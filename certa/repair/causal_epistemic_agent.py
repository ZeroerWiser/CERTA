"""CERA orchestration for certificate-guided repair.

CERA is the repair-decision LLM role in CERTA. This module coordinates
deterministic packet construction, optional CERA generation, and validator
checks. It never mutates the pipeline answer directly; commit authority
remains in the top-level inference pipeline and is disabled by default.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from eval_utils import normalize_text

from certa.derivations import (
    build_admissible_candidate_set,
    build_audit_derivation_pool,
    build_basis_relative_behavior_classes,
    build_compact_behavioral_contrast_v3,
    build_compact_behavioral_contrast_v2,
    build_decision_derivation_pool,
    build_derivation_lattice,
    build_minimal_contrast_set,
    build_original_support_symmetry_v3,
    build_role_binding_substitution_pairs,
    build_sample_fixed_role_intervention_basis,
    build_symmetric_derivation_frontier,
    materialize_derivations,
    reconstruct_original_support_hypotheses,
)
from certa.egra.evidence_cards import build_structural_evidence_cards
from certa.egra.planner_view import build_role_aligned_planner_view
from certa.egra.query_role_contract import (
    CORE_SIGNATURE_IDS,
    request_query_role_contract,
    validate_query_role_contract,
)
from certa.egra.retrieval import build_card_index, retrieve_structural_cards
from certa.derivations.answer_equivalence import inference_answer_key, inference_answers_equivalent
from certa.derivations.project import answers_equivalent
from certa.evidence.chains import build_causal_evidence_packet, stable_packet_hash
from certa.grounding import build_plan_closure, partition_support
from certa.logging.cera_audit import build_cera_request_audit, stable_hash_json, stable_hash_text
from certa.operations.contracts import LOOKUP_ACTIVE_SIGNATURE_IDS
from certa.planner import (
    CERAPlannerBoundary,
    build_proposal_aware_diagnostic_planner_view,
    build_proposal_blind_planner_view,
    build_typed_derivation_planner_prompt,
    build_typed_planner_response_schema,
    coerce_planner_boundary,
    planner_constraint_schema_hash,
    planner_reference_domain,
    planner_boundary_telemetry,
    validate_diagnostic_boundary_runtime,
    validate_typed_planner_output,
)
from certa.planner.schema_view import build_canonical_structural_group_catalog
from certa.repair.evidence_packet import CERACommitResult, CERAOutput, CertifiedCandidateFull
from certa.repair.method_context import MethodInferenceContext, assert_method_context_clean
from certa.repair.repair_prompt import CERA_V3_TEMPLATE_VERSION, DEFAULT_CERA_TEMPLATE_VERSION, build_cera_prompt as _build_cera_prompt
from certa.repair.safety_validator import validate_cera_output, validate_cera_output_v3
from certa.reproducibility.canonical_json import canonical_json, canonical_json_hash
from certa.semantics.query_contract import build_pre_evidence_query_contract
from certa.traces import (
    PATCH_VERSION,
    build_intent_prompt,
    build_intent_response_schema,
    build_role_binding_prompt,
    build_role_binding_response_schema,
    build_minimal_structural_patch_registry,
    build_typed_executable_traces,
    build_validation_failure_records,
    first_verifiable_failure,
    validate_intent_output,
    validate_role_binding_output,
)


ROUND11_CLOSURE_AUDIT_VERSION = "round11_operation_closure_audit_v1"
ROUND10_CLOSURE_AUDIT_VERSION = "round10_closure_audit_v1"
ROUND12_SEMANTIC_TYPE_AUDIT_VERSION = "round12_semantic_type_audit_v1"
EGRA_PARENT_SHA = "9c15effaa23eba4c5fe0b00a69136453326e3854"
EGRA_ARMS = (
    "C0_FLAT_SCHEMA_CURRENT",
    "C1_ROLE_ALIGNED_FLAT",
    "C2_EGRA",
)


def build_cera_prompt(packet: Any, **kwargs: Any) -> str:
    return _build_cera_prompt(packet, **kwargs)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _candidate_rows(cert_info: Mapping[str, Any]) -> List[CertifiedCandidateFull]:
    rows = cert_info.get("certified_candidates_full") or []
    out: List[CertifiedCandidateFull] = []
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, Mapping):
                out.append(CertifiedCandidateFull.from_dict(row))
    return out


def _candidate_sort_key(candidate: CertifiedCandidateFull) -> Tuple[int, int, float, float, float, int]:
    cert = candidate.certificate or {}
    return (
        int(not bool(cert.get("evidence_fallback", False))),
        int(bool(cert.get("path_verified", False))),
        _as_float(cert.get("candidate_effective_evidence_coverage"), 0.0),
        _as_float(cert.get("scci"), 0.0),
        _as_float(cert.get("binding_confidence"), 0.0),
        -int(candidate.priority or 99),
    )


def select_review_candidate(
    cert_info: Mapping[str, Any],
    *,
    original_answer: str,
) -> Optional[CertifiedCandidateFull]:
    candidates = [c for c in _candidate_rows(cert_info) if str(c.denotation).strip()]
    if not candidates:
        return None
    alternatives = [c for c in candidates if not inference_answers_equivalent(c.denotation, original_answer)]
    pool = alternatives or candidates
    return sorted(pool, key=_candidate_sort_key, reverse=True)[0]


def _candidate_index(candidate_id: str) -> Optional[int]:
    if not candidate_id.startswith("cand_"):
        return None
    try:
        return int(candidate_id.split("_", 1)[1])
    except (IndexError, ValueError):
        return None


def match_live_exec_candidate(candidate: CertifiedCandidateFull, all_exec_candidates: Optional[Sequence[Any]]) -> Any:
    if not all_exec_candidates:
        return None
    idx = _candidate_index(candidate.candidate_id)
    if idx is not None and 0 <= idx < len(all_exec_candidates):
        indexed = all_exec_candidates[idx]
        if inference_answers_equivalent(getattr(indexed, "denotation", ""), candidate.denotation):
            return indexed
    for live in all_exec_candidates:
        if inference_answers_equivalent(getattr(live, "denotation", ""), candidate.denotation):
            return live
    return None


def _candidate_from_derivation(derivation: Any) -> CertifiedCandidateFull:
    metadata = dict(getattr(derivation, "operation_metadata", {}) or {})
    metadata.setdefault("operation_family", getattr(derivation, "operation_family", "UNKNOWN"))
    polarity = getattr(derivation, "comparison_polarity", "unknown")
    if polarity and polarity != "unknown":
        metadata.setdefault("comparison_polarity", polarity)
    cells = [dict(item) for item in (getattr(derivation, "operand_metadata", []) or [])]
    answer = str(getattr(derivation, "projected_answer", "") or "")
    frontier_source = "symmetric_derivation_frontier_v1"
    source_candidate = dict(getattr(derivation, "source_candidate", {}) or {})
    source = str(source_candidate.get("source") or frontier_source)
    frontier_generated = (
        str(getattr(derivation, "source_candidate_id", "")).startswith("frontier_")
        or source == frontier_source
    )
    certificate = {
        "path_verified": False if frontier_generated else bool(getattr(derivation, "provenance_complete", False)),
        "evidence_fallback": False,
        "frontier_generated": bool(frontier_generated),
        "planner_generated": source == "typed_derivation_planner_agent",
        "requires_independent_derivation_verification": bool(frontier_generated),
        "derivation_provenance_complete": bool(getattr(derivation, "provenance_complete", False)),
        "derivation_id": getattr(derivation, "derivation_id", ""),
        "required_edge_count": len(getattr(derivation, "required_edge_triples", []) or []),
    }
    if not frontier_generated:
        certificate.update({
            "candidate_evidence_coverage": 1.0 if getattr(derivation, "provenance_complete", False) else 0.0,
            "candidate_effective_evidence_coverage": 1.0 if getattr(derivation, "provenance_complete", False) else 0.0,
        })
    return CertifiedCandidateFull(
        candidate_id=str(getattr(derivation, "source_candidate_id", "") or getattr(derivation, "derivation_id", "")),
        denotation=answer,
        normalized_denotation=normalize_text(answer),
        operation=str(getattr(derivation, "operation_family", "UNKNOWN")).lower(),
        priority=5,
        cells_used=cells,
        computation_trace=f"symmetric frontier derivation {getattr(derivation, 'derivation_id', '')}",
        operation_metadata=metadata,
        source=source,
        certificate=certificate,
    )


def _mapping_get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return default


def _count_prompt_tokens(generator: Any, prompt: str) -> int:
    if generator is not None and hasattr(generator, "count_generation_prompt_tokens"):
        try:
            return int(generator.count_generation_prompt_tokens(prompt))
        except Exception:
            pass
    return max(1, (len(prompt or "") + 3) // 4) if prompt else 0


def _planner_enabled(args: Any) -> bool:
    if args is None:
        return False
    return bool(
        getattr(args, "cera_enable_typed_planner", False)
        or getattr(args, "enable_typed_derivation_planner", False)
    )


class StructuredOutputUnsupportedError(RuntimeError):
    """Raised when RCPC cannot exercise exact schema-constrained generation."""


def call_typed_planner_agent(
    generator: Any,
    prompt: str,
    *,
    max_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
    response_schema: Optional[Mapping[str, Any]] = None,
    schema_name: str = "certa_typed_planner",
    require_structured_output: bool = False,
) -> Dict[str, Any]:
    if generator is None:
        return {"text": "", "generation_seconds": 0.0, "error": "no_generator"}
    start = time.time()
    if require_structured_output:
        method = getattr(generator, "generate_json_schema", None)
        if not callable(method):
            raise StructuredOutputUnsupportedError(
                "structured_output_unsupported: generator has no generate_json_schema method"
            )
        if not isinstance(response_schema, Mapping):
            raise ValueError("structured_output_schema_missing")
        output = dict(method(
            prompt,
            response_schema=dict(response_schema),
            schema_name=schema_name,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        ))
    else:
        outputs = generator.generate(
            [prompt],
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            logprobs=0,
        )
        output = dict(outputs[0] if outputs else {"text": ""})
    output.setdefault("generation_seconds", time.time() - start)
    return output


def _build_typed_planner_request_audit(
    *,
    prompt: str,
    view: Mapping[str, Any],
    args: Any = None,
    generator: Any = None,
    generation_output: Optional[Mapping[str, Any]] = None,
    constraint_schema_hash: str = "",
    structured_output_mechanism: str = "",
) -> Dict[str, Any]:
    prompt_hash = stable_hash_text(prompt)
    view_hash = stable_hash_json(view)
    sampling = {
        "max_tokens": getattr(args, "cera_planner_max_tokens", None),
        "temperature": getattr(args, "cera_planner_temperature", None),
        "top_p": getattr(args, "top_p", None),
    }
    input_tokens = int(_mapping_get(generation_output or {}, "input_token_count", 0) or 0)
    if not input_tokens:
        input_tokens = _count_prompt_tokens(generator, prompt)
    output_tokens = int(_mapping_get(generation_output or {}, "generated_token_count", 0) or 0)
    audit = {
        "planner_view_hash": view_hash,
        "prompt_hash": prompt_hash,
        "model": str(
            _mapping_get(generation_output or {}, "api_model", "")
            or getattr(generator, "model", "")
            or getattr(args, "api_model", "")
            or getattr(args, "model_path", "")
        ),
        "backend": str(
            _mapping_get(generation_output or {}, "generator_backend", "")
            or getattr(generator, "backend_name", "")
            or getattr(args, "generator_backend", "")
        ),
        "api_base_url": str(
            _mapping_get(generation_output or {}, "api_base_url", "")
            or getattr(generator, "api_base_url", "")
            or getattr(args, "api_base_url", "")
        ),
        "sampling": sampling,
        "api_cache_hit": bool(_mapping_get(generation_output or {}, "api_cache_hit", False)),
        "api_cache_mode": str(
            _mapping_get(generation_output or {}, "api_cache_mode", "")
            or getattr(args, "api_cache_mode", "")
        ),
        "latency_seconds": float(_mapping_get(generation_output or {}, "generation_seconds", 0.0) or 0.0),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "constraint_schema_hash": str(constraint_schema_hash or ""),
        "structured_output_mechanism": str(structured_output_mechanism or ""),
        "signature_allowlist": list((view.get("operation_ontology") or {}).get("signature_ids") or []),
    }
    audit["signature_allowlist_hash"] = stable_hash_json(audit["signature_allowlist"])
    audit["request_hash"] = stable_hash_json({
        "planner_view_hash": view_hash,
        "prompt_hash": prompt_hash,
        "model": audit["model"],
        "backend": audit["backend"],
        "api_base_url": audit["api_base_url"],
        "sampling": audit["sampling"],
        "constraint_schema_hash": audit["constraint_schema_hash"],
        "structured_output_mechanism": audit["structured_output_mechanism"],
        "signature_allowlist_hash": audit["signature_allowlist_hash"],
    })
    return audit


def _typed_planner_disabled_metadata() -> Dict[str, Any]:
    return {
        "cera_planner_enabled": False,
        "cera_planner_called": False,
        "cera_planner_derivation_count": 0,
        "cera_planner_compile_failure_count": 0,
    }


def _closure_audit_records(closure: Any) -> List[Dict[str, Any]]:
    assignments = {
        str(getattr(item, "derivation_id", "")): item
        for item in (getattr(closure, "assignments", ()) or ())
        if str(getattr(item, "derivation_id", ""))
    }
    records: List[Dict[str, Any]] = []
    for derivation in getattr(closure, "executable_derivations", ()) or ():
        derivation_id = str(getattr(derivation, "derivation_id", ""))
        assignment = assignments.get(derivation_id)
        metadata = dict(getattr(derivation, "operation_metadata", {}) or {})
        plan_ids = list(metadata.get("plan_ids") or [])
        if assignment is not None:
            plan_ids = list(getattr(assignment, "plan_ids", ()) or plan_ids)
        provenance_ids = set(str(item) for item in (getattr(derivation, "evidence_ids", []) or []) if str(item))
        provenance_ids.update(str(item) for item in (getattr(derivation, "operand_node_ids", []) or []) if str(item))
        for source, target, _ in getattr(derivation, "required_edge_triples", []) or []:
            provenance_ids.add(str(source))
            provenance_ids.add(str(target))
        records.append({
            "derivation_id": derivation_id,
            "plan_ids": sorted(str(item) for item in plan_ids if str(item)),
            "closure_version": str(getattr(closure, "closure_version", "")),
            "operation_contract_version": str(
                getattr(closure, "operation_contract_version", "")
            ),
            "operation_family": str(getattr(derivation, "operation_family", "")),
            "roles": {
                str(role): list(values)
                for role, values in dict(getattr(assignment, "role_bindings", {}) or {}).items()
            } if assignment is not None else {},
            "projected_answer": str(getattr(derivation, "projected_answer", "")),
            "execution_outcome": str(
                getattr(assignment, "execution_outcome", "EXECUTED")
            ),
            "projection_outcome": str(getattr(assignment, "projection_outcome", "PROJECTED")),
            "comparison_polarity": str(getattr(assignment, "comparison_polarity", "")),
            "projection_operator": str(getattr(assignment, "projection_operator", "")),
            "answer_domain": str(getattr(assignment, "answer_domain", "")),
            "resolved_atomic_operands": [
                list(item)
                for item in (getattr(assignment, "resolved_atomic_operands", ()) or ())
            ],
            "resolved_scope_node_ids": list(
                getattr(assignment, "resolved_scope_node_ids", ()) or ()
            ),
            "resolved_entity_value_relation": [
                list(item)
                for item in (
                    getattr(assignment, "resolved_entity_value_relation", ()) or ()
                )
            ],
            "canonical_program_id": str(getattr(assignment, "canonical_program_id", "")),
            "resource_complete": bool(getattr(assignment, "resource_complete", True)),
            "provenance_ids": sorted(provenance_ids),
        })
    return records


def _semantic_type_audit_records(closure: Any) -> List[Dict[str, Any]]:
    return [
        {
            "assignment_id": str(getattr(assignment, "assignment_id", "")),
            "plan_ids": list(getattr(assignment, "plan_ids", ()) or ()),
            "signature_id": str(getattr(assignment, "signature_id", "")),
            "operation_family": str(getattr(assignment, "operation_family", "")),
            "semantic_result_role": str(
                getattr(assignment, "semantic_result_role", "")
            ),
            "closure_outcome": str(
                getattr(getattr(assignment, "outcome", ""), "value", getattr(assignment, "outcome", ""))
            ),
            "execution_outcome": str(getattr(assignment, "execution_outcome", "NOT_RUN")),
            "projection_outcome": str(getattr(assignment, "projection_outcome", "NOT_RUN")),
            "projection_result": dict(
                getattr(assignment, "projection_result", {}) or {}
            ),
            "failure_reasons": list(getattr(assignment, "failure_reasons", ()) or ()),
        }
        for assignment in (getattr(closure, "assignments", ()) or ())
    ]


def _validate_stepwise_trace_runtime(args: Any) -> None:
    if str(getattr(args, "cera_stage", "E71")).upper() != "E71":
        raise ValueError("Round 12 stepwise trace requires E71")
    if not bool(getattr(args, "cera_shadow_only", True)):
        raise ValueError("Round 12 stepwise trace requires shadow-only runtime")
    if bool(getattr(args, "cera_commit_approved_repair", False)):
        raise ValueError("Round 12 stepwise trace forbids CERA commit approval")
    if str(getattr(args, "cera_planner_boundary", "")) != "proposal_blind_schema_only":
        raise ValueError("Round 12 stepwise trace requires proposal-blind schema-only boundary")
    if str(getattr(args, "cera_planner_legacy_query_semantics_mode", "")) != "audit_only":
        raise ValueError("Round 12 stepwise trace requires legacy query semantics audit_only")
    if str(getattr(args, "cera_planner_contract", "")) != "rcpc_signature_v2":
        raise ValueError("Round 12 stepwise trace requires rcpc_signature_v2")


def _run_typed_stepwise_trace(
    *,
    view: Mapping[str, Any],
    graph: Any,
    generator: Any,
    args: Any,
    metadata: Dict[str, Any],
) -> Tuple[List[Any], Dict[str, Any], Any]:
    """Run the same Typed Derivation Planner Agent in two constrained stages."""
    metadata.update({
        "cera_round12_trace_version": "typed_executable_reasoning_trace_v1",
        "cera_round12_trace_planner_call_count": 0,
        "cera_round12_trace_intent_count": 0,
        "cera_round12_trace_role_step_count": 0,
        "cera_round12_trace_count": 0,
        "cera_round12_trace_executable_count": 0,
        "cera_round12_trace_resource_complete": False,
        "cera_round12_trace_records": [],
        "cera_round12_trace_fvf_records": [],
        "cera_round12_trace_fvf_stage_counts": {},
        "cera_round12_trace_validation_failure_count": 0,
        "cera_round12_trace_request_audits": [],
        "cera_round12_trace_raw_outputs": [],
        "cera_round12_patch_shadow_enabled": bool(
            getattr(args, "cera_round12_minimal_patch_shadow", False)
        ),
        "cera_round12_patch_version": PATCH_VERSION,
        "cera_round12_patch_eligible_source_trace_count": 0,
        "cera_round12_patch_local_domain_count": 0,
        "cera_round12_patch_candidate_records": [],
        "cera_round12_patch_minimal_executable_count": 0,
        "cera_round12_patch_minimal_records": [],
        "cera_round12_patch_model_calls": 0,
    })
    if generator is None:
        metadata["cera_planner_skipped_reason"] = "no_generator"
        return [], metadata, None

    stage_records: List[Dict[str, Any]] = []

    def record_validation_fvf(raw: Any, errors: Sequence[str], boundary: str) -> None:
        records = [
            item.to_dict()
            for item in build_validation_failure_records(raw, errors, boundary)
        ]
        counts: Dict[str, int] = {}
        for record in records:
            stage = str(record.get("stage", ""))
            counts[stage] = counts.get(stage, 0) + 1
        metadata.update({
            "cera_round12_trace_validation_failure_count": len(records),
            "cera_round12_trace_fvf_records": records,
            "cera_round12_trace_fvf_stage_counts": counts,
        })

    def update_stage_aggregates() -> None:
        audits = [dict(item["request_audit"]) for item in stage_records]
        if not audits:
            return
        expected_schema_hashes = [
            str(item.get("constraint_schema_hash", "")) for item in stage_records
        ]
        reported_schema_hashes = [
            str(item.get("reported_schema_hash", "")) for item in stage_records
        ]
        raw_output_hashes = [
            str(item.get("raw_output_hash", "")) for item in stage_records
        ]
        mechanisms = sorted({
            str(item.get("structured_output_mechanism", ""))
            for item in stage_records
            if str(item.get("structured_output_mechanism", ""))
        })
        metadata.update({
            "cera_planner_prompt_hash": canonical_json_hash(
                [item.get("prompt_hash", "") for item in stage_records]
            ),
            "cera_planner_request_hash": canonical_json_hash(
                [audit.get("request_hash", "") for audit in audits]
            ),
            "cera_planner_model": audits[0].get("model", ""),
            "cera_planner_backend": audits[0].get("backend", ""),
            "cera_planner_api_base_url": audits[0].get("api_base_url", ""),
            "cera_planner_sampling": audits[0].get("sampling", {}),
            "cera_planner_api_cache_hit": all(
                audit.get("api_cache_hit") for audit in audits
            ),
            "cera_planner_api_cache_mode": audits[0].get("api_cache_mode", ""),
            "cera_planner_latency_seconds": sum(
                float(audit.get("latency_seconds", 0.0) or 0.0) for audit in audits
            ),
            "cera_planner_input_tokens": sum(
                int(audit.get("input_tokens", 0) or 0) for audit in audits
            ),
            "cera_planner_output_tokens": sum(
                int(audit.get("output_tokens", 0) or 0) for audit in audits
            ),
            "cera_planner_request_audit": {"stages": audits},
            "cera_planner_constraint_schema_hash": canonical_json_hash(
                expected_schema_hashes
            ),
            "cera_planner_structured_output_schema_hash": canonical_json_hash(
                reported_schema_hashes
            ),
            "cera_planner_raw_output_hash": canonical_json_hash(raw_output_hashes),
            "cera_planner_constraint_fallback_used": any(
                bool(item.get("structured_output_fallback_used", False))
                for item in stage_records
            ),
            "cera_planner_structured_output_mechanism": (
                mechanisms[0]
                if len(mechanisms) == 1
                else (f"mixed:{canonical_json_hash(mechanisms)}" if mechanisms else "")
            ),
            "cera_round12_trace_request_audits": list(stage_records),
        })

    def run_stage(
        *,
        stage_name: str,
        prompt: str,
        response_schema: Mapping[str, Any],
        schema_name: str,
    ) -> Tuple[str, str]:
        schema_hash = canonical_json_hash(response_schema)
        metadata["cera_round12_trace_planner_call_count"] += 1
        metadata["cera_planner_called"] = True
        started = time.time()
        try:
            generated = call_typed_planner_agent(
                generator,
                prompt,
                max_tokens=int(getattr(args, "cera_planner_max_tokens", 512)),
                temperature=float(getattr(args, "cera_planner_temperature", 0.0)),
                top_p=float(getattr(args, "top_p", 1.0)),
                response_schema=response_schema,
                schema_name=schema_name,
                require_structured_output=True,
            )
        except Exception as exc:
            elapsed = time.time() - started
            audit = _build_typed_planner_request_audit(
                prompt=prompt,
                view=view,
                args=args,
                generator=generator,
                generation_output={"generation_seconds": elapsed},
                constraint_schema_hash=schema_hash,
                structured_output_mechanism="",
            )
            stage_records.append({
                "stage": stage_name,
                "schema_name": schema_name,
                "prompt_hash": stable_hash_text(prompt),
                "constraint_schema_hash": schema_hash,
                "reported_schema_hash": "",
                "raw_output_hash": stable_hash_text(""),
                "structured_output_fallback_used": False,
                "structured_output_mechanism": "",
                "request_audit": audit,
                "generation_exception": str(exc),
            })
            update_stage_aggregates()
            metadata["cera_planner_failure_kind"] = "generation_exception"
            return "", f"{stage_name}_generation_exception:{exc}"
        audit = _build_typed_planner_request_audit(
            prompt=prompt,
            view=view,
            args=args,
            generator=generator,
            generation_output=generated,
            constraint_schema_hash=schema_hash,
            structured_output_mechanism=str(
                generated.get("structured_output_mechanism", "") or ""
            ),
        )
        raw = str(generated.get("text", "") or "")
        fallback_used = bool(generated.get("structured_output_fallback_used", False))
        mechanism = str(generated.get("structured_output_mechanism", "") or "")
        reported_hash = str(generated.get("structured_output_schema_hash", "") or "")
        record = {
            "stage": stage_name,
            "schema_name": schema_name,
            "prompt_hash": stable_hash_text(prompt),
            "constraint_schema_hash": schema_hash,
            "reported_schema_hash": reported_hash,
            "raw_output_hash": stable_hash_text(raw),
            "structured_output_fallback_used": fallback_used,
            "structured_output_mechanism": mechanism,
            "request_audit": audit,
        }
        stage_records.append(record)
        if bool(getattr(args, "cera_log_planner_raw_output", False)):
            metadata["cera_round12_trace_raw_outputs"].append({
                "stage": stage_name,
                "raw_output": raw,
            })
        update_stage_aggregates()
        if generated.get("error"):
            return "", f"{stage_name}_generation_error:{generated.get('error')}"
        if fallback_used:
            return "", f"{stage_name}_structured_output_fallback_forbidden"
        if not bool(generated.get("structured_output_requested", False)):
            return "", f"{stage_name}_structured_output_request_not_confirmed"
        if mechanism != "response_format.type=json_schema":
            return "", f"{stage_name}_structured_output_mechanism_mismatch"
        if reported_hash != schema_hash:
            return "", f"{stage_name}_structured_output_schema_hash_mismatch"
        return raw, ""

    intent_prompt = build_intent_prompt(view)
    intent_schema = build_intent_response_schema(view)
    metadata.update({
        "cera_planner_prompt_hash": stable_hash_text(intent_prompt),
        "cera_planner_constraint_schema_hash": canonical_json_hash(intent_schema),
    })
    intent_raw, error = run_stage(
        stage_name="INTENT_HYPOTHESIS",
        prompt=intent_prompt,
        response_schema=intent_schema,
        schema_name="certa_round12_trace_intent_v1",
    )
    if error:
        metadata["cera_planner_generation_error"] = error
        return [], metadata, None
    intent_validation = validate_intent_output(intent_raw, view)
    metadata.update({
        "cera_round12_trace_intent_parse_ok": bool(intent_validation.parse_ok),
        "cera_round12_trace_intent_validation_ok": bool(intent_validation.ok),
        "cera_round12_trace_intent_validation_errors": list(intent_validation.errors),
        "cera_round12_trace_intent_count": len(intent_validation.intents),
        "cera_round12_trace_intent_hypotheses": [
            item.to_dict() for item in intent_validation.intents
        ],
    })
    if not intent_validation.ok:
        metadata["cera_planner_validation_errors"] = list(intent_validation.errors)
        record_validation_fvf(
            intent_raw,
            intent_validation.errors,
            "INTENT_CONTRACT",
        )
        return [], metadata, None

    role_prompt = build_role_binding_prompt(view, intent_validation.intents)
    role_schema = build_role_binding_response_schema(view, intent_validation.intents)
    role_raw, error = run_stage(
        stage_name="ROLE_BINDING",
        prompt=role_prompt,
        response_schema=role_schema,
        schema_name="certa_round12_trace_roles_v1",
    )
    if error:
        metadata["cera_planner_generation_error"] = error
        return [], metadata, None
    role_validation = validate_role_binding_output(
        role_raw,
        view,
        intent_validation.intents,
    )
    metadata.update({
        "cera_round12_trace_role_parse_ok": bool(role_validation.parse_ok),
        "cera_round12_trace_role_validation_ok": bool(role_validation.ok),
        "cera_round12_trace_role_validation_errors": list(role_validation.errors),
        "cera_round12_trace_role_step_count": len(role_validation.role_steps),
        "cera_round12_trace_role_steps": [
            item.to_dict() for item in role_validation.role_steps
        ],
    })
    if not role_validation.ok:
        metadata["cera_planner_validation_errors"] = list(role_validation.errors)
        record_validation_fvf(
            role_raw,
            role_validation.errors,
            "ROLE_BINDING",
        )
        return [], metadata, None

    max_assignments = int(getattr(args, "cera_trace_max_assignments", 512))
    closure = build_plan_closure(
        role_validation.normalized_payload,
        graph,
        max_assignments=max_assignments,
    )
    traces = build_typed_executable_traces(
        intent_validation.intents,
        role_validation.role_steps,
        closure,
        graph,
    )
    trace_records = [item.to_dict() for item in traces]
    fvf_records = [
        failure.to_dict()
        for failure in (first_verifiable_failure(item) for item in traces)
        if failure is not None
    ]
    fvf_stage_counts: Dict[str, int] = {}
    for record in fvf_records:
        stage = str(record.get("stage", ""))
        fvf_stage_counts[stage] = fvf_stage_counts.get(stage, 0) + 1

    patch_registry: Dict[str, Any] = {}
    if bool(getattr(args, "cera_round12_minimal_patch_shadow", False)):
        patch_registry = build_minimal_structural_patch_registry(
            intent_validation.intents,
            role_validation.role_steps,
            closure,
            graph,
        )

    schema_hashes = [str(item["constraint_schema_hash"]) for item in stage_records]
    raw_hashes = [str(item["raw_output_hash"]) for item in stage_records]
    metadata.update({
        "cera_planner_parse_ok": True,
        "cera_planner_validation_ok": True,
        "cera_planner_validation_errors": [],
        "cera_planner_valid_plan_count": len(role_validation.normalized_payload.get("plans") or []),
        "cera_planner_derivation_count": len(closure.executable_derivations),
        "cera_planner_compile_failure_count": sum(
            1 for item in closure.assignments if item.outcome.value != "UNIQUE_EXECUTABLE"
        ),
        "cera_planner_constraint_schema_hash": canonical_json_hash(schema_hashes),
        "cera_planner_structured_output_mechanism": "response_format.type=json_schema",
        "cera_planner_structured_output_schema_hash": canonical_json_hash(schema_hashes),
        "cera_planner_raw_output_hash": canonical_json_hash(raw_hashes),
        "cera_round12_trace_request_audits": stage_records,
        "cera_round12_trace_count": len(traces),
        "cera_round12_trace_executable_count": sum(1 for item in traces if item.executable),
        "cera_round12_trace_resource_complete": bool(getattr(closure, "resource_complete", True)),
        "cera_round12_trace_records": trace_records,
        "cera_round12_trace_fvf_records": fvf_records,
        "cera_round12_trace_fvf_stage_counts": fvf_stage_counts,
        "cera_round12_patch_eligible_source_trace_count": int(
            patch_registry.get("eligible_source_trace_count", 0)
        ),
        "cera_round12_patch_local_domain_count": int(
            patch_registry.get("local_patch_domain_count", 0)
        ),
        "cera_round12_patch_candidate_records": list(
            patch_registry.get("candidate_records", [])
        ),
        "cera_round12_patch_minimal_executable_count": int(
            patch_registry.get("minimal_executable_patch_count", 0)
        ),
        "cera_round12_patch_minimal_records": list(
            patch_registry.get("minimal_patch_records", [])
        ),
        "cera_round12_patch_model_calls": int(
            patch_registry.get("candidate_model_calls", 0)
        ),
        "cera_round11_closure_resource_complete": bool(getattr(closure, "resource_complete", True)),
        "cera_round11_closure_declared_assignment_count": int(getattr(closure, "declared_assignment_count", 0)),
        "cera_round11_closure_realized_assignment_count": int(getattr(closure, "realized_assignment_count", 0)),
        "cera_round10_closure_audit_records": _closure_audit_records(closure),
        "cera_round12_semantic_type_audit_records": _semantic_type_audit_records(closure),
    })
    return list(closure.executable_derivations), metadata, closure


def _run_typed_derivation_planner(
    *,
    question: str,
    graph: Any,
    table_json: Optional[Mapping[str, Any]],
    pre_contract: Any,
    generator: Any,
    args: Any,
    original_answer: str,
) -> Tuple[List[Any], Dict[str, Any], Any]:
    if not _planner_enabled(args):
        return [], _typed_planner_disabled_metadata(), None

    stepwise_trace = bool(getattr(args, "cera_stepwise_trace", False))
    if stepwise_trace:
        _validate_stepwise_trace_runtime(args)

    boundary = coerce_planner_boundary(
        getattr(args, "cera_planner_boundary", CERAPlannerBoundary.PROPOSAL_BLIND_SCHEMA_ONLY.value)
        if args is not None
        else CERAPlannerBoundary.PROPOSAL_BLIND_SCHEMA_ONLY.value
    )
    boundary_fields = planner_boundary_telemetry(boundary)
    metadata: Dict[str, Any] = {
        "cera_planner_enabled": True,
        "cera_planner_called": False,
        "cera_planner_validation_ok": False,
        "cera_planner_valid_plan_count": 0,
        "cera_planner_derivation_count": 0,
        "cera_planner_compile_failure_count": 0,
        "cera_planner_compile_failures": [],
        "cera_planner_validation_errors": [],
        "cera_planner_parse_ok": False,
        "cera_round10_closure_audit_version": ROUND10_CLOSURE_AUDIT_VERSION,
        "cera_round11_closure_audit_version": ROUND11_CLOSURE_AUDIT_VERSION,
        "cera_round10_closure_audit_records": [],
        "cera_round12_semantic_type_audit_version": ROUND12_SEMANTIC_TYPE_AUDIT_VERSION,
        "cera_round12_semantic_type_audit_records": [],
        "cera_planner_boundary_condition": boundary_fields["planner_boundary_condition"],
        "cera_planner_proposal_visible_to_planner": boundary_fields["proposal_visible_to_planner"],
        "cera_planner_table_values_visible_to_planner": boundary_fields["table_values_visible_to_planner"],
        "cera_planner_boundary_ablation_arm": boundary_fields["boundary_ablation_arm"],
        "cera_planner_contract_version": str(
            getattr(args, "cera_planner_contract", "legacy_v1") if args is not None else "legacy_v1"
        ),
        "cera_planner_reference_domain_count": 0,
        "cera_planner_reference_domain_hash": "",
        "cera_planner_constraint_schema_hash": "",
        "cera_planner_structured_output_mechanism": "",
        "cera_planner_structured_output_schema_hash": "",
        "cera_planner_constraint_fallback_used": False,
        "cera_planner_invalid_generated_reference_count": 0,
    }
    egra_arm = str(
        getattr(args, "certa_egra_arm", "") if args is not None else ""
    ).strip()
    if egra_arm and egra_arm not in EGRA_ARMS:
        raise ValueError(f"unknown_certa_egra_arm:{egra_arm}")
    if egra_arm:
        if (
            str(getattr(args, "cera_stage", "E71")).upper() != "E71"
            or not bool(getattr(args, "cera_shadow_only", True))
            or bool(getattr(args, "cera_commit_approved_repair", False))
        ):
            raise ValueError("certa_egra_requires_e71_shadow_no_commit")
        metadata.update({
            "certa_egra_arm": egra_arm,
            "certa_egra_role_contract_called": False,
            "certa_egra_role_contract_valid": False,
            "certa_egra_role_contract_errors": [],
        })
    if graph is None:
        metadata["cera_planner_skipped_reason"] = "no_graph"
        return [], metadata, None

    configured_allowlist = str(
        getattr(args, "cera_planner_signature_allowlist", "") if args is not None else ""
    ).strip()
    allowed_signature_ids = None
    if egra_arm and configured_allowlist:
        raise ValueError("certa_egra_forbids_legacy_signature_allowlist")
    if egra_arm:
        allowed_signature_ids = CORE_SIGNATURE_IDS
    elif configured_allowlist:
        expected_allowlist = ",".join(LOOKUP_ACTIVE_SIGNATURE_IDS)
        if configured_allowlist != expected_allowlist:
            raise ValueError(
                f"invalid_cera_planner_signature_allowlist:{configured_allowlist}"
            )
        allowed_signature_ids = LOOKUP_ACTIVE_SIGNATURE_IDS

    proposal_aware = boundary == CERAPlannerBoundary.PROPOSAL_AWARE_DIAGNOSTIC
    legacy_query_semantics_mode = str(
        getattr(args, "cera_planner_legacy_query_semantics_mode", "active")
        if args is not None
        else "active"
    )
    validate_diagnostic_boundary_runtime(
        boundary,
        cera_stage=str(getattr(args, "cera_stage", "E71") if args is not None else "E71"),
        cera_shadow_only=bool(getattr(args, "cera_shadow_only", True) if args is not None else True),
        cera_commit_approved_repair=bool(
            getattr(args, "cera_commit_approved_repair", False) if args is not None else False
        ),
    )
    planner_contract = str(
        getattr(args, "cera_planner_contract", "legacy_v1")
        if args is not None
        else "legacy_v1"
    )
    if egra_arm:
        if boundary != CERAPlannerBoundary.PROPOSAL_BLIND_SCHEMA_ONLY:
            raise ValueError("certa_egra_requires_proposal_blind_schema_only")
        if planner_contract != "rcpc_signature_v2":
            raise ValueError("certa_egra_requires_rcpc_signature_v2")
        proposal_aware = False

    if egra_arm == "C0_FLAT_SCHEMA_CURRENT":
        view = build_proposal_blind_planner_view(
            question=question,
            graph=graph,
            table_json=table_json,
            query_contract=pre_contract,
            include_table_values=False,
            legacy_query_semantics_mode=legacy_query_semantics_mode,
            allowed_signature_ids=allowed_signature_ids,
        )
    elif egra_arm in {"C1_ROLE_ALIGNED_FLAT", "C2_EGRA"}:
        if generator is None:
            metadata["cera_planner_skipped_reason"] = "no_generator"
            return [], metadata, None
        try:
            frozen_roles = getattr(
                args,
                "_certa_egra_frozen_role_by_question_hash",
                None,
            )
            if frozen_roles is None:
                role_validation, role_audit = request_query_role_contract(
                    generator,
                    question,
                )
                role_reused = False
            else:
                question_hash = canonical_json_hash({"question": str(question or "")})
                frozen_role = frozen_roles.get(question_hash)
                if frozen_role is None:
                    raise ValueError("missing_frozen_role_contract")
                role_validation = validate_query_role_contract(
                    frozen_role.get("contract")
                )
                role_audit = dict(frozen_role.get("audit") or {})
                role_audit["frozen_source_calls"] = int(
                    role_audit.get("calls", 0) or 0
                )
                role_audit["calls"] = 0
                role_audit["frozen_reuse"] = True
                role_reused = True
        except Exception as exc:
            metadata.update({
                "certa_egra_role_contract_called": True,
                "certa_egra_role_contract_errors": [f"generation_exception:{exc}"],
                "cera_planner_skipped_reason": "query_role_generation_exception",
            })
            return [], metadata, None
        metadata.update({
            "certa_egra_role_contract_called": True,
            "certa_egra_role_contract_valid": bool(role_validation.ok),
            "certa_egra_role_contract_parse_ok": bool(role_validation.parse_ok),
            "certa_egra_role_contract_errors": list(role_validation.errors),
            "certa_egra_role_contract": dict(role_validation.normalized_payload),
            "certa_egra_role_contract_audit": role_audit,
            "certa_egra_role_contract_reused": role_reused,
        })
        if not role_validation.ok:
            metadata["cera_planner_skipped_reason"] = "invalid_query_role_contract"
            return [], metadata, None
        role_contract = role_validation.normalized_payload
        if not role_contract["supported_by_core_signatures"]:
            metadata["cera_planner_skipped_reason"] = "unsupported_by_core_signatures"
            return [], metadata, None
        allowed_signature_ids = tuple(role_contract["signature_candidates"])

        reference_ids = None
        selected_cards = None
        if egra_arm == "C2_EGRA":
            if not isinstance(table_json, Mapping):
                metadata["cera_planner_skipped_reason"] = "no_table_json"
                return [], metadata, None
            encoder = getattr(args, "_certa_egra_encoder", None)
            embedding_hash = str(
                getattr(args, "certa_egra_embedding_file_tree_sha256", "")
            )
            if encoder is None:
                metadata["cera_planner_skipped_reason"] = "no_egra_encoder"
                return [], metadata, None
            if len(embedding_hash) != 64:
                raise ValueError("invalid_certa_egra_embedding_file_tree_sha256")
            try:
                catalog_started = time.time()
                catalog = build_canonical_structural_group_catalog(
                    graph=graph,
                    table_json=table_json,
                )
                cards = build_structural_evidence_cards(catalog)
                catalog_seconds = time.time() - catalog_started
                index_started = time.time()
                table_sha256 = canonical_json_hash(table_json)
                index = build_card_index(
                    cards,
                    encoder,
                    parent_sha=EGRA_PARENT_SHA,
                    table_sha256=table_sha256,
                    embedding_file_tree_sha256=embedding_hash,
                )
                index_seconds = time.time() - index_started
                retrieval_started = time.time()
                retrieval = retrieve_structural_cards(
                    index,
                    cards,
                    question=question,
                    contract=role_contract,
                    encoder=encoder,
                )
                retrieval_seconds = time.time() - retrieval_started
            except Exception as exc:
                metadata.update({
                    "certa_egra_construction_error": str(exc),
                    "cera_planner_skipped_reason": "egra_construction_error",
                })
                return [], metadata, None
            cards_by_id = {str(card["card_id"]): card for card in cards}
            selected_cards = [
                cards_by_id[card_id]
                for card_id in retrieval["selected_card_ids"]
            ]
            reference_ids = retrieval["reference_node_ids"]
            metadata.update({
                "certa_egra_table_sha256": table_sha256,
                "certa_egra_catalog_sha256": str(catalog["catalog_sha256"]),
                "certa_egra_card_count": len(cards),
                "certa_egra_active_card_count": len(index["card_ids"]),
                "certa_egra_structural_cards": selected_cards,
                "certa_egra_index_cache_key": str(index["cache_key"]),
                "certa_egra_index_sha256": str(index["index_sha256"]),
                "certa_egra_retrieval": retrieval,
                "certa_egra_catalog_seconds": catalog_seconds,
                "certa_egra_index_seconds": index_seconds,
                "certa_egra_retrieval_seconds": retrieval_seconds,
                "certa_egra_index_cache_hit": False,
            })
        view_build = build_role_aligned_planner_view(
            question=question,
            graph=graph,
            table_json=table_json,
            contract=role_contract,
            reference_node_ids=reference_ids,
            selected_cards=selected_cards,
        )
        if not view_build.eligible:
            metadata["cera_planner_skipped_reason"] = view_build.reason
            return [], metadata, None
        view = view_build.view
    elif proposal_aware:
        view = build_proposal_aware_diagnostic_planner_view(
            question=question,
            graph=graph,
            table_json=table_json,
            query_contract=pre_contract,
            initial_proposal_diagnostic=original_answer,
        )
    else:
        view = build_proposal_blind_planner_view(
            question=question,
            graph=graph,
            table_json=table_json,
            query_contract=pre_contract,
            include_table_values=(boundary == CERAPlannerBoundary.PROPOSAL_BLIND_VALUE_AWARE),
            legacy_query_semantics_mode=legacy_query_semantics_mode,
            allowed_signature_ids=allowed_signature_ids,
        )
    prompt = build_typed_derivation_planner_prompt(view, proposal_aware=proposal_aware)
    rcpc_enabled = planner_contract in {"rcpc_v1", "rcpc_signature_v2"}
    signature_contract_enabled = planner_contract == "rcpc_signature_v2"
    reference_domain = planner_reference_domain(view)
    response_schema = (
        build_typed_planner_response_schema(
            view,
            require_signature_id=signature_contract_enabled,
        )
        if rcpc_enabled
        else None
    )
    constraint_schema_hash = (
        planner_constraint_schema_hash(
            view,
            require_signature_id=signature_contract_enabled,
        )
        if rcpc_enabled
        else ""
    )
    metadata.update({
        "cera_planner_view_version": str(view.get("planner_view_version", "")),
        "cera_planner_schema_node_count": len(view.get("schema_nodes") or []),
        "cera_planner_schema_edge_count": len(view.get("schema_edges") or []),
        "cera_planner_prompt_hash": stable_hash_text(prompt),
        "cera_planner_view_hash": stable_hash_json(view),
        "cera_planner_legacy_query_semantics_mode": legacy_query_semantics_mode,
        "cera_planner_legacy_query_semantics_public": "query_semantics" in view,
        "cera_planner_reference_domain_count": len(reference_domain),
        "cera_planner_reference_domain_hash": stable_hash_json(list(reference_domain)),
        "cera_planner_constraint_schema_hash": constraint_schema_hash,
        "cera_planner_signature_allowlist": list(
            (view.get("operation_ontology") or {}).get("signature_ids") or []
        ),
        "cera_planner_signature_allowlist_hash": stable_hash_json(
            (view.get("operation_ontology") or {}).get("signature_ids") or []
        ),
    })
    if stepwise_trace:
        return _run_typed_stepwise_trace(
            view=view,
            graph=graph,
            generator=generator,
            args=args,
            metadata=metadata,
        )
    if generator is None:
        metadata["cera_planner_skipped_reason"] = "no_generator"
        return [], metadata, None

    try:
        gen = call_typed_planner_agent(
            generator,
            prompt,
            max_tokens=int(getattr(args, "cera_planner_max_tokens", 512) if args is not None else 512),
            temperature=float(getattr(args, "cera_planner_temperature", 0.0) if args is not None else 0.0),
            top_p=float(getattr(args, "top_p", 1.0) if args is not None else 1.0),
            response_schema=response_schema,
            schema_name=(
                "certa_typed_planner_signature_v2"
                if signature_contract_enabled
                else "certa_typed_planner_rcpc_v1"
            ),
            require_structured_output=rcpc_enabled,
        )
    except Exception as exc:
        metadata.update({
            "cera_planner_called": True,
            "cera_planner_generation_error": str(exc),
            "cera_planner_failure_kind": "generation_exception",
        })
        return [], metadata, None
    audit = _build_typed_planner_request_audit(
        prompt=prompt,
        view=view,
        args=args,
        generator=generator,
        generation_output=gen,
        constraint_schema_hash=constraint_schema_hash,
        structured_output_mechanism=str(gen.get("structured_output_mechanism", "") or ""),
    )
    fallback_used = bool(gen.get("structured_output_fallback_used", False))
    reported_schema_hash = str(gen.get("structured_output_schema_hash", "") or "")
    structured_output_mechanism = str(gen.get("structured_output_mechanism", "") or "")
    metadata.update({
        "cera_planner_called": True,
        "cera_planner_request_hash": audit.get("request_hash"),
        "cera_planner_model": audit.get("model"),
        "cera_planner_backend": audit.get("backend"),
        "cera_planner_api_base_url": audit.get("api_base_url"),
        "cera_planner_sampling": audit.get("sampling"),
        "cera_planner_api_cache_hit": audit.get("api_cache_hit"),
        "cera_planner_api_cache_mode": audit.get("api_cache_mode"),
        "cera_planner_latency_seconds": audit.get("latency_seconds"),
        "cera_planner_input_tokens": audit.get("input_tokens"),
        "cera_planner_output_tokens": audit.get("output_tokens"),
        "cera_planner_request_audit": audit,
        "cera_planner_structured_output_mechanism": structured_output_mechanism,
        "cera_planner_structured_output_schema_hash": reported_schema_hash,
        "cera_planner_constraint_fallback_used": fallback_used,
    })
    raw = str(gen.get("text", "") or "")
    metadata["cera_planner_raw_output_hash"] = stable_hash_text(raw)
    if bool(getattr(args, "cera_log_planner_raw_output", False) if args is not None else False):
        metadata["cera_planner_raw_output"] = raw
    if gen.get("error"):
        metadata["cera_planner_generation_error"] = str(gen.get("error"))
        return [], metadata, None
    if rcpc_enabled and fallback_used:
        metadata["cera_planner_generation_error"] = "structured_output_fallback_forbidden"
        return [], metadata, None
    if rcpc_enabled and not bool(gen.get("structured_output_requested", False)):
        metadata["cera_planner_generation_error"] = "structured_output_request_not_confirmed"
        return [], metadata, None
    if rcpc_enabled and structured_output_mechanism != "response_format.type=json_schema":
        metadata["cera_planner_generation_error"] = "structured_output_mechanism_mismatch"
        return [], metadata, None
    if rcpc_enabled and reported_schema_hash != constraint_schema_hash:
        metadata["cera_planner_generation_error"] = "structured_output_schema_hash_mismatch"
        return [], metadata, None

    validation = validate_typed_planner_output(
        raw,
        view,
        require_signature_id=signature_contract_enabled,
    )
    invalid_generated_reference_count = sum(
        1
        for rejection in (getattr(validation, "plan_rejections", []) or [])
        for reason in (rejection.get("reasons") or [])
        if str(reason).startswith("unknown_schema_id:")
    )
    metadata.update({
        "cera_planner_parse_ok": bool(validation.parse_ok),
        "cera_planner_validation_ok": bool(validation.ok),
        "cera_planner_validation_errors": list(validation.errors),
        "cera_planner_valid_plan_count": int(validation.valid_plan_count),
        "cera_planner_plan_rejection_count": len(getattr(validation, "plan_rejections", []) or []),
        "cera_planner_plan_rejections": list(getattr(validation, "plan_rejections", []) or []),
        "cera_planner_resource_warnings": list(getattr(validation, "resource_warnings", []) or []),
        "cera_planner_invalid_generated_reference_count": invalid_generated_reference_count,
    })
    if not validation.ok:
        return [], metadata, None

    closure = build_plan_closure(
        validation.normalized_payload,
        graph,
        allowed_signature_ids=allowed_signature_ids,
    )
    non_executable = [
        {
            "assignment_id": assignment.assignment_id,
            "assignment_key": assignment.assignment_key,
            "outcome": assignment.outcome.value,
            "resolution_state": assignment.resolution_state,
            "matched_cell_ids": list(assignment.matched_cell_ids),
            "failure_reasons": list(assignment.failure_reasons),
            "operation_family": str(getattr(assignment, "operation_family", "")),
            "operation_contract_version": str(
                getattr(closure, "operation_contract_version", "")
            ),
            "comparison_polarity": str(getattr(assignment, "comparison_polarity", "")),
            "projection_operator": str(getattr(assignment, "projection_operator", "")),
            "answer_domain": str(getattr(assignment, "answer_domain", "")),
            "canonical_program_id": str(getattr(assignment, "canonical_program_id", "")),
            "resource_complete": bool(getattr(assignment, "resource_complete", True)),
        }
        for assignment in closure.assignments
        if assignment.outcome.value != "UNIQUE_EXECUTABLE"
    ]
    metadata.update({
        "cera_planner_derivation_count": len(closure.executable_derivations),
        "cera_planner_compile_failure_count": len(non_executable),
        "cera_planner_compile_failures": non_executable,
        "cera_round9_plan_closure_version": closure.closure_version,
        "cera_round9_closure_assignment_count": len(closure.assignments),
        "cera_round9_closure_executable_count": len(closure.executable_derivations),
        "cera_round9_closure_outcome_counts": dict(closure.outcome_counts),
        "cera_round11_operation_contract_version": str(
            getattr(closure, "operation_contract_version", "")
        ),
        "cera_round11_closure_resource_complete": bool(
            getattr(closure, "resource_complete", True)
        ),
        "cera_round11_closure_declared_assignment_count": int(
            getattr(closure, "declared_assignment_count", len(closure.assignments))
        ),
        "cera_round11_closure_realized_assignment_count": int(
            getattr(closure, "realized_assignment_count", len(closure.assignments))
        ),
        "cera_round11_closure_deduplicated_program_count": int(
            getattr(closure, "deduplicated_program_count", len(closure.executable_derivations))
        ),
        "cera_round10_closure_audit_records": _closure_audit_records(closure),
        "cera_round12_semantic_type_audit_records": _semantic_type_audit_records(closure),
    })
    return list(closure.executable_derivations), metadata, closure


def run_egra_constructor_shadow(
    *,
    question: str,
    original_answer: str,
    graph: Any,
    table_json: Mapping[str, Any],
    generator: Any,
    args: Any,
) -> Dict[str, Any]:
    """Run only role/retrieval/Planner/closure for a frozen EGRA arm."""
    if not str(original_answer or "").strip():
        return {
            "certa_egra_construction_only": True,
            "certa_egra_intervention_generated": False,
            "certa_egra_decision_executed": False,
            "cera_planner_called": False,
            "cera_planner_skipped_reason": "B0_INVALID",
        }
    pre_contract = build_pre_evidence_query_contract(
        question=question,
        question_frame=None,
        result_context={},
        initial_answer=original_answer,
        graph_stats=graph.stats() if hasattr(graph, "stats") else None,
    )
    _derivations, metadata, closure = _run_typed_derivation_planner(
        question=question,
        graph=graph,
        table_json=table_json,
        pre_contract=pre_contract,
        generator=generator,
        args=args,
        original_answer=original_answer,
    )
    metadata.update({
        "certa_egra_construction_only": True,
        "certa_egra_intervention_generated": False,
        "certa_egra_decision_executed": False,
    })
    if closure is not None:
        support = partition_support(
            closure,
            initial_proposal_answer=original_answer,
        )
        metadata.update({
            "cera_round9_partition_original_count": len(support.original_support),
            "cera_round9_partition_alternative_count": len(support.alternative_support),
            "cera_round9_partition_disjoint": bool(support.disjoint),
            "cera_round9_partition_exhaustive": bool(support.exhaustive),
            "cera_round9_partition_equivalence_policy": support.equivalence_policy,
        })
    return metadata


def call_cera_agent(
    generator: Any,
    prompt: str,
    *,
    max_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> Dict[str, Any]:
    if generator is None:
        return {"text": "", "generation_seconds": 0.0, "error": "no_generator"}
    start = time.time()
    outputs = generator.generate(
        [prompt],
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        logprobs=0,
    )
    output = dict(outputs[0] if outputs else {"text": ""})
    output.setdefault("generation_seconds", time.time() - start)
    return output


def _strip_fenced_json(text: str) -> str:
    value = str(text or "").strip()
    if value.startswith("```"):
        if value.lower().startswith("```json"):
            value = value[7:]
        else:
            value = value[3:]
        if value.endswith("```"):
            value = value[:-3]
    first = value.find("{")
    last = value.rfind("}")
    if first >= 0 and last >= first:
        return value[first:last + 1]
    return value.strip()


def parse_cera_output(raw_response: Any) -> Tuple[Optional[CERAOutput], Optional[str]]:
    if isinstance(raw_response, CERAOutput):
        return raw_response, None
    if isinstance(raw_response, Mapping):
        return CERAOutput.from_dict(raw_response), None
    try:
        payload = json.loads(_strip_fenced_json(str(raw_response or "")))
    except Exception:
        return None, "json_parse_error"
    if not isinstance(payload, Mapping):
        return None, "json_parse_error"
    return CERAOutput.from_dict(payload), None


def _base_result(
    *,
    enabled: bool,
    stage: str,
    shadow_only: bool,
    reject_reason: str = "",
    legacy_heuristic_usage_count: int = 0,
    metadata: Optional[Mapping[str, Any]] = None,
) -> CERACommitResult:
    return CERACommitResult(
        enabled=enabled,
        stage=stage,
        shadow_only=shadow_only,
        reject_reason=reject_reason,
        legacy_heuristic_usage_count=legacy_heuristic_usage_count,
        metadata=dict(metadata or {}),
    )


def _admissibility_metadata(admissible_set: Any) -> Dict[str, Any]:
    payload = admissible_set.to_dict() if hasattr(admissible_set, "to_dict") else {}
    derivations = payload.get("derivations") or []
    admissible = payload.get("admissible_derivations") or []
    classes = payload.get("projected_answer_classes") or {}
    return {
        "cera_derivation_count": len(derivations),
        "cera_admissible_derivation_count": len(admissible),
        "cera_projected_answer_class_count": len(classes),
        "cera_review_eligible": bool(payload.get("review_eligible", False)),
        "cera_admissibility_reject_reason": str(payload.get("reject_reason", "")),
    }


def _query_semantic_provenance(
    pre_contract: Any,
    result_context: Mapping[str, Any],
    planner_metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    question_frame = result_context.get("question_frame")
    qf = dict(question_frame) if isinstance(question_frame, Mapping) else {}
    contract_metadata = dict(getattr(pre_contract, "metadata", {}) or {})
    legacy_used = bool(contract_metadata.get("legacy_question_frame_used", False))
    if legacy_used:
        source = str(
            contract_metadata.get("legacy_question_frame_provenance")
            or "structural_cert_utils.parse_question_frame"
        )
    elif result_context.get("question_operation"):
        source = "result_context.question_operation"
    else:
        source = "default_unknown"
    semantic_rejection_prefixes = (
        "query_operation_incompatible:",
        "query_answer_domain_incompatible:",
        "query_projection_incompatible:",
    )
    rejection_reasons = [
        str(reason)
        for rejection in (planner_metadata.get("cera_planner_plan_rejections") or [])
        if isinstance(rejection, Mapping)
        for reason in (rejection.get("reasons") or [])
        if str(reason).startswith(semantic_rejection_prefixes)
    ]
    return {
        "cera_query_semantic_source": source,
        "cera_legacy_question_frame_used": legacy_used,
        "cera_question_frame_operator": str(
            qf.get("operator") or result_context.get("question_operation") or "unknown"
        ),
        "cera_question_frame_polarity": str(qf.get("polarity") or "neutral"),
        "cera_allowed_operation_hypotheses": list(
            getattr(pre_contract, "candidate_independent_operation_hypotheses", []) or []
        ),
        "cera_allowed_answer_domains": list(
            getattr(pre_contract, "allowed_answer_domains", []) or []
        ),
        "cera_allowed_projection_operators": list(
            getattr(pre_contract, "allowed_projection_operators", []) or []
        ),
        "cera_query_semantic_rejection_reasons": rejection_reasons,
    }


def _round6_metadata(lattice: Any, contrast: Any, support_v3: Any) -> Dict[str, Any]:
    lattice_payload = lattice.to_dict() if hasattr(lattice, "to_dict") else {}
    contrast_payload = contrast.to_dict() if hasattr(contrast, "to_dict") else {}
    support_payload = support_v3.to_dict() if hasattr(support_v3, "to_dict") else {}
    stage_counts = dict(lattice_payload.get("stage_counts") or {})
    members = lattice_payload.get("members") or []
    mismatch_count = sum(
        1 for item in members
        if isinstance(item, Mapping) and not bool(item.get("candidate_observation_equivalent", False))
    )
    return {
        "cera_round6_contract_version": "E71_v4",
        "cera_lattice_stage_counts": stage_counts,
        "cera_lattice_member_count": int(stage_counts.get("L0_explored_derivations", 0) or 0),
        "cera_lattice_l1_roundtrip_valid_count": int(stage_counts.get("L1_roundtrip_valid", 0) or 0),
        "cera_lattice_l4_evidence_grounded_count": int(stage_counts.get("L4_evidence_grounded", 0) or 0),
        "cera_lattice_l6_quotient_class_count": int(stage_counts.get("L6_quotient_classes", 0) or 0),
        "cera_lattice_answer_class_count": int(lattice_payload.get("answer_class_count", 0) or 0),
        "cera_lattice_compression_ratio": float(lattice_payload.get("compression_ratio", 0.0) or 0.0),
        "cera_lattice_candidate_observation_mismatch_count": mismatch_count,
        "cera_contrast_ready": bool(contrast_payload.get("ready_for_cera", False)),
        "cera_contrast_original_class_count": len(contrast_payload.get("original_classes") or []),
        "cera_contrast_alternative_class_count": len(contrast_payload.get("alternative_classes") or []),
        "cera_contrast_alternative_answer_class_count": int(contrast_payload.get("retained_alternative_answer_class_count", 0) or 0),
        "cera_contrast_unresolved_ambiguity_count": len(contrast_payload.get("unresolved_ambiguities") or []),
        "cera_contrast_unresolved_ambiguities": list(contrast_payload.get("unresolved_ambiguities") or []),
        "cera_original_support_v3_hypothesis_count": len(support_payload.get("hypotheses") or []),
        "cera_original_support_v3_roundtrip_executable": bool(support_payload.get("contains_executable_roundtrip_support", False)),
        "cera_original_support_v3_graph_anchor_only": bool(support_payload.get("contains_graph_anchor_only", False)),
        "cera_original_support_v3_ambiguity_count": int(support_payload.get("ambiguity_count", 0) or 0),
        "cera_original_support_v3_level_distribution": dict(support_payload.get("support_level_distribution") or {}),
    }


def _representative_id_from_class_payload(quotient_class: Mapping[str, Any]) -> str:
    representatives = [str(item) for item in quotient_class.get("representative_ids") or [] if str(item)]
    if representatives:
        return sorted(representatives)[0]
    members = [str(item) for item in quotient_class.get("member_derivation_ids") or [] if str(item)]
    return sorted(members)[0] if members else ""


def _round7_paired_role_interventions(lattice: Any, derivations: Sequence[Any], graph: Any) -> List[Any]:
    if graph is None:
        return []
    lattice_payload = lattice.to_dict() if hasattr(lattice, "to_dict") else {}
    classes = [
        item for item in lattice_payload.get("quotient_classes") or []
        if isinstance(item, Mapping)
    ]
    original_classes = sorted(
        [
            item for item in classes
            if item.get("original_support_members")
            and item.get("roundtrip_valid")
            and item.get("provenance_complete")
            and item.get("evidence_grounded")
        ],
        key=lambda item: _class_id_sort_key(str(item.get("class_id", ""))),
    )
    alternative_classes = sorted(
        [
            item for item in classes
            if item.get("alternative_members")
            and not item.get("original_support_members")
            and item.get("roundtrip_valid")
            and item.get("contract_compatible")
            and item.get("provenance_complete")
            and item.get("evidence_grounded")
            and not item.get("fallback_only")
        ],
        key=lambda item: _class_id_sort_key(str(item.get("class_id", ""))),
    )
    if not original_classes or not alternative_classes:
        return []
    derivation_by_id = {str(getattr(item, "derivation_id", "")): item for item in derivations}
    left = derivation_by_id.get(_representative_id_from_class_payload(original_classes[0]))
    right = derivation_by_id.get(_representative_id_from_class_payload(alternative_classes[0]))
    if left is None or right is None:
        return []
    pairs: List[Any] = []
    for role in ("TARGET_ENTITY", "TARGET_MEASURE"):
        try:
            pairs.extend(build_role_binding_substitution_pairs(left, right, graph, role=role))
        except Exception:
            continue
    return pairs


def _round7_metadata(compact_contrast: Any) -> Dict[str, Any]:
    payload = compact_contrast.to_dict() if hasattr(compact_contrast, "to_dict") else {}
    registry = payload.get("registry") if isinstance(payload.get("registry"), Mapping) else {}
    states = payload.get("states") if isinstance(payload.get("states"), Mapping) else {}
    return {
        "cera_round7_compact_contrast_version": str(payload.get("contrast_version", "")),
        "cera_round7_contrast_constructible": bool(states.get("contrast_constructible", False)),
        "cera_round7_contrast_compact": bool(states.get("contrast_compact", False)),
        "cera_round7_repair_eligible": bool(states.get("repair_eligible", False)),
        "cera_round7_paired_intervention_count": len(payload.get("paired_interventions") or []),
        "cera_round7_separating_intervention_count": len(payload.get("separating_interventions") or []),
        "cera_round7_contrast_unknown_count": len(payload.get("unknowns") or []),
        "cera_round7_contrast_unknowns": list(payload.get("unknowns") or []),
        "cera_round7_registry_evidence_count": len(registry.get("evidence_records") or []),
        "cera_round7_registry_derivation_count": len(registry.get("derivation_records") or []),
        "cera_round7_registry_hypothesis_count": len(registry.get("hypothesis_records") or []),
        "cera_round7_registry_intervention_count": len(registry.get("intervention_records") or []),
    }


def _round8_metadata(compact_contrast: Any, basis: Sequence[Any], behavior_classes: Sequence[Any]) -> Dict[str, Any]:
    payload = compact_contrast.to_dict() if hasattr(compact_contrast, "to_dict") else {}
    registry = payload.get("registry") if isinstance(payload.get("registry"), Mapping) else {}
    states = payload.get("states") if isinstance(payload.get("states"), Mapping) else {}
    return {
        "cera_round8_compact_contrast_version": str(payload.get("contrast_version", "")),
        "cera_round8_contrast_constructible": bool(states.get("contrast_constructible", False)),
        "cera_round8_contrast_registry_complete": bool(states.get("contrast_registry_complete", False)),
        "cera_round8_contrast_compact": bool(states.get("contrast_compact", False)),
        "cera_round8_repair_eligible": bool(states.get("repair_eligible", False)),
        "cera_round8_basis_count": len(basis or []),
        "cera_round8_behavior_class_count": len(behavior_classes or []),
        "cera_round8_separating_intervention_count": len(payload.get("separating_interventions") or []),
        "cera_round8_contrast_unknown_count": len(payload.get("unknowns") or []),
        "cera_round8_contrast_unknowns": list(payload.get("unknowns") or []),
        "cera_round8_registry_evidence_count": len(registry.get("evidence_records") or []),
        "cera_round8_registry_derivation_count": len(registry.get("derivation_records") or []),
        "cera_round8_registry_hypothesis_count": len(registry.get("hypothesis_records") or []),
        "cera_round8_registry_intervention_count": len(registry.get("intervention_records") or []),
    }


def _class_id_sort_key(class_id: str) -> Tuple[int, str]:
    if class_id.startswith("QC"):
        try:
            return int(class_id[2:]), class_id
        except ValueError:
            return 10**9, class_id
    return 10**9, class_id


def _contrast_anchor_derivation_id(contrast: Any, lattice: Any) -> str:
    contrast_payload = contrast.to_dict() if hasattr(contrast, "to_dict") else {}
    lattice_payload = lattice.to_dict() if hasattr(lattice, "to_dict") else {}
    classes = {
        str(item.get("class_id", "")): item
        for item in lattice_payload.get("quotient_classes") or []
        if isinstance(item, Mapping)
    }
    for class_id in sorted((contrast_payload.get("alternative_classes") or []), key=_class_id_sort_key):
        qclass = classes.get(str(class_id))
        if not qclass:
            continue
        representatives = [str(item) for item in qclass.get("representative_ids") or [] if str(item)]
        if representatives:
            return sorted(representatives)[0]
        members = [str(item) for item in qclass.get("member_derivation_ids") or [] if str(item)]
        if members:
            return sorted(members)[0]
    return ""


def _compact_lattice_packet_payload(lattice: Any, contrast: Any) -> Dict[str, Any]:
    lattice_payload = lattice.to_dict() if hasattr(lattice, "to_dict") else {}
    contrast_payload = contrast.to_dict() if hasattr(contrast, "to_dict") else {}
    retained = set(contrast_payload.get("original_classes") or []) | set(contrast_payload.get("alternative_classes") or [])
    quotient_classes = [
        item for item in lattice_payload.get("quotient_classes") or []
        if isinstance(item, Mapping) and item.get("class_id") in retained
    ]
    return {
        "lattice_version": lattice_payload.get("lattice_version", ""),
        "stage_counts": lattice_payload.get("stage_counts", {}),
        "answer_class_count": lattice_payload.get("answer_class_count", 0),
        "quotient_class_count": lattice_payload.get("quotient_class_count", 0),
        "compression_ratio": lattice_payload.get("compression_ratio", 0.0),
        "budget_trace": lattice_payload.get("budget_trace", []),
        "notes": lattice_payload.get("notes", []),
        "retained_quotient_classes": quotient_classes,
    }


def _compact_v3_alternative_derivation(compact_contrast: Any, derivations: Sequence[Any]) -> Optional[Any]:
    payload = compact_contrast.to_dict() if hasattr(compact_contrast, "to_dict") else {}
    alternative = payload.get("alternative_hypothesis") if isinstance(payload.get("alternative_hypothesis"), Mapping) else {}
    derivation_id = str(alternative.get("derivation_id") or "")
    if not derivation_id:
        derivation_ref = str(alternative.get("derivation_ref") or "")
        registry = payload.get("registry") if isinstance(payload.get("registry"), Mapping) else {}
        for record in registry.get("derivation_records") or []:
            if isinstance(record, Mapping) and str(record.get("derivation_ref") or "") == derivation_ref:
                derivation_id = str(record.get("derivation_id") or "")
                break
    if not derivation_id:
        return None
    return next((derivation for derivation in derivations if str(getattr(derivation, "derivation_id", "")) == derivation_id), None)


def _promote_admissibility_from_compact_v3(
    admissible_set: Any,
    compact_contrast: Any,
    derivations: Sequence[Any],
) -> Tuple[Any, Optional[Any]]:
    if not getattr(compact_contrast, "repair_eligible", False):
        return admissible_set, None
    reviewed_derivation = _compact_v3_alternative_derivation(compact_contrast, derivations)
    if reviewed_derivation is None:
        return admissible_set, None
    answer_class = inference_answer_key(getattr(reviewed_derivation, "projected_answer", "")).compact()
    notes = list(getattr(admissible_set, "notes", []) or [])
    notes.append("round8_compact_v3_repair_eligible_promoted_legacy_admissibility")
    promoted = replace(
        admissible_set,
        admissible_derivations=[reviewed_derivation],
        projected_answer_classes={answer_class: [getattr(reviewed_derivation, "derivation_id", "")]},
        review_eligible=True,
        selected_derivation_id=str(getattr(reviewed_derivation, "derivation_id", "")),
        selected_candidate_id=str(getattr(reviewed_derivation, "source_candidate_id", "")),
        ambiguity_count=0,
        reject_reason="",
        notes=notes,
    )
    return promoted, reviewed_derivation


def _packet_metadata_fields(packet: Any, packet_hash: str, original_answer: str) -> Dict[str, Any]:
    cf_available = 0
    cf_observed = 0
    for item in getattr(packet, "counterfactual_chain", []) or []:
        observed = getattr(item, "observed_effect", {}) or {}
        if observed.get("available") and observed.get("candidate_specific"):
            cf_available += 1
            if observed.get("changed"):
                cf_observed += 1
    outside_support = sum(
        1 for item in getattr(packet, "support_chain", []) or []
        if getattr(item, "provenance", "") == "executor_cell_not_in_evidence_subgraph"
    )
    metadata = getattr(packet, "metadata", {}) or {}
    original_hypotheses = (getattr(packet, "original_support_hypothesis_set", {}) or {}).get("hypotheses", [])
    packet_json = packet.to_json(pretty=False) if hasattr(packet, "to_json") else canonical_json(packet)
    return {
        "cera_original_answer": str(original_answer or ""),
        "cera_candidate_under_review": str(getattr(packet, "candidate_under_review", "") or ""),
        "cera_packet_hash": packet_hash,
        "cera_packet_token_length": max(1, (len(packet_json) + 3) // 4) if packet_json else 0,
        "cera_query_contract_hash": metadata.get("query_contract_hash", ""),
        "cera_query_contract_pre_evidence": bool((getattr(packet, "query_contract", {}) or {}).get("metadata", {}).get("pre_evidence")),
        "cera_original_certificate_available": bool(getattr(packet, "original_certificate_available", False)),
        "cera_original_support_hypothesis_count": len(original_hypotheses) if isinstance(original_hypotheses, list) else 0,
        "cera_original_support_executable": bool((getattr(packet, "original_support_hypothesis_set", {}) or {}).get("contains_executable_derivation")),
        "cera_candidate_counterfactual_available_count": cf_available,
        "cera_candidate_observed_counterfactual_count": cf_observed,
        "cera_outside_evidence_support_count": outside_support,
        "cera_row_major_context_cell_count": int(metadata.get("row_major_context_cell_count", 0) or 0),
        "cera_allow_row_major_context": bool(metadata.get("allow_row_major_context", False)),
        "cera_reviewed_derivation_id": metadata.get("reviewed_derivation_id", ""),
        "cera_reviewed_derivation_replay_mode": metadata.get("reviewed_derivation_replay_mode", ""),
        "cera_admissible_derivation_count": int(metadata.get("admissible_derivation_count", 0) or 0),
        "cera_projected_answer_class_count": int(metadata.get("projected_answer_class_count", 0) or 0),
    }


def run_causal_epistemic_repair(
    *,
    question: str,
    original_answer: str,
    cert_info: Mapping[str, Any],
    graph: Any = None,
    evidence: Any = None,
    table_json: Optional[Mapping[str, Any]] = None,
    all_exec_candidates: Optional[Sequence[Any]] = None,
    generator: Any = None,
    args: Any = None,
    result_context: Optional[Mapping[str, Any]] = None,
    legacy_heuristic_usage_count: int = 0,
) -> CERACommitResult:
    cera_stage = str(getattr(args, "cera_stage", "E71") if args is not None else "E71").upper()
    round6_e71_v4 = bool(getattr(args, "cera_round6_e71_v4", False) if args is not None else False)
    stage = "E72_cera_shadow" if cera_stage == "E72" else ("E71_v4_packet_shadow" if round6_e71_v4 else "E71_packet_shadow")
    shadow_only = bool(getattr(args, "cera_shadow_only", True) if args is not None else True)
    template_version = str(getattr(args, "cera_template_version", DEFAULT_CERA_TEMPLATE_VERSION) if args is not None else DEFAULT_CERA_TEMPLATE_VERSION)
    use_cera_v3 = template_version == CERA_V3_TEMPLATE_VERSION
    enabled = True
    if not str(original_answer or "").strip():
        return _base_result(
            enabled=enabled,
            stage=stage,
            shadow_only=shadow_only,
            reject_reason="B0_INVALID",
            legacy_heuristic_usage_count=legacy_heuristic_usage_count,
            metadata={"cera_planner_called": False},
        )
    if isinstance(result_context, MethodInferenceContext):
        context = result_context.to_dict()
    else:
        assert_method_context_clean(result_context or {})
        context = dict(result_context or {})
    pre_contract = build_pre_evidence_query_contract(
        question=question,
        question_frame=context.get("question_frame"),
        result_context=context,
        initial_answer=original_answer,
        graph_stats=context.get("graph_stats"),
    )
    planner_derivations, planner_metadata, plan_closure = _run_typed_derivation_planner(
        question=question,
        graph=graph,
        table_json=table_json,
        pre_contract=pre_contract,
        generator=generator,
        args=args,
        original_answer=original_answer,
    )
    if bool(getattr(args, "cera_stepwise_trace", False) if args is not None else False):
        trace_metadata = dict(planner_metadata)
        trace_metadata.update(_query_semantic_provenance(pre_contract, context, planner_metadata))
        return _base_result(
            enabled=enabled,
            stage="ROUND12_TYPED_TRACE_SHADOW",
            shadow_only=True,
            reject_reason="round12_trace_shadow_only",
            legacy_heuristic_usage_count=legacy_heuristic_usage_count,
            metadata=trace_metadata,
        )
    candidate_derivations = materialize_derivations(
        certified_candidates=_candidate_rows(cert_info),
        live_candidates=all_exec_candidates,
        graph=graph,
    )
    planner_closure_required = bool(use_cera_v3 and _planner_enabled(args))
    plan_closure_unavailable = bool(planner_closure_required and plan_closure is None)
    frontier_derivations = build_symmetric_derivation_frontier(
        contract=pre_contract,
        graph=graph,
        evidence=evidence,
        existing_derivations=list(candidate_derivations) + list(planner_derivations),
        max_group_size=int(getattr(args, "cera_frontier_max_group_size", 16) if args is not None else 16),
    )
    audit_pool = build_audit_derivation_pool(
        candidate_derivations=candidate_derivations,
        planner_derivations=planner_derivations,
        frontier_derivations=frontier_derivations,
    )
    decision_pool = build_decision_derivation_pool(
        planner_derivations=planner_derivations,
        original_answer=original_answer,
        contract=pre_contract,
        source_policy=(
            "plan_closure_v1"
            if plan_closure is not None
            else ("plan_closure_unavailable" if plan_closure_unavailable else "planner_only")
        ),
    )
    all_derivations = audit_pool.derivations
    decision_derivations = list(decision_pool.derivations)
    frontier_metadata = {
        "cera_candidate_derivation_count": len(candidate_derivations),
        "cera_planner_derivation_count": len(planner_derivations),
        "cera_frontier_derivation_count": len(frontier_derivations),
        "cera_symmetric_frontier_enabled": bool(frontier_derivations),
    }
    frontier_metadata.update(planner_metadata)
    frontier_metadata.update(_query_semantic_provenance(pre_contract, context, planner_metadata))
    frontier_metadata.update(audit_pool.metadata())
    frontier_metadata.update(decision_pool.metadata())
    if plan_closure is not None:
        support_partition = partition_support(plan_closure, initial_proposal_answer=original_answer)
        frontier_metadata.update({
            "cera_round9_partition_original_count": len(support_partition.original_support),
            "cera_round9_partition_alternative_count": len(support_partition.alternative_support),
            "cera_round9_partition_disjoint": bool(support_partition.disjoint),
            "cera_round9_partition_exhaustive": bool(support_partition.exhaustive),
            "cera_round9_partition_equivalence_policy": support_partition.equivalence_policy,
            "cera_round9_initial_proposal_answer_key": support_partition.initial_proposal_answer_key,
        })
    repair_derivations = [
        derivation for derivation in all_derivations
        if not answers_equivalent(derivation.projected_answer, original_answer)
    ]
    admissible_set = build_admissible_candidate_set(
        contract=pre_contract,
        derivations=repair_derivations,
        graph=graph,
        evidence=evidence,
    )
    original_support_hypothesis_set = reconstruct_original_support_hypotheses(
        original_answer=original_answer,
        derivations=all_derivations,
        graph=graph,
        evidence=evidence,
    )
    budget_trace = [{
        "budget_name": "cera_frontier_max_group_size",
        "budget_value": int(getattr(args, "cera_frontier_max_group_size", 16) if args is not None else 16),
        "pre_budget_count": len(candidate_derivations),
        "post_budget_count": len(frontier_derivations),
        "truncated": False,
        "truncation_policy": "frontier_group_generation_skips_groups_larger_than_budget",
    }]
    derivation_lattice = build_derivation_lattice(
        contract=pre_contract,
        derivations=all_derivations,
        original_answer=original_answer,
        graph=graph,
        evidence=evidence,
        budget_trace=budget_trace,
    )
    minimal_contrast_set = build_minimal_contrast_set(
        lattice=derivation_lattice,
        original_support_hypothesis_set=original_support_hypothesis_set,
    )
    original_support_v3 = build_original_support_symmetry_v3(
        original_answer=original_answer,
        lattice=derivation_lattice,
        original_support_hypothesis_set=original_support_hypothesis_set,
    )
    compact_contrast_v2 = build_compact_behavioral_contrast_v2(
        lattice=derivation_lattice,
        original_support_symmetry_v3=original_support_v3,
        paired_interventions=_round7_paired_role_interventions(derivation_lattice, all_derivations, graph),
        query_semantics={
            "answer_domain": pre_contract.answer_domain,
            "allowed_answer_domains": list(pre_contract.allowed_answer_domains),
            "allowed_projection_operators": list(pre_contract.allowed_projection_operators),
            "candidate_independent_operation_hypotheses": list(pre_contract.candidate_independent_operation_hypotheses),
        },
    )
    fixed_basis = build_sample_fixed_role_intervention_basis(decision_derivations, graph)
    behavior_classes = build_basis_relative_behavior_classes(decision_derivations, graph, fixed_basis)
    compact_contrast_v3 = build_compact_behavioral_contrast_v3(
        derivations=decision_derivations,
        behavior_classes=behavior_classes,
        basis=fixed_basis,
        original_answer=original_answer,
        query_semantics={
            "answer_domain": pre_contract.answer_domain,
            "allowed_answer_domains": list(pre_contract.allowed_answer_domains),
            "allowed_projection_operators": list(pre_contract.allowed_projection_operators),
            "candidate_independent_operation_hypotheses": list(pre_contract.candidate_independent_operation_hypotheses),
        },
    )
    round6_metadata = _round6_metadata(derivation_lattice, minimal_contrast_set, original_support_v3)
    round7_metadata = _round7_metadata(compact_contrast_v2)
    round8_metadata = _round8_metadata(compact_contrast_v3, fixed_basis, behavior_classes)
    if use_cera_v3 and compact_contrast_v3.repair_eligible and not admissible_set.review_eligible:
        original_reject_reason = admissible_set.reject_reason
        admissible_set, compact_selected = _promote_admissibility_from_compact_v3(
            admissible_set,
            compact_contrast_v3,
            decision_derivations,
        )
        round8_metadata.update({
            "cera_round8_compact_v3_promoted_legacy_admissibility": compact_selected is not None,
            "cera_round8_compact_v3_legacy_reject_reason": original_reject_reason,
            "cera_round8_compact_v3_selected_derivation_id": (
                str(getattr(compact_selected, "derivation_id", "")) if compact_selected is not None else ""
            ),
        })
    else:
        round8_metadata.update({
            "cera_round8_compact_v3_promoted_legacy_admissibility": False,
            "cera_round8_compact_v3_legacy_reject_reason": "",
            "cera_round8_compact_v3_selected_derivation_id": "",
        })
    planner_failure_reason = "planner_generation_error" if planner_metadata.get("cera_planner_generation_error") else ""
    if plan_closure_unavailable:
        metadata = _admissibility_metadata(admissible_set)
        metadata.update(frontier_metadata)
        metadata.update(round6_metadata)
        metadata.update(round7_metadata)
        metadata.update(round8_metadata)
        return _base_result(
            enabled=enabled,
            stage=stage,
            shadow_only=shadow_only,
            reject_reason=planner_failure_reason or "plan_closure_unavailable",
            legacy_heuristic_usage_count=legacy_heuristic_usage_count,
            metadata=metadata,
        )
    if use_cera_v3 and not compact_contrast_v3.repair_eligible:
        metadata = _admissibility_metadata(admissible_set)
        metadata.update(frontier_metadata)
        metadata.update(round6_metadata)
        metadata.update(round7_metadata)
        metadata.update(round8_metadata)
        return _base_result(
            enabled=enabled,
            stage=stage,
            shadow_only=shadow_only,
            reject_reason="round8_repair_not_eligible",
            legacy_heuristic_usage_count=legacy_heuristic_usage_count,
            metadata=metadata,
        )
    if not admissible_set.review_eligible:
        reviewed_derivation = None
        if round6_e71_v4 and minimal_contrast_set.ready_for_cera:
            anchor_id = _contrast_anchor_derivation_id(minimal_contrast_set, derivation_lattice)
            reviewed_derivation = next((d for d in all_derivations if d.derivation_id == anchor_id), None)
        if reviewed_derivation is not None:
            candidate = next(
                (c for c in _candidate_rows(cert_info) if c.candidate_id == reviewed_derivation.source_candidate_id),
                None,
            )
            if candidate is None:
                candidate = _candidate_from_derivation(reviewed_derivation)
            cert = candidate.certificate or {}
            live_candidate = match_live_exec_candidate(candidate, all_exec_candidates)
            max_excerpt_cells = int(getattr(args, "cera_table_excerpt_max_cells", 32) if args is not None else 32)
            allow_row_major_context = bool(getattr(args, "cera_allow_row_major_context", False) if args is not None else False)
            packet = build_causal_evidence_packet(
                question=question,
                original_answer=original_answer,
                candidate=candidate,
                graph=graph,
                evidence=evidence,
                table_json=table_json,
                cert_info=cert_info,
                exec_candidate=live_candidate,
                pre_evidence_contract=pre_contract,
                admissible_candidate_set=admissible_set,
                reviewed_derivation=reviewed_derivation,
                original_support_hypothesis_set=original_support_hypothesis_set,
                derivation_lattice=_compact_lattice_packet_payload(derivation_lattice, minimal_contrast_set),
                minimal_contrast_set=minimal_contrast_set.to_dict(),
                compact_behavioral_contrast_v2=compact_contrast_v2.to_dict(),
                compact_behavioral_contrast_v3=compact_contrast_v3.to_dict(),
                original_support_symmetry_v3=original_support_v3.to_dict(),
                question_frame=context.get("question_frame"),
                graph_stats=context.get("graph_stats"),
                edge_reliability_diag=context.get("edge_reliability_diag"),
                layout_risk=context.get("layout_risk", 0.0),
                max_excerpt_cells=max_excerpt_cells,
                allow_row_major_context=allow_row_major_context,
            )
            packet_hash = packet.metadata.get("packet_hash") or stable_packet_hash(packet)
            result = CERACommitResult(
                enabled=enabled,
                packet_built=True,
                stage=stage,
                triggered=True,
                shadow_only=shadow_only,
                final_committed=False,
                evidence_packet_hash=packet_hash,
                support_chain_len=len(packet.support_chain),
                counterfactual_chain_len=len(packet.counterfactual_chain),
                candidate_scci=_as_float(cert.get("scci"), 0.0),
                candidate_effective_coverage=_as_float(cert.get("candidate_effective_evidence_coverage"), 0.0),
                legacy_heuristic_usage_count=legacy_heuristic_usage_count,
                packet=packet,
                reject_reason=admissible_set.reject_reason,
                metadata=_packet_metadata_fields(packet, packet_hash, original_answer),
            )
            result.metadata.update(frontier_metadata)
            result.metadata.update(round6_metadata)
            result.metadata.update(round7_metadata)
            result.metadata.update(round8_metadata)
            result.metadata["cera_contrast_anchor_derivation_id"] = reviewed_derivation.derivation_id
            if cera_stage != "E72":
                return result
        metadata = _admissibility_metadata(admissible_set)
        metadata.update(frontier_metadata)
        metadata.update(round6_metadata)
        metadata.update(round7_metadata)
        metadata.update(round8_metadata)
        return _base_result(
            enabled=enabled,
            stage=stage,
            shadow_only=shadow_only,
            reject_reason=planner_failure_reason or admissible_set.reject_reason,
            legacy_heuristic_usage_count=legacy_heuristic_usage_count,
            metadata=metadata,
        )
    reviewed_derivation = next(
        (
            derivation for derivation in admissible_set.admissible_derivations
            if derivation.derivation_id == admissible_set.selected_derivation_id
        ),
        None,
    )
    candidate = next(
        (c for c in _candidate_rows(cert_info) if c.candidate_id == admissible_set.selected_candidate_id),
        None,
    )
    if candidate is None and reviewed_derivation is not None:
        candidate = _candidate_from_derivation(reviewed_derivation)
    if candidate is None:
        metadata = _admissibility_metadata(admissible_set)
        metadata.update(frontier_metadata)
        metadata.update(round6_metadata)
        metadata.update(round7_metadata)
        metadata.update(round8_metadata)
        return _base_result(
            enabled=enabled,
            stage=stage,
            shadow_only=shadow_only,
            reject_reason="no_admissible_rescue_candidate",
            legacy_heuristic_usage_count=legacy_heuristic_usage_count,
            metadata=metadata,
        )

    cert = candidate.certificate or {}
    live_candidate = match_live_exec_candidate(candidate, all_exec_candidates)
    max_excerpt_cells = int(getattr(args, "cera_table_excerpt_max_cells", 32) if args is not None else 32)
    allow_row_major_context = bool(getattr(args, "cera_allow_row_major_context", False) if args is not None else False)
    packet = build_causal_evidence_packet(
        question=question,
        original_answer=original_answer,
        candidate=candidate,
        graph=graph,
        evidence=evidence,
        table_json=table_json,
        cert_info=cert_info,
        exec_candidate=live_candidate,
        pre_evidence_contract=pre_contract,
        admissible_candidate_set=admissible_set,
        reviewed_derivation=reviewed_derivation,
        original_support_hypothesis_set=original_support_hypothesis_set,
        derivation_lattice=_compact_lattice_packet_payload(derivation_lattice, minimal_contrast_set),
        minimal_contrast_set=minimal_contrast_set.to_dict(),
        compact_behavioral_contrast_v2=compact_contrast_v2.to_dict(),
        compact_behavioral_contrast_v3=compact_contrast_v3.to_dict(),
        original_support_symmetry_v3=original_support_v3.to_dict(),
        question_frame=context.get("question_frame"),
        graph_stats=context.get("graph_stats"),
        edge_reliability_diag=context.get("edge_reliability_diag"),
        layout_risk=context.get("layout_risk", 0.0),
        max_excerpt_cells=max_excerpt_cells,
        allow_row_major_context=allow_row_major_context,
    )
    packet_hash = packet.metadata.get("packet_hash") or stable_packet_hash(packet)
    result = CERACommitResult(
        enabled=enabled,
        packet_built=True,
        stage=stage,
        triggered=True,
        shadow_only=shadow_only,
        final_committed=False,
        evidence_packet_hash=packet_hash,
        support_chain_len=len(packet.support_chain),
        counterfactual_chain_len=len(packet.counterfactual_chain),
        candidate_scci=_as_float(cert.get("scci"), 0.0),
        candidate_effective_coverage=_as_float(cert.get("candidate_effective_evidence_coverage"), 0.0),
        legacy_heuristic_usage_count=legacy_heuristic_usage_count,
        packet=packet,
        metadata=_packet_metadata_fields(packet, packet_hash, original_answer),
    )
    result.metadata.update(frontier_metadata)
    result.metadata.update(round6_metadata)
    result.metadata.update(round7_metadata)
    result.metadata.update(round8_metadata)

    if use_cera_v3:
        if cera_stage != "E72":
            return result
    else:
        if bool(cert.get("evidence_fallback", False)):
            result.triggered = False
            result.reject_reason = "evidence_fallback"
            return result
        if not packet.support_chain:
            result.triggered = False
            result.reject_reason = "no_support_chain"
            return result
        if (
            not packet.counterfactual_chain
            and bool(getattr(args, "cera_require_counterfactual_reference", True) if args is not None else True)
            and not bool(getattr(args, "cera_allow_support_only", False) if args is not None else False)
        ):
            result.reject_reason = "no_counterfactual_chain"
            return result
        if cera_stage != "E72":
            return result
    if generator is None:
        result.reject_reason = "no_generator"
        return result

    require_derivation_program = bool(getattr(args, "cera_require_derivation_program", True) if args is not None else True)
    require_counterfactual_reference = bool(getattr(args, "cera_require_counterfactual_reference", True) if args is not None else True)
    allow_support_only = bool(getattr(args, "cera_allow_support_only", False) if args is not None else False)
    prompt = build_cera_prompt(
        packet,
        template_version=template_version,
        require_derivation_program=require_derivation_program,
        require_counterfactual_reference=require_counterfactual_reference,
    )
    result.prompt = prompt
    gen = call_cera_agent(
        generator,
        prompt,
        max_tokens=int(getattr(args, "cera_max_tokens", 512) if args is not None else 512),
        temperature=float(getattr(args, "cera_temperature", 0.0) if args is not None else 0.0),
        top_p=float(getattr(args, "top_p", 1.0) if args is not None else 1.0),
    )
    audit = build_cera_request_audit(
        prompt=prompt,
        packet=packet,
        query_contract=packet.query_contract,
        args=args,
        generator=generator,
        generation_output=gen,
    )
    result.metadata.update({
        "cera_prompt_hash": audit.get("prompt_hash"),
        "cera_request_hash": audit.get("request_hash"),
        "cera_model": audit.get("model"),
        "cera_backend": audit.get("backend"),
        "cera_api_base_url": audit.get("api_base_url"),
        "cera_sampling": audit.get("sampling"),
        "cera_api_cache_hit": audit.get("api_cache_hit"),
        "cera_latency_seconds": audit.get("latency_seconds"),
        "cera_input_tokens": audit.get("input_tokens"),
        "cera_output_tokens": audit.get("output_tokens"),
        "cera_request_audit": audit,
    })
    result.llm_called = True
    result.raw_response = str(gen.get("text", ""))
    output, parse_error = parse_cera_output(result.raw_response)
    if output is None:
        result.json_parse_success = False
        result.reject_reason = parse_error or "json_parse_error"
        result.validator_reject_reason = result.reject_reason
        result.validator = {"accepted": False, "reject_reason": result.reject_reason}
        return result

    result.output = output
    result.metadata["cera_proposed_repair_answer"] = str(output.final_answer or "")
    result.json_parse_success = True
    if use_cera_v3:
        validator = validate_cera_output_v3(output, packet)
    else:
        validator = validate_cera_output(
            output,
            packet,
            require_derivation_program=require_derivation_program,
            require_counterfactual_reference=require_counterfactual_reference,
            allow_support_only=allow_support_only,
            allow_outside_evidence_support=bool(getattr(args, "cera_allow_outside_evidence_support", False) if args is not None else False),
        )
    result.validator = validator.to_dict()
    result.validator_accept = validator.accepted
    result.validator_reject_reason = validator.reject_reason
    if not validator.accepted:
        result.reject_reason = validator.reject_reason
        return result

    decision = output.decision.strip().upper()
    result.would_commit = decision == "USE_REPAIRED"
    result.would_keep = decision == "KEEP_ORIGINAL"
    result.insufficient = decision == "INSUFFICIENT_CERTIFICATE"
    result.reject_reason = ""
    result.final_committed = False
    return result
