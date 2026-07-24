#!/usr/bin/env python3
"""Resumable runner for bounded executable search and selective proof."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from graph_builder import build_hceg  # noqa: E402
from run_cscr_pipeline import (  # noqa: E402
    OpenAIChatGenerator,
    build_structure_aware_prompt,
    extract_answer,
)

from certa.active_v1.artifact_authority import (  # noqa: E402
    ArtifactContext,
    serialize_plan_closure_v3,
)
from certa.active_v1.answer_authority import active_answer_hash  # noqa: E402
from certa.active_v1.dataset_adapter_v1 import HiTabAdapterV1  # noqa: E402
from certa.active_v1.final_method_v1 import (  # noqa: E402
    build_complete_domain_c2_view,
    canonical_typed_plan_identity,
)
from certa.active_v1.planner_bridge_v3 import (  # noqa: E402
    _constructor_active_role_ids,
    build_v3_arm_view,
    close_compiled_payload,
    compile_active_planner_payload,
)
from certa.active_v1.planner_transport_projection import (  # noqa: E402
    build_planner_transport_schema,
)
from certa.active_v1.role_contract_v3 import (  # noqa: E402
    build_role_v3_prompt,
    derive_role_v3_record,
)
from certa.egra.retrieval import FrozenE5Encoder  # noqa: E402
from certa.grounding.plan_closure import PlanClosure  # noqa: E402
from certa.planner.schema_view import build_proposal_blind_planner_view  # noqa: E402
from certa.planner.typed_planner import (  # noqa: E402
    build_typed_derivation_planner_prompt,
    build_typed_planner_response_schema,
)
from certa.reproducibility.canonical_json import canonical_json, canonical_json_hash  # noqa: E402
from certa.v2.authority import materialize_executed_registry  # noqa: E402
from certa.v2.candidates import build_candidate_universe  # noqa: E402
from certa.v2.decision import decide_proof_dominance, decide_proof_verifier  # noqa: E402
from certa.v2.proof import build_proof_record, validate_proof_record  # noqa: E402
from certa.v2.runtime import (  # noqa: E402
    build_structural_challenge_prompt,
    canonicalize_structural_response,
    challenge_applicability,
    project_structural_evidence,
)
from certa.v2.runtime import CHALLENGE_IDS  # noqa: E402
from certa.v2.search import SEARCH_SCHEDULE, merge_search_attempts  # noqa: E402
from tools.certa_final_method import EMBEDDING_FREEZE, _retrieval  # noqa: E402


BASE = Path("/home/hsh/ME/Table/EMNLP2026")
DEFAULT_OUT = (
    BASE
    / "certa_v2_outputs"
    / "CERTA_V2_BOUNDED_EXECUTABLE_PROOF_SEARCH"
)
V1_OUT = (
    BASE
    / "certa_active_v1_outputs"
    / "CERTA_FINAL_MULTI_DATASET_ADAPTER_AND_METHOD_COMPLETION"
)
ROLE_ROOT = (
    BASE
    / "certa_active_v1_outputs"
    / "CERTA_ACTIVE_V1_ROLE_V3_FINAL"
    / "freeze"
)
MATRIX_PATH = (
    BASE
    / "certa_active_v1_outputs"
    / "CERTA_ACTIVE_V1_FINAL_ASSIGNMENT_LEVEL_GROUNDING_AUTHORITY_REPLAY"
    / "freeze"
    / "CONSTRUCTOR_CAPABILITY_MATRIX.json"
)
PROOF_SCHEMA = REPO / "schemas" / "v2" / "STRUCTURAL_CHALLENGE_RESPONSE.schema.json"
VERIFIER_SCHEMA = REPO / "schemas" / "v2" / "PAIRWISE_VERIFIER.schema.json"
MODEL = "Qwen3-8B"
API_BASE = "http://127.0.0.1:30338/v1"
TEMPERATURE = 0.4
TOP_P = 1.0
MAX_FULL_EVIDENCE_CHARS = 64_000
VERIFIER_MAX_TOKENS = 512


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_json_mapping(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _readl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _writel(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(canonical_json(dict(row)) + "\n" for row in rows),
        encoding="utf-8",
    )


def _transport_schema(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _transport_schema(child)
            for key, child in value.items()
            if key != "uniqueItems"
        }
    if isinstance(value, list):
        return [_transport_schema(child) for child in value]
    return value


def build_search_views(
    *,
    question: str,
    graph: Any,
    table: Mapping[str, Any],
    role: Mapping[str, Any],
    retrieval: Mapping[str, Any] | None,
    matrix: Mapping[str, Any],
    role_schema: Mapping[str, Any],
    role_registry: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Construct the exact three proposal-blind view families."""
    active_ids = _constructor_active_role_ids(matrix)
    broad = lambda values: build_proposal_blind_planner_view(
        question=question,
        graph=graph,
        table_json=table,
        query_contract=None,
        include_table_values=values,
        legacy_query_semantics_mode="audit_only",
        allowed_signature_ids=active_ids,
    )
    if role.get("supported") is True:
        role_view = build_v3_arm_view(
            "C1_ROLE_ONLY",
            question,
            graph,
            table,
            role,
            None,
            matrix,
            output_schema=role_schema,
            canonical_registry=role_registry,
        ).view
        retrieval_view = (
            build_complete_domain_c2_view(
                question,
                graph,
                table,
                role,
                retrieval,
                matrix,
                output_schema=role_schema,
                canonical_registry=role_registry,
            ).view
            if retrieval is not None
            else broad(False)
        )
    else:
        role_view = broad(False)
        retrieval_view = broad(False)
        retrieval_view["retrieval_advisory"] = {
            "authority": "ADVISORY_UNAVAILABLE_UNSUPPORTED_ROLE",
            "reference_node_ids": [],
            "reference_count": 0,
            "complete_schema_node_count": len(retrieval_view["schema_nodes"]),
            "complete_schema_edge_count": len(retrieval_view["schema_edges"]),
        }
    return {
        "ROLE_COMPLETE": role_view,
        "RETRIEVAL_COMPLETE": retrieval_view,
        "VALUE_AWARE_PROPOSAL_BLIND": broad(True),
    }


def _call(
    generator: OpenAIChatGenerator,
    *,
    output: Path,
    split: str,
    sample_id: str,
    call_id: str,
    prompt: str,
    max_tokens: int,
    seed: int,
    schema: Mapping[str, Any] | None = None,
    temperature: float = TEMPERATURE,
) -> dict[str, Any]:
    path = output / split / "model_outputs" / sample_id / f"{call_id}.json"
    identity = {
        "prompt_sha256": canonical_json_hash(prompt),
        "schema_sha256": canonical_json_hash(schema) if schema is not None else "",
        "seed": seed,
        "temperature": temperature,
        "top_p": TOP_P,
        "max_tokens": max_tokens,
    }
    if path.is_file():
        cached = _read(path)
        if any(cached.get(key) != value for key, value in identity.items()):
            raise RuntimeError(f"cached_call_identity_mismatch:{sample_id}:{call_id}")
        return cached
    response_format = (
        {
            "type": "json_schema",
            "json_schema": {
                "name": call_id.lower().replace("-", "_"),
                "schema": dict(schema),
                "strict": True,
            },
        }
        if schema is not None
        else None
    )
    kwargs = generator._completion_request_kwargs(
        prompt=prompt,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=TOP_P,
        response_format=response_format,
    )
    kwargs["seed"] = seed
    started = time.time()
    response = generator.client.chat.completions.create(**kwargs)
    text = response.choices[0].message.content or ""
    usage = response.usage
    record = {
        "schema_version": "certa_v2_model_call_v1",
        "sample_id": sample_id,
        "call_id": call_id,
        **identity,
        "prompt": prompt,
        "request": kwargs,
        "text": text,
        "usage": {
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        },
        "generation_seconds": time.time() - started,
    }
    _write(path, record)
    ledger_path = output / "logs" / "ENDPOINT_LEDGER.jsonl"
    ledger = _readl(ledger_path) if ledger_path.is_file() else []
    ledger.append(
        {
            key: record[key]
            for key in (
                "schema_version",
                "sample_id",
                "call_id",
                "prompt_sha256",
                "schema_sha256",
                "seed",
                "temperature",
                "top_p",
                "max_tokens",
                "usage",
                "generation_seconds",
            )
        }
    )
    _writel(ledger_path, ledger)
    return record


def _namespace_closure_plan(closure: PlanClosure, plan_id: str) -> PlanClosure:
    return replace(
        closure,
        plan_id=plan_id,
        assignments=tuple(
            replace(item, plan_id=plan_id, plan_ids=(plan_id,))
            for item in closure.assignments
        ),
        executable_derivations=tuple(
            replace(
                item,
                operation_metadata={**item.operation_metadata, "plan_ids": [plan_id]},
            )
            for item in closure.executable_derivations
        ),
    )


def _merge_closures(closures: Sequence[PlanClosure]) -> PlanClosure:
    assignments = []
    derivations = {}
    seen_assignments = set()
    seen_derivations = set()
    counts: dict[str, int] = {}
    for closure in closures:
        for assignment in closure.assignments:
            assignment_identity = canonical_json(assignment.to_dict())
            if assignment_identity in seen_assignments:
                continue
            seen_assignments.add(assignment_identity)
            if (
                not assignment.derivation_id
                or assignment.derivation_id not in seen_derivations
            ):
                assignments.append(assignment)
                if assignment.derivation_id:
                    seen_derivations.add(assignment.derivation_id)
        for derivation in closure.executable_derivations:
            derivations.setdefault(derivation.derivation_id, derivation)
        for key, value in closure.outcome_counts.items():
            counts[key] = counts.get(key, 0) + int(value)
    return PlanClosure(
        plan_id="V2_BOUNDED_UNION",
        operation_family="MULTI_SIGNATURE",
        planner_version="typed_derivation_planner_v1",
        assignments=tuple(assignments),
        executable_derivations=tuple(derivations[key] for key in sorted(derivations)),
        outcome_counts=counts,
        construction_trace=("exact_six_call_typed_plan_union",),
        declared_assignment_count=sum(item.declared_assignment_count for item in closures),
        realized_assignment_count=len(assignments),
        deduplicated_program_count=len(derivations),
        resource_complete=all(item.resource_complete for item in closures),
    )


def _search(
    *,
    generator: OpenAIChatGenerator,
    output: Path,
    split: str,
    sample_id: str,
    views: Mapping[str, Mapping[str, Any]],
    graph: Any,
    matrix: Mapping[str, Any],
) -> tuple[PlanClosure, dict[str, Any]]:
    attempts = []
    compilations = {}
    for slot in SEARCH_SCHEDULE:
        view = views[slot.view_kind]
        full_schema = build_typed_planner_response_schema(
            view, require_signature_id=True
        )
        transport_schema = build_planner_transport_schema(full_schema)
        prompt = (
            build_typed_derivation_planner_prompt(view)
            + f"\n\nFrozen search slot: {slot.call_id}; propose a distinct legal proof path."
        )
        call = _call(
            generator,
            output=output,
            split=split,
            sample_id=sample_id,
            call_id=f"PLANNER_{slot.call_id}",
            prompt=prompt,
            max_tokens=512,
            seed=slot.seed,
            schema=transport_schema,
        )
        compilation = compile_active_planner_payload(call["text"], view, matrix)
        compilations[slot.call_id] = compilation
        attempts.append(
            {
                "call_id": slot.call_id,
                "view_kind": slot.view_kind,
                "call_index": slot.call_index,
                "status": "OK" if compilation.ok else "INVALID",
                "plans": list(compilation.normalized_payload.get("plans", ()))
                if compilation.ok
                else [],
                "view": view,
                "errors": list(compilation.errors),
            }
        )
    merged = merge_search_attempts(attempts)
    closures = []
    for plan in merged["plans"]:
        identity = canonical_typed_plan_identity(plan)
        source = next(
            compilation
            for compilation in compilations.values()
            if compilation.ok
            and any(
                canonical_typed_plan_identity(item) == identity
                for item in compilation.normalized_payload["plans"]
            )
        )
        payload = {
            "planner_version": source.normalized_payload["planner_version"],
            "query_semantics": source.normalized_payload["query_semantics"],
            "plans": [plan],
            "unresolved_semantics": source.normalized_payload["unresolved_semantics"],
        }
        compiled = compile_active_planner_payload(
            payload, views["VALUE_AWARE_PROPOSAL_BLIND"], matrix
        )
        if compiled.ok:
            closures.append(
                _namespace_closure_plan(
                    close_compiled_payload(compiled, graph, matrix),
                    str(plan["plan_id"]),
                )
            )
    return _merge_closures(closures), {
        **merged,
        "attempts": [
            {key: value for key, value in attempt.items() if key != "view"}
            for attempt in attempts
        ],
    }


def _proof_prompt(packet: Mapping[str, Any]) -> str:
    return (
        "CERTA Candidate Proof Agent\nReturn JSON only. Evaluate every frozen "
        "challenge for the supplied candidate. Copy all identities exactly. "
        "Cite only packet artifact_ref values. SUPPORTED requires nonempty "
        "artifact_refs and CITED_SUPPORT. FOUND requires nonempty artifact_refs "
        "and CITED_CONTRADICTION. Use UNKNOWN and INSUFFICIENT_EVIDENCE when "
        "positive support is absent. Copy challenge_applicability exactly; for "
        "false use NOT_SUPPORTED, NOT_FOUND, empty artifact_refs, and only "
        "NOT_APPLICABLE_BY_SIGNATURE. Do not generate or rank answers.\n\n"
        + canonical_json(packet)
    )


def _a_decision(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    alternatives = [
        item for item in candidates if item["candidate_source"] == "EXECUTED_REGISTRY"
    ]
    b0 = candidates[0]
    if (
        not b0["registry_refs"]
        and len(alternatives) == 1
        and len(alternatives[0]["registry_refs"]) == 1
    ):
        return {
            "action": "REPAIR",
            "selected_candidate_id": alternatives[0]["candidate_id"],
            "selected_registry_entry_id": alternatives[0]["registry_refs"][0],
            "selected_answer_hash": alternatives[0]["candidate_answer_hash"],
            "validator_approved": True,
            "failure_reasons": [],
        }
    return {
        "action": "KEEP_B0",
        "selected_candidate_id": b0["candidate_id"],
        "selected_registry_entry_id": "",
        "selected_answer_hash": b0["candidate_answer_hash"],
        "validator_approved": True,
        "failure_reasons": ["v1_deterministic_selection_not_unique"],
    }


def _materialize(
    decision: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    registry: Sequence[Mapping[str, Any]],
    b0_answer: Any,
) -> Any:
    if decision.get("action") != "REPAIR":
        return b0_answer
    candidate = next(
        item
        for item in candidates
        if item["candidate_id"] == decision["selected_candidate_id"]
    )
    matches = [
        item
        for item in registry
        if item["registry_entry_id"] == decision["selected_registry_entry_id"]
        and item["answer_hash"] == candidate["candidate_answer_hash"]
    ]
    if len(matches) != 1:
        raise ValueError(f"selected_materializer_join:{len(matches)}")
    return matches[0]["executed_answer"]


def _run_sample(
    *,
    generator: OpenAIChatGenerator,
    output: Path,
    split: str,
    runtime: Mapping[str, Any],
    adapter: HiTabAdapterV1,
    matrix: Mapping[str, Any],
    role_schema: Mapping[str, Any],
    role_registry: Mapping[str, Any],
    role_cards: Mapping[str, Any],
    encoder: FrozenE5Encoder | None,
) -> dict[str, Any]:
    sample_id = str(runtime["id"])
    table_id = str(runtime["table_id"])
    question = str(runtime["question"])
    native = adapter.resolve_table(table_id, runtime_record=runtime)
    table = adapter.canonicalize_table(native)["table_payload"]["graph_payload"]
    graph = build_hceg(table, question)
    graph_hash = canonical_json_hash(graph.to_dict())
    if split == "development":
        frozen = _read(V1_OUT / "validation" / "samples" / f"{sample_id}.json")
        b0_answer = frozen["b0_answer"]
        role = frozen["role"]
        retrieval = (frozen.get("retrieval") or {}).get("retrieval")
    else:
        b0_call = _call(
            generator,
            output=output,
            split=split,
            sample_id=sample_id,
            call_id="B0",
            prompt=build_structure_aware_prompt(table, question),
            max_tokens=32,
            seed=1728,
            temperature=0,
        )
        b0_answer = extract_answer(b0_call["text"])
        role_call = _call(
            generator,
            output=output,
            split=split,
            sample_id=sample_id,
            call_id="ROLE_V3",
            prompt=build_role_v3_prompt(question, role_cards),
            max_tokens=64,
            seed=1728,
            schema=role_schema,
            temperature=0,
        )
        role = derive_role_v3_record(
            role_call["text"], role_schema, role_registry
        )
        retrieval = None
        if role["supported"] and encoder is not None:
            retrieval = _retrieval(
                role=role,
                graph=graph,
                table=table,
                question=question,
                encoder=encoder,
                parent_sha=str(output.name),
                embedding_sha=_read(EMBEDDING_FREEZE)["file_tree_sha256"],
            )["retrieval"]
    role_hash = canonical_json_hash(role)
    views = build_search_views(
        question=question,
        graph=graph,
        table=table,
        role=role,
        retrieval=retrieval,
        matrix=matrix,
        role_schema=role_schema,
        role_registry=role_registry,
    )
    closure, search = _search(
        generator=generator,
        output=output,
        split=split,
        sample_id=sample_id,
        views=views,
        graph=graph,
        matrix=matrix,
    )
    bundle = serialize_plan_closure_v3(
        closure,
        context=ArtifactContext(
            sample_id,
            table_id,
            "C1_C2_EXACT_PROGRAM_UNION",
            str(role["role_id"]),
            role_record_sha256=role_hash,
        ),
        initial_answer=b0_answer,
    )
    vault = [
        {
            "schema_version": "certa_executed_answer_vault_v1",
            "sample_id": sample_id,
            "table_id": table_id,
            "variant_id": "CERTA_V2_BOUNDED_SEARCH",
            "arm": "C1_C2_EXACT_PROGRAM_UNION",
            "canonical_program_id": item.operation_metadata["canonical_program_id"],
            "derivation_id": item.derivation_id,
            "answer_hash": active_answer_hash(item.projected_answer),
            "executed_answer": item.projected_answer,
        }
        for item in closure.executable_derivations
    ]
    graph_refs = set(graph.nodes)
    graph_refs.update(str(edge.edge_type.value) for edge in graph.edges)
    registry = materialize_executed_registry(
        sample_id=sample_id,
        table_id=table_id,
        role_record_sha256=role_hash,
        graph_sha256=graph_hash,
        raw_groundings=bundle.raw_groundings,
        raw_derivations=bundle.raw_derivations,
        registry_entries=bundle.registry_entries,
        answer_vault=vault,
        executed_derivations=closure.executable_derivations,
        graph_artifact_refs=graph_refs,
    )
    candidates = build_candidate_universe(sample_id, b0_answer, registry)
    evidence = [
        {
            "artifact_ref": node_id,
            "text": str(node.text or ""),
            "row": node.row,
            "col": node.col,
            "node_type": str(node.node_type.value),
        }
        for node_id, node in sorted(graph.nodes.items())
    ]
    proof_evidence = (
        project_structural_evidence(evidence)
        if len(canonical_json(evidence)) > MAX_FULL_EVIDENCE_CHARS
        else evidence
    )
    proof_evidence_refs = {
        str(item["artifact_ref"]) for item in proof_evidence
    }
    proofs = []
    proof_failures = []
    for candidate in candidates:
        witness = next(
            (
                item
                for item in registry
                if item["registry_entry_id"]
                == min(candidate["registry_refs"], default="")
            ),
            None,
        )
        signature_id = str(
            witness["signature_id"] if witness is not None else role["role_id"]
        )
        applicability = challenge_applicability(signature_id)
        packet = build_structural_challenge_prompt(
            candidate,
            sample_id=sample_id,
            question=question,
            role_record_sha256=role_hash,
            graph_sha256=graph_hash,
            evidence=proof_evidence,
            registry_entries=registry,
            signature_id=signature_id,
        )
        call = _call(
            generator,
            output=output,
            split=split,
            sample_id=sample_id,
            call_id=f"PROOF_V5_{candidate['candidate_id']}",
            prompt=_proof_prompt(packet),
            max_tokens=1536,
            seed=1729,
            schema=_transport_schema(_read(PROOF_SCHEMA)),
        )
        response_error = ""
        try:
            response = canonicalize_structural_response(
                json.loads(call["text"]),
                expected_applicability=applicability,
            )
        except (ValueError, json.JSONDecodeError) as exc:
            response, response_error = {}, f"challenge_json_invalid:{type(exc).__name__}"
        proof_kwargs = {
            "role_record_sha256": role_hash,
            "graph_sha256": graph_hash,
            "witness": witness,
            "packet_sha256": packet["packet_sha256"],
            "allowed_artifact_refs": proof_evidence_refs,
            "expected_role_id": str(role["role_id"]),
            "expected_challenge_applicability": applicability,
        }
        try:
            if response_error:
                raise ValueError(response_error)
            proof = build_proof_record(
                candidate, structural_response=response, **proof_kwargs
            )
        except ValueError as exc:
            proof_failures.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "reason": str(exc),
                    "fail_closed_state": "UNKNOWN",
                }
            )
            response = {
                "sample_id": sample_id,
                "candidate_id": candidate["candidate_id"],
                "candidate_answer_hash": candidate["candidate_answer_hash"],
                "packet_sha256": packet["packet_sha256"],
                "responses": [
                    {
                        "challenge_id": challenge_id,
                        "applicable": applicability[challenge_id],
                        "support": (
                            "UNKNOWN"
                            if applicability[challenge_id]
                            else "NOT_SUPPORTED"
                        ),
                        "contradiction": (
                            "UNKNOWN"
                            if applicability[challenge_id]
                            else "NOT_FOUND"
                        ),
                        "artifact_refs": [],
                        "claim_codes": [
                            (
                                "INSUFFICIENT_EVIDENCE"
                                if applicability[challenge_id]
                                else "NOT_APPLICABLE_BY_SIGNATURE"
                            )
                        ],
                    }
                    for challenge_id in CHALLENGE_IDS
                ],
            }
            proof = build_proof_record(
                candidate, structural_response=response, **proof_kwargs
            )
        validate_proof_record(
            proof,
            candidate=candidate,
            registry_entries=registry,
            role_record_sha256=role_hash,
            graph_sha256=graph_hash,
            allowed_artifact_refs=graph_refs,
            expected_role_id=str(role["role_id"]),
            expected_challenge_applicability=applicability,
        )
        proofs.append(proof)
    b0_proof = next(item for item in proofs if item["candidate_source"] == "B0")
    alt_proofs = [
        item for item in proofs if item["candidate_source"] == "EXECUTED_REGISTRY"
    ]
    roster_hash = canonical_json_hash([item["candidate_id"] for item in candidates])
    decision_a = _a_decision(candidates)
    decision_b = decide_proof_dominance(
        b0_proof=b0_proof,
        alternative_proofs=alt_proofs,
        candidates=candidates,
        registry_entries=registry,
        roster_sha256=roster_hash,
    )
    verifier = None
    if decision_b["action"] == "REPAIR":
        alt = next(
            item
            for item in alt_proofs
            if item["candidate_id"] == decision_b["selected_candidate_id"]
        )
        verifier_packet = {
            "b0_proof": b0_proof,
            "alternative_proof": alt,
            "eligible_differing_node_ids": decision_b["differing_node_ids"],
        }
        call = _call(
            generator,
            output=output,
            split=split,
            sample_id=sample_id,
            call_id="PAIRWISE_VERIFIER_V3",
            prompt=(
                "CERTA Pairwise Proof Verifier\nReturn JSON only. Confirm or veto "
                "one already eligible cited node difference. Do not create an answer.\n\n"
                + canonical_json(verifier_packet)
            ),
            max_tokens=VERIFIER_MAX_TOKENS,
            seed=1729,
            schema=_transport_schema(_read(VERIFIER_SCHEMA)),
        )
        verifier = _parse_json_mapping(call["text"])
    decision_c = decide_proof_verifier(
        b0_proof=b0_proof,
        alternative_proofs=alt_proofs,
        candidates=candidates,
        registry_entries=registry,
        roster_sha256=roster_hash,
        verifier_response=verifier or {},
    )
    decisions = {
        "V2-A_EXECUTABLE_SEARCH_ONLY": decision_a,
        "V2-B_PROOF_DOMINANCE": decision_b,
        "V2-C_PROOF_VERIFIER": decision_c,
    }
    selected = {
        variant: _materialize(decision, candidates, registry, b0_answer)
        for variant, decision in decisions.items()
    }
    return {
        "schema_version": "certa_v2_sample_master_v1",
        "sample_id": sample_id,
        "table_id": table_id,
        "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
        "graph_sha256": graph_hash,
        "role": role,
        "role_record_sha256": role_hash,
        "b0_answer": b0_answer,
        "search": search,
        "registry": registry,
        "candidates": candidates,
        "proofs": proofs,
        "proof_failures": proof_failures,
        "verifier": verifier,
        "decisions": decisions,
        "selected_finals": selected,
    }


def run_split(output: Path, split: str, limit: int | None) -> dict[str, Any]:
    if split not in {"development", "validation"}:
        raise ValueError(f"unsupported_split:{split}")
    if split == "validation":
        freeze = _read(output / "freeze" / "V2_METHOD_FREEZE.json")
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip()
        if freeze.get("method_commit") != head:
            raise RuntimeError("validation_method_commit_not_frozen")
        if subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO, text=True
        ).strip():
            raise RuntimeError("validation_requires_clean_worktree")
    runtime_path = (
        V1_OUT / "data" / "hitab" / "validation_runtime_v3.jsonl"
        if split == "development"
        else output / "data" / "fresh_validation_runtime.jsonl"
    )
    runtime_rows = _readl(runtime_path)
    if limit is not None:
        runtime_rows = runtime_rows[:limit]
    matrix = _read(MATRIX_PATH)
    role_schema = _read(ROLE_ROOT / "ROLE_V3_OUTPUT_SCHEMA.json")
    role_registry = _read(ROLE_ROOT / "ROLE_V3_CANONICAL_REGISTRY.json")
    role_cards = _read(ROLE_ROOT / "ROLE_V3_ROLE_CARDS.json")
    adapter = HiTabAdapterV1(REPO / "dataset" / "hitab" / "tables" / "raw")
    generator = OpenAIChatGenerator(
        model=MODEL,
        api_base_url=API_BASE,
        api_key_env="EMPTY",
        timeout=120,
        max_retries=0,
        max_model_len=32768,
        cache_mode="off",
        backend_name="vllm_chat",
    )
    encoder = FrozenE5Encoder(device="cpu") if split == "validation" else None
    rows = []
    for runtime in runtime_rows:
        sample_path = (
            output / split / "samples" / f"{runtime['id']}.json"
        )
        if sample_path.is_file():
            row = _read(sample_path)
        else:
            row = _run_sample(
                generator=generator,
                output=output,
                split=split,
                runtime=runtime,
                adapter=adapter,
                matrix=matrix,
                role_schema=role_schema,
                role_registry=role_registry,
                role_cards=role_cards,
                encoder=encoder,
            )
            _write(sample_path, row)
        rows.append(row)
        _writel(output / split / "SAMPLE_MASTER.jsonl", rows)
    proofs = [proof for row in rows for proof in row["proofs"]]
    verifiers = [
        {"sample_id": row["sample_id"], **row["verifier"]}
        for row in rows
        if row["verifier"] is not None
    ]
    _writel(output / split / "PROOF_RECORDS.jsonl", proofs)
    _writel(output / split / "VERIFIER_RECORDS.jsonl", verifiers)
    if split == "development":
        _writel(output / "development" / "SAMPLE_MASTER.jsonl", rows)
        _writel(output / "proof" / "PROOF_RECORDS.jsonl", proofs)
        _writel(output / "proof" / "VERIFIER_RECORDS.jsonl", verifiers)
        _write(
            output / "proof" / "PROOF_SCHEMA.json",
            _read(REPO / "schemas" / "v2" / "PROOF_RECORD.schema.json"),
        )
    return {
        "split": split,
        "rows": len(rows),
        "proof_records": len(proofs),
        "verifier_records": len(verifiers),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--split", choices=("development", "validation"), required=True
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    print(json.dumps(run_split(args.output, args.split, args.limit), sort_keys=True))


if __name__ == "__main__":
    main()
