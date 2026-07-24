"""Pure contracts for the bounded CERTA final-method variants.

This module changes neither the frozen Role V3 semantics nor the executor.  It
only defines (1) the complete-domain C2 Planner interface, (2) exact typed-plan
union, and (3) fail-closed support-state decision authority.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from graph_builder import HCEG

from certa.active_v1.answer_authority import active_answer_hash
from certa.active_v1.artifact_authority import (
    reconcile_registry_entry,
    validate_grounding_record_v3,
)
from certa.active_v1.planner_adapter import ActiveCompilationResult, PlannerViewBuild
from certa.active_v1.planner_bridge_v3 import (
    _constructor_active_role_ids,
    build_v3_arm_view,
    compile_active_planner_payload,
)
from certa.active_v1.role_contract_v3 import (
    ROLE_V3_SCHEMA_VERSION,
    derive_role_v3_record,
)
from certa.derivations.answer_equivalence import inference_answers_equivalent
from certa.operations.contracts import OPERATION_SIGNATURES
from certa.reproducibility.canonical_json import canonical_json, canonical_json_hash


VARIANT_IDS = (
    "V0_LEGACY_C2_HARD_FILTER",
    "V1_C2_COMPLETE_DOMAIN",
    "V2_C1_C2_EXACT_PROGRAM_UNION",
)
SUPPORT_STATES = ("NoSupport", "OriginalOnly", "AlternativeOnly", "BothSide")


@dataclass(frozen=True)
class ProgramUnion:
    payload: Dict[str, Any]
    lineage: Tuple[Dict[str, Any], ...]
    payload_sha256: str
    compilation: ActiveCompilationResult


@dataclass(frozen=True)
class ProgramUnionInput:
    arm: str
    sample_id: str
    table_id: str
    graph: Mapping[str, Any]
    role_record: Mapping[str, Any]
    planner_view: Mapping[str, Any]
    compilation: ActiveCompilationResult


@dataclass(frozen=True)
class SupportState:
    state: str
    original_ids: Tuple[str, ...]
    alternative_ids: Tuple[str, ...]


@dataclass(frozen=True)
class RegistryDecision:
    schema_version: str
    sample_id: str
    table_id: str
    variant_id: str
    role_record_sha256: str
    action: str
    selected_arm: str
    selected_program_id: str
    selected_derivation_id: str
    selected_answer_hash: str
    validator_approved: bool
    failure_reasons: Tuple[str, ...]


@dataclass(frozen=True)
class RegistrySupportState:
    sample_id: str
    table_id: str
    variant_id: str
    role_record_sha256: str
    capability_matrix_sha256: str
    state: str
    programs: Tuple[Dict[str, Any], ...]
    valid: bool
    failure_reasons: Tuple[str, ...]


@dataclass(frozen=True)
class MaterializedSelection:
    action: str
    answer: Any
    answer_hash: str
    program_id: str
    failure_reasons: Tuple[str, ...]


def build_complete_domain_c2_view(
    question: str,
    graph: HCEG,
    table_json: Mapping[str, Any],
    role: Mapping[str, Any],
    retrieval: Mapping[str, Any],
    capability_matrix: Mapping[str, Any],
    *,
    output_schema: Optional[Mapping[str, Any]] = None,
    canonical_registry: Optional[Mapping[str, Any]] = None,
) -> PlannerViewBuild:
    """Build C2 with a complete legal schema domain and advisory retrieval."""
    if not isinstance(retrieval, Mapping):
        raise ValueError("c2_retrieval_result_required")
    complete = build_v3_arm_view(
        "C1_ROLE_ONLY",
        question,
        graph,
        table_json,
        role,
        None,
        capability_matrix,
        output_schema=output_schema,
        canonical_registry=canonical_registry,
    )
    if retrieval.get("role_record_sha256") != complete.role_record_sha256:
        raise ValueError("c2_role_record_sha256_mismatch")
    raw_references = retrieval.get("reference_node_ids")
    if not isinstance(raw_references, list) or not raw_references:
        raise ValueError("c2_retrieval_reference_ids_empty")
    references = tuple(dict.fromkeys(str(item) for item in raw_references))
    schema_ids = {
        str(item.get("node_id") or "")
        for item in complete.view.get("schema_nodes", ())
        if isinstance(item, Mapping)
    }
    outside = sorted(set(references) - schema_ids)
    if outside:
        raise ValueError(f"retrieval_reference_outside_schema:{','.join(outside)}")
    view = dict(complete.view)
    view["retrieval_advisory"] = {
        "authority": "ADVISORY_ONLY_NO_DOMAIN_FILTER",
        "reference_node_ids": list(references),
        "reference_count": len(references),
        "complete_schema_node_count": len(complete.view.get("schema_nodes", ())),
        "complete_schema_edge_count": len(complete.view.get("schema_edges", ())),
    }
    return PlannerViewBuild(
        "C2_ROLE_RETRIEVAL",
        view,
        complete.role_record_sha256,
        references,
    )


def canonical_typed_plan_identity(plan: Mapping[str, Any]) -> str:
    """Return pre-ground typed-plan identity after removing only ``plan_id``."""
    if not isinstance(plan, Mapping):
        raise ValueError("typed_program_not_object")
    semantic = {str(key): value for key, value in plan.items() if key != "plan_id"}
    return canonical_json(semantic)


def union_exact_typed_programs(
    inputs: Sequence[ProgramUnionInput],
    *,
    full_domain_view: Mapping[str, Any],
    capability_matrix: Mapping[str, Any],
) -> ProgramUnion:
    """Union normalized Planner payloads by exact semantic program identity."""
    if not inputs:
        raise ValueError("program_union_inputs_empty")
    for field in ("sample_id", "table_id"):
        values = {str(getattr(item, field) or "") for item in inputs}
        if len(values) != 1 or "" in values:
            raise ValueError(f"program_union_context_mismatch:{field}")
    graph_hashes = {canonical_json_hash(item.graph) for item in inputs}
    role_hashes = {canonical_json_hash(item.role_record) for item in inputs}
    if len(graph_hashes) != 1:
        raise ValueError("program_union_context_mismatch:graph")
    if len(role_hashes) != 1:
        raise ValueError("program_union_context_mismatch:role_record")
    if len({item.arm for item in inputs}) != len(inputs):
        raise ValueError("program_union_duplicate_arm")
    for item in inputs:
        if item.arm not in {"C1_ROLE_ONLY", "C2_ROLE_RETRIEVAL"}:
            raise ValueError(f"program_union_invalid_arm:{item.arm}")
        compilation = item.compilation
        if not compilation.ok:
            raise ValueError(f"program_union_input_invalid:{item.arm}")
        if (
            canonical_json_hash(compilation.normalized_payload)
            != compilation.canonical_payload_sha256
        ):
            raise ValueError(f"program_union_compilation_hash_mismatch:{item.arm}")
        recomputed = compile_active_planner_payload(
            compilation.normalized_payload,
            item.planner_view,
            capability_matrix,
        )
        if (
            not recomputed.ok
            or recomputed.normalized_payload != compilation.normalized_payload
            or recomputed.canonical_payload_sha256
            != compilation.canonical_payload_sha256
        ):
            raise ValueError(f"program_union_compilation_revalidation_failed:{item.arm}")
    versions = {
        str(item.compilation.normalized_payload.get("planner_version") or "")
        for item in inputs
    }
    semantics = {
        canonical_json(item.compilation.normalized_payload.get("query_semantics"))
        for item in inputs
    }
    if len(versions) != 1 or "" in versions:
        raise ValueError("program_union_planner_version_mismatch")
    if len(semantics) != 1:
        raise ValueError("program_union_query_semantics_mismatch")

    by_identity: Dict[str, Dict[str, Any]] = {}
    source_arms: Dict[str, set[str]] = {}
    source_plan_ids: Dict[str, set[str]] = {}
    top_unresolved = set()
    for item in inputs:
        arm, payload = item.arm, item.compilation.normalized_payload
        plans = payload.get("plans")
        if not isinstance(plans, list):
            raise ValueError(f"program_union_plans_not_list:{arm}")
        unresolved = payload.get("unresolved_semantics")
        if not isinstance(unresolved, list) or any(
            not isinstance(item, str) for item in unresolved
        ):
            raise ValueError(f"program_union_unresolved_semantics_invalid:{arm}")
        top_unresolved.update(unresolved)
        for plan in plans:
            identity = canonical_typed_plan_identity(plan)
            by_identity.setdefault(
                identity,
                {key: value for key, value in plan.items() if key != "plan_id"},
            )
            source_arms.setdefault(identity, set()).add(str(arm))
            source_plan_ids.setdefault(identity, set()).add(str(plan.get("plan_id") or ""))

    plans = []
    lineage = []
    for index, identity in enumerate(sorted(by_identity)):
        program_id = f"TP-{canonical_json_hash(by_identity[identity], 24)}"
        plan = {"plan_id": f"P{index}", **by_identity[identity]}
        plans.append(plan)
        lineage.append({
            "union_plan_id": plan["plan_id"],
            "typed_program_id": program_id,
            "typed_program_sha256": canonical_json_hash(by_identity[identity]),
            "source_arms": sorted(source_arms[identity]),
            "source_plan_ids": sorted(source_plan_ids[identity]),
        })
    if not plans:
        raise ValueError("program_union_has_no_plans")
    first_payload = inputs[0].compilation.normalized_payload
    payload = {
        "planner_version": next(iter(versions)),
        "query_semantics": dict(first_payload["query_semantics"]),
        "plans": plans,
        "unresolved_semantics": sorted(top_unresolved),
    }
    compiled_union = compile_active_planner_payload(
        payload,
        full_domain_view,
        capability_matrix,
    )
    if not compiled_union.ok:
        raise ValueError(
            "program_union_full_domain_validation_failed:"
            + "|".join(compiled_union.errors)
        )
    if compiled_union.normalized_payload != payload:
        raise ValueError("program_union_normalization_drift")
    return ProgramUnion(
        payload,
        tuple(lineage),
        canonical_json_hash(payload),
        compiled_union,
    )


def classify_support_state(
    original_ids: Iterable[str],
    alternative_ids: Iterable[str],
) -> SupportState:
    """Classify the four exhaustive proposal-relative support states."""
    original = tuple(sorted(set(str(item) for item in original_ids if str(item))))
    alternative = tuple(sorted(set(str(item) for item in alternative_ids if str(item))))
    if set(original) & set(alternative):
        raise ValueError("support_partition_overlap")
    if original and alternative:
        state = "BothSide"
    elif original:
        state = "OriginalOnly"
    elif alternative:
        state = "AlternativeOnly"
    else:
        state = "NoSupport"
    return SupportState(state, original, alternative)


def _one(
    records: Sequence[Any],
    predicate: Any,
    failure: str,
    failures: list[str],
) -> Any:
    matches = [item for item in records if predicate(item)]
    if len(matches) != 1:
        failures.append(f"{failure}:{len(matches)}")
        return None
    return matches[0]


def _program_id(derivation: Any) -> str:
    try:
        program = json.loads(str(getattr(derivation, "executable_program", "") or ""))
    except (TypeError, ValueError):
        return ""
    if canonical_json(program) != getattr(derivation, "executable_program", ""):
        return ""
    return f"CP-{canonical_json_hash(program, 24)}"


def build_registry_support_state(
    *,
    sample_id: str,
    table_id: str,
    variant_id: str,
    selected_arms: Sequence[str],
    role_record: Mapping[str, Any],
    role_output_schema: Mapping[str, Any],
    role_registry: Mapping[str, Any],
    capability_matrix: Mapping[str, Any],
    b0_answer: Any,
    executed_derivations: Sequence[Any],
    raw_groundings_v3: Sequence[Mapping[str, Any]],
    raw_derivations: Sequence[Mapping[str, Any]],
    registry_entries: Sequence[Mapping[str, Any]],
    answer_vault_records: Sequence[Mapping[str, Any]],
) -> RegistrySupportState:
    """Recompute registry support through exact source-record joins."""
    failures: list[str] = []
    if variant_id not in VARIANT_IDS:
        failures.append(f"unknown_variant:{variant_id}")
    arms = tuple(sorted(set(str(item) for item in selected_arms if str(item))))
    if not arms:
        failures.append("selected_arms_empty")
    role_id = str(role_record.get("role_id") or "")
    try:
        expected_role = derive_role_v3_record(
            {"schema_version": ROLE_V3_SCHEMA_VERSION, "role_id": role_id},
            role_output_schema,
            role_registry,
        )
        if dict(role_record) != expected_role:
            failures.append("role_record_not_canonical")
    except (KeyError, TypeError, ValueError) as exc:
        failures.append(f"role_record_invalid:{type(exc).__name__}")
    role_sha = canonical_json_hash(role_record)
    capability_sha = canonical_json_hash(capability_matrix)
    try:
        active_role_ids = _constructor_active_role_ids(capability_matrix)
        if role_id not in active_role_ids:
            failures.append(f"role_not_constructor_active:{role_id}")
    except ValueError as exc:
        failures.append(f"capability_matrix_invalid:{exc}")
    signature = OPERATION_SIGNATURES.get(role_id)
    if signature is None:
        failures.append(f"role_signature_unknown:{role_id}")

    scoped_groundings = [
        item for item in raw_groundings_v3
        if item.get("sample_id") == sample_id
        and item.get("table_id") == table_id
        and item.get("arm") in arms
    ]
    for grounding in scoped_groundings:
        try:
            validate_grounding_record_v3(grounding)
        except ValueError as exc:
            failures.append(f"grounding_invalid:{exc}")
        if grounding.get("role_record_sha256") != role_sha:
            failures.append("grounding_role_sha256_mismatch")

    programs: list[Dict[str, Any]] = []
    by_program: Dict[str, Dict[str, Any]] = {}
    for derivation in executed_derivations:
        derivation_id = str(getattr(derivation, "derivation_id", "") or "")
        metadata = getattr(derivation, "operation_metadata", {}) or {}
        raw = _one(
            raw_derivations,
            lambda item: item.get("sample_id") == sample_id
            and item.get("arm") in arms
            and item.get("derivation_id") == derivation_id,
            f"raw_derivation_join:{derivation_id}",
            failures,
        )
        if raw is None:
            continue
        arm = str(raw.get("arm") or "")
        program_id = _program_id(derivation)
        if not program_id or metadata.get("canonical_program_id") != program_id:
            failures.append(f"canonical_program_identity_mismatch:{derivation_id}")
        if getattr(derivation, "availability", "") != "available":
            failures.append(f"derivation_not_available:{derivation_id}")
        if getattr(derivation, "provenance_complete", False) is not True:
            failures.append(f"derivation_provenance_incomplete:{derivation_id}")
        if not getattr(derivation, "operand_node_ids", ()):
            failures.append(f"derivation_operands_empty:{derivation_id}")
        if signature is not None and (
            getattr(derivation, "typed_signature", "") != role_id
            or getattr(derivation, "operation_family", "") != signature.operation_family
            or getattr(derivation, "projection_operator", "") != signature.projection_operator
            or getattr(derivation, "output_domain", "") != signature.answer_domain
        ):
            failures.append(f"derivation_role_contract_mismatch:{derivation_id}")

        registry = _one(
            registry_entries,
            lambda item: item.get("sample_id") == sample_id
            and item.get("arm") == arm
            and item.get("derivation_id") == derivation_id,
            f"registry_join:{derivation_id}",
            failures,
        )
        vault = _one(
            answer_vault_records,
            lambda item: item.get("sample_id") == sample_id
            and item.get("table_id") == table_id
            and item.get("variant_id") == variant_id
            and item.get("arm") == arm
            and item.get("canonical_program_id") == program_id
            and item.get("derivation_id") == derivation_id,
            f"answer_vault_join:{derivation_id}",
            failures,
        )
        grounding = None
        hypothesis = None
        grounding = _one(
            scoped_groundings,
            lambda item: item.get("arm") == arm
            and item.get("plan_id") == raw.get("plan_id"),
            f"grounding_join:{derivation_id}",
            failures,
        )
        if grounding is not None:
            hypothesis = _one(
                list(grounding.get("grounding_hypotheses") or ()),
                lambda item: item.get("binding_id") == raw.get("binding_id")
                and item.get("derivation_id") == derivation_id
                and item.get("canonical_program_id") == program_id,
                f"grounding_hypothesis_join:{derivation_id}",
                failures,
            )
            if hypothesis is not None and (
                hypothesis.get("binding_id") not in grounding.get("authorized_binding_ids", ())
                or hypothesis.get("resolution_state") != "EXACT"
                or hypothesis.get("grounding_valid") is not True
                or "resource_incomplete" in hypothesis.get("failure_reasons", ())
            ):
                failures.append(f"grounding_not_authorized:{derivation_id}")
        if metadata.get("resource_complete") is False:
            failures.append(f"derivation_resource_incomplete:{derivation_id}")
        if raw is not None:
            expected_answer_hash = active_answer_hash(
                getattr(derivation, "projected_answer", None)
            )
            expected_side = (
                "ORIGINAL"
                if inference_answers_equivalent(
                    getattr(derivation, "projected_answer", None), b0_answer
                )
                else "ALTERNATIVE"
            )
            raw_checks = (
                raw.get("signature_id") == role_id,
                raw.get("canonical_program_id") == program_id,
                raw.get("projected_answer_hash") == expected_answer_hash,
                raw.get("execution_status") == "EXECUTED",
                raw.get("projection_status") == "VALID",
                raw.get("side") == expected_side,
                bool(raw.get("provenance_ids")),
            )
            if not all(raw_checks):
                failures.append(f"raw_derivation_mismatch:{derivation_id}")
        if raw is not None and registry is not None:
            try:
                reconcile_registry_entry(registry, raw)
            except ValueError as exc:
                failures.append(f"registry_reconciliation_failed:{derivation_id}:{exc}")
        if raw is not None and vault is not None:
            answer = vault.get("executed_answer")
            if (
                active_answer_hash(answer) != raw.get("projected_answer_hash")
                or vault.get("answer_hash") != raw.get("projected_answer_hash")
                or not inference_answers_equivalent(
                    answer, getattr(derivation, "projected_answer", None)
                )
            ):
                failures.append(f"answer_vault_mismatch:{derivation_id}")
        if raw is None or registry is None or vault is None or grounding is None or hypothesis is None:
            continue
        program_record = {
            "arm": arm,
            "canonical_program_id": program_id,
            "derivation_id": derivation_id,
            "binding_id": raw["binding_id"],
            "registry_entry_id": registry["registry_entry_id"],
            "answer_hash": raw["projected_answer_hash"],
            "side": raw["side"],
        }
        previous = by_program.get(program_id)
        if previous is not None and previous != program_record:
            failures.append(f"canonical_program_conflict:{program_id}")
        else:
            by_program[program_id] = program_record

    programs = sorted(
        by_program.values(),
        key=lambda item: (item["canonical_program_id"], item["derivation_id"]),
    )
    support = classify_support_state(
        (item["canonical_program_id"] for item in programs if item["side"] == "ORIGINAL"),
        (item["canonical_program_id"] for item in programs if item["side"] == "ALTERNATIVE"),
    )
    unique_failures = tuple(dict.fromkeys(failures))
    return RegistrySupportState(
        sample_id=sample_id,
        table_id=table_id,
        variant_id=variant_id,
        role_record_sha256=role_sha,
        capability_matrix_sha256=capability_sha,
        state=support.state,
        programs=tuple(programs),
        valid=not unique_failures,
        failure_reasons=unique_failures,
    )


def select_registry_policy(
    state: RegistrySupportState,
) -> RegistryDecision:
    """Select only an authority ID/hash; never materialize an answer here."""
    failures = list(state.failure_reasons)
    if not state.valid:
        failures.append("registry_support_state_invalid")
    if state.state != "AlternativeOnly":
        failures.append(f"support_state_not_alternative_only:{state.state}")
    alternatives = [item for item in state.programs if item["side"] == "ALTERNATIVE"]
    hashes = {item["answer_hash"] for item in alternatives}
    if not alternatives:
        failures.append("alternative_missing")
    if len(hashes) != 1:
        failures.append("alternative_answer_class_not_unique")
    unique_failures = tuple(dict.fromkeys(failures))
    if unique_failures:
        return RegistryDecision(
            "certa_registry_validated_selection_v1",
            state.sample_id,
            state.table_id,
            state.variant_id,
            state.role_record_sha256,
            "KEEP_B0",
            "",
            "",
            "",
            "",
            False,
            unique_failures,
        )
    program_ids = {item["canonical_program_id"] for item in alternatives}
    if len(program_ids) != 1:
        return RegistryDecision(
            "certa_registry_validated_selection_v1",
            state.sample_id,
            state.table_id,
            state.variant_id,
            state.role_record_sha256,
            "KEEP_B0", "", "", "", "", False,
            ("alternative_program_not_unique",),
        )
    selected = alternatives[0]
    return RegistryDecision(
        "certa_registry_validated_selection_v1",
        state.sample_id,
        state.table_id,
        state.variant_id,
        state.role_record_sha256,
        "USE_ALTERNATIVE",
        selected["arm"],
        selected["canonical_program_id"],
        selected["derivation_id"],
        selected["answer_hash"],
        True,
        (),
    )


def materialize_registry_selection(
    selection: RegistryDecision,
    state: RegistrySupportState,
    answer_vault_records: Sequence[Mapping[str, Any]],
    b0_answer: Any,
) -> MaterializedSelection:
    """Recheck policy and resolve one exact vault record or return B0."""
    recomputed = select_registry_policy(state)
    if selection != recomputed or selection.action != "USE_ALTERNATIVE":
        failures = (
            ("selection_recomputation_mismatch",)
            if selection != recomputed
            else selection.failure_reasons
        )
        return MaterializedSelection(
            "KEEP_B0", b0_answer, active_answer_hash(b0_answer), "", failures,
        )
    matches = [
        item for item in answer_vault_records
        if item.get("sample_id") == state.sample_id
        and item.get("table_id") == state.table_id
        and item.get("variant_id") == state.variant_id
        and item.get("arm") == selection.selected_arm
        and item.get("canonical_program_id") == selection.selected_program_id
        and item.get("derivation_id") == selection.selected_derivation_id
        and item.get("answer_hash") == selection.selected_answer_hash
    ]
    if len(matches) != 1:
        return MaterializedSelection(
            "KEEP_B0", b0_answer, active_answer_hash(b0_answer), "",
            (f"materializer_vault_join:{len(matches)}",),
        )
    answer = matches[0].get("executed_answer")
    if active_answer_hash(answer) != selection.selected_answer_hash:
        return MaterializedSelection(
            "KEEP_B0", b0_answer, active_answer_hash(b0_answer), "",
            ("materializer_answer_hash_mismatch",),
        )
    return MaterializedSelection(
        "USE_ALTERNATIVE",
        answer,
        selection.selected_answer_hash,
        selection.selected_program_id,
        (),
    )
