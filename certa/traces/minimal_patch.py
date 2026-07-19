"""Deterministic First Verifiable Failure patch replay for CERTA Round 12."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from graph_builder import EdgeType, NodeType

from certa.grounding import build_plan_closure
from certa.grounding.structural_resolvers import ResolutionState, resolve_atomic_operand
from certa.hcer.experiment import typed_answer_equivalence
from certa.operations.contracts import OPERATION_SIGNATURES, get_operation_signature
from certa.reproducibility.canonical_json import canonical_json_hash

from .typed_trace import (
    FirstVerifiableFailure,
    IntentHypothesis,
    RoleBindingStep,
    TypedExecutableReasoningTrace,
    VerificationStage,
    VerificationStatus,
    build_typed_executable_traces,
    first_verifiable_failure,
)
from .structured_proposal import replay_compiled_actions


PATCH_VERSION = "round12_minimal_structural_patch_v1"
_BINDING_EDGE_TYPES = {
    EdgeType.ROW_PATH,
    EdgeType.COL_PATH,
    EdgeType.VALUE_UNDER_HEADER,
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(child) for child in value]
    return value


def _flatten(value: Any) -> Tuple[str, ...]:
    if isinstance(value, (tuple, list)):
        return tuple(reference for item in value for reference in _flatten(item))
    return (str(value),) if str(value) else ()


def _canonical_binding(value: Any) -> Any:
    if isinstance(value, (tuple, list)):
        if value and all(isinstance(item, (tuple, list)) for item in value):
            return tuple(sorted({_canonical_binding(item) for item in value}))
        return tuple(sorted({str(item) for item in value if str(item)}))
    return str(value)


def _prefix_payload(
    trace: TypedExecutableReasoningTrace,
    boundary: FirstVerifiableFailure,
) -> Dict[str, Any]:
    prefix_stages = {
        VerificationStage.INTENT_CONTRACT,
        VerificationStage.ROLE_SHAPE,
        VerificationStage.ROLE_REFERENCE,
    }
    steps = [
        {
            "step_id": step.step_id,
            "stage": step.stage.value,
            "status": step.status.value,
            "responsible_fields": list(step.responsible_fields),
        }
        for step in trace.verification_steps
        if step.stage in prefix_stages
    ]
    return {
        "trace_version": trace.to_dict()["trace_version"],
        "intent": trace.intent.to_dict(),
        "role_domain_declarations": [item.to_dict() for item in trace.role_steps],
        "verified_prefix_steps": steps,
        "suffix_boundary": {
            "step_id": boundary.step_id,
            "stage": boundary.stage.value,
        },
    }


def _prefix_hash(
    trace: TypedExecutableReasoningTrace,
    boundary: FirstVerifiableFailure,
) -> str:
    return canonical_json_hash(_prefix_payload(trace, boundary))


def _declared_candidates(
    trace: TypedExecutableReasoningTrace,
    failure: FirstVerifiableFailure,
) -> list[Tuple[Dict[str, Any], str]]:
    selected = {role: _canonical_binding(value) for role, value in trace.selected_role_bindings}
    candidates = []
    for step in trace.role_steps:
        if step.role_name not in failure.responsible_fields:
            continue
        for option in step.binding_options:
            normalized = _canonical_binding(option)
            if normalized == selected.get(step.role_name):
                continue
            patched = dict(selected)
            patched[step.role_name] = normalized
            candidates.append((patched, "planner_declared_role_alternative"))
    return candidates


def _schema_neighbors(graph: Any, cell_id: str) -> set[str]:
    values = set()
    for target_id, edge in graph.neighbors(cell_id):
        node = graph.nodes.get(target_id)
        if (
            node is not None
            and node.node_type in {NodeType.HEADER, NodeType.AGGREGATOR}
            and edge.edge_type in _BINDING_EDGE_TYPES
        ):
            values.add(str(target_id))
    return values


def _minimal_hitting_sets(
    families: Sequence[Sequence[str]],
) -> list[frozenset[str]]:
    """Enumerate the exact set-inclusion-minimal transversals of a hypergraph."""
    frontier = {frozenset()}
    for family in families:
        expanded = {
            base | {item}
            for base in frontier
            for item in sorted(set(family))
        }
        minimal = []
        for candidate in sorted(expanded, key=lambda item: (len(item), tuple(sorted(item)))):
            if not any(existing <= candidate for existing in minimal):
                minimal.append(candidate)
        frontier = set(minimal)
    return sorted(frontier, key=lambda item: (len(item), tuple(sorted(item))))


def _minimal_discriminating_extensions(
    graph: Any, conjunction: Sequence[str]
) -> list[Tuple[Tuple[str, ...], str]]:
    resolution = resolve_atomic_operand(graph, conjunction)
    if resolution.state != ResolutionState.AMBIGUOUS:
        return []
    candidates = tuple(resolution.candidate_node_ids)
    neighbors = {cell_id: _schema_neighbors(graph, cell_id) for cell_id in candidates}
    raw: set[frozenset[str]] = set()
    target_by_extension: Dict[frozenset[str], set[str]] = {}
    for target in candidates:
        difference_families = [
            sorted(neighbors[target] - neighbors[other])
            for other in candidates
            if other != target
        ]
        if not difference_families or any(not family for family in difference_families):
            continue
        for hitting_set in _minimal_hitting_sets(difference_families):
            extension = hitting_set - set(conjunction)
            if not extension:
                continue
            resolved = resolve_atomic_operand(graph, (*conjunction, *sorted(extension)))
            if resolved.state == ResolutionState.UNIQUE and resolved.unique_node_id == target:
                raw.add(extension)
                target_by_extension.setdefault(extension, set()).add(target)
    minimal = [
        extension for extension in raw
        if not any(other < extension for other in raw)
    ]
    return [
        (tuple(sorted(extension)), sorted(target_by_extension[extension])[0])
        for extension in sorted(minimal, key=lambda item: (len(item), tuple(sorted(item))))
    ]


def _ambiguous_contexts(
    trace: TypedExecutableReasoningTrace,
    failure: FirstVerifiableFailure,
) -> list[Tuple[Tuple[str, ...], Tuple[str, ...], int]]:
    """Return conjunction, eligible simple roles, and scope member index."""
    selected = {role: _canonical_binding(value) for role, value in trace.selected_role_bindings}
    signature = get_operation_signature(trace.intent.signature_id)
    operation = signature.operation_family if signature is not None else ""
    if operation in {"DIFF", "RATIO", "PAIR_COMPARE"}:
        return [
            (_flatten(selected.get(role, ())), (role,), -1)
            for role in failure.responsible_fields
            if selected.get(role)
        ]
    if operation == "LOOKUP":
        roles = tuple(role for role in failure.responsible_fields if selected.get(role))
        conjunction = tuple(
            reference for role in roles for reference in _flatten(selected.get(role, ()))
        )
        return [(conjunction, roles, -1)] if conjunction else []
    if operation in {"SUM", "AVERAGE", "COUNT", "ARGMAX", "ARGMIN"}:
        reasons = failure.failure_reasons
        member_text = next((
            reason.split(":", 1)[1]
            for reason in reasons
            if reason.startswith("ambiguous_scope_member:")
        ), "")
        member = tuple(sorted(item for item in member_text.split(",") if item))
        scope = tuple(selected.get("AGGREGATION_SCOPE", ()))
        member_index = next((
            index for index, item in enumerate(scope) if _canonical_binding(item) == member
        ), -1)
        shared = tuple(
            reference
            for role in ("TARGET_MEASURE", "GROUP_SCOPE", "TIME_SCOPE")
            for reference in _flatten(selected.get(role, ()))
        )
        if member_index >= 0:
            return [((*member, *shared), ("AGGREGATION_SCOPE",), member_index)]
    return []


def _eligible_refinement_roles(
    graph: Any,
    target_cell: str,
    extension: Sequence[str],
    roles: Sequence[str],
) -> Tuple[str, ...]:
    if "TARGET_ENTITY" not in roles:
        return tuple(roles)
    edge_types = {
        edge.edge_type
        for target_id, edge in graph.neighbors(target_cell)
        if target_id in set(extension)
    }
    if edge_types and edge_types <= {EdgeType.ROW_PATH}:
        return ("TARGET_ENTITY",)
    non_entity = tuple(role for role in roles if role != "TARGET_ENTITY")
    return non_entity or tuple(roles)


def _refinement_candidates(
    trace: TypedExecutableReasoningTrace,
    failure: FirstVerifiableFailure,
    graph: Any,
) -> list[Tuple[Dict[str, Any], str]]:
    if failure.status != VerificationStatus.AMBIGUOUS:
        return []
    selected = {role: _canonical_binding(value) for role, value in trace.selected_role_bindings}
    candidates = []
    for conjunction, roles, member_index in _ambiguous_contexts(trace, failure):
        for extension, target_cell in _minimal_discriminating_extensions(graph, conjunction):
            if member_index >= 0:
                patched = dict(selected)
                scope = list(selected.get("AGGREGATION_SCOPE", ()))
                scope[member_index] = tuple(sorted({*scope[member_index], *extension}))
                patched["AGGREGATION_SCOPE"] = tuple(scope)
                candidates.append((patched, "exact_minimal_structural_refinement"))
                continue
            for role in _eligible_refinement_roles(graph, target_cell, extension, roles):
                patched = dict(selected)
                patched[role] = tuple(sorted({*_flatten(selected.get(role, ())), *extension}))
                candidates.append((patched, "exact_minimal_structural_refinement"))
    return candidates


def _fixed_plan(intent: IntentHypothesis, bindings: Mapping[str, Any]) -> Dict[str, Any]:
    signature = get_operation_signature(intent.signature_id)
    if signature is None:
        raise ValueError(f"unknown_signature:{intent.signature_id}")
    plan: Dict[str, Any] = {
        "plan_id": "P1",
        "signature_id": signature.signature_id,
        "operation_family": signature.operation_family,
        "semantic_result_role": signature.semantic_result_role,
        "answer_domain": signature.answer_domain,
        "projection_operator": signature.projection_operator,
        "role_domains": {
            role: [_jsonable(value)] for role, value in sorted(bindings.items())
        },
        "unresolved_semantics": list(intent.unresolved_semantics),
    }
    if intent.comparison_polarity:
        plan["comparison_polarity"] = intent.comparison_polarity
    return plan


def _replay_candidate(
    source: TypedExecutableReasoningTrace,
    bindings: Mapping[str, Any],
    graph: Any,
) -> TypedExecutableReasoningTrace:
    payload = {
        "planner_version": source.to_dict()["trace_version"],
        "plans": [_fixed_plan(source.intent, bindings)],
        "unresolved_semantics": [],
    }
    closure = build_plan_closure(payload, graph, max_assignments=1)
    traces = build_typed_executable_traces(
        (source.intent,), source.role_steps, closure, graph
    )
    if len(traces) != 1:
        raise ValueError(f"patch_suffix_replay_trace_count:{len(traces)}")
    return traces[0]


def _patch_record(
    source: TypedExecutableReasoningTrace,
    failure: FirstVerifiableFailure,
    bindings: Mapping[str, Any],
    patch_source: str,
    graph: Any,
) -> Dict[str, Any]:
    patched = _replay_candidate(source, bindings, graph)
    old_bindings = {role: _canonical_binding(value) for role, value in source.selected_role_bindings}
    old_refs = {
        reference for value in old_bindings.values() for reference in _flatten(value)
    }
    new_refs = {
        reference for value in bindings.values() for reference in _flatten(value)
    }
    changed_refs = sorted(old_refs.symmetric_difference(new_refs))
    prefix_hash = _prefix_hash(source, failure)
    patched_prefix_hash = _prefix_hash(patched, failure)
    grounding = next(
        (
            step for step in patched.verification_steps
            if step.stage == VerificationStage.STRUCTURAL_GROUNDING
            and step.step_id == failure.step_id
        ),
        None,
    )
    executable = bool(
        patched.executable
        and grounding is not None
        and grounding.status == VerificationStatus.PASS
    )
    distance = [1, len(changed_refs), 0]
    identity = {
        "patch_version": PATCH_VERSION,
        "source_trace_id": source.trace_id,
        "first_failure_step_id": failure.step_id,
        "bindings": _jsonable(bindings),
        "patch_source": patch_source,
    }
    return {
        "patch_version": PATCH_VERSION,
        "patch_id": f"RP-{canonical_json_hash(identity, 24)}",
        "source_trace_id": source.trace_id,
        "first_failure_step_id": failure.step_id,
        "first_failure_stage": failure.stage.value,
        "first_failure_status": failure.status.value,
        "changed_step_ids": [failure.step_id],
        "old_step": failure.to_dict(),
        "new_step": grounding.to_dict() if grounding is not None else {},
        "old_selected_role_bindings": _jsonable(old_bindings),
        "new_selected_role_bindings": _jsonable(bindings),
        "changed_reference_ids": changed_refs,
        "signature_id": source.intent.signature_id,
        "changed_signature_field_count": 0,
        "patch_distance": distance,
        "prefix_hash": prefix_hash,
        "patched_prefix_hash": patched_prefix_hash,
        "prefix_preserved": prefix_hash == patched_prefix_hash,
        "suffix_start": failure.step_id,
        "suffix_replay_length": sum(
            step.stage in {
                VerificationStage.STRUCTURAL_GROUNDING,
                VerificationStage.EXECUTION,
                VerificationStage.PROJECTION,
                VerificationStage.OUTPUT_TYPE,
            }
            for step in patched.verification_steps
        ),
        "patch_source": patch_source,
        "resource_complete": patched.resource_complete,
        "patched_executable": executable,
        "patched_answer": patched.projected_answer if executable else "",
        "patched_output_domain": patched.output_domain if executable else "",
        "patched_trace": patched.to_dict(),
        "planner_call_count": 0,
        "cera_call_count": 0,
        "final_answer_mutated": False,
        "minimal_executable": False,
    }


def build_minimal_structural_patch_registry(
    intents: Sequence[IntentHypothesis],
    role_steps: Sequence[RoleBindingStep],
    closure: Any,
    graph: Any,
) -> Dict[str, Any]:
    """Replay every finite one-FVF patch and retain all distance minima."""
    traces = build_typed_executable_traces(intents, role_steps, closure, graph)
    candidate_records = []
    eligible_sources = []
    for source in traces:
        failure = first_verifiable_failure(source)
        if (
            failure is None
            or failure.stage != VerificationStage.STRUCTURAL_GROUNDING
            or failure.status not in {VerificationStatus.UNRESOLVED, VerificationStatus.AMBIGUOUS}
            or not failure.resource_complete
        ):
            continue
        raw_candidates = [
            *_declared_candidates(source, failure),
            *_refinement_candidates(source, failure, graph),
        ]
        deduplicated: Dict[str, Tuple[Dict[str, Any], str]] = {}
        for bindings, patch_source in raw_candidates:
            key = canonical_json_hash({
                "bindings": _jsonable(bindings),
                "patch_source": patch_source,
            })
            deduplicated[key] = (bindings, patch_source)
        if deduplicated:
            eligible_sources.append(source.trace_id)
        for key in sorted(deduplicated):
            bindings, patch_source = deduplicated[key]
            candidate_records.append(
                _patch_record(source, failure, bindings, patch_source, graph)
            )
    minimal = []
    for source_trace_id in sorted(set(eligible_sources)):
        executable = [
            item for item in candidate_records
            if item["source_trace_id"] == source_trace_id
            and item["patched_executable"]
            and item["prefix_preserved"]
            and item["resource_complete"]
        ]
        if not executable:
            continue
        minimum = min(tuple(item["patch_distance"]) for item in executable)
        for item in executable:
            if tuple(item["patch_distance"]) == minimum:
                item["minimal_executable"] = True
                minimal.append(item)
    candidate_records.sort(key=lambda item: item["patch_id"])
    minimal.sort(key=lambda item: item["patch_id"])
    return {
        "patch_version": PATCH_VERSION,
        "source_trace_count": len(traces),
        "eligible_source_trace_count": len(set(eligible_sources)),
        "eligible_source_trace_ids": sorted(set(eligible_sources)),
        "local_patch_domain_count": len(candidate_records),
        "candidate_records": candidate_records,
        "minimal_executable_patch_count": len(minimal),
        "minimal_patch_records": minimal,
        "candidate_model_calls": 0,
        "cera_calls": 0,
        "final_answer_mutations": 0,
    }


@dataclass(frozen=True)
class LocalRepairResult:
    base_answer: str
    local_repair_answer: str
    would_commit: bool
    commit_reason: str
    first_fault_step_id: str = ""
    candidate_count: int = 0
    minimal_executable_patch_count: int = 0
    minimal_patch_records: Tuple[Mapping[str, Any], ...] = ()
    semantic_answer_classes: Tuple[str, ...] = ()
    model_call_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return _jsonable(asdict(self))


def _public_prefix_hash(compiled: Any, fault_index: int) -> str:
    return canonical_json_hash({
        "structured_proposal_version": "certa_minimal_semantic_actions_v1",
        "verified_prefix": [step.to_dict() for step in compiled.steps[:fault_index]],
        "fault_step_id": compiled.steps[fault_index].step_id,
    })


def _same_structural_domain(source: Any, candidate: Any) -> bool:
    return bool(
        source.typed_domain == candidate.typed_domain
        and ((source.row_header_path and source.row_header_path == candidate.row_header_path)
             or (source.column_header_path
                 and source.column_header_path == candidate.column_header_path))
    )


def _action_candidates(compiled: Any, fault_index: int) -> Tuple[Tuple[Dict[str, Any], str], ...]:
    source = compiled.actions[fault_index]
    candidates = []
    for arg_index, operand in enumerate(source.get("args") or ()):
        if operand.get("kind") != "TABLE_REF":
            continue
        entry = compiled.catalog.resolve(str(operand["ref"]))
        for alternative in compiled.catalog.entries:
            if (alternative.handle == entry.handle
                    or not _same_structural_domain(entry, alternative)):
                continue
            action = _jsonable(source)
            action["args"][arg_index] = {"kind": "TABLE_REF", "ref": alternative.handle}
            candidates.append((action, "exact_structural_catalog_alternative"))
    signature = get_operation_signature(source.get("op"))
    if signature is not None:
        for alternative in sorted(
            OPERATION_SIGNATURES.values(), key=lambda item: item.signature_id
        ):
            if (alternative.signature_id == signature.signature_id
                    or alternative.operation_family != signature.operation_family
                    or bool(alternative.comparison_polarities)
                    != bool(signature.comparison_polarities)):
                continue
            action = _jsonable(source)
            action["op"] = alternative.signature_id
            candidates.append((action, "compatible_registry_signature_variant"))
    deduplicated = {
        canonical_json_hash(action): (action, source)
        for action, source in candidates
    }
    return tuple(deduplicated[key] for key in sorted(deduplicated))


def _changed_operand_count(source: Mapping[str, Any], candidate: Mapping[str, Any]) -> int:
    return int(source.get("op") != candidate.get("op")) + sum(
        left != right
        for left, right in zip(source.get("args") or (), candidate.get("args") or ())
    )


def _action_patch_record(
    compiled: Any,
    *,
    fault_index: int,
    replacement: Mapping[str, Any],
    patch_source: str,
    graph: Any,
    dataset: str,
) -> Dict[str, Any]:
    replayed, replay_ids, unchanged_ids = replay_compiled_actions(
        compiled, graph, dataset=dataset, replacements={fault_index: replacement}
    )
    replay_indices = tuple(int(step_id[1:]) for step_id in replay_ids)
    replay_valid = all(
        replayed.steps[index].step_result.execution_status == "PROJECTED"
        for index in replay_indices
    )
    unchanged_valid = all(
        replayed.steps[int(step_id[1:])] is compiled.steps[int(step_id[1:])]
        and replayed.steps[int(step_id[1:])].to_dict()
        == compiled.steps[int(step_id[1:])].to_dict()
        for step_id in unchanged_ids
    )
    executable = all(
        step.step_result.execution_status == "PROJECTED" for step in replayed.steps
    )
    final_answer = replayed.steps[-1].executed_value if executable else ""
    source_action = compiled.actions[fault_index]
    changed_refs = [
        str(candidate.get("ref"))
        for source_operand, candidate in zip(
            source_action.get("args") or (), replacement.get("args") or ()
        )
        if source_operand != candidate and candidate.get("kind") == "TABLE_REF"
    ]
    prefix_hash = _public_prefix_hash(compiled, fault_index)
    return {
        "patch_id": canonical_json_hash({
            "fault_step_id": f"S{fault_index}",
            "replacement": _jsonable(replacement),
            "patch_source": patch_source,
        }),
        "patch_source": patch_source,
        "replacement_action": _jsonable(replacement),
        "changed_reference_ids": changed_refs,
        "changed_step_ids": [f"S{fault_index}"],
        "edited_step_count": 1,
        "public_prefix_hash": prefix_hash,
        "patched_public_prefix_hash": prefix_hash,
        "public_prefix_preserved": True,
        "replayed_step_ids": list(replay_ids),
        "unchanged_step_ids": list(unchanged_ids),
        "suffix_replay_length": len(replay_ids),
        "patch_distance": [
            1, _changed_operand_count(source_action, replacement), len(replay_ids)
        ],
        "final_typed_projection": final_answer,
        "dependency_replay_valid": replay_valid and unchanged_valid,
        "true_step_result_replay": replay_valid and unchanged_valid,
        "resource_complete": all(step.resource_complete for step in replayed.steps),
        "patched_executable": executable,
        "minimal_executable": False,
        "patched_trace": replayed.to_dict(),
    }


def _semantic_answer_representatives(
    answers: Sequence[str], dataset: str
) -> Tuple[str, ...]:
    representatives = []
    for answer in sorted(set(str(value) for value in answers)):
        if any(
            typed_answer_equivalence(dataset, representative, answer)["equivalent"]
            for representative in representatives
        ):
            continue
        representatives.append(answer)
    return tuple(representatives)


def _empty_local_result(compiled: Any, reason: str) -> LocalRepairResult:
    return LocalRepairResult(
        compiled.answer, compiled.answer, False, reason,
        first_fault_step_id=(
            compiled.first_verifiable_fault.step_id
            if compiled.first_verifiable_fault is not None
            else ""
        ),
    )


def build_first_fault_local_repair(
    compiled: Any, graph: Any, *, dataset: str
) -> LocalRepairResult:
    public_fault = compiled.first_verifiable_fault
    if public_fault is None:
        return _empty_local_result(compiled, "no_first_verifiable_fault")
    if not public_fault.resource_complete:
        return _empty_local_result(compiled, "fault_resource_incomplete")
    fault_index = next(
        index
        for index, step in enumerate(compiled.steps)
        if step.step_id == public_fault.step_id
    )
    records = [
        _action_patch_record(
            compiled, fault_index=fault_index, replacement=replacement,
            patch_source=patch_source, graph=graph, dataset=dataset,
        )
        for replacement, patch_source in _action_candidates(compiled, fault_index)
    ]
    executable = [
        record
        for record in records
        if record["patched_executable"]
        and record["resource_complete"]
        and record["public_prefix_preserved"]
        and record["dependency_replay_valid"]
        and record["edited_step_count"] == 1
        and record["changed_step_ids"] == [public_fault.step_id]
    ]
    if not executable:
        return LocalRepairResult(
            compiled.answer, compiled.answer, False, "no_executable_minimum_patch",
            first_fault_step_id=public_fault.step_id,
            candidate_count=len(records),
        )
    minimum = min(tuple(record["patch_distance"]) for record in executable)
    minimal = tuple(
        record for record in executable
        if tuple(record["patch_distance"]) == minimum
    )
    for record in minimal:
        record["minimal_executable"] = True
    answers = tuple(str(record["final_typed_projection"]) for record in minimal)
    classes = _semantic_answer_representatives(answers, dataset)
    unique = len(classes) == 1
    repair_answer = classes[0] if unique else compiled.answer
    would_commit = bool(unique and not typed_answer_equivalence(
        dataset, compiled.answer, repair_answer
    )["equivalent"])
    reason = (
        "multiple_minimum_semantic_answer_classes" if not unique
        else "unique_minimum_semantic_answer_class" if would_commit
        else "repair_answer_unchanged"
    )
    return LocalRepairResult(
        compiled.answer, repair_answer if would_commit else compiled.answer,
        would_commit, reason,
        first_fault_step_id=public_fault.step_id,
        candidate_count=len(records),
        minimal_executable_patch_count=len(minimal),
        minimal_patch_records=minimal,
        semantic_answer_classes=classes,
    )
