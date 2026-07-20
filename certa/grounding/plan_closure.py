"""Pure operation-typed plan-conditioned grounding closure for CERTA."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from itertools import product
from math import prod
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from graph_builder import HCEG, NodeType

from certa.derivations.project import ProjectionStatus, execute_typed_projection_from_nodes
from certa.derivations.schema import ExecutableDerivation, to_jsonable
from certa.grounding.structural_resolvers import (
    EntityMeasureRelationResolution,
    ResolutionState,
    resolve_atomic_operand,
    resolve_entity_measure_relation,
    resolve_finite_scope,
)
from certa.operations.contracts import (
    FINAL_SUPPORTED_OPERATIONS,
    OPERATION_CONTRACT_VERSION,
    OperationContract,
    get_operation_contract,
    resolve_operation_signature,
    validate_operation_plan,
)
from certa.reproducibility.canonical_json import canonical_json, canonical_json_hash


PLAN_CLOSURE_VERSION = "plan_conditioned_operation_closure_v1"


class ClosureOutcome(str, Enum):
    UNIQUE_EXECUTABLE = "UNIQUE_EXECUTABLE"
    AMBIGUOUS_BINDING = "AMBIGUOUS_BINDING"
    UNRESOLVED_BINDING = "UNRESOLVED_BINDING"
    STRUCTURALLY_INVALID = "STRUCTURALLY_INVALID"
    EXECUTION_FAILED = "EXECUTION_FAILED"
    RESOURCE_INCOMPLETE = "RESOURCE_INCOMPLETE"


@dataclass(frozen=True)
class GroundedAssignment:
    plan_id: str
    assignment_id: str
    assignment_key: str
    role_bindings: Dict[str, Any]
    outcome: ClosureOutcome
    plan_ids: Tuple[str, ...] = ()
    target_entity_ids: Tuple[str, ...] = ()
    target_measure_ids: Tuple[str, ...] = ()
    time_scope_ids: Tuple[str, ...] = ()
    resolution_state: str = ""
    matched_cell_ids: Tuple[str, ...] = ()
    required_edge_triples: Tuple[Tuple[str, str, str], ...] = ()
    derivation_id: str = ""
    failure_reasons: Tuple[str, ...] = ()
    operation_family: str = ""
    signature_id: str = ""
    semantic_result_role: str = ""
    comparison_polarity: str = ""
    projection_operator: str = ""
    answer_domain: str = ""
    resolved_atomic_operands: Tuple[Tuple[str, str], ...] = ()
    resolved_scope_node_ids: Tuple[str, ...] = ()
    resolved_entity_value_relation: Tuple[Tuple[str, str, float], ...] = ()
    canonical_program_id: str = ""
    execution_outcome: str = "NOT_RUN"
    projection_outcome: str = "NOT_RUN"
    projection_result: Dict[str, Any] = field(default_factory=dict)
    projected_answer: str = ""
    resource_complete: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True)
class PlanClosure:
    plan_id: str
    operation_family: str
    planner_version: str = ""
    closure_version: str = PLAN_CLOSURE_VERSION
    operation_contract_version: str = OPERATION_CONTRACT_VERSION
    assignments: Tuple[GroundedAssignment, ...] = ()
    executable_derivations: Tuple[ExecutableDerivation, ...] = ()
    outcome_counts: Dict[str, int] = field(default_factory=dict)
    construction_trace: Tuple[str, ...] = field(default_factory=tuple)
    declared_assignment_count: int = 0
    realized_assignment_count: int = 0
    deduplicated_program_count: int = 0
    resource_complete: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True)
class _GroundingRecord:
    outcome: ClosureOutcome
    resolution_state: str
    matched_cell_ids: Tuple[str, ...] = ()
    required_edge_triples: Tuple[Tuple[str, str, str], ...] = ()
    resolved_atomic_operands: Tuple[Tuple[str, str], ...] = ()
    resolved_scope_node_ids: Tuple[str, ...] = ()
    relation: Optional[EntityMeasureRelationResolution] = None
    failure_reasons: Tuple[str, ...] = ()
    resource_complete: bool = True


@dataclass(frozen=True)
class _PlanAssignmentSpace:
    plan: Mapping[str, Any]
    plan_id: str
    operation_family: str
    contract: Optional[OperationContract] = None
    role_bindings: Mapping[str, Any] = field(default_factory=dict)
    assignment_domains: Tuple[Tuple[str, Tuple[Any, ...]], ...] = ()
    declared_count: int = 1
    invalid_assignment: Optional[GroundedAssignment] = None


def _as_nonempty_str(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _payload_plans(payload: Mapping[str, Any]) -> Tuple[Tuple[Mapping[str, Any], ...], str, str]:
    plans = payload.get("plans")
    if isinstance(plans, list):
        typed_plans = tuple(item for item in plans if isinstance(item, Mapping))
        planner_version = str(payload.get("planner_version") or "")
        plan_id = str(payload.get("closure_id") or payload.get("plan_id") or "payload")
        return typed_plans, planner_version, plan_id
    return (payload,), str(payload.get("planner_version") or ""), str(payload.get("plan_id") or "P0")


def _reference_ids(graph: HCEG) -> Tuple[str, ...]:
    return tuple(sorted(
        node_id
        for node_id, node in graph.nodes.items()
        if node.node_type == NodeType.HEADER
        or (node.node_type == NodeType.AGGREGATOR and node.header_level >= 0)
    ))


def _role_options(
    role: str,
    bindings: Mapping[str, Any],
    domains: Mapping[str, Tuple[Any, ...]],
) -> Tuple[Any, ...]:
    if role in domains:
        return tuple(domains[role])
    if role in bindings:
        return (bindings[role],)
    return ()


def _assignment_domains(
    contract: OperationContract,
    bindings: Mapping[str, Any],
    domains: Mapping[str, Tuple[Any, ...]],
) -> Tuple[Tuple[str, Tuple[Any, ...]], ...]:
    rows = []
    for role in contract.allowed_roles:
        options = _role_options(role.name, bindings, domains)
        if options:
            rows.append((role.name, tuple(sorted(set(options)))))
    return tuple(rows)


def _assignment_key(
    operation: str,
    role_bindings: Mapping[str, Any],
    polarity: str,
    projection_operator: str,
    answer_domain: str,
) -> str:
    roles = "|".join(
        f"{role}={canonical_json(values)}"
        for role, values in sorted(role_bindings.items())
    )
    parts = [
        operation,
        roles,
        f"projection_operator={projection_operator}",
        f"answer_domain={answer_domain}",
    ]
    if polarity:
        parts.append(f"comparison_polarity={polarity}")
    return "|".join(parts)


def _invalid_assignment(
    *,
    plan_id: str,
    operation_family: str,
    outcome: ClosureOutcome,
    failure_reasons: Iterable[str],
    role_bindings: Optional[Mapping[str, Any]] = None,
    signature_id: str = "",
    semantic_result_role: str = "",
    comparison_polarity: str = "",
    projection_operator: str = "",
    answer_domain: str = "",
    resource_complete: bool = True,
) -> GroundedAssignment:
    bindings = {str(role): values for role, values in (role_bindings or {}).items()}
    reasons = tuple(sorted({str(item) for item in failure_reasons if str(item)}))
    return GroundedAssignment(
        plan_id=plan_id,
        plan_ids=(plan_id,),
        assignment_id=f"{plan_id}:A0",
        assignment_key=(
            f"{outcome.value}:"
            f"{_assignment_key(operation_family, bindings, comparison_polarity, projection_operator, answer_domain)}"
        ),
        role_bindings=bindings,
        outcome=outcome,
        operation_family=operation_family,
        signature_id=signature_id,
        semantic_result_role=semantic_result_role,
        comparison_polarity=comparison_polarity,
        projection_operator=projection_operator,
        answer_domain=answer_domain,
        resolution_state=outcome.value,
        failure_reasons=reasons,
        resource_complete=resource_complete,
    )


def _resolution_outcome(state: ResolutionState) -> ClosureOutcome:
    if state == ResolutionState.UNIQUE:
        return ClosureOutcome.UNIQUE_EXECUTABLE
    if state == ResolutionState.AMBIGUOUS:
        return ClosureOutcome.AMBIGUOUS_BINDING
    if state == ResolutionState.RESOURCE_INCOMPLETE:
        return ClosureOutcome.RESOURCE_INCOMPLETE
    return ClosureOutcome.UNRESOLVED_BINDING


def _ground_atomic_operation(
    graph: HCEG,
    role_bindings: Mapping[str, Any],
    roles: Sequence[str],
) -> _GroundingRecord:
    resolved = []
    matched = []
    edges = []
    for role in roles:
        resolution = resolve_atomic_operand(graph, role_bindings.get(role, ()))
        if resolution.state != ResolutionState.UNIQUE:
            return _GroundingRecord(
                outcome=_resolution_outcome(resolution.state),
                resolution_state=resolution.state.value,
                matched_cell_ids=resolution.candidate_node_ids,
                required_edge_triples=resolution.required_edge_triples,
                failure_reasons=resolution.failure_reasons
                or (f"{role.lower()}_binding_{resolution.state.value.lower()}",),
                resource_complete=resolution.resource_complete,
            )
        node = graph.nodes[resolution.unique_node_id]
        if node.numeric_value is None:
            return _GroundingRecord(
                outcome=ClosureOutcome.UNRESOLVED_BINDING,
                resolution_state=ResolutionState.UNRESOLVED.value,
                matched_cell_ids=(resolution.unique_node_id,),
                required_edge_triples=resolution.required_edge_triples,
                failure_reasons=(f"non_numeric_operand:{role}:{resolution.unique_node_id}",),
            )
        resolved.append((role, resolution.unique_node_id))
        matched.append(resolution.unique_node_id)
        edges.extend(resolution.required_edge_triples)
    return _GroundingRecord(
        outcome=ClosureOutcome.UNIQUE_EXECUTABLE,
        resolution_state=ResolutionState.UNIQUE.value,
        matched_cell_ids=tuple(matched),
        required_edge_triples=tuple(sorted(set(edges))),
        resolved_atomic_operands=tuple(resolved),
    )


def _ground_lookup(graph: HCEG, role_bindings: Mapping[str, Any]) -> _GroundingRecord:
    binding_ids = (
        *role_bindings.get("TARGET_ENTITY", ()),
        *role_bindings.get("TARGET_MEASURE", ()),
        *role_bindings.get("TIME_SCOPE", ()),
    )
    resolution = resolve_atomic_operand(graph, binding_ids)
    return _GroundingRecord(
        outcome=_resolution_outcome(resolution.state),
        resolution_state=resolution.state.value,
        matched_cell_ids=resolution.candidate_node_ids,
        required_edge_triples=resolution.required_edge_triples,
        resolved_atomic_operands=(
            (("TARGET", resolution.unique_node_id),)
            if resolution.state == ResolutionState.UNIQUE
            else ()
        ),
        failure_reasons=resolution.failure_reasons,
        resource_complete=resolution.resource_complete,
    )


def _scope_member_conjunctions(value: Any) -> Tuple[Tuple[str, ...], ...]:
    members = tuple(value or ())
    if members and all(not isinstance(member, (list, tuple)) for member in members):
        return tuple((str(member),) for member in members)
    return tuple(tuple(str(item) for item in member) for member in members)


def _scope_inputs(
    role_bindings: Mapping[str, Any],
) -> Tuple[Tuple[Tuple[str, ...], ...], Tuple[str, ...]]:
    members = _scope_member_conjunctions(
        role_bindings.get("AGGREGATION_SCOPE", ())
    )
    shared = (
        *role_bindings.get("TARGET_MEASURE", ()),
        *role_bindings.get("GROUP_SCOPE", ()),
        *role_bindings.get("TIME_SCOPE", ()),
    )
    return members, tuple(shared)


def _ground_scope(
    graph: HCEG,
    role_bindings: Mapping[str, Any],
    *,
    require_numeric: bool,
) -> _GroundingRecord:
    members, shared = _scope_inputs(role_bindings)
    scope = resolve_finite_scope(
        graph,
        members,
        shared_binding_ids=shared,
        require_numeric=require_numeric,
    )
    return _GroundingRecord(
        outcome=_resolution_outcome(scope.state),
        resolution_state=scope.state.value,
        matched_cell_ids=scope.member_node_ids,
        required_edge_triples=scope.required_edge_triples,
        resolved_scope_node_ids=scope.member_node_ids,
        failure_reasons=scope.failure_reasons,
        resource_complete=scope.resource_complete,
    )


def _ground_relation(
    graph: HCEG,
    role_bindings: Mapping[str, Any],
) -> _GroundingRecord:
    members = _scope_member_conjunctions(
        role_bindings.get("AGGREGATION_SCOPE", ())
    )
    relation = resolve_entity_measure_relation(
        graph,
        members,
        measure_binding_ids=role_bindings.get("TARGET_MEASURE", ()),
        shared_binding_ids=(
            *role_bindings.get("GROUP_SCOPE", ()),
            *role_bindings.get("TIME_SCOPE", ()),
        ),
    )
    return _GroundingRecord(
        outcome=_resolution_outcome(relation.state),
        resolution_state=relation.state.value,
        matched_cell_ids=tuple(member.value_node_id for member in relation.members),
        required_edge_triples=relation.required_edge_triples,
        resolved_scope_node_ids=tuple(member.value_node_id for member in relation.members),
        relation=relation,
        failure_reasons=relation.failure_reasons,
        resource_complete=relation.resource_complete,
    )


def _ground_operation(
    contract: OperationContract,
    graph: HCEG,
    role_bindings: Mapping[str, Any],
) -> _GroundingRecord:
    if contract.resolution_mode == "atomic_lookup":
        return _ground_lookup(graph, role_bindings)
    if contract.resolution_mode == "ordered_atomic_operands":
        return _ground_atomic_operation(graph, role_bindings, ("LEFT_OPERAND", "RIGHT_OPERAND"))
    if contract.resolution_mode in {"finite_numeric_scope", "finite_scope"}:
        return _ground_scope(
            graph,
            role_bindings,
            require_numeric=contract.resolution_mode == "finite_numeric_scope",
        )
    if contract.resolution_mode == "entity_value_relation":
        return _ground_relation(graph, role_bindings)
    return _GroundingRecord(
        outcome=ClosureOutcome.STRUCTURALLY_INVALID,
        resolution_state=ClosureOutcome.STRUCTURALLY_INVALID.value,
        failure_reasons=(f"unknown_resolution_mode:{contract.resolution_mode}",),
    )


def _relation_identity(relation: Optional[EntityMeasureRelationResolution]) -> Tuple[Tuple[str, str, float], ...]:
    if relation is None:
        return ()
    return tuple(
        (
            ",".join(member.entity_binding_ids),
            member.value_node_id,
            member.numeric_value,
        )
        for member in relation.members
    )


def _role_reference_ids(value: Any) -> Tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(
            reference
            for item in value
            for reference in _role_reference_ids(item)
        )
    return (str(value),) if str(value) else ()


def _program_payload(
    *,
    contract: OperationContract,
    role_bindings: Mapping[str, Any],
    grounding: _GroundingRecord,
    projection_operator: str,
    answer_domain: str,
    comparison_polarity: str,
) -> Dict[str, Any]:
    canonical_roles = {
        role.name: list(role_bindings.get(role.name, ()))
        for role in contract.allowed_roles
        if role.name in role_bindings
    }
    return {
        "contract_version": OPERATION_CONTRACT_VERSION,
        "signature_id": contract.signature_id,
        "operation_family": contract.operation_family,
        "semantic_result_role": contract.semantic_result_role,
        "role_bindings": canonical_roles,
        "atomic_operands": [list(item) for item in grounding.resolved_atomic_operands],
        "scope_node_ids": list(grounding.resolved_scope_node_ids),
        "entity_value_relation": [list(item) for item in _relation_identity(grounding.relation)],
        "comparison_polarity": comparison_polarity,
        "projection_operator": projection_operator,
        "answer_domain": answer_domain,
    }


def _operand_metadata(
    graph: HCEG,
    grounding: _GroundingRecord,
    role_bindings: Mapping[str, Any],
) -> list[Dict[str, Any]]:
    relation_by_node = {
        member.value_node_id: member
        for member in (grounding.relation.members if grounding.relation is not None else ())
    }
    metadata = []
    for node_id in grounding.matched_cell_ids:
        node = graph.nodes[node_id]
        item: Dict[str, Any] = {
            "node_id": node_id,
            "row": node.row,
            "col": node.col,
            "target_entity_ids": list(role_bindings.get("TARGET_ENTITY", ())),
            "target_measure_ids": list(role_bindings.get("TARGET_MEASURE", ())),
            "time_scope_ids": list(role_bindings.get("TIME_SCOPE", ())),
        }
        relation_member = relation_by_node.get(node_id)
        if relation_member is not None:
            item["row_headers"] = [relation_member.entity_label]
            item["entity_binding_ids"] = list(relation_member.entity_binding_ids)
        metadata.append(item)
    return metadata


def _execute_grounded(
    *,
    contract: OperationContract,
    plan: Mapping[str, Any],
    graph: HCEG,
    role_bindings: Mapping[str, Any],
    grounding: _GroundingRecord,
) -> Tuple[GroundedAssignment, Optional[ExecutableDerivation]]:
    plan_id = str(plan.get("plan_id") or "P0")
    polarity = str(plan.get("comparison_polarity") or "")
    projection = str(plan.get("projection_operator") or "")
    answer_domain = str(plan.get("answer_domain") or "")
    assignment_key = _assignment_key(
        contract.operation_family,
        role_bindings,
        polarity,
        projection,
        answer_domain,
    )
    if grounding.outcome != ClosureOutcome.UNIQUE_EXECUTABLE:
        return GroundedAssignment(
            plan_id=plan_id,
            plan_ids=(plan_id,),
            assignment_id="",
            assignment_key=assignment_key,
            role_bindings=dict(role_bindings),
            outcome=grounding.outcome,
            operation_family=contract.operation_family,
            signature_id=contract.signature_id,
            semantic_result_role=contract.semantic_result_role,
            comparison_polarity=polarity,
            projection_operator=projection,
            answer_domain=answer_domain,
            target_entity_ids=role_bindings.get("TARGET_ENTITY", ()),
            target_measure_ids=role_bindings.get("TARGET_MEASURE", ()),
            time_scope_ids=role_bindings.get("TIME_SCOPE", ()),
            resolution_state=grounding.resolution_state,
            matched_cell_ids=grounding.matched_cell_ids,
            required_edge_triples=grounding.required_edge_triples,
            resolved_atomic_operands=grounding.resolved_atomic_operands,
            resolved_scope_node_ids=grounding.resolved_scope_node_ids,
            resolved_entity_value_relation=_relation_identity(grounding.relation),
            failure_reasons=grounding.failure_reasons,
            resource_complete=grounding.resource_complete,
        ), None

    program = _program_payload(
        contract=contract,
        role_bindings=role_bindings,
        grounding=grounding,
        projection_operator=projection,
        answer_domain=answer_domain,
        comparison_polarity=polarity,
    )
    program_id = f"CP-{canonical_json_hash(program, 24)}"
    derivation_id = f"PLC-{canonical_json_hash(program, 20)}"
    operand_ids = list(grounding.matched_cell_ids)
    shell = ExecutableDerivation(
        derivation_id=derivation_id,
        source_candidate_id=f"closure:{program_id}",
        operation_family=contract.operation_family,
        operand_node_ids=operand_ids,
        required_edge_triples=list(grounding.required_edge_triples),
        typed_signature=contract.signature_id,
        projection_operator=projection,
        projected_answer="",
        output_domain=answer_domain,
        evidence_ids=sorted({
            *operand_ids,
            *[
                reference
                for value in role_bindings.values()
                for reference in _role_reference_ids(value)
            ],
        }),
        executable_program=canonical_json(program),
        provenance_complete=bool(grounding.required_edge_triples),
        availability="available",
        operand_metadata=_operand_metadata(graph, grounding, role_bindings),
        source_candidate={"source": "plan_conditioned_operation_closure", "plan_ids": [plan_id]},
        operation_metadata={
            "planner_compiled": False,
            "plan_conditioned_closure": True,
            "closure_version": PLAN_CLOSURE_VERSION,
            "operation_contract_version": OPERATION_CONTRACT_VERSION,
            "signature_id": contract.signature_id,
            "semantic_result_role": contract.semantic_result_role,
            "plan_ids": [plan_id],
            "assignment_key": assignment_key,
            "canonical_program_id": program_id,
            "role_bindings": {role: list(values) for role, values in role_bindings.items()},
            "resolved_scope_node_ids": list(grounding.resolved_scope_node_ids),
            "resolved_entity_value_relation": [list(item) for item in _relation_identity(grounding.relation)],
            "resource_complete": grounding.resource_complete,
        },
        comparison_polarity=polarity or "unknown",
    )
    nodes = [graph.nodes[node_id] for node_id in operand_ids]
    projection_result = execute_typed_projection_from_nodes(shell, nodes, graph=graph)
    if projection_result.status != ProjectionStatus.PROJECTED:
        assignment = GroundedAssignment(
            plan_id=plan_id,
            plan_ids=(plan_id,),
            assignment_id="",
            assignment_key=assignment_key,
            role_bindings=dict(role_bindings),
            outcome=ClosureOutcome.EXECUTION_FAILED,
            operation_family=contract.operation_family,
            signature_id=contract.signature_id,
            semantic_result_role=contract.semantic_result_role,
            comparison_polarity=polarity,
            projection_operator=projection,
            answer_domain=answer_domain,
            target_entity_ids=role_bindings.get("TARGET_ENTITY", ()),
            target_measure_ids=role_bindings.get("TARGET_MEASURE", ()),
            time_scope_ids=role_bindings.get("TIME_SCOPE", ()),
            resolution_state=grounding.resolution_state,
            matched_cell_ids=grounding.matched_cell_ids,
            required_edge_triples=grounding.required_edge_triples,
            resolved_atomic_operands=grounding.resolved_atomic_operands,
            resolved_scope_node_ids=grounding.resolved_scope_node_ids,
            resolved_entity_value_relation=_relation_identity(grounding.relation),
            canonical_program_id=program_id,
            failure_reasons=tuple(projection_result.failure_reasons or ("projection_execution_failed",)),
            execution_outcome="EXECUTED",
            projection_outcome=projection_result.status.value,
            projection_result=dict(projection_result.to_dict()),
        )
        return assignment, None

    output_domain = projection_result.output_domain
    if output_domain != contract.answer_domain:
        assignment = GroundedAssignment(
            plan_id=plan_id,
            plan_ids=(plan_id,),
            assignment_id="",
            assignment_key=assignment_key,
            role_bindings=dict(role_bindings),
            outcome=ClosureOutcome.EXECUTION_FAILED,
            operation_family=contract.operation_family,
            signature_id=contract.signature_id,
            semantic_result_role=contract.semantic_result_role,
            comparison_polarity=polarity,
            projection_operator=projection,
            answer_domain=answer_domain,
            target_entity_ids=role_bindings.get("TARGET_ENTITY", ()),
            target_measure_ids=role_bindings.get("TARGET_MEASURE", ()),
            time_scope_ids=role_bindings.get("TIME_SCOPE", ()),
            resolution_state=grounding.resolution_state,
            matched_cell_ids=grounding.matched_cell_ids,
            required_edge_triples=grounding.required_edge_triples,
            resolved_atomic_operands=grounding.resolved_atomic_operands,
            resolved_scope_node_ids=grounding.resolved_scope_node_ids,
            resolved_entity_value_relation=_relation_identity(grounding.relation),
            canonical_program_id=program_id,
            failure_reasons=(
                f"projected_answer_domain_mismatch:{output_domain}!={contract.answer_domain}",
            ),
            execution_outcome="EXECUTED",
            projection_outcome="TYPE_MISMATCH",
            projection_result=dict(projection_result.to_dict()),
            projected_answer=projection_result.value,
        )
        return assignment, None
    derivation = replace(shell, projected_answer=projection_result.value, output_domain=output_domain)
    assignment = GroundedAssignment(
        plan_id=plan_id,
        plan_ids=(plan_id,),
        assignment_id="",
        assignment_key=assignment_key,
        role_bindings=dict(role_bindings),
        outcome=ClosureOutcome.UNIQUE_EXECUTABLE,
        operation_family=contract.operation_family,
        signature_id=contract.signature_id,
        semantic_result_role=contract.semantic_result_role,
        comparison_polarity=polarity,
        projection_operator=projection,
        answer_domain=answer_domain,
        target_entity_ids=role_bindings.get("TARGET_ENTITY", ()),
        target_measure_ids=role_bindings.get("TARGET_MEASURE", ()),
        time_scope_ids=role_bindings.get("TIME_SCOPE", ()),
        resolution_state=grounding.resolution_state,
        matched_cell_ids=grounding.matched_cell_ids,
        required_edge_triples=grounding.required_edge_triples,
        resolved_atomic_operands=grounding.resolved_atomic_operands,
        resolved_scope_node_ids=grounding.resolved_scope_node_ids,
        resolved_entity_value_relation=_relation_identity(grounding.relation),
        canonical_program_id=program_id,
        derivation_id=derivation_id,
        execution_outcome="EXECUTED",
        projection_outcome="PROJECTED",
        projection_result=dict(projection_result.to_dict()),
        projected_answer=projection_result.value,
    )
    return assignment, derivation


def _prepare_plan_assignment_space(
    *,
    plan: Mapping[str, Any],
    graph: HCEG,
    allowed_signature_ids: Optional[Sequence[str]] = None,
) -> _PlanAssignmentSpace:
    normalized_plan = dict(plan)
    raw_domains = plan.get("role_domains") or {}
    if isinstance(raw_domains, Mapping):
        canonical_domains: Dict[str, list[Any]] = {}
        for role, options in raw_domains.items():
            if not isinstance(options, list):
                canonical_domains[str(role)] = options
                continue
            by_key = {
                canonical_json(option): option
                for option in options
            }
            canonical_domains[str(role)] = [by_key[key] for key in sorted(by_key)]
        normalized_plan["role_domains"] = canonical_domains
    plan = normalized_plan
    plan_id = str(plan.get("plan_id") or "P0")
    operation_family = str(plan.get("operation_family") or "")
    comparison_polarity = str(plan.get("comparison_polarity") or "")
    projection_operator = str(plan.get("projection_operator") or "")
    answer_domain = str(plan.get("answer_domain") or "")
    contract = resolve_operation_signature(plan)
    if contract is None and not plan.get("signature_id"):
        contract = get_operation_contract(operation_family)
    if contract is None:
        return _PlanAssignmentSpace(
            plan=plan,
            plan_id=plan_id,
            operation_family=operation_family,
            invalid_assignment=_invalid_assignment(
                plan_id=plan_id,
                operation_family=operation_family,
                outcome=ClosureOutcome.STRUCTURALLY_INVALID,
                failure_reasons=[f"unsupported_operation_family:{operation_family}"],
                comparison_polarity=comparison_polarity,
                projection_operator=projection_operator,
                answer_domain=answer_domain,
            ),
        )
    if allowed_signature_ids is not None and contract.signature_id not in set(allowed_signature_ids):
        return _PlanAssignmentSpace(
            plan=plan,
            plan_id=plan_id,
            operation_family=operation_family,
            invalid_assignment=_invalid_assignment(
                plan_id=plan_id,
                operation_family=operation_family,
                outcome=ClosureOutcome.STRUCTURALLY_INVALID,
                failure_reasons=[f"signature_outside_active_allowlist:{contract.signature_id}"],
                signature_id=contract.signature_id,
                semantic_result_role=contract.semantic_result_role,
                comparison_polarity=comparison_polarity,
                projection_operator=projection_operator,
                answer_domain=answer_domain,
            ),
        )
    validation = validate_operation_plan(plan, _reference_ids(graph))
    if not validation.ok:
        return _PlanAssignmentSpace(
            plan=plan,
            plan_id=plan_id,
            operation_family=operation_family,
            invalid_assignment=_invalid_assignment(
                plan_id=plan_id,
                operation_family=operation_family,
                outcome=ClosureOutcome.STRUCTURALLY_INVALID,
                failure_reasons=validation.errors,
                role_bindings=validation.role_bindings_dict(),
                signature_id=str(plan.get("signature_id") or ""),
                semantic_result_role=str(plan.get("semantic_result_role") or ""),
                comparison_polarity=comparison_polarity,
                projection_operator=projection_operator,
                answer_domain=answer_domain,
            ),
        )
    bindings = validation.role_bindings_dict()
    domains = validation.role_domains_dict()
    assignment_domains = _assignment_domains(contract, bindings, domains)
    declared_count = prod(len(options) for _, options in assignment_domains)
    return _PlanAssignmentSpace(
        plan=plan,
        plan_id=plan_id,
        operation_family=operation_family,
        contract=contract,
        role_bindings=bindings,
        assignment_domains=assignment_domains,
        declared_count=declared_count,
    )


def _realize_plan_assignment_space(
    space: _PlanAssignmentSpace,
    graph: HCEG,
) -> list[Tuple[GroundedAssignment, Optional[ExecutableDerivation]]]:
    if space.invalid_assignment is not None or space.contract is None:
        return [
            (space.invalid_assignment, None)
        ] if space.invalid_assignment is not None else []
    rows = []
    role_names = tuple(role for role, _ in space.assignment_domains)
    for option_values in product(*(options for _, options in space.assignment_domains)):
        role_bindings = {
            role: tuple(values)
            for role, values in zip(role_names, option_values)
        }
        grounding = _ground_operation(space.contract, graph, role_bindings)
        rows.append(_execute_grounded(
            contract=space.contract,
            plan=space.plan,
            graph=graph,
            role_bindings=role_bindings,
            grounding=grounding,
        ))
    return rows


def _resource_incomplete_assignment(
    space: _PlanAssignmentSpace,
    *,
    total_declared: int,
    max_assignments: int,
) -> GroundedAssignment:
    return _invalid_assignment(
        plan_id=space.plan_id,
        operation_family=space.operation_family,
        outcome=ClosureOutcome.RESOURCE_INCOMPLETE,
        failure_reasons=[f"global_assignment_cap_exceeded:{total_declared}>{max_assignments}"],
        role_bindings=space.role_bindings,
        signature_id=(space.contract.signature_id if space.contract is not None else ""),
        semantic_result_role=(
            space.contract.semantic_result_role if space.contract is not None else ""
        ),
        comparison_polarity=str(space.plan.get("comparison_polarity") or ""),
        projection_operator=str(space.plan.get("projection_operator") or ""),
        answer_domain=str(space.plan.get("answer_domain") or ""),
        resource_complete=False,
    )


def _merge_assignment(existing: GroundedAssignment, incoming: GroundedAssignment) -> GroundedAssignment:
    return replace(
        existing,
        plan_ids=tuple(sorted({*existing.plan_ids, *incoming.plan_ids})),
        failure_reasons=tuple(sorted({*existing.failure_reasons, *incoming.failure_reasons})),
        resource_complete=existing.resource_complete and incoming.resource_complete,
    )


def _count_outcomes(assignments: Iterable[GroundedAssignment]) -> Dict[str, int]:
    counts = {outcome.value: 0 for outcome in ClosureOutcome}
    for assignment in assignments:
        counts[assignment.outcome.value] += 1
    return counts


def build_plan_closure(
    payload: Mapping[str, Any],
    graph: HCEG,
    *,
    max_assignments: Optional[int] = None,
    allowed_signature_ids: Optional[Sequence[str]] = None,
) -> PlanClosure:
    """Construct the complete finite operation-typed closure for a Planner payload."""
    plans, planner_version, closure_plan_id = _payload_plans(payload)
    spaces = tuple(
        _prepare_plan_assignment_space(
            plan=plan,
            graph=graph,
            allowed_signature_ids=allowed_signature_ids,
        )
        for plan in plans
    )
    rows: list[Tuple[GroundedAssignment, Optional[ExecutableDerivation]]] = []
    declared_assignment_count = sum(space.declared_count for space in spaces)
    valid_declared_count = sum(
        space.declared_count
        for space in spaces
        if space.invalid_assignment is None
    )
    resource_blocked = (
        max_assignments is not None
        and valid_declared_count > max_assignments
    )
    realized_assignment_count = 0
    if resource_blocked:
        for space in spaces:
            if space.invalid_assignment is not None:
                rows.append((space.invalid_assignment, None))
            else:
                rows.append((
                    _resource_incomplete_assignment(
                        space,
                        total_declared=valid_declared_count,
                        max_assignments=max_assignments,
                    ),
                    None,
                ))
    else:
        for space in spaces:
            rows.extend(_realize_plan_assignment_space(space, graph))
            if space.invalid_assignment is None:
                realized_assignment_count += space.declared_count

    by_key: Dict[str, Tuple[GroundedAssignment, Optional[ExecutableDerivation]]] = {}
    operation_rank = {operation: index for index, operation in enumerate(FINAL_SUPPORTED_OPERATIONS)}
    for assignment, derivation in rows:
        dedupe_identity = assignment.canonical_program_id or assignment.assignment_key
        dedupe_key = f"{assignment.outcome.value}|{dedupe_identity}"
        if dedupe_key in by_key:
            existing, existing_derivation = by_key[dedupe_key]
            merged = _merge_assignment(existing, assignment)
            if existing_derivation is not None:
                metadata = dict(existing_derivation.operation_metadata)
                metadata["plan_ids"] = list(merged.plan_ids)
                existing_derivation = replace(
                    existing_derivation,
                    source_candidate={"source": "plan_conditioned_operation_closure", "plan_ids": list(merged.plan_ids)},
                    operation_metadata=metadata,
                )
            by_key[dedupe_key] = (merged, existing_derivation)
        else:
            by_key[dedupe_key] = (assignment, derivation)

    ordered_rows = sorted(
        by_key.values(),
        key=lambda row: (
            operation_rank.get(row[0].operation_family, len(operation_rank)),
            row[0].assignment_key,
            row[0].outcome.value,
        ),
    )
    normalized_rows = []
    for index, (assignment, derivation) in enumerate(ordered_rows, start=1):
        normalized_rows.append((replace(assignment, assignment_id=f"A{index}"), derivation))
    assignments = tuple(row[0] for row in normalized_rows)
    derivations = tuple(row[1] for row in normalized_rows if row[1] is not None)
    operations = tuple(sorted(
        {str(plan.get("operation_family") or "") for plan in plans if plan.get("operation_family")},
        key=lambda operation: operation_rank.get(operation, len(operation_rank)),
    ))
    active_roles = sorted({
        role
        for assignment in assignments
        for role, values in assignment.role_bindings.items()
        if values
    })
    resource_complete = all(assignment.resource_complete for assignment in assignments)
    return PlanClosure(
        plan_id=closure_plan_id,
        operation_family=",".join(operations),
        planner_version=planner_version,
        assignments=assignments,
        executable_derivations=derivations,
        outcome_counts=_count_outcomes(assignments),
        construction_trace=(
            f"contract={OPERATION_CONTRACT_VERSION}",
            f"plans={len(plans)}",
            f"roles={','.join(active_roles)}",
            f"declared_assignments={declared_assignment_count}",
            f"realized_assignments={realized_assignment_count}",
            f"deduplicated_programs={len(derivations)}",
            f"resource_complete={str(resource_complete).lower()}",
        ),
        declared_assignment_count=declared_assignment_count,
        realized_assignment_count=realized_assignment_count,
        deduplicated_program_count=len(derivations),
        resource_complete=resource_complete,
    )
