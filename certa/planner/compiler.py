"""Deterministic compiler from typed planner skeletons to derivations."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Dict, Mapping, Optional, Tuple

from graph_builder import HCEG

from certa.derivations.answer_equivalence import inference_answer_key
from certa.derivations.schema import ExecutableDerivation
from certa.planner.lookup_resolver import resolve_lookup_binding


@dataclass
class PlanCompilationResult:
    derivations: list[ExecutableDerivation] = field(default_factory=list)
    failures: list[Dict[str, Any]] = field(default_factory=list)

    @property
    def failure_count(self) -> int:
        return len(self.failures)


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value or ""))


def _as_plans(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    plans = payload.get("plans") or []
    if not isinstance(plans, list):
        return []
    return [item for item in plans if isinstance(item, Mapping)]


def _role_ids(plan: Mapping[str, Any], role: str) -> list[str]:
    bindings = plan.get("role_bindings") or {}
    if not isinstance(bindings, Mapping):
        return []
    values = bindings.get(role) or []
    if not isinstance(values, list):
        return []
    return [str(item) for item in values]


def _role_options(plan: Mapping[str, Any], role: str) -> list[list[str]]:
    bound = _role_ids(plan, role)
    if bound:
        return [bound]
    domains = plan.get("role_domains") or {}
    values = domains.get(role) or [] if isinstance(domains, Mapping) else []
    if not isinstance(values, list):
        return []
    return [
        [str(item) for item in option]
        for option in values
        if isinstance(option, list) and option
    ]


def _lookup_assignments(plan: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    entities = _role_options(plan, "TARGET_ENTITY")
    measures = _role_options(plan, "TARGET_MEASURE")
    times = _role_options(plan, "TIME_SCOPE") or [[]]
    if not entities or not measures:
        return [plan]
    assignments = []
    combinations = list(product(entities, measures, times))
    for index, (entity, measure, time_scope) in enumerate(combinations):
        expanded = dict(plan)
        bindings = dict(plan.get("role_bindings") or {})
        bindings.update({"TARGET_ENTITY": entity, "TARGET_MEASURE": measure})
        if time_scope:
            bindings["TIME_SCOPE"] = time_scope
        expanded["role_bindings"] = bindings
        expanded.pop("role_domains", None)
        if len(combinations) > 1:
            expanded["plan_id"] = f"{str(plan.get('plan_id') or 'P0')}D{index}"
        assignments.append(expanded)
    return assignments


def _output_domain(value: Any) -> str:
    key = inference_answer_key(value)
    if key.category.startswith("NUMERIC"):
        return "SCALAR"
    if key.category == "BOOLEAN_EXACT":
        return "BOOLEAN"
    if key.category == "SET_EXACT_NORMALIZED":
        return "SET"
    return "ENTITY"


def _compile_lookup(
    plan: Mapping[str, Any],
    graph: HCEG,
    index: int,
) -> Tuple[Optional[ExecutableDerivation], Optional[Dict[str, Any]]]:
    plan_id = str(plan.get("plan_id") or f"P{index}")
    entity_ids = _role_ids(plan, "TARGET_ENTITY")
    measure_ids = _role_ids(plan, "TARGET_MEASURE")
    time_scope_ids = _role_ids(plan, "TIME_SCOPE")
    if not entity_ids or not measure_ids:
        return None, {"plan_id": plan_id, "reason": "missing_lookup_role_binding"}
    resolution = resolve_lookup_binding(
        graph,
        target_entity_ids=entity_ids,
        target_measure_ids=measure_ids,
        time_scope_ids=time_scope_ids,
    )
    if not resolution.unique:
        reason = "ambiguous_lookup_binding" if resolution.state == "ambiguous" else "lookup_binding_unresolved"
        return None, {
            "plan_id": plan_id,
            "reason": reason,
            "resolution_state": resolution.state,
            "match_count": len(resolution.matched_cell_ids),
            "matched_cell_ids": list(resolution.matched_cell_ids),
        }
    cell_id = resolution.matched_cell_ids[0]
    node = graph.nodes[cell_id]
    required_edges = list(resolution.required_edge_triples)
    signature_roles = [
        f"TARGET_ENTITY={','.join(entity_ids)}",
        f"TARGET_MEASURE={','.join(measure_ids)}",
    ]
    if time_scope_ids:
        signature_roles.append(f"TIME_SCOPE={','.join(time_scope_ids)}")
    derivation = ExecutableDerivation(
        derivation_id=f"TPD-{plan_id}-0",
        source_candidate_id=f"planner:{plan_id}",
        operation_family="LOOKUP",
        operand_node_ids=[cell_id],
        required_edge_triples=required_edges,
        typed_signature=f"LOOKUP|{'|'.join(signature_roles)}|VALUE_PROJECTION",
        projection_operator="VALUE_PROJECTION",
        projected_answer=str(node.text or ""),
        output_domain=_output_domain(node.text),
        evidence_ids=[cell_id],
        executable_program=json.dumps(
            {
                "op": "LOOKUP",
                "target_cell": cell_id,
                "target_entity": entity_ids,
                "target_measure": measure_ids,
                "time_scope": time_scope_ids,
            },
            sort_keys=True,
        ),
        provenance_complete=bool(required_edges),
        availability="available",
        operand_metadata=[
            {
                "node_id": cell_id,
                "row": node.row,
                "col": node.col,
                "target_entity_ids": entity_ids,
                "target_measure_ids": measure_ids,
                "time_scope_ids": time_scope_ids,
            }
        ],
        source_candidate={"source": "typed_derivation_planner_agent", "plan_id": plan_id},
        operation_metadata={"planner_compiled": True, "plan_id": plan_id},
    )
    return derivation, None


def compile_typed_plans_to_derivations(payload: Mapping[str, Any], graph: HCEG) -> PlanCompilationResult:
    derivations: list[ExecutableDerivation] = []
    failures: list[Dict[str, Any]] = []
    for index, plan in enumerate(_as_plans(payload)):
        plan_id = str(plan.get("plan_id") or f"P{index}")
        operation = str(plan.get("operation_family") or "")
        if operation != "LOOKUP":
            failures.append({"plan_id": plan_id, "reason": "unsupported_operation_family", "operation_family": operation})
            continue
        for assignment in _lookup_assignments(plan):
            derivation, failure = _compile_lookup(assignment, graph, index)
            if failure is not None:
                failures.append(failure)
                continue
            if derivation is not None:
                derivations.append(derivation)
    return PlanCompilationResult(derivations=derivations, failures=failures)
