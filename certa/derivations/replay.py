"""Replay executable derivations under graph interventions."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from .project import answers_equivalent, execute_projection_from_nodes
from .schema import ExecutableDerivation, ReplayResult


def _edge_set(graph: Any) -> set[tuple[str, str, str]]:
    out: set[tuple[str, str, str]] = set()
    for edge in getattr(graph, "edges", []) or []:
        etype = getattr(getattr(edge, "edge_type", ""), "value", getattr(edge, "edge_type", ""))
        out.add((str(getattr(edge, "source", "")), str(getattr(edge, "target", "")), str(etype)))
    return out


def _intervention_type(intervention: Any) -> str:
    value = getattr(intervention, "intervention_type", "")
    return str(getattr(value, "value", value or ""))


def _finish(
    *,
    intervention_id: str,
    derivation: ExecutableDerivation,
    intervention_type: str,
    post_answer: Optional[str],
    operation_executed: bool,
    projection_executed: bool,
    required_nodes_valid: bool,
    required_edges_valid: bool,
    operand_resolution_valid: bool,
    failure_reason: str = "",
    missing_node_ids: Optional[List[str]] = None,
    missing_edge_triples: Optional[List[Tuple[str, str, str]]] = None,
) -> ReplayResult:
    available = failure_reason != "missing_intervened_graph"
    changed = False
    if available:
        changed = not answers_equivalent(derivation.projected_answer, post_answer)
    elif failure_reason in {
        "required_nodes_missing",
        "required_edges_missing",
        "operand_resolution_failed",
        "projection_execution_failed",
    }:
        changed = True
    return ReplayResult(
        intervention_id=intervention_id,
        derivation_id=derivation.derivation_id,
        intervention_type=intervention_type,
        replay_mode="typed_same_derivation_replay_v1",
        pre_projected_answer=derivation.projected_answer,
        post_projected_answer=post_answer,
        operation_executed=operation_executed,
        projection_executed=projection_executed,
        required_nodes_valid=required_nodes_valid,
        required_edges_valid=required_edges_valid,
        operand_resolution_valid=operand_resolution_valid,
        changed=changed,
        available=available,
        failure_reason=failure_reason,
        missing_node_ids=missing_node_ids or [],
        missing_edge_triples=missing_edge_triples or [],
    )


def replay_derivation_under_intervention(
    *,
    intervention_id: str,
    derivation: ExecutableDerivation,
    intervention: Any,
) -> ReplayResult:
    post_graph = getattr(intervention, "intervened_graph", None)
    intervention_type = _intervention_type(intervention)
    if post_graph is None:
        return _finish(
            intervention_id=intervention_id,
            derivation=derivation,
            intervention_type=intervention_type,
            post_answer=None,
            operation_executed=False,
            projection_executed=False,
            required_nodes_valid=False,
            required_edges_valid=False,
            operand_resolution_valid=False,
            failure_reason="missing_intervened_graph",
        )

    nodes = getattr(post_graph, "nodes", {}) or {}
    missing_nodes = [node_id for node_id in derivation.operand_node_ids if node_id not in nodes]
    if missing_nodes:
        return _finish(
            intervention_id=intervention_id,
            derivation=derivation,
            intervention_type=intervention_type,
            post_answer=None,
            operation_executed=False,
            projection_executed=False,
            required_nodes_valid=False,
            required_edges_valid=True,
            operand_resolution_valid=False,
            failure_reason="required_nodes_missing",
            missing_node_ids=missing_nodes,
        )

    existing_edges = _edge_set(post_graph)
    missing_edges = [
        tuple(edge)
        for edge in derivation.required_edge_triples
        if tuple(edge) not in existing_edges
    ]
    if missing_edges:
        return _finish(
            intervention_id=intervention_id,
            derivation=derivation,
            intervention_type=intervention_type,
            post_answer=None,
            operation_executed=False,
            projection_executed=False,
            required_nodes_valid=True,
            required_edges_valid=False,
            operand_resolution_valid=True,
            failure_reason="required_edges_missing",
            missing_edge_triples=missing_edges,
        )

    resolved = [nodes[node_id] for node_id in derivation.operand_node_ids]
    if len(resolved) != len(derivation.operand_node_ids):
        return _finish(
            intervention_id=intervention_id,
            derivation=derivation,
            intervention_type=intervention_type,
            post_answer=None,
            operation_executed=False,
            projection_executed=False,
            required_nodes_valid=True,
            required_edges_valid=True,
            operand_resolution_valid=False,
            failure_reason="operand_resolution_failed",
        )

    post_answer, projection_failures = execute_projection_from_nodes(
        derivation,
        resolved,
        graph=post_graph,
    )
    if projection_failures or post_answer is None:
        return _finish(
            intervention_id=intervention_id,
            derivation=derivation,
            intervention_type=intervention_type,
            post_answer=post_answer,
            operation_executed=True,
            projection_executed=False,
            required_nodes_valid=True,
            required_edges_valid=True,
            operand_resolution_valid=True,
            failure_reason=projection_failures[0] if projection_failures else "projection_execution_failed",
        )

    return _finish(
        intervention_id=intervention_id,
        derivation=derivation,
        intervention_type=intervention_type,
        post_answer=post_answer,
        operation_executed=True,
        projection_executed=True,
        required_nodes_valid=True,
        required_edges_valid=True,
        operand_resolution_valid=True,
    )
