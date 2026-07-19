"""Counterfactual serialization helpers for CERTA evidence packets."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from certa.reproducibility.canonical_json import canonical_json_hash


def graph_state_fingerprint(graph: Any, n: int = 16) -> str:
    """Return a stable short fingerprint for an HCEG-like graph."""
    payload = {}
    if hasattr(graph, "to_dict"):
        try:
            payload = graph.to_dict()
        except Exception:
            payload = {}
    if not payload:
        payload = {
            "nodes": sorted(list(getattr(graph, "nodes", {}) or {}))[:256],
            "edge_count": len(getattr(graph, "edges", []) or []),
        }
    return canonical_json_hash(payload, n=n)


def serialize_removed_edge(edge: Any) -> Dict[str, Any]:
    edge_type = getattr(getattr(edge, "edge_type", ""), "value", getattr(edge, "edge_type", ""))
    return {
        "source": str(getattr(edge, "source", "")),
        "target": str(getattr(edge, "target", "")),
        "edge_type": str(edge_type),
        "weight": getattr(edge, "weight", None),
        "metadata": dict(getattr(edge, "metadata", {}) or {}),
    }


def serialize_intervention(intervention: Any) -> Dict[str, Any]:
    intervention_type = getattr(getattr(intervention, "intervention_type", ""), "value", getattr(intervention, "intervention_type", ""))
    intervened_graph = getattr(intervention, "intervened_graph", None)
    return {
        "intervention_type": str(intervention_type),
        "removed_nodes": [str(x) for x in (getattr(intervention, "removed_nodes", []) or [])],
        "removed_edges": [serialize_removed_edge(edge) for edge in (getattr(intervention, "removed_edges", []) or [])],
        "modified_nodes": [str(x) for x in (getattr(intervention, "modified_nodes", []) or [])],
        "description": str(getattr(intervention, "description", "")),
        "post_graph_hash": graph_state_fingerprint(intervened_graph) if intervened_graph is not None else "",
    }


def observed_effect_dict(
    *,
    pre_denotation: Any,
    post_denotation: Any,
    changed: bool,
    available: bool,
    executor_name: str = "GraphAwareExecutor",
) -> Dict[str, Any]:
    return {
        "available": bool(available),
        "executor": executor_name,
        "pre_denotation": "" if pre_denotation is None else str(pre_denotation),
        "post_denotation": None if post_denotation is None else str(post_denotation),
        "changed": bool(changed),
    }


def expected_effect_for_intervention(intervention_type: str) -> str:
    if intervention_type == "benign_irrelevant":
        return "candidate should stay invariant if the removed node is irrelevant"
    if intervention_type in {"support_delete", "required_edge_delete", "binding_swap", "anchor_shift", "sibling_substitute", "operator_replace"}:
        return "candidate should change or lose support if the modified graph state is necessary"
    return "effect must be read from observed executor output"


def intervention_flags(intervention_type: str, observed_effect: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "is_benign": intervention_type == "benign_irrelevant",
        "is_adversarial": intervention_type != "benign_irrelevant",
        "observed_effect_available": bool(observed_effect.get("available")),
        "post_denotation_null": observed_effect.get("post_denotation") is None,
        "changed": bool(observed_effect.get("changed")),
    }
