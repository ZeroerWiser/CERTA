"""Candidate-specific intervention replay for CERA packets.

This module never recomputes the generic top answer. It asks whether the same
certified candidate keeps the support cells, operation family, graph path, and
answer projection it needs after a graph intervention.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

from eval_utils import normalize_text

from certa.evidence.counterfactuals import graph_state_fingerprint, serialize_intervention


def _as_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "to_dict"):
        try:
            return dict(value.to_dict())
        except Exception:
            return {}
    return {}


def _candidate_dict(candidate: Any) -> Dict[str, Any]:
    payload = _as_dict(candidate)
    if payload:
        return payload
    out: Dict[str, Any] = {}
    for key in ("candidate_id", "denotation", "operation", "cells_used", "certificate"):
        if hasattr(candidate, key):
            out[key] = getattr(candidate, key)
    return out


def _node_id_from_cell(cell: Mapping[str, Any]) -> str:
    row = cell.get("row")
    col = cell.get("col")
    return str(cell.get("node_id") or f"cell_{row}_{col}")


def _candidate_support_node_ids(candidate: Any, support_chain: Optional[Sequence[Any]] = None) -> List[str]:
    ids: List[str] = []
    for item in support_chain or []:
        payload = _as_dict(item)
        node_id = payload.get("node_id")
        if node_id:
            ids.append(str(node_id))
    if ids:
        return sorted(dict.fromkeys(ids))
    cand = _candidate_dict(candidate)
    for cell in cand.get("cells_used") or []:
        payload = _as_dict(cell)
        if payload:
            ids.append(_node_id_from_cell(payload))
    return sorted(dict.fromkeys(ids))


def _certificate_graph_path(candidate: Any) -> List[str]:
    cand = _candidate_dict(candidate)
    cert = _as_dict(cand.get("certificate"))
    path = cert.get("graph_path") or []
    if not isinstance(path, list):
        return []
    return [str(x) for x in path if str(x)]


def _candidate_operation(candidate: Any) -> str:
    cand = _candidate_dict(candidate)
    value = cand.get("operation")
    return str(getattr(value, "value", value or ""))


def _candidate_denotation(candidate: Any) -> str:
    cand = _candidate_dict(candidate)
    return str(cand.get("denotation", ""))


def _candidate_id(candidate: Any) -> str:
    cand = _candidate_dict(candidate)
    return str(cand.get("candidate_id", ""))


def _node_set(graph: Any) -> Set[str]:
    return {str(x) for x in (getattr(graph, "nodes", {}) or {}).keys()}


def _removed_support_nodes(support_nodes: Iterable[str], post_graph: Any) -> List[str]:
    post_nodes = _node_set(post_graph)
    return sorted(str(node_id) for node_id in support_nodes if str(node_id) not in post_nodes)


def _lost_graph_path_nodes(path_nodes: Iterable[str], post_graph: Any) -> List[str]:
    post_nodes = _node_set(post_graph)
    return sorted(str(node_id) for node_id in path_nodes if str(node_id) and str(node_id) not in post_nodes)


@dataclass
class CandidateInterventionEffect:
    intervention_id: str
    candidate_id: str
    intervention_type: str
    pre_graph_hash: str
    post_graph_hash: str
    replay_mode: str
    pre_denotation: str
    post_denotation: Optional[str]
    support_valid: bool
    changed: bool
    available: bool
    failure_reason: str = ""
    support_node_ids: List[str] = field(default_factory=list)
    removed_support_nodes: List[str] = field(default_factory=list)
    graph_path_node_ids: List[str] = field(default_factory=list)
    lost_graph_path_nodes: List[str] = field(default_factory=list)
    operation_family: str = ""
    answer_projection: str = "candidate_denotation"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_observed_effect(self) -> Dict[str, Any]:
        return {
            "available": self.available,
            "executor": "candidate_specific_intervention_replay",
            "replay_mode": self.replay_mode,
            "candidate_specific": True,
            "candidate_id": self.candidate_id,
            "intervention_id": self.intervention_id,
            "intervention_type": self.intervention_type,
            "pre_graph_hash": self.pre_graph_hash,
            "post_graph_hash": self.post_graph_hash,
            "pre_denotation": self.pre_denotation,
            "post_denotation": self.post_denotation,
            "support_valid": self.support_valid,
            "changed": self.changed,
            "failure_reason": self.failure_reason,
            "support_node_ids": self.support_node_ids,
            "removed_support_nodes": self.removed_support_nodes,
            "graph_path_node_ids": self.graph_path_node_ids,
            "lost_graph_path_nodes": self.lost_graph_path_nodes,
            "operation_family": self.operation_family,
            "answer_projection": self.answer_projection,
        }


def execute_candidate_under_intervention(
    *,
    intervention_id: str,
    candidate: Any,
    intervention: Any,
    original_graph: Any = None,
    support_chain: Optional[Sequence[Any]] = None,
) -> CandidateInterventionEffect:
    """Replay a candidate against an intervention without generic top-answer recomputation."""
    serialized = serialize_intervention(intervention)
    post_graph = getattr(intervention, "intervened_graph", None)
    pre_hash = graph_state_fingerprint(original_graph) if original_graph is not None else ""
    post_hash = serialized.get("post_graph_hash") or (graph_state_fingerprint(post_graph) if post_graph is not None else "")
    candidate_id = _candidate_id(candidate)
    support_nodes = _candidate_support_node_ids(candidate, support_chain=support_chain)
    path_nodes = _certificate_graph_path(candidate)
    pre_denotation = _candidate_denotation(candidate)
    operation_family = _candidate_operation(candidate)
    if post_graph is None:
        return CandidateInterventionEffect(
            intervention_id=intervention_id,
            candidate_id=candidate_id,
            intervention_type=str(serialized.get("intervention_type", "")),
            pre_graph_hash=pre_hash,
            post_graph_hash=post_hash,
            replay_mode="candidate_support_projection_replay_v1",
            pre_denotation=pre_denotation,
            post_denotation=None,
            support_valid=False,
            changed=False,
            available=False,
            failure_reason="missing_intervened_graph",
            support_node_ids=support_nodes,
            graph_path_node_ids=path_nodes,
            operation_family=operation_family,
        )
    if not support_nodes:
        return CandidateInterventionEffect(
            intervention_id=intervention_id,
            candidate_id=candidate_id,
            intervention_type=str(serialized.get("intervention_type", "")),
            pre_graph_hash=pre_hash,
            post_graph_hash=post_hash,
            replay_mode="candidate_support_projection_replay_v1",
            pre_denotation=pre_denotation,
            post_denotation=None,
            support_valid=False,
            changed=False,
            available=False,
            failure_reason="no_candidate_support_cells",
            support_node_ids=support_nodes,
            graph_path_node_ids=path_nodes,
            operation_family=operation_family,
        )
    removed_support = _removed_support_nodes(support_nodes, post_graph)
    lost_path = _lost_graph_path_nodes(path_nodes, post_graph)
    support_valid = not removed_support and not lost_path
    if support_valid:
        post_denotation: Optional[str] = pre_denotation
        changed = False
        failure_reason = ""
    else:
        post_denotation = None
        changed = bool(normalize_text(pre_denotation))
        failure_reason = "candidate_support_or_path_removed"
    return CandidateInterventionEffect(
        intervention_id=intervention_id,
        candidate_id=candidate_id,
        intervention_type=str(serialized.get("intervention_type", "")),
        pre_graph_hash=pre_hash,
        post_graph_hash=post_hash,
        replay_mode="candidate_support_projection_replay_v1",
        pre_denotation=pre_denotation,
        post_denotation=post_denotation,
        support_valid=support_valid,
        changed=changed,
        available=True,
        failure_reason=failure_reason,
        support_node_ids=support_nodes,
        removed_support_nodes=removed_support,
        graph_path_node_ids=path_nodes,
        lost_graph_path_nodes=lost_path,
        operation_family=operation_family,
    )
