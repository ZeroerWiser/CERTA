"""Original-answer support hypothesis reconstruction."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Set

from .project import answers_equivalent
from .schema import (
    ExecutableDerivation,
    OriginalSupportHypothesis,
    OriginalSupportHypothesisSet,
)


def _node_text(node: Any) -> str:
    text = str(getattr(node, "text", "") or "")
    if text:
        return text
    numeric = getattr(node, "numeric_value", None)
    return "" if numeric is None else str(numeric)


def _graph_answer_nodes(graph: Any, original_answer: str, evidence: Any = None, max_nodes: int = 8) -> List[str]:
    if graph is None:
        return []
    evidence_nodes: Optional[Set[str]] = None
    if evidence is not None:
        evidence_nodes = set(getattr(evidence, "evidence_nodes", set()) or set())
    matches: List[str] = []
    for node_id, node in sorted((getattr(graph, "nodes", {}) or {}).items()):
        if evidence_nodes is not None and evidence_nodes and node_id not in evidence_nodes:
            continue
        if answers_equivalent(_node_text(node), original_answer):
            matches.append(str(node_id))
        if len(matches) >= max_nodes:
            break
    return matches


def reconstruct_original_support_hypotheses(
    *,
    original_answer: str,
    derivations: Sequence[ExecutableDerivation],
    graph: Any = None,
    evidence: Any = None,
    max_graph_anchor_nodes: int = 8,
) -> OriginalSupportHypothesisSet:
    hypotheses: List[OriginalSupportHypothesis] = []
    for idx, derivation in enumerate(derivations, start=1):
        if not answers_equivalent(derivation.projected_answer, original_answer):
            continue
        source = (derivation.source_candidate or {}).get("source") if isinstance(derivation.source_candidate, dict) else ""
        support_level = (
            "EXECUTABLE_RECONSTRUCTED"
            if source == "symmetric_derivation_frontier_v1" or str(derivation.source_candidate_id).startswith("frontier_")
            else "EXECUTABLE_EXISTING"
        )
        hypotheses.append(OriginalSupportHypothesis(
            hypothesis_id=f"OH{len(hypotheses) + 1}",
            source="projected_answer_equivalent_executable_derivation",
            derivation_id=derivation.derivation_id,
            projected_answer=derivation.projected_answer,
            answer_equivalent=True,
            operation_family=derivation.operation_family,
            operand_node_ids=list(derivation.operand_node_ids),
            evidence_ids=list(derivation.evidence_ids or derivation.operand_node_ids),
            required_edge_triples=list(derivation.required_edge_triples),
            projection_operator=derivation.projection_operator,
            provenance_complete=derivation.provenance_complete,
            availability="executable" if derivation.provenance_complete else "unavailable",
            failure_reasons=list(derivation.failure_reasons),
            support_level=support_level if derivation.provenance_complete else "UNAVAILABLE",
        ))

    contains_executable = any(h.availability == "executable" for h in hypotheses)
    contains_graph_anchor_only = False
    notes: List[str] = []
    if not hypotheses:
        node_ids = _graph_answer_nodes(
            graph,
            original_answer,
            evidence=evidence,
            max_nodes=max_graph_anchor_nodes,
        )
        for node_id in node_ids:
            hypotheses.append(OriginalSupportHypothesis(
                hypothesis_id=f"OH{len(hypotheses) + 1}",
                source="answer_anchored_graph_node",
                derivation_id=None,
                projected_answer=original_answer,
                answer_equivalent=True,
                operation_family="UNKNOWN",
                operand_node_ids=[node_id],
                evidence_ids=[node_id],
                required_edge_triples=[],
                projection_operator="UNKNOWN",
                provenance_complete=False,
                availability="graph_anchor_only",
                failure_reasons=["not_executable_derivation"],
                support_level="GRAPH_ANCHORED",
            ))
        contains_graph_anchor_only = bool(node_ids)
        if not node_ids:
            notes.append("no_original_support_hypothesis_available")
    ambiguity = max(0, len(hypotheses) - 1)
    return OriginalSupportHypothesisSet(
        original_answer=str(original_answer or ""),
        hypotheses=hypotheses,
        ambiguity_count=ambiguity,
        contains_executable_derivation=contains_executable,
        contains_graph_anchor_only=contains_graph_anchor_only,
        notes=notes,
    )
