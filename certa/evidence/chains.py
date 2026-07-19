"""Construct support and counterfactual chains for CERA evidence packets."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from eval_utils import normalize_text
from graph_builder import EdgeType, HCEG, NodeType
from structural_cert_utils import generate_candidate_targeted_interventions

from certa.derivations import (
    build_admissible_candidate_set,
    inference_answers_equivalent,
    materialize_derivations,
    reconstruct_original_support_hypotheses,
    replay_derivation_under_intervention,
)
from certa.derivations.schema import (
    AdmissibleCandidateSet,
    ExecutableDerivation,
    OriginalSupportHypothesisSet,
    PreEvidenceQueryContract,
)
from certa.evidence.candidate_intervention import execute_candidate_under_intervention
from certa.evidence.counterfactuals import (
    expected_effect_for_intervention,
    intervention_flags,
    serialize_intervention,
)
from certa.repair.evidence_packet import (
    CausalEvidencePacket,
    CertifiedCandidateFull,
    CounterfactualChainElement,
    EvidenceState,
    SupportChainElement,
)
from certa.repair.evidence_semanticizer import semanticize_packet
from certa.reproducibility.canonical_json import canonical_json_hash
from certa.semantics.query_contract import build_pre_evidence_query_contract, query_contract_hash


CRITICAL_PATH_EDGES = {
    EdgeType.ENTITY_MENTION,
    EdgeType.VALUE_UNDER_HEADER,
    EdgeType.ROW_PATH,
    EdgeType.COL_PATH,
    EdgeType.AGGREGATE_DEPENDS,
    EdgeType.COMPARISON_BETWEEN,
    EdgeType.PART_OF,
}


def stable_packet_hash(value: Any, n: int = 16) -> str:
    return canonical_json_hash(value, n=n)


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _serialize_node(node_id: str, node: Any) -> Dict[str, Any]:
    return {
        "node_id": node_id,
        "node_type": _enum_value(getattr(node, "node_type", "")),
        "row": getattr(node, "row", None),
        "col": getattr(node, "col", None),
        "text": str(getattr(node, "text", ""))[:160],
        "numeric_value": getattr(node, "numeric_value", None),
        "header_level": getattr(node, "header_level", None),
        "aggregation_type": str(getattr(node, "aggregation_type", "")),
        "metadata": dict(getattr(node, "metadata", {}) or {}),
    }


def _serialize_edge(edge: Any) -> Dict[str, Any]:
    return {
        "source": str(getattr(edge, "source", "")),
        "target": str(getattr(edge, "target", "")),
        "edge_type": _enum_value(getattr(edge, "edge_type", "")),
        "weight": getattr(edge, "weight", None),
        "metadata": dict(getattr(edge, "metadata", {}) or {}),
    }


def _candidate_from_payload(candidate: Any) -> CertifiedCandidateFull:
    if isinstance(candidate, CertifiedCandidateFull):
        return candidate
    if isinstance(candidate, Mapping):
        return CertifiedCandidateFull.from_dict(candidate)
    return CertifiedCandidateFull(
        candidate_id="",
        denotation=str(getattr(candidate, "denotation", "")),
        normalized_denotation=normalize_text(getattr(candidate, "denotation", "")),
        operation=_enum_value(getattr(candidate, "operation", "")),
        priority=getattr(candidate, "priority", None),
        cells_used=[
            {
                "row": getattr(ref, "row", None),
                "col": getattr(ref, "col", None),
                "value": getattr(ref, "value", ""),
                "row_headers": list(getattr(ref, "row_headers", []) or []),
                "col_headers": list(getattr(ref, "col_headers", []) or []),
            }
            for ref in (getattr(candidate, "cells_used", []) or [])
        ],
        computation_trace=str(getattr(candidate, "computation_trace", "")),
        operation_metadata=dict(getattr(candidate, "operation_metadata", {}) or {}),
        source="executor_live",
        certificate={},
    )


def _cell_node_id(row: Any, col: Any) -> str:
    return f"cell_{row}_{col}"


def _edge_path_for_node(graph: Optional[HCEG], node_id: str, max_edges: int = 8) -> List[Dict[str, Any]]:
    if graph is None or node_id not in getattr(graph, "nodes", {}):
        return []
    edges: List[Any] = []
    for _src, edge in graph.predecessors(node_id, CRITICAL_PATH_EDGES):
        edges.append(edge)
    for _tgt, edge in graph.neighbors(node_id, CRITICAL_PATH_EDGES):
        edges.append(edge)
    return [_serialize_edge(edge) for edge in edges[:max_edges]]


def build_evidence_state(
    *,
    question: str,
    graph: Optional[HCEG] = None,
    evidence: Any = None,
    cert_info: Optional[Mapping[str, Any]] = None,
    question_frame: Optional[Mapping[str, Any]] = None,
    graph_stats: Optional[Mapping[str, Any]] = None,
    edge_reliability_diag: Optional[Mapping[str, Any]] = None,
    layout_risk: float = 0.0,
    max_nodes: int = 64,
    max_edges: int = 96,
) -> EvidenceState:
    evidence_nodes = sorted(list(getattr(evidence, "evidence_nodes", set()) or []))
    anchor_nodes = [str(x) for x in (getattr(evidence, "anchor_nodes", []) or [])]
    graph_obj = graph or getattr(evidence, "graph", None)
    serialized_nodes: List[Dict[str, Any]] = []
    serialized_edges: List[Dict[str, Any]] = []
    if graph_obj is not None:
        for node_id in evidence_nodes[:max_nodes]:
            node = graph_obj.nodes.get(node_id)
            if node is not None:
                serialized_nodes.append(_serialize_node(node_id, node))
        for edge in (getattr(evidence, "evidence_edges", []) or [])[:max_edges]:
            serialized_edges.append(_serialize_edge(edge))
    overview = {
        "num_candidates": (cert_info or {}).get("num_candidates", 0),
        "scci_mode": (cert_info or {}).get("scci_mode", ""),
        "dominance_source": (cert_info or {}).get("dominance_source", ""),
    }
    return EvidenceState(
        question=question,
        graph_stats=dict(graph_stats or (graph_obj.stats() if graph_obj is not None else {})),
        question_frame=dict(question_frame or {}),
        edge_reliability_diag=dict(edge_reliability_diag or {}),
        layout_risk=_safe_float(layout_risk, 0.0),
        evidence_ids=evidence_nodes,
        anchor_nodes=anchor_nodes,
        evidence_nodes=serialized_nodes,
        evidence_edges=serialized_edges,
        certificate_overview=overview,
    )


def build_support_chain(
    candidate: Any,
    *,
    graph: Optional[HCEG] = None,
    evidence: Any = None,
    max_items: int = 16,
    evidence_prefix: str = "S",
) -> List[SupportChainElement]:
    full = _candidate_from_payload(candidate)
    evidence_node_ids = set(getattr(evidence, "evidence_nodes", set()) or set())
    cert = dict(full.certificate or {})
    cells = list(full.cells_used or [])
    support: List[SupportChainElement] = []
    for idx, cell in enumerate(cells[:max_items], start=1):
        row = cell.get("row")
        col = cell.get("col")
        node_id = str(cell.get("node_id") or _cell_node_id(row, col))
        provenance = "executor_cell"
        if evidence is not None and node_id not in evidence_node_ids:
            provenance = "executor_cell_not_in_evidence_subgraph"
        support.append(SupportChainElement(
            evidence_id=f"{evidence_prefix}{idx}",
            node_id=node_id,
            row=row,
            col=col,
            cell_value=str(cell.get("value", cell.get("cell_value", ""))),
            row_headers=[str(x) for x in (cell.get("row_headers") or [])],
            col_headers=[str(x) for x in (cell.get("col_headers") or [])],
            support_role="candidate_cell",
            provenance=provenance,
            edge_path=_edge_path_for_node(graph, node_id),
            certificate_link={
                "candidate_id": full.candidate_id,
                "operation": full.operation,
                "path_verified": bool(cert.get("path_verified", False)),
                "evidence_fallback": bool(cert.get("evidence_fallback", False)),
                "candidate_evidence_coverage": cert.get("candidate_evidence_coverage"),
                "candidate_effective_evidence_coverage": cert.get("candidate_effective_evidence_coverage"),
                "scci": cert.get("scci"),
                "bir": cert.get("bir"),
                "asr": cert.get("asr"),
            },
        ))
    return support


def _table_texts(table_json: Optional[Mapping[str, Any]]) -> List[List[Any]]:
    if not table_json:
        return []
    texts = table_json.get("texts") or table_json.get("table_array") or []
    return texts if isinstance(texts, list) else []


def _cell_excerpt(row: int, col: int, value: Any, *, source: str, evidence_id: str = "") -> Dict[str, Any]:
    return {
        "row": row,
        "col": col,
        "value": str(value),
        "source": source,
        "evidence_id": evidence_id,
    }


def build_relevant_table_excerpt(
    table_json: Optional[Mapping[str, Any]],
    *,
    evidence: Any = None,
    support_chain: Optional[Sequence[SupportChainElement]] = None,
    max_cells: int = 32,
    allow_row_major_context: bool = False,
) -> List[Dict[str, Any]]:
    texts = _table_texts(table_json)
    excerpt: List[Dict[str, Any]] = []
    seen: set[Tuple[int, int]] = set()

    for item in support_chain or []:
        if item.row is None or item.col is None:
            continue
        key = (int(item.row), int(item.col))
        if key in seen:
            continue
        seen.add(key)
        excerpt.append(_cell_excerpt(key[0], key[1], item.cell_value, source="support_chain", evidence_id=item.evidence_id))
        if len(excerpt) >= max_cells:
            return excerpt

    graph = getattr(evidence, "graph", None)
    for node_id in sorted(list(getattr(evidence, "evidence_nodes", set()) or [])):
        node = getattr(graph, "nodes", {}).get(node_id) if graph is not None else None
        if node is None:
            continue
        if getattr(node, "node_type", None) not in {NodeType.CELL, NodeType.AGGREGATOR, NodeType.HEADER}:
            continue
        row = getattr(node, "row", -1)
        col = getattr(node, "col", -1)
        if row is None or col is None or row < 0 or col < 0:
            continue
        key = (int(row), int(col))
        if key in seen:
            continue
        seen.add(key)
        excerpt.append(_cell_excerpt(key[0], key[1], getattr(node, "text", ""), source="evidence_subgraph"))
        if len(excerpt) >= max_cells:
            return excerpt

    if allow_row_major_context:
        for r, row_values in enumerate(texts):
            if not isinstance(row_values, list):
                continue
            for c, value in enumerate(row_values):
                key = (r, c)
                if key in seen:
                    continue
                seen.add(key)
                excerpt.append(_cell_excerpt(r, c, value, source="row_major_context"))
                if len(excerpt) >= max_cells:
                    return excerpt
    return excerpt


def _candidate_rows(cert_info: Optional[Mapping[str, Any]]) -> List[CertifiedCandidateFull]:
    rows = (cert_info or {}).get("certified_candidates_full") or []
    out: List[CertifiedCandidateFull] = []
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, Mapping):
                out.append(CertifiedCandidateFull.from_dict(row))
    return out


def find_original_equivalent_candidate(
    cert_info: Optional[Mapping[str, Any]],
    original_answer: str,
) -> Optional[CertifiedCandidateFull]:
    if not str(original_answer or "").strip():
        return None
    for candidate in _candidate_rows(cert_info):
        candidate_answer = candidate.denotation or candidate.normalized_denotation
        if inference_answers_equivalent(candidate_answer, original_answer):
            return candidate
    return None


def build_counterfactual_chain(
    candidate: Any,
    *,
    exec_candidate: Any = None,
    derivation: Optional[ExecutableDerivation] = None,
    graph: Optional[HCEG] = None,
    evidence: Any = None,
    question: str = "",
    support_chain: Optional[Sequence[SupportChainElement]] = None,
    max_items: int = 8,
) -> List[CounterfactualChainElement]:
    if graph is None:
        return []
    full = _candidate_from_payload(candidate)
    intervention_candidate = exec_candidate if exec_candidate is not None else full
    try:
        interventions = generate_candidate_targeted_interventions(
            graph=graph,
            evidence=evidence,
            candidate=intervention_candidate,
            derivation=derivation,
            max_benign=3,
        )
    except Exception:
        return []
    chain: List[CounterfactualChainElement] = []
    for idx, intervention in enumerate(interventions[:max_items], start=1):
        cf_id = f"CF{idx}"
        serialized = serialize_intervention(intervention)
        intervention_type = serialized.get("intervention_type", "")
        if derivation is not None:
            effect = replay_derivation_under_intervention(
                intervention_id=cf_id,
                derivation=derivation,
                intervention=intervention,
            )
            observed = effect.to_observed_effect()
            if effect.changed:
                interpretation = "same executable derivation changed or became invalid under the graph intervention"
            else:
                interpretation = "same executable derivation stayed invariant under the graph intervention"
            support_valid = effect.required_nodes_valid and effect.required_edges_valid
            failure_reason = effect.failure_reason
        else:
            legacy_effect = execute_candidate_under_intervention(
                intervention_id=cf_id,
                candidate=full,
                intervention=intervention,
                original_graph=graph,
                support_chain=support_chain,
            )
            observed = legacy_effect.to_observed_effect()
            if legacy_effect.available and legacy_effect.changed:
                interpretation = "same candidate lost support or changed under the graph intervention"
            elif legacy_effect.available:
                interpretation = "same candidate support stayed invariant under the graph intervention"
            else:
                interpretation = "candidate-specific intervention effect was not available"
            support_valid = legacy_effect.support_valid
            failure_reason = legacy_effect.failure_reason
        flags = intervention_flags(intervention_type, observed)
        flags.update({
            "candidate_specific": True,
            "support_valid": support_valid,
            "failure_reason": failure_reason,
        })
        chain.append(CounterfactualChainElement(
            cf_id=cf_id,
            intervention_type=intervention_type,
            removed_nodes=serialized.get("removed_nodes", []),
            removed_edges=serialized.get("removed_edges", []),
            modified_nodes=serialized.get("modified_nodes", []),
            expected_effect=expected_effect_for_intervention(intervention_type),
            observed_effect=observed,
            causal_interpretation=interpretation,
            flags=flags,
            description=serialized.get("description", ""),
        ))
    return chain


def _compact_admissible_candidate_set(admissible_set: AdmissibleCandidateSet) -> Dict[str, Any]:
    return {
        "derivation_count": len(admissible_set.derivations),
        "admissible_derivation_count": len(admissible_set.admissible_derivations),
        "projected_answer_classes": dict(admissible_set.projected_answer_classes),
        "review_eligible": admissible_set.review_eligible,
        "selected_derivation_id": admissible_set.selected_derivation_id,
        "selected_candidate_id": admissible_set.selected_candidate_id,
        "ambiguity_count": admissible_set.ambiguity_count,
        "reject_reason": admissible_set.reject_reason,
        "notes": list(admissible_set.notes),
    }


def build_causal_evidence_packet(
    *,
    question: str,
    original_answer: str,
    candidate: Any,
    graph: Optional[HCEG] = None,
    evidence: Any = None,
    table_json: Optional[Mapping[str, Any]] = None,
    cert_info: Optional[Mapping[str, Any]] = None,
    exec_candidate: Any = None,
    pre_evidence_contract: Optional[PreEvidenceQueryContract] = None,
    admissible_candidate_set: Optional[AdmissibleCandidateSet] = None,
    reviewed_derivation: Optional[ExecutableDerivation] = None,
    original_support_hypothesis_set: Optional[OriginalSupportHypothesisSet] = None,
    derivation_lattice: Optional[Mapping[str, Any]] = None,
    minimal_contrast_set: Optional[Mapping[str, Any]] = None,
    compact_behavioral_contrast_v2: Optional[Mapping[str, Any]] = None,
    compact_behavioral_contrast_v3: Optional[Mapping[str, Any]] = None,
    original_support_symmetry_v3: Optional[Mapping[str, Any]] = None,
    question_frame: Optional[Mapping[str, Any]] = None,
    graph_stats: Optional[Mapping[str, Any]] = None,
    edge_reliability_diag: Optional[Mapping[str, Any]] = None,
    layout_risk: float = 0.0,
    max_excerpt_cells: int = 32,
    allow_row_major_context: bool = False,
) -> CausalEvidencePacket:
    full = _candidate_from_payload(candidate)
    pre_contract = pre_evidence_contract or build_pre_evidence_query_contract(
        question=question,
        question_frame=question_frame,
        initial_answer=original_answer,
        graph_stats=graph_stats,
    )
    if admissible_candidate_set is None:
        derivations = materialize_derivations(
            certified_candidates=(cert_info or {}).get("certified_candidates_full") or [full.to_dict()],
            live_candidates=[exec_candidate] if exec_candidate is not None else None,
            graph=graph,
        )
        admissible_candidate_set = build_admissible_candidate_set(
            contract=pre_contract,
            derivations=derivations,
            graph=graph,
            evidence=evidence,
        )
    if reviewed_derivation is None and admissible_candidate_set.selected_derivation_id:
        reviewed_derivation = next(
            (
                item for item in admissible_candidate_set.admissible_derivations
                if item.derivation_id == admissible_candidate_set.selected_derivation_id
            ),
            None,
        )
    support_chain = build_support_chain(full, graph=graph, evidence=evidence)
    if reviewed_derivation is not None:
        reviewed_derivation.evidence_ids = [item.evidence_id for item in support_chain]
    original_candidate = find_original_equivalent_candidate(cert_info, original_answer)
    original_support_chain = (
        build_support_chain(original_candidate, graph=graph, evidence=evidence, evidence_prefix="OS")
        if original_candidate is not None else []
    )
    original_notes = []
    if original_candidate is None:
        original_notes.append("no_equivalent_candidate_certificate_for_original_answer")
    elif not original_support_chain:
        original_notes.append("original_equivalent_candidate_has_no_support_chain")
    counterfactual_chain = build_counterfactual_chain(
        full,
        exec_candidate=exec_candidate,
        derivation=reviewed_derivation,
        graph=graph,
        evidence=evidence,
        question=question,
        support_chain=support_chain,
    )
    table_excerpt = build_relevant_table_excerpt(
        table_json,
        evidence=evidence,
        support_chain=support_chain,
        max_cells=max_excerpt_cells,
        allow_row_major_context=allow_row_major_context,
    )
    evidence_state = build_evidence_state(
        question=question,
        graph=graph,
        evidence=evidence,
        cert_info=cert_info,
        question_frame=question_frame,
        graph_stats=graph_stats,
        edge_reliability_diag=edge_reliability_diag,
        layout_risk=layout_risk,
    )
    if original_support_hypothesis_set is None:
        original_support_hypothesis_set = reconstruct_original_support_hypotheses(
            original_answer=original_answer,
            derivations=admissible_candidate_set.derivations,
            graph=graph,
            evidence=evidence,
        )
    packet = CausalEvidencePacket(
        question=question,
        original_answer=str(original_answer or ""),
        candidate_under_review=full.denotation,
        candidate=full,
        evidence_state=evidence_state,
        query_contract=pre_contract.to_dict(),
        original_certificate_available=bool(
            original_support_hypothesis_set.contains_executable_derivation
            or original_support_hypothesis_set.contains_graph_anchor_only
            or original_support_chain
        ),
        original_equivalent_candidate_id=original_candidate.candidate_id if original_candidate is not None else "",
        original_certificate=original_candidate.certificate if original_candidate is not None else {},
        original_support_chain=original_support_chain,
        original_support_chain_notes=original_notes,
        original_support_hypothesis_set=original_support_hypothesis_set.to_dict(),
        support_chain=support_chain,
        counterfactual_chain=counterfactual_chain,
        admissible_candidate_set=_compact_admissible_candidate_set(admissible_candidate_set),
        reviewed_derivation=reviewed_derivation.to_dict() if reviewed_derivation is not None else {},
        derivation_lattice=dict(derivation_lattice or {}),
        minimal_contrast_set=dict(minimal_contrast_set or {}),
        compact_behavioral_contrast_v2=dict(compact_behavioral_contrast_v2 or {}),
        compact_behavioral_contrast_v3=dict(compact_behavioral_contrast_v3 or {}),
        original_support_symmetry_v3=dict(original_support_symmetry_v3 or {}),
        table_excerpt=table_excerpt,
        metadata={
            "packet_hash_algorithm": "sha256_16_sorted_json",
            "contains_gold_answer": False,
            "max_excerpt_cells": max_excerpt_cells,
            "allow_row_major_context": bool(allow_row_major_context),
            "row_major_context_cell_count": sum(1 for item in table_excerpt if item.get("source") == "row_major_context"),
            "query_contract_hash": query_contract_hash(pre_contract),
            "original_certificate_available": bool(
                original_support_hypothesis_set.contains_executable_derivation
                or original_support_hypothesis_set.contains_graph_anchor_only
                or original_support_chain
            ),
            "reviewed_derivation_id": reviewed_derivation.derivation_id if reviewed_derivation is not None else "",
            "reviewed_derivation_replay_mode": "typed_same_derivation_replay_v1" if reviewed_derivation is not None else "",
            "admissible_derivation_count": len(admissible_candidate_set.admissible_derivations),
            "projected_answer_class_count": len(admissible_candidate_set.projected_answer_classes),
            "round6_lattice_present": bool(derivation_lattice),
            "round6_minimal_contrast_present": bool(minimal_contrast_set),
            "round6_original_support_symmetry_v3_present": bool(original_support_symmetry_v3),
            "round7_compact_behavioral_contrast_v2_present": bool(compact_behavioral_contrast_v2),
            "round8_compact_behavioral_contrast_v3_present": bool(compact_behavioral_contrast_v3),
        },
    )
    packet.evidence_semantic_statements = semanticize_packet(packet)
    packet.metadata["packet_hash"] = stable_packet_hash(packet)
    return packet
