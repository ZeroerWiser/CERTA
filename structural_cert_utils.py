"""Lightweight structural-certification utilities for CSCR v10.

The utilities are intentionally small: no training, no new model calls, and no
new heavyweight graph library. They turn HCEG rules into auditable structural
priors and make candidate-SCCI reliable under no-anchor fallback.
"""

from __future__ import annotations

import copy
import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

from graph_builder import HCEG, EdgeType, GraphEdge, NodeType
from evidence_retriever import EvidenceSubgraph, InterventionResult, InterventionType

_STOPWORDS = {
    "what", "which", "how", "many", "much", "is", "are", "was", "were",
    "the", "a", "an", "of", "in", "for", "to", "and", "or", "by", "with",
    "from", "at", "on", "as", "than", "between", "among", "that", "this",
}


def parse_question_frame(question: str) -> Dict[str, Any]:
    q = (question or "").strip()
    ql = q.lower()
    operator = "lookup"
    if re.search(r"\b(total|sum|altogether|combined|合计|总共|总计)\b", ql):
        operator = "sum"
    elif re.search(r"\b(average|mean|avg|平均)\b", ql):
        operator = "average"
    elif re.search(r"\b(more|less|higher|lower|larger|smaller|greater|最大|最小|更高|更低)\b", ql):
        operator = "compare"
    elif re.search(r"\b(difference|increase|decrease|change|gap|差|变化)\b", ql):
        operator = "difference"
    elif re.search(r"\b(percent|percentage|ratio|share|proportion|占比|比例)\b", ql):
        operator = "ratio"
    elif re.search(r"\b(how many|number of|count|多少个|数量)\b", ql):
        operator = "count"

    polarity = "neutral"
    if re.search(r"\b(max|highest|largest|most|maximum|最大|最高|最多)\b", ql):
        polarity = "max"
    elif re.search(r"\b(min|lowest|smallest|least|minimum|最小|最低|最少)\b", ql):
        polarity = "min"

    keywords = [
        t for t in re.findall(r"[\w\u4e00-\u9fff]+", ql)
        if t not in _STOPWORDS and len(t) > 1
    ]
    return {
        "operator": operator,
        "polarity": polarity,
        "numeric_demand": operator in {"sum", "average", "compare", "difference", "ratio", "count"},
        "keywords": keywords[:24],
    }


def layout_assumption_risk(table_json: Dict[str, Any]) -> Dict[str, Any]:
    texts = table_json.get("texts") or table_json.get("table_array") or []
    n_rows = len(texts)
    n_cols = max((len(r) for r in texts), default=0)
    flags: List[str] = []
    if n_rows == 0 or n_cols == 0:
        return {"risk": 1.0, "flags": ["empty_or_unreadable_table"]}
    row_lens = [len(r) for r in texts]
    if len(set(row_lens)) > 1:
        flags.append("ragged_rows")
    total_slots = max(n_rows * n_cols, 1)
    blank_ratio = sum(1 for row in texts for x in row if not str(x).strip()) / total_slots
    if blank_ratio > 0.25:
        flags.append("sparse_or_blank_inheritance_required")
    top_headers = int(table_json.get("top_header_rows_num", table_json.get("top_header_rows", 1)) or 1)
    left_headers = int(table_json.get("left_header_columns_num", table_json.get("left_header_cols", 1)) or 1)
    if top_headers == 0 and left_headers == 0:
        flags.append("no_explicit_header_region")
    if top_headers > 2 or left_headers > 2:
        flags.append("deep_header_hierarchy")
    norm_rows = ["|".join(str(x).strip().lower() for x in row) for row in texts]
    repeated_rows = sum(c - 1 for c in Counter(norm_rows).values() if c > 1)
    if repeated_rows > 0:
        flags.append("repeated_rows_or_section_headers")
    merged = table_json.get("merged_regions") or []
    if blank_ratio > 0.15 and not merged:
        flags.append("missing_merge_metadata_suspected")
    risk = min(1.0, 0.12 + 0.16 * len(flags))
    return {
        "risk": round(risk, 4),
        "flags": flags,
        "blank_ratio": round(blank_ratio, 4),
        "top_header_rows": top_headers,
        "left_header_cols": left_headers,
    }


def annotate_edge_reliability(
    graph: HCEG,
    table_json: Dict[str, Any],
    question_frame: Optional[Dict[str, Any]] = None,
    apply_to_weight: bool = False,
    min_weight: float = 0.05,
) -> Dict[str, Any]:
    qf = question_frame or {}
    layout = layout_assumption_risk(table_json)
    layout_penalty = 1.0 - 0.45 * float(layout["risk"])
    base = {
        EdgeType.ENTITY_MENTION: 0.72,
        EdgeType.VALUE_UNDER_HEADER: 0.72,
        EdgeType.ROW_PATH: 0.68,
        EdgeType.COL_PATH: 0.68,
        EdgeType.AGGREGATE_DEPENDS: 0.58,
        EdgeType.PARENT_HEADER: 0.74,
        EdgeType.CHILD_HEADER: 0.74,
        EdgeType.HEADER_OF: 0.70,
        EdgeType.SPAN_OF: 0.70,
        EdgeType.MERGED_INTO: 0.70,
        EdgeType.PART_OF: 0.86,
        EdgeType.OP_DEMAND: 0.64,
        EdgeType.CONSTRAINT_TARGET: 0.62,
    }
    provenance = {
        EdgeType.ENTITY_MENTION: "lexical_schema_linking",
        EdgeType.VALUE_UNDER_HEADER: "layout_binding_prior",
        EdgeType.ROW_PATH: "row_header_traversal_prior",
        EdgeType.COL_PATH: "column_header_traversal_prior",
        EdgeType.AGGREGATE_DEPENDS: "aggregation_keyword_prior",
        EdgeType.PARENT_HEADER: "header_tree_prior",
        EdgeType.CHILD_HEADER: "header_tree_prior",
        EdgeType.HEADER_OF: "header_region_prior",
        EdgeType.SPAN_OF: "merge_metadata_prior",
        EdgeType.MERGED_INTO: "merge_metadata_prior",
        EdgeType.PART_OF: "value_extraction_prior",
        EdgeType.OP_DEMAND: "linguistic_operator_prior",
        EdgeType.CONSTRAINT_TARGET: "question_constraint_prior",
    }
    reliabilities: List[float] = []
    for e in graph.edges:
        r = base.get(e.edge_type, 0.55)
        if e.edge_type in {EdgeType.ROW_PATH, EdgeType.COL_PATH, EdgeType.VALUE_UNDER_HEADER, EdgeType.AGGREGATE_DEPENDS}:
            r *= layout_penalty
        if e.edge_type == EdgeType.AGGREGATE_DEPENDS and qf.get("operator") not in {"sum", "average", "ratio", "count", "lookup"}:
            r *= 0.85
        if e.edge_type == EdgeType.ENTITY_MENTION:
            r = 0.5 * r + 0.5 * max(0.0, min(1.0, float(e.weight)))
        e.metadata["provenance"] = provenance.get(e.edge_type, "structural_prior")
        e.metadata["reliability"] = round(r, 4)
        e.metadata["layout_risk"] = layout["risk"]
        if apply_to_weight:
            e.weight = max(min_weight, e.weight * r)
        reliabilities.append(r)
    return {
        "enabled": True,
        "applied_to_weight": bool(apply_to_weight),
        "layout_risk": layout["risk"],
        "layout_flags": layout["flags"],
        "layout_blank_ratio": layout.get("blank_ratio", 0.0),
        "mean_edge_reliability": round(sum(reliabilities) / max(len(reliabilities), 1), 4),
        "min_edge_reliability": round(min(reliabilities), 4) if reliabilities else 0.0,
    }


def evidence_is_fallback(evidence: Optional[EvidenceSubgraph]) -> bool:
    if evidence is None:
        return True
    metadata = getattr(evidence, "metadata", {}) or {}
    return bool(metadata.get("fallback")) or len(getattr(evidence, "anchor_nodes", []) or []) == 0


def evidence_ib_mdl_score(
    evidence: EvidenceSubgraph,
    lambda_complexity: float = 0.025,
    lambda_risk: float = 0.20,
) -> Tuple[float, Dict[str, Any]]:
    edges = list(getattr(evidence, "evidence_edges", []) or [])
    nodes: Set[str] = set(getattr(evidence, "evidence_nodes", set()) or set())
    anchors = list(getattr(evidence, "anchor_nodes", []) or [])
    if not nodes:
        return -1e9, {"error": "empty_evidence", "fallback": True}
    fallback = evidence_is_fallback(evidence)
    edge_relevance = sum(float(e.metadata.get("reliability", e.weight)) for e in edges) / max(len(edges), 1)
    anchor_cover = 0.0 if fallback else min(1.0, len(anchors) / 3.0)
    complexity = math.log1p(len(nodes)) + 0.5 * math.log1p(len(edges))
    risk = sum(1.0 - float(e.metadata.get("reliability", 0.55)) for e in edges) / max(len(edges), 1)
    fallback_penalty = 0.25 if fallback else 0.0
    score = 0.55 * edge_relevance + 0.35 * anchor_cover - lambda_complexity * complexity - lambda_risk * risk - fallback_penalty
    diag = {
        "score": round(score, 4),
        "edge_relevance": round(edge_relevance, 4),
        "anchor_cover": round(anchor_cover, 4),
        "complexity": round(complexity, 4),
        "risk": round(risk, 4),
        "fallback": bool(fallback),
        "fallback_penalty": fallback_penalty,
        "num_nodes": len(nodes),
        "num_edges": len(edges),
    }
    return score, diag


def cellref_node_ids(candidate: Any) -> Set[str]:
    ids: Set[str] = set()
    for ref in getattr(candidate, "cells_used", []) or []:
        row = getattr(ref, "row", -1)
        col = getattr(ref, "col", -1)
        if row is not None and col is not None and row >= 0 and col >= 0:
            ids.add(f"cell_{row}_{col}")
    return ids


def candidate_evidence_alignment(candidate: Any, evidence: Optional[EvidenceSubgraph]) -> Dict[str, Any]:
    used = cellref_node_ids(candidate)
    ev = set(getattr(evidence, "evidence_nodes", set()) or set()) if evidence else set()
    covered = used & ev
    raw_coverage = len(covered) / max(len(used), 1)
    fallback = evidence_is_fallback(evidence)
    effective_coverage = 0.0 if fallback else raw_coverage
    return {
        "candidate_cells": sorted(used),
        "covered_cells": sorted(covered),
        "coverage": round(raw_coverage, 4),
        "effective_coverage": round(effective_coverage, 4),
        "missing_cells": sorted(used - ev),
        "evidence_fallback": bool(fallback),
        "path_complete_by_cells": bool(used) and effective_coverage == 1.0,
    }


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value or ""))


def _edge_type_from_value(value: Any) -> Optional[EdgeType]:
    text = _enum_value(value)
    for edge_type in EdgeType:
        if edge_type.value == text or edge_type.name == text:
            return edge_type
    return None


def _edge_from_triple(graph: HCEG, triple: Tuple[str, str, str]) -> Optional[GraphEdge]:
    source, target, edge_type_value = triple
    for edge in getattr(graph, "edges", []) or []:
        if (
            str(getattr(edge, "source", "")) == str(source)
            and str(getattr(edge, "target", "")) == str(target)
            and _enum_value(getattr(edge, "edge_type", "")) == str(edge_type_value)
        ):
            return edge
    return None


def _required_edge_triples(derivation: Any) -> List[Tuple[str, str, str]]:
    triples: List[Tuple[str, str, str]] = []
    for edge in getattr(derivation, "required_edge_triples", []) or []:
        if isinstance(edge, (list, tuple)) and len(edge) == 3:
            triples.append((str(edge[0]), str(edge[1]), str(edge[2])))
    return triples


def _candidate_binding_swap_target(graph: HCEG, edge: GraphEdge) -> Optional[str]:
    source = str(getattr(edge, "source", ""))
    target = str(getattr(edge, "target", ""))
    edge_type = getattr(edge, "edge_type", None)
    source_node = graph.nodes.get(source)
    target_node = graph.nodes.get(target)
    if source_node is None or target_node is None:
        return None
    candidates: List[Tuple[int, str]] = []
    for node_id, node in graph.nodes.items():
        if node_id == target or getattr(node, "node_type", None) != NodeType.HEADER:
            continue
        if edge_type == EdgeType.ROW_PATH and getattr(node, "row", None) == getattr(source_node, "row", None):
            candidates.append((abs(int(getattr(node, "col", 0)) - int(getattr(target_node, "col", 0))), node_id))
        elif edge_type in {EdgeType.COL_PATH, EdgeType.VALUE_UNDER_HEADER} and getattr(node, "col", None) == getattr(source_node, "col", None):
            candidates.append((abs(int(getattr(node, "row", 0)) - int(getattr(target_node, "row", 0))), node_id))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1]


def generate_candidate_targeted_interventions(
    graph: HCEG,
    evidence: Optional[EvidenceSubgraph],
    candidate: Any,
    derivation: Any = None,
    max_benign: int = 5,
) -> List[InterventionResult]:
    if evidence_is_fallback(evidence):
        return []
    interventions: List[InterventionResult] = []
    support = cellref_node_ids(candidate)
    ev_nodes = set(getattr(evidence, "evidence_nodes", set()) or set()) if evidence else set()
    dependency_edges = _required_edge_triples(derivation)
    dependency_nodes: Set[str] = set(support)
    for source, target, _etype in dependency_edges:
        dependency_nodes.add(source)
        dependency_nodes.add(target)
    for nid in sorted(support):
        if nid not in graph.nodes:
            continue
        g2 = copy.deepcopy(graph)
        g2.remove_node(nid)
        interventions.append(InterventionResult(
            intervention_type=InterventionType.SUPPORT_DELETE,
            intervened_graph=g2,
            removed_nodes=[nid],
            description=f"Deleted candidate support cell {nid}",
        ))
    for triple in dependency_edges:
        edge = _edge_from_triple(graph, triple)
        if edge is None:
            continue
        edge_type = _edge_type_from_value(getattr(edge, "edge_type", ""))
        g2 = copy.deepcopy(graph)
        removed = g2.remove_edge(edge.source, edge.target, edge_type)
        if removed:
            interventions.append(InterventionResult(
                intervention_type=InterventionType.REQUIRED_EDGE_DELETE,
                intervened_graph=g2,
                removed_edges=removed,
                description=f"Deleted required derivation edge {edge.source}->{edge.target}:{_enum_value(edge.edge_type)}",
            ))
        if edge_type in {EdgeType.ROW_PATH, EdgeType.COL_PATH, EdgeType.VALUE_UNDER_HEADER}:
            replacement = _candidate_binding_swap_target(graph, edge)
            if replacement:
                g3 = copy.deepcopy(graph)
                removed_swap = g3.remove_edge(edge.source, edge.target, edge_type)
                g3.add_edge(GraphEdge(
                    source=edge.source,
                    target=replacement,
                    edge_type=edge_type,
                    weight=getattr(edge, "weight", 1.0),
                    metadata={
                        "intervention": "binding_swap",
                        "original_target": edge.target,
                    },
                ))
                interventions.append(InterventionResult(
                    intervention_type=InterventionType.BINDING_SWAP,
                    intervened_graph=g3,
                    removed_edges=removed_swap,
                    modified_nodes=[edge.target, replacement],
                    description=(
                        f"Swapped required binding edge {edge.source}->{edge.target}:"
                        f"{_enum_value(edge.edge_type)} to {replacement}"
                    ),
                ))
    benign = []
    for nid, node in graph.nodes.items():
        if (
            node.node_type == NodeType.CELL
            and node.is_numeric
            and nid not in dependency_nodes
            and nid not in ev_nodes
        ):
            benign.append(nid)
    for nid in benign[:max_benign]:
        g2 = copy.deepcopy(graph)
        g2.remove_node(nid)
        interventions.append(InterventionResult(
            intervention_type=InterventionType.BENIGN_IRRELEVANT,
            intervened_graph=g2,
            removed_nodes=[nid],
            description=f"Deleted non-evidence numeric cell {nid}",
        ))
    return interventions
