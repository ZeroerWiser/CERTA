"""Symmetric structural derivation frontier generation."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from graph_builder import EdgeType, NodeType

from .materialize import materialize_derivations
from .project import canonical_answer_key
from .schema import ExecutableDerivation, PreEvidenceQueryContract


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value or ""))


def _node_text(node: Any) -> str:
    text = str(getattr(node, "text", "") or "")
    if text:
        return text
    numeric = getattr(node, "numeric_value", None)
    return "" if numeric is None else str(numeric)


def _node_number(node: Any) -> Optional[float]:
    value = getattr(node, "numeric_value", None)
    if value is not None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    from certa.repair.evidence_dsl import parse_number

    return parse_number(_node_text(node))


def _header_labels(graph: Any, node_id: str, edge_types: Iterable[EdgeType]) -> List[str]:
    labels: List[str] = []
    wanted = set(edge_types)
    for target, edge in graph.neighbors(node_id, wanted):
        node = graph.nodes.get(target)
        text = _node_text(node)
        if text:
            labels.append(text)
    return labels


def _cell_payload(graph: Any, node_id: str) -> Dict[str, Any]:
    node = graph.nodes[node_id]
    return {
        "node_id": node_id,
        "row": getattr(node, "row", None),
        "col": getattr(node, "col", None),
        "value": _node_text(node),
        "row_headers": _header_labels(graph, node_id, [EdgeType.ROW_PATH]),
        "col_headers": _header_labels(graph, node_id, [EdgeType.COL_PATH, EdgeType.VALUE_UNDER_HEADER]),
    }


def _evidence_cell_ids(graph: Any, evidence: Any) -> List[str]:
    if graph is None:
        return []
    evidence_nodes = set(getattr(evidence, "evidence_nodes", set()) or set())
    if not evidence_nodes:
        return []
    out: List[str] = []
    for node_id in sorted(evidence_nodes):
        node = graph.nodes.get(node_id)
        if node is None:
            continue
        if getattr(node, "node_type", None) not in {NodeType.CELL, NodeType.AGGREGATOR}:
            continue
        if not _node_text(node).strip():
            continue
        out.append(str(node_id))
    return out


def _allowed(contract: PreEvidenceQueryContract, family: str) -> bool:
    allowed = {str(item) for item in contract.candidate_independent_operation_hypotheses or []}
    if not allowed or "UNKNOWN" in allowed or family in allowed:
        return True
    if "LOOKUP" in allowed and family == "LOOKUP_AGGREGATE":
        return True
    if "PAIR_COMPARE" in allowed and family in {"ARGMAX", "ARGMIN"}:
        return True
    return False


def _lookup_payloads(graph: Any, cell_ids: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, node_id in enumerate(cell_ids, start=1):
        cell = _cell_payload(graph, node_id)
        value = str(cell.get("value") or "")
        if not value.strip():
            continue
        family = "LOOKUP_AGGREGATE" if getattr(graph.nodes[node_id], "node_type", None) == NodeType.AGGREGATOR else "LOOKUP"
        rows.append({
            "candidate_id": f"frontier_lookup_{idx}",
            "denotation": value,
            "normalized_denotation": value,
            "operation": "lookup_aggregate" if family == "LOOKUP_AGGREGATE" else "lookup_cell",
            "priority": 5,
            "cells_used": [cell],
            "computation_trace": f"structural frontier lookup {node_id} -> {value}",
            "source": "symmetric_derivation_frontier_v1",
            "operation_metadata": {
                "operation_family": family,
                "projection_operator": "VALUE_PROJECTION",
                "answer_domain": "SCALAR" if _node_number(graph.nodes[node_id]) is not None else "ENTITY",
                "semantic_source": "symmetric_derivation_frontier_v1",
            },
            "certificate": {
                "path_verified": False,
                "evidence_fallback": False,
                "frontier_generated": True,
                "requires_independent_derivation_verification": True,
            },
        })
    return rows


def _row_label(cell: Mapping[str, Any]) -> str:
    values = cell.get("row_headers") or []
    if isinstance(values, list) and values:
        return str(values[0])
    return str(cell.get("value", ""))


def _numeric_groups(graph: Any, cell_ids: Sequence[str]) -> Dict[Tuple[str, ...], List[Tuple[str, Dict[str, Any], float]]]:
    groups: Dict[Tuple[str, ...], List[Tuple[str, Dict[str, Any], float]]] = {}
    for node_id in cell_ids:
        node = graph.nodes.get(node_id)
        number = _node_number(node)
        if number is None:
            continue
        cell = _cell_payload(graph, node_id)
        key = tuple(str(x) for x in (cell.get("col_headers") or []))
        if not key:
            key = ("__evidence_numeric_cells__",)
        groups.setdefault(key, []).append((node_id, cell, number))
    return groups


def _extreme_payloads(
    graph: Any,
    cell_ids: Sequence[str],
    *,
    families: Sequence[str],
    max_group_size: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    groups = _numeric_groups(graph, cell_ids)
    counter = 0
    for key, items in sorted(groups.items()):
        if len(items) < 2 or len(items) > max_group_size:
            continue
        for family in families:
            values = [number for _node_id, _cell, number in items]
            selected_idx = values.index(max(values)) if family == "ARGMAX" else values.index(min(values))
            selected = items[selected_idx][1]
            counter += 1
            rows.append({
                "candidate_id": f"frontier_{family.lower()}_{counter}",
                "denotation": _row_label(selected),
                "normalized_denotation": _row_label(selected),
                "operation": "arithmetic",
                "priority": 5,
                "cells_used": [cell for _node_id, cell, _number in items],
                "computation_trace": f"structural frontier {family.lower()} over {list(key)} -> {_row_label(selected)}",
                "source": "symmetric_derivation_frontier_v1",
                "operation_metadata": {
                    "operation_family": family,
                    "projection_operator": "ROW_ENTITY_PROJECTION",
                    "answer_domain": "ENTITY",
                    "comparison_polarity": "max" if family == "ARGMAX" else "min",
                    "selected_operand_index": selected_idx,
                    "semantic_source": "symmetric_derivation_frontier_v1",
                },
                "certificate": {
                    "path_verified": False,
                    "evidence_fallback": False,
                    "frontier_generated": True,
                    "requires_independent_derivation_verification": True,
                },
            })
    return rows


def _dedupe_derivations(
    derivations: Sequence[ExecutableDerivation],
    existing_derivations: Sequence[ExecutableDerivation],
) -> List[ExecutableDerivation]:
    seen = {
        (
            canonical_answer_key(item.projected_answer),
            item.operation_family,
            tuple(item.operand_node_ids),
            item.projection_operator,
        )
        for item in existing_derivations or []
    }
    out: List[ExecutableDerivation] = []
    for item in derivations:
        key = (
            canonical_answer_key(item.projected_answer),
            item.operation_family,
            tuple(item.operand_node_ids),
            item.projection_operator,
        )
        if key in seen:
            continue
        seen.add(key)
        item.derivation_id = f"FD{len(out) + 1}"
        out.append(item)
    return out


def build_symmetric_derivation_frontier(
    *,
    contract: PreEvidenceQueryContract,
    graph: Any = None,
    evidence: Any = None,
    existing_derivations: Optional[Sequence[ExecutableDerivation]] = None,
    max_group_size: int = 16,
) -> List[ExecutableDerivation]:
    """Generate graph-local executable derivations for both original and repair views."""
    if graph is None or evidence is None:
        return []
    cell_ids = _evidence_cell_ids(graph, evidence)
    if not cell_ids:
        return []
    payloads: List[Dict[str, Any]] = []
    if _allowed(contract, "LOOKUP") or _allowed(contract, "LOOKUP_AGGREGATE"):
        payloads.extend(_lookup_payloads(graph, cell_ids))
    extreme_families = [family for family in ("ARGMAX", "ARGMIN") if _allowed(contract, family)]
    if extreme_families:
        payloads.extend(_extreme_payloads(
            graph,
            cell_ids,
            families=extreme_families,
            max_group_size=max_group_size,
        ))
    if not payloads:
        return []
    derivations = materialize_derivations(certified_candidates=payloads, graph=graph)
    return _dedupe_derivations(derivations, existing_derivations or [])
