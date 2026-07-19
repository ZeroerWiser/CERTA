"""Materialize typed executable derivations from executor candidates."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from graph_builder import EdgeType, HCEG

from .answer_equivalence import inference_answer_key
from .project import canonical_answer_key
from .schema import ANSWER_DOMAINS, PROJECTION_OPERATORS, ExecutableDerivation


SIGNATURES = {
    "LOOKUP": "(EntityKey, Attribute) -> Value",
    "LOOKUP_AGGREGATE": "(AggregateScope, Attribute) -> Scalar",
    "SUM": "Set[Scalar] -> Scalar",
    "AVERAGE": "Set[Scalar] -> Scalar",
    "DIFF": "(Scalar, Scalar) -> Scalar",
    "RATIO": "(Scalar, Scalar) -> Scalar",
    "COUNT": "Set[Entity] -> Integer",
    "ARGMAX": "(Set[Entity], Entity -> Scalar) -> Entity",
    "ARGMIN": "(Set[Entity], Entity -> Scalar) -> Entity",
    "PAIR_COMPARE": "(Entity1, Scalar1, Entity2, Scalar2, Polarity) -> Entity | Boolean",
    "UNKNOWN": "unknown",
}


CRITICAL_EDGE_TYPES = {
    EdgeType.VALUE_UNDER_HEADER,
    EdgeType.ROW_PATH,
    EdgeType.COL_PATH,
    EdgeType.AGGREGATE_DEPENDS,
    EdgeType.COMPARISON_BETWEEN,
    EdgeType.PART_OF,
}


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value or ""))


def _as_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "to_dict"):
        try:
            return dict(value.to_dict())
        except Exception:
            return {}
    return {}


def _candidate_dict(candidate: Any, candidate_id: str = "") -> Dict[str, Any]:
    payload = _as_dict(candidate)
    if payload:
        return payload
    cells = []
    for ref in getattr(candidate, "cells_used", []) or []:
        cells.append({
            "row": getattr(ref, "row", None),
            "col": getattr(ref, "col", None),
            "value": getattr(ref, "value", ""),
            "row_headers": list(getattr(ref, "row_headers", []) or []),
            "col_headers": list(getattr(ref, "col_headers", []) or []),
        })
    return {
        "candidate_id": candidate_id,
        "denotation": str(getattr(candidate, "denotation", "")),
        "normalized_denotation": str(getattr(candidate, "denotation", "")),
        "operation": _enum_value(getattr(candidate, "operation", "")),
        "priority": getattr(candidate, "priority", None),
        "cells_used": cells,
        "computation_trace": str(getattr(candidate, "computation_trace", "")),
        "operation_metadata": _as_dict(getattr(candidate, "operation_metadata", {})),
        "certificate": _as_dict(getattr(candidate, "certificate", {})),
    }


def _cell_node_id(cell: Mapping[str, Any]) -> str:
    row = cell.get("row")
    col = cell.get("col")
    return str(cell.get("node_id") or f"cell_{row}_{col}")


def _operation_metadata(payload: Mapping[str, Any]) -> Dict[str, Any]:
    return _as_dict(payload.get("operation_metadata"))


def _operation_family(operation: str, trace: str, metadata: Optional[Mapping[str, Any]] = None) -> str:
    family = str((metadata or {}).get("operation_family") or "").upper()
    if family in SIGNATURES:
        return family
    op = operation.lower()
    tr = trace.lower()
    if op == "lookup_cell":
        return "LOOKUP"
    if op == "lookup_aggregate":
        return "LOOKUP_AGGREGATE"
    if op == "compare":
        return "PAIR_COMPARE"
    if op != "arithmetic":
        return "UNKNOWN"
    if tr.startswith("argmax"):
        return "ARGMAX"
    if tr.startswith("argmin"):
        return "ARGMIN"
    if tr.startswith("average("):
        return "AVERAGE"
    if tr.startswith("count"):
        return "COUNT"
    if " / " in tr:
        return "RATIO"
    if " - " in tr:
        return "DIFF"
    if " + " in tr:
        return "SUM"
    return "UNKNOWN"


def _contains_answer(headers: Iterable[Any], answer: str) -> bool:
    answer_key = canonical_answer_key(answer)
    return any(canonical_answer_key(value) == answer_key for value in headers or [])


def _projection_operator(
    family: str,
    cells: Sequence[Mapping[str, Any]],
    answer: str,
    metadata: Optional[Mapping[str, Any]] = None,
) -> str:
    if family in {"SUM", "AVERAGE", "DIFF", "RATIO", "COUNT"}:
        return "SCALAR_RESULT_PROJECTION"
    if family in {"LOOKUP", "LOOKUP_AGGREGATE"}:
        return "VALUE_PROJECTION"
    if family in {"ARGMAX", "ARGMIN", "PAIR_COMPARE"}:
        if any(_contains_answer(cell.get("row_headers") or [], answer) for cell in cells):
            return "ROW_ENTITY_PROJECTION"
        if any(_contains_answer(cell.get("col_headers") or [], answer) for cell in cells):
            return "COLUMN_ENTITY_PROJECTION"
        if answer.lower() in {"true", "false"}:
            return "BOOLEAN_PROJECTION"
        hinted = str((metadata or {}).get("projection_operator") or "")
        if hinted in {"ROW_ENTITY_PROJECTION", "COLUMN_ENTITY_PROJECTION", "BOOLEAN_PROJECTION"}:
            return hinted
        return "UNKNOWN"
    hinted = str((metadata or {}).get("projection_operator") or "")
    if hinted in PROJECTION_OPERATORS:
        return hinted
    return "UNKNOWN"


def _output_domain(projection: str, answer: str) -> str:
    if projection == "SCALAR_RESULT_PROJECTION":
        return "SCALAR"
    if projection == "BOOLEAN_PROJECTION":
        return "BOOLEAN"
    if projection in {"ROW_ENTITY_PROJECTION", "COLUMN_ENTITY_PROJECTION"}:
        return "ENTITY"
    if projection == "VALUE_PROJECTION":
        if inference_answer_key(answer).category.startswith("NUMERIC"):
            return "SCALAR"
        return "ENTITY"
    return "UNKNOWN"


def _comparison_polarity(family: str, trace: str, metadata: Mapping[str, Any]) -> str:
    polarity = str(metadata.get("comparison_polarity") or "").lower()
    mapping = {
        "max": "max",
        "argmax": "max",
        "greater": "greater",
        "greater_than": "greater",
        "greater_equal": "greater_equal",
        "min": "min",
        "argmin": "min",
        "less": "less",
        "less_than": "less",
        "less_equal": "less_equal",
        "equal": "equal",
        "equals": "equal",
    }
    if polarity in mapping:
        return mapping[polarity]
    tr = trace.lower()
    if family == "ARGMAX" or tr.startswith("argmax"):
        return "max"
    if family == "ARGMIN" or tr.startswith("argmin"):
        return "min"
    if family == "PAIR_COMPARE":
        return "greater_equal"
    return "unknown"


def _edge_triple(edge: Any) -> Tuple[str, str, str]:
    return (
        str(getattr(edge, "source", "")),
        str(getattr(edge, "target", "")),
        _enum_value(getattr(edge, "edge_type", "")),
    )


def _dependency_edge_types(family: str, projection: str) -> set[EdgeType]:
    edge_types = {EdgeType.PART_OF}
    if projection == "ROW_ENTITY_PROJECTION":
        edge_types.add(EdgeType.ROW_PATH)
    elif projection == "COLUMN_ENTITY_PROJECTION":
        edge_types.update({EdgeType.COL_PATH, EdgeType.VALUE_UNDER_HEADER})
    elif projection in {"VALUE_PROJECTION", "SCALAR_RESULT_PROJECTION"}:
        edge_types.update({EdgeType.ROW_PATH, EdgeType.COL_PATH, EdgeType.VALUE_UNDER_HEADER})
    if family in {"LOOKUP_AGGREGATE", "SUM", "AVERAGE", "COUNT"}:
        edge_types.add(EdgeType.AGGREGATE_DEPENDS)
    if family in {"ARGMAX", "ARGMIN", "PAIR_COMPARE"}:
        edge_types.update({
            EdgeType.ROW_PATH,
            EdgeType.COL_PATH,
            EdgeType.VALUE_UNDER_HEADER,
            EdgeType.COMPARISON_BETWEEN,
        })
    return edge_types


def _required_edges(
    graph: Optional[HCEG],
    operand_node_ids: Sequence[str],
    *,
    family: str = "UNKNOWN",
    projection: str = "UNKNOWN",
) -> List[Tuple[str, str, str]]:
    if graph is None:
        return []
    triples: List[Tuple[str, str, str]] = []
    edge_types = _dependency_edge_types(family, projection)
    operand_set = set(operand_node_ids)
    for node_id in operand_node_ids:
        if node_id not in getattr(graph, "nodes", {}):
            continue
        for target, edge in graph.neighbors(node_id, edge_types):
            triples.append(_edge_triple(edge))
        for source, edge in graph.predecessors(node_id, edge_types):
            if getattr(edge, "edge_type", None) == EdgeType.AGGREGATE_DEPENDS or source in operand_set:
                triples.append(_edge_triple(edge))
    seen = set()
    out: List[Tuple[str, str, str]] = []
    for triple in triples:
        if triple not in seen:
            seen.add(triple)
            out.append(triple)
    return out


def _header_ids(graph: Optional[HCEG], node_id: str, edge_types: set[EdgeType]) -> List[str]:
    if graph is None or node_id not in getattr(graph, "nodes", {}):
        return []
    out: List[str] = []
    for target, edge in graph.neighbors(node_id, edge_types):
        out.append(str(target))
    return sorted(dict.fromkeys(out))


def _operand_metadata(
    cells: Sequence[Mapping[str, Any]],
    operand_node_ids: Sequence[str],
    *,
    graph: Optional[HCEG],
) -> List[Dict[str, Any]]:
    metadata: List[Dict[str, Any]] = []
    for cell, node_id in zip(cells, operand_node_ids):
        payload = dict(cell)
        payload.setdefault("node_id", node_id)
        payload["target_entity_ids"] = _header_ids(graph, node_id, {EdgeType.ROW_PATH})
        payload["target_measure_ids"] = _header_ids(graph, node_id, {EdgeType.COL_PATH, EdgeType.VALUE_UNDER_HEADER})
        metadata.append(payload)
    return metadata


def _sanitized_source_candidate(payload: Mapping[str, Any], candidate_id: str) -> Dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "operation": str(payload.get("operation") or ""),
        "source": str(payload.get("source") or ""),
        "operation_metadata": _operation_metadata(payload),
    }


def _candidate_pairs(
    certified_candidates: Optional[Sequence[Any]],
    live_candidates: Optional[Sequence[Any]],
) -> List[Tuple[str, Dict[str, Any]]]:
    rows = list(certified_candidates or [])
    live = list(live_candidates or [])
    pairs: List[Tuple[str, Dict[str, Any]]] = []
    for idx, row in enumerate(rows):
        payload = _candidate_dict(row, candidate_id=f"cand_{idx}")
        candidate_id = str(payload.get("candidate_id") or f"cand_{idx}")
        pairs.append((candidate_id, payload))
    if pairs:
        return pairs
    for idx, row in enumerate(live):
        payload = _candidate_dict(row, candidate_id=f"cand_{idx}")
        candidate_id = str(payload.get("candidate_id") or f"cand_{idx}")
        payload["candidate_id"] = candidate_id
        pairs.append((candidate_id, payload))
    return pairs


def materialize_derivations(
    *,
    certified_candidates: Optional[Sequence[Any]] = None,
    live_candidates: Optional[Sequence[Any]] = None,
    graph: Optional[HCEG] = None,
) -> List[ExecutableDerivation]:
    derivations: List[ExecutableDerivation] = []
    for idx, (candidate_id, payload) in enumerate(_candidate_pairs(certified_candidates, live_candidates), start=1):
        cells = [_as_dict(cell) for cell in (payload.get("cells_used") or []) if _as_dict(cell)]
        operand_node_ids = [_cell_node_id(cell) for cell in cells]
        operation = str(payload.get("operation") or "")
        trace = str(payload.get("computation_trace") or "")
        metadata = _operation_metadata(payload)
        family = _operation_family(operation, trace, metadata)
        projected_answer = str(payload.get("denotation") or "")
        projection = _projection_operator(family, cells, projected_answer, metadata)
        output_domain = _output_domain(projection, projected_answer)
        metadata_domain = str(metadata.get("answer_domain") or "")
        if output_domain == "UNKNOWN" and metadata_domain in ANSWER_DOMAINS:
            output_domain = metadata_domain
        polarity = _comparison_polarity(family, trace, metadata)
        required_edges = _required_edges(
            graph,
            operand_node_ids,
            family=family,
            projection=projection,
        )
        failures: List[str] = []
        if family == "UNKNOWN":
            failures.append("unknown_operation_family")
        if not projected_answer.strip():
            failures.append("empty_projected_answer")
        if not operand_node_ids:
            failures.append("missing_operands")
        if graph is not None:
            missing = [node_id for node_id in operand_node_ids if node_id not in getattr(graph, "nodes", {})]
            if missing:
                failures.append("required_nodes_missing")
        if projection == "UNKNOWN":
            failures.append("unknown_projection_operator")
        if family in {"DIFF", "RATIO", "PAIR_COMPARE"} and len(operand_node_ids) != 2:
            failures.append("operation_arity_mismatch")
        if family in {"SUM", "AVERAGE", "ARGMAX", "ARGMIN"} and len(operand_node_ids) < 1:
            failures.append("operation_arity_mismatch")
        provenance_complete = not failures
        derivation_id = f"D{idx}"
        program_args = ",".join(operand_node_ids)
        derivations.append(ExecutableDerivation(
            derivation_id=derivation_id,
            source_candidate_id=candidate_id,
            operation_family=family,
            operand_node_ids=operand_node_ids,
            required_edge_triples=required_edges,
            typed_signature=SIGNATURES.get(family, "unknown"),
            projection_operator=projection,
            projected_answer=projected_answer,
            output_domain=output_domain,
            evidence_ids=[],
            executable_program=f"{family}[{polarity}]({program_args})->{projection}" if polarity != "unknown" else f"{family}({program_args})->{projection}",
            provenance_complete=provenance_complete,
            availability="available" if provenance_complete else "unavailable",
            failure_reasons=failures,
            operand_metadata=_operand_metadata(cells, operand_node_ids, graph=graph),
            source_candidate=_sanitized_source_candidate(payload, candidate_id),
            operation_metadata=metadata,
            comparison_polarity=polarity,
        ))
    return derivations
