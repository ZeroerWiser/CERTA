"""Shared structural resolver for typed LOOKUP role bindings."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from graph_builder import EdgeType, HCEG, NodeType


@dataclass(frozen=True)
class LookupResolution:
    state: str
    matched_cell_ids: tuple[str, ...] = ()
    required_edge_triples: tuple[tuple[str, str, str], ...] = ()
    role_bindings: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def unique(self) -> bool:
        return self.state == "unique" and len(self.matched_cell_ids) == 1


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value or ""))


def _cell_has_header(graph: HCEG, cell_id: str, header_id: str, edge_types: set[EdgeType]) -> bool:
    for neighbor_id, edge in graph.neighbors(cell_id):
        if neighbor_id == header_id and edge.edge_type in edge_types:
            return True
    return False


def _required_header_edges(graph: HCEG, cell_id: str, header_ids: Iterable[str]) -> tuple[tuple[str, str, str], ...]:
    wanted = {str(item) for item in header_ids}
    triples: list[tuple[str, str, str]] = []
    for neighbor_id, edge in graph.neighbors(cell_id):
        if neighbor_id in wanted:
            triples.append((cell_id, neighbor_id, _enum_value(edge.edge_type)))
    return tuple(sorted(triples))


def resolve_lookup_binding(
    graph: HCEG,
    *,
    target_entity_ids: Iterable[str],
    target_measure_ids: Iterable[str],
    time_scope_ids: Iterable[str] = (),
) -> LookupResolution:
    """Resolve LOOKUP role bindings to zero, unique, or ambiguous cells."""
    entity_ids = tuple(str(item) for item in target_entity_ids if str(item))
    measure_ids = tuple(str(item) for item in target_measure_ids if str(item))
    time_ids = tuple(str(item) for item in time_scope_ids if str(item))
    row_edges = {EdgeType.ROW_PATH}
    measure_edges = {EdgeType.COL_PATH, EdgeType.VALUE_UNDER_HEADER}
    scope_edges = {EdgeType.ROW_PATH, EdgeType.COL_PATH, EdgeType.VALUE_UNDER_HEADER}
    matches: list[str] = []
    for node_id, node in sorted(graph.nodes.items()):
        if node.node_type != NodeType.CELL:
            continue
        if not all(_cell_has_header(graph, node_id, entity_id, row_edges) for entity_id in entity_ids):
            continue
        if not all(_cell_has_header(graph, node_id, measure_id, measure_edges) for measure_id in measure_ids):
            continue
        if not all(_cell_has_header(graph, node_id, time_id, scope_edges) for time_id in time_ids):
            continue
        matches.append(node_id)
    if not matches:
        state = "zero"
    elif len(matches) == 1:
        state = "unique"
    else:
        state = "ambiguous"
    required_edges: tuple[tuple[str, str, str], ...] = ()
    if len(matches) == 1:
        required_edges = _required_header_edges(graph, matches[0], (*entity_ids, *measure_ids, *time_ids))
    return LookupResolution(
        state=state,
        matched_cell_ids=tuple(matches),
        required_edge_triples=required_edges,
        role_bindings={
            "TARGET_ENTITY": entity_ids,
            "TARGET_MEASURE": measure_ids,
            "TIME_SCOPE": time_ids,
        },
    )
