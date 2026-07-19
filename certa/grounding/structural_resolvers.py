"""Exact graph-structural resolvers for operation-typed plan closure."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Optional, Sequence, Tuple

from graph_builder import EdgeType, HCEG, NodeType


class ResolutionState(str, Enum):
    UNRESOLVED = "UNRESOLVED"
    UNIQUE = "UNIQUE"
    AMBIGUOUS = "AMBIGUOUS"
    RESOURCE_INCOMPLETE = "RESOURCE_INCOMPLETE"


@dataclass(frozen=True)
class AtomicResolution:
    state: ResolutionState
    binding_ids: Tuple[str, ...] = ()
    candidate_node_ids: Tuple[str, ...] = ()
    unique_node_id: str = ""
    required_edge_triples: Tuple[Tuple[str, str, str], ...] = ()
    failure_reasons: Tuple[str, ...] = ()
    resource_complete: bool = True


@dataclass(frozen=True)
class ScopeMemberProvenance:
    member_binding_ids: Tuple[str, ...]
    shared_binding_ids: Tuple[str, ...]
    node_id: str
    required_edge_triples: Tuple[Tuple[str, str, str], ...]


@dataclass(frozen=True)
class ScopeResolution:
    state: ResolutionState
    member_node_ids: Tuple[str, ...] = ()
    member_provenance: Tuple[ScopeMemberProvenance, ...] = ()
    required_edge_triples: Tuple[Tuple[str, str, str], ...] = ()
    unresolved_members: Tuple[Tuple[str, ...], ...] = ()
    failure_reasons: Tuple[str, ...] = ()
    resource_complete: bool = True


@dataclass(frozen=True)
class EntityValueMember:
    entity_binding_ids: Tuple[str, ...]
    entity_label: str
    value_node_id: str
    numeric_value: float
    required_edge_triples: Tuple[Tuple[str, str, str], ...]


@dataclass(frozen=True)
class EntityMeasureRelationResolution:
    state: ResolutionState
    members: Tuple[EntityValueMember, ...] = ()
    required_edge_triples: Tuple[Tuple[str, str, str], ...] = ()
    tie_value_groups: Tuple[Tuple[float, Tuple[str, ...]], ...] = ()
    unresolved_members: Tuple[Tuple[str, ...], ...] = ()
    failure_reasons: Tuple[str, ...] = ()
    resource_complete: bool = True


_BINDING_EDGE_TYPES = {
    EdgeType.ROW_PATH,
    EdgeType.COL_PATH,
    EdgeType.VALUE_UNDER_HEADER,
}


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value or ""))


def _canonical_ids(values: Iterable[Any]) -> Tuple[str, ...]:
    return tuple(sorted({str(value) for value in values if str(value)}))


def _binding_edges(
    graph: HCEG,
    node_id: str,
    binding_ids: Sequence[str],
) -> Tuple[Tuple[str, str, str], ...]:
    wanted = set(binding_ids)
    return tuple(sorted({
        (node_id, target_id, _enum_value(edge.edge_type))
        for target_id, edge in graph.neighbors(node_id)
        if target_id in wanted and edge.edge_type in _BINDING_EDGE_TYPES
    }))


def resolve_atomic_operand(
    graph: HCEG,
    binding_ids: Iterable[str],
    *,
    max_candidates: Optional[int] = None,
) -> AtomicResolution:
    """Resolve one exact structural conjunction to zero, one, or many cells."""
    bindings = _canonical_ids(binding_ids)
    if not bindings:
        return AtomicResolution(
            state=ResolutionState.UNRESOLVED,
            failure_reasons=("empty_atomic_binding",),
        )
    unknown = tuple(binding for binding in bindings if binding not in graph.nodes)
    if unknown:
        return AtomicResolution(
            state=ResolutionState.UNRESOLVED,
            binding_ids=bindings,
            failure_reasons=tuple(f"unknown_structural_reference:{item}" for item in unknown),
        )
    non_structural = tuple(
        binding
        for binding in bindings
        if graph.nodes[binding].node_type not in {NodeType.HEADER, NodeType.AGGREGATOR}
    )
    if non_structural:
        return AtomicResolution(
            state=ResolutionState.UNRESOLVED,
            binding_ids=bindings,
            failure_reasons=tuple(f"non_schema_reference:{item}" for item in non_structural),
        )

    matches = []
    provenance = []
    for node_id, node in sorted(graph.nodes.items()):
        if node.node_type != NodeType.CELL:
            continue
        edges = _binding_edges(graph, node_id, bindings)
        if {target for _, target, _ in edges} == set(bindings):
            matches.append(node_id)
            provenance.extend(edges)
    if max_candidates is not None and len(matches) > max_candidates:
        return AtomicResolution(
            state=ResolutionState.RESOURCE_INCOMPLETE,
            binding_ids=bindings,
            failure_reasons=(f"candidate_cap_exceeded:{len(matches)}>{max_candidates}",),
            resource_complete=False,
        )
    candidate_ids = tuple(matches)
    if not candidate_ids:
        state = ResolutionState.UNRESOLVED
    elif len(candidate_ids) == 1:
        state = ResolutionState.UNIQUE
    else:
        state = ResolutionState.AMBIGUOUS
    return AtomicResolution(
        state=state,
        binding_ids=bindings,
        candidate_node_ids=candidate_ids,
        unique_node_id=candidate_ids[0] if state == ResolutionState.UNIQUE else "",
        required_edge_triples=tuple(sorted(set(provenance))),
    )


def resolve_finite_scope(
    graph: HCEG,
    member_bindings: Sequence[Sequence[str]],
    *,
    shared_binding_ids: Iterable[str] = (),
    require_numeric: bool = False,
    max_members: Optional[int] = None,
    max_candidates_per_member: Optional[int] = None,
) -> ScopeResolution:
    """Resolve an explicitly enumerated finite structural member scope."""
    members = tuple(sorted({_canonical_ids(member) for member in member_bindings if member}))
    shared = _canonical_ids(shared_binding_ids)
    if not members:
        return ScopeResolution(
            state=ResolutionState.UNRESOLVED,
            failure_reasons=("empty_scope",),
        )
    if max_members is not None and len(members) > max_members:
        return ScopeResolution(
            state=ResolutionState.RESOURCE_INCOMPLETE,
            failure_reasons=(f"member_cap_exceeded:{len(members)}>{max_members}",),
            resource_complete=False,
        )

    resolved_members = []
    provenance = []
    all_edges = []
    for member in members:
        atomic = resolve_atomic_operand(
            graph,
            (*member, *shared),
            max_candidates=max_candidates_per_member,
        )
        if atomic.state == ResolutionState.RESOURCE_INCOMPLETE:
            return ScopeResolution(
                state=ResolutionState.RESOURCE_INCOMPLETE,
                unresolved_members=(member,),
                failure_reasons=atomic.failure_reasons,
                resource_complete=False,
            )
        if atomic.state == ResolutionState.AMBIGUOUS:
            return ScopeResolution(
                state=ResolutionState.AMBIGUOUS,
                unresolved_members=(member,),
                failure_reasons=(f"ambiguous_scope_member:{','.join(member)}",),
            )
        if atomic.state != ResolutionState.UNIQUE:
            return ScopeResolution(
                state=ResolutionState.UNRESOLVED,
                unresolved_members=(member,),
                failure_reasons=atomic.failure_reasons
                or (f"unresolved_scope_member:{','.join(member)}",),
            )
        node = graph.nodes[atomic.unique_node_id]
        if require_numeric and node.numeric_value is None:
            return ScopeResolution(
                state=ResolutionState.UNRESOLVED,
                unresolved_members=(member,),
                failure_reasons=(f"non_numeric_scope_member:{atomic.unique_node_id}",),
            )
        resolved_members.append(atomic.unique_node_id)
        member_edges = tuple(sorted(atomic.required_edge_triples))
        provenance.append(
            ScopeMemberProvenance(
                member_binding_ids=member,
                shared_binding_ids=shared,
                node_id=atomic.unique_node_id,
                required_edge_triples=member_edges,
            )
        )
        all_edges.extend(member_edges)
    return ScopeResolution(
        state=ResolutionState.UNIQUE,
        member_node_ids=tuple(sorted(set(resolved_members))),
        member_provenance=tuple(sorted(provenance, key=lambda item: item.member_binding_ids)),
        required_edge_triples=tuple(sorted(set(all_edges))),
    )


def resolve_entity_measure_relation(
    graph: HCEG,
    entity_member_bindings: Sequence[Sequence[str]],
    *,
    measure_binding_ids: Iterable[str],
    shared_binding_ids: Iterable[str] = (),
    max_members: Optional[int] = None,
    max_candidates_per_member: Optional[int] = None,
) -> EntityMeasureRelationResolution:
    """Resolve a complete finite entity-to-numeric-value relation."""
    measure = _canonical_ids(measure_binding_ids)
    shared = _canonical_ids(shared_binding_ids)
    scope = resolve_finite_scope(
        graph,
        entity_member_bindings,
        shared_binding_ids=(*measure, *shared),
        require_numeric=True,
        max_members=max_members,
        max_candidates_per_member=max_candidates_per_member,
    )
    if scope.state != ResolutionState.UNIQUE:
        return EntityMeasureRelationResolution(
            state=scope.state,
            required_edge_triples=scope.required_edge_triples,
            unresolved_members=scope.unresolved_members,
            failure_reasons=scope.failure_reasons,
            resource_complete=scope.resource_complete,
        )
    relation_members = []
    values: dict[float, list[str]] = {}
    for item in scope.member_provenance:
        node = graph.nodes[item.node_id]
        numeric_value = float(node.numeric_value)
        entity_label = " | ".join(
            str(graph.nodes[binding].text or "")
            for binding in item.member_binding_ids
            if binding in graph.nodes and str(graph.nodes[binding].text or "")
        )
        relation_members.append(
            EntityValueMember(
                entity_binding_ids=item.member_binding_ids,
                entity_label=entity_label,
                value_node_id=item.node_id,
                numeric_value=numeric_value,
                required_edge_triples=item.required_edge_triples,
            )
        )
        values.setdefault(numeric_value, []).append(item.node_id)
    ties = tuple(sorted(
        (value, tuple(sorted(node_ids)))
        for value, node_ids in values.items()
        if len(node_ids) > 1
    ))
    return EntityMeasureRelationResolution(
        state=ResolutionState.UNIQUE,
        members=tuple(sorted(relation_members, key=lambda item: item.entity_binding_ids)),
        required_edge_triples=scope.required_edge_triples,
        tie_value_groups=ties,
    )
