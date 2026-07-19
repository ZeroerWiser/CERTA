"""Round 6 derivation lattice and quotient-space construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from evidence_retriever import InterventionResult, InterventionType
from graph_builder import EdgeType, HCEG

from .admissibility import admissibility_result, check_candidate_contract
from .answer_equivalence import inference_answer_key, inference_answers_equivalent
from .project import execute_projection_from_nodes
from .replay import replay_derivation_under_intervention
from .schema import ExecutableDerivation, PreEvidenceQueryContract, to_jsonable


ROUND6_LATTICE_VERSION = "derivation_lattice_v1"


@dataclass
class LatticeStageTrace:
    entered_stage: str
    left_stage: str
    loss_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(self)


@dataclass
class DerivationLatticeMember:
    derivation_id: str
    answer_key: str
    answer_key_category: str
    projected_answer: str
    roundtrip_executable: bool
    roundtrip_answer: str = ""
    candidate_observation_equivalent: bool = False
    candidate_observation_mismatch_reason: str = ""
    contract_compatible: bool = False
    provenance_complete: bool = False
    evidence_grounded: bool = False
    intervention_evaluable: bool = False
    provenance_state: str = "UNAVAILABLE"
    program_class: str = ""
    support_footprint: str = ""
    intervention_signature: str = ""
    support_evidence_ids: List[str] = field(default_factory=list)
    projection_endpoint_ids: List[str] = field(default_factory=list)
    required_edge_roles: List[str] = field(default_factory=list)
    intervention_observations: List[Dict[str, Any]] = field(default_factory=list)
    fallback_dependency: bool = False
    original_answer_equivalent: bool = False
    stage_trace: List[LatticeStageTrace] = field(default_factory=list)
    failure_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(self)


@dataclass
class DerivationQuotientClass:
    class_id: str
    answer_key: str
    member_derivation_ids: List[str]
    operation_families: List[str]
    program_classes: List[str]
    support_footprints: List[str]
    intervention_signatures: List[str]
    representative_ids: List[str]
    original_support_members: List[str]
    alternative_members: List[str]
    support_evidence_ids: List[str] = field(default_factory=list)
    projection_endpoint_ids: List[str] = field(default_factory=list)
    required_edge_roles: List[str] = field(default_factory=list)
    intervention_observations: List[Dict[str, Any]] = field(default_factory=list)
    provenance_states: List[str] = field(default_factory=list)
    fallback_only: bool = False
    contract_compatible: bool = False
    provenance_complete: bool = False
    evidence_grounded: bool = False
    roundtrip_valid: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(self)


@dataclass
class DerivationLatticeAudit:
    lattice_version: str
    members: List[DerivationLatticeMember]
    quotient_classes: List[DerivationQuotientClass]
    stage_counts: Dict[str, int]
    answer_class_count: int
    quotient_class_count: int
    compression_ratio: float
    budget_trace: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(self)


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value or ""))


def _edge_role(edge: Sequence[Any]) -> str:
    if len(edge) < 3:
        return "UNKNOWN_EDGE"
    return str(edge[2] or "UNKNOWN_EDGE")


def _graph_nodes(graph: Any) -> Mapping[str, Any]:
    return getattr(graph, "nodes", {}) or {}


def _node_role(graph: Any, node_id: str) -> str:
    node = _graph_nodes(graph).get(node_id)
    if node is None:
        return "missing"
    node_type = _enum_value(getattr(node, "node_type", ""))
    header_level = getattr(node, "header_level", None)
    if header_level is None or header_level == -1:
        return node_type
    return f"{node_type}:h{header_level}"


def _projection_endpoints(derivation: ExecutableDerivation, graph: Any) -> List[str]:
    if graph is None:
        return []
    edge_types: set[EdgeType] = set()
    if derivation.projection_operator == "ROW_ENTITY_PROJECTION":
        edge_types = {EdgeType.ROW_PATH}
    elif derivation.projection_operator == "COLUMN_ENTITY_PROJECTION":
        edge_types = {EdgeType.COL_PATH, EdgeType.VALUE_UNDER_HEADER}
    elif derivation.projection_operator == "VALUE_PROJECTION":
        edge_types = {EdgeType.ROW_PATH, EdgeType.COL_PATH, EdgeType.VALUE_UNDER_HEADER}
    endpoints: set[str] = set()
    for node_id in derivation.operand_node_ids:
        if node_id not in _graph_nodes(graph):
            continue
        for target, edge in graph.neighbors(node_id, edge_types):
            endpoints.add(str(target))
        for source, edge in graph.predecessors(node_id, edge_types):
            endpoints.add(str(source))
    return sorted(endpoints)


def _program_class(derivation: ExecutableDerivation, graph: Any) -> str:
    role_structure = [
        _node_role(graph, node_id)
        for node_id in derivation.operand_node_ids
    ]
    return "|".join([
        derivation.operation_family,
        derivation.comparison_polarity,
        derivation.projection_operator,
        derivation.output_domain,
        ",".join(role_structure),
    ])


def _support_footprint(derivation: ExecutableDerivation, projection_endpoints: Sequence[str]) -> str:
    operands = ",".join(sorted(str(node_id) for node_id in derivation.operand_node_ids))
    endpoints = ",".join(sorted(str(node_id) for node_id in projection_endpoints))
    edges = ",".join(
        sorted(f"{source}>{target}:{etype}" for source, target, etype in derivation.required_edge_triples)
    )
    return "|".join([operands, endpoints, edges])


def _response_symbol(replay: Any) -> str:
    if not getattr(replay, "available", False):
        return "UNEVALUABLE"
    if (
        not getattr(replay, "required_nodes_valid", False)
        or not getattr(replay, "required_edges_valid", False)
        or not getattr(replay, "operand_resolution_valid", False)
        or not getattr(replay, "projection_executed", False)
    ):
        return "INVALIDATED"
    if getattr(replay, "changed", False):
        return "ANSWER_CHANGED"
    return "INVARIANT"


def _observation_from_replay(
    *,
    observation_id: str,
    intervention_basis_id: str,
    derivation: ExecutableDerivation,
    intervention: InterventionResult,
) -> Dict[str, Any]:
    replay = replay_derivation_under_intervention(
        intervention_id=observation_id,
        derivation=derivation,
        intervention=intervention,
    )
    symbol = _response_symbol(replay)
    return {
        "observation_id": observation_id,
        "derivation_id": derivation.derivation_id,
        "intervention_basis_id": intervention_basis_id,
        "intervention_type": str(getattr(intervention.intervention_type, "value", intervention.intervention_type)),
        "intervention_evaluable": bool(getattr(replay, "available", False)),
        "derivation_executable_post": bool(
            getattr(replay, "operation_executed", False)
            and getattr(replay, "projection_executed", False)
            and getattr(replay, "required_nodes_valid", False)
            and getattr(replay, "required_edges_valid", False)
            and getattr(replay, "operand_resolution_valid", False)
        ),
        "answer_changed": bool(symbol == "ANSWER_CHANGED"),
        "invalidated": bool(symbol == "INVALIDATED"),
        "post_answer": getattr(replay, "post_projected_answer", None),
        "failure_reason": str(getattr(replay, "failure_reason", "") or ""),
        "response_symbol": symbol,
    }


def _benign_node_id(graph: Any, dependency_nodes: set[str], evidence: Any) -> str:
    evidence_nodes = set(getattr(evidence, "evidence_nodes", set()) or set())
    for node_id, node in sorted(_graph_nodes(graph).items()):
        if node_id in dependency_nodes or node_id in evidence_nodes:
            continue
        if _enum_value(getattr(node, "node_type", "")) != "cell":
            continue
        if getattr(node, "numeric_value", None) is None:
            continue
        return str(node_id)
    return ""


def _intervened_graph(
    graph: Any,
    *,
    removed_nodes: Optional[set[str]] = None,
    removed_edge_triples: Optional[set[Tuple[str, str, str]]] = None,
) -> Tuple[HCEG, List[Any]]:
    removed_nodes = removed_nodes or set()
    removed_edge_triples = removed_edge_triples or set()
    post = HCEG()
    for node_id, node in _graph_nodes(graph).items():
        if str(node_id) in removed_nodes:
            continue
        post.add_node(node)
    removed_edges: List[Any] = []
    for edge in getattr(graph, "edges", []) or []:
        triple = (
            str(getattr(edge, "source", "")),
            str(getattr(edge, "target", "")),
            _enum_value(getattr(edge, "edge_type", "")),
        )
        if triple in removed_edge_triples or triple[0] in removed_nodes or triple[1] in removed_nodes:
            removed_edges.append(edge)
            continue
        post.add_edge(edge)
    return post, removed_edges


def _intervention_signature(
    derivation: ExecutableDerivation,
    *,
    graph: Any = None,
    evidence: Any = None,
) -> Tuple[str, bool, List[str], List[Dict[str, Any]]]:
    if graph is None or evidence is None:
        return "", False, [], []
    evidence_nodes = set(getattr(evidence, "evidence_nodes", set()) or set())
    if not evidence_nodes:
        return "", False, [], []
    observations: List[Dict[str, Any]] = []
    dependency_nodes = set(str(node_id) for node_id in derivation.operand_node_ids)
    endpoints = _projection_endpoints(derivation, graph)
    dependency_nodes.update(endpoints)
    for source, target, _etype in derivation.required_edge_triples:
        dependency_nodes.add(str(source))
        dependency_nodes.add(str(target))
    observation_idx = 1
    if derivation.operand_node_ids:
        removed_nodes = sorted(
            node_id
            for node_id in {str(node_id) for node_id in derivation.operand_node_ids}
            if node_id in _graph_nodes(graph)
        )
        if removed_nodes:
            post_graph, removed_edges = _intervened_graph(graph, removed_nodes=set(removed_nodes))
            intervention = InterventionResult(
                intervention_type=InterventionType.SUPPORT_DELETE,
                intervened_graph=post_graph,
                removed_nodes=removed_nodes,
                removed_edges=removed_edges,
                description="Deleted derivation support footprint",
            )
            observations.append(_observation_from_replay(
                observation_id=f"{derivation.derivation_id}:I{observation_idx}",
                intervention_basis_id="SUPPORT_FOOTPRINT_DELETE",
                derivation=derivation,
                intervention=intervention,
            ))
            observation_idx += 1
    edge_roles = sorted({_edge_role(edge) for edge in derivation.required_edge_triples})
    for role in edge_roles:
        edge_triples: set[Tuple[str, str, str]] = set()
        for source, target, etype in derivation.required_edge_triples:
            if str(etype) != role:
                continue
            edge_triples.add((str(source), str(target), str(etype)))
        if not edge_triples:
            continue
        post_graph, removed_edges = _intervened_graph(graph, removed_edge_triples=edge_triples)
        if not removed_edges:
            continue
        intervention = InterventionResult(
            intervention_type=InterventionType.REQUIRED_EDGE_DELETE,
            intervened_graph=post_graph,
            removed_edges=removed_edges,
            description=f"Deleted required derivation edge role {role}",
        )
        observations.append(_observation_from_replay(
            observation_id=f"{derivation.derivation_id}:I{observation_idx}",
            intervention_basis_id=f"REQUIRED_EDGE_DELETE:{role}",
            derivation=derivation,
            intervention=intervention,
        ))
        observation_idx += 1
    benign = _benign_node_id(graph, dependency_nodes, evidence)
    if benign:
        post_graph, removed_edges = _intervened_graph(graph, removed_nodes={benign})
        intervention = InterventionResult(
            intervention_type=InterventionType.BENIGN_IRRELEVANT,
            intervened_graph=post_graph,
            removed_nodes=[benign],
            removed_edges=removed_edges,
            description=f"Deleted benign non-dependency node {benign}",
        )
        observations.append(_observation_from_replay(
            observation_id=f"{derivation.derivation_id}:I{observation_idx}",
            intervention_basis_id=f"BENIGN_NODE_DELETE:{benign}",
            derivation=derivation,
            intervention=intervention,
        ))
    signature = "|".join(
        f"{item['intervention_basis_id']}={item['response_symbol']}"
        for item in sorted(observations, key=lambda obs: str(obs.get("intervention_basis_id", "")))
    )
    return signature, bool(observations), edge_roles, observations


def _roundtrip(
    derivation: ExecutableDerivation,
    graph: Any,
) -> Tuple[bool, str, str]:
    if graph is None:
        return False, "", "missing_graph"
    nodes = _graph_nodes(graph)
    missing = [node_id for node_id in derivation.operand_node_ids if node_id not in nodes]
    if missing:
        return False, "", "required_nodes_missing"
    executed, failures = execute_projection_from_nodes(
        derivation,
        [nodes[node_id] for node_id in derivation.operand_node_ids],
        graph=graph,
    )
    if failures or executed is None:
        return False, "" if executed is None else str(executed), failures[0] if failures else "projection_execution_failed"
    return True, str(executed), ""


def _failure_stage(
    *,
    roundtrip_valid: bool,
    contract_ok: bool,
    provenance_complete: bool,
    evidence_grounded: bool,
    intervention_evaluable: bool,
    failure_reasons: Sequence[str],
) -> List[LatticeStageTrace]:
    traces = [LatticeStageTrace("L0", "L0")]
    if not roundtrip_valid:
        return [LatticeStageTrace("L0", "L1", next(iter(failure_reasons), "roundtrip_invalid"))]
    traces.append(LatticeStageTrace("L1", "L1"))
    if not contract_ok:
        return traces + [LatticeStageTrace("L1", "L2", "query_contract_incompatible")]
    traces.append(LatticeStageTrace("L2", "L2"))
    if not provenance_complete:
        return traces + [LatticeStageTrace("L2", "L3", "provenance_incomplete")]
    traces.append(LatticeStageTrace("L3", "L3"))
    if not evidence_grounded:
        return traces + [LatticeStageTrace("L3", "L4", "not_evidence_grounded")]
    traces.append(LatticeStageTrace("L4", "L4"))
    if not intervention_evaluable:
        return traces + [LatticeStageTrace("L4", "L5", "intervention_basis_unavailable")]
    return traces + [
        LatticeStageTrace("L5", "L5"),
        LatticeStageTrace("L6", "L6"),
    ]


def _evidence_grounded(derivation: ExecutableDerivation, evidence: Any) -> bool:
    if evidence is None:
        return True
    evidence_nodes = set(getattr(evidence, "evidence_nodes", set()) or set())
    return bool(evidence_nodes) and all(node_id in evidence_nodes for node_id in derivation.operand_node_ids)


def _fallback_dependency(derivation: ExecutableDerivation) -> bool:
    source = derivation.source_candidate if isinstance(derivation.source_candidate, Mapping) else {}
    source_text = " ".join([
        str(source.get("source", "")),
        str(source.get("operation", "")),
        " ".join(str(item) for item in derivation.failure_reasons),
    ]).lower()
    return "fallback" in source_text


def _provenance_state(derivation: ExecutableDerivation) -> str:
    if derivation.availability != "available":
        return "UNAVAILABLE"
    if derivation.provenance_complete:
        return "COMPLETE"
    return "INCOMPLETE"


def _member_sort_key(member: DerivationLatticeMember) -> Tuple[str, str]:
    return (member.support_footprint, member.derivation_id)


def _support_atom_set(member: DerivationLatticeMember) -> set[str]:
    atoms = set(member.support_evidence_ids)
    if member.support_footprint:
        atoms.update(part for part in member.support_footprint.split("|") if part)
    return atoms


def _dominates(left: DerivationLatticeMember, right: DerivationLatticeMember) -> bool:
    if left.answer_key != right.answer_key:
        return False
    if left.roundtrip_executable < right.roundtrip_executable:
        return False
    if left.provenance_state != "COMPLETE" and right.provenance_state == "COMPLETE":
        return False
    if not _support_atom_set(left).issubset(_support_atom_set(right)):
        return False
    left_basis = len(left.intervention_signature.split("|")) if left.intervention_signature else 0
    right_basis = len(right.intervention_signature.split("|")) if right.intervention_signature else 0
    if left_basis < right_basis:
        return False
    if left.fallback_dependency and not right.fallback_dependency:
        return False
    return _member_sort_key(left) <= _member_sort_key(right)


def _representatives(members: Sequence[DerivationLatticeMember]) -> List[str]:
    representatives: List[DerivationLatticeMember] = []
    for member in sorted(members, key=_member_sort_key):
        dominated = any(_dominates(existing, member) for existing in representatives)
        if dominated:
            continue
        representatives = [
            existing for existing in representatives
            if not _dominates(member, existing)
        ]
        representatives.append(member)
    return [member.derivation_id for member in representatives]


def build_derivation_lattice(
    *,
    contract: PreEvidenceQueryContract,
    derivations: Sequence[ExecutableDerivation],
    original_answer: str,
    graph: Any = None,
    evidence: Any = None,
    budget_trace: Optional[Sequence[Mapping[str, Any]]] = None,
) -> DerivationLatticeAudit:
    members: List[DerivationLatticeMember] = []
    for derivation in derivations:
        roundtrip_ok, roundtrip_answer, roundtrip_failure = _roundtrip(derivation, graph)
        observation_equivalent = roundtrip_ok and inference_answers_equivalent(
            roundtrip_answer,
            derivation.projected_answer,
        )
        answer_key = inference_answer_key(roundtrip_answer if roundtrip_ok else derivation.projected_answer)
        contract_check = check_candidate_contract(derivation, contract)
        admissibility = admissibility_result(derivation, contract=contract, graph=graph, evidence=evidence)
        grounded = _evidence_grounded(derivation, evidence)
        signature, evaluable, edge_roles, intervention_observations = _intervention_signature(
            derivation,
            graph=graph,
            evidence=evidence,
        )
        endpoints = _projection_endpoints(derivation, graph)
        failures = list(admissibility.failure_reasons)
        if roundtrip_failure:
            failures.append(roundtrip_failure)
        if roundtrip_ok and not observation_equivalent:
            failures.append("candidate_observation_mismatch")
        failure_unique = sorted(set(str(item) for item in failures if item))
        roundtrip_valid = roundtrip_ok and observation_equivalent
        member = DerivationLatticeMember(
            derivation_id=derivation.derivation_id,
            answer_key=answer_key.compact(),
            answer_key_category=answer_key.category,
            projected_answer=derivation.projected_answer,
            roundtrip_executable=roundtrip_valid,
            roundtrip_answer=roundtrip_answer,
            candidate_observation_equivalent=observation_equivalent,
            candidate_observation_mismatch_reason="" if observation_equivalent else (roundtrip_failure or "executed_answer_differs_from_observation"),
            contract_compatible=contract_check.ok,
            provenance_complete=derivation.provenance_complete,
            evidence_grounded=grounded,
            intervention_evaluable=evaluable,
            provenance_state=_provenance_state(derivation),
            program_class=_program_class(derivation, graph),
            support_footprint=_support_footprint(derivation, endpoints),
            intervention_signature=signature,
            support_evidence_ids=sorted(str(node_id) for node_id in derivation.operand_node_ids),
            projection_endpoint_ids=endpoints,
            required_edge_roles=edge_roles,
            intervention_observations=intervention_observations,
            fallback_dependency=_fallback_dependency(derivation),
            original_answer_equivalent=inference_answers_equivalent(
                roundtrip_answer if roundtrip_ok else derivation.projected_answer,
                original_answer,
            ),
            failure_reasons=failure_unique,
        )
        member.stage_trace = _failure_stage(
            roundtrip_valid=roundtrip_valid,
            contract_ok=member.contract_compatible,
            provenance_complete=member.provenance_complete,
            evidence_grounded=member.evidence_grounded,
            intervention_evaluable=member.intervention_evaluable,
            failure_reasons=failure_unique,
        )
        members.append(member)

    stage_counts = {
        "L0_explored_derivations": len(members),
        "L1_roundtrip_valid": sum(1 for item in members if item.roundtrip_executable),
        "L2_query_contract_compatible": sum(1 for item in members if item.roundtrip_executable and item.contract_compatible),
        "L3_provenance_complete": sum(1 for item in members if item.roundtrip_executable and item.contract_compatible and item.provenance_complete),
        "L4_evidence_grounded": sum(1 for item in members if item.roundtrip_executable and item.contract_compatible and item.provenance_complete and item.evidence_grounded),
        "L5_intervention_evaluable": sum(1 for item in members if item.roundtrip_executable and item.contract_compatible and item.provenance_complete and item.evidence_grounded and item.intervention_evaluable),
    }
    quotient_source = [
        item for item in members
        if item.roundtrip_executable and item.contract_compatible and item.provenance_complete and item.evidence_grounded
    ]
    grouped: Dict[Tuple[str, str, str, str], List[DerivationLatticeMember]] = {}
    derivation_by_id = {item.derivation_id: item for item in derivations}
    for item in quotient_source:
        grouped.setdefault(
            (item.answer_key, item.program_class, item.support_footprint, item.intervention_signature),
            [],
        ).append(item)
    quotient_classes: List[DerivationQuotientClass] = []
    for idx, (_key, class_members) in enumerate(sorted(grouped.items(), key=lambda pair: pair[0]), start=1):
        member_ids = sorted(item.derivation_id for item in class_members)
        operation_families = sorted({
            derivation_by_id[member_id].operation_family
            for member_id in member_ids
            if member_id in derivation_by_id
        })
        support_ids = sorted({sid for item in class_members for sid in item.support_evidence_ids})
        endpoint_ids = sorted({sid for item in class_members for sid in item.projection_endpoint_ids})
        edge_roles = sorted({role for item in class_members for role in item.required_edge_roles})
        intervention_observations: List[Dict[str, Any]] = []
        seen_observations: set[Tuple[str, str, str]] = set()
        for member in sorted(class_members, key=lambda item: item.derivation_id):
            for observation in member.intervention_observations:
                observation_key = (
                    str(observation.get("intervention_basis_id", "")),
                    str(observation.get("response_symbol", "")),
                    str(observation.get("failure_reason", "")),
                )
                if observation_key in seen_observations:
                    continue
                seen_observations.add(observation_key)
                intervention_observations.append(dict(observation))
        quotient_classes.append(DerivationQuotientClass(
            class_id=f"QC{idx}",
            answer_key=class_members[0].answer_key,
            member_derivation_ids=member_ids,
            operation_families=operation_families,
            program_classes=sorted({item.program_class for item in class_members}),
            support_footprints=sorted({item.support_footprint for item in class_members}),
            intervention_signatures=sorted({item.intervention_signature for item in class_members}),
            representative_ids=_representatives(class_members),
            original_support_members=sorted(item.derivation_id for item in class_members if item.original_answer_equivalent),
            alternative_members=sorted(item.derivation_id for item in class_members if not item.original_answer_equivalent),
            support_evidence_ids=support_ids,
            projection_endpoint_ids=endpoint_ids,
            required_edge_roles=edge_roles,
            intervention_observations=intervention_observations,
            provenance_states=sorted({item.provenance_state for item in class_members}),
            fallback_only=all(item.fallback_dependency for item in class_members),
            contract_compatible=all(item.contract_compatible for item in class_members),
            provenance_complete=all(item.provenance_complete for item in class_members),
            evidence_grounded=all(item.evidence_grounded for item in class_members),
            roundtrip_valid=all(item.roundtrip_executable for item in class_members),
        ))
    stage_counts["L6_quotient_classes"] = len(quotient_classes)
    answer_classes = sorted({item.answer_key for item in quotient_source})
    compression_ratio = (
        float(len(quotient_source)) / float(len(quotient_classes))
        if quotient_classes else 0.0
    )
    notes: List[str] = []
    if not quotient_classes:
        notes.append("no_evidence_grounded_roundtrip_quotient_classes")
    if any(not item.candidate_observation_equivalent for item in members):
        notes.append("candidate_observation_mismatch_present")
    return DerivationLatticeAudit(
        lattice_version=ROUND6_LATTICE_VERSION,
        members=members,
        quotient_classes=quotient_classes,
        stage_counts=stage_counts,
        answer_class_count=len(answer_classes),
        quotient_class_count=len(quotient_classes),
        compression_ratio=round(compression_ratio, 6),
        budget_trace=[dict(item) for item in (budget_trace or [])],
        notes=notes,
    )
