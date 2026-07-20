"""Intervention-aligned derivation equivalence helpers for Round 7."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

from graph_builder import HCEG

from .answer_equivalence import inference_answer_key, inference_answers_equivalent
from .schema import ExecutableDerivation
from certa.planner.lookup_resolver import resolve_lookup_binding


@dataclass(frozen=True)
class RoleInterventionObservation:
    intervention_id: str
    derivation_id: str
    role: str
    target_schema_ids: tuple[str, ...]
    response_symbol: str
    pre_answer: str
    post_answer: Optional[str]
    answer_key: str = ""
    reason_class: str = ""
    benign_control: bool = False


@dataclass(frozen=True)
class PairedInterventionObservation:
    intervention_id: str
    left: RoleInterventionObservation
    right: RoleInterventionObservation


@dataclass(frozen=True)
class RoleInterventionBasisItem:
    intervention_id: str
    role: str
    target_schema_ids: tuple[str, ...]


@dataclass(frozen=True)
class HypothesisBehaviorVector:
    derivation_id: str
    behavior_key: tuple[str, ...]
    responses: tuple[RoleInterventionObservation, ...]


@dataclass(frozen=True)
class BasisRelativeBehaviorClass:
    class_id: str
    behavior_key: tuple[str, ...]
    member_derivation_ids: tuple[str, ...]
    response_vectors: tuple[HypothesisBehaviorVector, ...]


def _metadata(derivation: ExecutableDerivation) -> Mapping[str, Any]:
    if not derivation.operand_metadata:
        return {}
    first = derivation.operand_metadata[0]
    return first if isinstance(first, Mapping) else {}


def _current_role_ids(derivation: ExecutableDerivation, role: str) -> list[str]:
    metadata = _metadata(derivation)
    if role == "TARGET_ENTITY":
        return [str(item) for item in metadata.get("target_entity_ids") or []]
    if role == "TARGET_MEASURE":
        return [str(item) for item in metadata.get("target_measure_ids") or []]
    if role == "TIME_SCOPE":
        return [str(item) for item in metadata.get("time_scope_ids") or []]
    return []


def _resolve_lookup_answer(
    derivation: ExecutableDerivation,
    graph: HCEG,
    *,
    target_entity_ids: Iterable[str],
    target_measure_ids: Iterable[str],
) -> tuple[Optional[str], str]:
    resolution = resolve_lookup_binding(
        graph,
        target_entity_ids=target_entity_ids,
        target_measure_ids=target_measure_ids,
        time_scope_ids=_current_role_ids(derivation, "TIME_SCOPE"),
    )
    if not resolution.unique:
        if resolution.state == "ambiguous":
            return None, "role_binding_ambiguous"
        return None, "role_binding_unresolved"
    return str(graph.nodes[resolution.matched_cell_ids[0]].text or ""), ""


def _observe_role_substitution(
    derivation: ExecutableDerivation,
    graph: HCEG,
    *,
    role: str,
    target_schema_ids: tuple[str, ...],
) -> RoleInterventionObservation:
    intervention_id = f"ROLE_BINDING_SUBSTITUTE:{role}:{','.join(target_schema_ids)}"
    current_entity_ids = _current_role_ids(derivation, "TARGET_ENTITY")
    current_measure_ids = _current_role_ids(derivation, "TARGET_MEASURE")
    if role == "TARGET_ENTITY":
        next_entity_ids = list(target_schema_ids)
        next_measure_ids = current_measure_ids
        benign_control = next_entity_ids == current_entity_ids
    elif role == "TARGET_MEASURE":
        next_entity_ids = current_entity_ids
        next_measure_ids = list(target_schema_ids)
        benign_control = next_measure_ids == current_measure_ids
    else:
        return RoleInterventionObservation(
            intervention_id=intervention_id,
            derivation_id=derivation.derivation_id,
            role=role,
            target_schema_ids=target_schema_ids,
            response_symbol="UNEVALUABLE",
            pre_answer=derivation.projected_answer,
            post_answer=None,
            reason_class="unsupported_role",
            benign_control=False,
        )
    post_answer, failure_reason = _resolve_lookup_answer(
        derivation,
        graph,
        target_entity_ids=next_entity_ids,
        target_measure_ids=next_measure_ids,
    )
    if post_answer is None:
        return RoleInterventionObservation(
            intervention_id=intervention_id,
            derivation_id=derivation.derivation_id,
            role=role,
            target_schema_ids=target_schema_ids,
            response_symbol="INVALIDATED",
            pre_answer=derivation.projected_answer,
            post_answer=None,
            reason_class=failure_reason,
            benign_control=benign_control,
        )
    response = "INVARIANT" if inference_answers_equivalent(derivation.projected_answer, post_answer) else "ANSWER_CHANGED"
    return RoleInterventionObservation(
        intervention_id=intervention_id,
        derivation_id=derivation.derivation_id,
        role=role,
        target_schema_ids=target_schema_ids,
        response_symbol=response,
        pre_answer=derivation.projected_answer,
        post_answer=post_answer,
        answer_key=inference_answer_key(post_answer).compact(),
        benign_control=benign_control,
    )


def build_role_binding_substitution_pairs(
    left: ExecutableDerivation,
    right: ExecutableDerivation,
    graph: HCEG,
    *,
    role: str,
) -> list[PairedInterventionObservation]:
    """Instantiate the same role-level substitution IDs on two derivations."""
    targets = sorted({tuple(_current_role_ids(left, role)), tuple(_current_role_ids(right, role))})
    pairs: list[PairedInterventionObservation] = []
    for target in targets:
        if not target:
            continue
        left_obs = _observe_role_substitution(left, graph, role=role, target_schema_ids=target)
        right_obs = _observe_role_substitution(right, graph, role=role, target_schema_ids=target)
        pairs.append(PairedInterventionObservation(intervention_id=left_obs.intervention_id, left=left_obs, right=right_obs))
    return pairs


def build_sample_fixed_role_intervention_basis(
    derivations: Iterable[ExecutableDerivation],
    graph: HCEG,
    *,
    roles: tuple[str, ...] = ("TARGET_ENTITY", "TARGET_MEASURE"),
) -> tuple[RoleInterventionBasisItem, ...]:
    """Construct one ordered role-intervention basis before pair comparison."""
    del graph
    targets_by_role: dict[str, set[tuple[str, ...]]] = {role: set() for role in roles}
    for derivation in derivations:
        if derivation.operation_family != "LOOKUP":
            continue
        for role in roles:
            target = tuple(_current_role_ids(derivation, role))
            if target:
                targets_by_role.setdefault(role, set()).add(target)
    basis: list[RoleInterventionBasisItem] = []
    for role in roles:
        for target in sorted(targets_by_role.get(role, set())):
            basis.append(RoleInterventionBasisItem(
                intervention_id=f"ROLE_BINDING_SUBSTITUTE:{role}:{','.join(target)}",
                role=role,
                target_schema_ids=target,
            ))
    return tuple(basis)


def evaluate_derivation_on_basis(
    derivation: ExecutableDerivation,
    graph: HCEG,
    basis: Iterable[RoleInterventionBasisItem],
) -> tuple[RoleInterventionObservation, ...]:
    """Evaluate one hypothesis on the complete sample-fixed basis."""
    return tuple(
        _observe_role_substitution(
            derivation,
            graph,
            role=item.role,
            target_schema_ids=item.target_schema_ids,
        )
        for item in basis
    )


def _signature_item(observation: RoleInterventionObservation) -> str:
    suffix = observation.answer_key or observation.reason_class
    return f"{observation.intervention_id}={observation.response_symbol}:{suffix}"


def _behavior_vector(
    derivation: ExecutableDerivation,
    responses: tuple[RoleInterventionObservation, ...],
) -> HypothesisBehaviorVector:
    response_key = tuple(_signature_item(response) for response in responses)
    behavior_key = (
        f"answer={inference_answer_key(derivation.projected_answer).compact()}",
        f"domain={derivation.output_domain}",
        f"projection={derivation.projection_operator}",
        f"operation={derivation.operation_family}",
        f"signature={derivation.typed_signature}",
        *response_key,
    )
    return HypothesisBehaviorVector(
        derivation_id=derivation.derivation_id,
        behavior_key=behavior_key,
        responses=responses,
    )


def build_basis_relative_behavior_classes(
    derivations: Iterable[ExecutableDerivation],
    graph: HCEG,
    basis: Iterable[RoleInterventionBasisItem],
) -> tuple[BasisRelativeBehaviorClass, ...]:
    """Group executable hypotheses by exact basis-relative behavior."""
    groups: dict[tuple[str, ...], list[HypothesisBehaviorVector]] = {}
    basis_tuple = tuple(basis)
    for derivation in derivations:
        responses = evaluate_derivation_on_basis(derivation, graph, basis_tuple)
        vector = _behavior_vector(derivation, responses)
        groups.setdefault(vector.behavior_key, []).append(vector)
    classes: list[BasisRelativeBehaviorClass] = []
    for idx, (key, vectors) in enumerate(sorted(groups.items(), key=lambda item: item[0]), start=1):
        ordered_vectors = tuple(sorted(vectors, key=lambda item: item.derivation_id))
        classes.append(BasisRelativeBehaviorClass(
            class_id=f"BC{idx}",
            behavior_key=key,
            member_derivation_ids=tuple(item.derivation_id for item in ordered_vectors),
            response_vectors=ordered_vectors,
        ))
    return tuple(classes)


def iade_behavior_signatures(pairs: Iterable[PairedInterventionObservation]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    left = []
    right = []
    for pair in pairs:
        left.append(_signature_item(pair.left))
        right.append(_signature_item(pair.right))
    return tuple(sorted(left)), tuple(sorted(right))
