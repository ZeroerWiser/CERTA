"""Round 6/7 causal contrast diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field, is_dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .answer_equivalence import inference_answer_key, inference_answers_equivalent
from .lattice import DerivationLatticeAudit, DerivationQuotientClass
from .admissibility import check_candidate_contract
from .schema import OriginalSupportHypothesisSet, PreEvidenceQueryContract, to_jsonable


ROUND6_CONTRAST_VERSION = "minimal_contrast_set_v1"
ROUND7_COMPACT_CONTRAST_VERSION = "compact_behavioral_contrast_v2"
ROUND8_COMPACT_CONTRAST_VERSION = "compact_behavioral_contrast_v3"


@dataclass
class MinimalContrastSet:
    contrast_version: str
    original_classes: List[str]
    alternative_classes: List[str]
    shared_evidence_atoms: List[str]
    distinguishing_evidence_atoms: List[str]
    intervention_observations: List[Dict[str, Any]]
    unresolved_ambiguities: List[str]
    construction_trace: List[str]
    ready_for_cera: bool = False
    original_support_absent: bool = False
    retained_class_count: int = 0
    retained_alternative_answer_class_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(self)


@dataclass
class CompactBehavioralContrastV2:
    contrast_version: str
    states: Dict[str, bool]
    query_semantics: Dict[str, Any]
    original_hypothesis: Dict[str, Any]
    alternative_hypotheses: List[Dict[str, Any]]
    shared_evidence_refs: List[str]
    distinguishing_evidence_refs: List[str]
    separating_interventions: List[Dict[str, Any]]
    paired_interventions: List[Dict[str, Any]]
    unknowns: List[str]
    registry: Dict[str, Any]
    construction_trace: List[str] = field(default_factory=list)

    @property
    def contrast_constructible(self) -> bool:
        return bool(self.states.get("contrast_constructible", False))

    @property
    def contrast_compact(self) -> bool:
        return bool(self.states.get("contrast_compact", False))

    @property
    def repair_eligible(self) -> bool:
        return bool(self.states.get("repair_eligible", False))

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(self)


@dataclass
class CompactBehavioralContrastV3:
    contrast_version: str
    states: Dict[str, bool]
    query_semantics: Dict[str, Any]
    original_hypothesis: Dict[str, Any]
    alternative_hypothesis: Dict[str, Any]
    alternative_hypotheses: List[Dict[str, Any]]
    separating_interventions: List[Dict[str, Any]]
    unknowns: List[str]
    registry: Dict[str, Any]
    construction_trace: List[str] = field(default_factory=list)

    @property
    def contrast_constructible(self) -> bool:
        return bool(self.states.get("contrast_constructible", False))

    @property
    def contrast_registry_complete(self) -> bool:
        return bool(self.states.get("contrast_registry_complete", False))

    @property
    def contrast_compact(self) -> bool:
        return bool(self.states.get("contrast_compact", False))

    @property
    def repair_eligible(self) -> bool:
        return bool(self.states.get("repair_eligible", False))

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(self)


def _class_by_id(lattice: DerivationLatticeAudit) -> Dict[str, DerivationQuotientClass]:
    return {item.class_id: item for item in lattice.quotient_classes}


def _class_support_atoms(quotient_class: DerivationQuotientClass) -> set[str]:
    atoms = set(quotient_class.support_evidence_ids)
    atoms.update(quotient_class.projection_endpoint_ids)
    atoms.update(f"EDGE_ROLE:{role}" for role in quotient_class.required_edge_roles)
    return {str(item) for item in atoms if str(item)}


def _class_interventions(quotient_class: DerivationQuotientClass) -> List[Dict[str, Any]]:
    observations: List[Dict[str, Any]] = []
    if quotient_class.intervention_observations:
        for item in quotient_class.intervention_observations:
            payload = dict(item)
            payload["class_id"] = quotient_class.class_id
            observations.append(payload)
        return observations
    for signature in quotient_class.intervention_signatures:
        for raw in str(signature or "").split("|"):
            if not raw or "=" not in raw:
                continue
            label, response = raw.split("=", 1)
            observations.append({
                "class_id": quotient_class.class_id,
                "intervention_basis_id": label,
                "response_symbol": response,
            })
    return observations


def _eligible_alternative(quotient_class: DerivationQuotientClass) -> bool:
    return (
        quotient_class.roundtrip_valid
        and quotient_class.contract_compatible
        and quotient_class.provenance_complete
        and quotient_class.evidence_grounded
        and not quotient_class.fallback_only
        and bool(quotient_class.alternative_members)
    )


def _class_id_sort_key(class_id: str) -> tuple[int, str]:
    if str(class_id).startswith("QC"):
        try:
            return int(str(class_id)[2:]), str(class_id)
        except ValueError:
            return 10**9, str(class_id)
    return 10**9, str(class_id)


def _representative_derivation_id(quotient_class: DerivationQuotientClass) -> str:
    representatives = [str(item) for item in quotient_class.representative_ids if str(item)]
    if representatives:
        return sorted(representatives)[0]
    members = [str(item) for item in quotient_class.member_derivation_ids if str(item)]
    return sorted(members)[0] if members else ""


def _register_evidence_atom(
    *,
    registry: Dict[str, Any],
    atom_to_ref: Dict[tuple[str, str], str],
    class_id: str,
    atom_type: str,
    value: str,
) -> str:
    key = (str(atom_type), str(value))
    existing = atom_to_ref.get(key)
    if existing:
        for record in registry["evidence_records"]:
            if record.get("evidence_id") == existing:
                class_ids = set(str(item) for item in record.get("class_ids", []))
                class_ids.add(str(class_id))
                record["class_ids"] = sorted(class_ids, key=_class_id_sort_key)
                break
        return existing
    evidence_id = f"E{len(registry['evidence_records']) + 1}"
    atom_to_ref[key] = evidence_id
    registry["evidence_records"].append({
        "evidence_id": evidence_id,
        "atom_type": str(atom_type),
        "value": str(value),
        "class_ids": [str(class_id)],
    })
    return evidence_id


def _class_evidence_refs(
    quotient_class: DerivationQuotientClass,
    *,
    registry: Dict[str, Any],
    atom_to_ref: Dict[tuple[str, str], str],
) -> List[str]:
    refs: List[str] = []
    for node_id in quotient_class.support_evidence_ids:
        refs.append(_register_evidence_atom(
            registry=registry,
            atom_to_ref=atom_to_ref,
            class_id=quotient_class.class_id,
            atom_type="support_node",
            value=str(node_id),
        ))
    for node_id in quotient_class.projection_endpoint_ids:
        refs.append(_register_evidence_atom(
            registry=registry,
            atom_to_ref=atom_to_ref,
            class_id=quotient_class.class_id,
            atom_type="projection_endpoint",
            value=str(node_id),
        ))
    for role in quotient_class.required_edge_roles:
        refs.append(_register_evidence_atom(
            registry=registry,
            atom_to_ref=atom_to_ref,
            class_id=quotient_class.class_id,
            atom_type="required_edge_role",
            value=str(role),
        ))
    return sorted(dict.fromkeys(refs))


def _compact_role_observation(value: Any) -> Dict[str, Any]:
    if is_dataclass(value):
        payload = to_jsonable(value)
    elif isinstance(value, Mapping):
        payload = dict(value)
    else:
        payload = {}
    return {
        "intervention_id": str(payload.get("intervention_id", "")),
        "derivation_id": str(payload.get("derivation_id", "")),
        "role": str(payload.get("role", "")),
        "target_schema_ids": [str(item) for item in (payload.get("target_schema_ids") or [])],
        "response_symbol": str(payload.get("response_symbol", "")),
        "answer_key": str(payload.get("answer_key", "")),
        "reason_class": str(payload.get("reason_class", "")),
    }


def _observation_signature(value: Mapping[str, Any]) -> str:
    suffix = str(value.get("answer_key") or value.get("reason_class") or "")
    return f"{value.get('response_symbol', '')}:{suffix}"


def _compact_paired_interventions(paired_interventions: Sequence[Any]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    records: List[Dict[str, Any]] = []
    separating: List[Dict[str, Any]] = []
    for pair in paired_interventions:
        payload = to_jsonable(pair) if is_dataclass(pair) else dict(pair or {})
        left = _compact_role_observation(payload.get("left"))
        right = _compact_role_observation(payload.get("right"))
        role_intervention_id = str(payload.get("intervention_id") or left.get("intervention_id") or right.get("intervention_id") or "")
        intervention_ref = f"I{len(records) + 1}"
        original_signature = _observation_signature(left)
        alternative_signature = _observation_signature(right)
        original_evaluable = left.get("response_symbol") not in {"", "UNEVALUABLE"}
        alternative_evaluable = right.get("response_symbol") not in {"", "UNEVALUABLE"}
        is_separating = bool(original_evaluable and alternative_evaluable and original_signature != alternative_signature)
        record = {
            "intervention_ref": intervention_ref,
            "role_intervention_id": role_intervention_id,
            "role": left.get("role") or right.get("role"),
            "target_schema_ids": left.get("target_schema_ids") or right.get("target_schema_ids") or [],
            "original": left,
            "alternative": right,
            "original_signature": original_signature,
            "alternative_signature": alternative_signature,
            "evaluable_on_both_sides": bool(original_evaluable and alternative_evaluable),
            "separating": is_separating,
        }
        records.append(record)
        if is_separating:
            separating.append({
                "intervention_ref": intervention_ref,
                "role_intervention_id": role_intervention_id,
                "original_signature": original_signature,
                "alternative_signature": alternative_signature,
            })
    return records, separating


def _hypothesis_payload(
    *,
    side: str,
    hypothesis_id: str,
    derivation_ref: str,
    quotient_class: DerivationQuotientClass,
    evidence_refs: Sequence[str],
    response_vector: Mapping[str, str],
) -> Dict[str, Any]:
    return {
        "hypothesis_id": hypothesis_id,
        "side": side,
        "class_id": quotient_class.class_id,
        "answer_key": quotient_class.answer_key,
        "derivation_ref": derivation_ref,
        "evidence_refs": list(evidence_refs),
        "response_vector": dict(response_vector),
    }


def build_compact_behavioral_contrast_v2(
    *,
    lattice: DerivationLatticeAudit,
    original_support_symmetry_v3: Any,
    paired_interventions: Sequence[Any] = (),
    query_semantics: Optional[Mapping[str, Any]] = None,
) -> CompactBehavioralContrastV2:
    """Build a prompt-sized Round 7 contrast with explicit eligibility states."""
    original_classes = sorted(
        [
            item for item in lattice.quotient_classes
            if item.original_support_members and item.roundtrip_valid and item.provenance_complete and item.evidence_grounded
        ],
        key=lambda item: _class_id_sort_key(item.class_id),
    )
    alternative_classes = sorted(
        [
            item for item in lattice.quotient_classes
            if item.class_id not in {original.class_id for original in original_classes}
            and _eligible_alternative(item)
        ],
        key=lambda item: _class_id_sort_key(item.class_id),
    )
    registry: Dict[str, Any] = {
        "evidence_records": [],
        "derivation_records": [],
        "hypothesis_records": [],
        "intervention_records": [],
    }
    atom_to_ref: Dict[tuple[str, str], str] = {}
    paired_records, separating = _compact_paired_interventions(paired_interventions)
    registry["intervention_records"] = paired_records

    original_hypotheses: List[Dict[str, Any]] = []
    alternative_hypotheses: List[Dict[str, Any]] = []
    response_vector_original = {
        item["intervention_ref"]: item["original_signature"]
        for item in paired_records
    }
    response_vector_alternative = {
        item["intervention_ref"]: item["alternative_signature"]
        for item in paired_records
    }

    derivation_counter = 1
    hypothesis_counter = 1
    for side, classes in (("original", original_classes), ("alternative", alternative_classes)):
        for quotient_class in classes:
            representative_id = _representative_derivation_id(quotient_class)
            derivation_ref = f"D{derivation_counter}"
            derivation_counter += 1
            evidence_refs = _class_evidence_refs(quotient_class, registry=registry, atom_to_ref=atom_to_ref)
            registry["derivation_records"].append({
                "derivation_ref": derivation_ref,
                "derivation_id": representative_id,
                "class_id": quotient_class.class_id,
                "side": side,
                "answer_key": quotient_class.answer_key,
                "operation_families": list(quotient_class.operation_families),
            })
            hypothesis_id = f"H{hypothesis_counter}"
            hypothesis_counter += 1
            response_vector = response_vector_original if side == "original" else response_vector_alternative
            hypothesis = _hypothesis_payload(
                side=side,
                hypothesis_id=hypothesis_id,
                derivation_ref=derivation_ref,
                quotient_class=quotient_class,
                evidence_refs=evidence_refs,
                response_vector=response_vector,
            )
            registry["hypothesis_records"].append({
                "hypothesis_id": hypothesis_id,
                "side": side,
                "class_id": quotient_class.class_id,
                "derivation_ref": derivation_ref,
            })
            if side == "original":
                original_hypotheses.append(hypothesis)
            else:
                alternative_hypotheses.append(hypothesis)

    evidence_ref_sets = [
        set(item.get("evidence_refs") or [])
        for item in original_hypotheses + alternative_hypotheses
    ]
    if evidence_ref_sets:
        shared = sorted(set.intersection(*evidence_ref_sets))
        distinguishing = sorted(set.union(*evidence_ref_sets) - set(shared))
    else:
        shared = []
        distinguishing = []

    known_evidence_refs = {str(item.get("evidence_id")) for item in registry["evidence_records"]}
    cited_evidence_refs = set().union(*evidence_ref_sets) if evidence_ref_sets else set()
    all_evidence_registry_addressable = cited_evidence_refs.issubset(known_evidence_refs)
    common_evaluable = any(bool(item.get("evaluable_on_both_sides")) for item in paired_records)
    fallback_required = any(item.fallback_only for item in original_classes + alternative_classes)

    unknowns: List[str] = []
    if not original_hypotheses:
        unknowns.append("no_original_executable_support")
    if len(original_hypotheses) > 1:
        unknowns.append("multiple_original_hypotheses")
    if not alternative_hypotheses:
        unknowns.append("no_alternative_hypothesis")
    if len(alternative_hypotheses) > 1:
        unknowns.append("multiple_alternative_hypotheses")
    if not common_evaluable:
        unknowns.append("no_common_role_aligned_intervention")
    if not separating:
        unknowns.append("no_separating_intervention")
    if fallback_required:
        unknowns.append("fallback_only_evidence_required")
    if not all_evidence_registry_addressable:
        unknowns.append("unaddressed_prompt_evidence")

    support_payload = (
        original_support_symmetry_v3.to_dict()
        if hasattr(original_support_symmetry_v3, "to_dict")
        else dict(original_support_symmetry_v3 or {})
    )
    if not bool(support_payload.get("contains_executable_roundtrip_support", False)):
        unknowns.append("original_support_not_roundtrip_executable")

    constructible = bool(original_hypotheses or alternative_hypotheses)
    compact = bool(constructible and registry["hypothesis_records"] and all_evidence_registry_addressable)
    repair_eligible = bool(
        compact
        and len(original_hypotheses) == 1
        and len(alternative_hypotheses) == 1
        and common_evaluable
        and separating
        and not sorted(set(unknowns))
    )
    states = {
        "contrast_constructible": constructible,
        "contrast_compact": compact,
        "repair_eligible": repair_eligible,
    }
    trace = [
        f"quotient_classes={len(lattice.quotient_classes)}",
        f"original_hypotheses={len(original_hypotheses)}",
        f"alternative_hypotheses={len(alternative_hypotheses)}",
        f"paired_interventions={len(paired_records)}",
        f"separating_interventions={len(separating)}",
    ]
    return CompactBehavioralContrastV2(
        contrast_version=ROUND7_COMPACT_CONTRAST_VERSION,
        states=states,
        query_semantics=dict(query_semantics or {}),
        original_hypothesis=original_hypotheses[0] if len(original_hypotheses) == 1 else {},
        alternative_hypotheses=alternative_hypotheses,
        shared_evidence_refs=shared,
        distinguishing_evidence_refs=distinguishing,
        separating_interventions=separating,
        paired_interventions=paired_records,
        unknowns=sorted(set(unknowns)),
        registry=registry,
        construction_trace=trace,
    )


def _response_signature(response: Any) -> str:
    suffix = str(getattr(response, "answer_key", "") or getattr(response, "reason_class", "") or "")
    return f"{getattr(response, 'response_symbol', '')}:{suffix}"


def _class_representative_derivation(behavior_class: Any, derivation_by_id: Mapping[str, Any]) -> Any:
    member_ids = [str(item) for item in getattr(behavior_class, "member_derivation_ids", ()) if str(item)]
    for member_id in sorted(member_ids):
        derivation = derivation_by_id.get(member_id)
        if derivation is not None:
            return derivation
    return None


def _register_v3_evidence(
    registry: Dict[str, Any],
    *,
    class_id: str,
    derivation: Any,
) -> List[str]:
    refs: List[str] = []
    for node_id in getattr(derivation, "operand_node_ids", []) or []:
        evidence_id = f"E{len(registry['evidence_records']) + 1}"
        registry["evidence_records"].append({
            "evidence_id": evidence_id,
            "class_id": class_id,
            "atom_type": "operand_node",
            "value": str(node_id),
        })
        refs.append(evidence_id)
    for source, target, edge_type in getattr(derivation, "required_edge_triples", []) or []:
        evidence_id = f"E{len(registry['evidence_records']) + 1}"
        registry["evidence_records"].append({
            "evidence_id": evidence_id,
            "class_id": class_id,
            "atom_type": "required_edge",
            "value": f"{source}>{target}:{edge_type}",
        })
        refs.append(evidence_id)
    return refs


def _vector_for_derivation(behavior_class: Any, derivation_id: str) -> Any:
    vectors = list(getattr(behavior_class, "response_vectors", ()) or [])
    for vector in vectors:
        if getattr(vector, "derivation_id", "") == derivation_id:
            return vector
    return None


def _hypothesis_v3(
    *,
    side: str,
    hypothesis_id: str,
    derivation_ref: str,
    behavior_class: Any,
    derivation: Any,
    evidence_refs: Sequence[str],
    response_vector: Mapping[str, str],
) -> Dict[str, Any]:
    return {
        "hypothesis_id": hypothesis_id,
        "side": side,
        "behavior_class_id": getattr(behavior_class, "class_id", ""),
        "derivation_ref": derivation_ref,
        "derivation_id": getattr(derivation, "derivation_id", ""),
        "executed_answer": getattr(derivation, "projected_answer", ""),
        "answer_key": inference_answer_key(getattr(derivation, "projected_answer", "")).compact(),
        "answer_domain": getattr(derivation, "output_domain", ""),
        "projection_operator": getattr(derivation, "projection_operator", ""),
        "operation_family": getattr(derivation, "operation_family", ""),
        "evidence_refs": list(evidence_refs),
        "response_vector": dict(response_vector),
    }


def _query_contract_from_semantics(query_semantics: Optional[Mapping[str, Any]]) -> PreEvidenceQueryContract:
    semantics = query_semantics or {}
    return PreEvidenceQueryContract(
        question="",
        answer_domain=str(semantics.get("answer_domain") or "UNKNOWN"),
        allowed_answer_domains=[str(item) for item in semantics.get("allowed_answer_domains") or []],
        allowed_projection_operators=[str(item) for item in semantics.get("allowed_projection_operators") or []],
        candidate_independent_operation_hypotheses=[
            str(item) for item in semantics.get("candidate_independent_operation_hypotheses") or []
        ],
    )


def _query_semantic_unknowns_for_hypothesis(
    hypothesis: Mapping[str, Any],
    derivation_by_id: Mapping[str, Any],
    query_semantics: Optional[Mapping[str, Any]],
    *,
    side: str,
) -> List[str]:
    derivation_id = str(hypothesis.get("derivation_id") or "")
    derivation = derivation_by_id.get(derivation_id)
    if derivation is None:
        return [f"query_semantic_unchecked_{side}_derivation_missing"]
    check = check_candidate_contract(derivation, _query_contract_from_semantics(query_semantics))
    if check.ok:
        return []
    suffix = ",".join(sorted(check.failure_reasons)) if check.failure_reasons else "unknown"
    return [f"query_semantic_incompatible_{side}:{suffix}"]


def build_compact_behavioral_contrast_v3(
    *,
    derivations: Sequence[Any],
    behavior_classes: Sequence[Any],
    basis: Sequence[Any],
    original_answer: str,
    query_semantics: Optional[Mapping[str, Any]] = None,
) -> CompactBehavioralContrastV3:
    """Build a Round 8 compact contrast from exact fixed-basis behavior classes."""
    derivation_by_id = {str(getattr(item, "derivation_id", "")): item for item in derivations}
    original_classes: List[Any] = []
    alternative_classes: List[Any] = []
    for behavior_class in sorted(behavior_classes, key=lambda item: getattr(item, "class_id", "")):
        derivation = _class_representative_derivation(behavior_class, derivation_by_id)
        if derivation is None:
            continue
        if inference_answers_equivalent(getattr(derivation, "projected_answer", ""), original_answer):
            original_classes.append(behavior_class)
        else:
            alternative_classes.append(behavior_class)

    registry: Dict[str, Any] = {
        "hypothesis_records": [],
        "derivation_records": [],
        "evidence_records": [],
        "intervention_records": [],
    }
    original_hypotheses: List[Dict[str, Any]] = []
    alternative_hypotheses: List[Dict[str, Any]] = []
    selected_original_vector = None
    selected_alternative_vector = None
    missing_behavior_vector_ids: List[str] = []

    for side, classes in (("original", original_classes), ("alternative", alternative_classes)):
        for behavior_class in classes:
            derivation = _class_representative_derivation(behavior_class, derivation_by_id)
            if derivation is None:
                continue
            hypothesis_id = f"H{len(registry['hypothesis_records']) + 1}"
            derivation_ref = f"D{len(registry['derivation_records']) + 1}"
            evidence_refs = _register_v3_evidence(registry, class_id=getattr(behavior_class, "class_id", ""), derivation=derivation)
            vector = _vector_for_derivation(behavior_class, getattr(derivation, "derivation_id", ""))
            if vector is None:
                missing_behavior_vector_ids.append(str(getattr(derivation, "derivation_id", "")))
            response_vector = {}
            if vector is not None:
                response_vector = {
                    f"I{idx}": _response_signature(response)
                    for idx, response in enumerate(getattr(vector, "responses", ()) or [], start=1)
                }
            hypothesis = _hypothesis_v3(
                side=side,
                hypothesis_id=hypothesis_id,
                derivation_ref=derivation_ref,
                behavior_class=behavior_class,
                derivation=derivation,
                evidence_refs=evidence_refs,
                response_vector=response_vector,
            )
            registry["hypothesis_records"].append({
                "hypothesis_id": hypothesis_id,
                "side": side,
                "behavior_class_id": getattr(behavior_class, "class_id", ""),
                "derivation_ref": derivation_ref,
            })
            registry["derivation_records"].append({
                "derivation_ref": derivation_ref,
                "hypothesis_id": hypothesis_id,
                "derivation_id": getattr(derivation, "derivation_id", ""),
                "member_derivation_ids": list(getattr(behavior_class, "member_derivation_ids", ()) or []),
                "executed_answer": getattr(derivation, "projected_answer", ""),
                "answer_key": hypothesis["answer_key"],
                "evidence_refs": evidence_refs,
                "fallback_only": bool((getattr(derivation, "source_candidate", {}) or {}).get("certificate", {}).get("evidence_fallback", False)),
            })
            if side == "original":
                original_hypotheses.append(hypothesis)
                if len(original_classes) == 1:
                    selected_original_vector = vector
            else:
                alternative_hypotheses.append(hypothesis)
                if len(alternative_classes) == 1:
                    selected_alternative_vector = vector

    separating: List[Dict[str, Any]] = []
    common_evaluable = False
    if selected_original_vector is not None and selected_alternative_vector is not None:
        original_responses = list(getattr(selected_original_vector, "responses", ()) or [])
        alternative_responses = list(getattr(selected_alternative_vector, "responses", ()) or [])
        for idx, item in enumerate(basis, start=1):
            original_response = original_responses[idx - 1] if idx <= len(original_responses) else None
            alternative_response = alternative_responses[idx - 1] if idx <= len(alternative_responses) else None
            original_sig = _response_signature(original_response) if original_response is not None else "UNEVALUABLE:missing_response"
            alternative_sig = _response_signature(alternative_response) if alternative_response is not None else "UNEVALUABLE:missing_response"
            original_symbol = getattr(original_response, "response_symbol", "UNEVALUABLE")
            alternative_symbol = getattr(alternative_response, "response_symbol", "UNEVALUABLE")
            evaluable = original_symbol != "UNEVALUABLE" and alternative_symbol != "UNEVALUABLE"
            common_evaluable = common_evaluable or evaluable
            intervention_ref = f"I{idx}"
            is_separating = bool(evaluable and original_sig != alternative_sig)
            record = {
                "intervention_ref": intervention_ref,
                "role_intervention_id": getattr(item, "intervention_id", ""),
                "role": getattr(item, "role", ""),
                "target_schema_ids": list(getattr(item, "target_schema_ids", ()) or []),
                "original_signature": original_sig,
                "alternative_signature": alternative_sig,
                "evaluable_on_both_sides": evaluable,
                "separating": is_separating,
            }
            registry["intervention_records"].append(record)
            if is_separating:
                separating.append(record)
    else:
        for idx, item in enumerate(basis, start=1):
            registry["intervention_records"].append({
                "intervention_ref": f"I{idx}",
                "role_intervention_id": getattr(item, "intervention_id", ""),
                "role": getattr(item, "role", ""),
                "target_schema_ids": list(getattr(item, "target_schema_ids", ()) or []),
                "evaluable_on_both_sides": False,
                "separating": False,
            })

    evidence_ids = {item["evidence_id"] for item in registry["evidence_records"]}
    derivation_refs = {item["derivation_ref"] for item in registry["derivation_records"]}
    intervention_refs = {item["intervention_ref"] for item in registry["intervention_records"]}
    registry_complete = True
    for hypothesis in original_hypotheses + alternative_hypotheses:
        registry_complete = registry_complete and hypothesis["derivation_ref"] in derivation_refs
        registry_complete = registry_complete and set(hypothesis["evidence_refs"]).issubset(evidence_ids)
        registry_complete = registry_complete and set(hypothesis["response_vector"]).issubset(intervention_refs)

    unknowns: List[str] = []
    if not original_hypotheses:
        unknowns.append("no_original_behavior_class")
    if len(original_hypotheses) > 1:
        unknowns.append("multiple_original_behavior_classes")
    if not alternative_hypotheses:
        unknowns.append("no_alternative_behavior_class")
    if len(alternative_hypotheses) > 1:
        unknowns.append("multiple_alternative_behavior_classes")
    if len(original_hypotheses) == 1 and len(alternative_hypotheses) == 1 and not common_evaluable:
        unknowns.append("no_common_evaluable_basis_intervention")
    if len(original_hypotheses) == 1 and len(alternative_hypotheses) == 1 and not separating:
        unknowns.append("no_separating_intervention")
    if any(item.get("fallback_only") for item in registry["derivation_records"]):
        unknowns.append("fallback_only_evidence_required")
    if missing_behavior_vector_ids:
        unknowns.append("missing_behavior_vector")
    if not registry_complete:
        unknowns.append("registry_incomplete")
    if len(original_hypotheses) == 1:
        unknowns.extend(_query_semantic_unknowns_for_hypothesis(
            original_hypotheses[0],
            derivation_by_id,
            query_semantics,
            side="original",
        ))
    if len(alternative_hypotheses) == 1:
        unknowns.extend(_query_semantic_unknowns_for_hypothesis(
            alternative_hypotheses[0],
            derivation_by_id,
            query_semantics,
            side="alternative",
        ))

    constructible = bool(original_hypotheses and alternative_hypotheses)
    compact = bool(
        constructible
        and registry_complete
        and len(original_hypotheses) == 1
        and len(alternative_hypotheses) == 1
    )
    repair_eligible = bool(compact and common_evaluable and separating and not sorted(set(unknowns)))
    states = {
        "contrast_constructible": constructible,
        "contrast_registry_complete": registry_complete,
        "contrast_compact": compact,
        "repair_eligible": repair_eligible,
    }
    return CompactBehavioralContrastV3(
        contrast_version=ROUND8_COMPACT_CONTRAST_VERSION,
        states=states,
        query_semantics=dict(query_semantics or {}),
        original_hypothesis=original_hypotheses[0] if len(original_hypotheses) == 1 else {},
        alternative_hypothesis=alternative_hypotheses[0] if len(alternative_hypotheses) == 1 else {},
        alternative_hypotheses=alternative_hypotheses,
        separating_interventions=separating,
        unknowns=sorted(set(unknowns)),
        registry=registry,
        construction_trace=[
            f"behavior_classes={len(behavior_classes)}",
            f"original_behavior_classes={len(original_hypotheses)}",
            f"alternative_behavior_classes={len(alternative_hypotheses)}",
            f"basis_size={len(basis)}",
            f"separating_interventions={len(separating)}",
        ],
    )


def build_minimal_contrast_set(
    *,
    lattice: DerivationLatticeAudit,
    original_support_hypothesis_set: OriginalSupportHypothesisSet,
) -> MinimalContrastSet:
    classes = _class_by_id(lattice)
    original_class_ids = sorted(
        item.class_id for item in lattice.quotient_classes
        if item.original_support_members
    )
    original_support_absent = not original_class_ids
    alternative_class_ids = sorted(
        item.class_id for item in lattice.quotient_classes
        if item.class_id not in set(original_class_ids) and _eligible_alternative(item)
    )
    alternative_answer_keys = {
        classes[class_id].answer_key
        for class_id in alternative_class_ids
        if class_id in classes
    }

    retained_ids = original_class_ids + alternative_class_ids
    retained_atoms = {
        class_id: _class_support_atoms(classes[class_id])
        for class_id in retained_ids
        if class_id in classes
    }
    if retained_atoms:
        atom_sets = list(retained_atoms.values())
        shared = sorted(set.intersection(*atom_sets)) if atom_sets else []
        union = set.union(*atom_sets)
        distinguishing = sorted(union - set(shared))
    else:
        shared = []
        distinguishing = []

    interventions: List[Dict[str, Any]] = []
    seen_interventions: set[tuple[str, str, str]] = set()
    for class_id in retained_ids:
        quotient_class = classes.get(class_id)
        if quotient_class is None:
            continue
        for observation in _class_interventions(quotient_class):
            key = (
                str(observation.get("class_id", "")),
                str(observation.get("intervention_basis_id", "")),
                str(observation.get("response_symbol", "")),
                str(observation.get("failure_reason", "")),
            )
            if key in seen_interventions:
                continue
            seen_interventions.add(key)
            interventions.append(observation)

    unresolved: List[str] = []
    support_payload = original_support_hypothesis_set.to_dict()
    support_hypotheses = support_payload.get("hypotheses") or []
    if len(original_class_ids) > 1:
        unresolved.append("original_answer_supported_by_multiple_classes")
    if len(support_hypotheses) > 1:
        unresolved.append("original_answer_supported_by_multiple_programs")
    if not original_class_ids:
        unresolved.append("original_support_absent")
    if len(alternative_answer_keys) > 1:
        unresolved.append("multiple_alternative_answer_classes")
    if not alternative_class_ids:
        unresolved.append("no_alternative_contrast_class")

    trace = [
        f"lattice_members={len(lattice.members)}",
        f"quotient_classes={len(lattice.quotient_classes)}",
        f"original_classes={len(original_class_ids)}",
        f"alternative_classes={len(alternative_class_ids)}",
        f"alternative_answer_classes={len(alternative_answer_keys)}",
    ]
    if original_support_absent:
        trace.append("represented_explicit_original_support_absence")
    ready = (
        (bool(original_class_ids) or original_support_absent)
        and bool(alternative_class_ids)
        and all(classes[class_id].roundtrip_valid for class_id in alternative_class_ids if class_id in classes)
        and bool(interventions)
    )
    return MinimalContrastSet(
        contrast_version=ROUND6_CONTRAST_VERSION,
        original_classes=original_class_ids,
        alternative_classes=alternative_class_ids,
        shared_evidence_atoms=shared,
        distinguishing_evidence_atoms=distinguishing,
        intervention_observations=interventions,
        unresolved_ambiguities=sorted(set(unresolved)),
        construction_trace=trace,
        ready_for_cera=ready,
        original_support_absent=original_support_absent,
        retained_class_count=len(retained_ids),
        retained_alternative_answer_class_count=len(alternative_answer_keys),
    )
