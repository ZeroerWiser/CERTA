"""Original Support Symmetry v3 metadata for Round 6."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .lattice import DerivationLatticeAudit, DerivationQuotientClass
from .schema import OriginalSupportHypothesisSet, to_jsonable


ROUND6_SUPPORT_SYMMETRY_VERSION = "original_support_symmetry_v3"


@dataclass
class EvidenceRegistry:
    evidence_atoms: List[Dict[str, Any]] = field(default_factory=list)
    derivation_records: List[Dict[str, Any]] = field(default_factory=list)
    hypothesis_records: List[Dict[str, Any]] = field(default_factory=list)
    intervention_records: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(self)


@dataclass
class OriginalSupportSymmetryV3:
    support_version: str
    original_answer: str
    hypotheses: List[Dict[str, Any]]
    ambiguity_count: int
    contains_executable_roundtrip_support: bool
    contains_graph_anchor_only: bool
    support_level_distribution: Dict[str, int]
    evidence_registry: EvidenceRegistry
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(self)


def _legacy_support_level_by_derivation(
    support_set: OriginalSupportHypothesisSet,
) -> Dict[str, str]:
    levels: Dict[str, str] = {}
    for hypothesis in support_set.hypotheses:
        if hypothesis.derivation_id:
            levels[str(hypothesis.derivation_id)] = str(hypothesis.support_level or "UNAVAILABLE")
    return levels


def _support_level(
    quotient_class: DerivationQuotientClass,
    *,
    legacy_levels: Dict[str, str],
) -> str:
    if quotient_class.roundtrip_valid:
        return "EXECUTABLE_ROUNDTRIP_VALID"
    for derivation_id in quotient_class.original_support_members:
        level = legacy_levels.get(derivation_id)
        if level in {"EXECUTABLE_RECONSTRUCTED", "EXECUTABLE_EXISTING", "GRAPH_ANCHORED"}:
            return level
    return "UNAVAILABLE"


def _intervention_ids(quotient_class: DerivationQuotientClass, registry: EvidenceRegistry) -> List[str]:
    ids: List[str] = []
    if quotient_class.intervention_observations:
        for observation in quotient_class.intervention_observations:
            intervention_id = f"I{len(registry.intervention_records) + 1}"
            payload = dict(observation)
            payload.update({
                "intervention_id": intervention_id,
                "class_id": quotient_class.class_id,
            })
            registry.intervention_records.append(payload)
            ids.append(intervention_id)
        return ids
    for signature in quotient_class.intervention_signatures:
        for raw in str(signature or "").split("|"):
            if not raw or "=" not in raw:
                continue
            label, response = raw.split("=", 1)
            intervention_id = f"I{len(registry.intervention_records) + 1}"
            registry.intervention_records.append({
                "intervention_id": intervention_id,
                "class_id": quotient_class.class_id,
                "basis": label,
                "response_symbol": response,
            })
            ids.append(intervention_id)
    return ids


def _register_evidence_atoms(
    quotient_class: DerivationQuotientClass,
    registry: EvidenceRegistry,
) -> List[str]:
    ids: List[str] = []
    atom_payloads = []
    for node_id in quotient_class.support_evidence_ids:
        atom_payloads.append(("support_node", node_id))
    for node_id in quotient_class.projection_endpoint_ids:
        atom_payloads.append(("projection_endpoint", node_id))
    for role in quotient_class.required_edge_roles:
        atom_payloads.append(("required_edge_role", role))
    for atom_type, value in sorted(set(atom_payloads)):
        evidence_id = f"E{len(registry.evidence_atoms) + 1}"
        registry.evidence_atoms.append({
            "evidence_id": evidence_id,
            "class_id": quotient_class.class_id,
            "atom_type": atom_type,
            "value": value,
        })
        ids.append(evidence_id)
    return ids


def build_original_support_symmetry_v3(
    *,
    original_answer: str,
    lattice: DerivationLatticeAudit,
    original_support_hypothesis_set: OriginalSupportHypothesisSet,
) -> OriginalSupportSymmetryV3:
    legacy_levels = _legacy_support_level_by_derivation(original_support_hypothesis_set)
    registry = EvidenceRegistry()
    hypotheses: List[Dict[str, Any]] = []
    for quotient_class in lattice.quotient_classes:
        if not quotient_class.original_support_members:
            continue
        support_evidence_ids = _register_evidence_atoms(quotient_class, registry)
        intervention_ids = _intervention_ids(quotient_class, registry)
        support_level = _support_level(quotient_class, legacy_levels=legacy_levels)
        hypothesis_id = f"H{len(hypotheses) + 1}"
        derivation_ids = list(quotient_class.original_support_members)
        registry.derivation_records.extend({
            "derivation_record_id": f"DREG{len(registry.derivation_records) + 1}",
            "hypothesis_id": hypothesis_id,
            "derivation_id": derivation_id,
            "class_id": quotient_class.class_id,
        } for derivation_id in derivation_ids)
        ambiguity_notes: List[str] = []
        if len(quotient_class.operation_families) > 1:
            ambiguity_notes.append("multiple_operation_families")
        if len(quotient_class.representative_ids) > 1:
            ambiguity_notes.append("multiple_non_dominated_representatives")
        hypothesis = {
            "hypothesis_id": hypothesis_id,
            "answer_class_id": quotient_class.class_id,
            "derivation_ids": derivation_ids,
            "representative_derivation_ids": list(quotient_class.representative_ids),
            "support_evidence_ids": support_evidence_ids,
            "projection_endpoint_ids": list(quotient_class.projection_endpoint_ids),
            "operation_families": list(quotient_class.operation_families),
            "provenance_state": ",".join(quotient_class.provenance_states),
            "support_level": support_level,
            "intervention_signature_ids": intervention_ids,
            "ambiguity_notes": ambiguity_notes,
        }
        hypotheses.append(hypothesis)
        registry.hypothesis_records.append({
            "hypothesis_id": hypothesis_id,
            "answer_class_id": quotient_class.class_id,
            "support_level": support_level,
        })

    if not hypotheses and original_support_hypothesis_set.contains_graph_anchor_only:
        hypothesis_id = "H1"
        evidence_ids = []
        for legacy in original_support_hypothesis_set.hypotheses:
            for node_id in legacy.operand_node_ids:
                evidence_id = f"E{len(registry.evidence_atoms) + 1}"
                registry.evidence_atoms.append({
                    "evidence_id": evidence_id,
                    "atom_type": "graph_anchor",
                    "value": node_id,
                })
                evidence_ids.append(evidence_id)
        hypotheses.append({
            "hypothesis_id": hypothesis_id,
            "answer_class_id": "",
            "derivation_ids": [],
            "representative_derivation_ids": [],
            "support_evidence_ids": evidence_ids,
            "projection_endpoint_ids": [],
            "operation_families": ["UNKNOWN"],
            "provenance_state": "GRAPH_ANCHOR_ONLY",
            "support_level": "GRAPH_ANCHORED",
            "intervention_signature_ids": [],
            "ambiguity_notes": ["not_executable_derivation"],
        })
        registry.hypothesis_records.append({
            "hypothesis_id": hypothesis_id,
            "answer_class_id": "",
            "support_level": "GRAPH_ANCHORED",
        })

    support_levels: Dict[str, int] = {}
    for hypothesis in hypotheses:
        level = str(hypothesis.get("support_level", "UNAVAILABLE"))
        support_levels[level] = support_levels.get(level, 0) + 1
    notes = list(original_support_hypothesis_set.notes)
    if not hypotheses:
        notes.append("no_original_support_symmetry_v3_hypothesis")
    return OriginalSupportSymmetryV3(
        support_version=ROUND6_SUPPORT_SYMMETRY_VERSION,
        original_answer=str(original_answer or ""),
        hypotheses=hypotheses,
        ambiguity_count=max(0, len(hypotheses) - 1),
        contains_executable_roundtrip_support=any(
            hypothesis.get("support_level") == "EXECUTABLE_ROUNDTRIP_VALID"
            for hypothesis in hypotheses
        ),
        contains_graph_anchor_only=any(
            hypothesis.get("support_level") == "GRAPH_ANCHORED"
            for hypothesis in hypotheses
        ),
        support_level_distribution=dict(sorted(support_levels.items())),
        evidence_registry=registry,
        notes=notes,
    )
