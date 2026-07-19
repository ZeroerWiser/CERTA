"""Serializable typed derivation records for CERTA Round 4."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from certa.reproducibility.canonical_json import canonicalize
from certa.operations.contracts import FINAL_SUPPORTED_OPERATIONS


OPERATION_FAMILIES = {*FINAL_SUPPORTED_OPERATIONS, "UNKNOWN"}

PROJECTION_OPERATORS = {
    "VALUE_PROJECTION",
    "ROW_ENTITY_PROJECTION",
    "COLUMN_ENTITY_PROJECTION",
    "BOOLEAN_PROJECTION",
    "SCALAR_RESULT_PROJECTION",
    "UNKNOWN",
}

ANSWER_DOMAINS = {"ENTITY", "SCALAR", "SET", "BOOLEAN", "INTERVAL", "UNKNOWN"}


def to_jsonable(value: Any) -> Any:
    return canonicalize(value)


@dataclass
class PreEvidenceQueryContract:
    question: str
    answer_domain: str = "UNKNOWN"
    allowed_answer_domains: List[str] = field(default_factory=lambda: ["UNKNOWN"])
    allowed_projection_operators: List[str] = field(default_factory=lambda: ["UNKNOWN"])
    candidate_independent_operation_hypotheses: List[str] = field(default_factory=list)
    unit_or_scale_constraints: List[str] = field(default_factory=list)
    initial_answer_surface_type: str = "unknown"
    field_provenance: Dict[str, str] = field(default_factory=dict)
    unknown_fields: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(self)


@dataclass
class CandidateContractCheck:
    derivation_id: str
    candidate_id: str
    answer_domain_ok: bool
    projection_operator_ok: bool
    operation_family_ok: bool
    ambiguous_contract: bool = False
    failure_reasons: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.answer_domain_ok and self.projection_operator_ok and self.operation_family_ok

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(self)


@dataclass
class ExecutableDerivation:
    derivation_id: str
    source_candidate_id: str
    operation_family: str
    operand_node_ids: List[str]
    required_edge_triples: List[Tuple[str, str, str]]
    typed_signature: str
    projection_operator: str
    projected_answer: str
    output_domain: str
    evidence_ids: List[str]
    executable_program: str
    provenance_complete: bool
    availability: str
    failure_reasons: List[str] = field(default_factory=list)
    operand_metadata: List[Dict[str, Any]] = field(default_factory=list)
    source_candidate: Dict[str, Any] = field(default_factory=dict)
    operation_metadata: Dict[str, Any] = field(default_factory=dict)
    comparison_polarity: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(self)


@dataclass
class CandidateAdmissibilityResult:
    derivation: ExecutableDerivation
    candidate_id: str
    admissible: bool
    contract_check: CandidateContractCheck
    failure_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(self)


@dataclass
class AdmissibleCandidateSet:
    pre_evidence_contract: PreEvidenceQueryContract
    derivations: List[ExecutableDerivation]
    results: List[CandidateAdmissibilityResult]
    admissible_derivations: List[ExecutableDerivation]
    projected_answer_classes: Dict[str, List[str]]
    review_eligible: bool
    selected_derivation_id: str = ""
    selected_candidate_id: str = ""
    ambiguity_count: int = 0
    reject_reason: str = ""
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(self)


@dataclass
class ReplayResult:
    intervention_id: str
    derivation_id: str
    intervention_type: str
    replay_mode: str
    pre_projected_answer: str
    post_projected_answer: Optional[str]
    operation_executed: bool
    projection_executed: bool
    required_nodes_valid: bool
    required_edges_valid: bool
    operand_resolution_valid: bool
    changed: bool
    available: bool
    failure_reason: str = ""
    missing_node_ids: List[str] = field(default_factory=list)
    missing_edge_triples: List[Tuple[str, str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(self)

    def to_observed_effect(self) -> Dict[str, Any]:
        return {
            "available": self.available,
            "executor": "typed_same_derivation_replay",
            "replay_mode": self.replay_mode,
            "candidate_specific": True,
            "derivation_id": self.derivation_id,
            "intervention_id": self.intervention_id,
            "intervention_type": self.intervention_type,
            "pre_projected_answer": self.pre_projected_answer,
            "post_projected_answer": self.post_projected_answer,
            "operation_executed": self.operation_executed,
            "projection_executed": self.projection_executed,
            "required_nodes_valid": self.required_nodes_valid,
            "required_edges_valid": self.required_edges_valid,
            "operand_resolution_valid": self.operand_resolution_valid,
            "changed": self.changed,
            "failure_reason": self.failure_reason,
            "missing_node_ids": self.missing_node_ids,
            "missing_edge_triples": to_jsonable(self.missing_edge_triples),
        }


@dataclass
class OriginalSupportHypothesis:
    hypothesis_id: str
    source: str
    derivation_id: Optional[str]
    projected_answer: str
    answer_equivalent: bool
    operation_family: str
    operand_node_ids: List[str]
    evidence_ids: List[str]
    required_edge_triples: List[Tuple[str, str, str]]
    projection_operator: str
    provenance_complete: bool
    availability: str
    failure_reasons: List[str] = field(default_factory=list)
    support_level: str = "UNAVAILABLE"

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(self)


@dataclass
class OriginalSupportHypothesisSet:
    original_answer: str
    hypotheses: List[OriginalSupportHypothesis]
    ambiguity_count: int
    contains_executable_derivation: bool
    contains_graph_anchor_only: bool
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(self)
