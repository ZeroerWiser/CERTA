"""Serializable evidence and CERA result structures.

These dataclasses are intentionally plain. They carry evidence, certificate,
counterfactual, and validator facts without importing the main CSCR pipeline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, is_dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional

from certa.reproducibility.canonical_json import canonical_json, canonicalize


PACKET_VERSION = "certa_e71_packet_v2"
CERA_OUTPUT_VERSION = "cera_json_v2"


def to_jsonable(value: Any) -> Any:
    """Convert dataclasses, enums, and containers into JSON-friendly values."""
    return canonicalize(value)


def compact_json(value: Any) -> str:
    return canonical_json(value)


def pretty_json(value: Any) -> str:
    return canonical_json(value, pretty=True)


def _as_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value):
        return to_jsonable(value)
    return {}


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


@dataclass
class EvidenceState:
    """Gold-free evidence state observed by CERA."""

    question: str
    graph_stats: Dict[str, Any] = field(default_factory=dict)
    question_frame: Dict[str, Any] = field(default_factory=dict)
    edge_reliability_diag: Dict[str, Any] = field(default_factory=dict)
    layout_risk: float = 0.0
    evidence_ids: List[str] = field(default_factory=list)
    anchor_nodes: List[str] = field(default_factory=list)
    evidence_nodes: List[Dict[str, Any]] = field(default_factory=list)
    evidence_edges: List[Dict[str, Any]] = field(default_factory=list)
    certificate_overview: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvidenceState":
        return cls(
            question=str(payload.get("question", "")),
            graph_stats=_as_dict(payload.get("graph_stats")),
            question_frame=_as_dict(payload.get("question_frame")),
            edge_reliability_diag=_as_dict(payload.get("edge_reliability_diag")),
            layout_risk=float(payload.get("layout_risk", 0.0) or 0.0),
            evidence_ids=[str(x) for x in _as_list(payload.get("evidence_ids"))],
            anchor_nodes=[str(x) for x in _as_list(payload.get("anchor_nodes"))],
            evidence_nodes=[_as_dict(x) for x in _as_list(payload.get("evidence_nodes"))],
            evidence_edges=[_as_dict(x) for x in _as_list(payload.get("evidence_edges"))],
            certificate_overview=_as_dict(payload.get("certificate_overview")),
        )


@dataclass
class CertifiedCandidateFull:
    """Full candidate certificate record exposed additively by the calibrator."""

    candidate_id: str
    denotation: str
    normalized_denotation: str
    operation: str
    priority: Optional[int] = None
    cells_used: List[Dict[str, Any]] = field(default_factory=list)
    computation_trace: str = ""
    operation_metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    certificate: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CertifiedCandidateFull":
        return cls(
            candidate_id=str(payload.get("candidate_id", "")),
            denotation=str(payload.get("denotation", "")),
            normalized_denotation=str(payload.get("normalized_denotation", "")),
            operation=str(payload.get("operation", "")),
            priority=payload.get("priority"),
            cells_used=[_as_dict(x) for x in _as_list(payload.get("cells_used"))],
            computation_trace=str(payload.get("computation_trace", "")),
            operation_metadata=_as_dict(payload.get("operation_metadata")),
            source=str(payload.get("source", "")),
            certificate=_as_dict(payload.get("certificate")),
        )


@dataclass
class SupportChainElement:
    """A table/evidence support item cited by CERA as S1, S2, ..."""

    evidence_id: str
    node_id: str
    row: Optional[int]
    col: Optional[int]
    cell_value: str
    row_headers: List[str] = field(default_factory=list)
    col_headers: List[str] = field(default_factory=list)
    support_role: str = "candidate_cell"
    provenance: str = "executor_cell"
    edge_path: List[Dict[str, Any]] = field(default_factory=list)
    certificate_link: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SupportChainElement":
        return cls(
            evidence_id=str(payload.get("evidence_id", "")),
            node_id=str(payload.get("node_id", "")),
            row=payload.get("row"),
            col=payload.get("col"),
            cell_value=str(payload.get("cell_value", "")),
            row_headers=[str(x) for x in _as_list(payload.get("row_headers"))],
            col_headers=[str(x) for x in _as_list(payload.get("col_headers"))],
            support_role=str(payload.get("support_role", "candidate_cell")),
            provenance=str(payload.get("provenance", "executor_cell")),
            edge_path=[_as_dict(x) for x in _as_list(payload.get("edge_path"))],
            certificate_link=_as_dict(payload.get("certificate_link")),
        )


@dataclass
class CounterfactualChainElement:
    """A counterfactual intervention and its observed executor effect."""

    cf_id: str
    intervention_type: str
    removed_nodes: List[str] = field(default_factory=list)
    removed_edges: List[Dict[str, Any]] = field(default_factory=list)
    modified_nodes: List[str] = field(default_factory=list)
    expected_effect: str = ""
    observed_effect: Dict[str, Any] = field(default_factory=dict)
    causal_interpretation: str = ""
    flags: Dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CounterfactualChainElement":
        return cls(
            cf_id=str(payload.get("cf_id", "")),
            intervention_type=str(payload.get("intervention_type", "")),
            removed_nodes=[str(x) for x in _as_list(payload.get("removed_nodes"))],
            removed_edges=[_as_dict(x) for x in _as_list(payload.get("removed_edges"))],
            modified_nodes=[str(x) for x in _as_list(payload.get("modified_nodes"))],
            expected_effect=str(payload.get("expected_effect", "")),
            observed_effect=_as_dict(payload.get("observed_effect")),
            causal_interpretation=str(payload.get("causal_interpretation", "")),
            flags=_as_dict(payload.get("flags")),
            description=str(payload.get("description", "")),
        )


@dataclass
class CausalEvidencePacket:
    """Gold-free evidence packet given to CERA."""

    question: str
    original_answer: str
    candidate_under_review: str
    candidate: CertifiedCandidateFull
    evidence_state: EvidenceState
    query_contract: Dict[str, Any] = field(default_factory=dict)
    original_certificate_available: bool = False
    original_equivalent_candidate_id: str = ""
    original_certificate: Dict[str, Any] = field(default_factory=dict)
    original_support_chain: List[SupportChainElement] = field(default_factory=list)
    original_support_chain_notes: List[str] = field(default_factory=list)
    original_support_hypothesis_set: Dict[str, Any] = field(default_factory=dict)
    support_chain: List[SupportChainElement] = field(default_factory=list)
    counterfactual_chain: List[CounterfactualChainElement] = field(default_factory=list)
    admissible_candidate_set: Dict[str, Any] = field(default_factory=dict)
    reviewed_derivation: Dict[str, Any] = field(default_factory=dict)
    derivation_lattice: Dict[str, Any] = field(default_factory=dict)
    minimal_contrast_set: Dict[str, Any] = field(default_factory=dict)
    compact_behavioral_contrast_v2: Dict[str, Any] = field(default_factory=dict)
    compact_behavioral_contrast_v3: Dict[str, Any] = field(default_factory=dict)
    original_support_symmetry_v3: Dict[str, Any] = field(default_factory=dict)
    evidence_semantic_statements: List[Dict[str, Any]] = field(default_factory=list)
    table_excerpt: List[Dict[str, Any]] = field(default_factory=list)
    packet_version: str = PACKET_VERSION
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(self)

    def to_json(self, *, pretty: bool = False) -> str:
        return pretty_json(self) if pretty else compact_json(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CausalEvidencePacket":
        candidate = CertifiedCandidateFull.from_dict(_as_dict(payload.get("candidate")))
        evidence_state = EvidenceState.from_dict(_as_dict(payload.get("evidence_state")))
        return cls(
            question=str(payload.get("question", "")),
            original_answer=str(payload.get("original_answer", "")),
            candidate_under_review=str(payload.get("candidate_under_review", "")),
            candidate=candidate,
            evidence_state=evidence_state,
            query_contract=_as_dict(payload.get("query_contract")),
            original_certificate_available=bool(payload.get("original_certificate_available", False)),
            original_equivalent_candidate_id=str(payload.get("original_equivalent_candidate_id", "")),
            original_certificate=_as_dict(payload.get("original_certificate")),
            original_support_chain=[
                SupportChainElement.from_dict(_as_dict(x))
                for x in _as_list(payload.get("original_support_chain"))
            ],
            original_support_chain_notes=[str(x) for x in _as_list(payload.get("original_support_chain_notes"))],
            original_support_hypothesis_set=_as_dict(payload.get("original_support_hypothesis_set")),
            support_chain=[
                SupportChainElement.from_dict(_as_dict(x))
                for x in _as_list(payload.get("support_chain"))
            ],
            counterfactual_chain=[
                CounterfactualChainElement.from_dict(_as_dict(x))
                for x in _as_list(payload.get("counterfactual_chain"))
            ],
            admissible_candidate_set=_as_dict(payload.get("admissible_candidate_set")),
            reviewed_derivation=_as_dict(payload.get("reviewed_derivation")),
            derivation_lattice=_as_dict(payload.get("derivation_lattice")),
            minimal_contrast_set=_as_dict(payload.get("minimal_contrast_set")),
            compact_behavioral_contrast_v2=_as_dict(payload.get("compact_behavioral_contrast_v2")),
            compact_behavioral_contrast_v3=_as_dict(payload.get("compact_behavioral_contrast_v3")),
            original_support_symmetry_v3=_as_dict(payload.get("original_support_symmetry_v3")),
            evidence_semantic_statements=[_as_dict(x) for x in _as_list(payload.get("evidence_semantic_statements"))],
            table_excerpt=[_as_dict(x) for x in _as_list(payload.get("table_excerpt"))],
            packet_version=str(payload.get("packet_version", PACKET_VERSION)),
            metadata=_as_dict(payload.get("metadata")),
        )

    @classmethod
    def from_json(cls, payload: str) -> "CausalEvidencePacket":
        return cls.from_dict(json.loads(payload))


@dataclass
class CERAOutput:
    """Parsed JSON response from CERA."""

    decision: str
    final_answer: str = ""
    chosen_hypothesis_id: str = ""
    self_assessed_confidence: Optional[float] = None
    query_contract_check: Any = field(default_factory=dict)
    query_semantics_assessment: Any = field(default_factory=dict)
    original_defense: Any = field(default_factory=dict)
    original_assessment: Any = field(default_factory=dict)
    candidate_case: Any = field(default_factory=dict)
    alternative_assessment: Any = field(default_factory=dict)
    counterfactual_assessment: Any = field(default_factory=dict)
    separating_intervention_refs: Any = field(default_factory=list)
    blocking_unknowns: Any = field(default_factory=list)
    uncertainty_assessment: Any = field(default_factory=dict)
    derivation_program: Any = field(default_factory=dict)
    rationale: Any = ""
    safety_notes: Any = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)
    output_version: str = CERA_OUTPUT_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CERAOutput":
        confidence = payload.get("self_assessed_confidence", payload.get("confidence"))
        try:
            confidence = None if confidence is None else float(confidence)
        except (TypeError, ValueError):
            confidence = None
        return cls(
            decision=str(payload.get("decision", "")),
            final_answer=str(payload.get("final_answer", "")),
            chosen_hypothesis_id=str(payload.get("chosen_hypothesis_id", "")),
            self_assessed_confidence=confidence,
            query_contract_check=payload.get("query_contract_check", {}),
            query_semantics_assessment=payload.get("query_semantics_assessment", {}),
            original_defense=payload.get("original_defense", {}),
            original_assessment=payload.get("original_assessment", {}),
            candidate_case=payload.get("candidate_case", {}),
            alternative_assessment=payload.get("alternative_assessment", {}),
            counterfactual_assessment=payload.get("counterfactual_assessment", {}),
            separating_intervention_refs=payload.get("separating_intervention_refs", []),
            blocking_unknowns=payload.get("blocking_unknowns", []),
            uncertainty_assessment=payload.get("uncertainty_assessment", {}),
            derivation_program=payload.get("derivation_program", {}),
            rationale=payload.get("rationale", ""),
            safety_notes=payload.get("safety_notes", []),
            raw=dict(payload),
            output_version=str(payload.get("output_version", CERA_OUTPUT_VERSION)),
        )


@dataclass
class CERACommitResult:
    """Shadow or future commit result from CERA orchestration."""

    enabled: bool = False
    packet_built: bool = False
    stage: str = ""
    triggered: bool = False
    shadow_only: bool = True
    final_committed: bool = False
    reject_reason: str = ""
    evidence_packet_hash: str = ""
    support_chain_len: int = 0
    counterfactual_chain_len: int = 0
    candidate_scci: Optional[float] = None
    candidate_effective_coverage: Optional[float] = None
    legacy_heuristic_usage_count: int = 0
    llm_called: bool = False
    json_parse_success: bool = False
    validator_accept: bool = False
    validator_reject_reason: str = ""
    would_commit: bool = False
    would_keep: bool = False
    insufficient: bool = False
    unsafe_accept: bool = False
    runtime_error: str = ""
    packet: Optional[CausalEvidencePacket] = None
    prompt: str = ""
    raw_response: str = ""
    output: Optional[CERAOutput] = None
    validator: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(self)

    def to_prediction_fields(
        self,
        *,
        log_full_prompt: bool = False,
        log_evidence_packet: bool = True,
    ) -> Dict[str, Any]:
        fields: Dict[str, Any] = {
            "cera_enabled": self.enabled,
            "cera_packet_built": self.packet_built,
            "cera_stage": self.stage,
            "cera_triggered": self.triggered,
            "cera_shadow_only": self.shadow_only,
            "cera_final_committed": self.final_committed,
            "cera_reject_reason": self.reject_reason,
            "cera_evidence_packet_hash": self.evidence_packet_hash,
            "cera_support_chain_len": self.support_chain_len,
            "cera_counterfactual_chain_len": self.counterfactual_chain_len,
            "cera_candidate_scci": self.candidate_scci,
            "cera_candidate_effective_coverage": self.candidate_effective_coverage,
            "legacy_heuristic_usage_count": self.legacy_heuristic_usage_count,
            "cera_llm_called": self.llm_called,
            "cera_json_parse_success": self.json_parse_success,
            "cera_validator_accept": self.validator_accept,
            "cera_validator_reject_reason": self.validator_reject_reason,
            "cera_would_commit": self.would_commit,
            "cera_would_keep": self.would_keep,
            "cera_insufficient": self.insufficient,
            "cera_unsafe_accept": self.unsafe_accept,
            "cera_runtime_error": self.runtime_error,
            "cera_validator": self.validator,
        }
        for key in (
            "cera_original_answer",
            "cera_candidate_under_review",
            "cera_proposed_repair_answer",
            "cera_packet_token_length",
            "cera_packet_hash",
            "cera_query_contract_hash",
            "cera_query_contract_pre_evidence",
            "cera_prompt_hash",
            "cera_request_hash",
            "cera_model",
            "cera_backend",
            "cera_api_base_url",
            "cera_sampling",
            "cera_api_cache_hit",
            "cera_latency_seconds",
            "cera_input_tokens",
            "cera_output_tokens",
            "cera_request_audit",
            "cera_original_certificate_available",
            "cera_original_support_hypothesis_count",
            "cera_original_support_executable",
            "cera_candidate_counterfactual_available_count",
            "cera_candidate_observed_counterfactual_count",
            "cera_outside_evidence_support_count",
            "cera_row_major_context_cell_count",
            "cera_allow_row_major_context",
            "cera_derivation_count",
            "cera_admissible_derivation_count",
            "cera_projected_answer_class_count",
            "cera_review_eligible",
            "cera_admissibility_reject_reason",
            "cera_reviewed_derivation_id",
            "cera_reviewed_derivation_replay_mode",
            "cera_candidate_derivation_count",
            "cera_planner_enabled",
            "cera_planner_called",
            "cera_planner_skipped_reason",
            "cera_planner_view_version",
            "cera_planner_schema_node_count",
            "cera_planner_schema_edge_count",
            "cera_planner_prompt_hash",
            "cera_planner_view_hash",
            "cera_planner_request_hash",
            "cera_planner_model",
            "cera_planner_backend",
            "cera_planner_api_base_url",
            "cera_planner_sampling",
            "cera_planner_api_cache_hit",
            "cera_planner_api_cache_mode",
            "cera_planner_latency_seconds",
            "cera_planner_input_tokens",
            "cera_planner_output_tokens",
            "cera_planner_request_audit",
            "cera_planner_raw_output_hash",
            "cera_planner_raw_output",
            "cera_planner_generation_error",
            "cera_planner_parse_ok",
            "cera_planner_validation_ok",
            "cera_planner_validation_errors",
            "cera_planner_valid_plan_count",
            "cera_planner_plan_rejection_count",
            "cera_planner_plan_rejections",
            "cera_planner_resource_warnings",
            "cera_planner_derivation_count",
            "cera_planner_compile_failure_count",
            "cera_planner_compile_failures",
            "cera_planner_boundary_condition",
            "cera_planner_proposal_visible_to_planner",
            "cera_planner_table_values_visible_to_planner",
            "cera_planner_boundary_ablation_arm",
            "cera_planner_contract_version",
            "cera_planner_legacy_query_semantics_mode",
            "cera_planner_legacy_query_semantics_public",
            "cera_planner_reference_domain_count",
            "cera_planner_reference_domain_hash",
            "cera_planner_constraint_schema_hash",
            "cera_planner_structured_output_mechanism",
            "cera_planner_structured_output_schema_hash",
            "cera_planner_constraint_fallback_used",
            "cera_planner_invalid_generated_reference_count",
            "cera_query_semantic_source",
            "cera_legacy_question_frame_used",
            "cera_question_frame_operator",
            "cera_question_frame_polarity",
            "cera_allowed_operation_hypotheses",
            "cera_allowed_answer_domains",
            "cera_allowed_projection_operators",
            "cera_query_semantic_rejection_reasons",
            "cera_round10_closure_audit_version",
            "cera_round11_closure_audit_version",
            "cera_round10_closure_audit_records",
            "cera_round12_semantic_type_audit_version",
            "cera_round12_semantic_type_audit_records",
            "cera_round11_closure_resource_complete",
            "cera_round11_closure_declared_assignment_count",
            "cera_round11_closure_realized_assignment_count",
            "cera_round12_trace_version",
            "cera_round12_trace_planner_call_count",
            "cera_round12_trace_intent_parse_ok",
            "cera_round12_trace_intent_validation_ok",
            "cera_round12_trace_intent_validation_errors",
            "cera_round12_trace_intent_count",
            "cera_round12_trace_intent_hypotheses",
            "cera_round12_trace_role_parse_ok",
            "cera_round12_trace_role_validation_ok",
            "cera_round12_trace_role_validation_errors",
            "cera_round12_trace_role_step_count",
            "cera_round12_trace_role_steps",
            "cera_round12_trace_count",
            "cera_round12_trace_executable_count",
            "cera_round12_trace_resource_complete",
            "cera_round12_trace_records",
            "cera_round12_trace_fvf_records",
            "cera_round12_trace_fvf_stage_counts",
            "cera_round12_trace_validation_failure_count",
            "cera_round12_trace_request_audits",
            "cera_round12_trace_raw_outputs",
            "cera_round12_patch_shadow_enabled",
            "cera_round12_patch_version",
            "cera_round12_patch_eligible_source_trace_count",
            "cera_round12_patch_local_domain_count",
            "cera_round12_patch_candidate_records",
            "cera_round12_patch_minimal_executable_count",
            "cera_round12_patch_minimal_records",
            "cera_round12_patch_model_calls",
            "cera_frontier_derivation_count",
            "cera_symmetric_frontier_enabled",
            "cera_round8b_audit_derivation_count",
            "cera_round8b_audit_candidate_derivation_count",
            "cera_round8b_audit_planner_derivation_count",
            "cera_round8b_audit_frontier_derivation_count",
            "cera_round8b_decision_source_policy",
            "cera_round8b_decision_derivation_count",
            "cera_round8b_decision_original_derivation_count",
            "cera_round8b_decision_alternative_derivation_count",
            "cera_round8b_decision_rejected_derivation_count",
            "cera_round8b_decision_rejections",
            "cera_round6_contract_version",
            "cera_lattice_stage_counts",
            "cera_lattice_member_count",
            "cera_lattice_l1_roundtrip_valid_count",
            "cera_lattice_l4_evidence_grounded_count",
            "cera_lattice_l6_quotient_class_count",
            "cera_lattice_answer_class_count",
            "cera_lattice_compression_ratio",
            "cera_lattice_candidate_observation_mismatch_count",
            "cera_contrast_ready",
            "cera_contrast_anchor_derivation_id",
            "cera_contrast_original_class_count",
            "cera_contrast_alternative_class_count",
            "cera_contrast_alternative_answer_class_count",
            "cera_contrast_unresolved_ambiguity_count",
            "cera_contrast_unresolved_ambiguities",
            "cera_original_support_v3_hypothesis_count",
            "cera_original_support_v3_roundtrip_executable",
            "cera_original_support_v3_graph_anchor_only",
            "cera_original_support_v3_ambiguity_count",
            "cera_original_support_v3_level_distribution",
            "cera_round7_compact_contrast_version",
            "cera_round7_contrast_constructible",
            "cera_round7_contrast_compact",
            "cera_round7_repair_eligible",
            "cera_round7_paired_intervention_count",
            "cera_round7_separating_intervention_count",
            "cera_round7_contrast_unknown_count",
            "cera_round7_contrast_unknowns",
            "cera_round7_registry_evidence_count",
            "cera_round7_registry_derivation_count",
            "cera_round7_registry_hypothesis_count",
            "cera_round7_registry_intervention_count",
            "cera_round8_compact_contrast_version",
            "cera_round8_contrast_constructible",
            "cera_round8_contrast_registry_complete",
            "cera_round8_contrast_compact",
            "cera_round8_repair_eligible",
            "cera_round8_basis_count",
            "cera_round8_behavior_class_count",
            "cera_round8_separating_intervention_count",
            "cera_round8_contrast_unknown_count",
            "cera_round8_contrast_unknowns",
            "cera_round8_registry_evidence_count",
            "cera_round8_registry_derivation_count",
            "cera_round8_registry_hypothesis_count",
            "cera_round8_registry_intervention_count",
            "cera_round8_compact_v3_promoted_legacy_admissibility",
            "cera_round8_compact_v3_legacy_reject_reason",
            "cera_round8_compact_v3_selected_derivation_id",
        ):
            if key in self.metadata:
                fields[key] = self.metadata[key]
        if self.output is not None:
            fields["cera_output"] = self.output.to_dict()
        if self.raw_response:
            fields["cera_raw_response"] = self.raw_response
        if log_full_prompt and self.prompt:
            fields["cera_prompt"] = self.prompt
        if log_evidence_packet and self.packet is not None:
            fields["cera_evidence_packet"] = self.packet.to_dict()
        return {k: v for k, v in fields.items() if v is not None}


def collect_ids(items: Iterable[Any], key: str) -> List[str]:
    out: List[str] = []
    for item in items:
        payload = item.to_dict() if hasattr(item, "to_dict") else _as_dict(item)
        value = payload.get(key)
        if value:
            out.append(str(value))
    return out
