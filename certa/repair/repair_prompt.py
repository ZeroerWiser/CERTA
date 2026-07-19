"""Prompt construction for the Causal-Epistemic Repair Agent (CERA)."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping

from .evidence_packet import CausalEvidencePacket, pretty_json


DEFAULT_CERA_TEMPLATE_VERSION = "cera_repair_v2"
CERA_V3_TEMPLATE_VERSION = "cera_repair_v3"


def _packet_payload(packet: Any) -> Dict[str, Any]:
    if isinstance(packet, CausalEvidencePacket):
        return packet.to_dict()
    if isinstance(packet, dict):
        return dict(packet)
    return {}


def _prompt_safe_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(k): _prompt_safe_payload(v)
            for k, v in value.items()
            if str(k) != "contains_gold_answer"
        }
    if isinstance(value, list):
        return [_prompt_safe_payload(v) for v in value]
    return value


def _as_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "to_dict"):
        try:
            return dict(value.to_dict())
        except Exception:
            return {}
    return {}


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _compact_edges(edges: Iterable[Any], limit: int = 8) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for edge in list(edges or [])[:limit]:
        payload = _as_dict(edge)
        out.append({
            "source": payload.get("source"),
            "target": payload.get("target"),
            "edge_type": payload.get("edge_type"),
        })
    return out


def _compact_support(items: Any, limit: int = 12) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in _as_list(items)[:limit]:
        payload = _as_dict(item)
        out.append({
            "evidence_id": payload.get("evidence_id"),
            "node_id": payload.get("node_id"),
            "row": payload.get("row"),
            "col": payload.get("col"),
            "cell_value": payload.get("cell_value"),
            "row_headers": _as_list(payload.get("row_headers"))[:4],
            "col_headers": _as_list(payload.get("col_headers"))[:4],
            "support_role": payload.get("support_role"),
            "provenance": payload.get("provenance"),
        })
    return out


def _compact_certificate(cert: Mapping[str, Any]) -> Dict[str, Any]:
    payload = _as_dict(cert)
    keys = [
        "path_verified",
        "evidence_fallback",
        "candidate_effective_evidence_coverage",
        "candidate_evidence_coverage",
        "binding_confidence",
        "scci",
        "bir",
        "asr",
        "ib_mdl_score",
        "operation_compatible",
    ]
    return {key: payload.get(key) for key in keys if key in payload}


def _compact_candidate(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    payload = _as_dict(candidate)
    return {
        "candidate_id": payload.get("candidate_id"),
        "denotation": payload.get("denotation"),
        "operation": payload.get("operation"),
        "priority": payload.get("priority"),
        "operation_metadata": _as_dict(payload.get("operation_metadata")),
        "certificate_summary": _compact_certificate(_as_dict(payload.get("certificate"))),
        "support_cell_count": len(_as_list(payload.get("cells_used"))),
    }


def _compact_derivation(derivation: Mapping[str, Any]) -> Dict[str, Any]:
    payload = _as_dict(derivation)
    return {
        "derivation_id": payload.get("derivation_id"),
        "source_candidate_id": payload.get("source_candidate_id"),
        "operation_family": payload.get("operation_family"),
        "comparison_polarity": payload.get("comparison_polarity"),
        "typed_signature": payload.get("typed_signature"),
        "projection_operator": payload.get("projection_operator"),
        "projected_answer": payload.get("projected_answer"),
        "output_domain": payload.get("output_domain"),
        "evidence_ids": _as_list(payload.get("evidence_ids")),
        "executable_program": payload.get("executable_program"),
        "provenance_complete": payload.get("provenance_complete"),
        "availability": payload.get("availability"),
        "failure_reasons": _as_list(payload.get("failure_reasons")),
        "operand_node_ids": _as_list(payload.get("operand_node_ids")),
        "required_edge_triples": _as_list(payload.get("required_edge_triples"))[:12],
    }


def _compact_hypothesis_set(value: Any) -> Dict[str, Any]:
    payload = _as_dict(value)
    hypotheses = []
    for hyp in _as_list(payload.get("hypotheses"))[:6]:
        row = _as_dict(hyp)
        hypotheses.append({
            "hypothesis_id": row.get("hypothesis_id"),
            "source": row.get("source"),
            "projected_answer": row.get("projected_answer"),
            "operation_family": row.get("operation_family"),
            "projection_operator": row.get("projection_operator"),
            "support_level": row.get("support_level"),
            "support_node_ids": _as_list(row.get("support_node_ids")),
            "required_edge_triples": _as_list(row.get("required_edge_triples"))[:8],
            "notes": _as_list(row.get("notes")),
        })
    return {
        "contains_executable_derivation": payload.get("contains_executable_derivation"),
        "contains_graph_anchor_only": payload.get("contains_graph_anchor_only"),
        "hypotheses": hypotheses,
        "notes": _as_list(payload.get("notes")),
    }


def _compact_counterfactuals(items: Any, limit: int = 8) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in _as_list(items)[:limit]:
        payload = _as_dict(item)
        observed = _as_dict(payload.get("observed_effect"))
        flags = _as_dict(payload.get("flags"))
        out.append({
            "cf_id": payload.get("cf_id"),
            "intervention_type": payload.get("intervention_type"),
            "removed_nodes": _as_list(payload.get("removed_nodes"))[:8],
            "removed_edges": _compact_edges(payload.get("removed_edges"), limit=8),
            "modified_nodes": _as_list(payload.get("modified_nodes"))[:8],
            "expected_effect": payload.get("expected_effect"),
            "observed_effect": {
                "available": observed.get("available"),
                "changed": observed.get("changed"),
                "candidate_specific": observed.get("candidate_specific"),
                "support_valid": observed.get("support_valid"),
                "failure_reason": observed.get("failure_reason"),
                "post_projected_answer": observed.get("post_projected_answer"),
            },
            "flags": {
                "is_benign": flags.get("is_benign"),
                "is_adversarial": flags.get("is_adversarial"),
                "candidate_specific": flags.get("candidate_specific"),
                "support_valid": flags.get("support_valid"),
                "failure_reason": flags.get("failure_reason"),
            },
            "causal_interpretation": payload.get("causal_interpretation"),
        })
    return out


def _compact_semantic_statements(items: Any, limit: int = 24) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in _as_list(items)[:limit]:
        payload = _as_dict(item)
        out.append({
            "statement_id": payload.get("statement_id"),
            "category": payload.get("category"),
            "source_object_ids": _as_list(payload.get("source_object_ids")),
            "natural_language": payload.get("natural_language"),
            "availability": payload.get("availability"),
            "provenance": payload.get("provenance"),
        })
    return out


def _compact_table_excerpt(items: Any, limit: int = 16) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in _as_list(items)[:limit]:
        payload = _as_dict(item)
        out.append({
            "row": payload.get("row"),
            "col": payload.get("col"),
            "value": payload.get("value"),
            "source": payload.get("source"),
            "evidence_id": payload.get("evidence_id"),
        })
    return out


def build_cera_prompt_view(packet: Any) -> Dict[str, Any]:
    payload = _prompt_safe_payload(_packet_payload(packet))
    candidate = _as_dict(payload.get("candidate"))
    return {
        "prompt_view_version": "cera_prompt_view_v1",
        "query_contract": payload.get("query_contract", {}),
        "original_answer_record": {
            "answer": payload.get("original_answer", ""),
            "certificate_available": payload.get("original_certificate_available", False),
            "equivalent_candidate_id": payload.get("original_equivalent_candidate_id", ""),
            "support_hypothesis_set": _compact_hypothesis_set(payload.get("original_support_hypothesis_set", {})),
            "support_chain": _compact_support(payload.get("original_support_chain", [])),
            "notes": payload.get("original_support_chain_notes", []),
        },
        "candidate_under_review": {
            "answer": payload.get("candidate_under_review", ""),
            "candidate_summary": _compact_candidate(candidate),
            "reviewed_derivation": _compact_derivation(payload.get("reviewed_derivation", {})),
            "support_chain": _compact_support(payload.get("support_chain", [])),
        },
        "admissible_candidate_set": payload.get("admissible_candidate_set", {}),
        "compact_behavioral_contrast_v2": payload.get("compact_behavioral_contrast_v2", {}),
        "candidate_specific_counterfactuals": _compact_counterfactuals(payload.get("counterfactual_chain", [])),
        "semantic_statements": _compact_semantic_statements(payload.get("evidence_semantic_statements", [])),
        "table_excerpt": _compact_table_excerpt(payload.get("table_excerpt", [])),
        "packet_metadata": {
            "packet_hash": _as_dict(payload.get("metadata")).get("packet_hash"),
            "query_contract_hash": _as_dict(payload.get("metadata")).get("query_contract_hash"),
            "reviewed_derivation_id": _as_dict(payload.get("metadata")).get("reviewed_derivation_id"),
            "reviewed_derivation_replay_mode": _as_dict(payload.get("metadata")).get("reviewed_derivation_replay_mode"),
            "row_major_context_cell_count": _as_dict(payload.get("metadata")).get("row_major_context_cell_count"),
            "allow_row_major_context": _as_dict(payload.get("metadata")).get("allow_row_major_context"),
        },
    }


def build_cera_prompt_view_v3(packet: Any) -> Dict[str, Any]:
    payload = _prompt_safe_payload(_packet_payload(packet))
    contrast = _as_dict(payload.get("compact_behavioral_contrast_v3"))
    metadata = _as_dict(payload.get("metadata"))
    return {
        "prompt_view_version": "cera_prompt_view_v3",
        "query_semantics": {
            "query_contract": payload.get("query_contract", {}),
            "contrast_query_semantics": _as_dict(contrast.get("query_semantics")),
        },
        "compact_behavioral_contrast_v3": {
            "contrast_version": contrast.get("contrast_version", ""),
            "states": _as_dict(contrast.get("states")),
            "original_hypothesis": _as_dict(contrast.get("original_hypothesis")),
            "alternative_hypothesis": _as_dict(contrast.get("alternative_hypothesis")),
            "alternative_hypotheses": _as_list(contrast.get("alternative_hypotheses")),
            "separating_interventions": _as_list(contrast.get("separating_interventions")),
            "unknowns": _as_list(contrast.get("unknowns")),
            "registry": _as_dict(contrast.get("registry")),
        },
        "packet_metadata": {
            "packet_hash": metadata.get("packet_hash"),
            "query_contract_hash": metadata.get("query_contract_hash"),
            "reviewed_derivation_id": metadata.get("reviewed_derivation_id"),
        },
    }


def build_cera_prompt(
    packet: Any,
    *,
    template_version: str = DEFAULT_CERA_TEMPLATE_VERSION,
    require_derivation_program: bool = True,
    require_counterfactual_reference: bool = True,
) -> str:
    payload = _packet_payload(packet)
    if template_version == CERA_V3_TEMPLATE_VERSION:
        evidence_sections = {
            "prompt_view": build_cera_prompt_view_v3(payload),
        }
        requirements = {
            "decision_labels": ["KEEP_ORIGINAL", "USE_REPAIRED", "INSUFFICIENT_CERTIFICATE"],
            "must_output_valid_json_only": True,
            "must_not_use_gold_or_external_information": True,
            "must_use_only_compact_behavioral_contrast_v3": True,
            "must_not_regenerate_or_execute_derivation_programs": True,
            "must_cite_registry_ids": {
                "hypothesis_ids": "H*",
                "derivation_refs": "D*",
                "evidence_refs": "E*",
                "intervention_refs": "I*",
            },
            "use_repaired_requires": [
                "repair_eligible=true in the packet states",
                "chosen_hypothesis_id equals the unique alternative hypothesis",
                "final_answer equals that hypothesis executed_answer",
                "at least one separating intervention_ref",
                "no blocking unknowns",
            ],
            "self_assessed_confidence_not_used": True,
        }
        response_schema = {
            "decision": "KEEP_ORIGINAL | USE_REPAIRED | INSUFFICIENT_CERTIFICATE",
            "chosen_hypothesis_id": "H*; required for USE_REPAIRED and KEEP_ORIGINAL when a side is chosen",
            "final_answer": "executed answer string; empty unless decision is USE_REPAIRED",
            "query_semantics_assessment": {
                "answer_domain_ok": "true | false | unknown",
                "operation_family_ok": "true | false | unknown",
                "projection_operator_ok": "true | false | unknown",
                "notes": "short string",
            },
            "original_assessment": {
                "hypothesis_id": "H*",
                "derivation_ref": "D*",
                "evidence_refs": ["E1"],
                "intervention_refs": ["I1"],
                "summary": "brief statement grounded only in registry IDs",
            },
            "alternative_assessment": {
                "hypothesis_id": "H*",
                "derivation_ref": "D*",
                "evidence_refs": ["E2"],
                "intervention_refs": ["I1"],
                "summary": "brief statement grounded only in registry IDs",
            },
            "separating_intervention_refs": ["I1"],
            "blocking_unknowns": ["unknown labels from packet.unknowns, or empty list"],
            "rationale": "brief rationale grounded only in cited H/D/E/I IDs",
            "safety_notes": ["brief notes about uncertainty or missing support"],
        }
        return (
            "You are the Causal-Epistemic Repair Agent for Table Question Answering.\n"
            "Use only the compact_behavioral_contrast_v3 registry. Do not use hidden labels, "
            "gold answers, external lookup, unstated table values, table excerpts, full lattices, "
            "or derivation-program expressions. The deterministic records already contain executed "
            "derivations; your job is to choose among registry hypotheses or declare insufficiency.\n\n"
            f"Template version: {template_version}\n\n"
            "1. Requirements:\n"
            f"{pretty_json(requirements)}\n\n"
            "2. Required JSON response schema:\n"
            f"{pretty_json(response_schema)}\n\n"
            "3. Compact registry-backed contrast:\n"
            f"{pretty_json(evidence_sections)}\n\n"
            "4. Role checklist:\n"
            "- Original Defender: cite the original H/D/E/I registry IDs.\n"
            "- Alternative Advocate: cite the alternative H/D/E/I registry IDs.\n"
            "- Intervention Skeptic: cite only I* records marked separating and evaluable_on_both_sides.\n"
            "- Epistemic Arbiter: choose INSUFFICIENT_CERTIFICATE when repair_eligible is false or unknowns block repair.\n\n"
            "Return only one JSON object matching the schema."
        )

    evidence_sections = {
        "prompt_view": build_cera_prompt_view(payload),
    }
    requirements = {
        "decision_labels": ["KEEP_ORIGINAL", "USE_REPAIRED", "INSUFFICIENT_CERTIFICATE"],
        "must_output_valid_json_only": True,
        "must_not_use_gold_or_external_information": True,
        "must_check_query_contract": True,
        "must_cite_role_specific_support_ids": True,
        "must_cite_candidate_specific_counterfactual_ids_for_use_repaired": bool(require_counterfactual_reference),
        "must_use_evidence_dsl_for_use_repaired": bool(require_derivation_program),
        "must_not_use_repaired_when_final_answer_equals_original_answer": True,
        "must_treat_missing_certificates_as_insufficient_by_default": True,
        "allowed_dsl": [
            "SELECT(S1)",
            "COUNT(S1,S2,...)",
            "SUM(S1,S2,...)",
            "DIFF(S1,S2)",
            "RATIO(S1,S2)",
            "COMPARE(S1,\">\",S2)",
            "ARGMAX(S1,S2,...)",
            "ARGMIN(S1,S2,...)",
        ],
    }
    response_schema = {
        "decision": "KEEP_ORIGINAL | USE_REPAIRED | INSUFFICIENT_CERTIFICATE",
        "final_answer": "answer string; empty unless decision is USE_REPAIRED",
        "self_assessed_confidence": "diagnostic number from 0 to 1; not a validation score",
        "query_contract_check": {
            "answer_domain_ok": "true | false | unknown",
            "operation_signature_ok": "true | false | unknown",
            "projection_target_ok": "true | false | unknown",
            "notes": "short string",
        },
        "original_defense": {
            "summary": "why the original answer may be retained",
            "evidence_ids": ["OS1"],
        },
        "candidate_case": {
            "summary": "why the candidate is supported or not",
            "evidence_ids": ["S1"],
        },
        "counterfactual_assessment": {
            "summary": "what the counterfactual chain shows",
            "cf_ids": ["CF1"],
        },
        "uncertainty_assessment": {
            "missing_or_unknown_fields": ["unit_dimension"],
            "blocking_uncertainties": ["short strings"],
        },
        "derivation_program": {
            "expression": "Evidence DSL expression such as SELECT(S1)",
            "evidence_ids": ["S1"],
        },
        "rationale": "brief rationale grounded only in cited packet IDs",
        "safety_notes": ["brief notes about uncertainty or missing support"],
    }
    return (
        "You are the Causal-Epistemic Repair Agent for Table Question Answering.\n"
        "You make one shadow-only decision about whether a certified candidate should repair "
        "the Initial Proposal Agent answer. Use exactly these internal roles before answering: "
        "Original Defender, Candidate Advocate, Counterfactual Skeptic, and Epistemic Arbiter. "
        "Do not infer facts beyond the packet. Do not use hidden labels, gold answers, external "
        "lookup, or unstated table values.\n\n"
        f"Template version: {template_version}\n\n"
        "1. Requirements:\n"
        f"{pretty_json(requirements)}\n\n"
        "2. Required JSON response schema:\n"
        f"{pretty_json(response_schema)}\n\n"
        "3. Query contract, original record, reviewed derivation, candidate support, candidate-specific interventions, and semantic statements:\n"
        f"{pretty_json(evidence_sections)}\n\n"
        "4. Role checklist:\n"
        "- Original Defender: cite only OS evidence IDs when defending the original answer.\n"
        "- Candidate Advocate: cite only S evidence IDs when supporting USE_REPAIRED.\n"
        "- Counterfactual Skeptic: cite only CF IDs and require candidate_specific=true when relying on interventions.\n"
        "- Epistemic Arbiter: choose INSUFFICIENT_CERTIFICATE when the packet does not prove the repair.\n\n"
        "Return only one JSON object matching the schema."
    )
