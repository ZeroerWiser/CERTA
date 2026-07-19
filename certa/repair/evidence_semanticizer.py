"""Gold-free evidence semanticization for CERA prompts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping

from .evidence_packet import CausalEvidencePacket, to_jsonable


def _as_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "to_dict"):
        try:
            return dict(value.to_dict())
        except Exception:
            return {}
    return {}


@dataclass
class EvidenceSemanticStatement:
    statement_id: str
    category: str
    source_object_ids: List[str]
    raw_facts: Dict[str, Any] = field(default_factory=dict)
    natural_language: str = ""
    availability: str = "available"
    provenance: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def semanticize_support_chain(items: Any, *, prefix: str, category: str, provenance: str) -> List[EvidenceSemanticStatement]:
    statements: List[EvidenceSemanticStatement] = []
    for idx, item in enumerate(items or [], start=1):
        payload = _as_dict(item)
        evidence_id = str(payload.get("evidence_id") or f"{prefix}{idx}")
        row = payload.get("row")
        col = payload.get("col")
        value = str(payload.get("cell_value", ""))
        role = str(payload.get("support_role", "support_cell"))
        statements.append(EvidenceSemanticStatement(
            statement_id=f"{prefix}{idx}",
            category=category,
            source_object_ids=[evidence_id, str(payload.get("node_id", ""))],
            raw_facts={
                "evidence_id": evidence_id,
                "node_id": payload.get("node_id"),
                "row": row,
                "col": col,
                "cell_value": value,
                "support_role": role,
                "provenance": payload.get("provenance"),
            },
            natural_language=f"{evidence_id} cites {role} at row {row}, col {col} with value {value}.",
            availability="available" if value else "unknown",
            provenance=provenance,
        ))
    return statements


def semanticize_counterfactual_chain(items: Any, *, prefix: str = "CS") -> List[EvidenceSemanticStatement]:
    statements: List[EvidenceSemanticStatement] = []
    for idx, item in enumerate(items or [], start=1):
        payload = _as_dict(item)
        cf_id = str(payload.get("cf_id") or f"CF{idx}")
        observed = _as_dict(payload.get("observed_effect"))
        available = "available" if observed.get("available") else "unavailable"
        changed = bool(observed.get("changed"))
        candidate_specific = bool(observed.get("candidate_specific"))
        statements.append(EvidenceSemanticStatement(
            statement_id=f"{prefix}{idx}",
            category="counterfactual_intervention",
            source_object_ids=[cf_id],
            raw_facts={
                "cf_id": cf_id,
                "intervention_type": payload.get("intervention_type"),
                "observed_effect_available": observed.get("available"),
                "candidate_specific": candidate_specific,
                "changed": changed,
                "support_valid": observed.get("support_valid"),
                "failure_reason": observed.get("failure_reason"),
            },
            natural_language=(
                f"{cf_id} is a {payload.get('intervention_type')} intervention; "
                f"candidate-specific replay is {available} and changed={changed}."
            ),
            availability=available,
            provenance="candidate_intervention_replay" if candidate_specific else "counterfactual_chain",
        ))
    return statements


def semanticize_query_contract(contract: Any) -> List[EvidenceSemanticStatement]:
    payload = _as_dict(contract)
    if not payload:
        return []
    operation = payload.get("operation_signature")
    if not operation:
        operation = ",".join(str(x) for x in (payload.get("candidate_independent_operation_hypotheses") or []))
    return [
        EvidenceSemanticStatement(
            statement_id="QC1",
            category="query_contract",
            source_object_ids=["query_contract"],
            raw_facts={
                "answer_domain": payload.get("answer_domain"),
                "allowed_answer_domains": payload.get("allowed_answer_domains", []),
                "allowed_projection_operators": payload.get("allowed_projection_operators", []),
                "operation_signature": operation,
                "unknown_fields": payload.get("unknown_fields", []),
                "pre_evidence": _as_dict(payload.get("metadata")).get("pre_evidence"),
            },
            natural_language=(
                "The pre-evidence query contract allows answer domains "
                f"{payload.get('allowed_answer_domains', [payload.get('answer_domain', 'UNKNOWN')])} "
                f"and operation hypotheses {operation or 'unknown'}."
            ),
            availability="available",
            provenance="pre_evidence_query_contract" if _as_dict(payload.get("metadata")).get("pre_evidence") else "typed_query_contract",
        )
    ]


def semanticize_reviewed_derivation(derivation: Any) -> List[EvidenceSemanticStatement]:
    payload = _as_dict(derivation)
    if not payload:
        return []
    derivation_id = str(payload.get("derivation_id") or "D?")
    return [
        EvidenceSemanticStatement(
            statement_id="TD1",
            category="typed_executable_derivation",
            source_object_ids=[derivation_id, str(payload.get("source_candidate_id", ""))],
            raw_facts={
                "operation_family": payload.get("operation_family"),
                "projection_operator": payload.get("projection_operator"),
                "output_domain": payload.get("output_domain"),
                "projected_answer": payload.get("projected_answer"),
                "availability": payload.get("availability"),
                "failure_reasons": payload.get("failure_reasons", []),
            },
            natural_language=(
                f"{derivation_id} executes {payload.get('operation_family')} with "
                f"{payload.get('projection_operator')} and projects "
                f"{payload.get('projected_answer')}."
            ),
            availability=str(payload.get("availability") or "unknown"),
            provenance="typed_executable_derivation",
        )
    ]


def semanticize_original_support_hypotheses(hypothesis_set: Any) -> List[EvidenceSemanticStatement]:
    payload = _as_dict(hypothesis_set)
    hypotheses = payload.get("hypotheses") or []
    if not isinstance(hypotheses, list):
        return []
    statements: List[EvidenceSemanticStatement] = []
    for idx, item in enumerate(hypotheses, start=1):
        hyp = _as_dict(item)
        hyp_id = str(hyp.get("hypothesis_id") or f"OH{idx}")
        statements.append(EvidenceSemanticStatement(
            statement_id=f"OH{idx}",
            category="original_support_hypothesis",
            source_object_ids=[hyp_id, str(hyp.get("derivation_id", ""))],
            raw_facts={
                "projected_answer": hyp.get("projected_answer"),
                "answer_equivalent": hyp.get("answer_equivalent"),
                "operation_family": hyp.get("operation_family"),
                "projection_operator": hyp.get("projection_operator"),
                "availability": hyp.get("availability"),
                "provenance_complete": hyp.get("provenance_complete"),
            },
            natural_language=(
                f"{hyp_id} is an original-answer support hypothesis with "
                f"availability={hyp.get('availability')} and projected answer "
                f"{hyp.get('projected_answer')}."
            ),
            availability=str(hyp.get("availability") or "unknown"),
            provenance=str(hyp.get("source") or "original_support_hypothesis"),
        ))
    if not statements and payload:
        statements.append(EvidenceSemanticStatement(
            statement_id="OH0",
            category="original_support_hypothesis",
            source_object_ids=["original_support_hypothesis_set"],
            raw_facts={"notes": payload.get("notes", [])},
            natural_language="No original-answer support hypothesis is available.",
            availability="unavailable",
            provenance="original_support_reconstruction",
        ))
    return statements


def semanticize_packet(packet: Any) -> List[Dict[str, Any]]:
    payload = packet.to_dict() if isinstance(packet, CausalEvidencePacket) else _as_dict(packet)
    statements: List[EvidenceSemanticStatement] = []
    statements.extend(semanticize_query_contract(payload.get("query_contract")))
    statements.extend(semanticize_original_support_hypotheses(payload.get("original_support_hypothesis_set") or {}))
    statements.extend(semanticize_support_chain(
        payload.get("original_support_chain") or [],
        prefix="OS",
        category="original_answer_support",
        provenance="original_certificate_support_chain",
    ))
    statements.extend(semanticize_support_chain(
        payload.get("support_chain") or [],
        prefix="SS",
        category="candidate_support",
        provenance="candidate_support_chain",
    ))
    statements.extend(semanticize_reviewed_derivation(payload.get("reviewed_derivation") or {}))
    statements.extend(semanticize_counterfactual_chain(payload.get("counterfactual_chain") or []))
    return [to_jsonable(stmt) for stmt in statements]
