"""Gold-free query contracts for CERA.

Round 4 builds a pre-evidence contract before reviewed-candidate selection.
The legacy candidate-conditioned builder is kept as a compatibility wrapper,
but candidate/support inputs no longer affect the contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from certa.derivations.schema import PreEvidenceQueryContract
from certa.operations.contracts import signature_domains, signature_projections
from certa.repair.evidence_dsl import parse_number
from certa.reproducibility.canonical_json import canonical_json_hash


ANSWER_DOMAINS = {"ENTITY", "SCALAR", "SET", "BOOLEAN", "INTERVAL", "UNKNOWN"}
SCALAR_QUESTION_OPERATORS = {"sum", "average", "difference", "ratio", "count"}
SCALAR_CANDIDATE_OPERATIONS = {"arithmetic"}
LOOKUP_CANDIDATE_OPERATIONS = {"lookup_cell", "lookup_aggregate"}


def _as_mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _candidate_dict(candidate: Any) -> Dict[str, Any]:
    if isinstance(candidate, Mapping):
        return dict(candidate)
    if hasattr(candidate, "to_dict"):
        try:
            return dict(candidate.to_dict())
        except Exception:
            return {}
    out: Dict[str, Any] = {}
    for key in ("candidate_id", "denotation", "operation", "priority", "cells_used", "certificate"):
        if hasattr(candidate, key):
            out[key] = getattr(candidate, key)
    return out


def _support_values(support_chain: Optional[Sequence[Any]]) -> List[str]:
    values: List[str] = []
    for item in support_chain or []:
        payload = item.to_dict() if hasattr(item, "to_dict") else _as_mapping(item)
        if "cell_value" in payload:
            values.append(str(payload.get("cell_value", "")))
    return values


def _infer_answer_domain(
    *,
    question_operator: str,
    candidate_operation: str,
    candidate_denotation: str,
    support_values: Sequence[str],
) -> tuple[str, str]:
    if question_operator in SCALAR_QUESTION_OPERATORS:
        return "SCALAR", "question_frame.operator"
    if candidate_operation in SCALAR_CANDIDATE_OPERATIONS:
        return "SCALAR", "candidate.operation"
    if question_operator == "compare":
        return "UNKNOWN", "question_frame.compare_is_ambiguous_entity_or_boolean"
    if candidate_operation in LOOKUP_CANDIDATE_OPERATIONS:
        den_num = parse_number(candidate_denotation)
        if den_num is not None:
            return "SCALAR", "candidate.denotation_numeric_parse"
        if any(parse_number(value) is not None for value in support_values):
            return "SCALAR", "support_chain.cell_value_numeric_parse"
        return "UNKNOWN", "lookup_candidate_without_answer_role_proof"
    return "UNKNOWN", "no_structural_domain_proof"


def _unknown_fields(payload: Mapping[str, Any]) -> List[str]:
    unknown: List[str] = []
    for key, value in payload.items():
        if key in {"question", "field_provenance", "unknown_fields", "metadata"}:
            continue
        if value in ("", None, "unknown", "UNKNOWN", [], {}):
            unknown.append(key)
    return unknown


def _answer_surface_type(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "unknown"
    if text.lower() in {"true", "false", "yes", "no"}:
        return "boolean"
    if parse_number(text) is not None:
        return "scalar"
    if "|" in text:
        return "set"
    return "entity_or_text"


def _operation_hypotheses(
    question_operator: str,
    polarity: str = "neutral",
    surface_type: str = "unknown",
) -> List[str]:
    if polarity == "max":
        return ["ARGMAX", "PAIR_COMPARE", "LOOKUP"]
    if polarity == "min":
        return ["ARGMIN", "PAIR_COMPARE", "LOOKUP"]
    mapping = {
        "sum": ["SUM"],
        "average": ["AVERAGE"],
        "difference": ["DIFF"],
        "ratio": ["RATIO", "LOOKUP"],
        "count": ["COUNT"],
        "lookup": ["LOOKUP"],
        "compare": ["PAIR_COMPARE", "ARGMAX", "ARGMIN", "LOOKUP"],
    }
    return list(mapping.get(question_operator, ["UNKNOWN"]))


def _pre_unknown_fields(contract: PreEvidenceQueryContract) -> List[str]:
    unknown: List[str] = []
    payload = contract.to_dict()
    for key, value in payload.items():
        if key in {"question", "field_provenance", "unknown_fields", "metadata"}:
            continue
        if value in ("", None, "unknown", "UNKNOWN", [], {}):
            unknown.append(key)
        elif isinstance(value, list) and "UNKNOWN" in value:
            unknown.append(key)
    return unknown


def build_pre_evidence_query_contract(
    *,
    question: str,
    question_frame: Optional[Mapping[str, Any]] = None,
    result_context: Optional[Mapping[str, Any]] = None,
    initial_answer: Any = None,
    graph_stats: Optional[Mapping[str, Any]] = None,
) -> PreEvidenceQueryContract:
    """Build a candidate-independent query contract before evidence review."""
    qf = _as_mapping(question_frame)
    context = _as_mapping(result_context)
    question_operator = str(qf.get("operator") or context.get("question_operation") or "unknown")
    polarity = str(qf.get("polarity") or "neutral")
    surface_type = _answer_surface_type(initial_answer)
    semantic_surface_type = "unknown"
    operation_hypotheses = _operation_hypotheses(
        question_operator,
        polarity,
        semantic_surface_type,
    )
    allowed_domains = list(signature_domains(operation_hypotheses)) or ["UNKNOWN"]
    allowed_projections = list(signature_projections(operation_hypotheses)) or ["UNKNOWN"]
    contract = PreEvidenceQueryContract(
        question=str(question or ""),
        answer_domain=allowed_domains[0] if allowed_domains and allowed_domains[0] != "UNKNOWN" else "UNKNOWN",
        allowed_answer_domains=allowed_domains,
        allowed_projection_operators=allowed_projections,
        candidate_independent_operation_hypotheses=operation_hypotheses,
        unit_or_scale_constraints=[],
        initial_answer_surface_type=surface_type,
        field_provenance={
            "question": "dataset.question",
            "answer_domain": "question_frame.operator",
            "allowed_answer_domains": "question_frame.operator",
            "allowed_projection_operators": "question_frame.operator",
            "candidate_independent_operation_hypotheses": "question_frame.operator",
            "unit_or_scale_constraints": "uncomputed",
            "initial_answer_surface_type": "diagnostic_initial_proposal_answer_surface_type",
        },
        metadata={
            "contains_gold_answer": False,
            "pre_evidence": True,
            "initial_answer_surface_type_diagnostic_only": True,
            "legacy_question_frame_used": bool(qf),
            "legacy_question_frame_provenance": "structural_cert_utils.parse_question_frame" if qf else "",
            "question_polarity": polarity,
            "graph_stats_available": bool(graph_stats),
        },
    )
    contract.unknown_fields = _pre_unknown_fields(contract)
    return contract


@dataclass
class TypedQueryContract:
    question: str
    answer_domain: str = "UNKNOWN"
    operation_signature: str = "unknown"
    quantifier_scope: str = "unknown"
    comparison_polarity: str = "unknown"
    unit_dimension: str = "unknown"
    scale: str = "unknown"
    projection_target: str = "unknown"
    evidence_scope_constraints: List[Dict[str, Any]] = field(default_factory=list)
    field_provenance: Dict[str, str] = field(default_factory=dict)
    unknown_fields: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_typed_query_contract(
    *,
    question: str,
    question_frame: Optional[Mapping[str, Any]] = None,
    candidate: Any = None,
    support_chain: Optional[Sequence[Any]] = None,
    result_context: Optional[Mapping[str, Any]] = None,
) -> TypedQueryContract:
    """Compatibility wrapper for the Round 4 pre-evidence contract.

    Candidate and support-chain arguments are intentionally ignored.
    """
    return build_pre_evidence_query_contract(
        question=question,
        question_frame=question_frame,
        result_context=result_context,
        initial_answer=(result_context or {}).get("initial_answer") if isinstance(result_context, Mapping) else None,
        graph_stats=(result_context or {}).get("graph_stats") if isinstance(result_context, Mapping) else None,
    )


def build_candidate_conditioned_query_contract(
    *,
    question: str,
    question_frame: Optional[Mapping[str, Any]] = None,
    candidate: Any = None,
    support_chain: Optional[Sequence[Any]] = None,
    result_context: Optional[Mapping[str, Any]] = None,
) -> TypedQueryContract:
    """Legacy Round 3 candidate-conditioned contract, retained for audits."""
    qf = _as_mapping(question_frame)
    context = _as_mapping(result_context)
    cand = _candidate_dict(candidate)
    certificate = _as_mapping(cand.get("certificate"))
    question_operator = str(qf.get("operator") or context.get("question_operation") or "unknown")
    polarity = str(qf.get("polarity") or "unknown")
    if polarity == "neutral":
        polarity = "unknown"
    candidate_operation = str(cand.get("operation") or "unknown")
    support_values = _support_values(support_chain)
    answer_domain, answer_domain_source = _infer_answer_domain(
        question_operator=question_operator,
        candidate_operation=candidate_operation,
        candidate_denotation=str(cand.get("denotation", "")),
        support_values=support_values,
    )
    if answer_domain not in ANSWER_DOMAINS:
        answer_domain = "UNKNOWN"
    operation_signature = (
        f"question_operator={question_operator};"
        f"candidate_operation={candidate_operation};"
        f"candidate_id={cand.get('candidate_id', '')}"
    )
    evidence_scope_constraints = [
        {
            "constraint": "candidate_support_cells",
            "evidence_ids": [
                str((item.to_dict() if hasattr(item, "to_dict") else _as_mapping(item)).get("evidence_id"))
                for item in support_chain or []
                if (item.to_dict() if hasattr(item, "to_dict") else _as_mapping(item)).get("evidence_id")
            ],
            "provenance": "support_chain",
        },
        {
            "constraint": "candidate_certificate_path",
            "graph_path": [str(x) for x in (certificate.get("graph_path") or [])[:16]],
            "provenance": "candidate.certificate.graph_path",
        },
    ]
    contract = TypedQueryContract(
        question=str(question or ""),
        answer_domain=answer_domain,
        operation_signature=operation_signature,
        quantifier_scope=question_operator if question_operator in SCALAR_QUESTION_OPERATORS else "unknown",
        comparison_polarity=polarity,
        unit_dimension="unknown",
        scale="unknown",
        projection_target="candidate_denotation" if str(cand.get("denotation", "")).strip() else "unknown",
        evidence_scope_constraints=evidence_scope_constraints,
        field_provenance={
            "question": "dataset.question",
            "answer_domain": answer_domain_source,
            "operation_signature": "question_frame_and_candidate_certificate",
            "quantifier_scope": "question_frame.operator" if question_operator in SCALAR_QUESTION_OPERATORS else "uncomputed",
            "comparison_polarity": "question_frame.polarity" if polarity != "unknown" else "uncomputed",
            "unit_dimension": "uncomputed",
            "scale": "uncomputed",
            "projection_target": "candidate.denotation" if str(cand.get("denotation", "")).strip() else "uncomputed",
            "evidence_scope_constraints": "support_chain_and_candidate_certificate",
        },
        metadata={
            "contains_gold_answer": False,
            "legacy_question_frame_used": bool(qf),
            "legacy_question_frame_provenance": "structural_cert_utils.parse_question_frame" if qf else "",
        },
    )
    contract.unknown_fields = _unknown_fields(contract.to_dict())
    return contract


def query_contract_hash(contract: Any, n: int = 16) -> str:
    payload = contract.to_dict() if hasattr(contract, "to_dict") else contract
    return canonical_json_hash(payload, n=n)
