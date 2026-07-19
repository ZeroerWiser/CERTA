"""Structural admissibility checks for executable derivations."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Set

from .project import canonical_answer_key
from .schema import (
    AdmissibleCandidateSet,
    CandidateAdmissibilityResult,
    CandidateContractCheck,
    ExecutableDerivation,
    PreEvidenceQueryContract,
)


def _edge_set(graph: Any) -> Set[tuple[str, str, str]]:
    edges = getattr(graph, "edges", []) or []
    out: Set[tuple[str, str, str]] = set()
    for edge in edges:
        etype = getattr(getattr(edge, "edge_type", ""), "value", getattr(edge, "edge_type", ""))
        out.add((str(getattr(edge, "source", "")), str(getattr(edge, "target", "")), str(etype)))
    return out


def _allowed(value: str, allowed_values: Sequence[str]) -> bool:
    allowed = {str(item) for item in allowed_values or []}
    return not allowed or "UNKNOWN" in allowed or value in allowed


def _operation_allowed(family: str, hypotheses: Sequence[str]) -> bool:
    allowed = {str(item) for item in hypotheses or []}
    if not allowed or "UNKNOWN" in allowed:
        return True
    if family in allowed:
        return True
    if "LOOKUP" in allowed and family == "LOOKUP_AGGREGATE":
        return True
    if "PAIR_COMPARE" in allowed and family in {"ARGMAX", "ARGMIN"}:
        return True
    return False


def check_candidate_contract(
    derivation: ExecutableDerivation,
    contract: PreEvidenceQueryContract,
) -> CandidateContractCheck:
    failures: List[str] = []
    ambiguous = (
        "UNKNOWN" in set(contract.allowed_answer_domains or [])
        or "UNKNOWN" in set(contract.allowed_projection_operators or [])
        or not contract.candidate_independent_operation_hypotheses
    )
    answer_ok = _allowed(derivation.output_domain, contract.allowed_answer_domains)
    projection_ok = _allowed(derivation.projection_operator, contract.allowed_projection_operators)
    operation_ok = _operation_allowed(
        derivation.operation_family,
        contract.candidate_independent_operation_hypotheses,
    )
    if not answer_ok:
        failures.append("output_domain_incompatible_with_pre_evidence_contract")
    if not projection_ok:
        failures.append("projection_operator_incompatible_with_pre_evidence_contract")
    if not operation_ok:
        failures.append("operation_family_incompatible_with_pre_evidence_contract")
    return CandidateContractCheck(
        derivation_id=derivation.derivation_id,
        candidate_id=derivation.source_candidate_id,
        answer_domain_ok=answer_ok,
        projection_operator_ok=projection_ok,
        operation_family_ok=operation_ok,
        ambiguous_contract=ambiguous,
        failure_reasons=failures,
    )


def _required_nodes_present(derivation: ExecutableDerivation, graph: Any) -> bool:
    if graph is None:
        return True
    nodes = getattr(graph, "nodes", {}) or {}
    return all(node_id in nodes for node_id in derivation.operand_node_ids)


def _required_edges_present(derivation: ExecutableDerivation, graph: Any) -> bool:
    if graph is None:
        return True
    existing = _edge_set(graph)
    return all(tuple(edge) in existing for edge in derivation.required_edge_triples)


def _outside_evidence_absent(derivation: ExecutableDerivation, evidence: Any) -> bool:
    if evidence is None:
        return True
    evidence_nodes = set(getattr(evidence, "evidence_nodes", set()) or set())
    if not evidence_nodes:
        return False
    return all(node_id in evidence_nodes for node_id in derivation.operand_node_ids)


def admissibility_result(
    derivation: ExecutableDerivation,
    *,
    contract: PreEvidenceQueryContract,
    graph: Any = None,
    evidence: Any = None,
) -> CandidateAdmissibilityResult:
    failures = list(derivation.failure_reasons or [])
    if derivation.availability != "available":
        failures.append("derivation_unavailable")
    if not derivation.provenance_complete:
        failures.append("provenance_incomplete")
    if not _required_nodes_present(derivation, graph):
        failures.append("required_nodes_missing")
    if not _required_edges_present(derivation, graph):
        failures.append("required_edges_missing")
    if not str(derivation.projected_answer or "").strip():
        failures.append("projected_answer_empty")
    if not _outside_evidence_absent(derivation, evidence):
        failures.append("outside_evidence_support")

    contract_check = check_candidate_contract(derivation, contract)
    failures.extend(contract_check.failure_reasons)
    seen = set()
    unique_failures: List[str] = []
    for failure in failures:
        if failure not in seen:
            seen.add(failure)
            unique_failures.append(failure)
    return CandidateAdmissibilityResult(
        derivation=derivation,
        candidate_id=derivation.source_candidate_id,
        admissible=not unique_failures,
        contract_check=contract_check,
        failure_reasons=unique_failures,
    )


def build_admissible_candidate_set(
    *,
    contract: PreEvidenceQueryContract,
    derivations: Sequence[ExecutableDerivation],
    graph: Any = None,
    evidence: Any = None,
) -> AdmissibleCandidateSet:
    results = [
        admissibility_result(derivation, contract=contract, graph=graph, evidence=evidence)
        for derivation in derivations
    ]
    admissible = [result.derivation for result in results if result.admissible]
    classes: Dict[str, List[str]] = {}
    for derivation in admissible:
        key = canonical_answer_key(derivation.projected_answer)
        classes.setdefault(key, []).append(derivation.derivation_id)
    class_count = len(classes)
    review_eligible = class_count == 1
    selected_derivation_id = ""
    selected_candidate_id = ""
    reject_reason = ""
    if class_count == 0:
        reject_reason = "no_admissible_projected_answer_class"
    elif class_count > 1:
        reject_reason = "ambiguous_admissible_projected_answer_classes"
    else:
        selected_derivation_id = next(iter(next(iter(classes.values()))), "")
        selected = next((d for d in admissible if d.derivation_id == selected_derivation_id), None)
        selected_candidate_id = selected.source_candidate_id if selected is not None else ""
    return AdmissibleCandidateSet(
        pre_evidence_contract=contract,
        derivations=list(derivations),
        results=results,
        admissible_derivations=admissible,
        projected_answer_classes=classes,
        review_eligible=review_eligible,
        selected_derivation_id=selected_derivation_id,
        selected_candidate_id=selected_candidate_id,
        ambiguity_count=max(0, class_count - 1),
        reject_reason=reject_reason,
        notes=[],
    )
