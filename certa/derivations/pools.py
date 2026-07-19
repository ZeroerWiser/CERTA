"""Audit and decision derivation pools for CERTA Round 8B."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from .admissibility import check_candidate_contract
from .answer_equivalence import inference_answers_equivalent
from .schema import ExecutableDerivation, PreEvidenceQueryContract, to_jsonable


@dataclass
class AuditDerivationPool:
    candidate_derivations: list[ExecutableDerivation] = field(default_factory=list)
    planner_derivations: list[ExecutableDerivation] = field(default_factory=list)
    frontier_derivations: list[ExecutableDerivation] = field(default_factory=list)

    @property
    def derivations(self) -> list[ExecutableDerivation]:
        return [*self.candidate_derivations, *self.planner_derivations, *self.frontier_derivations]

    def metadata(self) -> dict[str, Any]:
        return {
            "cera_round8b_audit_derivation_count": len(self.derivations),
            "cera_round8b_audit_candidate_derivation_count": len(self.candidate_derivations),
            "cera_round8b_audit_planner_derivation_count": len(self.planner_derivations),
            "cera_round8b_audit_frontier_derivation_count": len(self.frontier_derivations),
        }


@dataclass
class DecisionDerivationPool:
    derivations: list[ExecutableDerivation] = field(default_factory=list)
    original_derivations: list[ExecutableDerivation] = field(default_factory=list)
    alternative_derivations: list[ExecutableDerivation] = field(default_factory=list)
    rejected_derivations: list[dict[str, Any]] = field(default_factory=list)
    source_policy: str = "planner_only"

    def metadata(self) -> dict[str, Any]:
        return {
            "cera_round8b_decision_source_policy": self.source_policy,
            "cera_round8b_decision_derivation_count": len(self.derivations),
            "cera_round8b_decision_original_derivation_count": len(self.original_derivations),
            "cera_round8b_decision_alternative_derivation_count": len(self.alternative_derivations),
            "cera_round8b_decision_rejected_derivation_count": len(self.rejected_derivations),
            "cera_round8b_decision_rejections": to_jsonable(self.rejected_derivations),
        }


def build_audit_derivation_pool(
    *,
    candidate_derivations: Sequence[ExecutableDerivation] = (),
    planner_derivations: Sequence[ExecutableDerivation] = (),
    frontier_derivations: Sequence[ExecutableDerivation] = (),
) -> AuditDerivationPool:
    return AuditDerivationPool(
        candidate_derivations=list(candidate_derivations),
        planner_derivations=list(planner_derivations),
        frontier_derivations=list(frontier_derivations),
    )


def build_decision_derivation_pool(
    *,
    planner_derivations: Sequence[ExecutableDerivation],
    original_answer: str,
    contract: PreEvidenceQueryContract,
    source_policy: str = "planner_only",
) -> DecisionDerivationPool:
    accepted: list[ExecutableDerivation] = []
    originals: list[ExecutableDerivation] = []
    alternatives: list[ExecutableDerivation] = []
    rejected: list[dict[str, Any]] = []
    for derivation in planner_derivations:
        check = check_candidate_contract(derivation, contract)
        if not check.ok:
            rejected.append({
                "derivation_id": derivation.derivation_id,
                "source_candidate_id": derivation.source_candidate_id,
                "reasons": list(check.failure_reasons),
            })
            continue
        accepted.append(derivation)
        if inference_answers_equivalent(derivation.projected_answer, original_answer):
            originals.append(derivation)
        else:
            alternatives.append(derivation)
    return DecisionDerivationPool(
        derivations=accepted,
        original_derivations=originals,
        alternative_derivations=alternatives,
        rejected_derivations=rejected,
        source_policy=source_policy,
    )
