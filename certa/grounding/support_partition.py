"""Proposal-relative support partition for CERTA Round 9 closure records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from certa.derivations.answer_equivalence import inference_answer_key, inference_answers_equivalent
from certa.derivations.schema import ExecutableDerivation, to_jsonable
from certa.grounding.plan_closure import PlanClosure


@dataclass(frozen=True)
class SupportPartition:
    initial_proposal_answer_key: str
    original_support: Tuple[ExecutableDerivation, ...]
    alternative_support: Tuple[ExecutableDerivation, ...]
    disjoint: bool
    exhaustive: bool
    equivalence_policy: str = "inference_answers_equivalent"

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(self)


def partition_support(closure: PlanClosure, *, initial_proposal_answer: Any) -> SupportPartition:
    """Partition executable closure derivations by Initial-Proposal equivalence."""
    if not closure.resource_complete:
        return SupportPartition(
            initial_proposal_answer_key=inference_answer_key(initial_proposal_answer).compact(),
            original_support=(),
            alternative_support=(),
            disjoint=True,
            exhaustive=True,
        )
    original = []
    alternative = []
    for derivation in closure.executable_derivations:
        if inference_answers_equivalent(derivation.projected_answer, initial_proposal_answer):
            original.append(derivation)
        else:
            alternative.append(derivation)
    original_ids = {item.derivation_id for item in original}
    alternative_ids = {item.derivation_id for item in alternative}
    executable_ids = {item.derivation_id for item in closure.executable_derivations}
    return SupportPartition(
        initial_proposal_answer_key=inference_answer_key(initial_proposal_answer).compact(),
        original_support=tuple(original),
        alternative_support=tuple(alternative),
        disjoint=not bool(original_ids & alternative_ids),
        exhaustive=(original_ids | alternative_ids) == executable_ids,
    )
