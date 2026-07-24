"""Fixed-budget proposal-blind typed-plan search contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from certa.active_v1.final_method_v1 import canonical_typed_plan_identity
from certa.reproducibility.canonical_json import canonical_json_hash


@dataclass(frozen=True)
class SearchSlot:
    view_kind: str
    call_index: int
    seed: int
    call_id: str


_VIEW_KINDS = (
    "ROLE_COMPLETE",
    "RETRIEVAL_COMPLETE",
    "VALUE_AWARE_PROPOSAL_BLIND",
)
SEARCH_SCHEDULE = tuple(
    SearchSlot(view_kind, index, 1729 + offset * 2 + index, f"{view_kind}-{index}")
    for offset, view_kind in enumerate(_VIEW_KINDS)
    for index in range(2)
)
_FORBIDDEN_KEYS = frozenset(
    {
        "answerproposal",
        "b0",
        "b0answer",
        "candidateanswer",
        "correct",
        "correctness",
        "gold",
        "goldanswer",
        "initialanswer",
        "initialproposal",
        "labelfield",
        "selectedanswer",
        "selectedfinal",
        "targetanswer",
    }
)


def _normalized_key(value: Any) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def assert_proposal_blind(value: Any) -> None:
    """Reject proposal, label, or selected-output fields at any depth."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = _normalized_key(key)
            if normalized in _FORBIDDEN_KEYS or any(
                token in normalized for token in ("correctness", "goldanswer")
            ):
                raise ValueError(f"proposal_blind_field_forbidden:{key}")
            assert_proposal_blind(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            assert_proposal_blind(child)


def merge_search_attempts(attempts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate the exact six-call roster and union plans without voting."""
    expected = {slot.call_id: slot for slot in SEARCH_SCHEDULE}
    actual_ids = [str(attempt.get("call_id") or "") for attempt in attempts]
    if len(attempts) != len(SEARCH_SCHEDULE) or set(actual_ids) != set(expected):
        raise ValueError("search_attempt_roster_mismatch")
    if len(actual_ids) != len(set(actual_ids)):
        raise ValueError("search_attempt_roster_duplicate")

    by_identity: dict[str, dict[str, Any]] = {}
    sources: dict[str, set[str]] = {}
    for attempt in attempts:
        call_id = str(attempt["call_id"])
        slot = expected[call_id]
        if (
            attempt.get("view_kind") != slot.view_kind
            or attempt.get("call_index") != slot.call_index
        ):
            raise ValueError(f"search_attempt_slot_mismatch:{call_id}")
        assert_proposal_blind(attempt.get("view", {}))
        status = attempt.get("status")
        if status not in {"OK", "EMPTY", "INVALID", "ERROR"}:
            raise ValueError(f"search_attempt_status_invalid:{call_id}")
        plans = attempt.get("plans")
        if not isinstance(plans, list):
            raise ValueError(f"search_attempt_plans_not_list:{call_id}")
        if status != "OK" and plans:
            raise ValueError(f"search_attempt_failed_with_plans:{call_id}")
        for plan in plans:
            if not isinstance(plan, Mapping):
                raise ValueError(f"search_plan_not_object:{call_id}")
            identity = canonical_typed_plan_identity(plan)
            semantic = {str(key): value for key, value in plan.items() if key != "plan_id"}
            by_identity.setdefault(identity, semantic)
            sources.setdefault(identity, set()).add(call_id)

    merged = []
    lineage = []
    for index, identity in enumerate(sorted(by_identity)):
        semantic = by_identity[identity]
        plan_id = f"P{index}"
        merged.append({"plan_id": plan_id, **semantic})
        lineage.append(
            {
                "union_plan_id": plan_id,
                "typed_program_sha256": canonical_json_hash(semantic),
                "source_call_ids": sorted(sources[identity]),
            }
        )
    return {
        "schema_version": "certa_v2_bounded_search_union_v1",
        "attempt_count": len(attempts),
        "schedule_sha256": canonical_json_hash(
            [
                {
                    "view_kind": slot.view_kind,
                    "call_index": slot.call_index,
                    "seed": slot.seed,
                    "call_id": slot.call_id,
                }
                for slot in SEARCH_SCHEDULE
            ]
        ),
        "plans": merged,
        "lineage": lineage,
    }
