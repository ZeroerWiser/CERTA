"""Thin role-aligned adapter for flat and retrieved Planner views."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence

from certa.egra.query_role_contract import validate_query_role_contract
from certa.operations.contracts import OPERATION_SIGNATURES
from certa.planner.schema_view import build_proposal_blind_planner_view


@dataclass(frozen=True)
class PlannerViewBuild:
    eligible: bool
    reason: str
    view: Dict[str, Any] = field(default_factory=dict)


def build_role_aligned_planner_view(
    *,
    question: str,
    graph: Any,
    table_json: Mapping[str, Any],
    contract: Mapping[str, Any],
    reference_node_ids: Optional[Sequence[str]] = None,
    selected_cards: Optional[Sequence[Mapping[str, Any]]] = None,
) -> PlannerViewBuild:
    """Build C1 flat or C2 narrowed views; invalid roles never fall back."""
    validation = validate_query_role_contract(contract)
    if not validation.ok:
        return PlannerViewBuild(
            False,
            f"invalid_query_role_contract:{'|'.join(validation.errors)}",
        )
    role = validation.normalized_payload
    if not role["supported_by_core_signatures"]:
        return PlannerViewBuild(False, "unsupported_by_core_signatures")

    signature_ids = tuple(str(item) for item in role["signature_candidates"])
    signatures = [OPERATION_SIGNATURES[item] for item in signature_ids]
    query_contract = {
        "answer_domain": str(role["answer_domain"]),
        "allowed_answer_domains": sorted({item.answer_domain for item in signatures}),
        "allowed_projection_operators": sorted({
            item.projection_operator for item in signatures
        }),
        "candidate_independent_operation_hypotheses": sorted({
            item.operation_family for item in signatures
        }),
        "unit_or_scale_constraints": [
            name
            for name, required in (
                ("TIME_SCOPE", role["requires_time_scope"]),
                ("UNIT_CONSISTENCY", role["requires_unit_consistency"]),
            )
            if required
        ],
    }
    view = build_proposal_blind_planner_view(
        question=question,
        graph=graph,
        table_json=table_json,
        query_contract=query_contract,
        include_table_values=False,
        legacy_query_semantics_mode="active",
        allowed_signature_ids=signature_ids,
    )
    view["planner_view_version"] = "certa_egra_role_aligned_flat_v1"

    if reference_node_ids is None:
        if selected_cards is not None:
            return PlannerViewBuild(False, "cards_without_reference_domain")
        return PlannerViewBuild(True, "", view)

    requested_ids = tuple(dict.fromkeys(str(item) for item in reference_node_ids))
    if not requested_ids:
        return PlannerViewBuild(False, "empty_egra_reference_domain")
    known_ids = {str(item["node_id"]) for item in view["schema_nodes"]}
    unknown_ids = sorted(set(requested_ids) - known_ids)
    if unknown_ids:
        return PlannerViewBuild(
            False,
            f"unknown_egra_reference_ids:{','.join(unknown_ids)}",
        )
    selected_set = set(requested_ids)
    compact_cards = []
    for card in selected_cards or ():
        header_ids = [str(item) for item in card.get("header_node_ids") or []]
        if not header_ids or not set(header_ids) <= selected_set:
            return PlannerViewBuild(
                False,
                f"card_outside_reference_domain:{card.get('card_id', '')}",
            )
        compact_cards.append({
            "card_id": str(card["card_id"]),
            "unit_kind": str(card["unit_kind"]),
            "human_readable_text": str(card["human_readable_text"]),
            "header_node_ids": header_ids,
        })
    view["schema_nodes"] = [
        item for item in view["schema_nodes"] if str(item["node_id"]) in selected_set
    ]
    view["schema_edges"] = [
        item for item in view["schema_edges"]
        if str(item["source"]) in selected_set and str(item["target"]) in selected_set
    ]
    view["structural_cards"] = compact_cards
    view["planner_view_version"] = "certa_egra_retrieved_structural_view_v1"
    return PlannerViewBuild(True, "", view)
