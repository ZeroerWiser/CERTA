"""Minimal canonical-class Role V3 contract for CERTA Active V1."""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Sequence, Tuple

import jsonschema


ROLE_V3_SCHEMA_VERSION = "certa_active_role_contract_v3"
ROLE_V3_RECORD_VERSION = "certa_active_role_v3_canonical_record_v1"
ROLE_V3_MAX_TOKENS = 64
ROLE_V3_ROLE_IDS = (
    "LOOKUP_VALUE_SCALAR", "LOOKUP_VALUE_ENTITY", "COUNT_SCALAR", "SUM_SCALAR",
    "AVERAGE_SCALAR", "DIFF_SCALAR", "RATIO_SCALAR", "ARGMAX_ENTITY",
    "ARGMAX_ENTITY_SET", "ARGMIN_ENTITY", "ARGMIN_ENTITY_SET",
    "PAIR_COMPARE_BOOLEAN", "UNSUPPORTED",
)
_MODEL_FIELDS = frozenset(("schema_version", "role_id"))
_DERIVED_FIELDS = (
    "supported", "intent", "answer_role", "projection", "cardinality", "operation_family",
)


def _ordered_cards(cards: Mapping[str, Any]) -> Sequence[Dict[str, Any]]:
    rows = cards.get("cards")
    if not isinstance(rows, list):
        raise ValueError("role_v3_cards_missing")
    return [{
        "role_id": row["role_id"],
        "positive_definition": row["positive_definition"],
        "nearest_neighbor_exclusions": row["nearest_neighbor_exclusions"],
        "answer_form": row["answer_form"],
    } for row in rows]


def validate_role_v3_artifacts(
    cards: Mapping[str, Any],
    output_schema: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> Tuple[str, ...]:
    """Require a single, ordered role authority across all frozen artifacts."""
    jsonschema.Draft202012Validator.check_schema(dict(output_schema))
    card_ids = tuple(row["role_id"] for row in _ordered_cards(cards))
    schema_ids = tuple(output_schema.get("properties", {}).get("role_id", {}).get("enum", ()))
    registry_rows = registry.get("roles")
    if not isinstance(registry_rows, list):
        raise ValueError("role_v3_registry_missing")
    registry_ids = tuple(row.get("role_id") for row in registry_rows)
    if card_ids != ROLE_V3_ROLE_IDS or schema_ids != card_ids or registry_ids != card_ids:
        raise ValueError("role_v3_authority_order_mismatch")
    if set(output_schema.get("properties", {})) != _MODEL_FIELDS:
        raise ValueError("role_v3_model_field_set_mismatch")
    if set(output_schema.get("required", ())) != _MODEL_FIELDS:
        raise ValueError("role_v3_required_field_set_mismatch")
    if output_schema.get("additionalProperties") is not False:
        raise ValueError("role_v3_additional_properties_not_false")
    if registry.get("authority") != "role_id_is_the_only_model_selected_field":
        raise ValueError("role_v3_registry_authority_mismatch")
    deferred = registry.get("deferred_authorities", {})
    if deferred != {
        "requires_time_scope": "DEFERRED_TO_GROUNDING",
        "requires_unit_consistency": "DEFERRED_TO_EXECUTION",
    }:
        raise ValueError("role_v3_deferred_authority_mismatch")
    for row in registry_rows:
        if any(field not in row for field in _DERIVED_FIELDS):
            raise ValueError(f"role_v3_registry_row_incomplete:{row.get('role_id')}")
    return card_ids


def build_role_v3_prompt_template(cards: Mapping[str, Any]) -> str:
    role_cards = json.dumps(_ordered_cards(cards), ensure_ascii=False, separators=(",", ":"))
    return (
        "CERTA Active V1 Role V3 Canonical-Class Contract\n"
        "Return JSON only. Select exactly one role_id from the supplied mutually exclusive role cards. "
        "Use only the question. Do not infer from table values, benchmark labels, candidate answers, "
        "or prior predictions. UNSUPPORTED is the selective abstention class when no registered role applies.\n"
        f"Schema version: {ROLE_V3_SCHEMA_VERSION}\n"
        f"Role cards: {role_cards}\n"
        "Question: {{QUESTION_JSON_STRING}}\n"
    )


def build_role_v3_prompt(question: str, cards: Mapping[str, Any]) -> str:
    """Insert the sole sample-specific input as one JSON string."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("role_v3_question_empty")
    return build_role_v3_prompt_template(cards).replace(
        "{{QUESTION_JSON_STRING}}", json.dumps(question, ensure_ascii=False)
    )


def parse_role_v3_output(value: Any, output_schema: Mapping[str, Any]) -> Dict[str, str]:
    """Parse and validate the exact two-field model payload without repair."""
    parsed = json.loads(value) if isinstance(value, str) else dict(value)
    jsonschema.Draft202012Validator(dict(output_schema)).validate(parsed)
    return dict(parsed)


def derive_role_v3_record(
    value: Any,
    output_schema: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> Dict[str, Any]:
    """Derive every non-model field exclusively from the frozen registry."""
    payload = parse_role_v3_output(value, output_schema)
    rows = registry.get("roles")
    if not isinstance(rows, list):
        raise ValueError("role_v3_registry_missing")
    by_id = {row.get("role_id"): row for row in rows}
    row = by_id.get(payload["role_id"])
    if row is None:
        raise ValueError("role_v3_role_id_not_in_registry")
    deferred = registry.get("deferred_authorities", {})
    return {
        "schema_version": ROLE_V3_RECORD_VERSION,
        "role_id": payload["role_id"],
        **{field: row[field] for field in _DERIVED_FIELDS},
        "requires_time_scope": deferred["requires_time_scope"],
        "requires_unit_consistency": deferred["requires_unit_consistency"],
    }


def role_v3_to_planner_query_contract(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Build a V3 Planner view without passing deferred states through V2 booleans."""
    if record.get("supported") is not True:
        raise ValueError("unsupported_role_has_no_planner_query_contract")
    if record.get("requires_time_scope") != "DEFERRED_TO_GROUNDING":
        raise ValueError("role_v3_time_scope_not_deferred")
    if record.get("requires_unit_consistency") != "DEFERRED_TO_EXECUTION":
        raise ValueError("role_v3_unit_consistency_not_deferred")
    return {
        "answer_domain": record["answer_role"],
        "allowed_answer_domains": [record["answer_role"]],
        "allowed_projection_operators": [record["projection"]],
        "candidate_independent_operation_hypotheses": [record["operation_family"]],
        "allowed_signature_ids": [record["role_id"]],
        "time_scope_authority": "DEFERRED_TO_GROUNDING",
        "unit_consistency_authority": "DEFERRED_TO_EXECUTION",
        "unit_or_scale_constraints": [],
    }
