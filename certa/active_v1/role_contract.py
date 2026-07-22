"""Question-only role contract for the CERTA Active V1 thin adapter."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Tuple

import jsonschema

from certa.operations.contracts import OPERATION_SIGNATURES, operation_signature_telemetry
from certa.reproducibility.canonical_json import canonical_json


ROLE_SCHEMA_VERSION = "certa_active_role_contract_v2"
ROLE_MAX_TOKENS = 192
ROLE_FIELDS = (
    "schema_version",
    "supported",
    "intent",
    "answer_role",
    "projection",
    "signature",
    "cardinality",
    "requires_time_scope",
    "requires_unit_consistency",
)
ROLE_TUPLES: Dict[str, Tuple[str, str, str, str]] = {
    "LOOKUP_VALUE_SCALAR": ("DIRECT_READ", "SCALAR", "VALUE_PROJECTION", "SINGLE"),
    "LOOKUP_VALUE_ENTITY": ("DIRECT_READ", "ENTITY", "VALUE_PROJECTION", "SINGLE"),
    "COUNT_SCALAR": ("COUNT", "SCALAR", "SCALAR_RESULT_PROJECTION", "SINGLE"),
    "SUM_SCALAR": ("SUM", "SCALAR", "SCALAR_RESULT_PROJECTION", "SINGLE"),
    "AVERAGE_SCALAR": ("AVERAGE", "SCALAR", "SCALAR_RESULT_PROJECTION", "SINGLE"),
    "DIFF_SCALAR": ("DIFFERENCE", "SCALAR", "SCALAR_RESULT_PROJECTION", "SINGLE"),
    "RATIO_SCALAR": ("RATIO", "SCALAR", "SCALAR_RESULT_PROJECTION", "SINGLE"),
    "ARGMAX_ENTITY": ("ARGMAX", "ENTITY", "ROW_ENTITY_PROJECTION", "SINGLE"),
    "ARGMAX_ENTITY_SET": ("ARGMAX", "SET", "ROW_ENTITY_PROJECTION", "MULTIPLE"),
    "ARGMIN_ENTITY": ("ARGMIN", "ENTITY", "ROW_ENTITY_PROJECTION", "SINGLE"),
    "ARGMIN_ENTITY_SET": ("ARGMIN", "SET", "ROW_ENTITY_PROJECTION", "MULTIPLE"),
    "PAIR_COMPARE_BOOLEAN": ("PAIR_COMPARE", "BOOLEAN", "BOOLEAN_PROJECTION", "SINGLE"),
}
UNSUPPORTED_TUPLE = ("UNSUPPORTED", "UNSUPPORTED", "UNSUPPORTED", "UNKNOWN")


@dataclass(frozen=True)
class RoleValidation:
    ok: bool
    parse_ok: bool
    wire_valid: bool
    semantic_schema_valid: bool
    local_validator_valid: bool
    payload: Dict[str, Any] = field(default_factory=dict)
    parse_errors: Tuple[str, ...] = ()
    wire_errors: Tuple[str, ...] = ()
    semantic_errors: Tuple[str, ...] = ()
    local_errors: Tuple[str, ...] = ()


def _active_ids(values: Iterable[str]) -> Tuple[str, ...]:
    ids = tuple(sorted(str(item) for item in values))
    if not ids:
        raise ValueError("active_role_signature_set_empty")
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate_active_role_signature")
    unknown = sorted(set(ids) - set(ROLE_TUPLES))
    if unknown:
        raise ValueError(f"unsupported_active_role_signature:{','.join(unknown)}")
    return ids


def build_role_prompt(question: str, active_signature_ids: Iterable[str]) -> str:
    """Build the single-call role prompt; the question is the only sample input."""
    ids = _active_ids(active_signature_ids)
    instructions = {
        "task": "Classify the question into exactly one frozen CERTA Active V1 role tuple.",
        "schema_version": ROLE_SCHEMA_VERSION,
        "active_signatures": {
            signature_id: {
                "role_tuple": list(ROLE_TUPLES[signature_id]),
                "operation_contract": operation_signature_telemetry(signature_id),
            }
            for signature_id in ids
        },
        "rules": [
            "Use only the supplied question.",
            "Return supported=false and the exact UNSUPPORTED tuple when no active signature applies.",
            "Choose exactly one active signature; do not combine interpretations.",
            "The two requirement flags describe the question and do not activate a signature.",
            "Return JSON conforming exactly to the constrained schema.",
        ],
    }
    return (
        "CERTA Active V1 Question Role Contract\n"
        "Return JSON only.\n"
        f"Instructions:\n{canonical_json(instructions)}\n"
        f"Input:\n{canonical_json({'question': str(question or '')})}"
    )


def build_role_wire_schema(active_signature_ids: Iterable[str]) -> Dict[str, Any]:
    """Return the flat transport-safe nine-field schema."""
    ids = _active_ids(active_signature_ids)
    tuples = [ROLE_TUPLES[item] for item in ids]
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "CERTA Active V1 role wire contract",
        "type": "object",
        "properties": {
            "schema_version": {"type": "string", "const": ROLE_SCHEMA_VERSION},
            "supported": {"type": "boolean"},
            "intent": {"type": "string", "enum": sorted({item[0] for item in tuples} | {"UNSUPPORTED"})},
            "answer_role": {"type": "string", "enum": sorted({item[1] for item in tuples} | {"UNSUPPORTED"})},
            "projection": {"type": "string", "enum": sorted({item[2] for item in tuples} | {"UNSUPPORTED"})},
            "signature": {"type": "string", "enum": [*ids, "UNSUPPORTED"]},
            "cardinality": {"type": "string", "enum": sorted({item[3] for item in tuples} | {"UNKNOWN"})},
            "requires_time_scope": {"type": "boolean"},
            "requires_unit_consistency": {"type": "boolean"},
        },
        "required": list(ROLE_FIELDS),
        "additionalProperties": False,
    }


def build_role_semantic_schema(active_signature_ids: Iterable[str]) -> Dict[str, Any]:
    """Return the local full schema enumerating every exact authorized core tuple."""
    ids = _active_ids(active_signature_ids)
    schema = build_role_wire_schema(ids)
    variants = []
    for signature_id in ids:
        intent, answer_role, projection, cardinality = ROLE_TUPLES[signature_id]
        variants.append({
            "properties": {
                "supported": {"const": True},
                "intent": {"const": intent},
                "answer_role": {"const": answer_role},
                "projection": {"const": projection},
                "signature": {"const": signature_id},
                "cardinality": {"const": cardinality},
            },
        })
    variants.append({
        "properties": {
            "supported": {"const": False},
            "intent": {"const": UNSUPPORTED_TUPLE[0]},
            "answer_role": {"const": UNSUPPORTED_TUPLE[1]},
            "projection": {"const": UNSUPPORTED_TUPLE[2]},
            "signature": {"const": "UNSUPPORTED"},
            "cardinality": {"const": UNSUPPORTED_TUPLE[3]},
            "requires_time_scope": {"const": False},
            "requires_unit_consistency": {"const": False},
        },
    })
    schema["title"] = "CERTA Active V1 full semantic role contract"
    schema["oneOf"] = variants
    return schema


def _parse(payload: Any) -> tuple[Dict[str, Any], Tuple[str, ...]]:
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as error:
            return {}, (f"invalid_json:{error.msg}",)
    if not isinstance(payload, Mapping):
        return {}, ("payload_not_object",)
    return dict(payload), ()


def _schema_errors(payload: Mapping[str, Any], schema: Mapping[str, Any]) -> Tuple[str, ...]:
    errors = jsonschema.Draft202012Validator(schema).iter_errors(payload)
    return tuple(sorted(
        f"{'.'.join(str(item) for item in error.absolute_path) or '$'}:{error.validator}:{error.message}"
        for error in errors
    ))


def _local_errors(payload: Mapping[str, Any], active_ids: Tuple[str, ...]) -> Tuple[str, ...]:
    errors = []
    if set(payload) != set(ROLE_FIELDS):
        errors.append("role_field_set_mismatch")
    if payload.get("schema_version") != ROLE_SCHEMA_VERSION:
        errors.append("role_schema_version_mismatch")
    if type(payload.get("supported")) is not bool:
        errors.append("supported_not_boolean")
    for field_name in ("intent", "answer_role", "projection", "signature", "cardinality"):
        if not isinstance(payload.get(field_name), str):
            errors.append(f"{field_name}_not_string")
    for field_name in ("requires_time_scope", "requires_unit_consistency"):
        if type(payload.get(field_name)) is not bool:
            errors.append(f"{field_name}_not_boolean")
    signature = str(payload.get("signature") or "")
    actual = (
        str(payload.get("intent") or ""),
        str(payload.get("answer_role") or ""),
        str(payload.get("projection") or ""),
        str(payload.get("cardinality") or ""),
    )
    if payload.get("supported") is True:
        if signature not in active_ids or ROLE_TUPLES.get(signature) != actual:
            errors.append("role_tuple_not_authorized")
    elif payload.get("supported") is False:
        if signature != "UNSUPPORTED" or actual != UNSUPPORTED_TUPLE:
            errors.append("unsupported_tuple_not_canonical")
        if payload.get("requires_time_scope") is not False or payload.get("requires_unit_consistency") is not False:
            errors.append("unsupported_requirement_flags_not_canonical")
    return tuple(sorted(set(errors)))


def validate_role_contract(payload: Any, active_signature_ids: Iterable[str]) -> RoleValidation:
    """Apply wire, full semantic-schema, and independent local validation without repair."""
    ids = _active_ids(active_signature_ids)
    parsed, parse_errors = _parse(payload)
    if parse_errors:
        return RoleValidation(False, False, False, False, False, {}, parse_errors)
    wire_errors = _schema_errors(parsed, build_role_wire_schema(ids))
    semantic_errors = _schema_errors(parsed, build_role_semantic_schema(ids))
    local_errors = _local_errors(parsed, ids)
    return RoleValidation(
        ok=not wire_errors and not semantic_errors and not local_errors,
        parse_ok=True,
        wire_valid=not wire_errors,
        semantic_schema_valid=not semantic_errors,
        local_validator_valid=not local_errors,
        payload=parsed,
        wire_errors=wire_errors,
        semantic_errors=semantic_errors,
        local_errors=local_errors,
    )


def to_egra_retrieval_contract(role: Mapping[str, Any]) -> Dict[str, Any]:
    """Derive the non-authoritative legacy retrieval-only compatibility view."""
    supported = role.get("supported") is True
    signature_id = str(role.get("signature") or "")
    intent = str(role.get("intent") or "")
    expected = ROLE_TUPLES.get(signature_id) if supported else UNSUPPORTED_TUPLE
    actual = (
        str(role.get("intent") or ""), str(role.get("answer_role") or ""),
        str(role.get("projection") or ""), str(role.get("cardinality") or ""),
    )
    if expected != actual or (supported and signature_id not in ROLE_TUPLES):
        raise ValueError("invalid_active_role_for_retrieval_projection")
    intent_family = {"ARGMAX": "RANK_MAX", "ARGMIN": "RANK_MIN"}.get(intent, intent)
    return {
        "supported_by_core_signatures": supported,
        "answer_domain": str(role["answer_role"]),
        "intent_family": intent_family,
        "signature_candidates": [signature_id] if supported else [],
        "projection_candidates": [str(role["projection"])] if supported else [],
        "cardinality": str(role["cardinality"]),
        "rank_direction": {"ARGMAX": "MAX", "ARGMIN": "MIN"}.get(intent, "NONE"),
        "rank_k": None,
        "requires_time_scope": bool(role["requires_time_scope"]),
        "requires_unit_consistency": bool(role["requires_unit_consistency"]),
        "unknowns": [],
    }


def role_to_query_contract(role: Mapping[str, Any]) -> Dict[str, Any]:
    """Map one authoritative role to the existing Planner query-semantics shape."""
    if role.get("supported") is not True:
        raise ValueError("unsupported_role_has_no_planner_query_contract")
    signature_id = str(role.get("signature") or "")
    signature = OPERATION_SIGNATURES.get(signature_id)
    if signature is None or ROLE_TUPLES.get(signature_id) != (
        role.get("intent"), role.get("answer_role"), role.get("projection"), role.get("cardinality")
    ):
        raise ValueError("invalid_active_role_for_planner_projection")
    constraints = []
    if role.get("requires_time_scope"):
        constraints.append("time_scope_required")
    if role.get("requires_unit_consistency"):
        constraints.append("unit_consistency_required")
    return {
        "answer_domain": signature.answer_domain,
        "allowed_answer_domains": [signature.answer_domain],
        "allowed_projection_operators": [signature.projection_operator],
        "candidate_independent_operation_hypotheses": [signature.operation_family],
        "unit_or_scale_constraints": constraints,
    }
