"""Question-only answer-role contract for CERTA-EGRA."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping

import jsonschema

from certa.operations.contracts import OPERATION_SIGNATURES, operation_signature_telemetry
from certa.reproducibility.canonical_json import canonical_json, canonical_json_hash


QUERY_ROLE_CONTRACT_VERSION = "certa_egra_query_contract_v1"
QUERY_ROLE_MAX_TOKENS = 256
FROZEN_MODEL = "Qwen3-8B"
FROZEN_BACKEND = "vllm_chat"
FROZEN_API_BASE_URL = "http://127.0.0.1:30338/v1"
FROZEN_THINKING = {"enable_thinking": False}
CORE_SIGNATURE_IDS = (
    "LOOKUP_VALUE_SCALAR",
    "LOOKUP_VALUE_ENTITY",
    "COUNT_SCALAR",
    "DIFF_SCALAR",
    "RATIO_SCALAR",
    "ARGMAX_ENTITY",
    "ARGMAX_ENTITY_SET",
    "ARGMIN_ENTITY",
    "ARGMIN_ENTITY_SET",
)

_ANSWER_DOMAINS = ("SCALAR", "ENTITY", "SET", "BOOLEAN", "UNSUPPORTED")
_INTENT_FAMILIES = (
    "DIRECT_READ",
    "COUNT",
    "DIFFERENCE",
    "RATIO",
    "RANK_MAX",
    "RANK_MIN",
    "UNSUPPORTED",
)
_PROJECTIONS = (
    "VALUE_PROJECTION",
    "SCALAR_RESULT_PROJECTION",
    "ROW_ENTITY_PROJECTION",
    "UNSUPPORTED",
)
_INTENT_OPERATION = {
    "DIRECT_READ": "LOOKUP",
    "COUNT": "COUNT",
    "DIFFERENCE": "DIFF",
    "RATIO": "RATIO",
    "RANK_MAX": "ARGMAX",
    "RANK_MIN": "ARGMIN",
}


@dataclass(frozen=True)
class QueryRoleValidation:
    ok: bool
    parse_ok: bool
    errors: tuple[str, ...] = ()
    normalized_payload: Dict[str, Any] = field(default_factory=dict)


def build_query_role_response_schema() -> Dict[str, Any]:
    """Return the frozen constrained-output schema from the EGRA Pack."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "schema_version": {"const": QUERY_ROLE_CONTRACT_VERSION},
            "supported_by_core_signatures": {"type": "boolean"},
            "answer_domain": {"enum": list(_ANSWER_DOMAINS)},
            "intent_family": {"enum": list(_INTENT_FAMILIES)},
            "signature_candidates": {
                "type": "array",
                "items": {"enum": list(CORE_SIGNATURE_IDS)},
                "minItems": 0,
                "maxItems": 3,
                "uniqueItems": True,
            },
            "projection_candidates": {
                "type": "array",
                "items": {"enum": list(_PROJECTIONS)},
                "minItems": 0,
                "maxItems": 2,
                "uniqueItems": True,
            },
            "cardinality": {
                "enum": ["SINGLE", "MULTIPLE", "EXACT_K", "UNKNOWN"],
            },
            "rank_direction": {
                "enum": ["MAX", "MIN", "KTH", "NONE", "UNKNOWN"],
            },
            "rank_k": {"type": ["integer", "null"], "minimum": 1},
            "requires_time_scope": {"type": "boolean"},
            "requires_unit_consistency": {"type": "boolean"},
            "unknowns": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "schema_version",
            "supported_by_core_signatures",
            "answer_domain",
            "intent_family",
            "signature_candidates",
            "projection_candidates",
            "cardinality",
            "rank_direction",
            "rank_k",
            "requires_time_scope",
            "requires_unit_consistency",
            "unknowns",
        ],
        "additionalProperties": False,
        "allOf": [{
            "if": {
                "properties": {"supported_by_core_signatures": {"const": False}},
            },
            "then": {
                "properties": {
                    "answer_domain": {"const": "UNSUPPORTED"},
                    "intent_family": {"const": "UNSUPPORTED"},
                    "signature_candidates": {"maxItems": 0},
                    "projection_candidates": {"maxItems": 0},
                }
            },
        }],
        "title": "CERTA-EGRA question-only role contract",
    }


def build_query_role_prompt(question: str) -> str:
    """Build the only prompt in this stage; its sole sample input is the question."""
    fixed_contract = {
        "task": "Classify the question into the frozen CERTA-EGRA answer-role contract.",
        "core_signature_semantics": {
            signature_id: operation_signature_telemetry(signature_id)
            for signature_id in CORE_SIGNATURE_IDS
        },
        "rules": [
            "Use only the supplied question.",
            "Return unsupported when none of the frozen signatures expresses the question.",
            "Do not infer a KTH ranking because the frozen signatures support only extrema.",
            "Order signature_candidates with the primary interpretation first.",
            "answer_domain, intent_family, cardinality, and rank fields describe the primary signature.",
            "When a secondary signature differs in intent or answer domain, name that field in unknowns.",
            "Return JSON conforming exactly to the constrained schema.",
        ],
    }
    return (
        "CERTA-EGRA Question Role Contract\n"
        f"Instructions:\n{canonical_json(fixed_contract)}\n"
        f"Input:\n{canonical_json({'question': str(question or '')})}"
    )


def _parse_payload(payload: Any) -> tuple[Dict[str, Any], list[str]]:
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return {}, ["invalid_json"]
    if not isinstance(payload, Mapping):
        return {}, ["payload_not_object"]
    return dict(payload), []


def validate_query_role_contract(payload: Any) -> QueryRoleValidation:
    """Apply schema and registry-derived semantic checks without repairing output."""
    parsed, errors = _parse_payload(payload)
    if errors:
        return QueryRoleValidation(False, False, tuple(errors), {})
    try:
        jsonschema.validate(parsed, build_query_role_response_schema())
    except jsonschema.ValidationError as error:
        return QueryRoleValidation(
            False,
            True,
            (f"schema_violation:{error.validator}:{error.message}",),
            parsed,
        )

    if not parsed["supported_by_core_signatures"]:
        return QueryRoleValidation(True, True, (), parsed)

    semantic_errors: list[str] = []
    signature_ids = list(parsed["signature_candidates"])
    if not signature_ids:
        semantic_errors.append("supported_without_signature_candidate")
    intent = str(parsed["intent_family"])
    expected_operation = _INTENT_OPERATION.get(intent, "")
    expected_projections = set()
    unknowns = {str(item) for item in parsed["unknowns"]}
    for index, signature_id in enumerate(signature_ids):
        signature = OPERATION_SIGNATURES[signature_id]
        if signature.operation_family != expected_operation:
            if index == 0:
                semantic_errors.append(f"signature_intent_mismatch:{signature_id}")
            elif "intent_family" not in unknowns:
                semantic_errors.append("unnamed_candidate_uncertainty:intent_family")
        if signature.answer_domain != parsed["answer_domain"]:
            if index == 0:
                semantic_errors.append(f"signature_answer_domain_mismatch:{signature_id}")
            elif "answer_domain" not in unknowns:
                semantic_errors.append("unnamed_candidate_uncertainty:answer_domain")
        expected_projections.add(signature.projection_operator)
    if set(parsed["projection_candidates"]) != expected_projections:
        semantic_errors.append("projection_candidates_do_not_match_signatures")

    direction = str(parsed["rank_direction"])
    expected_direction = {
        "RANK_MAX": "MAX",
        "RANK_MIN": "MIN",
    }.get(intent, "NONE")
    if direction == "KTH":
        semantic_errors.append("unsupported_rank_direction:KTH")
    elif direction != expected_direction:
        semantic_errors.append(
            f"rank_direction_mismatch:{direction}!={expected_direction}"
        )
    if parsed["rank_k"] is not None:
        semantic_errors.append("rank_k_forbidden_for_core_signatures")
    cardinality = str(parsed["cardinality"])
    if cardinality == "EXACT_K":
        semantic_errors.append("unsupported_cardinality:EXACT_K")
    if signature_ids:
        expected_cardinality = (
            "MULTIPLE"
            if OPERATION_SIGNATURES[signature_ids[0]].answer_domain == "SET"
            else "SINGLE"
        )
        if cardinality != "EXACT_K" and cardinality != expected_cardinality:
            semantic_errors.append(
                f"cardinality_mismatch:{cardinality}!={expected_cardinality}"
            )

    return QueryRoleValidation(
        not semantic_errors,
        True,
        tuple(semantic_errors),
        parsed,
    )


def request_query_role_contract(
    generator: Any,
    question: str,
) -> tuple[QueryRoleValidation, Dict[str, Any]]:
    """Issue one strict non-thinking contract call and return its audit record."""
    transport_identity = {
        "model": str(getattr(generator, "model", "")),
        "backend": str(getattr(generator, "backend_name", "")),
        "api_base_url": str(getattr(generator, "api_base_url", "")),
        "thinking": dict(getattr(generator, "chat_template_kwargs", {}) or {}),
    }
    expected_identity = {
        "model": FROZEN_MODEL,
        "backend": FROZEN_BACKEND,
        "api_base_url": FROZEN_API_BASE_URL,
        "thinking": FROZEN_THINKING,
    }
    if transport_identity != expected_identity:
        raise ValueError(
            "query_role_transport_identity_mismatch:"
            f"{canonical_json(transport_identity)}"
        )
    prompt = build_query_role_prompt(question)
    schema = build_query_role_response_schema()
    schema_hash = canonical_json_hash(schema)
    output = dict(generator.generate_json_schema(
        prompt,
        response_schema=schema,
        schema_name=QUERY_ROLE_CONTRACT_VERSION,
        max_new_tokens=QUERY_ROLE_MAX_TOKENS,
        temperature=0.0,
        top_p=1.0,
    ))
    transport_errors = []
    if output.get("error"):
        transport_errors.append(f"generation_error:{output['error']}")
    if not output.get("structured_output_requested", False):
        transport_errors.append("structured_output_request_not_confirmed")
    if output.get("structured_output_fallback_used", False):
        transport_errors.append("structured_output_fallback_forbidden")
    if output.get("structured_output_mechanism") != "response_format.type=json_schema":
        transport_errors.append("structured_output_mechanism_mismatch")
    if output.get("structured_output_schema_hash") != schema_hash:
        transport_errors.append("structured_output_schema_hash_mismatch")
    if str(output.get("api_model") or "") != FROZEN_MODEL:
        transport_errors.append("output_model_identity_mismatch")
    if str(output.get("generator_backend") or "") != FROZEN_BACKEND:
        transport_errors.append("output_backend_identity_mismatch")
    if str(output.get("api_base_url") or "") != FROZEN_API_BASE_URL:
        transport_errors.append("output_api_base_url_identity_mismatch")
    if dict(output.get("chat_template_kwargs") or {}) != FROZEN_THINKING:
        transport_errors.append("output_thinking_identity_mismatch")

    validation = validate_query_role_contract(output.get("text", ""))
    if transport_errors:
        validation = QueryRoleValidation(
            False,
            validation.parse_ok,
            tuple(transport_errors) + validation.errors,
            validation.normalized_payload,
        )
    model = str(output.get("api_model") or getattr(generator, "model", ""))
    api_base_url = str(
        output.get("api_base_url") or getattr(generator, "api_base_url", "")
    )
    backend = str(
        output.get("generator_backend") or getattr(generator, "backend_name", "")
    )
    sampling = {
        "max_tokens": QUERY_ROLE_MAX_TOKENS,
        "temperature": 0.0,
        "top_p": 1.0,
    }
    thinking = dict(
        output.get("chat_template_kwargs")
        or getattr(generator, "chat_template_kwargs", {})
        or {}
    )
    raw_output = str(output.get("text", "") or "")
    request_identity = {
        "prompt_sha256": canonical_json_hash({"prompt": prompt}),
        "schema_sha256": schema_hash,
        "model": model,
        "backend": backend,
        "api_base_url": api_base_url,
        "sampling": sampling,
        "thinking": thinking,
    }
    audit = {
        "schema_version": "certa_egra_query_contract_audit_v1",
        "calls": 1,
        "request_sha256": canonical_json_hash(request_identity),
        "prompt_sha256": request_identity["prompt_sha256"],
        "schema_sha256": schema_hash,
        "raw_output_sha256": canonical_json_hash({"text": raw_output}),
        "normalized_output_sha256": canonical_json_hash(
            validation.normalized_payload
        ),
        "normalized_output": validation.normalized_payload,
        "model": model,
        "backend": backend,
        "api_base_url": api_base_url,
        "sampling": sampling,
        "thinking": thinking,
        "cache": {
            "hit": bool(output.get("api_cache_hit", False)),
            "mode": str(
                output.get("api_cache_mode")
                or getattr(generator, "cache_mode", "")
            ),
        },
        "prompt_tokens": int(output.get("input_token_count", 0) or 0),
        "completion_tokens": int(output.get("generated_token_count", 0) or 0),
        "latency_seconds": float(output.get("generation_seconds", 0.0) or 0.0),
        "structured_output_fallback_used": bool(
            output.get("structured_output_fallback_used", False)
        ),
        "parse_ok": validation.parse_ok,
        "valid": validation.ok,
        "errors": list(validation.errors),
    }
    return validation, audit
