"""Prompt and validator for the Round 7 Typed Derivation Planner Agent."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping

from certa.derivations.schema import ANSWER_DOMAINS, PROJECTION_OPERATORS
from certa.operations.contracts import (
    FINAL_SUPPORTED_OPERATIONS,
    OPERATION_CONTRACT_VERSION,
    OPERATION_SIGNATURES,
    ROLE_VOCABULARY,
    build_operation_plan_schema,
    build_operation_role_schema_defs,
    operation_signature_telemetry,
    resolve_operation_signature,
    validate_operation_plan,
)
from certa.reproducibility.canonical_json import canonical_json, canonical_json_hash


PLANNER_VERSION = "typed_derivation_planner_v1"
PLAN_MIN_COUNT = 1
PLAN_MAX_COUNT = 6
TOP_LEVEL_FIELDS = {"planner_version", "query_semantics", "plans", "unresolved_semantics"}
QUERY_SEMANTICS_FIELDS = {"operation_family", "answer_domain", "projection_operator"}
PLAN_FIELDS = {
    "plan_id",
    "signature_id",
    "operation_family",
    "semantic_result_role",
    "answer_domain",
    "projection_operator",
    "role_bindings",
    "role_domains",
    "comparison_polarity",
    "unresolved_semantics",
}
FORBIDDEN_PROPOSAL_BLIND_KEYS = {
    "a0",
    "initial_answer",
    "original_answer",
    "final_answer",
    "candidate_answer",
    "gold",
    "gold_answer",
    "correct",
    "correctness",
}
STRUCTURAL_PLAN_ID_RE = re.compile(r"^P[0-9]+$")


@dataclass
class PlannerValidationResult:
    ok: bool
    parse_ok: bool = False
    errors: list[str] = field(default_factory=list)
    normalized_payload: Dict[str, Any] = field(default_factory=dict)
    valid_plan_count: int = 0
    plan_rejections: list[Dict[str, Any]] = field(default_factory=list)
    resource_warnings: list[str] = field(default_factory=list)


def _first_forbidden_key(value: Any, key_names: set[str]) -> str:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            if key_text in key_names:
                return key_text
            nested = _first_forbidden_key(child, key_names)
            if nested:
                return nested
    elif isinstance(value, list):
        for item in value:
            nested = _first_forbidden_key(item, key_names)
            if nested:
                return nested
    return ""


def build_typed_derivation_planner_prompt(
    view: Mapping[str, Any],
    *,
    proposal_aware: bool = False,
) -> str:
    """Render the common JSON-only Planner prompt."""
    if not proposal_aware:
        forbidden_key = _first_forbidden_key(view, FORBIDDEN_PROPOSAL_BLIND_KEYS)
        if forbidden_key:
            raise ValueError(f"forbidden_planner_request_key:{forbidden_key}")
    ontology = view.get("operation_ontology") or {}
    declared_signature_values = ontology.get("signature_ids")
    declared_signature_ids = (
        sorted(str(item) for item in declared_signature_values)
        if isinstance(declared_signature_values, list)
        else sorted(OPERATION_SIGNATURES)
    )
    view_signature_variants = ontology.get("signature_variants") or {}
    instructions = {
        "task": "Return typed derivation plan skeletons. Role references must use only schema_nodes[*].node_id values from the Planner view.",
        "operation_contract_version": OPERATION_CONTRACT_VERSION,
        "operation_signature_variants": {
            signature_id: dict(
                view_signature_variants.get(signature_id)
                or operation_signature_telemetry(signature_id)
            )
            for signature_id in declared_signature_ids
        },
        "role_domain_rule": (
            "role_bindings contain one exact value of the signature-declared role shape. "
            "For each role, use exactly one declaration source: role_bindings for one exact "
            "binding or role_domains for finite alternatives; never declare the same role in both."
        ),
        "role_shape_rule": (
            "structural_conjunction is [schema_id, ...]. finite_scope_members is "
            "[[member schema_id, ...], ...]. Do not flatten a finite member scope."
        ),
        "forbidden": [
            "Do not produce final answers.",
            "Do not mention gold answers.",
            "Do not invent schema IDs.",
            "Do not use table cell text as a role reference.",
        ],
        "output_contract": {
            "planner_version": PLANNER_VERSION,
            "query_semantics_fields": sorted(QUERY_SEMANTICS_FIELDS),
            "plan_fields": sorted(PLAN_FIELDS),
            "role_names": sorted(ROLE_VOCABULARY),
            "role_reference_rule": "Every terminal role reference must be copied exactly from schema_nodes[*].node_id.",
            "role_container_grammar": "Each role exactly follows its canonical signature role shape and cardinality.",
            "plan_id_rule": "plan_id is optional; when present it must match P followed by digits and is normalized by output order.",
        },
    }
    return (
        "CERTA Typed Derivation Planner Agent\n"
        "Return JSON only.\n\n"
        f"Instructions:\n{canonical_json(instructions)}\n\n"
        f"Planner view:\n{canonical_json(view)}"
    )


def planner_reference_domain(view: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the exact canonical schema-header node ID domain for a view."""
    return tuple(sorted(_schema_ids(view)))


def _enum_schema(values: Any) -> Dict[str, Any]:
    return {"type": "string", "enum": sorted(str(item) for item in values)}


def build_typed_planner_response_schema(
    view: Mapping[str, Any],
    *,
    require_signature_id: bool = False,
) -> Dict[str, Any]:
    """Build the strict per-view RCPC JSON Schema used by constrained generation."""
    reference_ids = planner_reference_domain(view)
    ontology = view.get("operation_ontology") or {}
    operation_values = ontology.get("operation_families")
    signature_values = ontology.get("signature_ids")
    declared_operations = {
        str(item) for item in (
            operation_values if isinstance(operation_values, list) else FINAL_SUPPORTED_OPERATIONS
        )
    }
    declared_signatures = {
        str(item) for item in (
            signature_values if isinstance(signature_values, list) else OPERATION_SIGNATURES
        )
    }
    signatures = [
        signature
        for signature_id, signature in OPERATION_SIGNATURES.items()
        if signature_id in declared_signatures and signature.operation_family in declared_operations
    ]
    domain_values = ontology.get("answer_domains")
    projection_values = ontology.get("projection_operators")
    domains = (
        domain_values
        if isinstance(domain_values, list)
        else sorted(ANSWER_DOMAINS - {"UNKNOWN"})
    )
    projections = (
        projection_values
        if isinstance(projection_values, list)
        else sorted(PROJECTION_OPERATORS - {"UNKNOWN"})
    )
    unresolved = {"type": "array", "items": {"type": "string"}}
    query_variants = []
    for signature in signatures:
        if signature.projection_operator not in projections or signature.answer_domain not in domains:
            continue
        query_variants.append({
            "type": "object",
            "properties": {
                "operation_family": {"type": "string", "const": signature.operation_family},
                "answer_domain": {"type": "string", "const": signature.answer_domain},
                "projection_operator": {"type": "string", "const": signature.projection_operator},
            },
            "required": sorted(QUERY_SEMANTICS_FIELDS),
            "additionalProperties": False,
        })
    query_semantics = {"anyOf": query_variants}
    plan_identifiers = (
        [signature.signature_id for signature in signatures]
        if require_signature_id
        else sorted({signature.operation_family for signature in signatures})
    )
    plan = {"anyOf": [
        build_operation_plan_schema(
            identifier,
            reference_ids,
            include_shared_defs=False,
            require_signature_id=require_signature_id,
        )
        for identifier in plan_identifiers
    ]}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "planner_version": {"type": "string", "const": PLANNER_VERSION},
            "query_semantics": query_semantics,
            "plans": {"type": "array", "items": plan, "minItems": PLAN_MIN_COUNT},
            "unresolved_semantics": unresolved,
        },
        "required": sorted(TOP_LEVEL_FIELDS),
        "additionalProperties": False,
        "$defs": build_operation_role_schema_defs(reference_ids),
    }


def planner_constraint_schema_hash(
    view: Mapping[str, Any],
    *,
    require_signature_id: bool = False,
) -> str:
    return canonical_json_hash(build_typed_planner_response_schema(
        view,
        require_signature_id=require_signature_id,
    ))


def _parse_payload(payload: Any) -> tuple[Dict[str, Any], list[str]]:
    if isinstance(payload, str):
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            return {}, ["invalid_json"]
        if not isinstance(parsed, dict):
            return {}, ["json_root_not_object"]
        return parsed, []
    if isinstance(payload, Mapping):
        return dict(payload), []
    return {}, ["payload_not_object"]


def _contains_key_recursive(value: Any, key_names: set[str]) -> bool:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key) in key_names:
                return True
            if _contains_key_recursive(child, key_names):
                return True
    elif isinstance(value, list):
        return any(_contains_key_recursive(item, key_names) for item in value)
    return False


def _schema_ids(view: Mapping[str, Any]) -> set[str]:
    return {str(item.get("node_id")) for item in view.get("schema_nodes", []) if item.get("node_id")}


def _allowed_from_view(view: Mapping[str, Any], key: str, fallback: set[str]) -> set[str]:
    ontology = view.get("operation_ontology") or {}
    values = ontology.get(key)
    if isinstance(values, list):
        return {str(item) for item in values}
    return set(fallback)


def _query_semantics(view: Mapping[str, Any]) -> Mapping[str, Any]:
    semantics = view.get("query_semantics") or {}
    return semantics if isinstance(semantics, Mapping) else {}


def _semantic_allowed_values(semantics: Mapping[str, Any], key: str) -> set[str]:
    values = semantics.get(key)
    if isinstance(values, list):
        return {str(item) for item in values if str(item) and str(item) != "UNKNOWN"}
    value = str(values or "")
    return {value} if value and value != "UNKNOWN" else set()


def _operation_semantically_compatible(operation_family: str, allowed_ops: set[str]) -> bool:
    if not allowed_ops:
        return True
    if operation_family in allowed_ops:
        return True
    if operation_family == "LOOKUP_AGGREGATE" and "LOOKUP" in allowed_ops:
        return True
    if operation_family in {"ARGMAX", "ARGMIN"} and "PAIR_COMPARE" in allowed_ops:
        return True
    return False


def _query_semantic_errors(
    *,
    plan_id: str,
    operation_family: str,
    answer_domain: str,
    projection_operator: str,
    semantics: Mapping[str, Any],
) -> list[str]:
    errors: list[str] = []
    allowed_ops = _semantic_allowed_values(semantics, "candidate_independent_operation_hypotheses")
    allowed_domains = _semantic_allowed_values(semantics, "allowed_answer_domains")
    allowed_projections = _semantic_allowed_values(semantics, "allowed_projection_operators")
    if not _operation_semantically_compatible(operation_family, allowed_ops):
        errors.append(f"query_operation_incompatible:{plan_id}:{operation_family}")
    if allowed_domains and answer_domain not in allowed_domains:
        errors.append(f"query_answer_domain_incompatible:{plan_id}:{answer_domain}")
    if allowed_projections and projection_operator not in allowed_projections:
        errors.append(f"query_projection_incompatible:{plan_id}:{projection_operator}")
    return errors


def _plan_signature(plan: Mapping[str, Any]) -> str:
    return canonical_json(
        {
            "signature_id": plan.get("signature_id"),
            "operation_family": plan.get("operation_family"),
            "semantic_result_role": plan.get("semantic_result_role"),
            "answer_domain": plan.get("answer_domain"),
            "projection_operator": plan.get("projection_operator"),
            "role_bindings": plan.get("role_bindings") or {},
            "role_domains": plan.get("role_domains") or {},
        }
    )


def _validate_role_container(
    *,
    plan_id: str,
    container_name: str,
    container: Any,
    schema_ids: set[str],
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    if container is None:
        return {}, errors
    if not isinstance(container, Mapping):
        return {}, [f"{container_name}_not_object:{plan_id}"]
    normalized: dict[str, Any] = {}
    for role, value in container.items():
        role_name = str(role)
        if role_name not in ROLE_VOCABULARY:
            errors.append(f"unknown_role:{plan_id}:{role_name}")
        normalized[role_name] = value
    return normalized, errors


def _json_role_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_role_value(item) for item in value]
    return value


def _legacy_json_domain_value(value: Any) -> Any:
    converted = _json_role_value(value)
    if (
        isinstance(converted, list)
        and converted
        and all(isinstance(option, list) and len(option) == 1 for option in converted)
    ):
        return [option[0] for option in converted]
    return converted


def validate_typed_planner_output(
    payload: Any,
    view: Mapping[str, Any],
    *,
    require_signature_id: bool = False,
) -> PlannerValidationResult:
    parsed, errors = _parse_payload(payload)
    if errors:
        return PlannerValidationResult(ok=False, parse_ok=False, errors=errors)

    fatal_errors: list[str] = []
    plan_rejections: list[Dict[str, Any]] = []
    resource_warnings: list[str] = []
    unknown_top_fields = sorted(set(parsed) - TOP_LEVEL_FIELDS)
    fatal_errors.extend(f"unknown_top_level_field:{key}" for key in unknown_top_fields)
    if "unresolved_semantics" not in parsed:
        fatal_errors.append("missing_top_level_unresolved_semantics")
        top_unresolved_semantics: list[str] = []
    elif not isinstance(parsed["unresolved_semantics"], list):
        fatal_errors.append("top_level_unresolved_semantics_not_list")
        top_unresolved_semantics = []
    else:
        top_unresolved_semantics = list(parsed["unresolved_semantics"])
        if any(not isinstance(item, str) for item in top_unresolved_semantics):
            fatal_errors.append("top_level_unresolved_semantics_item_not_string")

    query_semantics = parsed.get("query_semantics") or {}
    if not isinstance(query_semantics, Mapping):
        fatal_errors.append("query_semantics_not_object")
        query_semantics = {}
    unknown_semantic_fields = sorted(set(query_semantics) - QUERY_SEMANTICS_FIELDS)
    fatal_errors.extend(f"unknown_query_semantics_field:{key}" for key in unknown_semantic_fields)
    missing_semantic_fields = sorted(QUERY_SEMANTICS_FIELDS - set(query_semantics))
    fatal_errors.extend(f"missing_query_semantics_field:{key}" for key in missing_semantic_fields)
    normalized_query_semantics = {
        key: query_semantics[key]
        for key in QUERY_SEMANTICS_FIELDS
        if key in query_semantics
    }
    query_operation = str(query_semantics.get("operation_family") or "")
    query_domain = str(query_semantics.get("answer_domain") or "")
    query_projection = str(query_semantics.get("projection_operator") or "")
    query_contract = resolve_operation_signature({
        "operation_family": query_operation,
        "answer_domain": query_domain,
        "projection_operator": query_projection,
    })
    query_allowed_ops = _allowed_from_view(
        view,
        "operation_families",
        set(FINAL_SUPPORTED_OPERATIONS),
    )
    if query_operation not in query_allowed_ops:
        fatal_errors.append(f"invalid_query_operation_family:{query_operation}")
    elif query_contract is None:
        fatal_errors.append(
            "invalid_query_projection_domain_pair:"
            f"{query_operation}:{query_projection}:{query_domain}"
        )
    declared_signature_ids = _allowed_from_view(
        view,
        "signature_ids",
        set(OPERATION_SIGNATURES),
    )
    declared_query_tuples = {
        (
            OPERATION_SIGNATURES[signature_id].operation_family,
            OPERATION_SIGNATURES[signature_id].projection_operator,
            OPERATION_SIGNATURES[signature_id].answer_domain,
        )
        for signature_id in declared_signature_ids
        if signature_id in OPERATION_SIGNATURES
    }
    if (query_operation, query_projection, query_domain) not in declared_query_tuples:
        fatal_errors.append(
            "query_projection_domain_tuple_not_declared:"
            f"{query_operation}:{query_projection}:{query_domain}"
        )

    if parsed.get("planner_version") != PLANNER_VERSION:
        fatal_errors.append("planner_version_invalid")

    plans = parsed.get("plans")
    if not isinstance(plans, list):
        fatal_errors.append("plans_not_list")
        plans = []
    if len(plans) < PLAN_MIN_COUNT:
        fatal_errors.append("too_few_plans")
    if len(plans) > PLAN_MAX_COUNT:
        resource_warnings.append(f"plan_count_exceeds_budget:{len(plans)}>{PLAN_MAX_COUNT}")

    schema_ids = _schema_ids(view)
    allowed_ops = _allowed_from_view(view, "operation_families", set(FINAL_SUPPORTED_OPERATIONS))
    allowed_domains = _allowed_from_view(view, "answer_domains", ANSWER_DOMAINS - {"UNKNOWN"})
    allowed_projections = _allowed_from_view(view, "projection_operators", PROJECTION_OPERATORS - {"UNKNOWN"})
    allowed_signature_ids = _allowed_from_view(view, "signature_ids", set(OPERATION_SIGNATURES))
    semantics = _query_semantics(view)
    seen_signatures: set[str] = set()
    normalized_plans = []

    for index, raw_plan in enumerate(plans):
        plan_id = f"P{index}"
        plan_errors: list[str] = []
        if not isinstance(raw_plan, Mapping):
            plan_rejections.append({"plan_id": plan_id, "reasons": [f"plan_not_object:{plan_id}"]})
            continue
        plan = dict(raw_plan)
        plan_id = str(plan.get("plan_id") or plan_id)
        unknown_plan_fields = sorted(set(plan) - PLAN_FIELDS)
        plan_errors.extend(f"unknown_plan_field:{plan_id}:{key}" for key in unknown_plan_fields)
        if not STRUCTURAL_PLAN_ID_RE.fullmatch(plan_id):
            plan_errors.append(f"plan_id_not_structural:{plan_id}")
        if "unresolved_semantics" not in plan:
            plan_errors.append(f"missing_plan_unresolved_semantics:{plan_id}")
            plan_unresolved_semantics: list[str] = []
        elif not isinstance(plan["unresolved_semantics"], list):
            plan_errors.append(f"plan_unresolved_semantics_not_list:{plan_id}")
            plan_unresolved_semantics = []
        else:
            plan_unresolved_semantics = list(plan["unresolved_semantics"])
            if any(not isinstance(item, str) for item in plan_unresolved_semantics):
                plan_errors.append(f"plan_unresolved_semantics_item_not_string:{plan_id}")

        operation_family = str(plan.get("operation_family") or "")
        strict_plan = bool(plan.get("signature_id"))
        if require_signature_id and not strict_plan:
            plan_errors.append(f"missing_signature_id:{plan_id}")
        signature_id = str(plan.get("signature_id") or "")
        semantic_result_role = str(plan.get("semantic_result_role") or "")
        answer_domain = str(plan.get("answer_domain") or "")
        projection_operator = str(plan.get("projection_operator") or "")
        if operation_family not in allowed_ops:
            plan_errors.append(f"unknown_operation_family:{plan_id}:{operation_family}")
        if signature_id and signature_id not in allowed_signature_ids:
            plan_errors.append(f"unknown_signature_id:{plan_id}:{signature_id}")
        if answer_domain not in allowed_domains:
            plan_errors.append(f"invalid_answer_domain:{plan_id}:{answer_domain}")
        if projection_operator not in allowed_projections:
            plan_errors.append(f"invalid_projection_operator:{plan_id}:{projection_operator}")
        plan_errors.extend(_query_semantic_errors(
            plan_id=plan_id,
            operation_family=operation_family,
            answer_domain=answer_domain,
            projection_operator=projection_operator,
            semantics=semantics,
        ))

        role_bindings, role_binding_errors = _validate_role_container(
            plan_id=plan_id,
            container_name="role_bindings",
            container=plan.get("role_bindings"),
            schema_ids=schema_ids,
        )
        role_domains, role_domain_errors = _validate_role_container(
            plan_id=plan_id,
            container_name="role_domains",
            container=plan.get("role_domains"),
            schema_ids=schema_ids,
        )
        plan_errors.extend(role_binding_errors)
        plan_errors.extend(role_domain_errors)
        contract_plan = {
            "signature_id": signature_id,
            "operation_family": operation_family,
            "semantic_result_role": semantic_result_role,
            "answer_domain": answer_domain,
            "projection_operator": projection_operator,
            "role_bindings": role_bindings,
            "role_domains": role_domains,
            "unresolved_semantics": plan_unresolved_semantics,
        }
        if "comparison_polarity" in plan:
            contract_plan["comparison_polarity"] = plan.get("comparison_polarity")
        contract_validation = validate_operation_plan(contract_plan, schema_ids)
        plan_errors.extend(
            f"operation_contract:{plan_id}:{error}"
            for error in contract_validation.errors
        )
        for error in contract_validation.errors:
            if error.startswith("unknown_schema_id:"):
                _, role_name, schema_id = error.split(":", 2)
                plan_errors.append(
                    f"unknown_schema_id:{plan_id}:{role_name}:{schema_id}"
                )
        if contract_validation.ok:
            signature_id = contract_validation.signature_id
            signature = OPERATION_SIGNATURES[signature_id]
            semantic_result_role = signature.semantic_result_role
            role_bindings = {
                role: _json_role_value(values)
                for role, values in contract_validation.normalized_role_bindings
            }
            role_domains = {
                role: (
                    _json_role_value(options)
                    if strict_plan
                    else _legacy_json_domain_value(options)
                )
                for role, options in contract_validation.normalized_role_domains
            }
        normalized_plan: Dict[str, Any] = {
            "plan_id": "",
            "operation_family": operation_family,
            "answer_domain": answer_domain,
            "projection_operator": projection_operator,
            "role_bindings": role_bindings,
            "unresolved_semantics": plan_unresolved_semantics,
        }
        if strict_plan:
            normalized_plan["signature_id"] = signature_id
            normalized_plan["semantic_result_role"] = semantic_result_role
        if "comparison_polarity" in plan:
            normalized_plan["comparison_polarity"] = str(plan.get("comparison_polarity") or "")
        if role_domains:
            normalized_plan["role_domains"] = role_domains
        elif "role_domains" in plan:
            normalized_plan["role_domains"] = {}
        signature = _plan_signature(normalized_plan)
        seen_signatures.add(signature)
        if plan_errors:
            plan_rejections.append({"plan_id": plan_id, "reasons": plan_errors})
            errors.extend(plan_errors)
            continue
        normalized_plan["plan_id"] = f"P{len(normalized_plans)}"
        normalized_plans.append(normalized_plan)

    normalized = {
        "planner_version": parsed.get("planner_version"),
        "query_semantics": normalized_query_semantics,
        "plans": normalized_plans,
        "unresolved_semantics": top_unresolved_semantics,
    }
    ok = bool(normalized_plans) and not fatal_errors
    return PlannerValidationResult(
        ok=ok,
        parse_ok=True,
        errors=fatal_errors + errors,
        normalized_payload=normalized,
        valid_plan_count=len(normalized_plans),
        plan_rejections=plan_rejections,
        resource_warnings=resource_warnings,
    )
