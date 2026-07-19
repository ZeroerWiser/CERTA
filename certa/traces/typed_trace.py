"""Proposal-blind typed stepwise reasoning representation for CERTA Round 12."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from certa.operations.contracts import (
    OPERATION_CONTRACT_VERSION,
    OPERATION_SIGNATURES,
    OperationSignatureVariant,
    get_operation_signature,
    validate_operation_plan,
)
from certa.grounding.structural_resolvers import resolve_atomic_operand
from certa.reproducibility.canonical_json import canonical_json_hash


TRACE_VERSION = "typed_executable_reasoning_trace_v1"
_INTENT_ID_RE = re.compile(r"^I[0-9]+$")
_ROLE_STEP_ID_RE = re.compile(r"^R[0-9]+$")
_FORBIDDEN_KEYS = {
    "proposal",
    "initial_answer",
    "candidate_answer",
    "final_answer",
    "gold",
    "gold_answer",
    "correct",
    "correctness",
    "confidence",
    "score",
}


class VerificationStage(str, Enum):
    INTENT_CONTRACT = "INTENT_CONTRACT"
    ROLE_SHAPE = "ROLE_SHAPE"
    ROLE_REFERENCE = "ROLE_REFERENCE"
    STRUCTURAL_GROUNDING = "STRUCTURAL_GROUNDING"
    EXECUTION = "EXECUTION"
    PROJECTION = "PROJECTION"
    OUTPUT_TYPE = "OUTPUT_TYPE"


class VerificationStatus(str, Enum):
    PASS = "PASS"
    STRUCTURALLY_INVALID = "STRUCTURALLY_INVALID"
    UNRESOLVED = "UNRESOLVED"
    AMBIGUOUS = "AMBIGUOUS"
    RESOURCE_INCOMPLETE = "RESOURCE_INCOMPLETE"
    EXECUTION_FAILED = "EXECUTION_FAILED"
    PROJECTION_FAILED = "PROJECTION_FAILED"
    TYPE_MISMATCH = "TYPE_MISMATCH"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True)
class IntentHypothesis:
    intent_id: str
    signature_id: str
    comparison_polarity: str = ""
    unresolved_semantics: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "intent_id": self.intent_id,
            "signature_id": self.signature_id,
            "unresolved_semantics": list(self.unresolved_semantics),
        }
        if self.comparison_polarity:
            payload["comparison_polarity"] = self.comparison_polarity
        return payload


@dataclass(frozen=True)
class RoleBindingStep:
    step_id: str
    intent_id: str
    role_name: str
    role_shape: str
    binding_options: Tuple[Any, ...]
    unresolved_semantics: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "intent_id": self.intent_id,
            "role_name": self.role_name,
            "role_shape": self.role_shape,
            "binding_options": _jsonable(self.binding_options),
            "unresolved_semantics": list(self.unresolved_semantics),
        }


@dataclass(frozen=True)
class TraceVerificationStep:
    step_id: str
    stage: VerificationStage
    status: VerificationStatus
    responsible_fields: Tuple[str, ...] = ()
    structural_references: Tuple[str, ...] = ()
    candidate_match_count: int = 0
    required_edge_provenance: Tuple[Tuple[str, str, str], ...] = ()
    failure_reasons: Tuple[str, ...] = ()
    resource_complete: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "stage": self.stage.value,
            "status": self.status.value,
            "responsible_fields": list(self.responsible_fields),
            "structural_references": list(self.structural_references),
            "candidate_match_count": self.candidate_match_count,
            "required_edge_provenance": [list(item) for item in self.required_edge_provenance],
            "failure_reasons": list(self.failure_reasons),
            "resource_complete": self.resource_complete,
        }


@dataclass(frozen=True)
class FirstVerifiableFailure:
    trace_id: str
    step_id: str
    stage: VerificationStage
    status: VerificationStatus
    responsible_fields: Tuple[str, ...]
    structural_references: Tuple[str, ...]
    candidate_match_count: int
    required_edge_provenance: Tuple[Tuple[str, str, str], ...]
    failure_reasons: Tuple[str, ...]
    resource_complete: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "step_id": self.step_id,
            "stage": self.stage.value,
            "status": self.status.value,
            "responsible_fields": list(self.responsible_fields),
            "structural_references": list(self.structural_references),
            "candidate_match_count": self.candidate_match_count,
            "required_edge_provenance": [list(item) for item in self.required_edge_provenance],
            "failure_reasons": list(self.failure_reasons),
            "resource_complete": self.resource_complete,
        }


@dataclass(frozen=True)
class TypedExecutableReasoningTrace:
    trace_id: str
    semantic_assignment_id: str
    intent: IntentHypothesis
    role_steps: Tuple[RoleBindingStep, ...]
    selected_role_bindings: Tuple[Tuple[str, Any], ...]
    verification_steps: Tuple[TraceVerificationStep, ...]
    projected_answer: str = ""
    output_domain: str = ""
    executable: bool = False
    resource_complete: bool = True
    local_alternative_count: int = 0
    repair_eligible: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_version": TRACE_VERSION,
            "trace_id": self.trace_id,
            "semantic_assignment_id": self.semantic_assignment_id,
            "intent": self.intent.to_dict(),
            "role_steps": [item.to_dict() for item in self.role_steps],
            "selected_role_bindings": {
                role: _jsonable(value) for role, value in self.selected_role_bindings
            },
            "verification_steps": [item.to_dict() for item in self.verification_steps],
            "projected_answer": self.projected_answer,
            "output_domain": self.output_domain,
            "executable": self.executable,
            "resource_complete": self.resource_complete,
            "local_alternative_count": self.local_alternative_count,
            "repair_eligible": self.repair_eligible,
        }


@dataclass(frozen=True)
class IntentValidationResult:
    ok: bool
    parse_ok: bool
    intents: Tuple[IntentHypothesis, ...] = ()
    errors: Tuple[str, ...] = ()
    normalized_payload: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RoleBindingValidationResult:
    ok: bool
    parse_ok: bool
    role_steps: Tuple[RoleBindingStep, ...] = ()
    errors: Tuple[str, ...] = ()
    normalized_payload: Dict[str, Any] = field(default_factory=dict)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(child) for child in value]
    return value


def _parse_payload(raw: Any) -> Tuple[Optional[Mapping[str, Any]], Tuple[str, ...]]:
    if isinstance(raw, Mapping):
        return raw, ()
    try:
        value = json.loads(str(raw or ""))
    except (TypeError, json.JSONDecodeError):
        return None, ("invalid_json",)
    if not isinstance(value, Mapping):
        return None, ("root_not_object",)
    return value, ()


def _first_forbidden_key(value: Any) -> str:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in _FORBIDDEN_KEYS:
                return str(key)
            nested = _first_forbidden_key(child)
            if nested:
                return nested
    elif isinstance(value, list):
        for child in value:
            nested = _first_forbidden_key(child)
            if nested:
                return nested
    return ""


def _reference_ids(view: Mapping[str, Any]) -> Tuple[str, ...]:
    return tuple(sorted({
        str(item.get("node_id"))
        for item in view.get("schema_nodes", [])
        if isinstance(item, Mapping) and item.get("node_id")
    }))


def _allowed_signatures(view: Mapping[str, Any]) -> Tuple[OperationSignatureVariant, ...]:
    ontology = view.get("operation_ontology") or {}
    declared = set(str(item) for item in ontology.get("signature_ids") or OPERATION_SIGNATURES)
    return tuple(
        OPERATION_SIGNATURES[key]
        for key in sorted(OPERATION_SIGNATURES)
        if key in declared
    )


def _intent_schema(signature: OperationSignatureVariant) -> Dict[str, Any]:
    properties: Dict[str, Any] = {
        "intent_id": {"type": "string", "pattern": r"^I[0-9]+$"},
        "signature_id": {"type": "string", "const": signature.signature_id},
        "unresolved_semantics": {"type": "array", "items": {"type": "string"}},
    }
    required = list(properties)
    if signature.comparison_polarities:
        properties["comparison_polarity"] = {
            "type": "string",
            "enum": list(signature.comparison_polarities),
        }
        required.append("comparison_polarity")
    return {
        "type": "object",
        "properties": properties,
        "required": sorted(required),
        "additionalProperties": False,
    }


def build_intent_response_schema(view: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the exact constrained schema for public intent hypotheses."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "trace_version": {"type": "string", "const": TRACE_VERSION},
            "intent_hypotheses": {
                "type": "array",
                "items": {"anyOf": [_intent_schema(item) for item in _allowed_signatures(view)]},
                "minItems": 1,
            },
            "unresolved_semantics": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["intent_hypotheses", "trace_version", "unresolved_semantics"],
        "additionalProperties": False,
    }


def build_intent_prompt(view: Mapping[str, Any]) -> str:
    """Build the proposal-blind public request for semantic intent hypotheses."""
    public_request = {
        "task": "emit_finite_typed_intent_hypotheses",
        "rules": [
            "Use only canonical signature IDs declared in operation_ontology.",
            "Emit every semantically plausible signature without voting or scoring.",
            "Record unresolved semantic distinctions as strings; do not explain reasoning.",
            "Do not emit role bindings, answers, rationales, confidence, or scores.",
        ],
        "planner_view": dict(view),
    }
    return (
        "CERTA Typed Derivation Planner Agent: intent stage.\n"
        "Return only JSON conforming exactly to the supplied response schema.\n"
        f"PUBLIC_REQUEST={json.dumps(public_request, sort_keys=True, separators=(',', ':'))}"
    )


def validate_intent_output(raw: Any, view: Mapping[str, Any]) -> IntentValidationResult:
    payload, parse_errors = _parse_payload(raw)
    if payload is None:
        return IntentValidationResult(False, False, errors=parse_errors)
    errors = []
    forbidden = _first_forbidden_key(payload)
    if forbidden:
        errors.append(f"forbidden_key:{forbidden}")
    allowed_top = {"trace_version", "intent_hypotheses", "unresolved_semantics"}
    for key in payload:
        if key not in allowed_top:
            errors.append(f"unknown_top_level_field:{key}")
    if payload.get("trace_version") != TRACE_VERSION:
        errors.append(f"trace_version_mismatch:{payload.get('trace_version')}")
    rows = payload.get("intent_hypotheses")
    if not isinstance(rows, list) or not rows:
        errors.append("intent_hypotheses_not_nonempty_list")
        rows = []
    allowed = {item.signature_id: item for item in _allowed_signatures(view)}
    seen = set()
    intents = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            errors.append(f"intent_not_object:{index}")
            continue
        intent_id = str(row.get("intent_id") or "")
        signature_id = str(row.get("signature_id") or "")
        signature = allowed.get(signature_id)
        if not _INTENT_ID_RE.fullmatch(intent_id):
            errors.append(f"invalid_intent_id:{intent_id}")
        elif intent_id in seen:
            errors.append(f"duplicate_intent_id:{intent_id}")
        seen.add(intent_id)
        if signature is None:
            errors.append(f"unknown_or_undeclared_signature:{signature_id}")
            continue
        allowed_fields = {"intent_id", "signature_id", "unresolved_semantics"}
        polarity = str(row.get("comparison_polarity") or "")
        if signature.comparison_polarities:
            allowed_fields.add("comparison_polarity")
            if polarity not in signature.comparison_polarities:
                errors.append(f"invalid_comparison_polarity:{intent_id}:{polarity}")
        elif "comparison_polarity" in row:
            errors.append(f"forbidden_comparison_polarity:{intent_id}")
        for key in row:
            if key not in allowed_fields:
                errors.append(f"unknown_intent_field:{intent_id}:{key}")
        unresolved = row.get("unresolved_semantics")
        if not isinstance(unresolved, list) or any(not isinstance(item, str) for item in unresolved):
            errors.append(f"invalid_unresolved_semantics:{intent_id}")
            unresolved = []
        intents.append(IntentHypothesis(
            intent_id=intent_id,
            signature_id=signature_id,
            comparison_polarity=polarity,
            unresolved_semantics=tuple(str(item) for item in unresolved),
        ))
    intents = sorted(intents, key=lambda item: item.intent_id)
    normalized = {
        "trace_version": TRACE_VERSION,
        "intent_hypotheses": [item.to_dict() for item in intents],
        "unresolved_semantics": list(payload.get("unresolved_semantics") or []),
    }
    return IntentValidationResult(
        ok=not errors,
        parse_ok=True,
        intents=tuple(intents),
        errors=tuple(errors),
        normalized_payload=normalized,
    )


def _conjunction_schema(reference_ids: Sequence[str], role: Any) -> Dict[str, Any]:
    schema: Dict[str, Any] = {
        "type": "array",
        "items": {"type": "string", "enum": list(reference_ids)},
        "minItems": role.min_items,
    }
    if role.max_items is not None:
        schema["maxItems"] = role.max_items
    return schema


def _binding_schema(reference_ids: Sequence[str], role: Any) -> Dict[str, Any]:
    conjunction = _conjunction_schema(reference_ids, role)
    if role.shape == "structural_conjunction":
        return conjunction
    if role.shape == "finite_scope_members":
        schema: Dict[str, Any] = {
            "type": "array",
            "items": _conjunction_schema(reference_ids, role),
            "minItems": role.min_items,
        }
        if role.max_items is not None:
            schema["maxItems"] = role.max_items
        return schema
    raise ValueError(f"unsupported_role_shape:{role.name}:{role.shape}")


def build_role_binding_response_schema(
    view: Mapping[str, Any],
    intents: Sequence[IntentHypothesis],
) -> Dict[str, Any]:
    """Return role-step schema variants fixed to accepted intent hypotheses."""
    references = _reference_ids(view)
    variants = []
    for intent in intents:
        signature = get_operation_signature(intent.signature_id)
        if signature is None:
            continue
        for role in signature.allowed_roles:
            variants.append({
                "type": "object",
                "properties": {
                    "step_id": {"type": "string", "pattern": r"^R[0-9]+$"},
                    "intent_id": {"type": "string", "const": intent.intent_id},
                    "role_name": {"type": "string", "const": role.name},
                    "role_shape": {"type": "string", "const": role.shape},
                    "binding_options": {
                        "type": "array",
                        "items": _binding_schema(references, role),
                        "minItems": 1,
                    },
                    "unresolved_semantics": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "binding_options",
                    "intent_id",
                    "role_name",
                    "role_shape",
                    "step_id",
                    "unresolved_semantics",
                ],
                "additionalProperties": False,
            })
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "trace_version": {"type": "string", "const": TRACE_VERSION},
            "role_steps": {
                "type": "array",
                "items": {"anyOf": variants},
                "minItems": 1,
            },
            "unresolved_semantics": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["role_steps", "trace_version", "unresolved_semantics"],
        "additionalProperties": False,
    }


def build_role_binding_prompt(
    view: Mapping[str, Any],
    intents: Sequence[IntentHypothesis],
) -> str:
    """Build the public request for finite role-binding domains."""
    public_request = {
        "task": "emit_finite_typed_role_binding_domains",
        "rules": [
            "Use only exact schema node IDs present in planner_view.",
            "Emit one role step for every required canonical role.",
            "Preserve each declared operational role shape exactly.",
            "Expose complete finite binding options without ranking, voting, or scoring.",
            "Do not emit answers, rationales, confidence, or scores.",
        ],
        "accepted_intent_hypotheses": [item.to_dict() for item in intents],
        "planner_view": dict(view),
    }
    return (
        "CERTA Typed Derivation Planner Agent: role-binding stage.\n"
        "Return only JSON conforming exactly to the supplied response schema.\n"
        f"PUBLIC_REQUEST={json.dumps(public_request, sort_keys=True, separators=(',', ':'))}"
    )


def _flatten_references(value: Any) -> Tuple[str, ...]:
    if isinstance(value, (tuple, list)):
        return tuple(
            reference
            for item in value
            for reference in _flatten_references(item)
        )
    return (str(value),) if str(value) else ()


def validate_role_binding_output(
    raw: Any,
    view: Mapping[str, Any],
    intents: Sequence[IntentHypothesis],
) -> RoleBindingValidationResult:
    payload, parse_errors = _parse_payload(raw)
    if payload is None:
        return RoleBindingValidationResult(False, False, errors=parse_errors)
    errors = []
    forbidden = _first_forbidden_key(payload)
    if forbidden:
        errors.append(f"forbidden_key:{forbidden}")
    allowed_top = {"trace_version", "role_steps", "unresolved_semantics"}
    for key in payload:
        if key not in allowed_top:
            errors.append(f"unknown_top_level_field:{key}")
    if payload.get("trace_version") != TRACE_VERSION:
        errors.append(f"trace_version_mismatch:{payload.get('trace_version')}")
    rows = payload.get("role_steps")
    if not isinstance(rows, list) or not rows:
        errors.append("role_steps_not_nonempty_list")
        rows = []
    intent_by_id = {item.intent_id: item for item in intents}
    seen_step_ids = set()
    seen_roles = set()
    raw_by_intent: Dict[str, Dict[str, Mapping[str, Any]]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            errors.append(f"role_step_not_object:{index}")
            continue
        step_id = str(row.get("step_id") or "")
        intent_id = str(row.get("intent_id") or "")
        role_name = str(row.get("role_name") or "")
        intent = intent_by_id.get(intent_id)
        if not _ROLE_STEP_ID_RE.fullmatch(step_id):
            errors.append(f"invalid_role_step_id:{step_id}")
        elif step_id in seen_step_ids:
            errors.append(f"duplicate_role_step_id:{step_id}")
        seen_step_ids.add(step_id)
        if intent is None:
            errors.append(f"unknown_intent_reference:{intent_id}")
            continue
        signature = get_operation_signature(intent.signature_id)
        role = signature.role(role_name) if signature is not None else None
        if role is None:
            errors.append(f"forbidden_or_unknown_role:{intent_id}:{role_name}")
            continue
        role_key = (intent_id, role_name)
        if role_key in seen_roles:
            errors.append(f"duplicate_role_step:{intent_id}:{role_name}")
        seen_roles.add(role_key)
        if str(row.get("role_shape") or "") != role.shape:
            errors.append(
                f"role_shape_declaration_mismatch:{intent_id}:{role_name}:"
                f"{row.get('role_shape')}!={role.shape}"
            )
        allowed_fields = {
            "step_id", "intent_id", "role_name", "role_shape",
            "binding_options", "unresolved_semantics",
        }
        for key in row:
            if key not in allowed_fields:
                errors.append(f"unknown_role_step_field:{step_id}:{key}")
        options = row.get("binding_options")
        if not isinstance(options, list) or not options:
            errors.append(f"binding_options_not_nonempty:{intent_id}:{role_name}")
        unresolved = row.get("unresolved_semantics")
        if not isinstance(unresolved, list) or any(not isinstance(item, str) for item in unresolved):
            errors.append(f"invalid_unresolved_semantics:{step_id}")
        raw_by_intent.setdefault(intent_id, {})[role_name] = row

    references = _reference_ids(view)
    normalized_plans = []
    normalized_steps = []
    for plan_index, intent in enumerate(sorted(intents, key=lambda item: item.intent_id), start=1):
        signature = get_operation_signature(intent.signature_id)
        if signature is None:
            errors.append(f"unknown_signature:{intent.intent_id}:{intent.signature_id}")
            continue
        declared = raw_by_intent.get(intent.intent_id, {})
        for role_name in signature.required_role_names:
            if role_name not in declared:
                errors.append(f"missing_required_role:{intent.intent_id}:{role_name}")
        plan: Dict[str, Any] = {
            "plan_id": f"P{plan_index}",
            "signature_id": signature.signature_id,
            "operation_family": signature.operation_family,
            "semantic_result_role": signature.semantic_result_role,
            "answer_domain": signature.answer_domain,
            "projection_operator": signature.projection_operator,
            "role_domains": {
                role_name: row.get("binding_options")
                for role_name, row in sorted(declared.items())
            },
            "unresolved_semantics": sorted({
                *intent.unresolved_semantics,
                *[
                    str(item)
                    for row in declared.values()
                    for item in (row.get("unresolved_semantics") or [])
                ],
            }),
        }
        if intent.comparison_polarity:
            plan["comparison_polarity"] = intent.comparison_polarity
        validation = validate_operation_plan(plan, references)
        errors.extend(validation.errors)
        if not validation.ok:
            continue
        normalized_plan = dict(plan)
        normalized_plan["role_domains"] = {
            role: _jsonable(options)
            for role, options in validation.role_domains_dict().items()
        }
        normalized_plans.append(normalized_plan)
        for role_name, options in validation.role_domains_dict().items():
            row = declared[role_name]
            normalized_steps.append(RoleBindingStep(
                step_id=str(row.get("step_id") or ""),
                intent_id=intent.intent_id,
                role_name=role_name,
                role_shape=str(signature.role(role_name).shape),
                binding_options=tuple(options),
                unresolved_semantics=tuple(str(item) for item in row.get("unresolved_semantics") or []),
            ))
    normalized_steps.sort(key=lambda item: (item.intent_id, item.role_name, item.step_id))
    normalized_payload = {
        "planner_version": TRACE_VERSION,
        "plans": normalized_plans,
        "unresolved_semantics": list(payload.get("unresolved_semantics") or []),
    }
    return RoleBindingValidationResult(
        ok=not errors,
        parse_ok=True,
        role_steps=tuple(normalized_steps),
        errors=tuple(sorted(set(errors))),
        normalized_payload=normalized_payload,
    )


def build_validation_failure_records(
    raw: Any,
    errors: Sequence[str],
    boundary: str,
) -> Tuple[FirstVerifiableFailure, ...]:
    """Represent the earliest failed public generation boundary as an FVF."""
    if not errors:
        return ()
    payload, _ = _parse_payload(raw)
    payload = payload or {}
    reasons = tuple(str(item) for item in errors)
    boundary = str(boundary or "")
    step_id = ""
    responsible = ()
    references = ()
    if boundary == VerificationStage.INTENT_CONTRACT.value:
        stage = VerificationStage.INTENT_CONTRACT
        rows = payload.get("intent_hypotheses") or []
        rows = [item for item in rows if isinstance(item, Mapping)]
        candidates = []
        for reason in reasons:
            tokens = reason.split(":")
            prefix = tokens[0]
            target_index = None
            if prefix == "duplicate_intent_id" and len(tokens) > 1:
                matches = [
                    index for index, row in enumerate(rows)
                    if str(row.get("intent_id") or "") == tokens[1]
                ]
                target_index = matches[1] if len(matches) > 1 else None
            elif prefix == "unknown_or_undeclared_signature" and len(tokens) > 1:
                target_index = next((
                    index for index, row in enumerate(rows)
                    if str(row.get("signature_id") or "") == tokens[1]
                ), None)
            else:
                target_index = next((
                    index for index, row in enumerate(rows)
                    if str(row.get("intent_id") or "") in tokens
                ), None)
            if target_index is None:
                target_index = -1 if prefix in {
                    "forbidden_key", "unknown_top_level_field",
                    "trace_version_mismatch", "intent_hypotheses_not_nonempty_list",
                } else len(rows)
            candidates.append((target_index, reason))
        selected_index = min(index for index, _ in candidates)
        selected_reasons = tuple(
            reason for index, reason in candidates if index == selected_index
        )
        row = rows[selected_index] if 0 <= selected_index < len(rows) else {}
        raw_step_id = str(row.get("intent_id") or "INTENT_RESPONSE")
        occurrence = sum(
            1 for prior in rows[:selected_index + 1]
            if str(prior.get("intent_id") or "") == raw_step_id
        ) if row else 1
        step_id = f"{raw_step_id}@{occurrence}" if occurrence > 1 else raw_step_id
        responsible = ("signature_id",)
    elif boundary == "ROLE_BINDING":
        rows = payload.get("role_steps") or []
        rows = [item for item in rows if isinstance(item, Mapping)]
        candidates = []
        for reason in reasons:
            tokens = reason.split(":")
            prefix = tokens[0]
            candidate_stage = (
                VerificationStage.ROLE_REFERENCE
                if prefix == "unknown_schema_id"
                else VerificationStage.ROLE_SHAPE
            )
            target_index = None
            if prefix == "duplicate_role_step_id" and len(tokens) > 1:
                matches = [
                    index for index, row in enumerate(rows)
                    if str(row.get("step_id") or "") == tokens[1]
                ]
                target_index = matches[1] if len(matches) > 1 else None
            elif prefix == "duplicate_role_step" and len(tokens) > 2:
                matches = [
                    index for index, row in enumerate(rows)
                    if str(row.get("intent_id") or "") == tokens[1]
                    and str(row.get("role_name") or "") == tokens[2]
                ]
                target_index = matches[1] if len(matches) > 1 else None
            elif prefix == "unknown_schema_id" and len(tokens) > 2:
                target_index = next((
                    index for index, row in enumerate(rows)
                    if str(row.get("role_name") or "") == tokens[1]
                    and tokens[2] in _flatten_references(row.get("binding_options") or [])
                ), None)
            if target_index is None:
                target_index = next((
                    index for index, row in enumerate(rows)
                    if str(row.get("step_id") or "") in tokens
                    or str(row.get("role_name") or "") in tokens
                ), None)
            if target_index is None:
                target_index = -1 if prefix in {
                    "forbidden_key", "unknown_top_level_field",
                    "trace_version_mismatch", "role_steps_not_nonempty_list",
                } else len(rows)
            candidates.append((
                target_index,
                0 if candidate_stage == VerificationStage.ROLE_SHAPE else 1,
                reason,
                candidate_stage,
            ))
        selected_index, selected_stage_order, _, stage = min(candidates)
        selected_reasons = tuple(
            reason for row_index, stage_order, reason, _ in candidates
            if row_index == selected_index and stage_order == selected_stage_order
        )
        row = rows[selected_index] if 0 <= selected_index < len(rows) else {}
        selected_reason = selected_reasons[0]
        tokens = selected_reason.split(":")
        role_name = str(row.get("role_name") or next(
            (
                token for token in tokens
                if token in {
                    "TARGET_ENTITY", "TARGET_MEASURE", "TIME_SCOPE",
                    "AGGREGATION_SCOPE", "GROUP_SCOPE",
                    "LEFT_OPERAND", "RIGHT_OPERAND",
                }
            ),
            "role_binding",
        ))
        responsible = (role_name,)
        raw_step_id = str(row.get("step_id") or f"ROLE:{role_name}")
        occurrence = sum(
            1 for prior in rows[:selected_index + 1]
            if str(prior.get("step_id") or "") == raw_step_id
        ) if row else 1
        step_id = f"{raw_step_id}@{occurrence}" if occurrence > 1 else raw_step_id
        references = tuple(sorted(set(_flatten_references(row.get("binding_options") or []))))
    else:
        raise ValueError(f"unsupported_validation_boundary:{boundary}")
    identity = {
        "trace_version": TRACE_VERSION,
        "validation_boundary": boundary,
        "step_id": step_id,
        "payload": _jsonable(payload),
        "errors": list(selected_reasons),
    }
    trace_id = f"TRV-{canonical_json_hash(identity, 24)}"
    return (FirstVerifiableFailure(
        trace_id=trace_id,
        step_id=step_id,
        stage=stage,
        status=VerificationStatus.STRUCTURALLY_INVALID,
        responsible_fields=responsible,
        structural_references=references,
        candidate_match_count=0,
        required_edge_provenance=(),
        failure_reasons=selected_reasons,
        resource_complete=True,
    ),)


def _status_from_assignment(assignment: Any) -> VerificationStatus:
    outcome = str(getattr(getattr(assignment, "outcome", ""), "value", getattr(assignment, "outcome", "")))
    return {
        "UNIQUE_EXECUTABLE": VerificationStatus.PASS,
        "UNRESOLVED_BINDING": VerificationStatus.UNRESOLVED,
        "AMBIGUOUS_BINDING": VerificationStatus.AMBIGUOUS,
        "STRUCTURALLY_INVALID": VerificationStatus.STRUCTURALLY_INVALID,
        "RESOURCE_INCOMPLETE": VerificationStatus.RESOURCE_INCOMPLETE,
        "EXECUTION_FAILED": VerificationStatus.PASS,
    }.get(outcome, VerificationStatus.STRUCTURALLY_INVALID)


def _responsible_fields(assignment: Any) -> Tuple[str, ...]:
    operation = str(getattr(assignment, "operation_family", ""))
    reasons = tuple(str(item) for item in getattr(assignment, "failure_reasons", ()) or ())
    bindings = dict(getattr(assignment, "role_bindings", {}) or {})
    if operation in {"DIFF", "RATIO", "PAIR_COMPARE"}:
        for role in ("LEFT_OPERAND", "RIGHT_OPERAND"):
            if any(role.lower() in reason.lower() for reason in reasons):
                return (role,)
        return ("LEFT_OPERAND", "RIGHT_OPERAND")
    if operation in {"SUM", "AVERAGE", "COUNT", "ARGMAX", "ARGMIN"}:
        return tuple(
            role for role in (
                "TARGET_MEASURE", "GROUP_SCOPE", "TIME_SCOPE", "AGGREGATION_SCOPE",
            )
            if role in bindings
        )
    if operation == "LOOKUP":
        return tuple(
            role for role in ("TARGET_ENTITY", "TIME_SCOPE", "TARGET_MEASURE")
            if role in bindings
        )
    return ("signature_id",)


def _blocked_step(trace_prefix: str, stage: VerificationStage) -> TraceVerificationStep:
    return TraceVerificationStep(
        step_id=f"{trace_prefix}:{stage.value}",
        stage=stage,
        status=VerificationStatus.BLOCKED,
        failure_reasons=("blocked_by_earlier_verification_failure",),
    )


def _failure_resolution(assignment: Any, graph: Any) -> Tuple[int, Tuple[Tuple[str, str, str], ...]]:
    native_matches = tuple(getattr(assignment, "matched_cell_ids", ()) or ())
    native_edges = tuple(
        tuple(str(part) for part in edge)
        for edge in getattr(assignment, "required_edge_triples", ()) or ()
    )
    if graph is None:
        return len(native_matches), native_edges
    bindings = dict(getattr(assignment, "role_bindings", {}) or {})
    operation = str(getattr(assignment, "operation_family", ""))
    reasons = tuple(str(item) for item in getattr(assignment, "failure_reasons", ()) or ())
    conjunction: Tuple[str, ...] = ()
    if operation == "LOOKUP":
        conjunction = tuple(
            reference
            for role in ("TARGET_ENTITY", "TARGET_MEASURE", "TIME_SCOPE")
            for reference in _flatten_references(bindings.get(role, ()))
        )
    elif operation in {"DIFF", "RATIO", "PAIR_COMPARE"}:
        responsible = _responsible_fields(assignment)
        role = responsible[0] if len(responsible) == 1 else ""
        conjunction = _flatten_references(bindings.get(role, ())) if role else ()
    elif operation in {"SUM", "AVERAGE", "COUNT", "ARGMAX", "ARGMIN"}:
        member = next(
            (
                reason.split(":", 1)[1]
                for reason in reasons
                if reason.startswith(("ambiguous_scope_member:", "unresolved_scope_member:"))
            ),
            "",
        )
        member_refs = tuple(item for item in member.split(",") if item)
        shared = tuple(
            reference
            for role in ("TARGET_MEASURE", "GROUP_SCOPE", "TIME_SCOPE")
            for reference in _flatten_references(bindings.get(role, ()))
        )
        conjunction = (*member_refs, *shared) if member_refs else ()
    if not conjunction:
        return len(native_matches), native_edges
    resolution = resolve_atomic_operand(graph, conjunction)
    return len(resolution.candidate_node_ids), tuple(resolution.required_edge_triples)


def _verification_steps(
    trace_prefix: str,
    assignment: Any,
    graph: Any,
    intent: IntentHypothesis,
    role_steps: Sequence[RoleBindingStep],
) -> Tuple[TraceVerificationStep, ...]:
    bindings = dict(getattr(assignment, "role_bindings", {}) or {})
    all_references = tuple(sorted({
        reference
        for value in bindings.values()
        for reference in _flatten_references(value)
    }))
    candidate_match_count, edges = _failure_resolution(assignment, graph)
    reasons = tuple(str(item) for item in getattr(assignment, "failure_reasons", ()) or ())
    responsible = _responsible_fields(assignment)
    grounding_status = _status_from_assignment(assignment)
    steps = [TraceVerificationStep(
        step_id=intent.intent_id,
        stage=VerificationStage.INTENT_CONTRACT,
        status=VerificationStatus.PASS,
        responsible_fields=("signature_id",),
    )]
    for role_step in role_steps:
        role_references = tuple(sorted(set(_flatten_references(
            bindings.get(role_step.role_name, ())
        ))))
        steps.extend((
            TraceVerificationStep(
                step_id=role_step.step_id,
                stage=VerificationStage.ROLE_SHAPE,
                status=VerificationStatus.PASS,
                responsible_fields=(role_step.role_name,),
            ),
            TraceVerificationStep(
                step_id=role_step.step_id,
                stage=VerificationStage.ROLE_REFERENCE,
                status=VerificationStatus.PASS,
                responsible_fields=(role_step.role_name,),
                structural_references=role_references,
            ),
        ))
    operation = str(getattr(assignment, "operation_family", ""))
    grounding_references = (
        tuple(sorted(set(
            reference
            for role in responsible
            for reference in _flatten_references(bindings.get(role, ()))
        )))
        if operation in {"DIFF", "RATIO", "PAIR_COMPARE"}
        else all_references
    )
    grounding_step_id = (
        "G1"
        if responsible != ("RIGHT_OPERAND",)
        else "G2"
    )
    steps.append(TraceVerificationStep(
        step_id=grounding_step_id,
        stage=VerificationStage.STRUCTURAL_GROUNDING,
        status=grounding_status,
        responsible_fields=responsible,
        structural_references=grounding_references,
        candidate_match_count=candidate_match_count,
        required_edge_provenance=edges,
        failure_reasons=reasons if grounding_status != VerificationStatus.PASS else (),
        resource_complete=bool(getattr(assignment, "resource_complete", True)),
    ))
    if grounding_status != VerificationStatus.PASS:
        steps.extend(_blocked_step(trace_prefix, stage) for stage in (
            VerificationStage.EXECUTION,
            VerificationStage.PROJECTION,
            VerificationStage.OUTPUT_TYPE,
        ))
        return tuple(steps)
    execution_status = (
        VerificationStatus.PASS
        if str(getattr(assignment, "execution_outcome", "")) == "EXECUTED"
        else VerificationStatus.EXECUTION_FAILED
    )
    steps.append(TraceVerificationStep(
        step_id=f"{trace_prefix}:EXECUTION",
        stage=VerificationStage.EXECUTION,
        status=execution_status,
        failure_reasons=reasons if execution_status != VerificationStatus.PASS else (),
    ))
    if execution_status != VerificationStatus.PASS:
        steps.extend(_blocked_step(trace_prefix, stage) for stage in (
            VerificationStage.PROJECTION,
            VerificationStage.OUTPUT_TYPE,
        ))
        return tuple(steps)
    projection_outcome = str(getattr(assignment, "projection_outcome", ""))
    projection_status = (
        VerificationStatus.PASS
        if projection_outcome in {"PROJECTED", "TYPE_MISMATCH"}
        else VerificationStatus.PROJECTION_FAILED
    )
    steps.append(TraceVerificationStep(
        step_id=f"{trace_prefix}:PROJECTION",
        stage=VerificationStage.PROJECTION,
        status=projection_status,
        failure_reasons=reasons if projection_status != VerificationStatus.PASS else (),
    ))
    if projection_status != VerificationStatus.PASS:
        steps.append(_blocked_step(trace_prefix, VerificationStage.OUTPUT_TYPE))
        return tuple(steps)
    result = getattr(assignment, "projection_result", {}) or {}
    actual_domain = str(result.get("output_domain") or "")
    declared_domain = str(getattr(assignment, "answer_domain", ""))
    output_status = (
        VerificationStatus.PASS
        if actual_domain == declared_domain and projection_outcome != "TYPE_MISMATCH"
        else VerificationStatus.TYPE_MISMATCH
    )
    steps.append(TraceVerificationStep(
        step_id=f"{trace_prefix}:OUTPUT_TYPE",
        stage=VerificationStage.OUTPUT_TYPE,
        status=output_status,
        responsible_fields=("answer_domain", "projection_operator"),
        failure_reasons=reasons if output_status != VerificationStatus.PASS else (),
    ))
    return tuple(steps)


def first_verifiable_failure(
    trace: TypedExecutableReasoningTrace,
) -> Optional[FirstVerifiableFailure]:
    """Return the earliest mechanically non-PASS public verification step."""
    for step in trace.verification_steps:
        if step.status not in {VerificationStatus.PASS, VerificationStatus.BLOCKED}:
            return FirstVerifiableFailure(
                trace_id=trace.trace_id,
                step_id=step.step_id,
                stage=step.stage,
                status=step.status,
                responsible_fields=step.responsible_fields,
                structural_references=step.structural_references,
                candidate_match_count=step.candidate_match_count,
                required_edge_provenance=step.required_edge_provenance,
                failure_reasons=step.failure_reasons,
                resource_complete=step.resource_complete,
            )
    return None


def _local_alternative_count(
    assignment: Any,
    role_steps: Sequence[RoleBindingStep],
    responsible_fields: Sequence[str],
) -> int:
    selected = dict(getattr(assignment, "role_bindings", {}) or {})
    count = 0
    for step in role_steps:
        if step.role_name not in responsible_fields:
            continue
        selected_value = _jsonable(selected.get(step.role_name))
        count += sum(
            1 for option in step.binding_options
            if _jsonable(option) != selected_value
        )
    return count


def build_typed_executable_traces(
    intents: Sequence[IntentHypothesis],
    role_steps: Sequence[RoleBindingStep],
    closure: Any,
    graph: Any,
) -> Tuple[TypedExecutableReasoningTrace, ...]:
    """Convert deterministic closure assignments into canonically identified traces."""
    ordered_intents = tuple(sorted(intents, key=lambda item: item.intent_id))
    intent_by_plan = {
        f"P{index}": intent for index, intent in enumerate(ordered_intents, start=1)
    }
    steps_by_intent = {}
    for intent in ordered_intents:
        signature = get_operation_signature(intent.signature_id)
        role_order = {
            role.name: index
            for index, role in enumerate(signature.allowed_roles if signature else ())
        }
        if signature and signature.operation_family in {
            "SUM", "AVERAGE", "COUNT", "ARGMAX", "ARGMIN",
        }:
            role_order["AGGREGATION_SCOPE"] = len(role_order) + 1
        if signature and signature.operation_family == "LOOKUP":
            role_order.update({
                "TARGET_ENTITY": 0,
                "TIME_SCOPE": 1,
                "TARGET_MEASURE": 2,
            })
        steps_by_intent[intent.intent_id] = tuple(sorted(
            (step for step in role_steps if step.intent_id == intent.intent_id),
            key=lambda item: (role_order.get(item.role_name, 999), item.step_id),
        ))
    traces = []
    for assignment in getattr(closure, "assignments", ()) or ():
        plan_ids = tuple(getattr(assignment, "plan_ids", ()) or ())
        if not plan_ids:
            plan_ids = (str(getattr(assignment, "plan_id", "")),)
        for plan_id in plan_ids:
            intent = intent_by_plan.get(str(plan_id))
            if intent is None:
                continue
            intent_steps = steps_by_intent.get(intent.intent_id, ())
            selected = tuple(sorted(
                dict(getattr(assignment, "role_bindings", {}) or {}).items()
            ))
            identity = {
                "trace_version": TRACE_VERSION,
                "operation_contract_version": OPERATION_CONTRACT_VERSION,
                "intent": intent.to_dict(),
                "role_steps": [item.to_dict() for item in intent_steps],
                "selected_role_bindings": {
                    role: _jsonable(value) for role, value in selected
                },
                "assignment_key": str(getattr(assignment, "assignment_key", "")),
            }
            trace_id = f"TR-{canonical_json_hash(identity, 24)}"
            verification = _verification_steps(
                trace_id, assignment, graph, intent, intent_steps
            )
            fvf_step = next(
                (step for step in verification if step.status not in {VerificationStatus.PASS, VerificationStatus.BLOCKED}),
                None,
            )
            local_count = _local_alternative_count(
                assignment,
                intent_steps,
                fvf_step.responsible_fields if fvf_step is not None else (),
            )
            resource_complete = bool(getattr(assignment, "resource_complete", True))
            executable = all(step.status == VerificationStatus.PASS for step in verification)
            projection_result = getattr(assignment, "projection_result", {}) or {}
            patchable_fvf = bool(
                fvf_step is not None
                and fvf_step.stage == VerificationStage.STRUCTURAL_GROUNDING
                and fvf_step.status in {
                    VerificationStatus.UNRESOLVED,
                    VerificationStatus.AMBIGUOUS,
                }
            )
            traces.append(TypedExecutableReasoningTrace(
                trace_id=trace_id,
                semantic_assignment_id=str(
                    getattr(assignment, "assignment_id", "") or ""
                ),
                intent=intent,
                role_steps=intent_steps,
                selected_role_bindings=selected,
                verification_steps=verification,
                projected_answer=str(getattr(assignment, "projected_answer", "") or ""),
                output_domain=str(projection_result.get("output_domain") or ""),
                executable=executable,
                resource_complete=resource_complete,
                local_alternative_count=local_count,
                repair_eligible=bool(
                    patchable_fvf and resource_complete and local_count > 0
                ),
            ))
    return tuple(sorted(traces, key=lambda item: item.trace_id))
