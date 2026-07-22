"""Validated closure-payload adapter for CERTA Active V1."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

from graph_builder import HCEG

from certa.active_v1.role_contract import RoleValidation, role_to_query_contract
from certa.grounding.plan_closure import PlanClosure, build_plan_closure
from certa.operations.contracts import OPERATION_SIGNATURES
from certa.planner.schema_view import build_proposal_blind_planner_view
from certa.planner.typed_planner import validate_typed_planner_output
from certa.reproducibility.canonical_json import canonical_json, canonical_json_hash


ARMS = ("C0_SCHEMA_ONLY", "C1_ROLE_ONLY", "C2_ROLE_RETRIEVAL")
CAPABILITY_FIELDS = (
    "registry_present",
    "active_compiler_fixture_pass",
    "closure_fixture_pass",
    "deterministic_executor_fixture_pass",
    "projection_fixture_pass",
    "serialization_roundtrip_fixture_pass",
)


@dataclass(frozen=True)
class PlannerViewBuild:
    arm: str
    view: Dict[str, Any]
    role_record_sha256: str = ""
    retrieval_reference_node_ids: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ActiveCompilationResult:
    ok: bool
    normalized_payload: Dict[str, Any] = field(default_factory=dict)
    canonical_payload: str = ""
    canonical_payload_sha256: str = ""
    allowed_signature_ids: Tuple[str, ...] = ()
    errors: Tuple[str, ...] = ()


def active_signature_ids(matrix: Mapping[str, Any]) -> Tuple[str, ...]:
    """Recompute and enforce the constructor-capability activation equation."""
    rows = matrix.get("rows")
    if not isinstance(rows, list):
        raise ValueError("capability_matrix_rows_not_list")
    seen = set()
    active = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("capability_matrix_row_not_object")
        signature_id = str(row.get("signature_id") or "")
        if signature_id not in OPERATION_SIGNATURES:
            raise ValueError(f"capability_unknown_signature:{signature_id}")
        if signature_id in seen:
            raise ValueError(f"capability_duplicate_signature:{signature_id}")
        seen.add(signature_id)
        if any(type(row.get(field_name)) is not bool for field_name in CAPABILITY_FIELDS):
            raise ValueError(f"capability_fixture_boolean_missing:{signature_id}")
        equation = all(bool(row[field_name]) for field_name in CAPABILITY_FIELDS)
        if row.get("constructor_active") is not equation or row.get("active") is not equation:
            raise ValueError(f"capability_activation_equation_mismatch:{signature_id}")
        if equation:
            active.append(signature_id)
    if not active:
        raise ValueError("capability_matrix_has_no_active_signature")
    return tuple(sorted(active))


def _validated_role(role: Any, active_ids: Tuple[str, ...]) -> Mapping[str, Any]:
    if not isinstance(role, RoleValidation):
        raise ValueError("role_validation_record_required")
    if not role.ok:
        raise ValueError("invalid_role_has_no_active_planner_view")
    payload = role.payload
    if payload.get("supported") is not True:
        raise ValueError("unsupported_role_has_no_active_planner_view")
    if payload.get("signature") not in active_ids:
        raise ValueError("inactive_role_signature")
    return payload


def build_arm_view(
    arm: str,
    question: str,
    graph: HCEG,
    table_json: Mapping[str, Any],
    role: Optional[RoleValidation],
    retrieval: Optional[Mapping[str, Any]],
    capability_matrix: Mapping[str, Any],
) -> PlannerViewBuild:
    """Build one proposal-blind, value-firewalled arm view."""
    if arm not in ARMS:
        raise ValueError(f"unknown_active_constructor_arm:{arm}")
    active_ids = active_signature_ids(capability_matrix)
    role_payload: Optional[Mapping[str, Any]] = None
    if arm == "C0_SCHEMA_ONLY":
        allowed_ids = active_ids
        query_contract = None
        semantics_mode = "audit_only"
    else:
        role_payload = _validated_role(role, active_ids)
        allowed_ids = (str(role_payload["signature"]),)
        query_contract = role_to_query_contract(role_payload)
        semantics_mode = "active"
    role_hash = canonical_json_hash(role_payload) if role_payload is not None else ""
    view = build_proposal_blind_planner_view(
        question=str(question or ""),
        graph=graph,
        table_json=table_json,
        query_contract=query_contract,
        include_table_values=False,
        legacy_query_semantics_mode=semantics_mode,
        allowed_signature_ids=allowed_ids,
    )
    references: Tuple[str, ...] = ()
    if arm == "C2_ROLE_RETRIEVAL":
        if not isinstance(retrieval, Mapping):
            raise ValueError("c2_retrieval_result_required")
        if retrieval.get("role_record_sha256") != role_hash:
            raise ValueError("c2_role_record_sha256_mismatch")
        values = retrieval.get("reference_node_ids")
        if not isinstance(values, list) or not values:
            raise ValueError("c2_retrieval_reference_ids_empty")
        references = tuple(dict.fromkeys(str(item) for item in values))
        complete = {str(item.get("node_id")) for item in view["schema_nodes"]}
        outside = sorted(set(references) - complete)
        if outside:
            raise ValueError(f"retrieval_reference_outside_schema:{','.join(outside)}")
        selected = set(references)
        view["schema_nodes"] = [item for item in view["schema_nodes"] if item["node_id"] in selected]
        view["schema_edges"] = [
            item for item in view["schema_edges"]
            if item["source"] in selected and item["target"] in selected
        ]
    return PlannerViewBuild(arm, view, role_hash, references)


def compile_active_planner_payload(
    raw: Any,
    view: Mapping[str, Any],
    capability_matrix: Mapping[str, Any],
) -> ActiveCompilationResult:
    """Validate, capability-check, and canonically round-trip a Planner payload."""
    active_ids = active_signature_ids(capability_matrix)
    ontology = view.get("operation_ontology") if isinstance(view, Mapping) else None
    raw_allowed = ontology.get("signature_ids") if isinstance(ontology, Mapping) else None
    if not isinstance(raw_allowed, list) or not raw_allowed:
        return ActiveCompilationResult(False, errors=("planner_view_signature_allowlist_missing",))
    allowed_ids = tuple(str(item) for item in raw_allowed)
    if len(set(allowed_ids)) != len(allowed_ids):
        return ActiveCompilationResult(False, errors=("planner_view_signature_allowlist_duplicate",))
    inactive_view_ids = sorted(set(allowed_ids) - set(active_ids))
    if inactive_view_ids:
        return ActiveCompilationResult(
            False,
            errors=(f"planner_view_inactive_signature:{','.join(inactive_view_ids)}",),
        )
    validation = validate_typed_planner_output(raw, view, require_signature_id=True)
    errors = list(validation.errors)
    errors.extend(
        f"plan_rejected:{item.get('plan_id')}:{'|'.join(item.get('reasons') or [])}"
        for item in validation.plan_rejections
    )
    if not validation.ok or errors:
        return ActiveCompilationResult(False, errors=tuple(sorted(set(errors or ["planner_validation_failed"]))))
    payload = validation.normalized_payload
    for plan in payload.get("plans") or []:
        signature_id = str(plan.get("signature_id") or "")
        if signature_id not in active_ids:
            errors.append(f"inactive_plan_signature:{signature_id}")
    if errors:
        return ActiveCompilationResult(False, errors=tuple(sorted(set(errors))))
    serialized = canonical_json(payload)
    if canonical_json(json.loads(serialized)) != serialized:
        return ActiveCompilationResult(False, errors=("canonical_serialization_roundtrip_failed",))
    return ActiveCompilationResult(
        True,
        normalized_payload=payload,
        canonical_payload=serialized,
        canonical_payload_sha256=canonical_json_hash(payload),
        allowed_signature_ids=allowed_ids,
    )


def close_compiled_payload(
    compilation: ActiveCompilationResult,
    graph: HCEG,
    capability_matrix: Mapping[str, Any],
    *,
    max_assignments: Optional[int] = None,
) -> PlanClosure:
    """Close and execute only a valid payload against the same frozen allowlist."""
    if not compilation.ok:
        raise ValueError("cannot_close_invalid_active_compilation")
    active_ids = active_signature_ids(capability_matrix)
    allowed_ids = compilation.allowed_signature_ids
    if not allowed_ids:
        raise ValueError("active_compilation_allowlist_missing")
    inactive_allowed_ids = sorted(set(allowed_ids) - set(active_ids))
    if inactive_allowed_ids:
        raise ValueError(f"active_compilation_inactive_allowlist:{','.join(inactive_allowed_ids)}")
    payload_signature_ids = {
        str(plan.get("signature_id") or "")
        for plan in compilation.normalized_payload.get("plans") or []
        if isinstance(plan, Mapping)
    }
    outside_allowlist = sorted(payload_signature_ids - set(allowed_ids))
    if outside_allowlist:
        raise ValueError(f"active_compilation_signature_outside_allowlist:{','.join(outside_allowlist)}")
    closure = build_plan_closure(
        compilation.normalized_payload,
        graph,
        max_assignments=max_assignments,
        allowed_signature_ids=allowed_ids,
    )
    if not closure.resource_complete:
        raise ValueError("active_closure_resource_incomplete")
    for derivation in closure.executable_derivations:
        signature_id = str(derivation.typed_signature or "")
        signature = OPERATION_SIGNATURES.get(signature_id)
        if signature_id not in allowed_ids or signature is None:
            raise ValueError(f"active_closure_inactive_signature:{signature_id}")
        if derivation.projection_operator != signature.projection_operator or derivation.output_domain != signature.answer_domain:
            raise ValueError(f"active_closure_projection_contract_mismatch:{signature_id}")
    return closure
