"""Frozen Role V3 bridge to the existing Active V1 Planner authorities."""

from __future__ import annotations

import re
from typing import Any, Mapping, Optional, Tuple

from graph_builder import HCEG

from certa.active_v1 import planner_adapter
from certa.active_v1.planner_adapter import ARMS, PlannerViewBuild
from certa.active_v1.role_contract_v3 import (
    ROLE_V3_RECORD_VERSION,
    ROLE_V3_ROLE_IDS,
    ROLE_V3_SCHEMA_VERSION,
    derive_role_v3_record,
    role_v3_to_planner_query_contract,
)
from certa.planner.schema_view import build_proposal_blind_planner_view
from certa.reproducibility.canonical_json import canonical_json_hash


__all__ = (
    "build_v3_arm_view",
    "close_compiled_payload",
    "compile_active_planner_payload",
)


CONSTRUCTOR_CAPABILITY_VERSION = "certa_active_v1_constructor_capability_v1"
ROLE_V3_REGISTRY_FILE_SHA256 = (
    "114065916322ce70d1ca8122e6a40f7866ee4dfaa7d9c93eba58ab741d0bf3be"
)
ROLE_V3_OUTPUT_SCHEMA_CANONICAL_SHA256 = (
    "e2070502c3948cf43827a17b96d5ac39885bff774b3ac72a3823b3237671d971"
)
ROLE_V3_REGISTRY_CANONICAL_SHA256 = (
    "d23a67a56506e96eb3f749209f6793a32f17250d0f020c5487135945f654f4a8"
)
CONSTRUCTOR_CAPABILITY_FIELDS = (
    "role_registry_present",
    "v3_bridge_fixture_pass",
    "planner_schema_fixture_pass",
    "active_compiler_fixture_pass",
    "grounding_fixture_pass",
    "closure_fixture_pass",
    "deterministic_executor_fixture_pass",
    "projection_fixture_pass",
    "provenance_fixture_pass",
    "registry_serialization_fixture_pass",
    "negative_fixture_pass",
)
_CONSTRUCTOR_ROW_FIELDS = frozenset((
    "role_id", *CONSTRUCTOR_CAPABILITY_FIELDS, "constructor_active",
    "failure_reasons", "fixture_artifact_sha256",
))
_SUPPORTED_ROLE_IDS = frozenset(ROLE_V3_ROLE_IDS) - {"UNSUPPORTED"}
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _constructor_active_role_ids(matrix: Mapping[str, Any]) -> Tuple[str, ...]:
    """Validate the frozen Pack matrix and return its exact active role IDs."""
    if not isinstance(matrix, Mapping) or set(matrix) != {
        "schema_version", "role_registry_sha256", "rows",
    }:
        raise ValueError("constructor_capability_matrix_field_set_mismatch")
    if matrix.get("schema_version") != CONSTRUCTOR_CAPABILITY_VERSION:
        raise ValueError("constructor_capability_matrix_version_mismatch")
    if matrix.get("role_registry_sha256") != ROLE_V3_REGISTRY_FILE_SHA256:
        raise ValueError("constructor_capability_role_registry_sha256_mismatch")
    rows = matrix.get("rows")
    if not isinstance(rows, list):
        raise ValueError("constructor_capability_rows_not_list")

    seen = set()
    active = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != _CONSTRUCTOR_ROW_FIELDS:
            raise ValueError("constructor_capability_row_field_set_mismatch")
        role_id = row.get("role_id")
        if role_id not in _SUPPORTED_ROLE_IDS:
            raise ValueError(f"constructor_capability_unknown_role:{role_id or ''}")
        if role_id in seen:
            raise ValueError(f"constructor_capability_duplicate_role:{role_id}")
        seen.add(role_id)
        if any(type(row.get(name)) is not bool for name in CONSTRUCTOR_CAPABILITY_FIELDS):
            raise ValueError(f"constructor_capability_boolean_missing:{role_id}")
        equation = all(row[name] for name in CONSTRUCTOR_CAPABILITY_FIELDS)
        if row.get("constructor_active") is not equation:
            raise ValueError(f"constructor_capability_equation_mismatch:{role_id}")
        reasons = row.get("failure_reasons")
        if not isinstance(reasons, list) or any(not isinstance(item, str) for item in reasons):
            raise ValueError(f"constructor_capability_failure_reasons_invalid:{role_id}")
        if bool(reasons) is equation:
            raise ValueError(f"constructor_capability_failure_reasons_mismatch:{role_id}")
        fixture_sha = row.get("fixture_artifact_sha256")
        if not isinstance(fixture_sha, str) or _SHA256_PATTERN.fullmatch(fixture_sha) is None:
            raise ValueError(f"constructor_capability_fixture_sha256_invalid:{role_id}")
        if equation:
            active.append(role_id)

    if not active:
        raise ValueError("capability_matrix_has_no_active_signature")
    if seen != _SUPPORTED_ROLE_IDS:
        missing = ",".join(sorted(_SUPPORTED_ROLE_IDS - seen))
        raise ValueError(f"constructor_capability_role_set_mismatch:{missing}")
    return tuple(sorted(active))


def _legacy_capability_projection(matrix: Mapping[str, Any]) -> Mapping[str, Any]:
    """Project the eleven-field Pack equation onto the frozen six-field API."""
    _constructor_active_role_ids(matrix)
    projected_rows = []
    for row in matrix["rows"]:
        constructor_active = bool(row["constructor_active"])
        projected_rows.append({
            "signature_id": row["role_id"],
            "registry_present": all(row[name] for name in (
                "role_registry_present", "v3_bridge_fixture_pass",
                "planner_schema_fixture_pass", "grounding_fixture_pass",
            )),
            "active_compiler_fixture_pass": row["active_compiler_fixture_pass"],
            "closure_fixture_pass": row["closure_fixture_pass"],
            "deterministic_executor_fixture_pass": row[
                "deterministic_executor_fixture_pass"
            ],
            "projection_fixture_pass": (
                row["projection_fixture_pass"] and row["provenance_fixture_pass"]
            ),
            "serialization_roundtrip_fixture_pass": (
                row["registry_serialization_fixture_pass"]
                and row["negative_fixture_pass"]
            ),
            "constructor_active": constructor_active,
            "constructor_failure_reasons": list(row["failure_reasons"]),
            "active": constructor_active,
        })
    projected = {
        "schema_version": "certa_active_v1_signature_capability_v1",
        "rows": projected_rows,
    }
    planner_adapter.active_signature_ids(projected)
    return projected


def _validated_v3_role(
    role: Any,
    active_ids: Tuple[str, ...],
    output_schema: Optional[Mapping[str, Any]],
    canonical_registry: Optional[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if not isinstance(role, Mapping):
        raise ValueError("role_v3_canonical_record_required")
    if not isinstance(output_schema, Mapping) or not isinstance(
        canonical_registry, Mapping,
    ):
        raise ValueError("role_v3_frozen_artifacts_required")
    if canonical_json_hash(output_schema) != ROLE_V3_OUTPUT_SCHEMA_CANONICAL_SHA256:
        raise ValueError("role_v3_output_schema_identity_mismatch")
    if canonical_json_hash(canonical_registry) != ROLE_V3_REGISTRY_CANONICAL_SHA256:
        raise ValueError("role_v3_canonical_registry_identity_mismatch")
    if role.get("schema_version") != ROLE_V3_RECORD_VERSION:
        raise ValueError("role_v3_canonical_record_version_mismatch")
    role_id = role.get("role_id")
    if not isinstance(role_id, str):
        raise ValueError("role_v3_role_id_not_string")
    expected = derive_role_v3_record({
        "schema_version": ROLE_V3_SCHEMA_VERSION,
        "role_id": role_id,
    }, output_schema, canonical_registry)
    if dict(role) != expected or canonical_json_hash(role) != canonical_json_hash(expected):
        raise ValueError("role_v3_canonical_record_mismatch")
    if role.get("supported") is not True:
        raise ValueError("unsupported_role_has_no_active_planner_view")
    if role_id not in active_ids:
        raise ValueError(f"inactive_role_signature:{role_id or ''}")
    query_contract = role_v3_to_planner_query_contract(role)
    if query_contract.get("allowed_signature_ids") != [role_id]:
        raise ValueError("role_v3_query_contract_signature_mismatch")
    return role, query_contract


def build_v3_arm_view(
    arm: str,
    question: str,
    graph: HCEG,
    table_json: Mapping[str, Any],
    role: Optional[Mapping[str, Any]],
    retrieval: Optional[Mapping[str, Any]],
    capability_matrix: Mapping[str, Any],
    *,
    output_schema: Optional[Mapping[str, Any]] = None,
    canonical_registry: Optional[Mapping[str, Any]] = None,
) -> PlannerViewBuild:
    """Build a value-firewalled C0/C1/C2 view from a frozen V3 record."""
    if arm not in ARMS:
        raise ValueError(f"unknown_active_constructor_arm:{arm}")
    active_ids = _constructor_active_role_ids(capability_matrix)
    role_payload: Optional[Mapping[str, Any]] = None
    if arm == "C0_SCHEMA_ONLY":
        allowed_ids = active_ids
        query_contract = None
        semantics_mode = "audit_only"
    else:
        role_payload, query_contract = _validated_v3_role(
            role, active_ids, output_schema, canonical_registry,
        )
        allowed_ids = (str(role_payload["role_id"]),)
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
        complete_schema_domain = {str(item["node_id"]) for item in view["schema_nodes"]}
        outside = sorted(set(references) - complete_schema_domain)
        if outside:
            raise ValueError(f"retrieval_reference_outside_schema:{','.join(outside)}")
        selected = set(references)
        view["schema_nodes"] = [
            item for item in view["schema_nodes"]
            if str(item["node_id"]) in selected
        ]
        view["schema_edges"] = [
            item for item in view["schema_edges"]
            if str(item["source"]) in selected and str(item["target"]) in selected
        ]

    return PlannerViewBuild(arm, view, role_hash, references)


def compile_active_planner_payload(
    raw: Any,
    view: Mapping[str, Any],
    capability_matrix: Mapping[str, Any],
) -> planner_adapter.ActiveCompilationResult:
    """Delegate compilation after exact Pack-to-legacy capability projection."""
    return planner_adapter.compile_active_planner_payload(
        raw, view, _legacy_capability_projection(capability_matrix),
    )


def close_compiled_payload(
    compilation: planner_adapter.ActiveCompilationResult,
    graph: HCEG,
    capability_matrix: Mapping[str, Any],
    *,
    max_assignments: Optional[int] = None,
) -> Any:
    """Delegate closure after exact Pack-to-legacy capability projection."""
    return planner_adapter.close_compiled_payload(
        compilation,
        graph,
        _legacy_capability_projection(capability_matrix),
        max_assignments=max_assignments,
    )
