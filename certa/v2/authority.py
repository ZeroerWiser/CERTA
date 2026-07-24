"""Exact V1 execution-artifact reconciliation for the V2 registry."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from certa.active_v1.answer_authority import active_answer_hash
from certa.active_v1.artifact_authority import (
    reconcile_registry_entry,
    validate_grounding_record_v3,
)
from certa.reproducibility.canonical_json import canonical_json, canonical_json_hash


def _exactly_one(
    rows: Sequence[Any], predicate: Any, reason: str
) -> Any:
    matches = [row for row in rows if predicate(row)]
    if len(matches) != 1:
        raise ValueError(f"{reason}:{len(matches)}")
    return matches[0]


def _executed_provenance(derivation: Any) -> list[str]:
    values = {
        str(value)
        for value in (
            *getattr(derivation, "evidence_ids", ()),
            *getattr(derivation, "operand_node_ids", ()),
        )
        if str(value)
    }
    for source, _, target in getattr(derivation, "required_edge_triples", ()):
        if str(source):
            values.add(str(source))
        if str(target):
            values.add(str(target))
    return sorted(values)


def materialize_executed_registry(
    *,
    sample_id: str,
    table_id: str,
    role_record_sha256: str,
    graph_sha256: str,
    raw_groundings: Sequence[Mapping[str, Any]],
    raw_derivations: Sequence[Mapping[str, Any]],
    registry_entries: Sequence[Mapping[str, Any]],
    answer_vault: Sequence[Mapping[str, Any]],
    executed_derivations: Sequence[Any],
    graph_artifact_refs: set[str],
) -> list[dict[str, Any]]:
    """Join each registry row to one grounding, execution, projection, and vault."""
    materialized = []
    seen_registry_refs: set[str] = set()
    for registry in sorted(
        registry_entries, key=lambda row: str(row.get("registry_entry_id") or "")
    ):
        registry_ref = str(registry.get("registry_entry_id") or "")
        if not registry_ref or registry_ref in seen_registry_refs:
            raise ValueError("authority_registry_ref_invalid")
        seen_registry_refs.add(registry_ref)
        derivation_id = str(registry.get("derivation_id") or "")
        raw = _exactly_one(
            raw_derivations,
            lambda row: row.get("sample_id") == sample_id
            and row.get("derivation_id") == derivation_id,
            "authority_raw_derivation_join",
        )
        if (
            raw.get("sample_id") != sample_id
            or registry.get("sample_id") != sample_id
        ):
            raise ValueError("authority_sample_mismatch")
        raw_refs = set(str(item) for item in raw.get("provenance_ids", ()))
        if (
            not raw_refs
            or not raw_refs.issubset(graph_artifact_refs)
            or not set(str(item) for item in raw.get("operand_node_ids", ())).issubset(
                graph_artifact_refs
            )
        ):
            raise ValueError("authority_provenance_outside_graph")
        reconcile_registry_entry(registry, raw)

        grounding = _exactly_one(
            raw_groundings,
            lambda row: row.get("sample_id") == sample_id
            and row.get("table_id") == table_id
            and row.get("arm") == raw.get("arm")
            and row.get("plan_id") == raw.get("plan_id"),
            "authority_grounding_join",
        )
        validate_grounding_record_v3(grounding)
        if grounding.get("role_record_sha256") != role_record_sha256:
            raise ValueError("authority_role_hash_mismatch")
        hypothesis = _exactly_one(
            list(grounding.get("grounding_hypotheses") or ()),
            lambda row: row.get("binding_id") == raw.get("binding_id")
            and row.get("derivation_id") == derivation_id
            and row.get("canonical_program_id")
            == raw.get("canonical_program_id"),
            "authority_grounding_hypothesis_join",
        )
        if (
            hypothesis.get("binding_id")
            not in grounding.get("authorized_binding_ids", ())
            or hypothesis.get("resolution_state") != "EXACT"
            or hypothesis.get("grounding_valid") is not True
            or hypothesis.get("failure_reasons")
        ):
            raise ValueError("authority_grounding_not_exact")

        executed = _exactly_one(
            executed_derivations,
            lambda row: getattr(row, "derivation_id", "") == derivation_id,
            "authority_executed_derivation_join",
        )
        try:
            program = json.loads(str(executed.executable_program))
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("authority_program_not_json") from exc
        if canonical_json(program) != executed.executable_program:
            raise ValueError("authority_program_not_canonical")
        program_id = f"CP-{canonical_json_hash(program, 24)}"
        if (
            program_id != raw.get("canonical_program_id")
            or executed.operation_metadata.get("canonical_program_id") != program_id
        ):
            raise ValueError("authority_program_identity_mismatch")
        if (
            getattr(executed, "availability", "") != "available"
            or getattr(executed, "provenance_complete", False) is not True
            or executed.operation_metadata.get("resource_complete") is False
        ):
            raise ValueError("authority_execution_incomplete")
        if _executed_provenance(executed) != raw.get("provenance_ids"):
            raise ValueError("authority_provenance_replay_mismatch")
        if (
            getattr(executed, "typed_signature", "") != raw.get("signature_id")
            or getattr(executed, "projection_operator", "")
            != raw.get("projection")
            or getattr(executed, "output_domain", "") != raw.get("answer_role")
        ):
            raise ValueError("authority_operation_projection_mismatch")
        answer_hash = active_answer_hash(executed.projected_answer)
        if answer_hash != raw.get("projected_answer_hash"):
            raise ValueError("authority_executed_answer_hash_mismatch")

        vault = _exactly_one(
            answer_vault,
            lambda row: row.get("sample_id") == sample_id
            and row.get("table_id") == table_id
            and row.get("arm") == raw.get("arm")
            and row.get("canonical_program_id") == program_id
            and row.get("derivation_id") == derivation_id
            and row.get("answer_hash") == answer_hash,
            "authority_vault_join",
        )
        if (
            active_answer_hash(vault.get("executed_answer")) != answer_hash
            or vault.get("executed_answer") != executed.projected_answer
        ):
            raise ValueError("authority_vault_answer_mismatch")
        materialized.append(
            {
                "sample_id": sample_id,
                "table_id": table_id,
                "registry_entry_id": registry_ref,
                "derivation_id": derivation_id,
                "canonical_program_id": program_id,
                "binding_id": str(raw["binding_id"]),
                "role_record_sha256": role_record_sha256,
                "graph_sha256": graph_sha256,
                "answer_hash": answer_hash,
                "executed_answer": vault["executed_answer"],
                "validator_approved": True,
                "materializer_approved": True,
                "resolution_state": "EXACT",
                "resource_complete": True,
                "operation_contract_valid": True,
                "execution_outcome": "EXECUTED",
                "projection_outcome": "PROJECTED",
                "provenance_ids": list(raw["provenance_ids"]),
                "signature_id": str(raw["signature_id"]),
            }
        )
    return materialized
