"""Exact answer-class candidate universe over executed registry authority."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from certa.active_v1.answer_authority import active_answer_hash
from certa.reproducibility.canonical_json import canonical_json_hash


_REGISTRY_REQUIRED = frozenset(
    {
        "sample_id",
        "table_id",
        "registry_entry_id",
        "derivation_id",
        "canonical_program_id",
        "binding_id",
        "role_record_sha256",
        "graph_sha256",
        "answer_hash",
        "executed_answer",
        "validator_approved",
        "materializer_approved",
        "resolution_state",
        "resource_complete",
        "operation_contract_valid",
        "execution_outcome",
        "projection_outcome",
        "provenance_ids",
    }
)


def validate_registry_entry(entry: Mapping[str, Any], sample_id: str) -> None:
    missing = sorted(_REGISTRY_REQUIRED - set(entry))
    if missing:
        raise ValueError(f"registry_fields_missing:{','.join(missing)}")
    if entry.get("sample_id") != sample_id:
        raise ValueError("registry_sample_mismatch")
    for field in (
        "table_id",
        "registry_entry_id",
        "derivation_id",
        "canonical_program_id",
        "binding_id",
    ):
        if not isinstance(entry.get(field), str) or not entry[field]:
            raise ValueError(f"registry_{field}_invalid")
    for field in ("role_record_sha256", "graph_sha256", "answer_hash"):
        value = entry.get(field)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"registry_{field}_invalid")
    if active_answer_hash(entry.get("executed_answer")) != entry.get("answer_hash"):
        raise ValueError("registry_answer_hash_mismatch")
    if entry.get("validator_approved") is not True:
        raise ValueError("registry_validator_not_approved")
    if entry.get("materializer_approved") is not True:
        raise ValueError("registry_materializer_not_approved")
    if entry.get("resolution_state") != "EXACT":
        raise ValueError("registry_binding_not_exact")
    if entry.get("resource_complete") is not True:
        raise ValueError("registry_resource_incomplete")
    if entry.get("operation_contract_valid") is not True:
        raise ValueError("registry_operation_invalid")
    if entry.get("execution_outcome") != "EXECUTED":
        raise ValueError("registry_execution_invalid")
    if entry.get("projection_outcome") != "PROJECTED":
        raise ValueError("registry_projection_invalid")
    provenance = entry.get("provenance_ids")
    if (
        not isinstance(provenance, list)
        or not provenance
        or provenance != sorted(set(provenance))
        or any(not isinstance(item, str) or not item for item in provenance)
    ):
        raise ValueError("registry_provenance_invalid")


def _candidate(
    sample_id: str,
    answer: Any,
    answer_hash: str,
    source: str,
    registry_refs: list[str],
) -> dict[str, Any]:
    identity = {
        "sample_id": sample_id,
        "candidate_answer_hash": answer_hash,
        "candidate_source": source,
    }
    return {
        "sample_id": sample_id,
        "candidate_id": f"CAND-{canonical_json_hash(identity)}",
        "candidate_answer": answer,
        "candidate_answer_hash": answer_hash,
        "candidate_source": source,
        "registry_refs": registry_refs,
    }


def build_candidate_universe(
    sample_id: str,
    b0_answer: Any,
    executed_registry: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return B0 plus one candidate per distinct executed answer hash."""
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("candidate_sample_id_invalid")
    b0_hash = active_answer_hash(b0_answer)
    by_hash: dict[str, list[Mapping[str, Any]]] = {}
    seen_refs: set[str] = set()
    table_ids: set[str] = set()
    for entry in executed_registry:
        validate_registry_entry(entry, sample_id)
        ref = str(entry["registry_entry_id"])
        if ref in seen_refs:
            raise ValueError(f"registry_entry_id_duplicate:{ref}")
        seen_refs.add(ref)
        table_ids.add(str(entry["table_id"]))
        by_hash.setdefault(str(entry["answer_hash"]), []).append(entry)
    if len(table_ids) > 1:
        raise ValueError("registry_table_mismatch")

    b0_refs = sorted(
        str(entry["registry_entry_id"]) for entry in by_hash.pop(b0_hash, [])
    )
    candidates = [_candidate(sample_id, b0_answer, b0_hash, "B0", b0_refs)]
    for answer_hash in sorted(by_hash):
        entries = sorted(by_hash[answer_hash], key=lambda item: item["registry_entry_id"])
        answer = entries[0]["executed_answer"]
        if any(active_answer_hash(entry["executed_answer"]) != answer_hash for entry in entries):
            raise ValueError("registry_answer_class_inconsistent")
        candidates.append(
            _candidate(
                sample_id,
                answer,
                answer_hash,
                "EXECUTED_REGISTRY",
                [str(entry["registry_entry_id"]) for entry in entries],
            )
        )
    return candidates
