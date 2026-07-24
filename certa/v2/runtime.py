"""Candidate-conditioned structural challenge prompt and response boundary."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from certa.reproducibility.canonical_json import canonical_json_hash


CHALLENGE_IDS = (
    "ROLE_OPERATION_SEMANTIC_FIT",
    "SCOPE_COMPLETENESS",
    "TEMPORAL_ALIGNMENT",
    "UNIT_SCALE_CONSISTENCY",
    "ORDER_AND_POLARITY",
    "TIE_AND_ENTITY_PROJECTION",
    "LOOKUP_ENTITY_MEASURE_IDENTITY",
)
_CHALLENGE_ORDER = {challenge_id: index for index, challenge_id in enumerate(CHALLENGE_IDS)}
_SUPPORT = frozenset(("SUPPORTED", "NOT_SUPPORTED", "UNKNOWN"))
_CONTRADICTION = frozenset(("FOUND", "NOT_FOUND", "UNKNOWN"))
_CLAIM_CODES = frozenset(
    (
        "CITED_SUPPORT",
        "CITED_CONTRADICTION",
        "INSUFFICIENT_EVIDENCE",
        "NOT_APPLICABLE_BY_SIGNATURE",
    )
)
_REGISTRY_PROMPT_FIELDS = (
    "registry_entry_id",
    "canonical_program_id",
    "derivation_id",
    "binding_id",
    "answer_hash",
    "executed_answer",
    "provenance_ids",
)


def challenge_applicability(signature_id: str) -> dict[str, bool]:
    operation = str(signature_id).split("_", 1)[0]
    return {
        "ROLE_OPERATION_SEMANTIC_FIT": True,
        "SCOPE_COMPLETENESS": operation in {
            "COUNT", "SUM", "AVERAGE", "RATIO", "ARGMAX", "ARGMIN"
        },
        "TEMPORAL_ALIGNMENT": operation != "LOOKUP",
        "UNIT_SCALE_CONSISTENCY": operation in {
            "SUM", "AVERAGE", "RATIO", "ARGMAX", "ARGMIN"
        },
        "ORDER_AND_POLARITY": operation in {"RATIO", "ARGMAX", "ARGMIN"},
        "TIE_AND_ENTITY_PROJECTION": operation in {"ARGMAX", "ARGMIN"},
        "LOOKUP_ENTITY_MEASURE_IDENTITY": operation == "LOOKUP",
    }


def project_structural_evidence(
    evidence: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Remove numeric value mirrors only when their source cell is present."""
    cell_coordinates = {
        (item.get("row"), item.get("col"))
        for item in evidence
        if item.get("node_type") == "cell"
    }
    return [
        dict(item)
        for item in evidence
        if item.get("node_type") != "value"
        or (item.get("row"), item.get("col")) not in cell_coordinates
    ]


def build_structural_challenge_prompt(
    candidate: Mapping[str, Any],
    *,
    sample_id: str,
    question: str,
    role_record_sha256: str,
    graph_sha256: str,
    evidence: Sequence[Mapping[str, Any]],
    registry_entries: Sequence[Mapping[str, Any]],
    signature_id: str = "",
) -> dict[str, Any]:
    """Build the symmetric closed packet; only the target tuple may vary."""
    if candidate.get("sample_id") != sample_id:
        raise ValueError("challenge_candidate_sample_mismatch")
    representatives: dict[str, Mapping[str, Any]] = {}
    for entry in sorted(
        registry_entries, key=lambda item: str(item.get("registry_entry_id") or "")
    ):
        representatives.setdefault(str(entry.get("answer_hash") or ""), entry)
    sanitized_registry = [
        {field: entry.get(field) for field in _REGISTRY_PROMPT_FIELDS}
        for entry in representatives.values()
    ]
    packet = {
        "schema_version": "certa_v2_structural_challenge_packet_v1",
        "sample_id": sample_id,
        "question": str(question),
        "role_record_sha256": role_record_sha256,
        "graph_sha256": graph_sha256,
        "candidate_id": candidate.get("candidate_id"),
        "candidate_answer_hash": candidate.get("candidate_answer_hash"),
        "candidate_answer": candidate.get("candidate_answer"),
        "challenge_ids": list(CHALLENGE_IDS),
        "challenge_applicability": challenge_applicability(signature_id),
        "evidence": [dict(item) for item in evidence],
        "executed_registry_evidence": sanitized_registry,
        "instructions": (
            "For the supplied candidate only, cite support and contradictions "
            "from the packet. Do not generate, rank, or select an answer."
        ),
    }
    return {**packet, "packet_sha256": canonical_json_hash(packet)}


def canonicalize_structural_response(
    response: Mapping[str, Any],
    *,
    expected_applicability: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    """Canonicalize lists and impose deterministic challenge applicability."""
    result = dict(response)
    rows = []
    for row in response.get("responses", ()):
        if not isinstance(row, Mapping):
            continue
        canonical = {
            **dict(row),
            "artifact_refs": sorted(set(row.get("artifact_refs", ()))),
            "claim_codes": sorted(set(row.get("claim_codes", ()))),
        }
        challenge_id = canonical.get("challenge_id")
        if expected_applicability is not None and challenge_id in expected_applicability:
            canonical["applicable"] = expected_applicability[challenge_id]
            if not canonical["applicable"]:
                canonical.update(
                    support="NOT_SUPPORTED",
                    contradiction="NOT_FOUND",
                    artifact_refs=[],
                    claim_codes=["NOT_APPLICABLE_BY_SIGNATURE"],
                )
        rows.append(canonical)
    rows.sort(key=lambda row: _CHALLENGE_ORDER.get(row.get("challenge_id"), len(CHALLENGE_IDS)))
    result["responses"] = rows
    return result


def validate_structural_challenge_response(
    response: Mapping[str, Any],
    *,
    candidate: Mapping[str, Any],
    packet_sha256: str,
    allowed_artifact_refs: set[str],
    expected_applicability: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    """Validate identities, full catalog coverage, and citation allowlists."""
    expected_fields = {
        "sample_id",
        "candidate_id",
        "candidate_answer_hash",
        "packet_sha256",
        "responses",
    }
    if set(response) != expected_fields:
        raise ValueError("challenge_response_fields_mismatch")
    identities = (
        response.get("sample_id") == candidate.get("sample_id")
        and response.get("candidate_id") == candidate.get("candidate_id")
        and response.get("candidate_answer_hash")
        == candidate.get("candidate_answer_hash")
        and response.get("packet_sha256") == packet_sha256
    )
    if not identities:
        raise ValueError("challenge_identity_mismatch")
    rows = response.get("responses")
    if not isinstance(rows, list) or [
        row.get("challenge_id") if isinstance(row, Mapping) else None for row in rows
    ] != list(CHALLENGE_IDS):
        raise ValueError("challenge_roster_mismatch")
    row_fields = {
        "challenge_id",
        "applicable",
        "support",
        "contradiction",
        "artifact_refs",
        "claim_codes",
    }
    for row in rows:
        if set(row) != row_fields:
            raise ValueError("challenge_row_fields_mismatch")
        if row.get("support") not in _SUPPORT:
            raise ValueError("challenge_support_invalid")
        if row.get("contradiction") not in _CONTRADICTION:
            raise ValueError("challenge_contradiction_invalid")
        if type(row.get("applicable")) is not bool:
            raise ValueError("challenge_applicability_invalid")
        if expected_applicability is not None and row["applicable"] is not bool(
            expected_applicability[row["challenge_id"]]
        ):
            raise ValueError("challenge_applicability_mismatch")
        refs = row.get("artifact_refs")
        codes = row.get("claim_codes")
        if (
            not isinstance(refs, list)
            or refs != sorted(set(refs))
            or not set(refs).issubset(allowed_artifact_refs)
        ):
            raise ValueError("challenge_artifact_ref_invalid")
        if (
            not isinstance(codes, list)
            or codes != sorted(set(codes))
            or not set(codes).issubset(_CLAIM_CODES)
        ):
            raise ValueError("challenge_claim_code_invalid")
        if row["support"] == "SUPPORTED" and (
            not refs or "CITED_SUPPORT" not in codes
        ):
            raise ValueError("challenge_positive_support_uncited")
        if row["contradiction"] == "FOUND" and (
            not refs or "CITED_CONTRADICTION" not in codes
        ):
            raise ValueError("challenge_contradiction_uncited")
        if not row["applicable"] and (
            row["support"] != "NOT_SUPPORTED"
            or row["contradiction"] != "NOT_FOUND"
            or refs
            or codes != ["NOT_APPLICABLE_BY_SIGNATURE"]
        ):
            raise ValueError("challenge_nonapplicable_response_invalid")
    return dict(response)
