from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from tests.v2.test_bounded_proof_contracts import (
    ROLE_HASH,
    complete_alt_proof,
)


REPO = Path(__file__).resolve().parents[2]


def schema(name: str) -> dict:
    return json.loads(
        (REPO / "schemas" / "v2" / name).read_text(encoding="utf-8")
    )


def test_proof_schema_accepts_closed_record_and_rejects_extra_fields() -> None:
    _, _, proof = complete_alt_proof()
    jsonschema.validate(proof, schema("PROOF_RECORD.schema.json"))
    proof["confidence"] = 1.0
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(proof, schema("PROOF_RECORD.schema.json"))


def test_structural_response_schema_is_closed_and_has_exact_catalog_size() -> None:
    candidate, _, _ = complete_alt_proof()
    response = {
        "sample_id": candidate["sample_id"],
        "candidate_id": candidate["candidate_id"],
        "candidate_answer_hash": candidate["candidate_answer_hash"],
        "packet_sha256": "e" * 64,
        "responses": [
            {
                    "challenge_id": challenge_id,
                    "applicable": True,
                "support": "SUPPORTED",
                "contradiction": "NOT_FOUND",
                "artifact_refs": ["node-1"],
                "claim_codes": ["CITED_SUPPORT"],
            }
            for challenge_id in (
                "ROLE_OPERATION_SEMANTIC_FIT",
                "SCOPE_COMPLETENESS",
                "TEMPORAL_ALIGNMENT",
                "UNIT_SCALE_CONSISTENCY",
                "ORDER_AND_POLARITY",
                "TIE_AND_ENTITY_PROJECTION",
                "LOOKUP_ENTITY_MEASURE_IDENTITY",
            )
        ],
    }
    jsonschema.validate(
        response, schema("STRUCTURAL_CHALLENGE_RESPONSE.schema.json")
    )
    response["generated_answer"] = "55"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(
            response, schema("STRUCTURAL_CHALLENGE_RESPONSE.schema.json")
        )


def test_pairwise_verifier_schema_has_no_score_or_answer_generation_field() -> None:
    response = {
        "b0_candidate_id": "CAND-" + "a" * 64,
        "b0_candidate_hash": "b" * 64,
        "alternative_candidate_id": "CAND-" + "c" * 64,
        "alternative_candidate_hash": "d" * 64,
        "differing_node_id": "ROLE",
        "b0_node_state": "UNKNOWN",
        "alternative_node_state": "PASS",
        "artifact_refs": [ROLE_HASH],
        "decision": "ALTERNATIVE_DOMINATES",
    }
    jsonschema.validate(response, schema("PAIRWISE_VERIFIER.schema.json"))
    response["score"] = 0.99
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(response, schema("PAIRWISE_VERIFIER.schema.json"))
