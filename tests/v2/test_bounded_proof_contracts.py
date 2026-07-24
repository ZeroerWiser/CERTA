from __future__ import annotations

from copy import deepcopy

import pytest

from certa.active_v1.answer_authority import active_answer_hash
from certa.reproducibility.canonical_json import canonical_json_hash
from certa.v2.candidates import build_candidate_universe
from certa.v2.decision import (
    decide_proof_dominance,
    decide_proof_verifier,
)
from certa.v2.proof import (
    NODE_IDS,
    build_proof_record,
    validate_proof_record,
)
from certa.v2.runtime import (
    CHALLENGE_IDS,
    build_structural_challenge_prompt,
    canonicalize_structural_response,
    challenge_applicability,
    project_structural_evidence,
    validate_structural_challenge_response,
)
from certa.v2.search import SEARCH_SCHEDULE, merge_search_attempts


SAMPLE_ID = "sample-1"
TABLE_ID = "table-1"
ROLE_HASH = "a" * 64
GRAPH_HASH = "b" * 64


def registry_entry(
    *,
    answer: object = "55",
    registry_entry_id: str = "REG-1",
    program_id: str = "CP-" + "c" * 24,
    derivation_id: str = "DER-1",
    binding_id: str = "B-" + "d" * 24,
) -> dict:
    return {
        "sample_id": SAMPLE_ID,
        "table_id": TABLE_ID,
        "registry_entry_id": registry_entry_id,
        "derivation_id": derivation_id,
        "canonical_program_id": program_id,
        "binding_id": binding_id,
        "role_record_sha256": ROLE_HASH,
        "graph_sha256": GRAPH_HASH,
        "answer_hash": active_answer_hash(answer),
        "executed_answer": answer,
        "validator_approved": True,
        "materializer_approved": True,
        "resolution_state": "EXACT",
        "resource_complete": True,
        "operation_contract_valid": True,
        "execution_outcome": "EXECUTED",
        "projection_outcome": "PROJECTED",
        "provenance_ids": ["node-1"],
        "signature_id": "LOOKUP_VALUE_SCALAR",
    }


def candidate_and_witness(answer: object = "55") -> tuple[dict, dict]:
    entry = registry_entry(answer=answer)
    candidates = build_candidate_universe(SAMPLE_ID, "63%", [entry])
    candidate = next(
        item for item in candidates if item["candidate_source"] == "EXECUTED_REGISTRY"
    )
    return candidate, entry


def challenge_response(candidate: dict, *, support: str = "SUPPORTED") -> dict:
    return {
        "sample_id": SAMPLE_ID,
        "candidate_id": candidate["candidate_id"],
        "candidate_answer_hash": candidate["candidate_answer_hash"],
        "packet_sha256": "e" * 64,
        "responses": [
            {
                "challenge_id": challenge_id,
                "applicable": True,
                "support": support,
                "contradiction": "NOT_FOUND",
                "artifact_refs": ["node-1"],
                "claim_codes": ["CITED_SUPPORT"],
            }
            for challenge_id in CHALLENGE_IDS
        ],
    }


def complete_alt_proof(answer: object = "55") -> tuple[dict, dict, dict]:
    candidate, witness = candidate_and_witness(answer)
    response = challenge_response(candidate)
    proof = build_proof_record(
        candidate,
        role_record_sha256=ROLE_HASH,
        graph_sha256=GRAPH_HASH,
        witness=witness,
        structural_response=response,
        packet_sha256=response["packet_sha256"],
        allowed_artifact_refs={"node-1"},
    )
    return candidate, witness, proof


def b0_unknown_proof() -> tuple[dict, dict]:
    candidate = build_candidate_universe(SAMPLE_ID, "63%", [registry_entry()])[0]
    response = challenge_response(candidate, support="UNKNOWN")
    proof = build_proof_record(
        candidate,
        role_record_sha256=ROLE_HASH,
        graph_sha256=GRAPH_HASH,
        witness=None,
        structural_response=response,
        packet_sha256=response["packet_sha256"],
        allowed_artifact_refs={"node-1"},
    )
    return candidate, proof


def test_search_schedule_is_exactly_two_calls_for_each_frozen_view() -> None:
    assert len(SEARCH_SCHEDULE) == 6
    assert [(slot.view_kind, slot.call_index) for slot in SEARCH_SCHEDULE] == [
        ("ROLE_COMPLETE", 0),
        ("ROLE_COMPLETE", 1),
        ("RETRIEVAL_COMPLETE", 0),
        ("RETRIEVAL_COMPLETE", 1),
        ("VALUE_AWARE_PROPOSAL_BLIND", 0),
        ("VALUE_AWARE_PROPOSAL_BLIND", 1),
    ]
    assert len({slot.call_id for slot in SEARCH_SCHEDULE}) == 6


def test_search_requires_complete_registered_roster_and_deduplicates_plans() -> None:
    plan = {"plan_id": "P0", "signature_id": "LOOKUP_VALUE_SCALAR"}
    attempts = [
        {
            "call_id": slot.call_id,
            "view_kind": slot.view_kind,
            "call_index": slot.call_index,
            "status": "OK",
            "plans": [{**plan, "plan_id": f"P{index}"}],
        }
        for index, slot in enumerate(SEARCH_SCHEDULE)
    ]
    merged = merge_search_attempts(attempts)
    assert len(merged["plans"]) == 1
    assert merged["attempt_count"] == 6
    assert merged["lineage"][0]["source_call_ids"] == sorted(
        slot.call_id for slot in SEARCH_SCHEDULE
    )
    assert set(merged["plans"][0]) == {"plan_id", "signature_id"}
    with pytest.raises(ValueError, match="search_attempt_roster"):
        merge_search_attempts(attempts[:-1])
    with pytest.raises(ValueError, match="search_attempt_roster"):
        merge_search_attempts([*attempts[:-1], attempts[0]])


@pytest.mark.parametrize(
    "leak",
    [
        {"b0_answer": "63%"},
        {"candidate_answer": "55"},
        {"initial_proposal": "63%"},
        {"selected_final": "55"},
        {"correctness": True},
        {"nested": {"gold_answer": "55"}},
    ],
)
def test_search_rejects_proposal_or_outcome_leakage(leak: dict) -> None:
    attempts = [
        {
            "call_id": slot.call_id,
            "view_kind": slot.view_kind,
            "call_index": slot.call_index,
            "status": "EMPTY",
            "plans": [],
            "view": leak if slot.call_index == 0 else {"schema_nodes": []},
        }
        for slot in SEARCH_SCHEDULE
    ]
    with pytest.raises(ValueError, match="proposal_blind"):
        merge_search_attempts(attempts)


def test_candidate_universe_is_b0_plus_unique_exact_registry_answer_classes() -> None:
    entries = [
        registry_entry(answer="55", registry_entry_id="REG-2"),
        registry_entry(answer="55", registry_entry_id="REG-1"),
        registry_entry(answer="63%", registry_entry_id="REG-B0"),
    ]
    candidates = build_candidate_universe(SAMPLE_ID, "63%", entries)
    assert len(candidates) == 2
    assert candidates[0]["candidate_source"] == "B0"
    assert candidates[0]["registry_refs"] == ["REG-B0"]
    assert candidates[1]["candidate_source"] == "EXECUTED_REGISTRY"
    assert candidates[1]["registry_refs"] == ["REG-1", "REG-2"]


def test_candidate_universe_recomputes_hash_and_rejects_cross_sample_or_duplicate_refs() -> None:
    forged = registry_entry()
    forged["answer_hash"] = "f" * 64
    with pytest.raises(ValueError, match="registry_answer_hash"):
        build_candidate_universe(SAMPLE_ID, "63%", [forged])
    cross_sample = registry_entry()
    cross_sample["sample_id"] = "sample-2"
    with pytest.raises(ValueError, match="registry_sample"):
        build_candidate_universe(SAMPLE_ID, "63%", [cross_sample])
    with pytest.raises(ValueError, match="registry_entry_id_duplicate"):
        build_candidate_universe(
            SAMPLE_ID,
            "63%",
            [registry_entry(), registry_entry()],
        )


def test_candidate_ids_are_full_hash_bound_and_permutation_invariant() -> None:
    entries = [
        registry_entry(answer="55", registry_entry_id="REG-2"),
        registry_entry(answer="54", registry_entry_id="REG-1"),
    ]
    first = build_candidate_universe(SAMPLE_ID, "63%", entries)
    second = build_candidate_universe(SAMPLE_ID, "63%", list(reversed(entries)))
    assert first == second
    assert all(len(candidate["candidate_id"].removeprefix("CAND-")) == 64 for candidate in first)


def test_proof_has_exact_closed_seven_node_dag_and_revalidates() -> None:
    candidate, witness, proof = complete_alt_proof()
    assert set(proof) == {
        "sample_id",
        "candidate_id",
        "candidate_answer_hash",
        "candidate_source",
        "nodes",
        "overall_state",
        "registry_refs",
    }
    assert [node["node_id"] for node in proof["nodes"]] == list(NODE_IDS)
    assert proof["overall_state"] == "PASS"
    assert (
        validate_proof_record(
            proof,
            candidate=candidate,
            registry_entries=[witness],
            role_record_sha256=ROLE_HASH,
            graph_sha256=GRAPH_HASH,
            allowed_artifact_refs={"node-1"},
        )
        == proof
    )


def test_b0_without_registry_witness_is_unknown_never_fail() -> None:
    _, proof = b0_unknown_proof()
    states = {node["node_id"]: node["state"] for node in proof["nodes"]}
    assert states["ROLE"] == "UNKNOWN"
    assert states["CANDIDATE_PROOF_STATE"] == "UNKNOWN"
    assert proof["overall_state"] == "UNKNOWN"


def test_role_signature_contradiction_fails_role_and_blocks_dependents() -> None:
    candidate, witness = candidate_and_witness()
    response = challenge_response(candidate)
    proof = build_proof_record(
        candidate,
        role_record_sha256=ROLE_HASH,
        graph_sha256=GRAPH_HASH,
        witness=witness,
        structural_response=response,
        packet_sha256=response["packet_sha256"],
        allowed_artifact_refs={"node-1"},
        expected_role_id="RATIO_SCALAR",
    )
    states = {node["node_id"]: node["state"] for node in proof["nodes"]}
    assert states["ROLE"] == "FAIL"
    assert states["BINDING"] == "UNKNOWN"
    assert proof["overall_state"] == "FAIL"


@pytest.mark.parametrize(
    "mutation,error",
    [
        (lambda p: p.update({"sample_id": "sample-2"}), "proof_identity"),
        (
            lambda p: p["nodes"][1].update({"witness_ref": "REG-forged"}),
            "proof_witness",
        ),
        (
            lambda p: p["nodes"][2].update({"state": "PASS", "depends_on": []}),
            "proof_node_dependencies",
        ),
        (
            lambda p: p["nodes"][6].update({"state": "UNKNOWN"}),
            "proof_aggregate",
        ),
        (lambda p: p.update({"overall_state": "UNKNOWN"}), "proof_overall_state"),
        (
            lambda p: p["nodes"][4]["artifact_refs"].append("foreign-node"),
            "proof_artifact_ref",
        ),
    ],
)
def test_proof_validator_rejects_forgery_stitching_and_state_drift(
    mutation, error: str
) -> None:
    candidate, witness, proof = complete_alt_proof()
    mutation(proof)
    with pytest.raises(ValueError, match=error):
        validate_proof_record(
            proof,
            candidate=candidate,
            registry_entries=[witness],
            role_record_sha256=ROLE_HASH,
            graph_sha256=GRAPH_HASH,
            allowed_artifact_refs={"node-1"},
        )


def test_structural_prompt_is_hash_bound_symmetric_and_has_no_outcome_fields() -> None:
    b0, _ = b0_unknown_proof()
    alt, witness, _ = complete_alt_proof()
    duplicate_answer = registry_entry(registry_entry_id="REG-2")
    other_answer = registry_entry(
        answer="56",
        registry_entry_id="REG-3",
        program_id="CP-" + "e" * 24,
        derivation_id="DER-3",
        binding_id="B-" + "f" * 24,
    )
    common = {
        "sample_id": SAMPLE_ID,
        "question": "What is the value?",
        "role_record_sha256": ROLE_HASH,
        "graph_sha256": GRAPH_HASH,
        "evidence": [{"artifact_ref": "node-1", "text": "Row evidence"}],
        "registry_entries": [duplicate_answer, other_answer, witness],
    }
    b0_prompt = build_structural_challenge_prompt(b0, **common)
    alt_prompt = build_structural_challenge_prompt(alt, **common)
    assert set(b0_prompt) == set(alt_prompt)
    assert b0_prompt["challenge_ids"] == list(CHALLENGE_IDS)
    forbidden = {
        "gold",
        "correctness",
        "candidate_source",
        "selected_final",
        "planner_frequency",
    }
    assert not forbidden.intersection(str(b0_prompt).lower())
    assert b0_prompt["packet_sha256"] != alt_prompt["packet_sha256"]
    assert [
        item["registry_entry_id"]
        for item in b0_prompt["executed_registry_evidence"]
    ] == ["REG-1", "REG-3"]


def test_structural_evidence_projection_removes_only_mirrored_value_nodes() -> None:
    evidence = [
        {"artifact_ref": "cell_1_1", "node_type": "cell", "row": 1, "col": 1},
        {"artifact_ref": "val_1_1", "node_type": "value", "row": 1, "col": 1},
        {"artifact_ref": "val_2_2", "node_type": "value", "row": 2, "col": 2},
        {"artifact_ref": "header", "node_type": "header", "row": 0, "col": 0},
    ]
    assert [
        item["artifact_ref"] for item in project_structural_evidence(evidence)
    ] == ["cell_1_1", "val_2_2", "header"]


def test_structural_response_requires_exact_candidate_packet_and_challenge_roster() -> None:
    candidate, _, _ = complete_alt_proof()
    response = challenge_response(candidate)
    validated = validate_structural_challenge_response(
        response,
        candidate=candidate,
        packet_sha256=response["packet_sha256"],
        allowed_artifact_refs={"node-1"},
    )
    assert validated == response
    forged = deepcopy(response)
    forged["candidate_answer_hash"] = "f" * 64
    with pytest.raises(ValueError, match="challenge_identity"):
        validate_structural_challenge_response(
            forged,
            candidate=candidate,
            packet_sha256=response["packet_sha256"],
            allowed_artifact_refs={"node-1"},
        )
    incomplete = deepcopy(response)
    incomplete["responses"].pop()
    with pytest.raises(ValueError, match="challenge_roster"):
        validate_structural_challenge_response(
            incomplete,
            candidate=candidate,
            packet_sha256=response["packet_sha256"],
            allowed_artifact_refs={"node-1"},
        )


def test_structural_response_canonicalization_only_sorts_and_deduplicates_lists() -> None:
    candidate, _, _ = complete_alt_proof()
    response = challenge_response(candidate)
    response["responses"][0]["artifact_refs"] = ["node-2", "node-1", "node-2"]
    response["responses"][0]["claim_codes"] = [
        "INSUFFICIENT_EVIDENCE",
        "CITED_SUPPORT",
    ]
    canonical = canonicalize_structural_response(response)
    assert canonical["responses"][0]["artifact_refs"] == ["node-1", "node-2"]
    assert canonical["responses"][0]["claim_codes"] == [
        "CITED_SUPPORT",
        "INSUFFICIENT_EVIDENCE",
    ]


def test_structural_response_canonicalization_imposes_signature_applicability() -> None:
    candidate, _, _ = complete_alt_proof()
    response = challenge_response(candidate)
    response["responses"][0]["applicable"] = False
    response["responses"][1]["applicable"] = True
    applicability = challenge_applicability("LOOKUP_VALUE_SCALAR")
    canonical = canonicalize_structural_response(
        response, expected_applicability=applicability
    )
    role, scope, *_, lookup = canonical["responses"]
    assert role["applicable"] is True
    assert role["support"] == "SUPPORTED"
    assert scope == {
        "challenge_id": "SCOPE_COMPLETENESS",
        "applicable": False,
        "support": "NOT_SUPPORTED",
        "contradiction": "NOT_FOUND",
        "artifact_refs": [],
        "claim_codes": ["NOT_APPLICABLE_BY_SIGNATURE"],
    }
    assert lookup["applicable"] is True


def test_structural_response_canonicalization_restores_frozen_challenge_order() -> None:
    candidate, _, _ = complete_alt_proof()
    response = challenge_response(candidate)
    response["responses"].reverse()
    canonical = canonicalize_structural_response(response)
    assert [row["challenge_id"] for row in canonical["responses"]] == list(
        CHALLENGE_IDS
    )


def test_challenge_applicability_is_signature_derived_not_model_selected() -> None:
    count = challenge_applicability("COUNT_SCALAR")
    assert count["ROLE_OPERATION_SEMANTIC_FIT"] is True
    assert count["SCOPE_COMPLETENESS"] is True
    assert count["ORDER_AND_POLARITY"] is False
    assert count["LOOKUP_ENTITY_MEASURE_IDENTITY"] is False
    lookup = challenge_applicability("LOOKUP_VALUE_SCALAR")
    assert lookup["LOOKUP_ENTITY_MEASURE_IDENTITY"] is True
    assert lookup["SCOPE_COMPLETENESS"] is False


def test_dominance_commits_only_one_complete_registry_bound_alternative() -> None:
    b0_candidate, b0_proof = b0_unknown_proof()
    alt_candidate, witness, alt_proof = complete_alt_proof()
    roster_hash = canonical_json_hash(
        [b0_candidate["candidate_id"], alt_candidate["candidate_id"]]
    )
    decision = decide_proof_dominance(
        b0_proof=b0_proof,
        alternative_proofs=[alt_proof],
        candidates=[b0_candidate, alt_candidate],
        registry_entries=[witness],
        roster_sha256=roster_hash,
    )
    assert decision["action"] == "REPAIR"
    assert decision["selected_candidate_id"] == alt_candidate["candidate_id"]
    assert decision["selected_registry_entry_id"] == "REG-1"
    assert decision["validator_approved"] is True


@pytest.mark.parametrize("case", ["zero", "multiple", "b0_pass", "alt_unknown"])
def test_dominance_keeps_b0_for_every_nonunique_or_incomplete_case(case: str) -> None:
    b0_candidate, b0_proof = b0_unknown_proof()
    alt_candidate, witness, alt_proof = complete_alt_proof()
    alternatives = [alt_proof]
    candidates = [b0_candidate, alt_candidate]
    registry = [witness]
    if case == "zero":
        alternatives = []
        candidates = [b0_candidate]
        registry = []
    elif case == "multiple":
        other_candidate, other_witness, other_proof = complete_alt_proof("54")
        alternatives.append(other_proof)
        candidates.append(other_candidate)
        registry.append(other_witness)
    elif case == "b0_pass":
        b0_proof = deepcopy(alt_proof)
        b0_proof.update(
            {
                "candidate_id": b0_candidate["candidate_id"],
                "candidate_answer_hash": b0_candidate["candidate_answer_hash"],
                "candidate_source": "B0",
            }
        )
    else:
        alt_proof["nodes"][5]["state"] = "UNKNOWN"
        alt_proof["nodes"][6]["state"] = "UNKNOWN"
        alt_proof["overall_state"] = "UNKNOWN"
    roster_hash = canonical_json_hash(
        sorted(candidate["candidate_id"] for candidate in candidates)
    )
    decision = decide_proof_dominance(
        b0_proof=b0_proof,
        alternative_proofs=alternatives,
        candidates=candidates,
        registry_entries=registry,
        roster_sha256=roster_hash,
    )
    assert decision["action"] == "KEEP_B0"


def test_pairwise_verifier_is_confirmation_only_and_exactly_hash_bound() -> None:
    b0_candidate, b0_proof = b0_unknown_proof()
    alt_candidate, witness, alt_proof = complete_alt_proof()
    candidates = [b0_candidate, alt_candidate]
    roster_hash = canonical_json_hash(
        sorted(candidate["candidate_id"] for candidate in candidates)
    )
    verifier = {
        "b0_candidate_id": b0_candidate["candidate_id"],
        "b0_candidate_hash": b0_candidate["candidate_answer_hash"],
        "alternative_candidate_id": alt_candidate["candidate_id"],
        "alternative_candidate_hash": alt_candidate["candidate_answer_hash"],
        "differing_node_id": "ROLE",
        "b0_node_state": "UNKNOWN",
        "alternative_node_state": "PASS",
        "artifact_refs": [ROLE_HASH],
        "decision": "ALTERNATIVE_DOMINATES",
    }
    decision = decide_proof_verifier(
        b0_proof=b0_proof,
        alternative_proofs=[alt_proof],
        candidates=candidates,
        registry_entries=[witness],
        roster_sha256=roster_hash,
        verifier_response=verifier,
    )
    assert decision["action"] == "REPAIR"
    forged = deepcopy(verifier)
    forged["alternative_candidate_hash"] = "f" * 64
    rejected = decide_proof_verifier(
        b0_proof=b0_proof,
        alternative_proofs=[alt_proof],
        candidates=candidates,
        registry_entries=[witness],
        roster_sha256=roster_hash,
        verifier_response=forged,
    )
    assert rejected["action"] == "KEEP_B0"
    assert "verifier_identity_mismatch" in rejected["failure_reasons"]


def test_pairwise_verifier_cannot_create_eligibility() -> None:
    b0_candidate, b0_proof = b0_unknown_proof()
    decision = decide_proof_verifier(
        b0_proof=b0_proof,
        alternative_proofs=[],
        candidates=[b0_candidate],
        registry_entries=[],
        roster_sha256=canonical_json_hash([b0_candidate["candidate_id"]]),
        verifier_response={
            "decision": "ALTERNATIVE_DOMINATES",
            "alternative_candidate_id": "CAND-forged",
        },
    )
    assert decision["action"] == "KEEP_B0"
    assert "dominance_not_eligible" in decision["failure_reasons"]
