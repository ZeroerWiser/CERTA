"""Closed seven-node proof DAG and deterministic validation."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from certa.v2.candidates import validate_registry_entry
from certa.v2.runtime import CHALLENGE_IDS, validate_structural_challenge_response


NODE_IDS = (
    "ROLE",
    "BINDING",
    "OPERATION",
    "EXECUTION",
    "PROJECTION",
    "STRUCTURAL_CHALLENGE",
    "CANDIDATE_PROOF_STATE",
)
NODE_DEPENDENCIES = {
    "ROLE": (),
    "BINDING": ("ROLE",),
    "OPERATION": ("ROLE", "BINDING"),
    "EXECUTION": ("BINDING", "OPERATION"),
    "PROJECTION": ("EXECUTION",),
    "STRUCTURAL_CHALLENGE": (
        "ROLE",
        "BINDING",
        "OPERATION",
        "EXECUTION",
        "PROJECTION",
    ),
    "CANDIDATE_PROOF_STATE": (
        "ROLE",
        "BINDING",
        "OPERATION",
        "EXECUTION",
        "PROJECTION",
        "STRUCTURAL_CHALLENGE",
    ),
}
_TOP_FIELDS = {
    "sample_id",
    "candidate_id",
    "candidate_answer_hash",
    "candidate_source",
    "nodes",
    "overall_state",
    "registry_refs",
}
_NODE_FIELDS = {
    "node_id",
    "required",
    "depends_on",
    "state",
    "authority",
    "witness_ref",
    "artifact_refs",
    "reason_codes",
    "challenge_responses",
}
_STATES = frozenset(("PASS", "FAIL", "UNKNOWN"))


def _node(
    node_id: str,
    *,
    state: str,
    authority: str,
    witness_ref: str,
    artifact_refs: Sequence[str],
    reason_codes: Sequence[str] = (),
    challenge_responses: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    return {
        "node_id": node_id,
        "required": True,
        "depends_on": list(NODE_DEPENDENCIES[node_id]),
        "state": state,
        "authority": authority,
        "witness_ref": witness_ref,
        "artifact_refs": sorted(set(artifact_refs)),
        "reason_codes": sorted(set(reason_codes)),
        "challenge_responses": [dict(item) for item in challenge_responses],
    }


def _witness_artifacts(witness: Mapping[str, Any]) -> set[str]:
    values = {
        str(witness[field])
        for field in (
            "registry_entry_id",
            "derivation_id",
            "canonical_program_id",
            "binding_id",
            "role_record_sha256",
            "graph_sha256",
            "answer_hash",
        )
    }
    values.update(str(item) for item in witness.get("provenance_ids", ()))
    return values


def _validate_witness(
    witness: Mapping[str, Any],
    candidate: Mapping[str, Any],
    role_record_sha256: str,
    graph_sha256: str,
) -> None:
    validate_registry_entry(witness, str(candidate["sample_id"]))
    if witness.get("registry_entry_id") not in candidate.get("registry_refs", ()):
        raise ValueError("proof_witness_outside_candidate")
    if witness.get("answer_hash") != candidate.get("candidate_answer_hash"):
        raise ValueError("proof_witness_candidate_hash_mismatch")
    if witness.get("role_record_sha256") != role_record_sha256:
        raise ValueError("proof_witness_role_hash_mismatch")
    if witness.get("graph_sha256") != graph_sha256:
        raise ValueError("proof_witness_graph_hash_mismatch")


def _structural_state(
    responses: Sequence[Mapping[str, Any]], deterministic_pass: bool
) -> tuple[str, list[str]]:
    if not deterministic_pass:
        return "UNKNOWN", ["BLOCKED_BY_DEPENDENCY"]
    applicable = [row for row in responses if row.get("applicable") is True]
    if any(row.get("contradiction") == "FOUND" for row in applicable):
        return "FAIL", ["CITED_STRUCTURAL_CONTRADICTION"]
    if all(
        row.get("support") == "SUPPORTED"
        and row.get("contradiction") == "NOT_FOUND"
        for row in applicable
    ):
        return "PASS", ["ALL_APPLICABLE_CHALLENGES_SUPPORTED"]
    return "UNKNOWN", ["STRUCTURAL_SUPPORT_INCOMPLETE"]


def _aggregate(states: Sequence[str]) -> str:
    if "FAIL" in states:
        return "FAIL"
    if states and all(state == "PASS" for state in states):
        return "PASS"
    return "UNKNOWN"


def build_proof_record(
    candidate: Mapping[str, Any],
    *,
    role_record_sha256: str,
    graph_sha256: str,
    witness: Mapping[str, Any] | None,
    structural_response: Mapping[str, Any],
    packet_sha256: str,
    allowed_artifact_refs: set[str],
    expected_role_id: str = "",
    expected_challenge_applicability: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    """Materialize a proof from one candidate and at most one witness chain."""
    witness_ref = ""
    if witness is not None:
        _validate_witness(witness, candidate, role_record_sha256, graph_sha256)
        witness_ref = str(witness["registry_entry_id"])
        allowed_refs = _witness_artifacts(witness)
    else:
        allowed_refs = {role_record_sha256, graph_sha256}
    allowed_refs.update(allowed_artifact_refs)
    response = validate_structural_challenge_response(
        structural_response,
        candidate=candidate,
        packet_sha256=packet_sha256,
        allowed_artifact_refs=allowed_refs,
        expected_applicability=expected_challenge_applicability,
    )

    if witness is None:
        deterministic_state = "UNKNOWN"
        deterministic_reason = ["NO_EXECUTABLE_WITNESS"]
        artifacts = {
            "ROLE": [role_record_sha256],
            "BINDING": [],
            "OPERATION": [],
            "EXECUTION": [],
            "PROJECTION": [],
        }
    else:
        role_matches = (
            not expected_role_id
            or witness.get("signature_id") == expected_role_id
        )
        deterministic_state = "PASS" if role_matches else "UNKNOWN"
        deterministic_reason = [] if role_matches else ["BLOCKED_BY_ROLE"]
        artifacts = {
            "ROLE": [role_record_sha256],
            "BINDING": [
                str(witness["binding_id"]),
                *[str(item) for item in witness["provenance_ids"]],
            ],
            "OPERATION": [
                str(witness["canonical_program_id"]),
                str(witness["derivation_id"]),
            ],
            "EXECUTION": [
                str(witness["canonical_program_id"]),
                str(witness["derivation_id"]),
            ],
            "PROJECTION": [
                str(witness["canonical_program_id"]),
                str(witness["answer_hash"]),
                *[str(item) for item in witness["provenance_ids"]],
            ],
        }
    role_state = (
        "FAIL"
        if witness is not None
        and expected_role_id
        and witness.get("signature_id") != expected_role_id
        else deterministic_state
    )
    nodes = []
    for node_id in NODE_IDS[:5]:
        nodes.append(
            _node(
                node_id,
                state=role_state if node_id == "ROLE" else deterministic_state,
                authority="DETERMINISTIC",
                witness_ref=witness_ref,
                artifact_refs=artifacts[node_id],
                reason_codes=(
                    ["ROLE_SIGNATURE_CONTRADICTION"]
                    if node_id == "ROLE" and role_state == "FAIL"
                    else deterministic_reason
                ),
            )
        )
    structural_responses = response["responses"]
    structural_state, structural_reasons = _structural_state(
        structural_responses, deterministic_state == "PASS"
    )
    structural_refs = [
        str(ref) for row in structural_responses for ref in row["artifact_refs"]
    ]
    nodes.append(
        _node(
            "STRUCTURAL_CHALLENGE",
            state=structural_state,
            authority="MODEL_SEMANTIC_VALIDATED",
            witness_ref=witness_ref,
            artifact_refs=structural_refs,
            reason_codes=structural_reasons,
            challenge_responses=structural_responses,
        )
    )
    overall = _aggregate([node["state"] for node in nodes])
    aggregate_refs = [
        str(ref) for node in nodes for ref in node["artifact_refs"]
    ]
    nodes.append(
        _node(
            "CANDIDATE_PROOF_STATE",
            state=overall,
            authority="DETERMINISTIC_DERIVED",
            witness_ref=witness_ref,
            artifact_refs=aggregate_refs,
            reason_codes=[f"DERIVED_{overall}"],
        )
    )
    return {
        "sample_id": candidate["sample_id"],
        "candidate_id": candidate["candidate_id"],
        "candidate_answer_hash": candidate["candidate_answer_hash"],
        "candidate_source": candidate["candidate_source"],
        "nodes": nodes,
        "overall_state": overall,
        "registry_refs": list(candidate["registry_refs"]),
    }


def validate_proof_record(
    record: Mapping[str, Any],
    *,
    candidate: Mapping[str, Any],
    registry_entries: Sequence[Mapping[str, Any]],
    role_record_sha256: str,
    graph_sha256: str,
    allowed_artifact_refs: set[str],
    expected_role_id: str = "",
    expected_challenge_applicability: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    """Fail closed on identity, DAG, witness, citation, or aggregate drift."""
    if set(record) != _TOP_FIELDS:
        raise ValueError("proof_top_fields_mismatch")
    if any(
        record.get(field) != candidate.get(field)
        for field in (
            "sample_id",
            "candidate_id",
            "candidate_answer_hash",
            "candidate_source",
        )
    ):
        raise ValueError("proof_identity_mismatch")
    if record.get("registry_refs") != candidate.get("registry_refs"):
        raise ValueError("proof_registry_refs_mismatch")
    nodes = record.get("nodes")
    if not isinstance(nodes, list) or [
        node.get("node_id") if isinstance(node, Mapping) else None for node in nodes
    ] != list(NODE_IDS):
        raise ValueError("proof_node_roster_mismatch")
    by_ref = {
        str(entry.get("registry_entry_id") or ""): entry for entry in registry_entries
    }
    if len(by_ref) != len(registry_entries):
        raise ValueError("proof_registry_duplicate_ref")
    witness_refs = {str(node.get("witness_ref") or "") for node in nodes}
    if len(witness_refs) != 1:
        raise ValueError("proof_witness_stitching")
    witness_ref = next(iter(witness_refs))
    witness = by_ref.get(witness_ref) if witness_ref else None
    if witness_ref and witness is None:
        raise ValueError("proof_witness_missing")
    if witness is not None:
        _validate_witness(witness, candidate, role_record_sha256, graph_sha256)
        allowed_refs = _witness_artifacts(witness)
    else:
        allowed_refs = {role_record_sha256, graph_sha256}
    allowed_refs.update(allowed_artifact_refs)

    states: dict[str, str] = {}
    for node in nodes:
        if set(node) != _NODE_FIELDS:
            raise ValueError("proof_node_fields_mismatch")
        node_id = str(node["node_id"])
        if node.get("required") is not True:
            raise ValueError("proof_node_required_mismatch")
        if node.get("depends_on") != list(NODE_DEPENDENCIES[node_id]):
            raise ValueError("proof_node_dependencies_mismatch")
        if node.get("state") not in _STATES:
            raise ValueError("proof_node_state_invalid")
        if node.get("artifact_refs") != sorted(set(node.get("artifact_refs", ()))):
            raise ValueError("proof_artifact_ref_not_canonical")
        if not set(node.get("artifact_refs", ())).issubset(allowed_refs):
            raise ValueError("proof_artifact_ref_outside_authority")
        if node.get("reason_codes") != sorted(set(node.get("reason_codes", ()))):
            raise ValueError("proof_reason_codes_not_canonical")
        if any(states.get(dependency) != "PASS" for dependency in node["depends_on"]):
            if node["state"] == "PASS":
                raise ValueError("proof_dependency_pass_violation")
        states[node_id] = str(node["state"])

    role_matches = (
        witness is not None
        and (
            not expected_role_id
            or witness.get("signature_id") == expected_role_id
        )
    )
    expected_role = (
        "PASS" if role_matches else ("FAIL" if witness is not None else "UNKNOWN")
    )
    expected_deterministic = "PASS" if role_matches else "UNKNOWN"
    if states["ROLE"] != expected_role or any(
        states[node_id] != expected_deterministic for node_id in NODE_IDS[1:5]
    ):
        raise ValueError("proof_deterministic_state_mismatch")
    challenge_rows = nodes[5]["challenge_responses"]
    challenge_record = {
        "sample_id": record["sample_id"],
        "candidate_id": record["candidate_id"],
        "candidate_answer_hash": record["candidate_answer_hash"],
        "packet_sha256": "_validator_internal_",
        "responses": challenge_rows,
    }
    validate_structural_challenge_response(
        challenge_record,
        candidate=candidate,
        packet_sha256="_validator_internal_",
        allowed_artifact_refs=allowed_refs,
        expected_applicability=expected_challenge_applicability,
    )
    expected_structural, _ = _structural_state(
        challenge_rows, expected_deterministic == "PASS"
    )
    if states["STRUCTURAL_CHALLENGE"] != expected_structural:
        raise ValueError("proof_structural_state_mismatch")
    expected_aggregate = _aggregate([states[node_id] for node_id in NODE_IDS[:6]])
    if states["CANDIDATE_PROOF_STATE"] != expected_aggregate:
        raise ValueError("proof_aggregate_mismatch")
    if record.get("overall_state") != expected_aggregate:
        raise ValueError("proof_overall_state_mismatch")
    return dict(record)
