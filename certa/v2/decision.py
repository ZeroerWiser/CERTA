"""Fail-closed proof-dominance and confirmation-only verifier gates."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from certa.active_v1.answer_authority import active_answer_hash
from certa.reproducibility.canonical_json import canonical_json_hash


def _keep(*reasons: str) -> dict[str, Any]:
    return {
        "action": "KEEP_B0",
        "selected_candidate_id": "",
        "selected_registry_entry_id": "",
        "selected_answer_hash": "",
        "validator_approved": False,
        "failure_reasons": sorted(set(reasons)),
    }


def _candidate_map(
    candidates: Sequence[Mapping[str, Any]], roster_sha256: str
) -> dict[str, Mapping[str, Any]]:
    ids = [str(candidate.get("candidate_id") or "") for candidate in candidates]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("candidate_roster_invalid")
    accepted_hashes = {
        canonical_json_hash(ids),
        canonical_json_hash(sorted(ids)),
    }
    if roster_sha256 not in accepted_hashes:
        raise ValueError("candidate_roster_hash_mismatch")
    return {str(candidate["candidate_id"]): candidate for candidate in candidates}


def _node_map(proof: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    nodes = proof.get("nodes")
    if not isinstance(nodes, list):
        return {}
    return {
        str(node.get("node_id") or ""): node
        for node in nodes
        if isinstance(node, Mapping)
    }


def _eligible(
    *,
    b0_proof: Mapping[str, Any],
    alternative_proofs: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    registry_entries: Sequence[Mapping[str, Any]],
    roster_sha256: str,
) -> tuple[dict[str, Any], Mapping[str, Any] | None, Mapping[str, Any] | None]:
    try:
        candidate_by_id = _candidate_map(candidates, roster_sha256)
    except ValueError as exc:
        return _keep(str(exc)), None, None
    b0_candidate = candidate_by_id.get(str(b0_proof.get("candidate_id") or ""))
    if (
        b0_candidate is None
        or b0_candidate.get("candidate_source") != "B0"
        or b0_proof.get("candidate_answer_hash")
        != b0_candidate.get("candidate_answer_hash")
    ):
        return _keep("b0_proof_identity_mismatch"), None, None
    if b0_proof.get("overall_state") == "PASS":
        return _keep("b0_proof_pass"), None, None
    b0_nodes = _node_map(b0_proof)
    blocking = [
        node_id
        for node_id, node in b0_nodes.items()
        if node.get("required") is True and node.get("state") in {"FAIL", "UNKNOWN"}
    ]
    if not blocking:
        return _keep("b0_no_blocking_required_node"), None, None

    pass_alternatives = [
        proof for proof in alternative_proofs if proof.get("overall_state") == "PASS"
    ]
    if len(pass_alternatives) != 1:
        return _keep(f"complete_pass_alternative_count:{len(pass_alternatives)}"), None, None
    alternative = pass_alternatives[0]
    alternative_candidate = candidate_by_id.get(
        str(alternative.get("candidate_id") or "")
    )
    if (
        alternative_candidate is None
        or alternative_candidate.get("candidate_source") != "EXECUTED_REGISTRY"
        or alternative.get("candidate_answer_hash")
        != alternative_candidate.get("candidate_answer_hash")
    ):
        return _keep("alternative_proof_identity_mismatch"), None, None
    if len(alternative_proofs) != len(
        {
            str(proof.get("candidate_id") or "")
            for proof in alternative_proofs
        }
    ):
        return _keep("alternative_proof_duplicate_candidate"), None, None
    alt_nodes = _node_map(alternative)
    if not alt_nodes or any(
        node.get("required") is True and node.get("state") != "PASS"
        for node in alt_nodes.values()
    ):
        return _keep("alternative_required_node_not_pass"), None, None
    differing = [
        node_id
        for node_id in alt_nodes
        if alt_nodes[node_id].get("required") is True
        and alt_nodes[node_id].get("state") == "PASS"
        and b0_nodes.get(node_id, {}).get("state") in {"FAIL", "UNKNOWN"}
    ]
    if not differing:
        return _keep("no_exact_proof_node_difference"), None, None

    witness_refs = {
        str(node.get("witness_ref") or "") for node in alt_nodes.values()
    }
    if len(witness_refs) != 1 or "" in witness_refs:
        return _keep("alternative_witness_not_unique"), None, None
    witness_ref = next(iter(witness_refs))
    matches = [
        entry
        for entry in registry_entries
        if entry.get("registry_entry_id") == witness_ref
        and entry.get("sample_id") == alternative_candidate.get("sample_id")
        and entry.get("answer_hash")
        == alternative_candidate.get("candidate_answer_hash")
        and entry.get("validator_approved") is True
        and entry.get("materializer_approved") is True
        and active_answer_hash(entry.get("executed_answer"))
        == alternative_candidate.get("candidate_answer_hash")
    ]
    if len(matches) != 1:
        return _keep(f"registry_materializer_exact_match_count:{len(matches)}"), None, None
    if witness_ref not in alternative_candidate.get("registry_refs", ()):
        return _keep("selected_witness_outside_candidate"), None, None
    decision = {
        "action": "REPAIR",
        "selected_candidate_id": alternative_candidate["candidate_id"],
        "selected_registry_entry_id": witness_ref,
        "selected_answer_hash": alternative_candidate["candidate_answer_hash"],
        "validator_approved": True,
        "differing_node_ids": sorted(differing),
        "failure_reasons": [],
    }
    return decision, alternative, matches[0]


def decide_proof_dominance(
    *,
    b0_proof: Mapping[str, Any],
    alternative_proofs: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    registry_entries: Sequence[Mapping[str, Any]],
    roster_sha256: str,
) -> dict[str, Any]:
    """Commit only a unique complete, exactly materializable alternative."""
    decision, _, _ = _eligible(
        b0_proof=b0_proof,
        alternative_proofs=alternative_proofs,
        candidates=candidates,
        registry_entries=registry_entries,
        roster_sha256=roster_sha256,
    )
    return decision


def decide_proof_verifier(
    *,
    b0_proof: Mapping[str, Any],
    alternative_proofs: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    registry_entries: Sequence[Mapping[str, Any]],
    roster_sha256: str,
    verifier_response: Mapping[str, Any],
) -> dict[str, Any]:
    """Allow the verifier to confirm or veto, never create, eligibility."""
    eligible, alternative, _ = _eligible(
        b0_proof=b0_proof,
        alternative_proofs=alternative_proofs,
        candidates=candidates,
        registry_entries=registry_entries,
        roster_sha256=roster_sha256,
    )
    if eligible["action"] != "REPAIR" or alternative is None:
        return _keep("dominance_not_eligible", *eligible["failure_reasons"])
    expected_fields = {
        "b0_candidate_id",
        "b0_candidate_hash",
        "alternative_candidate_id",
        "alternative_candidate_hash",
        "differing_node_id",
        "b0_node_state",
        "alternative_node_state",
        "artifact_refs",
        "decision",
    }
    if set(verifier_response) != expected_fields:
        return _keep("verifier_fields_mismatch")
    identities = (
        verifier_response.get("b0_candidate_id") == b0_proof.get("candidate_id")
        and verifier_response.get("b0_candidate_hash")
        == b0_proof.get("candidate_answer_hash")
        and verifier_response.get("alternative_candidate_id")
        == alternative.get("candidate_id")
        and verifier_response.get("alternative_candidate_hash")
        == alternative.get("candidate_answer_hash")
    )
    if not identities:
        return _keep("verifier_identity_mismatch")
    node_id = str(verifier_response.get("differing_node_id") or "")
    b0_node = _node_map(b0_proof).get(node_id)
    alt_node = _node_map(alternative).get(node_id)
    if (
        b0_node is None
        or alt_node is None
        or b0_node.get("required") is not True
        or alt_node.get("required") is not True
        or b0_node.get("state") not in {"FAIL", "UNKNOWN"}
        or alt_node.get("state") != "PASS"
        or verifier_response.get("b0_node_state") != b0_node.get("state")
        or verifier_response.get("alternative_node_state") != alt_node.get("state")
    ):
        return _keep("verifier_node_difference_invalid")
    refs = verifier_response.get("artifact_refs")
    allowed_refs = set(b0_node.get("artifact_refs", ())) | set(
        alt_node.get("artifact_refs", ())
    )
    if (
        not isinstance(refs, list)
        or not refs
        or refs != sorted(set(refs))
        or not set(refs).issubset(allowed_refs)
    ):
        return _keep("verifier_artifact_refs_invalid")
    if verifier_response.get("decision") != "ALTERNATIVE_DOMINATES":
        return _keep("verifier_did_not_confirm")
    return {**eligible, "verifier_approved": True, "verifier_node_id": node_id}
