from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from certa.active_v1.answer_authority import active_answer_hash
from certa.active_v1.artifact_authority import recompute_binding_id_v3
from certa.reproducibility.canonical_json import canonical_json, canonical_json_hash
from certa.v2.authority import materialize_executed_registry


def fixture() -> dict:
    sample_id = "sample-1"
    table_id = "table-1"
    role_hash = "a" * 64
    graph_hash = "b" * 64
    answer = "55"
    program = {"operation": "LOOKUP", "operands": ["cell-1"]}
    program_id = "CP-" + canonical_json_hash(program, 24)
    derivation_id = "DER-1"
    grounding = {
        "schema_version": "certa_active_grounding_record_v3",
        "fixture_only": False,
        "sample_id": sample_id,
        "table_id": table_id,
        "arm": "C1_C2_EXACT_PROGRAM_UNION",
        "role_record_sha256": role_hash,
        "plan_id": "P0",
        "required_operand_roles": ["TARGET_ENTITY"],
        "grounding_hypotheses": [],
        "authorized_binding_ids": [],
        "rejected_binding_ids": [],
        "exact_hypothesis_count": 1,
        "ambiguous_hypothesis_count": 0,
        "unresolved_hypothesis_count": 0,
        "resource_incomplete_hypothesis_count": 0,
        "first_match_used": False,
    }
    hypothesis = {
        "assignment_id": "A1",
        "assignment_key": "LOOKUP|cell-1",
        "role_bindings": {"TARGET_ENTITY": ["cell-1"]},
        "role_bindings_sha256": canonical_json_hash(
            {"TARGET_ENTITY": ["cell-1"]}
        ),
        "operand_node_ids": ["cell-1"],
        "resolution_state": "EXACT",
        "grounding_valid": True,
        "derivation_id": derivation_id,
        "canonical_program_id": program_id,
        "failure_reasons": [],
    }
    hypothesis["binding_id"] = recompute_binding_id_v3(grounding, hypothesis)
    grounding["grounding_hypotheses"] = [hypothesis]
    grounding["authorized_binding_ids"] = [hypothesis["binding_id"]]
    raw = {
        "schema_version": "certa_active_derivation_record_v2",
        "fixture_only": False,
        "sample_id": sample_id,
        "arm": "C1_C2_EXACT_PROGRAM_UNION",
        "derivation_id": derivation_id,
        "plan_id": "P0",
        "binding_id": hypothesis["binding_id"],
        "side": "ALTERNATIVE",
        "signature_id": "LOOKUP_VALUE_SCALAR",
        "answer_role": "SCALAR",
        "projection": "VALUE_PROJECTION",
        "canonical_program_id": program_id,
        "answer_class_id": "AC-" + active_answer_hash(answer)[:24],
        "projected_answer_hash": active_answer_hash(answer),
        "operand_node_ids": ["cell-1"],
        "provenance_ids": ["cell-1"],
        "execution_status": "EXECUTED",
        "projection_status": "VALID",
    }
    registry_base = {
        "schema_version": "certa_active_registry_entry_v2",
        "fixture_only": False,
        "sample_id": sample_id,
        "arm": raw["arm"],
        "derivation_id": derivation_id,
        "side": "ALTERNATIVE",
        "canonical_program_id": program_id,
        "answer_class_id": raw["answer_class_id"],
        "answer_hash": raw["projected_answer_hash"],
        "provenance_ids": ["cell-1"],
    }
    registry = {
        **registry_base,
        "registry_entry_id": "REG-" + canonical_json_hash(registry_base, 24),
    }
    vault = {
        "schema_version": "certa_executed_answer_vault_v1",
        "sample_id": sample_id,
        "table_id": table_id,
        "variant_id": "CERTA_V2_BOUNDED_SEARCH",
        "arm": raw["arm"],
        "canonical_program_id": program_id,
        "derivation_id": derivation_id,
        "answer_hash": active_answer_hash(answer),
        "executed_answer": answer,
    }
    executed = SimpleNamespace(
        derivation_id=derivation_id,
        executable_program=canonical_json(program),
        projected_answer=answer,
        provenance_complete=True,
        operand_node_ids=("cell-1",),
        evidence_ids=("cell-1",),
        required_edge_triples=(),
        typed_signature="LOOKUP_VALUE_SCALAR",
        projection_operator="VALUE_PROJECTION",
        output_domain="SCALAR",
        availability="available",
        operation_metadata={
            "canonical_program_id": program_id,
            "resource_complete": True,
        },
    )
    return {
        "sample_id": sample_id,
        "table_id": table_id,
        "role_hash": role_hash,
        "graph_hash": graph_hash,
        "grounding": grounding,
        "raw": raw,
        "registry": registry,
        "vault": vault,
        "executed": executed,
    }


def materialize(data: dict) -> list[dict]:
    return materialize_executed_registry(
        sample_id=data["sample_id"],
        table_id=data["table_id"],
        role_record_sha256=data["role_hash"],
        graph_sha256=data["graph_hash"],
        raw_groundings=[data["grounding"]],
        raw_derivations=[data["raw"]],
        registry_entries=[data["registry"]],
        answer_vault=[data["vault"]],
        executed_derivations=[data["executed"]],
        graph_artifact_refs={"cell-1"},
    )


def test_materializer_reconciles_every_authority_and_recomputes_program_answer() -> None:
    data = fixture()
    result = materialize(data)
    assert len(result) == 1
    assert result[0]["registry_entry_id"] == data["registry"]["registry_entry_id"]
    assert result[0]["binding_id"] == data["raw"]["binding_id"]
    assert result[0]["answer_hash"] == active_answer_hash("55")
    assert result[0]["validator_approved"] is True
    assert result[0]["materializer_approved"] is True


@pytest.mark.parametrize(
    "mutation,error",
    [
        (
            lambda d: d["vault"].update({"executed_answer": "54"}),
            "authority_vault_answer",
        ),
        (
            lambda d: d["raw"].update({"binding_id": "B-forged"}),
            "authority_grounding_hypothesis",
        ),
        (
            lambda d: d["registry"].update({"answer_hash": "f" * 64}),
            "registry_derivation_mismatch",
        ),
        (
            lambda d: setattr(
                d["executed"], "executable_program", json.dumps({"forged": True})
            ),
            "authority_program",
        ),
        (
            lambda d: d["raw"].update({"provenance_ids": ["foreign"]}),
            "authority_provenance",
        ),
    ],
)
def test_materializer_fails_closed_on_forged_or_cross_artifact_state(
    mutation, error: str
) -> None:
    data = deepcopy(fixture())
    mutation(data)
    with pytest.raises(ValueError, match=error):
        materialize(data)
