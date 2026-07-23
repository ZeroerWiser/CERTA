import json
from dataclasses import replace
from pathlib import Path

import jsonschema
import pytest

from certa.active_v1.artifact_authority import (
    ArtifactContext,
    recompute_binding_id_v3,
    serialize_plan_closure_v3,
    validate_grounding_record_v3,
)
from certa.derivations.schema import ExecutableDerivation
from certa.grounding.plan_closure import ClosureOutcome, GroundedAssignment, PlanClosure
from certa.reproducibility.canonical_json import canonical_json, canonical_json_hash
from tools.certa_active_v1_completion import verify_replay_state


REPO = Path(__file__).resolve().parents[2]
SCHEMA = json.loads((REPO / "schemas/active_v1/RAW_GROUNDING_RECORD_V3.schema.json").read_text())
SHAPES = json.loads((Path(__file__).parent / "fixtures/grounding_authority/real_table_shapes.json").read_text())["rows"]
ROLE_SHA = "a" * 64


def _assignment(
    index,
    state="EXACT",
    *,
    executable=False,
    answer=None,
    resource_complete=True,
):
    plan_id = "P0"
    role_bindings = {"AGGREGATION_SCOPE": ((f"scope-{index}",),), "TARGET_MEASURE": (f"measure-{index}",)}
    matched = (f"cell-{index}",) if state == "EXACT" else ((f"cell-{index}-a", f"cell-{index}-b") if state == "AMBIGUOUS" else ())
    outcome = {"EXACT": ClosureOutcome.UNIQUE_EXECUTABLE, "AMBIGUOUS": ClosureOutcome.AMBIGUOUS_BINDING, "UNRESOLVED": ClosureOutcome.UNRESOLVED_BINDING}[state]
    derivation_id = canonical_program_id = executable_program = ""
    if executable:
        program = {"answer_domain": "SCALAR", "operation_family": "COUNT", "plan_id": plan_id, "projected_answer": str(answer), "projection_operator": "SCALAR_RESULT_PROJECTION", "signature_id": "COUNT_SCALAR"}
        executable_program = canonical_json(program)
        canonical_program_id = f"CP-{canonical_json_hash(program, 24)}"
        derivation_id = f"PLC-{canonical_json_hash({'cp': canonical_program_id}, 20)}"
    assignment = GroundedAssignment(
        plan_id=plan_id,
        plan_ids=(plan_id,),
        assignment_id=f"A{index}",
        assignment_key=f"COUNT|assignment={index}",
        role_bindings=role_bindings,
        outcome=outcome,
        resolution_state={"EXACT": "UNIQUE"}.get(state, state),
        matched_cell_ids=matched,
        required_edge_triples=((matched[0], "MEMBER_OF", f"scope-{index}"),) if matched else (),
        derivation_id=derivation_id,
        failure_reasons=() if state == "EXACT" else (f"{state.lower()}_binding",),
        operation_family="COUNT",
        signature_id="COUNT_SCALAR",
        semantic_result_role="CARDINALITY",
        projection_operator="SCALAR_RESULT_PROJECTION",
        answer_domain="SCALAR",
        canonical_program_id=canonical_program_id,
        execution_outcome="EXECUTED" if executable else "NOT_RUN",
        projection_outcome="PROJECTED" if executable else "NOT_RUN",
        projected_answer=str(answer) if executable else "",
        resource_complete=resource_complete,
    )
    derivation = None
    if executable:
        derivation = ExecutableDerivation(
            derivation_id=derivation_id, source_candidate_id=f"closure:{canonical_program_id}",
            operation_family="COUNT", operand_node_ids=list(matched),
            required_edge_triples=list(assignment.required_edge_triples), typed_signature="COUNT_SCALAR",
            projection_operator="SCALAR_RESULT_PROJECTION", projected_answer=str(answer),
            output_domain="SCALAR", evidence_ids=[f"edge-{index}"], executable_program=executable_program,
            provenance_complete=True, availability="available",
            operation_metadata={"canonical_program_id": canonical_program_id},
        )
    return assignment, derivation


def _closure(pairs, *, reverse=False):
    assignments = [pair[0] for pair in pairs]
    derivations = [pair[1] for pair in pairs if pair[1] is not None]
    if reverse:
        assignments.reverse()
        derivations.reverse()
    return PlanClosure(plan_id="CLOSURE", operation_family="COUNT",
                       assignments=tuple(assignments), executable_derivations=tuple(derivations),
                       declared_assignment_count=len(assignments), realized_assignment_count=len(assignments),
                       deduplicated_program_count=len(derivations), resource_complete=True)


def _context(role_sha=ROLE_SHA):
    return ArtifactContext(sample_id="FX_SAMPLE", table_id="FX_TABLE", arm="C1_ROLE_ONLY",
                           role_id="COUNT_SCALAR", fixture_only=True, role_record_sha256=role_sha)


def _bundle(pairs, *, initial_answer="10", reverse=False, context=None):
    return serialize_plan_closure_v3(_closure(pairs, reverse=reverse),
                                     context=context or _context(), initial_answer=initial_answer)


def test_v3_authorizes_every_exact_assignment_without_plan_singleton():
    bundle = _bundle([_assignment(1, executable=True, answer="10"),
                      _assignment(2, executable=True, answer="11")])
    record = bundle.raw_groundings[0]
    jsonschema.validate(record, SCHEMA)
    validate_grounding_record_v3(record)
    assert "selected_binding_id" not in record
    assert record["first_match_used"] is False
    assert len(record["authorized_binding_ids"]) == 2
    assert record["authorized_binding_ids"] == sorted(record["authorized_binding_ids"])
    assert {row["binding_id"] for row in record["grounding_hypotheses"]} == set(record["authorized_binding_ids"])
    assert {row["resolution_state"] for row in record["grounding_hypotheses"]} == {"EXACT"}
    assert {row["binding_id"] for row in bundle.raw_derivations} == set(record["authorized_binding_ids"])


def test_unresolved_and_intra_assignment_ambiguous_are_rejected():
    bundle = _bundle(
        [
            _assignment(1, executable=True, answer="10"),
            _assignment(2, "AMBIGUOUS"),
            _assignment(3, "UNRESOLVED"),
            _assignment(4, resource_complete=False),
        ]
    )
    record = bundle.raw_groundings[0]
    assert len(record["authorized_binding_ids"]) == 1
    assert len(record["rejected_binding_ids"]) == 3
    assert record["ambiguous_hypothesis_count"] == 1
    assert record["unresolved_hypothesis_count"] == 1
    assert record["resource_incomplete_hypothesis_count"] == 1
    assert all(row["resolution_state"] != "AMBIGUOUS" or not row["grounding_valid"]
               for row in record["grounding_hypotheses"])


def test_executable_derivation_cannot_reference_rejected_assignment():
    assignment, derivation = _assignment(1, executable=True, answer="10")
    assignment = replace(
        assignment,
        outcome=ClosureOutcome.AMBIGUOUS_BINDING,
        resolution_state="AMBIGUOUS",
    )
    with pytest.raises(ValueError, match="rejected_assignment_has_derivation_id"):
        _bundle([(assignment, derivation)])


def test_binding_identity_includes_role_record_and_canonical_role_bindings():
    pair = _assignment(1, executable=True, answer="10")
    first = _bundle([pair])
    changed_role = _bundle([pair], context=_context("b" * 64))
    changed_bindings = _bundle([(replace(pair[0], role_bindings={"TARGET_MEASURE": ("other",)}), pair[1])])
    ids = {
        first.raw_groundings[0]["authorized_binding_ids"][0],
        changed_role.raw_groundings[0]["authorized_binding_ids"][0],
        changed_bindings.raw_groundings[0]["authorized_binding_ids"][0],
    }
    assert len(ids) == 3
    for bundle in (first, changed_role, changed_bindings):
        record = bundle.raw_groundings[0]
        hypothesis = record["grounding_hypotheses"][0]
        assert hypothesis["binding_id"] == recompute_binding_id_v3(record, hypothesis)


def test_authority_is_proposal_blind_and_canonical_order_invariant():
    pairs = [
        _assignment(1, executable=True, answer="10"),
        _assignment(2, executable=True, answer="11"),
        _assignment(3, "AMBIGUOUS"),
    ]
    first = _bundle(pairs, initial_answer="10")
    proposal_mutant = _bundle(pairs, initial_answer="not-an-answer")
    reversed_bundle = _bundle(pairs, initial_answer="10", reverse=True)
    assert first.raw_groundings == proposal_mutant.raw_groundings
    assert first == reversed_bundle
    assert [row["side"] for row in first.raw_derivations] != [
        row["side"] for row in proposal_mutant.raw_derivations
    ]


@pytest.mark.parametrize("shape", SHAPES, ids=lambda row: row["sample_id"])
def test_real_table_assignment_shapes_are_fully_accounted(shape):
    exact = shape["exact_count"]
    ambiguous = shape["ambiguous_count"]
    executed = shape["executed_derivation_count"]
    pairs = []
    for index in range(1, shape["candidate_count"] + 1):
        if index <= exact:
            pairs.append(_assignment(index, executable=index <= executed,
                                     answer=str(index) if index <= executed else None))
        elif index <= exact + ambiguous:
            pairs.append(_assignment(index, "AMBIGUOUS"))
        else:
            pairs.append(_assignment(index, "UNRESOLVED"))
    bundle = _bundle(pairs)
    record = bundle.raw_groundings[0]
    assert len(record["grounding_hypotheses"]) == shape["candidate_count"]
    assert len(record["authorized_binding_ids"]) == exact
    assert record["ambiguous_hypothesis_count"] == ambiguous
    assert record["unresolved_hypothesis_count"] == shape["unresolved_count"]
    assert len(bundle.raw_derivations) == executed
    assert len(bundle.registry_entries) == executed
    assert len(record["authorized_binding_ids"]) + len(record["rejected_binding_ids"]) == shape["candidate_count"]


def test_closed_schema_forbids_legacy_selector_and_first_match():
    record = _bundle([_assignment(1, executable=True, answer="10")]).raw_groundings[0]
    selector = dict(record, selected_binding_id=record["authorized_binding_ids"][0])
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(selector, SCHEMA)
    first_match = dict(record, first_match_used=True)
    with pytest.raises((jsonschema.ValidationError, ValueError)):
        jsonschema.validate(first_match, SCHEMA)
        validate_grounding_record_v3(first_match)


@pytest.mark.parametrize("field,value", (("lexical_score", 1.0), ("embedding_score", 1.0),
    ("model_confidence", 1.0), ("answer_agreement", True), ("gold_label", "10"), ("rank", 1)))
def test_closed_schema_forbids_heuristic_or_answer_based_authority(field, value):
    record = _bundle([_assignment(1, executable=True, answer="10")]).raw_groundings[0]
    mutant = json.loads(json.dumps(record))
    mutant["grounding_hypotheses"][0][field] = value
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(mutant, SCHEMA)


def test_offline_replay_verifier_requires_payload_and_closure_identity():
    closure = _closure([_assignment(1, executable=True, answer="10")])
    payload = {"planner_version": "fixture", "plans": []}
    state = {
        "sample_id": "FX_SAMPLE",
        "arm": "C1_ROLE_ONLY",
        "normalized_payload": payload,
        "closure_sha256": canonical_json_hash(closure.to_dict()),
    }
    local_validation = {
        "ok": True,
        "normalized_payload_sha256": canonical_json_hash(payload),
    }
    record = verify_replay_state(state, local_validation, closure)
    assert record["closure_match"] is True
    assert record["normalized_payload_match"] is True
    with pytest.raises(ValueError, match="replay_normalized_payload_drift"):
        verify_replay_state(state, dict(local_validation, normalized_payload_sha256="0" * 64), closure)
    with pytest.raises(ValueError, match="replay_closure_drift"):
        verify_replay_state(dict(state, closure_sha256="0" * 64), local_validation, closure)
