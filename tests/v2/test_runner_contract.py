from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

from graph_builder import build_hceg

from certa.active_v1.dataset_adapter_v1 import HiTabAdapterV1
from certa.derivations.schema import ExecutableDerivation
from certa.grounding.plan_closure import (
    ClosureOutcome,
    GroundedAssignment,
    PlanClosure,
)
from certa.v2.search import SEARCH_SCHEDULE, assert_proposal_blind
import tools.certa_v2_run as runner
from tools.certa_v2_run import build_search_views


BASE = Path("/home/hsh/ME/Table/EMNLP2026")
V1 = (
    BASE
    / "certa_active_v1_outputs"
    / "CERTA_FINAL_MULTI_DATASET_ADAPTER_AND_METHOD_COMPLETION"
)
ROLE_ROOT = (
    BASE
    / "certa_active_v1_outputs"
    / "CERTA_ACTIVE_V1_ROLE_V3_FINAL"
    / "freeze"
)
MATRIX = (
    BASE
    / "certa_active_v1_outputs"
    / "CERTA_ACTIVE_V1_FINAL_ASSIGNMENT_LEVEL_GROUNDING_AUTHORITY_REPLAY"
    / "freeze"
    / "CONSTRUCTOR_CAPABILITY_MATRIX.json"
)


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_runner_builds_exact_three_proposal_blind_search_views() -> None:
    sample = read(V1 / "validation" / "samples" / "a922b621c6b56fd4c47dcd185347a4db.json")
    runtime = next(
        json.loads(line)
        for line in (
            V1 / "data" / "hitab" / "validation_runtime_v3.jsonl"
        ).read_text(encoding="utf-8").splitlines()
        if json.loads(line)["id"] == sample["sample_id"]
    )
    adapter = HiTabAdapterV1(
        BASE / "CERTA" / "dataset" / "hitab" / "tables" / "raw"
    )
    table = adapter.canonicalize_table(
        adapter.resolve_table(sample["table_id"], runtime_record=runtime)
    )["table_payload"]["graph_payload"]
    graph = build_hceg(table, runtime["question"])
    views = build_search_views(
        question=runtime["question"],
        graph=graph,
        table=table,
        role=sample["role"],
        retrieval=sample["retrieval"]["retrieval"],
        matrix=read(MATRIX),
        role_schema=read(ROLE_ROOT / "ROLE_V3_OUTPUT_SCHEMA.json"),
        role_registry=read(ROLE_ROOT / "ROLE_V3_CANONICAL_REGISTRY.json"),
    )
    assert set(views) == {slot.view_kind for slot in SEARCH_SCHEDULE}
    assert "table_values" not in views["ROLE_COMPLETE"]
    assert "retrieval_advisory" in views["RETRIEVAL_COMPLETE"]
    assert "table_values" in views["VALUE_AWARE_PROPOSAL_BLIND"]
    for view in views.values():
        assert_proposal_blind(view)


def test_runner_merge_deduplicates_executable_assignments_within_a_closure() -> None:
    assignment = GroundedAssignment(
        plan_id="P0",
        assignment_id="A0",
        assignment_key="COUNT:P0",
        role_bindings={},
        outcome=ClosureOutcome.UNIQUE_EXECUTABLE,
        derivation_id="D0",
    )
    derivation = ExecutableDerivation(
        derivation_id="D0",
        source_candidate_id="closure:P0",
        operation_family="COUNT",
        operand_node_ids=[],
        required_edge_triples=[],
        typed_signature="COUNT_SCALAR",
        projection_operator="SCALAR_RESULT_PROJECTION",
        projected_answer="1",
        output_domain="SCALAR",
        evidence_ids=[],
        executable_program="{}",
        provenance_complete=True,
        availability="available",
    )
    closure = PlanClosure(
        plan_id="P0",
        operation_family="COUNT",
        assignments=(assignment, assignment),
        executable_derivations=(derivation, derivation),
    )
    merged = runner._merge_closures([closure])
    assert merged.assignments == (assignment,)
    assert merged.executable_derivations == (derivation,)
    namespaced = runner._namespace_closure_plan(closure, "P7")
    assert {item.plan_id for item in namespaced.assignments} == {"P7"}
    assert {item.plan_ids for item in namespaced.assignments} == {("P7",)}
    assert namespaced.executable_derivations[0].operation_metadata["plan_ids"] == [
        "P7"
    ]
    ambiguous = replace(
        assignment,
        derivation_id="",
        outcome=ClosureOutcome.AMBIGUOUS_BINDING,
    )
    merged_ambiguous = runner._merge_closures(
        [
            PlanClosure(
                plan_id="P0",
                operation_family="COUNT",
                assignments=(ambiguous, ambiguous),
            )
        ]
    )
    assert merged_ambiguous.assignments == (ambiguous,)


def test_runner_merges_six_valid_calls_before_grounding(
    monkeypatch, tmp_path: Path
) -> None:
    sample = read(V1 / "validation" / "samples" / "a922b621c6b56fd4c47dcd185347a4db.json")
    runtime = next(
        json.loads(line)
        for line in (
            V1 / "data" / "hitab" / "validation_runtime_v3.jsonl"
        ).read_text(encoding="utf-8").splitlines()
        if json.loads(line)["id"] == sample["sample_id"]
    )
    adapter = HiTabAdapterV1(
        BASE / "CERTA" / "dataset" / "hitab" / "tables" / "raw"
    )
    table = adapter.canonicalize_table(
        adapter.resolve_table(sample["table_id"], runtime_record=runtime)
    )["table_payload"]["graph_payload"]
    graph = build_hceg(table, runtime["question"])
    matrix = read(MATRIX)
    views = build_search_views(
        question=runtime["question"],
        graph=graph,
        table=table,
        role=sample["role"],
        retrieval=sample["retrieval"]["retrieval"],
        matrix=matrix,
        role_schema=read(ROLE_ROOT / "ROLE_V3_OUTPUT_SCHEMA.json"),
        role_registry=read(ROLE_ROOT / "ROLE_V3_CANONICAL_REGISTRY.json"),
    )
    outputs = V1 / "validation" / "model_outputs" / sample["sample_id"]

    def fake_call(*_args, call_id: str, **_kwargs):
        source = (
            "PLANNER_C2_COMPLETE.json"
            if "RETRIEVAL" in call_id
            else "PLANNER_C1_COMPLETE.json"
        )
        return read(outputs / source)

    monkeypatch.setattr(runner, "_call", fake_call)
    closure, search = runner._search(
        generator=None,
        output=tmp_path,
        split="development",
        sample_id=sample["sample_id"],
        views=views,
        graph=graph,
        matrix=matrix,
    )
    assert search["attempt_count"] == 6
    assert len(search["attempts"]) == 6
    assert closure.resource_complete is True
    assert closure.realized_assignment_count > 0


def test_structured_transport_removes_only_unsupported_uniqueness_keywords() -> None:
    full = read(
        BASE
        / "CERTA"
        / "schemas"
        / "v2"
        / "STRUCTURAL_CHALLENGE_RESPONSE.schema.json"
    )
    transport = runner._transport_schema(full)
    assert "uniqueItems" in json.dumps(full)
    assert "uniqueItems" not in json.dumps(transport)
    assert transport["additionalProperties"] is False
    assert transport["properties"]["responses"]["minItems"] == 7


def test_malformed_pairwise_verifier_output_fails_closed() -> None:
    assert runner._parse_json_mapping('{"decision":"UNKNOWN"}') == {
        "decision": "UNKNOWN"
    }
    assert runner._parse_json_mapping('{"decision":"unterminated') == {}
    assert runner._parse_json_mapping("[]") == {}


def test_pairwise_verifier_has_frozen_nontruncating_output_budget() -> None:
    assert runner.VERIFIER_MAX_TOKENS == 512
    profile = (
        BASE
        / "CERTA"
        / "configs"
        / "profiles"
        / "certa_v2_bounded_proof_search.env"
    ).read_text(encoding="utf-8")
    assert "CERTA_V2_VERIFIER_MAX_TOKENS=512\n" in profile
    assert "CERTA_V2_PRIMARY_VARIANT=V2-C_PROOF_VERIFIER\n" in profile
