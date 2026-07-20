"""Targeted P0-1..P0-5 validity reproducers for the final lookup-only round."""

import ast
import json
import unittest
from pathlib import Path
from typing import Optional, Tuple

import jsonschema

from graph_builder import EdgeType, GraphEdge, GraphNode, HCEG, NodeType
from executor import ExecutorResult, OperationType

from certa.derivations.contrast import build_compact_behavioral_contrast_v3
from certa.grounding.plan_closure import ClosureOutcome, build_plan_closure
from certa.planner.schema_view import build_proposal_blind_planner_view
from certa.planner.typed_planner import (
    build_typed_planner_response_schema,
    validate_typed_planner_output,
)
from certa.repair.causal_epistemic_agent import run_causal_epistemic_repair
from tools.certa_round1_artifacts import _frozen_input


LOOKUP_SIGNATURE = "LOOKUP_VALUE_SCALAR"


def pipeline_arbitrate():
    """Load the exact function without importing optional training dependencies."""
    path = Path(__file__).resolve().parents[1] / "run_cscr_pipeline.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    node = next(
        item for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == "arbitrate"
    )
    namespace = {
        "Optional": Optional,
        "Tuple": Tuple,
        "ExecutorResult": ExecutorResult,
        "OperationType": OperationType,
    }
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["arbitrate"]


def lookup_graph():
    graph = HCEG()
    graph.add_node(GraphNode("measure", NodeType.HEADER, row=0, col=1, text="Value"))
    graph.add_node(GraphNode("entity", NodeType.HEADER, row=1, col=0, text="A"))
    graph.add_node(GraphNode("cell", NodeType.CELL, row=1, col=1, text="42", numeric_value=42.0))
    graph.add_edge(GraphEdge("cell", "entity", EdgeType.ROW_PATH))
    graph.add_edge(GraphEdge("cell", "measure", EdgeType.COL_PATH))
    return graph


def lookup_payload():
    return {
        "planner_version": "typed_derivation_planner_v1",
        "query_semantics": {
            "operation_family": "LOOKUP",
            "answer_domain": "SCALAR",
            "projection_operator": "VALUE_PROJECTION",
        },
        "plans": [{
            "plan_id": "P0",
            "signature_id": LOOKUP_SIGNATURE,
            "operation_family": "LOOKUP",
            "semantic_result_role": "VALUE",
            "answer_domain": "SCALAR",
            "projection_operator": "VALUE_PROJECTION",
            "role_bindings": {
                "TARGET_ENTITY": ["entity"],
                "TARGET_MEASURE": ["measure"],
            },
            "unresolved_semantics": [],
        }],
        "unresolved_semantics": [],
    }


def lookup_view():
    return build_proposal_blind_planner_view(
        question="What is the value for A?",
        graph=lookup_graph(),
        table_json={"texts": [["Name", "Value"], ["A", "42"]]},
        query_contract=None,
        include_table_values=False,
        legacy_query_semantics_mode="audit_only",
    )


class R2UpstreamP0ValidityTests(unittest.TestCase):
    def test_p0_1_active_view_schema_validator_and_closure_are_lookup_only(self):
        view = lookup_view()
        ontology = view["operation_ontology"]
        self.assertEqual(ontology["signature_ids"], [LOOKUP_SIGNATURE])
        self.assertEqual(ontology["operation_families"], ["LOOKUP"])

        schema = build_typed_planner_response_schema(view, require_signature_id=True)
        schema_text = json.dumps(schema, sort_keys=True)
        self.assertIn(LOOKUP_SIGNATURE, schema_text)
        self.assertNotIn("ARGMAX_ENTITY", schema_text)

        invalid = lookup_payload()
        invalid["query_semantics"] = {
            "operation_family": "ARGMAX",
            "answer_domain": "ENTITY",
            "projection_operator": "ARGMAX_ENTITY_PROJECTION",
        }
        invalid["plans"][0].update({
            "signature_id": "ARGMAX_ENTITY_SET",
            "operation_family": "ARGMAX",
            "semantic_result_role": "ENTITY",
            "answer_domain": "ENTITY",
            "projection_operator": "ARGMAX_ENTITY_PROJECTION",
            "role_bindings": {
                "AGGREGATION_SCOPE": [["entity"]],
                "TARGET_MEASURE": ["measure"],
            },
        })
        validation = validate_typed_planner_output(
            json.dumps(invalid), view, require_signature_id=True
        )
        self.assertFalse(validation.ok)

        closure = build_plan_closure(
            invalid,
            lookup_graph(),
            allowed_signature_ids=(LOOKUP_SIGNATURE,),
        )
        self.assertFalse(closure.executable_derivations)
        self.assertEqual(closure.assignments[0].outcome, ClosureOutcome.STRUCTURALLY_INVALID)

    def test_p0_2_runtime_row_physically_excludes_operation_and_answer_annotations(self):
        runtime = _frozen_input({
            "id": "s1",
            "table_id": "t1",
            "table_source": "fixture",
            "question": "What is A?",
            "dataset": "wrong-source-value",
            "aggregation": ["none"],
            "answer": ["42"],
            "answer_formulas": ["=B2"],
            "linked_cells": {"answer": [1, 1]},
            "reference_cells_map": {"B2": "(1, 1)"},
        })
        self.assertEqual(
            runtime,
            {
                "id": "s1",
                "table_id": "t1",
                "table_source": "fixture",
                "question": "What is A?",
                "dataset": "hitab",
            },
        )

    def test_p0_3_generated_lookup_schema_is_compiler_isomorphic(self):
        view = lookup_view()
        view["operation_ontology"] = {
            "operation_families": ["LOOKUP"],
            "signature_ids": [LOOKUP_SIGNATURE],
            "signature_variants": {
                LOOKUP_SIGNATURE: view["operation_ontology"]["signature_variants"][LOOKUP_SIGNATURE]
            },
            "projection_operators": ["VALUE_PROJECTION"],
            "answer_domains": ["SCALAR"],
        }
        payload = lookup_payload()
        schema = build_typed_planner_response_schema(view, require_signature_id=True)
        jsonschema.validate(payload, schema)

        validation = validate_typed_planner_output(
            json.dumps(payload), view, require_signature_id=True
        )
        self.assertTrue(validation.ok, validation.errors)
        closure = build_plan_closure(validation.normalized_payload, lookup_graph())
        self.assertEqual(closure.outcome_counts["UNIQUE_EXECUTABLE"], 1)
        self.assertTrue(all(
            count == 0
            for outcome, count in closure.outcome_counts.items()
            if outcome != "UNIQUE_EXECUTABLE"
        ))
        self.assertEqual(len(closure.executable_derivations), 1)

    def test_p0_4_empty_b0_fails_closed_before_planner_or_cera(self):
        final_answer, source, _ = pipeline_arbitrate()(
            "",
            ExecutorResult("42", OperationType.LOOKUP_CELL, 1),
            0.0,
        )
        self.assertEqual((final_answer, source), ("", "B0_INVALID"))

        result = run_causal_epistemic_repair(
            question="What is A?",
            original_answer="   ",
            cert_info={},
            graph=lookup_graph(),
            table_json={"texts": [["Name", "Value"], ["A", "42"]]},
            all_exec_candidates=[],
            generator=None,
            args=None,
            result_context={},
        )
        self.assertEqual(result.reject_reason, "B0_INVALID")
        self.assertFalse(result.packet_built)
        self.assertFalse(result.llm_called)
        self.assertFalse(result.metadata.get("cera_planner_called", False))

    def test_p0_5_empty_registry_is_incomplete_and_nonconstructible(self):
        contrast = build_compact_behavioral_contrast_v3(
            derivations=[],
            behavior_classes=[],
            basis=[],
            original_answer="42",
            query_semantics={},
        )
        self.assertFalse(contrast.contrast_registry_complete)
        self.assertFalse(contrast.contrast_constructible)
        self.assertFalse(contrast.repair_eligible)
        self.assertIn("registry_incomplete", contrast.unknowns)


if __name__ == "__main__":
    unittest.main()
