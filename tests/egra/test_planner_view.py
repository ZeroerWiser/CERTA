import json
import unittest

from graph_builder import EdgeType, GraphEdge, GraphNode, HCEG, NodeType

from certa.egra.planner_view import build_role_aligned_planner_view
from certa.planner.schema_view import build_proposal_blind_planner_view
from certa.planner.typed_planner import (
    build_typed_derivation_planner_prompt,
    build_typed_planner_response_schema,
    planner_reference_domain,
    validate_typed_planner_output,
)


CORE_SIGNATURES = (
    "LOOKUP_VALUE_SCALAR",
    "LOOKUP_VALUE_ENTITY",
    "COUNT_SCALAR",
    "DIFF_SCALAR",
    "RATIO_SCALAR",
    "ARGMAX_ENTITY",
    "ARGMAX_ENTITY_SET",
    "ARGMIN_ENTITY",
    "ARGMIN_ENTITY_SET",
)


def graph_and_table():
    graph = HCEG()
    graph.add_node(GraphNode("measure", NodeType.HEADER, row=0, col=1, text="Value"))
    graph.add_node(GraphNode("entity", NodeType.HEADER, row=1, col=0, text="A"))
    graph.add_node(GraphNode("cell", NodeType.CELL, row=1, col=1, text="42", numeric_value=42.0))
    graph.add_edge(GraphEdge("cell", "entity", EdgeType.ROW_PATH))
    graph.add_edge(GraphEdge("cell", "measure", EdgeType.COL_PATH))
    return graph, {"texts": [["Name", "Value"], ["A", "42"]]}


def scalar_lookup_contract():
    return {
        "schema_version": "certa_egra_query_contract_v1",
        "supported_by_core_signatures": True,
        "answer_domain": "SCALAR",
        "intent_family": "DIRECT_READ",
        "signature_candidates": ["LOOKUP_VALUE_SCALAR"],
        "projection_candidates": ["VALUE_PROJECTION"],
        "cardinality": "SINGLE",
        "rank_direction": "NONE",
        "rank_k": None,
        "requires_time_scope": False,
        "requires_unit_consistency": False,
        "unknowns": [],
    }


def argmax_plan_with_excluded_scalar_query():
    return {
        "planner_version": "typed_derivation_planner_v1",
        "query_semantics": {
            "operation_family": "ARGMAX",
            "answer_domain": "SCALAR",
            "projection_operator": "VALUE_PROJECTION",
        },
        "plans": [{
            "plan_id": "P0",
            "signature_id": "ARGMAX_ENTITY",
            "operation_family": "ARGMAX",
            "semantic_result_role": "ENTITY",
            "answer_domain": "ENTITY",
            "projection_operator": "ROW_ENTITY_PROJECTION",
            "role_bindings": {
                "AGGREGATION_SCOPE": [["entity"]],
                "TARGET_MEASURE": ["measure"],
            },
            "unresolved_semantics": [],
        }],
        "unresolved_semantics": [],
    }


class PlannerViewTests(unittest.TestCase):
    def test_shared_prompt_schema_and_validator_use_exact_view_signatures(self):
        graph, table = graph_and_table()
        view = build_proposal_blind_planner_view(
            question="Which entity is largest?",
            graph=graph,
            table_json=table,
            query_contract=None,
            include_table_values=False,
            legacy_query_semantics_mode="audit_only",
            allowed_signature_ids=CORE_SIGNATURES,
        )
        prompt = build_typed_derivation_planner_prompt(view)
        schema_text = json.dumps(build_typed_planner_response_schema(view, require_signature_id=True))
        for signature_id in CORE_SIGNATURES:
            self.assertIn(signature_id, prompt)
            self.assertIn(signature_id, schema_text)
        for excluded in (
            "LOOKUP_VALUE_BOOLEAN",
            "SUM_SCALAR",
            "AVERAGE_SCALAR",
            "ARGMAX_VALUE",
            "ARGMIN_VALUE",
            "PAIR_COMPARE_BOOLEAN",
        ):
            self.assertNotIn(excluded, prompt)
            self.assertNotIn(excluded, schema_text)

        validation = validate_typed_planner_output(
            argmax_plan_with_excluded_scalar_query(),
            view,
            require_signature_id=True,
        )
        self.assertFalse(validation.ok)
        self.assertIn(
            "query_projection_domain_tuple_not_declared:ARGMAX:VALUE_PROJECTION:SCALAR",
            validation.errors,
        )

    def test_explicit_empty_signature_set_is_fail_closed(self):
        graph, table = graph_and_table()
        view = build_proposal_blind_planner_view(
            question="Unsupported question",
            graph=graph,
            table_json=table,
            query_contract=None,
            include_table_values=False,
            legacy_query_semantics_mode="audit_only",
            allowed_signature_ids=(),
        )
        self.assertEqual(view["operation_ontology"]["signature_ids"], [])
        prompt = build_typed_derivation_planner_prompt(view)
        for signature_id in CORE_SIGNATURES:
            self.assertNotIn(signature_id, prompt)
        schema = build_typed_planner_response_schema(view, require_signature_id=True)
        self.assertEqual(schema["properties"]["query_semantics"]["anyOf"], [])
        self.assertEqual(schema["properties"]["plans"]["items"]["anyOf"], [])

    def test_c1_is_flat_and_c2_only_narrows_existing_header_references(self):
        graph, table = graph_and_table()
        contract = scalar_lookup_contract()
        c1 = build_role_aligned_planner_view(
            question="What is the value for A?",
            graph=graph,
            table_json=table,
            contract=contract,
        )
        self.assertTrue(c1.eligible, c1.reason)
        self.assertNotIn("structural_cards", c1.view)
        self.assertEqual(set(planner_reference_domain(c1.view)), {"entity", "measure"})
        self.assertEqual(
            c1.view["operation_ontology"]["signature_ids"],
            ["LOOKUP_VALUE_SCALAR"],
        )

        selected_card = {
            "card_id": "R0",
            "unit_kind": "ROW_PATH",
            "human_readable_text": "A",
            "header_node_ids": ["entity"],
        }
        c2 = build_role_aligned_planner_view(
            question="What is the value for A?",
            graph=graph,
            table_json=table,
            contract=contract,
            reference_node_ids=["entity"],
            selected_cards=[selected_card],
        )
        self.assertTrue(c2.eligible, c2.reason)
        self.assertEqual(planner_reference_domain(c2.view), ("entity",))
        self.assertEqual(c2.view["structural_cards"], [selected_card])
        self.assertNotIn("R0", planner_reference_domain(c2.view))

    def test_invalid_or_unsupported_role_contract_never_falls_back(self):
        graph, table = graph_and_table()
        unsupported = scalar_lookup_contract()
        unsupported.update({
            "supported_by_core_signatures": False,
            "answer_domain": "UNSUPPORTED",
            "intent_family": "UNSUPPORTED",
            "signature_candidates": [],
            "projection_candidates": [],
            "cardinality": "UNKNOWN",
            "rank_direction": "UNKNOWN",
        })
        result = build_role_aligned_planner_view(
            question="unsupported",
            graph=graph,
            table_json=table,
            contract=unsupported,
        )
        self.assertFalse(result.eligible)
        self.assertEqual(result.reason, "unsupported_by_core_signatures")
        self.assertEqual(result.view, {})

        invalid = scalar_lookup_contract()
        invalid["signature_candidates"] = []
        result = build_role_aligned_planner_view(
            question="invalid",
            graph=graph,
            table_json=table,
            contract=invalid,
        )
        self.assertFalse(result.eligible)
        self.assertTrue(result.reason.startswith("invalid_query_role_contract:"))
        self.assertEqual(result.view, {})


if __name__ == "__main__":
    unittest.main()
