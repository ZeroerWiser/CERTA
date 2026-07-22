import unittest

from graph_builder import EdgeType, GraphEdge, GraphNode, HCEG, NodeType

from certa.active_v1.planner_adapter import (
    active_signature_ids,
    build_arm_view,
    close_compiled_payload,
    compile_active_planner_payload,
)
from certa.active_v1.role_contract import validate_role_contract
from certa.operations.contracts import OPERATION_SIGNATURES
from certa.planner.typed_planner import PLANNER_VERSION
from tools.certa_active_v1 import (
    build_signature_capability_matrix,
    build_signature_capability_schema,
)

import jsonschema


def operation_graph():
    graph = HCEG()
    for node in (
        GraphNode("measure_numeric", NodeType.HEADER, row=0, col=1, text="Value"),
        GraphNode("entity_a", NodeType.HEADER, row=1, col=0, text="A"),
        GraphNode("entity_b", NodeType.HEADER, row=2, col=0, text="B"),
        GraphNode("numeric_a", NodeType.CELL, row=1, col=1, text="4", numeric_value=4.0),
        GraphNode("numeric_b", NodeType.CELL, row=2, col=1, text="2", numeric_value=2.0),
    ):
        graph.add_node(node)
    for cell, entity in (("numeric_a", "entity_a"), ("numeric_b", "entity_b")):
        graph.add_edge(GraphEdge(cell, entity, EdgeType.ROW_PATH))
        graph.add_edge(GraphEdge(cell, "measure_numeric", EdgeType.COL_PATH))
    return graph


def active_matrix(*signature_ids):
    return {
        "schema_version": "certa_active_v1_signature_capability_v1",
        "rows": [
            {
                "signature_id": signature_id,
                "registry_present": True,
                "active_compiler_fixture_pass": True,
                "closure_fixture_pass": True,
                "deterministic_executor_fixture_pass": True,
                "projection_fixture_pass": True,
                "serialization_roundtrip_fixture_pass": True,
                "constructor_active": True,
                "constructor_failure_reasons": [],
                "active": True,
            }
            for signature_id in signature_ids
        ],
    }


def count_role():
    return validate_role_contract({
        "schema_version": "certa_active_role_contract_v2",
        "supported": True,
        "intent": "COUNT",
        "answer_role": "SCALAR",
        "projection": "SCALAR_RESULT_PROJECTION",
        "signature": "COUNT_SCALAR",
        "cardinality": "SINGLE",
        "requires_time_scope": False,
        "requires_unit_consistency": False,
    }, ("COUNT_SCALAR",))


def count_payload():
    return {
        "planner_version": PLANNER_VERSION,
        "query_semantics": {
            "operation_family": "COUNT",
            "answer_domain": "SCALAR",
            "projection_operator": "SCALAR_RESULT_PROJECTION",
        },
        "plans": [{
            "plan_id": "P0",
            "signature_id": "COUNT_SCALAR",
            "operation_family": "COUNT",
            "semantic_result_role": "CARDINALITY",
            "answer_domain": "SCALAR",
            "projection_operator": "SCALAR_RESULT_PROJECTION",
            "role_bindings": {
                "AGGREGATION_SCOPE": [["entity_a"], ["entity_b"]],
                "TARGET_MEASURE": ["measure_numeric"],
            },
            "unresolved_semantics": [],
        }],
        "unresolved_semantics": [],
    }


class ActiveCapabilityPlannerTests(unittest.TestCase):
    def setUp(self):
        self.graph = operation_graph()
        self.table = {
            "texts": [["Entity", "Value"], ["A", "4"], ["B", "2"]],
            "top_header_rows_num": 1,
            "left_header_columns_num": 1,
        }
        self.matrix = active_matrix("COUNT_SCALAR")

    def test_capability_matrix_requires_all_fixture_dimensions(self):
        self.assertEqual(active_signature_ids(self.matrix), ("COUNT_SCALAR",))
        broken = active_matrix("COUNT_SCALAR")
        broken["rows"][0]["projection_fixture_pass"] = False
        with self.assertRaisesRegex(ValueError, "capability_activation_equation_mismatch"):
            active_signature_ids(broken)

    def test_c1_and_c2_bind_one_role_signature_without_fallback(self):
        c1 = build_arm_view(
            "C1_ROLE_ONLY", "How many?", self.graph, self.table,
            count_role(), None, self.matrix,
        )
        self.assertEqual(c1.view["operation_ontology"]["signature_ids"], ["COUNT_SCALAR"])
        c2 = build_arm_view(
            "C2_ROLE_RETRIEVAL", "How many?", self.graph, self.table,
            count_role(), {"reference_node_ids": ["entity_a", "entity_b", "measure_numeric"]}, self.matrix,
        )
        self.assertEqual({item["node_id"] for item in c2.view["schema_nodes"]}, {"entity_a", "entity_b", "measure_numeric"})
        with self.assertRaisesRegex(ValueError, "retrieval_reference_outside_schema"):
            build_arm_view(
                "C2_ROLE_RETRIEVAL", "How many?", self.graph, self.table,
                count_role(), {"reference_node_ids": ["missing"]}, self.matrix,
            )

    def test_validated_payload_compiles_roundtrips_and_closes(self):
        built = build_arm_view(
            "C1_ROLE_ONLY", "How many?", self.graph, self.table,
            count_role(), None, self.matrix,
        )
        compiled = compile_active_planner_payload(count_payload(), built.view, self.matrix)
        self.assertTrue(compiled.ok, compiled.errors)
        closure = close_compiled_payload(compiled, self.graph, self.matrix)
        self.assertEqual(len(closure.executable_derivations), 1)
        derivation = closure.executable_derivations[0]
        self.assertEqual(derivation.typed_signature, "COUNT_SCALAR")
        self.assertEqual(derivation.projected_answer, "2")
        self.assertTrue(derivation.provenance_complete)

    def test_inactive_signature_and_invalid_payload_fail_closed(self):
        built = build_arm_view(
            "C1_ROLE_ONLY", "How many?", self.graph, self.table,
            count_role(), None, self.matrix,
        )
        bad = count_payload()
        bad["plans"][0]["signature_id"] = "SUM_SCALAR"
        bad["plans"][0]["operation_family"] = OPERATION_SIGNATURES["SUM_SCALAR"].operation_family
        result = compile_active_planner_payload(bad, built.view, self.matrix)
        self.assertFalse(result.ok)
        self.assertTrue(result.errors)

    def test_executor_backed_capability_matrix_activates_only_design_signatures(self):
        matrix = build_signature_capability_matrix()
        jsonschema.validate(matrix, build_signature_capability_schema())
        expected = {
            "LOOKUP_VALUE_SCALAR", "LOOKUP_VALUE_ENTITY", "COUNT_SCALAR", "SUM_SCALAR",
            "AVERAGE_SCALAR", "DIFF_SCALAR", "RATIO_SCALAR", "ARGMAX_ENTITY",
            "ARGMAX_ENTITY_SET", "ARGMIN_ENTITY", "ARGMIN_ENTITY_SET", "PAIR_COMPARE_BOOLEAN",
        }
        self.assertEqual({row["signature_id"] for row in matrix["rows"]}, expected)
        for row in matrix["rows"]:
            self.assertTrue(row["constructor_active"], row)
            self.assertTrue(row["negative_fixture_pass"], row)
            self.assertTrue(row["canonical_program_id"], row)
            self.assertTrue(row["projected_answer_sha256"], row)


if __name__ == "__main__":
    unittest.main()
