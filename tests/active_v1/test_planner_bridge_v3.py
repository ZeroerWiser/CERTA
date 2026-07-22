import json
import unittest
from pathlib import Path
from unittest import mock

from graph_builder import EdgeType, GraphEdge, GraphNode, HCEG, NodeType

from certa.active_v1 import planner_adapter
from certa.active_v1.planner_adapter import build_arm_view
from certa.active_v1.planner_bridge_v3 import (
    build_v3_arm_view,
    close_compiled_payload,
    compile_active_planner_payload,
)
from certa.active_v1.role_contract_v3 import derive_role_v3_record
from certa.planner.typed_planner import PLANNER_VERSION
from certa.reproducibility.canonical_json import canonical_json_hash


PACKS = Path("/home/hsh/ME/Table/EMNLP2026/certa_goal_packs")
ROLE_PACK = PACKS / "CERTA_ACTIVE_V1_ROLE_V3_FINAL_METHOD_PACK"
COMPLETION_PACK = PACKS / "CERTA_ACTIVE_V1_FROZEN_ROLE_V3_FINAL_METHOD_COMPLETION_PACK"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def operation_graph():
    graph = HCEG()
    nodes = (
        ("measure_numeric", NodeType.HEADER, 0, 1, "Value", None),
        ("entity_a", NodeType.HEADER, 1, 0, "A", None),
        ("entity_b", NodeType.HEADER, 2, 0, "B", None),
        ("numeric_a", NodeType.CELL, 1, 1, "4", 4.0),
        ("numeric_b", NodeType.CELL, 2, 1, "2", 2.0),
    )
    for node_id, kind, row, col, text, number in nodes:
        graph.add_node(GraphNode(
            node_id, kind, row=row, col=col, text=text, numeric_value=number,
        ))
    for cell, entity in (("numeric_a", "entity_a"), ("numeric_b", "entity_b")):
        graph.add_edge(GraphEdge(cell, entity, EdgeType.ROW_PATH))
        graph.add_edge(GraphEdge(cell, "measure_numeric", EdgeType.COL_PATH))
    return graph


def legacy_matrix():
    row = {
        "signature_id": "COUNT_SCALAR", "registry_present": True,
        "active_compiler_fixture_pass": True, "closure_fixture_pass": True,
        "deterministic_executor_fixture_pass": True,
        "projection_fixture_pass": True,
        "serialization_roundtrip_fixture_pass": True,
        "constructor_active": True, "constructor_failure_reasons": [], "active": True,
    }
    return {"schema_version": "certa_active_v1_signature_capability_v1", "rows": [row]}


def count_payload():
    return {
        "planner_version": PLANNER_VERSION,
        "query_semantics": {"operation_family": "COUNT", "answer_domain": "SCALAR",
                            "projection_operator": "SCALAR_RESULT_PROJECTION"},
        "plans": [{"plan_id": "P0", "signature_id": "COUNT_SCALAR",
                   "operation_family": "COUNT", "semantic_result_role": "CARDINALITY",
                   "answer_domain": "SCALAR", "projection_operator": "SCALAR_RESULT_PROJECTION",
                   "role_bindings": {"AGGREGATION_SCOPE": [["entity_a"], ["entity_b"]],
                                     "TARGET_MEASURE": ["measure_numeric"]},
                   "unresolved_semantics": []}],
        "unresolved_semantics": [],
    }


class PlannerBridgeV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = load(ROLE_PACK / "ROLE_V3_OUTPUT_SCHEMA.json")
        cls.registry = load(ROLE_PACK / "ROLE_V3_CANONICAL_REGISTRY.json")
        cls.matrix_path = COMPLETION_PACK / "fixtures/CONSTRUCTOR_CAPABILITY_MATRIX.example.json"

    def setUp(self):
        self.graph = operation_graph()
        self.table = {"texts": [["Entity", "Value"], ["A", "4"], ["B", "2"]],
                      "top_header_rows_num": 1, "left_header_columns_num": 1}
        self.count = derive_role_v3_record(
            {"schema_version": "certa_active_role_contract_v3", "role_id": "COUNT_SCALAR"},
            self.schema, self.registry,
        )

    def matrix(self):
        return load(self.matrix_path)

    def bridge(self, arm, role=None, retrieval=None, matrix=None):
        artifacts = {} if arm == "C0_SCHEMA_ONLY" else {
            "output_schema": self.schema, "canonical_registry": self.registry,
        }
        return build_v3_arm_view(
            arm, "How many?", self.graph, self.table, role, retrieval,
            matrix or self.matrix(), **artifacts,
        )

    def test_valid_pack_matrix_drives_c0_without_legacy_field_names(self):
        built = self.bridge("C0_SCHEMA_ONLY")
        ids = built.view["operation_ontology"]["signature_ids"]
        self.assertEqual(len(ids), 12)
        self.assertIn("COUNT_SCALAR", ids)

    def test_corrupted_derived_semantics_are_not_role_authority(self):
        corruptions = {"operation_family": "SUM", "projection": "VALUE_PROJECTION",
                       "answer_role": "ENTITY", "cardinality": "MULTIPLE"}
        for field_name, value in corruptions.items():
            with self.subTest(field_name=field_name), self.assertRaisesRegex(
                ValueError, "role_v3_canonical_record_mismatch",
            ):
                self.bridge("C1_ROLE_ONLY", dict(self.count, **{field_name: value}))

    def test_current_v2_boundary_rejects_the_frozen_v3_record(self):
        with self.assertRaisesRegex(ValueError, "role_validation_record_required"):
            build_arm_view(
                "C1_ROLE_ONLY", "How many?", self.graph, self.table,
                self.count, None, legacy_matrix(),
            )

    def test_frozen_v3_record_traverses_the_active_planner_boundary(self):
        built = self.bridge("C1_ROLE_ONLY", self.count)
        self.assertEqual(built.view["operation_ontology"]["signature_ids"], ["COUNT_SCALAR"])
        self.assertNotIn("table_values", built.view)
        self.assertEqual(built.view["query_semantics"], {
            "answer_domain": "SCALAR", "allowed_answer_domains": ["SCALAR"],
            "allowed_projection_operators": ["SCALAR_RESULT_PROJECTION"],
            "candidate_independent_operation_hypotheses": ["COUNT"],
            "unit_or_scale_constraints": [],
        })

    def test_c0_uses_all_active_signatures_while_c1_never_broadens(self):
        c0, c1 = self.bridge("C0_SCHEMA_ONLY"), self.bridge("C1_ROLE_ONLY", self.count)
        self.assertEqual(len(c0.view["operation_ontology"]["signature_ids"]), 12)
        self.assertNotIn("query_semantics", c0.view)
        self.assertEqual(c1.view["operation_ontology"]["signature_ids"], ["COUNT_SCALAR"])

    def test_c2_requires_matching_hash_and_complete_deduplicated_references(self):
        c1 = self.bridge("C1_ROLE_ONLY", self.count)
        refs = ["entity_a", "entity_a", "entity_b", "measure_numeric"]
        c2 = self.bridge("C2_ROLE_RETRIEVAL", self.count, {
            "role_record_sha256": c1.role_record_sha256, "reference_node_ids": refs,
        })
        self.assertEqual(c1.role_record_sha256, canonical_json_hash(self.count))
        self.assertEqual(c2.role_record_sha256, c1.role_record_sha256)
        self.assertEqual(c2.retrieval_reference_node_ids,
                         ("entity_a", "entity_b", "measure_numeric"))
        self.assertEqual({row["node_id"] for row in c2.view["schema_nodes"]},
                         {"entity_a", "entity_b", "measure_numeric"})
        self.assertNotIn("table_values", c2.view)
        invalid = (
            (None, "c2_retrieval_result_required"),
            ({"role_record_sha256": "0" * 64, "reference_node_ids": ["entity_a"]},
             "c2_role_record_sha256_mismatch"),
            ({"role_record_sha256": c1.role_record_sha256, "reference_node_ids": []},
             "c2_retrieval_reference_ids_empty"),
            ({"role_record_sha256": c1.role_record_sha256, "reference_node_ids": ["missing"]},
             "retrieval_reference_outside_schema:missing"),
        )
        for retrieval, error in invalid:
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                self.bridge("C2_ROLE_RETRIEVAL", self.count, retrieval)

    def test_unsupported_and_inactive_roles_fail_without_fallback(self):
        unsupported = derive_role_v3_record(
            {"schema_version": "certa_active_role_contract_v3", "role_id": "UNSUPPORTED"},
            self.schema, self.registry,
        )
        with self.assertRaisesRegex(ValueError, "unsupported_role_has_no_active_planner_view"):
            self.bridge("C1_ROLE_ONLY", unsupported)
        inactive = derive_role_v3_record(
            {"schema_version": "certa_active_role_contract_v3", "role_id": "SUM_SCALAR"},
            self.schema, self.registry,
        )
        matrix = self.matrix()
        row = next(item for item in matrix["rows"] if item["role_id"] == "SUM_SCALAR")
        row.update(negative_fixture_pass=False, constructor_active=False,
                   failure_reasons=["negative_fixture_failed"])
        with self.assertRaisesRegex(ValueError, "inactive_role_signature:SUM_SCALAR"):
            self.bridge("C1_ROLE_ONLY", inactive, matrix=matrix)

    def test_compiler_and_closure_delegate_through_deterministic_projection(self):
        matrix, built = self.matrix(), self.bridge("C1_ROLE_ONLY", self.count)
        with mock.patch.object(planner_adapter, "compile_active_planner_payload",
                               wraps=planner_adapter.compile_active_planner_payload) as delegate:
            compiled = compile_active_planner_payload(count_payload(), built.view, matrix)
        self.assertTrue(compiled.ok, compiled.errors)
        projected = delegate.call_args.args[2]
        self.assertEqual(projected["schema_version"], "certa_active_v1_signature_capability_v1")
        self.assertTrue(next(row for row in projected["rows"]
                             if row["signature_id"] == "COUNT_SCALAR")["active"])
        with mock.patch.object(planner_adapter, "close_compiled_payload",
                               wraps=planner_adapter.close_compiled_payload) as delegate:
            closure = close_compiled_payload(compiled, self.graph, matrix)
        self.assertEqual(closure.executable_derivations[0].projected_answer, "2")
        self.assertEqual(delegate.call_args.args[2]["schema_version"],
                         "certa_active_v1_signature_capability_v1")


if __name__ == "__main__":
    unittest.main()
