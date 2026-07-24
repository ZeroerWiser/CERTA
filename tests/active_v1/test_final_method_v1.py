import copy
import json
import unittest
from dataclasses import replace
from pathlib import Path

import jsonschema
from graph_builder import build_hceg

from certa.active_v1.answer_authority import active_answer_hash
from certa.active_v1.final_method_v1 import (
    build_registry_support_state,
    build_complete_domain_c2_view,
    canonical_typed_plan_identity,
    classify_support_state,
    materialize_registry_selection,
    ProgramUnionInput,
    select_registry_policy,
    union_exact_typed_programs,
)
from certa.active_v1.artifact_authority import ArtifactContext, serialize_plan_closure_v3
from certa.active_v1.planner_bridge_v3 import build_v3_arm_view
from certa.active_v1.planner_bridge_v3 import compile_active_planner_payload
from certa.active_v1.role_contract_v3 import derive_role_v3_record
from certa.reproducibility.canonical_json import canonical_json_hash
from tests.active_v1.test_assignment_level_grounding_authority import (
    _assignment,
    _closure,
)
from tests.active_v1.test_planner_bridge_v3 import count_payload, operation_graph


ROOT = Path(__file__).resolve().parents[2]
ROLE_ROOT = Path(
    "/home/hsh/ME/Table/EMNLP2026/certa_active_v1_outputs/"
    "CERTA_ACTIVE_V1_ROLE_V3_FINAL/freeze"
)
MATRIX_PATH = Path(
    "/home/hsh/ME/Table/EMNLP2026/certa_active_v1_outputs/"
    "CERTA_ACTIVE_V1_FINAL_ASSIGNMENT_LEVEL_GROUNDING_AUTHORITY_REPLAY/"
    "freeze/CONSTRUCTOR_CAPABILITY_MATRIX.json"
)


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


class FinalMethodV1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = _json(ROLE_ROOT / "ROLE_V3_OUTPUT_SCHEMA.json")
        cls.registry = _json(ROLE_ROOT / "ROLE_V3_CANONICAL_REGISTRY.json")
        cls.matrix = _json(MATRIX_PATH)
        cls.role = derive_role_v3_record(
            {"schema_version": "certa_active_role_contract_v3", "role_id": "COUNT_SCALAR"},
            cls.schema,
            cls.registry,
        )
        cls.table = {
            "title": "Fixture",
            "texts": [["Year", "Value"], ["2020", "1"], ["2021", "2"]],
            "top_header_rows_num": 1,
            "left_header_columns_num": 1,
        }
        cls.graph = build_hceg(cls.table, "How many years are listed?")

    def test_complete_domain_c2_preserves_c1_reference_domain(self):
        c1 = build_v3_arm_view(
            "C1_ROLE_ONLY", "How many years are listed?", self.graph, self.table,
            self.role, None, self.matrix, output_schema=self.schema,
            canonical_registry=self.registry,
        )
        refs = [c1.view["schema_nodes"][0]["node_id"]]
        retrieval = {
            "role_record_sha256": canonical_json_hash(self.role),
            "reference_node_ids": refs,
        }
        complete = build_complete_domain_c2_view(
            "How many years are listed?", self.graph, self.table, self.role,
            retrieval, self.matrix, output_schema=self.schema,
            canonical_registry=self.registry,
        )
        self.assertEqual(complete.view["schema_nodes"], c1.view["schema_nodes"])
        self.assertEqual(complete.view["schema_edges"], c1.view["schema_edges"])
        self.assertEqual(
            complete.view["retrieval_advisory"]["reference_node_ids"], refs,
        )
        self.assertEqual(
            complete.view["retrieval_advisory"]["authority"],
            "ADVISORY_ONLY_NO_DOMAIN_FILTER",
        )

        legacy = build_v3_arm_view(
            "C2_ROLE_RETRIEVAL", "How many years are listed?", self.graph,
            self.table, self.role, retrieval, self.matrix,
            output_schema=self.schema, canonical_registry=self.registry,
        )
        self.assertLess(len(legacy.view["schema_nodes"]), len(c1.view["schema_nodes"]))

    def test_complete_domain_rejects_reference_outside_schema(self):
        retrieval = {
            "role_record_sha256": canonical_json_hash(self.role),
            "reference_node_ids": ["not-a-node"],
        }
        with self.assertRaisesRegex(ValueError, "retrieval_reference_outside_schema"):
            build_complete_domain_c2_view(
                "How many years are listed?", self.graph, self.table, self.role,
                retrieval, self.matrix, output_schema=self.schema,
                canonical_registry=self.registry,
            )

    def test_program_identity_excludes_only_plan_id(self):
        plan = {
            "plan_id": "P0",
            "signature_id": "COUNT_SCALAR",
            "operation_family": "COUNT",
            "semantic_result_role": "CARDINALITY",
            "answer_domain": "SCALAR",
            "projection_operator": "SCALAR_RESULT_PROJECTION",
            "role_bindings": {
                "AGGREGATION_SCOPE": [["h0"]],
                "TARGET_MEASURE": ["h1"],
            },
            "role_domains": {},
            "unresolved_semantics": [],
        }
        renamed = dict(plan, plan_id="P99")
        self.assertEqual(
            canonical_typed_plan_identity(plan),
            canonical_typed_plan_identity(renamed),
        )
        changed = copy.deepcopy(plan)
        changed["role_bindings"]["AGGREGATION_SCOPE"] = [["h2"]]
        self.assertNotEqual(
            canonical_typed_plan_identity(plan),
            canonical_typed_plan_identity(changed),
        )

    def _union_fixture(self, c1_scope, c2_scope):
        graph = operation_graph()
        table = {
            "texts": [["Entity", "Value"], ["A", "4"], ["B", "2"]],
            "top_header_rows_num": 1,
            "left_header_columns_num": 1,
        }
        c1_view = build_v3_arm_view(
            "C1_ROLE_ONLY", "How many?", graph, table, self.role, None,
            self.matrix, output_schema=self.schema,
            canonical_registry=self.registry,
        )
        retrieval = {
            "role_record_sha256": canonical_json_hash(self.role),
            "reference_node_ids": ["entity_a", "entity_b", "measure_numeric"],
        }
        c2_view = build_complete_domain_c2_view(
            "How many?", graph, table, self.role, retrieval, self.matrix,
            output_schema=self.schema, canonical_registry=self.registry,
        )
        compilations = []
        for scope, view in (
            (c1_scope, c1_view.view),
            (c2_scope, c2_view.view),
        ):
            payload = count_payload()
            payload["plans"][0]["role_bindings"]["AGGREGATION_SCOPE"] = [
                [scope]
            ]
            compilation = compile_active_planner_payload(
                payload, view, self.matrix,
            )
            self.assertTrue(compilation.ok, compilation.errors)
            compilations.append(compilation)
        return (
            ProgramUnionInput(
                "C1_ROLE_ONLY", "S1", "T1", graph.to_dict(), self.role,
                c1_view.view, compilations[0],
            ),
            ProgramUnionInput(
                "C2_ROLE_RETRIEVAL", "S1", "T1", graph.to_dict(), self.role,
                c2_view.view, compilations[1],
            ),
        ), c1_view.view

    def test_exact_union_deduplicates_same_program_and_preserves_lineage(self):
        inputs, full_view = self._union_fixture("entity_a", "entity_a")
        union = union_exact_typed_programs(
            inputs,
            full_domain_view=full_view,
            capability_matrix=self.matrix,
        )
        self.assertEqual(len(union.payload["plans"]), 1)
        self.assertEqual(union.payload["plans"][0]["plan_id"], "P0")
        self.assertEqual(
            union.lineage[0]["source_arms"],
            ["C1_ROLE_ONLY", "C2_ROLE_RETRIEVAL"],
        )

    def test_exact_union_keeps_distinct_bindings_and_is_order_invariant(self):
        inputs, full_view = self._union_fixture("entity_a", "entity_b")
        left = union_exact_typed_programs(
            inputs,
            full_domain_view=full_view,
            capability_matrix=self.matrix,
        )
        right = union_exact_typed_programs(
            tuple(reversed(inputs)),
            full_domain_view=full_view,
            capability_matrix=self.matrix,
        )
        self.assertEqual(len(left.payload["plans"]), 2)
        self.assertEqual(left.payload, right.payload)
        self.assertEqual(left.lineage, right.lineage)
        self.assertEqual(left.payload_sha256, right.payload_sha256)

    def test_exact_union_rejects_mismatched_context(self):
        inputs, full_view = self._union_fixture("entity_a", "entity_b")
        mismatched = (
            inputs[0],
            ProgramUnionInput(
                inputs[1].arm,
                inputs[1].sample_id,
                "T2",
                inputs[1].graph,
                inputs[1].role_record,
                inputs[1].planner_view,
                inputs[1].compilation,
            ),
        )
        with self.assertRaisesRegex(ValueError, "program_union_context_mismatch:table_id"):
            union_exact_typed_programs(
                mismatched,
                full_domain_view=full_view,
                capability_matrix=self.matrix,
            )

    def test_support_states_are_explicit(self):
        self.assertEqual(classify_support_state([], []).state, "NoSupport")
        self.assertEqual(classify_support_state(["a"], []).state, "OriginalOnly")
        self.assertEqual(classify_support_state([], ["b"]).state, "AlternativeOnly")
        self.assertEqual(classify_support_state(["a"], ["b"]).state, "BothSide")

    def test_union_closure_has_truthful_versioned_artifact_arm(self):
        pair = _assignment(1, executable=True, answer="42")
        role_sha = canonical_json_hash(self.role)
        bundle = serialize_plan_closure_v3(
            _closure([pair]),
            context=ArtifactContext(
                sample_id="S1",
                table_id="T1",
                arm="C1_C2_EXACT_PROGRAM_UNION",
                role_id="COUNT_SCALAR",
                role_record_sha256=role_sha,
                fixture_only=True,
            ),
            initial_answer="10",
        )
        schema = _json(ROOT / "schemas/active_v1/RAW_GROUNDING_RECORD_V3.schema.json")
        jsonschema.validate(bundle.raw_groundings[0], schema)
        self.assertEqual(
            bundle.raw_groundings[0]["arm"],
            "C1_C2_EXACT_PROGRAM_UNION",
        )

    def _authority_fixture(self, answers=("42",)):
        pairs = [_assignment(index + 1, executable=True, answer=answer)
                 for index, answer in enumerate(answers)]
        arm = "C1_ROLE_ONLY"
        closure = _closure(pairs)
        derivations = closure.executable_derivations
        role_sha = canonical_json_hash(self.role)
        bundle = serialize_plan_closure_v3(
            closure,
            context=ArtifactContext(
                sample_id="S1",
                table_id="T1",
                arm=arm,
                role_id="COUNT_SCALAR",
                role_record_sha256=role_sha,
                fixture_only=True,
            ),
            initial_answer="10",
        )
        vault = [
            {
                "sample_id": "S1",
                "table_id": "T1",
                "variant_id": "V1_C2_COMPLETE_DOMAIN",
                "arm": arm,
                "canonical_program_id": pair[0].canonical_program_id,
                "derivation_id": pair[1].derivation_id,
                "answer_hash": active_answer_hash(pair[1].projected_answer),
                "executed_answer": pair[1].projected_answer,
            }
            for pair in pairs
        ]
        state = build_registry_support_state(
            sample_id="S1",
            table_id="T1",
            variant_id="V1_C2_COMPLETE_DOMAIN",
            selected_arms=[arm],
            role_record=self.role,
            role_output_schema=self.schema,
            role_registry=self.registry,
            capability_matrix=self.matrix,
            b0_answer="10",
            executed_derivations=derivations,
            raw_groundings_v3=bundle.raw_groundings,
            raw_derivations=bundle.raw_derivations,
            registry_entries=bundle.registry_entries,
            answer_vault_records=vault,
        )
        return state, vault

    def test_registry_policy_is_joined_and_materialized_independently(self):
        state, vault = self._authority_fixture()
        self.assertTrue(state.valid, state.failure_reasons)
        self.assertEqual(state.state, "AlternativeOnly")
        selection = select_registry_policy(state)
        self.assertEqual(selection.action, "USE_ALTERNATIVE")
        self.assertFalse(hasattr(selection, "selected_answer"))
        materialized = materialize_registry_selection(
            selection, state, vault, "10",
        )
        self.assertEqual(
            (materialized.action, materialized.answer),
            ("USE_ALTERNATIVE", "42"),
        )

    def test_registry_policy_rejects_forged_or_ambiguous_vault_authority(self):
        state, vault = self._authority_fixture(("42", "43"))
        selection = select_registry_policy(state)
        self.assertEqual(selection.action, "KEEP_B0")
        self.assertIn("alternative_answer_class_not_unique", selection.failure_reasons)

        _state, valid_vault = self._authority_fixture()
        forged = [dict(valid_vault[0], executed_answer="external")]
        invalid = build_registry_support_state(
            sample_id="S1",
            table_id="T1",
            variant_id="V1_C2_COMPLETE_DOMAIN",
            selected_arms=["C1_ROLE_ONLY"],
            role_record=self.role,
            role_output_schema=self.schema,
            role_registry=self.registry,
            capability_matrix=self.matrix,
            b0_answer="10",
            executed_derivations=[
                _assignment(1, executable=True, answer="42")[1],
            ],
            raw_groundings_v3=(),
            raw_derivations=(),
            registry_entries=(),
            answer_vault_records=forged,
        )
        self.assertFalse(invalid.valid)
        self.assertEqual(select_registry_policy(invalid).action, "KEEP_B0")


if __name__ == "__main__":
    unittest.main()
