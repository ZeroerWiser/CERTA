import json
import unittest
from types import SimpleNamespace

import numpy as np

from graph_builder import EdgeType, GraphEdge, GraphNode, HCEG, NodeType

from certa.egra.query_role_contract import build_query_role_response_schema
from certa.repair.evidence_packet import CERACommitResult
from certa.repair.causal_epistemic_agent import (
    _run_typed_derivation_planner,
    run_egra_constructor_shadow,
)
from certa.reproducibility.canonical_json import canonical_json_hash


def graph_and_table():
    graph = HCEG()
    graph.add_node(GraphNode("measure", NodeType.HEADER, row=0, col=1, text="Population"))
    graph.add_node(GraphNode("time", NodeType.HEADER, row=0, col=2, text="2020"))
    graph.add_node(GraphNode("entity", NodeType.HEADER, row=1, col=0, text="North"))
    graph.add_node(GraphNode("value_a", NodeType.CELL, row=1, col=1, text="42", numeric_value=42.0))
    graph.add_node(GraphNode("value_b", NodeType.CELL, row=1, col=2, text="99", numeric_value=99.0))
    graph.add_edge(GraphEdge("value_a", "entity", EdgeType.ROW_PATH))
    graph.add_edge(GraphEdge("value_a", "measure", EdgeType.COL_PATH))
    graph.add_edge(GraphEdge("value_b", "entity", EdgeType.ROW_PATH))
    graph.add_edge(GraphEdge("value_b", "time", EdgeType.COL_PATH))
    return graph, {
        "texts": [["Region", "Population", "2020"], ["North", "42", "99"]],
        "top_header_rows_num": 1,
        "left_header_columns_num": 1,
    }


def role_payload(*, supported=True):
    if not supported:
        return {
            "schema_version": "certa_egra_query_contract_v1",
            "supported_by_core_signatures": False,
            "answer_domain": "UNSUPPORTED",
            "intent_family": "UNSUPPORTED",
            "signature_candidates": [],
            "projection_candidates": [],
            "cardinality": "UNKNOWN",
            "rank_direction": "UNKNOWN",
            "rank_k": None,
            "requires_time_scope": False,
            "requires_unit_consistency": False,
            "unknowns": ["operation"],
        }
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


def lookup_plan():
    return {
        "planner_version": "typed_derivation_planner_v1",
        "query_semantics": {
            "operation_family": "LOOKUP",
            "answer_domain": "SCALAR",
            "projection_operator": "VALUE_PROJECTION",
        },
        "plans": [{
            "plan_id": "P0",
            "signature_id": "LOOKUP_VALUE_SCALAR",
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


class FakeEncoder:
    def encode(self, texts):
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)


class ArmGenerator:
    def __init__(self, *, supported=True):
        self.supported = supported
        self.calls = []
        self.model = "Qwen3-8B"
        self.api_base_url = "http://127.0.0.1:30338/v1"
        self.backend_name = "vllm_chat"
        self.chat_template_kwargs = {"enable_thinking": False}
        self.cache_mode = "readwrite"

    def generate_json_schema(self, prompt, **kwargs):
        self.calls.append(kwargs["schema_name"])
        payload = (
            role_payload(supported=self.supported)
            if kwargs["schema_name"] == "certa_egra_query_contract_v1"
            else lookup_plan()
        )
        return {
            "text": json.dumps(payload),
            "structured_output_requested": True,
            "structured_output_mechanism": "response_format.type=json_schema",
            "structured_output_schema_hash": canonical_json_hash(kwargs["response_schema"]),
            "structured_output_fallback_used": False,
            "input_token_count": 10,
            "generated_token_count": 10,
            "generation_seconds": 0.01,
            "api_model": self.model,
            "api_base_url": self.api_base_url,
            "generator_backend": self.backend_name,
            "api_cache_hit": False,
            "api_cache_mode": self.cache_mode,
            "chat_template_kwargs": self.chat_template_kwargs,
        }


def args(arm):
    return SimpleNamespace(
        cera_enable_typed_planner=True,
        cera_planner_boundary="proposal_blind_schema_only",
        cera_stage="E71",
        cera_shadow_only=True,
        cera_commit_approved_repair=False,
        cera_planner_contract="rcpc_signature_v2",
        cera_planner_signature_allowlist="",
        cera_planner_legacy_query_semantics_mode="active",
        cera_planner_max_tokens=512,
        cera_planner_temperature=0.0,
        cera_stepwise_trace=False,
        top_p=1.0,
        certa_egra_arm=arm,
        certa_egra_embedding_file_tree_sha256="f" * 64,
        _certa_egra_encoder=FakeEncoder(),
    )


class RuntimeIntegrationTests(unittest.TestCase):
    def test_constructor_shadow_stops_after_planner_closure_and_partitions_b0(self):
        graph, table = graph_and_table()
        generator = ArmGenerator()
        metadata = run_egra_constructor_shadow(
            question="What is the population of North?",
            original_answer="42",
            graph=graph,
            table_json=table,
            generator=generator,
            args=args("C2_EGRA"),
        )
        self.assertTrue(metadata["certa_egra_construction_only"])
        self.assertFalse(metadata["certa_egra_intervention_generated"])
        self.assertFalse(metadata["certa_egra_decision_executed"])
        self.assertEqual(metadata["cera_round9_partition_original_count"], 1)
        self.assertEqual(metadata["cera_round9_partition_alternative_count"], 0)

    def test_egra_blind_metadata_survives_prediction_projection(self):
        metadata = {
            "certa_egra_arm": "C2_EGRA",
            "certa_egra_role_contract_called": True,
            "certa_egra_role_contract_valid": True,
            "certa_egra_role_contract": role_payload(),
            "certa_egra_role_contract_audit": {"calls": 1},
            "certa_egra_retrieval": {"selected_card_ids": ["R0"]},
            "certa_egra_index_sha256": "a" * 64,
            "cera_round9_partition_original_count": 1,
            "cera_round9_partition_alternative_count": 2,
            "cera_round9_partition_disjoint": True,
            "cera_round9_partition_exhaustive": True,
        }
        fields = CERACommitResult(metadata=metadata).to_prediction_fields()
        for key, value in metadata.items():
            self.assertEqual(fields[key], value)

    def test_c0_remains_flat_and_does_not_call_role_contract(self):
        graph, table = graph_and_table()
        generator = ArmGenerator()
        derivations, metadata, closure = _run_typed_derivation_planner(
            question="What is the population of North?",
            graph=graph,
            table_json=table,
            pre_contract=None,
            generator=generator,
            args=args("C0_FLAT_SCHEMA_CURRENT"),
            original_answer="42",
        )
        self.assertEqual(generator.calls, ["certa_typed_planner_signature_v2"])
        self.assertFalse(metadata["certa_egra_role_contract_called"])
        self.assertEqual(metadata["certa_egra_arm"], "C0_FLAT_SCHEMA_CURRENT")
        self.assertEqual(metadata["cera_planner_view_version"], "certa_planner_boundary_view_v1")
        self.assertEqual(len(derivations), 1)
        self.assertIsNotNone(closure)

    def test_c1_calls_role_then_uses_flat_role_aligned_view(self):
        graph, table = graph_and_table()
        generator = ArmGenerator()
        derivations, metadata, _ = _run_typed_derivation_planner(
            question="What is the population of North?",
            graph=graph,
            table_json=table,
            pre_contract=None,
            generator=generator,
            args=args("C1_ROLE_ALIGNED_FLAT"),
            original_answer="42",
        )
        self.assertEqual(generator.calls, [
            "certa_egra_query_contract_v1",
            "certa_typed_planner_signature_v2",
        ])
        self.assertTrue(metadata["certa_egra_role_contract_valid"])
        self.assertEqual(metadata["cera_planner_view_version"], "certa_egra_role_aligned_flat_v1")
        self.assertNotIn("certa_egra_retrieval", metadata)
        self.assertEqual(len(derivations), 1)

    def test_c1_reuses_frozen_role_contract_and_calls_only_planner(self):
        graph, table = graph_and_table()
        generator = ArmGenerator()
        frozen_args = args("C1_ROLE_ALIGNED_FLAT")
        question = "What is the population of North?"
        frozen_args._certa_egra_frozen_role_by_question_hash = {
            canonical_json_hash({"question": question}): {
                "contract": role_payload(),
                "audit": {"calls": 1, "request_sha256": "a" * 64},
            }
        }
        derivations, metadata, _ = _run_typed_derivation_planner(
            question=question,
            graph=graph,
            table_json=table,
            pre_contract=None,
            generator=generator,
            args=frozen_args,
            original_answer="42",
        )
        self.assertEqual(generator.calls, ["certa_typed_planner_signature_v2"])
        self.assertTrue(metadata["certa_egra_role_contract_reused"])
        self.assertEqual(metadata["certa_egra_role_contract_audit"]["calls"], 0)
        self.assertEqual(metadata["certa_egra_role_contract_audit"]["frozen_source_calls"], 1)
        self.assertEqual(len(derivations), 1)

    def test_c2_retrieves_cards_and_narrows_existing_reference_ids(self):
        graph, table = graph_and_table()
        generator = ArmGenerator()
        derivations, metadata, _ = _run_typed_derivation_planner(
            question="What is the population of North?",
            graph=graph,
            table_json=table,
            pre_contract=None,
            generator=generator,
            args=args("C2_EGRA"),
            original_answer="42",
        )
        self.assertEqual(generator.calls, [
            "certa_egra_query_contract_v1",
            "certa_typed_planner_signature_v2",
        ])
        self.assertEqual(metadata["cera_planner_view_version"], "certa_egra_retrieved_structural_view_v1")
        self.assertTrue(metadata["certa_egra_retrieval"]["selected_card_ids"])
        card_texts = [
            card["human_readable_text"]
            for card in metadata["certa_egra_structural_cards"]
        ]
        self.assertNotIn("42", card_texts)
        self.assertNotIn("99", card_texts)
        self.assertEqual(len(derivations), 1)

    def test_unsupported_role_stops_before_planner_without_legacy_fallback(self):
        graph, table = graph_and_table()
        generator = ArmGenerator(supported=False)
        derivations, metadata, closure = _run_typed_derivation_planner(
            question="List the two most similar regions",
            graph=graph,
            table_json=table,
            pre_contract=None,
            generator=generator,
            args=args("C2_EGRA"),
            original_answer="North",
        )
        self.assertEqual(generator.calls, ["certa_egra_query_contract_v1"])
        self.assertEqual(derivations, [])
        self.assertIsNone(closure)
        self.assertEqual(metadata["cera_planner_skipped_reason"], "unsupported_by_core_signatures")

    def test_constructor_arm_forbids_decision_authority_before_gate_c(self):
        graph, table = graph_and_table()
        for changed in (
            {"cera_stage": "E72"},
            {"cera_shadow_only": False},
            {"cera_commit_approved_repair": True},
        ):
            runtime_args = args("C2_EGRA")
            for key, value in changed.items():
                setattr(runtime_args, key, value)
            generator = ArmGenerator()
            with self.assertRaisesRegex(ValueError, "certa_egra_requires_e71_shadow_no_commit"):
                _run_typed_derivation_planner(
                    question="What is the population of North?",
                    graph=graph,
                    table_json=table,
                    pre_contract=None,
                    generator=generator,
                    args=runtime_args,
                    original_answer="42",
                )
            self.assertEqual(generator.calls, [])


class PackSchemaSanityTests(unittest.TestCase):
    def test_test_fixture_still_matches_frozen_role_schema(self):
        self.assertEqual(
            canonical_json_hash(build_query_role_response_schema()),
            "f58e8e84edb768689e406f8012c39813f79ad153e8587e8cd3341a031c1559d7",
        )


if __name__ == "__main__":
    unittest.main()
