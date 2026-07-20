import json
import unittest
from pathlib import Path

import jsonschema

from certa.egra.artifacts import (
    build_constructor_sample_rows,
    freeze_b0_rows,
    freeze_role_contract_rows,
    unblind_constructor_sample_rows,
)
from tests.egra.test_query_role_contract import FakeStructuredGenerator, scalar_lookup_payload


def runtime_row():
    return {
        "dataset": "hitab",
        "id": "s1",
        "question": "What is A?",
        "table_id": "t1",
        "table_source": "fixture",
    }


class EgraArtifactTests(unittest.TestCase):
    def test_blind_constructor_projection_uses_closure_evidence_and_pack_schema(self):
        prediction = {
            "id": "s1",
            "table_id": "t1",
            "question": "What is A?",
            "certa_egra_arm": "C2_EGRA",
            "llm_answer": "42",
            "final_answer": "42",
            "certa_egra_role_contract_valid": True,
            "certa_egra_role_contract": scalar_lookup_payload(),
            "certa_egra_role_contract_audit": {"normalized_output_sha256": "a" * 64},
            "certa_egra_retrieval": {
                "selected_card_ids": ["R0", "C0", "X0"],
                "reference_node_ids": ["entity", "measure"],
                "budgets": {"row_top_k": 4},
                "similarity_threshold": None,
            },
            "cera_planner_request_hash": "b" * 64,
            "cera_planner_valid_plan_count": 2,
            "cera_planner_input_tokens": 100,
            "cera_planner_latency_seconds": 0.5,
            "cera_round9_closure_outcome_counts": {"UNIQUE_EXECUTABLE": 2},
            "cera_planner_derivation_count": 2,
            "cera_round9_partition_original_count": 1,
            "cera_round9_partition_alternative_count": 1,
            "cera_round10_closure_audit_records": [
                {
                    "canonical_program_id": "p1",
                    "provenance_ids": ["entity"],
                    "resource_complete": True,
                    "projected_answer": "42",
                },
                {
                    "canonical_program_id": "p2",
                    "provenance_ids": ["measure"],
                    "resource_complete": True,
                    "projected_answer": "41",
                },
            ],
            "cera_round12_semantic_type_audit_records": [
                {"closure_outcome": "UNIQUE_EXECUTABLE", "signature_id": "LOOKUP_VALUE_SCALAR"},
                {"closure_outcome": "UNIQUE_EXECUTABLE", "signature_id": "LOOKUP_VALUE_SCALAR"},
            ],
            "cera_planner_proposal_visible_to_planner": False,
            "cera_planner_table_values_visible_to_planner": False,
        }
        rows = build_constructor_sample_rows(
            [runtime_row()],
            [prediction],
            split="dev",
        )
        self.assertTrue(rows[0]["paired_executable"])
        self.assertTrue(rows[0]["constructor_registry_ready"])
        self.assertTrue(rows[0]["unique_operand_resolution"])
        self.assertEqual(rows[0]["gold_join_status"], "NOT_ACCESSED")
        self.assertIsNone(rows[0]["oracle_repairable_postfreeze"])
        schema = json.loads(
            (Path(__file__).resolve().parents[3]
             / "certa_goal_packs/CERTA_EGRA_V0_CONSTRUCTION_AND_CONDITIONAL_DECISION_GATE_PACK/SAMPLE_LEVEL_MASTER_SCHEMA.json").read_text()
        )
        jsonschema.validate(rows[0], schema)
        unblinded = unblind_constructor_sample_rows(
            rows,
            [{"sample_id": "s1", "table_id": "t1", "gold_answer": ["41"]}],
        )
        self.assertEqual(unblinded[0]["gold_join_status"], "JOINED_POSTFREEZE")
        self.assertTrue(unblinded[0]["gold_answer_in_executable_space_postfreeze"])
        self.assertTrue(unblinded[0]["oracle_repairable_postfreeze"])

    def test_role_freeze_uses_only_runtime_question_and_is_resumable(self):
        generator = FakeStructuredGenerator(scalar_lookup_payload())
        rows = freeze_role_contract_rows([runtime_row()], generator)
        self.assertEqual(len(generator.calls), 1)
        self.assertEqual(rows[0]["sample_id"], "s1")
        self.assertEqual(rows[0]["contract"], scalar_lookup_payload())
        self.assertNotIn("question", rows[0])
        serialized = json.dumps(rows[0])
        self.assertNotIn("gold_answer", serialized)
        self.assertNotIn('"answer":', serialized)

        resumed = freeze_role_contract_rows([runtime_row()], generator, rows)
        self.assertEqual(resumed, rows)
        self.assertEqual(len(generator.calls), 1)

        contaminated = dict(runtime_row(), answer=["42"])
        with self.assertRaisesRegex(ValueError, "role_runtime_fields_mismatch"):
            freeze_role_contract_rows([contaminated], generator)

    def test_b0_freeze_is_an_exact_answer_preserving_projection(self):
        prediction = {
            "id": "s1",
            "table_id": "t1",
            "question": "What is A?",
            "llm_raw_output": "The answer is 42.",
            "llm_answer": "42",
            "final_answer": "42",
            "black_box_api_generator": True,
            "api_model": "Qwen3-8B",
            "api_base_url": "http://127.0.0.1:30338/v1",
            "api_key_env": "EMPTY",
            "generator_backend": "vllm_chat",
            "api_cache_hit": False,
            "api_cache_mode": "readwrite",
            "chat_template_kwargs": {"enable_thinking": False},
            "generated_token_count": 6,
            "llm_generation_seconds": 0.2,
            "api_usage": {"prompt_tokens": 10, "completion_tokens": 6},
        }
        rows = freeze_b0_rows([runtime_row()], [prediction])
        self.assertEqual(rows[0]["generation"]["text"], "The answer is 42.")
        self.assertTrue(rows[0]["generation"]["black_box_api"])
        self.assertNotIn("gold", json.dumps(rows[0]))

        prediction["final_answer"] = "changed"
        with self.assertRaisesRegex(ValueError, "b0_prediction_mutated"):
            freeze_b0_rows([runtime_row()], [prediction])


if __name__ == "__main__":
    unittest.main()
