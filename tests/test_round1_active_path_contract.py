import json
import unittest

from graph_builder import EdgeType, GraphEdge, GraphNode, HCEG, NodeType

from certa.planner.schema_view import build_proposal_blind_planner_view
from certa.round1.contracts import (
    build_blind_sample_master_row,
    select_table_disjoint_cohorts,
    validate_shadow_prediction,
    validate_shadow_runtime_config,
)


class GoldGuardRow(dict):
    def get(self, key, default=None):
        if key in {"answer", "gold", "gold_answer", "correctness", "error_type"}:
            raise AssertionError(f"forbidden selection access: {key}")
        return super().get(key, default)


def synthetic_rows():
    rows = []
    for table_index in range(12):
        for sample_index in range(3):
            rows.append(
                GoldGuardRow(
                    id=f"s-{table_index}-{sample_index}",
                    table_id=f"t-{table_index}",
                    question=f"question {table_index} {sample_index}",
                    aggregation=["none" if sample_index % 2 == 0 else "sum"],
                )
            )
    return rows


def clean_profile():
    return {
        "mode": "full_cert",
        "dataset": "hitab",
        "generator_backend": "vllm_chat",
        "api_model": "Qwen3-8B",
        "api_base_url": "http://127.0.0.1:30338/v1",
        "api_cache_mode": "off",
        "main_cert_profile": True,
        "enable_cera_repair": True,
        "cera_stage": "E71",
        "cera_shadow_only": True,
        "cera_commit_approved_repair": False,
        "cera_enable_typed_planner": True,
        "cera_planner_boundary": "proposal_blind_schema_only",
        "cera_planner_contract": "rcpc_signature_v2",
        "cera_planner_legacy_query_semantics_mode": "audit_only",
        "cera_stepwise_trace": False,
        "adaptive_prompt": False,
        "credal_probe": False,
        "credal_gate": False,
        "question_type_router": False,
        "online_normalizer": False,
        "oracle_online_normalizer": False,
        "api_format_normalizer": "off",
        "hceg_fallback": False,
        "certificate_commit_boundary": False,
        "self_consistency": False,
        "source_risk_calibration": "off",
        "operation_commit_gate_mode": "diagnostic",
        "black_box_commit_policy": "certified",
    }


class Round1ActivePathContractTests(unittest.TestCase):
    def test_cohort_selection_is_deterministic_gold_blind_and_table_disjoint(self):
        first = select_table_disjoint_cohorts(synthetic_rows(), seed=20260720, dev_size=8, holdout_size=8)
        second = select_table_disjoint_cohorts(synthetic_rows(), seed=20260720, dev_size=8, holdout_size=8)
        self.assertEqual(first, second)
        self.assertEqual(len(first["dev"]), 8)
        self.assertEqual(len(first["holdout"]), 8)
        self.assertTrue(first["table_disjoint"])
        self.assertFalse({row["table_id"] for row in first["dev"]} & {row["table_id"] for row in first["holdout"]})
        self.assertFalse({"answer", "question"} & set(first["dev"][0]))

    def test_runtime_config_is_the_only_authority_and_all_mutators_are_off(self):
        self.assertEqual(validate_shadow_runtime_config(clean_profile()), ())
        for field, unsafe_value in (
            ("adaptive_prompt", True),
            ("online_normalizer", True),
            ("hceg_fallback", True),
            ("certificate_commit_boundary", True),
            ("cera_commit_approved_repair", True),
            ("operation_commit_gate_mode", "conservative"),
        ):
            with self.subTest(field=field):
                config = clean_profile()
                config[field] = unsafe_value
                self.assertTrue(validate_shadow_runtime_config(config))

    def test_shadow_telemetry_cannot_grant_commit_authority(self):
        record = {
            "llm_answer": "42",
            "final_answer": "42",
            "cera_stage": "E71_v4_packet_shadow",
            "cera_shadow_only": True,
            "cera_would_commit": True,
            "cera_commit_requested": False,
            "cera_commit_applied": False,
            "cera_final_committed": False,
            "cera_llm_called": False,
            "cera_planner_called": True,
            "operation_support_commit_applied": False,
            "legacy_commit_path_used": False,
            "non_certificate_answer_mutation_used": False,
            "cera_planner_proposal_visible_to_planner": False,
            "cera_planner_table_values_visible_to_planner": False,
        }
        self.assertEqual(validate_shadow_prediction(record), ())
        record["final_answer"] = "43"
        self.assertIn("final_answer_differs_from_b0", validate_shadow_prediction(record))

    def test_proposal_blind_view_excludes_proposal_gold_and_table_values(self):
        graph = HCEG()
        graph.add_node(GraphNode("measure", NodeType.HEADER, row=0, col=1, text="Value"))
        graph.add_node(GraphNode("entity", NodeType.HEADER, row=1, col=0, text="A"))
        graph.add_node(GraphNode("value", NodeType.CELL, row=1, col=1, text="42", numeric_value=42.0))
        graph.add_edge(GraphEdge("value", "entity", EdgeType.ROW_PATH))
        graph.add_edge(GraphEdge("value", "measure", EdgeType.COL_PATH))
        view = build_proposal_blind_planner_view(
            question="What is A?",
            graph=graph,
            table_json={"texts": [["Name", "Value"], ["A", "42"]]},
            query_contract=None,
            include_table_values=False,
        )
        serialized = json.dumps(view, sort_keys=True).lower()
        for forbidden in ("initial_proposal", "gold_answer", "correctness", "error_type", '"42"'):
            self.assertNotIn(forbidden, serialized)

    def test_blind_sample_master_preserves_complete_failure_and_registry_state(self):
        record = {
            "id": "s-1",
            "table_id": "t-1",
            "llm_answer": "42",
            "final_answer": "42",
            "llm_input_audit": {"request_sha256": "r", "rendered_prompt_sha256": "p"},
            "api_cache_hit": False,
            "cera_planner_called": True,
            "cera_planner_view_hash": "v",
            "cera_planner_constraint_schema_hash": "s",
            "cera_planner_prompt_hash": "pp",
            "cera_planner_request_hash": "pr",
            "cera_planner_parse_ok": True,
            "cera_planner_valid_plan_count": 1,
            "cera_round11_closure_declared_assignment_count": 2,
            "cera_round11_closure_realized_assignment_count": 2,
            "cera_round11_closure_resource_complete": True,
            "cera_round10_closure_audit_records": [
                {"closure_outcome": "UNIQUE_EXECUTABLE", "signature_id": "LOOKUP_VALUE_SCALAR"}
            ],
            "cera_planner_derivation_count": 2,
            "cera_round9_partition_original_count": 1,
            "cera_round9_partition_alternative_count": 1,
            "cera_round8_basis_count": 1,
            "cera_round8_separating_intervention_count": 1,
            "cera_round8_contrast_registry_complete": True,
            "cera_evidence_packet": {
                "compact_behavioral_contrast_v3": {
                    "original_hypothesis": {
                        "hypothesis_id": "H1",
                        "executed_answer": "42",
                        "response_vector": {"I1": "INVARIANT:42"},
                    },
                    "alternative_hypotheses": [{
                        "hypothesis_id": "H2",
                        "executed_answer": "7",
                        "response_vector": {"I1": "ANSWER_CHANGED:7"},
                    }],
                    "separating_interventions": [{"intervention_ref": "I1"}],
                    "registry": {
                        "hypothesis_records": [
                            {"hypothesis_id": "H1", "derivation_ref": "D1", "side": "original"},
                            {"hypothesis_id": "H2", "derivation_ref": "D2", "side": "alternative"},
                        ],
                        "derivation_records": [
                            {"derivation_ref": "D1", "executed_answer": "42"},
                            {"derivation_ref": "D2", "executed_answer": "7"},
                        ],
                        "evidence_records": [{"evidence_id": "E1"}],
                        "intervention_records": [{"intervention_ref": "I1", "evaluable_on_both_sides": True, "separating": True}],
                    },
                    "states": {"contrast_registry_complete": True},
                    "unknowns": [],
                }
            },
            "cera_stage": "E71_v4_packet_shadow",
            "cera_shadow_only": True,
            "cera_commit_requested": False,
            "cera_commit_applied": False,
            "cera_final_committed": False,
            "cera_llm_called": False,
            "legacy_commit_path_used": False,
            "non_certificate_answer_mutation_used": False,
            "operation_support_commit_applied": False,
            "cera_planner_proposal_visible_to_planner": False,
            "cera_planner_table_values_visible_to_planner": False,
            "input_token_count": 10,
            "generated_token_count": 2,
            "cera_planner_input_tokens": 20,
            "cera_planner_output_tokens": 4,
            "pipeline_recorded_seconds": 0.5,
        }
        row = build_blind_sample_master_row(
            record,
            {"sample_id": "s-1", "table_id": "t-1", "source_order": 0},
            dataset_hash="d",
            cohort_hash="c",
            supported_signatures=("LOOKUP_VALUE_SCALAR",),
        )
        self.assertEqual(row["round1_final_answer"], "42")
        self.assertEqual(row["common_evaluable_intervention_count"], 1)
        self.assertEqual(row["separating_intervention_count"], 1)
        self.assertEqual(row["registry_outside_answer_count"], 0)
        self.assertTrue(row["paired_executable"])
        self.assertEqual(row["logical_calls"], 2)


if __name__ == "__main__":
    unittest.main()
