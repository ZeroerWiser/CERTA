import json
import tempfile
import unittest
from pathlib import Path

from tools.certa_round1_artifacts import analyze_round1, prepare_round1


class Round1ArtifactTests(unittest.TestCase):
    @staticmethod
    def _runtime_config():
        return {
            "mode": "full_cert",
            "dataset": "hitab",
            "generator_backend": "vllm_chat",
            "api_model": "Qwen3-8B",
            "api_base_url": "http://127.0.0.1:30338/v1",
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

    def test_prepare_freezes_table_disjoint_manifests_and_gold_blind_runtime_inputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "clean.jsonl"
            tables = root / "tables"
            output = root / "output"
            tables.mkdir()
            rows = []
            for table_index in range(8):
                (tables / f"{table_index}.json").write_text(
                    json.dumps({"texts": [["name", "value"], ["A", str(table_index)]]}),
                    encoding="utf-8",
                )
                for sample_index in range(2):
                    rows.append({
                        "id": f"s-{table_index}-{sample_index}",
                        "table_id": str(table_index),
                        "table_source": "fixture",
                        "question": f"question {table_index} {sample_index}",
                        "answer": [table_index],
                        "answer_formulas": ["=B2"],
                        "linked_cells": {"quantity_link": {"[ANSWER]": {"(1, 1)": table_index}}},
                        "reference_cells_map": {"B2": "(1, 1)"},
                        "aggregation": ["none"],
                    })
            dataset.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

            summary = prepare_round1(
                dataset_path=dataset,
                table_root=tables,
                output_root=output,
                dev_size=4,
                holdout_size=4,
            )

            self.assertEqual(summary["dev_count"], 4)
            self.assertEqual(summary["holdout_count"], 4)
            dev = [json.loads(line) for line in (output / "freeze/DEV_COHORT.jsonl").read_text().splitlines()]
            holdout = [json.loads(line) for line in (output / "freeze/HOLDOUT_COHORT_SEALED.jsonl").read_text().splitlines()]
            self.assertFalse({row["table_id"] for row in dev} & {row["table_id"] for row in holdout})
            runtime_text = (output / "inputs/dev_blind.jsonl").read_text()
            for forbidden in ("answer", "linked_cells", "reference_cells_map", "answer_formulas", "[ANSWER]"):
                self.assertNotIn(forbidden, runtime_text)
            runtime_rows = [json.loads(line) for line in runtime_text.splitlines()]
            self.assertEqual([row["id"] for row in runtime_rows], [row["sample_id"] for row in dev])
            sealed_text = (output / "freeze/HOLDOUT_COHORT_SEALED.jsonl").read_text()
            self.assertNotIn("question", sealed_text)
            self.assertNotIn("answer", sealed_text)

    def test_analyze_hashes_blind_master_before_dev_gold_join_and_preserves_oracle_as_diagnostic(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "clean.jsonl"
            tables = root / "tables"
            output = root / "output"
            primary = root / "primary"
            tables.mkdir()
            primary.mkdir()
            source_rows = []
            for table_index in range(2):
                (tables / f"{table_index}.json").write_text(
                    json.dumps({"texts": [["name", "value"], ["A", str(table_index + 1)]]}),
                    encoding="utf-8",
                )
                source_rows.append({
                    "id": f"s-{table_index}",
                    "table_id": str(table_index),
                    "table_source": "fixture",
                    "question": "what is the value?",
                    "answer": [table_index + 1],
                    "aggregation": ["none"],
                })
            dataset.write_text("".join(json.dumps(row) + "\n" for row in source_rows), encoding="utf-8")
            prepare_round1(dataset_path=dataset, table_root=tables, output_root=output, dev_size=1, holdout_size=1)
            cohort = json.loads((output / "freeze/DEV_COHORT.jsonl").read_text())
            source = next(row for row in source_rows if row["id"] == cohort["sample_id"])
            gold = str(source["answer"][0])
            record = {
                "id": source["id"],
                "table_id": source["table_id"],
                "llm_answer": "wrong",
                "final_answer": "wrong",
                "executor_answer": "wrong",
                "llm_input_audit": {"request_sha256": "r", "rendered_prompt_sha256": "p"},
                "cera_enabled": True,
                "cera_stage": "E71_v4_packet_shadow",
                "cera_shadow_only": True,
                "cera_planner_called": True,
                "cera_planner_parse_ok": True,
                "cera_planner_valid_plan_count": 2,
                "cera_planner_proposal_visible_to_planner": False,
                "cera_planner_table_values_visible_to_planner": False,
                "cera_planner_derivation_count": 2,
                "cera_round9_partition_original_count": 1,
                "cera_round9_partition_alternative_count": 1,
                "cera_round8_basis_count": 1,
                "cera_round8_separating_intervention_count": 1,
                "cera_round8_contrast_registry_complete": True,
                "cera_round11_closure_resource_complete": True,
                "cera_round10_closure_audit_records": [
                    {"closure_outcome": "UNIQUE_EXECUTABLE", "signature_id": "LOOKUP_VALUE_SCALAR"}
                ],
                "cera_evidence_packet": {"compact_behavioral_contrast_v3": {
                    "original_hypothesis": {
                        "hypothesis_id": "H1", "executed_answer": "wrong", "answer_key": "wrong",
                        "response_vector": {"I1": "INVARIANT:wrong"},
                    },
                    "alternative_hypotheses": [{
                        "hypothesis_id": "H2", "executed_answer": gold, "answer_key": gold,
                        "response_vector": {"I1": f"ANSWER_CHANGED:{gold}"},
                    }],
                    "separating_interventions": [{"intervention_ref": "I1"}],
                    "registry": {
                        "hypothesis_records": [
                            {"hypothesis_id": "H1", "derivation_ref": "D1", "side": "original"},
                            {"hypothesis_id": "H2", "derivation_ref": "D2", "side": "alternative"},
                        ],
                        "derivation_records": [
                            {"derivation_ref": "D1", "executed_answer": "wrong"},
                            {"derivation_ref": "D2", "executed_answer": gold},
                        ],
                        "evidence_records": [],
                        "intervention_records": [{
                            "intervention_ref": "I1", "original_signature": "INVARIANT:wrong",
                            "alternative_signature": f"ANSWER_CHANGED:{gold}",
                            "original_benign_control": True, "alternative_benign_control": False,
                            "evaluable_on_both_sides": True, "separating": True,
                        }],
                    },
                    "states": {"contrast_registry_complete": True},
                    "unknowns": [],
                }},
            }
            (primary / "predictions.debug.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
            (primary / "run_config.json").write_text(json.dumps(self._runtime_config()), encoding="utf-8")

            summary = analyze_round1(
                output_root=output,
                dataset_path=dataset,
                primary_output_dir=primary,
                diagnostic_output_dirs={},
            )

            blind_text = (output / "results/sample_master.blind.jsonl").read_text()
            self.assertNotIn("gold", blind_text)
            self.assertNotIn("correct", blind_text)
            unblind = json.loads((output / "results/sample_master.dev_unblind.jsonl").read_text())
            self.assertFalse(unblind["b0_correct"])
            self.assertTrue(unblind["candidate_oracle_correct"])
            self.assertTrue(unblind["oracle_repairable"])
            self.assertEqual(unblind["round1_final_answer"], "wrong")
            self.assertEqual(summary["oracle_repairable_dev_count"], 1)


if __name__ == "__main__":
    unittest.main()
