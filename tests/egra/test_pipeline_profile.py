import os
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import run_cscr_pipeline
from graph_builder import HCEG


class PipelineProfileTests(unittest.TestCase):
    def test_frozen_b0_loader_exact_joins_sample_table_question_and_transport(self):
        item = {"id": "s1", "table_id": "t1", "question": "What is A?"}
        row = {
            "schema_version": "certa_egra_b0_freeze_v1",
            "sample_id": "s1",
            "table_id": "t1",
            "question_sha256": run_cscr_pipeline.canonical_json_hash(
                {"question": item["question"]}
            ),
            "generation": {
                "text": "42",
                "black_box_api": True,
                "api_model": "Qwen3-8B",
                "api_base_url": "http://127.0.0.1:30338/v1",
                "generator_backend": "vllm_chat",
                "chat_template_kwargs": {"enable_thinking": False},
            },
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "b0.jsonl"
            path.write_text(json.dumps(row) + "\n")
            loaded = run_cscr_pipeline._load_certa_egra_frozen_b0(path, [item])
            self.assertEqual(loaded["s1"], row)
            row["table_id"] = "wrong"
            path.write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "frozen_b0_table_mismatch"):
                run_cscr_pipeline._load_certa_egra_frozen_b0(path, [item])

    def test_frozen_role_loader_exact_joins_without_labels(self):
        item = {"id": "s1", "table_id": "t1", "question": "What is A?"}
        question_sha256 = run_cscr_pipeline.canonical_json_hash(
            {"question": item["question"]}
        )
        row = {
            "schema_version": "certa_egra_role_freeze_row_v1",
            "sample_id": "s1",
            "table_id": "t1",
            "question_sha256": question_sha256,
            "contract": {
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
            },
            "audit": {
                "calls": 1,
                "model": "Qwen3-8B",
                "backend": "vllm_chat",
                "api_base_url": "http://127.0.0.1:30338/v1",
                "thinking": {"enable_thinking": False},
            },
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "roles.jsonl"
            path.write_text(json.dumps(row) + "\n")
            loaded = run_cscr_pipeline._load_certa_egra_frozen_roles(path, [item])
            self.assertEqual(loaded[question_sha256], {
                "contract": row["contract"],
                "audit": row["audit"],
            })
            row["question_sha256"] = "0" * 64
            path.write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "frozen_role_question_mismatch"):
                run_cscr_pipeline._load_certa_egra_frozen_roles(path, [item])

    def test_non_c0_arm_reuses_exact_frozen_b0_without_primary_call(self):
        args = SimpleNamespace(
            top_k_logprobs=5,
            certa_egra_arm="C2_EGRA",
            _certa_egra_frozen_b0_by_id={
                "s1": {
                    "sample_id": "s1",
                    "table_id": "t1",
                    "question_sha256": run_cscr_pipeline.canonical_json_hash(
                        {"question": "What is A?"}
                    ),
                    "generation": {
                        "text": "42",
                        "logprobs": None,
                        "black_box_api": True,
                        "api_model": "Qwen3-8B",
                        "api_base_url": "http://127.0.0.1:30338/v1",
                        "generator_backend": "vllm_chat",
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                }
            },
        )
        prepared = {
            "result": {"id": "s1", "table_id": "t1"},
            "item": {"id": "s1", "table_id": "t1", "question": "What is A?"},
        }
        generator = MagicMock()
        with patch.object(run_cscr_pipeline, "prepare_non_llm_steps", return_value=prepared):
            with patch.object(run_cscr_pipeline, "finalize_after_llm", return_value={"ok": True}) as finalize:
                self.assertEqual(
                    run_cscr_pipeline.process_single(
                        prepared["item"], {}, generator, args, "full_cert"
                    ),
                    {"ok": True},
                )
        generator.generate.assert_not_called()
        reused = finalize.call_args.args[1]
        self.assertEqual(reused["text"], "42")
        self.assertTrue(reused["certa_egra_frozen_b0_reused"])

    def test_constructor_finalize_returns_before_intervention_and_decision(self):
        args = SimpleNamespace(
            certa_egra_arm="C0_FLAT_SCHEMA_CURRENT",
            generator_backend="vllm_chat",
            api_model="Qwen3-8B",
            api_base_url="http://127.0.0.1:30338/v1",
            api_key_env="EMPTY",
            api_cache_mode="readwrite",
            black_box_commit_policy="freeze",
            api_format_normalizer="off",
            prompt_style="structure_aware",
            structural_prior_weighting=False,
            dataset="hitab",
            main_cert_profile=False,
        )
        prepared = {
            "result": {"id": "s1", "table_id": "t1", "non_llm_preparation_seconds": 0.1},
            "graph": None,
            "evidence": None,
            "interventions": None,
            "executor_result": None,
            "all_exec_candidates": [],
            "item": {"id": "s1", "table_id": "t1", "dataset": "hitab", "question": "What is A?"},
            "table_json": {"texts": [["Name", "Value"], ["A", "42"]]},
        }
        generation = {
            "text": "42",
            "logprobs": None,
            "generated_token_count": 1,
            "generation_seconds": 0.01,
            "black_box_api": True,
            "api_model": "Qwen3-8B",
            "api_base_url": "http://127.0.0.1:30338/v1",
            "api_key_env": "EMPTY",
            "generator_backend": "vllm_chat",
            "api_cache_hit": False,
            "api_cache_mode": "readwrite",
            "chat_template_kwargs": {"enable_thinking": False},
        }
        with patch.object(run_cscr_pipeline, "_build_hceg_and_retrieve", return_value=(HCEG(), object(), {})):
            with patch.object(run_cscr_pipeline, "run_egra_constructor_shadow", return_value={"cera_planner_called": True}) as constructor:
                with patch.object(run_cscr_pipeline.InterventionEngine, "generate_interventions", side_effect=AssertionError("forbidden")):
                    result = run_cscr_pipeline.finalize_after_llm(
                        prepared,
                        generation,
                        args,
                        "full_cert",
                        generator=object(),
                    )
        constructor.assert_called_once()
        self.assertTrue(result["certa_egra_construction_only"])
        self.assertFalse(result["certa_egra_intervention_generated"])
        self.assertFalse(result["certa_egra_decision_executed"])
        self.assertEqual(result["final_answer"], "42")
        self.assertFalse(result["b0_mutation"])

    def test_frozen_profile_and_runner_preserve_answer_and_method_boundaries(self):
        root = Path(__file__).resolve().parents[2]
        profile = (root / "configs/profiles/certa_egra_v0.env").read_text()
        runner = (root / "scripts/06_run_certa_egra_v0.sh").read_text()
        for frozen in (
            'CSCR_GPUS="4"',
            'CSCR_PYTHON="/home/hsh/anaconda3/envs/cond/bin/python"',
            'CSCR_BLACK_BOX_COMMIT_POLICY="freeze"',
            'CERTA_EGRA_EMBEDDING_DEVICE="cuda:0"',
            'CERTA_EGRA_EMBEDDING_FILE_TREE_SHA256="f7b400dfd56a18cacb3f584d097722aba842a961a0b57448c915f9166d5eb521"',
            'CSCR_OPERATION_CERTIFICATE_PROFILE=""',
            'CERTA_CERA_COMMIT_APPROVED_REPAIR="0"',
        ):
            self.assertIn(frozen, profile)
        for required in (
            'EXPECTED_EXTERNAL_ROOT="/home/hsh/ME/Table/EMNLP2026/certa_egra_outputs/CERTA_EGRA_V0_20260720T152831Z"',
            'export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"',
            "CONSTRUCTOR_CONFIG_FREEZE.json",
            "EARLY_SENTINEL_GATE.json",
            "--certa-egra-arm",
            "--certa-egra-frozen-b0-file",
            "--certa-egra-frozen-role-file",
            "--cera-stage E71",
            "--cera-shadow-only",
            "--cera-planner-contract rcpc_signature_v2",
            "--cera-planner-boundary proposal_blind_schema_only",
            "freeze-role",
            "constructor-master",
        ):
            self.assertIn(required, runner)
        for forbidden in (
            "--enable-cera-repair",
            "--cera-commit-approved-repair",
            "--cera-stage E72",
            "--self-consistency",
            "--question-type-router",
        ):
            self.assertNotIn(forbidden, runner)

    def test_c2_initializes_one_frozen_encoder_on_the_explicit_device(self):
        args = SimpleNamespace(
            certa_egra_arm="C2_EGRA",
            certa_egra_embedding_device="cuda:0",
            certa_egra_embedding_file_tree_sha256=(
                "f7b400dfd56a18cacb3f584d097722aba842a961a0b57448c915f9166d5eb521"
            ),
        )
        sentinel = object()
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "4"}, clear=False):
            with patch("run_cscr_pipeline._verify_certa_egra_embedding_files", return_value={}):
                with patch("certa.egra.retrieval.FrozenE5Encoder", return_value=sentinel) as factory:
                    run_cscr_pipeline._initialize_certa_egra_runtime(args)
        factory.assert_called_once_with(device="cuda:0")
        self.assertIs(args._certa_egra_encoder, sentinel)
        self.assertGreaterEqual(args._certa_egra_model_load_seconds, 0.0)

    def test_non_retrieval_arms_do_not_load_embedding_model(self):
        for arm in ("", "C0_FLAT_SCHEMA_CURRENT", "C1_ROLE_ALIGNED_FLAT"):
            args = SimpleNamespace(certa_egra_arm=arm)
            with patch("certa.egra.retrieval.FrozenE5Encoder") as factory:
                run_cscr_pipeline._initialize_certa_egra_runtime(args)
            factory.assert_not_called()
            self.assertFalse(hasattr(args, "_certa_egra_encoder"))

    def test_c2_rejects_embedding_identity_or_gpu_drift(self):
        baseline = {
            "certa_egra_arm": "C2_EGRA",
            "certa_egra_embedding_device": "cuda:0",
            "certa_egra_embedding_file_tree_sha256": (
                "f7b400dfd56a18cacb3f584d097722aba842a961a0b57448c915f9166d5eb521"
            ),
        }
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "3"}, clear=False):
            with self.assertRaisesRegex(ValueError, "requires_CUDA_VISIBLE_DEVICES_4"):
                run_cscr_pipeline._initialize_certa_egra_runtime(SimpleNamespace(**baseline))

        changed = dict(baseline)
        changed["certa_egra_embedding_file_tree_sha256"] = "0" * 64
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "4"}, clear=False):
            with self.assertRaisesRegex(ValueError, "embedding_file_tree_sha256_mismatch"):
                run_cscr_pipeline._initialize_certa_egra_runtime(SimpleNamespace(**changed))


if __name__ == "__main__":
    unittest.main()
