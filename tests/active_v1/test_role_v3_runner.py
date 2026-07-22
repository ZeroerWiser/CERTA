import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run_cscr_pipeline
from tools import certa_active_v1


PACK = Path("/home/hsh/ME/Table/EMNLP2026/certa_goal_packs/CERTA_ACTIVE_V1_ROLE_V3_FINAL_METHOD_PACK")
PROFILE = Path("/home/hsh/ME/Table/EMNLP2026/CERTA/configs/profiles/certa_active_v1.env")
HEAD = "a" * 40


class FakeRoleV3Generator:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        self.chat_template_kwargs = {"enable_thinking": False}
        FakeRoleV3Generator.instances.append(self)

    def _completion_request_kwargs(self, **kwargs):
        response_format = kwargs.pop("response_format")
        return {
            "model": self.kwargs["model"],
            "messages": [{"role": "user", "content": kwargs.pop("prompt")}],
            "response_format": response_format,
            "extra_body": {"chat_template_kwargs": self.chat_template_kwargs},
            "max_tokens": kwargs.pop("max_new_tokens"),
            **kwargs,
        }

    def generate_json_schema(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return {
            "text": json.dumps({
                "schema_version": "certa_active_role_contract_v3",
                "role_id": "COUNT_SCALAR",
            }),
            "api_usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
            "generation_seconds": 0.25,
            "api_cache_hit": False,
            "api_cache_mode": "off",
            "api_model": "Qwen3-8B",
            "structured_output_fallback_used": False,
        }


def fake_git(*args):
    if args == ("status", "--porcelain"):
        return ""
    if args == ("rev-parse", "HEAD"):
        return HEAD
    raise AssertionError(args)


class RoleV3RunnerTests(unittest.TestCase):
    def setUp(self):
        FakeRoleV3Generator.instances.clear()

    def test_freeze_binds_all_model_facing_authorities_before_calls(self):
        with tempfile.TemporaryDirectory() as td, patch.object(certa_active_v1, "_git", side_effect=fake_git):
            root = Path(td)
            result = certa_active_v1.freeze_role_v3_interface(root, PROFILE)
            self.assertEqual(result["interface_commit"], HEAD)
            self.assertEqual(result["model"]["max_new_tokens"], 64)
            self.assertEqual(result["model"]["cache_mode"], "off")
            self.assertEqual(result["thresholds"]["wire_required"], 36)
            self.assertEqual(result["thresholds"]["accepted_role_precision_min"], 0.95)
            for name in (
                "ROLE_V3_INTERFACE_FREEZE.json", "ROLE_V3_SOURCE_MANIFEST.json",
                "ROLE_V3_PROMPT_TEMPLATE.txt", "ROLE_V3_ROLE_CARDS.json",
                "ROLE_V3_OUTPUT_SCHEMA.json", "ROLE_V3_CANONICAL_REGISTRY.json",
                "ROLE_V3_FRESH_QUESTIONS.json", "ROLE_V3_GATE_THRESHOLDS.json",
            ):
                self.assertTrue((root / "freeze" / name).is_file(), name)
            frozen_text = (root / "freeze/ROLE_V3_INTERFACE_FREEZE.json").read_text()
            self.assertNotIn("canonical_label_path", frozen_text)

    def test_runner_makes_exactly_36_uncached_post_attempts_and_closes(self):
        with tempfile.TemporaryDirectory() as td, patch.object(certa_active_v1, "_git", side_effect=fake_git):
            root = Path(td)
            certa_active_v1.freeze_role_v3_interface(root, PROFILE)
            with patch.object(run_cscr_pipeline, "OpenAIChatGenerator", FakeRoleV3Generator):
                result = certa_active_v1.run_role_v3_predictions(
                    PACK / "ROLE_V3_FRESH_QUESTIONS.json",
                    root / "freeze/ROLE_V3_INTERFACE_FREEZE.json",
                    root,
                )
            generator = FakeRoleV3Generator.instances[0]
            self.assertEqual(generator.kwargs["cache_mode"], "off")
            self.assertEqual(generator.kwargs["max_retries"], 0)
            self.assertEqual(len(generator.calls), 36)
            self.assertTrue(all(call[1]["max_new_tokens"] == 64 for call in generator.calls))
            self.assertEqual(len(result["items"]), 36)
            self.assertTrue(all(set(row["prediction"]) == {"schema_version", "role_id"} for row in result["items"]))
            ledger = [json.loads(line) for line in (root / "logs/ROLE_V3_ENDPOINT_LEDGER.jsonl").read_text().splitlines()]
            self.assertEqual(len(ledger), 36)
            self.assertTrue(all(row["method"] == "POST" for row in ledger))
            self.assertTrue(all(row["path"] == "/v1/chat/completions" for row in ledger))
            self.assertTrue(all(row["transport_attempts"] == 1 and not row["cache_hit"] for row in ledger))
            close = json.loads((root / "role_v3/ROLE_V3_PREDICTION_CLOSE.json").read_text())
            self.assertEqual(close["logical_calls"], 36)
            self.assertEqual(close["transport_attempts"], 36)
            self.assertEqual(close["raw_request_count"], 36)
            self.assertEqual(close["raw_response_count"], 36)
            self.assertTrue(close["worktree_clean"])

    def test_runner_refuses_to_overwrite_closed_predictions(self):
        with tempfile.TemporaryDirectory() as td, patch.object(certa_active_v1, "_git", side_effect=fake_git):
            root = Path(td)
            certa_active_v1.freeze_role_v3_interface(root, PROFILE)
            (root / "role_v3").mkdir(exist_ok=True)
            (root / "role_v3/ROLE_V3_PREDICTIONS.json").write_text("{}\n")
            with self.assertRaisesRegex(FileExistsError, "refusing_to_overwrite_role_v3_predictions"):
                certa_active_v1.run_role_v3_predictions(
                    PACK / "ROLE_V3_FRESH_QUESTIONS.json",
                    root / "freeze/ROLE_V3_INTERFACE_FREEZE.json",
                    root,
                )


if __name__ == "__main__":
    unittest.main()
