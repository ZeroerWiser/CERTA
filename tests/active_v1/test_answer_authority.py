import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from certa.active_v1.answer_authority import active_answer_hash
from certa.reproducibility.canonical_json import canonical_json_hash
from tools.certa_active_v1 import run_active_b0


class ActiveAnswerAuthorityTests(unittest.TestCase):
    def test_hash_uses_the_frozen_inference_equivalence_key(self):
        expected = canonical_json_hash({"equivalence_key": "NUMERIC_EXACT_CANONICAL:2"})
        self.assertEqual(active_answer_hash("2.0"), expected)
        self.assertEqual(active_answer_hash("2"), expected)
        self.assertNotEqual(active_answer_hash("3"), expected)

    def test_b0_runner_freezes_post_requests_and_proves_cache_replay(self):
        class FakeGenerator:
            def __init__(self, **kwargs):
                self.cache_mode = kwargs["cache_mode"]
                self.cache_hits = 0
                self.cache_misses = 0
                Path(kwargs["cache_path"]).parent.mkdir(parents=True, exist_ok=True)
                Path(kwargs["cache_path"]).write_text("fixture cache\n", encoding="utf-8")

            def _completion_request_kwargs(self, **kwargs):
                return {
                    "model": "Qwen3-8B", "messages": [{"role": "user", "content": kwargs["prompt"]}],
                    "temperature": kwargs["temperature"], "top_p": kwargs["top_p"],
                    "max_tokens": kwargs["max_new_tokens"],
                }

            def generate(self, prompts, **kwargs):
                hit = self.cache_mode == "require"
                if hit:
                    self.cache_hits += len(prompts)
                else:
                    self.cache_misses += len(prompts)
                return [{
                    "text": "Answer: 2", "api_usage": {"prompt_tokens": 3, "completion_tokens": 2},
                    "generation_seconds": 0.1, "api_cache_hit": hit,
                } for _ in prompts]

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runtime = root / "runtime.jsonl"
            rows = [{
                "dataset": "hitab", "id": f"s{index}", "question": f"q{index}",
                "table_id": f"t{index}", "table_source": "fixture",
            } for index in range(64)]
            runtime.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            import run_cscr_pipeline
            with (
                patch.object(run_cscr_pipeline, "OpenAIChatGenerator", FakeGenerator),
                patch.object(run_cscr_pipeline, "load_table_for_cscr", return_value={}),
                patch.object(run_cscr_pipeline, "build_structure_aware_prompt", side_effect=lambda table, question: question),
                patch.object(run_cscr_pipeline, "extract_answer", return_value="2"),
            ):
                proof = run_active_b0(runtime, root, root / "out", root / "cache/cache.jsonl")
            self.assertTrue(proof["byte_and_answer_hash_match"])
            self.assertEqual(proof["cache_hits"], 64)
            records = [json.loads(line) for line in (root / "out/b0/DEV_B0_FREEZE.jsonl").read_text().splitlines()]
            ledger = [json.loads(line) for line in (root / "out/logs/B0_ENDPOINT_LEDGER.jsonl").read_text().splitlines()]
            self.assertEqual(len(records), 64)
            self.assertEqual(len(ledger), 64)
            self.assertTrue(all(row["method"] == "POST" for row in ledger))


if __name__ == "__main__":
    unittest.main()
