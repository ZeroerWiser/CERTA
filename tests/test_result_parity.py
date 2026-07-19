import json
import tempfile
import unittest
from pathlib import Path

from tools.result_parity import compare_dual_parity, compare_run_directories


class ResultParityTests(unittest.TestCase):
    def test_ignores_only_declared_volatile_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            target = root / "target"
            source.mkdir()
            target.mkdir()
            record = {
                "sample_id": "s-1",
                "final_answer": "42",
                "llm_answer": "42",
                "answer_source": "original",
                "operation": "sum",
                "certificate": {"valid": True},
                "run_id": "old-id",
                "elapsed_seconds": 1.0,
            }
            for directory, run_id, elapsed in ((source, "old-id", 1.0), (target, "new-id", 2.0)):
                (directory / "predictions.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
                (directory / "predictions.debug.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
                (directory / "metrics.json").write_text(json.dumps({"EM_max_accuracy": 1.0}), encoding="utf-8")
                (directory / "run_config.json").write_text(json.dumps({"run_id": run_id, "elapsed_seconds": elapsed}), encoding="utf-8")
            report = compare_run_directories(source, target)
            self.assertEqual(report["status"], "PASS")

    def test_reports_stable_answer_difference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            target = root / "target"
            source.mkdir()
            target.mkdir()
            for directory, answer in ((source, "42"), (target, "43")):
                record = {"sample_id": "s-1", "final_answer": answer}
                for name in ("predictions.jsonl", "predictions.debug.jsonl"):
                    (directory / name).write_text(json.dumps(record) + "\n", encoding="utf-8")
                (directory / "metrics.json").write_text("{}", encoding="utf-8")
                (directory / "run_config.json").write_text("{}", encoding="utf-8")
            report = compare_run_directories(source, target)
            self.assertEqual(report["status"], "FAIL")
            self.assertEqual(report["differences"][0]["field"], "final_answer")

    def test_model_identity_difference_fails(self):
        self._assert_config_difference_fails("model_path", "Qwen3-8B", "Qwen2.5-7B-Instruct")

    def test_input_checksum_difference_fails(self):
        self._assert_config_difference_fails("input_sha256", "a" * 64, "b" * 64)

    def test_nested_candidate_run_id_difference_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, target = self._make_runs(Path(tmp))
            record = {"sample_id": "s-1", "candidate": {"run_id": "candidate-a"}}
            changed = {"sample_id": "s-1", "candidate": {"run_id": "candidate-b"}}
            for directory, payload in ((source, record), (target, changed)):
                for name in ("predictions.jsonl", "predictions.debug.jsonl"):
                    (directory / name).write_text(json.dumps(payload) + "\n", encoding="utf-8")
            self.assertEqual(compare_run_directories(source, target)["status"], "FAIL")

    def test_answer_equivalence_does_not_replace_strict_artifact_parity(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, target = self._make_runs(Path(tmp))
            source_record = {
                "sample_id": "s-1",
                "llm_answer": "42",
                "final_answer": "42",
                "certificate": {"reject_reasons": ["a", "b"]},
            }
            target_record = {
                "sample_id": "s-1",
                "llm_answer": "42",
                "final_answer": "42",
                "certificate": {"reject_reasons": ["b", "a"]},
            }
            for directory, record in ((source, source_record), (target, target_record)):
                for name in ("predictions.jsonl", "predictions.debug.jsonl"):
                    (directory / name).write_text(json.dumps(record) + "\n", encoding="utf-8")
            report = compare_dual_parity(source, target)
            self.assertEqual(report["answer_equivalent"]["status"], "PASS")
            self.assertEqual(report["artifact_strict_parity"]["status"], "FAIL")

    def _make_runs(self, root):
        source = root / "source"
        target = root / "target"
        source.mkdir()
        target.mkdir()
        for directory in (source, target):
            for name in ("predictions.jsonl", "predictions.debug.jsonl"):
                (directory / name).write_text(json.dumps({"sample_id": "s-1"}) + "\n", encoding="utf-8")
            (directory / "metrics.json").write_text("{}", encoding="utf-8")
            (directory / "run_config.json").write_text("{}", encoding="utf-8")
        return source, target

    def _assert_config_difference_fails(self, key, source_value, target_value):
        with tempfile.TemporaryDirectory() as tmp:
            source, target = self._make_runs(Path(tmp))
            (source / "run_config.json").write_text(json.dumps({key: source_value}), encoding="utf-8")
            (target / "run_config.json").write_text(json.dumps({key: target_value}), encoding="utf-8")
            self.assertEqual(compare_run_directories(source, target)["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
