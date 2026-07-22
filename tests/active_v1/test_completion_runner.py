import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from certa.reproducibility.canonical_json import canonical_json_hash
from tools.certa_active_v1_completion import cost_ledger, decision_artifact_paths, model_call, request_record, role_calculator_record, v3_retrieval_contract


ROLE = {"schema_version": "certa_active_role_v3_canonical_record_v1", "role_id": "COUNT_SCALAR", "supported": True, "intent": "COUNT", "answer_role": "SCALAR", "projection": "SCALAR_RESULT_PROJECTION", "cardinality": "SINGLE", "operation_family": "COUNT", "requires_time_scope": "DEFERRED_TO_GROUNDING", "requires_unit_consistency": "DEFERRED_TO_EXECUTION"}


class CompletionRunnerContractTests(unittest.TestCase):
    def test_v3_compatibility_views_are_deterministic_and_non_mutating(self):
        before = dict(ROLE)
        retrieval = v3_retrieval_contract(ROLE)
        self.assertEqual((retrieval["rank_direction"], retrieval["rank_k"]), ("NONE", None))
        self.assertEqual(retrieval["signature_candidates"], ["COUNT_SCALAR"])
        record = role_calculator_record("S1", ROLE)
        self.assertEqual(record["signature"], "COUNT_SCALAR")
        self.assertEqual(record["record_sha256"], canonical_json_hash(ROLE))
        self.assertEqual(ROLE, before)

    def test_every_chat_request_is_recorded_as_post(self):
        record = request_record("PLANNER", 3, "S1", {"model": "Qwen3-8B"})
        self.assertEqual(record["method"], "POST")
        self.assertEqual(record["path"], "/v1/chat/completions")

    def test_dev_decision_eligibility_uses_the_pack_bound_path(self):
        paths = decision_artifact_paths("dev")
        self.assertEqual(paths["eligibility"].name, "DECISION_ELIGIBILITY.blind.json")
        self.assertEqual(paths["close"].name, "DEV_SELECTED_FINAL_PREDICTION_CLOSE.json")

    def test_failed_post_attempt_is_preserved_in_the_endpoint_ledger(self):
        class FailingGenerator:
            def _completion_request_kwargs(self, **kwargs): return {"model": "Qwen3-8B"}
            def generate(self, *args, **kwargs): raise TimeoutError("fixture")
        with TemporaryDirectory() as root, patch("tools.certa_active_v1_completion.OUT", Path(root)):
            with self.assertRaises(TimeoutError): model_call(FailingGenerator(), "DEV_PLANNER", "S1", "prompt", 32)
            ledger = cost_ledger()
            self.assertEqual((ledger["logical_calls"], ledger["transport_attempts"]), (1, 1))
            response = next((Path(root) / "raw/dev_planner").glob("*_response.json"))
            self.assertFalse(json.loads(response.read_text())["ok"])

    def test_stage_cost_ledger_does_not_charge_constructor_calls_to_decision(self):
        rows = [{"logical_call_type": kind, "transport_attempts": 1, "usage": {}, "generation_seconds": 1} for kind in ("DEV_PLANNER_C2", "DEV_CERA")]
        with TemporaryDirectory() as root, patch("tools.certa_active_v1_completion.OUT", Path(root)):
            path = Path(root) / "logs/ENDPOINT_LEDGER.jsonl"; path.parent.mkdir()
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            self.assertEqual(cost_ledger(("DEV_CERA",))["logical_calls"], 1)


if __name__ == "__main__":
    unittest.main()
