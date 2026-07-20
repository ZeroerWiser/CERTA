import sys
import unittest
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from cscr_astra_eval import (  # noqa: E402
    build_astra_payload,
    compute_metrics,
    evaluate_payload,
    exact_match,
    official_match,
    prediction_by_id,
)


class AstraEvaluatorContractTests(unittest.TestCase):
    def test_empty_or_missing_prediction_never_matches_empty_gold(self):
        self.assertFalse(exact_match("", ""))
        self.assertFalse(exact_match(None, ""))
        self.assertFalse(official_match("hitab", "", ""))

    def test_empty_or_missing_gold_is_an_explicit_invalid_reference(self):
        payload = {
            "table_results": [
                {
                    "data_index": 0,
                    "table_uid": "t-1",
                    "results": [
                        {
                            "question_index": 0,
                            "sample_id": "s-1",
                            "question": "q",
                            "correct_answer": "",
                            "generated_answer": "",
                            "symbolic_answer": "",
                        }
                    ],
                }
            ]
        }
        row = evaluate_payload(payload, dataset="hitab")[0]
        self.assertFalse(row["reference_valid"])
        self.assertEqual(row["reference_invalid_reason"], "missing_or_empty_gold")
        self.assertEqual(row["EM_textual_label"], 0)
        self.assertEqual(row["EM_symbolic_label"], 0)

    def test_duplicate_and_missing_prediction_ids_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, "duplicate prediction IDs: s-1"):
            prediction_by_id([{"id": "s-1"}, {"id": "s-1"}])
        with self.assertRaisesRegex(ValueError, "missing prediction ID"):
            prediction_by_id([{"final_answer": "42"}])

    def test_duplicate_reference_ids_fail_explicitly(self):
        clean = [
            {"id": "s-1", "table_id": "t-1", "question": "q1", "answer": ["1"]},
            {"id": "s-1", "table_id": "t-1", "question": "q2", "answer": ["2"]},
        ]
        with self.assertRaisesRegex(ValueError, "duplicate reference IDs: s-1"):
            build_astra_payload(clean, [{"id": "s-1", "final_answer": "1"}], (), "hitab")

    def test_missing_and_extra_prediction_ids_fail_explicitly(self):
        clean = [{"id": "s-1", "table_id": "t-1", "question": "q", "answer": ["1"]}]
        with self.assertRaisesRegex(ValueError, "missing prediction IDs: s-1"):
            build_astra_payload(clean, [], (), "hitab")
        with self.assertRaisesRegex(ValueError, "extra prediction IDs: s-2"):
            build_astra_payload(
                clean,
                [
                    {"id": "s-1", "final_answer": "1"},
                    {"id": "s-2", "final_answer": "2"},
                ],
                (),
                "hitab",
            )

    def test_existing_numeric_list_and_percent_matches_remain_supported(self):
        self.assertTrue(exact_match("1,000", "1000"))
        self.assertTrue(exact_match("[a, b]", "[a, b]"))
        self.assertTrue(exact_match("52.1%", "52.1"))

    def test_em_max_stays_oracle_union_and_selected_final_uses_textual_channel(self):
        rows = [
            {"EM_textual_label": 1, "EM_symbolic_label": 0},
            {"EM_textual_label": 0, "EM_symbolic_label": 1},
        ]
        metrics = compute_metrics(rows, use_judge=False, dataset="hitab")
        self.assertEqual(metrics["EM_max_accuracy"], 1.0)
        self.assertEqual(metrics["EM_max_semantics"], "oracle_union_textual_or_symbolic")
        self.assertEqual(metrics["selected_final_accuracy"], 0.5)
        self.assertEqual(metrics["selected_final_semantics"], "actual_final_answer_textual_channel")


if __name__ == "__main__":
    unittest.main()
