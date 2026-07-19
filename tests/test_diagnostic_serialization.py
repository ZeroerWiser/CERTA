import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_cscr_pipeline import _stable_unique_strings


class DiagnosticSerializationTests(unittest.TestCase):
    def test_stable_unique_reasons_preserve_content_and_sort_order(self):
        reasons = [
            "measure_unit:target_measure_phrase_unbound",
            "operation_expression:candidate_not_arithmetic",
            "measure_unit:target_measure_phrase_unbound",
            "aggregate_echo:echo_detected",
        ]
        ordered = _stable_unique_strings(reasons)
        self.assertEqual(set(ordered), set(reasons))
        self.assertEqual(
            ordered,
            [
                "aggregate_echo:echo_detected",
                "measure_unit:target_measure_phrase_unbound",
                "operation_expression:candidate_not_arithmetic",
            ],
        )


if __name__ == "__main__":
    unittest.main()
