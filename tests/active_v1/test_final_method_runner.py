import unittest

from tools.certa_final_method import (
    development_gold_answers,
    variant_artifact_arm,
    variant_planner_call_types,
)


class FinalMethodRunnerTests(unittest.TestCase):
    def test_variant_call_matrix_is_bounded_and_explicit(self):
        self.assertEqual(
            variant_planner_call_types("V0_LEGACY_C2_HARD_FILTER"),
            ("C2_LEGACY",),
        )
        self.assertEqual(
            variant_planner_call_types("V1_C2_COMPLETE_DOMAIN"),
            ("C2_COMPLETE",),
        )
        self.assertEqual(
            variant_planner_call_types("V2_C1_C2_EXACT_PROGRAM_UNION"),
            ("C1_COMPLETE", "C2_COMPLETE"),
        )
        with self.assertRaisesRegex(ValueError, "unknown_variant"):
            variant_planner_call_types("V3")

    def test_union_has_truthful_artifact_arm(self):
        self.assertEqual(
            variant_artifact_arm("V2_C1_C2_EXACT_PROGRAM_UNION"),
            "C1_C2_EXACT_PROGRAM_UNION",
        )
        self.assertEqual(
            variant_artifact_arm("V1_C2_COMPLETE_DOMAIN"),
            "C2_ROLE_RETRIEVAL",
        )

    def test_development_gold_parser_preserves_multi_answer_labels(self):
        self.assertEqual(
            development_gold_answers(
                {"id": "S1", "labels": {"answer": [1, "one"]}}
            ),
            [1, "one"],
        )
        for invalid in (
            {"id": "S1", "answer": [1]},
            {"id": "S1", "labels": {"answer": []}},
            {"id": "S1", "labels": {"answer": "1"}},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                development_gold_answers(invalid)


if __name__ == "__main__":
    unittest.main()
