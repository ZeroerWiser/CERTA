import unittest

from certa.active_v1.experiment_contract_v1 import (
    clopper_pearson_interval,
    compute_policy_metrics,
)


class ExperimentContractV1Tests(unittest.TestCase):
    def test_transition_and_unsafe_commit_invariants_are_exact(self):
        rows = [
            {"id": "cc", "table_id": "t1", "b0_correct": True,
             "selected_correct": True, "changed": False},
            {"id": "cw", "table_id": "t2", "b0_correct": True,
             "selected_correct": False, "changed": True},
            {"id": "wc", "table_id": "t3", "b0_correct": False,
             "selected_correct": True, "changed": True},
            {"id": "ww0", "table_id": "t4", "b0_correct": False,
             "selected_correct": False, "changed": False},
            {"id": "ww1", "table_id": "t5", "b0_correct": False,
             "selected_correct": False, "changed": True},
        ]
        metrics = compute_policy_metrics(rows)
        self.assertEqual(
            (metrics["CC"], metrics["CW"], metrics["WC"], metrics["WW"]),
            (1, 1, 1, 2),
        )
        self.assertEqual(metrics["changed_WW"], 1)
        self.assertEqual(metrics["commit_count"], 3)
        self.assertEqual(metrics["correct_commit_count"], 1)
        self.assertEqual(metrics["unsafe_commit_count"], 2)
        self.assertEqual(
            metrics["unsafe_commit_count"],
            metrics["CW"] + metrics["changed_WW"],
        )

    def test_duplicate_ids_and_inconsistent_changed_flags_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "duplicate_sample_id"):
            compute_policy_metrics([
                {"id": "x", "table_id": "t1", "b0_correct": True,
                 "selected_correct": True, "changed": False},
                {"id": "x", "table_id": "t2", "b0_correct": True,
                 "selected_correct": True, "changed": False},
            ])
        with self.assertRaisesRegex(ValueError, "boolean_field_invalid"):
            compute_policy_metrics([
                {"id": "x", "table_id": "t1", "b0_correct": 1,
                 "selected_correct": True, "changed": False},
            ])

    def test_clopper_pearson_boundaries_and_zero_denominator(self):
        self.assertEqual(clopper_pearson_interval(0, 0), {
            "numerator": 0, "denominator": 0, "estimate": None,
            "lower": None, "upper": None, "confidence": 0.95,
        })
        zero = clopper_pearson_interval(0, 6)
        self.assertEqual(zero["lower"], 0.0)
        self.assertGreater(zero["upper"], 0.0)
        full = clopper_pearson_interval(6, 6)
        self.assertEqual(full["upper"], 1.0)
        self.assertLess(full["lower"], 1.0)


if __name__ == "__main__":
    unittest.main()
