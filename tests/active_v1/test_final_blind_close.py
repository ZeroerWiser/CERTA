import unittest

from tools.certa_final_blind_close import exact_runtime_master_join


class FinalBlindCloseTests(unittest.TestCase):
    def test_exact_ordered_join_and_gold_firewall(self):
        runtime = [{"id": "S1", "table_id": "T1"}]
        master = [{"sample_id": "S1", "table_id": "T1"}]
        exact_runtime_master_join(runtime, master)
        with self.assertRaisesRegex(ValueError, "ordered_id_mismatch"):
            exact_runtime_master_join(
                runtime,
                [{"sample_id": "S2", "table_id": "T1"}],
            )
        with self.assertRaisesRegex(ValueError, "blind_master_gold_field"):
            exact_runtime_master_join(
                runtime,
                [{"sample_id": "S1", "table_id": "T1", "labels": {}}],
            )

    def test_duplicate_ids_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "runtime_identity_not_unique_complete"):
            exact_runtime_master_join(
                [
                    {"id": "S1", "table_id": "T1"},
                    {"id": "S1", "table_id": "T2"},
                ],
                [
                    {"sample_id": "S1", "table_id": "T1"},
                    {"sample_id": "S2", "table_id": "T2"},
                ],
            )


if __name__ == "__main__":
    unittest.main()
