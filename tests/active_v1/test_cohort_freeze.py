import json
import tempfile
import unittest
from pathlib import Path

from certa.active_v1.cohort import ACTIVE_COHORT_SEED, select_active_cohorts


class ActiveCohortFreezeTests(unittest.TestCase):
    def test_selection_is_label_free_alias_safe_and_table_disjoint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tables = root / "tables"
            tables.mkdir()
            dev = []
            train = []
            for split, rows, offset in (("dev", dev, 0), ("train", train, 1000)):
                for index in range(72):
                    table_id = f"{split}_{index}"
                    value = index + offset
                    if split == "train" and index < 4:
                        value = index
                    (tables / f"{table_id}.json").write_text(
                        json.dumps({"title": table_id, "texts": [["Value"], [str(value)]]}),
                        encoding="utf-8",
                    )
                    rows.append({
                        "dataset": "hitab",
                        "id": f"sample_{table_id}",
                        "question": f"question {table_id}",
                        "table_id": table_id,
                        "table_source": "fixture",
                    })
            historical = [{"sample_id": "old", "table_id": "dev_71"}]

            first = select_active_cohorts(dev, train, tables, historical)
            second = select_active_cohorts(list(reversed(dev)), list(reversed(train)), tables, historical)

            self.assertEqual(first, second)
            self.assertEqual(first["seed"], ACTIVE_COHORT_SEED)
            self.assertEqual(len(first["dev"]), 64)
            self.assertEqual(len(first["holdout"]), 64)
            self.assertEqual(first["integration16"], first["dev"][:16])
            self.assertNotIn("dev_71", {row["table_id"] for row in first["dev"]})
            self.assertFalse(
                {row["table_id"] for row in first["dev"]}
                & {row["table_id"] for row in first["holdout"]}
            )
            self.assertFalse(
                {row["table_content_sha256"] for row in first["dev"]}
                & {row["table_content_sha256"] for row in first["holdout"]}
            )
            for row in first["dev"] + first["holdout"]:
                self.assertEqual(
                    set(row["runtime"]),
                    {"dataset", "id", "question", "table_id", "table_source"},
                )

    def test_forbidden_or_noncanonical_identity_fields_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tables = Path(temp_dir)
            row = {
                "dataset": "hitab", "id": "s", "question": "q",
                "table_id": "t", "table_source": "fixture", "answer": ["1"],
            }
            with self.assertRaisesRegex(ValueError, "identity_fields_mismatch"):
                select_active_cohorts([row], [], tables, [])


if __name__ == "__main__":
    unittest.main()
