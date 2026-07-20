import json
import stat
import tempfile
import unittest
from pathlib import Path

import jsonschema

from tools.certa_egra_artifacts import (
    prepare_egra_cohorts,
    sanitize_egra_source,
    seal_egra_gold,
)


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class CohortPreparationTests(unittest.TestCase):
    def test_hash_only_alias_safe_selection_and_gold_runtime_firewall(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw = root / "raw"
            raw.mkdir()
            dev_source = root / "dev.jsonl"
            train_source = root / "train.jsonl"
            historical = root / "historical.jsonl"
            output = root / "output"
            sealed = root / "sealed"
            dev_rows = []
            train_rows = []
            for split, rows, count in (("dev", dev_rows, 72), ("train", train_rows, 80)):
                for index in range(count):
                    table_id = f"{split}_{index}"
                    value = index if split == "dev" else index + 1000
                    if split == "train" and index < 8:
                        value = index
                    table = {
                        "title": f"ignored {table_id}",
                        "texts": [["Name", "Value"], ["A", str(value)]],
                    }
                    (raw / f"{table_id}.json").write_text(json.dumps(table), encoding="utf-8")
                    rows.append({
                        "id": f"sample_{table_id}",
                        "table_id": table_id,
                        "table_source": "fixture",
                        "question": f"question {table_id}",
                        "answer": [str(value)],
                        "aggregation": ["none"],
                        "answer_formulas": ["=B2"],
                        "linked_cells": {"answer": [1, 1]},
                        "reference_cells_map": {"B2": "(1,1)"},
                    })
            dev_source.write_text("".join(json.dumps(row) + "\n" for row in dev_rows), encoding="utf-8")
            train_source.write_text("".join(json.dumps(row) + "\n" for row in train_rows), encoding="utf-8")
            historical.write_text(json.dumps({"sample_id": "old", "table_id": "dev_71"}) + "\n", encoding="utf-8")
            dev_identity = root / "dev_identity.jsonl"
            train_identity = root / "train_identity.jsonl"
            sanitize_egra_source(dev_source, dev_identity)
            sanitize_egra_source(train_source, train_identity)

            summary = prepare_egra_cohorts(
                dev_source=dev_identity,
                train_source=train_identity,
                table_root=raw,
                historical_cohort_paths=[historical],
                output_root=output,
                sealed_gold_root=sealed,
            )
            self.assertEqual(summary["dev_count"], 64)
            self.assertEqual(summary["holdout_count"], 64)
            manifest = json.loads((output / "freeze/COHORT_MANIFEST.json").read_text())
            schema = json.loads(
                (Path(__file__).resolve().parents[3]
                 / "certa_goal_packs/CERTA_EGRA_V0_CONSTRUCTION_AND_CONDITIONAL_DECISION_GATE_PACK/COHORT_MANIFEST_SCHEMA.json").read_text()
            )
            jsonschema.validate(manifest, schema)
            self.assertEqual(manifest["selection_fields"], ["sample_id", "table_id", "stable_hash"])
            self.assertFalse(set(manifest["dev_table_ids"]) & set(manifest["holdout_table_ids"]))
            self.assertNotIn("dev_71", manifest["dev_table_ids"] + manifest["holdout_table_ids"])

            dev_members = jsonl(output / "freeze/DEV_COHORT.jsonl")
            holdout_members = jsonl(output / "freeze/HOLDOUT_COHORT_SEALED.jsonl")
            self.assertFalse(
                {row["table_content_sha256"] for row in dev_members}
                & {row["table_content_sha256"] for row in holdout_members}
            )
            runtime = jsonl(output / "inputs/dev_runtime.jsonl")
            self.assertEqual(
                set(runtime[0]),
                {"dataset", "id", "question", "table_id", "table_source"},
            )
            runtime_text = (output / "inputs/dev_runtime.jsonl").read_text()
            for forbidden in (
                "answer_formulas",
                "linked_cells",
                "reference_cells_map",
                "aggregation",
                '"answer"',
            ):
                self.assertNotIn(forbidden, runtime_text)
            self.assertNotIn("question", (output / "freeze/HOLDOUT_COHORT_SEALED.jsonl").read_text())

            self.assertFalse((sealed / "dev_gold.jsonl").exists())
            self.assertFalse((sealed / "holdout_gold.jsonl").exists())
            sealed_summary = seal_egra_gold(
                dev_source=dev_source,
                train_source=train_source,
                output_root=output,
                sealed_gold_root=sealed,
            )
            repeated_summary = seal_egra_gold(
                dev_source=dev_source,
                train_source=train_source,
                output_root=output,
                sealed_gold_root=sealed,
            )
            self.assertEqual(sealed_summary, repeated_summary)
            for name in ("dev_gold.jsonl", "holdout_gold.jsonl"):
                path = sealed / name
                self.assertTrue(path.exists())
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o440)
                self.assertFalse(str(path).startswith(str(output)))

            reversed_output = root / "reversed_output"
            reversed_sealed = root / "reversed_sealed"
            reversed_dev = root / "dev_reversed.jsonl"
            reversed_train = root / "train_reversed.jsonl"
            reversed_dev.write_text("".join(json.dumps(row) + "\n" for row in reversed(dev_rows)), encoding="utf-8")
            reversed_train.write_text("".join(json.dumps(row) + "\n" for row in reversed(train_rows)), encoding="utf-8")
            reversed_dev_identity = root / "dev_reversed_identity.jsonl"
            reversed_train_identity = root / "train_reversed_identity.jsonl"
            sanitize_egra_source(reversed_dev, reversed_dev_identity)
            sanitize_egra_source(reversed_train, reversed_train_identity)
            second = prepare_egra_cohorts(
                dev_source=reversed_dev_identity,
                train_source=reversed_train_identity,
                table_root=raw,
                historical_cohort_paths=[historical],
                output_root=reversed_output,
                sealed_gold_root=reversed_sealed,
            )
            self.assertEqual(summary["dev_fingerprint"], second["dev_fingerprint"])
            self.assertEqual(summary["holdout_fingerprint"], second["holdout_fingerprint"])


if __name__ == "__main__":
    unittest.main()
