from __future__ import annotations

import json
from pathlib import Path

from tools.certa_v2_prepare import prepare_v2_state


BASE = Path("/home/hsh/ME/Table/EMNLP2026")
REPO = BASE / "CERTA"
RESIDUAL = (
    BASE
    / "certa_v2_outputs"
    / "CERTA_V2_BOUNDED_EXECUTABLE_PROOF_SEARCH"
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_prepare_reproduces_lineage_and_split_selection_without_labels(tmp_path: Path) -> None:
    result = prepare_v2_state(output=tmp_path, base=BASE, repo=REPO)
    expected_lineage = read_json(
        RESIDUAL / "lineage" / "V1_NEGATIVE_BASELINE_BINDING.json"
    )
    assert result["lineage"] == expected_lineage

    expected_split = read_json(RESIDUAL / "data" / "SPLIT_MANIFESTS.json")
    actual_split = result["split_manifests"]
    expected_split["fresh_validation"]["runtime"] = str(
        tmp_path / "data" / "fresh_validation_runtime.jsonl"
    )
    assert actual_split == expected_split
    assert (
        tmp_path / "data" / "fresh_validation_runtime.jsonl"
    ).read_bytes() == (
        RESIDUAL / "data" / "fresh_validation_runtime.jsonl"
    ).read_bytes()


def test_prepare_records_nonaccess_and_never_materializes_label_paths(
    tmp_path: Path,
) -> None:
    result = prepare_v2_state(output=tmp_path, base=BASE, repo=REPO)
    lineage = result["lineage"]
    split = result["split_manifests"]
    assert lineage["holdout_access"] == {
        "authorized": False,
        "holdout_output_files": 0,
        "labels_accessed": False,
        "model_calls": 0,
        "predictions_generated": False,
    }
    assert split["fresh_validation"]["labels_status"] == (
        "NOT_MATERIALIZED_OPERATOR_RELEASE_REQUIRED"
    )
    assert "labels" not in split["preserved_holdout"]
