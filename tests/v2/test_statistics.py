from __future__ import annotations

from certa.v2.statistics import (
    clopper_pearson,
    paired_metrics,
    table_clustered_bootstrap,
    validation_unlock,
)


ROWS = [
    {"table_id": "t1", "b0_correct": False, "selected_correct": True, "changed": True},
    {"table_id": "t2", "b0_correct": False, "selected_correct": True, "changed": True},
    {"table_id": "t3", "b0_correct": True, "selected_correct": False, "changed": True},
    {"table_id": "t4", "b0_correct": True, "selected_correct": True, "changed": False},
    {"table_id": "t5", "b0_correct": False, "selected_correct": False, "changed": True},
]


def test_paired_metrics_and_exact_commit_interval() -> None:
    metrics = paired_metrics(ROWS)
    assert metrics["WC"] == 2
    assert metrics["CW"] == 1
    assert metrics["CC"] == 1
    assert metrics["changed_WW"] == 1
    assert metrics["selected_correct"] == 3
    assert metrics["b0_correct"] == 2
    assert metrics["commit_count"] == 4
    assert metrics["correct_commit_count"] == 2
    assert metrics["commit_precision"] == 0.5
    assert metrics["commit_precision_interval"] == clopper_pearson(2, 4)


def test_table_bootstrap_is_seeded_and_reports_paired_gain() -> None:
    first = table_clustered_bootstrap(ROWS, replicates=1000, seed=1729)
    second = table_clustered_bootstrap(ROWS, replicates=1000, seed=1729)
    assert first == second
    assert first["observed_gain"] == 0.2
    assert first["replicates"] == 1000


def test_validation_unlock_is_exactly_mechanical() -> None:
    metrics = paired_metrics(ROWS)
    assert validation_unlock(
        metrics,
        registry_external_commits=0,
        validator_bypassed_commits=0,
        hashes_frozen=True,
    )["unlocked"] is True
    assert validation_unlock(
        {**metrics, "WC": 1},
        registry_external_commits=0,
        validator_bypassed_commits=0,
        hashes_frozen=True,
    )["unlocked"] is False
