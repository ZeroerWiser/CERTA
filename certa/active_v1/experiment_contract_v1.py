"""Frozen-ready transition and exact-interval contracts."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from scipy.stats import beta


def clopper_pearson_interval(
    numerator: int,
    denominator: int,
    *,
    confidence: float = 0.95,
) -> dict[str, Any]:
    if (
        type(numerator) is not int
        or type(denominator) is not int
        or numerator < 0
        or denominator < 0
        or numerator > denominator
    ):
        raise ValueError("binomial_count_invalid")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence_invalid")
    if denominator == 0:
        return {
            "numerator": numerator,
            "denominator": denominator,
            "estimate": None,
            "lower": None,
            "upper": None,
            "confidence": confidence,
        }
    alpha = 1.0 - confidence
    lower = (
        0.0
        if numerator == 0
        else float(beta.ppf(alpha / 2.0, numerator, denominator - numerator + 1))
    )
    upper = (
        1.0
        if numerator == denominator
        else float(beta.ppf(
            1.0 - alpha / 2.0,
            numerator + 1,
            denominator - numerator,
        ))
    )
    return {
        "numerator": numerator,
        "denominator": denominator,
        "estimate": numerator / denominator,
        "lower": lower,
        "upper": upper,
        "confidence": confidence,
    }


def compute_policy_metrics(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not rows:
        raise ValueError("metric_rows_empty")
    ids = [str(row.get("id") or "") for row in rows]
    if any(not value for value in ids):
        raise ValueError("sample_id_empty")
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate_sample_id")
    transitions = {"CC": 0, "CW": 0, "WC": 0, "WW": 0}
    changed_ww = 0
    changed_cc = 0
    commits = 0
    correct_commits = 0
    for row in rows:
        fields = ("b0_correct", "selected_correct", "changed")
        if any(type(row.get(field)) is not bool for field in fields):
            raise ValueError(f"boolean_field_invalid:{row.get('id') or ''}")
        b0 = row["b0_correct"]
        selected = row["selected_correct"]
        changed = row["changed"]
        transition = (
            "CC" if b0 and selected
            else "CW" if b0
            else "WC" if selected
            else "WW"
        )
        transitions[transition] += 1
        changed_ww += int(transition == "WW" and changed)
        changed_cc += int(transition == "CC" and changed)
        commits += int(changed)
        correct_commits += int(changed and selected)
    unsafe = commits - correct_commits
    if unsafe != transitions["CW"] + changed_ww:
        raise ValueError("unsafe_commit_transition_invariant_failed")
    return {
        "row_count": len(rows),
        "b0_correct": sum(bool(row["b0_correct"]) for row in rows),
        "selected_correct": sum(bool(row["selected_correct"]) for row in rows),
        **transitions,
        "changed_CC": changed_cc,
        "changed_WW": changed_ww,
        "commit_count": commits,
        "correct_commit_count": correct_commits,
        "unsafe_commit_count": unsafe,
        "commit_precision": clopper_pearson_interval(
            correct_commits, commits,
        ),
        "unsafe_commit_rate": clopper_pearson_interval(
            unsafe, commits,
        ),
    }
