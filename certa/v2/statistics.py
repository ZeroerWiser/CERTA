"""Paired, exact, and table-clustered CERTA V2 statistics."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Mapping, Sequence

from scipy.stats import beta, binomtest


def clopper_pearson(
    successes: int, trials: int, alpha: float = 0.05
) -> dict[str, Any]:
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("invalid_binomial_counts")
    lower = 0.0 if successes == 0 else float(
        beta.ppf(alpha / 2, successes, trials - successes + 1)
    )
    upper = 1.0 if successes == trials else float(
        beta.ppf(1 - alpha / 2, successes + 1, trials - successes)
    )
    return {
        "successes": successes,
        "trials": trials,
        "alpha": alpha,
        "lower": lower,
        "upper": upper,
    }


def paired_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("paired_rows_empty")
    wc = [row for row in rows if not row["b0_correct"] and row["selected_correct"]]
    cw = [row for row in rows if row["b0_correct"] and not row["selected_correct"]]
    commits = [row for row in rows if row["changed"]]
    correct_commits = [row for row in commits if row["selected_correct"]]
    count = len(rows)
    result = {
        "rows": count,
        "CC": sum(row["b0_correct"] and row["selected_correct"] for row in rows),
        "CW": len(cw),
        "WC": len(wc),
        "unchanged_WW": sum(
            not row["b0_correct"]
            and not row["selected_correct"]
            and not row["changed"]
            for row in rows
        ),
        "changed_WW": sum(
            not row["b0_correct"]
            and not row["selected_correct"]
            and row["changed"]
            for row in rows
        ),
        "b0_correct": sum(bool(row["b0_correct"]) for row in rows),
        "selected_correct": sum(bool(row["selected_correct"]) for row in rows),
        "commit_count": len(commits),
        "correct_commit_count": len(correct_commits),
        "wc_table_count": len({str(row["table_id"]) for row in wc}),
    }
    result["b0_accuracy"] = result["b0_correct"] / count
    result["selected_accuracy"] = result["selected_correct"] / count
    result["accuracy_gain"] = (
        result["selected_accuracy"] - result["b0_accuracy"]
    )
    result["commit_precision"] = (
        len(correct_commits) / len(commits) if commits else None
    )
    result["commit_precision_interval"] = clopper_pearson(
        len(correct_commits), len(commits)
    )
    result["mcnemar_exact_p"] = (
        float(binomtest(min(len(wc), len(cw)), len(wc) + len(cw), 0.5).pvalue)
        if wc or cw
        else 1.0
    )
    return result


def table_clustered_bootstrap(
    rows: Sequence[Mapping[str, Any]], *, replicates: int = 10000, seed: int = 1729
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["table_id"])].append(row)
    tables = sorted(grouped)
    if not tables or replicates <= 0:
        raise ValueError("bootstrap_inputs_invalid")
    rng = random.Random(seed)
    gains = []
    for _ in range(replicates):
        sampled = [rng.choice(tables) for _ in tables]
        draw = [row for table in sampled for row in grouped[table]]
        gains.append(
            sum(bool(row["selected_correct"]) - bool(row["b0_correct"]) for row in draw)
            / len(draw)
        )
    ordered = sorted(gains)
    quantile = lambda p: ordered[min(len(ordered) - 1, int(p * len(ordered)))]
    observed = sum(
        bool(row["selected_correct"]) - bool(row["b0_correct"]) for row in rows
    ) / len(rows)
    return {
        "seed": seed,
        "replicates": replicates,
        "table_count": len(tables),
        "observed_gain": observed,
        "percentile_95": [quantile(0.025), quantile(0.975)],
    }


def validation_unlock(
    metrics: Mapping[str, Any],
    *,
    registry_external_commits: int,
    validator_bypassed_commits: int,
    hashes_frozen: bool,
) -> dict[str, Any]:
    checks = {
        "selected_accuracy_gt_b0": metrics["selected_accuracy"]
        > metrics["b0_accuracy"],
        "wc_gt_cw": metrics["WC"] > metrics["CW"],
        "at_least_two_wc": metrics["WC"] >= 2,
        "wc_spans_two_tables": metrics["wc_table_count"] >= 2,
        "registry_external_zero": registry_external_commits == 0,
        "validator_bypassed_zero": validator_bypassed_commits == 0,
        "hashes_frozen": hashes_frozen,
    }
    return {"unlocked": all(checks.values()), "checks": checks}
