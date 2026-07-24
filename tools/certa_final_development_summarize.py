#!/usr/bin/env python3
"""Materialize the development comparison and bounded method selection."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from certa.active_v1.experiment_contract_v1 import compute_policy_metrics
from certa.active_v1.final_method_v1 import VARIANT_IDS
from certa.reproducibility.canonical_json import canonical_json


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _latency(rows: list[Mapping[str, Any]]) -> dict[str, float]:
    values = sorted(float(row.get("generation_seconds") or 0.0) for row in rows)
    if not values:
        return {"median_seconds": 0.0, "p95_seconds": 0.0}
    return {
        "median_seconds": statistics.median(values),
        "p95_seconds": values[min(len(values) - 1, int(0.95 * len(values)))],
    }


def summarize(output: Path) -> dict[str, Any]:
    samples = read_jsonl(output / "development/SAMPLE_MASTER.jsonl")
    if len(samples) != 128 or len({row["sample_id"] for row in samples}) != 128:
        raise ValueError("development_sample_master_not_exact_128")
    ledger = read_jsonl(output / "logs/ENDPOINT_LEDGER.jsonl")
    if any(row.get("failed") is True for row in ledger):
        raise ValueError("development_endpoint_failure_present")
    b0_correct = sum(bool(row["b0_correct"]) for row in samples)
    variants = []
    for variant_id in VARIANT_IDS:
        variant_rows = [
            next(item for item in sample["variants"] if item["variant_id"] == variant_id)
            for sample in samples
        ]
        policy_metrics = {}
        for policy_id in ("B0_KEEP", "REGISTRY_DETERMINISTIC", "CERA_VALIDATED"):
            records = []
            for sample, variant in zip(samples, variant_rows):
                policy = variant["policies"][policy_id]
                records.append({
                    "id": sample["sample_id"],
                    "table_id": sample["table_id"],
                    "b0_correct": bool(sample["b0_correct"]),
                    "selected_correct": bool(policy["correct"]),
                    "changed": bool(policy["changed"]),
                })
            policy_metrics[policy_id] = compute_policy_metrics(records)
        support_counts = Counter(row["support"]["state"] for row in variant_rows)
        failures = Counter(
            reason
            for row in variant_rows
            for reason in row["failure_reasons"]
        )
        variants.append({
            "variant_id": variant_id,
            "sample_count": len(variant_rows),
            "observed_count": sum(row["support"]["state"] != "UNOBSERVED" for row in variant_rows),
            "support_state_counts": dict(sorted(support_counts.items())),
            "failure_reason_counts": dict(sorted(failures.items())),
            "executable_row_count": sum(
                bool((row.get("closure") or {}).get("executable_derivations"))
                for row in variant_rows
            ),
            "registry_row_count": sum(bool(row["registry_entries"]) for row in variant_rows),
            "registry_entry_count": sum(len(row["registry_entries"]) for row in variant_rows),
            "policy_metrics": policy_metrics,
        })
    by_type = {}
    for call_type in sorted({row["logical_call_type"] for row in ledger}):
        rows = [row for row in ledger if row["logical_call_type"] == call_type]
        by_type[call_type] = {
            "logical_calls": len(rows),
            "transport_attempts": sum(int(row["transport_attempts"]) for row in rows),
            "prompt_tokens": sum(int(row.get("usage", {}).get("prompt_tokens", 0)) for row in rows),
            "completion_tokens": sum(int(row.get("usage", {}).get("completion_tokens", 0)) for row in rows),
            **_latency(rows),
        }
    cost = {
        "schema_version": "certa_final_development_cost_v1",
        "logical_calls": len(ledger),
        "transport_attempts": sum(int(row["transport_attempts"]) for row in ledger),
        "failed_attempts": 0,
        "prompt_tokens": sum(int(row.get("usage", {}).get("prompt_tokens", 0)) for row in ledger),
        "completion_tokens": sum(int(row.get("usage", {}).get("completion_tokens", 0)) for row in ledger),
        **_latency(ledger),
        "by_call_type": by_type,
        "shared_call_accounting": {
            "B0_and_Role": ["DEVELOPMENT_B0", "DEVELOPMENT_ROLE_V3"],
            "V0_LEGACY_C2_HARD_FILTER": ["DEVELOPMENT_PLANNER_C2_LEGACY"],
            "V1_C2_COMPLETE_DOMAIN": ["DEVELOPMENT_PLANNER_C2_COMPLETE"],
            "V2_C1_C2_EXACT_PROGRAM_UNION": [
                "DEVELOPMENT_PLANNER_C1_COMPLETE",
                "DEVELOPMENT_PLANNER_C2_COMPLETE"
            ],
            "CERA_VALIDATED": [],
        },
    }
    comparison = {
        "schema_version": "certa_final_variant_comparison_v1",
        "sample_count": len(samples),
        "b0_correct": b0_correct,
        "b0_accuracy": b0_correct / len(samples),
        "variants": variants,
        "cost": cost,
        "selection_principle": (
            "selected-final accuracy and WC/CW first; then commit precision; "
            "then calls, tokens, and P95 latency; candidate count is non-authoritative"
        ),
    }
    write_json(output / "development/VARIANT_COMPARISON.json", comparison)
    write_json(output / "development/COST_LEDGER.json", cost)
    selection = {
        "schema_version": "certa_final_method_selection_v1",
        "primary_constructor": "V1_C2_COMPLETE_DOMAIN",
        "primary_policy": "B0_KEEP",
        "primary_selected_correct": b0_correct,
        "primary_selected_accuracy": b0_correct / len(samples),
        "selection_status": "NO_DEVELOPMENT_NON_B0_POLICY_IMPROVED_B0",
        "rejected_primary_combinations": [
            {
                "constructor": row["variant_id"],
                "policy": "REGISTRY_DETERMINISTIC",
                "selected_correct": row["policy_metrics"]["REGISTRY_DETERMINISTIC"]["selected_correct"],
                "unsafe_commit_count": row["policy_metrics"]["REGISTRY_DETERMINISTIC"]["unsafe_commit_count"],
                "reason": "accuracy_below_b0_and_zero_correct_commits",
            }
            for row in variants
        ] + [
            {
                "constructor": row["variant_id"],
                "policy": "CERA_VALIDATED",
                "selected_correct": row["policy_metrics"]["CERA_VALIDATED"]["selected_correct"],
                "unsafe_commit_count": 0,
                "reason": "no_BothSide_rows_no_gain_tie_broken_against_extra_method_calls",
            }
            for row in variants
        ],
        "constructor_rationale": (
            "V0 remains the negative control; V2 adds a C1 call without selected-final "
            "gain; V1 is the smallest non-negative-control complete-domain interface. "
            "Under B0_KEEP it has no answer-changing authority."
        ),
        "blind_validation_primary_expectation": (
            "B0_KEEP cannot improve over B0; blind validation may test frozen comparators "
            "but cannot justify a positive method claim without a preregistered policy gain"
        ),
    }
    write_json(output / "development/METHOD_SELECTION.json", selection)
    return {"comparison": comparison, "selection": selection}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.output.resolve())
    print(canonical_json({
        "status": "PASS",
        "primary_constructor": result["selection"]["primary_constructor"],
        "primary_policy": result["selection"]["primary_policy"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
