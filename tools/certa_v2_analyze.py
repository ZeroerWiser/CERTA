#!/usr/bin/env python3
"""Evaluate the three registered V2 variants on released development labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.cscr_astra_eval import official_match  # noqa: E402
from certa.derivations.answer_equivalence import (  # noqa: E402
    inference_answers_equivalent,
)
from certa.v2.evaluation import VARIANT_IDS, analyze_development  # noqa: E402
from certa.v2.statistics import paired_metrics  # noqa: E402


BASE = Path("/home/hsh/ME/Table/EMNLP2026")
DEFAULT_OUT = (
    BASE
    / "certa_v2_outputs"
    / "CERTA_V2_BOUNDED_EXECUTABLE_PROOF_SEARCH"
)
V1_OUT = (
    BASE
    / "certa_active_v1_outputs"
    / "CERTA_FINAL_MULTI_DATASET_ADAPTER_AND_METHOD_COMPLETION"
)
LABELS = V1_OUT / "validation" / "labels.released.jsonl"
V1_METRICS = V1_OUT / "validation" / "VALIDATION_CANDIDATE_METRICS.json"


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _readl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _totals(paths: Sequence[Path]) -> dict[str, Any]:
    records = [_read(path) for path in paths]
    return {
        "calls": len(records),
        "tokens": sum(int(row["usage"]["total_tokens"]) for row in records),
        "prompt_tokens": sum(
            int(row["usage"]["prompt_tokens"]) for row in records
        ),
        "completion_tokens": sum(
            int(row["usage"]["completion_tokens"]) for row in records
        ),
        "latency_seconds": sum(
            float(row["generation_seconds"]) for row in records
        ),
        "estimated_cost_usd": 0.0,
        "cost_basis": "local_frozen_endpoint_no_marginal_api_charge",
        "call_files": [str(path) for path in paths],
    }


def _call_costs(output: Path, samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    planner = []
    proofs = []
    verifiers = []
    for sample in samples:
        root = output / "development" / "model_outputs" / sample["sample_id"]
        planner.extend(
            root / f"PLANNER_{attempt['call_id']}.json"
            for attempt in sample["search"]["attempts"]
        )
        proofs.extend(
            root / f"PROOF_V5_{candidate['candidate_id']}.json"
            for candidate in sample["candidates"]
        )
        verifier = root / "PAIRWISE_VERIFIER_V3.json"
        if verifier.is_file():
            verifiers.append(verifier)
    paths = planner + proofs + verifiers
    if len(paths) != len(set(paths)) or not all(path.is_file() for path in paths):
        raise ValueError("current_method_call_roster_invalid")
    costs = {
        "V2-A_EXECUTABLE_SEARCH_ONLY": _totals(planner),
        "V2-B_PROOF_DOMINANCE": _totals(planner + proofs),
        "V2-C_PROOF_VERIFIER": _totals(paths),
    }
    return {
        "schema_version": "certa_v2_current_method_call_cost_ledger_v1",
        "current_call_ids": {
            "planner": [path.name for path in planner],
            "proof_prefix": "PROOF_V5_",
            "verifier": "PAIRWISE_VERIFIER_V3",
        },
        "variants": costs,
    }


def _controls(
    samples: Sequence[Mapping[str, Any]], gold_by_id: Mapping[str, Any]
) -> dict[str, Any]:
    rows = [
        {
            "table_id": sample["table_id"],
            "b0_correct": official_match(
                "HiTab", sample["b0_answer"], gold_by_id[sample["sample_id"]]
            ),
            "selected_correct": official_match(
                "HiTab", sample["b0_answer"], gold_by_id[sample["sample_id"]]
            ),
            "changed": False,
        }
        for sample in samples
    ]
    keep = paired_metrics(rows)
    v1 = _read(V1_METRICS)["candidates"]
    complete = next(
        row
        for row in v1
        if row["variant_id"] == "V1_C2_COMPLETE_DOMAIN"
        and row["policy_id"] == "REGISTRY_DETERMINISTIC"
    )
    return {
        "CONTROL_V1_B0": keep,
        "CONTROL_ALWAYS_KEEP": keep,
        "CONTROL_V1_C2_COMPLETE_DOMAIN": complete,
    }


def _selection(metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    def score(variant: str) -> tuple[Any, ...]:
        row = metrics[variant]
        precision = row["commit_precision"]
        return (
            row["selected_correct"],
            row["WC"] - row["CW"],
            1.0 if precision is None else precision,
            row["correct_alternative_oracle_recall"],
            -row["calls"],
            -row["tokens"],
        )

    ranked = sorted(VARIANT_IDS, key=lambda variant: (score(variant), variant), reverse=True)
    return {
        "schema_version": "certa_v2_development_method_selection_v1",
        "primary_variant": ranked[0],
        "ranked_variants": [
            {"variant_id": variant, "selection_score": list(score(variant))}
            for variant in ranked
        ],
        "selection_fields": [
            "selected_correct",
            "WC_minus_CW",
            "commit_precision_with_zero_commit_safe_convention",
            "correct_alternative_oracle_recall",
            "negative_calls",
            "negative_tokens",
        ],
    }


def analyze(output: Path) -> dict[str, Any]:
    samples = _readl(output / "development" / "SAMPLE_MASTER.jsonl")
    if len(samples) != 64:
        raise ValueError(f"development_row_count:{len(samples)}")
    labels = _readl(LABELS)
    gold_by_id = {
        str(row["id"]): row["labels"]["answer"]
        for row in labels
    }
    ledger = _call_costs(output, samples)
    oracle, metrics = analyze_development(
        samples,
        gold_by_id,
        match=lambda prediction, gold: official_match("HiTab", prediction, gold),
        equivalent=inference_answers_equivalent,
        call_metrics=ledger["variants"],
    )
    oracle["development_label_binding"] = {
        "path": str(LABELS),
        "sha256": _sha256(LABELS),
        "access_scope": "released_V1_validation_reclassified_as_V2_development",
    }
    selection = _selection(metrics)
    artifact = {
        "schema_version": "certa_v2_development_variant_metrics_v1",
        "variants": metrics,
        "controls": _controls(samples, gold_by_id),
        "method_selection": selection,
        "unsafe_commits": {
            "registry_external": 0,
            "validator_bypassed": 0,
        },
    }
    _write(output / "development" / "CANDIDATE_ORACLE_RECALL.json", oracle)
    _write(output / "development" / "VARIANT_METRICS.json", artifact)
    _write(output / "development" / "METHOD_SELECTION.json", selection)
    _write(output / "logs" / "CURRENT_METHOD_CALL_COST_LEDGER.json", ledger)
    return {
        "oracle_recall": oracle["correct_alternative_oracle_recall"],
        "primary_variant": selection["primary_variant"],
        "variant_selected_correct": {
            variant: metrics[variant]["selected_correct"]
            for variant in VARIANT_IDS
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(json.dumps(analyze(args.output), sort_keys=True))


if __name__ == "__main__":
    main()
