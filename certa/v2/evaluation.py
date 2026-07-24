"""Registered CERTA V2 development evaluation."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Callable, Mapping, Sequence

from certa.v2.statistics import paired_metrics


VARIANT_IDS = (
    "V2-A_EXECUTABLE_SEARCH_ONLY",
    "V2-B_PROOF_DOMINANCE",
    "V2-C_PROOF_VERIFIER",
)


def _state_counts(values: Sequence[str]) -> dict[str, int]:
    counts = Counter(values)
    return {state: counts[state] for state in ("PASS", "FAIL", "UNKNOWN")}


def analyze_development(
    samples: Sequence[Mapping[str, Any]],
    gold_by_id: Mapping[str, Any],
    *,
    match: Callable[[Any, Any], bool],
    call_metrics: Mapping[str, Any],
    equivalent: Callable[[Any, Any], bool] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute label-separated candidate recall and registered variant metrics."""
    if not samples or {str(row["sample_id"]) for row in samples} != set(gold_by_id):
        raise ValueError("development_sample_label_roster_mismatch")
    equivalent = equivalent or (lambda left, right: left == right)
    b0_wrong = 0
    correct_alternative_rows = 0
    candidate_universe_correct = 0
    correct_alternative_classes = 0
    by_signature: dict[str, Counter[str]] = defaultdict(Counter)
    b0_states = []
    alternative_states = []
    expected_proofs = observed_proofs = proof_failures = 0
    paired: dict[str, list[dict[str, Any]]] = {
        variant: [] for variant in VARIANT_IDS
    }

    for sample in samples:
        sample_id = str(sample["sample_id"])
        gold = gold_by_id[sample_id]
        b0 = sample["b0_answer"]
        b0_correct = match(b0, gold)
        alternatives = [
            candidate
            for candidate in sample["candidates"]
            if candidate["candidate_source"] == "EXECUTED_REGISTRY"
        ]
        correct_alternatives = [
            candidate
            for candidate in alternatives
            if match(candidate["candidate_answer"], gold)
        ]
        signature = str(sample.get("role", {}).get("role_id") or "UNKNOWN")
        by_signature[signature]["rows"] += 1
        if not b0_correct:
            b0_wrong += 1
            by_signature[signature]["b0_wrong_rows"] += 1
            if correct_alternatives:
                correct_alternative_rows += 1
                by_signature[signature]["correct_alternative_rows"] += 1
        correct_alternative_classes += len(correct_alternatives)
        candidate_universe_correct += bool(b0_correct or correct_alternatives)

        expected_proofs += len(sample["candidates"])
        observed_proofs += len(sample["proofs"])
        proof_failures += len(sample.get("proof_failures", ()))
        for proof in sample["proofs"]:
            target = (
                b0_states
                if proof["candidate_source"] == "B0"
                else alternative_states
            )
            target.append(str(proof["overall_state"]))

        for variant in VARIANT_IDS:
            selected = sample["selected_finals"][variant]
            paired[variant].append(
                {
                    "sample_id": sample_id,
                    "table_id": str(sample["table_id"]),
                    "b0_correct": b0_correct,
                    "selected_correct": match(selected, gold),
                    "changed": not equivalent(selected, b0),
                    "b0_answer": b0,
                    "selected_answer": selected,
                }
            )

    oracle = {
        "schema_version": "certa_v2_candidate_oracle_recall_v1",
        "rows": len(samples),
        "b0_wrong_rows": b0_wrong,
        "correct_alternative_rows": correct_alternative_rows,
        "correct_alternative_classes": correct_alternative_classes,
        "correct_alternative_oracle_recall": (
            correct_alternative_rows / b0_wrong if b0_wrong else None
        ),
        "answer_class_recall": {
            "registry_alternative_rows": correct_alternative_rows,
            "registry_alternative_rate_all_rows": correct_alternative_rows
            / len(samples),
            "candidate_universe_correct_rows": candidate_universe_correct,
            "candidate_universe_recall": candidate_universe_correct / len(samples),
        },
        "by_signature": {
            signature: dict(counts)
            for signature, counts in sorted(by_signature.items())
        },
    }
    proof_summary = {
        "proof_coverage": {
            "expected": expected_proofs,
            "observed": observed_proofs,
            "rate": observed_proofs / expected_proofs,
            "contract_valid": observed_proofs - proof_failures,
            "contract_valid_rate": (observed_proofs - proof_failures)
            / expected_proofs,
            "fail_closed_responses": proof_failures,
        },
        "b0_proof_states": _state_counts(b0_states),
        "alternative_proof_states": _state_counts(alternative_states),
    }
    metrics = {}
    for variant in VARIANT_IDS:
        result = paired_metrics(paired[variant])
        costs = call_metrics[variant] if variant in call_metrics else call_metrics
        wc = result["WC"]
        result.update(
            proof_summary,
            correct_alternative_oracle_recall=oracle[
                "correct_alternative_oracle_recall"
            ],
            answer_class_recall=oracle["answer_class_recall"],
            calls=int(costs.get("calls", 0)),
            tokens=int(costs.get("tokens", 0)),
            latency_seconds=float(costs.get("latency_seconds", 0.0)),
            estimated_cost_usd=float(costs.get("estimated_cost_usd", 0.0)),
            calls_per_WC=(int(costs.get("calls", 0)) / wc if wc else None),
            tokens_per_WC=(int(costs.get("tokens", 0)) / wc if wc else None),
            latency_seconds_per_WC=(
                float(costs.get("latency_seconds", 0.0)) / wc if wc else None
            ),
            cost_per_WC_usd=(
                float(costs.get("estimated_cost_usd", 0.0)) / wc
                if wc
                else None
            ),
            transitions=paired[variant],
        )
        metrics[variant] = result
    return oracle, metrics
