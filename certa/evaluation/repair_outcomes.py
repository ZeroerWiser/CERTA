"""Post-hoc repair outcome metrics for CERA shadow runs.

These helpers may read gold answers only after inference. They must not be
called from packet construction, candidate selection, prompt construction, or
validation.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping

from dataset_adapters import dataset_answer_match, normalize_dataset_name
from eval_utils import evaluate_answer_multi_caliber


def _answer_ok(answer: Any, gold: Any, dataset: str) -> bool:
    dataset = normalize_dataset_name(dataset)
    if dataset in {"aitqa", "tablebench"}:
        return bool(dataset_answer_match(dataset, gold, answer))
    return bool(evaluate_answer_multi_caliber(answer, gold).get("hitab_official_em", False))


def _candidate_answer(prediction: Mapping[str, Any]) -> str:
    output = prediction.get("cera_output")
    if isinstance(output, Mapping):
        value = output.get("final_answer")
        if value is not None:
            return str(value)
    return str(prediction.get("cera_proposed_repair_answer", "") or "")


def compute_repair_outcome(prediction: Mapping[str, Any], *, dataset: str = "") -> Dict[str, Any]:
    dataset_name = dataset or str(prediction.get("dataset", "hitab"))
    gold = prediction.get("gold_answer", prediction.get("expected_answer", ""))
    original_answer = str(prediction.get("cera_original_answer", prediction.get("llm_answer", "")) or "")
    repair_answer = _candidate_answer(prediction)
    would_commit = bool(prediction.get("cera_would_commit"))
    original_correct = _answer_ok(original_answer, gold, dataset_name) if str(gold) or isinstance(gold, list) else False
    proposed_repair_correct = (
        _answer_ok(repair_answer, gold, dataset_name)
        if would_commit and repair_answer and (str(gold) or isinstance(gold, list))
        else False
    )
    repair_gain = bool(would_commit and not original_correct and proposed_repair_correct)
    repair_loss = bool(would_commit and original_correct and not proposed_repair_correct)
    unsafe_accept = bool(would_commit and not proposed_repair_correct)
    wrong_to_wrong = bool(would_commit and not original_correct and not proposed_repair_correct)
    safe_keep = bool(not would_commit and original_correct)
    return {
        "cera_original_correct": original_correct,
        "cera_proposed_repair_correct": proposed_repair_correct,
        "cera_repair_gain": repair_gain,
        "cera_repair_loss": repair_loss,
        "cera_wrong_to_wrong": wrong_to_wrong,
        "cera_safe_keep": safe_keep,
        "cera_unsafe_accept": unsafe_accept,
        "cera_net_shadow_delta_em": (1 if repair_gain else 0) - (1 if repair_loss else 0),
    }


def aggregate_repair_outcomes(predictions: Iterable[Mapping[str, Any]], *, dataset: str = "") -> Dict[str, Any]:
    rows = [p for p in predictions if p.get("cera_enabled")]
    if not rows:
        return {}
    outcomes = [compute_repair_outcome(row, dataset=dataset or str(row.get("dataset", "hitab"))) for row in rows]
    count = len(outcomes)
    def _sum(key: str) -> int:
        return sum(1 for item in outcomes if item.get(key))
    gain = _sum("cera_repair_gain")
    loss = _sum("cera_repair_loss")
    return {
        "posthoc_outcome_count": count,
        "original_correct_count": _sum("cera_original_correct"),
        "proposed_repair_correct_count": _sum("cera_proposed_repair_correct"),
        "repair_gain_count": gain,
        "repair_loss_count": loss,
        "wrong_to_wrong_count": _sum("cera_wrong_to_wrong"),
        "safe_keep_count": _sum("cera_safe_keep"),
        "unsafe_accept_count": _sum("cera_unsafe_accept"),
        "net_shadow_delta_em": gain - loss,
        "net_shadow_delta_em_rate": (gain - loss) / count if count else 0.0,
    }
