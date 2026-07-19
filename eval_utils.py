"""
eval_utils.py — CSCR Phase 0: 四口径 EM 评估工具

提供四种独立的 Exact Match 评估口径：
1. strict_em:   纯文本精确匹配（小写 + 去空白/标点）
2. numeric_em:  数值感知匹配（允许 52.1 == "52.1%"，但不做 pred/100≈gold 自动转换）
3. set_em:      集合匹配（多答案问题，分割后比较 frozenset）
4. hitab_official_em: HiTab 标注匹配（denotation 级：数值用容差比较，字符串用规范化比较）
"""

import argparse
import json
import math
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# 文本规范化
# ---------------------------------------------------------------------------

def normalize_text(value: Any) -> str:
    """基本规范化：小写、去标点、去货币符号、合并空白"""
    text = "" if value is None else str(value)
    text = text.lower().strip()
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = re.sub(r"[\$\u00a3\u20ac]", "", text)
    text = re.sub(r"[\[\]\(\)\{\}\"'`]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .,:;")



# ---------------------------------------------------------------------------
# 数值解析
# ---------------------------------------------------------------------------

def parse_number_strict(value: Any) -> Optional[float]:
    """严格数值解析：仅当字符串字面包含 '%' 时才除以 100"""
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = str(value).strip()
    # 移除逗号
    text = text.replace(",", "")
    # 匹配数值（可能带 % 后缀）
    match = re.match(r"^[-+]?\d*\.?\d+\s*(%?)$", text)
    if not match:
        return None
    is_percent = bool(match.group(1))
    try:
        number = float(text.rstrip("% "))
    except ValueError:
        return None
    if is_percent:
        return number / 100.0
    return number


def parse_number_raw(value: Any) -> Optional[float]:
    """原始数值解析：不做百分比转换，只提取数值"""
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = str(value).strip().replace(",", "")
    match = re.search(r"[-+]?\d*\.?\d+", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# 集合分割
# ---------------------------------------------------------------------------

def split_answer_set(value: Any) -> List[str]:
    """将答案分割为规范化元素列表"""
    if isinstance(value, list):
        raw_items = value
    else:
        raw_items = re.split(r"\s*,\s*|\s*;\s*|\s+/\s+|\s+\band\b\s+", str(value))
    return [normalize_text(item) for item in raw_items if normalize_text(item)]


# ---------------------------------------------------------------------------
# 四口径 EM 函数
# ---------------------------------------------------------------------------

def strict_em(prediction: Any, gold: Any) -> bool:
    """口径 1：纯文本精确匹配"""
    gold_values = gold if isinstance(gold, list) else [gold]
    pred_text = normalize_text(prediction)
    gold_texts = [normalize_text(g) for g in gold_values]
    # 单值匹配
    if pred_text in gold_texts:
        return True
    # 多值：合并为逗号分隔串
    if len(gold_values) > 1:
        gold_merged = ", ".join(sorted(normalize_text(g) for g in gold_values))
        pred_merged = ", ".join(sorted(split_answer_set(prediction)))
        if gold_merged == pred_merged:
            return True
    return False


def numeric_em(prediction: Any, gold: Any) -> bool:
    """口径 2：数值感知匹配。仅在 gold 或 pred 的原始字符串包含 '%' 时做百分比转换"""
    gold_values = gold if isinstance(gold, list) else [gold]
    for g in gold_values:
        pred_num = parse_number_strict(prediction)
        gold_num = parse_number_strict(g)
        if pred_num is not None and gold_num is not None:
            if math.isclose(pred_num, gold_num, rel_tol=1e-4, abs_tol=1e-6):
                return True
        # 同时尝试原始数值（不做百分比转换）
        pred_raw = parse_number_raw(prediction)
        gold_raw = parse_number_raw(g)
        if pred_raw is not None and gold_raw is not None:
            if math.isclose(pred_raw, gold_raw, rel_tol=1e-4, abs_tol=1e-6):
                return True
    return False


def set_em(prediction: Any, gold: Any) -> bool:
    """口径 3：集合匹配"""
    gold_values = gold if isinstance(gold, list) else [gold]
    pred_set = frozenset(split_answer_set(prediction))
    gold_set = frozenset(split_answer_set(gold_values))
    if not pred_set or not gold_set:
        return False
    return pred_set == gold_set


def hitab_official_em(prediction: Any, gold: Any) -> bool:
    """口径 4：HiTab denotation 匹配

    规则：
    1. 如果 gold 是数值列表且只有一个元素，比较数值
    2. 如果 gold 是字符串列表且只有一个元素，比较规范化文本
    3. 如果 gold 有多个元素，按集合比较
    4. 数值比较容差 rel_tol=1e-4
    """
    gold_values = gold if isinstance(gold, list) else [gold]

    # 尝试数值匹配（denotation 级）
    if len(gold_values) == 1:
        g = gold_values[0]
        # 数值匹配
        gold_num = parse_number_raw(g)
        pred_num = parse_number_raw(prediction)
        if gold_num is not None and pred_num is not None:
            if math.isclose(pred_num, gold_num, rel_tol=1e-4, abs_tol=1e-6):
                return True
        # 文本匹配
        if normalize_text(prediction) == normalize_text(g):
            return True
        return False

    # 多值集合匹配
    pred_set = frozenset(split_answer_set(prediction))
    gold_set = frozenset(split_answer_set(gold_values))
    if pred_set and gold_set and pred_set == gold_set:
        return True

    # 数值集合匹配
    pred_nums = set()
    for p in split_answer_set(prediction):
        n = parse_number_raw(p)
        if n is not None:
            pred_nums.add(round(n, 6))
    gold_nums = set()
    for g in gold_values:
        n = parse_number_raw(g)
        if n is not None:
            gold_nums.add(round(n, 6))
    if pred_nums and gold_nums and pred_nums == gold_nums:
        return True

    return False


# ---------------------------------------------------------------------------
# 综合评估
# ---------------------------------------------------------------------------

def evaluate_single(prediction: Any, gold_answer: Any, aggregation: Any = None) -> Dict[str, Any]:
    """评估单个样本，返回四口径结果"""
    result = {
        "strict_em": strict_em(prediction, gold_answer),
        "numeric_em": numeric_em(prediction, gold_answer),
        "set_em": set_em(prediction, gold_answer),
        "hitab_official_em": hitab_official_em(prediction, gold_answer),
    }
    result["is_correct_any"] = any(result.values())
    return result


def classify_error(prediction: Any, gold: Any, aggregation: Any = None) -> str:
    """错误分类：将不正确的预测归类为错误类型

    类别：
    - format_error: 数值正确但格式不同（52.1 vs 52.1%）
    - computation_error: 涉及计算的聚合类型答案错误
    - binding_error: lookup 类但值不在表格的 gold 行列
    - semantic_error: 文本语义不匹配（如 "women of african ancestry" vs "africa"）
    - correct: 答案正确
    """
    if hitab_official_em(prediction, gold):
        return "correct"

    gold_values = gold if isinstance(gold, list) else [gold]

    # 格式错误：原始数值相同但字符串不同
    pred_raw = parse_number_raw(prediction)
    for g in gold_values:
        gold_raw = parse_number_raw(g)
        if pred_raw is not None and gold_raw is not None:
            if math.isclose(pred_raw, gold_raw, rel_tol=0.05):
                return "format_error"

    # 计算错误：聚合类型问题
    agg = aggregation if isinstance(aggregation, list) else [aggregation] if aggregation else ["none"]
    agg_str = agg[0] if agg else "none"
    if agg_str in ("diff", "sum", "ratio", "percentage", "count", "average"):
        return "computation_error"

    # 语义错误 vs 绑定错误
    pred_text = normalize_text(prediction)
    gold_texts = [normalize_text(g) for g in gold_values]

    # 如果 pred 包含 gold 的子串或反之，可能是语义错误
    for gt in gold_texts:
        if gt in pred_text or pred_text in gt:
            return "semantic_error"

    # 如果都是数值但差异大，是绑定错误（访问了错误的单元格）
    if pred_raw is not None and any(parse_number_raw(g) is not None for g in gold_values):
        return "binding_error"

    return "semantic_error"


# ---------------------------------------------------------------------------
# 批量评估
# ---------------------------------------------------------------------------

def evaluate_batch(predictions: List[Dict]) -> Dict[str, Any]:
    """批量评估，返回四口径聚合指标"""
    calibers = ["strict_em", "numeric_em", "set_em", "hitab_official_em"]
    counts = {c: 0 for c in calibers}
    total = 0
    error_types = {}
    per_agg = {}  # per aggregation type breakdown

    for pred in predictions:
        final_answer = pred.get("answers", {}).get("final", "")
        if not final_answer and "prediction" in pred:
            final_answer = pred["prediction"]
        gold = pred.get("expected_answer", pred.get("answer", []))
        aggregation = pred.get("aggregation", ["none"])

        result = evaluate_single(final_answer, gold, aggregation)
        total += 1

        for c in calibers:
            if result[c]:
                counts[c] += 1

        # 错误分类
        err = classify_error(final_answer, gold, aggregation)
        error_types[err] = error_types.get(err, 0) + 1

        # Per-aggregation breakdown
        agg_key = aggregation[0] if isinstance(aggregation, list) and aggregation else str(aggregation)
        if agg_key not in per_agg:
            per_agg[agg_key] = {c: {"correct": 0, "total": 0} for c in calibers}
        for c in calibers:
            per_agg[agg_key][c]["total"] += 1
            if result[c]:
                per_agg[agg_key][c]["correct"] += 1

    metrics = {
        "total": total,
        "calibers": {},
        "error_taxonomy": error_types,
        "per_aggregation": {},
    }
    for c in calibers:
        metrics["calibers"][c] = {
            "correct": counts[c],
            "accuracy": counts[c] / total if total else 0.0,
        }
    for agg_key, agg_data in per_agg.items():
        metrics["per_aggregation"][agg_key] = {}
        for c in calibers:
            t = agg_data[c]["total"]
            cor = agg_data[c]["correct"]
            metrics["per_aggregation"][agg_key][c] = {
                "correct": cor,
                "total": t,
                "accuracy": cor / t if t else 0.0,
            }
    return metrics


def recalculate_metrics(predictions_jsonl_path: str, output_path: str) -> Dict[str, Any]:
    """重新评估已有的 predictions.jsonl，输出四口径指标"""
    predictions = []
    with open(predictions_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                predictions.append(json.loads(line))

    metrics = evaluate_batch(predictions)
    metrics["source_file"] = predictions_jsonl_path
    metrics["recalculated_at"] = __import__("datetime").datetime.now().isoformat()

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"Recalculated metrics for {len(predictions)} predictions:")
    for c, v in metrics["calibers"].items():
        print(f"  {c}: {v['correct']}/{metrics['total']} = {v['accuracy']:.4f}")
    print(f"  Error taxonomy: {metrics['error_taxonomy']}")

    return metrics


# ---------------------------------------------------------------------------
# Pipeline 兼容别名
# ---------------------------------------------------------------------------

def evaluate_answer_multi_caliber(prediction: Any, gold_answer: Any,
                                   aggregation: Any = None) -> Dict[str, Any]:
    """evaluate_single 的别名 (run_cscr_pipeline.py 使用此名称)"""
    return evaluate_single(prediction, gold_answer, aggregation)


def batch_evaluate(predictions: List[Dict],
                   answer_key: str = "final_answer",
                   gold_key: str = "gold_answer") -> Dict[str, Any]:
    """带自定义 key 的批量评估包装器

    与 evaluate_batch 不同，此函数接受 answer_key/gold_key 参数，
    适配 run_cscr_pipeline.py 的数据格式。
    """
    calibers = ["strict_em", "numeric_em", "set_em", "hitab_official_em"]
    counts = {c: 0 for c in calibers}
    total = 0
    error_types: Dict[str, int] = {}

    for pred in predictions:
        final_answer = pred.get(answer_key, "")
        gold = pred.get(gold_key, pred.get("answer", []))
        aggregation = pred.get("aggregation", ["none"])

        result = evaluate_single(final_answer, gold, aggregation)
        total += 1

        for c in calibers:
            if result[c]:
                counts[c] += 1

        err = classify_error(final_answer, gold, aggregation)
        error_types[err] = error_types.get(err, 0) + 1

    metrics: Dict[str, Any] = {
        "total": total,
        "error_taxonomy": error_types,
    }
    for c in calibers:
        metrics[f"{c}_count"] = counts[c]
        metrics[f"{c}_rate"] = counts[c] / total if total else 0.0

    return metrics


def compute_calibration_metrics(
    confidences: Sequence[float],
    correctness: Sequence[bool],
    n_bins: int = 10,
) -> Dict[str, float]:
    """计算校准指标: ECE, Brier score, Selective AUC

    用于 run_cscr_pipeline.py 中的校准评估。
    """
    n = len(confidences)
    if n == 0:
        return {"ece": 0.0, "brier": 0.0, "selective_auc": 0.0}

    # ECE (Expected Calibration Error)
    bins: List[List[Tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for conf, corr in zip(confidences, correctness):
        idx = min(int(conf * n_bins), n_bins - 1)
        bins[idx].append((conf, corr))

    ece = 0.0
    for b in bins:
        if not b:
            continue
        avg_conf = sum(c for c, _ in b) / len(b)
        avg_acc = sum(int(c) for _, c in b) / len(b)
        ece += abs(avg_conf - avg_acc) * len(b) / n

    # Brier score
    brier = sum((conf - int(corr)) ** 2 for conf, corr in zip(confidences, correctness)) / n

    # Selective AUC (area under accuracy vs coverage curve)
    indexed = sorted(enumerate(zip(confidences, correctness)),
                     key=lambda x: -x[1][0])
    cumulative_correct = 0
    selective_auc = 0.0
    for i, (_, (conf, corr)) in enumerate(indexed):
        cumulative_correct += int(corr)
        coverage = (i + 1) / n
        accuracy = cumulative_correct / (i + 1)
        selective_auc += accuracy / n

    return {
        "ece": round(ece, 6),
        "brier": round(brier, 6),
        "selective_auc": round(selective_auc, 6),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSCR Phase 0: 四口径 EM 评估")
    parser.add_argument("--predictions_file", required=True, help="Path to predictions.jsonl")
    parser.add_argument("--output_file", default=None, help="Output path for metrics JSON")
    args = parser.parse_args()

    if args.output_file is None:
        dirname = os.path.dirname(args.predictions_file)
        args.output_file = os.path.join(dirname, "metrics_recalculated.json")

    recalculate_metrics(args.predictions_file, args.output_file)
