"""v9.0 答案归一化与问题类型路由（轻量后处理）。

仅在 finalize_after_llm 之后、写出 prediction 之前调用。无 LLM 反馈，无参数训练。

- normalize_numeric_answer: 把 LLM 输出的 "4.2%", "2.3 times", "1,270",
  "$18.4 million" 等格式与 gold 的不同表示形态对齐，输出多个候选数值表达。
- align_to_gold_form: 给定 gold 的字符串形态，挑选与 gold 形态一致的候选。
  这只在评测层调和，**不会修改最终 final_answer**，但会在 prediction 中
  写入 normalized_candidates 字段，供 v9.x KG-fallback 与论文消融用。
- coarse_question_type: 在 QuestionAnalyzer 之上做更细粒度分类（lookup_cell
  / lookup_aggregate / arithmetic / compare / superlative / count / proportion /
  trend），用于 v9.0 question-type router。

为什么要这样做（理论锚点）：
诊断 E25-E38 的 373 个 universal_hard 样本时发现，约 25-30% 是数值格式或
单复数对齐导致的"假性错误"，从模型与 reachability 视角它们是确定性答案，
只是表达形态与 gold 不同。把这部分当作 v9.0 的 free lunch 处理，可以为
后续真正的 KG-fallback 留出更纯净的实验信号。
"""
import re
from typing import Any, List, Optional

_PERCENT_RE = re.compile(r"^([-+]?\d*\.?\d+)\s*%$")
_TIMES_RE = re.compile(r"^([-+]?\d*\.?\d+)\s*(?:times|x)$", re.IGNORECASE)
_MONEY_RE = re.compile(
    r"^[\$£€]?\s*([-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)\s*(million|billion|thousand|k|m|b)?$",
    re.IGNORECASE,
)
_PURE_NUMBER_RE = re.compile(r"^[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?$|^[-+]?\d+(?:\.\d+)?$")

_SCALE_MULTIPLIERS = {
    "thousand": 1_000.0,
    "k": 1_000.0,
    "million": 1_000_000.0,
    "m": 1_000_000.0,
    "billion": 1_000_000_000.0,
    "b": 1_000_000_000.0,
}


def _to_float(text: str) -> Optional[float]:
    text = text.replace(",", "").strip()
    try:
        return float(text)
    except ValueError:
        return None


def normalize_numeric_answer(answer: str) -> List[str]:
    """生成多种数值表达候选，按优先级倒序。

    例如 "4.2%" -> ["4.2%", "4.2", "0.042"]
        "2.3 times" -> ["2.3 times", "2.3"]
        "$18.4 million" -> ["18400000", "18.4", "18.4 million"]
    保持原值始终在第一位，避免破坏既有评测路径。
    """
    if not isinstance(answer, str):
        return [str(answer)]

    raw = answer.strip()
    if not raw:
        return [""]

    candidates: List[str] = [raw]

    m = _PERCENT_RE.match(raw)
    if m:
        val = _to_float(m.group(1))
        if val is not None:
            candidates.append(f"{val}")
            candidates.append(f"{val / 100.0}")
            candidates.append(f"{val / 100.0:.6f}".rstrip("0").rstrip("."))
        return _dedupe(candidates)

    m = _TIMES_RE.match(raw)
    if m:
        val = _to_float(m.group(1))
        if val is not None:
            candidates.append(f"{val}")
        return _dedupe(candidates)

    m = _MONEY_RE.match(raw)
    if m:
        base = _to_float(m.group(1))
        scale = (m.group(2) or "").lower()
        if base is not None:
            candidates.append(f"{base}")
            mul = _SCALE_MULTIPLIERS.get(scale)
            if mul is not None:
                expanded = base * mul
                # 同时保留 int 与 float 形态
                if abs(expanded - round(expanded)) < 1e-6:
                    candidates.append(str(int(round(expanded))))
                else:
                    candidates.append(f"{expanded}")
        return _dedupe(candidates)

    if _PURE_NUMBER_RE.match(raw):
        val = _to_float(raw)
        if val is not None:
            # 提供 0-100 与 0-1 的互译
            if 0.0 < val < 1.0:
                candidates.append(f"{val * 100.0}")
                candidates.append(f"{val * 100.0:.4f}".rstrip("0").rstrip("."))
            elif 1.0 <= val <= 100.0:
                candidates.append(f"{val / 100.0}")
                candidates.append(f"{val / 100.0:.6f}".rstrip("0").rstrip("."))
            candidates.append(f"{val}")
            if abs(val - round(val)) < 1e-6:
                candidates.append(str(int(round(val))))
        return _dedupe(candidates)

    return _dedupe(candidates)


def deployable_normalize_answer(answer: str, question: str = "", dataset: str = "") -> str:
    """Gold-free answer surface normalization for deployment.

    This only applies format-preserving rewrites that can be inferred from the
    answer/question text. It deliberately does not choose between scale variants
    such as 14 and 0.14 because that requires evaluator/gold knowledge on HiTab.
    """
    if not isinstance(answer, str):
        return str(answer)

    raw = answer.strip()
    if not raw:
        return raw

    dataset_key = (dataset or "").strip().lower().replace("_", "-")
    q = (question or "").lower()

    cleaned = re.sub(r"^(?:final\s+answer|answer)\s*:\s*", "", raw, flags=re.IGNORECASE).strip()
    cleaned = cleaned.strip(" \t\r\n\"'")
    if cleaned != raw:
        raw = cleaned

    m = _PERCENT_RE.match(raw)
    if m and re.search(r"\b(?:percent|percentage|proportion|share|ratio|rate)\b", q):
        val = _to_float(m.group(1))
        if val is not None:
            return f"{val}".rstrip("0").rstrip(".") if "." in f"{val}" else f"{val}"

    m = _TIMES_RE.match(raw)
    if m and "how many" in q:
        val = _to_float(m.group(1))
        if val is not None:
            return f"{val}".rstrip("0").rstrip(".") if "." in f"{val}" else f"{val}"

    if _PURE_NUMBER_RE.match(raw):
        val = _to_float(raw)
        if val is not None and abs(val - round(val)) < 1e-6:
            return str(int(round(val)))

    if dataset_key in {"tablebench", "table-bench"}:
        stripped = _strip_tablebench_auxiliary_parenthetical(raw, q)
        if stripped != raw:
            return stripped

    return raw


def _strip_tablebench_auxiliary_parenthetical(answer: str, question: str) -> str:
    """Remove count/rank annotations from entity answers, e.g. ``team (5)``.

    This is deliberately TableBench-only. AIT-QA/HiTab sometimes encode answer
    meaning in parentheses, so applying the same rewrite globally would be unsafe.
    """
    if not answer or "(" not in answer or ")" not in answer:
        return answer
    if not re.search(r"\b(?:which|who|what|where|when|name)\b", question):
        return answer
    if re.fullmatch(r"\(?\s*[-+]?\d[\d,]*(?:\.\d+)?\s*\)?%?", answer.strip()):
        return answer
    stripped = re.sub(r"\s*\((?:#?\d+(?:\.\d+)?|rank\s*\d+|count\s*[:=]?\s*\d+|total\s*[:=]?\s*\d+)\)", "", answer, flags=re.IGNORECASE)
    stripped = re.sub(r"\s+", " ", stripped).strip(" ,;")
    return stripped or answer


def _dedupe(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def align_to_gold_form(answer: str, gold: Any) -> Optional[str]:
    """如果 gold 是数值，且 answer 的某个候选数值与 gold 在容差内相等，
    返回与 gold 形态一致（保留小数位、不带 %）的候选；否则返回 None。

    v9.0 调用约定：仅供诊断写入字段，不替换 final_answer。
    """
    gold_val = _gold_to_float(gold)
    if gold_val is None:
        return None
    cand = normalize_numeric_answer(answer)
    for c in cand:
        v = _to_float(c)
        if v is None:
            continue
        if _close(v, gold_val):
            # 输出与 gold 形态一致（去掉百分号 / 千分位）
            return _format_like_gold(v, gold_val)
    return None


def _gold_to_float(gold: Any) -> Optional[float]:
    if isinstance(gold, (int, float)):
        return float(gold)
    if isinstance(gold, list) and gold:
        return _gold_to_float(gold[0])
    if isinstance(gold, str):
        return _to_float(gold.replace("%", "").strip())
    return None


def _close(a: float, b: float, rel: float = 1e-3, abs_tol: float = 1e-3) -> bool:
    if a == b:
        return True
    diff = abs(a - b)
    return diff <= max(rel * max(abs(a), abs(b)), abs_tol)


def _format_like_gold(value: float, gold: float) -> str:
    if abs(gold - round(gold)) < 1e-6:
        return str(int(round(value)))
    # 保留 gold 的小数位数
    s = f"{gold}"
    if "." in s:
        decimals = len(s.split(".")[1])
        return f"{value:.{decimals}f}"
    return f"{value}"


# ---------------------------------------------------------------------------
# 问题类型粗分类（v9.0 question-type router 的输入）
# ---------------------------------------------------------------------------

_QUESTION_TYPE_PATTERNS = [
    ("count", r"\bhow many\b(?!\s+(?:percent|times|years|days))"),
    ("proportion", r"\b(?:percentage|percent|share|proportion|ratio)\b"),
    ("times", r"\bhow many times\b|\b\d+\.?\d*\s*times\b"),
    ("compare", r"\b(?:more|less|higher|lower|fewer|larger|smaller|greater|than)\b"),
    ("superlative", r"\b(?:largest|smallest|highest|lowest|most|least|maximum|minimum|top|bottom)\b"),
    ("trend", r"\b(?:increase|decrease|rise|fall|grow|shrink|change|trend)\b"),
    ("arithmetic", r"\b(?:total|sum|average|mean|median|difference|sum of|combined)\b"),
    ("lookup", r"^\s*(?:what|which|who|when|where)\b"),
]


def coarse_question_type(question: str) -> str:
    """返回粗粒度问题类型，用于 v9.0 router 的 routing key。"""
    if not question:
        return "lookup"
    q = question.lower()
    for label, pattern in _QUESTION_TYPE_PATTERNS:
        if re.search(pattern, q):
            return label
    return "lookup"
