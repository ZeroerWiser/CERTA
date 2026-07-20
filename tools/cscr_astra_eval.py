#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from eval_utils import evaluate_answer_multi_caliber
from dataset_adapters import dataset_answer_match, normalize_dataset_name
from _common_io import write_json


DEFAULT_CLEAN_TEST = "/home/hsh/ME/Table/EMNLP2026/CausalityAwareTableQA/dataset/hitab/test_samples_clean.jsonl"
DEFAULT_AITQA_CLEAN_TEST = "/home/hsh/ME/Table/EMNLP2026/CausalityAwareTableQA/dataset/AIT-QA/aitqa_clean_questions.json"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, dict)]
        if isinstance(payload, dict):
            for key in ("data", "questions", "items", "samples"):
                rows = payload.get(key)
                if isinstance(rows, list):
                    return [row for row in rows if isinstance(row, dict)]
        raise ValueError(f"Unsupported JSON dataset shape in {path}")
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows



def normalize_answer(answer: Any) -> str:
    if answer is None:
        return ""
    text = str(answer).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s.-]", "", text)
    return text.strip()


def is_missing_or_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple, set)):
        return not value or all(is_missing_or_empty(item) for item in value)
    if isinstance(value, dict):
        return not value
    return not str(value).strip()


def exact_match(prediction: Any, ground_truth: Any) -> bool:
    if is_missing_or_empty(prediction) or is_missing_or_empty(ground_truth):
        return False
    pred_norm = normalize_answer(prediction)
    gt_norm = normalize_answer(ground_truth)
    if pred_norm == gt_norm:
        return True
    pred_num = parse_number(prediction)
    gt_num = parse_number(ground_truth)
    if pred_num is not None and gt_num is not None:
        return abs(pred_num - gt_num) < 1e-6
    return False


def parse_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).replace(",", "")
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def astra_answer_text(value: Any) -> str:
    if isinstance(value, list):
        if len(value) == 1:
            return "" if value[0] is None else str(value[0])
        return "[" + ", ".join(str(v) for v in value) + "]"
    return "" if value is None else str(value)


def sample_answer_value(sample: Dict[str, Any], fallback: Any = "") -> Any:
    if "answer" in sample:
        return sample.get("answer", fallback)
    if "answers" in sample:
        return sample.get("answers", fallback)
    if "gold_answer" in sample:
        return sample.get("gold_answer", fallback)
    return fallback


def dataset_label_prefix(dataset: str) -> str:
    normalized = normalize_dataset_name(dataset)
    if normalized == "hitab":
        return "HiTab"
    if normalized == "aitqa":
        return "AITQA"
    return normalized.upper()


def official_match(dataset: str, prediction: Any, ground_truth: Any) -> bool:
    if is_missing_or_empty(prediction) or is_missing_or_empty(ground_truth):
        return False
    normalized = normalize_dataset_name(dataset)
    gold = parse_gold_literal(ground_truth)
    if normalized == "hitab":
        return bool(evaluate_answer_multi_caliber(prediction, gold).get("hitab_official_em", False))
    return bool(dataset_answer_match(normalized, gold, prediction))


def normalize_question_key(question: Any) -> str:
    return re.sub(r"\s+", " ", str(question or "").strip().lower())


def reference_key(table_id: Any, question: Any, answer: Any) -> Tuple[str, str, str]:
    return (
        str(table_id or "").strip(),
        normalize_question_key(question),
        normalize_answer(answer),
    )


def iter_reference_rows(payload: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(payload, list):
        for row in payload:
            if isinstance(row, dict):
                yield row
        return
    if isinstance(payload, dict):
        for table_result in payload.get("table_results", []) or []:
            table_id = table_result.get("table_uid", table_result.get("table_id", ""))
            data_index = table_result.get("data_index")
            for row in table_result.get("results", []) or []:
                if isinstance(row, dict):
                    out = dict(row)
                    out.setdefault("table_id", table_id)
                    out.setdefault("data_index", data_index)
                    yield out


def load_reference_subset(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    keys = set()
    rows = []
    for row in iter_reference_rows(payload):
        table_id = row.get("table_id", row.get("table_uid", ""))
        question = row.get("question", "")
        answer = row.get("correct_answer", row.get("answer", ""))
        key = reference_key(table_id, question, answer)
        if key[0] and key[1]:
            keys.add(key)
            rows.append({
                "table_id": key[0],
                "question": question,
                "correct_answer": answer,
                "data_index": row.get("data_index"),
                "question_index": row.get("question_index"),
            })
    return {
        "path": str(path),
        "reference_rows": len(rows),
        "unique_keys": len(keys),
        "keys": keys,
        "rows": rows,
    }


def filter_clean_rows_by_reference(
    clean_rows: Sequence[Dict[str, Any]],
    reference: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    keys = reference.get("keys") or set()
    selected: List[Dict[str, Any]] = []
    omitted: List[Dict[str, Any]] = []
    seen = set()
    for clean_index, sample in enumerate(clean_rows):
        key = reference_key(
            sample.get("table_id", ""),
            sample.get("question", ""),
            astra_answer_text(sample_answer_value(sample, "")),
        )
        row = dict(sample)
        row["_cscr_clean_index"] = clean_index
        if key in keys:
            selected.append(row)
            seen.add(key)
        else:
            omitted.append({
                "clean_index": clean_index,
                "id": sample.get("id", ""),
                "table_id": sample.get("table_id", ""),
                "question": sample.get("question", ""),
                "answer": astra_answer_text(sample_answer_value(sample, "")),
            })
    missing_reference = [
        row for row in reference.get("rows", [])
        if reference_key(row.get("table_id", ""), row.get("question", ""), row.get("correct_answer", "")) not in seen
    ]
    report = {
        "reference_path": reference.get("path", ""),
        "reference_rows": reference.get("reference_rows", 0),
        "reference_unique_keys": reference.get("unique_keys", 0),
        "clean_rows_before_filter": len(clean_rows),
        "clean_rows_after_filter": len(selected),
        "clean_rows_omitted": len(omitted),
        "reference_keys_not_found_in_clean": len(missing_reference),
    }
    return selected, omitted, {"summary": report, "omitted_clean_rows": omitted, "missing_reference_rows": missing_reference}


def prediction_by_id(predictions: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    duplicates = set()
    for pred in predictions:
        sid = str(pred.get("id") or "")
        if not sid:
            raise ValueError("missing prediction ID")
        if sid in out:
            duplicates.add(sid)
        out[sid] = pred
    if duplicates:
        raise ValueError(f"duplicate prediction IDs: {', '.join(sorted(duplicates))}")
    return out


def choose_symbolic_answer(pred: Dict[str, Any], fields: Sequence[str]) -> Any:
    for field in fields:
        value = pred.get(field)
        if value not in (None, ""):
            return value
    return ""


def build_astra_payload(
    clean_rows: Sequence[Dict[str, Any]],
    predictions: Sequence[Dict[str, Any]],
    symbolic_fields: Sequence[str],
    dataset: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    pred_map = prediction_by_id(predictions)
    reference_ids: List[str] = []
    seen_reference_ids = set()
    duplicate_reference_ids = set()
    for sample in clean_rows:
        sid = str(sample.get("id") or "")
        if not sid:
            raise ValueError("missing reference ID")
        if sid in seen_reference_ids:
            duplicate_reference_ids.add(sid)
        seen_reference_ids.add(sid)
        reference_ids.append(sid)
    if duplicate_reference_ids:
        raise ValueError(f"duplicate reference IDs: {', '.join(sorted(duplicate_reference_ids))}")
    reference_id_set = set(reference_ids)
    missing_prediction_ids = sorted(reference_id_set - set(pred_map))
    if missing_prediction_ids:
        raise ValueError(f"missing prediction IDs: {', '.join(missing_prediction_ids)}")
    extra_prediction_ids = sorted(set(pred_map) - reference_id_set)
    if extra_prediction_ids:
        raise ValueError(f"extra prediction IDs: {', '.join(extra_prediction_ids)}")
    prefix = dataset_label_prefix(dataset)
    table_results: List[Dict[str, Any]] = []
    missing: List[Dict[str, Any]] = []
    for enum_index, sample in enumerate(clean_rows):
        clean_index = int(sample.get("_cscr_clean_index", enum_index))
        sid = str(sample.get("id") or "")
        pred = pred_map.get(sid)
        if pred is None:
            missing.append({
                "clean_index": clean_index,
                "id": sid,
                "table_id": sample.get("table_id", ""),
                "question": sample.get("question", ""),
                "answer": sample.get("answer", ""),
            })
            continue
        question_data = {
            "question_index": 0,
            "sample_id": sid,
            "question": sample.get("question", pred.get("question", "")),
            "correct_answer": astra_answer_text(sample_answer_value(sample, pred.get("gold_answer", ""))),
            "generated_answer": astra_answer_text(pred.get("final_answer", "")),
            "symbolic_answer": astra_answer_text(choose_symbolic_answer(pred, symbolic_fields)),
            "extra_answer": astra_answer_text(pred.get("llm_answer", "")),
            "cscr_answer_source": pred.get("answer_source", ""),
            "cscr_prompt_type": pred.get("prompt_type", ""),
            "cscr_primary_em": bool(
                pred.get(f"{prefix.lower()}_official_em", pred.get("primary_em", pred.get("hitab_official_em", False)))
            ),
            "cscr_error_type": pred.get("error_type", ""),
        }
        table_results.append({
            "data_index": clean_index,
            "table_uid": sample.get("table_id", pred.get("table_id", "")),
            "questions_count": 1,
            "results": [question_data],
        })
    payload = {
        "batch_info": {
            "source": "cscr_predictions",
            "clean_test_rows": len(clean_rows),
            "matched_questions": len(table_results),
            "missing_questions": len(missing),
        },
        "table_results": table_results,
    }
    return payload, missing


class ChatJudgeClient:
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str = "EMPTY",
        timeout: float = 120.0,
        max_retries: int = 2,
        max_tokens: int = 64,
        temperature: float = 0.0,
        cache_path: str = "",
        cache_mode: str = "readwrite",
    ):
        if cache_mode not in {"off", "readwrite", "readonly", "require"}:
            raise ValueError(f"Unsupported judge cache mode: {cache_mode}")
        self.model = model
        self.base_url = base_url.rstrip("/")
        if not self.base_url.endswith("/v1"):
            self.base_url = f"{self.base_url}/v1"
        self.api_key = api_key or "EMPTY"
        self.timeout = timeout
        self.max_retries = max(1, int(max_retries))
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.cache_mode = cache_mode
        self.cache_path = Path(cache_path).expanduser() if cache_path and cache_mode != "off" else None
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        if self.cache_path and self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = str(row.get("cache_key") or "")
                value = row.get("value")
                if key and isinstance(value, dict):
                    self.cache[key] = value

    @property
    def chat_url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def cache_key(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "base_url": self.base_url,
            "prompt": prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def append_cache(self, key: str, value: Dict[str, Any]) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"cache_key": key, "value": value}, ensure_ascii=False) + "\n")

    def generate(self, prompt: str) -> Dict[str, Any]:
        key = self.cache_key(prompt)
        if key in self.cache:
            self.cache_hits += 1
            cached = dict(self.cache[key])
            cached["cache_hit"] = True
            return cached
        if self.cache_path and self.cache_mode == "require":
            raise RuntimeError(f"judge cache miss in require mode: {key}")

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a strict answer-equivalence judge for table question answering. "
                        "Compare only the predicted answer with the gold answer."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last_error = ""
        t0 = time.time()
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.chat_url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                if response.status_code != 200:
                    raise RuntimeError(f"HTTP {response.status_code}: {response.text[:1000]}")
                data = response.json()
                text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                value = {
                    "text": text or "",
                    "cache_hit": False,
                    "latency_seconds": time.time() - t0,
                    "usage": data.get("usage", {}),
                }
                self.cache_misses += 1
                self.cache[key] = dict(value)
                if self.cache_mode == "readwrite":
                    self.append_cache(key, value)
                return value
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt + 1 >= self.max_retries:
                    raise RuntimeError(last_error) from exc
                time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(last_error or "unknown judge error")


def build_judge_prompt(
    question: str,
    prediction: Any,
    ground_truth: Any,
    prompt_version: str = "json_strict",
) -> str:
    if prompt_version == "legacy":
        return f"""Decide whether the predicted answer is equivalent to the gold answer.

Rules:
- Mark CORRECT if the two answers have the same denotation.
- Ignore harmless formatting differences, brackets, quotes, units, casing, and list order.
- For numeric answers, allow tiny rounding or formatting differences.
- If the prediction is empty, an error message, or answers a different value, mark INCORRECT.
- Do not solve the table question from scratch.

Question:
{question}

Gold answer:
{ground_truth}

Predicted answer:
{prediction}

Return exactly one label: CORRECT or INCORRECT.
"""
    return f"""You are judging answer equivalence for table question answering.

Only compare the predicted answer with the gold answer. Do not solve the table question from scratch.

Equivalence rules:
- CORRECT: same denotation after harmless formatting, casing, bracket, unit, order, or tiny numeric-rounding differences.
- INCORRECT: empty prediction, different entity/value, wrong aggregation result, wrong comparison target, wrong unit, or only a related but non-equivalent answer.
- If uncertain, choose INCORRECT.

Question:
{question}

Gold answer:
{ground_truth}

Predicted answer:
{prediction}

Return one compact JSON object only:
{{"label":"CORRECT","reason":"same denotation"}}
or
{{"label":"INCORRECT","reason":"brief reason"}}
"""


def parse_judge_response(response: str) -> Tuple[bool, str]:
    text = response.strip()
    if not text:
        return False, "empty_response"

    visible = re.sub(r"<think>.*?</think>", " ", text, flags=re.IGNORECASE | re.DOTALL).strip()
    if not visible:
        visible = text

    json_match = re.search(r"\{.*?\}", visible, flags=re.DOTALL)
    if json_match:
        try:
            payload = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            payload = {}
        label = str(payload.get("label", "")).strip().upper()
        if label == "CORRECT":
            return True, "json_label:CORRECT"
        if label == "INCORRECT":
            return False, "json_label:INCORRECT"

    compact = re.sub(r"\s+", " ", visible).strip().lower()
    final_label = re.search(
        r"(?:^|[\s,;:.\[{(])(?:final\s+answer|final\s+label|answer|label|judg(?:e)?ment|verdict)"
        r"\s*[:=\-]\s*(correct|incorrect|true|false|yes|no)\b",
        compact,
    )
    if final_label:
        label = final_label.group(1)
        if label in {"correct", "true", "yes"}:
            return True, f"explicit_label:{label}"
        return False, f"explicit_label:{label}"

    first_token = re.match(r"^(correct|incorrect|true|false|yes|no)\b", compact)
    if first_token:
        label = first_token.group(1)
        if label in {"correct", "true", "yes"}:
            return True, f"leading_label:{label}"
        return False, f"leading_label:{label}"

    for pattern in (
        "not correct", "not equivalent", "not consistent", "different denotation",
        "错误", "不正确", "不一致", "不等价",
    ):
        if pattern in compact:
            return False, f"phrase_negative:{pattern}"

    labels = re.findall(r"\b(correct|incorrect|true|false|yes|no)\b", compact)
    if labels:
        normalized = [("correct" if x in {"correct", "true", "yes"} else "incorrect") for x in labels]
        if len(set(normalized)) == 1:
            return normalized[-1] == "correct", f"single_label_family:{normalized[-1]}"
        return False, "conflicting_labels_default_false"
    for pattern in (
        "equivalent", "same denotation", "consistent", "正确", "一致", "相同", "等价",
    ):
        if pattern in compact:
            return True, f"phrase_positive:{pattern}"
    return False, "unparsed_default_false"


def iter_astra_questions(payload: Dict[str, Any]) -> Iterable[Tuple[Dict[str, Any], Dict[str, Any]]]:
    for table_result in payload.get("table_results", []):
        for question_data in table_result.get("results", []):
            if question_data:
                yield table_result, question_data


def evaluate_payload(
    payload: Dict[str, Any],
    dataset: str,
    judge_client: Optional[ChatJudgeClient] = None,
    judge_target: str = "both",
    judge_prompt_version: str = "json_strict",
    limit: Optional[int] = None,
    resume_results: Optional[List[Dict[str, Any]]] = None,
    save_path: Optional[Path] = None,
    save_every: int = 20,
) -> List[Dict[str, Any]]:
    prefix = dataset_label_prefix(dataset)
    results: List[Dict[str, Any]] = list(resume_results or [])
    seen = {
        (row.get("data_index"), row.get("table_id"), row.get("question_index"))
        for row in results
    }
    pending = list(iter_astra_questions(payload))
    if limit is not None:
        pending = pending[:limit]
    completed_since_save = 0
    for table_result, q in pending:
        key = (table_result.get("data_index"), table_result.get("table_uid"), q.get("question_index", 0))
        if key in seen:
            continue
        question = q.get("question", "")
        correct = q.get("correct_answer", "")
        generated = q.get("generated_answer", "")
        symbolic = q.get("symbolic_answer", "")
        reference_valid = not is_missing_or_empty(correct)
        textual_label = int(reference_valid and exact_match(generated, correct))
        symbolic_label = int(reference_valid and exact_match(symbolic, correct))
        official_textual = int(reference_valid and official_match(dataset, generated, correct))
        official_symbolic = int(reference_valid and official_match(dataset, symbolic, correct))

        row = {
            "question_index": q.get("question_index", 0),
            "data_index": table_result.get("data_index"),
            "table_id": table_result.get("table_uid", ""),
            "sample_id": q.get("sample_id", ""),
            "question": question,
            "correct_answer": correct,
            "generated_answer": generated,
            "symbolic_answer": symbolic,
            "extra_answer": q.get("extra_answer", ""),
            "reference_valid": reference_valid,
            "reference_invalid_reason": "" if reference_valid else "missing_or_empty_gold",
            "EM_textual_label": textual_label,
            "EM_symbolic_label": symbolic_label,
            "Official_textual_label": official_textual,
            "Official_symbolic_label": official_symbolic,
            f"{prefix}_official_textual_label": official_textual,
            f"{prefix}_official_symbolic_label": official_symbolic,
            "LLM_textual_label": 0,
            "LLM_symbolic_label": 0,
            "cscr_answer_source": q.get("cscr_answer_source", ""),
            "cscr_prompt_type": q.get("cscr_prompt_type", ""),
            "cscr_error_type": q.get("cscr_error_type", ""),
        }
        if prefix == "HiTab":
            row["HiTab_official_textual_label"] = official_textual
            row["HiTab_official_symbolic_label"] = official_symbolic
        if judge_client is not None:
            for target, pred_value in (("textual", generated), ("symbolic", symbolic)):
                if judge_target not in {target, "both"}:
                    continue
                prompt = build_judge_prompt(
                    question,
                    pred_value,
                    correct,
                    prompt_version=judge_prompt_version,
                )
                try:
                    call = judge_client.generate(prompt)
                    label, parse_method = parse_judge_response(call.get("text", ""))
                    row[f"LLM_{target}_label"] = int(label)
                    row[f"judge_{target}_response"] = call.get("text", "")
                    row[f"judge_{target}_parse"] = parse_method
                    row[f"judge_{target}_cache_hit"] = bool(call.get("cache_hit"))
                    row[f"judge_{target}_latency_seconds"] = call.get("latency_seconds", 0.0)
                except Exception as exc:
                    row[f"judge_{target}_error"] = str(exc)
        if judge_client is not None:
            row["judge_model"] = judge_client.model
            row["judge_base_url"] = judge_client.base_url
        results.append(row)
        seen.add(key)
        completed_since_save += 1
        if save_path and save_every > 0 and completed_since_save >= save_every:
            write_json(save_path, results)
            completed_since_save = 0
    return results


def parse_gold_literal(text: Any) -> Any:
    if not isinstance(text, str):
        return text
    stripped = text.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        try:
            return json.loads(stripped.replace("'", '"'))
        except json.JSONDecodeError:
            inner = stripped[1:-1].strip()
            if not inner:
                return []
            return [x.strip() for x in inner.split(",")]
    return text


def compute_metrics(results: Sequence[Dict[str, Any]], use_judge: bool, dataset: str = "hitab") -> Dict[str, Any]:
    def label_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if value is None:
            return False
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "y", "correct"}:
            return True
        if text in {"0", "false", "no", "n", "incorrect", ""}:
            return False
        return False

    total = len(results)
    prefix = dataset_label_prefix(dataset)

    def official_label(row: Dict[str, Any], target: str) -> Any:
        for key in (
            f"Official_{target}_label",
            f"{prefix}_official_{target}_label",
            f"HiTab_official_{target}_label",
        ):
            if key in row:
                return row.get(key)
        return 0

    em_textual = sum(1 for row in results if label_bool(row.get("EM_textual_label", 0)))
    em_symbolic = sum(1 for row in results if label_bool(row.get("EM_symbolic_label", 0)))
    official_textual = sum(1 for row in results if label_bool(official_label(row, "textual")))
    official_symbolic = sum(1 for row in results if label_bool(official_label(row, "symbolic")))
    llm_textual = sum(1 for row in results if label_bool(row.get("LLM_textual_label", 0)))
    llm_symbolic = sum(1 for row in results if label_bool(row.get("LLM_symbolic_label", 0)))
    em_max = sum(1 for row in results if label_bool(row.get("EM_textual_label")) or label_bool(row.get("EM_symbolic_label")))
    official_max = sum(
        1
        for row in results
        if label_bool(official_label(row, "textual")) or label_bool(official_label(row, "symbolic"))
    )
    llm_max = sum(1 for row in results if label_bool(row.get("LLM_textual_label")) or label_bool(row.get("LLM_symbolic_label")))

    def rate(n: int) -> float:
        return n / total if total else 0.0

    metrics = {
        "total_questions": total,
        "EM_textual_accuracy": rate(em_textual),
        "EM_symbolic_accuracy": rate(em_symbolic),
        "EM_max_accuracy": rate(em_max),
        "EM_max_semantics": "oracle_union_textual_or_symbolic",
        "selected_final_accuracy": rate(em_textual),
        "selected_final_correct": em_textual,
        "selected_final_semantics": "actual_final_answer_textual_channel",
        "invalid_reference_count": sum(1 for row in results if row.get("reference_valid") is False),
        "Official_textual_accuracy": rate(official_textual),
        "Official_symbolic_accuracy": rate(official_symbolic),
        "Official_max_accuracy": rate(official_max),
        f"{prefix}_official_textual_accuracy": rate(official_textual),
        f"{prefix}_official_symbolic_accuracy": rate(official_symbolic),
        f"{prefix}_official_max_accuracy": rate(official_max),
        "LLM_textual_accuracy": rate(llm_textual) if use_judge else 0.0,
        "LLM_symbolic_accuracy": rate(llm_symbolic) if use_judge else 0.0,
        "LLM_max_accuracy": rate(llm_max) if use_judge else 0.0,
        "EM_textual_correct": em_textual,
        "EM_symbolic_correct": em_symbolic,
        "EM_max_correct": em_max,
        "Official_textual_correct": official_textual,
        "Official_symbolic_correct": official_symbolic,
        "Official_max_correct": official_max,
        f"{prefix}_official_textual_correct": official_textual,
        f"{prefix}_official_symbolic_correct": official_symbolic,
        f"{prefix}_official_max_correct": official_max,
        "LLM_textual_correct": llm_textual if use_judge else 0,
        "LLM_symbolic_correct": llm_symbolic if use_judge else 0,
        "LLM_max_correct": llm_max if use_judge else 0,
    }
    if use_judge:
        for target in ("textual", "symbolic"):
            parse_dist: Dict[str, int] = {}
            error_count = 0
            tp = fp = fn = tn = 0
            for row in results:
                parse_key = str(row.get(f"judge_{target}_parse") or "")
                if parse_key:
                    parse_dist[parse_key] = parse_dist.get(parse_key, 0) + 1
                if row.get(f"judge_{target}_error"):
                    error_count += 1
                official_ok = label_bool(official_label(row, target))
                judge_ok = label_bool(row.get(f"LLM_{target}_label", 0))
                if official_ok and judge_ok:
                    tp += 1
                elif (not official_ok) and judge_ok:
                    fp += 1
                elif official_ok and (not judge_ok):
                    fn += 1
                else:
                    tn += 1
            metrics[f"judge_{target}_parse_distribution"] = dict(sorted(parse_dist.items()))
            metrics[f"judge_{target}_unparsed_count"] = int(parse_dist.get("unparsed_default_false", 0))
            metrics[f"judge_{target}_error_count"] = error_count
            confusion = {
                "true_positive": tp,
                "false_positive_official_wrong_judge_correct": fp,
                "false_negative_official_correct_judge_wrong": fn,
                "true_negative": tn,
            }
            metrics[f"judge_{target}_vs_official"] = confusion
            metrics[f"judge_{target}_vs_{prefix.lower()}_official"] = confusion
        max_tp = max_fp = max_fn = max_tn = 0
        for row in results:
            official_ok = label_bool(official_label(row, "textual")) or label_bool(official_label(row, "symbolic"))
            judge_ok = label_bool(row.get("LLM_textual_label", 0)) or label_bool(row.get("LLM_symbolic_label", 0))
            if official_ok and judge_ok:
                max_tp += 1
            elif (not official_ok) and judge_ok:
                max_fp += 1
            elif official_ok and (not judge_ok):
                max_fn += 1
            else:
                max_tn += 1
        max_confusion = {
            "true_positive": max_tp,
            "false_positive_official_wrong_judge_correct": max_fp,
            "false_negative_official_correct_judge_wrong": max_fn,
            "true_negative": max_tn,
        }
        metrics["judge_max_vs_official_max"] = max_confusion
        metrics[f"judge_max_vs_{prefix.lower()}_official_max"] = max_confusion
    if prefix == "HiTab":
        metrics["HiTab_official_textual_accuracy"] = metrics["Official_textual_accuracy"]
        metrics["HiTab_official_symbolic_accuracy"] = metrics["Official_symbolic_accuracy"]
        metrics["HiTab_official_max_accuracy"] = metrics["Official_max_accuracy"]
        metrics["HiTab_official_textual_correct"] = metrics["Official_textual_correct"]
        metrics["HiTab_official_symbolic_correct"] = metrics["Official_symbolic_correct"]
        metrics["HiTab_official_max_correct"] = metrics["Official_max_correct"]
        if use_judge:
            for target in ("textual", "symbolic"):
                metrics[f"judge_{target}_vs_hitab_official"] = metrics[f"judge_{target}_vs_official"]
            metrics["judge_max_vs_hitab_official_max"] = metrics["judge_max_vs_official_max"]
    return metrics


def load_resume(path: Path) -> List[Dict[str, Any]]:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CSCR predictions with ASTRA-compatible metrics and judges.")
    parser.add_argument("--dataset", choices=["hitab", "aitqa"], default="hitab")
    parser.add_argument("--predictions", required=True, help="CSCR predictions.jsonl")
    parser.add_argument("--clean-test", default="", help="Clean dataset file; supports JSONL and JSON array.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--subset-reference-results",
        default="",
        help="Optional ASTRA evaluation_results.json or astra_compatible_results.json; evaluate only the overlapping clean rows.",
    )
    parser.add_argument("--symbolic-fields", default="operation_support_reranked_denotation,executor_answer,hceg_fallback_candidate")
    parser.add_argument("--judge-backend", choices=["none", "openai_chat", "vllm_chat"], default="none")
    parser.add_argument("--judge-base-url", default="")
    parser.add_argument("--judge-model", default="")
    parser.add_argument("--judge-api-key-env", default="")
    parser.add_argument("--judge-api-key", default="")
    parser.add_argument("--judge-cache-path", default="")
    parser.add_argument("--judge-cache-mode", choices=["off", "readwrite", "readonly", "require"], default="readwrite")
    parser.add_argument("--judge-target", choices=["textual", "symbolic", "both"], default="both")
    parser.add_argument("--judge-prompt-version", choices=["json_strict", "legacy"], default="json_strict")
    parser.add_argument("--judge-timeout", type=float, default=120.0)
    parser.add_argument("--judge-max-retries", type=int, default=2)
    parser.add_argument("--judge-max-tokens", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-every", type=int, default=20)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    dataset = normalize_dataset_name(args.dataset)
    clean_test = args.clean_test or (DEFAULT_AITQA_CLEAN_TEST if dataset == "aitqa" else DEFAULT_CLEAN_TEST)
    clean_rows = read_jsonl(Path(clean_test))
    subset_report: Optional[Dict[str, Any]] = None
    if args.subset_reference_results:
        reference = load_reference_subset(Path(args.subset_reference_results))
        clean_rows, omitted_rows, subset_report = filter_clean_rows_by_reference(clean_rows, reference)
        write_json(output_dir / "subset_filter_report.json", subset_report)
    predictions = read_jsonl(Path(args.predictions))
    symbolic_fields = [x.strip() for x in args.symbolic_fields.split(",") if x.strip()]
    payload, missing = build_astra_payload(clean_rows, predictions, symbolic_fields, dataset=dataset)
    write_json(output_dir / "astra_compatible_results.json", payload)
    write_json(output_dir / "missing_clean_predictions.json", missing)

    judge_client = None
    if args.judge_backend != "none":
        if args.judge_backend == "vllm_chat":
            base_url = args.judge_base_url or "http://127.0.0.1:7781/v1"
            model = args.judge_model or "Llama-2-7b-chat-hf"
            api_key = args.judge_api_key or "EMPTY"
        else:
            base_url = args.judge_base_url or "https://api.lkeap.cloud.tencent.com/v1"
            model = args.judge_model or "deepseek-v3-0324"
            key_env = args.judge_api_key_env or "LKEAP_API_KEY"
            api_key = args.judge_api_key or os.environ.get(key_env, "")
            if not api_key:
                raise RuntimeError(f"Judge API key missing. Set {key_env} or pass --judge-api-key.")
        judge_client = ChatJudgeClient(
            model=model,
            base_url=base_url,
            api_key=api_key,
            timeout=args.judge_timeout,
            max_retries=args.judge_max_retries,
            max_tokens=args.judge_max_tokens,
            cache_path=args.judge_cache_path,
            cache_mode=args.judge_cache_mode,
        )

    result_path = output_dir / "evaluation_results.json"
    resume_rows = load_resume(result_path) if args.resume else []
    results = evaluate_payload(
        payload,
        dataset=dataset,
        judge_client=judge_client,
        judge_target=args.judge_target,
        judge_prompt_version=args.judge_prompt_version,
        limit=args.limit,
        resume_results=resume_rows,
        save_path=result_path,
        save_every=args.save_every,
    )
    metrics = compute_metrics(results, use_judge=judge_client is not None, dataset=dataset)
    metrics["dataset"] = dataset
    if judge_client is not None:
        metrics["judge_model"] = judge_client.model
        metrics["judge_base_url"] = judge_client.base_url
        metrics["judge_cache_hits"] = judge_client.cache_hits
        metrics["judge_cache_misses"] = judge_client.cache_misses
        metrics["judge_prompt_version"] = args.judge_prompt_version
    metrics["clean_test"] = clean_test
    metrics["clean_test_rows"] = len(clean_rows)
    metrics["matched_clean_predictions"] = payload["batch_info"]["matched_questions"]
    metrics["missing_clean_predictions"] = len(missing)
    metrics["coverage"] = payload["batch_info"]["matched_questions"] / len(clean_rows) if clean_rows else 0.0
    if subset_report:
        metrics["subset_reference_results"] = args.subset_reference_results
        metrics["subset_reference_rows"] = subset_report.get("summary", {}).get("reference_rows", 0)
        metrics["subset_clean_rows_before_filter"] = subset_report.get("summary", {}).get("clean_rows_before_filter", 0)
        metrics["subset_clean_rows_omitted"] = subset_report.get("summary", {}).get("clean_rows_omitted", 0)
        metrics["subset_reference_keys_not_found_in_clean"] = subset_report.get("summary", {}).get("reference_keys_not_found_in_clean", 0)
    write_json(result_path, results)
    write_json(output_dir / "evaluation_metrics.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
