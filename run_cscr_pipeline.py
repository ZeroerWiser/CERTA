"""
run_cscr_pipeline.py — CSCR 统一实验管线

整合 Phase 0/1A/1B/2/3/4/6 的所有模块，支持多种实验模式：

模式:
  baseline_a_plus   — 结构感知 Prompt + logit 熵校准 (无图/执行器)
  executor_only     — 结构感知 Prompt + 执行器验证 (无图)
  full              — 完整 CSCR: HCEG + 检索 + 执行器 + LLM (v3 仲裁)
  full_cert         — CSCR + Certificate Matrix + Dominance 决策 (Phase 6)
  recalculate       — 从已有 predictions.jsonl 重新计算四口径 EM

管线步骤 (full_cert 模式):
  1. 加载数据 + 构建结构感知 Prompt
  2. vLLM 推理获取 LLM 答案 + logprobs
  3. 构建 HCEG + 证据检索
  4. 执行器执行 Lookup-Before-Compute
  4b. 结构干预 + SCCI 计算 (Phase 2)
  5. Certificate-aware 仲裁 (Phase 6): Certificate Matrix + Dominance
  6. 四口径 EM 评估 + 校准指标

v9.0 多卡改动:
  - VLLMGeneratorWithLogprobs: 新增 dtype/gpu_memory_utilization/max_num_seqs
    + EOS token 自动感知模型类型 + enable_prefix_caching
  - run_pipeline: 批量推理重构 (--batch-inference)
    将非LLM步骤和LLM步骤分离，实现批量 prompt 收集后一次推理
"""

import argparse
import json
import logging
import math
import os
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("VLLM_ENGINE_ITERATION_TIMEOUT_S", "1800")
os.environ.setdefault("VLLM_RPC_TIMEOUT", "1800000")
os.environ.setdefault("RAY_CGRAPH_get_timeout", "1800")

# ---------------------------------------------------------------------------
# 本地模块导入
# ---------------------------------------------------------------------------

from eval_utils import (
    evaluate_answer_multi_caliber,
    batch_evaluate,
    compute_calibration_metrics,
    classify_error,
)
from structure_aware_formatter import (
    QuestionAnalyzer,
    build_structure_aware_prompt,
    build_scm_cot_prompt,
    build_baseline_e_prompt,
    build_selective_evidence_prompt,
    build_intersection_hint_prompt,
    build_table_pruned_prompt,
    build_table_focus_prompt,
    compute_first_token_entropy,
    greedy_confidence_from_logprobs,
)
from executor import (
    TypedExecutor,
    ExecutorResult,
    OperationType,
    generate_candidates,
    candidates_summary,
    executor_result_summary,
    _entity_match_score,
)
from graph_builder import (
    HCEG,
    NodeType,
    EdgeType,
    build_hceg,
)
from evidence_retriever import (
    EvidenceRetriever,
    EvidenceSubgraph,
    InterventionEngine,
    InterventionResult,
)
from dataset_adapters import (
    dataset_answer_match,
    load_table_for_cscr,
    normalize_dataset_name,
    normalize_item_for_cscr,
)

# v8.6: Credal Probe 诊断层（纯只读，不改变答案）
from credal_probe import compute_probe_diagnostics, aggregate_probe_metrics


# v9.1: HCEG-Fallback（KG 直检兜底）
from hceg_fallback import (
    hceg_direct_retrieve,
    should_trigger_fallback,
)

# v9.0: 轻量答案归一化 + 问题类型路由诊断
from answer_normalizer import (
    normalize_numeric_answer,
    align_to_gold_form,
    deployable_normalize_answer,
    coarse_question_type,
)

# Phase 6: Certificate Matrix + Dominance
from certificate_calibrator import (
    certificate_aware_arbitrate,
    ConformalAbstainer,
)
from causal_predictor import load_predictor

from structural_cert_utils import (
    parse_question_frame,
    annotate_edge_reliability,
    evidence_ib_mdl_score,
)
from experiment_logging import (
    build_run_metadata,
    make_debug_prediction_record,
    prompt_profile,
    resolve_run_id,
    select_prediction_records,
    table_profile,
)
from certa.derivations.answer_equivalence import inference_answer_key
from certa.evaluation.repair_outcomes import aggregate_repair_outcomes, compute_repair_outcome
from certa.diagnostics.heuristic_audit import summarize_legacy_heuristics
from certa.repair.causal_epistemic_agent import run_causal_epistemic_repair
from certa.repair.method_context import (
    PosthocEvaluationRecord,
    build_method_inference_context,
)
from certa.reproducibility.canonical_json import canonical_json, canonical_json_hash, canonical_text_hash



# ---------------------------------------------------------------------------
# I/O 工具
# ---------------------------------------------------------------------------

def _env_flag(names: Sequence[str], default: bool = False) -> bool:
    for name in names:
        raw = os.environ.get(name)
        if raw is None:
            continue
        return raw.strip().lower() in {"1", "true", "yes", "y", "on"}
    return default

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    if path.lower().endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, dict)]
        if isinstance(payload, dict):
            for key in ("data", "questions", "items", "samples"):
                rows = payload.get(key)
                if isinstance(rows, list):
                    return [row for row in rows if isinstance(row, dict)]
        raise ValueError(f"Unsupported JSON dataset shape in {path}")
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_jsonl(path: str, rows) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sha256_text(text: str) -> str:
    return canonical_text_hash(text or "")


def _stable_json_sha256(obj: Any) -> str:
    return canonical_json_hash(obj)


def _stable_unique_strings(values: Sequence[str]) -> List[str]:
    """Deduplicate diagnostic reasons without process-dependent set ordering."""
    return sorted({str(value) for value in values})


def _llm_input_audit_mode(args: argparse.Namespace) -> str:
    mode = str(getattr(args, "save_llm_inputs", "off") or "off").strip().lower()
    return mode if mode in {"off", "hash", "full"} else "off"


def _llm_input_audit_path(args: argparse.Namespace) -> Path:
    raw = str(getattr(args, "llm_input_audit_file", "") or "").strip()
    if raw:
        p = Path(raw)
        if not p.is_absolute():
            p = Path(args.output_dir) / p
        return p
    return Path(args.output_dir) / "llm_input_audit.jsonl"


def _generator_system_prompt(generator: Any) -> str:
    if hasattr(generator, "system_prompt_for_generation"):
        try:
            return str(generator.system_prompt_for_generation() or "")
        except Exception:
            return ""
    return ""


def _generator_messages_for_prompt(generator: Any, prompt: str) -> List[Dict[str, str]]:
    if hasattr(generator, "build_chat_messages"):
        try:
            messages = generator.build_chat_messages(prompt)
            if isinstance(messages, list):
                return [
                    {
                        "role": str(m.get("role", "")),
                        "content": str(m.get("content", "")),
                    }
                    for m in messages
                    if isinstance(m, dict)
                ]
        except Exception:
            pass
    return [{"role": "user", "content": prompt}]


def _generator_rendered_prompt(generator: Any, prompt: str) -> str:
    if hasattr(generator, "format_prompt_for_generation"):
        try:
            return str(generator.format_prompt_for_generation(prompt) or "")
        except Exception:
            return prompt
    return prompt


def _make_llm_input_audit_record(
    *,
    prepared: Dict[str, Any],
    prompt: str,
    generator: Any,
    args: argparse.Namespace,
    prompt_kind: str,
    prompt_type: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    logprobs: int,
    sequence_index: int = 0,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """Build sidecar audit record and compact prediction reference.

    mode=full: sidecar includes complete messages and rendered prompt.
    mode=hash: sidecar includes only hashes and lengths.
    mode=off: returns (None, disabled ref).
    """
    mode = _llm_input_audit_mode(args)
    item = prepared.get("item", {}) or {}
    result = prepared.get("result", {}) or {}
    sample_id = str(item.get("id", result.get("id", "")))
    table_id = str(item.get("table_id", result.get("table_id", "")))
    dataset = str(result.get("dataset", item.get("dataset", getattr(args, "dataset", ""))))
    question = str(item.get("question", result.get("question", "")))
    system_prompt = _generator_system_prompt(generator)
    messages = _generator_messages_for_prompt(generator, prompt)
    rendered_prompt = _generator_rendered_prompt(generator, prompt)

    request_core = {
        "backend": str(getattr(args, "generator_backend", "")),
        "model_path": str(getattr(args, "model_path", "") or ""),
        "api_model": str(getattr(args, "api_model", "") or ""),
        "api_base_url": str(getattr(args, "api_base_url", "") or ""),
        "prompt_kind": prompt_kind,
        "prompt_type": prompt_type,
        "messages": messages,
        "rendered_prompt_for_generation": rendered_prompt,
        "max_new_tokens": int(max_new_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "logprobs": int(logprobs or 0),
        "chat_template_kwargs": dict(getattr(generator, "chat_template_kwargs", {}) or {}),
    }
    request_sha = _stable_json_sha256(request_core)

    ref = {
        "enabled": mode != "off",
        "mode": mode,
        "audit_file": str(_llm_input_audit_path(args)),
        "request_sha256": request_sha,
        "prompt_kind": prompt_kind,
        "prompt_type": prompt_type,
        "system_prompt_present": bool(system_prompt),
        "system_prompt_chars": len(system_prompt),
        "user_prompt_chars": len(prompt or ""),
        "rendered_prompt_chars": len(rendered_prompt or ""),
        "messages_sha256": _stable_json_sha256(messages),
        "rendered_prompt_sha256": _sha256_text(rendered_prompt),
    }

    if mode == "off":
        return None, ref

    record = {
        "sample_id": sample_id,
        "table_id": table_id,
        "dataset": dataset,
        "question": question,
        "prompt_kind": prompt_kind,
        "prompt_type": prompt_type,
        "sequence_index": int(sequence_index),
        "request_sha256": request_sha,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "generator_backend": str(getattr(args, "generator_backend", "")),
        "model_path": str(getattr(args, "model_path", "") or ""),
        "api_model": str(getattr(args, "api_model", "") or ""),
        "api_base_url": str(getattr(args, "api_base_url", "") or ""),
        "api_key_env": str(getattr(args, "api_key_env", "") or ""),
        "sampling": {
            "max_new_tokens": int(max_new_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "logprobs": int(logprobs or 0),
        },
        "lengths": {
            "system_prompt_chars": len(system_prompt),
            "user_prompt_chars": len(prompt or ""),
            "rendered_prompt_chars": len(rendered_prompt or ""),
        },
        "hashes": {
            "messages_sha256": _stable_json_sha256(messages),
            "rendered_prompt_sha256": _sha256_text(rendered_prompt),
        },
    }

    if mode == "full":
        record["system_prompt"] = system_prompt
        record["messages"] = messages
        record["user_prompt"] = prompt
        record["rendered_prompt_for_generation"] = rendered_prompt

    return record, ref


def _append_llm_input_audit(args: argparse.Namespace, records: List[Dict[str, Any]]) -> None:
    if not records:
        return
    path = _llm_input_audit_path(args)
    path.parent.mkdir(parents=True, exist_ok=True)
    append_jsonl(str(path), records)


def setup_logger(log_file: str) -> logging.Logger:
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger("cscr_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(ch)
    return logger


# ---------------------------------------------------------------------------
# 表格加载
# ---------------------------------------------------------------------------

def load_table_json(item: Dict[str, Any], table_dir: str,
                     cache: Dict[str, Dict[str, Any]],
                     dataset: str = "hitab") -> Dict[str, Any]:
    return load_table_for_cscr(item, table_dir, cache, dataset)


def _table_texts_nonempty(table: Mapping[str, Any]) -> bool:
    texts = table.get("texts") or table.get("table_array") or []
    return isinstance(texts, list) and any(isinstance(row, list) and row for row in texts)


def _round8b_graph_preflight_required(args: Any) -> bool:
    if not bool(getattr(args, "enable_cera_repair", False)):
        return False
    if str(getattr(args, "mode", "")) != "full_cert":
        return False
    return bool(
        getattr(args, "cera_enable_typed_planner", False)
        or str(getattr(args, "cera_template_version", "")) == "cera_repair_v3"
    )


def _cera_commit_authorized(args: Any, cera_repair: Any, repaired_answer: str) -> Tuple[bool, str]:
    """Return whether explicit runtime config authorizes a validated CERA commit."""
    if not bool(getattr(args, "cera_commit_approved_repair", False)):
        return False, "commit_not_requested"
    if cera_repair is None:
        return False, "no_cera_result"
    if str(getattr(args, "cera_stage", "") or "").upper() != "E72":
        return False, "stage_not_authorized"
    if bool(getattr(args, "cera_shadow_only", True)):
        return False, "shadow_runtime"
    if not bool(getattr(cera_repair, "validator_accept", False)):
        return False, "validator_rejected"
    if bool(getattr(cera_repair, "would_keep", False)):
        return False, "keep_original"
    if bool(getattr(cera_repair, "insufficient", False)):
        return False, "insufficient_certificate"
    if not bool(getattr(cera_repair, "would_commit", False)):
        return False, "no_validated_use_repaired_decision"
    cera_output = getattr(cera_repair, "output", None)
    if not str(getattr(cera_output, "chosen_hypothesis_id", "") or "").strip():
        return False, "missing_registered_alternative"
    if not str(repaired_answer or "").strip():
        return False, "empty_repaired_answer"
    return True, ""


def validate_graph_required_table_source(args: Any, items: Sequence[Mapping[str, Any]], *, sample_size: int = 3) -> None:
    """Fail fast when graph-required CERTA modes would load empty tables."""
    if not _round8b_graph_preflight_required(args):
        return
    dataset = normalize_dataset_name(str(getattr(args, "dataset", "hitab")))
    table_dir = str(getattr(args, "table_dir", "") or "")
    if dataset == "hitab" and not os.path.isdir(table_dir):
        raise ValueError(f"Round8B graph-required mode needs a valid --table_dir; missing directory: {table_dir}")
    cache: Dict[str, Dict[str, Any]] = {}
    checked = 0
    for item in list(items)[: max(1, int(sample_size))]:
        table = load_table_json(dict(item), table_dir, cache, dataset=dataset)
        checked += 1
        if not _table_texts_nonempty(table):
            table_id = item.get("table_id") or item.get("context") or item.get("id", "")
            raise ValueError(
                "Round8B graph-required mode loaded an empty table; "
                f"check --table_dir for dataset={dataset}, table_id={table_id}"
            )
    if checked == 0:
        raise ValueError("Round8B graph-required mode has no input rows to preflight")


def evaluate_answer_for_dataset(prediction: Any, gold_answer: Any, dataset: str) -> Dict[str, Any]:
    result = evaluate_answer_multi_caliber(prediction, gold_answer)
    dataset = normalize_dataset_name(dataset)
    if dataset == "aitqa":
        result["aitqa_official_em"] = dataset_answer_match("aitqa", gold_answer, prediction)
    elif dataset == "tablebench":
        result["tablebench_official_em"] = dataset_answer_match("tablebench", gold_answer, prediction)
    elif dataset == "sstqa_zh":
        result["sstqa_zh_official_em"] = dataset_answer_match("sstqa_zh", gold_answer, prediction)
    return result


# ---------------------------------------------------------------------------
# 答案提取
# ---------------------------------------------------------------------------

def extract_answer(text: str) -> str:
    """从 LLM 输出中提取答案"""
    if not text:
        return ""
    thinking_block = re.match(r"^\s*<think>(.*?)</think>\s*(.*)$", text, flags=re.IGNORECASE | re.DOTALL)
    if thinking_block:
        text = thinking_block.group(2).strip()
        if not text:
            return ""
    elif re.match(r"^\s*<think>\s*", text, flags=re.IGNORECASE):
        return ""

    # JSON 格式
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            parsed = {}
        for key in ("answer", "final_answer", "final", "prediction"):
            if key in parsed:
                return str(parsed[key]).strip()

    # XML 标签
    tag = re.search(r"<answer>(.*?)</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if tag:
        return tag.group(1).strip()

    # boxed
    boxed = re.search(r"\\boxed\s*\{(.*?)\}", text, flags=re.DOTALL)
    if boxed:
        return boxed.group(1).strip()

    # "Answer:" 模式
    answer_m = re.search(r"Answer\s*:\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)
    if answer_m:
        matched_text = answer_m.group(1).strip()
        # 按行分割
        lines = matched_text.splitlines()
        # 检查列表是否为空，如果非空则取第一行，否则返回空字符串（或自定义默认值）
        first_line = lines[0].strip() if lines else ""

        # 如果后续逻辑依赖于 first_line 非空，可以在这里做判断或记录日志
        if not first_line:
            print(f"Warning: Empty answer extracted from regex. Full match: {answer_m.group(0)}")

        return first_line

    # 多行推理文本中查找最终答案
    lines = text.strip().splitlines()
    first_line = lines[0].strip() if lines else ""
    direct_final = re.search(
        r'(?:final\s+answer|answer\s+is|the\s+answer\s+is)\s*:?\s*(.+)',
        first_line,
        flags=re.IGNORECASE,
    )
    if direct_final:
        candidate = direct_final.group(1).strip().rstrip(".")
        num_matches = re.findall(r'[$£€]?\(?[-+]?\d[\d,]*(?:\.\d+)?\)?\s*%?', candidate)
        if num_matches:
            return num_matches[-1].strip()
        if candidate and len(candidate) < 100:
            return candidate
    if len(lines) > 3:
        for line in reversed(lines):
            line_stripped = line.strip()
            final_match = re.search(
                r'(?:answer\s+is|the\s+answer)\s*[:\s]*(.+)',
                line_stripped, flags=re.IGNORECASE
            )
            if final_match:
                candidate = final_match.group(1).strip().rstrip('.')
                if candidate and len(candidate) < 80:
                    return candidate

    if len(first_line) > 50:
        number_pattern = r'[$£€]?\(?[-+]?\d[\d,]*(?:\.\d+)?\)?\s*%?'
        cue_matches = list(re.finditer(
            r'\b(?:answer|was|is|equals?|total(?:ed)?|amount(?:ed)?\s+to)\b\s+(.+)',
            first_line,
            flags=re.IGNORECASE,
        ))
        if cue_matches:
            cue_numbers = re.findall(number_pattern, cue_matches[-1].group(1))
            if cue_numbers:
                return cue_numbers[-1].strip()
        num_matches = re.findall(number_pattern, first_line)
        if num_matches:
            return num_matches[0].strip()

    return first_line if len(first_line) <= 100 else first_line[:100]


def _api_chat_template_kwargs(backend_name: str, model: str) -> Dict[str, Any]:
    """Qwen3 transport compatibility for OpenAI-compatible vLLM only."""
    if backend_name == "vllm_chat" and "qwen3" in str(model or "").lower():
        return {"enable_thinking": False}
    return {}


def _canonical_answer_key(answer: Any) -> str:
    return inference_answer_key(answer).compact()


def _answer_surface_type(answer: Any) -> str:
    text = "" if answer is None else str(answer).strip()
    if not text:
        return "empty"
    lower = text.lower()
    if re.fullmatch(r"(?:19|20)\d{2}(?:[-/]\d{1,2})?(?:[-/]\d{1,2})?", lower):
        return "date"
    has_alpha = bool(re.search(r"[A-Za-z]", lower))
    has_num = bool(re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", lower))
    if has_alpha and has_num:
        return "mixed"
    if has_num:
        return "numeric"
    if has_alpha:
        return "entity"
    return "other"


def _question_expects_numeric(question: str, coarse_type: str, operation: str) -> bool:
    q = question.lower()
    numeric_cues = (
        "how many", "how much", "what percentage", "what percent", "what proportion",
        "what rate", "what ratio", "what number", "what amount", "total", "average",
        "sum", "difference", "increase", "decrease", "change", "score", "rank",
        "percentage point", "million dollars", "per 100,000",
    )
    if any(cue in q for cue in numeric_cues):
        return True
    if re.search(
        r"\b(?:what|which)\s+(?:is|was|were|are)\s+(?:the\s+)?"
        r"(?:percentage|percent|proportion|rate|rates|ratio|number|amount|value|score|total|average)\b",
        q,
    ):
        return True
    if re.search(
        r"\b(?:highest|lowest|largest|smallest|maximum|minimum)\s+"
        r"(?:percentage|percent|proportion|rate|rates|ratio|number|amount|value|score)\b",
        q,
    ):
        return True
    return coarse_type in {"count", "arithmetic", "proportion", "times", "trend"} or operation in {
        "count", "sum", "average", "difference", "ratio", "proportion", "trend",
    }


def _hceg_candidate_compatible(
    candidate: Any,
    question: str,
    coarse_type: str,
    operation: str,
) -> bool:
    surface_type = _answer_surface_type(candidate)
    if surface_type == "empty":
        return False
    q = question.lower()
    entity_question = re.search(r"\b(which|who|whom|where|whose)\b", q)
    numeric_entity_exception = re.search(
        r"\b(which|what)\s+(number|value|percentage|percent|rate|ratio|amount|score)\b",
        q,
    )
    if entity_question and not numeric_entity_exception:
        return surface_type in {"entity", "mixed", "date"}
    expects_numeric = _question_expects_numeric(question, coarse_type, operation)
    if expects_numeric:
        return surface_type in {"numeric", "mixed", "date"}
    return True


def _compact_text_key(text: Any) -> str:
    key = _canonical_answer_key(text)
    key = key.replace("-", " ")
    key = re.sub(r"\s+", " ", key).strip()
    return key


def _candidate_supported_by_question(candidate: Any, question: str) -> bool:
    cand_key = _compact_text_key(candidate)
    if not cand_key or len(cand_key) < 2:
        return False
    q_key = _compact_text_key(question)
    return bool(re.search(rf"(?<!\w){re.escape(cand_key)}(?!\w)", q_key))


def _looks_like_structural_label(candidate: Any) -> bool:
    key = _compact_text_key(candidate)
    if not key:
        return True
    blocked_exact = {
        "total", "subtotal", "overall", "all", "none", "n/a", "na",
        "value", "values", "rate", "rates", "ratio", "percent", "percentage",
        "share", "number", "amount", "score", "beta", "coefficient",
        "coefficients", "volunteer rate",
    }
    if key in blocked_exact:
        return True
    if re.fullmatch(r"(?:19|20)\d{2}", key):
        return True
    structural_cues = (
        "coefficient", "percentage", "percent", "rate", "ratio",
        "share", "total", "average", "mean", "median",
    )
    return any(cue in key for cue in structural_cues)


_COMPARE_STOPWORD_CANDIDATES = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "in", "is", "it", "of", "on", "or", "per", "than", "the", "their",
    "to", "was", "were", "with",
}

_COMPARE_GENERIC_PARTIAL_TOKENS = {
    "female", "females", "male", "males", "man", "men", "woman", "women",
    "person", "people", "worker", "workers", "group", "groups",
}


def _key_tokens(text: Any) -> List[str]:
    return re.findall(r"[a-z0-9]+", _compact_text_key(text))


def _candidate_is_stopword_or_too_short(candidate: Any) -> bool:
    key = _compact_text_key(candidate)
    if not key:
        return True
    tokens = _key_tokens(key)
    if not tokens:
        return True
    if len(tokens) == 1 and (tokens[0] in _COMPARE_STOPWORD_CANDIDATES or len(tokens[0]) <= 2):
        return True
    return all(tok in _COMPARE_STOPWORD_CANDIDATES for tok in tokens)


def _clean_compare_alternative(text: str) -> str:
    alt = re.sub(r"\s+", " ", text or "").strip(" \t\r\n,;:.?()[]")
    alt = re.sub(
        r"^(?:and|whether|between|which|who|what|where|when|among|within|for|in|of|the|a|an)\s+",
        "",
        alt,
        flags=re.IGNORECASE,
    ).strip(" \t\r\n,;:.?()[]")
    return alt


def _extract_compare_alternatives(question: str) -> List[str]:
    q = re.sub(r"\s+", " ", str(question or "")).strip()
    if not q:
        return []
    matches = list(re.finditer(r"\s+(?:or|versus|vs\.?)\s+", q, flags=re.IGNORECASE))
    if not matches:
        return []
    match = matches[-1]
    left_context = q[: match.start()].strip(" ,;:.?")
    right = q[match.end() :].strip(" ,;:.?")
    right = re.split(r"[?;]", right, maxsplit=1)[0].strip(" ,;:.")
    cut = max(left_context.rfind(","), left_context.rfind(";"), left_context.rfind(":"), left_context.rfind("?"))
    left = left_context[cut + 1 :].strip(" ,;:.?")
    alternatives = [_clean_compare_alternative(left), _clean_compare_alternative(right)]
    alternatives = [alt for alt in alternatives if _compact_text_key(alt)]
    return alternatives[:2] if len(alternatives) == 2 else []


def _candidate_tokens_in_alt(candidate_tokens: List[str], alt_tokens: List[str]) -> bool:
    if not candidate_tokens or not alt_tokens or len(candidate_tokens) > len(alt_tokens):
        return False
    width = len(candidate_tokens)
    for idx in range(0, len(alt_tokens) - width + 1):
        if alt_tokens[idx : idx + width] != candidate_tokens:
            continue
        prefix = alt_tokens[max(0, idx - 2) : idx]
        if prefix[-1:] in (["non"], ["not"], ["no"]) or prefix == ["not", "a"]:
            continue
        return True
    return False


def _candidate_alt_matches(candidate: Any, alternatives: List[str]) -> List[int]:
    cand_key = _compact_text_key(candidate)
    cand_tokens = _key_tokens(candidate)
    matches: List[int] = []
    for idx, alt in enumerate(alternatives):
        alt_key = _compact_text_key(alt)
        alt_tokens = _key_tokens(alt)
        if not alt_key or not alt_tokens:
            continue
        if cand_key == alt_key:
            matches.append(idx)
        elif _candidate_tokens_in_alt(cand_tokens, alt_tokens):
            matches.append(idx)
        elif len(alt_tokens) >= 2 and _candidate_tokens_in_alt(alt_tokens, cand_tokens):
            matches.append(idx)
    return matches


def _compare_polarity(question: str) -> str:
    q = _compact_text_key(question)
    lower_cues = (
        "less likely", "less common", "lower", "lowest", "fewer", "least",
        "smaller", "less populous", "less than",
    )
    higher_cues = (
        "more likely", "more common", "higher", "highest", "greater", "larger",
        "most", "more populous", "more than",
    )
    if any(cue in q for cue in lower_cues):
        return "lower"
    if any(cue in q for cue in higher_cues):
        return "higher"
    return "unknown"


def _certificate_compare_direction_verifier(
    result: Dict[str, Any],
    candidate: Any,
    surface_heuristic_mode: str = "diagnostic",
) -> Dict[str, Any]:
    candidate_text = "" if candidate is None else str(candidate).strip()
    question = result.get("question", "")
    alternatives = _extract_compare_alternatives(question)
    candidate_stopword = _candidate_is_stopword_or_too_short(candidate_text)
    matches = _candidate_alt_matches(candidate_text, alternatives)
    reasons: List[str] = []
    matched_alt = alternatives[matches[0]] if len(matches) == 1 else ""
    cand_tokens = _key_tokens(candidate_text)
    matched_alt_tokens = _key_tokens(matched_alt)

    if candidate_stopword:
        reasons.append("compare_candidate_stopword_or_too_short")
    if len(alternatives) < 2:
        reasons.append("compare_alternatives_not_extracted")
    elif not matches:
        reasons.append("compare_candidate_not_in_contrast_set")
    elif len(matches) > 1:
        reasons.append("compare_candidate_ambiguous_between_alternatives")
    elif (
        len(cand_tokens) == 1
        and cand_tokens[0] in _COMPARE_GENERIC_PARTIAL_TOKENS
        and len(matched_alt_tokens) >= 4
    ):
        reasons.append("compare_candidate_too_partial_for_contrast")

    return {
        "pass": not reasons,
        "reject_reasons": reasons,
        "features": {
            "surface_heuristic_mode": _normalise_surface_heuristic_mode(surface_heuristic_mode),
            "surface_heuristic_only": _normalise_surface_heuristic_mode(surface_heuristic_mode) != "legacy",
            "compare_polarity": _compare_polarity(question),
            "contrast_alternatives": alternatives,
            "matched_alternative": matched_alt,
            "candidate_stopword_or_too_short": candidate_stopword,
            "alternative_match_count": len(matches),
        },
    }


def _numeric_from_text(text: Any) -> Optional[float]:
    if text is None:
        return None
    raw = str(text).strip()
    if not raw:
        return None
    cleaned = raw.replace(",", "")
    cleaned = re.sub(r"[%$£€]", "", cleaned)
    match = re.search(r"[-+]?\d+(?:\.\d+)?", cleaned)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _node_text_for_certificate(graph: Optional[HCEG], node_id: str) -> str:
    if graph is None or node_id not in graph.nodes:
        return ""
    node = graph.nodes[node_id]
    if node.text:
        return str(node.text)
    if node.numeric_value is not None:
        return str(node.numeric_value)
    return ""


def _phrase_header_match_score(phrase: str, header_text: str) -> float:
    phrase_tokens = _key_tokens(phrase)
    header_tokens = _key_tokens(header_text)
    if not phrase_tokens or not header_tokens:
        return 0.0
    phrase_key = " ".join(phrase_tokens)
    header_key = " ".join(header_tokens)
    if phrase_key == header_key:
        return 8.0
    if _candidate_tokens_in_alt(phrase_tokens, header_tokens):
        return 6.0
    if len(header_tokens) >= 2 and _candidate_tokens_in_alt(header_tokens, phrase_tokens):
        return 4.5
    overlap = len(set(phrase_tokens) & set(header_tokens))
    union = len(set(phrase_tokens) | set(header_tokens))
    return 3.0 * overlap / union if union else 0.0


def _top_header_nodes_for_phrase(
    graph: HCEG,
    phrase: str,
    *,
    min_score: float = 3.0,
    top_k: int = 12,
) -> List[Tuple[str, float]]:
    scored: List[Tuple[str, float]] = []
    for node_id, node in graph.nodes.items():
        if node.node_type not in (NodeType.HEADER, NodeType.SPAN, NodeType.AGGREGATOR):
            continue
        score = _phrase_header_match_score(phrase, node.text)
        if score >= min_score:
            scored.append((node_id, score))
    scored.sort(key=lambda x: (-x[1], x[0]))
    return scored[:top_k]


def _question_year_row_bounds(graph: HCEG, question: str) -> List[Tuple[int, int]]:
    years = set(re.findall(r"\b(?:19|20)\d{2}\b", question or ""))
    if not years:
        return []
    year_rows: List[int] = []
    selected_rows: List[int] = []
    max_row = max((n.row for n in graph.nodes.values()), default=-1)
    for node in graph.nodes.values():
        if node.row < 0:
            continue
        key = _compact_text_key(node.text)
        if not re.fullmatch(r"(?:19|20)\d{2}", key or ""):
            continue
        if node.node_type in (NodeType.HEADER, NodeType.SPAN, NodeType.AGGREGATOR) or node.col <= 1:
            year_rows.append(node.row)
            if key in years:
                selected_rows.append(node.row)
    if not selected_rows:
        return []
    all_year_rows = sorted(set(year_rows))
    bounds: List[Tuple[int, int]] = []
    for row in sorted(set(selected_rows)):
        next_rows = [r for r in all_year_rows if r > row]
        end = next_rows[0] if next_rows else max_row + 1
        bounds.append((row + 1, end))
    return bounds


def _mentioned_constraint_headers(
    graph: HCEG,
    question: str,
    alternatives: Sequence[str],
) -> Dict[str, float]:
    question_key = _compact_text_key(question)
    question_tokens = set(_key_tokens(question))
    alt_tokens = set()
    for alt in alternatives:
        alt_tokens.update(_key_tokens(alt))
    generic_headers = {
        "number", "percent", "percentage", "rate", "total", "subtotal",
        "men", "women", "male", "female",
    }
    generic_overlap_tokens = {
        "certificate", "diploma", "degree", "workers", "worker",
        "population", "overall", "type", "people", "persons",
    } | _COMPARE_STOPWORD_CANDIDATES
    constraints: Dict[str, float] = {}
    for node_id, node in graph.nodes.items():
        if node.node_type not in (NodeType.HEADER, NodeType.SPAN, NodeType.AGGREGATOR):
            continue
        tokens = set(_key_tokens(node.text))
        if not tokens:
            continue
        key = " ".join(_key_tokens(node.text))
        if key in generic_headers:
            continue
        if tokens and tokens <= alt_tokens:
            continue
        exact = key and re.search(rf"(?<!\w){re.escape(key)}(?!\w)", question_key)
        overlap = len(tokens & question_tokens)
        specific_overlap = len(tokens & (question_tokens - generic_overlap_tokens))
        if exact:
            constraints[node_id] = max(constraints.get(node_id, 0.0), 6.0)
        elif overlap >= 2:
            constraints[node_id] = max(
                constraints.get(node_id, 0.0),
                min(6.0, 1.0 + overlap + 1.5 * specific_overlap),
            )
    return constraints


def _cell_header_ids(graph: HCEG, cell_id: str, edge_types: Optional[Sequence[EdgeType]] = None) -> List[str]:
    types = set(edge_types) if edge_types is not None else {
        EdgeType.VALUE_UNDER_HEADER,
        EdgeType.ROW_PATH,
        EdgeType.COL_PATH,
    }
    return [neighbor for neighbor, _edge in graph.neighbors(cell_id, types)]


def _cell_numeric_support(
    graph: HCEG,
    cell_id: str,
    alt_header_scores: Dict[str, float],
    constraint_scores: Dict[str, float],
    year_bounds: Sequence[Tuple[int, int]],
    question: str,
) -> Optional[Dict[str, Any]]:
    node = graph.nodes.get(cell_id)
    if node is None or node.node_type != NodeType.CELL:
        return None
    value = node.numeric_value if node.numeric_value is not None else _numeric_from_text(node.text)
    if value is None:
        return None
    header_ids = set(_cell_header_ids(graph, cell_id))
    alt_hits = header_ids & set(alt_header_scores)
    if not alt_hits:
        return None
    row_header_ids = set(_cell_header_ids(graph, cell_id, [EdgeType.ROW_PATH]))
    score = 10.0 + max(alt_header_scores[h] for h in alt_hits)
    matched_constraints = sorted(header_ids & set(constraint_scores))
    for hid in matched_constraints:
        score += constraint_scores[hid]
    if year_bounds:
        in_year_scope = any(start <= node.row < end for start, end in year_bounds)
        score += 4.0 if in_year_scope else -8.0
    q_key = _compact_text_key(question)
    q_has_gender = any(tok in q_key.split() for tok in ("women", "woman", "men", "man", "female", "male", "gender"))
    row_text = " ".join(_compact_text_key(_node_text_for_certificate(graph, hid)) for hid in row_header_ids)
    if not q_has_gender:
        if any(tok in row_text.split() for tok in ("total", "overall", "all")):
            score += 2.0
        if any(tok in row_text.split() for tok in ("women", "men", "female", "male")):
            score -= 2.0
    return {
        "cell_id": cell_id,
        "value": value,
        "score": score,
        "row": node.row,
        "col": node.col,
        "text": node.text,
        "alt_header_ids": sorted(alt_hits),
        "matched_constraints": matched_constraints,
        "row_headers": [
            _node_text_for_certificate(graph, hid)
            for hid in sorted(row_header_ids)
            if _node_text_for_certificate(graph, hid)
        ][:8],
    }


def _best_numeric_support_for_alternative(
    graph: HCEG,
    alternative: str,
    alternatives: Sequence[str],
    question: str,
) -> Optional[Dict[str, Any]]:
    alt_headers = _top_header_nodes_for_phrase(graph, alternative)
    if not alt_headers:
        return None
    alt_header_scores = dict(alt_headers)
    constraint_scores = _mentioned_constraint_headers(graph, question, alternatives)
    year_bounds = _question_year_row_bounds(graph, question)
    supports: List[Dict[str, Any]] = []
    for node_id, node in graph.nodes.items():
        if node.node_type != NodeType.CELL:
            continue
        support = _cell_numeric_support(
            graph,
            node_id,
            alt_header_scores,
            constraint_scores,
            year_bounds,
            question,
        )
        if support is not None:
            supports.append(support)
    if not supports:
        return None
    supports.sort(key=lambda s: (-float(s["score"]), s["row"], s["col"], s["cell_id"]))
    best = supports[0]
    best["top_alternative_headers"] = [
        {"node_id": nid, "text": _node_text_for_certificate(graph, nid), "score": score}
        for nid, score in alt_headers[:6]
    ]
    best["constraint_count"] = len(best.get("matched_constraints", []))
    return best


def _certificate_numeric_direction_verifier(
    result: Dict[str, Any],
    candidate: Any,
    graph: Optional[HCEG] = None,
) -> Dict[str, Any]:
    candidate_text = "" if candidate is None else str(candidate).strip()
    question = result.get("question", "")
    alternatives = _extract_compare_alternatives(question)
    polarity = _compare_polarity(question)
    matches = _candidate_alt_matches(candidate_text, alternatives)
    reasons: List[str] = []
    features: Dict[str, Any] = {
        "numeric_direction_status": "not_applicable",
        "compare_polarity": polarity,
        "contrast_alternatives": alternatives,
        "matched_alternative_index": matches[0] if len(matches) == 1 else None,
    }
    if graph is None or len(alternatives) != 2 or len(matches) != 1 or polarity not in {"higher", "lower"}:
        features["numeric_direction_status"] = "unknown"
        return {"pass": True, "verified": False, "reject_reasons": reasons, "features": features}

    matched_alt = alternatives[matches[0]]
    if _compact_text_key(candidate_text) != _compact_text_key(matched_alt):
        features["numeric_direction_status"] = "partial_candidate_skipped"
        features["matched_alternative"] = matched_alt
        return {"pass": True, "verified": False, "reject_reasons": reasons, "features": features}

    supports = [
        _best_numeric_support_for_alternative(graph, alt, alternatives, question)
        for alt in alternatives
    ]
    if any(s is None for s in supports):
        features["numeric_direction_status"] = "support_not_found"
        features["numeric_direction_supports"] = supports
        return {"pass": True, "verified": False, "reject_reasons": reasons, "features": features}

    min_score = min(float(s.get("score", 0.0)) for s in supports if s is not None)
    values = [float(s["value"]) for s in supports if s is not None]
    if min_score < 12.0 or math.isclose(values[0], values[1], rel_tol=1e-9, abs_tol=1e-9):
        features["numeric_direction_status"] = "low_confidence_or_equal"
        features["numeric_direction_supports"] = supports
        features["numeric_direction_values"] = values
        features["numeric_direction_min_score"] = min_score
        return {"pass": True, "verified": False, "reject_reasons": reasons, "features": features}

    winner_idx = 0 if values[0] >= values[1] else 1
    if polarity == "lower":
        winner_idx = 0 if values[0] <= values[1] else 1
    verified = True
    if winner_idx != matches[0]:
        reasons.append("compare_numeric_direction_conflict")
    features.update({
        "numeric_direction_status": "verified",
        "numeric_direction_supports": supports,
        "numeric_direction_values": values,
        "numeric_direction_winner_index": winner_idx,
        "numeric_direction_winner": alternatives[winner_idx],
        "numeric_direction_margin": abs(values[0] - values[1]),
        "numeric_direction_min_score": min_score,
    })
    return {
        "pass": not reasons,
        "verified": verified,
        "reject_reasons": reasons,
        "features": features,
    }


def _certificate_operation_verifier(
    result: Dict[str, Any],
    candidate: Any,
    surface_heuristic_mode: str = "diagnostic",
) -> Dict[str, Any]:
    candidate_text = "" if candidate is None else str(candidate).strip()
    question = result.get("question", "")
    coarse_type = str(result.get("coarse_question_type", "") or "").lower()
    operation = str(result.get("question_operation", "") or "").lower()
    expected_role = str(result.get("hceg_fallback_expected_role", "") or "").lower()
    surface_type = _answer_surface_type(candidate_text)
    structural_label = _looks_like_structural_label(candidate_text)
    supported_by_question = _candidate_supported_by_question(candidate_text, question)
    mode = _normalise_surface_heuristic_mode(surface_heuristic_mode)
    reasons: List[str] = []
    supported_operation = coarse_type == "compare" and operation == "compare"

    if not supported_operation:
        reasons.append("operation_not_supported_for_commit")
    if supported_operation:
        if expected_role == "entity" and structural_label and mode == "legacy":
            reasons.append("structural_label_candidate")
        if expected_role == "entity" and surface_type not in {"entity", "mixed", "date"}:
            reasons.append("compare_candidate_not_entity")
        if expected_role == "entity" and not supported_by_question and mode == "legacy":
            reasons.append("compare_candidate_not_in_question")
        if expected_role != "entity":
            reasons.append("compare_non_entity_commit_unsupported")
    elif expected_role == "entity" and structural_label and mode == "legacy":
        reasons.append("structural_label_candidate")

    return {
        "pass": not reasons,
        "reject_reasons": reasons,
        "features": {
            "surface_heuristic_mode": mode,
            "surface_structural_label_diagnostic": structural_label,
            "surface_question_support_diagnostic": supported_by_question,
            "surface_heuristic_used_for_reject": mode == "legacy",
            "coarse_question_type": coarse_type,
            "question_operation": operation,
            "expected_role": expected_role,
            "candidate_type": surface_type,
            "candidate_supported_by_question": supported_by_question,
            "structural_label_candidate": structural_label,
            "supported_operation": supported_operation,
        },
    }


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _effective_black_box_commit_policy(args: argparse.Namespace, result: Dict[str, Any]) -> str:
    raw = str(getattr(args, "black_box_commit_policy", "auto") or "auto").lower()
    if raw == "auto":
        return "format_only" if result.get("black_box_api_generator") else "off"
    return raw


def _should_freeze_black_box_answer(policy: str, result: Dict[str, Any]) -> bool:
    return bool(result.get("black_box_api_generator")) and policy in {"freeze", "format_only", "certified"}


def _black_box_semantic_commit_allowed(args: argparse.Namespace, result: Dict[str, Any]) -> bool:
    if not result.get("black_box_api_generator"):
        return True
    return _effective_black_box_commit_policy(args, result) in {"off", "certified"}


def _effective_api_format_normalizer(
    args: argparse.Namespace,
    result: Dict[str, Any],
    black_box_commit_policy: str,
) -> str:
    raw = str(getattr(args, "api_format_normalizer", "auto") or "auto").lower()
    if raw == "auto":
        dataset = normalize_dataset_name(result.get("dataset", getattr(args, "dataset", "hitab")))
        if (
            dataset == "tablebench"
            and result.get("black_box_api_generator")
            and black_box_commit_policy in {"format_only", "certified"}
        ):
            return "conservative"
        return "off"
    return raw


def _certificate_conformal_score_from_features(features: Dict[str, Any]) -> float:
    """v10: lexicographic certificate level used by conformal calibration."""
    operation_ok = bool(features.get("operation_verifier_pass"))
    compare_ok = bool(features.get("compare_direction_verifier_pass"))
    numeric_ok = bool(features.get("numeric_direction_verifier_pass"))
    numeric_verified = bool(features.get("numeric_direction_verifier_verified"))
    role_ok = bool(features.get("candidate_compatible")) and not bool(features.get("role_mismatch"))
    role_mapped = bool(features.get("role_aware_changed"))
    expected_entity = features.get("expected_role") == "entity"

    if operation_ok and compare_ok and numeric_verified and role_ok and (role_mapped or not expected_entity):
        return 1.0
    if operation_ok and compare_ok and numeric_ok and role_ok and (role_mapped or not expected_entity):
        return 0.76
    if operation_ok and compare_ok and role_ok and (role_mapped or not expected_entity):
        return 0.71
    if operation_ok and role_ok:
        return 0.50
    if role_ok:
        return 0.27
    return 0.0


def _certificate_commit_boundary(
    result: Dict[str, Any],
    candidate: Any,
    args: argparse.Namespace,
    graph: Optional[HCEG] = None,
) -> Dict[str, Any]:
    """v9.6: conservative diagnostic boundary for committing structural candidates."""
    candidate_text = "" if candidate is None else str(candidate).strip()
    final_answer = result.get("final_answer", "")
    source = result.get("answer_source", "")
    protected_source = source in ("path_verified_consensus", "consensus_cert")
    compatible = bool(result.get("hceg_fallback_candidate_compatible"))
    role_mismatch = bool(result.get("hceg_fallback_role_mismatch"))
    should_trigger = bool(result.get("hceg_fallback_should_trigger"))
    role_changed = bool(result.get("hceg_fallback_role_aware_changed"))
    expected_role = result.get("hceg_fallback_expected_role", "unknown")
    coarse_type = result.get("coarse_question_type", "unknown")
    operation = result.get("question_operation", "unknown")
    llm_conf = _safe_float(result.get("llm_confidence", result.get("final_confidence", 0.5)), 0.5)
    probe = result.get("probe_diagnostics") or {}
    credal = probe.get("credal_probe") or {}
    cw = _safe_float(credal.get("credal_width"), 0.0)
    entropy = _safe_float(result.get("first_token_entropy"), 0.0)
    max_llm_conf = _safe_float(getattr(args, "certificate_commit_max_llm_confidence", 1.0), 1.0)
    min_cw = _safe_float(getattr(args, "certificate_commit_min_credal_width", 0.0), 0.0)
    allow_diagnostic = bool(getattr(args, "certificate_commit_allow_diagnostic_candidates", False))
    operation_verifier_enabled = bool(getattr(args, "certificate_operation_verifier", False))
    compare_direction_verifier_enabled = bool(getattr(args, "certificate_compare_direction_verifier", False))
    numeric_direction_verifier_enabled = bool(getattr(args, "certificate_numeric_direction_verifier", False))
    if numeric_direction_verifier_enabled and graph is None:
        raise RuntimeError("certificate numeric direction verifier requires HCEG graph")
    conformal_enabled = bool(getattr(args, "certificate_conformal_boundary", False))
    conformal_threshold = _safe_float(getattr(args, "certificate_conformal_threshold", 1.01), 1.01)
    conformal_alpha = _safe_float(getattr(args, "certificate_conformal_alpha", 0.1), 0.1)
    surface_heuristic_mode = _normalise_surface_heuristic_mode(
        getattr(args, "surface_heuristic_mode", "diagnostic")
    )
    operation_check = _certificate_operation_verifier(
        result,
        candidate_text,
        surface_heuristic_mode=surface_heuristic_mode,
    )
    compare_direction_check = _certificate_compare_direction_verifier(
        result,
        candidate_text,
        surface_heuristic_mode=surface_heuristic_mode,
    )
    numeric_direction_check = (
        _certificate_numeric_direction_verifier(result, candidate_text, graph)
        if numeric_direction_verifier_enabled
        else {"pass": True, "verified": False, "reject_reasons": [], "features": {}}
    )
    conformal_features = {
        "answer_source": source,
        "expected_role": expected_role,
        "candidate_compatible": compatible,
        "role_mismatch": role_mismatch,
        "role_aware_changed": role_changed,
        "operation_verifier_pass": bool(operation_check["pass"]),
        "compare_direction_verifier_pass": bool(compare_direction_check["pass"]),
        "numeric_direction_verifier_pass": bool(numeric_direction_check["pass"]),
        "numeric_direction_verifier_verified": bool(numeric_direction_check.get("verified")),
        "numeric_direction_verifier_features": numeric_direction_check["features"],
    }
    conformal_score = _certificate_conformal_score_from_features(conformal_features)
    conformal_accept = conformal_score >= conformal_threshold
    expected_role_source = str(result.get("hceg_fallback_expected_role_source") or "unknown")
    expected_role_is_surface = bool(result.get("hceg_fallback_expected_role_source_is_surface_heuristic"))
    expected_role_structural_supported = (
        expected_role_source.startswith("operation:")
        or expected_role_source.startswith("coarse:")
        or bool(result.get("hceg_fallback_role_aware_changed"))
        or bool(result.get("hceg_fallback_candidate_compatible") and expected_role == "numeric")
    )
    compare_direction_surface_only = bool(
        compare_direction_check.get("features", {}).get("surface_heuristic_only")
    )

    reject_reasons: List[str] = []
    if not candidate_text:
        reject_reasons.append("empty_candidate")
    if protected_source:
        reject_reasons.append("protected_answer_source")
    if not compatible:
        reject_reasons.append("role_incompatible")
    if role_mismatch:
        reject_reasons.append("role_mismatch")
    if not should_trigger and not allow_diagnostic:
        reject_reasons.append("not_gate_triggered")
    if expected_role == "entity" and not role_changed:
        reject_reasons.append("entity_candidate_not_role_mapped")
    if expected_role_is_surface and not expected_role_structural_supported:
        reject_reasons.append("surface_role_without_structural_or_executor_support")
    if llm_conf > max_llm_conf:
        reject_reasons.append("llm_confidence_too_high")
    if cw < min_cw and entropy < 0.05:
        reject_reasons.append("risk_signal_too_low")
    if _canonical_answer_key(candidate_text) == _canonical_answer_key(final_answer):
        reject_reasons.append("same_as_current_answer")
    if operation_verifier_enabled and not operation_check["pass"]:
        reject_reasons.extend(operation_check["reject_reasons"])
    if compare_direction_verifier_enabled and not compare_direction_check["pass"]:
        reject_reasons.extend(compare_direction_check["reject_reasons"])
    if compare_direction_verifier_enabled and compare_direction_surface_only and not (
        numeric_direction_verifier_enabled and numeric_direction_check.get("verified")
    ):
        reject_reasons.append("compare_direction_surface_only_without_numeric_certificate")
    if numeric_direction_verifier_enabled and not numeric_direction_check["pass"]:
        reject_reasons.extend(numeric_direction_check["reject_reasons"])
    if conformal_enabled and not conformal_accept:
        reject_reasons.append("conformal_score_below_threshold")

    shadow_reject_reasons = [r for r in reject_reasons if r != "not_gate_triggered"]
    verifier_pass = (not operation_verifier_enabled) or operation_check["pass"]
    compare_direction_pass = (
        (not compare_direction_verifier_enabled)
        or compare_direction_check["pass"]
    )
    numeric_direction_pass = (
        (not numeric_direction_verifier_enabled)
        or numeric_direction_check["pass"]
    )
    conformal_pass = (not conformal_enabled) or conformal_accept
    positive_tuple = [
        int(bool(candidate_text)),
        int(not protected_source),
        int(compatible),
        int(not role_mismatch),
        int(should_trigger or allow_diagnostic),
        int(expected_role != "entity" or role_changed),
        int((not expected_role_is_surface) or expected_role_structural_supported),
        int(llm_conf <= max_llm_conf),
        int(cw >= min_cw or entropy >= 0.05),
        int(verifier_pass),
        int(compare_direction_pass),
        int((not compare_direction_verifier_enabled) or (not compare_direction_surface_only) or bool(numeric_direction_check.get("verified"))),
        int(numeric_direction_pass),
        int(conformal_pass),
    ]
    shadow_tuple = list(positive_tuple)
    shadow_tuple[4] = 1
    decision = "commit" if sum(positive_tuple) == len(positive_tuple) and not reject_reasons else "reject"
    shadow_decision = "commit" if sum(shadow_tuple) == len(shadow_tuple) and not shadow_reject_reasons else "reject"
    return {
        "candidate": candidate_text,
        "decision": decision,
        "shadow_decision": shadow_decision,
        "mode": getattr(args, "certificate_commit_mode", "diagnostic"),
        "reject_reasons": reject_reasons,
        "shadow_reject_reasons": shadow_reject_reasons,
        "dominance_tuple": positive_tuple,
        "shadow_dominance_tuple": shadow_tuple,
        "features": {
            "answer_source": source,
            "expected_role": expected_role,
            "expected_role_source": expected_role_source,
            "expected_role_is_surface_heuristic": expected_role_is_surface,
            "expected_role_structural_supported": bool(expected_role_structural_supported),
            "surface_heuristic_mode": surface_heuristic_mode,
            "coarse_question_type": coarse_type,
            "question_operation": operation,
            "candidate_type": _answer_surface_type(candidate_text),
            "should_trigger": should_trigger,
            "role_aware_changed": role_changed,
            "candidate_compatible": compatible,
            "role_mismatch": role_mismatch,
            "llm_confidence": round(llm_conf, 4),
            "credal_width": round(cw, 4),
            "first_token_entropy": round(entropy, 4),
            "operation_verifier_enabled": operation_verifier_enabled,
            "operation_verifier_pass": bool(operation_check["pass"]),
            "operation_verifier_reject_reasons": operation_check["reject_reasons"],
            "operation_verifier_features": operation_check["features"],
            "compare_direction_verifier_enabled": compare_direction_verifier_enabled,
            "compare_direction_verifier_pass": bool(compare_direction_check["pass"]),
            "compare_direction_verifier_reject_reasons": compare_direction_check["reject_reasons"],
            "compare_direction_verifier_features": compare_direction_check["features"],
            "numeric_direction_verifier_enabled": numeric_direction_verifier_enabled,
            "numeric_direction_verifier_pass": bool(numeric_direction_check["pass"]),
            "numeric_direction_verifier_verified": bool(numeric_direction_check.get("verified")),
            "numeric_direction_verifier_reject_reasons": numeric_direction_check["reject_reasons"],
            "numeric_direction_verifier_features": numeric_direction_check["features"],
            "conformal_boundary_enabled": conformal_enabled,
            "conformal_score_mode": "lexicographic_certificate_level",
            "conformal_score": round(conformal_score, 4),
            "conformal_threshold": round(conformal_threshold, 4),
            "conformal_alpha": round(conformal_alpha, 4),
            "conformal_accept": conformal_accept,
        },
    }


# ---------------------------------------------------------------------------
# VLLM 生成器 (带 logprobs 支持，v9.0 多卡扩展)
# ---------------------------------------------------------------------------

# EOS token 映射：(关键词列表, eos_token列表)
# 依据"语义保持Token位置"理论：每类模型的结束token在深层隐藏状态中
# 集中表达语义完整性，正确设置stop token确保logprobs完整捕获答案语义
_EOS_MAP = [
    # Qwen3（包括 Qwen3-MoE）
    (["qwen3"],          ["<|im_end|>", "</s>"]),
    # Qwen2.5 / Qwen2（含 Qwen2.5-32B-Instruct 等）
    (["qwen2", "qwen"],  ["<|im_end|>", "</s>"]),
    # LLaMA-3.x（含 3.1, 3.2, 3.3）
    (["llama-3", "llama3", "meta-llama-3"],
                         ["<|eot_id|>", "<|end_of_text|>"]),
    # LLaMA-2
    (["llama-2", "llama2"],
                         ["</s>", "[/INST]"]),
    # Mistral / Mixtral
    (["mistral", "mixtral"],
                         ["</s>", "[/INST]"]),
    # Gemma / Gemma2
    (["gemma"],          ["<end_of_turn>", "</s>"]),
    # InternLM2
    (["internlm2"],      ["<|im_end|>", "</s>"]),
    # DeepSeek
    (["deepseek"],       ["<|EOT|>", "</s>"]),
]

_EOS_DEFAULT = ["<|im_end|>", "</s>"]


def _detect_eos_tokens(model_path: str) -> List[str]:
    """根据模型路径自动选择 EOS tokens"""
    path_lower = model_path.lower()
    for keywords, eos_tokens in _EOS_MAP:
        if any(kw in path_lower for kw in keywords):
            return eos_tokens
    return _EOS_DEFAULT


def _set_llm_kwarg(llm_cls, llm_kwargs: Dict[str, Any], name: str, value: Any) -> None:
    """Add a configured vLLM kwarg; the pinned vLLM environment should accept it."""
    if value is not None:
        llm_kwargs[name] = value


def _disable_vllm_pynccl_if_requested() -> None:
    """Force vLLM TP collectives to use PyTorch distributed instead of PyNCCL."""
    if os.environ.get("CSCR_DISABLE_VLLM_PYNCCL", "1") != "1":
        return
    import vllm.distributed.parallel_state as parallel_state

    original_init = parallel_state.GroupCoordinator.__init__
    if getattr(original_init, "_cscr_disable_pynccl_patch", False):
        return

    def patched_init(self, *args, **kwargs):
        if "use_pynccl" in kwargs:
            kwargs["use_pynccl"] = False
        elif len(args) >= 4:
            args = list(args)
            args[3] = False
            args = tuple(args)
        return original_init(self, *args, **kwargs)

    patched_init._cscr_disable_pynccl_patch = True
    parallel_state.GroupCoordinator.__init__ = patched_init


def set_global_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class VLLMGeneratorWithLogprobs:
    """扩展 VLLMGenerator，支持返回 logprobs，支持多卡 Tensor Parallel

    v9.0 新增参数:
      dtype: 推理数据类型 (bfloat16/float16/auto)，大模型推荐 bfloat16
      gpu_memory_utilization: 显存利用率 0~1，多卡大模型建议 0.90~0.95
      max_num_seqs: vLLM 并发序列数，多卡时可放大以提升批量吞吐
      enable_prefix_caching: HiTab 同一表格的不同问题共享前缀，开启可加速

    EOS 自动感知:
      根据 model_path 关键词自动选择合适的 stop token。
      依据"语义保持Token位置"理论：不同模型在深层（后半层）隐藏状态中
      用特定位置 token 集中表达语义完整性；正确的 stop token 保证
      greedy_confidence_from_logprobs 准确反映模型的答案确定性。
    """

    def __init__(
        self,
        model_path: str,
        max_model_len: int = 16384,
        tensor_parallel_size: int = 1,
        dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.90,
        max_num_seqs: int = 256,
        max_num_batched_tokens: Optional[int] = None,
        swap_space: float = 1,
        cpu_offload_gb: float = 0,
        disable_custom_all_reduce: bool = False,
        enable_prefix_caching: bool = True,
        enable_chunked_prefill: bool = False,
        kv_cache_dtype: str = "auto",
        use_fast_image_processor: bool = True,
        distributed_executor_backend: str = "mp",
        enforce_eager: bool = False,
        seed: int = 0,
    ):
        import os
        from transformers import AutoTokenizer
        from vllm import LLM

        _disable_vllm_pynccl_if_requested()

        # 自动感知 EOS token（依据模型类型）
        self.eos = _detect_eos_tokens(model_path)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

        # 构建 vLLM LLM 实例。单机多卡默认用 mp，避免 Ray worker
        # 在 CUDA_VISIBLE_DEVICES 物理编号下触发 invalid device ordinal。
        if "gemma" in model_path.lower():
            enable_prefix_caching = False

        # NCCL configuration for multi-GPU stability with non-contiguous device IDs.
        # run_cscr.sh normally pins NCCL_SOCKET_IFNAME to the route interface.
        # This fallback avoids container bridges when the Python entrypoint is
        # launched directly.
        if tensor_parallel_size > 1:
            os.environ.setdefault("NCCL_DEBUG", "INFO")
            os.environ.setdefault("NCCL_SOCKET_IFNAME", "^lo,docker,br,veth")
            os.environ.setdefault("NCCL_IB_DISABLE", "1")
            os.environ.setdefault("NCCL_P2P_DISABLE", "0")
            os.environ.setdefault("NCCL_NET_PLUGIN", "none")
            os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
            os.environ.setdefault("NCCL_MIN_NCHANNELS", "4")
            os.environ.setdefault("NCCL_MAX_NCHANNELS", "8")
            os.environ.setdefault("NCCL_BUFFSIZE", "8388608")
            os.environ.setdefault("NCCL_SHM_DISABLE", "0")

        llm_kwargs: Dict[str, Any] = dict(
            model=model_path,
            max_model_len=max_model_len,
            trust_remote_code=True,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
            enable_prefix_caching=enable_prefix_caching,
            tensor_parallel_size=tensor_parallel_size,
        )
        _set_llm_kwarg(LLM, llm_kwargs, "seed", int(seed))
        _set_llm_kwarg(LLM, llm_kwargs, "max_num_batched_tokens", max_num_batched_tokens)
        _set_llm_kwarg(LLM, llm_kwargs, "swap_space", swap_space)
        _set_llm_kwarg(LLM, llm_kwargs, "cpu_offload_gb", cpu_offload_gb)
        _set_llm_kwarg(LLM, llm_kwargs, "disable_custom_all_reduce", disable_custom_all_reduce)
        _set_llm_kwarg(LLM, llm_kwargs, "enforce_eager", enforce_eager)
        _set_llm_kwarg(LLM, llm_kwargs, "enable_chunked_prefill", enable_chunked_prefill)
        _set_llm_kwarg(LLM, llm_kwargs, "kv_cache_dtype", kv_cache_dtype)
        if use_fast_image_processor:
            _set_llm_kwarg(LLM, llm_kwargs, "mm_processor_kwargs", {"use_fast": True})
        if tensor_parallel_size > 1 and distributed_executor_backend:
            llm_kwargs["distributed_executor_backend"] = distributed_executor_backend

        self.model = LLM(**llm_kwargs)
        self._model_path = model_path
        self._tensor_parallel_size = tensor_parallel_size
        self.seed = int(seed)
        self.requested_max_model_len = max_model_len
        self.max_model_len = max_model_len
        model_config = getattr(getattr(self.model, "llm_engine", None), "model_config", None)
        effective_len = getattr(model_config, "max_model_len", None)
        if effective_len is not None:
            self.max_model_len = int(effective_len)
            if self.max_model_len != int(max_model_len):
                logging.getLogger("cscr_pipeline").warning(
                    "vLLM effective max_model_len=%s differs from requested=%s; "
                    "no-truncation audits will use the effective value.",
                    self.max_model_len,
                    max_model_len,
                )

    def system_prompt_for_generation(self) -> str:
        return ""


    def build_chat_messages(self, prompt: str) -> List[Dict[str, str]]:
        return [{"role": "user", "content": prompt}]


    def format_prompt_for_generation(self, prompt: str) -> str:
        """Return the exact string sent to local vLLM after chat-template formatting."""
        messages = self.build_chat_messages(prompt)
        if not hasattr(self.tokenizer, "apply_chat_template"):
            return prompt
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        except (AttributeError, TypeError, ValueError):
            return prompt

    def count_generation_prompt_tokens(self, prompt: str) -> int:
        """Count tokens for the exact generation input, including chat template tokens."""
        formatted = self.format_prompt_for_generation(prompt)
        return len(self.tokenizer(formatted, add_special_tokens=False).input_ids)

    def generate(
        self,
        prompts: Sequence[str],
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        top_p: float = 1.0,
        logprobs: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        批量推理接口。接受任意数量的 prompts，一次性提交给 vLLM。

        返回 [{"text": str, "logprobs": list_or_none}, ...]
        logprobs > 0 时返回 top-K logprobs
        """
        from vllm import SamplingParams

        # 应用 chat template（批量）。后续 no-truncation audit 必须与这里的真实输入一致。
        chat_prompts = [self.format_prompt_for_generation(prompt) for prompt in prompts]
        prompt_token_counts = [
            len(self.tokenizer(p, add_special_tokens=False).input_ids)
            for p in chat_prompts
        ]
        context_budget = max(1, self.max_model_len - max_new_tokens)

        params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=self.eos,
            logprobs=logprobs if logprobs > 0 else None,
            skip_special_tokens=False,  # 保留 special tokens 以便正确提取 logprobs
        )

        outputs = self.model.generate(
            prompts=chat_prompts, sampling_params=params, use_tqdm=True
        )

        results = []
        for out in outputs:
            text = out.outputs[0].text
            lp_data = None

            if logprobs > 0 and out.outputs[0].logprobs:
                lp_data = []
                for token_lp in out.outputs[0].logprobs:
                    # token_lp 是 dict: {token_id: Logprob(logprob, rank, decoded_token)}
                    token_dict = {}
                    for tid, lp_obj in token_lp.items():
                        # 兼容不同版本 vLLM 的 decoded_token 字段名
                        decoded = (
                            getattr(lp_obj, "decoded_token", None)
                            or getattr(lp_obj, "token", None)
                            or str(tid)
                        )
                        # 过滤掉 special token 防止污染 top-K logprobs 排名
                        if decoded and not decoded.startswith("<|") and decoded != "</s>":
                            token_dict[decoded] = lp_obj.logprob
                        elif not token_dict:
                            # 如果全是 special token，至少保留一个
                            token_dict[decoded] = lp_obj.logprob
                    lp_data.append(token_dict)

            token_ids = getattr(out.outputs[0], "token_ids", None) or []
            prompt_tokens = prompt_token_counts[len(results)] if len(results) < len(prompt_token_counts) else None
            results.append({
                "text": text,
                "logprobs": lp_data,
                "generated_token_count": len(token_ids) if token_ids else (len(lp_data) if lp_data else 0),
                "input_token_count": prompt_tokens,
                "context_budget": context_budget,
                "context_pressure_ratio": (prompt_tokens / context_budget) if prompt_tokens is not None else None,
            })

        return results


def _load_api_cache(cache_path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    """Load the optional API response cache shared by chat generator backends."""
    cache: Dict[str, Dict[str, Any]] = {}
    if cache_path and cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = str(row.get("cache_key") or "")
            value = row.get("value")
            if key and isinstance(value, dict):
                cache[key] = value
    return cache


class _ApiGeneratorMixin:
    """Shared prompt formatting and JSONL cache append logic for API generators."""

    def system_prompt_for_generation(self) -> str:
        return ""

    def build_chat_messages(self, prompt: str) -> List[Dict[str, str]]:
        return [{"role": "user", "content": prompt}]

    def format_prompt_for_generation(self, prompt: str) -> str:
        return prompt
    def _append_cache(self, cache_key: str, value: Dict[str, Any]) -> None:
        cache_path = getattr(self, "cache_path", None)
        if not cache_path:
            return
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("a", encoding="utf-8") as f:
            f.write(canonical_json({"cache_key": cache_key, "value": value}) + "\n")


class OpenAIChatGenerator(_ApiGeneratorMixin):
    """OpenAI-compatible chat-completions generator for black-box API models.

    This backend only replaces the language generator. Local table formatting,
    HCEG construction, evidence retrieval, executor diagnostics, credal probes,
    and certificate audits remain unchanged.
    """

    def __init__(
        self,
        model: str,
        api_base_url: str,
        api_key_env: str = "LKEAP_API_KEY",
        timeout: float = 120.0,
        max_retries: int = 3,
        rate_limit_seconds: float = 0.0,
        max_model_len: int = 32768,
        cache_path: str = "",
        cache_mode: str = "readwrite",
        backend_name: str = "openai_chat",
    ):
        from openai import OpenAI

        if cache_mode not in {"off", "readwrite", "readonly", "require"}:
            raise ValueError(f"Unsupported API cache mode: {cache_mode}")
        api_key = os.environ.get(api_key_env)
        base_url = api_base_url.rstrip("/")
        if backend_name == "vllm_chat" and not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"
        if not api_key and (
            api_key_env.upper() in {"EMPTY", "NONE", "DUMMY"}
            or re.search(r"https?://(?:127\.0\.0\.1|localhost)(?::\d+)?(?:/|$)", base_url)
        ):
            api_key = "EMPTY"
        if not api_key:
            raise RuntimeError(
                f"API key environment variable {api_key_env} is not set. "
                "Set it before launching the API-backed CSCR run."
            )
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )
        self.model = model
        self.api_base_url = base_url
        self.api_key_env = api_key_env
        self.backend_name = backend_name
        self.chat_template_kwargs = _api_chat_template_kwargs(backend_name, model)
        self.timeout = timeout
        self.max_retries = max_retries
        self.rate_limit_seconds = rate_limit_seconds
        self.requested_max_model_len = max_model_len
        self.max_model_len = max_model_len
        self.eos: List[str] = []
        self._token_counter = self._init_token_counter(model)
        self.cache_mode = cache_mode
        self.cache_path = Path(cache_path).expanduser() if cache_path and cache_mode != "off" else None
        self._cache = _load_api_cache(self.cache_path)
        self.cache_hits = 0
        self.cache_misses = 0

    @staticmethod
    def _init_token_counter(model: str):
        import tiktoken

        try:
            return tiktoken.encoding_for_model(model)
        except KeyError:
            return tiktoken.get_encoding("cl100k_base")

    def count_generation_prompt_tokens(self, prompt: str) -> int:
        formatted = self.format_prompt_for_generation(prompt)
        if self._token_counter is not None:
            return len(self._token_counter.encode(formatted))
        return max(1, math.ceil(len(formatted) / 4))

    def _cache_key(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        response_format: Optional[Mapping[str, Any]] = None,
    ) -> str:
        payload = {
            "backend": self.backend_name,
            "model": self.model,
            "base_url": self.api_base_url,
            "messages": self.build_chat_messages(prompt),
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "chat_template_kwargs": dict(getattr(self, "chat_template_kwargs", {}) or {}),
        }
        if response_format is not None:
            payload["response_format"] = dict(response_format)
        return canonical_json_hash(payload)

    def _completion_request_kwargs(self, *, prompt: str, max_new_tokens: int, temperature: float, top_p: float, response_format: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"model": self.model, "messages": self.build_chat_messages(prompt), "temperature": temperature, "top_p": top_p, "max_tokens": max_new_tokens}
        if response_format is not None:
            payload["response_format"] = response_format
        if self.chat_template_kwargs:
            payload["extra_body"] = {"chat_template_kwargs": dict(self.chat_template_kwargs)}
        return payload

    def generate_json_schema(
        self,
        prompt: str,
        *,
        response_schema: Mapping[str, Any],
        schema_name: str,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> Dict[str, Any]:
        """Generate once with strict JSON Schema; never retry unconstrained."""
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": str(schema_name),
                "schema": dict(response_schema),
                "strict": True,
            },
        }
        schema_hash = canonical_json_hash(response_schema)
        cache_key = self._cache_key(
            prompt,
            max_new_tokens,
            temperature,
            top_p,
            response_format=response_format,
        )
        if cache_key in self._cache:
            cached = dict(self._cache[cache_key])
            cached["api_cache_hit"] = True
            cached["api_cache_mode"] = self.cache_mode
            cached.setdefault("generation_seconds", 0.0)
            self.cache_hits += 1
            return cached
        if self.cache_path and self.cache_mode == "require":
            raise RuntimeError(
                f"API cache miss in require mode for structured model={self.model}, key={cache_key}."
            )

        prompt_tokens_est = self.count_generation_prompt_tokens(prompt)
        context_budget = max(1, self.max_model_len - max_new_tokens)
        t0 = time.time()
        response = self.client.chat.completions.create(**self._completion_request_kwargs(prompt=prompt, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, response_format=response_format))
        elapsed = time.time() - t0
        choice = response.choices[0]
        text = choice.message.content or ""
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None) if usage is not None else None
        completion_tokens = getattr(usage, "completion_tokens", None) if usage is not None else None
        total_tokens = getattr(usage, "total_tokens", None) if usage is not None else None
        if prompt_tokens is None:
            prompt_tokens = prompt_tokens_est
        if completion_tokens is None:
            completion_tokens = max(1, math.ceil(len(text) / 4)) if text else 0
        value = {
            "text": text,
            "logprobs": None,
            "generated_token_count": int(completion_tokens or 0),
            "input_token_count": int(prompt_tokens or 0),
            "context_budget": context_budget,
            "context_pressure_ratio": float(prompt_tokens or 0) / context_budget,
            "generation_seconds": elapsed,
            "api_usage": {
                "prompt_tokens": int(prompt_tokens or 0),
                "completion_tokens": int(completion_tokens or 0),
                "total_tokens": int(total_tokens or ((prompt_tokens or 0) + (completion_tokens or 0))),
            },
            "api_model": self.model,
            "api_base_url": self.api_base_url,
            "api_key_env": self.api_key_env,
            "generator_backend": self.backend_name,
            "logprobs_available": False,
            "black_box_api": True,
            "api_cache_hit": False,
            "api_cache_mode": self.cache_mode,
            "structured_output_requested": True,
            "structured_output_mechanism": "response_format.type=json_schema",
            "structured_output_schema_hash": schema_hash,
            "structured_output_fallback_used": False,
            "chat_template_kwargs": dict(self.chat_template_kwargs),
        }
        self.cache_misses += 1
        self._cache[cache_key] = dict(value)
        if self.cache_mode == "readwrite":
            self._append_cache(cache_key, value)
        return value

    def generate(
        self,
        prompts: Sequence[str],
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        top_p: float = 1.0,
        logprobs: int = 0,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        context_budget = max(1, self.max_model_len - max_new_tokens)
        for prompt in prompts:
            cache_key = self._cache_key(prompt, max_new_tokens, temperature, top_p)
            if cache_key in self._cache:
                cached = dict(self._cache[cache_key])
                cached["api_cache_hit"] = True
                cached["api_cache_mode"] = self.cache_mode
                cached.setdefault("generation_seconds", 0.0)
                cached.setdefault("chat_template_kwargs", dict(self.chat_template_kwargs))
                self.cache_hits += 1
                results.append(cached)
                continue
            if self.cache_path and self.cache_mode == "require":
                raise RuntimeError(
                    f"API cache miss in require mode for model={self.model}, key={cache_key}. "
                    "Populate the cache with CSCR_API_CACHE_MODE=readwrite first."
                )
            if self.rate_limit_seconds > 0 and results:
                time.sleep(self.rate_limit_seconds)
            prompt_tokens_est = self.count_generation_prompt_tokens(prompt)
            t0 = time.time()
            response = self.client.chat.completions.create(**self._completion_request_kwargs(prompt=prompt, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p))
            elapsed = time.time() - t0
            choice = response.choices[0]
            text = choice.message.content or ""
            usage = getattr(response, "usage", None)
            prompt_tokens = getattr(usage, "prompt_tokens", None) if usage is not None else None
            completion_tokens = getattr(usage, "completion_tokens", None) if usage is not None else None
            total_tokens = getattr(usage, "total_tokens", None) if usage is not None else None
            if prompt_tokens is None:
                prompt_tokens = prompt_tokens_est
            if completion_tokens is None:
                completion_tokens = max(1, math.ceil(len(text) / 4)) if text else 0
            value = {
                "text": text,
                "logprobs": None,
                "generated_token_count": int(completion_tokens or 0),
                "input_token_count": int(prompt_tokens or 0),
                "context_budget": context_budget,
                "context_pressure_ratio": (float(prompt_tokens or 0) / context_budget),
                "generation_seconds": elapsed,
                "api_usage": {
                    "prompt_tokens": int(prompt_tokens or 0),
                    "completion_tokens": int(completion_tokens or 0),
                    "total_tokens": int(total_tokens or ((prompt_tokens or 0) + (completion_tokens or 0))),
                },
                "api_model": self.model,
                "api_base_url": self.api_base_url,
                "api_key_env": self.api_key_env,
                "generator_backend": self.backend_name,
                "logprobs_available": False,
                "black_box_api": True,
                "api_cache_hit": False,
                "api_cache_mode": self.cache_mode,
                "chat_template_kwargs": dict(self.chat_template_kwargs),
            }
            self.cache_misses += 1
            self._cache[cache_key] = dict(value)
            if self.cache_mode == "readwrite":
                self._append_cache(cache_key, value)
            results.append(value)
        return results


class GeminiChatGenerator(_ApiGeneratorMixin):
    """Gemini chat generator for black-box API transfer experiments."""

    def __init__(
        self,
        model: str,
        api_key_env: str = "GEMINI_API_KEY",
        timeout: float = 120.0,
        max_retries: int = 3,
        rate_limit_seconds: float = 0.0,
        max_model_len: int = 32768,
        cache_path: str = "",
        cache_mode: str = "readwrite",
    ):
        if cache_mode not in {"off", "readwrite", "readonly", "require"}:
            raise ValueError(f"Unsupported API cache mode: {cache_mode}")
        api_keys_raw = os.environ.get(api_key_env) or os.environ.get("GEMINI_API_KEYS") or os.environ.get("GOOGLE_API_KEY")
        if not api_keys_raw:
            raise RuntimeError(
                f"Gemini API key is not set. Set {api_key_env}, GEMINI_API_KEYS, or GOOGLE_API_KEY before launching."
            )
        import google.generativeai as genai

        self.genai = genai
        self.api_keys = [k.strip() for k in re.split(r"[,;\s]+", api_keys_raw) if k.strip()]
        if not self.api_keys:
            raise RuntimeError(f"No usable Gemini API key found in {api_key_env}/GEMINI_API_KEYS/GOOGLE_API_KEY")
        self.key_index = random.randrange(len(self.api_keys))
        self.model = model
        self.api_key_env = api_key_env
        self.timeout = timeout
        self.max_retries = max(1, int(max_retries or 1))
        self.rate_limit_seconds = rate_limit_seconds
        self.requested_max_model_len = max_model_len
        self.max_model_len = max_model_len
        self.eos: List[str] = []
        self.cache_mode = cache_mode
        self.cache_path = Path(cache_path).expanduser() if cache_path and cache_mode != "off" else None
        self._cache = _load_api_cache(self.cache_path)
        self.cache_hits = 0
        self.cache_misses = 0
        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
    def system_prompt_for_generation(self) -> str:
        return "You are a helpful AI bot."


    def build_chat_messages(self, prompt: str) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": self.system_prompt_for_generation()},
            {"role": "user", "content": prompt},
        ]
    def count_generation_prompt_tokens(self, prompt: str) -> int:
        return max(1, math.ceil(len(self.format_prompt_for_generation(prompt)) / 4))

    def _cache_key(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        payload = {
            "backend": "gemini_chat",
            "model": self.model,
            "messages": self.build_chat_messages(prompt),
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        return canonical_json_hash(payload)

    def _build_model(self, temperature: float, top_p: float, max_new_tokens: int):
        self.genai.configure(api_key=self.api_keys[self.key_index])
        generation_config = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": 64,
            "max_output_tokens": max_new_tokens,
            "response_mime_type": "text/plain",
        }
        return self.genai.GenerativeModel(
            model_name=self.model,
            safety_settings=self.safety_settings,
            generation_config=generation_config,
            system_instruction="You are a helpful AI bot.",
        )

    def _usage_from_response(self, response: Any, prompt_tokens_est: int, text: str) -> Tuple[int, int, int]:
        usage = getattr(response, "usage_metadata", None)
        prompt_tokens = getattr(usage, "prompt_token_count", None) if usage is not None else None
        completion_tokens = getattr(usage, "candidates_token_count", None) if usage is not None else None
        total_tokens = getattr(usage, "total_token_count", None) if usage is not None else None
        if prompt_tokens is None:
            prompt_tokens = prompt_tokens_est
        if completion_tokens is None:
            completion_tokens = max(1, math.ceil(len(text) / 4)) if text else 0
        if total_tokens is None:
            total_tokens = int(prompt_tokens or 0) + int(completion_tokens or 0)
        return int(prompt_tokens or 0), int(completion_tokens or 0), int(total_tokens or 0)

    def generate(
        self,
        prompts: Sequence[str],
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        top_p: float = 1.0,
        logprobs: int = 0,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        context_budget = max(1, self.max_model_len - max_new_tokens)
        for prompt in prompts:
            cache_key = self._cache_key(prompt, max_new_tokens, temperature, top_p)
            if cache_key in self._cache:
                cached = dict(self._cache[cache_key])
                cached["api_cache_hit"] = True
                cached["api_cache_mode"] = self.cache_mode
                cached.setdefault("generation_seconds", 0.0)
                self.cache_hits += 1
                results.append(cached)
                continue
            if self.cache_path and self.cache_mode == "require":
                raise RuntimeError(
                    f"Gemini API cache miss in require mode for model={self.model}, key={cache_key}. "
                    "Populate the cache with CSCR_API_CACHE_MODE=readwrite first."
                )
            if self.rate_limit_seconds > 0 and results:
                time.sleep(self.rate_limit_seconds)
            prompt_tokens_est = self.count_generation_prompt_tokens(prompt)
            last_error: Optional[Exception] = None
            t0 = time.time()
            text = ""
            response = None
            for attempt in range(self.max_retries):
                try:
                    model = self._build_model(temperature, top_p, max_new_tokens)
                    response = model.generate_content(prompt)
                    text = getattr(response, "text", "") or ""
                    last_error = None
                    break
                except ValueError as exc:
                    raise RuntimeError(f"Gemini blocked or malformed response: {exc}") from exc
                except Exception as exc:
                    last_error = exc
                    if len(self.api_keys) > 1:
                        self.key_index = random.randrange(len(self.api_keys))
                    if attempt + 1 < self.max_retries:
                        time.sleep(max(2.0, self.rate_limit_seconds))
            if last_error is not None:
                raise RuntimeError(f"Gemini generation failed after {self.max_retries} attempts: {last_error}") from last_error
            elapsed = time.time() - t0
            prompt_tokens, completion_tokens, total_tokens = self._usage_from_response(response, prompt_tokens_est, text)
            value = {
                "text": text,
                "logprobs": None,
                "generated_token_count": completion_tokens,
                "input_token_count": prompt_tokens,
                "context_budget": context_budget,
                "context_pressure_ratio": (float(prompt_tokens or 0) / context_budget),
                "generation_seconds": elapsed,
                "api_usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                },
                "api_model": self.model,
                "api_base_url": "google-generativeai",
                "api_key_env": self.api_key_env,
                "generator_backend": "gemini_chat",
                "logprobs_available": False,
                "black_box_api": True,
                "api_cache_hit": False,
                "api_cache_mode": self.cache_mode,
            }
            self.cache_misses += 1
            self._cache[cache_key] = dict(value)
            if self.cache_mode == "readwrite":
                self._append_cache(cache_key, value)
            results.append(value)
        return results


def _entropy_from_token_logprobs(token_logprobs: Dict[str, float]) -> float:
    if not token_logprobs:
        return 0.0
    vals = [float(v) for v in token_logprobs.values()]
    max_lp = max(vals)
    probs = [math.exp(v - max_lp) for v in vals]
    z = sum(probs)
    if z <= 0:
        return 0.0
    norm = [p / z for p in probs]
    return -sum(p * math.log(max(p, 1e-12)) for p in norm)


def _compute_entropy_trajectory_diagnostics(
    logprobs_list: Optional[List[Dict[str, float]]],
) -> Dict[str, Any]:
    if not logprobs_list:
        return {}
    entropies = [
        _entropy_from_token_logprobs(tok_lp)
        for tok_lp in logprobs_list
        if isinstance(tok_lp, dict) and tok_lp
    ]
    n = len(entropies)
    if n == 0:
        return {}

    ordered = sorted(entropies)
    hi = ordered[min(n - 1, int((n - 1) * 0.70))]
    lo = ordered[min(n - 1, int((n - 1) * 0.30))]
    low_patience = 2
    segments: List[Tuple[int, int]] = []
    active_start: Optional[int] = None
    low_run = 0

    for idx, ent in enumerate(entropies):
        if active_start is None:
            if ent >= hi:
                active_start = idx
                low_run = 0
            continue
        if ent <= lo:
            low_run += 1
            if low_run >= low_patience:
                end = max(active_start, idx - low_run)
                segments.append((active_start, end))
                active_start = None
                low_run = 0
        else:
            low_run = 0
    if active_start is not None:
        segments.append((active_start, n - 1))

    denom = max(1, n - 1)
    mass = sum(end - start + 1 for start, end in segments)
    if mass:
        centroid = sum(
            (end - start + 1) * ((start + end) / 2.0) / denom
            for start, end in segments
        ) / mass
        late_mass = sum(
            end - start + 1
            for start, end in segments
            if ((start + end) / 2.0) / denom >= 0.5
        )
    else:
        centroid = 0.0
        late_mass = 0

    tail_start = max(0, int(n * 2 / 3))
    tail = entropies[tail_start:]
    tail_phase_ratio = (
        sum(1 for ent in tail if ent >= hi) / len(tail)
        if tail else 0.0
    )

    return {
        "length": n,
        "mean_entropy": sum(entropies) / n,
        "max_entropy": max(entropies),
        "tail_mean_entropy": sum(tail) / len(tail) if tail else 0.0,
        "hep_count": len(segments),
        "hep_total_mass": mass,
        "entropy_centroid": centroid,
        "late_hep_mass": late_mass,
        "tail_entropy_phase_ratio": tail_phase_ratio,
    }


# ---------------------------------------------------------------------------
# 仲裁器: LLM 答案 vs 执行器答案
# ---------------------------------------------------------------------------

def arbitrate(
    llm_answer: str,
    executor_result: Optional[ExecutorResult],
    llm_confidence: float,
    executor_confidence: float = 1.0,
) -> Tuple[str, str, float]:
    """
    仲裁 LLM 和执行器的答案。

    返回: (final_answer, source, confidence)

    v3 策略 (LLM-first，基于实验诊断):

    实验数据表明:
    - LLM 单独: 60.9% EM
    - Executor 单独: 8.3% EM
    - Consensus (两者一致): 66.7% EM

    因此策略:
    1. 两者一致 → consensus (最高置信)
    2. LLM 高置信度 (≥0.7) → 信任 LLM
    3. 执行器 lookup 且 LLM 低置信度 (<0.3) 且执行器答案看起来像数值 → 考虑执行器
    4. 默认 → 信任 LLM

    关键原则: 执行器只有在 LLM 明确不确定时才作为备选。
    """
    if executor_result is None or not executor_result.executor_valid:
        return llm_answer, "llm_only", llm_confidence

    exec_answer = executor_result.denotation
    if not exec_answer or not exec_answer.strip():
        return llm_answer, "llm_only", llm_confidence

    # 检查一致性 (最优先)
    from eval_utils import normalize_text as _eval_norm
    llm_norm = _eval_norm(llm_answer)
    exec_norm = _eval_norm(exec_answer)

    if llm_norm == exec_norm:
        boosted = min(1.0, max(llm_confidence, executor_confidence) * 1.15)
        return llm_answer, "consensus", boosted

    # 数值一致性检查 (52.1 vs 52.1%)
    from eval_utils import parse_number_strict
    llm_num = parse_number_strict(llm_answer)
    exec_num = parse_number_strict(exec_answer)
    if llm_num is not None and exec_num is not None:
        import math
        if math.isclose(llm_num, exec_num, rel_tol=1e-4, abs_tol=1e-6):
            boosted = min(1.0, max(llm_confidence, executor_confidence) * 1.15)
            return llm_answer, "consensus_numeric", boosted

    # v3: executor_lookup_aggregate 历史 EM=0%，不信任
    if executor_result.operation == OperationType.LOOKUP_AGGREGATE:
        return llm_answer, "llm_preferred", llm_confidence

    # LLM 高置信度 → 直接信任 LLM
    if llm_confidence >= 0.7:
        return llm_answer, "llm_confident", llm_confidence

    # LLM 有答案 → 默认信任 LLM
    if llm_answer.strip():
        if (llm_confidence < 0.3
            and executor_result.priority <= 2
            and exec_num is not None
            and executor_result.confidence >= 0.9):
            return exec_answer, f"executor_{executor_result.operation.value}", executor_confidence * 0.8

        return llm_answer, "llm_preferred", llm_confidence

    # LLM 无答案 → 用执行器
    return exec_answer, f"executor_{executor_result.operation.value}", executor_confidence * 0.5


# ---------------------------------------------------------------------------
# HCEG 依赖校验
# ---------------------------------------------------------------------------

HCEG_PROMPT_STYLES = {"scm_cot", "selective_evidence", "table_focus", "table_pruned"}


def _build_hceg_and_retrieve(
    table_json: Dict[str, Any],
    question: str,
    structural_prior_weighting: bool = False,
) -> Tuple[HCEG, EvidenceSubgraph, Dict[str, Any]]:
    graph = build_hceg(table_json=table_json, question=question)
    question_frame = parse_question_frame(question)
    edge_diag = annotate_edge_reliability(
        graph=graph,
        table_json=table_json,
        question_frame=question_frame,
        apply_to_weight=structural_prior_weighting,
    )
    retriever = EvidenceRetriever(graph, max_expansion_hops=3, max_evidence_cells=30)
    evidence = retriever.retrieve(question)
    ib_score, ib_diag = evidence_ib_mdl_score(evidence)
    evidence.metadata["ib_mdl_score"] = ib_score
    evidence.metadata["ib_mdl_diag"] = ib_diag
    stats = {
        "evidence_num_anchors": len(evidence.anchor_nodes),
        "evidence_num_cells": evidence.num_cells,
        "evidence_has_aggregator": evidence.has_aggregator,
        "evidence_score": evidence.retrieval_score,
        "graph_stats": graph.stats(),
        "question_frame": question_frame,
        "edge_reliability_diag": edge_diag,
        "layout_risk": edge_diag.get("layout_risk", 0.0),
        "layout_flags": edge_diag.get("layout_flags", []),
        "evidence_ib_mdl_score": ib_score,
        "evidence_ib_mdl_diag": ib_diag,
        "evidence_no_anchor_fallback": bool(ib_diag.get("fallback", False)),
        "structural_prior_weighting": bool(structural_prior_weighting),
    }
    return graph, evidence, stats



def _require_hceg_state(
    graph: Optional[HCEG],
    evidence: Optional[EvidenceSubgraph],
    stage: str,
) -> Tuple[HCEG, EvidenceSubgraph]:
    """禁止 HCEG 依赖模块在缺图/缺证据时静默降级。"""
    if graph is None:
        raise RuntimeError(f"{stage} requires HCEG graph, but graph was not constructed")
    if evidence is None:
        raise RuntimeError(f"{stage} requires EvidenceSubgraph, but evidence was not retrieved")
    return graph, evidence


def _require_full_cert_state(
    graph: Optional[HCEG],
    evidence: Optional[EvidenceSubgraph],
    interventions: Optional[List[InterventionResult]],
    all_exec_candidates: Optional[List[ExecutorResult]],
) -> Tuple[HCEG, EvidenceSubgraph, List[InterventionResult], List[ExecutorResult]]:
    """full_cert 不能退化成 full/LLM-only 路径。"""
    graph, evidence = _require_hceg_state(graph, evidence, "full_cert")
    if interventions is None:
        raise RuntimeError("full_cert requires intervention generation, but interventions were not generated")
    if all_exec_candidates is None:
        raise RuntimeError("full_cert requires executor candidate generation, but candidates were not generated")
    return graph, evidence, interventions, all_exec_candidates


# ---------------------------------------------------------------------------
# 核心管线 — 非LLM步骤（可在 LLM 推理前批量执行）
# ---------------------------------------------------------------------------

def prepare_non_llm_steps(
    item: Dict[str, Any],
    table_json: Dict[str, Any],
    args: argparse.Namespace,
    mode: str,
) -> Dict[str, Any]:
    """
    执行所有非LLM步骤，构建 prompt 并收集中间结果。

    返回包含以下字段的 dict:
      - result: 部分结果 dict（含 id/question/graph_stats 等）
      - prompt: 构建好的 prompt 字符串
      - graph, evidence, interventions, executor_result, all_exec_candidates

    v9.0 批量推理重构的第一阶段：
    先对整个 batch 并行执行所有非 LLM 步骤（图构建、执行器、prompt 构建），
    收集所有 prompt 后一次性提交给 vLLM，避免逐样本串行推理的低效。
    """
    preparation_start = time.time()
    question = item.get("question", "")
    sample_id = item.get("id", "")

    result: Dict[str, Any] = {
        "id": sample_id,
        "table_id": item.get("table_id", ""),
        "dataset": item.get("dataset", getattr(args, "dataset", "hitab")),
        "question": question,
        "main_cert_profile": bool(getattr(args, "main_cert_profile", False)),
        "heuristic_surface_used_for_commit": False,
        "legacy_commit_path_used": False,
        "commit_decision_is_boolean_conjunction": False,
        "disable_candidate_scci": bool(getattr(args, "disable_candidate_scci", False)),
    }
    result["table"] = table_profile(table_json)
    result["query_table_entity_anchors"] = _table_grounded_query_entity_anchors(question, table_json)
    result["query_table_literal_anchors"] = _table_grounded_query_literal_anchors(question, table_json)
    for meta_key in ("dataset_question_type", "dataset_question_subtype", "row_hierarchy_needed", "paraphrase_group"):
        if meta_key in item:
            result[meta_key] = item.get(meta_key, "")
    result["coarse_question_type"] = coarse_question_type(question)

    use_scm_cot = getattr(args, "scm_cot", False) and mode in ("full", "full_cert")
    prompt_style = getattr(args, "prompt_style", "structure_aware")
    dataset_prompt_policy = getattr(args, "dataset_prompt_policy", "auto")
    if prompt_style in HCEG_PROMPT_STYLES and mode not in ("full", "full_cert"):
        raise ValueError(
            f"prompt_style={prompt_style!r} requires mode='full' or mode='full_cert'; "
            "it cannot run without HCEG construction"
        )

    graph = None
    evidence = None
    interventions = None
    executor_result = None
    all_exec_candidates = None

    # --- Phase A: 图构建 + 证据检索 (SCM-CoT / selective_evidence / table_focus / table_pruned 模式下提前) ---
    need_graph_before_llm = (
        use_scm_cot or prompt_style in ("selective_evidence", "table_focus", "table_pruned")
    ) and mode in ("full", "full_cert")

    if need_graph_before_llm:
        try:
            graph, evidence, hceg_stats = _build_hceg_and_retrieve(table_json, question, getattr(args, "structural_prior_weighting", False))
            result.update(hceg_stats)
        except Exception as e:
            raise RuntimeError("HCEG construction/evidence retrieval failed") from e

    # --- Phase B: 执行器 ---
    # v9.0: 始终运行 QuestionAnalyzer，把 question_operation 写入 result，便于 op-level 诊断
    analyzer = QuestionAnalyzer(question)
    result["question_operation"] = analyzer.operation_type
    if mode in ("executor_only", "full", "full_cert"):
        try:
            executor = TypedExecutor(table_json)
            executor_result = executor.execute(question, operation_type=analyzer.operation_type)

            if mode == "full_cert":
                all_exec_candidates = generate_candidates(
                    table_json, question, operation_type=analyzer.operation_type
                )
                result["num_exec_candidates"] = len(all_exec_candidates)
                result["exec_candidates_summary"] = candidates_summary(all_exec_candidates)

        except Exception as e:
            raise RuntimeError("TypedExecutor execution failed") from e

        if executor_result:
            result["executor_answer"] = executor_result.denotation
            result["executor_operation"] = executor_result.operation.value
            result["executor_priority"] = executor_result.priority
            result["executor_valid"] = executor_result.executor_valid
            result["executor_trace"] = executor_result.computation_trace
            if getattr(args, "operation_support_diagnostics", False):
                expected_role = _expected_role_for_question(
                    question,
                    result.get("coarse_question_type", ""),
                    analyzer.operation_type,
                    surface_heuristic_mode=getattr(args, "surface_heuristic_mode", "diagnostic"),
                )
                result["operation_support_diagnostic_enabled"] = True
                result["operation_support_expected_role"] = expected_role
                result["operation_support"] = executor_result_summary(executor_result)
                result["operation_support_cell_count"] = len(executor_result.cells_used)
                result["operation_support_valid"] = bool(executor_result.executor_valid)
                result["operation_support_operation"] = executor_result.operation.value
                result["operation_support_denotation"] = executor_result.denotation
                if all_exec_candidates is not None:
                    result["operation_support_candidates"] = [
                        executor_result_summary(c, max_cells=8)
                        for c in all_exec_candidates[:6]
                    ]
                if getattr(args, "operation_role_target_diagnostics", False):
                    _attach_role_target_support_diagnostics(
                        result=result,
                        question=question,
                        coarse_type=result.get("coarse_question_type", ""),
                        question_operation=analyzer.operation_type,
                        executor_result=executor_result,
                        all_exec_candidates=all_exec_candidates or [executor_result],
                        operation_commit_gate_diagnostics=getattr(
                            args, "operation_commit_gate_diagnostics", False
                        ),
                        operation_commit_dataset_scope=getattr(
                            args, "operation_commit_dataset_scope", "tablebench"
                        ),
                        surface_heuristic_mode=getattr(args, "surface_heuristic_mode", "diagnostic"),
                        operation_commit_version=getattr(args, "operation_commit_version", "E67"),
                    )

    # --- v9.0b: Question-Type Router ---
    # 根据 coarse_question_type 动态覆盖 prompt_style（数据驱动，E40/E42 切片验证）
    # lookup/proportion/superlative → table_focus（+1.46pp/+0.74pp/+1.13pp）
    # count/compare/arithmetic/times/trend → baseline_e（避免退化）
    if getattr(args, "question_type_router", False):
        coarse = result.get("coarse_question_type", "")
        if coarse in ("lookup", "proportion", "superlative"):
            prompt_style = "table_focus"
            result["prompt_style_routed"] = "table_focus"
        else:
            prompt_style = "baseline_e"
            result["prompt_style_routed"] = "baseline_e"
        result["prompt_style_routing_reason"] = f"coarse_type={coarse}"
        # table_focus 需要 graph/evidence，确保已构建
        if prompt_style == "table_focus" and graph is None and mode in ("full", "full_cert"):
            graph, evidence, hceg_stats = _build_hceg_and_retrieve(table_json, question, getattr(args, "structural_prior_weighting", False))
            result.update(hceg_stats)

    if prompt_style in HCEG_PROMPT_STYLES or use_scm_cot:
        graph, evidence = _require_hceg_state(graph, evidence, f"prompt_style={prompt_style}")

    # --- Step 1: 构建 Prompt ---
    if use_scm_cot and prompt_style == "scm_cot":
        prompt = build_scm_cot_prompt(
            table_json=table_json,
            question=question,
            evidence=evidence,
            graph=graph,
            exec_candidates=all_exec_candidates,
            graph_stats=result.get("graph_stats"),
        )
        result["prompt_type"] = "scm_cot"
    elif prompt_style == "baseline_e":
        prompt = build_baseline_e_prompt(
            table_json,
            question,
            dataset_prompt_policy=dataset_prompt_policy,
        )
        result["prompt_type"] = "baseline_e"
    elif prompt_style == "selective_evidence":
        prompt = build_selective_evidence_prompt(
            table_json=table_json,
            question=question,
            evidence=evidence,
            graph=graph,
        )
        result["prompt_type"] = "selective_evidence"
    elif prompt_style == "table_focus":
        prompt = build_table_focus_prompt(
            table_json=table_json,
            question=question,
            evidence=evidence,
            graph=graph,
            dataset_prompt_policy=dataset_prompt_policy,
        )
        result["prompt_type"] = "table_focus"
    elif prompt_style == "table_pruned":
        prompt = build_table_pruned_prompt(
            table_json=table_json,
            question=question,
            evidence=evidence,
            graph=graph,
        )
        result["prompt_type"] = "table_pruned"
    else:
        prompt = build_structure_aware_prompt(table_json, question)
        result["prompt_type"] = "structure_aware"

    result["prompt_length"] = len(prompt)
    result["dataset_prompt_policy"] = dataset_prompt_policy
    result["prompt"] = prompt_profile(
        prompt,
        prompt_type=result.get("prompt_type", ""),
        template_version=getattr(args, "prompt_template_version", ""),
        serialization_version=getattr(args, "table_serialization_version", ""),
        max_table_chars=getattr(args, "max_table_chars", None),
    )
    result["non_llm_preparation_seconds"] = time.time() - preparation_start

    return {
        "result": result,
        "prompt": prompt,
        "graph": graph,
        "evidence": evidence,
        "interventions": interventions,
        "executor_result": executor_result,
        "all_exec_candidates": all_exec_candidates,
        "item": item,
        "table_json": table_json,
    }


def finalize_after_llm(
    prepared: Dict[str, Any],
    gen_output: Dict[str, Any],
    args: argparse.Namespace,
    mode: str,
    top_k_logprobs: int = 5,
    generator: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    在 LLM 推理之后完成所有后续步骤（图干预、仲裁、评估）。

    参数:
      prepared: prepare_non_llm_steps 的返回值
      gen_output: {"text": str, "logprobs": list_or_none}
      args: 命令行参数
      mode: 实验模式
      top_k_logprobs: logprobs top-K 数量

    v9.0 批量推理重构的第二阶段：
    接收 vLLM 批量推理的单个结果，完成仲裁和评估。
    """
    finalize_start = time.time()
    result = dict(prepared["result"])
    graph = prepared["graph"]
    evidence = prepared["evidence"]
    interventions = prepared["interventions"]
    executor_result = prepared["executor_result"]
    all_exec_candidates = prepared["all_exec_candidates"]
    item = prepared["item"]
    table_json = prepared["table_json"]
    question = item.get("question", "")
    use_scm_cot = getattr(args, "scm_cot", False) and mode in ("full", "full_cert")

    # --- Step 2: 处理 LLM 输出 ---
    llm_answer = ""
    llm_confidence = 0.5
    llm_logprobs = None

    text = gen_output.get("text", "")
    llm_logprobs = gen_output.get("logprobs")
    result["llm_generation_seconds"] = float(gen_output.get("generation_seconds", 0.0) or 0.0)
    result["generated_token_count"] = int(gen_output.get("generated_token_count", 0) or 0)
    result["generator_backend"] = gen_output.get(
        "generator_backend",
        getattr(args, "generator_backend", "vllm"),
    )
    result["llm_logprobs_available"] = bool(llm_logprobs)
    if gen_output.get("black_box_api"):
        result["black_box_api_generator"] = True
        result["api_model"] = gen_output.get("api_model", getattr(args, "api_model", ""))
        result["api_base_url"] = gen_output.get("api_base_url", getattr(args, "api_base_url", ""))
        result["api_key_env"] = gen_output.get("api_key_env", getattr(args, "api_key_env", ""))
        result["api_usage"] = gen_output.get("api_usage", {})
        result["api_logprobs_unavailable"] = True
        result["api_cache_hit"] = bool(gen_output.get("api_cache_hit", False))
        result["api_cache_mode"] = gen_output.get("api_cache_mode", getattr(args, "api_cache_mode", "readwrite"))
        result["chat_template_kwargs"] = dict(gen_output.get("chat_template_kwargs", {}) or {})

    llm_answer = extract_answer(text)

    if llm_logprobs:
        llm_confidence = greedy_confidence_from_logprobs(llm_logprobs, max_tokens=3)
        first_token_entropy = compute_first_token_entropy(
            llm_logprobs[0] if llm_logprobs else {},
            top_k=top_k_logprobs,
        )
        result["first_token_entropy"] = first_token_entropy
        trajectory_diag = _compute_entropy_trajectory_diagnostics(llm_logprobs)
        if trajectory_diag:
            result["entropy_trajectory_diagnostics"] = trajectory_diag
            result["entropy_centroid"] = trajectory_diag.get("entropy_centroid", 0.0)
            result["late_hep_mass"] = trajectory_diag.get("late_hep_mass", 0)
            result["tail_entropy_phase_ratio"] = trajectory_diag.get("tail_entropy_phase_ratio", 0.0)
    else:
        llm_confidence = 0.5
        if result.get("black_box_api_generator"):
            result["llm_confidence_source"] = "black_box_default_no_logprobs"

    result["llm_raw_output"] = text
    result["llm_answer"] = llm_answer
    result["llm_confidence"] = llm_confidence
    black_box_commit_policy = _effective_black_box_commit_policy(args, result)
    result["black_box_commit_policy"] = black_box_commit_policy

    # --- Step 3: HCEG + 证据检索 (非 SCM-CoT/selective_evidence/table_focus/table_pruned 模式) ---
    prompt_style = getattr(args, "prompt_style", "structure_aware")
    need_graph_before_llm = (
        use_scm_cot or prompt_style in ("selective_evidence", "table_focus", "table_pruned")
    ) and mode in ("full", "full_cert")

    if not need_graph_before_llm and mode in ("full", "full_cert"):
        try:
            graph, evidence, hceg_stats = _build_hceg_and_retrieve(table_json, question, getattr(args, "structural_prior_weighting", False))
            result.update(hceg_stats)
        except Exception as e:
            raise RuntimeError("HCEG construction/evidence retrieval failed") from e

    # --- Step 4: 结构干预 + SCCI (full_cert 模式) ---
    if mode == "full_cert":
        try:
            graph, evidence = _require_hceg_state(graph, evidence, "intervention generation")
            engine = InterventionEngine(graph, evidence)
            interventions = engine.generate_interventions()
            result["num_interventions"] = len(interventions)
            if interventions:
                result["intervention_types"] = [iv.intervention_type.value for iv in interventions]
        except Exception as e:
            raise RuntimeError("Intervention generation failed") from e

    # --- Step 5: 仲裁 ---
    if mode == "baseline_a_plus":
        final_answer = llm_answer
        answer_source = "llm_only"
        final_confidence = llm_confidence
    elif mode == "full_cert":
        graph, evidence, interventions, all_exec_candidates = _require_full_cert_state(
            graph,
            evidence,
            interventions,
            all_exec_candidates,
        )
        _conformal = getattr(args, "_conformal_abstainer", None)
        _sp = getattr(args, "_success_predictor", None)
        final_answer, answer_source, final_confidence, cert_info = certificate_aware_arbitrate(
            llm_answer=llm_answer,
            llm_confidence=llm_confidence,
            executor_result=executor_result,
            table_json=table_json,
            question=question,
            evidence=evidence,
            graph=graph,
            interventions=interventions,
            scci_threshold=0.1,
            all_candidates=all_exec_candidates,
            conformal_abstainer=_conformal,
            success_predictor=_sp,
            pipeline_result=result,
        )
        result["certificate_info"] = cert_info
    else:
        final_answer, answer_source, final_confidence = arbitrate(
            llm_answer=llm_answer,
            executor_result=executor_result,
            llm_confidence=llm_confidence,
        )
    cera_repair = None
    if mode == "full_cert" and bool(getattr(args, "enable_cera_repair", False)):
        try:
            graph, evidence = _require_hceg_state(graph, evidence, "CERA repair shadow")
            heuristic_diag = summarize_legacy_heuristics(result)
            result.update(heuristic_diag)
            cera_repair = run_causal_epistemic_repair(
                question=question,
                original_answer=llm_answer,
                cert_info=result.get("certificate_info", {}),
                graph=graph,
                evidence=evidence,
                table_json=table_json,
                all_exec_candidates=all_exec_candidates,
                generator=generator,
                args=args,
                result_context=build_method_inference_context(result),
                legacy_heuristic_usage_count=int(result.get("legacy_heuristic_usage_count", 0) or 0),
            )
            result.update(cera_repair.to_prediction_fields(
                log_full_prompt=bool(getattr(args, "cera_log_full_prompt", False)),
                log_evidence_packet=bool(getattr(args, "cera_log_evidence_packet", False)),
            ))
        except Exception as e:
            if bool(getattr(args, "cera_strict_debug", False)):
                raise RuntimeError("CERA repair shadow failed while enabled") from e
            result.update({
                "cera_enabled": True,
                "cera_packet_built": False,
                "cera_stage": str(getattr(args, "cera_stage", "E71")),
                "cera_triggered": False,
                "cera_shadow_only": True,
                "cera_final_committed": False,
                "cera_runtime_error": str(e),
                "cera_reject_reason": "runtime_error",
                "cera_validator_accept": False,
                "cera_would_commit": False,
                "cera_would_keep": False,
                "cera_insufficient": False,
            })

    if _should_freeze_black_box_answer(black_box_commit_policy, result):
        result["black_box_answer_freeze_active"] = True
        if str(llm_answer).strip():
            answer_changed_by_freeze = _canonical_answer_key(final_answer) != _canonical_answer_key(llm_answer)
            source_changed_by_freeze = answer_source != "llm_only"
            if (
                answer_changed_by_freeze
                or source_changed_by_freeze
            ):
                result["final_answer_pre_black_box_freeze"] = final_answer
                result["answer_source_pre_black_box_freeze"] = answer_source
                result["final_confidence_pre_black_box_freeze"] = final_confidence
                result["black_box_answer_freeze_applied"] = True
                if answer_changed_by_freeze:
                    result["black_box_answer_freeze_answer_changed"] = True
                elif source_changed_by_freeze:
                    result["black_box_answer_freeze_source_only"] = True
            final_answer = llm_answer
            answer_source = "black_box_frozen_llm"
            final_confidence = llm_confidence
        else:
            result["black_box_answer_freeze_skipped_empty_llm"] = True

    result["final_answer"] = final_answer
    result["answer_source"] = answer_source
    result["final_confidence"] = final_confidence
    cera_commit_requested = bool(
        getattr(args, "cera_commit_approved_repair", False)
    )
    repaired_answer = ""

    if cera_commit_requested:
        result["cera_commit_applied"] = False

        if cera_repair is None:
            result["cera_commit_block_reason"] = "no_cera_result"

        else:
            cera_output = getattr(cera_repair, "output", None)

            repaired_answer = str(
                getattr(cera_output, "final_answer", "") or ""
            ).strip()

            can_commit, commit_block_reason = _cera_commit_authorized(
                args,
                cera_repair,
                repaired_answer,
            )

            if can_commit:
                result["final_answer_pre_cera_commit"] = final_answer
                result["answer_source_pre_cera_commit"] = answer_source

                final_answer = repaired_answer
                answer_source = "cera_v3_validated_repair"

                repair_confidence = getattr(
                    cera_output,
                    "self_assessed_confidence",
                    None,
                )
                if repair_confidence is not None:
                    final_confidence = float(repair_confidence)

                cera_repair.final_committed = True

                result["cera_commit_applied"] = True
                result["cera_commit_block_reason"] = ""

            else:
                result["cera_commit_block_reason"] = commit_block_reason

            # Refresh prediction fields after the commit state changes.
            result.update(
                cera_repair.to_prediction_fields(
                    log_full_prompt=bool(
                        getattr(args, "cera_log_full_prompt", False)
                    ),
                    log_evidence_packet=bool(
                        getattr(args, "cera_log_evidence_packet", False)
                    ),
                )
            )
    result["final_answer"] = final_answer
    result["answer_source"] = answer_source
    result["final_confidence"] = final_confidence
    result["normalized_answer_candidates"] = normalize_numeric_answer(str(final_answer))[:8]
    result["cera_commit_requested"] = bool(
        cera_commit_requested
    )

    result["cera_commit_gate"] = {
        "has_cera_result": cera_repair is not None,
        "stage": str(
            getattr(cera_repair, "stage", "")
        ) if cera_repair is not None else "",
        "stage_ok": bool(
            cera_repair is not None
            and str(getattr(args, "cera_stage", "") or "").upper() == "E72"
            and not bool(getattr(args, "cera_shadow_only", True))
        ),
        "validator_accept": bool(
            getattr(
                cera_repair,
                "validator_accept",
                False,
            )
        ) if cera_repair is not None else False,
        "would_commit": bool(
            getattr(
                cera_repair,
                "would_commit",
                False,
            )
        ) if cera_repair is not None else False,
        "repaired_answer_nonempty": bool(
            repaired_answer
        ),
    }
    api_format_mode = _effective_api_format_normalizer(args, result, black_box_commit_policy)
    result["api_format_normalizer_mode"] = api_format_mode
    if api_format_mode == "conservative":
        try:
            new_answer = deployable_normalize_answer(
                str(final_answer),
                question,
                item.get("dataset", getattr(args, "dataset", "hitab")),
            )
            if new_answer is not None and str(new_answer) != str(final_answer):
                result["final_answer_pre_api_format_normalizer"] = final_answer
                result["answer_source_pre_api_format_normalizer"] = answer_source
                result["final_answer"] = new_answer
                result["answer_source"] = "api_format_normalized"
                result["api_format_normalizer_applied"] = True
                final_answer = new_answer
                answer_source = "api_format_normalized"
                result["normalized_answer_candidates"] = normalize_numeric_answer(str(final_answer))[:8]
        except Exception as e:
            raise RuntimeError("API format normalizer failed") from e

    # --- v9.0b: Online Normalizer ---
    # 主路径只允许 gold-free 的表面格式归一化；读 gold 的版本仅作为 oracle 诊断。
    if getattr(args, "online_normalizer", False):
        try:
            ent = result.get("first_token_entropy")
            if ent is None:
                ent = 1.0
            if result.get("answer_source") != "path_verified_consensus" and ent < 0.05:
                new_answer = deployable_normalize_answer(
                    str(final_answer),
                    question,
                    item.get("dataset", getattr(args, "dataset", "hitab")),
                )
                if new_answer is not None and str(new_answer) != str(final_answer):
                    result["final_answer_pre_normalizer"] = final_answer
                    result["final_answer"] = new_answer
                    result["normalizer_applied"] = True
                    result["normalizer_mode"] = "deployable"
                    final_answer = new_answer
                    result["normalized_answer_candidates"] = normalize_numeric_answer(str(final_answer))[:8]
        except Exception as e:
            raise RuntimeError("Online normalizer failed") from e

    # --- Step 5.5: Credal Probe 诊断（v8.6 纯只读，不改变答案）---
    if getattr(args, "credal_probe", False):
        try:
            graph, evidence = _require_hceg_state(graph, evidence, "credal_probe")
            _logprobs_list = gen_output.get("logprobs") if gen_output else None
            probe_diag = compute_probe_diagnostics(
                graph=graph,
                evidence=evidence,
                executor_result=executor_result,
                cert_info=result.get("certificate_info", {}),
                first_token_entropy=result.get("first_token_entropy", 0.0),
                llm_confidence=llm_confidence,
                logprobs_list=_logprobs_list,
            )
            result["probe_diagnostics"] = probe_diag
        except Exception as e:
            raise RuntimeError("Credal probe diagnostics failed") from e

    # --- Step 5.6: HCEG-Fallback (v9.1) ---
    # 高 cw 或 compare 错误时，用 KG 直检替换 LLM 答案
    if getattr(args, "hceg_fallback", False):
        try:
            graph, evidence = _require_hceg_state(graph, evidence, "hceg_fallback")
            probe = result.get("probe_diagnostics") or {}
            credal_section = probe.get("credal_probe") or {}
            cw_val = credal_section.get("credal_width")
            if cw_val is None:
                cw_val = 0.0
            cw_threshold = getattr(args, "hceg_fallback_cw", 0.30)
            compare_thr = getattr(args, "hceg_fallback_compare_cw", 0.15)
            diff_thr = getattr(args, "hceg_fallback_diff_cw", 0.10)
            fallback_policy = getattr(args, "hceg_fallback_policy", "candidate_only")
            hceg_role_aware = bool(getattr(args, "hceg_role_aware", False))
            diagnostic_policy = getattr(args, "hceg_diagnostic_candidates", "triggered")
            coarse_type = result.get("coarse_question_type", "")
            question_operation = result.get("question_operation", "")
            expected_role_info = _infer_answer_role_commitment(
                question,
                coarse_type,
                question_operation,
                surface_heuristic_mode=getattr(args, "surface_heuristic_mode", "diagnostic"),
            )
            expected_role = expected_role_info.get("answer_role", "unknown")
            should_trigger, reason = should_trigger_fallback(
                credal_width=float(cw_val),
                coarse_type=coarse_type,
                question_operation=question_operation,
                answer_source=result.get("answer_source", ""),
                cw_threshold=cw_threshold,
                compare_cw_threshold=compare_thr,
                diff_cw_threshold=diff_thr,
            )
            role_sensitive = (
                expected_role == "entity"
                and (
                    coarse_type in {"compare", "superlative"}
                    or question_operation in {"compare", "argmax", "argmin"}
                )
            )
            diagnostic_candidate = False
            should_generate_candidate = bool(should_trigger)
            if not should_generate_candidate:
                if diagnostic_policy == "all":
                    should_generate_candidate = True
                    diagnostic_candidate = True
                elif diagnostic_policy == "role_sensitive" and role_sensitive:
                    should_generate_candidate = True
                    diagnostic_candidate = True
            result["hceg_fallback_should_trigger"] = should_trigger
            result["hceg_fallback_trigger_reason"] = reason
            result["hceg_fallback_policy"] = fallback_policy
            result["hceg_fallback_role_aware"] = hceg_role_aware
            result["hceg_fallback_expected_role"] = expected_role
            result["hceg_fallback_expected_role_source"] = expected_role_info.get("role_source", "unknown")
            result["hceg_fallback_expected_role_source_is_surface_heuristic"] = bool(
                expected_role_info.get("role_source_is_surface_heuristic")
            )
            result["hceg_fallback_surface_heuristic_mode"] = expected_role_info.get("surface_heuristic_mode")
            result["hceg_fallback_surface_expected_role"] = expected_role_info.get("surface_answer_role")
            result["hceg_fallback_surface_role_source"] = expected_role_info.get("surface_role_source")
            result["hceg_fallback_structural_expected_role"] = expected_role_info.get("structural_answer_role")
            result["hceg_fallback_structural_role_source"] = expected_role_info.get("structural_role_source")
            result["hceg_fallback_surface_structural_role_agreement"] = (
                expected_role_info.get("surface_structural_role_agreement")
            )
            result["hceg_fallback_diagnostic_policy"] = diagnostic_policy
            result["hceg_fallback_role_sensitive"] = role_sensitive
            result["hceg_fallback_diagnostic_candidate"] = diagnostic_candidate
            if should_generate_candidate:
                raw_fallback_ans = hceg_direct_retrieve(
                    graph=graph,
                    evidence=evidence,
                    question=question,
                    coarse_type=coarse_type,
                    question_operation=question_operation,
                    role_aware=False,
                )
                result["hceg_fallback_raw_candidate"] = raw_fallback_ans
                fallback_ans = raw_fallback_ans
                if hceg_role_aware:
                    role_fallback_ans = hceg_direct_retrieve(
                        graph=graph,
                        evidence=evidence,
                        question=question,
                        coarse_type=coarse_type,
                        question_operation=question_operation,
                        role_aware=True,
                    )
                    result["hceg_fallback_role_aware_candidate"] = role_fallback_ans
                    fallback_ans = role_fallback_ans or raw_fallback_ans
                    result["hceg_fallback_role_aware_changed"] = (
                        raw_fallback_ans is not None
                        and fallback_ans is not None
                        and _canonical_answer_key(raw_fallback_ans) != _canonical_answer_key(fallback_ans)
                    )
                if fallback_ans:
                    result["hceg_fallback_candidate"] = fallback_ans
                    result["hceg_fallback_candidate_type"] = _answer_surface_type(fallback_ans)
                    candidate_compatible = _hceg_candidate_compatible(
                        fallback_ans,
                        question,
                        result.get("coarse_question_type", ""),
                        result.get("question_operation", ""),
                    )
                    result["hceg_fallback_candidate_compatible"] = candidate_compatible
                    result["hceg_fallback_role_mismatch"] = not candidate_compatible
                    can_replace = should_trigger and fallback_policy == "replace"
                    if fallback_policy == "conservative":
                        exec_ans = result.get("executor_answer")
                        can_replace = (
                            should_trigger
                            and exec_ans is not None
                            and _canonical_answer_key(fallback_ans) == _canonical_answer_key(exec_ans)
                            and candidate_compatible
                        )
                    if can_replace and not _black_box_semantic_commit_allowed(args, result):
                        result["black_box_semantic_commit_blocked"] = True
                        result["black_box_semantic_commit_blocked_stage"] = "hceg_fallback"
                        can_replace = False
                    if can_replace and str(fallback_ans).strip() != str(final_answer).strip():
                        result["final_answer_pre_hceg_fallback"] = final_answer
                        result["final_answer"] = fallback_ans
                        result["answer_source"] = "hceg_fallback"
                        result["hceg_fallback_applied"] = True
                        final_answer = fallback_ans
                    if getattr(args, "certificate_commit_boundary", False):
                        decision = _certificate_commit_boundary(result, fallback_ans, args, graph=graph)
                        result["certificate_commit_candidate"] = True
                        result["certificate_commit_decision"] = decision.get("decision")
                        result["certificate_commit_recommended"] = decision.get("decision") == "commit"
                        result["certificate_commit_reject_reasons"] = decision.get("reject_reasons", [])
                        result["certificate_commit_dominance_tuple"] = decision.get("dominance_tuple", [])
                        result["certificate_commit_shadow_decision"] = decision.get("shadow_decision")
                        result["certificate_commit_shadow_recommended"] = decision.get("shadow_decision") == "commit"
                        result["certificate_commit_shadow_reject_reasons"] = decision.get("shadow_reject_reasons", [])
                        result["certificate_commit_shadow_dominance_tuple"] = decision.get("shadow_dominance_tuple", [])
                        result["certificate_commit_features"] = decision.get("features", {})
                        op_features = decision.get("features", {}).get("operation_verifier_features", {})
                        result["certificate_commit_operation_verified"] = (
                            decision.get("features", {}).get("operation_verifier_pass") is True
                        )
                        result["certificate_commit_operation_reject_reasons"] = (
                            decision.get("features", {}).get("operation_verifier_reject_reasons", [])
                        )
                        result["certificate_commit_operation_features"] = op_features
                        dir_features = decision.get("features", {}).get(
                            "compare_direction_verifier_features", {}
                        )
                        result["certificate_commit_compare_direction_verified"] = (
                            decision.get("features", {}).get("compare_direction_verifier_enabled") is True
                            and decision.get("features", {}).get("compare_direction_verifier_pass") is True
                        )
                        result["certificate_commit_compare_direction_reject_reasons"] = (
                            decision.get("features", {}).get(
                                "compare_direction_verifier_reject_reasons",
                                [],
                            )
                        )
                        result["certificate_commit_compare_direction_features"] = dir_features
                        numdir_features = decision.get("features", {}).get(
                            "numeric_direction_verifier_features", {}
                        )
                        result["certificate_commit_numeric_direction_verified"] = (
                            decision.get("features", {}).get("numeric_direction_verifier_enabled") is True
                            and decision.get("features", {}).get("numeric_direction_verifier_verified") is True
                        )
                        result["certificate_commit_numeric_direction_reject_reasons"] = (
                            decision.get("features", {}).get(
                                "numeric_direction_verifier_reject_reasons",
                                [],
                            )
                        )
                        result["certificate_commit_numeric_direction_features"] = numdir_features
                        result["certificate_commit_conformal_score"] = (
                            decision.get("features", {}).get("conformal_score")
                        )
                        conformal_score_pass = (
                            decision.get("features", {}).get("conformal_boundary_enabled") is True
                            and decision.get("features", {}).get("conformal_accept") is True
                        )
                        result["certificate_commit_conformal_accepted"] = (
                            conformal_score_pass
                        )
                        result["certificate_commit_conformal_score_pass"] = (
                            conformal_score_pass
                        )
                        result["certificate_commit_conformal_recommended"] = (
                            conformal_score_pass and decision.get("decision") == "commit"
                        )
                        result["certificate_commit_conformal_shadow_accepted"] = (
                            conformal_score_pass and decision.get("shadow_decision") == "commit"
                        )
                        result["certificate_commit_conformal_threshold"] = (
                            decision.get("features", {}).get("conformal_threshold")
                        )
                        can_commit = (
                            decision.get("decision") == "commit"
                            and getattr(args, "certificate_commit_mode", "diagnostic") == "conservative"
                        )
                        if can_commit and not _black_box_semantic_commit_allowed(args, result):
                            result["black_box_semantic_commit_blocked"] = True
                            result["black_box_semantic_commit_blocked_stage"] = "certificate_commit"
                            can_commit = False
                        if can_commit and str(fallback_ans).strip() != str(final_answer).strip():
                            result["final_answer_pre_certificate_commit"] = final_answer
                            result["answer_source_pre_certificate_commit"] = result.get("answer_source", "")
                            result["final_answer"] = fallback_ans
                            result["answer_source"] = "certificate_commit_boundary"
                            result["certificate_commit_applied"] = True
                            final_answer = fallback_ans
        except Exception as e:
            raise RuntimeError("HCEG fallback failed while enabled") from e

    final_answer = result.get("final_answer", final_answer)
    answer_source = result.get("answer_source", answer_source)
    final_confidence = result.get("final_confidence", final_confidence)
    final_answer, answer_source, final_confidence = _maybe_apply_operation_commit_gate(
        result,
        args,
        final_answer,
        answer_source,
        final_confidence,
    )

    _apply_source_risk_calibration(result, args)
    final_answer = result.get("final_answer", final_answer)
    final_confidence = result.get("final_confidence", final_confidence)
    result["main_cert_profile"] = bool(getattr(args, "main_cert_profile", False))
    result["legacy_commit_path_used"] = bool(
        result.get("hceg_fallback_applied")
        or result.get("certificate_commit_applied")
    )
    result["non_certificate_answer_mutation_used"] = bool(
        result.get("api_format_normalizer_applied")
        or result.get("normalizer_applied")
        or result.get("oracle_normalizer_applied")
        or result.get("hceg_fallback_applied")
        or result.get("certificate_commit_applied")
        or result.get("self_consistency_changed")
    )
    result["heuristic_surface_used_for_commit"] = bool(
        result.get("operation_support_commit_role_source_is_surface_heuristic")
        or result.get("operation_support_commit_surface_named_entity_anchor_used")
    )
    result.update(summarize_legacy_heuristics(result))

    # Gold and correctness enter only after all live method inference returns.
    gold_answer = item.get("answer", "")
    evaluation_record = PosthocEvaluationRecord(gold_answer=gold_answer)
    gold_answer = evaluation_record.to_dict()["gold_answer"]
    result["gold_answer"] = gold_answer

    gold_aligned = align_to_gold_form(str(final_answer), gold_answer)
    if gold_aligned is not None:
        result["gold_aligned_numeric_answer"] = gold_aligned

    if getattr(args, "oracle_online_normalizer", False):
        try:
            ent = result.get("first_token_entropy")
            if ent is None:
                ent = 1.0
            if (
                gold_aligned is not None
                and result.get("answer_source") != "path_verified_consensus"
                and ent < 0.05
            ):
                cands = result["normalized_answer_candidates"]
                gold_raw = gold_answer
                if isinstance(gold_raw, list):
                    gold_strs = [str(g).strip().lower() for g in gold_raw]
                else:
                    gold_strs = [str(gold_raw).strip().lower()]
                fa_str = str(final_answer).strip().lower()
                if fa_str not in gold_strs:
                    new_answer = next(
                        (
                            candidate
                            for candidate in cands
                            if str(candidate).strip().lower() in gold_strs
                        ),
                        None,
                    )
                    if new_answer is None:
                        aligned_text = str(gold_aligned)
                        if aligned_text.strip().lower() in gold_strs:
                            new_answer = aligned_text
                    if new_answer is not None and str(new_answer) != str(final_answer):
                        result["final_answer_pre_oracle_normalizer"] = final_answer
                        result["final_answer"] = new_answer
                        result["oracle_normalizer_applied"] = True
                        result["normalizer_mode"] = "oracle_posthoc"
                        result["non_certificate_answer_mutation_used"] = True
                        final_answer = new_answer
        except Exception as e:
            raise RuntimeError("Post-hoc oracle normalizer failed") from e

    # --- Step 6: 四口径 EM 评估 ---
    em_results = evaluate_answer_for_dataset(
        final_answer,
        gold_answer,
        item.get("dataset", getattr(args, "dataset", "hitab")),
    )
    result.update(em_results)
    if result.get("cera_enabled"):
        result.update(compute_repair_outcome(
            result,
            dataset=item.get("dataset", getattr(args, "dataset", "hitab")),
        ))

    _attach_operation_support_outcome(
        result,
        gold_answer=gold_answer,
        dataset=item.get("dataset", getattr(args, "dataset", "hitab")),
    )

    # --- 错误类型分类 ---
    aggregation = item.get("aggregation", ["none"])
    result["error_type"] = classify_error(final_answer, gold_answer, aggregation)
    result["post_llm_finalize_seconds"] = time.time() - finalize_start
    result["pipeline_recorded_seconds"] = (
        float(result.get("non_llm_preparation_seconds", 0.0) or 0.0)
        + float(result.get("llm_generation_seconds", 0.0) or 0.0)
        + float(result.get("post_llm_finalize_seconds", 0.0) or 0.0)
    )

    return result


# ---------------------------------------------------------------------------
# 向后兼容的 process_single（内部调用新的两阶段函数）
# ---------------------------------------------------------------------------

def process_single(
    item: Dict[str, Any],
    table_json: Dict[str, Any],
    generator: Optional[VLLMGeneratorWithLogprobs],
    args: argparse.Namespace,
    mode: str,
) -> Dict[str, Any]:
    """处理单个样本，返回预测结果 dict

    向后兼容接口：内部使用 prepare_non_llm_steps + finalize_after_llm 实现。
    非批量推理模式下仍逐样本调用 generator.generate()。
    """
    top_k_logprobs = args.top_k_logprobs if hasattr(args, "top_k_logprobs") else 5

    prepared = prepare_non_llm_steps(item, table_json, args, mode)

    gen_output = {"text": "", "logprobs": None}
    if generator is not None:
        if getattr(args, "skip_overlong_primary", False):
            _keep, _prompts, skipped, _audit = filter_prompts_by_context_budget(
                generator=generator,
                prompts=[prepared["prompt"]],
                max_model_len=_effective_max_model_len(generator, args),
                max_new_tokens=args.max_answer_tokens,
                logger=logging.getLogger("cscr_pipeline"),
                ids=[prepared["item"].get("id")],
            )
            if skipped:
                return _make_primary_context_overflow_result(prepared, skipped[0])
        else:
            assert_no_truncation(
                generator=generator,
                prompts=[prepared["prompt"]],
                max_model_len=_effective_max_model_len(generator, args),
                max_new_tokens=args.max_answer_tokens,
                logger=logging.getLogger("cscr_pipeline"),
                ids=[prepared["item"].get("id")],
            )
        record, ref = _make_llm_input_audit_record(
            prepared=prepared,
            prompt=prepared["prompt"],
            generator=generator,
            args=args,
            prompt_kind="primary",
            prompt_type=prepared.get("result", {}).get("prompt_type", ""),
            max_new_tokens=args.max_answer_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            logprobs=top_k_logprobs,
            sequence_index=0,
        )
        prepared["result"]["llm_input_audit"] = ref
        if record is not None:
            _append_llm_input_audit(args, [record])

        gen_start = time.time()
        gen_results = generator.generate(
            [prepared["prompt"]],
            max_new_tokens=args.max_answer_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            logprobs=top_k_logprobs,
        )
        gen_output = gen_results[0]
        gen_output["generation_seconds"] = time.time() - gen_start

    return finalize_after_llm(prepared, gen_output, args, mode, top_k_logprobs, generator=generator)


def _should_run_self_consistency(result: Dict[str, Any], args: argparse.Namespace) -> bool:
    if result.get("answer_source") in ("path_verified_consensus", "consensus_cert"):
        return False
    trigger = getattr(args, "self_consistency_trigger", "hceg")
    hceg_risk = bool(result.get("hceg_fallback_should_trigger"))
    if trigger == "all":
        return True
    if trigger == "hceg":
        return hceg_risk
    ent = float(result.get("first_token_entropy", 0.0) or 0.0)
    entropy_risk = ent >= float(getattr(args, "entropy_threshold_low", 0.05))
    if trigger == "entropy":
        return entropy_risk
    if hceg_risk or entropy_risk:
        return True
    tier = (result.get("probe_diagnostics") or {}).get("probe_risk_tier", "")
    return tier in ("medium_risk", "high_risk")


def _common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    k = 0
    while k < n and a[k] == b[k]:
        k += 1
    return k


def build_prefix_stable_apr_prompt(
    prepared: Dict[str, Any],
    old_result: Dict[str, Any],
    suffix_mode: str = "intersection_hint",
) -> str:
    """
    Prefix-stable APR / self-consistency prompt.

    设计目标：
    1) 原始完整输入不删除、不截断、不重排；
    2) Round-2 / SC 只在 Round-1 prompt 之后追加短控制后缀；
    3) 让 vLLM prefix caching 能复用 Round-1 的长前缀 KV；
    4) 把多轮推理从“重新输入世界”改成“同一感知场上的任务集切换”。
    """
    base_prompt = prepared.get("prompt", "") or ""
    item = prepared.get("item", {}) or {}
    question = item.get("question", "")

    result = prepared.get("result", {}) or {}
    executor_answer = result.get("executor_answer", "")
    executor_trace = result.get("executor_trace", "")
    question_operation = result.get("question_operation", "")
    coarse_type = old_result.get("coarse_question_type", result.get("coarse_question_type", ""))

    r1_answer = old_result.get("llm_answer", "")
    r1_final = old_result.get("final_answer", "")
    r1_entropy = old_result.get("first_token_entropy", None)
    r1_conf = old_result.get("final_confidence", None)
    answer_source = old_result.get("answer_source", "")

    if suffix_mode == "minimal":
        suffix = f"""

[ROUND-2 CONTROL: PREFIX-STABLE / MINIMAL]
Re-use the complete table/context above. Do not assume any missing context.
Question: {question}
Round-1 answer: {r1_answer}
Return only the final answer after a stricter evidence check.
"""
    elif suffix_mode == "causal_check":
        suffix = f"""

[ROUND-2 CONTROL: PREFIX-STABLE / CAUSAL-CHECK]
You have already received the complete table/context above. Re-use it exactly.
Question: {question}
Round-1 answer: {r1_answer}
Round-1 final answer after arbitration: {r1_final}
Uncertainty: first_token_entropy={r1_entropy}, final_confidence={r1_conf}, answer_source={answer_source}
Task hints: coarse_question_type={coarse_type}, question_operation={question_operation}
Executor weak signal: executor_answer={executor_answer}; executor_trace={executor_trace}

Causal verification policy:
1. Identify the target variable asked by the question.
2. Identify the minimal table evidence that causes/supports the target answer.
3. Reject answers supported only by lexical proximity or unrelated rows/columns.
4. If Round-1 is unsupported, correct it using the complete context above.
5. Return only the final answer in a concise form.
"""
    else:
        suffix = f"""

[APR ROUND-2 CONTROL SUFFIX: PREFIX-STABLE]
You have already received the complete table/context above. Do not ignore, replace, summarize, or truncate it.
Now re-check the answer using a stricter evidence-intersection policy.

Original question:
{question}

Round-1 answer:
{r1_answer}

Round-1 final answer after arbitration:
{r1_final}

Round-1 uncertainty:
first_token_entropy={r1_entropy}, final_confidence={r1_conf}, answer_source={answer_source}

Task type hints:
coarse_question_type={coarse_type}, question_operation={question_operation}

Executor-side weak signal, if available:
executor_answer={executor_answer}
executor_trace={executor_trace}

Re-evaluation policy:
1. Re-use the full table/context above. Do not request or assume missing context.
2. Identify the smallest evidence intersection that supports the answer.
3. If Round-1 answer is unsupported by the table/context, correct it.
4. If the evidence is ambiguous, output the best directly supported answer, not a long explanation.
5. Return only the final answer in a concise form.
"""
    return base_prompt.rstrip() + suffix


def _build_self_consistency_variants(
    prepared: Dict[str, Any],
    result: Dict[str, Any],
    args: argparse.Namespace,
) -> List[Tuple[str, str]]:
    table_json = prepared.get("table_json") or {}
    question = prepared.get("item", {}).get("question", "")
    graph = prepared.get("graph")
    evidence = prepared.get("evidence")
    current_type = result.get("prompt_type", "")
    dataset_prompt_policy = result.get("dataset_prompt_policy", getattr(args, "dataset_prompt_policy", "auto"))

    # Full-input compute reachability branch: input prefix is unchanged; only control suffix varies.
    if getattr(args, "prefix_stable_apr", False):
        variants: List[Tuple[str, str]] = []
        suffix_modes = ["intersection_hint", "causal_check", "minimal"]
        for suffix_mode in suffix_modes:
            prompt = build_prefix_stable_apr_prompt(
                prepared=prepared,
                old_result=result,
                suffix_mode=suffix_mode,
            )
            variants.append((f"prefix_stable_{suffix_mode}", prompt))
        return variants[: max(1, int(getattr(args, "k_samples", 3)) - 1)]

    variants: List[Tuple[str, str]] = []
    builders = [
        ("baseline_e", lambda: build_baseline_e_prompt(table_json, question, dataset_prompt_policy=dataset_prompt_policy)),
        ("table_focus", lambda: build_table_focus_prompt(table_json, question, evidence, graph, dataset_prompt_policy=dataset_prompt_policy)),
        ("intersection_hint", lambda: build_intersection_hint_prompt(table_json, question, evidence, graph)),
    ]
    for name, builder in builders:
        if name == current_type:
            continue
        prompt = builder()
        if prompt:
            variants.append((name, prompt))
    if not variants and prepared.get("prompt"):
        variants.append((f"{current_type or 'current'}_sample", prepared["prompt"]))
    return variants[: max(1, int(getattr(args, "k_samples", 3)) - 1)]


def _merge_self_consistency_results(
    base_result: Dict[str, Any],
    alt_results: List[Dict[str, Any]],
    item: Dict[str, Any],
) -> Dict[str, Any]:
    candidates = [base_result] + alt_results
    groups: Dict[str, Dict[str, Any]] = {}
    source_bonus = {
        "path_verified_consensus": 0.8,
        "consensus_cert": 0.6,
        "hceg_fallback": 0.5,
        "llm_cert_adjusted": 0.25,
        "llm_only": 0.0,
    }
    for cand in candidates:
        key = _canonical_answer_key(cand.get("final_answer", ""))
        if not key:
            continue
        entry = groups.setdefault(key, {"score": 0.0, "votes": 0, "best": cand})
        entry["votes"] += 1
        src = cand.get("answer_source", "")
        conf = float(cand.get("final_confidence", 0.5) or 0.5)
        entry["score"] += 1.0 + source_bonus.get(src, 0.0) + 0.05 * conf
        best = entry["best"]
        if source_bonus.get(src, 0.0) > source_bonus.get(best.get("answer_source", ""), 0.0):
            entry["best"] = cand

    base_key = _canonical_answer_key(base_result.get("final_answer", ""))
    if not groups:
        out = dict(base_result)
        out["self_consistency_used"] = True
        out["self_consistency_changed"] = False
        out["self_consistency_empty_vote_group"] = True
        out["self_consistency_candidates"] = [
            {
                "answer": c.get("final_answer", ""),
                "source": c.get("answer_source", ""),
                "prompt_type": c.get("prompt_type", ""),
                "confidence": c.get("final_confidence", 0.0),
            }
            for c in candidates
        ]
        out["self_consistency_vote_summary"] = {}
        return out

    best_key, best_entry = max(groups.items(), key=lambda kv: (kv[1]["score"], kv[1]["votes"]))
    changed = best_key != base_key and best_entry["votes"] >= 2

    out = dict(base_result)
    out["self_consistency_used"] = True
    out["self_consistency_candidates"] = [
        {
            "answer": c.get("final_answer", ""),
            "source": c.get("answer_source", ""),
            "prompt_type": c.get("prompt_type", ""),
            "confidence": c.get("final_confidence", 0.0),
        }
        for c in candidates
    ]
    out["self_consistency_vote_summary"] = {
        k: {"votes": v["votes"], "score": round(v["score"], 4)}
        for k, v in sorted(groups.items())
    }
    out["self_consistency_changed"] = bool(changed)

    if changed:
        selected = best_entry["best"]
        out["final_answer_pre_self_consistency"] = out.get("final_answer", "")
        out["answer_source_pre_self_consistency"] = out.get("answer_source", "")
        out["final_answer"] = selected.get("final_answer", "")
        out["final_confidence"] = selected.get("final_confidence", out.get("final_confidence", 0.5))
        out["answer_source"] = "self_consistency_vote"
        gold_answer = out.get("gold_answer", item.get("answer", ""))
        out.update(evaluate_answer_for_dataset(
            out["final_answer"],
            gold_answer,
            item.get("dataset", getattr(args, "dataset", "hitab")),
        ))
        out["error_type"] = classify_error(
            out["final_answer"],
            gold_answer,
            item.get("aggregation", ["none"]),
        )
    return out


# ---------------------------------------------------------------------------
# 批量运行
# ---------------------------------------------------------------------------
def assert_no_truncation(generator, prompts, max_model_len, max_new_tokens, logger, ids=None):
    """
    Enforce a strict no-truncation contract against the exact generation input.

    This function intentionally raises instead of clipping tokens.  max_num_batched_tokens
    may be lower than max_model_len under chunked prefill; only max_model_len controls
    semantic input reachability.
    """
    budget = max_model_len - max_new_tokens
    bad = []
    lengths = []
    for i, p in enumerate(prompts):
        if hasattr(generator, "count_generation_prompt_tokens"):
            n = generator.count_generation_prompt_tokens(p)
        else:
            n = len(generator.tokenizer(p, add_special_tokens=False).input_ids)
        lengths.append(n)
        if n > budget:
            bad.append((ids[i] if ids else i, n, budget))

    pressure = [round(n / max(1, budget), 4) for n in lengths]
    logger.info(
        "NO_TRUNCATION_AUDIT prompt_token_lens=%s context_pressure=%s max_input_budget=%s ids=%s",
        lengths, pressure, budget, ids,
    )

    if bad:
        raise RuntimeError(
            "No-truncation contract violated: prompt exceeds max_model_len. "
            f"bad={bad}. Increase CSCR_MAX_LEN or use a long-context model; do not truncate."
        )

    return {
        "lengths": lengths,
        "pressure": pressure,
        "budget": budget,
    }


def filter_prompts_by_context_budget(
    generator,
    prompts,
    max_model_len,
    max_new_tokens,
    logger,
    ids=None,
):
    """
    Audit optional-pass prompts and return prompts that fit the context budget.

    The caller must treat a non-empty skipped list as a configuration error when an
    optional module is explicitly enabled; no prompt is truncated or silently bypassed.

    Primary generation uses assert_no_truncation directly so that a sample whose full
    input cannot fit the model context fails loudly instead of changing semantics.
    """
    budget = max_model_len - max_new_tokens
    keep_positions = []
    keep_prompts = []
    skipped = []
    lengths = []
    pressure = []

    for i, p in enumerate(prompts):
        if hasattr(generator, "count_generation_prompt_tokens"):
            n = generator.count_generation_prompt_tokens(p)
        else:
            n = len(generator.tokenizer(p, add_special_tokens=False).input_ids)
        pr = round(n / max(1, budget), 4)
        lengths.append(n)
        pressure.append(pr)
        sid = ids[i] if ids else i
        if n > budget:
            skipped.append({
                "local_index": i,
                "id": sid,
                "input_token_count": n,
                "budget": budget,
                "context_pressure_ratio": pr,
            })
        else:
            keep_positions.append(i)
            keep_prompts.append(p)

    logger.info(
        "OPTIONAL_PASS_CONTEXT_AUDIT prompt_token_lens=%s context_pressure=%s max_input_budget=%s ids=%s kept=%s skipped=%s",
        lengths, pressure, budget, ids, len(keep_prompts), len(skipped),
    )
    if skipped:
        logger.warning("OPTIONAL_PASS_NO_TRUNCATION_SKIP skipped=%s", skipped)

    return keep_positions, keep_prompts, skipped, {
        "lengths": lengths,
        "pressure": pressure,
        "budget": budget,
    }


def _pressure_tier(pressure: float) -> str:
    if pressure >= 0.90:
        return "critical"
    if pressure >= 0.75:
        return "high"
    if pressure >= 0.50:
        return "medium"
    return "low"


def _effective_max_model_len(generator, args: argparse.Namespace) -> int:
    return int(getattr(generator, "max_model_len", args.max_model_len))


def _make_primary_context_overflow_result(
    prepared: Dict[str, Any],
    skip: Dict[str, Any],
) -> Dict[str, Any]:
    item = prepared.get("item", {})
    result = dict(prepared.get("result", {}))
    gold_answer = item.get("answer", result.get("gold_answer", ""))
    result.update({
        "gold_answer": gold_answer,
        "final_answer": "",
        "llm_answer": "",
        "raw_answer": "",
        "answer_source": "context_overflow_skipped",
        "final_confidence": 0.0,
        "generated_token_count": 0,
        "llm_generation_seconds": 0.0,
        "primary_skipped_no_truncation": True,
        "primary_skipped": "primary_prompt_exceeds_context_budget",
        "primary_skip_reason": (
            f"prompt_tokens={skip['input_token_count']}>budget={skip['budget']}"
        ),
        "input_token_count": int(skip["input_token_count"]),
        "context_budget": int(skip["budget"]),
        "context_pressure_ratio": float(skip["context_pressure_ratio"]),
        "compute_pressure_tier": _pressure_tier(float(skip["context_pressure_ratio"])),
        "error": "primary_prompt_exceeds_context_budget",
        "error_type": "context_overflow",
    })
    result.update(evaluate_answer_for_dataset(
        "",
        gold_answer,
        item.get("dataset", "hitab") if isinstance(item, dict) else "hitab",
    ))
    return result


def _make_error_result(item: Dict[str, Any], args: argparse.Namespace, error: Exception, stage: str) -> Dict[str, Any]:
    result = {
        "id": item.get("id"),
        "table_id": item.get("table_id", ""),
        "dataset": item.get("dataset", getattr(args, "dataset", "hitab")),
        "question": item.get("question", ""),
        "gold_answer": item.get("answer", item.get("answers", "")),
        "final_answer": "",
        "answer_source": f"{stage}_error",
        "final_confidence": 0.0,
        "error": str(error),
        "error_stage": stage,
        "error_type": f"{stage}_error",
        "strict_em": False,
        "numeric_em": False,
        "set_em": False,
        "hitab_official_em": False,
        "sstqa_zh_official_em": False,
    }
    for meta_key in ("dataset_question_type", "dataset_question_subtype", "dataset_difficulty", "row_hierarchy_needed", "paraphrase_group"):
        if meta_key in item:
            result[meta_key] = item.get(meta_key, "")
    return result


def _apply_source_risk_calibration(result: Dict[str, Any], args: argparse.Namespace) -> None:
    mode = str(getattr(args, "source_risk_calibration", "auto") or "auto").lower()
    if mode == "off":
        return
    dataset = normalize_dataset_name(result.get("dataset", getattr(args, "dataset", "hitab")))
    if mode == "auto" and dataset != "tablebench":
        return
    if mode == "tablebench" and dataset != "tablebench":
        return
    source = str(result.get("answer_source", "") or "")
    if source != "llm_cert_adjusted":
        return
    cap = _safe_float(getattr(args, "source_risk_llm_cert_adjusted_cap", 0.74), 0.74)
    old_conf = _safe_float(result.get("final_confidence", 0.0), 0.0)
    if old_conf <= cap:
        return
    result["final_confidence_pre_source_risk_calibration"] = old_conf
    result["final_confidence"] = cap
    result["source_risk_calibration_applied"] = True
    result["source_risk_calibration_mode"] = mode
    result["source_risk_calibration_cap"] = cap


def _primary_metric_key_for_dataset(dataset: str) -> str:
    dataset = normalize_dataset_name(dataset)
    if dataset == "aitqa":
        return "aitqa_official_em"
    if dataset == "tablebench":
        return "tablebench_official_em"
    if dataset == "sstqa_zh":
        return "sstqa_zh_official_em"
    return "hitab_official_em"


def _parse_number_loose(text: Any) -> Optional[float]:
    if text is None:
        return None
    value = str(text).strip().replace(",", "")
    value = re.sub(r"[%,$£€]", "", value)
    match = re.search(r"[-+]?\d+(?:\.\d+)?", value)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _normalise_surface_heuristic_mode(mode: Any) -> str:
    mode = str(mode or "diagnostic").lower()
    return mode if mode in {"off", "diagnostic", "legacy"} else "diagnostic"


def _normalise_operation_commit_version(version: Any) -> str:
    text = str(version or "E67").strip().upper()
    if text in {"E67", "67", "V67"}:
        return "E67"
    if text in {"E65.3", "65.3", "V65.3"}:
        return "E65.3"
    return "E65.4"



def _role_source_is_surface_heuristic(source: Any) -> bool:
    source = str(source or "")
    return source.endswith("_surface") or source == "surface"


def _structural_answer_role_commitment(
    coarse_type: str = "",
    question_operation: str = "",
) -> Dict[str, str]:
    coarse_type = str(coarse_type or "").lower()
    question_operation = str(question_operation or "").lower()
    operation_role = question_operation or "unknown"
    if coarse_type in {"count", "arithmetic", "times", "trend", "proportion"}:
        return {
            "answer_role": "numeric",
            "operation_role": operation_role,
            "role_source": f"coarse:{coarse_type}",
        }
    if question_operation in {"count", "sum", "average", "diff", "difference", "ratio", "proportion"}:
        return {
            "answer_role": "numeric",
            "operation_role": operation_role,
            "role_source": f"operation:{question_operation}",
        }
    if coarse_type in {"compare", "superlative"} or question_operation in {"compare", "argmax", "argmin"}:
        return {
            "answer_role": "entity",
            "operation_role": operation_role,
            "role_source": f"operation:{question_operation or coarse_type}",
        }
    return {
        "answer_role": "unknown",
        "operation_role": operation_role,
        "role_source": "unknown",
    }


def _surface_answer_role_commitment(
    question: str,
    question_operation: str = "",
) -> Dict[str, str]:
    q = (question or "").lower().strip()
    q_compact = " ".join(q.split())
    operation_role = question_operation or "unknown"

    compound_patterns = (
        r"\band\s+which\b",
        r"\band\s+who\b",
        r"\band\s+where\b",
        r"\bwhich\b.+\band\s+how\s+many\b",
        r"\bhow\s+many\b.+\band\s+which\b",
    )
    if any(re.search(pattern, q_compact) for pattern in compound_patterns):
        return {
            "answer_role": "compound",
            "operation_role": operation_role,
            "role_source": "compound_surface",
        }

    numeric_patterns = (
        r"^how\s+many\b",
        r"^how\s+much\b",
        r"^what\s+(?:is|was|were|are)\s+(?:the\s+)?(?:total|sum|average|mean|difference|ratio|percentage|percent|proportion|rate|number|amount|value|score|count)\b",
        r"^what\s+(?:total|sum|average|mean|difference|ratio|percentage|percent|proportion|rate|number|amount|value|score|count)\b",
        r"\b(?:total|average|mean|sum|difference|ratio|percentage|percent|proportion)\s+(?:of|between|across|for)\b",
    )
    if any(re.search(pattern, q_compact) for pattern in numeric_patterns):
        return {
            "answer_role": "numeric",
            "operation_role": operation_role,
            "role_source": "numeric_surface",
        }

    if re.search(r"^(?:in\s+which|what)\s+(?:year|season|round|rank|position)\b", q_compact):
        return {
            "answer_role": "entity_numeric_label",
            "operation_role": operation_role,
            "role_source": "entity_numeric_label_surface",
        }

    entity_patterns = (
        r"^which\b",
        r"^who\b",
        r"^whom\b",
        r"^whose\b",
        r"^where\b",
        r"^in\s+which\b",
        r"^what\s+(?:team|teams|nation|nations|country|countries|player|players|driver|drivers|region|regions|city|cities|school|schools|company|companies|club|clubs|season|year|route|routes|state|states|province|provinces)\b",
        r"\bwhich\s+.+\b(?:has|had|have|won|finished|ranked|scored|earned|drove|recorded|included)\b",
    )
    if any(re.search(pattern, q_compact) for pattern in entity_patterns):
        return {
            "answer_role": "entity",
            "operation_role": operation_role,
            "role_source": "entity_surface",
        }
    return {
        "answer_role": "unknown",
        "operation_role": operation_role,
        "role_source": "unknown",
    }


def _infer_answer_role_commitment(
    question: str,
    coarse_type: str = "",
    question_operation: str = "",
    surface_heuristic_mode: str = "diagnostic",
) -> Dict[str, Any]:
    mode = _normalise_surface_heuristic_mode(surface_heuristic_mode)
    structural = _structural_answer_role_commitment(coarse_type, question_operation)
    surface = (
        _surface_answer_role_commitment(question, question_operation)
        if mode in {"diagnostic", "legacy"}
        else {
            "answer_role": "disabled",
            "operation_role": question_operation or "unknown",
            "role_source": "surface_disabled",
        }
    )
    if mode == "legacy" and surface.get("answer_role") not in {"unknown", "disabled"}:
        primary = surface
        primary_source = "surface_legacy"
    else:
        primary = structural
        primary_source = "structural"
    primary_role = primary.get("answer_role", "unknown")
    surface_role = surface.get("answer_role", "unknown")
    structural_role = structural.get("answer_role", "unknown")
    return {
        "answer_role": primary_role,
        "operation_role": primary.get("operation_role", question_operation or "unknown"),
        "role_source": primary.get("role_source", "unknown"),
        "role_primary_source": primary_source,
        "role_source_is_surface_heuristic": _role_source_is_surface_heuristic(primary.get("role_source")),
        "surface_heuristic_mode": mode,
        "surface_answer_role": surface_role,
        "surface_role_source": surface.get("role_source", "unknown"),
        "structural_answer_role": structural_role,
        "structural_role_source": structural.get("role_source", "unknown"),
        "surface_structural_role_agreement": (
            surface_role == structural_role
            if surface_role not in {"unknown", "disabled"} and structural_role != "unknown"
            else None
        ),
    }


def _entity_surface_requested(question: str) -> bool:
    q_compact = " ".join((question or "").lower().strip().split())
    return bool(re.search(
        r"\b(which|who|whom|where|in which|what (?:country|nation|team|player|driver|city|state|province|club|company|school|university|year|season|color|combination|category|rank|position|film|song|album|river|mountain|region|candidate|party|person|name))\b",
        q_compact,
    ))


def _build_answer_projection_certificate(result: Dict[str, Any]) -> Dict[str, Any]:
    answer_role = str(result.get("operation_support_answer_role") or "unknown")
    surface_role = str(result.get("operation_support_surface_answer_role") or "unknown")
    structural_role = str(result.get("operation_support_structural_answer_role") or "unknown")
    candidate = str(result.get("operation_support_reranked_denotation") or "")
    candidate_surface = _answer_surface_type(candidate)
    entity_projection = answer_role in {"entity", "entity_numeric_label", "compound"}
    reject_reasons: List[str] = []
    if entity_projection and candidate_surface in {"numeric", "mixed"}:
        reject_reasons.append("entity_projection_numeric_candidate")
    if answer_role == "numeric" and candidate_surface not in {"numeric", "mixed"}:
        reject_reasons.append("numeric_projection_non_numeric_candidate")
    projection_role = "entity" if entity_projection else answer_role
    cert = {
        "version": "E65.4_answer_projection",
        "projection_role": projection_role,
        "answer_role": answer_role,
        "surface_answer_role": surface_role,
        "structural_answer_role": structural_role,
        "surface_dependency_used_for_commit": False,
        "surface_structural_role_conflict_observed": (
            surface_role not in {"unknown", "disabled"}
            and structural_role != "unknown"
            and surface_role != structural_role
        ),
        "entity_projection_requested": entity_projection,
        "candidate_surface_type": candidate_surface,
        "certified": not reject_reasons,
        "reject_reasons": reject_reasons,
    }
    result["answer_projection_certificate"] = cert
    return cert


def _question_named_entity_anchors(question: str) -> List[str]:
    text = question or ""
    quoted = re.findall(r"['\"]([^'\"]{3,60})['\"]", text)
    capitalized = re.findall(r"\b(?:[A-Z][A-Za-z0-9&.-]+|[A-Z]{2,})(?:\s+(?:[A-Z][A-Za-z0-9&.-]+|[A-Z]{2,})){1,4}\b", text)
    generic = {
        "Arithmetic Calculation",
        "Table Bench",
        "What Is",
        "Which Is",
        "How Many",
        "How Much",
    }
    anchors: List[str] = []
    for phrase in quoted + capitalized:
        phrase = " ".join(str(phrase).split()).strip(" ?.,;:")
        if len(phrase) < 3 or phrase in generic:
            continue
        if phrase.lower() in {"total", "average", "sum", "rank", "value", "number"}:
            continue
        if phrase not in anchors:
            anchors.append(phrase)
    return anchors[:6]


def _support_named_entity_anchor_covered(question: str, target_cells: Sequence[Dict[str, Any]]) -> Tuple[bool, List[str], List[str]]:
    anchors = _question_named_entity_anchors(question)
    if not anchors:
        return True, [], []
    support_texts: List[str] = []
    for cell in target_cells or []:
        support_texts.append(str(cell.get("value", "") or ""))
        support_texts.extend(str(x) for x in cell.get("row_headers", []) or [])
        support_texts.extend(str(x) for x in cell.get("col_headers", []) or [])
    matched: List[str] = []
    for anchor in anchors:
        if any(
            anchor.lower() in text.lower()
            or _entity_match_score([anchor.lower()], text) >= 0.82
            for text in support_texts
        ):
            matched.append(anchor)
    return bool(matched), anchors, matched


def _serialize_cell_ref(ref: Any) -> Dict[str, Any]:
    return {
        "row": getattr(ref, "row", None),
        "col": getattr(ref, "col", None),
        "value": getattr(ref, "value", ""),
        "row_headers": list(getattr(ref, "row_headers", []) or [])[:6],
        "col_headers": list(getattr(ref, "col_headers", []) or [])[:6],
    }


def _candidate_role_compatible(candidate: Optional[ExecutorResult], answer_role: str) -> bool:
    if candidate is None:
        return False
    surface = _answer_surface_type(candidate.denotation)
    if answer_role == "numeric":
        return surface in {"numeric", "mixed"}
    if answer_role == "entity":
        return surface in {"entity", "mixed", "date"}
    if answer_role == "entity_numeric_label":
        return surface in {"numeric", "entity", "mixed", "other"}
    if answer_role == "compound":
        return surface == "mixed" or bool(re.search(r"[,;/]", str(candidate.denotation or "")))
    return bool(candidate.denotation)


def _split_filter_target_cells(
    candidate: Optional[ExecutorResult],
    answer_role: str,
) -> Dict[str, Any]:
    if candidate is None:
        return {
            "filter_cells": [],
            "target_cells": [],
            "target_labels": [],
            "target_cell_count": 0,
            "filter_cell_count": 0,
            "target_cell_rows": [],
            "target_cell_cols": [],
            "filter_cell_rows": [],
            "filter_cell_cols": [],
            "target_numeric_values": [],
            "filter_numeric_values": [],
        }

    denotation = str(candidate.denotation or "").strip().lower()
    filter_cells: List[Dict[str, Any]] = []
    target_cells: List[Dict[str, Any]] = []
    target_labels: List[str] = []
    target_numeric_values: List[float] = []
    filter_numeric_values: List[float] = []

    for ref in candidate.cells_used:
        cell = _serialize_cell_ref(ref)
        value = str(getattr(ref, "value", "") or "")
        headers = list(getattr(ref, "row_headers", []) or []) + list(getattr(ref, "col_headers", []) or [])
        header_match = bool(denotation) and any(denotation == str(h).strip().lower() for h in headers)
        value_match = bool(denotation) and denotation == value.strip().lower()
        parsed_number = _parse_number_loose(value)
        numeric_value = parsed_number is not None

        if answer_role == "numeric":
            if numeric_value:
                target_cells.append(cell)
                target_numeric_values.append(float(parsed_number))
            else:
                filter_cells.append(cell)
        elif answer_role in {"entity", "entity_numeric_label"}:
            if value_match or header_match:
                target_cells.append(cell)
                if header_match:
                    target_labels.extend([h for h in headers if denotation == str(h).strip().lower()])
            elif numeric_value:
                filter_cells.append(cell)
                filter_numeric_values.append(float(parsed_number))
            else:
                target_cells.append(cell)
        else:
            target_cells.append(cell)
            if numeric_value:
                target_numeric_values.append(float(parsed_number))

    if answer_role == "entity" and not target_cells:
        target_cells = [_serialize_cell_ref(ref) for ref in candidate.cells_used[:1]]
    return {
        "filter_cells": filter_cells[:16],
        "target_cells": target_cells[:16],
        "target_labels": sorted(set(str(x) for x in target_labels))[:8],
        "target_cell_count": len(target_cells),
        "filter_cell_count": len(filter_cells),
        "target_cell_rows": sorted({int(cell["row"]) for cell in target_cells if isinstance(cell.get("row"), int)}),
        "target_cell_cols": sorted({int(cell["col"]) for cell in target_cells if isinstance(cell.get("col"), int)}),
        "filter_cell_rows": sorted({int(cell["row"]) for cell in filter_cells if isinstance(cell.get("row"), int)}),
        "filter_cell_cols": sorted({int(cell["col"]) for cell in filter_cells if isinstance(cell.get("col"), int)}),
        "target_numeric_values": target_numeric_values,
        "filter_numeric_values": filter_numeric_values,
    }


def _candidate_operation_compatible(candidate: Optional[ExecutorResult], operation_role: str) -> bool:
    if candidate is None:
        return False
    op = candidate.operation.value
    if operation_role in {"sum", "average", "count", "diff", "difference", "ratio", "proportion"}:
        return op in {"arithmetic", "lookup_aggregate"}
    if operation_role in {"argmax", "argmin", "compare"}:
        return op in {"compare", "arithmetic", "lookup_cell"}
    if operation_role in {"lookup", "none", "auto", ""}:
        return op in {"lookup_cell", "lookup_aggregate"}
    return True


def _unresolved_scope_constraint_reasons(question: str) -> List[str]:
    q = " ".join((question or "").lower().split())
    if not q:
        return []
    checks = [
        ("rank_scope", r"\b(?:top|bottom)\s+\d+\b|\b(?:highest|lowest|largest|smallest|tallest|shortest|most|least)\b"),
        ("range_scope", r"\bbetween\b.+\band\b|\bfrom\b.+\bto\b|\b(?:since|until|before|after)\b"),
        ("threshold_scope", r"\b(?:greater|less|more|fewer)\s+than\b|\b(?:at\s+least|at\s+most|no\s+more\s+than|no\s+less\s+than|under|over|above|below)\b|[<>]=?"),
        ("condition_scope", r"\b(?:where|whose|that\s+(?:had|have|has|were|was|are|is)|with)\b"),
    ]
    return [name for name, pattern in checks if re.search(pattern, q)]


def _commit_dataset_allowed(dataset: str, scope: str) -> bool:
    dataset = (dataset or "").lower()
    scope = (scope or "tablebench").lower()
    if scope == "all":
        return dataset in {"tablebench", "hitab", "aitqa"}
    if scope in {"tablebench_hitab", "hitab_tablebench"}:
        return dataset in {"tablebench", "hitab"}
    if scope == "hitab":
        return dataset == "hitab"
    return dataset == "tablebench"


def _scope_signature(values: Sequence[Any]) -> Tuple[str, ...]:
    return tuple(
        " ".join(str(value).strip().lower().split())
        for value in (values or [])
        if str(value).strip()
    )


def _target_axis_scope_diagnostics(
    result: Dict[str, Any],
    operation_role: str,
) -> Dict[str, Any]:
    cells = list(result.get("operation_support_target_cells", []) or [])
    col_signatures = {
        _scope_signature(cell.get("col_headers", []))
        for cell in cells
        if isinstance(cell, dict)
    }
    row_signatures = {
        _scope_signature(cell.get("row_headers", []))
        for cell in cells
        if isinstance(cell, dict)
    }
    col_signatures.discard(())
    row_signatures.discard(())
    column_scope_consistent = len(col_signatures) == 1
    row_scope_consistent = len(row_signatures) == 1
    axis_scope_consistent = column_scope_consistent or (
        operation_role == "sum" and row_scope_consistent
    )
    diagnostics = {
        "target_column_scope_signature_count": len(col_signatures),
        "target_row_scope_signature_count": len(row_signatures),
        "target_column_scope_consistent": column_scope_consistent,
        "target_row_scope_consistent": row_scope_consistent,
        "target_axis_scope_consistent": axis_scope_consistent,
        "target_axis_scope_type": (
            "column"
            if column_scope_consistent
            else "row"
            if operation_role == "sum" and row_scope_consistent
            else "mixed"
        ),
    }
    result.update({
        f"operation_support_commit_{key}": value
        for key, value in diagnostics.items()
    })
    return diagnostics


def _tokenize_structural_text(text: Any) -> List[str]:
    tokens = re.findall(r"[a-z0-9]+(?:_[a-z0-9]+)?", str(text or "").lower())
    return [tok for tok in tokens if len(tok) > 1 or "_" in tok or any(ch.isdigit() for ch in tok)]


def _normalise_structural_phrase(text: Any) -> str:
    return " ".join(_tokenize_structural_text(text))


_TABLE_ENTITY_ANCHOR_STOP_PHRASES = {
    "all",
    "overall",
    "total",
    "subtotal",
    "grand total",
    "sum",
    "average",
    "mean",
    "number",
    "numbers",
    "value",
    "values",
    "amount",
    "count",
    "rate",
    "percent",
    "percentage",
}


def _structural_phrase_in_question(phrase: str, question_norm: str) -> bool:
    if not phrase or not question_norm:
        return False
    return f" {phrase} " in f" {question_norm} "


def _table_grounded_query_entity_anchors(question: str, table_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    question_norm = _normalise_structural_phrase(question)
    texts = table_json.get("texts", []) if isinstance(table_json, dict) else []
    if not question_norm or not isinstance(texts, list):
        return []
    top_rows = int(table_json.get("top_header_rows_num", 1) or 1)
    left_cols = int(table_json.get("left_header_columns_num", 1) or 1)
    anchors_by_norm: Dict[str, Dict[str, Any]] = {}
    for r, row in enumerate(texts):
        if not isinstance(row, list):
            continue
        for c, value in enumerate(row):
            raw = " ".join(str(value or "").strip().split())
            norm = _normalise_structural_phrase(raw)
            if not norm or norm in _TABLE_ENTITY_ANCHOR_STOP_PHRASES:
                continue
            if _parse_number_loose(raw) is not None:
                continue
            source = ""
            if r >= top_rows and c < left_cols:
                source = "row_header"
            elif r >= top_rows and c >= left_cols:
                source = "entity_cell"
            if not source or not _structural_phrase_in_question(norm, question_norm):
                continue
            entry = anchors_by_norm.setdefault(
                norm,
                {
                    "text": raw,
                    "norm": norm,
                    "sources": [],
                    "rows": [],
                    "cols": [],
                },
            )
            if source not in entry["sources"]:
                entry["sources"].append(source)
            if r not in entry["rows"]:
                entry["rows"].append(r)
            if c not in entry["cols"]:
                entry["cols"].append(c)
    anchors = list(anchors_by_norm.values())
    anchors.sort(key=lambda item: (-len(str(item.get("norm", "")).split()), str(item.get("norm", ""))))
    for anchor in anchors:
        anchor["rows"] = sorted(anchor.get("rows", []))[:24]
        anchor["cols"] = sorted(anchor.get("cols", []))[:24]
    return anchors[:8]


def _table_grounded_query_literal_anchors(question: str, table_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    question_norm = _normalise_structural_phrase(question)
    texts = table_json.get("texts", []) if isinstance(table_json, dict) else []
    if not question_norm or not isinstance(texts, list):
        return []
    top_rows = int(table_json.get("top_header_rows_num", 1) or 1)
    left_cols = int(table_json.get("left_header_columns_num", 1) or 1)
    anchors_by_norm: Dict[str, Dict[str, Any]] = {}
    for r, row in enumerate(texts):
        if not isinstance(row, list):
            continue
        for c, value in enumerate(row):
            in_row_header = r >= top_rows and c < left_cols
            in_col_header = r < top_rows and c >= left_cols
            if not (in_row_header or in_col_header):
                continue
            raw = " ".join(str(value or "").strip().split())
            norm = _normalise_structural_phrase(raw)
            if not norm or not any(ch.isdigit() for ch in norm):
                continue
            if not _structural_phrase_in_question(norm, question_norm):
                continue
            source = "row_header_literal" if in_row_header else "column_header_literal"
            entry = anchors_by_norm.setdefault(
                norm,
                {"text": raw, "norm": norm, "sources": [], "rows": [], "cols": []},
            )
            if source not in entry["sources"]:
                entry["sources"].append(source)
            if r not in entry["rows"]:
                entry["rows"].append(r)
            if c not in entry["cols"]:
                entry["cols"].append(c)
    anchors = list(anchors_by_norm.values())
    anchors.sort(key=lambda item: (-len(str(item.get("norm", "")).split()), str(item.get("norm", ""))))
    for anchor in anchors:
        anchor["rows"] = sorted(anchor.get("rows", []))[:24]
        anchor["cols"] = sorted(anchor.get("cols", []))[:24]
    return anchors[:8]


def _cell_axis_values(cells: Sequence[Dict[str, Any]], axis: str) -> set:
    values = set()
    for cell in cells or []:
        if not isinstance(cell, dict):
            continue
        value = cell.get(axis)
        if isinstance(value, int):
            values.add(value)
    return values


def _support_structural_text(cells: Sequence[Dict[str, Any]]) -> str:
    fields: List[str] = []
    for cell in cells or []:
        if not isinstance(cell, dict):
            continue
        fields.append(str(cell.get("value", "") or ""))
        fields.extend(str(x) for x in cell.get("row_headers", []) or [])
        fields.extend(str(x) for x in cell.get("col_headers", []) or [])
    return _normalise_structural_phrase(" ".join(fields))


def _build_query_table_entity_filter_certificate(result: Dict[str, Any]) -> Dict[str, Any]:
    anchors = [
        anchor for anchor in (result.get("query_table_entity_anchors") or [])
        if isinstance(anchor, dict) and anchor.get("norm")
    ]
    target_cells = list(result.get("operation_support_target_cells", []) or [])
    filter_cells = list(result.get("operation_support_filter_cells", []) or [])
    target_rows = {
        int(x) for x in (result.get("operation_support_target_cell_rows") or [])
        if isinstance(x, int)
    } or _cell_axis_values(target_cells, "row")
    target_cols = {
        int(x) for x in (result.get("operation_support_target_cell_cols") or [])
        if isinstance(x, int)
    } or _cell_axis_values(target_cells, "col")
    filter_rows = {
        int(x) for x in (result.get("operation_support_filter_cell_rows") or [])
        if isinstance(x, int)
    } or _cell_axis_values(filter_cells, "row")
    filter_cols = {
        int(x) for x in (result.get("operation_support_filter_cell_cols") or [])
        if isinstance(x, int)
    } or _cell_axis_values(filter_cells, "col")
    support_text = _support_structural_text(target_cells + filter_cells)

    entity_rows = set()
    entity_cols = set()
    covered: List[str] = []
    per_anchor: List[Dict[str, Any]] = []
    for anchor in anchors:
        rows = {int(x) for x in anchor.get("rows", []) if isinstance(x, int)}
        cols = {int(x) for x in anchor.get("cols", []) if isinstance(x, int)}
        norm = str(anchor.get("norm") or "")
        entity_rows.update(rows)
        entity_cols.update(cols)
        row_hit = bool(rows & (target_rows | filter_rows))
        col_hit = bool(cols & (target_cols | filter_cols))
        text_hit = _structural_phrase_in_question(norm, support_text)
        if row_hit or col_hit or text_hit:
            covered.append(norm)
        per_anchor.append({
            "norm": norm,
            "rows": sorted(rows)[:12],
            "cols": sorted(cols)[:12],
            "row_hit": row_hit,
            "col_hit": col_hit,
            "text_hit": text_hit,
        })

    all_anchors_covered = len(set(covered)) == len({str(a.get("norm") or "") for a in anchors})
    row_partition_bound = bool(target_rows and entity_rows and target_rows <= entity_rows)
    col_partition_bound = bool(target_cols and entity_cols and target_cols <= entity_cols)
    filter_partition_bound = bool(filter_cells) and all_anchors_covered
    bound = (not anchors) or (
        all_anchors_covered
        and (row_partition_bound or col_partition_bound or filter_partition_bound)
    )
    reject_reasons: List[str] = []
    if anchors and not all_anchors_covered:
        reject_reasons.append("table_entity_anchor_not_in_support")
    if anchors and all_anchors_covered and not (row_partition_bound or col_partition_bound or filter_partition_bound):
        reject_reasons.append("table_entity_partition_unbound")
    cert = {
        "version": "E65.4_query_table_entity_filter_binding",
        "anchors": anchors,
        "anchor_count": len(anchors),
        "covered_anchor_norms": sorted(set(covered)),
        "all_anchors_covered": all_anchors_covered,
        "entity_rows": sorted(entity_rows)[:32],
        "entity_cols": sorted(entity_cols)[:32],
        "target_rows": sorted(target_rows)[:32],
        "target_cols": sorted(target_cols)[:32],
        "filter_rows": sorted(filter_rows)[:32],
        "filter_cols": sorted(filter_cols)[:32],
        "row_partition_bound": row_partition_bound,
        "col_partition_bound": col_partition_bound,
        "filter_partition_bound": filter_partition_bound,
        "bound": bound,
        "reject_reasons": reject_reasons,
        "per_anchor": per_anchor[:8],
    }
    result["query_table_entity_filter_certificate"] = cert
    return cert


def _build_query_table_literal_filter_certificate(result: Dict[str, Any]) -> Dict[str, Any]:
    anchors = [
        anchor for anchor in (result.get("query_table_literal_anchors") or [])
        if isinstance(anchor, dict) and anchor.get("norm")
    ]
    target_cells = list(result.get("operation_support_target_cells", []) or [])
    filter_cells = list(result.get("operation_support_filter_cells", []) or [])
    target_rows = {
        int(x) for x in (result.get("operation_support_target_cell_rows") or [])
        if isinstance(x, int)
    } or _cell_axis_values(target_cells, "row")
    target_cols = {
        int(x) for x in (result.get("operation_support_target_cell_cols") or [])
        if isinstance(x, int)
    } or _cell_axis_values(target_cells, "col")
    filter_rows = {
        int(x) for x in (result.get("operation_support_filter_cell_rows") or [])
        if isinstance(x, int)
    } or _cell_axis_values(filter_cells, "row")
    filter_cols = {
        int(x) for x in (result.get("operation_support_filter_cell_cols") or [])
        if isinstance(x, int)
    } or _cell_axis_values(filter_cells, "col")
    support_text = _support_structural_text(target_cells + filter_cells)

    row_anchor_rows = set()
    col_anchor_cols = set()
    covered: List[str] = []
    per_anchor: List[Dict[str, Any]] = []
    for anchor in anchors:
        rows = {int(x) for x in anchor.get("rows", []) if isinstance(x, int)}
        cols = {int(x) for x in anchor.get("cols", []) if isinstance(x, int)}
        norm = str(anchor.get("norm") or "")
        sources = set(str(x) for x in anchor.get("sources", []) or [])
        if "row_header_literal" in sources:
            row_anchor_rows.update(rows)
        if "column_header_literal" in sources:
            col_anchor_cols.update(cols)
        row_hit = bool(rows & (target_rows | filter_rows))
        col_hit = bool(cols & (target_cols | filter_cols))
        text_hit = _structural_phrase_in_question(norm, support_text)
        if row_hit or col_hit or text_hit:
            covered.append(norm)
        per_anchor.append({
            "norm": norm,
            "sources": sorted(sources),
            "rows": sorted(rows)[:12],
            "cols": sorted(cols)[:12],
            "row_hit": row_hit,
            "col_hit": col_hit,
            "text_hit": text_hit,
        })

    all_anchors_covered = len(set(covered)) == len({str(a.get("norm") or "") for a in anchors})
    row_partition_bound = (not row_anchor_rows) or bool(target_rows and target_rows <= row_anchor_rows) or bool(filter_rows and row_anchor_rows <= filter_rows)
    col_partition_bound = (not col_anchor_cols) or bool(target_cols and target_cols <= col_anchor_cols) or bool(filter_cols and col_anchor_cols <= filter_cols)
    bound = (not anchors) or (all_anchors_covered and row_partition_bound and col_partition_bound)
    reject_reasons: List[str] = []
    if anchors and not all_anchors_covered:
        reject_reasons.append("table_literal_anchor_not_in_support")
    if anchors and all_anchors_covered and not row_partition_bound:
        reject_reasons.append("table_literal_row_partition_unbound")
    if anchors and all_anchors_covered and not col_partition_bound:
        reject_reasons.append("table_literal_column_partition_unbound")
    cert = {
        "version": "E65.4_query_table_literal_filter_binding",
        "anchors": anchors,
        "anchor_count": len(anchors),
        "covered_anchor_norms": sorted(set(covered)),
        "all_anchors_covered": all_anchors_covered,
        "row_anchor_rows": sorted(row_anchor_rows)[:32],
        "col_anchor_cols": sorted(col_anchor_cols)[:32],
        "target_rows": sorted(target_rows)[:32],
        "target_cols": sorted(target_cols)[:32],
        "filter_rows": sorted(filter_rows)[:32],
        "filter_cols": sorted(filter_cols)[:32],
        "row_partition_bound": row_partition_bound,
        "col_partition_bound": col_partition_bound,
        "bound": bound,
        "reject_reasons": reject_reasons,
        "per_anchor": per_anchor[:8],
    }
    result["query_table_literal_filter_certificate"] = cert
    return cert


def _cell_header_phrases(cells: Sequence[Dict[str, Any]], axis: str = "column") -> List[str]:
    phrases: List[str] = []
    seen = set()
    for cell in cells or []:
        if not isinstance(cell, dict):
            continue
        headers: List[Any] = []
        if axis in {"column", "mixed"}:
            headers.extend(cell.get("col_headers", []) or [])
        if axis in {"row", "mixed"}:
            headers.extend(cell.get("row_headers", []) or [])
        for header in headers:
            phrase = _normalise_structural_phrase(header)
            if phrase and phrase not in seen:
                phrases.append(phrase)
                seen.add(phrase)
    return phrases


def _cell_header_text(cells: Sequence[Dict[str, Any]], axis: str = "column") -> str:
    fields: List[str] = []
    for cell in cells or []:
        if not isinstance(cell, dict):
            continue
        if axis in {"column", "mixed"}:
            fields.extend(str(x) for x in cell.get("col_headers", []) or [])
        if axis in {"row", "mixed"}:
            fields.extend(str(x) for x in cell.get("row_headers", []) or [])
    return " ".join(fields)


def _extract_exact_measure_mentions(question: str) -> List[str]:
    mentions = re.findall(r"`([^`]{2,80})`", question or "")
    mentions += re.findall(r"['\"]([^'\"]{2,80})['\"]", question or "")
    return [" ".join(m.strip().split()) for m in mentions if m.strip()]


def _unit_signature(text: Any) -> List[str]:
    raw = str(text or "").lower()
    units: List[str] = []
    if "%" in raw or re.search(r"\bpercent(?:age)?\b|\bper\s*cent\b", raw):
        units.append("percent")
    for group in re.findall(r"\(([^)]{1,40})\)", raw):
        for token in _tokenize_structural_text(group):
            units.append(token)
    return sorted(set(units))


def _support_numeric_values(cells: Sequence[Dict[str, Any]]) -> List[float]:
    values: List[float] = []
    for cell in cells or []:
        if not isinstance(cell, dict):
            continue
        parsed = _parse_number_loose(cell.get("value"))
        if parsed is not None:
            values.append(parsed)
    return values


def _operation_terms(question: str) -> Dict[str, bool]:
    q = " ".join(str(question or "").lower().split())
    return {
        "sum_like": bool(re.search(r"\b(total|sum)\b", q)),
        "average_like": bool(re.search(r"\b(average|mean)\b", q)),
    }


_MEASURE_AXIS_WEAK_TOKENS = {
    "all",
    "and",
    "are",
    "at",
    "average",
    "avg",
    "by",
    "count",
    "date",
    "day",
    "december",
    "ended",
    "ending",
    "for",
    "from",
    "grand",
    "in",
    "is",
    "mean",
    "month",
    "number",
    "numbers",
    "of",
    "overall",
    "per",
    "percent",
    "percentage",
    "quarter",
    "rate",
    "sum",
    "table",
    "the",
    "to",
    "total",
    "value",
    "values",
    "was",
    "were",
    "what",
    "year",
}


def _content_tokens_for_measure_binding(text: Any) -> set:
    return {
        token for token in _tokenize_structural_text(text)
        if token not in _MEASURE_AXIS_WEAK_TOKENS and not token.isdigit()
    }


def _axis_header_phrases_for_cell(cell: Dict[str, Any], axis: str) -> List[str]:
    headers: List[Any] = []
    if axis in {"column", "mixed"}:
        headers.extend(cell.get("col_headers", []) or [])
    if axis in {"row", "mixed"}:
        headers.extend(cell.get("row_headers", []) or [])
    phrases: List[str] = []
    seen = set()
    for header in headers:
        phrase = _normalise_structural_phrase(header)
        if phrase and phrase not in seen:
            phrases.append(phrase)
            seen.add(phrase)
    return phrases


def _build_measure_axis_granularity_certificate(result: Dict[str, Any]) -> Dict[str, Any]:
    expression = result.get("operation_expression_certificate") or {}
    operation_role = str(expression.get("operation_role") or result.get("operation_support_operation_role") or "")
    target_cells = list(result.get("operation_support_target_cells", []) or [])
    target_axis = str(expression.get("target_axis") or "column")
    query_tokens = _content_tokens_for_measure_binding(result.get("question"))
    target_axis_phrases = _cell_header_phrases(target_cells, target_axis)
    target_axis_matches = [
        phrase for phrase in target_axis_phrases
        if _content_tokens_for_measure_binding(phrase) & query_tokens
    ]
    orthogonal_axis = "row" if target_axis == "column" else "column" if target_axis == "row" else "mixed"
    orthogonal_matches_by_cell: List[List[str]] = []
    orthogonal_signatures = set()
    orthogonal_matched_signatures = set()
    for cell in target_cells:
        if not isinstance(cell, dict):
            continue
        phrases = _axis_header_phrases_for_cell(cell, orthogonal_axis)
        signature = tuple(phrases)
        if signature:
            orthogonal_signatures.add(signature)
        matches = [
            phrase for phrase in phrases
            if _content_tokens_for_measure_binding(phrase) & query_tokens
        ]
        if matches:
            orthogonal_matched_signatures.add(tuple(matches))
        orthogonal_matches_by_cell.append(matches)

    duplicate_orthogonal_key_unbound = (
        operation_role in {"sum", "average"}
        and len(target_cells) > 1
        and len(orthogonal_signatures) == 1
        and not target_axis_matches
    )
    orthogonal_measure_bound = (
        bool(orthogonal_matches_by_cell)
        and all(bool(matches) for matches in orthogonal_matches_by_cell)
        and len(orthogonal_matched_signatures) == 1
    )
    bound = True
    reject_reasons: List[str] = []
    if operation_role in {"sum", "average"} and len(target_cells) > 1:
        if not target_axis_matches and not orthogonal_measure_bound:
            reject_reasons.append("measure_axis_unbound")
        if duplicate_orthogonal_key_unbound:
            reject_reasons.append("duplicate_orthogonal_key_unbound")
    if reject_reasons:
        bound = False
    cert = {
        "version": "E65.4_measure_axis_granularity",
        "operation_role": operation_role,
        "target_axis": target_axis,
        "query_content_tokens": sorted(query_tokens)[:32],
        "target_axis_phrases": target_axis_phrases[:24],
        "target_axis_matched_phrases": target_axis_matches[:24],
        "orthogonal_axis": orthogonal_axis,
        "orthogonal_signature_count": len(orthogonal_signatures),
        "orthogonal_measure_bound": orthogonal_measure_bound,
        "duplicate_orthogonal_key_unbound": duplicate_orthogonal_key_unbound,
        "bound": bound,
        "reject_reasons": reject_reasons,
    }
    result["measure_axis_granularity_certificate"] = cert
    return cert


def _build_measure_fiber_certificate(result: Dict[str, Any]) -> Dict[str, Any]:
    expression = result.get("operation_expression_certificate") or {}
    operation_role = str(expression.get("operation_role") or result.get("operation_support_operation_role") or "")
    target_cells = list(result.get("operation_support_target_cells", []) or [])
    target_axis = str(expression.get("target_axis") or "column")
    query_tokens = _content_tokens_for_measure_binding(result.get("question"))
    target_axis_phrases = _cell_header_phrases(target_cells, target_axis)
    target_axis_matches = [
        phrase for phrase in target_axis_phrases
        if _content_tokens_for_measure_binding(phrase) & query_tokens
    ]
    orthogonal_axis = "row" if target_axis == "column" else "column" if target_axis == "row" else "mixed"
    orthogonal_matches_by_cell: List[List[str]] = []
    matched_signatures = set()
    for cell in target_cells:
        if not isinstance(cell, dict):
            continue
        matches = [
            phrase for phrase in _axis_header_phrases_for_cell(cell, orthogonal_axis)
            if _content_tokens_for_measure_binding(phrase) & query_tokens
        ]
        if matches:
            matched_signatures.add(tuple(matches))
        orthogonal_matches_by_cell.append(matches)

    matched_cell_count = sum(1 for matches in orthogonal_matches_by_cell if matches)
    reject_reasons: List[str] = []
    if operation_role in {"sum", "average"} and len(target_cells) > 1:
        if not target_axis_matches and matched_cell_count == 0:
            reject_reasons.append("measure_fiber_unbound")
        if operation_role == "average" and matched_cell_count:
            if matched_cell_count != len(orthogonal_matches_by_cell):
                reject_reasons.append("orthogonal_measure_fiber_partial")
            if len(matched_signatures) != 1:
                reject_reasons.append("orthogonal_measure_fiber_mixed")
    cert = {
        "version": "E67_measure_fiber",
        "operation_role": operation_role,
        "target_axis": target_axis,
        "orthogonal_axis": orthogonal_axis,
        "query_content_tokens": sorted(query_tokens)[:32],
        "target_axis_matched_phrases": target_axis_matches[:24],
        "orthogonal_matched_cell_count": matched_cell_count,
        "orthogonal_matched_signature_count": len(matched_signatures),
        "bound": not reject_reasons,
        "reject_reasons": reject_reasons,
    }
    result["measure_fiber_certificate"] = cert
    return cert


_AGGREGATE_HEADER_TOKENS = {"total", "subtotal", "overall", "all"}


def _cell_has_aggregate_header(cell: Dict[str, Any]) -> bool:
    headers = list(cell.get("row_headers", []) or []) + list(cell.get("col_headers", []) or [])
    for header in headers:
        tokens = set(_tokenize_structural_text(header))
        if "grand" in tokens and "total" in tokens:
            return True
        if tokens & _AGGREGATE_HEADER_TOKENS:
            return True
    return False


def _build_aggregate_echo_certificate(result: Dict[str, Any]) -> Dict[str, Any]:
    expression = result.get("operation_expression_certificate") or {}
    operation_role = str(expression.get("operation_role") or result.get("operation_support_operation_role") or "")
    target_cells = list(result.get("operation_support_target_cells", []) or [])
    numeric_cells: List[Dict[str, Any]] = []
    for idx, cell in enumerate(target_cells):
        if not isinstance(cell, dict):
            continue
        value = _parse_number_loose(cell.get("value"))
        if value is None:
            continue
        numeric_cells.append({
            "index": idx,
            "value": value,
            "row": cell.get("row"),
            "col": cell.get("col"),
            "aggregate_header": _cell_has_aggregate_header(cell),
            "row_headers": list(cell.get("row_headers", []) or [])[:4],
            "col_headers": list(cell.get("col_headers", []) or [])[:4],
        })

    aggregate_cells = [cell for cell in numeric_cells if cell.get("aggregate_header")]
    reject_reasons: List[str] = []
    echo_cells: List[Dict[str, Any]] = []
    if operation_role in {"sum", "average"} and len(numeric_cells) > 1 and aggregate_cells:
        reject_reasons.append("aggregate_header_mixed_with_components")
        if operation_role == "sum":
            total_value = sum(float(cell["value"]) for cell in numeric_cells)
            for cell in aggregate_cells:
                others_sum = total_value - float(cell["value"])
                if math.isclose(float(cell["value"]), others_sum, rel_tol=1e-6, abs_tol=1e-6):
                    reject_reasons.append("aggregate_value_equals_component_sum")
                    echo_cells.append({
                        "row": cell.get("row"),
                        "col": cell.get("col"),
                        "value": cell.get("value"),
                        "component_sum": others_sum,
                    })
    cert = {
        "version": "E67_aggregate_echo",
        "operation_role": operation_role,
        "target_numeric_cell_count": len(numeric_cells),
        "aggregate_cell_count": len(aggregate_cells),
        "echo_cells": echo_cells[:8],
        "free": not reject_reasons,
        "reject_reasons": sorted(set(reject_reasons)),
    }
    result["aggregate_echo_certificate"] = cert
    return cert


def _build_candidate_source_certificate(result: Dict[str, Any]) -> Dict[str, Any]:
    before = result.get("operation_support_selected_candidate_before_rerank") or {}
    after = result.get("operation_support_selected_candidate_after_rerank") or {}
    before_answer = str(before.get("denotation") or "")
    after_answer = str(after.get("denotation") or result.get("operation_support_reranked_denotation") or "")
    changed = bool(before_answer or after_answer) and (
        _canonical_answer_key(before_answer) != _canonical_answer_key(after_answer)
    )
    reject_reasons = ["candidate_changed_by_rerank"] if changed else []
    cert = {
        "version": "E67_candidate_source_stability",
        "pre_rerank_answer": before_answer,
        "post_rerank_answer": after_answer,
        "pre_rerank_operation": before.get("operation", ""),
        "post_rerank_operation": after.get("operation", result.get("operation_support_reranked_operation", "")),
        "stable": not changed,
        "reject_reasons": reject_reasons,
    }
    result["candidate_source_certificate"] = cert
    return cert


def _build_operation_expression_certificate(result: Dict[str, Any]) -> Dict[str, Any]:
    operation_role = str(result.get("operation_support_operation_role") or result.get("question_operation") or "")
    candidate_operation = str(result.get("operation_support_reranked_operation") or "")
    target_count = int(result.get("operation_support_target_cell_count") or 0)
    filter_count = int(result.get("operation_support_filter_cell_count") or 0)
    target_cells = list(result.get("operation_support_target_cells", []) or [])
    filter_cells = list(result.get("operation_support_filter_cells", []) or [])
    axis_scope = _target_axis_scope_diagnostics(result, operation_role)
    terms = _operation_terms(str(result.get("question") or ""))

    expression_kind = "unknown"
    reject_reasons: List[str] = []
    if candidate_operation == "arithmetic" and operation_role in {"sum", "average"}:
        expression_kind = (
            f"{operation_role}(filter_rows,target_measure_column)"
            if filter_count > 0
            else f"{operation_role}(target_measure_column)"
        )
    elif candidate_operation:
        expression_kind = candidate_operation

    if candidate_operation != "arithmetic":
        reject_reasons.append("candidate_not_arithmetic")
    if operation_role not in {"sum", "average"}:
        reject_reasons.append("operation_expression_not_sum_or_average")
    if not bool(result.get("operation_support_reranked_valid")):
        reject_reasons.append("candidate_invalid")
    if not axis_scope.get("target_axis_scope_consistent"):
        reject_reasons.append("target_axis_scope_ambiguous")
    if operation_role in {"sum", "average"} and axis_scope.get("target_axis_scope_type") != "column":
        reject_reasons.append("target_axis_not_column_aggregate")
    if filter_count > 0:
        reject_reasons.append("filter_predicates_unbound")
    if operation_role in {"sum", "average"} and target_count < 2:
        reject_reasons.append("operation_arity_unsatisfied")
    if operation_role == "average" and terms["sum_like"] and terms["average_like"]:
        reject_reasons.append("operation_composition_ambiguous")

    executable = (
        candidate_operation == "arithmetic"
        and operation_role in {"sum", "average"}
        and bool(result.get("operation_support_reranked_valid"))
    )
    arity_satisfied = not (
        operation_role in {"sum", "average"} and target_count < 2
    )
    certified = executable and arity_satisfied and not reject_reasons
    cert = {
        "version": "E65.4_operation_expression",
        "kind": expression_kind,
        "operation_role": operation_role,
        "candidate_operation": candidate_operation,
        "target_axis": axis_scope.get("target_axis_scope_type", "unknown"),
        "target_column_scope_consistent": bool(axis_scope.get("target_column_scope_consistent")),
        "target_row_scope_consistent": bool(axis_scope.get("target_row_scope_consistent")),
        "target_cell_count": target_count,
        "filter_cell_count": filter_count,
        "arity_satisfied": arity_satisfied,
        "executable": executable,
        "ambiguous": bool(reject_reasons),
        "certified": certified,
        "reject_reasons": reject_reasons,
        "target_header_text": _cell_header_text(target_cells, axis_scope.get("target_axis_scope_type", "column")),
        "filter_cells_preview": filter_cells[:6],
        "target_cells_preview": target_cells[:8],
    }
    result["operation_expression_certificate"] = cert
    return cert


def _build_measure_unit_binding_certificate(result: Dict[str, Any]) -> Dict[str, Any]:
    question = str(result.get("question") or "")
    expression = result.get("operation_expression_certificate") or {}
    axis = str(expression.get("target_axis") or "column")
    target_cells = list(result.get("operation_support_target_cells", []) or [])
    target_header_text = _cell_header_text(target_cells, axis)
    target_values_text = " ".join(str(cell.get("value", "")) for cell in target_cells if isinstance(cell, dict))
    exact_mentions = _extract_exact_measure_mentions(question)
    query_units = _unit_signature(question)
    target_units = _unit_signature(target_header_text + " " + target_values_text)
    query_phrase = f" {_normalise_structural_phrase(question)} "
    target_header_phrases = _cell_header_phrases(target_cells, axis)
    matched_header_phrases = [
        phrase for phrase in target_header_phrases
        if f" {phrase} " in query_phrase
    ]
    exact_missed = [
        mention for mention in exact_mentions
        if _normalise_structural_phrase(mention) not in _normalise_structural_phrase(target_header_text)
    ]
    unit_missing = [unit for unit in query_units if unit not in target_units]
    header_present = bool(target_header_phrases or target_units)
    structural_binding_present = bool(matched_header_phrases) or bool(query_units and not unit_missing)
    status = "bound"
    reject_reasons: List[str] = []
    if not header_present:
        reject_reasons.append("target_header_uninformative")
    if exact_missed:
        reject_reasons.append("exact_measure_mention_unbound")
    if unit_missing:
        reject_reasons.append("unit_signature_unbound")
    if target_header_phrases and not structural_binding_present:
        reject_reasons.append("target_measure_phrase_unbound")
    if reject_reasons:
        status = "unbound"
    cert = {
        "version": "E65.4_measure_unit_binding",
        "binding_status": status,
        "bound": status == "bound",
        "target_header_phrases": target_header_phrases[:24],
        "matched_header_phrases": matched_header_phrases[:24],
        "exact_measure_mentions": exact_mentions,
        "exact_measure_mentions_unbound": exact_missed,
        "query_unit_signature": query_units,
        "target_unit_signature": target_units,
        "unit_signature_unbound": unit_missing,
        "target_header_text": target_header_text,
        "reject_reasons": reject_reasons,
    }
    result["measure_unit_binding_certificate"] = cert
    return cert


def _build_support_necessity_diagnostics(result: Dict[str, Any]) -> Dict[str, Any]:
    expression = result.get("operation_expression_certificate") or {}
    operation_role = str(expression.get("operation_role") or result.get("operation_support_operation_role") or "")
    target_cells = list(result.get("operation_support_target_cells", []) or [])
    target_count = int(result.get("operation_support_target_cell_count") or len(target_cells))
    target_preview_truncated = target_count > len(target_cells)
    full_values = [
        float(value)
        for value in (result.get("operation_support_target_numeric_values") or [])
        if isinstance(value, (int, float))
    ]
    values = full_values or _support_numeric_values(target_cells)
    candidate = str(result.get("operation_support_reranked_denotation") or "")
    candidate_num = _parse_number_loose(candidate)
    recomputed: Optional[float] = None
    if values and operation_role == "sum":
        recomputed = sum(values)
    elif values and operation_role == "average":
        recomputed = sum(values) / len(values)
    recomputation_consistent: Optional[bool] = None
    if recomputed is not None and candidate_num is not None:
        recomputation_consistent = math.isclose(
            recomputed,
            candidate_num,
            rel_tol=1e-3,
            abs_tol=1e-2,
        )
    target_axis = str(expression.get("target_axis") or "unknown")
    local_stability_status = "passed"
    reject_reasons: List[str] = []
    if operation_role in {"sum", "average"} and not values:
        reject_reasons.append("no_numeric_support_values")
    if operation_role in {"sum", "average"} and candidate_num is None:
        reject_reasons.append("candidate_not_numeric")
    if (
        operation_role in {"sum", "average"}
        and target_count > 0
        and len(values) != target_count
    ):
        reject_reasons.append("target_numeric_values_incomplete")
    if operation_role in {"sum", "average"} and target_preview_truncated and not full_values:
        reject_reasons.append("support_preview_truncated")
    if operation_role in {"sum", "average"} and recomputed is None:
        reject_reasons.append("denotation_not_recomputable")
    if recomputed is not None and recomputation_consistent is False:
        reject_reasons.append("denotation_not_recomputed_from_support")
    if target_axis == "mixed":
        reject_reasons.append("mixed_axis_support")
    if reject_reasons:
        local_stability_status = "failed"
    cert = {
        "version": "E65.4_support_necessity",
        "local_stability_status": local_stability_status,
        "passed": local_stability_status == "passed",
        "target_numeric_value_count": len(values),
        "target_cell_count": target_count,
        "target_preview_truncated": target_preview_truncated,
        "full_numeric_values_available": bool(full_values),
        "candidate_numeric_value": candidate_num,
        "recomputed_denotation": recomputed,
        "recomputation_consistent": recomputation_consistent,
        "remove_target_denotation_changed": True if values else None,
        "swap_sibling_measure_changed": None,
        "drop_filter_predicate_changed": None,
        "reject_reasons": reject_reasons,
    }
    result["support_necessity_diagnostics"] = cert
    return cert


def _attach_e65_4_certificates(result: Dict[str, Any]) -> None:
    _build_operation_expression_certificate(result)
    _build_measure_unit_binding_certificate(result)
    _build_measure_axis_granularity_certificate(result)
    _build_support_necessity_diagnostics(result)
    _build_answer_projection_certificate(result)
    _build_query_table_entity_filter_certificate(result)
    _build_query_table_literal_filter_certificate(result)
    _build_measure_fiber_certificate(result)
    _build_aggregate_echo_certificate(result)
    _build_candidate_source_certificate(result)


def _commit_gate_decision(
    result: Dict[str, Any],
    dataset_scope: str = "tablebench",
    operation_commit_version: str = "E67",
) -> Tuple[bool, List[str]]:
    version = _normalise_operation_commit_version(operation_commit_version)
    if version in {"E65.4", "E67"}:
        return _commit_gate_decision_e65_4(result, dataset_scope=dataset_scope)
    return _commit_gate_decision_e65_3(result, dataset_scope=dataset_scope)


def _commit_gate_decision_e65_4(result: Dict[str, Any], dataset_scope: str = "tablebench") -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    version = _normalise_operation_commit_version(result.get("operation_commit_version") or "E67")
    dataset = str(result.get("dataset") or "").lower()
    answer_role = str(result.get("operation_support_answer_role") or "")
    answer = str(result.get("operation_support_reranked_denotation") or "").strip()
    expression = result.get("operation_expression_certificate") or _build_operation_expression_certificate(result)
    measure = result.get("measure_unit_binding_certificate") or _build_measure_unit_binding_certificate(result)
    necessity = result.get("support_necessity_diagnostics") or _build_support_necessity_diagnostics(result)
    projection = result.get("answer_projection_certificate") or _build_answer_projection_certificate(result)
    entity_filter = result.get("query_table_entity_filter_certificate") or _build_query_table_entity_filter_certificate(result)
    literal_filter = result.get("query_table_literal_filter_certificate") or _build_query_table_literal_filter_certificate(result)
    granularity = result.get("measure_axis_granularity_certificate") or _build_measure_axis_granularity_certificate(result)
    measure_fiber = result.get("measure_fiber_certificate") or _build_measure_fiber_certificate(result)
    aggregate_echo = result.get("aggregate_echo_certificate") or _build_aggregate_echo_certificate(result)
    candidate_source = result.get("candidate_source_certificate") or _build_candidate_source_certificate(result)
    filter_count = int(result.get("operation_support_filter_cell_count") or 0)
    scope_reasons = _unresolved_scope_constraint_reasons(str(result.get("question") or ""))
    anchors: List[str] = []
    matched_anchors: List[str] = []
    anchor_covered = bool(entity_filter.get("bound"))
    result["operation_support_commit_dataset_scope"] = dataset_scope
    result["operation_support_commit_surface_risks_enforced"] = False
    result["operation_support_commit_structural_certificate_enforced"] = True
    result["operation_support_commit_surface_named_entity_anchor_used"] = False
    result["heuristic_surface_used_for_commit"] = False
    result["commit_decision_is_boolean_conjunction"] = True
    result["operation_support_commit_surface_heuristic_mode"] = _normalise_surface_heuristic_mode(
        result.get("operation_support_surface_heuristic_mode") or "diagnostic"
    )
    result["operation_support_commit_role_source_is_surface_heuristic"] = False
    result["operation_support_commit_role_structural_supported"] = True
    result["operation_support_commit_scope_constraint_reasons"] = list(
        scope_reasons if filter_count == 0 else []
    )
    result["operation_support_commit_named_entity_anchors"] = anchors
    result["operation_support_commit_matched_named_entity_anchors"] = matched_anchors
    result["operation_support_commit_entity_anchor_covered"] = bool(anchor_covered)
    result["operation_support_commit_table_entity_anchors"] = entity_filter.get("anchors", [])
    result["operation_support_commit_table_entity_filter_bound"] = bool(entity_filter.get("bound"))
    result["operation_support_commit_table_literal_anchors"] = literal_filter.get("anchors", [])
    result["operation_support_commit_table_literal_filter_bound"] = bool(literal_filter.get("bound"))
    result["operation_support_commit_measure_axis_granularity_bound"] = bool(granularity.get("bound"))
    result["operation_support_commit_measure_fiber_bound"] = bool(measure_fiber.get("bound"))
    result["operation_support_commit_aggregate_echo_free"] = bool(aggregate_echo.get("free"))
    result["operation_support_commit_candidate_source_stable"] = bool(candidate_source.get("stable"))
    result["operation_support_commit_surface_risk_reasons"] = _stable_unique_strings(
        (expression.get("reject_reasons") or [])
        + (measure.get("reject_reasons") or [])
        + [f"measure_axis:{reason}" for reason in granularity.get("reject_reasons", []) or []]
        + [f"measure_fiber:{reason}" for reason in measure_fiber.get("reject_reasons", []) or []]
        + [f"aggregate_echo:{reason}" for reason in aggregate_echo.get("reject_reasons", []) or []]
        + [f"candidate_source:{reason}" for reason in candidate_source.get("reject_reasons", []) or []]
        + (necessity.get("reject_reasons") or [])
        + (projection.get("reject_reasons") or [])
        + [f"entity_filter_binding:{reason}" for reason in entity_filter.get("reject_reasons", []) or []]
        + [f"literal_filter_binding:{reason}" for reason in literal_filter.get("reject_reasons", []) or []]
        + (scope_reasons if filter_count == 0 else [])
    )

    if not _commit_dataset_allowed(dataset, dataset_scope):
        reasons.append("dataset_not_in_commit_scope")
    if answer_role != "numeric":
        reasons.append("answer_role_not_numeric")
    if not bool(result.get("operation_support_reranked_valid")):
        reasons.append("candidate_invalid")
    if not bool(result.get("operation_support_reranked_role_compatible")):
        reasons.append("role_incompatible")
    if not bool(result.get("operation_support_reranked_operation_compatible")):
        reasons.append("operation_incompatible")
    if bool(result.get("operation_support_candidate_selection_ambiguous")):
        reasons.append("candidate_partial_order_ambiguous")
    if not expression.get("certified"):
        reasons.extend(f"operation_expression:{reason}" for reason in expression.get("reject_reasons", []) or ["not_certified"])
    if not measure.get("bound"):
        reasons.extend(f"measure_unit:{reason}" for reason in measure.get("reject_reasons", []) or ["unbound"])
    if not granularity.get("bound"):
        reasons.extend(f"measure_axis:{reason}" for reason in granularity.get("reject_reasons", []) or ["unbound"])
    if version == "E67" and not measure_fiber.get("bound"):
        reasons.extend(f"measure_fiber:{reason}" for reason in measure_fiber.get("reject_reasons", []) or ["unbound"])
    if version == "E67" and not aggregate_echo.get("free"):
        reasons.extend(f"aggregate_echo:{reason}" for reason in aggregate_echo.get("reject_reasons", []) or ["not_free"])
    if version == "E67" and not candidate_source.get("stable"):
        reasons.extend(f"candidate_source:{reason}" for reason in candidate_source.get("reject_reasons", []) or ["unstable"])
    if not necessity.get("passed"):
        reasons.extend(f"support_necessity:{reason}" for reason in necessity.get("reject_reasons", []) or ["failed"])
    if not projection.get("certified"):
        reasons.extend(f"answer_projection:{reason}" for reason in projection.get("reject_reasons", []) or ["not_certified"])
    if not entity_filter.get("bound"):
        reasons.extend(
            f"entity_filter_binding:{reason}"
            for reason in entity_filter.get("reject_reasons", []) or ["unbound"]
        )
    if not literal_filter.get("bound"):
        reasons.extend(
            f"literal_filter_binding:{reason}"
            for reason in literal_filter.get("reject_reasons", []) or ["unbound"]
        )
    if filter_count == 0 and scope_reasons:
        reasons.extend(f"query_scope:{reason}" for reason in scope_reasons)
    if not answer:
        reasons.append("empty_reranked_answer")
    return not reasons, sorted(set(reasons))


def _commit_gate_decision_e65_3(result: Dict[str, Any], dataset_scope: str = "tablebench") -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    dataset = str(result.get("dataset") or "").lower()
    subtype = str(result.get("dataset_question_subtype") or result.get("dataset_question_type") or "")
    answer_role = str(result.get("operation_support_answer_role") or "")
    role_source = str(result.get("operation_support_role_source") or "unknown")
    role_source_is_surface = bool(result.get("operation_support_role_source_is_surface_heuristic"))
    operation_role = str(result.get("operation_support_operation_role") or result.get("question_operation") or "")
    surface_mode = _normalise_surface_heuristic_mode(
        result.get("operation_support_surface_heuristic_mode") or "diagnostic"
    )
    reranked_op = str(result.get("operation_support_reranked_operation") or "")
    target_count = int(result.get("operation_support_target_cell_count") or 0)
    filter_count = int(result.get("operation_support_filter_cell_count") or 0)
    role_structural_supported = bool(
        role_source.startswith("operation:")
        or role_source.startswith("coarse:")
        or (
            answer_role == "numeric"
            and reranked_op in {"arithmetic", "lookup_aggregate"}
            and target_count > 0
        )
        or (
            answer_role in {"entity", "entity_numeric_label"}
            and reranked_op in {"compare", "lookup_cell"}
            and target_count > 0
        )
    )
    answer = str(result.get("operation_support_reranked_denotation") or "").strip()
    scope_reasons = _unresolved_scope_constraint_reasons(str(result.get("question") or ""))
    result["operation_support_commit_scope_constraint_reasons"] = scope_reasons
    result["operation_support_commit_role_source_is_surface_heuristic"] = role_source_is_surface
    result["operation_support_commit_role_structural_supported"] = role_structural_supported
    result["operation_support_commit_surface_heuristic_mode"] = surface_mode
    anchor_covered, anchors, matched_anchors = _support_named_entity_anchor_covered(
        str(result.get("question") or ""),
        result.get("operation_support_target_cells", []) or [],
    )
    result["operation_support_commit_named_entity_anchors"] = anchors
    result["operation_support_commit_matched_named_entity_anchors"] = matched_anchors
    result["operation_support_commit_entity_anchor_covered"] = bool(anchor_covered)
    surface_risk_reasons: List[str] = []
    if filter_count == 0 and scope_reasons:
        surface_risk_reasons.append("unresolved_scope_constraint")
    if not anchor_covered:
        surface_risk_reasons.append("entity_anchor_uncovered")
    result["operation_support_commit_surface_risk_reasons"] = surface_risk_reasons
    result["operation_support_commit_surface_risks_enforced"] = True
    axis_scope = _target_axis_scope_diagnostics(result, operation_role)

    result["operation_support_commit_dataset_scope"] = dataset_scope
    if not _commit_dataset_allowed(dataset, dataset_scope):
        reasons.append("dataset_not_in_commit_scope")
    if dataset == "tablebench" and subtype != "Aggregation":
        reasons.append("subtype_not_aggregation")
    if role_source_is_surface and not role_structural_supported:
        reasons.append("surface_role_without_structural_or_executor_support")
    if answer_role in {"numeric", "entity", "entity_numeric_label", "compound"} and not role_structural_supported:
        reasons.append("answer_role_not_structurally_supported")
    if answer_role != "numeric":
        reasons.append("answer_role_not_numeric")
    if operation_role not in {"sum", "average"}:
        reasons.append("operation_role_not_sum_or_average")
    if reranked_op != "arithmetic":
        reasons.append("reranked_op_not_arithmetic")
    if not result.get("operation_support_reranked_valid"):
        reasons.append("reranked_invalid")
    if not result.get("operation_support_reranked_role_compatible"):
        reasons.append("role_incompatible")
    if not result.get("operation_support_reranked_operation_compatible"):
        reasons.append("operation_incompatible")
    if target_count < 2:
        reasons.append("target_cell_count_lt_2")
    if filter_count > 0:
        reasons.append("filter_cell_count_gt_0")
    if filter_count == 0 and scope_reasons:
        reasons.append("query_scope_constraint_unresolved")
    if not anchor_covered:
        reasons.append("query_entity_anchor_uncovered")
    if operation_role in {"sum", "average"} and not axis_scope.get("target_column_scope_consistent"):
        reasons.append("arithmetic_column_scope_inconsistent")
    if not answer:
        reasons.append("empty_reranked_answer")
    return not reasons, reasons


def _operation_commit_certificate(
    result: Dict[str, Any],
    mode: str = "diagnostic",
    policy_allowed: Optional[bool] = None,
) -> Dict[str, Any]:
    dataset = str(result.get("dataset") or "").lower()
    dataset_scope = str(result.get("operation_support_commit_dataset_scope") or "tablebench")
    subtype = str(result.get("dataset_question_subtype") or result.get("dataset_question_type") or "")
    answer_role = str(result.get("operation_support_answer_role") or "")
    role_source = str(result.get("operation_support_role_source") or "unknown")
    role_source_is_surface = bool(result.get("operation_support_role_source_is_surface_heuristic"))
    role_structural_supported = bool(result.get("operation_support_commit_role_structural_supported"))
    surface_mode = _normalise_surface_heuristic_mode(
        result.get("operation_support_commit_surface_heuristic_mode")
        or result.get("operation_support_surface_heuristic_mode")
        or "diagnostic"
    )
    operation_role = str(result.get("operation_support_operation_role") or result.get("question_operation") or "")
    reranked_op = str(result.get("operation_support_reranked_operation") or "")
    target_count = int(result.get("operation_support_target_cell_count") or 0)
    filter_count = int(result.get("operation_support_filter_cell_count") or 0)
    scope_reasons = list(result.get("operation_support_commit_scope_constraint_reasons", []) or [])
    candidate = str(result.get("operation_support_commit_answer") or result.get("operation_support_reranked_denotation") or "").strip()
    axis_scope = _target_axis_scope_diagnostics(result, operation_role)
    version = _normalise_operation_commit_version(result.get("operation_commit_version") or "E67")
    conditions: Dict[str, Any] = {
        "dataset_in_commit_scope": _commit_dataset_allowed(dataset, dataset_scope),
        "answer_role_numeric": answer_role == "numeric",
        "answer_role_not_surface_only": (not role_source_is_surface) or role_structural_supported,
        "operation_sum_or_average": operation_role in {"sum", "average"},
        "candidate_operation_arithmetic": reranked_op == "arithmetic",
        "candidate_valid": bool(result.get("operation_support_reranked_valid")),
        "role_compatible": bool(result.get("operation_support_reranked_role_compatible")),
        "operation_compatible": bool(result.get("operation_support_reranked_operation_compatible")),
        "target_cell_coverage": target_count >= 2,
        "filter_cells_absent": filter_count == 0,
        "query_scope_constraints_resolved": not (filter_count == 0 and scope_reasons),
        "query_entity_anchors_covered": bool(result.get("operation_support_commit_entity_anchor_covered")),
        "non_empty_candidate": bool(candidate),
        "arithmetic_column_scope_consistent": bool(axis_scope.get("target_column_scope_consistent")),
    }
    if version in {"E65.4", "E67"}:
        expression = result.get("operation_expression_certificate") or _build_operation_expression_certificate(result)
        measure = result.get("measure_unit_binding_certificate") or _build_measure_unit_binding_certificate(result)
        necessity = result.get("support_necessity_diagnostics") or _build_support_necessity_diagnostics(result)
        projection = result.get("answer_projection_certificate") or _build_answer_projection_certificate(result)
        entity_filter = result.get("query_table_entity_filter_certificate") or _build_query_table_entity_filter_certificate(result)
        literal_filter = result.get("query_table_literal_filter_certificate") or _build_query_table_literal_filter_certificate(result)
        granularity = result.get("measure_axis_granularity_certificate") or _build_measure_axis_granularity_certificate(result)
        measure_fiber = result.get("measure_fiber_certificate") or _build_measure_fiber_certificate(result)
        aggregate_echo = result.get("aggregate_echo_certificate") or _build_aggregate_echo_certificate(result)
        candidate_source = result.get("candidate_source_certificate") or _build_candidate_source_certificate(result)
        result["operation_support_commit_table_entity_anchors"] = entity_filter.get("anchors", [])
        result["operation_support_commit_table_entity_filter_bound"] = bool(entity_filter.get("bound"))
        result["operation_support_commit_table_literal_anchors"] = literal_filter.get("anchors", [])
        result["operation_support_commit_table_literal_filter_bound"] = bool(literal_filter.get("bound"))
        result["operation_support_commit_measure_axis_granularity_bound"] = bool(granularity.get("bound"))
        result["operation_support_commit_measure_fiber_bound"] = bool(measure_fiber.get("bound"))
        result["operation_support_commit_aggregate_echo_free"] = bool(aggregate_echo.get("free"))
        result["operation_support_commit_candidate_source_stable"] = bool(candidate_source.get("stable"))
        scope_reasons = list(result.get("operation_support_commit_scope_constraint_reasons", []) or [])
        if "operation_support_commit_entity_anchor_covered" not in result:
            result["operation_support_commit_named_entity_anchors"] = []
            result["operation_support_commit_matched_named_entity_anchors"] = []
            result["operation_support_commit_entity_anchor_covered"] = bool(entity_filter.get("bound"))
        conditions = {
            "dataset_in_commit_scope": _commit_dataset_allowed(dataset, dataset_scope),
            "answer_role_numeric": answer_role == "numeric",
            "answer_projection_certified": bool(projection.get("certified")),
            "candidate_valid": bool(result.get("operation_support_reranked_valid")),
            "role_compatible": bool(result.get("operation_support_reranked_role_compatible")),
            "operation_compatible": bool(result.get("operation_support_reranked_operation_compatible")),
            "candidate_selection_unique": not bool(result.get("operation_support_candidate_selection_ambiguous")),
            "operation_expression_certified": bool(expression.get("certified")),
            "measure_unit_bound": bool(measure.get("bound")),
            "measure_axis_granularity_bound": bool(granularity.get("bound")),
            "support_necessity_passed": bool(necessity.get("passed")),
            "query_scope_constraints_resolved": not scope_reasons,
            "query_entity_anchors_covered": bool(result.get("operation_support_commit_entity_anchor_covered")),
            "query_table_entity_filter_bound": bool(entity_filter.get("bound")),
            "query_table_literal_filter_bound": bool(literal_filter.get("bound")),
            "non_empty_candidate": bool(candidate),
        }
        if version == "E67":
            conditions.update({
                "measure_fiber_bound": bool(measure_fiber.get("bound")),
                "aggregate_echo_free": bool(aggregate_echo.get("free")),
                "candidate_source_stable": bool(candidate_source.get("stable")),
            })
    elif dataset_scope == "tablebench" or dataset == "tablebench":
        conditions["dataset_tablebench"] = dataset == "tablebench"
        conditions["subtype_aggregation"] = subtype == "Aggregation"
    if policy_allowed is not None:
        conditions["semantic_commit_policy_allowed"] = bool(policy_allowed)
    hard_ok = all(bool(v) for v in conditions.values())
    reject_reasons = list(result.get("operation_support_commit_reject_reasons", []) or [])
    if policy_allowed is False and "semantic_commit_policy_blocked" not in reject_reasons:
        reject_reasons.append("semantic_commit_policy_blocked")
    return {
        "version": (
            "E67_measure_fiber_aggregate_echo_candidate_stability_certificate"
            if version == "E67"
            else "E65.4_expression_measure_support_entity_filter_certificate"
            if version == "E65.4"
            else "E65.3_scope_role_operation_axis_target_certificate"
        ),
        "mode": mode,
        "dataset_scope": dataset_scope,
        "eligible": bool(result.get("operation_support_commit_eligible")),
        "hard_conditions_satisfied": bool(hard_ok),
        "hard_conditions": conditions,
        "reject_reasons": reject_reasons,
        "candidate_answer": candidate,
        "candidate_answer_key": _canonical_answer_key(candidate),
        "candidate_operation": reranked_op,
        "answer_role": answer_role,
        "answer_role_source": role_source,
        "answer_role_source_is_surface_heuristic": role_source_is_surface,
        "answer_role_structural_supported": role_structural_supported,
        "surface_heuristic_mode": surface_mode,
        "operation_role": operation_role,
        "pre_commit_answer": result.get("final_answer_pre_operation_commit_gate", ""),
        "pre_commit_answer_source": result.get("answer_source_pre_operation_commit_gate", ""),
        "target_cell_count": target_count,
        "filter_cell_count": filter_count,
        "target_axis_scope": axis_scope,
        "target_cells_preview": list(result.get("operation_support_target_cells", []) or [])[:8],
        "filter_cells_preview": list(result.get("operation_support_filter_cells", []) or [])[:8],
        "scope_constraint_reasons": scope_reasons,
        "surface_risk_reasons": list(result.get("operation_support_commit_surface_risk_reasons", []) or []),
        "surface_risks_enforced": bool(result.get("operation_support_commit_surface_risks_enforced")),
        "structural_certificate_enforced": bool(result.get("operation_support_commit_structural_certificate_enforced")),
        "surface_named_entity_anchor_used_for_commit": bool(result.get("operation_support_commit_surface_named_entity_anchor_used")),
        "heuristic_surface_used_for_commit": bool(result.get("heuristic_surface_used_for_commit")),
        "legacy_commit_path_used": bool(result.get("legacy_commit_path_used")),
        "commit_decision_is_boolean_conjunction": bool(result.get("commit_decision_is_boolean_conjunction")),
        "operation_expression_certificate": result.get("operation_expression_certificate"),
        "measure_unit_binding_certificate": result.get("measure_unit_binding_certificate"),
        "measure_axis_granularity_certificate": result.get("measure_axis_granularity_certificate"),
        "measure_fiber_certificate": result.get("measure_fiber_certificate"),
        "aggregate_echo_certificate": result.get("aggregate_echo_certificate"),
        "candidate_source_certificate": result.get("candidate_source_certificate"),
        "support_necessity_diagnostics": result.get("support_necessity_diagnostics"),
        "answer_projection_certificate": result.get("answer_projection_certificate"),
        "query_table_entity_filter_certificate": result.get("query_table_entity_filter_certificate"),
        "query_table_literal_filter_certificate": result.get("query_table_literal_filter_certificate"),
        "named_entity_anchors": list(result.get("operation_support_commit_named_entity_anchors", []) or []),
        "matched_named_entity_anchors": list(result.get("operation_support_commit_matched_named_entity_anchors", []) or []),
        "table_entity_anchors": list(result.get("operation_support_commit_table_entity_anchors", []) or []),
        "table_literal_anchors": list(result.get("operation_support_commit_table_literal_anchors", []) or []),
        "applied": bool(result.get("operation_support_commit_applied")),
        "accepted": bool(result.get("operation_support_commit_accepted")),
        "same_answer": bool(result.get("operation_support_commit_same_answer")),
        "blocked_by_policy": bool(result.get("operation_support_commit_blocked_by_policy")),
        "blocked_empty_answer": bool(result.get("operation_support_commit_blocked_empty_answer")),
    }


def _maybe_apply_operation_commit_gate(
    result: Dict[str, Any],
    args: argparse.Namespace,
    final_answer: Any,
    answer_source: str,
    final_confidence: float,
) -> Tuple[Any, str, float]:
    mode = str(getattr(args, "operation_commit_gate_mode", "diagnostic") or "diagnostic").lower()
    result["operation_commit_version"] = _normalise_operation_commit_version(
        getattr(args, "operation_commit_version", "E67")
    )
    if result.get("operation_support_commit_gate_diagnostic_enabled"):
        result["operation_support_commit_gate_mode"] = mode
        result["operation_support_commit_certificate"] = _operation_commit_certificate(
            result,
            mode=mode,
            policy_allowed=_black_box_semantic_commit_allowed(args, result),
        )
    if mode != "conservative":
        return final_answer, answer_source, final_confidence
    if not result.get("operation_support_commit_gate_diagnostic_enabled"):
        return final_answer, answer_source, final_confidence
    if not result.get("operation_support_commit_eligible"):
        return final_answer, answer_source, final_confidence

    commit_answer = str(result.get("operation_support_commit_answer", "") or "").strip()
    if not commit_answer:
        result["operation_support_commit_blocked_empty_answer"] = True
        result["operation_support_commit_certificate"] = _operation_commit_certificate(
            result,
            mode=mode,
            policy_allowed=_black_box_semantic_commit_allowed(args, result),
        )
        return final_answer, answer_source, final_confidence
    if not _black_box_semantic_commit_allowed(args, result):
        result["operation_support_commit_blocked_by_policy"] = True
        result["black_box_semantic_commit_blocked"] = True
        result["black_box_semantic_commit_blocked_stage"] = "operation_commit_gate"
        result["operation_support_commit_certificate"] = _operation_commit_certificate(
            result,
            mode=mode,
            policy_allowed=False,
        )
        return final_answer, answer_source, final_confidence

    result["operation_support_commit_accepted"] = True
    if _canonical_answer_key(commit_answer) == _canonical_answer_key(final_answer):
        result["operation_support_commit_same_answer"] = True
        result["operation_support_commit_certificate"] = _operation_commit_certificate(
            result,
            mode=mode,
            policy_allowed=True,
        )
        return final_answer, answer_source, final_confidence

    result["final_answer_pre_operation_commit_gate"] = final_answer
    result["answer_source_pre_operation_commit_gate"] = answer_source
    result["final_confidence_pre_operation_commit_gate"] = final_confidence
    result["final_answer"] = commit_answer
    result["answer_source"] = "operation_commit_gate"
    result["operation_support_commit_applied"] = True
    result["operation_support_commit_certificate"] = _operation_commit_certificate(
        result,
        mode=mode,
        policy_allowed=True,
    )
    return commit_answer, "operation_commit_gate", final_confidence


def _select_role_aware_candidate(
    candidates: Sequence[ExecutorResult],
    answer_role: str,
    operation_role: str,
) -> Tuple[Optional[ExecutorResult], str]:
    if not candidates:
        return None, "no_candidates"

    def _features(candidate: ExecutorResult) -> Tuple[int, int, int, int]:
        role_ok = int(_candidate_role_compatible(candidate, answer_role))
        op_ok = int(_candidate_operation_compatible(candidate, operation_role))
        executable = int(bool(candidate.executor_valid and str(candidate.denotation or "").strip()))
        arithmetic_match = int(
            operation_role in {"sum", "average"}
            and candidate.operation == OperationType.ARITHMETIC
        )
        return executable, role_ok, op_ok, arithmetic_match

    feature_rows = [(candidate, _features(candidate)) for candidate in candidates]
    non_dominated: List[Tuple[ExecutorResult, Tuple[int, int, int, int]]] = []
    for candidate, feats in feature_rows:
        dominated = False
        for other, other_feats in feature_rows:
            if other is candidate:
                continue
            if all(o >= f for o, f in zip(other_feats, feats)) and any(o > f for o, f in zip(other_feats, feats)):
                dominated = True
                break
        if not dominated:
            non_dominated.append((candidate, feats))

    if len(non_dominated) == 1:
        selected = non_dominated[0][0]
        ambiguity = "partial_order_unique"
    else:
        selected = next((c for c, feats in non_dominated if feats[-1] == 1), None)
        ambiguity = "partial_order_ambiguous"
        if selected is None:
            selected = non_dominated[0][0] if non_dominated else candidates[0]
    reasons = []
    if _candidate_role_compatible(selected, answer_role):
        reasons.append("role_compatible")
    else:
        reasons.append("role_incompatible")
    if _candidate_operation_compatible(selected, operation_role):
        reasons.append("operation_compatible")
    else:
        reasons.append("operation_incompatible")
    reasons.append(ambiguity)
    return selected, ",".join(reasons)


def _attach_role_target_support_diagnostics(
    result: Dict[str, Any],
    question: str,
    coarse_type: str,
    question_operation: str,
    executor_result: Optional[ExecutorResult],
    all_exec_candidates: Optional[Sequence[ExecutorResult]],
    operation_commit_gate_diagnostics: bool = False,
    operation_commit_dataset_scope: str = "tablebench",
    surface_heuristic_mode: str = "diagnostic",
    operation_commit_version: str = "E67",
) -> None:
    role_info = _infer_answer_role_commitment(
        question,
        coarse_type,
        question_operation,
        surface_heuristic_mode=surface_heuristic_mode,
    )
    commit_version = _normalise_operation_commit_version(operation_commit_version)
    if commit_version in {"E65.4", "E67"}:
        structural_role = role_info.get("structural_answer_role") or "unknown"
        structural_source = role_info.get("structural_role_source") or "unknown"
        role_info["answer_role"] = structural_role
        role_info["role_source"] = structural_source
        role_info["role_primary_source"] = "structural_e65_4"
        role_info["role_source_is_surface_heuristic"] = False
    answer_role = role_info["answer_role"]
    operation_role = role_info["operation_role"]
    candidates = list(all_exec_candidates or [])
    selected_after, rerank_reason = _select_role_aware_candidate(
        candidates,
        answer_role,
        operation_role,
    )
    selected_before = executor_result or (candidates[0] if candidates else None)
    changed = (
        selected_after is not None
        and selected_before is not None
        and _canonical_answer_key(selected_after.denotation) != _canonical_answer_key(selected_before.denotation)
    )
    split = _split_filter_target_cells(selected_after or selected_before, answer_role)
    reason_parts = [x for x in rerank_reason.split(",") if x]
    if selected_after is not None and candidates and selected_after is not candidates[0]:
        reason_parts.append("candidate_priority_changed")
    reason_parts.append("denotation_changed" if changed else "denotation_stable")

    result["operation_support_role_target_diagnostic_enabled"] = True
    result["operation_support_answer_role"] = answer_role
    result["operation_support_operation_role"] = operation_role
    result["operation_support_role_source"] = role_info["role_source"]
    result["operation_support_role_primary_source"] = role_info.get("role_primary_source", "")
    result["operation_support_role_source_is_surface_heuristic"] = bool(
        role_info.get("role_source_is_surface_heuristic")
    )
    result["operation_support_surface_heuristic_mode"] = role_info.get("surface_heuristic_mode")
    result["operation_support_surface_answer_role"] = role_info.get("surface_answer_role")
    result["operation_support_surface_role_source"] = role_info.get("surface_role_source")
    result["operation_support_structural_answer_role"] = role_info.get("structural_answer_role")
    result["operation_support_structural_role_source"] = role_info.get("structural_role_source")
    result["operation_support_surface_structural_role_agreement"] = role_info.get(
        "surface_structural_role_agreement"
    )
    result["operation_support_legacy_expected_role"] = result.get("operation_support_expected_role", "")
    result["operation_support_expected_role"] = answer_role
    result["operation_support_selected_candidate_before_rerank"] = executor_result_summary(selected_before, max_cells=8)
    result["operation_support_selected_candidate_after_rerank"] = executor_result_summary(selected_after, max_cells=8)
    result["operation_support_reranked_denotation"] = getattr(selected_after, "denotation", "") if selected_after else ""
    result["operation_support_reranked_operation"] = (
        selected_after.operation.value if selected_after is not None else ""
    )
    result["operation_support_reranked_valid"] = bool(selected_after and selected_after.executor_valid)
    result["operation_support_reranked_role_compatible"] = _candidate_role_compatible(selected_after, answer_role)
    result["operation_support_reranked_operation_compatible"] = _candidate_operation_compatible(selected_after, operation_role)
    result["operation_support_rerank_changed"] = bool(changed)
    result["operation_support_rerank_reason"] = ",".join(reason_parts)
    result["operation_support_candidate_selection_ambiguous"] = "partial_order_ambiguous" in reason_parts
    result["operation_support_filter_cell_count"] = split["filter_cell_count"]
    result["operation_support_target_cell_count"] = split["target_cell_count"]
    result["operation_support_filter_cells"] = split["filter_cells"]
    result["operation_support_target_cells"] = split["target_cells"]
    result["operation_support_target_cell_rows"] = split.get("target_cell_rows", [])
    result["operation_support_target_cell_cols"] = split.get("target_cell_cols", [])
    result["operation_support_filter_cell_rows"] = split.get("filter_cell_rows", [])
    result["operation_support_filter_cell_cols"] = split.get("filter_cell_cols", [])
    result["operation_support_target_labels"] = split["target_labels"]
    result["operation_support_target_numeric_values"] = split.get("target_numeric_values", [])
    result["operation_support_filter_numeric_values"] = split.get("filter_numeric_values", [])
    result["operation_support_entity_surface_numeric_role"] = bool(
        _entity_surface_requested(question) and answer_role == "numeric"
    )
    result["operation_commit_version"] = commit_version
    if commit_version in {"E65.4", "E67"}:
        _attach_e65_4_certificates(result)
    if operation_commit_gate_diagnostics:
        eligible, reject_reasons = _commit_gate_decision(
            result,
            dataset_scope=operation_commit_dataset_scope,
            operation_commit_version=commit_version,
        )
        result["operation_support_commit_gate_diagnostic_enabled"] = True
        result["operation_support_commit_dataset_scope"] = operation_commit_dataset_scope
        result["operation_support_commit_eligible"] = bool(eligible)
        result["operation_support_commit_reject_reasons"] = reject_reasons
        result["operation_support_commit_answer"] = result.get("operation_support_reranked_denotation", "")
        result["operation_support_commit_certificate"] = _operation_commit_certificate(result)


def _expected_role_for_question(
    question: str,
    coarse_type: str,
    question_operation: str,
    surface_heuristic_mode: str = "diagnostic",
) -> str:
    role_info = _infer_answer_role_commitment(
        question,
        coarse_type,
        question_operation,
        surface_heuristic_mode=surface_heuristic_mode,
    )
    return str(role_info.get("answer_role") or "unknown")


def _attach_operation_support_outcome(result: Dict[str, Any], gold_answer: Any, dataset: str) -> None:
    if not result.get("operation_support_diagnostic_enabled"):
        return
    exec_answer = str(result.get("executor_answer", "") or "")
    final_answer = str(result.get("final_answer", "") or "")
    primary_key = _primary_metric_key_for_dataset(dataset)
    exec_eval = evaluate_answer_for_dataset(exec_answer, gold_answer, dataset) if exec_answer else {}
    result["operation_support_eval"] = exec_eval
    result["operation_support_official_correct"] = bool(exec_eval.get(primary_key, exec_eval.get("hitab_official_em", False)))
    result["operation_support_matches_final"] = (
        bool(exec_answer)
        and bool(final_answer)
        and _canonical_answer_key(exec_answer) == _canonical_answer_key(final_answer)
    )
    final_ok = bool(result.get(primary_key, result.get("hitab_official_em", False)))
    exec_ok = bool(result.get("operation_support_official_correct"))
    if exec_ok and final_ok:
        gap = "both_correct"
    elif exec_ok and not final_ok:
        gap = "executor_correct_final_wrong"
    elif final_ok and not exec_ok:
        gap = "final_correct_executor_wrong"
    elif result["operation_support_matches_final"]:
        gap = "both_wrong_same"
    else:
        gap = "both_wrong_different"
    result["operation_support_gap"] = gap

    if result.get("operation_support_role_target_diagnostic_enabled"):
        pre_commit_answer = str(result.get("final_answer_pre_operation_commit_gate", final_answer) or "")
        pre_commit_eval = evaluate_answer_for_dataset(pre_commit_answer, gold_answer, dataset) if pre_commit_answer else {}
        result["operation_support_commit_baseline_eval"] = pre_commit_eval
        result["operation_support_commit_baseline_official_correct"] = bool(
            pre_commit_eval.get(primary_key, pre_commit_eval.get("hitab_official_em", False))
        )
        reranked_answer = str(result.get("operation_support_reranked_denotation", "") or "")
        reranked_eval = evaluate_answer_for_dataset(reranked_answer, gold_answer, dataset) if reranked_answer else {}
        result["operation_support_reranked_eval"] = reranked_eval
        result["operation_support_reranked_official_correct"] = bool(
            reranked_eval.get(primary_key, reranked_eval.get("hitab_official_em", False))
        )
        result["operation_support_reranked_matches_final"] = (
            bool(reranked_answer)
            and bool(final_answer)
            and _canonical_answer_key(reranked_answer) == _canonical_answer_key(final_answer)
        )
        reranked_ok = bool(result.get("operation_support_reranked_official_correct"))
        if reranked_ok and final_ok:
            reranked_gap = "both_correct"
        elif reranked_ok and not final_ok:
            reranked_gap = "reranked_correct_final_wrong"
        elif final_ok and not reranked_ok:
            reranked_gap = "final_correct_reranked_wrong"
        elif result["operation_support_reranked_matches_final"]:
            reranked_gap = "both_wrong_same"
        else:
            reranked_gap = "both_wrong_different"
        result["operation_support_reranked_gap"] = reranked_gap

        if result.get("operation_support_commit_gate_diagnostic_enabled"):
            commit_answer = str(result.get("operation_support_commit_answer", "") or "")
            commit_eval = evaluate_answer_for_dataset(commit_answer, gold_answer, dataset) if commit_answer else {}
            result["operation_support_commit_eval"] = commit_eval
            result["operation_support_commit_official_correct"] = bool(
                commit_eval.get(primary_key, commit_eval.get("hitab_official_em", False))
            )
            commit_ok = bool(result.get("operation_support_commit_official_correct"))
            baseline_ok = bool(result.get("operation_support_commit_baseline_official_correct"))
            commit_matches_baseline = (
                bool(commit_answer)
                and bool(pre_commit_answer)
                and _canonical_answer_key(commit_answer) == _canonical_answer_key(pre_commit_answer)
            )
            if not result.get("operation_support_commit_eligible"):
                commit_gap = "not_eligible"
            elif commit_ok and baseline_ok:
                commit_gap = "both_correct"
            elif commit_ok and not baseline_ok:
                commit_gap = "commit_correct_final_wrong"
            elif baseline_ok and not commit_ok:
                commit_gap = "final_correct_commit_wrong"
            elif commit_matches_baseline:
                commit_gap = "both_wrong_same"
            else:
                commit_gap = "both_wrong_different"
            result["operation_support_commit_gap"] = commit_gap
            result["operation_support_commit_pre_official_correct"] = baseline_ok
            result["operation_support_commit_post_official_correct"] = commit_ok
            result["operation_support_commit_wrong_to_wrong_changed"] = (
                commit_gap == "both_wrong_different"
            )


def run_pipeline(args: argparse.Namespace) -> None:
    """执行完整管线"""
    os.makedirs(args.output_dir, exist_ok=True)
    if getattr(args, "overwrite", False):
        for name in (
            "predictions.jsonl",
            "predictions.debug.jsonl",
            "predictions_recalculated.jsonl",
            "metrics.json",
            "metrics_recalculated.json",
            "run_config.json",
            "run_metadata.json",
            "dry_run_prompts.jsonl",
        ):
            path = os.path.join(args.output_dir, name)
            if os.path.exists(path):
                os.remove(path)
    logger = setup_logger(os.path.join(args.output_dir, "run.log"))

    mode = args.mode
    args.dataset = normalize_dataset_name(getattr(args, "dataset", "hitab"))
    args.seed = int(getattr(args, "seed", 0) or 0)
    set_global_seed(args.seed)
    created_at = datetime.now(timezone.utc).isoformat()
    method = f"CSCR-{mode}"
    run_id = resolve_run_id(args, method=method, created_at=created_at)
    args.run_id = run_id
    logger.info(f"CSCR Pipeline starting in mode: {mode} | dataset={args.dataset} | run_id={run_id}")
    logger.info(f"Deterministic seed: {args.seed}")

    # 保存运行配置
    run_config = vars(args).copy()
    # 移除不可序列化的私有字段
    for k in list(run_config.keys()):
        if k.startswith("_"):
            run_config.pop(k)
    run_config["method"] = method
    run_config["created_at"] = created_at
    run_config["run_id"] = run_id
    run_config["api_chat_template_kwargs"] = _api_chat_template_kwargs(
        getattr(args, "generator_backend", ""),
        getattr(args, "api_model", None) or getattr(args, "model_path", ""),
    )
    run_config["modules"] = {
        "structure_aware_formatter": True,
        "executor": mode in ("executor_only", "full", "full_cert"),
        "graph_builder": mode in ("full", "full_cert"),
        "evidence_retriever": mode in ("full", "full_cert"),
        "certificate_calibrator": mode == "full_cert",
        "intervention_engine": mode == "full_cert",
        "logit_entropy": getattr(args, "generator_backend", "vllm") == "vllm",
        "batch_inference": getattr(args, "batch_inference", False),
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": getattr(args, "dtype", "bfloat16"),
        "operation_role_target_diagnostics": getattr(args, "operation_role_target_diagnostics", False),
        "operation_commit_gate_diagnostics": getattr(args, "operation_commit_gate_diagnostics", False),
        "operation_commit_gate_mode": getattr(args, "operation_commit_gate_mode", "diagnostic"),
        "operation_commit_dataset_scope": getattr(args, "operation_commit_dataset_scope", "tablebench"),
        "main_cert_profile": getattr(args, "main_cert_profile", False),
        "cera_repair": {
            "enabled": bool(getattr(args, "enable_cera_repair", False)),
            "stage": getattr(args, "cera_stage", "E71"),
            "round6_e71_v4": bool(getattr(args, "cera_round6_e71_v4", False)),
            "typed_planner_enabled": bool(getattr(args, "cera_enable_typed_planner", False)),
            "planner_boundary": getattr(args, "cera_planner_boundary", "proposal_blind_schema_only"),
            "planner_contract": getattr(args, "cera_planner_contract", "legacy_v1"),
            "planner_legacy_query_semantics_mode": getattr(
                args,
                "cera_planner_legacy_query_semantics_mode",
                "active",
            ),
            "planner_max_tokens": int(getattr(args, "cera_planner_max_tokens", 512)),
            "planner_temperature": float(getattr(args, "cera_planner_temperature", 0.0)),
            "shadow_only": bool(getattr(args, "cera_shadow_only", True)),
            "template_version": getattr(args, "cera_template_version", "cera_repair_v2"),
            "require_derivation_program": bool(getattr(args, "cera_require_derivation_program", True)),
            "require_counterfactual_reference": bool(getattr(args, "cera_require_counterfactual_reference", True)),
            "allow_support_only": bool(getattr(args, "cera_allow_support_only", False)),
        },
        "main_cert_profile_clean_contract": {
            "structural_certificate_gate_enabled": (
                getattr(args, "operation_commit_gate_diagnostics", False)
                and _normalise_operation_commit_version(
                    getattr(args, "operation_commit_version", "E67")
                ) in {"E65.4", "E67"}
            ),
            "legacy_commit_paths_disabled": not (
                getattr(args, "hceg_fallback", False)
                or getattr(args, "certificate_commit_boundary", False)
                or getattr(args, "self_consistency", False)
            ),
            "prompt_routing_disabled": not (
                getattr(args, "adaptive_prompt", False)
                or getattr(args, "question_type_router", False)
            ),
            "answer_normalizers_disabled": not (
                getattr(args, "online_normalizer", False)
                or getattr(args, "oracle_online_normalizer", False)
                or getattr(args, "api_format_normalizer", "auto") != "off"
            ),
            "source_risk_calibration_disabled": getattr(args, "source_risk_calibration", "auto") == "off",
            "credal_probe_disabled": not getattr(args, "credal_probe", False),
            "black_box_policy_certificate_only": getattr(args, "black_box_commit_policy", "auto") in {"off", "certified"},
            "surface_heuristic_mode": getattr(args, "surface_heuristic_mode", "diagnostic"),
        },
        "operation_commit_version": _normalise_operation_commit_version(
            getattr(args, "operation_commit_version", "E67")
        ),
        "e65_4_query_table_entity_filter_binding": _normalise_operation_commit_version(
            getattr(args, "operation_commit_version", "E67")
        ) in {"E65.4", "E67"},
        "e65_4_query_table_literal_filter_binding": _normalise_operation_commit_version(
            getattr(args, "operation_commit_version", "E67")
        ) in {"E65.4", "E67"},
        "e65_4_measure_axis_granularity": _normalise_operation_commit_version(
            getattr(args, "operation_commit_version", "E67")
        ) in {"E65.4", "E67"},
        "e67_measure_fiber": _normalise_operation_commit_version(
            getattr(args, "operation_commit_version", "E67")
        ) == "E67",
        "e67_aggregate_echo": _normalise_operation_commit_version(
            getattr(args, "operation_commit_version", "E67")
        ) == "E67",
        "e67_candidate_source_stability": _normalise_operation_commit_version(
            getattr(args, "operation_commit_version", "E67")
        ) == "E67",
        "llm_input_audit": _llm_input_audit_mode(args) != "off",
        "llm_input_audit_mode": _llm_input_audit_mode(args),
        "llm_input_audit_file": str(_llm_input_audit_path(args)),
        "generator_backend": getattr(args, "generator_backend", "vllm"),
        "api_chat_template_kwargs": dict(run_config.get("api_chat_template_kwargs", {})),
        "black_box_api_generator": getattr(args, "generator_backend", "vllm") in {"openai_chat", "gemini_chat", "vllm_chat"},
        "black_box_commit_policy": getattr(args, "black_box_commit_policy", "auto"),
        "api_format_normalizer": getattr(args, "api_format_normalizer", "auto"),
        "surface_heuristic_mode": getattr(args, "surface_heuristic_mode", "diagnostic"),
    }
    write_json(os.path.join(args.output_dir, "run_config.json"), run_config)
    write_json(
        os.path.join(args.output_dir, "run_metadata.json"),
        build_run_metadata(
            args,
            method=method,
            created_at=created_at,
            run_id=run_id,
            modules=run_config.get("modules", {}),
        ),
    )

    # --- recalculate 模式: 仅重新计算 EM ---
    if mode == "recalculate":
        logger.info("Recalculate mode: re-evaluating existing predictions")
        pred_file = args.recalculate_from or os.path.join(args.output_dir, "predictions.jsonl")
        predictions = read_jsonl(pred_file)
        logger.info(f"Loaded {len(predictions)} predictions from {pred_file}")

        # 去重
        seen_ids = set()
        deduped = []
        for pred in predictions:
            pid = pred.get("id", "")
            if pid not in seen_ids:
                seen_ids.add(pid)
                deduped.append(pred)
        if len(deduped) < len(predictions):
            logger.info(f"Deduplicated: {len(predictions)} -> {len(deduped)} predictions")
        predictions = deduped

        for pred in predictions:
            answers_dict = pred.get("answers", {})
            if isinstance(answers_dict, dict) and "final" in answers_dict:
                final = str(answers_dict["final"])
            else:
                final = pred.get("final_answer",
                        pred.get("ca2kg_answer",
                        pred.get("baseline_answer", "")))

            gold = pred.get("expected_answer",
                   pred.get("gold_answer",
                   pred.get("gold", pred.get("answer", ""))))

            aggregation = pred.get("aggregation", ["none"])

            em = evaluate_answer_multi_caliber(final, gold, aggregation)
            pred["final_answer"] = final
            pred["gold_answer"] = gold
            pred.update(em)

        out_file = os.path.join(args.output_dir, "predictions_recalculated.jsonl")
        with open(out_file, "w", encoding="utf-8") as f:
            for pred in predictions:
                f.write(json.dumps(pred, ensure_ascii=False) + "\n")

        metrics = batch_evaluate(predictions, answer_key="final_answer", gold_key="gold_answer")
        write_json(os.path.join(args.output_dir, "metrics_recalculated.json"), metrics)
        logger.info(f"Recalculated metrics saved. Summary: {json.dumps({k: v for k, v in metrics.items() if not isinstance(v, list)}, indent=2)}")
        return

    # --- 加载数据 ---
    items = [normalize_item_for_cscr(item, args.dataset) for item in read_jsonl(args.input_file)]
    if args.start_from:
        items = items[args.start_from:]
    if args.limit is not None:
        items = items[:args.limit]
    logger.info(f"Loaded {len(items)} items")
    validate_graph_required_table_source(args, items)

    # 初始化生成器
    generator = None
    if not args.dry_run:
        generator_backend = getattr(args, "generator_backend", "vllm")
        if generator_backend in {"openai_chat", "gemini_chat", "vllm_chat"}:
            api_model = getattr(args, "api_model", None) or args.model_path
            if not api_model:
                raise RuntimeError(f"--api-model or --model_path is required for {generator_backend} backend")
            if generator_backend in {"openai_chat", "vllm_chat"}:
                logger.info(
                    f"Loading OpenAI-compatible API generator: backend={generator_backend} | model={api_model} | "
                    f"base_url={args.api_base_url} | api_key_env={args.api_key_env} | "
                    f"timeout={args.api_timeout}s | max_retries={args.api_max_retries} | "
                    f"rate_limit={args.api_rate_limit_seconds}s | max_model_len={args.max_model_len}"
                )
                generator = OpenAIChatGenerator(
                    model=api_model,
                    api_base_url=args.api_base_url,
                    api_key_env=args.api_key_env,
                    timeout=args.api_timeout,
                    max_retries=args.api_max_retries,
                    rate_limit_seconds=args.api_rate_limit_seconds,
                    max_model_len=args.max_model_len,
                    cache_path=getattr(args, "api_cache_path", ""),
                    cache_mode=getattr(args, "api_cache_mode", "readwrite"),
                    backend_name=generator_backend,
                )
            else:
                logger.info(
                    f"Loading Gemini API generator: model={api_model} | "
                    f"api_key_env={args.api_key_env} | timeout={args.api_timeout}s | "
                    f"max_retries={args.api_max_retries} | rate_limit={args.api_rate_limit_seconds}s | "
                    f"max_model_len={args.max_model_len}"
                )
                generator = GeminiChatGenerator(
                    model=api_model,
                    api_key_env=args.api_key_env,
                    timeout=args.api_timeout,
                    max_retries=args.api_max_retries,
                    rate_limit_seconds=args.api_rate_limit_seconds,
                    max_model_len=args.max_model_len,
                    cache_path=getattr(args, "api_cache_path", ""),
                    cache_mode=getattr(args, "api_cache_mode", "readwrite"),
                )
            logger.info("API generator loaded. Logprobs/attention unavailable by design.")
        else:
            tp = args.tensor_parallel_size
            dtype = getattr(args, "dtype", "bfloat16")
            gpu_mem = getattr(args, "gpu_memory_utilization", 0.90)
            max_num_seqs = getattr(args, "max_num_seqs", 256)
            max_num_batched_tokens = getattr(args, "max_num_batched_tokens", None)
            swap_space = getattr(args, "swap_space", 1)
            cpu_offload_gb = getattr(args, "cpu_offload_gb", 0)
            disable_custom_all_reduce = getattr(args, "disable_custom_all_reduce", False)
            enable_prefix_caching = getattr(args, "enable_prefix_caching", True)
            enable_chunked_prefill = getattr(args, "enable_chunked_prefill", False)
            kv_cache_dtype = getattr(args, "kv_cache_dtype", "auto")
            use_fast_image_processor = getattr(args, "use_fast_image_processor", True)
            distributed_executor_backend = getattr(args, "distributed_executor_backend", "mp")
            enforce_eager = getattr(args, "enforce_eager", False)
            logger.info(
                f"Loading model: {args.model_path} | "
                f"TP={tp} | dtype={dtype} | "
                f"MAX_MODEL_LEN={args.max_model_len} | "
                f"gpu_mem_util={gpu_mem} | max_num_seqs={max_num_seqs} | "
                f"max_num_batched_tokens={max_num_batched_tokens or 'auto'} | "
                f"swap_space={swap_space}GiB/gpu | cpu_offload_gb={cpu_offload_gb}GiB/gpu | "
                f"disable_custom_all_reduce={disable_custom_all_reduce} | "
                f"enforce_eager={enforce_eager} | "
                f"enable_chunked_prefill={enable_chunked_prefill} | "
                f"kv_cache_dtype={kv_cache_dtype} | "
                f"seed={args.seed} | "
                f"use_fast_image_processor={use_fast_image_processor} | "
                f"distributed_executor_backend={distributed_executor_backend}"
            )
            generator = VLLMGeneratorWithLogprobs(
                model_path=args.model_path,
                max_model_len=args.max_model_len,
                tensor_parallel_size=tp,
                dtype=dtype,
                gpu_memory_utilization=gpu_mem,
                max_num_seqs=max_num_seqs,
                max_num_batched_tokens=max_num_batched_tokens,
                swap_space=swap_space,
                cpu_offload_gb=cpu_offload_gb,
                disable_custom_all_reduce=disable_custom_all_reduce,
                enforce_eager=enforce_eager,
                enable_prefix_caching=enable_prefix_caching,
                enable_chunked_prefill=enable_chunked_prefill,
                kv_cache_dtype=kv_cache_dtype,
                use_fast_image_processor=use_fast_image_processor,
                distributed_executor_backend=distributed_executor_backend,
                seed=args.seed,
            )
            logger.info(f"Model loaded. Detected EOS tokens: {generator.eos}")

    write_json(
        os.path.join(args.output_dir, "run_metadata.json"),
        build_run_metadata(
            args,
            method=method,
            created_at=created_at,
            run_id=run_id,
            modules=run_config.get("modules", {}),
            generator=generator,
        ),
    )

    # --- Dry run ---
    if args.dry_run:
        table_cache: Dict[str, Dict[str, Any]] = {}
        dry_rows = []
        for item in items[:args.batch_size]:
            table = load_table_json(item, args.table_dir, table_cache, dataset=args.dataset)
            question = item.get("question", "")
            prompt = build_structure_aware_prompt(table, question)
            dry_rows.append({
                "id": item.get("id"),
                "question": question,
                "prompt_length": len(prompt),
                "prompt_preview": prompt[:500],
            })
        append_jsonl(os.path.join(args.output_dir, "dry_run_prompts.jsonl"), dry_rows)
        logger.info(f"Dry run: wrote {len(dry_rows)} prompt previews")
        return

    # --- 处理 ---
    table_cache: Dict[str, Dict[str, Any]] = {}
    predictions: List[Dict[str, Any]] = []
    pred_file = os.path.join(args.output_dir, "predictions.jsonl")
    debug_pred_file = os.path.join(args.output_dir, "predictions.debug.jsonl")
    prediction_record_layout = getattr(args, "prediction_record_layout", "research")
    write_debug_predictions = bool(getattr(args, "write_debug_predictions", True))

    # Resume 支持：优先读取 debug sidecar，以保证中断续跑后的聚合指标仍能使用完整诊断字段。
    processed_ids = set()
    if args.resume and (os.path.exists(debug_pred_file) or os.path.exists(pred_file)):
        existing_path = debug_pred_file if os.path.exists(debug_pred_file) else pred_file
        existing = read_jsonl(existing_path)
        processed_ids = {p.get("id") for p in existing}
        predictions = existing
        logger.info(f"Resume: {len(processed_ids)} already processed from {existing_path}")

    total_start = time.time()
    batch_size = args.batch_size
    top_k_logprobs = args.top_k_logprobs if hasattr(args, "top_k_logprobs") else 5
    use_batch_inference = getattr(args, "batch_inference", False)

    # v7.0a: Success Predictor 加载
    if mode == "full_cert":
        sp_path = getattr(args, "success_predictor_model", None)
        if sp_path:
            if not os.path.exists(sp_path):
                raise FileNotFoundError(f"Success Predictor checkpoint not found: {sp_path}")
            args._success_predictor = load_predictor(sp_path)
            logger.info(f"v7.0a Success Predictor loaded from {sp_path}")
            logger.warning(
                "v8.5 safety: Success Predictor is treated as a conservative diagnostic/weak gate. "
                "If this checkpoint was trained on 7B predictions, do not interpret it as calibrated "
                "for 32B or API black-box models; SP overwrite now additionally requires path_verified=True."
            )
        else:
            args._success_predictor = None

    # v6.0: Conformal Abstention 初始化
    if mode == "full_cert":
        conformal_path = getattr(args, "conformal_calibrate_from", None)
        if conformal_path:
            conformal_alpha = getattr(args, "conformal_alpha", 0.05)
            abstainer = ConformalAbstainer(alpha=conformal_alpha)
            tau = abstainer.calibrate_from_predictions(conformal_path)
            logger.info(f"v6.1 Conformal Abstention: calibrated from {conformal_path}")
            stats = abstainer.get_stats()
            logger.info(f"  threshold τ* = {tau:.4f}")
            logger.info(f"  alpha = {stats.get('alpha')}, calibration_size = {stats.get('calibration_size')}")
            logger.info(f"  non_abstention_rate = {stats.get('non_abstention_rate'):.4f}")
            logger.info(f"  precision_at_threshold = {stats.get('precision_at_threshold'):.4f}")
            logger.info(f"  relaxed_fallback = {stats.get('relaxed_fallback', False)}")
            args._conformal_abstainer = abstainer
        else:
            args._conformal_abstainer = None

    # 按 batch 处理
    remaining = [item for item in items if item.get("id") not in processed_ids]
    logger.info(
        f"Processing {len(remaining)} items in batches of {batch_size} | "
        f"batch_inference={'ON' if use_batch_inference else 'OFF'}"
    )

    for batch_start in range(0, len(remaining), batch_size):
        batch = remaining[batch_start:batch_start + batch_size]
        batch_results = []

        if use_batch_inference and generator is not None:
            # ===== v9.0 批量推理模式 =====
            # Phase 1: 对整个 batch 并行执行非 LLM 步骤（图构建、执行器、prompt 构建）
            prepared_list = []
            for item in batch:
                table = load_table_json(item, args.table_dir, table_cache, dataset=args.dataset)
                try:
                    prepared = prepare_non_llm_steps(item, table, args, mode)
                    prepared_list.append(prepared)
                except Exception as e:
                    logger.error(f"Prepare error {item.get('id')}: {e}")
                    if getattr(args, "abort_on_generation_error", True):
                        raise RuntimeError(
                            "Non-LLM preparation failed; aborting to avoid invalid artifacts"
                        ) from e
                    prepared_list.append({
                        "result": _make_error_result(item, args, e, "prepare"),
                        "prompt": "",
                        "graph": None,
                        "evidence": None,
                        "interventions": None,
                        "executor_result": None,
                        "all_exec_candidates": None,
                        "item": item,
                        "table_json": table,
                        "_error": True,
                    })

            # Phase 2: 一次性批量推理（过滤掉错误样本的空 prompt）
            valid_indices = [i for i, p in enumerate(prepared_list) if not p.get("_error")]
            prompts_to_generate = [prepared_list[i]["prompt"] for i in valid_indices]

            gen_outputs_map: Dict[int, Dict[str, Any]] = {}
            if prompts_to_generate:
                try:
                    debug_ids = [prepared_list[i]["item"].get("id") for i in valid_indices]
                    effective_max_model_len = _effective_max_model_len(generator, args)

                    if getattr(args, "skip_overlong_primary", False):
                        original_valid_indices = list(valid_indices)
                        keep_local_positions, kept_prompts, skipped_primary, pre_audit = (
                            filter_prompts_by_context_budget(
                                generator=generator,
                                prompts=prompts_to_generate,
                                max_model_len=effective_max_model_len,
                                max_new_tokens=args.max_answer_tokens,
                                logger=logger,
                                ids=debug_ids,
                            )
                        )
                        for local_pos, tok_len, pressure in zip(
                            range(len(original_valid_indices)),
                            pre_audit["lengths"],
                            pre_audit["pressure"],
                        ):
                            idx = original_valid_indices[local_pos]
                            prepared_list[idx]["result"]["input_token_count"] = tok_len
                            prepared_list[idx]["result"]["context_budget"] = pre_audit["budget"]
                            prepared_list[idx]["result"]["context_pressure_ratio"] = pressure
                            prepared_list[idx]["result"]["compute_pressure_tier"] = _pressure_tier(
                                float(pressure)
                            )
                        if skipped_primary:
                            logger.warning(
                                "PRIMARY_PASS_NO_TRUNCATION_SKIP skipped=%s",
                                skipped_primary,
                            )
                            for skip in skipped_primary:
                                local_pos = int(skip["local_index"])
                                idx = original_valid_indices[local_pos]
                                prepared_list[idx]["_primary_context_skip"] = True
                                prepared_list[idx]["result"] = _make_primary_context_overflow_result(
                                    prepared_list[idx],
                                    skip,
                                )
                        valid_indices = [original_valid_indices[pos] for pos in keep_local_positions]
                        prompts_to_generate = kept_prompts
                        debug_ids = [prepared_list[i]["item"].get("id") for i in valid_indices]

                    if not prompts_to_generate:
                        audit = {"lengths": [], "pressure": [], "budget": effective_max_model_len - args.max_answer_tokens}
                    else:
                        audit = assert_no_truncation(
                            generator=generator,
                            prompts=prompts_to_generate,
                            max_model_len=effective_max_model_len,
                            max_new_tokens=args.max_answer_tokens,
                            logger=logger,
                            ids=debug_ids,
                        )
                    for idx, tok_len, pressure in zip(valid_indices, audit["lengths"], audit["pressure"]):
                        prepared_list[idx]["result"]["input_token_count"] = tok_len
                        prepared_list[idx]["result"]["context_budget"] = audit["budget"]
                        prepared_list[idx]["result"]["context_pressure_ratio"] = pressure
                        prepared_list[idx]["result"]["compute_pressure_tier"] = _pressure_tier(float(pressure))
                    if prompts_to_generate:
                        audit_records: List[Dict[str, Any]] = []
                        audit_refs: Dict[int, Dict[str, Any]] = {}

                        for local_idx, global_idx in enumerate(valid_indices):
                            prepared_i = prepared_list[global_idx]
                            prompt_i = prompts_to_generate[local_idx]
                            prompt_type_i = prepared_i.get("result", {}).get("prompt_type", "")
                            record, ref = _make_llm_input_audit_record(
                                prepared=prepared_i,
                                prompt=prompt_i,
                                generator=generator,
                                args=args,
                                prompt_kind="primary",
                                prompt_type=prompt_type_i,
                                max_new_tokens=args.max_answer_tokens,
                                temperature=args.temperature,
                                top_p=args.top_p,
                                logprobs=top_k_logprobs,
                                sequence_index=local_idx,
                            )
                            prepared_i["result"]["llm_input_audit"] = ref
                            if record is not None:
                                audit_records.append(record)
                            audit_refs[global_idx] = ref

                        _append_llm_input_audit(args, audit_records)

                        gen_start = time.time()
                        gen_results = generator.generate(
                            prompts_to_generate,
                            max_new_tokens=args.max_answer_tokens,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            logprobs=top_k_logprobs,
                        )
                        per_prompt_seconds = (time.time() - gen_start) / max(1, len(prompts_to_generate))
                        for local_idx, global_idx in enumerate(valid_indices):
                            gen_result = gen_results[local_idx]
                            gen_result.setdefault("generation_seconds", per_prompt_seconds)
                            gen_outputs_map[global_idx] = gen_result
                except Exception as e:
                    logger.error(f"Batch generate error: {e}")
                    if getattr(args, "abort_on_generation_error", True):
                        raise RuntimeError("Primary batch generation failed; aborting to avoid invalid artifacts") from e
                    # 回退：为每个有效样本返回空输出
                    for global_idx in valid_indices:
                        gen_outputs_map[global_idx] = {"text": "", "logprobs": None}

            # Phase 3: 逐样本完成后续步骤（仲裁、评估）
            for i, prepared in enumerate(prepared_list):
                if prepared.get("_error"):
                    batch_results.append(prepared["result"])
                    continue
                if prepared.get("_primary_context_skip"):
                    batch_results.append(prepared["result"])
                    continue
                try:
                    gen_output = gen_outputs_map.get(i, {"text": "", "logprobs": None})
                    result = finalize_after_llm(prepared, gen_output, args, mode, top_k_logprobs, generator=generator)
                    batch_results.append(result)
                except Exception as e:
                    logger.error(f"Finalize error {prepared['item'].get('id')}: {e}")
                    if getattr(args, "abort_on_generation_error", True):
                        raise RuntimeError(
                            "Finalize failed; aborting to avoid invalid artifacts"
                        ) from e
                    batch_results.append(_make_error_result(prepared["item"], args, e, "finalize"))

            # ===== Phase 4: Adaptive Prompt Router =====
            # v8.3: 纯熵路由 (entropy >= thr → intersection_hint)
            # v8.9 (credal-aware): 三模式 gating + non-degradation guard
            #   - 'cap'（默认，安全）: entropy>=thr AND cw < cw_high 才进 R2
            #     语义：cw 过高表示 reachability interval 过宽（高认知不确定性），R2 几乎不可能改善
            #     E33 实测：cw>=0.30 桶 R2_EM=23.3%（远低于 cw<0.05 桶的 53.6%）
            #   - 'floor'（v8.8 错误模式，已证明有害，仅作消融对比）：entropy>=thr AND cw>=cw_low
            #     E31 cw>=0.10 阻断 120 高潜力低 cw 样本，净损 -21
            #   - 'band'（双侧 gating）：entropy>=thr AND cw_low <= cw < cw_high 才进 R2
            adaptive_prompt = getattr(args, "adaptive_prompt", False)
            entropy_threshold_low = getattr(args, "entropy_threshold_low", 0.05)
            credal_gate = getattr(args, "credal_gate", False)
            credal_gate_mode = getattr(args, "credal_gate_mode", "cap")
            credal_gate_cw = getattr(args, "credal_gate_cw", 0.10)        # floor / band 下界
            credal_gate_cw_high = getattr(args, "credal_gate_cw_high", 0.30)  # cap / band 上界
            non_degradation_guard = getattr(args, "non_degradation_guard", False)

            if adaptive_prompt and generator is not None:
                # 收集需要 Round 2 重新推理的样本
                round2_indices: List[int] = []
                round2_prompts: List[str] = []
                round2_prompt_types: List[str] = []
                router_stats = {
                    "entropy_only": 0,
                    "credal_skipped_low": 0,   # band/floor 下界阻断
                    "credal_skipped_high": 0,  # cap/band 上界阻断（reachability cap）
                    "both_pass": 0,
                }

                for i, prepared in enumerate(prepared_list):
                    if prepared.get("_error"):
                        continue
                    result_i = batch_results[i] if i < len(batch_results) else None
                    if result_i is None or result_i.get("error"):
                        continue

                    entropy_val = result_i.get("first_token_entropy", 0.0)

                    # 第一道门：首 token 熵 >= 阈值
                    if entropy_val < entropy_threshold_low:
                        continue

                    # v8.9 第二道门（可选）：credal_width-based 三模式
                    if credal_gate:
                        cw_val = (
                            result_i.get("probe_diagnostics", {})
                            .get("credal_probe", {})
                            .get("credal_width", 0.0)
                        )
                        if credal_gate_mode == "cap":
                            # cap: cw 太高 → reachability 不足 → 跳过 R2 改用更确定路径
                            if cw_val >= credal_gate_cw_high:
                                router_stats["credal_skipped_high"] += 1
                                # 标记，便于 KG-fallback 等下游模块识别
                                result_i["credal_cap_triggered"] = True
                                result_i["credal_cap_cw"] = cw_val
                                continue
                            router_stats["both_pass"] += 1
                        elif credal_gate_mode == "floor":
                            # floor (v8.8): cw 太低 → 跳过（已证明有害，仅作消融）
                            if cw_val < credal_gate_cw:
                                router_stats["credal_skipped_low"] += 1
                                continue
                            router_stats["both_pass"] += 1
                        elif credal_gate_mode == "band":
                            # band: 仅在 [cw_low, cw_high) 区间触发 R2
                            if cw_val < credal_gate_cw:
                                router_stats["credal_skipped_low"] += 1
                                continue
                            if cw_val >= credal_gate_cw_high:
                                router_stats["credal_skipped_high"] += 1
                                result_i["credal_cap_triggered"] = True
                                result_i["credal_cap_cw"] = cw_val
                                continue
                            router_stats["both_pass"] += 1
                        else:
                            # 未知模式当作 entropy_only
                            router_stats["entropy_only"] += 1
                    else:
                        router_stats["entropy_only"] += 1

                    # 需要升级 prompt 重新推理。
                    # Prefix-stable 模式不重建完整 prompt，只在 R1 完整输入后追加控制后缀。
                    table_json_i = prepared.get("table_json")
                    question_i = prepared["item"].get("question", "")
                    evidence_i = prepared.get("evidence")
                    graph_i = prepared.get("graph")
                    prefix_stable_apr = getattr(args, "prefix_stable_apr", False)
                    apr_suffix_mode = getattr(args, "apr_control_suffix_mode", "intersection_hint")

                    if prefix_stable_apr:
                        old_prompt = prepared.get("prompt", "") or ""
                        new_prompt = build_prefix_stable_apr_prompt(
                            prepared=prepared,
                            old_result=result_i,
                            suffix_mode=apr_suffix_mode,
                        )
                        common_prefix = _common_prefix_len(old_prompt, new_prompt)
                        result_i["apr_prefix_stable"] = True
                        result_i["apr_prefix_common_chars"] = common_prefix
                        result_i["apr_prefix_reuse_ratio_char"] = common_prefix / max(1, len(old_prompt))
                        prompt_type_r2 = f"prefix_stable_{apr_suffix_mode}"
                    else:
                        # v8.3 legacy: 统一使用 intersection_hint（唯一被验证有效的升级路径）
                        new_prompt = build_intersection_hint_prompt(
                            table_json=table_json_i,
                            question=question_i,
                            evidence=evidence_i,
                            graph=graph_i,
                        )
                        prompt_type_r2 = "intersection_hint"

                    round2_indices.append(i)
                    round2_prompts.append(new_prompt)
                    round2_prompt_types.append(prompt_type_r2)

                if round2_prompts:
                    if credal_gate:
                        if credal_gate_mode == "cap":
                            gate_desc = f"entropy>={entropy_threshold_low} AND cw<{credal_gate_cw_high} (cap)"
                        elif credal_gate_mode == "floor":
                            gate_desc = f"entropy>={entropy_threshold_low} AND cw>={credal_gate_cw} (floor)"
                        elif credal_gate_mode == "band":
                            gate_desc = (
                                f"entropy>={entropy_threshold_low} AND "
                                f"{credal_gate_cw}<=cw<{credal_gate_cw_high} (band)"
                            )
                        else:
                            gate_desc = f"entropy>={entropy_threshold_low} (unknown mode)"
                    else:
                        gate_desc = f"entropy>={entropy_threshold_low}"
                    logger.info(
                        f"APR Round 2: re-inferring {len(round2_prompts)} samples "
                        f"(gate: {gate_desc}, prompt={round2_prompt_types[0] if round2_prompt_types else 'none'})"
                    )
                    if credal_gate:
                        logger.info(
                            f"  Router stats: both_pass={router_stats['both_pass']}, "
                            f"credal_skipped_low={router_stats['credal_skipped_low']}, "
                            f"credal_skipped_high={router_stats['credal_skipped_high']}"
                        )
                    try:
                        round2_debug_ids = [
                            prepared_list[idx]["item"].get("id")
                            for idx in round2_indices
                        ]

                        # APR Round 2 is enabled explicitly. Oversized R2 prompts are
                        # configuration errors, not a reason to silently preserve R1.
                        keep_local_positions, kept_round2_prompts, skipped_round2, round2_pre_audit = (
                            filter_prompts_by_context_budget(
                                generator=generator,
                                prompts=round2_prompts,
                                max_model_len=_effective_max_model_len(generator, args),
                                max_new_tokens=args.max_answer_tokens,
                                logger=logger,
                                ids=round2_debug_ids,
                            )
                        )

                        if skipped_round2:
                            raise RuntimeError(
                                f"APR Round 2 prompt exceeds context budget: {skipped_round2}"
                            )

                        round2_indices = [round2_indices[pos] for pos in keep_local_positions]
                        round2_prompt_types = [round2_prompt_types[pos] for pos in keep_local_positions]
                        round2_prompts = kept_round2_prompts

                        if not round2_prompts:
                            raise RuntimeError("APR Round 2 was enabled but no prompts remain after context-budget validation")

                        if round2_prompts:
                            # This should now be a no-op safety check, but it preserves
                            # the same audit format as primary generation and protects
                            # against unexpected tokenizer/template drift.
                            round2_debug_ids = [
                                prepared_list[idx]["item"].get("id")
                                for idx in round2_indices
                            ]
                            round2_audit = assert_no_truncation(
                                generator=generator,
                                prompts=round2_prompts,
                                max_model_len=_effective_max_model_len(generator, args),
                                max_new_tokens=args.max_answer_tokens,
                                logger=logger,
                                ids=round2_debug_ids,
                            )
                            for idx, tok_len, pressure in zip(round2_indices, round2_audit["lengths"], round2_audit["pressure"]):
                                batch_results[idx]["apr_round2_input_token_count"] = tok_len
                                batch_results[idx]["apr_round2_context_budget"] = round2_audit["budget"]
                                batch_results[idx]["apr_round2_context_pressure_ratio"] = pressure
                                batch_results[idx]["apr_round2_compute_pressure_tier"] = _pressure_tier(float(pressure))
                            audit_records: List[Dict[str, Any]] = []
                            for j, global_idx in enumerate(round2_indices):
                                prepared_j = prepared_list[global_idx]
                                record, ref = _make_llm_input_audit_record(
                                    prepared=prepared_j,
                                    prompt=round2_prompts[j],
                                    generator=generator,
                                    args=args,
                                    prompt_kind="apr_round2",
                                    prompt_type=round2_prompt_types[j],
                                    max_new_tokens=args.max_answer_tokens,
                                    temperature=args.temperature,
                                    top_p=args.top_p,
                                    logprobs=top_k_logprobs,
                                    sequence_index=j,
                                )
                                batch_results[global_idx]["apr_round2_llm_input_audit"] = ref
                                if record is not None:
                                    audit_records.append(record)
                            _append_llm_input_audit(args, audit_records)
                            r2_start = time.time()
                            round2_results = generator.generate(
                                round2_prompts,
                                max_new_tokens=args.max_answer_tokens,
                                temperature=args.temperature,
                                top_p=args.top_p,
                                logprobs=top_k_logprobs,
                            )
                            r2_per_prompt_seconds = (time.time() - r2_start) / max(1, len(round2_prompts))
                            # 用 Round 2 结果重新 finalize 这些样本
                            for j, global_idx in enumerate(round2_indices):
                                prepared_j = prepared_list[global_idx]
                                gen_output_r2 = round2_results[j]
                                gen_output_r2.setdefault("generation_seconds", r2_per_prompt_seconds)

                                # 记录 Round 1 信息
                                old_result = batch_results[global_idx]
                                r1_info = {
                                    "r1_answer": old_result.get("llm_answer", ""),
                                    "r1_entropy": old_result.get("first_token_entropy", 0.0),
                                    "r1_prompt_type": old_result.get("prompt_type", ""),
                                }

                                try:
                                    result_r2 = finalize_after_llm(
                                        prepared_j, gen_output_r2, args, mode, top_k_logprobs, generator=None
                                    )
                                    result_r2["apr_round"] = 2
                                    result_r2["apr_upgraded_prompt_type"] = round2_prompt_types[j]
                                    result_r2["apr_round1_info"] = r1_info
                                    result_r2["prompt_type"] = round2_prompt_types[j]
                                    result_r2["prompt_length"] = len(round2_prompts[j])
                                    result_r2["apr_round2_input_token_count"] = round2_audit["lengths"][j]
                                    result_r2["apr_round2_context_budget"] = round2_audit["budget"]
                                    result_r2["apr_round2_context_pressure_ratio"] = round2_audit["pressure"][j]
                                    result_r2["apr_round2_compute_pressure_tier"] = _pressure_tier(
                                        float(round2_audit["pressure"][j])
                                    )

                                    # v8.8: Non-degradation guard
                                    # E25 诊断显示 APR R2 在 273 样本中产生 41 gain vs 79 loss（净 -38）
                                    # 用置信度 + path consensus 仲裁：仅当 R2 不显著弱于 R1 时才接受
                                    if non_degradation_guard:
                                        r1_conf = old_result.get("final_confidence", 0.5)
                                        r2_conf = result_r2.get("final_confidence", 0.5)
                                        r1_source = old_result.get("answer_source", "")
                                        # 安全规则：
                                        # 1) 若 R1 是 path_verified_consensus（91% EM），不接受 R2
                                        # 2) 若 R2 置信度比 R1 低 > 0.05，不接受 R2
                                        if r1_source == "path_verified_consensus":
                                            result_r2 = old_result
                                            result_r2["apr_round2_skipped"] = "r1_path_consensus"
                                        elif r2_conf < r1_conf - 0.05:
                                            # R2 比 R1 弱，回退到 R1
                                            old_result["apr_round2_skipped"] = "r2_lower_conf"
                                            old_result["apr_round2_r2_answer"] = result_r2.get("llm_answer")
                                            old_result["apr_round2_r2_conf"] = r2_conf
                                            result_r2 = old_result

                                    batch_results[global_idx] = result_r2
                                except Exception as e:
                                    raise RuntimeError(
                                        f"APR Round 2 finalize error {prepared_j['item'].get('id')}"
                                    ) from e

                    except Exception as e:
                        logger.error(f"APR Round 2 batch generate error: {e}")
                        if getattr(args, "abort_on_generation_error", True):
                            raise RuntimeError("APR Round 2 generation failed; aborting to avoid invalid artifacts") from e
                        raise RuntimeError("APR Round 2 failed while enabled") from e

            # ===== Phase 5: v9.2 Diverse Self-Consistency =====
            if getattr(args, "self_consistency", False) and generator is not None:
                sc_plan: List[Tuple[int, List[Tuple[str, str]]]] = []
                max_sc = int(getattr(args, "self_consistency_max_samples", 512))
                for i, prepared in enumerate(prepared_list):
                    if len(sc_plan) >= max_sc:
                        break
                    if prepared.get("_error"):
                        continue
                    result_i = batch_results[i] if i < len(batch_results) else None
                    if not result_i or result_i.get("error"):
                        continue
                    if not _should_run_self_consistency(result_i, args):
                        continue
                    variants = _build_self_consistency_variants(prepared, result_i, args)
                    if variants:
                        sc_plan.append((i, variants))

                if sc_plan:
                    sc_prompts: List[str] = []
                    sc_meta: List[Tuple[int, str]] = []
                    for idx, variants in sc_plan:
                        for prompt_type, prompt in variants:
                            sc_meta.append((idx, prompt_type))
                            sc_prompts.append(prompt)
                    logger.info(
                        f"Self-Consistency: generating {len(sc_prompts)} candidates "
                        f"for {len(sc_plan)} risk samples"
                    )
                    try:
                        sc_debug_ids = [
                            prepared_list[idx]["item"].get("id")
                            for idx, _prompt_type in sc_meta
                        ]
                        keep_sc_positions, kept_sc_prompts, skipped_sc, _sc_pre_audit = (
                            filter_prompts_by_context_budget(
                                generator=generator,
                                prompts=sc_prompts,
                                max_model_len=_effective_max_model_len(generator, args),
                                max_new_tokens=args.max_answer_tokens,
                                logger=logger,
                                ids=sc_debug_ids,
                            )
                        )
                        if skipped_sc:
                            raise RuntimeError(
                                f"Self-consistency prompt exceeds context budget: {skipped_sc}"
                            )
                        sc_meta = [sc_meta[pos] for pos in keep_sc_positions]
                        sc_prompts = kept_sc_prompts

                        if not sc_prompts:
                            raise RuntimeError("Self-consistency was enabled but no prompts remain after context-budget validation")

                        if sc_prompts:
                            sc_debug_ids = [
                                prepared_list[idx]["item"].get("id")
                                for idx, _prompt_type in sc_meta
                            ]
                            sc_audit = assert_no_truncation(
                                generator=generator,
                                prompts=sc_prompts,
                                max_model_len=_effective_max_model_len(generator, args),
                                max_new_tokens=args.max_answer_tokens,
                                logger=logger,
                                ids=sc_debug_ids,
                            )
                            for meta_pos, (idx, prompt_type) in enumerate(sc_meta):
                                batch_results[idx].setdefault("self_consistency_prompt_audit", []).append({
                                    "prompt_type": prompt_type,
                                    "skipped": False,
                                    "input_token_count": sc_audit["lengths"][meta_pos],
                                    "context_budget": sc_audit["budget"],
                                    "context_pressure_ratio": sc_audit["pressure"][meta_pos],
                                    "compute_pressure_tier": _pressure_tier(float(sc_audit["pressure"][meta_pos])),
                                })
                            audit_records: List[Dict[str, Any]] = []
                            for j, (global_idx, prompt_type) in enumerate(sc_meta):
                                prepared_j = prepared_list[global_idx]
                                record, ref = _make_llm_input_audit_record(
                                    prepared=prepared_j,
                                    prompt=sc_prompts[j],
                                    generator=generator,
                                    args=args,
                                    prompt_kind="self_consistency",
                                    prompt_type=prompt_type,
                                    max_new_tokens=args.max_answer_tokens,
                                    temperature=getattr(args, "self_consistency_temperature", 0.35),
                                    top_p=args.top_p,
                                    logprobs=top_k_logprobs,
                                    sequence_index=j,
                                )
                                batch_results[global_idx].setdefault("self_consistency_llm_input_audit", []).append(ref)
                                if record is not None:
                                    audit_records.append(record)
                            _append_llm_input_audit(args, audit_records)
                            sc_start = time.time()
                            sc_outputs = generator.generate(
                                sc_prompts,
                                max_new_tokens=args.max_answer_tokens,
                                temperature=getattr(args, "self_consistency_temperature", 0.35),
                                top_p=args.top_p,
                                logprobs=top_k_logprobs,
                            )
                            sc_per_prompt_seconds = (time.time() - sc_start) / max(1, len(sc_prompts))
                            alt_by_idx: Dict[int, List[Dict[str, Any]]] = {}
                            for j, gen_output_sc in enumerate(sc_outputs):
                                global_idx, prompt_type = sc_meta[j]
                                prepared_j = prepared_list[global_idx]
                                try:
                                    gen_output_sc.setdefault("generation_seconds", sc_per_prompt_seconds)
                                    alt = finalize_after_llm(
                                        prepared_j, gen_output_sc, args, mode, top_k_logprobs, generator=None
                                    )
                                    alt["prompt_type"] = prompt_type
                                    alt["self_consistency_candidate"] = True
                                    alt_by_idx.setdefault(global_idx, []).append(alt)
                                except Exception as e:
                                    raise RuntimeError(
                                        f"Self-consistency finalize error {prepared_j['item'].get('id')}"
                                    ) from e
                            for global_idx, alts in alt_by_idx.items():
                                batch_results[global_idx] = _merge_self_consistency_results(
                                    batch_results[global_idx],
                                    alts,
                                    prepared_list[global_idx].get("item", {}),
                                )
                    except Exception as e:
                        logger.error(f"Self-Consistency batch generate error: {e}")
                        if getattr(args, "abort_on_generation_error", True):
                            raise RuntimeError("Self-consistency generation failed; aborting to avoid invalid artifacts") from e
                        raise RuntimeError("Self-consistency failed while enabled") from e

        else:
            # ===== 传统逐样本推理模式（向后兼容）=====
            for item in batch:
                table = load_table_json(item, args.table_dir, table_cache, dataset=args.dataset)
                try:
                    result = process_single(item, table, generator, args, mode)
                    batch_results.append(result)
                except Exception as e:
                    logger.error(f"Error processing {item.get('id')}: {e}")
                    if getattr(args, "abort_on_generation_error", True):
                        raise RuntimeError(
                            "Sample processing failed; aborting to avoid invalid artifacts"
                        ) from e
                    batch_results.append(_make_error_result(item, args, e, "process"))

        for row in batch_results:
            row["run_id"] = run_id
        predictions.extend(batch_results)
        append_jsonl(
            pred_file,
            select_prediction_records(batch_results, run_id=run_id, layout=prediction_record_layout),
        )
        if write_debug_predictions and prediction_record_layout != "legacy":
            append_jsonl(
                debug_pred_file,
                [make_debug_prediction_record(row, run_id=run_id) for row in batch_results],
            )

        # 进度
        done = batch_start + len(batch)
        elapsed = time.time() - total_start
        eta = elapsed / max(done, 1) * (len(remaining) - done)
        logger.info(f"Progress: {done}/{len(remaining)}, elapsed: {elapsed:.0f}s, ETA: {eta:.0f}s")

    total_time = time.time() - total_start
    logger.info(f"Total processing time: {total_time:.1f}s")

    # --- 汇总评估 ---
    metrics = compute_full_metrics(predictions, logger)
    metrics["runtime"] = {
        "total_seconds": total_time,
        "samples_per_second": len(predictions) / max(total_time, 1),
    }
    metrics["artifact_layout"] = {
        "prediction_record_layout": prediction_record_layout,
        "predictions_jsonl": "compact research rows" if prediction_record_layout == "research" else "legacy full rows",
        "debug_predictions_jsonl": bool(write_debug_predictions and prediction_record_layout != "legacy"),
    }
    write_json(os.path.join(args.output_dir, "metrics.json"), metrics)
    logger.info("Pipeline complete. Metrics saved.")


def compute_full_metrics(predictions: List[Dict[str, Any]], logger) -> Dict[str, Any]:
    """计算完整指标集"""
    total = len(predictions)
    if total == 0:
        return {"total": 0}

    def _safe_rate(num: int, den: int) -> float:
        return num / den if den else 0.0
    dataset = normalize_dataset_name(predictions[0].get("dataset", "hitab"))
    if dataset == "aitqa":
        primary_metric_name = "aitqa_official_em"
    elif dataset == "tablebench":
        primary_metric_name = "tablebench_official_em"
    elif dataset == "sstqa_zh":
        primary_metric_name = "sstqa_zh_official_em"
    else:
        primary_metric_name = "hitab_official_em"

    def _answer_ok(answer: Any, gold: Any) -> bool:
        if dataset in ("aitqa", "tablebench", "sstqa_zh"):
            return bool(dataset_answer_match(dataset, gold, answer))
        return bool(evaluate_answer_multi_caliber(answer, gold).get("hitab_official_em", False))

    for pred in predictions:
        if dataset == "aitqa" and "aitqa_official_em" not in pred:
            pred["aitqa_official_em"] = _answer_ok(pred.get("final_answer", ""), pred.get("gold_answer", ""))
        elif dataset == "tablebench" and "tablebench_official_em" not in pred:
            pred["tablebench_official_em"] = _answer_ok(pred.get("final_answer", ""), pred.get("gold_answer", ""))
        elif dataset == "sstqa_zh" and "sstqa_zh_official_em" not in pred:
            pred["sstqa_zh_official_em"] = _answer_ok(pred.get("final_answer", ""), pred.get("gold_answer", ""))

    def _primary_correct(pred: Dict[str, Any]) -> bool:
        if primary_metric_name in pred:
            return bool(pred.get(primary_metric_name, False))
        return _answer_ok(pred.get("final_answer", ""), pred.get("gold_answer", ""))

    # 四口径 EM
    calibers = ["strict_em", "numeric_em", "set_em", "hitab_official_em"]
    if primary_metric_name not in calibers:
        calibers.append(primary_metric_name)
    em_counts = {c: sum(1 for p in predictions if p.get(c, False)) for c in calibers}
    em_rates = {c: em_counts[c] / total for c in calibers}

    # 答案来源分布
    source_dist = {}
    for p in predictions:
        src = p.get("answer_source", "unknown")
        source_dist[src] = source_dist.get(src, 0) + 1

    # 按来源的 EM (hitab_official_em)
    source_em = {}
    for src in source_dist:
        src_preds = [p for p in predictions if p.get("answer_source") == src]
        if src_preds:
            source_em[src] = {
                "count": len(src_preds),
                "hitab_em": sum(1 for p in src_preds if p.get("hitab_official_em", False)) / len(src_preds),
                "primary_em": sum(1 for p in src_preds if _primary_correct(p)) / len(src_preds),
            }

    # 执行器统计 + operation-level EM
    executor_stats = {}
    operation_em = {}
    conformal_abstain_by_operation = {}
    exec_preds = [p for p in predictions if "executor_operation" in p]
    if exec_preds:
        op_dist = {}
        op_correct = {}
        op_abstain = {}
        for p in exec_preds:
            op = p.get("executor_operation", "unknown")
            op_dist[op] = op_dist.get(op, 0) + 1
            op_correct[op] = op_correct.get(op, 0) + int(_primary_correct(p))
            if p.get("answer_source") == "conformal_abstain":
                op_abstain[op] = op_abstain.get(op, 0) + 1
        executor_stats = {
            "total_executed": len(exec_preds),
            "valid_rate": sum(1 for p in exec_preds if p.get("executor_valid", False)) / len(exec_preds),
            "operation_distribution": op_dist,
        }
        operation_em = {
            op: {
                "count": count,
                "correct": op_correct.get(op, 0),
                "hitab_em": op_correct.get(op, 0) / count if count else 0.0,
            }
            for op, count in sorted(op_dist.items())
        }
        conformal_abstain_by_operation = {
            op: {
                "count": op_abstain.get(op, 0),
                "rate": op_abstain.get(op, 0) / count if count else 0.0,
            }
            for op, count in sorted(op_dist.items())
        }

    operation_support_metrics = {}
    support_preds = [p for p in predictions if p.get("operation_support_diagnostic_enabled")]
    if support_preds:
        def _new_support_bucket() -> Dict[str, Any]:
            return {
                "count": 0,
                "executor_official_correct": 0,
                "final_official_correct": 0,
                "matches_final": 0,
                "executor_valid": 0,
                "support_cell_count_sum": 0.0,
                "gap_distribution": {},
                "expected_role_distribution": {},
            }

        def _add_support_row(bucket: Dict[str, Any], pred: Dict[str, Any]) -> None:
            bucket["count"] += 1
            bucket["executor_official_correct"] += int(bool(pred.get("operation_support_official_correct")))
            bucket["final_official_correct"] += int(_primary_correct(pred))
            bucket["matches_final"] += int(bool(pred.get("operation_support_matches_final")))
            bucket["executor_valid"] += int(bool(pred.get("operation_support_valid")))
            bucket["support_cell_count_sum"] += float(pred.get("operation_support_cell_count", 0) or 0)
            gap = str(pred.get("operation_support_gap") or "unknown")
            bucket["gap_distribution"][gap] = bucket["gap_distribution"].get(gap, 0) + 1
            role = str(pred.get("operation_support_expected_role") or "unknown")
            bucket["expected_role_distribution"][role] = bucket["expected_role_distribution"].get(role, 0) + 1

        def _finish_support_bucket(bucket: Dict[str, Any]) -> Dict[str, Any]:
            count = int(bucket.get("count", 0))
            return {
                "count": count,
                "executor_official_em": _safe_rate(int(bucket.get("executor_official_correct", 0)), count),
                "final_official_em": _safe_rate(int(bucket.get("final_official_correct", 0)), count),
                "matches_final_rate": _safe_rate(int(bucket.get("matches_final", 0)), count),
                "executor_valid_rate": _safe_rate(int(bucket.get("executor_valid", 0)), count),
                "avg_support_cell_count": (float(bucket.get("support_cell_count_sum", 0.0)) / count if count else 0.0),
                "gap_distribution": bucket.get("gap_distribution", {}),
                "expected_role_distribution": bucket.get("expected_role_distribution", {}),
            }

        overall_support = _new_support_bucket()
        by_question_operation: Dict[str, Dict[str, Any]] = {}
        by_executor_operation: Dict[str, Dict[str, Any]] = {}
        by_expected_role: Dict[str, Dict[str, Any]] = {}
        for pred in support_preds:
            _add_support_row(overall_support, pred)
            qop = str(pred.get("question_operation") or "unknown")
            eop = str(pred.get("operation_support_operation") or pred.get("executor_operation") or "unknown")
            role = str(pred.get("operation_support_expected_role") or "unknown")
            _add_support_row(by_question_operation.setdefault(qop, _new_support_bucket()), pred)
            _add_support_row(by_executor_operation.setdefault(eop, _new_support_bucket()), pred)
            _add_support_row(by_expected_role.setdefault(role, _new_support_bucket()), pred)
        operation_support_metrics = {
            "enabled_count": len(support_preds),
            "overall": _finish_support_bucket(overall_support),
            "by_question_operation": {
                key: _finish_support_bucket(value)
                for key, value in sorted(by_question_operation.items())
            },
            "by_executor_operation": {
                key: _finish_support_bucket(value)
                for key, value in sorted(by_executor_operation.items())
            },
            "by_expected_role": {
                key: _finish_support_bucket(value)
                for key, value in sorted(by_expected_role.items())
            },
        }

    operation_role_target_metrics = {}
    role_target_preds = [
        p for p in predictions
        if p.get("operation_support_role_target_diagnostic_enabled")
    ]
    if role_target_preds:
        def _target_count_bin(count: int) -> str:
            if count <= 0:
                return "target_0"
            if count == 1:
                return "target_1"
            if count <= 4:
                return "target_2_4"
            if count <= 9:
                return "target_5_9"
            return "target_10_plus"

        def _new_role_target_bucket() -> Dict[str, Any]:
            return {
                "count": 0,
                "selected_official_correct": 0,
                "reranked_official_correct": 0,
                "final_official_correct": 0,
                "rerank_changed": 0,
                "reranked_role_compatible": 0,
                "reranked_operation_compatible": 0,
                "filter_cell_count_sum": 0.0,
                "target_cell_count_sum": 0.0,
                "target_cell_count_correct_sum": 0.0,
                "target_cell_count_wrong_sum": 0.0,
                "reranked_wrong": 0,
                "entity_surface_numeric_role": 0,
                "surface_role_source_count": 0,
                "surface_structural_role_agree": 0,
                "surface_structural_role_conflict": 0,
                "gap_distribution": {},
                "answer_role_distribution": {},
                "role_source_distribution": {},
                "surface_role_source_distribution": {},
                "structural_role_source_distribution": {},
                "target_cell_count_bins": {},
            }

        def _add_role_target_row(bucket: Dict[str, Any], pred: Dict[str, Any]) -> None:
            bucket["count"] += 1
            target_count = float(pred.get("operation_support_target_cell_count", 0) or 0)
            filter_count = float(pred.get("operation_support_filter_cell_count", 0) or 0)
            reranked_correct = bool(pred.get("operation_support_reranked_official_correct"))
            bucket["selected_official_correct"] += int(bool(pred.get("operation_support_official_correct")))
            bucket["reranked_official_correct"] += int(reranked_correct)
            bucket["final_official_correct"] += int(_primary_correct(pred))
            bucket["rerank_changed"] += int(bool(pred.get("operation_support_rerank_changed")))
            bucket["reranked_role_compatible"] += int(bool(pred.get("operation_support_reranked_role_compatible")))
            bucket["reranked_operation_compatible"] += int(bool(pred.get("operation_support_reranked_operation_compatible")))
            bucket["filter_cell_count_sum"] += filter_count
            bucket["target_cell_count_sum"] += target_count
            role = str(pred.get("operation_support_answer_role") or "unknown")
            if reranked_correct:
                bucket["target_cell_count_correct_sum"] += target_count
            else:
                bucket["target_cell_count_wrong_sum"] += target_count
                bucket["reranked_wrong"] += 1
            if pred.get("operation_support_entity_surface_numeric_role") or (
                _entity_surface_requested(str(pred.get("question") or "")) and role == "numeric"
            ):
                bucket["entity_surface_numeric_role"] += 1
            if pred.get("operation_support_role_source_is_surface_heuristic"):
                bucket["surface_role_source_count"] += 1
            agreement = pred.get("operation_support_surface_structural_role_agreement")
            if agreement is True:
                bucket["surface_structural_role_agree"] += 1
            elif agreement is False:
                bucket["surface_structural_role_conflict"] += 1
            gap = str(pred.get("operation_support_reranked_gap") or "unknown")
            bucket["gap_distribution"][gap] = bucket["gap_distribution"].get(gap, 0) + 1
            bucket["answer_role_distribution"][role] = bucket["answer_role_distribution"].get(role, 0) + 1
            source = str(pred.get("operation_support_role_source") or "unknown")
            bucket["role_source_distribution"][source] = bucket["role_source_distribution"].get(source, 0) + 1
            surface_source = str(pred.get("operation_support_surface_role_source") or "unknown")
            bucket["surface_role_source_distribution"][surface_source] = (
                bucket["surface_role_source_distribution"].get(surface_source, 0) + 1
            )
            structural_source = str(pred.get("operation_support_structural_role_source") or "unknown")
            bucket["structural_role_source_distribution"][structural_source] = (
                bucket["structural_role_source_distribution"].get(structural_source, 0) + 1
            )
            bin_key = _target_count_bin(int(target_count))
            bin_bucket = bucket["target_cell_count_bins"].setdefault(
                bin_key,
                {"count": 0, "reranked_correct": 0, "final_correct": 0, "target_cell_count_sum": 0.0, "filter_cell_count_sum": 0.0},
            )
            bin_bucket["count"] += 1
            bin_bucket["reranked_correct"] += int(reranked_correct)
            bin_bucket["final_correct"] += int(_primary_correct(pred))
            bin_bucket["target_cell_count_sum"] += target_count
            bin_bucket["filter_cell_count_sum"] += filter_count

        def _finish_role_target_bucket(bucket: Dict[str, Any]) -> Dict[str, Any]:
            count = int(bucket.get("count", 0))
            reranked_correct = int(bucket.get("reranked_official_correct", 0))
            reranked_wrong = int(bucket.get("reranked_wrong", 0))
            target_bins = {}
            for key, value in sorted(bucket.get("target_cell_count_bins", {}).items()):
                bin_count = int(value.get("count", 0))
                target_bins[key] = {
                    "count": bin_count,
                    "reranked_executor_official_em": _safe_rate(int(value.get("reranked_correct", 0)), bin_count),
                    "final_official_em": _safe_rate(int(value.get("final_correct", 0)), bin_count),
                    "avg_target_cell_count": (
                        float(value.get("target_cell_count_sum", 0.0)) / bin_count if bin_count else 0.0
                    ),
                    "avg_filter_cell_count": (
                        float(value.get("filter_cell_count_sum", 0.0)) / bin_count if bin_count else 0.0
                    ),
                }
            return {
                "count": count,
                "selected_executor_official_em": _safe_rate(int(bucket.get("selected_official_correct", 0)), count),
                "reranked_executor_official_em": _safe_rate(int(bucket.get("reranked_official_correct", 0)), count),
                "final_official_em": _safe_rate(int(bucket.get("final_official_correct", 0)), count),
                "rerank_changed_rate": _safe_rate(int(bucket.get("rerank_changed", 0)), count),
                "reranked_role_compatible_rate": _safe_rate(int(bucket.get("reranked_role_compatible", 0)), count),
                "reranked_operation_compatible_rate": _safe_rate(int(bucket.get("reranked_operation_compatible", 0)), count),
                "avg_filter_cell_count": (float(bucket.get("filter_cell_count_sum", 0.0)) / count if count else 0.0),
                "avg_target_cell_count": (float(bucket.get("target_cell_count_sum", 0.0)) / count if count else 0.0),
                "avg_target_cell_count_correct": (
                    float(bucket.get("target_cell_count_correct_sum", 0.0)) / reranked_correct if reranked_correct else 0.0
                ),
                "avg_target_cell_count_wrong": (
                    float(bucket.get("target_cell_count_wrong_sum", 0.0)) / reranked_wrong if reranked_wrong else 0.0
                ),
                "entity_surface_numeric_role": int(bucket.get("entity_surface_numeric_role", 0)),
                "entity_surface_numeric_role_rate": _safe_rate(int(bucket.get("entity_surface_numeric_role", 0)), count),
                "surface_role_source_count": int(bucket.get("surface_role_source_count", 0)),
                "surface_role_source_rate": _safe_rate(int(bucket.get("surface_role_source_count", 0)), count),
                "surface_structural_role_agree": int(bucket.get("surface_structural_role_agree", 0)),
                "surface_structural_role_conflict": int(bucket.get("surface_structural_role_conflict", 0)),
                "gap_distribution": bucket.get("gap_distribution", {}),
                "answer_role_distribution": bucket.get("answer_role_distribution", {}),
                "role_source_distribution": bucket.get("role_source_distribution", {}),
                "surface_role_source_distribution": bucket.get("surface_role_source_distribution", {}),
                "structural_role_source_distribution": bucket.get("structural_role_source_distribution", {}),
                "target_cell_count_bins": target_bins,
            }

        overall_rt = _new_role_target_bucket()
        by_question_operation_rt: Dict[str, Dict[str, Any]] = {}
        by_dataset_subtype_rt: Dict[str, Dict[str, Any]] = {}
        by_answer_role_rt: Dict[str, Dict[str, Any]] = {}
        for pred in role_target_preds:
            _add_role_target_row(overall_rt, pred)
            qop = str(pred.get("question_operation") or "unknown")
            subtype = str(pred.get("dataset_question_subtype") or pred.get("dataset_question_type") or "unknown")
            role = str(pred.get("operation_support_answer_role") or "unknown")
            _add_role_target_row(by_question_operation_rt.setdefault(qop, _new_role_target_bucket()), pred)
            _add_role_target_row(by_dataset_subtype_rt.setdefault(subtype, _new_role_target_bucket()), pred)
            _add_role_target_row(by_answer_role_rt.setdefault(role, _new_role_target_bucket()), pred)
        operation_role_target_metrics = {
            "enabled_count": len(role_target_preds),
            "overall": _finish_role_target_bucket(overall_rt),
            "by_question_operation": {
                key: _finish_role_target_bucket(value)
                for key, value in sorted(by_question_operation_rt.items())
            },
            "by_dataset_question_subtype": {
                key: _finish_role_target_bucket(value)
                for key, value in sorted(by_dataset_subtype_rt.items())
            },
            "by_answer_role": {
                key: _finish_role_target_bucket(value)
                for key, value in sorted(by_answer_role_rt.items())
            },
        }

    operation_commit_gate_metrics = {}
    commit_gate_preds = [
        p for p in predictions
        if p.get("operation_support_commit_gate_diagnostic_enabled")
    ]
    if commit_gate_preds:
        def _new_commit_bucket() -> Dict[str, Any]:
            return {
                "count": 0,
                "eligible": 0,
                "commit_official_correct": 0,
                "final_official_correct": 0,
                "total_baseline_official_correct": 0,
                "applied": 0,
                "target_cell_count_sum": 0.0,
                "filter_cell_count_sum": 0.0,
                "surface_role_source": 0,
                "surface_role_without_structural_support": 0,
                "gap_distribution": {},
                "applied_gap_distribution": {},
                "reject_reason_distribution": {},
                "surface_risk_reason_distribution": {},
            }

        def _add_commit_row(bucket: Dict[str, Any], pred: Dict[str, Any]) -> None:
            bucket["count"] += 1
            eligible = bool(pred.get("operation_support_commit_eligible"))
            bucket["eligible"] += int(eligible)
            baseline_ok = bool(pred.get("operation_support_commit_baseline_official_correct"))
            if "operation_support_commit_baseline_official_correct" not in pred:
                baseline_ok = _primary_correct(pred)
            bucket["total_baseline_official_correct"] += int(baseline_ok)
            if eligible:
                bucket["commit_official_correct"] += int(bool(pred.get("operation_support_commit_official_correct")))
                bucket["final_official_correct"] += int(baseline_ok)
                bucket["applied"] += int(bool(pred.get("operation_support_commit_applied")))
                bucket["target_cell_count_sum"] += float(pred.get("operation_support_target_cell_count", 0) or 0)
                bucket["filter_cell_count_sum"] += float(pred.get("operation_support_filter_cell_count", 0) or 0)
                gap = str(pred.get("operation_support_commit_gap") or "unknown")
                bucket["gap_distribution"][gap] = bucket["gap_distribution"].get(gap, 0) + 1
                if pred.get("operation_support_commit_applied"):
                    bucket["applied_gap_distribution"][gap] = bucket["applied_gap_distribution"].get(gap, 0) + 1
            if pred.get("operation_support_commit_role_source_is_surface_heuristic"):
                bucket["surface_role_source"] += 1
                if not pred.get("operation_support_commit_role_structural_supported"):
                    bucket["surface_role_without_structural_support"] += 1
            for reason in pred.get("operation_support_commit_reject_reasons", []) or []:
                reason = str(reason)
                bucket["reject_reason_distribution"][reason] = bucket["reject_reason_distribution"].get(reason, 0) + 1
            for reason in pred.get("operation_support_commit_surface_risk_reasons", []) or []:
                reason = str(reason)
                bucket["surface_risk_reason_distribution"][reason] = (
                    bucket["surface_risk_reason_distribution"].get(reason, 0) + 1
                )

        def _finish_commit_bucket(bucket: Dict[str, Any]) -> Dict[str, Any]:
            count = int(bucket.get("count", 0))
            eligible = int(bucket.get("eligible", 0))
            gaps = bucket.get("gap_distribution", {})
            applied_gaps = bucket.get("applied_gap_distribution", {})
            gain = int(gaps.get("commit_correct_final_wrong", 0))
            loss = int(gaps.get("final_correct_commit_wrong", 0))
            applied_gain = int(applied_gaps.get("commit_correct_final_wrong", 0))
            applied_loss = int(applied_gaps.get("final_correct_commit_wrong", 0))
            applied_w2w = int(applied_gaps.get("both_wrong_different", 0))
            total_baseline_correct = int(bucket.get("total_baseline_official_correct", 0))
            return {
                "count": count,
                "eligible_count": eligible,
                "coverage_rate": _safe_rate(eligible, count),
                "commit_executor_official_em": _safe_rate(int(bucket.get("commit_official_correct", 0)), eligible),
                "final_official_em_on_eligible": _safe_rate(int(bucket.get("final_official_correct", 0)), eligible),
                "applied_count": int(bucket.get("applied", 0)),
                "avg_target_cell_count_on_eligible": (
                    float(bucket.get("target_cell_count_sum", 0.0)) / eligible if eligible else 0.0
                ),
                "avg_filter_cell_count_on_eligible": (
                    float(bucket.get("filter_cell_count_sum", 0.0)) / eligible if eligible else 0.0
                ),
                "eligible_gain_count": gain,
                "eligible_loss_count": loss,
                "net_gain_count": gain - loss,
                "applied_gain_count": applied_gain,
                "applied_loss_count": applied_loss,
                "applied_wrong_to_wrong_count": applied_w2w,
                "applied_wrong_to_wrong_rate": _safe_rate(applied_w2w, int(bucket.get("applied", 0))),
                "surface_role_source_count": int(bucket.get("surface_role_source", 0)),
                "surface_role_source_rate": _safe_rate(int(bucket.get("surface_role_source", 0)), count),
                "surface_role_without_structural_support_count": int(
                    bucket.get("surface_role_without_structural_support", 0)
                ),
                "projected_primary_em_if_committed": _safe_rate(total_baseline_correct + gain - loss, count),
                "projected_delta_em_if_committed": _safe_rate(gain - loss, count),
                "gap_distribution": gaps,
                "applied_gap_distribution": applied_gaps,
                "reject_reason_distribution": bucket.get("reject_reason_distribution", {}),
                "surface_risk_reason_distribution": bucket.get("surface_risk_reason_distribution", {}),
            }

        overall_commit = _new_commit_bucket()
        by_subtype_commit: Dict[str, Dict[str, Any]] = {}
        by_qop_commit: Dict[str, Dict[str, Any]] = {}
        for pred in commit_gate_preds:
            _add_commit_row(overall_commit, pred)
            subtype = str(pred.get("dataset_question_subtype") or pred.get("dataset_question_type") or "unknown")
            qop = str(pred.get("question_operation") or "unknown")
            _add_commit_row(by_subtype_commit.setdefault(subtype, _new_commit_bucket()), pred)
            _add_commit_row(by_qop_commit.setdefault(qop, _new_commit_bucket()), pred)
        operation_commit_gate_metrics = {
            "enabled_count": len(commit_gate_preds),
            "overall": _finish_commit_bucket(overall_commit),
            "by_dataset_question_subtype": {
                key: _finish_commit_bucket(value)
                for key, value in sorted(by_subtype_commit.items())
            },
            "by_question_operation": {
                key: _finish_commit_bucket(value)
                for key, value in sorted(by_qop_commit.items())
            },
        }

    operation_commit_certificate_metrics = {}
    commit_cert_preds = [
        p for p in predictions
        if isinstance(p.get("operation_support_commit_certificate"), dict)
    ]
    if commit_cert_preds:
        condition_counts: Dict[str, int] = {}
        condition_denoms: Dict[str, int] = {}
        reject_dist: Dict[str, int] = {}
        hard_ok = 0
        eligible = 0
        applied = 0
        blocked_by_policy = 0
        for pred in commit_cert_preds:
            cert = pred.get("operation_support_commit_certificate") or {}
            hard_ok += int(bool(cert.get("hard_conditions_satisfied")))
            eligible += int(bool(cert.get("eligible")))
            applied += int(bool(cert.get("applied")))
            blocked_by_policy += int(bool(cert.get("blocked_by_policy")))
            for name, value in (cert.get("hard_conditions") or {}).items():
                condition_denoms[name] = condition_denoms.get(name, 0) + 1
                condition_counts[name] = condition_counts.get(name, 0) + int(bool(value))
            for reason in cert.get("reject_reasons", []) or []:
                reason = str(reason)
                reject_dist[reason] = reject_dist.get(reason, 0) + 1
        operation_commit_certificate_metrics = {
            "enabled_count": len(commit_cert_preds),
            "eligible_count": eligible,
            "hard_conditions_satisfied_count": hard_ok,
            "hard_conditions_satisfied_rate": hard_ok / len(commit_cert_preds),
            "applied_count": applied,
            "blocked_by_policy_count": blocked_by_policy,
            "condition_pass_rate": {
                key: condition_counts.get(key, 0) / condition_denoms[key]
                for key in sorted(condition_denoms)
            },
            "condition_pass_count": {
                key: condition_counts.get(key, 0)
                for key in sorted(condition_denoms)
            },
            "reject_reason_distribution": dict(sorted(reject_dist.items())),
        }

    # 校准指标
    confidences = [p.get("final_confidence", 0.5) for p in predictions]
    correctness = [_primary_correct(p) for p in predictions]
    cal_metrics = compute_calibration_metrics(confidences, correctness)

    # 错误类型分布
    error_dist = {}
    for p in predictions:
        if not _primary_correct(p):
            etype = p.get("error_type", "unknown")
            error_dist[etype] = error_dist.get(etype, 0) + 1

    def _slice_metrics(key: str) -> Dict[str, Any]:
        buckets: Dict[str, List[Dict[str, Any]]] = {}
        for pred in predictions:
            value = pred.get(key, "")
            if value not in (None, ""):
                buckets.setdefault(str(value), []).append(pred)
        return {
            name: {
                "count": len(rows),
                "primary_em": sum(1 for row in rows if _primary_correct(row)) / len(rows),
                "strict_em": sum(1 for row in rows if row.get("strict_em", False)) / len(rows),
                "numeric_em": sum(1 for row in rows if row.get("numeric_em", False)) / len(rows),
            }
            for name, rows in sorted(buckets.items())
            if rows
        }

    dataset_slices = {
        "dataset_question_type": _slice_metrics("dataset_question_type"),
        "dataset_question_subtype": _slice_metrics("dataset_question_subtype"),
        "row_hierarchy_needed": _slice_metrics("row_hierarchy_needed"),
    }

    # Logit 熵统计 + 熵桶 EM
    entropies = [p.get("first_token_entropy", -1) for p in predictions if "first_token_entropy" in p]
    entropy_stats = {}
    entropy_buckets = []
    if entropies:
        entropy_stats = {
            "mean": sum(entropies) / len(entropies),
            "min": min(entropies),
            "max": max(entropies),
            "count": len(entropies),
        }
        for lo, hi in [(0.0, 0.01), (0.01, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.50), (0.50, 1.10)]:
            bucket_preds = [
                p for p in predictions
                if lo <= float(p.get("first_token_entropy", 0.0) or 0.0) < hi
            ]
            if bucket_preds:
                bucket_errors = {}
                for p in bucket_preds:
                    if not _primary_correct(p):
                        etype = p.get("error_type", "unknown")
                        bucket_errors[etype] = bucket_errors.get(etype, 0) + 1
                entropy_buckets.append({
                    "range": [lo, hi],
                    "count": len(bucket_preds),
                    "hitab_em": sum(1 for p in bucket_preds if p.get("hitab_official_em", False)) / len(bucket_preds),
                    "primary_em": sum(1 for p in bucket_preds if _primary_correct(p)) / len(bucket_preds),
                    "error_distribution": bucket_errors,
                })

    # Prompt length 分布
    prompt_lengths = sorted(p.get("prompt_length", 0) for p in predictions if "prompt_length" in p)
    prompt_length_stats = {}
    if prompt_lengths:
        def _quantile(vals, q):
            idx = min(len(vals) - 1, max(0, int((len(vals) - 1) * q)))
            return vals[idx]
        prompt_length_stats = {
            "mean": sum(prompt_lengths) / len(prompt_lengths),
            "min": prompt_lengths[0],
            "p50": _quantile(prompt_lengths, 0.50),
            "p75": _quantile(prompt_lengths, 0.75),
            "p90": _quantile(prompt_lengths, 0.90),
            "p95": _quantile(prompt_lengths, 0.95),
            "max": prompt_lengths[-1],
            "count": len(prompt_lengths),
        }

    # v8.6: Credal Probe 分桶统计
    probe_metrics = {}
    probe_metrics = aggregate_probe_metrics(predictions)

    module_keys = [
        "prompt_style_routed",
        "normalizer_applied",
        "oracle_normalizer_applied",
        "hceg_fallback_should_trigger",
        "hceg_fallback_candidate",
        "hceg_fallback_diagnostic_candidate",
        "hceg_fallback_role_aware_changed",
        "hceg_fallback_applied",
        "certificate_commit_candidate",
        "certificate_commit_recommended",
        "certificate_commit_shadow_recommended",
        "certificate_commit_operation_verified",
        "certificate_commit_compare_direction_verified",
        "certificate_commit_numeric_direction_verified",
        "certificate_commit_conformal_score_pass",
        "certificate_commit_conformal_recommended",
        "certificate_commit_conformal_shadow_accepted",
        "certificate_commit_applied",
        "operation_support_diagnostic_enabled",
        "operation_support_role_target_diagnostic_enabled",
        "operation_support_rerank_changed",
        "operation_support_entity_surface_numeric_role",
        "operation_support_surface_heuristic_mode",
        "operation_support_role_source_is_surface_heuristic",
        "operation_support_surface_answer_role",
        "operation_support_surface_role_source",
        "operation_support_structural_answer_role",
        "operation_support_structural_role_source",
        "operation_support_surface_structural_role_agreement",
        "operation_support_commit_gate_diagnostic_enabled",
        "operation_support_commit_role_source_is_surface_heuristic",
        "operation_support_commit_role_structural_supported",
        "operation_support_commit_surface_heuristic_mode",
        "operation_support_commit_surface_risks_enforced",
        "operation_support_commit_eligible",
        "operation_support_commit_official_correct",
        "operation_support_commit_accepted",
        "operation_support_commit_applied",
        "operation_support_commit_same_answer",
        "operation_support_commit_blocked_by_policy",
        "operation_support_commit_blocked_empty_answer",
        "main_cert_profile",
        "heuristic_surface_used_for_commit",
        "legacy_commit_path_used",
        "non_certificate_answer_mutation_used",
        "commit_decision_is_boolean_conjunction",
        "source_risk_calibration_applied",
        "black_box_api_generator",
        "black_box_answer_freeze_active",
        "black_box_answer_freeze_applied",
        "black_box_answer_freeze_answer_changed",
        "black_box_answer_freeze_source_only",
        "black_box_answer_freeze_skipped_empty_llm",
        "black_box_semantic_commit_blocked",
        "api_format_normalizer_applied",
        "api_logprobs_unavailable",
        "primary_skipped_no_truncation",
        "apr_round2_skipped_no_truncation",
        "self_consistency_skipped_no_truncation",
        "self_consistency_used",
        "self_consistency_changed",
        "self_consistency_empty_vote_group",
    ]
    module_diagnostics = {
        key: {
            "count": sum(1 for p in predictions if p.get(key)),
            "rate": _safe_rate(sum(1 for p in predictions if p.get(key)), total),
        }
        for key in module_keys
    }

    hceg_triggered = [p for p in predictions if p.get("hceg_fallback_should_trigger")]
    hceg_candidate_without_trigger = sum(
        1 for p in predictions
        if p.get("hceg_fallback_candidate") and not p.get("hceg_fallback_should_trigger")
    )
    hceg_candidates = [
        p for p in predictions
        if p.get("hceg_fallback_candidate")
    ]
    hceg_triggered_candidates = [p for p in hceg_candidates if p.get("hceg_fallback_should_trigger")]
    hceg_diagnostic_candidates = [p for p in hceg_candidates if p.get("hceg_fallback_diagnostic_candidate")]
    hceg_candidate_correct = 0
    hceg_potential_gain = 0
    hceg_potential_loss = 0
    hceg_candidate_compatible = 0
    hceg_role_mismatch = 0
    hceg_role_aware_changed = 0
    hceg_raw_candidate_count = 0
    hceg_raw_candidate_correct = 0
    hceg_role_aware_gain_over_raw = 0
    hceg_role_aware_loss_over_raw = 0
    hceg_type_dist: Dict[str, int] = {}
    hceg_expected_role_dist: Dict[str, int] = {}
    hceg_by_type: Dict[str, Dict[str, Any]] = {}
    hceg_scope = hceg_candidates if hceg_candidates else hceg_triggered
    for p in hceg_scope:
        role = p.get("hceg_fallback_expected_role") or "unknown"
        hceg_expected_role_dist[role] = hceg_expected_role_dist.get(role, 0) + 1
    for p in hceg_candidates:
        candidate = p.get("hceg_fallback_candidate", "")
        gold = p.get("gold_answer", "")
        cand_ok = _answer_ok(candidate, gold)
        pre_answer = p.get(
            "final_answer_pre_certificate_commit",
            p.get("final_answer_pre_hceg_fallback", p.get("final_answer", "")),
        )
        pre_ok = _answer_ok(pre_answer, gold)
        hceg_candidate_correct += int(cand_ok)
        hceg_potential_gain += int(cand_ok and not pre_ok)
        hceg_potential_loss += int((not cand_ok) and pre_ok)
        compatible = bool(p.get("hceg_fallback_candidate_compatible"))
        hceg_candidate_compatible += int(compatible)
        hceg_role_mismatch += int(bool(p.get("hceg_fallback_role_mismatch")))
        raw_candidate = p.get("hceg_fallback_raw_candidate")
        if raw_candidate:
            raw_ok = _answer_ok(raw_candidate, gold)
            hceg_raw_candidate_count += 1
            hceg_raw_candidate_correct += int(raw_ok)
            if _canonical_answer_key(raw_candidate) != _canonical_answer_key(candidate):
                hceg_role_aware_changed += 1
                hceg_role_aware_gain_over_raw += int(cand_ok and not raw_ok)
                hceg_role_aware_loss_over_raw += int((not cand_ok) and raw_ok)
        surface_type = p.get("hceg_fallback_candidate_type") or _answer_surface_type(candidate)
        hceg_type_dist[surface_type] = hceg_type_dist.get(surface_type, 0) + 1

    for p in hceg_scope:
        key = f"{p.get('coarse_question_type', 'unknown')}/{p.get('question_operation', 'unknown')}"
        entry = hceg_by_type.setdefault(key, {
            "count": 0,
            "candidate_count": 0,
            "candidate_correct": 0,
            "potential_gain": 0,
            "potential_loss": 0,
            "applied": 0,
            "candidate_compatible": 0,
            "role_mismatch": 0,
        })
        entry["count"] += 1
        if p.get("hceg_fallback_candidate"):
            candidate = p.get("hceg_fallback_candidate", "")
            gold = p.get("gold_answer", "")
            cand_ok = _answer_ok(candidate, gold)
            pre_answer = p.get(
                "final_answer_pre_certificate_commit",
                p.get("final_answer_pre_hceg_fallback", p.get("final_answer", "")),
            )
            pre_ok = _answer_ok(pre_answer, gold)
            entry["candidate_count"] += 1
            entry["candidate_correct"] += int(cand_ok)
            entry["potential_gain"] += int(cand_ok and not pre_ok)
            entry["potential_loss"] += int((not cand_ok) and pre_ok)
            entry["candidate_compatible"] += int(bool(p.get("hceg_fallback_candidate_compatible")))
            entry["role_mismatch"] += int(bool(p.get("hceg_fallback_role_mismatch")))
        entry["applied"] += int(bool(p.get("hceg_fallback_applied")))

    for entry in hceg_by_type.values():
        entry["candidate_hitab_em"] = _safe_rate(entry["candidate_correct"], entry["candidate_count"])
        entry["candidate_compatible_rate"] = _safe_rate(entry["candidate_compatible"], entry["candidate_count"])
        entry["role_mismatch_rate"] = _safe_rate(entry["role_mismatch"], entry["candidate_count"])

    hceg_candidate_quality = {
        "trigger_count": len(hceg_triggered),
        "candidate_count": len(hceg_candidates),
        "triggered_candidate_count": len(hceg_triggered_candidates),
        "diagnostic_candidate_count": len(hceg_diagnostic_candidates),
        "candidate_rate": _safe_rate(len(hceg_triggered_candidates), len(hceg_triggered)),
        "candidate_per_total_rate": _safe_rate(len(hceg_candidates), total),
        "candidate_correct": hceg_candidate_correct,
        "candidate_hitab_em": _safe_rate(hceg_candidate_correct, len(hceg_candidates)),
        "candidate_compatible_count": hceg_candidate_compatible,
        "candidate_compatible_rate": _safe_rate(hceg_candidate_compatible, len(hceg_candidates)),
        "role_mismatch_count": hceg_role_mismatch,
        "role_mismatch_rate": _safe_rate(hceg_role_mismatch, len(hceg_candidates)),
        "role_aware_changed": hceg_role_aware_changed,
        "role_aware_gain_over_raw": hceg_role_aware_gain_over_raw,
        "role_aware_loss_over_raw": hceg_role_aware_loss_over_raw,
        "raw_candidate_count": hceg_raw_candidate_count,
        "raw_candidate_correct": hceg_raw_candidate_correct,
        "raw_candidate_hitab_em": _safe_rate(hceg_raw_candidate_correct, hceg_raw_candidate_count),
        "potential_gain": hceg_potential_gain,
        "potential_loss": hceg_potential_loss,
        "candidate_type_distribution": hceg_type_dist,
        "expected_role_distribution": hceg_expected_role_dist,
        "by_question_type": dict(sorted(hceg_by_type.items())),
        "candidate_without_trigger_count": hceg_candidate_without_trigger,
    }

    certificate_commit_candidates = [p for p in predictions if p.get("certificate_commit_candidate")]
    certificate_commit_recommended = [
        p for p in certificate_commit_candidates
        if p.get("certificate_commit_decision") == "commit"
    ]
    certificate_commit_shadow_recommended = [
        p for p in certificate_commit_candidates
        if p.get("certificate_commit_shadow_decision") == "commit"
    ]
    certificate_commit_applied = [p for p in predictions if p.get("certificate_commit_applied")]
    certificate_operation_verified = [
        p for p in certificate_commit_candidates
        if p.get("certificate_commit_operation_verified")
    ]
    certificate_compare_direction_verified = [
        p for p in certificate_commit_candidates
        if p.get("certificate_commit_compare_direction_verified")
    ]
    certificate_numeric_direction_verified = [
        p for p in certificate_commit_candidates
        if p.get("certificate_commit_numeric_direction_verified")
    ]
    certificate_conformal_accepted = [
        p for p in certificate_commit_candidates
        if p.get("certificate_commit_conformal_accepted")
    ]
    certificate_conformal_score_pass = [
        p for p in certificate_commit_candidates
        if p.get("certificate_commit_conformal_score_pass")
        or p.get("certificate_commit_conformal_accepted")
    ]
    certificate_conformal_recommended = [
        p for p in certificate_commit_candidates
        if p.get("certificate_commit_conformal_recommended")
    ]
    certificate_conformal_shadow_accepted = [
        p for p in certificate_commit_candidates
        if p.get("certificate_commit_conformal_shadow_accepted")
    ]
    cert_reject_reasons: Dict[str, int] = {}
    cert_shadow_reject_reasons: Dict[str, int] = {}
    cert_operation_reject_reasons: Dict[str, int] = {}
    cert_compare_direction_reject_reasons: Dict[str, int] = {}
    cert_numeric_direction_reject_reasons: Dict[str, int] = {}
    cert_conformal_scores = [
        float(p.get("certificate_commit_conformal_score"))
        for p in certificate_commit_candidates
        if p.get("certificate_commit_conformal_score") is not None
    ]
    cert_by_type: Dict[str, Dict[str, Any]] = {}
    cert_rec_correct = 0
    cert_rec_gain = 0
    cert_rec_loss = 0
    cert_shadow_correct = 0
    cert_shadow_gain = 0
    cert_shadow_loss = 0
    cert_conformal_rec_correct = 0
    cert_conformal_rec_gain = 0
    cert_conformal_rec_loss = 0
    cert_conformal_shadow_correct = 0
    cert_conformal_shadow_gain = 0
    cert_conformal_shadow_loss = 0
    cert_applied_gain = 0
    cert_applied_loss = 0
    for p in certificate_commit_candidates:
        for reason in p.get("certificate_commit_reject_reasons", []) or []:
            cert_reject_reasons[reason] = cert_reject_reasons.get(reason, 0) + 1
        for reason in p.get("certificate_commit_shadow_reject_reasons", []) or []:
            cert_shadow_reject_reasons[reason] = cert_shadow_reject_reasons.get(reason, 0) + 1
        for reason in p.get("certificate_commit_operation_reject_reasons", []) or []:
            cert_operation_reject_reasons[reason] = cert_operation_reject_reasons.get(reason, 0) + 1
        for reason in p.get("certificate_commit_compare_direction_reject_reasons", []) or []:
            cert_compare_direction_reject_reasons[reason] = (
                cert_compare_direction_reject_reasons.get(reason, 0) + 1
            )
        for reason in p.get("certificate_commit_numeric_direction_reject_reasons", []) or []:
            cert_numeric_direction_reject_reasons[reason] = (
                cert_numeric_direction_reject_reasons.get(reason, 0) + 1
            )
    for p in certificate_commit_recommended:
        candidate = p.get("hceg_fallback_candidate", "")
        gold = p.get("gold_answer", "")
        cand_ok = _answer_ok(candidate, gold)
        pre_answer = p.get(
            "final_answer_pre_certificate_commit",
            p.get("final_answer_pre_hceg_fallback", p.get("final_answer", "")),
        )
        pre_ok = _answer_ok(pre_answer, gold)
        cert_rec_correct += int(cand_ok)
        cert_rec_gain += int(cand_ok and not pre_ok)
        cert_rec_loss += int((not cand_ok) and pre_ok)
    for p in certificate_commit_shadow_recommended:
        candidate = p.get("hceg_fallback_candidate", "")
        gold = p.get("gold_answer", "")
        cand_ok = _answer_ok(candidate, gold)
        pre_answer = p.get(
            "final_answer_pre_certificate_commit",
            p.get("final_answer_pre_hceg_fallback", p.get("final_answer", "")),
        )
        pre_ok = _answer_ok(pre_answer, gold)
        cert_shadow_correct += int(cand_ok)
        cert_shadow_gain += int(cand_ok and not pre_ok)
        cert_shadow_loss += int((not cand_ok) and pre_ok)
    for p in certificate_conformal_recommended:
        candidate = p.get("hceg_fallback_candidate", "")
        gold = p.get("gold_answer", "")
        cand_ok = _answer_ok(candidate, gold)
        pre_answer = p.get(
            "final_answer_pre_certificate_commit",
            p.get("final_answer_pre_hceg_fallback", p.get("final_answer", "")),
        )
        pre_ok = _answer_ok(pre_answer, gold)
        cert_conformal_rec_correct += int(cand_ok)
        cert_conformal_rec_gain += int(cand_ok and not pre_ok)
        cert_conformal_rec_loss += int((not cand_ok) and pre_ok)
    for p in certificate_conformal_shadow_accepted:
        candidate = p.get("hceg_fallback_candidate", "")
        gold = p.get("gold_answer", "")
        cand_ok = _answer_ok(candidate, gold)
        pre_answer = p.get(
            "final_answer_pre_certificate_commit",
            p.get("final_answer_pre_hceg_fallback", p.get("final_answer", "")),
        )
        pre_ok = _answer_ok(pre_answer, gold)
        cert_conformal_shadow_correct += int(cand_ok)
        cert_conformal_shadow_gain += int(cand_ok and not pre_ok)
        cert_conformal_shadow_loss += int((not cand_ok) and pre_ok)
    for p in certificate_commit_applied:
        candidate = p.get("final_answer", "")
        gold = p.get("gold_answer", "")
        cand_ok = _answer_ok(candidate, gold)
        pre_answer = p.get("final_answer_pre_certificate_commit", "")
        pre_ok = _answer_ok(pre_answer, gold)
        cert_applied_gain += int(cand_ok and not pre_ok)
        cert_applied_loss += int((not cand_ok) and pre_ok)
    for p in certificate_commit_candidates:
        key = f"{p.get('coarse_question_type', 'unknown')}/{p.get('question_operation', 'unknown')}"
        entry = cert_by_type.setdefault(key, {
            "count": 0,
            "recommended": 0,
            "applied": 0,
            "candidate_correct": 0,
            "potential_gain": 0,
            "potential_loss": 0,
            "shadow_recommended": 0,
            "shadow_candidate_correct": 0,
            "shadow_potential_gain": 0,
            "shadow_potential_loss": 0,
        })
        entry["count"] += 1
        if p.get("certificate_commit_decision") == "commit":
            candidate = p.get("hceg_fallback_candidate", "")
            gold = p.get("gold_answer", "")
            cand_ok = _answer_ok(candidate, gold)
            pre_answer = p.get(
                "final_answer_pre_certificate_commit",
                p.get("final_answer_pre_hceg_fallback", p.get("final_answer", "")),
            )
            pre_ok = _answer_ok(pre_answer, gold)
            entry["recommended"] += 1
            entry["candidate_correct"] += int(cand_ok)
            entry["potential_gain"] += int(cand_ok and not pre_ok)
            entry["potential_loss"] += int((not cand_ok) and pre_ok)
        if p.get("certificate_commit_shadow_decision") == "commit":
            candidate = p.get("hceg_fallback_candidate", "")
            gold = p.get("gold_answer", "")
            cand_ok = _answer_ok(candidate, gold)
            pre_answer = p.get(
                "final_answer_pre_certificate_commit",
                p.get("final_answer_pre_hceg_fallback", p.get("final_answer", "")),
            )
            pre_ok = _answer_ok(pre_answer, gold)
            entry["shadow_recommended"] += 1
            entry["shadow_candidate_correct"] += int(cand_ok)
            entry["shadow_potential_gain"] += int(cand_ok and not pre_ok)
            entry["shadow_potential_loss"] += int((not cand_ok) and pre_ok)
        entry["applied"] += int(bool(p.get("certificate_commit_applied")))
    for entry in cert_by_type.values():
        entry["recommended_candidate_hitab_em"] = _safe_rate(entry["candidate_correct"], entry["recommended"])
        entry["shadow_candidate_hitab_em"] = _safe_rate(
            entry["shadow_candidate_correct"],
            entry["shadow_recommended"],
        )
    certificate_commit_metrics = {
        "candidate_count": len(certificate_commit_candidates),
        "candidate_rate": _safe_rate(len(certificate_commit_candidates), total),
        "recommended_count": len(certificate_commit_recommended),
        "recommended_rate": _safe_rate(len(certificate_commit_recommended), len(certificate_commit_candidates)),
        "shadow_recommended_count": len(certificate_commit_shadow_recommended),
        "shadow_recommended_rate": _safe_rate(
            len(certificate_commit_shadow_recommended),
            len(certificate_commit_candidates),
        ),
        "operation_verified_count": len(certificate_operation_verified),
        "operation_verified_rate": _safe_rate(
            len(certificate_operation_verified),
            len(certificate_commit_candidates),
        ),
        "compare_direction_verified_count": len(certificate_compare_direction_verified),
        "compare_direction_verified_rate": _safe_rate(
            len(certificate_compare_direction_verified),
            len(certificate_commit_candidates),
        ),
        "numeric_direction_verified_count": len(certificate_numeric_direction_verified),
        "numeric_direction_verified_rate": _safe_rate(
            len(certificate_numeric_direction_verified),
            len(certificate_commit_candidates),
        ),
        "conformal_accepted_count": len(certificate_conformal_accepted),
        "conformal_accepted_rate": _safe_rate(
            len(certificate_conformal_accepted),
            len(certificate_commit_candidates),
        ),
        "conformal_score_pass_count": len(certificate_conformal_score_pass),
        "conformal_score_pass_rate": _safe_rate(
            len(certificate_conformal_score_pass),
            len(certificate_commit_candidates),
        ),
        "conformal_recommended_count": len(certificate_conformal_recommended),
        "conformal_recommended_rate": _safe_rate(
            len(certificate_conformal_recommended),
            len(certificate_commit_candidates),
        ),
        "conformal_recommended_candidate_correct": cert_conformal_rec_correct,
        "conformal_recommended_candidate_hitab_em": _safe_rate(
            cert_conformal_rec_correct,
            len(certificate_conformal_recommended),
        ),
        "conformal_recommended_potential_gain": cert_conformal_rec_gain,
        "conformal_recommended_potential_loss": cert_conformal_rec_loss,
        "conformal_shadow_accepted_count": len(certificate_conformal_shadow_accepted),
        "conformal_shadow_accepted_rate": _safe_rate(
            len(certificate_conformal_shadow_accepted),
            len(certificate_commit_candidates),
        ),
        "conformal_shadow_candidate_correct": cert_conformal_shadow_correct,
        "conformal_shadow_candidate_hitab_em": _safe_rate(
            cert_conformal_shadow_correct,
            len(certificate_conformal_shadow_accepted),
        ),
        "conformal_shadow_potential_gain": cert_conformal_shadow_gain,
        "conformal_shadow_potential_loss": cert_conformal_shadow_loss,
        "conformal_score_mean": (
            sum(cert_conformal_scores) / len(cert_conformal_scores)
            if cert_conformal_scores else 0.0
        ),
        "conformal_score_max": max(cert_conformal_scores) if cert_conformal_scores else 0.0,
        "conformal_threshold": (
            next(
                (
                    p.get("certificate_commit_conformal_threshold")
                    for p in certificate_commit_candidates
                    if p.get("certificate_commit_conformal_threshold") is not None
                ),
                None,
            )
        ),
        "applied_count": len(certificate_commit_applied),
        "applied_rate": _safe_rate(len(certificate_commit_applied), total),
        "recommended_candidate_correct": cert_rec_correct,
        "recommended_candidate_hitab_em": _safe_rate(cert_rec_correct, len(certificate_commit_recommended)),
        "recommended_potential_gain": cert_rec_gain,
        "recommended_potential_loss": cert_rec_loss,
        "shadow_candidate_correct": cert_shadow_correct,
        "shadow_candidate_hitab_em": _safe_rate(
            cert_shadow_correct,
            len(certificate_commit_shadow_recommended),
        ),
        "shadow_potential_gain": cert_shadow_gain,
        "shadow_potential_loss": cert_shadow_loss,
        "applied_potential_gain": cert_applied_gain,
        "applied_potential_loss": cert_applied_loss,
        "reject_reason_distribution": dict(sorted(cert_reject_reasons.items())),
        "shadow_reject_reason_distribution": dict(sorted(cert_shadow_reject_reasons.items())),
        "operation_reject_reason_distribution": dict(sorted(cert_operation_reject_reasons.items())),
        "compare_direction_reject_reason_distribution": dict(
            sorted(cert_compare_direction_reject_reasons.items())
        ),
        "numeric_direction_reject_reason_distribution": dict(
            sorted(cert_numeric_direction_reject_reasons.items())
        ),
        "by_question_type": dict(sorted(cert_by_type.items())),
    }

    sc_used = [p for p in predictions if p.get("self_consistency_used")]
    sc_changed = [p for p in predictions if p.get("self_consistency_changed")]
    sc_empty_vote_group = [p for p in predictions if p.get("self_consistency_empty_vote_group")]
    self_consistency_diagnostics = {
        "used": len(sc_used),
        "used_rate": _safe_rate(len(sc_used), total),
        "changed": len(sc_changed),
        "changed_rate": _safe_rate(len(sc_changed), len(sc_used)),
        "empty_vote_group": len(sc_empty_vote_group),
        "empty_vote_group_rate": _safe_rate(len(sc_empty_vote_group), len(sc_used)),
    }

    candidate_hit_ks = [1, 3, 5, 10, 20, 50]
    def _new_candidate_rank_bucket() -> Dict[str, Any]:
        return {
            "evaluated": 0,
            "total_candidates": 0,
            "mrr_sum": 0.0,
            "hit_counts": {k: 0 for k in candidate_hit_ks},
        }

    def _add_candidate_rank_row(bucket: Dict[str, Any], pred: Dict[str, Any]) -> None:
        candidates = pred.get("exec_candidates_summary") or []
        if not candidates:
            return
        bucket["evaluated"] += 1
        bucket["total_candidates"] += len(candidates)
        rank = None
        gold = pred.get("gold_answer", "")
        for idx, cand in enumerate(candidates, start=1):
            denotation = cand.get("denotation", "") if isinstance(cand, dict) else cand
            if _answer_ok(denotation, gold):
                rank = idx
                break
        if rank is not None:
            bucket["mrr_sum"] += 1.0 / rank
            for k in candidate_hit_ks:
                if rank <= k:
                    bucket["hit_counts"][k] += 1

    def _finish_candidate_rank_bucket(bucket: Dict[str, Any]) -> Dict[str, Any]:
        evaluated = int(bucket.get("evaluated", 0))
        hit_counts = bucket.get("hit_counts", {})
        return {
            "evaluated": evaluated,
            "avg_candidates": (
                float(bucket.get("total_candidates", 0)) / evaluated
                if evaluated else 0.0
            ),
            "mrr": (
                float(bucket.get("mrr_sum", 0.0)) / evaluated
                if evaluated else 0.0
            ),
            "hit_at": {
                str(k): _safe_rate(int(hit_counts.get(k, 0)), evaluated)
                for k in candidate_hit_ks
            },
        }

    candidate_overall = _new_candidate_rank_bucket()
    candidate_by_qop: Dict[str, Dict[str, Any]] = {}
    candidate_by_subtype: Dict[str, Dict[str, Any]] = {}
    for p in predictions:
        _add_candidate_rank_row(candidate_overall, p)
        qop = str(p.get("question_operation") or "unknown")
        subtype = str(p.get("dataset_question_subtype") or p.get("dataset_question_type") or "unknown")
        _add_candidate_rank_row(candidate_by_qop.setdefault(qop, _new_candidate_rank_bucket()), p)
        _add_candidate_rank_row(candidate_by_subtype.setdefault(subtype, _new_candidate_rank_bucket()), p)

    candidate_ranking_metrics = _finish_candidate_rank_bucket(candidate_overall)
    candidate_ranking_metrics["by_question_operation"] = {
        key: _finish_candidate_rank_bucket(value)
        for key, value in sorted(candidate_by_qop.items())
        if value.get("evaluated", 0)
    }
    candidate_ranking_metrics["by_dataset_question_subtype"] = {
        key: _finish_candidate_rank_bucket(value)
        for key, value in sorted(candidate_by_subtype.items())
        if value.get("evaluated", 0)
    }

    def _mean(vals: List[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    evidence_preds = [
        p for p in predictions
        if isinstance(p.get("graph_stats"), dict) and p.get("evidence_num_cells") is not None
    ]
    evidence_filtering_metrics: Dict[str, Any] = {}
    if evidence_preds:
        graph_cell_counts = []
        evidence_cell_counts = []
        filtering_rates = []
        aggregator_hits = 0
        for p in evidence_preds:
            node_types = (p.get("graph_stats") or {}).get("node_types") or {}
            graph_cells = int(node_types.get("cell", 0) or 0)
            evidence_cells = int(p.get("evidence_num_cells", 0) or 0)
            graph_cell_counts.append(graph_cells)
            evidence_cell_counts.append(evidence_cells)
            if graph_cells > 0:
                usage_rate = min(1.0, max(0.0, evidence_cells / graph_cells))
                filtering_rates.append(1.0 - usage_rate)
            aggregator_hits += int(bool(p.get("evidence_has_aggregator")))
        evidence_filtering_metrics = {
            "count": len(evidence_preds),
            "avg_graph_cells": sum(graph_cell_counts) / len(graph_cell_counts),
            "avg_evidence_cells": sum(evidence_cell_counts) / len(evidence_cell_counts),
            "avg_filtering_rate": sum(filtering_rates) / len(filtering_rates) if filtering_rates else 0.0,
            "avg_cell_usage_rate": 1.0 - (sum(filtering_rates) / len(filtering_rates) if filtering_rates else 0.0),
            "aggregator_evidence_rate": aggregator_hits / len(evidence_preds),
        }

    structural_certification_metrics: Dict[str, Any] = {}
    structural_preds = [p for p in predictions if p.get("edge_reliability_diag") or p.get("evidence_ib_mdl_diag")]
    if structural_preds:
        layout_risks = [float(p.get("layout_risk", 0.0) or 0.0) for p in structural_preds]
        ib_scores = [float(p.get("evidence_ib_mdl_score", 0.0) or 0.0) for p in structural_preds]
        mean_rels = [float((p.get("edge_reliability_diag") or {}).get("mean_edge_reliability", 0.0) or 0.0) for p in structural_preds]
        fallback_count = sum(1 for p in structural_preds if p.get("evidence_no_anchor_fallback") or p.get("evidence_num_anchors", 0) == 0)
        risk_buckets = {"low": 0, "medium": 0, "high": 0}
        risk_correct = {"low": 0, "medium": 0, "high": 0}
        for p in structural_preds:
            risk = float(p.get("layout_risk", 0.0) or 0.0)
            bucket = "low" if risk < 0.30 else ("medium" if risk < 0.60 else "high")
            risk_buckets[bucket] += 1
            risk_correct[bucket] += int(_primary_correct(p))
        cert_rows = [p.get("certificate_info", {}) for p in predictions if isinstance(p.get("certificate_info"), dict)]
        candidate_details = []
        scci_modes: Dict[str, int] = {}
        for ci in cert_rows:
            mode = str(ci.get("scci_mode", "unknown"))
            scci_modes[mode] = scci_modes.get(mode, 0) + 1
            rows = ci.get("candidate_details") or []
            if isinstance(rows, list):
                candidate_details.extend(r for r in rows if isinstance(r, dict))
        candidate_coverages = [float(r.get("candidate_evidence_coverage", 0.0) or 0.0) for r in candidate_details]
        effective_coverages = [float(r.get("candidate_effective_evidence_coverage", r.get("candidate_evidence_coverage", 0.0)) or 0.0) for r in candidate_details]
        candidate_scci = [float(r.get("scci", 0.0) or 0.0) for r in candidate_details]
        candidate_interventions = []
        for ci in cert_rows:
            vals = ci.get("candidate_intervention_counts") or []
            if isinstance(vals, list):
                candidate_interventions.extend(float(v or 0.0) for v in vals)
        structural_certification_metrics = {
            "count": len(structural_preds),
            "structural_prior_weighting_count": sum(1 for p in structural_preds if p.get("structural_prior_weighting")),
            "avg_layout_risk": _mean(layout_risks),
            "avg_edge_reliability": _mean(mean_rels),
            "avg_evidence_ib_mdl_score": _mean(ib_scores),
            "no_anchor_fallback_rate": fallback_count / len(structural_preds),
            "layout_risk_bucket_counts": risk_buckets,
            "layout_risk_bucket_primary_em": {k: _safe_rate(risk_correct.get(k, 0), risk_buckets.get(k, 0)) for k in risk_buckets},
            "scci_mode_distribution": scci_modes,
            "avg_candidate_evidence_coverage": _mean(candidate_coverages),
            "avg_candidate_effective_evidence_coverage": _mean(effective_coverages),
            "avg_candidate_scci": _mean(candidate_scci),
            "avg_candidate_targeted_interventions": _mean(candidate_interventions),
        }

    operation_support_cell_usage_metrics: Dict[str, Any] = {}
    support_usage_preds = [
        p for p in predictions
        if isinstance(p.get("graph_stats"), dict)
        and ((p.get("operation_support_cell_count") is not None) or (p.get("operation_support_target_cell_count") is not None))
    ]
    if support_usage_preds:
        support_counts: List[float] = []
        target_counts: List[float] = []
        support_usage_rates: List[float] = []
        target_usage_rates: List[float] = []
        applied_support_rates: List[float] = []
        for p in support_usage_preds:
            node_types = (p.get("graph_stats") or {}).get("node_types") or {}
            graph_cells = int(node_types.get("cell", 0) or 0)
            support_count = int(
                p.get("operation_support_cell_count")
                or (
                    int(p.get("operation_support_target_cell_count", 0) or 0)
                    + int(p.get("operation_support_filter_cell_count", 0) or 0)
                )
            )
            target_count = int(p.get("operation_support_target_cell_count", 0) or 0)
            support_counts.append(float(support_count))
            target_counts.append(float(target_count))
            if graph_cells > 0:
                support_rate = min(1.0, max(0.0, support_count / graph_cells))
                target_rate = min(1.0, max(0.0, target_count / graph_cells))
                support_usage_rates.append(support_rate)
                target_usage_rates.append(target_rate)
                if p.get("operation_support_commit_applied"):
                    applied_support_rates.append(support_rate)
        operation_support_cell_usage_metrics = {
            "count": len(support_usage_preds),
            "avg_operation_support_cells": _mean(support_counts),
            "avg_operation_target_cells": _mean(target_counts),
            "avg_operation_support_cell_usage_rate": _mean(support_usage_rates),
            "avg_operation_target_cell_usage_rate": _mean(target_usage_rates),
            "avg_applied_operation_support_cell_usage_rate": _mean(applied_support_rates),
        }

    trajectory_preds = [
        p for p in predictions
        if isinstance(p.get("entropy_trajectory_diagnostics"), dict)
    ]
    entropy_trajectory_metrics: Dict[str, Any] = {}
    if trajectory_preds:
        fields = [
            "length",
            "mean_entropy",
            "max_entropy",
            "tail_mean_entropy",
            "hep_count",
            "hep_total_mass",
            "entropy_centroid",
            "late_hep_mass",
            "tail_entropy_phase_ratio",
        ]

        def _traj_avg(preds: List[Dict[str, Any]], field: str) -> float:
            vals = [
                float((p.get("entropy_trajectory_diagnostics") or {}).get(field, 0.0) or 0.0)
                for p in preds
            ]
            return _mean(vals)

        correct_traj = [p for p in trajectory_preds if _primary_correct(p)]
        wrong_traj = [p for p in trajectory_preds if not _primary_correct(p)]
        entropy_trajectory_metrics = {
            "count": len(trajectory_preds),
            "overall": {field: _traj_avg(trajectory_preds, field) for field in fields},
            "correct": {field: _traj_avg(correct_traj, field) for field in fields},
            "wrong": {field: _traj_avg(wrong_traj, field) for field in fields},
        }

    time_preds = [p for p in predictions if p.get("llm_generation_seconds") is not None]
    correct_preds = [p for p in predictions if _primary_correct(p)]
    wrong_preds = [p for p in predictions if not _primary_correct(p)]

    refusal_pattern = re.compile(
        r"(?:\b(?:sorry|cannot|can't|unable|not\s+able)\b|抱歉|无法|不能|不给|不提供)",
        re.IGNORECASE,
    )

    def _refusal_like_answer(pred: Dict[str, Any]) -> bool:
        text = " ".join(str(pred.get(k, "") or "") for k in ("final_answer", "llm_answer", "llm_raw_output"))
        return bool(refusal_pattern.search(text))

    answer_quality_metrics = {
        "empty_final_answer_count": sum(1 for p in predictions if not str(p.get("final_answer", "") or "").strip()),
        "missing_answer_source_count": sum(1 for p in predictions if not str(p.get("answer_source", "") or "").strip()),
        "refusal_like_answer_count": sum(1 for p in predictions if _refusal_like_answer(p)),
        "api_usage_missing_count": sum(
            1 for p in predictions
            if p.get("black_box_api_generator") and not isinstance(p.get("api_usage"), dict)
        ),
    }

    def _avg_field(preds: List[Dict[str, Any]], field: str) -> float:
        vals = [float(p.get(field, 0.0) or 0.0) for p in preds if p.get(field) is not None]
        return _mean(vals)

    efficiency_metrics = {
        "avg_llm_generation_seconds": _avg_field(time_preds, "llm_generation_seconds"),
        "avg_generated_tokens": _avg_field(predictions, "generated_token_count"),
        "avg_generated_tokens_correct": _avg_field(correct_preds, "generated_token_count"),
        "avg_generated_tokens_wrong": _avg_field(wrong_preds, "generated_token_count"),
        "avg_prompt_length": _avg_field(predictions, "prompt_length"),
        "avg_prompt_length_correct": _avg_field(correct_preds, "prompt_length"),
        "avg_prompt_length_wrong": _avg_field(wrong_preds, "prompt_length"),
        "avg_final_answer_chars_correct": _mean([len(str(p.get("final_answer", ""))) for p in correct_preds]),
        "avg_final_answer_chars_wrong": _mean([len(str(p.get("final_answer", ""))) for p in wrong_preds]),
    }

    api_preds = [p for p in predictions if p.get("black_box_api_generator")]
    external_generator_metrics: Dict[str, Any] = {}
    if api_preds:
        usage_rows = [p.get("api_usage", {}) for p in api_preds if isinstance(p.get("api_usage", {}), dict)]
        prompt_tokens = [float(u.get("prompt_tokens", 0) or 0) for u in usage_rows]
        completion_tokens = [float(u.get("completion_tokens", 0) or 0) for u in usage_rows]
        total_tokens = [float(u.get("total_tokens", 0) or 0) for u in usage_rows]
        model_dist: Dict[str, int] = {}
        backend_dist: Dict[str, int] = {}
        cache_mode_dist: Dict[str, int] = {}
        for p in api_preds:
            model_key = str(p.get("api_model") or "unknown")
            backend_key = str(p.get("generator_backend") or "unknown")
            cache_mode_key = str(p.get("api_cache_mode") or "unknown")
            model_dist[model_key] = model_dist.get(model_key, 0) + 1
            backend_dist[backend_key] = backend_dist.get(backend_key, 0) + 1
            cache_mode_dist[cache_mode_key] = cache_mode_dist.get(cache_mode_key, 0) + 1
        api_cache_hits = sum(1 for p in api_preds if p.get("api_cache_hit"))
        external_generator_metrics = {
            "enabled_count": len(api_preds),
            "backend_distribution": backend_dist,
            "api_model_distribution": model_dist,
            "api_cache_mode_distribution": cache_mode_dist,
            "logprobs_available_count": sum(1 for p in api_preds if p.get("llm_logprobs_available")),
            "logprobs_unavailable_count": sum(1 for p in api_preds if p.get("api_logprobs_unavailable")),
            "cache_hit_count": api_cache_hits,
            "cache_miss_count": len(api_preds) - api_cache_hits,
            "cache_hit_rate": api_cache_hits / len(api_preds) if api_preds else 0.0,
            "avg_prompt_tokens": _mean(prompt_tokens),
            "avg_completion_tokens": _mean(completion_tokens),
            "avg_total_tokens": _mean(total_tokens),
            "total_prompt_tokens": sum(prompt_tokens),
            "total_completion_tokens": sum(completion_tokens),
            "total_tokens": sum(total_tokens),
        }

    context_preds = [p for p in predictions if p.get("input_token_count") is not None]
    context_reachability_metrics: Dict[str, Any] = {}
    if context_preds:
        input_lengths = [float(p.get("input_token_count", 0.0) or 0.0) for p in context_preds]
        pressures = [float(p.get("context_pressure_ratio", 0.0) or 0.0) for p in context_preds]
        tier_counts: Dict[str, int] = {}
        for p in context_preds:
            tier = p.get("compute_pressure_tier", "unknown")
            tier_counts[tier] = tier_counts.get(tier, 0) + 1

        apr_r2 = [p for p in predictions if p.get("apr_round") == 2 or p.get("apr_round2_input_token_count") is not None]
        apr_r2_pressures = [
            float(p.get("apr_round2_context_pressure_ratio", 0.0) or 0.0)
            for p in apr_r2
            if p.get("apr_round2_context_pressure_ratio") is not None
        ]
        prefix_reuse = [
            float(p.get("apr_prefix_reuse_ratio_char", 0.0) or 0.0)
            for p in predictions
            if p.get("apr_prefix_reuse_ratio_char") is not None
        ]

        sc_audits = []
        for p in predictions:
            audits = p.get("self_consistency_prompt_audit") or []
            if isinstance(audits, list):
                sc_audits.extend(a for a in audits if isinstance(a, dict))
        sc_pressures = [
            float(a.get("context_pressure_ratio", 0.0) or 0.0)
            for a in sc_audits
            if a.get("context_pressure_ratio") is not None
        ]
        context_reachability_metrics = {
            "count": len(context_preds),
            "avg_input_tokens": _mean(input_lengths),
            "max_input_tokens": max(input_lengths) if input_lengths else 0.0,
            "avg_context_pressure": _mean(pressures),
            "max_context_pressure": max(pressures) if pressures else 0.0,
            "pressure_tier_distribution": tier_counts,
            "apr_round2_count": len(apr_r2),
            "apr_round2_skipped_no_truncation": sum(1 for p in predictions if p.get("apr_round2_skipped_no_truncation")),
            "avg_apr_round2_context_pressure": _mean(apr_r2_pressures),
            "avg_apr_prefix_reuse_ratio_char": _mean(prefix_reuse),
            "self_consistency_prompt_count": len(sc_audits),
            "self_consistency_skipped_no_truncation": sum(1 for p in predictions if p.get("self_consistency_skipped_no_truncation")),
            "avg_self_consistency_context_pressure": _mean(sc_pressures),
        }

    repair_preds = [p for p in predictions if p.get("cera_enabled")]
    repair_metrics: Dict[str, Any] = {}
    if repair_preds:
        reject_dist: Dict[str, int] = {}
        validator_reject_dist: Dict[str, int] = {}
        stage_dist: Dict[str, int] = {}
        packet_token_lengths = [
            int(p.get("cera_packet_token_length", 0) or 0)
            for p in repair_preds
            if int(p.get("cera_packet_token_length", 0) or 0) > 0
        ]
        def _summary(vals: List[int]) -> Dict[str, float]:
            if not vals:
                return {"count": 0, "mean": 0.0, "min": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
            vals_sorted = sorted(vals)
            def _percentile(pct: float) -> float:
                idx = min(len(vals_sorted) - 1, max(0, int(round((len(vals_sorted) - 1) * pct))))
                return float(vals_sorted[idx])
            return {
                "count": len(vals_sorted),
                "mean": _mean([float(v) for v in vals_sorted]),
                "min": float(vals_sorted[0]),
                "p50": _percentile(0.50),
                "p95": _percentile(0.95),
                "max": float(vals_sorted[-1]),
            }
        def _has_operation_multiplicity(pred: Mapping[str, Any]) -> bool:
            cert_info = pred.get("certificate_info")
            if not isinstance(cert_info, Mapping):
                return False
            rows = cert_info.get("certified_candidates_full") or []
            operations = {
                str(row.get("operation"))
                for row in rows
                if isinstance(row, Mapping) and str(row.get("operation", "")).strip()
            }
            return len(operations) > 1
        def _answer_disagrees(pred: Mapping[str, Any]) -> bool:
            original = pred.get("cera_original_answer", pred.get("llm_answer", ""))
            candidate = pred.get("cera_candidate_under_review", "")
            return bool(candidate) and str(original).strip().lower() != str(candidate).strip().lower()
        for pred in repair_preds:
            reason = str(pred.get("cera_reject_reason") or "")
            if reason:
                reject_dist[reason] = reject_dist.get(reason, 0) + 1
            validator_reason = str(pred.get("cera_validator_reject_reason") or "")
            if validator_reason:
                validator_reject_dist[validator_reason] = validator_reject_dist.get(validator_reason, 0) + 1
            stage = str(pred.get("cera_stage") or "unknown")
            stage_dist[stage] = stage_dist.get(stage, 0) + 1
        packet_built_count = sum(1 for p in repair_preds if p.get("cera_packet_built"))
        opportunity_count = sum(1 for p in repair_preds if p.get("cera_triggered"))
        support_available_count = sum(1 for p in repair_preds if int(p.get("cera_support_chain_len", 0) or 0) > 0)
        original_support_available_count = sum(1 for p in repair_preds if p.get("cera_original_certificate_available"))
        candidate_cf_available_count = sum(1 for p in repair_preds if int(p.get("cera_candidate_counterfactual_available_count", 0) or 0) > 0)
        observed_cf_count = sum(1 for p in repair_preds if int(p.get("cera_candidate_observed_counterfactual_count", 0) or 0) > 0)
        fallback_blocked_count = sum(1 for p in repair_preds if p.get("cera_reject_reason") == "evidence_fallback")
        outside_support_count = sum(1 for p in repair_preds if int(p.get("cera_outside_evidence_support_count", 0) or 0) > 0)
        row_major_context_cell_count = sum(int(p.get("cera_row_major_context_cell_count", 0) or 0) for p in repair_preds)
        row_major_context_packet_count = sum(1 for p in repair_preds if int(p.get("cera_row_major_context_cell_count", 0) or 0) > 0)
        pre_evidence_contract_count = sum(1 for p in repair_preds if p.get("cera_query_contract_pre_evidence"))
        admissible_derivation_count = sum(int(p.get("cera_admissible_derivation_count", 0) or 0) for p in repair_preds)
        derivation_available_count = sum(1 for p in repair_preds if int(p.get("cera_admissible_derivation_count", 0) or 0) > 0)
        unique_admissible_class_count = sum(1 for p in repair_preds if int(p.get("cera_projected_answer_class_count", 0) or 0) == 1)
        ambiguous_admissible_class_count = sum(1 for p in repair_preds if int(p.get("cera_projected_answer_class_count", 0) or 0) > 1)
        original_support_hypothesis_count = sum(int(p.get("cera_original_support_hypothesis_count", 0) or 0) for p in repair_preds)
        original_support_executable_count = sum(1 for p in repair_preds if p.get("cera_original_support_executable"))
        lattice_member_count = sum(int(p.get("cera_lattice_member_count", 0) or 0) for p in repair_preds)
        lattice_l1_count = sum(int(p.get("cera_lattice_l1_roundtrip_valid_count", 0) or 0) for p in repair_preds)
        lattice_l4_count = sum(int(p.get("cera_lattice_l4_evidence_grounded_count", 0) or 0) for p in repair_preds)
        lattice_l6_count = sum(int(p.get("cera_lattice_l6_quotient_class_count", 0) or 0) for p in repair_preds)
        lattice_answer_class_count = sum(int(p.get("cera_lattice_answer_class_count", 0) or 0) for p in repair_preds)
        lattice_mismatch_count = sum(int(p.get("cera_lattice_candidate_observation_mismatch_count", 0) or 0) for p in repair_preds)
        lattice_compression_values = [
            float(p.get("cera_lattice_compression_ratio", 0.0) or 0.0)
            for p in repair_preds
            if float(p.get("cera_lattice_compression_ratio", 0.0) or 0.0) > 0.0
        ]
        contrast_ready_count = sum(1 for p in repair_preds if p.get("cera_contrast_ready"))
        contrast_alt_class_count = sum(int(p.get("cera_contrast_alternative_class_count", 0) or 0) for p in repair_preds)
        contrast_alt_answer_class_count = sum(int(p.get("cera_contrast_alternative_answer_class_count", 0) or 0) for p in repair_preds)
        contrast_unresolved_count = sum(int(p.get("cera_contrast_unresolved_ambiguity_count", 0) or 0) for p in repair_preds)
        original_support_v3_hypothesis_count = sum(int(p.get("cera_original_support_v3_hypothesis_count", 0) or 0) for p in repair_preds)
        original_support_v3_roundtrip_count = sum(1 for p in repair_preds if p.get("cera_original_support_v3_roundtrip_executable"))
        original_support_v3_graph_anchor_count = sum(1 for p in repair_preds if p.get("cera_original_support_v3_graph_anchor_only"))
        legacy_dependency_count = sum(1 for p in repair_preds if int(p.get("legacy_heuristic_usage_count", 0) or 0) > 0)
        operation_multiplicity_count = sum(1 for p in repair_preds if _has_operation_multiplicity(p))
        answer_disagreement_count = sum(1 for p in repair_preds if _answer_disagrees(p))
        posthoc_repair_metrics = aggregate_repair_outcomes(repair_preds, dataset=dataset)
        repair_metrics = {
            "enabled_count": len(repair_preds),
            "packet_built_count": packet_built_count,
            "packet_built_rate": _safe_rate(packet_built_count, len(repair_preds)),
            "opportunity_count": opportunity_count,
            "review_opportunity_rate": _safe_rate(opportunity_count, len(repair_preds)),
            "support_chain_available_count": support_available_count,
            "support_chain_available_rate": _safe_rate(support_available_count, len(repair_preds)),
            "original_support_available_count": original_support_available_count,
            "original_support_available_rate": _safe_rate(original_support_available_count, len(repair_preds)),
            "counterfactual_chain_available_count": sum(1 for p in repair_preds if int(p.get("cera_counterfactual_chain_len", 0) or 0) > 0),
            "candidate_specific_counterfactual_available_count": candidate_cf_available_count,
            "candidate_specific_counterfactual_available_rate": _safe_rate(candidate_cf_available_count, len(repair_preds)),
            "observed_counterfactual_count": observed_cf_count,
            "observed_counterfactual_rate": _safe_rate(observed_cf_count, len(repair_preds)),
            "fallback_blocked_count": fallback_blocked_count,
            "fallback_blocked_rate": _safe_rate(fallback_blocked_count, len(repair_preds)),
            "outside_evidence_support_count": outside_support_count,
            "outside_evidence_support_rate": _safe_rate(outside_support_count, len(repair_preds)),
            "row_major_context_cell_count": row_major_context_cell_count,
            "row_major_context_packet_count": row_major_context_packet_count,
            "row_major_context_packet_rate": _safe_rate(row_major_context_packet_count, len(repair_preds)),
            "pre_evidence_query_contract_count": pre_evidence_contract_count,
            "pre_evidence_query_contract_rate": _safe_rate(pre_evidence_contract_count, len(repair_preds)),
            "admissible_derivation_count": admissible_derivation_count,
            "derivation_available_count": derivation_available_count,
            "derivation_available_rate": _safe_rate(derivation_available_count, len(repair_preds)),
            "unique_admissible_projected_answer_class_count": unique_admissible_class_count,
            "unique_admissible_projected_answer_class_rate": _safe_rate(unique_admissible_class_count, len(repair_preds)),
            "ambiguous_admissible_projected_answer_class_count": ambiguous_admissible_class_count,
            "ambiguous_admissible_projected_answer_class_rate": _safe_rate(ambiguous_admissible_class_count, len(repair_preds)),
            "original_support_hypothesis_count": original_support_hypothesis_count,
            "original_support_executable_count": original_support_executable_count,
            "original_support_executable_rate": _safe_rate(original_support_executable_count, len(repair_preds)),
            "round6_lattice_member_count": lattice_member_count,
            "round6_lattice_l1_roundtrip_valid_count": lattice_l1_count,
            "round6_lattice_l1_roundtrip_valid_rate_over_members": _safe_rate(lattice_l1_count, lattice_member_count),
            "round6_lattice_l4_evidence_grounded_count": lattice_l4_count,
            "round6_lattice_l4_evidence_grounded_rate_over_members": _safe_rate(lattice_l4_count, lattice_member_count),
            "round6_lattice_l6_quotient_class_count": lattice_l6_count,
            "round6_lattice_answer_class_count": lattice_answer_class_count,
            "round6_lattice_candidate_observation_mismatch_count": lattice_mismatch_count,
            "round6_lattice_avg_compression_ratio": _mean(lattice_compression_values),
            "round6_contrast_ready_count": contrast_ready_count,
            "round6_contrast_ready_rate": _safe_rate(contrast_ready_count, len(repair_preds)),
            "round6_contrast_alternative_class_count": contrast_alt_class_count,
            "round6_contrast_alternative_answer_class_count": contrast_alt_answer_class_count,
            "round6_contrast_unresolved_ambiguity_count": contrast_unresolved_count,
            "round6_original_support_v3_hypothesis_count": original_support_v3_hypothesis_count,
            "round6_original_support_v3_roundtrip_executable_count": original_support_v3_roundtrip_count,
            "round6_original_support_v3_roundtrip_executable_rate": _safe_rate(original_support_v3_roundtrip_count, len(repair_preds)),
            "round6_original_support_v3_graph_anchor_only_count": original_support_v3_graph_anchor_count,
            "packet_token_length_distribution": _summary(packet_token_lengths),
            "legacy_heuristic_dependency_count": legacy_dependency_count,
            "legacy_heuristic_dependency_rate": _safe_rate(legacy_dependency_count, len(repair_preds)),
            "operation_family_multiplicity_count": operation_multiplicity_count,
            "operation_family_multiplicity_rate": _safe_rate(operation_multiplicity_count, len(repair_preds)),
            "answer_disagreement_count": answer_disagreement_count,
            "answer_disagreement_rate": _safe_rate(answer_disagreement_count, len(repair_preds)),
            "no_rescue_candidate_count": sum(1 for p in repair_preds if p.get("cera_reject_reason") == "no_rescue_candidate"),
            "no_admissible_rescue_candidate_count": sum(1 for p in repair_preds if p.get("cera_reject_reason") == "no_admissible_rescue_candidate"),
            "no_admissible_projected_answer_class_count": sum(1 for p in repair_preds if p.get("cera_reject_reason") == "no_admissible_projected_answer_class"),
            "ambiguous_admissible_projected_answer_classes_count": sum(1 for p in repair_preds if p.get("cera_reject_reason") == "ambiguous_admissible_projected_answer_classes"),
            "cera_llm_called_count": sum(1 for p in repair_preds if p.get("cera_llm_called")),
            "json_parse_success_count": sum(1 for p in repair_preds if p.get("cera_json_parse_success")),
            "validator_accept_count": sum(1 for p in repair_preds if p.get("cera_validator_accept")),
            "would_commit_count": sum(1 for p in repair_preds if p.get("cera_would_commit")),
            "would_keep_count": sum(1 for p in repair_preds if p.get("cera_would_keep")),
            "insufficient_count": sum(1 for p in repair_preds if p.get("cera_insufficient")),
            "unsafe_accept_count": sum(1 for p in repair_preds if p.get("cera_unsafe_accept")),
            "reject_reason_distribution": dict(sorted(reject_dist.items())),
            "validator_reject_reason_distribution": dict(sorted(validator_reject_dist.items())),
            "stage_distribution": dict(sorted(stage_dist.items())),
        }
        repair_metrics.update(posthoc_repair_metrics)

    main_cert_profile_audit = {
        "enabled": bool(getattr(args, "main_cert_profile", False)),
        "operation_commit_version": _normalise_operation_commit_version(
            getattr(args, "operation_commit_version", "E67")
        ),
        "operation_commit_gate_mode": getattr(args, "operation_commit_gate_mode", "diagnostic"),
        "operation_commit_gate_diagnostics": bool(
            getattr(args, "operation_commit_gate_diagnostics", False)
        ),
        "surface_heuristic_mode": getattr(args, "surface_heuristic_mode", "diagnostic"),
        "structural_certificate_gate_enabled": (
            bool(getattr(args, "operation_commit_gate_diagnostics", False))
            and _normalise_operation_commit_version(
                getattr(args, "operation_commit_version", "E67")
            ) in {"E65.4", "E67"}
        ),
        "legacy_commit_paths_disabled": not (
            getattr(args, "hceg_fallback", False)
            or getattr(args, "certificate_commit_boundary", False)
            or getattr(args, "self_consistency", False)
        ),
        "prompt_routing_disabled": not (
            getattr(args, "adaptive_prompt", False)
            or getattr(args, "question_type_router", False)
        ),
        "answer_normalizers_disabled": not (
            getattr(args, "online_normalizer", False)
            or getattr(args, "oracle_online_normalizer", False)
            or getattr(args, "api_format_normalizer", "auto") != "off"
        ),
        "source_risk_calibration_disabled": getattr(args, "source_risk_calibration", "auto") == "off",
        "credal_probe_disabled": not getattr(args, "credal_probe", False),
        "black_box_policy_certificate_only": getattr(args, "black_box_commit_policy", "auto") in {"off", "certified"},
        "heuristic_surface_used_for_commit_count": sum(
            1 for p in predictions if p.get("heuristic_surface_used_for_commit")
        ),
        "legacy_commit_path_used_count": sum(
            1 for p in predictions if p.get("legacy_commit_path_used")
        ),
        "non_certificate_answer_mutation_count": sum(
            1 for p in predictions if p.get("non_certificate_answer_mutation_used")
        ),
        "boolean_conjunction_commit_count": sum(
            1 for p in predictions if p.get("commit_decision_is_boolean_conjunction")
        ),
    }

    metrics = {
        "total": total,
        "dataset": dataset,
        "primary_metric_name": primary_metric_name,
        "primary_em": em_rates.get(primary_metric_name, 0.0),
        "em_counts": em_counts,
        "em_rates": em_rates,
        "dataset_slices": dataset_slices,
        "answer_source_distribution": source_dist,
        "source_em": source_em,
        "executor_stats": executor_stats,
        "operation_em": operation_em,
        "operation_support_metrics": operation_support_metrics,
        "operation_role_target_metrics": operation_role_target_metrics,
        "operation_commit_gate_metrics": operation_commit_gate_metrics,
        "operation_commit_certificate_metrics": operation_commit_certificate_metrics,
        "conformal_abstain_by_operation": conformal_abstain_by_operation,
        "calibration": cal_metrics,
        "error_distribution": error_dist,
        "entropy_stats": entropy_stats,
        "entropy_buckets": entropy_buckets,
        "prompt_length_stats": prompt_length_stats,
        "probe_metrics": probe_metrics,
        "module_diagnostics": module_diagnostics,
        "hceg_candidate_quality": hceg_candidate_quality,
        "certificate_commit_metrics": certificate_commit_metrics,
        "self_consistency_diagnostics": self_consistency_diagnostics,
        "candidate_ranking_metrics": candidate_ranking_metrics,
        "evidence_filtering_metrics": evidence_filtering_metrics,
        "structural_certification_metrics": structural_certification_metrics,
        "operation_support_cell_usage_metrics": operation_support_cell_usage_metrics,
        "entropy_trajectory_metrics": entropy_trajectory_metrics,
        "efficiency_metrics": efficiency_metrics,
        "answer_quality_metrics": answer_quality_metrics,
        "context_reachability_metrics": context_reachability_metrics,
        "external_generator_metrics": external_generator_metrics,
        "repair_metrics": repair_metrics,
        "main_cert_profile_audit": main_cert_profile_audit,
    }

    # 打印摘要
    logger.info("=" * 60)
    logger.info("CSCR Pipeline Results Summary")
    logger.info("=" * 60)
    logger.info(f"Total samples: {total}")
    for c in calibers:
        logger.info(f"  {c}: {em_counts[c]}/{total} = {em_rates[c]*100:.2f}%")
    logger.info(f"Answer sources: {source_dist}")
    if executor_stats:
        logger.info(f"Executor valid rate: {executor_stats.get('valid_rate', 0)*100:.1f}%")
    if operation_em:
        logger.info(f"Operation EM: {operation_em}")
    if operation_support_metrics:
        logger.info(f"Operation support metrics: {operation_support_metrics}")
    if operation_role_target_metrics:
        logger.info(f"Operation role-target metrics: {operation_role_target_metrics}")
    if operation_commit_gate_metrics:
        logger.info(f"Operation commit gate metrics: {operation_commit_gate_metrics}")
    if operation_commit_certificate_metrics:
        logger.info(f"Operation commit certificate metrics: {operation_commit_certificate_metrics}")
    if external_generator_metrics:
        logger.info(f"External generator metrics: {external_generator_metrics}")
    if answer_quality_metrics:
        logger.info(f"Answer quality metrics: {answer_quality_metrics}")
    if prompt_length_stats:
        logger.info(f"Prompt length stats: {prompt_length_stats}")
    logger.info(f"Candidate ranking metrics: {candidate_ranking_metrics}")
    if evidence_filtering_metrics:
        logger.info(f"Evidence filtering metrics: {evidence_filtering_metrics}")
    if structural_certification_metrics:
        logger.info(f"Structural certification metrics: {structural_certification_metrics}")
    if entropy_trajectory_metrics:
        logger.info(f"Entropy trajectory metrics: {entropy_trajectory_metrics}")
    if efficiency_metrics:
        logger.info(f"Efficiency metrics: {efficiency_metrics}")
    if context_reachability_metrics:
        logger.info(f"Context reachability metrics: {context_reachability_metrics}")
    if repair_metrics:
        logger.info(f"CERA repair metrics: {repair_metrics}")
    if module_diagnostics:
        logger.info(f"Module diagnostics: {module_diagnostics}")
    if hceg_candidate_quality.get("trigger_count", 0) or hceg_candidate_quality.get("candidate_count", 0):
        logger.info(f"HCEG candidate quality: {hceg_candidate_quality}")
    if certificate_commit_metrics.get("candidate_count", 0):
        logger.info(f"Certificate commit metrics: {certificate_commit_metrics}")
    if self_consistency_diagnostics.get("used", 0):
        logger.info(f"Self-consistency diagnostics: {self_consistency_diagnostics}")
    if operation_support_cell_usage_metrics.get("count", 0):
        logger.info(f"Operation support cell usage: {operation_support_cell_usage_metrics}")
    logger.info(f"Calibration ECE: {cal_metrics.get('ece', -1):.4f}")
    logger.info(f"Calibration Brier: {cal_metrics.get('brier', -1):.4f}")
    logger.info("=" * 60)

    return metrics


# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CSCR Pipeline - 统一实验管线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
实验模式:
  baseline_a_plus   结构感知 Prompt + logit 熵校准 (无执行器/图)
  executor_only     结构感知 Prompt + 执行器验证
  full              完整 CSCR 管线 (HCEG + 检索 + 执行器 + LLM)
  recalculate       从已有预测重新计算四口径 EM

示例:
  python run_cscr_pipeline.py --mode baseline_a_plus --model_path /path/to/qwen \\
      --output_dir outputs/cscr/baseline_a_plus

  python run_cscr_pipeline.py --mode recalculate \\
      --recalculate_from outputs/ca2kg_tables/.../predictions.jsonl \\
      --output_dir outputs/cscr/recalculated

  # 多卡运行 Qwen2.5-32B（4卡）
  python run_cscr_pipeline.py --mode full_cert \\
      --model_path /data/hesihao/llm/Qwen/Qwen2.5-32B-Instruct \\
      --tensor_parallel_size 4 --dtype bfloat16 --gpu-memory-utilization 0.92 \\
      --batch-inference --batch_size 128 \\
      --prompt-style selective_evidence \\
      --success-predictor-model outputs/cscr/success_predictor_v2.pt \\
      --output_dir outputs/cscr/full_cert_32b
""",
    )

    # 数据参数
    parser.add_argument(
        "--dataset",
        default="hitab",
        choices=["hitab", "aitqa", "ait-qa", "hi-tab", "sstqa_zh", "sstqa-zh", "sstqazh", "tablebench", "table-bench"],
    )
    parser.add_argument("--input_file", default="/data/hesihao/HiTab/data/test_samples.jsonl")
    parser.add_argument("--table_dir", default="/data/hesihao/HiTab/data/tables/raw")
    parser.add_argument("--output_dir", required=True)

    # 模式
    parser.add_argument("--mode", choices=["baseline_a_plus", "executor_only", "full", "full_cert", "recalculate"],
                        default="baseline_a_plus")

    # 模型参数
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--generator-backend",
                        choices=["vllm", "openai_chat", "gemini_chat", "vllm_chat"],
                        default="vllm",
                        dest="generator_backend",
                        help="Language generator backend. API backends only replace generation and keep local CSCR evidence/certificate modules unchanged.")
    parser.add_argument("--api-base-url",
                        default="https://api.lkeap.cloud.tencent.com/v1",
                        dest="api_base_url",
                        help="OpenAI-compatible API base URL for --generator-backend=openai_chat")
    parser.add_argument("--api-key-env",
                        default="LKEAP_API_KEY",
                        dest="api_key_env",
                        help="Environment variable that stores the API key")
    parser.add_argument("--api-model",
                        default=None,
                        dest="api_model",
                        help="Remote chat model name; defaults to --model_path when omitted")
    parser.add_argument("--api-timeout", type=float, default=120.0,
                        dest="api_timeout",
                        help="API request timeout in seconds")
    parser.add_argument("--api-max-retries", type=int, default=3,
                        dest="api_max_retries",
                        help="OpenAI SDK max retries")
    parser.add_argument("--api-rate-limit-seconds", type=float, default=0.0,
                        dest="api_rate_limit_seconds",
                        help="Optional sleep between API requests in seconds")
    parser.add_argument("--api-cache-path",
                        default="",
                        dest="api_cache_path",
                        help="Optional JSONL cache for OpenAI-compatible API responses, keyed by model/prompt/sampling parameters")
    parser.add_argument("--api-cache-mode",
                        choices=["off", "readwrite", "readonly", "require"],
                        default="readwrite",
                        dest="api_cache_mode",
                        help="API cache mode. require fails on cache miss and is recommended for final replay audits.")
    parser.add_argument("--save-llm-inputs",
                        choices=["off", "hash", "full"],
                        default="off",
                        dest="save_llm_inputs",
                        help="Audit exact LLM inputs. off=disabled; hash=hash/length only; full=write full messages and rendered prompt to sidecar JSONL.")
    parser.add_argument("--llm-input-audit-file",
                        default="llm_input_audit.jsonl",
                        dest="llm_input_audit_file",
                        help="Sidecar JSONL path for LLM input audit. Relative paths are resolved under --output_dir.")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="vLLM Tensor Parallel 维度，等于使用的 GPU 数量")
    parser.add_argument("--max_model_len", type=int, default=16384)

    # v9.0: 大模型 / 多卡新增参数
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["bfloat16", "float16", "float32", "auto"],
                        help="推理数据类型 (默认: bfloat16，大模型推荐)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90,
                        dest="gpu_memory_utilization",
                        help="vLLM 显存利用率 0.0~1.0 (默认: 0.90，大模型建议 0.92~0.95)")
    parser.add_argument("--max-num-seqs", type=int, default=256,
                        dest="max_num_seqs",
                        help="vLLM 最大并发序列数 (默认: 256，多卡时可放大)")
    parser.add_argument("--max-num-batched-tokens", type=int, default=None,
                        dest="max_num_batched_tokens",
                        help="vLLM 每次调度的最大 token 数；内存紧张时可设 16384~32768")
    parser.add_argument("--swap-space", type=float, default=1,
                        dest="swap_space",
                        help="vLLM 每张 GPU 的 CPU swap 空间 GiB；降低可减少节点内存占用")
    parser.add_argument("--cpu-offload-gb", type=float, default=0,
                        dest="cpu_offload_gb",
                        help="vLLM 每张 GPU 的 CPU offload GiB；节点内存紧张时保持 0")
    parser.add_argument("--disable-custom-all-reduce", action="store_true",
                        dest="disable_custom_all_reduce",
                        help="禁用 vLLM custom all-reduce，降低多卡通信兼容性风险")
    parser.add_argument("--enforce-eager", action="store_true",
                        dest="enforce_eager", default=True,
                        help="禁用 CUDA graph capture/torch compile 图捕获路径，规避 TP 多卡 NCCL pending work 等待 (默认开启)")
    parser.add_argument("--no-enforce-eager", action="store_false",
                        dest="enforce_eager",
                        help="显式允许 vLLM 使用 CUDA graph；仅建议在稳定性复验通过后用于吞吐消融")
    parser.add_argument("--enable-chunked-prefill", action="store_true",
                        dest="enable_chunked_prefill",
                        help="启用 vLLM chunked prefill，长上下文/长 prompt 场景吞吐更稳")
    parser.add_argument("--kv-cache-dtype", default="auto",
                        dest="kv_cache_dtype",
                        help="vLLM KV cache dtype，默认 auto")
    parser.add_argument("--distributed-executor-backend", default="mp",
                        choices=["mp", "ray", "uni", "external_launcher"],
                        dest="distributed_executor_backend",
                        help="vLLM 多卡执行后端；单机默认 mp，避免 Ray GPU ordinal 映射问题")
    parser.add_argument("--use-fast-image-processor", action="store_true",
                        default=True,
                        dest="use_fast_image_processor",
                        help="向多模态 processor 传 use_fast=True，避免 slow image processor 警告")
    parser.add_argument("--no-use-fast-image-processor", action="store_false",
                        dest="use_fast_image_processor",
                        help="禁用 fast image processor 透传")
    parser.add_argument("--batch-inference", action="store_true",
                        dest="batch_inference",
                        help="v9.0: 启用批量推理模式（收集整批 prompt 后一次 vLLM 调用）")

    # 推理参数
    parser.add_argument("--batch_size", type=int, default=128,
                        help="分段保存粒度（批量推理模式下也是推理批大小）")
    parser.add_argument("--max_answer_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k_logprobs", type=int, default=5,
                        help="首 token logprobs 的 top-K 数量")
    parser.add_argument("--seed", type=int, default=0,
                        help="Python/torch/vLLM 随机种子；默认 0，用于跨实验复验审计")

    # 数据范围
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start_from", type=int, default=0)
    parser.add_argument("--max_table_chars", type=int, default=24000)

    # Recalculate 模式
    parser.add_argument("--recalculate_from", default=None,
                        help="recalculate 模式: 输入 predictions.jsonl 路径")

    # 控制
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--abort-on-generation-error", action="store_true",
                        dest="abort_on_generation_error", default=True,
                        help="Preparation/finalization/vLLM failures abort the run instead of writing empty-answer artifacts (default)")
    parser.add_argument("--continue-on-generation-error", action="store_false",
                        dest="abort_on_generation_error",
                        help="Opt into legacy behavior: write per-sample error artifacts and continue")
    parser.add_argument("--skip-overlong-primary", action="store_true", default=False,
                        dest="skip_overlong_primary",
                        help="Legacy smoke-test behavior: record over-context primary prompts as context_overflow rows and continue")
    parser.add_argument("--run-id", default="", dest="run_id",
                        help="Optional stable run identifier. When omitted, one is derived from method/model/config/timestamp.")
    parser.add_argument("--prediction-record-layout",
                        choices=["research", "legacy"],
                        default="research",
                        dest="prediction_record_layout",
                        help="research=compact predictions.jsonl plus debug sidecar; legacy=old full flat rows in predictions.jsonl")
    parser.add_argument("--write-debug-predictions", action="store_true", default=True,
                        dest="write_debug_predictions",
                        help="Write full diagnostic rows to predictions.debug.jsonl when using research layout")
    parser.add_argument("--no-write-debug-predictions", action="store_false",
                        dest="write_debug_predictions",
                        help="Disable predictions.debug.jsonl sidecar")
    parser.add_argument("--prompt-template-version", default="cscr_prompt_v11",
                        dest="prompt_template_version",
                        help="Version label recorded in run_metadata/prompt profile; does not alter prompting")
    parser.add_argument("--table-serialization-version", default="structure_aware_formatter_v11",
                        dest="table_serialization_version",
                        help="Version label for table-to-prompt serialization; does not alter prompting")
    parser.add_argument("--answer-normalization-version", default="answer_normalizer_v1_eval_utils_v1",
                        dest="answer_normalization_version",
                        help="Version label for answer normalization metadata")
    parser.add_argument("--evaluator-name", default="cscr_multi_caliber_eval",
                        dest="evaluator_name",
                        help="Evaluator name recorded in run_metadata")
    parser.add_argument("--evaluator-version", default="eval_utils_v1",
                        dest="evaluator_version",
                        help="Evaluator version recorded in run_metadata")
    parser.add_argument("--numeric-tolerance", default="rel_tol=1e-6,abs_tol=1e-6",
                        dest="numeric_tolerance",
                        help="Numeric tolerance label recorded in run_metadata")
    parser.add_argument("--error-taxonomy-version", default="classify_error_v1",
                        dest="error_taxonomy_version",
                        help="Error taxonomy version label recorded in run_metadata")

    # v6.0: Conformal Abstention
    parser.add_argument("--conformal-calibrate-from", default=None,
                        dest="conformal_calibrate_from",
                        help="v6.0: 从已有 predictions.jsonl 校准 Conformal Abstention 阈值")
    parser.add_argument("--conformal-alpha", type=float, default=0.05,
                        dest="conformal_alpha",
                        help="v6.0: Conformal Abstention 的 α 值 (default: 0.05)")

    # v7.0a: Success Predictor
    parser.add_argument("--success-predictor-model", default=None,
                        dest="success_predictor_model",
                        help="v7.0a: Success Predictor 模型路径 (.pt)")

    # v8.0a: SCM-CoT
    parser.add_argument("--scm-cot", action="store_true",
                        dest="scm_cot",
                        help="v8.0a: 启用 SCM-CoT prompt (因果证据路径+结构化候选注入)")

    # v8.0b/v8.5: Prompt 风格选择
    parser.add_argument("--prompt-style",
                        choices=["structure_aware", "baseline_e", "selective_evidence", "scm_cot", "table_pruned", "table_focus"],
                        default="structure_aware",
                        dest="prompt_style",
                        help="Prompt 构建风格 (default: structure_aware). "
                             "baseline_e=简洁prompt, selective_evidence=选择性证据注入, "
                             "table_focus=v8.5完整表格+结构焦点提示, "
                             "table_pruned=v8.4硬剪枝复现实验用，不推荐作为主线")
    parser.add_argument("--structural-prior-weighting", action="store_true", default=False,
                        dest="structural_prior_weighting",
                        help="Apply provenance/reliability scores to HCEG edge weights; default is diagnostic-only.")
    parser.add_argument("--disable-candidate-scci", action="store_true", default=False,
                        dest="disable_candidate_scci",
                        help="Use legacy sample-level SCCI instead of candidate-specific targeted SCCI.")

    parser.add_argument("--dataset-prompt-policy",
                        choices=["auto", "legacy", "benchmark", "operation"],
                        default="auto",
                        dest="dataset_prompt_policy",
                        help="auto=HiTab/TableBench 使用已验证旧模板、AIT-QA 使用通用评测格式；"
                             "legacy=全部使用旧 HiTab 风格模板；benchmark=非 HiTab 使用通用评测格式；"
                             "operation=TableBench 使用操作提示消融")
    parser.add_argument("--source-risk-calibration",
                        choices=["auto", "off", "tablebench", "all"],
                        default="auto",
                        dest="source_risk_calibration",
                        help="置信度风险校准，不改答案。auto/tablebench 仅校准 TableBench 的 llm_cert_adjusted")
    parser.add_argument("--source-risk-llm-cert-adjusted-cap",
                        type=float,
                        default=0.74,
                        dest="source_risk_llm_cert_adjusted_cap",
                        help="source-risk calibration 对 llm_cert_adjusted 的置信度上限")
    parser.add_argument("--black-box-commit-policy",
                        choices=["auto", "off", "freeze", "format_only", "certified"],
                        default="auto",
                        dest="black_box_commit_policy",
                        help="闭源 API 生成器的提交策略。auto=format_only for black-box; "
                             "freeze/format_only 禁止证据层语义改写 final answer，"
                             "certified 先冻结生成器答案，仅允许后续证书显式提交。")
    parser.add_argument("--api-format-normalizer",
                        choices=["auto", "off", "conservative"],
                        default="auto",
                        dest="api_format_normalizer",
                        help="闭源 API gold-free 格式归一化。auto 在 format_only/certified 黑盒策略下启用 conservative。")
    parser.add_argument("--surface-heuristic-mode",
                        choices=["off", "diagnostic", "legacy"],
                        default="diagnostic",
                        dest="surface_heuristic_mode",
                        help="控制词表/regex 表面语义规则。diagnostic=只落盘不作为 role 主证据；legacy=复现旧规则；off=关闭 surface role 诊断。")
    parser.add_argument("--operation-support-diagnostics",
                        action="store_true",
                        default=False,
                        dest="operation_support_diagnostics",
                        help="E63: 只记录执行器 support-set、答案角色和 denotation gap；不改变答案或提交决策")
    parser.add_argument("--operation-role-target-diagnostics",
                        action="store_true",
                        default=False,
                        dest="operation_role_target_diagnostics",
                        help="E64: 分离 answer-role / operation-role / filter-target cells，并记录 role-aware rerank；不改变 final answer")
    parser.add_argument("--operation-commit-gate-diagnostics",
                        action="store_true",
                        default=False,
                        dest="operation_commit_gate_diagnostics",
                        help="E65: 基于 E64 role-target 支持集记录保守提交 gate 的 gain/loss；不改变 final answer")
    parser.add_argument("--operation-commit-gate-mode",
                        choices=["diagnostic", "conservative"],
                        default="diagnostic",
                        dest="operation_commit_gate_mode",
                        help="E65 提交模式。diagnostic 只记录 projected gain/loss；conservative 在显式 certified 黑盒策略下提交 eligible answer。")
    parser.add_argument("--operation-commit-version",
                        choices=["E65.3", "E65.4", "E67"],
                        default="E67",
                        dest="operation_commit_version",
                        help="结构提交证书版本。E67 在 E65.4 上增加 measure fiber、aggregate echo 与候选稳定性证书；E65.3 保留历史 scope-role gate。")
    parser.add_argument("--operation-commit-dataset-scope",
                        choices=["tablebench", "hitab", "tablebench_hitab", "all"],
                        default="tablebench",
                        dest="operation_commit_dataset_scope",
                        help="E65 可提交证书的数据集作用域。默认仅 TableBench；HiTab actual-commit smoke 必须显式放宽。")
    parser.add_argument("--main-cert-profile",
                        action="store_true",
                        default=False,
                        dest="main_cert_profile",
                        help="启用论文主线结构证书 profile 审计：只允许 E67/E65.4 boolean certificate gate 改写，旧 fallback/路由/normalizer 必须关闭。")
    parser.add_argument("--enable-cera-repair",
                        action="store_true",
                        default=_env_flag(["CERTA_ENABLE_CERTIFICATE_REPAIR", "CSCR_ENABLE_CERTIFICATE_REPAIR"], False),
                        dest="enable_cera_repair",
                        help="Round 3 CERTA-R: enable CERA repair instrumentation. Default stage E71 builds packet/opportunity shadow only.")
    parser.add_argument("--cera-stage",
                        choices=["E71", "E72"],
                        default=os.environ.get("CERTA_CERA_STAGE", os.environ.get("CSCR_CERA_STAGE", "E71")).upper(),
                        dest="cera_stage",
                        help="CERA experiment stage. E71 builds packets without LLM calls; E72 calls CERA in shadow-only mode.")
    parser.add_argument("--cera-shadow-only",
                        action=argparse.BooleanOptionalAction,
                        default=_env_flag(["CERTA_CERA_SHADOW_ONLY", "CSCR_CERA_SHADOW_ONLY"], True),
                        dest="cera_shadow_only",
                        help="Keep CERA shadow-only. Round 3 does not mutate final answers.")
    parser.add_argument(
        "--cera-commit-approved-repair",
        action="store_true",
        default=_env_flag(
            ["CERTA_CERA_COMMIT_APPROVED_REPAIR"],
            False,
        ),
        dest="cera_commit_approved_repair",
        help=(
            "Commit an E72 repair to final_answer only when the CERA v3 "
            "validator accepts a USE_REPAIRED decision. Disabled by default."
        ),
    )
    parser.add_argument("--cera-round6-e71-v4",
                        action="store_true",
                        default=_env_flag(["CERTA_CERA_ROUND6_E71_V4"], False),
                        dest="cera_round6_e71_v4",
                        help="Round 6: attach deterministic derivation lattice, quotient, contrast, and support-symmetry v3 audit objects to E71 packets.")
    parser.add_argument("--cera-enable-typed-planner",
                        action=argparse.BooleanOptionalAction,
                        default=_env_flag(["CERTA_CERA_ENABLE_TYPED_PLANNER"], False),
                        dest="cera_enable_typed_planner",
                        help="Round 8: call the schema-only Typed Derivation Planner Agent and compile validated plans into repair derivations.")
    parser.add_argument(
        "--cera-planner-boundary",
        choices=[
            "proposal_blind_schema_only",
            "proposal_blind_value_aware",
            "proposal_aware_diagnostic",
        ],
        default="proposal_blind_schema_only",
        dest="cera_planner_boundary",
        help="Round 10 Planner information boundary. A/B remain proposal-blind; C is E71 shadow diagnostic only.",
    )
    parser.add_argument(
        "--cera-planner-contract",
        choices=["legacy_v1", "rcpc_v1", "rcpc_signature_v2"],
        default="legacy_v1",
        dest="cera_planner_contract",
        help=(
            "Planner generation contract. rcpc_v1 preserves the historical schema; "
            "rcpc_signature_v2 requires canonical signature IDs and typed role shapes."
        ),
    )
    parser.add_argument(
        "--cera-planner-legacy-query-semantics-mode",
        choices=["active", "audit_only"],
        default="active",
        dest="cera_planner_legacy_query_semantics_mode",
        help=(
            "Round 12 flat-plan control: active exposes legacy query semantics "
            "and applies compatibility checks; audit_only omits them from the "
            "Planner public request and validation authority."
        ),
    )
    parser.add_argument("--cera-planner-temperature",
                        type=float,
                        default=0.0,
                        dest="cera_planner_temperature",
                        help="Generation temperature for Typed Derivation Planner Agent calls.")
    parser.add_argument("--cera-planner-max-tokens",
                        type=int,
                        default=512,
                        dest="cera_planner_max_tokens",
                        help="Maximum generated tokens for the Typed Derivation Planner JSON response.")
    parser.add_argument("--cera-log-planner-raw-output",
                        action="store_true",
                        default=False,
                        dest="cera_log_planner_raw_output",
                        help="Write raw Typed Derivation Planner output to debug predictions. Off by default.")
    parser.add_argument(
        "--cera-stepwise-trace",
        action="store_true",
        default=False,
        dest="cera_stepwise_trace",
        help=(
            "Round 12 terminal diagnostic: run two constrained Typed Derivation "
            "Planner stages and deterministic typed traces in E71 shadow-only mode."
        ),
    )
    parser.add_argument(
        "--cera-trace-max-assignments",
        type=int,
        default=512,
        dest="cera_trace_max_assignments",
        help=(
            "Engineering cap for complete deterministic Round 12 trace-domain "
            "expansion; truncation is logged as RESOURCE_INCOMPLETE."
        ),
    )
    parser.add_argument(
        "--cera-round12-minimal-patch-shadow",
        action="store_true",
        default=False,
        dest="cera_round12_minimal_patch_shadow",
        help=(
            "Replay finite one-FVF structural patches without Planner or CERA "
            "calls; requires the Round 12 stepwise E71 shadow contract."
        ),
    )
    parser.add_argument("--cera-template-version",
                        default="cera_repair_v2",
                        dest="cera_template_version",
                        help="CERA prompt template version label.")
    parser.add_argument("--cera-temperature",
                        type=float,
                        default=0.0,
                        dest="cera_temperature",
                        help="CERA generation temperature for E72 shadow-only runs.")
    parser.add_argument("--cera-max-tokens",
                        type=int,
                        default=512,
                        dest="cera_max_tokens",
                        help="Maximum generated tokens for the CERA JSON response.")
    parser.add_argument("--cera-require-derivation-program",
                        action=argparse.BooleanOptionalAction,
                        default=True,
                        dest="cera_require_derivation_program",
                        help="Require accepted USE_REPAIRED CERA outputs to include a valid Evidence DSL derivation.")
    parser.add_argument("--cera-require-counterfactual-reference",
                        action=argparse.BooleanOptionalAction,
                        default=True,
                        dest="cera_require_counterfactual_reference",
                        help="Require accepted USE_REPAIRED CERA outputs to cite at least one counterfactual ID.")
    parser.add_argument("--cera-allow-support-only",
                        action="store_true",
                        default=False,
                        dest="cera_allow_support_only",
                        help="Allow support-chain-only CERA validation when no counterfactual chain exists. Disabled by default.")
    parser.add_argument("--cera-table-excerpt-max-cells",
                        type=int,
                        default=32,
                        dest="cera_table_excerpt_max_cells",
                        help="Maximum gold-free table cells included in the CERA evidence packet excerpt.")
    parser.add_argument("--cera-allow-row-major-context",
                        action="store_true",
                        default=False,
                        dest="cera_allow_row_major_context",
                        help="Debug only: allow row-major table-context fallback in CERA packets. Disabled by default.")
    parser.add_argument("--cera-log-full-prompt",
                        action="store_true",
                        default=False,
                        dest="cera_log_full_prompt",
                        help="Write full CERA prompts to debug predictions. Off by default.")
    parser.add_argument("--cera-log-evidence-packet",
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        dest="cera_log_evidence_packet",
                        help="Write CERA evidence packets to debug predictions.")
    parser.add_argument("--cera-strict-debug",
                        action="store_true",
                        default=False,
                        dest="cera_strict_debug",
                        help="Raise CERA shadow exceptions instead of fail-open logging. Off by default.")

    # v8.3: Adaptive Prompt Router (APR)
    parser.add_argument("--adaptive-prompt", action="store_true",
                        dest="adaptive_prompt",
                        help="v8.3: 启用 APR（首 token 熵路由）。低熵样本保持 Round 1，"
                             "高熵样本统一用 intersection_hint 进行 Round 2 推理")
    parser.add_argument("--entropy-threshold-low", type=float, default=0.05,
                        dest="entropy_threshold_low",
                        help="v8.3: APR 低熵阈值 (default: 0.05)。低于此值的样本保持原答案")
    parser.add_argument("--entropy-threshold-high", type=float, default=0.20,
                        dest="entropy_threshold_high",
                        help="v8.3: APR 高熵阈值已不使用，保留参数向后兼容")
    parser.add_argument("--prefix-stable-apr", action="store_true", default=False,
                        dest="prefix_stable_apr",
                        help="启用 prefix-stable APR/SC：保留 Round-1 完整 prompt 为精确前缀，仅追加控制后缀，避免重建长上下文")
    parser.add_argument("--apr-control-suffix-mode", type=str, default="intersection_hint",
                        choices=["intersection_hint", "causal_check", "minimal"],
                        dest="apr_control_suffix_mode",
                        help="prefix-stable APR 的控制后缀类型")
    parser.add_argument("--credal-probe", action="store_true", default=False,
                        dest="credal_probe",
                        help="v8.6: 启用 Credal Probe 纯诊断层（只记录不改答案）")

    # v8.9: Credal-Aware APR Routing (cap/floor/band 三模式)
    parser.add_argument("--credal-gate", action="store_true", default=False,
                        dest="credal_gate",
                        help="v8.9: 启用 credal_width 联合门控（默认 cap 模式: cw 太高跳过 R2）")
    parser.add_argument("--credal-gate-mode", type=str, default="cap",
                        choices=["cap", "floor", "band"],
                        dest="credal_gate_mode",
                        help=("v8.9 模式: "
                              "cap=cw>=cw_high 跳过 R2（默认，安全）; "
                              "floor=v8.8 错误模式 cw<cw_low 跳过 R2（消融对比）; "
                              "band=仅 [cw_low, cw_high) 区间触发 R2"))
    parser.add_argument("--credal-gate-cw", type=float, default=0.10,
                        dest="credal_gate_cw",
                        help="v8.9: floor/band 下界（cw<此值 → 跳过 R2）。default: 0.10")
    parser.add_argument("--credal-gate-cw-high", type=float, default=0.30,
                        dest="credal_gate_cw_high",
                        help="v8.9: cap/band 上界（cw>=此值 → 跳过 R2，因为 R2 几乎修不好）。default: 0.30")
    parser.add_argument("--non-degradation-guard", action="store_true", default=False,
                        dest="non_degradation_guard",
                        help="v8.8: APR R2 答案非退化保护。"
                             "若 R1 是 path_consensus 或 R2 置信度显著低于 R1，回退到 R1")

    # v9.0b: Question-Type Router + Online Normalizer
    parser.add_argument("--question-type-router", action="store_true", default=False,
                        dest="question_type_router",
                        help="v9.0b: 根据 coarse_question_type 路由 prompt_style。"
                             "lookup/proportion/superlative → table_focus; "
                             "其他 → baseline_e（数据驱动，E40/E42 切片验证）")
    parser.add_argument("--online-normalizer", action="store_true", default=False,
                        dest="online_normalizer",
                        help="v9.0b: gold-free 表面格式归一化（只做可由 answer/question 推断的改写）")
    parser.add_argument("--oracle-online-normalizer", action="store_true", default=False,
                        dest="oracle_online_normalizer",
                        help="v9.0b diagnostic only: 允许用 gold 选择数值格式，用于估计 normalizer 上界，"
                             "不得进入主实验。")

    # v9.1: HCEG-Fallback (KG 直检兜底)
    parser.add_argument("--hceg-fallback", action="store_true", default=False,
                        dest="hceg_fallback",
                        help="v9.1: 高 cw 或 compare 错时启用 HCEG 直检兜底替换 LLM 答案")
    parser.add_argument("--hceg-fallback-cw", type=float, default=0.30,
                        dest="hceg_fallback_cw",
                        help="v9.1: HCEG-Fallback 高 cw 阈值 (default: 0.30)")
    parser.add_argument("--hceg-fallback-compare-cw", type=float, default=0.15,
                        dest="hceg_fallback_compare_cw",
                        help="v9.1: compare 类型早触发阈值 (default: 0.15)")
    parser.add_argument("--hceg-fallback-diff-cw", type=float, default=0.10,
                        dest="hceg_fallback_diff_cw",
                        help="v9.1: compare/diff 操作早触发阈值 (default: 0.10)")
    parser.add_argument("--hceg-fallback-policy", type=str, default="candidate_only",
                        choices=["candidate_only", "conservative", "replace"],
                        dest="hceg_fallback_policy",
                        help="v9.1: candidate_only=只记录候选; conservative=仅强证据替换; replace=触发即替换")
    parser.add_argument("--hceg-role-aware", action="store_true", default=False,
                        dest="hceg_role_aware",
                        help="v9.5: HCEG 候选按问题期望答案角色从数值证据回映射到实体/行标签")
    parser.add_argument("--hceg-diagnostic-candidates", type=str, default="triggered",
                        choices=["triggered", "role_sensitive", "all"],
                        dest="hceg_diagnostic_candidates",
                        help="v9.5: HCEG 候选诊断覆盖范围；triggered=仅门控触发样本，role_sensitive=额外覆盖实体角色 compare/superlative，all=全样本诊断")
    parser.add_argument("--certificate-commit-boundary", action="store_true", default=False,
                        dest="certificate_commit_boundary",
                        help="v9.6: 为 HCEG 候选记录 lexicographic certificate commit/reject 诊断")
    parser.add_argument("--certificate-commit-mode", type=str, default="diagnostic",
                        choices=["diagnostic", "conservative"],
                        dest="certificate_commit_mode",
                        help="v9.6: diagnostic=只记录边界; conservative=仅满足全部硬门控时替换")
    parser.add_argument("--certificate-commit-max-llm-confidence", type=float, default=1.0,
                        dest="certificate_commit_max_llm_confidence",
                        help="v9.6: HCEG 候选提交所允许的最高 LLM 置信度")
    parser.add_argument("--certificate-commit-min-credal-width", type=float, default=0.0,
                        dest="certificate_commit_min_credal_width",
                        help="v9.6: HCEG 候选提交所需的最低 credal width 或等价风险信号")
    parser.add_argument("--certificate-commit-allow-diagnostic-candidates",
                        action="store_true", default=False,
                        dest="certificate_commit_allow_diagnostic_candidates",
                        help="v9.6: 允许 role-sensitive diagnostic 候选进入提交边界诊断；默认仅门控触发候选可提交")
    parser.add_argument("--certificate-operation-verifier",
                        action="store_true", default=False,
                        dest="certificate_operation_verifier",
                        help="v9.7: 启用 operation-aware 证书边界，仅允许通过操作角色验证的候选进入提交/ shadow 口径")
    parser.add_argument("--certificate-compare-direction-verifier",
                        action="store_true", default=False,
                        dest="certificate_compare_direction_verifier",
                        help="v9.8: 对 compare/entity 候选进行显式对比答案项与方向证书验证")
    parser.add_argument("--certificate-numeric-direction-verifier",
                        action="store_true", default=False,
                        dest="certificate_numeric_direction_verifier",
                        help="v9.9: 对通过显式对比集合的 compare 候选进行数值方向证书验证")
    parser.add_argument("--certificate-conformal-boundary",
                        action="store_true", default=False,
                        dest="certificate_conformal_boundary",
                        help="v10.0: 启用 conformalized certificate boundary 分数阈值")
    parser.add_argument("--certificate-conformal-threshold", type=float, default=1.01,
                        dest="certificate_conformal_threshold",
                        help="v10.0: 证书提交所需 conformal score 阈值；由校准集脚本生成")
    parser.add_argument("--certificate-conformal-alpha", type=float, default=0.10,
                        dest="certificate_conformal_alpha",
                        help="v10.0: conformal/risk-control 目标错误率，仅记录到 artifact 与 metrics")

    # v9.2: Diverse self-consistency
    parser.add_argument("--self-consistency", action="store_true", default=False,
                        dest="self_consistency",
                        help="v9.2: 对风险样本运行多 prompt 候选投票")
    parser.add_argument("--k-samples", type=int, default=3,
                        dest="k_samples",
                        help="v9.2: self-consistency 总候选预算，包含当前答案 (default: 3)")
    parser.add_argument("--self-consistency-temperature", type=float, default=0.35,
                        dest="self_consistency_temperature",
                        help="v9.2: 多样化推理采样温度 (default: 0.35)")
    parser.add_argument("--self-consistency-max-samples", type=int, default=512,
                        dest="self_consistency_max_samples",
                        help="v9.2: 每次运行最多触发 self-consistency 的样本数")
    parser.add_argument("--self-consistency-trigger", type=str, default="hceg",
                        choices=["hceg", "entropy", "risk", "all"],
                        dest="self_consistency_trigger",
                        help="v9.2: self-consistency 触发策略 (default: hceg)")

    args = parser.parse_args()

    if args.mode != "recalculate" and not args.dry_run:
        if args.generator_backend == "vllm" and not args.model_path:
            parser.error("--model_path is required for --generator-backend=vllm")
        if args.generator_backend in {"openai_chat", "gemini_chat", "vllm_chat"} and not (args.api_model or args.model_path):
            parser.error(f"--api-model or --model_path is required for --generator-backend={args.generator_backend}")
    if args.cera_stage not in {"E71", "E72"}:
        parser.error("--cera-stage must be E71 or E72")
    if args.cera_planner_boundary == "proposal_aware_diagnostic":
        if args.cera_stage != "E71" or not args.cera_shadow_only or args.cera_commit_approved_repair:
            parser.error(
                "--cera-planner-boundary proposal_aware_diagnostic requires "
                "--cera-stage E71, --cera-shadow-only, and no --cera-commit-approved-repair"
            )
    if args.cera_stepwise_trace:
        trace_errors = []
        if args.mode != "full_cert":
            trace_errors.append("use --mode full_cert")
        if not args.enable_cera_repair:
            trace_errors.append("enable --enable-cera-repair")
        if not args.cera_enable_typed_planner:
            trace_errors.append("enable --cera-enable-typed-planner")
        if args.cera_stage != "E71":
            trace_errors.append("use --cera-stage E71")
        if not args.cera_shadow_only:
            trace_errors.append("use --cera-shadow-only")
        if args.cera_commit_approved_repair:
            trace_errors.append("disable --cera-commit-approved-repair")
        if args.cera_planner_boundary != "proposal_blind_schema_only":
            trace_errors.append("use proposal_blind_schema_only")
        if args.cera_planner_legacy_query_semantics_mode != "audit_only":
            trace_errors.append("use legacy query semantics audit_only")
        if args.cera_planner_contract != "rcpc_signature_v2":
            trace_errors.append("use --cera-planner-contract rcpc_signature_v2")
        if args.cera_trace_max_assignments <= 0:
            trace_errors.append("set --cera-trace-max-assignments above zero")
        if trace_errors:
            parser.error("--cera-stepwise-trace requires: " + "; ".join(trace_errors))
    if args.cera_round12_minimal_patch_shadow and not args.cera_stepwise_trace:
        parser.error(
            "--cera-round12-minimal-patch-shadow requires --cera-stepwise-trace"
        )
    if args.enable_cera_repair and args.mode != "full_cert":
        parser.error("--enable-cera-repair requires --mode full_cert")
    if args.cera_commit_approved_repair:
        if not args.enable_cera_repair:
            parser.error(
                "--cera-commit-approved-repair requires --enable-cera-repair"
            )

        if args.cera_stage != "E72":
            parser.error(
                "--cera-commit-approved-repair requires --cera-stage E72"
            )

        if args.cera_shadow_only:
            parser.error(
                "--cera-commit-approved-repair requires --no-cera-shadow-only"
            )

        if args.cera_template_version != "cera_repair_v3":
            parser.error(
                "--cera-commit-approved-repair requires "
                "--cera-template-version cera_repair_v3"
            )

        if not args.cera_enable_typed_planner:
            parser.error(
                "--cera-commit-approved-repair requires "
                "--cera-enable-typed-planner"
            )

    elif args.enable_cera_repair and not args.cera_shadow_only:
        parser.error(
            "Non-shadow CERA execution requires "
            "--cera-commit-approved-repair"
        )

    if args.main_cert_profile:
        profile_violations = []
        if not args.operation_commit_gate_diagnostics:
            profile_violations.append("enable --operation-commit-gate-diagnostics")
        if _normalise_operation_commit_version(args.operation_commit_version) not in {"E65.4", "E67"}:
            profile_violations.append("use --operation-commit-version E65.4 or E67")
        if args.hceg_fallback:
            profile_violations.append("disable --hceg-fallback")
        if args.certificate_commit_boundary:
            profile_violations.append("disable --certificate-commit-boundary")
        if args.self_consistency:
            profile_violations.append("disable --self-consistency")
        if args.adaptive_prompt:
            profile_violations.append("disable --adaptive-prompt")
        if args.question_type_router:
            profile_violations.append("disable --question-type-router")
        if args.online_normalizer or args.oracle_online_normalizer:
            profile_violations.append("disable online normalizers")
        if args.api_format_normalizer != "off":
            profile_violations.append("set --api-format-normalizer off")
        if args.source_risk_calibration != "off":
            profile_violations.append("set --source-risk-calibration off")
        if args.credal_probe:
            profile_violations.append("disable --credal-probe")
        if args.black_box_commit_policy not in {"off", "certified"}:
            profile_violations.append("set --black-box-commit-policy off or certified")
        if profile_violations:
            parser.error(
                "--main-cert-profile requires the clean paper profile: "
                + "; ".join(profile_violations)
            )

    return args


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
