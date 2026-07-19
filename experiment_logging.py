"""Experiment logging helpers for CSCR / Ca2KG TableQA runs.

This module is deliberately side-effect free: it only builds compact prediction
records, debug records, run metadata, and stable hashes. It must not participate
in answer generation, arbitration, or metric decisions.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

PREDICTION_SCHEMA_VERSION = "research_prediction_v2"
DEBUG_SCHEMA_VERSION = "debug_prediction_v1"
RUN_METADATA_SCHEMA_VERSION = "run_metadata_v1"

RUN_LEVEL_SAMPLE_KEYS = {
    "api_base_url",
    "api_key_env",
    "api_cache_mode",
    "api_model",
    "generator_backend",
    "dataset_prompt_policy",
    "black_box_commit_policy",
    "api_format_normalizer_mode",
    "operation_commit_version",
    "operation_support_commit_gate_mode",
    "llm_confidence_source",
    "api_logprobs_unavailable",
    "llm_logprobs_available",
    "black_box_api_generator",
}

LARGE_DEBUG_KEYS = {
    "operation_support_commit_certificate",
    "operation_support_candidates",
    "exec_candidates_summary",
    "certificate_info",
    "cera_evidence_packet",
    "cera_output",
    "cera_prompt",
    "cera_raw_response",
    "cera_request_audit",
    "cera_round10_closure_audit_records",
    "cera_validator",
    "probe_diagnostics",
    "entropy_trajectory_diagnostics",
    "self_consistency_alternatives",
    "self_consistency_candidates",
    "self_consistency_vote_details",
    "apr_round1_info",
    "llm_logprobs",
    "logprobs",
    "prompts",
    "raw_outputs",
}

CORE_PREDICTION_KEYS = [
    "id",
    "table_id",
    "dataset",
    "dataset_question_type",
    "dataset_question_subtype",
    "question",
    "gold_answer",
    "expected_answer",
    "aggregation",
    "coarse_question_type",
    "question_operation",
    "prompt_type",
    "prompt_length",
    "llm_raw_output",
    "llm_answer",
    "final_answer",
    "answer_source",
    "final_confidence",
    "normalized_answer_candidates",
    "gold_aligned_numeric_answer",
    "strict_em",
    "numeric_em",
    "set_em",
    "is_correct_any",
    "hitab_official_em",
    "aitqa_official_em",
    "tablebench_official_em",
    "error_type",
]

COMPACT_DIAGNOSTIC_PREFIXES = (
    "operation_support_commit_",
    "certificate_commit_",
    "hceg_fallback_",
    "black_box_answer_freeze_",
    "black_box_semantic_commit_",
    "api_format_normalizer_",
    "normalizer_",
    "oracle_normalizer_",
    "apr_",
    "self_consistency_",
    "cera_",
)

COMPACT_DIAGNOSTIC_KEYS = {
    "main_cert_profile",
    "legacy_commit_path_used",
    "non_certificate_answer_mutation_used",
    "heuristic_surface_used_for_commit",
    "commit_decision_is_boolean_conjunction",
    "disable_candidate_scci",
    "first_token_entropy",
    "entropy_centroid",
    "late_hep_mass",
    "tail_entropy_phase_ratio",
    "num_interventions",
    "intervention_types",
    "num_exec_candidates",
    "evidence_num_anchors",
    "evidence_num_cells",
    "evidence_has_aggregator",
    "evidence_score",
    "graph_stats",
    "executor_answer",
    "executor_operation",
    "executor_priority",
    "executor_valid",
    "executor_trace",
    "operation_support",
    "operation_support_answer_role",
    "operation_support_operation_role",
    "operation_support_cell_count",
    "operation_support_valid",
    "operation_support_operation",
    "operation_support_denotation",
    "operation_support_target_cell_count",
    "operation_support_filter_cell_count",
    "operation_support_reranked_denotation",
    "operation_support_reranked_operation",
    "operation_support_reranked_valid",
    "operation_support_reranked_role_compatible",
    "operation_support_reranked_operation_compatible",
    "operation_support_candidate_selection_ambiguous",
    "query_table_entity_anchors",
    "query_table_literal_anchors",
    "legacy_heuristic_usage_count",
}

EVALUATOR_KEYS = [
    "strict_em",
    "numeric_em",
    "set_em",
    "is_correct_any",
    "hitab_official_em",
    "aitqa_official_em",
    "tablebench_official_em",
    "error_type",
]

EFFICIENCY_KEYS = [
    "input_token_count",
    "generated_token_count",
    "llm_generation_seconds",
    "non_llm_preparation_seconds",
    "post_llm_finalize_seconds",
    "pipeline_recorded_seconds",
    "context_budget",
    "context_pressure_ratio",
    "compute_pressure_tier",
    "api_usage",
    "api_cache_hit",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def sha256_text(text: str, n: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def sha256_json(value: Any, n: int = 16) -> str:
    return sha256_text(stable_json_dumps(value), n=n)


def _safe_package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_commit(repo_root: Optional[str]) -> Optional[str]:
    if not repo_root:
        return None
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def _git_dirty(repo_root: Optional[str]) -> Optional[bool]:
    if not repo_root:
        return None
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return bool(status.strip())
    except Exception:
        return None


def resolve_run_id(args: Any, method: str, created_at: Optional[str] = None) -> str:
    explicit = str(getattr(args, "run_id", "") or "").strip()
    if explicit:
        return explicit
    created_at = created_at or utc_now_iso()
    payload = {
        "method": method,
        "dataset": getattr(args, "dataset", ""),
        "model_path": getattr(args, "model_path", ""),
        "api_model": getattr(args, "api_model", ""),
        "generator_backend": getattr(args, "generator_backend", ""),
        "prompt_style": getattr(args, "prompt_style", ""),
        "temperature": getattr(args, "temperature", None),
        "top_p": getattr(args, "top_p", None),
        "max_answer_tokens": getattr(args, "max_answer_tokens", None),
        "seed": getattr(args, "seed", None),
        "created_at": created_at,
    }
    stem = f"{str(getattr(args, 'dataset', 'dataset')).replace('-', '_')}_{method.lower().replace('-', '_')}"
    return f"{stem}_{sha256_json(payload, n=10)}"


def table_profile(table_json: Mapping[str, Any], serialized_text: str = "") -> Dict[str, Any]:
    texts = table_json.get("texts") if isinstance(table_json, Mapping) else None
    row_count = len(texts) if isinstance(texts, list) else None
    col_count = max((len(r) for r in texts if isinstance(r, list)), default=0) if isinstance(texts, list) else None
    profile = {
        "title": table_json.get("title") if isinstance(table_json, Mapping) else None,
        "row_count": row_count,
        "column_count": col_count,
        "top_header_rows_num": table_json.get("top_header_rows_num") if isinstance(table_json, Mapping) else None,
        "left_header_columns_num": table_json.get("left_header_columns_num") if isinstance(table_json, Mapping) else None,
        "table_hash": sha256_json(table_json) if table_json else None,
    }
    if serialized_text:
        profile.update(
            {
                "serialized_table_chars": len(serialized_text),
                "serialized_table_hash": sha256_text(serialized_text),
            }
        )
    return {k: v for k, v in profile.items() if v is not None}


def prompt_profile(
    prompt: str,
    prompt_type: str = "",
    template_version: str = "",
    serialization_version: str = "",
    max_table_chars: Optional[int] = None,
    original_table_chars: Optional[int] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "prompt_type": prompt_type,
        "prompt_length": len(prompt or ""),
        "prompt_hash": sha256_text(prompt or ""),
    }
    if template_version:
        out["prompt_template_version"] = template_version
    if serialization_version:
        out["table_serialization_version"] = serialization_version
    if max_table_chars is not None:
        out["max_table_chars"] = int(max_table_chars)
    if original_table_chars is not None:
        out["original_table_chars"] = int(original_table_chars)
        if max_table_chars is not None:
            out["truncation_applied"] = bool(max_table_chars > 0 and original_table_chars > max_table_chars)
    return out


def _public_args(args: Any) -> Dict[str, Any]:
    values = dict(vars(args)) if hasattr(args, "__dict__") else {}
    clean: Dict[str, Any] = {}
    for key, value in values.items():
        if key.startswith("_"):
            continue
        if "key" in key.lower() and key != "api_key_env":
            clean[key] = "<redacted>"
        else:
            clean[key] = value
    return clean


def build_run_metadata(
    args: Any,
    method: str,
    created_at: str,
    run_id: str,
    modules: Optional[Mapping[str, Any]] = None,
    generator: Any = None,
    repo_root: Optional[str] = None,
) -> Dict[str, Any]:
    repo_root = repo_root or str(Path(__file__).resolve().parent)
    backend = getattr(args, "generator_backend", "vllm")
    api_model = getattr(args, "api_model", None) or getattr(args, "model_path", None)
    effective_max_model_len = getattr(generator, "max_model_len", getattr(args, "max_model_len", None))
    requested_max_model_len = getattr(generator, "requested_max_model_len", getattr(args, "max_model_len", None))
    config = _public_args(args)
    return {
        "schema_version": RUN_METADATA_SCHEMA_VERSION,
        "run_id": run_id,
        "method": method,
        "created_at": created_at,
        "dataset": {
            "name": getattr(args, "dataset", ""),
            "input_file": getattr(args, "input_file", ""),
            "table_dir": getattr(args, "table_dir", ""),
            "start_from": getattr(args, "start_from", 0),
            "limit": getattr(args, "limit", None),
            "input_file_hash": file_sha256(getattr(args, "input_file", "")),
        },
        "model": {
            "backend": backend,
            "model_path": getattr(args, "model_path", None),
            "api_model": api_model,
            "dtype": getattr(args, "dtype", None),
            "tensor_parallel_size": getattr(args, "tensor_parallel_size", None),
            "requested_max_model_len": requested_max_model_len,
            "effective_max_model_len": effective_max_model_len,
        },
        "inference": {
            "temperature": getattr(args, "temperature", None),
            "top_p": getattr(args, "top_p", None),
            "top_k_logprobs": getattr(args, "top_k_logprobs", None),
            "max_answer_tokens": getattr(args, "max_answer_tokens", None),
            "batch_size": getattr(args, "batch_size", None),
            "batch_inference": getattr(args, "batch_inference", None),
            "seed": getattr(args, "seed", None),
        },
        "prompt": {
            "prompt_style": getattr(args, "prompt_style", None),
            "dataset_prompt_policy": getattr(args, "dataset_prompt_policy", None),
            "prompt_template_version": getattr(args, "prompt_template_version", None),
            "table_serialization_version": getattr(args, "table_serialization_version", None),
            "max_table_chars": getattr(args, "max_table_chars", None),
        },
        "normalization_and_evaluation": {
            "answer_normalization_version": getattr(args, "answer_normalization_version", None),
            "evaluator_name": getattr(args, "evaluator_name", None),
            "evaluator_version": getattr(args, "evaluator_version", None),
            "numeric_tolerance": getattr(args, "numeric_tolerance", None),
            "error_taxonomy_version": getattr(args, "error_taxonomy_version", None),
        },
        "code": {
            "repo_root": repo_root,
            "git_commit": _git_commit(repo_root),
            "git_dirty": _git_dirty(repo_root),
            "config_hash": sha256_json(config),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "dependency_versions": {
                name: _safe_package_version(name)
                for name in ("torch", "transformers", "vllm", "openai", "tiktoken", "numpy")
            },
        },
        "modules": dict(modules or {}),
        "full_config": config,
    }


def file_sha256(path: str, n: int = 16) -> Optional[str]:
    if not path or not os.path.exists(path) or not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()[:n]
    except OSError:
        return None


def compact_value(value: Any, max_string: int = 2000, max_list: int = 12, max_dict: int = 20) -> bool:
    if value is None or isinstance(value, (bool, int, float)):
        return True
    if isinstance(value, str):
        return len(value) <= max_string
    if isinstance(value, list):
        if len(value) > max_list:
            return False
        return all(compact_value(v, max_string=400, max_list=6, max_dict=8) for v in value)
    if isinstance(value, dict):
        if len(value) > max_dict:
            return False
        return all(isinstance(k, str) and compact_value(v, max_string=400, max_list=6, max_dict=8) for k, v in value.items())
    return False


def _copy_present(src: Mapping[str, Any], keys: Iterable[str]) -> Dict[str, Any]:
    return {key: src[key] for key in keys if key in src}


def _diagnostic_subset(result: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in result.items():
        if key in CORE_PREDICTION_KEYS or key in EFFICIENCY_KEYS or key in RUN_LEVEL_SAMPLE_KEYS or key in LARGE_DEBUG_KEYS:
            continue
        if key in COMPACT_DIAGNOSTIC_KEYS or key.startswith(COMPACT_DIAGNOSTIC_PREFIXES):
            if compact_value(value):
                out[key] = value
    return out


def make_research_prediction_record(result: Mapping[str, Any], run_id: str) -> Dict[str, Any]:
    """Build the compact predictions.jsonl row used for paper statistics.

    The row preserves top-level fields needed by existing metric scripts, while
    moving bulky certificates/candidates/probes to predictions.debug.jsonl.
    """
    row: Dict[str, Any] = {
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "run_id": run_id,
    }
    row.update(_copy_present(result, CORE_PREDICTION_KEYS))
    for key in EFFICIENCY_KEYS:
        if key in result and key != "api_usage":
            row[key] = result[key]
    if "api_usage" in result:
        usage = result.get("api_usage") or {}
        if isinstance(usage, Mapping):
            row["api_usage"] = {
                k: usage.get(k)
                for k in ("prompt_tokens", "completion_tokens", "total_tokens")
                if k in usage
            }
    diagnostics = _diagnostic_subset(result)
    row.update(diagnostics)
    metric_block = _copy_present(result, EVALUATOR_KEYS)
    if metric_block:
        row["metrics"] = metric_block
    efficiency_block = _copy_present(row, EFFICIENCY_KEYS)
    if efficiency_block:
        row["efficiency"] = efficiency_block
    table_block = result.get("table")
    if isinstance(table_block, Mapping):
        row["table"] = dict(table_block)
    prompt_block = result.get("prompt")
    if isinstance(prompt_block, Mapping):
        row["prompt"] = dict(prompt_block)
    return {k: v for k, v in row.items() if v is not None}


def make_debug_prediction_record(result: Mapping[str, Any], run_id: str) -> Dict[str, Any]:
    row = dict(result)
    row.setdefault("schema_version", DEBUG_SCHEMA_VERSION)
    row["run_id"] = run_id
    return row


def make_ca2kg_research_prediction_record(result: Mapping[str, Any], run_id: str) -> Dict[str, Any]:
    answers = result.get("answers") if isinstance(result.get("answers"), Mapping) else {}
    correctness = result.get("correctness") if isinstance(result.get("correctness"), Mapping) else {}
    confidence = result.get("confidence") if isinstance(result.get("confidence"), Mapping) else {}
    final_answer = answers.get("final", result.get("final_answer", "")) if isinstance(answers, Mapping) else result.get("final_answer", "")
    gold = result.get("expected_answer", result.get("gold_answer", ""))
    row: Dict[str, Any] = {
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "run_id": run_id,
        "id": result.get("id"),
        "table_id": result.get("table_id"),
        "dataset": result.get("dataset", "hitab"),
        "question": result.get("question", ""),
        "gold_answer": gold,
        "expected_answer": gold,
        "final_answer": final_answer,
        "answer_source": result.get("answer_source", "ca2kg_frequency_panel"),
        "answers": dict(answers) if isinstance(answers, Mapping) else {},
        "confidence": dict(confidence) if isinstance(confidence, Mapping) else {},
        "correctness": dict(correctness) if isinstance(correctness, Mapping) else {},
        "panel": result.get("panel", {}),
        "table_chars": result.get("table_chars"),
        "has_table": result.get("has_table"),
    }
    if isinstance(result.get("table"), Mapping):
        row["table"] = dict(result["table"])
    if isinstance(result.get("prompt"), Mapping):
        row["prompt"] = dict(result["prompt"])
    if isinstance(result.get("efficiency"), Mapping):
        row["efficiency"] = dict(result["efficiency"])
    return {k: v for k, v in row.items() if v is not None}


def select_prediction_records(rows: Sequence[Mapping[str, Any]], run_id: str, layout: str = "research") -> List[Dict[str, Any]]:
    if layout == "legacy":
        return [dict(row, run_id=run_id) for row in rows]
    return [make_research_prediction_record(row, run_id=run_id) for row in rows]


def select_ca2kg_prediction_records(rows: Sequence[Mapping[str, Any]], run_id: str, layout: str = "research") -> List[Dict[str, Any]]:
    if layout == "legacy":
        return [dict(row, run_id=run_id) for row in rows]
    return [make_ca2kg_research_prediction_record(row, run_id=run_id) for row in rows]
