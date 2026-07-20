"""Pure contracts for the CERTA Round 1 shadow-only active path."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple


PROFILE_REQUIRED_VALUES = {
    "mode": "full_cert",
    "dataset": "hitab",
    "generator_backend": "vllm_chat",
    "api_model": "Qwen3-8B",
    "api_base_url": "http://127.0.0.1:30338/v1",
    "main_cert_profile": True,
    "enable_cera_repair": True,
    "cera_stage": "E71",
    "cera_shadow_only": True,
    "cera_commit_approved_repair": False,
    "cera_enable_typed_planner": True,
    "cera_planner_boundary": "proposal_blind_schema_only",
    "cera_planner_contract": "rcpc_signature_v2",
    "cera_planner_legacy_query_semantics_mode": "audit_only",
    "cera_stepwise_trace": False,
    "adaptive_prompt": False,
    "credal_probe": False,
    "credal_gate": False,
    "question_type_router": False,
    "online_normalizer": False,
    "oracle_online_normalizer": False,
    "api_format_normalizer": "off",
    "hceg_fallback": False,
    "certificate_commit_boundary": False,
    "self_consistency": False,
    "source_risk_calibration": "off",
    "operation_commit_gate_mode": "diagnostic",
    "black_box_commit_policy": "certified",
}

_INVALID_REPLAY_PREFIXES = ("INVALIDATED", "UNEVALUABLE")
_ANSWER_KEY_SPACE = re.compile(r"\s+")


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_json_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _hash_text(payload)


def _answer_key(value: Any) -> str:
    return _ANSWER_KEY_SPACE.sub(" ", str(value or "").strip().lower())


def _operation_stratum(row: Mapping[str, Any]) -> str:
    aggregation = row.get("aggregation", [])
    if isinstance(aggregation, list):
        values = sorted(str(value).strip().lower() for value in aggregation if str(value).strip())
    else:
        values = [str(aggregation).strip().lower()] if str(aggregation).strip() else []
    return "+".join(values) if values else "unknown"


def _stratified_order(rows: Sequence[Tuple[int, Mapping[str, Any]]], seed: int) -> list[Tuple[int, Mapping[str, Any]]]:
    by_stratum: Dict[str, list[Tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    for source_index, row in rows:
        by_stratum[_operation_stratum(row)].append((source_index, row))
    for stratum, items in by_stratum.items():
        items.sort(key=lambda item: _hash_text(f"{seed}|sample|{stratum}|{item[1].get('id', '')}"))
    strata = sorted(by_stratum, key=lambda value: _hash_text(f"{seed}|stratum|{value}"))
    ordered: list[Tuple[int, Mapping[str, Any]]] = []
    offset = 0
    while True:
        added = False
        for stratum in strata:
            items = by_stratum[stratum]
            if offset < len(items):
                ordered.append(items[offset])
                added = True
        if not added:
            break
        offset += 1
    return ordered


def _cohort_records(
    ordered_rows: Sequence[Tuple[int, Mapping[str, Any]]],
    *,
    cohort: str,
    seed: int,
    limit: int,
) -> list[Dict[str, Any]]:
    records = []
    for source_order, (source_index, row) in enumerate(ordered_rows[:limit]):
        sample_id = str(row.get("id") or "")
        table_id = str(row.get("table_id") or "")
        stratum = _operation_stratum(row)
        records.append({
            "sample_id": sample_id,
            "table_id": table_id,
            "source_index": source_index,
            "source_order": source_order,
            "cohort": cohort,
            "seed": seed,
            "operation_stratum": stratum,
            "selection_sha256": _hash_text(f"{seed}|{cohort}|{table_id}|{stratum}|{sample_id}"),
        })
    return records


def select_table_disjoint_cohorts(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    dev_size: int,
    holdout_size: int,
) -> Dict[str, Any]:
    """Select stable table-disjoint cohorts without reading answer or outcome fields."""
    indexed = []
    by_table: Dict[str, list[Tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    seen_ids = set()
    for source_index, row in enumerate(rows):
        sample_id = str(row.get("id") or "")
        table_id = str(row.get("table_id") or "")
        if not sample_id or not table_id:
            continue
        if sample_id in seen_ids:
            raise ValueError(f"duplicate source sample ID: {sample_id}")
        seen_ids.add(sample_id)
        indexed.append((source_index, row))
        by_table[table_id].append((source_index, row))
    table_ids = sorted(by_table, key=lambda value: _hash_text(f"{seed}|table|{value}"))
    dev_tables = set(table_ids[::2])
    holdout_tables = set(table_ids[1::2])
    dev_order = _stratified_order([item for item in indexed if str(item[1].get("table_id")) in dev_tables], seed)
    holdout_order = _stratified_order([item for item in indexed if str(item[1].get("table_id")) in holdout_tables], seed)
    dev = _cohort_records(dev_order, cohort="dev", seed=seed, limit=dev_size)
    holdout = _cohort_records(holdout_order, cohort="holdout", seed=seed, limit=holdout_size)
    dev_table_ids = {row["table_id"] for row in dev}
    holdout_table_ids = {row["table_id"] for row in holdout}
    return {
        "seed": seed,
        "source_row_count": len(indexed),
        "source_table_count": len(by_table),
        "dev": dev,
        "holdout": holdout,
        "dev_shortfall": max(0, dev_size - len(dev)),
        "holdout_shortfall": max(0, holdout_size - len(holdout)),
        "table_disjoint": not bool(dev_table_ids & holdout_table_ids),
    }


def validate_shadow_runtime_config(config: Mapping[str, Any]) -> Tuple[str, ...]:
    errors = []
    for field, expected in PROFILE_REQUIRED_VALUES.items():
        actual = config.get(field)
        if actual != expected:
            errors.append(f"runtime_config_mismatch:{field}:{actual!r}!={expected!r}")
    return tuple(errors)


def validate_shadow_prediction(record: Mapping[str, Any]) -> Tuple[str, ...]:
    errors = []
    if str(record.get("final_answer", "")) != str(record.get("llm_answer", "")):
        errors.append("final_answer_differs_from_b0")
    if not bool(record.get("cera_enabled", True)):
        errors.append("cera_shadow_not_enabled")
    if not str(record.get("cera_stage", "")).startswith("E71"):
        errors.append("cera_stage_not_e71")
    if record.get("cera_shadow_only") is not True:
        errors.append("cera_not_shadow_only")
    if record.get("cera_planner_called") is not True:
        errors.append("planner_not_called")
    if record.get("cera_planner_proposal_visible_to_planner") is not False:
        errors.append("proposal_visible_to_planner")
    if record.get("cera_planner_table_values_visible_to_planner") is not False:
        errors.append("table_values_visible_to_primary_planner")
    forbidden_true = (
        "cera_commit_requested",
        "cera_commit_applied",
        "cera_final_committed",
        "cera_llm_called",
        "operation_support_commit_applied",
        "certificate_commit_applied",
        "hceg_fallback_applied",
        "api_format_normalizer_applied",
        "normalizer_applied",
        "oracle_normalizer_applied",
        "self_consistency_changed",
        "legacy_commit_path_used",
        "non_certificate_answer_mutation_used",
    )
    errors.extend(f"forbidden_runtime_mutation:{field}" for field in forbidden_true if bool(record.get(field, False)))
    if int(record.get("legacy_heuristic_usage_count", 0) or 0) != 0:
        errors.append("legacy_heuristic_usage_nonzero")
    return tuple(errors)


def _contrast(record: Mapping[str, Any]) -> Mapping[str, Any]:
    packet = record.get("cera_evidence_packet") or {}
    if not isinstance(packet, Mapping):
        return {}
    contrast = packet.get("compact_behavioral_contrast_v3") or {}
    return contrast if isinstance(contrast, Mapping) else {}


def _response_status(value: Any) -> str:
    return str(value or "").split(":", 1)[0].upper() or "MISSING"


def _response_vectors(contrast: Mapping[str, Any]) -> Tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    original = contrast.get("original_hypothesis") or {}
    original_vector = original.get("response_vector") or {} if isinstance(original, Mapping) else {}
    alternatives = []
    for hypothesis in contrast.get("alternative_hypotheses") or []:
        if isinstance(hypothesis, Mapping) and isinstance(hypothesis.get("response_vector"), Mapping):
            alternatives.append(hypothesis["response_vector"])
    return original_vector, alternatives


def _intervention_metrics(contrast: Mapping[str, Any]) -> Dict[str, Any]:
    original, alternatives = _response_vectors(contrast)
    common = set()
    paired_replays = 0
    statuses = Counter()
    for value in original.values():
        statuses[_response_status(value)] += 1
    for alternative in alternatives:
        for value in alternative.values():
            statuses[_response_status(value)] += 1
        for intervention_id in set(original) & set(alternative):
            left = _response_status(original[intervention_id])
            right = _response_status(alternative[intervention_id])
            if not left.startswith(_INVALID_REPLAY_PREFIXES) and not right.startswith(_INVALID_REPLAY_PREFIXES):
                common.add(intervention_id)
                paired_replays += 1
    separating = contrast.get("separating_interventions") or []
    return {
        "common_evaluable": len(common),
        "paired_replays": paired_replays,
        "separating": len(separating),
        "status_counts": dict(sorted(statuses.items())),
    }


def _registry_metrics(contrast: Mapping[str, Any]) -> Dict[str, Any]:
    registry = contrast.get("registry") or {}
    if not isinstance(registry, Mapping):
        registry = {}
    derivations = {
        str(row.get("derivation_ref") or ""): row
        for row in registry.get("derivation_records") or []
        if isinstance(row, Mapping) and str(row.get("derivation_ref") or "")
    }
    hypothesis_payloads = []
    original = contrast.get("original_hypothesis")
    if isinstance(original, Mapping):
        hypothesis_payloads.append(original)
    hypothesis_payloads.extend(row for row in contrast.get("alternative_hypotheses") or [] if isinstance(row, Mapping))
    payload_by_id = {str(row.get("hypothesis_id") or ""): row for row in hypothesis_payloads}
    outside = 0
    for hypothesis in registry.get("hypothesis_records") or []:
        if not isinstance(hypothesis, Mapping):
            outside += 1
            continue
        derivation = derivations.get(str(hypothesis.get("derivation_ref") or ""))
        payload = payload_by_id.get(str(hypothesis.get("hypothesis_id") or ""))
        if derivation is None or payload is None:
            outside += 1
            continue
        if _answer_key(derivation.get("executed_answer")) != _answer_key(payload.get("executed_answer")):
            outside += 1
    states = contrast.get("states") or {}
    complete = bool(
        isinstance(states, Mapping)
        and states.get("contrast_registry_complete", False)
        and registry.get("hypothesis_records")
        and registry.get("derivation_records")
        and registry.get("evidence_records")
        and registry.get("intervention_records")
    )
    return {
        "complete": complete,
        "outside_answer_count": outside,
        "hypothesis_count": len(registry.get("hypothesis_records") or []),
        "derivation_count": len(registry.get("derivation_records") or []),
        "evidence_count": len(registry.get("evidence_records") or []),
        "intervention_count": len(registry.get("intervention_records") or []),
    }


def build_blind_sample_master_row(
    record: Mapping[str, Any],
    cohort_record: Mapping[str, Any],
    *,
    dataset_hash: str,
    cohort_hash: str,
    supported_signatures: Iterable[str],
) -> Dict[str, Any]:
    contrast = _contrast(record)
    intervention = _intervention_metrics(contrast)
    registry = _registry_metrics(contrast)
    closure_records = [row for row in record.get("cera_round10_closure_audit_records") or [] if isinstance(row, Mapping)]
    outcome_counts = Counter(str(row.get("closure_outcome") or "UNKNOWN") for row in closure_records)
    original_count = int(record.get("cera_round9_partition_original_count", 0) or 0)
    alternative_count = int(record.get("cera_round9_partition_alternative_count", 0) or 0)
    original_hypothesis = contrast.get("original_hypothesis") or {}
    alternatives = [row for row in contrast.get("alternative_hypotheses") or [] if isinstance(row, Mapping)]
    alternative_answer_keys = {str(row.get("answer_key") or _answer_key(row.get("executed_answer"))) for row in alternatives}
    validation_errors = list(validate_shadow_prediction(record))
    for key in ("cera_runtime_error", "cera_planner_generation_error"):
        if record.get(key):
            validation_errors.append(f"{key}:{record[key]}")
    if record.get("cera_planner_validation_ok") is False:
        validation_errors.extend(str(value) for value in record.get("cera_planner_validation_errors") or ["planner_validation_failed"])
    if record.get("cera_round11_closure_resource_complete") is False:
        validation_errors.append("closure_resource_incomplete")
    logical_calls = 1 + int(bool(record.get("cera_planner_called"))) + int(bool(record.get("cera_llm_called")))
    cache_hits = (
        int(bool(record.get("api_cache_hit")))
        + int(bool(record.get("cera_planner_api_cache_hit")))
        + int(bool(record.get("cera_api_cache_hit")))
    )
    llm_audit = record.get("llm_input_audit") or {}
    planner_schema_hash = record.get("cera_planner_structured_output_schema_hash") or record.get("cera_planner_constraint_schema_hash", "")
    supported = tuple(sorted(set(str(value) for value in supported_signatures)))
    return {
        "sample_id": str(record.get("id") or cohort_record.get("sample_id") or ""),
        "table_id": str(record.get("table_id") or cohort_record.get("table_id") or ""),
        "split": "dev",
        "source_order": int(cohort_record.get("source_order", 0) or 0),
        "dataset_hash": dataset_hash,
        "cohort_manifest_hash": cohort_hash,
        "b0_answer": str(record.get("llm_answer", "")),
        "b0_answer_key": _answer_key(record.get("llm_answer", "")),
        "b0_request_hash": str(llm_audit.get("request_sha256") or ""),
        "b0_prompt_hash": str(llm_audit.get("rendered_prompt_sha256") or record.get("prompt", {}).get("prompt_hash", "")),
        "b0_cache_status": "hit" if record.get("api_cache_hit") else "miss",
        "planner_boundary": str(record.get("cera_planner_boundary_condition") or "proposal_blind_schema_only"),
        "planner_called": bool(record.get("cera_planner_called")),
        "planner_view_hash": str(record.get("cera_planner_view_hash") or ""),
        "planner_schema_hash": str(planner_schema_hash),
        "planner_prompt_hash": str(record.get("cera_planner_prompt_hash") or ""),
        "planner_request_hash": str(record.get("cera_planner_request_hash") or ""),
        "planner_parse_status": "PASS" if record.get("cera_planner_parse_ok") else "FAIL",
        "planner_valid_plan_count": int(record.get("cera_planner_valid_plan_count", 0) or 0),
        "declared_assignment_count": int(record.get("cera_round11_closure_declared_assignment_count", 0) or 0),
        "realized_assignment_count": int(record.get("cera_round11_closure_realized_assignment_count", 0) or 0),
        "closure_resource_complete": bool(record.get("cera_round11_closure_resource_complete", False)),
        "closure_outcome_counts": dict(sorted(outcome_counts.items())),
        "executable_derivation_count": int(record.get("cera_planner_derivation_count", 0) or 0),
        "supported_operation_signatures": list(supported),
        "original_support_count": original_count,
        "alternative_support_count": alternative_count,
        "paired_executable": original_count > 0 and alternative_count > 0,
        "original_answer_class_count": 1 if isinstance(original_hypothesis, Mapping) and original_hypothesis else 0,
        "alternative_answer_class_count": len(alternative_answer_keys),
        "ambiguity_state": "CLEAR" if not contrast.get("unknowns") else "AMBIGUOUS",
        "intervention_basis_hash": _canonical_json_hash((contrast.get("registry") or {}).get("intervention_records") or []),
        "intervention_basis_count": int(record.get("cera_round8_basis_count", 0) or 0),
        "paired_replay_count": intervention["paired_replays"],
        "common_evaluable_intervention_count": intervention["common_evaluable"],
        "separating_intervention_count": intervention["separating"],
        "intervention_status_counts": intervention["status_counts"],
        "registry_complete": registry["complete"],
        "registry_outside_answer_count": registry["outside_answer_count"],
        "registry_hypothesis_count": registry["hypothesis_count"],
        "registry_derivation_count": registry["derivation_count"],
        "registry_evidence_count": registry["evidence_count"],
        "registry_intervention_count": registry["intervention_count"],
        "round1_final_answer": str(record.get("final_answer", "")),
        "round1_answer_source": "B0",
        "legacy_answer_mutation_used": bool(record.get("legacy_commit_path_used") or record.get("non_certificate_answer_mutation_used")),
        "cera_enabled": bool(record.get("cera_enabled")),
        "cera_stage": str(record.get("cera_stage") or ""),
        "cera_would_commit": bool(record.get("cera_would_commit")),
        "cera_commit_applied": bool(record.get("cera_commit_applied") or record.get("cera_final_committed")),
        "failure_stage": "" if not validation_errors else "ROUND1_CONTRACT",
        "failure_reasons": sorted(set(validation_errors)),
        "logical_calls": logical_calls,
        "actual_attempts": max(0, logical_calls - cache_hits),
        "retries": 0,
        "cache_hits": cache_hits,
        "cache_misses": max(0, logical_calls - cache_hits),
        "prompt_tokens": int(record.get("input_token_count", 0) or 0) + int(record.get("cera_planner_input_tokens", 0) or 0) + int(record.get("cera_input_tokens", 0) or 0),
        "completion_tokens": int(record.get("generated_token_count", 0) or 0) + int(record.get("cera_planner_output_tokens", 0) or 0) + int(record.get("cera_output_tokens", 0) or 0),
        "latency_seconds": float(record.get("pipeline_recorded_seconds", 0.0) or 0.0),
    }
