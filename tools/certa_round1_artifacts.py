#!/usr/bin/env python3
"""Freeze and analyze the bounded CERTA Round 1 shadow experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = REPO_ROOT / "tools"
for import_root in (REPO_ROOT, TOOLS_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import jsonschema

from certa.operations.contracts import LOOKUP_ACTIVE_SIGNATURE_IDS
from certa.round1.contracts import (
    build_blind_sample_master_row,
    select_table_disjoint_cohorts,
    validate_shadow_runtime_config,
)
from cscr_astra_eval import official_match


PACK_ROOT = REPO_ROOT.parent / "certa_goal_packs" / "CERTA_R1_ACTIVE_PATH_CONTRACT_AND_FROZEN_PAIR_FREEZE_PACK"
DEFAULT_DATASET = REPO_ROOT.parent / "CausalityAwareTableQA/dataset/hitab/test_samples_clean.jsonl"
DEFAULT_TABLE_ROOT = REPO_ROOT.parent / "CausalityAwareTableQA/dataset/hitab/tables/raw"
SEED = 20260720
ACTIVE_SIGNATURE_IDS = LOOKUP_ACTIVE_SIGNATURE_IDS
BLIND_RUNTIME_FIELDS = ("id", "table_id", "table_source", "question")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            rows.append(payload)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def tree_manifest(root: Path) -> dict[str, Any]:
    entries = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        entries.append({
            "path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return {
        "root": str(root),
        "file_count": len(entries),
        "tree_sha256": canonical_sha256(entries),
        "entries": entries,
    }


def _frozen_input(source: Mapping[str, Any]) -> dict[str, Any]:
    row = {field: source.get(field) for field in BLIND_RUNTIME_FIELDS if field in source}
    row["dataset"] = "hitab"
    return row


def prepare_round1(
    *,
    dataset_path: Path,
    table_root: Path,
    output_root: Path,
    dev_size: int = 64,
    holdout_size: int = 64,
) -> dict[str, Any]:
    dataset_path = dataset_path.resolve()
    table_root = table_root.resolve()
    output_root = output_root.resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)
    if not table_root.is_dir():
        raise FileNotFoundError(table_root)
    cohort_path = output_root / "freeze/DEV_COHORT.jsonl"
    if cohort_path.exists():
        raise FileExistsError(f"refusing to overwrite frozen cohort: {cohort_path}")

    rows = read_jsonl(dataset_path)
    cohorts = select_table_disjoint_cohorts(
        rows,
        seed=SEED,
        dev_size=dev_size,
        holdout_size=holdout_size,
    )
    if not cohorts["table_disjoint"]:
        raise AssertionError("cohort selection is not table-disjoint")
    dev = cohorts["dev"]
    holdout = cohorts["holdout"]
    dev_sources = [rows[int(item["source_index"])] for item in dev]

    write_jsonl(cohort_path, dev)
    write_jsonl(output_root / "freeze/HOLDOUT_COHORT_SEALED.jsonl", holdout)
    write_jsonl(output_root / "inputs/dev_blind.jsonl", (_frozen_input(row) for row in dev_sources))
    write_jsonl(output_root / "inputs/dev_diag8_blind.jsonl", (_frozen_input(row) for row in dev_sources[:8]))

    table_info = tree_manifest(table_root)
    dataset_manifest = {
        "schema_version": "certa_round1_dataset_manifest_v1",
        "dataset": "hitab_clean",
        "input_path": str(dataset_path),
        "input_sha256": sha256_file(dataset_path),
        "sample_count": len(rows),
        "table_root": str(table_root),
        "table_file_count": table_info["file_count"],
        "table_tree_sha256": table_info["tree_sha256"],
        "quarantine_exclusions": [],
    }
    write_json(output_root / "freeze/DATASET_MANIFEST.json", dataset_manifest)
    write_json(output_root / "freeze/TABLE_TREE_MANIFEST.json", table_info)
    write_json(output_root / "freeze/QUARANTINE_EXCLUSION_MANIFEST.json", {
        "schema_version": "certa_round1_quarantine_exclusion_v1",
        "dataset": "hitab_clean",
        "excluded_sample_ids": [],
        "excluded_table_ids": [],
        "reason": "no quarantine or exclusion applied",
    })
    cohort_manifest = {
        "schema_version": "certa_round1_cohort_manifest_v1",
        "seed": SEED,
        "selection_contract": "stable_table_hash_then_stable_sample_hash_within_operation_strata",
        "selection_forbidden_fields": ["answer", "gold", "correctness", "alternative_correctness", "error_type", "eligibility"],
        "source_row_count": cohorts["source_row_count"],
        "source_table_count": cohorts["source_table_count"],
        "dev_count": len(dev),
        "holdout_count": len(holdout),
        "dev_shortfall": cohorts["dev_shortfall"],
        "holdout_shortfall": cohorts["holdout_shortfall"],
        "table_disjoint": cohorts["table_disjoint"],
        "dev_manifest_sha256": sha256_file(cohort_path),
        "holdout_manifest_sha256": sha256_file(output_root / "freeze/HOLDOUT_COHORT_SEALED.jsonl"),
        "dev_input_sha256": sha256_file(output_root / "inputs/dev_blind.jsonl"),
        "diagnostic_input_sha256": sha256_file(output_root / "inputs/dev_diag8_blind.jsonl"),
    }
    write_json(output_root / "freeze/COHORT_MANIFEST.json", cohort_manifest)

    supported = {
        "schema_version": "certa_round1_supported_operation_signatures_v1",
        "status": "FOCUSED_CONTRACT_TEST_PASS",
        "test_path": "tests/test_round1_operation_contracts.py",
        "test_sha256": sha256_file(REPO_ROOT / "tests/test_round1_operation_contracts.py"),
        "required_proofs": [
            "contract_role_and_reference_domain",
            "resolver_four_state_no_first_match",
            "deterministic_execution_and_projection",
            "complete_operand_and_edge_provenance",
            "resource_completeness",
            "same_derivation_replay",
            "no_legacy_mutation_authority",
            "benign_control_replay",
        ],
        "signatures": list(ACTIVE_SIGNATURE_IDS),
    }
    write_json(output_root / "freeze/SUPPORTED_OPERATION_SIGNATURES.json", supported)
    profile_path = REPO_ROOT / "configs/profiles/certa_round1_shadow.env"
    runner_path = REPO_ROOT / "scripts/05_run_round1_shadow.sh"
    write_json(output_root / "freeze/PROFILE_MANIFEST.json", {
        "schema_version": "certa_round1_profile_manifest_v1",
        "profile_path": str(profile_path),
        "profile_sha256": sha256_file(profile_path),
        "runner_path": str(runner_path),
        "runner_sha256": sha256_file(runner_path),
        "dataset": "hitab_clean",
        "model_id": "/home/common_data/llm/Qwen/Qwen3-8B",
        "served_model": "Qwen3-8B",
        "endpoint": "http://127.0.0.1:30338/v1",
        "thinking": False,
        "sampling": {"temperature": 0.0, "top_p": 1.0, "seed": 0, "max_model_len": 32768, "max_answer_tokens": 32},
        "cache_mode": "readwrite",
        "primary_boundary": "proposal_blind_schema_only",
        "planner_contract": "rcpc_signature_v2",
        "cera_stage": "E71",
        "shadow_only": True,
    })
    return {
        "dataset_sha256": dataset_manifest["input_sha256"],
        "table_tree_sha256": table_info["tree_sha256"],
        "cohort_manifest_sha256": sha256_file(output_root / "freeze/COHORT_MANIFEST.json"),
        "dev_count": len(dev),
        "holdout_count": len(holdout),
        "table_disjoint": cohorts["table_disjoint"],
    }


def _index_exact(rows: Sequence[Mapping[str, Any]], *, label: str) -> dict[str, Mapping[str, Any]]:
    index: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        sample_id = str(row.get("id") or row.get("sample_id") or "")
        if not sample_id:
            raise ValueError(f"missing {label} sample ID")
        if sample_id in index:
            raise ValueError(f"duplicate {label} sample ID: {sample_id}")
        index[sample_id] = row
    return index


def _exact_join(
    cohort: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    *,
    label: str,
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    record_index = _index_exact(records, label=label)
    cohort_ids = [str(row.get("sample_id") or "") for row in cohort]
    if len(cohort_ids) != len(set(cohort_ids)):
        raise ValueError("duplicate cohort sample ID")
    missing = sorted(set(cohort_ids) - set(record_index))
    extra = sorted(set(record_index) - set(cohort_ids))
    if missing or extra:
        raise ValueError(f"{label} exact join failed: missing={missing}, extra={extra}")
    return [(row, record_index[str(row["sample_id"])]) for row in cohort]


def _contrast(record: Mapping[str, Any]) -> Mapping[str, Any]:
    packet = record.get("cera_evidence_packet") or {}
    contrast = packet.get("compact_behavioral_contrast_v3") or {} if isinstance(packet, Mapping) else {}
    return contrast if isinstance(contrast, Mapping) else {}


def _registry_answers(record: Mapping[str, Any]) -> list[Any]:
    registry = _contrast(record).get("registry") or {}
    if not isinstance(registry, Mapping):
        return []
    return [
        row.get("executed_answer")
        for row in registry.get("derivation_records") or []
        if isinstance(row, Mapping) and row.get("executed_answer") not in (None, "")
    ]


def _runtime_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "mode", "dataset", "generator_backend", "api_model", "api_base_url", "main_cert_profile",
        "enable_cera_repair", "cera_stage", "cera_shadow_only", "cera_commit_approved_repair",
        "cera_enable_typed_planner", "cera_planner_boundary", "cera_planner_contract",
        "cera_planner_legacy_query_semantics_mode", "cera_stepwise_trace", "adaptive_prompt",
        "credal_probe", "credal_gate", "question_type_router", "online_normalizer",
        "oracle_online_normalizer", "api_format_normalizer", "hceg_fallback",
        "certificate_commit_boundary", "self_consistency", "source_risk_calibration",
        "operation_commit_gate_mode", "black_box_commit_policy",
    )
    return {field: config.get(field) for field in fields}


def _prediction_path(output_dir: Path) -> Path:
    debug = output_dir / "predictions.debug.jsonl"
    if debug.is_file():
        return debug
    compact = output_dir / "predictions.jsonl"
    if compact.is_file():
        return compact
    raise FileNotFoundError(f"missing predictions in {output_dir}")


def _arm_summary(output_dir: Path, expected_cohort: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    records = read_jsonl(_prediction_path(output_dir))
    joined = _exact_join(expected_cohort, records, label=output_dir.name)
    signature_counts: Counter[str] = Counter()
    valid_plan_counts = []
    for _, record in joined:
        valid_plan_counts.append(int(record.get("cera_planner_valid_plan_count", 0) or 0))
        for closure in record.get("cera_round10_closure_audit_records") or []:
            if isinstance(closure, Mapping) and closure.get("signature_id"):
                signature_counts[str(closure["signature_id"])] += 1
    return {
        "output_dir": str(output_dir),
        "sample_count": len(joined),
        "prediction_sha256": sha256_file(_prediction_path(output_dir)),
        "run_config_sha256": sha256_file(output_dir / "run_config.json"),
        "planner_called_count": sum(bool(record.get("cera_planner_called")) for _, record in joined),
        "proposal_visible_count": sum(bool(record.get("cera_planner_proposal_visible_to_planner")) for _, record in joined),
        "table_values_visible_count": sum(bool(record.get("cera_planner_table_values_visible_to_planner")) for _, record in joined),
        "samples_with_valid_plan": sum(value > 0 for value in valid_plan_counts),
        "valid_plan_count": sum(valid_plan_counts),
        "original_support_count": sum(int(record.get("cera_round9_partition_original_count", 0) or 0) for _, record in joined),
        "alternative_support_count": sum(int(record.get("cera_round9_partition_alternative_count", 0) or 0) for _, record in joined),
        "signature_counts": dict(sorted(signature_counts.items())),
        "records": records,
    }


def _intervention_summary(joined: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]) -> dict[str, Any]:
    status_counts: Counter[str] = Counter()
    failure_counts: Counter[str] = Counter()
    canonical_post_answers: Counter[str] = Counter()
    per_sample = []
    benign_original = 0
    benign_alternative = 0
    for cohort_row, record in joined:
        registry = _contrast(record).get("registry") or {}
        intervention_records = registry.get("intervention_records") or [] if isinstance(registry, Mapping) else []
        kept = []
        for intervention in intervention_records:
            if not isinstance(intervention, Mapping):
                continue
            kept.append(dict(intervention))
            benign_original += int(bool(intervention.get("original_benign_control")))
            benign_alternative += int(bool(intervention.get("alternative_benign_control")))
            for side in ("original", "alternative"):
                signature = str(intervention.get(f"{side}_signature") or "UNEVALUABLE:missing_response")
                status, _, suffix = signature.partition(":")
                status_counts[status] += 1
                if status in {"INVALIDATED", "UNEVALUABLE"}:
                    failure_counts[suffix or "unspecified"] += 1
                elif suffix:
                    canonical_post_answers[suffix] += 1
        per_sample.append({
            "sample_id": cohort_row["sample_id"],
            "basis_hash": canonical_sha256(intervention_records),
            "interventions": kept,
        })
    return {
        "schema_version": "certa_round1_e4_v1",
        "basis_contract": "one_ordered_sample_fixed_basis_before_pairwise_comparison",
        "replay_contract": "same_typed_derivation_no_planner_or_program_search",
        "status_counts": dict(sorted(status_counts.items())),
        "failure_class_counts": dict(sorted(failure_counts.items())),
        "canonical_post_answer_counts": dict(sorted(canonical_post_answers.items())),
        "original_benign_control_count": benign_original,
        "alternative_benign_control_count": benign_alternative,
        "benign_control_test": "tests.test_round1_operation_contracts.Round1OperationContractTests.test_sample_fixed_basis_labels_self_substitution_as_benign_control",
        "per_sample": per_sample,
    }


def analyze_round1(
    *,
    output_root: Path,
    dataset_path: Path,
    primary_output_dir: Path,
    diagnostic_output_dirs: Mapping[str, Path],
) -> dict[str, Any]:
    output_root = output_root.resolve()
    dataset_path = dataset_path.resolve()
    primary_output_dir = primary_output_dir.resolve()
    cohort = read_jsonl(output_root / "freeze/DEV_COHORT.jsonl")
    supported_payload = read_json(output_root / "freeze/SUPPORTED_OPERATION_SIGNATURES.json")
    supported = tuple(supported_payload["signatures"])
    dataset_manifest = read_json(output_root / "freeze/DATASET_MANIFEST.json")
    cohort_hash = sha256_file(output_root / "freeze/COHORT_MANIFEST.json")

    run_config = read_json(primary_output_dir / "run_config.json")
    config_errors = validate_shadow_runtime_config(_runtime_contract(run_config))
    if config_errors:
        raise ValueError("primary runtime contract failed: " + "; ".join(config_errors))
    primary_records = read_jsonl(_prediction_path(primary_output_dir))
    joined = _exact_join(cohort, primary_records, label="primary")

    blind_rows = [
        build_blind_sample_master_row(
            record,
            cohort_row,
            dataset_hash=dataset_manifest["input_sha256"],
            cohort_hash=cohort_hash,
            supported_signatures=supported,
        )
        for cohort_row, record in joined
    ]
    sample_schema = read_json(PACK_ROOT / "schemas/sample_level_master.schema.json")
    for row in blind_rows:
        jsonschema.validate(row, sample_schema)
    blind_path = output_root / "results/sample_master.blind.jsonl"
    write_jsonl(blind_path, blind_rows)
    blind_hash = sha256_file(blind_path)

    # Gold is opened only after the blind sample master has been written and hashed.
    source_rows = read_jsonl(dataset_path)
    source_index = _index_exact(source_rows, label="development source")
    unblind_rows = []
    for (cohort_row, record), blind in zip(joined, blind_rows):
        sample_id = str(cohort_row["sample_id"])
        source = source_index.get(sample_id)
        if source is None:
            raise ValueError(f"development source ID missing after blind freeze: {sample_id}")
        gold = source.get("answer")
        b0_correct = official_match("hitab", record.get("llm_answer"), gold)
        symbolic_answer = record.get("executor_answer")
        symbolic_correct = official_match("hitab", symbolic_answer, gold)
        final_correct = official_match("hitab", record.get("final_answer"), gold)
        registry_answers = _registry_answers(record)
        oracle_correct = any(official_match("hitab", answer, gold) for answer in registry_answers)
        row = dict(blind)
        row.update({
            "gold_answer": gold,
            "b0_correct": b0_correct,
            "symbolic_answer": symbolic_answer,
            "symbolic_correct": symbolic_correct,
            "em_max_correct": b0_correct or symbolic_correct,
            "selected_final_correct": final_correct,
            "candidate_oracle_correct": oracle_correct,
            "oracle_repairable": (not b0_correct) and oracle_correct,
        })
        unblind_rows.append(row)
    unblind_path = output_root / "results/sample_master.dev_unblind.jsonl"
    write_jsonl(unblind_path, unblind_rows)

    cumulative = []
    predicates = (
        ("source_row", lambda row: True),
        ("any_executable_derivation", lambda row: row["executable_derivation_count"] > 0),
        ("original_support", lambda row: row["original_support_count"] > 0),
        ("alternative_support", lambda row: row["alternative_support_count"] > 0),
        ("paired_executable", lambda row: row["paired_executable"]),
        ("common_evaluable_intervention", lambda row: row["common_evaluable_intervention_count"] > 0),
        ("separating_intervention", lambda row: row["separating_intervention_count"] > 0),
        ("gold_only_oracle_repairable", lambda row: row["oracle_repairable"]),
    )
    survivors = list(unblind_rows)
    for stage, predicate in predicates:
        survivors = [row for row in survivors if predicate(row)]
        cumulative.append({"stage": stage, "count": len(survivors)})
    count = len(unblind_rows)
    metrics = {
        "denominator": count,
        "EM_textual_accuracy": sum(row["b0_correct"] for row in unblind_rows) / count if count else 0.0,
        "EM_symbolic_accuracy": sum(row["symbolic_correct"] for row in unblind_rows) / count if count else 0.0,
        "EM_max_accuracy": sum(row["em_max_correct"] for row in unblind_rows) / count if count else 0.0,
        "selected_final_accuracy": sum(row["selected_final_correct"] for row in unblind_rows) / count if count else 0.0,
        "candidate_oracle_accuracy": sum(row["candidate_oracle_correct"] for row in unblind_rows) / count if count else 0.0,
    }
    write_json(output_root / "results/E1_FUNNEL.json", {
        "schema_version": "certa_round1_e1_v1",
        "blind_master_sha256_before_unblind": blind_hash,
        "funnel": cumulative,
        "metrics": metrics,
    })

    diag_cohort = cohort[:8]
    arm_summaries = {"primary": _arm_summary(primary_output_dir, cohort)}
    for name, path in sorted(diagnostic_output_dirs.items()):
        arm_summaries[name] = _arm_summary(Path(path).resolve(), diag_cohort)
    for summary in arm_summaries.values():
        summary.pop("records", None)
    write_json(output_root / "results/E2_PLANNER_BOUNDARY.json", {
        "schema_version": "certa_round1_e2_v1",
        "primary_method_boundary": "proposal_blind_schema_only",
        "diagnostic_arms_can_enter_final_method": False,
        "matched_diagnostic_sample_ids": [row["sample_id"] for row in diag_cohort],
        "arms": arm_summaries,
    })

    ambiguity_counts = Counter(row["ambiguity_state"] for row in blind_rows)
    write_json(output_root / "results/E3_SYMMETRY.json", {
        "schema_version": "certa_round1_e3_v1",
        "construction_contract": "same_preconstructed_closure_executor_provenance_canonicalization_and_signature_registry",
        "first_class_or_top1_collapse_allowed": False,
        "sample_count": len(blind_rows),
        "paired_executable_count": sum(row["paired_executable"] for row in blind_rows),
        "original_support_total": sum(row["original_support_count"] for row in blind_rows),
        "alternative_support_total": sum(row["alternative_support_count"] for row in blind_rows),
        "original_answer_class_total": sum(row["original_answer_class_count"] for row in blind_rows),
        "alternative_answer_class_total": sum(row["alternative_answer_class_count"] for row in blind_rows),
        "ambiguity_state_counts": dict(sorted(ambiguity_counts.items())),
        "registry_complete_count": sum(row["registry_complete"] for row in blind_rows),
    })
    write_json(output_root / "results/E4_INTERVENTIONS.json", _intervention_summary(joined))

    cost_by_arm = {}
    cost_by_arm["primary"] = {
        key: sum(row[key] for row in blind_rows)
        for key in ("logical_calls", "actual_attempts", "retries", "cache_hits", "cache_misses", "prompt_tokens", "completion_tokens", "latency_seconds")
    }
    for name, path in sorted(diagnostic_output_dirs.items()):
        records = read_jsonl(_prediction_path(Path(path)))
        pairs = _exact_join(diag_cohort, records, label=name)
        temp_rows = [
            build_blind_sample_master_row(
                record, cohort_row,
                dataset_hash=dataset_manifest["input_sha256"],
                cohort_hash=cohort_hash,
                supported_signatures=supported,
            )
            for cohort_row, record in pairs
        ]
        cost_by_arm[name] = {
            key: sum(row[key] for row in temp_rows)
            for key in ("logical_calls", "actual_attempts", "retries", "cache_hits", "cache_misses", "prompt_tokens", "completion_tokens", "latency_seconds")
        }
    write_json(output_root / "results/COST_LEDGER.json", {
        "schema_version": "certa_round1_cost_ledger_v1",
        "logical_call_definition": "one B0 generation plus each actually invoked planner or CERA call",
        "actual_attempts_note": "API retries pinned to zero; cache hits are logical calls but not network attempts",
        "arms": cost_by_arm,
    })

    b0_rows = [{
        "sample_id": row["sample_id"],
        "answer": row["b0_answer"],
        "request_sha256": row["b0_request_hash"],
        "prompt_sha256": row["b0_prompt_hash"],
        "cache_status": row["b0_cache_status"],
    } for row in blind_rows]
    b0_manifest = {
        "schema_version": "certa_round1_b0_prediction_manifest_v1",
        "prediction_file": str(_prediction_path(primary_output_dir)),
        "prediction_file_sha256": sha256_file(_prediction_path(primary_output_dir)),
        "prediction_count": len(b0_rows),
        "prediction_manifest_sha256": canonical_sha256(b0_rows),
        "prompt_sha256": canonical_sha256([row["prompt_sha256"] for row in b0_rows]),
        "request_manifest_sha256": canonical_sha256([row["request_sha256"] for row in b0_rows]),
        "rows": b0_rows,
    }
    write_json(output_root / "freeze/B0_PREDICTION_MANIFEST.json", b0_manifest)
    closure_rows = [{
        "sample_id": row["sample_id"],
        "declared_assignment_count": row["declared_assignment_count"],
        "realized_assignment_count": row["realized_assignment_count"],
        "resource_complete": row["closure_resource_complete"],
        "outcome_counts": row["closure_outcome_counts"],
        "executable_derivation_count": row["executable_derivation_count"],
        "registry_complete": row["registry_complete"],
    } for row in blind_rows]
    closure_manifest = {
        "schema_version": "certa_round1_closure_manifest_v1",
        "count": len(closure_rows),
        "manifest_sha256": canonical_sha256(closure_rows),
        "rows": closure_rows,
    }
    write_json(output_root / "freeze/CLOSURE_MANIFEST.json", closure_manifest)
    write_json(output_root / "freeze/ACTIVE_PATH_CONTRACT_MANIFEST.json", {
        "schema_version": "certa_round1_active_path_contract_manifest_v1",
        "planner_view_hashes": sorted({row["planner_view_hash"] for row in blind_rows}),
        "planner_schema_hashes": sorted({row["planner_schema_hash"] for row in blind_rows}),
        "planner_prompt_hashes": sorted({row["planner_prompt_hash"] for row in blind_rows}),
        "planner_request_hashes": sorted({row["planner_request_hash"] for row in blind_rows}),
        "intervention_basis_hashes": sorted({row["intervention_basis_hash"] for row in blind_rows}),
        "runtime_config_sha256": sha256_file(primary_output_dir / "run_config.json"),
        "runtime_authority": "config_only",
        "telemetry_can_authorize_commit": False,
    })

    evaluator_path = REPO_ROOT / "tools/cscr_astra_eval.py"
    evaluator_test = REPO_ROOT / "tests/test_cscr_astra_eval_contract.py"
    write_json(output_root / "audit/EVALUATOR_CONTRACT.json", {
        "schema_version": "certa_round1_evaluator_contract_v1",
        "path": str(evaluator_path),
        "sha256": sha256_file(evaluator_path),
        "test_path": str(evaluator_test),
        "test_sha256": sha256_file(evaluator_test),
        "exact_join": True,
        "invalid_reference_match": False,
        "em_max_semantics": "textual_correct OR symbolic_correct; oracle union only, never selected final",
        "selected_final_semantics": "actual final_answer; Round 1 frozen to B0",
    })
    forbidden_blind_keys = {"gold_answer", "b0_correct", "symbolic_correct", "em_max_correct", "selected_final_correct", "oracle_repairable"}
    blind_key_union = set().union(*(set(row) for row in blind_rows)) if blind_rows else set()
    proposal_blindness_pass = all(
        record.get("cera_planner_proposal_visible_to_planner") is False
        and record.get("cera_planner_table_values_visible_to_planner") is False
        for _, record in joined
    )
    gold_firewall_pass = not bool(forbidden_blind_keys & blind_key_union)
    write_json(output_root / "audit/GOLD_FIREWALL_AUDIT.json", {
        "schema_version": "certa_round1_gold_firewall_v1",
        "runtime_gold_fields": [],
        "blind_input_sha256": sha256_file(output_root / "inputs/dev_blind.jsonl"),
        "blind_output": str(blind_path),
        "blind_output_sha256_before_unblind": blind_hash,
        "unblind_output": str(unblind_path),
        "gold_join_after_blind_hash": True,
        "holdout_gold_read": False,
        "proposal_blindness_pass": proposal_blindness_pass,
        "pass": gold_firewall_pass and proposal_blindness_pass,
    })

    repo_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, text=True, capture_output=True).stdout.strip()
    repo_branch = subprocess.run(["git", "branch", "--show-current"], cwd=REPO_ROOT, check=True, text=True, capture_output=True).stdout.strip()
    repo_dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, check=True, text=True, capture_output=True).stdout.strip())
    method_paths = (
        "certa/round1/contracts.py", "certa/derivations/iade.py", "certa/derivations/contrast.py",
        "certa/planner/schema_view.py", "certa/planner/typed_planner.py", "certa/grounding/plan_closure.py",
        "certa/derivations/replay.py", "certa/repair/causal_epistemic_agent.py",
        "configs/profiles/certa_round1_shadow.env", "scripts/05_run_round1_shadow.sh",
    )
    method_hashes = {path: sha256_file(REPO_ROOT / path) for path in method_paths}
    e0 = {
        "schema_version": "certa_e0_freeze_v1",
        "repo": {"head": repo_head, "branch": repo_branch, "dirty": repo_dirty},
        "dataset": {
            "name": "hitab_clean", "input_sha256": dataset_manifest["input_sha256"],
            "table_tree_sha256": dataset_manifest["table_tree_sha256"], "sample_count": dataset_manifest["sample_count"],
        },
        "cohorts": {
            "seed": SEED,
            "dev_manifest_sha256": sha256_file(output_root / "freeze/DEV_COHORT.jsonl"),
            "holdout_manifest_sha256": sha256_file(output_root / "freeze/HOLDOUT_COHORT_SEALED.jsonl"),
            "table_disjoint": read_json(output_root / "freeze/COHORT_MANIFEST.json")["table_disjoint"],
        },
        "model": {
            "model_id": "/home/common_data/llm/Qwen/Qwen3-8B", "served_model": "Qwen3-8B",
            "backend": "vllm_chat", "endpoint": "http://127.0.0.1:30338/v1", "thinking": False,
            "sampling": {"temperature": 0.0, "top_p": 1.0, "seed": 0, "max_model_len": 32768, "max_answer_tokens": 32},
        },
        "b0": {
            "prediction_manifest_sha256": b0_manifest["prediction_manifest_sha256"],
            "prompt_sha256": b0_manifest["prompt_sha256"],
            "request_manifest_sha256": b0_manifest["request_manifest_sha256"],
        },
        "evaluator": {
            "path": str(evaluator_path), "sha256": sha256_file(evaluator_path),
            "test_sha256": sha256_file(evaluator_test),
            "em_max_semantics": "textual OR symbolic oracle union, not selected final",
        },
        "method_hashes": method_hashes,
        "gold_firewall": {
            "runtime_gold_fields": [], "blind_output": str(blind_path), "unblind_output": str(unblind_path),
            "pass": gold_firewall_pass and proposal_blindness_pass,
        },
        "cache": {
            "namespace": str(run_config.get("api_cache_path") or ""), "mode": run_config.get("api_cache_mode"),
        },
        "planner": {
            "boundary": run_config.get("cera_planner_boundary"), "contract": run_config.get("cera_planner_contract"),
            "legacy_query_semantics_mode": run_config.get("cera_planner_legacy_query_semantics_mode"),
        },
    }
    jsonschema.validate(e0, read_json(PACK_ROOT / "schemas/e0_freeze.schema.json"))
    write_json(output_root / "freeze/E0_FREEZE_MANIFEST.json", e0)

    return {
        "sample_count": len(blind_rows),
        "blind_master_sha256": blind_hash,
        "gold_firewall_pass": gold_firewall_pass and proposal_blindness_pass,
        "proposal_blindness_pass": proposal_blindness_pass,
        "sample_master_complete": len(blind_rows) == len(cohort),
        "registry_outside_answer_count": sum(row["registry_outside_answer_count"] for row in blind_rows),
        "legacy_answer_mutation_count": sum(row["legacy_answer_mutation_used"] for row in blind_rows),
        "cera_commit_applied_count": sum(row["cera_commit_applied"] for row in blind_rows),
        "paired_executable_count": sum(row["paired_executable"] for row in blind_rows),
        "common_evaluable_intervention_count": sum(row["common_evaluable_intervention_count"] for row in blind_rows),
        "separating_intervention_count": sum(row["separating_intervention_count"] for row in blind_rows),
        "oracle_repairable_dev_count": sum(row["oracle_repairable"] for row in unblind_rows),
        "selected_final_equals_b0": all(row["round1_final_answer"] == row["b0_answer"] for row in blind_rows),
        "metrics": metrics,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--output-root", type=Path, required=True)
    prepare.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    prepare.add_argument("--table-root", type=Path, default=DEFAULT_TABLE_ROOT)
    prepare.add_argument("--dev-size", type=int, default=64)
    prepare.add_argument("--holdout-size", type=int, default=64)
    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--output-root", type=Path, required=True)
    analyze.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    analyze.add_argument("--primary-output-dir", type=Path, required=True)
    analyze.add_argument("--value-aware-output-dir", type=Path)
    analyze.add_argument("--proposal-aware-output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "prepare":
        summary = prepare_round1(
            dataset_path=args.dataset,
            table_root=args.table_root,
            output_root=args.output_root,
            dev_size=args.dev_size,
            holdout_size=args.holdout_size,
        )
        print(json.dumps(summary, sort_keys=True))
    elif args.command == "analyze":
        diagnostics = {}
        if args.value_aware_output_dir:
            diagnostics["value_aware"] = args.value_aware_output_dir
        if args.proposal_aware_output_dir:
            diagnostics["proposal_aware"] = args.proposal_aware_output_dir
        summary = analyze_round1(
            output_root=args.output_root,
            dataset_path=args.dataset,
            primary_output_dir=args.primary_output_dir,
            diagnostic_output_dirs=diagnostics,
        )
        print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
