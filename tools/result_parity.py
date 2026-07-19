#!/usr/bin/env python3
"""Field-level comparison of two saved, non-blind CERTA run directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_FILES = ("predictions.jsonl", "predictions.debug.jsonl", "metrics.json", "run_config.json")
VOLATILE_KEYS = {"timestamp", "started_at", "finished_at", "host", "hostname", "pid", "process_id", "elapsed_seconds", "duration_seconds", "temporary_directory", "temp_dir"}

IGNORED_LOCATIONS = {
    "metrics.json.efficiency_metrics.avg_llm_generation_seconds",
    "metrics.json.runtime.samples_per_second",
    "metrics.json.runtime.total_seconds",
    "run_config.json.created_at",
    "run_config.json.output_dir",
    "run_config.json.run_id",
    "run_config.json.modules.llm_input_audit_file",
}


def _ignored_location(path: str) -> bool:
    if path in IGNORED_LOCATIONS:
        return True
    for artifact in ("predictions.jsonl", "predictions.debug.jsonl"):
        if not path.startswith(artifact + "["):
            continue
        suffixes = (
            ".efficiency.llm_generation_seconds", ".efficiency.non_llm_preparation_seconds",
            ".efficiency.pipeline_recorded_seconds", ".efficiency.post_llm_finalize_seconds",
            ".llm_generation_seconds", ".non_llm_preparation_seconds", ".pipeline_recorded_seconds",
            ".post_llm_finalize_seconds", ".llm_input_audit.audit_file",
        )
        return path.endswith(suffixes) or path.rsplit("[", 1)[-1].endswith("].run_id")
    return False


def _load(path: Path) -> Any:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize(value: Any, allow_top_level_run_id: bool = False, depth: int = 0) -> Any:
    if isinstance(value, dict):
        return {
            key: _normalize(item, allow_top_level_run_id, depth + 1)
            for key, item in value.items()
            if key not in VOLATILE_KEYS and not (allow_top_level_run_id and depth == 0 and key == "run_id")
        }
    if isinstance(value, list):
        return [_normalize(item, allow_top_level_run_id, depth + 1) for item in value]
    return value


def _differences(source: Any, target: Any, path: str = "", sample_id: str | None = None) -> list[dict[str, Any]]:
    if _ignored_location(path):
        return []
    if isinstance(source, dict) and isinstance(target, dict):
        found: list[dict[str, Any]] = []
        effective_sample_id = str(source.get("sample_id", target.get("sample_id", sample_id or "")))
        for key in sorted(set(source) | set(target)):
            found.extend(_differences(source.get(key), target.get(key), f"{path}.{key}" if path else key, effective_sample_id))
        return found
    if isinstance(source, list) and isinstance(target, list):
        found = []
        for index in range(max(len(source), len(target))):
            left = source[index] if index < len(source) else None
            right = target[index] if index < len(target) else None
            found.extend(_differences(left, right, f"{path}[{index}]", sample_id))
        return found
    if source != target:
        return [{
            "sample_id": sample_id or "",
            "artifact": path.split(".", 1)[0].split("[", 1)[0],
            "field": path.rsplit(".", 1)[-1],
            "location": path,
            "source": source,
            "target": target,
        }]
    return []


def compare_run_directories(source_dir: Path, target_dir: Path) -> dict[str, Any]:
    missing = [name for name in REQUIRED_FILES if not (source_dir / name).is_file() or not (target_dir / name).is_file()]
    if missing:
        return {"status": "UNVERIFIED", "reason": "missing_required_artifact", "missing_files": missing, "differences": []}
    differences: list[dict[str, Any]] = []
    for name in REQUIRED_FILES:
        allow_top_level_run_id = name == "run_config.json"
        differences.extend(_differences(_normalize(_load(source_dir / name), allow_top_level_run_id), _normalize(_load(target_dir / name), allow_top_level_run_id), name))
    return {"status": "PASS" if not differences else "FAIL", "differences": differences}


def _prediction_index(rows: Any, label: str) -> tuple[dict[str, dict[str, Any]] | None, list[dict[str, Any]]]:
    if not isinstance(rows, list):
        return None, [{"field": "predictions.jsonl", "reason": f"{label}_predictions_not_jsonl"}]
    indexed: dict[str, dict[str, Any]] = {}
    differences: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            differences.append({"field": "predictions.jsonl", "reason": f"{label}_row_not_object", "index": index})
            continue
        sample_id = str(row.get("sample_id", row.get("id", ""))).strip()
        if not sample_id:
            differences.append({"field": "sample_id", "reason": f"{label}_missing_sample_id", "index": index})
            continue
        if sample_id in indexed:
            differences.append({"field": "sample_id", "reason": f"{label}_duplicate_sample_id", "sample_id": sample_id})
            continue
        indexed[sample_id] = row
    return indexed, differences


def compare_answer_equivalence(source_dir: Path, target_dir: Path) -> dict[str, Any]:
    """Compare only row cardinality, sample identities, LLM answers, and final answers."""
    source_path = source_dir / "predictions.jsonl"
    target_path = target_dir / "predictions.jsonl"
    if not source_path.is_file() or not target_path.is_file():
        missing = [str(path) for path in (source_path, target_path) if not path.is_file()]
        return {"status": "UNVERIFIED", "reason": "missing_predictions", "missing_files": missing, "differences": []}
    source_rows = _load(source_path)
    target_rows = _load(target_path)
    source_index, differences = _prediction_index(source_rows, "source")
    target_index, target_differences = _prediction_index(target_rows, "target")
    differences.extend(target_differences)
    if source_index is None or target_index is None:
        return {"status": "UNVERIFIED", "reason": "invalid_predictions", "differences": differences}
    if len(source_rows) != len(target_rows):
        differences.append({"field": "row_count", "source": len(source_rows), "target": len(target_rows)})
    source_ids = set(source_index)
    target_ids = set(target_index)
    if source_ids != target_ids:
        differences.append({
            "field": "sample_id_set",
            "source_only": sorted(source_ids - target_ids),
            "target_only": sorted(target_ids - source_ids),
        })
    for sample_id in sorted(source_ids & target_ids):
        for field in ("llm_answer", "final_answer"):
            if source_index[sample_id].get(field) != target_index[sample_id].get(field):
                differences.append({
                    "sample_id": sample_id,
                    "field": field,
                    "source": source_index[sample_id].get(field),
                    "target": target_index[sample_id].get(field),
                })
    return {
        "status": "PASS" if not differences else "FAIL",
        "source_row_count": len(source_rows),
        "target_row_count": len(target_rows),
        "differences": differences,
    }


def compare_dual_parity(source_dir: Path, target_dir: Path) -> dict[str, Any]:
    """Keep answer equivalence and full-artifact parity as separate non-substitutable reports."""
    return {
        "answer_equivalent": compare_answer_equivalence(source_dir, target_dir),
        "artifact_strict_parity": compare_run_directories(source_dir, target_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_run", type=Path)
    parser.add_argument("target_run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare_dual_parity(args.source_run, args.target_run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    strict = report["artifact_strict_parity"]
    answers = report["answer_equivalent"]
    print(json.dumps({
        "ANSWER_EQUIVALENT": answers["status"],
        "ARTIFACT_STRICT_PARITY": strict["status"],
        "strict_differences": len(strict.get("differences", [])),
    }, ensure_ascii=False))
    if strict["status"] == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
