#!/usr/bin/env python3
"""Reproduce CERTA V2 lineage and identity-only split manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from certa.active_v1.cohort import (  # noqa: E402
    ACTIVE_COHORT_DOMAIN,
    ACTIVE_COHORT_SEED,
    _project,
    _representatives,
    _table_content_hash,
)
from certa.reproducibility.canonical_json import canonical_json  # noqa: E402


V1_BRANCH = "research/certa-final-multidataset-method"
V1_COMMIT = "dc58b91284e14201418367757482208ee05bd64f"
SOURCE_PATHS = (
    "certa/active_v1/dataset_adapter_v1.py",
    "certa/active_v1/role_contract_v3.py",
    "certa/active_v1/planner_bridge_v3.py",
    "certa/active_v1/final_method_v1.py",
    "certa/active_v1/artifact_authority.py",
    "certa/planner/typed_planner.py",
    "certa/grounding/plan_closure.py",
    "certa/grounding/structural_resolvers.py",
    "certa/derivations/project.py",
    "certa/derivations/answer_equivalence.py",
    "certa/reproducibility/canonical_json.py",
    "tools/cscr_astra_eval.py",
)
V1_ARTIFACT_RELATIVE_PATHS = (
    "validation/BLIND_SAMPLE_MASTER.jsonl",
    "validation/REGISTRY.jsonl",
    "validation/PREDICTION_CLOSE.json",
    "validation/VALIDATION_CANDIDATE_METRICS.json",
    "validation/labels.released.jsonl",
    "data/hitab/validation_runtime_v3.jsonl",
    "data/hitab/holdout_runtime_v3.jsonl",
    "terminal/FINAL_TERMINAL_STATE.json",
    "terminal/SHA256SUMS.txt",
    "terminal/verified_git.bundle",
)
EXCLUSION_RELATIVE_PATHS = (
    "certa_round1_outputs/CERTA_R1_20260720T070718Z/freeze/DEV_COHORT.jsonl",
    "certa_egra_outputs/CERTA_EGRA_V0_20260720T152831Z/freeze/DEV_COHORT.jsonl",
    "certa_final_workspace/development/development_runtime.jsonl",
    "certa_final_firewall/public/fresh_dev_runtime.jsonl",
    "certa_final_firewall/public/fresh_holdout_runtime.jsonl",
    "certa_final_strict_v2/public/fresh_validation_runtime_v2.jsonl",
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _line_set_hash(values: Sequence[str] | set[str]) -> str:
    return _sha256_bytes(
        "".join(f"{value}\n" for value in sorted(set(values))).encode("utf-8")
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(canonical_json(dict(row)) + "\n" for row in rows),
        encoding="utf-8",
    )


def _git(repo: Path, *args: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout if binary else result.stdout.decode("utf-8").strip()


def build_v1_lineage(*, base: Path, repo: Path) -> dict[str, Any]:
    v1_root = (
        base
        / "certa_active_v1_outputs"
        / "CERTA_FINAL_MULTI_DATASET_ADAPTER_AND_METHOD_COMPLETION"
    )
    metrics_path = v1_root / "validation" / "VALIDATION_CANDIDATE_METRICS.json"
    terminal_path = v1_root / "terminal" / "FINAL_TERMINAL_STATE.json"
    manifest_path = v1_root / "terminal" / "SHA256SUMS.txt"
    bundle_path = v1_root / "terminal" / "verified_git.bundle"
    artifacts = [
        {"path": str(v1_root / relative), "sha256": _sha256_file(v1_root / relative)}
        for relative in V1_ARTIFACT_RELATIVE_PATHS
    ]
    sources = []
    for path in SOURCE_PATHS:
        blob = _git(repo, "show", f"{V1_COMMIT}:{path}", binary=True)
        assert isinstance(blob, bytes)
        sources.append(
            {
                "git_blob": str(_git(repo, "rev-parse", f"{V1_COMMIT}:{path}")),
                "path": path,
                "sha256": _sha256_bytes(blob.rstrip(b"\n")),
            }
        )
    metrics = _read_json(metrics_path)
    candidate_rows = metrics["candidates"]
    b0_row = next(row for row in candidate_rows if row["candidate_id"].endswith("+B0_KEEP"))
    deterministic = [
        row
        for row in candidate_rows
        if row["policy_id"] == "REGISTRY_DETERMINISTIC"
    ]
    bundle_verify = subprocess.run(
        ["git", "bundle", "verify", str(bundle_path)],
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    local_commit = str(_git(repo, "rev-parse", V1_COMMIT))
    remote_commit = str(
        _git(repo, "rev-parse", f"refs/remotes/origin/{V1_BRANCH}")
    )
    terminal = _read_json(terminal_path)
    return {
        "artifact_hashes": artifacts,
        "holdout_access": {
            "authorized": False,
            "holdout_output_files": 0,
            "labels_accessed": False,
            "model_calls": 0,
            "predictions_generated": False,
        },
        "negative_validation": {
            "b0_correct": b0_row["b0_correct"],
            "cera_calls": 0,
            "deterministic_non_b0_cw": sorted(
                set(row["CW"] for row in deterministic)
            ),
            "deterministic_non_b0_wc": max(row["WC"] for row in deterministic),
            "holdout_calls": 0,
            "sample_count": b0_row["row_count"],
        },
        "schema_version": "certa_v2_v1_negative_baseline_binding_v1",
        "source_blobs": sources,
        "v1_branch": V1_BRANCH,
        "v1_bundle": {
            "path": str(bundle_path),
            "sha256": _sha256_file(bundle_path),
            "verified": bundle_verify.returncode == 0,
            "verify_returncode": bundle_verify.returncode,
        },
        "v1_commit": V1_COMMIT,
        "v1_commit_verified": local_commit == V1_COMMIT,
        "v1_manifest": {
            "path": str(manifest_path),
            "sha256": _sha256_file(manifest_path),
        },
        "v1_remote_commit": remote_commit,
        "v1_terminal": {
            "path": str(terminal_path),
            "record": terminal,
            "sha256": _sha256_file(terminal_path),
        },
    }


def build_split_manifests(
    *, base: Path, repo: Path, runtime_output: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    table_root = repo / "dataset" / "hitab" / "tables" / "raw"
    selector_source = (
        base
        / "certa_egra_outputs"
        / "CERTA_EGRA_V0_20260720T152831Z"
        / "inputs"
        / "dev_identity_source.jsonl"
    )
    holdout_runtime = (
        base
        / "certa_final_strict_v2"
        / "public"
        / "fresh_holdout_runtime_v2.jsonl"
    )
    canonical_holdout = (
        base
        / "certa_active_v1_outputs"
        / "CERTA_FINAL_MULTI_DATASET_ADAPTER_AND_METHOD_COMPLETION"
        / "data"
        / "hitab"
        / "holdout_runtime_v3.jsonl"
    )
    exclusions = [base / relative for relative in EXCLUSION_RELATIVE_PATHS]
    exclusion_records = [
        {
            "path": str(path),
            "row_count": len(_read_jsonl(path)),
            "sha256": _sha256_file(path),
        }
        for path in exclusions
    ]
    development_rows = [
        row for path in exclusions for row in _read_jsonl(path)
    ]
    development_table_ids = {
        str(row.get("table_id") or "") for row in development_rows
    }
    if "" in development_table_ids:
        raise ValueError("development_exclusion_table_id_empty")
    table_hashes: dict[str, str] = {}
    development_content_hashes = {
        table_hashes.setdefault(
            table_id, _table_content_hash(table_root / f"{table_id}.json")
        )
        for table_id in development_table_ids
    }
    identity_rows = _read_jsonl(selector_source)
    projected = _project(
        identity_rows, table_root, "fresh_validation", table_hashes
    )
    candidates = _representatives(
        projected, development_table_ids, development_content_hashes
    )
    selected = candidates[:64]
    if len(selected) != 64:
        raise ValueError(f"fresh_validation_insufficient:{len(selected)}")
    for source_order, row in enumerate(selected):
        row["source_order"] = source_order
    runtime_rows = [dict(row["runtime"]) for row in selected]
    _write_jsonl(runtime_output, runtime_rows)

    holdout_rows = _read_jsonl(holdout_runtime)
    holdout_table_ids = {str(row["table_id"]) for row in holdout_rows}
    holdout_content_hashes = {
        _table_content_hash(table_root / f"{table_id}.json")
        for table_id in holdout_table_ids
    }
    selected_table_ids = {str(row["table_id"]) for row in selected}
    selected_content_hashes = {
        str(row["table_content_sha256"]) for row in selected
    }
    dev_validation_overlap = development_table_ids & selected_table_ids
    dev_holdout_overlap = development_table_ids & holdout_table_ids
    validation_holdout_overlap = selected_table_ids & holdout_table_ids
    validation_reserved_content_overlap = (
        selected_content_hashes & holdout_content_hashes
    )
    manifest = {
        "aitqa": {"use": "labeled_development_only"},
        "development": {
            "exclusion_source_count": len(exclusions),
            "exclusion_sources": exclusion_records,
            "policy": "all_previously_unblinded_or_answer_exposed",
            "sorted_table_ids_sha256": _line_set_hash(development_table_ids),
            "unique_table_count": len(development_table_ids),
        },
        "disjointness": {
            "development_holdout_table_overlap": len(dev_holdout_overlap),
            "development_validation_table_overlap": len(dev_validation_overlap),
            "pass": not any(
                (
                    dev_holdout_overlap,
                    dev_validation_overlap,
                    validation_holdout_overlap,
                    validation_reserved_content_overlap,
                )
            ),
            "validation_holdout_table_overlap": len(validation_holdout_overlap),
            "validation_reserved_content_overlap": len(
                validation_reserved_content_overlap
            ),
        },
        "fresh_validation": {
            "audit": {
                "candidate_content_class_count": len(candidates),
                "overlap_content_count": len(
                    selected_content_hashes & development_content_hashes
                ),
                "overlap_table_count": len(dev_validation_overlap),
                "selected_content_class_count": len(selected_content_hashes),
                "selected_count": len(selected),
                "selected_table_count": len(selected_table_ids),
            },
            "content_class_count": len(selected_content_hashes),
            "labels_status": "NOT_MATERIALIZED_OPERATOR_RELEASE_REQUIRED",
            "rows": len(runtime_rows),
            "runtime": str(runtime_output),
            "runtime_sha256": _sha256_file(runtime_output),
            "selection": selected,
            "sorted_content_hashes_sha256": _line_set_hash(selected_content_hashes),
            "sorted_sample_ids_sha256": _line_set_hash(
                {str(row["sample_id"]) for row in selected}
            ),
            "sorted_table_ids_sha256": _line_set_hash(selected_table_ids),
            "table_count": len(selected_table_ids),
        },
        "preserved_holdout": {
            "canonical_runtime": str(canonical_holdout),
            "canonical_runtime_sha256": _sha256_file(canonical_holdout),
            "eligible": True,
            "labels_accessed": False,
            "predictions_generated": False,
            "rows": len(holdout_rows),
            "runtime": str(holdout_runtime),
            "runtime_sha256": _sha256_file(holdout_runtime),
            "sorted_table_ids_sha256": _line_set_hash(holdout_table_ids),
            "table_count": len(holdout_table_ids),
        },
        "schema_version": "certa_v2_split_manifests_v1",
        "selector": {
            "domain_separator": ACTIVE_COHORT_DOMAIN.decode("utf-8"),
            "seed": ACTIVE_COHORT_SEED,
            "selector_source": "certa/active_v1/cohort.py",
            "selector_source_sha256": _sha256_file(
                repo / "certa" / "active_v1" / "cohort.py"
            ),
            "source": str(selector_source),
            "source_sha256": _sha256_file(selector_source),
        },
        "sstqa_zh": {"use": "post_freeze_robustness_only"},
    }
    return manifest, runtime_rows


def prepare_v2_state(
    *, output: Path, base: Path, repo: Path
) -> dict[str, Any]:
    output = Path(output)
    base = Path(base)
    repo = Path(repo)
    lineage = build_v1_lineage(base=base, repo=repo)
    runtime_output = output / "data" / "fresh_validation_runtime.jsonl"
    split_manifests, _ = build_split_manifests(
        base=base, repo=repo, runtime_output=runtime_output
    )
    _write_json(
        output / "lineage" / "V1_NEGATIVE_BASELINE_BINDING.json", lineage
    )
    _write_json(output / "data" / "SPLIT_MANIFESTS.json", split_manifests)
    return {"lineage": lineage, "split_manifests": split_manifests}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/home/hsh/ME/Table/EMNLP2026/certa_v2_outputs/"
            "CERTA_V2_BOUNDED_EXECUTABLE_PROOF_SEARCH"
        ),
    )
    parser.add_argument(
        "--base", type=Path, default=Path("/home/hsh/ME/Table/EMNLP2026")
    )
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    result = prepare_v2_state(output=args.output, base=args.base, repo=args.repo)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "v1_commit": result["lineage"]["v1_commit"],
                "fresh_validation_rows": result["split_manifests"][
                    "fresh_validation"
                ]["rows"],
                "holdout_labels_accessed": result["lineage"]["holdout_access"][
                    "labels_accessed"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
