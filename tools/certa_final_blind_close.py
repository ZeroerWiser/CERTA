#!/usr/bin/env python3
"""Fail-closed prediction materialization and close for CERTA final cohorts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from certa.active_v1.answer_authority import active_answer_hash
from certa.active_v1.final_method_v1 import VARIANT_IDS
from certa.reproducibility.canonical_json import canonical_json


POLICIES = ("B0_KEEP", "REGISTRY_DETERMINISTIC", "CERA_VALIDATED")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(canonical_json(dict(row)) + "\n" for row in rows),
        encoding="utf-8",
    )
    return sha256(path)


def write_json(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return sha256(path)


def exact_runtime_master_join(
    runtime: Sequence[Mapping[str, Any]],
    master: Sequence[Mapping[str, Any]],
) -> None:
    for name, rows, key in (
        ("runtime", runtime, "id"),
        ("master", master, "sample_id"),
    ):
        ids = [str(row.get(key) or "") for row in rows]
        if any(not value for value in ids) or len(ids) != len(set(ids)):
            raise ValueError(f"{name}_identity_not_unique_complete")
    if [row["id"] for row in runtime] != [row["sample_id"] for row in master]:
        raise ValueError("runtime_master_ordered_id_mismatch")
    if any(
        runtime_row["table_id"] != master_row["table_id"]
        for runtime_row, master_row in zip(runtime, master)
    ):
        raise ValueError("runtime_master_table_id_mismatch")
    forbidden = {"gold", "gold_answer", "gold_answers", "labels", "correct"}
    for row in master:
        if forbidden & set(row):
            raise ValueError(f"blind_master_gold_field:{row['sample_id']}")


def close_split(
    *,
    output: Path,
    split: str,
    runtime_path: Path,
    freeze_path: Path,
) -> dict[str, Any]:
    if split not in {"validation", "holdout"}:
        raise ValueError(f"invalid_split:{split}")
    runtime = read_jsonl(runtime_path)
    master_path = output / split / "BLIND_SAMPLE_MASTER.jsonl"
    master = read_jsonl(master_path)
    exact_runtime_master_join(runtime, master)
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
    ).strip()
    origin = subprocess.check_output(
        ["git", "rev-parse", f"origin/{freeze['branch']}"],
        cwd=REPO,
        text=True,
    ).strip()
    if head != origin or head != freeze["method_commit"]:
        raise ValueError("close_commit_not_frozen_and_pushed")
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=REPO, text=True,
    ).strip():
        raise ValueError("close_repository_dirty")
    artifacts = {}
    registries = []
    for sample in master:
        for variant in sample["variants"]:
            registries.extend(variant["registry_entries"])
    registry_path = output / split / "REGISTRY.jsonl"
    artifacts["REGISTRY.jsonl"] = write_jsonl(registry_path, registries)
    for variant_id in VARIANT_IDS:
        for policy_id in POLICIES:
            rows = []
            for runtime_row, sample in zip(runtime, master):
                variant = next(
                    item for item in sample["variants"]
                    if item["variant_id"] == variant_id
                )
                policy = variant["policies"][policy_id]
                answer = policy["selected_answer"]
                rows.append({
                    "schema_version": "certa_final_selected_final_v1",
                    "split": split,
                    "sample_id": sample["sample_id"],
                    "table_id": sample["table_id"],
                    "variant_id": variant_id,
                    "policy_id": policy_id,
                    "action": policy["action"],
                    "selected_answer": answer,
                    "selected_answer_hash": active_answer_hash(answer),
                    "runtime_table_artifact_sha256": runtime_row[
                        "table_artifact_sha256"
                    ],
                })
            name = f"SELECTED_FINALS_{variant_id}_{policy_id}.jsonl"
            artifacts[name] = write_jsonl(output / split / name, rows)
    primary_name = (
        f"SELECTED_FINALS_{freeze['primary_constructor']}_"
        f"{freeze['primary_policy']}.jsonl"
    )
    close = {
        "schema_version": "certa_prediction_close_v3",
        "split": split,
        "method_commit": head,
        "runtime_sha256": sha256(runtime_path),
        "selected_finals_sha256": artifacts[primary_name],
        "primary_policy": (
            f"{freeze['primary_constructor']}+{freeze['primary_policy']}"
        ),
        "closed_at": datetime.now(timezone.utc).isoformat(),
        "runtime_path": str(runtime_path),
        "runtime_row_count": len(runtime),
        "ordered_sample_ids_sha256": hashlib.sha256(
            canonical_json([row["id"] for row in runtime]).encode("utf-8")
        ).hexdigest(),
        "blind_sample_master_path": str(master_path),
        "blind_sample_master_sha256": sha256(master_path),
        "freeze_path": str(freeze_path),
        "freeze_sha256": sha256(freeze_path),
        "registry_sha256": artifacts["REGISTRY.jsonl"],
        "all_policy_artifact_sha256": artifacts,
        "primary_selected_finals_path": str(output / split / primary_name),
        "primary_selected_finals_name": primary_name,
        "repository_clean": True,
        "repository_pushed": True,
        "label_access_before_close": False,
    }
    close_path = output / split / "PREDICTION_CLOSE.json"
    close_sha = write_json(close_path, close)
    return {
        "status": "PASS",
        "prediction_close_path": str(close_path),
        "prediction_close_sha256": close_sha,
        "primary_selected_finals_sha256": artifacts[primary_name],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "holdout"), required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    args = parser.parse_args()
    print(canonical_json(close_split(
        output=args.output.resolve(),
        split=args.split,
        runtime_path=args.runtime.resolve(),
        freeze_path=args.freeze.resolve(),
    )))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
