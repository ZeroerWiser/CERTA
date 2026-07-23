#!/usr/bin/env python3
"""Compute and finalize Gate C V3 from frozen replay artifacts without model access."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PACK = Path(__file__).resolve().parent
BIND = json.loads((PACK / "SOURCE_AND_OUTPUT_BINDINGS.json").read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(args: list[str], *, cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, text=True, capture_output=True, check=check)


def git(repo: Path, *args: str) -> str:
    return run(["git", *args], cwd=repo).stdout.strip()


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def verify_repo(repo: Path) -> dict[str, Any]:
    run(["git", "fetch", "--prune", "origin"], cwd=repo)
    actual = {
        "branch": git(repo, "branch", "--show-current"),
        "head": git(repo, "rev-parse", "HEAD"),
        "origin_head": git(repo, "rev-parse", "origin/" + BIND["origin_branch"]),
        "clean": git(repo, "status", "--porcelain") == "",
    }
    expected = {
        "branch": BIND["branch"],
        "head": BIND["commit"],
        "origin_head": BIND["commit"],
        "clean": True,
    }
    if actual != expected:
        raise RuntimeError("repository_binding_mismatch:" + json.dumps(actual, sort_keys=True))
    blobs = {}
    for path, expected_blob in BIND["source_git_blobs"].items():
        actual_blob = git(repo, "rev-parse", f"HEAD:{path}")
        if actual_blob != expected_blob:
            raise RuntimeError(f"source_blob_mismatch:{path}:{actual_blob}")
        blobs[path] = actual_blob
    return {"repository": actual, "source_git_blobs": blobs}


def verify_failed_replay(root: Path) -> dict[str, str]:
    bound = {
        "terminal/FINAL_TERMINAL_STATE.json": BIND["failed_terminal_sha256"],
        "replay/REPLAY_FAILURE.json": BIND["failure_record_sha256"],
        "terminal/SHA256SUMS.txt": BIND["failed_manifest_sha256"],
        "terminal/CERTA_ACTIVE_V1_GROUNDING_AUTHORITY_FINAL.bundle": BIND["failed_bundle_sha256"],
    }
    for relative, expected in bound.items():
        path = root / relative
        if not path.is_file() or sha(path) != expected:
            raise RuntimeError("bound_failed_replay_mismatch:" + relative)
    manifest = root / "terminal/SHA256SUMS.txt"
    rows = [line for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != BIND["failed_manifest_entries"]:
        raise RuntimeError("failed_manifest_entry_count_mismatch")
    for line in rows:
        expected, relative = line.split("  ", 1)
        path = root / relative
        if not path.is_file() or sha(path) != expected:
            raise RuntimeError("failed_manifest_content_mismatch:" + relative)
    terminal = json.loads((root / "terminal/FINAL_TERMINAL_STATE.json").read_text(encoding="utf-8"))
    if terminal.get("terminal_state") != BIND["failed_terminal"]:
        raise RuntimeError("failed_terminal_identity_mismatch")
    missing = [relative for relative in BIND["required_replay_inputs"] if not (root / relative).is_file()]
    if missing:
        raise RuntimeError("required_replay_input_missing:" + "|".join(missing))
    access = json.loads((root / "logs/OFFLINE_REPLAY_ACCESS_LEDGER.json").read_text(encoding="utf-8"))
    nonzero = {key: access.get(key) for key in BIND["forbidden_accesses"] if access.get(key) != 0}
    if nonzero:
        raise RuntimeError("forbidden_replay_access:" + json.dumps(nonzero, sort_keys=True))
    return {relative: sha(root / relative) for relative in BIND["required_replay_inputs"]}


def compute(repo: Path, replay_root: Path, recovery_root: Path) -> None:
    if recovery_root.exists():
        raise RuntimeError("recovery_root_already_exists")
    repository = verify_repo(repo)
    inputs = verify_failed_replay(replay_root)
    staging = recovery_root.with_name(recovery_root.name + ".staging")
    if staging.exists():
        raise RuntimeError("recovery_staging_already_exists")
    staging.mkdir(parents=True)
    try:
        for relative in BIND["required_replay_inputs"]:
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(replay_root / relative, target)
        for relative in (
            "terminal/FINAL_TERMINAL_STATE.json",
            "replay/REPLAY_FAILURE.json",
            "terminal/SHA256SUMS.txt",
        ):
            target = staging / "historical_failed_replay" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(replay_root / relative, target)
        audits = replay_root / "audits"
        if audits.is_dir():
            shutil.copytree(audits, staging / "prior_audits")

        sys.path.insert(0, str(repo))
        from tools.compute_certa_active_constructor_gate_v3 import THRESHOLDS, compute_gate

        if dict(THRESHOLDS) != BIND["gate_thresholds"]:
            raise RuntimeError("gate_threshold_drift")
        gate = compute_gate(
            identities=jsonl(staging / "constructor/DEV64_IDENTITIES.blind.jsonl"),
            role_records=jsonl(staging / "constructor/DEV64_ROLE_V3_RECORDS.blind.jsonl"),
            groundings=jsonl(staging / "constructor/RAW_GROUNDINGS_V3.jsonl"),
            derivations=jsonl(staging / "constructor/RAW_DERIVATIONS.jsonl"),
            registry=jsonl(staging / "constructor/FROZEN_REGISTRY.jsonl"),
            cost_ledger=json.loads((staging / "constructor/CONSTRUCTOR_COST_LEDGER.json").read_text(encoding="utf-8")),
            allow_fixture=False,
        )
        gate["recovery_execution"] = {
            "schema_version": "certa_active_v1_gate_c_v3_recovery_execution_v1",
            "method_sha": BIND["commit"],
            "entrypoint": "imported_compute_gate",
            "repo_root_injected_before_import": True,
            "model_calls": 0,
            "gold_accesses": 0,
            "sealed_label_accesses": 0,
            "decision_calls": 0,
            "source_artifacts": inputs,
            "computed_at_unix": time.time(),
        }
        write_json(staging / "constructor/CONSTRUCTOR_GATE_C_V3.json", gate)
        write_json(staging / "recovery/GATE_C_V3_RECOVERY_RECORD.json", {
            "schema_version": "certa_active_v1_gate_c_v3_recovery_record_v1",
            "repository": repository,
            "failed_replay_root": str(replay_root),
            "failed_replay_manifest_sha256": BIND["failed_manifest_sha256"],
            "gate_sha256": sha(staging / "constructor/CONSTRUCTOR_GATE_C_V3.json"),
            "gate_pass": bool(gate["pass"]),
            "failure_reasons": list(gate["failure_reasons"]),
            "endpoint_calls": 0,
            "gold_accesses": 0,
            "sealed_label_accesses": 0,
            "created_at_unix": time.time(),
        })
        os.replace(staging, recovery_root)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    print("GATE_C_V3_PASS" if gate["pass"] else "GATE_C_V3_VALID_FAILURE")


def finalize(repo: Path, recovery_root: Path) -> None:
    repository = verify_repo(repo)
    gate_path = recovery_root / "constructor/CONSTRUCTOR_GATE_C_V3.json"
    if not gate_path.is_file():
        raise RuntimeError("gate_c_v3_missing")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    reports = (
        "reviews/GATE_ENTRYPOINT_ROOT_CAUSE_AUDIT.md",
        "reviews/GATE_C_V3_ARTIFACT_AUDIT.md",
        "reviews/GATE_C_V3_HOSTILE_AAAI_AUDIT.md",
    )
    missing = [relative for relative in reports if not (recovery_root / relative).is_file()]
    if missing:
        raise RuntimeError("gate_recovery_audit_missing:" + "|".join(missing))
    terminal_state = (
        "FREEZE_CERTA_ACTIVE_METHOD_READY_FOR_FINAL_DECISION_EXECUTION"
        if gate.get("pass") is True
        else "FREEZE_CERTA_ACTIVE_GROUNDING_AUTHORITY_VALID_NO_PAIRED_CONTRAST"
    )
    terminal = recovery_root / "terminal"
    if terminal.exists():
        raise RuntimeError("terminal_already_exists")
    terminal.mkdir(parents=True)
    bundle = terminal / "CERTA_ACTIVE_V1_GATE_C_V3_RECOVERY.bundle"
    run(["git", "bundle", "create", str(bundle), BIND["branch"]], cwd=repo)
    verified = run(["git", "bundle", "verify", str(bundle)], cwd=repo)
    write_json(terminal / "BUNDLE_VERIFICATION.json", {
        "verified": True,
        "bundle_sha256": sha(bundle),
        "stdout": verified.stdout.strip(),
        "stderr": verified.stderr.strip(),
    })
    required = [*BIND["required_replay_inputs"], "constructor/CONSTRUCTOR_GATE_C_V3.json", "recovery/GATE_C_V3_RECOVERY_RECORD.json", *reports]
    write_json(terminal / "REQUIRED_ARTIFACTS.json", {
        "schema_version": "certa_active_v1_gate_c_v3_recovery_required_artifacts_v1",
        "terminal_state": terminal_state,
        "artifacts": [{"path": relative, "status": "PRESENT", "sha256": sha(recovery_root / relative)} for relative in required],
    })
    write_json(terminal / "FINAL_METHOD_FREEZE_MANIFEST.json", {
        "schema_version": "certa_active_v1_gate_c_v3_recovery_freeze_v1",
        "terminal_state": terminal_state,
        "method_sha": BIND["commit"],
        "branch": BIND["branch"],
        "repository": repository,
        "gate_sha256": sha(gate_path),
        "gate_pass": bool(gate["pass"]),
        "model_calls": 0,
        "gold_accesses": 0,
        "sealed_label_accesses": 0,
        "final_decision_execution_authorized": bool(gate["pass"]),
        "created_at_unix": time.time(),
    })
    write_json(terminal / "FINAL_TERMINAL_STATE.json", {
        "schema_version": "certa_active_v1_gate_c_v3_recovery_terminal_v1",
        "terminal_state": terminal_state,
        "method_sha": BIND["commit"],
        "method_frozen": True,
        "gate_c_v3_pass": bool(gate["pass"]),
        "decision_authorized": bool(gate["pass"]),
        "created_at_unix": time.time(),
    })
    (terminal / "TERMINAL_REPORT.md").write_text(
        "# CERTA Active V1 Gate C V3 Recovery\n\n"
        f"Terminal: `{terminal_state}`\n\n"
        f"Method: `{BIND['commit']}`\n\n"
        f"Gate C V3 pass: `{gate['pass']}`\n\n"
        f"Failure reasons: `{gate.get('failure_reasons', [])}`\n\n"
        "Endpoint, gold, sealed-label and Decision calls: `0`\n",
        encoding="utf-8",
    )
    files = sorted(path for path in recovery_root.rglob("*") if path.is_file() and path.name != "SHA256SUMS.txt")
    (terminal / "SHA256SUMS.txt").write_text(
        "".join(f"{sha(path)}  {path.relative_to(recovery_root)}\n" for path in files),
        encoding="utf-8",
    )
    print(terminal_state)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("compute", "finalize"))
    parser.add_argument("--repo", default=BIND["repository"])
    parser.add_argument("--replay-root", default=BIND["failed_replay_root"])
    parser.add_argument("--recovery-root", default=BIND["recovery_root"])
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    replay_root = Path(args.replay_root).resolve()
    recovery_root = Path(args.recovery_root).resolve()
    if args.command == "compute":
        compute(repo, replay_root, recovery_root)
    else:
        finalize(repo, recovery_root)


if __name__ == "__main__":
    main()
