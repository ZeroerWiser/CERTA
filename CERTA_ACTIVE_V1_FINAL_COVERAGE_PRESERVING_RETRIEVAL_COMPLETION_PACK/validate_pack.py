#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

PACK_NAME = "CERTA_ACTIVE_V1_FINAL_COVERAGE_PRESERVING_RETRIEVAL_COMPLETION_PACK"
REQUIRED = {
    "RESEARCH_DIRECTOR_DECISION.md",
    "METHOD_CONTRACT.md",
    "IMPLEMENTATION_BOUNDARY.md",
    "SCIENTIFIC_EXECUTION_PROTOCOL.md",
    "SUBAGENT_PROTOCOL.md",
    "SOURCE_AND_EVIDENCE_BINDINGS.json",
    "FINAL_METHOD_CLAIM_LEDGER.md",
    "REQUIRED_ARTIFACTS.md",
    "GOAL_MODE_COMMAND.txt",
    "CHANGE_NOTE.md",
    "SHA256SUMS.txt",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack-root", required=True)
    args = parser.parse_args()
    root = Path(args.pack_root).resolve()
    errors: list[str] = []
    if root.name != PACK_NAME:
        errors.append("pack_name_mismatch")
    missing = sorted(name for name in REQUIRED if not (root / name).is_file())
    if missing:
        errors.append("missing:" + ",".join(missing))

    binding_path = root / "SOURCE_AND_EVIDENCE_BINDINGS.json"
    if binding_path.is_file():
        try:
            binding = json.loads(binding_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            errors.append("binding_json_invalid")
        else:
            if binding.get("base_commit") != "a6818af3c157f3416bdff84925e003e36b3c4583":
                errors.append("base_commit_mismatch")
            if binding.get("target_branch") != "research/certa-active-v1-coverage-preserving-retrieval-final":
                errors.append("target_branch_mismatch")
            if len(binding.get("source_git_blobs") or {}) < 7:
                errors.append("source_binding_incomplete")
            if binding.get("prior_assignment_authority_replay", {}).get("manifest_entries") != 538:
                errors.append("prior_replay_binding_incomplete")
            if binding.get("prior_gate_recovery", {}).get("gate_sha256") != "21b5cf7a07e7f157f4f2c45305f5be85682e12857ea2f9267d79e843845a1c67":
                errors.append("prior_gate_binding_mismatch")

    goal = (root / "GOAL_MODE_COMMAND.txt").read_text(encoding="utf-8") if (root / "GOAL_MODE_COMMAND.txt").is_file() else ""
    required_phrases = (
        "/goal CERTA_ACTIVE_V1_FINAL_COVERAGE_PRESERVING_RETRIEVAL_COMPLETION",
        "corrected C2 must retain byte-identical `schema_nodes` and `schema_edges` to C1",
        "Endpoint calls for C0 and C1 must be zero",
        "temperature 0",
        "zero retries",
        "FREEZE_CERTA_ACTIVE_METHOD_READY_FOR_FINAL_DECISION_EXECUTION",
        "FREEZE_CERTA_ACTIVE_COVERAGE_PRESERVING_RETRIEVAL_VALID_NO_PAIRED_CONTRAST",
        "Decision, CERA, dev gold, unblinding and holdout are forbidden",
        "Do not modify Gate thresholds",
        "This is the last authorized CERTA Active V1 method-completion Goal",
    )
    for phrase in required_phrases:
        if phrase not in goal:
            errors.append("goal_missing:" + phrase)

    manifest = root / "SHA256SUMS.txt"
    if manifest.is_file():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                expected, relative = line.split("  ", 1)
            except ValueError:
                errors.append("manifest_line_invalid")
                continue
            path = root / relative
            if not path.is_file() or sha256(path) != expected:
                errors.append("manifest_mismatch:" + relative)

    if errors:
        raise SystemExit("\n".join(sorted(set(errors))))
    print("PASS " + PACK_NAME)


if __name__ == "__main__":
    main()
