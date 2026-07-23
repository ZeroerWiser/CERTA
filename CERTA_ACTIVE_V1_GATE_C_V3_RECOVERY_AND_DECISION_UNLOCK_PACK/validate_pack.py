#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import py_compile
from pathlib import Path

REQUIRED = (
    "RESEARCH_DIRECTOR_DECISION.md",
    "SOURCE_AND_OUTPUT_BINDINGS.json",
    "GATE_C_V3_RECOVERY.py",
    "GATE_RECOVERY_PROTOCOL.md",
    "SUBAGENT_PROTOCOL.md",
    "CONDITIONAL_NEXT_STAGE.md",
    "GOAL_MODE_COMMAND.txt",
    "REQUIRED_ARTIFACTS.md",
    "CHANGE_NOTE.md",
    "SHA256SUMS.txt",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack-root", default=str(Path(__file__).resolve().parent))
    args = parser.parse_args()
    root = Path(args.pack_root).resolve()
    missing = [name for name in REQUIRED if not (root / name).is_file()]
    if missing:
        raise SystemExit("missing:" + "|".join(missing))
    for line in (root / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        if relative != "SHA256SUMS.txt" and sha(root / relative) != expected:
            raise SystemExit("checksum:" + relative)
    binding = json.loads((root / "SOURCE_AND_OUTPUT_BINDINGS.json").read_text(encoding="utf-8"))
    if binding["commit"] != "a6818af3c157f3416bdff84925e003e36b3c4583":
        raise SystemExit("method_commit")
    if binding["failed_manifest_sha256"] != "419766026150f9d2e3b826aa67f6e784e32f9a6da1addcd56454c2589487cd0e":
        raise SystemExit("failed_manifest")
    if binding["failed_manifest_entries"] != 538:
        raise SystemExit("manifest_entries")
    if binding["gate_thresholds"] != {
        "c2_paired_min": 8,
        "paired_gain_min": 4,
        "c2_registry_complete_paired_min": 6,
        "registry_gain_min": 3,
        "paired_tables_min": 4,
        "role_compatible_precision": 1.0,
    }:
        raise SystemExit("gate_thresholds")
    tool = (root / "GATE_C_V3_RECOVERY.py").read_text(encoding="utf-8")
    for forbidden in ("urllib.request", "requests.", "OpenAIChatGenerator", "subprocess.Popen", "os.kill(", "os.killpg"):
        if forbidden in tool:
            raise SystemExit("forbidden_runtime_token:" + forbidden)
    for required in ("sys.path.insert(0, str(repo))", "from tools.compute_certa_active_constructor_gate_v3 import THRESHOLDS, compute_gate", "verify_failed_replay", "forbidden_replay_access"):
        if required not in tool:
            raise SystemExit("missing_recovery_token:" + required)
    goal = (root / "GOAL_MODE_COMMAND.txt").read_text(encoding="utf-8")
    if not goal.startswith("/goal CERTA_ACTIVE_V1_GATE_C_V3_ZERO_CALL_RECOVERY_AND_DECISION_UNLOCK"):
        raise SystemExit("goal_identity")
    if "Do not execute `tools/compute_certa_active_constructor_gate_v3.py` directly" not in goal:
        raise SystemExit("direct_script_prohibition")
    py_compile.compile(str(root / "GATE_C_V3_RECOVERY.py"), doraise=True)
    py_compile.compile(str(root / "validate_pack.py"), doraise=True)
    print("PASS CERTA_ACTIVE_V1_GATE_C_V3_RECOVERY_AND_DECISION_UNLOCK_PACK")


if __name__ == "__main__":
    main()
