#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, py_compile
from pathlib import Path

REQUIRED = (
    "RESEARCH_DIRECTOR_DECISION.md",
    "INPUT_BINDINGS.md",
    "SET_VALUED_GROUNDING_AUTHORITY_CONTRACT.md",
    "OFFLINE_REPLAY_AND_GATES.md",
    "GATE_Q.json",
    "SUBAGENTS.md",
    "EXECUTION_DAG.md",
    "GOAL_MODE_COMMAND.txt",
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
    gate = json.loads((root / "GATE_Q.json").read_text(encoding="utf-8"))
    if gate["old_method_sha"] != "a1e8a7c761fc1f51b56d5d029a94901477eafb55":
        raise SystemExit("method_sha")
    if gate["gold_access_allowed"] is not False or gate["endpoint_calls_before_gate_q"] != 0:
        raise SystemExit("firewall")
    if gate["gate_q"]["all_original_thresholds_unchanged"] is not True:
        raise SystemExit("threshold_drift")
    combined = "\n".join((root / name).read_text(encoding="utf-8") for name in REQUIRED if name.endswith((".md", ".txt")))
    required_tokens = (
        "SET_VALUED_GROUNDING_AUTHORITY_FINALIZATION",
        "FINITE_SET_OF_INDIVIDUALLY_UNIQUE_BINDINGS",
        "NO_SECOND_GROUNDING_FIX",
        "FREEZE_CERTA_ACTIVE_SET_VALUED_AUTHORITY_INSUFFICIENT",
    )
    for token in required_tokens:
        if token not in combined:
            raise SystemExit("missing_contract:" + token)
    forbidden = ("confidence score module", "new retriever", "agent voting module")
    for token in forbidden:
        if token in combined.lower():
            raise SystemExit("forbidden_component:" + token)
    py_compile.compile(str(root / "validate_pack.py"), doraise=True)
    print("PASS CERTA_ACTIVE_V1_SET_VALUED_GROUNDING_AUTHORITY_FINALIZATION_PACK")


if __name__ == "__main__":
    main()
