#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, py_compile
from pathlib import Path

REQUIRED=(
"RESEARCH_DIRECTOR_DECISION.md","AUDIT_AND_DIAGNOSIS.md","GROUNDING_AUTHORITY_CONTRACT.md",
"SOURCE_AND_OUTPUT_BINDINGS.json","IMPLEMENTATION_BOUNDARY.md","OFFLINE_REPLAY_AND_GATE_PROTOCOL.md",
"SUBAGENT_PROTOCOL.md","TERMINAL_STATES.md","GOAL_MODE_COMMAND.txt","SHA256SUMS.txt")

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--pack-root",default=str(Path(__file__).resolve().parent)); a=ap.parse_args(); root=Path(a.pack_root)
    missing=[x for x in REQUIRED if not (root/x).is_file()]
    if missing: raise SystemExit("missing:"+"|".join(missing))
    for line in (root/"SHA256SUMS.txt").read_text().splitlines():
        if not line.strip(): continue
        expected,rel=line.split("  ",1)
        if rel!="SHA256SUMS.txt" and sha(root/rel)!=expected: raise SystemExit("checksum:"+rel)
    b=json.loads((root/"SOURCE_AND_OUTPUT_BINDINGS.json").read_text())
    if b["start_commit"]!="a1e8a7c761fc1f51b56d5d029a94901477eafb55": raise SystemExit("start_commit")
    goal=(root/"GOAL_MODE_COMMAND.txt").read_text()
    required=("assignment-level","zero-endpoint","27 prior","FREEZE_CERTA_ACTIVE_METHOD_READY_FOR_FINAL_DECISION_EXECUTION")
    for token in required:
        if token not in goal: raise SystemExit("goal_missing:"+token)
    forbidden=("modify Role V3","relax Gate C thresholds","first-match selection is allowed")
    for token in forbidden:
        if token in goal: raise SystemExit("forbidden:"+token)
    py_compile.compile(str(root/"validate_pack.py"),doraise=True)
    print("PASS CERTA_ACTIVE_V1_FINAL_ASSIGNMENT_LEVEL_GROUNDING_AUTHORITY_COMPLETION_PACK")
if __name__=="__main__": main()
