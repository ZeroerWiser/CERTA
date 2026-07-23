#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, py_compile
from pathlib import Path

REQUIRED=[
"RESEARCH_DIRECTOR_DECISION.md","SOURCE_AND_LINEAGE_BINDINGS.json","RUNTIME_LIFECYCLE_POLICY.md","RUNTIME_CONTROLLER.py","CHECKPOINT_RESUME_PROTOCOL.md","SCIENTIFIC_FAILURE_TAXONOMY.md","SCHEMA_PROJECTION_FREEZE.json","INTEGRATION16_PROTOCOL.md","MATCHED_CONSTRUCTOR_PROTOCOL.md","BLIND_DECISION_PROTOCOL.md","HOLDOUT_PROTOCOL.md","ABSTRACT_CLAIM_EVIDENCE_LEDGER.md","AGENTS.md","REQUIRED_ARTIFACTS.md","EXECUTION_DAG.md","GOAL_MODE_COMMAND.txt","SHA256SUMS.txt"]
FORBIDDEN_TOKENS=["ROLE_V3_FRESH_QUESTIONS","ROLE_V3_FRESH_LABELS.json","role cards:","new operation ontology"]
SCAN_FILES=[x for x in REQUIRED if x not in {"validate_pack.py","SHA256SUMS.txt"}]

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--pack-root",default=str(Path(__file__).resolve().parent)); a=ap.parse_args(); root=Path(a.pack_root).resolve()
    missing=[x for x in REQUIRED if not (root/x).is_file()]
    if missing: raise SystemExit("missing:"+"|".join(missing))
    for line in (root/"SHA256SUMS.txt").read_text().splitlines():
        if not line.strip(): continue
        expected,rel=line.split("  ",1)
        if rel=="SHA256SUMS.txt": continue
        if sha(root/rel)!=expected: raise SystemExit("checksum:"+rel)
    binding=json.loads((root/"SOURCE_AND_LINEAGE_BINDINGS.json").read_text())
    if binding["head"]!="a1e8a7c761fc1f51b56d5d029a94901477eafb55": raise SystemExit("method_head")
    freeze=json.loads((root/"SCHEMA_PROJECTION_FREEZE.json").read_text())
    if freeze["projection_edits_authorized"] is not False: raise SystemExit("projection_not_frozen")
    goal=(root/"GOAL_MODE_COMMAND.txt").read_text()
    if len(goal.split())>1250: raise SystemExit("goal_too_long")
    all_text="\n".join((root/name).read_text(errors="ignore") for name in SCAN_FILES if (root/name).is_file())
    for token in FORBIDDEN_TOKENS:
        if token in all_text: raise SystemExit("forbidden_content:"+token)
    py_compile.compile(str(root/"RUNTIME_CONTROLLER.py"),doraise=True)
    py_compile.compile(str(root/"validate_pack.py"),doraise=True)
    print("PASS CERTA_ACTIVE_V1_RUNTIME_STABLE_FINAL_SCIENTIFIC_EXECUTION_MICRO_REVISED_PACK")
if __name__=="__main__": main()
