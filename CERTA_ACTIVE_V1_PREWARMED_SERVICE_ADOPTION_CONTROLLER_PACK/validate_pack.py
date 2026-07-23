#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import py_compile
from pathlib import Path

REQUIRED = (
    "RESEARCH_DIRECTOR_DECISION.md",
    "SOURCE_BINDINGS.json",
    "PREWARMED_SERVICE_ADOPTION_PROTOCOL.md",
    "ATOMIC_FINALIZATION_PROTOCOL.md",
    "RUNTIME_CONTROLLER_ADOPT.py",
    "REQUIRED_ARTIFACTS.md",
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
        if relative == "SHA256SUMS.txt":
            continue
        if sha(root / relative) != expected:
            raise SystemExit("checksum:" + relative)

    binding = json.loads((root / "SOURCE_BINDINGS.json").read_text(encoding="utf-8"))
    if binding["head"] != "a1e8a7c761fc1f51b56d5d029a94901477eafb55":
        raise SystemExit("method_head")
    if binding["prewarmed_evidence_archive"]["sha256"] != "694fcfb57bf7215cbd21b55a6860de5e86582c1312de311ab78a8502a6308a64":
        raise SystemExit("evidence_archive_sha")
    expected = binding["prewarmed_identity"]
    if expected != {
        "path": "/home/hsh/ME/Table/EMNLP2026/certa_runtime_evidence/QWEN3_8B_PREWARMED_SERVICE_20260723/PREWARMED_VLLM_SERVICE_IDENTITY.json",
        "pid": 1349993,
        "cuda_visible_devices": "3",
        "host": "127.0.0.1",
        "port": 30338,
        "endpoint": "http://127.0.0.1:30338",
        "api_base_url": "http://127.0.0.1:30338/v1",
        "served_model_name": "Qwen3-8B",
        "model_root": "/home/common_data/llm/Qwen/Qwen3-8B",
        "max_model_len": 32768,
    }:
        raise SystemExit("prewarmed_identity_binding")

    controller = (root / "RUNTIME_CONTROLLER_ADOPT.py").read_text(encoding="utf-8")
    forbidden_controller_tokens = (
        "subprocess.Popen", "os.kill(", "os.killpg", "signal.SIG", "terminate()", "--safetensors-load-strategy",
    )
    for token in forbidden_controller_tokens:
        if token in controller:
            raise SystemExit("controller_lifecycle_authority:" + token)
    required_controller_tokens = (
        "bootstrap-adopt-existing-service", "PREWARMED_VLLM_SERVICE_IDENTITY.json",
        "readiness_three_rounds", "atomic_finalize", "RUNTIME_CONTROLLER_STATE.json",
    )
    for token in required_controller_tokens:
        if token not in controller:
            raise SystemExit("controller_missing:" + token)

    goal = (root / "GOAL_MODE_COMMAND.txt").read_text(encoding="utf-8")
    if "/goal CERTA_ACTIVE_V1_ADOPT_PREWARMED_SERVICE_AND_RUN_FINAL_SCIENTIFIC_DAG" not in goal:
        raise SystemExit("goal_identity")
    if "--safetensors-load-strategy" in goal:
        raise SystemExit("forbidden_safetensors_strategy")

    py_compile.compile(str(root / "RUNTIME_CONTROLLER_ADOPT.py"), doraise=True)
    py_compile.compile(str(root / "validate_pack.py"), doraise=True)
    print("PASS CERTA_ACTIVE_V1_PREWARMED_SERVICE_ADOPTION_CONTROLLER_PACK")


if __name__ == "__main__":
    main()
