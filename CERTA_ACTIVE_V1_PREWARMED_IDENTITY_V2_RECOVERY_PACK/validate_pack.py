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
    "PREWARMED_IDENTITY_V2_PROTOCOL.md",
    "PREWARMED_IDENTITY_REFREEZE.py",
    "RUNTIME_CONTROLLER_ADOPT_V2.py",
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
    service = binding["current_service"]
    if service != {
        "pid": 1349993,
        "cuda_visible_devices": "3",
        "host": "127.0.0.1",
        "port": 30338,
        "endpoint": "http://127.0.0.1:30338",
        "api_base_url": "http://127.0.0.1:30338/v1",
        "served_model_name": "Qwen3-8B",
        "model_root": "/home/common_data/llm/Qwen/Qwen3-8B",
        "max_model_len": 32768,
        "diagnostic_normalized_cmdline_sha256": "df765d9af5cc5aafc518428dd4a545ebee34bfd67a512e9d3f9d8d21ec337977",
    }:
        raise SystemExit("current_service_binding")
    old = binding["superseded_identity"]
    if old["expected_cmdline_sha256"] != "d86efb9407136fc0127fd63e1387632b70f29c0bd5cdf971884fb345413ac3c4":
        raise SystemExit("superseded_cmdline_binding")
    if old["blocked_failure_sha256"] != "56dbe5495c9a5d95deb1354347ed210ca65bb489ed0975e4a418e81523a932f0":
        raise SystemExit("blocked_failure_binding")
    if service["diagnostic_normalized_cmdline_sha256"] == old["expected_cmdline_sha256"]:
        raise SystemExit("mixed_lineage_not_expressed")

    refreeze = (root / "PREWARMED_IDENTITY_REFREEZE.py").read_text(encoding="utf-8")
    controller = (root / "RUNTIME_CONTROLLER_ADOPT_V2.py").read_text(encoding="utf-8")
    forbidden_lifecycle = ("subprocess.Popen", "os.kill(", "os.killpg", "signal.SIG", ".terminate(", ".kill(", "--safetensors-load-strategy")
    for name, text in (("refreeze", refreeze), ("controller", controller)):
        for token in forbidden_lifecycle:
            if token in text:
                raise SystemExit(f"{name}_lifecycle_authority:{token}")
    for token in (
        "PREWARMED_VLLM_SERVICE_IDENTITY_V2.json",
        "log_identity",
        "diagnostic_normalized_cmdline_sha256",
        "root_or_descendant_owns_port",
        "ProxyHandler({})",
        "PREWARMED_IDENTITY_V2_FROZEN",
    ):
        if token not in refreeze:
            raise SystemExit("refreeze_missing:" + token)
    for token in (
        "bootstrap-adopt-existing-service-v2",
        "V2_IDENTITY_VALIDATION_FAILURE.json",
        "v2_cmdline_sha256_mismatch",
        "verify_log_identity",
        "atomic_finalize",
        "ProxyHandler({})",
    ):
        if token not in controller:
            raise SystemExit("controller_missing:" + token)

    goal = (root / "GOAL_MODE_COMMAND.txt").read_text(encoding="utf-8")
    if not goal.startswith("/goal CERTA_ACTIVE_V1_PREWARMED_IDENTITY_V2_REFREEZE_AND_RESUME_FINAL_DAG"):
        raise SystemExit("goal_identity")
    if len(goal.split()) > 1800:
        raise SystemExit("goal_too_long")

    py_compile.compile(str(root / "PREWARMED_IDENTITY_REFREEZE.py"), doraise=True)
    py_compile.compile(str(root / "RUNTIME_CONTROLLER_ADOPT_V2.py"), doraise=True)
    py_compile.compile(str(root / "validate_pack.py"), doraise=True)
    print("PASS CERTA_ACTIVE_V1_PREWARMED_IDENTITY_V2_RECOVERY_PACK")


if __name__ == "__main__":
    main()
