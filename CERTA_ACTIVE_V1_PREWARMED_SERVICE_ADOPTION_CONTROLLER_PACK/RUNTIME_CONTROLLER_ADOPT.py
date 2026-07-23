#!/usr/bin/env python3
"""Adopt a user-managed prewarmed vLLM service and run the frozen CERTA DAG."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Mapping

PACK = Path(__file__).resolve().parent
BIND = json.loads((PACK / "SOURCE_BINDINGS.json").read_text(encoding="utf-8"))


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path | str) -> str:
    return sha_bytes(Path(path).read_bytes())


def write_json(path: Path | str, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(args: list[str], *, cwd: Path | str | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, text=True, capture_output=True, check=check)


def git(repo: Path, *args: str) -> str:
    return run(["git", *args], cwd=repo).stdout.strip()


def pick(record: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in record:
            return record[name]
    raise KeyError("missing_identity_field:" + "|".join(names))


def normalize_cmdline(value: str) -> str:
    return " ".join(str(value).replace("\x00", " ").split())


def initial_state(output_root: Path) -> dict[str, Any]:
    state = {
        "schema_version": "certa_active_v1_prewarmed_controller_state_v1",
        "status": "ADOPTION_VALIDATION_IN_PROGRESS",
        "method_commit": BIND["head"],
        "service_managed_by": "USER_RUNTIME_OPERATOR",
        "controller_service_lifecycle_authority": False,
        "scientific_output_root": str(output_root),
        "created_at_unix": time.time(),
        "current_stage": "bootstrap-adopt-existing-service",
    }
    write_json(output_root / "runtime/RUNTIME_CONTROLLER_STATE.json", state)
    return state


def save_state(output_root: Path, state: Mapping[str, Any]) -> None:
    write_json(output_root / "runtime/RUNTIME_CONTROLLER_STATE.json", dict(state))


def verify_repository(repo: Path) -> dict[str, Any]:
    git(repo, "fetch", "--prune", "origin")
    actual = {
        "branch": git(repo, "branch", "--show-current"),
        "head": git(repo, "rev-parse", "HEAD"),
        "origin_head": git(repo, "rev-parse", "origin/" + BIND["branch"]),
        "origin_master": git(repo, "rev-parse", "origin/master"),
        "clean": git(repo, "status", "--porcelain") == "",
    }
    expected = {
        "branch": BIND["branch"],
        "head": BIND["head"],
        "origin_head": BIND["origin_head"],
        "origin_master": BIND["origin_master"],
        "clean": True,
    }
    if actual != expected:
        raise RuntimeError("repository_binding_mismatch:" + canonical(actual))
    blobs = {}
    for path, expected_blob in BIND["frozen_method_blobs"].items():
        actual_blob = git(repo, "rev-parse", f"HEAD:{path}")
        if actual_blob != expected_blob:
            raise RuntimeError(f"method_blob_mismatch:{path}:{actual_blob}")
        blobs[path] = actual_blob
    return {"repository": actual, "git_blobs": blobs}


def identity_from_archive(archive: Path, deployed: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_archive_sha = BIND["prewarmed_evidence_archive"]["sha256"]
    if not archive.is_file() or archive.is_symlink():
        raise RuntimeError("prewarmed_archive_not_regular")
    actual_archive_sha = sha_file(archive)
    if actual_archive_sha != expected_archive_sha:
        raise RuntimeError("prewarmed_archive_sha256_mismatch")
    if not deployed.is_file() or deployed.is_symlink():
        raise RuntimeError("prewarmed_identity_not_regular")
    with tarfile.open(archive, "r:gz") as bundle:
        members = [m for m in bundle.getmembers() if m.isfile() and m.name.endswith("/PREWARMED_VLLM_SERVICE_IDENTITY.json")]
        if len(members) != 1:
            raise RuntimeError("prewarmed_archive_identity_member_count")
        handle = bundle.extractfile(members[0])
        if handle is None:
            raise RuntimeError("prewarmed_archive_identity_unreadable")
        archived_bytes = handle.read()
    deployed_bytes = deployed.read_bytes()
    if sha_bytes(archived_bytes) != sha_bytes(deployed_bytes):
        raise RuntimeError("deployed_identity_not_archive_identical")
    identity = json.loads(deployed_bytes)
    evidence = {
        "archive_path": str(archive),
        "archive_sha256": actual_archive_sha,
        "archive_identity_member": members[0].name,
        "identity_path": str(deployed),
        "identity_sha256": sha_bytes(deployed_bytes),
    }
    return identity, evidence


def process_socket_owns_port(pid: int, port: int) -> bool:
    inodes: set[str] = set()
    fd_root = Path(f"/proc/{pid}/fd")
    for fd in fd_root.iterdir():
        try:
            target = os.readlink(fd)
        except OSError:
            continue
        if target.startswith("socket:[") and target.endswith("]"):
            inodes.add(target[8:-1])
    expected_port = f"{port:04X}"
    for table in (Path("/proc/net/tcp"), Path("/proc/net/tcp6")):
        if not table.is_file():
            continue
        for line in table.read_text().splitlines()[1:]:
            fields = line.split()
            if len(fields) < 10:
                continue
            local = fields[1]
            state = fields[3]
            inode = fields[9]
            local_port = local.rsplit(":", 1)[-1].upper()
            if local_port == expected_port and state == "0A" and inode in inodes:
                return True
    return False


def verify_process_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    frozen = BIND["prewarmed_identity"]
    pid = int(pick(identity, "pid"))
    if pid != int(frozen["pid"]):
        raise RuntimeError("prewarmed_pid_mismatch")
    proc = Path(f"/proc/{pid}")
    if not proc.is_dir():
        raise RuntimeError("prewarmed_process_missing")

    start_ticks = (proc / "stat").read_text().split()[21]
    expected_ticks = str(pick(identity, "process_start_ticks", "start_ticks"))
    if start_ticks != expected_ticks:
        raise RuntimeError("prewarmed_start_ticks_mismatch")

    cmdline = normalize_cmdline((proc / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace"))
    frozen_cmdline = normalize_cmdline(str(pick(identity, "cmdline", "normalized_cmdline")))
    if cmdline != frozen_cmdline:
        raise RuntimeError("prewarmed_cmdline_mismatch")
    cmdline_sha = sha_bytes(cmdline.encode())
    expected_cmdline_sha = identity.get("cmdline_sha256")
    if expected_cmdline_sha and cmdline_sha != expected_cmdline_sha:
        raise RuntimeError("prewarmed_cmdline_sha256_mismatch")

    executable = os.readlink(proc / "exe")
    expected_executable = str(pick(identity, "executable", "python_executable", "exe"))
    if Path(executable).resolve() != Path(expected_executable).resolve():
        raise RuntimeError("prewarmed_executable_mismatch")

    environ = {}
    for item in (proc / "environ").read_bytes().split(b"\0"):
        if b"=" in item:
            key, value = item.split(b"=", 1)
            environ[key.decode("utf-8", "replace")] = value.decode("utf-8", "replace")
    actual_gpu = environ.get("CUDA_VISIBLE_DEVICES", "")
    expected_gpu = str(pick(identity, "cuda_visible_devices", "CUDA_VISIBLE_DEVICES"))
    if actual_gpu != expected_gpu or actual_gpu != frozen["cuda_visible_devices"]:
        raise RuntimeError("prewarmed_gpu_identity_mismatch")

    if not process_socket_owns_port(pid, int(frozen["port"])):
        raise RuntimeError("prewarmed_pid_does_not_own_port")

    required_tokens = [
        str(frozen["model_root"]), "--served-model-name", str(frozen["served_model_name"]),
        "--port", str(frozen["port"]), "--max-model-len", str(frozen["max_model_len"]),
    ]
    for token in required_tokens:
        if token not in cmdline:
            raise RuntimeError("prewarmed_cmdline_required_token_missing:" + token)

    identity_model = str(pick(identity, "model_root", "model_path"))
    identity_model_name = str(pick(identity, "served_model_name", "model_name"))
    identity_max_len = int(pick(identity, "max_model_len"))
    if identity_model != frozen["model_root"] or identity_model_name != frozen["served_model_name"] or identity_max_len != frozen["max_model_len"]:
        raise RuntimeError("prewarmed_model_identity_mismatch")

    log_file = Path(str(pick(identity, "log_file", "server_log", "log_path")))
    if not log_file.is_file() or log_file.is_symlink() or not os.access(log_file, os.R_OK):
        raise RuntimeError("prewarmed_log_not_readable_regular")
    log_stat = log_file.stat()
    if "log_device" in identity and int(identity["log_device"]) != log_stat.st_dev:
        raise RuntimeError("prewarmed_log_device_mismatch")
    if "log_inode" in identity and int(identity["log_inode"]) != log_stat.st_ino:
        raise RuntimeError("prewarmed_log_inode_mismatch")

    return {
        "pid": pid,
        "process_start_ticks": start_ticks,
        "cmdline": cmdline,
        "cmdline_sha256": cmdline_sha,
        "executable": executable,
        "cuda_visible_devices": actual_gpu,
        "host": frozen["host"],
        "port": frozen["port"],
        "served_model_name": frozen["served_model_name"],
        "model_root": frozen["model_root"],
        "max_model_len": frozen["max_model_len"],
        "log_file": str(log_file),
        "log_device": log_stat.st_dev,
        "log_inode": log_stat.st_ino,
        "log_size_at_adoption": log_stat.st_size,
    }


def http_status(url: str, timeout: int = 10) -> tuple[int, bytes]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.status, response.read()


def readiness_three_rounds(output_root: Path) -> list[dict[str, Any]]:
    endpoint = BIND["prewarmed_identity"]["endpoint"]
    ledger = []
    for index in range(1, 4):
        health_status, health_body = http_status(endpoint + "/health")
        model_status, model_body = http_status(endpoint + "/v1/models")
        models = json.loads(model_body or b"{}")
        ids = [str(item.get("id")) for item in models.get("data", [])]
        row = {
            "round": index,
            "health_status": health_status,
            "health_body_bytes": len(health_body),
            "models_status": model_status,
            "model_ids": ids,
            "passed": health_status == 200 and model_status == 200 and ids == [BIND["prewarmed_identity"]["served_model_name"]],
            "checked_at_unix": time.time(),
        }
        ledger.append(row)
        write_json(output_root / "runtime/PREWARMED_READINESS_LEDGER.json", ledger)
        if not row["passed"]:
            raise RuntimeError("prewarmed_readiness_round_failed:" + str(index))
        if index < 3:
            time.sleep(1)
    return ledger


def load_runner(repo: Path, output_root: Path):
    runner_path = repo / BIND["frozen_runner"]
    spec = importlib.util.spec_from_file_location("certa_frozen_completion_runner", runner_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("frozen_runner_import_spec_failed")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(repo))
    spec.loader.exec_module(module)
    module.OUT = output_root
    return module


def controller_state(output_root: Path) -> dict[str, Any]:
    return json.loads((output_root / "runtime/RUNTIME_CONTROLLER_STATE.json").read_text(encoding="utf-8"))


def verify_adopted_service(output_root: Path) -> dict[str, Any]:
    state = controller_state(output_root)
    identity_path = Path(state["identity_path"])
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    current = verify_process_identity(identity)
    frozen = state["service_identity"]
    for field in ("pid", "process_start_ticks", "cmdline_sha256", "executable", "cuda_visible_devices", "port", "served_model_name", "model_root", "max_model_len", "log_file", "log_device", "log_inode"):
        if current[field] != frozen[field]:
            raise RuntimeError("adopted_service_identity_drift:" + field)
    status, body = http_status(BIND["prewarmed_identity"]["endpoint"] + "/v1/models")
    ids = [str(item.get("id")) for item in json.loads(body or b"{}").get("data", [])]
    if status != 200 or ids != [BIND["prewarmed_identity"]["served_model_name"]]:
        raise RuntimeError("adopted_service_model_check_failed")
    return current


def required_artifact_paths() -> list[str]:
    values = []
    for line in (PACK / "REQUIRED_ARTIFACTS.md").read_text(encoding="utf-8").splitlines():
        value = line.strip().strip("`")
        if value and "/" in value and not value.startswith("#"):
            values.append(value)
    return values


def fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_finalize(repo: Path, output_root: Path, terminal_state: str, *, stage: str, error: BaseException | None = None) -> None:
    terminal = output_root / "terminal"
    if terminal.exists():
        raise RuntimeError("terminal_already_exists")
    staging = output_root / (".terminal_staging_" + uuid.uuid4().hex)
    staging.mkdir(parents=True)
    try:
        state = controller_state(output_root)
        runner = load_runner(repo, output_root)
        error_record = None
        if error is not None:
            error_record = {"type": type(error).__name__, "message_sha256": sha_bytes(str(error).encode())}

        cost = runner.cost_ledger() if hasattr(runner, "cost_ledger") else {"status": "NOT_AVAILABLE"}
        write_json(staging / "COST_LEDGER.json", cost)
        write_json(staging / "FINAL_TERMINAL_STATE.json", {
            "schema_version": "certa_active_v1_prewarmed_adoption_terminal_v1",
            "terminal_state": terminal_state,
            "stage": stage,
            "method_sha": BIND["head"],
            "method_frozen": True,
            "service_user_managed": True,
            "controller_service_lifecycle_authority": False,
            "error": error_record,
            "created_at_unix": time.time(),
        })
        write_json(staging / "FINAL_METHOD_FREEZE_MANIFEST.json", {
            "schema_version": "certa_active_v1_final_method_freeze_v1",
            "terminal_state": terminal_state,
            "method_sha": BIND["head"],
            "repository_clean": git(repo, "status", "--porcelain") == "",
            "service_identity": state.get("service_identity"),
            "created_at_unix": time.time(),
        })

        disposition = []
        for relative in required_artifact_paths():
            path = output_root / relative
            disposition.append({
                "path": relative,
                "status": "PRESENT" if path.is_file() else "NOT_REACHED",
                "sha256": sha_file(path) if path.is_file() else None,
            })
        write_json(staging / "REQUIRED_ARTIFACTS.json", {
            "schema_version": "certa_active_v1_required_artifacts_v1",
            "terminal_state": terminal_state,
            "artifacts": disposition,
        })

        bundle = staging / "CERTA_ACTIVE_V1_FINAL_COMPLETION.bundle"
        run(["git", "bundle", "create", str(bundle), BIND["branch"]], cwd=repo)
        run(["git", "bundle", "verify", str(bundle)], cwd=repo)

        report = [
            "# CERTA Active V1 Prewarmed-Service Scientific Execution",
            "",
            f"Terminal: `{terminal_state}`",
            f"Stage: `{stage}`",
            f"Method: `{BIND['head']}`",
            "Service lifecycle owner: `USER_RUNTIME_OPERATOR`",
            f"Logical calls: `{cost.get('logical_calls', 'NOT_RECORDED')}`",
        ]
        if error_record:
            report.append(f"Error: `{error_record['type']}` / `{error_record['message_sha256']}`")
        (staging / "TERMINAL_REPORT.md").write_text("\n\n".join(report) + "\n", encoding="utf-8")

        checksum_rows = []
        for path in sorted(output_root.rglob("*")):
            if not path.is_file() or staging in path.parents or path.name == "SHA256SUMS.txt":
                continue
            checksum_rows.append(f"{sha_file(path)}  {path.relative_to(output_root)}")
        for path in sorted(staging.rglob("*")):
            if path.is_file() and path.name != "SHA256SUMS.txt":
                checksum_rows.append(f"{sha_file(path)}  terminal/{path.relative_to(staging)}")
        (staging / "SHA256SUMS.txt").write_text("\n".join(checksum_rows) + "\n", encoding="utf-8")

        for path in staging.rglob("*"):
            if path.is_file():
                fsync_file(path)
        fsync_dir(staging)
        os.replace(staging, terminal)
        fsync_dir(output_root)
    except Exception:
        raise RuntimeError("BLOCKED_ATOMIC_FINALIZATION_FAILED:" + str(staging))


def classify_runtime_failure(error: BaseException) -> bool:
    text = (type(error).__name__ + ":" + str(error)).lower()
    return any(token in text for token in ("connection", "enginedead", "apitimeout", "process_missing", "identity_drift", "model_check"))


def bootstrap(args: argparse.Namespace) -> None:
    repo = Path(args.repo).resolve()
    output_root = Path(args.output_root).resolve()
    archive = Path(args.evidence_archive).resolve()
    identity_path = Path(args.identity).resolve()
    if output_root.exists():
        raise RuntimeError("new_output_root_already_exists")
    output_root.mkdir(parents=True)
    state = initial_state(output_root)
    try:
        repository = verify_repository(repo)
        identity, evidence = identity_from_archive(archive, identity_path)
        service = verify_process_identity(identity)
        readiness = readiness_three_rounds(output_root)
        state.update({
            "status": "PREWARMED_SERVICE_ADOPTED",
            "identity_path": str(identity_path),
            "evidence": evidence,
            "repository": repository,
            "service_identity": service,
            "readiness_rounds": len(readiness),
            "current_stage": "frozen-freeze-and-preflight",
        })
        save_state(output_root, state)
        write_json(output_root / "runtime/PREWARMED_SERVICE_ADOPTION.json", state)

        runner = load_runner(repo, output_root)
        runner.freeze()
        preflight_rc = runner.preflight()
        if preflight_rc:
            raise RuntimeError("frozen_preflight_failed:" + str(preflight_rc))
        state.update({"status": "SCIENTIFIC_DAG_READY", "current_stage": "Integration16", "preflight_pass": True})
        save_state(output_root, state)
    except Exception as error:
        state.update({"status": "ADOPTION_FAILED", "failure_type": type(error).__name__, "failure_sha256": sha_bytes(str(error).encode())})
        save_state(output_root, state)
        terminal = "BLOCKED_PREWARMED_SERVICE_READINESS_FAILED" if "readiness" in str(error).lower() else "BLOCKED_PREWARMED_SERVICE_IDENTITY_FAILED"
        atomic_finalize(repo, output_root, terminal, stage="bootstrap-adopt-existing-service", error=error)
        raise


def stage_terminal(command: str) -> str:
    return {
        "integration16": "FREEZE_CERTA_ACTIVE_INTEGRATION_FAILED",
        "constructor64": "FREEZE_CERTA_ACTIVE_CONSTRUCTOR_FAILED",
        "decision-dev": "FREEZE_CERTA_ACTIVE_DECISION_FAILED",
        "unblind-dev": "FREEZE_CERTA_ACTIVE_DECISION_FAILED",
        "holdout-blind": "FREEZE_CERTA_ACTIVE_HOLDOUT_FAILED",
        "unblind-holdout": "FREEZE_CERTA_ACTIVE_HOLDOUT_FAILED",
    }[command]


def call_frozen_stage(runner: Any, command: str) -> int:
    if command == "integration16":
        return runner.constructor("dev", 16, runner.ARMS)
    if command == "constructor64":
        return runner.constructor("dev", 64, runner.ARMS)
    if command == "decision-dev":
        return runner.decision("dev")
    if command == "unblind-dev":
        return runner.unblind("dev")
    if command == "holdout-blind":
        return runner.holdout_blind()
    if command == "unblind-holdout":
        return runner.unblind("holdout")
    raise ValueError("unknown_stage:" + command)


def execute_stage(args: argparse.Namespace) -> None:
    repo = Path(args.repo).resolve()
    output_root = Path(args.output_root).resolve()
    state = controller_state(output_root)
    if state.get("status") not in {"SCIENTIFIC_DAG_READY", "SCIENTIFIC_STAGE_COMPLETE"}:
        raise RuntimeError("controller_not_scientific_ready")
    state.update({"current_stage": args.command, "status": "SCIENTIFIC_STAGE_RUNNING"})
    save_state(output_root, state)
    try:
        current = verify_adopted_service(output_root)
        write_json(output_root / f"runtime/SERVICE_IDENTITY_BEFORE_{args.command}.json", current)
        runner = load_runner(repo, output_root)
        rc = call_frozen_stage(runner, args.command)
        if rc:
            raise RuntimeError(f"scientific_stage_nonzero:{args.command}:{rc}")
        current_after = verify_adopted_service(output_root)
        write_json(output_root / f"runtime/SERVICE_IDENTITY_AFTER_{args.command}.json", current_after)
        state.update({"current_stage": args.command, "status": "SCIENTIFIC_STAGE_COMPLETE", "last_completed_stage": args.command})
        save_state(output_root, state)
    except Exception as error:
        state.update({"status": "STAGE_FAILED", "failure_type": type(error).__name__, "failure_sha256": sha_bytes(str(error).encode())})
        save_state(output_root, state)
        terminal = "BLOCKED_PREWARMED_SERVICE_LOST" if classify_runtime_failure(error) else stage_terminal(args.command)
        atomic_finalize(repo, output_root, terminal, stage=args.command, error=error)
        raise


def finalize(args: argparse.Namespace) -> None:
    repo = Path(args.repo).resolve()
    output_root = Path(args.output_root).resolve()
    verify_adopted_service(output_root)
    atomic_finalize(repo, output_root, args.state, stage="finalize")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=(
        "bootstrap-adopt-existing-service", "integration16", "constructor64", "decision-dev",
        "unblind-dev", "holdout-blind", "unblind-holdout", "finalize",
    ))
    parser.add_argument("--repo", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--evidence-archive", default=BIND["prewarmed_evidence_archive"]["path"])
    parser.add_argument("--identity", default=BIND["prewarmed_identity"]["path"])
    parser.add_argument("--state")
    args = parser.parse_args()
    if args.command == "bootstrap-adopt-existing-service":
        bootstrap(args)
    elif args.command == "finalize":
        if not args.state:
            parser.error("--state is required for finalize")
        finalize(args)
    else:
        execute_stage(args)


if __name__ == "__main__":
    main()
