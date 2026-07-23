#!/usr/bin/env python3
"""Adopt a V2-frozen user-managed vLLM service and run the frozen CERTA DAG."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
import shutil
import stat
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


class IdentityError(RuntimeError):
    def __init__(self, code: str, **details: Any) -> None:
        self.code = code
        self.details = details
        super().__init__(code)


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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
        raise IdentityError("repository_binding_mismatch", actual=actual, expected=expected)
    blobs = {}
    for path, expected_blob in BIND["frozen_method_blobs"].items():
        actual_blob = git(repo, "rev-parse", f"HEAD:{path}")
        if actual_blob != expected_blob:
            raise IdentityError("method_blob_mismatch", path=path, actual=actual_blob, expected=expected_blob)
        blobs[path] = actual_blob
    return {"repository": actual, "git_blobs": blobs}


def archive_identity(archive: Path, manifest: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if not archive.is_file() or archive.is_symlink():
        raise IdentityError("v2_archive_not_regular", path=str(archive))
    actual_archive_sha = sha_file(archive)
    if actual_archive_sha != manifest.get("archive_sha256"):
        raise IdentityError("v2_archive_sha256_mismatch", actual=actual_archive_sha, expected=manifest.get("archive_sha256"))
    expected_member = str(manifest.get("identity_member"))
    with tarfile.open(archive, "r:gz") as bundle:
        try:
            member = bundle.getmember(expected_member)
        except KeyError as exc:
            raise IdentityError("v2_identity_member_missing", member=expected_member) from exc
        if not member.isfile():
            raise IdentityError("v2_identity_member_not_file", member=expected_member)
        handle = bundle.extractfile(member)
        if handle is None:
            raise IdentityError("v2_identity_member_unreadable", member=expected_member)
        identity_bytes = handle.read()
    identity_sha = sha_bytes(identity_bytes)
    if identity_sha != manifest.get("identity_sha256"):
        raise IdentityError("v2_identity_sha256_mismatch", actual=identity_sha, expected=manifest.get("identity_sha256"))
    identity = json.loads(identity_bytes)
    return identity, {
        "archive_path": str(archive),
        "archive_sha256": actual_archive_sha,
        "identity_member": expected_member,
        "identity_sha256": identity_sha,
    }


def proc_argv(pid: int) -> list[str]:
    return [x.decode("utf-8", "replace") for x in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0") if x]


def proc_environ(pid: int) -> dict[str, str]:
    result = {}
    for item in Path(f"/proc/{pid}/environ").read_bytes().split(b"\0"):
        if b"=" in item:
            key, value = item.split(b"=", 1)
            result[key.decode("utf-8", "replace")] = value.decode("utf-8", "replace")
    return result


def argument_value(argv: list[str], flag: str) -> str | None:
    for index, item in enumerate(argv):
        if item == flag and index + 1 < len(argv):
            return argv[index + 1]
        if item.startswith(flag + "="):
            return item.split("=", 1)[1]
    return None


def descendants(root_pid: int) -> set[int]:
    ppid_map = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            ppid_map[int(entry.name)] = int((entry / "stat").read_text().split()[3])
        except Exception:
            continue
    result = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, ppid in ppid_map.items():
            if ppid in result and pid not in result:
                result.add(pid)
                changed = True
    return result


def socket_inodes(pid: int) -> set[str]:
    result = set()
    root = Path(f"/proc/{pid}/fd")
    if not root.is_dir():
        return result
    for fd in root.iterdir():
        try:
            target = os.readlink(fd)
        except OSError:
            continue
        if target.startswith("socket:[") and target.endswith("]"):
            result.add(target[8:-1])
    return result


def listening_inodes(port: int) -> set[str]:
    result = set()
    expected = f"{port:04X}"
    for table in (Path("/proc/net/tcp"), Path("/proc/net/tcp6")):
        if not table.is_file():
            continue
        for line in table.read_text().splitlines()[1:]:
            fields = line.split()
            if len(fields) >= 10 and fields[1].rsplit(":", 1)[-1].upper() == expected and fields[3] == "0A":
                result.add(fields[9])
    return result


def current_port_ownership(pid: int, port: int) -> dict[str, Any]:
    listening = listening_inodes(port)
    tree = sorted(descendants(pid))
    owners = {}
    for current in tree:
        owned = sorted(socket_inodes(current) & listening)
        if owned:
            owners[str(current)] = owned
    return {
        "listening_socket_inodes": sorted(listening),
        "process_tree_pids": tree,
        "owners_in_process_tree": owners,
        "root_pid_owns_port": bool(socket_inodes(pid) & listening),
        "root_or_descendant_owns_port": bool(owners),
    }


def fd_identity(pid: int, fd: int) -> dict[str, Any]:
    path = Path(f"/proc/{pid}/fd/{fd}")
    target = os.readlink(path)
    info = os.stat(path)
    return {
        "fd": fd,
        "target": target,
        "device": info.st_dev,
        "inode": info.st_ino,
        "mode": info.st_mode,
        "is_regular_file": stat.S_ISREG(info.st_mode),
    }


def verify_log_identity(pid: int, frozen: Mapping[str, Any]) -> dict[str, Any]:
    current_stdout = fd_identity(pid, 1)
    current_stderr = fd_identity(pid, 2)
    expected_stdout = frozen["stdout"]
    expected_stderr = frozen["stderr"]
    for name, current, expected in (
        ("stdout", current_stdout, expected_stdout),
        ("stderr", current_stderr, expected_stderr),
    ):
        for field in ("target", "device", "inode", "is_regular_file"):
            if current[field] != expected[field]:
                raise IdentityError("v2_log_fd_identity_mismatch", descriptor=name, field=field, actual=current[field], expected=expected[field])
    path = Path(str(frozen["log_file"]))
    if not path.is_file() or path.is_symlink():
        raise IdentityError("v2_live_log_not_regular", path=str(path))
    info = path.stat()
    if info.st_dev != int(frozen["log_device"]) or info.st_ino != int(frozen["log_inode"]):
        raise IdentityError("v2_live_log_inode_mismatch", actual_device=info.st_dev, actual_inode=info.st_ino, expected_device=frozen["log_device"], expected_inode=frozen["log_inode"])
    return {
        "log_file": str(path),
        "log_device": info.st_dev,
        "log_inode": info.st_ino,
        "log_size": info.st_size,
        "stdout": current_stdout,
        "stderr": current_stderr,
    }


def direct_opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def readiness(endpoint: str, expected_model: str) -> list[dict[str, Any]]:
    opener = direct_opener()
    ledger = []
    for round_id in range(1, 4):
        with opener.open(endpoint + "/health", timeout=10) as response:
            health_status = response.status
            health_body = response.read()
        with opener.open(endpoint + "/v1/models", timeout=10) as response:
            models_status = response.status
            models_body = response.read()
        ids = [str(item.get("id")) for item in json.loads(models_body or b"{}").get("data", [])]
        row = {
            "round": round_id,
            "health_status": health_status,
            "health_body_bytes": len(health_body),
            "models_status": models_status,
            "model_ids": ids,
            "pass": health_status == 200 and models_status == 200 and ids == [expected_model],
            "checked_at_unix": time.time(),
        }
        if not row["pass"]:
            raise IdentityError("v2_readiness_round_failed", round=round_id, row=row)
        ledger.append(row)
        if round_id < 3:
            time.sleep(1)
    return ledger


def verify_live_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    if identity.get("schema_version") != "certa_active_v1_prewarmed_vllm_service_identity_v2":
        raise IdentityError("v2_identity_schema_version")
    pid = int(identity["pid"])
    proc = Path(f"/proc/{pid}")
    if not proc.is_dir():
        raise IdentityError("v2_process_missing", pid=pid)
    start_ticks = (proc / "stat").read_text().split()[21]
    if start_ticks != str(identity["process_start_ticks"]):
        raise IdentityError("v2_process_start_ticks_mismatch", actual=start_ticks, expected=identity["process_start_ticks"])
    argv = proc_argv(pid)
    normalized = " ".join(argv)
    argv_sha = sha_bytes(canonical(argv).encode())
    cmdline_sha = sha_bytes(normalized.encode())
    if argv_sha != identity["argv_sha256"]:
        raise IdentityError("v2_argv_sha256_mismatch", actual=argv_sha, expected=identity["argv_sha256"])
    if cmdline_sha != identity["cmdline_sha256"]:
        raise IdentityError("v2_cmdline_sha256_mismatch", actual=cmdline_sha, expected=identity["cmdline_sha256"])
    executable = str(Path(os.readlink(proc / "exe")).resolve())
    if executable != identity["executable"]:
        raise IdentityError("v2_executable_mismatch", actual=executable, expected=identity["executable"])
    environment = proc_environ(pid)
    if environment.get("CUDA_VISIBLE_DEVICES") != identity["cuda_visible_devices"]:
        raise IdentityError("v2_gpu_mismatch", actual=environment.get("CUDA_VISIBLE_DEVICES"), expected=identity["cuda_visible_devices"])
    required = {
        "--model": identity["model_root"],
        "--served-model-name": identity["served_model_name"],
        "--host": identity["host"],
        "--port": str(identity["port"]),
        "--max-model-len": str(identity["max_model_len"]),
    }
    for flag, expected in required.items():
        actual = argument_value(argv, flag)
        if actual != expected:
            raise IdentityError("v2_semantic_argument_mismatch", flag=flag, actual=actual, expected=expected)
    ownership = current_port_ownership(pid, int(identity["port"]))
    if not ownership["root_or_descendant_owns_port"]:
        raise IdentityError("v2_process_tree_does_not_own_port", ownership=ownership)
    log = verify_log_identity(pid, identity["log_identity"])
    return {
        "pid": pid,
        "process_start_ticks": start_ticks,
        "argv_sha256": argv_sha,
        "cmdline_sha256": cmdline_sha,
        "executable": executable,
        "cuda_visible_devices": environment["CUDA_VISIBLE_DEVICES"],
        "port_ownership": ownership,
        "log_identity": log,
        "served_model_name": identity["served_model_name"],
        "model_root": identity["model_root"],
        "max_model_len": identity["max_model_len"],
    }


def load_runner(repo: Path, output_root: Path):
    path = repo / BIND["frozen_runner"]
    spec = importlib.util.spec_from_file_location("certa_frozen_completion_runner_v2", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("frozen_runner_import_failed")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(repo))
    spec.loader.exec_module(module)
    module.OUT = output_root
    return module


def state_path(output_root: Path) -> Path:
    return output_root / "runtime/RUNTIME_CONTROLLER_STATE.json"


def load_state(output_root: Path) -> dict[str, Any]:
    return json.loads(state_path(output_root).read_text(encoding="utf-8"))


def save_state(output_root: Path, state: Mapping[str, Any]) -> None:
    write_json(state_path(output_root), dict(state))


def fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def required_paths() -> list[str]:
    result = []
    for line in (PACK / "REQUIRED_ARTIFACTS.md").read_text(encoding="utf-8").splitlines():
        value = line.strip().lstrip("- ").strip("`")
        if value and "/" in value and not value.startswith("#"):
            result.append(value)
    return result


def atomic_finalize(repo: Path, output_root: Path, terminal_state: str, stage: str, error: BaseException | None = None) -> None:
    terminal = output_root / "terminal"
    if terminal.exists():
        raise RuntimeError("terminal_already_exists")
    staging = output_root / (".terminal_staging_" + uuid.uuid4().hex)
    staging.mkdir(parents=True)
    try:
        state = load_state(output_root)
        runner = load_runner(repo, output_root)
        if isinstance(error, IdentityError):
            error_record = {"type": type(error).__name__, "code": error.code, "details": error.details}
        elif error is not None:
            error_record = {"type": type(error).__name__, "message_sha256": sha_bytes(str(error).encode())}
        else:
            error_record = None
        cost = runner.cost_ledger() if hasattr(runner, "cost_ledger") else {"status": "NOT_AVAILABLE"}
        write_json(staging / "COST_LEDGER.json", cost)
        write_json(staging / "FINAL_TERMINAL_STATE.json", {
            "schema_version": "certa_active_v1_prewarmed_identity_v2_terminal_v1",
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
            "method_sha": BIND["head"],
            "terminal_state": terminal_state,
            "repository_clean": git(repo, "status", "--porcelain") == "",
            "identity_binding": state.get("identity_binding"),
            "service_identity": state.get("service_identity"),
        })
        disposition = []
        for relative in required_paths():
            path = output_root / relative
            disposition.append({"path": relative, "status": "PRESENT" if path.is_file() else "NOT_REACHED", "sha256": sha_file(path) if path.is_file() else None})
        write_json(staging / "REQUIRED_ARTIFACTS.json", {"terminal_state": terminal_state, "artifacts": disposition})
        bundle = staging / "CERTA_ACTIVE_V1_FINAL_COMPLETION.bundle"
        run(["git", "bundle", "create", str(bundle), BIND["branch"]], cwd=repo)
        run(["git", "bundle", "verify", str(bundle)], cwd=repo)
        report = [
            "# CERTA Active V1 V2-Identity Scientific Execution",
            "",
            f"Terminal: `{terminal_state}`",
            f"Stage: `{stage}`",
            f"Method: `{BIND['head']}`",
            "Service lifecycle owner: `USER_RUNTIME_OPERATOR`",
            f"Logical calls: `{cost.get('logical_calls', 'NOT_RECORDED')}`",
        ]
        if error_record:
            report.append("Error: `" + canonical(error_record) + "`")
        (staging / "TERMINAL_REPORT.md").write_text("\n\n".join(report) + "\n", encoding="utf-8")
        checksum_rows = []
        for path in sorted(output_root.rglob("*")):
            if path.is_file() and staging not in path.parents and path.name != "SHA256SUMS.txt":
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
    except Exception as exc:
        raise RuntimeError("BLOCKED_ATOMIC_FINALIZATION_FAILED:" + str(staging)) from exc


def bootstrap(args: argparse.Namespace) -> None:
    repo = Path(args.repo).resolve()
    output_root = Path(args.output_root).resolve()
    archive = Path(args.evidence_archive).resolve()
    binding_path = Path(args.binding_manifest).resolve()
    if output_root.exists():
        raise RuntimeError("new_output_root_already_exists")
    output_root.mkdir(parents=True)
    state = {
        "schema_version": "certa_active_v1_prewarmed_identity_v2_controller_state_v1",
        "status": "V2_IDENTITY_VALIDATION_IN_PROGRESS",
        "method_commit": BIND["head"],
        "service_managed_by": "USER_RUNTIME_OPERATOR",
        "controller_service_lifecycle_authority": False,
        "current_stage": "bootstrap-adopt-existing-service-v2",
        "created_at_unix": time.time(),
    }
    save_state(output_root, state)
    try:
        repository = verify_repository(repo)
        manifest = json.loads(binding_path.read_text(encoding="utf-8"))
        if manifest.get("method_sha") != BIND["head"]:
            raise IdentityError("v2_binding_method_sha_mismatch", actual=manifest.get("method_sha"), expected=BIND["head"])
        identity, evidence = archive_identity(archive, manifest)
        service = verify_live_identity(identity)
        readiness_ledger = readiness(identity["endpoint"], identity["served_model_name"])
        write_json(output_root / "runtime/V2_READINESS_LEDGER.json", readiness_ledger)
        state.update({
            "status": "PREWARMED_IDENTITY_V2_ADOPTED",
            "identity_binding": manifest,
            "evidence": evidence,
            "repository": repository,
            "service_identity": service,
            "readiness_rounds": 3,
            "current_stage": "frozen-freeze-and-preflight",
        })
        save_state(output_root, state)
        write_json(output_root / "runtime/PREWARMED_IDENTITY_V2_ADOPTION.json", state)
        runner = load_runner(repo, output_root)
        runner.freeze()
        rc = runner.preflight()
        if rc:
            raise RuntimeError("frozen_preflight_failed:" + str(rc))
        state.update({"status": "SCIENTIFIC_DAG_READY", "current_stage": "Integration16", "preflight_pass": True})
        save_state(output_root, state)
    except Exception as error:
        failure = {"type": type(error).__name__}
        if isinstance(error, IdentityError):
            failure.update({"code": error.code, "details": error.details})
        else:
            failure["message_sha256"] = sha_bytes(str(error).encode())
        write_json(output_root / "runtime/V2_IDENTITY_VALIDATION_FAILURE.json", failure)
        state.update({"status": "V2_ADOPTION_FAILED", "failure": failure})
        save_state(output_root, state)
        terminal = "BLOCKED_PREWARMED_SERVICE_V2_READINESS_FAILED" if isinstance(error, IdentityError) and "readiness" in error.code else "BLOCKED_PREWARMED_SERVICE_V2_IDENTITY_FAILED"
        atomic_finalize(repo, output_root, terminal, "bootstrap-adopt-existing-service-v2", error)
        raise


def verify_adopted(output_root: Path) -> dict[str, Any]:
    state = load_state(output_root)
    manifest = state["identity_binding"]
    identity, _ = archive_identity(Path(manifest["archive_path"]), manifest)
    return verify_live_identity(identity)


def stage_terminal(command: str) -> str:
    return {
        "integration16": "FREEZE_CERTA_ACTIVE_INTEGRATION_FAILED",
        "constructor64": "FREEZE_CERTA_ACTIVE_CONSTRUCTOR_FAILED",
        "decision-dev": "FREEZE_CERTA_ACTIVE_DECISION_FAILED",
        "unblind-dev": "FREEZE_CERTA_ACTIVE_DECISION_FAILED",
        "holdout-blind": "FREEZE_CERTA_ACTIVE_HOLDOUT_FAILED",
        "unblind-holdout": "FREEZE_CERTA_ACTIVE_HOLDOUT_FAILED",
    }[command]


def call_stage(runner: Any, command: str) -> int:
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
    state = load_state(output_root)
    if state.get("status") not in {"SCIENTIFIC_DAG_READY", "SCIENTIFIC_STAGE_COMPLETE"}:
        raise RuntimeError("controller_not_scientific_ready")
    state.update({"status": "SCIENTIFIC_STAGE_RUNNING", "current_stage": args.command})
    save_state(output_root, state)
    try:
        before = verify_adopted(output_root)
        write_json(output_root / f"runtime/SERVICE_IDENTITY_BEFORE_{args.command}.json", before)
        runner = load_runner(repo, output_root)
        rc = call_stage(runner, args.command)
        if rc:
            raise RuntimeError(f"scientific_stage_nonzero:{args.command}:{rc}")
        after = verify_adopted(output_root)
        write_json(output_root / f"runtime/SERVICE_IDENTITY_AFTER_{args.command}.json", after)
        state.update({"status": "SCIENTIFIC_STAGE_COMPLETE", "current_stage": args.command, "last_completed_stage": args.command})
        save_state(output_root, state)
    except Exception as error:
        if isinstance(error, IdentityError):
            terminal = "BLOCKED_PREWARMED_SERVICE_V2_LOST"
        else:
            terminal = stage_terminal(args.command)
        state.update({"status": "STAGE_FAILED", "failure_type": type(error).__name__})
        save_state(output_root, state)
        atomic_finalize(repo, output_root, terminal, args.command, error)
        raise


def finalize(args: argparse.Namespace) -> None:
    repo = Path(args.repo).resolve()
    output_root = Path(args.output_root).resolve()
    verify_adopted(output_root)
    atomic_finalize(repo, output_root, args.state, "finalize")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=(
        "bootstrap-adopt-existing-service-v2", "integration16", "constructor64", "decision-dev",
        "unblind-dev", "holdout-blind", "unblind-holdout", "finalize",
    ))
    parser.add_argument("--repo", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--evidence-archive", default=BIND["v2_evidence"]["archive"])
    parser.add_argument("--binding-manifest", default=BIND["v2_evidence"]["binding_manifest"])
    parser.add_argument("--state")
    args = parser.parse_args()
    if args.command == "bootstrap-adopt-existing-service-v2":
        bootstrap(args)
    elif args.command == "finalize":
        if not args.state:
            parser.error("--state is required for finalize")
        finalize(args)
    else:
        execute_stage(args)


if __name__ == "__main__":
    main()
