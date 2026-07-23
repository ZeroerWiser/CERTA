#!/usr/bin/env python3
"""Freeze a complete V2 identity for an already-running user-managed vLLM service."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import stat
import tarfile
import time
import urllib.request
from pathlib import Path
from typing import Any

PACK = Path(__file__).resolve().parent
BIND = json.loads((PACK / "SOURCE_BINDINGS.json").read_text(encoding="utf-8"))


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def normalize_argv(argv: list[str]) -> str:
    return " ".join(argv)


def proc_argv(pid: int) -> list[str]:
    raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    return [item.decode("utf-8", "replace") for item in raw.split(b"\0") if item]


def proc_environ(pid: int) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in Path(f"/proc/{pid}/environ").read_bytes().split(b"\0"):
        if b"=" in item:
            key, value = item.split(b"=", 1)
            result[key.decode("utf-8", "replace")] = value.decode("utf-8", "replace")
    return result


def argument_value(argv: list[str], flag: str) -> str | None:
    for index, item in enumerate(argv):
        if item == flag and index + 1 < len(argv):
            return argv[index + 1]
        prefix = flag + "="
        if item.startswith(prefix):
            return item[len(prefix):]
    return None


def descendants(root_pid: int) -> set[int]:
    ppid_map: dict[int, int] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().split()
            ppid_map[int(entry.name)] = int(fields[3])
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
    result: set[str] = set()
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
    expected = f"{port:04X}"
    result: set[str] = set()
    for table in (Path("/proc/net/tcp"), Path("/proc/net/tcp6")):
        if not table.is_file():
            continue
        for line in table.read_text().splitlines()[1:]:
            fields = line.split()
            if len(fields) >= 10 and fields[1].rsplit(":", 1)[-1].upper() == expected and fields[3] == "0A":
                result.add(fields[9])
    return result


def port_ownership(root_pid: int, port: int) -> dict[str, Any]:
    listening = listening_inodes(port)
    tree = sorted(descendants(root_pid))
    owners: dict[str, list[str]] = {}
    for pid in tree:
        owned = sorted(socket_inodes(pid) & listening)
        if owned:
            owners[str(pid)] = owned
    return {
        "listening_socket_inodes": sorted(listening),
        "process_tree_pids": tree,
        "owners_in_process_tree": owners,
        "root_pid_owns_port": bool(socket_inodes(root_pid) & listening),
        "root_or_descendant_owns_port": bool(owners),
    }


def fd_identity(pid: int, fd: int) -> dict[str, Any]:
    fd_path = Path(f"/proc/{pid}/fd/{fd}")
    target = os.readlink(fd_path)
    info = os.stat(fd_path)
    return {
        "fd": fd,
        "target": target,
        "device": info.st_dev,
        "inode": info.st_ino,
        "mode": info.st_mode,
        "is_regular_file": stat.S_ISREG(info.st_mode),
    }


def resolve_log_identity(pid: int) -> dict[str, Any]:
    stdout = fd_identity(pid, 1)
    stderr = fd_identity(pid, 2)
    if not stdout["is_regular_file"] and not stderr["is_regular_file"]:
        raise RuntimeError("live_log_not_regular_file")
    selected = stdout if stdout["is_regular_file"] else stderr
    if stdout["is_regular_file"] and stderr["is_regular_file"]:
        if (stdout["device"], stdout["inode"]) != (stderr["device"], stderr["inode"]):
            raise RuntimeError("stdout_stderr_log_identity_diverged")
    target = str(selected["target"])
    if target.endswith(" (deleted)"):
        raise RuntimeError("live_log_deleted")
    path = Path(target)
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise RuntimeError("live_log_path_not_readable_regular")
    path_info = path.stat()
    if (path_info.st_dev, path_info.st_ino) != (selected["device"], selected["inode"]):
        raise RuntimeError("live_log_path_fd_identity_mismatch")
    return {
        "mode": "stdout_stderr_regular_file",
        "log_file": str(path),
        "log_device": path_info.st_dev,
        "log_inode": path_info.st_ino,
        "log_size_at_refreeze": path_info.st_size,
        "stdout": stdout,
        "stderr": stderr,
    }


def direct_opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def readiness(endpoint: str, expected_model: str) -> list[dict[str, Any]]:
    opener = direct_opener()
    result = []
    for round_id in range(1, 4):
        with opener.open(endpoint + "/health", timeout=10) as response:
            health_status = response.status
            health_body = response.read()
        with opener.open(endpoint + "/v1/models", timeout=10) as response:
            model_status = response.status
            model_body = response.read()
        models = json.loads(model_body or b"{}")
        ids = [str(item.get("id")) for item in models.get("data", [])]
        row = {
            "round": round_id,
            "health_status": health_status,
            "health_body_bytes": len(health_body),
            "models_status": model_status,
            "model_ids": ids,
            "pass": health_status == 200 and model_status == 200 and ids == [expected_model],
            "checked_at_unix": time.time(),
        }
        if not row["pass"]:
            raise RuntimeError("readiness_round_failed:" + str(round_id))
        result.append(row)
        if round_id < 3:
            time.sleep(1)
    return result


def deterministic_archive(source_dir: Path, archive: Path) -> None:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as bundle:
        for path in sorted(source_dir.rglob("*")):
            if not path.is_file():
                continue
            relative = Path(source_dir.name) / path.relative_to(source_dir)
            data = path.read_bytes()
            info = tarfile.TarInfo(str(relative))
            info.size = len(data)
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mode = 0o440
            bundle.addfile(info, io.BytesIO(data))
    with archive.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            zipped.write(buffer.getvalue())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostic", default=BIND["diagnostic_result"]["path"])
    parser.add_argument("--evidence-dir", default=BIND["v2_evidence"]["directory"])
    parser.add_argument("--archive", default=BIND["v2_evidence"]["archive"])
    parser.add_argument("--binding-manifest", default=BIND["v2_evidence"]["binding_manifest"])
    args = parser.parse_args()

    diagnostic_path = Path(args.diagnostic).resolve()
    evidence_dir = Path(args.evidence_dir).resolve()
    archive = Path(args.archive).resolve()
    binding_path = Path(args.binding_manifest).resolve()
    for target in (evidence_dir, archive, binding_path):
        if target.exists():
            raise SystemExit("v2_target_already_exists:" + str(target))

    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    failure = diagnostic.get("first_critical_failure") or {}
    if failure.get("name") != BIND["diagnostic_result"]["required_first_failure"]:
        raise SystemExit("diagnostic_first_failure_mismatch")
    if failure.get("actual") != BIND["current_service"]["diagnostic_normalized_cmdline_sha256"]:
        raise SystemExit("diagnostic_actual_cmdline_sha_mismatch")
    if failure.get("expected") != BIND["superseded_identity"]["expected_cmdline_sha256"]:
        raise SystemExit("diagnostic_expected_cmdline_sha_mismatch")
    ownership = diagnostic.get("port_ownership") or {}
    if ownership.get("root_pid_owns_port") is not True or ownership.get("root_or_descendant_owns_port") is not True:
        raise SystemExit("diagnostic_port_ownership_not_verified")
    exception_text = canonical(diagnostic.get("diagnostic_exception") or {})
    if "missing_identity_field:log_file|server_log|log_path" not in exception_text:
        raise SystemExit("diagnostic_missing_log_failure_not_verified")

    frozen = BIND["current_service"]
    pid = int(frozen["pid"])
    proc = Path(f"/proc/{pid}")
    if not proc.is_dir():
        raise SystemExit("current_service_pid_missing")
    start_ticks = (proc / "stat").read_text().split()[21]
    argv = proc_argv(pid)
    normalized = normalize_argv(argv)
    normalized_sha = sha_bytes(normalized.encode())
    if normalized_sha != frozen["diagnostic_normalized_cmdline_sha256"]:
        raise SystemExit("live_cmdline_changed_since_diagnostic")
    executable = str(Path(os.readlink(proc / "exe")).resolve())
    environment = proc_environ(pid)
    if environment.get("CUDA_VISIBLE_DEVICES") != frozen["cuda_visible_devices"]:
        raise SystemExit("live_gpu_mismatch")

    required = {
        "--model": frozen["model_root"],
        "--served-model-name": frozen["served_model_name"],
        "--host": frozen["host"],
        "--port": str(frozen["port"]),
        "--max-model-len": str(frozen["max_model_len"]),
    }
    for flag, expected in required.items():
        if argument_value(argv, flag) != expected:
            raise SystemExit("live_required_argument_mismatch:" + flag)
    if "vllm.entrypoints.openai.api_server" not in argv:
        raise SystemExit("live_vllm_module_missing")

    current_ownership = port_ownership(pid, int(frozen["port"]))
    if not current_ownership["root_or_descendant_owns_port"]:
        raise SystemExit("live_process_tree_does_not_own_port")
    log_identity = resolve_log_identity(pid)
    readiness_ledger = readiness(frozen["endpoint"], frozen["served_model_name"])

    evidence_dir.mkdir(parents=True)
    identity = {
        "schema_version": "certa_active_v1_prewarmed_vllm_service_identity_v2",
        "pid": pid,
        "process_start_ticks": start_ticks,
        "argv": argv,
        "argv_sha256": sha_bytes(canonical(argv).encode()),
        "normalized_cmdline": normalized,
        "cmdline_sha256": normalized_sha,
        "executable": executable,
        "cuda_visible_devices": environment["CUDA_VISIBLE_DEVICES"],
        "host": frozen["host"],
        "port": frozen["port"],
        "endpoint": frozen["endpoint"],
        "api_base_url": frozen["api_base_url"],
        "served_model_name": frozen["served_model_name"],
        "model_root": frozen["model_root"],
        "max_model_len": frozen["max_model_len"],
        "port_ownership": current_ownership,
        "log_identity": log_identity,
        "readiness_cycles": 3,
        "readiness_pass": True,
        "created_at_unix": time.time(),
        "service_lifecycle_owner": "USER_RUNTIME_OPERATOR",
    }
    identity_path = evidence_dir / "PREWARMED_VLLM_SERVICE_IDENTITY_V2.json"
    write_json(identity_path, identity)
    write_json(evidence_dir / "READINESS_LEDGER.json", readiness_ledger)
    write_json(evidence_dir / "SOURCE_DIAGNOSTIC.json", diagnostic)
    write_json(evidence_dir / "REFREEZE_PROVENANCE.json", {
        "schema_version": "certa_active_v1_prewarmed_identity_v2_refreeze_provenance_v1",
        "superseded_archive_sha256": BIND["superseded_identity"]["archive_sha256"],
        "superseded_cmdline_sha256": BIND["superseded_identity"]["expected_cmdline_sha256"],
        "blocked_failure_sha256": BIND["superseded_identity"]["blocked_failure_sha256"],
        "diagnostic_path": str(diagnostic_path),
        "diagnostic_sha256": sha_file(diagnostic_path),
        "method_sha": BIND["head"],
        "service_restarted": False,
        "service_signaled": False,
    })

    checksum_rows = []
    for path in sorted(evidence_dir.iterdir()):
        if path.is_file() and path.name != "CONTENT_SHA256SUMS.txt":
            checksum_rows.append(f"{sha_file(path)}  {path.name}")
    (evidence_dir / "CONTENT_SHA256SUMS.txt").write_text("\n".join(checksum_rows) + "\n", encoding="utf-8")
    for path in evidence_dir.iterdir():
        if path.is_file():
            path.chmod(0o440)

    deterministic_archive(evidence_dir, archive)
    archive.chmod(0o440)
    binding = {
        "schema_version": "certa_active_v1_prewarmed_identity_v2_binding_v1",
        "method_sha": BIND["head"],
        "evidence_directory": str(evidence_dir),
        "archive_path": str(archive),
        "archive_sha256": sha_file(archive),
        "identity_member": f"{evidence_dir.name}/PREWARMED_VLLM_SERVICE_IDENTITY_V2.json",
        "identity_sha256": sha_file(identity_path),
        "pid": pid,
        "process_start_ticks": start_ticks,
        "cmdline_sha256": normalized_sha,
        "argv_sha256": identity["argv_sha256"],
        "log_identity_sha256": sha_bytes(canonical(log_identity).encode()),
        "readiness_ledger_sha256": sha_file(evidence_dir / "READINESS_LEDGER.json"),
        "created_at_unix": time.time(),
    }
    write_json(binding_path, binding)
    binding_path.chmod(0o440)
    print(json.dumps({"status": "PREWARMED_IDENTITY_V2_FROZEN", **binding}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
