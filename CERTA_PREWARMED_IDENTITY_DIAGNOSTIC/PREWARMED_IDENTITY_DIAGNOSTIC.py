#!/usr/bin/env python3
"""Read-only diagnosis for CERTA prewarmed vLLM identity adoption.

This script does not start, stop, signal, restart, or reconfigure any process.
It emits the exact first mismatching identity field instead of only a hash.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Mapping


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def normalize_cmdline(value: str) -> str:
    return " ".join(str(value).replace("\x00", " ").split())


def pick(record: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in record:
            return record[name]
    raise KeyError("missing_identity_field:" + "|".join(names))


def proc_ppid(pid: int) -> int | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().split()
        return int(fields[3])
    except Exception:
        return None


def descendants(root_pid: int) -> set[int]:
    parent_to_children: dict[int, set[int]] = {}
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        pid = int(proc.name)
        ppid = proc_ppid(pid)
        if ppid is not None:
            parent_to_children.setdefault(ppid, set()).add(pid)
    result = {root_pid}
    frontier = [root_pid]
    while frontier:
        parent = frontier.pop()
        for child in parent_to_children.get(parent, set()):
            if child not in result:
                result.add(child)
                frontier.append(child)
    return result


def listening_socket_inodes(port: int) -> set[str]:
    expected = f"{port:04X}"
    inodes: set[str] = set()
    for table in (Path("/proc/net/tcp"), Path("/proc/net/tcp6")):
        if not table.is_file():
            continue
        for line in table.read_text().splitlines()[1:]:
            fields = line.split()
            if len(fields) < 10:
                continue
            local_port = fields[1].rsplit(":", 1)[-1].upper()
            state = fields[3]
            if local_port == expected and state == "0A":
                inodes.add(fields[9])
    return inodes


def pid_socket_inodes(pid: int) -> set[str]:
    result: set[str] = set()
    fd_root = Path(f"/proc/{pid}/fd")
    if not fd_root.is_dir():
        return result
    for fd in fd_root.iterdir():
        try:
            target = os.readlink(fd)
        except OSError:
            continue
        if target.startswith("socket:[") and target.endswith("]"):
            result.add(target[8:-1])
    return result


def port_owners(root_pid: int, port: int) -> dict[str, Any]:
    listen = listening_socket_inodes(port)
    tree = descendants(root_pid)
    owners: dict[int, list[str]] = {}
    for pid in sorted(tree):
        overlap = sorted(pid_socket_inodes(pid) & listen)
        if overlap:
            owners[pid] = overlap
    return {
        "listening_socket_inodes": sorted(listen),
        "process_tree_pids": sorted(tree),
        "owners_in_process_tree": {str(k): v for k, v in owners.items()},
        "root_pid_owns_port": root_pid in owners,
        "root_or_descendant_owns_port": bool(owners),
    }


def no_proxy_opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def get(url: str, timeout: int = 10) -> tuple[int, bytes]:
    request = urllib.request.Request(url, method="GET")
    with no_proxy_opener().open(request, timeout=timeout) as response:
        return response.status, response.read()


def compare(checks: list[dict[str, Any]], name: str, expected: Any, actual: Any, *, critical: bool = True) -> None:
    checks.append({
        "name": name,
        "expected": expected,
        "actual": actual,
        "pass": expected == actual,
        "critical": critical,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--expected-archive-sha256", default="694fcfb57bf7215cbd21b55a6860de5e86582c1312de311ab78a8502a6308a64")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    archive = Path(args.archive).resolve()
    identity_path = Path(args.identity).resolve()
    report: dict[str, Any] = {
        "schema_version": "certa_prewarmed_identity_diagnostic_v1",
        "created_at_unix": time.time(),
        "read_only": True,
        "checks": [],
    }
    checks: list[dict[str, Any]] = report["checks"]

    compare(checks, "archive_exists_regular", True, archive.is_file() and not archive.is_symlink())
    if archive.is_file():
        compare(checks, "archive_sha256", args.expected_archive_sha256, sha_file(archive))
    compare(checks, "identity_exists_regular", True, identity_path.is_file() and not identity_path.is_symlink())

    identity: dict[str, Any] = {}
    if archive.is_file() and identity_path.is_file():
        with tarfile.open(archive, "r:gz") as bundle:
            members = [m for m in bundle.getmembers() if m.isfile() and m.name.endswith("/PREWARMED_VLLM_SERVICE_IDENTITY.json")]
            compare(checks, "archive_identity_member_count", 1, len(members))
            if len(members) == 1:
                handle = bundle.extractfile(members[0])
                archived = handle.read() if handle else b""
                deployed = identity_path.read_bytes()
                compare(checks, "identity_byte_identical_to_archive", sha_bytes(archived), sha_bytes(deployed))
                identity = json.loads(deployed)
                report["identity_member"] = members[0].name
                report["identity_sha256"] = sha_bytes(deployed)

    if identity:
        try:
            pid = int(pick(identity, "pid"))
            proc = Path(f"/proc/{pid}")
            compare(checks, "pid_exists", True, proc.is_dir())
            if proc.is_dir():
                actual_ticks = (proc / "stat").read_text().split()[21]
                compare(checks, "process_start_ticks", str(pick(identity, "process_start_ticks", "start_ticks")), actual_ticks)

                actual_cmdline = normalize_cmdline((proc / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace"))
                expected_cmdline = normalize_cmdline(str(pick(identity, "cmdline", "normalized_cmdline")))
                compare(checks, "cmdline_exact", expected_cmdline, actual_cmdline)
                if identity.get("cmdline_sha256"):
                    compare(checks, "cmdline_sha256", identity["cmdline_sha256"], sha_bytes(actual_cmdline.encode()))

                actual_exe = os.readlink(proc / "exe")
                expected_exe = str(pick(identity, "executable", "python_executable", "exe"))
                compare(checks, "executable_resolved", str(Path(expected_exe).resolve()), str(Path(actual_exe).resolve()))

                environ: dict[str, str] = {}
                for item in (proc / "environ").read_bytes().split(b"\0"):
                    if b"=" in item:
                        key, value = item.split(b"=", 1)
                        environ[key.decode("utf-8", "replace")] = value.decode("utf-8", "replace")
                expected_gpu = str(pick(identity, "cuda_visible_devices", "CUDA_VISIBLE_DEVICES"))
                compare(checks, "CUDA_VISIBLE_DEVICES", expected_gpu, environ.get("CUDA_VISIBLE_DEVICES", ""))

                port = int(pick(identity, "port"))
                report["port_ownership"] = port_owners(pid, port)
                compare(checks, "root_or_descendant_owns_listening_port", True, report["port_ownership"]["root_or_descendant_owns_port"])
                compare(checks, "root_pid_directly_owns_listening_port", True, report["port_ownership"]["root_pid_owns_port"], critical=False)

                expected_model = str(pick(identity, "model_root", "model_path"))
                expected_name = str(pick(identity, "served_model_name", "model_name"))
                expected_len = int(pick(identity, "max_model_len"))
                for token in (expected_model, "--served-model-name", expected_name, "--port", str(port), "--max-model-len", str(expected_len)):
                    checks.append({"name": "cmdline_contains:" + token, "expected": True, "actual": token in actual_cmdline, "pass": token in actual_cmdline, "critical": True})

                log_path = Path(str(pick(identity, "log_file", "server_log", "log_path")))
                compare(checks, "log_exists_readable_regular", True, log_path.is_file() and not log_path.is_symlink() and os.access(log_path, os.R_OK))
                if log_path.is_file():
                    st = log_path.stat()
                    if "log_device" in identity:
                        compare(checks, "log_device", int(identity["log_device"]), st.st_dev)
                    if "log_inode" in identity:
                        compare(checks, "log_inode", int(identity["log_inode"]), st.st_ino)
                    report["current_log"] = {"path": str(log_path), "device": st.st_dev, "inode": st.st_ino, "size": st.st_size}

                endpoint = str(identity.get("endpoint") or f"http://127.0.0.1:{port}")
                for round_index in range(1, 4):
                    health_status, health_body = get(endpoint + "/health")
                    models_status, models_body = get(endpoint + "/v1/models")
                    model_ids = [str(item.get("id")) for item in json.loads(models_body or b"{}").get("data", [])]
                    checks.append({
                        "name": f"readiness_round_{round_index}",
                        "expected": {"health_status": 200, "model_ids": [expected_name]},
                        "actual": {"health_status": health_status, "health_body_bytes": len(health_body), "models_status": models_status, "model_ids": model_ids},
                        "pass": health_status == 200 and models_status == 200 and model_ids == [expected_name],
                        "critical": True,
                    })
                    if round_index < 3:
                        time.sleep(1)
        except Exception as exc:
            report["diagnostic_exception"] = {"type": type(exc).__name__, "message": str(exc)}

    failures = [x for x in checks if x.get("critical") and not x.get("pass")]
    report["critical_failure_count"] = len(failures)
    report["first_critical_failure"] = failures[0] if failures else None
    report["diagnostic_pass"] = not failures and "diagnostic_exception" not in report

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "diagnostic_pass": report["diagnostic_pass"],
        "critical_failure_count": report["critical_failure_count"],
        "first_critical_failure": report["first_critical_failure"],
        "output": str(output),
    }, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["diagnostic_pass"] else 2)


if __name__ == "__main__":
    main()
