#!/usr/bin/env bash
set -euo pipefail

CERTA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly CERTA_SOURCE_COMMIT_DEFAULT="0135203cad30710ddd4a854c9228dd564c2fca84"

load_local_paths() {
    if [ -f "${CERTA_ROOT}/configs/paths.env" ]; then
        # shellcheck source=/dev/null
        source "${CERTA_ROOT}/configs/paths.env"
    fi
}

require_value() {
    local name="$1"
    if [ -z "${!name:-}" ]; then
        echo "Missing required ${name}. Set it explicitly in the environment or configs/paths.env." >&2
        return 2
    fi
}

require_runtime_configuration() {
    local name
    for name in CERTA_PYTHON CERTA_DATASET CERTA_INPUT_FILE CERTA_TABLE_DIR CERTA_OUTPUT_ROOT CERTA_RUN_ID CERTA_GENERATOR_BACKEND CERTA_MODEL_ID CERTA_API_BASE_URL CERTA_API_MODEL; do
        require_value "${name}"
    done
    if [ ! -x "${CERTA_PYTHON}" ]; then
        echo "CERTA_PYTHON is not executable: ${CERTA_PYTHON}" >&2
        return 2
    fi
    if [ ! -f "${CERTA_INPUT_FILE}" ] || [ ! -d "${CERTA_TABLE_DIR}" ]; then
        echo "CERTA_INPUT_FILE and CERTA_TABLE_DIR must exist before execution." >&2
        return 2
    fi
    if [ "${CERTA_DATASET}" != "aitqa" ] || [ "${CERTA_GENERATOR_BACKEND}" != "vllm_chat" ] || [ "${CERTA_MODEL_ID}" != "Qwen3-8B" ] || [ "${CERTA_API_MODEL}" != "Qwen3-8B" ] || [ "${CERTA_API_BASE_URL}" != "http://127.0.0.1:30338/v1" ]; then
        echo "This release accepts only AIT-QA clean with vllm_chat at Qwen3-8B on http://127.0.0.1:30338/v1." >&2
        return 2
    fi
}

freeze_public_main_profile() {
    export CERTA_MAIN_CERT_PROFILE="${CERTA_MAIN_CERT_PROFILE:-0}"
    export CERTA_SEED="${CERTA_SEED:-0}"
    export CERTA_PYTHONHASHSEED="${CERTA_PYTHONHASHSEED:-0}"
    export CERTA_CACHE_MODE="${CERTA_CACHE_MODE:-off}"
    export CERTA_MAX_ANSWER_TOKENS="${CERTA_MAX_ANSWER_TOKENS:-32}"
    export CERTA_MAX_MODEL_LEN="${CERTA_MAX_MODEL_LEN:-8192}"
    if [ "${CERTA_MAIN_CERT_PROFILE}" != "0" ] || [ "${CERTA_SEED}" != "0" ] || [ "${CERTA_PYTHONHASHSEED}" != "0" ] || [ "${CERTA_CACHE_MODE}" != "off" ] || [ "${CERTA_MAX_ANSWER_TOKENS}" != "32" ] || [ "${CERTA_MAX_MODEL_LEN}" != "8192" ]; then
        echo "The public Qwen3/AIT-QA main profile requires main_cert_profile=0, seed=0, PYTHONHASHSEED=0, cache_mode=off, max_answer_tokens=32, and max_model_len=8192." >&2
        return 2
    fi
}

map_public_environment() {
    local run_dir="$1"
    export CSCR_PROJECT_DIR="${CERTA_ROOT}"
    export CSCR_PYTHON="${CERTA_PYTHON}"
    export CSCR_DATASET="${CERTA_DATASET}"
    export CSCR_INPUT_FILE="${CERTA_INPUT_FILE}"
    export CSCR_TABLE_DIR="${CERTA_TABLE_DIR}"
    export CSCR_MODEL_PATH="${CERTA_MODEL_ID}"
    export CSCR_GENERATOR_BACKEND="${CERTA_GENERATOR_BACKEND}"
    export CSCR_API_BASE_URL="${CERTA_API_BASE_URL}"
    export CSCR_API_MODEL="${CERTA_API_MODEL}"
    export CSCR_API_KEY_ENV="EMPTY"
    export CSCR_OUTPUT_DIR="${run_dir}"
    export CSCR_RUN_TIMESTAMP="${CERTA_RUN_ID}"
    export PYTHONHASHSEED="${CERTA_PYTHONHASHSEED:-0}"
    export CSCR_MAIN_CERT_PROFILE="${CERTA_MAIN_CERT_PROFILE:-0}"
    export CSCR_SEED="${CERTA_SEED:-0}"
    export CSCR_API_CACHE_MODE="${CERTA_CACHE_MODE:-off}"
    export CSCR_MAX_ANSWER_TOKENS="${CERTA_MAX_ANSWER_TOKENS:-32}"
    export CSCR_MAX_LEN="${CERTA_MAX_MODEL_LEN:-8192}"
}

write_release_metadata() {
    local run_dir="$1"
    shift
    mkdir -p "${run_dir}"
    CERTA_METADATA_DIR="${run_dir}" CERTA_ROOT_FOR_METADATA="${CERTA_ROOT}" CERTA_PYTHON="${CERTA_PYTHON}" \
        CERTA_DATASET="${CERTA_DATASET}" CERTA_INPUT_FILE="${CERTA_INPUT_FILE}" CERTA_TABLE_DIR="${CERTA_TABLE_DIR}" \
        CERTA_GENERATOR_BACKEND="${CERTA_GENERATOR_BACKEND}" CERTA_MODEL_ID="${CERTA_MODEL_ID}" \
        CERTA_API_BASE_URL="${CERTA_API_BASE_URL}" CERTA_API_MODEL="${CERTA_API_MODEL}" \
        CERTA_SOURCE_COMMIT="${CERTA_SOURCE_COMMIT}" CSCR_SEED="${CSCR_SEED}" CSCR_API_CACHE_MODE="${CSCR_API_CACHE_MODE}" \
        CERTA_COMMAND_JSON="$(printf '%s\n' "$@" | "${CERTA_PYTHON}" -c 'import json,sys; print(json.dumps(sys.stdin.read().splitlines()))')" \
        "${CERTA_PYTHON}" - <<'PY'
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path

def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def tree_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(child.relative_to(path)).encode("utf-8"))
        digest.update(file_hash(child).encode("ascii"))
    return digest.hexdigest()

def version(name: str):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None

def git_head(root: Path):
    result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"

root = Path(os.environ["CERTA_ROOT_FOR_METADATA"])
python = os.environ["CERTA_PYTHON"]
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True).stdout.splitlines() if __import__("shutil").which("nvidia-smi") else []
payload = {
    "source_commit": os.environ.get("CERTA_SOURCE_COMMIT", "0135203cad30710ddd4a854c9228dd564c2fca84"),
    "target_commit": git_head(root),
    "python": {"executable": python, "version": platform.python_version()},
    "packages": {name: version(name) for name in ("torch", "vllm", "transformers", "openai")},
    "cuda": __import__("torch").version.cuda,
    "gpu": gpu,
    "dataset": os.environ["CERTA_DATASET"],
    "input_file": os.environ["CERTA_INPUT_FILE"],
    "input_sha256": file_hash(Path(os.environ["CERTA_INPUT_FILE"])),
    "table_dir": os.environ["CERTA_TABLE_DIR"],
    "table_sha256": tree_hash(Path(os.environ["CERTA_TABLE_DIR"])),
    "generator_backend": os.environ["CERTA_GENERATOR_BACKEND"],
    "model_id": os.environ["CERTA_MODEL_ID"],
    "api_base_url": os.environ["CERTA_API_BASE_URL"],
    "api_model": os.environ["CERTA_API_MODEL"],
    "main_cert_profile": os.environ.get("CERTA_MAIN_CERT_PROFILE", "0"),
    "seed": os.environ.get("CSCR_SEED", "0"),
    "cache_mode": os.environ.get("CSCR_API_CACHE_MODE", "off"),
    "max_answer_tokens": os.environ.get("CSCR_MAX_ANSWER_TOKENS", "32"),
    "max_model_len": os.environ.get("CSCR_MAX_LEN", "8192"),
    "pythonhashseed": os.environ.get("PYTHONHASHSEED", "0"),
    "command": json.loads(os.environ["CERTA_COMMAND_JSON"]),
    "release_validation_only": True,
    "not_for_method_selection_or_paper_claims": True,
}
Path(os.environ["CERTA_METADATA_DIR"], "release_metadata.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

run_legacy_mode() {
    local mode="$1"
    local run_dir="$2"
    local -a command=(bash "${CERTA_ROOT}/run_cscr.sh" "${mode}")
    if [ -n "${CERTA_LIMIT:-}" ]; then
        command+=(--limit "${CERTA_LIMIT}")
    fi
    write_release_metadata "${run_dir}" "${command[@]}"
    "${command[@]}" 2>&1 | tee "${run_dir}/run_info.log"
    local required
    for required in predictions.jsonl predictions.debug.jsonl metrics.json run_config.json; do
        if [ ! -f "${run_dir}/${required}" ]; then
            echo "Pipeline did not create required artifact: ${run_dir}/${required}" >&2
            return 3
        fi
    done
    finalize_run_metadata "${run_dir}"
}

finalize_run_metadata() {
    local run_dir="$1"
    CERTA_METADATA_DIR="${run_dir}" "${CERTA_PYTHON}" - <<'PY'
import hashlib
import json
from pathlib import Path

run_dir = Path(__import__("os").environ["CERTA_METADATA_DIR"])
files = []
for path in sorted(item for item in run_dir.rglob("*") if item.is_file()):
    if path.name in {"artifact_manifest.json", "checksums.sha256"}:
        continue
    files.append({"path": str(path.relative_to(run_dir)), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
(run_dir / "artifact_manifest.json").write_text(json.dumps({"files": files}, indent=2) + "\n", encoding="utf-8")
with (run_dir / "checksums.sha256").open("w", encoding="utf-8") as handle:
    for path in sorted(item for item in run_dir.rglob("*") if item.is_file() and item.name != "checksums.sha256"):
        handle.write(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(run_dir)}\n")
PY
}
