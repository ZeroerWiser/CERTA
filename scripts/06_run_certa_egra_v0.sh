#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/hsh/ME/Table/EMNLP2026/CERTA"
EXPECTED_EXTERNAL_ROOT="/home/hsh/ME/Table/EMNLP2026/certa_egra_outputs/CERTA_EGRA_V0_20260720T152831Z"
SEALED_ROOT="/home/hsh/ME/Table/EMNLP2026/certa_egra_sealed/CERTA_EGRA_V0_20260720T152831Z"
PACK_ROOT="/home/hsh/ME/Table/EMNLP2026/certa_goal_packs/CERTA_EGRA_V0_CONSTRUCTION_AND_CONDITIONAL_DECISION_GATE_PACK"
PROFILE="${REPO_ROOT}/configs/profiles/certa_egra_v0.env"
TABLE_DIR="${REPO_ROOT}/dataset/hitab/tables/raw"
DEV_RUNTIME="${EXPECTED_EXTERNAL_ROOT}/inputs/dev_runtime.jsonl"
R2_RUNTIME="/home/hsh/ME/Table/EMNLP2026/certa_round2_outputs/CERTA_R2_20260720T110557Z/inputs/lookup_sentinel16_runtime.jsonl"
ROLE_ROWS="${EXPECTED_EXTERNAL_ROOT}/freeze/DEV_ROLE_CONTRACTS.jsonl"
B0_ROWS="${EXPECTED_EXTERNAL_ROOT}/freeze/DEV_B0_FREEZE.jsonl"
EARLY_GATE="${EXPECTED_EXTERNAL_ROOT}/results/EARLY_SENTINEL_GATE.json"

usage() {
    echo "usage: $0 {role-r2|role-sentinel|sentinel-c0|freeze-b0-sentinel|sentinel-c2|early-gate|role-dev|dev-c0|freeze-b0-dev|dev-c1|dev-c2|constructor-gate}" >&2
}

if [ "$#" -ne 1 ]; then
    usage
    exit 2
fi
if [ "${CERTA_EGRA_EXTERNAL_ROOT:-${EXPECTED_EXTERNAL_ROOT}}" != "${EXPECTED_EXTERNAL_ROOT}" ]; then
    echo "external root identity mismatch" >&2
    exit 2
fi

phase="$1"
mkdir -p "${EXPECTED_EXTERNAL_ROOT}/runs/dev" "${EXPECTED_EXTERNAL_ROOT}/cache" "${EXPECTED_EXTERNAL_ROOT}/logs" "${EXPECTED_EXTERNAL_ROOT}/results"

"/home/hsh/anaconda3/envs/cond/bin/python" - "${REPO_ROOT}" "${EXPECTED_EXTERNAL_ROOT}" <<'PY'
import hashlib, json, subprocess, sys
from pathlib import Path
repo, root = map(Path, sys.argv[1:])
freeze = json.loads((root / "freeze/CONSTRUCTOR_CONFIG_FREEZE.json").read_text())
head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=repo, text=True).strip()
status = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True)
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
checks = {
    "method_sha": head,
    "branch": branch,
    "dev_runtime_sha256": sha(root / "inputs/dev_runtime.jsonl"),
    "profile_sha256": sha(repo / "configs/profiles/certa_egra_v0.env"),
    "runner_sha256": sha(repo / "scripts/06_run_certa_egra_v0.sh"),
}
for key, observed in checks.items():
    if freeze.get(key) != observed:
        raise SystemExit(f"constructor freeze mismatch: {key}")
if status:
    raise SystemExit("constructor method worktree is not clean")
if branch != "research/certa-egra-v0":
    raise SystemExit("constructor branch mismatch")
PY

# shellcheck disable=SC1090
source "${PROFILE}"

require_early_pass() {
    "${CSCR_PYTHON}" - "${EARLY_GATE}" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.is_file() or json.loads(path.read_text()).get("pass") is not True:
    raise SystemExit("machine early sentinel PASS required")
PY
}

verify_endpoint() {
    "${CSCR_PYTHON}" - "${EXPECTED_EXTERNAL_ROOT}/logs/ENDPOINT_LEDGER.jsonl" <<'PY'
import datetime, json, sys, urllib.request
from pathlib import Path
with urllib.request.urlopen("http://127.0.0.1:30338/v1/models", timeout=10) as response:
    payload = json.load(response)
models = sorted(str(item.get("id")) for item in payload.get("data", []))
if models != ["Qwen3-8B"]:
    raise SystemExit(f"endpoint model mismatch: {models}")
row = {
    "schema_version": "certa_egra_endpoint_ledger_v1",
    "checked_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "base_url": "http://127.0.0.1:30338/v1",
    "models": models,
    "thinking": {"enable_thinking": False},
}
with Path(sys.argv[1]).open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(row, sort_keys=True) + "\n")
PY
}

run_arm() {
    local stage="$1"
    local arm_name="$2"
    local arm output_dir
    case "${arm_name}" in
        c0) arm="C0_FLAT_SCHEMA_CURRENT" ;;
        c1) arm="C1_ROLE_ALIGNED_FLAT" ;;
        c2) arm="C2_EGRA" ;;
        *) usage; exit 2 ;;
    esac
    output_dir="${EXPECTED_EXTERNAL_ROOT}/runs/dev/${arm_name}"
    local -a bounded_args reuse_args
    bounded_args=()
    reuse_args=()
    if [ "${stage}" = "sentinel" ]; then
        bounded_args=(--limit 24)
        if [ -e "${output_dir}" ]; then
            echo "refusing to overwrite sentinel output: ${output_dir}" >&2
            exit 2
        fi
    elif [ "${arm_name}" = "c1" ]; then
        require_early_pass
        bounded_args=(--limit 32)
        if [ -e "${output_dir}" ]; then
            echo "refusing to overwrite C1 output: ${output_dir}" >&2
            exit 2
        fi
    else
        require_early_pass
        if [ ! -f "${output_dir}/predictions.debug.jsonl" ]; then
            echo "missing sentinel output for resume: ${output_dir}" >&2
            exit 2
        fi
        bounded_args=(--resume)
    fi
    if [ "${arm_name}" != "c0" ]; then
        reuse_args=(
            --certa-egra-frozen-b0-file "${B0_ROWS}"
            --certa-egra-frozen-role-file "${ROLE_ROWS}"
        )
    fi
    verify_endpoint
    export CSCR_INPUT_FILE="${DEV_RUNTIME}"
    export CSCR_TABLE_DIR="${TABLE_DIR}"
    export CSCR_OUTPUT_DIR="${output_dir}"
    export CSCR_API_CACHE_PATH="${EXPECTED_EXTERNAL_ROOT}/cache/qwen3_8b_nonthinking.jsonl"
    export CSCR_LLM_INPUT_AUDIT_FILE="llm_input_audit.jsonl"
    export CERTA_EGRA_ARM="${arm}"
    local -a command
    command=(
        bash "${REPO_ROOT}/run_cscr.sh" full_cert
        --cera-stage E71
        --cera-shadow-only
        --cera-round6-e71-v4
        --cera-enable-typed-planner
        --cera-planner-boundary proposal_blind_schema_only
        --cera-planner-contract rcpc_signature_v2
        --cera-planner-legacy-query-semantics-mode active
        --cera-planner-temperature 0
        --cera-planner-max-tokens 512
        --certa-egra-arm "${arm}"
        --certa-egra-embedding-device "${CERTA_EGRA_EMBEDDING_DEVICE}"
        --certa-egra-embedding-file-tree-sha256 "${CERTA_EGRA_EMBEDDING_FILE_TREE_SHA256}"
        "${reuse_args[@]}"
        "${bounded_args[@]}"
    )
    "${command[@]}" 2>&1 | tee "${EXPECTED_EXTERNAL_ROOT}/logs/${stage}_${arm_name}.log"
}

case "${phase}" in
    role-r2)
        verify_endpoint
        "${CSCR_PYTHON}" "${REPO_ROOT}/tools/certa_egra_artifacts.py" freeze-role \
            --runtime "${R2_RUNTIME}" \
            --output "${EXPECTED_EXTERNAL_ROOT}/audit/R2_ROLE_CONTRACTS.jsonl" \
            --manifest "${EXPECTED_EXTERNAL_ROOT}/audit/R2_ROLE_CONTRACT_FREEZE.json" \
            --cache "${EXPECTED_EXTERNAL_ROOT}/cache/qwen3_8b_nonthinking.jsonl"
        ;;
    role-sentinel)
        verify_endpoint
        "${CSCR_PYTHON}" "${REPO_ROOT}/tools/certa_egra_artifacts.py" freeze-role \
            --runtime "${DEV_RUNTIME}" --limit 24 \
            --output "${ROLE_ROWS}" \
            --manifest "${EXPECTED_EXTERNAL_ROOT}/freeze/ROLE_CONTRACT_FREEZE.json" \
            --cache "${EXPECTED_EXTERNAL_ROOT}/cache/qwen3_8b_nonthinking.jsonl"
        ;;
    sentinel-c0) run_arm sentinel c0 ;;
    freeze-b0-sentinel)
        "${CSCR_PYTHON}" "${REPO_ROOT}/tools/certa_egra_artifacts.py" freeze-b0 \
            --runtime "${DEV_RUNTIME}" --limit 24 \
            --predictions "${EXPECTED_EXTERNAL_ROOT}/runs/dev/c0/predictions.debug.jsonl" \
            --output "${B0_ROWS}"
        ;;
    sentinel-c2) run_arm sentinel c2 ;;
    early-gate)
        "${CSCR_PYTHON}" "${REPO_ROOT}/tools/certa_egra_artifacts.py" constructor-master \
            --runtime "${DEV_RUNTIME}" --split dev \
            --predictions "${EXPECTED_EXTERNAL_ROOT}/runs/dev/c0/predictions.debug.jsonl" \
            --predictions "${EXPECTED_EXTERNAL_ROOT}/runs/dev/c2/predictions.debug.jsonl" \
            --output "${EXPECTED_EXTERNAL_ROOT}/results/constructor_sample_master.early.blind.jsonl"
        "${CSCR_PYTHON}" "${PACK_ROOT}/tools/compute_egra_early_sentinel.py" \
            --sample-master "${EXPECTED_EXTERNAL_ROOT}/results/constructor_sample_master.early.blind.jsonl" \
            --output "${EARLY_GATE}"
        ;;
    role-dev)
        require_early_pass
        verify_endpoint
        "${CSCR_PYTHON}" "${REPO_ROOT}/tools/certa_egra_artifacts.py" freeze-role \
            --runtime "${DEV_RUNTIME}" --limit 64 --resume \
            --output "${ROLE_ROWS}" \
            --manifest "${EXPECTED_EXTERNAL_ROOT}/freeze/ROLE_CONTRACT_FREEZE.json" \
            --cache "${EXPECTED_EXTERNAL_ROOT}/cache/qwen3_8b_nonthinking.jsonl"
        ;;
    dev-c0) run_arm dev c0 ;;
    freeze-b0-dev)
        require_early_pass
        "${CSCR_PYTHON}" "${REPO_ROOT}/tools/certa_egra_artifacts.py" freeze-b0 \
            --runtime "${DEV_RUNTIME}" \
            --predictions "${EXPECTED_EXTERNAL_ROOT}/runs/dev/c0/predictions.debug.jsonl" \
            --output "${B0_ROWS}" --replace
        ;;
    dev-c1) run_arm dev c1 ;;
    dev-c2) run_arm dev c2 ;;
    constructor-gate)
        require_early_pass
        "${CSCR_PYTHON}" "${REPO_ROOT}/tools/certa_egra_artifacts.py" constructor-master \
            --runtime "${DEV_RUNTIME}" --split dev \
            --predictions "${EXPECTED_EXTERNAL_ROOT}/runs/dev/c0/predictions.debug.jsonl" \
            --predictions "${EXPECTED_EXTERNAL_ROOT}/runs/dev/c1/predictions.debug.jsonl" \
            --predictions "${EXPECTED_EXTERNAL_ROOT}/runs/dev/c2/predictions.debug.jsonl" \
            --output "${EXPECTED_EXTERNAL_ROOT}/results/constructor_sample_master.blind.jsonl"
        "${CSCR_PYTHON}" "${REPO_ROOT}/tools/certa_egra_artifacts.py" unblind-constructor \
            --blind "${EXPECTED_EXTERNAL_ROOT}/results/constructor_sample_master.blind.jsonl" \
            --gold "${SEALED_ROOT}/dev_gold.jsonl" \
            --output "${EXPECTED_EXTERNAL_ROOT}/results/constructor_sample_master.dev_unblind.jsonl"
        method_sha="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
        "${CSCR_PYTHON}" "${PACK_ROOT}/tools/compute_egra_constructor_gate.py" \
            --sample-master "${EXPECTED_EXTERNAL_ROOT}/results/constructor_sample_master.dev_unblind.jsonl" \
            --method-sha "${method_sha}" \
            --output "${EXPECTED_EXTERNAL_ROOT}/results/CONSTRUCTOR_GATE.json"
        ;;
    *) usage; exit 2 ;;
esac
