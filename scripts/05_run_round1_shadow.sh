#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/hsh/ME/Table/EMNLP2026/CERTA"
PROFILE="${REPO_ROOT}/configs/profiles/certa_round1_shadow.env"
TABLE_DIR="/home/hsh/ME/Table/EMNLP2026/CausalityAwareTableQA/dataset/hitab/tables/raw"

usage() {
    echo "usage: CERTA_R1_EXTERNAL_ROOT=/absolute/path $0 {primary|value_aware|proposal_aware}" >&2
}

if [ "$#" -ne 1 ]; then
    usage
    exit 2
fi
if [ -z "${CERTA_R1_EXTERNAL_ROOT:-}" ] || [ "${CERTA_R1_EXTERNAL_ROOT#/}" = "${CERTA_R1_EXTERNAL_ROOT}" ]; then
    echo "CERTA_R1_EXTERNAL_ROOT must be an absolute external output root" >&2
    exit 2
fi

arm="$1"
case "${arm}" in
    primary)
        planner_boundary="proposal_blind_schema_only"
        input_file="${CERTA_R1_EXTERNAL_ROOT}/inputs/dev_blind.jsonl"
        bounded_args=()
        ;;
    value_aware)
        planner_boundary="proposal_blind_value_aware"
        input_file="${CERTA_R1_EXTERNAL_ROOT}/inputs/dev_diag8_blind.jsonl"
        bounded_args=(--limit 8)
        ;;
    proposal_aware)
        planner_boundary="proposal_aware_diagnostic"
        input_file="${CERTA_R1_EXTERNAL_ROOT}/inputs/dev_diag8_blind.jsonl"
        bounded_args=(--limit 8)
        ;;
    *)
        usage
        exit 2
        ;;
esac

output_dir="${CERTA_R1_EXTERNAL_ROOT}/runs/${arm}"
cache_path="${CERTA_R1_EXTERNAL_ROOT}/cache/qwen3_8b_nonthinking.jsonl"
log_file="${CERTA_R1_EXTERNAL_ROOT}/logs/run_${arm}.log"
if [ ! -f "${input_file}" ]; then
    echo "missing frozen input: ${input_file}" >&2
    exit 2
fi
if [ -e "${output_dir}" ]; then
    echo "refusing to overwrite existing output: ${output_dir}" >&2
    exit 2
fi

# shellcheck disable=SC1090
source "${PROFILE}"
export CSCR_INPUT_FILE="${input_file}"
export CSCR_TABLE_DIR="${TABLE_DIR}"
export CSCR_OUTPUT_DIR="${output_dir}"
export CSCR_API_CACHE_PATH="${cache_path}"
export CSCR_LLM_INPUT_AUDIT_FILE="llm_input_audit.jsonl"
mkdir -p "${CERTA_R1_EXTERNAL_ROOT}/runs" "${CERTA_R1_EXTERNAL_ROOT}/cache" "${CERTA_R1_EXTERNAL_ROOT}/logs"

command=(
    bash "${REPO_ROOT}/run_cscr.sh" full_cert
    --enable-cera-repair
    --cera-stage E71
    --cera-shadow-only
    --cera-round6-e71-v4
    --cera-enable-typed-planner
    --cera-planner-boundary "${planner_boundary}"
    --cera-planner-contract rcpc_signature_v2
    --cera-planner-legacy-query-semantics-mode audit_only
    --cera-planner-signature-allowlist "${CSCR_CERA_PLANNER_SIGNATURE_ALLOWLIST}"
    --cera-planner-temperature 0
    --cera-planner-max-tokens 512
    --cera-template-version cera_repair_v3
    --cera-log-evidence-packet
    "${bounded_args[@]}"
)

printf 'ROUND1_COMMAND'
printf ' %q' "${command[@]}"
printf '\n'
"${command[@]}" 2>&1 | tee "${log_file}"
