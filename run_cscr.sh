#!/usr/bin/env bash
# =============================================================================
# run_cscr.sh — CSCR 实验启动脚本
#
# 用法:
#   bash run_cscr.sh <MODE> [OPTIONS]
#
# 模式:
#   recalculate      从 Baseline E 预测重新计算四口径 EM
#   baseline_a_plus  结构感知 Prompt + logit 熵校准
#   executor_only    结构感知 Prompt + 执行器验证
#   full             完整 CSCR 管线 (v3 仲裁)
#   full_cert        CSCR + Certificate Matrix + Dominance 决策 (Phase 6)
#   dry_run          仅生成 prompt 预览 (不加载模型)
#   all              依次运行所有模式
#   ablation_no_exec 消融: 全量 baseline_a_plus (无执行器，作为上界参照)
#
# 示例:
#   bash run_cscr.sh recalculate
#   bash run_cscr.sh baseline_a_plus
#   bash run_cscr.sh full --limit 100
#   bash run_cscr.sh all
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# 配置 (根据环境修改)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${CSCR_PROJECT_DIR:-${SCRIPT_DIR}}"
MODEL_PATH="${CSCR_MODEL_PATH:-/data1/hesihao1/text_llm/Qwen/Qwen2.5-7B-Instruct}"
ASTRA_DATASET_ROOT="${CSCR_ASTRA_DATASET_ROOT:-/home/hsh/ME/Table/EMNLP2026/CausalityAwareTableQA/dataset}"
PYTHON_BIN="${CSCR_PYTHON:-${CONDA_PREFIX:-/home/hsh/anaconda3/envs/table}/bin/python}"
PYTHON_ENV_BIN="$(dirname "${PYTHON_BIN}")"
export PATH="${PYTHON_ENV_BIN}:${PATH}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
CSCR_DATASET="${CSCR_DATASET:-hitab}"
case "${CSCR_DATASET}" in
    hitab|hi-tab)
        CSCR_DATASET="hitab"
        DEFAULT_INPUT_FILE="${ASTRA_DATASET_ROOT}/hitab/test_samples_clean.jsonl"
        DEFAULT_TABLE_DIR="${ASTRA_DATASET_ROOT}/hitab/tables/raw"
        DEFAULT_OUTPUT_BASE="${PROJECT_DIR}/outputs/cscr"
        DEFAULT_BASELINE_E_PREDS="${PROJECT_DIR}/outputs/ca2kg_tables/hitab_qwen7b_frequency_full_sequential/predictions.jsonl"
        ;;
    aitqa|ait-qa)
        CSCR_DATASET="aitqa"
        DEFAULT_INPUT_FILE="${ASTRA_DATASET_ROOT}/AIT-QA/aitqa_clean_questions.json"
        DEFAULT_TABLE_DIR="${ASTRA_DATASET_ROOT}/AIT-QA"
        DEFAULT_OUTPUT_BASE="${PROJECT_DIR}/outputs/cscr/aitqa"
        DEFAULT_BASELINE_E_PREDS="${PROJECT_DIR}/outputs/cscr/aitqa/baseline_e_predictions.jsonl"
        ;;
    tablebench|table-bench)
        CSCR_DATASET="tablebench"
        DEFAULT_INPUT_FILE="/data1/hesihao1/tableqa/test_tableqaevaluation/data/tablebench/test.jsonl"
        DEFAULT_TABLE_DIR="/data1/hesihao1/tableqa/test_tableqaevaluation/data/tablebench"
        DEFAULT_OUTPUT_BASE="${PROJECT_DIR}/outputs/cscr/tablebench"
        DEFAULT_BASELINE_E_PREDS="${PROJECT_DIR}/outputs/cscr/tablebench/baseline_e_predictions.jsonl"
        ;;
    *)
        echo "Unsupported CSCR_DATASET=${CSCR_DATASET}. Use hitab, aitqa, or tablebench." >&2
        exit 2
        ;;
esac
INPUT_FILE="${CSCR_INPUT_FILE:-${DEFAULT_INPUT_FILE}}"
TABLE_DIR="${CSCR_TABLE_DIR:-${DEFAULT_TABLE_DIR}}"
OUTPUT_BASE="${CSCR_OUTPUT_BASE:-${DEFAULT_OUTPUT_BASE}}"
BASELINE_E_PREDS="${CSCR_BASELINE_E_PREDS:-${DEFAULT_BASELINE_E_PREDS}}"

prepend_ld_path() {
    local candidate="$1"
    if [ -z "${candidate}" ] || [ ! -d "${candidate}" ]; then
        return
    fi
    case ":${LD_LIBRARY_PATH:-}:" in
        *":${candidate}:"*) ;;
        *) export LD_LIBRARY_PATH="${candidate}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
    esac
}

normalize_ld_path() {
    local old_path="${LD_LIBRARY_PATH:-}"
    local new_path=""
    local part
    IFS=':' read -ra _ld_parts <<< "$old_path"
    for part in "${_ld_parts[@]}"; do
        if [ -z "$part" ]; then
            continue
        fi
        case ":${new_path}:" in
            *":${part}:"*) ;;
            *) new_path="${new_path}${new_path:+:}${part}" ;;
        esac
    done
    if [ -n "$new_path" ]; then
        export LD_LIBRARY_PATH="$new_path"
    else
        unset LD_LIBRARY_PATH
    fi
}

# Keep CUDA / conda shared-library lookup deterministic for vLLM native wheels.
prepend_ld_path "${CUDA_HOME:-/usr/local/cuda-11.8}/lib64"
prepend_ld_path "${CONDA_PREFIX:-/home/hsh/anaconda3/envs/table}/lib"
normalize_ld_path

# ---------------------------------------------------------------------------
# 多卡配置
# ---------------------------------------------------------------------------
# CSCR_GPUS: 逗号分隔的 GPU 编号列表，例如 "0,1,2,3"（多卡）或 "4"（单卡）
# 内存保守默认使用 2 卡；需要 4 卡吞吐时启动前覆盖 CSCR_GPUS 和 CSCR_TP。
CSCR_GPUS="${CSCR_GPUS:-0,4}"
BANNED_GPUS="${CSCR_BANNED_GPUS-}"
ENABLE_GPU_BAN_GUARD="${CSCR_ENABLE_GPU_BAN_GUARD:-1}"

gpu_list_contains() {
    local haystack=",$1,"
    local needle="$2"
    [[ "${haystack}" == *",${needle},"* ]]
}

if [ "${ENABLE_GPU_BAN_GUARD}" = "1" ]; then
    for banned_gpu in $(echo "${BANNED_GPUS}" | tr ',' ' '); do
        if gpu_list_contains "${CSCR_GPUS}" "${banned_gpu}"; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: CSCR_GPUS=${CSCR_GPUS} contains banned GPU ${banned_gpu}. Set CSCR_ENABLE_GPU_BAN_GUARD=0 to override intentionally." >&2
            exit 2
        fi
    done
fi
export CUDA_VISIBLE_DEVICES="${CSCR_GPUS}"

# TENSOR_PARALLEL: 自动计算为 GPU 数量（逗号分隔的数量）
# 也可通过 CSCR_TP 手动覆盖
_AUTO_TP=$(echo "${CSCR_GPUS}" | tr ',' '\n' | wc -l | tr -d ' ')
TENSOR_PARALLEL="${CSCR_TP:-${_AUTO_TP}}"
if [ "${TENSOR_PARALLEL}" -gt "${_AUTO_TP}" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: CSCR_TP=${TENSOR_PARALLEL} exceeds visible GPU count ${_AUTO_TP} from CSCR_GPUS=${CSCR_GPUS}." >&2
    exit 2
fi

# 显存利用率：默认使用高吞吐档；可用 CSCR_GPU_MEM_UTIL 覆盖。
GPU_MEM_UTIL="${CSCR_GPU_MEM_UTIL:-0.85}"

# 数据类型：大模型推荐 bfloat16
DTYPE="${CSCR_DTYPE:-bfloat16}"

# 最大并发序列数：控制 vLLM 同时驻留的序列/KV cache，内存紧张时保持较小
MAX_NUM_SEQS="${CSCR_MAX_NUM_SEQS:-8}"

# 每次调度的最大 token 数：进一步限制峰值 KV cache / activation 压力
MAX_NUM_BATCHED_TOKENS="${CSCR_MAX_NUM_BATCHED_TOKENS:-24576}"

# vLLM CPU 侧资源：swap_space 是每张 GPU 的 CPU swap GiB，过大会吃节点内存
SWAP_SPACE="${CSCR_SWAP_SPACE:-1}"
CPU_OFFLOAD_GB="${CSCR_CPU_OFFLOAD_GB:-0}"
DISABLE_CUSTOM_ALL_REDUCE="${CSCR_DISABLE_CUSTOM_ALL_REDUCE:-1}"
ENABLE_CHUNKED_PREFILL="${CSCR_ENABLE_CHUNKED_PREFILL:-1}"
KV_CACHE_DTYPE="${CSCR_KV_CACHE_DTYPE:-auto}"
USE_FAST_IMAGE_PROCESSOR="${CSCR_USE_FAST_IMAGE_PROCESSOR:-1}"
DISTRIBUTED_EXECUTOR_BACKEND="${CSCR_DISTRIBUTED_EXECUTOR_BACKEND:-mp}"
if [ -n "${CSCR_ENFORCE_EAGER+x}" ]; then
    ENFORCE_EAGER="${CSCR_ENFORCE_EAGER}"
elif [ "${TENSOR_PARALLEL}" -gt 1 ]; then
    ENFORCE_EAGER="1"
else
    ENFORCE_EAGER="0"
fi

# Set this before vLLM imports CUDA to avoid the "overriding to spawn" warning.
export VLLM_USE_V1="${VLLM_USE_V1:-0}"
export CSCR_DISABLE_VLLM_PYNCCL="${CSCR_DISABLE_VLLM_PYNCCL:-1}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_ENGINE_ITERATION_TIMEOUT_S="${VLLM_ENGINE_ITERATION_TIMEOUT_S:-1800}"
export VLLM_RPC_TIMEOUT="${VLLM_RPC_TIMEOUT:-1800000}"
export RAY_CGRAPH_get_timeout="${RAY_CGRAPH_get_timeout:-1800}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"

is_physical_net_if() {
    case "${1:-}" in
        ""|lo|docker*|br-*|veth*)
            return 1
            ;;
        *)
            return 0
            ;;
    esac
}

detect_nccl_socket_ifname() {
    local route_if=""
    if command -v ip >/dev/null 2>&1; then
        route_if="$(ip route get 8.8.8.8 2>/dev/null | awk '{for (i=1; i<=NF; i++) if ($i=="dev") {print $(i+1); exit}}' || true)"
        if is_physical_net_if "${route_if}"; then
            echo "${route_if}"
            return 0
        fi
        route_if="$(ip -o -4 addr show scope global 2>/dev/null | awk '$2 !~ /^(lo|docker|br-|veth)/ {print $2; exit}' || true)"
        if is_physical_net_if "${route_if}"; then
            echo "${route_if}"
            return 0
        fi
    fi
    return 1
}

# NCCL configuration for multi-GPU with non-contiguous device IDs.
# Prefer the real route interface; container bridge interfaces such as docker0
# can make NCCL bootstrap pick an unusable path on single-node jobs.
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
if [ -z "${NCCL_SOCKET_IFNAME:-}" ]; then
    _AUTO_NCCL_SOCKET_IFNAME="$(detect_nccl_socket_ifname || true)"
    export NCCL_SOCKET_IFNAME="${_AUTO_NCCL_SOCKET_IFNAME:-^lo,docker,br,veth}"
else
    export NCCL_SOCKET_IFNAME
fi
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_NET_PLUGIN="${NCCL_NET_PLUGIN:-none}"
# Force NCCL to use visible devices only (critical for non-contiguous GPU IDs)
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
# Additional NCCL settings for stability
export NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-4}"
export NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-8}"
export NCCL_BUFFSIZE="${NCCL_BUFFSIZE:-8388608}"
# Use loopback if no network interface is available
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"

# Use flashinfer's sampler when installed; vLLM falls back if the env is unset.
if [ "${CSCR_USE_FLASHINFER_SAMPLER:-1}" = "1" ]; then
    export VLLM_USE_FLASHINFER_SAMPLER=1
fi

# 是否启用批量推理（将一批 prompt 一次性送入 vLLM，默认开启）
BATCH_INFERENCE="${CSCR_BATCH_INFERENCE:-1}"

# Prompt 风格：baseline_e | structure_aware | table_focus | table_pruned | selective_evidence | scm_cot
# v8.3 主线：baseline_e（32B Qwen2.5-Instruct 上 EM=73.42%）
PROMPT_STYLE="${CSCR_PROMPT_STYLE:-structure_aware}"

# v10/v11: provenance-aware structural certification. 默认诊断先行，避免改变检索行为。
STRUCTURAL_PRIOR_WEIGHTING="${CSCR_STRUCTURAL_PRIOR_WEIGHTING:-0}"
DISABLE_CANDIDATE_SCCI="${CSCR_DISABLE_CANDIDATE_SCCI:-0}"

# v8.6: Credal Probe 诊断层（只读旁路，不改变答案，为 v9.x meta-controller 收集数据）
# 0=关闭，1=开启
CREDAL_PROBE="${CSCR_CREDAL_PROBE:-0}"

# v8.9: Credal-Aware APR Routing — 反向门控（v8.8 的 floor 错误，改为 cap）
# 语义：cw >= CREDAL_GATE_CW_HIGH 时跳过 R2（高不确定性区间内 R2 几乎不可能改善）
# 0=关闭（保持 v8.3 纯熵路由），1=开启
CREDAL_GATE="${CSCR_CREDAL_GATE:-0}"
# v8.9 cap 阈值（向上的截断；超过此值不进 R2）
CREDAL_GATE_CW_HIGH="${CSCR_CREDAL_GATE_CW_HIGH:-0.30}"
# v8.8 floor 阈值（仅当 CSCR_CREDAL_GATE_MODE=floor 时启用）— 已证明有害，默认禁用
CREDAL_GATE_CW="${CSCR_CREDAL_GATE_CW:-0.10}"
# 模式：cap（默认，>= cw_high 跳过 R2）/ floor（v8.8 错误模式，>= cw 才进 R2）/ band（在 [cw_low, cw_high] 内才进 R2）
CREDAL_GATE_MODE="${CSCR_CREDAL_GATE_MODE:-cap}"
# v8.8: Non-degradation guard（保护 R1 path_consensus + R2 低置信回退）
NON_DEGRADATION_GUARD="${CSCR_NON_DEGRADATION_GUARD:-0}"

# v9.0b: Question-Type Router（根据 coarse_question_type 路由 prompt_style）
# lookup/proportion/superlative → table_focus; 其他 → baseline_e
# 0=关闭，1=开启
QUESTION_TYPE_ROUTER="${CSCR_QUESTION_TYPE_ROUTER:-0}"

# v9.0b: Online Normalizer（gold-free 表面格式归一化）
# 0=关闭，1=开启
ONLINE_NORMALIZER="${CSCR_ONLINE_NORMALIZER:-0}"
ORACLE_ONLINE_NORMALIZER="${CSCR_ORACLE_ONLINE_NORMALIZER:-0}"

# v9.1: HCEG-Fallback（高 cw 或 compare 错时生成 KG 直检候选）
# 0=关闭，1=开启
HCEG_FALLBACK="${CSCR_HCEG_FALLBACK:-0}"
HCEG_FALLBACK_CW="${CSCR_HCEG_FALLBACK_CW:-0.30}"
HCEG_FALLBACK_COMPARE_CW="${CSCR_HCEG_FALLBACK_COMPARE_CW:-0.15}"
HCEG_FALLBACK_DIFF_CW="${CSCR_HCEG_FALLBACK_DIFF_CW:-0.10}"
HCEG_FALLBACK_POLICY="${CSCR_HCEG_FALLBACK_POLICY:-candidate_only}"
HCEG_ROLE_AWARE="${CSCR_HCEG_ROLE_AWARE:-1}"
HCEG_DIAGNOSTIC_CANDIDATES="${CSCR_HCEG_DIAGNOSTIC_CANDIDATES:-triggered}"
CERT_COMMIT_BOUNDARY="${CSCR_CERT_COMMIT_BOUNDARY:-0}"
CERT_COMMIT_MODE="${CSCR_CERT_COMMIT_MODE:-diagnostic}"
CERT_COMMIT_MAX_LLM_CONFIDENCE="${CSCR_CERT_COMMIT_MAX_LLM_CONFIDENCE:-1.0}"
CERT_COMMIT_MIN_CREDAL_WIDTH="${CSCR_CERT_COMMIT_MIN_CREDAL_WIDTH:-0.0}"
CERT_COMMIT_ALLOW_DIAGNOSTIC="${CSCR_CERT_COMMIT_ALLOW_DIAGNOSTIC:-0}"
CERT_OPERATION_VERIFIER="${CSCR_CERT_OPERATION_VERIFIER:-0}"
CERT_COMPARE_DIRECTION_VERIFIER="${CSCR_CERT_COMPARE_DIRECTION_VERIFIER:-0}"
CERT_NUMERIC_DIRECTION_VERIFIER="${CSCR_CERT_NUMERIC_DIRECTION_VERIFIER:-0}"
CERT_CONFORMAL_BOUNDARY="${CSCR_CERT_CONFORMAL_BOUNDARY:-0}"
CERT_CONFORMAL_THRESHOLD="${CSCR_CERT_CONFORMAL_THRESHOLD:-1.01}"
CERT_CONFORMAL_ALPHA="${CSCR_CERT_CONFORMAL_ALPHA:-0.10}"

# v9.2: Diverse Self-Consistency（风险样本多 prompt 候选投票）
SELF_CONSISTENCY="${CSCR_SELF_CONSISTENCY:-0}"
K_SAMPLES="${CSCR_K_SAMPLES:-3}"
SELF_CONSISTENCY_TEMPERATURE="${CSCR_SELF_CONSISTENCY_TEMPERATURE:-0.35}"
SELF_CONSISTENCY_MAX_SAMPLES="${CSCR_SELF_CONSISTENCY_MAX_SAMPLES:-512}"
SELF_CONSISTENCY_TRIGGER="${CSCR_SELF_CONSISTENCY_TRIGGER:-hceg}"

# Full-input compute reachability: prefix-stable APR/SC
PREFIX_STABLE_APR="${CSCR_PREFIX_STABLE_APR:-1}"
APR_CONTROL_SUFFIX_MODE="${CSCR_APR_CONTROL_SUFFIX_MODE:-intersection_hint}"

# ---------------------------------------------------------------------------
# v8.3: Adaptive Prompt Router (APR) 配置 — 纯熵路由
# ---------------------------------------------------------------------------
# 是否启用 APR（首 token 熵路由），0=关闭，1=开启
# v8.3 修复历程: v8.1 三段路由→v8.2 新增 conflict 路由(净损-26/-11)→v8.3 纯熵路由
ADAPTIVE_PROMPT="${CSCR_ADAPTIVE_PROMPT:-0}"
# 低熵阈值（低于此值保持 Round 1 答案，高于此值用 intersection_hint 重推理）
ENTROPY_THRESHOLD_LOW="${CSCR_ENTROPY_LOW:-0.05}"
# 高熵阈值（v8.3 不使用，保留参数向后兼容）
ENTROPY_THRESHOLD_HIGH="${CSCR_ENTROPY_HIGH:-0.20}"

# 推理参数
if [ -n "${CSCR_MAX_LEN:-}" ]; then
    MAX_MODEL_LEN="${CSCR_MAX_LEN}"
else
    _MODEL_PATH_LOWER="$(echo "${MODEL_PATH}" | tr '[:upper:]' '[:lower:]')"
    MAX_MODEL_LEN=8192
fi
BATCH_SIZE=${CSCR_BATCH:-32}
MAX_ANSWER_TOKENS="${CSCR_MAX_ANSWER_TOKENS:-32}"
TEMPERATURE="${CSCR_TEMPERATURE:-0.0}"
TOP_P="${CSCR_TOP_P:-1.0}"
TOP_K_LOGPROBS="${CSCR_TOP_K_LOGPROBS:-5}"
SEED="${CSCR_SEED:-0}"
GENERATOR_BACKEND="${CSCR_GENERATOR_BACKEND:-vllm}"
case "${GENERATOR_BACKEND}" in
    vllm|openai_chat|gemini_chat|vllm_chat) ;;
    *)
        echo "Unsupported CSCR_GENERATOR_BACKEND=${GENERATOR_BACKEND}. Use vllm, openai_chat, gemini_chat, or vllm_chat." >&2
        exit 2
        ;;
esac
if [ -z "${CSCR_API_BASE_URL+x}" ] && [ "${GENERATOR_BACKEND}" = "vllm_chat" ]; then
    API_BASE_URL="http://127.0.0.1:30300/v1"
else
    API_BASE_URL="${CSCR_API_BASE_URL:-https://api.lkeap.cloud.tencent.com/v1}"
fi
if [ -z "${CSCR_API_KEY_ENV+x}" ] && [ "${GENERATOR_BACKEND}" = "vllm_chat" ]; then
    API_KEY_ENV="EMPTY"
else
    API_KEY_ENV="${CSCR_API_KEY_ENV:-LKEAP_API_KEY}"
fi
if [ -z "${CSCR_API_MODEL+x}" ] && [ "${GENERATOR_BACKEND}" = "vllm_chat" ]; then
    API_MODEL="$(basename "${MODEL_PATH%/}")"
else
    API_MODEL="${CSCR_API_MODEL:-${MODEL_PATH}}"
fi
API_TIMEOUT="${CSCR_API_TIMEOUT:-120}"
API_MAX_RETRIES="${CSCR_API_MAX_RETRIES:-3}"
API_RATE_LIMIT_SECONDS="${CSCR_API_RATE_LIMIT_SECONDS:-0}"
API_CACHE_PATH="${CSCR_API_CACHE_PATH:-}"
API_CACHE_MODE="${CSCR_API_CACHE_MODE:-readwrite}"
SAVE_LLM_INPUTS="${CSCR_SAVE_LLM_INPUTS:-full}"
LLM_INPUT_AUDIT_FILE="${CSCR_LLM_INPUT_AUDIT_FILE:-llm_input_audit.jsonl}"
case "${SAVE_LLM_INPUTS}" in
    off|hash|full) ;;
    *)
        echo "Unsupported CSCR_SAVE_LLM_INPUTS=${SAVE_LLM_INPUTS}. Use off, hash, or full." >&2
        exit 2
        ;;
esac
BLACK_BOX_COMMIT_POLICY="${CSCR_BLACK_BOX_COMMIT_POLICY:-auto}"
API_FORMAT_NORMALIZER="${CSCR_API_FORMAT_NORMALIZER:-auto}"
SURFACE_HEURISTIC_MODE="${CSCR_SURFACE_HEURISTIC_MODE:-diagnostic}"
case "${SURFACE_HEURISTIC_MODE}" in
    off|diagnostic|legacy) ;;
    *)
        echo "Unsupported CSCR_SURFACE_HEURISTIC_MODE=${SURFACE_HEURISTIC_MODE}. Use off, diagnostic, or legacy." >&2
        exit 2
        ;;
esac
DATASET_PROMPT_POLICY="${CSCR_DATASET_PROMPT_POLICY:-auto}"
case "${DATASET_PROMPT_POLICY}" in
    auto|legacy|benchmark|operation) ;;
    *)
        echo "Unsupported CSCR_DATASET_PROMPT_POLICY=${DATASET_PROMPT_POLICY}. Use auto, legacy, benchmark, or operation." >&2
        exit 2
        ;;
esac
SOURCE_RISK_CALIBRATION="${CSCR_SOURCE_RISK_CALIBRATION:-auto}"
case "${SOURCE_RISK_CALIBRATION}" in
    auto|off|tablebench|all) ;;
    *)
        echo "Unsupported CSCR_SOURCE_RISK_CALIBRATION=${SOURCE_RISK_CALIBRATION}. Use auto, off, tablebench, or all." >&2
        exit 2
        ;;
esac
SOURCE_RISK_LLM_CERT_ADJUSTED_CAP="${CSCR_SOURCE_RISK_LLM_CERT_ADJUSTED_CAP:-0.74}"
OPERATION_SUPPORT_DIAGNOSTICS="${CSCR_OPERATION_SUPPORT_DIAGNOSTICS:-0}"
OPERATION_ROLE_TARGET_DIAGNOSTICS="${CSCR_OPERATION_ROLE_TARGET_DIAGNOSTICS:-0}"
OPERATION_COMMIT_GATE_DIAGNOSTICS="${CSCR_OPERATION_COMMIT_GATE_DIAGNOSTICS:-0}"
OPERATION_COMMIT_GATE_MODE="${CSCR_OPERATION_COMMIT_GATE_MODE:-diagnostic}"
OPERATION_COMMIT_VERSION="${CSCR_OPERATION_COMMIT_VERSION:-E67}"
OPERATION_CERTIFICATE_PROFILE="${CSCR_OPERATION_CERTIFICATE_PROFILE:-}"
MAIN_CERT_PROFILE="${CSCR_MAIN_CERT_PROFILE:-0}"
case "${OPERATION_COMMIT_VERSION}" in
    E65.3|E65.4|E67) ;;
    *)
        echo "Unsupported CSCR_OPERATION_COMMIT_VERSION=${OPERATION_COMMIT_VERSION}. Use E65.3, E65.4, or E67." >&2
        exit 2
        ;;
esac
case "${MAIN_CERT_PROFILE}" in
    0|1) ;;
    *)
        echo "Unsupported CSCR_MAIN_CERT_PROFILE=${MAIN_CERT_PROFILE}. Use 0 or 1." >&2
        exit 2
        ;;
esac
OPERATION_COMMIT_DATASET_SCOPE="${CSCR_OPERATION_COMMIT_DATASET_SCOPE:-all}"
case "${OPERATION_CERTIFICATE_PROFILE}" in
    "")
        ;;
    strict)
        SURFACE_HEURISTIC_MODE="diagnostic"
        OPERATION_SUPPORT_DIAGNOSTICS=1
        OPERATION_ROLE_TARGET_DIAGNOSTICS=1
        OPERATION_COMMIT_GATE_DIAGNOSTICS=1
        OPERATION_COMMIT_GATE_MODE="conservative"
        BLACK_BOX_COMMIT_POLICY="certified"
        ;;
    diagnostic)
        SURFACE_HEURISTIC_MODE="diagnostic"
        OPERATION_SUPPORT_DIAGNOSTICS=1
        OPERATION_ROLE_TARGET_DIAGNOSTICS=1
        OPERATION_COMMIT_GATE_DIAGNOSTICS=1
        OPERATION_COMMIT_GATE_MODE="diagnostic"
        BLACK_BOX_COMMIT_POLICY="certified"
        ;;
    *)
        echo "Unsupported CSCR_OPERATION_CERTIFICATE_PROFILE=${OPERATION_CERTIFICATE_PROFILE}. Use strict or diagnostic." >&2
        exit 2
        ;;
esac
if [ "${MAIN_CERT_PROFILE}" = "1" ]; then
    SURFACE_HEURISTIC_MODE="diagnostic"
    ADAPTIVE_PROMPT=0
    PREFIX_STABLE_APR=0
    CREDAL_PROBE=0
    CREDAL_GATE=0
    NON_DEGRADATION_GUARD=0
    QUESTION_TYPE_ROUTER=0
    ONLINE_NORMALIZER=0
    ORACLE_ONLINE_NORMALIZER=0
    HCEG_FALLBACK=0
    CERT_COMMIT_BOUNDARY=0
    SELF_CONSISTENCY=0
    SOURCE_RISK_CALIBRATION="off"
    API_FORMAT_NORMALIZER="off"
    BLACK_BOX_COMMIT_POLICY="certified"
    OPERATION_SUPPORT_DIAGNOSTICS=1
    OPERATION_ROLE_TARGET_DIAGNOSTICS=1
    OPERATION_COMMIT_GATE_DIAGNOSTICS=1
fi
ABORT_ON_GENERATION_ERROR="${CSCR_ABORT_ON_GENERATION_ERROR:-1}"
SKIP_OVERLONG_PRIMARY="${CSCR_SKIP_OVERLONG_PRIMARY:-0}"

# 打印多卡配置信息
echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: GPU 配置: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, tensor_parallel_size=${TENSOR_PARALLEL}, banned_gpus=${BANNED_GPUS}, gpu_ban_guard=${ENABLE_GPU_BAN_GUARD}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: Python: ${PYTHON_BIN}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: 数据集配置: dataset=${CSCR_DATASET}, input=${INPUT_FILE}, table_dir=${TABLE_DIR}, output_base=${OUTPUT_BASE}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: 推理配置: backend=${GENERATOR_BACKEND}, api_model=${API_MODEL}, api_base_url=${API_BASE_URL}, api_key_env=${API_KEY_ENV}, api_cache_mode=${API_CACHE_MODE},llm_input_audit=${SAVE_LLM_INPUTS}, llm_input_audit_file=${LLM_INPUT_AUDIT_FILE}, black_box_commit_policy=${BLACK_BOX_COMMIT_POLICY}, api_format_normalizer=${API_FORMAT_NORMALIZER}, surface_heuristic_mode=${SURFACE_HEURISTIC_MODE}, dtype=${DTYPE}, max_model_len=${MAX_MODEL_LEN}, max_answer_tokens=${MAX_ANSWER_TOKENS}, temperature=${TEMPERATURE}, top_p=${TOP_P}, top_k_logprobs=${TOP_K_LOGPROBS}, seed=${SEED}, dataset_prompt_policy=${DATASET_PROMPT_POLICY}, source_risk_calibration=${SOURCE_RISK_CALIBRATION}@${SOURCE_RISK_LLM_CERT_ADJUSTED_CAP}, operation_support_diag=${OPERATION_SUPPORT_DIAGNOSTICS}, operation_role_target_diag=${OPERATION_ROLE_TARGET_DIAGNOSTICS}, operation_commit_gate_diag=${OPERATION_COMMIT_GATE_DIAGNOSTICS}, operation_commit_gate_mode=${OPERATION_COMMIT_GATE_MODE}, operation_commit_version=${OPERATION_COMMIT_VERSION}, operation_certificate_profile=${OPERATION_CERTIFICATE_PROFILE:-none}, main_cert_profile=${MAIN_CERT_PROFILE}, structural_prior_weighting=${STRUCTURAL_PRIOR_WEIGHTING}, disable_candidate_scci=${DISABLE_CANDIDATE_SCCI}, operation_commit_dataset_scope=${OPERATION_COMMIT_DATASET_SCOPE}, gpu_mem_util=${GPU_MEM_UTIL}, max_num_seqs=${MAX_NUM_SEQS}, max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS:-auto}, batch_size=${BATCH_SIZE}, batch_inference=${BATCH_INFERENCE}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: E67 证书: expression + measure/unit + measure-axis granularity + measure fiber + aggregate echo + candidate source stability + support necessity + answer projection + query-table entity/literal filter binding"
if [ "${OPERATION_COMMIT_GATE_MODE}" = "conservative" ] && [ "${BLACK_BOX_COMMIT_POLICY}" != "certified" ]; then
    case "${GENERATOR_BACKEND}" in
        openai_chat|vllm_chat|gemini_chat)
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARN: Structural certificate conservative commit with ${GENERATOR_BACKEND} is policy-blocked unless CSCR_BLACK_BOX_COMMIT_POLICY=certified; this run will remain shadow/diagnostic if the policy is not changed." >&2
            ;;
    esac
fi
echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: 资源配置: swap_space=${SWAP_SPACE}GiB/gpu, cpu_offload_gb=${CPU_OFFLOAD_GB}GiB/gpu, disable_custom_all_reduce=${DISABLE_CUSTOM_ALL_REDUCE}, disable_vllm_pynccl=${CSCR_DISABLE_VLLM_PYNCCL}, skip_overlong_primary=${SKIP_OVERLONG_PRIMARY}, enforce_eager=${ENFORCE_EAGER}, enable_chunked_prefill=${ENABLE_CHUNKED_PREFILL}, kv_cache_dtype=${KV_CACHE_DTYPE}, fast_image_processor=${USE_FAST_IMAGE_PROCESSOR}, distributed_executor_backend=${DISTRIBUTED_EXECUTOR_BACKEND}, OMP_NUM_THREADS=${OMP_NUM_THREADS:-unset}, RAY_memory_usage_threshold=${RAY_memory_usage_threshold:-unset}, VLLM_USE_V1=${VLLM_USE_V1}, VLLM_ENGINE_ITERATION_TIMEOUT_S=${VLLM_ENGINE_ITERATION_TIMEOUT_S}, VLLM_RPC_TIMEOUT=${VLLM_RPC_TIMEOUT}, RAY_CGRAPH_get_timeout=${RAY_CGRAPH_get_timeout}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: 路由配置: adaptive_prompt=${ADAPTIVE_PROMPT}, prefix_stable_apr=${PREFIX_STABLE_APR}, apr_control_suffix_mode=${APR_CONTROL_SUFFIX_MODE}, hceg_role_aware=${HCEG_ROLE_AWARE}, hceg_diag=${HCEG_DIAGNOSTIC_CANDIDATES}, cert_commit=${CERT_COMMIT_BOUNDARY}/${CERT_COMMIT_MODE}, cert_op_verifier=${CERT_OPERATION_VERIFIER}, cert_dir_verifier=${CERT_COMPARE_DIRECTION_VERIFIER}, cert_numdir_verifier=${CERT_NUMERIC_DIRECTION_VERIFIER}, cert_conformal=${CERT_CONFORMAL_BOUNDARY}@${CERT_CONFORMAL_THRESHOLD}, self_consistency=${SELF_CONSISTENCY}"
# ---------------------------------------------------------------------------
# 解析参数
# ---------------------------------------------------------------------------
MODE="${1:-help}"
shift || true
EXTRA_ARGS="$@"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ---------------------------------------------------------------------------
# 函数
# ---------------------------------------------------------------------------

log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $*"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $*" >&2
}

terminate_process_group() {
    local pgid="$1"
    if [ -z "${pgid}" ]; then
        return
    fi
    kill -TERM -- "-${pgid}" 2>/dev/null || true
    sleep 3
    kill -KILL -- "-${pgid}" 2>/dev/null || true
}

run_python_command() {
    local -a command=("$@")
    local status=0
    local child_pid=""

    if command -v setsid >/dev/null 2>&1; then
        set +e
        setsid "${command[@]}" ${EXTRA_ARGS} &
        child_pid=$!
        trap 'terminate_process_group "'"${child_pid}"'"; exit 130' INT
        trap 'terminate_process_group "'"${child_pid}"'"; exit 143' TERM
        wait "${child_pid}"
        status=$?
        trap - INT TERM
        set -e
        if [ "${status}" -ne 0 ]; then
            terminate_process_group "${child_pid}"
        fi
        return "${status}"
    fi

    "${command[@]}" ${EXTRA_ARGS}
}

run_mode() {
    local mode="$1"
    local output_dir="${CSCR_OUTPUT_DIR:-${OUTPUT_BASE}/${mode}_${TIMESTAMP}}"

    log_info "=========================================="
    log_info "Running CSCR mode: ${mode}"
    log_info "Output: ${output_dir}"
    log_info "=========================================="

    local cmd=(
        "${PYTHON_BIN}" "${PROJECT_DIR}/run_cscr_pipeline.py"
        --mode "${mode}"
        --dataset "${CSCR_DATASET}"
        --input_file "${INPUT_FILE}"
        --table_dir "${TABLE_DIR}"
        --output_dir "${output_dir}"
        --model_path "${MODEL_PATH}"
        --generator-backend "${GENERATOR_BACKEND}"
        --api-base-url "${API_BASE_URL}"
        --api-key-env "${API_KEY_ENV}"
        --api-model "${API_MODEL}"
        --api-timeout "${API_TIMEOUT}"
        --api-max-retries "${API_MAX_RETRIES}"
        --api-rate-limit-seconds "${API_RATE_LIMIT_SECONDS}"
        --api-cache-mode "${API_CACHE_MODE}"
        --save-llm-inputs "${SAVE_LLM_INPUTS}"
        --llm-input-audit-file "${LLM_INPUT_AUDIT_FILE}"
        --tensor_parallel_size "${TENSOR_PARALLEL}"
        --max_model_len "${MAX_MODEL_LEN}"
        --batch_size "${BATCH_SIZE}"
        --max_answer_tokens "${MAX_ANSWER_TOKENS}"
        --temperature "${TEMPERATURE}"
        --top_p "${TOP_P}"
        --top_k_logprobs "${TOP_K_LOGPROBS}"
        --seed "${SEED}"
        --dataset-prompt-policy "${DATASET_PROMPT_POLICY}"
        --source-risk-calibration "${SOURCE_RISK_CALIBRATION}"
        --source-risk-llm-cert-adjusted-cap "${SOURCE_RISK_LLM_CERT_ADJUSTED_CAP}"
        --black-box-commit-policy "${BLACK_BOX_COMMIT_POLICY}"
        --operation-commit-dataset-scope "${OPERATION_COMMIT_DATASET_SCOPE}"
        --operation-commit-version "${OPERATION_COMMIT_VERSION}"
        --api-format-normalizer "${API_FORMAT_NORMALIZER}"
        --surface-heuristic-mode "${SURFACE_HEURISTIC_MODE}"
        --dtype "${DTYPE}"
        --gpu-memory-utilization "${GPU_MEM_UTIL}"
        --max-num-seqs "${MAX_NUM_SEQS}"
        --swap-space "${SWAP_SPACE}"
        --cpu-offload-gb "${CPU_OFFLOAD_GB}"
        --kv-cache-dtype "${KV_CACHE_DTYPE}"
        --distributed-executor-backend "${DISTRIBUTED_EXECUTOR_BACKEND}"
        --prompt-style "${PROMPT_STYLE}"
    )

    if [ -n "${API_CACHE_PATH}" ]; then
        cmd+=(--api-cache-path "${API_CACHE_PATH}")
    fi

    if [ -n "${MAX_NUM_BATCHED_TOKENS}" ]; then
        cmd+=(--max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}")
    fi

    if [ "${DISABLE_CUSTOM_ALL_REDUCE}" = "1" ]; then
        cmd+=(--disable-custom-all-reduce)
    fi

    if [ "${ENFORCE_EAGER}" = "1" ]; then
        cmd+=(--enforce-eager)
    else
        cmd+=(--no-enforce-eager)
    fi

    if [ "${ENABLE_CHUNKED_PREFILL}" = "1" ]; then
        cmd+=(--enable-chunked-prefill)
    fi

    if [ "${USE_FAST_IMAGE_PROCESSOR}" = "1" ]; then
        cmd+=(--use-fast-image-processor)
    else
        cmd+=(--no-use-fast-image-processor)
    fi

    if [ "${ABORT_ON_GENERATION_ERROR}" = "1" ]; then
        cmd+=(--abort-on-generation-error)
    fi
    if [ "${SKIP_OVERLONG_PRIMARY}" = "1" ]; then
        cmd+=(--skip-overlong-primary)
    fi
    if [ "${OPERATION_SUPPORT_DIAGNOSTICS}" = "1" ]; then
        cmd+=(--operation-support-diagnostics)
    fi
    if [ "${OPERATION_ROLE_TARGET_DIAGNOSTICS}" = "1" ]; then
        cmd+=(--operation-support-diagnostics)
        cmd+=(--operation-role-target-diagnostics)
    fi
    if [ "${OPERATION_COMMIT_GATE_DIAGNOSTICS}" = "1" ]; then
        cmd+=(--operation-support-diagnostics)
        cmd+=(--operation-role-target-diagnostics)
        cmd+=(--operation-commit-gate-diagnostics)
        cmd+=(--operation-commit-gate-mode "${OPERATION_COMMIT_GATE_MODE}")
    fi
    if [ "${MAIN_CERT_PROFILE}" = "1" ]; then
        cmd+=(--main-cert-profile)
    fi
    if [ "${STRUCTURAL_PRIOR_WEIGHTING}" = "1" ]; then
        cmd+=(--structural-prior-weighting)
    fi
    if [ "${DISABLE_CANDIDATE_SCCI}" = "1" ]; then
        cmd+=(--disable-candidate-scci)
    fi

    # 批量推理模式
    if [ "${BATCH_INFERENCE}" = "1" ]; then
        cmd+=(--batch-inference)
    fi

    # v8.1: APR (Adaptive Prompt Router)
    if [ "${ADAPTIVE_PROMPT}" = "1" ]; then
        cmd+=(--adaptive-prompt)
        cmd+=(--entropy-threshold-low "${ENTROPY_THRESHOLD_LOW}")
        cmd+=(--entropy-threshold-high "${ENTROPY_THRESHOLD_HIGH}")
    fi

    if [ "${PREFIX_STABLE_APR}" = "1" ]; then
        cmd+=(--prefix-stable-apr)
        cmd+=(--apr-control-suffix-mode "${APR_CONTROL_SUFFIX_MODE}")
    fi

    # v8.6: Credal Probe 诊断层
    if [ "${CREDAL_PROBE}" = "1" ]; then
        cmd+=(--credal-probe)
    fi

    # v8.9: Credal-Aware APR Routing (cap/floor/band)
    if [ "${CREDAL_GATE}" = "1" ]; then
        cmd+=(--credal-gate)
        cmd+=(--credal-gate-mode "${CREDAL_GATE_MODE}")
        cmd+=(--credal-gate-cw "${CREDAL_GATE_CW}")
        cmd+=(--credal-gate-cw-high "${CREDAL_GATE_CW_HIGH}")
        # credal-gate 依赖 credal-probe（路由需要 cw 信号），自动开启
        if [ "${CREDAL_PROBE}" != "1" ]; then
            cmd+=(--credal-probe)
        fi
    fi
    # v8.8: Non-degradation guard
    if [ "${NON_DEGRADATION_GUARD}" = "1" ]; then
        cmd+=(--non-degradation-guard)
    fi

    # v9.0b: Question-Type Router
    if [ "${QUESTION_TYPE_ROUTER}" = "1" ]; then
        cmd+=(--question-type-router)
    fi
    # v9.0b: Online Normalizer
    if [ "${ONLINE_NORMALIZER}" = "1" ]; then
        cmd+=(--online-normalizer)
    fi
    if [ "${ORACLE_ONLINE_NORMALIZER}" = "1" ]; then
        cmd+=(--oracle-online-normalizer)
    fi
    # v9.1: HCEG-Fallback
    if [ "${HCEG_FALLBACK}" = "1" ]; then
        cmd+=(--hceg-fallback)
        cmd+=(--hceg-fallback-cw "${HCEG_FALLBACK_CW}")
        cmd+=(--hceg-fallback-compare-cw "${HCEG_FALLBACK_COMPARE_CW}")
        cmd+=(--hceg-fallback-diff-cw "${HCEG_FALLBACK_DIFF_CW}")
        cmd+=(--hceg-fallback-policy "${HCEG_FALLBACK_POLICY}")
        if [ "${HCEG_ROLE_AWARE}" = "1" ]; then
            cmd+=(--hceg-role-aware)
        fi
        cmd+=(--hceg-diagnostic-candidates "${HCEG_DIAGNOSTIC_CANDIDATES}")
        if [ "${CERT_COMMIT_BOUNDARY}" = "1" ]; then
            cmd+=(--certificate-commit-boundary)
            cmd+=(--certificate-commit-mode "${CERT_COMMIT_MODE}")
            cmd+=(--certificate-commit-max-llm-confidence "${CERT_COMMIT_MAX_LLM_CONFIDENCE}")
            cmd+=(--certificate-commit-min-credal-width "${CERT_COMMIT_MIN_CREDAL_WIDTH}")
            if [ "${CERT_COMMIT_ALLOW_DIAGNOSTIC}" = "1" ]; then
                cmd+=(--certificate-commit-allow-diagnostic-candidates)
            fi
            if [ "${CERT_OPERATION_VERIFIER}" = "1" ]; then
                cmd+=(--certificate-operation-verifier)
            fi
            if [ "${CERT_COMPARE_DIRECTION_VERIFIER}" = "1" ]; then
                cmd+=(--certificate-compare-direction-verifier)
            fi
            if [ "${CERT_NUMERIC_DIRECTION_VERIFIER}" = "1" ]; then
                cmd+=(--certificate-numeric-direction-verifier)
            fi
            if [ "${CERT_CONFORMAL_BOUNDARY}" = "1" ]; then
                cmd+=(--certificate-conformal-boundary)
                cmd+=(--certificate-conformal-threshold "${CERT_CONFORMAL_THRESHOLD}")
                cmd+=(--certificate-conformal-alpha "${CERT_CONFORMAL_ALPHA}")
            fi
        fi
        # HCEG-Fallback 依赖 credal-probe（需要 cw 信号），自动开启
        if [ "${CREDAL_PROBE}" != "1" ]; then
            cmd+=(--credal-probe)
        fi
    fi
    # v9.2: Diverse Self-Consistency
    if [ "${SELF_CONSISTENCY}" = "1" ]; then
        cmd+=(--self-consistency)
        cmd+=(--k-samples "${K_SAMPLES}")
        cmd+=(--self-consistency-temperature "${SELF_CONSISTENCY_TEMPERATURE}")
        cmd+=(--self-consistency-max-samples "${SELF_CONSISTENCY_MAX_SAMPLES}")
        cmd+=(--self-consistency-trigger "${SELF_CONSISTENCY_TRIGGER}")
    fi

    run_python_command "${cmd[@]}"

    log_info "Mode ${mode} completed. Results in: ${output_dir}"

    # 打印关键指标
    local metrics_file="${output_dir}/metrics.json"
    if [ -f "${metrics_file}" ]; then
        echo ""
        log_info "=== Key Metrics ==="
        "${PYTHON_BIN}" -c "
import json
with open('${metrics_file}') as f:
    m = json.load(f)
rates = m.get('em_rates', {})
cal = m.get('calibration', {})
src = m.get('answer_source_distribution', {})
mods = m.get('module_diagnostics', {})
hceg = m.get('hceg_candidate_quality', {})
cert_commit = m.get('certificate_commit_metrics', {})
sc = m.get('self_consistency_diagnostics', {})
ort = m.get('operation_role_target_metrics', {})
ocg = m.get('operation_commit_gate_metrics', {})
ext = m.get('external_generator_metrics', {})
eff = m.get('efficiency_metrics', {})
ctx = m.get('context_reachability_metrics', {})
evidence = m.get('evidence_filtering_metrics', {})
op_usage = m.get('operation_support_cell_usage_metrics', {})
dataset = m.get('dataset', '${CSCR_DATASET}')
primary_name = m.get('primary_metric_name', 'hitab_official_em')
primary_em = m.get('primary_em', rates.get('hitab_official_em', 0))
def rate_pct(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    return value if value > 1 else value * 100
print(f\"  dataset:          {dataset}\")
print(f\"  primary {primary_name}: {primary_em*100:.2f}%\")
print(f\"  strict_em:        {rates.get('strict_em', 0)*100:.2f}%\")
print(f\"  numeric_em:       {rates.get('numeric_em', 0)*100:.2f}%\")
print(f\"  set_em:           {rates.get('set_em', 0)*100:.2f}%\")
print(f\"  hitab_official_em:{rates.get('hitab_official_em', 0)*100:.2f}%\")
print(f\"  ECE:              {cal.get('ece', -1):.4f}\")
print(f\"  Brier:            {cal.get('brier', -1):.4f}\")
print(f\"  Answer sources:   {src}\")
if mods:
    compact = {k: v.get('count', 0) for k, v in mods.items() if v.get('count', 0)}
    print(f\"  Module triggers:  {compact}\")
if hceg and (hceg.get('trigger_count', 0) or hceg.get('candidate_count', 0)):
    print(f\"  HCEG quality:     trigger={hceg.get('trigger_count', 0)}, candidate={hceg.get('candidate_count', 0)}, diag={hceg.get('diagnostic_candidate_count', 0)}, cand_em={hceg.get('candidate_hitab_em', 0)*100:.2f}%, raw_em={hceg.get('raw_candidate_hitab_em', 0)*100:.2f}%, gain={hceg.get('potential_gain', 0)}, loss={hceg.get('potential_loss', 0)}, compat={hceg.get('candidate_compatible_rate', 0)*100:.2f}%, role_mismatch={hceg.get('role_mismatch_rate', 0)*100:.2f}%, changed={hceg.get('role_aware_changed', 0)}\")
if cert_commit and cert_commit.get('candidate_count', 0):
    print(f\"  Cert commit:      candidate={cert_commit.get('candidate_count', 0)}, op_verified={cert_commit.get('operation_verified_count', 0)}, dir_verified={cert_commit.get('compare_direction_verified_count', 0)}, numdir_verified={cert_commit.get('numeric_direction_verified_count', 0)}, conformal_score={cert_commit.get('conformal_score_pass_count', cert_commit.get('conformal_accepted_count', 0))}@{cert_commit.get('conformal_threshold', 'NA')}, conformal_shadow={cert_commit.get('conformal_shadow_accepted_count', cert_commit.get('shadow_recommended_count', 0))}, recommended={cert_commit.get('recommended_count', 0)}, shadow={cert_commit.get('shadow_recommended_count', 0)}, applied={cert_commit.get('applied_count', 0)}, rec_em={cert_commit.get('recommended_candidate_hitab_em', 0)*100:.2f}%, shadow_em={cert_commit.get('shadow_candidate_hitab_em', 0)*100:.2f}%, conf_shadow_em={cert_commit.get('conformal_shadow_candidate_hitab_em', cert_commit.get('shadow_candidate_hitab_em', 0))*100:.2f}%, rec_gain={cert_commit.get('recommended_potential_gain', 0)}, rec_loss={cert_commit.get('recommended_potential_loss', 0)}, shadow_gain={cert_commit.get('shadow_potential_gain', 0)}, shadow_loss={cert_commit.get('shadow_potential_loss', 0)}, conf_shadow_gain={cert_commit.get('conformal_shadow_potential_gain', cert_commit.get('shadow_potential_gain', 0))}, conf_shadow_loss={cert_commit.get('conformal_shadow_potential_loss', cert_commit.get('shadow_potential_loss', 0))}\")
if sc and sc.get('used', 0):
    empty = sc.get('empty_vote_group', 0)
    suffix = f\", empty_vote_group={empty}\" if empty else \"\"
    print(f\"  SC diagnostics:   used={sc.get('used', 0)}, changed={sc.get('changed', 0)}, changed_rate={sc.get('changed_rate', 0)*100:.2f}%{suffix}\")
if ort and ort.get('enabled_count', 0):
    overall = ort.get('overall', {})
    print(f\"  E64 role-target:  enabled={ort.get('enabled_count', 0)}, selected_em={overall.get('selected_executor_official_em', 0)*100:.2f}%, reranked_em={overall.get('reranked_executor_official_em', 0)*100:.2f}%, final_em={overall.get('final_official_em', 0)*100:.2f}%, rerank_changed={overall.get('rerank_changed_rate', 0)*100:.2f}%, avg_filter={overall.get('avg_filter_cell_count', 0):.2f}, avg_target={overall.get('avg_target_cell_count', 0):.2f}\")
if ocg and ocg.get('enabled_count', 0):
    overall = ocg.get('overall', {})
    print(f\"  Structural commit gate: enabled={ocg.get('enabled_count', 0)}, eligible={overall.get('eligible_count', 0)}, coverage={overall.get('coverage_rate', 0)*100:.2f}%, commit_em={overall.get('commit_executor_official_em', 0)*100:.2f}%, final_on_eligible={overall.get('final_official_em_on_eligible', 0)*100:.2f}%, gain/loss={overall.get('eligible_gain_count', 0)}/{overall.get('eligible_loss_count', 0)}, applied_gain/loss/w2w={overall.get('applied_gain_count', 0)}/{overall.get('applied_loss_count', 0)}/{overall.get('applied_wrong_to_wrong_count', 0)}, projected={overall.get('projected_primary_em_if_committed', 0)*100:.2f}% ({overall.get('projected_delta_em_if_committed', 0)*100:+.2f}pp), avg_target={overall.get('avg_target_cell_count_on_eligible', 0):.2f}, avg_filter={overall.get('avg_filter_cell_count_on_eligible', 0):.2f}, gaps={overall.get('gap_distribution', {})}\")
if ext and ext.get('enabled_count', 0):
    print(f\"  External API:     enabled={ext.get('enabled_count', 0)}, models={ext.get('api_model_distribution', {})}, logprobs={ext.get('logprobs_available_count', 0)}/{ext.get('enabled_count', 0)}, avg_prompt_tokens={ext.get('avg_prompt_tokens', 0):.1f}, avg_completion_tokens={ext.get('avg_completion_tokens', 0):.1f}, total_tokens={ext.get('total_tokens', 0):.0f}\")
if eff:
    print(f\"  Efficiency:       avg_tokens={eff.get('avg_generated_tokens', 0):.2f}, avg_sec={eff.get('avg_llm_generation_seconds', 0):.3f}, wrong_tokens={eff.get('avg_generated_tokens_wrong', 0):.2f}\")
if evidence or op_usage:
    print(f\"  Cell usage:       evidence={rate_pct(evidence.get('avg_cell_usage_rate', evidence.get('avg_cell_usage_percent', 0))):.2f}%, evidence_filter={rate_pct(evidence.get('avg_filtering_rate', 0)):.2f}%, op_support={rate_pct(op_usage.get('avg_operation_support_cell_usage_rate', 0)):.2f}%, op_target={rate_pct(op_usage.get('avg_operation_target_cell_usage_rate', 0)):.2f}%, applied_op_support={rate_pct(op_usage.get('avg_applied_operation_support_cell_usage_rate', 0)):.2f}%\")
if ctx:
    print(f\"  Context reach:    avg_tokens={ctx.get('avg_input_tokens', 0):.1f}, max_tokens={ctx.get('max_input_tokens', 0):.0f}, max_pressure={ctx.get('max_context_pressure', 0):.3f}, apr_skip={ctx.get('apr_round2_skipped_no_truncation', 0)}, sc_skip={ctx.get('self_consistency_skipped_no_truncation', 0)}\")
" 2>/dev/null || true
        echo ""
    fi
}

run_recalculate() {
    local output_dir="${OUTPUT_BASE}/recalculated_${TIMESTAMP}"

    log_info "=========================================="
    log_info "Running recalculate mode"
    log_info "Input: ${BASELINE_E_PREDS}"
    log_info "Output: ${output_dir}"
    log_info "=========================================="

    mkdir -p "${output_dir}"

    "${PYTHON_BIN}" "${PROJECT_DIR}/run_cscr_pipeline.py" \
        --mode recalculate \
        --dataset "${CSCR_DATASET}" \
        --recalculate_from "${BASELINE_E_PREDS}" \
        --output_dir "${output_dir}" \
        --input_file "${INPUT_FILE}" \
        --table_dir "${TABLE_DIR}" \
        ${EXTRA_ARGS}

    log_info "Recalculate completed. Results in: ${output_dir}"

    # 打印对比
    local metrics_file="${output_dir}/metrics_recalculated.json"
    if [ -f "${metrics_file}" ]; then
        echo ""
        log_info "=== Recalculated Metrics (四口径 EM) ==="
        "${PYTHON_BIN}" -c "
import json
with open('${metrics_file}') as f:
    m = json.load(f)
total = m.get('total', 0)
for cal in ['strict_em', 'numeric_em', 'set_em', 'hitab_official_em']:
    c = m.get(f'{cal}_count', 0)
    r = m.get(f'{cal}_rate', 0)
    print(f'  {cal}: {c}/{total} = {r*100:.2f}%')
" 2>/dev/null || true
        echo ""
    fi
}

run_dry_run() {
    local output_dir="${OUTPUT_BASE}/dry_run_${TIMESTAMP}"

    log_info "Dry run mode (no model loading)"

    "${PYTHON_BIN}" "${PROJECT_DIR}/run_cscr_pipeline.py" \
        --mode baseline_a_plus \
        --dataset "${CSCR_DATASET}" \
        --dry_run \
        --output_dir "${output_dir}" \
        --input_file "${INPUT_FILE}" \
        --table_dir "${TABLE_DIR}" \
        --batch_size 5 \
        --dataset-prompt-policy "${DATASET_PROMPT_POLICY}" \
        --source-risk-calibration "${SOURCE_RISK_CALIBRATION}" \
        --source-risk-llm-cert-adjusted-cap "${SOURCE_RISK_LLM_CERT_ADJUSTED_CAP}" \
        $( [ "${OPERATION_SUPPORT_DIAGNOSTICS}" = "1" ] && printf '%s' "--operation-support-diagnostics" ) \
        ${EXTRA_ARGS}

    log_info "Dry run complete. Check: ${output_dir}/dry_run_prompts.jsonl"
}

run_eval_utils_standalone() {
    # 使用 eval_utils.py 的 CLI 单独评估
    log_info "Running standalone eval_utils evaluation"

    "${PYTHON_BIN}" "${PROJECT_DIR}/eval_utils.py" \
        --predictions "${BASELINE_E_PREDS}" \
        --answer_key "ca2kg_answer" \
        --gold_key "gold" \
        ${EXTRA_ARGS}
}

print_help() {
    cat << 'EOF'
CSCR 实验启动脚本

用法:
  bash run_cscr.sh <MODE> [额外参数]

可用模式:
  recalculate      从 Baseline E 的 predictions.jsonl 重新计算四口径 EM
  baseline_a_plus  结构感知 Prompt + logit 熵校准 (Phase 1A)
  executor_only    结构感知 Prompt + 执行器 (Phase 1A + 4)
  full             完整 CSCR 管线 (Phase 1A + 1B + 3 + 4, v3 仲裁)
  full_cert        CSCR + Certificate + SCCI + Dominance (Phase 1A + 1B + 2 + 3 + 4 + 6)
  dry_run          仅生成 prompt 预览，不加载模型
  eval_standalone  使用 eval_utils.py 单独评估 Baseline E 预测
  all              依次运行 recalculate → baseline_a_plus → executor_only → full
  ablation_no_exec 消融: 全量 baseline_a_plus (无执行器，作为上界参照)
  help             显示此帮助信息

环境变量:
  CSCR_DATASET         数据集 hitab/aitqa/tablebench (默认: hitab)
  CSCR_INPUT_FILE      覆盖输入 jsonl
  CSCR_ASTRA_DATASET_ROOT ASTRA 数据集根目录 (默认: /home/hsh/ME/Table/EMNLP2026/CausalityAwareTableQA/dataset)
  CSCR_HITAB_CLEAN_TEST 保留兼容；HiTab 默认使用 ASTRA/dataset/hitab/test_samples_clean.jsonl
  CSCR_AITQA_CLEAN_TEST 保留兼容；AIT-QA 默认使用 ASTRA/dataset/AIT-QA/aitqa_clean_questions.json
  CSCR_TABLE_DIR       覆盖表格目录或 AIT-QA aitqa_tables.jsonl 所在目录
  CSCR_OUTPUT_BASE     覆盖输出根目录
  CSCR_MODEL_PATH      模型路径 (默认: /data1/hesihao1/llm/Qwen/Qwen2.5-7B-Instruct)
  CSCR_GENERATOR_BACKEND 生成器后端 vllm/openai_chat (默认: vllm)
  CSCR_API_BASE_URL    OpenAI-compatible API base URL (默认: https://api.lkeap.cloud.tencent.com/v1)
  CSCR_API_KEY_ENV     API key 所在环境变量名 (默认: LKEAP_API_KEY)
  CSCR_API_MODEL       API 模型名，例如 DeepSeek-V3-250324
  CSCR_API_TIMEOUT     API 请求超时秒数 (默认: 120)
  CSCR_API_MAX_RETRIES OpenAI SDK 重试次数 (默认: 3)
  CSCR_API_RATE_LIMIT_SECONDS 每次 API 请求间隔秒数 (默认: 0)
  CSCR_API_CACHE_PATH  可选 JSONL API response cache；用于闭源 API 复验降成本和增强可复现性
  CSCR_GPUS            逗号分隔的 GPU 编号列表 (默认: "1,2", 多卡示例: "0,1,2,3")
  CSCR_BANNED_GPUS     禁用 GPU 列表 (默认: 空；例如 "6,7")
  CSCR_ENABLE_GPU_BAN_GUARD 是否启用禁卡保护 1/0 (默认: 1；仅在 CSCR_BANNED_GPUS 非空时生效)
  CSCR_TP              tensor parallel size (默认: 自动=GPU数量)
  CSCR_GPU_MEM_UTIL    显存利用率 0.0~1.0 (默认: 0.85, 内存紧张可降到 0.55~0.65)
  CSCR_DTYPE           数据类型 bfloat16/float16/auto (默认: bfloat16)
  CSCR_MAX_NUM_SEQS    vLLM 并发序列数 (默认: 8, 内存紧张建议 4~8)
  CSCR_MAX_NUM_BATCHED_TOKENS vLLM 单次调度 token 上限 (默认: 24576, 内存紧张建议 16384~32768)
  CSCR_ENABLE_CHUNKED_PREFILL 启用 vLLM chunked prefill 1/0 (默认: 1)
  CSCR_KV_CACHE_DTYPE  vLLM KV cache dtype (默认: auto)
  CSCR_DISTRIBUTED_EXECUTOR_BACKEND vLLM 多卡执行后端 mp/ray (默认: mp)
  CSCR_USE_FAST_IMAGE_PROCESSOR Gemma3 多模态处理器使用 fast path 1/0 (默认: 1)
  CSCR_USE_FLASHINFER_SAMPLER 启用 vLLM FlashInfer sampler 1/0 (默认: 1)
  CSCR_SKIP_OVERLONG_PRIMARY 主轮超出模型上下文时写入 context_overflow 并继续 1/0 (默认: 0，投稿运行建议失败而不是跳过)
  CSCR_OUTPUT_DIR      固定输出目录；配合 --resume 可从已有 predictions.jsonl 继续
  VLLM_ENGINE_ITERATION_TIMEOUT_S vLLM engine step 超时秒数 (默认: 1800)
  VLLM_RPC_TIMEOUT    vLLM worker RPC 超时毫秒数 (默认: 1800000)
  RAY_CGRAPH_get_timeout Ray compiled graph 获取结果超时秒数 (默认: 1800)
  CSCR_SWAP_SPACE      vLLM 每张 GPU 的 CPU swap GiB (默认: 1，节点内存紧张不要调大)
  CSCR_CPU_OFFLOAD_GB  vLLM 每张 GPU 的 CPU offload GiB (默认: 0)
  CSCR_DISABLE_CUSTOM_ALL_REDUCE 禁用 vLLM custom all-reduce 1/0 (默认: 1)
  CSCR_ENFORCE_EAGER   禁用 CUDA graph capture，规避 TP 多卡 NCCL pending 等待 (默认: TP>1 时 1；Python 直跑默认也开启)
  CSCR_BATCH_INFERENCE 是否启用批量推理 1/0 (默认: 1=开启)
  CSCR_MAX_LEN         max model length (默认: 8192；正式 long-context 复验建议 32768)
  CSCR_PREFIX_STABLE_APR 1=APR/SC 使用同一完整输入前缀，只追加控制后缀
  CSCR_BATCH           batch size — 分段保存粒度 (默认: 32)
  CSCR_ADAPTIVE_PROMPT 是否启用 APR v8.3 纯熵路由 1/0 (默认: 0=关闭)
  CSCR_ENTROPY_LOW     APR 低熵阈值 (默认: 0.05, 低于此值保留 R1)
  CSCR_ENTROPY_HIGH    APR 高熵阈值 (v8.3 不使用，保留兼容)
  CSCR_QUESTION_TYPE_ROUTER v9.0b 问题类型 prompt 路由 1/0
  CSCR_ONLINE_NORMALIZER    v9.0b gold-free 格式归一化 1/0
  CSCR_ORACLE_ONLINE_NORMALIZER 仅诊断: gold oracle normalizer 1/0
  CSCR_HCEG_FALLBACK        v9.1 HCEG 候选生成 1/0
  CSCR_HCEG_FALLBACK_POLICY candidate_only/conservative/replace
  CSCR_HCEG_ROLE_AWARE      v9.5 HCEG 候选按答案角色回映射 1/0 (默认: 1)
  CSCR_HCEG_DIAGNOSTIC_CANDIDATES triggered/role_sensitive/all (默认: triggered)
  CSCR_SELF_CONSISTENCY     v9.2 多 prompt 候选投票 1/0
  CSCR_K_SAMPLES            v9.2 总候选预算，包含当前答案 (默认: 3)
  CSCR_SELF_CONSISTENCY_TRIGGER hceg/entropy/risk/all (默认: hceg)
  CSCR_DATASET_PROMPT_POLICY auto/legacy/benchmark/operation (默认: auto)
  CSCR_SOURCE_RISK_CALIBRATION auto/off/tablebench/all (默认: auto)
  CSCR_SOURCE_RISK_LLM_CERT_ADJUSTED_CAP llm_cert_adjusted 置信度封顶 (默认: 0.74)
  CSCR_SURFACE_HEURISTIC_MODE off/diagnostic/legacy (默认: diagnostic；主线只落盘 surface 诊断，不作为提交主证据)
  CSCR_MAIN_CERT_PROFILE  1/0 (默认: 0；1=关闭 APR/credal/router/normalizer/HCEG/旧 certificate commit，仅保留 E67 结构证书路径)
  CSCR_STRUCTURAL_PRIOR_WEIGHTING 1/0 (默认: 0；1=将 edge reliability 应用于 HCEG 边权，默认仅诊断落盘)
  CSCR_DISABLE_CANDIDATE_SCCI 1/0 (默认: 0；1=回退到旧的 sample-level SCCI 消融)
  CSCR_OPERATION_SUPPORT_DIAGNOSTICS E63 support-set 诊断 1/0 (默认: 0，不改答案)
  CSCR_OPERATION_ROLE_TARGET_DIAGNOSTICS E64 role-target 支持集诊断 1/0 (默认: 0，不改答案)
  CSCR_OPERATION_COMMIT_GATE_DIAGNOSTICS 结构证书保守提交 gate 诊断 1/0 (默认: 0，不改答案)
  CSCR_OPERATION_COMMIT_GATE_MODE diagnostic/conservative (默认: diagnostic；conservative 需配合 CSCR_BLACK_BOX_COMMIT_POLICY=certified)
  CSCR_OPERATION_COMMIT_VERSION E67/E65.4/E65.3 (默认: E67；E67 使用 measure fiber、aggregate echo 与候选稳定性证书)
  CSCR_OPERATION_CERTIFICATE_PROFILE strict/diagnostic (strict 会开启结构证书 conservative actual commit)
  CSCR_OPERATION_COMMIT_DATASET_SCOPE tablebench/hitab/tablebench_hitab/all (默认: all；主线先诊断 HiTab/AIT-QA)

多卡使用示例:
  # 4卡运行 Qwen2.5-32B（v8.3 当前主线: baseline_e + APR）
  CSCR_MODEL_PATH=/data1/hesihao1/llm/Qwen/Qwen2.5-32B-Instruct \
  CSCR_GPUS=0,1,2,3 \
  CSCR_GPU_MEM_UTIL=0.85 \
  CSCR_ADAPTIVE_PROMPT=1 \
  bash run_cscr.sh full_cert --prompt-style baseline_e \
    --success-predictor-model outputs/cscr/success_predictor_v2.pt

额外参数 (透传给 run_cscr_pipeline.py):
  --limit N        只处理前 N 个样本
  --start_from N   从第 N 个样本开始
  --resume         断点续跑
  --overwrite      覆盖已有输出
  --prompt-style baseline_e    v8.3: 当前最高主线 prompt
  --prompt-style table_focus   v8.5: 完整表格 + 结构焦点软提示（推荐验证）
  --prompt-style table_pruned  v8.4: 硬剪枝复现实验用；实验18/19/20已证实有害，不推荐主线
  --conformal-calibrate-from PATH  v6.0: 从 predictions.jsonl 校准 Conformal 阈值
  --conformal-alpha FLOAT          v6.0: Conformal α (default: 0.05)
  --adaptive-prompt                v8.3: 启用 APR 纯熵路由 (intersection_hint)
  CSCR_PREFIX_STABLE_APR=1         使用 prefix-stable APR，避免重建完整长 prompt
  --entropy-threshold-low FLOAT    v8.3: APR 低熵阈值 (default: 0.05)
  --entropy-threshold-high FLOAT   v8.3: (不使用，保留兼容)

v8.5 Safe Table Focus 实验 (完整表格 + 行列内聚团软提示):
 # 32B 模型 table_focus 消融
 CSCR_MODEL_PATH=/data1/hesihao1/llm/Qwen/Qwen2.5-32B-Instruct \
 CSCR_GPUS=0,1,2,3 CSCR_GPU_MEM_UTIL=0.85 \
 bash run_cscr.sh full_cert --prompt-style table_focus \
   --success-predictor-model outputs/cscr/success_predictor_v2.pt

 # 32B 模型 table_focus + APR 组合
 CSCR_MODEL_PATH=/data1/hesihao1/llm/Qwen/Qwen2.5-32B-Instruct \
 CSCR_GPUS=0,1,2,3 CSCR_GPU_MEM_UTIL=0.85 \
 CSCR_ADAPTIVE_PROMPT=1 \
 bash run_cscr.sh full_cert --prompt-style table_focus \
   --success-predictor-model outputs/cscr/success_predictor_v2.pt

v8.3 APR 路由实验 (纯熵路由 + intersection_hint):
 # 7B 模型 APR 实验
 CSCR_ADAPTIVE_PROMPT=1 \
 bash run_cscr.sh full_cert --prompt-style baseline_e \
   --success-predictor-model outputs/cscr/success_predictor_v2.pt

 # 32B 模型 APR 实验 (4卡)
 CSCR_MODEL_PATH=/data1/hesihao1/llm/Qwen/Qwen2.5-32B-Instruct \
 CSCR_GPUS=0,1,2,3 CSCR_GPU_MEM_UTIL=0.92 \
 CSCR_ADAPTIVE_PROMPT=1 \
 bash run_cscr.sh full_cert --prompt-style baseline_e \
   --success-predictor-model outputs/cscr/success_predictor_v2.pt

推荐实验顺序:
  1. bash run_cscr.sh recalculate        # 先验证四口径 EM，确认 66.35% 的真实组成
  2. bash run_cscr.sh dry_run            # 检查结构感知 prompt 质量
  3. bash run_cscr.sh baseline_a_plus    # Phase 1A 实验
  4. bash run_cscr.sh executor_only      # Phase 1A + 4 实验
  5. bash run_cscr.sh full               # 完整 CSCR 实验
  6. bash run_cscr.sh full_cert          # v6.0: Graph-Aware SCCI 全量实验

v6.0 Conformal 两阶段实验:
  Stage 1: bash run_cscr.sh full_cert   # 先跑完获取 predictions.jsonl
  Stage 2: bash run_cscr.sh full_cert --conformal-calibrate-from outputs/cscr/full_cert_XXXX/predictions.jsonl

快速验证 (前 100 个样本):
  bash run_cscr.sh baseline_a_plus --limit 100
EOF
}

# ---------------------------------------------------------------------------
# 主逻辑
# ---------------------------------------------------------------------------

case "${MODE}" in
    recalculate)
        run_recalculate
        ;;
    baseline_a_plus|executor_only|full|full_cert)
        run_mode "${MODE}"
        ;;
    dry_run)
        run_dry_run
        ;;
    eval_standalone)
        run_eval_utils_standalone
        ;;
    ablation_no_exec)
        log_info "Running ablation: full baseline_a_plus (no executor)"
        run_mode "baseline_a_plus"
        ;;
    all)
        log_info "Running all experiment modes sequentially"
        run_recalculate
        echo ""
        run_mode "baseline_a_plus"
        echo ""
        run_mode "executor_only"
        echo ""
        run_mode "full"
        echo ""
        run_mode "full_cert"
        log_info "All modes completed!"
        ;;
    help|--help|-h)
        print_help
        ;;
    *)
        log_error "Unknown mode: ${MODE}"
        echo ""
        print_help
        exit 1
        ;;
esac
