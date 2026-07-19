#!/usr/bin/env bash
set -euo pipefail

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    cat <<'EOF'
Usage: scripts/04_evaluate.sh RUN_DIR

Runs only the local ASTRA-compatible textual/symbolic evaluator with
--judge-backend none, then summarizes its saved metrics. No judge or model call
is permitted.
EOF
    exit 0
fi

run_dir="${1:?missing run directory}"
source "$(dirname "${BASH_SOURCE[0]}")/_release_common.sh"
load_local_paths
require_runtime_configuration
if [ ! -f "${run_dir}/predictions.jsonl" ]; then
    echo "Saved predictions are required: ${run_dir}/predictions.jsonl" >&2
    exit 2
fi
evaluation_dir="${run_dir}/evaluation"
"${CERTA_PYTHON}" "${CERTA_ROOT}/tools/cscr_astra_eval.py" \
    --dataset aitqa \
    --predictions "${run_dir}/predictions.jsonl" \
    --clean-test "${CERTA_INPUT_FILE}" \
    --output-dir "${evaluation_dir}" \
    --judge-backend none
"${CERTA_PYTHON}" "${CERTA_ROOT}/tools/summarize_astra_eval_metrics.py" \
    "${evaluation_dir}/evaluation_metrics.json" \
    --output-json "${evaluation_dir}/summary.json" > "${evaluation_dir}/summary.raw.md"
CERTA_EVALUATION_METRICS="${evaluation_dir}/evaluation_metrics.json" "${CERTA_PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

metrics = json.loads(Path(os.environ["CERTA_EVALUATION_METRICS"]).read_text(encoding="utf-8"))
print(json.dumps({key: metrics.get(key) for key in ("EM_textual_accuracy", "EM_symbolic_accuracy", "EM_max_accuracy")}, sort_keys=True))
PY
