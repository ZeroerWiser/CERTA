#!/usr/bin/env bash
set -euo pipefail

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    cat <<'EOF'
Usage: scripts/02_run_ablation.sh ARM_NAME

ARM_NAME must be present in configs/experiment_matrix.tsv and map to an
existing source mode. This wrapper does not define new ablation semantics.
EOF
    exit 0
fi

arm_name="${1:?missing registered arm name}"
source "$(dirname "${BASH_SOURCE[0]}")/_release_common.sh"
load_local_paths
export CERTA_PROFILE="${CERTA_PROFILE:-configs/profiles/ablation.env}"
export CERTA_WRAPPER="${BASH_SOURCE[0]}"
# shellcheck source=/dev/null
source "${CERTA_ROOT}/${CERTA_PROFILE}"
require_runtime_configuration
matrix_row="$(awk -F '\t' -v arm="${arm_name}" 'NR > 1 && $1 == arm {print; exit}' "${CERTA_ROOT}/configs/experiment_matrix.tsv")"
if [ -z "${matrix_row}" ]; then
    echo "Unregistered ablation arm: ${arm_name}" >&2
    exit 2
fi
IFS=$'\t' read -r _arm source_mode changed_existing_flag unchanged_flags legacy_only <<< "${matrix_row}"
run_dir="${CERTA_OUTPUT_ROOT}/${arm_name}_${CERTA_RUN_ID}"
map_public_environment "${run_dir}"
CERTA_ABLATION_METADATA_DIR="${run_dir}" CERTA_ABLATION_ARM="${arm_name}" \
CERTA_ABLATION_MODE="${source_mode}" CERTA_CHANGED_FLAG="${changed_existing_flag}" \
CERTA_UNCHANGED_FLAGS="${unchanged_flags}" CERTA_LEGACY_ONLY="${legacy_only}" \
"${CERTA_PYTHON:-python3}" - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["CERTA_ABLATION_METADATA_DIR"]) / "ablation_config.json"
path.write_text(json.dumps({
    "arm_name": os.environ["CERTA_ABLATION_ARM"],
    "base_run_id": os.environ.get("CERTA_RUN_ID", ""),
    "changed_existing_flag": os.environ["CERTA_CHANGED_FLAG"],
    "unchanged_flags": os.environ["CERTA_UNCHANGED_FLAGS"],
    "source_mode": os.environ["CERTA_ABLATION_MODE"],
    "legacy_compatibility_only": os.environ["CERTA_LEGACY_ONLY"] == "true",
    "dataset": os.environ.get("CERTA_DATASET", ""),
    "model": os.environ.get("CERTA_MODEL_PATH", ""),
    "source_commit": os.environ.get("CERTA_SOURCE_COMMIT", ""),
    "profile": os.environ.get("CERTA_PROFILE", ""),
}, indent=2) + "\n", encoding="utf-8")
PY
run_legacy_mode "${source_mode}" "${run_dir}"
