#!/usr/bin/env bash
set -euo pipefail

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    cat <<'EOF'
Usage: scripts/03_run_efficiency.sh --base-run PATH

Runs the source-pinned main mode under engineering-only settings. The paired
main run config must be saved at PATH/run_config.json.
EOF
    exit 0
fi

if [ "${1:-}" != "--base-run" ] || [ -z "${2:-}" ]; then
    echo "Usage: scripts/03_run_efficiency.sh --base-run PATH" >&2
    exit 2
fi
base_run="$2"
source "$(dirname "${BASH_SOURCE[0]}")/_release_common.sh"
load_local_paths
export CERTA_PROFILE="${CERTA_PROFILE:-configs/profiles/efficiency.env}"
export CERTA_WRAPPER="${BASH_SOURCE[0]}"
# shellcheck source=/dev/null
source "${CERTA_ROOT}/${CERTA_PROFILE}"
require_runtime_configuration
if [ ! -f "${base_run}/run_config.json" ] || [ ! -f "${base_run}/release_metadata.json" ]; then
    echo "Paired main run is missing run_config.json or release_metadata.json: ${base_run}" >&2
    exit 2
fi
run_dir="${CERTA_OUTPUT_ROOT}/efficiency_${CERTA_RUN_ID}"
map_public_environment "${run_dir}"
CERTA_EFFICIENCY_BASE_RUN="${base_run}" CERTA_EFFICIENCY_OUTPUT="${run_dir}" \
"${CERTA_PYTHON:-python3}" - <<'PY'
import json
import os
from pathlib import Path

base_dir = Path(os.environ["CERTA_EFFICIENCY_BASE_RUN"])
base = json.loads((base_dir / "run_config.json").read_text(encoding="utf-8"))
base_release = json.loads((base_dir / "release_metadata.json").read_text(encoding="utf-8"))
current_release = json.loads((Path(os.environ["CERTA_EFFICIENCY_OUTPUT"]) / "release_metadata.json").read_text(encoding="utf-8"))
paired_keys = ("source_commit", "profile", "dataset", "input_file", "table_dir", "model_path", "api_base_url", "api_model")
mismatches = {key: {"base": base_release.get(key), "current": current_release.get(key)} for key in paired_keys if base_release.get(key) != current_release.get(key)}
if mismatches:
    raise SystemExit("Efficiency run differs from paired main run: " + json.dumps(mismatches, sort_keys=True))
out = Path(os.environ["CERTA_EFFICIENCY_OUTPUT"]) / "efficiency_pairing.json"
out.write_text(json.dumps({"base_run_config": base, "paired_fields": list(paired_keys), "label": "engineering efficiency only"}, indent=2) + "\n", encoding="utf-8")
PY
run_legacy_mode "${CERTA_LEGACY_MODE}" "${run_dir}"
