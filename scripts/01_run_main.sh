#!/usr/bin/env bash
set -euo pipefail

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    cat <<'EOF'
Usage: scripts/01_run_main.sh

Runs the source-pinned full_cert legacy mode. Set paths and credentials only in
ignored configs/paths.env or the documented CERTA_* environment variables.
EOF
    exit 0
fi

source "$(dirname "${BASH_SOURCE[0]}")/_release_common.sh"
load_local_paths
export CERTA_PROFILE="configs/profiles/main.env"
export CERTA_WRAPPER="${BASH_SOURCE[0]}"
# shellcheck source=/dev/null
source "${CERTA_ROOT}/${CERTA_PROFILE}"
require_runtime_configuration
freeze_public_main_profile
run_dir="${CERTA_OUTPUT_ROOT}/full_cert_${CERTA_RUN_ID}"
map_public_environment "${run_dir}"
run_legacy_mode "${CERTA_LEGACY_MODE}" "${run_dir}"
