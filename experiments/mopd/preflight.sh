#!/bin/bash
# Data + GPU-count check. Does not start veRL.
set -euo pipefail

HOPD_HOME="${HOPD_HOME:-$(cd "$(dirname "$0")/../.." && pwd)}"
export HOPD_HOME
# shellcheck disable=SC1091
source "$HOPD_HOME/config.sh"

python3 "$HOPD_HOME/experiments/mopd/smoke_data.py"
echo "[ ok ] git $(git -C "$HOPD_HOME" log -1 --oneline)"

ngpu=0
if command -v nvidia-smi >/dev/null 2>&1; then
    ngpu=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
fi
echo "[info] nvidia-smi GPU count: $ngpu (MOPD train needs 3)"

n=$(find "${HOPD_DATA_ROOT:-/nonexistent}" -name 'mmfine_reason_sampled_55k_text_prompt.parquet' 2>/dev/null | wc -l | tr -d ' ')
if [[ "$n" -gt 0 ]]; then
    echo "[ ok ] train parquet under $HOPD_DATA_ROOT"
else
    echo "[warn] no mmfine parquet yet — download_data.py before train"
fi

if [[ -d "$HOPD_VERL_HOME/verl/trainer/config" ]]; then
    echo "[ ok ] veRL at $HOPD_VERL_HOME"
else
    echo "[warn] no veRL — setup/bootstrap_verl.sh before train"
fi

if [[ "${REQUIRE_3GPU:-0}" == "1" && "$ngpu" -lt 3 ]]; then
    echo "ERROR: need 3 GPUs for MOPD (student + VL teacher + text teacher)" >&2
    exit 1
fi
echo "preflight OK"
