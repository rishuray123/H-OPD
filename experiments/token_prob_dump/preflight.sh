#!/bin/bash
# Login-node OK except the CUDA check (skipped unless REQUIRE_CUDA=1).
set -euo pipefail

HOPD_HOME="${HOPD_HOME:-${SCRATCH:+$SCRATCH/hopd/H-OPD}}"
if [[ -z "${HOPD_HOME:-}" || ! -f "$HOPD_HOME/config.sh" ]]; then
    echo "ERROR: HOPD_HOME=$HOPD_HOME is not this repo. git pull first?" >&2
    exit 1
fi
export HOPD_HOME
export HOPD_VENV="${HOPD_VENV:-${SCRATCH:+$SCRATCH/hopd/.venv}}"
export HOPD_SCRATCH="${HOPD_SCRATCH:-${SCRATCH:+$SCRATCH/hopd}}"
export HOPD_ENV_SCRIPT="${HOPD_ENV_SCRIPT:-$HOPD_HOME/setup/env_vars.ls6.sh}"
export PATH="$HOME/.local/bin:$PATH"
# shellcheck disable=SC1091
source "$HOPD_HOME/config.sh"

DUMP="$HOPD_HOME/experiments/token_prob_dump/dump.py"
PLOT="$HOPD_HOME/experiments/token_prob_dump/plot.py"
[[ -f "$DUMP" ]] || { echo "ERROR: missing $DUMP — git pull origin main" >&2; exit 1; }
[[ -f "$PLOT" ]] || { echo "ERROR: missing $PLOT" >&2; exit 1; }

python3 -m py_compile "$DUMP" "$PLOT"
echo "[ ok ] py_compile dump.py plot.py"
echo "[ ok ] git $(git -C "$HOPD_HOME" log -1 --oneline)"

n=$(find "${HOPD_DATA_ROOT:-/nonexistent}" -name 'mmfine_reason_sampled_55k_text_prompt.parquet' 2>/dev/null | wc -l | tr -d ' ')
if [[ "$n" -gt 0 ]]; then
    echo "[ ok ] train parquet under $HOPD_DATA_ROOT"
else
    echo "[warn] no mmfine parquet — download_data.py on a GPU node first"
    if [[ "${REQUIRE_PARQUET:-0}" == "1" ]]; then
        exit 1
    fi
fi

if [[ "${REQUIRE_CUDA:-0}" == "1" ]]; then
    hopd_activate_env || exit 1
    python - <<'PY'
import torch, sys
print("[ ok ] python", sys.executable)
print("[ ok ] torch", torch.__version__, "cuda", torch.cuda.is_available(), "ngpu", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("ERROR: no CUDA in this job")
PY
fi
echo "preflight OK"
