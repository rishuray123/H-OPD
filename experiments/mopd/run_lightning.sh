#!/bin/bash
# Lightning 3-GPU MOPD. Usage:
#   bash experiments/mopd/run_lightning.sh smoke   # routed MOPD, 96 rows, 2 steps
#   bash experiments/mopd/run_lightning.sh full    # all 55k, 1 epoch
#   bash experiments/mopd/run_lightning.sh mix     # both teachers, p(y_t) mix + k1
#   bash experiments/mopd/run_lightning.sh paper   # paper H-OPD: union Ω_t + reverse KL
set -euo pipefail

MODE="${1:-smoke}"
HOPD_HOME="${HOPD_HOME:-$(cd "$(dirname "$0")/../.." && pwd)}"
export HOPD_HOME
export SKIP_INSTALL="${SKIP_INSTALL:-1}"

ngpu=0
if command -v nvidia-smi >/dev/null 2>&1; then
    ngpu=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
fi
if [[ "$ngpu" -lt 3 ]]; then
    echo "ERROR: $ngpu GPU(s); routed MOPD needs 3." >&2
    exit 1
fi

if [[ -z "${SCRATCH:-}" ]] || [[ ! -d "${SCRATCH:-/nonexistent}" ]]; then
    export HOPD_ENV_SCRIPT=""
    export HOPD_SCRATCH="${HOPD_SCRATCH:-$HOPD_HOME/scratch}"
    export HOPD_DATA_ROOT="${HOPD_DATA_ROOT:-$HOPD_SCRATCH/data}"
    export HOPD_CKPT_ROOT="${HOPD_CKPT_ROOT:-$HOPD_SCRATCH/checkpoints}"
    if [[ -n "${VIRTUAL_ENV:-}" ]]; then
        export HOPD_VENV="$VIRTUAL_ENV"
    fi
fi

case "$MODE" in
    smoke)
        # Overlong VL prompts are dropped by the dataset filter, so slice more
        # rows than the 2 steps x batch 8 the smoke actually consumes.
        export MOPD_MAX_ROWS="${MOPD_MAX_ROWS:-96}"
        export STEPS="${STEPS:-2}"
        export MOPD_FULL=0
        echo "SMOKE: real H-OPD parquet, first ${MOPD_MAX_ROWS} rows, STEPS=${STEPS}"
        ;;
    full)
        export MOPD_MAX_ROWS=0
        export MOPD_FULL=1
        unset STEPS || true
        echo "FULL: all rows of real H-OPD parquet, 1 epoch, ckpt every 50 steps"
        ;;
    mix)
        export MOPD_MAX_ROWS="${MOPD_MAX_ROWS:-96}"
        export STEPS="${STEPS:-2}"
        export MOPD_FULL=0
        export HOPD_MIX=1
        echo "MIX smoke: both teachers on each Y, p(y_t) entropy blend + k1, ${MOPD_MAX_ROWS} rows, STEPS=${STEPS}"
        ;;
    paper)
        export MOPD_MAX_ROWS="${MOPD_MAX_ROWS:-96}"
        export STEPS="${STEPS:-2}"
        export MOPD_FULL=0
        export HOPD_PAPER=1
        echo "PAPER smoke: union Ω_t mix, reverse KL, no PPO, ${MOPD_MAX_ROWS} rows, STEPS=${STEPS}"
        ;;
    *)
        echo "Usage: $0 smoke|full|mix|paper" >&2
        exit 1
        ;;
esac

exec bash "$HOPD_HOME/experiments/mopd/run_ls6.sh"
