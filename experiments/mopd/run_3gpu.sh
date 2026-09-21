#!/bin/bash
# MOPD train without Slurm (Lightning 3-GPU or LS6 after idev).
# 1 GPU student + 1 GPU VL teacher + 1 GPU text teacher. STEPS default 2.
set -euo pipefail

HOPD_HOME="${HOPD_HOME:-$(cd "$(dirname "$0")/../.." && pwd)}"
export HOPD_HOME
export STEPS="${STEPS:-2}"
export SKIP_INSTALL="${SKIP_INSTALL:-1}"

ngpu=0
if command -v nvidia-smi >/dev/null 2>&1; then
    ngpu=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
fi
if [[ "$ngpu" -lt 3 ]]; then
    echo "ERROR: this machine has $ngpu GPU(s). Routed MOPD needs 3." >&2
    echo "  Lightning: start a 3× GPU instance, then:" >&2
    echo "    bash experiments/mopd/preflight.sh" >&2
    echo "    STEPS=2 bash experiments/mopd/run_3gpu.sh" >&2
    echo "  Lonestar:  mkdir -p logs && sbatch experiments/mopd/run_ls6.sh" >&2
    echo "  1×H100 dump box cannot run this trainer." >&2
    exit 1
fi

# Lightning / local: do not load TACC modules.
if [[ -z "${SCRATCH:-}" ]] || [[ ! -d "${SCRATCH:-/nonexistent}" ]]; then
    export HOPD_ENV_SCRIPT=""
    export HOPD_SCRATCH="${HOPD_SCRATCH:-$HOPD_HOME/scratch}"
    export HOPD_DATA_ROOT="${HOPD_DATA_ROOT:-$HOPD_SCRATCH/data}"
    export HOPD_CKPT_ROOT="${HOPD_CKPT_ROOT:-$HOPD_SCRATCH/checkpoints}"
    if [[ -n "${VIRTUAL_ENV:-}" ]]; then
        export HOPD_VENV="$VIRTUAL_ENV"
    fi
fi

echo "MOPD 3-GPU smoke STEPS=$STEPS HOPD_HOME=$HOPD_HOME"
exec bash "$HOPD_HOME/experiments/mopd/run_ls6.sh"
