#!/bin/bash
# Site paths. Override any variable from the environment.

if [[ -z "${HOPD_HOME:-}" ]]; then
    HOPD_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
export HOPD_HOME

# Our veRL fork (not the MATH project's checkout).
# hopd  = this experiment (trainer patches). Branched from ls6.
# ls6   = frozen cu128 / vLLM 0.12 pin. Do not run fork main on LS6.
export HOPD_VERL_HOME="${HOPD_VERL_HOME:-$HOPD_HOME/verl}"
export HOPD_VERL_REPO="${HOPD_VERL_REPO:-https://github.com/rishuray123/verl.git}"
export HOPD_VERL_BRANCH="${HOPD_VERL_BRANCH:-hopd}"
export HOPD_VERL_COMMIT="${HOPD_VERL_COMMIT:-044bbba27e5861e1212727aba67813cccb2db291}"

export HOPD_VENV="${HOPD_VENV:-}"
export HOPD_ENV_SCRIPT="${HOPD_ENV_SCRIPT-$HOPD_HOME/setup/env_vars.ls6.sh}"

export HOPD_SCRATCH="${HOPD_SCRATCH:-${SCRATCH:-$HOPD_HOME/scratch}}"
export HOPD_DATA_ROOT="${HOPD_DATA_ROOT:-$HOPD_SCRATCH/data}"
export HOPD_CKPT_ROOT="${HOPD_CKPT_ROOT:-$HOPD_SCRATCH/checkpoints}"
export HOPD_LOG_DIR="${HOPD_LOG_DIR:-$HOPD_HOME/logs}"
export HOPD_WANDB_PROJECT="${HOPD_WANDB_PROJECT:-hopd}"
# tensorboard needs no login (unlike wandb); file writes machine-readable metrics.
export HOPD_LOGGER="${HOPD_LOGGER:-[\"console\",\"tensorboard\",\"file\"]}"

hopd_activate_env() {
    if [[ -n "${HOPD_ENV_SCRIPT:-}" && -f "$HOPD_ENV_SCRIPT" ]]; then
        # shellcheck disable=SC1090
        source "$HOPD_ENV_SCRIPT"
    fi
    if [[ -n "${HOPD_VENV:-}" ]]; then
        if [[ -f "$HOPD_VENV/bin/activate" ]]; then
            # shellcheck disable=SC1090
            source "$HOPD_VENV/bin/activate"
        else
            echo "ERROR: HOPD_VENV=$HOPD_VENV has no bin/activate" >&2
            return 1
        fi
    fi
    if [[ ! -d "$HOPD_VERL_HOME/verl/trainer/config" ]]; then
        echo "ERROR: no veRL at HOPD_VERL_HOME=$HOPD_VERL_HOME" >&2
        echo "       Run setup/bootstrap_verl.sh" >&2
        return 1
    fi
    export VERL_CONFIG_PATH="$HOPD_VERL_HOME/verl/trainer/config"
    export PYTHONPATH="$HOPD_VERL_HOME:${PYTHONPATH:-}"
    mkdir -p "$HOPD_LOG_DIR"
    return 0
}
