#!/bin/bash
#SBATCH --job-name=hopd-tokdump
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --partition=gpu-a100
#SBATCH -A ECS26006

# Offline token-prob dump (not the H-OPD trainer).
# Submit from the repo so Slurm logs land in $HOPD_HOME/logs:
#   cd $SCRATCH/hopd/H-OPD && mkdir -p logs
#   git pull --ff-only origin main
#   sbatch experiments/token_prob_dump/run_ls6.sh

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

export HOPD_HOME="${HOPD_HOME:-$SCRATCH/hopd/H-OPD}"
export HOPD_VENV="${HOPD_VENV:-$SCRATCH/hopd/.venv}"
export HOPD_SCRATCH="${HOPD_SCRATCH:-$SCRATCH/hopd}"
export HOPD_ENV_SCRIPT="${HOPD_ENV_SCRIPT:-$HOPD_HOME/setup/env_vars.ls6.sh}"

if [[ ! -f "$HOPD_HOME/config.sh" ]]; then
    echo "ERROR: $HOPD_HOME is not the H-OPD repo. git clone/pull first." >&2
    exit 1
fi

# shellcheck disable=SC1091
source "$HOPD_HOME/config.sh"
cd "$HOPD_HOME"
mkdir -p "$HOPD_HOME/logs" "$HOPD_DATA_ROOT/token_prob_dump"

if [[ ! -x "$HOPD_VENV/bin/python" ]]; then
    echo "No venv at $HOPD_VENV — running setup/install_ls6.sh"
    bash "$HOPD_HOME/setup/install_ls6.sh"
fi
# shellcheck disable=SC1090
source "$HOPD_VENV/bin/activate"
hopd_activate_env || exit 1

n=$(find "$HOPD_DATA_ROOT" -name 'mmfine_reason_sampled_55k_text_prompt.parquet' 2>/dev/null | wc -l | tr -d ' ')
if [[ "$n" -eq 0 ]]; then
    echo "No train parquet — download_data.py --out_dir $HOPD_DATA_ROOT"
    python "$HOPD_HOME/download_data.py" --out_dir "$HOPD_DATA_ROOT"
fi

REQUIRE_CUDA=1 REQUIRE_PARQUET=1 bash "$HOPD_HOME/experiments/token_prob_dump/preflight.sh"

python -c "import matplotlib" 2>/dev/null || uv pip install matplotlib

OUT="$HOPD_DATA_ROOT/token_prob_dump/run_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
N_PROMPTS="${N_PROMPTS:-8}"
N_SAMPLES="${N_SAMPLES:-4}"
MAX_NEW="${MAX_NEW:-64}"
TOPK="${TOPK:-16}"

echo "OUT=$OUT N_PROMPTS=$N_PROMPTS N_SAMPLES=$N_SAMPLES MAX_NEW=$MAX_NEW"

python "$HOPD_HOME/experiments/token_prob_dump/dump.py" \
    --backend hf \
    --out_dir "$OUT" \
    --n_prompts "$N_PROMPTS" \
    --n_samples "$N_SAMPLES" \
    --max_new_tokens "$MAX_NEW" \
    --topk "$TOPK"

python "$HOPD_HOME/experiments/token_prob_dump/plot.py" --dump_dir "$OUT"
echo "Dump: $OUT"
ls -lh "$OUT"
