# Token-prob dump (separate from H-OPD training)

Offline inspection. Not `hopd_vl_opd_ls6.sh`. Scripts must be **on GitHub `main`** and `git pull`'d on Lonestar or Slurm will not see them.

Outputs: `$HOPD_DATA_ROOT/token_prob_dump/<run>/` on scratch.

## Lonestar6

Login node (no GPU):

```bash
export HOPD_HOME=$SCRATCH/hopd/H-OPD
export HOPD_VENV=$SCRATCH/hopd/.venv
export HOPD_ENV_SCRIPT=$HOPD_HOME/setup/env_vars.ls6.sh
export HOPD_SCRATCH=$SCRATCH/hopd
export PATH="$HOME/.local/bin:$PATH"

git -C $HOPD_HOME pull --ff-only origin main
bash $HOPD_HOME/experiments/token_prob_dump/preflight.sh
```

If preflight warns **no mmfine parquet**, do not expect the dump to work until a GPU job downloads data. `run_ls6.sh` now runs `install_ls6.sh` and `download_data.py` when those are missing (first job is long).

```bash
cd $HOPD_HOME
mkdir -p logs
git pull --ff-only origin main
sbatch experiments/token_prob_dump/run_ls6.sh
squeue -u $USER
```

First GPU job is small (8 prompts × 4 samples). After it works:

```bash
sbatch --export=ALL,N_PROMPTS=64,N_SAMPLES=8,MAX_NEW=128 experiments/token_prob_dump/run_ls6.sh
```
