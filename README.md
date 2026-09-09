# H-OPD (this fork)

Harness for reproducing [H-OPD](https://arxiv.org/abs/2607.02592) and later variants.

veRL lives in **`verl/` inside this folder** (gitignored). It is still a separate GitHub repo so trainer patches stay reusable: [rishuray123/verl](https://github.com/rishuray123/verl) (**`hopd`** for this experiment, `ls6` is the frozen LS6 stack, `main` tracks upstream — do not install `main` on Lonestar).

Do not reuse `$SCRATCH/opdl/`. This project uses `$SCRATCH/hopd/`.

Paper authors did not release `run_hopd.sh`. Stage 1 is single-teacher VL OPD on their parquet. Entropy arbitration comes later as a `loss_mode` on our veRL fork.

## Lonestar6 (one GPU node)

```bash
ssh ls6
idev -p gpu-a100-dev -A ECS26006 -N 1 -n 1 -t 04:00:00

export HOPD_HOME=$SCRATCH/hopd/H-OPD
export HOPD_VENV=$SCRATCH/hopd/.venv
export HOPD_ENV_SCRIPT=$HOPD_HOME/setup/env_vars.ls6.sh
export HOPD_SCRATCH=$SCRATCH/hopd
export PATH="$HOME/.local/bin:$PATH"

mkdir -p $SCRATCH/hopd && cd $SCRATCH/hopd
if [[ ! -d $HOPD_HOME/.git ]]; then
    git clone https://github.com/rishuray123/H-OPD.git
else
    git -C $HOPD_HOME pull --ff-only origin main
fi

bash $HOPD_HOME/setup/install_ls6.sh   # clones or git fetch+checkout origin/hopd
source $HOPD_VENV/bin/activate
python $HOPD_HOME/download_data.py --out_dir $HOPD_SCRATCH/data
bash $HOPD_HOME/verify_setup.sh
bash $HOPD_HOME/hopd_vl_opd_ls6.sh --steps 5
```

Already cloned: `git -C $HOPD_HOME pull --ff-only origin main` then `bash $HOPD_HOME/setup/bootstrap_verl.sh` (fast trainer-only update) or `install_ls6.sh` (also refreshes the venv).

Student 2 GPUs FSDP, teacher 1 GPU vLLM. Defaults: Qwen3-VL-2B ← Qwen3-VL-4B, `k=8`, reverse KL.

## Citation

Yin et al., *H-OPD: Confidence Aware Heterogeneous Multi-Teacher Multimodal On-policy Distillation*, arXiv:2607.02592.
Data: [qixiangbupt/H-OPD-data](https://huggingface.co/datasets/qixiangbupt/H-OPD-data).
