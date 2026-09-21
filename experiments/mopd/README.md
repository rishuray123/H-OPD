# Routed MOPD (veRL), not paper H-OPD

veRL assigns **one** teacher per sample via `data_source`. That is **not** token-level mix of VL + text on the same \(Y\).

GPU split: student 2B on **1** GPU, VL 4B teacher on **1**, text `Qwen3-4B-Instruct-2507` on **1**. A 1×H100 dump machine cannot run this.

Even rows → VL teacher (images kept). Odd rows → text teacher (images cleared).

Smoke uses the **real** `mmfine_reason_sampled_55k_text_prompt.parquet`, first **96** rows (half VL, half text), 2 trainer steps. Full uses all rows for 1 epoch.

`max_prompt_length` is 2048 because image tokens inflate VL prompts far past their
text length, and `filter_overlong_prompts=True` must stay on — it measures the
processed length and drops rows that would otherwise break `DataProto.concat`.
That filtering is why the smoke slices more rows than 2 steps consume.

## Before any run

```bash
python3 experiments/mopd/doctor.py
```

Checks torch/CUDA, that `transformers.is_torch_available()` is True, and that
`verl.trainer.main_ppo` imports. A half-finished install shows up as
`cannot import name 'PreTrainedModel' from 'transformers'` — every torch-only
name is hidden when that flag is False, so do not patch the import sites.

Use the venv python explicitly (`scratch/.venv/bin/python`); on Lightning the
bare `python` can be the base conda interpreter even when the prompt says
`(.venv)`.

## Lightning (3 GPUs)

After clone / venv / `bootstrap_verl.sh` / `download_data.py`:

```bash
cd ~/H-OPD
source "$HOPD_VENV/bin/activate"
bash experiments/mopd/run_lightning.sh smoke   # wait for Done.
bash experiments/mopd/run_lightning.sh full
```

## Lonestar6

```bash
cd $SCRATCH/hopd/H-OPD
mkdir -p logs
MOPD_MAX_ROWS=32 STEPS=2 sbatch experiments/mopd/run_ls6.sh
# after that job succeeds:
MOPD_FULL=1 sbatch experiments/mopd/run_ls6.sh
```
