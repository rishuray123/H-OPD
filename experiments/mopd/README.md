# Routed MOPD and paper H-OPD (veRL)

Default: veRL assigns **one** teacher per sample via `data_source` (routed MOPD).
`HOPD_PAPER=1` is the paper recipe: both teachers on the same \(Y\), mix on
\(\Omega_t = K_V \cup K_T\), reverse KL \(D_{\mathrm{KL}}(\pi_\theta \| q_A)\).

GPU split: student 2B on **1** GPU, VL 4B teacher on **1**, text `Qwen3-4B-Instruct-2507` on **1**. A 1×H100 dump machine cannot run this.

Even rows → VL teacher (images kept). Odd rows → text teacher (images cleared).

Smoke uses the **real** `mmfine_reason_sampled_55k_text_prompt.parquet`, first **96** rows (half VL, half text), 2 trainer steps. Full uses all rows for 1 epoch.

`max_prompt_length` is 2048 because image tokens inflate VL prompts far past their
text length, and `filter_overlong_prompts=True` must stay on — it measures the
processed length and drops rows that would otherwise break `DataProto.concat`.
That filtering is why the smoke slices more rows than 2 steps consume.

A full-resolution image alone blows that budget, so `make_routed_parquet.py`
writes `max_pixels`/`min_pixels` into every image struct (`--image_max_pixels`,
default 256·32·32 ≈ 256 image tokens); `fetch_image` honours them in both the
filter and the rollout. Check the outcome on CPU before booking GPUs:

```bash
python3 experiments/mopd/probe_lengths.py --parquet scratch/data/mopd/train_routed_96.parquet
```

It prints prompt-length percentiles per `data_source` and shows the real
exception for rows veRL would silently skip. If `hopd_vl` shows `over budget:
0/8`, the VL rows will survive; the run's own `filter dataset len:` should then
be close to `dataset len:`.

`make_routed_parquet.py` must write a **new** Arrow schema (`bytes` + pixel
caps only). HuggingFace `datasets` reloads every struct field, so a pandas
round-trip that still carries a null `image` fails `process_image` even when
`pd.read_parquet` hides that key. After pulling this repo, also refresh the
nested trainer so `process_image` drops null fields (needed for val parquets):

```bash
bash setup/bootstrap_verl.sh
rm -f scratch/data/mopd/train_routed_96.parquet scratch/data/mopd/train_routed.parquet
```

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
bash experiments/mopd/run_lightning.sh mix     # both teachers, p(y_t) mix + k1
bash experiments/mopd/run_lightning.sh paper   # paper H-OPD: union Ω_t + reverse KL
bash experiments/mopd/run_lightning.sh full
```

## Watching a run

Every run writes to `logs/mopd_<id>/`, with `logs/mopd_latest` symlinked to the
newest one. Loggers default to `console`, `tensorboard`, and `file`, and each
training step dumps its rollouts, so all of this works mid-run:

```bash
python3 experiments/mopd/watch.py --follow            # metrics table, refreshes
python3 experiments/mopd/show_rollouts.py --n 3       # what the student just wrote
python3 experiments/mopd/show_rollouts.py --step all --stats
tensorboard --logdir logs/mopd_latest/tensorboard --port 6006
python3 experiments/mopd/watch.py --plot plots/ --csv metrics.csv
```

What to read as progress:

- `distill_loss` down and `teacher_mass` up — the student's distribution is
  moving onto the teacher's. This is the actual training objective.
- `acc` — accuracy from `task_reward.py`, computed on every rollout. It is
  **reported only**: `score` stays 0.0 so the distillation loss never sees it.
  This is the honest "is the student getting better" number.
- `entropy` falling with `resp_len` stable is normal sharpening; `resp_len`
  collapsing to a few tokens or `grad_norm` spiking means something is wrong.

Rollout JSONL lives in `logs/mopd_latest/rollouts/<step>.jsonl` (one row per
sample: prompt, response, `acc`, `is_vl`). Validation dumps land in `val/` when
`TEST_FREQ` is set — it is off by default since training rollouts already show
generations.

## Paper H-OPD (`HOPD_PAPER=1`)

This is the paper recipe on this stack. Both teachers score the same student
\(Y\). Each is renormalized on its own top-\(k\), then mixed on
\(\Omega_t = K_V \cup K_T\) with entropy weights \(\alpha \propto \exp(-H/\tau)\).
The student is trained with reverse KL \(D_{\mathrm{KL}}(\pi_\theta \| q_A)\)
on that support (`loss_mode=reverse_kl_topk`, `use_policy_gradient=False`).
The text teacher sees the question with VL image pads stripped and no pixels
(proxy for the paper's \(\tilde{x}\); we do not rewrite `[IMAGE DESCRIPTION]`
unless the parquet already has \(d\)).

```bash
bash experiments/mopd/run_lightning.sh paper        # 96 rows, 2 steps
# or:
HOPD_PAPER=1 MOPD_MAX_ROWS=96 STEPS=2 bash experiments/mopd/run_ls6.sh
```

`HOPD_MIX=1` is the older approximation: same dual scoring, but only \(p(y_t)\)
is mixed and the update is k1 + PPO. Do not stack paper reverse KL with
`HOPD_TASK_REWARD` / outcome GRPO in one job.

Still not paper *scale*: we use 20k/2048/512/bs8 on 3×L40S, not 55k/12384/128
on 8×B200. `acc` is always logged; it is not in the paper loss.

Rollouts gain `alpha_vl` (1 = VL teacher won that sample) and `omega_k` (union
support size) on the paper path.

## Lonestar6

```bash
cd $SCRATCH/hopd/H-OPD
mkdir -p logs
MOPD_MAX_ROWS=32 STEPS=2 sbatch experiments/mopd/run_ls6.sh
# after that job succeeds:
MOPD_FULL=1 sbatch experiments/mopd/run_ls6.sh
```
