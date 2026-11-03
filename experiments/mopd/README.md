# Routed MOPD and paper H-OPD (veRL)

Default: veRL assigns **one** teacher per sample via `data_source` (routed MOPD).
`HOPD_PAPER=1` is the paper recipe: both teachers on the same \(Y\), mix on
\(\Omega_t = K_V \cup K_T\), reverse KL \(D_{\mathrm{KL}}(\pi_\theta \| q^\star)\),
text teacher on the parquet `text_prompt` (\(\tilde{x}\) with `[IMAGE DESCRIPTION]`),
held-out **MathVista-200** `acc`.

GPU split: student 2B on **1** GPU, VL 4B teacher on **1**, text `Qwen3-4B-Instruct-2507` on **1**. A 1×H100 dump machine cannot run this.
Debug-only 2-GPU layout: `HOPD_TEACHERS_SHARE_GPU=1` (auto when only 2 GPUs are
visible) puts both teachers on GPU 1 as Ray fractional bundles (0.40 util each);
same code paths, slower teacher scoring.

Even rows → VL teacher (images kept). Odd rows → text teacher (images cleared).

Smoke uses the **real** `mmfine_reason_sampled_55k_text_prompt.parquet`, first **96** rows (half VL, half text), 2 trainer steps. Full uses all rows for 1 epoch.

`max_prompt_length` is 2048 because image tokens inflate VL prompts far past their
text length, and `filter_overlong_prompts=True` stays on — it measures the
processed length and drops rows that would otherwise break `DataProto.concat`.
The first pass writes an index cache under `~/.cache/verl/rlhf/overlong_filter/`;
later launches skip the tokenize. `HOPD_REFILTER=1` rebuilds. That filtering is
why the smoke slices more rows than 2 steps consume. `make_routed_parquet.py`
also skips when its `.stamp.json` still matches (`MOPD_REBUILD=1` to force).

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

One-shot setup (venv under `scratch/.venv`, veRL fork on `hopd`, paper overlay,
dataset incl. the three eval parquets, `doctor.py`):

```bash
git clone <this repo> ~/H-OPD && cd ~/H-OPD
bash setup/lightning_setup.sh          # ~20 min: wheels + 9 GB dataset
source scratch/env.sh                  # every new terminal
bash experiments/mopd/run_lightning.sh papercheck
python3 experiments/mopd/watch.py --follow   # second terminal
```

`papercheck` is the paper recipe end to end but short: 512 rows, 12 steps,
val on MathVista/MathVerse/MathVision (64 rows each) at step 0, 6, 12, one
checkpoint at step 12. It is the exact env block the TACC job gets (see
`run_lightning.sh`). Other modes:

```bash
bash experiments/mopd/run_lightning.sh smoke   # routed MOPD, 96 rows, 2 steps
bash experiments/mopd/run_lightning.sh mix     # both teachers, p(y_t) mix + k1
bash experiments/mopd/run_lightning.sh papersmoke # paper recipe, 96 rows, 2 steps, no val
bash experiments/mopd/run_lightning.sh paperdev   # -dev: 96 rows, 4 steps, MathVista-32
bash experiments/mopd/run_lightning.sh paper      # paper confirm: 256 rows, 20 steps, MathVista acc
bash experiments/mopd/run_lightning.sh paper2k    # 2048 rows, 80 steps
bash experiments/mopd/run_lightning.sh paper20k   # 20k rows, 200 steps
bash experiments/mopd/run_lightning.sh full
```

Knobs (env, any mode): `VAL_SETS=mathvista,mathverse,mathvision`,
`VAL_MAX_ROWS=0` (full sets), `HOPD_MIX_TAU` (paper 0.5), `STEPS`,
`TEST_FREQ`, `SAVE_FREQ`, `MOPD_MAX_ROWS`.

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
The student is trained with reverse KL \(D_{\mathrm{KL}}(\pi_\theta \| q^\star)\)
on that support (`loss_mode=reverse_kl_topk`, `use_policy_gradient=False`).
The 55k file already stores paper \(\tilde{x}\) in a top-level `text_prompt`
column (`question [IMAGE DESCRIPTION] <d>`). Student and VL teacher keep
`prompt` + image. The text teacher is scored on tokenized `text_prompt` (no
pixels). Mix jobs fail if that column is missing. Held-out acc comes from the
dataset's own eval parquets, picked by `VAL_SETS` (default `mathvista`;
`mathvista,mathverse,mathvision` for all three), cleaned into
`mopd/<set>_eval[_N].parquet` and passed as a `data.val_files` list so veRL
reports `val-core/<data_source>/acc/mean@1` per set. `watch.py` prints `acc`
(train rollouts), `val:<set>` per eval set, `val_acc` (all val rows),
`alpha_vl`, `H_vl`, `H_text`, `omega_k`, and `tilde_x` (1 = description path
used). Mix temperature defaults to the paper's \(\tau=0.5\) (`HOPD_MIX_TAU`).

```bash
# LS6 login. Cancel other GPU jobs first (-dev is 1 job/user; extra sbatch hurts Priority).
cd $SCRATCH/hopd/H-OPD && mkdir -p logs && git pull --ff-only
squeue -u $USER

# 1) debug: same block as `run_lightning.sh papercheck`, on gpu-a100-dev (2h).
#    No --exclude: if the job lands on 2-GPU c301-002, run_ls6.sh auto-enables
#    HOPD_TEACHERS_SHARE_GPU=1 (both 4B teachers on one GPU, Ray fractional
#    bundles, 0.40 util each). Same code paths, just slower. -t overrides the 12h header.
HOPD_PAPER=1 MOPD_MAX_ROWS=512 STEPS=12 TEST_FREQ=6 SAVE_FREQ=12 \
  VAL_SETS=mathvista,mathverse,mathvision VAL_MAX_ROWS=64 \
  sbatch -p gpu-a100-dev -t 02:00:00 --job-name=hopd-pchk \
  experiments/mopd/run_ls6.sh

# Interactive instead: idev -p gpu-a100-dev -m 120 -N 1 -n 1 ; then
#   cd $SCRATCH/hopd/H-OPD && bash experiments/mopd/run_lightning.sh papercheck
# (2 or 3 GPUs both work; tail logs/mopd_latest/training.log)

# 2) after the check shows tilde_x≈1, teacher_mass≈1, and val numbers:
HOPD_PAPER=1 MOPD_MAX_ROWS=2048 STEPS=80 TEST_FREQ=20 SAVE_FREQ=20 \
  VAL_SETS=mathvista,mathverse,mathvision VAL_MAX_ROWS=0 \
  sbatch -p gpu-a100 --exclude=c301-002 --job-name=hopd-p2k \
  experiments/mopd/run_ls6.sh
```

Known failure: before 2026-11-03 `run_ls6.sh` passed `--parquet train val` to
`filter_overlong.py`, whose flag was `action=append`; argparse rejected the
second path and exited **2** about ten minutes in, on every job that had a
separate val parquet (`HOPD_PAPER=1`). Fixed: one `--parquet` per file and the
flag now also accepts several paths.

`HOPD_MIX=1` is the older approximation: same dual scoring, but only \(p(y_t)\)
is mixed and the update is k1 + PPO. Do not stack paper reverse KL with
`HOPD_TASK_REWARD` / outcome GRPO in one job.

Still not paper *scale*: 256–20k / 2048 / 256 / bs8 on 3×A100, not 55k / 12384 /
128 on 8×B200, and acc is boxed MathVista-200 rather than VLMEvalKit. `acc` is
logged only; it is not in the paper loss. First paper job after this change
rebuilds the mix parquet (`MOPD_REBUILD=1` if a stale stamp is reused).

Rollouts gain `alpha_vl` (1 = VL teacher won that sample), `H_vl`/`H_text`
(top-k entropies), `omega_k` (union support size) and `used_text_prompt` on
the paper path. Every branch writes the same key set, including the
one-teacher-errored fallbacks, so `_postprocess` never hits a missing key.

## Lonestar6

```bash
cd $SCRATCH/hopd/H-OPD
mkdir -p logs
MOPD_MAX_ROWS=32 STEPS=2 sbatch experiments/mopd/run_ls6.sh
# after that job succeeds:
MOPD_FULL=1 sbatch experiments/mopd/run_ls6.sh
```
