# Paper H-OPD LS6 experiment log — 2026-11-18/19

**Claim level:** recipe confirmation + scale/response ablations on 3×A100-40GB. **Not Table‑1.**

Baseline writeup for the first smoke: [`mopd_3592945/paper_run_report.md`](mopd_3592945/paper_run_report.md).

---

## 1. Shared recipe (all runs unless noted)

| Knob | Value |
|------|--------|
| Flag | `HOPD_PAPER=1` |
| Student | `Qwen/Qwen3-VL-2B-Instruct` |
| VL teacher | `Qwen/Qwen3-VL-4B-Instruct` |
| Text teacher | `Qwen/Qwen3-4B-Instruct-2507` |
| Mix | both teachers on same \(Y\), \(\Omega_t=K_V\cup K_T\), \(\tau=0.5\), `topk=8` |
| Loss | `reverse_kl_topk`, no PPO / no task reward in loss |
| lr | `1e-6`, Adam `foreach=False` |
| Batch | `TRAIN_BSZ=2`, student vLLM util `0.25` |
| Prompt | `max_prompt_length=2048`, image pixel cap |
| \(\tilde{x}\) | parquet `text_prompt` — telemetry `tilde_x=1.00` on all logged steps |
| Val | MathVista-200 + MathVerse-200 + MathVision test (**n=704**), usually end-only |
| Hardware | LS6 `gpu-a100`, 3 GPUs |

---

## 2. Run configs & outcomes

| JobID | Name | Slurm | Data × steps | resp | Wall | **val_acc** | Vision | Verse | Vista | Notes |
|-------|------|-------|--------------|------|------|-------------|--------|-------|-------|-------|
| **3592945** | hopd-p2k | COMPLETED | 2k × 80 | 128 | ~smoke | **0.511** | 0.688 | 0.500 | 0.255 | First full smoke; clip_ratio=1.0 |
| **3593543** | hopd-p20k | COMPLETED | 20k × 200 | 128 | 2h01 | **0.504** | 0.668 | 0.500 | 0.260 | No gain; entropy 1.4→**9.0**; student_mass ↓ |
| **3593544** | hopd-r256 | COMPLETED | 2k × 40 | **256** | 1h03 | **0.543** | 0.678 | 0.495 | **0.385** | **Best**; student_mass≈0.22; clip≈0.5 |
| **3593882** | hopd-p55k | FAILED* | 55k × 500† | 128 | 4h19 | **0.531** | 0.688 | 0.500 | 0.325 | *Adam OOM step 501 after val; †`STEPS` ignored under `MOPD_FULL=1` |

\*Step-500 metrics/val are valid. Launcher now honors `STEPS` with `MOPD_FULL`.

### Training dynamics (first → last)

| Job | distill_loss | student_mass | entropy | resp_len mean / clip |
|-----|--------------|--------------|---------|----------------------|
| 3592945 | 6.88 → 4.57 | 0.019 → 0.040 | 1.11 → 3.20 | 128 / **1.0** |
| 3593543 | 6.36 → 4.37 | 0.057 → **0.010** | 1.36 → **8.97** | 128 / **1.0** |
| 3593544 | 5.36 → 4.02 | **0.226 → 0.223** | 1.29 → 2.25 | **222** / **0.5** |
| 3593882 | 5.46 → 4.08 | 0.008 → 0.081 | 0.40 → **8.52** | 128 / **1.0** |

**Interpretation:** At resp **128**, every train token hits the cap (`clip_ratio=1`). Longer runs get **high entropy + low student_mass** (diffuse student, weak top-k match) and **no held-out gain**. Resp **256** keeps mass ~0.22, milder entropy, and the only clear **Vista** lift.

---

## 3. Validation failure / success taxonomy (n=704 dumps)

Grading (`task_reward.py`): prefer `\boxed{}` via mathruler; else gold substring in **last 200 chars**. Almost never boxed.

| Job | acc | wrong/right | `\boxed` (R/W) | mid-sentence cut (R/W) | mean chars |
|-----|-----|-------------|----------------|------------------------|------------|
| 3592945 | 0.511 | 344/360 | 2/2 | 333/313 | ~465 |
| 3593544 | 0.543 | 322/382 | 9/11 | 314/267 | ~**840** |
| 3593543 | 0.504 | 349/355 | 8/5 | 335/316 | ~465 |
| 3593882 | 0.531 | 330/374 | 8/7 | 346/301 | ~440 |

### Success types (what “correct” usually means here)

1. **Gold appears near the end of truncated CoT** (substring grader) — dominant; many “right” answers are still mid-cut.
2. **Short / early numeric answer** that survives in the last 200 chars.
3. **Rare `\boxed{...}`** (&lt;2% of rows).

### Failure types

1. **Truncation before the answer** (resp 128 especially) — CoT cut mid-sentence, gold never lands in the scored tail.
2. **Wrong final number / reasoning** with full-ish text (more visible at resp 256).
3. **Gold mentioned early but not in last 200 chars** → scored wrong (grader artifact).
4. **False-ish correct**: gold digit string appears in working (e.g. intermediate) near the end.

### Train rollouts

- `n=2` per step → rollout `acc` is noise; use **val_acc**.
- `tilde_x=1` on checked steps (descriptions wired).
- Spike history (3592945): step 71 near-empty; step 72 loss/grad spike — recovered.

Artifacts: `mopd_<job>/{cluster,rollouts,val}/`. Gallery: prefer `mopd_3593544/val_gallery` with `--n 50`.

---

## 4. Conclusions → next runs

1. **Do not keep scaling data at resp 128 only** — 20k/55k did not beat 2k×80 meaningfully; entropy pathology.
2. **Resp 256 is the small change that helped** — especially MathVista.
3. Keep **bs2**, util **0.25**, end-only val (`TEST_FREQ=STEPS`) to avoid post-val Adam OOM.
4. Always set **`STEPS=...` even with `MOPD_FULL=1`** (fixed in launcher).
5. Still not paper scale (bs128 / 12k ctx / 8×B200 / VLMEvalKit).

### Queue next (small, safe)

**A — extend the winner (2k × 80 × resp 256)**  
Same data as best run, more steps:

```bash
cd $SCRATCH/hopd/H-OPD && git pull --ff-only origin main

HOPD_PAPER=1 MOPD_MAX_ROWS=2048 STEPS=80 TEST_FREQ=80 SAVE_FREQ=40 \
  VAL_SETS=mathvista,mathverse,mathvision VAL_MAX_ROWS=0 VAL_BEFORE_TRAIN=False \
  STUDENT_GPU_UTIL=0.25 TRAIN_BSZ=2 MAX_RESPONSE=256 \
  HOPD_SKIP_IMPORT_CHECK=1 \
  sbatch -p gpu-a100 --job-name=hopd-r256b --time=12:00:00 experiments/mopd/run_ls6.sh
```

**B — more data at resp 256 (8k × 120)**  
Only scale-up that matches the winning response length:

```bash
HOPD_PAPER=1 MOPD_MAX_ROWS=8192 STEPS=120 TEST_FREQ=120 SAVE_FREQ=40 \
  VAL_SETS=mathvista,mathverse,mathvision VAL_MAX_ROWS=0 VAL_BEFORE_TRAIN=False \
  STUDENT_GPU_UTIL=0.25 TRAIN_BSZ=2 MAX_RESPONSE=256 \
  HOPD_SKIP_IMPORT_CHECK=1 \
  sbatch -p gpu-a100 --job-name=hopd-r256-8k --time=18:00:00 experiments/mopd/run_ls6.sh
```

**Avoid for now:** bs4/8, resp 512 (Adam risk), mid-run full val every 20, τ/topk sweeps, uncapped `MOPD_FULL` without `STEPS`.

---

## 5. Job ID graveyard (failed / superseded)

| JobID | Fate |
|-------|------|
| 3592493 / 3592795 | Adam OOM after first full val @ resp128/bs4 |
| 3592785 | `VAL_SETS=all` typo |
| 3593297 / 3593342 | `UnboundLocalError: torch` (bad post-val patch) |
| 3593882 | Val@500 OK; OOM step 501 (`STEPS` ignored) |
