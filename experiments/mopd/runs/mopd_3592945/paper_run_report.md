# Paper-level H-OPD run report — LS6 job 3592945

**Status:** completed 80/80 steps (2026-11-19).  
**Claim level:** plumbing + recipe confirmation only. **Not** a reproduction of Yin et al. Table 1.

Related artifacts: [`watch_table.md`](watch_table.md) · [`README.md`](README.md) · cluster logs `logs/hopd-p2k_3592945.out` · `logs/mopd_3592945/` · KB `docs/LS6.md`.

---

## 1. What we ran (architecture)

### Roles

| Role | Model | HF id | Placement (3× A100-40GB) |
|------|--------|-------|---------------------------|
| Student (actor + colocated rollout) | Qwen3-VL **2B** Instruct | `Qwen/Qwen3-VL-2B-Instruct` | GPU 0 — FSDP train + vLLM rollout (sleep mode) |
| VL teacher | Qwen3-VL **4B** Instruct | `Qwen/Qwen3-VL-4B-Instruct` | GPU 1 — vLLM |
| Text teacher | Qwen3 **4B** Instruct (2507) | `Qwen/Qwen3-4B-Instruct-2507` | GPU 2 — vLLM |

### Distillation path (`HOPD_PAPER=1`)

1. Student samples on-policy \(Y \sim \pi_S(\cdot \mid x)\) (images on VL path).
2. **Both** teachers score the **same** \(Y\) (not routed one-teacher-per-row MOPD).
3. Text teacher sees paper \(\tilde{x}\): parquet `text_prompt` with `[IMAGE DESCRIPTION]` (no pixels). Tracked as `tilde_x` / `used_text_prompt`.
4. Mix over union support \(\Omega_t = K_V \cup K_T\) (top-\(k\) ids from each teacher; tables padded to width `2·topk` for batch concat).
5. Gate from top-\(k\) entropies with temperature \(\tau\): VL weight logged as `alpha_vl`; entropies `H_vl`, `H_text`; support size `omega_k`.
6. Loss: **`reverse_kl_topk`** as supervised distill (`use_policy_gradient=False`, `use_task_rewards=False`). No PPO / GRPO advantage in the distill objective. Task `acc` is logged only.
7. Optimizer: AdamW, `lr=1e-6`, `foreach=False` (single-tensor Adam), FSDP param + optimizer offload on student.

### Stack

- Trainer: nested veRL fork `hopd` + `experiments/mopd/verl_overlay/` (tilde-x scoring, top-k pad, pre-Adam `empty_cache`).
- Rollout / teachers: vLLM, `enforce_eager=True`, student `enable_sleep_mode=True`, `free_cache_engine=True`.
- Attention: SDPA (`attn_implementation=sdpa`); `use_remove_padding=False` (no flash_attn).
- Cluster: TACC Lonestar6 `gpu-a100`, job **3592945** (`hopd-p2k`).

---

## 2. Recipe we followed vs paper (matched / not matched)

| Knob | Paper (Yin et al. / our reading of recipe) | This run (3592945) | Match? |
|------|--------------------------------------------|--------------------|--------|
| Student | Qwen3-VL-2B class | `Qwen3-VL-2B-Instruct` | Yes (family/size) |
| VL teacher | Qwen3-VL-4B class | `Qwen3-VL-4B-Instruct` | Yes |
| Text teacher | Qwen3-4B class | `Qwen3-4B-Instruct-2507` | Yes (instruct revision may differ) |
| Both teachers on same \(Y\) | Yes | Yes (`mix_teachers`) | Yes |
| \(\tilde{x}\) image descriptions for text teacher | Yes | parquet `text_prompt` / `tilde_x=1.00` | Yes |
| Mix support | \(\Omega_t=K_V\cup K_T\) | union + pad to `2·topk` | Yes |
| Mix temperature \(\tau\) | **0.5** (Table 4) | `HOPD_MIX_TAU=0.5` | Yes |
| Loss | reverse KL (token / top-k) | `reverse_kl_topk`, `topk=8` | Yes (top-k approx of full vocab) |
| No task reward in distill | Yes | `use_task_rewards=False` | Yes |
| Learning rate | \(1\times10^{-6}\) | `1e-6` | Yes |
| Max response | **128** | `MAX_RESPONSE=128` | Yes |
| Train data | full mmfine / ~**55k** paper set | **2048**-row mix slice (`MOPD_MAX_ROWS=2048`) | **No — ~27× smaller** |
| Context / prompt budget | paper ~**12k** class | `max_prompt_length=2048`, image capped (~256 tokens via `max_pixels`) | **No** |
| Batch size | paper **128** (multi-GPU) | **`TRAIN_BSZ=2`** (memory) | **No** |
| Hardware | paper **8× B200**-class | **3× A100-40GB** | **No** |
| Steps / schedule | full paper schedule | **80** steps, 1 epoch cap via `total_training_steps=80` | **No — smoke horizon** |
| Eval | VLMEvalKit / multi-bench, large \(n\) | MathVista-200 + MathVerse-200 + MathVision test (**n=704**), one pass at step 80 | Partial sets; **not** paper eval harness |
| Mid-run val | paper-dependent | **`TEST_FREQ=80`** (val only at end) to avoid Adam OOM after full val | Operational deviation |
| Student vLLM util | n/a | `0.25` colocated with FSDP | Cluster memory knob |

**Bottom line:** algorithm shape matches the paper H-OPD mix + reverse-KL + \(\tilde{x}\) + \(\tau=0.5\) path. **Scale, batch, context, hardware, and eval protocol do not.** Do not cite this as Table 1.

---

## 3. Job config (exact)

```bash
HOPD_PAPER=1 MOPD_MAX_ROWS=2048 STEPS=80 TEST_FREQ=80 SAVE_FREQ=20 \
  VAL_SETS=mathvista,mathverse,mathvision VAL_MAX_ROWS=0 VAL_BEFORE_TRAIN=False \
  STUDENT_GPU_UTIL=0.25 TRAIN_BSZ=2 \
  sbatch -p gpu-a100 --job-name=hopd-p2k experiments/mopd/run_ls6.sh
```

| Setting | Value |
|---------|--------|
| Job ID | 3592945 |
| Partition | `gpu-a100` |
| Train parquet | `train_routed_2048_mix.parquet` |
| Val files | `mathvista_200_test_eval` · `mathverse_200_test_eval` · `mathvision_test_eval` |
| `topk` | 8 |
| `mix_temperature` | 0.5 |
| `loss_mode` | `reverse_kl_topk` |
| `use_policy_gradient` | False |
| `rollout.n` | 1 |
| `resume_mode` | `disable` (fresh weights; prior resume attempts failed launcher wiring) |

**Why this shape:** jobs **3592493** and **3592795** trained ~20 steps then died in `Adam.step` (~1.55 GiB needed, ~1.5 GiB free) immediately after the first **full** val. Lowering student util to 0.25 did not shrink the ~34 GiB actor footprint. Escape: `TRAIN_BSZ=2` + `TEST_FREQ=80` (no mid-run val → no step-21 Adam after val).

---

## 4. Results

### Held-out accuracy (step 80, n=704)

| Benchmark | Acc |
|-----------|-----|
| MathVision | 0.688 |
| MathVerse | 0.500 |
| MathVista | 0.255 |
| **Mean `val_acc`** | **0.511** |

Same-recipe reference (not identical schedule): 3592493 val ≈ 0.513→0.526 over 20 steps; 3592795 step-20 ≈ 0.516. **Flat within noise** at this scale — expected for 80 steps × bs2 on 2k rows.

### Pipeline telemetry (healthy)

- `tilde_x = 1.00` throughout (descriptions used).
- `teacher_mass ≈ 1` (teacher top-k normalized).
- `omega_k ≈ 8–9` (union support populated).
- `alpha_vl ≈ 0.69–0.87` (VL-heavy mix).

### Optimization (noisy)

- Distill loss first→last ≈ **6.88 → 4.57** (down, high variance).
- `student_mass` stayed low (~0.02–0.22).
- Entropy rose (~1.1 → ~3.2) — not a clean mode collapse.
- **Step 71:** loss 0.173, `resp_len=3.5` (near-empty).
- **Step 72:** loss **76.58**, `grad_norm=2219`, then recovered.
- Train `acc` with **n=2** is **not** a learning curve.

### Watch table (last 15 steps incl. final val)

```
step  distill_loss  teacher_mass  student_mass  entropy  grad_norm  resp_len  tok/s  step_s  val:mathvision  val:mathverse  val:mathvista    acc  n  alpha_vl  H_vl  H_text  omega_k  tilde_x  val_acc
----  ------------  ------------  ------------  -------  ---------  --------  -----  ------  --------------  -------------  -------------  -----  -  --------  ----  ------  -------  -------  -------
  66          9.82          1.00        0.0523     3.29     160.51    128.00  10.42   19.04               -              -              -  1.000  2     0.719  0.59    1.32      8.9     1.00        -
  67          2.59           0.5        0.0428     2.55     161.68    128.00  11.23   20.73               -              -              -  0.500  2     0.858  0.25    1.22      5.0     1.00        -
  68          5.25          1.00        0.0483     3.07      96.26    128.00   9.53   24.49               -              -              -  1.000  2     0.739  0.62    1.41      8.9     1.00        -
  69          5.07          1.00         0.221     2.74      56.18    128.00   8.69   21.11               -              -              -  0.000  2     0.686  0.52    1.14      9.0     1.00        -
  70          4.37          1.00        0.0337     3.66      50.49    128.00   9.70   28.70               -              -              -  0.500  2     0.742  0.74    1.52      8.7     1.00        -
  71         0.173          1.00          0.14     4.02       7.19      3.50  13.15   21.25               -              -              -  1.000  2     0.871  0.85    2.06      8.0     1.00        -
  72         76.58          1.00        0.0636     3.96       2219     66.00   8.23   24.62               -              -              -  0.000  2     0.780  0.90    1.78      8.5     1.00        -
  73          9.40          1.00        0.0298     3.32     288.51    128.00   9.29   27.34               -              -              -  0.500  2     0.780  0.51    1.50      8.8     1.00        -
  74          3.84          1.00         0.171     3.96      52.59    128.00  11.00   16.16               -              -              -  1.000  2     0.686  0.74    1.35      9.2     1.00        -
  75          4.77          1.00        0.0345     4.90      69.41    128.00  10.21   22.79               -              -              -  0.500  2     0.727  0.68    1.43      8.9     1.00        -
  76          4.30          1.00        0.0325     3.59      71.30    110.50   8.40   26.52               -              -              -  0.500  2     0.737  0.68    1.46      8.9     1.00        -
  77          3.18          1.00       0.00982     3.47      56.00     67.50   7.79   25.26               -              -              -  0.500  2     0.786  0.87    1.76      8.5     1.00        -
  78         10.45          1.00        0.0306     4.05     137.00    128.00   9.89   23.50               -              -              -  0.000  2     0.705  0.70    1.37      8.8     1.00        -
  79          4.13          1.00        0.0288     3.99      49.56    128.00  10.30   25.90               -              -              -  1.000  2     0.751  0.77    1.59      8.8     1.00        -
  80          4.57          1.00        0.0398     3.20      84.38    128.00   4.27   50.97           0.688            0.5          0.255  0.500  2     0.719  0.68    1.38      8.9     1.00    0.511

first -> last:  distill_loss 6.88->4.57 (down) | teacher_mass 1.00->1.00 (flat) | student_mass 0.0187->0.0398 (up) | entropy 1.11->3.20 (up) | grad_norm 110.11->84.38 (down) | resp_len 128.00->128.00 (flat) | tok/s 2.44->4.27 (up) | step_s 101.00->50.97 (down)
rollout acc:    0 -> 0.5 over 80 steps
val acc:        step 80: 0.511 (n=704)
```

Dump all steps 1..80 from cluster:

```bash
python3 experiments/mopd/watch.py --run logs/mopd_3592945 --last 999 \
  > experiments/mopd/runs/mopd_3592945/watch_table.txt
python3 experiments/mopd/watch.py --run logs/mopd_3592945 \
  --csv experiments/mopd/runs/mopd_3592945/metrics.csv
```

---

## 5. Comparison to paper *claims* (what we can / cannot say)

| Paper-style claim | This run |
|-------------------|----------|
| H-OPD mix + reverse KL trains stably at paper **scale** | **Not tested.** We showed **smoke stability** for 80 steps at bs2 / 2k rows / 3×A100 after memory workarounds. |
| Held-out multimodal math gains (Table 1 numbers) | **Cannot claim.** `val_acc=0.511` is a single end-of-smoke number, flat vs early vals, wrong scale/eval. |
| \(\tilde{x}\) text teacher path matters | **Supported as implemented:** `tilde_x=1` every step; text teacher fed descriptions. No ablation here. |
| Entropy mix with \(\tau=0.5\) produces sensible \(\alpha\) | **Supported qualitatively:** `alpha_vl` in ~0.7–0.85, `omega_k≈9`. |
| Student matches teacher top-k mass | **Weak:** `student_mass` stayed low; loss fell only modestly. |

---

## 6. Presentation one-liner

> On Lonestar6 (3×A100-40GB) we ran the Yin-style H-OPD recipe — Qwen3-VL-2B student, VL-4B + text-4B teachers, both scoring the same on-policy \(Y\), \(\tilde{x}\) descriptions, \(\Omega_t=K_V\cup K_T\), \(\tau=0.5\), `reverse_kl_topk`, lr \(1\mathrm{e}{-6}\), response 128 — for 80 steps on a 2048-row slice (bs=2, end-only val). Mix telemetry and \(\tilde{x}\) look correct; final held-out mean acc 0.511 (n=704). This confirms the **pipeline**, not Table 1.

---

## 7. Remaining gaps vs paper (besides raw scale)

| Gap | Notes |
|-----|--------|
| Eval harness | Paper: VLMEvalKit / broader benches. We: 3 HF eval slices, custom `acc`. |
| Mid-run val + Adam | Paper can val often; we need end-only val or post-val `empty_cache` on 40GB. |
| Image in val dumps | veRL `val/*.jsonl` is text-only (`skip_special_tokens`). Rejoin pixels via `export_val_gallery.py`. |
| Stability | Step 71 empty / step 72 grad spike — inspect before trusting longer curves. |
| Baselines | No single-teacher / no-mix control at same scale yet. |
| Student mass | Stays low — weak top-k match vs paper-scale runs. |

Scale (55k, bs128, ~12k ctx, 8×B200) is still the main accuracy blocker.

## 8. Next if scaling toward paper claims

Only change **scale knobs** that matter for accuracy; keep the recipe (models, \(\tau=0.5\), `reverse_kl_topk`, lr, resp 128, \(\tilde{x}\)).

1. Sync launcher/overlay (resume + post-val `empty_cache` + `VAL_SETS=all`) before mid-run val.
2. **Next paper-ish job (safe on 3×A100):** 8k–20k rows, 200 steps, `TRAIN_BSZ=2`, end-only or sparse val.
3. Do **not** yet: bs128, 12k ctx, flash_attn, task reward, \(\tau\) sweeps, model swaps.
4. Inspect step-71/72 rollouts; build val image gallery with `export_val_gallery.py`.
5. Still not Table‑1 until data/batch/HW/eval match.
