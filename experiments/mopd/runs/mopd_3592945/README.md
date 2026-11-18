# mopd_3592945 (`hopd-p2k`)

Paper H-OPD smoke on LS6 `gpu-a100` / `c303-003` (confirm node in `.out`).

| | |
|---|---|
| Job | **3592945** |
| Config | `HOPD_PAPER=1` `MOPD_MAX_ROWS=2048` `STEPS=80` `TRAIN_BSZ=2` `STUDENT_GPU_UTIL=0.25` `TEST_FREQ=80` `SAVE_FREQ=20` `VAL_BEFORE_TRAIN=False` |
| Why this shape | Mid-run full val + Adam OOM on 3592493/3592795; end-only val + bs2 to finish 80 steps |
| Logs | `$SCRATCH/hopd/H-OPD/logs/hopd-p2k_3592945.out` · run dir `logs/mopd_3592945/` |
| Full writeup | [`paper_run_report.md`](paper_run_report.md) — recipe, architecture, table, vs paper claims |
| Watch snapshot | [`watch_table.md`](watch_table.md) (paste from live watch; may be last-N only) |
| Full export | Prefer CSV from `metrics.jsonl` (command below) → `metrics.csv` |

## Refresh full table from cluster

On login (after `source $SCRATCH/hopd/.venv/bin/activate` · `cd $SCRATCH/hopd/H-OPD`):

```bash
mkdir -p experiments/mopd/runs/mopd_3592945
python3 experiments/mopd/watch.py --run logs/mopd_3592945 --last 999 \
  > experiments/mopd/runs/mopd_3592945/watch_table.txt
python3 experiments/mopd/watch.py --run logs/mopd_3592945 \
  --csv experiments/mopd/runs/mopd_3592945/metrics.csv
# then sync this folder into the laptop clone if needed
```

## Final result (step 80)

- **Completed 80/80** without post-val Adam OOM (end-only val).
- `val_acc` **0.511** (n=704): MathVision 0.688 · MathVerse 0.500 · MathVista 0.255.
- Compare prior mid-run vals on same recipe: 3592493 step0/20 ≈ 0.513→0.526; 3592795 step20 ≈ 0.516 — flat within noise at this scale.
- Not Table‑1.

## Reading the curve

- Plumbing OK: `tilde_x=1`, `omega_k≈8–9`, `teacher_mass≈1`.
- Loss weakly down (6.88→4.57) but spiky (step 72).
- Rollout `acc` with `n=2` is not a learning curve.
- See also `docs/LS6.md` status entry 2026-11-19.
