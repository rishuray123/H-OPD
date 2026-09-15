# Routed MOPD (veRL), not paper H-OPD

veRL assigns **one** teacher per sample via `data_source`. That is **not** token-level mix of VL + text on the same \(Y\).

GPU split on one LS6 node: student 2B on **1** A100, VL 4B teacher on **1**, text `Qwen3-4B-Instruct-2507` on **1**.

Even rows → VL teacher (images kept). Odd rows → text teacher (images cleared).

```bash
cd $HOPD_HOME
git pull --ff-only origin main
mkdir -p logs
sbatch experiments/mopd/run_ls6.sh
squeue -u $USER
```

Smoke: 5 steps. Install/download run if the dump job has not created them yet. Prefer not overlapping two first-time `install_ls6.sh` processes; queuing behind `hopd-tokdump` is fine.
