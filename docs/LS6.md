# Lonestar6 notes (H-OPD)

Cluster: TACC Lonestar6. Allocation **ECS26006**. User `rkr758`.

## Paths for this project

Separate from MATH OPDL (`$SCRATCH/opdl/`).

```
$SCRATCH/hopd/H-OPD           # this git repo
$SCRATCH/hopd/H-OPD/verl      # nested clone of rishuray123/verl (gitignored)
$SCRATCH/hopd/.venv
$SCRATCH/hopd/data
$SCRATCH/hopd/checkpoints
$SCRATCH/.cache/huggingface   # shared cache is fine
$SCRATCH/.cache/uv
```

Venvs and checkpoints on `$SCRATCH`, never `$WORK`.

## Queues

- `gpu-a100-dev`: 3× A100-40GB, max **2 hours**, **1 job per user**. Often sits in queue.
- `gpu-a100`: same node type, up to 48 hours, more jobs allowed. Prefer this if `idev` on `-dev` hangs.
- Do **not** use `gpu-a100-small` (virtual slice, not a full 3-GPU node). Do not use `gpu-h100` with this cu128 / vLLM 0.12 pin.

```bash
# cancel a stuck -dev wait (Ctrl-C), then:
idev -p gpu-a100 -A ECS26006 -N 1 -n 1 -t 04:00:00
```

## veRL

- GitHub: https://github.com/rishuray123/verl
- **Run branch `hopd`**: H-OPD trainer patches. Nested at `$HOPD_HOME/verl`.
- Floor / branch `ls6`: `044bbba27e5861e1212727aba67813cccb2db291` (cu128 / vLLM 0.12). `hopd` must contain this commit.
- `main` on that fork tracks upstream (currently CUDA 13 / vLLM 0.24 — do not install that on LS6).

Update an existing LS6 checkout:

```bash
git -C $HOPD_HOME pull --ff-only origin main
bash $HOPD_HOME/setup/bootstrap_verl.sh   # fetch+checkout origin/hopd
```

## Status log

- 2026-09-04: Forked H-OPD and verl. Added LS6 harness for stage-1 VL OPD. Install/smoke not run yet.
- 2026-09-05: veRL default path is `$HOPD_HOME/verl` (nested, gitignored), not a sibling checkout.
- 2026-09-09: Experiment branch `hopd` created on rishuray123/verl from `ls6` (`044bbba2`). Harness bootstraps `origin/hopd`. Install/smoke not run yet.
- 2026-09-09: Existing clones: `git pull --ff-only` on H-OPD, then `bootstrap_verl.sh` for nested `verl/` (`origin/hopd`).
- 2026-09-09: `gpu-a100-dev` often queues; use `idev -p gpu-a100` (full 3× A100). Not `gpu-a100-small`.
- 2026-09-09: Harness pushed to `rishuray123/H-OPD` `main` (`d4d0d37`). Cluster clone of older `dcff686` must `git pull` before `bootstrap_verl.sh`.
