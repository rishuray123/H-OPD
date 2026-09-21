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
- 2026-09-09: Login-node bootstrap succeeded: `$SCRATCH/hopd/H-OPD` at `a320708`, nested `verl/` on `hopd` (`044bbba2`). `install_ls6.sh` still needs a GPU node.
- 2026-09-15: Offline token-prob dump lives in `experiments/token_prob_dump/` (not the trainer). `sbatch experiments/token_prob_dump/run_ls6.sh` on `gpu-a100`. Writes `$SCRATCH/hopd/data/token_prob_dump/`.
- 2026-09-15: Login preflight at `a04ceff` OK; no mmfine parquet yet. Dump sbatch now installs venv and downloads data if missing.
- 2026-09-15: Jobs `3442692` (`idev`) and `3443317` (`hopd-tokdump`) pending on `gpu-a100` (`Priority`). Cancel leftover `idev` if using sbatch.
- 2026-09-15: Routed MOPD job: `sbatch experiments/mopd/run_ls6.sh` (1 GPU student + VL 4B + text 4B). Not H-OPD entropy mix.
- 2026-09-21: **flash_attn is required**, not optional. `_compute_old_log_prob` → `left_right_2_no_padding` → `verl.utils.attention_utils.unpad_input` imports `flash_attn.bert_padding` on CUDA regardless of `use_remove_padding` / `attn_implementation=sdpa`; without it step 1 dies with `ModuleNotFoundError: No module named 'flash_attn'`. Install the prebuilt wheel (no source build): `flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl` from the Dao-AILab v2.8.3 release. `install_ls6.sh` no longer uninstalls it.
- 2026-09-21: `data.filter_overlong_prompts=True` is mandatory for VL rows. It measures the **processed** prompt length (image tokens expanded); with it off, an overlong prompt reaches the agent loop and `DataProto.concat` dies with `Sizes of tensors must match except in dimension 0. Expected size 512 but got size 930`. `max_prompt_length` now defaults to 2048 because image tokens dwarf the text length; slice ~96 rows for a 2-step smoke since filtering drops some.
- 2026-09-21: veRL scores every rollout even with `distillation.use_task_rewards=False`, and `default_compute_score` only knows public dataset names — custom `data_source` values raise `NotImplementedError: Reward function is not implemented for data_source='hopd_text'`. Fixed with `reward.custom_reward_function.path=experiments/mopd/zero_reward.py` (returns 0.0).
- 2026-09-21: numpy must be **2.2.6**. veRL's `setup.py` declares `numpy<2.0.0` and the editable install drags numpy to 1.26, but vLLM's GPU worker imports numba (`numpy<=2.2`) and scipy needs `numpy>=2.0`. Wrong version = `Engine core initialization failed` / `WorkerProc failed to start` with `Numba needs NumPy 2.2 or less`. `install_ls6.sh` now pins numpy after `pip install -e verl`.
- 2026-09-21: Lightning gotchas: `ray start --block` in the background leaves GCS unreachable (`Failed to connect to GCS`) — start it in the foreground and wipe `/tmp/ray`. `transformers` hides `PreTrainedModel`/`MistralForSequenceClassification` whenever `is_torch_available()` is False (half-done install), so fix the env, not the imports. `experiments/mopd/doctor.py` checks the whole chain. Bare `python` on a Studio can be base conda even when the prompt shows `(.venv)`.
- 2026-09-19: MOPD smoke slices the real mmfine parquet (`MOPD_MAX_ROWS=32`, 2 steps). Full: `MOPD_FULL=1` (1 epoch). Lightning: `bash experiments/mopd/run_lightning.sh smoke` then `full`.
