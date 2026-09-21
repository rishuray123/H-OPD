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

- 2026-09-21: **Routed MOPD smoke passed end to end** on Lightning (3 of 4× L40S 48GB): 96-row slice, 2 steps, ~53s then ~22s per step, `actor/distillation/loss` ≈ −0.8, rewards flat 0.0 by design. Order of failures fixed to get there: Ray bootstrap → numpy 2.2.6 → zero reward fn → prompt-length filter → flash_attn removal → image struct normalization. Caveat: `prompt_length/max` was only 226, i.e. text-sized, so confirm VL rows survive filtering before trusting the mix. `response_length/clip_ratio` 0.5–0.625 at 512 tokens, so raise `MAX_RESPONSE` for real runs.

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
- 2026-09-21: Do **not** install flash_attn. `_compute_old_log_prob` → `left_right_2_no_padding` → `verl.utils.attention_utils` wanted `flash_attn.bert_padding` on CUDA regardless of `use_remove_padding` / `attn_implementation=sdpa`, but installing flash_attn 2.8.3 then breaks vLLM: `vllm_flash_attn` imports `flash_attn.cute` and dies with `cutlass.cute.core has no attribute ThrMma`. Fixed on verl `hopd` (`859553fa`) by falling back to `transformers.modeling_flash_attention_utils._unpad_input/_pad_input/_index_first_axis` + `einops.rearrange`, which are documented as FA2-equivalent.
- 2026-09-21: H-OPD image structs carry **both** `bytes` and `image` (one null), and `process_image` asserts they are never both present — a null field still counts. Result: every VL row silently dropped (`filter dataset len: 48` out of 96, all text). `make_routed_parquet.py` now strips null struct fields and prints how many rows carry images.
- 2026-09-21: `data.filter_overlong_prompts=True` is mandatory for VL rows. It measures the **processed** prompt length (image tokens expanded); with it off, an overlong prompt reaches the agent loop and `DataProto.concat` dies with `Sizes of tensors must match except in dimension 0. Expected size 512 but got size 930`. `max_prompt_length` now defaults to 2048 because image tokens dwarf the text length; slice ~96 rows for a 2-step smoke since filtering drops some.
- 2026-09-21: veRL scores every rollout even with `distillation.use_task_rewards=False`, and `default_compute_score` only knows public dataset names — custom `data_source` values raise `NotImplementedError: Reward function is not implemented for data_source='hopd_text'`. Fixed with `reward.custom_reward_function.path=experiments/mopd/task_reward.py`, which keeps `score=0.0` (so the distillation loss is untouched) and returns an `acc` extra that shows up in the rollout dumps and val metrics.
- 2026-09-21: numpy must be **2.2.6**. veRL's `setup.py` declares `numpy<2.0.0` and the editable install drags numpy to 1.26, but vLLM's GPU worker imports numba (`numpy<=2.2`) and scipy needs `numpy>=2.0`. Wrong version = `Engine core initialization failed` / `WorkerProc failed to start` with `Numba needs NumPy 2.2 or less`. `install_ls6.sh` now pins numpy after `pip install -e verl`.
- 2026-09-21: Lightning gotchas: `ray start --block` in the background leaves GCS unreachable (`Failed to connect to GCS`) — start it in the foreground and wipe `/tmp/ray`. `transformers` hides `PreTrainedModel`/`MistralForSequenceClassification` whenever `is_torch_available()` is False (half-done install), so fix the env, not the imports. `experiments/mopd/doctor.py` checks the whole chain. Bare `python` on a Studio can be base conda even when the prompt shows `(.venv)`.
- 2026-09-19: MOPD smoke slices the real mmfine parquet (`MOPD_MAX_ROWS=32`, 2 steps). Full: `MOPD_FULL=1` (1 epoch). Lightning: `bash experiments/mopd/run_lightning.sh smoke` then `full`.
- 2026-09-21: Clean image structs are not enough — at full resolution a single image outgrows `max_prompt_length=2048` on its own, so `filter dataset len: 48/96` again with every `hopd_vl` row gone. `make_routed_parquet.py` now writes `max_pixels`/`min_pixels` into each image struct (`--image_max_pixels`, default 256·32·32 ≈ 256 image tokens); `qwen_vl_utils.fetch_image` reads them, so the cap applies to both the filter and the rollout. `data.image_patch_size=16` matches Qwen3-VL (default 14 is Qwen2-VL). Verify on CPU with `experiments/mopd/probe_lengths.py --parquet <routed.parquet>`, which prints per-`data_source` prompt lengths and re-raises the exception veRL swallows as "Error processing one of the samples, skipping...".
- 2026-09-21: Run observability: `HOPD_LOGGER` defaults to `["console","tensorboard","file"]`. The `file` backend writes `$RUN_LOG_DIR/metrics.jsonl` (`VERL_FILE_LOGGER_PATH`), tensorboard goes to `$RUN_LOG_DIR/tensorboard` (`TENSORBOARD_DIR`), and `trainer.rollout_data_dir` / `validation_data_dir` dump one JSONL per step. `logs/mopd_latest` symlinks the current run. Read them with `experiments/mopd/watch.py --follow` (metric table + rollout accuracy) and `experiments/mopd/show_rollouts.py`. `TEST_FREQ` is -1 unless set; validation costs a full pass over the val file.
- 2026-09-21: pandas `image keys: ['bytes', 'max_pixels', 'min_pixels']` is not what the trainer sees. HuggingFace `datasets` reloads every Arrow struct field; `df.to_parquet` kept the source schema's null `image`, so `process_image` still hit `Cannot have both bytes and image` and veRL counted those as overlong (`filter dataset len: 48`). `make_routed_parquet.py` now writes an explicit `list<struct<bytes, max_pixels, min_pixels>>` schema. `process_image` on verl `hopd` drops null struct fields before the assert (mathvista val parquet has the same trap). After pull: `bash setup/bootstrap_verl.sh` then rebuild the routed parquet.
- 2026-09-22: Paper H-OPD mix is `HOPD_MIX=1` / `bash experiments/mopd/run_lightning.sh mix`. veRL `hopd` scores both teachers on the same student \(Y\) and blends \(p(y_t)\) by top-k entropy (`distillation.mix_teachers`). Routed MOPD (one teacher per `data_source`) stays the default. Mix writes a separate parquet (`*_mix.parquet`, images kept on every row). Task reward in the loss is `HOPD_TASK_REWARD=1` and needs `rollout.n>=4`; leave it off for `n=1`. Full runs checkpoint every 50 steps (`SAVE_FREQ_FULL`); Ctrl-C before the first save leaves no actor weights.
- 2026-09-22: `run_ls6.sh` with `set -u` died on Lightning: `SCRATCH: unbound variable` because the defaults were `$SCRATCH/hopd/...`. It now falls back to `$HOPD_HOME/scratch` and skips `env_vars.ls6.sh` when `SCRATCH` is unset. Direct `bash experiments/mopd/run_ls6.sh` on a Studio no longer needs `run_lightning.sh` for those paths.
- 2026-09-22: `datasets.load_dataset("parquet")` dies on the 20k mix file (`train_routed_20000_mix.parquet`) with `ArrowNotImplementedError: Nested data conversions not implemented for chunked array outputs` — every row carries a `list<struct<bytes>>` image. The 96-row mix smoke was small enough to load. veRL `hopd` now falls back to `pq.read_table(...).combine_chunks()` + `Dataset.from_arrow`. `make_routed_parquet.py` writes smaller row groups (1024). After `bootstrap_verl.sh`, rerun the same `HOPD_MIX=1` command; the existing mix parquet does not need a rebuild.
