#!/bin/bash
#SBATCH --job-name=hopd-mopd
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --partition=gpu-a100
#SBATCH -A ECS26006

# veRL routed MOPD on 3× A100-40GB (NOT paper H-OPD entropy mix).
#   1 GPU student 2B  +  1 GPU VL 4B teacher  +  1 GPU text 4B teacher
# Samples alternate hopd_vl / hopd_text via data_source.
#   cd $SCRATCH/hopd/H-OPD && mkdir -p logs && sbatch experiments/mopd/run_ls6.sh

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

export HOPD_HOME="${HOPD_HOME:-$SCRATCH/hopd/H-OPD}"
export HOPD_VENV="${HOPD_VENV:-$SCRATCH/hopd/.venv}"
export HOPD_SCRATCH="${HOPD_SCRATCH:-$SCRATCH/hopd}"
export HOPD_ENV_SCRIPT="${HOPD_ENV_SCRIPT:-$HOPD_HOME/setup/env_vars.ls6.sh}"

if [[ ! -f "$HOPD_HOME/config.sh" ]]; then
    echo "ERROR: $HOPD_HOME is not the H-OPD repo" >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$HOPD_HOME/config.sh"
cd "$HOPD_HOME"
mkdir -p "$HOPD_HOME/logs" "$HOPD_DATA_ROOT" "$HOPD_CKPT_ROOT"

if [[ ! -x "$HOPD_VENV/bin/python" ]]; then
    echo "No venv — setup/install_ls6.sh"
    bash "$HOPD_HOME/setup/install_ls6.sh"
fi
# shellcheck disable=SC1090
source "$HOPD_VENV/bin/activate"
hopd_activate_env || exit 1

SRC="$(find "$HOPD_DATA_ROOT" -name 'mmfine_reason_sampled_55k_text_prompt.parquet' | head -1 || true)"
if [[ -z "$SRC" || ! -f "$SRC" ]]; then
    python "$HOPD_HOME/download_data.py" --out_dir "$HOPD_DATA_ROOT"
    SRC="$(find "$HOPD_DATA_ROOT" -name 'mmfine_reason_sampled_55k_text_prompt.parquet' | head -1)"
fi
if [[ -z "$SRC" || ! -f "$SRC" ]]; then
    echo "ERROR: train parquet missing after download" >&2
    exit 1
fi

MOPD_TRAIN="$HOPD_DATA_ROOT/mopd/train_routed.parquet"
python "$HOPD_HOME/experiments/mopd/make_routed_parquet.py" --src "$SRC" --out "$MOPD_TRAIN"
VAL_FILE="$(find "$HOPD_DATA_ROOT" -name 'mathvista_200_test.parquet' | head -1 || true)"
if [[ -z "$VAL_FILE" || ! -f "$VAL_FILE" ]]; then
    VAL_FILE="$MOPD_TRAIN"
fi

export VLLM_USE_V1=1
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=8
export MALLOC_ARENA_MAX=4
export RAY_memory_monitor_refresh_ms=0
export RAY_memory_usage_threshold=0.99
export TOKENIZERS_PARALLELISM=false
export VERL_USE_UV=0

STUDENT_MODEL="Qwen/Qwen3-VL-2B-Instruct"
TEACHER_VL="Qwen/Qwen3-VL-4B-Instruct"
TEACHER_TEXT="Qwen/Qwen3-4B-Instruct-2507"
STEPS="${STEPS:-5}"
TRAIN_BSZ=8
PPO_MICRO_BSZ=1
MAX_PROMPT=512
MAX_RESPONSE=512
LR=1e-6
LOSS_MODE="k1"
TOPK=8
STUDENT_GPUS=1
TEACHER_GPUS=2
GPUS_ON_NODE=3
SAVE_DIR="$HOPD_CKPT_ROOT/hopd-mopd-${SLURM_JOB_ID:-local}"
RUN_LOG_DIR="$HOPD_LOG_DIR/mopd_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_LOG_DIR" "$SAVE_DIR"

cleanup() {
    ray stop --force 2>/dev/null || true
    kill $(jobs -p) 2>/dev/null || true
    wait
}
trap cleanup SIGINT SIGTERM EXIT

if [[ -n "${SLURM_JOB_NODELIST:-}" ]]; then
    nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
    nodes_array=($nodes)
    head_node=${nodes_array[0]}
    head_node_ip=$(getent hosts "$head_node" | awk '{print $1}')
else
    head_node_ip=$(hostname -I | awk '{print $1}')
fi

ray stop --force 2>/dev/null || true
RAY_PORT=6379
export ip_head=$head_node_ip:$RAY_PORT
export RAY_ADDRESS=$ip_head
ray start --head --node-ip-address="$head_node_ip" --port=$RAY_PORT \
    --num-cpus=32 --num-gpus=$GPUS_ON_NODE --block \
    > "$RUN_LOG_DIR/ray_head.log" 2>&1 &
sleep 15
ray status

MAX_NUM_TOKENS=$(( MAX_PROMPT + MAX_RESPONSE + 1 ))

python3 -m verl.trainer.main_ppo \
    --config-path=$VERL_CONFIG_PATH \
    --config-name=ppo_trainer.yaml \
    \
    data.train_files=$MOPD_TRAIN \
    data.val_files=$VAL_FILE \
    data.train_batch_size=$TRAIN_BSZ \
    data.max_prompt_length=$MAX_PROMPT \
    data.max_response_length=$MAX_RESPONSE \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=1 \
    data.dataloader_num_workers=2 \
    data.truncation='error' \
    data.shuffle=True \
    data.image_key=images \
    \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    \
    actor_rollout_ref.model.path=$STUDENT_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=False \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.use_torch_compile=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BSZ \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BSZ \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$(( PPO_MICRO_BSZ * (MAX_PROMPT + MAX_RESPONSE) )) \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.70 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.max_model_len=$MAX_NUM_TOKENS \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    \
    distillation.enabled=True \
    distillation.teacher_key=data_source \
    distillation.n_gpus_per_node=$TEACHER_GPUS \
    distillation.nnodes=1 \
    +distillation.teacher_models.vl.key=hopd_vl \
    +distillation.teacher_models.vl.model_path=$TEACHER_VL \
    +distillation.teacher_models.vl.num_replicas=1 \
    +distillation.teacher_models.vl.inference.name=vllm \
    +distillation.teacher_models.vl.inference.tensor_model_parallel_size=1 \
    +distillation.teacher_models.vl.inference.gpu_memory_utilization=0.70 \
    +distillation.teacher_models.vl.inference.enforce_eager=True \
    +distillation.teacher_models.vl.inference.max_model_len=$MAX_NUM_TOKENS \
    +distillation.teacher_models.text.key=hopd_text \
    +distillation.teacher_models.text.model_path=$TEACHER_TEXT \
    +distillation.teacher_models.text.num_replicas=1 \
    +distillation.teacher_models.text.inference.name=vllm \
    +distillation.teacher_models.text.inference.tensor_model_parallel_size=1 \
    +distillation.teacher_models.text.inference.gpu_memory_utilization=0.70 \
    +distillation.teacher_models.text.inference.enforce_eager=True \
    +distillation.teacher_models.text.inference.max_model_len=$MAX_NUM_TOKENS \
    distillation.distillation_loss.loss_mode=$LOSS_MODE \
    distillation.distillation_loss.topk=$TOPK \
    distillation.distillation_loss.use_policy_gradient=True \
    distillation.distillation_loss.use_task_rewards=False \
    distillation.distillation_loss.loss_max_clamp=10.0 \
    distillation.distillation_loss.log_prob_min_clamp=-10.0 \
    \
    trainer.logger="$HOPD_LOGGER" \
    trainer.project_name=$HOPD_WANDB_PROJECT \
    trainer.experiment_name=hopd-mopd-${SLURM_JOB_ID:-local} \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=$STUDENT_GPUS \
    trainer.resume_mode=disable \
    trainer.default_local_dir=$SAVE_DIR \
    trainer.val_before_train=False \
    trainer.total_epochs=1 \
    +trainer.total_training_steps=$STEPS \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    2>&1 | tee "$RUN_LOG_DIR/training.log"

echo "Checkpoints: $SAVE_DIR"
echo "Logs:        $RUN_LOG_DIR"
trap - EXIT
cleanup
echo "Done."
