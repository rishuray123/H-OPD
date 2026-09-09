#!/bin/bash
#SBATCH --job-name=hopd-vl-opd
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:00:00
#SBATCH --partition=gpu-a100
#SBATCH -A ECS26006

# Stage 1: single VL teacher OPD on Lonestar6 (3× A100-40GB).
#   idev -p gpu-a100-dev -A ECS26006 -N 1 -n 1 -t 02:00:00
#   bash hopd_vl_opd_ls6.sh --steps 5

if [[ -z "${HOPD_HOME:-}" ]]; then
    _cand="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
    [[ -f "$_cand/config.sh" ]] && HOPD_HOME="$_cand"
fi
if [[ -z "${HOPD_HOME:-}" || ! -f "$HOPD_HOME/config.sh" ]]; then
    echo "ERROR: set HOPD_HOME to this repo" >&2
    exit 1
fi

export HOPD_ENV_SCRIPT="${HOPD_ENV_SCRIPT:-$HOPD_HOME/setup/env_vars.ls6.sh}"
# shellcheck disable=SC1091
source "$HOPD_HOME/config.sh"
hopd_activate_env || exit 1
set -x

export VLLM_USE_V1=1
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=8
export MALLOC_ARENA_MAX=4
export RAY_memory_monitor_refresh_ms=0
export RAY_memory_usage_threshold=0.99
export TOKENIZERS_PARALLELISM=false
export VERL_USE_UV=0

STUDENT_MODEL="Qwen/Qwen3-VL-2B-Instruct"
TEACHER_MODEL="Qwen/Qwen3-VL-4B-Instruct"
FULL=0
STEPS=5
EPOCHS=1
DATA_DIR="$HOPD_DATA_ROOT"
TRAIN_FILE=""
VAL_FILE=""
SAVE_DIR=""
TRAIN_BSZ=8
PPO_MICRO_BSZ=1
MAX_PROMPT=512
MAX_RESPONSE=512
LR=1e-6
LOSS_MODE="k1"
TOPK=8
STUDENT_GPUS_PER_NODE=2
TEACHER_GPUS_PER_NODE=1
GPUS_ON_NODE=3

while [[ $# -gt 0 ]]; do
    case "$1" in
        --student)       STUDENT_MODEL="$2"; shift 2 ;;
        --teacher)       TEACHER_MODEL="$2"; shift 2 ;;
        --full)          FULL=1; shift ;;
        --steps)         STEPS="$2"; shift 2 ;;
        --data_dir)      DATA_DIR="$2"; shift 2 ;;
        --train_file)    TRAIN_FILE="$2"; shift 2 ;;
        --val_file)      VAL_FILE="$2"; shift 2 ;;
        --save_dir)      SAVE_DIR="$2"; shift 2 ;;
        --train_bsz)     TRAIN_BSZ="$2"; shift 2 ;;
        --max_prompt)    MAX_PROMPT="$2"; shift 2 ;;
        --max_response)  MAX_RESPONSE="$2"; shift 2 ;;
        --loss_mode)     LOSS_MODE="$2"; shift 2 ;;
        --topk)          TOPK="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$TRAIN_FILE" ]]; then
    TRAIN_FILE="$(find "$DATA_DIR" -name 'mmfine_reason_sampled_55k_text_prompt.parquet' | head -1)"
fi
if [[ -z "$VAL_FILE" ]]; then
    VAL_FILE="$(find "$DATA_DIR" -name 'mathvista_200_test.parquet' | head -1)"
fi
if [[ -z "$TRAIN_FILE" || ! -f "$TRAIN_FILE" ]]; then
    echo "ERROR: train parquet not found under $DATA_DIR" >&2
    echo "       python download_data.py --out_dir \$HOPD_DATA_ROOT" >&2
    exit 1
fi
if [[ -z "$VAL_FILE" || ! -f "$VAL_FILE" ]]; then
    VAL_FILE="$TRAIN_FILE"
    echo "WARN: no val parquet, using train file"
fi
if [[ -z "$SAVE_DIR" ]]; then
    SAVE_DIR="$HOPD_CKPT_ROOT/hopd-vl-opd-${SLURM_JOB_ID:-local}"
fi

RUN_LOG_DIR="$HOPD_LOG_DIR/run_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_LOG_DIR" "$SAVE_DIR" "$HOPD_HOME/logs"

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

EXTRA=()
if [[ "$FULL" == "1" ]]; then
    EXTRA+=(
        "trainer.total_epochs=$EPOCHS"
        "trainer.save_freq=50"
        "trainer.test_freq=50"
    )
else
    EXTRA+=(
        "trainer.total_epochs=1"
        "+trainer.total_training_steps=$STEPS"
        "trainer.save_freq=-1"
        "trainer.test_freq=-1"
    )
fi

MAX_NUM_TOKENS=$(( MAX_PROMPT + MAX_RESPONSE + 1 ))

python3 -m verl.trainer.main_ppo \
    --config-path=$VERL_CONFIG_PATH \
    --config-name=ppo_trainer.yaml \
    \
    data.train_files=$TRAIN_FILE \
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
    distillation.n_gpus_per_node=$TEACHER_GPUS_PER_NODE \
    distillation.nnodes=1 \
    distillation.teacher_models.teacher_model.model_path=$TEACHER_MODEL \
    distillation.teacher_models.teacher_model.inference.name=vllm \
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=1 \
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.70 \
    distillation.teacher_models.teacher_model.inference.enforce_eager=True \
    distillation.teacher_models.teacher_model.inference.max_model_len=$MAX_NUM_TOKENS \
    distillation.distillation_loss.loss_mode=$LOSS_MODE \
    distillation.distillation_loss.topk=$TOPK \
    distillation.distillation_loss.use_policy_gradient=True \
    distillation.distillation_loss.use_task_rewards=False \
    distillation.distillation_loss.loss_max_clamp=10.0 \
    distillation.distillation_loss.log_prob_min_clamp=-10.0 \
    \
    trainer.logger="$HOPD_LOGGER" \
    trainer.project_name=$HOPD_WANDB_PROJECT \
    trainer.experiment_name=hopd-vl-opd-${SLURM_JOB_ID:-local} \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=$STUDENT_GPUS_PER_NODE \
    trainer.resume_mode=disable \
    trainer.default_local_dir=$SAVE_DIR \
    trainer.val_before_train=False \
    "${EXTRA[@]}" \
    2>&1 | tee "$RUN_LOG_DIR/training.log"

echo "Checkpoints: $SAVE_DIR"
echo "Logs:        $RUN_LOG_DIR"
trap - EXIT
cleanup
echo "Done."
