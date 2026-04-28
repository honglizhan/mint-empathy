#!/bin/bash
# VERL GRPO training launcher for the paper's primary configuration.
# Quality + Diversity reward: PsychoCounsel RM (Q) + tactic diversity (D_KL, H).
#
# Reproduces Table 2 configurations:
#   Q+D_KL:     KL_GAMMA=1.0  ENTROPY_GAMMA=0.0
#   Q+H:        KL_GAMMA=0.0  ENTROPY_GAMMA=1.0
#   Q+D_KL+H:   KL_GAMMA=0.5  ENTROPY_GAMMA=0.5

set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPTS_DIR/.." && pwd)"
PYTHON="${PYTHON:-python}"

if [ -z "${CUDA_HOME:-}" ]; then
  if command -v nvcc >/dev/null 2>&1; then
    CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
    export CUDA_HOME
  else
    for CUDA_CANDIDATE in \
      /home1/apps/nvidia/Linux_x86_64/25.3/cuda \
      /home1/apps/nvidia/Linux_aarch64/25.3/cuda \
      /home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8 \
      /usr/local/cuda; do
      if [ -x "$CUDA_CANDIDATE/bin/nvcc" ]; then
        CUDA_HOME="$CUDA_CANDIDATE"
        export CUDA_HOME
        break
      fi
    done
  fi
fi

if [ -n "${CUDA_HOME:-}" ]; then
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi

if [ -n "${RAY_ADDRESS:-}" ]; then
  export RAY_ADDRESS
fi

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-1.7B}"
TAGGER_URLS="${TAGGER_URLS:-http://localhost:8100/v1}"
RM_SERVER_URL="${RM_SERVER_URL:-http://localhost:5100}"
PROMPTS_DIR="${PROMPTS_DIR:-$PROJECT_DIR/../tactic_tagger/prompts/}"
KL_GAMMA="${KL_GAMMA:-1.0}"
ENTROPY_GAMMA="${ENTROPY_GAMMA:-0.0}"
KL_COEF="${KL_COEF:-0.01}"
DIVERSITY_WEIGHT="${DIVERSITY_WEIGHT:-1.0}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.4}"
NNODES="${NNODES:-1}"
UID_BATCH_TIMEOUT_S="${UID_BATCH_TIMEOUT_S:-1800}"
TACTIC_TAGGER_TIMEOUT_S="${TACTIC_TAGGER_TIMEOUT_S:-20}"
TACTIC_TAGGER_MAX_RETRIES="${TACTIC_TAGGER_MAX_RETRIES:-6}"
TACTIC_TAGGER_RETRY_BASE_DELAY_S="${TACTIC_TAGGER_RETRY_BASE_DELAY_S:-1.0}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-24}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-$TRAIN_BATCH_SIZE}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-4}"
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-8}"
ROLLOUT_N="${ROLLOUT_N:-8}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
SAVE_FREQ="${SAVE_FREQ:-50}"
TEST_FREQ="${TEST_FREQ:--1}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
LOGGER=${LOGGER:-'["console","wandb"]'}
TRAIN_FILE="${TRAIN_FILE:-$PROJECT_DIR/data/train.parquet}"
VAL_FILE="${VAL_FILE:-$TRAIN_FILE}"

# Build experiment name from reward weights
MODEL_SHORT=$(basename "$MODEL_PATH")
TAG="q"
[ "$KL_GAMMA" != "0.0" ] && [ "$KL_GAMMA" != "0" ] && TAG="${TAG}-dkl${KL_GAMMA}"
[ "$ENTROPY_GAMMA" != "0.0" ] && [ "$ENTROPY_GAMMA" != "0" ] && TAG="${TAG}-h${ENTROPY_GAMMA}"
EXPERIMENT_NAME="${MODEL_SHORT}_bs${TRAIN_BATCH_SIZE}_n${ROLLOUT_N}_lr${LEARNING_RATE}_kl${KL_COEF}_rlen${MAX_RESPONSE_LENGTH}_ep${TOTAL_EPOCHS}_${TAG}"

CKPT_DIR="${CKPT_DIR:-./checkpoints/grpo-empathy/${EXPERIMENT_NAME}}"
mkdir -p "$CKPT_DIR"

echo "Experiment: $EXPERIMENT_NAME | kl=$KL_GAMMA ent=$ENTROPY_GAMMA diversity_weight=$DIVERSITY_WEIGHT"
echo "RM server: $RM_SERVER_URL | Tagger: $TAGGER_URLS"
echo "Rollout GPU util: $ROLLOUT_GPU_MEMORY_UTILIZATION | UID batch timeout: ${UID_BATCH_TIMEOUT_S}s"

VERL_ARGS=(
    "algorithm.adv_estimator=grpo"
    "algorithm.use_kl_in_reward=False"
    "reward_model.use_reward_loop=False"
    "data.train_files=$TRAIN_FILE"
    "data.val_files=$VAL_FILE"
    "data.train_batch_size=$TRAIN_BATCH_SIZE"
    "data.max_prompt_length=$MAX_PROMPT_LENGTH"
    "data.max_response_length=$MAX_RESPONSE_LENGTH"
    "data.filter_overlong_prompts=True"
    "data.truncation=error"
    "actor_rollout_ref.model.path=$MODEL_PATH"
    "actor_rollout_ref.model.lora_rank=0"
    "actor_rollout_ref.model.enable_gradient_checkpointing=True"
    "actor_rollout_ref.model.use_remove_padding=True"
    "actor_rollout_ref.actor.optim.lr=$LEARNING_RATE"
    "actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE"
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU"
    "actor_rollout_ref.actor.use_kl_loss=True"
    "actor_rollout_ref.actor.kl_loss_coef=$KL_COEF"
    "actor_rollout_ref.actor.kl_loss_type=low_var_kl"
    "actor_rollout_ref.actor.entropy_coeff=0"
    "actor_rollout_ref.actor.fsdp_config.param_offload=False"
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False"
    "actor_rollout_ref.rollout.name=vllm"
    "actor_rollout_ref.rollout.n=$ROLLOUT_N"
    "actor_rollout_ref.rollout.temperature=1.0"
    "actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTILIZATION"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=1"
    "actor_rollout_ref.rollout.load_format=safetensors"
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU"
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU"
    "actor_rollout_ref.ref.fsdp_config.param_offload=False"
    "custom_reward_function.path=$PROJECT_DIR/reward_verl.py"
    "custom_reward_function.name=compute_score"
    "reward_manager.name=batch"
)

VERL_ARGS+=(
    "+custom_reward_function.reward_kwargs.reward_mode=quality_plus_diversity"
    "+custom_reward_function.reward_kwargs.rm_server_url='$RM_SERVER_URL'"
    "+custom_reward_function.reward_kwargs.tactic_tagger_server_url='$TAGGER_URLS'"
    "+custom_reward_function.reward_kwargs.tactic_tagger_prompts_dir='$PROMPTS_DIR'"
    "+custom_reward_function.reward_kwargs.kl_gamma=$KL_GAMMA"
    "+custom_reward_function.reward_kwargs.entropy_gamma=$ENTROPY_GAMMA"
    "+custom_reward_function.reward_kwargs.bigram_gamma=0.0"
    "+custom_reward_function.reward_kwargs.trigram_gamma=0.0"
    "+custom_reward_function.reward_kwargs.diversity_weight=$DIVERSITY_WEIGHT"
    "+custom_reward_function.reward_kwargs.smoothing_alpha=0.1"
    "+custom_reward_function.reward_kwargs.tactic_tagger_temperature=0.1"
    "+custom_reward_function.reward_kwargs.tactic_tagger_max_tokens=64"
    "+custom_reward_function.reward_kwargs.tactic_tagger_max_concurrent=128"
    "+custom_reward_function.reward_kwargs.tactic_tagger_timeout_s=$TACTIC_TAGGER_TIMEOUT_S"
    "+custom_reward_function.reward_kwargs.tactic_tagger_max_retries=$TACTIC_TAGGER_MAX_RETRIES"
    "+custom_reward_function.reward_kwargs.tactic_tagger_retry_base_delay_s=$TACTIC_TAGGER_RETRY_BASE_DELAY_S"
    "+custom_reward_function.reward_kwargs.sample_log_dir='$CKPT_DIR/reward_samples'"
    "trainer.critic_warmup=0"
    "trainer.val_before_train=False"
    "trainer.n_gpus_per_node=1"
    "trainer.nnodes=$NNODES"
    "trainer.total_epochs=$TOTAL_EPOCHS"
    "trainer.save_freq=$SAVE_FREQ"
    "trainer.test_freq=$TEST_FREQ"
    "trainer.default_local_dir=$CKPT_DIR"
    "trainer.max_actor_ckpt_to_keep=null"
    "trainer.logger=$LOGGER"
    "trainer.project_name=grpo-empathy"
    "trainer.experiment_name=$EXPERIMENT_NAME"
)

"$PYTHON" "$PROJECT_DIR/validate_verl_reward_setup.py" "${VERL_ARGS[@]}"

if [ "${VERL_VALIDATE_ONLY:-0}" = "1" ]; then
    echo "Validation-only mode complete."
    exit 0
fi

"$PYTHON" -m verl.trainer.main_ppo "${VERL_ARGS[@]}"

echo "Training complete. Checkpoints at: $CKPT_DIR"