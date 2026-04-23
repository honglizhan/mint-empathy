#!/bin/bash
# VERL GRPO training launcher for GH200 (1 GPU, 120GB).
# Diversity-only reward: KL + entropy (no RM).
# Called from example_submit.sh with env vars set.

set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPTS_DIR/.." && pwd)"
PYTHON="${PYTHON:-python}"

export RAY_ADDRESS="${RAY_ADDRESS:-localhost:6379}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-1.7B}"
TAGGER_URLS="${TAGGER_URLS:-http://localhost:8100/v1}"
PROMPTS_DIR="${PROMPTS_DIR:-$PROJECT_DIR/../tactic_tagger/prompts/}"
KL_GAMMA="${KL_GAMMA:-1.0}"
BIGRAM_GAMMA="${BIGRAM_GAMMA:-0.0}"
TRIGRAM_GAMMA="${TRIGRAM_GAMMA:-0.0}"
ENTROPY_GAMMA="${ENTROPY_GAMMA:-0.0}"
KL_COEF="${KL_COEF:-0.01}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.4}"
NNODES="${NNODES:-1}"
UID_BATCH_TIMEOUT_S="${UID_BATCH_TIMEOUT_S:-1800}"
TACTIC_TAGGER_TIMEOUT_S="${TACTIC_TAGGER_TIMEOUT_S:-20}"
TACTIC_TAGGER_MAX_RETRIES="${TACTIC_TAGGER_MAX_RETRIES:-6}"
TACTIC_TAGGER_RETRY_BASE_DELAY_S="${TACTIC_TAGGER_RETRY_BASE_DELAY_S:-1.0}"

# Build experiment name from reward weights
MODEL_SHORT=$(basename "$MODEL_PATH")
TAG="rw"
[ "$KL_GAMMA" != "0.0" ] && [ "$KL_GAMMA" != "0" ] && TAG="${TAG}-kl${KL_GAMMA}"
[ "$BIGRAM_GAMMA" != "0.0" ] && [ "$BIGRAM_GAMMA" != "0" ] && TAG="${TAG}-bi${BIGRAM_GAMMA}"
[ "$TRIGRAM_GAMMA" != "0.0" ] && [ "$TRIGRAM_GAMMA" != "0" ] && TAG="${TAG}-tri${TRIGRAM_GAMMA}"
[ "$ENTROPY_GAMMA" != "0.0" ] && [ "$ENTROPY_GAMMA" != "0" ] && TAG="${TAG}-ent${ENTROPY_GAMMA}"
EXPERIMENT_NAME="${MODEL_SHORT}_bs24_n8_lr1e-6_kl${KL_COEF}_rlen2048_ep3_${TAG}"

CKPT_DIR="${CKPT_DIR:-./checkpoints/grpo-empathy/${EXPERIMENT_NAME}}"
mkdir -p "$CKPT_DIR"

echo "Experiment: $EXPERIMENT_NAME | kl=$KL_GAMMA bi=$BIGRAM_GAMMA tri=$TRIGRAM_GAMMA ent=$ENTROPY_GAMMA"
echo "Rollout GPU util: $ROLLOUT_GPU_MEMORY_UTILIZATION | UID batch timeout: ${UID_BATCH_TIMEOUT_S}s"

VERL_ARGS=(
    "algorithm.adv_estimator=grpo"
    "algorithm.use_kl_in_reward=False"
    "data.train_files=$PROJECT_DIR/data/train.parquet"
    "data.val_files=$PROJECT_DIR/data/train.parquet"
    "data.train_batch_size=24"
    "data.max_prompt_length=8192"
    "data.max_response_length=2048"
    "data.filter_overlong_prompts=True"
    "data.truncation=error"
    "actor_rollout_ref.model.path=$MODEL_PATH"
    "actor_rollout_ref.model.lora_rank=0"
    "actor_rollout_ref.model.enable_gradient_checkpointing=True"
    "actor_rollout_ref.model.use_remove_padding=True"
    "actor_rollout_ref.actor.optim.lr=1e-6"
    "actor_rollout_ref.actor.ppo_mini_batch_size=24"
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4"
    "actor_rollout_ref.actor.use_kl_loss=True"
    "actor_rollout_ref.actor.kl_loss_coef=$KL_COEF"
    "actor_rollout_ref.actor.kl_loss_type=low_var_kl"
    "actor_rollout_ref.actor.entropy_coeff=0"
    "actor_rollout_ref.actor.fsdp_config.param_offload=False"
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False"
    "actor_rollout_ref.rollout.name=vllm"
    "actor_rollout_ref.rollout.n=8"
    "actor_rollout_ref.rollout.temperature=1.0"
    "actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTILIZATION"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=1"
    "actor_rollout_ref.rollout.load_format=safetensors"
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8"
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8"
    "actor_rollout_ref.ref.fsdp_config.param_offload=False"
    "reward.num_workers=1"
    "reward.reward_manager.source=importlib"
    "reward.reward_manager.name=UidBatchRewardManager"
    "reward.reward_manager.module.path=$PROJECT_DIR/reward_manager_compat.py"
)

VERL_ARGS+=(
    "reward.custom_reward_function.path=$PROJECT_DIR/reward_verl.py"
    "reward.custom_reward_function.name=compute_score"
    "+reward.uid_batch_timeout_s=$UID_BATCH_TIMEOUT_S"
    "+reward.custom_reward_function.reward_kwargs.reward_mode=diversity_only"
    "+reward.custom_reward_function.reward_kwargs.tactic_tagger_server_url='$TAGGER_URLS'"
    "+reward.custom_reward_function.reward_kwargs.tactic_tagger_prompts_dir='$PROMPTS_DIR'"
    "+reward.custom_reward_function.reward_kwargs.kl_gamma=$KL_GAMMA"
    "+reward.custom_reward_function.reward_kwargs.bigram_gamma=$BIGRAM_GAMMA"
    "+reward.custom_reward_function.reward_kwargs.trigram_gamma=$TRIGRAM_GAMMA"
    "+reward.custom_reward_function.reward_kwargs.entropy_gamma=$ENTROPY_GAMMA"
    "+reward.custom_reward_function.reward_kwargs.smoothing_alpha=0.1"
    "+reward.custom_reward_function.reward_kwargs.tactic_tagger_temperature=0.1"
    "+reward.custom_reward_function.reward_kwargs.tactic_tagger_max_tokens=64"
    "+reward.custom_reward_function.reward_kwargs.tactic_tagger_max_concurrent=128"
    "+reward.custom_reward_function.reward_kwargs.tactic_tagger_timeout_s=$TACTIC_TAGGER_TIMEOUT_S"
    "+reward.custom_reward_function.reward_kwargs.tactic_tagger_max_retries=$TACTIC_TAGGER_MAX_RETRIES"
    "+reward.custom_reward_function.reward_kwargs.tactic_tagger_retry_base_delay_s=$TACTIC_TAGGER_RETRY_BASE_DELAY_S"
    "+reward.custom_reward_function.reward_kwargs.sample_log_dir='$CKPT_DIR/reward_samples'"
    "trainer.critic_warmup=0"
    "trainer.val_before_train=False"
    "trainer.n_gpus_per_node=1"
    "trainer.nnodes=$NNODES"
    "trainer.total_epochs=3"
    "trainer.save_freq=50"
    "trainer.test_freq=-1"
    "trainer.default_local_dir=$CKPT_DIR"
    "trainer.max_actor_ckpt_to_keep=null"
    'trainer.logger=["console","wandb"]'
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
