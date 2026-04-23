#!/bin/bash
#SBATCH -J mint-grpo
#SBATCH -o logs/mint_%j.log
#SBATCH -e logs/mint_%j.log
#SBATCH -p gh
#SBATCH -N 2
#SBATCH -n 2
#SBATCH --ntasks-per-node=1
#SBATCH -t 48:00:00
#SBATCH -A your-allocation-name
#SBATCH --mail-type=all
#SBATCH --mail-user=your-email@example.com

# Multi-node VERL GRPO on GH200 (1 GPU per node, 120GB each).
# Diversity-only reward (KL + entropy, no RM).
#
# Node layout:
#   Node 0 (server):    Tactic tagger (vLLM + 10 LoRA, port 8100)
#   Node 1 (training):  Ray head + VERL GRPO (1 GPU)
#
# Default: KL + entropy. Override via env vars:
#   KL_GAMMA=1.0 ENTROPY_GAMMA=0.0 sbatch example_submit.sh   # KL only
#   KL_GAMMA=0.0 ENTROPY_GAMMA=1.0 sbatch example_submit.sh   # entropy only

set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPTS_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPTS_DIR/.." && pwd)"
PYTHON="${PYTHON:-python}"
RAY="${RAY:-ray}"
CONDA_PREFIX="${CONDA_PREFIX:-$(dirname "$(dirname "$(which python)")")}"
PROMPTS_DIR="${PROMPTS_DIR:-$PROJECT_DIR/../tactic_tagger/prompts/}"
CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc)")")}"
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

LOG_DIR="$SCRIPTS_DIR/logs"
mkdir -p "$LOG_DIR"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

# Get node hostnames
NODELIST=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
SERVER_NODE="${NODELIST[0]}"
TRAIN_NODE="${NODELIST[1]}"

TAGGER_PORT=8100
TAGGER_URL="http://$SERVER_NODE:$TAGGER_PORT/v1"

echo "========================================"
echo "Job $SLURM_JOB_ID | $(date)"
echo "  Server node (node 0): $SERVER_NODE  [Tagger:$TAGGER_PORT]"
echo "  Training node (node 1): $TRAIN_NODE [Ray + VERL]"
echo "========================================"

echo ""
echo "=== [$(date)] Preflight config validation ==="
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-1.7B}"
export TAGGER_URLS="$TAGGER_URL"
export PROMPTS_DIR="$PROMPTS_DIR"
export KL_GAMMA="${KL_GAMMA:-1.0}"
export BIGRAM_GAMMA="${BIGRAM_GAMMA:-0.0}"
export TRIGRAM_GAMMA="${TRIGRAM_GAMMA:-0.0}"
export ENTROPY_GAMMA="${ENTROPY_GAMMA:-0.0}"
export KL_COEF="${KL_COEF:-0.01}"
VERL_VALIDATE_ONLY=1 bash "$SCRIPTS_DIR/example_run.sh" >/dev/null
echo "Preflight passed."

cleanup() {
    echo ""
    echo "Cleaning up... ($(date))"
    ssh $SSH_OPTS "$SERVER_NODE" \
        "pkill -f 'launch_tactic_tagger_server' 2>/dev/null; pkill -f 'vllm.entrypoints' 2>/dev/null" || true
    ssh $SSH_OPTS "$TRAIN_NODE" \
        "$RAY stop 2>/dev/null" || true
    echo "Cleanup done."
}
trap cleanup EXIT

# =========================================================================
# Node 0: Launch Tagger server
# =========================================================================
echo ""
echo "=== [$(date)] Starting tagger server on $SERVER_NODE ==="

ssh $SSH_OPTS "$SERVER_NODE" "bash -s" <<EOF
export PATH="$CONDA_PREFIX/bin:\$PATH"
export CUDA_HOME="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:\$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:\${LD_LIBRARY_PATH:-}"
export HF_HOME="$HF_HOME"
nohup $PYTHON "$PROJECT_DIR/launch_tactic_tagger_server.py" \
    --port=$TAGGER_PORT \
    --gpu_memory_utilization=0.95 \
    > "$LOG_DIR/tagger_server.log" 2>&1 &
echo "Tagger server launched (PID=\$!)"
EOF

TAGGER_URL="http://$SERVER_NODE:$TAGGER_PORT/v1"
echo "Waiting for tagger server at $TAGGER_URL..."
for i in $(seq 1 1200); do
    if curl -s "${TAGGER_URL}/models" > /dev/null 2>&1; then
        echo "Tagger server ready (waited ${i}s)"
        break
    fi
    if [ $((i % 60)) -eq 0 ]; then
        echo "  Still waiting for tagger... (${i}s elapsed)"
    fi
    sleep 1
done
curl -s "${TAGGER_URL}/models" > /dev/null 2>&1 \
    || { echo "ERROR: Tagger server timeout after 1200s. Check $LOG_DIR/tagger_server.log"; exit 1; }

# =========================================================================
# Node 1: Ray + VERL training
# =========================================================================
echo ""
echo "=== [$(date)] Starting Ray + training on $TRAIN_NODE ==="

ssh $SSH_OPTS "$TRAIN_NODE" "bash -s" <<EOF
export PATH="$CONDA_PREFIX/bin:\$PATH"
export CUDA_HOME="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:\$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:\${LD_LIBRARY_PATH:-}"
export HF_HOME="$HF_HOME"
$RAY start --head --port=6379 --num-gpus=1 --disable-usage-stats
EOF

sleep 10
echo "Ray cluster ready (1 GPU on $TRAIN_NODE)"

# Run training (blocks until complete)
ssh $SSH_OPTS "$TRAIN_NODE" "bash -s" <<EOF
set -euo pipefail
export PATH="$CONDA_PREFIX/bin:\$PATH"
export CUDA_HOME="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:\$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:\${LD_LIBRARY_PATH:-}"
export HF_HOME="$HF_HOME"
export RAY_ADDRESS="localhost:6379"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

    export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-1.7B}"
    export TAGGER_URLS="$TAGGER_URL"
    export PROMPTS_DIR="$PROMPTS_DIR"
    export KL_GAMMA="${KL_GAMMA:-1.0}"
    export BIGRAM_GAMMA="${BIGRAM_GAMMA:-0.0}"
    export TRIGRAM_GAMMA="${TRIGRAM_GAMMA:-0.0}"
    export ENTROPY_GAMMA="${ENTROPY_GAMMA:-0.0}"
    export KL_COEF="${KL_COEF:-0.01}"

bash "$SCRIPTS_DIR/example_run.sh" 2>&1
EOF

echo ""
echo "=== Training complete! $(date) ==="
