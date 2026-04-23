#!/bin/bash
# EVAL pipeline runner (1-GPU workflow).
#
# Each step uses the single GPU differently, so steps run sequentially
# with server swaps between them.
#
# Step 0: Preprocess gold (CPU only, no GPU)
# Step 1: Sample model responses via vLLM offline (1 GPU, no server needed)
# Step 2: Empathy eval via gpt-oss-120b server (1 GPU, vLLM serve on port 8000)
# Step 3: Tag tactics via tactic tagger LoRA server (1 GPU, vLLM serve on port 8100)
# Step 4: Analysis + stickiness (CPU only, no GPU)
#
# Usage:
#   bash run.sh                                            # all methods, all steps
#   bash run.sh --method baseline1_vanilla_Qwen3-1.7B        # single method, all steps
#   bash run.sh --step 1                                   # single step, all methods
#   bash run.sh --step 0                                   # preprocess gold only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${SCRIPT_DIR}/config.yml"

# Build ALL_METHODS from config.yml using Python (flat keys)
ALL_METHODS=($(python3 -c "
import yaml
with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)
for mk, mv in cfg['methods'].items():
    models = mv.get('models', {})
    if models:
        for ml in models:
            print(f'{mk}_{ml}')
    elif mk != 'gold':
        print(mk)
"))

# Parse args
METHOD=""
STEP=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --method) METHOD="$2"; shift 2 ;;
        --step)   STEP="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

run_step() {
    local step=$1
    local method=$2

    case $step in
        0)
            echo "=== Step 0: Preprocess gold ==="
            python3 "${SCRIPT_DIR}/step0_preprocess_gold.py" --config "$CONFIG"
            ;;
        1)
            echo "=== Step 1: Sample responses for ${method} ==="
            python3 "${SCRIPT_DIR}/step1_sample.py" --config "$CONFIG" --method "$method"
            ;;
        2)
            echo "=== Step 2: Empathy eval for ${method} ==="
            python3 "${SCRIPT_DIR}/step2_empathy_eval.py" --config "$CONFIG" --method "$method"
            ;;
        3)
            echo "=== Step 3: Tag tactics for ${method} ==="
            python3 "${SCRIPT_DIR}/step3_tag_tactics.py" --config "$CONFIG" --method "$method"
            ;;
        4)
            echo "=== Step 4: Analysis (all methods) ==="
            python3 "${SCRIPT_DIR}/step4_analyze.py" --config "$CONFIG"
            ;;
        *)
            echo "Unknown step: $step"; exit 1 ;;
    esac
}

# Single step mode
if [[ -n "$STEP" ]]; then
    if [[ "$STEP" == "0" || "$STEP" == "4" ]]; then
        # Steps 0 and 4 don't need a method
        run_step "$STEP" ""
    elif [[ -n "$METHOD" ]]; then
        run_step "$STEP" "$METHOD"
    else
        # Run this step for all methods
        for m in "${ALL_METHODS[@]}"; do
            run_step "$STEP" "$m"
        done
    fi
    exit 0
fi

# Full pipeline mode
# Step 0: Preprocess gold (one-time)
run_step 0 ""

# Steps 1-3 per method
if [[ -n "$METHOD" ]]; then
    METHODS=("$METHOD")
else
    METHODS=("${ALL_METHODS[@]}")
fi

for m in "${METHODS[@]}"; do
    echo ""
    echo "========================================"
    echo "  Processing method: ${m}"
    echo "========================================"
    run_step 1 "$m"
    run_step 2 "$m"    # Empathy eval (LLM judge)
    run_step 3 "$m"    # Tactic tagging
done

# Step 2 for gold (empathy eval only, no sampling/tagging)
echo ""
echo "=== Step 2: Empathy eval for gold ==="
python3 "${SCRIPT_DIR}/step2_empathy_eval.py" --config "$CONFIG" --method gold

# Step 4: Analysis
echo ""
run_step 4 ""

echo ""
echo "All done!"
