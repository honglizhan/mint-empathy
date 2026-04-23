#!/bin/bash
#SBATCH -J tagger-training
#SBATCH -o tagger-training.%j.out
#SBATCH -e tagger-training.%j.err
#SBATCH -p gpu-a100-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 24:00:00
#SBATCH --mail-user=your-email@example.com
#SBATCH --mail-type=all
#SBATCH -A your-allocation-name

# Load required modules
# module load cuda/12.1
# module load python3/3.9.7

# Activate your virtual environment if needed
# source /path/to/your/venv/bin/activate

# This script launches the sequence classification fine-tuning script
# using Hugging Face Accelerate for distributed training.

for tactic_file in ../data/tagger_annotations/train/*.jsonl; do
    # Extract tactic name from filename (remove path and extension)
    tactic=$(basename "$tactic_file" .jsonl)

    echo -e "\033[1;34mTraining model for tactic: $tactic\033[0m"

    python train_lora.py \
        --model_id="meta-llama/Llama-3.1-8B-Instruct" \
        --train_data_path="../data/tagger_annotations/train/${tactic}.jsonl" \
        --val_data_path="../data/tagger_annotations/val/${tactic}.jsonl" \
        --finetuned_model_path="./trained-tactic-tagger-models-lora_adapters/Llama-3.1-8B-Instruct-tagger-${tactic}" \
        --ckpt_output_path="./ckpt/Llama-3.1-8B-Instruct-tagger-${tactic}" \
        --training_epochs=3.0 \
        --batch_size=24

    echo -e "\033[1;32mCompleted training for tactic: $tactic\033[0m"
    echo -e "\033[1;33m----------------------------------------\033[0m"
done