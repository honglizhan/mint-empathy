# MINT: Multi-turn Inter-tactic Novelty Training

Code and data for the paper **"Breaking the Empathy Script: Discourse Diversity in Multi-Turn Empathic Dialogue"**.

MINT is a reinforcement learning framework that optimizes discourse move diversity across multi-turn empathic dialogue. It combines an empathy quality reward with a cross-turn tactic novelty signal using GRPO, training models to vary their empathy tactics across conversation turns rather than locking into repetitive patterns.

## Repository Structure

```
mint-empathy/
├── data/                          Data
│   ├── training/                  322 multi-turn emotional support conversations
│   ├── evaluation/                Lend-an-Ear gold evaluation data
│   └── tagger_annotations/        Human-annotated tactic labels for tagger training
│
├── tactic_tagger/                 Train 10 per-tactic LoRA adapters
│   ├── train_lora.py              LoRA fine-tuning on Llama-3.1-8B-Instruct
│   ├── prompts/                   10 tactic definition templates
│   ├── evaluate_lora.py           Evaluation with vLLM
│   └── tag_tactics.py             Tag training conversations with tactics
│
├── training/                      MINT: GRPO training with VERL
│   ├── reward_verl.py             Unified reward function (5 modes)
│   ├── reward_func_tactics_kl_bigram_entropy.py
│   ├── prepare_data_verl.py       Convert data to VERL parquet format
│   ├── launch_tactic_tagger_server.py
│   ├── scripts/                   Example SLURM launch scripts
│   └── tests/                     Reward function unit tests
│
├── evaluation/                    Evaluation pipeline
│   ├── config.yml                 Central configuration
│   ├── run.sh                     Orchestrator (steps 0-4)
│   ├── step0_preprocess_gold.py   Extract gold turns
│   ├── step1_sample.py            Generate model responses (vLLM)
│   ├── step2_empathy_eval.py      LLM-judge empathy scoring
│   ├── step3_tag_tactics.py       Tag responses with tactic adapters
│   ├── step4_analyze.py           Aggregate metrics + bootstrap significance
│   ├── outputs/                   Pre-computed model outputs
│   └── eval_outputs/              Pre-computed empathy ratings
│
└── analysis/                      Paper figures + tactic diversity analysis
    ├── analyze_tactic_diversity.py
    ├── pareto_front.py
    ├── plot_stickiness_and_prevalence.py
    ├── surface_vs_tactic_repetition.py
    └── generate_appendix_h.py
```

## Setup

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Pipeline Overview

### 1. Train Tactic Taggers

Train 10 binary LoRA adapters (one per empathy tactic) on Llama-3.1-8B-Instruct:

```bash
cd tactic_tagger
bash run_training.sh
```

### 2. Tag Training Data

Tag 322 conversations with tactic labels:

```bash
cd tactic_tagger
python tag_tactics.py
```

### 3. MINT Training (GRPO with VERL)

Prepare data and launch training:

```bash
cd training

# Start tactic tagger vLLM server (on a dedicated GPU)
python launch_tactic_tagger_server.py

# Convert data to VERL format
python prepare_data_verl.py --include_tactics

# Launch GRPO training (see scripts/ for SLURM examples)
bash scripts/example_run.sh
```

The reward function (`reward_verl.py`) supports 5 modes:
- `quality_only`: PsychoCounsel empathy quality reward only
- `diversity_only`: Tactic diversity reward only
- `quality_plus_diversity`: Q + D_KL (our best variant)
- `quality_times_diversity`: Q * diversity
- `quality_x_r1_zero_div`: Q with token-level entropy bonus

### 4. Evaluation

Run the full evaluation pipeline:

```bash
cd evaluation

# Edit config.yml with your server URLs and model paths

# Run all steps for a specific method
bash run.sh --method baseline1_vanilla_Qwen3-1.7B

# Or run individual steps
python step0_preprocess_gold.py --config config.yml
python step1_sample.py --config config.yml --method baseline1_vanilla_Qwen3-1.7B
python step2_empathy_eval.py --config config.yml --method baseline1_vanilla_Qwen3-1.7B
python step3_tag_tactics.py --config config.yml --method baseline1_vanilla_Qwen3-1.7B
python step4_analyze.py --config config.yml
```

Pre-computed outputs for all methods are included in `outputs/` and `eval_outputs/`.

### 5. Generate Paper Figures

```bash
cd analysis
python pareto_front.py
python plot_stickiness_and_prevalence.py
python surface_vs_tactic_repetition.py
```

### 6. Verify Paper Numbers

This repo includes a lightweight audit script for the main paper-facing numbers
that are directly reproducible from the public artifacts.

```bash
python analysis/verify_paper_numbers.py --config analysis/paper_numbers.yml
```

The check covers:
- dataset counts
- Table 1 non-word metrics
- tactic prevalence caption numbers
- prose claims derived from Table 1

The Table 1 word-count column is intentionally excluded because it is tracked as
a known snapshot difference between the paper source and the current public repo
outputs.

## Data

- **Training conversations** (`data/training/`): 322 multi-turn emotional support conversations from WildChat and SENSE-7, with sentence-level tactic tags.
- **Tagger annotations** (`data/tagger_annotations/`): Human-annotated sentences across 10 empathy tactics, split into train/val/test.
- **Evaluation data** (`data/evaluation/`): 50 Lend-an-Ear gold conversations with tactic annotations.

## Empathy Tactics

The 10 tactic categories used in MINT (5 rare tactics excluded: contextualizing, gratitude, solidarity, spirituality, terms of endearment):

| Tactic | Description |
|--------|-------------|
| Advice | Actionable solutions or coping strategies |
| Assistance | Offering personal help |
| Emotional Expression | Sharing own feelings or reactions |
| Empowerment | Positive statements about the user's character |
| Information | Factual statements, resources, or data |
| Paraphrasing | Restating the user's feelings or experiences |
| Questioning | Questions to understand the user better |
| Reappraisal | Prompting cognitive reappraisal |
| Self-Disclosure | Sharing personal experiences |
| Validation | Reassuring or normalizing feelings |

## Hardware Requirements

- **Tagger training**: 1x A100/H100 GPU
- **MINT training**: 4x H200 96GB GPUs (or equivalent)
- **Evaluation**: 1+ GPU for vLLM inference

## Citation

```bibtex
@article{zhan2026discourse,
  title   = {Discourse Diversity in Multi-Turn Empathic Dialogue},
  author  = {Zhan, Hongli and Gueorguieva, Emma S. and Hernandez, Javier and Suh, Jina and Ong, Desmond C. and Li, Junyi Jessy},
  year    = {2026},
  note    = {Under review}
}
```

## Project Page

See `project-page/` for the project website, or visit the [online version](https://honglizhan.github.io/mint-empathy/).

## License

This project is released under the [MIT License](LICENSE). Please cite the paper if you use this code or data.
