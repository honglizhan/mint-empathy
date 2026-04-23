import os
import json
import re
import pandas as pd
import nltk
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from termcolor import cprint
from nltk.tokenize import sent_tokenize
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from dataclasses import dataclass, field
from transformers import HfArgumentParser, AutoTokenizer

# Download nltk punkt if not available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# --- Arguments ---
@dataclass
class ModelArguments:
    model_id: str = field(default="meta-llama/Llama-3.1-8B-Instruct", metadata={"help": "Base model ID"})
    adapter_base_dir: str = field(default="./lora_adapters/", metadata={"help": "Directory containing LoRA adapters"})
    prompts_dir: str = field(default="../tactic_tagger/prompts/", metadata={"help": "Directory containing tactic instruction files"})
    temperature: float = field(default=0.1, metadata={"help": "Sampling temperature"})
    max_tokens: int = field(default=2048, metadata={"help": "Maximum tokens to generate"})
    top_p: float = field(default=0.9, metadata={"help": "Top P for sampling"})
    tensor_parallel_size: int = field(default=1, metadata={"help": "Number of tensor parallel GPUs"})
    pipeline_parallel_size: int = field(default=1, metadata={"help": "Number of pipeline parallel GPUs"})
    gpu_memory_utilization: float = field(default=0.8, metadata={"help": "GPU memory utilization"})
    max_model_len: int = field(default=8192*2, metadata={"help": "Maximum model length"})

@dataclass
class DataArguments:
    input_file: str = field(default="../data/training/conversations_322.json", metadata={"help": "Path to the input JSON file"})
    output_file: str = field(default="../data/training/conversations_322_tagged.json", metadata={"help": "Path to the output JSON file"})

# The 10 empathy tactics used in MINT training and evaluation.
TACTIC_NAMES = [
    "information", "assistance", "advice", "validation", "emotional_expression",
    "paraphrasing", "self_disclosure", "questioning", "reappraisal", "empowerment"
]

def load_tactic_info(prompts_dir):
    tactic_info = {}
    for tactic in TACTIC_NAMES:
        prompt_path = os.path.join(prompts_dir, f"{tactic}.txt")
        if not os.path.exists(prompt_path):
            cprint(f"Warning: Prompt file not found for {tactic} at {prompt_path}", "red")
            continue

        with open(prompt_path, 'r') as f:
            content = f.read()

        # Extract the capitalized tactic name used in the instructions
        match = re.search(r'contains "([^"]+)"', content)
        cap_name = match.group(1) if match else tactic.replace('_', ' ').title()

        system_prompt = f"You are a Fair Tagger Assistant, responsible for providing precise, objective tagging based on predefined criteria. Your task is to assess whether a given sentence contains \"{cap_name}\", ensuring consistency and adherence to strict tagging guidelines."

        tactic_info[tactic] = {
            "system_prompt": system_prompt,
            "user_template": content,
        }
    return tactic_info

def parse_score(output_text: str) -> int:
    match = re.search(r'<score>(\d+)</score>', output_text)
    if match:
        return int(match.group(1))
    return 0

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()

    # 1. Load tactic info
    tactic_info = load_tactic_info(model_args.prompts_dir)
    cprint(f"Loaded info for {len(tactic_info)} tactics", "green")

    # 2. Load LLM with LoRA support
    # Using settings from _load_tactic_loras.py reference
    llm = LLM(
        model                   = model_args.model_id,
        enable_lora             = True,
        max_lora_rank           = 64,
        dtype                   = "bfloat16",
        gpu_memory_utilization  = model_args.gpu_memory_utilization,
        max_model_len           = model_args.max_model_len,
        pipeline_parallel_size  = model_args.pipeline_parallel_size,
        tensor_parallel_size    = model_args.tensor_parallel_size,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_id)

    tagger_loras = {}
    for idx, tactic in enumerate(TACTIC_NAMES, 1):
        adapter_path = os.path.join(model_args.adapter_base_dir, f"Llama-3.1-8B-Instruct-tagger-{tactic}")
        if os.path.exists(adapter_path):
            tagger_loras[tactic] = LoRARequest(tactic, idx, adapter_path)
        else:
            cprint(f"Warning: Adapter not found for {tactic} at {adapter_path}", "red")

    cprint(f"Loaded {len(tagger_loras)} LoRA adapters", "green")

    sampling_params = SamplingParams(
        temperature = model_args.temperature,
        max_tokens  = model_args.max_tokens,
        top_p       = model_args.top_p,
        top_k       = 50,
        n           = 1,
    )

    # 3. Read input data
    with open(data_args.input_file, 'r') as f:
        data = json.load(f)
    cprint(f"Read {len(data)} conversations from {data_args.input_file}", "cyan")

    # 4. Extract all sentences that need tagging
    sentences_to_tag = []
    for conv_idx, conv_entry in enumerate(data):
        for msg_idx, message in enumerate(conv_entry["conversation"]):
            if message["role"] == "supporter":
                full_response = message["content"]
                sents = sent_tokenize(full_response)
                for sent_idx, sent in enumerate(sents):
                    sentences_to_tag.append({
                        "conv_idx": conv_idx,
                        "msg_idx": msg_idx,
                        "sent_idx": sent_idx,
                        "full_response": full_response,
                        "sentence": sent
                    })

    cprint(f"Total sentences to tag: {len(sentences_to_tag)}", "cyan")

    # 5. Tag tactic by tactic
    # (conv_idx, msg_idx, sent_idx) -> list of tactics
    tagging_results = {}

    for tactic, lora_req in tagger_loras.items():
        cprint(f"Tagging tactic: {tactic}...", "yellow")
        info = tactic_info[tactic]

        prompts = []
        for item in sentences_to_tag:
            user_msg = info["user_template"].replace("{Full_Response}", item["full_response"]).replace("{Sentence}", item["sentence"])

            messages = [
                {"role": "system", "content": info["system_prompt"]},
                {"role": "user", "content": user_msg}
            ]
            prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt_str)

        # Batch generate for this tactic
        outputs = llm.generate(prompts, sampling_params=sampling_params, lora_request=lora_req)

        for item, output in zip(sentences_to_tag, outputs):
            score = parse_score(output.outputs[0].text)
            if score == 1:
                key = (item["conv_idx"], item["msg_idx"], item["sent_idx"])
                if key not in tagging_results:
                    tagging_results[key] = []
                tagging_results[key].append(tactic)

    # 6. Reconstruct the data with tags
    for conv_idx, conv_entry in enumerate(data):
        for msg_idx, message in enumerate(conv_entry["conversation"]):
            if message["role"] == "supporter":
                full_response = message["content"]
                sents = sent_tokenize(full_response)

                sentence_tactics = []
                all_used_tactics = set()

                for sent_idx, sent in enumerate(sents):
                    key = (conv_idx, msg_idx, sent_idx)
                    sent_tactics = tagging_results.get(key, [])
                    sentence_tactics.append({
                        "sentence": sent,
                        "tactics": sent_tactics
                    })
                    all_used_tactics.update(sent_tactics)

                message["sentence_tactics"] = sentence_tactics
                message["all_tactics"] = list(all_used_tactics)

    # 7. Save output
    os.makedirs(os.path.dirname(data_args.output_file), exist_ok=True)
    with open(data_args.output_file, 'w') as f:
        json.dump(data, f, indent=2)
    cprint(f"Saved tagged data to {data_args.output_file}", "green")

if __name__ == "__main__":
    main()