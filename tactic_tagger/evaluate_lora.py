from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from absl import app, flags
import pandas as pd
import os
import json
from tqdm import tqdm
import re
from typing import List, Dict

FLAGS = flags.FLAGS
flags.DEFINE_string("model_id", "meta-llama/Llama-3.1-8B-Instruct", "Model ID for vLLM")
flags.DEFINE_string("tactic_adapter_base_dir", "./lora_adapters/", "Base directory for LoRA adapters")
flags.DEFINE_string("prompts_dir", "prompts/", "Directory containing prompt files")
flags.DEFINE_string("input_path", "", "Path to the input CSV or JSONL file")
flags.DEFINE_string("output_path", "", "Path to the output JSONL file")
flags.DEFINE_string("tactic_name", "", "Specific tactic name to evaluate. If empty, all tactics will be evaluated.")
flags.DEFINE_integer("max_tokens", 20, "Maximum tokens to generate")
flags.DEFINE_float("temperature", 0.1, "Temperature for sampling")
flags.DEFINE_float("top_p", 0.9, "Top P for sampling")
flags.DEFINE_integer("tensor_parallel_size", 1, "Tensor parallel size for vLLM")
flags.DEFINE_integer("pipeline_parallel_size", 3, "Pipeline parallel size for vLLM")
flags.DEFINE_float("gpu_memory_utilization", 0.8, "GPU memory utilization for vLLM")
flags.DEFINE_integer("max_model_len", 8192, "Maximum model length for vLLM")

def capitalize_tactic_name(text):
    return re.sub(r'\b([a-z]+(?:_[a-z]+)*)\b', lambda m: ' '.join(word.capitalize() for word in m.group(1).split('_')), text)

def read_file_to_dataframe(path):
    if path.endswith(".csv"):
        return pd.read_csv(path)
    elif path.endswith(".jsonl"):
        return pd.read_json(path, lines=True)
    else:
        raise ValueError("File format not supported.")

def prepare_batch_messages(input_df: pd.DataFrame, prompts_df: pd.DataFrame) -> List[Dict]:
    """Prepare all chat messages for batch processing."""
    batch_data = []

    for _, row in tqdm(input_df.iterrows(), total=len(input_df), desc="Preparing messages"):
        row_dict = row.to_dict()
        for _, prompt_row in prompts_df.iterrows():
            user_message = prompt_row["Prompt"].format(
                Full_Response=row_dict["whole_response"],
                Sentence=row_dict["sentence"]
            )
            system_message = f'You are a Fair Tagger Assistant, responsible for providing precise, objective tagging based on predefined criteria. Your task is to assess whether a given sentence contains {capitalize_tactic_name(prompt_row["Category"])}, ensuring consistency and adherence to strict tagging guidelines.'

            # Create messages in OpenAI chat format
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]

            batch_data.append({
                'messages': messages,
                'row_data': row_dict,
                'category': prompt_row["Category"],
                'user_message': user_message
            })

    return batch_data

# The 10 empathy tactics used in MINT training and evaluation.
tactic_names = [
    "information", "assistance", "advice", "validation", "emotional_expression",
    "paraphrasing", "self_disclosure", "questioning", "reappraisal", "empowerment",
]

def main(argv):
    os.makedirs(os.path.dirname(FLAGS.output_path), exist_ok=True)

    ### ------ Read and filter out already tagged posts ------
    input_df = read_file_to_dataframe(FLAGS.input_path)
    if os.path.exists(FLAGS.output_path):
        output_df = read_file_to_dataframe(FLAGS.output_path)
        input_df = input_df[~input_df[['postID', 'sentenceID']].apply(tuple, axis=1).isin(output_df[['postID', 'sentenceID']].apply(tuple, axis=1))]

    print(f"Processing {len(input_df)} samples...")

    if len(input_df) == 0:
        print("No samples to process. Exiting.")
        return

    ### ------ Read prompts ------
    my_prompts = []
    for filename in os.listdir(FLAGS.prompts_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(FLAGS.prompts_dir, filename)
            with open(file_path, 'r') as file:
                content = file.read()
            my_prompts.append([filename.replace(".txt", ""), content])
    prompts_df = pd.DataFrame(my_prompts, columns=['Category', 'Prompt'])

    # Filter to specific tactic if provided
    if FLAGS.tactic_name:
        prompts_df = prompts_df[prompts_df["Category"] == FLAGS.tactic_name]
        assert len(prompts_df) == 1, f"Found {len(prompts_df)} prompts for {FLAGS.tactic_name}."
    else:
        assert len(prompts_df) == len(tactic_names), f"Mismatch in number of prompts ({len(prompts_df)}) and defined tactics ({len(tactic_names)})."

    print(f"Loading model: {FLAGS.model_id}")

    llm = LLM(
        model=FLAGS.model_id,
        enable_lora=True,
        max_lora_rank=64,
        tensor_parallel_size=FLAGS.tensor_parallel_size,
        dtype="bfloat16",
        gpu_memory_utilization=FLAGS.gpu_memory_utilization,
        max_model_len=FLAGS.max_model_len,
        pipeline_parallel_size=FLAGS.pipeline_parallel_size,
    )

    tactic_adapter_base_dir = FLAGS.tactic_adapter_base_dir
    tagger_loras = {}
    for idx, tactic in enumerate(tactic_names, 1):
        adapter_path = f"{tactic_adapter_base_dir}Llama-3.1-8B-Instruct-tagger-{tactic}"
        tagger_loras[tactic] = LoRARequest(tactic, idx, adapter_path)

    sampling_params = SamplingParams(temperature=FLAGS.temperature, top_p=FLAGS.top_p, max_tokens=FLAGS.max_tokens)

    all_output_records = []
    RELEVANT_KEYS = [
        "post", "upworker_initial", "whole_response", "subgroup", "context", "dataset",
        "postID", "sentenceID", "sentenceRANK", "sentence", "response_writer", "coder"
    ]

    # If a specific tactic is provided, iterate only for that tactic
    tactics_to_evaluate = [FLAGS.tactic_name] if FLAGS.tactic_name else tactic_names

    for tactic in tactics_to_evaluate:
        current_prompts_df = prompts_df[prompts_df["Category"] == tactic]
        batch_data = prepare_batch_messages(input_df, current_prompts_df)
        print(f"Total messages to process for {tactic}: {len(batch_data)}")

        formatted_prompts = []
        for item in batch_data:
            formatted_prompt = llm.get_tokenizer().apply_chat_template(
                item['messages'],
                tokenize=False,
                add_generation_prompt=True
            )
            formatted_prompts.append(formatted_prompt)

        lora_req = tagger_loras[tactic]
        print(f"--- Sampling from LoRA Adapter [{tactic}] ---")
        outputs = llm.generate(formatted_prompts, sampling_params, lora_request=lora_req)

        for i, output in enumerate(outputs):
            row_dict = batch_data[i]['row_data'].copy()
            response = output.outputs[0].text.strip()
            row_dict[batch_data[i]['category']] = response
            row_dict[batch_data[i]['category'] + "_prompt"] = batch_data[i]['user_message']

            # Filter to only relevant keys
            keys_to_save = RELEVANT_KEYS + [tactic, f"{tactic}_prompt"]
            filtered_record = {k: row_dict.get(k, None) for k in keys_to_save}
            all_output_records.append(filtered_record)

    # Write all results to file
    with open(FLAGS.output_path, 'w') as f:
        for record in all_output_records:
            f.write(json.dumps(record) + "\n")

    print(f"Completed processing {len(all_output_records)} records.")
    print(f"Results saved to: {FLAGS.output_path}")

if __name__ == "__main__":
    app.run(main)