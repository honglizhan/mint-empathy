"""
Step 2: Run lend-an-ear empathy evaluation on sampled conversations.

Reads the flat sampled entries, reconstructs conversation format for
annotate_turns.py, and runs the empathy eval (6 dimensions).

Usage:
    python step2_empathy_eval.py --config config.yml --method baseline1_vanilla_Qwen3-1.7B
    python step2_empathy_eval.py --config config.yml --method gold
"""

import os
import sys
import json
import yaml
import subprocess
from collections import defaultdict

from absl import app, flags
from termcolor import cprint

FLAGS = flags.FLAGS

flags.DEFINE_string("config", "config.yml", "Path to config YAML")
flags.DEFINE_string("method", None, "Method name (key in config.methods)")

ROLE_MAP_REVERSE = {"user": "seeker", "assistant": "supporter"}


def entries_to_conversations(entries):
    """Reconstruct conversation format from flat entries for annotate_turns.py.

    Each entry has conversation_history (the context) and model_response.
    We group by conversation_id and rebuild the full conversation with
    model responses replacing supporter turns.
    """
    grouped = defaultdict(list)
    for entry in entries:
        grouped[entry["conversation_id"]].append(entry)

    conversations = []
    for conv_id, conv_entries in grouped.items():
        # Use the longest conversation_history (last turn) to get full context
        last_entry = max(conv_entries, key=lambda e: len(e["conversation_history"]))
        turns = []
        for msg in last_entry["conversation_history"]:
            role = ROLE_MAP_REVERSE.get(msg["role"], msg["role"])
            turns.append({"role": role, "content": msg["content"]})
        # Add the final model response
        turns.append({"role": "supporter", "content": last_entry["model_response"]})

        # Replace earlier supporter turns with model responses
        for entry in conv_entries:
            history_len = len(entry["conversation_history"])
            if history_len < len(turns):
                turns[history_len] = {"role": "supporter", "content": entry["model_response"]}

        conversations.append({
            "conversation_id": conv_id,
            "source": conv_entries[0].get("source", ""),
            "model": conv_entries[0].get("model", ""),
            "conversation": turns,
        })

    return conversations


def main(_):
    if not FLAGS.method:
        cprint("Error: --method is required", "red", force_color=True)
        sys.exit(1)

    with open(FLAGS.config) as f:
        cfg = yaml.safe_load(f)

    config_dir = os.path.dirname(os.path.abspath(FLAGS.config))
    method_name = FLAGS.method

    # Validate method exists (flat key like 'baseline1_vanilla_Qwen3-1.7B' or 'gold')
    methods = cfg["methods"]
    found = method_name in methods  # direct match (e.g. 'gold')
    if not found:
        for mk, mv in methods.items():
            model_label = method_name[len(mk) + 1:] if method_name.startswith(mk + "_") else None
            if model_label and model_label in mv.get("models", {}):
                found = True
                break
    if not found:
        cprint(f"Error: method '{method_name}' not found in config", "red", force_color=True)
        sys.exit(1)

    # Paths: use outputs/{method}/conversations.json
    sampled_json = os.path.join(config_dir, "outputs", method_name, "conversations.json")
    eval_dir = os.path.join(config_dir, "eval_outputs", method_name)

    ev = cfg["empathy_eval"]

    # Load flat entries and reconstruct conversation format
    cprint(f"Loading sampled entries from {sampled_json}...", "cyan", force_color=True)
    with open(sampled_json) as f:
        entries = json.load(f)
    cprint(f"  Loaded {len(entries)} entries", "green", force_color=True)

    conversations = entries_to_conversations(entries)
    cprint(f"  Reconstructed {len(conversations)} conversations", "green", force_color=True)

    # Write temp conversations JSON for annotate_turns.py
    tmp_convs_path = os.path.join(config_dir, f".tmp_eval_conversations_{method_name}.json")
    with open(tmp_convs_path, "w") as f:
        json.dump(conversations, f, indent=2, ensure_ascii=False)

    # Build the config that annotate_turns.py expects
    annotate_config = {
        "conversations_json": "dummy",
        "sub_components_json": ev["sub_components_json"],
        "framework_path": ev["framework_path"],
        "template_path": ev["template_path"],
        "output_dir": "dummy",
        "dimensions": ev["dimensions"],
        "models": [{
            "name": ev["model_name"],
            "model": ev["model_id"],
            "temperature": ev["temperature"],
            "max_tokens": ev["max_tokens"],
            "max_concurrency": ev["max_concurrency"],
        }],
        "vllm_base_urls": ev.get("vllm_base_urls", [ev.get("vllm_base_url", "http://localhost:8000/v1")]),
    }

    tmp_config_path = os.path.join(config_dir, f".tmp_eval_config_{method_name}.yaml")
    with open(tmp_config_path, "w") as f:
        yaml.dump(annotate_config, f, default_flow_style=False)

    lend_an_ear_dir = ev["lend_an_ear_dir"]
    annotate_script = os.path.join(lend_an_ear_dir, "annotate_turns.py")

    cprint(f"\nRunning empathy eval for method: {method_name}", "cyan", force_color=True)
    cprint(f"  Judge model: {ev['model_name']}", "cyan", force_color=True)
    cprint(f"  Conversations: {tmp_convs_path}", "cyan", force_color=True)
    cprint(f"  Output dir: {eval_dir}", "cyan", force_color=True)

    cmd = [
        sys.executable, annotate_script,
        "--config", tmp_config_path,
        "--run", ev["model_name"],
        "--conversations_json", tmp_convs_path,
        "--output_dir", eval_dir,
    ]

    result = subprocess.run(cmd)

    # Cleanup temp files
    for tmp in [tmp_config_path, tmp_convs_path]:
        if os.path.exists(tmp):
            os.remove(tmp)

    if result.returncode != 0:
        cprint(f"Empathy eval failed with exit code {result.returncode}", "red", force_color=True)
        sys.exit(result.returncode)

    cprint(f"\nEmpathy eval complete: {eval_dir}", "green", force_color=True)


if __name__ == "__main__":
    app.run(main)
