"""
Step 3: Tag each sampled supporter response with empathy tactics using the
TacticTagger (Llama-3.1-8B + LoRA adapters).

Supports multiple tagger servers for parallel throughput. Batches are
distributed round-robin across all reachable servers listed in config.yml
under tagger.server_urls.

Pre-requisites:
    Launch one tagger server per GPU (ports 8100-8103):
        CUDA_VISIBLE_DEVICES=0 python ../training/launch_tactic_tagger_server.py --port 8100
        CUDA_VISIBLE_DEVICES=1 python ../training/launch_tactic_tagger_server.py --port 8101
        ...

Usage:
    python step3_tag_tactics.py --config config.yml --method baseline1_vanilla_Qwen3-1.7B
"""

import os
import sys
import json
import yaml
import concurrent.futures

from absl import app, flags
from nltk.tokenize import sent_tokenize
from termcolor import cprint
from tqdm import tqdm

# Import tactic utilities from the training directory
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "training"
))
from reward_func_tactics_kl_bigram_entropy import (
    TACTIC_NAMES, load_tactic_info, parse_tactic_score,
)

from openai import OpenAI


class TacticTagger:
    """Synchronous HTTP client for the vLLM tactic tagger server.

    The server (launched via training/launch_tactic_tagger_server.py) exposes
    one LoRA adapter per tactic behind an OpenAI-compatible chat completions
    endpoint.  This class queries every adapter for every input sentence and
    returns the set of positively-tagged tactics.
    """

    def __init__(self, prompts_dir, server_url, temperature=0.1,
                 max_tokens=64, server_max_concurrent=128, lazy_init=False):
        self.tactic_info = load_tactic_info(prompts_dir)
        self.server_url = server_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(base_url=server_url, api_key="EMPTY")
        # Verify server is reachable (unless lazy)
        if not lazy_init:
            self.client.models.list()

    def tag_responses(self, texts, full_responses=None):
        """Tag a batch of sentences.

        Args:
            texts: List of individual sentences to evaluate.
            full_responses: Optional list of full response strings (one per
                sentence) providing context. When omitted, each sentence is
                used as its own context for backward compatibility.

        Returns a list (one per input text) of sets of tactic names that were
        positively tagged (score == 1).
        """
        if full_responses is None:
            full_responses = texts
        results = [set() for _ in texts]
        for tactic_name, info in self.tactic_info.items():
            for idx, text in enumerate(texts):
                user_msg = (
                    info["user_template"]
                    .replace("{Full_Response}", full_responses[idx])
                    .replace("{Sentence}", text)
                )
                messages = [
                    {"role": "system", "content": info["system_prompt"]},
                    {"role": "user", "content": user_msg},
                ]
                try:
                    resp = self.client.chat.completions.create(
                        model=tactic_name,
                        messages=messages,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        top_p=0.9,
                    )
                    score = parse_tactic_score(resp.choices[0].message.content)
                    if score == 1:
                        results[idx].add(tactic_name)
                except Exception:
                    pass  # skip on server error
        return results

FLAGS = flags.FLAGS

flags.DEFINE_string("config", "config.yml", "Path to config YAML")
flags.DEFINE_string("method", None, "Method name (key in config.methods)")


def init_taggers(server_urls, prompts_dir, temperature, max_tokens, max_concurrency):
    """Initialize one TacticTagger per reachable server URL."""
    taggers = []
    for url in server_urls:
        try:
            t = TacticTagger(
                prompts_dir=prompts_dir,
                server_url=url,
                temperature=temperature,
                max_tokens=max_tokens,
                server_max_concurrent=max_concurrency,
                lazy_init=False,
            )
            taggers.append(t)
        except Exception as e:
            cprint(f"  Skipping unreachable tagger at {url}: {e}", "yellow", force_color=True)
    return taggers


def tag_entries(entries, taggers, batch_size):
    """Tag entries by distributing batches round-robin across multiple taggers."""
    # Build flat list of (entry_idx, sent_idx, sentence_text, full_response)
    sentence_jobs = []
    entry_sentence_counts = []
    for i, entry in enumerate(entries):
        text = entry["model_response"].strip()
        if not text:
            entry_sentence_counts.append(0)
            continue
        sents = sent_tokenize(text)
        entry_sentence_counts.append(len(sents))
        for j, sent in enumerate(sents):
            sentence_jobs.append((i, j, sent, text))

    cprint(f"  Entries to tag: {sum(1 for c in entry_sentence_counts if c > 0)} (non-empty)", "cyan", force_color=True)
    cprint(f"  Total sentences: {len(sentence_jobs)}", "cyan", force_color=True)
    cprint(f"  Taggers: {len(taggers)}", "cyan", force_color=True)

    # Split into batches
    batches = []
    for start in range(0, len(sentence_jobs), batch_size):
        batches.append(sentence_jobs[start:start + batch_size])

    sentence_results = {}  # (entry_idx, sent_idx) -> sorted list of tactics

    if len(taggers) == 1:
        # Single tagger: sequential
        for batch in tqdm(batches, desc="Tagging"):
            batch_texts = [sent for _, _, sent, _ in batch]
            batch_full = [full for _, _, _, full in batch]
            batch_tactics = taggers[0].tag_responses(batch_texts, full_responses=batch_full)
            for (entry_idx, sent_idx, _, _), tactics in zip(batch, batch_tactics):
                sentence_results[(entry_idx, sent_idx)] = sorted(list(tactics))
    else:
        # Multiple taggers: round-robin with thread pool
        def process_batch(args):
            tagger_idx, batch = args
            batch_texts = [sent for _, _, sent, _ in batch]
            batch_full = [full for _, _, _, full in batch]
            batch_tactics = taggers[tagger_idx].tag_responses(batch_texts, full_responses=batch_full)
            results = []
            for (entry_idx, sent_idx, _, _), tactics in zip(batch, batch_tactics):
                results.append(((entry_idx, sent_idx), sorted(list(tactics))))
            return results

        work_items = [(i % len(taggers), batch) for i, batch in enumerate(batches)]

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(taggers)) as executor:
            futures = {executor.submit(process_batch, item): item for item in work_items}
            with tqdm(total=len(batches), desc="Tagging") as pbar:
                for future in concurrent.futures.as_completed(futures):
                    for key, tactics in future.result():
                        sentence_results[key] = tactics
                    pbar.update(1)

    # Assemble per-entry results
    tagged_entries = []
    for i, entry in enumerate(entries):
        tagged = dict(entry)
        n_sents = entry_sentence_counts[i]
        sentence_tactics = [sentence_results.get((i, j), []) for j in range(n_sents)]
        tactic_counts = {}
        for sent_tactics in sentence_tactics:
            for t in sent_tactics:
                tactic_counts[t] = tactic_counts.get(t, 0) + 1
        tagged["sentence_tactics"] = sentence_tactics
        tagged["tactic_counts"] = tactic_counts
        tagged_entries.append(tagged)

    return tagged_entries


def main(_):
    if not FLAGS.method:
        cprint("Error: --method is required", "red", force_color=True)
        sys.exit(1)

    with open(FLAGS.config) as f:
        cfg = yaml.safe_load(f)

    config_dir = os.path.dirname(os.path.abspath(FLAGS.config))
    method_name = FLAGS.method

    # Validate method and get config
    methods = cfg["methods"]
    mc = None
    if method_name in methods:
        mc = methods[method_name]
    else:
        for mk, mv in methods.items():
            model_label = method_name[len(mk) + 1:] if method_name.startswith(mk + "_") else None
            if model_label and model_label in mv.get("models", {}):
                mc = mv
                break
    if mc is None:
        cprint(f"Error: method '{method_name}' not found in config", "red", force_color=True)
        sys.exit(1)

    if mc.get("skip_tagging"):
        cprint(f"Method '{method_name}' has skip_tagging=true, skipping.", "yellow", force_color=True)
        return

    # Paths
    input_json = os.path.join(config_dir, "outputs", method_name, "conversations.json")
    output_json = os.path.join(config_dir, "outputs", method_name, "conversations_tagged.json")

    # Pre-check: skip if output already exists with expected row count
    if os.path.exists(output_json) and os.path.exists(input_json):
        with open(input_json) as f:
            expected_rows = len(json.load(f))
        with open(output_json) as f:
            existing_rows = len(json.load(f))
        if existing_rows == expected_rows:
            cprint(f"Skipping '{method_name}': {output_json} already exists with {existing_rows} entries", "yellow", force_color=True)
            return

    tc = cfg["tagger"]
    server_urls = tc["server_urls"]
    tagger_temperature = tc.get("temperature", 0.1)
    tagger_max_tokens = tc.get("max_tokens", 64)
    tagger_prompts_dir = tc["prompts_dir"]
    tagger_max_concurrency = tc.get("max_concurrency", 128)
    batch_size = tc["batch_size"]

    # Load sampled entries
    cprint(f"Loading sampled entries from {input_json}...", "cyan", force_color=True)
    with open(input_json) as f:
        entries = json.load(f)
    cprint(f"  Loaded {len(entries)} entries", "green", force_color=True)

    # Initialize taggers (one per reachable server)
    cprint(f"\nInitializing taggers across {len(server_urls)} servers...", "cyan", force_color=True)
    taggers = init_taggers(server_urls, tagger_prompts_dir, tagger_temperature, tagger_max_tokens, tagger_max_concurrency)
    if not taggers:
        cprint("Error: no reachable tagger servers", "red", force_color=True)
        sys.exit(1)
    cprint(f"  {len(taggers)}/{len(server_urls)} servers reachable", "green", force_color=True)

    # Tag entries
    cprint(f"\nTagging model responses...", "cyan", force_color=True)
    tagged_entries = tag_entries(entries, taggers, batch_size)

    # Save tagged JSON
    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(tagged_entries, f, indent=2, ensure_ascii=False)
    cprint(f"\nTagged entries saved to {output_json}", "green", force_color=True)

    # Quick stats
    n_tagged = sum(1 for e in tagged_entries if e["tactic_counts"])
    total_tactics = sum(sum(e["tactic_counts"].values()) for e in tagged_entries)
    total_sents = sum(len(e["sentence_tactics"]) for e in tagged_entries)
    cprint(f"  Entries with tactics: {n_tagged}/{len(tagged_entries)}", "cyan", force_color=True)
    cprint(f"  Total sentences: {total_sents}", "cyan", force_color=True)
    cprint(f"  Total tactic instances: {total_tactics}", "cyan", force_color=True)
    cprint(f"  Mean tactics/turn: {total_tactics / max(len(tagged_entries), 1):.2f}",
           "cyan", force_color=True)

    cprint(f"\nDone!", "green", force_color=True)


if __name__ == "__main__":
    app.run(main)
