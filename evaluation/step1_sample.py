"""
Step 1: Sample model responses at every supporter turn in the 50 Lend-an-Ear
test conversations, using independent (non-rolling) sampling.

Supports four prompt modes:
  - simple:                Static minimal system prompt
  - tactic_aware:          Static prompt listing 10 tactics
  - verl_history:          Dynamic per-turn prompt with tactic history counts
                           (uses build_system_prompt from prepare_data_verl.py)
  - verbalized_sampling:   VS wrapper around any base prompt mode (generates k
                           candidates with probabilities, samples one)

For verl_history mode, gold tactic history is extracted from turn_level_tactics.csv
and accumulated per-turn, the same way VERL training data was prepared.

Uses vLLM offline inference (no separate server needed).

Usage:
    python step1_sample.py --config config.yml --method baseline1_vanilla_Qwen3-1.7B
    python step1_sample.py --config config.yml --method baseline5_vs_tactic_Qwen3-4B
"""

import csv
import html
import json
import math
import os
import random
import re
import sys

import yaml
from absl import app, flags
from termcolor import cprint

# Import shared prompt builder and tactic names from the training directory
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "training"
))
from prepare_data_verl import build_tactic_system_prompt as build_verl_system_prompt
from reward_func_tactics_kl_bigram_entropy import TACTIC_NAMES as VERL_TACTICS

FLAGS = flags.FLAGS

flags.DEFINE_string("config", "config.yml", "Path to config YAML")
flags.DEFINE_string("method", None, "Method name, e.g. 'baseline1_vanilla_Qwen3-1.7B'")
flags.DEFINE_integer("limit", 0, "Limit conversations (0 = all)")

ROLE_MAP = {"seeker": "user", "supporter": "assistant"}

# JSON schema for vLLM guided decoding (VS mode).
# Includes a "reasoning" field so thinking models (Qwen3) can reason inside
# the structured output instead of needing <think> blocks (which JSON grammar
# suppresses).
VS_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "responses": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "probability": {"type": "number"}
                },
                "required": ["text", "probability"]
            }
        }
    },
    "required": ["reasoning", "responses"]
}


def resolve_method(cfg, flat_key):
    """Resolve a flat method key like 'baseline1_vanilla_Qwen3-1.7B'.

    Returns (method_config_dict, model_name_or_path).
    For methods without models (e.g. 'gold'), model_name_or_path is None.
    The returned method_config has sampling_defaults merged in.
    """
    methods = cfg["methods"]
    defaults = cfg.get("sampling_defaults", {})

    # Direct match (e.g. 'gold')
    if flat_key in methods:
        mc = {**defaults, **methods[flat_key]}
        return mc, None

    # Try each method prefix: find longest matching method key
    best = None
    for method_key in methods:
        if flat_key.startswith(method_key + "_"):
            if best is None or len(method_key) > len(best):
                best = method_key

    if best is None:
        cprint(f"Error: method '{flat_key}' not found in config", "red", force_color=True)
        all_keys = []
        for mk, mv in methods.items():
            models = mv.get("models", {})
            if models:
                all_keys.extend(f"{mk}_{ml}" for ml in models)
            else:
                all_keys.append(mk)
        cprint(f"Available: {all_keys}", "yellow", force_color=True)
        sys.exit(1)

    model_label = flat_key[len(best) + 1:]  # strip "method_key_"
    models = methods[best].get("models", {})
    if model_label not in models:
        cprint(f"Error: model '{model_label}' not found under method '{best}'", "red", force_color=True)
        cprint(f"Available models: {list(models.keys())}", "yellow", force_color=True)
        sys.exit(1)

    mc = {**defaults, **methods[best]}
    return mc, models[model_label]


def strip_think_blocks(text):
    """Remove <think>...</think> blocks from model output."""
    cleaned = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()
    return cleaned if cleaned else text.strip()


def extract_reasoning(text):
    """Extract content inside <think>...</think> blocks."""
    match = re.search(r'<think>(.*?)</think>', text, flags=re.DOTALL)
    return match.group(1).strip() if match else ""


def parse_conversation(full_text):
    """Parse 'Seeker: ...\\nSupporter: ...' text into structured turns."""
    turns = []
    current_role = None
    current_lines = []

    for line in full_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("Seeker: "):
            if current_role:
                turns.append({"role": current_role, "content": "\n".join(current_lines)})
            current_role = "seeker"
            current_lines = [line[len("Seeker: "):]]
        elif line.startswith("Supporter: "):
            if current_role:
                turns.append({"role": current_role, "content": "\n".join(current_lines)})
            current_role = "supporter"
            current_lines = [line[len("Supporter: "):]]
        else:
            current_lines.append(line)

    if current_role:
        turns.append({"role": current_role, "content": "\n".join(current_lines)})

    return turns


def load_tactic_tags_by_conv(csv_path):
    """Load turn-level tactic tags grouped by conversation_id.

    Returns dict: conv_id -> list of (turn_idx, tactic_list) sorted by turn_idx.
    """
    from collections import defaultdict
    conv_tactics = defaultdict(dict)
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            conv_id = row["conversation_id"]
            turn_idx = int(row["supporter_turn_index"])
            tactics = []
            for tactic in VERL_TACTICS:
                if int(row.get(tactic, 0)) > 0:
                    tactics.append(tactic)
            if turn_idx in conv_tactics[conv_id]:
                existing = set(conv_tactics[conv_id][turn_idx])
                existing.update(tactics)
                conv_tactics[conv_id][turn_idx] = sorted(existing)
            else:
                conv_tactics[conv_id][turn_idx] = sorted(tactics)
    return dict(conv_tactics)


# ---------------------------------------------------------------------------
# Verbalized Sampling helpers
# ---------------------------------------------------------------------------

def _fix_trailing_commas(text):
    """Remove trailing commas before ] or } (common in freeform LLM JSON)."""
    return re.sub(r',\s*([}\]])', r'\1', text)


def _clean_vs_text(text):
    """Normalize escaped/newline-heavy candidate text snippets."""
    text = html.unescape(text)
    text = text.replace("\\n", "\n")
    return text.strip()


def _normalize_candidate_fields(candidates):
    """Normalize field aliases: confidence/nll -> probability (per VS paper)."""
    for c in candidates:
        if "probability" not in c:
            for alias in ("confidence", "prob", "likelihood"):
                if alias in c:
                    c["probability"] = c.pop(alias)
                    break
            else:
                c["probability"] = 0.5
        if "text" not in c:
            for alias in ("response", "content", "answer"):
                if alias in c:
                    c["text"] = c.pop(alias)
                    break
        if "text" in c:
            c["text"] = _clean_vs_text(c["text"])
    return candidates


def _try_parse_json(text):
    """Try to parse JSON, with trailing comma fix as fallback."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(_fix_trailing_commas(text))
    except json.JSONDecodeError:
        return None


def _extract_responses(data):
    """Extract (response_list, reasoning) from parsed JSON data."""
    reasoning = ""
    if isinstance(data, dict):
        reasoning = data.get("reasoning", "")
        if "responses" in data:
            return _normalize_candidate_fields(data["responses"]), reasoning
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        return _normalize_candidate_fields(data), reasoning
    return None, reasoning


def _parse_xml_attr_responses(text):
    """Parse tag-style responses like <response text="..." probability="0.3"/>."""
    patterns = [
        re.compile(r'<[^>]*\btext="(.*?)"[^>]*\bprobability="(.*?)"[^>]*/?>', re.DOTALL),
        re.compile(r"<[^>]*\btext='(.*?)'[^>]*\bprobability='(.*?)'[^>]*/?>", re.DOTALL),
        re.compile(r'<[^>]*\bprobability="(.*?)"[^>]*\btext="(.*?)"[^>]*/?>', re.DOTALL),
        re.compile(r"<[^>]*\bprobability='(.*?)'[^>]*\btext='(.*?)'[^>]*/?>", re.DOTALL),
    ]
    for pattern_idx, pattern in enumerate(patterns):
        out = []
        for match in pattern.finditer(text):
            if pattern_idx < 2:
                cand_text, prob = match.group(1), match.group(2)
            else:
                prob, cand_text = match.group(1), match.group(2)
            cand_text = _clean_vs_text(cand_text)
            if cand_text:
                out.append({"text": cand_text, "probability": prob})
        if out:
            return out
    return []


def _parse_xml_nested_responses(text):
    """Parse nested tag responses like <response><text>...</text><probability>0.2</probability></response>."""
    patterns = [
        re.compile(
            r"<response\b[^>]*>.*?<text>(.*?)</text>.*?<probability>(.*?)</probability>.*?</response>",
            re.DOTALL,
        ),
        re.compile(
            r"<([A-Za-z][\w:-]*)\b[^>]*>(.*?)</\1>\s*<probability>(.*?)</probability>",
            re.DOTALL,
        ),
    ]
    for pattern_idx, pattern in enumerate(patterns):
        out = []
        for match in pattern.finditer(text):
            if pattern_idx == 0:
                cand_text, prob = match.group(1), match.group(2)
            else:
                tag_name, cand_text, prob = match.group(1), match.group(2), match.group(3)
                if tag_name.lower() == "probability":
                    continue
            cand_text = _clean_vs_text(re.sub(r"<[^>]+>", " ", cand_text))
            if cand_text:
                out.append({"text": cand_text, "probability": prob})
        if out:
            return out
    return []


def _parse_probability_lines(text):
    """Parse freeform blocks separated by explicit '(Probability: x)' lines."""
    parts = re.split(r"\(\s*Probability:\s*([0-9.]+)\s*\)", text, flags=re.IGNORECASE | re.DOTALL)
    if len(parts) < 3:
        return []
    out = []
    current_text = parts[0]
    for i in range(1, len(parts), 2):
        prob = parts[i]
        cand_text = _clean_vs_text(current_text)
        if cand_text:
            out.append({"text": cand_text, "probability": prob})
        current_text = parts[i + 1] if i + 1 < len(parts) else ""
    tail = _clean_vs_text(current_text)
    if out and tail:
        out[-1]["text"] = _clean_vs_text(out[-1]["text"] + "\n" + tail)
    return out


def parse_vs_json(raw_text):
    """Parse VS JSON output into (candidates, reasoning).

    Returns:
        (list of {text, probability} dicts, reasoning_string)

    Handles: <think> blocks, direct JSON, markdown code blocks,
    embedded JSON objects, field aliases, trailing commas.
    """
    text = strip_think_blocks(raw_text).strip()

    # Try direct JSON parse
    data = _try_parse_json(text)
    if data is not None:
        result, reasoning = _extract_responses(data)
        if result:
            return result, reasoning

    # Try extracting JSON from markdown code blocks
    match = re.search(r'```(?:json)?\s*(.*?)```', text, re.DOTALL)
    if match:
        data = _try_parse_json(match.group(1).strip())
        if data is not None:
            result, reasoning = _extract_responses(data)
            if result:
                return result, reasoning

    # Try finding a JSON object in the text (greedy: outermost braces)
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        data = _try_parse_json(match.group(0))
        if data is not None:
            result, reasoning = _extract_responses(data)
            if result:
                return result, reasoning

    # Try XML-ish response formats seen in malformed model outputs.
    for parser in (_parse_xml_attr_responses, _parse_xml_nested_responses, _parse_probability_lines):
        result = parser(text)
        if len(result) >= 2:
            return _normalize_candidate_fields(result), ""

    # Fallback: treat entire output as single response
    cprint(f"  [VS] JSON parse failed, using raw text as single candidate", "yellow", force_color=True)
    return [{"text": text, "probability": 1.0}], ""


def vs_select_response(candidates, weight_mode="elicited", tau=0.0,
                        min_k_survivors=3, seed=42):
    """Select one response from VS candidates.

    Args:
        candidates: list of {"text": str, "probability": float}
        weight_mode: "elicited" (renormalize), "softmax", or "uniform"
        tau: filter threshold (0.0 = keep all)
        min_k_survivors: minimum candidates to keep after filtering
        seed: random seed for reproducibility

    Returns:
        (selected_text, selected_idx, all_candidates_with_probs)
    """
    if not candidates:
        return "", -1, []

    # Clean up probabilities
    for c in candidates:
        p = c.get("probability", 0)
        if isinstance(p, str):
            p = p.replace("%", "")
            p = float(p)
            if p > 1.0:
                p = p / 100.0
        c["probability"] = max(float(p), 0.0)

    # Filter by tau (keep candidates with p >= tau; tau=0.0 keeps all)
    if tau > 0:
        filtered = [c for c in candidates if c["probability"] >= tau]
        if len(filtered) < min_k_survivors:
            filtered = sorted(candidates, key=lambda c: c["probability"], reverse=True)[:min_k_survivors]
    else:
        filtered = list(candidates)

    if not filtered:
        return candidates[0].get("text", ""), 0, candidates

    # Compute weights
    raw_probs = [c["probability"] for c in filtered]

    if weight_mode == "uniform":
        weights = [1.0 / len(filtered)] * len(filtered)
    elif weight_mode == "softmax":
        max_p = max(raw_probs)
        exp_probs = [math.exp(p - max_p) for p in raw_probs]
        total = sum(exp_probs)
        weights = [p / total for p in exp_probs]
    else:  # elicited (default)
        total = sum(raw_probs)
        if total > 0:
            weights = [p / total for p in raw_probs]
        else:
            weights = [1.0 / len(filtered)] * len(filtered)

    # Sample
    rng = random.Random(seed)
    idx = rng.choices(range(len(filtered)), weights=weights, k=1)[0]
    selected_text = filtered[idx].get("text", "")

    # Find original index
    orig_idx = candidates.index(filtered[idx]) if filtered[idx] in candidates else idx

    return selected_text, orig_idx, candidates


def build_base_system_prompt(vs_base_mode, mc, conv_tactic_tags, conv_id,
                             supporter_turn_idx, tactic_last_turn_counts,
                             tactic_history_counts):
    """Build the base system prompt for a given prompt mode."""
    if vs_base_mode == "simple":
        return mc["system_prompt"].strip()
    elif vs_base_mode == "tactic_aware":
        return mc["system_prompt"].strip()
    elif vs_base_mode == "verl_history":
        return build_verl_system_prompt(
            dict(tactic_last_turn_counts),
            dict(tactic_history_counts),
            VERL_TACTICS,
        )
    else:
        cprint(f"Error: unknown vs_base_prompt_mode '{vs_base_mode}'", "red", force_color=True)
        sys.exit(1)


def main(_):
    if not FLAGS.method:
        cprint("Error: --method is required", "red", force_color=True)
        sys.exit(1)

    with open(FLAGS.config) as f:
        cfg = yaml.safe_load(f)

    config_dir = os.path.dirname(os.path.abspath(FLAGS.config))
    method_name = FLAGS.method

    mc, model_path = resolve_method(cfg, method_name)

    if mc.get("skip_sampling"):
        cprint(f"Method '{method_name}' has skip_sampling=true, skipping.", "yellow", force_color=True)
        return

    output_json = os.path.join(config_dir, "outputs", method_name, "conversations.json")
    if os.path.isfile(output_json):
        with open(output_json) as f:
            existing = json.load(f)
        if len(existing) >= 315:
            cprint(f"Output complete ({len(existing)} entries): {output_json}, skipping.", "yellow", force_color=True)
            return
        cprint(f"Output incomplete ({len(existing)}/315 entries), regenerating.", "yellow", force_color=True)

    model_name = model_path
    prompt_mode = mc["prompt_mode"]
    temperature = mc["temperature"]
    max_tokens = mc["max_tokens"]
    is_thinking_model = mc.get("is_thinking_model", False)
    gpu_id = str(cfg.get("gpu_id", 0))
    dtype = mc.get("dtype", "bfloat16")
    gpu_memory_utilization = mc.get("gpu_memory_utilization", 0.90)
    max_model_len = mc.get("max_model_len", 8192)

    # VS-specific config
    is_vs = prompt_mode == "verbalized_sampling"
    vs_defaults = cfg.get("vs_defaults", {})
    vs_base_mode = mc.get("vs_base_prompt_mode", vs_defaults.get("vs_base_prompt_mode", "simple"))
    vs_k = mc.get("vs_k", vs_defaults.get("vs_k", 5))
    vs_tau = mc.get("vs_tau", vs_defaults.get("vs_tau", 0.0))
    vs_weight_mode = mc.get("vs_weight_mode", vs_defaults.get("vs_weight_mode", "elicited"))
    vs_seed = mc.get("vs_seed", vs_defaults.get("vs_seed", 42))
    vs_prompt_template = mc.get("vs_prompt_template", vs_defaults.get("vs_prompt_template", ""))

    if is_vs:
        cprint(f"  VS mode: base={vs_base_mode}, k={vs_k}, tau={vs_tau}, "
               f"weight_mode={vs_weight_mode}, seed={vs_seed}", "cyan", force_color=True)

    conversations_csv = cfg["test_set"]["conversations_csv"]

    # Set GPU before importing vllm (respect shell-level override)
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    # Ensure conda env's libstdc++ is found (needed on GH200 nodes).
    conda_lib = os.path.join(os.path.dirname(sys.executable), "..", "lib")
    conda_lib = os.path.abspath(conda_lib)
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if conda_lib not in ld_path:
        os.environ["LD_LIBRARY_PATH"] = f"{conda_lib}:{ld_path}"
    # Disable V1 multiprocessing to avoid spawned subprocess libstdc++ mismatch.
    # The EngineCore runs in-process instead, sidestepping the CXXABI_1.3.15 issue.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    from vllm import LLM, SamplingParams

    # Load conversations from CSV
    cprint(f"Loading conversations from {conversations_csv}...", "cyan", force_color=True)
    with open(conversations_csv, newline="") as f:
        reader = csv.DictReader(f)
        conv_rows = list(reader)
    cprint(f"  Loaded {len(conv_rows)} conversations", "green", force_color=True)

    limit = FLAGS.limit
    if limit > 0:
        conv_rows = conv_rows[:limit]
        cprint(f"  Limited to {len(conv_rows)} conversations", "yellow", force_color=True)

    # For verl_history mode (or VS wrapping verl_history), load gold tactic tags
    needs_tactic_tags = (prompt_mode == "verl_history") or (is_vs and vs_base_mode == "verl_history")
    conv_tactic_tags = None
    if needs_tactic_tags:
        tactics_csv = cfg["test_set"]["turn_level_tactics_csv"]
        cprint(f"Loading tactic tags from {tactics_csv}...", "cyan", force_color=True)
        conv_tactic_tags = load_tactic_tags_by_conv(tactics_csv)
        cprint(f"  Loaded tags for {len(conv_tactic_tags)} conversations", "green", force_color=True)

    # Build generation tasks
    tasks_info = []   # (conv_id, turn_idx, history)
    all_chat_messages = []
    per_turn_system_prompts = []

    for conv_row in conv_rows:
        conv_id = conv_row["conversation_id"]
        turns = parse_conversation(conv_row["full_conversation"])

        tactic_history_counts = {}
        tactic_last_turn_counts = {}
        supporter_turn_idx = 0

        for i, turn in enumerate(turns):
            if turn["role"] != "supporter":
                continue
            supporter_turn_idx += 1

            # Build conversation history
            history = []
            for prev in turns[:i]:
                role = ROLE_MAP.get(prev["role"], prev["role"])
                history.append({"role": role, "content": prev["content"]})

            if not any(m["role"] == "user" for m in history):
                if needs_tactic_tags and conv_tactic_tags:
                    turn_tactics = conv_tactic_tags.get(conv_id, {}).get(supporter_turn_idx, [])
                    tactic_last_turn_counts = {}
                    for t in turn_tactics:
                        tactic_history_counts[t] = tactic_history_counts.get(t, 0) + 1
                        tactic_last_turn_counts[t] = tactic_last_turn_counts.get(t, 0) + 1
                continue

            # Build system prompt
            if is_vs:
                base_prompt = build_base_system_prompt(
                    vs_base_mode, mc, conv_tactic_tags, conv_id,
                    supporter_turn_idx, tactic_last_turn_counts,
                    tactic_history_counts,
                )
                vs_instructions = vs_prompt_template.format(vs_k=vs_k).strip()
                system_prompt = f"{base_prompt}\n\n{vs_instructions}"
            elif prompt_mode == "simple":
                system_prompt = mc["system_prompt"].strip()
            elif prompt_mode == "tactic_aware":
                system_prompt = mc["system_prompt"].strip()
            elif prompt_mode == "verl_history":
                system_prompt = build_verl_system_prompt(
                    dict(tactic_last_turn_counts),
                    dict(tactic_history_counts),
                    VERL_TACTICS,
                )
            else:
                cprint(f"Error: unknown prompt_mode '{prompt_mode}'", "red", force_color=True)
                sys.exit(1)

            chat_messages = [{"role": "system", "content": system_prompt}] + history
            tasks_info.append((conv_id, i, history))
            all_chat_messages.append(chat_messages)
            per_turn_system_prompts.append(system_prompt)

            # Update tactic history (after building prompt)
            if needs_tactic_tags and conv_tactic_tags:
                turn_tactics = conv_tactic_tags.get(conv_id, {}).get(supporter_turn_idx, [])
                tactic_last_turn_counts = {}
                for t in turn_tactics:
                    tactic_history_counts[t] = tactic_history_counts.get(t, 0) + 1
                    tactic_last_turn_counts[t] = tactic_last_turn_counts.get(t, 0) + 1

    cprint(f"  Prepared {len(tasks_info)} generation tasks (supporter turns)",
           "cyan", force_color=True)

    # Initialize vLLM offline engine
    cprint(f"\nLoading model: {model_name} (gpu={gpu_id}, dtype={dtype})...",
           "cyan", force_color=True)
    llm = LLM(
        model=model_name,
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
    )

    # Build sampling params (with guided JSON for VS mode).
    # The JSON schema includes a "reasoning" field so thinking models can
    # reason inside the structured output without needing <think> blocks.
    sp_kwargs = {"temperature": temperature, "max_tokens": max_tokens}
    if is_vs:
        from vllm.sampling_params import StructuredOutputsParams
        sp_kwargs["structured_outputs"] = StructuredOutputsParams(json=VS_JSON_SCHEMA)
    sampling_params = SamplingParams(**sp_kwargs)

    # Generate all responses in one batch
    cprint(f"\nGenerating {len(all_chat_messages)} responses...", "cyan", force_color=True)
    outputs = llm.chat(all_chat_messages, sampling_params=sampling_params)

    # Build flat output
    output_entries = []
    empty_count = 0
    vs_parse_fail_count = 0

    for result_idx, (conv_id, turn_idx, history) in enumerate(tasks_info):
        raw_text = outputs[result_idx].outputs[0].text or ""

        if is_vs:
            # VS post-processing: parse JSON (reasoning comes from JSON field), select candidate
            candidates, reasoning = parse_vs_json(raw_text)
            parse_ok = len(candidates) > 1 or (len(candidates) == 1 and candidates[0].get("probability", 1.0) != 1.0)
            if not parse_ok:
                vs_parse_fail_count += 1
            per_turn_seed = vs_seed + result_idx
            text, selected_idx, all_cands = vs_select_response(
                candidates, weight_mode=vs_weight_mode, tau=vs_tau,
                min_k_survivors=3, seed=per_turn_seed,
            )
            entry = {
                "conversation_id": conv_id,
                "source": "lend_an_ear",
                "model": model_name,
                "system_prompt": per_turn_system_prompts[result_idx],
                "conversation_history": history,
                "reasoning": reasoning,
                "model_response": text,
                "vs_candidates": [{"text": c.get("text", ""), "probability": c.get("probability", 0)} for c in all_cands],
                "vs_selected_idx": selected_idx,
                "vs_parse_ok": parse_ok,
            }
        else:
            reasoning = extract_reasoning(raw_text) if is_thinking_model else ""
            text = strip_think_blocks(raw_text) if is_thinking_model else raw_text
            entry = {
                "conversation_id": conv_id,
                "source": "lend_an_ear",
                "model": model_name,
                "system_prompt": per_turn_system_prompts[result_idx],
                "conversation_history": history,
                "reasoning": reasoning,
                "model_response": text,
            }

        if not text.strip():
            empty_count += 1
        output_entries.append(entry)

    # Save output
    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(output_entries, f, indent=2, ensure_ascii=False)

    cprint(f"\nDone!", "green", force_color=True)
    cprint(f"  Method: {method_name}", "green", force_color=True)
    cprint(f"  Model: {model_name}", "green", force_color=True)
    cprint(f"  Prompt mode: {prompt_mode}", "green", force_color=True)
    cprint(f"  Entries: {len(output_entries)}", "green", force_color=True)
    if empty_count > 0:
        cprint(f"  Empty responses: {empty_count}", "red", force_color=True)
    if is_vs:
        cprint(f"  VS parse failures: {vs_parse_fail_count}/{len(output_entries)}", "yellow", force_color=True)
    cprint(f"  Output: {output_json}", "green", force_color=True)


if __name__ == "__main__":
    app.run(main)
