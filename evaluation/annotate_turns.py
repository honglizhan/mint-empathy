"""
Turn-level empathy annotation for 322 merged emotion-support conversations.

Evaluates every supporter turn on 6 empathy sub-components (1-5 scale),
then aggregates into a single empathy score per turn.

Adapted from open_source_replication/annotate_conversations.py (conversation-level)
and sense-7-lend_an_ear/empathy_judge-[lend_an_ear].py (turn-level pattern).

Usage:
    python annotate_turns.py --config config.yaml --run gpt-oss-120b
    python annotate_turns.py --config config.yaml --run gpt-oss-120b --limit 1
"""

import os
import re
import json
import asyncio
import yaml
import pandas as pd
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio
from termcolor import cprint
from absl import app, flags

load_dotenv()

FLAGS = flags.FLAGS

flags.DEFINE_string("config", "config.yaml", "Path to YAML config file")
flags.DEFINE_string("run", "all",
                    "Model name to run (must match a 'name' in config models list), or 'all'")
flags.DEFINE_integer("limit", 0, "Limit number of conversations to process (0 = all)")
flags.DEFINE_string("conversations_json", None,
                    "Override: path to conversations JSON (overrides config)")
flags.DEFINE_string("output_dir", None,
                    "Override: output directory (overrides config)")


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_conversations(json_path):
    """Loads conversations JSON and expands into per-supporter-turn rows.

    Returns a list of dicts, each with:
      conversation_id, turn_index, conversation_history, current_turn, turn_text
    """
    with open(json_path, 'r') as f:
        conversations = json.load(f)

    rows = []
    for conv in conversations:
        conv_id = conv["conversation_id"]
        turns = conv["conversation"]

        for i, turn in enumerate(turns):
            if turn["role"] != "supporter":
                continue

            # Build conversation history from all prior turns
            history_parts = []
            for prev_turn in turns[:i]:
                role_label = "Seeker" if prev_turn["role"] == "seeker" else "Supporter"
                history_parts.append(f"{role_label}: {prev_turn['content']}")

            history = "\n".join(history_parts) if history_parts else "No previous history."
            current_turn = f"Supporter: {turn['content']}"

            rows.append({
                "conversation_id": conv_id,
                "turn_index": i,
                "conversation_history": history,
                "current_turn": current_turn,
                "turn_text": turn["content"],
            })

    return rows


# ── Prompt Assembly ──────────────────────────────────────────────────────────

def format_few_shot(few_shot_examples):
    """Formats few-shot examples for a sub-component (turn-level format)."""
    parts = []
    for i, ex in enumerate(few_shot_examples, 1):
        parts.append(
            f"#### Example {i}\n\n"
            f"- Conversation History:\n\n{ex['conversation_history']}\n\n"
            f"- Current Turn:\n\n{ex['current_turn']}\n\n"
            f"<score>{ex['score']}</score>"
        )
    return "\n\n".join(parts)


def build_prompt(template, framework, component_def, history, current_turn):
    """Assembles a single prompt for one sub-component evaluation."""
    few_shot_str = format_few_shot(component_def["few_shot"])
    question_rubric = f"**Question:** {component_def['question']}"
    rubric = component_def["rubric"]

    return template.format(
        framework=framework,
        few_shot=few_shot_str,
        history=history,
        current_turn=current_turn,
        question=question_rubric,
        rubric=rubric,
    )


def parse_score(response_text):
    """Extracts integer 1-5 from <score>N</score> tags in the LLM response."""
    if response_text is None:
        return None
    match = re.search(r'<score>\s*([1-5])\s*</score>', response_text)
    if match:
        return int(match.group(1))
    return None


def aggregate_scores(raw_scores, sub_components):
    """Aggregates 6 sub-component scores into a single empathy score.

    1. Reverse-code proscriptive (negative-polarity) items: reversed = 6 - raw
    2. Average all 6 scores (now all on 1-5 scale, higher = more empathic)
    """
    coded = []
    for name, comp_def in sub_components.items():
        raw = raw_scores.get(name)
        if raw is None:
            continue
        if comp_def["polarity"] == "negative":
            coded.append(6 - raw)
        else:
            coded.append(raw)

    if not coded:
        return None

    return round(sum(coded) / len(coded), 4)


# ── API Call ─────────────────────────────────────────────────────────────────

async def call_llm(client, model, prompt, temperature, max_tokens, semaphore,
                   max_retries, request_delay=0):
    """Makes an async OpenAI-compatible (vLLM) API call with retry logic.

    Returns (text, reasoning_content, input_tokens, output_tokens).
    """
    async with semaphore:
        if request_delay > 0:
            await asyncio.sleep(request_delay)
        for attempt in range(max_retries):
            try:
                kwargs = dict(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                )
                if temperature is not None:
                    kwargs["temperature"] = temperature
                response = await client.chat.completions.create(**kwargs)
                usage = response.usage
                msg = response.choices[0].message
                content = msg.content or ""
                reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
                return (
                    content,
                    reasoning,
                    usage.prompt_tokens if usage else 0,
                    usage.completion_tokens if usage else 0,
                )
            except Exception as e:
                if attempt < max_retries - 1:
                    retry_after = re.search(r'retry after (\d+) seconds', str(e))
                    wait = int(retry_after.group(1)) + 1 if retry_after else 2 ** attempt + 1
                    cprint(f"  Retry {attempt+1}/{max_retries} after {wait}s: {e}", "red", force_color=True)
                    await asyncio.sleep(wait)
                else:
                    cprint(f"  FAILED after {max_retries} retries: {e}", "red", force_color=True)
                    return (None, None, 0, 0)


# ── Single Model Run ─────────────────────────────────────────────────────────

async def run_single_model(model_cfg, cfg, config_dir, turn_rows, framework, template, sub_components):
    """Runs turn-level annotation for one model."""
    name = model_cfg["name"]
    model = model_cfg["model"]
    temperature = model_cfg.get("temperature")
    max_tokens = model_cfg.get("max_tokens", 4096)
    max_concurrency = model_cfg.get("max_concurrency", 64)
    max_retries = model_cfg.get("max_retries", 5)
    request_delay = model_cfg.get("request_delay", 0)

    if FLAGS.output_dir:
        output_dir = os.path.abspath(FLAGS.output_dir)
    else:
        output_dir = os.path.join(config_dir, cfg.get("output_dir", "outputs"))
    ratings_path = os.path.join(output_dir, f"{name}_ratings.csv")
    predictions_path = os.path.join(output_dir, f"{name}_predictions.jsonl")
    summary_path = os.path.join(output_dir, f"{name}_summary.json")

    cprint(f"\n{'='*60}", "yellow", force_color=True)
    cprint(f"  Running: {name}  (vllm / {model})", "yellow", force_color=True)
    cprint(f"  temp={temperature}, max_tokens={max_tokens}, "
           f"concurrency={max_concurrency}", "yellow", force_color=True)
    cprint(f"{'='*60}", "yellow", force_color=True)

    dimensions = cfg["dimensions"]

    # Apply limit (by conversation)
    rows = turn_rows
    if FLAGS.limit > 0:
        # Limit by number of conversations, not individual turns
        conv_ids = []
        seen = set()
        for r in rows:
            if r["conversation_id"] not in seen:
                seen.add(r["conversation_id"])
                conv_ids.append(r["conversation_id"])
        limited_conv_ids = set(conv_ids[:FLAGS.limit])
        rows = [r for r in rows if r["conversation_id"] in limited_conv_ids]
        cprint(f"Limiting to {FLAGS.limit} conversation(s) "
               f"({len(rows)} supporter turns).", "cyan", force_color=True)

    n_turns = len(rows)
    cprint(f"Prepared {n_turns * len(dimensions)} prompts "
           f"({n_turns} supporter turns x {len(dimensions)} dimensions).",
           "cyan", force_color=True)

    # Build all prompts
    all_prompts = []
    prompt_index = []  # (turn_idx, dimension_name)

    for turn_idx, row in enumerate(rows):
        for dim_name in dimensions:
            prompt = build_prompt(template, framework, sub_components[dim_name],
                                  row["conversation_history"], row["current_turn"])
            all_prompts.append(prompt)
            prompt_index.append((turn_idx, dim_name))

    # Initialize vLLM client(s) - supports round-robin across multiple URLs
    from openai import AsyncOpenAI
    base_urls_raw = model_cfg.get("base_url", cfg.get("vllm_base_url", cfg.get("vllm_base_urls", ["http://localhost:8000/v1"])))
    if isinstance(base_urls_raw, str):
        base_urls_raw = [base_urls_raw]
    clients = [AsyncOpenAI(base_url=url, api_key="unused", max_retries=0) for url in base_urls_raw]
    semaphore = asyncio.Semaphore(max_concurrency)
    cprint(f"  Using {len(clients)} vLLM endpoint(s): {base_urls_raw}", "cyan", force_color=True)

    tasks = [
        call_llm(clients[i % len(clients)], model, p, temperature, max_tokens, semaphore,
                 max_retries, request_delay)
        for i, p in enumerate(all_prompts)
    ]

    cprint("Running inference...", "cyan", force_color=True)
    responses = await tqdm_asyncio.gather(*tasks, desc=f"{name}")

    # Parse responses and collect scores
    turn_scores = [{} for _ in range(n_turns)]
    turn_raw_responses = [{} for _ in range(n_turns)]
    turn_reasoning_responses = [{} for _ in range(n_turns)]
    total_input_tokens = 0
    total_output_tokens = 0

    for i, (response_text, reasoning_text, in_tok, out_tok) in enumerate(responses):
        turn_idx, dim_name = prompt_index[i]
        score = parse_score(response_text)
        if score is None and reasoning_text:
            score = parse_score(reasoning_text)
        turn_scores[turn_idx][dim_name] = score
        turn_raw_responses[turn_idx][dim_name] = response_text
        turn_reasoning_responses[turn_idx][dim_name] = reasoning_text
        total_input_tokens += in_tok
        total_output_tokens += out_tok

    # Build ratings CSV and predictions JSONL
    ratings_rows = []
    predictions = []

    for turn_idx, row in enumerate(rows):
        scores = turn_scores[turn_idx]
        agg = aggregate_scores(scores, sub_components)

        # Ratings row (flat CSV)
        ratings_row = {
            "conversation_id": row["conversation_id"],
            "turn_index": row["turn_index"],
        }
        for dim_name in dimensions:
            ratings_row[dim_name] = scores.get(dim_name)
        ratings_row["aggregated_empathy"] = agg
        ratings_rows.append(ratings_row)

        # Full prediction record (JSONL)
        pred = {
            "conversation_id": row["conversation_id"],
            "turn_index": row["turn_index"],
            "model": name,
            "turn_text": row["turn_text"][:200] + ("..." if len(row["turn_text"]) > 200 else ""),
        }
        for dim_name in dimensions:
            pred[dim_name] = scores.get(dim_name)
        pred["aggregated_empathy"] = agg
        pred["raw_responses"] = turn_raw_responses[turn_idx]
        reasoning = turn_reasoning_responses[turn_idx]
        if any(v is not None for v in reasoning.values()):
            pred["reasoning_responses"] = reasoning
        predictions.append(pred)

    # Save ratings CSV
    os.makedirs(output_dir, exist_ok=True)
    ratings_df = pd.DataFrame(ratings_rows)
    ratings_df.to_csv(ratings_path, index=False)
    cprint(f"Ratings saved to {ratings_path}", "green", force_color=True)

    # Save predictions JSONL
    with open(predictions_path, 'w') as f:
        for pred in predictions:
            f.write(json.dumps(pred, default=str) + "\n")
    cprint(f"Predictions saved to {predictions_path}", "green", force_color=True)

    # Stats
    total_cells = n_turns * len(dimensions)
    parsed_cells = sum(1 for s in turn_scores for d in dimensions if s.get(d) is not None)
    parse_rate = parsed_cells / total_cells * 100 if total_cells > 0 else 0

    cprint(f"\n  Supporter turns: {n_turns}", "yellow", force_color=True)
    cprint(f"  Parsed scores: {parsed_cells}/{total_cells} ({parse_rate:.1f}%)", "yellow", force_color=True)

    for dim_name in dimensions:
        dim_parsed = sum(1 for s in turn_scores if s.get(dim_name) is not None)
        cprint(f"  {dim_name}: {dim_parsed}/{n_turns} parsed", "yellow", force_color=True)

    total_tokens = total_input_tokens + total_output_tokens
    cprint(f"\n  Tokens: {total_input_tokens:,} input + {total_output_tokens:,} output = {total_tokens:,} total",
           "yellow", force_color=True)

    # Save summary
    summary = {
        "model": name,
        "model_id": model,
        "temperature": temperature,
        "conversations": len(set(r["conversation_id"] for r in rows)),
        "supporter_turns": n_turns,
        "total_prompts": total_cells,
        "parsed_scores": parsed_cells,
        "parse_rate": round(parse_rate, 2),
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    cprint(f"Summary saved to {summary_path}", "green", force_color=True)


# ── Main Pipeline ────────────────────────────────────────────────────────────

async def run_annotation():
    config_path = os.path.abspath(FLAGS.config)
    config_dir = os.path.dirname(config_path)

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Load conversations and expand to per-turn rows
    if FLAGS.conversations_json:
        conv_path = os.path.abspath(FLAGS.conversations_json)
    else:
        conv_path = os.path.join(config_dir, cfg["conversations_json"])
    cprint(f"Loading conversations from {conv_path}...", "cyan", force_color=True)
    turn_rows = load_conversations(conv_path)
    n_convs = len(set(r["conversation_id"] for r in turn_rows))
    cprint(f"Loaded {n_convs} conversations, {len(turn_rows)} supporter turns.", "cyan", force_color=True)

    # Load framework, template, sub-components
    with open(os.path.join(config_dir, cfg["framework_path"]), 'r') as f:
        framework = f.read()
    with open(os.path.join(config_dir, cfg["template_path"]), 'r') as f:
        template = f.read()
    with open(os.path.join(config_dir, cfg["sub_components_json"]), 'r') as f:
        sub_components = json.load(f)

    cprint(f"Dimensions: {cfg['dimensions']}", "cyan", force_color=True)

    # Determine which models to run
    all_models = cfg["models"]
    if FLAGS.run == "all":
        models_to_run = all_models
    else:
        models_to_run = [m for m in all_models if m["name"] == FLAGS.run]
        if not models_to_run:
            available = [m["name"] for m in all_models]
            raise ValueError(f"Model '{FLAGS.run}' not found in config. Available: {available}")

    for model_cfg in models_to_run:
        await run_single_model(model_cfg, cfg, config_dir, turn_rows, framework, template, sub_components)

    cprint(f"\n{'='*60}", "green", force_color=True)
    cprint("  All annotation runs complete.", "green", force_color=True)
    cprint(f"{'='*60}", "green", force_color=True)


def main(_):
    asyncio.run(run_annotation())


if __name__ == "__main__":
    app.run(main)
