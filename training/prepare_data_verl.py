"""
Convert tactics-tagged JSON to VERL parquet format.

Supports two modes via --include_tactics flag:
  --include_tactics (default): Full tactic-aware prompt with definitions,
      usage history, and format constraints. For diversity experiments.
  --noinclude_tactics: Minimal prompt with just role + format + length.
      For quality-only baselines.

Usage:
    python prepare_data_verl.py \
        --input_file /path/to/merged_emotion_support_conversations_322_tactics_tagged.json \
        --output_dir data/
    python prepare_data_verl.py --noinclude_tactics --output_dir data/
"""

import json
import os
import sys
from collections import Counter

# Ensure sibling modules are importable when run from the project root
_THIS_DIR = os.path.dirname(os.path.realpath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import pandas as pd
from absl import app, flags
from termcolor import cprint

from reward_func_tactics_kl_bigram_entropy import TACTIC_NAMES

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "input_file",
    "../data/training/conversations_322_tagged.json",
    "Path to tactics-tagged JSON input",
)
flags.DEFINE_string("output_dir", "data/", "Directory for output parquet files")
flags.DEFINE_boolean("include_tactics", True,
                     "Include tactic definitions and history in system prompt")

TACTIC_DEFINITIONS = {
    "information": "Provide factual statements, resources, or data to answer questions or calm anxieties (e.g., \"Flying is the safest form of travel\", \"Here is a resource: [x]\").",
    "assistance": "Explicitly offer to personally help or connect the user with someone who can (e.g., \"Please call me if you need anything\", \"Can I do anything to help?\").",
    "advice": "Suggest actionable solutions or coping strategies the user could employ (e.g., \"You could try...\", \"It might help if you...\").",
    "validation": "Reassure, normalize, or validate feelings, explicitly or implicitly (e.g., \"That makes sense\", \"You're not overreacting\", \"Everyone feels this way sometimes\").",
    "emotional_expression": "Share your own feelings, reactions, hopes, or opinions about the user's situation (e.g., \"I'm so sorry\", \"I hope things get better\").",
    "paraphrasing": "Communicate your understanding of the user's situation or feelings back to them (e.g., \"It sounds like you...\", \"What I'm hearing is...\").",
    "self_disclosure": "Share personal information or acknowledge similar past experiences (e.g., \"I've felt the same way\", \"When I was in college...\").",
    "questioning": "Ask questions to better understand the user's feelings or situation (e.g., \"How are you feeling?\", \"What happened?\").",
    "reappraisal": "Prompt cognitive reappraisal to change how the user thinks about the situation (e.g., \"It wasn't your fault\", \"Some things are beyond our control\").",
    "empowerment": "Make positive, uplifting statements about the user's character or capability (e.g., \"You are strong\", \"You've totally got this\").",
}

MINIMAL_SYSTEM_PROMPT = """\
### Role
You are a supporter helping a user who is seeking emotional support. Respond with empathy and understanding.

### Response Format
Write your response as a natural, conversational message.

### Response Length
Keep your response concise and under 200 tokens. Focus on quality over quantity."""

ROLE_MAP = {"seeker": "user", "supporter": "assistant"}


def build_tactic_system_prompt(tactic_last_turn_counts, tactic_history_counts, tactic_names):
    """Build the full system prompt with role, tactic definitions, and history."""
    lines = [
        "### Role",
        "You are a supporter helping a user who is seeking emotional support. Your task is to respond with empathy and understanding in a multi-turn conversation. Use a variety of empathy tactics when responding. The definition of empathy tactics is provided below. In addition, an overview of empathy tactics used in the previous turn is also provided. You can use it as a reference for determining which tactics to deploy in the current turn.",
        "",
        "### Empathy Tactics",
        "Empathy tactics are specific communication strategies used in supportive conversations. "
        "Using a variety of tactics makes your response more helpful and avoids repetitive patterns. "
        "Draw from these 10 tactics and aim to use different ones across your turns:",
        "",
    ]

    for i, tactic in enumerate(tactic_names, 1):
        definition = TACTIC_DEFINITIONS.get(tactic, "")
        lines.append(f"{i}. **{tactic}**: {definition}")
    lines.append("")

    lines.append("### Empathy Tactic Usage")

    has_last_turn = tactic_last_turn_counts and any(
        v is not None for v in tactic_last_turn_counts.values()
    )

    if not has_last_turn:
        lines.append("- This is the first turn. No tactics have been used yet.")
        lines.append("Try to use a variety of tactics in your response.")
    else:
        last_items = sorted(
            [(k, v) for k, v in tactic_last_turn_counts.items() if v is not None],
            key=lambda x: x[1],
            reverse=True,
        )
        last_parts = [f"{tactic} ({int(count)}x)" for tactic, count in last_items]
        lines.append(f"- Previous turn used: {', '.join(last_parts)}.")
        lines.append("Try to use *different* tactics than what was just used.")

        used_set = {k for k, v in tactic_history_counts.items() if v is not None} if tactic_history_counts else set()
        unused = [t for t in tactic_names if t not in used_set]
        if unused:
            lines.append(f"- Tactics not yet used in this conversation: {', '.join(unused)}.")
            lines.append("Prioritize using tactics you haven't tried yet.")
        else:
            lines.append("- All tactics have been used at least once. Try to vary your combinations further.")

    lines.append("")
    lines.append("### Response Format")
    lines.append(
        "Write your response as a natural, conversational message. "
        "The tactic definitions above are for your internal reference only. "
        "Do NOT include any of the following in your response:"
    )
    lines.append("- Tactic names as labels or headers (e.g., **Validation**: , *Paraphrasing*, (Empowerment))")
    lines.append("- Numbered tactic lists (e.g., **1. Reappraisal**, **2. Advice**)")
    lines.append("- Meta-commentary about tactics (e.g., \"Empathy Tactics Used So Far\", \"Tactic Count\", \"Tactics used in this turn\")")
    lines.append("- Markdown section headers for individual tactics (e.g., ### Validation)")
    lines.append("Your response should read like a genuine, caring human message with no visible structure or labeling of therapeutic techniques.")
    lines.append("")
    lines.append("### Response Length")
    lines.append("Keep your response concise and under 200 tokens. Focus on quality over quantity.")

    return "\n".join(lines)


def process_conversations(data, include_tactics):
    """Process conversations into VERL-compatible rows."""
    valid_tactics = set(TACTIC_NAMES)
    rows = []

    for conv_entry in data:
        conversation = conv_entry.get("conversation", [])
        assistant_indices = [
            i for i, msg in enumerate(conversation) if msg.get("role") == "supporter"
        ]

        for target_idx in assistant_indices:
            history = conversation[:target_idx]

            # Build prompt messages with role mapping
            prompt_messages = [
                {"role": ROLE_MAP.get(msg["role"], msg["role"]), "content": msg["content"]}
                for msg in history
            ]

            # Skip if no seeker message in history
            if not any(msg["role"] == "user" for msg in prompt_messages):
                continue

            prompt_text = "\n".join(f"{m['role']}: {m['content']}" for m in prompt_messages)

            if include_tactics:
                # Collect tactic history
                tactic_history_counts = {}
                tactic_last_turn_counts = {}
                tactic_history_bigrams = {}
                tactic_history_trigrams = {}
                assistant_turn_idx = 0

                for msg in history:
                    if msg.get("role") == "supporter":
                        # Use sentence_tactics for sentence-level counts
                        # (matches curr_counts unit in reward_verl.py)
                        sentence_tactics_data = msg.get("sentence_tactics") or []
                        if sentence_tactics_data:
                            flat_tactics = [
                                t
                                for sent in sentence_tactics_data
                                for t in sent.get("tactics", [])
                                if t in valid_tactics
                            ]
                        else:
                            # Fallback to all_tactics if sentence_tactics unavailable
                            flat_tactics = [t for t in msg.get("all_tactics", []) if t in valid_tactics]

                        for tactic in flat_tactics:
                            tactic_history_counts[tactic] = tactic_history_counts.get(tactic, 0) + 1

                        for i in range(len(flat_tactics) - 1):
                            src, dst = flat_tactics[i], flat_tactics[i + 1]
                            if src not in tactic_history_bigrams:
                                tactic_history_bigrams[src] = {}
                            tactic_history_bigrams[src][dst] = (
                                tactic_history_bigrams[src].get(dst, 0) + 1
                            )

                        for i in range(len(flat_tactics) - 2):
                            a, b, c = flat_tactics[i], flat_tactics[i + 1], flat_tactics[i + 2]
                            tactic_history_trigrams.setdefault(a, {}).setdefault(b, {})[c] = (
                                tactic_history_trigrams.get(a, {}).get(b, {}).get(c, 0) + 1
                            )

                        tactic_last_turn_counts = dict(Counter(flat_tactics))

                        assistant_turn_idx += 1

                system_prompt = build_tactic_system_prompt(
                    tactic_last_turn_counts, tactic_history_counts, TACTIC_NAMES
                )
                extra_info = {
                    "prompt_text": prompt_text,
                    "tactic_history_counts": tactic_history_counts,
                    "tactic_last_turn_counts": tactic_last_turn_counts,
                    "tactic_history_bigrams": tactic_history_bigrams,
                    "tactic_history_trigrams": tactic_history_trigrams,
                    "num_previous_assistant_turns": assistant_turn_idx,
                }
            else:
                system_prompt = MINIMAL_SYSTEM_PROMPT
                num_prev_assistant = sum(1 for msg in history if msg.get("role") == "supporter")
                extra_info = {
                    "prompt_text": prompt_text,
                    "num_previous_assistant_turns": num_prev_assistant,
                }

            prompt_with_system = [
                {"role": "system", "content": system_prompt}
            ] + prompt_messages

            rows.append({
                "data_source": "tactic_diversity",
                "prompt": prompt_with_system,
                "ability": "empathy",
                "reward_model": {"style": "rule", "ground_truth": ""},
                "extra_info": extra_info,
            })

    return rows


def main(_):
    cprint(f"Reading input: {FLAGS.input_file}", "cyan", force_color=True)
    with open(FLAGS.input_file, "r") as f:
        data = json.load(f)
    cprint(f"Loaded {len(data)} conversations", "cyan", force_color=True)
    cprint(f"Include tactics: {FLAGS.include_tactics}", "yellow", force_color=True)

    rows = process_conversations(data, FLAGS.include_tactics)
    cprint(f"Created {len(rows)} training examples", "green", force_color=True)

    # Write parquet
    os.makedirs(FLAGS.output_dir, exist_ok=True)
    train_path = os.path.join(FLAGS.output_dir, "train.parquet")
    pd.DataFrame(rows).to_parquet(train_path)
    cprint(f"Written: {train_path}", "green", force_color=True)

    # Verify round-trip
    cprint("\nVerifying round-trip...", "yellow", force_color=True)
    df = pd.read_parquet(train_path)
    row = df.iloc[0]
    cprint(f"  data_source: {row['data_source']}", "cyan", force_color=True)
    cprint(f"  prompt type: {type(row['prompt'])}, len: {len(row['prompt'])}", "cyan", force_color=True)
    cprint(f"  prompt[0] (system): {row['prompt'][0]['content'][:200]}...", "cyan", force_color=True)
    cprint(f"  reward_model: {row['reward_model']}", "cyan", force_color=True)
    cprint(f"  extra_info keys: {list(row['extra_info'].keys())}", "cyan", force_color=True)
    cprint(f"  extra_info.prompt_text: {row['extra_info']['prompt_text'][:200]}...", "cyan", force_color=True)

    cprint("\nDone!", "green", force_color=True)


if __name__ == "__main__":
    app.run(main)
