"""
Step 0: Preprocess gold (human) conversations from the Lend-an-Ear test set.

Converts the 50 Lend-an-Ear gold conversations into the same flat format used
by model outputs, so all downstream steps (empathy eval, analysis) work identically.

For each supporter turn:
  1. Parse conversation from CSV into structured turns
  2. Build conversation_history (all turns before it)
  3. Extract model_response (gold supporter text)
  4. Extract gold tactic tags from sentence_level_tactics.csv

Outputs:
  outputs/gold/conversations.json         (same schema as model outputs)
  outputs/gold/conversations_tagged.json  (adds sentence_tactics + tactic_counts)

Usage:
    python preprocess_gold.py --config config.yml
"""

import csv
import json
import os
import sys
from collections import defaultdict

import yaml
from absl import app, flags
from termcolor import cprint

FLAGS = flags.FLAGS

flags.DEFINE_string("config", "config.yml", "Path to config YAML")

ROLE_MAP = {"seeker": "user", "supporter": "assistant"}

# The 10 VERL-compatible tactics (subset of the 15 in the tactic CSV)
VERL_TACTICS = [
    "information", "assistance", "advice", "validation", "emotional_expression",
    "paraphrasing", "self_disclosure", "questioning", "reappraisal", "empowerment",
]


def parse_conversation(full_text):
    """Parse 'Seeker: ...\nSupporter: ...' text into structured turns."""
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
            # Continuation line
            current_lines.append(line)

    if current_role:
        turns.append({"role": current_role, "content": "\n".join(current_lines)})

    return turns


def load_tactic_tags(csv_path):
    """Load sentence-level tactic tags, grouped by (conversation_id, supporter_turn_index).

    Returns dict: (conv_id, turn_idx) -> {
        "sentence_tactics": [[tactic, ...], ...],  # per-sentence, ordered by sentenceRANK
        "tactic_counts": {tactic: count, ...},      # total counts across sentences
    }
    """
    # Group rows by (conv_id, turn_idx), keeping sentenceRANK for ordering
    raw = defaultdict(list)
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            conv_id = row["conversation_id"]
            turn_idx = int(row["supporter_turn_index"])
            rank = int(row["sentenceRANK"])
            # Collect tactics present in this sentence
            tactics = sorted(t for t in VERL_TACTICS if int(row.get(t, 0)) > 0)
            raw[(conv_id, turn_idx)].append((rank, tactics))

    # Build structured output per turn
    tags = {}
    for key, rows in raw.items():
        rows.sort(key=lambda x: x[0])  # sort by sentenceRANK
        sentence_tactics = [tactics for _, tactics in rows]
        # Count total occurrences across all sentences
        tactic_counts = {}
        for tactics in sentence_tactics:
            for t in tactics:
                tactic_counts[t] = tactic_counts.get(t, 0) + 1
        tags[key] = {
            "sentence_tactics": sentence_tactics,
            "tactic_counts": tactic_counts,
        }
    return tags


def main(_):
    with open(FLAGS.config) as f:
        cfg = yaml.safe_load(f)

    config_dir = os.path.dirname(os.path.abspath(FLAGS.config))
    conversations_csv = cfg["test_set"]["conversations_csv"]
    tactics_csv = cfg["test_set"]["sentence_level_tactics_csv"]

    # Load conversations
    cprint(f"Loading conversations from {conversations_csv}...", "cyan", force_color=True)
    with open(conversations_csv, newline="") as f:
        reader = csv.DictReader(f)
        conv_rows = list(reader)
    cprint(f"  Loaded {len(conv_rows)} conversations", "green", force_color=True)

    # Load tactic tags
    cprint(f"Loading tactic tags from {tactics_csv}...", "cyan", force_color=True)
    tactic_tags = load_tactic_tags(tactics_csv)
    cprint(f"  Loaded tags for {len(tactic_tags)} supporter turns", "green", force_color=True)

    # Process each conversation
    entries = []
    entries_tagged = []
    total_skipped = 0

    for conv_row in conv_rows:
        conv_id = conv_row["conversation_id"]
        turns = parse_conversation(conv_row["full_conversation"])

        supporter_turn_idx = 0
        for i, turn in enumerate(turns):
            if turn["role"] != "supporter":
                continue
            supporter_turn_idx += 1

            # Build conversation history (all prior turns)
            history = []
            for prev in turns[:i]:
                role = ROLE_MAP.get(prev["role"], prev["role"])
                history.append({"role": role, "content": prev["content"]})

            # Skip if no seeker message in history
            if not any(m["role"] == "user" for m in history):
                total_skipped += 1
                continue

            entry = {
                "conversation_id": conv_id,
                "source": "lend_an_ear_gold",
                "model": "human",
                "system_prompt": "",
                "conversation_history": history,
                "reasoning": "",
                "model_response": turn["content"],
            }
            entries.append(entry)

            # Tagged version: add sentence_tactics + tactic_counts from the tactic CSV
            tagged_entry = dict(entry)
            key = (conv_id, supporter_turn_idx)
            tag_data = tactic_tags.get(key, {"sentence_tactics": [], "tactic_counts": {}})
            tagged_entry["sentence_tactics"] = tag_data["sentence_tactics"]
            tagged_entry["tactic_counts"] = tag_data["tactic_counts"]
            entries_tagged.append(tagged_entry)

    # Save outputs
    output_dir = os.path.join(config_dir, "outputs", "gold")
    os.makedirs(output_dir, exist_ok=True)

    out_path = os.path.join(output_dir, "conversations.json")
    with open(out_path, "w") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)
    cprint(f"  Saved {len(entries)} entries to {out_path}", "green", force_color=True)

    out_tagged_path = os.path.join(output_dir, "conversations_tagged.json")
    with open(out_tagged_path, "w") as f:
        json.dump(entries_tagged, f, indent=2, ensure_ascii=False)
    cprint(f"  Saved {len(entries_tagged)} tagged entries to {out_tagged_path}", "green", force_color=True)

    if total_skipped > 0:
        cprint(f"  Skipped {total_skipped} turns (no prior seeker message)", "yellow", force_color=True)

    # Stats
    tagged_count = sum(1 for e in entries_tagged if e["tactic_counts"])
    total_tactics = sum(sum(e["tactic_counts"].values()) for e in entries_tagged)
    total_sentences = sum(len(e["sentence_tactics"]) for e in entries_tagged)
    cprint(f"\nStats:", "cyan", force_color=True)
    cprint(f"  Total entries: {len(entries)}", "cyan", force_color=True)
    cprint(f"  Entries with tactics: {tagged_count}", "cyan", force_color=True)
    cprint(f"  Total tactic instances: {total_tactics}", "cyan", force_color=True)
    cprint(f"  Total sentences: {total_sentences}", "cyan", force_color=True)
    cprint(f"  Mean tactics/turn: {total_tactics / max(len(entries), 1):.2f}", "cyan", force_color=True)

    cprint("\nDone!", "green", force_color=True)


if __name__ == "__main__":
    app.run(main)
