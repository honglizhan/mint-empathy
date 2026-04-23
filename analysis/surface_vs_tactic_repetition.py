"""Compare surface-level similarity vs tactic-level stickiness between consecutive turns.

Shows that lexical overlap and embedding cosine similarity fail to distinguish
human from LLM supporters, while tactic stickiness reveals a dramatic gap.

For the eval set: compares gold conversation_history response at t-1 (what the
model was conditioned on) vs model_response at t (what it generated).
For the training set: compares consecutive supporter turns from real multi-turn
conversations (where turn t was conditioned on turn t-1).
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from termcolor import cprint
from absl import app, flags
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score

FLAGS = flags.FLAGS
flags.DEFINE_string("training_json",
                    "../data/training/conversations_322_tagged.json",
                    "Path to training data tactics-tagged conversations")
flags.DEFINE_string("gold_json",
                    "../evaluation/outputs/gold/conversations_tagged.json",
                    "Path to gold (human) tagged conversations")
flags.DEFINE_string("vanilla_17b_json",
                    "../evaluation/outputs/baseline1_vanilla_Qwen3-1.7B/conversations_tagged.json",
                    "Path to Vanilla Qwen3-1.7B tagged conversations")
flags.DEFINE_string("vanilla_4b_json",
                    "../evaluation/outputs/baseline1_vanilla_Qwen3-4B/conversations_tagged.json",
                    "Path to Vanilla Qwen3-4B tagged conversations")
flags.DEFINE_string("output_dir", "results",
                    "Output directory for results")
flags.DEFINE_integer("bertscore_batch_size", 64,
                     "Batch size for BERTScore computation")

TACTIC_NAMES = [
    "advice", "information", "paraphrasing", "reappraisal", "validation",
    "empowerment", "questioning", "emotional_expression", "assistance",
    "self_disclosure",
]


def bertscore_pairs(text_pairs):
    """Compute BERTScore F1 for a list of (reference, candidate) pairs."""
    if not text_pairs:
        return []
    refs = [a for a, b in text_pairs]
    cands = [b for a, b in text_pairs]
    cprint(f"    Computing BERTScore for {len(refs)} pairs...", "yellow", force_color=True)
    P, R, F1 = bert_score(cands, refs, lang="en",
                           batch_size=FLAGS.bertscore_batch_size,
                           verbose=False)
    return [float(f) for f in F1]


def word_bigrams(text):
    """Extract word bigrams from text."""
    words = text.lower().split()
    return [tuple(words[i:i+2]) for i in range(len(words) - 1)]


def bigram_overlap_pct(text_a, text_b):
    """Fraction of bigrams in text_a that also appear in text_b (multiset)."""
    from collections import Counter
    a = Counter(word_bigrams(text_a))
    b = Counter(word_bigrams(text_b))
    denom = sum(a.values())
    if denom == 0:
        return 0.0
    numer = sum(min(a[k], b[k]) for k in a)
    return numer / denom


def bleu2(text_a, text_b):
    """BLEU-2 score: text_a is reference, text_b is hypothesis."""
    ref = text_a.lower().split()
    hyp = text_b.lower().split()
    if not ref or not hyp:
        return 0.0
    smooth = SmoothingFunction().method1
    return sentence_bleu([ref], hyp, weights=(0.5, 0.5), smoothing_function=smooth)


def build_gold_lookup(gold_path):
    """Build lookup: (conversation_id, turn_position) -> (text, tactic_set).

    turn_position is determined by conversation_history length.
    """
    with open(gold_path) as f:
        data = json.load(f)

    convos = defaultdict(list)
    for t in data:
        convos[t["conversation_id"]].append(t)

    lookup = {}
    for cid, turns in convos.items():
        turns.sort(key=lambda x: len(x["conversation_history"]))
        for pos, t in enumerate(turns):
            text = t.get("model_response", "")
            tacs = set(t.get("tactic_counts", {}).keys())
            lookup[(cid, pos)] = (text, tacs)

    return lookup


def extract_pairs_eval(model_path, gold_lookup):
    """Extract (gold_response_at_t-1, model_response_at_t) pairs from eval set.

    For each turn at position > 0, the pair is:
      - text_a: gold supporter response at t-1 (what was in conversation_history)
      - text_b: model response at t (what the model generated)
      - tac_a: gold tactics at t-1
      - tac_b: model tactics at t
    """
    with open(model_path) as f:
        data = json.load(f)

    convos = defaultdict(list)
    for t in data:
        convos[t["conversation_id"]].append(t)

    pairs = []
    for cid, turns in convos.items():
        turns.sort(key=lambda x: len(x["conversation_history"]))
        for i in range(1, len(turns)):
            # Gold response at t-1 (from gold lookup)
            gold_prev = gold_lookup.get((cid, i - 1))
            if gold_prev is None:
                continue
            text_a, tac_a = gold_prev

            # Model response at t
            text_b = turns[i].get("model_response", "")
            tac_b = set(turns[i].get("tactic_counts", {}).keys())

            if text_a.strip() and text_b.strip():
                pairs.append((text_a, text_b, tac_a, tac_b))

    return pairs


def extract_pairs_training(filepath):
    """Extract consecutive supporter turn pairs from training data (nested format).

    These are real multi-turn conversations: turn t was conditioned on turn t-1.
    """
    with open(filepath) as f:
        data = json.load(f)

    model_name_map = {
        "gpt-3.5-turbo": "GPT-3.5-turbo",
        "gpt-4": "GPT-4",
        "GPT4": "GPT-4",
        "GPT4-empathy": "GPT-4",
        "Llama2-70b": "Llama-2-70B",
        "IC": "GPT-3.5-turbo",
    }

    model_pairs = defaultdict(list)
    for conv in data:
        model = conv.get("model", "unknown")
        mapped = model_name_map.get(model, model)
        if mapped is None:
            continue

        supporter_turns = []
        for turn in conv["conversation"]:
            if turn["role"] != "supporter":
                continue
            text = turn.get("content", "")
            tactics = set()
            for st in turn.get("sentence_tactics", []):
                for t in st.get("tactics", []):
                    if t in TACTIC_NAMES:
                        tactics.add(t)
            supporter_turns.append((text, tactics))

        for i in range(1, len(supporter_turns)):
            t_a, tac_a = supporter_turns[i - 1]
            t_b, tac_b = supporter_turns[i]
            if t_a.strip() and t_b.strip():
                model_pairs[mapped].append((t_a, t_b, tac_a, tac_b))

    return model_pairs


def compute_metrics(pairs, source_name=""):
    """Compute all metrics for a list of (text_prev, text_curr, tac_prev, tac_curr) pairs."""
    cprint(f"  Computing metrics for {source_name}...", "yellow", force_color=True)

    # Lexical: bigram overlap % and BLEU-2
    bigram_scores = [bigram_overlap_pct(a, b) for a, b, _, _ in pairs]
    bleu_scores = [bleu2(a, b) for a, b, _, _ in pairs]

    # Semantic: BERTScore F1
    text_pairs = [(a, b) for a, b, _, _ in pairs]
    bertscore_f1 = bertscore_pairs(text_pairs)

    # Tactic stickiness: per-tactic conditional probability P(tac in t | tac in t-1),
    # averaged across tactics. Matches plot_human_vs_llm_stickiness.py methodology.
    pp = defaultdict(int)       # tactic present in both t-1 and t
    p_count = defaultdict(int)  # tactic present in t-1
    for _, _, tac_a, tac_b in pairs:
        for tac in TACTIC_NAMES:
            if tac in tac_a:
                p_count[tac] += 1
                if tac in tac_b:
                    pp[tac] += 1
    per_tactic_stickiness = {}
    for tac in TACTIC_NAMES:
        per_tactic_stickiness[tac] = pp[tac] / p_count[tac] if p_count[tac] > 0 else 0.0
    avg_stickiness = np.mean(list(per_tactic_stickiness.values()))

    return {
        "bigram_overlap": np.mean(bigram_scores) if bigram_scores else 0.0,
        "bigram_std": np.std(bigram_scores) if bigram_scores else 0.0,
        "bleu2": np.mean(bleu_scores) if bleu_scores else 0.0,
        "bleu2_std": np.std(bleu_scores) if bleu_scores else 0.0,
        "bertscore_f1": np.mean(bertscore_f1) if bertscore_f1 else 0.0,
        "bertscore_std": np.std(bertscore_f1) if bertscore_f1 else 0.0,
        "tactic_stickiness": float(avg_stickiness),
        "per_tactic_stickiness": per_tactic_stickiness,
        "n_pairs": len(pairs),
    }


def main(_):
    cprint("\n=== Surface Similarity vs Tactic Stickiness ===", "cyan", force_color=True)
    cprint("Comparison: gold response at t-1 vs model response at t", "cyan", force_color=True)

    out_dir = Path(FLAGS.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # Build gold lookup for eval set
    cprint("Building gold lookup...", "yellow", force_color=True)
    gold_lookup = build_gold_lookup(FLAGS.gold_json)

    # Human gold (eval set): gold[t-1] vs gold[t]
    cprint("Loading human gold...", "yellow", force_color=True)
    human_pairs = extract_pairs_eval(FLAGS.gold_json, gold_lookup)
    results["Human"] = compute_metrics(human_pairs, "Human")

    # LLM vanilla (eval set): gold[t-1] vs LLM[t]
    cprint("Loading Qwen vanilla models...", "yellow", force_color=True)
    for name, path in [("Qwen3-1.7B", FLAGS.vanilla_17b_json),
                        ("Qwen3-4B", FLAGS.vanilla_4b_json)]:
        pairs = extract_pairs_eval(path, gold_lookup)
        results[name] = compute_metrics(pairs, name)

    # Training data: real multi-turn conversations, turn[t-1] vs turn[t]
    cprint("Loading training data models...", "yellow", force_color=True)
    training_pairs = extract_pairs_training(FLAGS.training_json)
    for model_name, pairs in training_pairs.items():
        results[model_name] = compute_metrics(pairs, model_name)

    # Print results table
    cprint("\n  {:20s}  {:>10s}  {:>10s}  {:>10s}  {:>12s}  {:>6s}".format(
        "Source", "Bigram %", "BLEU-2", "BERTScore", "Tac Sticky", "Pairs"),
        "yellow", force_color=True)
    cprint("  " + "-" * 76, "yellow", force_color=True)

    for name in ["Human"] + [n for n in results if n != "Human"]:
        r = results[name]
        cprint("  {:20s}  {:>10.3f}  {:>10.3f}  {:>10.3f}  {:>12.3f}  {:>6d}".format(
            name, r["bigram_overlap"], r["bleu2"], r["bertscore_f1"],
            r["tactic_stickiness"], r["n_pairs"]),
            "green" if name == "Human" else "cyan", force_color=True)

    # Save results as JSON
    out_json = out_dir / "surface_vs_tactic_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    cprint(f"\n  Saved results to {out_json}", "green", force_color=True)

    cprint("\nDone!", "cyan", force_color=True)


if __name__ == "__main__":
    app.run(main)
