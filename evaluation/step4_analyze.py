"""Step 4: Cross-method analysis with significance testing.

Computes all Table 4 metrics (empathy, word count, tactic diversity, stickiness)
with paired bootstrap significance tests against the Vanilla baseline.

Usage:
    python3 step4_analyze.py --config config.yml
"""

import csv
import json
import os
from collections import defaultdict

import nltk
import numpy as np
from absl import app, flags
from termcolor import cprint

FLAGS = flags.FLAGS
flags.DEFINE_string("config", "config.yml", "Path to config YAML")

EMPATHY_COLS = [
    "validating_emotions", "encouraging_elaboration", "demonstrating_understanding",
    "advice_giving", "self_oriented", "dismissing_emotions", "aggregated_empathy",
]

TACTICS = [
    "information", "assistance", "advice", "validation",
    "emotional_expression", "paraphrasing", "self_disclosure",
    "questioning", "reappraisal", "empowerment",
]
TACTIC_SET = set(TACTICS)

# Table 4 row order: (display_name, output_folder, model_size, is_vanilla_baseline)
# Numbering matches paper Section 6.3: baselines 1-8, ours 1-3
METHODS = [
    # 1.7B
    ("1.7B 1) Vanilla",              "baseline1_vanilla_Qwen3-1.7B",              "1.7B", True),
    ("1.7B 2) Tactic Prompt",        "baseline2_tactic_prompt_Qwen3-1.7B",        "1.7B", False),
    ("1.7B 3) Tactic+History",       "baseline3_tactic_history_Qwen3-1.7B",       "1.7B", False),
    ("1.7B 4) VS (Vanilla)",         "baseline4_vs_vanilla_Qwen3-1.7B",           "1.7B", False),
    ("1.7B 5) VS (Tactic)",          "baseline5_vs_tactic_Qwen3-1.7B",            "1.7B", False),
    ("1.7B 6) VS (Tactic+History)",  "baseline6_vs_tactic_history_Qwen3-1.7B",    "1.7B", False),
    ("1.7B 7) PsychoCounsel",        "baseline7_psychocounsel_Qwen3-1.7B",        "1.7B", False),
    ("1.7B 8) R1-Zero-Div",          "baseline8_r1zerodiv_Qwen3-1.7B",            "1.7B", False),
    ("1.7B Ours 1) Q+DKL",           "ours1_q_dkl_Qwen3-1.7B",                   "1.7B", False),
    ("1.7B Ours 2) Q+H",             "ours2_q_h_Qwen3-1.7B",                     "1.7B", False),
    ("1.7B Ours 3) Q+DKL+H",         "ours3_q_dkl_h_Qwen3-1.7B",                "1.7B", False),
    # 4B
    ("4B 1) Vanilla",                "baseline1_vanilla_Qwen3-4B",                "4B", True),
    ("4B 2) Tactic Prompt",          "baseline2_tactic_prompt_Qwen3-4B",          "4B", False),
    ("4B 3) Tactic+History",         "baseline3_tactic_history_Qwen3-4B",         "4B", False),
    ("4B 4) VS (Vanilla)",           "baseline4_vs_vanilla_Qwen3-4B",             "4B", False),
    ("4B 5) VS (Tactic)",            "baseline5_vs_tactic_Qwen3-4B",              "4B", False),
    ("4B 6) VS (Tactic+History)",    "baseline6_vs_tactic_history_Qwen3-4B",      "4B", False),
    ("4B 7) PsychoCounsel",          "baseline7_psychocounsel_Qwen3-4B",          "4B", False),
    ("4B 8) R1-Zero-Div",            "baseline8_r1zerodiv_Qwen3-4B",              "4B", False),
    ("4B Ours 1) Q+DKL",             "ours1_q_dkl_Qwen3-4B",                     "4B", False),
    ("4B Ours 2) Q+H",               "ours2_q_h_Qwen3-4B",                       "4B", False),
    ("4B Ours 3) Q+DKL+H",           "ours3_q_dkl_h_Qwen3-4B",                  "4B", False),
]

VANILLA_FOLDERS = {
    "1.7B": "baseline1_vanilla_Qwen3-1.7B",
    "4B":   "baseline1_vanilla_Qwen3-4B",
}

# Significance direction: higher-is-better vs lower-is-better
POSITIVE_DIMS = {"validating_emotions", "encouraging_elaboration",
                 "demonstrating_understanding", "aggregated_empathy"}
NEGATIVE_DIMS = {"advice_giving", "self_oriented", "dismissing_emotions", "stickiness"}

TABLE_COLS = ["words", "validating_emotions", "encouraging_elaboration",
              "demonstrating_understanding", "advice_giving", "self_oriented",
              "dismissing_emotions", "aggregated_empathy", "tac_per_turn", "stickiness"]
COL_SHORT = {
    "words": "Words", "validating_emotions": "Valid.", "encouraging_elaboration": "Elab.",
    "demonstrating_understanding": "Underst.", "advice_giving": "Unsol.Adv.",
    "self_oriented": "Self-Or.", "dismissing_emotions": "Dismiss.",
    "aggregated_empathy": "Agg.", "tac_per_turn": "Tac/Turn", "stickiness": "Stick.",
}


# ── Data loading ──────────────────────────────────────────────

def load_ratings(eval_dir, folder):
    path = os.path.join(eval_dir, "eval_outputs", folder, "gpt-oss-120b_ratings.csv")
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            row = {}
            for col in EMPATHY_COLS:
                val = r[col].strip()
                row[col] = float(val) if val else np.nan
            rows.append(row)
    return rows


def load_tagged(eval_dir, folder):
    path = os.path.join(eval_dir, "outputs", folder, "conversations_tagged.json")
    with open(path) as f:
        return json.load(f)


def load_conversations(eval_dir, folder):
    path = os.path.join(eval_dir, "outputs", folder, "conversations.json")
    with open(path) as f:
        return json.load(f)


# ── Metric arrays (per-turn, for bootstrap) ───────────────────

def empathy_arrays(ratings):
    return {col: np.array([r[col] for r in ratings]) for col in EMPATHY_COLS}


def words_array(conversations):
    return np.array([len(nltk.word_tokenize(item["model_response"]))
                     for item in conversations], dtype=float)


def tac_per_turn_array(tagged):
    return np.array([sum(1 for t, c in item.get("tactic_counts", {}).items()
                         if t in TACTIC_SET and c > 0)
                     for item in tagged], dtype=float)


# ── Stickiness ────────────────────────────────────────────────

def group_by_conversation(tagged):
    convs = defaultdict(list)
    for item in tagged:
        convs[item["conversation_id"]].append(item)
    for cid in convs:
        convs[cid].sort(key=lambda x: len(x.get("conversation_history", [])))
    return dict(convs)


def tactic_set_of(item):
    tc = item.get("tactic_counts", {})
    return {t for t, c in tc.items() if t in TACTIC_SET and c > 0}


def build_stickiness_pairs(gold_tagged, method_tagged):
    gold_convs = group_by_conversation(gold_tagged)
    method_convs = group_by_conversation(method_tagged)
    pairs = []
    for cid in sorted(gold_convs):
        g_turns = gold_convs[cid]
        m_turns = method_convs[cid]
        for i in range(1, len(g_turns)):
            pairs.append((tactic_set_of(g_turns[i - 1]), tactic_set_of(m_turns[i])))
    return pairs


def stickiness_from_pairs(pairs):
    hits = {t: [] for t in TACTICS}
    for prev_gold, curr_model in pairs:
        for t in TACTICS:
            if t in prev_gold:
                hits[t].append(1 if t in curr_model else 0)
    probs = [np.mean(hits[t]) for t in TACTICS if hits[t]]
    return np.mean(probs) if probs else 0.0


def stickiness_from_indices(pairs, indices):
    hits = {t: [] for t in TACTICS}
    for idx in indices:
        prev_gold, curr_model = pairs[idx]
        for t in TACTICS:
            if t in prev_gold:
                hits[t].append(1 if t in curr_model else 0)
    probs = [np.mean(hits[t]) for t in TACTICS if hits[t]]
    return np.mean(probs) if probs else 0.0


# ── Paired bootstrap ─────────────────────────────────────────

def bootstrap_means(method_arr, vanilla_arr, n_boot=10000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(method_arr)
    obs = np.nanmean(method_arr) - np.nanmean(vanilla_arr)
    flips = sum(
        1 for _ in range(n_boot)
        if (lambda d: (obs > 0 and d < 0) or (obs < 0 and d > 0) or obs == 0)(
            np.nanmean(method_arr[(idx := rng.randint(0, n, n))]) -
            np.nanmean(vanilla_arr[idx])
        )
    )
    return flips / n_boot, obs


def bootstrap_stickiness(m_pairs, v_pairs, n_boot=10000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(m_pairs)
    obs = stickiness_from_pairs(m_pairs) - stickiness_from_pairs(v_pairs)
    flips = sum(
        1 for _ in range(n_boot)
        if (lambda d: (obs > 0 and d < 0) or (obs < 0 and d > 0) or obs == 0)(
            stickiness_from_indices(m_pairs, (idx := rng.randint(0, n, n))) -
            stickiness_from_indices(v_pairs, idx)
        )
    )
    return flips / n_boot, obs


# ── Significance markers ─────────────────────────────────────

def stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""


def sig_marker(p, diff, dim):
    s = stars(p)
    if not s:
        return ""
    if dim in POSITIVE_DIMS:
        return f"\\sym{{{s}}}" if diff > 0 else f"\\losssym{{{s}}}"
    if dim in NEGATIVE_DIMS:
        return f"\\sym{{{s}}}" if diff < 0 else f"\\losssym{{{s}}}"
    return ""


# ── Main ──────────────────────────────────────────────────────

def main(_):
    eval_dir = os.path.dirname(os.path.abspath(FLAGS.config))

    cprint("Loading gold...", "yellow", force_color=True)
    gold_tagged = load_tagged(eval_dir, "gold")
    gold_ratings = load_ratings(eval_dir, "gold")

    cprint(f"Loading {len(METHODS)} methods...", "yellow", force_color=True)
    data = {}
    for display, folder, size, is_vanilla in METHODS:
        cprint(f"  {folder}", "white", force_color=True)
        ratings = load_ratings(eval_dir, folder)
        tagged = load_tagged(eval_dir, folder)
        convs = load_conversations(eval_dir, folder)
        data[folder] = {
            "emp": empathy_arrays(ratings),
            "words": words_array(convs),
            "tac": tac_per_turn_array(tagged),
            "stick_pairs": build_stickiness_pairs(gold_tagged, tagged),
        }
        data[folder]["stick_val"] = stickiness_from_pairs(data[folder]["stick_pairs"])

    # Gold reference values
    gold_emp = empathy_arrays(gold_ratings)
    gold_convs = load_conversations(eval_dir, "gold")
    gold_words = words_array(gold_convs)
    gold_tac = tac_per_turn_array(gold_tagged)
    gold_stick_pairs = build_stickiness_pairs(gold_tagged, gold_tagged)
    gold_stick_val = stickiness_from_pairs(gold_stick_pairs)

    # Compute values + significance for each method
    cprint("\nRunning paired bootstrap (10k resamples)...", "yellow", force_color=True)
    results = {}  # (display, col_short) -> (value_str, marker, p_str)

    # Gold row
    for col in TABLE_COLS:
        if col == "words":
            v = np.mean(gold_words)
            results[("Gold (Human)", COL_SHORT[col])] = (f"{v:.1f}", "", "")
        elif col in EMPATHY_COLS:
            v = np.nanmean(gold_emp[col])
            results[("Gold (Human)", COL_SHORT[col])] = (f"{v:.2f}", "", "")
        elif col == "tac_per_turn":
            v = np.mean(gold_tac)
            results[("Gold (Human)", COL_SHORT[col])] = (f"{v:.2f}", "", "")
        elif col == "stickiness":
            results[("Gold (Human)", COL_SHORT[col])] = (f"{gold_stick_val:.2f}", "", "")

    # Model rows
    for display, folder, size, is_vanilla in METHODS:
        d = data[folder]
        v_folder = VANILLA_FOLDERS[size]
        vd = data[v_folder]

        for col in TABLE_COLS:
            if col == "words":
                val = np.mean(d["words"])
                val_str = f"{val:.1f}"
                marker, p_str = "", ""
            elif col in EMPATHY_COLS:
                val = np.nanmean(d["emp"][col])
                val_str = f"{val:.2f}"
                if not is_vanilla:
                    p, diff = bootstrap_means(d["emp"][col], vd["emp"][col])
                    marker = sig_marker(p, diff, col)
                    p_str = f"p={p:.4f}"
                else:
                    marker, p_str = "", ""
            elif col == "tac_per_turn":
                val = np.mean(d["tac"])
                val_str = f"{val:.2f}"
                marker, p_str = "", ""
            elif col == "stickiness":
                val = d["stick_val"]
                val_str = f"{val:.2f}"
                if not is_vanilla:
                    p, diff = bootstrap_stickiness(d["stick_pairs"], vd["stick_pairs"])
                    marker = sig_marker(p, diff, col)
                    p_str = f"p={p:.4f}"
                else:
                    marker, p_str = "", ""

            results[(display, COL_SHORT[col])] = (val_str, marker, p_str)

        cprint(f"  {display} done", "green", force_color=True)

    # Print compact table
    col_names = [COL_SHORT[c] for c in TABLE_COLS]
    header = f"{'Method':<30}"
    for cn in col_names:
        header += f" {cn:>16}"
    sep = "=" * (30 + 17 * len(col_names))

    cprint(f"\n{sep}", "cyan", force_color=True)
    cprint("TABLE 4: FULL RESULTS WITH SIGNIFICANCE", "cyan", force_color=True)
    cprint(sep, "cyan", force_color=True)
    cprint(header, "cyan", force_color=True)
    cprint("-" * len(sep), "cyan", force_color=True)

    # Gold row
    row_str = f"{'Gold (Human)':<30}"
    for cn in col_names:
        val_str, marker, _ = results[("Gold (Human)", cn)]
        cell = f"{val_str} {marker}".strip()
        row_str += f" {cell:>16}"
    cprint(row_str, "yellow", force_color=True)
    cprint("-" * len(sep), "cyan", force_color=True)

    # Method rows
    prev_size = None
    for display, folder, size, is_vanilla in METHODS:
        if size != prev_size and prev_size is not None:
            cprint("-" * len(sep), "cyan", force_color=True)
        prev_size = size

        row_str = f"{display:<30}"
        for cn in col_names:
            val_str, marker, _ = results[(display, cn)]
            cell = f"{val_str} {marker}".strip()
            row_str += f" {cell:>16}"
        color = "white" if is_vanilla else "green"
        cprint(row_str, color, force_color=True)

    cprint(sep, "cyan", force_color=True)

    # Detailed p-values
    cprint("\nDETAILED P-VALUES (non-baseline methods only):", "yellow", force_color=True)
    cprint(f"{'Method':<30} {'Column':<12} {'Value':<8} {'Marker':<20} {'p-value':<12}", "yellow", force_color=True)
    cprint("-" * 82, "yellow", force_color=True)
    for display, folder, size, is_vanilla in METHODS:
        if is_vanilla:
            continue
        for col in TABLE_COLS:
            cn = COL_SHORT[col]
            val_str, marker, p_str = results[(display, cn)]
            if p_str:
                m_disp = marker if marker else "(none)"
                cprint(f"{display:<30} {cn:<12} {val_str:<8} {m_disp:<20} {p_str:<12}", "white", force_color=True)

    cprint("\nStep 4 analysis done.", "green", force_color=True)


if __name__ == "__main__":
    app.run(main)
