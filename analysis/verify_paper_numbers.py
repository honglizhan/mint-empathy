"""Verify paper-facing numbers against the public repo artifacts.

This script intentionally ignores the Table 2 word-count column, which is
tracked separately as a known mismatch between the paper snapshot and the
current public repo outputs.
"""

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from absl import app, flags
from termcolor import cprint

FLAGS = flags.FLAGS
flags.DEFINE_string("config", "analysis/paper_numbers.yml", "Path to YAML config")

TACTICS = [
    "information",
    "assistance",
    "advice",
    "validation",
    "emotional_expression",
    "paraphrasing",
    "self_disclosure",
    "questioning",
    "reappraisal",
    "empowerment",
]
TACTIC_SET = set(TACTICS)

TABLE_FOLDER_MAP = {
    "Gold": "gold",
    "1.7B Vanilla": "baseline1_vanilla_Qwen3-1.7B",
    "1.7B Tactic Prompt": "baseline2_tactic_prompt_Qwen3-1.7B",
    "1.7B Tactic+History": "baseline3_tactic_history_Qwen3-1.7B",
    "1.7B VS (Vanilla)": "baseline4_vs_vanilla_Qwen3-1.7B",
    "1.7B VS (Tactic)": "baseline5_vs_tactic_Qwen3-1.7B",
    "1.7B VS (Tactic+History)": "baseline6_vs_tactic_history_Qwen3-1.7B",
    "1.7B PsychoCounsel": "baseline7_psychocounsel_Qwen3-1.7B",
    "1.7B R1-Zero-Div": "baseline8_r1zerodiv_Qwen3-1.7B",
    "1.7B Ours Q+DKL": "ours1_q_dkl_Qwen3-1.7B",
    "1.7B Ours Q+H": "ours2_q_h_Qwen3-1.7B",
    "1.7B Ours Q+DKL+H": "ours3_q_dkl_h_Qwen3-1.7B",
    "4B Vanilla": "baseline1_vanilla_Qwen3-4B",
    "4B Tactic Prompt": "baseline2_tactic_prompt_Qwen3-4B",
    "4B Tactic+History": "baseline3_tactic_history_Qwen3-4B",
    "4B VS (Vanilla)": "baseline4_vs_vanilla_Qwen3-4B",
    "4B VS (Tactic)": "baseline5_vs_tactic_Qwen3-4B",
    "4B VS (Tactic+History)": "baseline6_vs_tactic_history_Qwen3-4B",
    "4B PsychoCounsel": "baseline7_psychocounsel_Qwen3-4B",
    "4B R1-Zero-Div": "baseline8_r1zerodiv_Qwen3-4B",
    "4B Ours Q+DKL": "ours1_q_dkl_Qwen3-4B",
    "4B Ours Q+H": "ours2_q_h_Qwen3-4B",
    "4B Ours Q+DKL+H": "ours3_q_dkl_h_Qwen3-4B",
}


def load_config():
    with open(FLAGS.config) as f:
        return yaml.safe_load(f)


def mean_ignore_nan(values):
    clean = [v for v in values if not math.isnan(v)]
    return sum(clean) / len(clean)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_ratings(eval_dir, folder):
    rows = []
    path = eval_dir / "eval_outputs" / folder / "gpt-oss-120b_ratings.csv"
    with open(path) as f:
        for row in csv.DictReader(f):
            parsed = {}
            for key in [
                "validating_emotions",
                "encouraging_elaboration",
                "demonstrating_understanding",
                "advice_giving",
                "self_oriented",
                "dismissing_emotions",
                "aggregated_empathy",
            ]:
                value = row[key].strip()
                parsed[key] = float(value) if value else math.nan
            rows.append(parsed)
    return rows


def load_tagged(eval_dir, folder):
    return load_json(eval_dir / "outputs" / folder / "conversations_tagged.json")


def group_by_conversation(tagged):
    convs = defaultdict(list)
    for item in tagged:
        convs[item["conversation_id"]].append(item)
    for cid in convs:
        convs[cid].sort(key=lambda item: len(item.get("conversation_history", [])))
    return convs


def tactic_set_of(item):
    counts = item.get("tactic_counts", {})
    return {t for t, c in counts.items() if t in TACTIC_SET and c > 0}


def build_stickiness_pairs(gold_tagged, method_tagged):
    gold_convs = group_by_conversation(gold_tagged)
    method_convs = group_by_conversation(method_tagged)
    pairs = []
    for cid in sorted(gold_convs):
        gold_turns = gold_convs[cid]
        method_turns = method_convs[cid]
        for idx in range(1, len(gold_turns)):
            pairs.append((tactic_set_of(gold_turns[idx - 1]), tactic_set_of(method_turns[idx])))
    return pairs


def stickiness_from_pairs(pairs):
    hits = {t: [] for t in TACTICS}
    for previous_gold, current_method in pairs:
        for tactic in TACTICS:
            if tactic not in previous_gold:
                continue
            hits[tactic].append(1 if tactic in current_method else 0)
    probabilities = [sum(values) / len(values) for values in hits.values() if values]
    return sum(probabilities) / len(probabilities)


def tac_per_turn(tagged):
    counts = []
    for item in tagged:
        tactic_count = 0
        for tactic, value in item.get("tactic_counts", {}).items():
            if tactic in TACTIC_SET and value > 0:
                tactic_count += 1
        counts.append(tactic_count)
    return sum(counts) / len(counts)


def compute_table_metrics(eval_dir, folder, gold_tagged):
    ratings = load_ratings(eval_dir, folder)
    tagged = load_tagged(eval_dir, folder)
    metrics = {
        "validating_emotions": mean_ignore_nan([row["validating_emotions"] for row in ratings]),
        "encouraging_elaboration": mean_ignore_nan([row["encouraging_elaboration"] for row in ratings]),
        "demonstrating_understanding": mean_ignore_nan([row["demonstrating_understanding"] for row in ratings]),
        "advice_giving": mean_ignore_nan([row["advice_giving"] for row in ratings]),
        "self_oriented": mean_ignore_nan([row["self_oriented"] for row in ratings]),
        "dismissing_emotions": mean_ignore_nan([row["dismissing_emotions"] for row in ratings]),
        "aggregated_empathy": mean_ignore_nan([row["aggregated_empathy"] for row in ratings]),
        "tac_per_turn": tac_per_turn(tagged),
        "stickiness": stickiness_from_pairs(build_stickiness_pairs(gold_tagged, tagged)),
    }
    return metrics


def round1(value):
    return round(value + 1e-9, 1)


def verify_counts(repo_root, expected, errors):
    train_conversations = load_json(repo_root / "data" / "training" / "conversations_322_tagged.json")
    gold_turns = load_json(repo_root / "evaluation" / "outputs" / "gold" / "conversations.json")
    nonempty_tagger_rows = 0
    with open(repo_root / "data" / "tagger_annotations" / "all_tagged_sentences.csv") as f:
        for row in csv.DictReader(f):
            if row["sentence"].strip():
                nonempty_tagger_rows += 1

    observed = {
        "training_conversations": len(train_conversations),
        "training_supporter_turns": sum(
            1
            for conv in train_conversations
            for turn in conv["conversation"]
            if turn["role"] == "supporter"
        ),
        "sense7_conversations": sum(1 for conv in train_conversations if conv["source"] == "sense-7"),
        "gold_conversations": len({row["conversation_id"] for row in gold_turns}),
        "gold_supporter_turns": len(gold_turns),
        "tagger_sentences_nonempty": nonempty_tagger_rows,
    }

    for key, expected_value in expected.items():
        observed_value = observed[key]
        if observed_value != expected_value:
            errors.append(f"Count mismatch for {key}: expected {expected_value}, got {observed_value}")
            cprint(errors[-1], "red", force_color=True)
            continue
        cprint(f"[ok] {key}: {observed_value}", "green", force_color=True)


def verify_table(eval_dir, expected_rows, tolerance, errors):
    gold_tagged = load_tagged(eval_dir, "gold")
    for label, expected_metrics in expected_rows.items():
        folder = TABLE_FOLDER_MAP[label]
        observed = compute_table_metrics(eval_dir, folder, gold_tagged)
        bad = []
        for key, expected_value in expected_metrics.items():
            diff = abs(observed[key] - expected_value)
            if diff > tolerance:
                bad.append((key, expected_value, observed[key], diff))
        if bad:
            details = "; ".join(
                f"{key}: expected {expected_value:.2f}, got {observed_value:.3f}"
                for key, expected_value, observed_value, _ in bad
            )
            errors.append(f"Table metrics mismatch for {label}: {details}")
            cprint(errors[-1], "red", force_color=True)
            continue
        cprint(f"[ok] table metrics: {label}", "green", force_color=True)


def verify_prevalence(repo_root, expected, errors):
    train_data = load_json(repo_root / "data" / "training" / "conversations_322_tagged.json")
    gold_data = load_json(repo_root / "evaluation" / "outputs" / "gold" / "conversations_tagged.json")

    model_map = {
        "gpt-3.5-turbo": "GPT-3.5",
        "gpt-4": "GPT-4",
        "GPT4": "GPT-4",
        "GPT4-empathy": "GPT-4",
        "Llama2-70b": "Llama-2-70B",
        "IC": "GPT-3.5",
    }

    model_turn_sets = defaultdict(list)
    for conv in train_data:
        model = model_map.get(conv["model"])
        if model is None:
            continue
        turns = []
        for turn in conv["conversation"]:
            if turn["role"] != "supporter":
                continue
            turn_tactics = set()
            for sentence_tactics in turn.get("sentence_tactics", []):
                for tactic in sentence_tactics.get("tactics", []):
                    if tactic in TACTIC_SET:
                        turn_tactics.add(tactic)
            turns.append(turn_tactics)
        model_turn_sets[model].append(turns)

    human_questioning = 100 * sum(
        1 for item in gold_data if item.get("tactic_counts", {}).get("questioning", 0) > 0
    ) / len(gold_data)
    llm_prevalence = defaultdict(dict)
    for model, conversations in model_turn_sets.items():
        all_turns = [turn for conv in conversations for turn in conv]
        for tactic in ["advice", "information", "questioning"]:
            llm_prevalence[model][tactic] = 100 * sum(1 for turn in all_turns if tactic in turn) / len(all_turns)

    observed = {
        "advice_min_percent": round(min(llm_prevalence[model]["advice"] for model in llm_prevalence)),
        "advice_max_percent": round(max(llm_prevalence[model]["advice"] for model in llm_prevalence)),
        "information_min_percent": round(min(llm_prevalence[model]["information"] for model in llm_prevalence)),
        "information_max_percent": round(max(llm_prevalence[model]["information"] for model in llm_prevalence)),
        "questioning_min_percent": round(min(llm_prevalence[model]["questioning"] for model in llm_prevalence)),
        "questioning_max_percent": round(max(llm_prevalence[model]["questioning"] for model in llm_prevalence)),
        "human_questioning_percent": round(human_questioning),
    }

    for key, expected_value in expected.items():
        observed_value = observed[key]
        if observed_value != expected_value:
            errors.append(f"Prevalence mismatch for {key}: expected {expected_value}, got {observed_value}")
            cprint(errors[-1], "red", force_color=True)
            continue
        cprint(f"[ok] prevalence {key}: {observed_value}", "green", force_color=True)


def verify_claims(expected_rows, expected_claims, errors):
    rows = expected_rows
    observed = {
        "avg_agg_empathy_improvement_percent": round1(
            100
            * (
                ((rows["1.7B Ours Q+DKL"]["aggregated_empathy"] - rows["1.7B Vanilla"]["aggregated_empathy"])
                 / rows["1.7B Vanilla"]["aggregated_empathy"])
                + ((rows["4B Ours Q+DKL"]["aggregated_empathy"] - rows["4B Vanilla"]["aggregated_empathy"])
                   / rows["4B Vanilla"]["aggregated_empathy"])
            )
            / 2
        ),
        "4b_stickiness_reduction_percent": round1(
            100
            * (
                rows["4B Vanilla"]["stickiness"] - rows["4B Ours Q+DKL"]["stickiness"]
            )
            / rows["4B Vanilla"]["stickiness"]
        ),
        "1p7b_unsolicited_advice_drop_percent": round1(
            100
            * (
                rows["1.7B Vanilla"]["advice_giving"] - rows["1.7B Ours Q+DKL"]["advice_giving"]
            )
            / rows["1.7B Vanilla"]["advice_giving"]
        ),
        "4b_unsolicited_advice_drop_percent": round1(
            100
            * (
                rows["4B Vanilla"]["advice_giving"] - rows["4B Ours Q+DKL"]["advice_giving"]
            )
            / rows["4B Vanilla"]["advice_giving"]
        ),
        "1p7b_encouraging_elaboration_gain_percent": round1(
            100
            * (
                rows["1.7B Ours Q+DKL"]["encouraging_elaboration"]
                - rows["1.7B Vanilla"]["encouraging_elaboration"]
            )
            / rows["1.7B Vanilla"]["encouraging_elaboration"]
        ),
        "4b_encouraging_elaboration_gain_percent": round1(
            100
            * (
                rows["4B Ours Q+DKL"]["encouraging_elaboration"]
                - rows["4B Vanilla"]["encouraging_elaboration"]
            )
            / rows["4B Vanilla"]["encouraging_elaboration"]
        ),
    }

    for key, expected_value in expected_claims.items():
        observed_value = observed[key]
        if observed_value != expected_value:
            errors.append(f"Claim mismatch for {key}: expected {expected_value}, got {observed_value}")
            cprint(errors[-1], "red", force_color=True)
            continue
        cprint(f"[ok] claim {key}: {observed_value}", "green", force_color=True)


def main(_):
    config = load_config()
    repo_root = Path(config["repo_root"]).resolve()
    eval_dir = repo_root / "evaluation"
    errors = []

    cprint("Verifying dataset counts...", "yellow", force_color=True)
    verify_counts(repo_root, config["expected"]["counts"], errors)

    cprint("Verifying Table 2 non-word metrics...", "yellow", force_color=True)
    verify_table(
        eval_dir,
        config["expected"]["table_2_non_word_metrics"],
        config["tolerances"]["table_metric_abs"],
        errors,
    )

    cprint("Verifying tactic prevalence caption numbers...", "yellow", force_color=True)
    verify_prevalence(repo_root, config["expected"]["prevalence_caption"], errors)

    cprint("Verifying prose claim numbers derived from Table 2...", "yellow", force_color=True)
    verify_claims(
        config["expected"]["table_2_non_word_metrics"],
        config["expected"]["derived_claims"],
        errors,
    )

    if errors:
        cprint("", "red", force_color=True)
        cprint(f"Verification failed with {len(errors)} issue(s).", "red", force_color=True)
        raise SystemExit(1)

    cprint("", "green", force_color=True)
    cprint("All configured paper-number checks passed.", "green", force_color=True)
    cprint("Word-count checks are intentionally excluded.", "yellow", force_color=True)


if __name__ == "__main__":
    app.run(main)