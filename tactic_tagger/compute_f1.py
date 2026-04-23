import json
import re
from collections import Counter
from pathlib import Path

from absl import app, flags
from termcolor import cprint

FLAGS = flags.FLAGS

TACTIC_NAMES = [
    "advice",
    "assistance",
    "emotional_expression",
    "empowerment",
    "information",
    "paraphrasing",
    "questioning",
    "reappraisal",
    "self_disclosure",
    "validation",
]

flags.DEFINE_string(
    "gold_dir",
    "../data/tagger_annotations/test",
    "Directory with gold tactic jsonl files.",
)
flags.DEFINE_string(
    "predictions_dir",
    "./predictions/Llama-3.1-8B-Instruct-trained-tagger-lora",
    "Directory with per-tactic prediction jsonl files.",
)
flags.DEFINE_string(
    "output_json",
    "",
    "Optional path to save the summary as JSON.",
)


def f1_for_label(y_true: list[int], y_pred: list[int], label: int) -> float:
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == label and yp == label)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != label and yp == label)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == label and yp != label)
    if tp == 0 and fp == 0 and fn == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def macro_binary_f1(y_true: list[int], y_pred: list[int]) -> float:
    return (f1_for_label(y_true, y_pred, 0) + f1_for_label(y_true, y_pred, 1)) / 2


def extract_score(text: str | None) -> int | None:
    if text is None:
        return None
    match = re.search(r"<score>\[?(\d+)\]?</score>", text)
    if match:
        return int(match.group(1))
    match = re.search(r"<score>(\d+)\[\]</score>", text)
    if match:
        return int(match.group(1))
    return None


def extract_context_and_sentence(user_text: str) -> tuple[str | None, str | None]:
    pattern = (
        r"### Input:\n"
        r"- Context \(Full Empathic Response\):\n(.*?)\n\n"
        r"- Sentence to Evaluate:\n(.*?)\n\n"
        r"### Response:"
    )
    match = re.search(pattern, user_text, flags=re.DOTALL)
    if not match:
        return None, None
    return match.group(1), match.group(2)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def parse_gold_row(row: dict, tactic: str) -> dict:
    messages = row["messages"]
    user_text = messages[1]["content"]
    assistant_text = messages[2]["content"]
    whole_response, sentence = extract_context_and_sentence(user_text)
    return {
        "key": (whole_response, sentence),
        "label": extract_score(assistant_text),
        "whole_response": whole_response,
        "sentence": sentence,
    }


def parse_prediction_row(row: dict, tactic: str) -> dict:
    if tactic in row:
        whole_response = row.get("whole_response")
        sentence = row.get("sentence")
        pred = extract_score(row.get(tactic))
        post_id = row.get("postID")
        sentence_id = row.get("sentenceID")
        return {
            "key": (whole_response, sentence),
            "id_key": (post_id, sentence_id),
            "label": pred,
            "whole_response": whole_response,
            "sentence": sentence,
        }

    messages = row["messages"]
    user_text = messages[1]["content"]
    assistant_text = messages[2]["content"]
    whole_response, sentence = extract_context_and_sentence(user_text)
    return {
        "key": (whole_response, sentence),
        "id_key": None,
        "label": extract_score(assistant_text),
        "whole_response": whole_response,
        "sentence": sentence,
    }


def align_rows(gold_rows: list[dict], pred_rows: list[dict]) -> tuple[list[int], list[int], str]:
    gold_keys = [row["key"] for row in gold_rows]
    pred_keys = [row["key"] for row in pred_rows]

    if len(gold_rows) == len(pred_rows):
        sentence_matches = sum(
            1
            for i in range(len(gold_rows))
            if gold_rows[i].get("sentence") == pred_rows[i].get("sentence")
        )
        sentence_match_rate = sentence_matches / len(gold_rows) if gold_rows else 0.0
        if gold_keys == pred_keys or sentence_match_rate >= 0.95:
            y_true = [row["label"] for row in gold_rows]
            y_pred = [row["label"] for row in pred_rows]
            alignment = "row_order" if gold_keys == pred_keys else "row_order_sentence_match"
            return y_true, y_pred, alignment

    gold_counter = Counter(gold_keys)
    pred_counter = Counter(pred_keys)
    if gold_counter == pred_counter and all(v == 1 for v in gold_counter.values()):
        pred_by_key = {row["key"]: row for row in pred_rows}
        y_true = []
        y_pred = []
        for row in gold_rows:
            y_true.append(row["label"])
            y_pred.append(pred_by_key[row["key"]]["label"])
        return y_true, y_pred, "text_key"

    raise ValueError(
        f"Could not align gold and prediction rows. gold={len(gold_rows)} pred={len(pred_rows)}"
    )


def compute_one_tactic(gold_path: Path, pred_path: Path, tactic: str) -> dict:
    gold_rows = [parse_gold_row(row, tactic) for row in load_jsonl(gold_path)]
    pred_rows = [parse_prediction_row(row, tactic) for row in load_jsonl(pred_path)]
    y_true, y_pred, alignment = align_rows(gold_rows, pred_rows)

    kept_true = []
    kept_pred = []
    skipped = 0
    for true_label, pred_label in zip(y_true, y_pred):
        if true_label is None or pred_label is None:
            skipped += 1
            continue
        kept_true.append(true_label)
        kept_pred.append(pred_label)

    if not kept_true:
        raise ValueError(f"No valid aligned rows for {tactic}")

    f1 = macro_binary_f1(kept_true, kept_pred)
    return {
        "tactic": tactic,
        "f1": float(f1),
        "n_examples": len(kept_true),
        "n_skipped": skipped,
        "alignment": alignment,
    }


def main(argv):
    del argv
    gold_dir = Path(FLAGS.gold_dir)
    pred_dir = Path(FLAGS.predictions_dir)

    results = []
    for tactic in sorted(TACTIC_NAMES):
        gold_path = gold_dir / f"{tactic}.jsonl"
        pred_path = pred_dir / f"{tactic}.jsonl"
        if not gold_path.exists():
            raise FileNotFoundError(f"Missing gold file: {gold_path}")
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing prediction file: {pred_path}")

        result = compute_one_tactic(gold_path, pred_path, tactic)
        results.append(result)
        cprint(
            f"{tactic:>20}  F1={result['f1']:.3f}  n={result['n_examples']}  skipped={result['n_skipped']}  align={result['alignment']}",
            "green",
            force_color=True,
        )

    mean_f1 = sum(item["f1"] for item in results) / len(results)
    median_f1 = sorted(item["f1"] for item in results)[len(results) // 2]
    best = max(results, key=lambda x: x["f1"])
    worst = min(results, key=lambda x: x["f1"])

    cprint("", "white", force_color=True)
    cprint("F1 summary", "cyan", force_color=True)
    cprint(f"Mean F1: {mean_f1:.3f}", "yellow", force_color=True)
    cprint(f"Median F1: {median_f1:.3f}", "yellow", force_color=True)
    cprint(f"Best tactic: {best['tactic']} ({best['f1']:.3f})", "yellow", force_color=True)
    cprint(f"Worst tactic: {worst['tactic']} ({worst['f1']:.3f})", "yellow", force_color=True)

    summary = {
        "mean_f1": mean_f1,
        "median_f1": median_f1,
        "best_tactic": best,
        "worst_tactic": worst,
        "per_tactic": results,
    }

    if FLAGS.output_json:
        output_path = Path(FLAGS.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2))
        cprint(f"Saved summary to {output_path}", "green", force_color=True)


if __name__ == "__main__":
    app.run(main)