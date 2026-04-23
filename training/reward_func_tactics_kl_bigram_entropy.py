"""Shared tactic reward utilities used by VERL reward/runtime scripts."""

import math
import os
import re
from typing import Dict, List, Set, Tuple

from termcolor import cprint

TACTIC_NAMES = [
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
NUM_TACTICS = len(TACTIC_NAMES)


def load_tactic_info(prompts_dir: str) -> Dict[str, Dict[str, str]]:
    """Load tactic prompt templates from files."""
    tactic_info: Dict[str, Dict[str, str]] = {}
    for tactic in TACTIC_NAMES:
        prompt_path = os.path.join(prompts_dir, f"{tactic}.txt")
        if not os.path.exists(prompt_path):
            cprint(f"Warning: Prompt file not found for {tactic} at {prompt_path}", "red")
            continue

        with open(prompt_path, "r") as f:
            content = f.read()

        match = re.search(r'contains "([^"]+)"', content)
        cap_name = match.group(1) if match else tactic.replace("_", " ").title()
        system_prompt = (
            "You are a Fair Tagger Assistant, responsible for providing precise, "
            "objective tagging based on predefined criteria. Your task is to assess "
            f"whether a given sentence contains \"{cap_name}\", ensuring consistency "
            "and adherence to strict tagging guidelines."
        )

        tactic_info[tactic] = {
            "system_prompt": system_prompt,
            "user_template": content,
        }

    return tactic_info


def parse_tactic_score(output_text: str) -> int:
    """Parse `<score>X</score>` from tagger output."""
    match = re.search(r"<score>(\d+)</score>", output_text)
    if match:
        return int(match.group(1))
    return 0


def extract_response_from_thinking_model(text: str) -> str:
    """Strip `<think>...</think>` reasoning blocks and return visible response."""
    cleaned = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()
    return cleaned if cleaned else text.strip()


def clean_history_structs(
    raw_hist_counts: Dict[str, int] | None,
    raw_hist_bigrams: Dict[str, Dict[str, int]] | None,
    raw_hist_trigrams: Dict[str, Dict[str, Dict[str, int]]] | None = None,
) -> Tuple[Dict[str, int], Dict[str, Dict[str, int]], Dict[str, Dict[str, Dict[str, int]]]]:
    """Drop `None` keys/values injected by Arrow/parquet in nested history structs."""
    counts_in = raw_hist_counts or {}
    bigrams_in = raw_hist_bigrams or {}
    trigrams_in = raw_hist_trigrams or {}

    hist_counts = {k: v for k, v in counts_in.items() if v is not None}

    hist_bigrams: Dict[str, Dict[str, int]] = {}
    for src, dests in bigrams_in.items():
        if dests is None:
            continue
        cleaned_dests = {dst: count for dst, count in dests.items() if count is not None}
        if cleaned_dests:
            hist_bigrams[src] = cleaned_dests

    hist_trigrams: Dict[str, Dict[str, Dict[str, int]]] = {}
    for a, mid_dict in trigrams_in.items():
        if mid_dict is None:
            continue
        cleaned_mid: Dict[str, Dict[str, int]] = {}
        for b, dst_dict in mid_dict.items():
            if dst_dict is None:
                continue
            cleaned_dst = {c: count for c, count in dst_dict.items() if count is not None}
            if cleaned_dst:
                cleaned_mid[b] = cleaned_dst
        if cleaned_mid:
            hist_trigrams[a] = cleaned_mid

    return hist_counts, hist_bigrams, hist_trigrams


def compute_kl_divergence(
    hist_counts: Dict[str, int],
    curr_counts: Dict[str, int],
    alpha: float = 0.1,
) -> float:
    """KL(Q_current || P_history) with Laplace smoothing."""
    if not curr_counts:
        return 0.0

    total_hist = sum(hist_counts.values()) + NUM_TACTICS * alpha
    total_curr = sum(curr_counts.values()) + NUM_TACTICS * alpha

    kl = 0.0
    for tactic in TACTIC_NAMES:
        p_h = (hist_counts.get(tactic, 0) + alpha) / total_hist
        q_c = (curr_counts.get(tactic, 0) + alpha) / total_curr
        kl += q_c * math.log(q_c / p_h)

    return min(kl, 5.0)


def compute_bigram_surprisal(
    hist_bigrams: Dict[str, Dict[str, int]],
    sentence_tactics: List[Set[str]],
    alpha: float = 0.1,
) -> float:
    """Average -log P_history(t_{k+1} | t_k) over consecutive sentence transitions."""
    if len(sentence_tactics) < 2:
        return 0.0

    total_surprisal = 0.0
    num_transitions = 0

    for k in range(len(sentence_tactics) - 1):
        src_set = sentence_tactics[k]
        dst_set = sentence_tactics[k + 1]

        if not src_set or not dst_set:
            continue

        for src in src_set:
            src_hist = hist_bigrams.get(src, {})
            total_from_src = sum(src_hist.values()) + NUM_TACTICS * alpha
            for dst in dst_set:
                p = (src_hist.get(dst, 0) + alpha) / total_from_src
                total_surprisal += -math.log(p)
                num_transitions += 1

    return total_surprisal / num_transitions if num_transitions > 0 else 0.0


def compute_trigram_surprisal(
    hist_trigrams: Dict[str, Dict[str, Dict[str, int]]],
    sentence_tactics: List[Set[str]],
    alpha: float = 0.1,
) -> float:
    """Average -log P_history(t_{k+2} | t_k, t_{k+1}) over consecutive sentence triples."""
    if len(sentence_tactics) < 3:
        return 0.0

    total_surprisal = 0.0
    num_transitions = 0

    for k in range(len(sentence_tactics) - 2):
        src_set = sentence_tactics[k]
        mid_set = sentence_tactics[k + 1]
        dst_set = sentence_tactics[k + 2]

        if not src_set or not mid_set or not dst_set:
            continue

        for a in src_set:
            for b in mid_set:
                ab_hist = hist_trigrams.get(a, {}).get(b, {})
                total_from_ab = sum(ab_hist.values()) + NUM_TACTICS * alpha
                for c in dst_set:
                    p = (ab_hist.get(c, 0) + alpha) / total_from_ab
                    total_surprisal += -math.log(p)
                    num_transitions += 1

    return total_surprisal / num_transitions if num_transitions > 0 else 0.0


def compute_within_turn_entropy(curr_counts: Dict[str, int]) -> float:
    """Shannon entropy of tactic distribution within the current response."""
    total = sum(curr_counts.values())
    if total == 0:
        return 0.0

    entropy = 0.0
    for count in curr_counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log(p)

    return entropy
