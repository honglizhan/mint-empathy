"""
Sanity check for the 3-component tactic diversity reward.
Tests the math functions directly with hand-crafted inputs (no vLLM server needed).

Usage:
    python test_reward_sanity.py
"""
from termcolor import cprint

from reward_func_tactics_kl_bigram_entropy import (
    compute_kl_divergence,
    compute_bigram_surprisal,
    compute_trigram_surprisal,
    compute_within_turn_entropy,
    TACTIC_NAMES,
    NUM_TACTICS,
)

ALPHA = 0.1
KL_GAMMA = 1.0
BIGRAM_GAMMA = 1.0
TRIGRAM_GAMMA = 1.0
ENTROPY_GAMMA = 2.0


def combined_reward(kl, bigram, trigram, entropy):
    return KL_GAMMA * kl + BIGRAM_GAMMA * bigram + TRIGRAM_GAMMA * trigram + ENTROPY_GAMMA * entropy


def run_scenario(name, hist_counts, hist_bigrams, curr_counts, sentence_tactics, hist_trigrams=None):
    if hist_trigrams is None:
        hist_trigrams = {}

    cprint(f"\n{'='*70}", "cyan", force_color=True)
    cprint(f"Scenario: {name}", "cyan", force_color=True)
    cprint(f"{'='*70}", "cyan", force_color=True)

    cprint(f"  History counts:  {hist_counts}", "yellow", force_color=True)
    cprint(f"  History bigrams: {hist_bigrams}", "yellow", force_color=True)
    cprint(f"  History trigrams: {hist_trigrams}", "yellow", force_color=True)
    cprint(f"  Current counts:  {curr_counts}", "yellow", force_color=True)
    cprint(f"  Sentence tactics: {[sorted(s) for s in sentence_tactics]}", "yellow", force_color=True)

    kl = compute_kl_divergence(hist_counts, curr_counts, ALPHA)
    bigram = compute_bigram_surprisal(hist_bigrams, sentence_tactics, ALPHA)
    trigram = compute_trigram_surprisal(hist_trigrams, sentence_tactics, ALPHA)
    entropy = compute_within_turn_entropy(curr_counts)
    total = combined_reward(kl, bigram, trigram, entropy)

    cprint(f"  KL divergence:     {kl:.4f}  (weighted: {KL_GAMMA * kl:.4f})", "green", force_color=True)
    cprint(f"  Bigram surprisal:  {bigram:.4f}  (weighted: {BIGRAM_GAMMA * bigram:.4f})", "green", force_color=True)
    cprint(f"  Trigram surprisal: {trigram:.4f}  (weighted: {TRIGRAM_GAMMA * trigram:.4f})", "green", force_color=True)
    cprint(f"  Within-turn H:     {entropy:.4f}  (weighted: {ENTROPY_GAMMA * entropy:.4f})", "green", force_color=True)
    cprint(f"  TOTAL REWARD:      {total:.4f}", "magenta", force_color=True)

    return total


def main():
    cprint("Tactic Diversity Reward Sanity Check", "cyan", force_color=True)
    cprint(f"Gamma weights: KL={KL_GAMMA}, bigram={BIGRAM_GAMMA}, trigram={TRIGRAM_GAMMA}, entropy={ENTROPY_GAMMA}", "cyan", force_color=True)
    cprint(f"Smoothing alpha: {ALPHA}", "cyan", force_color=True)
    cprint(f"Tactics ({NUM_TACTICS}): {TACTIC_NAMES}", "cyan", force_color=True)

    # ----------------------------------------------------------------
    # History: a conversation that's been doing P-V-P for 3 turns
    # ----------------------------------------------------------------
    pvp_hist_counts = {"paraphrasing": 3, "validation": 3}
    pvp_hist_bigrams = {
        "paraphrasing": {"validation": 2},
        "validation": {"paraphrasing": 2},
    }
    pvp_hist_trigrams = {
        "paraphrasing": {"validation": {"paraphrasing": 2}},
        "validation": {"paraphrasing": {"validation": 2}},
    }

    # Scenario 1: Model continues the P-V-P pattern (BAD - should get LOW reward)
    s1 = run_scenario(
        "Continue P-V-P (should be LOWEST)",
        hist_counts=pvp_hist_counts,
        hist_bigrams=pvp_hist_bigrams,
        curr_counts={"paraphrasing": 2, "validation": 2},
        sentence_tactics=[
            {"paraphrasing"}, {"validation"}, {"paraphrasing"}, {"validation"},
        ],
        hist_trigrams=pvp_hist_trigrams,
    )

    # Scenario 2: Model uses rare tactics in novel order (GOOD - should get HIGH reward)
    s2 = run_scenario(
        "Novel tactics + novel transitions (should be HIGHEST)",
        hist_counts=pvp_hist_counts,
        hist_bigrams=pvp_hist_bigrams,
        curr_counts={"reappraisal": 2, "empowerment": 1, "questioning": 1},
        sentence_tactics=[
            {"questioning"}, {"reappraisal"}, {"empowerment"}, {"reappraisal"},
        ],
        hist_trigrams=pvp_hist_trigrams,
    )

    # Scenario 3: Model uses ONE rare tactic but fills entire response with it (MEDIOCRE)
    s3 = run_scenario(
        "One rare tactic repeated (high KL, zero entropy)",
        hist_counts=pvp_hist_counts,
        hist_bigrams=pvp_hist_bigrams,
        curr_counts={"reappraisal": 4},
        sentence_tactics=[
            {"reappraisal"}, {"reappraisal"}, {"reappraisal"}, {"reappraisal"},
        ],
        hist_trigrams=pvp_hist_trigrams,
    )

    # Scenario 4: First turn, empty history (should get reasonable baseline)
    s4 = run_scenario(
        "First turn (empty history, diverse response)",
        hist_counts={},
        hist_bigrams={},
        curr_counts={"questioning": 2, "validation": 1, "advice": 1},
        sentence_tactics=[
            {"questioning"}, {"validation"}, {"questioning"}, {"advice"},
        ],
    )

    # Scenario 5: First turn, empty history, monotactic
    s5 = run_scenario(
        "First turn (empty history, monotactic)",
        hist_counts={},
        hist_bigrams={},
        curr_counts={"paraphrasing": 4},
        sentence_tactics=[
            {"paraphrasing"}, {"paraphrasing"}, {"paraphrasing"}, {"paraphrasing"},
        ],
    )

    # Scenario 6: Single sentence response
    s6 = run_scenario(
        "Single sentence (no bigrams possible)",
        hist_counts=pvp_hist_counts,
        hist_bigrams=pvp_hist_bigrams,
        curr_counts={"empowerment": 1},
        sentence_tactics=[
            {"empowerment"},
        ],
        hist_trigrams=pvp_hist_trigrams,
    )

    # Scenario 7: Empty response (no tactics detected)
    s7 = run_scenario(
        "Empty response (no tactics)",
        hist_counts=pvp_hist_counts,
        hist_bigrams=pvp_hist_bigrams,
        curr_counts={},
        sentence_tactics=[],
        hist_trigrams=pvp_hist_trigrams,
    )

    # ----------------------------------------------------------------
    # Summary: rank scenarios by reward
    # ----------------------------------------------------------------
    results = [
        ("Continue P-V-P", s1),
        ("Novel tactics + transitions", s2),
        ("One rare tactic repeated", s3),
        ("First turn, diverse", s4),
        ("First turn, monotactic", s5),
        ("Single sentence", s6),
        ("Empty response", s7),
    ]
    results.sort(key=lambda x: x[1], reverse=True)

    cprint(f"\n{'='*70}", "cyan", force_color=True)
    cprint("RANKING (highest to lowest reward):", "cyan", force_color=True)
    cprint(f"{'='*70}", "cyan", force_color=True)
    for rank, (name, score) in enumerate(results, 1):
        cprint(f"  {rank}. {score:+.4f}  {name}", "magenta", force_color=True)

    # ----------------------------------------------------------------
    # Sanity assertions
    # ----------------------------------------------------------------
    cprint(f"\n{'='*70}", "cyan", force_color=True)
    cprint("Sanity assertions:", "cyan", force_color=True)

    checks = [
        ("Novel > Continue P-V-P", s2 > s1),
        ("Novel > One rare repeated", s2 > s3),
        ("One rare repeated > Continue P-V-P", s3 > s1),
        ("First diverse > First monotactic", s4 > s5),
        ("Empty response = 0", s7 == 0.0),
    ]
    for desc, passed in checks:
        status = "PASS" if passed else "FAIL"
        color = "green" if passed else "red"
        cprint(f"  [{status}] {desc}", color, force_color=True)

    if all(passed for _, passed in checks):
        cprint("\nAll sanity checks passed!", "green", force_color=True)
    else:
        cprint("\nSome checks FAILED. Review the component magnitudes.", "red", force_color=True)


if __name__ == "__main__":
    main()
