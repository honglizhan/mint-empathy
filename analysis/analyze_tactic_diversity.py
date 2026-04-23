"""
Analyze tactic repetition patterns in pre-tagged emotional support conversations.

This script loads the 322-conversation dataset with pre-tagged tactics and computes:
1. Overall reuse rates (new vs reused tactic usages)
2. Per-turn reuse fraction distribution
3. Per-tactic reuse rates
4. Simulated diversity reward distribution
5. Reward breakdown by conversation turn index
6. Stacked bar chart of mean new vs reused tactics by turn index
7. Line plot of mean simulated diversity reward by turn index

Generates 7 PNG plots saved to the same directory.
"""

import os
import sys
import json
import numpy as np
from collections import defaultdict, Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Import TACTIC_NAMES from the reward module in training
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training"))
from reward_func_tactics_kl_bigram_entropy import TACTIC_NAMES

# ── Config ──────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "data",
    "training",
    "conversations_322_tagged.json",
)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis_results")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Default reward parameters (same as tactic_diversity_reward_func defaults)
NEW_REWARD = 2.0
REUSE_BASE_PENALTY = -2.0
FREQUENCY_WEIGHT = 0.5
RECENCY_DECAY = 0.7


def load_data(path: str) -> list:
    with open(path) as f:
        return json.load(f)


def analyze(conversations: list):
    """Walk every conversation, build tactic history incrementally, and collect stats."""

    # ── Collectors ──────────────────────────────────────────────────────────
    # Per-turn stats
    turn_reuse_fractions = []       # fraction of tactics in this turn that are reused
    turn_rewards = []               # simulated diversity reward
    turn_indices_list = []          # assistant turn index within its conversation
    turn_new_counts = []
    turn_reused_counts = []

    # Per-tactic stats
    tactic_total_uses = Counter()   # total times each tactic appears
    tactic_reuse_count = Counter()  # times each tactic appears as a reuse

    # Co-occurrence across consecutive turns
    consecutive_cooccurrence = defaultdict(int)  # (tactic, tactic) -> count

    total_assistant_turns = 0

    for conv in conversations:
        messages = conv["conversation"]

        # Build history incrementally
        history_dict: dict = {}   # tactic -> {"count": int, "turn_indices": [int]}
        assistant_turn_idx = 0
        prev_turn_tactics = set()

        for msg in messages:
            if msg["role"] != "supporter":
                continue

            curr_tactics = set(msg.get("all_tactics", []))

            prev_set = {k for k, v in history_dict.items() if v is not None}

            new_tactics = curr_tactics - prev_set
            reused_tactics = curr_tactics & prev_set

            # ── Skip turn 0: it has no history so everything is trivially new ─
            if assistant_turn_idx > 0:
                total_assistant_turns += 1
                # ── Per-turn stats ──────────────────────────────────────────
                n_total = len(curr_tactics)
                n_reused = len(reused_tactics)
                reuse_frac = n_reused / n_total if n_total > 0 else 0.0
                turn_reuse_fractions.append(reuse_frac)
                turn_indices_list.append(assistant_turn_idx)
                turn_new_counts.append(len(new_tactics))
                turn_reused_counts.append(n_reused)

                # ── Simulated reward ────────────────────────────────────────
                new_score = len(new_tactics) * NEW_REWARD
                reuse_score = 0.0
                for tactic in reused_tactics:
                    info = history_dict[tactic]
                    count = info["count"]
                    most_recent = max(info["turn_indices"])
                    freq_w = 1.0 + FREQUENCY_WEIGHT * (count - 1) if count > 1 else 1.0
                    recency_w = RECENCY_DECAY ** (assistant_turn_idx - 1 - most_recent) if assistant_turn_idx > 0 else 0.0
                    reuse_score += REUSE_BASE_PENALTY * freq_w * recency_w
                turn_rewards.append(new_score + reuse_score)

                # ── Per-tactic stats ────────────────────────────────────────
                for t in curr_tactics:
                    tactic_total_uses[t] += 1
                    if t in prev_set:
                        tactic_reuse_count[t] += 1

                # ── Co-occurrence with previous turn ────────────────────────
                if prev_turn_tactics:
                    shared = curr_tactics & prev_turn_tactics
                    for t in shared:
                        for t2 in shared:
                            consecutive_cooccurrence[(t, t2)] += 1

            # ── Update history for next turn ────────────────────────────────
            for t in curr_tactics:
                if t not in history_dict or history_dict[t] is None:
                    history_dict[t] = {"count": 0, "turn_indices": []}
                history_dict[t]["count"] += 1
                history_dict[t]["turn_indices"].append(assistant_turn_idx)

            prev_turn_tactics = curr_tactics
            assistant_turn_idx += 1

    return {
        "total_assistant_turns": total_assistant_turns,
        "turn_reuse_fractions": np.array(turn_reuse_fractions),
        "turn_rewards": np.array(turn_rewards),
        "turn_indices": np.array(turn_indices_list),
        "turn_new_counts": np.array(turn_new_counts),
        "turn_reused_counts": np.array(turn_reused_counts),
        "tactic_total_uses": tactic_total_uses,
        "tactic_reuse_count": tactic_reuse_count,
        "consecutive_cooccurrence": consecutive_cooccurrence,
    }


def print_stats(stats: dict, n_convs: int):
    print("=" * 70)
    print("TACTIC DIVERSITY ANALYSIS")
    print("=" * 70)

    total_turns = stats["total_assistant_turns"]
    total_usages = int(stats["turn_new_counts"].sum() + stats["turn_reused_counts"].sum())
    total_reused = int(stats["turn_reused_counts"].sum())
    total_new = int(stats["turn_new_counts"].sum())

    print(f"\nDataset: {n_convs} conversations, {total_turns} assistant turns")
    print(f"Total tactic usages: {total_usages}")
    print(f"  New usages:    {total_new} ({100*total_new/total_usages:.1f}%)")
    print(f"  Reused usages: {total_reused} ({100*total_reused/total_usages:.1f}%)")

    fracs = stats["turn_reuse_fractions"]
    print(f"\nPer-turn reuse fraction:")
    print(f"  Mean:   {fracs.mean():.3f}")
    print(f"  Median: {np.median(fracs):.3f}")
    print(f"  Turns with any reuse:    {(fracs > 0).sum()}/{total_turns} ({100*(fracs > 0).mean():.1f}%)")
    print(f"  Turns with 75-100% reuse: {(fracs >= 0.75).sum()}/{total_turns} ({100*(fracs >= 0.75).mean():.1f}%)")

    rewards = stats["turn_rewards"]
    print(f"\nSimulated diversity reward (new={NEW_REWARD}, reuse_base={REUSE_BASE_PENALTY}, "
          f"freq_w={FREQUENCY_WEIGHT}, decay={RECENCY_DECAY}):")
    print(f"  Mean:   {rewards.mean():.3f}")
    print(f"  Median: {np.median(rewards):.3f}")
    print(f"  Std:    {rewards.std():.3f}")
    print(f"  Min:    {rewards.min():.3f}")
    print(f"  Max:    {rewards.max():.3f}")

    print(f"\nPer-tactic reuse rates:")
    print(f"  {'Tactic':<25s} {'Total':>6s} {'Reused':>7s} {'Rate':>7s}")
    print(f"  {'-'*25} {'-'*6} {'-'*7} {'-'*7}")
    for tactic in sorted(TACTIC_NAMES, key=lambda t: stats["tactic_total_uses"].get(t, 0), reverse=True):
        total = stats["tactic_total_uses"].get(tactic, 0)
        reused = stats["tactic_reuse_count"].get(tactic, 0)
        rate = reused / total if total > 0 else 0.0
        print(f"  {tactic:<25s} {total:>6d} {reused:>7d} {rate:>7.1%}")

    # Reward by turn index
    print(f"\nReuse fraction by assistant turn index:")
    indices = stats["turn_indices"]
    fracs = stats["turn_reuse_fractions"]
    max_idx = min(int(indices.max()), 9)
    print(f"  {'Turn':>5s} {'N':>5s} {'Mean new':>9s} {'Mean reused':>12s} {'Mean reuse':>11s} {'Mean reward':>12s}")
    for idx in range(max_idx + 1):
        mask = indices == idx
        n = mask.sum()
        if n > 0:
            mean_new = stats["turn_new_counts"][mask].mean()
            mean_reused = stats["turn_reused_counts"][mask].mean()
            mean_frac = fracs[mask].mean()
            mean_rew = stats["turn_rewards"][mask].mean()
            print(f"  {idx:>5d} {n:>5d} {mean_new:>9.2f} {mean_reused:>12.2f} {mean_frac:>11.3f} {mean_rew:>12.3f}")

    print("=" * 70)


def plot_all(stats: dict):
    """Generate 7 PNG plots."""

    fracs = stats["turn_reuse_fractions"]
    rewards = stats["turn_rewards"]
    indices = stats["turn_indices"]

    # ── 1. Histogram of per-turn reuse fraction ────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(fracs, bins=20, range=(0, 1), edgecolor="black", alpha=0.75, color="steelblue")
    ax.set_xlabel("Reuse Fraction (per assistant turn)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Per-Turn Tactic Reuse Fraction")
    ax.axvline(fracs.mean(), color="red", linestyle="--", label=f"Mean = {fracs.mean():.2f}")
    ax.legend()
    fig.tight_layout()
    path1 = os.path.join(OUTPUT_DIR, "plot_reuse_fraction_hist.png")
    fig.savefig(path1, dpi=150)
    plt.close(fig)
    print(f"Saved: {path1}")

    # ── 2. Histogram of simulated diversity rewards ────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(rewards, bins=30, edgecolor="black", alpha=0.75, color="salmon")
    ax.set_xlabel("Diversity Reward")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Simulated Tactic Diversity Rewards")
    ax.axvline(rewards.mean(), color="blue", linestyle="--", label=f"Mean = {rewards.mean():.2f}")
    ax.axvline(0, color="black", linestyle=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    path2 = os.path.join(OUTPUT_DIR, "plot_reward_distribution.png")
    fig.savefig(path2, dpi=150)
    plt.close(fig)
    print(f"Saved: {path2}")

    # ── 3. Per-tactic reuse rate bar chart ─────────────────────────────────
    tactics_sorted = sorted(
        TACTIC_NAMES,
        key=lambda t: stats["tactic_total_uses"].get(t, 0),
        reverse=True,
    )
    total_vals = [stats["tactic_total_uses"].get(t, 0) for t in tactics_sorted]
    reuse_vals = [stats["tactic_reuse_count"].get(t, 0) for t in tactics_sorted]
    reuse_rates = [r / t if t > 0 else 0 for r, t in zip(reuse_vals, total_vals)]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(tactics_sorted))
    width = 0.35
    ax.bar(x - width / 2, total_vals, width, label="Total uses", color="steelblue", edgecolor="black")
    ax.bar(x + width / 2, reuse_vals, width, label="Reused", color="salmon", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_", "\n") for t in tactics_sorted], fontsize=8)
    ax.set_ylabel("Count")
    ax.set_title("Per-Tactic Total Uses vs Reuses")
    ax.legend()

    # Add reuse rate labels
    for i, rate in enumerate(reuse_rates):
        ax.text(i + width / 2, reuse_vals[i] + 5, f"{rate:.0%}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    path3 = os.path.join(OUTPUT_DIR, "plot_per_tactic_reuse.png")
    fig.savefig(path3, dpi=150)
    plt.close(fig)
    print(f"Saved: {path3}")

    # ── 4. Reuse fraction vs turn index (box plot) ─────────────────────────
    max_idx = min(int(indices.max()), 9)
    groups = []
    labels = []
    for idx in range(max_idx + 1):
        mask = indices == idx
        if mask.sum() > 0:
            groups.append(fracs[mask])
            labels.append(str(idx))

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(groups, labels=labels, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("steelblue")
        patch.set_alpha(0.6)
    ax.set_xlabel("Assistant Turn Index")
    ax.set_ylabel("Reuse Fraction")
    ax.set_title("Tactic Reuse Fraction by Conversation Turn Index")
    # overlay means
    means = [g.mean() for g in groups]
    ax.plot(range(1, len(means) + 1), means, "ro-", markersize=5, label="Mean")
    ax.legend()
    fig.tight_layout()
    path4 = os.path.join(OUTPUT_DIR, "plot_reuse_vs_turn_index.png")
    fig.savefig(path4, dpi=150)
    plt.close(fig)
    print(f"Saved: {path4}")

    # ── 5. Heatmap of tactic co-occurrence across consecutive turns ────────
    cooc = stats["consecutive_cooccurrence"]
    # Use only tactics that appear in the data
    active_tactics = [t for t in tactics_sorted if stats["tactic_total_uses"].get(t, 0) > 0]
    n = len(active_tactics)
    matrix = np.zeros((n, n))
    for i, t1 in enumerate(active_tactics):
        for j, t2 in enumerate(active_tactics):
            matrix[i, j] = cooc.get((t1, t2), 0)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([t.replace("_", "\n") for t in active_tactics], fontsize=7, rotation=45, ha="right")
    ax.set_yticklabels([t.replace("_", "\n") for t in active_tactics], fontsize=7)
    ax.set_title("Tactic Co-occurrence in Consecutive Assistant Turns")
    fig.colorbar(im, ax=ax, label="Co-occurrence count")

    # Add value annotations
    for i in range(n):
        for j in range(n):
            val = int(matrix[i, j])
            if val > 0:
                color = "white" if val > matrix.max() * 0.6 else "black"
                ax.text(j, i, str(val), ha="center", va="center", fontsize=6, color=color)

    fig.tight_layout()
    path5 = os.path.join(OUTPUT_DIR, "plot_tactic_cooccurrence_heatmap.png")
    fig.savefig(path5, dpi=150)
    plt.close(fig)
    print(f"Saved: {path5}")

    # ── 6. Stacked bar — mean new vs reused tactics by turn index ─────────
    new_counts = stats["turn_new_counts"]
    reused_counts = stats["turn_reused_counts"]
    max_idx = min(int(indices.max()), 9)

    turn_idx_range = []
    mean_new_vals = []
    mean_reused_vals = []
    sample_sizes = []
    for idx in range(1, max_idx + 1):
        mask = indices == idx
        n = mask.sum()
        if n > 0:
            turn_idx_range.append(idx)
            mean_new_vals.append(new_counts[mask].mean())
            mean_reused_vals.append(reused_counts[mask].mean())
            sample_sizes.append(n)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(turn_idx_range))
    bars_new = ax.bar(x, mean_new_vals, color="steelblue", edgecolor="black", label="New tactics")
    bars_reused = ax.bar(x, mean_reused_vals, bottom=mean_new_vals, color="salmon", edgecolor="black", label="Reused tactics")

    # Annotate bar segments
    for i in range(len(turn_idx_range)):
        # New (bottom)
        if mean_new_vals[i] > 0.05:
            ax.text(i, mean_new_vals[i] / 2, f"{mean_new_vals[i]:.2f}", ha="center", va="center", fontsize=8, fontweight="bold")
        # Reused (top)
        if mean_reused_vals[i] > 0.05:
            ax.text(i, mean_new_vals[i] + mean_reused_vals[i] / 2, f"{mean_reused_vals[i]:.2f}", ha="center", va="center", fontsize=8, fontweight="bold")
        # Sample size above
        total_h = mean_new_vals[i] + mean_reused_vals[i]
        ax.text(i, total_h + 0.05, f"N={sample_sizes[i]}", ha="center", va="bottom", fontsize=7, color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in turn_idx_range])
    ax.set_xlabel("Assistant Turn Index")
    ax.set_ylabel("Mean Tactic Count")
    ax.set_title("Mean New vs Reused Tactics by Turn Index")
    ax.legend()
    fig.tight_layout()
    path6 = os.path.join(OUTPUT_DIR, "plot_new_vs_reused_by_turn.png")
    fig.savefig(path6, dpi=150)
    plt.close(fig)
    print(f"Saved: {path6}")

    # ── 7. Line plot — mean reward by turn index ─────────────────────────
    mean_rewards = []
    std_rewards = []
    for idx in turn_idx_range:
        mask = indices == idx
        mean_rewards.append(rewards[mask].mean())
        std_rewards.append(rewards[mask].std())
    mean_rewards = np.array(mean_rewards)
    std_rewards = np.array(std_rewards)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, mean_rewards, "o-", color="steelblue", linewidth=2, markersize=6, label="Mean reward")
    ax.fill_between(x, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2, color="steelblue", label="±1 std")
    ax.axhline(0, color="black", linestyle=":", alpha=0.5, label="y = 0")

    # Sample size labels above each data point
    for i in range(len(turn_idx_range)):
        ax.text(i, mean_rewards[i] + std_rewards[i] + 0.5, f"N={sample_sizes[i]}", ha="center", va="bottom", fontsize=7, color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in turn_idx_range])
    ax.set_xlabel("Assistant Turn Index")
    ax.set_ylabel("Mean Simulated Diversity Reward")
    ax.set_title("Mean Tactic Diversity Reward by Turn Index")
    ax.legend()
    fig.tight_layout()
    path7 = os.path.join(OUTPUT_DIR, "plot_mean_reward_by_turn.png")
    fig.savefig(path7, dpi=150)
    plt.close(fig)
    print(f"Saved: {path7}")


def main():
    print(f"Loading data from: {DATA_PATH}")
    conversations = load_data(DATA_PATH)
    print(f"Loaded {len(conversations)} conversations")

    stats = analyze(conversations)
    print_stats(stats, len(conversations))
    plot_all(stats)
    print("\nDone.")


if __name__ == "__main__":
    main()
