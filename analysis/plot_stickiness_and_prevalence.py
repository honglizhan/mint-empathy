"""Two-panel dot plot: tactic prevalence (left) + stickiness (right).

Data (same as Section 3.3):
  - Human: 50 Lend-an-Ear gold conversations (315 supporter turns)
  - LLM:   322 natural multi-turn conversations from WildChat/SENSE-7
           (GPT-3.5-turbo, GPT-4, Llama-2-70B-Chat)

Output: results/stickiness_and_prevalence.pdf
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_JSON = SCRIPT_DIR.parent / "data" / "training" / "conversations_322_tagged.json"
GOLD_JSON = SCRIPT_DIR.parent / "evaluation" / "outputs" / "gold" / "conversations_tagged.json"
OUT_DIR = SCRIPT_DIR / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TACTICS = ["information", "assistance", "advice", "validation", "emotional_expression",
           "paraphrasing", "self_disclosure", "questioning", "reappraisal", "empowerment"]
DISPLAY = {"advice": "Advice", "information": "Information", "paraphrasing": "Paraphrasing",
           "questioning": "Questioning", "emotional_expression": "Emot. expression",
           "assistance": "Assistance", "validation": "Validation", "reappraisal": "Reappraisal",
           "empowerment": "Empowerment", "self_disclosure": "Self-disclosure"}
MODEL_MAP = {"gpt-3.5-turbo": "GPT-3.5", "gpt-4": "GPT-4", "GPT4": "GPT-4",
             "GPT4-empathy": "GPT-4", "Llama2-70b": "Llama-2-70B", "IC": "GPT-3.5"}
LLM_MODELS = ["GPT-3.5", "GPT-4", "Llama-2-70B"]
# ORDER will be set after loading data (sorted by LLM prevalence median, descending)
HUMAN_COLOR = "#D66B4B"

# ── Load gold (flat format) ──
with open(GOLD_JSON) as f:
    gold_items = json.load(f)

# Group gold by conversation for stickiness
gold_convos = defaultdict(list)
for item in gold_items:
    gold_convos[item["conversation_id"]].append(item)
for cid in gold_convos:
    gold_convos[cid].sort(key=lambda x: len(x.get("conversation_history", [])))

# ── Load training data (nested format), group by model ──
with open(TRAIN_JSON) as f:
    train_data = json.load(f)

# Extract per-model supporter turn tactic sets
model_turn_sets = defaultdict(list)  # model -> list of (conv_idx, [set, set, ...])
for conv in train_data:
    model = MODEL_MAP.get(conv.get("model"), None)
    if model is None:
        continue
    turns = []
    for turn in conv["conversation"]:
        if turn["role"] != "supporter":
            continue
        present = {t for st in turn.get("sentence_tactics", []) for t in st.get("tactics", []) if t in TACTICS}
        turns.append(present)
    model_turn_sets[model].append(turns)


def get_tactic_set(item):
    tc = item.get("tactic_counts", {})
    return {t for t in TACTICS if tc.get(t, 0) > 0}


# ── Compute prevalence ──
human_prev = {}
for t in TACTICS:
    human_prev[t] = sum(1 for item in gold_items if item.get("tactic_counts", {}).get(t, 0) > 0) / len(gold_items) * 100

llm_prev = {}
for model, convos in model_turn_sets.items():
    all_turns = [s for conv in convos for s in conv]
    llm_prev[model] = {t: sum(1 for s in all_turns if t in s) / len(all_turns) * 100 for t in TACTICS}

# Sort tactics by LLM prevalence median (descending)
ORDER = sorted(TACTICS, key=lambda t: float(np.median([llm_prev[m][t] for m in LLM_MODELS])), reverse=True)

# ── Compute stickiness ──
# Human: consecutive gold turns
human_stick_pp = defaultdict(int)
human_stick_denom = defaultdict(int)
human_stick_ap = defaultdict(int)
human_stick_adenom = defaultdict(int)
for turns in gold_convos.values():
    for i in range(len(turns) - 1):
        curr = get_tactic_set(turns[i])
        nxt = get_tactic_set(turns[i + 1])
        for t in TACTICS:
            if t in curr:
                human_stick_denom[t] += 1
                if t in nxt:
                    human_stick_pp[t] += 1
            else:
                human_stick_adenom[t] += 1
                if t in nxt:
                    human_stick_ap[t] += 1

human_stick_pres = {t: (human_stick_pp[t] / human_stick_denom[t] * 100 if human_stick_denom[t] else 0) for t in TACTICS}
human_stick_abs = {t: (human_stick_ap[t] / human_stick_adenom[t] * 100 if human_stick_adenom[t] else 0) for t in TACTICS}

# LLM: consecutive supporter turns within same conversation
llm_stick_pres = {}
llm_stick_abs = {}
for model, convos in model_turn_sets.items():
    pp, denom, ap, adenom = defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int)
    for conv in convos:
        for i in range(1, len(conv)):
            prev, curr = conv[i - 1], conv[i]
            for t in TACTICS:
                if t in prev:
                    denom[t] += 1
                    if t in curr:
                        pp[t] += 1
                else:
                    adenom[t] += 1
                    if t in curr:
                        ap[t] += 1
    llm_stick_pres[model] = {t: (pp[t] / denom[t] * 100 if denom[t] else 0) for t in TACTICS}
    llm_stick_abs[model] = {t: (ap[t] / adenom[t] * 100 if adenom[t] else 0) for t in TACTICS}


def style_ax(ax):
    ax.grid(axis="x", color="#ECECEC", lw=0.7)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color("#B8B8B8")


# ── Plot ──
tactics = ORDER[::-1]
n = len(tactics)
y = np.arange(n)

fig, (ax_p, ax_s) = plt.subplots(1, 2, sharey=True, figsize=(9, 3.5))

# --- Right panel: Stickiness ---
STICK_BAND = "#B3CDE3"
STICK_MED = "#2C6BA0"
ABS_BAND = "#D8D8D8"
ABS_MED = "#999999"

for i, tac in enumerate(tactics):
    # Present conditional
    h_pres = human_stick_pres[tac]
    llm_pres_vals = [llm_stick_pres[m][tac] for m in LLM_MODELS]
    s_min, s_max, s_med = min(llm_pres_vals), max(llm_pres_vals), float(np.median(llm_pres_vals))
    # Absent conditional
    h_abs = human_stick_abs[tac]
    llm_abs_vals = [llm_stick_abs[m][tac] for m in LLM_MODELS]
    a_min, a_max, a_med = min(llm_abs_vals), max(llm_abs_vals), float(np.median(llm_abs_vals))

    # Connector
    ax_s.hlines(y[i], min(h_abs, a_min), max(h_pres, s_max), color="#ECECEC", lw=1.2, zorder=1)
    # Absent band + dot
    ax_s.hlines(y[i], a_min, a_max, color=ABS_BAND, lw=5, capstyle="round", zorder=2)
    ax_s.scatter(a_med, y[i], s=28, color=ABS_MED, edgecolors="white", lw=0.5, zorder=3)
    # Present band + dot
    ax_s.hlines(y[i], s_min, s_max, color=STICK_BAND, lw=7, capstyle="round", zorder=4)
    ax_s.scatter(s_med, y[i], s=42, color=STICK_MED, edgecolors="white", lw=0.6, zorder=5)

# Human absent (open grey diamonds)
ax_s.scatter([human_stick_abs[t] for t in tactics], y, s=45, facecolors="none",
             edgecolors="#999999", lw=1.0, zorder=6, marker="D")
# Human present (filled red diamonds)
ax_s.scatter([human_stick_pres[t] for t in tactics], y, s=55, color=HUMAN_COLOR,
             edgecolors="white", lw=0.6, zorder=7, marker="D")

ax_s.set_yticks(y)
ax_s.set_yticklabels([DISPLAY[t] for t in tactics])
ax_s.set_xlabel(r"Conditional probability of tactic in turn $t$  [%]", fontsize=8)
ax_s.set_title("Tactic Stickiness", fontsize=9, fontweight="bold", pad=6)
ax_s.set_xlim(-2, 100)
ax_s.set_ylim(-0.6, n - 0.4)
style_ax(ax_s)

# Stickiness legend: two groups with title headers, stacked vertically
title_fp = FontProperties(size=6, style="italic")
pres_handles = [
    Line2D([], [], marker="D", ls="None", ms=5, mfc=HUMAN_COLOR, mec="white", mew=0.5, label="Human"),
    Line2D([], [], marker="o", ls="None", ms=4.5, mfc=STICK_MED, mec="white", mew=0.5, label="LLM med."),
    Line2D([], [], lw=4, color=STICK_BAND, solid_capstyle="round", label="LLM range"),
]
abs_handles = [
    Line2D([], [], marker="D", ls="None", ms=5, mfc="none", mec="#999999", mew=0.8, label="Human"),
    Line2D([], [], marker="o", ls="None", ms=4.5, mfc=ABS_MED, mec="white", mew=0.5, label="LLM med."),
    Line2D([], [], lw=4, color=ABS_BAND, solid_capstyle="round", label="LLM range"),
]
leg_pres = ax_s.legend(handles=pres_handles,
    title=r"$P\,(\mathrm{in}\ t \mid \mathrm{in}\ t\!-\!1)$",
    title_fontproperties=title_fp, prop={"size": 5.5}, frameon=True,
    framealpha=0.9, edgecolor="#CCCCCC", loc="lower right",
    bbox_to_anchor=(1.0, 0.22), handlelength=1.2, handletextpad=0.5, labelspacing=0.25)
ax_s.add_artist(leg_pres)
ax_s.legend(handles=abs_handles,
    title=r"$P\,(\mathrm{in}\ t \mid \mathrm{NOT\ in}\ t\!-\!1)$",
    title_fontproperties=title_fp, prop={"size": 5.5}, frameon=True,
    framealpha=0.9, edgecolor="#CCCCCC", loc="lower right",
    bbox_to_anchor=(1.0, 0.0), handlelength=1.2, handletextpad=0.5, labelspacing=0.25)

# --- Left panel: Prevalence ---
PREV_BAND = "#B2DFBC"
PREV_MED = "#2D8E4E"

for i, tac in enumerate(tactics):
    h = human_prev[tac]
    vals = [llm_prev[m][tac] for m in LLM_MODELS]
    lo, hi, med = min(vals), max(vals), float(np.median(vals))
    ax_p.hlines(y[i], min(h, lo), max(h, hi), color="#ECECEC", lw=1.2, zorder=1)
    ax_p.hlines(y[i], lo, hi, color=PREV_BAND, lw=7, capstyle="round", zorder=2)
    ax_p.scatter(med, y[i], s=42, color=PREV_MED, edgecolors="white", lw=0.6, zorder=3)

ax_p.scatter([human_prev[t] for t in tactics], y, s=55, color=HUMAN_COLOR,
             edgecolors="white", lw=0.6, zorder=4, marker="D")

ax_p.set_xlabel("Turns containing tactic  [%]", fontsize=8)
ax_p.set_title("Tactic Prevalence", fontsize=9, fontweight="bold", pad=6)
ax_p.set_xlim(-2, 100)
style_ax(ax_p)

# Prevalence legend (lower right of plot area)
ax_p.legend(handles=[
    Line2D([], [], marker="D", ls="None", ms=6, mfc=HUMAN_COLOR, mec="white", mew=0.5, label="Human"),
    Line2D([], [], marker="o", ls="None", ms=5.5, mfc=PREV_MED, mec="white", mew=0.5, label="LLM median"),
    Line2D([], [], lw=5, color=PREV_BAND, solid_capstyle="round", label="LLM range (3 models)"),
], prop={"size": 6}, frameon=True, framealpha=0.9, edgecolor="#CCCCCC",
    loc="lower right", handlelength=1.2, handletextpad=0.6, labelspacing=0.3)

plt.tight_layout()
fig.subplots_adjust(wspace=0.05)
out_path = OUT_DIR / "stickiness_and_prevalence.pdf"
fig.savefig(out_path, bbox_inches="tight", dpi=300)
print(f"Saved {out_path}")
