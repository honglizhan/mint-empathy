"""Pareto front: aggregated empathy vs. stickiness for 1.7B and 4B models."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from pathlib import Path
from matplotlib.colors import to_rgba

OUT_DIR = Path(__file__).resolve().parent / "results"

# Format: {name: (stickiness, agg_empathy, family)}
# Stickiness = avg P(tactic in model | tactic in gold last turn), lower = more diverse

METHODS_17B = {
    "Human (Gold)":      (0.27, 2.90, "human"),
    "Vanilla":           (0.51, 3.60, "prompt"),
    "PsychoCounsel":     (0.65, 4.42, "quality_rl"),
    "R1-Zero-Div":       (0.63, 4.38, "token_div"),
    r"Q+D$_{\mathrm{KL}}$": (0.51, 4.54, "ours"),
}

METHODS_4B = {
    "Human (Gold)":      (0.27, 2.90, "human"),
    "Vanilla":           (0.57, 3.75, "prompt"),
    "PsychoCounsel":     (0.67, 4.62, "quality_rl"),
    "R1-Zero-Div":       (0.57, 4.58, "token_div"),
    r"Q+D$_{\mathrm{KL}}$": (0.42, 4.67, "ours"),
}

# ---------- Style ----------
FAMILY_STYLE = {
    "human":      dict(color="#B8860B", marker="*",  s=300, zorder=5),
    "prompt":     dict(color="#7c9ab5", marker="s",  s=80,  zorder=3),
    "quality_rl": dict(color="#d4944c", marker="D",  s=90,  zorder=3),
    "token_div":  dict(color="#9b82c4", marker="^",  s=90,  zorder=3),
    "ours":       dict(color="#2563eb", marker="o",  s=140, zorder=7),
}

# Legend labels: specific method names
FAMILY_LABELS = {
    "human": "Human (Gold)",
    "prompt": "Vanilla",
    "quality_rl": "PsychoCounsel (Quality RL)",
    "token_div": "R1-Zero-Div (Token div.)",
    "ours": r"\textsc{Mint} (discourse div., ours)",
}


def scatter_methods(ax, methods, filled, plotted_families):
    """Scatter one set of methods. filled=True for 4B, False for 1.7B (open markers)."""
    for name, (xval, agg, fam) in methods.items():
        base = FAMILY_STYLE[fam]
        if fam == "human":
            if "human" in plotted_families:
                continue
            ax.scatter(xval, agg, color=base["color"], marker=base["marker"],
                       s=base["s"], zorder=base["zorder"],
                       edgecolors="k", linewidths=0.6)
        elif filled:
            ax.scatter(xval, agg, color=base["color"], marker=base["marker"],
                       s=base["s"], zorder=base["zorder"],
                       edgecolors="#555" if fam != "ours" else "k",
                       linewidths=0.6)
            # Halo glow behind Q+DKL 4B
            if fam == "ours":
                ax.scatter(xval, agg, color=base["color"], marker="o",
                           s=500, zorder=base["zorder"] - 1,
                           alpha=0.12, edgecolors="none")
        else:
            ax.scatter(xval, agg, facecolors="none", marker=base["marker"],
                       s=base["s"], zorder=base["zorder"],
                       edgecolors=base["color"], linewidths=1.5)
        plotted_families.add(fam)


# ---------- Plot ----------
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "legend.fontsize": 8,
})

fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.2))

XLIM = (0.74, 0.20)
YLIM = (2.7, 4.88)

# ---------- Green rectangle around both Q+DKL points ----------
ideal_rect = mpatches.Rectangle(
    (0.37, 4.42), 0.18, 0.38,
    facecolor=to_rgba("#10b981", 0.08),
    edgecolor=to_rgba("#10b981", 0.30), linewidth=0.8,
    zorder=0, transform=ax.transData,
)
ax.add_patch(ideal_rect)

# ---------- Human crosshair ----------
human_x, human_y = 0.27, 2.90
ax.axvline(x=human_x, color="#C59D2A", linestyle=":", linewidth=0.8, alpha=0.4, zorder=1)
ax.axhline(y=human_y, color="#C59D2A", linestyle=":", linewidth=0.8, alpha=0.4, zorder=1)

# ---------- Connect 1.7B and 4B points per method ----------
for name in METHODS_17B:
    if name == "Human (Gold)":
        continue
    if name in METHODS_4B:
        x1, y1, fam = METHODS_17B[name]
        x2, y2, _ = METHODS_4B[name]
        color = FAMILY_STYLE[fam]["color"]
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=0.8,
                linestyle=":", alpha=0.4, zorder=2)

# Scatter
plotted_families = set()
scatter_methods(ax, METHODS_17B, filled=False, plotted_families=plotted_families)
scatter_methods(ax, METHODS_4B, filled=True, plotted_families=plotted_families)

ax.set_ylabel(r"Aggregated Empathy $\longrightarrow$ more empathic")
ax.set_xlabel(r"Stickiness $\longrightarrow$ more diverse")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.yaxis.grid(True, alpha=0.12, linewidth=0.4)
ax.xaxis.grid(False)
ax.set_ylim(*YLIM)
ax.set_xlim(*XLIM)

# ---------- Legend: Method (with specific names) + Model size ----------
family_handles = []
for fam in ["human", "prompt", "quality_rl", "token_div", "ours"]:
    s = FAMILY_STYLE[fam]
    h = mlines.Line2D([], [], color=s["color"], marker=s["marker"],
                       linestyle="None", markersize=7,
                       markeredgecolor="k", markeredgewidth=0.4,
                       label=FAMILY_LABELS[fam])
    family_handles.append(h)

leg1 = ax.legend(handles=family_handles, loc="lower left", framealpha=0.92,
                 edgecolor="#ccc", handletextpad=0.3, borderpad=0.5,
                 labelspacing=0.45, fontsize=7, title=r"\textbf{Method}",
                 title_fontsize=7.5)

size_filled = mlines.Line2D([], [], color="#2563eb", marker="o", linestyle="None",
                             markersize=8, markerfacecolor="#2563eb",
                             markeredgecolor="k", markeredgewidth=0.5,
                             label=r"Qwen-3 4B (\textbf{solid})")
size_open = mlines.Line2D([], [], color="#2563eb", marker="o", linestyle="None",
                           markersize=8, markerfacecolor="none",
                           markeredgecolor="#2563eb", markeredgewidth=1.5,
                           label=r"Qwen-3 1.7B (\textbf{outline})")

leg2 = ax.legend(handles=[size_filled, size_open], framealpha=0.92,
                 edgecolor="#ccc", handletextpad=0.3, borderpad=0.5,
                 labelspacing=0.45, fontsize=7, title=r"\textbf{Model}",
                 title_fontsize=7.5,
                 bbox_to_anchor=(0.0, 0.35), loc="lower left")
ax.add_artist(leg1)

fig.tight_layout()
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_DIR / "pareto_front.pdf", bbox_inches="tight", dpi=300)
print(f"Saved pareto_front.pdf to {OUT_DIR}")
