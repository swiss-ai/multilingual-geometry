"""
plot_ablation_results.py — raw COMET-Kiwi score distributions per dose,
forward vs. reversed, one panel per pair. No embedded monotonicity test
(that's visual — look at the boxplots) and no LID collapse flagging on
this plot (that's a separate check, run add_lid_to_ablation.py and look
at collapse_rate directly, or plot_score_by_dose.py's monotonicity view
if you want that specific test back).

alpha=0 (baseline, no ablation) is included as the leftmost box in each
panel, using each row's own baseline_comet_kiwi, so the distribution
shift is visible from a real "no intervention" starting point.

Usage:
    python plot_ablation_results.py \
        --results_dir ablation_results \
        --out_dir figures
"""

import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--results_dir", type=str, default="ablation_results")
parser.add_argument("--out_dir", type=str, default="figures")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

# Corrected per the project's established reasoning: Afrikaans's closest,
# highest-resource relative is DUTCH, not German — af_nl is the primary
# Germanic pair (like gl_es for Romance), af_de is the weaker/more
# distant secondary relationship.
PAIR_TYPE = {
    "gl_es": "primary",         "af_nl": "primary",
    "ca_es": "secondary",       "nl_de": "secondary",
    "af_de": "weaker",
    "gl_ca": "within-family null",
    "gl_de": "cross-family control", "af_es": "cross-family control",
}

# Per-pair colors, forward (saturated) / reversed (light), matching the
# original thesis figures' palette convention.
PAIR_COLOR = {
    "gl_es": ("#534AB7", "#AFA9EC"),
    "gl_ca": ("#D4537E", "#ED93B1"),
    "es_ca": ("#D85A30", "#F0997B"), "ca_es": ("#D85A30", "#F0997B"),
    "gl_de": ("#888780", "#C0BDBA"),
    "af_nl": ("#0F6E56", "#5DCAA5"),
    "af_de": ("#EF9F27", "#FAC775"),
    "nl_de": ("#185FA5", "#85B7EB"),
    "af_es": ("#888780", "#C0BDBA"),
}
DEFAULT_COLOR = ("#534AB7", "#AFA9EC")

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
})

PAIR_RE = re.compile(r"^([a-z]+)_([a-z]+)_ablation_scored(_with_lid)?\.csv$")


def find_pair_files(results_dir):
    """One file per pair; the plain _ablation_scored.csv and the
    _with_lid version have identical comet_kiwi/baseline columns, so
    either works here — LID columns are simply ignored by this script."""
    files = {}
    for path in glob.glob(os.path.join(results_dir, "*_ablation_scored*.csv")):
        m = PAIR_RE.match(os.path.basename(path))
        if not m:
            continue
        lr, hr, has_lid = m.group(1), m.group(2), m.group(3)
        key = f"{lr}_{hr}"
        if key not in files or has_lid:
            files[key] = path
    return files


def collect_boxplot_data(df, direction):
    """Returns (labels, list_of_score_arrays) with baseline as the first
    entry, then one entry per alpha, ascending."""
    sub = df[df["direction"] == direction]
    if sub.empty:
        return None, None
    alphas = sorted(sub["alpha"].unique())
    labels = ["baseline"] + [f"\u03b1={a}" for a in alphas]
    groups = [sub["baseline_comet_kiwi"].values]
    for a in alphas:
        groups.append(sub[sub["alpha"] == a]["comet_kiwi"].values)
    return labels, groups


pair_files = find_pair_files(args.results_dir)
if not pair_files:
    raise SystemExit(f"No *_ablation_scored*.csv files found in {args.results_dir}")

data = {key: pd.read_csv(path) for key, path in pair_files.items()}
print(f"Loaded {len(data)} pairs: {list(data.keys())}")

n_pairs = len(data)
n_cols = min(4, n_pairs)
n_rows = int(np.ceil(n_pairs / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.4 * n_cols, 3.8 * n_rows), squeeze=False)

for idx, (key, df) in enumerate(data.items()):
    ax = axes[idx // n_cols][idx % n_cols]
    color_fwd, color_rev = PAIR_COLOR.get(key, DEFAULT_COLOR)

    fwd_labels, fwd_groups = collect_boxplot_data(df, "forward")
    rev_labels, rev_groups = collect_boxplot_data(df, "reversed")
    labels = fwd_labels or rev_labels
    n_x = len(labels)
    x = np.arange(n_x)
    width = 0.32

    box_kwargs = dict(widths=width * 0.9, patch_artist=True, showfliers=False,
                       medianprops=dict(color="#222", linewidth=1.2))

    if fwd_groups is not None:
        bp = ax.boxplot(fwd_groups, positions=x - width / 2, **box_kwargs)
        for patch in bp["boxes"]:
            patch.set_facecolor(color_fwd)
            patch.set_alpha(0.85)
    if rev_groups is not None:
        bp = ax.boxplot(rev_groups, positions=x + width / 2, **box_kwargs)
        for patch in bp["boxes"]:
            patch.set_facecolor(color_rev)
            patch.set_alpha(0.85)

    ax.axvline(0.5, color="#ccc", linestyle=":", linewidth=0.8, zorder=0)
    pair_type = PAIR_TYPE.get(key, "unclassified")
    ax.set_title(f"{key.replace('_', '/')}  ({pair_type})", fontsize=9.5, fontweight="500", pad=6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5, rotation=0)
    ax.set_ylabel("COMET-Kiwi score", fontsize=8)
    ax.tick_params(labelsize=7.5)
    ax.grid(True, axis="y", alpha=0.15, linewidth=0.4)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=color_fwd, alpha=0.85),
               plt.Rectangle((0, 0), 1, 1, facecolor=color_rev, alpha=0.85)]
    ax.legend(handles, ["forward", "reversed"], fontsize=6.5, frameon=False, loc="best")

for idx in range(n_pairs, n_rows * n_cols):
    axes[idx // n_cols][idx % n_cols].axis("off")

fig.suptitle("COMET-Kiwi score distribution by dose \u2014 forward vs. reversed",
             fontsize=13, fontweight="bold", y=1.02)
fig.text(0.5, -0.02,
          "Boxes = quartiles, line = median. \"baseline\" = alpha=0, no ablation. "
          "LID collapse check is a separate step (add_lid_to_ablation.py).",
          ha="center", fontsize=8, color="#666", style="italic")
fig.tight_layout(rect=[0, 0, 1, 0.95])
grid_path = os.path.join(args.out_dir, "score_distribution_grid.png")
fig.savefig(grid_path, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {grid_path}")