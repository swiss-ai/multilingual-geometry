"""
visualise_projections.py — Plot per-layer projection magnitudes for each
source/axis pair, with selected layers highlighted.

Reads CSVs output by compute_projections.py and produces:
  1. proj_all_pairs.pdf   — all pairs on one plot, shaded selected layers
  2. proj_per_pair.pdf    — faceted, one panel per pair

Layer selection criterion (matching ablation logic):
  Layers in the second half (l >= 16) where mean_abs_proj exceeds
  the second-half mean by at least `--threshold_sigma` standard deviations.
  Intersection across all pairs is highlighted separately.

Usage:
    python visualise_projections.py \
        --input  probe-results-llama/proj_all_pairs.csv \
        --output_dir figures-llama/ \
        --threshold_sigma 0.5

    # Or individual CSVs:
    python visualise_projections.py \
        --input probe-results-llama/proj_gl_es.csv probe-results-llama/proj_af_nl.csv
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ── Args ─────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--input", nargs="+", required=True)
parser.add_argument("--output_dir", type=str, default="figures-llama")
parser.add_argument("--fmt", type=str, default="png",
                    choices=["pdf", "png", "svg"])
parser.add_argument("--dpi", type=int, default=200)
parser.add_argument("--threshold_sigma", type=float, default=0.5,
                    help="Threshold for layer selection: mean + k*std (default 0.5)")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────────────

input_files = []
for pattern in args.input:
    matched = glob.glob(pattern)
    input_files.extend(matched) if matched else (
        input_files.append(pattern) if os.path.isfile(pattern) else None
    )

df = pd.concat([pd.read_csv(f) for f in sorted(set(input_files))],
               ignore_index=True)

pairs = df.groupby(["source_lang", "axis_lang"]).size().reset_index()[
    ["source_lang", "axis_lang"]
].values.tolist()
pairs = [tuple(p) for p in pairs]

print(f"Loaded {len(pairs)} pairs: {pairs}")

# ── Compute selected layers per pair ─────────────────────────────────────────

COLORS = {
    ("gl", "es"): "#378ADD",
    ("gl", "ca"): "#1D9E75",
    ("af", "nl"): "#D85A30",
    ("af", "de"): "#8B5CA0",
}

pair_selected = {}
pair_data     = {}

for source, axis in pairs:
    sub = df[(df["source_lang"] == source) & (df["axis_lang"] == axis)].copy()
    sub = sub.sort_values("layer").reset_index(drop=True)

    second_half = sub[sub["layer"] >= 12]["mean_abs_proj"]
    threshold   = second_half.mean() + args.threshold_sigma * second_half.std()
    selected    = set(sub[
        (sub["layer"] >= 12) & (sub["mean_abs_proj"] >= threshold)
    ]["layer"].tolist())

    pair_selected[(source, axis)] = selected
    pair_data[(source, axis)]     = sub

    print(f"  {source}/{axis}: threshold={threshold:.4f}, "
          f"selected={sorted(selected)}")

romance_pairs  = [(s, a) for s, a in pairs if s == "gl"]
germanic_pairs = [(s, a) for s, a in pairs if s == "af"]

romance_intersection  = set.intersection(*[pair_selected[p] for p in romance_pairs])  if romance_pairs  else set()
germanic_intersection = set.intersection(*[pair_selected[p] for p in germanic_pairs]) if germanic_pairs else set()

print(f"\nRomance  intersection (gl): {sorted(romance_intersection)}")
print(f"Germanic intersection (af): {sorted(germanic_intersection)}")

from matplotlib.patches import Patch
from matplotlib.lines import Line2D

def savefig(name):
    path = os.path.join(args.output_dir, f"{name}.{args.fmt}")
    plt.savefig(path, dpi=args.dpi, bbox_inches="tight")
    plt.close()
    print(f"  → {path}")

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": args.dpi,
})

BG = "#FFFFFF"

INTERSECTION_COLORS = {
    "romance":  "#D85A30",   # coral
    "germanic": "#8B5CA0",   # purple
}

# ── Helper: draw one family plot ─────────────────────────────────────────────

def plot_family(family_pairs, family_intersection, family_name, int_color, fname):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    for l in family_intersection:
        ax.axvspan(l - 0.5, l + 0.5, color=int_color, alpha=0.15, zorder=0)

    ax.axvline(12, color="#999999", linewidth=0.8, linestyle=":", zorder=1)

    for source, axis in family_pairs:
        sub      = pair_data[(source, axis)]
        color    = COLORS.get((source, axis), "gray")
        selected = pair_selected[(source, axis)]

        layers = sub["layer"].values
        mean   = sub["mean_abs_proj"].values
        std    = sub["std_abs_proj"].values

        ax.fill_between(layers, mean - std, mean + std, alpha=0.08, color=color)
        ax.plot(layers, mean, color=color, linewidth=1.8,
                label=f"{source}/{axis}", zorder=3)

        sel_idx = [i for i, l in enumerate(layers) if l in selected]
        ax.scatter(layers[sel_idx], mean[sel_idx],
                   color=color, s=40, zorder=5, marker="o")

        for l in family_intersection:
            idx = list(layers).index(l) if l in layers else None
            if idx is not None:
                ax.vlines(l, ymin=0, ymax=mean[idx],
                          color=int_color, linewidth=1.2,
                          linestyle=":", zorder=4)

    ax.set_xlabel("Transformer layer")
    ax.set_ylabel(r"Mean $|\mathbf{h}^{(\ell)} \cdot \hat{\mathbf{a}}^{(\ell)}|$")
    ax.set_title(
        f"Projection magnitude onto high-resource axis — {family_name} family",
        fontsize=10, fontweight="bold"
    )
    ax.xaxis.set_major_locator(plt.MultipleLocator(4))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    handles, labels = ax.get_legend_handles_labels()
    handles += [
        Patch(facecolor=int_color, alpha=0.3,
              label=f"Intersection: L{sorted(family_intersection)}"),
        Line2D([0], [0], color="#999999", linestyle=":", linewidth=0.8,
               label="L16 boundary"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper left")
    sns.despine(ax=ax)
    fig.tight_layout()
    savefig(fname)

# ── 1. Romance plot ───────────────────────────────────────────────────────────

print("\n[1] Romance family plot...")
plot_family(romance_pairs, romance_intersection, "Romance",
            INTERSECTION_COLORS["romance"], "proj_romance")

# ── 2. Germanic plot ──────────────────────────────────────────────────────────

print("[2] Germanic family plot...")
plot_family(germanic_pairs, germanic_intersection, "Germanic",
            INTERSECTION_COLORS["germanic"], "proj_germanic")

# ── Summary table ─────────────────────────────────────────────────────────────

rows = []
for source, axis in pairs:
    sub      = pair_data[(source, axis)]
    selected = pair_selected[(source, axis)]
    family   = "romance" if source == "gl" else "germanic"
    fint     = romance_intersection if family == "romance" else germanic_intersection
    rows.append({
        "pair":              f"{source}/{axis}",
        "family":            family,
        "selected_layers":   str(sorted(selected)),
        "n_selected":        len(selected),
        "family_intersection": str(sorted(fint)),
        "threshold_sigma":   args.threshold_sigma,
        "peak_layer":        int(sub.loc[sub["mean_abs_proj"].idxmax(), "layer"]),
        "peak_proj":         round(sub["mean_abs_proj"].max(), 5),
    })

tbl = pd.DataFrame(rows)
tbl_path = os.path.join(args.output_dir, "proj_layer_selection.csv")
tbl.to_csv(tbl_path, index=False)
print(f"\nLayer selection summary → {tbl_path}")
print(tbl.to_string(index=False))
print("\nDone.")