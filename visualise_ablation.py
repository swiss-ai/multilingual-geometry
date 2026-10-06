"""
visualise_ablation.py — Visualise causal ablation results.

Produces:
  1. key_results.png       — forward pairs α=0.3, good only
  2. full_results.png      — forward + backward pairs × alpha × group
  3. alpha_effect.png      — dose-response per forward pair, good only
  4. asymmetry_grid.png    — forward vs reversed columns, all alphas, good only
  5. fwd_rev_lines.png     — forward vs reversed dose-response lines per pair
  6. heatmap.png           — all pairs × alpha heatmap, good translations
  7. scatter_asym.png      — geometric asymmetry vs causal effect scatter

Usage:
    python visualise_ablation.py \
        --input      ablation_results_diagnostic/ablation_summary_vs_none_n100.csv \
        --output_dir figures/ablation \
        --containment_csv figures/containment/containment_summary.csv
"""

import argparse, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--input",           type=str, default="ablation_results_final/ablation_summary_vs_none_n100.csv")
parser.add_argument("--output_dir",      type=str, default="figures/ablation")
parser.add_argument("--containment_csv", type=str, default="figures/containment/containment_summary.csv",
                    help="Optional: path to containment_summary.csv for scatter plot")
parser.add_argument("--dpi",             type=int, default=200)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
df = pd.read_csv(args.input)

if "source_lang" in df.columns:
    df = df.rename(columns={"source_lang": "source", "axis_lang": "axis"})
if "group_name" in df.columns:
    df = df.rename(columns={"group_name": "group"})

# ── Colours ────────────────────────────────────────────────────────────────────
COLORS = {
    ("gl", "es"): "#534AB7",  ("es", "gl"): "#AFA9EC",
    ("gl", "ca"): "#D4537E",  ("ca", "gl"): "#ED93B1",
    ("es", "ca"): "#D85A30",  ("ca", "es"): "#F0997B",
    ("gl", "de"): "#888780",  ("de", "gl"): "#C0BDBA",
    ("af", "nl"): "#0F6E56",  ("nl", "af"): "#5DCAA5",
    ("af", "de"): "#EF9F27",  ("de", "af"): "#FAC775",
    ("nl", "de"): "#185FA5",  ("de", "nl"): "#85B7EB",
    ("af", "es"): "#888780",  ("es", "af"): "#C0BDBA",
}

ROLE = {
    ("gl", "es"): "primary",   ("es", "gl"): "reversed",
    ("gl", "ca"): "control",   ("ca", "gl"): "reversed",
    ("es", "ca"): "secondary", ("ca", "es"): "reversed",
    ("gl", "de"): "xfamily",   ("de", "gl"): "reversed",
    ("af", "nl"): "primary",   ("nl", "af"): "reversed",
    ("af", "de"): "weaker",    ("de", "af"): "reversed",
    ("nl", "de"): "secondary", ("de", "nl"): "reversed",
    ("af", "es"): "xfamily",   ("es", "af"): "reversed",
}

def color(src, ax): return COLORS.get((src, ax), "#888780")
def role(src, ax):  return ROLE.get((src, ax), "unknown")

existing = set(zip(df["source"], df["axis"]))

PAIRS_FWD = [p for p in [
    ("gl", "es"), ("gl", "ca"), ("es", "ca"), ("gl", "de"),
    ("af", "nl"), ("af", "de"), ("nl", "de"), ("af", "es"),
] if p in existing]

PAIRS_REV = [(b, a) for (a, b) in PAIRS_FWD if (b, a) in existing]
BI_PAIRS  = [(f, (f[1], f[0])) for f in PAIRS_FWD if (f[1], f[0]) in existing]
PAIRS_ALL = PAIRS_FWD + [r for r in PAIRS_REV if r not in PAIRS_FWD]

alphas = sorted(df["alpha"].unique())

# ── Helpers ────────────────────────────────────────────────────────────────────
def savefig(name):
    path = os.path.join(args.output_dir, f"{name}.png")
    plt.savefig(path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  -> {path}")

def eb(ax, x, y, se):
    ax.errorbar(x, y, yerr=se, fmt="none", color="#444", capsize=3, linewidth=1.0, zorder=5)

def zl(ax):
    ax.axhline(0, color="#888", linewidth=0.7, linestyle="--", zorder=1)

def get(sub, src, ax):
    row = sub[(sub["source"] == src) & (sub["axis"] == ax)]
    if len(row) == 0: return None, None
    return float(row["delta_vs_none"].iloc[0]), float(row["se_delta_vs_none"].iloc[0])

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
})

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Key results — forward pairs α=0.3, good only
# ═══════════════════════════════════════════════════════════════════════════════
print("[1] Key results...")
fig, ax = plt.subplots(figsize=(11, 4.5))
sub = df[(df["alpha"] == 0.3) & (df["group"] == "good")]
xs = np.arange(len(PAIRS_FWD))
deltas, ses, cols = [], [], []
for p in PAIRS_FWD:
    d, s = get(sub, *p); deltas.append(d or 0); ses.append(s or 0); cols.append(color(*p))
ax.bar(xs, deltas, color=cols, width=0.5, zorder=3, edgecolor="white")
for x, y, se in zip(xs, deltas, ses):
    eb(ax, x, y, se)
    ax.text(x, y - 0.012 if y < 0 else y + 0.008, f"{y:+.3f}",
            ha="center", fontsize=8, fontweight="500", color="#2C2C2A")
zl(ax)
ax.set_xticks(xs); ax.set_xticklabels([f"{s}/{a}" for s, a in PAIRS_FWD], fontsize=9)
ax.set_ylabel("Δ COMET-Kiwi vs none", fontsize=9); ax.set_ylim(-0.32, 0.12)
ax.grid(True, axis="y", alpha=0.15, linewidth=0.4)
ax.set_title("Causal ablation — forward pairs, α=0.3, good translations", fontsize=10, fontweight="bold", pad=8)
plt.tight_layout(); savefig("key_results")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Full results grid — forward + backward × alpha × group
# ═══════════════════════════════════════════════════════════════════════════════
print("[2] Full results grid (fwd + rev)...")
groups = ["good", "bad"]
n_all = len(PAIRS_ALL)
fig, axes = plt.subplots(len(groups), len(alphas),
                          figsize=(4.5 * len(alphas), 4.0 * len(groups)),
                          sharey=True,
                          gridspec_kw={"hspace": 0.5, "wspace": 0.12})
if len(alphas) == 1: axes = axes.reshape(-1, 1)
for gi, group in enumerate(groups):
    for ai, alpha in enumerate(alphas):
        ax = axes[gi, ai]
        sub = df[(df["alpha"] == alpha) & (df["group"] == group)]
        deltas, ses, cols = [], [], []
        for p in PAIRS_ALL:
            d, s = get(sub, *p); deltas.append(d or 0); ses.append(s or 0)
            c = color(*p)
            cols.append(c if group == "good" else c + "99")
        xs = np.arange(n_all)
        ax.bar(xs, deltas, color=cols, width=0.5, zorder=3, edgecolor="white")
        for x, y, se in zip(xs, deltas, ses):
            eb(ax, x, y, se)
            ax.text(x, y - 0.008 if y < 0 else y + 0.006, f"{y:+.2f}",
                    ha="center", fontsize=6, color="#2C2C2A")
        zl(ax)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{s}/{a}" for s, a in PAIRS_ALL], fontsize=6, rotation=45, ha="right")
        ax.set_title(f"α={alpha} — {group}", fontsize=8, pad=3)
        ax.grid(True, axis="y", alpha=0.15, linewidth=0.4); ax.set_ylim(-0.35, 0.15)
        if ai == 0: ax.set_ylabel("Δ COMET-Kiwi vs none", fontsize=8)
        # shade reversed pairs
        rev_xs = [i for i, p in enumerate(PAIRS_ALL) if role(*p) == "reversed"]
        for rx in rev_xs:
            ax.axvspan(rx - 0.4, rx + 0.4, color="#f0f0f0", zorder=0, alpha=0.5)
fig.suptitle("Ablation — all pairs (forward + reversed), all α", fontsize=10, fontweight="bold", y=1.01)
fig.text(0.5, -0.01, "Shaded bars = reversed direction (hr source → lr axis)", ha="center", fontsize=7, color="#666", style="italic")
plt.tight_layout(); savefig("full_results")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Alpha dose-response — forward pairs, good only
# ═══════════════════════════════════════════════════════════════════════════════
print("[3] Alpha effect...")
n_fwd = len(PAIRS_FWD)
fig, axes = plt.subplots(1, n_fwd, figsize=(3.0 * n_fwd, 4.5), sharey=True, gridspec_kw={"wspace": 0.15})
if n_fwd == 1: axes = [axes]
for i, (src, ax_lang) in enumerate(PAIRS_FWD):
    ax = axes[i]
    sub = df[(df["source"] == src) & (df["axis"] == ax_lang) & (df["group"] == "good")]
    xs, deltas, ses = [], [], []
    for j, alpha in enumerate(alphas):
        row = sub[sub["alpha"] == alpha]
        d = float(row["delta_vs_none"].iloc[0]) if len(row) else 0.0
        s = float(row["se_delta_vs_none"].iloc[0]) if len(row) else 0.0
        xs.append(j); deltas.append(d); ses.append(s)
    ax.bar(xs, deltas, color=color(src, ax_lang), width=0.45, zorder=3, edgecolor="white")
    for x, y, se in zip(xs, deltas, ses):
        eb(ax, x, y, se)
        ax.text(x, y - 0.012 if y < 0 else y + 0.008, f"{y:+.3f}", ha="center", fontsize=7, color="#2C2C2A")
    zl(ax); ax.set_xticks(xs); ax.set_xticklabels([f"α={a}" for a in alphas], fontsize=7)
    ax.set_title(f"{src}/{ax_lang}", fontsize=9, fontweight="500", pad=4)
    ax.grid(True, axis="y", alpha=0.15, linewidth=0.4); ax.set_ylim(-0.35, 0.12)
    if i == 0: ax.set_ylabel("Δ COMET-Kiwi vs none", fontsize=8)
fig.suptitle("Dose-response — forward pairs, good translations", fontsize=10, fontweight="bold")
plt.tight_layout(); savefig("alpha_effect")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. Asymmetry grid — forward vs reversed columns, all alphas
# ═══════════════════════════════════════════════════════════════════════════════
print("[4] Asymmetry grid...")
if BI_PAIRS:
    n_bi = len(BI_PAIRS)
    fig, axes = plt.subplots(len(alphas), n_bi,
                              figsize=(3.5 * n_bi, 3.0 * len(alphas)),
                              sharey=True,
                              gridspec_kw={"hspace": 0.55, "wspace": 0.2})
    if len(alphas) == 1: axes = axes.reshape(1, -1)
    if n_bi == 1: axes = axes.reshape(-1, 1)
    sub_all = df[df["group"] == "good"]
    for ai, alpha in enumerate(alphas):
        sub = sub_all[sub_all["alpha"] == alpha]
        for bi, (fwd, rev) in enumerate(BI_PAIRS):
            ax = axes[ai, bi]
            d_f, se_f = get(sub, *fwd)
            d_r, se_r = get(sub, *rev)
            xs = [0, 1]; ds = [d_f or 0, d_r or 0]; ss = [se_f or 0, se_r or 0]
            cs = [color(*fwd), color(*rev)]
            lbls = [f"{fwd[0]}/{fwd[1]}\n→", f"{rev[0]}/{rev[1]}\n→"]
            ax.bar(xs, ds, color=cs, width=0.55, zorder=3, edgecolor="white")
            for x, y, se in zip(xs, ds, ss):
                eb(ax, x, y, se)
                ypos = y - 0.015 if y < 0 else y + 0.010
                ax.text(x, ypos, f"{y:+.3f}", ha="center", fontsize=8, fontweight="500", color="#2C2C2A")
            zl(ax); ax.set_xticks(xs); ax.set_xticklabels(lbls, fontsize=8)
            ax.grid(True, axis="y", alpha=0.15, linewidth=0.4); ax.set_ylim(-0.35, 0.15)
            if ai == 0: ax.set_title(f"{fwd[0]}/{fwd[1]}\nvs\n{rev[0]}/{rev[1]}", fontsize=8.5, fontweight="500", pad=6)
            if bi == 0: ax.set_ylabel(f"α={alpha}\nΔ COMET-Kiwi", fontsize=8)
    fig.suptitle("Causal asymmetry — forward vs reversed, good translations", fontsize=10, fontweight="bold", y=1.01)
    fig.text(0.5, -0.02,
             "Stepping-stone predicts: forward degrades more than reversed.\nEach column = one pair; each row = one α.",
             ha="center", fontsize=8, color="#5F5E5A", style="italic")
    plt.tight_layout(); savefig("asymmetry_grid")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. Forward vs reversed lines — dose-response on same axis per pair
# ═══════════════════════════════════════════════════════════════════════════════
print("[5] Forward vs reversed lines...")
if BI_PAIRS:
    n_bi = len(BI_PAIRS)
    fig, axes = plt.subplots(1, n_bi, figsize=(4.5 * n_bi, 4.5), sharey=True,
                              gridspec_kw={"wspace": 0.2})
    if n_bi == 1: axes = [axes]
    for i, (fwd, rev) in enumerate(BI_PAIRS):
        ax = axes[i]
        for pair, ls, lw, label in [(fwd, "-", 2.0, f"{fwd[0]}/{fwd[1]} (fwd)"),
                                     (rev, "--", 1.4, f"{rev[0]}/{rev[1]} (rev)")]:
            sub = df[(df["source"] == pair[0]) & (df["axis"] == pair[1]) & (df["group"] == "good")]
            ds, ses = [], []
            for alpha in alphas:
                row = sub[sub["alpha"] == alpha]
                ds.append(float(row["delta_vs_none"].iloc[0]) if len(row) else 0.0)
                ses.append(float(row["se_delta_vs_none"].iloc[0]) if len(row) else 0.0)
            ds = np.array(ds); ses = np.array(ses)
            c = color(*pair)
            ax.plot(alphas, ds, color=c, linewidth=lw, linestyle=ls,
                    marker="o", markersize=5, label=label, zorder=3)
            ax.fill_between(alphas, ds - ses, ds + ses, color=c, alpha=0.12)
        zl(ax)
        ax.set_xlabel("α (suppression strength)", fontsize=8)
        ax.set_title(f"{fwd[0]}/{fwd[1]}  ↔  {rev[0]}/{rev[1]}", fontsize=9, fontweight="500", pad=5)
        ax.legend(fontsize=7.5, frameon=True, framealpha=0.9, edgecolor="#ddd", handlelength=1.5)
        ax.grid(True, alpha=0.15, linewidth=0.4)
        ax.set_ylim(-0.35, 0.12)
        if i == 0: ax.set_ylabel("Δ COMET-Kiwi vs none (good)", fontsize=8)
    fig.suptitle("Forward vs reversed dose-response — good translations", fontsize=10, fontweight="bold")
    fig.text(0.5, -0.04,
             "Solid = forward (lr→hr axis), dashed = reversed (hr→lr axis). "
             "Gap between curves = causal asymmetry.",
             ha="center", fontsize=8, color="#5F5E5A", style="italic")
    plt.tight_layout(); savefig("fwd_rev_lines")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. Heatmap — all pairs × alpha, good translations
# ═══════════════════════════════════════════════════════════════════════════════
print("[6] Heatmap...")
sub = df[df["group"] == "good"]
all_pairs_sorted = PAIRS_FWD + [r for r in PAIRS_REV if r not in PAIRS_FWD]
pair_labels = [f"{s}/{a}" for s, a in all_pairs_sorted]

mat = np.zeros((len(all_pairs_sorted), len(alphas)))
for i, p in enumerate(all_pairs_sorted):
    for j, alpha in enumerate(alphas):
        d, _ = get(sub[sub["alpha"] == alpha], *p)
        mat[i, j] = d if d is not None else 0.0

fig, ax = plt.subplots(figsize=(8, 0.55 * len(all_pairs_sorted) + 1.5))
vmax = max(abs(mat).max(), 0.05)
im = ax.imshow(mat, aspect="auto", cmap="RdBu", vmin=-vmax, vmax=vmax)

ax.set_xticks(range(len(alphas)))
ax.set_xticklabels([f"α={a}" for a in alphas], fontsize=9)
ax.set_yticks(range(len(all_pairs_sorted)))
ax.set_yticklabels(pair_labels, fontsize=9)
ax.set_xlabel("Suppression strength α", fontsize=9)

# Annotate cells
for i in range(len(all_pairs_sorted)):
    for j in range(len(alphas)):
        val = mat[i, j]
        txt_color = "white" if abs(val) > vmax * 0.6 else "#2C2C2A"
        ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                fontsize=7, color=txt_color)

# Divider between forward and reversed
n_fwd_in_all = len(PAIRS_FWD)
ax.axhline(n_fwd_in_all - 0.5, color="#444", linewidth=1.2, linestyle="--")
ax.text(len(alphas) - 0.5, n_fwd_in_all - 0.6, "← forward  /  reversed →",
        ha="right", va="bottom", fontsize=7, color="#555", style="italic")

plt.colorbar(im, ax=ax, label="Δ COMET-Kiwi vs none", shrink=0.8, pad=0.02)
ax.set_title("Ablation effect heatmap — good translations\nRed = degrades, blue = improves",
             fontsize=9, fontweight="bold", pad=8)
plt.tight_layout(); savefig("heatmap")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. Scatter — geometric asymmetry vs causal effect
# ═══════════════════════════════════════════════════════════════════════════════
print("[7] Scatter: geometric vs causal...")
if os.path.exists(args.containment_csv):
    cont = pd.read_csv(args.containment_csv)
    # Expected columns: pair, mean_asymmetry_L16+
    # Merge with causal effect at α=0.3, good
    sub_causal = df[(df["alpha"] == 0.3) & (df["group"] == "good")].copy()
    sub_causal["pair"] = sub_causal["source"] + "/" + sub_causal["axis"]

    merged = sub_causal.merge(cont[["pair", "mean_asymmetry_L16+"]], on="pair", how="inner")

    if len(merged) > 0:
        fig, ax = plt.subplots(figsize=(7, 5))
        for _, row in merged.iterrows():
            src, ax_lang = row["source"], row["axis"]
            c = color(src, ax_lang)
            r = role(src, ax_lang)
            marker = "o" if r in ("primary", "secondary", "weaker") else "s"
            ax.scatter(row["mean_asymmetry_L16+"], row["delta_vs_none"],
                       color=c, s=60, marker=marker, zorder=3)
            ax.annotate(f"{src}/{ax_lang}",
                        (row["mean_asymmetry_L16+"], row["delta_vs_none"]),
                        textcoords="offset points", xytext=(6, 3), fontsize=7, color=c)
        # Fit line
        x = merged["mean_asymmetry_L16+"].values
        y = merged["delta_vs_none"].values
        if len(x) >= 3:
            m, b = np.polyfit(x, y, 1)
            xr = np.linspace(x.min() - 0.02, x.max() + 0.02, 100)
            ax.plot(xr, m * xr + b, color="#888", linewidth=1, linestyle="--", zorder=1)
            # Pearson r
            r_val = np.corrcoef(x, y)[0, 1]
            ax.text(0.97, 0.05, f"r = {r_val:.2f}", transform=ax.transAxes,
                    ha="right", fontsize=8, color="#555")

        ax.axhline(0, color="#888", linewidth=0.7, linestyle="--")
        ax.axvline(0, color="#888", linewidth=0.7, linestyle=":")
        ax.set_xlabel("Geometric containment asymmetry Δρ (mean L16+)", fontsize=9)
        ax.set_ylabel("Causal effect Δ COMET-Kiwi (α=0.3, good)", fontsize=9)
        ax.set_title("Geometric asymmetry vs causal ablation effect", fontsize=10, fontweight="bold", pad=8)
        ax.grid(True, alpha=0.15, linewidth=0.4)
        fwd_patch = mpatches.Patch(color="#534AB7", label="Forward pairs (●)")
        rev_patch  = mpatches.Patch(color="#AFA9EC", label="Reversed pairs (●)")
        ax.legend(handles=[fwd_patch, rev_patch], fontsize=8, frameon=True,
                  framealpha=0.9, edgecolor="#ddd")
        fig.text(0.5, -0.04,
                 "Each point is one language pair at α=0.3. "
                 "Positive Δρ = source geometrically contained in axis subspace. "
                 "Negative Δ COMET-Kiwi = ablation degrades translation.",
                 ha="center", fontsize=7.5, color="#5F5E5A", style="italic")
        plt.tight_layout(); savefig("scatter_asym")
    else:
        print("  No overlapping pairs between containment and ablation data, skipping scatter.")
else:
    print(f"  Containment CSV not found at {args.containment_csv}, skipping scatter.")

print("\nDone.")