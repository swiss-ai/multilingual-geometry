import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.stats import gaussian_kde
import glob
import os

os.makedirs("figures", exist_ok=True)

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "figure.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": True,
})

BOTTOM_COLOR = "#C0504D"
TOP_COLOR    = "#4472C4"
BG_COLOR     = "#FFFFFF"

files = sorted(glob.glob("translations/apertus/translations_*.csv"))
languages = []
for f in files:
    lang = f.split("translations_")[1].split("_apertus")[0].capitalize()
    df   = pd.read_csv(f)
    scores = df["comet_kiwi"].dropna().values
    q25  = np.percentile(scores, 25)
    q75  = np.percentile(scores, 75)
    languages.append({
        "name":   lang,
        "scores": scores,
        "bottom": scores[scores <= q25],
        "top":    scores[scores >= q75],
        "q25":    q25,
        "q75":    q75,
        "mean":   scores.mean(),
    })

languages = sorted(languages, key=lambda x: x["mean"], reverse=True)

fig = plt.figure(figsize=(12, 7))
fig.patch.set_facecolor(BG_COLOR)

fig.text(0.5, 0.97, "COMET-Kiwi score distribution by language",
         ha="center", va="top", fontsize=13, fontweight="bold",
         fontfamily="serif", color="#1a1a1a")
fig.text(0.5, 0.935, "Bottom quartile (Q1) vs top quartile (Q4) — sorted by mean score",
         ha="center", va="top", fontsize=9, color="#555555", fontfamily="serif")

for i, lang in enumerate(languages):
    ax = fig.add_subplot(2, 3, i + 1)
    ax.set_facecolor(BG_COLOR)

    bottom = lang["bottom"]
    top    = lang["top"]

    x_min = min(bottom.min(), top.min())
    x_max = max(bottom.max(), top.max())
    x_pad = (x_max - x_min) * 0.05

    for subset, color, alpha_fill in [
        (bottom, BOTTOM_COLOR, 0.2),
        (top,    TOP_COLOR,    0.2),
    ]:
        kde = gaussian_kde(subset, bw_method=0.25)
        x   = np.linspace(x_min - x_pad, x_max + x_pad, 500)
        y   = kde(x)
        ax.plot(x, y, linewidth=1.8, color=color)
        ax.fill_between(x, y, alpha=alpha_fill, color=color)

    ax.axvline(lang["mean"], color="#333333", linewidth=0.8,
               linestyle="--", alpha=0.5, zorder=5)
    ax.axvline(lang["q25"], color=BOTTOM_COLOR, linewidth=0.6,
               linestyle=":", alpha=0.7)
    ax.axvline(lang["q75"], color=TOP_COLOR, linewidth=0.6,
               linestyle=":", alpha=0.7)

    ax.set_title(lang["name"], fontsize=10, fontweight="bold",
                 color="#1a1a1a", pad=6, fontfamily="serif")

    ax.text(0.97, 0.92, f"μ = {lang['mean']:.3f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=7.5, color="#444444", fontfamily="serif",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="#cccccc", linewidth=0.5))

    ax.text(0.03, 0.97, f"Q1 \u2264 {lang['q25']:.3f}",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=7, color=BOTTOM_COLOR, fontfamily="serif")
    ax.text(0.03, 0.84, f"Q3 \u2265 {lang['q75']:.3f}",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=7, color=TOP_COLOR, fontfamily="serif")

    ax.set_xlabel("COMET-Kiwi score", fontsize=7.5)
    ax.set_ylabel("Density", fontsize=7.5)
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.spines["left"].set_color("#cccccc")
    ax.spines["left"].set_linewidth(0.8)
    ax.tick_params(axis="x", labelsize=7)
    ax.spines["bottom"].set_color("#cccccc")
    ax.spines["bottom"].set_linewidth(0.8)

legend_elements = [
    mpatches.Patch(facecolor=BOTTOM_COLOR, alpha=0.6, label="Bottom quartile (Q1)"),
    mpatches.Patch(facecolor=TOP_COLOR,    alpha=0.6, label="Top quartile (Q4)"),
    plt.Line2D([0], [0], color="#333333", linewidth=0.8, linestyle="--", label="Mean"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=3,
           frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 0.01),
           handlelength=1.5)

fig.subplots_adjust(top=0.88, bottom=0.12, hspace=0.55, wspace=0.25,
                    left=0.04, right=0.98)

fig.savefig("figures/fig_comet_quartiles.png", bbox_inches="tight",
            facecolor=BG_COLOR)
plt.close()
print("Saved -> figures/fig_comet_quartiles.png")