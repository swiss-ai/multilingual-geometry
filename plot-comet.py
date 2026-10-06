import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
import glob

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = {
    "Afrikaans": "#B04A3A",
    "Catalan":   "#3A5FA0",
    "Dutch":     "#2E8B57",
    "Galician":  "#8B5CA0",
    "German":    "#C47A1E",
    "Spanish":   "#2A7FA0",
}

files = sorted(glob.glob("translations/apertus/translations_*.csv"))

fig, ax = plt.subplots(figsize=(8, 4.5))

for f in files:
    lang = f.split("translations_")[1].split("_apertus")[0].capitalize()
    df   = pd.read_csv(f)
    scores = df["comet_kiwi"].dropna().values

    kde  = gaussian_kde(scores, bw_method=0.15)
    x    = np.linspace(scores.min(), scores.max(), 500)
    y    = kde(x)

    color = COLORS.get(lang, "gray")
    ax.plot(x, y, linewidth=2.0, color=color, label=f"{lang} (μ={scores.mean():.3f})")
    # ax.fill_between(x, y, alpha=0.08, color=color)
    ax.axvline(scores.mean(), linewidth=0.8, linestyle="--", color=color, alpha=0.6)

ax.set_xlabel("COMET-Kiwi score")
ax.set_ylabel("Density")
ax.set_title("Score distribution per language")
ax.axvline(0.8, linewidth=1, linestyle=":", color="black", alpha=0.4, label="0.8 threshold")
ax.legend(frameon=False, loc="upper left")
ax.set_xlim(0.60, 0.95)

fig.tight_layout()
fig.savefig("figures/fig_comet_kde.png", bbox_inches="tight")
plt.close()
print("Saved -> figures/fig_comet_kde2.png")