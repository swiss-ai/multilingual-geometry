"""
viz_language_identity.py — Plot language identity score across layers.
Two extreme groups overlaid. Detects pivot events as contiguous bands
of upward-crossing layers in the transition zone.

Usage:
    python viz_language_identity.py
"""

import numpy as np
import json
import glob
import matplotlib.pyplot as plt

# ============================================================
# Config — change these to switch language
# ============================================================

LANG        = "gl"
LANG_NAME   = "Galician"
HS_DIR      = "hidden_states_2/galician_extremes_50"
SAVE_PATH   = "references_new/new_refs_galician_pivot_events_50.png"

# ============================================================
# Load references
# ============================================================

with open("references_new/reference_pairs.json") as f:
    pairs = json.load(f)

vecs       = np.load("references_new/contrastive_vectors.npz")
en_ref     = np.array(pairs[LANG]["en_ref"])
target_ref = np.array(pairs[LANG]["target_ref"])
v_lang_all = vecs[f"v_{LANG}"]
n_layers   = 33
layers     = list(range(n_layers))

en_scores   = np.array([float(np.dot(en_ref[l],     v_lang_all[l])) for l in layers])
tgt_scores  = np.array([float(np.dot(target_ref[l], v_lang_all[l])) for l in layers])
gap         = tgt_scores - en_scores + 1e-8

# ============================================================
# Load and project hidden states
# ============================================================

def load_and_project(fpath):
    data               = np.load(fpath, allow_pickle=True)
    translation_hidden = data["translation_hidden"]
    group              = int(data["group"])
    comet              = float(data["comet_kiwi"])
    source             = str(data["source"])
    translation        = str(data["translation"])

    scores = []
    for l in layers:
        h      = translation_hidden[l]
        h_norm = h / (np.linalg.norm(h) + 1e-8)
        scores.append(float(np.dot(h_norm, v_lang_all[l])))

    identity = (np.array(scores) - en_scores) / gap

    return {
        "group":       group,
        "comet":       comet,
        "source":      source[:60],
        "translation": translation[:60],
        "identity":    identity,
        "file":        fpath.split("/")[-1],
    }

all_files  = sorted(glob.glob(f"{HS_DIR}/*.npz"))
results    = [load_and_project(f) for f in all_files]
bad_group  = [r for r in results if r["group"] == 0]
good_group = [r for r in results if r["group"] == 1]

print(f"Loaded {len(bad_group)} bad | {len(good_group)} good")

# ============================================================
# Detect pivot events — contiguous bands of upward-crossing
# layers in the transition zone (0.3–0.7)
# ============================================================

def detect_pivot_bands(group, lo=0.3, hi=0.7):
    """
    Find contiguous bands of layers where:
      - mean identity is in [lo, hi]
      - gradient is positive (moving toward target language)

    Returns list of (start_layer, end_layer) tuples.
    """
    if not group:
        return []

    mean = np.mean([r["identity"] for r in group], axis=0)
    grad = np.gradient(mean)

    # boolean mask: in band AND rising
    in_pivot = [
        0.3 <= mean[l] <= 0.7 and grad[l] > 0
        for l in layers
    ]

    # find contiguous runs of True
    bands = []
    start = None
    for l, flag in enumerate(in_pivot):
        if flag and start is None:
            start = l
        elif not flag and start is not None:
            bands.append((start, l - 1))
            start = None
    if start is not None:
        bands.append((start, n_layers - 1))

    return bands, mean


for name, group in [("bad", bad_group), ("good", good_group)]:
    comet_vals        = [r["comet"] for r in group]
    bands, mean       = detect_pivot_bands(group)
    print(f"\n{name}: {len(group)} sentences | "
          f"comet {min(comet_vals):.4f}–{max(comet_vals):.4f}")
    print(f"  Pivot bands: {bands}")
    for start, end in bands:
        print(f"    L{start}–L{end} | "
              f"identity {mean[start]:.3f}→{mean[end]:.3f}")

# ============================================================
# Plot
# ============================================================

fig, ax = plt.subplots(figsize=(14, 6))

# ── reference lines ───────────────────────────────────────────────────────────
ax.axhline(0, color="black", linestyle="--", linewidth=1.5,
           label="English reference (0)", zorder=3)
ax.axhline(1, color="green", linestyle="--", linewidth=1.5,
           label=f"{LANG_NAME} reference (1)", zorder=3)

# ── horizontal transition band ────────────────────────────────────────────────
ax.axhspan(0.3, 0.7, alpha=0.06, color="orange", zorder=1)
ax.axhline(0.3, color="orange", linestyle="-.", linewidth=1.0, zorder=3)
ax.axhline(0.7, color="orange", linestyle="-.", linewidth=1.0, zorder=3,
           label="Transition zone (0.3–0.7)")

# ── both groups ───────────────────────────────────────────────────────────────
band_colors = {"Bad": "tomato", "Good": "steelblue"}

for group, color, label_prefix in [
    (bad_group,  "tomato",    "Bad"),
    (good_group, "steelblue", "Good"),
]:
    comet_vals        = [r["comet"] for r in group]
    identities        = np.array([r["identity"] for r in group])
    mean              = identities.mean(axis=0)
    std               = identities.std(axis=0)
    bands, _          = detect_pivot_bands(group)

    # std band
    ax.fill_between(layers, mean - std, mean + std,
                    alpha=0.12, color=color, zorder=2)

    # mean trace
    ax.plot(layers, mean,
            color=color, linewidth=2.5, alpha=1.0, zorder=5,
            label=f"{label_prefix} (n={len(group)}, "
                  f"comet {min(comet_vals):.3f}–{max(comet_vals):.3f})")

    # pivot bands — shaded vertical regions
    for b_idx, (start, end) in enumerate(bands):
        ax.axvspan(start, end,
                   alpha=0.20, color=color, zorder=1)
        ax.axvline(start, color=color, linestyle=":",
                   linewidth=1.5, zorder=4)
        ax.axvline(end,   color=color, linestyle=":",
                   linewidth=1.5, zorder=4)
        # annotate only first band to avoid clutter
        if b_idx == 0:
            ax.annotate(
                f"{label_prefix}\nL{start}–L{end}",
                xy=((start + end) / 2,
                    1.45 if label_prefix == "Good" else 1.55),
                ha="center", fontsize=8,
                color=color, fontweight="bold"
            )

    # dots at pivot layers
    grad           = np.gradient(mean)
    pivot_layers   = [l for l in layers
                      if 0.3 <= mean[l] <= 0.7 and grad[l] > 0]
    pivot_values   = [mean[l] for l in pivot_layers]
    ax.scatter(pivot_layers, pivot_values,
               color=color, s=80, zorder=6,
               edgecolors="white", linewidths=0.8)

ax.set_xlabel("Layer", fontsize=12)
ax.set_ylabel(f"Language identity\n(0 = English,  1 = {LANG_NAME})",
              fontsize=11)
ax.set_ylim(-0.6, 1.7)
ax.set_xlim(0, 32)
ax.grid(True, alpha=0.25)
ax.legend(fontsize=9, loc="upper left")

plt.suptitle(
    f"Language identity across layers — translation hidden states\n"
    f"projected onto English→{LANG_NAME} contrastive axis\n"
    f"Shaded bands = contiguous upward pivot events in transition zone (0.3–0.7)",
    fontsize=12, fontweight="bold"
)
plt.tight_layout()
plt.savefig(SAVE_PATH, dpi=150)
print(f"\nSaved -> {SAVE_PATH}")