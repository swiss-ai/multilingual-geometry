"""
run_all_asymmetry.py — Stepping stone asymmetry experiments for Romance and Germanic families.

Produces:
  - One PNG per experiment (individual plots)
  - grid_romance.png       — 4×2 grid, Romance family, A4 landscape, shared y-axis
  - grid_germanic.png      — 4×2 grid, Germanic family, A4 landscape, shared y-axis
  - asymmetry_romance.png  — Δ asymmetry index per layer for all Romance pairs
  - asymmetry_germanic.png — Δ asymmetry index per layer for all Germanic pairs

Usage:
    python run_all_asymmetry.py \
        --hs_dir    hidden_states \
        --ref_json  references/reference_pairs.json \
        --vecs_npz  references/contrastive_vectors.npz \
        --output_dir figures/asymmetry

    # Single experiment:
    python run_all_asymmetry.py --experiment gl_via_es
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import json

# ============================================================
# Args
# ============================================================

parser = argparse.ArgumentParser()
parser.add_argument("--hs_dir",     type=str, default="hidden_states_2")
parser.add_argument("--ref_json",   type=str, default="references/reference_pairs.json")
parser.add_argument("--vecs_npz",   type=str, default="references/contrastive_vectors.npz")
parser.add_argument("--output_dir", type=str, default="figures/asymmetry")
parser.add_argument("--experiment", type=str, default=None)
parser.add_argument("--dpi",        type=int, default=150)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# ============================================================
# Language metadata
# ============================================================

LANG_META = {
    "gl": {"name": "Galician",  "color": "#E91E63", "hs_dir": "galician_extremes_100"},
    "es": {"name": "Spanish",   "color": "#9C27B0", "hs_dir": "spanish_extremes_100"},
    "ca": {"name": "Catalan",   "color": "#F44336", "hs_dir": "catalan_extremes_100"},
    "de": {"name": "German",    "color": "#2196F3", "hs_dir": "german_extremes_100"},
    "af": {"name": "Afrikaans", "color": "#FF9800", "hs_dir": "afrikaans_extremes_100"},
    "nl": {"name": "Dutch",     "color": "#4CAF50", "hs_dir": "dutch_extremes_100"},
}

# ============================================================
# Experiment configs
# ============================================================

ROMANCE_ORDER = [
    "gl_via_es", "es_via_gl",
    "gl_via_ca", "ca_via_gl",
    "es_via_ca", "ca_via_es",
    "gl_via_de", "de_via_gl",
]

GERMANIC_ORDER = [
    "af_via_nl", "nl_via_af",
    "af_via_de", "de_via_af",
    "nl_via_de", "de_via_nl",
    "af_via_es", "es_via_af",
]

# Asymmetry pairs: (lr_key, hr_key, label, color)
ROMANCE_ASYM_PAIRS = [
    ("gl_via_es", "es_via_gl", "gl / es", "#9C27B0"),
    ("gl_via_ca", "ca_via_gl", "gl / ca", "#E91E63"),
    ("es_via_ca", "ca_via_es", "es / ca", "#F44336"),
]
ROMANCE_CONTROL = ("gl_via_de", "de_via_gl", "gl / de")
ROMANCE_HIERARCHY = "es  >  gl  >  ca"

GERMANIC_ASYM_PAIRS = [
    ("af_via_nl", "nl_via_af", "af / nl", "#4CAF50"),
    ("af_via_de", "de_via_af", "af / de", "#FF9800"),
    ("nl_via_de", "de_via_nl", "nl / de", "#2196F3"),
]
GERMANIC_CONTROL = ("af_via_es", "es_via_af", "af / es")
GERMANIC_HIERARCHY = "nl  ≈  de  >  af"

CONFIGS = {
    # ── Romance ──────────────────────────────────────────────
    "gl_via_es": {"source": "gl", "axis": "es", "control": False, "short": "gl→es axis"},
    "es_via_gl": {"source": "es", "axis": "gl", "control": False, "short": "es→gl axis"},
    "gl_via_ca": {"source": "gl", "axis": "ca", "control": False, "short": "gl→ca axis"},
    "ca_via_gl": {"source": "ca", "axis": "gl", "control": False, "short": "ca→gl axis"},
    "es_via_ca": {"source": "es", "axis": "ca", "control": False, "short": "es→ca axis"},
    "ca_via_es": {"source": "ca", "axis": "es", "control": False, "short": "ca→es axis"},
    "gl_via_de": {"source": "gl", "axis": "de", "control": True,  "short": "gl→de axis (ctrl)"},
    "de_via_gl": {"source": "de", "axis": "gl", "control": True,  "short": "de→gl axis (ctrl)"},
    # ── Germanic ─────────────────────────────────────────────
    "af_via_nl": {"source": "af", "axis": "nl", "control": False, "short": "af→nl axis"},
    "nl_via_af": {"source": "nl", "axis": "af", "control": False, "short": "nl→af axis"},
    "af_via_de": {"source": "af", "axis": "de", "control": False, "short": "af→de axis"},
    "de_via_af": {"source": "de", "axis": "af", "control": False, "short": "de→af axis"},
    "nl_via_de": {"source": "nl", "axis": "de", "control": False, "short": "nl→de axis"},
    "de_via_nl": {"source": "de", "axis": "nl", "control": False, "short": "de→nl axis"},
    "af_via_es": {"source": "af", "axis": "es", "control": True,  "short": "af→es axis (ctrl)"},
    "es_via_af": {"source": "es", "axis": "af", "control": True,  "short": "es→af axis (ctrl)"},
}

# ============================================================
# Load references and vectors
# ============================================================

print(f"Loading references from {args.ref_json}...")
with open(args.ref_json) as f:
    ref_pairs = json.load(f)

print(f"Loading vectors from {args.vecs_npz}...")
vecs = np.load(args.vecs_npz)

# ============================================================
# Core helpers
# ============================================================

def load_axis(axis_code, n_layers):
    v          = vecs[f"v_{axis_code}"]
    en_ref     = np.array(ref_pairs[axis_code]["en_ref"])
    target_ref = np.array(ref_pairs[axis_code]["target_ref"])
    en_scores  = np.array([float(np.dot(en_ref[l],     v[l])) for l in range(n_layers)])
    tgt_scores = np.array([float(np.dot(target_ref[l], v[l])) for l in range(n_layers)])
    gap        = tgt_scores - en_scores + 1e-8
    return v, en_scores, gap


def project_file(fpath, v, en_scores, gap):
    data     = np.load(fpath, allow_pickle=True)
    hidden   = data["translation_hidden"]
    n_layers = hidden.shape[0]
    scores   = []
    for l in range(n_layers):
        h      = hidden[l]
        h_norm = h / (np.linalg.norm(h) + 1e-8)
        raw    = float(np.dot(h_norm, v[l]))
        scores.append((raw - en_scores[l]) / gap[l])
    return {
        "group": int(data["group"]),
        "comet": float(data["comet_kiwi"]),
        "proj":  np.array(scores),
    }


def compute_experiment(key, cfg):
    source_meta = LANG_META[cfg["source"]]
    hs_path     = os.path.join(args.hs_dir, source_meta["hs_dir"])
    files       = sorted(glob.glob(os.path.join(hs_path, "*.npz")))
    if not files:
        print(f"  WARNING: no files in {hs_path}")
        return None, None, None, None

    n_layers      = np.load(files[0], allow_pickle=True)["translation_hidden"].shape[0]
    v, en_sc, gap = load_axis(cfg["axis"], n_layers)
    results       = [project_file(f, v, en_sc, gap) for f in files]
    good          = [r for r in results if r["group"] == 1]
    bad           = [r for r in results if r["group"] == 0]
    layers        = list(range(n_layers))
    return good, bad, layers, n_layers


def robust_ylim(all_means_list, padding=0.2):
    """Compute a shared y-axis limit based on 5th–95th percentile across all means."""
    combined = np.concatenate(all_means_list)
    lo = np.nanpercentile(combined, 2)
    hi = np.nanpercentile(combined, 98)
    margin = (hi - lo) * padding
    return (lo - margin, hi + margin)


def draw_panel(ax, group_data, layers, source_meta, axis_meta,
               panel_title, show_ylabel=False, show_legend=False,
               fontsize=7, ylim=None):
    ax.set_facecolor("#fafafa")

    if not group_data:
        ax.set_title(f"{panel_title} (no data)", fontsize=fontsize)
        return

    means = np.mean([r["proj"] for r in group_data], axis=0)
    stds  = np.std( [r["proj"] for r in group_data], axis=0)

    ax.plot(layers, means, color=source_meta["color"],
            linewidth=1.4, label=f"{source_meta['name']}")
    ax.fill_between(layers, means - stds, means + stds,
                    color=source_meta["color"], alpha=0.10)

    ax.axhline(0, color="black",            linestyle="--", linewidth=0.7)
    ax.axhline(1, color=axis_meta["color"], linestyle="--", linewidth=0.7,
               label=f"{axis_meta['name']} ref (1)")
    ax.axhspan(0.3, 0.7, alpha=0.06, color="orange")
    ax.axhline(0.3, color="orange", linestyle="-.", linewidth=0.5)
    ax.axhline(0.7, color="orange", linestyle="-.", linewidth=0.5)

    ax.set_title(panel_title, fontsize=fontsize, pad=2)
    ax.set_xlim(0, max(layers))
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.tick_params(labelsize=fontsize - 1)
    ax.grid(True, alpha=0.2, linewidth=0.4)

    if show_ylabel:
        ax.set_ylabel(f"Identity\n(0=En, 1={axis_meta['name']})",
                      fontsize=fontsize - 1)
    if show_legend:
        ax.legend(fontsize=fontsize - 2, frameon=False,
                  loc="lower right", handlelength=1)


# ============================================================
# Individual plots
# ============================================================

def save_individual(key, cfg):
    good, bad, layers, _ = compute_experiment(key, cfg)
    if good is None:
        return

    source_meta = LANG_META[cfg["source"]]
    axis_meta   = LANG_META[cfg["axis"]]
    ctrl_str    = " (control)" if cfg["control"] else ""

    fig, axes_plot = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
    draw_panel(axes_plot[0], good, layers, source_meta, axis_meta,
               "Good translations", show_ylabel=True, show_legend=True, fontsize=10)
    draw_panel(axes_plot[1], bad,  layers, source_meta, axis_meta,
               "Bad translations",  show_legend=True, fontsize=10)

    title = (f"English→{source_meta['name']} hidden states\n"
             f"projected onto {axis_meta['name']} contrastive axis{ctrl_str}")
    plt.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()

    out = os.path.join(args.output_dir, f"{key}.png")
    plt.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ============================================================
# A4 grid with shared y-axis
# ============================================================

def save_grid(experiment_keys, family_name, filename):
    n_rows, n_exp_cols = 4, 2

    # ── Pre-compute all means to derive shared ylim ──────────
    all_means = []
    exp_data  = {}
    for key in experiment_keys:
        cfg  = CONFIGS[key]
        good, bad, layers, _ = compute_experiment(key, cfg)
        exp_data[key] = (good, bad, layers, cfg)
        for grp in [good, bad]:
            if grp:
                m = np.mean([r["proj"] for r in grp], axis=0)
                all_means.append(m)

    ylim = robust_ylim(all_means, padding=0.15)
    print(f"  Shared ylim for {family_name}: {ylim[0]:.2f} – {ylim[1]:.2f}")

    fig, axes = plt.subplots(
        n_rows, n_exp_cols * 2,
        figsize=(11.7, 8.3),
        gridspec_kw={"hspace": 0.55, "wspace": 0.25},
    )

    print(f"  Building {family_name} grid...")

    for idx, key in enumerate(experiment_keys):
        good, bad, layers, cfg = exp_data[key]
        row     = idx // n_exp_cols
        ecol    = idx %  n_exp_cols
        col_good = ecol * 2
        col_bad  = ecol * 2 + 1

        source_meta = LANG_META[cfg["source"]]
        axis_meta   = LANG_META[cfg["axis"]]
        show_y      = (ecol == 0)

        ax_good = axes[row, col_good]
        ax_bad  = axes[row, col_bad]

        draw_panel(ax_good, good, layers, source_meta, axis_meta,
                   "Good", show_ylabel=show_y, show_legend=False,
                   fontsize=6, ylim=ylim)
        draw_panel(ax_bad,  bad,  layers, source_meta, axis_meta,
                   "Bad",  show_ylabel=False, show_legend=False,
                   fontsize=6, ylim=ylim)

        ctrl_str   = " ✦" if cfg["control"] else ""
        cell_title = f"{cfg['short']}{ctrl_str}"
        mid_x = (ax_good.get_position().x0 + ax_bad.get_position().x1) / 2
        fig.text(mid_x, ax_good.get_position().y1 + 0.008,
                 cell_title, ha="center", va="bottom",
                 fontsize=6.5, fontweight="bold")

    for ecol, label in enumerate(["Pair A", "Pair B"]):
        ax0 = axes[0, ecol * 2]
        ax1 = axes[0, ecol * 2 + 1]
        mid_x = (ax0.get_position().x0 + ax1.get_position().x1) / 2
        fig.text(mid_x, ax0.get_position().y1 + 0.035,
                 label, ha="center", va="bottom",
                 fontsize=8, fontweight="bold", color="#444")

    fig.text(0.99, 0.01, "✦ = cross-family control pair",
             ha="right", va="bottom", fontsize=6, color="#888", style="italic")
    fig.suptitle(
        f"Stepping-stone asymmetry — {family_name} family\n"
        f"Language identity (0 = English, 1 = axis target language)",
        fontsize=9, fontweight="bold", y=0.995,
    )

    out = os.path.join(args.output_dir, filename)
    plt.savefig(out, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Grid saved → {out}")


# ============================================================
# Asymmetry summary: Δ(lr→hr) - Δ(hr→lr) per layer
# ============================================================

def save_asymmetry_summary(asym_pairs, control_pair, family_name,
                            resource_hierarchy, filename):
    def smooth(x, w=3):
        return np.convolve(x, np.ones(w)/w, mode="same")

    # ── Pre-compute deltas for main pairs to derive ylim ─────
    all_good_deltas = []
    all_bad_deltas  = []
    pair_data = {}

    for lr_key, hr_key, label, color in asym_pairs:
        lr_good, lr_bad, layers, _ = compute_experiment(lr_key, CONFIGS[lr_key])
        hr_good, hr_bad, _,      _ = compute_experiment(hr_key, CONFIGS[hr_key])
        pair_data[label] = (lr_good, lr_bad, hr_good, hr_bad, layers, color)
        if lr_good and hr_good:
            d = smooth(np.mean([r["proj"] for r in lr_good], axis=0) -
                       np.mean([r["proj"] for r in hr_good], axis=0))
            all_good_deltas.append(d)
        if lr_bad and hr_bad:
            d = smooth(np.mean([r["proj"] for r in lr_bad], axis=0) -
                       np.mean([r["proj"] for r in hr_bad], axis=0))
            all_bad_deltas.append(d)

    # Shared ylim from main pairs only (excludes control outlier)
    combined = np.concatenate(all_good_deltas + all_bad_deltas)
    lo = np.nanpercentile(combined, 2)  - 0.15
    hi = np.nanpercentile(combined, 98) + 0.15

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True,
                              gridspec_kw={"wspace": 0.08})

    legend_handles = []
    legend_labels  = []
    good_deltas_for_annot = {}

    for ax_idx, (group_label, panel_data_key) in enumerate([
        ("Good translations", "good"),
        ("Bad translations",  "bad"),
    ]):
        ax = axes[ax_idx]
        ax.set_facecolor("#ffffff")
        ax.set_ylim(lo, hi)

        ax.axhline(0, color="#333", linestyle="--", linewidth=0.8, zorder=2)
        ax.axvspan(16, 32, alpha=0.04, color="#2196F3", zorder=0)
        ax.axvline(16, color="#999", linestyle=":", linewidth=0.7, zorder=2)

        layers = None

        # Main pairs
        for lr_key, hr_key, label, color in asym_pairs:
            lr_good, lr_bad, hr_good, hr_bad, lyr, _ = pair_data[label]
            layers = lyr
            lr_grp = lr_good if panel_data_key == "good" else lr_bad
            hr_grp = hr_good if panel_data_key == "good" else hr_bad
            if not lr_grp or not hr_grp:
                continue

            delta = smooth(np.mean([r["proj"] for r in lr_grp], axis=0) -
                           np.mean([r["proj"] for r in hr_grp], axis=0))

            if panel_data_key == "good":
                good_deltas_for_annot[label] = (delta, color)

            h, = ax.plot(lyr, delta, color=color, linewidth=1.6,
                         label=label, zorder=3)
            ax.fill_between(lyr, 0, delta, color=color, alpha=0.07, zorder=1)

            if ax_idx == 0:
                legend_handles.append(h)
                legend_labels.append(label)

        # Control pair
        if control_pair:
            lr_key, hr_key, ctrl_label = control_pair
            lr_good, lr_bad, lyr2, _ = compute_experiment(lr_key, CONFIGS[lr_key])
            hr_good, hr_bad, _,    _ = compute_experiment(hr_key, CONFIGS[hr_key])
            lr_grp = lr_good if panel_data_key == "good" else lr_bad
            hr_grp = hr_good if panel_data_key == "good" else hr_bad
            if lr_grp and hr_grp:
                delta = smooth(np.mean([r["proj"] for r in lr_grp], axis=0) -
                               np.mean([r["proj"] for r in hr_grp], axis=0))
                h, = ax.plot(lyr2, delta, color="#aaaaaa", linewidth=1.0,
                             linestyle="--", label=f"{ctrl_label} (ctrl)", zorder=2)
                if ax_idx == 0:
                    legend_handles.append(h)
                    legend_labels.append(f"{ctrl_label} (ctrl)")

        if layers is None:
            continue

        ax.set_xlim(0, max(layers))
        ax.set_xlabel("Transformer layer", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.15, linewidth=0.4)
        ax.set_title(group_label, fontsize=8, pad=4)

    # ── Single legend at top centre ───────────────────────────
    fig.legend(legend_handles, legend_labels,
               fontsize=7, frameon=True, framealpha=0.9,
               edgecolor="#ddd", ncol=len(legend_handles),
               loc="upper center", bbox_to_anchor=(0.5, 1.01),
               handlelength=1.5, columnspacing=1.2)

    # ── Y-axis label ─────────────────────────────────────────
    axes[0].set_ylabel("Asymmetry index  Δ(ℓ) = (lr→hr) − (hr→lr)", fontsize=8)

    # ── Annotations on good panel ─────────────────────────────
    ax0 = axes[0]
    for label, (delta, color) in good_deltas_for_annot.items():
        mean_late = float(np.mean(delta[16:29]))
        direction = "contained ↑" if mean_late > 0.05 else \
                    "inverse ↓"   if mean_late < -0.05 else "symmetric"
        y_val  = float(np.clip(delta[22], lo + 0.05, hi - 0.05))
        offset = 0.12 if mean_late > 0 else -0.12
        y_text = float(np.clip(y_val + offset, lo + 0.02, hi - 0.02))
        ax0.annotate(
            direction,
            xy=(22, y_val),
            xytext=(22, y_text),
            fontsize=6, color=color, ha="center",
            arrowprops=dict(arrowstyle="-", color=color,
                            lw=0.7, connectionstyle="arc3,rad=0.0"),
        )

    # Resource hierarchy box
    ax0.text(0.98, 0.97, f"Inferred hierarchy:\n{resource_hierarchy}",
             transform=ax0.transAxes, fontsize=6.5,
             va="top", ha="right",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       edgecolor="#ccc", alpha=0.9))

    # ── Caption below figure ──────────────────────────────────
    caption = (
        "Δ(ℓ) > 0: low-resource language is more similar to the high-resource axis "
        "than vice versa (containment, consistent with the stepping-stone hypothesis).  "
        "Δ(ℓ) < 0: inverse relationship — high-resource language is contained in "
        "the low-resource axis.  Dashed grey line = cross-family control pair."
    )
    fig.text(0.5, -0.04, caption, ha="center", va="top",
             fontsize=6.5, color="#444", wrap=True,
             style="italic")

    fig.suptitle(
        f"Representational asymmetry — {family_name} family",
        fontsize=9, fontweight="bold", y=1.07,
    )
    plt.tight_layout()

    out = os.path.join(args.output_dir, filename)
    plt.savefig(out, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Asymmetry summary saved → {out}")


# ============================================================
# Run
# ============================================================

if args.experiment:
    cfg = CONFIGS[args.experiment]
    print(f"\nRunning single experiment: {args.experiment}")
    save_individual(args.experiment, cfg)
else:
    print("\n── Individual plots ──────────────────────────────────")
    for key, cfg in CONFIGS.items():
        print(f"\nExperiment: {key}")
        save_individual(key, cfg)

    print("\n── Grid plots ────────────────────────────────────────")
    save_grid(ROMANCE_ORDER,  "Romance",  "grid_romance.png")
    save_grid(GERMANIC_ORDER, "Germanic", "grid_germanic.png")

    print("\n── Asymmetry summary plots ───────────────────────────")
    save_asymmetry_summary(ROMANCE_ASYM_PAIRS,  ROMANCE_CONTROL,
                           "Romance",  ROMANCE_HIERARCHY,  "asymmetry_romance.png")
    save_asymmetry_summary(GERMANIC_ASYM_PAIRS, GERMANIC_CONTROL,
                           "Germanic", GERMANIC_HIERARCHY, "asymmetry_germanic.png")

print(f"\nAll done. Figures in {args.output_dir}/")
