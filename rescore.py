#!/usr/bin/env python3

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from comet import download_model, load_from_checkpoint

OUTPUT_DIR = "speedy-llama/flat"
PLOT_DIR = "speedy-llama/plots"
SUMMARY_CSV = os.path.join(OUTPUT_DIR, "ablation_summary.csv")
os.makedirs(PLOT_DIR, exist_ok=True)

PAIRS = [
    ("es", "ca", "Spanish → Catalan"),
    ("ca", "es", "Catalan → Spanish"),
    ("es", "gl", "Spanish → Galician"),
    ("gl", "es", "Galician → Spanish"),
    ("ca", "gl", "Catalan → Galician"),
    ("gl", "ca", "Galician → Catalan"),
    ("de", "af", "German → Afrikaans"),
    ("af", "de", "Afrikaans → German"),
    ("de", "nl", "German → Dutch"),
    ("nl", "de", "Dutch → German"),
    ("nl", "af", "Dutch → Afrikaans"),
    ("af", "nl", "Afrikaans → Dutch"),
]
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

print("Loading COMET-Kiwi model...")
comet_path = download_model("Unbabel/wmt22-cometkiwi-da")
comet_model = load_from_checkpoint(comet_path)
print("COMET model loaded.\n")

def add_sentence_key(df):
    df = df.copy()
    if "row_id" in df.columns:
        df["_sentence_key"] = df["row_id"].astype(str)
    else:
        df["_sentence_key"] = df["source"].astype(str)
    return df

def score_file(fpath):
    scored_path = fpath.replace(".csv", "_scored.csv")
    if os.path.exists(scored_path):
        print(f"  Loading existing score: {scored_path}")
        df = pd.read_csv(scored_path)
        if "ablation_comet" in df.columns:
            df["comet_score"] = df["ablation_comet"]
        elif "comet_score" not in df.columns:
            raise ValueError(f"Could not find COMET score column in {scored_path}")
        return df

    df = pd.read_csv(fpath)
    required = ["source", "generated", "reference"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"{fpath} is missing required column: {col}")

    data = [
        {"src": str(row["source"]), "mt": str(row["generated"]), "ref": str(row["reference"])}
        for _, row in df.iterrows()
    ]
    print(f"  Scoring {len(data)} translations...")
    scores = comet_model.predict(data, batch_size=8, gpus=1)
    df["comet_score"] = scores.scores
    df["ablation_comet"] = df["comet_score"]
    df.to_csv(scored_path, index=False)
    print(f"  Saved: {scored_path}")
    return df

summary = []

for source, axis, pair_title in PAIRS:
    print("=" * 70)
    print(f"PAIR: {pair_title}")
    print("=" * 70)

    none_files = sorted(glob.glob(os.path.join(
        OUTPUT_DIR, f"{source}_ablate_{axis}_none_*.csv"
    )))
    if not none_files:
        print(f"WARNING: No none baseline found for {source} -> {axis}")
        continue

    none_file = max(none_files, key=os.path.getmtime)
    print(f"None baseline: {os.path.basename(none_file)}")

    none_df = add_sentence_key(score_file(none_file))
    if "group" not in none_df.columns:
        raise ValueError(f"{none_file} does not contain a 'group' column.")

    none_lookup = none_df[["_sentence_key", "comet_score"]].copy()
    if none_lookup["_sentence_key"].duplicated().any():
        raise ValueError(f"Duplicate sentence keys in none file: {none_file}")

    ablation_files = sorted(glob.glob(os.path.join(
        OUTPUT_DIR, f"{source}_ablate_{axis}_probe_gap_*.csv"
    )))
    if not ablation_files:
        print(f"No probe_gap files found for {source} -> {axis}")
        continue

    for fpath in ablation_files:
        filename = os.path.basename(fpath)
        print(f"\nProcessing: {filename}")

        df = add_sentence_key(score_file(fpath))
        for col in ["group", "alpha", "condition"]:
            if col not in df.columns:
                raise ValueError(f"{filename} is missing required column: {col}")

        # ONLY GOOD TRANSLATIONS
        good = df[df["group"] == 1].copy()
        print(f"Good translations: {len(good)}/{len(df)}")
        if good.empty:
            continue

        # Drop duplicate sentence keys for this test run
        before = len(good)

        good = good.drop_duplicates(
            subset="_sentence_key",
            keep="first"
        ).copy()

        after = len(good)

        if before != after:
            print(
                f"  Dropped {before - after} duplicate sentence(s); "
                f"using {after} unique sentences."
            )

        merged = good.merge(
            none_lookup,
            on="_sentence_key",
            how="inner",
            validate="one_to_one",
            suffixes=("", "_none"),
        )

        if len(merged) != len(good):
            print(f"WARNING: only matched {len(merged)}/{len(good)} good sentences with none.")

        if merged.empty:
            print("ERROR: no good sentences matched. Skipping.")
            continue

        alpha = float(merged["alpha"].iloc[0])
        condition = str(merged["condition"].iloc[0])

        mean_none = float(merged["comet_score_none"].mean())
        mean_ablated = float(merged["comet_score"].mean())
        delta = (merged["comet_score"] - merged["comet_score_none"]).to_numpy()
        mean_delta = float(np.mean(delta))
        std_delta = float(np.std(delta, ddof=1)) if len(delta) > 1 else 0.0
        se_delta = std_delta / np.sqrt(len(delta))

        summary.append({
            "source": source,
            "axis": axis,
            "pair": f"{source} → {axis}",
            "pair_title": pair_title,
            "condition": condition,
            "alpha": alpha,
            "group": "good",
            "n": len(merged),
            "none_comet": mean_none,
            "ablation_comet": mean_ablated,
            "delta_none_relative": mean_delta,
            "se_delta": se_delta,
            "source_file": filename,
        })

# Add alpha=0 using the same good-sentence pool represented by the ablation files.
summary_df = pd.DataFrame(summary)
if summary_df.empty:
    raise RuntimeError("No results were produced. Check that none and probe_gap files exist.")

baseline_rows = []

for source, axis, pair_title in PAIRS:
    pair_rows = summary_df[
        (summary_df["source"] == source) & (summary_df["axis"] == axis)
    ]
    if pair_rows.empty:
        continue

    none_files = sorted(glob.glob(os.path.join(
        OUTPUT_DIR, f"{source}_ablate_{axis}_none_*.csv"
    )))
    ablation_files = sorted(glob.glob(os.path.join(
        OUTPUT_DIR, f"{source}_ablate_{axis}_probe_gap_*.csv"
    )))
    if not none_files or not ablation_files:
        continue

    none_file = max(none_files, key=os.path.getmtime)
    none_df = add_sentence_key(score_file(none_file))

    good_keys = set()
    for fpath in ablation_files:
        abl = add_sentence_key(pd.read_csv(fpath))
        if "group" in abl.columns:
            good_keys.update(abl.loc[abl["group"] == 1, "_sentence_key"].astype(str))

    baseline_good = none_df[
        none_df["_sentence_key"].astype(str).isin(good_keys)
    ].copy()

    if baseline_good.empty:
        continue

    baseline_rows.append({
        "source": source,
        "axis": axis,
        "pair": f"{source} → {axis}",
        "pair_title": pair_title,
        "condition": "none",
        "alpha": 0.0,
        "group": "good",
        "n": len(baseline_good),
        "none_comet": float(baseline_good["comet_score"].mean()),
        "ablation_comet": float(baseline_good["comet_score"].mean()),
        "delta_none_relative": 0.0,
        "se_delta": 0.0,
        "source_file": os.path.basename(none_file),
    })

if baseline_rows:
    summary_df = pd.concat([summary_df, pd.DataFrame(baseline_rows)], ignore_index=True)

summary_df = summary_df.sort_values(["source", "axis", "alpha"])
summary_df.to_csv(SUMMARY_CSV, index=False)
print(f"\nSaved summary → {SUMMARY_CSV}")

# ============================================================
# ONE LARGE GRAPH: good translations only, actual COMET score
# ============================================================

fig, ax = plt.subplots(figsize=(16, 10))

for source, axis, pair_title in PAIRS:
    grp = summary_df[
        (summary_df["source"] == source)
        & (summary_df["axis"] == axis)
        & (summary_df["group"] == "good")
    ].sort_values("alpha")

    if grp.empty:
        continue

    ax.plot(
        grp["alpha"].to_numpy(),
        grp["ablation_comet"].to_numpy(),
        marker="o",
        linewidth=2,
        markersize=5,
        label=pair_title,
    )

ax.set_xlabel("Ablation strength α", fontsize=14)
ax.set_ylabel("Mean COMET-Kiwi score", fontsize=14)
ax.set_title(
    "COMET-Kiwi under language-specific representation ablation\n"
    "Good translations only",
    fontsize=16,
)
ax.set_xlim(0.0, 0.5)
ax.set_xticks(ALPHAS)
ax.grid(True, alpha=0.25)
ax.legend(fontsize=10, ncol=2)

plt.tight_layout()

combined_path = os.path.join(
    PLOT_DIR, "dose_response_all_12_directions.png"
)
plt.savefig(combined_path, dpi=250, bbox_inches="tight")
plt.close()
print(f"Saved → {combined_path}")

# ============================================================
# Per-direction good-only graphs
# ============================================================

for source, axis, pair_title in PAIRS:
    grp = summary_df[
        (summary_df["source"] == source)
        & (summary_df["axis"] == axis)
        & (summary_df["group"] == "good")
    ].sort_values("alpha")

    if grp.empty:
        continue

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        grp["alpha"].to_numpy(),
        grp["ablation_comet"].to_numpy(),
        marker="o",
        linewidth=2,
        markersize=6,
    )
    ax.set_xlabel("Ablation strength α", fontsize=12)
    ax.set_ylabel("Mean COMET-Kiwi score", fontsize=12)
    ax.set_title(f"{pair_title}\nGood translations only", fontsize=14)
    ax.set_xlim(0.0, 0.5)
    ax.set_xticks(ALPHAS)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()

    save_path = os.path.join(PLOT_DIR, f"dose_response_good_{source}_{axis}.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved → {save_path}")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
print(f"Summary: {SUMMARY_CSV}")
print(f"Main graph: {combined_path}")
