"""
aggregate_for_visualisation.py — bridges rescore.py's per-sentence output
to visualise_ablation.py's expected input.

WHY THIS STEP EXISTS
    rescore.py produces one CSV per (source_lang, axis_lang, condition,
    alpha) run, with a per-sentence "comet_delta" column computed against
    "baseline_comet" — the STORED translation's original score from your
    earlier translation pass, a different generation run entirely.

    visualise_ablation.py wants "delta_vs_none" — each ablated sentence's
    score compared against a freshly-generated, UNABLATED run of that same
    sentence through the same code path (ablation.py's --condition none).
    That's a different, cleaner baseline than the stored one, and it's
    not something rescore.py computes at all.

    This script:
      1. Loads every rescored *.csv matching --input_glob
      2. Separately identifies the "none"-condition files (one needed per
         source_lang — axis_lang doesn't matter for an unhooked run, so
         if you ran --condition none multiple times with different
         --axis_lang values, they should be near-identical; this script
         averages duplicates rather than picking one arbitrarily)
      3. For every ablation-condition row, matches it to the "none" row
         for the SAME sentence (joined on the "source" text) and computes
         delta_vs_none = comet_generated(ablated) - comet_generated(none)
      4. Aggregates to mean + SE per (source_lang, axis_lang, alpha,
         group_name) — exactly the schema visualise_ablation.py expects

REQUIREMENT: you need at least one --condition none run per source_lang
before this script can do anything useful. If you haven't run that yet,
run it first — this script will tell you clearly which source_langs are
missing it rather than silently falling back to the stored baseline.

Usage:
    python agg.py \
        --input_glob "rescored_results/*.csv" \
        --out ablation_results/ablation_summary_vs_none_n2000.csv
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--input_glob", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()

files = sorted(glob.glob(args.input_glob))
if not files:
    raise SystemExit(f"No files matched {args.input_glob}")
print(f"Loading {len(files)} rescored file(s)...")

dfs = [pd.read_csv(f) for f in files]
df = pd.concat(dfs, ignore_index=True)
print(f"Total rows: {len(df)}")

required_cols = {"source_lang", "axis_lang", "condition", "alpha", "group_name",
                  "source", "comet_generated"}
missing = required_cols - set(df.columns)
if missing:
    raise SystemExit(f"Input files are missing expected columns: {missing}")

# ── Build the "none" baseline lookup, per (source_lang, source text) ────────

none_df = df[df["condition"] == "none"]
if none_df.empty:
    raise SystemExit(
        "No rows with condition=='none' found in the input files.\n"
        "You need to run ablation.py with --condition none for each "
        "source_lang before this aggregation step means anything — see "
        "this script's docstring for why the stored baseline_comet isn't "
        "an acceptable substitute."
    )

present_source_langs = set(df[df["condition"] != "none"]["source_lang"].unique())
none_source_langs = set(none_df["source_lang"].unique())
missing_none = present_source_langs - none_source_langs
if missing_none:
    print(f"WARNING: no 'none' condition run found for source_lang(s) "
          f"{sorted(missing_none)} — rows for these will be dropped, since "
          f"there's no valid vs_none baseline to compare against.")

# average duplicate none-condition runs for the same sentence (e.g. if you
# ran --condition none once per axis_lang out of habit — they should be
# near-identical since no hooks fire regardless of axis_lang)
none_baseline = (
    none_df.groupby(["source_lang", "source"])["comet_generated"]
    .mean()
    .rename("none_comet")
    .reset_index()
)
print(f"Built 'none' baseline for {none_baseline['source_lang'].nunique()} "
      f"source_lang(s), {len(none_baseline)} unique sentences total.")

# ── Join ablation-condition rows to their matching none baseline ────────────

ablated = df[~df["condition"].isin(["none", "baseline"])].copy()
before = len(ablated)
ablated = ablated.merge(none_baseline, on=["source_lang", "source"], how="inner")
after = len(ablated)
if after < before:
    print(f"NOTE: {before - after} rows dropped (no matching 'none'-condition "
          f"sentence found — check that the same sentence sample was used "
          f"for both the none run and the ablation runs).")

ablated["delta_vs_none"] = ablated["comet_generated"] - ablated["none_comet"]

# ── Aggregate to mean + SE per (source_lang, axis_lang, alpha, group_name) ──

def sem(x):
    return x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0

summary = (
    ablated.groupby(["source_lang", "axis_lang", "alpha", "group_name"])["delta_vs_none"]
    .agg(delta_vs_none="mean", se_delta_vs_none=sem, n="count")
    .reset_index()
)

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
summary.to_csv(args.out, index=False)
print(f"\nSaved -> {args.out}")
print(f"{len(summary)} (pair, alpha, group) rows, "
      f"{summary['source_lang'].nunique()} source langs, "
      f"{summary.groupby(['source_lang','axis_lang']).ngroups} pairs")
print("\nThis file is ready for visualise_ablation.py --input", args.out)