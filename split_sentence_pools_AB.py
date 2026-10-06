"""
split_sentence_pools_AB.py — Split A (reference) and B (layer_select) only,
excluding row_ids already used for Split C (ablation), which you've
already extracted.

Uses a PER-LANGUAGE PERCENTILE threshold for high-quality filtering by
default, not a fixed absolute COMET-Kiwi score — languages have different
score ceilings (e.g. German's max in this dataset is 0.8907), so a fixed
threshold like 0.89 can leave almost nothing for some languages while
being generous for others.
"""

import argparse
import glob
import json
import os
import re

import pandas as pd

LANGUAGES = {
    "es": {"csv": "translations_spanish_{model}.csv",   "hidden_dir": "spanish"},
    "ca": {"csv": "translations_catalan_{model}.csv",   "hidden_dir": "catalan"},
    "gl": {"csv": "translations_galician_{model}.csv",  "hidden_dir": "galician"},
    "de": {"csv": "translations_german_{model}.csv",    "hidden_dir": "german"},
    "nl": {"csv": "translations_dutch_{model}.csv",     "hidden_dir": "dutch"},
    "af": {"csv": "translations_afrikaans_{model}.csv", "hidden_dir": "afrikaans"},
}
MODEL_SLUG = "apertus-8b-instruct-2509"
ROW_ID_RE = re.compile(r"row(\d+)\.npz$")

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, required=True)
parser.add_argument("--hidden_states_base", type=str, required=True)
parser.add_argument("--min_comet", type=float, default=None,
                     help="Fixed absolute COMET-Kiwi threshold. NOT "
                          "recommended across languages with different "
                          "score ceilings. Use --min_comet_pct instead "
                          "unless you have a specific reason for an "
                          "absolute cutoff.")
parser.add_argument("--min_comet_pct", type=float, default=90.0,
                     help="Per-language percentile threshold. Default 90 "
                          "= top 10%% of each language's OWN score "
                          "distribution (computed after removing Split "
                          "C's rows). Ignored if --min_comet is set.")
parser.add_argument("--n_reference_json", type=str, default=None)
parser.add_argument("--n_reference_default", type=int, default=200)
parser.add_argument("--n_layer_select", type=int, default=200)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--out_dir", type=str, default="sentence_splits")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

if args.n_reference_json:
    with open(args.n_reference_json) as f:
        n_reference_by_lang = json.load(f)
else:
    n_reference_by_lang = {code: args.n_reference_default for code in LANGUAGES}

summary = {}

for code, info in LANGUAGES.items():
    csv_path = os.path.join(args.input_dir, info["csv"].format(model=MODEL_SLUG))
    hidden_dir = os.path.join(args.hidden_states_base, info["hidden_dir"])

    if not os.path.exists(csv_path):
        print(f"WARNING: {csv_path} not found, skipping {code}.")
        continue
    if not os.path.isdir(hidden_dir):
        print(f"WARNING: {hidden_dir} not found, skipping {code}.")
        continue

    print(f"{'='*60}")
    print(f"{code}")
    print(f"{'='*60}")

    npz_files = glob.glob(os.path.join(hidden_dir, "row*.npz"))
    split_c_ids = set()
    for f in npz_files:
        m = ROW_ID_RE.search(os.path.basename(f))
        if m:
            split_c_ids.add(int(m.group(1)))

    print(f"Found {len(split_c_ids)} row_ids already used for Split C "
          f"(from {hidden_dir})")

    if len(split_c_ids) == 0:
        print(f"  WARNING: no .npz files found matching row*.npz in {hidden_dir}")

    df = pd.read_csv(csv_path)
    print(f"Total sentences: {len(df)}")

    remaining = df[~df["row_id"].isin(split_c_ids)].reset_index(drop=True)
    print(f"Remaining after excluding Split C: {len(remaining)}")
    print(f"  Score range in remaining pool: "
          f"{remaining['comet_kiwi'].min():.4f}-{remaining['comet_kiwi'].max():.4f}")

    if args.min_comet is not None:
        threshold = args.min_comet
        print(f"Using fixed absolute threshold: comet_kiwi >= {threshold}")
    else:
        threshold = remaining["comet_kiwi"].quantile(args.min_comet_pct / 100.0)
        print(f"Using per-language {args.min_comet_pct:.0f}th percentile "
              f"threshold: comet_kiwi >= {threshold:.4f} (computed from "
              f"this language's own remaining-pool distribution)")

    hq_pool = remaining[remaining["comet_kiwi"] >= threshold].reset_index(drop=True)
    print(f"High-quality pool: {len(hq_pool)}")

    if len(hq_pool) == 0:
        print(f"  ERROR: no high-quality sentences remain for {code}. Skipping.")
        continue

    n_ref_this = min(n_reference_by_lang.get(code, args.n_reference_default), len(hq_pool))
    n_layer_this = min(args.n_layer_select, len(hq_pool) - n_ref_this)

    if n_layer_this <= 0:
        print(f"  WARNING: not enough sentences left for Split B.")
        n_layer_this = max(0, n_layer_this)

    split_a = hq_pool.sample(n=n_ref_this, random_state=args.seed).reset_index(drop=True)
    hq_remaining = hq_pool[~hq_pool["row_id"].isin(split_a["row_id"])].reset_index(drop=True)
    split_b = hq_remaining.sample(
        n=min(n_layer_this, len(hq_remaining)), random_state=args.seed
    ).reset_index(drop=True)

    print(f"Split A (reference):    {len(split_a)} sentences")
    print(f"Split B (layer_select): {len(split_b)} sentences")

    ids_a = set(split_a["row_id"])
    ids_b = set(split_b["row_id"])
    assert not (ids_a & ids_b), f"Split A/B overlap for {code}"
    assert not (ids_a & split_c_ids), f"Split A overlaps existing Split C for {code}"
    assert not (ids_b & split_c_ids), f"Split B overlaps existing Split C for {code}"
    print("Disjointness check: PASSED")

    lang_dir = os.path.join(args.out_dir, code)
    os.makedirs(lang_dir, exist_ok=True)
    split_a.to_csv(os.path.join(lang_dir, "split_A_reference.csv"), index=False)
    split_b.to_csv(os.path.join(lang_dir, "split_B_layer_select.csv"), index=False)
    print(f"Saved -> {lang_dir}/split_A_reference.csv, split_B_layer_select.csv\n")

    summary[code] = {
        "total": len(df),
        "split_C_already_used": len(split_c_ids),
        "threshold_used": float(threshold),
        "split_A_reference": len(split_a),
        "split_B_layer_select": len(split_b),
    }

print(f"{'='*60}")
print("Summary")
print(f"{'='*60}")
print(f"{'lang':>6}  {'total':>7}  {'C (existing)':>13}  {'threshold':>10}  {'A (ref)':>9}  {'B (layer)':>10}")
for code, s in summary.items():
    print(f"{code:>6}  {s['total']:>7}  {s['split_C_already_used']:>13}  "
          f"{s['threshold_used']:>10.4f}  {s['split_A_reference']:>9}  {s['split_B_layer_select']:>10}")

with open(os.path.join(args.out_dir, "split_summary_AB.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved summary -> {args.out_dir}/split_summary_AB.json")
