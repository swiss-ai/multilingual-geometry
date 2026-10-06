"""
comet-eval.py — Compute COMET-Kiwi quality-estimation scores for
one-way translations (EN → target language).

Scores each translation as a (source, mt) pair using the reference-free
COMET-Kiwi model. Appends a `comet_kiwi` column to each CSV.

Input CSVs are expected to have columns: source, translation
(as produced by translate.py)

Usage:
    # Score a single language
    python comet-eval.py --input_csv translations/apertus/translations_german_apertus-8b-instruct-2509.csv

    # Score all languages in a directory
    python comet-eval.py --input_dir translations/apertus

    # Score a single file sharded across 4 SLURM tasks
    python comet-eval.py --input_csv translations/apertus/translations_catalan_apertus-8b-instruct-2509.csv \
        --shard_id 0 --num_shards 4

    # Merge shards after all tasks complete
    python comet-eval.py --input_csv translations/apertus/translations_catalan_apertus-8b-instruct-2509.csv \
        --num_shards 4 --merge_only
"""

import argparse
import os
from glob import glob

import pandas as pd
import torch
from comet import download_model, load_from_checkpoint

# ============================================================
# Arguments
# ============================================================

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--input_csv", type=str,
                   help="Score a single CSV file")
group.add_argument("--input_dir", type=str,
                   help="Score all translations_*.csv files in this directory")

parser.add_argument("--output_dir",    type=str, default=None,
                    help="Directory to write scored CSVs. "
                         "Defaults to overwriting input files in place.")
parser.add_argument("--batch_size",    type=int, default=64)
parser.add_argument("--model",         type=str,
                    default="Unbabel/wmt22-cometkiwi-da")
parser.add_argument("--gpus",          type=int,
                    default=1 if torch.cuda.is_available() else 0)
parser.add_argument("--skip_existing", action="store_true",
                    help="Skip files that already have a comet_kiwi column")
parser.add_argument("--shard_id",      type=int, default=0,
                    help="Index of this shard (0-indexed)")
parser.add_argument("--num_shards",    type=int, default=1,
                    help="Total number of shards")
parser.add_argument("--merge_only",    action="store_true",
                    help="Skip scoring — just merge existing shard files")
args = parser.parse_args()

# ============================================================
# Resolve input files
# ============================================================

if args.input_csv:
    input_files = [args.input_csv]
else:
    input_files = sorted(glob(os.path.join(args.input_dir, "translations_*.csv")))
    if not input_files:
        raise FileNotFoundError(f"No translations_*.csv files found in {args.input_dir}")
    print(f"Found {len(input_files)} translation files:")
    for f in input_files:
        print(f"  {f}")

if args.output_dir:
    os.makedirs(args.output_dir, exist_ok=True)

# ============================================================
# Shard path helper
# ============================================================

def shard_path(output_path, shard_id, num_shards):
    base, ext = os.path.splitext(output_path)
    return f"{base}_shard{shard_id}of{num_shards}{ext}"

# ============================================================
# Merge shards
# ============================================================

def merge_shards(output_path, num_shards):
    base, ext = os.path.splitext(output_path)
    shard_files = [f"{base}_shard{i}of{num_shards}{ext}" for i in range(num_shards)]
    missing = [f for f in shard_files if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(f"Missing shard files: {missing}")

    merged = pd.concat(
        [pd.read_csv(f) for f in shard_files], ignore_index=True
    ).sort_index()

    merged.to_csv(output_path, index=False)
    print(f"  Merged {num_shards} shards -> {output_path} ({len(merged)} rows)")

    for f in shard_files:
        os.remove(f)
    print(f"  Shard files deleted.")

# ============================================================
# Load COMET-Kiwi model (skip if merge only)
# ============================================================

if not args.merge_only:
    print(f"\nLoading COMET-Kiwi model '{args.model}'...")
    model_path  = download_model(args.model)
    comet_model = load_from_checkpoint(model_path)
    print("Model loaded.\n")

# ============================================================
# Score each file
# ============================================================

def score_file(input_path, output_path):
    print(f"{'='*55}")
    print(f"Scoring: {input_path}")

    df = pd.read_csv(input_path)

    # Validate columns
    for col in ["source", "translation"]:
        if col not in df.columns:
            raise ValueError(
                f"Missing column '{col}' in {input_path}. "
                f"Found: {list(df.columns)}"
            )

    # Sharding
    if args.num_shards > 1:
        out_path = shard_path(output_path, args.shard_id, args.num_shards)
        df = df.iloc[args.shard_id::args.num_shards].reset_index(drop=True)
        print(f"  Shard {args.shard_id}/{args.num_shards} -> {len(df)} rows")
    else:
        out_path = output_path

    # Skip if already scored
    if args.skip_existing and os.path.exists(out_path):
        existing = pd.read_csv(out_path)
        if "comet_kiwi" in existing.columns:
            print(f"  Already scored -> {out_path} — skipping.")
            return

    # Fill nulls so COMET doesn't crash
    df["source"]      = df["source"].fillna("").astype(str)
    df["translation"] = df["translation"].fillna("").astype(str)

    # Flag empty translations before scoring
    n_empty = (df["translation"].str.strip() == "").sum()
    if n_empty:
        print(f"  WARNING: {n_empty} empty translations — will score as 0.0")

    # Build COMET input — empty translations get 0.0 directly
    scores = []
    non_empty_mask = df["translation"].str.strip() != ""
    non_empty_data = [
        {"src": row.source, "mt": row.translation}
        for row in df[non_empty_mask].itertuples(index=False)
    ]

    if non_empty_data:
        out = comet_model.predict(
            non_empty_data,
            batch_size=args.batch_size,
            gpus=args.gpus,
        )
        scored_iter = iter(out.scores)
    else:
        scored_iter = iter([])

    for is_non_empty in non_empty_mask:
        if is_non_empty:
            scores.append(round(next(scored_iter), 6))
        else:
            scores.append(0.0)

    df["comet_kiwi"] = scores

    # Summary stats (excluding empty rows)
    valid_scores = df.loc[non_empty_mask, "comet_kiwi"]
    print(f"  Rows scored : {non_empty_mask.sum()} / {len(df)}")
    print(f"  Mean  : {valid_scores.mean():.4f}")
    print(f"  Std   : {valid_scores.std():.4f}")
    print(f"  Min   : {valid_scores.min():.4f}")
    print(f"  Max   : {valid_scores.max():.4f}")
    print(f"  <0.4  : {(valid_scores < 0.4).sum()} rows (low quality)")
    print(f"  >0.8  : {(valid_scores > 0.8).sum()} rows (high quality)")

    df.to_csv(out_path, index=False)
    print(f"  Saved -> {out_path}")


for input_path in input_files:
    if args.output_dir:
        filename    = os.path.basename(input_path)
        output_path = os.path.join(args.output_dir, filename)
    else:
        output_path = input_path

    try:
        if args.merge_only:
            merge_shards(output_path, args.num_shards)
        else:
            score_file(input_path, output_path)
            if args.num_shards > 1 and args.shard_id == args.num_shards - 1:
                print("\nLast shard complete — merging...")
                merge_shards(output_path, args.num_shards)
    except Exception as e:
        print(f"  ERROR: {e}")
        continue

print(f"\n{'='*55}")
print("Done.")