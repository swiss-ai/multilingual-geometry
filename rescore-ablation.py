"""
rescore.py — Rescore ablation outputs with COMET-Kiwi (reference-free).

Reads one or more CSVs from ablation_results/, computes COMET-Kiwi on the
generated translations, and writes a new CSV with added columns:
  - comet_generated   : COMET-Kiwi score for the generated translation
  - comet_delta       : comet_generated - baseline_comet

Usage:
    # Single file
    python rescore.py \
        --input  ablation_results/gl_ablate_es_late_a0.1_t0.05.csv

    # Glob pattern
    python rescore.py \
        --input  "ablation_results/gl_ablate_es_*.csv"

    # Multiple explicit files
    python rescore.py \
        --input  ablation_results/gl_ablate_es_late_a0.1_t0.05.csv \
                 ablation_results/gl_ablate_es_early_a0.1_t0.05.csv

    # Override output directory
    python rescore.py \
        --input  "ablation_results/*.csv" \
        --output_dir rescored_results

Options:
    --batch_size   Sentences per COMET batch (default 32; reduce if OOM)
    --model_path   Override COMET model (default: Unbabel/wmt22-cometkiwi-da)
    --device       cuda / cpu (auto-detected if omitted)
    --skip_done    Skip files that already have a rescored output
"""

import argparse
import glob
import os
import sys

import pandas as pd
import torch
from comet import download_model, load_from_checkpoint

# ── Args ─────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--input", nargs="+", required=True,
                    help="Input CSV path(s) or glob pattern(s)")
parser.add_argument("--output_dir", type=str, default=None,
                    help="Output directory (default: same as each input file)")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--model_path", type=str,
                    default="Unbabel/wmt22-cometkiwi-da")
parser.add_argument("--device", type=str, default=None)
parser.add_argument("--skip_done", action="store_true",
                    help="Skip if rescored output already exists")
args = parser.parse_args()

# ── Resolve input files ───────────────────────────────────────────────────────

input_files = []
for pattern in args.input:
    matched = glob.glob(pattern)
    if matched:
        input_files.extend(matched)
    elif os.path.isfile(pattern):
        input_files.append(pattern)
    else:
        print(f"WARNING: no files matched pattern '{pattern}'")

input_files = sorted(
    set(f for f in input_files if not f.endswith("_rescored.csv"))
)
if not input_files:
    print("ERROR: no input files found.")
    sys.exit(1)

print(f"Found {len(input_files)} file(s) to rescore.")

# ── Device ───────────────────────────────────────────────────────────────────

if args.device:
    device = args.device
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {device}")

# ── Load COMET-Kiwi ──────────────────────────────────────────────────────────

print(f"\nLoading COMET model: {args.model_path} ...")
model_path = download_model(args.model_path)
comet_model = load_from_checkpoint(model_path)
comet_model.eval()
print("COMET model ready.\n")

# ── Helper: score a list of (src, mt) pairs ───────────────────────────────────

def score_pairs(sources, translations, batch_size):
    """
    Returns a list of floats — one COMET-Kiwi score per sentence pair.
    Skips rows where translation is empty/NaN and returns NaN for them.
    """
    assert len(sources) == len(translations)
    scores = [float("nan")] * len(sources)

    valid_idx = []
    valid_data = []
    for i, (src, mt) in enumerate(zip(sources, translations)):
        if not isinstance(mt, str) or not mt.strip():
            continue
        valid_idx.append(i)
        valid_data.append({"src": str(src), "mt": str(mt)})

    if not valid_data:
        return scores

    result = comet_model.predict(
        valid_data,
        batch_size=batch_size,
        gpus=1 if device == "cuda" else 0,
        progress_bar=True,
    )

    for i, score in zip(valid_idx, result.scores):
        scores[i] = float(score)

    return scores

# ── Process each file ────────────────────────────────────────────────────────

for fpath in input_files:
    fname = os.path.basename(fpath)

    # Determine output path
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, fname.replace(".csv", "_rescored.csv"))
    else:
        out_path = fpath.replace(".csv", "_rescored.csv")

    if args.skip_done and os.path.isfile(out_path):
        print(f"[skip] {fname} → already rescored at {out_path}")
        continue

    print(f"\n{'='*60}")
    print(f"File   : {fname}")
    print(f"Output : {out_path}")

    df = pd.read_csv(fpath)
    n = len(df)
    print(f"Rows   : {n}")

    condition = df["condition"].iloc[0] if "condition" in df.columns else "unknown"

    # Reference baseline: generated == reference, skip redundant scoring
    # but still compute so the output schema is consistent
    print(f"Condition: {condition}")
    print("Scoring generated translations...")

    scores_gen = score_pairs(
        df["source"].tolist(),
        df["generated"].tolist(),
        args.batch_size,
    )

    df["comet_generated"] = scores_gen
    df["comet_delta"] = df["comet_generated"] - df["baseline_comet"]

    # ── Per-group summary ──────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"  {'Group':<10} {'N':>5}  {'Δ mean':>8}  {'Δ std':>7}  "
          f"{'base mean':>10}  {'gen mean':>9}")
    print(f"{'─'*50}")
    for gname in ["bad", "good"]:
        sub = df[df["group_name"] == gname]
        if sub.empty:
            continue
        print(f"  {gname:<10} {len(sub):>5}  "
              f"{sub['comet_delta'].mean():>+8.4f}  "
              f"{sub['comet_delta'].std():>7.4f}  "
              f"{sub['baseline_comet'].mean():>10.4f}  "
              f"{sub['comet_generated'].mean():>9.4f}")
    print(f"{'─'*50}")
    overall = df["comet_delta"]
    print(f"  {'overall':<10} {len(df):>5}  "
          f"{overall.mean():>+8.4f}  "
          f"{overall.std():>7.4f}")
    print(f"{'─'*50}\n")

    df.to_csv(out_path, index=False)
    print(f"Saved → {out_path}")

print("\nAll files rescored.")