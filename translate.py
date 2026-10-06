"""
translate.py — One-way translation pipeline EN → target language.

Translates source sentences into a single target language using Apertus.
Shardable across SLURM nodes. Saves shards and merges into a single CSV.
Empty translations are retried immediately before moving to the next row.

Usage:
    python translate.py \
        --shard_id 0 --num_shards 1 \
        --target_lang "German" \
        --input_csv slurm/translations/rerun/rerun_german.csv \
        --output_dir slurm/translations/apertus

SLURM: one job per (language x shard).
"""

import pandas as pd
import openai
from tqdm import tqdm
import time
import signal
import argparse
import os
from glob import glob

# ============================================================
# Arguments
# ============================================================

parser = argparse.ArgumentParser()
parser.add_argument("--shard_id",    type=int, default=0)
parser.add_argument("--num_shards",  type=int, default=4)
parser.add_argument("--target_lang", type=str, required=True,
                    help="One of: German, Dutch, Afrikaans, Spanish, Catalan, Galician")
parser.add_argument("--model",       type=str,
                    default="swiss-ai/Apertus-8B-Instruct-2509")
parser.add_argument("--input_csv",   type=str,
                    default="thesis-final/codebase/filtered_translation_eval_10k.csv")
parser.add_argument("--output_dir",  type=str,
                    default="slurm/translations/apertus")
parser.add_argument("--limit",       type=int, default=10000)
args = parser.parse_args()

BATCH_SIZE            = 10
MAX_RETRIES           = 3
MAX_ROW_RETRIES       = 5   # extra attempts for rows that come back empty
SLEEP_BETWEEN_RETRIES = 1.0
REQUEST_TIMEOUT       = 60

lang_tag  = args.target_lang.lower().replace(" ", "_")
model_tag = args.model.split("/")[-1].lower()

temp_dir   = os.path.join(args.output_dir, "temp", lang_tag)
os.makedirs(temp_dir, exist_ok=True)

SHARD_CSV  = os.path.join(temp_dir,
             f"shard_{args.shard_id:03d}_of_{args.num_shards:03d}.csv")
MERGED_CSV = os.path.join(args.output_dir,
             f"translations_{lang_tag}_{model_tag}.csv")

# ============================================================
# API client
# ============================================================

client = openai.Client(
    api_key="sk-rc-eGGDBJN5V3ujGbDcNrpZsQ",
    base_url="https://api.swissai.cscs.ch/v1"
)

# ============================================================
# Timeout handling
# ============================================================

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("API call timed out")

signal.signal(signal.SIGALRM, timeout_handler)

# ============================================================
# Translation
# ============================================================

def translate_batch(sentences, batch_id, target_lang):
    prompt  = f"Translate the following sentences from English to {target_lang}.\n"
    prompt += "Output exactly one translated sentence per line, keeping the same order.\n"
    prompt += "Do not add numbering, explanations, or any extra text.\n\n"
    for s in sentences:
        prompt += f"{s}\n"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"  -> [EN->{target_lang}] Batch {batch_id}, attempt {attempt}")
            signal.alarm(REQUEST_TIMEOUT)

            res = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                stream=False,
            )
            signal.alarm(0)

            text  = res.choices[0].message.content.strip()
            lines = [l.strip() for l in text.splitlines() if l.strip()]

            # Strip any numbering the model may have added
            cleaned = []
            for line in lines:
                if len(line) > 2 and line[0].isdigit() and line[1] in ".):-":
                    line = line.split(None, 1)[-1].strip()
                cleaned.append(line)

            while len(cleaned) < len(sentences):
                cleaned.append("")

            return cleaned[:len(sentences)]

        except TimeoutException:
            print(f"  Batch {batch_id} timed out, retrying...")
        except Exception as e:
            print(f"  Batch {batch_id} failed: {e}")
        finally:
            signal.alarm(0)

        time.sleep(SLEEP_BETWEEN_RETRIES)

    print(f"  Batch {batch_id} gave up — returning empty strings")
    return [""] * len(sentences)

# ============================================================
# Load data
# ============================================================

print(f"Loading {args.input_csv}...")
df = pd.read_csv(args.input_csv).head(args.limit)

for col in ["path", "sentence"]:
    if col not in df.columns:
        raise ValueError(f"Missing column '{col}'. Found: {list(df.columns)}")

df = df.reset_index(drop=True)
df["row_id"] = df.index  # stable across all languages

# ============================================================
# Sharding
# ============================================================

shard_df = df[df["row_id"] % args.num_shards == args.shard_id].reset_index(drop=True)
print(f"Shard {args.shard_id}/{args.num_shards} -> {len(shard_df)} rows")
print(f"Target language : {args.target_lang}")
print(f"Model           : {args.model}")

# ============================================================
# Resume — only count rows with valid translations as done
# ============================================================

if os.path.exists(SHARD_CSV):
    done_df = pd.read_csv(SHARD_CSV)
    done_df = done_df[
        done_df["translation"].notna() &
        (done_df["translation"].astype(str).str.strip() != "")
    ]
    done_row_ids = set(done_df["row_id"].astype(int))
    print(f"Resuming — {len(done_row_ids)} rows with valid translations")
else:
    done_df      = pd.DataFrame()
    done_row_ids = set()

rows_to_process = shard_df[
    ~shard_df["row_id"].isin(done_row_ids)
].reset_index(drop=True)
print(f"Rows to process : {len(rows_to_process)}")

# ============================================================
# Translation loop — retry empty rows immediately before moving on
# ============================================================

results       = []
batch_id      = 0
total_batches = (len(rows_to_process) + BATCH_SIZE - 1) // BATCH_SIZE

for i in tqdm(range(0, len(rows_to_process), BATCH_SIZE),
              total=total_batches,
              desc=f"{args.target_lang} shard {args.shard_id}",
              unit="batch"):
    batch_id += 1
    batch     = rows_to_process.iloc[i:i + BATCH_SIZE]
    sentences = batch["sentence"].astype(str).tolist()

    translations = translate_batch(sentences, batch_id, args.target_lang)

    # Retry any empty translations one row at a time before saving
    for row, translation in zip(batch.itertuples(index=False), translations):
        if not translation or not translation.strip():
            print(f"  Row {row.row_id} empty — retrying individually...")
            for retry in range(1, MAX_ROW_RETRIES + 1):
                result      = translate_batch([str(row.sentence)], f"{batch_id}r{retry}", args.target_lang)
                translation = result[0] if result else ""
                if translation and translation.strip():
                    print(f"  Row {row.row_id} succeeded on retry {retry}")
                    break
                print(f"  Row {row.row_id} still empty on retry {retry}")
                time.sleep(SLEEP_BETWEEN_RETRIES)
            else:
                print(f"  Row {row.row_id} gave up after {MAX_ROW_RETRIES} retries")

        results.append({
            "row_id":      row.row_id,
            "path":        row.path,
            "source":      row.sentence,
            "target_lang": args.target_lang,
            "translation": translation,
            "model":       args.model,
        })

    # Save incrementally every 20 batches
    if len(results) >= BATCH_SIZE * 20:
        chunk   = pd.DataFrame(results)
        done_df = pd.concat([done_df, chunk], ignore_index=True)
        done_df.to_csv(SHARD_CSV, index=False)
        results = []

# Final flush
if results:
    chunk   = pd.DataFrame(results)
    done_df = pd.concat([done_df, chunk], ignore_index=True)

done_df.to_csv(SHARD_CSV, index=False)
print(f"Shard saved -> {SHARD_CSV}")

# ============================================================
# Merge — triggered only by shard 0 once all shards exist
# ============================================================

if args.shard_id == 0:
    shard_files = sorted(glob(os.path.join(temp_dir, "shard_*_of_*.csv")))
    expected    = args.num_shards

    if len(shard_files) < expected:
        print(f"Only {len(shard_files)}/{expected} shards found — skipping merge.")
        print(f"Re-run shard 0 after all shards complete to trigger merge.")
    else:
        merged = pd.concat(
            [pd.read_csv(f) for f in shard_files], ignore_index=True
        )

        # Drop duplicates — keep last (most recent retry wins)
        n_before = len(merged)
        merged   = merged.drop_duplicates(subset="row_id", keep="last")
        n_dupes  = n_before - len(merged)
        if n_dupes:
            print(f"WARNING: dropped {n_dupes} duplicate row_ids")

        # Report any remaining empty translations
        n_empty = (
            merged["translation"].isna() |
            (merged["translation"].astype(str).str.strip() == "")
        ).sum()
        if n_empty:
            print(f"WARNING: {n_empty} rows still have empty translations")

        merged = merged.sort_values("row_id").reset_index(drop=True)

        if len(merged) != args.limit:
            print(f"WARNING: expected {args.limit} rows, got {len(merged)}")
        else:
            print(f"All {len(merged)} rows present and aligned.")

        merged.to_csv(MERGED_CSV, index=False)
        print(f"Merged -> {MERGED_CSV}")