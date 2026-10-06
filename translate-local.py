"""
translate_local.py — One-way translation pipeline EN -> target language,
using a LOCALLY LOADED model (transformers) instead of CSCS's hosted
OpenAI-compatible API.

Use this instead of translate.py for any model CSCS doesn't host --
e.g. Gemma-7B, Llama-8B. Same batching, retry-on-empty, resume, and
sharding/merge logic as translate.py; only the generation backend
changed (local transformers .generate() instead of a remote API call).

WHAT'S DIFFERENT FROM translate.py, AND WHY
    - No openai.Client / API key. Model is loaded once per process with
      transformers, matching the same convention used elsewhere in this
      project (ablation.py): device_map="auto", bfloat16,
      attn_implementation="eager" (change to "sdpa" if you specifically
      want that instead -- see ablation.py's own switch for context).
    - Chat formatting is now YOUR responsibility, not the API's. The
      original script sent messages=[{"role":"user","content":prompt}]
      to a server that applied the model's chat template internally.
      Here, tokenizer.apply_chat_template(...) does that explicitly --
      Gemma and Llama format instruct prompts differently from each
      other and from Apertus, so this step matters and can't be skipped
      or shared across models.
    - No SIGALRM network-timeout handling -- that was for a remote HTTP
      call that could hang on the network; a local, deterministic
      (do_sample=False) generate() call doesn't have that failure mode
      the same way. Retry-on-exception and retry-on-empty-output are
      kept, since those can still happen locally (OOM, degenerate
      output, etc).

REQUIREMENT -- gated models. Gemma and Llama checkpoints on HuggingFace
Hub require you to accept each model's license on its HF page, logged
into the SAME account your HUGGING_FACE_HUB_TOKEN belongs to, before the
token will successfully authenticate for download. If you get a 401/403
on model load, this is almost always why -- check
https://huggingface.co/<model_id> for an "Agree and access" gate you
haven't clicked through yet, separately for EACH model (Gemma and Llama
are gated independently, accepting one doesn't grant the other).

Usage:
    python translate_local.py \
        --shard_id 0 --num_shards 1 \
        --target_lang "German" \
        --model "google/gemma-7b-it" \
        --input_csv slurm/translations/rerun/rerun_german.csv \
        --output_dir slurm/translations/gemma

    python translate_local.py \
        --shard_id 0 --num_shards 1 \
        --target_lang "German" \
        --model "meta-llama/Llama-3.1-8B-Instruct" \
        --input_csv slurm/translations/rerun/rerun_german.csv \
        --output_dir slurm/translations/llama

SLURM: one job per (language x shard), same as translate.py -- the model
loads once per process, so shard granularity should stay coarse enough
that checkpoint-loading time doesn't dominate (same consideration as
your existing ablation.py jobs).
"""

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import time
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
parser.add_argument("--model",       type=str, required=True,
                    help="HF model id, e.g. google/gemma-7b-it or "
                         "meta-llama/Llama-3.1-8B-Instruct")
parser.add_argument("--input_csv",   type=str,
                    default="thesis-final/codebase/filtered_translation_eval_10k.csv")
parser.add_argument("--output_dir",  type=str,
                    default="slurm/translations/local")
parser.add_argument("--limit",       type=int, default=10000)
parser.add_argument("--max_new_tokens", type=int, default=512,
                    help="raise this if batches of 10 sentences get truncated")
parser.add_argument("--prompt_batch_size", type=int, default=8,
                    help="number of 10-sentence PROMPTS processed together in one "
                         "model.generate() call. Higher = faster (better GPU "
                         "utilization) but more memory; lower if you hit OOM.")
args = parser.parse_args()

BATCH_SIZE            = 10   # sentences per prompt (unchanged from translate.py)
MAX_RETRIES           = 3
MAX_ROW_RETRIES       = 5   # extra attempts for rows that come back empty
SLEEP_BETWEEN_RETRIES = 1.0

lang_tag  = args.target_lang.lower().replace(" ", "_")
model_tag = args.model.split("/")[-1].lower()

temp_dir   = os.path.join(args.output_dir, "temp", lang_tag)
os.makedirs(temp_dir, exist_ok=True)

SHARD_CSV  = os.path.join(temp_dir,
             f"shard_{args.shard_id:03d}_of_{args.num_shards:03d}.csv")
MERGED_CSV = os.path.join(args.output_dir,
             f"translations_{lang_tag}_{model_tag}.csv")

# ============================================================
# Load model locally (once per process)
# ============================================================

print(f"Loading tokenizer: {args.model} ...")
tokenizer = AutoTokenizer.from_pretrained(args.model)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

print(f"Loading model: {args.model} ...")
model = AutoModelForCausalLM.from_pretrained(
    args.model,
    dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="eager",
    low_cpu_mem_usage=True,
)
model.eval()
first_device = next(model.parameters()).device
print(f"Model loaded on {first_device}.\n")

# ============================================================
# Translation
# ============================================================

def build_prompt(sentences, target_lang):
    prompt  = f"Translate the following sentences from English to {target_lang}.\n"
    prompt += "Output exactly one translated sentence per line, keeping the same order.\n"
    prompt += "Do not add numbering, explanations, or any extra text.\n\n"
    for s in sentences:
        prompt += f"{s}\n"
    return prompt


def clean_output(text, n_expected):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    cleaned = []
    for line in lines:
        # strip any numbering the model may have added
        if len(line) > 2 and line[0].isdigit() and line[1] in ".):-":
            line = line.split(None, 1)[-1].strip()
        cleaned.append(line)
    while len(cleaned) < n_expected:
        cleaned.append("")
    return cleaned[:n_expected]


def generate_grouped(chat_prompts):
    """
    Runs model.generate() ONCE on a whole list of chat-formatted prompts,
    batched together via left-padding (tokenizer.padding_side is set to
    "left" above -- required for this to work correctly with a causal
    LM: real content stays right-aligned, so every row's newly generated
    tokens start at the SAME column index after generation, regardless
    of that row's original prompt length before padding).

    Returns: list[str], one raw decoded output per input prompt, same
    order and length as chat_prompts.
    """
    inputs = tokenizer(chat_prompts, return_tensors="pt", truncation=True,
                        max_length=2048, padding=True).to(first_device)
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = out_ids[:, prompt_len:]
    return [tokenizer.decode(row, skip_special_tokens=True).strip() for row in gen_ids]


def translate_batches_grouped(sentence_batches, batch_ids, target_lang):
    """
    sentence_batches: list of lists, each an up-to-10-sentence prompt's
        worth of sentences (same granularity as translate.py's BATCH_SIZE)
    batch_ids: matching labels for logging only

    Builds one chat-formatted prompt per sentence_batch, generates for
    ALL of them in a single model.generate() call, cleans each result
    against its own expected sentence count.

    Returns: list of translation-lists, same length/order as sentence_batches.
    """
    chat_prompts = []
    for sentences in sentence_batches:
        user_prompt = build_prompt(sentences, target_lang)
        messages = [{"role": "user", "content": user_prompt}]
        # each model's chat template is different -- this is the step the
        # CSCS API used to do server-side, now explicit and per-prompt
        chat_prompts.append(
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"  -> [EN->{target_lang}] Batches {batch_ids[0]}-{batch_ids[-1]} "
                  f"({len(sentence_batches)} prompts grouped), attempt {attempt}")
            raw_outputs = generate_grouped(chat_prompts)
            return [clean_output(text, len(sentences))
                    for text, sentences in zip(raw_outputs, sentence_batches)]
        except Exception as e:
            print(f"  Group {batch_ids[0]}-{batch_ids[-1]} failed: {e}")
        time.sleep(SLEEP_BETWEEN_RETRIES)

    print(f"  Group {batch_ids[0]}-{batch_ids[-1]} gave up — returning empty strings")
    return [[""] * len(s) for s in sentence_batches]

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
# Translation loop — prompts grouped for batched generation,
# empty-row retries also grouped instead of one row at a time
# ============================================================

# chunk rows into sentence-batches of BATCH_SIZE (one prompt each) --
# same granularity as translate.py
sentence_batches = []
batch_rows = []
for i in range(0, len(rows_to_process), BATCH_SIZE):
    batch = rows_to_process.iloc[i:i + BATCH_SIZE]
    sentence_batches.append(batch["sentence"].astype(str).tolist())
    batch_rows.append(list(batch.itertuples(index=False)))

total_prompts = len(sentence_batches)
print(f"{total_prompts} prompts total, grouped {args.prompt_batch_size} at a time "
      f"per model.generate() call")

results = []
save_threshold = BATCH_SIZE * 20  # unchanged save-interval, now measured in sentences

for g_start in tqdm(range(0, total_prompts, args.prompt_batch_size),
                     total=(total_prompts + args.prompt_batch_size - 1) // args.prompt_batch_size,
                     desc=f"{args.target_lang} shard {args.shard_id}",
                     unit="group"):
    g_end = min(g_start + args.prompt_batch_size, total_prompts)
    group_sentence_batches = sentence_batches[g_start:g_end]
    group_batch_ids = list(range(g_start, g_end))
    group_rows = batch_rows[g_start:g_end]

    group_translations = translate_batches_grouped(
        group_sentence_batches, group_batch_ids, args.target_lang
    )

    # collect empty rows across this whole group for a batched retry,
    # instead of retrying one row at a time
    empty_rows = []       # row objects
    empty_positions = []  # (prompt_idx_in_group, sentence_idx_in_prompt)
    for p_idx, (rows_in_prompt, translations_in_prompt) in enumerate(
            zip(group_rows, group_translations)):
        for s_idx, (row, translation) in enumerate(zip(rows_in_prompt, translations_in_prompt)):
            if not translation or not translation.strip():
                empty_rows.append(row)
                empty_positions.append((p_idx, s_idx))

    if empty_rows:
        print(f"  {len(empty_rows)} empty translation(s) in this group — "
              f"retrying in batch...")
        still_empty = empty_rows
        for retry in range(1, MAX_ROW_RETRIES + 1):
            if not still_empty:
                break
            retry_batches = [[str(r.sentence)] for r in still_empty]
            retry_ids = list(range(len(retry_batches)))
            retry_results = translate_batches_grouped(retry_batches, retry_ids, args.target_lang)

            next_still_empty = []
            for row, result in zip(still_empty, retry_results):
                translation = result[0] if result else ""
                if translation and translation.strip():
                    # write the recovered translation back into the group's results
                    for p_idx, s_idx in [pos for pos, r in zip(empty_positions, empty_rows) if r.row_id == row.row_id]:
                        group_translations[p_idx][s_idx] = translation
                    print(f"  Row {row.row_id} succeeded on retry {retry}")
                else:
                    next_still_empty.append(row)
            still_empty = next_still_empty
            if still_empty:
                time.sleep(SLEEP_BETWEEN_RETRIES)
        for row in still_empty:
            print(f"  Row {row.row_id} gave up after {MAX_ROW_RETRIES} retries")

    # flatten this group's results into the output rows
    for rows_in_prompt, translations_in_prompt in zip(group_rows, group_translations):
        for row, translation in zip(rows_in_prompt, translations_in_prompt):
            results.append({
                "row_id":      row.row_id,
                "path":        row.path,
                "source":      row.sentence,
                "target_lang": args.target_lang,
                "translation": translation,
                "model":       args.model,
            })

    if len(results) >= save_threshold:
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

        n_before = len(merged)
        merged   = merged.drop_duplicates(subset="row_id", keep="last")
        n_dupes  = n_before - len(merged)
        if n_dupes:
            print(f"WARNING: dropped {n_dupes} duplicate row_ids")

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