"""
extract_hidden_states.py — Extract hidden states for extreme COMET groups.

Takes all 5000 sentences by COMET score and extracts
translation hidden states using the actual translation prompt.

Usage:
    python extract-hs.py \
        --input_csv translations/apertus/translations_spanish_apertus-8b-instruct-2509.csv \
        --output_dir hidden_states/spanish-apertus \
        --target_lang Spanish
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HOME"] = os.path.join(os.environ.get("SCRATCH", os.path.expanduser("~")), "hf_cache")

import argparse
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument("--input_csv",   type=str, required=True)
parser.add_argument("--output_dir",  type=str, default="hidden_states/spanish-apertus")
parser.add_argument("--target_lang", type=str, default="Spanish")
parser.add_argument("--n",           type=int, default=5000)
parser.add_argument("--seed",        type=int, default=42)
parser.add_argument("--array_id",    type=int, default=0)
parser.add_argument("--n_arrays",    type=int, default=1)
parser.add_argument("--model",       type=str,
                    default="swiss-ai/Apertus-8B-Instruct-2509")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

def make_prompt(source, target_lang):
    prompt  = f"Translate the following sentences from English to {target_lang}.\n"
    prompt += "Output exactly one translated sentence per line, keeping the same order.\n"
    prompt += "Do not add numbering, explanations, or any extra text.\n\n"
    prompt += source
    return prompt

# ============================================================
# Load top/bottom sentences
# ============================================================

df      = pd.read_csv(args.input_csv)
df_sort = df.sort_values("comet_kiwi").reset_index(drop=True)

bottom  = df_sort.head(args.n).copy()
top     = df_sort.tail(args.n).copy()
bottom["group"] = 0   # bad
top["group"]    = 1   # good

print(f"Bottom {args.n} (bad):")
print(f"  comet {bottom['comet_kiwi'].min():.4f}–"
      f"{bottom['comet_kiwi'].max():.4f} "
      f"| mean {bottom['comet_kiwi'].mean():.4f}")
print(f"Top {args.n} (good):")
print(f"  comet {top['comet_kiwi'].min():.4f}–"
      f"{top['comet_kiwi'].max():.4f} "
      f"| mean {top['comet_kiwi'].mean():.4f}")

sample = pd.concat([bottom, top]).reset_index(drop=True)
print(f"\nTotal: {len(sample)} sentences")

# ============================================================
# Slice this job's chunk
# ============================================================

total_rows = len(sample)
chunk_size = int(np.ceil(total_rows / args.n_arrays))
start      = args.array_id * chunk_size
end        = min(start + chunk_size, total_rows)
sample     = sample.iloc[start:end].reset_index(drop=True)

print(f"Array job {args.array_id}/{args.n_arrays}: "
      f"processing rows {start}–{end} ({len(sample)} sentences)\n")

# ============================================================
# Load model
# ============================================================

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(args.model)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    args.model,
    dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="eager",
    low_cpu_mem_usage=True,
)
model.eval()
torch.cuda.empty_cache()
first_device = next(model.parameters()).device
print("Model loaded.\n")

# ============================================================
# Extract hidden states
# ============================================================

def extract_hidden(text):
    inputs = tokenizer(
        text, return_tensors="pt",
        truncation=True, max_length=256
    )
    inputs = {k: v.to(first_device) for k, v in inputs.items()}

    with torch.no_grad():
        torch.cuda.empty_cache()
        outputs = model(
            input_ids=inputs["input_ids"],
            output_hidden_states=True,
            use_cache=False,
        )

    hidden = np.stack(
        [hs[0, :, :].mean(dim=0).detach().float().cpu().numpy()
         for hs in outputs.hidden_states],
        axis=0
    )  # (n_layers, hidden_dim)
    return hidden.astype(np.float32)

# ============================================================
# Run extraction
# ============================================================

total = len(sample)
for i, (_, row) in enumerate(sample.iterrows()):
    row_id      = int(row["row_id"])
    source      = str(row["source"])
    translation = str(row["translation"])
    comet       = float(row["comet_kiwi"])
    group       = int(row["group"])

    print(f"  [{i+1}/{total}] row={row_id:05d} | "
          f"{'good' if group == 1 else 'bad':4s} | comet={comet:.4f}")
    print(f"    source:      {source[:80]}")
    print(f"    translation: {translation[:80]}")

    prompt             = make_prompt(source, args.target_lang)
    prompt_hidden      = extract_hidden(prompt)
    translation_hidden = extract_hidden(translation)

    print(f"    prompt      hidden shape: {prompt_hidden.shape}  "
          f"norm@L16: {np.linalg.norm(prompt_hidden[16]):.2f}")
    print(f"    translation hidden shape: {translation_hidden.shape}  "
          f"norm@L16: {np.linalg.norm(translation_hidden[16]):.2f}")

    out_path = os.path.join(
        args.output_dir, f"row{row_id:05d}_group{group}.npz"
    )
    np.savez(
        out_path,
        prompt_hidden=prompt_hidden,
        translation_hidden=translation_hidden,
        source=np.array(source),
        translation=np.array(translation),
        comet_kiwi=np.array(comet),
        group=np.array(group),
        row_id=np.array(row_id),
    )
    print(f"    Saved -> {out_path}\n")

print("Done.")