#!/usr/bin/env python3

"""
translate_score_extract_llama.py

Pipeline:
    1. Translate N sentences with Llama 3.1 8B Instruct
    2. Score all translations with COMET-Kiwi
    3. Select the top N_TOP by COMET-Kiwi
    4. Extract mean hidden states only for those top sentences
    5. Save translations, scores, selected sentences, and hidden states

Hidden-state output:
    (N_TOP, 33, 4096)
    32 transformer layers + embedding output

Designed to minimize compute:
    - Batched generation
    - Batched COMET scoring
    - Hidden states ONLY for top N_TOP
    - No gradients
    - BF16 model
    - Mean-pool token representations immediately
    - Save hidden states as float16
"""

import os
import gc
import argparse

import numpy as np
import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from comet import download_model, load_from_checkpoint


# ============================================================
# Environment
# ============================================================

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if "SCRATCH" in os.environ:
    os.environ["HF_HOME"] = os.path.join(
        os.environ["SCRATCH"],
        "hf_cache",
    )


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

# Generation
GEN_BATCH_SIZE = 16
MAX_INPUT_LENGTH = 256
MAX_NEW_TOKENS = 128

# COMET
COMET_BATCH_SIZE = 32

# Hidden states
HIDDEN_BATCH_SIZE = 8

# Dataset
N_TOTAL = 5000
N_TOP = 2000

SEED = 42


# ============================================================
# Language prompts
# ============================================================

LANGUAGE_NAMES = {
    "es": "Spanish",
    "ca": "Catalan",
    "gl": "Galician",
    "de": "German",
    "nl": "Dutch",
    "af": "Afrikaans",
}


def make_messages(source_sentence, target_lang):
    target_name = LANGUAGE_NAMES[target_lang]

    return [
        {
            "role": "user",
            "content": (
                f"Translate the following sentence into {target_name}. "
                f"Return only the translation, with no explanation.\n\n"
                f"Sentence: {source_sentence}\n"
                f"Translation:"
            ),
        }
    ]


# ============================================================
# Arguments
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--input_csv",
    type=str,
    required=True,
)

parser.add_argument(
    "--output_dir",
    type=str,
    required=True,
)

parser.add_argument(
    "--target_lang",
    type=str,
    required=True,
    choices=list(LANGUAGE_NAMES.keys()),
)

parser.add_argument(
    "--source_col",
    type=str,
    default="sentence",
)

parser.add_argument(
    "--n",
    type=int,
    default=N_TOTAL,
)

parser.add_argument(
    "--top_n",
    type=int,
    default=N_TOP,
)

parser.add_argument(
    "--seed",
    type=int,
    default=SEED,
)

args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

torch.manual_seed(args.seed)
np.random.seed(args.seed)


# ============================================================
# Device
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU required.")

device = torch.device("cuda")


print("=" * 70)
print("LLAMA 3.1 8B TRANSLATION / SCORING / HIDDEN-STATE PIPELINE")
print("=" * 70)
print(f"Model:          {MODEL_NAME}")
print(f"Input:          {args.input_csv}")
print(f"Target:         {args.target_lang}")
print(f"Total:          {args.n}")
print(f"Top:            {args.top_n}")
print(f"Output:         {args.output_dir}")
print(f"CUDA device:    {torch.cuda.get_device_name(0)}")
print("=" * 70)


# ============================================================
# Load input
# ============================================================

df = pd.read_csv(args.input_csv)

if args.source_col not in df.columns:
    raise ValueError(
        f"Missing source column: {args.source_col}"
    )

df = df.copy()
df = df.head(args.n).reset_index(drop=True)
df["row_id"] = np.arange(len(df))

print(f"\nLoaded {len(df)} sentences.")


# ============================================================
# Load Llama 3.1 8B
# ============================================================

print("\nLoading Llama 3.1 8B tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "left"

print("Loading Llama 3.1 8B model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
).eval()

print("Llama 3.1 8B loaded.")


# ============================================================
# Batched generation
# ============================================================

def generate_translations(sentences, target_lang):

    translations = []
    total = len(sentences)

    for start in range(0, total, GEN_BATCH_SIZE):

        end = min(
            start + GEN_BATCH_SIZE,
            total,
        )

        batch_sentences = sentences[start:end]

        conversations = [
            make_messages(
                sentence,
                target_lang,
            )
            for sentence in batch_sentences
        ]

        inputs = tokenizer.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_INPUT_LENGTH,
        )

        inputs = {
            k: v.to(device)
            for k, v in inputs.items()
        }

        with torch.inference_mode():

            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Because padding is LEFT padding, the prompt length is
        # the same for the whole batch after padding.
        input_length = inputs["input_ids"].shape[-1]

        generated_ids = outputs[:, input_length:]

        decoded = tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )

        decoded = [
            text.strip()
            for text in decoded
        ]

        translations.extend(decoded)

        print(
            f"Translation: {end}/{total}",
            flush=True,
        )

        del inputs
        del outputs
        del generated_ids

        torch.cuda.empty_cache()

    return translations


# ============================================================
# Translation
# ============================================================

translation_path = os.path.join(
    args.output_dir,
    f"translations_{args.target_lang}_llama-3.1-8b.csv",
)

if os.path.exists(translation_path):

    print(
        f"\nExisting translations found: "
        f"{translation_path}"
    )

    df = pd.read_csv(
        translation_path
    )

else:

    print("\nGenerating translations...")

    df["translation"] = generate_translations(
        df[args.source_col].astype(str).tolist(),
        args.target_lang,
    )

    df.to_csv(
        translation_path,
        index=False,
    )

    print(
        f"Saved translations → "
        f"{translation_path}"
    )


# ============================================================
# Load COMET-Kiwi
# ============================================================

print("\nLoading COMET-Kiwi...")

comet_path = download_model(
    "Unbabel/wmt22-cometkiwi-da"
)

comet_model = load_from_checkpoint(
    comet_path
)

print("COMET-Kiwi loaded.")


# ============================================================
# COMET scoring
# ============================================================

scored_path = os.path.join(
    args.output_dir,
    f"translations_{args.target_lang}_llama-3.1-8b_scored.csv",
)

if os.path.exists(scored_path):

    print(
        f"\nExisting COMET scores found: "
        f"{scored_path}"
    )

    df = pd.read_csv(
        scored_path
    )

else:

    print("\nScoring translations...")

    comet_data = [
        {
            "src": str(row[args.source_col]),
            "mt": str(row["translation"]),
        }
        for _, row in df.iterrows()
    ]

    scores = []

    for start in range(
        0,
        len(comet_data),
        COMET_BATCH_SIZE,
    ):

        end = min(
            start + COMET_BATCH_SIZE,
            len(comet_data),
        )

        batch = comet_data[start:end]

        result = comet_model.predict(
            batch,
            batch_size=COMET_BATCH_SIZE,
            gpus=1,
        )

        scores.extend(
            result.scores
        )

        print(
            f"COMET: {end}/{len(comet_data)}",
            flush=True,
        )

    df["comet_kiwi"] = scores

    df.to_csv(
        scored_path,
        index=False,
    )

    print(
        f"Saved scored translations → "
        f"{scored_path}"
    )


# ============================================================
# Select top N_TOP
# ============================================================

print("\nSelecting highest-quality translations...")

df = df.sort_values(
    "comet_kiwi",
    ascending=False,
).reset_index(drop=True)

top_n = min(
    args.top_n,
    len(df),
)

top_df = df.head(top_n).copy()

top_path = os.path.join(
    args.output_dir,
    f"top{top_n}_{args.target_lang}_llama-3.1-8b.csv",
)

top_df.to_csv(
    top_path,
    index=False,
)

print(
    f"Selected top {top_n}."
)

print(
    f"COMET range: "
    f"{top_df['comet_kiwi'].min():.4f} – "
    f"{top_df['comet_kiwi'].max():.4f}"
)

print(
    f"Saved → {top_path}"
)


# ============================================================
# Free COMET model before hidden-state extraction
# ============================================================

del comet_model

gc.collect()

torch.cuda.empty_cache()


# ============================================================
# Hidden-state extraction
# ============================================================

print("\nExtracting hidden states for top sentences...")

hidden_path = os.path.join(
    args.output_dir,
    f"top{top_n}_{args.target_lang}_hidden_states.npz",
)

sentences = top_df[
    args.source_col
].astype(str).tolist()


def extract_hidden_states(sentences):

    all_hidden = []
    total = len(sentences)

    for start in range(
        0,
        total,
        HIDDEN_BATCH_SIZE,
    ):

        end = min(
            start + HIDDEN_BATCH_SIZE,
            total,
        )

        batch_sentences = sentences[start:end]

        conversations = [
            make_messages(
                sentence,
                args.target_lang,
            )
            for sentence in batch_sentences
        ]

        inputs = tokenizer.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_INPUT_LENGTH,
        )

        inputs = {
            k: v.to(device)
            for k, v in inputs.items()
        }

        attention_mask = inputs["attention_mask"]

        with torch.inference_mode():

            outputs = model(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
            )

        # ----------------------------------------------------
        # Mean pool over NON-PADDING tokens
        #
        # Shape:
        #     batch × layers × hidden_dim
        #
        # Llama 3.1 8B:
        #     33 × 4096
        # ----------------------------------------------------

        mask = attention_mask.float()

        hidden_layers = []

        for hidden in outputs.hidden_states:

            weighted = (
                hidden
                * mask.unsqueeze(-1)
            )

            pooled = (
                weighted.sum(dim=1)
                / mask.sum(
                    dim=1,
                    keepdim=True,
                )
            )

            hidden_layers.append(
                pooled.float().cpu().numpy()
            )

        batch_hidden = np.stack(
            hidden_layers,
            axis=1,
        )

        all_hidden.append(
            batch_hidden.astype(
                np.float16
            )
        )

        print(
            f"Hidden states: {end}/{total}",
            flush=True,
        )

        del inputs
        del outputs
        del hidden_layers
        del batch_hidden

        torch.cuda.empty_cache()

    return np.concatenate(
        all_hidden,
        axis=0,
    )


hidden_states = extract_hidden_states(
    sentences
)

print(
    f"\nHidden-state shape: "
    f"{hidden_states.shape}"
)


# ============================================================
# Save hidden states
# ============================================================

np.savez_compressed(
    hidden_path,
    hidden_states=hidden_states,
    row_id=top_df[
        "row_id"
    ].to_numpy(),
    comet_kiwi=top_df[
        "comet_kiwi"
    ].to_numpy(),
)

print(
    f"Saved hidden states → "
    f"{hidden_path}"
)


# ============================================================
# Final
# ============================================================

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

print(
    f"Translations: {translation_path}"
)

print(
    f"Scored:       {scored_path}"
)

print(
    f"Top {top_n}:   {top_path}"
)

print(
    f"Hidden states: {hidden_path}"
)

print(
    f"Hidden shape:  {hidden_states.shape}"
)