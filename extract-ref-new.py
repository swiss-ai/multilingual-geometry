"""
extract_references.py — Extract contrastive language pair vectors.

For each language, uses the SAME English sentences with two different
prompt languages to isolate pure language identity signal:
  - en_ref:     English sentences + English continuation prompt
  - target_ref: English sentences + target language continuation prompt

Usage:
    python extract_references.py \
        --input_dir translations/apertus/ \
        --min_comet 0.89 \
        --n_sentences 200
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HOME"] = os.path.join(os.environ.get("SCRATCH", os.path.expanduser("~")), "hf_cache")

import argparse
import json
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

LANGUAGES = {
    "es": {"name": "Spanish",   "csv": "translations_spanish_{model}.csv"},
    "ca": {"name": "Catalan",   "csv": "translations_catalan_{model}.csv"},
    "gl": {"name": "Galician",  "csv": "translations_galician_{model}.csv"},
    "de": {"name": "German",    "csv": "translations_german_{model}.csv"},
    "nl": {"name": "Dutch",     "csv": "translations_dutch_{model}.csv"},
    "af": {"name": "Afrikaans", "csv": "translations_afrikaans_{model}.csv"},
}

NEUTRAL_PROMPTS = {
    "de": "Vervollständige den folgenden Satz: {sentence}",
    "nl": "Maak de volgende zin af: {sentence}",
    "af": "Voltooi die volgende sin: {sentence}",
    "es": "Continúa la siguiente oración: {sentence}",
    "ca": "Continua la frase següent: {sentence}",
    "gl": "Continúa a seguinte oración: {sentence}",
    "en": "Continue the following sentence: {sentence}",
}

MODEL_SLUG = "apertus-8b-instruct-2509"

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir",   type=str,   required=True)
parser.add_argument("--min_comet",   type=float, default=0.89)
parser.add_argument("--n_sentences", type=int,   default=200)
parser.add_argument("--seed",        type=int,   default=42)
parser.add_argument("--output_dir",  type=str,   default="references_new")
parser.add_argument("--model",       type=str,
                    default="swiss-ai/Apertus-8B-Instruct-2509")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

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
# Helpers
# ============================================================

def load_sentences(csv_path, min_comet, n_sentences, seed):
    df       = pd.read_csv(csv_path)
    df_good  = df[df["comet_kiwi"] >= min_comet].reset_index(drop=True)

    print(f"  Quality filter: {len(df_good)}/{len(df)} sentences "
          f"pass comet_kiwi >= {min_comet}")

    if len(df_good) == 0:
        raise ValueError(f"No sentences pass quality threshold in {csv_path}")

    n         = min(n_sentences, len(df_good))
    df_sample = df_good.sample(n=n, random_state=seed)

    print(f"  Sampled {n} sentences "
          f"(comet range: {df_sample['comet_kiwi'].min():.4f} – "
          f"{df_sample['comet_kiwi'].max():.4f})")

    return df_sample["source"].tolist(), df_sample["translation"].tolist()


def extract_mean_hidden(sentences, prompt_template, desc, lang_code):
    """
    Extract mean-pooled hidden states.
    Prints the full prompt sent to the model for each sentence.
    """
    all_hidden = []

    for i, sentence in enumerate(sentences):
        prompt = prompt_template.format(sentence=sentence)

        print(f"\n  [{desc}] sentence {i+1}/{len(sentences)}")
        print(f"    prompt ({lang_code}): {prompt[:120]}")

        inputs = tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=256
        )
        inputs = {k: v.to(first_device) for k, v in inputs.items()}

        with torch.no_grad():
            torch.cuda.empty_cache()

            # generate one token so we can see what the model produces
            gen = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                use_cache=True,
            )
            generated = tokenizer.decode(
                gen[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            print(f"    model output:    {generated[:80]}")

            # now extract hidden states from forward pass
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
        all_hidden.append(hidden)

    ref   = np.stack(all_hidden, axis=0).mean(axis=0).astype(np.float32)
    norms = np.linalg.norm(ref, axis=1, keepdims=True) + 1e-8
    return ref / norms

# ============================================================
# English reference — extracted once from English sentences
# ============================================================

first_csv = os.path.join(
    args.input_dir,
    list(LANGUAGES.values())[0]["csv"].format(model=MODEL_SLUG)
)
en_sentences, _ = load_sentences(
    first_csv, args.min_comet, args.n_sentences, args.seed
)

print(f"\nExtracting English reference...")
print(f"  Prompt template: \"{NEUTRAL_PROMPTS['en'].format(sentence='...')}\"")
en_ref = extract_mean_hidden(
    en_sentences, NEUTRAL_PROMPTS["en"], desc="English", lang_code="en"
)
print(f"\nEnglish ref shape: {en_ref.shape}\n")

# ============================================================
# Per language: same English sentences, target language prompt
# ============================================================

reference_pairs  = {}
cosine_sims      = {}
contrastive_vecs = {}

for code, info in LANGUAGES.items():
    lang_name = info["name"]

    print(f"\n{'='*60}")
    print(f"Processing {lang_name} ({code})")
    print(f"  Prompt template: \"{NEUTRAL_PROMPTS[code].format(sentence='...')}\"")
    print(f"  Using same {len(en_sentences)} English sentences")
    print(f"{'='*60}")

    # same English sentences, target language prompt
    target_ref = extract_mean_hidden(
        en_sentences,
        NEUTRAL_PROMPTS[code],
        desc=lang_name,
        lang_code=code,
    )

    # Step 1 — cosine similarity between raw references per layer
    ref_sim = [
        float(np.dot(en_ref[i], target_ref[i]))
        for i in range(en_ref.shape[0])
    ]
    cosine_sims[code] = ref_sim

    # mask L0 and L32 for best layer selection
    masked = [s if 0 < i < 32 else 1.0 for i, s in enumerate(ref_sim)]
    best_layer = int(np.argmin(masked))

    print(f"\n  Cosine sim (en vs {lang_name}) by layer:")
    print(f"    Early  (L0–4):   {np.mean(ref_sim[:5]):.4f}")
    print(f"    Middle (L12–16): {np.mean(ref_sim[12:17]):.4f}")
    print(f"    Late   (L28–32): {np.mean(ref_sim[28:]):.4f}")
    print(f"    Best layer (ignoring L0/L32): L{best_layer} "
          f"({ref_sim[best_layer]:.4f})\n")

    # Step 2 — contrastive vector
    v_lang = target_ref - en_ref
    norms  = np.linalg.norm(v_lang, axis=1, keepdims=True) + 1e-8
    v_lang = v_lang / norms

    contrastive_vecs[f"v_{code}"] = v_lang
    reference_pairs[code] = {
        "language":            lang_name,
        "en_ref":              en_ref.tolist(),
        "target_ref":          target_ref.tolist(),
        "cosine_sim_by_layer": ref_sim,
        "best_layer":          best_layer,
        "n_sentences":         args.n_sentences,
        "min_comet":           args.min_comet,
        "seed":                args.seed,
    }

# ============================================================
# Save JSON pairs
# ============================================================

json_path = os.path.join(args.output_dir, "reference_pairs.json")
with open(json_path, "w") as f:
    json.dump(reference_pairs, f)
print(f"\nSaved reference pairs -> {json_path}")

# ============================================================
# Save contrastive vectors (npz — too large for JSON)
# ============================================================

npz_path = os.path.join(args.output_dir, "contrastive_vectors.npz")
np.savez(
    npz_path,
    en_ref=en_ref,
    **contrastive_vecs,
    **{f"cosine_sim_{code}": np.array(sims)
       for code, sims in cosine_sims.items()},
    language_codes=np.array(list(LANGUAGES.keys())),
    language_names=np.array([v["name"] for v in LANGUAGES.values()]),
)
print(f"Saved contrastive vectors -> {npz_path}")
for code in LANGUAGES:
    key = f"v_{code}"
    if key in contrastive_vecs:
        print(f"  {key}: {contrastive_vecs[key].shape}")