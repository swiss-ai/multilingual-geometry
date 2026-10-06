"""
test_lang_detection.py — Pivot window detection via hidden state geometry.

At each layer, computes cosine similarity between the generated token's
hidden state and reference vectors for English (source) and the target
language. The pivot window is the span of layers where English similarity
is falling but target similarity has not yet peaked.

After processing all sentences, produces two side-by-side heatmaps:
  Left:  English cosine similarity per layer per sentence (darker = stronger)
  Right: Target cosine similarity per layer per sentence (darker = stronger)
Sentences are sorted by COMET-Kiwi score so Q1 is on the left, Q4 on the right.

Usage:
    python test_lang_detection.py --lang german --target_code de
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# ============================================================
# Args
# ============================================================

LANG_CODE_MAP = {
    "german":    "de",
    "dutch":     "nl",
    "afrikaans": "af",
    "spanish":   "es",
    "catalan":   "ca",
    "galician":  "gl",
}

parser = argparse.ArgumentParser()
parser.add_argument("--lang",           type=str, default="german")
parser.add_argument("--target_code",    type=str, default=None)
parser.add_argument("--n_q1",          type=int, default=25,
                    help="Number of Q1 sentences to process")
parser.add_argument("--n_q4",          type=int, default=25,
                    help="Number of Q4 sentences to process")
parser.add_argument("--max_new_tokens", type=int, default=40)
parser.add_argument("--references_dir", type=str, default="references")
parser.add_argument("--translations_dir", type=str,
                    default="translations/apertus")
parser.add_argument("--output_dir",     type=str, default="pivot_results")
parser.add_argument("--model",          type=str,
                    default="swiss-ai/Apertus-8B-Instruct-2509")
args = parser.parse_args()

TARGET_LANG = args.lang.capitalize()
TARGET_CODE = args.target_code or LANG_CODE_MAP.get(args.lang, args.lang[:2])
MODEL_TAG   = "apertus-8b-instruct-2509"
INPUT_CSV   = os.path.join(
    args.translations_dir,
    f"translations_{args.lang}_{MODEL_TAG}.csv"
)
REFERENCES_PATH = os.path.join(args.references_dir, f"references_{TARGET_CODE}.npz")
OUTPUT_DIR      = os.path.join(args.output_dir, args.lang)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Load sentences
# ============================================================

print(f"Loading sentences from {INPUT_CSV}...")
df  = pd.read_csv(INPUT_CSV)
q25 = df["comet_kiwi"].quantile(0.25)
q75 = df["comet_kiwi"].quantile(0.75)

q1_df = df[df["comet_kiwi"] <= q25].head(args.n_q1)
q4_df = df[df["comet_kiwi"] >= q75].head(args.n_q4)

SENTENCES = q1_df["source"].tolist() + q4_df["source"].tolist()
LABELS    = ["Q1"] * len(q1_df) + ["Q4"] * len(q4_df)
SCORES    = q1_df["comet_kiwi"].tolist() + q4_df["comet_kiwi"].tolist()

print(f"Q1: {len(q1_df)} sentences (comet <= {q25:.4f})")
print(f"Q4: {len(q4_df)} sentences (comet >= {q75:.4f})")

# ============================================================
# Load reference vectors
# ============================================================

print(f"\nLoading reference vectors from {REFERENCES_PATH}...")
refs       = np.load(REFERENCES_PATH)
en_ref     = refs["en_ref"].astype(np.float32)
target_ref = refs["target_ref"].astype(np.float32)
print(f"Reference shapes: en={en_ref.shape}, target={target_ref.shape}")

# ============================================================
# Load model
# ============================================================

print("\nLoading tokenizer...")
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

def cosine_sim(a, b):
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))


def run_sentence(sentence, idx, label, comet_score):
    print(f"\n[{idx+1}/{len(SENTENCES)}] [{label}] comet={comet_score:.4f}: \"{sentence[:60]}\"")

    messages = [{"role": "user", "content":
                 f"Translate the following sentence from English to {TARGET_LANG}.\n\n{sentence}"}]
    prompt   = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(first_device) for k, v in inputs.items()}

    generated_ids    = inputs["input_ids"].clone()
    all_en_sim       = []
    all_target_sim   = []
    all_token_labels = []

    for step in range(args.max_new_tokens):
        with torch.no_grad():
            torch.cuda.empty_cache()
            outputs = model(
                input_ids=generated_ids,
                output_hidden_states=True,
                use_cache=False,
            )

        hidden_states  = outputs.hidden_states
        n_layers       = len(hidden_states)
        next_token_id  = outputs.logits[0, -1, :].argmax().item()
        next_token_str = tokenizer.decode([next_token_id]).strip()

        if next_token_id == tokenizer.eos_token_id or \
                next_token_str in {"<|assistant_end|>", ""}:
            break

        step_en, step_tgt = [], []
        for i, layer in enumerate(hidden_states):
            hs = layer[0, -1, :].detach().float().cpu().numpy()
            step_en.append(cosine_sim(hs, en_ref[i]))
            step_tgt.append(cosine_sim(hs, target_ref[i]))

        all_en_sim.append(step_en)
        all_target_sim.append(step_tgt)
        all_token_labels.append(next_token_str)

        next_tensor   = torch.tensor([[next_token_id]], device=generated_ids.device)
        generated_ids = torch.cat([generated_ids, next_tensor], dim=1)

    translation = tokenizer.decode(
        generated_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()

    # mean across steps -> (n_layers,)
    en_curve     = np.mean(all_en_sim,     axis=0)
    target_curve = np.mean(all_target_sim, axis=0)

    # pivot window
    threshold          = 0.5
    en_fade_layers     = np.where(en_curve < en_curve.max() * threshold)[0]
    target_rise_layers = np.where(target_curve > target_curve.max() * threshold)[0]
    source_faded  = int(en_fade_layers[0])     if len(en_fade_layers)     > 0 else n_layers - 1
    target_peaked = int(target_rise_layers[0]) if len(target_rise_layers) > 0 else n_layers - 1
    pivot_width   = max(0, target_peaked - source_faded)

    print(f"  pivot: L{source_faded}–L{target_peaked} (w={pivot_width})  "
          f"translation: {translation[:60]}")

    result = {
        "sentence":      sentence,
        "translation":   translation,
        "label":         label,
        "comet_kiwi":    comet_score,
        "en_sim":        en_curve.tolist(),
        "target_sim":    target_curve.tolist(),
        "source_faded":  source_faded,
        "target_peaked": target_peaked,
        "pivot_width":   pivot_width,
        "n_layers":      n_layers,
        "target_lang":   TARGET_LANG,
        "target_code":   TARGET_CODE,
    }
    out_path = os.path.join(OUTPUT_DIR, f"sent_{idx+1:03d}_{label}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


# ============================================================
# Run all sentences
# ============================================================

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

results = []
for idx, (sentence, label, score) in enumerate(zip(SENTENCES, LABELS, SCORES)):
    r = run_sentence(sentence, idx, label, score)
    results.append(r)
    torch.cuda.empty_cache()

# ============================================================
# Heatmap visualisation
# ============================================================

# Sort all results by COMET score (Q1 left → Q4 right)
results_sorted = sorted(results, key=lambda x: x["comet_kiwi"])
n_sentences    = len(results_sorted)
n_layers       = results_sorted[0]["n_layers"]

# Build matrices: rows=layers, cols=sentences (sorted by COMET)
en_matrix     = np.zeros((n_layers, n_sentences))
target_matrix = np.zeros((n_layers, n_sentences))

for col, r in enumerate(results_sorted):
    en_matrix[:, col]     = r["en_sim"]
    target_matrix[:, col] = r["target_sim"]

comet_scores  = [r["comet_kiwi"] for r in results_sorted]
pivot_widths  = [r["pivot_width"] for r in results_sorted]
labels_sorted = [r["label"] for r in results_sorted]

BG = "#F7F7F5"
fig, axes = plt.subplots(
    1, 2, figsize=(14, 8), facecolor=BG,
    gridspec_kw={"wspace": 0.08}
)
fig.suptitle(
    f"Pivot Window — English → {TARGET_LANG}\n"
    f"Cosine similarity to reference vectors  "
    f"(sentences sorted by COMET-Kiwi, low → high)",
    fontsize=11, y=1.01
)

vmin_en  = en_matrix.min()
vmax_en  = en_matrix.max()
vmin_tgt = target_matrix.min()
vmax_tgt = target_matrix.max()

# Left: English similarity — dark blue = strong English signal
im0 = axes[0].imshow(
    en_matrix, aspect="auto", origin="upper",
    cmap="Blues", vmin=vmin_en, vmax=vmax_en,
    interpolation="nearest"
)
axes[0].set_title("English similarity", fontsize=10)
axes[0].set_ylabel("Transformer layer")
axes[0].set_xlabel("Sentence (sorted by COMET ↑)")
axes[0].set_yticks(range(0, n_layers, 4))
axes[0].set_yticklabels([f"L{i}" for i in range(0, n_layers, 4)], fontsize=7)
axes[0].set_xticks([])
plt.colorbar(im0, ax=axes[0], shrink=0.6, pad=0.02)

# Right: Target similarity — dark green = strong target signal
im1 = axes[1].imshow(
    target_matrix, aspect="auto", origin="upper",
    cmap="Greens", vmin=vmin_tgt, vmax=vmax_tgt,
    interpolation="nearest"
)
axes[1].set_title(f"{TARGET_LANG} similarity", fontsize=10)
axes[1].set_xlabel("Sentence (sorted by COMET ↑)")
axes[1].set_yticks(range(0, n_layers, 4))
axes[1].set_yticklabels([f"L{i}" for i in range(0, n_layers, 4)], fontsize=7)
axes[1].set_xticks([])
plt.colorbar(im1, ax=axes[1], shrink=0.6, pad=0.02)

# Mark Q1/Q4 boundary on both axes
n_q1 = sum(1 for r in results_sorted if r["label"] == "Q1")
for ax in axes:
    ax.axvline(n_q1 - 0.5, color="#E8A838", linewidth=1.5, linestyle="--")
    ax.text(n_q1 * 0.5, n_layers + 0.5, "Q1", ha="center",
            fontsize=8, color="#E8A838")
    ax.text(n_q1 + (n_sentences - n_q1) * 0.5, n_layers + 0.5, "Q4",
            ha="center", fontsize=8, color="#E8A838")

plt.tight_layout()
fig_path = os.path.join(OUTPUT_DIR, f"heatmap_{args.lang}.png")
plt.savefig(fig_path, dpi=200, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"\nHeatmap saved -> {fig_path}")

# ============================================================
# Summary
# ============================================================

q1_results = [r for r in results if r["label"] == "Q1"]
q4_results = [r for r in results if r["label"] == "Q4"]

q1_pw = np.mean([r["pivot_width"] for r in q1_results])
q4_pw = np.mean([r["pivot_width"] for r in q4_results])

print(f"\n{'='*50}")
print(f"Language: {TARGET_LANG}")
print(f"Q1 mean pivot width: {q1_pw:.2f} layers")
print(f"Q4 mean pivot width: {q4_pw:.2f} layers")
print(f"Difference:          {q1_pw - q4_pw:.2f} layers")

summary = {
    "language":       TARGET_LANG,
    "target_code":    TARGET_CODE,
    "n_q1":           len(q1_results),
    "n_q4":           len(q4_results),
    "q1_mean_pivot":  float(q1_pw),
    "q4_mean_pivot":  float(q4_pw),
    "pivot_diff":     float(q1_pw - q4_pw),
    "q1_pivot_widths": [r["pivot_width"] for r in q1_results],
    "q4_pivot_widths": [r["pivot_width"] for r in q4_results],
}
summary_path = os.path.join(OUTPUT_DIR, f"summary_{args.lang}.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Summary saved -> {summary_path}")

print("\nDone.")