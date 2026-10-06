"""
ablation.py — Surgically ablate the high-resource language direction
from low-resource hidden states during inference.

Uses top and bottom N sentences by COMET score.

Conditions:
  baseline:   no generation — uses stored translation directly (ceiling / reference baseline)
  none:       generate normally with NO hooks (true model baseline)
  all_layers: ablate v_H at all selected decoding layers
  early:      ablate v_H at layers 0–15
  late:       ablate v_H at layers 16–31
  pivot:      ablate v_H at detected pivot layers only
  probe_gap:  ablate v_H at explicitly specified layers (--layers required)

SURGICAL CHANGE:
  Instead of ablating the whole hidden state tensor at every token,
  we only ablate the LAST token representation:
      hidden[:, -1, :]
  i.e. the token state directly used for next-token prediction.

OPTIONAL GATING:
  Only ablate when |projection| > proj_threshold.

NOTE: v_axis has shape (33, 4096) — index 0 is the embedding layer output.
      Hook layer i fires after transformer layer i, corresponding to v_axis[i+1].

Usage:
    # Romance family (gl source)
    python ablation.py \
        --source_lang gl \
        --axis_lang   es \
        --condition   probe_gap \
        --layers      25 27 \
        --alpha       0.1 \
        --input_csv   translations/gemma/translations_galician_apertus-8b-instruct-2509.csv

    # Germanic family (af source)
    python ablation.py \
        --source_lang af \
        --axis_lang   nl \
        --condition   probe_gap \
        --layers      27 28 31 \
        --alpha       0.1 \
        --input_csv   translations/apertus/translations_afrikaans_apertus-8b-instruct-2509.csv
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HOME"] = os.path.join(
    os.environ.get("SCRATCH", os.path.expanduser("~")), "hf_cache"
)

# ============================================================
# Args
# ============================================================

parser = argparse.ArgumentParser()
parser.add_argument("--source_lang", type=str, default="gl",
                    help="Low-resource target language code (e.g. gl)")
parser.add_argument("--axis_lang", type=str, default="es",
                    help="High-resource axis language to ablate (e.g. es)")
parser.add_argument("--condition", type=str, default="late",
                    choices=["baseline", "none", "all_layers", "early",
                             "late", "pivot", "probe_gap"])
parser.add_argument("--layers", nargs="+", type=int, default=None,
                    help="Explicit layer indices for probe_gap condition, "
                         "e.g. --layers 25 27 for Romance, --layers 27 28 31 for Germanic")
parser.add_argument("--input_csv", type=str, required=True)
parser.add_argument("--ref_json", type=str,
                    default="references/reference_pairs.json")
parser.add_argument("--vecs_npz", type=str,
                    default="references/contrastive_vectors.npz")
parser.add_argument("--n", type=int, default=250,
                    help="Number of sentences per group (top/bottom N)")
parser.add_argument("--alpha", type=float, default=0.1,
                    help="Ablation strength: 1.0=full removal, 0.5=partial")
parser.add_argument("--proj_threshold", type=float, default=0.0,
                    help="Only ablate if |projection| > threshold. 0.0 disables gating.")
parser.add_argument("--model", type=str,
                    default="swiss-ai/Apertus-8B-Instruct-2509")
parser.add_argument("--output_dir", type=str,
                    default="ablation_results")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--log_proj", action="store_true",
                    help="Log average last-token projection magnitudes per sentence")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

LANG_NAMES = {
    "gl": "Galician", "es": "Spanish",
    "af": "Afrikaans", "nl": "Dutch",
    "ca": "Catalan",   "de": "German",
    "en": "English",
}

# ============================================================
# Load contrastive vectors
# ============================================================

print("Loading references and contrastive vectors...")
with open(args.ref_json) as f:
    pairs = json.load(f)

vecs = np.load(args.vecs_npz)
v_axis = vecs[f"v_{args.axis_lang}"]          # (33, hidden_dim)
v_axis_t = torch.tensor(v_axis, dtype=torch.float32)

print(f"Axis language : {LANG_NAMES[args.axis_lang]}")
print(f"Axis shape    : {v_axis.shape}")
print(f"  → index 0 = embedding layer output")
print(f"  → index i+1 = transformer layer i output")

# ============================================================
# Load model — skip only for reference baseline
# ============================================================

if args.condition != "baseline":
    print("\nLoading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    model.eval()
    torch.cuda.empty_cache()
    first_device = next(model.parameters()).device
    n_layers = model.config.num_hidden_layers
    print(f"Model loaded | {n_layers} transformer layers | device: {first_device}\n")

    # Explicit model-family validation.
    if "gemma-7b" in args.model.lower():
        if n_layers != 28:
            raise RuntimeError(
                f"Expected Gemma 7B to have 28 transformer layers, got {n_layers}"
            )
        print("Gemma mapping: transformer L27 -> hidden-state L28 -> v_axis[28]")
else:
    print("\nBaseline condition — skipping model load.\n")
    tokenizer = None
    model = None
    first_device = None
    # Only used for layer-set construction in baseline mode.
    # Apertus is the historical default for this mode.
    n_layers = 32

# ============================================================
# Define ablation layer set
# ============================================================

PIVOT_LAYERS = {
    "gl": set(range(18, 27)),
    "af": set(range(20, 27)),
    "es": set(range(18, 28)),
    "nl": set(range(20, 27)),
}

if args.condition in {"baseline", "none"}:
    ablation_layers = set()
elif args.condition == "all_layers":
    ablation_layers = set(range(n_layers))
elif args.condition == "early":
    ablation_layers = set(range(0, n_layers // 2))
elif args.condition == "late":
    ablation_layers = set(range(n_layers // 2, n_layers))
elif args.condition == "pivot":
    ablation_layers = PIVOT_LAYERS.get(args.source_lang, set())
    if not ablation_layers:
        print(f"WARNING: No pivot layers defined for {args.source_lang}, "
              f"falling back to all layers.")
        ablation_layers = set(range(n_layers))
elif args.condition == "probe_gap":
    if not args.layers:
        raise ValueError(
            "--layers is required for probe_gap condition.\n"
            "  Romance (gl): --layers 25 27\n"
            "  Germanic (af): --layers 27 28 31"
        )
    ablation_layers = set(args.layers)

print(f"Condition        : {args.condition}")
print(f"Ablation alpha   : {args.alpha}")
print(f"Proj threshold   : {args.proj_threshold}")
if ablation_layers:
    print(f"Layers ablated   : {sorted(ablation_layers)}")
    print(f"  → using v_axis indices "
          f"{[l+1 for l in sorted(ablation_layers)]}")

    if "gemma-7b" in args.model.lower():
        print(
            "  → Gemma: transformer L27 corresponds to hidden-state/vector index 28"
        )
    print()
else:
    print(f"Layers ablated   : 0 (no hooks)\n")

# ============================================================
# Hook factory (SURGICAL VERSION)
#
# Only ablates the LAST token hidden state:
#   last = hidden[:, -1, :]
#
#   last' = last - alpha * proj_v(last)
#
# Optional gating:
#   only apply if |proj| > threshold
# ============================================================

registered_hooks = []
current_sentence_proj_log = []

def make_ablation_hook(layer_idx, alpha):
    def hook(module, input, output):
        if layer_idx not in ablation_layers:
            return output

        # Hidden-state vectors are indexed one past the transformer block:
        # Gemma:  L0=embedding, L1=block 0, ..., L28=block 27
        # Apertus: L0=embedding, L1=block 0, ..., L32=block 31
        v_idx = layer_idx + 1

        if v_idx >= v_axis_t.shape[0]:
            raise RuntimeError(
                f"Invalid vector index {v_idx} for transformer layer L{layer_idx}; "
                f"v_axis has {v_axis_t.shape[0]} entries."
            )

        # handle both Tensor and tuple output formats
        if isinstance(output, torch.Tensor):
            hidden = output.float()
            rest = None
            out_dtype = output.dtype
        elif isinstance(output, tuple):
            hidden = output[0].float()
            rest = output[1:]
            out_dtype = output[0].dtype
        else:
            return output

        # hidden shape: (batch, seq_len, d_model)
        if hidden.ndim != 3 or hidden.shape[1] == 0:
            return output

        v = v_axis_t[v_idx].to(hidden.device)
        v = v / (v.norm() + 1e-8)

        # ── SURGICAL CHANGE: only last token ──────────────────
        last = hidden[:, -1, :]                # (batch, d_model)
        proj = last @ v                        # (batch,)

        if args.log_proj:
            current_sentence_proj_log.append({
                "layer": layer_idx,
                "mean_abs_proj": float(proj.abs().mean().item()),
                "mean_proj": float(proj.mean().item()),
            })

        # Optional gating
        if args.proj_threshold > 0:
            mask = (proj.abs() > args.proj_threshold).float()   # (batch,)
        else:
            mask = torch.ones_like(proj)

        proj = proj.unsqueeze(-1)              # (batch, 1)
        mask = mask.unsqueeze(-1)              # (batch, 1)

        last = last - alpha * mask * proj * v
        hidden[:, -1, :] = last
        # ──────────────────────────────────────────────────────

        hidden = hidden.to(out_dtype)

        if rest is None:
            return hidden
        else:
            return (hidden,) + rest

    return hook


def register_hooks():
    for i, layer in enumerate(model.model.layers):
        h = layer.register_forward_hook(make_ablation_hook(i, args.alpha))
        registered_hooks.append(h)


def remove_hooks():
    for h in registered_hooks:
        h.remove()
    registered_hooks.clear()

# ============================================================
# Prompt template
# ============================================================

def make_prompt(source, target_lang_name):
    return (
        f"Translate the following sentences from English to {target_lang_name}.\n"
        "Output exactly one translated sentence per line, keeping the same order.\n"
        "Do not add numbering, explanations, or any extra text.\n\n"
        f"{source}"
    )

# ============================================================
# Load top and bottom N sentences by COMET
# ============================================================

# ============================================================
# Load top N sentences by COMET
# ============================================================

print("Loading sentences...")

df = pd.read_csv(args.input_csv)
df_sort = df.sort_values("comet_kiwi").reset_index(drop=True)

# Top N = highest-quality translations only
top = df_sort.tail(args.n).copy()
top["group"] = 1       # good

df_sample = top.reset_index(drop=True)

# Hard invariant: every experimental condition must generate exactly args.n
# unique input sentences. This prevents silently writing malformed result files.
if len(df_sample) != args.n:
    raise RuntimeError(
        f"Expected exactly {args.n} sampled rows, got {len(df_sample)}"
    )

if "sentence" not in df_sample.columns:
    raise RuntimeError("Input CSV is missing required column: sentence")

if df_sample["sentence"].isna().any():
    raise RuntimeError("Selected sample contains missing source sentences")

n_unique_sources = df_sample["sentence"].astype(str).nunique()
if n_unique_sources != args.n:
    raise RuntimeError(
        f"Selected sample contains duplicate source sentences: "
        f"{n_unique_sources} unique sources for {args.n} rows"
    )

print(f"Sample invariant OK: {len(df_sample)} rows / {n_unique_sources} unique sources")

print(
    f"Top {args.n} (good): comet "
    f"{top['comet_kiwi'].min():.4f}–"
    f"{top['comet_kiwi'].max():.4f} | "
    f"mean {top['comet_kiwi'].mean():.4f}"
)

print(f"Total: {len(df_sample)} sentences\n")
target_lang_name = LANG_NAMES[args.source_lang]

# ============================================================
# Register hooks if needed
# ============================================================

if args.condition not in {"baseline", "none"}:
    register_hooks()
    print(f"Hooks registered on {len(registered_hooks)} layers.\n")

# ============================================================
# Main generation loop
# ============================================================

results = []
total = len(df_sample)

seen_sources = set()

for i, row in df_sample.iterrows():
    source = str(row["sentence"])

    if source in seen_sources:
        raise RuntimeError(
            f"Duplicate source encountered during generation at sample index {i}: "
            f"{source[:150]!r}"
        )
    seen_sources.add(source)
    reference = str(row["translation"])
    baseline_comet = float(row["comet_kiwi"])
    group = int(row["group"])
    group_name = "good" if group == 1 else "bad"

    current_sentence_proj_log.clear()

    # ── Reference baseline (stored translation) ──────────────
    if args.condition == "baseline":
        generated = reference
        print(f"[{i+1}/{total}] {group_name:4s} | "
              f"comet={baseline_comet:.4f} | baseline (stored)")

        results.append({
            "source": source,
            "reference": reference,
            "generated": generated,
            "group": group,
            "group_name": group_name,
            "baseline_comet": baseline_comet,
            "condition": args.condition,
            "source_lang": args.source_lang,
            "axis_lang": args.axis_lang,
            "alpha": args.alpha,
            "proj_threshold": args.proj_threshold,
            "ablation_layers": "[]",
            "avg_abs_proj": np.nan,
            "avg_proj": np.nan,
        })
        continue

    # ── True model generation (with or without hooks) ────────
    prompt = make_prompt(source, target_lang_name)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )
    inputs = {k: v.to(first_device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
        )

    generated = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()

    if current_sentence_proj_log:
        avg_abs_proj = float(np.mean([x["mean_abs_proj"] for x in current_sentence_proj_log]))
        avg_proj = float(np.mean([x["mean_proj"] for x in current_sentence_proj_log]))
    else:
        avg_abs_proj = np.nan
        avg_proj = np.nan

    print(f"[{i+1}/{total}] {group_name:4s} | "
          f"comet={baseline_comet:.4f} | "
          f"condition={args.condition} | alpha={args.alpha}")
    print(f"  src : {source[:90]}")
    print(f"  ref : {reference[:90]}")
    print(f"  gen : {generated[:90]}")
    if args.log_proj:
        print(f"  avg_abs_proj: {avg_abs_proj:.4f}")
    print()

    results.append({
        "source": source,
        "reference": reference,
        "generated": generated,
        "group": group,
        "group_name": group_name,
        "baseline_comet": baseline_comet,
        "condition": args.condition,
        "source_lang": args.source_lang,
        "axis_lang": args.axis_lang,
        "alpha": args.alpha,
        "proj_threshold": args.proj_threshold,
        "ablation_layers": str(sorted(ablation_layers)),
        "avg_abs_proj": avg_abs_proj,
        "avg_proj": avg_proj,
    })

# ============================================================
# Remove hooks
# ============================================================

if args.condition not in {"baseline", "none"}:
    remove_hooks()
    print("Hooks removed.")

# ============================================================
# Save
# ============================================================

# Final hard invariants: never save a partial or duplicated experimental file.
if len(results) != args.n:
    raise RuntimeError(
        f"Refusing to save: expected {args.n} results, got {len(results)}"
    )

result_sources = [str(r["source"]) for r in results]
if len(set(result_sources)) != len(result_sources):
    raise RuntimeError(
        "Refusing to save: duplicate source detected in generated results"
    )

if set(result_sources) != set(df_sample["sentence"].astype(str)):
    raise RuntimeError(
        "Refusing to save: generated sources do not exactly match sampled sources"
    )

print(
    f"Final invariant OK: {len(results)} results / "
    f"{len(set(result_sources))} unique sources"
)

layers_str = "_".join(str(l) for l in sorted(ablation_layers)) if ablation_layers else "none"
out_file = os.path.join(
    args.output_dir,
    f"{args.source_lang}_ablate_{args.axis_lang}"
    f"_{args.condition}_L{layers_str}_a{args.alpha}_t{args.proj_threshold}.csv"
)
pd.DataFrame(results).to_csv(out_file, index=False)

print(f"\nSaved → {out_file}")
print(f"Total sentences written: {len(results)}")
print("Now run rescore.py to get COMET deltas.")