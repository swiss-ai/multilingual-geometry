#!/usr/bin/env python3

"""
ablation.py — Surgically ablate the high-resource language direction
from low-resource hidden states during inference.

FAST VERSION:
- Batched generation
- Left padding
- SDPA attention
- inference_mode()
- Hooks only on selected layers
- Only last-token representation converted to FP32

Uses top N sentences by COMET score.

Conditions:
  baseline:   no generation — uses stored translation directly
  none:       generate normally with NO hooks
  all_layers: ablate v_H at all selected decoding layers
  early:      ablate v_H at layers 0–15
  late:       ablate v_H at layers 16–31
  pivot:      ablate v_H at detected pivot layers only
  probe_gap:  ablate v_H at explicitly specified layers

Surgical ablation:
    hidden[:, -1, :]

Optional gating:
    Only ablate when |projection| > proj_threshold.
"""

# ============================================================
# Imports
# ============================================================

import os
import argparse
import json

import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# Environment
# ============================================================

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

os.environ["HF_HOME"] = os.path.join(
    os.environ.get("SCRATCH", os.path.expanduser("~")),
    "hf_cache"
)


# ============================================================
# Args
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--source_lang",
    type=str,
    default="gl",
    help="Low-resource target language code (e.g. gl)"
)

parser.add_argument(
    "--axis_lang",
    type=str,
    default="es",
    help="High-resource axis language to ablate (e.g. es)"
)

parser.add_argument(
    "--condition",
    type=str,
    default="late",
    choices=[
        "baseline",
        "none",
        "all_layers",
        "early",
        "late",
        "pivot",
        "probe_gap",
    ]
)

parser.add_argument(
    "--layers",
    nargs="+",
    type=int,
    default=None,
    help="Explicit layer indices for probe_gap"
)

parser.add_argument(
    "--input_csv",
    type=str,
    required=True
)

parser.add_argument(
    "--ref_json",
    type=str,
    default="references/reference_pairs.json"
)

parser.add_argument(
    "--vecs_npz",
    type=str,
    default="references/contrastive_vectors.npz"
)

parser.add_argument(
    "--n",
    type=int,
    default=250,
    help="Number of sentences"
)

parser.add_argument(
    "--alpha",
    type=float,
    default=0.1,
    help="Ablation strength: 1.0=full removal, 0.5=partial"
)

parser.add_argument(
    "--proj_threshold",
    type=float,
    default=0.0,
    help="Only ablate if |projection| > threshold"
)

parser.add_argument(
    "--model",
    type=str,
    default="swiss-ai/Apertus-8B-Instruct-2509"
)

parser.add_argument(
    "--output_dir",
    type=str,
    default="ablation_results"
)

parser.add_argument(
    "--seed",
    type=int,
    default=42
)

parser.add_argument(
    "--log_proj",
    action="store_true",
    help="Log last-token projection magnitudes per sentence"
)

# NEW:
parser.add_argument(
    "--batch_size",
    type=int,
    default=16,
    help="Generation batch size. Try 16 or 32 on GH200."
)

args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)


# ============================================================
# Seeds
# ============================================================

torch.manual_seed(args.seed)
np.random.seed(args.seed)


# ============================================================
# Language names
# ============================================================

LANG_NAMES = {
    "gl": "Galician",
    "es": "Spanish",
    "af": "Afrikaans",
    "nl": "Dutch",
    "ca": "Catalan",
    "de": "German",
    "en": "English",
}


# ============================================================
# Load contrastive vectors
# ============================================================

print("Loading references and contrastive vectors...")

with open(args.ref_json) as f:
    pairs = json.load(f)

vecs = np.load(args.vecs_npz)

v_axis = vecs[f"v_{args.axis_lang}"]

# Keep original vectors in FP32.
v_axis_t = torch.tensor(
    v_axis,
    dtype=torch.float32
)

print(f"Axis language : {LANG_NAMES[args.axis_lang]}")
print(f"Axis shape    : {v_axis.shape}")
print("  → index 0 = embedding layer output")
print("  → index i+1 = transformer layer i output")


# ============================================================
# Load model
# ============================================================

if args.condition != "baseline":

    print("\nLoading tokenizer and model...")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model
    )

    # IMPORTANT FOR BATCHED CAUSAL-LM GENERATION
    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )

    model.eval()

    torch.cuda.empty_cache()

    first_device = next(model.parameters()).device

    n_layers = model.config.num_hidden_layers

    print(
        f"Model loaded | {n_layers} layers | "
        f"device: {first_device}"
    )

else:

    print(
        "\nBaseline condition — skipping model load.\n"
    )

    tokenizer = None
    model = None
    first_device = None
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

    ablation_layers = set(
        range(n_layers // 2, n_layers)
    )

elif args.condition == "pivot":

    ablation_layers = PIVOT_LAYERS.get(
        args.source_lang,
        set()
    )

    if not ablation_layers:

        print(
            f"WARNING: No pivot layers defined for "
            f"{args.source_lang}, falling back to all layers."
        )

        ablation_layers = set(range(n_layers))

elif args.condition == "probe_gap":

    if not args.layers:

        raise ValueError(
            "--layers is required for probe_gap condition."
        )

    ablation_layers = set(args.layers)


print(f"Condition        : {args.condition}")
print(f"Ablation alpha   : {args.alpha}")
print(f"Proj threshold   : {args.proj_threshold}")

if ablation_layers:

    print(
        f"Layers ablated   : "
        f"{sorted(ablation_layers)}"
    )

    print(
        "  → using v_axis indices "
        f"{[l + 1 for l in sorted(ablation_layers)]}\n"
    )

else:

    print("Layers ablated   : 0 (no hooks)\n")


# ============================================================
# Hook factory
# ============================================================

registered_hooks = []

# Stores projection values for each batch.
# Only used if --log_proj is enabled.
current_batch_proj_logs = []


def make_ablation_hook(layer_idx, alpha):

    def hook(module, input, output):

        # This should normally never trigger now because
        # we only register hooks on selected layers.
        if layer_idx not in ablation_layers:
            return output

        v_idx = layer_idx + 1

        if v_idx >= v_axis_t.shape[0]:
            return output

        # ----------------------------------------------------
        # Handle Tensor / tuple output
        # ----------------------------------------------------

        if isinstance(output, torch.Tensor):

            hidden = output
            rest = None

        elif isinstance(output, tuple):

            hidden = output[0]
            rest = output[1:]

        else:

            return output

        # ----------------------------------------------------
        # Expected shape:
        #
        # (batch, seq_len, hidden_dim)
        # ----------------------------------------------------

        if hidden.ndim != 3:
            return output

        if hidden.shape[1] == 0:
            return output

        # ----------------------------------------------------
        # ONLY last token
        #
        # Do not convert the whole hidden tensor to FP32.
        # ----------------------------------------------------

        last = hidden[:, -1, :].float()

        # ----------------------------------------------------
        # Projection vector
        # ----------------------------------------------------

        v = v_axis_t[v_idx].to(
            device=last.device,
            dtype=last.dtype
        )

        v = v / (v.norm() + 1e-8)

        # (batch,)
        proj = last @ v

        # ----------------------------------------------------
        # Projection logging
        # ----------------------------------------------------

        if args.log_proj:

            current_batch_proj_logs.append(
                {
                    "layer": layer_idx,
                    "abs_proj": (
                        proj.detach()
                        .abs()
                        .cpu()
                        .numpy()
                    ),
                    "proj": (
                        proj.detach()
                        .cpu()
                        .numpy()
                    ),
                }
            )

        # ----------------------------------------------------
        # Optional gating
        # ----------------------------------------------------

        if args.proj_threshold > 0:

            mask = (
                proj.abs() >
                args.proj_threshold
            ).to(last.dtype)

        else:

            mask = torch.ones_like(
                proj,
                dtype=last.dtype
            )

        proj = proj.unsqueeze(-1)

        mask = mask.unsqueeze(-1)

        # ----------------------------------------------------
        # Ablate
        #
        # last' = last - alpha * proj * v
        # ----------------------------------------------------

        last = (
            last -
            alpha *
            mask *
            proj *
            v
        )

        # Convert ONLY modified last token back to
        # the model's original dtype.

        hidden[:, -1, :] = last.to(
            hidden.dtype
        )

        # ----------------------------------------------------
        # Return same structure as original output
        # ----------------------------------------------------

        if rest is None:

            return hidden

        else:

            return (
                (hidden,) +
                rest
            )

    return hook


# ============================================================
# Register only selected layers
# ============================================================

def register_hooks():

    for layer_idx in sorted(ablation_layers):

        h = model.model.layers[
            layer_idx
        ].register_forward_hook(
            make_ablation_hook(
                layer_idx,
                args.alpha
            )
        )

        registered_hooks.append(h)

    print(
        f"Hooks registered on "
        f"{len(registered_hooks)} selected layers."
    )


def remove_hooks():

    for h in registered_hooks:
        h.remove()

    registered_hooks.clear()


# ============================================================
# Prompt template
# ============================================================

def make_prompt(
    source,
    target_lang_name
):

    return (
        f"Translate the following sentences "
        f"from English to {target_lang_name}.\n"
        "Output exactly one translated sentence per line, "
        "keeping the same order.\n"
        "Do not add numbering, explanations, "
        "or any extra text.\n\n"
        f"{source}"
    )


# ============================================================
# Load top N sentences by COMET
# ============================================================

print("Loading sentences...")

df = pd.read_csv(args.input_csv)

df_sort = (
    df.sort_values("comet_kiwi")
    .reset_index(drop=True)
)

# Top N = highest-quality translations
top = df_sort.tail(args.n).copy()

top["group"] = 1

df_sample = (
    top.reset_index(drop=True)
)

print(
    f"Top {args.n} (good): comet "
    f"{top['comet_kiwi'].min():.4f}–"
    f"{top['comet_kiwi'].max():.4f} | "
    f"mean {top['comet_kiwi'].mean():.4f}"
)

print(
    f"Total: {len(df_sample)} sentences"
)

print(
    f"Batch size: {args.batch_size}"
)

target_lang_name = LANG_NAMES[
    args.source_lang
]


# ============================================================
# Register hooks
# ============================================================

if args.condition not in {
    "baseline",
    "none"
}:

    register_hooks()


# ============================================================
# Results
# ============================================================

results = []

total = len(df_sample)


# ============================================================
# BASELINE
# ============================================================

if args.condition == "baseline":

    print(
        "Running stored-translation baseline..."
    )

    for i, row in df_sample.iterrows():

        source = str(row["sentence"])
        reference = str(row["translation"])

        baseline_comet = float(
            row["comet_kiwi"]
        )

        group = int(row["group"])

        results.append(
            {
                "source": source,
                "reference": reference,
                "generated": reference,
                "group": group,
                "group_name": "good",
                "baseline_comet": baseline_comet,
                "condition": args.condition,
                "source_lang": args.source_lang,
                "axis_lang": args.axis_lang,
                "alpha": args.alpha,
                "proj_threshold": args.proj_threshold,
                "ablation_layers": "[]",
                "avg_abs_proj": np.nan,
                "avg_proj": np.nan,
            }
        )

    print(
        f"Processed {total}/{total}"
    )


# ============================================================
# BATCHED MODEL GENERATION
# ============================================================

else:

    for start in range(
        0,
        total,
        args.batch_size
    ):

        end = min(
            start + args.batch_size,
            total
        )

        batch_df = df_sample.iloc[
            start:end
        ]

        sources = (
            batch_df["sentence"]
            .astype(str)
            .tolist()
        )

        references = (
            batch_df["translation"]
            .astype(str)
            .tolist()
        )

        baseline_comets = (
            batch_df["comet_kiwi"]
            .astype(float)
            .tolist()
        )

        groups = (
            batch_df["group"]
            .astype(int)
            .tolist()
        )

        # ----------------------------------------------------
        # Prompts
        # ----------------------------------------------------

        prompts = [
            make_prompt(
                source,
                target_lang_name
            )
            for source in sources
        ]

        # ----------------------------------------------------
        # Tokenize entire batch
        # ----------------------------------------------------

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )

        inputs = {
            k: v.to(first_device)
            for k, v in inputs.items()
        }

        # ----------------------------------------------------
        # Reset projection logs
        # ----------------------------------------------------

        current_batch_proj_logs.clear()

        # ----------------------------------------------------
        # Generation
        # ----------------------------------------------------

        with torch.inference_mode():

            output_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Because we're using left padding, all examples
        # share the same padded input length.

        padded_input_len = (
            inputs["input_ids"].shape[1]
        )

        # ----------------------------------------------------
        # Projection statistics
        # ----------------------------------------------------

        if (
            args.log_proj
            and current_batch_proj_logs
        ):

            # Build per-sentence statistics.
            #
            # Each hook contributes an array of shape
            # (batch,).

            batch_size_actual = len(
                sources
            )

            per_sentence_abs = [
                []
                for _ in range(
                    batch_size_actual
                )
            ]

            per_sentence_proj = [
                []
                for _ in range(
                    batch_size_actual
                )
            ]

            for log in current_batch_proj_logs:

                for j in range(
                    batch_size_actual
                ):

                    per_sentence_abs[j].append(
                        log["abs_proj"][j]
                    )

                    per_sentence_proj[j].append(
                        log["proj"][j]
                    )

            batch_avg_abs_proj = [
                float(np.mean(x))
                if x else np.nan
                for x in per_sentence_abs
            ]

            batch_avg_proj = [
                float(np.mean(x))
                if x else np.nan
                for x in per_sentence_proj
            ]

        else:

            batch_avg_abs_proj = [
                np.nan
            ] * len(sources)

            batch_avg_proj = [
                np.nan
            ] * len(sources)

        # ----------------------------------------------------
        # Decode and save each example
        # ----------------------------------------------------

        for j in range(len(sources)):

            generated = tokenizer.decode(
                output_ids[j][
                    padded_input_len:
                ],
                skip_special_tokens=True,
            ).strip()

            group = groups[j]

            results.append(
                {
                    "source": sources[j],
                    "reference": references[j],
                    "generated": generated,
                    "group": group,
                    "group_name": (
                        "good"
                        if group == 1
                        else "bad"
                    ),
                    "baseline_comet": (
                        baseline_comets[j]
                    ),
                    "condition": args.condition,
                    "source_lang": args.source_lang,
                    "axis_lang": args.axis_lang,
                    "alpha": args.alpha,
                    "proj_threshold": (
                        args.proj_threshold
                    ),
                    "ablation_layers": str(
                        sorted(ablation_layers)
                    ),
                    "avg_abs_proj": (
                        batch_avg_abs_proj[j]
                    ),
                    "avg_proj": (
                        batch_avg_proj[j]
                    ),
                }
            )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        print(
            f"[{end}/{total}] "
            f"{args.condition} "
            f"alpha={args.alpha} "
            f"batch={len(sources)}"
        )


# ============================================================
# Remove hooks
# ============================================================

if args.condition not in {
    "baseline",
    "none"
}:

    remove_hooks()

    print("Hooks removed.")


# ============================================================
# Save
# ============================================================

layers_str = (
    "_".join(
        str(l)
        for l in sorted(ablation_layers)
    )
    if ablation_layers
    else "none"
)

out_file = os.path.join(
    args.output_dir,
    f"{args.source_lang}"
    f"_ablate_{args.axis_lang}"
    f"_{args.condition}"
    f"_L{layers_str}"
    f"_a{args.alpha}"
    f"_t{args.proj_threshold}.csv"
)

pd.DataFrame(
    results
).to_csv(
    out_file,
    index=False
)

print(
    f"\nSaved → {out_file}"
)

print(
    f"Total sentences written: "
    f"{len(results)}"
)

print(
    "Now run rescore.py to get COMET deltas."
)