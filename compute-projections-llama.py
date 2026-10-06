"""
compute_projections_llama.py

Compute mean projection magnitude of Llama hidden states onto
contrastive language-axis vectors, per layer.

Expected Llama hidden-state files:

    llama-gl/top2000_gl_hidden_states.npz
    llama-es/top2000_es_hidden_states.npz
    ...

Each file contains:

    hidden_states: (n_sentences, n_layers, hidden_dim)

For Llama 3 8B-style models:

    hidden_states = (N, 33, 4096)

where:
    index 0  = embedding/input hidden state
    index 1  = transformer layer 0 output
    ...
    index 32 = transformer layer 31 output

Contrastive vectors:

    v_es, v_ca, v_gl, v_de, v_nl, v_af

each with shape:

    (33, 4096)

Projection:

    p_i^(l) = h_i^(l) · â^(l)

where â^(l) is the independently normalized language-axis
vector at layer l.

Output:

    mean_abs_proj
    std_abs_proj
"""


import argparse
import os

import numpy as np
import pandas as pd


# ============================================================
# Arguments
# ============================================================

parser = argparse.ArgumentParser()

group = parser.add_mutually_exclusive_group(required=True)

group.add_argument(
    "--pairs",
    nargs="+",
    help="Explicit source:axis pairs, e.g. gl:es gl:ca af:nl af:de"
)

group.add_argument(
    "--source_langs",
    nargs="+",
    help="Source language codes"
)

parser.add_argument(
    "--axis_langs",
    nargs="+",
    help="Axis language codes (used with --source_langs)"
)

parser.add_argument(
    "--hidden_root",
    type=str,
    default=".",
    help="Root directory containing llama-XX directories"
)

parser.add_argument(
    "--vecs_npz",
    type=str,
    required=True,
    help="Llama contrastive vectors NPZ"
)

parser.add_argument(
    "--output_dir",
    type=str,
    default="probe-results-llama",
    help="Directory for projection CSVs"
)

args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)


# ============================================================
# Language → hidden-state file
# ============================================================

LANG_FILES = {
    "gl": "llama-3.1-gl/top2000_gl_hidden_states.npz",
    "es": "llama-3.1-es/top2000_es_hidden_states.npz",
    "ca": "llama-3.1-ca/top2000_ca_hidden_states.npz",
    "af": "llama-3.1-af/top2000_af_hidden_states.npz",
    "nl": "llama-3.1-nl/top2000_nl_hidden_states.npz",
    "de": "llama-3.1-de/top2000_de_hidden_states.npz",
}


# ============================================================
# Resolve pairs
# ============================================================

if args.pairs:

    pairs = []

    for p in args.pairs:

        if ":" not in p:
            raise ValueError(
                f"Invalid pair '{p}'. "
                f"Expected format source:axis, e.g. gl:es"
            )

        source, axis = p.split(":", 1)

        pairs.append((source, axis))

else:

    if not args.axis_langs:
        raise ValueError(
            "--axis_langs required when using --source_langs"
        )

    pairs = [
        (source, axis)
        for source in args.source_langs
        for axis in args.axis_langs
    ]


print(f"Pairs to process: {pairs}")


# ============================================================
# Load contrastive vectors
# ============================================================

print(
    f"\nLoading contrastive vectors from "
    f"{args.vecs_npz}..."
)

vecs = np.load(args.vecs_npz)

print(
    f"Available vectors: {list(vecs.keys())}"
)


# ============================================================
# Process pairs
# ============================================================

all_results = []


for source_lang, axis_lang in pairs:

    print(f"\n{'=' * 60}")
    print(f"Pair: {source_lang} → {axis_lang}")
    print(f"{'=' * 60}")


    # --------------------------------------------------------
    # Load axis vector
    # --------------------------------------------------------

    axis_key = f"v_{axis_lang}"

    if axis_key not in vecs:

        print(
            f"WARNING: {axis_key} not found in "
            f"{args.vecs_npz}, skipping."
        )

        continue


    v_axis = vecs[axis_key].astype(np.float32)

    print(
        f"Axis shape: {v_axis.shape}"
    )


    # --------------------------------------------------------
    # Validate axis
    # --------------------------------------------------------

    if v_axis.ndim != 2:

        raise ValueError(
            f"Expected axis to be 2D, "
            f"got {v_axis.shape}"
        )


    n_layers, hidden_dim = v_axis.shape


    # --------------------------------------------------------
    # Llama sanity check
    # --------------------------------------------------------

    if n_layers != 33:

        print(
            f"WARNING: expected 33 hidden states for "
            f"a 32-layer Llama model, but found {n_layers}."
        )

    if hidden_dim != 4096:

        print(
            f"WARNING: expected hidden dimension 4096 "
            f"for Llama 8B, but found {hidden_dim}."
        )


    # --------------------------------------------------------
    # Normalize axis independently at each layer
    # --------------------------------------------------------

    norms = np.linalg.norm(
        v_axis,
        axis=1,
        keepdims=True
    )

    norms = np.maximum(norms, 1e-8)

    v_unit = v_axis / norms


    # --------------------------------------------------------
    # Load source hidden states
    # --------------------------------------------------------

    if source_lang not in LANG_FILES:

        print(
            f"WARNING: no hidden-state mapping for "
            f"{source_lang}, skipping."
        )

        continue


    hidden_path = os.path.join(
        args.hidden_root,
        LANG_FILES[source_lang]
    )


    if not os.path.exists(hidden_path):

        print(
            f"WARNING: hidden states not found at "
            f"{hidden_path}, skipping."
        )

        continue


    print(
        f"Loading hidden states: {hidden_path}"
    )


    data = np.load(hidden_path)


    if "hidden_states" not in data:

        raise KeyError(
            f"'hidden_states' not found in {hidden_path}. "
            f"Available keys: {list(data.keys())}"
        )


    hidden = data["hidden_states"].astype(np.float32)


    print(
        f"Hidden-state shape: {hidden.shape}"
    )


    # --------------------------------------------------------
    # Validate hidden states
    # --------------------------------------------------------

    if hidden.ndim != 3:

        raise ValueError(
            f"Expected hidden states to be 3D, "
            f"got {hidden.shape}"
        )


    n_sentences, hidden_layers, hidden_size = hidden.shape


    if hidden_layers != n_layers:

        raise ValueError(
            f"Layer mismatch: hidden states have "
            f"{hidden_layers} layers but axis has "
            f"{n_layers}"
        )


    if hidden_size != hidden_dim:

        raise ValueError(
            f"Hidden dimension mismatch: hidden states "
            f"have {hidden_size} dimensions but axis has "
            f"{hidden_dim}"
        )


    # --------------------------------------------------------
    # Projection
    # --------------------------------------------------------
    #
    # hidden:
    #
    #     (N, L, D)
    #
    # v_unit:
    #
    #     (L, D)
    #
    # result:
    #
    #     (N, L)
    #
    # p_i^l = h_i^l dot a^l
    # --------------------------------------------------------

    projections = np.sum(
        hidden * v_unit[None, :, :],
        axis=2
    )


    print(
        f"Projection shape: {projections.shape}"
    )


    # --------------------------------------------------------
    # Absolute projection magnitude
    # --------------------------------------------------------

    abs_proj = np.abs(projections)

    mean_abs = abs_proj.mean(axis=0)

    std_abs = abs_proj.std(axis=0)


    print(
        f"Sentences: {n_sentences}"
    )


    print(
        f"Layer-wise mean |proj| — "
        f"min: {mean_abs.min():.4f}, "
        f"max: {mean_abs.max():.4f}, "
        f"peak @ L{mean_abs.argmax()}"
    )


    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    df = pd.DataFrame({

        "layer":
            np.arange(n_layers),

        "mean_abs_proj":
            mean_abs,

        "std_abs_proj":
            std_abs,

        "source_lang":
            source_lang,

        "axis_lang":
            axis_lang,

    })


    out_path = os.path.join(
        args.output_dir,
        f"proj_{source_lang}_{axis_lang}.csv"
    )


    df.to_csv(
        out_path,
        index=False
    )


    print(
        f"Saved → {out_path}"
    )


    all_results.append(df)


# ============================================================
# Combined CSV
# ============================================================

if all_results:

    combined = pd.concat(
        all_results,
        ignore_index=True
    )


    combined_path = os.path.join(
        args.output_dir,
        "proj_all_pairs.csv"
    )


    combined.to_csv(
        combined_path,
        index=False
    )


    print(
        f"\nCombined → {combined_path}"
    )


    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print(
        "\nSummary (L16–L32):"
    )

    print(
        f"{'Pair':<12} "
        f"{'Peak layer':>10} "
        f"{'Peak |proj|':>14} "
        f"{'Mean L16+':>12}"
    )


    for df in all_results:

        late = df[
            df["layer"] >= 16
        ]


        peak_layer = int(
            df.loc[
                df["mean_abs_proj"].idxmax(),
                "layer"
            ]
        )


        peak_val = (
            df["mean_abs_proj"].max()
        )


        mean_late = (
            late["mean_abs_proj"].mean()
        )


        pair = (
            f"{df['source_lang'].iloc[0]}/"
            f"{df['axis_lang'].iloc[0]}"
        )


        print(
            f"{pair:<12} "
            f"{peak_layer:>10} "
            f"{peak_val:>14.4f} "
            f"{mean_late:>12.4f}"
        )


else:

    print(
        "\nNo pairs were successfully processed."
    )


print("\nDone.")