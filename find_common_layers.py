"""
find_common_layers.py — Intersect accepted_layers across pairs within
each resource triangle.
"""

import argparse
import json
import os

TRIANGLES = {
    "ibero_romance": ["gl_es", "ca_es", "gl_ca"],
    "west_germanic":  ["af_de", "nl_de", "af_nl"],
}

parser = argparse.ArgumentParser()
parser.add_argument("--results_dir", type=str, default="layer_selection_results")
args = parser.parse_args()

results = {}
for fname in os.listdir(args.results_dir):
    if fname.endswith("_layers.json"):
        key = fname.replace("_layers.json", "")
        with open(os.path.join(args.results_dir, fname)) as f:
            results[key] = json.load(f)

for triangle_name, pairs in TRIANGLES.items():
    print(f"\n{'='*60}")
    print(f"{triangle_name}")
    print(f"{'='*60}")

    sets = {}
    for pair_key in pairs:
        if pair_key not in results:
            print(f"  WARNING: {pair_key} not found, skipping.")
            continue
        accepted = set(results[pair_key]["accepted_layers"])
        sets[pair_key] = accepted
        print(f"  {pair_key}: {sorted(accepted)}  ({len(accepted)} layers)")

    if len(sets) < len(pairs):
        print(f"  Incomplete — need all {len(pairs)} pairs.")
        continue

    intersection = set.intersection(*sets.values())
    print(f"\n  Common (intersection) layers: {sorted(intersection)}")

    if intersection:
        avg_asym = {}
        for l in intersection:
            vals = [results[p]["per_layer"][str(l)]["asymmetry_orth"] for p in pairs]
            avg_asym[l] = sum(vals) / len(vals)
        best = max(avg_asym, key=avg_asym.get)
        print(f"  Recommended single layer (highest avg asymmetry in intersection): "
              f"L{best} (avg asym={avg_asym[best]:.4f})")
        print(f"  Per-layer avg asymmetry in intersection:")
        for l in sorted(intersection, key=lambda x: -avg_asym[x]):
            print(f"    L{l}: {avg_asym[l]:.4f}")
    else:
        print("  No common layer across all pairs in this triangle.")
