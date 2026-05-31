#!/usr/bin/env python3
"""Compute per-image Jaccard similarity between two selection tensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selection_a", help="First .pt tensor, shape [N,k]")
    parser.add_argument("selection_b", help="Second .pt tensor, shape [N,k]")
    parser.add_argument("output", help="Output JSON path")
    args = parser.parse_args()

    sel_a = torch.load(args.selection_a, map_location="cpu")
    sel_b = torch.load(args.selection_b, map_location="cpu")
    if sel_a.shape != sel_b.shape:
        raise SystemExit(f"shape mismatch: {tuple(sel_a.shape)} vs {tuple(sel_b.shape)}")
    if sel_a.ndim != 2:
        raise SystemExit(f"expected [N,k] tensors, got {tuple(sel_a.shape)}")

    jaccards = []
    for row_a, row_b in zip(sel_a, sel_b):
        a = {int(x) for x in row_a.tolist() if int(x) >= 0}
        b = {int(x) for x in row_b.tolist() if int(x) >= 0}
        if not a and not b:
            jaccards.append(1.0)
        else:
            jaccards.append(len(a & b) / max(1, len(a | b)))

    values = torch.tensor(jaccards, dtype=torch.float32)
    result = {
        "n_images": int(values.numel()),
        "mean_jaccard": float(values.mean()),
        "median_jaccard": float(values.median()),
        "std": float(values.std(unbiased=False)),
        "min": float(values.min()),
        "max": float(values.max()),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
