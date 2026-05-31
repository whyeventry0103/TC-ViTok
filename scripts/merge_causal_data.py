#!/usr/bin/env python3
"""Merge causal-importance .pt shards along the image dimension."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    shards = [torch.load(path, map_location="cpu") for path in args.inputs]
    importances = torch.cat([s["importances"] for s in shards], dim=0)
    features = torch.cat([s["features"] for s in shards], dim=0)
    metadata = []
    for shard in shards:
        metadata.extend(shard.get("metadata", []))

    merged = {
        "importances": importances,
        "features": features,
        "metadata": metadata,
        "config": {
            "merged_from": args.inputs,
            "n_images": int(importances.shape[0]),
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, out)

    summary = {
        "output": str(out),
        "shape_importances": list(importances.shape),
        "shape_features": list(features.shape),
        "n_metadata": len(metadata),
        "importance_mean": float(importances.float().mean()),
        "importance_std": float(importances.float().std(unbiased=False)),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
