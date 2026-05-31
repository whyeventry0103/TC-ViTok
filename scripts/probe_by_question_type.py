#!/usr/bin/env python3
"""Train causal-importance probes split by simple GQA question type."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from probe_with_bootstrap import bootstrap_image_r2, train_probe
from train_causal_probe import r2_score


def classify_question(question: str) -> str:
    q = question.lower()
    if "what color" in q or "what colour" in q:
        return "color"
    if "how many" in q or "count" in q or "number of" in q:
        return "count"
    if "is there" in q or "are there" in q:
        return "existence"
    if any(k in q for k in ["where", "left", "right", "top", "bottom", "behind", "in front"]):
        return "spatial"
    if "what is" in q or "what's" in q or "what kind" in q:
        return "identity"
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--probe", choices=["linear", "mlp"], default="mlp")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--min-images", type=int, default=20)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = torch.load(args.data, map_location="cpu")
    features = data["features"].float()
    targets = data["importances"].float()
    metadata = data.get("metadata", [])
    if len(metadata) != features.shape[0]:
        raise ValueError(f"metadata length {len(metadata)} does not match features {features.shape[0]}")

    buckets: dict[str, list[int]] = {}
    for idx, meta in enumerate(metadata):
        buckets.setdefault(classify_question(meta.get("question", "")), []).append(idx)

    results = {}
    for qtype, indices in sorted(buckets.items()):
        if len(indices) < args.min_images:
            results[qtype] = {"n_images": len(indices), "skipped": f"below min_images={args.min_images}"}
            continue
        idx = torch.tensor(indices, dtype=torch.long)
        trained = train_probe(
            features=features[idx],
            targets=targets[idx],
            probe_type=args.probe,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=1e-3,
        )
        pred = trained["pred_val"]
        target = trained["target_val"]
        fixed_r2 = r2_score(pred.reshape(-1), target.reshape(-1))
        boot = bootstrap_image_r2(pred, target, args.n_bootstrap, seed=42)
        results[qtype] = {
            "n_images": len(indices),
            "fixed_split_r2": float(fixed_r2),
            "bootstrap_mean_r2": boot["mean_r2"],
            "ci_95_lower": boot["ci_95_lower"],
            "ci_95_upper": boot["ci_95_upper"],
            "n_val_images": trained["n_val_images"],
        }
        print(f"{qtype}: n={len(indices)}, R2={fixed_r2:.4f}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"probe": args.probe, "results": results}, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
