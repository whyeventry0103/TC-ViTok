#!/usr/bin/env python3
"""Probe alternative causal-importance target formulations.

This is a cheap diagnostic for Track C C2. It asks whether the negative
raw-scalar probe result is specific to the target formulation by trying
within-image ranks, top-k labels, per-image standardized importances, and a
position-only feature baseline on the already-computed C1 encoder features.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from probe_with_bootstrap import train_probe
from train_causal_probe import r2_score


def score_variant(
    features: torch.Tensor,
    targets: torch.Tensor,
    probe_type: str,
    epochs: int,
    batch_size: int,
) -> dict:
    trained = train_probe(
        features=features,
        targets=targets,
        probe_type=probe_type,
        epochs=epochs,
        batch_size=batch_size,
        lr=1e-3,
    )
    pred = trained["pred_val"]
    target = trained["target_val"]
    return {
        "fixed_split_r2": float(r2_score(pred.reshape(-1), target.reshape(-1))),
        "n_train_images": int(trained["n_train_images"]),
        "n_val_images": int(trained["n_val_images"]),
        "target_train_mean": trained["target_train_mean"],
        "target_train_std": trained["target_train_std"],
        "history_tail": trained["history"][-5:],
    }


def position_features(n_images: int, n_tokens: int) -> torch.Tensor:
    eye = torch.eye(n_tokens, dtype=torch.float32)
    return eye.unsqueeze(0).expand(n_images, n_tokens, n_tokens).contiguous()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="runs/c1_causal_1000/causal_data.pt")
    parser.add_argument("--top-k", type=int, default=72)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = torch.load(args.data, map_location="cpu")
    importances = data["importances"].float()
    encoder_features = data["features"].float()
    n_images, n_tokens = importances.shape
    if encoder_features.shape[:2] != importances.shape:
        raise SystemExit(f"Feature/target mismatch: {encoder_features.shape} vs {importances.shape}")
    if args.top_k <= 0 or args.top_k > n_tokens:
        raise SystemExit(f"--top-k must be in 1..{n_tokens}")

    ranks = importances.argsort(dim=-1).argsort(dim=-1).float() / float(n_tokens - 1)
    threshold = importances.topk(args.top_k, dim=-1).values[:, -1:]
    topk_binary = (importances >= threshold).float()
    centered = importances - importances.mean(dim=-1, keepdim=True)
    standardized = centered / importances.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)

    target_variants = {
        "raw": importances,
        "rank_normalized": ranks,
        "topk_binary": topk_binary,
        "standardized_per_image": standardized,
    }
    feature_variants = {
        "encoder_full": encoder_features,
        "position_only": position_features(n_images, n_tokens),
    }

    results: dict[str, dict] = {
        "data": args.data,
        "n_images": int(n_images),
        "n_tokens": int(n_tokens),
        "top_k": int(args.top_k),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "results": {},
    }

    for target_name, target_tensor in target_variants.items():
        for feature_name, feature_tensor in feature_variants.items():
            for probe_type in ("linear", "mlp"):
                key = f"{target_name}__{feature_name}__{probe_type}"
                score = score_variant(
                    features=feature_tensor,
                    targets=target_tensor,
                    probe_type=probe_type,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                )
                results["results"][key] = score
                print(f"{key}: R2={score['fixed_split_r2']:+.6f}", flush=True)
                torch.cuda.empty_cache()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
