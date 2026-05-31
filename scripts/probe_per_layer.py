#!/usr/bin/env python3
"""Train causal-importance probes for each saved mid-LLM layer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from probe_with_bootstrap import bootstrap_image_r2, train_probe
from train_causal_probe import r2_score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--causal-data", required=True)
    parser.add_argument("--midllm-features", required=True)
    parser.add_argument("--probe", choices=["linear", "mlp"], default="mlp")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    causal = torch.load(args.causal_data, map_location="cpu")
    targets = causal["importances"].float()
    mid = torch.load(args.midllm_features, map_location="cpu")
    layer_keys = sorted(
        [key for key, value in mid.items() if key.startswith("L") and isinstance(value, torch.Tensor)],
        key=lambda x: int(x[1:]),
    )

    results = {}
    for key in layer_keys:
        features = mid[key].float()
        if features.shape[:2] != targets.shape:
            raise ValueError(f"{key} feature shape {features.shape} mismatches targets {targets.shape}")
        trained = train_probe(
            features=features,
            targets=targets,
            probe_type=args.probe,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=1e-3,
        )
        pred = trained["pred_val"]
        target = trained["target_val"]
        fixed_r2 = r2_score(pred.reshape(-1), target.reshape(-1))
        boot = bootstrap_image_r2(pred, target, args.n_bootstrap, args.seed)
        results[key] = {
            "fixed_split_r2": float(fixed_r2),
            "bootstrap_mean_r2": boot["mean_r2"],
            "ci_95_lower": boot["ci_95_lower"],
            "ci_95_upper": boot["ci_95_upper"],
            "n_val_images": trained["n_val_images"],
            "d_in": trained["d_in"],
        }
        print(f"{key}: R2={fixed_r2:.4f}, boot_mean={boot['mean_r2']:.4f}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "probe": args.probe,
        "causal_data": args.causal_data,
        "midllm_features": args.midllm_features,
        "results": results,
    }, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
