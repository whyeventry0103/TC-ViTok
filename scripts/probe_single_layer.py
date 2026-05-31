#!/usr/bin/env python3
"""Train linear and MLP probes for one saved mid-LLM layer feature file."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from probe_with_bootstrap import bootstrap_image_r2, train_probe
from train_causal_probe import r2_score


def score_one_probe(features, targets, probe_type, epochs, batch_size, n_bootstrap, seed):
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
    fixed_r2 = r2_score(pred.reshape(-1), target.reshape(-1))
    boot = bootstrap_image_r2(pred, target, n_bootstrap, seed)
    return {
        "fixed_split_r2": float(fixed_r2),
        "n_bootstrap": boot["n_bootstrap"],
        "mean_r2": boot["mean_r2"],
        "std_r2": boot["std_r2"],
        "ci_95_lower": boot["ci_95_lower"],
        "ci_95_upper": boot["ci_95_upper"],
        "n_train_images": trained["n_train_images"],
        "n_val_images": trained["n_val_images"],
        "history_tail": trained["history"][-5:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-from", required=True)
    parser.add_argument("--targets-from", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--n-bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-append", required=True)
    parser.add_argument("--delete-features-after", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    features_data = torch.load(args.features_from, map_location="cpu")
    key = f"L{args.layer}"
    if key not in features_data:
        raise SystemExit(f"{args.features_from} does not contain {key}; keys={list(features_data.keys())}")
    features = features_data[key].float()

    targets_data = torch.load(args.targets_from, map_location="cpu")
    targets = targets_data["importances"].float()
    if features.shape[:2] != targets.shape:
        raise SystemExit(f"Feature/target shape mismatch: {features.shape} vs {targets.shape}")

    print(f"Layer {key} features: shape={tuple(features.shape)}, dtype={features.dtype}")
    print(f"Targets: shape={tuple(targets.shape)}, dtype={targets.dtype}")

    linear = score_one_probe(
        features, targets, "linear", args.epochs, args.batch_size, args.n_bootstrap, args.seed
    )
    print(f"{key} linear fixed R2={linear['fixed_split_r2']:.6f}, mean={linear['mean_r2']:.6f}")

    mlp = score_one_probe(
        features, targets, "mlp", args.epochs, args.batch_size, args.n_bootstrap, args.seed
    )
    print(f"{key} mlp fixed R2={mlp['fixed_split_r2']:.6f}, mean={mlp['mean_r2']:.6f}")

    entry = {
        "layer": args.layer,
        "key": key,
        "features_from": args.features_from,
        "targets_from": args.targets_from,
        "n_images": int(features.shape[0]),
        "n_tokens": int(features.shape[1]),
        "d": int(features.shape[2]),
        "vis_start": features_data.get("vis_start"),
        "vis_end": features_data.get("vis_end"),
        "linear": linear,
        "mlp": mlp,
    }

    out = Path(args.output_append)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        log = json.loads(out.read_text())
        log = [item for item in log if int(item.get("layer", -1)) != args.layer]
    else:
        log = []
    log.append(entry)
    log.sort(key=lambda item: int(item["layer"]))
    out.write_text(json.dumps(log, indent=2))
    print(json.dumps(entry, indent=2))
    print(f"Appended L{args.layer} results to {out}")

    with open("runs/TRACKC_LOG.md", "a") as f:
        f.write(
            f"C2 L{args.layer}: "
            f"linear R2={linear['mean_r2']:+.6f} "
            f"[{linear['ci_95_lower']:+.6f},{linear['ci_95_upper']:+.6f}], "
            f"MLP R2={mlp['mean_r2']:+.6f} "
            f"[{mlp['ci_95_lower']:+.6f},{mlp['ci_95_upper']:+.6f}]\n"
        )

    if args.delete_features_after:
        os.remove(args.features_from)
        print(f"Deleted {args.features_from}")


if __name__ == "__main__":
    main()
