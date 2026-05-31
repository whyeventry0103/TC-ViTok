#!/usr/bin/env python3
"""Train a causal-importance probe and bootstrap R2 over validation images."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train_causal_probe import LinearProbe, MLPProbe, r2_score


def load_features_and_targets(args):
    target_data = torch.load(args.targets_from or args.data, map_location="cpu")
    targets = target_data["importances"].float()
    if args.features_from:
        feature_data = torch.load(args.features_from, map_location="cpu")
        features = feature_data.get("features", feature_data)
        if isinstance(features, dict):
            raise ValueError("--features-from must resolve to a tensor or a dict with key 'features'")
        features = features.float()
    else:
        features = target_data["features"].float()
    if features.shape[:2] != targets.shape:
        raise ValueError(f"Feature/target shape mismatch: {features.shape} vs {targets.shape}")
    return features, targets, target_data


def train_probe(
    features: torch.Tensor,
    targets: torch.Tensor,
    probe_type: str,
    epochs: int,
    batch_size: int,
    lr: float,
):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    n_images, n_tokens, d_in = features.shape
    n_train_images = max(1, int(0.8 * n_images))
    if n_train_images >= n_images:
        n_train_images = max(1, n_images - 1)

    x_train = features[:n_train_images].reshape(-1, d_in)
    y_train = targets[:n_train_images].reshape(-1)
    x_val = features[n_train_images:].reshape(-1, d_in)
    y_val = targets[n_train_images:].reshape(-1)

    y_mean = y_train.mean()
    y_std = y_train.std(unbiased=False).clamp_min(1e-6)
    y_train_z = (y_train - y_mean) / y_std

    loader = DataLoader(
        TensorDataset(x_train, y_train_z),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    model = LinearProbe(d_in) if probe_type == "linear" else MLPProbe(d_in)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    history = []
    for epoch in range(epochs):
        total = 0.0
        n_seen = 0
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = nn.functional.mse_loss(pred, yb)
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu()) * xb.shape[0]
            n_seen += xb.shape[0]
        history.append({"epoch": epoch + 1, "train_mse_z": total / max(1, n_seen)})

    preds = []
    model.eval()
    with torch.no_grad():
        for start in range(0, x_val.shape[0], batch_size):
            xb = x_val[start:start + batch_size].to(device)
            pred_z = model(xb).cpu()
            preds.append(pred_z * y_std + y_mean)
    pred_val = torch.cat(preds, dim=0)
    return {
        "pred_val": pred_val.reshape(n_images - n_train_images, n_tokens),
        "target_val": targets[n_train_images:],
        "n_images": n_images,
        "n_train_images": n_train_images,
        "n_val_images": n_images - n_train_images,
        "d_in": d_in,
        "history": history,
        "target_train_mean": float(y_mean),
        "target_train_std": float(y_std),
    }


def bootstrap_image_r2(pred: torch.Tensor, target: torch.Tensor, n_bootstrap: int, seed: int):
    rng = random.Random(seed)
    n_images = pred.shape[0]
    values = []
    for _ in range(n_bootstrap):
        idx = torch.tensor([rng.randrange(n_images) for _ in range(n_images)], dtype=torch.long)
        values.append(r2_score(pred[idx].reshape(-1), target[idx].reshape(-1)))
    vals = torch.tensor(values, dtype=torch.float32)
    return {
        "n_bootstrap": int(n_bootstrap),
        "mean_r2": float(vals.mean()),
        "std_r2": float(vals.std(unbiased=False)),
        "ci_95_lower": float(torch.quantile(vals, 0.025)),
        "ci_95_upper": float(torch.quantile(vals, 0.975)),
        "all_r2": [float(x) for x in vals.tolist()],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=None, help="Causal .pt with both features and importances")
    parser.add_argument("--features-from", default=None, help="Optional .pt with alternative features")
    parser.add_argument("--targets-from", default=None, help="Optional causal .pt with importances")
    parser.add_argument("--probe", choices=["linear", "mlp"], default="mlp")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if not args.data and not (args.features_from and args.targets_from):
        raise SystemExit("Use --data or both --features-from and --targets-from")

    torch.manual_seed(args.seed)
    features, targets, _ = load_features_and_targets(args)
    trained = train_probe(
        features=features,
        targets=targets,
        probe_type=args.probe,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    pred = trained["pred_val"]
    target = trained["target_val"]
    fixed_r2 = r2_score(pred.reshape(-1), target.reshape(-1))
    boot = bootstrap_image_r2(pred, target, args.n_bootstrap, args.seed)

    result = {
        "probe": args.probe,
        "data": args.data,
        "features_from": args.features_from,
        "targets_from": args.targets_from,
        "n_images": int(trained["n_images"]),
        "n_train_images": int(trained["n_train_images"]),
        "n_val_images": int(trained["n_val_images"]),
        "d_in": int(trained["d_in"]),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "fixed_split_r2": float(fixed_r2),
        "bootstrap": boot,
        "target_train_mean": trained["target_train_mean"],
        "target_train_std": trained["target_train_std"],
        "history_tail": trained["history"][-5:],
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
