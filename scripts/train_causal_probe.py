#!/usr/bin/env python3
"""Train a probe to predict causal token importance from CLIP token features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class LinearProbe(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.linear = nn.Linear(d_in, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


class MLPProbe(nn.Module):
    def __init__(self, d_in: int, d_hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def r2_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    ss_res = ((target - pred) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    if float(ss_tot) <= 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--probe", choices=["linear", "mlp"], default="mlp")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", required=True)
    parser.add_argument("--d4-output", default=None, help="If set, write val R2 here.")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    data = torch.load(args.data, map_location="cpu")
    features = data["features"].float()
    targets = data["importances"].float()
    n_images, n_tokens, d_in = features.shape
    n_train_images = max(1, int(0.8 * n_images))
    if n_train_images >= n_images:
        n_train_images = max(1, n_images - 1)

    x_train = features[:n_train_images].reshape(-1, d_in)
    y_train = targets[:n_train_images].reshape(-1)
    x_val = features[n_train_images:].reshape(-1, d_in)
    y_val = targets[n_train_images:].reshape(-1)

    # Standardize target using train split only; R2 is invariant to affine target
    # scaling after inverse-transforming predictions, but this stabilizes training.
    y_mean = y_train.mean()
    y_std = y_train.std(unbiased=False).clamp_min(1e-6)
    y_train_z = (y_train - y_mean) / y_std
    y_val_z = (y_val - y_mean) / y_std

    loader = DataLoader(
        TensorDataset(x_train, y_train_z),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    model = LinearProbe(d_in) if args.probe == "linear" else MLPProbe(d_in)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    history = []
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        n_seen = 0
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

    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, x_val.shape[0], args.batch_size):
            xb = x_val[start:start + args.batch_size].to(device)
            preds.append(model(xb).cpu())
    pred_z = torch.cat(preds, dim=0)
    pred = pred_z * y_std + y_mean
    val_mse = float(nn.functional.mse_loss(pred, y_val))
    val_r2 = r2_score(pred, y_val)
    val_r2_z = r2_score(pred_z, y_val_z)

    result = {
        "probe": args.probe,
        "data": args.data,
        "n_images": int(n_images),
        "n_train_images": int(n_train_images),
        "n_val_images": int(n_images - n_train_images),
        "n_train_tokens": int(x_train.shape[0]),
        "n_val_tokens": int(x_val.shape[0]),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "val_mse": val_mse,
        "r2": val_r2,
        "r2_standardized": val_r2_z,
        "target_train_mean": float(y_mean),
        "target_train_std": float(y_std),
        "history_tail": history[-5:],
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

    if args.d4_output:
        d4 = Path(args.d4_output)
        d4.parent.mkdir(parents=True, exist_ok=True)
        d4.write_text(f"{val_r2:.6f}\n")
        print(f"Wrote {d4}")


if __name__ == "__main__":
    main()
