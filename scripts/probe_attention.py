#!/usr/bin/env python3
"""Analyze attention features against causal token importance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from probe_with_bootstrap import bootstrap_image_r2, train_probe
from train_causal_probe import r2_score


def rankdata(x: torch.Tensor) -> torch.Tensor:
    return x.argsort(dim=-1).argsort(dim=-1).float()


def spearman_per_image(scores: torch.Tensor, targets: torch.Tensor) -> dict:
    score_ranks = rankdata(scores)
    target_ranks = rankdata(targets)
    score_centered = score_ranks - score_ranks.mean(dim=1, keepdim=True)
    target_centered = target_ranks - target_ranks.mean(dim=1, keepdim=True)
    denom = (
        score_centered.square().sum(dim=1).sqrt()
        * target_centered.square().sum(dim=1).sqrt()
    ).clamp_min(1e-12)
    rho = (score_centered * target_centered).sum(dim=1) / denom
    return {
        "n": int(rho.numel()),
        "mean_spearman": float(rho.mean()),
        "median_spearman": float(rho.median()),
        "std_spearman": float(rho.std(unbiased=False)),
        "abs_mean": float(rho.abs().mean()),
        "ci_95_lower": float(torch.quantile(rho, 0.025)),
        "ci_95_upper": float(torch.quantile(rho, 0.975)),
    }


def score_probe(
    features: torch.Tensor,
    targets: torch.Tensor,
    probe_type: str,
    epochs: int,
    batch_size: int,
    n_bootstrap: int,
    seed: int,
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
    parser.add_argument("--attention", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--n-bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-append", required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    attn_data = torch.load(args.attention, map_location="cpu")
    target_data = torch.load(args.targets, map_location="cpu")
    targets = target_data["importances"].float()

    mean_key = f"L{args.layer}_mean"
    perhead_key = f"L{args.layer}_perhead"
    if mean_key not in attn_data or perhead_key not in attn_data:
        raise SystemExit(f"Missing {mean_key}/{perhead_key}; keys={list(attn_data.keys())}")

    mean_scores = attn_data[mean_key].float()
    perhead_features = attn_data[perhead_key].float()
    n = min(mean_scores.shape[0], targets.shape[0])
    mean_scores = mean_scores[:n]
    perhead_features = perhead_features[:n]
    targets = targets[:n]
    if mean_scores.shape != targets.shape:
        raise SystemExit(f"Mean attention/target mismatch: {mean_scores.shape} vs {targets.shape}")
    if perhead_features.shape[:2] != targets.shape:
        raise SystemExit(f"Per-head attention/target mismatch: {perhead_features.shape} vs {targets.shape}")

    spearman = spearman_per_image(mean_scores, targets)
    print(f"L{args.layer} mean attention Spearman: {spearman}")

    linear = score_probe(
        perhead_features,
        targets,
        "linear",
        args.epochs,
        args.batch_size,
        args.n_bootstrap,
        args.seed,
    )
    print(f"L{args.layer} attention linear fixed R2={linear['fixed_split_r2']:.6f}, mean={linear['mean_r2']:.6f}")

    mlp = score_probe(
        perhead_features,
        targets,
        "mlp",
        args.epochs,
        args.batch_size,
        args.n_bootstrap,
        args.seed,
    )
    print(f"L{args.layer} attention mlp fixed R2={mlp['fixed_split_r2']:.6f}, mean={mlp['mean_r2']:.6f}")

    entry = {
        "layer": args.layer,
        "attention": args.attention,
        "targets": args.targets,
        "n_images": int(n),
        "n_tokens": int(targets.shape[1]),
        "n_heads": int(perhead_features.shape[2]),
        "spearman_attention_vs_importance": spearman,
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
            f"C4 L{args.layer}: spearman_abs={spearman['abs_mean']:.6f} "
            f"linear R2={linear['mean_r2']:+.6f} "
            f"MLP R2={mlp['mean_r2']:+.6f}\n"
        )


if __name__ == "__main__":
    main()
