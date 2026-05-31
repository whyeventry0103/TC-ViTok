#!/usr/bin/env python3
"""Prepare and summarize lightweight Track C control experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def save_position_targets(causal_path: Path, out_dir: Path) -> None:
    causal = torch.load(causal_path, map_location="cpu")
    importances = causal["importances"]
    n_images, n_tokens = importances.shape
    side = int(round(n_tokens ** 0.5))
    if side * side != n_tokens:
        raise ValueError(f"Expected square visual grid, got {n_tokens} tokens")

    row = torch.arange(side, dtype=torch.float32).view(side, 1).expand(side, side).reshape(-1)
    col = torch.arange(side, dtype=torch.float32).view(1, side).expand(side, side).reshape(-1)
    row = (row / max(1, side - 1)).repeat(n_images, 1)
    col = (col / max(1, side - 1)).repeat(n_images, 1)

    out_dir.mkdir(parents=True, exist_ok=True)
    common = {
        "metadata": causal.get("metadata"),
        "source": str(causal_path),
        "description": "Positive-control spatial coordinate targets on the 24x24 visual grid.",
    }
    torch.save({**common, "importances": row}, out_dir / "position_row_targets.pt")
    torch.save({**common, "importances": col}, out_dir / "position_col_targets.pt")
    print(f"Wrote {out_dir / 'position_row_targets.pt'}")
    print(f"Wrote {out_dir / 'position_col_targets.pt'}")


def save_permuted_features(causal_path: Path, out_dir: Path, seed: int) -> None:
    causal = torch.load(causal_path, map_location="cpu")
    features = causal["features"].clone()
    generator = torch.Generator().manual_seed(seed)
    for i in range(features.shape[0]):
        perm = torch.randperm(features.shape[1], generator=generator)
        features[i] = features[i, perm]
    out = {
        "features": features,
        "metadata": causal.get("metadata"),
        "source": str(causal_path),
        "seed": seed,
        "description": "CLIP features with visual-token order independently permuted within each image.",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_dir / "clip_features_token_permuted.pt")
    print(f"Wrote {out_dir / 'clip_features_token_permuted.pt'}")


def rank_rows(x: np.ndarray) -> np.ndarray:
    """Tie-aware average ranks, equivalent to scipy.stats.rankdata(method='average')."""
    ranks = np.empty(x.shape, dtype=np.float32)
    for i, row in enumerate(x):
        order = np.argsort(row, kind="mergesort")
        sorted_row = row[order]
        row_ranks = np.empty(row.shape[0], dtype=np.float32)
        start = 0
        while start < row.shape[0]:
            end = start + 1
            while end < row.shape[0] and sorted_row[end] == sorted_row[start]:
                end += 1
            avg_rank = 0.5 * (start + end - 1)
            row_ranks[order[start:end]] = avg_rank
            start = end
        ranks[i] = row_ranks
    return ranks


def rowwise_spearman(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ar = rank_rows(a)
    br = rank_rows(b)
    ar = ar - ar.mean(axis=1, keepdims=True)
    br = br - br.mean(axis=1, keepdims=True)
    denom = np.sqrt((ar * ar).sum(axis=1) * (br * br).sum(axis=1))
    return (ar * br).sum(axis=1) / np.maximum(denom, 1e-12)


def c4_permutation_test(attn_paths: list[Path], targets_path: Path, output: Path, n_perm: int, seed: int) -> None:
    targets = torch.load(targets_path, map_location="cpu")["importances"].float().numpy()
    rng = np.random.default_rng(seed)
    results = {}

    for path in attn_paths:
        data = torch.load(path, map_location="cpu")
        for key, value in data.items():
            if not key.endswith("_mean"):
                continue
            attn = value.float().numpy()
            n = min(attn.shape[0], targets.shape[0])
            a = attn[:n]
            y = targets[:n]
            observed = rowwise_spearman(a, y)
            null_means = []
            for _ in range(n_perm):
                shuffled = np.empty_like(a)
                for i in range(n):
                    shuffled[i] = a[i, rng.permutation(a.shape[1])]
                null_means.append(float(rowwise_spearman(shuffled, y).mean()))
            null = np.array(null_means, dtype=np.float32)
            obs_mean = float(observed.mean())
            p_two_sided = float((np.abs(null) >= abs(obs_mean)).mean())
            results[key] = {
                "attention_file": str(path),
                "n_images": int(n),
                "n_tokens": int(a.shape[1]),
                "observed_mean_spearman": obs_mean,
                "observed_abs_mean_spearman": float(np.abs(observed).mean()),
                "observed_median_spearman": float(np.median(observed)),
                "observed_std_spearman": float(observed.std()),
                "null_mean": float(null.mean()),
                "null_std": float(null.std()),
                "null_ci_95": [float(np.percentile(null, 2.5)), float(np.percentile(null, 97.5))],
                "p_two_sided": p_two_sided,
                "n_permutations": int(n_perm),
                "seed": int(seed),
            }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


def read_probe(path: Path) -> dict:
    data = json.loads(path.read_text())
    boot = data.get("bootstrap", {})
    return {
        "fixed_split_r2": data.get("fixed_split_r2"),
        "mean_r2": boot.get("mean_r2"),
        "ci_95_lower": boot.get("ci_95_lower"),
        "ci_95_upper": boot.get("ci_95_upper"),
    }


def summarize(out_dir: Path) -> None:
    summary = {"positive_controls": {}, "permuted_feature_null": {}, "c4_permutation": {}}
    for target in ["row", "col"]:
        for probe in ["linear", "mlp"]:
            path = out_dir / f"positive_{target}_{probe}.json"
            if path.exists():
                summary["positive_controls"][f"{target}_{probe}"] = read_probe(path)
    for probe in ["linear", "mlp"]:
        path = out_dir / f"permuted_features_{probe}.json"
        if path.exists():
            summary["permuted_feature_null"][probe] = read_probe(path)
    c4 = out_dir / "c4_spearman_permutation.json"
    if c4.exists():
        summary["c4_permutation"] = json.loads(c4.read_text())
    out = out_dir / "trackc_controls_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    prep = sub.add_parser("prepare")
    prep.add_argument("--causal-data", default="runs/c1_causal_1000/causal_data.pt")
    prep.add_argument("--output-dir", default="runs/trackc_controls")
    prep.add_argument("--seed", type=int, default=42)

    perm = sub.add_parser("c4-permutation")
    perm.add_argument("--attention", nargs="+", default=[
        "runs/c4_attention/smoke_attn_n100.pt",
        "runs/c4_attention/smoke_attn_L1_n100.pt",
    ])
    perm.add_argument("--targets", default="runs/c1_causal_1000/causal_data.pt")
    perm.add_argument("--output", default="runs/trackc_controls/c4_spearman_permutation.json")
    perm.add_argument("--n-permutations", type=int, default=1000)
    perm.add_argument("--seed", type=int, default=42)

    summ = sub.add_parser("summarize")
    summ.add_argument("--output-dir", default="runs/trackc_controls")

    args = parser.parse_args()
    if args.cmd == "prepare":
        save_position_targets(Path(args.causal_data), Path(args.output_dir))
        save_permuted_features(Path(args.causal_data), Path(args.output_dir), args.seed)
    elif args.cmd == "c4-permutation":
        c4_permutation_test(
            [Path(p) for p in args.attention],
            Path(args.targets),
            Path(args.output),
            args.n_permutations,
            args.seed,
        )
    elif args.cmd == "summarize":
        summarize(Path(args.output_dir))


if __name__ == "__main__":
    main()
