#!/usr/bin/env python3
"""Summarize multi-k TC-ViTok baseline sweeps.

Expected input files follow the Experiment 1 naming convention:
    adaptive_k36_none.json
    adaptive_k36_random_drop.json
    ...

The script writes a CSV with one row per (k, baseline), prints TPN-vs-best
trivial deltas, and optionally produces a three-panel k-scaling plot.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METRICS = {
    "vqav2": ("vqa_score", "VQAv2"),
    "gqa": ("overall_accuracy", "GQA"),
    "pope": ("macro_f1", "POPE"),
}


def load_metric(path: Path) -> dict[str, float]:
    with path.open() as f:
        data = json.load(f)
    benches = data.get("benchmarks", data)
    row: dict[str, float] = {}
    for bench, (key, _) in METRICS.items():
        if bench not in benches or key not in benches[bench]:
            raise KeyError(f"{path} missing benchmarks.{bench}.{key}")
        row[bench] = float(benches[bench][key])
    return row


def write_plot(rows: list[dict[str, object]], output: Path) -> None:
    import matplotlib.pyplot as plt

    labels = {
        "none": "TPN",
        "random_drop": "Random",
        "uniform_grid": "Grid",
        "cls_attention": "CLS-attn",
    }
    markers = {
        "none": "o",
        "random_drop": "s",
        "uniform_grid": "^",
        "cls_attention": "D",
    }
    baselines = ["none", "random_drop", "uniform_grid", "cls_attention"]

    fig, axes = plt.subplots(1, 3, figsize=(9.5, 2.7), sharex=True)
    for ax, (bench, (_, title)) in zip(axes, METRICS.items()):
        for baseline in baselines:
            points = [
                (int(r["k"]), float(r[bench]))
                for r in rows
                if r["baseline"] == baseline
            ]
            points.sort()
            if not points:
                continue
            xs, ys = zip(*points)
            ax.plot(xs, ys, marker=markers[baseline], linewidth=1.4, markersize=4,
                    label=labels[baseline])
        ax.set_title(title)
        ax.set_xlabel("$k$")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Accuracy / F1")
    axes[-1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="runs/multi_k_baselines")
    parser.add_argument("--prefix", default="adaptive")
    parser.add_argument("--ks", nargs="+", type=int, default=[36, 72, 144, 216, 288])
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=["none", "random_drop", "uniform_grid", "cls_attention"],
    )
    parser.add_argument("--csv", default="runs/multi_k_baselines/summary.csv")
    parser.add_argument("--plot", default="figures/k_scaling.pdf")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    rows: list[dict[str, object]] = []
    missing: list[Path] = []

    for k in args.ks:
        for baseline in args.baselines:
            path = input_dir / f"{args.prefix}_k{k}_{baseline}.json"
            if not path.exists():
                missing.append(path)
                continue
            metrics = load_metric(path)
            rows.append({"k": k, "baseline": baseline, **metrics})

    if not rows:
        raise SystemExit(f"No result JSON files found in {input_dir}")

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["k", "baseline", "vqav2", "gqa", "pope"])
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (int(r["k"]), str(r["baseline"]))):
            writer.writerow(row)

    print(f"Wrote {csv_path}")
    if missing:
        print("\nMissing files:")
        for path in missing:
            print(f"  {path}")

    print("\nTPN minus strongest trivial baseline:")
    any_win = False
    for k in args.ks:
        by_baseline = {str(r["baseline"]): r for r in rows if int(r["k"]) == k}
        if "none" not in by_baseline:
            continue
        trivial_rows = [
            by_baseline[b]
            for b in ("random_drop", "uniform_grid", "cls_attention")
            if b in by_baseline
        ]
        if not trivial_rows:
            continue
        for bench in METRICS:
            tpn = float(by_baseline["none"][bench])
            best = max(float(r[bench]) for r in trivial_rows)
            delta = tpn - best
            any_win = any_win or delta > 0
            print(f"  k={k:3d} {bench:5s}: {delta:+.2f} pp")

    if any_win:
        print("\nResult: TPN beats the strongest trivial baseline in at least one cell.")
    else:
        print("\nResult: TPN does not beat the strongest trivial baseline in completed cells.")

    if args.plot:
        try:
            write_plot(rows, Path(args.plot))
            print(f"Wrote {args.plot}")
        except ImportError as exc:
            print(f"Skipping plot because matplotlib is unavailable: {exc}")


if __name__ == "__main__":
    main()
