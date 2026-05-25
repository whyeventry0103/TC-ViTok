#!/usr/bin/env python3
"""Fit accuracy ~ log2(k) for each selector and benchmark.

Slope is reported in percentage points per log2(k). Larger positive slope
means the selector benefits more from additional tokens; equivalently, it
degrades more sharply as the token budget is reduced.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict


BENCHMARKS = ["vqav2", "gqa", "pope"]


def linreg(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    ss_xx = sum((x - mean_x) ** 2 for x in xs)
    ss_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = ss_xy / ss_xx if ss_xx else 0.0
    intercept = mean_y - slope * mean_x
    preds = [slope * x + intercept for x in xs]
    ss_res = sum((y - p) ** 2 for y, p in zip(ys, preds))
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 0.0
    return slope, intercept, r2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default="runs/multi_k_baselines/summary.csv")
    args = parser.parse_args()

    rows = list(csv.DictReader(open(args.summary)))
    if not rows:
        raise SystemExit(f"No rows in {args.summary}")

    baselines = sorted({r["baseline"] for r in rows})
    slopes: dict[str, dict[str, tuple[float, float]]] = defaultdict(dict)

    print(f"{'baseline':<18} {'benchmark':<10} {'slope pp/log2(k)':>18} {'R^2':>8}")
    print("-" * 60)
    for baseline in baselines:
        for bench in BENCHMARKS:
            points = [
                (math.log2(int(r["k"])), float(r[bench]))
                for r in rows
                if r["baseline"] == baseline and r.get(bench)
            ]
            points.sort()
            if len(points) < 3:
                continue
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            slope, _, r2 = linreg(xs, ys)
            slopes[bench][baseline] = (slope, r2)
            print(f"{baseline:<18} {bench:<10} {slope:>+18.3f} {r2:>8.3f}")

    print("\nRobustness ranking by absolute slope (left = shallowest):")
    for bench in BENCHMARKS:
        ranking = sorted(slopes[bench].items(), key=lambda item: abs(item[1][0]))
        print(f"{bench:<10} " + " < ".join(name for name, _ in ranking))

    print("\nInterpretation:")
    print("- Shallower absolute slope means less sensitivity to token budget.")
    print("- Use only if TPN is shallowest, or clearly shallower than random, on a benchmark.")


if __name__ == "__main__":
    main()
