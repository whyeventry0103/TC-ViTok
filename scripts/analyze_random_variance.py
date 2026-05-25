#!/usr/bin/env python3
"""Compute random-drop seed variance for TC-ViTok evaluations."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


OUT = Path("runs/random_seed_variance")
TPN_REF = {"vqav2": 72.96, "gqa": 68.98, "pope": 79.19}
METRICS = {
    "vqav2": "vqa_score",
    "gqa": "overall_accuracy",
    "pope": "macro_f1",
}


def read_metric(path: Path, bench: str) -> float:
    data = json.load(path.open())
    benches = data.get("benchmarks", data)
    if bench not in benches:
        raise KeyError(f"{path} missing benchmark {bench}")
    value = benches[bench]
    if isinstance(value, dict):
        key = METRICS[bench]
        return float(value[key])
    return float(value)


def main() -> None:
    rows = []
    per_bench = {bench: [] for bench in METRICS}
    for path in sorted(OUT.glob("random_seed*.json")):
        if path.name.endswith(".per_question.json"):
            continue
        data = json.load(path.open())
        seed = int(data.get("seed", path.stem.replace("random_seed", "")))
        row = {"seed": seed}
        for bench in METRICS:
            row[bench] = read_metric(path, bench)
            per_bench[bench].append(row[bench])
        rows.append(row)

    if not rows:
        raise SystemExit(f"No random_seed*.json files found in {OUT}")

    summary_path = OUT / "summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "vqav2", "gqa", "pope"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {summary_path}")
    print(f"{'Benchmark':<10} {'Random mean':>12} {'Random std':>12} {'TPN':>8} {'Gap to mean':>12}")
    for bench, vals in per_bench.items():
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) >= 2 else 0.0
        tpn = TPN_REF[bench]
        print(f"{bench:<10} {mean:>12.3f} {std:>12.3f} {tpn:>8.3f} {tpn - mean:>+12.3f}")

    print("\nInterpretation:")
    print("- Random std measures selection-induced variance across random-drop seeds.")
    print("- If |TPN - random mean| < random std, TPN is within one random-seed sigma.")


if __name__ == "__main__":
    main()
