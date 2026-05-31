#!/usr/bin/env python3
"""Aggregate per-seed evaluation JSONs into mean/std benchmark metrics."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


HEADLINE = {
    "vqav2": "vqa_score",
    "gqa": "overall_accuracy",
    "pope": "macro_f1",
    "mmbench": "accuracy",
    "mme": "perception_total",
    "textvqa": "vqa_score",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="Directory containing seed*/eval.json files")
    args = parser.parse_args()

    files = sorted(Path(args.run_dir).glob("seed*/eval.json"))
    if not files:
        raise SystemExit(f"No eval.json files found under {args.run_dir}/seed*/eval.json")

    values: dict[str, list[float]] = {}
    for path in files:
        data = json.load(open(path))
        benches = data.get("benchmarks", data)
        for bench, key in HEADLINE.items():
            if bench in benches and key in benches[bench]:
                values.setdefault(bench, []).append(float(benches[bench][key]))

    print(f"Aggregated over {len(files)} seed files")
    print(f"{'benchmark':<12} {'mean':>10} {'std':>10} {'n':>4}")
    for bench, vals in values.items():
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(f"{bench:<12} {mean:>10.3f} {std:>10.3f} {len(vals):>4}")


if __name__ == "__main__":
    main()
