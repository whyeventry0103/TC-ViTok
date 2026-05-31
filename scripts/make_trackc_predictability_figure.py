#!/usr/bin/env python3
"""Create Track C predictability curve/table from probe JSON artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def add_point(points, label, source, path):
    if not path:
        return
    data = json.load(open(path))
    if "fixed_split_r2" in data:
        r2 = data["fixed_split_r2"]
        lo = data.get("bootstrap", {}).get("ci_95_lower")
        hi = data.get("bootstrap", {}).get("ci_95_upper")
    else:
        r2 = data["r2"]
        lo = hi = None
    points.append({"label": label, "source": source, "r2": r2, "lo": lo, "hi": hi, "path": path})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", default=None)
    parser.add_argument("--siglip", default=None)
    parser.add_argument("--dinov2", default=None)
    parser.add_argument("--per-layer", default=None)
    parser.add_argument("--output-pdf", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    points = []
    add_point(points, "CLIP encoder", "encoder", args.clip)
    add_point(points, "SigLIP encoder", "encoder", args.siglip)
    add_point(points, "DINOv2 encoder", "encoder", args.dinov2)

    if args.per_layer:
        per = json.load(open(args.per_layer))["results"]
        for key in sorted(per, key=lambda x: int(x[1:])):
            item = per[key]
            points.append({
                "label": key,
                "source": "llm_layer",
                "r2": item["fixed_split_r2"],
                "lo": item.get("ci_95_lower"),
                "hi": item.get("ci_95_upper"),
                "path": args.per_layer,
            })

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(points, indent=2))

    labels = [p["label"] for p in points]
    ys = [p["r2"] for p in points]
    xs = list(range(len(points)))
    colors = ["#4C78A8" if p["source"] == "encoder" else "#F58518" for p in points]
    yerr_lower = [0 if p["lo"] is None else max(0, p["r2"] - p["lo"]) for p in points]
    yerr_upper = [0 if p["hi"] is None else max(0, p["hi"] - p["r2"]) for p in points]

    plt.figure(figsize=(max(8, len(points) * 0.7), 4.5))
    plt.axhline(0.0, color="black", linewidth=1, alpha=0.5)
    plt.axhline(0.2, color="gray", linewidth=1, linestyle="--", alpha=0.6)
    plt.bar(xs, ys, color=colors)
    if any(v > 0 for v in yerr_lower + yerr_upper):
        plt.errorbar(xs, ys, yerr=[yerr_lower, yerr_upper], fmt="none", ecolor="black", capsize=3)
    plt.xticks(xs, labels, rotation=35, ha="right")
    plt.ylabel("Held-out probe R2")
    plt.title("Causal Token-Importance Predictability")
    plt.tight_layout()
    out_pdf = Path(args.output_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_pdf)
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
