#!/usr/bin/env python3
"""Analyze spatial selection coverage over the 24x24 CLIP patch grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selection", help=".pt tensor of selected indices, shape [N,k]")
    parser.add_argument("output", help="Output JSON path")
    parser.add_argument("--heatmap", default=None, help="Optional heatmap PDF/PNG path")
    args = parser.parse_args()

    selection = torch.load(args.selection, map_location="cpu").long()
    if selection.ndim != 2:
        raise SystemExit(f"expected [N,k] selection tensor, got {tuple(selection.shape)}")
    n_images, k = selection.shape

    valid = selection[(selection >= 0) & (selection < 576)]
    counts = torch.bincount(valid, minlength=576)
    freq = counts.float() / max(1, n_images)

    result = {
        "n_images": int(n_images),
        "k": int(k),
        "expected_uniform_freq": float(k / 576),
        "max_freq": float(freq.max()),
        "min_freq": float(freq.min()),
        "mean_freq": float(freq.mean()),
        "median_freq": float(freq.median()),
        "mass_balance": float(freq.max() - freq.min()),
        "dead_zone_count": int((freq < 0.02).sum()),
        "hot_spot_count": int((freq > 0.5).sum()),
        "freq_per_position": [float(x) for x in freq.tolist()],
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(
        f"max={result['max_freq']:.3f} min={result['min_freq']:.3f} "
        f"median={result['median_freq']:.3f} "
        f"dead_zones={result['dead_zone_count']} hot_spots={result['hot_spot_count']}"
    )

    if args.heatmap:
        try:
            import matplotlib.pyplot as plt

            grid = freq.reshape(24, 24).numpy()
            fig, ax = plt.subplots(figsize=(4.4, 4.0))
            im = ax.imshow(grid, cmap="magma", vmin=0.0, vmax=max(0.5, float(freq.max())))
            ax.set_title("Selection frequency")
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            heatmap = Path(args.heatmap)
            heatmap.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(heatmap)
            print(f"Wrote {heatmap}")
        except Exception as exc:
            print(f"[warning] heatmap skipped: {exc}")


if __name__ == "__main__":
    main()
