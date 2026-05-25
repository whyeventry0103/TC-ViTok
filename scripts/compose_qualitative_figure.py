#!/usr/bin/env python3
"""Compose qualitative_real.pdf from dump_tpn_heatmaps.py output.

Reads overlay PNGs and manifest from a dump directory and lays them out
in a 4-row grid with question / GT answer / model prediction annotations.

Usage:
    python scripts/compose_qualitative_figure.py \
        --dump-dir runs/qualitative_dump \
        --out "DL Project Report/figures/qualitative_real.pdf" \
        --n-examples 6
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


def pick_diverse(manifest, n):
    """Return n entries spread evenly across the manifest for visual diversity."""
    if len(manifest) <= n:
        return manifest
    step = len(manifest) // n
    return [manifest[i * step] for i in range(n)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dump-dir", default="runs/qualitative_dump")
    p.add_argument("--out", default="DL Project Report/figures/qualitative_real.pdf")
    p.add_argument("--n-examples", type=int, default=6)
    args = p.parse_args()

    dump_dir = Path(args.dump_dir)
    manifest_path = dump_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"manifest.json not found in {dump_dir}. "
            "Run dump_tpn_heatmaps.py first."
        )

    with open(manifest_path) as f:
        manifest = json.load(f)

    entries = pick_diverse(manifest, args.n_examples)
    n = len(entries)

    fig, axes = plt.subplots(n, 2, figsize=(7.0, n * 2.2))
    if n == 1:
        axes = [axes]

    for row, entry in enumerate(entries):
        npz_path = Path(entry["npz"])
        overlay_path = npz_path.parent / (npz_path.stem + "_overlay.png")

        data = np.load(npz_path, allow_pickle=True)

        # Load raw image from path stored in npz
        img_path = str(data["image_path"])
        raw_img = np.array(Image.open(img_path).convert("RGB"))

        # Left column: raw image
        axes[row][0].imshow(raw_img)
        axes[row][0].axis("off")
        if row == 0:
            axes[row][0].set_title("Input Image", fontsize=8, fontweight="bold")

        # Right column: TPN score overlay (render-overlays png) or heatmap fallback
        if overlay_path.exists():
            axes[row][1].imshow(np.array(Image.open(overlay_path).convert("RGB")))
        else:
            scores = data["scores"].astype(float)
            if scores.ndim == 1:
                side = int(round(scores.shape[0] ** 0.5))
                scores = scores[: side * side].reshape(side, side)
            # Resize heatmap to match image for overlay
            from PIL import Image as PILImage
            h, w = raw_img.shape[:2]
            heat = PILImage.fromarray(
                ((scores - scores.min()) / (scores.max() - scores.min() + 1e-8) * 255).astype(np.uint8)
            ).resize((w, h), resample=PILImage.BILINEAR)
            axes[row][1].imshow(raw_img, alpha=0.5)
            axes[row][1].imshow(np.array(heat), cmap="hot", alpha=0.6)
        axes[row][1].axis("off")
        if row == 0:
            axes[row][1].set_title("TPN Score Map (top-k selected)", fontsize=8, fontweight="bold")

        # Caption strip below panels
        q = str(entry.get("question", ""))[:72]
        gt = str(entry.get("answer", ""))
        pred = str(entry.get("pred", ""))
        match = "✓" if gt.lower().strip() == pred.lower().strip() else "✗"
        caption = f"Q: {q}\nGT: {gt}   Pred: {pred} {match}"
        axes[row][0].text(
            0.0, -0.04, caption,
            transform=axes[row][0].transAxes,
            fontsize=6, va="top", wrap=True,
            color="#222222",
        )

    plt.suptitle("TPN token selection on GQA val\\_balanced (k=144, Phase 1 Final)", fontsize=9, y=1.01)
    plt.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out), bbox_inches="tight")
    print(f"Saved: {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
