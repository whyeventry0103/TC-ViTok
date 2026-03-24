#!/usr/bin/env python3
"""Visualize TPN attention masks on sample images.

Usage:
    python scripts/visualize_masks.py --checkpoint runs/phase1/best.ckpt --images examples/*.jpg

This script:
1. Loads a trained TPN checkpoint
2. Runs inference on sample images
3. Visualizes which tokens are selected
4. Saves overlay visualizations

Useful for understanding what the TPN has learned and debugging failure modes.

Paper reference: Section 9 (Qualitative Analysis)
"""

import argparse
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# TODO (Phase D): Import TC-ViTok modules
# from tc_vitok.models import TCViTokModel


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize TPN masks")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--images",
        nargs="+",
        required=True,
        help="Paths to images to visualize",
    )
    parser.add_argument(
        "--questions",
        nargs="+",
        default=None,
        help="Questions to ask (one per image, or default question)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="visualizations",
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=144,
        help="Token budget",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference",
    )
    return parser.parse_args()


def visualize_mask(image, mask, output_path, question=""):
    """Create overlay visualization of selected tokens.

    Args:
        image: PIL Image
        mask: Selection mask [576] (24×24 grid)
        output_path: Where to save visualization
        question: Question text for title
    """
    # TODO (Phase D): Implement mask visualization
    # 1. Reshape mask to 24×24 spatial grid
    # 2. Upsample to image resolution
    # 3. Create heatmap overlay
    # 4. Save composite image

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Mask heatmap
    mask_grid = mask.reshape(24, 24)
    axes[1].imshow(mask_grid, cmap="hot", interpolation="nearest")
    axes[1].set_title(f"Selected Tokens (k={mask.sum():.0f})")
    axes[1].axis("off")

    # Overlay
    axes[2].imshow(image)
    axes[2].imshow(
        mask_grid,
        cmap="hot",
        alpha=0.5,
        interpolation="bilinear",
        extent=[0, image.width, image.height, 0],
    )
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.suptitle(f"Q: {question}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {args.checkpoint}")
    # TODO (Phase D): Load model
    # model = TCViTokModel(k=args.k)
    # checkpoint = torch.load(args.checkpoint, map_location=args.device)
    # model.load_state_dict(checkpoint["model_state_dict"])
    # model.eval()

    # Default question if not provided
    default_question = "What is in this image?"
    questions = args.questions or [default_question] * len(args.images)

    if len(questions) == 1 and len(args.images) > 1:
        questions = questions * len(args.images)

    # Process each image
    for i, (img_path, question) in enumerate(zip(args.images, questions)):
        print(f"\nProcessing {img_path}...")
        print(f"Question: {question}")

        # TODO (Phase D): Run inference
        # image = Image.open(img_path).convert("RGB")
        # mask, scores = model.get_mask(image, question)

        # TODO (Phase D): Visualize
        output_path = output_dir / f"mask_{i:03d}.png"
        # visualize_mask(image, mask, output_path, question)

        print(f"Saved to {output_path}")

    print(f"\nAll visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()
