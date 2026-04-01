#!/usr/bin/env python3
"""Visualize TPN attention masks on sample images.

Usage:
    python scripts/visualize_masks.py --checkpoint runs/phase1/best.ckpt --images examples/*.jpg
    python scripts/visualize_masks.py --checkpoint runs/phase1/best.ckpt --images img.jpg --questions "What is on the left?"

This script:
1. Loads a trained TPN checkpoint
2. Runs inference on sample images with given questions
3. Visualizes which tokens are selected on a 24x24 grid
4. Saves overlay visualizations (original, mask heatmap, overlay)

Useful for understanding what the TPN has learned and debugging failure modes.

Paper reference: Section 9 (Qualitative Analysis)
"""

import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image


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
        help="Questions to ask (one per image, or one for all)",
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
        default=None,
        help="Device for inference",
    )
    parser.add_argument(
        "--stub",
        action="store_true",
        help="Use StubLLM (for testing without LLaVA)",
    )
    return parser.parse_args()


def detect_device() -> str:
    """Auto-detect the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def visualize_mask(
    image: Image.Image,
    mask_grid: np.ndarray,
    output_path: str,
    question: str = "",
    k: int = 144,
) -> None:
    """Create overlay visualization of selected tokens.

    Produces a 3-panel figure:
    - Left: Original image
    - Center: 24x24 token selection heatmap
    - Right: Overlay of heatmap on original image

    Args:
        image: PIL Image (original resolution)
        mask_grid: Selection mask reshaped to (24, 24)
        output_path: Where to save visualization
        question: Question text for title
        k: Token budget (for annotation)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Panel 2: Mask heatmap (24x24 grid)
    selected_count = int(mask_grid.sum())
    axes[1].imshow(mask_grid, cmap="hot", interpolation="nearest")
    axes[1].set_title(f"Selected Tokens ({selected_count}/{24*24})")
    axes[1].axis("off")

    # Panel 3: Overlay
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

    title = f"Q: {question}" if question else "Token Selection Visualization"
    plt.suptitle(title, fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def idx_to_mask_grid(idx: torch.Tensor, N: int = 576) -> np.ndarray:
    """Convert gather indices to a 24x24 spatial mask grid.

    Args:
        idx: Top-k indices, Shape: [k] (single sample)
        N: Total tokens (576)

    Returns:
        mask_grid: Binary mask reshaped to (24, 24)
    """
    mask = np.zeros(N, dtype=np.float32)
    mask[idx.cpu().numpy()] = 1.0
    grid_size = int(N ** 0.5)  # 24
    return mask.reshape(grid_size, grid_size)


def main():
    args = parse_args()

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    device = args.device or detect_device()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print(f"Token budget: k={args.k}")

    # Load model
    from tc_vitok.models.tc_vitok_model import TCViTokModel

    checkpoint = torch.load(
        args.checkpoint, map_location=device, weights_only=False
    )
    ckpt_config = checkpoint.get("config", {})
    model_config = ckpt_config.get("model", {})

    model = TCViTokModel(
        k=args.k,
        tpn_config=model_config.get("tpn", {}),
        use_stub_llm=args.stub,
        lora_config=model_config.get("lora", {}),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device)
    model.eval()

    # Prepare transforms
    from tc_vitok.data.datasets import DINO_TRANSFORM, CLIP_TRANSFORM

    # Default question if not provided
    default_question = "What is in this image?"
    questions = args.questions or [default_question] * len(args.images)
    if len(questions) == 1 and len(args.images) > 1:
        questions = questions * len(args.images)

    assert len(questions) == len(args.images), (
        f"Got {len(questions)} questions for {len(args.images)} images"
    )

    # Process each image
    for i, (img_path, question) in enumerate(zip(args.images, questions)):
        print(f"\nProcessing {img_path}...")
        print(f"  Question: {question}")

        image = Image.open(img_path).convert("RGB")

        # Prepare inputs
        img_coarse = DINO_TRANSFORM(image).unsqueeze(0).to(device)
        img_full = CLIP_TRANSFORM(image).unsqueeze(0).to(device)

        # Run TPN to get selection indices
        with torch.no_grad():
            outputs = model(
                images_coarse=img_coarse,
                images_full=img_full,
                questions=[question],
                return_idx=True,
            )
            idx = outputs["idx"][0]  # [k] for first sample

        # Convert to spatial mask grid
        mask_grid = idx_to_mask_grid(idx, N=576)

        # Save visualization
        output_path = output_dir / f"mask_{i:03d}.png"
        visualize_mask(image, mask_grid, str(output_path), question, k=args.k)
        print(f"  Saved to {output_path}")

    print(f"\nAll visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()
