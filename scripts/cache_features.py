#!/usr/bin/env python3
"""Precompute and cache visual and text features to HDF5.

Usage:
    python scripts/cache_features.py --dataset vqav2 --split train --output data/cache/
    python scripts/cache_features.py --dataset gqa --split val --output data/cache/ --device mps

This significantly speeds up training by avoiding repeated encoding of:
- DINOv2-S coarse visual features [49, 384]
- CLIP text embeddings [512]

Full visual features (CLIP ViT-L/14) are NOT cached due to size.

Paper reference: Section 7.4 (Implementation Details)
"""

import argparse
import os
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Cache features to HDF5")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["vqav2", "gqa", "textvqa"],
        help="Dataset to cache",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split (train/val/test)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for HDF5 cache files",
    )
    parser.add_argument(
        "--annotations-path",
        type=str,
        default=None,
        help="Path to annotation file (VQAv2 only)",
    )
    parser.add_argument(
        "--questions-path",
        type=str,
        default=None,
        help="Path to questions file",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="Directory containing images",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for feature extraction (auto-detected if not specified)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for feature extraction",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of samples (for testing)",
    )
    return parser.parse_args()


def detect_device() -> str:
    """Auto-detect the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    args = parse_args()

    # MPS/Apple Silicon compatibility
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    device = args.device or detect_device()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / f"{args.dataset}_{args.split}.h5"

    print(f"Caching {args.dataset} {args.split} to {cache_path}")
    print(f"Device: {device}")

    # Load frozen encoders
    from tc_vitok.models.encoders import CoarseVisualEncoder, TextEncoder

    print("Loading DINOv2-S coarse encoder...")
    coarse_encoder = CoarseVisualEncoder().to(device).eval()

    print("Loading CLIP text encoder...")
    text_encoder = TextEncoder().to(device).eval()

    # Load dataset
    print(f"Loading {args.dataset} {args.split} dataset...")
    if args.dataset == "vqav2":
        from tc_vitok.data.datasets import VQAv2Dataset

        if args.annotations_path is None or args.questions_path is None:
            raise ValueError(
                "VQAv2 requires --annotations-path and --questions-path"
            )
        dataset = VQAv2Dataset(
            annotations_path=args.annotations_path,
            questions_path=args.questions_path,
            image_dir=args.image_dir or "data/vqav2/train2014",
            split=args.split,
            max_samples=args.max_samples,
        )
    elif args.dataset == "gqa":
        from tc_vitok.data.datasets import GQADataset

        if args.questions_path is None:
            raise ValueError("GQA requires --questions-path")
        dataset = GQADataset(
            questions_path=args.questions_path,
            image_dir=args.image_dir or "data/gqa/images",
            max_samples=args.max_samples,
        )
    elif args.dataset == "textvqa":
        raise NotImplementedError("TextVQA caching not yet supported")
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    print(f"Dataset loaded: {len(dataset)} samples")

    # Build cache
    from tc_vitok.data.caching import build_cache

    print("Building feature cache...")
    build_cache(
        dataset=dataset,
        cache_path=str(cache_path),
        coarse_encoder=coarse_encoder,
        text_encoder=text_encoder,
        batch_size=args.batch_size,
        device=device,
    )

    print(f"Cache saved to {cache_path}")
    print("Done!")


if __name__ == "__main__":
    main()
