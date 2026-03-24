#!/usr/bin/env python3
"""Precompute and cache visual and text features to HDF5.

Usage:
    python scripts/cache_features.py --dataset vqav2 --split train --output data/cache/

This significantly speeds up training by avoiding repeated encoding of:
- DINOv2-S coarse visual features [49, 384]
- CLIP text embeddings [512]

Full visual features (CLIP ViT-L/14) are NOT cached due to size.

Paper reference: Section 7.4 (Implementation Details)
"""

import argparse
from pathlib import Path
import torch

# TODO (Phase B): Import TC-ViTok modules
# from tc_vitok.models import CoarseVisualEncoder, TextEncoder
# from tc_vitok.data import VQAv2Dataset, GQADataset, build_cache


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
        "--image-dir",
        type=str,
        default=None,
        help="Directory containing images (optional, dataset-specific default)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for feature extraction",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for feature extraction",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_path = output_dir / f"{args.dataset}_{args.split}.h5"

    print(f"Caching {args.dataset} {args.split} to {cache_path}")
    print(f"Device: {args.device}")

    # TODO (Phase B): Load encoders
    print("Loading encoders...")
    # coarse_encoder = CoarseVisualEncoder().to(args.device).eval()
    # text_encoder = TextEncoder().to(args.device).eval()

    # TODO (Phase B): Load dataset
    print(f"Loading {args.dataset} dataset...")
    # if args.dataset == "vqav2":
    #     dataset = VQAv2Dataset(split=args.split, image_dir=args.image_dir)
    # elif args.dataset == "gqa":
    #     dataset = GQADataset(split=args.split, image_dir=args.image_dir)
    # ...

    # TODO (Phase B): Build cache
    print("Building feature cache...")
    # build_cache(
    #     dataset=dataset,
    #     cache_path=cache_path,
    #     coarse_encoder=coarse_encoder,
    #     text_encoder=text_encoder,
    #     device=args.device,
    # )

    print(f"Cache saved to {cache_path}")
    print("Done!")


if __name__ == "__main__":
    main()
