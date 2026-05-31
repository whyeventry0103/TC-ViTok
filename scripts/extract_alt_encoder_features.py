#!/usr/bin/env python3
"""Extract same-image patch features from SigLIP or DINOv2 for Track C C3."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from tc_vitok.data.datasets import GQADataset  # noqa: E402


ENCODERS = {
    "siglip": "google/siglip-large-patch16-384",
    "dinov2": "facebook/dinov2-large",
}


def resize_patch_grid(features: torch.Tensor, target_side: int = 24) -> torch.Tensor:
    """Resize [P,D] square patch grid to [target_side^2,D]."""
    n_patches, dim = features.shape
    side = int(round(n_patches ** 0.5))
    if side * side != n_patches:
        raise ValueError(f"Patch count {n_patches} is not a square")
    if side == target_side:
        return features
    grid = features.transpose(0, 1).reshape(1, dim, side, side).float()
    resized = F.interpolate(grid, size=(target_side, target_side), mode="bilinear", align_corners=False)
    return resized.reshape(dim, target_side * target_side).transpose(0, 1)


def strip_prefix_tokens(hidden: torch.Tensor, model_name: str) -> torch.Tensor:
    """Remove CLS/register tokens and return a square patch-token grid."""
    candidates = [
        ("cls", 1),
        ("cls_plus_4_registers", 5),
    ]
    for label, prefix in candidates:
        patches = hidden[prefix:]
        side = int(round(patches.shape[0] ** 0.5))
        if side * side == patches.shape[0]:
            return patches.detach().cpu()
    raise ValueError(
        f"{model_name} produced {hidden.shape[0]} tokens; neither CLS-only nor "
        "CLS+4-register stripping leaves a square patch grid"
    )


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", choices=sorted(ENCODERS), required=True)
    parser.add_argument("--causal-data", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--n-images", type=int, default=None, help="Optional smoke subset size")
    parser.add_argument("--start-index", type=int, default=0, help="Start index into causal-data metadata")
    parser.add_argument("--target-side", type=int, default=24, help="Output patch-grid side length")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data_root = args.data_root or os.environ.get("DATA_ROOT")
    if not data_root:
        raise SystemExit("Set --data-root or DATA_ROOT")
    data_root = str(Path(data_root).expanduser())
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    causal = torch.load(args.causal_data, map_location="cpu")
    metadata = causal.get("metadata")
    if metadata:
        image_ids = [str(item["image_id"]) for item in metadata]
    else:
        n_images = int(causal["importances"].shape[0])
        dataset = GQADataset(
            questions_path=f"{data_root}/gqa/val_balanced_questions.json",
            image_dir=f"{data_root}/gqa/images",
            max_samples=n_images,
        )
        image_ids = [str(sample["image_id"]) for sample in dataset.samples]

    end = None if args.n_images is None else args.start_index + args.n_images
    image_ids = image_ids[args.start_index:end]
    if not image_ids:
        raise SystemExit("No images selected; check --start-index/--n-images")

    if args.encoder == "siglip":
        from transformers import SiglipModel, SiglipProcessor

        processor = SiglipProcessor.from_pretrained(ENCODERS[args.encoder])
        model = SiglipModel.from_pretrained(ENCODERS[args.encoder]).to(device).eval()

        def encode(image: Image.Image) -> torch.Tensor:
            inputs = processor(images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            out = model.vision_model(pixel_values=pixel_values, return_dict=True)
            return out.last_hidden_state[0].detach().cpu()

    else:
        from transformers import AutoImageProcessor, AutoModel

        processor = AutoImageProcessor.from_pretrained(ENCODERS[args.encoder])
        model = AutoModel.from_pretrained(ENCODERS[args.encoder]).to(device).eval()

        def encode(image: Image.Image) -> torch.Tensor:
            inputs = processor(images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            out = model(pixel_values=pixel_values, return_dict=True)
            hidden = out.last_hidden_state[0]
            return strip_prefix_tokens(hidden, ENCODERS[args.encoder])

    features = []
    for image_id in tqdm(image_ids, desc=f"{args.encoder} features"):
        path = Path(data_root) / "gqa" / "images" / f"{image_id}.jpg"
        image = Image.open(path).convert("RGB")
        patch_features = resize_patch_grid(encode(image), target_side=args.target_side)
        features.append(patch_features.to(torch.float16))

    tensor = torch.stack(features, dim=0)
    data = {
        "features": tensor,
        "encoder": args.encoder,
        "model_name": ENCODERS[args.encoder],
        "image_ids": image_ids,
        "start_index": args.start_index,
        "n_images": len(image_ids),
        "target_side": args.target_side,
        "source_causal_data": args.causal_data,
        "feature_shape": list(tensor.shape),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, out)
    print(json.dumps({
        "output": str(out),
        "encoder": args.encoder,
        "model_name": ENCODERS[args.encoder],
        "shape": list(tensor.shape),
    }, indent=2))


if __name__ == "__main__":
    main()
