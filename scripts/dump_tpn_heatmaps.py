#!/usr/bin/env python3
"""Dump TPN selection heatmaps for the qualitative figure.

Runs the Phase 1 Final checkpoint on a fixed subset of GQA val_balanced and
saves, per example, the (raw image, TPN score grid, selected-token mask,
question, ground-truth answer, model prediction) tuple as a single .npz +
matching .png renders. The qualitative figure in the paper is built from
these dumps.

Output layout::

    <out_dir>/
      sample_0000.npz   image, scores, mask, idx, question, answer, pred
      sample_0000_overlay.png  (image + score heatmap)
      ...
      manifest.json    summary of all dumped samples

Usage::

    python scripts/dump_tpn_heatmaps.py \\
        --checkpoint runs/phase1_final/phase1/best.ckpt \\
        --data-root  $DATA_ROOT \\
        --out runs/qualitative_dump/ \\
        --n-samples 32 \\
        --benchmark gqa
"""

import os

# Match evaluate.py: HF offline before any HF import
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from tc_vitok.data.datasets import GQADataset, VQAv2Dataset, CLIP_TRANSFORM, DINO_TRANSFORM
from tc_vitok.models.tc_vitok_model import TCViTokModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", default=None)
    p.add_argument("--benchmark", default="gqa", choices=["gqa", "vqav2"])
    p.add_argument("--out", default="runs/qualitative_dump")
    p.add_argument("--n-samples", type=int, default=32)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--llava-model", default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--render-overlays", action="store_true",
                   help="Also save .png overlays of the heatmap on the raw image.")
    p.add_argument("--adaptive-budget", action="store_true",
                   help="Load the Adaptive checkpoint with BudgetPredictor "
                        "active. Each dump records the per-sample k_eff so the "
                        "budget-histogram figure can be built downstream.")
    p.add_argument("--k-min", type=int, default=36)
    p.add_argument("--k-max", type=int, default=288)
    return p.parse_args()


def detect_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def load_model_with_final_config(
    ckpt_path: str,
    device: str,
    llava_model: str,
    adaptive_budget: bool = False,
    k_min: int = 36,
    k_max: int = 288,
) -> TCViTokModel:
    """Load checkpoint with the LoRA-rank auto-detection used in evaluate.py.

    Set ``adaptive_budget=True`` when loading a Phase 1 Adaptive checkpoint;
    the BudgetPredictor weights are present in the state dict and need a
    matching module to overlay onto.
    """
    print(f"[load] Reading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)

    rank = 32
    for key, val in state_dict.items():
        if "lora_A" in key and "q_proj" in key:
            rank = val.shape[0]
            break
    targets = {"q_proj", "k_proj", "v_proj", "o_proj"}
    for key in state_dict:
        if "lora_A" in key:
            for t in ("gate_proj", "up_proj", "down_proj"):
                if t in key:
                    targets.add(t)

    lora_config = {"rank": rank, "alpha": rank * 2, "targets": sorted(targets), "dropout": 0.05}

    # Auto-detect adaptive_budget if the user did not set it explicitly: the
    # BudgetPredictor weights live under ``budget_predictor.*``.
    has_bp = any(k.startswith("budget_predictor.") for k in state_dict)
    if has_bp and not adaptive_budget:
        print("[load] Detected budget_predictor.* in checkpoint -- enabling adaptive_budget.")
        adaptive_budget = True

    model = TCViTokModel(
        k=144,
        tpn_config={"n_heads": 4, "d_model": 384, "d_coarse": 384,
                    "d_h": 512, "d_text": 512, "M": 49, "N": 576},
        use_stub_llm=False,
        llava_model_name=llava_model,
        lora_config=lora_config,
        load_encoders=True,
        merge_beta=0.3,
        merge_tau=0.1,
        adaptive_budget=adaptive_budget,
        k_min=k_min,
        k_max=k_max,
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[load] missing={len(missing)} | unexpected={len(unexpected)} "
          f"| LoRA rank={rank} targets={sorted(targets)} | adaptive={adaptive_budget}")
    model.eval().to(device)
    return model


def overlay_heatmap(pil_image: Image.Image, scores_24x24: np.ndarray) -> Image.Image:
    """Render a transparent jet overlay of TPN scores on the raw image."""
    import matplotlib.cm as cm

    img = pil_image.convert("RGB").resize((336, 336))
    s = scores_24x24
    # Min-max normalize per image for visualization (raw scores are post-instance-norm).
    s = (s - s.min()) / (s.max() - s.min() + 1e-8)
    s = np.kron(s, np.ones((14, 14)))  # upsample 24x24 → 336x336
    cmap = cm.get_cmap("inferno")
    rgba = (cmap(s) * 255).astype(np.uint8)
    overlay = Image.fromarray(rgba)

    base = img.convert("RGBA")
    overlay.putalpha(140)  # alpha
    return Image.alpha_composite(base, overlay).convert("RGB")


def main() -> None:
    args = parse_args()
    data_root = args.data_root or os.environ.get("DATA_ROOT")
    if not data_root:
        raise SystemExit("--data-root or $DATA_ROOT required")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    device = args.device or detect_device()
    model = load_model_with_final_config(
        args.checkpoint, device, args.llava_model,
        adaptive_budget=args.adaptive_budget,
        k_min=args.k_min, k_max=args.k_max,
    )

    if args.benchmark == "gqa":
        dataset = GQADataset(
            questions_path=f"{data_root}/gqa/val_balanced_questions.json",
            image_dir=f"{data_root}/gqa/images",
        )
        image_dir = Path(f"{data_root}/gqa/images")
    else:
        dataset = VQAv2Dataset(
            annotations_path=f"{data_root}/vqav2/v2_mscoco_val2014_annotations.json",
            questions_path=f"{data_root}/vqav2/v2_OpenEnded_mscoco_val2014_questions.json",
            image_dir=f"{data_root}/vqav2/val2014",
            split="val",
        )
        image_dir = Path(f"{data_root}/vqav2/val2014")

    manifest: list[dict] = []
    indices = list(range(args.start_index, args.start_index + args.n_samples))
    for n_dumped, idx in enumerate(indices):
        if idx >= len(dataset):
            break
        sample = dataset[idx]

        images_coarse = sample["images_coarse"].unsqueeze(0).to(device)
        images_full = sample["images_full"].unsqueeze(0).to(device)
        question = sample["questions"]

        info = model.get_selection_info(images_coarse, [question])
        scores = info["scores"][0].detach().cpu().numpy().reshape(24, 24)
        mask = info["mask"][0].detach().cpu().numpy().reshape(24, 24)
        idx_arr = info["idx"][0].detach().cpu().numpy()
        # Effective per-sample k (counts -1 padding entries used by adaptive mode)
        k_eff = int((idx_arr >= 0).sum())

        with torch.no_grad():
            preds = model.generate(
                images_coarse=images_coarse,
                images_full=images_full,
                questions=[question],
                max_new_tokens=20,
            )
        pred = preds[0]

        # Locate raw image to save alongside dump
        raw_image_path: Optional[Path] = None
        if args.benchmark == "gqa":
            cand = image_dir / f"{sample['image_id']}.jpg"
            if cand.exists():
                raw_image_path = cand
        else:
            iid = sample.get("image_id", "")
            try:
                cand = image_dir / f"COCO_val2014_{int(iid):012d}.jpg"
                if cand.exists():
                    raw_image_path = cand
            except (TypeError, ValueError):
                pass

        npz_path = out / f"sample_{n_dumped:04d}.npz"
        np.savez(
            npz_path,
            scores=scores,
            mask=mask,
            idx=idx_arr,
            k_eff=k_eff,
            question=question,
            answer=str(sample.get("answer_str", "")),
            pred=pred,
            image_path=str(raw_image_path) if raw_image_path else "",
        )
        manifest.append({
            "n": n_dumped,
            "dataset_index": idx,
            "question": question,
            "answer": str(sample.get("answer_str", "")),
            "pred": pred,
            "k_eff": k_eff,
            "image_path": str(raw_image_path) if raw_image_path else "",
            "npz": str(npz_path),
        })

        if args.render_overlays and raw_image_path and raw_image_path.exists():
            try:
                pil = Image.open(raw_image_path)
                ov = overlay_heatmap(pil, scores)
                ov.save(out / f"sample_{n_dumped:04d}_overlay.png")
            except Exception as e:
                print(f"  [warn] overlay failed for sample {n_dumped}: {e}")

        if (n_dumped + 1) % 4 == 0:
            print(f"[dump] {n_dumped + 1}/{len(indices)}: q='{question[:60]}' | pred='{pred[:30]}'")

    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[done] {len(manifest)} samples dumped to {out}")


if __name__ == "__main__":
    main()
