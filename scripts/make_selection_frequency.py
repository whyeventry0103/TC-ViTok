#!/usr/bin/env python3
"""Diagnostic figure: average TPN selection frequency over many samples.

For each of N samples, runs the TPN and accumulates the
binary selection mask. The result is a 24x24 heatmap of P(position selected),
averaged across questions and images. If the TPN's selection is
query-conditioned and content-driven, this heatmap should be roughly uniform
(every position is selected ~k/N_pos of the time). A strong spatial mode
indicates a content/query-independent positional prior.

Usage:
    DATA_ROOT=~/tc-vitok-data python3 scripts/make_selection_frequency.py \\
        --checkpoint runs/phase1_final/phase1/best.ckpt \\
        --out "DL Project Report/figures/selection_frequency.pdf" \\
        --n-samples 500
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from tc_vitok.data.datasets import GQADataset, VQAv2Dataset
from tc_vitok.models.tc_vitok_model import TCViTokModel


def load_model(ckpt_path: str, device: str, llava_model: str, adaptive_budget: bool):
    print(f"[load] {ckpt_path}")
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

    has_bp = any(k.startswith("budget_predictor.") for k in state_dict)
    if has_bp and not adaptive_budget:
        adaptive_budget = True

    model = TCViTokModel(
        k=144,
        tpn_config={"n_heads": 4, "d_model": 384, "d_coarse": 384,
                    "d_h": 512, "d_text": 512, "M": 49, "N": 576},
        use_stub_llm=False,
        llava_model_name=llava_model,
        lora_config=lora_config,
        load_encoders=True,
        merge_beta=0.3, merge_tau=0.1,
        adaptive_budget=adaptive_budget,
    )
    model.load_state_dict(state_dict, strict=False)
    model.eval().to(device)
    return model


def build_dataset(data_root: str, benchmark: str):
    if benchmark == "gqa":
        return GQADataset(
            questions_path=f"{data_root}/gqa/val_balanced_questions.json",
            image_dir=f"{data_root}/gqa/images",
        )
    if benchmark == "vqav2":
        return VQAv2Dataset(
            annotations_path=f"{data_root}/vqav2/v2_mscoco_val2014_annotations.json",
            questions_path=f"{data_root}/vqav2/v2_OpenEnded_mscoco_val2014_questions.json",
            image_dir=f"{data_root}/vqav2/val2014",
            split="val",
        )
    raise ValueError(f"Unsupported benchmark: {benchmark}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", default=None)
    p.add_argument("--out", default="DL Project Report/figures/selection_frequency.pdf")
    p.add_argument("--n-samples", type=int, default=500)
    p.add_argument("--device", default=None)
    p.add_argument("--llava-model", default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--adaptive-budget", action="store_true")
    p.add_argument("--title", default=None)
    p.add_argument("--benchmark", choices=["gqa", "vqav2"], default="gqa")
    p.add_argument("--score-threshold", type=float, default=None,
                   help="If set, print the bbox of positions with mean normalized score below this threshold.")
    p.add_argument("--save-npz", default=None,
                   help="Optional path to save freq and score_mean arrays for downstream diagnostics.")
    args = p.parse_args()

    data_root = args.data_root or os.environ.get("DATA_ROOT")
    if not data_root:
        raise SystemExit("--data-root or $DATA_ROOT required")

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device, args.llava_model, args.adaptive_budget)
    dataset = build_dataset(data_root, args.benchmark)

    n = min(args.n_samples, len(dataset))
    accum = np.zeros((24, 24), dtype=np.float64)
    score_accum = np.zeros((24, 24), dtype=np.float64)
    count = 0

    for i in range(n):
        sample = dataset[i]
        images_coarse = sample["images_coarse"].unsqueeze(0).to(device)
        question = sample["questions"]
        info = model.get_selection_info(images_coarse, [question])
        mask = info["mask"][0].detach().cpu().numpy().reshape(24, 24)
        scores = info["scores"][0].detach().cpu().numpy().reshape(24, 24)
        accum += mask
        # Normalize scores per-image (matches per-image inference perception)
        s = scores - scores.min()
        s = s / (s.max() + 1e-8)
        score_accum += s
        count += 1
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{n}] running mean P_select={accum.mean()/count:.3f}")

    freq = accum / count            # 24x24, in [0,1]
    score_mean = score_accum / count

    # Uniform baseline = k/N (e.g., 144/576 = 0.25)
    uniform_p = float(freq.mean())
    print(f"\n[done] N={count} | mean P_select={uniform_p:.3f} | "
          f"max={freq.max():.3f} | min={freq.min():.3f} | "
          f"hot/cold ratio={freq.max()/max(freq.min(),1e-3):.2f}x")

    if args.score_threshold is not None:
        low = np.argwhere(score_mean < args.score_threshold)
        if len(low) == 0:
            print(f"[threshold] no positions below {args.score_threshold:.3f}")
        else:
            r0, c0 = low.min(axis=0)
            r1, c1 = low.max(axis=0)
            print(
                f"[threshold] score_mean < {args.score_threshold:.3f}: "
                f"{len(low)} positions, bbox rows {r0}-{r1}, cols {c0}-{c1}"
            )

    if args.save_npz:
        out_npz = Path(args.save_npz)
        out_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_npz, freq=freq, score_mean=score_mean, n=count)
        print(f"Saved arrays: {out_npz}")

    # Two-panel figure: selection frequency + mean score
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.6))

    im1 = ax1.imshow(freq, cmap="hot", vmin=0, vmax=min(1.0, freq.max() * 1.05),
                     interpolation="nearest")
    ax1.set_title(rf"P(selected) over {count} samples", fontsize=10)
    ax1.set_xlabel("patch column (0=left)", fontsize=8)
    ax1.set_ylabel("patch row (0=top)", fontsize=8)
    ax1.set_xticks([0, 6, 12, 18, 23]); ax1.set_yticks([0, 6, 12, 18, 23])
    ax1.tick_params(labelsize=7)
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.ax.tick_params(labelsize=7)
    cbar1.set_label(f"P_select (uniform={uniform_p:.2f})", fontsize=7)

    im2 = ax2.imshow(score_mean, cmap="viridis", interpolation="nearest")
    ax2.set_title("Mean per-image normalized score", fontsize=10)
    ax2.set_xlabel("patch column", fontsize=8)
    ax2.set_xticks([0, 6, 12, 18, 23]); ax2.set_yticks([0, 6, 12, 18, 23])
    ax2.tick_params(labelsize=7)
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.ax.tick_params(labelsize=7)

    title = args.title or (
        f"TPN positional prior on {args.benchmark} val "
        f"(N={count}, $k$=144)"
    )
    plt.suptitle(title, fontsize=10, y=1.02)
    plt.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out), bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
