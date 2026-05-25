#!/usr/bin/env python3
"""Measure rank correlation between learned TPN scores and CLIP CLS attention."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from tc_vitok.data.datasets import GQADataset, collate_fn
from scripts.evaluate import load_model


def average_ranks(values: list[float]) -> list[float]:
    """Return average ranks for ties, 0-indexed."""
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        avg = (i + j - 1) / 2.0
        for pos in order[i:j]:
            ranks[pos] = avg
        i = j
    return ranks


def pearson(xs: list[float], ys: list[float]) -> float:
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return float("nan")
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / math.sqrt(vx * vy)


def spearman(xs: list[float], ys: list[float]) -> float:
    return pearson(average_ranks(xs), average_ranks(ys))


@torch.no_grad()
def cls_attention_scores(model, images_full: torch.Tensor) -> torch.Tensor:
    clip_vision = model.full_encoder.clip.vision_model
    outputs = clip_vision(
        pixel_values=images_full,
        output_hidden_states=True,
        return_dict=True,
    )
    last_block = clip_vision.encoder.layers[-1]
    hidden = last_block.layer_norm1(outputs.hidden_states[-2])
    attn = last_block.self_attn
    q = attn.q_proj(hidden)
    k = attn.k_proj(hidden)
    batch, seq_len, dim = q.shape
    num_heads = int(attn.num_heads)
    head_dim = dim // num_heads
    scale = float(getattr(attn, "scale", head_dim ** -0.5))
    q = q.view(batch, seq_len, num_heads, head_dim).transpose(1, 2) * scale
    k = k.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
    weights = torch.matmul(q, k.transpose(-2, -1)).softmax(dim=-1)
    return weights[:, :, 0, 1:].mean(dim=1)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", required=True)
    parser.add_argument("--llava-model", default="llava-hf/llava-1.5-7b-hf")
    args = parser.parse_args()

    data_root = args.data_root or os.environ.get("DATA_ROOT")
    if not data_root:
        raise SystemExit("Set --data-root or DATA_ROOT")
    data_root = str(Path(data_root).expanduser())

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = load_model(
        checkpoint_path=args.checkpoint,
        device=device,
        k=144,
        baseline="none",
        llava_model_name=args.llava_model,
        auto_adaptive_budget=False,
    ).eval()

    dataset = GQADataset(
        questions_path=f"{data_root}/gqa/val_balanced_questions.json",
        image_dir=f"{data_root}/gqa/images",
        max_samples=args.num_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    rhos: list[float] = []
    for batch in tqdm(loader, desc="TPN/CLS correlation"):
        images_coarse = batch["images_coarse"].to(device)
        images_full = batch["images_full"].to(device)
        questions = batch["questions"]

        v_coarse = model.coarse_encoder(images_coarse)
        q_embed = model.text_encoder(questions)
        _, tpn_scores, _, _ = model.tpn(
            v_coarse=v_coarse,
            q_embed=q_embed,
            k=144,
            tau=0.1,
            training=False,
            adaptive_budget=False,
            budget_predictor=None,
        )
        cls_scores = cls_attention_scores(model, images_full)
        if tpn_scores.shape != cls_scores.shape:
            raise RuntimeError(
                f"Score shape mismatch: TPN={tuple(tpn_scores.shape)} "
                f"CLS={tuple(cls_scores.shape)}"
            )

        for tpn_row, cls_row in zip(tpn_scores.cpu(), cls_scores.cpu()):
            rho = spearman(tpn_row.tolist(), cls_row.tolist())
            if not math.isnan(rho):
                rhos.append(rho)

    if not rhos:
        raise SystemExit("No finite correlations computed")

    mean = sum(rhos) / len(rhos)
    sorted_rhos = sorted(rhos)
    median = sorted_rhos[len(sorted_rhos) // 2]
    std = math.sqrt(sum((rho - mean) ** 2 for rho in rhos) / len(rhos))
    result = {
        "num_samples": len(rhos),
        "mean_spearman": mean,
        "median_spearman": median,
        "std_spearman": std,
        "min": min(rhos),
        "max": max(rhos),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5.0, 3.0))
        ax.hist(rhos, bins=30, edgecolor="black")
        ax.axvline(mean, color="red", linewidth=1.5, label=f"mean={mean:.3f}")
        ax.axvline(0, color="grey", linewidth=0.8)
        ax.set_xlabel("Spearman rho: TPN scores vs CLIP CLS-attention")
        ax.set_ylabel("Images")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output.with_suffix(".pdf"))
        print(f"Wrote {output.with_suffix('.pdf')}")
    except Exception as exc:
        print(f"[warning] plot skipped: {exc}")


if __name__ == "__main__":
    main()
