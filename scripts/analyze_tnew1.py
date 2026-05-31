#!/usr/bin/env python3
"""T_new1: correlate TPN scores with multiple candidate visual priors.

Outputs:
- runs/p1_5_1/multi_prior_correlation.json
- runs/phase1_decision/D3.txt

D3 is max mean absolute Spearman correlation across all successfully computed
priors. This is the decision-tree number for "did TPN learn any obvious prior?"
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.evaluate import load_model  # noqa: E402
from tc_vitok.data.datasets import GQADataset, collate_fn  # noqa: E402


CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def average_ranks_1d(values: torch.Tensor) -> torch.Tensor:
    """Average ranks for ties, 0-indexed. CPU implementation for stability."""
    xs = [float(x) for x in values.tolist()]
    order = sorted(range(len(xs)), key=xs.__getitem__)
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and xs[order[j]] == xs[order[i]]:
            j += 1
        avg = (i + j - 1) / 2.0
        for pos in order[i:j]:
            ranks[pos] = avg
        i = j
    return torch.tensor(ranks, dtype=torch.float32)


def spearman_1d(a: torch.Tensor, b: torch.Tensor) -> float:
    ra = average_ranks_1d(a.detach().float().cpu())
    rb = average_ranks_1d(b.detach().float().cpu())
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = ra.norm() * rb.norm()
    if float(denom) <= 0:
        return float("nan")
    return float((ra * rb).sum() / denom)


def summarize_rhos(rhos: list[float]) -> dict:
    finite = torch.tensor([x for x in rhos if math.isfinite(x)], dtype=torch.float32)
    if finite.numel() == 0:
        return {"n": 0, "mean": None, "median": None, "std": None, "abs_mean": None}
    return {
        "n": int(finite.numel()),
        "mean": float(finite.mean()),
        "median": float(finite.median()),
        "std": float(finite.std(unbiased=False)),
        "abs_mean": float(finite.abs().mean()),
        "min": float(finite.min()),
        "max": float(finite.max()),
    }


@torch.no_grad()
def clip_cls_attention_scores(model, images_full: torch.Tensor, layer: int) -> torch.Tensor:
    """Manual CLIP CLS-to-patch attention for a specific encoder layer."""
    clip_vision = model.full_encoder.clip.vision_model
    outputs = clip_vision(
        pixel_values=images_full,
        output_hidden_states=True,
        return_dict=True,
    )
    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("CLIP did not return hidden states")
    block = clip_vision.encoder.layers[layer]
    # Attention layer `layer` consumes hidden_states[layer].
    hidden = block.layer_norm1(hidden_states[layer])
    attn = block.self_attn
    q = attn.q_proj(hidden)
    k = attn.k_proj(hidden)
    batch, seq_len, dim = q.shape
    num_heads = int(attn.num_heads)
    head_dim = dim // num_heads
    scale = float(getattr(attn, "scale", head_dim ** -0.5))
    q = q.view(batch, seq_len, num_heads, head_dim).transpose(1, 2) * scale
    k = k.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
    weights = torch.matmul(q, k.transpose(-2, -1)).softmax(dim=-1)
    scores = weights[:, :, 0, 1:].mean(dim=1)
    if scores.shape[1] != 576:
        raise RuntimeError(f"Expected 576 patch scores, got {tuple(scores.shape)}")
    return scores


@torch.no_grad()
def fastv_layer2_scores(model, v_full: torch.Tensor, questions: list[str]) -> torch.Tensor:
    """LLM layer-2 text-to-visual attention scores, shape [B,576].

    This is best-effort because Transformers attention implementations differ.
    It constructs the same [visual prefix | text prompt] input as TC-ViTok
    inference and asks the LLM to return attentions.
    """
    device = v_full.device
    lm = model._llm
    tokenizer = model.processor.tokenizer
    embed_fn = lm.get_input_embeddings()
    lm_dtype = next(lm.parameters()).dtype

    prompts = [f"USER: {q}\nASSISTANT:" for q in questions]
    text_inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
        add_special_tokens=True,
    )
    input_ids = text_inputs.input_ids.to(device)
    text_mask = text_inputs.attention_mask.to(device)

    use_cuda = "cuda" in str(device)
    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_cuda
        else torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=False)
    )
    with amp_ctx:
        v_projected = model._mm_proj(v_full.to(dtype=lm_dtype))
        text_embeds = embed_fn(input_ids).to(dtype=v_projected.dtype)

        combined_embeds = torch.cat([v_projected, text_embeds], dim=1)
        vis_mask = torch.ones(v_full.shape[0], v_full.shape[1], device=device, dtype=torch.long)
        combined_mask = torch.cat([vis_mask, text_mask], dim=1)

        outputs = lm(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            output_attentions=True,
            return_dict=True,
        )
    attentions = getattr(outputs, "attentions", None)
    if attentions is None or len(attentions) <= 2 or attentions[2] is None:
        raise RuntimeError("LLM did not return layer-2 attentions")
    attn = attentions[2]  # [B, heads, seq, seq]
    text_start = v_full.shape[1]
    # Only real text query positions.
    scores = []
    for b in range(v_full.shape[0]):
        valid_text = text_mask[b].bool()
        if int(valid_text.sum()) == 0:
            scores.append(torch.zeros(v_full.shape[1], device=device))
            continue
        rows = attn[b, :, text_start:text_start + text_mask.shape[1], :v_full.shape[1]]
        rows = rows[:, valid_text, :]
        scores.append(rows.mean(dim=(0, 1)))
    return torch.stack(scores, dim=0)


def unnormalize_clip(images_full: torch.Tensor) -> torch.Tensor:
    mean = CLIP_MEAN.to(images_full.device, images_full.dtype)
    std = CLIP_STD.to(images_full.device, images_full.dtype)
    return (images_full * std + mean).clamp(0, 1)


def patch_luminance_scores(images_full: torch.Tensor) -> torch.Tensor:
    img = unnormalize_clip(images_full).float()
    lum = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
    pooled = F.avg_pool2d(lum, kernel_size=14, stride=14)
    return pooled.flatten(1)


def patch_gradient_scores(images_full: torch.Tensor) -> torch.Tensor:
    img = unnormalize_clip(images_full).float()
    lum = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
    gx = F.pad(lum[:, :, :, 1:] - lum[:, :, :, :-1], (0, 1, 0, 0))
    gy = F.pad(lum[:, :, 1:, :] - lum[:, :, :-1, :], (0, 0, 0, 1))
    grad = torch.sqrt(gx.square() + gy.square() + 1e-12)
    pooled = F.avg_pool2d(grad, kernel_size=14, stride=14)
    return pooled.flatten(1)


def center_prior(batch_size: int, device: torch.device | str) -> torch.Tensor:
    rows = torch.arange(24, device=device).float()
    cols = torch.arange(24, device=device).float()
    rr, cc = torch.meshgrid(rows, cols, indexing="ij")
    # Higher score near the center.
    dist = torch.sqrt((rr - 11.5).square() + (cc - 11.5).square())
    prior = -dist.flatten()
    return prior.unsqueeze(0).expand(batch_size, -1)


def add_correlations(bucket: dict[str, list[float]], tpn_scores: torch.Tensor, priors: dict[str, torch.Tensor]) -> None:
    for name, prior in priors.items():
        if prior.shape != tpn_scores.shape:
            raise RuntimeError(
                f"Prior {name} shape {tuple(prior.shape)} does not match "
                f"TPN scores {tuple(tpn_scores.shape)}"
            )
        for tpn_row, prior_row in zip(tpn_scores.cpu(), prior.cpu()):
            rho = spearman_1d(tpn_row, prior_row)
            if math.isfinite(rho):
                bucket.setdefault(name, []).append(rho)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", required=True)
    parser.add_argument("--d3-output", default="runs/phase1_decision/D3.txt")
    parser.add_argument("--skip-fastv", action="store_true", help="Skip LLM layer-2 attention prior.")
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
        auto_adaptive_budget=False,
        llava_attn_implementation=None if args.skip_fastv else "eager",
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

    rhos: dict[str, list[float]] = {}
    notes: dict[str, str] = {}
    fastv_failed = False

    for batch in tqdm(loader, desc="T_new1 multi-prior correlation"):
        images_coarse = batch["images_coarse"].to(device)
        images_full = batch["images_full"].to(device)
        questions = batch["questions"]

        v_full, _, tpn_scores, _ = model._compute_tpn_selection(images_coarse, images_full, questions)
        priors = {
            "clip_cls_last": clip_cls_attention_scores(model, images_full, layer=-1),
            "clip_cls_penultimate": clip_cls_attention_scores(model, images_full, layer=-2),
            "image_gradient": patch_gradient_scores(images_full),
            "pixel_luminance": patch_luminance_scores(images_full),
            "center_prior": center_prior(images_full.shape[0], images_full.device),
        }

        if not args.skip_fastv and not fastv_failed:
            try:
                priors["fastv_layer2_text_to_visual"] = fastv_layer2_scores(model, v_full, questions)
            except Exception as exc:
                fastv_failed = True
                notes["fastv_layer2_text_to_visual"] = f"skipped after failure: {type(exc).__name__}: {exc}"

        add_correlations(rhos, tpn_scores, priors)

    prior_results = {name: summarize_rhos(vals) for name, vals in sorted(rhos.items())}
    valid_abs = [
        float(stats["abs_mean"])
        for stats in prior_results.values()
        if stats.get("abs_mean") is not None
    ]
    d3 = max(valid_abs) if valid_abs else float("nan")
    best_prior = None
    if prior_results:
        best_prior = max(
            prior_results,
            key=lambda name: -1.0 if prior_results[name].get("abs_mean") is None else prior_results[name]["abs_mean"],
        )

    result = {
        "num_samples_requested": args.num_samples,
        "num_samples_correlated": max((stats["n"] for stats in prior_results.values()), default=0),
        "priors": prior_results,
        "D3_max_abs_mean_spearman": d3,
        "best_prior": best_prior,
        "notes": notes,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))

    d3_path = Path(args.d3_output)
    d3_path.parent.mkdir(parents=True, exist_ok=True)
    d3_path.write_text(f"{d3:.6f}\n")

    print(json.dumps(result, indent=2))
    print(f"D3 = {d3:.6f} ({best_prior})")
    print(f"Wrote {output}")
    print(f"Wrote {d3_path}")


if __name__ == "__main__":
    main()
