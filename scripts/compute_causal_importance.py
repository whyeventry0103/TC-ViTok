#!/usr/bin/env python3
"""Compute per-token causal importance by visual-token ablation.

For each GQA sample, this script runs the LoRA-576 path, then zeros each CLIP
visual token one at a time and measures the answer-token NLL increase:

    importance[token] = NLL(ablated token) - NLL(full context)

Positive values mean the token helps the model assign probability to the ground
truth answer. The output is a .pt file with:
    importances: [N, 576] float16
    features:    [N, 576, 1024] float16
    metadata:    list of question/image/answer ids
"""

from __future__ import annotations

import argparse
import json
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


def per_sample_answer_nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean NLL over non-ignored answer labels, matching training CE alignment."""
    min_len = min(logits.shape[1], labels.shape[1])
    logits = logits[:, :min_len].float()
    labels = labels[:, :min_len]
    valid = labels != -100
    safe_labels = labels.masked_fill(~valid, 0)
    log_probs = F.log_softmax(torch.nan_to_num(logits, nan=0.0, posinf=100.0, neginf=-100.0), dim=-1)
    token_nll = -log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    token_nll = token_nll * valid.float()
    denom = valid.sum(dim=1).clamp_min(1).float()
    return token_nll.sum(dim=1) / denom


@torch.no_grad()
def answer_nll(model, visual_tokens: torch.Tensor, questions: list[str], answers: list[str]) -> torch.Tensor:
    outputs = model._llava_forward(visual_tokens, questions, answers)
    return per_sample_answer_nll(outputs["logits"], outputs["seq_labels"])


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--n-images", type=int, default=200)
    parser.add_argument("--ablation-batch-size", type=int, default=8)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data_root = args.data_root or os.environ.get("DATA_ROOT")
    if not data_root:
        raise SystemExit("Set --data-root or DATA_ROOT")
    data_root = str(Path(data_root).expanduser())

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = load_model(
        checkpoint_path=args.checkpoint,
        device=device,
        k=576,
        baseline="full_576",
        auto_adaptive_budget=False,
    ).eval()

    dataset = GQADataset(
        questions_path=f"{data_root}/gqa/val_balanced_questions.json",
        image_dir=f"{data_root}/gqa/images",
        max_samples=args.start_index + args.n_images,
    )
    if args.start_index:
        dataset.samples = dataset.samples[args.start_index:args.start_index + args.n_images]
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    all_importances = []
    all_features = []
    metadata = []

    for batch in tqdm(loader, desc="causal importance"):
        images_full = batch["images_full"].to(device)
        question = batch["questions"][0]
        answer = batch["answer_str"][0]

        v_full = model.full_encoder(images_full)  # [1, 576, 1024]
        baseline_nll = answer_nll(model, v_full, [question], [answer])[0]

        sample_importance = torch.empty(576, device="cpu", dtype=torch.float32)
        for start in range(0, 576, args.ablation_batch_size):
            end = min(start + args.ablation_batch_size, 576)
            positions = list(range(start, end))
            v_rep = v_full.repeat(len(positions), 1, 1)
            for row, pos in enumerate(positions):
                v_rep[row, pos] = 0
            nll = answer_nll(
                model,
                v_rep,
                [question] * len(positions),
                [answer] * len(positions),
            )
            sample_importance[start:end] = (nll - baseline_nll).detach().float().cpu()

        all_importances.append(sample_importance.to(torch.float16))
        all_features.append(v_full[0].detach().cpu().to(torch.float16))
        metadata.append({
            "question_id": batch.get("question_id", [None])[0],
            "image_id": batch.get("image_id", [None])[0],
            "question": question,
            "answer": answer,
            "baseline_nll": float(baseline_nll.detach().cpu()),
            "importance_mean": float(sample_importance.mean()),
            "importance_std": float(sample_importance.std(unbiased=False)),
            "importance_min": float(sample_importance.min()),
            "importance_max": float(sample_importance.max()),
        })

    data = {
        "importances": torch.stack(all_importances, dim=0),
        "features": torch.stack(all_features, dim=0),
        "metadata": metadata,
        "config": {
            "checkpoint": args.checkpoint,
            "start_index": args.start_index,
            "n_images": args.n_images,
            "ablation_batch_size": args.ablation_batch_size,
            "metric": "answer_nll_increase_from_zero_ablation",
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, out)

    imps = data["importances"].float()
    summary = {
        "output": str(out),
        "shape_importances": list(data["importances"].shape),
        "shape_features": list(data["features"].shape),
        "mean": float(imps.mean()),
        "std": float(imps.std(unbiased=False)),
        "min": float(imps.min()),
        "max": float(imps.max()),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
