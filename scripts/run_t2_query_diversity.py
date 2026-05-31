#!/usr/bin/env python3
"""Measure how much TPN scores change when questions change for the same image."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.evaluate import load_model  # noqa: E402
from tc_vitok.data.datasets import GQADataset, collate_fn  # noqa: E402


ALT_QUESTIONS = [
    "What is in the image?",
    "Describe the main object.",
    "What colors are present?",
    "How many objects are there?",
    "What is on the left side?",
]


def symmetric_kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    return 0.5 * ((p * (p.log() - q.log())).sum(-1) + (q * (q.log() - p.log())).sum(-1))


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--n-images", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data_root = str(Path(args.data_root or os.environ["DATA_ROOT"]).expanduser())
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = load_model(
        checkpoint_path=args.checkpoint,
        device=device,
        k=144,
        baseline="none",
        auto_adaptive_budget=False,
    ).eval()

    dataset = GQADataset(
        questions_path=f"{data_root}/gqa/val_balanced_questions.json",
        image_dir=f"{data_root}/gqa/images",
        max_samples=args.n_images,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    diversities = []
    topk_jaccards = []
    for batch in tqdm(loader, desc="T2 query diversity"):
        images_coarse = batch["images_coarse"].to(device)
        images_full = batch["images_full"].to(device)
        batch_size = images_full.shape[0]
        probs = []
        selections = []
        for question in ALT_QUESTIONS:
            questions = [question] * batch_size
            _, idx, scores, _ = model._compute_tpn_selection(images_coarse, images_full, questions)
            probs.append(torch.softmax(scores.float(), dim=-1).cpu())
            selections.append(idx.cpu())

        for b in range(batch_size):
            pair_kls = []
            pair_js = []
            for i in range(len(ALT_QUESTIONS)):
                for j in range(i + 1, len(ALT_QUESTIONS)):
                    pair_kls.append(float(symmetric_kl(probs[i][b], probs[j][b])))
                    a = {int(x) for x in selections[i][b].tolist() if int(x) >= 0}
                    c = {int(x) for x in selections[j][b].tolist() if int(x) >= 0}
                    pair_js.append(len(a & c) / max(1, len(a | c)))
            diversities.append(sum(pair_kls) / len(pair_kls))
            topk_jaccards.append(sum(pair_js) / len(pair_js))

    kl = torch.tensor(diversities)
    jac = torch.tensor(topk_jaccards)
    result = {
        "n_images": int(kl.numel()),
        "n_questions_per_image": len(ALT_QUESTIONS),
        "mean_per_image_kl": float(kl.mean()),
        "median_per_image_kl": float(kl.median()),
        "std_per_image_kl": float(kl.std(unbiased=False)),
        "mean_topk_jaccard": float(jac.mean()),
        "median_topk_jaccard": float(jac.median()),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
