#!/usr/bin/env python3
"""Analyze information flow in the parameter-free token merge.

This diagnostic measures whether the merge meaningfully changes selected token
features and whether the post-merge representation aligns with the aggregated
unselected-token contribution.
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
from tc_vitok.models.tc_vitok_model import complement_gather, gather_tokens  # noqa: E402


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    """Linear CKA between two [n,d] matrices."""
    x = x.float()
    y = y.float()
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    xy = x.T @ y
    xx = x.T @ x
    yy = y.T @ y
    denom = xx.norm() * yy.norm()
    if float(denom) <= 0:
        return float("nan")
    return float((xy.norm() ** 2 / denom).clamp(min=0.0, max=1.0))


def compute_unselected_contribution(
    v_full: torch.Tensor,
    idx: torch.Tensor,
    scores: torch.Tensor,
    tau_merge: float,
) -> torch.Tensor:
    """Reproduce the unselected-token contribution term from informative merge."""
    v_selected = gather_tokens(v_full, idx)
    v_unsel, unsel_idx = complement_gather(v_full, idx)

    v_sel_norm = F.normalize(v_selected, dim=-1)
    v_unsel_norm = F.normalize(v_unsel, dim=-1)
    sim = torch.bmm(v_unsel_norm, v_sel_norm.transpose(1, 2))
    assignment = F.softmax(sim / tau_merge, dim=-1)

    unsel_scores = scores.gather(1, unsel_idx)
    score_weights = F.softmax(unsel_scores, dim=-1)
    weighted_assignment = assignment * score_weights.unsqueeze(-1)
    weighted_assignment = weighted_assignment / (
        weighted_assignment.sum(dim=1, keepdim=True) + 1e-8
    )
    return torch.bmm(weighted_assignment.transpose(1, 2), v_unsel)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--n-images", type=int, default=500)
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

    selected_all = []
    merged_all = []
    contribution_all = []
    rel_norms = []
    cosine_selected_contrib = []

    for batch in tqdm(loader, desc="T8 merge CKA"):
        images_coarse = batch["images_coarse"].to(device)
        images_full = batch["images_full"].to(device)
        questions = batch["questions"]

        v_full, idx, scores, _ = model._compute_tpn_selection(images_coarse, images_full, questions)
        v_selected = gather_tokens(v_full, idx)
        contribution = compute_unselected_contribution(v_full, idx, scores, model.merge_tau)
        v_merged = v_selected + model.merge_beta * contribution

        selected_all.append(v_selected.detach().cpu().reshape(-1, v_selected.shape[-1]))
        contribution_all.append(contribution.detach().cpu().reshape(-1, contribution.shape[-1]))
        merged_all.append(v_merged.detach().cpu().reshape(-1, v_merged.shape[-1]))

        delta = model.merge_beta * contribution
        rel = delta.norm(dim=-1) / (v_selected.norm(dim=-1) + 1e-8)
        rel_norms.extend(float(x) for x in rel.detach().cpu().reshape(-1))
        cos = F.cosine_similarity(v_selected.float(), contribution.float(), dim=-1)
        cosine_selected_contrib.extend(float(x) for x in cos.detach().cpu().reshape(-1))

    selected = torch.cat(selected_all, dim=0)
    merged = torch.cat(merged_all, dim=0)
    contribution = torch.cat(contribution_all, dim=0)
    rel_t = torch.tensor(rel_norms)
    cos_t = torch.tensor(cosine_selected_contrib)

    result = {
        "n_images": int(len(dataset)),
        "n_token_rows": int(selected.shape[0]),
        "merge_beta": float(model.merge_beta),
        "merge_tau": float(model.merge_tau),
        "cka_selected_vs_merged": linear_cka(selected, merged),
        "cka_selected_vs_unselected_contribution": linear_cka(selected, contribution),
        "cka_merged_vs_unselected_contribution": linear_cka(merged, contribution),
        "mean_relative_delta_norm": float(rel_t.mean()),
        "median_relative_delta_norm": float(rel_t.median()),
        "max_relative_delta_norm": float(rel_t.max()),
        "mean_cosine_selected_contribution": float(cos_t.mean()),
        "median_cosine_selected_contribution": float(cos_t.median()),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
