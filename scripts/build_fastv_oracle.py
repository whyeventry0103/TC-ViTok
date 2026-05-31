#!/usr/bin/env python3
"""Build FastV-style oracle selections from LLM layer-2 attention.

For each GQA sample, this script runs the full-576 visual-token path and
extracts layer-2 text-to-visual attention. The top-k visual positions become
the oracle selection used by P1.3.
"""

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

from scripts.analyze_tnew1 import fastv_layer2_scores  # noqa: E402
from scripts.evaluate import load_model  # noqa: E402
from tc_vitok.data.datasets import GQADataset, collate_fn  # noqa: E402


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--n-images", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--k", type=int, default=144)
    parser.add_argument("--output", required=True, help="Path for oracle top-k indices .pt")
    parser.add_argument("--scores-output", default=None, help="Optional path for raw FastV scores .pt")
    parser.add_argument("--metadata-output", default=None, help="Optional metadata JSON path")
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
        llava_attn_implementation="eager",
    ).eval()

    dataset = GQADataset(
        questions_path=f"{data_root}/gqa/val_balanced_questions.json",
        image_dir=f"{data_root}/gqa/images",
        max_samples=args.n_images,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    all_indices = []
    all_scores = []
    metadata = []
    for batch in tqdm(loader, desc="FastV oracle"):
        images_full = batch["images_full"].to(device)
        questions = batch["questions"]
        v_full = model.full_encoder(images_full)
        scores = fastv_layer2_scores(model, v_full, questions)
        idx = scores.topk(args.k, dim=-1).indices
        all_indices.append(idx.cpu())
        all_scores.append(scores.cpu())
        for qid, image_id, question in zip(
            batch.get("question_id", [None] * len(questions)),
            batch.get("image_id", [None] * len(questions)),
            questions,
        ):
            metadata.append({"question_id": qid, "image_id": image_id, "question": question})

    oracle = torch.cat(all_indices, dim=0)
    scores = torch.cat(all_scores, dim=0)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(oracle, out)

    if args.scores_output:
        scores_out = Path(args.scores_output)
        scores_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(scores, scores_out)

    if args.metadata_output:
        meta_out = Path(args.metadata_output)
        meta_out.parent.mkdir(parents=True, exist_ok=True)
        meta_out.write_text(json.dumps(metadata, indent=2))

    summary = {
        "n_images": int(oracle.shape[0]),
        "k": int(oracle.shape[1]),
        "oracle_shape": list(oracle.shape),
        "scores_shape": list(scores.shape),
        "output": str(out),
        "scores_output": args.scores_output,
        "metadata_output": args.metadata_output,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
