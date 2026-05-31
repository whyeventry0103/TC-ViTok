#!/usr/bin/env python3
"""Measure rough latency and optional FLOPs for TC-ViTok eval configurations."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.evaluate import load_model  # noqa: E402
from tc_vitok.data.datasets import GQADataset, collate_fn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


def load_one_batch(data_root: str, batch_size: int):
    dataset = GQADataset(
        questions_path=f"{data_root}/gqa/val_balanced_questions.json",
        image_dir=f"{data_root}/gqa/images",
        max_samples=batch_size,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return next(iter(loader))


@torch.no_grad()
def measure_latency(model, batch, device: str, n_warm: int, n_trial: int) -> dict:
    images_coarse = batch["images_coarse"].to(device)
    images_full = batch["images_full"].to(device)
    questions = batch["questions"]

    for _ in range(n_warm):
        model.generate(images_coarse, images_full, questions, max_new_tokens=8)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_trial):
        model.generate(images_coarse, images_full, questions, max_new_tokens=8)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return {
        "batch_size": int(images_full.shape[0]),
        "n_trial": int(n_trial),
        "ms_per_batch": 1000 * elapsed / n_trial,
        "ms_per_image": 1000 * elapsed / (n_trial * images_full.shape[0]),
    }


def try_measure_flops(model, batch, device: str) -> dict:
    try:
        from calflops import calculate_flops
    except Exception as exc:
        return {"flops": None, "macs": None, "params": None, "flop_note": f"calflops unavailable: {exc}"}

    images_coarse = batch["images_coarse"].to(device)
    images_full = batch["images_full"].to(device)
    questions = batch["questions"]

    class Wrapper(torch.nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped

        def forward(self, images_coarse, images_full):
            selected, _ = self.wrapped._select_tokens(images_coarse, images_full, questions)
            return selected

    try:
        flops, macs, params = calculate_flops(
            model=Wrapper(model),
            kwargs={"images_coarse": images_coarse, "images_full": images_full},
            print_results=False,
        )
        return {"flops": flops, "macs": macs, "params": params}
    except Exception as exc:
        return {"flops": None, "macs": None, "params": None, "flop_note": f"calflops failed: {exc}"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--baselines", nargs="+", default=["none", "random_drop", "uniform_grid", "cls_attention"])
    parser.add_argument("--k", type=int, default=144)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--n-warm", type=int, default=3)
    parser.add_argument("--n-trial", type=int, default=10)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data_root = str(Path(args.data_root or os.environ["DATA_ROOT"]).expanduser())
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    batch = load_one_batch(data_root, args.batch_size)

    results = {}
    for baseline in args.baselines:
        print(f"=== {baseline} ===")
        model = load_model(
            checkpoint_path=args.checkpoint,
            device=device,
            k=args.k,
            baseline=baseline,
            auto_adaptive_budget=False,
        ).eval()
        results[baseline] = {
            **measure_latency(model, batch, device, args.n_warm, args.n_trial),
            **try_measure_flops(model, batch, device),
        }
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
