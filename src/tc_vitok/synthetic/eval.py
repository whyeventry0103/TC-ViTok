"""Evaluate synthetic routing tasks with answer and target-patch metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

from tc_vitok.data.datasets import CLIP_TRANSFORM, DINO_TRANSFORM
from tc_vitok.evaluation.metrics import normalize_answer
from tc_vitok.synthetic import dense_text, needle, two_needle

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
from scripts.evaluate import load_model  # noqa: E402


BUILDERS = {
    "needle": needle.build_dataset,
    "two_needle": two_needle.build_dataset,
    "dense_text": dense_text.build_dataset,
}


@torch.no_grad()
def eval_task(model, task: str, n: int, batch_size: int, device: str) -> dict:
    samples = BUILDERS[task](n)
    n_answer = 0
    n_target = 0
    n_seen = 0

    for start in tqdm(range(0, len(samples), batch_size), desc=f"synthetic/{task}"):
        batch = samples[start:start + batch_size]
        images_coarse = torch.stack([DINO_TRANSFORM(s.image) for s in batch]).to(device)
        images_full = torch.stack([CLIP_TRANSFORM(s.image) for s in batch]).to(device)
        questions = [s.question for s in batch]
        answers = [s.answer for s in batch]

        preds = model.generate(images_coarse, images_full, questions, max_new_tokens=8)
        _, idx, _, _ = model._compute_tpn_selection(images_coarse, images_full, questions)

        for pred, answer, selected, sample in zip(preds, answers, idx.cpu(), batch):
            pred_norm = normalize_answer(pred)
            answer_norm = normalize_answer(answer)
            if answer_norm in pred_norm.split() or pred_norm == answer_norm:
                n_answer += 1
            if int(sample.target_patch_idx) in {int(x) for x in selected.tolist() if int(x) >= 0}:
                n_target += 1
            n_seen += 1

    return {
        "task": task,
        "n": n_seen,
        "answer_accuracy": n_answer / max(1, n_seen),
        "target_patch_selection_rate": n_target / max(1, n_seen),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tasks", nargs="+", default=["needle", "two_needle", "dense_text"], choices=sorted(BUILDERS))
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", required=True)
    parser.add_argument("--llava-model", default="llava-hf/llava-1.5-7b-hf")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = load_model(
        checkpoint_path=args.checkpoint,
        device=device,
        k=144,
        baseline="none",
        llava_model_name=args.llava_model,
        auto_adaptive_budget=False,
    ).eval()

    results = [eval_task(model, task, args.n, args.batch_size, device) for task in args.tasks]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
