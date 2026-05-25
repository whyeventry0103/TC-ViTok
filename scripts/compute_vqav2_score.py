#!/usr/bin/env python3
"""Compute soft VQAv2-style scores from saved mask ablation predictions.

The input is the JSON emitted by scripts/mask_dead_zone_eval.py after enabling
prediction saving. It must contain:
  - answers: list[list[str]]
  - preds_baseline, preds_dead_zone, preds_random_0, ...

For each condition, this script computes per-sample soft VQA score using
score = min(num_matching_answers / 3, 1.0), then reports deltas vs baseline
with bootstrap confidence intervals.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np


def normalize_answer(answer: str) -> str:
    answer = answer.lower().strip()
    answer = re.sub(r"[^\w\s]", "", answer)
    words = [w for w in answer.split() if w not in {"a", "an", "the"}]
    return " ".join(words).strip()


def vqa_score(prediction: str, answers: list[str]) -> float:
    pred = normalize_answer(prediction)
    gt = [normalize_answer(a) for a in answers]
    return min(sum(1 for a in gt if a == pred) / 3.0, 1.0)


def bootstrap_ci(diffs, n_boot=1000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    diffs = np.asarray(diffs, dtype=float)
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(diffs, size=len(diffs), replace=True)
        boots.append(sample.mean())
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def condition_from_pred_key(key: str) -> str:
    return key.removeprefix("preds_")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", required=True,
                   help="JSON from mask_dead_zone_eval.py with preds_* and answers.")
    p.add_argument("--out", default="runs/eval_results/m1_vqav2_soft.json")
    args = p.parse_args()

    with open(args.pairs) as f:
        data = json.load(f)

    if "answers" not in data:
        raise SystemExit(
            f"{args.pairs} has no 'answers' field. Re-run mask_dead_zone_eval.py "
            "after the prediction-saving patch."
        )
    answers = data["answers"]
    pred_keys = sorted(k for k in data if k.startswith("preds_"))
    if "preds_baseline" not in pred_keys:
        raise SystemExit(f"{args.pairs} has no preds_baseline field.")

    scores = {}
    for key in pred_keys:
        cond = condition_from_pred_key(key)
        preds = data[key]
        if len(preds) != len(answers):
            raise ValueError(f"{key} length {len(preds)} != answers length {len(answers)}")
        scores[cond] = [vqa_score(pred, ans) for pred, ans in zip(preds, answers)]

    baseline = scores["baseline"]
    out = {
        "pairs": args.pairs,
        "n": len(baseline),
        "conditions": {},
    }

    print(f"Soft VQAv2-score from {args.pairs} (n={len(baseline)})")
    print(f"{'condition':<16} {'soft_mean':>10} {'exact_mean':>10} {'delta':>10} {'95% CI':>22}")
    base_mean = float(np.mean(baseline))
    for cond in sorted(scores, key=lambda c: (c != "baseline", c)):
        vals = scores[cond]
        mean = float(np.mean(vals))
        diffs = [v - b for v, b in zip(vals, baseline)]
        delta = mean - base_mean
        lo, hi = bootstrap_ci(diffs)
        exact_mean = data.get(cond)
        exact_mean = float(np.mean(exact_mean)) if isinstance(exact_mean, list) else None
        print(
            f"{cond:<16} {mean:>10.4f} "
            f"{(exact_mean if exact_mean is not None else float('nan')):>10.4f} "
            f"{delta:>+10.4f} [{lo:+.4f}, {hi:+.4f}]"
        )
        out["conditions"][cond] = {
            "soft_scores": vals,
            "soft_mean": mean,
            "exact_mean": exact_mean,
            "delta_vs_baseline": delta,
            "ci95": [lo, hi],
        }

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(out, f)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
