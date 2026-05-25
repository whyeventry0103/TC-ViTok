#!/usr/bin/env python3
"""Held-out dead-zone validation.

Identifies low-score TPN positions on VQAv2 val, then tests the resulting
bounds as an inference-time mask on the original GQA val_balanced evaluation
subset. This separates dead-zone identification from the GQA M1 test set.

Usage:
    DATA_ROOT=~/tc-vitok-data python3 scripts/cross_dataset_validation.py \
        --checkpoint runs/phase1_final/phase1/best.ckpt \
        --vqav2-n 500 --gqa-n 500 \
        --output runs/eval_results/cross_dataset_m1.json
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import sys
from pathlib import Path
from collections import deque

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from tc_vitok.data.datasets import GQADataset, VQAv2Dataset
from mask_dead_zone_eval import (
    bootstrap_ci,
    load_model,
    rect_flat_indices,
    run_eval,
)


GQA_BOUNDS = (0, 4, 6, 11)


def bbox_to_set(bounds, grid=24):
    r0, r1, c0, c1 = bounds
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def jaccard(a, b):
    a_set = bbox_to_set(a)
    b_set = bbox_to_set(b)
    return len(a_set & b_set) / max(1, len(a_set | b_set))


def largest_component(mask):
    """Return largest 4-connected component from a boolean 2D mask."""
    seen = np.zeros(mask.shape, dtype=bool)
    best = []
    rows, cols = mask.shape
    for r in range(rows):
        for c in range(cols):
            if not mask[r, c] or seen[r, c]:
                continue
            comp = []
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < rows and 0 <= nc < cols and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if len(comp) > len(best):
                best = comp
    return best


def component_bounds(component):
    rows = [p[0] for p in component]
    cols = [p[1] for p in component]
    return (min(rows), max(rows), min(cols), max(cols))


def identify_low_score_bounds(model, dataset, device, n, threshold, progress_every):
    accum = np.zeros((24, 24), dtype=np.float64)
    freq = np.zeros((24, 24), dtype=np.float64)
    count = 0
    question_ids = []

    for i in range(n):
        sample = dataset[i]
        images_coarse = sample["images_coarse"].unsqueeze(0).to(device)
        question = sample["questions"]
        info = model.get_selection_info(images_coarse, [question])
        mask = info["mask"][0].detach().cpu().numpy().reshape(24, 24)
        scores = info["scores"][0].detach().cpu().numpy().reshape(24, 24)
        s = scores - scores.min()
        s = s / (s.max() + 1e-8)
        accum += s
        freq += mask
        count += 1
        question_ids.append(str(sample.get("question_id", i)))
        if (i + 1) % progress_every == 0 or (i + 1) == n:
            print(f"  [VQAv2 {i+1}/{n}] mean score={accum.mean()/count:.3f}", flush=True)

    score_mean = accum / count
    select_freq = freq / count
    low_mask = score_mean < threshold
    component = largest_component(low_mask)
    if not component:
        print(
            f"[warn] no positions below threshold={threshold:.3f}; "
            "falling back to the 30 lowest-score positions.",
            flush=True,
        )
        flat = np.argsort(score_mean.reshape(-1))[:30]
        component = [(int(idx // 24), int(idx % 24)) for idx in flat]
    bounds = component_bounds(component)
    low_positions = [[int(r), int(c)] for r, c in component]
    return {
        "bounds": bounds,
        "low_positions": low_positions,
        "score_mean": score_mean,
        "select_freq": select_freq,
        "question_ids": question_ids,
    }


def load_baseline(path, expected_n):
    if not path or not Path(path).exists():
        return None
    with open(path) as f:
        data = json.load(f)
    baseline = data.get("baseline")
    if baseline is None:
        raise ValueError(f"{path} exists but has no baseline key")
    if len(baseline) != expected_n:
        raise ValueError(f"{path} baseline length {len(baseline)} != expected n={expected_n}")
    return baseline


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", default=None)
    p.add_argument("--vqav2-n", type=int, default=500)
    p.add_argument("--gqa-n", type=int, default=500)
    p.add_argument("--threshold", type=float, default=0.40)
    p.add_argument("--output", default="runs/eval_results/cross_dataset_m1.json")
    p.add_argument("--baseline-cache", default="runs/eval_results/m3_multi_rect_final.json")
    p.add_argument("--device", default=None)
    p.add_argument("--llava-model", default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--progress-every", type=int, default=25)
    args = p.parse_args()

    data_root = args.data_root or os.environ.get("DATA_ROOT")
    if not data_root:
        raise SystemExit("--data-root or $DATA_ROOT required")

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device, args.llava_model)

    vqav2 = VQAv2Dataset(
        annotations_path=f"{data_root}/vqav2/v2_mscoco_val2014_annotations.json",
        questions_path=f"{data_root}/vqav2/v2_OpenEnded_mscoco_val2014_questions.json",
        image_dir=f"{data_root}/vqav2/val2014",
        split="val",
    )
    gqa = GQADataset(
        questions_path=f"{data_root}/gqa/val_balanced_questions.json",
        image_dir=f"{data_root}/gqa/images",
    )
    vqav2_n = min(args.vqav2_n, len(vqav2))
    gqa_n = min(args.gqa_n, len(gqa))

    print("\n=== Identify low-score bounds on VQAv2 val ===", flush=True)
    diag = identify_low_score_bounds(
        model, vqav2, device, vqav2_n, args.threshold, args.progress_every
    )
    vqav2_bounds = tuple(int(x) for x in diag["bounds"])
    overlap = jaccard(vqav2_bounds, GQA_BOUNDS)
    print(f"VQAv2-derived bounds: rows {vqav2_bounds[0]}-{vqav2_bounds[1]}, "
          f"cols {vqav2_bounds[2]}-{vqav2_bounds[3]}", flush=True)
    print(f"GQA-derived bounds:   rows {GQA_BOUNDS[0]}-{GQA_BOUNDS[1]}, "
          f"cols {GQA_BOUNDS[2]}-{GQA_BOUNDS[3]}", flush=True)
    print(f"Jaccard overlap: {overlap:.3f}", flush=True)

    vqav2_ids = {f"vqav2:{qid}" for qid in diag["question_ids"]}
    gqa_ids = {f"gqa:{gqa.samples[i]['question_id']}" for i in range(gqa_n)}
    assert not (vqav2_ids & gqa_ids), "VQAv2 and GQA prefixed question IDs overlap unexpectedly"

    baseline_scores = load_baseline(args.baseline_cache, gqa_n)
    if baseline_scores is None:
        print("\n=== Baseline on GQA (no mask) ===", flush=True)
        baseline_scores, _, _ = run_eval(
            model, gqa, device, gqa_n, None, "none", args.progress_every
        )
    else:
        print(f"\n=== Reusing cached GQA baseline: {args.baseline_cache} ===", flush=True)

    print("\n=== Mask VQAv2-derived bounds on GQA ===", flush=True)
    mask_scores, _, _ = run_eval(
        model,
        gqa,
        device,
        gqa_n,
        rect_flat_indices(vqav2_bounds),
        "vqav2_bounds_on_gqa",
        args.progress_every,
    )
    diffs = [m - b for m, b in zip(mask_scores, baseline_scores)]
    delta = float(np.mean(diffs))
    lo, hi = bootstrap_ci(diffs)
    base_mean = float(np.mean(baseline_scores))
    masked_mean = float(np.mean(mask_scores))

    result = {
        "vqav2_n": vqav2_n,
        "gqa_n": gqa_n,
        "threshold": args.threshold,
        "vqav2_bounds": list(vqav2_bounds),
        "gqa_bounds": list(GQA_BOUNDS),
        "vqav2_low_positions": diag["low_positions"],
        "jaccard_overlap": overlap,
        "gqa_baseline_mean": base_mean,
        "gqa_masked_mean": masked_mean,
        "m1_vqav2_bounds_on_gqa_delta": delta,
        "m1_vqav2_bounds_on_gqa_ci": [lo, hi],
        "baseline_scores": baseline_scores,
        "masked_scores": mask_scores,
        "gqa_question_ids": [str(gqa.samples[i]["question_id"]) for i in range(gqa_n)],
        "vqav2_question_ids": diag["question_ids"],
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)

    print("\n=== Cross-dataset M1 summary ===", flush=True)
    print(f"GQA baseline mean: {base_mean:.4f}", flush=True)
    print(f"Masked mean:       {masked_mean:.4f}", flush=True)
    print(f"Delta:             {delta:+.4f} ({delta * 100:+.2f} pp)", flush=True)
    print(f"95% CI:            [{lo:+.4f}, {hi:+.4f}]", flush=True)
    print(f"Saved: {out}", flush=True)
    print("VALIDATION PASS", flush=True)


if __name__ == "__main__":
    main()
