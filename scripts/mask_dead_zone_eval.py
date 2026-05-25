#!/usr/bin/env python3
"""Ablation: mask the TPN score dead zone at inference and measure VQAv2-score delta.

Runs two forward passes per sample: one with the standard TPN, one with
positions (row 0-4, col 6-11) forced to score = -1e9 so they are never
selected by top-k.  Reports per-sample exact-match (VQA-style) and mean
VQAv2-score for both conditions.

Usage:
    DATA_ROOT=~/tc-vitok-data python3 scripts/mask_dead_zone_eval.py \\
        --checkpoint runs/phase1_final/phase1/best.ckpt \\
        --n-samples 500
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from tc_vitok.data.datasets import GQADataset
from tc_vitok.models.tc_vitok_model import TCViTokModel


# Dead-zone patch positions: rows 0-4, cols 6-11 in the 24x24 grid.
def dead_zone_flat_indices(grid=24, row_lo=0, row_hi=4, col_lo=6, col_hi=11):
    indices = []
    for r in range(row_lo, row_hi + 1):
        for c in range(col_lo, col_hi + 1):
            indices.append(r * grid + c)
    return indices


def bbox_intersects(a, b):
    ar0, ar1, ac0, ac1 = a
    br0, br1, bc0, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def rect_flat_indices(bounds, grid=24):
    row_lo, row_hi, col_lo, col_hi = bounds
    return [
        r * grid + c
        for r in range(row_lo, row_hi + 1)
        for c in range(col_lo, col_hi + 1)
    ]


def random_rectangles(n, seed, grid=24, dead_zone=(0, 4, 6, 11)):
    """Sample n non-overlapping 5x6 rectangles that do not intersect dead_zone."""
    import random
    rng = random.Random(seed)
    rects = []
    bounds_seen = []
    attempts = 0
    max_attempts = max(1000, n * 500)

    while len(rects) < n and attempts < max_attempts:
        attempts += 1
        row_lo = rng.randint(0, grid - 5)
        col_lo = rng.randint(0, grid - 6)
        bounds = (row_lo, row_lo + 4, col_lo, col_lo + 5)
        if bbox_intersects(bounds, dead_zone):
            continue
        if any(bbox_intersects(bounds, existing) for existing in bounds_seen):
            continue
        bounds_seen.append(bounds)
        rects.append((rect_flat_indices(bounds, grid), bounds))

    if len(rects) != n:
        raise RuntimeError(
            f"Could only sample {len(rects)} non-overlapping rectangles "
            f"after {attempts} attempts; requested {n}."
        )
    return rects


def vqa_score(pred: str, answer: str) -> float:
    """Simple exact-match VQA score used for quick ablation."""
    return 1.0 if pred.strip().lower() == answer.strip().lower() else 0.0


def load_model(ckpt_path, device, llava_model):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    rank = 32
    for key, val in state_dict.items():
        if "lora_A" in key and "q_proj" in key:
            rank = val.shape[0]; break
    targets = {"q_proj", "k_proj", "v_proj", "o_proj"}
    for key in state_dict:
        if "lora_A" in key:
            for t in ("gate_proj", "up_proj", "down_proj"):
                if t in key: targets.add(t)
    lora_config = {"rank": rank, "alpha": rank * 2, "targets": sorted(targets), "dropout": 0.05}
    has_budget_predictor = any(k.startswith("budget_predictor.") for k in state_dict)
    model = TCViTokModel(
        k=144,
        tpn_config={"n_heads": 4, "d_model": 384, "d_coarse": 384,
                    "d_h": 512, "d_text": 512, "M": 49, "N": 576},
        use_stub_llm=False, llava_model_name=llava_model,
        lora_config=lora_config, load_encoders=True,
        merge_beta=0.3, merge_tau=0.1, adaptive_budget=has_budget_predictor,
    )
    model.load_state_dict(state_dict, strict=False)
    model.eval().to(device)
    return model


def run_eval(model, dataset, device, n, mask_indices=None, mask_label="none", progress_every=25):
    scores = []
    preds_out = []
    answers_out = []
    if mask_indices is not None:
        mask_idx_tensor = torch.tensor(mask_indices, dtype=torch.long)

    print(f"  Starting eval pass: mask={mask_label}, n={n}", flush=True)
    for i in range(n):
        sample = dataset[i]
        images_coarse = sample["images_coarse"].unsqueeze(0).to(device)
        images_full = sample["images_full"].unsqueeze(0).to(device)
        question = sample["questions"]
        answer = str(sample.get("answer_str", ""))
        answers = sample.get("answers", [answer])
        if isinstance(answers, str):
            answers = [answers]

        handle = None
        if mask_indices is not None:
            dz = mask_idx_tensor

            def hook_fn(module, input, output):
                if not isinstance(output, tuple):
                    out = output.clone()
                    out[:, dz.to(out.device)] = -1e9
                    return out
                idx_out, s_N = output[0], output[1]
                rest = output[2:]
                s_mod = s_N.clone()
                s_mod[:, dz.to(s_mod.device)] = -1e9
                k = idx_out.shape[-1]
                new_idx = torch.topk(s_mod, k, dim=-1).indices
                new_idx = torch.sort(new_idx, dim=-1).values
                B, N = (s_N.shape[0], s_N.shape[1])
                new_mask = torch.zeros(B, N, device=s_N.device)
                new_mask.scatter_(-1, new_idx, 1.0)
                return (new_idx, s_mod, new_mask) + rest[1:]

            handle = model.tpn.register_forward_hook(hook_fn)

        with torch.no_grad():
            preds = model.generate(
                images_coarse=images_coarse,
                images_full=images_full,
                questions=[question],
                max_new_tokens=20,
        )
        pred = preds[0]
        if handle is not None:
            handle.remove()

        preds_out.append(pred)
        answers_out.append([str(a) for a in answers])
        scores.append(vqa_score(pred, answer))
        if (i + 1) % progress_every == 0 or (i + 1) == n:
            print(f"  [{i+1}/{n}] running mean={np.mean(scores):.4f} "
                  f"(mask={mask_label})", flush=True)
    return scores, preds_out, answers_out


def bootstrap_ci(diffs, n_boot=1000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    diffs = np.asarray(diffs)
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(diffs, size=len(diffs), replace=True)
        boots.append(sample.mean())
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def load_baseline_cache(path):
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        if "baseline" not in data:
            raise ValueError(f"Baseline cache {path} does not contain a 'baseline' key.")
        return data["baseline"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported baseline cache format in {path}.")


def summarize_delta(label, scores, baseline_scores):
    mean = float(np.mean(scores))
    base_mean = float(np.mean(baseline_scores))
    diffs = [s - b for s, b in zip(scores, baseline_scores)]
    delta = mean - base_mean
    lo, hi = bootstrap_ci(diffs)
    print(f"{label:<18} mean={mean:.4f}  delta={delta:+.4f}  "
          f"95% CI=[{lo:+.4f}, {hi:+.4f}]", flush=True)
    return delta, (lo, hi), diffs


def validate_results(results, bounds, expected_n=500):
    assert "baseline" in results, "Missing baseline scores."
    assert len(results["baseline"]) == expected_n, (
        f"Expected {expected_n} baseline scores, got {len(results['baseline'])}."
    )
    lengths = {name: len(scores) for name, scores in results.items()}
    assert len(set(lengths.values())) == 1, f"Score lengths differ: {lengths}"
    dead_zone = tuple(bounds["dead_zone"])
    for name, b in bounds.items():
        if name.startswith("random_"):
            assert not bbox_intersects(tuple(b), dead_zone), (
                f"{name} bounds {b} intersect dead zone {dead_zone}."
            )
    print("VALIDATION PASS", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", default=None)
    p.add_argument("--n-samples", type=int, default=500)
    p.add_argument("--device", default=None)
    p.add_argument("--llava-model", default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--mask-mode", choices=["dead_zone", "random", "both"],
                   default="both",
                   help="dead_zone: M1; random: M3 control; both: run all three.")
    p.add_argument("--n-random", type=int, default=10,
                   help="Number of random rectangles to test.")
    p.add_argument("--random-seed-start", type=int, default=0)
    p.add_argument("--baseline-cache", default=None,
                   help="Optional JSON path with cached baseline scores.")
    p.add_argument("--save-pairs", required=True,
                   help="JSON path to save per-sample paired scores.")
    p.add_argument("--progress-every", type=int, default=25,
                   help="Print progress every N samples within each eval pass.")
    args = p.parse_args()

    data_root = args.data_root or os.environ.get("DATA_ROOT")
    if not data_root:
        raise SystemExit("--data-root or $DATA_ROOT required")

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device, args.llava_model)

    dataset = GQADataset(
        questions_path=f"{data_root}/gqa/val_balanced_questions.json",
        image_dir=f"{data_root}/gqa/images",
    )
    n = min(args.n_samples, len(dataset))

    results = {}
    bounds = {"dead_zone": [0, 4, 6, 11]}

    if args.baseline_cache:
        print(f"\n=== Baseline (cached): {args.baseline_cache} ===", flush=True)
        base_scores = load_baseline_cache(args.baseline_cache)
        assert len(base_scores) == n, (
            f"Cached baseline length {len(base_scores)} does not match n={n}."
        )
    else:
        print(f"\n=== Baseline (no mask) ===", flush=True)
        base_scores, base_preds, answers = run_eval(model, dataset, device, n, None, "none", args.progress_every)
    results["baseline"] = base_scores
    preds_by_label = {}
    if not args.baseline_cache:
        preds_by_label["preds_baseline"] = base_preds
        results["answers"] = answers

    if args.mask_mode in ("dead_zone", "both"):
        print(f"\n=== Masked dead zone (rows 0-4, cols 6-11) ===", flush=True)
        dz_scores, dz_preds, dz_answers = run_eval(model, dataset, device, n,
                                                   dead_zone_flat_indices(), "dead_zone", args.progress_every)
        results["dead_zone"] = dz_scores
        preds_by_label["preds_dead_zone"] = dz_preds
        if "answers" not in results:
            results["answers"] = dz_answers

    if args.mask_mode in ("random", "both") and args.n_random > 0:
        rects = random_rectangles(args.n_random, seed=args.random_seed_start)
        for j, (rand_idx, rand_bounds) in enumerate(rects):
            rr_lo, rr_hi, rc_lo, rc_hi = rand_bounds
            key = f"random_{j}"
            bounds[key] = [rr_lo, rr_hi, rc_lo, rc_hi]
            print(f"\n=== Masked random rect {j} (rows {rr_lo}-{rr_hi}, "
                  f"cols {rc_lo}-{rc_hi}; M3 control) ===", flush=True)
            rd_scores, rd_preds, rd_answers = run_eval(model, dataset, device, n, rand_idx, key, args.progress_every)
            results[key] = rd_scores
            preds_by_label[f"preds_{key}"] = rd_preds
            if "answers" not in results:
                results["answers"] = rd_answers

    print(f"\n{'='*60}", flush=True)
    base_mean = float(np.mean(base_scores))
    print(f"Baseline VQA-score:  {base_mean:.4f}  (n={n})", flush=True)

    dz_delta = None
    dz_diffs = None
    if "dead_zone" in results:
        dz_delta, _, dz_diffs = summarize_delta("dead_zone", results["dead_zone"], base_scores)

    random_keys = sorted(
        [k for k in results if k.startswith("random_")],
        key=lambda x: int(x.split("_")[1]),
    )
    random_deltas = []
    random_diffs = {}
    for key in random_keys:
        delta, _, diffs = summarize_delta(key, results[key], base_scores)
        random_deltas.append(delta)
        random_diffs[key] = diffs

    if random_deltas:
        rd = np.asarray(random_deltas, dtype=float)
        p5, p50, p95 = np.percentile(rd, [5, 50, 95])
        print("\nRandom-rectangle delta distribution:")
        print(f"  mean={rd.mean():+.4f}  std={rd.std(ddof=0):.4f}  "
              f"min={rd.min():+.4f}  max={rd.max():+.4f}")
        print(f"  p05={p5:+.4f}  p50={p50:+.4f}  p95={p95:+.4f}")
        if dz_delta is not None:
            percentile = 100.0 * float(np.mean(rd <= dz_delta))
            print(f"  Dead-zone percentile rank vs random deltas: {percentile:.1f}")

        if "dead_zone" in results:
            print("\nPaired differential: dead_zone - random_rect")
            for key in random_keys:
                diffs = [
                    dz - rr
                    for dz, rr in zip(results["dead_zone"], results[key])
                ]
                mean_diff = float(np.mean(diffs))
                lo, hi = bootstrap_ci(diffs)
                print(f"  {key:<10} mean={mean_diff:+.4f}  "
                      f"95% CI=[{lo:+.4f}, {hi:+.4f}]")

    validate_results(results, bounds, expected_n=500)

    out_data = dict(results)
    out_data.update(preds_by_label)
    out_data["bounds"] = bounds
    Path(args.save_pairs).parent.mkdir(parents=True, exist_ok=True)
    with open(args.save_pairs, "w") as f:
        json.dump(out_data, f)
    print(f"\nSaved per-sample paired scores: {args.save_pairs}")


if __name__ == "__main__":
    main()
