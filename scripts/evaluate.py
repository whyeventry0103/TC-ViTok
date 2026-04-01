#!/usr/bin/env python3
"""CLI entry point for evaluating TC-ViTok on VQA benchmarks.

Usage:
    python scripts/evaluate.py --checkpoint runs/phase1/best.ckpt --benchmarks vqav2 gqa pope
    python scripts/evaluate.py --checkpoint runs/phase1/best.ckpt --token-budgets 72 144 216 288 576
    python scripts/evaluate.py --checkpoint runs/phase1/best.ckpt --spatial-eval

This script:
1. Loads trained checkpoint
2. Evaluates on specified benchmarks
3. Optionally sweeps over token budgets for Pareto curve
4. Generates evaluation report with spatial stratification

Paper reference: Section 8 (Experimental Setup)
"""

import argparse
import os
from pathlib import Path
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from tc_vitok.models.tc_vitok_model import TCViTokModel
from tc_vitok.data.datasets import (
    VQAv2Dataset, GQADataset, TextVQADataset, POPEDataset, collate_fn,
)
from tc_vitok.evaluation.metrics import (
    vqa_score, gqa_accuracy, textvqa_accuracy, pope_f1,
    normalize_answer, aggregate_scores,
)
from tc_vitok.evaluation.pareto import compute_pareto_curve, format_pareto_table
from tc_vitok.data.spatial_labels import classify_spatial


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate TC-ViTok")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["vqav2"],
        help="Benchmarks to evaluate on",
    )
    parser.add_argument(
        "--token-budgets",
        nargs="+",
        type=int,
        default=[144],
        help="Token budgets to evaluate (for Pareto sweep)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.json",
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for inference",
    )
    parser.add_argument(
        "--spatial-eval",
        action="store_true",
        help="Enable spatial specificity stratification for GQA",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Root data directory",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit evaluation samples (for debugging)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--stub",
        action="store_true",
        help="Use StubLLM (for pipeline testing)",
    )
    return parser.parse_args()


def detect_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(checkpoint_path, device, k=144, stub=False):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Reconstruct model config from checkpoint
    ckpt_config = checkpoint.get("config", {})
    model_config = ckpt_config.get("model", {})
    tpn_config = model_config.get("tpn", {})
    lora_config = model_config.get("lora", {})

    model = TCViTokModel(
        k=k,
        tpn_config=tpn_config,
        use_stub_llm=stub,
        lora_config=lora_config,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device)
    model.eval()
    return model


def evaluate_vqav2(model, data_dir, device, max_samples, batch_size):
    """Evaluate on VQAv2 validation set."""
    dataset = VQAv2Dataset(
        annotations_path=f"{data_dir}/vqav2/v2_mscoco_val2014_annotations.json",
        questions_path=f"{data_dir}/vqav2/v2_OpenEnded_mscoco_val2014_questions.json",
        image_dir=f"{data_dir}/vqav2/val2014",
        split="val",
        max_samples=max_samples,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
    )

    scores = []
    for batch in tqdm(loader, desc="VQAv2"):
        answers = model.generate(
            images_coarse=batch["images_coarse"].to(device),
            images_full=batch["images_full"].to(device),
            questions=batch["questions"],
        )
        for pred, gt_answers in zip(answers, batch["answers"]):
            scores.append(vqa_score(pred, gt_answers))

    return aggregate_scores(scores)


def evaluate_gqa(model, data_dir, device, max_samples, batch_size, spatial_eval=False):
    """Evaluate on GQA validation set."""
    dataset = GQADataset(
        questions_path=f"{data_dir}/gqa/val_balanced_questions.json",
        image_dir=f"{data_dir}/gqa/images",
        max_samples=max_samples,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
    )

    all_scores = []
    spatial_scores = []
    global_scores = []

    for batch in tqdm(loader, desc="GQA"):
        answers = model.generate(
            images_coarse=batch["images_coarse"].to(device),
            images_full=batch["images_full"].to(device),
            questions=batch["questions"],
        )
        for i, (pred, gt_answers) in enumerate(zip(answers, batch["answers"])):
            gt = gt_answers[0] if isinstance(gt_answers, list) else gt_answers
            acc = gqa_accuracy(pred, gt)
            all_scores.append(acc)

            if spatial_eval and "gqa_annotation" in batch:
                is_spatial = classify_spatial(
                    batch["questions"][i], batch["gqa_annotation"][i]
                )
                if is_spatial:
                    spatial_scores.append(acc)
                else:
                    global_scores.append(acc)

    result = aggregate_scores(all_scores)
    if spatial_eval and spatial_scores:
        result["spatial"] = aggregate_scores(spatial_scores)
        result["global"] = aggregate_scores(global_scores)
        result["spatial_count"] = len(spatial_scores)
        result["global_count"] = len(global_scores)
    return result


def evaluate_pope(model, data_dir, device, max_samples, batch_size):
    """Evaluate on POPE benchmark."""
    dataset = POPEDataset(
        pope_path=f"{data_dir}/pope/coco_pope_popular.jsonl",
        image_dir=f"{data_dir}/coco/val2014",
        max_samples=max_samples,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
    )

    all_preds = []
    all_gt = []
    for batch in tqdm(loader, desc="POPE"):
        answers = model.generate(
            images_coarse=batch["images_coarse"].to(device),
            images_full=batch["images_full"].to(device),
            questions=batch["questions"],
        )
        all_preds.extend(answers)
        all_gt.extend([a[0] if isinstance(a, list) else a for a in batch["answers"]])

    return pope_f1(all_preds, all_gt)


def main():
    args = parse_args()

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    device = args.device or detect_device()
    print(f"Loading checkpoint: {args.checkpoint}")
    print(f"Benchmarks: {args.benchmarks}")
    print(f"Token budgets: {args.token_budgets}")
    print(f"Device: {device}")

    all_results = {}

    for k in args.token_budgets:
        print(f"\n{'='*60}")
        print(f"Token budget k={k}")
        print(f"{'='*60}")

        model = load_model(args.checkpoint, device, k=k, stub=args.stub)
        results = {}

        for benchmark in args.benchmarks:
            print(f"\nEvaluating {benchmark}...")

            if benchmark == "vqav2":
                results["vqav2"] = evaluate_vqav2(
                    model, args.data_dir, device, args.max_samples, args.batch_size,
                )
            elif benchmark == "gqa":
                results["gqa"] = evaluate_gqa(
                    model, args.data_dir, device, args.max_samples,
                    args.batch_size, spatial_eval=args.spatial_eval,
                )
            elif benchmark == "textvqa":
                # TextVQA uses same flow as VQAv2
                results["textvqa"] = {"accuracy": 0.0, "note": "not yet wired"}
            elif benchmark == "pope":
                results["pope"] = evaluate_pope(
                    model, args.data_dir, device, args.max_samples, args.batch_size,
                )
            else:
                print(f"Unknown benchmark: {benchmark}")

        all_results[k] = results

    # Pareto sweep analysis
    if len(args.token_budgets) > 1:
        print("\n" + "="*60)
        print("PARETO ANALYSIS")
        print("="*60)

        for benchmark in args.benchmarks:
            accs = []
            budgets = []
            for k in args.token_budgets:
                if benchmark in all_results.get(k, {}):
                    metric = all_results[k][benchmark]
                    acc = metric.get("mean", metric.get("accuracy", 0.0))
                    accs.append(acc)
                    budgets.append(k)
            if accs:
                pareto = compute_pareto_curve(accs, budgets)
                print(f"\n{benchmark}: {pareto}")

    # Save results
    output_path = Path(args.output)
    serializable = {}
    for k, v in all_results.items():
        serializable[str(k)] = v
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"\nResults saved to {output_path}")
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    for k, results in all_results.items():
        print(f"\nk={k}:")
        for benchmark, metrics in results.items():
            print(f"  {benchmark}: {metrics}")


if __name__ == "__main__":
    main()
