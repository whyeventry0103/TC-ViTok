#!/usr/bin/env python3
"""Evaluate TC-ViTok on VQAv2, GQA, and POPE.

Sets HF offline mode before any imports (prevents NCCL desync from HTTP timeouts — BUG-17).

Usage:
    # TC-ViTok (main model):
    python scripts/evaluate.py \\
        --checkpoint runs/phase1/best.ckpt \\
        --benchmarks vqav2 gqa pope \\
        --spatial-eval \\
        --data-root $DATA_ROOT \\
        --output runs/eval_results/tcvitok_k144.json

    # Random-drop baseline:
    python scripts/evaluate.py \\
        --checkpoint runs/phase1/best.ckpt \\
        --baseline random_drop \\
        --benchmarks vqav2 gqa pope \\
        --data-root $DATA_ROOT \\
        --output runs/eval_results/random_drop_k144.json

    # Image-only TPN baseline (ablation A3):
    python scripts/evaluate.py \\
        --checkpoint runs/phase1/best.ckpt \\
        --baseline image_only \\
        --benchmarks vqav2 gqa pope \\
        --data-root $DATA_ROOT \\
        --output runs/eval_results/image_only_k144.json

Paper reference: Section 8 (Experimental Design)
"""

import os
import sys

# ── Offline mode — MUST be set before any HuggingFace import ─────────────────
# Prevents background threads from hitting HF Hub (429 rate-limit noise) and
# prevents HTTP timeouts that can stall one rank during DDP (BUG-17).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import argparse
import json
import re
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── Add src/ to path ─────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from tc_vitok.models.tc_vitok_model import TCViTokModel, gather_tokens
from tc_vitok.data.datasets import (
    VQAv2Dataset, GQADataset, POPEDataset, collate_fn,
)
from tc_vitok.evaluation.metrics import (
    vqa_score, gqa_accuracy, pope_f1, normalize_answer,
)
from tc_vitok.data.spatial_labels import classify_spatial


# ─────────────────────────────────────────────────────────────────────────────
# Baseline model variants
# ─────────────────────────────────────────────────────────────────────────────

class RandomDropModel(TCViTokModel):
    """Baseline: uniformly random token selection (no TPN).

    Selects k tokens uniformly at random from N=576, bypassing TPN entirely.
    All other components (encoders, LLM) are identical to TC-ViTok.
    Corresponds to ablation A2 in the paper.
    """

    def _select_tokens(self, images_coarse, images_full, questions):
        v_full = self.full_encoder(images_full)
        N = v_full.shape[1]
        B = v_full.shape[0]
        idx = torch.stack([
            torch.randperm(N, device=v_full.device)[:self.k]
            for _ in range(B)
        ])
        return gather_tokens(v_full, idx)


class ImageOnlyTPNModel(TCViTokModel):
    """Baseline: TPN with zeroed query embedding (image-only selection).

    Passes zero query embedding to TPN, removing language conditioning.
    TPN still runs and uses its trained weights, but sees no text signal —
    selection becomes purely content-based (no query conditioning).
    Corresponds to ablation A3 in the paper.
    """

    def _select_tokens(self, images_coarse, images_full, questions):
        v_coarse = self.coarse_encoder(images_coarse)
        B = v_coarse.shape[0]
        q_embed_zero = torch.zeros(
            B, 512, device=v_coarse.device, dtype=v_coarse.dtype
        )
        idx, _, __ = self.tpn(
            v_coarse=v_coarse,
            q_embed=q_embed_zero,
            k=self.k,
            tau=0.1,
            training=False,
        )
        v_full = self.full_encoder(images_full)
        return gather_tokens(v_full, idx)


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(
    checkpoint_path: str,
    device: str,
    k: int = 144,
    baseline: str = "none",
    llava_model_name: str = "llava-hf/llava-1.5-7b-hf",
) -> TCViTokModel:
    """Load TCViTokModel from Phase 1 checkpoint.

    The checkpoint contains only trainable weights (tpn.*,
    lora_*, multi_modal_projector.*). The 7B LLaVA backbone is re-loaded
    from HF cache (offline) and trainable weights are overlaid via strict=False.
    """
    # Wiring config matching Phase 1 training exactly
    tpn_config = {
        "n_heads": 4,
        "d_model": 384,
        "d_coarse": 384,
        "d_h": 512,
        "d_text": 512,
        "M": 49,
        "N": 576,
    }
    lora_config = {
        "rank": 32,
        "alpha": 64,
        "targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "dropout": 0.05,
    }

    model_cls = {
        "none": TCViTokModel,
        "random_drop": RandomDropModel,
        "image_only": ImageOnlyTPNModel,
    }[baseline]

    # Auto-detect LoRA rank/alpha/targets from checkpoint before building model
    print(f"[load_model] Pre-loading checkpoint to detect LoRA config: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    for key, val in state_dict.items():
        if "lora_A" in key and "q_proj" in key:
            lora_config["rank"] = val.shape[0]
            lora_config["alpha"] = val.shape[0] * 2
            break
    lora_target_set = {"q_proj", "k_proj", "v_proj", "o_proj"}
    for key in state_dict:
        if "lora_A" in key:
            for t in ("gate_proj", "up_proj", "down_proj"):
                if t in key:
                    lora_target_set.add(t)
    lora_config["targets"] = sorted(lora_target_set)

    print(f"[load_model] baseline={baseline}, k={k}, device={device}")
    print(f"[load_model] Detected LoRA rank={lora_config['rank']}, "
          f"alpha={lora_config['alpha']}, targets={lora_config['targets']}")
    print(f"[load_model] Loading LLaVA backbone from HF cache (offline)...")
    t0 = time.time()

    model = model_cls(
        k=k,
        tpn_config=tpn_config,
        use_stub_llm=False,
        llava_model_name=llava_model_name,
        lora_config=lora_config,
        load_encoders=True,
    )

    print(f"[load_model] Backbone loaded in {time.time()-t0:.1f}s")
    print(f"[load_model] Overlaying checkpoint weights...")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    # Sanity check: TPN weights must be present
    tpn_keys  = [k for k in state_dict if k.startswith("tpn.")]
    lora_keys = [k for k in state_dict if "lora_" in k]
    mm_keys   = [k for k in state_dict if "multi_modal_projector." in k]
    if not tpn_keys:
        raise RuntimeError(
            f"No TPN weights found in checkpoint. "
            f"Keys in checkpoint: {list(state_dict.keys())[:10]}"
        )
    print(f"[load_model] Checkpoint key counts: "
          f"TPN={len(tpn_keys)} | LoRA={len(lora_keys)} | mm_proj={len(mm_keys)}")
    print(f"[load_model] load_state_dict: "
          f"missing={len(missing)} (expected: frozen backbone) | unexpected={len(unexpected)}")

    model.eval()
    model = model.to(device)
    print(f"[load_model] Model ready on {device}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Answer extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

def extract_yes_no(raw_answer: str) -> str:
    """Extract 'yes' or 'no' from a generated answer string.

    Handles cases like "Yes, there is a dog" → "yes" and
    "No, I don't see one" → "no".
    """
    if not raw_answer.strip():
        return "no"
    first_word = raw_answer.strip().lower().split()[0]
    first_word = re.sub(r"[^\w]", "", first_word)
    return "yes" if first_word.startswith("yes") else "no"


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark evaluation functions
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_vqav2(model, data_root, device, batch_size, max_samples=None, log_interval=200):
    """Evaluate on VQAv2 val2014 split using generative LM decoding.

    Calls model.generate() to produce free-form answer strings, then scores
    against the 10 ground-truth annotator answers using the VQA soft accuracy
    metric. This matches the paper's evaluation protocol (Section 8.2).
    """
    print("\n[VQAv2] Loading val2014 dataset...")
    dataset = VQAv2Dataset(
        annotations_path=f"{data_root}/vqav2/v2_mscoco_val2014_annotations.json",
        questions_path=f"{data_root}/vqav2/v2_OpenEnded_mscoco_val2014_questions.json",
        image_dir=f"{data_root}/vqav2/val2014",
        split="val",
        max_samples=max_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    print(f"[VQAv2] {len(dataset)} samples, {len(loader)} batches")

    scores = []
    per_question = []

    for step, batch in enumerate(tqdm(loader, desc="VQAv2 val")):
        with torch.no_grad():
            preds = model.generate(
                images_coarse=batch["images_coarse"].to(device),
                images_full=batch["images_full"].to(device),
                questions=batch["questions"],
                max_new_tokens=20,
            )  # List[str]

        # Diagnostic: first batch — show a few samples
        if step == 0:
            for i in range(min(4, len(preds))):
                gt_show = batch["answers"][i][:3] if isinstance(batch["answers"][i], list) else [batch["answers"][i]]
                score_i = vqa_score(preds[i], batch["answers"][i])
                print(f"  [diag] pred={repr(preds[i])} | gt={gt_show} | score={score_i:.3f}", flush=True)

        for pred, gt_answers, qid in zip(preds, batch["answers"], batch["question_id"]):
            score = vqa_score(pred, gt_answers)
            scores.append(score)
            per_question.append({
                "question_id": qid,
                "pred": pred,
                "score": round(score, 4),
            })

        if log_interval > 0 and (step + 1) % log_interval == 0:
            running = sum(scores) / len(scores) * 100
            print(
                f"[VQAv2] step {step+1}/{len(loader)} | "
                f"VQA-Score: {running:.2f}% | "
                f"samples: {len(scores)}",
                flush=True,
            )

    vqa_score_pct = sum(scores) / len(scores) * 100
    print(f"[VQAv2] FINAL VQA-Score: {vqa_score_pct:.2f}%  ({len(scores)} questions)")
    return {
        "vqa_score": round(vqa_score_pct, 2),
        "num_samples": len(scores),
        "per_question": per_question,
    }


def evaluate_gqa(model, data_root, device, batch_size, max_samples=None, spatial_eval=True, log_interval=200):
    """Evaluate on GQA val_balanced with optional spatial/global split.

    Uses model.generate() to produce free-form answer strings, scored against
    the single ground-truth GQA answer. Matches paper evaluation (Section 8.3).
    """
    print("\n[GQA] Loading val_balanced dataset...")
    dataset = GQADataset(
        questions_path=f"{data_root}/gqa/val_balanced_questions.json",
        image_dir=f"{data_root}/gqa/images",
        max_samples=max_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    print(f"[GQA] {len(dataset)} samples, {len(loader)} batches")

    all_correct = []
    spatial_correct = []
    global_correct = []
    per_question = []

    for step, batch in enumerate(tqdm(loader, desc="GQA val_balanced")):
        with torch.no_grad():
            preds = model.generate(
                images_coarse=batch["images_coarse"].to(device),
                images_full=batch["images_full"].to(device),
                questions=batch["questions"],
                max_new_tokens=20,
            )  # List[str]

        if step == 0:
            for i in range(min(4, len(preds))):
                gt_show = batch["answers"][i]
                gt = gt_show[0] if isinstance(gt_show, list) else gt_show
                print(f"  [diag] pred={repr(preds[i])} | gt={repr(gt)}", flush=True)

        for i, (pred, gt_answers, qid) in enumerate(
            zip(preds, batch["answers"], batch["question_id"])
        ):
            gt = gt_answers[0] if isinstance(gt_answers, list) else gt_answers
            correct = gqa_accuracy(pred, gt)
            all_correct.append(correct)

            is_spatial = False
            if spatial_eval and "gqa_annotation" in batch:
                is_spatial = classify_spatial(
                    batch["questions"][i],
                    batch["gqa_annotation"][i],
                )
            if is_spatial:
                spatial_correct.append(correct)
            else:
                global_correct.append(correct)

            per_question.append({
                "question_id": qid,
                "pred": pred,
                "gt": gt,
                "correct": bool(correct),
                "is_spatial": is_spatial,
            })

        if log_interval > 0 and (step + 1) % log_interval == 0:
            overall_run = sum(all_correct) / len(all_correct) * 100
            sp_str = ""
            if spatial_correct:
                sp_acc = sum(spatial_correct) / len(spatial_correct) * 100
                gl_acc = sum(global_correct) / len(global_correct) * 100 if global_correct else 0.0
                sp_str = f" | Spatial: {sp_acc:.2f}% ({len(spatial_correct)}) | Global: {gl_acc:.2f}% ({len(global_correct)})"
            print(
                f"[GQA] step {step+1}/{len(loader)} | "
                f"Overall: {overall_run:.2f}% | samples: {len(all_correct)}{sp_str}",
                flush=True,
            )

    overall_acc = sum(all_correct) / len(all_correct) * 100
    result = {
        "overall_accuracy": round(overall_acc, 2),
        "num_samples": len(all_correct),
    }
    if spatial_eval and spatial_correct:
        spatial_acc = sum(spatial_correct) / len(spatial_correct) * 100
        global_acc = sum(global_correct) / len(global_correct) * 100 if global_correct else 0.0
        result["spatial_accuracy"] = round(spatial_acc, 2)
        result["global_accuracy"] = round(global_acc, 2)
        result["spatial_count"] = len(spatial_correct)
        result["global_count"] = len(global_correct)
        print(f"[GQA] FINAL Overall: {overall_acc:.2f}% | "
              f"Spatial: {spatial_acc:.2f}% ({len(spatial_correct)}) | "
              f"Global: {global_acc:.2f}% ({len(global_correct)})")
    else:
        print(f"[GQA] FINAL Overall accuracy: {overall_acc:.2f}%  ({len(all_correct)} samples)")

    result["per_question"] = per_question
    return result


def evaluate_pope(model, data_root, device, batch_size, max_samples=None, log_interval=200):
    """Evaluate on all three POPE variants (popular, random, adversarial).

    POPE images are COCO val2014 images — same directory as VQAv2 val.
    Uses model.generate() to produce yes/no answers, then extracts the first
    word for binary classification. Returns per-variant F1 and macro-average F1.
    """
    # POPE uses COCO val2014 images (NOT a separate coco/val2014 directory)
    image_dir = f"{data_root}/vqav2/val2014"
    pope_dir = f"{data_root}/pope"

    variants = ["popular", "random", "adversarial"]
    results = {}

    for variant in variants:
        pope_path = f"{pope_dir}/coco_pope_{variant}.jsonl"
        if not Path(pope_path).exists():
            print(f"[POPE] WARNING: {pope_path} not found — skipping {variant}")
            continue

        print(f"\n[POPE/{variant}] Loading...")
        dataset = POPEDataset(
            pope_path=pope_path,
            image_dir=image_dir,
            max_samples=max_samples,
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        print(f"[POPE/{variant}] {len(dataset)} samples")

        all_preds = []
        all_gt = []

        pope_diag_done = False
        for step, batch in enumerate(tqdm(loader, desc=f"POPE/{variant}")):
            with torch.no_grad():
                raw_answers = model.generate(
                    images_coarse=batch["images_coarse"].to(device),
                    images_full=batch["images_full"].to(device),
                    questions=batch["questions"],
                    max_new_tokens=5,  # POPE answers are "yes" or "no"
                )  # List[str]

            # Extract yes/no from the first word of each generated answer
            preds = [extract_yes_no(a) for a in raw_answers]

            gt = [
                (a[0] if isinstance(a, list) else a).lower()
                for a in batch["answers"]
            ]
            all_preds.extend(preds)
            all_gt.extend(gt)

            if not pope_diag_done and step == 0:
                for i in range(min(4, len(preds))):
                    print(
                        f"  [diag] raw={repr(raw_answers[i])} | "
                        f"pred={repr(preds[i])} | gt={repr(gt[i])}",
                        flush=True,
                    )
                pope_diag_done = True

            if log_interval > 0 and (step + 1) % log_interval == 0:
                running = pope_f1(all_preds, all_gt)
                print(
                    f"[POPE/{variant}] step {step+1}/{len(loader)} | "
                    f"F1: {running['f1']*100:.2f}% | "
                    f"Prec: {running['precision']*100:.2f}% | "
                    f"Rec: {running['recall']*100:.2f}% | "
                    f"samples: {len(all_preds)}",
                    flush=True,
                )

        metrics = pope_f1(all_preds, all_gt)
        results[variant] = {
            "f1": round(metrics["f1"] * 100, 2),
            "precision": round(metrics["precision"] * 100, 2),
            "recall": round(metrics["recall"] * 100, 2),
            "accuracy": round(metrics["accuracy"] * 100, 2),
            "num_samples": len(all_preds),
        }
        print(f"[POPE/{variant}] F1={results[variant]['f1']:.2f}% | "
              f"Prec={results[variant]['precision']:.2f}% | "
              f"Rec={results[variant]['recall']:.2f}%")

    # Macro-average F1 across all available variants
    f1s = [v["f1"] for v in results.values()]
    results["macro_f1"] = round(sum(f1s) / len(f1s), 2) if f1s else 0.0
    print(f"[POPE] Macro-average F1: {results['macro_f1']:.2f}%")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate TC-ViTok on VQA benchmarks")
    parser.add_argument("--checkpoint", required=True, help="Path to best.ckpt")
    parser.add_argument(
        "--benchmarks", nargs="+", default=["vqav2", "gqa", "pope"],
        choices=["vqav2", "gqa", "pope"],
        help="Benchmarks to run",
    )
    parser.add_argument(
        "--baseline", default="none",
        choices=["none", "random_drop", "image_only"],
        help="Baseline variant (none = full TC-ViTok)",
    )
    parser.add_argument("--k", type=int, default=144, help="Token budget")
    parser.add_argument("--data-root", default=None,
                        help="Dataset root (default: $DATA_ROOT env var)")
    parser.add_argument("--device", default=None,
                        help="Inference device (default: cuda if available)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap samples per benchmark (for quick smoke tests)")
    parser.add_argument("--spatial-eval", action="store_true",
                        help="Enable spatial/global split on GQA")
    parser.add_argument("--log-interval", type=int, default=200,
                        help="Print running accuracy every N batches (default: 200)")
    parser.add_argument("--output", default="runs/eval_results/results.json",
                        help="Output JSON path")
    parser.add_argument("--llava-model", default="llava-hf/llava-1.5-7b-hf",
                        help="LLaVA model name in HF cache")
    return parser.parse_args()


def detect_device():
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def main():
    args = parse_args()

    # Resolve data_root
    data_root = args.data_root or os.environ.get("DATA_ROOT")
    if not data_root:
        print("ERROR: --data-root or $DATA_ROOT must be set")
        sys.exit(1)
    data_root = str(Path(data_root).expanduser())

    # Wire HF cache dirs from data_root (mirrors train.py)
    os.environ.setdefault("HF_HOME", f"{data_root}/hf_cache")
    os.environ.setdefault("TRANSFORMERS_CACHE", f"{data_root}/hf_cache")
    os.environ.setdefault("TORCH_HOME", f"{data_root}/torch_cache")

    device = args.device or detect_device()

    print("=" * 60)
    print("TC-ViTok Evaluation")
    print("=" * 60)
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Baseline   : {args.baseline}")
    print(f"  Benchmarks : {args.benchmarks}")
    print(f"  k          : {args.k}")
    print(f"  Device     : {device}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  Data root  : {data_root}")
    print(f"  Output     : {args.output}")
    print("=" * 60)

    # Load model once — reuse across all benchmarks
    model = load_model(
        checkpoint_path=args.checkpoint,
        device=device,
        k=args.k,
        baseline=args.baseline,
        llava_model_name=args.llava_model,
    )

    results = {
        "checkpoint": args.checkpoint,
        "baseline": args.baseline,
        "k": args.k,
        "device": device,
        "benchmarks": {},
    }

    t_start = time.time()

    if "vqav2" in args.benchmarks:
        results["benchmarks"]["vqav2"] = evaluate_vqav2(
            model, data_root, device, args.batch_size, args.max_samples,
            log_interval=args.log_interval,
        )

    if "gqa" in args.benchmarks:
        results["benchmarks"]["gqa"] = evaluate_gqa(
            model, data_root, device, args.batch_size,
            args.max_samples, spatial_eval=args.spatial_eval,
            log_interval=args.log_interval,
        )

    if "pope" in args.benchmarks:
        results["benchmarks"]["pope"] = evaluate_pope(
            model, data_root, device, args.batch_size, args.max_samples,
            log_interval=args.log_interval,
        )

    results["eval_time_minutes"] = round((time.time() - t_start) / 60, 1)

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Strip per_question from saved file to keep it small, save separately
    save_results = {k: v for k, v in results.items()}
    per_q_path = out_path.with_suffix(".per_question.json")
    per_question_data = {}
    for bench, bench_res in results["benchmarks"].items():
        if "per_question" in bench_res:
            per_question_data[bench] = bench_res.pop("per_question")

    with open(out_path, "w") as f:
        json.dump(save_results, f, indent=2)

    if per_question_data:
        with open(per_q_path, "w") as f:
            json.dump(per_question_data, f, indent=2)
        print(f"\nPer-question results saved to {per_q_path}")

    print(f"\nResults saved to {out_path}")

    # ── Summary table ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Baseline   : {args.baseline}")
    print(f"  k          : {args.k}")
    print(f"  Eval time  : {results['eval_time_minutes']} min")
    print()

    if "vqav2" in results["benchmarks"]:
        r = results["benchmarks"]["vqav2"]
        print(f"  VQAv2 val  : {r['vqa_score']:.2f}%  (paper target: ~76%)")

    if "gqa" in results["benchmarks"]:
        r = results["benchmarks"]["gqa"]
        print(f"  GQA overall: {r['overall_accuracy']:.2f}%  (paper target: ~61%)")
        if "spatial_accuracy" in r:
            print(f"  GQA spatial: {r['spatial_accuracy']:.2f}%  (paper target: ~59%)")
            print(f"  GQA global : {r['global_accuracy']:.2f}%")

    if "pope" in results["benchmarks"]:
        r = results["benchmarks"]["pope"]
        print(f"  POPE macro F1: {r['macro_f1']:.2f}%  (paper target: ~84%)")
        for v in ["popular", "random", "adversarial"]:
            if v in r:
                print(f"    {v:12s}: F1={r[v]['f1']:.2f}%")

    print("=" * 60)


if __name__ == "__main__":
    main()
