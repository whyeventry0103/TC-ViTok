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
import random
import re
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── Add src/ to path ─────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from tc_vitok.models.tc_vitok_model import (
    TCViTokModel, gather_tokens, informative_token_merge,
)
from tc_vitok.data.datasets import (
    VQAv2Dataset, GQADataset, POPEDataset, TextVQADataset,
    MMBenchDataset, MMEDataset, collate_fn,
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
        if self.adaptive_budget:
            raise NotImplementedError(
                "RandomDropModel does not support adaptive budget; "
                "use --baseline none for adaptive eval."
            )
        v_full = self.full_encoder(images_full)
        N = v_full.shape[1]
        B = v_full.shape[0]
        idx = torch.stack([
            torch.randperm(N, device=v_full.device)[:self.k]
            for _ in range(B)
        ])
        return gather_tokens(v_full, idx), None


class UniformGridModel(TCViTokModel):
    """Baseline: deterministic uniform-grid selection at fixed k.

    Selects evenly spaced positions from the CLIP-L/14 24x24 patch sequence,
    bypassing the learned TPN. This preserves broad spatial coverage while
    keeping all downstream components identical to the evaluated checkpoint.
    """

    def _select_tokens(self, images_coarse, images_full, questions):
        if self.adaptive_budget:
            raise NotImplementedError(
                "UniformGridModel does not support adaptive budget; "
                "use --baseline none for adaptive eval."
        )
        v_full = self.full_encoder(images_full)
        B, N, _ = v_full.shape
        if self.k > N:
            raise ValueError(f"UniformGridModel expects k <= N; got N={N}, k={self.k}")
        if N % self.k == 0:
            stride = N // self.k
            idx = torch.arange(0, N, stride, device=v_full.device)[: self.k]
        else:
            idx = torch.linspace(0, N - 1, steps=self.k, device=v_full.device).round().long()
        idx = idx.unsqueeze(0).expand(B, -1).contiguous()
        return gather_tokens(v_full, idx), None


class CLSAttentionModel(TCViTokModel):
    """Baseline: CLIP last-layer CLS-to-patch attention top-k selection.

    Uses the frozen CLIP ViT-L/14 vision encoder's final-layer attention from
    the CLS token to patch tokens as a query-agnostic importance prior.
    Computes attention weights manually because SDPA attention may not return
    them even when output_attentions=True.
    """

    def _select_tokens(self, images_coarse, images_full, questions):
        if self.adaptive_budget:
            raise NotImplementedError(
                "CLSAttentionModel does not support adaptive budget; "
                "use --baseline none for adaptive eval."
            )
        clip_vision = self.full_encoder.clip.vision_model
        with torch.no_grad():
            outputs = clip_vision(
                pixel_values=images_full,
                output_hidden_states=True,
                return_dict=True,
            )
        v_full = outputs.last_hidden_state[:, 1:, :].contiguous()
        assert v_full.shape[1] == 576, (
            f"CLSAttentionModel expects 576 patch tokens, got {v_full.shape[1]}"
        )
        if outputs.hidden_states is None or len(outputs.hidden_states) < 2:
            raise RuntimeError("CLIP vision model did not return hidden states.")

        last_block = clip_vision.encoder.layers[-1]
        hidden = last_block.layer_norm1(outputs.hidden_states[-2])
        attn = last_block.self_attn
        q = attn.q_proj(hidden)
        k = attn.k_proj(hidden)
        B, S, D = q.shape
        num_heads = int(attn.num_heads)
        head_dim = D // num_heads
        scale = float(getattr(attn, "scale", head_dim ** -0.5))
        q = q.view(B, S, num_heads, head_dim).transpose(1, 2) * scale
        k = k.view(B, S, num_heads, head_dim).transpose(1, 2)
        last_attn = torch.matmul(q, k.transpose(-2, -1)).softmax(dim=-1)
        cls_to_patches = last_attn[:, :, 0, 1:].mean(dim=1)
        assert cls_to_patches.shape == (v_full.shape[0], v_full.shape[1]), (
            f"Unexpected CLS attention shape {tuple(cls_to_patches.shape)} "
            f"for visual tokens {tuple(v_full.shape)}"
        )
        idx = cls_to_patches.topk(self.k, dim=-1).indices
        return gather_tokens(v_full, idx), None


class ImageOnlyTPNModel(TCViTokModel):
    """Baseline: TPN with zeroed query embedding (image-only selection).

    Passes zero query embedding to TPN, removing language conditioning.
    TPN still runs and uses its trained weights, but sees no text signal —
    selection becomes purely content-based (no query conditioning).
    Corresponds to ablation A3 in the paper (also tests Hypothesis H2).
    """

    def _select_tokens(self, images_coarse, images_full, questions):
        v_coarse = self.coarse_encoder(images_coarse)
        B = v_coarse.shape[0]
        q_embed_zero = torch.zeros(
            B, 512, device=v_coarse.device, dtype=v_coarse.dtype
        )
        idx, s_N, _, k_eff = self.tpn(
            v_coarse=v_coarse,
            q_embed=q_embed_zero,
            k=self.k,
            tau=0.1,
            training=False,
            adaptive_budget=self.adaptive_budget,
            budget_predictor=self.budget_predictor,
            k_min=self.k_min,
            k_max=self.k_max,
        )
        v_full = self.full_encoder(images_full)
        if self.adaptive_budget:
            v_selected = self._gather_adaptive(v_full, idx, s_N)
            k_per_sample = (idx >= 0).sum(dim=1)
            return v_selected, k_per_sample
        if self.merge_beta > 0:
            return informative_token_merge(
                v_fine=v_full, idx=idx, s_N=s_N,
                beta=self.merge_beta, tau_merge=self.merge_tau,
            ), None
        return gather_tokens(v_full, idx), None


class Full576Model(TCViTokModel):
    """Control: same LoRA / mm_projector as TC-ViTok Final, but full 576 tokens.

    Bypasses the TPN, the token merge, and any selection. The fine encoder
    output is forwarded to the LLM unmodified. This isolates the contribution
    of compression from the contribution of LoRA fine-tuning — the missing
    control flagged by the prior reviewer report (Step 2 / W1).

    Reuses the Phase 1 Final LoRA adapter and mm_projector verbatim.
    Returns 576 tokens unchanged.
    """

    def _select_tokens(self, images_coarse, images_full, questions):
        if self.adaptive_budget:
            raise NotImplementedError(
                "Full576Model is the LoRA-576 control and must run with fixed "
                "k=576; pass --baseline full_576 without --adaptive-budget."
            )
        v_full = self.full_encoder(images_full)  # [B, 576, 1024]
        assert v_full.shape[1] == 576, (
            f"Full576Model expects 576 visual tokens, got {v_full.shape[1]}"
        )
        return v_full, None


class MergeDisabledModel(TCViTokModel):
    """Ablation: Phase 1 Final checkpoint with merge disabled at eval.

    Same TPN selection, same LoRA, same k, but beta=0 → no token merge.
    Isolates the contribution of the merge mechanism (true single-variable
    ablation; v3 vs Final changes too many things at once).
    """

    def _select_tokens(self, images_coarse, images_full, questions):
        if self.adaptive_budget:
            raise NotImplementedError(
                "MergeDisabledModel runs against the fixed-k Final checkpoint; "
                "use --baseline none for adaptive checkpoints."
            )
        v_coarse = self.coarse_encoder(images_coarse)
        q_embed = self.text_encoder(questions)
        idx, s_N, _, k_eff = self.tpn(
            v_coarse=v_coarse, q_embed=q_embed, k=self.k, tau=0.1, training=False,
            adaptive_budget=False,
            budget_predictor=None,
        )
        v_full = self.full_encoder(images_full)
        # Force beta=0: no merge, hard top-k only.
        return gather_tokens(v_full, idx), None


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(
    checkpoint_path: str,
    device: str,
    k: int = 144,
    baseline: str = "none",
    llava_model_name: str = "llava-hf/llava-1.5-7b-hf",
    merge_beta: float = 0.3,
    merge_tau: float = 0.1,
    adaptive_budget: bool = False,
    k_min: int = 36,
    k_max: int = 288,
    zero_query: bool = False,
    tpn_version: str = "auto",
    auto_adaptive_budget: bool = True,
    llava_attn_implementation: str | None = None,
) -> TCViTokModel:
    """Load TCViTokModel from Phase 1 checkpoint.

    The checkpoint contains only trainable weights (tpn.*,
    lora_*, multi_modal_projector.*). The 7B LLaVA backbone is re-loaded
    from HF cache (offline) and trainable weights are overlaid via strict=False.
    """
    print(f"[load_model] Pre-loading checkpoint to detect config: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)

    if tpn_version == "auto":
        v3_markers = (
            "tpn.proj_in.",
            "tpn.attn.",
            "tpn.score_head.",
            "tpn.norm_attn.",
        )
        tpn_version = "v3" if any(
            key.startswith(v3_markers) for key in state_dict
        ) else "v1"
        print(f"[load_model] Auto-detected TPN version: {tpn_version}")

    has_budget_predictor = any(
        key.startswith("budget_predictor.") for key in state_dict
    )
    if auto_adaptive_budget and baseline == "none" and has_budget_predictor and not adaptive_budget:
        adaptive_budget = True
        print("[load_model] Auto-enabled adaptive_budget from checkpoint keys")

    # Wiring config matching Phase 1 training. v1 is cross-attention TPN.
    # For v3 (image-only scorer), we pass a slimmer config (no d_coarse/d_text/M).
    if tpn_version == "v3":
        tpn_config = {
            "d_visual": 1024,
            "d_h": 256,
            "n_heads": 4,
            "dropout": 0.1,
            "N": 576,
        }
    else:
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
        "uniform_grid": UniformGridModel,
        "cls_attention": CLSAttentionModel,
        "image_only": ImageOnlyTPNModel,
        "full_576": Full576Model,
        "merge_disabled": MergeDisabledModel,
    }[baseline]

    # Auto-detect LoRA rank/alpha/targets from checkpoint before building model.
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

    # Merge config: 0 disables merge entirely (used by merge_disabled / full_576).
    # For the Final model checkpoint, the training config used beta=0.3, tau=0.1.
    hard_drop_baselines = (
        "merge_disabled", "full_576", "random_drop", "uniform_grid", "cls_attention"
    )
    effective_merge_beta = 0.0 if baseline in hard_drop_baselines else merge_beta

    model = model_cls(
        k=k,
        tpn_config=tpn_config,
        use_stub_llm=False,
        llava_model_name=llava_model_name,
        lora_config=lora_config,
        load_encoders=True,
        merge_beta=effective_merge_beta,
        merge_tau=merge_tau,
        adaptive_budget=adaptive_budget,
        k_min=k_min,
        k_max=k_max,
        zero_query=zero_query,
        tpn_version=tpn_version,
        llava_attn_implementation=llava_attn_implementation,
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


def evaluate_mmbench(model, data_root, device, batch_size, max_samples=None, log_interval=200):
    """MMBench English dev. Multiple-choice; score = exact-match on letter."""
    print("\n[MMBench] Loading English dev split...")
    mmbench_dir = Path(f"{data_root}/mmbench")
    # Try TSV first (OpenCompass format), then HF parquet (lmms-lab/MMBench)
    tsv_candidates = list(mmbench_dir.glob("mmbench_dev_en*.tsv"))
    parquet_candidates = list(mmbench_dir.glob("en/dev-*.parquet")) + \
                         list(mmbench_dir.glob("en/*.parquet"))
    if tsv_candidates:
        bench_path = str(tsv_candidates[0])
    elif parquet_candidates:
        bench_path = str(sorted(parquet_candidates)[0])
    else:
        raise FileNotFoundError(
            f"MMBench not found in {mmbench_dir}. "
            f"Run: python -c \"from huggingface_hub import snapshot_download; "
            f"snapshot_download('lmms-lab/MMBench', repo_type='dataset', "
            f"local_dir='{mmbench_dir}')\""
        )
    print(f"[MMBench] Using: {bench_path}")
    dataset = MMBenchDataset(bench_path, max_samples=max_samples)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, collate_fn=collate_fn, pin_memory=True,
    )
    print(f"[MMBench] {len(dataset)} samples")

    correct: list[int] = []
    per_category: dict[str, list[int]] = {}
    per_question: list[dict] = []

    letter_re = re.compile(r"\b([A-D])\b")
    for step, batch in enumerate(tqdm(loader, desc="MMBench dev")):
        with torch.no_grad():
            preds = model.generate(
                images_coarse=batch["images_coarse"].to(device),
                images_full=batch["images_full"].to(device),
                questions=batch["questions"],
                max_new_tokens=4,
            )
        if step == 0:
            for i in range(min(4, len(preds))):
                print(
                    f"  [diag] pred={repr(preds[i])} | gt={batch['answers'][i][0]}",
                    flush=True,
                )
        for i, (pred, gt_list, qid) in enumerate(zip(
            preds, batch["answers"], batch["question_id"],
        )):
            gt = gt_list[0].upper() if isinstance(gt_list, list) else str(gt_list).upper()
            m = letter_re.search(pred.upper())
            extracted = m.group(1) if m else (pred.strip().upper()[:1] if pred.strip() else "")
            ok = int(extracted == gt)
            correct.append(ok)
            cat = batch.get("category", [""] * len(preds))[i] or ""
            per_category.setdefault(cat, []).append(ok)
            per_question.append({
                "question_id": qid,
                "pred": pred,
                "extracted": extracted,
                "gt": gt,
                "correct": bool(ok),
                "category": cat,
            })
        if log_interval > 0 and (step + 1) % log_interval == 0:
            run = sum(correct) / len(correct) * 100
            print(f"[MMBench] step {step+1}/{len(loader)} | acc: {run:.2f}%", flush=True)

    overall = sum(correct) / len(correct) * 100 if correct else 0.0
    by_cat = {
        c: round(sum(v) / len(v) * 100, 2)
        for c, v in per_category.items() if v
    }
    print(f"[MMBench] FINAL accuracy: {overall:.2f}%  ({len(correct)} samples)")
    return {
        "accuracy": round(overall, 2),
        "by_category": by_cat,
        "num_samples": len(correct),
        "per_question": per_question,
    }


def evaluate_mme(model, data_root, device, batch_size, max_samples=None, log_interval=200):
    """MME yes/no benchmark. Score = sum_subtask(accuracy + accuracy_plus)."""
    print("\n[MME] Loading benchmark...")
    mme_root = f"{data_root}/mme"
    dataset = MMEDataset(mme_root, max_samples=max_samples)
    if len(dataset) == 0:
        raise FileNotFoundError(
            f"MME data not found in {mme_root}. "
            f"Download: python -c \"from huggingface_hub import snapshot_download; "
            f"snapshot_download('lmms-lab/MME', repo_type='dataset', local_dir='{mme_root}')\""
        )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, collate_fn=collate_fn, pin_memory=True,
    )
    print(f"[MME] {len(dataset)} samples")

    # Track per-subtask, per-image pair correctness
    per_image: dict[tuple, list[int]] = {}
    per_q: list[dict] = []
    for step, batch in enumerate(tqdm(loader, desc="MME")):
        with torch.no_grad():
            preds = model.generate(
                images_coarse=batch["images_coarse"].to(device),
                images_full=batch["images_full"].to(device),
                questions=batch["questions"],
                max_new_tokens=4,
            )
        for i, (pred, gt_list, img_id, sub) in enumerate(zip(
            preds, batch["answers"],
            batch.get("image_id", [""] * len(preds)),
            batch.get("subtask", [""] * len(preds)),
        )):
            gt = (gt_list[0] if isinstance(gt_list, list) else gt_list).lower()
            yn = extract_yes_no(pred)
            ok = int(yn == gt)
            per_image.setdefault((sub, img_id), []).append(ok)
            per_q.append({"subtask": sub, "image": img_id, "pred": pred, "gt": gt, "correct": bool(ok)})

    # Per-subtask accuracy + accuracy_plus
    sub_metrics: dict[str, dict] = {}
    for (sub, _), oks in per_image.items():
        s = sub_metrics.setdefault(sub, {"acc_total": 0, "acc_n": 0, "plus_total": 0, "plus_n": 0})
        s["acc_total"] += sum(oks)
        s["acc_n"] += len(oks)
        s["plus_total"] += int(all(oks) and len(oks) >= 2)
        s["plus_n"] += 1
    sub_scores: dict[str, float] = {}
    perception_total = 0.0
    cognition_total = 0.0
    PERCEPTION = {"existence", "count", "position", "color", "posters",
                  "celebrity", "scene", "landmark", "artwork", "OCR"}
    for sub, s in sub_metrics.items():
        acc_pct = s["acc_total"] / max(1, s["acc_n"]) * 100
        plus_pct = s["plus_total"] / max(1, s["plus_n"]) * 100
        score = acc_pct + plus_pct
        sub_scores[sub] = round(score, 2)
        if sub in PERCEPTION:
            perception_total += score
        else:
            cognition_total += score

    print(f"[MME] Perception: {perception_total:.1f} | Cognition: {cognition_total:.1f}")
    return {
        "perception_total": round(perception_total, 2),
        "cognition_total": round(cognition_total, 2),
        "by_subtask": sub_scores,
        "num_samples": len(per_q),
        "per_question": per_q,
    }


def evaluate_textvqa(model, data_root, device, batch_size, max_samples=None, log_interval=200):
    """TextVQA val. Uses VQA soft-accuracy against the 10 annotator answers."""
    print("\n[TextVQA] Loading val split...")
    dataset = TextVQADataset(
        annotations_path=f"{data_root}/textvqa/TextVQA_0.5.1_val.json",
        image_dir=f"{data_root}/textvqa/train_images",
        max_samples=max_samples,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, collate_fn=collate_fn, pin_memory=True,
    )
    print(f"[TextVQA] {len(dataset)} samples")
    scores: list[float] = []
    per_q: list[dict] = []
    for step, batch in enumerate(tqdm(loader, desc="TextVQA val")):
        with torch.no_grad():
            preds = model.generate(
                images_coarse=batch["images_coarse"].to(device),
                images_full=batch["images_full"].to(device),
                questions=batch["questions"],
                max_new_tokens=20,
            )
        for pred, gt_list, qid in zip(preds, batch["answers"], batch["question_id"]):
            s = vqa_score(pred, gt_list)
            scores.append(s)
            per_q.append({"question_id": qid, "pred": pred, "score": round(s, 4)})
        if log_interval > 0 and (step + 1) % log_interval == 0:
            run = sum(scores) / len(scores) * 100
            print(f"[TextVQA] step {step+1}/{len(loader)} | VQA-Score: {run:.2f}%", flush=True)
    avg = sum(scores) / max(1, len(scores)) * 100
    print(f"[TextVQA] FINAL VQA-Score: {avg:.2f}%  ({len(scores)} samples)")
    return {"vqa_score": round(avg, 2), "num_samples": len(scores), "per_question": per_q}


def _summarize_seeds(per_seed: list[dict]) -> dict:
    """Compute mean/stddev for headline metric of each benchmark across seeds."""
    import statistics

    HEADLINE = {
        "vqav2": "vqa_score",
        "gqa": "overall_accuracy",
        "pope": "macro_f1",
        "mmbench": "accuracy",
        "mme": "perception_total",
        "textvqa": "vqa_score",
    }
    summary: dict = {}
    for bench, key in HEADLINE.items():
        vals = [s[bench][key] for s in per_seed if bench in s and key in s[bench]]
        if not vals:
            continue
        if len(vals) == 1:
            summary[bench] = {"mean": vals[0], "stddev": 0.0, "n_seeds": 1, "values": vals}
        else:
            summary[bench] = {
                "mean": round(statistics.mean(vals), 3),
                "stddev": round(statistics.stdev(vals), 3),
                "n_seeds": len(vals),
                "values": vals,
            }
    return summary


@torch.no_grad()
def _clip_cls_attention_scores(model: TCViTokModel, images_full: torch.Tensor) -> torch.Tensor:
    """Return final-layer CLIP CLS-to-patch attention, shape [B, 576]."""
    clip_vision = model.full_encoder.clip.vision_model
    outputs = clip_vision(
        pixel_values=images_full,
        output_hidden_states=True,
        return_dict=True,
    )
    if outputs.hidden_states is None or len(outputs.hidden_states) < 2:
        raise RuntimeError("CLIP vision model did not return hidden states.")

    last_block = clip_vision.encoder.layers[-1]
    hidden = last_block.layer_norm1(outputs.hidden_states[-2])
    attn = last_block.self_attn
    q = attn.q_proj(hidden)
    k = attn.k_proj(hidden)
    batch, seq_len, dim = q.shape
    num_heads = int(attn.num_heads)
    head_dim = dim // num_heads
    scale = float(getattr(attn, "scale", head_dim ** -0.5))
    q = q.view(batch, seq_len, num_heads, head_dim).transpose(1, 2) * scale
    k = k.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
    weights = torch.matmul(q, k.transpose(-2, -1)).softmax(dim=-1)
    scores = weights[:, :, 0, 1:].mean(dim=1)
    if scores.shape[1] != 576:
        raise RuntimeError(f"Expected 576 CLIP patch scores, got {tuple(scores.shape)}")
    return scores


def _forced_rows_to_indices(
    forced_rows: torch.Tensor,
    k: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert forced selection rows into [B,k] indices plus [B,576] scores."""
    rows = forced_rows.to(device)
    if rows.ndim != 2:
        raise ValueError(f"Forced selection tensor must be 2D, got {tuple(rows.shape)}")

    if rows.shape[1] == 576:
        scores = rows.float()
        idx = scores.topk(k, dim=-1).indices
        return idx.long(), scores

    idx = rows.long()
    if idx.shape[1] < k:
        raise ValueError(f"Forced index tensor has only {idx.shape[1]} columns; need k={k}")
    idx = idx[:, :k]
    scores = torch.zeros(idx.shape[0], 576, device=device, dtype=torch.float32)
    scores.scatter_(1, idx.clamp(0, 575), 1.0)
    return idx, scores


def install_diagnostic_selection_io(model: TCViTokModel, args, device: str) -> dict | None:
    """Attach evaluation-time selection save/load diagnostics to a model."""
    paths = [
        args.force_selection_from,
        args.save_scores_to,
        args.save_selection_to,
        args.save_attentions_from_clip,
    ]
    if not any(paths):
        return None

    if args.baseline != "none" and (
        args.save_scores_to or args.save_selection_to or args.save_attentions_from_clip
    ):
        print(
            "[warning] diagnostic score/selection saving is designed for "
            "--baseline none; forced selection still works for downstream evaluation."
        )

    forced = None
    if args.force_selection_from:
        forced = torch.load(args.force_selection_from, map_location="cpu")
        if not torch.is_tensor(forced):
            raise ValueError("--force-selection-from must point to a tensor .pt file")

    state = {
        "offset": 0,
        "forced": forced,
        "scores": [],
        "selection": [],
        "clip_attn": [],
        "save_scores_to": args.save_scores_to,
        "save_selection_to": args.save_selection_to,
        "save_attentions_from_clip": args.save_attentions_from_clip,
    }
    original_select = model._select_tokens

    @torch.no_grad()
    def diagnostic_select(images_coarse, images_full, questions):
        batch = images_full.shape[0]
        start = state["offset"]
        end = start + batch
        state["offset"] = end

        if state["forced"] is not None:
            if end > state["forced"].shape[0]:
                raise RuntimeError(
                    f"Forced selection has {state['forced'].shape[0]} rows, "
                    f"but evaluation requested rows through {end}."
                )
            v_full = model.full_encoder(images_full)
            idx, s_N = _forced_rows_to_indices(
                state["forced"][start:end], model.k, v_full.device
            )
            if model.adaptive_budget:
                v_selected = model._gather_adaptive(v_full, idx, s_N)
                k_per_sample = (idx >= 0).sum(dim=1)
                output = (v_selected, k_per_sample)
            elif model.merge_beta > 0:
                output = (
                    informative_token_merge(
                        v_fine=v_full,
                        idx=idx,
                        s_N=s_N,
                        beta=model.merge_beta,
                        tau_merge=model.merge_tau,
                    ),
                    None,
                )
            else:
                output = (gather_tokens(v_full, idx), None)
        elif hasattr(model, "_compute_tpn_selection"):
            v_full, idx, s_N, _ = model._compute_tpn_selection(
                images_coarse=images_coarse,
                images_full=images_full,
                questions=questions,
            )
            if model.adaptive_budget:
                v_selected = model._gather_adaptive(v_full, idx, s_N)
                k_per_sample = (idx >= 0).sum(dim=1)
                output = (v_selected, k_per_sample)
            elif model.merge_beta > 0:
                output = (
                    informative_token_merge(
                        v_fine=v_full,
                        idx=idx,
                        s_N=s_N,
                        beta=model.merge_beta,
                        tau_merge=model.merge_tau,
                    ),
                    None,
                )
            else:
                output = (gather_tokens(v_full, idx), None)
        else:
            output = original_select(images_coarse, images_full, questions)
            idx = None
            s_N = None

        if args.save_scores_to and s_N is not None:
            state["scores"].append(s_N.detach().cpu())
        if args.save_selection_to and idx is not None:
            state["selection"].append(idx.detach().cpu())
        if args.save_attentions_from_clip:
            state["clip_attn"].append(
                _clip_cls_attention_scores(model, images_full).detach().cpu()
            )
        return output

    model._select_tokens = diagnostic_select
    return state


def save_diagnostic_selection_io(state: dict | None) -> None:
    """Persist tensors accumulated by install_diagnostic_selection_io."""
    if state is None:
        return
    outputs = [
        ("save_scores_to", "scores", "scores"),
        ("save_selection_to", "selection", "selection"),
        ("save_attentions_from_clip", "clip_attn", "CLIP attentions"),
    ]
    for path_key, data_key, label in outputs:
        path = state.get(path_key)
        chunks = state.get(data_key) or []
        if not path:
            continue
        if not chunks:
            raise RuntimeError(f"No {label} were captured; refusing to write {path}")
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        tensor = torch.cat(chunks, dim=0)
        torch.save(tensor, out)
        print(f"Saved {label} to {out} with shape={tuple(tensor.shape)}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate TC-ViTok on VQA benchmarks")
    parser.add_argument("--checkpoint", required=True, help="Path to best.ckpt")
    parser.add_argument(
        "--benchmarks", nargs="+", default=["vqav2", "gqa", "pope"],
        choices=["vqav2", "gqa", "pope", "mmbench", "mme", "textvqa"],
        help="Benchmarks to run",
    )
    parser.add_argument(
        "--baseline", default="none",
        choices=[
            "none", "random_drop", "uniform_grid", "cls_attention",
            "image_only", "full_576", "merge_disabled",
        ],
        help=(
            "Baseline variant. 'none' = full TC-ViTok. "
            "'full_576' = same LoRA + mm_proj as TC-ViTok but pass all 576 tokens "
            "(critical missing control: isolates compression vs LoRA effect). "
            "'merge_disabled' = Final TPN selection but beta=0 at eval "
            "(true single-variable merge ablation). "
            "'image_only' = TPN with zeroed CLIP query (Hypothesis H2 test). "
            "'random_drop' = uniform random token selection at k. "
            "'uniform_grid' = deterministic every-N/k-th token selection. "
            "'cls_attention' = CLIP last-layer CLS-to-patch attention top-k."
        ),
    )
    parser.add_argument("--k", type=int, default=144, help="Token budget")
    parser.add_argument(
        "--merge-beta", type=float, default=0.3,
        help="Token-merge strength. 0 disables merge. Default 0.3 matches Final.",
    )
    parser.add_argument(
        "--merge-tau", type=float, default=0.1,
        help="Token-merge softmax temperature.",
    )
    parser.add_argument(
        "--adaptive-budget", action="store_true",
        help="Use BudgetPredictor for per-sample variable k (Adaptive variant).",
    )
    parser.add_argument(
        "--no-auto-adaptive-budget",
        action="store_false",
        dest="auto_adaptive_budget",
        help=(
            "Do not auto-enable BudgetPredictor just because checkpoint keys are "
            "present. Use this for fixed-k sweeps on adaptive checkpoints."
        ),
    )
    parser.set_defaults(auto_adaptive_budget=True)
    parser.add_argument("--k-min", type=int, default=36)
    parser.add_argument("--k-max", type=int, default=288)
    parser.add_argument(
        "--zero-query", action="store_true",
        help="Zero out CLIP text embedding before TPN (v1 with zero_query=True).",
    )
    parser.add_argument(
        "--force-query-zero", action="store_true",
        help="Alias for --zero-query used by diagnostic scripts.",
    )
    parser.add_argument(
        "--force-selection-from", default=None,
        help=(
            "Path to .pt tensor with forced selections. Supports [N,k] long "
            "indices or [N,576] float/bool masks/scores. Batches consume rows "
            "in evaluation order."
        ),
    )
    parser.add_argument(
        "--save-scores-to", default=None,
        help="Save per-image TPN score tensors [N,576] to this .pt file.",
    )
    parser.add_argument(
        "--save-selection-to", default=None,
        help="Save per-image selected top-k indices [N,k] to this .pt file.",
    )
    parser.add_argument(
        "--save-attentions-from-clip", default=None,
        help=(
            "Save per-image CLIP final-layer CLS-to-patch attention [N,576] "
            "to this .pt file."
        ),
    )
    parser.add_argument(
        "--tpn-version", default="auto", choices=["auto", "v1", "v3"],
        help="TPN architecture. 'auto' detects from checkpoint keys. "
             "'v1' = cross-attention TPN (DINOv2+CLIP-text). "
             "'v3' = image-only TokenScorerV3 (per-token MLP + self-attn on CLIP).",
    )
    parser.add_argument(
        "--n-eval-seeds", type=int, default=1,
        help=(
            "Number of generation seeds to average over for eval-time variance. "
            "Each seed re-runs the full benchmark; results are averaged and stddev "
            "is reported. Default 1 = no variance estimate. Recommended: 3-5."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed. Controls random_drop token sampling.",
    )
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
    if args.force_query_zero:
        args.zero_query = True

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
    print(f"  Zero query : {args.zero_query}")
    print(f"  TPN version: {args.tpn_version}")
    print(f"  Seed       : {args.seed}")
    print(f"  Device     : {device}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  Data root  : {data_root}")
    print(f"  Output     : {args.output}")
    print("=" * 60)

    # Load model once — reuse across all benchmarks AND across eval seeds
    model = load_model(
        checkpoint_path=args.checkpoint,
        device=device,
        k=args.k,
        baseline=args.baseline,
        llava_model_name=args.llava_model,
        merge_beta=args.merge_beta,
        merge_tau=args.merge_tau,
        adaptive_budget=args.adaptive_budget,
        k_min=args.k_min,
        k_max=args.k_max,
        zero_query=args.zero_query,
        tpn_version=args.tpn_version,
        auto_adaptive_budget=args.auto_adaptive_budget,
    )
    if args.zero_query and model.tpn_version == "v3":
        print(
            "[warning] --zero-query is a no-op for tpn-version=v3 because "
            "TokenScorerV3 has no CLIP text-query input."
        )

    diag = install_diagnostic_selection_io(model, args, device)

    results = {
        "checkpoint": args.checkpoint,
        "baseline": args.baseline,
        "k": args.k,
        "zero_query": bool(args.zero_query),
        "tpn_version": model.tpn_version,
        "max_samples": args.max_samples,
        "seed": int(args.seed),
        "merge_beta": float(model.merge_beta),
        "merge_tau": float(model.merge_tau),
        "adaptive_budget": bool(model.adaptive_budget),
        "n_eval_seeds": args.n_eval_seeds,
        "device": device,
        "benchmarks": {},
    }

    t_start = time.time()

    # Multi-seed eval loop. Determinism via torch.manual_seed only affects
    # any sampling in generate(); LLaVA greedy decoding is deterministic so
    # variance comes from any tie-breaking + non-deterministic GPU kernels.
    # We collect per-seed metrics and report mean ± stddev.
    seeds = [args.seed + i for i in range(args.n_eval_seeds)]
    per_seed_metrics: list[dict] = []
    for seed in seeds:
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        seed_results: dict = {}

        if "vqav2" in args.benchmarks:
            seed_results["vqav2"] = evaluate_vqav2(
                model, data_root, device, args.batch_size, args.max_samples,
                log_interval=args.log_interval,
            )
        if "gqa" in args.benchmarks:
            seed_results["gqa"] = evaluate_gqa(
                model, data_root, device, args.batch_size,
                args.max_samples, spatial_eval=args.spatial_eval,
                log_interval=args.log_interval,
            )
        if "pope" in args.benchmarks:
            seed_results["pope"] = evaluate_pope(
                model, data_root, device, args.batch_size, args.max_samples,
                log_interval=args.log_interval,
            )
        if "mmbench" in args.benchmarks:
            seed_results["mmbench"] = evaluate_mmbench(
                model, data_root, device, args.batch_size, args.max_samples,
                log_interval=args.log_interval,
            )
        if "mme" in args.benchmarks:
            seed_results["mme"] = evaluate_mme(
                model, data_root, device, args.batch_size, args.max_samples,
                log_interval=args.log_interval,
            )
        if "textvqa" in args.benchmarks:
            seed_results["textvqa"] = evaluate_textvqa(
                model, data_root, device, args.batch_size, args.max_samples,
                log_interval=args.log_interval,
            )
        per_seed_metrics.append(seed_results)

    # First-seed full results (with per_question lists) used for downstream analysis.
    results["benchmarks"] = per_seed_metrics[0]
    if args.n_eval_seeds > 1:
        results["per_seed"] = _summarize_seeds(per_seed_metrics)

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

    if "mmbench" in results["benchmarks"]:
        r = results["benchmarks"]["mmbench"]
        print(f"  MMBench dev: {r['accuracy']:.2f}%  ({r['num_samples']} samples)")

    if "mme" in results["benchmarks"]:
        r = results["benchmarks"]["mme"]
        print(f"  MME perception: {r['perception_total']:.1f}")
        print(f"  MME cognition : {r['cognition_total']:.1f}")

    if "textvqa" in results["benchmarks"]:
        r = results["benchmarks"]["textvqa"]
        print(f"  TextVQA val: {r['vqa_score']:.2f}%")

    if "per_seed" in results:
        print("\n  Multi-seed summary (mean ± stddev over n seeds):")
        for bench, stats in results["per_seed"].items():
            print(f"    {bench:10s}: {stats['mean']:.2f} ± {stats['stddev']:.2f}  (n={stats['n_seeds']})")

    save_diagnostic_selection_io(diag)
    print("=" * 60)


if __name__ == "__main__":
    main()
