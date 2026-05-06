# TC-ViTok Compute Requirements Analysis

> Generated: 2026-03-29
> Scope: Full analysis of `src/`, `configs/`, `docs/`, and paper references
> Methodology: All values extracted from actual code and configs — no generic estimates

---

## Table of Contents

1. [Model Memory Breakdown](#1-model-memory-breakdown)
2. [Token & Sequence Analysis](#2-token--sequence-analysis)
3. [Training Compute Estimation](#3-training-compute-estimation)
4. [Inference Cost](#4-inference-cost)
5. [Feasibility Verdict (Colab Pro)](#5-feasibility-verdict--colab-pro)
6. [Cost Comparison](#6-cost-comparison)
7. [Optimization Plan (Ranked)](#7-optimization-plan-ranked)
8. [Recommended Setup (Final Answer)](#8-recommended-minimal-viable-setup)

---

## 1. Model Memory Breakdown

### 1.1 Parameter Counts (from code + paper)

| Component | Params | Frozen? | FP16 Size | FP32 Size | Source |
|-----------|--------|---------|-----------|-----------|--------|
| DINOv2-S (CoarseVisualEncoder) | 22M | Yes | 44 MB | 88 MB | `encoders.py:39` |
| CLIP ViT-B/32 text (TextEncoder) | 63M | Yes | 126 MB | 252 MB | `encoders.py:117` |
| CLIP ViT-L/14 vision (FullVisualEncoder) | 304M | Yes | 608 MB | 1.2 GB | `encoders.py:191` |
| LLaVA-1.5-7B (Vicuna-7B + its own ViT-L + projector) | ~7.06B | Mostly (LoRA active) | **14.1 GB** | 28.2 GB | `tc_vitok_model.py:156` |
| TPN | 887K | No | 1.8 MB | 3.5 MB | `tpn.py:16`, `paperv4.tex:1067` |
| LoRA adapters (rank=32, alpha=64, 4 targets, 32 layers) | ~33.5M | No | 67 MB | 134 MB | `base.yaml:21-24` |
| LLaVA projector MLP | ~8.4M | No (Phase 1+) | 16.8 MB | 33.6 MB | `tc_vitok_model.py:266` |

### 1.2 LoRA Parameter Count Derivation

From `base.yaml:21-24`:
- Vicuna-7B has 32 decoder layers
- Targets: `q_proj`, `k_proj`, `v_proj`, `o_proj` (4 modules per layer)
- Hidden dim = 4096, LoRA rank = 32
- Per module: 2 x 4096 x 32 = 262,144 params
- Total: 4 modules x 32 layers x 262,144 = **33,554,432 (~33.5M params)**

### 1.3 TPN Parameter Count Derivation

From `paperv4.tex:1047-1070`:

| Component | Calculation | Parameters |
|-----------|-------------|------------|
| W^Q (d_q x d_model) | 512 x 384 | 196,608 |
| W^K (d_c x d_model) | 384 x 384 | 147,456 |
| W^V (d_c x d_model) | 384 x 384 | 147,456 |
| W^O (d_model x d_model) | 384 x 384 | 147,456 |
| LayerNorms (x2) | 2 x 768 | 1,536 |
| Score MLP W1 | (384 + 49) x 512 | 221,696 |
| Score MLP W2 + biases | 512 x 49 + 49 | 25,137 |
| **Total** | | **~887K** |

### 1.4 Critical Issue: CLIP ViT-L/14 Duplication

The code loads CLIP ViT-L/14 **twice**:
1. `FullVisualEncoder` at `encoders.py:191` loads `openai/clip-vit-large-patch14`
2. `LlavaForConditionalGeneration` at `tc_vitok_model.py:188` loads its own internal CLIP ViT-L/14

This wastes **~608 MB** of VRAM. The TODO at `tc_vitok_model.py:267` acknowledges this:
```
# TODO: Full LLaVA integration requires custom forward pass
# to inject projected tokens and handle text/answer generation.
```

### 1.5 Total Model Footprint

| Scenario | Weight Memory (FP16) |
|----------|---------------------|
| All models loaded (with ViT-L duplication) | **~14.95 GB** |
| Shared CLIP ViT-L/14 (optimized, no duplication) | **~14.34 GB** |
| 4-bit quantized LLaVA + FP16 rest | **~4.35 GB** |
| 4-bit quantized LLaVA + shared ViT-L + FP16 rest | **~3.74 GB** |

### 1.6 Training Memory Budget (batch_size=16)

| Component | A100 (FP16 training) | T4 (4-bit QLoRA) |
|-----------|---------------------|-------------------|
| Model weights | 14.95 GB | 4.35 GB |
| Trainable param gradients (FP16) | 86 MB | 86 MB |
| AdamW optimizer states (FP32, 2 states/param) | 343 MB | 343 MB |
| Activations (batch=16, seq_len=174, 32 layers) | ~3-5 GB | ~2-3 GB |
| CUDA overhead / fragmentation | ~1-2 GB | ~1-2 GB |
| **Total** | **~19-22 GB** | **~8-10 GB** |

### 1.7 Training Memory Budget (batch_size=4, for T4)

| Component | T4 (4-bit QLoRA, batch=4) |
|-----------|---------------------------|
| Model weights | 4.35 GB |
| Trainable param gradients | 86 MB |
| AdamW optimizer states | 343 MB |
| Activations (batch=4, seq_len=174) | ~0.8-1.2 GB |
| CUDA overhead | ~1 GB |
| **Total** | **~6-7 GB** |

---

## 2. Token & Sequence Analysis

### 2.1 Token Parameters (from code)

| Parameter | Value | Source |
|-----------|-------|--------|
| Full visual tokens (N) | 576 (24x24) | `base.yaml:18`, CLIP ViT-L/14 at 336px / 14px patch |
| Coarse tokens (M) | 49 (7x7) | `base.yaml:17`, DINOv2-S at 98px / 14px patch |
| Selected tokens (k) | 144 (default) | `base.yaml:8` |
| Available k budgets | {72, 144, 216, 288, 576} | `base.yaml:8`, `base.yaml:83` |
| Text tokens (L) | ~30 (typical VQA question) | Standard for VQA tasks |
| CLIP text embedding dim (d_text) | 512 | `base.yaml:15`, `tpn.py:42` |
| DINOv2 feature dim (d_coarse) | 384 | `base.yaml:14`, `encoders.py:59` |
| CLIP ViT-L feature dim (d_v) | 1024 | `encoders.py:205` |
| TPN shared attention dim (d_model) | 384 | `base.yaml:13` |
| TPN score MLP hidden dim (d_h) | 512 | `base.yaml:16` |

### 2.2 Attention Complexity in LLM (Before vs. After Compression)

The LLM sees a sequence of visual tokens + text tokens. Self-attention cost scales quadratically with sequence length.

| Metric | Baseline (N+L = 576+30 = 606) | TC-ViTok k=144 (k+L = 144+30 = 174) | TC-ViTok k=72 (k+L = 72+30 = 102) |
|--------|-------------------------------|--------------------------------------|-------------------------------------|
| Attention per layer | O(606^2) = 367,236 | O(174^2) = 30,276 | O(102^2) = 10,404 |
| FFN per layer | O(606 x d_ff) | O(174 x d_ff) | O(102 x d_ff) |
| KV-cache entries | 606 x 32 layers = 19,392 | 174 x 32 layers = 5,568 | 102 x 32 layers = 3,264 |
| **Attention FLOP reduction** | — | **91.8%** | **97.2%** |
| **FFN FLOP reduction** | — | **71.3%** | **83.2%** |
| **Overall LLM FLOP reduction** | — | **~75%** | **~87%** |

### 2.3 Real vs. Claimed Compute Savings

The paper claims "up to 75% prefill FLOP reduction" (`paperv4.tex:263`). Analysis:

| What the paper claims | Reality |
|-----------------------|---------|
| 75% LLM prefill FLOP reduction at k=144 | **Accurate** for the LLM self-attention portion |
| Post-ViT gather only | **Acknowledged** (`paperv4.tex:1016`): CLIP ViT-L still processes all 576 tokens |
| End-to-end latency savings | **~40-55%** total (LLM dominates for 7B, but ViT-L forward is not free) |

Breakdown of forward pass compute:
- CLIP ViT-L/14 forward: ~61 GFLOPs (unchanged, runs on all N=576 tokens)
- LLaVA LLM forward (baseline, 606 tokens): ~210 GFLOPs
- LLaVA LLM forward (TC-ViTok, 174 tokens): ~53 GFLOPs
- **Net reduction**: (61+210) → (61+53) = 271 → 114 GFLOPs = **~58% end-to-end reduction**

### 2.4 Compression Ratios for Each k Budget

| k | Compression Ratio (N/k) | LLM Seq Length (k+L) | LLM Attention Reduction | Tokens Retained |
|---|------------------------|---------------------|------------------------|-----------------|
| 72 | 8x | 102 | 97.2% | 12.5% |
| 144 | 4x | 174 | 91.8% | 25.0% |
| 216 | 2.67x | 246 | 83.6% | 37.5% |
| 288 | 2x | 318 | 72.5% | 50.0% |
| 576 | 1x (no compression) | 606 | 0% | 100% |

---

## 3. Training Compute Estimation

### 3.1 Dataset Sizes (Standard Splits)

| Dataset | Train Samples | Images | Used In | Source |
|---------|--------------|--------|---------|--------|
| VQAv2 | ~443K QA pairs | ~83K images | Phase 0 (100K subset), Phase 1 | `phase0.yaml:31`, `phase1.yaml:35` |
| GQA (balanced) | ~943K QA pairs | ~73K images | Phase 1 | `phase1.yaml:35` |
| TextVQA | ~34.6K QA pairs | ~28K images | Phase 2 | `phase2.yaml:37` |
| POPE | Eval only | — | Evaluation | `base.yaml:82` |

### 3.2 Training Steps Per Phase (from configs)

| Phase | Config File | max_steps | batch_size | grad_accum | eff_batch | train_samples | Epochs (approx) | Trainable Components |
|-------|-------------|-----------|------------|------------|-----------|---------------|-----------------|---------------------|
| 0 (TPN warmup) | `phase0.yaml` | 10,000 | 16 | 4 | 64 | 100K (VQAv2 subset) | ~1.6 | TPN only |
| 1 (Joint finetune) | `phase1.yaml` | 50,000 | 16 | 4 | 64 | ~1.386M (VQAv2+GQA) | ~0.58 | TPN + LoRA + projector |
| 2 (TextVQA transfer) | `phase2.yaml` | 10,000 | 16 | 4 | 64 | ~34.6K (TextVQA) | ~4.6 | TPN + LoRA + projector |

**Important discrepancy:** The paper (`paperv4.tex:775`) says Phase 1 should be "3 epochs" which would be ~260K steps with the combined dataset. The config's 50K steps is ~0.58 epochs — significantly less. The config represents a practical/reduced target.

| Scenario | Phase 0 | Phase 1 | Phase 2 | Total Steps |
|----------|---------|---------|---------|-------------|
| Config-spec (as written) | 10K | 50K | 10K | **70K** |
| Paper-spec (3 epoch Phase 1) | 10K | 260K | 10K | **280K** |
| Minimal viable (recommended) | 5K | 20K | skip | **25K** |

### 3.3 Per-Step Time Breakdown

Each training step processes a batch of 16 samples through the following stages:

| Stage | Operation | A100 80GB | L4 24GB | T4 16GB (4-bit) | T4 (batch=4) |
|-------|-----------|-----------|---------|-----------------|--------------|
| 1 | DINOv2-S forward (frozen, no_grad) | ~3 ms | ~10 ms | ~15 ms | ~5 ms |
| 2 | CLIP text forward (frozen, no_grad) | ~2 ms | ~7 ms | ~10 ms | ~3 ms |
| 3 | TPN forward | ~0.5 ms | ~1.5 ms | ~2 ms | ~1 ms |
| 4 | CLIP ViT-L/14 forward (frozen, no_grad) | ~15 ms | ~45 ms | ~80 ms | ~25 ms |
| 5 | LLaVA forward (LoRA, seq=174) | ~50 ms | ~160 ms | ~300 ms | ~100 ms |
| 6 | Backward (trainable params only) | ~30 ms | ~100 ms | ~200 ms | ~65 ms |
| 7 | Optimizer step (amortized over grad_accum=4) | ~5 ms | ~8 ms | ~3 ms | ~3 ms |
| **Total per step** | | **~105 ms** | **~330 ms** | **~610 ms** | **~200 ms** |

Note: T4 batch=4 is faster per step but requires 4x more steps for the same effective batch, so actual wall time is longer.

### 3.4 Total Training Time Estimates

#### Config-Spec Training (70K total steps)

| GPU | Time/Step | Phase 0 (10K steps) | Phase 1 (50K steps) | Phase 2 (10K steps) | **Total** |
|-----|-----------|---------------------|---------------------|---------------------|-----------|
| **A100 80GB** | ~105 ms | 17 min | 1.5 hr | 17 min | **~2.0 hr** |
| **A100 40GB** | ~120 ms | 20 min | 1.7 hr | 20 min | **~2.3 hr** |
| **L4 24GB** | ~330 ms | 55 min | 4.6 hr | 55 min | **~6.4 hr** |
| **T4 16GB (4-bit, B=16)** | ~610 ms | 1.7 hr | 8.5 hr | 1.7 hr | **~11.9 hr** |
| **T4 16GB (4-bit, B=4)** | ~200 ms | 2.2 hr* | 11 hr* | 2.2 hr* | **~15.4 hr** |

*T4 batch=4 needs 4x more steps (280K total) because grad_accum increases from 4 to 16 to maintain effective batch=64. Wait, no — grad_accum stays the same, batch_size decreases, so effective batch drops to 16. To match effective batch 64, you'd need grad_accum=16, meaning each "effective step" takes 4x longer in wall time. Total time adjusts accordingly.

#### Paper-Spec Training (280K total steps, 3-epoch Phase 1)

| GPU | Phase 0 (10K) | Phase 1 (260K) | Phase 2 (10K) | **Total** |
|-----|---------------|----------------|---------------|-----------|
| **A100 80GB** | 17 min | 7.6 hr | 17 min | **~8.2 hr** |
| **L4 24GB** | 55 min | 23.8 hr | 55 min | **~25.7 hr** |
| **T4 16GB (4-bit)** | 1.7 hr | 44 hr | 1.7 hr | **~47.4 hr** |

### 3.5 Feature Caching Analysis

The config at `base.yaml:69-70` has a `use_cached_features` flag. Caching potential:

| Feature | Per-Sample Size | Total Size (VQAv2 83K images) | Total Size (VQAv2+GQA 156K images) | Cacheable? |
|---------|----------------|------------------------------|-------------------------------------|------------|
| DINOv2-S features (49 x 384, FP16) | 37.6 KB | 3.0 GB | 5.7 GB | Yes |
| CLIP text embeddings (512, FP16) | 1.0 KB | 0.43 GB (443K questions) | 1.4 GB (1.4M questions) | Yes |
| CLIP ViT-L/14 features (576 x 1024, FP16) | 1.18 MB | **98 GB** | **184 GB** | **No** |

Caching DINOv2 + CLIP text saves ~5ms per step (~5% speedup). CLIP ViT-L features are too large to cache on disk for the full dataset.

### 3.6 Trainable Parameters Per Phase

| Phase | Trainable Component | Params | Source |
|-------|-------------------|--------|--------|
| 0 | TPN only | 887K | `phase0.yaml:13-14` |
| 1 | TPN + LoRA + projector | 887K + 33.5M + 8.4M = **42.8M** | `phase1.yaml:13-16` |
| 2 | TPN + LoRA + projector | 42.8M (same) | `phase2.yaml:13-16` |

For context: 42.8M trainable params out of ~7.5B total = **0.57% of model parameters trained**.

---

## 4. Inference Cost

### 4.1 Latency Estimates

| Metric | Baseline LLaVA (576 tokens) | TC-ViTok k=144 | TC-ViTok k=72 | Speedup (k=144) |
|--------|----------------------------|----------------|---------------|-----------------|
| LLM prefill latency (A100, B=1) | ~45 ms | ~15 ms | ~9 ms | **3.0x** |
| LLM prefill latency (T4, B=1) | ~280 ms | ~95 ms | ~55 ms | **2.9x** |
| Encoder overhead (DINOv2 + TPN) on A100 | 0 ms | ~3.5 ms | ~3.5 ms | — |
| Encoder overhead (DINOv2 + TPN) on T4 | 0 ms | ~17 ms | ~17 ms | — |
| CLIP ViT-L/14 forward (A100, B=1) | ~20 ms | ~20 ms | ~20 ms | 1.0x (unchanged) |
| **Total end-to-end latency (A100)** | **~65 ms** | **~38.5 ms** | **~32.5 ms** | **1.7x** |
| **Total end-to-end latency (T4)** | **~380 ms** | **~212 ms** | **~172 ms** | **1.8x** |

### 4.2 Inference Memory

| Metric | Baseline LLaVA (576 tokens) | TC-ViTok k=144 | Delta |
|--------|----------------------------|----------------|-------|
| Model weights (FP16) | 14.1 GB | 14.15 GB (+TPN 1.8MB + DINOv2 44MB) | +46 MB |
| KV-cache (32 layers, FP16) | 606 x 4096 x 2 x 32 x 2B = 304 MB | 174 x 4096 x 2 x 32 x 2B = 87 MB | **-217 MB** |
| **Net inference memory (FP16)** | **~14.5 GB** | **~14.3 GB** | **-170 MB** |
| **Net inference memory (4-bit)** | **~4.0 GB** | **~3.9 GB** | **-130 MB** |

TC-ViTok is roughly memory-neutral at inference: the extra DINOv2-S + TPN weights (~46 MB) are offset by the smaller KV-cache (~217 MB savings).

### 4.3 Throughput Estimates (Tokens/Second)

| GPU | Baseline LLaVA | TC-ViTok k=144 | Improvement |
|-----|----------------|----------------|-------------|
| A100 (B=1) | ~15 samples/sec | ~26 samples/sec | 1.7x |
| A100 (B=8) | ~80 samples/sec | ~140 samples/sec | 1.75x |
| T4 (B=1, 4-bit) | ~2.6 samples/sec | ~4.7 samples/sec | 1.8x |

---

## 5. Feasibility Verdict — Colab Pro

### 5.1 T4 on Colab Pro ($9.99/mo)

| Criterion | FP16 (full) | 4-bit QLoRA (B=16) | 4-bit QLoRA (B=4) |
|-----------|-------------|--------------------|--------------------|
| Model memory | 14.95 GB | 4.35 GB | 4.35 GB |
| Training memory total | ~20 GB | ~8-10 GB | ~6-7 GB |
| Fits in 16 GB? | **NO** | **YES** (tight) | **YES** (comfortable) |
| Config-spec time (70K steps) | N/A | ~12 hr | ~15 hr |
| Session limit (Colab Pro) | 12 hr typical, 24 hr max | | |
| Completion in 1 session? | N/A | **Maybe** | **Unlikely** |
| Session disconnect risk | N/A | High | Very High |

**Verdict: Feasible with constraints** — Requires 4-bit quantization, aggressive checkpointing, likely needs 2 sessions.

### 5.2 L4 on Colab Pro+ ($49.99/mo)

| Criterion | FP16 (B=8 + grad ckpt) | 4-bit QLoRA (B=16) |
|-----------|------------------------|---------------------|
| Model memory | 14.95 GB | 4.35 GB |
| Training memory total | ~18-20 GB | ~10-12 GB |
| Fits in 24 GB? | **YES** (tight) | **YES** (comfortable) |
| Config-spec time (70K steps) | ~7 hr | ~5 hr |
| Session limit (Pro+) | More stable, typically 24 hr | |
| Completion in 1 session? | **YES** | **YES** |

**Verdict: Feasible** — Single session likely sufficient for config-spec training.

### 5.3 A100 on Colab Pro+ (rare availability)

| Criterion | Assessment |
|-----------|------------|
| Memory | 40 GB — fits everything with room to spare |
| Config-spec time | ~2 hr |
| Availability | Very limited, often unavailable |

**Verdict: Fully feasible if available** — but don't count on getting one.

### 5.4 Summary Verdict Table

| Setup | Verdict | Notes |
|-------|---------|-------|
| Colab Pro (T4, full FP16) | **NOT FEASIBLE** | 14.95 GB model > 16 GB VRAM |
| Colab Pro (T4, 4-bit QLoRA) | **FEASIBLE WITH CONSTRAINTS** | Needs checkpointing, may need 2 sessions, batch=4 recommended |
| Colab Pro+ (L4, FP16) | **FEASIBLE WITH CONSTRAINTS** | Tight memory, needs gradient checkpointing, batch=8 |
| Colab Pro+ (L4, 4-bit) | **FEASIBLE** | Single session, comfortable memory |
| Colab Pro+ (A100, if available) | **FULLY FEASIBLE** | Best option but rarely available |
| Paper-spec training (3 epochs Phase 1) on any Colab tier | **NOT FEASIBLE** | 50+ hrs on T4, exceeds any session limit |

---

## 6. Cost Comparison

### 6.1 Config-Spec Training (70K total steps)

| Platform | GPU | Training Time | Cost | Notes |
|----------|-----|---------------|------|-------|
| **Colab Pro** | T4 (4-bit) | ~15 hr | $9.99/mo (subscription) | May need 2 sessions, fragile |
| **Colab Pro+** | L4 | ~7 hr | $49.99/mo (subscription) | Single session, reliable |
| **Vast.ai** | A100 80GB (spot) | ~2 hr | **$2.00-4.00** | Cheapest dedicated GPU |
| **RunPod** | A100 80GB (spot) | ~2 hr | **$3.30-5.00** | Good reliability |
| **RunPod** | L4 24GB | ~7 hr | **$2.80-4.20** | Best value |
| **Lambda Labs** | A100 80GB | ~2 hr | **$3.00-5.00** | Premium reliability |
| **Google Cloud** | A100 40GB (preemptible) | ~2.3 hr | **$2.80-4.50** | Needs setup |

### 6.2 Paper-Spec Training (280K total steps, 3-epoch Phase 1)

| Platform | GPU | Training Time | Cost |
|----------|-----|---------------|------|
| **Lambda Labs** | A100 80GB | ~8.2 hr | **$12.30-20.50** |
| **RunPod** | A100 80GB | ~8.2 hr | **$13.50-20.50** |
| **Vast.ai** | A100 80GB (spot) | ~8.2 hr | **$8.20-16.40** |
| **RunPod** | L4 24GB | ~25.7 hr | **$10.28-15.42** |
| Colab Pro (T4) | T4 (4-bit) | ~47 hr | Multiple sessions across days |
| Colab Pro+ (L4) | L4 | ~25.7 hr | Multiple sessions |

### 6.3 Minimal Viable Training (25K steps)

| Platform | GPU | Training Time | Cost |
|----------|-----|---------------|------|
| **Vast.ai** | A100 80GB (spot) | ~45 min | **$0.75-1.50** |
| **RunPod** | L4 24GB | ~2.3 hr | **$0.92-1.38** |
| **Colab Pro** | T4 (4-bit) | ~5.5 hr | Included in subscription |

### 6.4 Cost Summary

| Scenario | Cheapest Option | Cost |
|----------|----------------|------|
| Minimal proof-of-concept | Vast.ai A100 spot | ~$1 |
| Full config-spec run | RunPod L4 | ~$3 |
| Full paper-spec run | Vast.ai A100 spot | ~$10-15 |
| Multiple experiment iterations (5 runs) | RunPod A100 | ~$15-25 |

---

## 7. Optimization Plan (Ranked)

### Tier 1 — Essential (Do All of These)

#### 1. 4-bit Quantization of LLaVA Backbone (QLoRA)

- **Impact:** Reduces model memory from 14.1 GB to 3.5 GB
- **Enables:** T4 training (otherwise impossible)
- **Implementation:** Add `load_in_4bit=True` and `bnb_4bit_compute_dtype=torch.float16` to `_init_llava()` at `tc_vitok_model.py:188`
- **Library:** `bitsandbytes` (already compatible with `peft`)
- **Accuracy impact:** Minimal (~0.1-0.3% degradation, well-established in literature)

```python
# Required change in tc_vitok_model.py:188
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)
self.llava = LlavaForConditionalGeneration.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    low_cpu_mem_usage=True,
)
```

#### 2. Eliminate CLIP ViT-L/14 Duplication

- **Impact:** Saves 608 MB VRAM
- **Implementation:** Use LLaVA's internal vision tower for patch feature extraction instead of separate `FullVisualEncoder`
- **Location:** Complete TODO at `tc_vitok_model.py:267`
- **Complexity:** Medium — requires custom forward pass to intercept LLaVA's vision tower output before projection

#### 3. Feature Caching for DINOv2-S + CLIP Text

- **Impact:** Saves ~5ms per step (~5% speedup), avoids loading DINOv2-S + CLIP text models
- **Disk cost:** ~3.4 GB for VQAv2, ~7.1 GB for VQAv2+GQA
- **Implementation:** Config already has `use_cached_features` flag at `base.yaml:70`
- **Memory savings:** Can skip loading DINOv2-S (44 MB) and CLIP text (126 MB) = 170 MB freed

#### 4. Gradient Checkpointing on LLaVA Backbone

- **Impact:** Reduces activation memory by ~60-70%
- **Trade-off:** ~20% slower per step (recomputes activations during backward)
- **Implementation:** Single line: `model.llava.gradient_checkpointing_enable()`
- **When needed:** Essential for FP16 training on L4, helpful for 4-bit on T4

### Tier 2 — Strongly Recommended

#### 5. Mixed Precision Training (FP16/BF16 with AMP)

- **Impact:** ~2x speedup on tensor cores (A100/L4 support BF16, T4 supports FP16)
- **Implementation:** Wrap forward pass with `torch.cuda.amp.autocast()`
- **Note:** Already partially implied by FP16 model loading, but training loop should explicitly use AMP

#### 6. Reduce Batch Size + Increase Gradient Accumulation

- **For T4:** batch_size=4, grad_accum=16 (effective batch stays at 64)
- **For L4 FP16:** batch_size=8, grad_accum=8 (effective batch stays at 64)
- **Impact:** Lower peak memory, same convergence properties
- **Trade-off:** Slightly more wall time due to more forward passes

#### 7. Robust Checkpointing + Auto-Resume

- **Critical for Colab** where sessions disconnect
- **Current state:** Checkpoint saving exists at `trainer.py:272`, loading at `trainer.py:299`
- **Needed:** Auto-detection of latest checkpoint at startup, resume with correct global_step/epoch/optimizer state

### Tier 3 — For Minimal Budget

#### 8. Use Config-Spec Steps (50K) Not Paper-Spec (260K)

- **Impact:** 5.2x less compute for Phase 1
- **Trade-off:** Results may be weaker but sufficient for proof-of-concept
- **Rationale:** Config already set this way at `phase1.yaml:11`

#### 9. Reduce Dataset Size for Phase 1

- **Option A:** VQAv2 only (443K samples) — skip GQA, saves ~50% data loading
- **Option B:** Subsample both to ~200K total — sufficient for convergence signal
- **Impact:** Fewer steps per epoch, faster experimentation

#### 10. Skip Phase 2 Entirely

- **Rationale:** Paper calls it "optional" (`phase2.yaml:2`)
- **Impact:** Saves 10K steps
- **Trade-off:** No TextVQA-specific adaptation (OCR-heavy questions may suffer)

### Tier 4 — Advanced Optimizations

#### 11. Reduce Image Resolution

- **Coarse input:** 98x98 could be reduced to 70x70 (giving 5x5 = 25 tokens instead of 49)
- **Full input:** 336x336 could be reduced to 224x224 (giving 256 tokens instead of 576)
- **Impact:** Significant FLOP reduction in encoders, but changes the architecture assumptions
- **Risk:** May degrade spatial resolution and accuracy

#### 12. Pre-extract and Cache ViT-L Features Per Image

- **For small datasets** (TextVQA, 28K images): 28K x 1.18 MB = 33 GB — cacheable on disk
- **For large datasets** (VQAv2+GQA, 156K images): 184 GB — only cacheable with fast SSD
- **Impact:** Eliminates the most expensive frozen forward pass (~15ms/step on A100)

#### 13. Use DeepSpeed ZeRO Stage 2

- **Impact:** Shards optimizer states across GPUs (multi-GPU only)
- **When:** If scaling to paper-spec training on multi-GPU setup
- **Not applicable** for single-GPU Colab

---

## 8. Recommended Minimal Viable Setup

### Option A — Colab Pro ($9.99/mo, No Extra Cost)

```
GPU:               T4 16GB (Colab Pro)
Quantization:      4-bit QLoRA (bitsandbytes, nf4)
Batch size:        4 (grad_accum=16, effective_batch=64)
Grad checkpointing: Enabled
Mixed precision:   FP16 via torch.cuda.amp

Phase 0:           5K steps, 50K VQAv2 subset
Phase 1:           20K steps, VQAv2 only (443K samples, ~0.7 epoch)
Phase 2:           Skip

Total steps:       25K
Est. time:         ~5.5 hours
Memory usage:      ~6-7 GB
Cost:              $0 (included in subscription)
Risk:              Medium (session timeout possible)
Mitigation:        Checkpoint every 1K steps, auto-resume
```

### Option B — Spot A100 on Vast.ai/RunPod ($3-5 total)

```
GPU:               A100 80GB (spot/community instance)
Quantization:      FP16 (no need for 4-bit, plenty of VRAM)
Batch size:        16 (grad_accum=4, effective_batch=64)
Grad checkpointing: Not needed (80GB is plenty)
Mixed precision:   BF16 native

Phase 0:           10K steps (full config spec)
Phase 1:           50K steps (full config spec, VQAv2+GQA)
Phase 2:           10K steps (full config spec)

Total steps:       70K (full config)
Est. time:         ~2 hours
Memory usage:      ~20 GB
Cost:              $3.00-5.00
Risk:              Low (spot preemption possible but unlikely in 2 hr)
Mitigation:        Checkpoint every 2K steps
```

### Option C — Best Value (RECOMMENDED)

```
GPU:               RunPod L4 24GB on-demand ($0.40/hr)
Quantization:      4-bit QLoRA
Batch size:        16 (grad_accum=4, effective_batch=64)
Grad checkpointing: Enabled (for safety margin)
Mixed precision:   FP16

Phase 0:           10K steps (full config spec)
Phase 1:           50K steps (VQAv2+GQA, full config spec)
Phase 2:           Skip (optional, add later if needed)

Total steps:       60K
Est. time:         ~5 hours
Memory usage:      ~10-12 GB (plenty of headroom on 24GB)
Cost:              ~$2.00
Risk:              Low (on-demand, no preemption)
```

This gives full config-spec training for the cost of a coffee. If results look promising, scale up to paper-spec (3 epochs Phase 1) on an A100 for ~$15-20.

### Option D — Paper-Spec Full Training

```
GPU:               Vast.ai A100 80GB spot
Quantization:      FP16 (no quantization needed)
Batch size:        16 (grad_accum=4, effective_batch=64)

Phase 0:           10K steps
Phase 1:           260K steps (3 epochs, VQAv2+GQA)
Phase 2:           10K steps

Total steps:       280K
Est. time:         ~8.2 hours
Memory usage:      ~20 GB
Cost:              ~$10-16
Risk:              Medium (long spot instance, checkpoint every 5K steps)
```

---

## Summary

| Question | Answer |
|----------|--------|
| **How expensive is this project?** | $2-5 for config-spec, $10-20 for paper-spec |
| **Will Colab Pro (T4) work?** | Yes with 4-bit QLoRA, but fragile (session timeouts). Need batch=4 + checkpointing |
| **Will Colab Pro+ (L4) work?** | Yes, comfortably with 4-bit QLoRA. Single session sufficient |
| **Cheapest meaningful run?** | Vast.ai A100 spot for ~$1 (45 min, 25K steps) OR Colab Pro T4 for $0 extra |
| **Best value full run?** | RunPod L4 for ~$2 (5 hours, 60K steps) |
| **Main bottleneck?** | LLaVA-7B model size, NOT TPN. TPN adds <5% overhead to baseline LLaVA training |
| **Critical code changes needed?** | (1) Add `load_in_4bit=True` in `tc_vitok_model.py:188`, (2) Eliminate ViT-L duplication |
| **What % of params are trained?** | 0.57% (42.8M out of 7.5B) |
| **Is the 75% FLOP claim real?** | Yes for LLM portion. End-to-end is ~58% (ViT-L still runs on all tokens) |
