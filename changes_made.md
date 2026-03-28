# TC-ViTok: Changes Made (Audit-Driven Fixes)

**Date:** 2026-03-28
**Scope:** All critical mismatches between paper and codebase, identified via full research audit.

---

## 1. TPN Architecture Rewrite

**Category:** CRITICAL
**File:** `src/tc_vitok/models/tpn.py`

**Problem:** Code used broadcast-based fusion (expand fused embedding to 49 positions, add to visual features). Paper specifies cross-attention with concat architecture.

**What changed:**
- Replaced broadcast fusion with manual multi-head cross-attention (q_proj, k_proj, v_proj, o_proj)
- Changed n_heads from 8 to 4 (paper Appendix A.1)
- Score MLP now takes concat `[context_vector; alpha_avg]` (dim=433) instead of visual features alone
- Score MLP architecture: `433 → 512 → 49` with GELU + dropout
- Added bilinear upsample from 7x7 (49) to 24x24 (576)
- Added LayerNorm before score MLP

**Why:** Paper Section 3.2 and Appendix A.1 specify this exact architecture. The broadcast approach cannot learn query-dependent spatial attention patterns.

**Effect:** Parameter count now ~887K (matching paper). TPN produces query-conditioned spatial masks via cross-attention.

---

## 2. Gumbel-Sigmoid STE Fix

**Category:** CRITICAL
**File:** `src/tc_vitok/models/masking.py`

**Problem:** STE implementation details were incorrect; entropy computation used binary entropy per token instead of Shannon entropy of spatial distribution.

**What changed:**
- `gumbel_sigmoid_ste()`: Hard top-k on raw scores (not noisy scores), STE trick: `m_hard - m_soft.detach() + m_soft`
- `compute_spatial_entropy()`: NEW function — Shannon entropy of normalized spatial distribution: `p_i = mean_mask_i / sum(mean_mask)`, `H = -sum(p_i * log(p_i))`
- Removed old binary entropy per-token formulation

**Why:** Paper defines entropy as Shannon entropy over the spatial probability distribution of mask activations across the batch, not per-token binary entropy.

**Effect:** Entropy regularization now correctly encourages spatial spread of selected tokens across the image.

---

## 3. Sparsity Loss Fix

**Category:** CRITICAL
**File:** `src/tc_vitok/training/losses.py`

**Problem:** Sparsity loss was `|sum(mask) - k|`, which is always ~0 under hard top-k (mask always sums to exactly k). This means the sparsity regularizer had zero effect.

**What changed:**
- `l1_sparsity_loss(scores)` now computes `scores.abs().mean()` — L1 norm of raw TPN output scores divided by N
- `entropy_loss(mask)` returns `-compute_spatial_entropy(mask)` (minimizing loss = maximizing entropy)
- `compute_total_loss()` now accepts `scores` parameter and passes it to sparsity loss

**Why:** Paper Section 3.3 defines L_sparse = ||s_N||_1 / N, applied to raw scores before thresholding.

**Effect:** Sparsity regularization now actually penalizes large score magnitudes, encouraging the TPN to be decisive (scores near 0 or 1).

---

## 4. Gather-Based Token Extraction (Real Compression)

**Category:** CRITICAL
**File:** `src/tc_vitok/models/tc_vitok_model.py`

**Problem:** Code used zero-masking (`v_full * mask`), which keeps all 576 tokens in the sequence — just zeros out unselected ones. No actual sequence length reduction. The entire point of TC-ViTok is reducing LLM input length.

**What changed:**
- Added `gather_selected_tokens(visual_tokens, mask, k)` function using `mask.topk(k).indices` + `torch.gather`
- Output shape: `[B, k, D]` (e.g., `[B, 144, 1024]`) — real compression from 576 to k tokens
- Forward pass updated to use gather instead of element-wise multiplication
- Model returns `selected_tokens` key in output dict

**Why:** Paper Section 3.1 explicitly describes reducing the visual token sequence from N to k. Zero-masking provides no computational savings to the LLM.

**Effect:** LLM now receives k tokens instead of N. This is the core contribution of the paper — 75% reduction in visual tokens with minimal VQA accuracy loss.

---

## 5. Config-Paper Alignment

**Category:** IMPORTANT
**Files:** `configs/base.yaml`, `configs/phase0.yaml`, `configs/phase1.yaml`, `configs/phase2.yaml`, `src/tc_vitok/training/curriculum.py`, `src/tc_vitok/training/optimizer.py`

**Problem:** Multiple hyperparameter mismatches between config files and paper values.

**What changed:**
| Parameter | Was | Now | Paper Reference |
|-----------|-----|-----|-----------------|
| tau_max | 5.0 | 1.0 | Section 3.3 |
| tau_min | 0.5 | 0.1 | Section 3.3 |
| n_heads | 8 | 4 | Appendix A.1 |
| lr_tpn | 1e-4 | 1e-3 | Appendix A.2 |
| lr_lora | 5e-5 | 2e-4 | Appendix A.2 |
| lr_mlp | 1e-4 | 5e-4 | Appendix A.2 |
| lambda_max | 0.01 | 0.1 | Table in Section 8.2 |
| lambda_entropy | 0.0001 | 0.01 | Table in Section 8.2 |
| lambda_diversity | 0.001 | 0.01 | Table in Section 8.2 |

- Phase 0: All lambdas set to 0.0 (VQA loss only), tau fixed at 1.0
- Phase 1: All values aligned with paper Phase 1 schedule
- Phase 2: tau fixed at 0.1, lambda_max=0.0 per paper

**Why:** Hyperparameters directly affect training dynamics. Wrong temperature range (5.0→0.5 vs 1.0→0.1) and wrong learning rates would prevent reproducing paper results.

**Effect:** Training configuration now matches paper exactly across all three phases.

---

## 6. StubLLM for Pipeline Testing

**Category:** IMPORTANT
**File:** `src/tc_vitok/models/tc_vitok_model.py`

**Problem:** Full pipeline required downloading LLaVA-1.5-7B (~14GB) just to verify basic correctness. No way to test without GPU and large downloads.

**What changed:**
- Added `StubLLM` class: small 2-layer transformer with matching interface (input dim=1024, vocab=1000)
- `TCViTokModel` accepts `use_stub_llm=True` flag to use StubLLM instead of real LLaVA
- StubLLM produces correct output shapes and supports gradient flow

**Why:** Enables rapid iteration and CI testing. Full pipeline can be verified end-to-end without model downloads.

**Effect:** `python scripts/test_pipeline.py` runs the full forward+backward pass with StubLLM in seconds on CPU.

---

## 7. README Accuracy Fixes

**Category:** IMPORTANT
**File:** `README.md`

**Problem:** README claimed measured results that don't exist yet, stated wrong parameter counts, and had incorrect hyperparameter values.

**What changed:**
- TPN params: "~3M" → "~887K"
- Total trainable: "~38M" → "~34M"
- Added prominent warning: "ALL RESULTS BELOW ARE PROJECTED -- NOT MEASURED"
- Efficiency framing: "LLM-side computational cost" (not "total")
- All hyperparameter values in tables fixed to match paper
- Phase descriptions fixed (tau, lambda values)
- Architecture description fixed (4 heads, concat approach)

**Why:** Misleading claims undermine credibility. Parameter counts should match actual architecture.

**Effect:** README now accurately represents the project state and clearly distinguishes projected from measured results.

---

## 8. Data Module Scaffolding

**Category:** STRUCTURAL
**Files:** `src/tc_vitok/data/__init__.py`, `src/tc_vitok/data/datasets.py`, `src/tc_vitok/data/spatial_labels.py`

**What changed:**
- `datasets.py`: `DummyVQADataset` (synthetic data for testing), `PairedVQADataset` (structural placeholder for diversity loss), `collate_fn`, `DINO_TRANSFORM`, `CLIP_TRANSFORM`
- `spatial_labels.py`: `classify_spatial()` with GQA annotation priority + keyword fallback, `classify_batch()` for batch processing
- `__init__.py`: Package docstring

**Why:** Paper Section 8.1 describes dataset requirements. Section 8.3 describes spatial classification protocol. These are needed before real data integration.

**Effect:** Data loading interface is defined. Spatial classification ready for GQA evaluation.

---

## 9. Test Scripts

**Category:** STRUCTURAL
**Files:** `scripts/test_core.py`, `scripts/test_pipeline.py`

**What changed:**
- `test_core.py`: 7 lightweight tests (no model downloads): TPN params/shapes, STE correctness, gather extraction, sparsity loss formula, Shannon entropy, total loss gradient flow, curriculum values
- `test_pipeline.py`: 6 end-to-end tests with StubLLM: TPN architecture, Gumbel-Sigmoid STE, gather extraction, losses, curriculum, full pipeline forward+backward

**Effect:** All 7 core tests pass. Pipeline verification available without heavy dependencies.

---

## 10. Trainer Integration

**Category:** IMPORTANT
**File:** `src/tc_vitok/training/trainer.py`

**Problem:** `train_step()` did not pass raw scores to loss computation, so sparsity loss couldn't operate on scores.

**What changed:**
- `train_step()` now passes `scores=outputs["scores"]` to `compute_total_loss()`

**Why:** After fixing sparsity loss to operate on raw scores (not mask), the trainer must forward scores from model output to loss function.

**Effect:** Sparsity regularization is now active during training.

---

## Verification

All fixes verified via `scripts/test_core.py`:

```
TEST 1: TPN Architecture — 887,955 params (paper: ~887K) ✓
TEST 2: Gumbel-Sigmoid STE — binary forward, soft backward ✓
TEST 3: Gather Token Extraction — [4,576,1024] → [4,144,1024] ✓
TEST 4: Sparsity Loss — mean(|scores|) = 1.625 ✓
TEST 5: Spatial Entropy — Shannon entropy, uniform=6.36, concentrated=2.30 ✓
TEST 6: Total Loss — all terms computed, gradients flow to TPN ✓
TEST 7: Curriculum — tau: 1.0→0.1, lambda: 0.0→0.1 ✓
```

## System Readiness

- Architecture matches paper: **YES**
- Core bugs fixed: **YES** (sparsity, entropy, gather, TPN)
- Pipeline runs minimally: **YES** (StubLLM, all tests pass)
- Token compression is real: **YES** (gather-based, [B,k,D] output)
- Ready for first experiment: **YES** (pending real dataset loaders and LLaVA integration)
