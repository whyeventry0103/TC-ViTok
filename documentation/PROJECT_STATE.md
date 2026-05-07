# TC-ViTok: Complete Project State
*Last updated: 2026-05-07*

---

## What Is This Project

**TC-ViTok** (Task-Conditioned Visual Tokenization) is a research project that makes Vision-Language Models (VLMs) more efficient by learning *which* visual tokens to keep based on the input question.

### The Problem
LLaVA-1.5-7B encodes every image into exactly **576 visual tokens**, regardless of whether the question is "Is the sky blue?" (needs ~10 tokens) or "How many people are in the background by the red car?" (needs many more). Every query pays the full 576-token prefill cost in the LLM — wasteful and uniform.

### The Idea
Learn a **policy** that looks at the question and the image together, then selects only the most relevant visual tokens. The LLM then sees a compressed visual prefix (e.g. 144 tokens instead of 576), reducing LLM compute by 75%.

The full pipeline:
```
Question + Low-res Image → TPN → Token scores (576) → top-k selection
                                                          ↓
High-res Image → CLIP ViT-L/14 → 576 tokens → gather(idx) → 144 tokens → LLaVA-7B → Answer
```

---

## Architecture (All Components)

### 1. Coarse Visual Encoder — DINOv2-Small (frozen)
- Input: 98×98 image
- Output: 49 tokens (7×7 grid), 384-dim
- **Purpose:** Fast, low-res features for the TPN to score all 576 spatial positions

### 2. Text Encoder — CLIP ViT-B/32 (frozen)
- Input: question string
- Output: 512-dim [EOS] embedding
- **Purpose:** Question representation so token selection is query-conditioned

### 3. Tokenization Policy Network (TPN) — ~887K params (trainable)
- Input: 49 coarse tokens (384-dim) + question embedding (512-dim)
- Architecture: Cross-attention (H=4 heads) → concat [context; attention_weights] (433-dim) → Score MLP → bilinear upsample 7×7→24×24 → **instance normalize**
- Output: importance scores `s_N` [B, 576] + top-k indices + context vector (384-dim)
- **Key detail:** Instance normalization (mean=0, std=1 per sample) prevents score explosion and keeps gradients stable

### 4. Gumbel-Sigmoid STE — differentiable top-k selection
The core technical challenge: token selection is discrete (hard top-k), but training requires gradients.

**Fixed-k mode (Phase 1 Final):**
```
Forward:  m_hard = topk(s_N + gumbel_noise, k=144) → binary {0,1}
Backward: gradient flows through m_soft = sigmoid((s_N + g) / tau)
STE:      mask = m_hard - m_soft.detach() + m_soft
```

**Adaptive mode (Phase 1 Adaptive):**
- BudgetPredictor predicts per-sample fraction k_frac → k_target
- Threshold derived from sorted scores at position k_target
- Gradient to BudgetPredictor via: `threshold = threshold_hard - (k_target - k_target.detach())`

**STE correction (additive, tau-independent):**
After gathering visual tokens, multiply by `(1 + s - s.detach())` so VQA loss gradient reaches TPN scores even when tau→0 and sigmoid gradients vanish.

### 5. Informative Token Merge — parameter-free (Phase 1 Final key innovation)
After selecting 144 tokens, the 432 unselected tokens are not discarded — they are merged back into the 144 selected tokens using cosine-similarity weighted aggregation:

```
For each selected token i:
  similarity(i, j) = cosine_sim(v_selected_i, v_unselected_j)  for all j in unselected
  assignment = softmax(similarity)      # which selected token each unselected token belongs to
  contribution = assignment.T @ v_unselected   # weighted sum into each selected slot
  v_merged_i = v_selected_i + beta * contribution_i * tpn_weight_i
```

- `beta=0.3` — merge strength (0 = no merge, 1 = full replace)
- TPN scores used as per-selected-token weighting: tokens the TPN was more confident about get more informative merges
- **Zero new parameters** — no new modules to train, no co-adaptation instability

**Why this matters:** Previous approach (Phase 1.5, 8 failed versions) tried to recycle unselected tokens via a learned `UnselectedPooler` (cross-attention with 4 slots). The pooler and LoRA oscillated during training because they co-adapt: pooler output changes → LoRA must re-adapt → LoRA gradient changes → pooler re-adapts. After 8 versions and months of work, this was abandoned in favor of parameter-free token merge.

### 6. Full Visual Encoder — CLIP ViT-L/14 (frozen)
- Input: 336×336 image
- Output: 576 tokens (24×24 grid), 1024-dim
- **Purpose:** High-resolution features; only the `gather(idx)` call retrieves the k=144 selected tokens

### 7. MM Projector — LLaVA's mm_projector (trainable, lr=5e-4)
- Maps visual features from 1024-dim to LLM hidden dim
- Standard LLaVA component

### 8. LLaVA-1.5-7B + LoRA (frozen base + trainable LoRA)
- Base model frozen
- **LoRA rank=64, alpha=128** on `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj`
- gate/up targets included because token merge shifts the visual feature distribution — FFN layers need to adapt to interpret the merged tokens
- lr=1e-4 (lower than typical because rank=64 has more capacity; higher LR causes overshoot)

### 9. BudgetPredictor — ~25K params (Phase 1 Adaptive only)
- Input: TPN context vector (384-dim) + entropy of TPN scores (normalized)
- Architecture: 2-layer MLP (384→64→1) + sigmoid
- Output: per-sample token fraction k_frac ∈ [k_min/N, k_max/N]
- **Purpose:** Allocate more tokens to hard questions (spatial reasoning, counting) and fewer to easy ones (yes/no, color)

---

## Training Pipeline

### Why Multiple Phases?

Training everything from scratch simultaneously doesn't work:
- BudgetPredictor needs meaningful TPN scores to predict sensible budgets
- LoRA needs stable visual input to converge
- TPN needs gradient signal to learn meaningful token scores

The phased curriculum solves this by establishing a solid foundation before adding complexity.

---

### Phase 0 — TPN Warmup (COMPLETE)
**Checkpoint:** `runs/phase0_v4/phase0/best.ckpt`

- Only TPN trains; LoRA is rank=32 (lightweight, just for gradient signal)
- LLaVA backbone frozen
- Loss: VQA cross-entropy only (`lambda_max=0.01`, `lambda_entropy=0.01` for basic regularization)
- Dataset: VQAv2 subset
- **Goal:** TPN learns basic token scoring from question-image pairs before the full system runs

---

### Phase 1 Final — Joint Training with Token Merge (CURRENTLY RUNNING)
**Config:** `configs/phase1_final.yaml`
**Output:** `runs/phase1_final/`
**Steps:** 80,000 total | **Current:** ~55,000 ✅

**What it does:**
- Loads Phase 0 TPN weights; discards Phase 0 LoRA (rank=32) and reinitializes at rank=64
- Trains TPN + LoRA + MM projector jointly
- Fixed token budget k=144 (25% of 576)
- `informative_token_merge` active (beta=0.3): unselected tokens merged into selected
- Datasets: VQAv2 + GQA combined

**Key hyperparameters:**
| Parameter | Value | Reason |
|-----------|-------|--------|
| k | 144 | 75% compression, good accuracy tradeoff |
| LoRA rank/alpha | 64/128 | More capacity than Phase 0 for full VQA |
| lr_tpn | 3e-3 | TPN is small, learns quickly |
| lr_lora | 1e-4 | High rank needs lower LR to avoid overshoot |
| lr_mlp | 5e-4 | MM projector |
| tau annealing | 1.0→0.3 | Hard top-k behavior increases over training |
| lora_warmfreeze_steps | 3000 | TPN trains alone first to stabilize before LoRA unfreezes |
| lambda_entropy | 0.02 | Prevent spatial collapse (all tokens from one region) |
| lambda_diversity | 0.1 | Prevent query-ignoring (same tokens for all questions) |
| merge_beta | 0.3 | 30% merge strength |

**Current validation trajectory:**
| Step | val_loss | val_vqa |
|------|----------|---------|
| 5k   | 0.3275   | 0.4313  |
| 10k  | 0.3031   | 0.4053  |
| 15k  | 0.3395   | 0.4423  ← LoRA stabilization spike |
| 20k  | 0.3041   | 0.4075  |
| 30k  | 0.2856   | 0.3881  |
| 40k  | 0.2797   | 0.3828  |
| 45k  | 0.2530   | 0.3562  ← breakthrough |
| 50k  | 0.2459   | 0.3494  |
| **55k** | **0.2400** | **0.3438** |

**Baseline to beat:** Phase 1 v3 reached `val_vqa=0.3334` at 50k steps (LoRA rank=32, no merge).

Trend from 45k–55k: dropping ~0.006/5k steps. Projected at 80k: **~0.315–0.32**, beating the baseline around step 65k.

---

### Phase 1 Adaptive — Adaptive Token Budget (NEXT)
**Config:** `configs/phase1_adaptive.yaml`
**Loads from:** `runs/phase1_final/best.ckpt`
**Steps:** 40,000

**What it adds:**
- Initializes BudgetPredictor from scratch (not in Phase 1 Final checkpoint)
- All other weights (TPN, LoRA, MM projector) loaded from Phase 1 Final
- Fixed k=144 replaced by per-sample variable k ∈ [36, 288]
- Budget warmup: BudgetPredictor frozen for first 2000 steps (LoRA stabilizes first)
- LoRA warmfreeze: 1000 steps
- Budget penalty loss: `lambda_budget * mean(k_effective / N)` penalizes using more tokens

**Gradient paths for BudgetPredictor:**
| Path | Mechanism | Status |
|------|-----------|--------|
| A: Budget Loss → BP | `mean(k_eff/N)` → threshold → k_frac → BP weights | Active |
| B: VQA Loss → TPN | STE additive correction | Active |
| C: VQA Loss → BP | No direct path; indirect via ctx evolution | Blocked |
| D: Entropy Loss → BP | Pushes k_frac up; dominated 5× by budget loss | Leak (acceptable) |

**Why warm-start from Phase 1 Final:**
- TPN scores already meaningful → BudgetPredictor gets real entropy signal immediately
- LoRA already converged → only needs to fine-tune for variable-length inputs
- Only 40k more steps needed vs. training from scratch

**Pareto curve:** Sweep `lambda_budget ∈ {0.05, 0.1, 0.2, 0.4}` → each produces a different point on compression-vs-accuracy tradeoff. The adaptive model should outperform fixed-k at the same average token count because it allocates intelligently.

---

## Key Design Decisions and Why

### Why informative_token_merge instead of UnselectedPooler?

Phase 1.5 (8 versions) spent months trying to train a learned pooler to recycle unselected tokens. All 8 failed due to **co-adaptation oscillation**: pooler and LoRA fight each other because each changes the target distribution the other is trying to fit. Even with pooler LR as low as 2e-5, oscillation period was 10-15k steps.

Token merge is parameter-free — there's nothing to oscillate. The merge weights come directly from TPN scores and cosine similarity, both of which are already stable. This eliminated the oscillation completely and produces cleaner training curves.

### Why instance normalization on TPN scores?

Without it, scores grow unbounded over training (TPN learns to push high-importance scores increasingly large). Large absolute scores cause Gumbel-Sigmoid to saturate (sigmoid →1 for all selected, →0 for all unselected), which kills the soft gradient in the backward pass. Instance norm (mean=0, std=1) keeps scores in a stable range throughout training.

### Why the STE additive correction?

```python
scores_at_selected = gather(s_N, idx)
v_selected = v_selected * (1 + scores_at_selected - scores_at_selected.detach())
```

At `tau=0.1` (late training), `sigmoid'(x/tau)` ≈ 0 for all |x| > 1 — the soft mask gradient path vanishes. Without this correction, TPN receives no gradient from VQA loss at low temperature. The additive correction is a second independent gradient path that works at any temperature, ensuring TPN always gets signal from VQA.

### Why rank=64 for Phase 1 Final (vs rank=32 in Phase 1 v3)?

Token merge shifts the visual feature distribution — merged tokens look different from pure selected tokens because they carry aggregated background information. LoRA needs more capacity to adapt the LLM to interpret this new distribution. Rank=64 provides that capacity. Rank=32 was sufficient when there was no merge; rank=64 is needed now.

### Why gate/up proj in LoRA targets?

The MM projector maps visual features to LLM space, but the FFN layers (gate/up) handle the actual non-linear processing inside the LLM. Token merge changes what the LLM sees — the FFN needs to adapt to correctly process merged tokens that encode both selected content and compressed background context.

---

## Loss Function

```
L_total = L_vqa                                    [main task]
        + lambda_entropy * (-H)                    [spatial diversity]
        + lambda_diversity * L_div                 [query diversity]
        + lambda_budget * mean(k_effective / N)    [Phase 1 Adaptive only]
```

**Spatial entropy (`lambda_entropy=0.02`):**
```
H = -Σ p_i * log(p_i)    where p_i = batch-mean selection frequency at position i
loss_ent = -H             (minimize loss = maximize entropy = spread tokens spatially)
```
Prevents TPN from always selecting center tokens regardless of question.

**Diversity loss (`lambda_diversity=0.1`):**
Penalizes when two different questions from the same batch produce similar score vectors. Implemented as within-batch pairing (first half vs second half) using raw TPN scores. Not identical to the original PairedVQADataset formulation but works as a within-batch regularizer.

**Budget loss (Phase 1 Adaptive, `lambda_budget=0.1`):**
`mean(k_effective / N)` — penalizes using more tokens. Higher lambda_budget → smaller average k. Sweep to trace Pareto curve.

---

## Infrastructure

**Server:** Ubuntu 22.04, 2× RTX 6000 Ada (49GB each), CUDA 12.4
**Training:** 2-GPU DDP via `scripts/launch_multigpu.sh`
**Precision:** BF16 mixed precision
**Effective batch size:** 4 samples/GPU × 16 grad accum × 2 GPUs = 128
**Speed:** ~1.95 it/s → 80k steps ≈ 11.4 hours

**Key paths on server:**
```
~/TC-ViTok/                              # repo
~/tc-vitok-data/                         # VQAv2, GQA, HF model weights
runs/phase0_v4/phase0/best.ckpt          # Phase 0 checkpoint
runs/phase1_final/                       # Phase 1 Final (in progress)
```

**Sync local → server:**
```bash
git add -A && git commit -m "..." && git push
# on server:
git -c credential.helper= pull origin main
```

**Launch / monitor:**
```bash
# Start Phase 1 Final (if not already running)
bash scripts/launch_multigpu.sh --config configs/phase1_final.yaml --output runs/phase1_final

# Check progress
tmux attach -t <session_name>
tail -f logs/train_*.log | tr '\r' '\n' | grep -E "Step|val"
```

---

## Current Status Summary

| Phase | Status | Best val_vqa | Notes |
|-------|--------|-------------|-------|
| Phase 0 | ✅ Complete | — | TPN warmup done |
| Phase 1 v3 | ✅ Complete (old baseline) | 0.3334 @ 50k | LoRA rank=32, no merge |
| Phase 1.5 (v1–v8b) | ❌ All failed/killed | 0.3663 @ v8 step 45k | Co-adaptation oscillation |
| **Phase 1 Final** | 🟡 Running (55k/80k) | **0.3438 @ 55k** | Stable, on track to beat 0.3334 |
| Phase 1 Adaptive | ⏳ Waiting | — | Starts after Phase 1 Final |

**Expected outcome:** Phase 1 Final will reach ~0.315–0.32 by step 80k, beating the Phase 1 v3 baseline (0.3334) around step 65k. Phase 1 Adaptive then adds per-sample variable budget on top of this foundation.

---

## What Comes After Phase 1 Adaptive

1. **Evaluate on VQAv2, GQA, POPE benchmarks** using `scripts/evaluate.py`
2. **Pareto sweep** — run Phase 1 Adaptive with `lambda_budget ∈ {0.05, 0.1, 0.2, 0.4}` to trace compression-vs-accuracy tradeoff
3. **Ablations for paper:**
   - Token merge vs. no merge (hard drop) — quantifies merge contribution
   - Adaptive vs. fixed-k at same average token count — shows adaptive allocation benefit
   - Query-conditioned vs. image-only TPN — proves query conditioning works (GQA spatial gap)
4. **Optional:** Spatial spread penalty, teacher distillation — only if Phase 1 Adaptive doesn't hit target accuracy

---

## Bugs Fixed (Relevant Ones)

| Bug | File | Fix |
|-----|------|-----|
| Token-0 overwrite: invalid idx clamped to 0 corrupted position 0 in scatter_ | masking.py | Use topk indices directly (always valid), scatter valid_mask values |
| LoRA rank mismatch on cross-phase checkpoint load: rank=32 checkpoint → rank=64 model crashes even with strict=False | trainer.py | Filter state_dict to remove shape-mismatched keys before load_state_dict |
| Phase 0 TPN zero gradient (lambda_max=0 → no loss signal) | phase0.yaml | lambda_max=0.01, lambda_entropy=0.01 |
| DDP param mismatch from frozen vision tower | train.py | find_unused_parameters=True |
| NCCL timeout from HF HTTP desync across ranks | train.py | HF_HUB_OFFLINE=1, 30min PG timeout |
| DINOv2-S expects 518px but receives 98px | encoders.py | img_size=98 in timm.create_model |
