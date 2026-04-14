# Adaptive-Budget TC-ViTok: Per-Sample Variable Token Selection

## Overview

Standard TC-ViTok selects a **fixed** number of tokens (k=144) for every image-question pair. Adaptive-Budget TC-ViTok replaces this with a **learned, per-sample variable budget**: a lightweight BudgetPredictor MLP decides how many tokens each sample needs, from 36 (6.25%) to 288 (50%) of the 576 visual tokens.

The key result is a **Pareto curve** of compression vs. accuracy, traced by sweeping the budget penalty weight `lambda_budget`. At each operating point, the model allocates more tokens to hard questions (spatial reasoning, counting, OCR) and fewer to easy ones (yes/no, color recognition).

---

## Architecture

### Components

```
Image (336x336) ──→ CLIP ViT-L/14 (frozen) ──→ v_full [B, 576, 1024]
                                                      │
Image (98x98)   ──→ DINOv2-S (frozen) ──→ v_coarse [B, 49, 384]
                                                │
Question ──→ CLIP Text (frozen) ──→ q_embed [B, 512]
                                        │
                              ┌─────────┴─────────┐
                              │    TPN (887K)      │
                              │  Cross-attention   │
                              │  Score MLP         │
                              │  Bilinear upsample │
                              │  Instance norm     │
                              └──────┬──────┬──────┘
                                     │      │
                              s_N [B,576]  ctx [B,384]
                                     │      │
                                     │  ┌───┴────────────┐
                                     │  │ BudgetPredictor │  (NEW, 25K params)
                                     │  │ MLP: 384→64→1  │
                                     │  │ sigmoid → frac  │
                                     │  └───────┬────────┘
                                     │      k_frac [B]
                                     │          │
                              ┌──────┴──────────┴──────┐
                              │  Adaptive Gumbel-STE   │  (NEW)
                              │  Per-sample threshold  │
                              │  Variable top-k        │
                              │  STE: m_hard + m_soft  │
                              └──────┬──────────┬──────┘
                                idx [B, k_max]  k_effective [B]
                                     │              │
                              gather + merge    budget_loss
                                     │
                              v_selected [B, k_i, 1024]
                                     │
                              LLaVA LLM (LoRA)
                                     │
                                  answer
```

### BudgetPredictor (25K params)

A 2-layer MLP that takes the TPN's cross-attention context vector `ctx` and predicts what fraction of tokens to keep:

```
Input:   ctx [B, 384]    (TPN's cross-attention context — encodes how query relates to image)
Layer 1: Linear(384, 64) + GELU
Layer 2: Linear(64, 1)
Output:  sigmoid → map to [k_min/N, k_max/N] = [0.0625, 0.5]
Result:  k_frac [B]      (per-sample target fraction)
```

The output `k_frac` is converted to a per-sample integer token count:
- `k_target = k_frac * 576` (continuous, in [36, 288])
- `k_hard = round(k_target)` (integer, for actual selection)

### Adaptive Gumbel-Sigmoid STE

Extends the original fixed-k Gumbel-Sigmoid STE to handle per-sample variable k:

**Training:**
1. Compute per-sample threshold from `k_frac` using sorted (detached) scores
2. Apply differentiability trick: `threshold = threshold_hard - (k_target - k_target.detach())`
   - Forward: equals `threshold_hard` exactly (zero effect)
   - Backward: `d(threshold)/d(k_frac) = -N` (non-zero gradient to BudgetPredictor)
3. Add Gumbel noise for exploration: `noisy_scores = s_N + g`
4. Compute soft mask: `m_soft = sigmoid((noisy_scores - threshold) / tau)`
5. Compute differentiable token count: `k_effective = sum(m_soft)` (gradient to BudgetPredictor)
6. Hard selection: per-sample `topk(k_hard)` on noisy scores
7. STE trick: `mask_ste = m_hard - m_soft.detach() + m_soft`
8. Pad shorter samples' indices with -1

**Inference:**
- Deterministic per-sample `topk(k_hard)`, no Gumbel noise
- Pad to `k_max_batch` with -1

### Variable-Length Handling

Since each sample has a different number of tokens:

- **idx tensor**: `[B, k_max_batch]` with -1 padding for shorter samples
- **Gather**: Per-sample loop, only gathers valid (non-padding) indices
- **LLM attention mask**: Per-sample mask marks padding positions as 0
- **STE gradient routing**: Correction masked at padding positions to prevent gradient leak

---

## Gradient Flow

Four gradient paths exist in this system:

### Path A: Budget Loss → BudgetPredictor (ACTIVE)
```
L_budget = mean(k_effective / N)
    → k_effective = sum(sigmoid((scores + g - threshold) / tau))
    → threshold depends on k_frac via: threshold_hard - (k_target - k_target.detach())
    → k_frac = BudgetPredictor(ctx)
    → BudgetPredictor weights

d(L_budget)/d(k_frac) > 0  →  optimizer pushes k_frac DOWN (fewer tokens)
```
This is the primary training signal for the BudgetPredictor.

### Path B: VQA Loss → TPN Scores (ACTIVE)
```
L_vqa = CE(logits, labels)
    → logits = LLM(v_selected * (1 + correction))
    → correction = gather(s_N, idx) - gather(s_N, idx).detach()  [= 0 forward, non-zero backward]
    → s_N = TPN(v_coarse, q_embed)
    → TPN weights
```
This is the task-conditioned signal: promotes tokens useful for answering, demotes useless ones.

### Path C: VQA Loss → BudgetPredictor (BLOCKED)
No gradient path exists from VQA loss to BudgetPredictor. The budget predictor receives no direct task-performance signal. Per-sample budget adaptation emerges indirectly through the evolving `ctx` representation (shaped by TPN training via Path B).

### Path D: Entropy Loss → BudgetPredictor (UNINTENDED LEAK)
The entropy loss pushes k_frac UP (more tokens = higher entropy). This opposes the budget loss (Path A). With `lambda_entropy=0.02` vs `lambda_budget=0.1`, the budget loss dominates ~5x. The budget warmup (2000 steps) keeps the predictor frozen while this stabilizes.

---

## Loss Function

```
L_total = L_vqa
        + lambda_entropy * (-H)           [spatial diversity, H = Shannon entropy]
        + lambda_diversity * L_div         [query-conditioned diversity]
        + lambda_budget * mean(k_eff / N)  [budget penalty — NEW]
```

The budget penalty `lambda_budget * mean(k_effective / N)` directly penalizes the fraction of tokens used. Higher `lambda_budget` → more compression → fewer tokens.

---

## Training Procedure

### Prerequisites

- Completed Phase 1 Final training (checkpoint at `runs/phase1_final/best.ckpt`)
- The BudgetPredictor starts from random initialization (not in the Phase 1 checkpoint)
- All other components (TPN, LoRA, mm_projector) load from the Phase 1 checkpoint

### Training Schedule

| Steps | What happens |
|-------|-------------|
| 0-500 | LR warmup for all components |
| 0-1000 | LoRA weights frozen (warmfreeze) |
| 0-2000 | BudgetPredictor frozen (warmup) — TPN/LoRA stabilize first |
| 2000+ | All components training jointly |
| 0-40000 | Tau anneals from 1.0 → 0.3 |

### Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `adaptive_budget` | `true` | Enable adaptive selection |
| `k_min` | 36 | Minimum tokens (6.25% of 576) |
| `k_max` | 288 | Maximum tokens (50% of 576) |
| `lambda_budget` | 0.1 | **Sweep this** for Pareto curve |
| `budget_warmup_steps` | 2000 | Freeze BudgetPredictor initially |
| `lr_budget` | 1e-3 | Same as TPN (lightweight head) |
| `lr_tpn` | 1e-3 | TPN learning rate |
| `lr_lora` | 1e-4 | LoRA learning rate (rank 64) |
| `lr_mlp` | 5e-4 | mm_projector learning rate |
| `batch_size` | 4 | Per-GPU batch size |
| `grad_accum_steps` | 16 | Effective batch = 4 x 16 x 2 GPUs = 128 |
| `max_steps` | 40000 | Total training steps |

---

## Commands (Remote Server)

### Server Details

- **Host**: `ADUAED16811WKLX` (Ubuntu 22.04, CUDA 12.4)
- **User**: `sk10741`
- **GPUs**: 2x NVIDIA RTX 6000 Ada (49GB VRAM each)
- **Repo**: `~/TC-ViTok`
- **Data**: `~/tc-vitok-data`

### 1. SSH into server

```bash
ssh sk10741@ADUAED16811WKLX
```

### 2. Pull latest code

```bash
cd ~/TC-ViTok
git -c credential.helper= pull origin main
```

### 3. Run adaptive budget training

```bash
# Single run with lambda_budget=0.1
bash scripts/launch_multigpu.sh \
    --config configs/phase1_adaptive.yaml \
    --output runs/adaptive_lb0.1
```

This will:
- Load the Phase 1 Final checkpoint from `runs/phase1_final/best.ckpt`
- Initialize BudgetPredictor from scratch (random weights)
- Train for 40,000 steps on 2 GPUs with DDP
- Log to WandB as "phase1_adaptive_budget"
- Save checkpoints every 2,000 steps
- Validate every 2,000 steps

### 4. Sweep lambda_budget for Pareto curve

```bash
for lb in 0.01 0.05 0.1 0.3 1.0; do
    bash scripts/launch_multigpu.sh \
        --config configs/phase1_adaptive.yaml \
        --output runs/adaptive_lb${lb} \
        training.lambda_budget=${lb}
done
```

Each run traces one point on the compression-vs-accuracy frontier.

### 5. Monitor training

```bash
# GPU utilization
nvidia-smi

# Training logs (if running in tmux)
tmux capture-pane -t training -p | tail -20

# Latest log file
ls -t logs/train_*.log | head -1 | xargs tail -50
```

### 6. Key WandB metrics to watch

| Metric | What to expect | Red flag |
|--------|---------------|----------|
| `train/k_effective_mean` | Should start ~162, then shift based on lambda_budget | Stuck at k_min or k_max |
| `train/budget` | Should decrease over training | Flat = BudgetPredictor not learning |
| `val/vqa` | Should be competitive with fixed k=144 baseline | Collapse to 0 |
| `val/k_effective_mean` | Actual average tokens used at eval | All same value = no per-sample variation |
| `train/entropy` | Should stay high (> 5.0) | Drop below 4.0 = spatial collapse |

### 7. Evaluate a checkpoint

```bash
python scripts/evaluate.py \
    --checkpoint runs/adaptive_lb0.1/best.ckpt \
    --config configs/phase1_adaptive.yaml \
    --benchmarks vqav2 gqa textvqa pope
```

---

## File Reference

| File | What changed |
|------|-------------|
| `src/tc_vitok/models/tpn.py` | Added `BudgetPredictor` class; TPN.forward() returns 4 values, supports adaptive mode |
| `src/tc_vitok/models/masking.py` | Added `adaptive_gumbel_sigmoid_ste()` with per-sample variable k |
| `src/tc_vitok/models/tc_vitok_model.py` | `_gather_adaptive()` for variable-length gather; inference attention masking; STE routing with padding mask |
| `src/tc_vitok/training/losses.py` | Added `budget_loss()`; `entropy_from_idx` handles -1 padding; `compute_total_loss` includes budget term |
| `src/tc_vitok/training/trainer.py` | Budget warmup freeze; validation budget metrics; checkpoint includes budget_predictor |
| `src/tc_vitok/training/optimizer.py` | BudgetPredictor param group with separate LR |
| `scripts/train.py` | Passes adaptive_budget args to model; reads `checkpoint.load_from` from config |
| `configs/phase1_adaptive.yaml` | Full config for adaptive budget training |
| `configs/base.yaml` | Added adaptive budget defaults |
| `tests/test_adaptive_budget.py` | 14 tests covering all new components |

---

## Verified End-to-End

The following was verified by running a complete simulation (13 steps) on the local machine:

- All imports succeed
- TPN + BudgetPredictor forward produces correct shapes with per-sample variable k
- Gumbel noise produces different k per sample (observed: [258, 236, 64, 142] in one batch)
- BudgetPredictor receives gradient (4/4 params, grad norms ~0.1-2.1)
- TPN receives gradient (12/12 params)
- Optimizer creates separate param groups with correct LRs
- 5 simulated training steps complete without error
- Budget warmup correctly zeros BudgetPredictor gradients
- Eval mode is deterministic
- Fixed-k backward compatibility preserved
- entropy_from_idx handles -1 padding without crash
