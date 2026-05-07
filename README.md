# TC-ViTok: Task-Conditioned Visual Tokenization with Adaptive Budget

**Learning an Adaptive Token Selection Policy for Efficient Vision-Language Reasoning**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![PyTorch 2.6+](https://img.shields.io/badge/PyTorch-2.6%2B-orange.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Institution: LUMS](https://img.shields.io/badge/Institution-LUMS-red.svg)](https://lums.edu.pk)

Vision-Language Models like LLaVA-1.5 feed all **576 visual tokens** from every image into a 7B LLM regardless of what the question asks. **TC-ViTok** replaces this with a learned, query-conditioned token selection policy: a lightweight Tokenization Policy Network (TPN, ~887K params) that scores all 576 tokens and keeps only the ones relevant to the question. We build on this with two extensions: **(1)** parameter-free **Informative Token Merge** that recycles discarded tokens into the selected set at zero additional parameter cost, and **(2)** an **Adaptive Budget** that learns per-sample variable token counts from 36 to 288 — more tokens for hard spatial or counting questions, fewer for easy ones.

> **Efficiency note:** CLIP ViT-L/14 still processes all 576 tokens during encoding. The compression savings are on the LLM prefill side: 75% FLOPs reduction at k=144, 94% at k=36. End-to-end savings depend on what fraction of inference time is LLM prefill.

---

## Key Contributions

1. **Base TC-ViTok** — Query-conditioned token selection via TPN + Gumbel-Sigmoid STE. Frames visual tokenization as differentiable policy learning rather than static cropping or heuristic pruning.

2. **Informative Token Merge** — Parameter-free cosine-similarity-based merging of the 432 discarded tokens back into the 144 selected tokens, weighted by TPN importance scores. Zero new parameters, zero co-adaptation risk, no train-eval mismatch. Inspired by PruMerge-style approaches but query-conditioned via TPN scores.

3. **Adaptive Budget** — A BudgetPredictor MLP (~25K params) predicts per-sample token fraction from TPN context and score entropy. A differentiable threshold trick makes the discrete token count differentiable, allowing the predictor to learn compression-accuracy tradeoffs from the VQA signal. Sweeping `lambda_budget` traces a Pareto curve.

---

## Method

### 1. Core Idea: Visual Tokenization as Policy Learning

Standard VLMs apply fixed tokenization: every image → 576 tokens, regardless of question complexity. We reframe this as a policy learning problem:

```
π: (Query Q, Image I_coarse) → Token Selection Indices S ⊆ {1, ..., 576}
```

The policy (TPN) observes the question and a low-resolution image summary (49×384 from DINOv2-S), then selects which spatial regions of the full high-resolution CLIP encoding to pass to the LLM. Selected tokens are gathered with `torch.gather` — the LLM sees exactly k tokens instead of 576.

---

### 2. Tokenization Policy Network (TPN) — ~887K params

**Internal architecture (step by step):**

```
q_embed [B, 512]   ──► W_Q ──► Q [B, 1, 384]    (question as single query)
v_coarse [B, 49, 384] ──► W_K, W_V ──► K, V [B, 49, 384]

Multi-Head Cross-Attention (4 heads):
  attn = softmax(QK^T / √d_k)    [B, 4, 1, 49]
  ctx = attn @ V                 [B, 1, 384]     ← attended visual context
  α_avg = mean(attn, heads)      [B, 49]          ← spatial attention weights

Score MLP:
  LayerNorm([ctx; α_avg])        [B, 433]
  → MLP → s_M                   [B, 49]          ← coarse importance scores

Bilinear upsample 7×7 → 24×24:  [B, 576]
Instance normalization:          mean=0, std=1    ← s_N per sample
```

**Why instance normalization?** Without it, Adam pushes scores to ±100 within ~1000 steps. At that magnitude, Gumbel noise (τ≈1.0) is negligible → deterministic top-144 → same tokens always selected → entropy collapse. Instance norm anchors scores to mean=0, std=1 permanently. Since only relative ordering matters for top-k, no information is lost. No learnable parameters means the optimizer cannot tune it away.

---

### 3. Gumbel-Sigmoid Straight-Through Estimator (STE)

Token selection is discrete (hard top-k), but training requires gradients. We use:

**Forward pass — hard top-k with Gumbel exploration:**
```
g ~ Gumbel(0, 1)                                     # exploration noise
noisy = s_N + g                                       # perturbed scores
m_soft = sigmoid(noisy / τ)                           # soft mask (has gradient)
idx = topk(noisy, k).indices                          # hard selection (no gradient)
m_hard = scatter(idx, 1.0)                            # binary mask
mask_ste = m_hard − m_soft.detach() + m_soft          # STE: forward=m_hard, backward=∂m_soft
```

**Backward pass:** Gradient flows through `m_soft` (continuous). The `detach()` ensures the forward value is binary while the backward Jacobian is `∂(sigmoid)/∂(noisy/τ) × (1/τ)`.

**Temperature τ annealing:** 1.0 → 0.3 over training. τ=1.0 = maximum exploration (diverse selections). τ=0.3 = sharper selections (near binary). Floor of 0.3 prevents STE gradient starvation at very low temperature.

**Tau-independent STE correction:** At low τ, sigmoid saturates for |s|>1 → gradients vanish → TPN stops learning. Fix: multiply gathered visual tokens by an additive correction that is zero in the forward pass but non-zero in the backward:
```python
scores_at_selected = gather(s_N, idx)
v_selected = v_selected * (1 + scores_at_selected − scores_at_selected.detach())
# Forward: multiplies by 1. Backward: routes d(loss)/d(s_N[j]) = Σ_d g_v[j,d] × v_selected[j,d]
```
This gradient is tau-independent and remains non-zero throughout training.

---

### 4. Informative Token Merge

After selecting k=144 tokens, the 432 unselected tokens are not discarded — their information is absorbed into the selected tokens via parameter-free cosine-similarity-based merging:

```python
# 1. Separate selected and unselected
v_selected = gather(v_fine, idx)                       # [B, 144, 1024]
v_unsel, unsel_idx = complement_gather(v_fine, idx)   # [B, 432, 1024]

# 2. Soft assignment: each unselected token → nearest selected (by cosine sim)
sim = bmm(normalize(v_unsel), normalize(v_selected).transpose(1, 2))  # [B, 432, 144]
assignment = softmax(sim / τ_merge, dim=-1)            # τ_merge=0.1

# 3. TPN-score weighting: near-threshold unselected tokens contribute more
score_weights = softmax(gather(s_N, unsel_idx), dim=-1)  # [B, 432]
weighted = assignment * score_weights.unsqueeze(-1)
weighted = weighted / (weighted.sum(dim=1, keepdim=True) + 1e-8)

# 4. Aggregate and merge (β=0.3)
contributions = bmm(weighted.transpose(1, 2), v_unsel)  # [B, 144, 1024]
v_merged = v_selected + β * contributions               # [B, 144, 1024]
```

**Why this works better than alternatives:**

| Alternative | Problem |
|-------------|---------|
| Hard drop | Loses all information from 432 tokens. PruMerge-style merging recovers ~2% VQAv2 over pure pruning at the same budget. |
| Learned pooler (Phase 1.5) | Creates co-adaptation oscillation with LoRA — we trained 8 versions, all failed (see Architecture Evolution). Token merge has zero new parameters, so nothing can oscillate. |
| Random merge | Not query-conditioned. TPN scores weight the merge: for "what color is the car?", car-adjacent tokens get high scores → contribute strongly; distant background contributes weakly. |

**No train-eval mismatch:** Both training and evaluation receive exactly k=144 tokens. This was a critical bug in the Phase 1.5 pooler approach that corrupted all evaluation metrics.

---

### 5. Adaptive Budget

The fixed-k=144 model compresses every image equally. The Adaptive Budget extension learns *how many* tokens each sample needs.

**BudgetPredictor (~25K params):** A 2-layer MLP predicting per-sample token fraction:
```
ctx [B, 384] + normalize(entropy(s_N)) [B, 1]
  → Linear(385, 64) → GELU → Linear(64, 1) → sigmoid
  → k_frac ∈ [k_min/N, k_max/N] = [36/576, 288/576]
```
`ctx` is the TPN's attended visual context (encodes image-question relevance). Entropy of `s_N` indicates how concentrated the importance distribution is — low entropy (clustered) means a small region is clearly relevant → fewer tokens needed.

**Adaptive Gumbel-Sigmoid STE:** Extends fixed-k STE with a per-sample threshold:

1. Compute threshold from sorted (detached) scores at position `k_target = k_frac × 576`
2. **Differentiability trick:**
   ```python
   threshold = threshold_hard − (k_target − k_target.detach())
   ```
   - Forward: `threshold = threshold_hard` (zero effect)
   - Backward: `∂threshold/∂k_frac = −N` (non-zero gradient to BudgetPredictor)
3. Shifted soft mask: `m_soft = sigmoid((noisy − threshold) / τ)`
4. Differentiable count: `k_effective = sum(m_soft)` — used in budget loss
5. Hard selection: per-sample `topk(k_hard)`; shorter samples padded with `-1` in idx

**Budget loss:** `L_budget = λ_budget × mean(k_effective / N)` — penalizes using more tokens. Higher λ_budget → fewer average tokens → more compression.

**Gradient paths in adaptive mode:**

| Path | Route | Status |
|------|-------|--------|
| A: Budget Loss → BudgetPredictor | `mean(k_eff/N)` → threshold → k_frac → BP weights | Active — primary training signal |
| B: VQA Loss → TPN | STE additive correction | Active |
| C: VQA Loss → BudgetPredictor | No direct path (discrete k blocks it) | Blocked |
| D: Entropy Loss → BudgetPredictor | Entropy pushes k_frac up | Small leak (~5× dominated by Path A) |

---

## Architecture

### Fixed-k Mode (Phase 1 Final)

| Component | Params | Trainable | Input → Output |
|-----------|--------|-----------|----------------|
| DINOv2-S (coarse encoder) | 22M | ✗ | 98×98 RGB → [B, 49, 384] |
| CLIP ViT-B/32 (text encoder) | 63M | ✗ | question → [B, 512] |
| **TPN** | **~887K** | **✓** | [B,49,384]+[B,512] → idx[B,144] + s_N[B,576] + ctx[B,384] |
| CLIP ViT-L/14 (full encoder) | 304M | ✗ | 336×336 RGB → [B, 576, 1024] |
| **Informative Token Merge** | **0** | — | v_fine + idx + s_N → v_merged [B, 144, 1024] |
| MM Projector | ~7M | **✓** | [B, 144, 1024] → [B, 144, 4096] |
| LLaVA-1.5-7B + LoRA (r=64) | ~129M | **✓ (LoRA only)** | [B, 144, 4096] + text → text |
| **Total Trainable** | **~137M** | | |

LoRA: rank=64, alpha=128, targets: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj`.
FFN targets (gate/up) are included because token merging shifts the visual feature distribution — the FFN layers must adapt to the merged representation, not just the attention layers.

### Adaptive Budget Mode (Phase 1 Adaptive — additions only)

| Change | Detail |
|--------|--------|
| + BudgetPredictor | ~25K new params, added to TPN module |
| Token count k | Fixed 144 → variable [36, 288] per sample |
| idx tensor | [B, 144] → [B, k_max] with -1 padding for short samples |
| Attention mask | Per-sample variable-length mask |
| LoRA targets | 4 targets (q/k/v/o — gate/up removed for adaptive stability) |

### Full Forward Pass

```
images, questions → (3 frozen encoders) → v_coarse [B,49,384], q_embed [B,512], v_fine [B,576,1024]
                                         ↓
                               TPN → s_N [B,576], idx [B,k], ctx [B,384]
                                         ↓
     Fixed-k:  gather(v_fine, idx) → v_selected [B,k,1024]
               + informative_merge(v_fine, idx, s_N, β=0.3) → v_merged [B,k,1024]
                                         ↓
     Adaptive: BudgetPredictor(ctx, entropy(s_N)) → k_frac → adaptive_gumbel_sigmoid_ste
                                         ↓
               mm_projector → [B,k,4096] → LLaVA-1.5-7B+LoRA → answer tokens
```

---

## Training Phases

Training uses a three-phase curriculum where each phase adds one component on top of the previous foundation.

### Phase 0: TPN Warmup

| Setting | Value |
|---------|-------|
| What trains | TPN only — LoRA and MM projector frozen |
| Dataset | VQAv2 |
| Duration | ~10K steps |
| Loss | L_VQA only |
| Temperature | τ = 1.0 (fixed, maximum exploration) |
| Purpose | Cold-start TPN with meaningful scores before LoRA co-training begins |

**Why Phase 0 matters:** If LoRA trains alongside an untrained TPN, LoRA compensates for bad selections rather than forcing the TPN to improve. With LoRA frozen, the only way to reduce L_VQA is better selection quality. This gives TPN a foundation before joint training.

### Phase 1 Final: Joint Training with Token Merge

| Setting | Value |
|---------|-------|
| What trains | TPN + LoRA (r=64) + MM projector |
| Dataset | VQAv2 + GQA |
| Duration | 80K steps |
| LoRA warmfreeze | LoRA frozen steps 0–3000 (TPN stabilizes first) |
| Loss | L_VQA + 0.0×L_max + 0.02×(−H) + 0.1×L_div |
| Temperature | τ: 1.0 → 0.3 (linear) |
| Token merge | β=0.3, τ_merge=0.1 |
| LR (TPN / LoRA / MLP) | 3e-3 / 1e-4 / 5e-4 |
| Effective batch | 128 (4 × 16 grad steps × 2 GPUs) |
| Loads from | Phase 0 checkpoint |
| **Measured result** | **val_vqa loss = 0.3342 at step 80k** |

**Why LoRA warm-freeze?** In the first 3000 steps, LoRA receives gradients frozen at 0. This forces the VQA loss to decrease purely through improved TPN selection quality. After step 3000, LoRA unfreezes and jointly optimizes with TPN. Without this, LoRA immediately learns to compensate for the TPN's early mistakes — the TPN never learns to be selective.

### Phase 1 Adaptive: Variable Budget Training

| Setting | Value |
|---------|-------|
| What trains | BudgetPredictor (new) + TPN + LoRA + MM projector |
| Dataset | VQAv2 + GQA |
| Duration | 40K steps |
| BudgetPredictor warmup | Frozen steps 0–2000 (TPN/LoRA stabilize first) |
| LoRA warmfreeze | Frozen steps 0–1000 (shorter — TPN already trained) |
| Loss | L_VQA + 0.02×(−H) + 0.1×L_div + λ_budget×mean(k_eff/N) |
| Token range | k ∈ [36, 288] per sample |
| LR (TPN / LoRA / BP / MLP) | 1e-3 / 2e-4 / 1e-3 / 5e-4 |
| Loads from | Phase 1 Final checkpoint (`runs/phase1_final/phase1/best.ckpt`) |
| λ_budget | Sweep {0.05, 0.1, 0.2, 0.4} to trace Pareto curve |

---

## Loss Function

| Term | Weight | Purpose |
|------|--------|---------|
| L_VQA | 1.0 | Cross-entropy on answer tokens only (NTP on generative LLM) |
| −H (spatial entropy) | 0.02 | Prevent TPN from always selecting the same spatial region. H = Shannon entropy of batch-mean selection frequency. |
| L_div (query diversity) | 0.1 | Prevent TPN from ignoring the question. Penalizes similar selections for samples with dissimilar queries. |
| λ_budget × mean(k_eff/N) | sweep 0.05–0.4 | Budget penalty — Phase 1 Adaptive only. Sweep λ to trace the Pareto frontier. |
| L_max (sparsity) | 0.0 | L1 sparsity on scores — disabled; instance normalization makes L1 constant (mean=0 always). |

**val_vqa metric:** The logged `val_vqa` is cross-entropy loss on VQAv2 validation set — lower is better. It is not the same as VQA-Score % (which is higher-is-better and requires beam search decoding + answer normalization).

---

## Results

### Measured Training Metrics

| Run | val_vqa loss | Steps | Configuration |
|-----|-------------|-------|---------------|
| Phase 1 v3 (baseline) | 0.3334 | 50k | LoRA r=32, hard drop, no merge |
| **Phase 1 Final** | **0.3342** | 80k | LoRA r=64 + informative token merge |
| Phase 1 Adaptive | in progress | 40k | λ_budget=0.1 (running) |

Phase 1 Final essentially matches the Phase 1 v3 baseline. Token merge at β=0.3 did not clearly improve over hard drop at these compression ratios; it also did not hurt. The rank=64 LoRA upgrade provides capacity for the adaptive phase.

> val_vqa loss is a training proxy metric (lower = better). Full benchmark evaluation (VQA-Score %, GQA accuracy %, POPE F1) via `scripts/evaluate.py` is pending.

### Competitive Context (Published Results at ~25% Token Retention)

Results on LLaVA-1.5-7B, VQAv2 validation set, k≈144 tokens:

| Method | Tokens | VQAv2 | Trainable params | Core technique |
|--------|--------|-------|-----------------|----------------|
| LLaVA-1.5-7B (full) | 576 | 78.5% | — | Upper bound |
| SparseVLM | ~144+8 recycled | ~78% | Yes (staged) | Text-conditioned selection + cross-attn recycler |
| LLaVA-PruMerge | ~144 | ~77% | No | CLS-attention prune + cosine-sim merge into neighbors |
| FastV | ~144 | ~76–77% | No | LLM layer-2 attention pruning |
| CLS attention top-144 | 144 | 75–76% | No | Static ViT CLS attention ranking |
| Uniform grid 12×12 | 144 | ~74% | No | Every 4th token (no selection) |
| Random 144 | 144 | 72.5–73.5% | No | Uniform random |
| **TC-ViTok (k=144)** | **144** | **TBD** | **Yes (query-conditioned)** | **TPN + informative merge** |
| **TC-ViTok Adaptive** | **36–288** | **TBD** | **Yes** | **BudgetPredictor** |

The total range from random (73%) to best (78%) is only ~5 points. Token merging (PruMerge: +2% over pruning) is the single most impactful parameter-free technique. Query conditioning (SparseVLM: +1%) adds on top of that. TC-ViTok combines query-conditioned selection with merge; achieving 77%+ is the target.

### Planned Ablations

| Ablation | Tests |
|----------|-------|
| No token merge (hard drop) | Contribution of merge step |
| No STE tau-independent correction | Gradient path at low temperature |
| No LoRA warm-freeze | Training stability of joint training |
| Random selection baseline | TPN contribution vs. random |
| Fixed-k vs. Adaptive at same avg k | Whether adaptive Pareto-dominates fixed-k |
| No budget warmup | BudgetPredictor training stability |

---

## Architecture Evolution

### Phase 1 v3 → Phase 1.5 (Learnable Pooler — 8 versions, all failed)

After Phase 1 v3 (val_vqa=0.3334, LoRA r=32, hard drop), we attempted Phase 1.5: a cross-attention **UnselectedPooler** with 4 learned slot vectors (~512K params) to soft-recycle the 432 discarded tokens. This ran through 8 versions before the approach was abandoned.

**Three fatal design mistakes:**

**Mistake 1: Circular Co-Adaptation (structural, unfixable)**
```
Pooler output shifts → LLM input distribution changes → LoRA must re-adapt
→ LoRA gradient to pooler changes → Pooler must re-adapt → cycle repeats
```
This is GAN-style instability. Reducing pooler LR slows the oscillation period but does not eliminate it:

| Pooler LR | Oscillation period | Best val_vqa |
|-----------|-------------------|-------------|
| 5e-4 (v4–v7) | ~5k steps | 0.39–0.44 (cycling) |
| 2e-5 (v8) | ~15k steps | 0.3663 at step 45k |
| 0.0 (v8b, frozen) | None | Started at 0.4540 (optimizer catapult), unclear ceiling |

**Mistake 2: 9 Conflicting Loss Terms**  
At peak complexity, the TPN received gradients from: VQA, entropy, diversity, boundary, sparsity, spread, distillation, residual distillation, and slot orthogonality. With instance-normalized scores (finite gradient budget), auxiliary losses literally steal gradient from VQA.

**Mistake 3: Train-Eval Mismatch (critical, undetected)**  
During training, the LLM received k+4=148 tokens (4 pooler summaries + 144 selected). During evaluation (`generate()`), only 144 selected tokens were passed. The first 4 positions contained visual content instead of the summary tokens the model expected. Every Phase 1.5 validation metric was corrupted by this mismatch.

**Conclusion:** Token merge solves all three problems simultaneously: zero new parameters → no co-adaptation; no extra sequence positions → no train-eval mismatch; no additional loss terms.

### Phase 1.5 → Phase 1 Final (Token Merge + Rank 64)

After abandoning the pooler, we rebuilt Phase 1 with:
- **Informative Token Merge** (parameter-free, no co-adaptation, no mismatch)
- **LoRA rank 32 → 64** (2× capacity for more LoRA adaptation to merged tokens)
- **gate/up LoRA targets** added (FFN must adapt to merged feature distribution)
- **lr_lora 2e-4 → 1e-4** (rank-64 LoRA needs lower LR to avoid overshoot)

### Phase 1 Final → Phase 1 Adaptive (BudgetPredictor)

Phase 1 Final establishes stable TPN+LoRA foundation. The BudgetPredictor needs meaningful, converged TPN scores to predict sensible budgets — it cannot be trained from scratch alongside TPN. Loading from Phase 1 Final provides a warm start where TPN scores already encode question-conditioned importance.

---

## Repository Structure

```
TC-ViTok/
├── configs/
│   ├── base.yaml                  # Default hyperparameters (all keys with defaults)
│   ├── phase0.yaml                # TPN warmup (LLaVA frozen)
│   ├── phase1_final.yaml          # Joint training with token merge (80K steps)
│   └── phase1_adaptive.yaml       # Adaptive budget (loads from phase1_final)
├── src/tc_vitok/
│   ├── models/
│   │   ├── tpn.py                 # TPN (~887K) + BudgetPredictor (~25K)
│   │   ├── encoders.py            # DINOv2-S (img_size=98) + CLIP text/full
│   │   ├── masking.py             # gumbel_sigmoid_ste + adaptive_gumbel_sigmoid_ste
│   │   └── tc_vitok_model.py      # Full model: encode → select → merge → LLaVA
│   ├── training/
│   │   ├── losses.py              # VQA, entropy, diversity, budget losses
│   │   ├── curriculum.py          # tau and lambda schedules
│   │   ├── trainer.py             # DDP training loop, validation, checkpointing
│   │   └── optimizer.py           # Per-group AdamW (TPN/LoRA/MLP/BP separate LRs)
│   ├── data/
│   │   ├── datasets.py            # VQAv2, GQA datasets + retry loop for corrupt JPEGs
│   │   └── spatial_labels.py      # GQA spatial/global question classification
│   └── evaluation/
│       ├── metrics.py             # VQA-Score, GQA accuracy, POPE F1
│       ├── spatial_eval.py        # Spatial vs. global stratification
│       ├── collapse_detection.py  # Entropy + IoU collapse diagnostics
│       └── pareto.py              # Pareto curve plotting utilities
├── scripts/
│   ├── train.py                   # Training entry point (Phase 0/1/1-adaptive)
│   ├── evaluate.py                # Evaluation + Pareto sweep
│   ├── launch_multigpu.sh         # 2-GPU DDP launcher (sets all NCCL/HF env vars)
│   ├── cache_features.py          # Pre-compute DINOv2 + CLIP text → HDF5
│   ├── visualize_masks.py         # Heatmap of TPN token selections
│   └── diagnose_routing.py        # Verify TPN routes differently per question
├── tests/
│   ├── test_tpn.py                # TPN architecture + gradient flow
│   ├── test_masking.py            # STE correctness + tau-independence
│   └── test_adaptive_budget.py    # 14 tests: BudgetPredictor + adaptive STE
└── documentation/                 # Extended design docs (see below)
    ├── solutions.md               # Architecture decisions + Phase 1.5 post-mortem
    ├── understanding.md           # Deep-dive: full forward pass with tensor shapes
    ├── innovations.md             # SOTA literature review + novelty analysis
    ├── ADAPTIVE_BUDGET.md         # Adaptive budget design rationale
    ├── computations.md            # VRAM and FLOP breakdowns
    └── run.md                     # Full setup + dataset download guide
```

---

## Setup and Installation

### Requirements

- Python 3.11 (3.12 works; 3.14 does not have PyTorch packages)
- **PyTorch ≥ 2.6.0** — MANDATORY due to CVE-2025-32434 (transformers ≥ 4.40 requires `weights_only=True` in `torch.load`)
- CUDA 12.1+ GPU with ≥ 24 GB VRAM for actual training (2× RTX 6000 Ada recommended)
- Apple Silicon with MPS supported for development/testing

### Installation

```bash
git clone https://github.com/whyeventry0103/TC-ViTok.git
cd TC-ViTok

conda create -n tcvitok python=3.11 -y
conda activate tcvitok

# CUDA (for training on server)
pip install torch>=2.6.0 torchvision>=0.21.0 --index-url https://download.pytorch.org/whl/cu121

# Apple Silicon (for development)
pip install torch>=2.6.0 torchvision>=0.21.0

pip install -r requirements.txt
pip install -e .
pip install peft h5py   # LoRA + feature caching (verify installed)
```

### Model Downloads

Pre-download models to avoid HuggingFace HTTP timeouts during DDP training:

```bash
# LLaVA-1.5-7B (includes CLIP ViT-L/14 and mm_projector) — ~13.5 GB
huggingface-cli download llava-hf/llava-1.5-7b-hf

# CLIP ViT-L/14 standalone (needed for feature extraction) — ~890 MB
python -c "from transformers import CLIPModel; CLIPModel.from_pretrained('openai/clip-vit-large-patch14-336')"

# DINOv2-S and CLIP ViT-B/32 download automatically on first run via timm/transformers
```

### Dataset Setup

```bash
export DATA_ROOT=~/tc-vitok-data
mkdir -p $DATA_ROOT/{vqav2,gqa,pope,coco}

# VQAv2 + COCO images (~25 GB total)
cd $DATA_ROOT/coco
wget http://images.cocodataset.org/zips/train2014.zip && unzip train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip && unzip val2014.zip

cd $DATA_ROOT/vqav2
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip && unzip *.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip && unzip *.zip

# Symlinks (required for evaluate.py)
ln -s ../coco/train2014 $DATA_ROOT/vqav2/train2014
ln -s ../coco/val2014 $DATA_ROOT/vqav2/val2014

# GQA (~20 GB)
cd $DATA_ROOT/gqa
wget https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip && unzip images.zip
wget https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip && unzip questions1.2.zip
```

### Verify Installation

```bash
# Core tests (no model downloads needed)
PYTHONPATH=src python -m pytest tests/ -q   # expected: all pass

# Dry run (StubLLM + dummy data, ~30 seconds)
PYTHONPATH=src python scripts/train.py \
    --config configs/phase0.yaml --output /tmp/tc_vitok_test \
    --stub --dummy-data --no-wandb --max-steps 2
# Expected: /tmp/tc_vitok_test/phase0/best.ckpt created
```

---

## Running Experiments

### Multi-GPU Training (2× GPU, recommended)

```bash
# Phase 0: TPN warmup
tmux new-session -d -s phase0
tmux send-keys -t phase0 "cd ~/TC-ViTok && bash scripts/launch_multigpu.sh \
    --config configs/phase0.yaml --output runs/phase0" Enter

# Phase 1 Final: token merge + LoRA r=64 (loads phase0 checkpoint automatically)
tmux new-session -d -s phase1_final
tmux send-keys -t phase1_final "cd ~/TC-ViTok && bash scripts/launch_multigpu.sh \
    --config configs/phase1_final.yaml --output runs/phase1_final" Enter

# Phase 1 Adaptive: variable budget (loads phase1_final checkpoint automatically)
tmux new-session -d -s phase1_adaptive
tmux send-keys -t phase1_adaptive "cd ~/TC-ViTok && bash scripts/launch_multigpu.sh \
    --config configs/phase1_adaptive.yaml --output runs/phase1_adaptive" Enter
```

### Single-GPU Training

```bash
export DATA_ROOT=~/tc-vitok-data
PYTHONPATH=src python scripts/train.py --config configs/phase0.yaml --output runs/phase0
PYTHONPATH=src python scripts/train.py --config configs/phase1_final.yaml --output runs/phase1_final
PYTHONPATH=src python scripts/train.py --config configs/phase1_adaptive.yaml --output runs/phase1_adaptive
```

### Pareto Sweep (Adaptive Budget)

```bash
for lb in 0.05 0.1 0.2 0.4; do
    tmux new-session -d -s adaptive_lb${lb}
    tmux send-keys -t adaptive_lb${lb} "cd ~/TC-ViTok && bash scripts/launch_multigpu.sh \
        --config configs/phase1_adaptive.yaml \
        --output runs/adaptive_lb${lb} \
        training.lambda_budget=${lb}" Enter
done
```

### Monitoring

```bash
tmux attach -t phase1_final             # reattach to running session
tail -f logs/train_*.log | tr '\r' '\n' | grep "Step"   # live metrics
grep "val" logs/train_*.log             # validation checkpoints only
nvidia-smi                              # GPU utilization
ls -lh runs/phase1_final/phase1/        # checkpoint list
```

---

## Evaluation

```bash
# Phase 1 Final: VQAv2 + GQA + POPE
python scripts/evaluate.py \
    --checkpoint runs/phase1_final/phase1/best.ckpt \
    --config configs/phase1_final.yaml \
    --benchmarks vqav2 gqa pope \
    --data-root ~/tc-vitok-data \
    --output runs/eval_results/phase1_final.json

# Phase 1 Adaptive: same benchmarks
python scripts/evaluate.py \
    --checkpoint runs/phase1_adaptive/phase1/best.ckpt \
    --config configs/phase1_adaptive.yaml \
    --benchmarks vqav2 gqa pope \
    --data-root ~/tc-vitok-data \
    --output runs/eval_results/phase1_adaptive.json

# Pareto sweep: one model, multiple k values
python scripts/evaluate.py \
    --checkpoint runs/phase1_final/phase1/best.ckpt \
    --token-budgets 72 144 216 288 576 \
    --benchmarks vqav2 gqa \
    --output runs/eval_results/pareto_sweep.json

# GQA with spatial stratification (spatial vs. global reasoning)
python scripts/evaluate.py \
    --checkpoint runs/phase1_final/phase1/best.ckpt \
    --benchmarks gqa --spatial-eval \
    --output runs/eval_results/gqa_spatial.json

# Random-drop baseline (ablation)
python scripts/evaluate.py \
    --checkpoint runs/phase1_final/phase1/best.ckpt \
    --baseline random_drop --benchmarks vqav2 gqa \
    --output runs/eval_results/random_drop.json
```

---

## Known Failure Modes and Mitigations

| Failure | Symptom | Fix |
|---------|---------|-----|
| **Spatial collapse** | All selected tokens from the same image region; entropy < 5.0 in logs | Increase `lambda_entropy` from 0.02 to 0.05 |
| **Query-ignoring** | Same tokens selected regardless of question; IoU > 0.9 across batch | Increase `lambda_diversity`; verify STE additive correction is active |
| **STE gradient vanishing** | TPN not learning after step 10k; loss plateau | Verify tau-independent correction: `v_selected *= (1 + sc - sc.detach())`; increase `tau_min` from 0.3 to 0.5 |
| **Score magnitude explosion** | Loss NaN; select entropy = log(k) (always same tokens) | Verify instance normalization in TPN output; check `s_N.mean()` ≈ 0 in logs |
| **Budget collapse** | `k_effective` stuck at k_min or k_max throughout training | Verify `budget_warmup_steps > 0`; adjust `lambda_budget`; check Path A gradient |
| **Co-adaptation oscillation** | val_vqa oscillates ±0.04 every ~10k steps | Never train a learnable pooler simultaneously with LoRA; use token merge instead |
| **Cross-phase checkpoint mismatch** | `RuntimeError: size mismatch for lora_A` | Trainer filters shape-mismatched keys before `load_state_dict` — verify `strict=False` |
| **DDP rank desync** | NCCL timeout; SIGABRT instead of Python traceback | `HF_HUB_OFFLINE=1` must be set (use `launch_multigpu.sh`); all HF downloads before training |

---

## Key Design Decisions

**Why two phases (Phase 0 + Phase 1)?** Phase 0 cold-starts TPN with meaningful importance scores before joint training. If LoRA trains from step 0, it compensates for bad selections and TPN never learns selectivity. Phase 0 establishes the selection foundation; Phase 1 co-optimizes everything.

**Why Phase 1 Adaptive needs Phase 1 Final first?** BudgetPredictor predicts how many tokens a sample needs based on TPN scores. If TPN scores are not yet meaningful, the predictor receives noise → learns a useless function. Loading from Phase 1 Final gives BudgetPredictor access to stable, question-conditioned scores from step 0.

**Why token merge instead of a learned pooler?** Eight training runs with a cross-attention pooler (Phase 1.5) all failed due to circular co-adaptation with LoRA. Token merge has zero new parameters — nothing can oscillate. It also eliminates the train-eval mismatch that corrupted all Phase 1.5 evaluation metrics.

**Why instance normalization on TPN scores?** Without it, Adam pushes scores to ±100 → Gumbel noise becomes negligible → deterministic fixed selection → entropy collapse. `tanh` saturates too (gradient ~0 at ±100). Instance norm anchors scores to mean=0, std=1 permanently with no learnable parameters that Adam can optimize away.

---

## Citation

```bibtex
@article{shah2025tcvitok,
  title     = {TC-ViTok: Task-Conditioned Visual Tokenization with Adaptive Budget
               for Efficient Vision-Language Reasoning},
  author    = {Shah, Muhammad Ali and Ul Haq, Talha Hassan},
  institution = {Lahore University of Management Sciences},
  year      = {2025}
}
```

**Authors:** Muhammad Ali Shah (27100172), Talha Hassan Ul Haq (27100306)  
**Institution:** Lahore University of Management Sciences (LUMS)

---

*MIT License — see [LICENSE](LICENSE) for details.*
