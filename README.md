# TC-ViTok: Task-Conditioned Visual Tokenization with Adaptive Budget

**Learning a Query-Conditioned Token Selection Policy for Efficient Vision-Language Reasoning**

![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Institution: LUMS](https://img.shields.io/badge/Institution-LUMS-red.svg)

Vision-Language Models like LLaVA encode a fixed 576 visual tokens for every image, regardless of query complexity. TC-ViTok introduces a lightweight **Tokenization Policy Network** (TPN, ~887K params) that learns a query-conditioned selection policy, dynamically choosing only k << 576 relevant tokens per sample. The **Fixed-k** variant (k=144) reduces LLM-side prefill cost by 75% while preserving unselected-token information through a parameter-free **informative token merge**. The **Adaptive Budget** extension adds a **BudgetPredictor** (~25K params) that predicts a per-sample variable budget (36–288 tokens), allowing the model to allocate more capacity to hard questions (spatial reasoning, counting, OCR) and less to easy ones (yes/no, color). By sweeping the budget penalty weight `lambda_budget`, we trace a Pareto curve of compression vs. accuracy. Both modes use Gumbel-Sigmoid straight-through estimation for end-to-end differentiable discrete selection.

> **Note on compression scope:** The full CLIP ViT-L/14 encoder still processes all 576 tokens. Savings accrue at the LLM prefill stage, where the LLM sees only k + L_text tokens instead of 576 + L_text.

---

## Key Contributions

- **Base TC-ViTok:** A query-conditioned TPN (~887K params) learns to select k of 576 CLIP ViT-L/14 tokens via Gumbel-Sigmoid STE. Per-sample instance normalization on TPN scores prevents score saturation and gradient vanishing throughout training.
- **Fixed-k with Informative Token Merge:** Unselected tokens (N-k = 432) are not discarded but merged back into the k selected tokens using a parameter-free cosine-similarity assignment weighted by TPN importance scores (beta=0.3). This recovers background information at zero parameter cost.
- **Adaptive Budget:** A 2-layer BudgetPredictor MLP takes the TPN cross-attention context vector and score entropy as input, predicts a per-sample token fraction k_frac ∈ [k_min/N, k_max/N], and uses a differentiable threshold-shift trick to propagate gradients from the budget penalty loss back to the predictor. LoRA targets are narrowed to q/k/v/o (4 modules) for stability.

---

## Method

### 1. Core Idea: Visual Tokenization as Policy Learning

Standard VLMs pass all N=576 spatial tokens from CLIP ViT-L/14 to the LLM. TC-ViTok reframes this as a learned routing problem:

```
π: (Query Q, Image I_coarse) → Token Selection Indices S ⊆ {0, ..., N-1}
```

The TPN receives a low-resolution coarse image (DINOv2-S, 49 tokens) and the CLIP text embedding of the query. It produces per-token importance scores over the 576 spatial positions of the full CLIP encoder, then selects the top-k positions. This makes token selection conditioned on *what is being asked*, not just what is visually salient.

### 2. Token Selection: Gumbel-Sigmoid STE

Discrete top-k selection is non-differentiable. We use the Gumbel-Sigmoid straight-through estimator (STE):

```
u ~ Uniform(0, 1),  eps = 1e-8
g = -log(-log(clamp(u, eps, 1-eps)))          # Gumbel(0, 1)
noisy = s_N + g                                # noisy scores [B, N]
m_soft = sigmoid(noisy / tau)                  # soft mask [B, N]
idx    = topk(noisy, k).indices                # hard selection on noisy scores
m_hard = scatter(idx, 1.0)                     # binary mask [B, N]
mask_ste = m_hard - m_soft.detach() + m_soft   # STE: forward=m_hard, grad via m_soft
```

- **Forward:** exact hard top-k selection (binary mask, real compression via `torch.gather`).
- **Backward:** gradient flows through `m_soft`, reaching all N TPN scores.
- **Temperature annealing:** tau decays linearly from `tau_max` to `tau_min` over training. High tau = soft exploration; low tau = sharp exploitation. Phase 1 Final uses tau_min=0.3 (not 0.1) to avoid STE gradient starvation.
- **Instance normalization:** TPN scores are normalized to zero-mean, unit-std per sample before STE. This prevents Adam from pushing scores to O(100), which would overwhelm Gumbel noise and collapse selection to a static crop.

### 3. Informative Token Merge (Fixed-k Extension)

After TPN selects k=144 tokens, the remaining N-k=432 unselected tokens are merged back rather than discarded:

```python
# v_fine: CLIP features [B, N, D]; idx: selected indices [B, k]; s_N: TPN scores [B, N]
v_selected = gather(v_fine, idx)                      # [B, k, D]
v_unsel, unsel_idx = complement_gather(v_fine, idx)   # [B, N-k, D]

sim = bmm(normalize(v_unsel), normalize(v_selected).T)   # [B, N-k, k] cosine sim
assignment = softmax(sim / tau_merge, dim=-1)             # [B, N-k, k], tau_merge=0.1

unsel_scores = s_N[unsel_idx]                            # [B, N-k]
score_weights = softmax(unsel_scores, dim=-1)            # near-threshold tokens weight more

weighted_assignment = (assignment * score_weights.unsqueeze(-1))
weighted_assignment /= weighted_assignment.sum(dim=1, keepdim=True)

contributions = bmm(weighted_assignment.T, v_unsel)      # [B, k, D]
v_merged = v_selected + beta * contributions             # beta=0.3
```

Key properties:
- **Zero new parameters.** Merge is fully determined by cosine similarities and TPN scores.
- **Query-conditioned weighting:** near-threshold unselected tokens (high TPN score) contribute more than low-scored background tokens, so the merge reflects the query's relevance signal.
- **Residual form:** `v_merged = v_selected + beta * contributions` preserves the original token and adds enrichment, keeping the LLM's input distribution close to what it was trained on.
- This replaced the failed UnselectedPooler (Phase 1.5, 8 versions) which suffered from co-adaptation oscillation between the pooler and LoRA.

### 4. Adaptive Budget (Major Extension)

The fixed-k assumption treats all questions equally. The Adaptive Budget extension replaces the fixed k with a per-sample predicted budget.

**BudgetPredictor architecture:**
```
ctx [B, 384]  →  Linear(384, 64) + GELU  →  h [B, 64]
                                             + entropy_proj(score_entropy [B, 1])
                                          →  Linear(64, 1) + sigmoid
                                          →  k_frac = k_min/N + out * (k_max - k_min)/N
```
- Input: TPN cross-attention context vector `ctx` [B, 384] + score entropy (stop-gradient complexity proxy).
- Output: per-sample `k_frac` ∈ [k_min/N, k_max/N] = [36/576, 288/576].
- Parameters: ~25K.

**Adaptive STE — differentiable threshold:**

```python
k_target = k_frac * N                              # continuous [B]
k_hard   = round(k_target).clamp(k_min, k_max)    # integer [B]

# Threshold from sorted scores (stop-grad)
threshold_hard = (scores_sorted[b, k_hard-1] + scores_sorted[b, k_hard]) / 2  # [B]

# Differentiable trick: zero forward effect, d(threshold)/d(k_frac) = -N
threshold = threshold_hard - (k_target - k_target.detach())   # [B]

shifted = noisy - threshold.unsqueeze(1)            # [B, N]
m_soft  = sigmoid(shifted / tau)                    # [B, N]
k_effective = m_soft.sum(dim=-1)                    # [B] — differentiable budget
```

**Gradient paths in adaptive mode:**

| Path | Direction | Mechanism |
|------|-----------|-----------|
| A (primary) | Budget loss → BudgetPredictor | `L_budget = mean(k_eff/N)` → threshold → k_frac → BP weights |
| B | VQA loss → TPN scores | STE correction `v*(1 + s - s.detach())` routes grad to TPN |
| C | VQA loss → BudgetPredictor | Blocked; BP adapts indirectly through evolving ctx |
| D (minor leak) | Entropy loss → BudgetPredictor | Pushes k_frac up; dominated ~5x by path A |

**Variable-length batching:** idx tensor is padded to `k_max` width, with -1 marking invalid positions. Attention masking in the LLM handles variable-length visual prefixes within a batch.

---

## Architecture

### Fixed-k Mode

| Component | Parameters | Trainable | Input | Output |
|-----------|-----------|-----------|-------|--------|
| DINOv2-S (coarse encoder) | 22M | Frozen | 98×98×3 | [B, 49, 384] |
| CLIP ViT-B/32 (text encoder) | 63M | Frozen | text tokens | [B, 512] |
| **TPN** | **~887K** | **Yes** | [B,49,384] + [B,512] | idx [B,k], s_N [B,576], ctx [B,384] |
| CLIP ViT-L/14 (full encoder) | 304M | Frozen | 336×336×3 | [B, 576, 1024] |
| Informative Token Merge | 0 | — | [B,576,1024], idx, s_N | [B, k, 1024] |
| LLaVA-1.5 backbone | 7B | Frozen | — | — |
| LoRA (rank 64, alpha 128) | ~35M | **Yes** | — | text logits |
| Vision projector (MLP) | ~4M | **Yes** | [B, k, 1024] | [B, k, 4096] |
| **Total trainable** | **~36M** | — | — | — |

LoRA targets in Phase 1 Final: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj` (6 modules). The MLP target set (gate/up) was added to help the LLM adapt its FFN layers to the shifted visual distribution from token merge.

### Adaptive Budget Mode

Changes relative to Fixed-k:

| Change | Details |
|--------|---------|
| **BudgetPredictor added** | ~25K params; takes ctx [B,384] + score entropy → k_frac [B] |
| **k becomes per-sample variable** | k ∈ [36, 288] per sample (vs. fixed 144) |
| **Adaptive STE** | threshold-shift trick makes k_effective differentiable |
| **idx padded** | shape [B, k_max] with -1 for invalid positions |
| **Attention masking** | LLM uses per-sample attention mask for variable-length visual prefixes |
| **LoRA targets reduced** | q/k/v/o only (4 modules, no gate/up) for stability |
| **lr_budget = 1e-3** | Separate learning rate for BudgetPredictor |

---

## Training

### Phase 0: TPN Warmup

| Item | Value |
|------|-------|
| What trains | TPN only |
| Frozen | LLaVA backbone, LoRA, projector |
| Data | VQAv2 (100K subsample) |
| Steps | ~10,000 |
| Loss | L_VQA + lambda_entropy * L_ent + lambda_diversity * L_div |
| tau | 1.0 fixed (no annealing) |
| lr_tpn | 1e-3 |

Purpose: give TPN a head start at query-conditioned routing before LoRA is introduced, so LoRA does not compensate for poor token selection.

### Phase 1 Final: Joint Training with Token Merge

| Item | Value |
|------|-------|
| What trains | TPN + LoRA (rank 64/alpha 128) + vision projector |
| Frozen | LLaVA 7B weights, CLIP encoders, DINOv2 |
| Data | VQAv2 + GQA (full training sets) |
| Steps | 80,000 |
| Loss | L_VQA + 0.02 * L_ent + 0.1 * L_div |
| tau schedule | 1.0 → 0.3 (linear; tau_min=0.3 to avoid gradient starvation) |
| LoRA warm-freeze | LoRA gradients zeroed for first 3,000 steps |
| lr_tpn / lr_lora / lr_mlp | 3e-3 / 1e-4 / 5e-4 |
| Effective batch | 128 (4 samples × 16 grad accum × 2 GPUs) |
| Load from | Phase 0 best checkpoint |

Note: lambda_max=0.0 — L1 sparsity is disabled because instance normalization makes `mean(|s_N|) ≈ 0.798` constant, providing no useful gradient signal.

### Phase 1 Adaptive: Variable Budget Training

| Item | Value |
|------|-------|
| What trains | TPN + LoRA (rank 64/alpha 128) + projector + BudgetPredictor |
| Frozen | LLaVA 7B, CLIP encoders, DINOv2 |
| Data | VQAv2 + GQA |
| Steps | 40,000 |
| Loss | L_VQA + 0.02 * L_ent + 0.1 * L_div + lambda_budget * L_budget |
| lambda_budget | Sweep: {0.05, 0.1, 0.2, 0.4} to trace Pareto curve |
| tau schedule | 1.0 → 0.3 (linear) |
| Budget warmup | BudgetPredictor frozen for first 2,000 steps |
| LoRA warm-freeze | LoRA frozen for first 1,000 steps (shorter than Phase 1 Final; TPN already trained) |
| lr_tpn / lr_lora / lr_mlp / lr_budget | 1e-3 / 2e-4 / 5e-4 / 1e-3 |
| Load from | Phase 1 Final best checkpoint |

The budget warmup prevents the BudgetPredictor from learning a trivially low budget before TPN and LoRA have stabilized, which would cause premature accuracy collapse.

---

## Loss Function

| Term | Active In | Weight | Purpose |
|------|-----------|--------|---------|
| `L_VQA` | All phases | 1.0 | Cross-entropy on generated answer tokens |
| `L_ent = -H` | All phases | 0.02 (Ph1) / 0.01 (Ph0) | Maximize Shannon entropy of batch-mean spatial selection; prevents spatial collapse |
| `L_div` | All phases | 0.1 (Ph1) / 0.02 (Ph0) | Hinge on mask similarity for pairs with dissimilar queries; prevents query-ignoring |
| `L_budget` | Adaptive only | lambda_budget (sweep) | `mean(k_eff/N)` — penalizes token fraction; gradient to BudgetPredictor |
| `L_sparsity` | Disabled (=0) | 0.0 | L1 on TPN scores; disabled because instance norm makes it constant |

Entropy is computed via the STE mask (`mask_ste`) during training so gradients reach all N=576 token positions, not just the selected k. At eval or when mask_ste is unavailable, entropy is computed from the idx tensor directly.

---

## Results

### Fixed-k Baseline (Phase 1 Final, k=144)

| Metric | Value |
|--------|-------|
| Steps | 80,000 |
| Val VQA loss (final) | **0.3342** |
| Compression (LLM tokens) | 75% (144 of 576) |
| Full VQA-Score eval | Pending |
| Full GQA accuracy eval | Pending |

The val_vqa=0.3342 is the actual measured training-time validation metric (VQA cross-entropy loss on held-out samples). Full benchmark evaluation (VQA-Score %, GQA accuracy %) has not yet been run.

### Adaptive Budget (Phase 1 Adaptive)

Training in progress. Results pending.

Planned: sweep lambda_budget ∈ {0.05, 0.1, 0.2, 0.4} and report (mean_k, val_vqa) for each, tracing the compression–accuracy Pareto frontier.

### Ablations (Planned)

| Ablation | Question |
|----------|---------|
| No token merge (beta=0) vs. beta=0.3 | How much does informative merge help over hard drop? |
| Fixed k=144 vs. adaptive budget | Does per-sample variable k beat fixed k at same mean budget? |
| LoRA targets: 4 vs. 6 (with gate/up) | Is MLP adaptation needed after token merge? |
| Orthogonal deflation (Phase 1.5 legacy) | Does it help as a regularizer? |
| tau_min=0.1 vs. 0.3 | What is the STE gradient starvation threshold? |

---

## Repository Structure

```
TC-ViTok/
├── README.md
├── setup.py
├── train.py                          # Top-level training entry point
├── evaluate.py                       # Evaluation entry point
├── launch_multigpu.sh                # Multi-GPU DDP launcher
├── visualize_masks.py                # TPN selection visualization
├── diagnose_routing.py               # Routing collapse diagnostics
│
├── configs/
│   ├── base.yaml                     # Default hyperparameters
│   ├── phase0.yaml                   # Phase 0: TPN warmup
│   ├── phase1.yaml                   # Phase 1: basic joint finetuning
│   ├── phase1_final.yaml             # Phase 1 Final: token merge + rank 64 LoRA
│   ├── phase1_adaptive.yaml          # Phase 1 Adaptive: variable budget [NEW]
│   └── phase2.yaml                   # Phase 2: domain transfer (optional)
│
├── src/tc_vitok/
│   ├── models/
│   │   ├── tpn.py                    # TokenizationPolicyNetwork + BudgetPredictor
│   │   ├── encoders.py               # DINOv2-S (coarse) + CLIP text/full encoders
│   │   ├── masking.py                # gumbel_sigmoid_ste, adaptive_gumbel_sigmoid_ste
│   │   └── tc_vitok_model.py         # Full model: gather_tokens, informative_token_merge
│   │
│   ├── training/
│   │   ├── losses.py                 # compute_total_loss + all individual loss terms
│   │   ├── curriculum.py             # get_tau, get_lambda, get_lr_schedule
│   │   ├── trainer.py                # TCViTokTrainer: multi-phase training loop
│   │   └── optimizer.py              # AdamW with per-group learning rates
│   │
│   ├── data/
│   │   ├── datasets.py               # VQAv2, GQA, TextVQA, POPE loaders
│   │   ├── caching.py                # HDF5 feature cache (coarse ViT + CLIP text)
│   │   └── spatial_labels.py         # Spatial specificity classifier (GQA annotations)
│   │
│   └── evaluation/
│       ├── metrics.py                # VQA-Score, GQA accuracy, POPE F1
│       ├── spatial_eval.py           # Spatial vs. global question stratification
│       ├── collapse_detection.py     # Spatial collapse & query-ignoring diagnostics
│       └── pareto.py                 # Pareto curve analysis for adaptive budget
│
└── tests/                            # Unit and integration tests
```

---

## Setup and Installation

```bash
# 1. Create environment
conda create -n tc-vitok python=3.10
conda activate tc-vitok

# 2. Install package
pip install -e .

# 3. Download model weights (LLaVA-1.5-7B, DINOv2-S, CLIP ViT-L/14)
#    Models are fetched from HuggingFace Hub on first run.
#    Pre-download to avoid rank desync during DDP training:
HF_HUB_OFFLINE=0 python -c "
from transformers import AutoModelForCausalLM
import timm, open_clip
timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=True)
"

# 4. Download datasets
#    VQAv2: https://visualqa.org/download.html  → ~/tc-vitok-data/vqav2/
#    GQA:   https://cs.stanford.edu/people/dorarad/gqa/download.html → ~/tc-vitok-data/gqa/
```

---

## Running Experiments

### Standard Fixed-k Pipeline

```bash
# Phase 0: TPN warmup (~10K steps, TPN only)
bash launch_multigpu.sh \
  --config configs/phase0.yaml \
  --output runs/phase0

# Phase 1 Final: joint training with token merge (80K steps)
bash launch_multigpu.sh \
  --config configs/phase1_final.yaml \
  --output runs/phase1_final
```

### Adaptive Budget Pipeline

```bash
# Phase 1 Adaptive: variable budget training (40K steps, loads from Phase 1 Final)
# Sweep lambda_budget to trace the Pareto curve
for lam in 0.05 0.1 0.2 0.4; do
  bash launch_multigpu.sh \
    --config configs/phase1_adaptive.yaml \
    --output runs/phase1_adaptive_lam${lam} \
    training.lambda_budget=${lam}
done
```

### Evaluation

```bash
# Evaluate a checkpoint
python evaluate.py \
  --checkpoint runs/phase1_final/phase1/best.ckpt \
  --config configs/phase1_final.yaml \
  --datasets vqav2 gqa \
  --output results/phase1_final/

# Pareto analysis across adaptive budget runs
python -c "
from src.tc_vitok.evaluation.pareto import plot_pareto
plot_pareto('results/adaptive/', output='pareto_curve.png')
"
```

---

## Known Failure Modes and Mitigations

| Failure Mode | Symptom | Root Cause | Fix |
|---|---|---|---|
| **Spatial collapse** | Same spatial region selected regardless of query | Entropy = log(k) instead of log(N); all samples select same tokens | Entropy regularization (lambda_entropy=0.02) + Gumbel noise on noisy scores (not raw scores) |
| **Score saturation** | TPN scores reach O(100), static crop from step 1000 | Adam pushes scores to large values, overwhelming Gumbel noise | Per-sample instance normalization on s_N — enforces zero-mean, unit-std; cannot be saturated |
| **Query-ignoring** | Different questions → same token selection | TPN learns image-only saliency; text input ignored | Diversity loss + LoRA warm-freeze (3K steps): forces TPN to route VQA signal before LoRA compensates |
| **Budget collapse** | BudgetPredictor immediately outputs k_min | Budget loss gradient before TPN/LoRA are stable | Budget warmup: BudgetPredictor frozen for first 2K steps |
| **Co-adaptation oscillation** | val_vqa oscillates on ~10K step period | Mutual dependency between pooler and LoRA updates | Removed pooler entirely; replaced with parameter-free informative token merge |
| **Gradient vanishing through STE** | mask.grad_fn is None; TPN receives no VQA signal | STE not implemented correctly; mask disconnected from soft path | `mask_ste = m_hard - m_soft.detach() + m_soft`; assert grad_fn is not None at train time |

---

## Citation

```bibtex
@article{tcvitok2026,
  title     = {TC-ViTok: Task-Conditioned Visual Tokenization with Adaptive Budget},
  author    = {Shah, Muhammad Ali},
  year      = {2026},
  institution = {Lahore University of Management Sciences (LUMS)},
  note      = {Preprint}
}
```
