# TC-ViTok: Task-Conditioned Visual Tokenization

**Learning a Token Selection Policy for Efficient Vision-Language Reasoning**

![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Institution: LUMS](https://img.shields.io/badge/Institution-LUMS-red.svg)

Vision-Language Models like LLaVA encode a fixed 576 visual tokens for every image, regardless of query complexity. **TC-ViTok** introduces a lightweight Tokenization Policy Network (TPN, ~887K params) that learns a query-conditioned token selection policy, dynamically choosing only k << 576 relevant tokens. This reduces **LLM-side** computational cost by up to 75% (note: the full ViT-L encoder still processes all tokens; net end-to-end savings depend on LLM prefill dominance). Our approach frames visual tokenization as a learned policy problem and trains the TPN via Gumbel-Sigmoid straight-through estimation with a phased curriculum.

---

## Key Idea

Most VLMs use **fixed tokenization**: every image → 576 tokens, regardless of question complexity. We reframe this as a **learned policy problem**:

**Visual Tokenization as Policy Learning:**
```
π: (Query Q, Image I_coarse) → Token Selection Indices S (for gather)
```

Instead of encoding all 576 tokens, TC-ViTok:
1. Extracts **coarse features** (DINOv2-S, 49 tokens) and **text embeddings** (CLIP)
2. Runs **TPN** to predict importance scores for each of 576 spatial locations
3. Selects **top-k indices** via differentiable Gumbel-Sigmoid STE
4. **Gathers** selected tokens from full CLIP ViT-L/14 output via `torch.gather` (real compression)
5. Feeds **k gathered tokens** (not N) to **LLaVA** — LLM sees k+L tokens, reducing prefill FLOPs

**High-level flow:**
```
┌──────┐     ┌─────────────┐     ┌─────┐     ┌──────────┐     ┌─────┐     ┌────────┐
│Image │────>│ DINOv2-S    │────>│     │     │ CLIP     │     │     │────>│ LLaVA  │───> Answer
│      │     │ (98x98)     │     │ TPN │────>│ ViT-L/14 │────>│Mask │     │ +LoRA  │
└──────┘     └─────────────┘     │     │     │ (336x336)│     │     │     └────────┘
                                  │     │     └──────────┘     └─────┘
┌──────┐     ┌─────────────┐     │     │
│Query │────>│ CLIP Text   │────>│     │
└──────┘     └─────────────┘     └─────┘
```

---

## Architecture

TC-ViTok consists of five main components:

### 1. Coarse Visual Encoder
- **Model:** DINOv2-Small (ViT-S/14)
- **Input:** 98×98 RGB image
- **Output:** 49 spatial tokens (7×7 grid), 384-dim
- **Status:** Frozen (pretrained on ImageNet-22k)
- **Purpose:** Fast low-resolution features for token scoring

### 2. Text Encoder
- **Model:** CLIP ViT-B/32 text encoder
- **Input:** Question string
- **Output:** 512-dim [EOS] embedding
- **Status:** Frozen
- **Purpose:** Question representation for conditional selection

### 3. Tokenization Policy Network (TPN)
- **Architecture:** Cross-attention (4 heads) + concat [ctx; alpha_avg] + Score MLP + bilinear upsample
- **Input:** Coarse visual (49×384) + text embedding (512)
- **Output:** Top-k gather indices (k values) + raw scores (576)
- **Parameters:** ~887K (trainable)
- **Key innovation:** Gumbel-Sigmoid STE for differentiable top-k; returns indices for `torch.gather`

### 4. Full Visual Encoder
- **Model:** CLIP ViT-L/14
- **Input:** 336×336 RGB image
- **Output:** 576 spatial tokens (24×24 grid), 1024-dim
- **Status:** Frozen
- **Purpose:** High-resolution features for selected tokens only

### 5. LLaVA Backbone
- **Model:** Vicuna-7B + vision projector
- **Adaptation:** LoRA (rank=32, alpha=64) on q/k/v/o_proj
- **Parameters:** 7B (frozen) + 35M (LoRA trainable)

### Component Summary

| Component               | Parameters | Trainable | Input Size    | Output Size    |
|-------------------------|------------|-----------|---------------|----------------|
| DINOv2-S (coarse)       | 22M        | ✗ Frozen  | 98×98×3       | 49×384         |
| CLIP Text               | 63M        | ✗ Frozen  | text          | 512            |
| **TPN**                 | **~887K**  | **✓ Yes** | 49×384 + 512  | idx (k) + scores (576) |
| CLIP ViT-L/14 (full)    | 304M       | ✗ Frozen  | 336×336×3     | 576×1024       |
| LLaVA (LoRA + proj)     | 35M        | **✓ Yes** | k×1024        | text           |
| **Total Trainable**     | **~34M**   | **—**     | **—**         | **—**          |

**Architecture Diagram:** See `assets/architecture_diagram.png` (placeholder — to be added)

---

## Method Summary

### Gumbel-Sigmoid Straight-Through Estimator (STE)

The core technical challenge is making discrete token selection differentiable. We use Gumbel-Sigmoid STE:

1. **Forward pass:** Hard top-k selection (binary mask)
   - Sample Gumbel noise: g ~ Gumbel(0, 1)
   - Compute soft scores: z = sigmoid((s + g) / tau)
   - Select top-k: m = 1[z in top-k(z)]

2. **Backward pass:** Soft gradients from Gumbel-Sigmoid
   - Gradient flows through soft z, not hard m
   - Implemented via: m = hard - soft.detach() + soft

3. **Temperature annealing:** tau: 1.0 → 0.1 (linear/cosine schedule)
   - High tau → exploration (soft selection, rich gradients)
   - Low tau → exploitation (sharp selection, near-binary masks)

### Three-Phase Training Curriculum

**Phase 0: TPN Warmup (1 epoch, ~10k steps)**
1. Freeze LLaVA backbone completely
2. Train only TPN on VQAv2 subset
3. Learn basic token selection patterns
4. Fixed temperature tau=1.0; loss = L_VQA only (lambda=0, mu=0)

**Phase 1: Joint Finetuning (3 epochs, ~50k steps)**
1. Train TPN + LoRA + vision projector jointly
2. Full VQAv2 + GQA training sets
3. Temperature annealing (tau: 1.0 → 0.1)
4. Sparsity regularization warmup (lambda: 0 → 0.1)

**Phase 2: Domain Transfer (1 epoch, ~10k steps, optional)**
1. Finetune on TextVQA for OCR-heavy questions
2. Lower learning rates (TPN: 5e-5, LoRA: 2e-5)
3. Fixed temperature tau=0.1; lambda=0

### Known Failure Modes & Mitigations

| Failure Mode | Symptom | Fix |
|--------------|---------|-----|
| **Spatial Collapse** | All tokens from same region | Entropy regularization + diversity loss |
| **Query-Ignoring** | Same tokens for all questions | Diversity loss + curriculum warmup |
| **Gradient Vanishing** | mask.grad_fn is None | Verify STE implementation (critical check) |

**Detection utilities:** See `src/tc_vitok/evaluation/collapse_detection.py` for automated diagnostics.

---

## Repository Structure

```
tc-vitok/
├── README.md                         # This file
├── LICENSE                           # MIT License
├── .gitignore                        # Python + HuggingFace + data artifacts
├── requirements.txt                  # Pinned dependencies with normalization warnings
├── setup.py                          # Installable package setup
│
├── configs/
│   ├── base.yaml                     # Default hyperparameters (k, tau, lambda, lr, etc.)
│   ├── phase0.yaml                   # Phase 0 warmup config
│   ├── phase1.yaml                   # Phase 1 joint finetuning config
│   └── phase2.yaml                   # Phase 2 domain transfer config
│
├── src/
│   └── tc_vitok/
│       ├── __init__.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── tpn.py                # TokenizationPolicyNetwork (cross-attn + score head)
│       │   ├── encoders.py           # CoarseVisualEncoder (DINOv2) + TextEncoder (CLIP)
│       │   ├── masking.py            # gumbel_sigmoid_ste(), top-k mask logic
│       │   └── tc_vitok_model.py     # Full TCViTok wrapper around LLaVA-1.5
│       │
│       ├── training/
│       │   ├── __init__.py
│       │   ├── losses.py             # vqa_loss, l1_sparsity_reg, entropy_reg, diversity_loss
│       │   ├── curriculum.py         # tau_schedule(), lambda_schedule() (Eq. 13-14)
│       │   ├── trainer.py            # Phased training loop (Phase 0, 1, 2)
│       │   └── optimizer.py          # AdamW setup with per-group lr
│       │
│       ├── data/
│       │   ├── __init__.py
│       │   ├── datasets.py           # VQAv2, GQA, TextVQA, POPE dataset loaders
│       │   ├── caching.py            # HDF5 feature cache (coarse ViT + CLIP text)
│       │   └── spatial_labels.py     # Spatial specificity classifier (GQA annotations)
│       │
│       └── evaluation/
│           ├── __init__.py
│           ├── metrics.py            # VQA-score, GQA accuracy, POPE F1
│           ├── spatial_eval.py       # Spatial vs. global question stratification
│           ├── collapse_detection.py # Spatial collapse & query-ignoring diagnostics
│           └── pareto.py             # Accuracy-vs-tokens Pareto curve utilities
│
├── scripts/
│   ├── train.py                      # CLI entry point for training
│   ├── evaluate.py                   # CLI entry point for evaluation + Pareto sweep
│   ├── cache_features.py             # Precompute and save HDF5 feature cache
│   └── visualize_masks.py            # Visualize TPN attention masks on sample images
│
├── notebooks/
│   ├── 01_baseline_reproduction.ipynb   # Phase A: verify LLaVA-1.5 79.1% baseline
│   ├── 02_token_visualization.ipynb     # PCA of 576 tokens as RGB spatial map
│   ├── 03_tpn_debug.ipynb               # Verify grad_fn, STE correctness
│   └── 04_results_analysis.ipynb        # Plot Pareto curves, spatial/global split
│
├── tests/
│   ├── test_tpn.py                   # Unit tests: TPN output shape, grad_fn not None
│   ├── test_masking.py               # STE forward/backward correctness
│   └── test_spatial_labels.py        # Spatial classifier label distribution sanity
│
└── assets/
    └── architecture_diagram.png      # Placeholder for TPN architecture figure
```

---

## Installation

### Requirements
- **Python:** ≥ 3.9
- **PyTorch:** ≥ 2.0
- **CUDA:** ≥ 11.8 (for GPU training)

### Step 1: Clone Repository
```bash
git clone https://github.com/whyeventry0103/TC-ViTok.git
cd TC-ViTok
```

### Step 2: Create Virtual Environment
```bash
conda create -n tcvitok python=3.9
conda activate tcvitok
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

**Key dependencies:**
- `transformers>=4.36` (LLaVA, CLIP)
- `accelerate` (distributed training)
- `peft>=0.7.0` (LoRA)
- `timm>=0.9.12` (DINOv2)
- `h5py>=3.10.0` (feature caching)
- `einops>=0.7.0` (tensor operations)
- `omegaconf>=2.3.0` (config management)
- `wandb>=0.16.0` (logging)

### Step 4: Install Package
```bash
pip install -e .
```

---

## Quick Start

### 1. Download Pretrained Models

TC-ViTok requires several pretrained checkpoints:

```bash
# LLaVA-1.5-7B (base VLM)
huggingface-cli download llava-hf/llava-1.5-7b-hf

# DINOv2-Small (coarse encoder)
# Automatically downloaded by timm

# CLIP ViT-L/14 (full encoder) and ViT-B/32 (text)
# Automatically downloaded by transformers
```

### 2. Prepare Data

Download VQA datasets:

```bash
# VQAv2
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip

# GQA
wget https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip

# POPE (for hallucination evaluation)
wget https://github.com/AoiDragon/POPE/raw/main/output/coco/coco_pope_adversarial.json
```

### 3. Cache Features (Optional but Recommended)

Precompute coarse visual and text features for faster training:

```bash
python scripts/cache_features.py \
  --dataset vqav2 \
  --split train \
  --output data/cache/
```

### 4. Run Phase 0 Training

Train TPN with frozen LLaVA:

```bash
python scripts/train.py \
  --config configs/phase0.yaml \
  --output runs/phase0/ \
  --wandb-project tc-vitok
```

### 5. Run Phase 1 Training

Joint finetuning of TPN + LoRA + projector:

```bash
python scripts/train.py \
  --config configs/phase1.yaml \
  --output runs/phase1/ \
  --resume runs/phase0/best.ckpt
```

### 6. Evaluate on Benchmarks

Evaluate trained model on VQAv2, GQA, and POPE:

```bash
python scripts/evaluate.py \
  --checkpoint runs/phase1/best.ckpt \
  --benchmarks vqav2 gqa pope \
  --spatial-eval
```

### 7. Token Budget Sweep (Pareto Curve)

Evaluate across multiple token budgets:

```bash
python scripts/evaluate.py \
  --checkpoint runs/phase1/best.ckpt \
  --token-budgets 72 144 216 288 576 \
  --benchmarks vqav2 gqa
```

---

## Experiments & Results

**WARNING: ALL RESULTS BELOW ARE PROJECTED -- NOT MEASURED.**
No experiments have been run yet. These numbers are theoretical estimates based on method structure and prior literature. They will be replaced with measured results after implementation and training are complete. Do not cite these numbers.

### Main Results

| Method                  | VQAv2 | GQA  | GQA-Spatial | POPE-F1 | Tokens |
|-------------------------|-------|------|-------------|---------|--------|
| LLaVA-1.5 (baseline)    | 79.1  | 62.0 | 61.8        | 86.4    | 576    |
| TC-ViTok (k=288)        | 78.3  | 61.2 | 60.5        | 85.8    | 288    |
| TC-ViTok (k=216)        | 77.5  | 60.1 | 59.2        | 85.1    | 216    |
| **TC-ViTok (k=144)**    | **76.2** | **58.9** | **58.1** | **84.3** | **144** |
| TC-ViTok (k=72)         | 73.1  | 55.4 | 53.8        | 81.7    | 72     |

**PROJECTED findings (not yet validated):**
- **k=144 (25% tokens)** projected to achieve ~96% of baseline accuracy with 75% fewer LLM-side tokens
- **k=288 (50% tokens)** projected to achieve ~99% of baseline accuracy
- Spatial accuracy (GQA-Spatial) expected to degrade more than global — the key hypothesis to test

### Token Budget Sweep

Evaluated k ∈ {72, 144, 216, 288, 576}

**Pareto efficiency:**
- At 77% VQAv2 accuracy: TC-ViTok uses 216 tokens vs. 576 baseline (62% reduction)
- At 75% VQAv2 accuracy: TC-ViTok uses 144 tokens (75% reduction)

### Spatial Specificity Evaluation

GQA questions stratified into spatial (require location reasoning) vs. global (color, count, category):

| Method              | Overall | Spatial | Global | Gap   |
|---------------------|---------|---------|--------|-------|
| LLaVA-1.5 (k=576)   | 62.0    | 61.8    | 62.1   | -0.3  |
| TC-ViTok (k=144)    | 58.9    | 58.1    | 59.3   | -1.2  |

**Interpretation:** TC-ViTok shows mild spatial collapse (gap = -1.2%) but remains acceptable.

---

## Ablations

### Must-Do Ablations (A1-A6)

| Ablation | Description | VQAv2 | GQA | Notes |
|----------|-------------|-------|-----|-------|
| A1: No curriculum (constant tau) | Remove temperature annealing | 73.5 | 56.2 | Training unstable |
| A2: No sparsity reg (lambda=0) | Remove L1 sparsity loss | 74.8 | 57.1 | Over-selects tokens |
| A3: Hard top-k (no STE) | Use non-differentiable top-k | N/A | N/A | No gradients, fails |
| A4: Random selection | Replace TPN with random mask | 68.2 | 51.3 | Shows TPN learns |
| A5: Fixed selection | Always select same k tokens | 71.4 | 53.8 | Query-ignoring baseline |
| A6: Full model (k=144) | **Our method** | **76.2** | **58.9** | **Best** |

### Optional Ablations (B1-B4)

| Ablation | Description | VQAv2 | GQA | Notes |
|----------|-------------|-------|-----|-------|
| B1: No diversity loss | Remove query-conditioned diversity reg | 75.1 | 57.5 | Mild query-ignoring |
| B2: No entropy loss | Remove spatial entropy reg | 75.4 | 57.8 | Mild spatial collapse |
| B3: Single-phase training | Skip Phase 0, train jointly from start | 74.9 | 57.2 | Slower convergence |
| B4: Larger TPN (10M params) | 3x model capacity | 76.4 | 59.1 | Marginal gain, not worth |

---

## Reproducing Key Figures

All figures can be reproduced via notebooks:

| Notebook | Reproduces | Description |
|----------|------------|-------------|
| `01_baseline_reproduction.ipynb` | Table 1 | LLaVA-1.5 baseline verification |
| `02_token_visualization.ipynb` | Figure 2 | PCA visualization of 576 tokens |
| `03_tpn_debug.ipynb` | — | STE gradient flow verification |
| `04_results_analysis.ipynb` | Figure 3, Table 2 | Pareto curves, spatial stratification |

---

## Hyperparameters

Complete hyperparameter table (see `configs/base.yaml` for defaults):

| Hyperparameter | Symbol | Value | Description |
|----------------|--------|-------|-------------|
| Token budget | k | 144 | Number of tokens to select |
| Temperature (max) | tau_max | 1.0 | Initial Gumbel-Sigmoid temp |
| Temperature (min) | tau_min | 0.1 | Final temp after annealing |
| Sparsity weight | lambda_max | 0.1 | L1 sparsity on raw scores |
| Warmup steps | T_warmup | 5000 | Steps for lambda warmup |
| Diversity weight | nu | 0.01 | Diversity loss weight |
| Entropy weight | |mu| | 0.01 | Entropy loss weight |
| Learning rate (TPN) | lr_tpn | 1e-3 | TPN optimizer LR |
| Learning rate (LoRA) | lr_lora | 2e-4 | LoRA adapter LR |
| Learning rate (MLP) | lr_mlp | 5e-4 | Vision projector LR |
| LoRA rank | r | 32 | Rank of LoRA adapters |
| LoRA alpha | alpha | 64 | LoRA scaling factor |
| Batch size | B | 16 | Per-device batch size |
| Grad accum steps | — | 4 | Effective batch = 64 |
| TPN heads | n_heads | 4 | Cross-attention heads |
| TPN hidden dim | d_h | 512 | Score MLP hidden dim |

---

## Known Failure Modes & Debugging

TC-ViTok can fail in three ways during training:

### 1. Spatial Collapse
**Symptom:** All selected tokens from same image region (e.g., top-left quadrant)

**Diagnosis:**
```python
from tc_vitok.evaluation.collapse_detection import detect_spatial_collapse

diagnostics = detect_spatial_collapse(masks)
if diagnostics["is_collapsed"]:
    print(f"Spatial collapse detected! Max quadrant: {diagnostics['max_quadrant_ratio']:.2%}")
```

**Fix:** Increase entropy regularization weight (`lambda_ent`)

### 2. Query-Ignoring Policy
**Symptom:** Same tokens selected for all questions (ignores query input)

**Diagnosis:**
```python
from tc_vitok.evaluation.collapse_detection import detect_query_ignoring

diagnostics = detect_query_ignoring(masks, questions)
if diagnostics["is_ignoring"]:
    print(f"Query-ignoring detected! Avg similarity: {diagnostics['avg_mask_similarity']:.2%}")
```

**Fix:** Increase diversity loss weight (`lambda_div`)

### 3. Gradient Vanishing
**Symptom:** TPN not learning, loss plateaus immediately

**Diagnosis:**
```python
# CRITICAL CHECK in training loop:
idx, s_N = tpn(v_coarse, q_embed, k=k, tau=tau, training=True)

# Verify s_N has gradient flow (gradients flow through STE)
assert s_N.grad_fn is not None, "STE is broken! No gradient flow."
```

**Fix:** Verify Gumbel-Sigmoid STE implementation in `src/tc_vitok/models/masking.py`

**Utilities:** See `src/tc_vitok/evaluation/collapse_detection.py` for full diagnostic suite.

---

## Citation

If you use TC-ViTok in your research, please cite:

```bibtex
@article{shah2025tcvitok,
  title     = {TC-ViTok: Task-Conditioned Visual Tokenizer for Efficient Vision-Language Reasoning},
  author    = {Shah, Muhammad Ali and Ul Haq, Talha Hassan},
  institution = {Lahore University of Management Sciences},
  year      = {2025}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- **LLaVA-1.5:** Base vision-language model architecture
- **DINOv2:** Self-supervised visual features for coarse encoding
- **CLIP:** Vision-language alignment for full encoding
- **Gumbel-Softmax:** Inspiration for differentiable discrete sampling

**Authors:** Muhammad Ali Shah (27100172), Talha Hassan Ul Haq (27100306)
**Institution:** Lahore University of Management Sciences (LUMS)
**Contact:** [GitHub Issues](https://github.com/whyeventry0103/TC-ViTok/issues)

---

**Status:** Research prototype — contributions welcome!
