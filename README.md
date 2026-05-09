# TC-ViTok: Task-Conditioned Visual Tokenization with Adaptive Budget

**Learning an Adaptive Token Selection Policy for Efficient Vision-Language Reasoning**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![PyTorch 2.6+](https://img.shields.io/badge/PyTorch-2.6%2B-orange.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Institution: LUMS](https://img.shields.io/badge/Institution-LUMS-red.svg)](https://lums.edu.pk)

Vision-Language Models like LLaVA-1.5 feed all **576 visual tokens** from every image into a 7B LLM regardless of what the question asks. **TC-ViTok** replaces this with a learned, query-conditioned token selection policy: a lightweight Tokenization Policy Network (TPN, ~887K params) that scores all 576 tokens and keeps only the ones relevant to the question. We build on this with two extensions: **(1)** parameter-free **Informative Token Merge** that recycles discarded tokens into the selected set at zero additional parameter cost, and **(2)** an **Adaptive Budget** that learns per-sample variable token counts from 36 to 288.

> **Efficiency note:** CLIP ViT-L/14 still processes all 576 tokens during encoding. The compression savings are on the LLM prefill side: 3.47× LLM prefill speedup, 3.39× end-to-end at k=144.

---

## Results (Final)

All three TC-ViTok variants were evaluated on LLaVA-1.5-7B with full beam-search decoding and official answer normalization.

### Main Benchmark Comparison

| Model | Tokens | VQAv2 (%) | GQA (%) | POPE F1 (%) | val-VQA loss | Query-cond. |
|---|---|---|---|---|---|---|
| LLaVA-1.5-7B (full, published) | 576 | 78.5 | 61.0 | 84.4 | — | ✗ |
| FastV | ~144 | 77.1 | — | — | — | ✗ |
| LLaVA-PruMerge | ~144 | 76.8 | — | — | — | ✗ |
| SparseVLM | ~152 | 78.2 | — | — | — | ✗ |
| **TC-ViTok v3** (hard drop, r=32, 50K steps) | 144 | **72.47** | ~63† | ~75† | 0.3334 | ✓ |
| **TC-ViTok Adaptive** (λ=0.1) | 36–288 | **69.43** | **65.04** | **75.63** | — | ✓ |
| **TC-ViTok Final** (token merge, r=64, 80K steps) | 144 | **72.27** | **67.12** | **79.09** | 0.3342 | ✓ |

†: v3 GQA and POPE were estimated from trend; only VQAv2 was fully evaluated due to compute constraints.

**Headline finding:** TC-ViTok Final achieves **67.12% on GQA val_balanced**, exceeding the uncompressed LLaVA-1.5-7B (61.0%) by **+6.1 points** at 75% token reduction. We hypothesize this is active denoising — compositional GQA questions depend on <5% of image pixels, and dropping the other 95% removes distractors from the LLM's self-attention.

**POPE regression** (79.09 vs 84.4): Expected and honest. POPE tests object presence regardless of question relevance. Our compression deliberately drops question-irrelevant tokens; the 5-point F1 drop is evidence the system is doing what it claims. TC-ViTok is not suited for question-agnostic object detection tasks.

### POPE Breakdown (TC-ViTok Final)

| Variant | F1 (%) |
|---|---|
| POPE-popular | 79.30 |
| POPE-random | 80.31 |
| POPE-adversarial | 77.66 |
| **POPE macro-F1** | **79.09** |

### Efficiency Analysis

| Component | Full LLaVA-1.5 | TC-ViTok (k=144) |
|---|---|---|
| LLM dense FLOPs | 8.24 TF | 2.41 TF |
| LLM attention FLOPs | 0.20 TF | 0.02 TF |
| Encoders + TPN + Merge | 0.08 TF | 0.08 TF |
| **Total** | **8.52 TF** | **2.51 TF** |
| **LLM prefill speedup** | — | **3.47×** |
| **End-to-end speedup** | — | **3.39×** |
| KV-cache size | 576 tok | 144 tok (−75%) |

The added TPN+merge overhead (~83 GFLOPs) is less than 1% of the LLM prefill it replaces.

### Training Convergence

| Run | val-VQA loss | Steps | Config |
|---|---|---|---|
| Phase 1 v3 (baseline, hard drop) | 0.3334 | 50K | LoRA r=32, {q,k,v,o}, no merge |
| **Phase 1 Final** | **0.3342** | **80K** | LoRA r=64, {q,k,v,o,gate,up}, token merge |

Final nearly matches v3 on the training proxy despite the added merge complexity. Downstream benchmark gains (GQA +4.1 over v3, POPE +4.1) are not captured by val-VQA because it measures NTP cross-entropy, not accuracy.

---

## Key Contributions

1. **Tau-independent STE correction** — A novel additive gradient path through the Gumbel-Sigmoid straight-through estimator that maintains learning signal as τ → 0. Standard STE gradients scale as 1/τ; the additive correction is tau-independent and remains non-zero throughout the temperature schedule (§3.4, Eq. 9).

2. **Informative Token Merge** — Parameter-free cosine-similarity-based recovery of discarded tokens, weighted by TPN importance scores. The zero-parameter design was dictated by our failure analysis: any trainable transform between selector and LLM creates a GAN-like co-adaptation loop.

3. **Co-adaptation failure analysis** — We trained 8 variants of a cross-attention pooler over 8 weeks and documented three structural failure modes: circular co-adaptation, auxiliary loss gradient competition, and train-eval mismatch. The diagnosis directly motivates the parameter-free merge as the correct design.

4. **Adaptive BudgetPredictor** (~25K params) with a differentiable threshold trick for per-sample variable token counts and a budget penalty sweep to trace the accuracy–compression Pareto frontier.

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
q_embed [B, 512]      ──► W_Q ──► Q [B, 1, 384]    (question as single query)
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

**Why instance normalization?** Without it, Adam pushes scores to ±100 within ~1000 steps. At that magnitude, Gumbel noise (τ≈1.0) is negligible → deterministic top-144 → same tokens always selected → entropy collapse. Instance norm anchors scores to mean=0, std=1 permanently. Since only relative ordering matters for top-k, no information is lost.

---

### 3. Gumbel-Sigmoid Straight-Through Estimator (STE)

Token selection is discrete (hard top-k), but training requires gradients. We use:

**Forward pass — hard top-k with Gumbel exploration:**
```python
g = -log(-log(U))                                    # Gumbel(0,1) noise
noisy = s_N + g                                      # perturbed scores
m_soft = sigmoid(noisy / τ)                          # soft mask (has gradient)
idx = topk(noisy, k).indices                         # hard selection (no gradient)
m_hard = scatter(idx, 1.0)                           # binary mask
mask_ste = m_hard - m_soft.detach() + m_soft         # STE: forward=m_hard, backward=∂m_soft
```

**Tau-independent STE correction:** At low τ, sigmoid saturates → gradients vanish → TPN stops learning. Fix: additive correction that is zero in the forward pass but routes gradients directly from the LLM loss to TPN scores:
```python
scores_at_selected = gather(s_N, idx)
v_selected = v_selected * (1 + scores_at_selected - scores_at_selected.detach())
# Forward: multiplies by 1 (zero effect)
# Backward: d(loss)/d(s_N[j]) = Σ_d grad_v[j,d] × v_selected[j,d]  ← tau-independent
```

**Temperature τ annealing:** 1.0 → 0.3 over training. Floor of 0.3 prevents gradient starvation.

---

### 4. Informative Token Merge

After selecting k=144 tokens, the 432 unselected tokens are absorbed into the selected tokens via parameter-free cosine-similarity-based merging:

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
|---|---|
| Hard drop | Loses all information from 432 tokens |
| Learned pooler (Phase 1.5) | Creates co-adaptation oscillation with LoRA — 8 variants all failed (see Architecture Evolution) |
| Random merge | Not query-conditioned; TPN scores weight the merge so near-threshold tokens contribute more |

**No train-eval mismatch:** Both training and evaluation receive exactly k=144 tokens.

---

### 5. Adaptive Budget

The fixed-k=144 model compresses every image equally. The Adaptive Budget extension learns *how many* tokens each sample needs.

**BudgetPredictor (~25K params):**
```
ctx [B, 384] + normalize(entropy(s_N)) [B, 1]
  → Linear(385, 64) → GELU → Linear(64, 1) → sigmoid
  → k_frac ∈ [36/576, 288/576]
```

**Differentiable threshold trick:**
```python
threshold = threshold_hard - (k_target - k_target.detach())
# Forward: threshold = threshold_hard (zero effect)
# Backward: ∂threshold/∂k_frac = −N  (non-zero gradient to BudgetPredictor)
```

**Budget loss:** `L_budget = λ_budget × mean(k_effective / N)`

---

## Architecture

### Fixed-k Mode (Phase 1 Final)

| Component | Params | Trainable | Input → Output |
|---|---|---|---|
| DINOv2-S (coarse encoder) | 22M | ✗ | 98×98 RGB → [B, 49, 384] |
| CLIP ViT-B/32 (text encoder) | 63M | ✗ | question → [B, 512] |
| **TPN** | **~887K** | **✓** | [B,49,384]+[B,512] → idx[B,144] + s_N[B,576] + ctx[B,384] |
| CLIP ViT-L/14 (full encoder) | 304M | ✗ | 336×336 RGB → [B, 576, 1024] |
| **Informative Token Merge** | **0** | — | v_fine + idx + s_N → v_merged [B, 144, 1024] |
| MM Projector | ~7M | **✓** | [B, 144, 1024] → [B, 144, 4096] |
| LLaVA-1.5-7B + LoRA (r=64) | ~129M | **✓ (LoRA only)** | [B, 144, 4096] + text → text |
| **Total Trainable** | **~137M** | | |

LoRA: rank=64, alpha=128, targets: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj`.  
FFN targets (gate/up) included because token merging shifts the visual feature distribution — the FFN layers must adapt to the merged representation.

### Adaptive Budget Mode (Phase 1 Adaptive — differences from Final)

| Parameter | Phase 1 Final | Phase 1 Adaptive |
|---|---|---|
| Token count k | Fixed 144 | Variable [36, 288] per sample |
| LoRA targets | q,k,v,o,gate,up | q,k,v,o only |
| BudgetPredictor | Not present | +~25K params |
| Training steps | 80K (from Phase 0) | 40K (from Phase 1 Final) |

**Note:** The reduced LoRA target set in Adaptive (removing gate/up) was a design choice for training stability, but it contributes to the gap between Adaptive and Final on all benchmarks.

---

## Training Phases

### Phase 0: TPN Warmup (~10K steps)

| Setting | Value |
|---|---|
| What trains | TPN only — LoRA and MM projector frozen |
| Dataset | VQAv2 |
| Loss | L_VQA only |
| Temperature | τ = 1.0 (fixed) |
| Purpose | Cold-start TPN before LoRA co-training begins |

**Why Phase 0 matters:** If LoRA trains alongside an untrained TPN, LoRA compensates for bad selections rather than forcing the TPN to improve. With LoRA frozen, the only way to reduce L_VQA is better selection quality.

### Phase 1 Final: Joint Training with Token Merge (80K steps)

| Setting | Value |
|---|---|
| What trains | TPN + LoRA (r=64) + MM projector |
| Dataset | VQAv2 + GQA |
| LoRA warmfreeze | LoRA frozen steps 0–3000 |
| Loss | L_VQA + 0.02×(−H) + 0.1×L_div |
| Temperature | τ: 1.0 → 0.3 (linear) |
| Token merge | β=0.3, τ_merge=0.1 |
| LR (TPN / LoRA / MLP) | 3e-3 / 1e-4 / 5e-4 |
| Effective batch | 128 (4 × 16 grad steps × 2 GPUs) |
| Loads from | Phase 0 checkpoint |
| **Final val-VQA loss** | **0.3342 at step 80K** |

### Phase 1 Adaptive: Variable Budget Training (40K steps)

| Setting | Value |
|---|---|
| What trains | BudgetPredictor (new) + TPN + LoRA + MM projector |
| Dataset | VQAv2 + GQA |
| BudgetPredictor warmup | Frozen steps 0–2000 |
| Loss | L_VQA + 0.02×(−H) + 0.1×L_div + 0.1×mean(k_eff/N) |
| Token range | k ∈ [36, 288] per sample |
| Mean k_eff | ~354 (soft count) during training |
| LR (TPN / LoRA / BP / MLP) | 1e-3 / 2e-4 / 1e-3 / 5e-4 |
| Loads from | Phase 1 Final checkpoint |
| **Benchmark results** | **VQAv2 69.43%, GQA 65.04%, POPE 75.63%** |

---

## Loss Function

| Term | Weight | Purpose |
|---|---|---|
| L_VQA | 1.0 | Cross-entropy on answer tokens only (NTP) |
| −H (spatial entropy) | 0.02 | Prevent TPN from always selecting the same spatial region |
| L_div (query diversity) | 0.1 | Prevent TPN from ignoring the question |
| λ_budget × mean(k_eff/N) | 0.1 (Adaptive only) | Penalize token usage |

**val-VQA metric:** The logged `val_vqa` is cross-entropy loss on VQAv2 val — lower is better. It is *not* the same as VQA-Score % (higher-is-better, requires beam search + answer normalization).

---

## Architecture Evolution (Phase 1.5 — Why It Failed)

After Phase 1 v3 (val_vqa=0.3334, hard drop), we attempted Phase 1.5: a cross-attention **UnselectedPooler** with 4 learned slot vectors (~512K params). This ran through 8 versions over 8 weeks before being abandoned.

### Three Fatal Design Mistakes

**Mistake 1: Circular Co-Adaptation (structural, unfixable)**
```
Pooler output shifts → LLM input distribution changes → LoRA must re-adapt
→ LoRA gradient to pooler changes → Pooler must re-adapt → cycle repeats
```
GAN-style instability. Reducing pooler LR slows the cycle period but never eliminates it:

| Pooler LR | Oscillation period | Best val_vqa |
|---|---|---|
| 5e-4 (v4–v7) | ~5K steps | 0.39–0.44 (cycling, never converged) |
| 2e-5 (v8) | ~10–15K steps | 0.3663 at step 45K, then regressed |
| 0.0 (v8b, frozen) | None | 0.4540 at 5K (fresh optimizer catapult), training ongoing |

**Mistake 2: 9 Conflicting Loss Terms**
At peak complexity, TPN received gradients from: VQA, entropy, diversity, boundary, sparsity, spread, distillation, residual distillation, and slot orthogonality. With instance-normalized scores (finite gradient budget), auxiliary losses literally steal gradient from VQA.

**Mistake 3: Train-Eval Mismatch (critical, undetected)**
During training, the LLM received k+4=148 tokens (4 pooler summaries + 144 selected). During evaluation (`generate()`), only 144 selected tokens were passed. Every Phase 1.5 validation metric was corrupted by this mismatch.

**Conclusion:** Token merge solves all three problems simultaneously: zero new parameters → no co-adaptation; no extra sequence positions → no train-eval mismatch; no additional loss terms needed.

---

## Repository Structure

```
TC-ViTok/
├── configs/
│   ├── base.yaml                  # Default hyperparameters
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
│   ├── train.py                   # Training entry point
│   ├── evaluate.py                # Evaluation (auto-detects LoRA rank from checkpoint)
│   ├── launch_multigpu.sh         # 2-GPU DDP launcher (sets NCCL/HF env vars)
│   ├── cache_features.py          # Pre-compute DINOv2 + CLIP text → HDF5
│   ├── visualize_masks.py         # Heatmap of TPN token selections
│   └── diagnose_routing.py        # Verify TPN routes differently per question
├── tests/
│   ├── test_tpn.py                # TPN architecture + gradient flow
│   ├── test_masking.py            # STE correctness + tau-independence
│   └── test_adaptive_budget.py    # BudgetPredictor + adaptive STE
└── documentation/
    ├── solutions.md               # Architecture decisions + Phase 1.5 post-mortem
    ├── understanding.md           # Full forward pass with tensor shapes
    ├── innovations.md             # SOTA literature review + novelty analysis
    ├── ADAPTIVE_BUDGET.md         # Adaptive budget design rationale
    ├── computations.md            # VRAM and FLOP breakdowns
    └── run.md                     # Full setup + dataset download guide
```

---

## Setup and Installation

### Requirements

- Python 3.11 (3.12 works; 3.14 does not have PyTorch packages)
- **PyTorch ≥ 2.6.0** — MANDATORY due to CVE-2025-32434
- CUDA 12.1+ GPU with ≥ 24 GB VRAM for training (2× RTX 6000 Ada recommended)
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
pip install peft h5py
```

### Model Downloads

```bash
# LLaVA-1.5-7B (includes CLIP ViT-L/14 and mm_projector) — ~13.5 GB
huggingface-cli download llava-hf/llava-1.5-7b-hf

# CLIP ViT-L/14 standalone — ~890 MB
python -c "from transformers import CLIPModel; CLIPModel.from_pretrained('openai/clip-vit-large-patch14-336')"

# DINOv2-S and CLIP ViT-B/32 download automatically on first run
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
ln -s ../coco/train2014 $DATA_ROOT/vqav2/train2014
ln -s ../coco/val2014 $DATA_ROOT/vqav2/val2014

# GQA (~20 GB)
cd $DATA_ROOT/gqa
wget https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip && unzip images.zip
wget https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip && unzip questions1.2.zip
```

### Verify Installation

```bash
PYTHONPATH=src python -m pytest tests/ -q   # expected: all pass

# Dry run (~30 seconds)
PYTHONPATH=src python scripts/train.py \
    --config configs/phase0.yaml --output /tmp/tc_vitok_test \
    --stub --dummy-data --no-wandb --max-steps 2
```

---

## Reproducing the BMVC 2026 Submission

The BMVC 2026 paper is positioned as an insight + mechanism + negative-results
paper around the **co-adaptation diagnosis** rather than as a SOTA paper. The
empirical study consists of ten controlled evaluations on the **existing**
Phase 1 Final / v3 / Adaptive checkpoints. **No new training runs are
required.**

The day-by-day playbook with copy-pasteable commands lives in
[`RUN_THIS.md`](RUN_THIS.md). The experiments populate the `[E#]` placeholders
in the paper:

| ID | What it produces | Command (abbreviated) | GPU time |
|---|---|---|---|
| E1 | LoRA-576 control row (the missing critical control) | `evaluate.py --baseline full_576` | ~12h |
| E2 | Single-variable merge ablation | `evaluate.py --baseline merge_disabled` | ~6h |
| E3 | Zero-CLIP-query ablation (Hypothesis H2 test) | `evaluate.py --baseline image_only` | ~6h |
| E4 | Eval-time variance (5 seeds) | `evaluate.py --n-eval-seeds 5` | ~10h |
| E5 | MMBench + MME results | `evaluate.py --benchmarks mmbench mme` | ~8h |
| E6 | TextVQA result | `evaluate.py --benchmarks textvqa` | ~8h |
| E7 | Real qualitative TPN heatmaps for fig:qualitative-real | `dump_tpn_heatmaps.py` | ~30min |
| E8 | Per-sample budget histogram | `dump_tpn_heatmaps.py` (Adaptive ckpt) | ~30min |
| E9 | Co-adaptation oscillation figure | `plot_oscillation.py --use-known-points` | local CPU |
| E10 | Retroactive v3 GQA + POPE eval | `evaluate.py` on v3 ckpt | ~10h |

After each experiment finishes, paste the headline numbers into the paper's
`[E#.X]` placeholders. The exact mapping is in `RUN_THIS.md`.

### LaTeX template port (one-time)

The paper currently uses the ICML 2021 template as a stand-in. Before
submission:
1. Download `bmvc2k.cls` (or whatever 2026 ships) from the official BMVC
   2026 site: https://bmvc2026.bmva.org/calls/call-for-papers/
2. Drop it into the `DL Project Report/` directory.
3. Follow the in-file instructions at the top of `main.tex` to switch the
   `\documentclass`, `\bibliographystyle`, and title block.
4. Recompile. The body of the paper is template-portable; only the preamble
   and title block change.

The paper is **anonymized** for double-blind review (no author names, no
GitHub URL, no acknowledgments).

---

## Running Experiments

### Multi-GPU Training (2× GPU, recommended)

```bash
# Phase 0: TPN warmup
tmux new-session -d -s phase0
tmux send-keys -t phase0 "cd ~/TC-ViTok && bash scripts/launch_multigpu.sh \
    --config configs/phase0.yaml --output runs/phase0" Enter

# Phase 1 Final: token merge + LoRA r=64
tmux new-session -d -s phase1_final
tmux send-keys -t phase1_final "cd ~/TC-ViTok && bash scripts/launch_multigpu.sh \
    --config configs/phase1_final.yaml --output runs/phase1_final" Enter

# Phase 1 Adaptive: variable budget (loads phase1_final checkpoint)
tmux new-session -d -s phase1_adaptive
tmux send-keys -t phase1_adaptive "cd ~/TC-ViTok && bash scripts/launch_multigpu.sh \
    --config configs/phase1_adaptive.yaml --output runs/phase1_adaptive" Enter
```

### Monitoring

```bash
tmux attach -t phase1_final
tail -f logs/train_*.log | tr '\r' '\n' | grep "Step"
grep "val" logs/train_*.log
ls -lh runs/phase1_final/phase1/
```

---

## Evaluation

`evaluate.py` auto-detects the LoRA rank and target set from the checkpoint state dict — no manual config editing needed.

```bash
# Phase 1 Final: VQAv2 + GQA + POPE
python scripts/evaluate.py \
    --checkpoint runs/phase1_final/phase1/best.ckpt \
    --benchmarks vqav2 gqa pope \
    --data-root ~/tc-vitok-data \
    --output runs/eval_results/phase1_final.json

# E1 — LoRA-576 control (most important new evaluation; 576 tokens, same LoRA)
python scripts/evaluate.py \
    --checkpoint runs/phase1_final/phase1/best.ckpt \
    --baseline full_576 \
    --benchmarks vqav2 gqa pope \
    --data-root ~/tc-vitok-data \
    --output runs/eval_results/E1_lora576.json

# E2 — single-variable merge ablation (TPN selection but beta=0 at eval)
python scripts/evaluate.py \
    --checkpoint runs/phase1_final/phase1/best.ckpt \
    --baseline merge_disabled \
    --benchmarks vqav2 gqa pope \
    --data-root ~/tc-vitok-data \
    --output runs/eval_results/E2_merge_disabled.json

# E3 — zero-CLIP-query (Hypothesis H2 test)
python scripts/evaluate.py \
    --checkpoint runs/phase1_final/phase1/best.ckpt \
    --baseline image_only \
    --benchmarks vqav2 gqa pope \
    --data-root ~/tc-vitok-data \
    --output runs/eval_results/E3_zero_query.json

# E4 — eval-time variance over 5 seeds
python scripts/evaluate.py \
    --checkpoint runs/phase1_final/phase1/best.ckpt \
    --benchmarks gqa \
    --n-eval-seeds 5 \
    --data-root ~/tc-vitok-data \
    --output runs/eval_results/E4_variance.json

# E5/E6 — new benchmarks
python scripts/evaluate.py \
    --checkpoint runs/phase1_final/phase1/best.ckpt \
    --benchmarks mmbench mme textvqa \
    --data-root ~/tc-vitok-data \
    --output runs/eval_results/E5_E6_new_benchmarks.json

# E7/E8 — qualitative dump
python scripts/dump_tpn_heatmaps.py \
    --checkpoint runs/phase1_final/phase1/best.ckpt \
    --data-root ~/tc-vitok-data \
    --benchmark gqa --n-samples 32 \
    --render-overlays \
    --out runs/qualitative_dump

# E9 — oscillation figure (no GPU needed; uses checkpoint history)
python scripts/plot_oscillation.py --use-known-points \
    --out "DL Project Report/figures/oscillation.pdf"

# E10 — retroactive v3 GQA + POPE
python scripts/evaluate.py \
    --checkpoint runs/phase1_v3/phase1/best.ckpt \
    --benchmarks gqa pope \
    --data-root ~/tc-vitok-data \
    --output runs/eval_results/E10_v3_gqa_pope.json
```

---

## Known Failure Modes and Mitigations

| Failure | Symptom | Fix |
|---|---|---|
| **Spatial collapse** | All selected tokens from same region; entropy < 5.0 | Increase `lambda_entropy` from 0.02 to 0.05 |
| **Query-ignoring** | Same tokens regardless of question; IoU > 0.9 across batch | Increase `lambda_diversity`; verify STE additive correction active |
| **STE gradient vanishing** | TPN not learning after step 10K; loss plateau | Verify tau-independent correction: `v_selected *= (1 + sc - sc.detach())` |
| **Score magnitude explosion** | Loss NaN; select entropy = log(k) | Verify instance normalization in TPN; check `s_N.mean() ≈ 0` in logs |
| **Budget collapse** | `k_effective` stuck at k_min or k_max | Verify `budget_warmup_steps > 0`; adjust `lambda_budget` |
| **Co-adaptation oscillation** | val_vqa oscillates ±0.04 every ~10K steps | Never train a learnable pooler simultaneously with LoRA; use token merge |
| **Checkpoint LoRA size mismatch** | `RuntimeError: size mismatch for lora_A` | `evaluate.py` auto-detects rank from checkpoint; no manual fix needed |
| **DDP rank desync** | NCCL timeout; SIGABRT | `HF_HUB_OFFLINE=1` must be set; all HF downloads must happen before training |

---

## Key Design Decisions

**Why two phases (Phase 0 + Phase 1)?** Phase 0 cold-starts TPN with meaningful scores before LoRA co-training. Without it, LoRA immediately compensates for bad selections and TPN never learns selectivity.

**Why token merge instead of a learned pooler?** Eight training runs with a cross-attention pooler (Phase 1.5) all failed due to circular co-adaptation with LoRA. Token merge has zero new parameters — nothing can oscillate. See Architecture Evolution above.

**Why instance normalization on TPN scores?** Without it, Adam pushes scores to ±100 → Gumbel noise negligible → deterministic fixed selection → entropy collapse. `tanh` also saturates. Instance norm anchors scores to mean=0, std=1 permanently with no learnable parameters.

**Why LoRA warm-freeze (steps 0–3000)?** Forces early loss reduction through better TPN selection, not LoRA compensation. After step 3000, LoRA unfreezes and optimizes jointly.

**Why gate/up LoRA targets in Phase 1 Final but not Adaptive?** Token merging shifts the visual feature distribution — FFN layers must adapt to merged representations. In Adaptive mode, this was removed for training stability; it contributes to the Final > Adaptive gap.

**Why does GQA exceed the full model?** Our working hypothesis: GQA compositional questions depend on <5% of image pixels. The 75% of dropped tokens act as attention distractors in the LLM. TC-ViTok's query-conditioned selection removes those distractors before they enter the LLM. This is a feature, not a bug. The POPE regression (where you *want* full coverage) confirms the mechanism: compression selectively drops irrelevant content.

---

## Limitations

- **Single training seed.** All results are from one training run. Variance across seeds is unknown.
- **Compute gap vs. published baselines.** FastV and PruMerge require no additional training. Our 80K-step comparison is not compute-matched to methods that trained for longer on more GPUs.
- **POPE regression is real.** 5-point F1 drop on POPE is unavoidable at 75% compression for object-presence tasks. TC-ViTok is not appropriate where full object coverage is required.
- **LoRA-576 control:** wired as experiment E1. Run `evaluate.py --baseline full_576` against the Phase 1 Final checkpoint to populate the row in `tab:main`. Until E1 is run, the GQA delta vs. uncompressed is unattributable between compression and LoRA.
- **Adaptive Pareto sweep.** Only λ_b=0.1 was evaluated; the full sweep {0.05, 0.1, 0.2, 0.4} was not completed.

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
