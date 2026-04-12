# TC-ViTok: Innovation Roadmap to 76% VQAv2

**Current state:** ~72.4% VQAv2, 144 tokens (25% of 576), val entropy -5.13  
**Target:** 76% VQAv2 (paper claim), GQA spatial > image-only TPN by >3%  
**Gap:** ~3.6% — primarily from hard-dropping 432 tokens; soft recycling is the main fix

---

## SOTA Literature Insights

### SparseVLM (ICML/ICLR 2025) — Token Recycling
Explicitly warns against hard-dropping visual tokens. Uses LLM self-attention
patterns to recycle pruned tokens into compact representations. Key insight:
discarded tokens carry contextual information (background, global scene) that
helps the LLM decode answers even when not directly attended to.

**Relevance:** Directly validates the soft recycling direction. Their method
uses attention inside LLM layers; we must do it pre-LLM (after TPN, before the
LLM prefix). The mechanism must therefore be based on TPN scores or a
lightweight learned pooling rather than LLM attention.

### QG-VTC (Neurocomputing, April 2025) — Hierarchical Soft Compression
Compresses progressively inside CLIP-ViT layers (576→492→408→324→240→156→72)
using soft attention-weighted averaging. Gets 75.31% VQAv2 with only 72 tokens
(half our budget). Directly proves soft recycling recovers ~2-3% over hard drop.

**Relevance:** Strongest evidence for the expected gain. Their hierarchical
approach is inside the vision encoder (incompatible with our TPN post-encoder
architecture), but the core idea — aggregate discarded information into retained
tokens — is portable.

### ToDRE (arXiv 2025) — Spatial Diversity via k-Center
Visual token pruning based only on importance scores leads to spatial clustering
(tokens concentrated in one region). Solves with a greedy k-center algorithm to
ensure retained tokens are spread across the image.

**Relevance:** Directly addresses our val entropy = -5.13 problem (only 0.16
nats above static crop = -4.97). Their greedy k-center is non-differentiable;
we need a differentiable spatial spread penalty on the soft selection
distribution.

### VFlowOpt (ICCV 2025) — Progressive Aggregation for Fine-grained Details
Achieves 10% token retention with 90% accuracy by progressively aggregating
pruned information guided by attention-based importance maps. Key finding:
single-step large compression loses high-frequency details (OCR, small objects);
progressive compression preserves them.

**Relevance:** We drop 75% of tokens in one step. Single-step compression risks
losing fine-grained details. VFlowOpt's insight motivates using multiple
diverse summary tokens (each specializing in different content) rather than a
single mean-pool summary.

---

## Priority 1 — Soft Token Recycling (+2-3% VQAv2 expected)

**Why:** We hard-drop 432 tokens. QG-VTC gets 75.31% with 72 tokens using soft
recycling — that's better than us with 144. The information in the 432 dropped
tokens (background context, scene structure, global layout) helps the LLM even
when not directly related to the selected region.

### Q1: TPN-score weighted sum vs cross-attention pooling?

**Short answer: cross-attention pooling with 4 learned slots is more
mathematically sound; score-weighted sum is the fast first implementation.**

TPN scores (s_N) optimize for *which tokens to select*, not for *how to
summarize what we dropped*. A token ranked 432nd (barely unselected) is
weighted highest by softmax(s_N * unsel_mask), but it may not be the best
representative of the unselected content — it's just the most "important"
unselected token. The summary computed this way is biased toward near-threshold
tokens and under-represents global context.

Cross-attention pooling with 4 learned slot vectors fixes this:
- Each slot learns to aggregate a different "aspect" of the unselected content
  (global layout, background texture, peripheral objects, color histogram)
- Slots specialize independently via gradient descent
- No score-dependency — the summary is optimized for LLM usefulness, not TPN selection

**Implementation plan — two stages:**

Stage A (fast, no new params): score-weighted single summary token

```python
# In tc_vitok_model.py, after gather_tokens()
v_selected = gather_tokens(v_fine, idx)          # [B, 144, D]

sel_mask = torch.zeros(B, N, device=v_fine.device)
sel_mask.scatter_(1, idx, 1.0)
unsel_mask = 1.0 - sel_mask                      # [B, N]

# Weight unselected by TPN scores (higher score = more informative summary)
unsel_weights = torch.softmax(
    s_N * unsel_mask - 1e9 * sel_mask, dim=-1
)                                                 # [B, N]
v_summary = (v_fine * unsel_weights.unsqueeze(-1)).sum(1, keepdim=True)
                                                  # [B, 1, D]

# LLM sees 145 tokens: 1 summary + 144 selected
v_input = torch.cat([v_summary, v_selected], dim=1)  # [B, 145, D]
```

Stage B (better, ~512K new params): cross-attention pooling with 4 slots

```python
# New module: UnselectedPooler
class UnselectedPooler(nn.Module):
    def __init__(self, d_model: int, n_slots: int = 4, n_heads: int = 4):
        super().__init__()
        self.slots = nn.Parameter(torch.randn(1, n_slots, d_model))
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, v_unselected: torch.Tensor) -> torch.Tensor:
        # v_unselected: [B, 432, D]
        B = v_unselected.shape[0]
        q = self.slots.expand(B, -1, -1)           # [B, 4, D]
        out, _ = self.cross_attn(q, v_unselected, v_unselected)
        return self.norm(out)                       # [B, 4, D]

# In forward():
v_unselected = complement_gather(v_fine, idx, N)  # [B, 432, D]
v_summary = self.unsel_pooler(v_unselected)        # [B, 4, D]
v_input = torch.cat([v_summary, v_selected], dim=1) # [B, 148, D]
```

**Recommended implementation order:** Stage A first (zero new params, can be
added without retraining — just re-run Phase 1 from Phase 0 checkpoint). If
+1.5% or more, proceed to Stage B.

### Q3: Single-step 75% compression and fine-grained detail loss

**Yes, this is a real risk.** Averaging 432 tokens smooths high-frequency
content. OCR tokens (text in images) and small object tokens have high
spatial-frequency patterns that get washed out by mean-pooling.

**Mitigations in Stage B (cross-attention pooling):**

1. **Multiple slots specialize**: 4 slots independently learn what to aggregate.
   One slot may specialize in high-frequency textures (OCR, small objects)
   without being averaged with low-frequency background slots.

2. **Score-aware attention mask**: During cross-attention, up-weight high-scored
   unselected tokens (they were "almost selected" — likely fine-grained
   important content) by adding a score bias to the attention logits:
   ```python
   # Add TPN score bias to cross-attention logits
   score_bias = unsel_scores.unsqueeze(1)  # [B, 1, 432]
   attn_bias = score_bias.expand(-1, n_slots, -1)  # [B, 4, 432]
   # Pass as attn_mask to nn.MultiheadAttention
   ```

3. **Frequency-split summary**: Compute two summary tokens per slot — one from
   low-scoring unselected tokens (global context) and one from high-scoring
   unselected tokens (fine-grained details). Concatenate → 8 summary tokens.
   This mirrors VFlowOpt's progressive aggregation in a single step.

---

## Priority 1b — Teacher Attention Distillation (+1-2% VQAv2 expected, denser TPN supervision)

**Source concept:** DynamicViT (ICCV 2021) adapted for VLMs.

**Why:** TPN currently learns only from sparse VQA cross-entropy via STE — one
bit of feedback per forward pass. Teacher distillation provides dense, per-token
supervision at every step. The teacher's answer-token attention (or hidden-state
similarity) to visual tokens gives a ground-truth "token importance" map that
directly tells TPN which spatial regions the LLM relied on.

**Memory:** Sequential execution (teacher no_grad first, free, then student with
grad). Peak per GPU = max(19GB teacher, 34GB student) = 34GB. Fits in 49GB with
15GB headroom.

**Implementation choice:** use last-layer hidden state cosine similarity (NOT raw
attention weights). Attention weights have LLaMA positional spikes (BOS, first
visual token). Hidden states are more stable and avoid the FlashAttention
compatibility issue (no need for `attn_implementation="eager"`).

```python
def extract_teacher_importance(llava, input_ids, attn_mask, n_visual=576,
                                answer_start=None):
    with torch.no_grad():
        out = llava(input_ids=input_ids, attention_mask=attn_mask,
                    output_hidden_states=True, use_cache=False)
    h = out.hidden_states[-1]                          # [B, seq_len, D]
    h_vis = h[:, :n_visual, :]                        # [B, 576, D]
    h_ans = h[:, answer_start:, :].mean(1, keepdim=True)  # [B, 1, D]
    return F.cosine_similarity(h_ans.expand_as(h_vis), h_vis, dim=-1).detach()
    # returns [B, 576] — importance of each visual token from teacher's perspective

def attention_distillation_loss(s_N, teacher_importance, T=2.0):
    p_t = F.softmax(teacher_importance / T, dim=-1)
    lp_s = F.log_softmax(s_N / T, dim=-1)
    return F.kl_div(lp_s, p_t, reduction='batchmean') * T**2
```

**Lambda:** lambda_distill = 0.2 (tunable 0.1-0.5). Run teacher with
`llava.disable_adapters()` to use base model without LoRA as teacher.

**Training overhead:** one extra frozen forward pass per step (seq_len=626, no
grad). At 3.34 it/s current speed, overhead ≈ 25-30% slower (~1.3× total time).

---

## Priority 2 — Fix GQA Training Data (+1-2% GQA expected)

**Why:** Phase 1 trained on VQAv2 only. `build_loaders()` reads a singular
`dataset` key; the `datasets: ["vqav2", "gqa"]` config list is silently ignored.
GQA's 148k images with spatial/relational questions would improve TPN routing
and GQA accuracy. VQAv2 accuracy should also improve slightly from more diverse
training distribution.

**Implementation:** Fix `build_loaders()` in `scripts/train.py` to handle list:

```python
# Current (broken for list):
dataset_name = cfg.data.dataset  # KeyError if config uses "datasets"

# Fix:
dataset_names = cfg.data.get("datasets", [cfg.data.get("dataset", "vqav2")])
datasets = [build_single_dataset(name, cfg, split) for name in dataset_names]
loader = DataLoader(ConcatDataset(datasets), ...)
```

---

## Priority 3 — Decision Gate: Wait for Full Eval Results

**Critical metric:** GQA spatial accuracy (TC-ViTok) vs GQA spatial (image-only
TPN baseline). This tells us whether TPN learned query-conditioned routing.

| GQA spatial gap (TC vs image-only) | Interpretation | Action |
|---|---|---|
| > 3% | Routing works | Soft recycling + GQA data fix → retrain Phase 1 |
| 1–3% | Routing weak | Also increase lambda_diversity, longer Phase 0 |
| < 1% | Routing failed | Spatial diversity penalty (Priority 4) first |

---

## Priority 4 — Differentiable Spatial Diversity Penalty (if routing weak)

**Why:** Val entropy -5.13 is only 0.16 nats above static crop (-4.97). TPN has
residual spatial bias — it still favors certain image regions regardless of
query. ToDRE's k-center addresses this but is non-differentiable.

**Differentiable spatial spread penalty** on the soft selection distribution:

```python
# In losses.py, add spatial_spread_loss()
def spatial_spread_loss(mask_soft: torch.Tensor, N: int = 576, grid: int = 24):
    """Penalize spatial clustering in soft selection distribution.

    Maximizes variance of the expected selection position, forcing TPN to
    spread selections across the image rather than clustering in one region.

    Args:
        mask_soft: Soft selection weights [B, N] (e.g., sigmoid(s_N / tau))
        N: Total tokens (576)
        grid: Grid side length (24 for 24x24)

    Returns:
        loss: -variance (minimize to maximize spatial spread)
    """
    device = mask_soft.device
    positions = torch.arange(N, device=device).float()
    row = positions // grid   # [N] — row coordinate
    col = positions % grid    # [N] — col coordinate

    # Normalize to probability distribution
    p = mask_soft / (mask_soft.sum(dim=-1, keepdim=True) + 1e-8)  # [B, N]

    # Expected position
    E_row = (p * row).sum(dim=-1)   # [B]
    E_col = (p * col).sum(dim=-1)   # [B]

    # Variance of selection position (high = spread, low = clustered)
    var_row = (p * (row - E_row.unsqueeze(-1))**2).sum(dim=-1)  # [B]
    var_col = (p * (col - E_col.unsqueeze(-1))**2).sum(dim=-1)  # [B]

    # Return negative variance (minimize loss = maximize spread)
    return -(var_row + var_col).mean()
```

Add to `compute_total_loss()` with weight `lambda_spread` (suggested: 0.005).

**Why this is better than ToDRE's k-center:** Fully differentiable via m_soft.
Gradient d(spread)/d(s_N) flows to all 576 TPN scores. ToDRE's greedy k-center
is post-hoc and non-differentiable — it cannot train TPN weights.

---

## Priority 5 — Minor Improvements (implement together with P1/P2)

| Change | Expected gain | Complexity | Rationale |
|---|---|---|---|
| LoRA rank 32→64 | ~0.5–1% | Low (config) | More LLM adaptation capacity |
| Phase 1 steps 50k→70k with LR decay | ~0.5% | Low | Model converged at 40k; cosine decay from 50k may extract more |
| Entropy schedule: 0.02→0.005 over last 20k steps | Prevents over-regularization | Low | Entropy pressure unnecessary when model converged |
| lambda_diversity decay after step 30k | Cleaner gradient signal | Low | Diversity loss oscillates; decay stabilizes late training |
| Learnable position bias in TPN cross-attention | +0.3–0.5% | Medium | Encodes spatial structure of visual tokens into TPN |

---

## Recommended Retraining Order (after eval results confirm routing works)

```
Step 1: Implement Stage A soft recycling (no new params)
        + Fix GQA data loading
        → Retrain Phase 1 from runs/phase0_v4/best.ckpt
        → Expected: 74–75% VQAv2

Step 2: If still below 76%, implement Stage B cross-attention pooling (4 slots)
        + Add spatial spread penalty (lambda_spread=0.005) if GQA gap < 3%
        + LoRA rank 32→64
        → Retrain Phase 1 again
        → Expected: 75–77% VQAv2

Step 3: Minor tuning (entropy decay schedule, longer training, LR warmup)
        → Expected: 76–77% VQAv2
```

---

---

## 2026 Literature: Three Advanced Concepts (Reviewed)

### Concept 1 — Adaptive Token Budget / Dynamic k
**Source:** "Small VLMs are Smart Compressors" (arXiv 2026)

**Verdict: DEFER to Phase 2. Do not implement in Phase 1.5.**

Per-sample dynamic k breaks LLaVA batching (variable sequence lengths →
variable-length attention). Per-batch dynamic k is feasible but requires a
FLOPs/latency penalty to prevent k from collapsing to k_max. Without that
penalty the TPN has no incentive to use fewer tokens. Adds complexity without
addressing the core 72.4% → 76% gap. Revisit as a Phase 2 efficiency ablation.

### Concept 2 — REINFORCE / Policy Gradient Instead of STE
**Source:** MARC / RL Token (2026)

**Verdict: REJECT. STE did not fail; REINFORCE is the wrong tool.**

REINFORCE's variance scales as O(action_space²) = O(576²) ≈ 330k. For sparse
VQA-Score rewards (0/1 per question), the signal-to-noise ratio would be
unusable. STE with tau-independent correction already provides dense gradient
to all k=144 selected positions per step. The 72.4% gap is from hard-dropping
432 tokens (no recycling), not from routing failure. Fix the architecture first.

REINFORCE is the right tool only when: (a) no differentiable proxy exists, and
(b) the true reward is uncorrelated with CE loss. Neither holds here.

### Concept 3 — Debiased / Orthogonal Token Rebalancing
**Source:** Efficient3D (April 2026)

**Verdict: IMPLEMENT — folds into Soft Recycling as the correct pooling strategy.**

Naive score-weighted summary of unselected tokens computes a summary correlated
with near-threshold tokens (ranked 145-180), which resemble the already-selected
set. This provides redundant information to the LLM. Orthogonal deflation removes
the selected subspace from unselected tokens before cross-attention pooling,
ensuring the 4 summary slots carry complementary (not redundant) information.

Implementation: project unselected tokens orthogonal to mean(v_selected) before
passing to UnselectedPooler. O(432 × D) extra compute — negligible.

---

## Phase 1.5 Architecture: Complete Forward Pass

Incorporating: Orthogonal Soft Recycling + Spatial Diversity Penalty + Debiased
Pooling. New trainable params: ~512K (UnselectedPooler only).

```python
def forward(self, images, questions, answer_strings=None):

    # 1. Encode
    v_coarse = coarse_encoder(images)          # [B, 49, 384]  DINOv2
    v_fine   = fine_encoder(images)            # [B, 576, 1024] CLIP
    q_embed  = text_encoder(questions)         # [B, 512]

    # 2. TPN importance scores (instance-normalized)
    s_N = tpn(v_coarse, q_embed)               # [B, 576]

    # 3. Selection
    if training:
        idx = (s_N + gumbel_noise).topk(k).indices   # [B, k] diverse
    else:
        idx = s_N.topk(k).indices                     # [B, k] deterministic

    # 4. Soft masks for regularization losses
    m_soft   = sigmoid(s_N / tau)              # [B, 576] for spread loss
    mask_ste = build_ste_mask(s_N, idx, tau)   # [B, 576] for entropy loss

    # 5. Gather selected + tau-independent STE correction
    v_selected = gather_tokens(v_fine, idx)    # [B, k, D]
    sc = gather(s_N, idx); v_selected *= (1 + sc - sc.detach()).unsqueeze(-1)

    # 6. Orthogonal soft recycling (NEW)
    v_unselected = complement_gather(v_fine, idx)     # [B, 576-k, D]
    v_sel_mean   = v_selected.mean(dim=1)             # [B, D]
    proj = (v_unselected * v_sel_mean.unsqueeze(1)).sum(-1, keepdim=True) \
           / (v_sel_mean.norm(dim=-1)**2 + 1e-8).unsqueeze(-1)
    v_unsel_orth = v_unselected - proj * v_sel_mean.unsqueeze(1)  # [B, 432, D]
    v_summary = unsel_pooler(v_unsel_orth)            # [B, 4, D]  ← 4 slots

    # 7. Compose LLM prefix: [4 summary | k selected]
    v_input = cat([v_summary, v_selected], dim=1)     # [B, k+4, D]

    # 8. LLM forward
    logits, seq_labels = _llava_forward(v_input, questions, answer_strings)

    # 9. Losses
    if training:
        loss_spread = spatial_spread_loss(m_soft)     # NEW: -(var_row+var_col)
        losses = compute_total_loss(
            logits, seq_labels, s_N, idx, k,
            mask_ste=mask_ste, q_embed=q_embed,
            lambda_entropy=λ_ent,    # 0.02 → 0.005 decay
            lambda_diversity=λ_div,  # 0.1  → 0.02 decay at step 30k
            lambda_spread=λ_spread,  # NEW: 0.005
        )
    return logits, seq_labels, losses
```

### New module: UnselectedPooler (with 2D sinusoidal PE)

CLIP-ViT bakes 1D positional info into v_fine features, but orthogonal deflation
partially destroys it. Reinject 2D sinusoidal PE using known unselected indices
before cross-attention — zero new parameters.

```python
def sinusoidal_2d_pe(flat_indices: torch.Tensor, D: int, grid: int = 24) -> torch.Tensor:
    """2D sinusoidal positional encoding for unselected token positions.
    flat_indices: [B, N-k] — flat grid indices of unselected tokens
    Returns:      [B, N-k, D]
    """
    row  = (flat_indices // grid).float()
    col  = (flat_indices % grid).float()
    half = D // 4
    div  = torch.exp(torch.arange(half).float() * (-math.log(10000.0) / half))
    pe   = torch.zeros(*flat_indices.shape, D)
    pe[..., 0:half]        = torch.sin(row.unsqueeze(-1) * div)
    pe[..., half:2*half]   = torch.cos(row.unsqueeze(-1) * div)
    pe[..., 2*half:3*half] = torch.sin(col.unsqueeze(-1) * div)
    pe[..., 3*half:]       = torch.cos(col.unsqueeze(-1) * div)
    return pe.to(flat_indices.device)

class UnselectedPooler(nn.Module):
    """4-slot cross-attention pooler for orthogonal unselected tokens.
    Injects 2D sinusoidal PE so summary slots know WHERE background content is."""
    def __init__(self, d_model=1024, n_slots=4, n_heads=4):
        super().__init__()
        self.slots     = nn.Parameter(torch.randn(1, n_slots, d_model))
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm       = nn.LayerNorm(d_model)

    def forward(self, v_unsel_orth, unsel_indices):   # [B, 432, D], [B, 432]
        B = v_unsel_orth.shape[0]
        # Inject spatial awareness into orthogonal features
        pe  = sinusoidal_2d_pe(unsel_indices, v_unsel_orth.shape[-1])  # [B, 432, D]
        kv  = v_unsel_orth + pe                                         # [B, 432, D]
        q   = self.slots.expand(B, -1, -1)                             # [B, 4, D]
        out, _ = self.cross_attn(q, kv, kv)
        return self.norm(out)                                           # [B, 4, D]
```

### spatial_spread_loss: annealing schedule

λ_spread = 0.005 is NOT safe at full strength throughout training. At tau=0.1
(end of training), the spread gradient is ~13% of VQA gradient for OCR-style
local-focus queries — enough to hurt accuracy on "what word is written on the
tiny sign?" questions.

Fix: anneal spread loss to zero in the last 20k steps. Active early to prevent
static crop, gone late so VQA gradient can freely zoom in.

```python
# In curriculum.py
def get_lambda_spread(step: int, max_val: float = 0.005,
                      decay_start: int = 30000, decay_end: int = 50000) -> float:
    if step >= decay_end:   return 0.0
    if step < decay_start:  return max_val
    frac = (step - decay_start) / (decay_end - decay_start)
    return max_val * (1.0 - frac)
```

### Component summary

| Component | Status | New params |
|---|---|---|
| TPN, encoders, STE correction | Unchanged | 0 |
| m_soft / mask_ste / diversity loss | Unchanged | 0 |
| Orthogonal deflation | NEW | 0 |
| 2D sinusoidal PE for unselected indices | NEW | 0 |
| UnselectedPooler (4 slots, H=4) | NEW | ~512K |
| spatial_spread_loss with annealing | NEW | 0 |
| _llava_forward (k+4 prefix) | Minor edit | 0 |

---

---

## Future Work (Paper Discussion Section)

### Adversarial Representation Matching (GAN / Professor Forcing)

**Verdict: Implement after Phase 1.5. Cite in Discussion now.**

The natural extension of KL distillation is adversarial distribution matching: train
a lightweight discriminator to classify whether LLM answer-token hidden states came
from the full 576-token teacher or the 148-token compressed student. The TPN acts as
generator, trained to fool the discriminator — forcing the compressed prefix to produce
LLM representations indistinguishable from the full-prefix representations.

**Why not Phase 1.5:** Five existing loss terms + Gumbel-STE + 32-layer gradient path
= unstable training environment. GANs require alternating updates, gradient penalties
(WGAN-GP, λ_gp=10), and delayed start (after step 20k). Implementation risk high,
marginal gain over KL distillation.

**If implemented:** discriminator operates on mean-pooled answer-token hidden states
[B, 4096] — same shape for teacher and student regardless of visual prefix length.
Use WGAN-GP for stability. lambda_gan = 0.01-0.05. Start only after VQA + KL
distillation loss have converged (~step 20k).

**Paper framing:** *"A natural extension is adversarial representation matching, where
the compressed visual prefix is trained to be indistinguishable from the full-token
prefix in LLM hidden state space — a Professor Forcing analog for visual compression."*

### Adaptive Token Budget (Dynamic k per batch)

See Concept 1 in the 2026 Literature section above. Requires a FLOPs penalty signal
to prevent k from collapsing to k_max. Good Phase 2 efficiency ablation.

---

## What NOT to Do

- **Don't change TPN architecture capacity** — 887K params is sufficient; the
  problem is what happens to the 432 dropped tokens, not TPN expressivity
- **Don't increase k beyond 144** — defeats the paper's compute efficiency claim
- **Don't retrain Phase 0** — it's converged; the gains are all in Phase 1
- **Don't implement spatial spread penalty before seeing eval results** — val
  entropy -5.13 may be sufficient if GQA spatial >> image-only TPN baseline
- **Don't implement Stage B (cross-attention pooler) before Stage A** — verify
  the gain from recycling concept before adding params

---

## Eval Checklist (paste results here when done)

```
TC-ViTok k=144 (Phase 1 v3 — runs/phase1_v3/phase1/best.ckpt):
  VQAv2 VQA-Score:      _______  (target: 76.0%)
  GQA overall:          _______  (target: 61.0%)
  GQA spatial:          _______  (target: 59.0%)
  GQA global:           _______  (target: 62.0%)
  POPE macro-F1:        _______  (target: 84.0%)
  Eval status:          [OK] run completed 2026-04-10 — read runs/eval_results/*.json

Random-drop baseline:
  VQAv2:                _______
  Eval status:          [FAIL] — needs rerun

Image-only TPN baseline:
  VQAv2:                _______
  GQA spatial:          _______

Key gap (TC vs image-only, GQA spatial): _______  → routing verdict: WORKS / WEAK / FAILED
```

---

## Phase 1.5 Implementation Status (as of 2026-04-12)

All components from the "Phase 1.5 Architecture: Complete Forward Pass" section above
are **fully implemented and running**. Smoke test passed (3 steps, 4 optimizer groups,
no OOM, no crash). Full training running at `runs/phase1p5_v1`.

### What Was Built (Session 5: 2026-04-11)

| Component | File | Status |
|---|---|---|
| `complement_gather(V, idx)` | `tc_vitok_model.py` | ✅ |
| `sinusoidal_2d_pe(flat_indices, D)` | `tc_vitok_model.py` | ✅ |
| `UnselectedPooler(d_model=1024, n_slots=4, n_heads=4)` | `tc_vitok_model.py` | ✅ |
| Orthogonal deflation in `forward()` | `tc_vitok_model.py` | ✅ |
| `_get_teacher_importance()` — full prompt + hidden state cosine sim | `tc_vitok_model.py` | ✅ |
| `spatial_spread_loss(mask_soft, N, grid)` | `losses.py` | ✅ |
| `attention_distillation_loss(s_N, teacher_imp, T=1.0, teacher_scale=10.0)` | `losses.py` | ✅ |
| `compute_total_loss` extended with spread/distill params | `losses.py` | ✅ |
| `get_lambda_spread(step, max_val, decay_start, decay_end)` | `curriculum.py` | ✅ |
| `unsel_pooler` param group in optimizer | `optimizer.py` | ✅ |
| Teacher pass in `train_step` | `trainer.py` | ✅ |
| `datasets: ["vqav2", "gqa"]` via ConcatDataset | `scripts/train.py` | ✅ |
| `configs/phase1p5.yaml` | configs/ | ✅ |
| `tests/test_phase1p5.py` — 52 tests | tests/ | ✅ |

### Bugs Fixed During Phase 1.5 Implementation

| ID | Issue | Fix |
|---|---|---|
| BUG-38 | Padding corruption in teacher importance — h[i, ans_start:] averages padding zeros | Store seq_lens before padding; slice h[i, ans_start:seq_lens[i]] |
| BUG-39 | Wrong LLaVA prompt structure — visual tokens prepended before BOS | Split: prefix_embeds ("USER: ") + v_proj_full + suffix_embeds ("\nQ\nASSISTANT:"); track visual_starts |
| BUG-40 | VRAM spikes — lm_out (~19GB) held alive during student forward | Explicit del chain: all_embeds → padded_embeds/masks → lm_out → h |
| BUG-41 | Teacher distribution flat — T=2.0, cosine_sim std≈0.1 → p_top only 1.16× uniform | teacher_scale=10.0, T=1.0; maps cosine_sim std 0.1→1.0; ~19× concentration verified |
| BUG-42 | configs/phase1p5.yaml wrong checkpoint path | Fixed: runs/phase0_v4/best.ckpt → runs/phase0_v4/phase0/best.ckpt |
| BUG-43 | diagnose_routing.py used nonexistent from_pretrained(); wrong image sizes | Fixed to constructor + load_state_dict; dual preprocessors (98px DINOv2, 336px CLIP) |

---

## Phase 1.5 Training Analysis (live — step ~21k of 50k)

### Val Metrics vs Phase 1 v3 Baseline

| Step | p1.5 val_vqa | p1v3 val_vqa | Gap | Interpretation |
|------|-------------|-------------|-----|----------------|
| 5k   | 0.5129      | 0.4411      | +0.072 | UnselectedPooler injecting noise |
| 10k  | 0.4951      | 0.4053      | +0.090 | Gap widening — distill too strong |
| 15k  | 0.5001      | 0.4032      | +0.097 | Regression (pooler warmup plateau) |
| 20k  | 0.4610      | 0.3617      | +0.099 | Recovery; distill declining |

**Phase 1.5 converges 35% slower** than Phase 1 v3 (0.052 vs 0.079 VQA CE improvement
over steps 5k→20k). Root causes:
1. UnselectedPooler randomly initialized — adds noise prefix tokens for first ~15k steps
2. lambda_distill=0.2 contributes 43% of VQA gradient magnitude, pulling TPN toward
   teacher's full-token routing preferences rather than 144-token VQA-optimal routing
3. GQA training data harder distribution

**Positive signs:**
- Distill IS declining: 1.2-1.4 (step 11k) → 0.7-1.1 (step 21k)
- Spread IS increasing: -107 → -113 (more spatial diversity, above uniform -95.8)
- No collapse, no NaN, entropy stable at -5.83 train

### Plan of Action

**Step 25k is the decision point.** Monitor these specific numbers:

| Metric | Green: add distill decay, extend 100k | Yellow: extend, watch 30k | Red: restart |
|--------|---------------------------------------|--------------------------|--------------|
| val_vqa | < 0.42 | 0.42–0.45 | > 0.45 |
| Gap vs p1v3 | < 0.085 (narrowing) | 0.085–0.095 | > 0.095 (widening) |
| distill train | < 0.70 | 0.70–0.85 | > 0.85 (plateau) |

**If Green or Yellow:** Add distill annealing in trainer.py:
```python
# After step 20k, decay lambda_distill from 0.2 to 0.02 over next 30k steps
lambda_distill = max(0.02, 0.2 * (1.0 - (step - 20000) / 30000))
```
Extend max_steps to 100k. Tau stays at 0.1 floor from step 50k onward (fine).

**If Red:** Restart from `runs/phase0_v4/phase0/best.ckpt` with:
- lambda_distill: 0.2 → 0.05
- pooler_warmup_steps: 5000 (freeze TPN, only train pooler + LoRA first 5k steps)
- max_steps: 100000

---

## Novelty Analysis (Reviewer #2 Assessment — 2026-04-12)

### Component Novelty Breakdown

| Component | Novelty | Prior art |
|---|---|---|
| Query-conditioned TPN routing | **Standard engineering** | Q-Former (BLIP-2), InstructBLIP |
| Gumbel-Softmax + STE | **Standard engineering** | Jang et al. 2017, Maddison et al. 2017 |
| Teacher distillation for token selection | **Standard engineering** | EViT (Liang et al. 2022), DynamicViT |
| Cross-attention 4-slot pooler | **Standard engineering** | Slot Attention, Perceiver, Q-Former |
| **Orthogonal deflation of complement set** | **NOVEL — Golden Ticket** | No prior paper found |
| Spatial spread loss (variance penalty) | Incremental contribution | ToDRE addresses same problem differently |

### Publication Odds (assuming results hit target)

| Venue | Odds | Condition |
|---|---|---|
| CVPR/ICCV/ECCV main | 12-18% | Must beat FastV + PruMerge, not just random drop |
| NeurIPS | 15-22% | Theoretical framing of orthogonal deflation + principled ablations |
| ECCV/EMNLP workshop | 75%+ | Results don't need to be SOTA |

### Required Ablations to Defend Against Reviewer #2

1. **Orthogonal deflation ablation**: Table with (a) no background pooling, (b) pooling without deflation, (c) pooling with deflation. If (b)→(c) gives ≥ 0.5% VQAv2, the claim is defensible.
2. **Comparison to FastV, LLaVA-PruMerge, ToMe-for-LLaVA at same 144-token budget** — must beat them, not just random drop. Run these baselines BEFORE writing paper.
3. **Query-conditioning ablation**: GQA spatial accuracy (TC-ViTok) vs (image-only TPN baseline). If gap > 3%, the query-conditioning narrative holds.

---

## Phase 1.5 v4 Design: Slot Diversity via Soft Orthogonality Penalty

**Status:** Planned — implement after v3 validates the no-warmup training regime.

### Problem Statement (from v3 design analysis)

The current UnselectedPooler has a gap: the 4 output slots can attend to identical
background tokens and produce correlated/redundant summaries. The existing deflation
only removes the selected-token subspace from the INPUTS — it does not enforce that
the 4 OUTPUT slots are diverse.

### Why Hard Gram-Schmidt is Wrong (two traps, user-identified 2026-04-12)

**Trap 1 — LayerNorm explosion on zero-residuals:**
If Slot 2 is collinear with Slot 1, GS zeroes its residual. LayerNorm backward
amplifies the gradient by 1/sqrt(var + eps) ≈ 316× (at var≈0). In BF16 (max ~65504),
three such amplifications → overflow → NaN loss. The instability is in the backward
pass, not the forward.

**Trap 2 — Self-reinforcing slot death:**
A zeroed slot → LLM adapts to ignore it → ∂L/∂slot_k ≈ 0 → no gradient reaches
the attention weights → slot stays dead. The "gradient-informed" claim breaks down
once the LLM adapts to ignore the dead slot. Hard GS hides redundancy from the LLM
rather than resolving it.

**Why iterative deflation (matching pursuit) is also wrong:**
- 4× slower (slots are sequential, not parallelizable)
- Gradient instability: ∂L/∂slot_1 flows through residuals from slots 2,3,4
- BF16 drift: repeated subtraction `residual = residual - proj` accumulates fp16 error

### The Correct Solution: Soft Cosine Orthogonality Loss

Add `lambda_slot_orth × L_slot_orth` to the total training loss, where:

```python
def slot_orthogonality_loss(v_summary: torch.Tensor) -> torch.Tensor:
    """Penalize cosine similarity between all pairs of slot output vectors.

    Operates on pre-LayerNorm slot embeddings to avoid LN scale interaction.
    In high-D space (D=1024), random vectors are nearly orthogonal by default
    (E[cosine_sim] ≈ 0), so the penalty starts near zero and fires only when
    slots genuinely over-specialize on the same background features.

    Args:
        v_summary: Pre-norm slot outputs [B, n_slots, D]

    Returns:
        loss: mean squared cosine similarity across all slot pairs (scalar)
    """
    # L2-normalize each slot
    v_norm = F.normalize(v_summary, dim=-1)          # [B, n_slots, D]
    # Gram matrix of cosine similarities
    gram = torch.bmm(v_norm, v_norm.transpose(1, 2)) # [B, n_slots, n_slots]
    # Only off-diagonal pairs (i ≠ j) — mask diagonal
    n = v_summary.shape[1]
    mask = ~torch.eye(n, dtype=torch.bool, device=gram.device)
    return gram[:, mask].pow(2).mean()
```

**Properties:**
- Always-nonzero gradient: drives attention weights to diverge continuously
- No near-zero vectors: operates on L2-normalized vectors → no LN explosion risk
- Numerically safe in BF16: F.normalize clamps implicitly, pow(2) is bounded
- Starts near 0 for random slots (high-D orthogonality), grows only when slots correlate
- Fully symmetric: treats all 4 slots equally (no slot hierarchy like GS)
- Paper claim: "trained with explicit orthogonality penalty ensuring slot diversity"

### Implementation Plan for v4

**losses.py:** Add `slot_orthogonality_loss(v_summary)` function.

**tc_vitok_model.py:** Expose pre-norm slot outputs in `forward()` return dict:
```python
# In UnselectedPooler.forward():
out, _ = self.cross_attn(q, kv, kv)
v_summary_prenorm = out                  # return this for loss
return self.norm(out), v_summary_prenorm # [B, n_slots, D] each
```

**trainer.py:** Compute and add slot orthogonality loss when `lambda_slot_orth > 0`.

**configs/phase1p5_v4.yaml:**
```yaml
lambda_slot_orth: 0.01    # small — slots start near-orthogonal in 1024-D space
```

### Hyperparameter: lambda_slot_orth = 0.01

Rationale: At slot diversity = 0 (perfectly orthogonal), loss = 0. At slot collapse
(all 4 slots identical), loss = 1.0 (mean sq cosine sim of 1.0). Contribution at
collapse: 0.01 × 1.0 = 0.01 << VQA ≈ 0.5. The penalty fires hard enough to prevent
collapse but never dominates. If early training shows val_vqa degrading, reduce to 0.005.

### Ablation for Paper

| Variant | lambda_slot_orth | Expected val_vqa |
|---|---|---|
| v3 (baseline) | 0 | TBD |
| v4 (soft penalty) | 0.01 | TBD |
| v4 no-deflation ablation | 0.01 | TBD (lower — proves deflation matters) |

If (v4 - v3) ≥ 0.3% VQAv2, the slot diversity mechanism has a defensible paper contribution.
