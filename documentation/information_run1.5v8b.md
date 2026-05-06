# TC-ViTok — Complete Deep-Dive Reference
*Viva preparation document. Covers architecture, loss design, training procedure, all scripts/configs/files.*

---

## Table of Contents
1. [The Core Problem](#1-the-core-problem)
2. [Full Architecture](#2-full-architecture)
3. [Every Component in Detail](#3-every-component-in-detail)
4. [Loss Functions](#4-loss-functions)
5. [Training Phases](#5-training-phases)
6. [Co-Adaptation Problem & Version History](#6-co-adaptation-problem--version-history)
7. [File Map — Every File Explained](#7-file-map--every-file-explained)
8. [Configs — Every Key Explained](#8-configs--every-key-explained)
9. [Scripts — Commands, Arguments, and What They Do](#9-scripts--commands-arguments-and-what-they-do)
10. [Exact tmux Training Commands](#10-exact-tmux-training-commands)
11. [Exact tmux Evaluation Commands](#11-exact-tmux-evaluation-commands)
12. [Monitoring & Debugging](#12-monitoring--debugging)
13. [Key Numbers to Know](#13-key-numbers-to-know)

---

## 1. The Core Problem

**Standard LLaVA-1.5** feeds all **576 visual tokens** (from CLIP ViT-L/14 at 336×336px with 14×14 patches = 24×24 = 576 patches) into the 7B language model for every image, regardless of the question. This is:
- **Computationally wasteful**: the LLM's attention is O(n²) in sequence length. Halving tokens → 4× attention savings.
- **Informationally noisy**: "What color is the car?" doesn't need sky/background tokens. Irrelevant tokens fill the context window, diluting the signal.

**TC-ViTok's solution**: A lightweight **Tokenization Policy Network (TPN)** scores all 576 tokens and selects the **144 most question-relevant** ones. The other 432 ("background") tokens are compressed into **4 summary slots** by the UnselectedPooler. The LLM sees **148 tokens** instead of 576 — a 3.9× reduction.

**Why this works**: Natural images have high spatial redundancy. The question signal is strong enough to bias the selection: "person in the image" → faces/bodies; "text on the sign" → sign patches; "background color" → background patches.

**The challenge**: Making token selection differentiable (it's a discrete argmax operation), stable during training (many failure modes discovered), and actually better than naive random dropping.

---

## 2. Full Architecture

```
INPUT: image (any res) + question (string)
                │
                ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 1 — FROZEN ENCODERS (no gradient, no weight update)      │
│                                                                  │
│  image  ──[resize 98×98]──► DINOv2-S ──► v_coarse [B, 49, 384] │
│  image  ──[resize 336×336]─ CLIP ViT-L/14 ──► v_full [B,576,1024]│
│  question string ──────────► CLIP text enc ──► q_embed [B, 512] │
└─────────────────────────────────────────────────────────────────┘
                │ v_coarse, q_embed
                ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 2 — TPN (trainable, 887K params)                         │
│                                                                  │
│  Cross-attn: q_embed → v_coarse                                 │
│  Score MLP → 49 coarse scores → bilinear upsample → 576 scores  │
│  Instance norm → Gumbel-Sigmoid STE → idx [B, 144]              │
│                                       s_N [B, 576]              │
└─────────────────────────────────────────────────────────────────┘
                │ idx, s_N
                ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 3 — GATHER + POOL (Phase 1.5)                            │
│                                                                  │
│  gather(v_full, idx) ──────────────────► v_sel   [B, 144, 1024] │
│  complement_gather(v_full, idx) ──────► v_unsel  [B, 432, 1024] │
│                                                                  │
│  Orthogonal Deflation:                                          │
│    v_unsel -= proj(v_unsel → mean(v_sel))   [remove redundancy] │
│                                             ► v_unsel_orth      │
│  UnselectedPooler (trainable, ~512K params):                    │
│    4 slot queries cross-attn v_unsel_orth + 2D pos enc          │
│                                             ► summary [B, 4, 1024]│
│                                                                  │
│  v_input = concat([summary | v_sel]) ────► [B, 148, 1024]       │
└─────────────────────────────────────────────────────────────────┘
                │ v_input
                ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 4 — LLaVA-1.5 + LoRA (partial training)                  │
│                                                                  │
│  mm_projector (trainable): [B, 148, 1024] → [B, 148, 4096]      │
│  Concat with tokenized question → full sequence                  │
│  LoRA-adapted Llama-7B → logits [B, seq_len, vocab]              │
│  L_VQA = cross_entropy(logits, answer_token_labels)              │
└─────────────────────────────────────────────────────────────────┘
```

**Gradient flow**: VQA loss → LoRA → mm_projector → v_input → v_sel → (additive STE correction) → s_N → TPN weights

---

## 3. Every Component in Detail

### 3.1 Coarse Visual Encoder (DINOv2-S)
- **What**: ViT-Small with patch size 14, pretrained self-supervised on ImageNet-22k via DINO
- **Input**: 98×98 RGB image (deliberately small to be fast)
- **Why 98px**: 98 / 14 = 7 patches per side → 7×7 = **49 spatial tokens** at 384-dim. This is the coarse spatial map for the TPN.
- **Why DINOv2 not CLIP**: DINOv2 is self-supervised so it preserves fine-grained spatial locality (DINO's self-distillation objective forces patch-level features to be meaningful). CLIP text-image contrastive training is less spatially precise.
- **Frozen**: always. Never updated. Its job is just to give the TPN a cheap semantic spatial map.
- **Parameters**: 22M (none trainable)

### 3.2 Text Encoder (CLIP ViT-B/32)
- **What**: CLIP's text branch (ViT-B/32 text transformer)
- **Input**: question string, tokenized via CLIP tokenizer
- **Output**: EOS token embedding → **512-dim** vector representing the question semantics
- **Why EOS**: The EOS (end-of-sequence) token in CLIP accumulates global meaning of the full sentence — it's the standard CLIP text embedding used for image-text similarity.
- **Frozen**: always.
- **Why CLIP not BERT**: The TPN cross-attends text over visual tokens. CLIP was trained to align text and image features in the same embedding space, so CLIP text features are naturally better suited to attend over CLIP-like visual features (DINOv2 and CLIP ViT-L are both in similar visual representation spaces).

### 3.3 Full Visual Encoder (CLIP ViT-L/14)
- **What**: CLIP's large vision transformer, ViT-Large with patch size 14
- **Input**: 336×336 RGB image → 24×24 = **576 spatial tokens** at **1024-dim**
- **Why 336px**: LLaVA-1.5 natively uses 336px CLIP ViT-L/14. We reuse the same encoder so our visual features are already aligned to LLaVA's LLM.
- **Frozen**: always. We never retrain it — its features are already what LLaVA's LLM expects.
- **Parameters**: 304M (none trainable)

### 3.4 Tokenization Policy Network (TPN) — `tpn.py`
- **What**: Lightweight cross-attention network, **887K params**, **trainable**
- **Why introduced**: We need a *question-conditioned* scorer for visual tokens. A simple image-only saliency map (e.g., DINOv2 attention) ignores the question. We need to ask "which patches matter for *this* question?"

**Internal architecture step by step:**

**Step 1 — Linear Projections**
```
q_embed [B, 512] ──► W_Q ──► Q [B, 1, 384]    (question as single query)
v_coarse [B, 49, 384] ──► W_K ──► K [B, 49, 384]
v_coarse [B, 49, 384] ──► W_V ──► V [B, 49, 384]
```
The text query (512-dim) is projected into the same 384-dim space as the DINOv2 features so they can be compared.

**Step 2 — Multi-Head Cross-Attention (4 heads)**
```
attn_logits = Q @ K^T / sqrt(d_k)    [B, 4, 1, 49]
alpha = softmax(attn_logits)          [B, 4, 1, 49]   ← where does the question attend?
ctx = alpha @ V                       [B, 1, 384]      ← attended visual context
ctx = W_O(ctx) + LayerNorm            [B, 384]
alpha_avg = mean(alpha, dim=heads)    [B, 49]          ← spatial attention map
```
After this: `ctx` = what visual information is relevant to the question; `alpha_avg` = spatial attention weights showing which coarse regions the question looks at.

**Step 3 — Score MLP**
```
combined = concat([ctx, alpha_avg])   [B, 384+49=433]
combined = LayerNorm(combined)
s_M = MLP(combined)                   [B, 49]          ← one score per coarse patch
```
The MLP takes both the attended context vector AND the raw attention weights — the context says "what" was important, the attention weights say "where."

**Step 4 — Bilinear Upsample 7×7 → 24×24**
```
s_M [B, 49] ──► reshape ──► [B, 1, 7, 7] ──► bilinear ──► [B, 1, 24, 24] ──► flatten ──► s_N [B, 576]
```
Coarse scores at 7×7 resolution are upsampled to 24×24 to match the full CLIP ViT-L/14 spatial resolution. Bilinear interpolation ensures smooth gradients.

**Step 5 — Instance Normalization**
```
s_N = (s_N - mean(s_N)) / std(s_N)   per sample, across the 576 positions
```
**Why**: Without this, Adam pushes scores to ±100 within ~1000 steps. At that scale, Gumbel noise (magnitude ≈ τ ≈ 1.0) is negligible → deterministic top-144 selection → same 576 tokens always chosen regardless of question (static crop). Two failed alternatives:
- `tanh(s_N)`: saturates at ±1 within 1000 steps → tanh'(x) ≈ 0.001 → gradient vanishes
- No normalization: scores diverge, Gumbel noise ineffective, collapse

Instance norm permanently anchors scores to mean=0, std=1. Since only relative ordering matters for topk selection, this loses no information. No learnable parameters → optimizer cannot tune it away.

**Step 6 — Gumbel-Sigmoid STE**
```python
g = -log(-log(Uniform(0,1)))         # Gumbel(0,1) noise
noisy_scores = s_N + g               # per-sample diversity
m_soft = sigmoid(noisy_scores / tau) # soft mask (has gradient)
idx = topk(noisy_scores, k=144)      # hard selection (no gradient)
m_hard = scatter(idx, 1.0)           # binary mask
mask_ste = m_hard - m_soft.detach() + m_soft   # STE trick
```
**STE trick**: `m_hard - m_soft.detach() + m_soft` = `m_hard` in the forward pass (m_soft cancels), but has gradient from m_soft in the backward pass. This allows gradients to flow through a discrete argmax.

**Why noisy scores for topk** (not raw scores): Raw scores always break ties the same way (lowest index first) → all batch samples select identical tokens → entropy = log(144) → collapse. Gumbel noise makes each sample's selection unique.

**Temperature τ**: As τ → 0, sigmoid becomes a step function → m_soft → m_hard → the STE becomes exact (forward = backward). As τ → 1, sigmoid is soft → smooth gradients. We anneal 1.0 → 0.1 so training starts smooth (exploration) and ends sharp (exploitation).

**Returns**: `idx [B, 144]` (gather indices), `s_N [B, 576]` (raw scores for loss), `mask_ste [B, 576]` (for entropy loss gradient)

### 3.5 Gather Operation — `gather_tokens` in `tc_vitok_model.py`
```python
idx_exp = idx.unsqueeze(-1).expand(-1, -1, d_v)   # [B, 144, 1024]
v_selected = torch.gather(v_full, dim=1, index=idx_exp)  # [B, 144, 1024]
```
Real compression — LLM input sequence goes from 576 → 144. `torch.gather` is non-differentiable w.r.t. `idx` (it's an integer index operation), which is why the STE exists.

### 3.6 Additive Tau-Independent STE — `tc_vitok_model.py:500`
```python
scores_at_selected = torch.gather(s_N, dim=1, index=idx)      # [B, 144]
correction = scores_at_selected - scores_at_selected.detach() # = 0 in forward
v_selected = v_selected * (1.0 + correction.unsqueeze(-1))    # identity in forward
```
**Why**: The standard STE (multiply by `mask_ste`) has gradient ∝ m_soft × (1 − m_soft) / τ. As τ → 0.1, m_soft saturates → gradient → 0 → TPN stops learning. This killed TPN after step 15k in early experiments.

**Additive STE**: The correction term is zero in the forward pass (cancels itself). In the backward pass, the gradient of `loss` w.r.t. `s_N[b, j]` = Σ_d (g_v[b,j,d] × v_selected[b,j,d]) — a dot product of the VQA loss gradient with the visual token's features. This is **tau-independent** and remains non-zero throughout training.

### 3.7 Complement Gather — `complement_gather` in `tc_vitok_model.py`
```python
# Build boolean mask: True = selected
sel_mask.scatter_(1, idx, True)
# All positions that are NOT selected
unsel_idx = all_positions[~sel_mask].reshape(B, N-k)   # [B, 432]
v_unsel = torch.gather(v_full, dim=1, index=idx_exp)   # [B, 432, 1024]
```
Returns the 432 unselected token features AND their spatial indices (needed for positional encoding and the distillation loss).

### 3.8 Orthogonal Deflation — `tc_vitok_model.py:519`
```python
v_sel_mean = v_selected.mean(dim=1)                    # [B, 1024] — centroid of selected
proj_coeff = dot(v_unsel, v_sel_mean) / ||v_sel_mean||² # [B, 432, 1]
v_unsel_orth = v_unsel - proj_coeff * v_sel_mean        # [B, 432, 1024]
```
**Why**: Without deflation, the unselected tokens that are near the selection boundary are very similar to the selected tokens (they're near-threshold). The pooler would spend its capacity summarizing near-redundant information. By projecting out the selected-token subspace, we force the pooler slots to capture information that is **orthogonal** (complementary) to what was selected. The 4 slots now summarize background information the LLM is not already getting from the 144 selected tokens.

### 3.9 UnselectedPooler — `tc_vitok_model.py:116`
- **What**: 4-slot cross-attention module, ~512K trainable params
- **Why introduced**: Phase 1 (no pooler) hard-drops 432 tokens. This creates an information gap — any information only in those 432 tokens is permanently lost. The pooler "soft-recycles" this information into 4 compact summary vectors.

**Architecture**:
```
query: self.slots [1, 4, 1024] (learned slot queries, expanded to batch)
key/value: v_unsel_orth + sinusoidal_2d_pe(unsel_indices)
→ MultiheadAttention(4 heads) → [B, 4, 1024]
→ LayerNorm → summary [B, 4, 1024]
```

**Positional encoding**: Before pooling, each unselected token's spatial position is re-injected via 2D sinusoidal PE. During orthogonal deflation, positional structure is preserved in the features (DINOv2/CLIP are position-encoded), but we add explicit positional signal so the pooler's attention can spatially organize: "attend to the top-left background" vs "attend to the bottom-right background."

**Sinusoidal 2D PE formula**:
```
row = flat_index // 24,  col = flat_index % 24
pe[0:D/4]   = sin(row × div),  pe[D/4:D/2]  = cos(row × div)
pe[D/2:3D/4] = sin(col × div), pe[3D/4:D]   = cos(col × div)
div = exp(-i × log(10000) / (D/4))   for i in [0, D/4)
```

**Orthogonal initialization of slot queries**:
```python
slots_data = torch.empty(4, 1024)
nn.init.orthogonal_(slots_data)   # 4 unit-norm orthogonal vectors
```
**Why**: With random init (scale ~0.02), projected Q vectors are near-zero → softmax(QK^T/√d) ≈ uniform for all 4 slots → all 4 slots attend identically → all 4 slots produce identical outputs → slot_orth loss = 1.0 → collapsed. Unit-norm orthogonal queries produce maximally diverse attention distributions from step 0, so slot_orth starts near 0.

**Returns**: `summary [B, 4, 1024]` (post-norm, fed to LLM), `v_summary_prenorm [B, 4, 1024]` (pre-norm, for slot_orth loss), `attn_weights [B, 4, 432]` (for residual_distill loss)

**LLM prefix**: `concat([summary | v_selected], dim=1)` → `[B, 148, 1024]`. The 4 summary slots come **first**, then the 144 selected tokens. LLaVA's LLM attends to all 148 tokens jointly.

### 3.10 LLaVA-1.5 + LoRA — `tc_vitok_model.py:332`
- **What**: LlavaForConditionalGeneration (LLaMA-7B + mm_projector + frozen CLIP ViT). We bypass LLaVA's internal ViT and inject our own visual tokens.
- **LoRA targets**: `q_proj`, `k_proj`, `v_proj`, `o_proj` — the 4 attention projections in all 32 LLM transformer layers
- **Phase 1 LoRA rank**: 32, alpha=64 → effective rank scaling = alpha/rank = 2
- **Phase 1.5 LoRA rank**: 64, alpha=128 → same effective scaling but 2× capacity
- **Why LoRA not full finetuning**: LoRA adds ~35M params total vs 7B in the LLM. Prevents catastrophic forgetting of pretrained knowledge while adapting to our visual token format.
- **mm_projector**: LLaVA's 2-layer MLP that projects 1024-dim visual tokens → 4096-dim LLM hidden dim. Trainable in Phase 1.
- **lora_warmfreeze_steps**: LoRA gradients are zeroed for the first 3000 steps. This forces the VQA loss to be reduced through TPN routing quality alone (not by LoRA compensating for poor selection). After step 3000, LoRA unfreezes and jointly optimizes.

### 3.11 Teacher Importance Map — `_get_teacher_importance` in `tc_vitok_model.py`
Used in Phase 1.5 to provide dense supervision for the TPN and pooler.

```python
# Run frozen LLaVA with all 576 tokens (no selection) under torch.no_grad()
teacher_out = frozen_llava(all_576_tokens, questions, answers)
h_ans = teacher_out.hidden_states[-1][:, answer_positions, :]   # [B, ans_len, H]
h_vis = teacher_out.hidden_states[-1][:, visual_positions, :]   # [B, 576, H]
# Per-token importance: how much does each visual token correlate with the answer?
mean_h_ans = h_ans.mean(dim=1)   # [B, H]
teacher_importance = cosine_similarity(h_vis, mean_h_ans)        # [B, 576]
```
This gives us a ground-truth "importance map" of which tokens a fully-informed model (seeing all 576) would find most relevant to the answer. The TPN is supervised to agree with this map, providing dense gradient signal at every step vs the sparse VQA signal.

---

## 4. Loss Functions

All losses are implemented in `src/tc_vitok/training/losses.py` and combined in `compute_total_loss`.

### 4.1 VQA Loss (L_VQA) — always active
```python
# Next-token prediction on answer tokens only
# Labels: -100 (ignore) for visual + question tokens; token IDs for answer tokens
loss = cross_entropy(logits, labels, ignore_index=-100)
```
**Why NTP not classification**: LLaVA is a generative model, not a 3000-class classifier. NTP trains it to generate the correct answer string token by token. This is the standard approach for instruction-following LLMs.

**val_vqa metric**: this is the NTP cross-entropy loss on the validation set (lower = better). It is NOT the actual VQA-Score % (higher = better) — val_vqa is just the training loss proxy. Actual VQA-Score is computed by `evaluate.py`.

**Handling NaN**: logits are cast to float32 and nan_to_num'd before cross-entropy. BF16 attention can produce NaN/±inf from magnitude overflow; this ensures the loss is always finite.

### 4.2 Entropy Regularization (L_ent) — always active
```python
m_bar = mask.mean(dim=0)       # [576] — batch-averaged selection frequency
p = m_bar / m_bar.sum()        # normalized probability distribution
H = -sum(p * log(p))           # Shannon entropy, max at log(576) ≈ 6.36
L_ent = -H                     # MINIMIZING this MAXIMIZES entropy
```
**Why**: Without this, the TPN quickly learns to always select the same spatial region (e.g., center of the image where answers are most frequent in VQAv2). Maximizing entropy forces the TPN to spread its selections across different spatial positions across the batch. Higher entropy = more query-conditioned selection (different questions select different regions). Gradient flows through `mask_ste` (not `idx`) to reach all 576 positions.

**Typical weight**: λ_ent = 0.01. Contributes ≈ −0.06 to total loss (loss_ent = −6.0 × 0.01).

### 4.3 Diversity Loss (L_div) — active in Phase 1
```python
# Pair adjacent batch samples
score_sim = cosine_similarity(scores[:half], scores[half:])  # [B/2] — mask similarity
q_sim = cosine_similarity(q_embed[:half], q_embed[half:])    # [B/2] — question similarity
L_div = mean(score_sim × (1 - q_sim))
```
**Why**: TPN might select the same patches regardless of question (e.g., always select salient objects, ignoring whether the question asks about them). This loss penalizes: two samples with *different* questions (q_sim low) but *similar* selections (score_sim high). If the selections are already different (question-conditioned), this loss is near zero. If questions are similar, similar selections are acceptable.

**Typical weight**: λ_div = 0.04 (Phase 1.5). Contributes ≈ 0.02 to total loss.

### 4.4 Boundary Margin Loss (L_boundary) — Phase 1.5+
```python
# Teacher's ranked tokens: k-th (last selected) and (k+1)-th (first rejected)
s_kth  = s_N[teacher's 144th-ranked token]   # TPN score at teacher's boundary-in
s_kp1  = s_N[teacher's 145th-ranked token]   # TPN score at teacher's boundary-out
L_boundary = relu(margin - (s_kth - s_kp1))  # hinge: zero if s_kth > s_kp1 + margin
```
**Why introduced**: The full KL distillation loss (L_distill, v1–v3) pushed all 576 TPN scores toward the teacher's full distribution — but instance-normalized scores sum to zero, so pushing all scores toward positive values collapses the distribution. The boundary loss only constrains **two tokens** — the teacher's k-th and (k+1)-th — with a hinge. This teaches "which side of the boundary each token is on" without perturbing the overall distribution shape. Compatible with instance normalization.

**margin = 0.1**: In instance-norm space (std=1), this means the k-th token's score must exceed the (k+1)-th by at least 0.1σ. Typical loss ≈ 0.0–0.1.

**Typical contribution**: 0.05 × 0.05 = 0.0025 at typical hinge value.

### 4.5 Slot Orthogonality Loss (L_slot_orth) — Phase 1.5 v4–v7
```python
v_norm = F.normalize(v_summary_prenorm, dim=-1)    # unit-norm slot outputs
gram = bmm(v_norm, v_norm.transpose(1,2))          # [B, 4, 4] cosine similarity matrix
off_diagonal = gram[:, mask]                        # exclude diagonal (self-similarity)
L_slot_orth = mean(off_diagonal ** 2)              # squared cosine similarity
```
**Why**: Cross-attention with identical slot queries produces identical outputs. Penalizing cosine² similarity between slot outputs encourages each slot to specialize in attending to a different subset of the 432 background tokens. Loss ∈ [0, 1]: 0 = all orthogonal, 1 = all identical. With D=1024 random vectors, E[cos²] ≈ 1/1024 ≈ 0.001, so random slots would score near 0.

**Pre-norm vs post-norm**: Operates on pre-LayerNorm outputs because LayerNorm changes the cosine geometry (it normalizes feature magnitudes differently per slot), making cosine² less interpretable as a true similarity measure.

**Disabled in v8b** (lambda_slot_orth=0.0) because pooler is frozen — no point penalizing parameters that don't update.

### 4.6 Residual Pooler Distillation Loss (L_residual_distill) — Phase 1.5 v5–v7
```python
# Teacher's importance over the 432 unselected tokens, sorted by rank
teacher_residual = teacher_dist.gather(1, unsel_idx)   # [B, 432]
_, rank_order = teacher_residual.sort(descending=True) # [B, 432]

# Divide into 4 quartiles: top-25%, 25-50%, 50-75%, bottom-25%
# Slot 0 must attend to teacher's top-25% of unselected tokens
# Slot 1 must attend to 25-50%, etc.
for slot_i in range(4):
    band_positions = rank_order[:, band_start:band_end]
    target = teacher_residual.at(band_positions)  # teacher prob in this band
    slot_attn_band = pooler_attn[slot_i].at(band_positions)
    L_per_slot = KL(slot_attn_band || target)
```
**Why rank-banded (v6 fix)**: v5 used naive KL over all 432 tokens aggregated across all 4 slots. Since aggregation was a sum, every slot received the **same gradient** → all 4 slots learned the same distribution → slot_orth rose from 0.001 to 0.45 → val_vqa regressed from 0.4230 to 0.4650. By assigning each slot a **different quartile** of the teacher's residual importance ordering, each slot receives a structurally different gradient signal. Diversity is enforced at the architectural level.

**Teacher scale = 10.0**: Teacher importance is cosine similarity ∈ [−1,1] with typical std ≈ 0.1. The student s_N has std=1 (instance-normed). Applying softmax to cosine sims directly produces near-uniform distributions (top token only 1.16× higher than uniform). Multiplying by 10 maps teacher std 0.1 → 1.0, matching s_N's scale. At matched scales, the teacher distribution is 19× concentrated at the most important token — a clear, non-degenerate training signal.

**Disabled in v8b** (lambda_residual_distill=0.0) because pooler is frozen.

### 4.7 Total Loss Formula
```
L_total = L_VQA 
        + λ_ent    × L_ent 
        + λ_div    × L_div 
        + λ_boundary × L_boundary 
        + λ_slot_orth × L_slot_orth 
        + λ_residual_distill × L_residual_distill
        + λ_sparsity × L_sparsity   (0 in v4+ since instance norm makes L1 constant)
        + λ_spread × L_spread       (0 in v4+, was used in v1-v3)
        + λ_distill × L_distill     (0 in v4+, replaced by boundary)
```

---

## 5. Training Phases

### Phase 0 — TPN Warmup (`configs/phase0.yaml`)
**Goal**: Teach TPN to score tokens meaningfully before introducing the co-adaptation problem.

| What trains | What's frozen |
|---|---|
| TPN (887K) | All encoders, LLM, LoRA, mm_projector |

**Duration**: 10,000 steps  
**τ**: Fixed at 1.0 (no annealing — maximally soft selection, smooth gradients)  
**λ_max**: 0.0 (L1 sparsity disabled — instance norm makes L1 constant anyway)  
**Checkpoint saved to**: `runs/phase0_v4/phase0/best.ckpt`

**Why freeze LoRA**: If LoRA were active, it could compensate for bad TPN selections by learning to handle whatever tokens the TPN accidentally picks. The TPN would never learn to be selective. With LoRA frozen, the only way to reduce L_VQA is to improve selection quality.

### Phase 1 — Joint Finetuning (`configs/phase1.yaml`)
**Goal**: Co-train TPN + LoRA + mm_projector to converge to a jointly optimal solution.

| What trains | What's frozen |
|---|---|
| TPN + LoRA + mm_projector | All encoders, LLM base weights |

**Duration**: 50,000 steps  
**τ**: 1.0 → 0.1 (linear anneal)  
**lora_warmfreeze_steps**: 3000 — LoRA frozen for first 3k steps  
**Phase 1 v3 result**: val_vqa = **0.3334** at step 50k (baseline to beat)  
**Checkpoint**: `runs/phase1_v3/phase1/best.ckpt`

### Phase 1.5 — Joint Finetuning + UnselectedPooler (`configs/phase1p5_v8b.yaml`)
Same as Phase 1 but adds the UnselectedPooler and all Phase 1.5 losses.

| What trains | What's frozen |
|---|---|
| TPN + LoRA + mm_projector + pooler | All encoders, LLM base weights |

**v8b specifics**: pooler frozen (lr=0), loaded from v8 step-45k checkpoint, tau_max=0.595 (continuation)

### Phase 2 — Domain Transfer (`configs/phase2.yaml`)
Transfer-learn to TextVQA (OCR-heavy, different from VQAv2 natural scenes). Fine-tune on TextVQA questions. Not yet run.

---

## 6. Co-Adaptation Problem & Version History

The core instability in Phase 1.5: **the pooler and LoRA cannot be trained simultaneously without oscillation.**

**Mechanism**:
1. Pooler output shifts → LLM input prefix changes
2. LoRA must re-adapt to the new prefix
3. LoRA weights shift → gradient to pooler changes
4. Pooler must re-adapt → oscillation loop

Period of oscillation ∝ 1/lr_pooler. This created two distinct failure regimes:

| Version | lr_pooler | Result | Killed at |
|---------|-----------|--------|-----------|
| v1–v3 | — | Architecture bugs (slot collapse, gradient issues) | various |
| v4 | 5e-4 | fast 5k-step regression cycles | step 30k |
| v5 | 5e-4 | rank-banded distill fixed slot collapse, same lr regression | step 25k |
| v6 | 5e-4 | boundary+slot_orth, still 5k regression | step 20k |
| v7 | 5e-4 | reduced lambdas, still 5k regression | step 15k |
| v8 | **2e-5** | 5k cycles gone, but slow 10-15k cycles; best val_vqa=0.3663 at step 45k | step 63k |
| v8b | **0.0** | pooler frozen, no oscillation; catapult at step 5k (fresh optimizer) | **RUNNING** |

**v8b: Fresh Optimizer Catapult**  
Loading `best.ckpt` (step 45k weights) without optimizer state → AdamW starts with zero running variance → effective LR = base_LR / sqrt(ε) (huge) → parameter drift from checkpoint optimum → val_vqa jumps to 0.4540 at step 5k. This is expected, not a bug. By step 15k, AdamW builds variance estimates and converges to stable high-LR update magnitudes.

**Decision table (step 15k val_vqa)**:

| val_vqa | Action |
|---------|--------|
| < 0.40 | Continue — frozen pooler working, LoRA converging |
| 0.40–0.42 | Continue with managed expectations |
| > 0.42 | Kill v8b; start no-pooler rank-64/128 run |

---

## 7. File Map — Every File Explained

### Directory Structure
```
TC-ViTok/
├── src/tc_vitok/           ← Python package (installed via pip -e .)
│   ├── models/             ← Neural network architecture
│   ├── training/           ← Loss, optimizer, trainer, curriculum
│   ├── data/               ← Datasets, caching, spatial labels
│   └── evaluation/         ← Metrics, diagnostics, collapse detection
├── configs/                ← YAML hyperparameter files
├── scripts/                ← CLI entry points (train, eval, utilities)
├── tests/                  ← Unit and integration tests
└── runs/                   ← Output directory (checkpoints, logs)
```

---

### `src/tc_vitok/models/`

#### `encoders.py`
**Purpose**: Wrappers around the three frozen pre-trained encoders.  
**Why here**: All encoder code in one place; each returns a fixed-shape tensor contract. Separating from tc_vitok_model.py keeps the main model class focused on the TC-ViTok logic.

- `CoarseVisualEncoder`: DINOv2-S (timm), `img_size=98`. Returns `[B, 49, 384]`. Frozen on init.
- `TextEncoder`: CLIP ViT-B/32 (HuggingFace), extracts EOS token. Returns `[B, 512]`. Frozen.
- `FullVisualEncoder`: CLIP ViT-L/14 (HuggingFace), pixel_values input. Returns `[B, 576, 1024]`. Frozen.

#### `tpn.py`
**Purpose**: The TPN — cross-attention scorer + Gumbel STE selection.  
**Why separate file**: The TPN is the core novel contribution; it has its own paper section. Keeping it in its own file makes it easy to test, cite, and ablate in isolation.

Key: `TokenizationPolicyNetwork.forward()` returns `(idx, s_N, mask_ste)`.

#### `masking.py`
**Purpose**: The differentiable top-k selection primitive.  
**Why separate file**: The STE is a reusable mathematical building block. Multiple loss functions (entropy, diversity) need it. Tests can verify it independently.

Key functions:
- `gumbel_sigmoid_ste(scores, k, tau)` → `(mask_ste, idx)`
- `compute_spatial_entropy(mask)` → scalar H
- `compute_diversity_loss(mask_1, mask_2, q_embed_1, q_embed_2)` → scalar

#### `tc_vitok_model.py`
**Purpose**: The full TC-ViTok model — integrates all components and handles both training and inference.  
**Why one file**: This is the system integration layer. It owns the forward pass contract that the trainer calls.

Key classes/functions:
- `gather_tokens(V, idx)` → `[B, k, D]`
- `complement_gather(V, idx)` → `([B, N-k, D], [B, N-k])`
- `sinusoidal_2d_pe(indices, D)` → `[B, L, D]`
- `UnselectedPooler` (class) — 4-slot cross-attn pooler with orthogonal init
- `StubLLM` (class) — tiny transformer for testing without 7B params
- `TCViTokModel` (class) — main model; `forward()` runs the full pipeline
  - `_init_llava()` — loads LLaVA + LoRA
  - `_llava_forward()` — NTP forward: project → tokenize → label → LLM → logits
  - `_get_teacher_importance()` — teacher cosine-sim importance map
  - `set_training_phase()` — freezes/unfreezes components per phase
  - `forward_cached()` — backward-compat wrapper using pre-computed features

---

### `src/tc_vitok/training/`

#### `losses.py`
**Purpose**: All loss functions, individually and combined.  
**Why separate file**: Loss logic is pure math with no model state. It can be unit-tested and compared to paper equations independently. Keeping it separate prevents `trainer.py` from becoming a 1000-line monolith.

- `vqa_loss(logits, labels)` — CE with nan_to_num guard
- `entropy_loss(mask)` — returns -H (minimize → maximize entropy)
- `diversity_loss(mask_1, mask_2, q_embed_1, q_embed_2)` — hinge on mask similarity
- `boundary_margin_loss(s_N, teacher_importance, k, margin)` — hinge on k-th boundary
- `slot_orthogonality_loss(v_summary_prenorm)` — mean cos² off-diagonal
- `residual_pooler_distill_loss(pooler_attn, teacher_importance, unsel_idx)` — rank-banded KL
- `compute_total_loss(...)` — weighted sum; returns dict of all component losses

#### `curriculum.py`
**Purpose**: All the training schedules (tau, lambda, LR).  
**Why separate file**: Schedules are pure functions of `step` — no model state needed. Easy to test and plot. Keeps trainer loop clean.

- `get_tau(step, T, tau_min, tau_max)` — linear anneal
- `get_lambda(step, T_warmup, lambda_max)` — linear warmup
- `get_lambda_spread(step)` — flat then linear decay to 0
- `get_lr_schedule(step, warmup_steps, total_steps, lr_base)` — linear warmup + cosine decay

#### `optimizer.py`
**Purpose**: Creates AdamW with per-component param groups.  
**Why separate file**: Optimizer creation is complex enough (handles stub vs real LLM, handles lr_pooler=0 case, handles old vs new LLaVA architecture for finding LoRA params) that it warrants its own file and tests.

- `create_optimizer(model, lr_tpn, lr_lora, lr_mlp, lr_pooler, phase)` — returns AdamW
- Param groups: TPN | LoRA | mm_projector | UnselectedPooler (if present)
- If `lr_pooler=0.0`: pooler params ARE in the optimizer but with lr=0 → no update
- If `lr_pooler=None`: falls back to `lr_mlp`

#### `trainer.py`
**Purpose**: The main training loop — orchestrates everything.  
**Why separate file**: The training loop contains DDP logic, checkpointing, validation, logging — none of which is model-specific. It calls the model's `forward()` and `losses.compute_total_loss()` as black boxes.

Key methods:
- `__init__()` — builds optimizer, sets phase, initializes DDP state
- `train()` — outer loop over steps; handles DDP barrier sync for val/checkpoint
- `train_step(batch)` — single gradient step: forward → loss → backward → clip → optimizer
- `validate()` — eval loop, rank-0 only, returns `{val_loss, val_vqa, ...}`
- `save_checkpoint()` / `load_checkpoint()` — saves model + optimizer state

---

### `src/tc_vitok/data/`

#### `datasets.py`
**Purpose**: PyTorch Dataset implementations for all benchmarks.  
**Why here**: Data loading logic is cleanly separated from model/training logic. Datasets know how to read files and apply transforms; the model doesn't.

- `DummyVQADataset` — random tensors, for pipeline smoke tests
- `VQAv2Dataset` — reads VQAv2 JSON annotations + COCO images; returns both 98px (DINO) and 336px (CLIP) crops; top-3129 answer vocab
- `GQADataset` — reads GQA balanced questions + scene graph images
- `collate_fn` — pads variable-length sequences in a batch
- `DINO_TRANSFORM` — resize 98×98, ImageNet normalize
- `CLIP_TRANSFORM` — resize 336×336, CLIP normalize

**Retry loop for corrupt JPEGs**: If a JPEG fails to decode, tries up to 5 adjacent samples (BUG-11 fix). Without this, a DataLoader worker crashes → SIGABRT.

#### `caching.py`
**Purpose**: Pre-compute and cache DINOv2 + CLIP text features to HDF5.  
**Why**: DINOv2 and CLIP text encoders are frozen — running them at every training step wastes compute. Cache them once → 2× speedup. Full ViT-L/14 features (2.4MB/sample) are NOT cached due to disk size.

- `build_cache(dataset, cache_path, ...)` — runs encoders, writes v_coarse + q_embed to HDF5
- `CachedFeatureDataset` — reads cached features, runs only ViT-L/14 live

#### `spatial_labels.py`
**Purpose**: Classify VQA questions as "spatially specific" vs general.  
**Why**: The paper claims TC-ViTok is especially good on spatially specific questions ("left side", "corner", etc.). This module labels questions for stratified evaluation.

#### `__init__.py`
Exports the public API of the data subpackage.

---

### `src/tc_vitok/evaluation/`

#### `metrics.py`
**Purpose**: VQA metric implementations.

- `normalize_answer(s)` — lowercase, remove punctuation, remove articles
- `vqa_score(predicted, ground_truth)` — soft accuracy: min(#annotators who said predicted/3, 1.0). 10 annotators, partial credit.
- `gqa_accuracy` — exact match after normalization
- `pope_f1` — binary F1 for hallucination eval (yes/no questions)

#### `collapse_detection.py`
**Purpose**: Detect TPN collapse during training (always selecting same tokens).  
Checks: entropy of selection distribution, variance of mask across batch samples.

#### `pareto.py`
**Purpose**: Plot accuracy vs token-budget Pareto frontier for paper figures.

#### `spatial_eval.py`
**Purpose**: Stratify eval metrics by question spatial specificity.

---

### `configs/`

All YAML files use OmegaConf with `defaults: [base]` to inherit from `base.yaml`.

#### `base.yaml`
**Purpose**: Default values for ALL config keys. Every other config only overrides what changes.  
**Why important**: OmegaConf merges base + override, so any key not in the phase-specific config falls back to base.yaml. The trainer reads `config["training"]["key"]` — if a key is missing from the phase config but present in base.yaml, it is automatically available.

Key sections:
- `model.k: 144` — token budget
- `model.tpn.*` — TPN architecture (n_heads=4, d_model=384, etc.)
- `model.lora.*` — LoRA rank/alpha/targets
- `training.*` — all hyperparameters
- `data.*` — paths via `${oc.env:DATA_ROOT,data}` (reads env var or falls back to `data/`)
- `logging.*` — wandb, intervals, output_dir
- `evaluation.*` — benchmarks list, token budgets to sweep

#### `phase0.yaml`
Overrides: `phase: 0`, `max_steps: 10000`, `tau_min: 1.0` (no anneal), `lambda_max: 0.0`, `lambda_entropy: 0.01`, `trainable_components: [tpn]`

#### `phase1.yaml`
Overrides: `phase: 1`, `max_steps: 50000`, `tau: 1.0→0.1`, `lr_tpn: 3e-3`, `lora_warmfreeze_steps: 3000`, `datasets: [vqav2, gqa]`

#### `phase1p5_v8.yaml`
Adds: `lr_pooler: 2e-5`, `lambda_boundary: 0.05`, `lambda_slot_orth: 0.02`, `lambda_residual_distill: 0.02`, `lora_rank: 64`

#### `phase1p5_v8b.yaml` (CURRENT)
Changes from v8: `lr_pooler: 0.0`, `lambda_slot_orth: 0.0`, `lambda_residual_distill: 0.0`, `tau_max: 0.595`, `max_steps: 55000`, `load_from: runs/phase1p5_v8/phase1/best.ckpt`

---

### `scripts/`

#### `train.py`
**Purpose**: CLI entry point for training. Reads config, builds model + datasets, instantiates trainer, calls `trainer.train()`.  
**Why not trainer.py directly**: `train.py` handles CLI argument parsing, DDP init (`torch.distributed.init_process_group`), environment setup (HF_HUB_OFFLINE=1), and config merging. `trainer.py` is a pure class that assumes these are already done.

#### `launch_multigpu.sh`
**Purpose**: DDP launcher wrapper around `torchrun`. Sets all necessary environment variables (NCCL, HF offline, OMP threads), kills leftover GPU processes from crashed runs, creates log file.  
**Why not just torchrun directly**: The env vars are critical (missing `HF_HUB_OFFLINE=1` causes HTTP timeout NCCL desync on multi-GPU; missing `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` causes SIGABRT instead of Python traceback).

#### `evaluate.py`
**Purpose**: Full VQA evaluation — loads checkpoint, runs on val split, computes VQA-Score %, GQA accuracy, POPE F1. Outputs JSON with all results.  
**Why separate from train.py**: Evaluation can be run independently on any checkpoint, with different benchmarks, without re-running training. It also supports baseline variants (`--baseline random_drop`, `--baseline image_only`).

#### `cache_features.py`
**Purpose**: Pre-compute DINOv2 + CLIP text features and save to HDF5. Run once before training for speed.

#### `extract_tpn_checkpoint.py`
**Purpose**: Extract only TPN weights from a full checkpoint (useful for Phase 0 → Phase 1 handoff without the 7B LLM weights).

#### `visualize_masks.py`
**Purpose**: Plot heatmaps of which tokens TPN selects on example images. Generates figures for paper and debugging.

#### `diagnose_routing.py`
**Purpose**: Check if TPN actually routes differently for different questions. If TPN selects the same 144 tokens regardless of question, something is wrong (static crop / entropy collapse).

#### `test_core.py`, `test_nccl.py`, `test_pipeline.py`
Quick smoke tests: DDP communication, pipeline end-to-end, GPU availability. Run before a training launch to catch environment issues.

---

## 8. Configs — Every Key Explained

### The Most Important Training Keys

```yaml
training:
  max_steps: 55000          # Total gradient steps (not epochs)
  batch_size: 4             # Per-GPU batch size
  grad_accum_steps: 16      # effective batch = 4 × 16 × 2 GPUs = 128

  # Temperature curriculum
  tau_max: 0.595            # Starting temperature (1.0 = new run; 0.595 = continuation)
  tau_min: 0.1              # Ending temperature (sharpest selection)

  # Sparsity regularization (disabled in v4+ — instance norm makes L1 constant)
  lambda_max: 0.0
  T_warmup: 5000            # Steps to warm lambda 0 → lambda_max

  # Regularization weights (Phase 1.5 v8b)
  lambda_entropy: 0.01      # Maximize spatial diversity (−H weight)
  lambda_diversity: 0.04    # Query-conditioning penalty
  lambda_boundary: 0.05     # Teacher boundary hinge loss weight
  lambda_slot_orth: 0.0     # Slot diversity (disabled — pooler frozen in v8b)
  lambda_residual_distill: 0.0  # Rank-banded KL (disabled — pooler frozen in v8b)
  boundary_margin: 0.1      # Hinge margin in instance-norm space

  # LoRA warm-freeze: freeze LoRA for first N steps (Phase 1 only)
  lora_warmfreeze_steps: 0  # 0 = off (v8b loads from step 45k where LoRA was already warm)

  # Pooler warm-up: freeze TPN for first N steps (used in early v4 experiments)
  pooler_warmup_steps: 0    # 0 = off

  # Learning rates (per parameter group)
  lr_tpn: 3e-3              # TPN: highest LR (small network, 887K)
  lr_lora: 2e-4             # LoRA: careful finetuning
  lr_mlp: 5e-4              # mm_projector: medium
  lr_pooler: 0.0            # UnselectedPooler: 0 = frozen (v8b key setting)

  weight_decay: 0.01
  max_grad_norm: 1.0
  mixed_precision: "bf16"   # BF16 on RTX 6000 Ada; avoids NaN from FP16 overflow

checkpoint:
  load_from: "runs/phase1p5_v8/phase1/best.ckpt"  # Load weights (not optimizer state)
  strict: false              # Allow missing/extra keys (enables Phase 0 → Phase 1 transfer)

logging:
  wandb_run_name: "phase1p5_v8b_frozen_pooler"
  eval_interval: 5000       # Run validation every N steps
  max_val_batches: 100      # Limit val to 100 batches (~55s) vs full 13K (~2h)
  log_interval: 100         # Log training metrics every N steps
  save_interval: 2000       # Save checkpoint every N steps
```

### Config Inheritance (OmegaConf)

```yaml
# In any phase config:
defaults:
  - base                    # Merges base.yaml first, then overrides below
```

Any key in the phase config **overwrites** the same key in base.yaml. Any key NOT in the phase config uses base.yaml's value. This is why you can write a minimal `phase1p5_v8b.yaml` (only ~30 lines) that still has all 50+ config keys available at runtime.

### Environment Variable Resolution

```yaml
data_root: ${oc.env:DATA_ROOT,data}
# → reads env var DATA_ROOT; falls back to "data/" if not set
```

On the remote server: `export DATA_ROOT=~/tc-vitok-data`, then OmegaConf resolves all paths like `${data.data_root}/vqav2/...` to `/home/user/tc-vitok-data/vqav2/...`.

---

## 9. Scripts — Commands, Arguments, and What They Do

### `train.py` — Full Argument Reference

```
--config PATH        (REQUIRED) Path to YAML config file
--output DIR         Output directory for checkpoints and logs (default: "runs")
--resume PATH        Resume from checkpoint (overrides config.checkpoint.load_from)
--wandb-project NAME WandB project name (default: "tc-vitok")
--max-steps N        Override max training steps (useful for quick tests: --max-steps 5)
--run-both-phases    After Phase 0, auto-run Phase 1 from configs/phase1.yaml
--device DEVICE      Override device (default: auto-detected cuda)
--stub               Use StubLLM instead of real LLaVA (no 7B model needed, fast)
--dummy-data         Use DummyVQADataset instead of real VQAv2 (no data needed)
--cache-path PATH    Path to HDF5 feature cache (overrides config cache_dir)
--no-wandb           Disable WandB logging (for debugging / offline testing)
```

**Examples**:
```bash
# Quick smoke test (2 steps, no GPU memory, no data needed):
python scripts/train.py \
    --config configs/phase0.yaml \
    --output /tmp/test \
    --stub --dummy-data --no-wandb --max-steps 2

# Full Phase 0 training:
python scripts/train.py \
    --config configs/phase0.yaml \
    --output runs/phase0_v5 \
    --wandb-project tc-vitok

# Resume from checkpoint:
python scripts/train.py \
    --config configs/phase1p5_v8b.yaml \
    --output runs/phase1p5_v8b \
    --resume runs/phase1p5_v8/phase1/best.ckpt
```

### `launch_multigpu.sh` — Full Argument Reference

All arguments after `--config` and `--output` are **passed through to train.py** exactly.

```bash
bash scripts/launch_multigpu.sh \
    --config CONFIG_YAML \          # required
    --output OUTPUT_DIR \           # required
    [--stub] \                      # passed to train.py
    [--dummy-data] \                # passed to train.py
    [--no-wandb] \                  # passed to train.py
    [--max-steps N] \               # passed to train.py
    [--resume CKPT_PATH]            # passed to train.py

# Environment variables:
NUM_GPUS=2          # override GPU count (default: auto-detected)
DATA_ROOT=/path     # dataset root (wired to HF_HOME, TRANSFORMERS_CACHE, etc.)
LOG_DIR=/path       # where to write log file (default: ~/TC-ViTok/logs/)
NCCL_DEBUG=INFO     # verbose NCCL logging (default: WARN)
```

**What the script does internally**:
1. Kills leftover GPU Python/torchrun processes (crash cleanup)
2. Counts GPUs with `torch.cuda.device_count()`
3. Sets `PYTHONPATH=src/`, `PYTHONUNBUFFERED=1`, `OMP_NUM_THREADS=1`
4. Sets NCCL env vars (`NCCL_P2P_DISABLE=1`, `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`, `NCCL_TIMEOUT=1800`)
5. Creates log file at `logs/train_YYYYMMDD_HHMMSS.log`
6. Calls `torchrun --standalone --nproc_per_node=N scripts/train.py $@`
7. Redirects all output to log file

### `evaluate.py` — Full Argument Reference

```
--checkpoint PATH    (REQUIRED) Path to trained checkpoint .ckpt
--benchmarks LIST    Space-separated list: vqav2 gqa pope textvqa (default: vqav2 gqa pope)
--data-root PATH     Root directory of datasets (override DATA_ROOT env var)
--output PATH        JSON output file for results (default: runs/eval_results/results.json)
--baseline TYPE      Evaluation variant: random_drop | image_only (ablations)
--spatial-eval       Enable spatial-specificity stratification of results
--k N               Token budget override (default: from checkpoint config, 144)
--batch-size N       Eval batch size (default: 8)
--num-workers N      DataLoader workers (default: 4)
--no-wandb           Disable WandB eval logging
```

---

## 10. Exact tmux Training Commands

### Phase 0 Training
```bash
tmux new-session -d -s phase0_v5
tmux send-keys -t phase0_v5 \
  "cd ~/TC-ViTok && bash scripts/launch_multigpu.sh \
   --config configs/phase0.yaml \
   --output runs/phase0_v5" Enter
# Detach: Ctrl-b d
# Reattach: tmux attach -t phase0_v5
```

### Phase 1 Training (from Phase 0 checkpoint)
```bash
tmux new-session -d -s phase1_v4
tmux send-keys -t phase1_v4 \
  "cd ~/TC-ViTok && bash scripts/launch_multigpu.sh \
   --config configs/phase1.yaml \
   --output runs/phase1_v4" Enter
# Note: configs/phase1.yaml already has:
#   checkpoint:
#     load_from: "runs/phase0/best.ckpt"
```

### Phase 1.5 v8b (current run)
```bash
# v8b is already running in tmux session phase1p5_v8b
tmux attach -t phase1p5_v8b
# Detach: Ctrl-b d

# To start fresh if killed:
tmux new-session -d -s phase1p5_v8b
tmux send-keys -t phase1p5_v8b \
  "cd ~/TC-ViTok && bash scripts/launch_multigpu.sh \
   --config configs/phase1p5_v8b.yaml \
   --output runs/phase1p5_v8b" Enter
```

### New Phase 1 Final Run (no pooler, rank 64/128)
```bash
tmux new-session -d -s phase1_final
tmux send-keys -t phase1_final \
  "cd ~/TC-ViTok && bash scripts/launch_multigpu.sh \
   --config configs/phase1_final.yaml \
   --output runs/phase1_final" Enter
```

### Dry-run Smoke Test (2 steps, 30 seconds, no data needed)
```bash
tmux new-session -d -s smoke_test
tmux send-keys -t smoke_test \
  "cd ~/TC-ViTok && bash scripts/launch_multigpu.sh \
   --config configs/phase0.yaml \
   --output /tmp/smoke_test \
   --stub --dummy-data --no-wandb --max-steps 2" Enter
```

### Kill and restart a run
```bash
# Kill
tmux kill-session -t phase1p5_v8b

# Check GPU is clear before restarting
nvidia-smi

# Restart
tmux new-session -d -s phase1p5_v8b
tmux send-keys -t phase1p5_v8b \
  "cd ~/TC-ViTok && bash scripts/launch_multigpu.sh \
   --config configs/phase1p5_v8b.yaml \
   --output runs/phase1p5_v8b" Enter
```

---

## 11. Exact tmux Evaluation Commands

### Full VQAv2 + GQA + POPE Evaluation
```bash
tmux new-session -d -s eval_v8
tmux send-keys -t eval_v8 \
  "cd ~/TC-ViTok && python scripts/evaluate.py \
   --checkpoint runs/phase1p5_v8/phase1/best.ckpt \
   --benchmarks vqav2 gqa pope \
   --data-root ~/tc-vitok-data \
   --output runs/eval_results/phase1p5_v8_k144.json" Enter
# Attach to monitor: tmux attach -t eval_v8
```

### Evaluate Phase 1 v3 baseline (get actual VQA-Score %)
```bash
tmux new-session -d -s eval_v3_baseline
tmux send-keys -t eval_v3_baseline \
  "cd ~/TC-ViTok && python scripts/evaluate.py \
   --checkpoint runs/phase1_v3/phase1/best.ckpt \
   --benchmarks vqav2 \
   --data-root ~/tc-vitok-data \
   --output runs/eval_results/phase1_v3_vqav2.json" Enter
```

### Random-drop ablation (paper ablation A2)
```bash
tmux new-session -d -s eval_baseline
tmux send-keys -t eval_baseline \
  "cd ~/TC-ViTok && python scripts/evaluate.py \
   --checkpoint runs/phase1_v3/phase1/best.ckpt \
   --baseline random_drop \
   --benchmarks vqav2 gqa \
   --data-root ~/tc-vitok-data \
   --output runs/eval_results/random_drop_k144.json" Enter
```

### Read eval results
```bash
# On the server, after evaluation finishes:
cat runs/eval_results/phase1p5_v8_k144.json | python3 -m json.tool
# or
python3 -c "import json; d=json.load(open('runs/eval_results/phase1p5_v8_k144.json')); print(d)"
```

---

## 12. Monitoring & Debugging

### Monitor training progress
```bash
# Tail log (strip carriage returns from tqdm):
tail -f ~/TC-ViTok/logs/train_20260414_090703.log | tr '\r' '\n' | grep "Step"

# Val checkpoints only:
grep "val" ~/TC-ViTok/logs/train_20260414_090703.log

# Watch loss components:
grep "Step" ~/TC-ViTok/logs/train_20260414_090703.log | tail -50
```

### Check checkpoints
```bash
ls -lh runs/phase1p5_v8b/phase1/          # list checkpoints
ls -lh runs/phase1p5_v8/phase1/best.ckpt  # v8 best (step 45k)
```

### GPU memory
```bash
nvidia-smi                                 # snapshot
watch -n 2 nvidia-smi                     # live
```

### Sync with local machine
```bash
git -c credential.helper= pull origin main   # on server
```

### tmux quick reference
```bash
tmux ls                       # list sessions
tmux attach -t SESSION        # reattach
Ctrl-b d                      # detach without killing
Ctrl-b [                      # scroll mode (q to exit)
tmux kill-session -t SESSION  # kill
```

---

## 13. Key Numbers to Know

| Fact | Value |
|------|-------|
| Total visual tokens | 576 (24×24 grid, CLIP ViT-L/14 at 336px) |
| Selected tokens | 144 (k = 25% of 576) |
| Unselected tokens | 432 |
| Summary slots (pooler) | 4 |
| LLM input after pooler | 148 tokens (4 + 144) |
| Token reduction | 3.9× |
| TPN parameters | ~887K |
| UnselectedPooler parameters | ~512K |
| LoRA parameters (rank 32) | ~35M |
| LoRA parameters (rank 64) | ~70M |
| LLM base parameters | 7B (frozen) |
| DINOv2-S parameters | 22M (frozen) |
| CLIP text parameters | ~63M (frozen) |
| CLIP ViT-L/14 parameters | 304M (frozen) |
| DINOv2-S input size | 98×98 px → 7×7 = 49 patches |
| CLIP ViT-L/14 input size | 336×336 px → 24×24 = 576 patches |
| Effective batch size | 4 × 16 × 2 GPU = 128 |
| Phase 1 v3 baseline | val_vqa = **0.3334** |
| Phase 1.5 v8 best | val_vqa = **0.3663** at step 45k |
| Phase 1.5 v8b step 5k | val_vqa = 0.4540 (catapult expected) |
| τ range | 1.0 → 0.1 (linear over max_steps) |
| τ at v8 step 45k | 0.595 (= 1.0 − 0.9×45/100) |
| val_vqa to beat (paper target) | < 0.30 at step 100k |

---

*Generated from source code at commit 3737a22 (2026-04-14). All code paths verified.*
