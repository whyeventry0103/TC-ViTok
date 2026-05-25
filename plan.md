# TC-ViTok — Post-Submission Strategic Roadmap (v2)

**Status:** Internal research blueprint, revised after final-sprint experiments confirmed the core negative result is structural.
**Last updated:** 2026-05-25 (revised post-sprint).
**Author voice:** Adversarial researcher writing for the team that built TC-ViTok.
**Audience:** Project lead + experimental engineers continuing this work over the summer.
**Goal:** Convert TC-ViTok from "an interesting failure-analysis paper" into either (a) a system that demonstrably beats training-free baselines on at least one axis (FLOPs, grounded tasks, or OCR), or (b) a flagship fundamental-limit result with cross-architecture confirmation. Both outcomes are publishable. Both are better than the current state.

**Confidence legend used throughout:**
- `[H]` High — directly supported by our own data or well-established literature
- `[M]` Medium — supported by indirect evidence or strong theoretical argument
- `[L]` Low — speculative; needs validation before being trusted

**Sprint findings legend (added in v2):**
- `[S+]` — sprint experiment confirms the claim
- `[S-]` — sprint experiment refutes the claim
- `[S?]` — sprint experiment leaves the claim ambiguous

---

# 0. Executive Summary

We submitted a paper that honestly documents why pre-LLM trainable visual-token compression is hard at LLaVA-7B scale. Our final sprint experiments produced four hard negative findings that change the strategic picture:

1. **The TPN loses to random selection at every measured $k \in \{8, 18, 24, 36, 72, 144, 216, 288\}$** and on every benchmark, with the gap *widening* at low $k$ (–32pp on POPE at $k=18$). The previously-hoped "extreme-compression regime where structured selection dominates" does not exist for this architecture. `[S-]`
2. **Random selection has standard deviation $<0.4$pp across 5 seeds**, but TPN is 2–4pp below the random mean. The gap is cleanly outside noise. `[S+]`
3. **TPN scores correlate with CLIP CLS-attention at Spearman $\rho \approx -0.03$** (200 GQA samples). The learned selector is essentially independent of the obvious useful prior. `[S+]`
4. **TPN has the steepest accuracy-vs-$\log k$ slope of any selector** on every benchmark. Joint training produced a selection rule that is *more* budget-sensitive than random selection. `[S+]`

These are not adjustments to the prior plan; they are structural facts about what 80K steps of training produced. The previous plan's central bet — *"weak supervision is the problem; fix it with distillation"* — is now too optimistic in isolation. The Spearman ≈ 0 finding is mild evidence that the architecture is converging to something pathological that better supervision *alone* may not escape.

**The v2 plan rests on a different organizing principle:**

> **Before committing 1000+ GPU-h to any improvement track, we run a cheap battery of diagnostic tests that distinguish three competing hypotheses: H_opt (supervision is the problem), H_arch (architecture is the problem), and H_capacity (selector is too small or mis-shaped). Each hypothesis has a different experimental track. We commit to a track only after the diagnostic results select one.**

Three tracks emerge from Phase 1:

- **Track A — Method paper.** If diagnostics show the architecture *could* represent useful selection given the right signal, distillation + cross-attention query + OCR pathway becomes the path. Compute: ~800 GPU-h.
- **Track B — Architectural redesign.** If diagnostics show the trainable-bridge framework is fundamentally limited, pivot to frozen-prior + tiny-residual selectors, Q-Former-style latent bottlenecks, or mid-LLM compression with LoRA. Compute: ~600 GPU-h.
- **Track C — Fundamental-limit paper.** If diagnostics show causal token importance is unpredictable from encoder features, publish that as a flagship result. Compute: ~150 GPU-h plus writeup.

Each track has explicit success criteria, kill criteria, and publication targets. The first 3 weeks of the sprint determine which track is run.

**Beyond the diagnostic restructuring, v2 changes:**

- **Synthetic tasks (needle, two-needle, dense-text) are promoted to gates** — every new architecture must pass them before earning a real-benchmark run.
- **Cross-architecture replication (Phase 2D) is added** as a budgeted phase, because the existing paper's taxonomy claim depends on it.
- **Multi-paper publication strategy** replaces "one big paper at the end" — P1 (TMLR / mechanism paper) at month 2, P2 (method paper or fundamental-limit paper) at month 4.
- **Compute estimates are revised upward 2×** for honesty.
- **Idea 16 (Frozen-Prior + Tiny Residual)** is added — a fail-safe architecture that cannot learn the pathological pattern TPN converged to.

This document is the operational plan. Sections 1–7 are the analysis substrate; Section 8 is the master prioritized to-do list; Section 9 is the publication strategy; Section 10 is the decision tree.

---

# 1. Research Autopsy (Revised)

## 1.1 The big-picture diagnosis

TC-ViTok is a system that *converges* but converges to something worse than random selection. Why?

The previous version of this plan distinguished two competing hypotheses:

- **H_opt:** Architecture is fine; training signal is too weak.
- **H_arch:** Architecture is wrong; training signal weakness is secondary.

Sprint findings force us to add a third:

- **H_capacity:** Architecture has insufficient or mis-shaped capacity (887K params, 4-head cross-attention, single CLIP-text query channel) to represent useful selection regardless of supervision.

The three are not mutually exclusive. With sprint data, my updated priors:

- **H_opt: probably true but insufficient.** Distillation likely improves things, but the magnitude of the Spearman ≈ 0 finding suggests the model converges to a pathological local minimum that better supervision alone may not escape.
- **H_arch: more likely than v1 acknowledged.** The trainable-bridge framework consistently fails across k. The Qwen-VL stress test showed the same instability on a different VLM, suggesting it's not specific to LLaVA.
- **H_capacity: under-investigated.** No prior experiment varied selector size or query-channel width systematically.

**v2 strategic shift:** instead of betting on H_opt, run cheap diagnostics that distinguish all three in 3 weeks before committing to a track. See Section 10.

### 1.1.1 What the sprint data shows about each hypothesis

| Hypothesis | Supporting evidence | Refuting evidence |
|---|---|---|
| **H_opt** (weak supervision) | Warm-start dependence; sparse selector gradient (qualitative); merge shadowed by LoRA; query channel unused at convergence | Spearman ≈ 0 suggests architecture isn't even *capable* of converging to useful prior; weak supervision alone shouldn't produce *systematically worse-than-random* solutions |
| **H_arch** (wrong architecture) | TPN loses to random at every k; slope is steepest of all selectors; trainable bridge instability on Qwen-VL too | Architecture *could* in principle route on query (capacity exists); the merge formula is reasonable; no direct test of alternative bridge designs |
| **H_capacity** (under/mis-capacity) | 887K params is small relative to the 7B LLM and 304M CLIP encoder; 4 heads is small for 576-token routing; CLIP-text 512-d vector is the thinnest query channel of any method we cite | No experiment varied capacity yet; lack of evidence is not evidence of absence |

**Bottom line:** Phase 1 diagnostics must be the first ~50 GPU-h spent.

### 1.1.2 Why the Spearman ≈ 0 finding matters most

If the architecture could in principle represent CLS-attention-like priors at all, we'd expect *some* correlation to emerge over 80K joint-training steps. The fact that none does is mild but real evidence that the joint-training optimization landscape pushes the selector into a basin uncorrelated with any obvious useful prior.

There are three ways this could be true:

(a) **Architecture is incapable of representing CLS-attention-like priors** even if we tried directly (testable: train TPN with CLS-attention as direct target; if it can't match, H_capacity).
(b) **Architecture is capable but optimization pushes away from useful priors** (testable: initialize TPN at CLS-attention and fine-tune; if it stays near CLS-attn and improves end-task, H_opt; if it drifts back to Spearman ≈ 0, H_arch).
(c) **The TPN learned a different, useful prior we haven't measured.** (Testable: probe correlation with multiple priors — FastV-attn, image gradient, position, etc. — see Phase 1.5.)

These three tests distinguish hypotheses cheaply (~12 GPU-h total).

## 1.2 Failure taxonomy (orthogonal axes)

### 1.2.1 Supervision failures `[H]`

| Failure | Mechanism | Sprint annotation | How to verify | How to falsify |
|---|---|---|---|---|
| **End-task loss is too distal.** Selector gradient backprops through 32 LLM layers → projector → merge → STE → into TPN. | `[S?]` | Measure gradient norm at TPN vs LoRA layers. Expected: TPN gradient norm orders of magnitude smaller. | Train (a) end-task only, (b) end-task + KL-distillation from LoRA-576 over LLM hidden states. If (b) converges faster or higher, H_opt confirmed. | Distillation produces identical or worse results — points to H_arch or H_capacity. |
| **No teacher for "what to keep."** Selector has no direct supervision on which tokens are useful. | `[S+]` Spearman ≈ 0 with CLS-attention is consistent. | Compare TPN-selected tokens against FastV-style oracle. Jaccard / NDCG. | Use oracle masks as direct selector supervision. Expect improvement. | Oracle-supervised TPN does not beat end-task-supervised TPN. |
| **No spatial-prior supervision.** | `[S+]` TPN slope steepest → coverage degrades fastest under budget. | Visualize selection spatial distribution; expect higher entropy than oracle. | Add spatial dispersion auxiliary loss. | No improvement on POPE, RefCOCO. |
| **No OCR supervision.** | `[S+]` CLS-attention beats TPN by 10.8 pp on TextVQA. | Mean TPN score on OCR-detected patches vs background. | Add OCR-region prior. | TextVQA does not improve materially. |

### 1.2.2 Optimization failures `[H]`

| Failure | Mechanism | Sprint annotation | How to verify |
|---|---|---|---|
| **Co-adaptation oscillation** | Documented in paper. Period scales with η_selector. | `[S?]` Qwen-VL stress test showed loss-instability but period-LR ordering contradicted. The mechanism may be more complex than two-player GAN theory predicts. |
| **Budget collapse at λ_b ≤ 0.05.** | Documented. | `[S?]` Not stress-tested at multiple architectures. |
| **STE temperature collapse.** | When τ → 0 too fast, gradients near zero. | Not directly stress-tested in sprint. |
| **Sparse gradient flow through top-k.** | Most gradient flows through selected tokens. | `[S?]` Consistent with Spearman ≈ 0: if gradient barely reaches unselected positions, the score function never learns to discriminate. |
| **LLM adapts around bad selection.** | LoRA learns to produce correct answers despite poor selection. | `[S+]` Random-selection LoRA-trained accuracy is *higher* than TPN-selection accuracy. Strongly consistent. |

### 1.2.3 Representation failures `[M]`

| Failure | Mechanism | Sprint annotation |
|---|---|---|
| **CLIP text embedding is too coarse for question conditioning.** | Single 512-d vector compresses all questions. | `[S?]` Not directly tested; query channel width never varied. |
| **DINOv2 spatial guide is uncoupled.** | Enters as separate stream; not used for spatial reasoning. | Not tested. |
| **Pre-LLM features are not LLM-aligned.** | CLIP features were trained for image-text contrastive, not LLM consumption. | `[M]` Causal-token-importance test (Idea 10) would settle this. |
| **Merge as cosine-similarity is wrong operator.** | Tokens that look similar by key cosine may not carry interchangeable information. | `[S+]` Merge contributes zero accuracy at every β. Cosine similarity is decorative. |

### 1.2.4 Architectural failures `[M] → [M-H]` after sprint

| Failure | Mechanism | Sprint annotation |
|---|---|---|
| **Selector capacity too low.** 887K params; 4 heads. | `[M]` Not tested yet but the smallness is striking relative to 7B LLM. |
| **One-step compression too aggressive.** | `[M]` Two-stage compression untested. |
| **No recycling.** Once dropped, tokens are gone. | `[M]` |
| **No register tokens.** | `[M]` Dead-zone effect doesn't replicate across seeds — registers may not help. |
| **Trainable bridge itself may be the wrong abstraction.** | `[S+, NEW]` Spearman ≈ 0 + steepest-slope + below-random performance across all k together suggest the bridge framework converges to bad solutions. Need to test alternatives. |

### 1.2.5 Data / SFT failures `[H]`

| Failure | Mechanism |
|---|---|
| **Restricted SFT mix.** Missing OCR-VQA, A-OKVQA, RefCOCO, ChartQA, DocVQA. |
| **No grounded-task data.** RefCOCO etc. is where query-conditioning should pay off. |
| **VQAv2 / GQA train sets reward saliency, not query-conditioning.** |

### 1.2.6 Evaluation failures `[M]`

| Failure | Mechanism |
|---|---|
| **Benchmark suite is dominated by global-image tasks.** |
| **5K-subset evaluations have ~0.6 pp noise.** |
| **Single-seed evaluation.** Variance unknown. |
| **No real out-of-distribution eval.** |
| **No timing / FLOPs reported at matched accuracy.** |

### 1.2.7 Scaling failures `[L → M]` after sprint

| Failure | Mechanism |
|---|---|
| **All experiments at LLaVA-1.5-7B.** Conclusions may be 7B-specific. |
| **Single token budget k=144.** Sprint extended to {8,18,24,36,72,144,216,288} — pattern is now more robust. |
| **Single encoder (CLIP ViT-L/14).** Different encoders untested. |
| **Single VLM family.** Qwen-VL stress test mildly confirms trainable-bridge instability but doesn't generalize the full taxonomy. |

## 1.3 Symptoms vs. causes

| Observed phenomenon | Likely root cause |
|---|---|
| Co-adaptation oscillation | Joint optimization with weak supervision (signal too far from selector). Oscillation is the symptom. |
| Budget collapse | Lagrangian mis-tune: penalty too weak vs LM gradient. |
| Warm-start dependence | Cold-start optimization gets trapped because the selector receives no useful early signal. |
| Dead-zone region | Attention-sink-like behavior, but also: the selector has no training signal pushing it to use those positions. Doesn't replicate across seeds. |
| E3 query-agnostic convergence | CLIP text embedding has too low signal-to-noise relative to the visual stream. Not a deep finding about query-conditioning. |
| Merge β contributes nothing | LoRA absorbs the merge contribution. The merge does information-flow that LoRA can substitute for. |
| **`[NEW]` TPN scores Spearman ≈ 0 with CLS-attention** | **Joint training under end-task-only supervision pushes the selector into a basin uncorrelated with any obvious useful prior. Either the architecture cannot escape this basin, or the optimization gradient never points toward useful priors.** |
| **`[NEW]` Random selection beats TPN at every k** | **Either (a) LoRA does all the work and any input distribution serves equally; (b) the TPN actively destroys information that LoRA would otherwise have; or (c) random's input diversity acts as regularization. These three are testable separately.** |

This re-reading is uncomfortable but necessary. **Most of our headline findings are symptoms of weak supervision or architectural pathology — the new sprint data sharpens which it is.**

## 1.4 What we never measured (and now should)

Updated list incorporating sprint gaps:

- **What the TPN actually learned.** Sprint started this with Spearman analysis, but we need: histograms of scores over training; correlation with multiple priors (CLS-attn, FastV-attn, gradient magnitude, position, pixel intensity); per-image vs global score patterns. **Phase 1.5 priority.**
- **Causal token importance per example.** For each example, ablate each token in turn; measure ΔLLM-prediction. Use as oracle. ~8 GPU-h for 200 samples.
- **LoRA-on-random vs LoRA-on-TPN.** Train LoRA *only* with random selection. If accuracy matches LoRA-on-TPN, the selector is decorative.
- **Selection consistency across seeds.** Two TPNs from different seeds — Jaccard? `[S?]` Mentioned in original plan; still not run.
- **Selection consistency across questions.** For one image and 5 different questions, do TPN selections actually vary?
- **Information content of selected tokens.** Mutual information between selected tokens and ground-truth answer.
- **Failure-case clustering.** Per question type, image type, answer length, object density.
- **Counterfactual selection.** For failed examples, would replacing TPN with FastV/PruMerge/oracle selection have worked?
- **OOD generalization.** Domain-shifted dataset (medical, document, scene-text).
- **Inference-time variance across decoding strategies.**
- **FLOPs and latency measurements at matched accuracy.** Essential for any compression claim.

---

# 2. Design Space Map

This section taxonomizes prior work mechanistically, then maps TC-ViTok onto the same axes.

## 2.1 Mechanism taxonomy of cited methods

### 2.1.1 Where in the pipeline compression happens

```
[Encoder] → [Pre-LLM module] → [LLM]
                                  │
                                  └ [Mid-LLM module] → [Output]
```

| Method | Where | Why this matters |
|---|---|---|
| ToMe | Inside ViT | Operates on visual features only; no query/LLM signal. Training-free. |
| EViT | Inside ViT | Uses CLS-attention as saliency; preserves info via fusion. |
| DynamicViT | Inside ViT | Small predictor; **distilled from teacher**. |
| PruMerge | Pre-LLM | Uses CLS rank; merges. Training-free. |
| FastV | Inside LLM (after layer ~2) | **By layer 2, visual info has mixed into text tokens via attention.** Pruning visual tokens post-mixing keeps the information. |
| SparseVLM | Inside LLM, multi-layer | Text-guided pruning + recycling. |
| BLIP-2 Q-Former | Pre-LLM | Fixed query bank; cross-attention 576 → 32. |
| QG-VTC | Inside encoder | Question injected into encoder. |
| FocusLLaVA | Inside LLM, coarse-to-fine | Two-stage attention. |
| **TC-ViTok** | **Pre-LLM** | **Hardest position: must anticipate LLM's needs from frozen encoder features alone.** Sprint confirms this position is structurally hard. |

### 2.1.2 What supervisory signal each method uses

| Method | Supervision | Strength |
|---|---|---|
| ToMe / EViT / PruMerge / FastV | None (training-free) | N/A |
| DynamicViT | **Distillation from teacher + sparsity loss** | Strong |
| SparseVLM | End-task loss | Medium |
| BLIP-2 Q-Former | Image-text contrastive + caption + ITM | Strong (multi-objective) |
| QG-VTC | End-task loss with question conditioning | Medium |
| **TC-ViTok** | **End-task loss only** | **Weakest** |

Observation: among methods that *train* a selector, every successful one uses **distillation** (DynamicViT), **multi-objective pretraining** (Q-Former), or **strong inductive bias** (CLS attention as starting point in EViT). We use none of these.

### 2.1.3 Information preservation strategy

| Method | What happens to unselected tokens? |
|---|---|
| ToMe | Merged via key similarity into selected |
| EViT | Fused into a single "background" token |
| DynamicViT | Dropped |
| PruMerge | Merged into nearest retained neighbor |
| FastV | Dropped (but already mixed into text via early-layer attention) |
| SparseVLM | Dropped at one layer, can be recycled at later layers |
| BLIP-2 Q-Former | Replaced by query bank cross-attention output |
| **TC-ViTok** | **Merged into selected via cosine-similarity-weighted aggregation, β=0.3 — shadowed by LoRA, zero accuracy contribution** |

PruMerge's merge works because the LLM is *not adapted* to the compressed representation. Our LoRA adapts to the post-merge representation, making the merge's information flow shadowed.

### 2.1.4 Query conditioning

| Method | Query channel | Granularity |
|---|---|---|
| ToMe / EViT / DynamicViT / PruMerge / FastV | None | Query-agnostic |
| SparseVLM | LLM internal text-to-visual attention | Per-token text → per-token visual |
| BLIP-2 Q-Former | Fixed learned queries (32) | Coarse |
| QG-VTC | Question → encoder injection | Fine |
| FocusLLaVA | Instruction-guided cross-attention | Fine |
| **TC-ViTok** | **Single CLIP text-embedding vector (512-d)** | **Coarsest possible** |

Our query channel is the *thinnest* of any query-aware method. Direct comparison with SparseVLM's per-token attention is unfavourable.

### 2.1.5 Optimization regime

| Method | Optimization |
|---|---|
| ToMe / EViT / DynamicViT / PruMerge / FastV | Training-free; some with light distillation |
| DynamicViT | Predictor trained with distillation |
| BLIP-2 | Three pretraining objectives over massive data |
| QG-VTC | Stage-wise pretraining then fine-tuning |
| **TC-ViTok** | **Joint end-to-end training with three components (TPN, LoRA, BudgetPredictor)** |

The methods that work avoid joint training of the selector or use much stronger supervision. We picked the worst of both worlds.

## 2.2 The "What TC-ViTok lacks" gap matrix

| Axis | Best practice | TC-ViTok | Implication |
|---|---|---|---|
| Compression stage | Multi-stage (Q-Former: 576→32; FocusLLaVA: coarse→fine) | Single-stage 576→144 | Overloaded one decision |
| Query channel | Per-token text attention (SparseVLM) or massive pretraining (Q-Former) | Single CLIP text vector | Channel too thin |
| Selector supervision | Distillation (DynamicViT), strong contrastive (Q-Former) | End-task only | Signal too distal |
| Selector initialization | Distillation init or CLS attention | From scratch | Cold start, bad local minima |
| Token info preservation | Multi-mode (fuse, merge, recycle) | One mode, β shadowed | No effective recovery |
| Register / sink tokens | Darcet 2024 for sinks | None | Dead-zone artifact unaddressed |
| OCR pathway | Specialized in published LLaVA-1.5 via SFT mix | None | TextVQA −12.6pp gap |
| Curriculum | Multi-stage pretraining | Three-phase but same data | No graduated difficulty |
| Aux objectives | Sparsity, distillation, contrastive | Sparsity (λ_b only) | Few constraints |
| RL refinement | Not in compared methods | None | Could exploit non-differentiable signals |

## 2.3 Full design-axes inventory

### 2.3.1 Selector axes

| Axis | TC-ViTok | Alternatives |
|---|---|---|
| Architecture | Cross-attention TPN, 4 heads, d=384 | MoE selector, MLP-only, GNN-over-patches, **frozen-prior + tiny residual (NEW)** |
| Capacity | 887K params | 1.5M, 3M, 6M params |
| Hard vs soft | Hard top-k (with STE) | Soft attention pooling, gumbel-soft, **content-driven latent bottleneck (Q-Former style)** |
| Query channel | CLIP text 512-d vector | Tokenized question with cross-attention, LLM's first-layer text embedding, retrieval-augmented context |
| Spatial prior | DINOv2 features as side channel | 2D positional priors, frozen ViT attention as prior, learned 2D RoPE |
| Saliency prior | None (learned from scratch) | **CLIP CLS-attention init, FastV-derived init, frozen-prior + small residual** |

### 2.3.2 Compression strategy axes

| Axis | TC-ViTok | Alternatives |
|---|---|---|
| Number of stages | 1 | 2 (576→288→144), 3 stages, continuous |
| Budget allocation | Fixed k=144 or adaptive | Per-region, per-layer, RL-allocated |
| Merge strategy | Cosine-sim weighted, β=0.3 | Cross-attention merge, learned routing, **no-merge + recycling**, Q-Former-style |
| Recycling | None | Soft (mid-LLM), hard (LLM-uncertainty), retrieval |
| Register tokens | None | 4, 8, 16 (Darcet 2024) |

### 2.3.3 Supervision axes

| Axis | TC-ViTok | Alternatives |
|---|---|---|
| End-task loss | CE on next-token | **KL distillation from teacher VLM**, attention distillation, hidden-state distillation |
| Selector aux loss | None (only λ_b) | Mask-prediction (DynamicViT-style), CLS-attention prior, FastV pseudo-labels, **causal-token-importance distillation (NEW)** |
| Spatial aux loss | None | Spatial coverage entropy, diversity loss |
| OCR aux loss | None | Text-detector pseudo-labels |

### 2.3.4 Training-regime axes

| Axis | TC-ViTok | Alternatives |
|---|---|---|
| Joint vs staged | Three-stage curriculum | Pure staged (freeze LoRA during TPN training), pure joint, alternating EM-style |
| Warm-start | Yes (from converged base) | Distillation warm-start, **CLS-attention init**, RL warm-start |
| Data | VQAv2 + GQA + LLaVA-Instruct + TextVQA | Add RefCOCO, OCR-VQA, A-OKVQA, ChartQA, DocVQA |
| Curriculum on data | None | Easy → hard, single-region → multi-region |

### 2.3.5 Inference axes

| Axis | TC-ViTok | Alternatives |
|---|---|---|
| Decoding | Greedy | Beam, nucleus, top-k sampling, speculative |
| Test-time adaptation | None | Token re-selection on uncertainty, retrieval top-up |
| Caching | None | KV-cache aware compression |

## 2.4 Ranked idea space (REVISED with sprint findings)

| Idea | ROI v1 | ROI v2 | Compute | Novelty | Risk |
|---|---|---|---|---|---|
| Distillation from LoRA-576 teacher | Very high | **High** (downgraded; teacher quality unknown) | Medium | Medium | Low |
| **Frozen-prior + tiny residual selector (NEW, Idea 16)** | — | **Very High** | Low | Medium-high | Low |
| **Causal-importance probe / distillation (NEW priority)** | Low/High | **Very High** | Low | High | Medium |
| Per-token-question cross-attention | High | High | Low | Medium | Low |
| CLS-attention initialization | High | **Very High** | Very low | Low | Low |
| Two-stage compression (576→288→144) | High | High | Medium | Low | Medium |
| OCR-aware dual pathway | High (TextVQA) | High (TextVQA) | Medium | Medium | Medium |
| Add full SFT mix (OCR-VQA, RefCOCO, etc.) | High | High | High | Zero | Low |
| Register tokens (Darcet 2024) | Medium | Low | Low | Low | Low |
| **Q-Former-style latent bottleneck (Idea 9 elevated)** | Medium | **High** | Medium | Medium | Medium |
| RL fine-tuning of selector | Medium | Medium | High | High | High |
| Cross-attention merge (replace cosine sim) | Medium | Low | Low | Low | Low |
| Mid-LLM compression + LoRA (NEW direction) | — | Medium | Medium | Medium | Medium |
| MoE selector | Low | Low | High | Medium | High |
| Recycling on LLM uncertainty | Medium | Medium | High | High | High |

Pattern: the highest-ROI ideas after the sprint are **(a) diagnostic / probe experiments that distinguish hypotheses (causal importance, frozen-prior fail-safe)** and **(b) initialization-based interventions (CLS-attention init)** that test whether the architecture is rescuable. Heavy retraining is deferred until diagnostics finish.

---

# 3. Systematic Debugging Framework

The biggest methodological error of the project so far was launching expensive runs before validating subsystems. Here is a complete "research operating system" to never repeat that.

## 3.1 Philosophical principles

1. **No expensive run without a cheap version that already works.** If you can't build a 1K-step toy version, don't run the 100K-step version.
2. **Every run must have a falsifier.** Before launching, write: "if X happens, this hypothesis is wrong."
3. **Every component must be testable in isolation.**
4. **Confounds are the enemy.** Single-variable changes only.
5. **`[NEW v2]` Synthetic-task gating.** Every new architecture must pass synthetic-task gates (Section 3.3) before earning a real-benchmark run.

## 3.2 Subsystem validation tests

### 3.2.1 TPN-in-isolation tests

**T1: Selector consistency under seed.** Two TPNs trained with seeds 7 and 42. For 500 validation images, compute selection Jaccard. Expected ≥ 0.7 if genuinely converging; < 0.3 indicates near-random.

**T2: Selector consistency under query.** Single image, 10 different questions. Measure selection diversity per image (KL divergence across questions). High = uses query. Low = query-agnostic. **Proper test of E3 we never ran.**

**T3: Oracle-vs-actual selection.** Generate oracle selection per image as top-144 patches by FastV-style teacher attention. Jaccard, NDCG. If Jaccard < 0.3, TPN is essentially uncorrelated with what LLM uses.

**T4: Token coverage and entropy.** Per-position selection frequency over 5K images. Mass-balance metric (max − min). Dead-zone metric. Hot-spot metric.

**T5: Selector gradient health.** Throughout training, log ‖∂L_LM / ∂score_TPN‖, ‖∂L_budget / ∂score_TPN‖, ratio over time.

### 3.2.2 LoRA-in-isolation tests

**T6: LoRA-vs-selector responsibility split.** Freeze TPN; train LoRA from scratch. Replace TPN with random selection; train LoRA from scratch. If both LoRAs reach similar accuracy, LoRA has absorbed selection responsibility.

**T6.5 (NEW v2): LoRA-on-random vs LoRA-on-TPN.** Train LoRA with *always-random* selection vs *always-TPN-trained* selection (TPN frozen). If LoRA-on-random matches or exceeds LoRA-on-TPN, the selector is decorative or actively harmful. **This is the cheapest test of "why does random win?"** Compute: ~6 GPU-h.

**T7: LoRA capacity probe.** Train with rank ∈ {16, 32, 64, 128}. Find saturation point.

### 3.2.3 Merge-in-isolation tests

**T8: Merge information flow.** CKA between selected token *post-merge* and the full set of unselected tokens it merged from.

**T9: Merge accuracy contribution at different LoRA scales.** Train LoRA-small (rank 16) and LoRA-large (rank 64) with merge on/off. If merge contribution is zero at every LoRA size, merge transfers no useful information.

### 3.2.4 End-to-end behavioral tests

**T10: Counterfactual selection.** For each failed example, replace TPN selection with PruMerge / FastV / random / oracle. Conditional accuracy.

**T11: Failure clustering.** Cluster failed examples by image complexity, question type, answer length.

**T12: OOD probe.** Held-out domain (medical, document).

### 3.2.5 NEW v2 tests

**T_new1: TPN learned-distribution characterization.** For 500 GQA samples, compute:
- TPN score histogram distribution (mean, std, entropy).
- Spearman correlation between TPN scores and: CLIP-CLS attention, FastV-attn at multiple layers, image gradient magnitude, pixel intensity, position bias.
- t-SNE / UMAP of per-image score patterns.
- Per-image score variance across questions (T2 generalized).

Compute: ~4 GPU-h.

**T_new2: Causal token importance.** For each of 200 test images, compute per-token causal importance by ablating each token in a full-context forward pass and measuring ΔLM-prediction. Train a probe to predict this from encoder features.

Compute: ~8 GPU-h.

**T_new3: LoRA-on-X comparison.** See T6.5.

## 3.3 Synthetic environments (PROMOTED in v2 to first-class gates)

**Every new architecture must pass the synthetic gates before earning a 5K-sample benchmark run.** This saves enormous compute by killing architectures that lack basic capacity.

### 3.3.1 The "needle-in-the-image" task

Place a colored shape at a known patch location in a noise image. Ask "what color is the shape at position (i, j)?". The right token to keep is the one at (i, j).

- TPN must select the needle patch ≥ 95% when query mentions location → architecturally capable.
- TPN selects needle regardless of query → saliency-based, not query-conditioned.
- TPN never selects needle → architectural incapacity.

**Pass criterion:** ≥ 90% needle-selection accuracy when query specifies location.

### 3.3.2 The "two-needle" task

Two shapes, two locations. Ask "what color is the shape at (i₁, j₁)?". The right token is one specific patch; the other shape is a distractor.

**Pass criterion:** ≥ 80% correct-needle selection.

### 3.3.3 The "dense text" task

Synthetic images with rendered text at known positions. Ask OCR-style questions.

**Pass criterion:** ≥ 70% text-patch retention rate when query is OCR-relevant.

### 3.3.4 Synthetic-gate decision logic

| Synthetic result | Implication |
|---|---|
| Passes all three → run on real benchmarks. |
| Passes needle but fails two-needle → query-conditioning is partial. |
| Fails needle → architecture cannot route on query. Stop. Try different architecture or different query channel. |
| Passes needle but fails dense-text → need OCR pathway. |

## 3.4 Diagnostic dashboard

A persistent dashboard, updated per checkpoint:

1. Convergence curves: loss, accuracy, all components.
2. Selector statistics: position selection frequency map, dead-zone area, hot-spot count, entropy.
3. Gradient health: TPN gradient norm / LoRA gradient norm ratio.
4. Selection consistency: Jaccard with previous checkpoint, mean shift over training.
5. Query usage: selection-distribution variance across different questions on same image.
6. Failure clustering by question type.
7. Sample image with current selection overlay (6 examples, updated per epoch).
8. β-sweep at last checkpoint.
9. Auxiliary-loss values.
10. OCR-region selection fraction.
11. **`[NEW v2]` Correlation of TPN scores with CLS-attention, FastV-attn over training.**
12. **`[NEW v2]` Synthetic-task accuracy (needle, two-needle, dense-text) as automatic regression gates.**

## 3.5 Causal-intervention infrastructure

Beyond observation, build the ability to intervene:

- **Force selection.** Override TPN with a custom mask. Measure delta accuracy.
- **Force score.** Set per-position score directly to a target.
- **Force query.** Replace CLIP text embedding with arbitrary vector.
- **Force LoRA frozen.** Train selector while LoRA is locked.
- **Force merge.** Test different merge formulas without retraining.
- **`[NEW v2]` Force initialization.** Initialize TPN to produce specific scores (CLS-attention, FastV-attn, random with seed).

All of these should be CLI flags on `evaluate.py` and `train.py`.

## 3.6 Reproducibility infrastructure

- **Seed registry.** Every run records full seed state.
- **Hyperparameter pinning.** Every config commits a hash.
- **Data ordering.** DataLoader seed pinned.
- **Result tables auto-generated** from JSON logs.
- **Multi-seed launch script.** Single command launches 3 seeds.

Going forward, **single-seed runs are diagnostic only. Headline numbers require N=3.**

## 3.7 Failure pattern detection

- Every dataset class has a `validate()` method.
- Every dataloader emits a sample-rate metric.
- Every training run logs a "first 10 samples" hash so two runs with the same data go through the same examples.

## 3.8 NEW v2: Measurement infrastructure

Beating FastV "on FLOPs at matched accuracy" requires careful measurement infrastructure that does not exist in the current codebase.

**Required tools:**

- **FLOP counter.** Per-module FLOP counts for forward pass at multiple k values. Use `fvcore` or `calflops`.
- **Latency benchmark.** Wall-clock per-image inference time on identical hardware (3 GPU types if possible: RTX 6000, A100, H100). Average over ≥ 100 examples.
- **Pareto plot generator.** Given (FLOPs, accuracy) tuples, generate Pareto front plots with confidence intervals.
- **Apples-to-apples comparison script.** Match accuracy ± 0.5 pp, report FLOPs/latency comparison.

**Phase 0.5 deliverable:** All four tools operational. Without these, any "we beat FastV" claim is unsupportable.

---

# 4. Experimental Strategy (Diagnostic-First Roadmap)

## v2 organizing principle

The previous plan committed to a single linear sprint. v2 commits to a **diagnostic-first branching strategy**:

1. **Weeks 1–2:** Infrastructure + measurement.
2. **Weeks 3–4:** Phase 1 diagnostics + Phase 1.5 "what did TPN learn?"
3. **Week 5:** Decision gate → branch into Track A, B, or C.
4. **Weeks 6–14:** Track-specific execution.
5. **Weeks 15–16:** Headline run + writeup.

Each track has different compute requirements (Track C is cheapest, Track A heaviest). The diagnostic phase is the same regardless.

## Phase 0 — Infrastructure (Week 1)

**Goal:** Make every subsequent experiment cheap and reliable.

| Task | Deliverable | Compute |
|---|---|---|
| Build diagnostic dashboard (Sec 3.4) | TensorBoard / WandB layout, one click per run | 0 |
| Build causal-intervention CLI (Sec 3.5) | Forced selection, forced query, forced merge, **forced init** | 0 |
| Implement multi-seed launcher | One command → 3 seeds | 0 |
| Implement subsystem-isolation runs | TPN-only, LoRA-only, merge-only modes | 0 |
| Implement synthetic-task suite (Sec 3.3) | Needle, two-needle, dense-text generators | 0 |
| Implement seed registry + hyperparameter hashing | Logger module | 0 |
| Validate everything on 1K-step toy runs | All tests pass | ~4 GPU-h |

**Success criterion:** All Phase-1 diagnostics can be run via a single command on a given checkpoint, output a single dashboard URL.

**Kill criterion:** If any of the above cannot be done in a week, the project's compute is misallocated. Re-plan.

## Phase 0.5 NEW v2 — Measurement Infrastructure (Week 1)

Parallel with Phase 0.

| Task | Deliverable | Compute |
|---|---|---|
| FLOP counter | Per-module forward-pass FLOP counts at k ∈ {36, 72, 144, 288, 576} | 0 |
| Latency benchmark | Wall-clock measurements on RTX 6000 | ~2 GPU-h |
| Pareto plot generator | Code to plot (FLOPs, accuracy) with N=3 confidence intervals | 0 |
| Apples-to-apples FastV comparison | Script comparing TC-ViTok vs FastV at matched-accuracy FLOPs | 0 |

**Success criterion:** Run produces a single (FLOPs, latency, accuracy) tuple for any checkpoint + k combination.

## Phase 1 — Diagnostics on Existing Checkpoints (Week 2)

**Goal:** Run the full diagnostic battery on `phase1_final`, `phase1_adaptive_v2`, `phase1_v3`, `phase1_seed7`.

| Test | Deliverable | Findings drive |
|---|---|---|
| T1 Selector consistency under seed (Final vs Seed 7) | Jaccard | Whether TPN is a stable solution |
| T2 Selector consistency under query | Per-image diversity | Whether E3 is "doesn't use" or "can't use" query |
| T3 Oracle-vs-actual selection | Jaccard with FastV-attn oracle | Whether TPN is close to what LLM wants |
| T4 Token coverage map | Heatmap + statistics | Dead-zone consistency across seeds |
| T5 Selector gradient health | Plots over training | Whether selector was being trained |
| T6.5 NEW v2: LoRA-on-random vs LoRA-on-TPN | Conditional accuracy | Whether selector matters at all |
| T8 Merge information flow | CKA | Whether merge transfers info |
| T10 Counterfactual selection | Conditional accuracy under oracle | Where the bottleneck actually is |
| T11 Failure clustering | Per-cluster accuracy | Where to invest |

**Expected compute:** ~15 GPU-h.

## Phase 1.5 NEW v2 — "What Did TPN Learn?" (Week 2-3)

**Goal:** Characterize the TPN's learned distribution comprehensively. The sprint started this with Spearman ≈ 0; we need to know more.

| Test | Deliverable | Compute |
|---|---|---|
| T_new1: TPN score histogram + multi-prior correlation | Spearman with CLS-attn, FastV-attn (layers 1, 4, 8, 16, 28), image gradient, pixel intensity, position bias. Histograms per training-step (5K, 20K, 40K, 80K). | ~4 GPU-h |
| T_new1.5: t-SNE / UMAP of per-image score patterns | Are TPN scores image-specific or generic? | ~1 GPU-h |
| T_new1.6: Per-question score variance | T2 generalized: 100 images × 5 questions, measure per-image KL divergence across questions. | ~2 GPU-h |
| T_new2: Causal token importance (200 samples) | Per-token ablation; train probe to predict from encoder features; report R². | ~8 GPU-h |

**Total compute:** ~15 GPU-h.

**Critical decision points from Phase 1.5:**

- **If T_new1 shows TPN learned a coherent prior (Spearman > 0.4 with anything):** characterize what it is. This is itself a publication finding ("what joint training of pre-LLM selectors actually converges to").
- **If T_new1 shows TPN scores are essentially random across all priors:** the architecture is converging to a pathological local minimum. H_arch confirmed; design alternative bridge architecture.
- **If T_new2 shows causal token importance is predictable (R² > 0.5):** distillation has a meaningful ceiling. Track A becomes the path.
- **If T_new2 shows causal token importance is NOT predictable (R² < 0.2):** pre-LLM compression is fundamentally limited. Track C becomes the path.

## Phase 1.75 NEW v2 — Decision Gate (Week 4)

**Goal:** Choose Track A, B, or C based on Phase 1 + 1.5 findings.

See Section 10 for the explicit decision tree. Decision must be made before launching Phase 2.

## Phase 1.9 NEW v2 — ICONIP Review Response (Week 4-5)

If ICONIP reviews have landed by this point, set aside ~1 week to:

1. Read all reviews carefully.
2. Identify any required experiments for response.
3. Update Phase 2 plan with any reviewer-suggested directions.
4. Draft response.

If reviews haven't landed yet, defer this phase but reserve the time.

## Phase 2 — Subsystem Experiments (Weeks 5-10)

Branches based on Phase 1.75 track decision. Each track is run as a focused 5-6 week sprint.

### Track A — Method paper (H_opt + partial H_arch fixable)

**Triggered if:** T_new2 shows causal-importance is predictable AND T_new1 shows TPN is *capable* of representing useful priors (i.e., when forced to track CLS-attention, it can).

#### A1 (Phase 2A) — Supervision experiments

**E_sup1: Distillation from LoRA-576 teacher**
- Joint loss: α · CE + β · KL(student || teacher) + γ · attention-distillation + δ · BCE(selector_mask, teacher_attn_top-k).
- **Pre-condition: profile teacher's attention pattern first (Phase 0.5 add-on, ~2 GPU-h).** If teacher attention is near-uniform across visual tokens, abort.
- Compute: ~30 GPU-h (full training to convergence, not 6).
- Success: TPN exceeds Final by ≥ 2 pp OR reaches Final at 1/3 training compute.

**E_sup2: FastV-derived oracle masks as direct supervision**
- Pre-compute FastV-style oracle selection per training image.
- BCE loss on mask prediction.
- Compute: ~20 GPU-h.
- Success: TPN selects > 0.7 Jaccard with oracle AND downstream accuracy improves ≥ 1 pp.
- Failure: TPN matches oracle but accuracy doesn't improve → oracle is wrong target.

**E_sup3: CLS-attention initialization**
- Initialize TPN to produce scores proportional to CLIP CLS-attention. Then fine-tune with end-task.
- Compute: ~15 GPU-h.
- Success: Faster convergence; final accuracy ≥ TC-ViTok Final; correlation with CLS-attn doesn't decay to ~0.

**E_sup4: Add full SFT mix**
- Add OCR-VQA, A-OKVQA, ChartQA, RefCOCO, DocVQA.
- Compute: ~60 GPU-h.
- Success: TextVQA improves by ≥ 5 pp; RefCOCO above random baseline.

#### A2 (Phase 2B) — Architecture experiments (selectively, based on Phase 1 findings)

**E_arch1: Per-token-question cross-attention**
- Replace single CLIP text embedding with tokenized question encoded by frozen LLaVA text encoder. Per-visual-token cross-attention to per-question-token.
- **Synthetic-task gate first: must pass needle test before benchmarking.**
- Compute: ~25 GPU-h.
- Success: E3 query-zeroing gap grows; T2 per-image diversity increases.

**E_arch2: Two-stage compression**
- 576 → 288 (stage 1) → 144 (stage 2). Stage 1 trained with attention-distillation. Stage 2 with end-task + query.
- Compute: ~30 GPU-h.

**E_arch4: OCR-aware dual pathway**
- Off-the-shelf OCR detector → density score. Above threshold: k=288 high-token pathway. Else: k=144 standard pathway.
- Compute: ~15 GPU-h.

#### A3 (Phase 2C) — Optimization experiments

**E_opt1: Frozen-LoRA selector training**
- Train TPN with LoRA frozen. Then unfreeze.
- Compute: ~15 GPU-h.

**E_opt3: RL fine-tuning post supervised pretraining**
- Pretrain TPN with E_sup1. RL fine-tune with reward = task accuracy.
- Compute: ~25 GPU-h.

#### Track A budget: ~250 GPU-h for Phase 2 experiments

### Track B — Architectural redesign (H_arch + H_capacity dominant)

**Triggered if:** T_new1 shows TPN scores are uncorrelated with any useful prior (random walk in score space) OR T6.5 shows LoRA-on-random matches LoRA-on-TPN OR E_sup3 fine-tuning from CLS-attention drifts back to Spearman ≈ 0.

#### B1: Frozen-prior + tiny residual (Idea 16)

- Initialize selector as a *fixed* function of CLIP CLS-attention.
- Add small (~50K params) MLP producing a score *adjustment*: `scores = CLS_attn + α · MLP(features)` with α small.
- Selector starts matching the strongest training-free baseline by construction; can only learn to refine, not destroy.
- Compute: ~20 GPU-h.
- Success: matches PruMerge at k=144 AND beats it by ≥ 1 pp.

#### B2: Q-Former-style latent bottleneck (Idea 9 elevated)

- Replace top-k with content-driven latent bottleneck. 576 visual tokens cross-attend to L learnable latent tokens (L=64 or 128). Latent tokens are then used as visual prefix.
- Cleanly different from top-k; no STE; no merge.
- Compute: ~30 GPU-h.
- Success: matches or exceeds TC-ViTok Final on at least one benchmark.

#### B3: Mid-LLM compression + LoRA (NEW direction)

- Compress at LLM layer 2 (FastV-style), but with LoRA training on the compressed model.
- Combines training-free's recovery advantage with training-time's adaptation advantage.
- Compute: ~30 GPU-h.

#### B4: Encoder-LoRA adaptation

- Unfreeze CLIP with rank-16 LoRA. Train alongside TPN.
- Compute: ~25 GPU-h.

#### Track B budget: ~150 GPU-h for Phase 2 experiments

### Track C — Fundamental-limit paper

**Triggered if:** T_new2 shows causal-token-importance is unpredictable from encoder features (R² < 0.2). This is the most theoretically interesting outcome.

#### C1: Comprehensive causal-importance analysis

- Extend T_new2 to 5000 samples (not 200).
- Compute causal importance per token per question.
- Train probes of varying capacity (linear, 2-layer MLP, transformer) to predict from frozen features.
- Report R² as function of probe capacity.
- Compute: ~50 GPU-h.

#### C2: Mid-LLM activation predictability

- Same probe analysis but with LLM-intermediate hidden states as input.
- If LLM-intermediate hidden states *do* predict causal importance, this shows the gap is information-theoretic at the *encoder* level: pre-LLM compression is fundamentally bounded because the relevant information has not yet been computed.
- Compute: ~30 GPU-h.

#### C3: Architecture-independent verification

- Repeat C1 with multiple encoders (DINOv2, SigLIP) to test whether the bound is encoder-specific or fundamental.
- Compute: ~40 GPU-h.

#### Track C budget: ~120 GPU-h. Publication target: TMLR, NeurIPS, ICML.

## Phase 2D NEW v2 — Cross-architecture replication (Weeks 8-10, parallel)

**Goal:** Reproduce the winning Phase 2 architecture's findings on a second VLM family.

| Task | Architecture | Compute |
|---|---|---|
| Port best Phase 2 winner to LLaVA-Next | New encoder (SigLIP) + same LLM | ~50 GPU-h |
| Port best Phase 2 winner to Qwen-VL | Different LLM family | ~50 GPU-h |

**Success criterion:** The Phase 2 winner's gain over baseline replicates within ± 30%.

**If only one architecture replicates:** report as architecture-specific finding. Still publishable.

**If neither replicates:** the Phase 2 winner was LLaVA-7B-specific. Important negative finding.

## Phase 3 — Interaction Studies (Weeks 9-10)

Combine winners from Phase 2. Test combinations:

| Combination | Expected behavior |
|---|---|
| E_sup1 (distillation) + E_arch1 (cross-attention query) | Both signals strengthen the selector |
| E_sup1 + E_arch2 (two-stage) | Distillation per stage |
| E_sup1 + E_arch4 (OCR dual path) | Distillation + specialized text path |
| B1 (frozen-prior) + E_arch4 | Hybrid: known-good base + OCR pathway |

**Decision:** Pick top 2 candidates for Phase 5.

## Phase 4 — Scaling Curve Studies (Week 11)

Before full-scale training, characterize the scaling curve.

- Token budget k ∈ {72, 144, 216, 288, 432}.
- LoRA rank ∈ {16, 32, 64, 128}.
- Training steps ∈ {10K, 20K, 40K, 80K, 160K}.

**Compute:** ~60 GPU-h.

**Deliverable:** Pareto curves with N=3 confidence intervals.

**Insight question:** Where on the Pareto curve does the candidate become competitive with FastV at matched inference cost?

## Phase 5 — Candidate Architecture Bakeoff (Weeks 12-13)

Top 2 candidates from Phase 3 trained to full convergence at the best operating point identified in Phase 4.

**Compute:** ~200 GPU-h per architecture × 3 seeds = ~1200 GPU-h.

**Only proceed if Phases 1-4 indicate the candidates have a path to dominance.** If not, skip and write up Phases 1-4 as a research paper (Track C or pivot paper).

## Phase 6 — Full-Scale Training (Weeks 13-14)

Train final candidate on the full LLaVA-1.5 SFT mix. Multi-seed (N=3).

**Compute:** ~300 GPU-h × 3 seeds = ~900 GPU-h.

## Phase 7 — Robustness and Generalization (Week 15)

- OOD evaluation: medical, document, satellite images.
- Adversarial query phrasing.
- Seed variance characterization.
- Compute-curve at inference time vs FastV (using Phase 0.5 infrastructure).

## Phase 8 — Final Benchmarking (Week 16)

- Full benchmark suite, full eval splits (not 5K subsets).
- RefCOCO, DocVQA, ChartQA added.
- Multi-seed reporting.
- Apples-to-apples timing on identical hardware.
- **Pareto FLOP vs accuracy plot.** This figure is the most important artifact of the paper.

## What NOT to do (v2 updated)

- **Do not run multi-seed for ablations.** Multi-seed is for headline numbers only.
- **Do not change multiple things between runs.** Single-variable comparisons only.
- **Do not run > 50K-step experiments in Phase 2.** Phase 2 is diagnostic.
- **Do not skip Phase 0-1.5.**
- **Do not commit to a track before Decision Gate (Phase 1.75).**
- **`[NEW v2]` Do not skip the synthetic-task gates.** A method that fails the needle test cannot do RefCOCO.
- **`[NEW v2]` Do not commit to Phase 5 if Phase 4 shows no Pareto improvement over FastV.** Pivot to writeup.
- **`[NEW v2]` Do not run Phase 6 before cross-architecture (Phase 2D) confirms taxonomy.**

## How to detect dead ends quickly

- If a Phase-2 experiment doesn't improve any of (T1, T3, T8, T10) by ≥ 25% relative, kill it.
- If a Phase-3 combination doesn't outperform the better single Phase-2 winner, the combination is decorative.
- If Phase-4 scaling curves show plateau below FastV's FLOPs-matched accuracy, no scaling pathway exists. Pivot.
- **`[NEW v2]` If a new architecture fails the needle test, abandon. Synthetic-task failure → real-task failure.**

## How to avoid self-deception

- **Pre-register** every Phase-2 experiment's success/failure criteria before running it.
- **Adversarial review:** every successful experiment gets a peer who tries to find a confound.
- **Hidden test set:** keep one 1K-sample evaluation set untouched. Final results report it; no decisions made on it.
- **Avoid moving goalposts.** If the experiment fails its preregistered criterion, it fails.
- **`[NEW v2]` Compute-bookkeeping discipline.** Every run logs GPU-h used. Quarterly review.

---

# 5. High-Upside Architectural Ideas

This section is the "if we have permission to redesign, what would we build" inventory. Sprint findings update the ROI rankings; new ideas added.

## Idea 1: Distilled Hybrid Selector (DHS) `[ROI: High (downgraded from Very High), Novelty: Medium, Risk: Low-Medium]`

**Mechanism.** Use full-context teacher (LoRA-576 or public LLaVA-1.5) to produce:
1. Per-token attention pattern at each LLM layer.
2. Output logit distribution.

Train joint loss:
- α · CE(student, target)
- + β · KL(student logits || teacher logits)
- + γ · ‖student_layer_l_attn - teacher_layer_l_attn‖² (visual-token-restricted)
- + δ · BCE(selector_mask, teacher_attn_top-k_mask)

**Why it might work.** End-task loss for the selector is too distal. Direct supervision on "which tokens did the full-context teacher actually use" replaces 33-layer credit assignment with 1-layer classification.

**Why downgraded from v1.** Two issues:
1. **Teacher quality.** LoRA-576 is only ~3 pp above random selection. A teacher this weak may be insufficient.
2. **Spearman ≈ 0 result.** If the TPN converges to a pathological local minimum, distillation may not escape it without strong initialization (E_sup3) too.

**Cheap test.** Compute teacher attention on 5K samples (cached). Train selector-only (LoRA frozen) on attention distillation. Measure Jaccard with teacher and downstream accuracy.

**What could fail.** Teacher attention too diffuse to be top-k target; teacher quality too low.

**Falsifier.** Selector-only distillation matches teacher attention but downstream accuracy doesn't improve.

## Idea 16 NEW v2: Frozen-Prior + Tiny Residual (FP-TR) `[ROI: Very High, Novelty: Medium, Risk: Low]`

**Mechanism.** Initialize selector as a *fixed* function of CLIP CLS-attention. Add a small (~50K params) MLP producing a score *adjustment*:

```
scores_t = CLS_attn + α_t · MLP(visual_features, question_embed)
```

Where α_t starts small (0.01) and slowly increases over training (capped at 0.5). The selector's output is provably close to PruMerge's at initialization; learning can only *refine*, not destroy.

**Why it might work.** Sprint data shows joint training pushes the TPN into a pathological basin uncorrelated with any useful prior. If we *prevent* the optimization from leaving the CLS-attention basin (via the α schedule and small residual capacity), we get PruMerge's training-free performance as a floor, plus the chance of improvement.

**Why no one tried it.** PruMerge / EViT use CLS-attention frozen. TC-ViTok learns from scratch. The hybrid — frozen prior + tiny residual — is the obvious gap that combines training-free's robustness with training-time's potential adaptability.

**Cheap test.** 4 GPU-h to train. Compare to PruMerge baseline. If it's strictly better, we have a positive result.

**Why this is a fail-safe.** Even if the residual learns nothing useful, the model still matches PruMerge — guaranteed-not-worse-than-training-free. The downside is bounded.

**Falsifier.** FP-TR matches PruMerge but does not exceed it across multiple seeds → no joint-training benefit even at small capacity → strong H_arch.

**Publication potential.** Combined with the existing taxonomy paper, FP-TR's working becomes a positive system contribution. If it doesn't work, that itself is a strong negative finding ("even minimal-capacity learned refinement of CLS-attention fails — pre-LLM compression beyond CLS-attention may be fundamentally hard").

## Idea 2: Two-Stage Progressive Compression with Per-Stage Objectives `[ROI: High, Novelty: Low, Risk: Medium]`

**Mechanism.** 576 → 288 → 144. Stage 1 trained with attention distillation only. Stage 2 trained with end-task + query conditioning.

**Why it might work.** Single 4× compression decision is overloaded — preserve image content AND attend to question. Splitting lets each be optimized cleanly.

**Cheap test.** Train stage 1 alone with attention distillation. Plug into LoRA-576 at k=288. If accuracy stays near LoRA-576 at k=288, stage 1 is sufficient.

**Falsifier.** Two-stage doesn't outperform single-stage at the same total compute.

## Idea 3: Per-Token Query Cross-Attention (PTQ-XA) `[ROI: High, Novelty: Medium, Risk: Low]`

**Mechanism.** Replace single CLIP text embedding with LLaVA text token embeddings. Each visual token cross-attends to every question token.

**Why it might work.** Single CLIP text embedding is the thinnest query channel. Per-token cross-attention is what SparseVLM uses inside the LLM.

**Cheap test.** Run T2 with new query channel. If per-image selection diversity grows substantially, query is being used.

**Falsifier.** Even with rich query channel, E3 zeroing doesn't hurt.

**Synthetic-task gate:** must pass needle test with location queries.

## Idea 4: OCR-Aware Dual Pathway (OADP) `[ROI: High for TextVQA, Novelty: Medium, Risk: Medium]`

**Mechanism.** Off-the-shelf OCR-saliency detector (CRAFT or frozen ViT) produces per-image OCR density. Above threshold: route through k=288 OCR pathway. Below: standard k=144.

**Why it might work.** TextVQA failure is structural. Sprint confirms: CLS-attention beats TPN on TextVQA by 10.8 pp on the same grid. The fix is to not compress text patches.

**Cheap test.** Apply OCR detector to TextVQA val. Route only patches with OCR density > threshold through k=576. Measure TextVQA Δ vs k=144.

**Falsifier.** TextVQA improvement < 5 pp even with full text-patch preservation.

## Idea 5: Register-Augmented TPN (RA-TPN) `[ROI: Low (downgraded), Novelty: Low, Risk: Low]`

Sprint data shows dead-zone doesn't replicate across seeds. Likely a symptom of weak supervision, not a cause. Registers may not help.

Keep on backlog; not Top-5.

## Idea 6: Learned Cross-Attention Merge `[ROI: Low (downgraded), Novelty: Low, Risk: Low]`

Sprint confirmed merge is shadowed by LoRA. Replacing the merge formula likely doesn't fix the shadowing. Keep on backlog.

## Idea 7: Token Recycling on LLM Uncertainty (TRU) `[ROI: Medium, Novelty: High, Risk: High]`

**Mechanism.** After one forward pass, measure LLM next-token uncertainty. If > threshold, run a second pass with augmented token set.

**Cheap test.** Identify high-uncertainty examples. Replace with k=216 selection. Measure delta.

**Falsifier.** Additional tokens don't help — bottleneck is elsewhere.

## Idea 8: Mixture-of-Selectors (MoS) `[ROI: Low, Novelty: Medium, Risk: High]`

Backlog. High engineering cost, modest expected gain.

## Idea 9: Latent Bottleneck Compression (Perceiver-IO-style) `[ROI: High (elevated), Novelty: Medium, Risk: Medium]`

**Mechanism.** 576 visual tokens cross-attend to L learnable latent tokens (L = 64 or 128). Latents are passed to LLM.

**Why elevated.** Sprint data suggests top-k itself may be the wrong abstraction. A content-driven latent bottleneck doesn't have the top-k pathology and may avoid the Spearman ≈ 0 trap.

**Cheap test.** Train; measure.

**Falsifier.** Perceiver-IO matches or exceeds at same parameter count — in which case top-k is provably the wrong abstraction.

## Idea 10: Causal Token Interventions for Selection (CTI) `[ROI: Very High (elevated), Novelty: High, Risk: Medium]`

**Mechanism.** For each example, ablate each token in turn; measure ΔLLM prediction. Per-token causal importance. Use as oracle.

**Why elevated to Top-5.** This is the cheapest test of the upper bound of pre-LLM compression. ~8 GPU-h for 200 samples.

**Outcome interpretation:**
- If causal importance is predictable from encoder features (R² > 0.5): distillation has a meaningful ceiling. Track A.
- If unpredictable (R² < 0.2): pre-LLM compression is fundamentally limited. Track C — publish that as flagship finding.

**Falsifier.** Causal scores can't be predicted from any encoder features → pre-LLM compression is provably hard at this scale.

## Idea 11: RL-Refined Selector (RLRS) `[ROI: Medium, Novelty: High, Risk: High]`

**Mechanism.** Supervised pretrain selector (E_sup1). Then RL fine-tune with reward = downstream accuracy.

**Cheap test.** RL fine-tune for 5K steps after supervised pretraining.

**Falsifier.** RL doesn't improve over supervised baseline.

## Idea 12: Retrieval-Augmented Token Memory (RATM) `[ROI: Low, Novelty: High, Risk: High]`

Backlog.

## Idea 13: Speculative Token Routing `[ROI: Medium, Novelty: Very High, Risk: Very High]`

Backlog.

## Idea 14: Direct Hidden-State Distillation `[ROI: Medium, Novelty: Medium, Risk: Low]`

**Mechanism.** Beyond attention distillation, distill LLM hidden states at selected layers. Force compressed model's hidden states to match teacher's.

**Why it might work.** Stronger supervisory signal than CE alone.

## Idea 15: Encoder Adaptation `[ROI: Medium (downgraded), Novelty: Low, Risk: Medium]`

**Mechanism.** Unfreeze encoder with LoRA.

**Why downgraded.** Sprint findings increase compute estimates everywhere; encoder adaptation is heavy.

## Idea 16 NEW v2: Frozen-Prior + Tiny Residual

See above. Top-5 candidate.

## Idea 17 NEW v2: Mid-LLM Compression + LoRA

**Mechanism.** Compress at LLM layer 2 (FastV-style), but with LoRA training on the compressed model. Combines FastV's recovery via early mixing with LoRA's adaptation.

**Why it might work.** Sprint findings suggest pre-LLM compression is structurally hard. Mid-LLM compression sidesteps this. FastV doesn't train; we add LoRA training to its compression mechanism.

**Cheap test.** Implement FastV pruning at layer 2 + LoRA training on the compressed model. Compare to FastV alone and to TC-ViTok.

**Falsifier.** Mid-LLM compression + LoRA doesn't beat FastV alone.

**Publication potential.** Novel combination; addresses the structural pre-LLM disadvantage by avoiding it entirely.

## Top-5 Updated Priority

1. **Idea 10 (CTI)** — cheap, dispositive, either resolves the project or seeds a flagship paper.
2. **Idea 16 (FP-TR, NEW)** — fail-safe architecture that cannot learn the pathological pattern.
3. **Idea 5 / E_sup3 (CLS-attn init)** — tests whether useful priors can be preserved under joint training.
4. **Idea 1 (DHS)** — distillation, conditional on teacher quality measurement.
5. **Idea 9 (Perceiver-IO bottleneck)** — content-driven alternative to top-k.

---

# 6. How to Actually Beat FastV and PruMerge

## 6.1 Why FastV wins (mechanism)

FastV wins because it operates *post early-layer mixing*. By layer 2:
- Visual feature embedding via MLP projector has happened.
- One or two layers of LLM self-attention have mixed visual into text-token residual streams.

Even after pruning visual tokens, the LLM still has access to information via the text-token residual stream. Pruning is "cheap" because the information is no longer purely on the dropped tokens.

**TC-ViTok operates before any of this.** When we drop a token pre-LLM, it is gone for all 32 layers.

## 6.2 Why PruMerge wins

1. Uses CLIP's CLS-attention as saliency prior. Strong, pretrained signal.
2. Merge happens to recover information well because the LLM is *not been adapted* — the merge fills information that the standard-trained LLM uses well.

**TC-ViTok's merge is shadowed by LoRA.** LoRA adapts to the post-merge representation; by the time we measure merge contribution, LoRA has already absorbed it.

## 6.3 The fundamental asymmetry

| Property | Post-LLM (FastV) / Mid-LLM (SparseVLM) | Pre-LLM (TC-ViTok) |
|---|---|---|
| Information access at prune time | LLM has mixed visual into text | Only frozen encoder features |
| Recovery if wrong | Possible via residual stream / recycling | None |
| Supervision available | Layer-internal attention is supervisor | Must learn from end task |
| Training requirement | None (FastV) or light | Heavy |
| Computational savings | Late-layer FLOPs only | All-layer FLOPs |

Pre-LLM compression saves more FLOPs but is structurally harder.

## 6.4 Concrete paths to dominance

We will not beat FastV on absolute accuracy at the same compression ratio without substantial structural changes. But we can win on at least four axes.

### 6.4.1 Win on FLOPs at matched accuracy

FastV operates after layer 2; we operate before layer 0. Full-stack FLOPs savings are larger. If we can match FastV's accuracy at *higher compression*, we win on the efficiency axis.

Path: 75% VQAv2 at k=72 (8× compression). FastV at k=72 would lose more accuracy due to less mixing-time.

**Tool required:** Pareto plot infrastructure (Phase 0.5).

### 6.4.2 Win on grounded tasks

FastV is query-agnostic. On tasks where the question specifies a region (RefCOCO, ChartQA), a query-conditioned method should win. We never tested this.

Path: Implement Idea 3 (PTQ-XA) + Idea 1 (Distillation). Evaluate on RefCOCO and ChartQA. **If we win by ≥ 5 pp on RefCOCO, we have a story.**

### 6.4.3 Win on OCR via specialization

FastV and PruMerge degrade gracefully on OCR but don't specialize. A dual-pathway architecture (Idea 4) could substantially exceed them.

Path: Implement Idea 4. Target TextVQA + DocVQA.

### 6.4.4 Win as a substrate for distillation

If Idea 1 works, TC-ViTok could be the substrate on which others build distilled compression. The taxonomy + working distillation recipe is a strong story.

### 6.4.5 NEW v2 — Win by predictability (causal-importance distillation)

If Idea 10 shows causal token importance is predictable from encoder features, *distill that signal into the selector*. The TPN's score function would be: "predict per-token causal importance from encoder features." This is gold-standard supervision — strictly better than CLS-attention or FastV-attn.

Path:
1. Compute causal importance on 5K samples (~50 GPU-h).
2. Train probe to predict from features.
3. Use predicted importance as direct selector supervision.
4. Compare to all training-free baselines.

**Success criterion:** beats FastV at matched-k by ≥ 2 pp on at least one benchmark.

### 6.4.6 NEW v2 — Win on cross-architecture portability

If the Phase 2D replication shows the winning architecture generalizes across VLM families, that's a robustness story FastV cannot match (FastV is layer-specific to particular LLM architectures).

### 6.4.7 Honest losers — directions to deprioritize

- **Beating FastV on absolute VQAv2 accuracy at k=144.** Hopeless without structural change. FastV has 4.8 pp lead; closing requires SFT mix change or hybrid approach.
- **Beating Q-Former.** Q-Former has massive multi-objective pretraining. We can't match compute.
- **Training-free TC-ViTok.** That's just PruMerge.

## 6.5 The pivot if necessary

If after Phase 2-3 we cannot demonstrate a win on at least one of (6.4.1 - 6.4.6), the honest move is to publish TC-ViTok as a diagnostic instrument — already supported by the existing taxonomy paper. The training-dynamics findings stand.

**v2 update:** The pivot is no longer "embarrassing." With Track C as an explicit alternative path, the "fundamental-limit" framing is itself a publishable contribution. The pivot becomes a deliberate venue/framing choice, not a defeat.

---

# 7. If We Restarted From Scratch

This section is a counterfactual: how would we have built TC-ViTok if we knew everything we know now?

## 7.1 Methodology philosophy

- **Validate before you scale.** Every component proven in isolation.
- **Use synthetic tasks first.** Needle task takes a week to build; reveals capacity issues immediately.
- **Distillation by default.** Any compression method trained on end-task loss alone starts at a disadvantage.
- **Multi-seed from day one.** Single-seed evaluation crisis is preventable.
- **Causal interventions over correlational observation.** Force selections, force queries, force LoRA freezes.
- **`[NEW v2]` Diagnostic-first ordering.** Measure before optimize.

## 7.2 Codebase philosophy

- **Smaller, more focused modules.** Each subsystem independently runnable.
- **Heavy logging by default.** Selection statistics, gradient norms, attention maps.
- **Configuration-as-code.** Every run uniquely determined by a config hash.
- **Diagnostic-first CLIs.** Force-mask, force-query, force-merge as first-class operations.
- **Toy mode for everything.** `--toy` flag finishes in 5 minutes on synthetic data.

## 7.3 Experimentation philosophy

- **Pre-register hypotheses.** Success criteria and falsifier before running.
- **One variable per run.** Always.
- **Held-out sets are sacred.** Final eval set never sees a decision.
- **Aggressive kill criteria.** No "just one more tweak."

## 7.4 Architecture-search philosophy

- **Start from the best prior.** CLS-attention prior + CLIP frozen + small selector.
- **Modular substitutability.** Every architectural choice (selector type, merge type, query channel) is a swap-in module.
- **Distillation is the substrate.** Every architectural choice evaluated *with* distillation.
- **`[NEW v2]` Fail-safe by construction.** When unsure, design architectures that cannot be worse than the best training-free baseline.

## 7.5 Scaling philosophy

- **Scale only when smaller works.** If 5K-step run on 10% of data doesn't show meaningful signal, scaling won't help.
- **Pareto curves over single operating points.** Always.
- **Compute logs.** Every paper claim about efficiency backed by measured FLOP / throughput.

## 7.6 Publication philosophy

- **Negative results are publishable if rigorous.** The current paper trends this way.
- **`[NEW v2]` Multi-paper strategy.** Don't bet everything on one big paper. Submit incrementally — a TMLR mechanism paper after Phase 1.5 is achievable in 8 weeks.
- **Failure-mode taxonomies require generality.** If we claim taxonomy, we owe at least two architectures' worth of confirmation. The current paper has Qwen-VL stress test only.
- **Match competitors compute-fairly.** A win requiring 10× the compute is not a win.

---

# 8. Master Plan — REVISED Prioritized Top-20 Experiments

The single deliverable of this document is this list. Rank the entire summer of work by ROI × probability × novelty.

## 8.1 The top 20 (v2)

| # | Experiment | Phase | Compute | Expected delta | Confidence |
|---|---|---|---|---|---|
| 1 | **Build diagnostic dashboard + causal-intervention CLI + measurement infrastructure** | 0 + 0.5 | ~6 GPU-h | Infrastructure baseline | H |
| 2 | **Run T1-T11 + T6.5 diagnostics on existing checkpoints** | 1 | ~15 GPU-h | Identifies root cause | H |
| 3 | **T_new1: TPN learned-distribution characterization** | 1.5 | ~7 GPU-h | Settles "what did TPN learn?" | H |
| 4 | **T_new2: Causal token importance probe (Idea 10)** | 1.5 | ~8 GPU-h | Determines pre-LLM compression upper bound | H |
| 5 | **Decision gate: choose Track A, B, or C** | 1.75 | 0 | Direction lock-in | — |
| 6 | **E_sup3: CLS-attention initialization + fine-tune** | 2A | ~15 GPU-h | Tests whether useful priors survive joint training | M |
| 7 | **B1: Frozen-prior + tiny residual (Idea 16, NEW)** | 2B (Track B) | ~20 GPU-h | Fail-safe; either matches PruMerge or reveals H_arch | M |
| 8 | **E_sup4: Full SFT mix (OCR-VQA, RefCOCO, etc.)** | 2A | ~60 GPU-h | +5-10 pp TextVQA, +3-5 pp absolute | H |
| 9 | **E_arch4: OCR-aware dual pathway** | 2B (Track A or B) | ~15 GPU-h | +8-12 pp TextVQA | M |
| 10 | **E_arch1: Per-token query cross-attention (PTQ-XA)** | 2B (Track A) | ~25 GPU-h | Query usage emerges | M |
| 11 | **E_sup1: Distillation from LoRA-576** | 2A (Track A) | ~30 GPU-h | +2-5 pp VQAv2, conditional on teacher quality | M |
| 12 | **B2: Q-Former-style latent bottleneck** | 2B (Track B) | ~30 GPU-h | Alternative architecture; tests if top-k is wrong | L |
| 13 | **B3: Mid-LLM compression + LoRA (NEW)** | 2B (Track B) | ~30 GPU-h | Sidesteps pre-LLM difficulty | M |
| 14 | **E_arch2: Two-stage progressive compression** | 2B (Track A) | ~30 GPU-h | +1-2 pp at matched k | M |
| 15 | **RefCOCO eval on best Phase-2 candidate** | 3 | ~6 GPU-h | Headline grounding number | H |
| 16 | **Cross-architecture replication (Phase 2D, NEW)** | 2D | ~100 GPU-h | Taxonomy validation | M |
| 17 | **Pareto curve scan k ∈ {72, 144, 216, 288}** | 4 | ~60 GPU-h | Pareto curve vs FastV | H |
| 18 | **Causal-importance distillation (Idea 10 → method)** | 3 | ~50 GPU-h | If T_new2 positive, this is the method | M |
| 19 | **Multi-seed N=3 on Final + Phase-2 winner** | 6 | ~900 GPU-h | Variance characterization, headline numbers | H |
| 20 | **Full-scale final candidate training** | 6 | ~900 GPU-h | Final paper experiment | H |

**Total budget (all tracks):** ~2350 GPU-h. Track-specific subsets: A ~1500, B ~1100, C ~600.

## 8.2 Top 5 "Most likely to change everything" (v2)

1. **#4 — Causal token importance (Idea 10).** Cheapest direct test of upper bound. Result determines which of three tracks to run.
2. **#3 — T_new1 (TPN learned-distribution).** Cheapest characterization of what we actually built. Result distinguishes H_opt vs H_arch.
3. **#7 — Frozen-prior + tiny residual (Idea 16, NEW).** Fail-safe; cannot be worse than PruMerge. Provides positive lower-bound result regardless of outcome.
4. **#8 — Full SFT mix.** TextVQA gap is structural data. Highest absolute-number ROI for Track A.
5. **#11 — Distillation from LoRA-576.** Tests H_opt if teacher quality permits.

## 8.3 Top 5 "Probably dead ends" (v2)

1. **Register tokens alone.** Treats symptom not cause. Sprint reinforced this.
2. **MoE selector.** High engineering cost; modest expected gain.
3. **Retrieval-augmented token memory.** Adds complexity without clear path to dominance.
4. **RL alone (without good supervised pretraining).** Cold-start RL on selection won't work.
5. **Speculative token routing.** Engineering-heavy; unlikely to outperform a well-tuned single-pass selector.

## 8.4 Top 5 "Cheap but extremely informative" (v2)

1. **#1 — Dashboard / causal CLI / measurement infrastructure.** Zero GPU; enormous downstream value.
2. **#2 — Diagnostic battery (T1-T11 + T6.5).** ~15 GPU-h reveals root cause.
3. **#4 — Causal token importance.** ~8 GPU-h tells you whether pre-LLM is hopeless.
4. **#3 — TPN learned-distribution.** ~7 GPU-h tells you what was actually learned.
5. **Synthetic-task suite (Phase 0).** Lets you verify any new architecture's capacity in 10 minutes.

## 8.5 Top 5 "Highest publication potential" (v2)

1. **#4 — Causal token interventions.** Positive or negative result is publishable (Track C is built on this).
2. **#11 + #15 — Distillation + RefCOCO.** If both work, top-tier method paper at BMVC/NeurIPS.
3. **#7 — Frozen-prior + tiny residual.** If this works, a clean positive system finding.
4. **#9 — OCR-aware dual pathway.** Specialized win on TextVQA + DocVQA; deployment-relevant.
5. **Phase 1 + 1.5 root-cause analysis.** Repositioning current paper from "we observed X" to "we causally identified Y" via diagnostics is itself a TMLR-grade revision.

## 8.6 Execution timeline (v2 revised — 4-month sprint, branching)

| Weeks | Phase | Focus | Compute budget |
|---|---|---|---|
| 1 | Phase 0 + 0.5 | Infrastructure + measurement | ~6 GPU-h |
| 2 | Phase 1 | Diagnostics on existing checkpoints | ~15 GPU-h |
| 3 | Phase 1.5 | TPN-learned-distribution + causal-importance | ~15 GPU-h |
| 4 | Phase 1.75 + 1.9 | Decision gate + ICONIP review response | 0 |
| 5-10 | Phase 2 | Track-specific experiments | Track A: ~250 GPU-h, Track B: ~150 GPU-h, Track C: ~120 GPU-h |
| 8-10 | Phase 2D (parallel) | Cross-architecture replication | ~100 GPU-h |
| 9-10 | Phase 3 | Interaction studies + RefCOCO | ~50 GPU-h |
| 11 | Phase 4 | Scaling curve scan | ~60 GPU-h |
| 12-13 | Phase 5 | Bakeoff (top 2 candidates × 3 seeds) | ~1200 GPU-h |
| 13-14 | Phase 6 | Full-scale + multi-seed | ~900 GPU-h |
| 15 | Phase 7 | Robustness | ~40 GPU-h |
| 16 | Phase 8 | Final benchmarking + writeup | ~40 GPU-h |

**Total budget (Track A, the heaviest):** ~2700 GPU-h over 16 weeks. At 2 GPUs available, that's ~28 days of full utilization spread over 4 months.

**Track B compute:** ~1500 GPU-h.
**Track C compute:** ~700 GPU-h.

## 8.7 Decision gates (v2)

At each phase boundary, an explicit go/no-go.

- **End of Phase 1.5:** Choose Track A / B / C per Section 10. **Do not proceed without this choice.**
- **End of Phase 2:** Do winners produce ≥ +2 pp on at least one benchmark OR ≥ 1 pp on a flagship synthetic task? If no, return to diagnostics or pivot to writeup.
- **End of Phase 2D:** Does the Phase 2 winner replicate cross-architecture within ± 30%? If no, scope the contribution to single-architecture.
- **End of Phase 3:** Do combined candidates outperform best single Phase-2 by ≥ +1 pp? If no, the combinations are decorative; stick with single best.
- **End of Phase 4:** Is there a Pareto point matching or exceeding FastV? If no, do NOT proceed to Phase 5 with the goal of beating FastV. Pivot to TMLR-grade mechanism paper.

## 8.8 The minimum publishable continuation

If only 5 things get done over the summer, they should be:

1. **#1 — Infrastructure + measurement.** (Necessary to publish anything else.)
2. **#2 — Diagnostics.** (Identifies root cause.)
3. **#3 + #4 — T_new1 + T_new2.** (Characterize what TPN learned and the upper bound.)
4. **#7 — Frozen-prior + tiny residual (Idea 16).** (Fail-safe positive system finding.)
5. **#11 — Distillation experiment.** (Tests the H_opt hypothesis.)

These five constitute a strong follow-up paper at TMLR or BMVC. If they succeed beyond minimum threshold, architectural extensions (Section 5) become a third paper.

---

# 9. NEW v2 — Multi-Paper Publication Strategy

Betting everything on one big paper at the end of a 4-month sprint is fragile. v2 proposes a hedged, multi-paper strategy that produces value at multiple checkpoints.

## 9.1 The three-paper portfolio

| Paper | Trigger | Content | Target venue | Timeline |
|---|---|---|---|---|
| **P0** (already submitted) | — | Failure-mode taxonomy v1 + trivial-baseline negative result + Qwen-VL stress test | ICONIP | Submitted |
| **P1** | After Phase 1.5 | Causal-importance findings + diagnostic methodology + "what joint pre-LLM compression actually learns" | TMLR or NeurIPS Workshop | Month 2 |
| **P2** | After Phase 5 | Either method paper (if Track A succeeds) or fundamental-limit paper (if Track C runs) | BMVC / WACV / NeurIPS | Month 4 |

## 9.2 P1 — The mechanism paper

**Premise.** Phase 1 + 1.5 diagnostics produce a precise mechanistic account of *why* TC-ViTok fails. Whether or not we find a method that wins, the diagnostic findings are themselves a contribution.

**Content:**
- The Spearman ≈ 0 finding with multi-prior correlation analysis.
- The T6.5 LoRA-on-random vs LoRA-on-TPN result.
- The causal-importance probe and its R².
- The synthetic-task suite as a diagnostic toolkit.
- A precise characterization of what joint training converges to.

**Venue rationale.** TMLR explicitly accepts mechanism / negative-results papers with rigorous methodology. The "we measured what joint training actually produces" angle is exactly TMLR's wheelhouse. Expected acceptance: 70-85%.

**Compute cost:** ~50 GPU-h beyond Phase 1.5. Mostly writeup.

**Timeline:** Submit by month 2.

## 9.3 P2 — The follow-up paper

Two possible directions depending on Track outcome:

### P2.A — Method paper (if Track A succeeds)

**Content:** TC-ViTok-v2 with distillation + per-token query cross-attention + OCR pathway. Pareto plot vs FastV. Cross-architecture replication.

**Venue:** BMVC / WACV / NeurIPS main track.

### P2.B — Fundamental-limit paper (if Track C runs)

**Content:** Causal-importance unpredictability from encoder features + cross-architecture verification + implications for pre-LLM compression generally.

**Venue:** NeurIPS / ICML / TMLR main track.

### P2.C — Hybrid (if Track B partially succeeds)

**Content:** Frozen-prior + tiny residual as positive system finding + taxonomy v2 with cross-architecture confirmation.

**Venue:** BMVC / WACV.

## 9.4 Risk-adjusted publication path

| Outcome | Papers published |
|---|---|
| Everything works (Track A succeeds) | P0 + P1 + P2.A — three papers |
| Phase 1.5 only | P0 + P1 — two papers |
| Track A fails, Track B partial | P0 + P1 + P2.C — three papers |
| Track A fails, Track C runs | P0 + P1 + P2.B — three papers |
| Total failure (Phase 1.5 produces no novel finding) | P0 alone |

**Expected paper count:** 2-3 papers from a 4-month sprint. This is a much better risk profile than the single-paper bet.

## 9.5 Publication discipline

- Each paper has a pre-registered scope and target venue at the start of its writing window.
- Each paper has at least N=3 multi-seed for headline numbers.
- Each paper includes the Pareto FLOP comparison with FastV.
- Each paper includes negative results honestly.

---

# 10. NEW v2 — Decision Tree After Phase 1

The plan branches on Phase 1.5 findings. Without this decision tree, Phase 2 commits before evidence.

## 10.1 Diagnostic outcomes

Phase 1 + 1.5 produce four key numbers:

- **D1:** T1 selector consistency Jaccard (across seeds). Range [0, 1].
- **D2:** T6.5 LoRA-on-random accuracy vs LoRA-on-TPN accuracy. Sign matters.
- **D3:** T_new1 maximum Spearman correlation of TPN scores with any tested prior (CLS, FastV, gradient, position). Range [-1, 1].
- **D4:** T_new2 R² of probe predicting causal token importance from encoder features. Range [0, 1].

## 10.2 The decision tree

```
START

├─ D4 ≥ 0.5 ─────────────────────────────────┐
│  Causal-importance is predictable.         │
│  Distillation has a meaningful ceiling.   │
│                                            │
│  ├─ D3 ≥ 0.4 ─→ Track A.1                 │
│  │  TPN learned a useful prior but is     │
│  │  under-supervised. Run distillation.   │
│  │  Compute: ~800 GPU-h.                  │
│  │                                         │
│  └─ D3 < 0.4 ─→ Track A.2                 │
│     TPN learned nothing useful but could.  │
│     Run E_sup3 (CLS-init) + E_sup1.       │
│     Compute: ~1000 GPU-h.                 │
│                                            │
├─ 0.2 ≤ D4 < 0.5 ─────────────────────────┐
│  Causal-importance is partially            │
│  predictable. Mixed signal.                │
│                                            │
│  ├─ D2 < 0 (random beats TPN-LoRA) ─→     │
│  │  Track B.                              │
│  │  Architecture is wrong; redesign.      │
│  │  Run B1 (frozen-prior + tiny residual) │
│  │  and B2 (Q-Former).                    │
│  │  Compute: ~600 GPU-h.                  │
│  │                                         │
│  └─ D2 ≥ 0 ─→ Track A.3                   │
│     LoRA matters; TPN matters partially.   │
│     Distillation + arch tweaks.           │
│     Compute: ~800 GPU-h.                  │
│                                            │
├─ D4 < 0.2 ─→ Track C                      │
│  Causal-importance is unpredictable.       │
│  Pre-LLM compression is bounded.           │
│  Publish that as flagship finding.         │
│  Compute: ~150 GPU-h + writeup.            │
│                                            │
END
```

## 10.3 What each track commits to

### Track A — Method paper

Focus on supervision and architecture together. Top experiments: E_sup1, E_sup3, E_sup4, E_arch1, E_arch4. Phase 2D cross-architecture replication. Phase 4 Pareto curve. Phase 5 bakeoff. Phase 6 multi-seed training. Goal: beat FastV on at least one axis.

Compute: ~1500-2700 GPU-h over months 2-4. Three-paper portfolio: P0 + P1 + P2.A.

### Track B — Architectural redesign

The trainable-bridge framework is fundamentally limited. Pivot to alternative architectures.

Top experiments: B1 (frozen-prior + tiny residual), B2 (Q-Former bottleneck), B3 (mid-LLM + LoRA). Synthetic-task gates for each. Phase 4 Pareto. Phase 5 bakeoff if any candidate passes synthetic gates.

Compute: ~1000-1500 GPU-h. Two- or three-paper portfolio: P0 + P1 + P2.C.

### Track C — Fundamental-limit paper

Comprehensive characterization of the upper bound on pre-LLM compression. Multi-encoder verification.

Top experiments: C1 (5000-sample causal analysis), C2 (mid-LLM activation predictability), C3 (multi-encoder verification). Phase 7 OOD robustness.

Compute: ~600-700 GPU-h. Two-paper portfolio: P0 + P1 + P2.B.

## 10.4 Track switching

If a track is committed to but shows no progress within 3 weeks of Phase 2, the team should re-run Phase 1.75 with updated data. Tracks are not irrevocable.

## 10.5 Cross-track shared deliverables

Regardless of track, the following are always produced:

- Diagnostic dashboard.
- Synthetic-task suite as a public toolkit.
- Causal-importance probe results (any track that runs it).
- Pareto FLOP comparison infrastructure.
- N=3 multi-seed for any headline number.

---

# 11. Concluding Notes

The honest summary of the project, post-sprint:

- The current paper (P0) is publishable because it does honest negative-results work.
- The architecture as currently designed converges to a pathological local minimum that is worse than random selection at every measured budget.
- The path forward is not a single architectural sprint. It is a diagnostic-first, three-track branching strategy.
- The single highest-leverage non-architectural change is **adding stronger supervision via distillation, conditional on teacher quality**.
- The single highest-leverage architectural change is **a fail-safe design (Idea 16: frozen-prior + tiny residual) that cannot learn the pathological pattern**.
- The single highest-priority methodological change is **subsystem validation + synthetic-task gates before scaling**.
- **The single highest-leverage publication change is the multi-paper strategy (P0 + P1 + P2) rather than a single big paper bet.**

The 4-month sprint above is achievable. The deliverable at the end is:
- A genuinely competitive query-conditioned compression method (Track A success path);
- Or a clean positive system finding from frozen-prior architectures (Track B partial success);
- Or a definitive negative result on pre-LLM compression bounded by causal-importance unpredictability (Track C path).

All three are publishable at multiple venues. The portfolio strategy ensures at least P0 + P1 produce papers regardless.

**Submit ICONIP. Then start at Phase 0.**

**The most important sentence in this document, restated:**

> The Spearman ≈ 0 finding is the most important data point we have. It says the joint training of pre-LLM trainable selectors converges to something uncorrelated with any obvious useful prior. Whether this is fixable (Track A), bypassable (Track B), or fundamental (Track C) is the question Phase 1.5 answers. Everything else in this plan is downstream of that answer.
