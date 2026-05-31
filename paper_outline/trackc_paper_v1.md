# Causal Token Importance is Unpredictable from Frozen LLM Representations

## Abstract

We measure per-token causal visual importance in a frozen LLaVA-1.5-7B and ask
whether it can be predicted from common compression-relevant representations.
Under linear and MLP probes with bootstrap confidence intervals, CLIP encoder
features, visual-token residual streams, and text-to-visual attention all fail to
predict causal importance. This reframes visual-token compression from importance
identification toward diversity preservation.

## 1. Introduction

- Visual-token compression often assumes useful tokens can be identified before or
  during early LLM computation.
- We test this assumption with ablation-defined causal importance.
- The core finding is negative across representation families.

## 2. Related Work

- Pre-LLM compressors: ToMe, EViT, PruMerge, TC-ViTok.
- Mid-LLM compressors: FastV, SparseVLM.
- Saliency, attribution, and token-importance probing.

## 3. Method

### 3.1 Per-token causal importance via ablation
### 3.2 Probe formulation and image-level bootstrap confidence intervals
### 3.3 Representations tested
### 3.4 Target variants and robustness checks

## 4. Encoder Feature Unpredictability

- C1 N=1000 linear and MLP results.
- Per-question-type breakdown.
- Target formulation diagnostics.

## 5. Visual-Token Residual Unpredictability

- L0, L4, L16, L28 visual residual probes.
- No onset curve; MLP remains negative and linear worsens with depth.

## 6. Attention Weight Unpredictability

- Text-to-visual attention at L1, L2, L16.
- L2 is FastV-relevant and still does not track causal importance.

## 7. Implication: Diversity Preservation, Not Importance Identification

- Compression methods may succeed by preserving broad recoverable evidence rather
  than selecting causally important tokens.
- This changes the design target for future pre-LLM and mid-LLM compression.

## 8. Supporting Phase 1 Evidence

- Seed-dependent selector solutions.
- Query-agnostic top-k behavior.
- Below-random FastV-oracle overlap.
- Coverage collapse into hot and dead positions.

## 9. Limitations

- LLaVA-1.5-7B and GQA val focus.
- Single-token ablation target.
- Linear and 2-layer MLP probes are strong but not exhaustive.
- Attention smoke is N=100 and should be framed as a gate, not a full sweep.

## 10. Cross-Encoder Verification (DINOv2)

**Motivation.** The C1 result is CLIP-specific by construction. A reviewer could argue CLIP's contrastive objective discards fine-grained spatial structure, and richer self-supervised encoders could recover the causal-importance signal.

**Design.** We repeat the C1 probe protocol on DINOv2-large (self-supervised ViT-L, 1024-dimensional patch features). Same 1000 GQA images, same LoRA-576 single-token-ablation targets, same linear+MLP probes, bootstrap n=200. Native DINOv2 patch tokens are bilinearly resized to 24×24. Prefix tokens are removed as needed; remaining patch tokens are validated to form a square grid before resizing.

**Results.**
- Linear: bootstrap mean R² = −0.076, 95% CI [−0.127, −0.038]
- MLP: bootstrap mean R² = −0.298, 95% CI [−0.424, −0.206]

Both CIs strictly below zero. Mirrors C1: increasing probe capacity makes R² more negative rather than recovering signal — signature of overfitting noise, not learning causal importance.

**Interpretation.** The encoder-feature unpredictability is not solely an artefact of CLIP's feature space. DINOv2's self-supervised spatial representations, trained with a different objective, also fail to recover the causal-importance distribution defined by LLaVA-1.5-7B. We do not claim all encoders must fail; two representative encoder families commonly used in visual-token compression both fail under standard linear and MLP probes.

**Artifact.** `runs/all_results_consolidated.json` updated: 9 representation rows (C1, C2 at L0/L4/L16/L28, C3 DINOv2, C4 at L1/L2/L16) = 18 probe measurements total.

## 11. Conclusion
