# Reviewer Prompt: TC-ViTok Final Acceptance-Maximization Review

You are acting as a strict but constructive ICONIP 2026 reviewer and meta-reviewer. Review the attached paper with one goal only: maximize its acceptance probability before submission. Be technical, skeptical, and specific. Do not spend space on generic praise unless it changes the strategic recommendation.

Assume the authors can still make text edits, table/caption edits, and small presentation changes. Assume no major new experiments can be completed unless you explicitly argue that one is essential. The paper is currently at the 15-page LLNCS limit, so every addition should come with a suggested cut.

## Paper Context

The paper is framed as a controlled negative-result / diagnostic study, not as a new state-of-the-art compression method.

Core setup:
- Method: `TC-ViTok`, a query-conditioned pre-LLM visual token selector.
- Stack: LLaVA-1.5-7B + CLIP ViT-L/14 visual tokens + LoRA adapters.
- Compression: 576 visual tokens to `k=144` before the LLM.
- Selector: TPN using DINOv2-S coarse image features plus CLIP text/query embedding.
- Training: selector, LLaVA projector, and LoRA adapters are jointly trained.
- Main negative result: the learned query-conditioned selector loses to a matched no-compression `LoRA-576` reference and fails to surpass the strongest question-blind trivial selector on any evaluated benchmark.
- Intended contribution: a controlled negative result plus three failure-mode hypotheses and reproduction protocols:
  1. FM1: co-adaptation oscillation in active trainable selector/pooler bridges.
  2. FM2: adaptive BudgetPredictor collapse and a stable adaptive-budget window.
  3. FM3: warm-start dependence under end-task supervision alone.

The authors want to know whether the current draft is now acceptance-ready, what still threatens acceptance, and exactly what to edit next.

## Important Current State of the Draft

Please treat the following as the current paper state:

- Title: “Why Jointly Learned Pre-LLM Visual Token Selectors Fail: A Diagnostic Study.”
- The paper explicitly says evidence is primarily from one LLaVA-1.5-7B stack.
- The Qwen2-VL attempt is now framed as a failed portability attempt: it did not reach non-trivial task accuracy and is not used as cross-architecture validation.
- The paper no longer claims a broad taxonomy or universality.
- The abstract and main results now use the factually safe claim: the learned TPN fails to surpass the strongest question-blind trivial selector on any evaluated benchmark.
- The earlier factual error has been fixed:
  - On POPE, TPN slightly beats CLS-attention top-k but loses to random/grid.
  - On MMBench, TPN beats random/grid but is within noise of CLS-attention top-k and does not surpass it.
- MME has been removed from the experimental setup because it is not in the result table.
- MMBench and TextVQA standard-error caveats have been added.
- NTP is defined as next-token-prediction loss.
- Negative `R^2` in probe results is explained as worse-than-mean prediction.
- The N=100 attention analysis is explicitly called a pilot smoke test.
- The dead-zone / selection-frequency figure and paragraph were cut to recover page budget.
- The paper currently compiles to exactly 15 pages.

## Main Evidence Now in the Paper

### Matched No-Compression Control

The paper introduces `LoRA-576`: the same LoRA adapter, projector, data, optimizer, rank, alpha, dropout, target modules, and checkpoint step as `TC-ViTok Final`, but forwarding all 576 visual tokens. This isolates compression cost from fine-tuning effects.

Main gaps to `LoRA-576`:
- VQAv2: TC-ViTok 72.27 vs LoRA-576 76.38, gap 4.11 pp.
- GQA: TC-ViTok 67.12 vs LoRA-576 70.55, gap 3.43 pp.
- POPE: TC-ViTok 79.09 vs LoRA-576 86.46, gap 7.37 pp.

### Trivial Selector Baselines

All trivial selectors use the same LoRA adapter and same hard-drop path at `k=144`.

Key values:
- VQAv2:
  - TPN 72.27
  - random 74.05
  - grid 74.65
  - CLS-attention 74.91
- GQA:
  - TPN 67.12
  - random 70.40
  - grid 69.26
  - CLS-attention 68.16
- POPE:
  - TPN 79.09
  - random 83.92
  - grid 82.78
  - CLS-attention 78.94
- MMBench:
  - TPN 62.0
  - random 61.24
  - grid 61.93
  - CLS-attention 62.05
  - Difference to CLS is only 0.05 pp and is within binomial noise.
- TextVQA:
  - TPN 25.8
  - random 29.63
  - grid 29.61
  - CLS-attention 36.57
  - CLS exceeds TPN by 10.77 pp.

The paper’s safe claim is: the TPN never surpasses the strongest trivial selector on any evaluated benchmark.

### k-Scaling Sweep

The learned TPN remains below the strongest trivial selector at every measured `k` on VQAv2, GQA, and POPE for `k ∈ {36,72,144,216,288}`. The figure caption defines “strongest” as the per-k maximum of random, grid, and CLS-attention top-k.

### STE Correction Evidence

The paper includes three pieces of evidence:

50K Phase-1 ablation:
- With STE correction: final validation NTP loss 0.3511.
- Without STE correction: final validation NTP loss 0.3846.

12K Phase-1 gradient diagnostic:
- With STE correction at tau≈0.10:
  - final TPN gradient norm 0.0324
  - LoRA gradient norm 0.4091
- Without STE correction at tau≈0.10:
  - final TPN gradient norm 0.0030
  - LoRA gradient norm 0.4277

tau_min=0.5 diagnostic without STE:
- final TPN gradient norm 0.0029
- validation NTP loss 0.4069 at 12K
- The text explicitly says the 12K tau05 validation loss is not directly comparable to the 50K no-STE result; the matched evidence is the TPN gradient norm.

### FM1 Evidence

Current framing:
- FM1 is a prediction, not a theorem.
- The paper says oscillations are consistent with co-adaptation dynamics but do not uniquely identify them; alternative optimization-path explanations remain possible.
- Seven active trainable-pooler configurations showed oscillatory validation-VQA trajectories.
- The frozen-bridge control did not oscillate but produced fixed noise tokens the LLM could not use.
- A 25x selector learning-rate reduction lengthened period from roughly 5K to roughly 15K steps.
- The text no longer claims the period changes with either group’s learning rate; it only discusses selector LR because that is what was varied.
- The parameter-free informative merge is described as an architectural removal of the trainable bridge, not an accuracy-improving contribution.

### FM2 Evidence

The FM2 table now includes seed-42 and seed-7 values:

- λ_b=0.03:
  - seed-42: k_eff=287.97, best Val=0.3898
  - seed-7: k_eff=288.00, best Val=0.2466
  - outcome: collapsed in both seeds
  - Important nuance: collapse is defined as budget saturation to full budget, not necessarily worse validation loss in every seed.
- λ_b=0.08:
  - seed-42: k_eff=280.15, best Val=0.3368
  - seed-7: k_eff=281.83, best Val=0.3285
  - outcome: stable in both seeds
- λ_b=0.10:
  - seed-42: k_eff=271.00, best Val=0.3177
  - seed-7: k_eff=270.37, best Val=0.3391
  - outcome: stable in both seeds
- λ_b=0.15:
  - seed-42: k_eff=247.23, best Val=0.3176
  - seed-7: k_eff=245.71, best Val=0.3078
  - outcome: stable in both seeds
- λ_b=0.20:
  - seed-42: k_eff=227.85, best Val=0.3291
  - seed-7: not run
  - outcome: stronger budget pressure

Current FM2 wording:
- Lower collapse at λ_b=0.03 is replicated in both seeds.
- Stable non-collapsed region λ_b ∈ [0.08, 0.15] is replicated in both seeds.
- λ_b=0.20 remains single-seed and is described only as stronger budget pressure.

Please evaluate whether this is now enough to neutralize the earlier FM2 replication criticism.

### FM3 Evidence

Current framing:
- FM3 is the weakest of the three hypotheses.
- The paper does not claim to solve FM3.
- It says from-scratch joint training plateaus above the warm-started ceiling under end-task supervision alone.
- Distillation, oracle importance, or strong selector priors are described as untested falsification paths, not completed mitigations.
- The reproduction protocol gives a concrete falsification criterion: FM3 is falsified in this stack if a distillation-augmented random-init run reaches Val-VQA ≤ 0.33 within 80K steps.

Warm-start table:
- warm-start Phase-1 checkpoint: best Val-VQA 0.3177 at 40K.
- from-scratch v1: 0.4915 oscillating at 48K.
- from-scratch v3: 0.4380 → 0.39 regressing at 40K.
- image-only from-scratch v3: 0.3334, still descending at 50K, single seed.
- Footnote now clarifies this approached a from-scratch plateau near 0.33, not the warm-started 0.3177 ceiling.

## Your Review Task

Please produce a review under the sections below. Be precise and acceptance-oriented.

### 1. Overall Acceptance Assessment

Give:
- current estimated acceptance probability for ICONIP 2026;
- likely reviewer scores by category: novelty, empirical rigor, technical soundness, clarity, significance, reproducibility;
- whether the paper now reads as a credible negative-result / diagnostic paper;
- whether it is currently weak accept, borderline, or weak reject.

### 2. Top Remaining Rejection Risks

Rank the top 5 remaining reasons a technically competent reviewer might reject the current draft.

For each risk, state:
- severity: fatal / critical / moderate / minor;
- fixability: text only / table-caption edit / requires experiment / cannot fix now;
- exact location if identifiable;
- exact replacement wording or table-caption change if possible.

Pay special attention to:
- single-stack limitation;
- FM3 distillation not tested;
- whether FM2 is now sufficiently replicated;
- whether λ_b=0.03 “collapse” is confusing because seed-7’s best validation loss is low despite k_eff=288;
- whether the Qwen2-VL failed portability attempt should remain in abstract/limitations or be removed entirely;
- whether MMBench 0.05 pp is handled safely;
- whether the paper still sounds too universal despite its caveats.

### 3. Claim-Evidence Consistency Audit

Audit for every remaining claim-evidence mismatch.

Specifically check:
- title;
- abstract;
- contribution list;
- FM1 prediction/mechanism/signature/remedy;
- FM2 prediction/signature/table/remedy;
- FM3 prediction/table/remedy/reproduction protocol;
- main results text;
- k-scaling claim;
- STE/tau05 paragraph;
- probe paragraph;
- limitations;
- conclusion.

For every mismatch, give exact edit text.

### 4. FM2 Table and Text Review

Review the FM2 evidence after seed-7 λ_b=0.03 and 0.15 are integrated.

Please answer:
- Is the table design with final k_eff and best validation NTP loss acceptable?
- Does the low seed-7 Val at λ_b=0.03 undermine calling it “collapsed,” or is the budget-saturation definition enough?
- Should the table include final validation loss, last-5 mean, or a footnote instead?
- Is “[0.08, 0.15] two-seed replicated non-collapsed region” now safe?
- Should λ_b=0.20 remain in the table as single-seed “stronger budget pressure,” or should it be removed/footnoted to avoid asymmetry?
- Give exact preferred Table 2 caption wording.

### 5. FM3 Distillation Decision

The paper does not include a clean FM3 distillation experiment.

Please decide:
- Is this acceptable for ICONIP given the current framing?
- Does the current falsification-protocol wording make FM3 scientific enough?
- Should the authors explicitly say “we do not test this mitigation” in the abstract, body, or limitations?
- Would adding an underpowered distillation run help or hurt?
- What is the minimum credible distillation experiment if the authors later decide to run one?

### 6. Page-Budget and Presentation Review

The paper is exactly 15 pages.

Please identify:
- any low-value text still worth cutting;
- any missing high-value clarity aid that is worth adding only if space is created;
- whether a pipeline diagram is still worth forcing in;
- whether any table is too dense or likely to confuse;
- any caption-level fixes that improve clarity without page cost.

### 7. Writing and Positioning

Review acceptance-oriented positioning:
- Does the title oversell “selectors fail” given single-stack scope?
- Does the abstract foreground the LoRA-576 control enough?
- Does the introduction explain why the negative result is informative rather than obvious?
- Does the conclusion leave a useful practical takeaway?
- Is the tone appropriately negative-result/diagnostic rather than failed-method defensive?
- Is the matched LoRA-576 control emphasized enough as a methodological contribution?

### 8. Final Action List

Give a prioritized list:
- Must Do Before Submission;
- Should Do If Time Allows;
- Optional;
- Do Not Do.

For each item, include expected acceptance-probability impact in percentage points if possible.

### 9. Final Verdict

End with:
- current verdict: accept / weak accept / borderline / weak reject / reject;
- estimated acceptance probability after current FM2 integration;
- whether any remaining issue is fatal;
- the single highest-value edit to make next.

Be strict. Assume reviewers are skeptical, detail-oriented, and likely to check numbers against tables. The output should be actionable enough that the authors can directly edit the paper from your recommendations.
