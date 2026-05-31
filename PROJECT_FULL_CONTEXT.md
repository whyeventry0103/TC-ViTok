# TC-ViTok Full Project Context

Last updated: 2026-05-31

This document is the single handoff file for TC-ViTok. It is written so a future
LLM, collaborator, reviewer-response agent, or maintainer can read one file and
understand the project framing, infrastructure, experiments, results, failures,
and paper state.

Important: this file supersedes older broad summaries such as `everything.md`
and `documentation/PROJECT_STATE.md` where they conflict with the final paper.
Those older files are useful historical records, but some of their framing treats
TC-ViTok as a successful compression method. The current paper is explicitly a
controlled negative-result and diagnostic case study.

## 1. One-Paragraph Summary

TC-ViTok tests a natural idea: learn a query-conditioned visual token selector
that compresses 576 CLIP-L/14 image patches to k=144 before LLaVA-1.5-7B LoRA
training, so the LLM receives fewer visual tokens and pays less visual prefill
cost. The main result is negative but controlled. The learned selector loses to a
compute-matched no-compression LoRA-576 reference and fails to surpass the
strongest question-blind trivial selector on all five evaluated benchmarks. The
paper uses this failure as a diagnostic case study and identifies three
failure-mode (FM) hypotheses: coupled selector-adapter oscillation, adaptive
budget collapse, and warm-start dependence. The evidence is primarily from one
LLaVA-1.5-7B stack, with a failed Qwen2-VL portability attempt reported only as
non-validation.

## 2. Current Paper Framing

Current title:

`Why Jointly Learned Pre-LLM Visual Token Selectors Can Fail: A Diagnostic Case Study`

Current intended venue:

ICONIP 2026, Springer LLNCS format, 15 page limit.

Current source:

`ICONIP2026_G5_TCVitok/main.tex`

Current compiled PDF:

`ICONIP2026_G5_TCVitok/main.pdf`

Core framing rules:

- TC-ViTok is not proposed as a competitive compression method.
- The contribution is a controlled negative result plus failure-mode diagnosis.
- Evidence is primarily single-stack: LLaVA-1.5-7B plus CLIP ViT-L/14 plus TPN.
- Claims must be scoped to "in this stack" or "in stacks similar to this one"
  unless a result is directly benchmark-measured.
- The learned selector does not surpass the strongest trivial baseline on any
  measured benchmark. This is the safe central claim.
- Do not claim it loses to every trivial selector on every benchmark. That is
  false because TC-ViTok slightly beats CLS-attention on POPE and is within noise
  of CLS on MMBench.
- Do not claim FM3 is solved. Distillation and oracle importance are untested
  falsification paths, not completed mitigations.
- Do not use the Qwen2-VL attempt as cross-architecture validation. It failed to
  reach non-trivial task accuracy.

## 3. Key Results at a Glance

### 3.1 Main Benchmark Table

All TC-ViTok and trivial-selector rows use the same LoRA and hard-drop path where
applicable. TC-ViTok Final uses k=144.

| Method | Tokens | VQAv2 | GQA | POPE | MMBench | TextVQA |
|---|---:|---:|---:|---:|---:|---:|
| LLaVA-1.5-7B published full | 576 | 78.5 | 61.0 | 84.4 | 64.3 | 58.2 |
| LoRA-576 reference | 576 | 76.38 | 70.55 | 86.46 | 61.7 | 38.4 |
| TC-ViTok v3 hard drop | 144 | 72.47 | 59.87 | 78.50 | n/a | n/a |
| TC-ViTok Adaptive | 36-288 | 69.43 | 65.04 | 75.63 | n/a | n/a |
| TC-ViTok Final | 144 | 72.27 | 67.12 | 79.09 | 62.0 | 25.8 |
| Final, merge disabled at eval | 144 | 72.27 | 67.12 | 79.09 | n/a | n/a |
| Final, zero CLIP query at eval | 144 | 72.67 | 67.45 | 79.37 | n/a | n/a |
| Random drop | 144 | 74.05 | 70.40 | 83.92 | 61.24 | 29.63 |
| Uniform grid | 144 | 74.65 | 69.26 | 82.78 | 61.93 | 29.61 |
| CLS-attention top-k | 144 | 74.91 | 68.16 | 78.94 | 62.05 | 36.57 |

Safe interpretation:

- LoRA-576 beats TC-ViTok Final by 4.11 pp on VQAv2, 3.43 pp on GQA, and
  7.37 pp on POPE.
- The strongest trivial selector beats or ties TC-ViTok on every evaluated
  benchmark.
- On MMBench, 62.0 vs 62.05 is statistically indistinguishable because MMBench
  dev is small, with SE about 2.5 pp.
- TextVQA is the clearest trivial-baseline failure: CLS-attention top-k scores
  36.57 vs TC-ViTok Final 25.8, a 10.77 pp gap.

### 3.2 Seed Replication for the Main Compression Gap

Both seeds use 5K-sample evaluations from the validation sets at the Phase-1
Final checkpoint.

| Seed | Method | VQAv2 | GQA | POPE |
|---:|---|---:|---:|---:|
| 42 | TC-ViTok Final | 72.27 | 67.12 | 79.09 |
| 42 | LoRA-576 | 76.38 | 70.55 | 86.46 |
| 7 | TC-ViTok Final | 72.34 | 67.58 | 78.87 |
| 7 | LoRA-576 | 76.65 | 71.36 | 86.78 |

Safe interpretation:

- The second seed verifies that the compression gap is not unique to seed 42.
- Do not claim statistical significance across random initializations. There are
  only two seeds.

### 3.3 FM2 Adaptive-Budget Replication

Table reports final effective budget `k_eff` and best observed validation
next-token-prediction (NTP) loss. Collapse is operationally defined as
`k_eff > 0.99 * k_max` for at least 2K consecutive steps. Here `k_max=288`.

| lambda_b | Seed 42: k_eff, Val | Seed 7: k_eff, Val | Status |
|---:|---:|---:|---|
| 0.03 | 287.97, 0.3898 | 288.00, 0.2466 | collapsed in both seeds |
| 0.08 | 280.15, 0.3368 | 281.83, 0.3285 | stable in both seeds |
| 0.10 | 271.00, 0.3177 | 270.37, 0.3391 | stable in both seeds |
| 0.15 | 247.23, 0.3176 | 245.71, 0.3078 | stable in both seeds |
| 0.20 | 227.85, 0.3291 | n/a | stronger budget pressure, single seed |

Important interpretation:

- The two-seed replicated non-collapsed region is lambda_b in [0.08, 0.15].
- lambda_b=0.20 is exploratory single-seed evidence only.
- The seed-7 lambda_b=0.03 row has the lowest NTP loss, but it is still
  collapsed because it uses the full k=288 token budget. Collapse is a loss of
  adaptivity and token savings, not necessarily a loss degradation.
- Val NTP losses are not direct selector-quality rankings across rows because
  token counts and continuation trajectories differ. Collapsed rows use the full
  token budget, and all adaptive rows are extra fine-tuning continuations.

### 3.4 FM3 Warm-Start Dependence

| Initialization | TPN version | LoRA rank/alpha | Best Val-VQA | Steps | Note |
|---|---|---:|---:|---:|---|
| Warm-start from Phase-1 Final | v1 | 64/128 | 0.3177 | 40K | best warm-start ceiling |
| From scratch | v1 | 64/128 | 0.4915 | 48K | oscillating |
| From scratch | v3 | 32/64 | 0.4380 to about 0.39 | 40K | regressing |
| From scratch image-only | v3 | 32/64 | 0.3334 | 50K | single seed, still descending |

Safe interpretation:

- Warm-start is necessary in the tested stack for the best training trajectory.
- FM3 is the weakest hypothesis because an effective supervisory augmentation,
  such as distillation or oracle importance, would falsify it.
- FM3 falsification protocol: run from-scratch joint training with additional
  token-level distillation or oracle importance supervision. If it reaches
  Val-VQA <= 0.33 within 80K steps, FM3 is falsified for that avenue.
- 80K is twice the 40K warm-start horizon used for the Phase-1 checkpoint.

### 3.5 STE and Tau-Floor Diagnostics

The method uses a tau-independent straight-through gradient route for
Gumbel-Sigmoid selection:

```text
mask_STE = mask_hard.detach()
         + alpha * (scores - scores.detach())
         + beta  * (mask_soft - mask_soft.detach())
```

Key results:

- Disabling the tau-independent correction worsens 50K validation NTP loss from
  0.3511 to 0.3846.
- At 12K steps, TPN gradient norm is 0.0324 with the correction vs 0.0030
  without it.
- LoRA gradient norms remain comparable: 0.4091 with correction vs 0.4277
  without correction.
- Raising the temperature floor to tau_min=0.5 without the correction gives TPN
  gradient norm 0.0029 and val NTP 0.4069 at 12K.
- Therefore the STE route provides gradient connectivity not recovered by this
  simple temperature-floor adjustment.
- The 0.4069 tau05 NTP value is a 12K diagnostic and is not directly comparable
  to the 50K no-STE loss of 0.3846. The matched evidence is the gradient norm.

## 4. Architecture and Design Choices

### 4.1 Components

The stack combines three encoders or token sources:

- CLIP ViT-L/14 visual tokens, `v_f`: 576 fine-grained patch tokens that are
  selected, merged, projected, and sent to LLaVA.
- DINOv2/CLIP coarse visual features, `v_c`: lower-resolution visual context
  consumed by the selector.
- CLIP text/question embedding, `q`: query signal consumed by the selector.

The TPN consumes `(q, v_c)` and scores the 576 fine CLIP-L tokens `v_f`. The
TPN does not receive `v_f` directly as its main input. The selected fine tokens
are passed through the LLaVA projector and then through LLaVA-1.5-7B with LoRA
adapters.

Current model code:

- Main model: `src/tc_vitok/models/tc_vitok_model.py`
- Token selector: `src/tc_vitok/models/tpn.py`
- Encoders: `src/tc_vitok/models/encoders.py`
- Training loop: `src/tc_vitok/training/trainer.py`
- Losses: `src/tc_vitok/training/losses.py`

### 4.2 Selector and Selection

The selector predicts per-token scores for the 576 candidate fine tokens. The
hard selection path uses top-k / masks for token dropping. During training, the
Gumbel-Sigmoid STE path provides a differentiable surrogate so the selector can
receive gradients.

Design choice:

- Independent binary token decisions with Gumbel-Sigmoid are easier to combine
  with adaptive budgets than a categorical top-k-only view.
- The tau-independent correction was added because low-temperature annealing
  starved TPN gradients.

### 4.3 Merge

The final version uses a parameter-free informative token merge rather than a
trainable bridge or pooler.

Why:

- Trainable bridges created a coupled gradient circuit between selector and
  LoRA adapter.
- Active trainable-pooler configurations exhibited oscillation.
- Removing the trainable bridge removed oscillation in the tested stack.
- The merge itself was not benchmark-improving, but it made the training
  dynamics analyzable and removed one source of co-adaptation.

### 4.4 Adaptive Budget

The BudgetPredictor is an adaptive head that tries to choose per-sample token
budgets. Its main pathology is collapse to the maximum-budget policy when the
budget penalty is too weak.

The goal of adaptive budgeting is to reduce average token cost below `k_max`.
If the head saturates at `k_eff = k_max`, it may still obtain a low NTP loss
because it effectively uses all tokens, but it eliminates the computational
savings that adaptive budgeting was meant to provide.

### 4.5 LoRA-576 Reference

LoRA-576 is a compute-matched no-compression reference. It uses the same LoRA
rank, training data, and fine-tuning setup as TC-ViTok but forwards all 576
visual tokens. This is the essential control because it separates compression
cost from ordinary LoRA fine-tuning effects.

## 5. Failure Modes

### FM1: Coupled Selector-Adapter Oscillation

Observation:

- Seven active trainable-pooler configurations showed non-monotonic,
  limit-cycle-like validation trajectories.
- A representative selector learning rate of 5e-4 produced a cycle period of
  about 5K steps.
- Reducing selector learning rate to 2e-5, matched to the LoRA learning rate,
  lengthened the cycle period to about 15K steps.
- A frozen bridge did not oscillate but selected fixed/noisy tokens that were
  unusable.

Mechanistic interpretation:

- The GAN analogy is only intuition.
- The narrower claim is that a learnable pre-LLM bridge closes a coupled
  two-timescale gradient circuit.
- Both selector and LoRA minimize the same VQA loss, but each changes the
  gradient field seen by the other.
- Alternative optimization-path explanations remain possible.

Remedy:

- Removing the trainable bridge and using a parameter-free merge removed
  oscillation in the tested stack.

### FM2: Adaptive-Budget Collapse

Observation:

- Low budget penalty at lambda_b=0.03 collapses to k_eff near 288 in both seeds.
- lambda_b in [0.08, 0.15] is stable in both seeds.
- lambda_b=0.20 is non-collapsed but high-pressure and single-seed only.

Key nuance:

- Collapse is defined by final k_eff saturation, not by validation loss.
- A collapsed run can still achieve low loss because it behaves like a
  fixed-full-budget model.
- That does not make it a successful adaptive budget, because it loses the token
  savings objective.

Remedy:

- Operate in the two-seed replicated non-collapsed region, currently
  lambda_b in [0.08, 0.15].
- Treat lambda_b=0.20 as exploratory single-seed upper-pressure evidence.

### FM3: Warm-Start Dependence

Observation:

- From-scratch joint training plateaus or regresses above the warm-started
  trajectory in the tested stack.
- The best warm-started run reaches 0.3177 Val-VQA at 40K.
- The best from-scratch image-only run reached about 0.3334 by 50K on one seed
  and was still descending, which is near the from-scratch plateau, not the
  warm-start ceiling.

What is claimed:

- FM3 is an observed training pathology and falsifiable hypothesis.
- It is not a validated mechanism with a completed mitigation.
- Distillation, oracle importance, and strong selector priors are open
  falsification paths.

What is not claimed:

- The paper does not prove that from-scratch training can never work.
- The paper does not prove distillation would fail.
- The paper does not solve FM3.

## 6. Probe and Diagnostic Findings

Track C was selected because causal visual importance was not recoverable from
the tested representations under the tested probe families.

Important local summary:

`runs/trackc_decision_summary.md`

Observed diagnostics:

- Query-conditioned top-k Jaccard was about 0.958, indicating near full overlap
  across questions and weak practical query dependence.
- Synthetic needle target rate was about 0.27, near chance level 0.25.
- Coverage extremes and hot/dead positions indicated degenerate position policy.
- CLIP/DINOv2 feature probes failed to recover single-token ablation importance
  under tested probes and sample sizes.
- Coordinate controls were positive, with R2 around 0.39 to 0.52, showing the
  probe pipeline can recover simple spatial signals.
- Negative R2 means worse-than-mean prediction, not a formatting error.
- The FastV-related attention smoke test was only N=100 and must remain labeled
  as a pilot.

Paper-level probe details:

- CLIP ViT-L/14 R2 values: about -0.04 and -0.33.
- DINOv2-large R2 values: about -0.08 and -0.30.
- Token-permuted CLIP null: about -0.05 and -0.31.
- Text-to-visual attention R2 around -0.02 in the pilot.
- Visual attention mass was about 89 percent in the pilot, showing the pipeline
  identifies expected visual attention concentration, but not causal importance.

Safe wording:

- "The tested probes and sample sizes do not recover token importance."
- Do not say "token importance is unrecoverable" in an absolute sense.

## 7. Experiment Inventory

This section lists the major runs and what each one established. The raw logs
are not reproduced line-by-line because they are large, but this gives the
result, purpose, and known artifact locations.

### 7.1 Phase 0 Infrastructure and Warmup

Purpose:

- Build and validate the real-data training pipeline.
- Train selector components before full joint training.

Key discoveries:

- In early gather-based Option B, integer gather blocked VQA gradients to TPN.
- Phase 0 needed direct sparsity/score losses to give TPN any gradient.
- Full validation during Phase 0 was too expensive and effectively hung the run.

Key fixes:

- Skip Phase 0 validation by setting a very large `eval_interval`.
- Use BF16 where possible.
- Use single GPU for Phase 0 because DDP overhead exceeded benefit.

### 7.2 Phase 1 Joint Training

Purpose:

- Jointly train selector path and LLaVA LoRA adapters.

Artifacts:

- Final paper refers to Phase-1 Final checkpoint after 80K steps.
- Local logs include `train & eval logs/Train_logs.txt`.

Takeaway:

- Joint training can produce a trainable model but does not learn a selector that
  beats trivial question-blind selectors.

### 7.3 TC-ViTok v3 Hard Drop

Purpose:

- Evaluate an earlier hard-drop variant.

Results:

- VQAv2 72.47
- GQA 59.87
- POPE 78.50

Artifact:

- `train & eval logs/eval_v3.log`

Takeaway:

- Hard drop was trainable but weak, especially on GQA.

### 7.4 TC-ViTok Adaptive

Purpose:

- Evaluate adaptive-budget continuation.

Results:

- VQAv2 69.43
- GQA 65.04
- POPE 75.63

Artifact:

- `train & eval logs/eval_adaptive.log`

Takeaway:

- The adaptive head can be trained in a stable regime, but the final benchmark
  result is not competitive with trivial selectors or LoRA-576.

### 7.5 TC-ViTok Final

Purpose:

- Main final selector result.

Results:

- VQAv2 72.27
- GQA 67.12
- POPE 79.09
- MMBench 62.0
- TextVQA 25.8

Artifact:

- `train & eval logs/eval_final.log`

Takeaway:

- Final trained query-conditioned selector does not surpass the strongest
  question-blind trivial selector on any of the five evaluated benchmarks.

### 7.6 Trivial Selector Baselines

Purpose:

- Test whether a learned query-conditioned selector beats cheap question-blind
  token selection at the same budget and LoRA path.

Selectors:

- Random drop.
- Uniform grid.
- CLS-attention top-k.

Results are in Section 3.1.

Takeaway:

- Trivial selectors are unexpectedly strong.
- The strongest trivial selector beats TC-ViTok Final on every benchmark or is
  statistically indistinguishable in the MMBench case.

### 7.7 LoRA-576 Reference

Purpose:

- Separate compression cost from LoRA fine-tuning effects.

Results:

- VQAv2 76.38
- GQA 70.55
- POPE 86.46
- MMBench 61.7
- TextVQA 38.4

Takeaway:

- Compression, not LoRA fine-tuning itself, explains the main TC-ViTok gap.

### 7.8 Seed-7 Replication

Purpose:

- Verify the compression gap and FM2 adaptive-budget behavior are not seed-42
  artifacts.

Main gap:

- Seed 7 TC-ViTok Final: 72.34 / 67.58 / 78.87.
- Seed 7 LoRA-576: 76.65 / 71.36 / 86.78.

FM2:

- lambda_b=0.03 collapses.
- lambda_b=0.08, 0.10, and 0.15 remain stable.

Takeaway:

- The key negative result and stable-window evidence replicate across two seeds.
- Do not overstate robustness beyond two seeds.

### 7.9 FM2 Seed-7 Runs

Remote logs:

- `runs/fm2_seed7/lb003.log`
- `runs/fm2_seed7/lb008.log`
- `runs/fm2_seed7/lb010.log`
- `runs/fm2_seed7/lb015.log`

Remote summary artifact:

- `runs/fm2_seed7/summary_lb003_lb008_lb010_lb015.json`

Seed-7 details from the final summary:

| Run | Final loss | Final vqa | Final budget | Final k_eff | Best loss | Best step | Last-5 loss | Last-5 k_eff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lb003 | 0.2468 | 0.3479 | 0.5000 | 288.0000 | 0.2466 | 34000 | 0.2471 | 288.0000 |
| lb008 | 0.2619 | 0.3386 | 0.4893 | 281.8250 | 0.2517 | 32000 | 0.2606 | 281.0595 |
| lb010 | 0.2704 | 0.3392 | 0.4694 | 270.3650 | 0.2702 | 36000 | 0.2716 | 268.4290 |
| lb015 | 0.3125 | 0.3626 | 0.4266 | 245.7125 | 0.3078 | 26000 | 0.3136 | 244.0000 |

### 7.10 Trainable-Pooler / FM1 Runs

Purpose:

- Test whether trainable bridges cause oscillatory co-adaptation.

Observed variants:

- Seven active trainable-pooler configurations oscillated.
- The frozen bridge did not oscillate but was unusable.
- Variants changed selector learning rate, cross-attention head count, auxiliary
  projection depth, learned-query constraints, and bridge activity.

Representative results:

- Selector LR 5e-4, LoRA LR 2e-5: about 5K period.
- Selector LR 2e-5, LoRA LR 2e-5: about 15K period.
- Frozen pooler: no oscillation, but non-informative selections.

Takeaway:

- Active learned bridges are consistent with FM1 in this stack.
- This is not a theorem and not cross-stack evidence.

### 7.11 STE Ablations

Purpose:

- Verify whether the tau-independent STE route is necessary.

Runs:

- `phase1_final_ste_logged`
- `phase1_final_no_ste_logged`
- `phase1_final_no_ste_tau05_logged`

Configs:

- `configs/phase1_final_ste_logged.yaml`
- `configs/phase1_final_no_ste_logged.yaml`
- `configs/phase1_final_no_ste_tau05_logged.yaml`

Script:

- `scripts/summarize_ste_ablation.py`

Takeaway:

- The additive STE route is supported by both final NTP loss and gradient-norm
  diagnostics.

### 7.12 k-Scaling Evaluation

Purpose:

- Test whether TC-ViTok beats trivial selectors at other budgets.

Budgets:

- k in {36, 72, 144, 216, 288}

Safe claim:

- On the Phase-1 Adaptive checkpoint, TC-ViTok does not beat the strongest
  trivial baseline at any measured k.

Figure:

- Current paper Figure 3.

### 7.13 Query Ablations

Purpose:

- Test whether the CLIP query affects final selection behavior.

Key result:

- Zeroing the CLIP query at evaluation improved TC-ViTok Final slightly on
  VQAv2/GQA/POPE: 72.67 / 67.45 / 79.37 vs 72.27 / 67.12 / 79.09.

Safe interpretation:

- The trained selector does not exploit the query at inference in a robust way.
- This does not prove the query is useless during optimization. Image-only and
  query-conditioned from-scratch training follow different trajectories.

### 7.14 Merge Ablations

Purpose:

- Test whether the informative merge itself drives benchmark gains.

Key result:

- Disabling merge at evaluation gives 72.27 / 67.12 / 79.09, matching TC-ViTok
  Final on VQAv2/GQA/POPE.

Safe interpretation:

- Merge is an architectural/training-stability intervention, not a benchmark
  improvement in the final model.

### 7.15 Qwen2-VL Portability Attempt

Purpose:

- Try an FM1 replication on Qwen2-VL-7B.

Outcome:

- Task accuracy stayed near chance.
- The run is a failed portability attempt.
- It is not validation for FM1 generality.

Code:

- `src/tc_vitok/qwenvl/`

Safe paper use:

- Mention only as a limitation or failed attempt.
- Do not put Qwen numbers in the main evidence path.

### 7.16 Synthetic and Needle Diagnostics

Purpose:

- Test whether token selection can recover a known target under simplified
  setups.

Code:

- `src/tc_vitok/synthetic/`

Finding:

- Synthetic needle target rate was about 0.27, near chance level 0.25.

Safe interpretation:

- Supports the story that the current selector fails to learn robust targeted
  importance from the available signals, but do not overgeneralize.

## 8. Remote Setup

Remote machine:

- User: `sk10741`
- Host: `ADUAED16811WKLX`
- Repo path: `~/TC-ViTok`
- Data path: `~/tc-vitok-data`
- OS: Ubuntu 22.04
- Python: 3.10
- GPUs: 2 x NVIDIA RTX 6000 Ada Generation
- VRAM: 49140 MiB per GPU

Observed driver/CUDA states:

- Older setup record: driver 550.127.05, CUDA 12.4.
- May 31 observed `nvidia-smi`: driver 570.207, CUDA 12.8.

Environment variables:

```bash
export DATA_ROOT=~/tc-vitok-data
export HF_HOME=${DATA_ROOT}/hf_cache
export TORCH_HOME=${DATA_ROOT}/torch_cache
```

Offline model loading was important during DDP:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

Useful remote checks:

```bash
cd ~/TC-ViTok
tmux ls
nvidia-smi
df -h .
```

Final observed GPU idle state after FM2 runs:

- GPU 0: about 15 MiB, 0 percent utilization.
- GPU 1: about 15 MiB, 0 percent utilization.

Disk warning:

- The root filesystem reached 100 percent usage during final runs.
- At one point `/dev/nvme1n1p2` had only 3.9 to 6.8 GB free out of 3.5 TB.
- Checkpoints can fill disk quickly.

Cleanup commands used during final FM2 sprint:

```bash
cd ~/TC-ViTok
rm -f runs/fm2_seed7/lb008/phase1/best.ckpt
rm -f runs/fm2_seed7/lb010/phase1/best.ckpt
find runs/fm2_seed7 -name "checkpoint_step_*.pt" -type f -delete
df -h .
```

Use cleanup carefully. Do not remove logs or summary JSON files. Do not remove a
checkpoint unless the result has been summarized and the paper no longer needs
that checkpoint for evaluation.

## 9. Long-Run Commands

### 9.1 DDP Launch Pattern

General multi-GPU launch:

```bash
cd ~/TC-ViTok
export DATA_ROOT=~/tc-vitok-data
bash scripts/launch_multigpu.sh \
  --config configs/phase1_final.yaml \
  --output runs/phase1_final \
  --no-wandb
```

### 9.2 Single-GPU tmux Launch Pattern

For long single-GPU continuations:

```bash
cd ~/TC-ViTok
tmux new -d -s fm2_lb015_s7 \
  'cd ~/TC-ViTok && CUDA_VISIBLE_DEVICES=0 DATA_ROOT=~/tc-vitok-data python3 scripts/train.py --config configs/phase1_adaptive_lb015_seed7.yaml --output runs/fm2_seed7/lb015 --seed 7 --no-wandb 2>&1 | tee runs/fm2_seed7/lb015.log'
```

Second GPU in parallel:

```bash
cd ~/TC-ViTok
tmux new -d -s fm2_lb003_s7 \
  'cd ~/TC-ViTok && CUDA_VISIBLE_DEVICES=1 DATA_ROOT=~/tc-vitok-data python3 scripts/train.py --config configs/phase1_adaptive_lb003_seed7.yaml --output runs/fm2_seed7/lb003 --seed 7 --no-wandb 2>&1 | tee runs/fm2_seed7/lb003.log'
```

Monitor:

```bash
tmux ls
nvidia-smi
tail -f runs/fm2_seed7/lb015.log
grep "val |" runs/fm2_seed7/lb015.log | tail -25
```

Attach/detach:

```bash
tmux attach -t fm2_lb015_s7
# detach with Ctrl-b then d
```

### 9.3 Config Generation Pattern for Lambda Sweep

Example:

```bash
cd ~/TC-ViTok
cp configs/phase1_adaptive.yaml configs/phase1_adaptive_lb003_seed7.yaml
python3 - <<'PY'
from pathlib import Path
p = Path("configs/phase1_adaptive_lb003_seed7.yaml")
s = p.read_text()
s = s.replace("lambda_budget: 0.1", "lambda_budget: 0.03")
s = s.replace('wandb_run_name: "phase1_adaptive_budget"', 'wandb_run_name: "fm2_seed7_lb003"')
p.write_text(s)
PY
mkdir -p runs/fm2_seed7/lb003
grep -n "lambda_budget\|load_from\|max_steps\|save_interval" configs/phase1_adaptive_lb003_seed7.yaml
```

### 9.4 Summarizing FM2 Logs

The remote summary parsed lines of this form:

```text
[Step 40000] val | loss: ... | vqa: ... | sparsity: ... | entropy: ... | budget: ... | k_effective_mean: ...
```

The summary output was:

`runs/fm2_seed7/summary_lb003_lb008_lb010_lb015.json`

If re-running the summary, preserve:

- number of validation rows
- final row
- best loss row
- best vqa row
- last-5 mean

### 9.5 Paper Compile Command

Local compile:

```bash
cd /Users/apple/Desktop/Junior_spring/TC-ViTok/ICONIP2026_G5_TCVitok
latexmk -pdf -interaction=nonstopmode main.tex
```

Known benign warnings:

- `amsmath` unable to redefine `\vec`.
- `caption` unknown class warning for LLNCS.
- Hyperref linebreak in title PDF string.
- Font shape substitutions.

Important:

- Verify page count after any edit.
- The target is exactly 15 pages or fewer.

## 10. Local Setup

Local workspace:

```bash
/Users/apple/Desktop/Junior_spring/TC-ViTok
```

Local shell:

```bash
zsh
```

Local paper tools:

- TinyTeX installed under `/Users/apple/Library/TinyTeX`.
- `latexmk` is available and has been used for the paper.

Local state:

- The git worktree may be dirty with paper build files and code edits.
- Do not revert unrelated user changes.
- Generated LaTeX files such as `.aux`, `.bbl`, `.blg`, `.log`, `.out`, `.fdb_latexmk`, and `.fls` may appear after compile.

## 11. Codebase Map

High-level files:

- `README.md`: older project overview.
- `remote.md`: detailed historical remote setup and bug log.
- `everything.md`: older broad summary, now partially outdated.
- `PROJECT_FULL_CONTEXT.md`: this current single source of truth.
- `ICONIP2026_G5_TCVitok/main.tex`: current paper.
- `ICONIP2026_G5_TCVitok/main.bib`: references.

Core model and training:

- `src/tc_vitok/models/tc_vitok_model.py`
- `src/tc_vitok/models/tpn.py`
- `src/tc_vitok/models/encoders.py`
- `src/tc_vitok/training/trainer.py`
- `src/tc_vitok/training/losses.py`
- `src/tc_vitok/training/optimizer.py`
- `src/tc_vitok/training/curriculum.py`

Entry points:

- `scripts/train.py`
- `scripts/evaluate.py`
- `scripts/launch_multigpu.sh`

Diagnostics and analysis scripts:

- `scripts/aggregate_multiseed.py`
- `scripts/analyze_t1.py`
- `scripts/analyze_t4_coverage.py`
- `scripts/analyze_t8_merge_cka.py`
- `scripts/analyze_tnew1.py`
- `scripts/build_fastv_oracle.py`
- `scripts/check_trackc_artifacts.py`
- `scripts/compute_causal_importance.py`
- `scripts/decision_gate.py`
- `scripts/extract_alt_encoder_features.py`
- `scripts/extract_attention_features.py`
- `scripts/extract_mid_llm_features.py`
- `scripts/generate_trackc_artifacts.py`
- `scripts/measure_compute.py`
- `scripts/probe_attention.py`
- `scripts/probe_by_question_type.py`
- `scripts/probe_per_layer.py`
- `scripts/probe_single_layer.py`
- `scripts/probe_target_variants.py`
- `scripts/probe_with_bootstrap.py`
- `scripts/run_t2_query_diversity.py`
- `scripts/run_trackc_controls.py`
- `scripts/summarize_ste_ablation.py`
- `scripts/train_causal_probe.py`

Special modules:

- `src/tc_vitok/qwenvl/`: Qwen2-VL portability attempt.
- `src/tc_vitok/synthetic/`: synthetic and needle diagnostics.

Logs:

- `train & eval logs/Train_logs.txt`
- `train & eval logs/eval_final.log`
- `train & eval logs/eval_v3.log`
- `train & eval logs/eval_adaptive.log`

## 12. Infrastructure Failures and Fixes

This section condenses the long bug history from `remote.md`.

### 12.1 Model and Library Compatibility

Problems:

- Newer `transformers` changed LLaVA internals.
- `LlavaForConditionalGeneration` paths differed between versions.
- DINOv2 image size default did not match the required 98x98 coarse path.

Fixes:

- Added fallbacks for `language_model` vs nested `model.language_model`.
- Stored direct references to the LLaVA language model and multimodal projector.
- Chose PEFT task type based on actual LLaVA internal object.
- Set DINOv2 `img_size=98`.

### 12.2 DDP and GPU Issues

Problems:

- DataLoader worker crashes from corrupt or missing JPEGs caused SIGABRT.
- Zombie processes held GPU memory after crashes.
- `device_map="cpu"` installed Accelerate hooks that prevented normal `.to(cuda)`.
- DDP forward was bypassed by calling raw model methods, so ranks diverged.
- Rank 0 validation/checkpointing desynchronized ranks.
- Phase 0 DDP was slower than single GPU.

Fixes:

- Added dataset retry loops.
- Added GPU process cleanup to launch script.
- Removed `device_map="cpu"`.
- Ensured DDP training calls go through the DDP wrapper.
- Added rank barriers after validation/checkpoint.
- Ran Phase 0 single-GPU.

### 12.3 HuggingFace and NCCL

Problems:

- HuggingFace HEAD requests desynchronized DDP ranks.
- NCCL timed out during model load and barriers.

Fixes:

- Set `HF_HUB_OFFLINE=1`.
- Set `TRANSFORMERS_OFFLINE=1`.
- Increased NCCL process group timeout to 30 minutes.
- Removed unnecessary barriers before model transfer.

### 12.4 Phase 0 Training Logic

Problem:

- In gather-based selection, integer indices blocked VQA gradient to TPN.
- With zero sparsity losses, Phase 0 gave no useful TPN gradient.

Fix:

- Added non-zero direct losses such as sparsity or L1-style terms for TPN.
- Disabled expensive Phase 0 validation.

## 13. Paper Revision and Reviewer-Defense History

Major issues that were fixed:

- Abstract originally overclaimed that TC-ViTok lost to all trivial selectors on
  all benchmarks. Corrected to: it fails to surpass the strongest trivial
  baseline on any benchmark.
- Section 5.2 originally misstated the GQA/POPE CLS comparisons. Corrected.
- MME was listed in setup without results. Removed from setup list.
- MMBench 0.05 pp gap to CLS was reframed as statistically indistinguishable.
- `NTP` is now defined on first use.
- FM1 no longer overclaims a theorem or GAN-like proof.
- FM2 replication now includes seed 7 at lambda_b in {0.03, 0.08, 0.10, 0.15}.
- FM3 is explicitly an untested falsification path, not a solved mitigation.
- Qwen2-VL was demoted to a failed portability attempt.
- The title changed from "Fail" to "Can Fail".
- The introduction now explains why the negative result is not obvious.
- The conclusion now gives a practical takeaway for similar stacks.
- An architecture diagram was added to the paper.

TA feedback addressed or to keep watching:

- Contributions are itemized.
- Related work was expanded compared with earlier drafts, but it remains compact
  because of the 15-page limit.
- Figure formatting should always be checked after LaTeX edits.
- Key equations should be displayed, not buried in paragraphs.
- Failure modes are abbreviated as FM after first use.
- Avoid em dashes in final prose where possible.
- Remedies are framed after observed signatures and results, with care that they
  do not imply all failures are solved.

## 14. Open Risks

These are the remaining honest weaknesses.

### Single-Stack Evidence

All main failure-mode evidence comes from the LLaVA-1.5-7B plus CLIP plus TPN
stack. This is the largest structural limitation.

Defense:

- The title says "Can Fail".
- The paper is a diagnostic case study.
- The abstract, contributions, limitations, and conclusion scope the evidence.
- Reproduction protocols are a contribution.

### FM3 Distillation Not Tested

Defense:

- The paper says this explicitly.
- The falsification criterion is concrete.
- Underpowered distillation would create more attack surface than it resolves.

### FM1 Replication

FM1 is supported by multiple variants in one stack, not by independent
architectures.

Defense:

- It is phrased as a prediction/hypothesis, not a theorem.
- The trainable-bridge removal result is stack-specific.

### Query Conditioning Looks Weak

Zeroing the CLIP query at evaluation improves the final model slightly.

Defense:

- This is part of the negative result.
- It shows end-task supervision does not reliably exploit question information.
- It does not prove the query has no role during optimization.

### Compute/FLOP Quantification

The motivation is pre-LLM token reduction, but the main paper is primarily
diagnostic rather than a compute benchmark.

Potential future addition:

- A small table mapping token count to visual attention cost would help:
  576 tokens = 1.0x visual-token attention baseline, 144 tokens = 1/16 of
  visual-token self-attention in a simple quadratic token-only view. Actual LLM
  prefill savings depend on text length and implementation.

## 15. What Not To Claim

Do not claim:

- TC-ViTok is a competitive compression method.
- The failure modes are universal across all VLMs.
- The learned selector loses to every trivial selector on every benchmark.
- FM3 is solved.
- Distillation was tested.
- Qwen2-VL validates cross-architecture generality.
- Token importance is impossible to recover.
- The MMBench 0.05 pp gap is meaningful.
- The seed-7 lambda_b=0.03 row proves collapse is beneficial.
- The low NTP loss in FM2 directly ranks selector quality across lambda rows.

Safe claims:

- In the tested stack, the learned selector fails to surpass the strongest
  question-blind trivial baseline on all measured benchmarks.
- The LoRA-576 matched control shows that compression, not LoRA fine-tuning
  alone, explains the performance gap.
- The STE route improves TPN gradient connectivity and task loss in this setup.
- FM2 has a two-seed replicated stable region at lambda_b in [0.08, 0.15] and
  replicated collapse at lambda_b=0.03.
- FM3 is an observed pathology and falsifiable hypothesis.

## 16. Likely Reviewer Questions and Best Answers

Question: Why is this negative result not obvious?

Answer: Query-conditioned compression can succeed with stronger priors or
supervision, and a query-conditioned selector should plausibly beat
question-blind random/grid/CLS heuristics at the same budget. The controlled
finding is that end-task supervision alone does not realize that advantage in
this stack.

Question: Why call lambda_b=0.03 collapse if seed 7 gets the best Val loss?

Answer: Collapse is operationally defined by budget saturation, not loss. The
run uses k=288 tokens and behaves as a fixed-full-budget model, eliminating the
adaptive token savings goal. Val losses are not direct selector-quality rankings
across rows because token counts and continuation trajectories differ.

Question: Why not run distillation?

Answer: A clean distillation experiment would need a full from-scratch
distillation condition, adequate horizon, and preferably multiple seeds. A weak
or short run would not resolve FM3 and would introduce new attack surface. The
paper instead provides a precise falsification protocol.

Question: Why should reviewers care about a single-stack result?

Answer: The result is framed as a diagnostic case study, not a universal law. It
provides a matched control, trivial baselines, replicated FM2 behavior, and
reproduction protocols for independent stacks.

Question: Does zero-query improvement mean the query encoder is useless?

Answer: It means the final trained selector does not robustly exploit query
information at inference. It does not prove the query has no optimization role,
because image-only and query-conditioned from-scratch training follow different
trajectories.

Question: Is the learned selector worse than CLS-attention everywhere?

Answer: No. It slightly beats CLS on POPE but loses to random and grid there. It
is statistically indistinguishable from CLS on MMBench. The safe claim is that
it does not surpass the strongest trivial baseline on any benchmark.

## 17. Recommended Next Steps if Work Continues

Highest-value scientific follow-ups:

1. Run a clean FM3 distillation falsification experiment:
   - from-scratch joint training
   - auxiliary token-level importance supervision from LoRA-576 single-token
     ablation importance or CLS-attention scores
   - 80K steps
   - at least two seeds
   - success criterion: Val-VQA <= 0.33

2. Replicate the full study on another VLM stack:
   - Qwen2-VL attempt must first reach non-trivial accuracy
   - only then can FM1/FM2/FM3 be tested cross-architecture

3. Add a small compute table:
   - tokens vs relative visual prefill cost
   - clarify that practical savings depend on text length and implementation

4. Improve qualitative visualization:
   - show token maps for TC-ViTok, random/grid/CLS, and query variants
   - avoid overinterpreting heatmaps unless tied to quantitative metrics

5. Expand probe datasets:
   - larger attention pilot beyond N=100
   - richer probes and causal targets
   - bootstrap confidence intervals where possible

## 18. Minimal Context for a Future LLM

If a future LLM only has time to read one section, read this:

- TC-ViTok is a diagnostic negative-result paper, not a method paper.
- The learned query-conditioned selector compresses 576 tokens to k=144 before
  LLaVA-1.5-7B LoRA training.
- It loses to a compute-matched LoRA-576 reference and does not surpass the
  strongest trivial selector on any of five benchmarks.
- The three FMs are:
  - FM1: trainable bridge creates oscillatory coupled selector-adapter dynamics.
  - FM2: adaptive budget can collapse to full-budget policy when penalty is too
    weak; stable region [0.08, 0.15] is replicated in two seeds.
  - FM3: from-scratch joint training depends on warm-start; distillation is an
    untested falsification path.
- Single-stack scope is the main limitation.
- Never use Qwen2-VL as validation.
- Never say all trivial selectors beat TC-ViTok on all benchmarks.
- The final paper lives at `ICONIP2026_G5_TCVitok/main.tex`.

