# Codex Task — FM1 Reproduction on Qwen-VL (Cross-Architecture Validation)

**Purpose:** Reproduce the FM1 co-adaptation-oscillation finding on a
second VLM stack (Qwen2-VL-7B + LoRA + trainable pooler bridge). A
positive reproduction would close the single-architecture rejection
vector (R1) in the ICONIP 2026 submission and push acceptance
probability from ~88% to ~92–95%.

**Compute budget:** ~24–36 GPU-hours total on 2× RTX 6000 Ada or
equivalent.

**Why this experiment, specifically:**
The text-only acceptance-hardening edits already in `main.tex` predict
FM1 will appear in any architecture with a trainable parameter bridge
between a learnable pre-LLM selector and adapter-trained LLM weights.
The single highest-impact empirical strengthening of the paper is to
*show* that prediction holding on a second architecture. We deliberately
pick FM1 (oscillation) over FM2 (collapse) or FM3 (warm-start) because:
(a) FM1 is the cheapest to demonstrate — a single training run with a
trainable pooler exhibits the cycle within 10K steps; (b) FM1's
mechanism (GAN-style two-player gradient circuit) is the strongest
theoretical claim in the paper; (c) the existing TC-ViTok codebase
already has a trainable-pooler variant we can port.

---

## High-level plan

1. **Setup (4 GPU-h):** Build a Qwen2-VL-7B + LoRA training rig with a
   trainable pooler module sitting between the frozen Qwen2-VL vision
   encoder and the Qwen2 LLM backbone.
2. **Two runs at two selector learning rates (16–24 GPU-h):**
   - **Run A** (high LR): trainable-pooler $\etasel = 5 \times 10^{-4}$,
     LLM-LoRA standard LR.
   - **Run B** (low LR): trainable-pooler $\etasel = 2 \times 10^{-5}$
     ($25\times$ lower), LLM-LoRA standard LR.
   - Train each to 12K steps minimum (Run A is expected to show
     oscillation within ~5K; Run B within ~15K).
3. **Diagnostics (1 GPU-h):** Run validation-VQA evaluation every 500
   training steps to produce a high-resolution time-series.
4. **Paper integration (text-only, ~1 hr human time):** Add a single
   "FM1 cross-architecture confirmation" paragraph and one
   sub-panel to Fig. 1 (or a new Fig. 1b).

---

## CODEX TASK: Qwen-VL trainable-pooler FM1 reproduction

```
CODEX TASK: FM1 oscillation reproduction on Qwen2-VL-7B
PRIORITY: P0 — only experimental work that pushes ICONIP acceptance
above ~88%.

CONTEXT
The TC-ViTok ICONIP submission predicts (Section 4.1, Prediction
statement) that any architecture with a trainable parameter bridge
between a learnable pre-LLM selector and adapter-trained LLM weights
will exhibit limit-cycle oscillation in joint training. The paper has
this prediction stated formally but only one architecture's worth of
evidence (CLIP ViT-L/14 + LLaVA-1.5-7B). The single most valuable
follow-up experiment is to reproduce FM1 on a different stack and
add a confirmatory figure.

DEPENDENCIES
- ~/TC-ViTok/ existing codebase (selector, merge, training rig)
- huggingface transformers >= 4.45 (Qwen2-VL is in transformers)
- 2x RTX 6000 Ada or equivalent GPUs
- LLaVA dataset already on disk at ~/tc-vitok-data
- VQAv2 + GQA training data already configured

REPO CHANGES — create a new subpackage:
  src/tc_vitok/qwenvl/
    __init__.py
    qwen_backbone.py       # Qwen2-VL-7B loader, frozen vision encoder
    qwen_pooler.py         # The TRAINABLE bridge — what we want to
                           # show oscillates
    qwen_lora.py           # LoRA adapters on Qwen2's LLM
    qwen_train.py          # Joint training loop with same loss + Gumbel STE
                           # but explicitly NO parameter-free merge —
                           # the trainable bridge is the whole point
    qwen_eval.py           # Val-VQA accuracy on GQA val_balanced
                           # every 500 steps
    configs/
      qwenvl_run_a.yaml    # eta_selector = 5e-4
      qwenvl_run_b.yaml    # eta_selector = 2e-5

DETAILED MODULE SPECIFICATIONS

==== qwen_backbone.py ====
- Load Qwen/Qwen2-VL-7B-Instruct from HF Hub.
- Identify and FREEZE: vision encoder (Qwen2VisionTransformer), all
  base LLM weights, the multi-modal projector.
- Expose:
    encode_image(pixel_values) -> [B, N_visual, D_v]
    encode_text(input_ids) -> question embedding (CLS-pooled or
                              last-token-pooled embedding from a
                              frozen pass)
    llm_forward(visual_tokens, text_input_ids, lora_state) ->
                              logits, hidden states

==== qwen_pooler.py ====
This is the TRAINABLE bridge. Architecture:
- Cross-attention with 4 heads, d_model = 384
- Input:
    Q = question_embedding   [B, 1, 512] -> linear to [B, 1, 384]
    K, V = coarse_visual    [B, M, D_v_qwen] (sub-sampled visual tokens)
- Output:
    scores       [B, N]   (per-position importance)
    summary_ctx  [B, D_v] (the "pooled" representation that the LLM
                          will receive in addition to selected tokens)
- INCLUDE the same Gumbel-Sigmoid STE with tau-independent additive
  correction as Eq.~1 in main.tex.
- Top-k selection at k = 144 visual tokens.
- Forward pass:
    selected_tokens = top_k_select_with_STE(visual, scores, k=144)
    pooled = summary_ctx (this is the trainable bridge edge)
    llm_input = [selected_tokens; pooled.unsqueeze(1)]  # length k+1
- Trainable parameters: cross-attention QKV projections + score MLP =
  roughly 1.0–1.2M params.

==== qwen_lora.py ====
- LoRA on Qwen2 LLM: rank 32, alpha 64, targets {q_proj, k_proj,
  v_proj, o_proj} on every layer.
- Trainable: only LoRA matrices.
- Standard implementation; use peft library if convenient.

==== qwen_train.py ====
- DataLoader: VQAv2 + GQA, batch_size 64 (effective batch 128 via
  gradient accumulation if memory tight).
- Loss: standard NTP cross-entropy on Qwen2-VL chat template
  formatted (question, answer) pairs.
- Optimizer: two parameter groups:
    Group "selector"  -> qwen_pooler.parameters(),
                          lr = eta_selector (from config),
                          weight_decay = 0.01
    Group "lora"      -> all LoRA params,
                          lr = 2e-5,
                          weight_decay = 0.0
- Mixed precision: bf16.
- Training horizon: 12K steps (sufficient to see at least one full
  oscillation cycle even at eta_selector=2e-5).
- Logging:
    Every 500 steps, run qwen_eval.py on GQA val_balanced (full eval,
    not subset) and log val-VQA cross-entropy AND VQA-Score accuracy.
    Write to a CSV: step, val_loss, val_acc.
- Checkpoint every 2K steps.

==== qwen_eval.py ====
- Greedy decoding, max_new_tokens=20, batched eval.
- Output: scalar val_loss (NTP CE) and val_acc (exact-match VQA-Score).

==== configs/qwenvl_run_a.yaml ====
eta_selector: 5.0e-4
eta_lora: 2.0e-5
batch_size: 64
grad_accum: 2
steps: 12000
log_every: 500
save_every: 2000
seed: 42

==== configs/qwenvl_run_b.yaml ====
eta_selector: 2.0e-5      # 25x lower
eta_lora: 2.0e-5
batch_size: 64
grad_accum: 2
steps: 12000
log_every: 500
save_every: 2000
seed: 42

OUTPUT FILES (after both runs complete)
  runs/qwenvl_fm1_run_a/
    log.csv                 # step, val_loss, val_acc
    summary.json            # {oscillation_period_estimate: ..., ...}
    best.ckpt
  runs/qwenvl_fm1_run_b/
    log.csv
    summary.json
    best.ckpt

ANALYSIS SCRIPT
Create scripts/analyze_qwenvl_fm1.py:
- Load both CSVs.
- Compute rolling-mean detrended val_acc series.
- Estimate oscillation period from autocorrelation peaks
  (statsmodels.tsa.stattools.acf, find first non-trivial peak).
- Print:
    Run A (eta=5e-4):  period = X steps,  peak-to-trough amplitude = Y
    Run B (eta=2e-5):  period = X' steps, peak-to-trough amplitude = Y'
    Period ratio (B/A) = ...     [FM1 predicts ratio > 1]
    Both cycles consistent with FM1? [Y/N]
- Generate a 2-panel figure (matplotlib, saved as
  figures/oscillation_qwenvl.pdf):
    Top panel: Run A trajectory + period annotation
    Bottom panel: Run B trajectory + period annotation
- This figure will be added to the paper as a new sub-panel.

VALIDATION CHECKS
- Assert each run reached >= 10K steps without crashing.
- Assert log.csv has >= 20 datapoints (so we have >= 10K steps / 500).
- Assert val_loss is finite throughout (no NaN/inf).
- Assert at least one full oscillation cycle is detected in Run A.

RUN COMMAND (after implementation):
DATA_ROOT=~/tc-vitok-data python -m tc_vitok.qwenvl.qwen_train \
    --config src/tc_vitok/qwenvl/configs/qwenvl_run_a.yaml \
    --output runs/qwenvl_fm1_run_a/ \
    2>&1 | tee runs/qwenvl_fm1_run_a/train.log

# Then once Run A is done (or in parallel on a different GPU):
DATA_ROOT=~/tc-vitok-data python -m tc_vitok.qwenvl.qwen_train \
    --config src/tc_vitok/qwenvl/configs/qwenvl_run_b.yaml \
    --output runs/qwenvl_fm1_run_b/ \
    2>&1 | tee runs/qwenvl_fm1_run_b/train.log

# Then:
python scripts/analyze_qwenvl_fm1.py \
    --run-a runs/qwenvl_fm1_run_a/log.csv \
    --run-b runs/qwenvl_fm1_run_b/log.csv \
    --output figures/oscillation_qwenvl.pdf

EXPECTED RUNTIMES
- Run A: ~10 GPU-h on 2x RTX 6000 Ada (Qwen2-VL-7B inference is
  comparable to LLaVA-1.5-7B, slight overhead from longer context).
- Run B: ~10 GPU-h.
- If run sequentially: ~24 hours total wall-clock.
- If run in parallel on different GPUs: ~12 hours wall-clock.

EXPECTED OUTCOMES (most likely)
- Run A (eta=5e-4): val_acc oscillates with period ~5K steps,
  matching LLaVA observation.
- Run B (eta=2e-5): val_acc oscillates with period ~15K steps,
  matching LLaVA observation.
- Period ratio B/A is consistent with the 25x learning-rate drop's
  monotonic prediction (FM1 confirmed on Qwen2-VL).

FALLBACK OUTCOMES
- If Run A does NOT oscillate within 12K steps:
  This would falsify FM1's strongest formulation. The paper would
  need to revise the FM1 prediction to scope it to LLaVA-class
  architectures. This is publishable as a refinement of the
  taxonomy but reduces the acceptance lift.
- If both runs oscillate but with the WRONG period ordering (Run B
  faster than Run A): suggests FM1's period-inverse-LR claim is
  more complex than reported. Discuss honestly.
- If training is unstable (NaN early in either run): debug
  initialisation/LR schedule before drawing conclusions.

PAPER INTEGRATION (text-only, AFTER results return):

EDIT 1: Add a sub-figure
- The current Fig. 1 in main.tex (oscillation.pdf) is the LLaVA
  observation. Either:
  (a) Replace with a 2-panel figure combining LLaVA (top) and
      Qwen2-VL (bottom), or
  (b) Add a new Figure 1b after Fig. 1 showing only Qwen2-VL.
  Recommend (b) to preserve the LLaVA figure intact.

  New figure caption template:
  "FM1 cross-architecture confirmation on Qwen2-VL-7B with a
  trainable pooler. \emph{Top:} eta_selector=5e-4, period ~ [X] K
  steps. \emph{Bottom:} eta_selector=2e-5, period ~ [Y] K steps.
  The same monotone relationship between selector learning rate
  and oscillation period observed in LLaVA-1.5-7B (Fig. 1)
  reproduces here. We do not run the frozen-pooler control on
  Qwen2-VL."

EDIT 2: Insert one paragraph at the end of Section 4.1 (FM1),
immediately after the existing "Remedy" paragraph. Template:

"\paragraph{Cross-architecture confirmation.}
We reproduce the FM1 prediction on Qwen2-VL-7B + LoRA with a
trainable pooler bridge. With selector learning rate
eta_selector = 5x10^-4, we observe oscillation with period
approximately [X] K steps; at eta_selector = 2x10^-5, the period
extends to approximately [Y] K steps (Fig.\ [new-fig-ref]).
The qualitative pattern matches our LLaVA-1.5-7B observation:
the same gradient-circuit mechanism produces limit cycles in a
different (encoder, LLM, selector) triple, and the cycle period
grows monotonically as selector learning rate decreases. This is
consistent with the prediction in this section and supports the
portability of FM1 beyond the LLaVA-CLIP-ViT-L/14 stack on which
the taxonomy was originally derived."

EDIT 3: Update §4.4 "Testing the Taxonomy" — the FM1 reproduction
protocol paragraph. Add at the end:
"We executed this protocol on Qwen2-VL-7B (see §4.1) and observed
the predicted period relationship; the FM3 protocol on Qwen2-VL is
left for future work."

EDIT 4: Update the abstract — add ONE sentence to convey that the
taxonomy now has cross-architecture confirmation.
Insert after "...a three-stage curriculum on a converged base":

"FM1 reproduces qualitatively on Qwen2-VL-7B + LoRA with a trainable
pooler, supporting cross-architecture portability of the
oscillation finding."

POST-EDIT VERIFICATION
- pdflatex main.tex; bibtex main; pdflatex x2
- grep -c "FILL\|VALUE\|INSERT" main.tex   # must be 0
- New figure renders correctly at print resolution
- Page count remains <= 15

TIME ESTIMATE
Implementation work for Codex: ~4 hours (Qwen2-VL boilerplate +
training loop adaptation).
GPU runtime: ~12-24 hours.
Paper integration after results: ~1 hour.
Total: ~36 hours from start to integrated paper.
```

---

## Optional second experiment (if compute allows): FM2 on Qwen-VL

If both Qwen-VL FM1 runs succeed and there's GPU time remaining
before submission, the second-best experiment is **FM2 (BudgetPredictor
collapse) on Qwen-VL** — would close even more of R1.

The setup is the same as above (Qwen-VL backbone) but with the
adaptive-budget head. Two $\lbb$ values: 0.03 (predicted collapse) and
0.10 (predicted stable). ~10 GPU-h total. Would add a second
cross-architecture confirmation paragraph at the end of Section 4.2.

If both FM1 and FM2 reproduce on Qwen-VL, the taxonomy's portability
becomes empirically demonstrated (not just predicted), and the
single-architecture rejection vector R1 essentially disappears. Acceptance
probability ceiling moves to ~93–95%.

---

## Fallback if Qwen-VL setup is too painful

If Qwen-VL's HuggingFace integration is fragile (LoRA + custom
trainable bridge often hits compatibility issues), a viable alternative
is **LLaVA-Next-7B** (Vicuna-1.5-7B with a different visual encoder,
SigLIP). This shares the LLaVA tokenizer and most of the existing
TC-ViTok pipeline but uses a different vision encoder. A successful
FM1 reproduction on LLaVA-Next would still close R1 partially — though
less compellingly than a true second-family confirmation like Qwen-VL.

Pick this fallback only if Codex hits >2 hours of integration issues
with Qwen-VL.

---

## Submission timing

Given a 5-day deadline:

| Day | Activity |
|---|---|
| Day 1 | Codex implements Qwen-VL FM1 module (~4 hours). Run A starts on GPU 0. |
| Day 2 | Run A completes ~hour 12. Run B starts. Codex implements analysis script. |
| Day 3 | Run B completes ~hour 24. Analysis script runs. Figure generated. |
| Day 4 | Paper integration (Edits 1–4 above). Recompile, verify, eyeball. |
| Day 5 | Final review, submit. |

If Qwen-VL setup hits trouble, drop to LLaVA-Next fallback on Day 2 and
slide everything by 1 day.
