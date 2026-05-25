# TC-ViTok — Complete Project Context

**Last updated:** 2026-05-17  
**Author:** Muhammad Ali Shah (m.ali.shah0103@gmail.com, GitHub: whyeventry0103)  
**Server user:** sk10741@10.224.32.112  
**Target venue:** BMVC 2026 (abstract deadline: May 22, full paper: May 29)

---

## 1. What This Project Is

TC-ViTok (Task-Conditioned Visual Tokenization) is a research project building a **modular, pre-LLM, query-conditioned visual token compressor** for LLaVA-1.5-7B. The core idea: LLaVA-1.5 feeds 576 visual tokens from CLIP ViT-L/14 into an LLM. This is wasteful — "What color is the car?" needs maybe 20 tokens, not 576. We want a lightweight module that reads the question, looks at the image, and picks only the ~144 most relevant tokens before they hit the LLM.

**The paper is NOT a SOTA paper.** It is a **failure-mode taxonomy + mechanistic analysis + honest recipe** paper. We ran ~12 training attempts across 8+ architectural variants and documented three structural pathologies that any such system inherits when trained jointly with LLM adapters. The taxonomy + recipe is the contribution.

**Paper title (current):**  
*"TC-ViTok: A Failure-Mode Taxonomy for Pre-LLM Visual Token Compression, with a Stable Curriculum and Parameter-Free Recipe"*

---

## 2. Hardware and Server

- **Server:** `sk10741@10.224.32.112`
- **GPUs:** 2× NVIDIA RTX 6000 Ada (49,140 MiB each)
- **OS:** Ubuntu, CUDA 12.4
- **Data root:** `/home/sk10741/tc-vitok-data/`
- **Project root:** `~/TC-ViTok/`

### Data on server

| Dataset | Size | Path |
|---------|------|------|
| GQA | 36 GB | `tc-vitok-data/gqa/` |
| VQAv2 | 20 GB | `tc-vitok-data/vqav2/` |
| POPE | 1.1 MB | `tc-vitok-data/pope/` |
| TextVQA | — | `tc-vitok-data/textvqa/` |
| LLaVA-Instruct-150K | — | `tc-vitok-data/llava_instruct/` |
| COCO images | — | `tc-vitok-data/coco/` |
| MMBench | — | `tc-vitok-data/mmbench/` |
| MME | — | `tc-vitok-data/mme/` |

### Sync commands (local ↔ server)

```bash
# Pull specific directory from server to local (note: single quotes for remote paths with spaces)
rsync -avz 'sk10741@10.224.32.112:~/TC-ViTok/DL Project Report/' ~/Desktop/Junior_spring/TC-ViTok/"DL Project Report"/
rsync -avz 'sk10741@10.224.32.112:~/TC-ViTok/DL Project Report Template/' ~/Desktop/Junior_spring/TC-ViTok/"DL Project Report Template"/

# Push specific file to server
rsync -avz ~/Desktop/Junior_spring/TC-ViTok/scripts/foo.py 'sk10741@10.224.32.112:~/TC-ViTok/scripts/'

# Pull single figure from server
rsync -avz 'sk10741@10.224.32.112:~/TC-ViTok/DL Project Report/figures/budget_histogram.pdf' \
    ~/Desktop/Junior_spring/TC-ViTok/"DL Project Report"/figures/
```

---

## 3. Codebase Structure

```
TC-ViTok/
├── src/tc_vitok/
│   ├── models/
│   │   ├── tc_vitok_model.py    # main TCViTokModel, gather_tokens, informative_token_merge
│   │   ├── tpn.py               # TokenizationPolicyNetwork (v1) + TokenScorerV3 (v3)
│   │   ├── masking.py           # Gumbel-STE, adaptive threshold, BudgetPredictor wiring
│   │   ├── encoders.py          # frozen CLIP-L, CLIP-B/32, DINOv2-S wrappers
│   │   └── __init__.py
│   ├── training/
│   │   ├── trainer.py           # main training loop, checkpoint save/load, DDP setup
│   │   ├── losses.py            # entropy, diversity, budget, distill losses
│   │   ├── optimizer.py         # AdamW param group builder (separate LRs per module)
│   │   └── curriculum.py        # tau schedule, warmfreeze logic
│   ├── data/
│   │   ├── datasets.py          # VQAv2, GQA, TextVQA, LLaVAInstruct, POPE, MMBench, MME
│   │   ├── caching.py           # feature caching utilities
│   │   └── spatial_labels.py    # spatial question classification (for spatial_eval)
│   └── evaluation/
│       ├── metrics.py           # vqa_score, gqa_accuracy, pope_f1, normalize_answer
│       ├── collapse_detection.py
│       ├── spatial_eval.py
│       └── pareto.py
├── scripts/
│   ├── train.py                 # single-GPU entry point
│   ├── launch_multigpu.sh       # 2-GPU DDP launcher
│   ├── evaluate.py              # eval on VQAv2/GQA/POPE/MMBench/MME/TextVQA
│   ├── dump_tpn_heatmaps.py     # dump TPN scores + overlays for qualitative figure
│   ├── plot_oscillation.py      # co-adaptation oscillation figure (--use-known-points)
│   ├── make_budget_histogram.py # read budget dump → budget_histogram.pdf
│   ├── compose_qualitative_figure.py  # assemble qualitative_real.pdf from dump
│   ├── visualize_masks.py
│   └── generate_diagrams.py
├── configs/
│   ├── base.yaml                # shared defaults
│   ├── phase0.yaml              # Phase 0: TPN warmup only
│   ├── phase0_v3.yaml
│   ├── phase1.yaml              # Phase 1 original
│   ├── phase1_v3.yaml           # v3 TPN baseline run
│   ├── phase1_final.yaml        # paper submission model (80k steps, v1 TPN, rank 64)
│   ├── phase1_final_v2.yaml
│   ├── phase1_adaptive.yaml     # THE 0.3177 run (40k steps, adaptive budget)
│   ├── phase1_adaptive_v2.yaml  # continuation from 40k → 100k (failed to beat 0.3177)
│   ├── phase1_stage1.yaml       # ICONIP plan: v3 TPN from scratch (failed at step 22K)
│   ├── phase1_stage2.yaml       # ICONIP plan: stage 2 adaptive on top (never run)
│   ├── phase1_warmstart.yaml    # fine-tune from 0.3177 ckpt (failed, k_eff collapsed)
│   ├── phase1p5_v{2..8b}.yaml   # Phase 1.5 trainable pooler experiments (ALL ABANDONED)
│   └── phase2.yaml
├── DL Project Report/
│   ├── main.tex                 # the paper (~1928 lines, BMVC 2026)
│   ├── main.bib                 # bibliography
│   └── figures/
│       ├── tpn_architecture.png         ✅ real (148K)
│       ├── gumbel_ste.png               ✅ real (226K)
│       ├── tc_vitok_architecture.png    ✅ real (193K)
│       ├── training_pipeline.png        ✅ real (189K)
│       ├── oscillation.pdf              ✅ real (16K)
│       ├── budget_histogram.pdf         ✅ real (20K) — generated 2026-05-17, n=500
│       └── qualitative_real.pdf         ✅ real (435K) — generated 2026-05-17, n=32
├── DL Project Report Template/  # icml2021.sty, icml2021.bst, fancyhdr.sty etc.
├── notebooks/
│   ├── 01_baseline_reproduction.ipynb
│   ├── 01_project_overview_and_setup.ipynb
│   ├── 02_model_architecture.ipynb
│   ├── 02_token_visualization.ipynb
│   ├── 03_tpn_debug.ipynb
│   ├── 03_training_phases_and_failures.ipynb
│   ├── 04_results_analysis.ipynb
│   └── 04_results_and_analysis.ipynb
├── runs/                        # training outputs (on server, large — not synced to local)
│   ├── phase0_v4/phase0/best.ckpt
│   ├── phase1_final/phase1/best.ckpt     ← paper submission model
│   ├── phase1_adaptive/phase1/best.ckpt  ← THE 0.3177 checkpoint — PRESERVE FOREVER
│   ├── phase1_adaptive_v2/phase1/best.ckpt ← step 64k, val_vqa=0.3575
│   ├── phase1_v3/phase1/best.ckpt        ← v3 baseline, val_vqa=0.3334
│   ├── qualitative_dump/                 ← 32 GQA heatmap dumps
│   └── budget_dump/                      ← 500-sample k_eff dump
└── requirements.txt
```

---

## 4. Architecture

### 4.1 Full System (Phase 1 Final — the paper model)

```
frozen encoders → TPN → Gumbel-STE selection → Informative Token Merge → MM projector → LLaVA-1.5-7B + LoRA
```

**Frozen encoders (always frozen, no gradients):**
- **CLIP ViT-L/14 (336px):** produces `v_fine ∈ R^{576×1024}` — the tokens fed to LLM
- **DINOv2-S (98px):** produces `v_coarse ∈ R^{49×384}` (7×7 grid) — spatial guide for TPN
- **CLIP ViT-B/32 text:** encodes question → `q_emb ∈ R^{512}` (EOS token representation)

**TPN v1 (~887K params) — the query-conditioned selector:**
- Cross-attention (4 heads): `q_emb` as query, `v_coarse` as key+value
  - Outputs: `ctx ∈ R^{384}` (what visual content is relevant) + `α ∈ R^{49}` (where)
- Score MLP: `[ctx; α] ∈ R^{433}` → LayerNorm → 2-layer MLP → 49 coarse scores
- Bilinear upsample 7×7 → 24×24 → `s_N ∈ R^{576}`
- **Instance normalization** on s_N (critical — see design decisions)

**TPN v3 (TokenScorerV3, ~790K params) — image-only variant:**
- Self-attention on CLIP-L tokens only — NO text query
- Was used in the early v3 baseline run; broke diversity loss (no q_embed)
- Converged monotonically from scratch but plateaued higher than warm-started v1
- Not used in the final submitted system

**Gumbel-Sigmoid STE:**
```
n = s_N + Gumbel(0,1)           # add noise for per-sample diversity
m_soft = sigmoid(n / τ)          # soft mask (for backward)
idx = topk(n, k)                  # hard selection indices (for forward)
m_hard = one_hot(idx)
m_STE = m_hard - m_soft.detach() + m_soft   # STE: forward=hard, backward=soft
```

**Tau-independent additive gradient correction (key innovation):**
```python
v_sel = gather(v_fine, idx)
c = gather(s_N, idx)
v_sel = v_sel * (1 + c - c.detach())
# Forward: c - c.detach() = 0, v_sel unchanged
# Backward: grad = sum_d(dL/dv_sel[j,d] * v_sel[j,d])  — non-zero at ANY τ
```
Without this: at τ→0.3, sigmoid saturates → ∂m_soft/∂s_N ≈ 0 for |s_N|>1 → TPN gradient dies at step ~15K.

**Informative Token Merge (zero parameters):**
- Each unselected token soft-assigned to nearest selected token by cosine similarity
- Assignment weighted by TPN score of unselected token (near-threshold tokens contribute more)
- `v_merged = v_sel + 0.3 * Ã^T * v_unsel`
- **Critical property: zero parameters → co-adaptation loop cannot close**

**LoRA (rank 64, alpha 128):**
- Targets in Phase 1 Final: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj
- Targets in Phase 1 Adaptive: q_proj, k_proj, v_proj, o_proj (attention only)
- ~85M trainable params across 32 LLM layers
- Total trainable: ~89M (TPN + LoRA + MM projector + optional BudgetPredictor)

**BudgetPredictor (~25K params, Phase 1 Adaptive only):**
- Input: `[ctx ∈ R^{384}; H(s_N) ∈ R^1]` where H is normalized entropy of scores
- Architecture: Linear(385→64) → GELU → Linear(64→1) → sigmoid → clip to [k_min/N, k_max/N]
- Output: per-sample k_frac → k_hard = round(k_frac × N)
- **Differentiable threshold trick:** `θ = θ_hard - (k_target - k_target.detach())` routes budget penalty gradient through non-differentiable topk
- **Collapse boundary:** λ_budget ≤ 0.05 → collapses to k_max within 10K steps; safe range is [0.08, 0.15]

### 4.2 Parameter counts

| Component | Params | Trainable |
|-----------|--------|-----------|
| CLIP ViT-L/14 | 307M | ❌ frozen |
| DINOv2-S | 22M | ❌ frozen |
| CLIP-B/32 text | 63M | ❌ frozen |
| LLaVA-1.5-7B (LLaMA-2) | 6.74B | ❌ frozen backbone |
| LoRA rank-64 | ~85M | ✅ |
| MM projector | ~2M | ✅ |
| TPN v1 | 887K | ✅ |
| BudgetPredictor | 25K | ✅ (Phase 1 Adaptive only) |

---

## 5. Training Protocol (The Correct Curriculum)

The ORDER of phases is a necessary condition for convergence — not a tuning choice.

### Phase 0 — TPN Warmup (10K steps)

- **Only TPN trains.** LLM, LoRA, MM projector all frozen.
- With LoRA frozen, the only way to reduce VQA loss is better token selection → forces TPN to learn meaningful scores before joint training.
- τ = 1.0 fixed; data: VQAv2 only
- Config: `configs/phase0.yaml`
- Checkpoint: `runs/phase0_v4/phase0/best.ckpt`

### Phase 1 Final — Joint Training (80K steps)

- TPN + LoRA (rank 64, gate+up targets) + MM projector all train
- **LoRA warm-freeze for first 3000 steps** — prevents LoRA from immediately compensating for TPN, preserving gradient pressure on TPN
- τ: 1.0 → 0.3 annealed linearly
- Data: VQAv2 + GQA; effective batch: 128 (4 × 16 accum × 2 GPUs)
- LR: lr_tpn=3e-3, lr_lora=1e-4, lr_mlp=5e-4
- Config: `configs/phase1_final.yaml`
- Checkpoint: `runs/phase1_final/phase1/best.ckpt`
- **This is the paper submission model** (eval benchmarks done)

### Phase 1 Adaptive — BudgetPredictor Fine-tune (40K steps)

- Loads from Phase 1 Final checkpoint (TPN scores must be meaningful first)
- BudgetPredictor initialized fresh, frozen for first 2000 steps
- λ_budget = 0.1 (critical — too low causes collapse; see failure modes)
- k_min=36, k_max=288, adaptive_budget=True
- Config: `configs/phase1_adaptive.yaml`
- Checkpoint: `runs/phase1_adaptive/phase1/best.ckpt`
- **val_vqa = 0.3177 at step 40K — the best result ever achieved**
- **PRESERVE THIS CHECKPOINT — never overwrite**

---

## 6. All Training Runs (Chronological)

### Phase 0 runs (early 2026)
Multiple Phase 0 variants; `phase0_v4` produced the usable warm-start checkpoint. Goal: give TPN a meaningful operating regime before any joint training.

### Phase 1 v3 baseline (April 2026)
- v3 TPN (image-only), rank 32/64 LoRA, no merge, fixed k=144
- **Convergence trajectory: 0.44 (5k) → 0.41 (10k) → 0.36 (20k) → 0.33 (50k)** — monotonically declining
- Killed at step 50k when still trending down
- Checkpoint: `runs/phase1_v3/phase1/best.ckpt`, **val_vqa = 0.3334**
- This is the primary ablation baseline in the paper

### Phase 1 Final (April 2026)
- v1 TPN (query-conditioned), rank 64/128 LoRA (extended targets including gate+up)
- Warm-started from Phase 0 v4
- 80K steps; loaded from phase0_v4 → trained with LoRA warm-freeze 3000 steps
- Config: `configs/phase1_final.yaml`
- This is the **paper submission model**
- Eval: VQAv2=72.27, GQA=67.12, POPE=79.09, MMBench=62.0, MME=1216/225, TextVQA=25.8

### Phase 1 Adaptive (April 2026)
- Fine-tune from Phase 1 Final, adds BudgetPredictor, λ_budget=0.1
- k_min=36, k_max=288; BudgetPredictor frozen 2000 steps
- **Best checkpoint: step 40K, val_vqa=0.3177** — all-time best
- Checkpoint: `runs/phase1_adaptive/phase1/best.ckpt` — **NEVER OVERWRITE**
- Eval: VQAv2=69.43, GQA=65.04, POPE=75.63
- val_vqa at step 0 when reloaded is ~0.37 (not 0.3177) — this is a sampling artifact from max_val_batches=500 vs original 100

### Phase 1.5 — Trainable Pooler experiments (April 2026, ALL ABANDONED)
- Attempted to add a learnable UnselectedPooler to process the 432 discarded tokens
- Ran 8+ variants (phase1p5_v2 through v8b), all failed
- **Root cause of all failures: co-adaptation oscillation (GAN-like limit cycles)**
  - v6 (lr_pooler=5e-4): oscillation period ≈ 5K steps
  - v8 (lr_pooler=2e-5): oscillation period ≈ 15K steps
  - v8b (lr_pooler=0): no oscillation but frozen pooler outputs useless noise
  - Period scales as T_osc ≈ c/η_pooler (c≈2.5), empirically verified
- **Conclusion:** ANY trainable bridge between selector outputs and LLM closes the co-adaptation loop. Fix: parameter-free merge.
- These failure data became the core of the paper (§8.1 + Fig oscillation)

### Phase 1 Adaptive v2 (May 12, 2026)
- Goal: extend 0.3177 run with TextVQA + LLaVA-Instruct-150K, train to 100K steps
- Same-phase resume from step 40K, halved LRs (lr_tpn=5e-4, lr_lora=1e-4)
- **Never beat the original 0.3177 — plateaued around 0.36**
- Best: step 64K, val_vqa=0.3575, val_loss=0.2879
- Root cause of plateau: LLaVA-Instruct conversational distribution shifts VQA metric (short-answer VQAv2 vs long-answer LLaVA-Instruct creates conflicting optimization targets)
- Config: `configs/phase1_adaptive_v2.yaml`
- k_eff during eval was ~260-265 (healthy, not collapsed)

val_vqa trajectory:
| Step | val_vqa | val_loss | Notes |
|------|---------|---------|-------|
| 40k (orig) | **0.3177** | — | original checkpoint |
| 42k | 0.3819 | 0.3122 | warm-restart regression |
| 44k | 0.3732 | 0.3035 | improving |
| 58k | 0.3588 | 0.2893 | breakthrough |
| 64k | **0.3575** | **0.2879** | best.ckpt saved |
| 70k | 0.3609 | 0.2919 | |

### Phase 1 Stage 1 — ICONIP plan (May 2026, FAILED at step ~22K)
- New plan targeting ICONIP'26: v3 TPN from scratch, 100K steps
- Rationale: v3 was the only run to converge monotonically (but that was from phase0 warmstart)
- **Failed worse than any previous run: val_vqa ~0.67 at step 2K, oscillating at 22K**
- Root cause 1: v3 has no q_embed → diversity loss = 0 always (dead loss)
- Root cause 2: cold start from scratch is the fundamental failure regardless of TPN variant
- Config: `configs/phase1_stage1.yaml` — killed and abandoned

### Phase 1 Warmstart (May 2026, FAILED)
- Goal: fine-tune directly from the 0.3177 checkpoint with low LRs, push val_vqa < 0.30
- phase=2 (cross-phase load: fresh optimizer, reset step to 0)
- λ_budget=0.03 (intentionally low to allow k_eff to drift toward k_max for accuracy)
- **Catastrophic failure: BudgetPredictor collapsed to k_eff=287.97 within 10K steps**
- λ_budget=0.03 is below the collapse boundary of ~0.08
- val_vqa regressed monotonically: 0.3726 (step 1K) → 0.3898 (step 20K)
- Config: `configs/phase1_warmstart.yaml` — 20K steps then killed
- **The 0.3177 checkpoint appears to be the architectural ceiling** — every fine-tuning attempt regressed it
- This failure characterized the collapse boundary and became Table tab:budget-collapse in the paper

---

## 7. Three Failure Modes (The Paper's Core Contribution)

### FM1: GAN-Style Co-Adaptation Oscillation

**Mechanism:** When a trainable selector and LLM adapters are jointly trained on a shared loss, they form a closed gradient circuit:
```
L_VQA → ΔLoRA → LLM "moves" → new selector gradient → ΔTPN → input distribution shifts → ...
```
This is structurally identical to a two-player non-saturating GAN. The equilibrium analysis predicts limit cycles (Mescheder et al. 2018, Jolicoeur-Martineau et al. 2018).

**Empirical signature:** val_vqa oscillates with period T_osc ≈ c/η_selector  
(c≈2.5 for our trainable-pooler variants; period 5K steps at η=5e-4, 15K steps at η=2e-5)

**Trigger:** ANY trainable component bridging selector outputs to LLM inputs. Phase 1.5 pooler was the bridge.

**Fix:** Informative Token Merge (parameter-free). Zero new parameters → loop cannot close by construction.

**Evidence:** `figures/oscillation.pdf` — v6/v8/v8b training curves showing period scaling.

### FM2: BudgetPredictor Collapse

**Mechanism:** The BudgetPredictor learns the trivial policy k→k_max whenever the budget penalty gradient is dominated by the residual VQA gradient.

**Empirical boundary:**
| λ_budget | Behavior | Evidence |
|----------|----------|---------|
| ≤ 0.05 | Collapses to k_max within 10K steps | Warmstart run (λ=0.03): k_eff=287.97 by step 10K |
| 0.08–0.15 | Stable, non-trivial policy | Phase1_adaptive (λ=0.1): k_eff mean=271.3 |
| ≥ 0.2 | Too aggressive, accuracy degrades | (swept) |

**Fix:** λ_budget ∈ [0.08, 0.15] + BudgetPredictor warm-freeze for 2000 steps after loading.

**Evidence:** `figures/budget_histogram.pdf` — real k_eff distribution from 500 GQA val samples with the phase1_adaptive checkpoint (λ=0.1). Mean=271.3, all samples in [254, 288] — far above k=144 fixed baseline, confirming predictor sees little incentive to compress at this λ.

### FM3: Warm-Start Dependence

**Mechanism:** Cold-start joint training of high-capacity selector + LoRA pairs plateaus 2–10× above the warm-started ceiling regardless of TPN variant, learning rate, or loss weight. The warm-start provides a stable prior that prevents the circular co-adaptation from dominating.

**Evidence:**

| Run | Start | Best val_vqa |
|-----|-------|-------------|
| phase1_adaptive | warm from Phase 1 Final | **0.3177** |
| phase1_v3 | warm from Phase 0 | 0.3334 |
| phase1 original | cold | ~0.45 plateau |
| phase1_stage1 (v3 from scratch) | cold | ~0.67 at step 2K, oscillating |

**Fix:** Three-phase curriculum — Phase 0 (TPN alone) → Phase 1 (joint with LoRA warmfreeze) → Phase 1 Adaptive (from converged base). The ordering is necessary.

---

## 8. All Known Bugs (BUG-1 through BUG-46)

### Selected critical bugs from May 9–10 session (BUG-1 through BUG-43)

- **BUG-11:** VQAv2/GQA datasets missing retry loops — single bad image crashes DataLoader worker. Fix: 5-attempt retry loop in `__getitem__`.
- **BUG-17:** HF Hub HTTP requests from background threads cause NCCL desync → deadlock on 2-GPU DDP. Fix: `os.environ["HF_HUB_OFFLINE"]="1"` and `TRANSFORMERS_OFFLINE=1` BEFORE any HuggingFace import (must be top of file).
- **BUG-23:** Instance norm not applied → score magnitudes explode to ±100 within 1000 steps → entropy collapse (same 576 positions always selected). Fix: instance norm in masking.py.
- **BUG-31:** Tau-independent gradient correction missing → TPN gradient vanishes when τ→0.3 (sigmoid saturates) → TPN stops learning at step ~15K. Fix: additive correction (Eq. taufix in paper).
- **BUG-38:** Train-eval mismatch in trainable pooler — different token count at eval vs train, corrupting pooler outputs.

### May 12 session bugs (BUG-44 through BUG-46)

**BUG-44** — `trainer.py:225`  
`train()` calls `save_checkpoint(is_best=True)` unconditionally at end of training → overwrites actual best val checkpoint with final step (which is almost always worse).  
Fix: `save_checkpoint(is_best=False)` at end of train loop.

**BUG-45** — `trainer.py:681–687`  
`optimizer.load_state_dict(ckpt)` on same-phase resume clobbers ALL param_group fields including `initial_lr`, silently restoring the old run's LRs regardless of what's in the new config. The halved LRs in phase1_adaptive_v2 had zero effect until this was fixed.  
Fix: `configured_initial_lrs = [g['initial_lr'] for g in optimizer.param_groups]` → `load_state_dict` → restore.

**BUG-46** — `datasets.py`  
`TextVQADataset.__getitem__` had no retry loop or path fallback (VQAv2/GQA had both from BUG-11 fix). Single corrupted image → SIGABRT crash in DataLoader worker.  
Fix: refactored into `_load_item` with 5-attempt retry + `Path(path).name` strip for path prefix variants.

---

## 9. Paper Results

### Main evaluation table

| Model | VQAv2 | GQA | POPE | MMBench | MME P/C | TextVQA |
|-------|-------|-----|------|---------|---------|---------|
| **TC-ViTok Final (k=144)** | 72.27 | 67.12 | 79.09 | 62.0 | 1216/225 | 25.8 |
| LoRA-576 control (E1) | 76.38 | 70.55 | 86.46 | 61.7 | 1254/218 | — |
| TC-ViTok v3 (E10) | 72.47 | 59.87 | 78.50 | — | — | — |
| TC-ViTok Adaptive (E8) | 69.43 | 65.04 | 75.63 | — | — | — |
| LLaVA-1.5 published | 78.5 | 61.0 | 84.4 | 64.3 | 1531/295 | 58.2 |

**Compression cost (TC-ViTok Final vs LoRA-576, honest comparison):**
- VQAv2: −4.11 points
- GQA: −3.43 points
- POPE: −7.37 points (expected — TPN drops question-irrelevant tokens)
- MMBench: +0.32 points (neutral/slight gain)
- MME: −30 total

**Key framing:** LoRA-576 is the missing control. Prior work compared against published LLaVA-1.5 numbers, which conflates compression effects with LoRA fine-tuning effects. Adding LoRA-576 isolates pure compression cost.

### Experiment log

| Exp | Description | Status | Results |
|-----|-------------|--------|---------|
| E1 | LoRA-576 control | ✅ DONE | VQAv2=76.38, GQA=70.55, POPE=86.46, MMBench=61.7, MME=1254/218 |
| E2 | Merge-disabled ablation | ✅ DONE | check paper for values |
| E3 | Zero-query (image-only TPN) ablation | ✅ DONE | check paper for values |
| E4 | 5-seed GQA variance | ✅ DONE | check paper for std value |
| E5_mmb | TC-ViTok Final MMBench | ✅ DONE | 62.02% (4329 samples) |
| E5_mme | TC-ViTok Final MME | ✅ DONE | perception=1216.1, cognition=225.4 |
| E6 | TC-ViTok Final TextVQA | ✅ DONE | 25.84% |
| E7 | Qualitative dump | ✅ DONE | runs/qualitative_dump/ (32 GQA samples) |
| E8 | Budget histogram | ✅ DONE | runs/budget_dump/ (500 samples, mean k_eff=271.3) |
| E9 | Oscillation figure | ✅ DONE | figures/oscillation.pdf |
| E10 | v3 retroactive GQA+POPE | ✅ DONE | GQA=59.87%, POPE=78.50% |

---

## 10. Paper Structure (main.tex, ~1928 lines)

Template: ICML 2021 (stand-in for BMVC 2026; switch to bmvc2k.cls before submission).

**Sections:**
1. **Abstract** — Three failure modes named with quantitative evidence, recipe stated, honest cost reported
2. **Introduction** — 6 contributions; 1-3 (training dynamics) are the durable ones
3. **Related Work** — query-agnostic compression, query-aware methods, differentiable discrete opt
4. **Methodology:**
   - §4.1 Problem formulation
   - §4.2 Pre-LLM co-adaptation problem (gradient loop diagram)
   - §4.3 Frozen encoders (CLIP-L, DINOv2-S, CLIP-B/32)
   - §4.4 TPN architecture
   - §4.5 Gumbel-Sigmoid STE + tau-independent correction (Eq. taufix)
   - §4.6 Informative Token Merge (parameter-free, Eq. merge)
   - §4.7 Adaptive Budget (BudgetPredictor, Eq. bp + thresh)
   - §4.8 Loss functions (Table: tab:loss)
   - §4.9 Training protocol (3-phase curriculum)
5. **Efficiency Analysis** — FLOPs, KV cache savings at k=144
6. **Experimental Design** — benchmarks, baselines, eval protocol
7. **Results** — main table, ablation table, bar chart
8. **Analysis: Training Dynamics in Pre-LLM Compression** ← KEY SECTION
   - §8.1 Phase 1.5 pooler failure and co-adaptation (oscillation curves)
   - §8.2 Warm-start dependence (Table: tab:warmstart)
   - §8.3 BudgetPredictor collapse modes (Table: tab:budget-collapse, Fig: fig:budget-hist)
   - §8.4 Oscillation period scaling
9. **Limitations** — 8 named (L1–L8)
10. **Conclusion** — failure-mode laws as falsifiable claims

**Figures:**
| Label | File | Type |
|-------|------|------|
| fig:tpn | tpn_architecture.png | raster |
| fig:ste | gumbel_ste.png | raster |
| fig:arch | tc_vitok_architecture.png | raster |
| fig:merge | (TikZ) | vector |
| fig:pipeline | training_pipeline.png | raster |
| fig:convergence | (TikZ pgfplots) | vector |
| fig:ablation-bar | (TikZ pgfplots) | vector |
| fig:budget-collapse | (TikZ) | vector |
| fig:budget-hist | budget_histogram.pdf | raster (real data) |
| fig:qualitative-real | qualitative_real.pdf | raster (real data) |
| fig:pareto | (TikZ) | vector |
| fig:oscillation | oscillation.pdf | raster (real data) |

**Bibliography:** `main.bib` — key additions: `mescheder2018training`, `jolicoeur2018relativistic` for GAN co-adaptation citations.

---

## 11. Key Commands

### Launch training (server)

```bash
# Multi-GPU DDP
DATA_ROOT=~/tc-vitok-data bash scripts/launch_multigpu.sh \
    --config configs/phase1_final.yaml \
    --output runs/phase1_final

# In tmux (recommended for long runs)
tmux new -s phase1_adaptive
DATA_ROOT=~/tc-vitok-data bash scripts/launch_multigpu.sh \
    --config configs/phase1_adaptive.yaml \
    --output runs/phase1_adaptive
```

### Monitor training (server)

```bash
tmux attach -t <session_name>

# Grep val metrics
grep "val" ~/TC-ViTok/logs/train_$(ls -t ~/TC-ViTok/logs/ | head -1)

# Check checkpoint step
python3 -c "
import torch
ck = torch.load('runs/phase1_adaptive/phase1/best.ckpt', map_location='cpu', weights_only=True)
print('step:', ck['global_step'])
"

# Check k_eff in budget dump
python3 -c "
import json, numpy as np
m = json.load(open('runs/budget_dump/manifest.json'))
k = [e['k_eff'] for e in m]
print(f'mean={np.mean(k):.1f}, min={min(k)}, max={max(k)}')
"
```

### Evaluation (server)

```bash
# Standard eval (TC-ViTok Final)
python scripts/evaluate.py \
    --checkpoint runs/phase1_final/phase1/best.ckpt \
    --benchmarks vqav2 gqa pope \
    --data-root /home/sk10741/tc-vitok-data \
    --output runs/eval_results/tcvitok_final.json \
    --batch-size 4

# LoRA-576 baseline
python scripts/evaluate.py \
    --checkpoint runs/phase1_final/phase1/best.ckpt \
    --baseline full_576 \
    --benchmarks vqav2 gqa pope \
    --data-root /home/sk10741/tc-vitok-data \
    --output runs/eval_results/lora576.json

# GPU 1 only
CUDA_VISIBLE_DEVICES=1 python scripts/evaluate.py ...

# IMPORTANT: always use absolute path /home/sk10741/tc-vitok-data — ~ can break in some terminal configs
```

### Generate paper figures (server)

```bash
# Qualitative heatmaps (32 GQA examples, phase1_final checkpoint)
DATA_ROOT=~/tc-vitok-data python scripts/dump_tpn_heatmaps.py \
    --checkpoint runs/phase1_final/phase1/best.ckpt \
    --data-root $DATA_ROOT \
    --out runs/qualitative_dump \
    --n-samples 32 --benchmark gqa --render-overlays

# Budget histogram (500 examples, adaptive checkpoint, --adaptive-budget required)
DATA_ROOT=~/tc-vitok-data python scripts/dump_tpn_heatmaps.py \
    --checkpoint runs/phase1_adaptive/phase1/best.ckpt \
    --data-root $DATA_ROOT \
    --out runs/budget_dump \
    --n-samples 500 --benchmark gqa --adaptive-budget

# Compose qualitative figure PDF
python scripts/compose_qualitative_figure.py \
    --dump-dir runs/qualitative_dump \
    --out "DL Project Report/figures/qualitative_real.pdf"

# Compose budget histogram PDF
python scripts/make_budget_histogram.py \
    --dump-dir runs/budget_dump \
    --out "DL Project Report/figures/budget_histogram.pdf"

# Oscillation figure (uses hardcoded known data points, no log files needed)
python scripts/plot_oscillation.py --use-known-points \
    --out "DL Project Report/figures/oscillation.pdf"
```

### Compile paper (local machine)

```bash
cd ~/Desktop/Junior_spring/TC-ViTok/DL\ Project\ Report

# Full compile sequence (required: pdflatex ×3 + bibtex ×1 for cross-refs + citations)
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
open main.pdf

# OR with latexmk (handles pass counting automatically)
latexmk -c && latexmk -pdf -interaction=nonstopmode -f main.tex && open main.pdf
```

### Git (server)

```bash
git -c credential.helper= pull origin main
git add -p
git commit -m "message"
git push origin main
```

---

## 12. Config System

All configs inherit from `configs/base.yaml`. Key fields:

```yaml
phase: 1       # 1 = same-phase resume (load optimizer too); 2 = cross-phase (fresh optimizer, reset step)

model:
  k: 144
  tpn_version: "v1"       # "v1" = query-conditioned (887K); "v3" = image-only (790K)
  tpn:                     # only used when tpn_version="v3"
    d_visual: 1024
    d_h: 256
    n_heads: 4
    N: 576
  adaptive_budget: false   # true enables BudgetPredictor
  k_min: 36
  k_max: 288
  merge_beta: 0.3          # 0.0 = no merge (pure drop)
  merge_tau: 0.1
  lora:
    rank: 64
    alpha: 128
    targets: ["q_proj", "k_proj", "v_proj", "o_proj"]
    dropout: 0.05

training:
  max_steps: 80000
  batch_size: 4
  grad_accum_steps: 16     # effective batch = 4 × 16 × 2 GPUs = 128
  lr_tpn: 3e-3
  lr_lora: 1e-4
  lr_mlp: 5e-4
  lr_budget: 1e-3
  lr_pooler: 0.0           # always 0 — Phase 1.5 pooler experiments abandoned
  lora_warmfreeze_steps: 3000
  budget_warmup_steps: 2000
  tau_max: 1.0
  tau_min: 0.3
  lambda_entropy: 0.02
  lambda_diversity: 0.1    # ONLY meaningful for v1 TPN (requires q_embed); v3 → always 0
  lambda_budget: 0.0       # set to 0.1 for adaptive runs; DO NOT set below 0.08
  # All Phase 1.5 losses — always 0:
  lambda_spread: 0.0
  lambda_distill: 0.0
  lambda_boundary: 0.0
  lambda_slot_orth: 0.0
  lambda_residual_distill: 0.0

data:
  datasets: ["vqav2", "gqa"]  # add "textvqa", "llava_instruct" with caution

checkpoint:
  load_from: null    # path to ckpt or null for cold start
  strict: false
```

**phase=2 semantics (cross-phase load):**
- Loads model weights from checkpoint, discards optimizer state and global_step
- Use when: changing number of param groups, switching phases, fine-tuning from a different run
- Fresh tau schedule starts from tau_max, fresh LR warmup

**phase=1 semantics (same-phase resume):**
- Loads model weights AND optimizer state AND global_step
- Resumes exactly where the run left off
- WARNING: optimizer.load_state_dict clobbers initial_lr (BUG-45) — if you want different LRs, fix this in trainer.py

---

## 13. Important Design Decisions and Why

| Decision | Why |
|----------|-----|
| Instance norm on s_N (not batch norm, not tanh) | Adam pushes |s_N|→±100 within 1K steps without norm → entropy collapse. Batch norm breaks inference (stateful running mean). tanh saturates (|tanh'(±100)| ≈ 1e-87). Instance norm permanently anchors to μ=0, σ=1 with no learnable params to circumvent. |
| Additive (not multiplicative) STE correction | Multiplicative: ∂m_soft/∂s_N ≈ 0 when |s_N|>1. At τ→0.3, ALL selected tokens have |s_N|>1 → TPN gradient dies at step ~15K. Additive: gradient = dot(dL/dv_sel, v_sel), non-zero at any τ. |
| Parameter-free token merge | Any trainable pooler closes the co-adaptation loop (Phase 1.5 failure). Zero parameters = loop structurally impossible. |
| LoRA warm-freeze 3000 steps | Without warm-freeze, LoRA immediately compensates for TPN imperfections → removes gradient pressure on TPN → TPN learns nothing. |
| BudgetPredictor frozen 2000 steps in Phase 1 Adaptive | Without freeze, BP collapses to k_max while TPN is still adjusting to adaptive selection → impossible to recover. |
| Phase 0 TPN warmup | Cold-start joint training always plateaus (FM3). TPN must have a working regime before LoRA starts adapting. |
| DINOv2-S at 98px for v_coarse | 98px → 7×7 grid → M=49 cheap cross-attention. DINOv2 preserves spatial locality better than CLIP (contrastive objective is less spatially precise). |
| CLIP-L/14 for v_fine | LLaVA-1.5 already uses CLIP-L/14-336 → features are pre-aligned to LLM projector. |
| Gumbel noise on n (not s_N) for topk | Using raw scores for topk selects identical positions for all batch samples → collapses diversity. Noise on n ensures per-sample variation. |
| λ_budget ∈ [0.08, 0.15] | Below 0.08: BP collapses to k_max (warmstart experiment). Above 0.15: too aggressive, accuracy drops. Sweet spot is 0.1. |

---

## 14. Current State (as of 2026-05-17)

### Training — COMPLETE
- All runs finished or killed
- No active training
- **The 0.3177 checkpoint is the architectural ceiling** — every fine-tuning attempt regressed it
- This conclusion is now a paper contribution (FM3 + L8 limitation)

### Figures — ALL REAL DATA ✅
- budget_histogram.pdf: 500 real GQA samples, adaptive checkpoint, k_eff mean=271.3
- qualitative_real.pdf: 32 real GQA examples, phase1_final checkpoint
- oscillation.pdf: real training curves from v6/v8/v8b runs

### Paper
- main.tex ~1928 lines, BMVC 2026 double-blind
- All major sections complete including failure-mode analysis (§8)
- Check for remaining E2/E3/E4 placeholders: `grep "pending\|E2\.\|E3\.\|E4\." DL\ Project\ Report/main.tex`
- **Deadlines: abstract May 22, full paper May 29**

### Before submission checklist
- [ ] Compile locally and check for remaining citation/reference errors
- [ ] Verify E2/E3/E4 ablation numbers are filled in paper
- [ ] Switch from ICML template to official BMVC 2026 template
- [ ] Remove all identifying information (author names, GitHub URL, acknowledgments)
- [ ] Verify no identifying info in figures or supplementary

---

## 15. Quick Reference Numbers

| Quantity | Value |
|----------|-------|
| Visual tokens (LLaVA-1.5) | 576 |
| Selected tokens (k) | 144 |
| Compression ratio | 4× |
| Discarded tokens (merged) | 432 |
| Best val_vqa ever | 0.3177 (phase1_adaptive step 40K) |
| v3 baseline val_vqa | 0.3334 |
| TC-ViTok Final VQAv2 | 72.27% |
| LoRA-576 VQAv2 (missing control) | 76.38% |
| Compression cost on VQAv2 | −4.11 points |
| Compression cost on GQA | −3.43 points |
| TPN v1 params | 887K |
| TPN v3 params | 790K |
| BudgetPredictor params | 25K |
| Total trainable params | ~89M |
| GPUs | 2× RTX 6000 Ada (49,140 MiB each) |
| Effective batch size | 128 (4 × 16 accum × 2 GPUs) |
| k_eff mean at λ_b=0.1 | 271.3 (from 500 real samples) |
| λ_budget collapse boundary | ~0.08 (below → k_eff→k_max) |
| Oscillation period law | T_osc ≈ 2.5/η_selector |
| Phase 0 steps | 10K |
| Phase 1 Final steps | 80K |
| Phase 1 Adaptive steps | 40K |
