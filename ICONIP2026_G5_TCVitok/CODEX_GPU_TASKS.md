# Codex Task Bundle — TC-ViTok ICONIP 2026 GPU Work

**Run order (priority):**
1. **Trivial-baseline evaluation at k=144** (NEW CODE, ~6 GPU-h) — **highest paper-strengthening ROI**
2. **Qwen-VL FM1 reproduction** (EXISTING CODE, ~24 GPU-h) — closes cross-architecture gap
3. **Optional: STE gradient-norm diagnostic** (LIGHT, ~2 GPU-h) — only if time permits

Both Task 1 and Task 2 are *independent*; they can run in parallel on separate GPUs. Task 2 has an existing implementation in `src/tc_vitok/qwenvl/` that needs only to be smoke-tested then launched.

If you must pick one and skip the rest: **do Task 1**. It is the missing within-paper control that an adversarial reviewer will weaponize, and it does not require any new VLM family — only the existing Phase-1-Final checkpoint.

---

## TASK 1 — Trivial-Baseline Evaluation at k=144 (REVISED)

> **Revision note.** A prior draft of this task incorrectly proposed a separate `trivial_baselines.py` with fabricated imports (`TCViTok`, `VQAv2Eval`, etc.). The actual eval entry point is `scripts/evaluate.py`, and a `RandomDropModel` baseline already exists in that file. The revised plan below extends the existing pattern instead.

### Why this matters
The ICONIP paper currently compares TC-ViTok Final against:
- `LoRA-576` (no compression)
- Published query-agnostic methods (different SFT mix)

The repo already has a `random_drop` baseline (uniform random selection at $k$), but the paper does not currently report it in Table 5. A reviewer will additionally ask: *"Does your learned selector beat a deterministic uniform grid or a CLS-attention top-$k$ prior?"* These two trivial baselines are missing entirely. This task adds them.

### Expected outcomes
- **Best case:** TC-ViTok Final ≥ all three trivial baselines (random, uniform grid, CLS-attention) by 3+ pp on VQAv2/GQA/POPE.
  → Strong positive result. Add a row block to Table 5 and a paragraph to §5.1 framing this as evidence that learned selection is meaningful. Flip Limitation (7) to a strength.
- **Neutral case:** TC-ViTok Final beats trivial baselines by 0–3 pp.
  → Honest reporting; modest learned-selector advantage.
- **Worst case:** TC-ViTok Final loses to or ties trivial baselines on any benchmark.
  → Honest reporting; the taxonomy still characterises *why* training fails to extract value over trivial selection, which is even more interesting in this case.

All three cases improve the paper because they convert an open question into a measured one.

### Repo structure (confirmed)

- **Eval entry point:** `scripts/evaluate.py` (already supports `none`, `random_drop`, `image_only`, `full_576`, `merge_disabled` baselines).
- **Base class:** `TCViTokModel` from `src/tc_vitok/models/tc_vitok_model.py`. The selection logic lives in the overridable method `_select_tokens(self, images_coarse, images_full, questions)` which returns `(v_selected, k_per_sample_or_None)`.
- **Helpers (already imported in evaluate.py):** `gather_tokens`, `informative_token_merge` from `tc_vitok.models.tc_vitok_model`.
- **Existing pattern to mirror:** `RandomDropModel` at `scripts/evaluate.py:76–97`. It uses `gather_tokens` (hard drop). New baselines should follow the same pattern so they are directly comparable to `random_drop`.
- **Full visual encoder:** `self.full_encoder` is a `FullVisualEncoder` (`src/tc_vitok/models/encoders.py:177`). Inside it, `self.full_encoder.clip.vision_model` is the underlying `transformers` CLIP vision model that can return attentions when called with `output_attentions=True`.

### What to implement

Make three edits to `scripts/evaluate.py` only. Do not create new files.

#### Edit 1 — Add `UniformGridModel` class (place after `RandomDropModel`, ~line 98)

```python
class UniformGridModel(TCViTokModel):
    """Baseline: deterministic uniform-grid selection at k=144.

    Selects every 4th token from N=576 (since 576/144=4), bypassing TPN.
    The output is identical across runs and images: positions [0, 4, 8, ..., 572].
    This is the strongest trivial baseline because it preserves spatial coverage.
    All other components (encoders, LoRA, LLM) are identical to TC-ViTok.
    """

    def _select_tokens(self, images_coarse, images_full, questions):
        if self.adaptive_budget:
            raise NotImplementedError(
                "UniformGridModel does not support adaptive budget; "
                "use --baseline none for adaptive eval."
            )
        v_full = self.full_encoder(images_full)             # [B, 576, 1024]
        B, N, _ = v_full.shape
        assert N % self.k == 0, (
            f"UniformGridModel expects N divisible by k for clean grid; "
            f"got N={N}, k={self.k}"
        )
        stride = N // self.k                                # 576 // 144 = 4
        grid_idx = torch.arange(0, N, stride, device=v_full.device)[: self.k]
        idx = grid_idx.unsqueeze(0).expand(B, -1).contiguous()  # [B, k]
        return gather_tokens(v_full, idx), None
```

#### Edit 2 — Add `CLSAttentionModel` class (place after `UniformGridModel`)

```python
class CLSAttentionModel(TCViTokModel):
    """Baseline: CLS-attention top-k selection (EViT / PruMerge style).

    Uses CLIP last-layer [CLS]->patch attention as the per-position score,
    then takes top-k. This is the standard 'query-agnostic learned prior'
    baseline that the literature treats as the strongest trivial selector.
    All other components are identical to TC-ViTok.
    """

    def _select_tokens(self, images_coarse, images_full, questions):
        if self.adaptive_budget:
            raise NotImplementedError(
                "CLSAttentionModel does not support adaptive budget; "
                "use --baseline none for adaptive eval."
            )
        # We need both features AND CLS->patch attention from the SAME forward.
        # FullVisualEncoder.forward() does not return attentions, so call the
        # underlying CLIP vision model directly with output_attentions=True.
        clip_vision = self.full_encoder.clip.vision_model
        with torch.no_grad():
            outputs = clip_vision(
                pixel_values=images_full,
                output_attentions=True,
                return_dict=True,
            )
        # last_hidden_state: [B, 577, 1024]  (CLS at index 0, patches 1..576)
        v_full = outputs.last_hidden_state[:, 1:, :].contiguous()    # [B, 576, 1024]
        # attentions: tuple of length n_layers, each [B, heads, 577, 577]
        last_attn = outputs.attentions[-1]                           # [B, heads, 577, 577]
        cls_to_patches = last_attn[:, :, 0, 1:].mean(dim=1)          # [B, 576]
        idx = cls_to_patches.topk(self.k, dim=-1).indices            # [B, k]
        return gather_tokens(v_full, idx), None
```

**Attention semantics.** CLIP's `[CLS]` token attends to patches via standard self-attention. We use the last-layer attention because that is closest to the final image representation; averaging across heads matches the convention used by EViT and PruMerge.

#### Edit 3 — Register the new baselines in `load_model` and `parse_args`

In `load_model` (around line 264), extend the `model_cls` dict:

```python
model_cls = {
    "none": TCViTokModel,
    "random_drop": RandomDropModel,
    "image_only": ImageOnlyTPNModel,
    "full_576": Full576Model,
    "merge_disabled": MergeDisabledModel,
    "uniform_grid": UniformGridModel,        # NEW
    "cls_attention": CLSAttentionModel,      # NEW
}[baseline]
```

In `parse_args` (around line 860), extend the `--baseline` choices list:

```python
parser.add_argument(
    "--baseline", default="none",
    choices=["none", "random_drop", "image_only", "full_576",
             "merge_disabled", "uniform_grid", "cls_attention"],   # NEW choices added
    help=(
        "Baseline variant. 'none' = full TC-ViTok. "
        "'full_576' = same LoRA + mm_proj as TC-ViTok but pass all 576 tokens. "
        "'merge_disabled' = Final TPN selection with beta=0 at eval. "
        "'image_only' = TPN with zeroed CLIP query. "
        "'random_drop' = uniform random selection at k. "
        "'uniform_grid' = every 4th token (deterministic). "
        "'cls_attention' = CLIP last-layer CLS->patch attention top-k."
    ),
)
```

### Run commands

Run four evaluations against the Phase-1-Final checkpoint, all on 5K subsets to match Table 5's conventions:

```bash
CKPT=runs/phase1_final/phase1/best.ckpt        # confirm actual path on your server
OUT=runs/trivial_baselines
mkdir -p $OUT

# Reference: TPN (TC-ViTok Final). Re-run for direct comparison.
DATA_ROOT=~/tc-vitok-data python scripts/evaluate.py \
    --checkpoint $CKPT --baseline none \
    --benchmarks vqav2 gqa pope \
    --max-samples 5000 \
    --output $OUT/tpn.json 2>&1 | tee $OUT/tpn.log

# Random drop (already in repo)
DATA_ROOT=~/tc-vitok-data python scripts/evaluate.py \
    --checkpoint $CKPT --baseline random_drop \
    --benchmarks vqav2 gqa pope \
    --max-samples 5000 \
    --output $OUT/random_drop.json 2>&1 | tee $OUT/random_drop.log

# Uniform grid (new)
DATA_ROOT=~/tc-vitok-data python scripts/evaluate.py \
    --checkpoint $CKPT --baseline uniform_grid \
    --benchmarks vqav2 gqa pope \
    --max-samples 5000 \
    --output $OUT/uniform_grid.json 2>&1 | tee $OUT/uniform_grid.log

# CLS-attention top-k (new)
DATA_ROOT=~/tc-vitok-data python scripts/evaluate.py \
    --checkpoint $CKPT --baseline cls_attention \
    --benchmarks vqav2 gqa pope \
    --max-samples 5000 \
    --output $OUT/cls_attention.json 2>&1 | tee $OUT/cls_attention.log
```

Confirm the exact `--output` flag with `python scripts/evaluate.py --help` if it differs (could be `--output-dir`, `--output-json`, etc.).

Expected wall-clock: ~1.5 GPU-h per baseline (3 benchmarks × 5K samples). Four baselines total → ~6 GPU-h. Can be parallelized across GPUs.

### Sanity checks (do BEFORE the 5K runs)

1. **TPN baseline reproducibility.** Run `--baseline none --max-samples 200` and confirm VQAv2/GQA/POPE are within ±1 pp of (72.27 / 67.12 / 79.09). If not, the eval pipeline has drifted and the comparison won't be apples-to-apples.
2. **Uniform-grid determinism.** Run `--baseline uniform_grid --max-samples 200` twice. Outputs must be byte-identical.
3. **CLS-attention shape sanity.** Add a one-time assertion in `CLSAttentionModel._select_tokens`: `assert v_full.shape[1] == 576` to catch any image-size mismatch.
4. **CLS-attention per-image variation.** Ensure `cls_to_patches` has shape `[B, 576]` with per-image variation (not broadcast). A common bug is selecting the same patches for every image; this would be a clear red flag in a 200-sample run.

If any sanity check fails, do not proceed to the 5K runs. Debug first.

### Paper integration (when results land)

Add this row block to Table 5 (`tab:main`). All three trivial baselines use the same LoRA adapter and the same hard-drop path (no merge) as the existing `random_drop` baseline, so they are directly comparable to each other and to TC-ViTok Final modulo the (empirically zero-effect) merge:

```latex
\midrule
\multicolumn{7}{l}{\emph{Trivial-selector baselines (same LoRA, $k{=}144$, hard drop):}} \\
Random drop                     & 144 & ?? & ?? & ?? & --- & --- \\
Uniform grid                    & 144 & ?? & ?? & ?? & --- & --- \\
CLS-attention top-$k$           & 144 & ?? & ?? & ?? & --- & --- \\
```

If TC-ViTok Final beats all three trivial baselines (best case), replace Limitation (7) with a positive paragraph in §5.1:

> *"To establish that the learned selector beats trivial selection, we evaluate TC-ViTok Final against three trivial baselines at the same $k{=}144$ and same LoRA adapter: random drop, uniform grid (every 4th patch), and CLS-attention top-$k$. TC-ViTok Final exceeds the strongest trivial baseline by $\Delta$VQAv2 / $\Delta$GQA / $\Delta$POPE points (Table~\ref{tab:main}); the learned selection rule extracts value beyond random selection, spatial uniformity, and CLIP's own CLS-attention prior."*

If the result is *negative* (TC-ViTok loses or ties on any benchmark), the framing changes:

> *"Comparing TC-ViTok Final against trivial baselines at the same $k{=}144$ reveals that the learned selector does not beat $X$ baseline on $Y$ benchmark. This is consistent with FM3 (warm-start dependence): even our converged learned selector does not extract value over trivial selection on every benchmark. The taxonomy's value is not invalidated --- it explains why learned selection underperforms in this regime --- but we cannot claim that learned selection is uniformly better than trivial here."*

In both cases, replace Limitation (7) with a brief, factual statement of what was measured (e.g., *"We compare against random-drop, uniform-grid, and CLS-attention top-$k$ baselines at $k{=}144$, Table~\ref{tab:main}"*).

---

## TASK 2 — Qwen-VL FM1 Reproduction (EXISTING CODE)

The Qwen-VL implementation already exists in `src/tc_vitok/qwenvl/` from a prior session. It has passed local syntax checks but has NOT been smoke-tested on GPU. Do the smoke test first.

### Smoke test (required before full run)

```bash
mkdir -p runs/qwenvl_fm1_smoke

CUDA_VISIBLE_DEVICES=0 DATA_ROOT=~/tc-vitok-data PYTHONPATH=src \
python3 -m tc_vitok.qwenvl.qwen_train \
  --config src/tc_vitok/qwenvl/configs/qwenvl_run_a.yaml \
  --output runs/qwenvl_fm1_smoke \
  --max-steps 2 \
  --eval-samples 4 \
  --train-samples-per-dataset 8 \
  --log-every 1 \
  --save-every 2 \
  2>&1 | tee runs/qwenvl_fm1_smoke/train.log
```

This runs 2 training steps + 1 eval pass on 8-sample subsets. Watch for:
- HF Qwen2-VL loading (transformers >= 4.45 required)
- Visual encoder output shape (variable per image — confirm padding works)
- LoRA attaching to q_proj/k_proj/v_proj/o_proj
- Forward pass through the trainable pooler
- LLM forward pass with visual prefix
- Backward pass (gradient through pooler + LoRA only, encoder frozen)
- Eval at step 2

Common failure modes and fixes:
- `OutOfMemoryError`: Check `nvidia-smi`. GPU 0 may have stale processes from prior sessions. Kill them with `nvidia-smi --query-compute-apps=pid --format=csv | tail -n +2 | xargs -r kill -9`, then retry.
- `KeyError` on image_grid_thw: Qwen2-VL processor version mismatch. Pin transformers >= 4.45 in the environment.
- `RuntimeError: mat1 and mat2 shapes cannot be multiplied`: Pooler `visual_dim` was inferred from a small batch but actual Qwen visual hidden dim differs. The `ensure_pooler` call in `QwenFM1Model.forward_loss` runs at first batch — verify the first batch produces sensible shapes.
- NaN loss in step 2: lower learning rate temporarily (`--config` overrides) or increase gradient clipping.

### Run A (η_sel = 5e-4)

```bash
tmux new -s qwenvl_a
mkdir -p runs/qwenvl_fm1_run_a

CUDA_VISIBLE_DEVICES=0 DATA_ROOT=~/tc-vitok-data PYTHONPATH=src \
python3 -m tc_vitok.qwenvl.qwen_train \
  --config src/tc_vitok/qwenvl/configs/qwenvl_run_a.yaml \
  --output runs/qwenvl_fm1_run_a \
  2>&1 | tee runs/qwenvl_fm1_run_a/train.log
```

12K steps, ~12 GPU-h. Validation runs every 500 steps and is logged to `runs/qwenvl_fm1_run_a/log.csv`.

**Early-signal check at step 5K:**

```bash
tail -10 runs/qwenvl_fm1_run_a/log.csv
```

If oscillation is visible (val_acc rising, falling, rising over 4–6 validation points), FM1 has reproduced on Qwen2-VL. This is the critical positive signal for the paper.

### Run B (η_sel = 2e-5)

```bash
tmux new -s qwenvl_b
mkdir -p runs/qwenvl_fm1_run_b

CUDA_VISIBLE_DEVICES=1 DATA_ROOT=~/tc-vitok-data PYTHONPATH=src \
python3 -m tc_vitok.qwenvl.qwen_train \
  --config src/tc_vitok/qwenvl/configs/qwenvl_run_b.yaml \
  --output runs/qwenvl_fm1_run_b \
  2>&1 | tee runs/qwenvl_fm1_run_b/train.log
```

Use a *separate* GPU (CUDA_VISIBLE_DEVICES=1) if available; otherwise launch sequentially after Run A.

### Analysis

```bash
python scripts/analyze_qwenvl_fm1.py \
  --run-a runs/qwenvl_fm1_run_a/log.csv \
  --run-b runs/qwenvl_fm1_run_b/log.csv \
  --output figures/oscillation_qwenvl.pdf
```

Outputs `figures/oscillation_qwenvl.pdf` and prints period estimates for both runs.

### Paper integration (when results land)

Add Figure 1b after Figure 1 in §4.1:

```latex
\begin{figure}[t]
\centering
\includegraphics[width=0.95\textwidth]{figures/oscillation_qwenvl.pdf}
\caption{Cross-architecture FM1 confirmation on Qwen2-VL-7B with a
trainable pooler. \emph{Top:} $\etasel{=}5{\times}10^{-4}$, period
$\approx [X]$K steps. \emph{Bottom:} $\etasel{=}2{\times}10^{-5}$,
period $\approx [Y]$K steps.}
\label{fig:oscillation_qwenvl}
\end{figure}
```

Add a paragraph to §4.1 just before the Remedy paragraph:

> **Cross-architecture confirmation.** We reproduce FM1 on Qwen2-VL-7B + LoRA with a trainable cross-attention pooler. At η_sel = 5×10⁻⁴, we observe oscillation with period approximately [X] K steps; at η_sel = 2×10⁻⁵, the period extends to approximately [Y] K steps (Fig. fig:oscillation_qwenvl). The same monotone relationship between selector learning rate and oscillation period observed on LLaVA-1.5-7B (Fig. 1) reproduces on a different (encoder, LLM, selector) triple. We did not run a parameter-free-merge control on Qwen2-VL; the existence of the cycle is the cross-architecture confirmation we claim.

Update the abstract (one sentence) — insert after "...a three-stage curriculum on a converged base":

> *"FM1 reproduces qualitatively on Qwen2-VL-7B + LoRA with a trainable cross-attention pooler, supporting cross-architecture portability of the oscillation finding."*

Update Limitation (2) to soften: change *"the predictions in Section 4.4 are stated to make cross-architecture testing concrete, not because we have run that test"* → *"FM1 has been reproduced on Qwen2-VL-7B (Fig. fig:oscillation_qwenvl); FM2 and FM3 cross-architecture tests remain future work."*

---

## TASK 3 (OPTIONAL) — STE Gradient-Norm Diagnostic

This converts the FM1 STE-correction ablation from "trust us, gradients die" to a measurable diagnostic. Only do this if Tasks 1 and 2 finish with time to spare.

### What to do

If existing training logs from the 8 trainable-pooler variants (used for FM1) include per-step TPN gradient norms, no GPU is needed — just plot from logs:

```bash
ls runs/pooler_variants_*/  # check what's available
grep -E "grad_norm.*TPN" runs/pooler_variants_*/train.log | head
```

If gradient norms are NOT logged (likely), re-run one short pooler-variant training (~6K steps) with the STE correction enabled vs disabled, logging `||∇_TPN L||_2` every 100 steps. Expected: without correction, the norm collapses to <1e-6 by step ~15K; with correction, it stays finite.

### Code skeleton

Create `scripts/diagnose_ste.py`:

```python
"""STE-correction gradient diagnostic.

Loads two short pooler-variant runs (with/without τ-independent correction),
extracts TPN gradient norms per step from logs, and plots them.
"""
import argparse, re, json
import numpy as np
import matplotlib.pyplot as plt

def parse_log(path):
    steps, norms = [], []
    pattern = re.compile(r"step=(\d+).*?tpn_grad_norm=([\deE\.\-\+]+)")
    with open(path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                steps.append(int(m.group(1)))
                norms.append(float(m.group(2)))
    return np.array(steps), np.array(norms)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--with-correction", required=True)
    p.add_argument("--without-correction", required=True)
    p.add_argument("--output", default="figures/ste_grad_norm.pdf")
    args = p.parse_args()

    s1, n1 = parse_log(args.with_correction)
    s0, n0 = parse_log(args.without_correction)

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.semilogy(s0, n0, label="without τ-independent correction", lw=1.5)
    ax.semilogy(s1, n1, label="with τ-independent correction", lw=1.5)
    ax.set_xlabel("Training step")
    ax.set_ylabel(r"$\|\nabla_{\mathrm{TPN}} \mathcal{L}\|_2$  (log scale)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output)
```

### Paper integration

If this works, add a small inset figure to §3.3 (STE section) and update the relevant sentence:

> *"Without Eq. eq:taufix, TPN gradient norm decays exponentially to numerical zero by step 15K (Fig. fig:ste_grad_norm); with the correction, it remains finite throughout training."*

This converts "gradients die" from an unsupported claim into a measured one. Optional but strengthens §3.3.

---

## EXPECTED WALL-CLOCK PLAN (assuming 2 GPUs)

| Time | GPU 0 | GPU 1 |
|---|---|---|
| Hour 0 | Smoke test Task 1 + Task 2 (in parallel, on the same GPU is fine for smoke tests since they're tiny) | (idle) |
| Hour 1–7 | Task 1: trivial-baseline eval (6 GPU-h) | Task 2: Qwen-VL Run A starts |
| Hour 1–13 | (Task 1 finishes at hour 7) | Task 2: Qwen-VL Run A continues |
| Hour 13–25 | Task 2: Qwen-VL Run B (if GPU 1 still busy with Run A, run sequentially on GPU 0 from hour 7) | Task 2: Qwen-VL Run A completes ~hour 13 |
| Hour 25–28 | Analysis scripts + paper integration | (idle) |

Total: ~28 hours wall-clock with 2 GPUs. ~36 hours sequential on 1 GPU.

If only 1 GPU is available, prioritize Task 1 (6 GPU-h) and submit with the trivial-baseline result. Defer Qwen-VL to a workshop submission or revision round.

---

## DELIVERABLES BACK TO HUMAN

After each task completes, the human (Muhammad Ali Shah) needs:

### From Task 1
- Four JSON files: `runs/trivial_baselines/{tpn,random_drop,uniform_grid,cls_attention}.json` — each with VQAv2 / GQA / POPE accuracy on 5K samples
- One sentence stating which case occurred (best / neutral / worst) and the exact deltas (TC-ViTok minus best-trivial-baseline, per benchmark)
- The proposed Table 5 row block (LaTeX snippet, filled in with the measured numbers)

### From Task 2
- `runs/qwenvl_fm1_run_a/log.csv` — val_acc trajectory
- `runs/qwenvl_fm1_run_b/log.csv` — val_acc trajectory
- `figures/oscillation_qwenvl.pdf` — the 2-panel figure
- The summary JSON with period estimates
- The proposed §4.1 paragraph + abstract sentence (LaTeX snippets)

### From Task 3 (if attempted)
- `figures/ste_grad_norm.pdf`
- The proposed §3.3 sentence update

---

## DO NOT

- Do not modify `main.tex` directly. Return LaTeX snippets to the human; the human controls page count.
- Do not run Task 1 with `--max-samples` smaller than 5000 except for the sanity-check passes. The paper's Table 5 reports 5K-sample evaluations.
- Do not commit checkpoints to git. The `runs/` directory should be gitignored.
- Do not skip the smoke tests. The Qwen-VL implementation has not been verified on GPU and is the highest failure-mode risk.
- Do not reuse the CLS-attention selector inside Task 2. It is specific to the CLIP ViT-L/14 architecture in TC-ViTok; Qwen-VL has a different vision encoder.

---

## SANITY-CHECK AT END

Before reporting results to the human, verify:

```bash
# Task 1
ls runs/trivial_baselines/tpn.json \
   runs/trivial_baselines/random_drop.json \
   runs/trivial_baselines/uniform_grid.json \
   runs/trivial_baselines/cls_attention.json
python3 -c "
import json
for name in ['tpn','random_drop','uniform_grid','cls_attention']:
    d = json.load(open(f'runs/trivial_baselines/{name}.json'))
    print(name, d)
"
# Each file must contain finite VQAv2 / GQA / POPE numbers.

# Task 2
ls runs/qwenvl_fm1_run_a/log.csv runs/qwenvl_fm1_run_b/log.csv
wc -l runs/qwenvl_fm1_run_a/log.csv  # should be >= 24 (12K steps / 500 + header)
ls figures/oscillation_qwenvl.pdf
```

If any of these fail, do not report to human until fixed.
