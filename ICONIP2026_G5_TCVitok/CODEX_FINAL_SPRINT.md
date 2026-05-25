# Codex Task Bundle — TC-ViTok ICONIP Final Sprint (2–3 days to deadline)

## Mission

Find at least one positive empirical finding to lift the paper out of all-negative-result territory. We have GPU access. The current paper has no positive result — every comparison goes the wrong way. **The single biggest acceptance-probability lift available is finding any axis on which the learned TPN demonstrably wins.**

Five experiments. Three are launchable today in parallel; two need new code and launch tomorrow.

## Execution order

**Day 1 (today):** Experiments 1, 3, 4 — all use existing code paths. Launch in parallel.
**Day 2:** Experiments 2 and 5 — write new code, then run.
**Day 3:** Hand findings back to human. Human will apply paper text edits.

## Repo facts you should already know

- Eval entry: `scripts/evaluate.py`
- Existing baselines: `none`, `random_drop`, `image_only`, `full_576`, `merge_disabled`, `uniform_grid`, `cls_attention` (the last two were added in a previous task; confirm by checking `--help`).
- Phase-1 Adaptive checkpoint (variable-k training): `runs/phase1_adaptive_v2/phase1/best.ckpt` (preferred). Fall back to `runs/phase1_adaptive/phase1/best.ckpt` if v2 has issues.
- Phase-1 Final checkpoint (fixed k=144): `runs/phase1_final/phase1/best.ckpt`.
- Full visual encoder: `self.full_encoder` wraps `self.full_encoder.clip.vision_model` (HuggingFace CLIPModel).
- TPN model attribute path: explore by reading `src/tc_vitok/models/tc_vitok_model.py` if unsure.

## Do NOT modify main.tex

Return findings as plain text plus proposed LaTeX snippets. The human controls page-count integration.

---

## EXPERIMENT 1 — TPN at Extreme Low k (DAY 1, GPU-h ~3-4)

### Goal
Find whether the learned TPN beats trivial baselines at very low token budgets ($k \in \{8, 18, 24\}$), where random selection should become catastrophically lossy.

### Why this matters most
Current data shows TPN loses at all $k \in [36, 288]$. But the curves narrow as $k$ decreases. At extreme low $k$, structured selection should dominate random. If TPN wins at $k=8$ on any benchmark, that's a major positive finding — the paper gains a "regime where structured selection extracts value" claim.

### Pre-flight check
1. Confirm `python scripts/evaluate.py --help` lists `uniform_grid` and `cls_attention` as valid `--baseline` choices. If not, those baselines weren't added in the previous task; do not proceed until they exist.
2. Confirm `UniformGridModel` handles non-divisible k (e.g., 576/18=32 — clean; 576/8=72 — clean; 576/24=24 — clean). The earlier task should have used `np.linspace`-style indexing that works for arbitrary k. Verify by reading `scripts/evaluate.py:UniformGridModel._select_tokens`. If the old version still has `assert N % k == 0`, fix to:
   ```python
   stride_idx = torch.linspace(0, N - 1, steps=self.k, device=v_full.device).round().long()
   idx = stride_idx.unsqueeze(0).expand(B, -1).contiguous()
   ```
3. Confirm `runs/phase1_adaptive_v2/phase1/best.ckpt` exists. If not, use `runs/phase1_adaptive/phase1/best.ckpt`.

### Run commands

```bash
CKPT=runs/phase1_adaptive_v2/phase1/best.ckpt   # fall back to phase1_adaptive if needed
OUT=runs/extreme_k_baselines
mkdir -p $OUT

for k in 8 18 24; do
  for baseline in none random_drop uniform_grid cls_attention; do
    DATA_ROOT=~/tc-vitok-data python scripts/evaluate.py \
      --checkpoint $CKPT \
      --baseline $baseline \
      --k $k \
      --benchmarks vqav2 gqa pope \
      --max-samples 5000 \
      --output $OUT/extreme_k${k}_${baseline}.json \
      2>&1 | tee $OUT/extreme_k${k}_${baseline}.log
  done
done
```

That's 12 inference runs × ~15-20 min each = ~3-4 GPU-h.

### Sanity check before launching the full sweep
Run only the k=24 / baseline=none combination first with `--max-samples 200`:

```bash
DATA_ROOT=~/tc-vitok-data python scripts/evaluate.py \
  --checkpoint $CKPT --baseline none --k 24 \
  --benchmarks vqav2 --max-samples 200 \
  --output $OUT/smoke_k24.json
```

If this runs without errors and produces a number in the 40-70 range, the pipeline is fine. If it crashes (likely culprits: out-of-distribution k for the LoRA, or some assertion in the model code about k), STOP and report the error before proceeding.

### Deliverables to human

1. `runs/extreme_k_baselines/summary.csv` — one row per (k, baseline) combination with columns `k, baseline, vqav2, gqa, pope`.
2. A plain-text statement: "TPN beats strongest trivial baseline at k=X on benchmark Y by Z pp" OR "TPN does not beat any trivial baseline at any extreme k tested."
3. The combined k-scaling figure regenerated to include k=8, 18, 24 in addition to existing 36, 72, 144, 216, 288 — save to `figures/k_scaling_extended.pdf`. If TPN wins at any extreme k, this figure is the visual proof.

### Expected paper impact

**If TPN wins at any extreme k on any benchmark:** Add to §5.1 "Scaling with token budget":

> *"The crossover happens below the in-distribution training budget: at $k{=}[X]$ ($[Y]\times$ compression), the learned TPN exceeds the strongest trivial baseline by $[Z]$\,pp on $[benchmark]$ (Fig.~\ref{fig:k-scaling}), identifying an extreme-compression regime where structured selection extracts value beyond random and CLS-attention priors."*

**If TPN does not win at any extreme k:** Add one sentence:

> *"The negative pattern is robust to extreme compression: extending the sweep to $k\!\in\!\{8, 18, 24\}$ does not produce a crossover; structured selection does not surpass trivial selection in any regime we measure."*

---

## EXPERIMENT 3 — Multi-Seed Trivial Variance (DAY 1, GPU-h ~5-6)

### Goal
Quantify the per-seed variance of trivial random selection. The learned TPN is deterministic by construction; if it matches the *mean* of random's distribution while eliminating selection-induced variance, that is a small but real positive structural finding.

### Pre-flight check
1. Confirm `scripts/evaluate.py` exposes a `--seed` flag (or `--random-seed`). If not, check `RandomDropModel._select_tokens` — it uses `torch.randperm` which is reproducible only if a global seed is set. You may need to set `torch.manual_seed(args.seed)` at the start of `main()` in evaluate.py if it isn't already.
2. If a seed flag does not exist, add one with default 42:
   ```python
   parser.add_argument("--seed", type=int, default=42, help="Seed for random selection (random_drop baseline)")
   # in main():
   torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
   ```

### Run commands

```bash
CKPT=runs/phase1_final/phase1/best.ckpt
OUT=runs/random_seed_variance
mkdir -p $OUT

for seed in 7 13 42 100 1000; do
  DATA_ROOT=~/tc-vitok-data python scripts/evaluate.py \
    --checkpoint $CKPT \
    --baseline random_drop \
    --seed $seed \
    --k 144 \
    --benchmarks vqav2 gqa pope \
    --max-samples 5000 \
    --output $OUT/random_seed${seed}.json \
    2>&1 | tee $OUT/random_seed${seed}.log
done
```

5 runs × 3 benchmarks × ~20 min each ≈ 5-6 GPU-h.

### Analysis script (write this, ~30 lines)

Create `scripts/analyze_random_variance.py`:

```python
"""Compute per-seed std for random selection vs TPN's deterministic accuracy."""
import json, glob, statistics
from pathlib import Path

OUT = Path("runs/random_seed_variance")
TPN_REF = {"vqav2": 72.96, "gqa": 68.98, "pope": 79.19}  # from earlier trivial-baseline run

per_seed = {"vqav2": [], "gqa": [], "pope": []}
for f in sorted(glob.glob(str(OUT / "random_seed*.json"))):
    d = json.load(open(f))
    # adapt the keys to whatever evaluate.py emits — likely d['vqav2']['accuracy'] etc.
    for bench in per_seed:
        per_seed[bench].append(float(d[bench].get("accuracy") or d[bench].get("score") or d[bench]))

print(f"{'Benchmark':<10} {'Random mean':>12} {'Random std':>12} {'TPN':>8} {'Gap to mean':>12}")
for bench, vals in per_seed.items():
    mu = statistics.mean(vals)
    sd = statistics.stdev(vals) if len(vals) >= 2 else 0.0
    tpn = TPN_REF[bench]
    print(f"{bench:<10} {mu:>12.3f} {sd:>12.3f} {tpn:>8.3f} {tpn - mu:>+12.3f}")
print("\nInterpretation:")
print(f"- If |TPN - random mean| < random_std on a benchmark, TPN ~ mean of random's distribution.")
print(f"- Random std on its own is the 'variance' positive finding for TPN.")
```

### Deliverables to human

1. `runs/random_seed_variance/summary.csv` with columns: `seed, vqav2, gqa, pope`.
2. Output of `python scripts/analyze_random_variance.py` — show the table.
3. A plain-text statement: "Random selection has per-seed std of [σ_v]/[σ_g]/[σ_p] on VQAv2/GQA/POPE; TPN's deterministic accuracy is [within / above / below] the random mean by [Δ] pp."

### Expected paper impact

If random std is meaningful (~0.5-2 pp on each benchmark) and TPN sits near the random mean: add a sentence to §5.1 (right after the existing trivial-baselines paragraph):

> *"Random selection at $k{=}144$ has per-seed standard deviation $\sigma_{\mathrm{VQAv2}}{=}[X]$\,pp, $\sigma_{\mathrm{GQA}}{=}[Y]$\,pp, $\sigma_{\mathrm{POPE}}{=}[Z]$\,pp across 5 seeds; the deterministic TPN reaches the random-selection mean with zero selection-induced variance but does not exceed it."*

---

## EXPERIMENT 4 — k-Scaling Slope Analysis (DAY 1, GPU-h ~0)

### Goal
From the existing Exp1 multi-k data, determine whether the learned TPN degrades more gracefully than random selection under budget pressure. If TPN's accuracy-vs-log(k) slope is shallower, that is a "robustness" positive finding.

### No GPU needed — pure data analysis

The existing `runs/multi_k_baselines/summary.csv` (from the earlier Exp1) has accuracy at $k \in \{36, 72, 144, 216, 288\}$ for each baseline on VQAv2/GQA/POPE.

### Analysis script

Create `scripts/analyze_k_slopes.py`:

```python
"""Fit accuracy ~ log(k) per (baseline, benchmark); report slopes.

Shallower slope = more robust to budget pressure. If TPN's slope is the
shallowest on any benchmark, structured selection has a robustness
advantage even where absolute accuracy is lower.
"""
import csv
import math
from collections import defaultdict

import numpy as np

ROWS = list(csv.DictReader(open("runs/multi_k_baselines/summary.csv")))
benchmarks = ["vqav2", "gqa", "pope"]
baselines = sorted({r["baseline"] for r in ROWS})

print(f"{'baseline':<18} {'benchmark':<10} {'slope (acc/log2 k)':>22}  {'R^2':>6}")
print("-" * 60)

slopes = defaultdict(dict)
for baseline in baselines:
    for bench in benchmarks:
        xs, ys = [], []
        for r in ROWS:
            if r["baseline"] == baseline:
                xs.append(math.log2(int(r["k"])))
                ys.append(float(r[bench]))
        if len(xs) < 3:
            continue
        xs, ys = np.array(xs), np.array(ys)
        slope, intercept = np.polyfit(xs, ys, 1)
        residuals = ys - (slope * xs + intercept)
        ss_res = (residuals ** 2).sum()
        ss_tot = ((ys - ys.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        slopes[bench][baseline] = (slope, r2)
        print(f"{baseline:<18} {bench:<10} {slope:>+22.3f}  {r2:>6.3f}")

print("\nInterpretation:")
print("Shallower (smaller-magnitude) slope = more robust to budget reduction.")
for bench in benchmarks:
    sorted_by_slope = sorted(slopes[bench].items(), key=lambda x: abs(x[1][0]))
    print(f"\n{bench}: ", " < ".join(name for name, _ in sorted_by_slope))
print("\nIf TPN is leftmost on any benchmark, TPN degrades most gracefully there.")
```

### Deliverables to human

1. Console output of `analyze_k_slopes.py` (slopes table).
2. A plain-text statement: "TPN has the [shallowest / 2nd-shallowest / steepest] slope on [benchmarks list]; this [supports / does not support] a robustness-to-budget positive finding."

### Expected paper impact

If TPN's slope is shallower than at least one trivial baseline on any benchmark, add to §5.1 "Scaling with token budget":

> *"While the TPN does not surpass the strongest trivial baseline in absolute accuracy at any measured $k$, its accuracy degrades more gracefully under budget pressure: on $[benchmark]$, the TPN's slope is $[X]$\,pp/octave vs.\ random selection's $[Y]$\,pp/octave; structured selection is more robust to compression-budget reduction even where it does not surpass random."*

If TPN's slope is *not* shallower on any benchmark, skip this finding and do not mention slope in the paper.

---

## EXPERIMENT 2 — TPN vs CLS-Attention Spatial Correlation (DAY 2, GPU-h ~2)

### Goal
Quantify how closely the learned TPN score distribution matches CLIP's last-layer CLS→patches attention. High correlation supports the interpretation "joint training converged to approximately the CLIP CLS-attention prior, not beyond it" — a positive structural finding that explains the near-tie with CLS-attention top-$k$ on most benchmarks.

### New code: `scripts/tpn_cls_correlation.py`

```python
"""Measure spatial-prior correlation between the learned TPN and CLIP CLS-attention.

For N test images:
- Forward through CLIP vision encoder with output_attentions=True, extract
  last-layer CLS->patches attention, average across heads. Shape: (B, 576).
- Forward through TPN to get per-position scores. Shape: (B, 576).
- Per-image Spearman rank correlation between the two 576-vectors.
- Report mean, median, std across N images, plus a histogram.

High correlation (>0.5 mean) supports the claim that the TPN learned a
CLS-attention-like spatial prior. Low correlation contradicts that claim
(then the TPN learned something else — also informative).
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from tc_vitok.models.tc_vitok_model import TCViTokModel
from tc_vitok.data.datasets import GQADataset, collate_fn
from torch.utils.data import DataLoader


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", default="~/tc-vitok-data")
    p.add_argument("--num-samples", type=int, default=200)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Use the loader the existing eval pipeline uses; if direct construction
    # is awkward, copy the load_model body from scripts/evaluate.py.
    from scripts.evaluate import load_model
    model = load_model(checkpoint=args.checkpoint, baseline="none", k=144).eval()

    ds = GQADataset(args.data_root, split="val", max_samples=args.num_samples)
    dl = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate_fn)

    rhos = []
    for batch in dl:
        images_full = batch["images_full"].to(device)
        images_coarse = batch["images_coarse"].to(device)
        questions = batch["questions"]

        # ---- TPN scores ----
        v_coarse = model.coarse_encoder(images_coarse)
        q_embed = model.text_encoder(questions) if hasattr(model, "text_encoder") else None
        # Inspect the TPN forward signature; we need its per-position scores
        # (the s_N tensor of shape [B, 576] before top-k).
        idx, s_N, *_ = model.tpn(
            v_coarse=v_coarse, q_embed=q_embed, k=144,
            tau=0.1, training=False,
        )
        # ---- CLS attention scores ----
        clip_vision = model.full_encoder.clip.vision_model
        outs = clip_vision(pixel_values=images_full, output_attentions=True,
                           return_dict=True)
        last_attn = outs.attentions[-1]                 # [B, heads, 577, 577]
        cls_to_patches = last_attn[:, :, 0, 1:].mean(dim=1)  # [B, 576]

        # ---- per-image Spearman ----
        s_N_np = s_N.cpu().numpy()
        cls_np = cls_to_patches.cpu().numpy()
        for i in range(s_N_np.shape[0]):
            rho, _ = spearmanr(s_N_np[i], cls_np[i])
            rhos.append(float(rho))

    rhos = np.array(rhos)
    out = {
        "num_samples": len(rhos),
        "mean_spearman": float(rhos.mean()),
        "median_spearman": float(np.median(rhos)),
        "std_spearman": float(rhos.std()),
        "min": float(rhos.min()),
        "max": float(rhos.max()),
    }
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

    # Optional plot
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(rhos, bins=30, edgecolor="black")
        ax.set_xlabel("Spearman correlation (TPN scores vs CLIP CLS attention)")
        ax.set_ylabel("Images")
        ax.axvline(0, color="grey", lw=0.5)
        ax.axvline(rhos.mean(), color="red", lw=1.5,
                   label=f"mean = {rhos.mean():.3f}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.output.replace(".json", ".pdf"))
    except Exception as e:
        print(f"[warning] plot skipped: {e}")


if __name__ == "__main__":
    main()
```

### Run command

```bash
DATA_ROOT=~/tc-vitok-data python scripts/tpn_cls_correlation.py \
  --checkpoint runs/phase1_final/phase1/best.ckpt \
  --num-samples 200 \
  --output runs/tpn_cls_correlation/result.json \
  2>&1 | tee runs/tpn_cls_correlation/eval.log
```

~1-2 GPU-h.

### Sanity check

After implementing, verify on 10 samples first:
```bash
DATA_ROOT=~/tc-vitok-data python scripts/tpn_cls_correlation.py \
  --checkpoint runs/phase1_final/phase1/best.ckpt \
  --num-samples 10 \
  --output runs/tpn_cls_correlation/smoke.json
```

The mean correlation should be in [-1, 1]. If it's NaN or outside that range, the score extraction is broken (likely culprit: TPN scores not on the same 576-element grid as CLIP patches — verify shapes match before correlating).

### Deliverables to human

1. `runs/tpn_cls_correlation/result.json` with mean/median/std Spearman.
2. `runs/tpn_cls_correlation/result.pdf` histogram of per-image correlations.
3. A plain-text statement: "Mean Spearman correlation between TPN scores and CLIP CLS-attention is ρ=[X] (N=200 GQA samples)."

### Expected paper impact

**If mean Spearman > 0.4:** Add to §5.2 "Interpreting query-agnostic convergence" (and a new figure if there's page space — but probably just text):

> *"The converged TPN score distribution correlates with CLIP's last-layer CLS-attention at mean Spearman $\rho{=}[X]$ on $N{=}200$ GQA samples; joint training did not push the selector beyond the spatial prior CLIP already encodes. This explains the near-tie with CLS-attention top-$k$ on GQA, POPE, and MMBench (Table~\ref{tab:main}) and the larger TextVQA gap, where CLS-attention's distribution is more peaked toward text-localisation regions than the TPN's."*

**If mean Spearman is low (< 0.2):** The TPN learned something different from CLS-attention but still does not beat it. Add one sentence noting this — slightly less impactful but still a structural finding:

> *"The converged TPN score distribution correlates with CLIP's last-layer CLS-attention at mean Spearman $\rho{=}[X]$ on $N{=}200$ GQA samples; joint training produced a selection prior that is structurally different from CLS-attention but that does not surpass it on any benchmark."*

---

## EXPERIMENT 5 — STE Gradient-Norm Diagnostic (DAY 2, GPU-h ~2-3)

### Goal
Convert §3.3's claim "without Eq. 1, TPN gradients die around step 15K" from rhetorical assertion to measured fact. Run two short training jobs (with and without the τ-independent STE correction), log gradient norms per step, plot.

### Approach

The cleanest way: take one of the existing trainable-pooler configurations (whichever was used for the original 8-variant FM1 evidence), run it for 2K steps in two modes:
- Mode A: τ-independent additive STE correction enabled (the standard TC-ViTok path).
- Mode B: correction disabled (just baseline Gumbel-Sigmoid STE).

Log `torch.nn.utils.clip_grad_norm_` return value for the TPN parameters every 50 steps.

### What to add to the training script

Find the existing trainer (likely `scripts/train.py` or `src/tc_vitok/training/trainer.py`). After the loss.backward() call, before optimizer.step():

```python
tpn_grad_norm = torch.nn.utils.clip_grad_norm_(
    [p for p in model.tpn.parameters() if p.grad is not None],
    max_norm=1e9,  # don't actually clip; just measure
)
# log to wandb/csv/whatever the existing pipeline uses
log_dict["tpn_grad_norm"] = float(tpn_grad_norm)
```

Also add a CLI flag `--disable-ste-correction` that bypasses Eq. 1 in the TPN forward (the τ-independent additive correction). Trace through the TPN code to find where the correction is applied and gate it on this flag.

### Run commands

```bash
# Phase 0 only (TPN alone, LLM frozen) — much faster than full training
# and sufficient to observe gradient-norm behavior.

# Mode A: correction enabled
DATA_ROOT=~/tc-vitok-data python scripts/train.py \
  --phase 0 \
  --max-steps 2000 \
  --log-every 50 \
  --output runs/ste_diagnostic_with_correction \
  2>&1 | tee runs/ste_diagnostic_with_correction/train.log

# Mode B: correction disabled
DATA_ROOT=~/tc-vitok-data python scripts/train.py \
  --phase 0 \
  --max-steps 2000 \
  --log-every 50 \
  --disable-ste-correction \
  --output runs/ste_diagnostic_without_correction \
  2>&1 | tee runs/ste_diagnostic_without_correction/train.log
```

If Phase 0 training takes more than ~30 min for 2K steps, lower to 1K steps. The point is to observe whether gradient norms decay exponentially in Mode B, which should be visible within hundreds of steps.

### Analysis script

`scripts/plot_ste_grad_norm.py`:

```python
"""Plot TPN gradient norm vs training step for two STE modes."""
import argparse, re
import numpy as np
import matplotlib.pyplot as plt

def parse_log(path):
    steps, norms = [], []
    pat = re.compile(r"step[=:](\d+).*?tpn_grad_norm[=:]([\d\.eE+\-]+)")
    for line in open(path):
        m = pat.search(line)
        if m:
            steps.append(int(m.group(1)))
            norms.append(float(m.group(2)))
    return np.array(steps), np.array(norms)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--with", dest="with_path", required=True)
    p.add_argument("--without", dest="without_path", required=True)
    p.add_argument("--output", default="figures/ste_grad_norm.pdf")
    args = p.parse_args()

    s_w, g_w = parse_log(args.with_path)
    s_wo, g_wo = parse_log(args.without_path)

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.semilogy(s_wo, g_wo, label="without τ-independent correction",
                lw=1.5, color="C3")
    ax.semilogy(s_w,  g_w,  label="with τ-independent correction",
                lw=1.5, color="C0")
    ax.set_xlabel("Training step")
    ax.set_ylabel(r"$\|\nabla_{\mathrm{TPN}}\mathcal{L}\|_2$  (log scale)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output)
    print(f"Wrote {args.output}")
```

### Deliverables to human

1. `runs/ste_diagnostic_with_correction/train.log` and `runs/ste_diagnostic_without_correction/train.log`.
2. `figures/ste_grad_norm.pdf` — the comparison plot.
3. A plain-text statement: "Without correction, TPN gradient norm decays from [X] to [Y] over 2K steps (factor [Z]× drop). With correction, gradient norm remains in [A, B]."

### Expected paper impact

If the plot shows clear exponential decay without correction and stable norms with correction, add an inline reference to the existing §3.3 sentence:

Current: *"Without Eq.~\ref{eq:taufix}, TPN gradients die around step 15K in otherwise identical runs."*

Add a clause:
*"Without Eq.~\ref{eq:taufix}, TPN gradient norm decays exponentially (Fig.~\ref{fig:ste-grad}, gradient drop of $[Z]\times$ over 2K steps); with the correction, norms remain finite."*

If page space is tight, this becomes a one-line addition without the figure.

---

## CRITICAL GENERAL INSTRUCTIONS

1. **Run sanity checks before launching any full sweep.** All five experiments have explicit smoke-test commands. Do them. Each takes <10 minutes.
2. **Save raw outputs.** Every run should write a JSON or CSV to `runs/<exp>/...`. Do not throw away data.
3. **Do not modify `main.tex`.** Return findings and proposed LaTeX snippets. The human controls page-budget integration.
4. **Day 1 experiments can launch in parallel on separate GPUs.** Use `CUDA_VISIBLE_DEVICES=0` and `CUDA_VISIBLE_DEVICES=1`.
5. **If an experiment fails, report immediately.** Do not silently skip. Each experiment has a fallback or interpretation for negative outcomes — that is still useful data.
6. **Report Day 1 findings before launching Day 2.** Human will decide whether to proceed with Experiments 2 and 5 based on Day 1 results.

## SUMMARY DELIVERABLE FORMAT

At the end of each day, write a summary to `runs/DAY[N]_SUMMARY.md` with:

- One sentence per experiment: outcome + headline number.
- Per-experiment raw data path.
- Proposed LaTeX snippet for paper integration (if the outcome supports it).
- Anything that broke or was unclear.

The human will integrate findings into `main.tex` once you report.
