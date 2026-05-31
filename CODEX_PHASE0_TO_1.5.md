# Codex Task Bundle — Phase 0 to Phase 1.75 (Diagnostic-First Sprint)

## Mission

Execute the first 4 weeks of the post-submission roadmap (`plan.md` v2). Goal: in ~36 GPU-h of work, produce four diagnostic numbers (D1, D2, D3, D4) that determine which of three tracks (A: method paper, B: architectural redesign, C: fundamental-limit paper) to commit the rest of the summer to.

**Do not run any heavy training in this bundle.** Heavy training comes after the decision gate (Phase 1.75). The whole point of this sprint is to *prevent* premature commitment to a doomed track.

## Output everyone is waiting for

By the end of this bundle, you should produce one file: `runs/phase1_decision/decision_summary.md` containing:

```
D1 (T1 selector consistency Jaccard, Final vs Seed 7): ___
D2 (T6.5 LoRA-on-random ACC − LoRA-on-TPN ACC, avg over VQAv2/GQA/POPE): ___
D3 (T_new1 max |Spearman| of TPN scores with any prior): ___
D4 (T_new2 R² of probe predicting causal importance from encoder features): ___

Track recommendation: A / B / C
Justification: (one paragraph mapping the four numbers to the §10 decision tree in plan.md)
```

That file is the single deliverable. Everything else exists to produce it.

## Execution order

```
Week 1: Infrastructure (Phase 0 + 0.5)
   ├─ P0.1 Causal-intervention CLI                     [no GPU]
   ├─ P0.2 Synthetic-task data generators              [no GPU]
   ├─ P0.3 Multi-seed launcher                         [no GPU]
   └─ P0.4 FLOP counter + latency benchmark            [~2 GPU-h]

Week 2: Diagnostics on existing checkpoints (Phase 1)
   ├─ P1.1 T1 selector consistency (Final vs Seed 7)   [~1 GPU-h]
   ├─ P1.2 T2 selection diversity per query            [~2 GPU-h]
   ├─ P1.3 T3 oracle vs TPN Jaccard                    [~3 GPU-h]
   ├─ P1.4 T4 token coverage map                       [~1 GPU-h]
   ├─ P1.5 T6.5 LoRA-on-random vs LoRA-on-TPN          [~8 GPU-h]  ← largest
   ├─ P1.6 T8 merge information flow (CKA)             [~1 GPU-h]
   ├─ P1.7 T10 counterfactual selection                [~2 GPU-h]
   └─ P1.8 T11 failure clustering                      [~1 GPU-h]

Week 2-3: What did TPN learn (Phase 1.5)
   ├─ P1.5.1 T_new1 multi-prior correlation            [~4 GPU-h]
   └─ P1.5.2 T_new2 causal importance + probe          [~8 GPU-h]  ← novel

Week 3-4: Decision (Phase 1.75)
   └─ P1.75 Decision-gate analysis                     [no GPU]
```

Total: ~33 GPU-h. Fits in 3-4 days on 2 GPUs with parallelism.

## Critical instructions

1. **No new architectures.** This bundle is pure diagnostics on existing checkpoints. No retraining the TPN.
2. **Save every intermediate result.** All raw outputs go to `runs/phaseN_*/...`. Do not throw away data.
3. **Sanity check first.** Each task has a 200-sample smoke test. Run it before the full sweep.
4. **Report blocking issues immediately.** If a task can't run (missing checkpoint, code path broken), stop and report. Do not silently skip.
5. **Do not modify `main.tex`.** Already-submitted paper.
6. **Do not commit to a track before P1.75 analysis.** That is the entire point of this sprint.

## Repo facts to assume

- Primary checkpoint: `runs/phase1_final/phase1/best.ckpt` (k=144 fixed, single seed).
- Second seed checkpoint: `runs/phase1_seed7/phase1/checkpoint_step_80000.pt`.
- Adaptive checkpoint: `runs/phase1_adaptive_v2/phase1/best.ckpt` (variable k).
- Eval entry: `scripts/evaluate.py` (supports baselines: `none`, `random_drop`, `image_only`, `full_576`, `merge_disabled`, `uniform_grid`, `cls_attention`).
- Model code: `src/tc_vitok/models/tc_vitok_model.py` (class `TCViTokModel`, method `_select_tokens(images_coarse, images_full, questions)`).
- Encoders: `src/tc_vitok/models/encoders.py` (class `FullVisualEncoder` wrapping `transformers` CLIP).
- Datasets: `src/tc_vitok/data/datasets.py` (`VQAv2Dataset`, `GQADataset`, `POPEDataset`, `TextVQADataset`, `MMBenchDataset`, `MMEDataset`).
- Metrics: `src/tc_vitok/evaluation/metrics.py` (`vqa_score`, `gqa_accuracy`, `pope_f1`).

If any of these have moved or been renamed, search before assuming. Run `python scripts/evaluate.py --help` to confirm CLI.

---

# PHASE 0 — Infrastructure (Week 1)

## P0.1 — Causal-intervention CLI flags on evaluate.py

### Why
Section §3.5 of `plan.md` calls out causal interventions as required research instruments. Several Phase 1 tests depend on them.

### What to add to `scripts/evaluate.py`

Add three new CLI flags (do not remove anything existing):

```python
parser.add_argument(
    "--force-selection-from", default=None,
    help="Path to .pt file containing per-image selection masks [num_images, 576]. "
         "If set, TPN scores are overridden with these masks."
)
parser.add_argument(
    "--force-query-zero", action="store_true",
    help="Replace the CLIP question embedding with zeros before TPN runs."
)
parser.add_argument(
    "--save-scores-to", default=None,
    help="If set, save per-image TPN scores [B, 576] to this .pt file during eval."
)
parser.add_argument(
    "--save-selection-to", default=None,
    help="If set, save per-image top-k indices [B, k] to this .pt file during eval."
)
parser.add_argument(
    "--save-attentions-from-clip", default=None,
    help="If set, also save per-image CLIP last-layer CLS→patches attention [B, 576] to this .pt file."
)
```

Implement these as monkey-patches on `TCViTokModel`:

```python
# in scripts/evaluate.py, after load_model()
if args.force_query_zero:
    original_select = model._select_tokens
    def _wrapped(images_coarse, images_full, questions):
        # Pass zero strings or modify in TPN forward to zero the query embedding
        # Simplest: override the question embedding tensor at TPN call site
        return original_select(images_coarse, images_full, [""] * len(questions))
    model._select_tokens = _wrapped
```

For `--save-scores-to` and `--save-selection-to`, attach hooks to the TPN's output of `s_N` (the score tensor) and `idx` (top-k indices). Concatenate across batches; save at end.

For `--save-attentions-from-clip`, run CLIP vision forward once per image with `output_attentions=True`, extract last-layer CLS→patches attention, average across heads. Save concatenated.

### Sanity check

```bash
DATA_ROOT=~/tc-vitok-data python scripts/evaluate.py \
  --checkpoint runs/phase1_final/phase1/best.ckpt \
  --baseline none \
  --benchmarks gqa --max-samples 50 \
  --save-scores-to runs/p0_sanity/scores.pt \
  --save-selection-to runs/p0_sanity/selection.pt \
  --save-attentions-from-clip runs/p0_sanity/cls_attn.pt \
  --output runs/p0_sanity/eval.json
```

Verify the three .pt files exist and have correct shapes:
- `scores.pt`: tensor of shape `[50, 576]`, dtype float.
- `selection.pt`: tensor of shape `[50, 144]`, dtype long, values in `[0, 576)`.
- `cls_attn.pt`: tensor of shape `[50, 576]`, dtype float, rows sum near 1 (it's a softmax distribution).

If any shape is wrong, fix before moving on.

### Deliverable

Confirmation in `runs/p0_sanity/PASS` (empty file) that all three flags work on a 50-sample subset.

---

## P0.2 — Synthetic-task data generators

### Why
§3.3 of plan.md: every new architecture must pass synthetic gates before earning a real benchmark run. We need the gates first.

### What to build

Create `src/tc_vitok/synthetic/`:

```
src/tc_vitok/synthetic/
├── __init__.py
├── needle.py          # single-shape-at-known-location
├── two_needle.py      # two-shape-distractor
├── dense_text.py      # rendered text at known positions
└── eval.py            # wrappers that evaluate a TCViTokModel on each task
```

### Needle generator (`needle.py`)

```python
"""Needle-in-the-image synthetic task.

Generates 336x336 images with one colored shape at a known patch position.
The image is otherwise Gaussian noise. The 'right' answer is the shape color.
The 'right' patch to select is the one containing the shape center.

This task exists to test whether a selector can pick a specific patch
when the query specifies its location.
"""
from __future__ import annotations
import math
import random
from typing import NamedTuple

import numpy as np
import torch
from PIL import Image, ImageDraw

COLORS = {
    "red":    (220, 30, 30),
    "green":  (40, 180, 40),
    "blue":   (40, 60, 220),
    "yellow": (220, 200, 30),
    "purple": (160, 30, 200),
    "orange": (240, 140, 30),
}

class NeedleSample(NamedTuple):
    image: Image.Image
    question: str
    answer: str
    target_patch_idx: int   # index in [0, 576) of the patch holding the shape


def gen_sample(rng: random.Random, image_size: int = 336, patch_size: int = 14):
    n_patches_per_side = image_size // patch_size           # 24
    n_patches_total = n_patches_per_side * n_patches_per_side   # 576

    # Background: low-contrast noise
    arr = (np.random.RandomState(rng.randint(0, 2**31 - 1))
           .randint(120, 140, (image_size, image_size, 3), dtype=np.uint8))
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)

    # Pick a random patch position
    row = rng.randint(0, n_patches_per_side - 1)
    col = rng.randint(0, n_patches_per_side - 1)
    patch_idx = row * n_patches_per_side + col

    # Pick a color
    color_name = rng.choice(list(COLORS.keys()))
    color = COLORS[color_name]

    # Draw the shape (circle) centered on that patch
    cx = col * patch_size + patch_size // 2
    cy = row * patch_size + patch_size // 2
    r = patch_size // 2 - 1
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)

    # Build question / answer
    # Two question variants: with location, without location.
    if rng.random() < 0.5:
        q = f"What color is the shape at row {row}, column {col}?"
    else:
        q = "What color is the shape in the image?"
    a = color_name

    return NeedleSample(image=img, question=q, answer=a, target_patch_idx=patch_idx)


def build_dataset(n: int, seed: int = 7):
    rng = random.Random(seed)
    return [gen_sample(rng) for _ in range(n)]
```

### Two-needle generator (`two_needle.py`)

Similar but with two shapes. Question always asks about one of them by location.

### Dense-text generator (`dense_text.py`)

Use PIL's text drawing. Place 5-10 short words/numbers at random patches. Question asks "what is written at row r, column c?".

### Eval wrapper (`eval.py`)

```python
"""Evaluate a TCViTokModel on synthetic tasks.

For each task, we measure:
- Answer accuracy (whether the model gets the right color/text)
- Target-patch selection rate (fraction of times the right patch was in the selection)
"""
import torch
from tc_vitok.models.tc_vitok_model import TCViTokModel
from .needle import build_dataset as build_needle
from .two_needle import build_dataset as build_two_needle
from .dense_text import build_dataset as build_dense_text


@torch.no_grad()
def eval_synthetic(model: TCViTokModel, task: str, n: int = 500, k: int = 144):
    if task == "needle":
        ds = build_needle(n)
    elif task == "two_needle":
        ds = build_two_needle(n)
    elif task == "dense_text":
        ds = build_dense_text(n)
    else:
        raise ValueError(task)

    n_answer_correct = 0
    n_patch_in_selection = 0

    for sample in ds:
        # Run model.generate or equivalent on (image, question)
        # capture pred answer
        # also capture which patches were selected
        # the model needs --save-selection-to functionality or equivalent

        # Pseudocode:
        pred = model.generate(sample.image, sample.question)
        selected_idx = model.last_selection  # [144], assumed exposed

        if pred.lower().strip() == sample.answer.lower().strip():
            n_answer_correct += 1
        if sample.target_patch_idx in selected_idx.tolist():
            n_patch_in_selection += 1

    return {
        "task": task,
        "n": n,
        "answer_accuracy": n_answer_correct / n,
        "target_patch_selection_rate": n_patch_in_selection / n,
    }
```

### Sanity check

Run all three on TC-ViTok Final with n=100. Expected (based on what we know):

- **needle answer accuracy:** > 60% (with location query, the LLM can usually identify the color even without good selection).
- **needle target_patch_selection_rate:** This is the key number. If TPN is genuinely capable of routing on the query, > 70%. If TPN ignores the query, ~ 24% (random would get 144/576 = 25%).

Save outputs to `runs/p0_synthetic/sanity_results.json`. **Report the target_patch_selection_rate prominently** — this is the cleanest single test of "does TPN actually use the query?"

### Deliverable

`runs/p0_synthetic/sanity_results.json` with three rows (needle, two_needle, dense_text), each containing answer_accuracy and target_patch_selection_rate.

---

## P0.3 — Multi-seed launcher

### Why
Every Phase 6 headline number requires N=3 seeds. Manual launching is error-prone.

### What to build

`scripts/launch_multiseed.sh`:

```bash
#!/usr/bin/env bash
# Usage: ./scripts/launch_multiseed.sh CONFIG_YAML OUTPUT_DIR N_SEEDS
set -euo pipefail
CONFIG=$1
OUT=$2
N=${3:-3}

mkdir -p $OUT
for seed in $(seq 0 $((N-1))); do
  ACTUAL_SEED=$((42 + 100 * seed))
  echo "=== Launching seed $ACTUAL_SEED ==="
  CUDA_VISIBLE_DEVICES=$seed DATA_ROOT=~/tc-vitok-data PYTHONPATH=src \
    python scripts/train.py \
    --config $CONFIG \
    --seed $ACTUAL_SEED \
    --output $OUT/seed$ACTUAL_SEED \
    2>&1 | tee $OUT/seed$ACTUAL_SEED.log &
done
wait
echo "All seeds done."
```

### Aggregation script

`scripts/aggregate_multiseed.py`:

```python
"""Aggregate per-seed JSON results into mean ± std per benchmark."""
import json
import glob
import statistics
import sys
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python scripts/aggregate_multiseed.py RUN_DIR"); sys.exit(1)

run_dir = Path(sys.argv[1])
files = sorted(run_dir.glob("seed*/eval.json"))

if not files:
    print(f"No eval.json files found under {run_dir}/seed*/eval.json"); sys.exit(1)

per_bench = {}
for f in files:
    data = json.load(open(f))
    for bench, score in data.items():
        per_bench.setdefault(bench, []).append(float(score) if not isinstance(score, dict) else float(score.get("accuracy", 0.0)))

print(f"Aggregated over {len(files)} seeds")
print(f"{'Benchmark':<12} {'Mean':>8} {'Std':>8} {'N':>3}")
for bench, vals in per_bench.items():
    mu = statistics.mean(vals)
    sd = statistics.stdev(vals) if len(vals) >= 2 else 0.0
    print(f"{bench:<12} {mu:>8.3f} {sd:>8.3f} {len(vals):>3}")
```

### Deliverable

Both scripts executable. Smoke-test by launching N=2 short eval runs (5K-sample) in parallel; confirm the aggregation script combines them correctly.

---

## P0.4 — FLOP counter + latency benchmark

### Why
Section 6.4.1 of plan.md says "win on FLOPs at matched accuracy." We cannot make that claim without measurement infrastructure.

### What to build

`scripts/measure_compute.py`:

```python
"""Measure FLOPs and wall-clock latency for TC-ViTok configurations."""
import argparse
import time

import torch
import calflops  # pip install calflops
from PIL import Image

from scripts.evaluate import load_model


def measure_flops(model, image_tensor, question="What is in the image?"):
    """Forward pass with calflops."""
    from calflops import calculate_flops
    # may need to wrap model.forward to accept batched (image, question)
    flops, macs, params = calculate_flops(
        model=model.forward,
        kwargs={"images_full": image_tensor.unsqueeze(0), "questions": [question]},
        print_results=False,
    )
    return {"flops": flops, "macs": macs, "params": params}


def measure_latency(model, image_tensor, question, n_warm=10, n_trial=100):
    """Wall-clock latency in milliseconds."""
    device = next(model.parameters()).device
    for _ in range(n_warm):
        _ = model.generate(image_tensor.to(device), question)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_trial):
        _ = model.generate(image_tensor.to(device), question)
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return {"ms_per_image": 1000 * (t1 - t0) / n_trial, "n_trial": n_trial}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--baselines", nargs="+",
                   default=["none", "random_drop", "uniform_grid", "cls_attention"])
    p.add_argument("--k", type=int, default=144)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    # Use a fixed test image
    image = Image.open("path/to/test_image.jpg").convert("RGB")
    # ... preprocess to tensor

    results = {}
    for baseline in args.baselines:
        model = load_model(checkpoint=args.checkpoint, baseline=baseline, k=args.k)
        model.eval().cuda()
        results[baseline] = {
            **measure_flops(model, image_tensor),
            **measure_latency(model, image_tensor, "What is in the image?"),
        }
        del model
        torch.cuda.empty_cache()

    import json
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
```

### Sanity check

Run on TC-ViTok Final and CLS-attention at k=144. Compare FLOPs and ms/image. Sanity expectations:

- CLS-attention should have slightly higher visual-encoder FLOPs (extra attention extraction).
- TC-ViTok should have higher selector FLOPs (TPN cross-attention).
- LLM FLOPs should dominate everything (~95% of total).
- ms/image should be in the 100-500ms range on RTX 6000.

If LLM FLOPs are not dominant, the FLOP counter is undercounting somewhere — fix before relying on it.

### Deliverable

`runs/p0_compute/baseline_compute.json` with (FLOPs, MACs, params, ms_per_image) for each baseline at k=144.

---

# PHASE 1 — Diagnostics on existing checkpoints (Week 2)

All Phase 1 tasks use the existing `phase1_final` and `phase1_seed7` checkpoints. No retraining.

## P1.1 — T1: Selector consistency under seed

### Goal
Measure Jaccard similarity of TPN selections between seed-42 (Final) and seed-7 (Seed7) checkpoints on the same images. Quantifies whether the TPN is a stable solution or near-random.

### Run

Save selection from both checkpoints on the same 500 GQA samples, then compute per-image Jaccard.

```bash
mkdir -p runs/p1_t1

DATA_ROOT=~/tc-vitok-data python scripts/evaluate.py \
  --checkpoint runs/phase1_final/phase1/best.ckpt \
  --baseline none \
  --benchmarks gqa --max-samples 500 \
  --save-selection-to runs/p1_t1/selection_final.pt \
  --output runs/p1_t1/eval_final.json

DATA_ROOT=~/tc-vitok-data python scripts/evaluate.py \
  --checkpoint runs/phase1_seed7/phase1/checkpoint_step_80000.pt \
  --baseline none \
  --benchmarks gqa --max-samples 500 \
  --save-selection-to runs/p1_t1/selection_seed7.pt \
  --output runs/p1_t1/eval_seed7.json
```

### Analysis script `scripts/analyze_t1.py`

```python
"""Compute per-image Jaccard between two selection tensors."""
import torch, json, sys

sel_a = torch.load(sys.argv[1])   # [N, k] long
sel_b = torch.load(sys.argv[2])   # [N, k] long
assert sel_a.shape == sel_b.shape, f"shape mismatch: {sel_a.shape} vs {sel_b.shape}"

jaccards = []
for i in range(sel_a.shape[0]):
    a, b = set(sel_a[i].tolist()), set(sel_b[i].tolist())
    j = len(a & b) / len(a | b)
    jaccards.append(j)

jaccards = torch.tensor(jaccards)
result = {
    "n_images": len(jaccards),
    "mean_jaccard": float(jaccards.mean()),
    "median_jaccard": float(jaccards.median()),
    "std": float(jaccards.std()),
    "min": float(jaccards.min()),
    "max": float(jaccards.max()),
}
print(json.dumps(result, indent=2))
json.dump(result, open(sys.argv[3], "w"), indent=2)
```

```bash
python scripts/analyze_t1.py \
  runs/p1_t1/selection_final.pt \
  runs/p1_t1/selection_seed7.pt \
  runs/p1_t1/jaccard_result.json
```

### Interpretation
- Jaccard ≥ 0.7: TPN converges to a stable solution across seeds → selection has structure.
- 0.3 ≤ Jaccard < 0.7: partial structure → moderate consistency.
- Jaccard < 0.3: TPN selection is approximately random across seeds → no consistent solution.

**This is D1.** Save to `runs/phase1_decision/D1.txt` as a single float.

### Deliverable
`runs/p1_t1/jaccard_result.json` and `runs/phase1_decision/D1.txt`.

---

## P1.2 — T2: Selection diversity per query

### Goal
For the same image with multiple different questions, measure how much TPN's selection varies. High variance = uses the query. Low variance = query-agnostic.

### Approach
For 100 GQA images, take the existing question plus 4 alternative questions (e.g., generic "what is in the image?", "describe the colors", "count the objects", "what is on the left?"). Run TPN selection 5 times per image with different questions.

### New script `scripts/run_t2_query_diversity.py`

```python
"""For each image, run TPN selection 5 times with 5 different questions.
Measure per-image KL divergence between the resulting selection distributions.
"""
import argparse, json
from pathlib import Path
import torch
from torch.nn.functional import kl_div

from scripts.evaluate import load_model

ALT_QUESTIONS = [
    "What is in the image?",
    "Describe the main object.",
    "What colors are present?",
    "How many objects are there?",
    "What is on the left side?",
]

@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", default="~/tc-vitok-data")
    p.add_argument("--n-images", type=int, default=100)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    model = load_model(checkpoint=args.checkpoint, baseline="none", k=144).eval().cuda()
    # Use GQA val loader to get 100 images
    from tc_vitok.data.datasets import GQADataset, collate_fn
    from torch.utils.data import DataLoader
    ds = GQADataset(args.data_root, split="val", max_samples=args.n_images)
    dl = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_fn)

    per_image_diversities = []
    for batch in dl:
        images_full = batch["images_full"].cuda()
        images_coarse = batch["images_coarse"].cuda()
        # Run TPN 5 times with different questions
        scores_list = []
        for q in ALT_QUESTIONS:
            questions = [q] * images_full.shape[0]
            # Get scores (not just top-k). Hook into the TPN to grab s_N.
            # ... implementation specific to TCViTokModel internals
            _, s_N = model._compute_tpn_scores_only(images_coarse, images_full, questions)
            # s_N shape: [B, 576]
            scores_list.append(torch.softmax(s_N, dim=-1).cpu())
        # Per-image KL divergence between the 5 question variants
        # Average pairwise KL divergence: 1/10 * sum_{i<j} KL(p_i || p_j) + KL(p_j || p_i)) / 2
        B = scores_list[0].shape[0]
        for b in range(B):
            ps = [s[b] for s in scores_list]
            pairwise = []
            for i in range(5):
                for j in range(i+1, 5):
                    kl_ij = (ps[i] * (ps[i].log() - ps[j].log())).sum()
                    kl_ji = (ps[j] * (ps[j].log() - ps[i].log())).sum()
                    pairwise.append(float((kl_ij + kl_ji) / 2))
            per_image_diversities.append(sum(pairwise) / len(pairwise))

    result = {
        "n_images": len(per_image_diversities),
        "n_questions_per_image": 5,
        "mean_per_image_kl": float(torch.tensor(per_image_diversities).mean()),
        "median_per_image_kl": float(torch.tensor(per_image_diversities).median()),
        "std": float(torch.tensor(per_image_diversities).std()),
    }
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
```

**Note:** the `_compute_tpn_scores_only` method may not exist. If not, add it to `TCViTokModel`:

```python
@torch.no_grad()
def _compute_tpn_scores_only(self, images_coarse, images_full, questions):
    v_coarse = self.coarse_encoder(images_coarse)
    q_embed = self.text_encoder(questions)
    # Run TPN forward, return idx and s_N before top-k
    idx, s_N, _, k_eff = self.tpn(
        v_coarse=v_coarse, q_embed=q_embed, k=self.k,
        tau=0.1, training=False,
    )
    return idx, s_N
```

### Interpretation
- Mean per-image KL ≥ 1.0: TPN substantially varies selection across questions → uses query.
- Mean per-image KL 0.1-1.0: partial query usage.
- Mean per-image KL < 0.1: TPN is approximately query-agnostic.

### Deliverable
`runs/p1_t2/diversity_result.json`. Report mean KL prominently.

---

## P1.3 — T3: Oracle vs TPN selection Jaccard

### Goal
Compare TPN's selection to an "oracle" selection derived from FastV-style attention. Quantifies how close TPN is to what the LLM actually uses.

### Approach
For each image, run a *teacher* forward pass — the LoRA-576 baseline (no compression). Capture the LLM's layer-2 attention from text tokens to visual tokens. Average across heads and text positions. Top-144 of these scores = oracle.

### New script `scripts/build_fastv_oracle.py`

```python
"""Build FastV-style oracle: top-144 visual tokens by LLM layer-2 attention.

For each image:
1. Forward through LoRA-576 (no compression).
2. Hook LLM layer 2 attention.
3. Extract text→visual attention rows (last few text tokens).
4. Average across heads and text positions.
5. Top-144 indices = oracle selection.
"""
import argparse, json
from pathlib import Path
import torch
from scripts.evaluate import load_model

# Hook to capture layer-2 attention
captured_attn = {}

def hook_factory(name):
    def hook(module, inp, out):
        # out structure depends on the LLM implementation
        if isinstance(out, tuple) and len(out) >= 2 and out[1] is not None:
            captured_attn[name] = out[1].detach()  # [B, heads, seq, seq]
    return hook


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", default="~/tc-vitok-data")
    p.add_argument("--n-images", type=int, default=500)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    # Load LoRA-576 (the no-compression baseline)
    model = load_model(checkpoint=args.checkpoint, baseline="full_576", k=576).eval().cuda()
    # Find the layer-2 attention module of the LLaVA LLM
    # Path: model.llava.language_model.model.layers[2].self_attn
    llm_layer2_attn = (
        model.llava.language_model.model.layers[2].self_attn
        if hasattr(model.llava, "language_model")
        else model.llava.model.layers[2].self_attn
    )
    handle = llm_layer2_attn.register_forward_hook(hook_factory("L2"))

    # Loop over GQA val 500 samples
    from tc_vitok.data.datasets import GQADataset, collate_fn
    from torch.utils.data import DataLoader
    ds = GQADataset(args.data_root, split="val", max_samples=args.n_images)
    dl = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate_fn)

    oracle_indices = []
    image_ids = []
    for batch in dl:
        # Forward; hook captures attention
        _ = model.forward(
            images_coarse=batch["images_coarse"].cuda(),
            images_full=batch["images_full"].cuda(),
            questions=batch["questions"],
            max_new_tokens=1,  # just need one forward step
        )
        attn = captured_attn["L2"]   # [B, heads, seq, seq]
        # Identify visual-token positions: typically positions [vis_start, vis_start+576)
        # depending on LLaVA tokenization. Verify by inspecting one batch.
        # Heuristic: visual tokens are the first 576 non-text positions.
        B, H, seq, _ = attn.shape
        # Text tokens query → visual tokens
        # For now, assume visual tokens are at positions [1, 577) in the input sequence
        # (token 0 is BOS, then 576 visual, then text). Adjust based on actual layout.
        vis_start, vis_end = 1, 577
        # text positions: anything from vis_end onward
        text_positions = list(range(vis_end, seq))
        # Mean attention from text tokens to each visual token
        text_to_visual = attn[:, :, text_positions, vis_start:vis_end]   # [B, H, T, 576]
        avg = text_to_visual.mean(dim=(1, 2))    # [B, 576]
        top144 = avg.topk(144, dim=-1).indices   # [B, 144]
        oracle_indices.append(top144.cpu())

    handle.remove()
    oracle = torch.cat(oracle_indices, dim=0)   # [N, 144]
    torch.save(oracle, args.output)
    print(f"Saved oracle to {args.output}, shape={oracle.shape}")

if __name__ == "__main__":
    main()
```

### Now compute Jaccard against TPN selection

The TPN selection from P1.1 is already saved as `runs/p1_t1/selection_final.pt`. Reuse it.

```bash
mkdir -p runs/p1_t3

# Build oracle (this requires LoRA-576 forward, expect ~3 GPU-h)
DATA_ROOT=~/tc-vitok-data python scripts/build_fastv_oracle.py \
  --checkpoint runs/phase1_final/phase1/best.ckpt \
  --n-images 500 \
  --output runs/p1_t3/oracle.pt

# Compute Jaccard
python scripts/analyze_t1.py \
  runs/p1_t1/selection_final.pt \
  runs/p1_t3/oracle.pt \
  runs/p1_t3/oracle_vs_tpn_jaccard.json
```

### Sanity check
**Important:** verify the visual-token positions in the LLaVA layer-2 attention before running on 500 samples. Run with `--n-images 5` first and print the attention shape; confirm that the visual-token region is correctly identified. If LLaVA's tokenization differs from the assumption above, fix the indexing.

### Interpretation
- Jaccard ≥ 0.5: TPN is broadly consistent with what the LLM actually attends to → fixable via supervision.
- Jaccard 0.2-0.5: partial overlap → mixed signal.
- Jaccard < 0.2: TPN is essentially uncorrelated with LLM's actual usage → strong H_arch.

### Deliverable
`runs/p1_t3/oracle_vs_tpn_jaccard.json`.

---

## P1.4 — T4: Token coverage map

### Goal
Per-position selection frequency map. Dead zones, hot spots, mass-balance.

### Run

Already have `runs/p1_t1/selection_final.pt` (5K-sample selection). Just analyze.

`scripts/analyze_t4_coverage.py`:

```python
"""Position-selection-frequency map and dead-zone metrics."""
import torch, json, sys
import numpy as np

selection = torch.load(sys.argv[1])   # [N, k] long
n, k = selection.shape

# Per-position selection frequency
counts = torch.zeros(576, dtype=torch.long)
for i in range(n):
    counts.scatter_add_(0, selection[i], torch.ones(k, dtype=torch.long))
freq = counts.float() / n   # [576]

result = {
    "n_images": n,
    "k": k,
    "expected_uniform_freq": k / 576,   # 0.25 for k=144
    "max_freq": float(freq.max()),
    "min_freq": float(freq.min()),
    "median_freq": float(freq.median()),
    "mass_balance": float(freq.max() - freq.min()),
    "dead_zone_count": int((freq < 0.02).sum()),
    "hot_spot_count": int((freq > 0.5).sum()),
    "freq_per_position": freq.tolist(),
}
print(f"max: {result['max_freq']:.3f}, min: {result['min_freq']:.3f}, "
      f"dead_zones (<0.02): {result['dead_zone_count']}, "
      f"hot_spots (>0.5): {result['hot_spot_count']}")
json.dump(result, open(sys.argv[2], "w"), indent=2)
```

### Deliverable
`runs/p1_t4/coverage_stats.json`. Generate a 24×24 heatmap PDF.

---

## P1.5 — T6.5: LoRA-on-random vs LoRA-on-TPN (largest single task)

### Goal
The most important test in the diagnostic battery. Train a LoRA from scratch with two different input distributions:

(a) **LoRA-on-random:** Selection is always uniform random at k=144. Train LoRA to compensate.
(b) **LoRA-on-TPN-frozen:** Selection is the trained TPN's output (frozen at the Phase-1-Final checkpoint). Train LoRA to compensate.

If accuracy is similar, LoRA absorbs selection responsibility and **the selector is decorative**. If LoRA-on-TPN > LoRA-on-random by ≥ 2 pp, the selector is genuinely contributing. If LoRA-on-random > LoRA-on-TPN, the selector is actively harmful.

### Approach

Reuse the existing training pipeline but constrain the selection.

**Step 1:** Pre-compute random selections (with a fixed seed) for the entire training set. Save as a `.pt` file: shape `[N_train, 144]`, dtype long. This makes "random" reproducible and lets the training loop just index into the pre-computed selections.

**Step 2:** Pre-compute TPN-frozen selections similarly using the Phase-1-Final TPN.

**Step 3:** Modify `scripts/train.py` to accept `--force-selection-from <path>` and use the pre-computed selections instead of computing on-the-fly.

**Step 4:** Run two 20K-step training runs (Phase 1 length, smaller than full to save compute):
- Run A: `--force-selection-from random.pt`
- Run B: `--force-selection-from tpn_frozen.pt`

Both with identical hyperparameters. Same seed (e.g., 42).

**Step 5:** Evaluate both on VQAv2/GQA/POPE 5K subsets.

### Run

```bash
mkdir -p runs/p1_t65

# Step 1+2: pre-compute selections on training data
DATA_ROOT=~/tc-vitok-data python scripts/precompute_selections.py \
  --mode random --seed 42 \
  --output runs/p1_t65/random_selections.pt

DATA_ROOT=~/tc-vitok-data python scripts/precompute_selections.py \
  --mode tpn-frozen \
  --tpn-checkpoint runs/phase1_final/phase1/best.ckpt \
  --output runs/p1_t65/tpn_frozen_selections.pt

# Step 3+4: train two LoRAs in parallel
tmux new -s lora_random
CUDA_VISIBLE_DEVICES=0 DATA_ROOT=~/tc-vitok-data python scripts/train.py \
  --config configs/phase1_lora_only.yaml \
  --force-selection-from runs/p1_t65/random_selections.pt \
  --max-steps 20000 \
  --output runs/p1_t65/lora_on_random \
  2>&1 | tee runs/p1_t65/lora_on_random/train.log
# detach, then in second tmux:

tmux new -s lora_tpn
CUDA_VISIBLE_DEVICES=1 DATA_ROOT=~/tc-vitok-data python scripts/train.py \
  --config configs/phase1_lora_only.yaml \
  --force-selection-from runs/p1_t65/tpn_frozen_selections.pt \
  --max-steps 20000 \
  --output runs/p1_t65/lora_on_tpn_frozen \
  2>&1 | tee runs/p1_t65/lora_on_tpn_frozen/train.log

# Step 5: evaluate both
DATA_ROOT=~/tc-vitok-data python scripts/evaluate.py \
  --checkpoint runs/p1_t65/lora_on_random/phase1/best.ckpt \
  --baseline none \
  --benchmarks vqav2 gqa pope \
  --max-samples 5000 \
  --output runs/p1_t65/eval_lora_on_random.json

DATA_ROOT=~/tc-vitok-data python scripts/evaluate.py \
  --checkpoint runs/p1_t65/lora_on_tpn_frozen/phase1/best.ckpt \
  --baseline none \
  --benchmarks vqav2 gqa pope \
  --max-samples 5000 \
  --output runs/p1_t65/eval_lora_on_tpn_frozen.json
```

### New code

`scripts/precompute_selections.py`:

```python
"""Pre-compute per-image selection masks for the training set.

Mode:
- random: uniform random top-144 from 576, fixed seed.
- tpn-frozen: load a checkpoint and run TPN to get selections; cache.
"""
import argparse, torch
from pathlib import Path

@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["random", "tpn-frozen"], required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tpn-checkpoint", default=None)
    p.add_argument("--output", required=True)
    p.add_argument("--data-root", default="~/tc-vitok-data")
    args = p.parse_args()

    # Compute number of training samples (e.g., 500K for VQAv2+GQA+Instruct)
    # For each training image, generate or compute selection
    # Save as one big tensor indexed by image_id

    if args.mode == "random":
        g = torch.Generator().manual_seed(args.seed)
        # For each image_id in training set, generate top-144 random indices
        # ... requires knowing the number of training samples
        n_train = 500_000   # adjust to actual
        sel = torch.stack([
            torch.randperm(576, generator=g)[:144]
            for _ in range(n_train)
        ])
        torch.save(sel, args.output)
    elif args.mode == "tpn-frozen":
        # Load model with the given TPN checkpoint, run on training data, save selections
        # Compute may take a few hours
        raise NotImplementedError("Implement once verified on smoke test")

if __name__ == "__main__":
    main()
```

Modify `scripts/train.py` to accept `--force-selection-from` and override the selection step.

`configs/phase1_lora_only.yaml`: existing phase1 config but with the TPN parameters frozen.

### Sanity check
Run a 200-step toy training with `--mode random` on a 1K-sample subset. Confirm:
- Loss decreases.
- Final eval on 200-sample GQA is in the 30-50% range (not random-classifier 1-2%).
- Selection masks logged from training match the pre-computed random masks.

If loss doesn't decrease at all, the `--force-selection-from` plumbing is broken.

### Compute estimate
~8 GPU-h (two 20K-step runs + evals).

### Interpretation
- **|LoRA-on-random ACC − LoRA-on-TPN ACC| < 1 pp:** selector is decorative; LoRA does the work.
- **LoRA-on-TPN beats by 2-5 pp:** selector contributes meaningfully.
- **LoRA-on-random beats LoRA-on-TPN by any margin:** selector is actively harmful (consistent with sprint findings).

**This is D2.** Save the average gap (LoRA-on-TPN − LoRA-on-random, averaged across the three benchmarks) to `runs/phase1_decision/D2.txt`.

### Deliverable
`runs/p1_t65/eval_lora_on_random.json`, `runs/p1_t65/eval_lora_on_tpn_frozen.json`, and `runs/phase1_decision/D2.txt`.

---

## P1.6 — T8: Merge information flow (CKA)

### Goal
Check whether the parameter-free merge transfers useful information by measuring CKA (Centered Kernel Alignment) between merged tokens and the unselected tokens they merged from.

### Approach
For 500 GQA samples:
1. Run TPN to get selected indices and unselected indices.
2. Compute `v_selected` (selected token features) and `v_merged` (after merge formula).
3. Compute CKA between `v_merged` and `v_unselected`.

High CKA = merge actually carries information. Low CKA = merge is decorative.

`scripts/analyze_t8_merge_cka.py`:

```python
"""CKA between merged tokens and unselected token features."""
import argparse, json
import torch
from scripts.evaluate import load_model

def cka(X, Y):
    """Centered Kernel Alignment between two [n, d] matrices."""
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    HSIC = (X.T @ Y).norm()**2
    return float(HSIC / ((X.T @ X).norm() * (Y.T @ Y).norm()))


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--n-images", type=int, default=500)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    model = load_model(checkpoint=args.checkpoint, baseline="none", k=144).eval().cuda()
    # ... iterate GQA val 500 samples
    # For each batch, compute v_full (576x1024), idx (selected), and v_merged
    # Then collect v_merged and v_unselected pairs

    ckas = []
    # ... pseudocode here; fill in based on model internals

    result = {
        "n_images": args.n_images,
        "mean_cka_merged_vs_unselected": float(torch.tensor(ckas).mean()),
        # baseline: CKA between v_selected and v_unselected (without merge)
    }
    json.dump(result, open(args.output, "w"), indent=2)
```

### Deliverable
`runs/p1_t8/cka_result.json`. Report mean CKA.

---

## P1.7 — T10: Counterfactual selection

### Goal
For each failed example (where TPN was wrong), check whether using a different selection rule would have answered correctly.

### Approach
Run TC-ViTok Final on GQA 1K subset. For each failed example, re-run with selections from:
- Random (5 seeds; report best)
- Uniform grid
- CLS-attention top-k
- FastV-oracle (from P1.3)

For each, conditional accuracy on the failed-only subset.

### Compute
~2 GPU-h.

### Deliverable
`runs/p1_t10/counterfactual_results.json` with conditional accuracies.

---

## P1.8 — T11: Failure clustering

### Goal
Cluster failed examples by image complexity, question type, answer length.

### Approach
Pure analysis on existing eval logs. Use simple heuristics:
- Image complexity: number of detected objects (use a frozen detector if available, else use image entropy).
- Question type: keyword-based ("what color", "how many", "is there", etc.).
- Answer length: word count.

### Compute
~1 GPU-h (mostly for the detector forward).

### Deliverable
`runs/p1_t11/failure_clusters.json`. Identify whether failures cluster by image complexity (compression-bottleneck) or by question type (query-conditioning).

---

# PHASE 1.5 — What did TPN learn (Week 2-3)

## P1.5.1 — T_new1: Multi-prior correlation

### Goal
For 500 GQA samples, compute Spearman rank correlation between TPN's per-position scores and *six* candidate priors:

1. CLIP last-layer CLS→patches attention.
2. FastV-style LLM layer-2 attention from text→visual.
3. CLIP penultimate-layer CLS attention.
4. Image gradient magnitude per patch (computed from raw image).
5. Pixel intensity (luminance) per patch.
6. Position bias (distance from image center).

The sprint already measured #1 (Spearman ≈ -0.03). This task extends to all six.

### Approach
Reuse the score-saving infrastructure from P0.1. Save TPN scores once. Compute each prior separately. Loop over 200 GQA images.

### New script `scripts/analyze_tnew1.py`

```python
"""Multi-prior correlation analysis for TPN scores.

For each image, compute Spearman rank correlation between TPN scores and
each of six candidate priors. Report mean, median, std per prior.
"""
import argparse, json
import torch
import numpy as np
from scipy.stats import spearmanr

# ... load TPN scores from runs/p1_t4/scores_final.pt or compute fresh
# ... load CLS attentions from P0.1 sanity check or compute fresh
# ... compute FastV attention (reuse runs/p1_t3/oracle_scores.pt, but as scores not top-k)
# ... compute image-gradient magnitude per patch from raw images
# ... compute pixel-intensity per patch
# ... compute distance-from-center per patch (one fixed prior for all images)

results = {}
for prior_name, prior_scores in {
    "cls_last_layer": cls_attn,
    "fastv_layer2":  fastv_attn,
    "cls_penultimate": cls_attn_penult,
    "image_gradient": grad_mag,
    "pixel_intensity": intensity,
    "center_distance": center_dist,
}.items():
    rhos = []
    for i in range(n_images):
        rho, _ = spearmanr(tpn_scores[i], prior_scores[i])
        rhos.append(float(rho))
    rhos = np.array(rhos)
    results[prior_name] = {
        "mean": float(rhos.mean()),
        "median": float(np.median(rhos)),
        "std": float(rhos.std()),
        "abs_mean": float(np.abs(rhos).mean()),
    }

# D3 is the max absolute mean Spearman across all priors
D3 = max(r["abs_mean"] for r in results.values())
results["D3"] = D3
print(json.dumps(results, indent=2))
```

### Compute
~4 GPU-h.

### Interpretation
- **Max |Spearman| ≥ 0.4 with any prior:** TPN learned something correlated with a known useful prior. Strong → Track A.1.
- **Max |Spearman| 0.2-0.4:** partial structure → Track A.2 / mixed.
- **Max |Spearman| < 0.2:** TPN learned nothing correlated with any obvious prior → Track B.

**This is D3.** Save the max |Spearman| across all six priors to `runs/phase1_decision/D3.txt`.

### Deliverable
`runs/p1_5_1/multi_prior_correlation.json` and `runs/phase1_decision/D3.txt`. Plus a small markdown table showing each prior's mean Spearman.

---

## P1.5.2 — T_new2: Causal token importance + probe

### Goal
Compute *causal* per-token importance: for each test image, ablate each visual token in turn from a full-context forward pass; measure ΔLLM-prediction. Then train a probe to predict this from encoder features.

If the probe's R² is high (≥ 0.5), causal importance is predictable from features → distillation has a meaningful ceiling → Track A.
If R² is low (< 0.2), causal importance is fundamentally unpredictable from features → pre-LLM compression is bounded → Track C.

### Approach

**Step 1:** For 200 GQA samples, compute per-token causal importance.

`scripts/compute_causal_importance.py`:

```python
"""Compute per-token causal importance via ablation.

For each (image, question) pair:
1. Run LoRA-576 (full-context) forward; record LLM prediction logits at the
   answer position (or top-token log-probability).
2. For each visual-token position t in [0, 576):
   - Ablate position t (zero it out, or mask it).
   - Re-run forward; record new logits / log-prob.
   - Causal importance[t] = original_logprob - ablated_logprob.

This is O(576 * N) forward passes per image. With batching this is doable
for 200 samples in ~8 GPU-h.
"""
import argparse, json
from pathlib import Path
import torch
from scripts.evaluate import load_model

@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--n-images", type=int, default=200)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    model = load_model(checkpoint=args.checkpoint, baseline="full_576", k=576).eval().cuda()
    # GQA val loader, 200 samples
    from tc_vitok.data.datasets import GQADataset, collate_fn
    from torch.utils.data import DataLoader
    ds = GQADataset(args.data_root, split="val", max_samples=args.n_images)

    importances = []
    encoder_features = []  # save for the probe stage

    for sample in ds:
        # Compute baseline logprob
        baseline_logprob = compute_answer_logprob(model, sample.image, sample.question, sample.answer)
        # Extract encoder features
        v_full = model.full_encoder(sample.image_tensor.unsqueeze(0).cuda())  # [1, 576, 1024]
        encoder_features.append(v_full[0].cpu())

        # Ablate each position in batches of 32
        per_position_delta = torch.zeros(576)
        # Batch 18 ablations at a time (576/32 = 18 batches)
        for batch_start in range(0, 576, 32):
            batch_end = min(batch_start + 32, 576)
            mask = torch.ones(batch_end - batch_start, 576)
            for j, pos in enumerate(range(batch_start, batch_end)):
                mask[j, pos] = 0
            # Run forward with the masked visual tokens
            # ... return per-batch-element logprobs
            ablated_logprobs = compute_answer_logprobs_batched(
                model, v_full * mask.unsqueeze(-1).cuda(), sample.question, sample.answer
            )
            per_position_delta[batch_start:batch_end] = baseline_logprob - ablated_logprobs
        importances.append(per_position_delta)

    importances = torch.stack(importances)        # [200, 576]
    features = torch.stack(encoder_features)      # [200, 576, 1024]

    torch.save({"importances": importances, "features": features}, args.output)
```

**Step 2:** Train a probe to predict causal importance from encoder features.

`scripts/train_causal_probe.py`:

```python
"""Train a probe to predict per-token causal importance from encoder features."""
import argparse, json
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, random_split

class LinearProbe(nn.Module):
    def __init__(self, d_in=1024):
        super().__init__()
        self.linear = nn.Linear(d_in, 1)
    def forward(self, x):
        return self.linear(x).squeeze(-1)

class MLP Probe(nn.Module):
    def __init__(self, d_in=1024, d_hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="From compute_causal_importance.py")
    p.add_argument("--probe", choices=["linear", "mlp"], default="mlp")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    data = torch.load(args.data)
    importances = data["importances"]   # [N, 576]
    features = data["features"]         # [N, 576, 1024]

    # Flatten across (N, 576) → samples of 1024-d features predicting one scalar
    N, P, D = features.shape
    X = features.reshape(N * P, D)      # [N*576, 1024]
    Y = importances.reshape(N * P)      # [N*576]

    # Train/val split per image (avoid leakage)
    n_train = int(0.8 * N)
    train_idx = torch.arange(0, n_train * P)
    val_idx = torch.arange(n_train * P, N * P)

    probe = LinearProbe() if args.probe == "linear" else MLPProbe()
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)

    for epoch in range(50):
        # ... train loop
        pass

    # Final R² on val set
    probe.eval()
    pred = probe(X[val_idx])
    target = Y[val_idx]
    ss_res = ((target - pred) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    r2 = float(1 - ss_res / ss_tot)

    result = {"probe": args.probe, "r2": r2, "n_train_images": n_train, "n_val_images": N - n_train}
    json.dump(result, open(args.output, "w"), indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
```

### Run

```bash
mkdir -p runs/p1_5_2

# Step 1: compute causal importance (8 GPU-h)
DATA_ROOT=~/tc-vitok-data python scripts/compute_causal_importance.py \
  --checkpoint runs/phase1_final/phase1/best.ckpt \
  --n-images 200 \
  --output runs/p1_5_2/causal_data.pt

# Step 2: train both probes (fast, < 30 min on GPU)
python scripts/train_causal_probe.py \
  --data runs/p1_5_2/causal_data.pt \
  --probe linear \
  --output runs/p1_5_2/probe_linear.json

python scripts/train_causal_probe.py \
  --data runs/p1_5_2/causal_data.pt \
  --probe mlp \
  --output runs/p1_5_2/probe_mlp.json
```

### Sanity check
Causal importance values should be non-zero and have variation. If all 576 positions have causal importance ≈ 0, either the ablation isn't actually changing the prediction (mask not applied correctly) or the LLM is robust to single-token ablation (unlikely at 1/576). Verify on 5 samples first.

### Interpretation
- **MLP R² ≥ 0.5:** causal importance is predictable from features. Distillation has a real target. Track A.
- **MLP R² in [0.2, 0.5]:** partially predictable. Distillation likely helps but ceiling unclear.
- **MLP R² < 0.2:** causal importance is fundamentally unpredictable from encoder features. Pre-LLM compression is bounded. **Publish that as Track C flagship.**

**This is D4.** Save the MLP R² to `runs/phase1_decision/D4.txt`.

### Deliverable
`runs/p1_5_2/probe_linear.json`, `runs/p1_5_2/probe_mlp.json`, and `runs/phase1_decision/D4.txt`.

---

# PHASE 1.75 — Decision gate

## P1.75 — Decision-gate analysis

### Goal
Combine D1, D2, D3, D4 → choose Track A / B / C.

### Run

`scripts/decision_gate.py`:

```python
"""Apply the decision tree in plan.md §10."""
import json
from pathlib import Path

def main():
    decision_dir = Path("runs/phase1_decision")
    D1 = float(open(decision_dir / "D1.txt").read().strip())
    D2 = float(open(decision_dir / "D2.txt").read().strip())
    D3 = float(open(decision_dir / "D3.txt").read().strip())
    D4 = float(open(decision_dir / "D4.txt").read().strip())

    print(f"D1 (T1 selector consistency Jaccard): {D1:.3f}")
    print(f"D2 (T6.5 LoRA-on-TPN - LoRA-on-random, avg over 3 bench): {D2:+.3f}")
    print(f"D3 (T_new1 max |Spearman| with any prior): {D3:.3f}")
    print(f"D4 (T_new2 probe R²): {D4:.3f}")

    # Decision tree from plan.md §10
    if D4 >= 0.5:
        if D3 >= 0.4:
            track = "A.1"
            justification = (
                f"D4={D4:.2f} ≥ 0.5: causal importance is predictable; distillation has a target. "
                f"D3={D3:.2f} ≥ 0.4: TPN learned a useful prior already. "
                f"Track A.1: run E_sup1 (distillation) + E_sup4 (SFT mix) + E_arch1 (per-token query)."
            )
        else:
            track = "A.2"
            justification = (
                f"D4={D4:.2f} ≥ 0.5: causal importance predictable. "
                f"D3={D3:.2f} < 0.4: TPN learned nothing useful but architecture could. "
                f"Track A.2: run E_sup3 (CLS-init) + E_sup1 (distill) to bootstrap."
            )
    elif D4 >= 0.2:
        if D2 < 0:
            track = "B"
            justification = (
                f"D4={D4:.2f} in [0.2, 0.5]: mixed signal. "
                f"D2={D2:+.2f} < 0: LoRA-on-random beats LoRA-on-TPN — selector is harmful. "
                f"Track B: architectural redesign. Run B1 (frozen-prior + tiny residual), "
                f"B2 (Q-Former bottleneck)."
            )
        else:
            track = "A.3"
            justification = (
                f"D4={D4:.2f} in [0.2, 0.5]: mixed signal. "
                f"D2={D2:+.2f} ≥ 0: LoRA-on-TPN matches or beats LoRA-on-random — selector contributes. "
                f"Track A.3: distillation + architectural tweaks (PTQ-XA, two-stage)."
            )
    else:
        track = "C"
        justification = (
            f"D4={D4:.2f} < 0.2: causal importance is unpredictable from encoder features. "
            f"Pre-LLM compression is fundamentally bounded by this. "
            f"Track C: write the fundamental-limit paper. Extend C1 (5000-sample analysis), "
            f"C2 (mid-LLM activation predictability), C3 (cross-encoder verification)."
        )

    summary = f"""# Phase 1 Decision Summary

D1 (T1 selector consistency Jaccard): {D1:.3f}
D2 (T6.5 LoRA-on-TPN - LoRA-on-random): {D2:+.3f}
D3 (T_new1 max |Spearman| with any prior): {D3:.3f}
D4 (T_new2 probe R²): {D4:.3f}

Track recommendation: {track}

Justification:
{justification}

Next phase: see plan.md §10 and §4 for the track-specific Phase 2 plan.
"""

    (decision_dir / "decision_summary.md").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
```

### Deliverable
`runs/phase1_decision/decision_summary.md`. This is the single output of the whole bundle.

---

# SUMMARY OF DELIVERABLES

| Phase | Deliverable | Path |
|-------|-------------|------|
| P0.1 | Causal-intervention CLI works | `runs/p0_sanity/PASS` |
| P0.2 | Synthetic-task sanity results | `runs/p0_synthetic/sanity_results.json` |
| P0.3 | Multi-seed launcher | `scripts/launch_multiseed.sh` |
| P0.4 | Compute baselines | `runs/p0_compute/baseline_compute.json` |
| P1.1 | Selector consistency (D1) | `runs/p1_t1/jaccard_result.json` |
| P1.2 | Query diversity | `runs/p1_t2/diversity_result.json` |
| P1.3 | Oracle Jaccard | `runs/p1_t3/oracle_vs_tpn_jaccard.json` |
| P1.4 | Coverage map | `runs/p1_t4/coverage_stats.json` |
| P1.5 | LoRA-on-X (D2) | `runs/p1_t65/eval_*.json` |
| P1.6 | Merge CKA | `runs/p1_t8/cka_result.json` |
| P1.7 | Counterfactual | `runs/p1_t10/counterfactual_results.json` |
| P1.8 | Failure clusters | `runs/p1_t11/failure_clusters.json` |
| P1.5.1 | Multi-prior correlation (D3) | `runs/p1_5_1/multi_prior_correlation.json` |
| P1.5.2 | Causal probe R² (D4) | `runs/p1_5_2/probe_mlp.json` |
| P1.75 | **Decision summary** | **`runs/phase1_decision/decision_summary.md`** |

---

# CRITICAL DO-NOTs

- **Do not retrain the TPN.** Phase 1 is diagnostics on existing checkpoints.
- **Do not start Phase 2 experiments until P1.75 decision is logged.**
- **Do not commit large checkpoints to git.** Add `runs/` to `.gitignore`.
- **Do not skip sanity checks.** Each task has one; run it first.
- **Do not change multiple things between sanity check and full run.** If sanity passes, run the full sweep with identical settings.
- **Do not modify the submitted `ICONIP2026_G5_TCVitok/main.tex`.** That paper is done.

---

# REPORTING BACK

After each task completes, append a one-line entry to `runs/PHASE1_LOG.md`:

```markdown
- [DATE] P1.1: D1 = 0.45 (Jaccard between Final and Seed7 on 500 GQA)
- [DATE] P1.2: mean per-image KL = 0.034 (low — TPN approximately query-agnostic)
- [DATE] P1.3: oracle Jaccard = 0.19 (TPN nearly uncorrelated with what LLM uses)
- [DATE] P1.5: D2 = -1.8 pp (LoRA-on-random beats LoRA-on-TPN; selector is harmful)
- [DATE] P1.5.1: D3 = 0.07 (TPN scores essentially uncorrelated with any prior)
- [DATE] P1.5.2: D4 = 0.15 (causal importance not strongly predictable)
- [DATE] P1.75: Track recommendation = C
```

Then send the contents of `runs/phase1_decision/decision_summary.md` to the human, plus any blocking issues encountered.

---

# WHAT HAPPENS NEXT

Once the decision summary is written and the human signs off, the next bundle will be:

- **If Track A:** `CODEX_PHASE2_TRACKA.md` — distillation + per-token query + OCR pathway experiments.
- **If Track B:** `CODEX_PHASE2_TRACKB.md` — frozen-prior + Q-Former bottleneck + mid-LLM compression.
- **If Track C:** `CODEX_PHASE2_TRACKC.md` — full 5000-sample causal analysis + cross-encoder verification.

The human will write that bundle after reviewing the Phase 1 results. Do not assume a track and start Phase 2 work autonomously.
