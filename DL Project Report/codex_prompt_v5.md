# GPT-5.5 Codex Prompt — Round 5
# Final reviewer-response fixes before BMVC submission

Apply every change verbatim. Ordered by priority.

---

## CHANGE 1 — Fix matplotlib title escape bug in make_selection_frequency.py
**File:** `scripts/make_selection_frequency.py`
**Lines 153-156**

The `rf"..."` string passes `val\_balanced` to matplotlib, which renders the
backslash literally. Fix by using a plain string.

Find:
```python
    title = args.title or (
        rf"TPN positional prior on GQA val\_balanced "
        rf"(N={count}, $k{{=}}144$)"
    )
```

Replace with:
```python
    title = args.title or (
        f"TPN positional prior on GQA val_balanced "
        f"(N={count}, $k$=144)"
    )
```

---

## CHANGE 2 — New script: dead-zone masking ablation (M1 experiment)
**Create file:** `scripts/mask_dead_zone_eval.py`

This script masks the dead-zone positions (row 0-4, col 6-11) at inference —
setting their TPN scores to -inf so they are never selected — and evaluates
VQAv2-score on 500 GQA val_balanced samples against the unmasked baseline.
The outcome will tell us whether the dead zone is a genuine attention sink
(masking it changes nothing) or carries information (masking hurts).

```python
#!/usr/bin/env python3
"""Ablation: mask the TPN score dead zone at inference and measure VQAv2-score delta.

Runs two forward passes per sample: one with the standard TPN, one with
positions (row 0-4, col 6-11) forced to score = -1e9 so they are never
selected by top-k.  Reports per-sample exact-match (VQA-style) and mean
VQAv2-score for both conditions.

Usage:
    DATA_ROOT=~/tc-vitok-data python3 scripts/mask_dead_zone_eval.py \\
        --checkpoint runs/phase1_final/phase1/best.ckpt \\
        --n-samples 500
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from tc_vitok.data.datasets import GQADataset
from tc_vitok.models.tc_vitok_model import TCViTokModel


# Dead-zone patch positions: row 0-4, col 6-11 in the 24x24 grid.
def dead_zone_flat_indices(grid=24, row_lo=0, row_hi=4, col_lo=6, col_hi=11):
    indices = []
    for r in range(row_lo, row_hi + 1):
        for c in range(col_lo, col_hi + 1):
            indices.append(r * grid + c)
    return indices


DEAD_ZONE = dead_zone_flat_indices()


def vqa_score(pred: str, answer: str) -> float:
    """Simple exact-match VQA score used for quick ablation."""
    return 1.0 if pred.strip().lower() == answer.strip().lower() else 0.0


def load_model(ckpt_path, device, llava_model):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    rank = 32
    for key, val in state_dict.items():
        if "lora_A" in key and "q_proj" in key:
            rank = val.shape[0]; break
    targets = {"q_proj", "k_proj", "v_proj", "o_proj"}
    for key in state_dict:
        if "lora_A" in key:
            for t in ("gate_proj", "up_proj", "down_proj"):
                if t in key: targets.add(t)
    lora_config = {"rank": rank, "alpha": rank * 2, "targets": sorted(targets), "dropout": 0.05}
    model = TCViTokModel(
        k=144,
        tpn_config={"n_heads": 4, "d_model": 384, "d_coarse": 384,
                    "d_h": 512, "d_text": 512, "M": 49, "N": 576},
        use_stub_llm=False, llava_model_name=llava_model,
        lora_config=lora_config, load_encoders=True,
        merge_beta=0.3, merge_tau=0.1, adaptive_budget=False,
    )
    model.load_state_dict(state_dict, strict=False)
    model.eval().to(device)
    return model


def run_eval(model, dataset, device, n, mask_dead_zone=False):
    scores = []
    dead_idx = torch.tensor(DEAD_ZONE, dtype=torch.long)
    for i in range(n):
        sample = dataset[i]
        images_coarse = sample["images_coarse"].unsqueeze(0).to(device)
        images_full = sample["images_full"].unsqueeze(0).to(device)
        question = sample["questions"]
        answer = str(sample.get("answer_str", ""))

        if mask_dead_zone:
            # Hook to zero out TPN scores at dead-zone positions before top-k
            def hook_fn(module, input, output):
                # output shape: (B, N) or (B, N, 1)
                out = output.clone()
                if out.dim() == 3:
                    out[:, dead_idx, :] = -1e9
                else:
                    out[:, dead_idx] = -1e9
                return out
            # Attach to TPN scoring head
            tpn = model.tpn
            handle = tpn.register_forward_hook(hook_fn)

        with torch.no_grad():
            preds = model.generate(
                images_coarse=images_coarse,
                images_full=images_full,
                questions=[question],
                max_new_tokens=20,
            )
        pred = preds[0]

        if mask_dead_zone:
            handle.remove()

        scores.append(vqa_score(pred, answer))
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{n}] running mean={np.mean(scores):.4f} "
                  f"(masked={mask_dead_zone})")
    return float(np.mean(scores))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", default=None)
    p.add_argument("--n-samples", type=int, default=500)
    p.add_argument("--device", default=None)
    p.add_argument("--llava-model", default="llava-hf/llava-1.5-7b-hf")
    args = p.parse_args()

    data_root = args.data_root or os.environ.get("DATA_ROOT")
    if not data_root:
        raise SystemExit("--data-root or $DATA_ROOT required")

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device, args.llava_model)

    dataset = GQADataset(
        questions_path=f"{data_root}/gqa/val_balanced_questions.json",
        image_dir=f"{data_root}/gqa/images",
    )
    n = min(args.n_samples, len(dataset))

    print(f"\n=== Baseline (no mask) ===")
    base_score = run_eval(model, dataset, device, n, mask_dead_zone=False)

    print(f"\n=== Masked dead zone (rows 0-4, cols 6-11) ===")
    masked_score = run_eval(model, dataset, device, n, mask_dead_zone=True)

    delta = masked_score - base_score
    print(f"\n{'='*50}")
    print(f"Baseline VQA-score:  {base_score:.4f}")
    print(f"Masked VQA-score:    {masked_score:.4f}")
    print(f"Delta (mask - base): {delta:+.4f}")
    if abs(delta) < 0.005:
        verdict = "UNCHANGED (consistent with attention-sink hypothesis)"
    elif delta < 0:
        verdict = "DEGRADES (dead zone carries information; refutes sink)"
    else:
        verdict = "IMPROVES (dead zone is noise; supports sink hypothesis)"
    print(f"Verdict: {verdict}")


if __name__ == "__main__":
    main()
```

---

## CHANGE 3 — Update paper title (drop "Stable")
**File:** `DL Project Report/main.tex`
**Lines 77-78**

Find:
```latex
\icmltitle{TC-ViTok: A Failure-Mode Taxonomy for Pre-LLM Visual Token\\
           Compression, with a Stable Curriculum and Parameter-Free Recipe}
```

Replace with:
```latex
\icmltitle{TC-ViTok: A Failure-Mode Taxonomy for Pre-LLM Visual Token\\
           Compression, with a Parameter-Free Recipe}
```

Also update the running head on line 60:

Find:
```latex
\icmltitlerunning{TC-ViTok: Task-Conditioned Visual Tokenization}
```

Replace with:
```latex
\icmltitlerunning{TC-ViTok: Failure-Mode Taxonomy for Pre-LLM Token Compression}
```

---

## CHANGE 4 — Reorder introduction to lead with the failure finding
**File:** `DL Project Report/main.tex`
**Lines 114-148**

Find the entire intro block (from "Modern VLMs" through "removed every component
of that loop that we could."):

```latex
Modern VLMs such as LLaVA-1.5~\cite{liu2024llava} attach a large language model (LLM)
to a frozen CLIP ViT-L/14 vision encoder via a learned multimodal projector.
At inference, the encoder maps a $336\times336$ image to $576$ patch tokens
($24\times24$ grid, patch size $14\times14$), which are prepended to the tokenized
question and fed into the LLM.
Since attention cost is quadratic in sequence length, these 576 visual tokens
(which far outnumber the $\sim$30 typical question tokens) dominate prefill FLOPs.

This fixed-token representation is informationally wasteful.
A question asking ``What is written on the sign?'' requires only the small image region
containing text, yet the LLM still receives all 576 tokens including sky, background,
and irrelevant objects. Conversely, ``Is there a person in the image?'' needs a global
scan rather than fine-grained local detail. The token budget is determined by image
resolution, not by what the question requires.

Prior work has addressed the redundancy half of this problem through token compression:
Token Merging (ToMe)~\cite{bolya2022token} and EViT~\cite{liang2022evit} reduce tokens
within the ViT; FastV~\cite{chen2024fastv} and LLaVA-PruMerge~\cite{shang2024llava}
prune after encoding; SparseVLM~\cite{zhang2025sparsevlm} recycles pruned tokens inside
the LLM. However, all these methods decide which tokens to keep based on \emph{image
content alone} — the user's question plays no role. This is the query-conditioning gap.

The few approaches that incorporate query signals do so either inside the LLM's attention
layers (SparseVLM, FocusLLaVA~\cite{zhu2024focusllava}), which still incurs the full
early-layer LLM cost, or inside the vision encoder (QG-VTC~\cite{li2025qgvtc}), which
tightly couples the compression logic to a specific encoder architecture and sacrifices
modularity.
We are unaware of any method that provides query-conditioned token selection as a
\emph{separate, lightweight module} operating on pre-computed frozen features
\emph{before} the LLM. In this paper we attempt that design and report, in addition to
its benchmark behavior, a structural finding: the obvious way to train such a module
(jointly with LLM adapters, end-to-end from random initialization) does not converge,
and the reason is not a tuning failure but a closed gradient loop whose dynamics we
characterize quantitatively. The architectural recipe we ultimately propose is what
remains after we removed every component of that loop that we could.
```

Replace with:

```latex
Pre-LLM visual token compression — reducing the visual sequence before it enters the
LLM, saving all prefill FLOPs — does not naively converge when the selector is
co-trained with LLM adapters. This paper is about why, and what to do about it.

Modern VLMs such as LLaVA-1.5~\cite{liu2024llava} attach a large language model (LLM)
to a frozen CLIP ViT-L/14 vision encoder. At inference, the encoder maps a
$336\times336$ image to $576$ patch tokens ($24\times24$ grid), which are prepended to
the tokenized question and fed into the LLM. Since attention cost is quadratic in
sequence length, these 576 visual tokens dominate prefill FLOPs — and the budget is
determined by image resolution, not by what the question requires.

Prior work compresses tokens either query-agnostically (ToMe~\cite{bolya2022token},
EViT~\cite{liang2022evit}, FastV~\cite{chen2024fastv},
PruMerge~\cite{shang2024llava}) or inside the LLM's attention layers, after all tokens
have been injected (SparseVLM~\cite{zhang2025sparsevlm},
FocusLLaVA~\cite{zhu2024focusllava}). Pre-LLM compression with a learned
query-conditioned selector is the architecturally natural design: it saves LLM FLOPs
and is modular. It also does not converge in the obvious training setup.

We attempted this design and found that the obvious joint training procedure fails in
three coupled ways: co-adaptation oscillation, budget-head collapse, and warm-start
dependence. We characterize each failure quantitatively, derive an architectural recipe
that sidesteps two of them by construction and characterizes the third empirically, and
report an honest compression cost against the matched same-LoRA baseline
(\textsc{LoRA-576}). The system we build (TC-ViTok) is the substrate for these
findings, not the contribution itself.
```

---

## CHANGE 5 — Soften Contribution 2 (sidesteps two failures, characterizes third)
**File:** `DL Project Report/main.tex`

Find:
```latex
\textbf{(2)~An architectural recipe that defeats each failure.}
(i)~A tau-independent additive STE correction (Eq.~\ref{eq:taufix}) that keeps
TPN-score gradients non-zero at any annealed temperature; (ii)~\emph{Informative
Token Merge}, parameter-free by design, which eliminates the co-adaptation loop by
removing the trainable bridge between selector and LLM; and (iii)~a
\emph{TPN warmup $\to$ joint warm-freeze $\to$ adaptive budget} curriculum whose
stage ordering, not its individual components, is the necessary convergence condition.
```

Replace with:
```latex
\textbf{(2)~An architectural recipe that sidesteps two failures and characterizes
the third.}
(i)~A tau-independent additive STE correction (Eq.~\ref{eq:taufix}) that keeps
TPN-score gradients non-zero at any annealed temperature; (ii)~\emph{Informative
Token Merge}, parameter-free by design, which eliminates the co-adaptation loop by
construction — removing the trainable bridge removes the closed gradient circuit;
and (iii)~a \emph{TPN warmup $\to$ joint warm-freeze $\to$ adaptive budget}
curriculum whose stage ordering is a necessary convergence condition in every
configuration we tested. The warm-start dependence (FM3) is characterized
empirically but not defeated: from-scratch training does not converge at this scale
regardless of architecture or curriculum order.
```

---

## CHANGE 6 — Rename §5.5 section title
**File:** `DL Project Report/main.tex`

Find:
```latex
\subsection{Mechanistic Hypotheses for the GQA Result}
\label{sec:hypotheses}
\label{sec:gqa-mechanism}
```

Replace with:
```latex
\subsection{Decomposing the GQA Gain: LoRA vs.\ Compression}
\label{sec:hypotheses}
\label{sec:gqa-mechanism}
```

---

## CHANGE 7 — Soften "lattice-like, not regional" claims in three places

**File:** `DL Project Report/main.tex`

### 7a — §5.5 prose (line ~1343)
Find:
```latex
than 5\%, with no contiguous regional mode. Per-image score statistics are
nearly uniform across positions \emph{except} for one localized dead zone in
the upper-center where mean normalized score drops by roughly $2{\times}$
relative to the rest of the image (Fig.~\ref{fig:positional-prior}, right).
Per-image qualitative inspection (Fig.~\ref{fig:qualitative-real}) can give
the impression of a regional bias because per-image min-max normalization
saturates whichever patches rank highest within a single image; the
aggregate-over-samples view shows the underlying pattern is structured but
not regional.
```

Replace with:
```latex
than 5\%. There is a mild upper-region preference in aggregate (rows 0--5 contain
more high-frequency positions than rows 12--17), but it is substantially weaker than
the per-sample qualitative figure suggested. Per-image score statistics are nearly
uniform across positions \emph{except} for one localized dead zone in the
upper-center where mean normalized score drops by roughly $2{\times}$ relative to
the rest of the image (Fig.~\ref{fig:positional-prior}, right).
Per-image qualitative inspection (Fig.~\ref{fig:qualitative-real}) can exaggerate the
regional appearance because per-image min-max normalization saturates on whichever
patches rank highest within that single image; the aggregate view shows a structured
but only mildly regional pattern.
```

### 7b — fig:qualitative-real caption (line ~1369)
Find:
```latex
shows the underlying selection pattern is lattice-structured rather than
regionally clustered.
```

Replace with:
```latex
shows the underlying pattern is heterogeneous and only mildly regional — much
weaker than per-sample visualization suggests.
```

### 7c — fig:positional-prior caption (line ~1385)
Find:
```latex
spatial structure is lattice-like rather than regionally clustered: specific
positions are selected on $>60\%$ of samples while others are selected on
$<5\%$, with no contiguous regional mode.
```

Replace with:
```latex
spatial structure is heterogeneous: specific positions are selected on $>60\%$ of
samples while others are selected on $<5\%$. There is a mild upper-region preference
in aggregate (rows 0--5), but it is substantially weaker than per-sample qualitative
inspection suggested; the dominant structure is position-specific heterogeneity, not
a regional blob.
```

### 7d — L9 limitation (line ~1849)
Find:
```latex
heterogeneity in selection frequency (lattice-like, not regional), and a
```

Replace with:
```latex
heterogeneity in selection frequency (position-specific, with mild upper-region
preference), and a
```

### 7e — L7 appended text (line ~1825)
Find:
```latex
lattice-like heterogeneity while mean score is nearly uniform except for a
localized upper-center dead zone, so the model is in practice a
```

Replace with:
```latex
position-specific heterogeneity with mild upper-region preference, while mean score
is nearly uniform except for a localized upper-center dead zone, so the model is in
practice a
```

---

## CHANGE 8 — Add TextVQA context to §L6
**File:** `DL Project Report/main.tex`

Find:
```latex
\textbf{(L6)~Benchmark coverage.}
We evaluate on VQAv2, GQA, POPE, MMBench (English dev), MME, and TextVQA
(\texttt{[E5]}, \texttt{[E6]}). We do not evaluate on RefCOCO, DocVQA,
ChartQA, or SEED-Bench. The first three would test query-conditioning's
strongest case (referent grounding, dense text); we view them as the
most natural follow-on experiments.
```

Replace with:
```latex
\textbf{(L6)~Benchmark coverage.}
We evaluate on VQAv2, GQA, POPE, MMBench (English dev), MME, and TextVQA
(\texttt{[E5]}, \texttt{[E6]}). We do not evaluate on RefCOCO, DocVQA,
ChartQA, or SEED-Bench. The first three would test query-conditioning's
strongest case (referent grounding, dense text); we view them as the
most natural follow-on experiments.
TextVQA scores low (25.8\%) relative to published methods ($\sim$45--55\%): this is
expected behavior. OCR requires dense local coverage across the entire image; our
$4{\times}$ compression with $k{=}144$ deliberately discards most tokens, and the
positional prior documented in L9 concentrates selection on a subset of positions
regardless of OCR needs. We do not present TC-ViTok as competitive on
OCR-centric benchmarks, and the LoRA-576 control (which also scores lower than
full-SFT published systems) confirms that the gap is primarily a compute-budget
constraint.
```

---

## CHANGE 9 — Add M1 result placeholder in L9 (to be filled after server run)

**File:** `DL Project Report/main.tex`
**Location:** end of the L9 paragraph (after `direct experiments would compare
against...`).

Find the sentence:
```latex
We do not test this conjecture in this paper;
direct experiments would compare against (a) register-token-equipped CLIP
encoders, (b) masking the dead-zone positions at inference, and (c)
explicit anti-sink regularization during training.
```

Replace with:
```latex
We partially test this conjecture: masking the dead-zone positions (rows 0--4,
cols 6--11) at inference by setting their TPN scores to $-\infty$ before top-$k$
selection changes VQA-score by $\Delta{=}$\textbf{[FILL: +X.XX or -X.XX or
$\approx$0.00]} over 500 GQA val samples — consistent with
\textbf{[FILL: attention-sink (delta$\approx$0) / information-carrying
(delta$<$0) / noise-carrying (delta$>$0)]} interpretation.
Full sink-vs-register analysis comparing register-token-equipped CLIP encoders
and explicit anti-sink regularization is left for future work.
```

(The user will run `scripts/mask_dead_zone_eval.py` on the server, get the delta,
and replace the `[FILL:...]` placeholders with the real number and verdict.)

---

## CHANGE 10 — Compile and verify

```bash
cd "DL Project Report"
pdflatex main.tex && pdflatex main.tex
grep -c "FILL" main.tex  # should return 2 (the M1 placeholders)
```

---

## AFTER CODEX: Server commands for the user

### Regenerate selection_frequency.pdf (fixes title bug):
```bash
cd ~/TC-ViTok
DATA_ROOT=~/tc-vitok-data python3 scripts/make_selection_frequency.py \
    --checkpoint runs/phase1_final/phase1/best.ckpt \
    --out "DL Project Report/figures/selection_frequency.pdf" \
    --n-samples 500
```

### Run M1 dead-zone masking ablation:
```bash
DATA_ROOT=~/tc-vitok-data python3 scripts/mask_dead_zone_eval.py \
    --checkpoint runs/phase1_final/phase1/best.ckpt \
    --n-samples 500
```

Note the delta output, then manually fill in the two `[FILL:...]` placeholders
in main.tex L9 with the real number and verdict.

### Sync figures and recompile:
```bash
# Local:
rsync -av 'sk10741@10.224.32.112:~/TC-ViTok/DL Project Report/figures/selection_frequency.pdf' \
    "/Users/apple/Desktop/Junior_spring/TC-ViTok/DL Project Report/figures/"
cd "/Users/apple/Desktop/Junior_spring/TC-ViTok/DL Project Report"
pdflatex main.tex && pdflatex main.tex
```
