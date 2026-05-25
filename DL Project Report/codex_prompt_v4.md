# GPT-5.5 Codex Prompt — Round 4
# Refine paper interpretation based on aggregate selection-frequency diagnostic

The selection-frequency diagnostic (Fig.\ positional-prior) revealed a more nuanced
picture than the per-image qualitative figure suggested:

- **P(selected)** is heterogeneous spatially (range 0-0.7, baseline 0.25, hot/cold
  ratio ~20x) but the pattern is *lattice-like*, NOT regionally clustered.
- **Mean per-image normalized score** is approximately uniform (~0.65 everywhere)
  *except* for a single "dead zone" of about 4x4 patches in the upper-center
  region (row 0-4, col 7-10) where mean score drops to ~0.35.
- The "upper-left bias" reading from the per-image qualitative figure was a
  visualization artifact of per-image min-max normalization saturating on
  whichever patches happen to be ranked highest within each image.

Apply the following changes verbatim.

---

## CHANGE 1 — Replace fig:positional-prior caption with corrected interpretation

**File:** `DL Project Report/main.tex`
**Location:** `\caption{...}` for `\label{fig:positional-prior}` (inserted Round 3).

**Replace the caption** with:

```latex
\caption{\textbf{Aggregate TPN selection diagnostics on 500 GQA val\_balanced
samples.}
\emph{Left:} per-position selection frequency $P_\text{select}$. Under fully
content-driven selection at $k{=}144$, the uniform baseline is
$144/576{=}0.25$ (per-position). The measured distribution is strongly
heterogeneous (range 0--0.7, hot/cold ratio $\approx 20{\times}$) but its
spatial structure is lattice-like rather than regionally clustered: specific
positions are selected on $>60\%$ of samples while others are selected on
$<5\%$, with no contiguous regional mode.
\emph{Right:} per-image min-max normalized score, averaged across samples,
is approximately uniform ($\bar{s}\!\approx\!0.65$) \emph{except} for a
localized $\sim\!4{\times}4$-patch ``dead zone'' in the upper-center
($\bar{s}\!\approx\!0.35$, row 0--4, col 7--10) that the TPN systematically
deprioritizes regardless of content. We conjecture this corresponds to an
attention-sink position learned during joint training; testing this with
register tokens~\cite{darcet2024vision} or sink masking is a direct follow-on.
Together with the E3 ablation (\S\ref{sec:limits}, L7), these aggregate
statistics give a geometric account of the convergent TPN behavior that
per-sample qualitative inspection (Fig.~\ref{fig:qualitative-real}) cannot
provide.}
```

---

## CHANGE 2 — Rewrite §5.4 paragraph with corrected mechanism

**File:** `DL Project Report/main.tex`
**Location:** the §5.4 paragraph (rewritten Round 3) that begins `TC-ViTok Final
outperforms uncompressed LLaVA-1.5 by 6.1 points on GQA`.

**Replace** the entire paragraph with:

```latex
TC-ViTok Final outperforms uncompressed LLaVA-1.5 by 6.1 points on GQA, which
we attribute primarily to LoRA fine-tuning of the LLM (see the matched
LoRA-576 control, Section~\ref{sec:control}, which captures most of this
gain). The compression-specific contribution is a heterogeneous per-position
selection prior with a $\sim\!20{\times}$ hot/cold ratio
(Fig.~\ref{fig:positional-prior}, left): certain spatial positions are
selected on more than 60\% of samples while others are selected on fewer
than 5\%, with no contiguous regional mode. Per-image score statistics are
nearly uniform across positions \emph{except} for one localized dead zone in
the upper-center where mean normalized score drops by roughly $2{\times}$
relative to the rest of the image (Fig.~\ref{fig:positional-prior}, right).
Per-image qualitative inspection (Fig.~\ref{fig:qualitative-real}) can give
the impression of a regional bias because per-image min-max normalization
saturates whichever patches rank highest within a single image; the
aggregate-over-samples view shows the underlying pattern is structured but
not regional. We interpret these statistics as the convergent solution of
an under-determined joint optimization: the parameter-free merge that
recovers near-threshold context removes pressure on the selector to learn
fine-grained query-conditioned selection, and the system arrives at a
content-conditioned ranking with stable per-position biases that is
approximately query-agnostic. This is consistent with the E3 zero-query
ablation in Section~\ref{sec:limits} (L7).
```

---

## CHANGE 3 — Update fig:qualitative-real caption to explain the viz artifact

**File:** `DL Project Report/main.tex`
**Location:** `\caption{...}` for `\label{fig:qualitative-real}`.

**Replace the caption** with:

```latex
\caption{\textbf{TPN selection on six GQA val\_balanced examples.}
Each row: raw image (left), top-$k{=}144$ selection overlay (right). The
overlay tints only selected patches with warm colors; alpha is modulated by
the per-image min-max normalized TPN score. Per-image normalization
saturates the colormap on whichever positions rank highest \emph{within
that image}, which can produce the visual impression of a regional bias.
The aggregate-over-500-samples diagnostic (Fig.~\ref{fig:positional-prior})
shows the underlying selection pattern is lattice-structured rather than
regionally clustered. We include these examples to make per-sample
selection legible, not as evidence of question-conditioned focus, which
the aggregate analysis refutes.}
```

---

## CHANGE 4 — Rewrite L9 with the corrected finding

**File:** `DL Project Report/main.tex`
**Location:** the `(L9)` paragraph inserted Round 3.

**Replace the entire L9 paragraph** with:

```latex
\textbf{(L9)~Structured per-position selection prior with a localized score
dead zone.}
The aggregate selection diagnostic (Fig.~\ref{fig:positional-prior}) reveals
two stable properties of the converged TPN: a $\sim\!20{\times}$ spatial
heterogeneity in selection frequency (lattice-like, not regional), and a
localized dead zone of $\sim\!16$ patches in the upper-center where mean
score is systematically depressed. We conjecture the dead zone is an
attention-sink artifact analogous to those Darcet et
al.~\cite{darcet2024vision} identify in ViTs, learned during joint
fine-tuning of TPN with LoRA. We do not test this conjecture in this paper;
direct experiments would compare against (a) register-token-equipped CLIP
encoders, (b) masking the dead-zone positions at inference, and (c)
explicit anti-sink regularization during training. The broader scope
statement is: TC-ViTok is best described as a \emph{content-conditioned
ranker with stable per-position biases}, not as a query-conditioned
selector at this scale.
```

---

## CHANGE 5 — Update the conclusion falsifiability sentence

**File:** `DL Project Report/main.tex`
**Location:** the falsifiability sentence (last updated Round 3).

**Replace** the sentence with:

```latex
Concretely: the oscillation-period law is falsified by any architecture where
$T_\text{osc}$ does not increase as $\eta_\text{selector}$ decreases under
otherwise identical conditions; the budget-collapse law is falsified by a
stable $k_\text{eff}$ outside $[0.08, 0.15]$ at the same LLM scale; the
warm-start dependence is falsified by any from-scratch run that converges
below 0.34 Val-VQA at LLaVA-1.5-7B scale without warm initialization; and
the convergent-prior finding is falsified by a TPN+merge system whose
aggregate $P_\text{select}$ heterogeneity is within $2{\times}$ of the
uniform baseline, or whose mean per-image score is uniform across all
positions (no dead zone), under the same data and scale.
```

---

## CHANGE 6 — Add the register-token reference

**File:** `DL Project Report/main.bib`

Check whether `darcet2024vision` is already in the .bib. If not, add:

```bibtex
@inproceedings{darcet2024vision,
  title={Vision Transformers Need Registers},
  author={Darcet, Timoth\'{e}e and Oquab, Maxime and Mairal, Julien and Bojanowski, Piotr},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2024}
}
```

If your bib filename is different (e.g., `references.bib`), apply there.

---

## CHANGE 7 — Compile and verify

```bash
cd "DL Project Report"
pdflatex main.tex && pdflatex main.tex
```

The figure caption now references row/col coordinates (0--4, 7--10) for the
dead zone. If the regenerated `selection_frequency.pdf` shows the dead zone
in a meaningfully different location, update the L9 paragraph and the
fig:positional-prior caption accordingly to match the actual coordinates.
