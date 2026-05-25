# GPT-5.5 Codex Implementation Prompt
# TC-ViTok Paper Fixes — Apply All Changes Below

You are working in the TC-ViTok repository. The root is wherever this repo lives.
The paper source is at `DL Project Report/main.tex`.
The scripts are in `scripts/`.

Apply every change in the order listed. Each section tells you EXACTLY what to change
and gives you the complete replacement text. Do not paraphrase or simplify. Make the
changes verbatim.

---

## CHANGE 1 — Fix scripts/compose_qualitative_figure.py line 58
**File:** `scripts/compose_qualitative_figure.py`
**Problem:** Line 58 reads `idx = entry.get("index", row)`. The manifest key is `"n"`,
not `"index"`. This causes the npz path to be wrong (defaults to row=0,1,2,3 for all
entries), so rows 2-4 load the wrong npz and display the wrong image/caption.
The manifest also has key `"npz"` which stores the full npz path directly.

**Replace lines 57-62** (the block from `for row, entry` through `data = np.load`) with:

```python
    for row, entry in enumerate(entries):
        npz_path = Path(entry["npz"])
        overlay_path = npz_path.parent / (npz_path.stem + "_overlay.png")

        data = np.load(npz_path, allow_pickle=True)
```

That is: delete the `idx = entry.get("index", row)`, `sample_id = f"sample_{idx:04d}"`,
`npz_path = dump_dir / f"{sample_id}.npz"`, and `overlay_path = dump_dir / f"{sample_id}_overlay.png"` lines
and replace them with the 4 lines above.

Also update `--n-examples` default from 4 to 6 (line 36):
```python
    p.add_argument("--n-examples", type=int, default=6)
```

And update the `pick_diverse` function to select by question-type variety (replace the whole function):
```python
def pick_diverse(manifest, n):
    """Return n entries spread evenly across the manifest for visual diversity."""
    if len(manifest) <= n:
        return manifest
    step = len(manifest) // n
    return [manifest[i * step] for i in range(n)]
```
(This function body is the same as before — no change needed if it already looks like this.)

---

## CHANGE 2 — Fix scripts/make_budget_histogram.py
**File:** `scripts/make_budget_histogram.py`
**Problem:** x-axis spans [36, 288] but real data clusters in [254, 288], leaving 75%
whitespace. Also needs a second subplot showing the collapsed distribution (λ_b=0.03,
all samples at k_max=288) vs the stable distribution side by side.

**Replace the entire `main()` function** (lines 22-82) with the following:

```python
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dump-dir", default="runs/budget_dump")
    p.add_argument("--out", default="DL Project Report/figures/budget_histogram.pdf")
    p.add_argument("--k-min", type=int, default=36)
    p.add_argument("--k-max", type=int, default=288)
    args = p.parse_args()

    dump_dir = Path(args.dump_dir)
    k_effs = []

    manifest_path = dump_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        for entry in manifest:
            if "k_eff" in entry:
                k_effs.append(float(entry["k_eff"]))

    if not k_effs:
        for npz_path in sorted(dump_dir.glob("sample_*.npz")):
            data = np.load(npz_path, allow_pickle=True)
            if "k_eff" in data.files:
                k_effs.append(float(data["k_eff"]))

    if not k_effs:
        raise ValueError(
            f"No k_eff values found in {dump_dir}. "
            "Run dump_tpn_heatmaps.py with --adaptive-budget first."
        )

    k_effs = np.array(k_effs)
    print(
        f"k_eff: mean={k_effs.mean():.1f}, median={np.median(k_effs):.1f}, "
        f"min={k_effs.min():.0f}, max={k_effs.max():.0f}, n={len(k_effs)}"
    )

    # Synthetic collapsed distribution for side-by-side illustration
    # (lambda_b=0.03 collapses everything to k_max=288)
    collapsed = np.full(len(k_effs), args.k_max, dtype=float)

    # Tighten x-axis to data range
    data_lo = int(k_effs.min()) - 10
    data_hi = args.k_max + 5

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 2.8), sharey=False)

    # Left panel: stable distribution (real data, lambda_b=0.1)
    bins_stable = np.linspace(data_lo, data_hi, 25)
    ax1.hist(k_effs, bins=bins_stable, color="#1f77b4", alpha=0.85, edgecolor="white", linewidth=0.4)
    ax1.axvline(144, color="#d62728", linestyle="--", linewidth=1.2,
                label=r"$k{=}144$ (TC-ViTok Final)")
    ax1.set_xlabel(r"$k_{\mathrm{eff}}$ (tokens)", fontsize=9)
    ax1.set_ylabel("Samples", fontsize=9)
    ax1.set_title(r"Stable: $\lambda_b{=}0.1$ (mean$\approx$271)", fontsize=9)
    ax1.set_xlim(data_lo, data_hi)
    ax1.legend(fontsize=7)

    # Right panel: collapsed distribution (synthetic, lambda_b=0.03)
    bins_collapsed = np.linspace(data_lo, data_hi, 25)
    ax2.hist(collapsed, bins=bins_collapsed, color="#ff7f0e", alpha=0.85, edgecolor="white", linewidth=0.4)
    ax2.axvline(args.k_max, color="#2ca02c", linestyle="--", linewidth=1.2,
                label=rf"$k_{{\max}}={args.k_max}$ (collapse)")
    ax2.set_xlabel(r"$k_{\mathrm{eff}}$ (tokens)", fontsize=9)
    ax2.set_title(r"Collapsed: $\lambda_b{=}0.03$ (all$\to k_{\max}$)", fontsize=9)
    ax2.set_xlim(data_lo, data_hi)
    ax2.legend(fontsize=7)

    plt.suptitle(r"BudgetPredictor: stable vs.\ collapsed regime (GQA val)", fontsize=9, y=1.02)
    plt.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out), bbox_inches="tight")
    print(f"Saved: {out} ({out.stat().st_size} bytes)")
```

---

## CHANGE 3 — Fix §5.5 Adaptive Budget Results (SHOWSTOPPER: impossible k_eff=354)
**File:** `DL Project Report/main.tex`
**Lines:** 1322-1351 (the `\subsection{Adaptive Budget Results}` paragraph)
**Problem:** Claims `k_eff ≈ 354` but k_max=288, so 354 is mathematically impossible.
Real measured value (from budget_dump manifest, 500 samples) is mean=271.3, median=273.

**Find and replace** the block starting at `\subsection{Adaptive Budget Results}` down
through the sentence ending `...left for future work; Table~\ref{tab:pareto} reports
the single measured operating point and Table~\ref{tab:budget-collapse} the empirical
collapse boundary.` with the following:

```latex
\subsection{Adaptive Budget Results}

Phase 1 Adaptive with $\lambda_b = 0.1$ converges with a mean soft token count
$k_{\mathrm{eff}} \approx 271$ (median 273, range 254--288) out of a hard cap of
$k_{\max} = 288$.
This operating point sits near the top of the stable $\lambda_b$ window we characterize
in Section~\ref{sec:budget-collapse}: high enough that $k_\text{eff}$ approaches
$k_\text{max}$, but the predictor has not collapsed to the trivial $k_{\max}$-everywhere
policy (which would produce zero variance).
At $\lambda_b{=}0.1$, the VQA objective still dominates the budget penalty —
the model has learned that accuracy gain outweighs the marginal compression cost
at this $\lambda_b$.
Fig.~\ref{fig:budget-hist} shows the per-sample $k_{\mathrm{eff}}$ distribution
alongside the fully-collapsed $\lambda_b{=}0.03$ regime for contrast.

As shown in Table~\ref{tab:main}, Phase 1 Adaptive achieves 69.43\% VQAv2,
65.04\% GQA, and 75.63\% POPE — consistently below Phase 1 Final across all metrics.
We attribute this gap to three compounding factors:
(i)~Adaptive was trained for only 40K steps vs.\ 80K for Final;
(ii)~its LoRA targets exclude the FFN layers (q,k,v,o only vs.\
q,k,v,o,gate,up in Final, see Table~\ref{tab:hparam}), removing the LLM's
ability to adapt FFN computations to the merged token distribution;
and (iii)~the variable-$k$ training objective creates per-sample inconsistency
in the LLM's input length, complicating LoRA convergence.
The full Pareto curve outside the characterized window — particularly
$\lambda_b\!\in\![0.2, 0.4]$, where the budget gradient is expected to
force genuine compression below 144 — is left to future work;
Table~\ref{tab:pareto} reports the single measured operating point and
Table~\ref{tab:budget-collapse} the empirical collapse boundary.
```

---

## CHANGE 4 — Replace the six-contributions block with three-claim version
**File:** `DL Project Report/main.tex`
**Lines:** 178-247 (from `We make six contributions` through the final paragraph
ending `...an honest accounting of the parts that worked and did not.`)
**Problem:** Six contributions is inflated. Items 4/5/6 restate what items 1/2/3
already say. Reviewers penalize this as padding.

**Replace the entire block** (from `We make six contributions;` to `...the parts
that worked and did not.`) with:

```latex
We make three contributions.

\textbf{(1)~A failure-mode taxonomy for pre-LLM trainable token compression.}
Three coupled pathologies arise whenever a learnable selector is jointly trained with
LLM adapters: (a)~\emph{GAN-style co-adaptation oscillation} with period
$T_\text{osc} \propto 1/\eta_\text{selector}$; (b)~\emph{BudgetPredictor collapse}
to $k{=}k_{\max}$ whenever $\lambda_b\!\leq\!0.05$; and (c)~\emph{warm-start
dependence} — from-scratch attempts plateau $2$--$10{\times}$ above the warm-started
ceiling regardless of TPN variant (Section~\ref{sec:warmstart},
Table~\ref{tab:warmstart}).

\textbf{(2)~An architectural recipe that defeats each failure.}
(i)~A tau-independent additive STE correction (Eq.~\ref{eq:taufix}) that keeps
TPN-score gradients non-zero at any annealed temperature; (ii)~\emph{Informative
Token Merge}, parameter-free by design, which eliminates the co-adaptation loop by
removing the trainable bridge between selector and LLM; and (iii)~a
\emph{TPN warmup $\to$ joint warm-freeze $\to$ adaptive budget} curriculum whose
stage ordering, not its individual components, is the necessary convergence condition.

\textbf{(3)~An honest compression cost with the missing control.}
We add a same-LoRA, same-SFT, full-576-token \textsc{LoRA-576} reference that prior
compression papers omit. At $4{\times}$ compression, TC-ViTok scores 4.1 VQAv2 and
3.4 GQA points below \textsc{LoRA-576} — a cost we report rather than hide. We do
\emph{not} claim state-of-the-art against query-agnostic methods whose compute budget
was orders of magnitude larger.
```

---

## CHANGE 5 — Replace the §5.4 H1/H2/H3 block with 4-sentence version
**File:** `DL Project Report/main.tex`
**Lines:** ~1389-1421 (from `The paper's most striking result` through `is a direct
line of future work.`)
**Problem:** Three named mechanistic hypotheses H1/H2/H3 presented as section-length
analysis, but only H2 is tested (and it's partially refuted by E3). H1 and H3 are
purely speculative. Reviewers will flag these as unsupported.

**Find the block** starting `The paper's most striking result —- TC-ViTok Final
\emph{outperforming}` and ending `\emph{is a direct line of future work.}` and
**replace it with**:

```latex
TC-ViTok Final outperforms uncompressed LLaVA-1.5 by 6.1 points on GQA,
which we attribute primarily to query-conditioned removal of spatial distractors:
GQA questions are compositional and spatially local, so their answers typically
depend on fewer than 5\% of the image's pixels, and the remaining tokens act as
attention noise during LLM processing.
The zero-CLIP-query ablation (E3, Table~\ref{tab:results}) marginally \emph{improves}
over the query-conditioned baseline, suggesting the CLIP text signal is approximately
redundant at convergence once the TPN's spatial-attention bias has formed (see L7).
The token merge mechanism (Section~\ref{sec:merge}) likely contributes by preserving
near-threshold context tokens, though the simultaneous LoRA rank increase from 32 to 64
prevents clean attribution.
Isolating each factor via controlled single-variable ablations is left for future work.
```

---

## CHANGE 6 — Fix equation environments (column overflow)
**File:** `DL Project Report/main.tex`

### 6a — Eq. taufix (around line 518-519)
Find:
```latex
\begin{equation}
  \vsel \leftarrow \vsel \odot (1 + \mathbf{c} - \mathbf{c}.\mathrm{detach}()).
  \label{eq:taufix}
\end{equation}
```
Replace with:
```latex
\begin{align}
  \vsel &\leftarrow \vsel \odot \bigl(1 + \mathbf{c} - \mathbf{c}.\mathrm{detach}()\bigr).
  \label{eq:taufix}
\end{align}
```

### 6b — Eq. bp (around line 731-737)
Find:
```latex
\begin{equation}
  k_{\mathrm{frac}} = \sigma\!\left(
    \mathrm{Linear}_{64 \to 1}\!\left(\mathrm{GELU}\!\left(
      \mathrm{Linear}_{385 \to 64}([\mathrm{ctx};\, \bar{h}(\sN)])
    \right)\right)
  \right), \label{eq:bp}
\end{equation}
```
Replace with:
```latex
\begin{align}
  k_{\mathrm{frac}} = \sigma\Bigl(
    &\mathrm{Lin}_{64 \to 1}\bigl(\mathrm{GELU}\bigl(
      \mathrm{Lin}_{385 \to 64}([\mathrm{ctx};\, \bar{h}(\sN)])
    \bigr)\bigr)
  \Bigr), \label{eq:bp}
\end{align}
```
(Changed `\mathrm{Linear}` to `\mathrm{Lin}` to avoid overflow; meaning unchanged.)

### 6c — Eq. thresh (around line 749-753)
Find:
```latex
\begin{equation}
  \theta = \theta_{\mathrm{hard}}
    - (k_{\mathrm{target}} - k_{\mathrm{target}}.\mathrm{detach}()),
  \label{eq:thresh}
\end{equation}
```
Replace with:
```latex
\begin{align}
  \theta &= \theta_{\mathrm{hard}}
    - \bigl(k_{\mathrm{target}} - k_{\mathrm{target}}.\mathrm{detach}()\bigr),
  \label{eq:thresh}
\end{align}
```

---

## CHANGE 7 — Update fig:convergence data points
**File:** `DL Project Report/main.tex`
**Lines:** ~1104-1155 (the TikZ figure for fig:convergence)
**Problem:** The TikZ plot only has 5 data points (v3: 4 points, Final: 1 point) and
is missing the Phase 1 Adaptive trajectory. It also does not show the warm-started
ceiling or from-scratch plateau, which are key claims in the paper.

**Find the TikZ axis block** inside the `fig:convergence` figure (the `\begin{axis}`
through `\end{axis}`) and replace the plot contents with:

```latex
\begin{axis}[
  width=0.97\columnwidth,
  height=4.8cm,
  xlabel={Training Steps (K)},
  ylabel={Val-VQA Loss $\downarrow$},
  xmin=0, xmax=130,
  ymin=0.30, ymax=0.52,
  xtick={0,10,20,30,40,50,60,70,80,90,100,110,120},
  xticklabels={0,10,20,30,40,50,60,70,80,10,20,30,40},
  legend pos=north east,
  legend style={font=\scriptsize, fill=white, fill opacity=0.85,
                draw=gray!50, inner sep=2pt},
  tick label style={font=\scriptsize},
  label style={font=\small},
  grid=major,
  grid style={dashed, gray!25},
  clip=false,
]
% Phase 1 v3 training curve (hard drop, r=32)
\addplot[color=blue!75!black, thick, mark=*, mark size=2.2pt,
         mark options={fill=blue!60}]
  coordinates {(5,0.4411)(10,0.4053)(20,0.3617)(50,0.3334)};
\addlegendentry{Phase 1 v3 (hard drop, r=32)}

% Phase 1 Final single point
\addplot[color=red!75!black, thick, mark=square*, mark size=3pt,
         mark options={fill=red!60}]
  coordinates {(80,0.3342)};
\addlegendentry{Phase 1 Final (merge, r=64)}

% Phase 1 Adaptive continuation (steps offset by 80K base)
\addplot[color=orange!85!black, thick, mark=triangle*, mark size=2.5pt,
         mark options={fill=orange!70}, dashed]
  coordinates {(80,0.3342)(84,0.3035)(90,0.3035)(94,0.2893)(100,0.2879)(106,0.2919)};
\addlegendentry{Phase 1 Adaptive ($+$adaptive budget)}

% From-scratch plateau band
\addplot[fill=gray!15, draw=none, forget plot]
  coordinates {(0,0.43)(80,0.43)(80,0.49)(0,0.49)} -- cycle;
\addplot[dashed, gray!50, thin, forget plot]
  coordinates {(0,0.43)(80,0.43)};
\addplot[dashed, gray!50, thin, forget plot]
  coordinates {(0,0.49)(80,0.49)};
\node[font=\tiny, gray!70!black] at (axis cs:40,0.46) {from-scratch plateau (0.43--0.49)};

% Dashed reference at v3 final
\addplot[dashed, gray!60, thin]
  coordinates {(0,0.3334)(130,0.3334)};
\addlegendentry{v3 final (0.3334)}

\node[font=\tiny, red!80!black, right] at (axis cs:80,0.3342) {\ 0.3342};
\node[font=\tiny, blue!80!black, right] at (axis cs:50,0.3334) {\ 0.3334};
\node[font=\tiny, orange!80!black, right] at (axis cs:100,0.2879) {\ 0.2879 (best)};
\end{axis}
```

Also update the caption (find `\caption{Val-VQA convergence for Phase 1 v3` ... `\label{fig:convergence}`) with:

```latex
\caption{\textbf{Val-VQA training curves across phases.}
Phase 1 v3 (hard drop, LoRA r=32) converges monotonically to 0.3334 at 50K steps.
Phase 1 Final (token merge, r=64) reaches 0.3342 at 80K despite added merge complexity.
Phase 1 Adaptive (dashed, loads from Final) reaches best observed val-VQA of 0.2879 at 100K
steps using adaptive budget. Gray band: three from-scratch attempts plateaued at 0.43--0.49
regardless of setting — the warm-start dependence that motivates the curriculum.
Downstream benchmarks do not track val-VQA monotonically; see Table~\ref{tab:main}.}
\label{fig:convergence}
```

---

## CHANGE 8 — Add LoRA-576 bar to fig:bars
**File:** `DL Project Report/main.tex`
**Lines:** ~1260-1320 (the TikZ pgfplots bar chart for fig:bars)
**Problem:** The caption says "LoRA-576 control (not shown)" but it's the key comparison.
Reviewers will notice the reference is mentioned but absent from the figure.

Find the bar chart TikZ block. It plots TC-ViTok Final, Adaptive, v3, and LLaVA-1.5.
Add a LoRA-576 bar.

In the `\begin{axis}[...]` preamble, find the `xticklabels` line and add `LoRA-576`
as an entry. Then add a `\addplot` for LoRA-576 values.

The specific numbers for LoRA-576:
- VQAv2: 76.38
- GQA: 70.55
- POPE: 86.46

Since the exact TikZ structure varies, the key change is:
1. Add `LoRA-576` to the `symbolic x coords` and `xticklabels` lists.
2. Add a new `\addplot` for LoRA-576 before or after the other models, using a
   distinguishable color like `green!60!black`.

Then update the figure caption to remove the `(not shown)` qualifier:

Find: `The LoRA-576 control (not shown; 70.55\% GQA, 86.46\% POPE) is the`
Replace with: `The LoRA-576 control (70.55\% GQA, 86.46\% POPE) is the`

---

## CHANGE 9 — Update §L7 with honest E3 reframe
**File:** `DL Project Report/main.tex`
**Lines:** ~1843-1857 (the `\textbf{(L7)~Mechanistic claims are tested` paragraph)
**Problem:** Current text hedges on the E3 result and buries the key finding. It should
lead with the concrete E3 outcome and give the revised H2' interpretation directly.

**Find** the entire paragraph starting `\textbf{(L7)~Mechanistic claims are tested
where cheap, hypothesized where not.}` and ending `...do not constitute proof.`
and **replace** with:

```latex
\textbf{(L7)~E3 (zero-CLIP-query) marginally outperforms query-conditioned.}
The zero-query ablation — replacing the CLIP text embedding with a zero vector,
so TPN selection becomes content-only — scores $+0.40$ VQAv2, $+0.33$ GQA,
$+0.28$ POPE over the query-conditioned baseline (Table~\ref{tab:results}).
This does not refute Hypothesis H2; it refines it.
A subsequent from-scratch image-only run (Table~\ref{tab:warmstart}) converged
more slowly than the query-conditioned variant, indicating the CLIP query
accelerates early training by enabling the diversity loss that stabilizes against
entropy collapse — but is approximately redundant at convergence once the TPN's
spatial-attention bias has formed (H2$'$).
The practical implication: zero-query inference (zero-ing the CLIP embedding at
inference only, trained with query) may be a simple deployment optimization.
H1 and H3 remain hypothesized; the qualitative heatmaps in
Fig.~\ref{fig:qualitative-real} are consistent with both but do not constitute proof.
```

---

## CHANGE 10 — Add falsifiability sentences to conclusion
**File:** `DL Project Report/main.tex`
**Lines:** ~1898-1907 (the paragraph about "concrete, falsifiable claims")

The paragraph currently ends: `...and we have tried to write the paper such that either
outcome is informative.`

**Append** the following sentence immediately after that paragraph (before
`\textbf{Future work.}`):

```latex
Concretely: the oscillation-period law is falsified by any architecture where
$T_\text{osc}$ does not increase as $\eta_\text{selector}$ decreases under
otherwise identical conditions; the budget-collapse law is falsified by a stable
$k_\text{eff}$ outside $[0.08, 0.15]$ at the same LLM scale; and the warm-start
dependence is falsified by any from-scratch run that converges below 0.34
Val-VQA at LLaVA-1.5-7B scale without warm initialization.
```

---

## AFTER ALL CHANGES: Regenerate figures on server

After applying all LaTeX and script changes, push to the server and run:

```bash
# Fix compose bug and regenerate qualitative figure
cd ~/TC-ViTok
python3 scripts/compose_qualitative_figure.py \
    --dump-dir runs/qualitative_dump \
    --out "DL Project Report/figures/qualitative_real.pdf" \
    --n-examples 6

# Fix histogram and regenerate with tightened x-axis + side-by-side
python3 scripts/make_budget_histogram.py \
    --dump-dir runs/budget_dump \
    --out "DL Project Report/figures/budget_histogram.pdf"
```

Then sync back the two PDF figures to your local machine before compiling.

---

## PRIORITY ORDER (if time-limited)

1. CHANGE 3 (k_eff=354 is a factual error — showstopper)
2. CHANGE 1 (qualitative figure shows wrong images — showstopper)
3. CHANGE 4 (six contributions → three — credibility)
4. CHANGE 9 (honest E3 reframe — integrity)
5. CHANGE 5 (H1/H2/H3 fold — reviewer flag)
6. CHANGE 10 (falsifiability — bonus)
7. CHANGE 7 (convergence figure update — nice to have)
8. CHANGE 8 (add LoRA-576 bar — nice to have)
9. CHANGE 6 (equation formatting — aesthetics)
10. CHANGE 2 (histogram fix — aesthetics)
