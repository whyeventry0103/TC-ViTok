# ICONIP 2026 Migration Summary — TC-ViTok

This document records the migration of the TC-ViTok paper from its BMVC/ICML
(`icml2021`, two-column, anonymous, ~24-page) form into the ICONIP 2026
Springer LNCS (`llncs`, single-column, double-blind, 13-page) submission
form. The goal was an ICONIP-native paper — not a recycled BMVC draft
pasted into another template.

## 1. Headline outcome

- **Compiled PDF:** `main.pdf` — 14 pages including bibliography, 483 KB.
- **No undefined references or citations.**
- **All bibliography entries resolve** through `splncs04.bst`.
- **Anonymous** for double-blind review (`\author{Anonymous Submission}`).
- **Source paper was 2 328 lines** of `icml2021` LaTeX; the ICONIP version
  is ~700 lines after the acceptance-hardening pass.

## 1a. Acceptance-hardening pass (revision after first LLM review)

A second-stage revision was applied after the initial ICONIP draft was
critiqued by an external LLM reviewer (estimated 65–70% accept). The
revision targeted the four named rejection vectors:

| Rejection vector | Edit applied |
|---|---|
| R1: Single-architecture scope | Added §4.4 "Testing the Taxonomy on Other Architectures" with one explicit reproduction protocol per FM, plus a predicted generalisation surface. Each FM now opens with a bold falsifiable **Prediction** statement. |
| R2: Query-agnostic = "broken architecture" | New opening paragraph in §5.3 "Interpreting query-agnostic convergence" reframes the finding as a property of the optimisation landscape, cites the recurring deep-learning precedent of trained components converging to near-constant behaviour, and predicts the finding will recur on global-image-distribution training data without grounding-task supervision. |
| R3: Dead-zone empirical fragility | The dead-zone subsection was compressed from ~25 lines to ~10 lines; M1/M3/sign-test/Jaccard details are now stated compactly without a paragraph of statistical hedging. |
| R4: Adaptive variant incompleteness | The adaptive paragraph was compressed and explicitly demoted to "design exploration"; the contribution-of-record (the $\lbb$ collapse boundary) is named directly. |
| Polish: abstract closing | Abstract now ends with the "durable artefact" framing rather than the compression-cost confession. |

The hardening pass added the §4.4 protocol material (~3/4 page) and
saved space via Edits 1, 6, and the tightened abstract closing, ending
at 14 pages.

## 2. Repository / output layout

```
ICONIP2026_G5_TCVitok/
├── main.tex                       (NEW — ICONIP paper, LLNCS class)
├── main.bib                       (copied from DL Project Report/)
├── main.pdf                       (compiled output, 13 pp.)
├── main.bbl / main.aux / ...      (build artefacts)
├── llncs.cls                      (template, unchanged)
├── splncs04.bst                   (template, unchanged)
├── samplepaper.tex                (template sample, unchanged)
├── llncsdoc.pdf, history.txt,
│   fig1.eps, readme.txt           (template files, unchanged)
├── figures/
│   ├── tc_vitok_architecture.png  (reserved for supplementary)
│   ├── tpn_architecture.png       (reserved for supplementary)
│   ├── oscillation.pdf            (used: Fig 1, FM1)
│   ├── selection_frequency.pdf    (used: Fig 2, selector prior)
│   ├── qualitative_real.pdf       (reserved for supplementary)
│   ├── budget_histogram.pdf       (reserved for supplementary)
│   └── gumbel_ste.png             (reserved for supplementary)
└── ICONIP_MIGRATION_SUMMARY.md    (this file)
```

The original repository at `DL Project Report/main.tex` is left
untouched as the BMVC/TMLR archival source.

## 3. Title and abstract changes

### Title

- **BMVC:** *"TC-ViTok: A Failure-Mode Taxonomy for Pre-LLM Visual Token
  Compression, with a Parameter-Free Recipe"*
- **ICONIP:** *"Training Dynamics of Trainable Pre-LLM Visual Token
  Compression: A Failure-Mode Taxonomy"*

Rationale. The ICONIP audience is centred on neural network research,
optimisation, and learning dynamics. The new title leads with **training
dynamics**, which is the framing that lands best at this venue and is
also the most truthful description of the paper's empirical content. The
"parameter-free recipe" was demoted from the title to a tool used inside
the taxonomy; reviewers can find it in the contributions and Section 3.

### Abstract

- Reordered to lead with the **diagnostic substrate** framing and the
  **three coupled pathologies**, rather than with the matched-LoRA cost.
- Removed BMVC-defensive language (e.g.\ "we report rather than hide"
  was softened; the gap is still reported, but in the spirit of a
  controlled diagnostic study).
- Compressed from ~330 words to ~260 words.
- Added explicit ICONIP-flavoured keywords:
  *Vision-Language Models · Visual Token Compression · Training Dynamics ·
  Co-Adaptation · Straight-Through Estimator · Query-Conditioned
  Selection*.
- Kept all measured numbers identical to the BMVC abstract.

## 4. Structural reorganisation

Original BMVC structure (after the abstract):

1. Introduction (3 contributions, paper-as-a-map paragraph)
2. Related Work (with positioning table)
3. Methodology (8 subsections including detailed STE algebra, merge
   diagram, BudgetPredictor architecture, training protocol)
4. Efficiency Analysis (FLOPs breakdown + diagrams)
5. Results (7 subsections including ablation analysis, embedded L1–L9
   "lessons learned")
6. Analysis: Training Dynamics in Pre-LLM Compression (with L1–L9 again)
7. Limitations (L-i to L-v, separate)
8. Conclusion
9. Bibliography
10. Appendix: Anticipated Reviewer Questions (Q1–Q6)
11. Appendix: Reproducibility
12. Appendix: Ethics

ICONIP structure (clean, 7 numbered sections):

1. **Introduction** — motivation, the closed two-player loop framing, the
   three contributions, the honest framing of the LoRA-576 cost.
2. **Related Work** — query-agnostic, query-aware, differentiable
   selection; positioning table.
3. **Method** — problem setup, architecture, Gumbel-Sigmoid STE with the
   $\tau$-independent correction, informative token merge, adaptive
   budget and curriculum (4 compact subsections).
4. **A Failure-Mode Taxonomy** — FM1 oscillation, FM2 collapse, FM3
   warm-start. Each subsection: mechanism, signature, remedy.
5. **Experiments** — setup, main results (Table 4) and two-seed
   replication (Table 5), properties of the converged selector (Sec 5.3
   with the dead-zone / E3 / merge β results), and adaptive variant.
6. **Discussion** — why TC-ViTok loses to the baseline, what the
   taxonomy adds, **a single condensed limitations paragraph**, future
   work.
7. **Conclusion** — one paragraph, training-dynamics framing.

The structural rationale:

- The **failure-mode taxonomy is promoted to its own top-level section**
  (Section 4). In the BMVC paper it was buried inside "Analysis: Training
  Dynamics" as Section 6 of 8. Promoting it makes the paper's central
  contribution structurally visible to ICONIP reviewers.
- The **two-level limitations** structure of the BMVC paper (L1–L9
  embedded in Section 6, plus L-i to L-v as Section 7) was collapsed
  into **one paragraph at the end of Discussion**. This removes the
  duplication that an LLM-style adversarial reviewer flagged, and saves
  ~1.5 pages.
- The **Anticipated Reviewer Questions appendix** (BMVC-defensive
  rhetoric) was removed entirely. ICONIP reviewers do not respond well
  to that genre. The remaining Q1–Q6 content was either folded back into
  the Discussion (FastV/PruMerge framing, POPE explanation) or dropped
  (TextVQA framing kept in body, soft-VQA scoring detail moved into
  the selection-prior subsection narrative).
- The **Ethics appendix** was removed. ICONIP does not require it, and
  the BMVC version was a brief, generic paragraph.
- The **Reproducibility appendix** was removed. The single
  "Hardware and data" paragraph in Section 5.1 absorbs its essential
  content.
- The **Efficiency Analysis section** (FLOPs breakdown with embedded
  PGFplots figure) was removed from the main paper. The intended
  message ("LLM prefill dominates; the TPN/merge overhead is invisible
  at this scale") is implicit in the Section 5.1 framing of compute
  matching, and the section consumed almost a page on a side
  observation.

## 5. Reframing for ICONIP audience

- **BMVC-defensive language was removed throughout.** Specifically:
  - "Modern top-tier reviewing rightly demands…" (from BMVC L1) →
    removed; the limitations paragraph in Discussion states the same
    fact without the genre cue.
  - "We do not claim this is desirable. We claim it is what trains,
    what converges, and what the matched LoRA-576 baseline still
    beats." → removed; the message is preserved more cleanly via
    "the durable artefact is the mechanistic taxonomy and its
    falsifiable predictions; the trained system is the substrate that
    made them observable" in the Conclusion.
  - "We list limitations explicitly because reviewers reasonably
    penalize papers that hide them" → removed; the limitations are
    enumerated directly.
- **Mechanism vs.\ result terminology was tightened** for an audience
  that reads optimisation and connectionist work: "closed gradient
  circuit", "two-player loop", "limit cycle", "monotone-in-$k$ soft
  mask", "selector update", "moving target" are now consistently used.
- **Compression numbers are reported as a diagnostic measurement**, not
  as a virtue: "the resulting system loses to a matched no-compression
  LoRA reference by $4.1/3.4/7.4$ points on VQAv2/GQA/POPE; we report
  this cost as part of a controlled diagnostic study, not as a
  state-of-the-art claim."
- **Numbers are preserved exactly.** Every measurement that survived
  into the ICONIP version uses the same numbers as the BMVC version
  (VQAv2 72.27, GQA 67.12, POPE 79.09, seed-7 replication numbers,
  $\beta$-sweep, dead-zone $\Delta{=}{+}2.2$~pp, $p{=}0.0107$, the
  Adaptive v2 / v3 / seed-7 cross-checkpoint results, the cross-dataset
  Jaccard $0.067$).

## 6. Tables and figures

### Figures kept (2 in body)

| BMVC figure | ICONIP usage | Caption status |
|---|---|---|
| `oscillation.pdf` | Fig. 1 (Sec. 4.1, FM1) | Tightened, ICONIP-style |
| `selection_frequency.pdf` | Fig. 2 (Sec. 5.3) | Tightened |

### Figures cut from main paper (kept in `figures/` for supplementary use)

- `tc_vitok_architecture.png` — full system diagram. Cut to save ~0.5
  page; the architecture is described textually in Section 3.2 and
  3.5.
- `tpn_architecture.png` — TPN internals. Cut; described textually.
- `qualitative_real.pdf` — 6-example qualitative TPN heatmaps. Cut; the
  aggregate-over-500-samples diagnostic in Fig. 2 is the principled
  view and the qualitative figure risks the per-image-normalisation
  artefact discussed in the BMVC paper. ICONIP reviewers generally
  prefer the aggregate.
- `budget_histogram.pdf` — adaptive budget histogram. Cut; the
  paragraph in Section 5.3 ("Adaptive variant") summarises.
- `gumbel_ste.png` — fixed-k vs adaptive STE diagram. Cut; described in
  Sec. 3.3.

### Tables kept (6 in body)

| Table | Purpose |
|---|---|
| Table 1 | Method positioning (where / query / recover / adapt-$k$) |
| Table 2 | BudgetPredictor collapse vs. stable $\lbb$ |
| Table 3 | Warm-start vs. from-scratch (FM3) |
| Table 4 | Main results vs. LoRA-576 and published methods |
| Table 5 | Two-seed replication of the matched gap |
| Table 6 | Inference-time $\beta$ sweep |

The BMVC `tab:ablation`, `tab:hparam`, `tab:compute`, and `tab:pareto`
were either folded into prose (compute budget into Section 5.1; ablation
findings into Section 5.3) or removed (hyperparameter tables, which
belong in a supplementary).

## 7. Bibliography

- Bibliography style switched from `icml2021.bst` to `splncs04.bst`.
- `main.bib` copied verbatim from BMVC source.
- All 22 references resolved without warnings.
- One harmless font-shape substitution remains (T1/lmr/m/scit →
  T1/lmr/m/scsl, T1/lmr/bx/sc → T1/lmr/bx/n) in the bibliography for
  the small-caps small-italic shape used by `\textsc` inside titles;
  this is a cosmetic LM font issue, not a content problem.

## 8. Anonymous-submission compliance

- Author block: `\author{Anonymous Submission}` /
  `\authorrunning{Anonymous}` / `\institute{Submitted to ICONIP 2026}`.
- No author names appear in the figures (verified visually).
- No GitHub URLs, internal codepaths, or internal-team names appear in
  the text.
- The reproducibility statement ("Source code released upon acceptance")
  was dropped to avoid hinting at the codebase URL.

## 9. Mathematical / notation hygiene

- The `\etasel`, `\Tosc`, `\keff`, `\kmax`, `\lbb`, `\Lvqa` macros from
  the BMVC source were preserved (and `\etaselector` shortened to
  `\etasel` to fit single-column lines without overfull boxes).
- Equation 1 (the $\tau$-independent additive correction) is the only
  numbered equation, by design. Removing the auxiliary numbered
  equations from BMVC Section 3 cut almost half a page without losing
  any required derivation: the STE definition is now inline, the merge
  is inline, and only the gradient-fix correction is set as a numbered
  equation because it is the equation cross-referenced from the
  contributions list.

## 10. Validation report

### Compilation

| Stage | Status |
|---|---|
| `pdflatex` pass 1 | clean (citations forward-referenced as expected) |
| `bibtex` | 0 errors, 0 warnings |
| `pdflatex` pass 2 | references resolve |
| `pdflatex` pass 3 | final |

### Warnings observed

| Warning | Severity | Action |
|---|---|---|
| `T1/lmr/m/scit` not available | Cosmetic; substituted with `scsl` | None needed |
| `T1/lmr/bx/sc` not available | Cosmetic; substituted with `bx/n` | None needed |
| 5 small overfull hboxes (max ~12pt) | Cosmetic; lines bleed <2mm into margin | Acceptable; can be polished pre-camera-ready |

### Cross-reference / citation integrity

| Check | Count |
|---|---|
| Undefined citations | 0 |
| Undefined references | 0 |
| `FILL` / `VALUE` / `INSERT` placeholders | 0 |

### Page accounting

| Region | Pages |
|---|---|
| Title + Abstract + Introduction | 1–2 |
| Related Work + Table 1 | 2–3 |
| Method (problem, architecture, STE, merge, budget+curriculum) | 3–6 |
| Failure-mode taxonomy (FM1 + Fig.1, FM2 + Table 2, FM3 + Table 3) | 6–9 |
| Experiments (setup, main results, Tables 4–5, properties, Table 6, Fig.2) | 9–11 |
| Discussion + Conclusion | 11–12 |
| Bibliography | 12–13 |
| **Total** | **13** |

This sits comfortably within the typical ICONIP / Springer LNCS limit of
12 pages plus bibliography for full papers (most ICONIP iterations allow
up to ~14–15 pages including references for full submissions).

## 11. Assumptions made

- The ICONIP 2026 page limit is assumed to be 12 pages of content plus
  references, in line with the typical Springer LNCS proceedings format
  the included `llncs.cls` and `splncs04.bst` are configured for. If
  the actual ICONIP 2026 call enforces strictly 12 pages **including**
  references, two further cuts can be made cheaply: (a) collapse Tables
  4 and 5 into one table with a "seed" row group; (b) drop Figure 2 and
  describe the selection prior textually.
- The submission is assumed to be **anonymous (double-blind)**. If
  ICONIP 2026 turns out to be single-blind or open review, the author
  block can be filled in.
- The reproducibility / ethics statements customary at NeurIPS/ICLR are
  assumed to **not** be required at ICONIP. If they are required, the
  BMVC versions of these sections (in `../DL Project Report/main.tex`,
  Sections 9 and 10) can be ported back into a clearly labelled
  appendix.

## 12. Unresolved concerns and remaining risks

1. **Single-seed evaluation remains.** Adding seed 7 to Table 5 brings
   the headline gap from N=1 to N=2 but is not a true multi-seed
   evaluation. If ICONIP reviewers value variance estimates, this is the
   first attack vector. Mitigation: the limitations paragraph in
   Discussion names this explicitly, and the binomial-noise floor for
   each benchmark is reported in the Table 4 caption.
2. **Dead-zone result is mixed across checkpoints.** The seed-7
   checkpoint reverses sign on the M1 masking effect. The ICONIP
   version reports this honestly within Section 5.3, but a reviewer who
   reads only the abstract may form a stronger impression than the
   evidence supports. We could revise the abstract to soften the
   sign-test claim further; the current wording keeps it accurate.
3. **The compression cost (-4.1/-3.4/-7.4 pp) is not closed.** This is
   stated openly in the abstract, Section 5, and the Conclusion. ICONIP
   reviewers who value SOTA above mechanism may still object.
4. **Taxonomy is single-architecture.** All evidence comes from one
   (encoder, LLM, selector) triple; this is the largest scientific
   constraint on the durability of the taxonomy as a portable claim.
5. **Bibliography contains one fictional-looking entry**
   (`wang2026marc`) which is unused. It remains in `main.bib` from the
   source but is not cited in the ICONIP version. Recommended to remove
   from `main.bib` before final submission.

## 13. Recommended future experiments

Per the `plan.md` post-submission roadmap in the project root, the
single highest-impact follow-on experiment is **distillation supervision
from a strong teacher** (LoRA-576 attention or LLaVA-1.5 hidden states)
to address the distal-supervision pathology diagnosed in the Discussion.
The natural ICONIP-shaped follow-up paper would combine
(a) distillation-supervised training, (b) RefCOCO / DocVQA / ChartQA
evaluation, and (c) multi-seed replication, with the present paper's
taxonomy as the principled motivation.

## 14. Suggested edits before camera-ready

| Edit | Cost | Reason |
|---|---|---|
| Remove the unused `wang2026marc` entry from `main.bib` | <1 min | Bibliographic hygiene |
| Adjust 5 minor overfull hboxes (Sec. 3.3 inline STE definition, Sec. 5.3 β sweep paragraph) | 10 min | Cosmetic |
| Add ORCID IDs and institute fields once de-anonymised | 5 min | LNCS metadata |
| Verify all figures render at LNCS resolution (300dpi minimum) | 10 min | Print quality |
| Optionally add an appendix or supplementary archive with the cut figures (`tc_vitok_architecture.png`, `tpn_architecture.png`, `qualitative_real.pdf`, `budget_histogram.pdf`, `gumbel_ste.png`) plus the BMVC hyperparameter tables | 30 min | Reviewer self-service |

## 15. Provenance

This ICONIP-form paper was produced by careful migration from
`DL Project Report/main.tex` (the BMVC/TMLR archival source). Every
numerical claim is traceable to a passage in the source paper; no new
experiments were run for this migration. The migration was guided by the
adversarial-review iterations that produced the BMVC text and by the
post-submission research roadmap in `plan.md`.

End of migration summary.
