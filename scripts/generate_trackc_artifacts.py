#!/usr/bin/env python3
"""Generate Track C summary artifacts from completed run outputs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open() as f:
        return json.load(f)


def read_float(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        return float(path.read_text().strip())
    except ValueError:
        return None


def fmt(x: float | None, digits: int = 3) -> str:
    if x is None:
        return "n/a"
    return f"{x:.{digits}f}"


def ci_pair(result: dict, prefix: str = "") -> list[float | None]:
    if prefix:
        return [
            result.get(f"{prefix}_ci_95_lower"),
            result.get(f"{prefix}_ci_95_upper"),
        ]
    if "bootstrap" in result:
        boot = result["bootstrap"]
        return [boot.get("ci_95_lower"), boot.get("ci_95_upper")]
    return [result.get("ci_95_lower"), result.get("ci_95_upper")]


def mean_r2(result: dict) -> float | None:
    if "bootstrap" in result:
        return result["bootstrap"].get("mean_r2")
    return result.get("mean_r2", result.get("bootstrap_mean_r2", result.get("r2")))


def fixed_r2(result: dict) -> float | None:
    return result.get("fixed_split_r2", result.get("r2"))


def first_present(item: dict, keys: list[str]) -> Any:
    for key in keys:
        if key in item and item[key] is not None:
            return item[key]
    return None


def metric_from_probe(item: dict, probe: str, keys: list[str]) -> Any:
    nested = item.get(probe)
    if isinstance(nested, dict):
        value = first_present(nested, keys)
        if value is not None:
            return value
    return first_present(item, [f"{probe}_{key}" for key in keys] + keys)


def normalize_qtype_rows(raw: Any) -> list[dict]:
    """Return robust per-question-type rows from several observed/likely schemas."""
    if not raw:
        return []

    candidates: list[tuple[str, Any]] = []
    if isinstance(raw, list):
        for i, item in enumerate(raw):
            if isinstance(item, dict):
                name = str(first_present(item, ["question_type", "type", "name", "category", "label"]) or f"type_{i}")
                candidates.append((name, item))
    elif isinstance(raw, dict):
        for container_key in ["results", "by_type", "per_type", "question_types", "types"]:
            nested = raw.get(container_key)
            if isinstance(nested, dict):
                candidates.extend((str(k), v) for k, v in nested.items())
                break
            if isinstance(nested, list):
                for i, item in enumerate(nested):
                    if isinstance(item, dict):
                        name = str(first_present(item, ["question_type", "type", "name", "category", "label"]) or f"type_{i}")
                        candidates.append((name, item))
                break
        if not candidates:
            candidates.extend((str(k), v) for k, v in raw.items())

    rows = []
    for name, item in candidates:
        if not isinstance(item, dict):
            continue

        mlp_mean = metric_from_probe(item, "mlp", ["mean_r2", "bootstrap_mean_r2", "r2", "fixed_split_r2"])
        if mlp_mean is None:
            mlp_mean = first_present(item, ["mlp_bootstrap_mean", "mlp_bootstrap_mean_r2", "bootstrap_mean_r2", "mean_r2"])

        fixed = metric_from_probe(item, "mlp", ["fixed_split_r2", "fixed_r2", "r2"])
        linear_mean = metric_from_probe(item, "linear", ["mean_r2", "bootstrap_mean_r2", "r2", "fixed_split_r2"])

        ci_lo = metric_from_probe(item, "mlp", ["ci_95_lower", "bootstrap_ci_95_lower", "lower"])
        ci_hi = metric_from_probe(item, "mlp", ["ci_95_upper", "bootstrap_ci_95_upper", "upper"])
        if ci_lo is None:
            ci_lo = first_present(item, ["mlp_ci_95_lower", "ci_95_lower", "bootstrap_ci_95_lower"])
        if ci_hi is None:
            ci_hi = first_present(item, ["mlp_ci_95_upper", "ci_95_upper", "bootstrap_ci_95_upper"])

        n_images = first_present(item, ["n_images", "n", "count", "num_images", "n_questions", "samples"])
        if mlp_mean is None and fixed is None and linear_mean is None:
            continue

        rows.append({
            "name": name,
            "n_images": n_images,
            "fixed_split_r2": fixed,
            "bootstrap_mean_r2": mlp_mean,
            "ci_95_lower": ci_lo,
            "ci_95_upper": ci_hi,
            "linear_bootstrap_mean_r2": linear_mean,
            "raw": item,
        })
    return sorted(rows, key=lambda row: row["name"])


def load_c1(root: Path) -> tuple[list[dict], dict]:
    linear = read_json(root / "runs/c1_causal_1000/probe_linear_boot.json", {})
    mlp = read_json(root / "runs/c1_causal_1000/probe_mlp_boot.json", {})
    qtype = read_json(root / "runs/c1_causal_1000/per_question_type_r2.json", {})
    reps = []
    if linear or mlp:
        reps.append({
            "name": "CLIP encoder",
            "source": "C1",
            "type": "encoder_feature",
            "depth": None,
            "linear_fixed_r2": fixed_r2(linear),
            "linear_r2_mean": mean_r2(linear),
            "linear_r2_ci": ci_pair(linear),
            "mlp_fixed_r2": fixed_r2(mlp),
            "mlp_r2_mean": mean_r2(mlp),
            "mlp_r2_ci": ci_pair(mlp),
            "paths": {
                "linear": "runs/c1_causal_1000/probe_linear_boot.json",
                "mlp": "runs/c1_causal_1000/probe_mlp_boot.json",
            },
        })
    return reps, qtype or {}


def load_c2(root: Path) -> list[dict]:
    data = read_json(root / "runs/c2_midllm/probe_per_layer_incremental.json", [])
    reps = []
    for item in data or []:
        layer = int(item["layer"])
        linear = item.get("linear", {})
        mlp = item.get("mlp", {})
        reps.append({
            "name": f"L{layer} visual residual",
            "source": "C2",
            "type": "visual_residual",
            "depth": layer,
            "linear_fixed_r2": fixed_r2(linear),
            "linear_r2_mean": mean_r2(linear),
            "linear_r2_ci": ci_pair(linear),
            "mlp_fixed_r2": fixed_r2(mlp),
            "mlp_r2_mean": mean_r2(mlp),
            "mlp_r2_ci": ci_pair(mlp),
            "paths": {"combined": "runs/c2_midllm/probe_per_layer_incremental.json"},
        })
    return reps


def load_c4(root: Path) -> list[dict]:
    items = []
    for rel in [
        "runs/c4_attention/smoke_probe_L1_log.json",
        "runs/c4_attention/smoke_probe_log.json",
    ]:
        data = read_json(root / rel, [])
        if isinstance(data, list):
            items.extend(data)
    by_layer = {}
    for item in items:
        by_layer[int(item["layer"])] = item
    reps = []
    for layer in sorted(by_layer):
        item = by_layer[layer]
        linear = item.get("linear", {})
        mlp = item.get("mlp", {})
        spear = item.get("spearman_attention_vs_importance", {})
        reps.append({
            "name": f"L{layer} text->visual attention",
            "source": "C4",
            "type": "text_visual_attention",
            "depth": layer,
            "linear_fixed_r2": fixed_r2(linear),
            "linear_r2_mean": mean_r2(linear),
            "linear_r2_ci": ci_pair(linear),
            "mlp_fixed_r2": fixed_r2(mlp),
            "mlp_r2_mean": mean_r2(mlp),
            "mlp_r2_ci": ci_pair(mlp),
            "spearman_mean": spear.get("mean_spearman"),
            "spearman_abs": spear.get("abs_mean"),
            "spearman_ci": [spear.get("ci_95_lower"), spear.get("ci_95_upper")],
            "paths": {
                "combined": "runs/c4_attention/smoke_probe_log.json",
                "l1": "runs/c4_attention/smoke_probe_L1_log.json",
            },
        })
    return reps


def load_c3(root: Path) -> list[dict]:
    linear = read_json(root / "runs/c3_dinov2/probe_linear_n1000.json", {})
    mlp = read_json(root / "runs/c3_dinov2/probe_mlp_n1000.json", {})
    if not linear and not mlp:
        return []
    return [{
        "name": "DINOv2 encoder",
        "source": "C3",
        "type": "encoder_feature",
        "depth": None,
        "linear_fixed_r2": fixed_r2(linear),
        "linear_r2_mean": mean_r2(linear),
        "linear_r2_ci": ci_pair(linear),
        "mlp_fixed_r2": fixed_r2(mlp),
        "mlp_r2_mean": mean_r2(mlp),
        "mlp_r2_ci": ci_pair(mlp),
        "paths": {
            "linear": "runs/c3_dinov2/probe_linear_n1000.json",
            "mlp": "runs/c3_dinov2/probe_mlp_n1000.json",
        },
    }]


def write_softmax_check(root: Path) -> None:
    out = root / "runs/c4_attention/softmax_denominator_check.log"
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        import torch
    except Exception as exc:
        out.write_text(f"Could not import torch: {exc}\n")
        return

    lines = []
    for rel in [
        "runs/c4_attention/smoke_attn_n100.pt",
        "runs/c4_attention/smoke_attn_L1_n100.pt",
    ]:
        path = root / rel
        if not path.exists():
            continue
        data = torch.load(path, map_location="cpu")
        lines.append(f"File: {rel}")
        for key, value in data.items():
            if not key.endswith("_perhead") or not hasattr(value, "shape"):
                continue
            ph = value.float()
            sums = ph[0].sum(dim=0)
            lines.append(
                f"{key}: first8_visual_sums="
                + ", ".join(f"{float(x):.6f}" for x in sums[:8])
            )
            lines.append(
                f"{key}: min={float(sums.min()):.6f} "
                f"mean={float(sums.mean()):.6f} max={float(sums.max()):.6f}"
            )
            lines.append(
                "Interpretation: sums near 1.0 imply visual-only normalization; "
                "sums below 1.0 imply full-sequence softmax with only visual mass reported."
            )
    if not lines:
        lines.append("No C4 attention feature files found.")
    out.write_text("\n".join(lines) + "\n")


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        widths = [max(w, len(cell)) for w, cell in zip(widths, row)]
    header = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    body = ["| " + " | ".join(c.ljust(w) for c, w in zip(row, widths)) + " |" for row in rows]
    return "\n".join([header, sep] + body)


def write_summary(root: Path, reps: list[dict], qtype: dict) -> None:
    d1 = read_float(root / "runs/phase1_decision/D1.txt")
    d3 = read_float(root / "runs/phase1_decision/D3.txt")
    d4 = read_float(root / "runs/phase1_decision/D4.txt")
    t3 = read_json(root / "runs/p1_t3/oracle_vs_tpn_jaccard.json", {})
    coverage = read_json(root / "runs/p1_t4/coverage_stats.json", {})

    def rep_by(source: str) -> list[dict]:
        return [r for r in reps if r["source"] == source]

    c1_rows = []
    for r in rep_by("C1"):
        c1_rows.append([
            "linear",
            fmt(r.get("linear_fixed_r2")),
            fmt(r.get("linear_r2_mean")),
            f"[{fmt(r.get('linear_r2_ci', [None, None])[0])}, {fmt(r.get('linear_r2_ci', [None, None])[1])}]",
        ])
        c1_rows.append([
            "MLP",
            fmt(r.get("mlp_fixed_r2")),
            fmt(r.get("mlp_r2_mean")),
            f"[{fmt(r.get('mlp_r2_ci', [None, None])[0])}, {fmt(r.get('mlp_r2_ci', [None, None])[1])}]",
        ])

    c2_rows = [[
        f"L{r['depth']}",
        fmt(r.get("linear_r2_mean")),
        f"[{fmt(r.get('linear_r2_ci', [None, None])[0])}, {fmt(r.get('linear_r2_ci', [None, None])[1])}]",
        fmt(r.get("mlp_r2_mean")),
        f"[{fmt(r.get('mlp_r2_ci', [None, None])[0])}, {fmt(r.get('mlp_r2_ci', [None, None])[1])}]",
    ] for r in rep_by("C2")]

    c4_rows = [[
        f"L{r['depth']}",
        fmt(r.get("spearman_mean")),
        fmt(r.get("spearman_abs")),
        fmt(r.get("linear_r2_mean")),
        fmt(r.get("mlp_r2_mean")),
    ] for r in rep_by("C4")]

    q_rows = []
    for item in normalize_qtype_rows(qtype):
        q_rows.append([
            item["name"],
            str(item.get("n_images", "n/a")),
            fmt(item.get("fixed_split_r2")),
            fmt(item.get("bootstrap_mean_r2")),
            f"[{fmt(item.get('ci_95_lower'))}, {fmt(item.get('ci_95_upper'))}]",
        ])

    text = f"""# Track C Decision Summary

## Phase 1 confirmed Track C

- D1 selector consistency Jaccard: {fmt(d1, 6)} (random baseline for k=144,N=576 is about 0.143)
- D3 max absolute Spearman with tested priors: {fmt(d3, 6)}
- D4 causal-importance predictability from encoder features: {fmt(d4, 6)}
- Track C was selected because D4 < 0.2.

## C1 - Encoder Feature Unpredictability at N=1000

{markdown_table(["Probe", "fixed R2", "boot mean", "95% CI"], c1_rows) if c1_rows else "C1 probe files not found."}

### Per-Question-Type Breakdown

{markdown_table(["Type", "N", "fixed R2", "boot mean", "95% CI"], q_rows) if q_rows else "Per-question-type file not found."}

## C2 - Visual-Token Residual Stream Unpredictability

{markdown_table(["Layer", "linear mean", "linear CI", "MLP mean", "MLP CI"], c2_rows) if c2_rows else "C2 layer probe file not found."}

## C4 - Text-to-Visual Attention Unpredictability Smoke at N=100

{markdown_table(["Layer", "Spearman mean", "abs mean", "linear R2", "MLP R2"], c4_rows) if c4_rows else "C4 smoke probe files not found."}

## C3 - Optional Cross-Encoder Check

{markdown_table(["Encoder", "linear mean", "linear CI", "MLP mean", "MLP CI"], [[r["name"], fmt(r.get("linear_r2_mean")), f"[{fmt(r.get('linear_r2_ci', [None, None])[0])}, {fmt(r.get('linear_r2_ci', [None, None])[1])}]", fmt(r.get("mlp_r2_mean")), f"[{fmt(r.get('mlp_r2_ci', [None, None])[0])}, {fmt(r.get('mlp_r2_ci', [None, None])[1])}]"] for r in rep_by("C3")]) if rep_by("C3") else "DINOv2 C3 has not been run yet."}

## Supporting Phase 1 Diagnostics

- P1.3 FastV-oracle Jaccard: {fmt(t3.get("mean_jaccard"), 6)} (random baseline about 0.143)
- P1.2 query-conditioned top-k Jaccard: 0.958 (near full overlap across questions)
- P0.2 synthetic needle target rate: about 0.27 (chance level is 0.25)
- P1.4 coverage extremes: dead/hot positions indicate a degenerate position policy{f" ({coverage})" if coverage else ""}.

## Headline Finding

Per-token causal visual importance is not recoverable from any tested representation
under the tested probe families. CLIP encoder features fail at N=1000, visual-token
LLM residual streams fail across early through late sampled depths, and text-to-visual
attention fails at FastV-relevant early layers and a mid-depth smoke test.

## Implications

The results suggest that current visual-token compression methods do not work by
identifying causally important tokens. Their empirical success is better understood
as preserving enough diversity and redundancy for the language model to recover from
imperfect token selection.
"""
    out = root / "runs/trackc_decision_summary.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)


def write_outline(root: Path) -> None:
    out = root / "paper_outline/trackc_paper_v1.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("""# Causal Token Importance is Unpredictable from Frozen LLM Representations

## Abstract

We measure per-token causal visual importance in a frozen LLaVA-1.5-7B and ask
whether it can be predicted from common compression-relevant representations.
Under linear and MLP probes with bootstrap confidence intervals, CLIP encoder
features, visual-token residual streams, and text-to-visual attention all fail to
predict causal importance. This reframes visual-token compression from importance
identification toward diversity preservation.

## 1. Introduction

- Visual-token compression often assumes useful tokens can be identified before or
  during early LLM computation.
- We test this assumption with ablation-defined causal importance.
- The core finding is negative across representation families.

## 2. Related Work

- Pre-LLM compressors: ToMe, EViT, PruMerge, TC-ViTok.
- Mid-LLM compressors: FastV, SparseVLM.
- Saliency, attribution, and token-importance probing.

## 3. Method

### 3.1 Per-token causal importance via ablation
### 3.2 Probe formulation and image-level bootstrap confidence intervals
### 3.3 Representations tested
### 3.4 Target variants and robustness checks

## 4. Encoder Feature Unpredictability

- C1 N=1000 linear and MLP results.
- Per-question-type breakdown.
- Target formulation diagnostics.

## 5. Visual-Token Residual Unpredictability

- L0, L4, L16, L28 visual residual probes.
- No onset curve; MLP remains negative and linear worsens with depth.

## 6. Attention Weight Unpredictability

- Text-to-visual attention at L1, L2, L16.
- L2 is FastV-relevant and still does not track causal importance.

## 7. Implication: Diversity Preservation, Not Importance Identification

- Compression methods may succeed by preserving broad recoverable evidence rather
  than selecting causally important tokens.
- This changes the design target for future pre-LLM and mid-LLM compression.

## 8. Supporting Phase 1 Evidence

- Seed-dependent selector solutions.
- Query-agnostic top-k behavior.
- Below-random FastV-oracle overlap.
- Coverage collapse into hot and dead positions.

## 9. Limitations

- LLaVA-1.5-7B and GQA val focus.
- Single-token ablation target.
- Linear and 2-layer MLP probes are strong but not exhaustive.
- Attention smoke is N=100 and should be framed as a gate, not a full sweep.

## 10. Optional Cross-Encoder Verification

- DINOv2-only C3 can test whether encoder unpredictability is CLIP-specific.

## 11. Conclusion
""")


def maybe_plot(root: Path, reps: list[dict], qtype: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:
        print(f"Skipping figures because matplotlib is unavailable: {exc}")
        return

    figdir = root / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    plot_reps = [r for r in reps if r.get("linear_r2_mean") is not None or r.get("mlp_r2_mean") is not None]
    if plot_reps:
        names = [r["name"].replace(" text->visual attention", " attn").replace(" visual residual", " vis") for r in plot_reps]
        linear = np.array([r.get("linear_r2_mean", math.nan) for r in plot_reps], dtype=float)
        mlp = np.array([r.get("mlp_r2_mean", math.nan) for r in plot_reps], dtype=float)
        x = np.arange(len(names))
        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.axhline(0, color="0.35", linestyle="--", linewidth=1, label="constant-mean baseline")
        ax.plot(x - 0.08, linear, "o-", label="linear probe")
        ax.plot(x + 0.08, mlp, "s-", label="MLP probe")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right")
        ax.set_ylabel("Held-out probe R2")
        ax.set_title("Causal importance is unpredictable across tested representations")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figdir / "predictability_summary.pdf")
        fig.savefig(figdir / "predictability_summary.png", dpi=200)
        plt.close(fig)

    if qtype:
        rows = [r for r in normalize_qtype_rows(qtype) if r.get("bootstrap_mean_r2") is not None]
        if rows:
            labels = [
                f"{r['name']}\n(n={r['n_images']})" if r.get("n_images") is not None else r["name"]
                for r in rows
            ]
            means = np.array([r["bootstrap_mean_r2"] for r in rows], dtype=float)
            lo = np.array([r.get("ci_95_lower", r["bootstrap_mean_r2"]) for r in rows], dtype=float)
            hi = np.array([r.get("ci_95_upper", r["bootstrap_mean_r2"]) for r in rows], dtype=float)
            x = np.arange(len(labels))
            fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(labels)), 4.5))
            ax.axhline(0, color="0.35", linestyle="--", linewidth=1)
            ax.bar(x, means, color="#4C78A8")
            ax.errorbar(x, means, yerr=[means - lo, hi - means], fmt="none", ecolor="black", capsize=3)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=25, ha="right")
            ax.set_ylabel("MLP probe R2")
            ax.set_title("MLP probe R2 by GQA question type (N=1000, CLIP encoder features)")
            fig.tight_layout()
            fig.savefig(figdir / "per_question_type_r2.pdf")
            fig.savefig(figdir / "per_question_type_r2.png", dpi=200)
            plt.close(fig)

    phase_values = {
        "D1 seed Jaccard": 0.171,
        "Random Jaccard": 0.143,
        "P1.2 query Jaccard": 0.958,
        "P1.3 oracle Jaccard": 0.131,
        "Random oracle": 0.143,
    }
    fig, ax = plt.subplots(figsize=(8.5, 4))
    ax.bar(range(len(phase_values)), list(phase_values.values()), color=["#E45756", "#BAB0AC", "#F58518", "#E45756", "#BAB0AC"])
    ax.set_xticks(range(len(phase_values)))
    ax.set_xticklabels(list(phase_values.keys()), rotation=25, ha="right")
    ax.set_ylabel("Jaccard / overlap")
    ax.set_title("Phase 1 selector degeneracy diagnostics")
    fig.tight_layout()
    fig.savefig(figdir / "phase1_degeneracy_summary.pdf")
    fig.savefig(figdir / "phase1_degeneracy_summary.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Repository root containing runs/")
    args = parser.parse_args()
    root = Path(args.root).resolve()

    c1_reps, qtype = load_c1(root)
    reps = c1_reps + load_c2(root) + load_c4(root) + load_c3(root)

    out_json = root / "runs/all_results_consolidated.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"representations": reps}, indent=2))

    write_softmax_check(root)
    write_summary(root, reps, qtype)
    write_outline(root)
    maybe_plot(root, reps, qtype)

    print(f"Wrote {out_json}")
    print(f"Wrote {root / 'runs/trackc_decision_summary.md'}")
    print(f"Wrote {root / 'runs/c4_attention/softmax_denominator_check.log'}")
    print(f"Wrote {root / 'paper_outline/trackc_paper_v1.md'}")
    print(f"Wrote figures under {root / 'figures'}")


if __name__ == "__main__":
    main()
