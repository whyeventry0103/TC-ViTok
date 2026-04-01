"""Pareto frontier analysis for accuracy-vs-tokens trade-off.

This module provides utilities to evaluate TC-ViTok across different token
budgets (k ∈ {72, 144, 216, 288, 576}) and plot the Pareto curve showing
the accuracy-efficiency trade-off.

Paper reference: Section 8.4 (Token Budget Sweep)
"""

from typing import List, Dict, Tuple
import numpy as np


def compute_pareto_curve(
    accuracies: List[float],
    token_budgets: List[int],
) -> Dict[str, any]:
    """Compute Pareto frontier for accuracy-vs-tokens trade-off.

    Args:
        accuracies: List of accuracies (parallel to token_budgets)
        token_budgets: List of k values evaluated

    Returns:
        pareto_data: Dictionary with:
            - pareto_points: List of (k, acc) tuples on frontier
            - dominated_points: List of (k, acc) tuples dominated
            - efficiency_ratio: Accuracy per token (slope at k=144)

    A point (k1, acc1) dominates (k2, acc2) if:
        k1 < k2 AND acc1 >= acc2  (fewer tokens, same or better accuracy)

    Paper reference: Figure 3 (Pareto curves)
    """

    points = list(zip(token_budgets, accuracies))
    points.sort()  # Sort by token budget

    # Find Pareto frontier
    pareto = []
    dominated = []

    max_acc = -float("inf")
    for k, acc in points:
        if acc >= max_acc:
            pareto.append((k, acc))
            max_acc = acc
        else:
            dominated.append((k, acc))

    # Compute efficiency ratio (accuracy per token)
    # Use k=144 as reference point (common budget)
    k144_point = [(k, acc) for k, acc in pareto if k == 144]
    if k144_point:
        k144, acc144 = k144_point[0]
        efficiency_ratio = acc144 / k144
    else:
        efficiency_ratio = 0.0

    return {
        "pareto_points": pareto,
        "dominated_points": dominated,
        "efficiency_ratio": efficiency_ratio,
        "num_pareto": len(pareto),
        "num_dominated": len(dominated),
    }


def compare_pareto_curves(
    baseline_data: Dict[int, float],
    tcvitok_data: Dict[int, float],
) -> Dict[str, any]:
    """Compare Pareto curves between baseline and TC-ViTok.

    Args:
        baseline_data: {k: accuracy} for baseline (e.g., LLaVA-1.5)
        tcvitok_data: {k: accuracy} for TC-ViTok

    Returns:
        comparison: Dictionary with:
            - accuracy_gap: TC-ViTok acc - baseline acc at each k
            - efficiency_gain: How many fewer tokens for same accuracy
            - area_under_curve: Integral of Pareto curve

    Paper reference: Table 2 (Main Results)
    """

    # Compute accuracy gaps at each k
    common_ks = set(baseline_data.keys()) & set(tcvitok_data.keys())
    accuracy_gaps = {
        k: tcvitok_data[k] - baseline_data[k]
        for k in common_ks
    }

    # Find efficiency gain
    # For target accuracy, how many fewer tokens does TC-ViTok need?
    # Example: If baseline needs 288 tokens for 70% acc,
    # and TC-ViTok achieves 70% with 144 tokens, gain = 144 tokens saved

    efficiency_gains = {}
    for target_acc in [0.65, 0.70, 0.75, 0.80]:
        # Find minimum k to achieve target_acc for each method
        baseline_k = min(
            [k for k, acc in baseline_data.items() if acc >= target_acc],
            default=None,
        )
        tcvitok_k = min(
            [k for k, acc in tcvitok_data.items() if acc >= target_acc],
            default=None,
        )

        if baseline_k and tcvitok_k:
            efficiency_gains[target_acc] = baseline_k - tcvitok_k

    # Compute area under curve (AUC) using trapezoidal rule
    baseline_ks = sorted(baseline_data.keys())
    tcvitok_ks = sorted(tcvitok_data.keys())

    baseline_auc = np.trapz(
        [baseline_data[k] for k in baseline_ks],
        baseline_ks,
    )
    tcvitok_auc = np.trapz(
        [tcvitok_data[k] for k in tcvitok_ks],
        tcvitok_ks,
    )

    return {
        "accuracy_gaps": accuracy_gaps,
        "efficiency_gains": efficiency_gains,
        "baseline_auc": baseline_auc,
        "tcvitok_auc": tcvitok_auc,
        "auc_improvement": tcvitok_auc - baseline_auc,
    }


def format_pareto_table(
    results: Dict[int, Dict[str, float]],
    benchmarks: List[str] = ["vqav2", "gqa", "pope"],
) -> str:
    """Format Pareto sweep results as markdown table.

    Args:
        results: {k: {benchmark: accuracy}} nested dict
        benchmarks: List of benchmark names

    Returns:
        table: Markdown table string

    Example output:
        | k   | VQAv2 | GQA   | POPE  |
        |-----|-------|-------|-------|
        | 72  | 76.2  | 58.1  | 85.3  |
        | 144 | 78.5  | 61.4  | 86.7  |
        | ... | ...   | ...   | ...   |

    Paper reference: Table 2
    """

    # Header
    header = "| k   |" + "|".join(f" {b:^7s} |" for b in benchmarks)
    separator = "|-----|" + "|".join("-------|" for _ in benchmarks)

    # Rows
    rows = []
    for k in sorted(results.keys()):
        row_data = [f"{k:<3d}"]
        for bench in benchmarks:
            acc = results[k].get(bench, 0.0)
            row_data.append(f"{acc:^7.1f}")
        rows.append("| " + " | ".join(row_data) + " |")

    table = "\n".join([header, separator] + rows)
    return table
