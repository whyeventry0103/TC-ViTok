#!/usr/bin/env python3
"""Estimate the oscillation-period scaling slope from the reported points.

The paper currently reports two finite nonzero pooler-LR oscillation periods
plus one frozen-pooler control with no oscillation. A log-log slope can be fit
to the two finite points, but a percentile bootstrap CI is underdetermined with
fewer than three distinct finite measurements. This script records that fact
explicitly instead of inventing an extra point.
"""
import json
from pathlib import Path

import numpy as np


# From Section "Oscillation Period": v6 and v8 have finite measured periods.
# v8b is a control at lr=0 with no cycle and is excluded from log-log fitting.
FINITE_POINTS = [
    {"run": "v6", "eta_selector": 5e-4, "T_osc_ksteps": 5.0},
    {"run": "v8", "eta_selector": 2e-5, "T_osc_ksteps": 15.0},
]
EXCLUDED_CONTROLS = [
    {"run": "v8b", "eta_selector": 0.0, "T_osc_ksteps": None, "reason": "frozen pooler; no oscillation"},
]


def fit_slope(points):
    x = np.log([p["eta_selector"] for p in points])
    y = np.log([p["T_osc_ksteps"] for p in points])
    alpha, beta = np.polyfit(x, y, deg=1)
    y_hat = alpha * x + beta
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return float(alpha), float(beta), r2


def bootstrap_slope(points, n_boot=10000, seed=0):
    if len(points) < 3:
        return None
    rng = np.random.default_rng(seed)
    slopes = []
    for _ in range(n_boot):
        idx = rng.choice(len(points), size=len(points), replace=True)
        sample = [points[i] for i in idx]
        if len({p["eta_selector"] for p in sample}) < 2:
            continue
        slope, _, _ = fit_slope(sample)
        slopes.append(slope)
    if not slopes:
        return None
    lo, hi = np.quantile(slopes, [0.025, 0.975])
    return float(lo), float(hi), len(slopes)


def main():
    alpha, beta, r2 = fit_slope(FINITE_POINTS)
    ci = bootstrap_slope(FINITE_POINTS)
    result = {
        "finite_points": FINITE_POINTS,
        "excluded_controls": EXCLUDED_CONTROLS,
        "alpha_hat": alpha,
        "beta_hat": beta,
        "r2": r2,
        "target_alpha": -1.0,
        "ci95": None,
        "note": (
            "Only two finite nonzero oscillation-period points are reported; "
            "the frozen-pooler control has no finite period. A bootstrap CI is "
            "therefore underdetermined until another finite eta/T_osc point is added."
        ),
    }
    if ci is not None:
        result["ci95"] = [ci[0], ci[1]]
        result["n_boot_valid"] = ci[2]

    out = Path("runs/eval_results/oscillation_slope.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)

    print("Oscillation slope fit")
    for p in FINITE_POINTS:
        print(f"  {p['run']}: eta={p['eta_selector']:.2e}, T={p['T_osc_ksteps']:.1f}K")
    print("  excluded: v8b eta=0, no finite oscillation period")
    print(f"alpha_hat={alpha:+.3f}, R^2={r2:.3f}, target alpha=-1.000")
    if result["ci95"] is None:
        print("95% CI: underdetermined with only two finite points")
    else:
        lo, hi = result["ci95"]
        print(f"95% CI=[{lo:+.3f}, {hi:+.3f}]")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
