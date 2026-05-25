#!/usr/bin/env python3
"""Analyze Qwen2-VL FM1 oscillation logs."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_csv(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append((int(row["step"]), float(row["val_acc"])))
    if len(rows) < 3:
        raise ValueError(f"{path} has too few validation points")
    return np.asarray(rows, dtype=float)


def detrend(y, window=3):
    if len(y) <= window:
        return y - y.mean()
    kernel = np.ones(window) / window
    trend = np.convolve(y, kernel, mode="same")
    return y - trend


def estimate_period(series):
    steps = series[:, 0]
    y = detrend(series[:, 1])
    y = y - y.mean()
    denom = np.dot(y, y)
    if denom <= 1e-12:
        return None, 0.0
    ac = np.correlate(y, y, mode="full")[len(y)-1:] / denom
    if len(ac) < 3:
        return None, float(series[:, 1].max() - series[:, 1].min())
    peaks = []
    for i in range(1, len(ac) - 1):
        if ac[i] > ac[i - 1] and ac[i] > ac[i + 1] and ac[i] > 0:
            peaks.append(i)
    period = None
    if peaks:
        lag = peaks[0]
        period = float((steps[1] - steps[0]) * lag)
    amp = float(series[:, 1].max() - series[:, 1].min())
    return period, amp


def plot(a, b, period_a, period_b, output):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(6.8, 4.8), sharex=True)
    for ax, data, title, period in [
        (axes[0], a, r"Run A: $\eta_{selector}=5\times10^{-4}$", period_a),
        (axes[1], b, r"Run B: $\eta_{selector}=2\times10^{-5}$", period_b),
    ]:
        ax.plot(data[:, 0] / 1000.0, data[:, 1] * 100.0, marker="o", lw=1.5)
        label = "period: unavailable" if period is None else f"period ~ {period/1000:.1f}K steps"
        ax.set_title(f"{title} ({label})", fontsize=10)
        ax.set_ylabel("GQA acc (%)")
        ax.grid(True, alpha=0.25)
    axes[1].set_xlabel("Training step (K)")
    fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-a", required=True)
    p.add_argument("--run-b", required=True)
    p.add_argument("--output", default="figures/oscillation_qwenvl.pdf")
    args = p.parse_args()
    a = load_csv(args.run_a)
    b = load_csv(args.run_b)
    assert len(a) >= 20, "Run A needs >=20 validation points"
    assert len(b) >= 20, "Run B needs >=20 validation points"
    assert np.isfinite(a).all() and np.isfinite(b).all()
    period_a, amp_a = estimate_period(a)
    period_b, amp_b = estimate_period(b)
    ratio = None if period_a in (None, 0) or period_b is None else period_b / period_a
    summary = {
        "run_a": {"period_steps": period_a, "amplitude": amp_a},
        "run_b": {"period_steps": period_b, "amplitude": amp_b},
        "period_ratio_b_over_a": ratio,
        "fm1_consistent": bool(period_a and period_b and period_b > period_a),
    }
    print(json.dumps(summary, indent=2))
    plot(a, b, period_a, period_b, args.output)
    Path(args.output).with_suffix(".json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
