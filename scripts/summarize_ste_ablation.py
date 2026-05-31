#!/usr/bin/env python3
"""Summarize STE ablation training logs.

Parses the normal TC-ViTok trainer log format and writes a compact JSON with
final validation metrics plus selector-gradient diagnostics.
"""

import argparse
import json
import re
from pathlib import Path


def parse_metric(line: str, key: str):
    match = re.search(rf"{re.escape(key)}: (-?[0-9.eE+-]+)", line)
    return float(match.group(1)) if match else None


def parse_log(path: Path, grad_threshold: float):
    train_grad = []
    val_rows = []

    for line in path.read_text(errors="ignore").splitlines():
        step_match = re.search(r"\[Step (\d+)\]", line)
        if not step_match:
            continue
        step = int(step_match.group(1))

        if "train |" in line and "grad_norm_tpn:" in line:
            train_grad.append(
                {
                    "step": step,
                    "tau": parse_metric(line, "tau"),
                    "grad_norm_tpn": parse_metric(line, "grad_norm_tpn"),
                    "grad_norm_lora": parse_metric(line, "grad_norm_lora"),
                    "grad_norm_mm_projector": parse_metric(line, "grad_norm_mm_projector"),
                }
            )

        if "val |" in line:
            val_rows.append(
                {
                    "step": step,
                    "loss": parse_metric(line, "loss"),
                    "vqa": parse_metric(line, "vqa"),
                    "k_effective_mean": parse_metric(line, "k_effective_mean"),
                }
            )

    below = [row for row in train_grad if row["grad_norm_tpn"] is not None and row["grad_norm_tpn"] < grad_threshold]
    first_below = below[0]["step"] if below else None

    return {
        "log_path": str(path),
        "n_train_grad_rows": len(train_grad),
        "n_val_rows": len(val_rows),
        "final_val": val_rows[-1] if val_rows else None,
        "first_tpn_grad_below_threshold_step": first_below,
        "grad_threshold": grad_threshold,
        "final_train_grad": train_grad[-1] if train_grad else None,
        "min_tpn_grad_norm": min((row["grad_norm_tpn"] for row in train_grad if row["grad_norm_tpn"] is not None), default=None),
        "max_tpn_grad_norm": max((row["grad_norm_tpn"] for row in train_grad if row["grad_norm_tpn"] is not None), default=None),
    }


def main():
    parser = argparse.ArgumentParser(description="Summarize STE ablation logs")
    parser.add_argument("--with-ste", type=Path, required=True)
    parser.add_argument("--no-ste", type=Path, required=True)
    parser.add_argument("--tau05", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grad-threshold", type=float, default=1e-4)
    args = parser.parse_args()

    summary = {
        "with_ste": parse_log(args.with_ste, args.grad_threshold),
        "no_ste": parse_log(args.no_ste, args.grad_threshold),
    }
    if args.tau05 is not None:
        summary["no_ste_tau05"] = parse_log(args.tau05, args.grad_threshold)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

