#!/usr/bin/env python3
"""Apply the Phase 1.75 decision tree from plan.md section 10."""

from __future__ import annotations

from pathlib import Path


def read_float(path: Path) -> float:
    try:
        return float(path.read_text().strip())
    except FileNotFoundError as exc:
        raise SystemExit(f"Missing decision input: {path}") from exc


def read_optional_float(path: Path) -> float | None:
    if not path.exists():
        return None
    return float(path.read_text().strip())


def main() -> None:
    decision_dir = Path("runs/phase1_decision")
    d1 = read_float(decision_dir / "D1.txt")
    d2 = read_optional_float(decision_dir / "D2.txt")
    d3 = read_float(decision_dir / "D3.txt")
    d4 = read_float(decision_dir / "D4.txt")

    if d4 >= 0.5:
        if d3 >= 0.4:
            track = "A.1"
            justification = (
                f"D4={d4:.3f} >= 0.5, so causal importance is predictable from "
                f"encoder features. D3={d3:.3f} >= 0.4, so the current TPN also "
                "aligns with at least one useful prior. Proceed with distillation "
                "and targeted architecture tweaks."
            )
        else:
            track = "A.2"
            justification = (
                f"D4={d4:.3f} >= 0.5, so causal importance is predictable, but "
                f"D3={d3:.3f} < 0.4, so the current TPN did not learn a useful "
                "prior. Use CLS/prior initialization plus distillation."
            )
    elif d4 >= 0.2:
        if d2 is None:
            raise SystemExit(
                "D2 is required when 0.2 <= D4 < 0.5 because it gates Track B vs A.3"
            )
        if d2 < 0:
            track = "B"
            justification = (
                f"D4={d4:.3f} is mixed and D2={d2:+.3f} is negative: LoRA-on-random "
                "beats LoRA-on-TPN on average. Treat the selector as harmful and "
                "move to architectural redesign."
            )
        else:
            track = "A.3"
            justification = (
                f"D4={d4:.3f} is mixed and D2={d2:+.3f} is non-negative: the selector "
                "contributes at least some value. Use a mixed distillation plus "
                "architecture-tweak track."
            )
    else:
        track = "C"
        justification = (
            f"D4={d4:.3f} < 0.2: causal importance is poorly predictable from "
            "encoder features. Treat this as a fundamental-limit result and "
            "expand causal analysis/cross-encoder verification. D2 was not run "
            "because this branch is decided by D4 alone; run it later only as "
            "supporting evidence."
        )

    d2_text = "not run (non-gating because D4 < 0.2)" if d2 is None else f"{d2:+.3f}"
    summary = f"""# Phase 1 Decision Summary

D1 (T1 selector consistency Jaccard): {d1:.3f}
D2 (T6.5 LoRA-on-TPN - LoRA-on-random): {d2_text}
D3 (T_new1 max |Spearman| with any prior): {d3:.3f}
D4 (T_new2 probe R^2): {d4:.3f}

Track recommendation: {track}

Justification:
{justification}
"""
    decision_dir.mkdir(parents=True, exist_ok=True)
    out = decision_dir / "decision_summary.md"
    out.write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
