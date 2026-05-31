#!/usr/bin/env python3
"""Lightweight consistency checks for generated Track C artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()

    summary_path = root / "runs/trackc_decision_summary.md"
    consolidated_path = root / "runs/all_results_consolidated.json"
    softmax_path = root / "runs/c4_attention/softmax_denominator_check.log"

    summary = summary_path.read_text() if summary_path.exists() else ""
    consolidated = json.loads(consolidated_path.read_text()) if consolidated_path.exists() else {"representations": []}
    reps = consolidated.get("representations", [])

    print("=== Files ===")
    for path in [summary_path, consolidated_path, softmax_path]:
        print(f"{path.relative_to(root)}: {'OK' if path.exists() else 'MISSING'}")

    print("\n=== Representation Completeness ===")
    print(f"total representations: {len(reps)}")
    print(f"sources covered: {sorted({r.get('source') for r in reps})}")
    print(f"C2 depths: {sorted(r.get('depth') for r in reps if r.get('source') == 'C2')}")
    print(f"C4 depths: {sorted(r.get('depth') for r in reps if r.get('source') == 'C4')}")
    c3 = [r for r in reps if r.get("source") == "C3"]
    print(f"C3 rows: {len(c3)} {'(DINOv2 present)' if c3 else '(not run yet)'}")

    print("\n=== Key Number Cross-References ===")
    json_text = json.dumps(consolidated)
    for number in [
        "0.171286",
        "0.116980",
        "-0.268089",
        "-0.327110",
        "-0.207584",
        "-0.227896",
        "-0.229328",
        "-0.020984",
        "0.131008",
    ]:
        print(
            f"{number}: "
            f"summary={'yes' if number in summary else 'no'} "
            f"consolidated={'yes' if number in json_text else 'no'}"
        )

    print("\n=== Figure Files ===")
    for rel in [
        "figures/predictability_summary.pdf",
        "figures/predictability_summary.png",
        "figures/per_question_type_r2.pdf",
        "figures/per_question_type_r2.png",
        "figures/phase1_degeneracy_summary.pdf",
        "figures/phase1_degeneracy_summary.png",
    ]:
        path = root / rel
        print(f"{rel}: {'OK' if path.exists() else 'MISSING'}")


if __name__ == "__main__":
    main()
