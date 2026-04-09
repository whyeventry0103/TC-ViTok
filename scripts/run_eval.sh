#!/usr/bin/env bash
# run_eval.sh — Orchestrate all TC-ViTok evaluations on the server.
#
# Run inside a tmux session:
#   tmux new -s eval
#   bash scripts/run_eval.sh
#
# All output is tee'd to logs/eval_TIMESTAMP.log.
# Results land in runs/eval_results/*.json

set -euo pipefail

# ── Environment ──────────────────────────────────────────────────────────────
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export KMP_DUPLICATE_LIB_OK=TRUE

DATA_ROOT="${DATA_ROOT:-${HOME}/tc-vitok-data}"
CHECKPOINT="${1:-runs/phase1/best.ckpt}"
OUTDIR="runs/eval_results"
LOGDIR="logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="${LOGDIR}/eval_${TIMESTAMP}.log"

# ── Sanity checks ────────────────────────────────────────────────────────────
if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "ERROR: Checkpoint not found: ${CHECKPOINT}"
    echo "Usage: bash scripts/run_eval.sh [checkpoint_path]"
    echo "       Default: runs/phase1/best.ckpt"
    exit 1
fi

if [[ ! -d "${DATA_ROOT}" ]]; then
    echo "ERROR: DATA_ROOT not found: ${DATA_ROOT}"
    echo "Set DATA_ROOT environment variable or pass correct path."
    exit 1
fi

mkdir -p "${OUTDIR}" "${LOGDIR}"

# Redirect all output through tee so we get both console and file log
exec > >(tee -a "${LOGFILE}") 2>&1

echo "============================================================"
echo "  TC-ViTok Evaluation Suite"
echo "  $(date)"
echo "  Checkpoint : ${CHECKPOINT}"
echo "  Data root  : ${DATA_ROOT}"
echo "  Output dir : ${OUTDIR}"
echo "  Log file   : ${LOGFILE}"
echo "============================================================"
echo ""

# ── Helper: run one eval and report exit status ───────────────────────────────
run_eval() {
    local label="$1"
    shift
    echo "------------------------------------------------------------"
    echo "  START: ${label}"
    echo "  $(date)"
    echo "------------------------------------------------------------"
    local rc=0
    python scripts/evaluate.py "$@" || rc=$?
    if [[ ${rc} -eq 0 ]]; then
        echo "[OK] ${label} completed"
    else
        echo "[WARN] ${label} FAILED (exit ${rc}) — continuing with remaining evals"
    fi
    echo ""
}

# ── 1. TC-ViTok (main model, k=144) ──────────────────────────────────────────
run_eval "TC-ViTok k=144 — VQAv2 + GQA + POPE" \
    --checkpoint "${CHECKPOINT}" \
    --benchmarks vqav2 gqa pope \
    --spatial-eval \
    --data-root "${DATA_ROOT}" \
    --output "${OUTDIR}/tcvitok_k144.json"

# ── 2. Baseline A2: Random drop ───────────────────────────────────────────────
run_eval "Baseline: random_drop k=144 — VQAv2 + GQA + POPE" \
    --checkpoint "${CHECKPOINT}" \
    --baseline random_drop \
    --benchmarks vqav2 gqa pope \
    --spatial-eval \
    --data-root "${DATA_ROOT}" \
    --output "${OUTDIR}/random_drop_k144.json"

# ── 3. Baseline A3: Image-only TPN (ablation) ─────────────────────────────────
run_eval "Baseline: image_only TPN k=144 — VQAv2 + GQA + POPE" \
    --checkpoint "${CHECKPOINT}" \
    --baseline image_only \
    --benchmarks vqav2 gqa pope \
    --spatial-eval \
    --data-root "${DATA_ROOT}" \
    --output "${OUTDIR}/image_only_k144.json"

# ── 4. Summary comparison table ──────────────────────────────────────────────
echo "============================================================"
echo "  SUMMARY — $(date)"
echo "============================================================"

python - "${OUTDIR}" <<'PYEOF'
import json
import sys
from pathlib import Path

outdir = Path(sys.argv[1])

models = [
    ("TC-ViTok k=144",   "tcvitok_k144.json"),
    ("Random drop k=144","random_drop_k144.json"),
    ("Image-only TPN",   "image_only_k144.json"),
]

paper_targets = {
    "vqav2": 76.0,
    "gqa":   61.0,
    "gqa_s": 59.0,
    "gqa_g": 62.0,
    "pope":  84.0,
}

def fmt(v):
    """Format a value that is already in % (or None) for display."""
    if v is None:
        return "  N/A  "
    return f"{float(v):6.2f}%"

header = f"{'Model':<22} {'VQAv2':>8} {'GQA':>8} {'GQA-Sp':>8} {'GQA-Gl':>8} {'POPE-F1':>9}"
sep    = "-" * len(header)
print(header)
print(sep)

for label, fname in models:
    fpath = outdir / fname
    if not fpath.exists():
        print(f"{label:<22} {'(not found)':>48}")
        continue
    data = json.loads(fpath.read_text())
    # Results are stored under "benchmarks" key
    benches = data.get("benchmarks", data)  # fallback if flat
    vqav2 = benches.get("vqav2", {}).get("vqa_score")
    gqa   = benches.get("gqa",   {}).get("overall_accuracy")
    gqa_s = benches.get("gqa",   {}).get("spatial_accuracy")
    gqa_g = benches.get("gqa",   {}).get("global_accuracy")
    pope  = benches.get("pope",  {})
    # pope_f1: prefer macro_f1; fallback to mean of variants
    pope_f1 = pope.get("macro_f1")
    if pope_f1 is None:
        f1s = [pope.get(v, {}).get("f1") for v in ("popular","random","adversarial")]
        f1s = [x for x in f1s if x is not None]
        pope_f1 = sum(f1s) / len(f1s) if f1s else None
    print(f"{label:<22} {fmt(vqav2):>8} {fmt(gqa):>8} {fmt(gqa_s):>8} {fmt(gqa_g):>8} {fmt(pope_f1):>9}")

print(sep)
print(f"{'Paper targets':<22} {fmt(paper_targets['vqav2']):>8} "
      f"{fmt(paper_targets['gqa']):>8} "
      f"{fmt(paper_targets['gqa_s']):>8} "
      f"{fmt(paper_targets['gqa_g']):>8} "
      f"{fmt(paper_targets['pope']):>9}")
print()
PYEOF

echo "============================================================"
echo "  All evaluations complete."
echo "  Results saved to: ${OUTDIR}/"
echo "  Full log at    : ${LOGFILE}"
echo "  $(date)"
echo "============================================================"
