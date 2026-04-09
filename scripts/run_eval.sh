#!/usr/bin/env bash
# run_eval.sh — Orchestrate all TC-ViTok evaluations on the server.
#
# Run inside a tmux session:
#   tmux new -s eval
#   bash scripts/run_eval.sh [checkpoint_path]
#
# Logs:
#   logs/eval_<TIMESTAMP>.log          — master log (everything)
#   logs/eval_tcvitok_<TIMESTAMP>.log  — TC-ViTok run only
#   logs/eval_random_<TIMESTAMP>.log   — random-drop baseline
#   logs/eval_imgonly_<TIMESTAMP>.log  — image-only TPN baseline
#
# Results land in runs/eval_results/*.json

set -uo pipefail

# ── Environment ──────────────────────────────────────────────────────────────
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export KMP_DUPLICATE_LIB_OK=TRUE

DATA_ROOT="${DATA_ROOT:-${HOME}/tc-vitok-data}"
CHECKPOINT="${1:-runs/phase1/best.ckpt}"
OUTDIR="runs/eval_results"
LOG_DIR="${LOG_DIR:-logs}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

MASTER_LOG="${LOG_DIR}/eval_${TIMESTAMP}.log"
LOG_TC="${LOG_DIR}/eval_tcvitok_${TIMESTAMP}.log"
LOG_RD="${LOG_DIR}/eval_random_${TIMESTAMP}.log"
LOG_IO="${LOG_DIR}/eval_imgonly_${TIMESTAMP}.log"

# ── Sanity checks ────────────────────────────────────────────────────────────
if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "ERROR: Checkpoint not found: ${CHECKPOINT}"
    echo "Usage: bash scripts/run_eval.sh [checkpoint_path]"
    echo "       Default: runs/phase1/best.ckpt"
    exit 1
fi

if [[ ! -d "${DATA_ROOT}" ]]; then
    echo "ERROR: DATA_ROOT not found: ${DATA_ROOT}"
    echo "Set DATA_ROOT environment variable before running."
    exit 1
fi

mkdir -p "${OUTDIR}" "${LOG_DIR}"
echo "Logging to: ${MASTER_LOG}"

# ── run_one: run a single evaluate.py call, tee to sub-log + master log ──────
run_one() {
    local label="$1"
    local sublog="$2"
    shift 2

    {
        echo "============================================================"
        echo "  START: ${label}"
        echo "  $(date)"
        echo "============================================================"

        local rc=0
        python scripts/evaluate.py "$@" || rc=$?

        echo "============================================================"
        if [[ ${rc} -eq 0 ]]; then
            echo "  [OK] ${label} completed — $(date)"
        else
            echo "  [FAIL] ${label} exited ${rc} — $(date)"
            echo "  Full log: ${sublog}"
        fi
        echo "============================================================"
        echo ""
    } 2>&1 | tee -a "${sublog}" >> "${MASTER_LOG}"

    # Echo status line to terminal (master log already has detail)
    if grep -q "\[OK\]" "${sublog}" 2>/dev/null; then
        echo "[OK]   ${label}"
    else
        echo "[FAIL] ${label}  — see ${sublog}"
    fi
}

# ── Header ───────────────────────────────────────────────────────────────────
{
echo "============================================================"
echo "  TC-ViTok Evaluation Suite — ${TIMESTAMP}"
echo "  Checkpoint : ${CHECKPOINT}"
echo "  Data root  : ${DATA_ROOT}"
echo "  Output dir : ${OUTDIR}"
echo "  Master log : ${MASTER_LOG}"
echo "============================================================"
echo ""
} | tee -a "${MASTER_LOG}"

# ── 1. TC-ViTok (main model, k=144) ──────────────────────────────────────────
run_one "TC-ViTok k=144 (VQAv2 + GQA + POPE)" "${LOG_TC}" \
    --checkpoint "${CHECKPOINT}" \
    --benchmarks vqav2 gqa pope \
    --spatial-eval \
    --data-root "${DATA_ROOT}" \
    --output "${OUTDIR}/tcvitok_k144.json"

# ── 2. Baseline A2: Random drop ──────────────────────────────────────────────
run_one "Baseline: random_drop k=144 (VQAv2 + GQA + POPE)" "${LOG_RD}" \
    --checkpoint "${CHECKPOINT}" \
    --baseline random_drop \
    --benchmarks vqav2 gqa pope \
    --spatial-eval \
    --data-root "${DATA_ROOT}" \
    --output "${OUTDIR}/random_drop_k144.json"

# ── 3. Baseline A3: Image-only TPN ───────────────────────────────────────────
run_one "Baseline: image_only TPN k=144 (VQAv2 + GQA + POPE)" "${LOG_IO}" \
    --checkpoint "${CHECKPOINT}" \
    --baseline image_only \
    --benchmarks vqav2 gqa pope \
    --spatial-eval \
    --data-root "${DATA_ROOT}" \
    --output "${OUTDIR}/image_only_k144.json"

# ── 4. Summary comparison table ──────────────────────────────────────────────
{
echo ""
echo "============================================================"
echo "  RESULTS SUMMARY — $(date)"
echo "============================================================"

python - "${OUTDIR}" <<'PYEOF'
import json
import sys
from pathlib import Path

outdir = Path(sys.argv[1])

models = [
    ("TC-ViTok k=144",    "tcvitok_k144.json"),
    ("Random drop k=144", "random_drop_k144.json"),
    ("Image-only TPN",    "image_only_k144.json"),
]

paper_targets = {
    "vqav2": 76.0,
    "gqa":   61.0,
    "gqa_s": 59.0,
    "gqa_g": 62.0,
    "pope":  84.0,
}

def fmt(v):
    if v is None:
        return "   N/A  "
    return f"{float(v):6.2f}%"

header = f"{'Model':<24} {'VQAv2':>8} {'GQA':>8} {'GQA-Sp':>8} {'GQA-Gl':>8} {'POPE-F1':>9}"
sep = "-" * len(header)
print(header)
print(sep)

for label, fname in models:
    fpath = outdir / fname
    if not fpath.exists():
        print(f"{label:<24} {'(not found — eval failed)':>43}")
        continue
    data = json.loads(fpath.read_text())
    b = data.get("benchmarks", data)
    vqav2 = b.get("vqav2", {}).get("vqa_score")
    gqa   = b.get("gqa",   {}).get("overall_accuracy")
    gqa_s = b.get("gqa",   {}).get("spatial_accuracy")
    gqa_g = b.get("gqa",   {}).get("global_accuracy")
    pope  = b.get("pope",  {})
    pope_f1 = pope.get("macro_f1")
    if pope_f1 is None:
        f1s = [pope.get(v, {}).get("f1") for v in ("popular", "random", "adversarial")]
        f1s = [x for x in f1s if x is not None]
        pope_f1 = sum(f1s) / len(f1s) if f1s else None
    print(f"{label:<24} {fmt(vqav2):>8} {fmt(gqa):>8} {fmt(gqa_s):>8} {fmt(gqa_g):>8} {fmt(pope_f1):>9}")

print(sep)
print(f"{'Paper targets':<24} {fmt(paper_targets['vqav2']):>8} "
      f"{fmt(paper_targets['gqa']):>8} "
      f"{fmt(paper_targets['gqa_s']):>8} "
      f"{fmt(paper_targets['gqa_g']):>8} "
      f"{fmt(paper_targets['pope']):>9}")
print()
PYEOF

echo "Results JSON : ${OUTDIR}/"
echo "Master log   : ${MASTER_LOG}"
echo "Per-eval logs: ${LOG_DIR}/eval_*_${TIMESTAMP}.log"
echo "============================================================"
} | tee -a "${MASTER_LOG}"
