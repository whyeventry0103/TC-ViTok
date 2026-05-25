#!/usr/bin/env bash
set -euo pipefail

# Run Experiment 1: TPN vs trivial selectors across token budgets.
#
# Usage examples:
#   GPU=0 scripts/run_multi_k_baselines.sh 36 72 144
#   GPU=1 scripts/run_multi_k_baselines.sh 216 288
#
# Environment overrides:
#   CKPT=/path/to/best.ckpt
#   OUT=runs/multi_k_baselines
#   DATA_ROOT=~/tc-vitok-data

GPU="${GPU:-0}"
OUT="${OUT:-runs/multi_k_baselines}"
DATA_ROOT="${DATA_ROOT:-$HOME/tc-vitok-data}"
CKPT="${CKPT:-}"

if [[ -z "$CKPT" ]]; then
  for candidate in \
    runs/phase1_adaptive_v2/phase1/best.ckpt \
    runs/phase1_adaptive_v2_killed_20260512/phase1/best.ckpt \
    runs/phase1_adaptive/phase1/best.ckpt \
    runs/phase1_final/phase1/best.ckpt
  do
    if [[ -f "$candidate" ]]; then
      CKPT="$candidate"
      break
    fi
  done
fi

if [[ -z "$CKPT" || ! -f "$CKPT" ]]; then
  echo "Could not find checkpoint. Set CKPT=/path/to/best.ckpt." >&2
  exit 1
fi

if [[ "$#" -gt 0 ]]; then
  KS=("$@")
else
  KS=(36 72 144 216 288)
fi

BASELINES=(none random_drop uniform_grid cls_attention)
mkdir -p "$OUT"

echo "Checkpoint: $CKPT"
echo "GPU: $GPU"
echo "Output: $OUT"
echo "k values: ${KS[*]}"

for k in "${KS[@]}"; do
  for baseline in "${BASELINES[@]}"; do
    json="$OUT/adaptive_k${k}_${baseline}.json"
    log="$OUT/adaptive_k${k}_${baseline}.log"
    if [[ -f "$json" ]]; then
      echo "Skipping existing $json"
      continue
    fi
    echo "Running k=$k baseline=$baseline"
    CUDA_VISIBLE_DEVICES="$GPU" DATA_ROOT="$DATA_ROOT" python3 scripts/evaluate.py \
      --checkpoint "$CKPT" \
      --baseline "$baseline" \
      --k "$k" \
      --no-auto-adaptive-budget \
      --benchmarks vqav2 gqa pope \
      --max-samples 5000 \
      --output "$json" \
      2>&1 | tee "$log"
  done
done

python3 scripts/summarize_multi_k_baselines.py \
  --input-dir "$OUT" \
  --csv "$OUT/summary.csv" \
  --plot "$OUT/k_scaling.pdf"
