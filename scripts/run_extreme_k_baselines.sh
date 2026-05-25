#!/usr/bin/env bash
set -euo pipefail

# Experiment 1: extreme-low-k selector sweep.
#
# Usage:
#   GPU=0 scripts/run_extreme_k_baselines.sh
#
# Optional overrides:
#   CKPT=/path/to/best.ckpt
#   OUT=runs/extreme_k_baselines
#   DATA_ROOT=~/tc-vitok-data

GPU="${GPU:-0}"
OUT="${OUT:-runs/extreme_k_baselines}"
DATA_ROOT="${DATA_ROOT:-$HOME/tc-vitok-data}"
CKPT="${CKPT:-}"

if [[ -z "$CKPT" ]]; then
  for candidate in \
    runs/phase1_adaptive_v2/phase1/best.ckpt \
    runs/phase1_adaptive_v2_killed_20260512/phase1/best.ckpt \
    runs/phase1_adaptive/phase1/best.ckpt
  do
    if [[ -f "$candidate" ]]; then
      CKPT="$candidate"
      break
    fi
  done
fi

if [[ -z "$CKPT" || ! -f "$CKPT" ]]; then
  echo "Could not find adaptive checkpoint. Set CKPT=/path/to/best.ckpt." >&2
  exit 1
fi

mkdir -p "$OUT"
echo "Checkpoint: $CKPT"
echo "GPU: $GPU"
echo "Output: $OUT"

BASELINES=(none random_drop uniform_grid cls_attention)
for k in 8 18 24; do
  for baseline in "${BASELINES[@]}"; do
    json="$OUT/extreme_k${k}_${baseline}.json"
    log="$OUT/extreme_k${k}_${baseline}.log"
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
  --prefix extreme \
  --ks 8 18 24 \
  --csv "$OUT/summary.csv" \
  --plot "$OUT/k_scaling_extreme.pdf"
