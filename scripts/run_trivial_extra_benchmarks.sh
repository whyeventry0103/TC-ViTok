#!/usr/bin/env bash
set -euo pipefail

# Run Experiment 3: fill MMBench/TextVQA cells for trivial selectors.
#
# Usage:
#   GPU=1 scripts/run_trivial_extra_benchmarks.sh
#
# Environment overrides:
#   CKPT=runs/phase1_final/phase1/best.ckpt
#   OUT=runs/trivial_baselines_extra
#   DATA_ROOT=~/tc-vitok-data

GPU="${GPU:-0}"
CKPT="${CKPT:-runs/phase1_final/phase1/best.ckpt}"
OUT="${OUT:-runs/trivial_baselines_extra}"
DATA_ROOT="${DATA_ROOT:-$HOME/tc-vitok-data}"

if [[ ! -f "$CKPT" ]]; then
  echo "Could not find checkpoint: $CKPT" >&2
  exit 1
fi

BASELINES=(random_drop uniform_grid cls_attention)
BENCHMARKS=(mmbench textvqa)
mkdir -p "$OUT"

echo "Checkpoint: $CKPT"
echo "GPU: $GPU"
echo "Output: $OUT"

for baseline in "${BASELINES[@]}"; do
  for bench in "${BENCHMARKS[@]}"; do
    json="$OUT/${baseline}_${bench}.json"
    log="$OUT/${baseline}_${bench}.log"
    if [[ -f "$json" ]]; then
      echo "Skipping existing $json"
      continue
    fi
    echo "Running baseline=$baseline benchmark=$bench"
    CUDA_VISIBLE_DEVICES="$GPU" DATA_ROOT="$DATA_ROOT" python3 scripts/evaluate.py \
      --checkpoint "$CKPT" \
      --baseline "$baseline" \
      --benchmarks "$bench" \
      --max-samples 5000 \
      --output "$json" \
      2>&1 | tee "$log"
  done
done
