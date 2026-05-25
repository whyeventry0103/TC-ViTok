#!/usr/bin/env bash
set -euo pipefail

# Experiment 3: random-drop seed variance at k=144.
#
# Usage:
#   GPU=1 scripts/run_random_seed_variance.sh
#
# Optional overrides:
#   CKPT=runs/phase1_final/phase1/best.ckpt
#   OUT=runs/random_seed_variance
#   DATA_ROOT=~/tc-vitok-data

GPU="${GPU:-0}"
CKPT="${CKPT:-runs/phase1_final/phase1/best.ckpt}"
OUT="${OUT:-runs/random_seed_variance}"
DATA_ROOT="${DATA_ROOT:-$HOME/tc-vitok-data}"

if [[ ! -f "$CKPT" ]]; then
  echo "Could not find checkpoint: $CKPT" >&2
  exit 1
fi

mkdir -p "$OUT"
echo "Checkpoint: $CKPT"
echo "GPU: $GPU"
echo "Output: $OUT"

for seed in 7 13 42 100 1000; do
  json="$OUT/random_seed${seed}.json"
  log="$OUT/random_seed${seed}.log"
  if [[ -f "$json" ]]; then
    echo "Skipping existing $json"
    continue
  fi
  echo "Running random_drop seed=$seed"
  CUDA_VISIBLE_DEVICES="$GPU" DATA_ROOT="$DATA_ROOT" python3 scripts/evaluate.py \
    --checkpoint "$CKPT" \
    --baseline random_drop \
    --seed "$seed" \
    --k 144 \
    --benchmarks vqav2 gqa pope \
    --max-samples 5000 \
    --output "$json" \
    2>&1 | tee "$log"
done

python3 scripts/analyze_random_variance.py | tee "$OUT/analysis.txt"
