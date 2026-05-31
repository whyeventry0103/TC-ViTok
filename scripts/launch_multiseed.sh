#!/usr/bin/env bash
# Launch short multi-seed training jobs. Keep heavy runs behind human approval.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 CONFIG_YAML OUTPUT_DIR [N_SEEDS]" >&2
  exit 1
fi

CONFIG=$1
OUT=$2
N=${3:-3}
GPU_COUNT=${GPU_COUNT:-2}

mkdir -p "$OUT"
for i in $(seq 0 $((N - 1))); do
  SEED=$((42 + 100 * i))
  GPU=$((i % GPU_COUNT))
  RUN_DIR="$OUT/seed${SEED}"
  mkdir -p "$RUN_DIR"
  echo "=== Launching seed ${SEED} on GPU ${GPU} ==="
  CUDA_VISIBLE_DEVICES=$GPU DATA_ROOT=${DATA_ROOT:-$HOME/tc-vitok-data} PYTHONPATH=src \
    python3 scripts/train.py \
      --config "$CONFIG" \
      --seed "$SEED" \
      --output "$RUN_DIR" \
      2>&1 | tee "$RUN_DIR/train.log" &
done

wait
echo "All seeds done."
