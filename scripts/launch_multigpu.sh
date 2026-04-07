#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# launch_multigpu.sh — launch TC-ViTok training on 2× RTX 6000 Ada via torchrun
#
# Usage:
#   # Phase 0 only:
#   bash scripts/launch_multigpu.sh --config configs/phase0.yaml --output runs/
#
#   # Both phases back-to-back:
#   bash scripts/launch_multigpu.sh --config configs/phase0.yaml --output runs/ --run-both-phases
#
#   # Dry-run (2 steps, stub model, no data needed):
#   bash scripts/launch_multigpu.sh --config configs/phase0.yaml --output /tmp/test \
#       --stub --dummy-data --no-wandb --max-steps 2
#
# Environment variables (optional):
#   DATA_ROOT  — path to dataset root (Google Drive mount or local dir)
#                e.g. export DATA_ROOT=/mnt/gdrive/tc-vitok-data
#   HF_HOME    — HuggingFace cache dir (auto-set from DATA_ROOT if not given)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# ── GPU count ────────────────────────────────────────────────────────────────
NUM_GPUS=${NUM_GPUS:-$(python3 -c "import torch; print(torch.cuda.device_count())")}
echo "Launching on ${NUM_GPUS} GPU(s)"

# ── Python / env ─────────────────────────────────────────────────────────────
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1            # avoid CPU over-subscription in workers
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

# ── DATA_ROOT → HF_HOME wiring (done in train.py too, belt-and-suspenders) ──
if [[ -n "${DATA_ROOT:-}" ]]; then
    export HF_HOME="${HF_HOME:-${DATA_ROOT}/hf_cache}"
    export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${DATA_ROOT}/hf_cache}"
    export TORCH_HOME="${TORCH_HOME:-${DATA_ROOT}/torch_cache}"
    export HF_HUB_DISABLE_SYMLINKS_WARNING=1
    echo "DATA_ROOT  = ${DATA_ROOT}"
    echo "HF_HOME    = ${HF_HOME}"
    echo "TORCH_HOME = ${TORCH_HOME}"
fi

# ── Launch ───────────────────────────────────────────────────────────────────
torchrun \
    --standalone \
    --nproc_per_node="${NUM_GPUS}" \
    "${REPO_ROOT}/scripts/train.py" \
    "$@"
