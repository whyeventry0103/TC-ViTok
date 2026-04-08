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

# ── Kill zombie GPU processes from prior crashed runs ────────────────────────
# A previous SIGABRT can leave python/torchrun holding GPU memory, causing the
# next run to OOM on model.to(device) → DDP param mismatch across ranks.
echo "Checking for leftover GPU processes..."
if command -v nvidia-smi &>/dev/null; then
    # Get PIDs of processes using the GPUs (skip header line)
    _gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u || true)
    if [[ -n "$_gpu_pids" ]]; then
        echo "WARNING: Found processes on GPU: $_gpu_pids"
        echo "Killing leftover GPU processes to free VRAM..."
        for _pid in $_gpu_pids; do
            # Only kill python/torchrun processes (safety: don't kill other users' work)
            _cmd=$(ps -p "$_pid" -o comm= 2>/dev/null || true)
            if [[ "$_cmd" == *python* ]] || [[ "$_cmd" == *torchrun* ]]; then
                echo "  Killing PID $_pid ($_cmd)"
                kill -9 "$_pid" 2>/dev/null || true
            else
                echo "  Skipping PID $_pid ($_cmd) — not a python/torchrun process"
            fi
        done
        sleep 2  # wait for GPU memory to be released
        echo "GPU cleanup done."
    else
        echo "GPUs are clean — no leftover processes."
    fi
fi

# ── GPU count ────────────────────────────────────────────────────────────────
NUM_GPUS=${NUM_GPUS:-$(python3 -c "import torch; print(torch.cuda.device_count())")}
echo "Launching on ${NUM_GPUS} GPU(s)"

# ── Python / env ─────────────────────────────────────────────────────────────
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1            # avoid CPU over-subscription in workers
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1  # surface NCCL errors as Python exceptions (not SIGABRT)
export NCCL_TIMEOUT=1800                  # 30-min timeout before NCCL gives up on a stalled rank

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

# ── Log file (captures full traceback for post-mortem debugging) ─────────────
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to: ${LOG_FILE}"

# ── Launch ───────────────────────────────────────────────────────────────────
torchrun \
    --standalone \
    --nproc_per_node="${NUM_GPUS}" \
    "${REPO_ROOT}/scripts/train.py" \
    "$@" 2>&1 | tee "${LOG_FILE}"
