# TC-ViTok Remote Server Guide

Complete record of everything done to set up and run TC-ViTok on the remote server with 2× NVIDIA RTX 6000 Ada GPUs.

---

## Server Specs

```
GPU 0: NVIDIA RTX 6000 Ada Generation — 49,140 MiB VRAM
GPU 1: NVIDIA RTX 6000 Ada Generation — 49,140 MiB VRAM
CUDA: 12.4
Driver: 550.127.05
OS: Linux (Ubuntu)
Python: 3.10
User: sk10741
Host: ADUAED16811WKLX
Repo path: ~/TC-ViTok
Disk: 3.5TB total, ~199GB free (single partition at /)
```

---

## Storage Decision

Google Drive was initially considered but has a 15GB free limit — not enough for 55GB of data. The server has 199GB free on local disk which is more than sufficient, so **everything is stored locally on the server** at `~/tc-vitok-data`. Google Drive is not used.

---

## Environment Variables (set in ~/.bashrc)

```bash
export DATA_ROOT=~/tc-vitok-data
export HF_HOME=${DATA_ROOT}/hf_cache
export TORCH_HOME=${DATA_ROOT}/torch_cache
```

These are set permanently in `~/.bashrc`. All scripts and training code read from `DATA_ROOT` — changing this one variable is all you need to redirect everything.

---

## What Was Changed in the Codebase

### 1. Multi-GPU Support (DDP + BF16)

**`scripts/train.py`**
- Added `init_ddp()` — reads `LOCAL_RANK` env var set by `torchrun`, initializes NCCL process group
- Model wrapped in `nn.parallel.DistributedDataParallel` when multi-GPU
- DataLoaders use `DistributedSampler` for correct data sharding across GPUs
- All logging, checkpointing, and WandB calls gated to `is_main` (rank 0 only)
- `DATA_ROOT` env var automatically sets `HF_HOME` and `TORCH_HOME` before any imports

**`src/tc_vitok/training/trainer.py`**
- Accepts `local_rank` parameter, detects DDP via `model.module`
- Forward pass wrapped in `torch.autocast(dtype=torch.bfloat16)` — RTX 6000 Ada supports BF16 natively
- Checkpoints always save `model.module.state_dict()` (plain, not DDP-wrapped)
- All prints and saves gated to `is_main`

**`configs/base.yaml`**
- `batch_size: 32` (from 16) — per GPU
- `grad_accum_steps: 2` — effective batch = 32 × 2 × 2 GPUs = 128
- `num_workers: 8`
- `mixed_precision: "bf16"`
- All data paths use OmegaConf env-var interpolation: `${oc.env:DATA_ROOT,data}`

### 2. Path Integration

**`configs/base.yaml`**
```yaml
data_root: ${oc.env:DATA_ROOT,data}
train_image_dir: "${data.data_root}/vqav2/train2014"
# all paths follow this pattern
```

**`scripts/train.py`**
```python
if data_root:
    os.environ.setdefault("HF_HOME", f"{data_root}/hf_cache")
    os.environ.setdefault("TORCH_HOME", f"{data_root}/torch_cache")
```

### 3. New Scripts Created

| Script | Purpose |
|---|---|
| `scripts/launch_multigpu.sh` | Runs `torchrun --nproc_per_node=2` with all env vars set |
| `scripts/setup_gdrive.sh` | rclone-based Google Drive setup (not used, kept for reference) |
| `scripts/download_assets.sh` | Downloads all models and datasets to `$DATA_ROOT` |

### 4. All Bugs Fixed

| Bug | Was | Fixed to |
|---|---|---|
| DINOv2 loader | HuggingFace `facebook/dinov2-small` | `timm.create_model("vit_small_patch14_dinov2.lvd142m")` |
| CLIP for FullVisualEncoder | `openai/clip-vit-large-patch14` | `openai/clip-vit-large-patch14-336` |
| CLIP for TextEncoder | not downloaded at all | `openai/clip-vit-base-patch32` |
| LLaVA download class | `AutoModelForCausalLM` | `LlavaForConditionalGeneration` |
| Script aborts on any error | `set -euo pipefail` | `set -uo pipefail` (removed `-e`) |
| Large downloads restart from 0 | no resume flag | `wget -c --tries=5 --timeout=120` |
| COCO image URL | `http://` | `https://` |
| numpy conflict with system packages | numpy 2.2.6 | `pip install "numpy<2.0"` |
| `AutoFeatureExtractor` removed in transformers 5.x | `AutoFeatureExtractor` | `AutoImageProcessor` |
| `HF_HOME` still pointing to Drive after switch | old `.bashrc` line | `sed -i '/HF_HOME/d' ~/.bashrc` then re-added correctly |
| `TORCH_HOME` not set | missing from `.bashrc` | added `export TORCH_HOME=${DATA_ROOT}/torch_cache` |

---

## Part 1 — Initial Setup (Done, do not repeat)

### Install Python packages
```bash
cd ~/TC-ViTok
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install "numpy<2.0" --force-reinstall
```

### Set environment variables permanently
```bash
echo 'export DATA_ROOT=~/tc-vitok-data' >> ~/.bashrc
echo 'export HF_HOME=${DATA_ROOT}/hf_cache' >> ~/.bashrc
echo 'export TORCH_HOME=${DATA_ROOT}/torch_cache' >> ~/.bashrc
source ~/.bashrc
```

Verify:
```bash
echo $DATA_ROOT    # /home/sk10741/tc-vitok-data
echo $HF_HOME      # /home/sk10741/tc-vitok-data/hf_cache
echo $TORCH_HOME   # /home/sk10741/tc-vitok-data/torch_cache
```

---

## Part 2 — Download Everything (Currently Running)

Download is running in tmux session `download3`. Started on 2026-04-08.

Check progress:
```bash
tmux capture-pane -t download3 -p | tail -20
```

Done when you see:
```
──────────────────────────────────────────
 All downloads complete
──────────────────────────────────────────
```

### What gets downloaded

| Item | Size | Destination |
|---|---|---|
| DINOv2-S (via timm) | ~85 MB | `~/tc-vitok-data/torch_cache/` |
| CLIP ViT-B/32 | ~340 MB | `~/tc-vitok-data/hf_cache/` |
| CLIP ViT-L/14-336 | ~1.7 GB | `~/tc-vitok-data/hf_cache/` |
| LLaVA-1.5-7B | ~14 GB | `~/tc-vitok-data/hf_cache/` |
| VQAv2 annotations | ~130 MB | `~/tc-vitok-data/vqav2/` |
| COCO train2014 images | ~13 GB | `~/tc-vitok-data/vqav2/train2014/` |
| COCO val2014 images | ~6 GB | `~/tc-vitok-data/vqav2/val2014/` |
| GQA questions | ~660 MB | `~/tc-vitok-data/gqa/` |
| GQA images | ~20 GB | `~/tc-vitok-data/gqa/images/` |
| **Total** | **~55 GB** | |

### If download crashes midway
Just re-run — skips already-downloaded files:
```bash
tmux attach -t download3
bash scripts/download_assets.sh
# or start a new session:
tmux new -s download4
bash scripts/download_assets.sh
```

### Verify downloads after completion
```bash
ls ~/tc-vitok-data/hf_cache/
ls ~/tc-vitok-data/torch_cache/
ls ~/tc-vitok-data/vqav2/
ls ~/tc-vitok-data/gqa/
```

---

## Part 3 — Training (Run After Downloads Complete)

### Step 1 — Verify env vars are set
```bash
echo $DATA_ROOT
echo $HF_HOME
echo $TORCH_HOME
```
If any are empty, run `source ~/.bashrc` first.

### Step 2 — Dry run (verify pipeline works, ~30 seconds, no real data)
```bash
cd ~/TC-ViTok
PYTHONPATH=src bash scripts/launch_multigpu.sh \
    --config configs/phase0.yaml \
    --output /tmp/tc_vitok_dryrun \
    --stub --dummy-data --no-wandb --max-steps 5
```

Expected output:
```
DDP enabled — local_rank=0, world_size=2
Mixed precision: torch.bfloat16
Starting Phase 0 Training
```

### Step 3 — Real training (both phases, unattended)
```bash
tmux new -s training
export DATA_ROOT=~/tc-vitok-data
bash scripts/launch_multigpu.sh \
    --config configs/phase0.yaml \
    --output runs/ \
    --run-both-phases \
    --no-wandb
```
Detach: `Ctrl+B` then `D`

Check progress:
```bash
tmux capture-pane -t training -p | tail -20
```

### What training does
- **Phase 0** (10,000 steps): Trains TPN only, LLaVA frozen, VQAv2 data
- **Phase 1** (50,000 steps): Trains TPN + LoRA + MLP jointly, VQAv2 + GQA data
- Checkpoints saved to `runs/phase0/` and `runs/phase1/` every 5,000 steps
- Best checkpoint: `runs/phase0/best.ckpt`, `runs/phase1/best.ckpt`

### Resume from checkpoint
```bash
tmux new -s training
export DATA_ROOT=~/tc-vitok-data
bash scripts/launch_multigpu.sh \
    --config configs/phase0.yaml \
    --output runs/ \
    --resume runs/phase0/checkpoint_step_5000.pt \
    --no-wandb
```

### Config reference
| Parameter | Value | Meaning |
|---|---|---|
| `k` | 144 | Tokens selected out of 576 |
| `batch_size` | 32 | Per GPU |
| `grad_accum_steps` | 2 | Effective batch = 128 total |
| `mixed_precision` | `"bf16"` | BF16 autocast |
| `max_steps` (Phase 0) | 10,000 | ~1 epoch VQAv2 |
| `max_steps` (Phase 1) | 50,000 | ~3 epochs VQAv2+GQA |

---

## Part 4 — After Every Reconnect

Env vars in `~/.bashrc` persist across sessions automatically. Just verify on reconnect:
```bash
echo $DATA_ROOT   # should print /home/sk10741/tc-vitok-data
```
If empty: `source ~/.bashrc`

No rclone mount needed — data is on local disk.

---

## Useful Commands

```bash
# Check both GPUs
nvidia-smi

# Check disk space
df -h ~

# Check what's been downloaded
ls ~/tc-vitok-data/
ls ~/tc-vitok-data/hf_cache/
ls ~/tc-vitok-data/vqav2/
ls ~/tc-vitok-data/gqa/

# List all tmux sessions
tmux ls

# Attach to a session
tmux attach -t download3
tmux attach -t training

# Check session output without attaching
tmux capture-pane -t download3 -p | tail -20
tmux capture-pane -t training -p | tail -20

# Kill a tmux session
tmux kill-session -t <session-name>

# Send Ctrl+C to a tmux session without attaching
tmux send-keys -t <session-name> C-c Enter
```

---

## Troubleshooting

### Env vars missing after reconnect
```bash
source ~/.bashrc
```

### CUDA out of memory
```yaml
# in configs/base.yaml:
training:
  batch_size: 16
  grad_accum_steps: 4   # keeps effective batch = 128
```

### NCCL / DDP error
```bash
python3 -c "import torch; print(torch.cuda.device_count())"  # must print 2
export NCCL_DEBUG=INFO
PYTHONPATH=src bash scripts/launch_multigpu.sh \
    --config configs/phase0.yaml \
    --output /tmp/t --stub --dummy-data --no-wandb --max-steps 2
```

### Download crashed midway
```bash
tmux new -s download4
export DATA_ROOT=~/tc-vitok-data
bash scripts/download_assets.sh   # skips completed files automatically
```

### numpy/pandas conflict
```bash
pip install "numpy<2.0" --force-reinstall
```

### HuggingFace model not found during training
```bash
export DATA_ROOT=~/tc-vitok-data
bash scripts/download_assets.sh --models-only
```

### Check if a specific model downloaded correctly
```bash
ls ~/tc-vitok-data/hf_cache/models--llava-hf--llava-1.5-7b-hf/
ls ~/tc-vitok-data/hf_cache/models--openai--clip-vit-large-patch14-336/
ls ~/tc-vitok-data/hf_cache/models--openai--clip-vit-base-patch32/
ls ~/tc-vitok-data/torch_cache/hub/
```
