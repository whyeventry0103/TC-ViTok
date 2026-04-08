# TC-ViTok Remote Server Guide

Complete record of everything done to set up and run TC-ViTok on the remote server with 2× NVIDIA RTX 6000 Ada GPUs.

---

## Server Specs

```
GPU 0: NVIDIA RTX 6000 Ada Generation — 49,140 MiB VRAM
GPU 1: NVIDIA RTX 6000 Ada Generation — 49,140 MiB VRAM
CUDA: 12.4
Driver: 550.127.05
OS: Linux (Ubuntu 22.04)
Python: 3.10
User: sk10741
Host: ADUAED16811WKLX
Repo path: ~/TC-ViTok
Disk: 3.5TB total, ~190GB free (single partition at /)
aria2c: installed at /usr/bin/aria2c (v1.36.0)
```

---

## Storage Decision

Google Drive was initially considered but has a 15GB free limit — not enough for 55GB of data. The server has ~190GB free on local disk which is more than sufficient, so **everything is stored locally on the server** at `~/tc-vitok-data`. Google Drive is not used.

---

## Environment Variables (set in ~/.bashrc)

```bash
export DATA_ROOT=~/tc-vitok-data
export HF_HOME=${DATA_ROOT}/hf_cache
export TORCH_HOME=${DATA_ROOT}/torch_cache
```

These are set permanently in `~/.bashrc`. All scripts and training code read from `DATA_ROOT`.

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

### 2. New Scripts Created

| Script | Purpose |
|---|---|
| `scripts/launch_multigpu.sh` | Runs `torchrun --nproc_per_node=2` with all env vars set |
| `scripts/setup_gdrive.sh` | rclone-based Google Drive setup (not used, kept for reference) |
| `scripts/download_assets.sh` | Downloads all models and datasets to `$DATA_ROOT` |

### 3. All Bugs Fixed in download_assets.sh

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
| COCO images unzip failure | `wget -P` (directory prefix) saved to wrong path | `wget -O "${ZIP}"` (explicit output file) + zip validation before extract |
| Corrupt zip re-attempted on re-run | no validation check | `unzip -t` check before download; deletes corrupt zip automatically |
| COCO SSL certificate error | no cert flag | `wget --no-check-certificate` |
| GQA questions individual JSON URLs | direct JSON URLs return 404 (removed from Stanford) | download `questions1.2.zip` bundle instead |
| GQA questions 503 with aria2c 16 connections | Stanford throttles parallel connections | aria2c falls back to 3 working connections; or wget fallback |
| COCO images slow at CDN (< 1 KB/s with single connection) | single wget connection throttled | `aria2c -x 16 -s 16` for fast parallel download (~60 MB/s) |

---

## Part 1 — Initial Setup (Done, do not repeat)

### Install Python packages
```bash
cd ~/TC-ViTok
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install "numpy<2.0" --force-reinstall
sudo apt-get install -y aria2   # for fast parallel downloads
```

### Set environment variables permanently
```bash
echo 'export DATA_ROOT=~/tc-vitok-data' >> ~/.bashrc
echo 'export HF_HOME=${DATA_ROOT}/hf_cache' >> ~/.bashrc
echo 'export TORCH_HOME=${DATA_ROOT}/torch_cache' >> ~/.bashrc
source ~/.bashrc
```

---

## Part 2 — Downloads (Final Status as of 2026-04-08)

| Item | Size | Destination | Status |
|---|---|---|---|
| DINOv2-S (via timm/HF) | ~85 MB | `~/tc-vitok-data/hf_cache/` | ✓ Done |
| CLIP ViT-B/32 | ~340 MB | `~/tc-vitok-data/hf_cache/` | ✓ Done |
| CLIP ViT-L/14-336 | ~1.7 GB | `~/tc-vitok-data/hf_cache/` | ✓ Done |
| LLaVA-1.5-7B | ~14 GB | `~/tc-vitok-data/hf_cache/` | ✓ Done |
| VQAv2 annotations (4 JSON files) | ~130 MB | `~/tc-vitok-data/vqav2/` | ✓ Done |
| COCO train2014 (82,783 images) | ~13 GB | `~/tc-vitok-data/vqav2/train2014/` | ✓ Done |
| COCO val2014 (40,504 images) | ~6 GB | `~/tc-vitok-data/vqav2/val2014/` | ✓ Done |
| GQA images (148,854 images) | ~20 GB | `~/tc-vitok-data/gqa/images/` | ✓ Done |
| GQA questions (questions1.2.zip) | ~1.4 GB | `~/tc-vitok-data/gqa/` | Downloading (aria2c, ~2.5h) |
| **Total** | **~55 GB** | | |

### How COCO images were downloaded (fast method)
Single wget connections to `images.cocodataset.org` were throttled to < 1 KB/s. Fixed by using aria2c with 16 parallel connections:
```bash
aria2c -x 16 -s 16 --check-certificate=false \
    -d ~/tc-vitok-data/vqav2/ \
    -o train2014.zip \
    "https://images.cocodataset.org/zips/train2014.zip"
unzip ~/tc-vitok-data/vqav2/train2014.zip -d ~/tc-vitok-data/vqav2/
rm -f ~/tc-vitok-data/vqav2/train2014.zip
```
Achieved ~58 MB/s (finished in ~4 minutes for 13 GB).

### How GQA questions are downloaded
Individual JSON URLs (e.g. `train_balanced_questions.json`) return 404 — Stanford removed them. The zip bundle still exists:
```bash
export DATA_ROOT=~/tc-vitok-data
bash scripts/download_assets.sh --gqa-only
```
The script downloads `questions1.2.zip` via aria2c (falls back to wget if 503). Stanford throttles to ~150 KB/s for this file.

### Git pull auth issue on server
VS Code's credential helper interferes with git. Use:
```bash
git -c credential.helper= pull origin main
# If that also fails (VS Code socket still intercepting), use rsync from local Mac:
rsync -avz --exclude='.git' /path/to/local/TC-ViTok/ sk10741@ADUAED16811WKLX:~/TC-ViTok/
```

### Verify all downloads
```bash
ls ~/tc-vitok-data/vqav2/train2014/ | wc -l   # expect 82783
ls ~/tc-vitok-data/vqav2/val2014/ | wc -l      # expect 40504
ls ~/tc-vitok-data/gqa/images/ | wc -l         # expect ~148854
ls ~/tc-vitok-data/gqa/*.json 2>/dev/null | wc -l  # expect 3
ls ~/tc-vitok-data/hf_cache/models--llava-hf--llava-1.5-7b-hf/
ls ~/tc-vitok-data/hf_cache/models--openai--clip-vit-large-patch14-336/
ls ~/tc-vitok-data/hf_cache/models--openai--clip-vit-base-patch32/
```

### If any download is missing
```bash
export DATA_ROOT=~/tc-vitok-data
bash scripts/download_assets.sh              # re-run everything (skips completed)
bash scripts/download_assets.sh --models-only
bash scripts/download_assets.sh --vqav2-only
bash scripts/download_assets.sh --gqa-only
```

---

## Part 3 — Training (Run After Downloads Complete)

### Step 1 — Verify env vars
```bash
echo $DATA_ROOT   # /home/sk10741/tc-vitok-data
echo $HF_HOME
echo $TORCH_HOME
```
If empty: `source ~/.bashrc`

### Step 2 — Dry run (~30 seconds, no real data needed)
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

```bash
echo $DATA_ROOT   # should print /home/sk10741/tc-vitok-data
```
If empty: `source ~/.bashrc`

---

## Useful Commands

```bash
# Check both GPUs
nvidia-smi

# Check disk space
df -h ~

# Check downloaded data
ls ~/tc-vitok-data/vqav2/train2014/ | wc -l
ls ~/tc-vitok-data/vqav2/val2014/ | wc -l
ls ~/tc-vitok-data/gqa/images/ | wc -l
ls ~/tc-vitok-data/gqa/*.json | wc -l

# tmux
tmux ls
tmux attach -t <session>
tmux capture-pane -t <session> -p | tail -20
tmux kill-session -t <session>
```

---

## Troubleshooting

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

### COCO images missing / slow download
Single connections to `images.cocodataset.org` are throttled to < 1 KB/s. Always use aria2c:
```bash
aria2c -x 16 -s 16 --check-certificate=false \
    -d ~/tc-vitok-data/vqav2/ -o train2014.zip \
    "https://images.cocodataset.org/zips/train2014.zip"
unzip ~/tc-vitok-data/vqav2/train2014.zip -d ~/tc-vitok-data/vqav2/
rm -f ~/tc-vitok-data/vqav2/train2014.zip
# repeat for val2014.zip
```

### GQA questions missing (Stanford 404)
Individual JSON URLs no longer exist. Use the script which downloads the zip bundle:
```bash
export DATA_ROOT=~/tc-vitok-data
bash scripts/download_assets.sh --gqa-only
```

### HuggingFace model not found during training
```bash
export DATA_ROOT=~/tc-vitok-data
bash scripts/download_assets.sh --models-only
```

### numpy/pandas conflict
```bash
pip install "numpy<2.0" --force-reinstall
```

### git pull fails (VS Code credential socket error)
```bash
git -c credential.helper= pull origin main
```
If still failing, push from Mac and rsync to server manually.
