# TC-ViTok GPU Training Guide

Complete, step-by-step instructions for training on the remote server (2× NVIDIA RTX 6000 Ada, 49 GB VRAM each) using Google Drive as the data/model store.

---

## The Big Picture: Why Google Drive and How It Works

The challenge: datasets (~40 GB) and models (~15 GB) are large. Uploading them from your laptop to the remote server is slow and wasteful if you ever change machines.

The solution:
1. Mount your Google Drive directly on the remote server using `rclone` — it appears as a local filesystem at `/mnt/gdrive`.
2. Run `download_assets.sh` on the remote server. It downloads **directly from the internet into the mounted Drive folder** — your laptop is not involved at all.
3. Training reads data straight from that Drive mount as if it were a local disk.
4. HuggingFace models (LLaVA-1.5-7B, CLIP, DINOv2) are also downloaded into Drive. Once downloaded, any machine that mounts your Drive has them instantly.

You never manually upload anything from your laptop.

---

## Prerequisites

### On the remote server, verify:
```bash
# Python (need 3.10+)
python3 --version

# CUDA toolkit
nvcc --version    # should show 12.x

# GPUs
nvidia-smi        # should show 2× RTX 6000 Ada

# Basic tools
wget --version
unzip -v
curl --version
```

### Python environment
The remote server needs a conda/venv environment with all packages. If starting fresh:
```bash
# Option A: conda (recommended)
conda create -n tc-vitok python=3.11 -y
conda activate tc-vitok

# Option B: venv
python3 -m venv ~/envs/tc-vitok
source ~/envs/tc-vitok/bin/activate
```

Install all dependencies (do this inside the repo directory):
```bash
cd ~/TC-ViTok    # wherever the repo is on the remote server
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

> Note: Install PyTorch first with the correct CUDA wheel (`cu124` matches CUDA 12.4 on the server), then install the rest from `requirements.txt`.

---

## PART 1 — Set Up Google Drive on the Remote Server

This is a one-time setup. After this, Drive stays mounted and survives across training runs (as long as you re-mount after a reboot).

### Step 1.1 — Run the setup script
```bash
cd ~/TC-ViTok
bash scripts/setup_gdrive.sh
```

This installs `rclone` if missing, then launches the interactive config wizard. Follow these steps exactly inside the wizard:

```
> n                              # new remote
Name: gdrive                     # type exactly: gdrive
Storage type: 18                 # or type 'drive' — Google Drive
client_id: (leave blank, press Enter)
client_secret: (leave blank, press Enter)
scope: 1                         # Full access
root_folder_id: (leave blank, press Enter)
service_account_file: (leave blank, press Enter)
Edit advanced config? n
Use auto config? n               # ← IMPORTANT: say n (headless server)
```

At this point rclone prints a URL like:
```
https://accounts.google.com/o/oauth2/auth?...
```

Open that URL in your **local browser** (not on the server), sign into the Google account where your Drive lives, click Allow. Copy the verification code shown and paste it back into the terminal on the server.

```
> (paste the code here)
Configure as Shared Drive? n
y                                # confirm and save
q                                # quit config
```

**Verify it worked:**
```bash
rclone lsd gdrive:
# Should list your Google Drive folders
```

### Step 1.2 — Mount Google Drive
```bash
bash scripts/setup_gdrive.sh --mount
```

This runs `rclone mount` in daemon mode. Drive is now accessible at `/mnt/gdrive`.

Verify:
```bash
ls /mnt/gdrive
# Should list the contents of your Google Drive root
```

### Step 1.3 — Create the data folder in Drive
```bash
mkdir -p /mnt/gdrive/tc-vitok-data
```

This folder will hold all datasets and model weights. It's a real folder on your Google Drive — you can see it from drive.google.com.

### Step 1.4 — Set the environment variable
Add these two lines to your shell profile on the remote server so they persist across sessions:
```bash
echo 'export DATA_ROOT=/mnt/gdrive/tc-vitok-data' >> ~/.bashrc
echo 'export HF_HOME=${DATA_ROOT}/hf_cache' >> ~/.bashrc
source ~/.bashrc
```

Check:
```bash
echo $DATA_ROOT
# /mnt/gdrive/tc-vitok-data
```

---

## PART 2 — Download Datasets and Models

Everything downloads directly to Google Drive. Run this on the remote server.

### Step 2.1 — Download everything at once (~55 GB total)
```bash
cd ~/TC-ViTok
export DATA_ROOT=/mnt/gdrive/tc-vitok-data
bash scripts/download_assets.sh
```

This downloads in order:
1. DINOv2-S weights via HuggingFace (~85 MB)
2. CLIP ViT-L/14 weights via HuggingFace (~890 MB)
3. LLaVA-1.5-7B weights via HuggingFace (~14 GB) — this is the slow one
4. VQAv2 annotations (4 JSON files, ~130 MB total)
5. COCO train2014 images (~13 GB)
6. COCO val2014 images (~6 GB)
7. GQA question files (~660 MB)
8. GQA images (~20 GB)

Or download only specific parts:
```bash
bash scripts/download_assets.sh --models-only    # only DINOv2 + CLIP + LLaVA
bash scripts/download_assets.sh --vqav2-only     # only VQAv2 annotations + images
bash scripts/download_assets.sh --gqa-only       # only GQA questions + images
```

### What the folder structure looks like after download

On the remote server (via the mount):
```
/mnt/gdrive/tc-vitok-data/
├── hf_cache/                          ← HuggingFace model weights
│   ├── models--facebook--dinov2-small/
│   ├── models--openai--clip-vit-large-patch14/
│   └── models--llava-hf--llava-1.5-7b-hf/
├── vqav2/
│   ├── v2_mscoco_train2014_annotations.json
│   ├── v2_OpenEnded_mscoco_train2014_questions.json
│   ├── v2_mscoco_val2014_annotations.json
│   ├── v2_OpenEnded_mscoco_val2014_questions.json
│   ├── train2014/                     ← 83,000 JPEG images
│   └── val2014/                       ← 40,000 JPEG images
└── gqa/
    ├── train_balanced_questions.json
    ├── val_balanced_questions.json
    ├── testdev_balanced_questions.json
    └── images/                        ← 113,000 images
```

In your Google Drive (visible at drive.google.com):
```
My Drive/
└── tc-vitok-data/        ← same folder, accessible anywhere
    ├── hf_cache/
    ├── vqav2/
    └── gqa/
```

### How the code finds the data

`configs/base.yaml` uses OmegaConf env-var interpolation:
```yaml
data_root: ${oc.env:DATA_ROOT,data}
train_image_dir: "${data.data_root}/vqav2/train2014"
```

When `DATA_ROOT=/mnt/gdrive/tc-vitok-data`, all paths resolve to the Drive mount automatically. If `DATA_ROOT` is not set, they fall back to `./data/` relative to the repo.

`scripts/train.py` also sets `HF_HOME` automatically:
```python
if data_root:
    hf_home = os.path.join(data_root, "hf_cache")
    os.environ.setdefault("HF_HOME", hf_home)
```

So models are found in Drive's `hf_cache/` folder without any extra configuration.

---

## PART 3 — Verify the Setup (Dry Run)

Before committing to a full training run, verify the full pipeline works. This requires no data and no real model — it uses a stub LLM and synthetic data.

```bash
cd ~/TC-ViTok
export PYTHONPATH=src

PYTHONPATH=src python scripts/train.py \
    --config configs/phase0.yaml \
    --output /tmp/tc_vitok_dryrun \
    --stub \
    --dummy-data \
    --no-wandb \
    --max-steps 5
```

Expected output:
```
Loaded config from configs/phase0.yaml
Phase: 0
Device: cuda:0
DDP enabled — local_rank=0, world_size=1
Model created (stub=True, k=144)
Trainable params: ...
============================================================
Starting Phase 0 Training
============================================================
Max steps: 5
Mixed precision: torch.bfloat16
...
Phase 0 Training Complete
```

If you see this, the pipeline is working.

For a DDP dry run (both GPUs):
```bash
bash scripts/launch_multigpu.sh \
    --config configs/phase0.yaml \
    --output /tmp/tc_vitok_dryrun_ddp \
    --stub \
    --dummy-data \
    --no-wandb \
    --max-steps 5
```

Expected: same output but shows `DDP world size: 2`.

---

## PART 4 — Running Real Training

### Step 4.1 — (Optional) Set up WandB
WandB tracks loss curves, logs, and checkpoints online.
```bash
pip install wandb
wandb login
# Enter your API key from wandb.ai/authorize
```

If you don't want WandB, always pass `--no-wandb`.

### Step 4.2 — Phase 0 only (TPN warmup, ~10k steps)
```bash
export DATA_ROOT=/mnt/gdrive/tc-vitok-data

bash scripts/launch_multigpu.sh \
    --config configs/phase0.yaml \
    --output runs/ \
    --no-wandb
```

What this does:
- Runs `torchrun --nproc_per_node=2` automatically (detects 2 GPUs)
- Loads VQAv2 from Drive via `DATA_ROOT`
- Trains only the TPN (LLaVA frozen), BF16, 32 samples/GPU/step
- Saves checkpoints to `runs/phase0/checkpoint_step_*.pt` every 5000 steps
- Saves best checkpoint to `runs/phase0/best.ckpt`

### Step 4.3 — Both phases back-to-back (recommended)
```bash
export DATA_ROOT=/mnt/gdrive/tc-vitok-data

bash scripts/launch_multigpu.sh \
    --config configs/phase0.yaml \
    --output runs/ \
    --run-both-phases \
    --no-wandb
```

What this does:
- Runs Phase 0 (TPN warmup, 10k steps on VQAv2)
- Automatically transitions to Phase 1 (joint finetuning TPN + LoRA + MLP, 50k steps on VQAv2 + GQA)
- Phase 1 resumes from `runs/phase0/best.ckpt`

### Step 4.4 — Resuming from a checkpoint
If training was interrupted:
```bash
export DATA_ROOT=/mnt/gdrive/tc-vitok-data

bash scripts/launch_multigpu.sh \
    --config configs/phase0.yaml \
    --output runs/ \
    --resume runs/phase0/checkpoint_step_5000.pt \
    --no-wandb
```

### Step 4.5 — With WandB enabled
```bash
export DATA_ROOT=/mnt/gdrive/tc-vitok-data

bash scripts/launch_multigpu.sh \
    --config configs/phase0.yaml \
    --output runs/ \
    --run-both-phases \
    --wandb-project tc-vitok
```

---

## PART 5 — Configuration Reference

All training hyperparameters live in `configs/`. The active config is `configs/base.yaml`, which phase configs override selectively.

### Key values in `configs/base.yaml` (already set for these GPUs):
| Parameter | Value | Meaning |
|---|---|---|
| `mixed_precision` | `"bf16"` | BF16 autocast — RTX 6000 Ada supports this natively |
| `batch_size` | `32` | Per-GPU batch size |
| `grad_accum_steps` | `2` | Effective batch = 32 × 2 × 2 GPUs = 128 |
| `num_workers` | `8` | DataLoader workers (4 per GPU) |
| `k` | `144` | Token budget (144 of 576 visual tokens selected) |
| `max_steps` | `50000` | Phase 1 training steps |

### To change token budget k:
Edit `configs/base.yaml`:
```yaml
model:
  k: 144   # options: 72, 144, 216, 288, 576
```

### Phase configs:
| File | Phase | Steps | Trains |
|---|---|---|---|
| `configs/phase0.yaml` | 0 | 10,000 | TPN only (LLaVA frozen) |
| `configs/phase1.yaml` | 1 | 50,000 | TPN + LoRA + MLP (full) |
| `configs/phase2.yaml` | 2 | — | TextVQA domain transfer |

---

## PART 6 — Monitoring and Checkpoints

### Watch training progress live
```bash
# In a second SSH terminal
tail -f /tmp/rclone_mount.log        # rclone Drive mount health
ls -lh runs/phase0/                  # checkpoint files
```

### Checkpoint format
Each checkpoint saved to `runs/phase{N}/checkpoint_step_{S}.pt` contains:
```python
{
    "global_step": int,
    "epoch": int,
    "model_state_dict": {...},      # plain state dict (not DDP-wrapped)
    "optimizer_state_dict": {...},
    "config": {...},
}
```

### Checkpoints are also on Drive
Because `runs/` is on the local server, not Drive. If you want checkpoints on Drive too (for persistence across machines), output there:
```bash
bash scripts/launch_multigpu.sh \
    --config configs/phase0.yaml \
    --output /mnt/gdrive/tc-vitok-data/runs/ \
    --run-both-phases
```

Note: writing checkpoints to Drive is slower. Recommended workflow: keep `runs/` local during training, copy the final checkpoint to Drive when done.

---

## PART 7 — After a Reboot: Re-mount Drive

The rclone mount does not survive a server reboot. After every reboot:
```bash
bash scripts/setup_gdrive.sh --mount
export DATA_ROOT=/mnt/gdrive/tc-vitok-data
export HF_HOME=${DATA_ROOT}/hf_cache
```

To make this permanent (auto-mount on login), add to `~/.bashrc`:
```bash
# Re-mount if not already mounted
if ! mountpoint -q /mnt/gdrive 2>/dev/null; then
    rclone mount gdrive: /mnt/gdrive \
        --daemon \
        --vfs-cache-mode full \
        --vfs-cache-max-size 50G \
        --buffer-size 256M \
        --dir-cache-time 72h \
        --log-file /tmp/rclone_mount.log &
fi
export DATA_ROOT=/mnt/gdrive/tc-vitok-data
export HF_HOME=${DATA_ROOT}/hf_cache
```

---

## PART 8 — Troubleshooting

### "DATA_ROOT not set / paths not found"
```bash
echo $DATA_ROOT               # must be set
ls $DATA_ROOT/vqav2/          # must contain JSON files and train2014/
ls $DATA_ROOT/hf_cache/       # must contain model folders
```

### "Drive mount not responding / hangs"
```bash
cat /tmp/rclone_mount.log     # check rclone errors
# If stale mount:
fusermount -u /mnt/gdrive
bash scripts/setup_gdrive.sh --mount
```

### "NCCL error / DDP fails"
```bash
# Test both GPUs are visible:
python3 -c "import torch; print(torch.cuda.device_count())"  # must print 2

# Check NCCL:
export NCCL_DEBUG=INFO
bash scripts/launch_multigpu.sh --config configs/phase0.yaml \
    --output /tmp/t --stub --dummy-data --no-wandb --max-steps 2
```

### "CUDA out of memory"
Reduce batch size in `configs/base.yaml`:
```yaml
training:
  batch_size: 16        # down from 32
  grad_accum_steps: 4   # keep effective batch = 16×4×2=128
```

### "HuggingFace model not found"
The model may not have downloaded completely. Re-run:
```bash
export DATA_ROOT=/mnt/gdrive/tc-vitok-data
bash scripts/download_assets.sh --models-only
```

### "rclone: command not found"
```bash
curl https://rclone.org/install.sh | sudo bash
```

### "fusermount: command not found" (for unmount)
```bash
sudo apt-get install fuse   # Ubuntu/Debian
```

---

## Quick Reference Card

```bash
# ── One-time setup ──────────────────────────────────────────────────────
bash scripts/setup_gdrive.sh               # configure rclone (first time only)
bash scripts/setup_gdrive.sh --mount       # mount Drive at /mnt/gdrive
mkdir -p /mnt/gdrive/tc-vitok-data
export DATA_ROOT=/mnt/gdrive/tc-vitok-data
bash scripts/download_assets.sh            # download all data + models to Drive

# ── After every reboot ──────────────────────────────────────────────────
bash scripts/setup_gdrive.sh --mount
export DATA_ROOT=/mnt/gdrive/tc-vitok-data
export HF_HOME=${DATA_ROOT}/hf_cache

# ── Dry run (verify pipeline, no data needed) ───────────────────────────
bash scripts/launch_multigpu.sh \
    --config configs/phase0.yaml \
    --output /tmp/test \
    --stub --dummy-data --no-wandb --max-steps 5

# ── Real training (both phases) ─────────────────────────────────────────
export DATA_ROOT=/mnt/gdrive/tc-vitok-data
bash scripts/launch_multigpu.sh \
    --config configs/phase0.yaml \
    --output runs/ \
    --run-both-phases \
    --no-wandb

# ── Resume from checkpoint ──────────────────────────────────────────────
bash scripts/launch_multigpu.sh \
    --config configs/phase0.yaml \
    --output runs/ \
    --resume runs/phase0/checkpoint_step_5000.pt \
    --no-wandb
```
