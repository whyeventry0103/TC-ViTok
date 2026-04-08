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

### 3. All Bugs Fixed in Training Pipeline (Session 2: Apr 8 2026)

These bugs were discovered and fixed during real-data test runs on the server.
All fixes are committed to `main` and pushed to GitHub.

| Bug | Error | File:Line | Fix |
|---|---|---|---|
| BUG-7 | `AttributeError: 'LlavaForConditionalGeneration' has no attribute 'language_model'` | `tc_vitok_model.py:_init_llava` | Added fallback to `model` attr — old transformers uses `language_model`, new uses `model` |
| BUG-8 | `AttributeError: 'LlavaModel' has no attribute 'prepare_inputs_for_generation'` | `tc_vitok_model.py:_init_llava` | Detected transformers arch version; OLD → `TaskType.CAUSAL_LM` on `LlamaForCausalLM`; NEW → `TaskType.FEATURE_EXTRACTION` on `LlamaModel` |
| BUG-9 | `AssertionError: Input height (98) doesn't match model (518)` | `encoders.py:CoarseVisualEncoder` | Added `img_size=98` to `timm.create_model()` — DINOv2-S default is 518×518, we need 98×98 |
| BUG-10 | `multi_modal_projector` inaccessible after arch change | `tc_vitok_model.py:_run_llm` | Store `self._llm` and `self._mm_proj` as direct refs at init; use them everywhere instead of `self.llava.language_model` / `self.llava.multi_modal_projector` |

**Root cause of BUG-7/8/10:** The server has a newer version of `transformers` (4.45+) where `LlavaForConditionalGeneration` restructured its internals:
- OLD arch (4.36–4.44): `llava.language_model` = `LlamaForCausalLM`, `llava.multi_modal_projector` at top level
- NEW arch (4.45+): `llava.model.language_model` = `LlamaModel` (no head), `llava.model.multi_modal_projector` nested inside

**Key design after fix:**
- `self._llm` = LoRA-wrapped language model (set at `_init_llava`, correct for both arch versions)
- `self._mm_proj` = mm_projector (set at `_init_llava`, correct for both arch versions)
- `set_training_phase()` uses `self._mm_proj.parameters()` for phase 1 unfreezing
- LoRA task type auto-selected: `CAUSAL_LM` if `LlamaForCausalLM`, `FEATURE_EXTRACTION` if `LlamaModel`

### 4. All Bugs Fixed in Training Loop (Session 2 continued: Apr 8 2026)

These bugs caused the first real training run to crash with `Signal 6 (SIGABRT)`.

**Crash details:**
```
rank: 1 (local_rank: 1)
exitcode: -6 (pid: 1737968)
traceback: Signal 6 (SIGABRT) received by PID 1737968
```
Root cause: SIGABRT in PyTorch DDP training means a DataLoader worker silently crashed (corrupt/missing JPEG → worker dies → main process gets SIGABRT). Additionally found a design bug where TPN received zero gradient in Phase 0.

| Bug | Error / Problem | Files | Fix |
|---|---|---|---|
| BUG-11 | `SIGABRT` — DataLoader worker crash on corrupt/missing JPEG image | `datasets.py:VQAv2Dataset.__getitem__`, `GQADataset.__getitem__` | Added retry loop: attempts up to 5 adjacent samples before raising; prevents silent worker death |
| BUG-12 | TPN receives zero gradient in Phase 0 | `configs/phase0.yaml` | `lambda_max: 0.0 → 0.01`, `lambda_entropy: 0.0 → 0.01`. In Option B (gather-based), VQA loss has NO gradient path to TPN because gather uses integer `idx` (non-differentiable). TPN can ONLY learn from L1 sparsity. `lambda=0` meant TPN got zero gradient and Phase 0 did nothing. |
| BUG-13 | Unnecessary activation memory from frozen 7B LLM in Phase 0 | `tc_vitok_model.py:_run_llm` | Wrap LLM forward in `torch.no_grad()` when all LLM params are frozen. Detected via `any(p.requires_grad for p in lm.parameters())`. |
| BUG-14 | SIGABRT instead of Python traceback on NCCL errors | `scripts/launch_multigpu.sh` | Added `NCCL_ASYNC_ERROR_HANDLING=1` (surfaces errors as Python exceptions) and `NCCL_TIMEOUT=1800` (30-min timeout) |
| BUG-15 | No log file — crash leaves no full traceback to diagnose | `scripts/launch_multigpu.sh` | Added `tee logs/train_TIMESTAMP.log` so full output is always saved |

**Config changes (safety):**
- `batch_size: 32 → 16` per GPU (same effective batch 128 via grad_accum 2→4)
- `num_workers: 8 → 4` (fewer worker processes = fewer crash points)

**Key insight on TPN gradient flow (Option B):**
In the gather-based approach, `gumbel_sigmoid_ste` returns `(mask_ste, idx)`. The TPN forward in `tc_vitok_model.py` uses only `idx` (integer tensor) for gathering tokens. `mask_ste` is computed but NOT applied multiplicatively to tokens. Therefore:
- VQA loss → no gradient to TPN (integer gather is non-differentiable)
- Entropy loss → no gradient to TPN (entropy uses `idx` which is integer)
- **L1 sparsity** → ONLY gradient path to TPN (directly on raw scores `s_N`)

Phase 0 must use `lambda_max > 0` for TPN to train at all.

### 5. All Bugs Fixed in download_assets.sh

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

> All downloads complete. GQA questions were the last item (questions1.2.zip, ~1.4GB via aria2c).

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

## Part 3 — Training Status

### Current Status (as of 2026-04-08 ~14:00)
- **Dry run**: PASSED (2 steps, 2 GPUs, BF16, stub mode) ✓
- **Real-data test**: PASSED (2 steps, 2 GPUs, BF16, real VQAv2 + all encoders) ✓
- **First full training attempt**: CRASHED at ~13:33 with SIGABRT (BUG-11 through BUG-15 fixed)
- **Status**: All fixes applied and pushed. Ready to restart.

### Before restarting — pull fixes first
```bash
git -c credential.helper= pull origin main
```

### Start training (after pulling fixes)
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

### Expected completion (after restart)
| Phase | Steps | Est. duration | Est. done |
|---|---|---|---|
| Phase 0 (TPN only, 887k params) | 10,000 | ~9-10 hours | ~Apr 9 morning |
| Phase 1 (TPN + LoRA + MLP) | 50,000 | ~55-70 hours | ~Apr 10-11 |

Throughput: ~3.1 s/step at BF16, batch_size=16/GPU (was 32; reduced for safety).

### Verified correct parameter counts
- **Before `set_training_phase()`**: 324,012,493 trainable (LoRA + mm_proj + VQA head + TPN)
- **After `set_training_phase(0)`**: 887,955 trainable (TPN only — Phase 0 correct)
- These two numbers appearing sequentially in logs is **normal and correct**

### GPU utilization during training
Both GPUs should be at ~100% with ~16-18GB VRAM each:
```
GPU 0: 18167MiB / 49140MiB, 100% utilization
GPU 1: 16367MiB / 49140MiB, 100% utilization
```

---

## Part 4 — Training Commands (Run After Downloads Complete)

### Step 1 — Verify env vars (do this on every reconnect)
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
nvidia-smi   # both GPUs should be ~100% if training is running
```

> **ALREADY RUNNING as of Apr 8 2026 13:25** — training started, no need to re-run unless it crashes.

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

## Part 5 — After Every Reconnect

```bash
# 1. Check env vars
echo $DATA_ROOT   # should print /home/sk10741/tc-vitok-data
# If empty:
source ~/.bashrc

# 2. Check if training is still running
tmux ls           # should show: training: 1 windows
tmux capture-pane -t training -p | tail -20   # see latest output

# 3. Check GPU utilization
nvidia-smi        # both GPUs should be ~100% if training is active

# 4. Check disk space
df -h ~           # make sure checkpoints aren't filling the disk
```

### If training finished normally
Checkpoints saved to `runs/phase0/` and `runs/phase1/`.
Best checkpoints: `runs/phase0/best.ckpt` and `runs/phase1/best.ckpt`.

### If training crashed (GPUs idle, tmux session shows error)
Check the error, fix it, then resume from latest checkpoint:
```bash
# Find latest checkpoint
ls -t runs/phase0/checkpoint_step_*.pt | head -1

# Resume
tmux new -s training
export DATA_ROOT=~/tc-vitok-data
bash scripts/launch_multigpu.sh \
    --config configs/phase0.yaml \
    --output runs/ \
    --resume runs/phase0/checkpoint_step_XXXX.pt \
    --no-wandb
```

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
