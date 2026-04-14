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

### 5. DDP Parameter Mismatch Fix (Session 3: Apr 8 2026)

**Error:**
```
RuntimeError: DDP expects same model across all ranks, but Rank 1 has 474 params,
while rank 0 has inconsistent 0 params.
```
Followed by NCCL timeout after 600 seconds. Also deprecation warning: `NCCL_ASYNC_ERROR_HANDLING` → `TORCH_NCCL_ASYNC_ERROR_HANDLING`.

**Root cause:** Rank 0 crashed/hung during `model.to(device)` — most likely because zombie python processes from the previous SIGABRT crash still held GPU 0 memory. Both ranks independently load the 14GB LLaVA model; if old processes hold GPU 0 VRAM, rank 0 OOMs. Rank 1 then waits for rank 0's DDP init → NCCL timeout.

Additionally, `LlavaForConditionalGeneration.from_pretrained()` without `device_map` may auto-place tensors on GPU during loading (before the explicit `.to(device)` call), consuming extra VRAM.

| Bug | Error / Problem | Files | Fix |
|---|---|---|---|
| BUG-16a | Zombie GPU processes from prior crash block VRAM | `scripts/launch_multigpu.sh` | Added automatic GPU process cleanup before launch: queries `nvidia-smi` for PIDs, kills python/torchrun processes |
| BUG-16b | LLaVA auto-places on GPU during `from_pretrained` | `tc_vitok_model.py:_init_llava` | ~~Initially added `device_map="cpu"`~~ REVERTED: `device_map` installs accelerate dispatch hooks that intercept `.to()`, preventing GPU transfer. Use only `low_cpu_mem_usage=True` (no device_map). |
| BUG-16c | No synchronization between model creation and DDP wrapping | `scripts/train.py` | Added `dist.barrier()` before `model.to(device)` and before DDP wrapping |
| BUG-16d | OOM on `.to(device)` gives no useful error | `scripts/train.py` | Wrapped `model.to(device)` in try/except with rank-specific error message |
| BUG-16e | Param count only logged on rank 0 | `scripts/train.py` | Now logs on ALL ranks for DDP debugging |
| BUG-16f | Frozen vision_tower never used in forward → DDP mismatch | `scripts/train.py` | Added `find_unused_parameters=True` to DDP constructor |
| BUG-16g | Deprecated env var `NCCL_ASYNC_ERROR_HANDLING` | `scripts/launch_multigpu.sh` | Changed to `TORCH_NCCL_ASYNC_ERROR_HANDLING` |

**Before restarting training after a crash:**
```bash
# 1. Kill any zombie GPU processes (now automatic in launch script, but manual option)
nvidia-smi                         # check for leftover processes
pkill -9 -f "scripts/train.py"    # kill any orphaned training processes
sleep 2

# 2. Verify GPUs are free
nvidia-smi                         # should show 0 MiB used / no processes

# 3. Pull latest fixes
git -c credential.helper= pull origin main

# 4. Restart
tmux new -s training
export DATA_ROOT=~/tc-vitok-data
bash scripts/launch_multigpu.sh \
    --config configs/phase0.yaml \
    --output runs/ \
    --run-both-phases \
    --no-wandb
```

### 6. NCCL Timeout During DDP Init (Session 3 continued: Apr 8 2026)

**Error:**
```
[rank1]: Watchdog caught collective operation timeout: WorkNCCL(SeqNum=1, OpType=ALLREDUCE,
NumelIn=1, NumelOut=1, Timeout(ms)=600000) ran for 600073 milliseconds before timing out.
```

**Root cause:** Both ranks load all 4 encoders (DINOv2-S, CLIP-B/32, CLIP-L/14, LLaVA-7B) simultaneously. Even though models are cached locally, the HuggingFace client makes HEAD requests to check for model updates. These HTTP requests timeout intermittently (`'The read operation timed out'`, `504 Gateway Time-out`), causing one rank's model creation to take 5+ minutes longer than the other. After model creation, the `dist.barrier()` call triggers an NCCL ALLREDUCE — but NCCL's default 600s timeout had already been consumed by the desynchronized model loading.

Additionally, the `dist.barrier()` calls added in BUG-16c were themselves part of the problem — they introduced additional synchronization points that could timeout.

| Bug | Error / Problem | Files | Fix |
|---|---|---|---|
| BUG-17a | HuggingFace HTTP requests timeout, desync ranks | `scripts/train.py` | Set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` — all models are pre-downloaded, no need for network |
| BUG-17b | `dist.barrier()` before model.to() caused NCCL timeout | `scripts/train.py` | Removed barriers entirely — DDP constructor handles its own synchronization |
| BUG-17c | NCCL default timeout (600s) too short for large model init | `scripts/train.py` | Set `init_process_group(timeout=timedelta(minutes=30))` |

### 7. device_map="cpu" Prevents GPU Transfer (Session 3 continued: Apr 8 2026)

**Error:**
```
RuntimeError: DDP expects same model across all ranks, but Rank 1 has 474 params,
while rank 0 has inconsistent 0 params.
```
Same DDP error as before, but with a Python traceback this time (thanks to NCCL async error handling).

**Root cause:** `device_map="cpu"` in `LlavaForConditionalGeneration.from_pretrained()` installs HuggingFace `accelerate` dispatch hooks that **intercept `.to()` calls**. When `model.to("cuda:1")` is called, the dispatch hooks prevent the actual tensor transfer — the model thinks it moved but tensors stay on CPU. This is why `nvidia-smi` consistently showed:
- GPU 0: 16,367 MiB (full model moved correctly — possibly accelerate handles rank 0 differently)
- GPU 1: 727 MiB (only small non-LLaVA components moved, LLaVA still on CPU)

DDP's `_verify_param_shape_across_processes` then fails because rank 1's params are on CPU (not visible to NCCL), so it appears to have 0 params from rank 0's perspective.

| Bug | Fix |
|---|---|
| BUG-18 | Removed `device_map="cpu"` from `from_pretrained()`. Use only `low_cpu_mem_usage=True` (loads weights lazily on CPU without dispatch hooks). Added GPU memory logging and param device verification before DDP wrapping. |

### 8. DDP Forward Bypass — Rank 1 Dies Silently (Session 4: Apr 8 2026)

**Symptom:** Both ranks initialize and wrap with DDP correctly, but GPU 1 drops to 0% utilization and rank 1's process dies. Training continues on GPU 0 only at ~1.85 it/s instead of ~3.5 it/s.

**Root cause:** Three interrelated bugs:

| Bug | Error / Problem | Files | Fix |
|---|---|---|---|
| BUG-19a | `train_step()` calls `raw_model.forward_cached()` / `raw_model()` — bypasses DDP wrapper, so gradient all-reduce hooks never fire. Ranks diverge, rank 1 eventually crashes. | `trainer.py:219-246`, `tc_vitok_model.py:279` | Unified `forward()` on `TCViTokModel` that dispatches to cached or full pipeline internally. `train_step()` now always calls `self.model(**kwargs)` (through DDP wrapper). |
| BUG-19b | `validate()` runs on rank 0 only while rank 1 continues to next training batch — ranks desynchronize permanently. | `trainer.py:166-169` | Added `dist.barrier()` after validation and checkpointing so non-main ranks wait. |
| BUG-19c | Checkpointing on rank 0 only without synchronization — same desync problem as validation. | `trainer.py:175-177` | Added `dist.barrier()` after checkpoint save. |

**Key design change:**
- `TCViTokModel.forward()` now accepts both raw-image kwargs (`images_coarse`, `questions`) and cached kwargs (`v_coarse`, `q_embed`). It auto-detects which path to use based on whether `v_coarse`/`q_embed` are provided.
- `forward_cached()` still exists for backward compatibility (e.g., evaluation scripts) but is NOT called during DDP training.
- `raw_model = self.model.module` is still used for attribute access (e.g., `raw_model.k`, `raw_model.set_training_phase()`) — only the forward call must go through the DDP wrapper.

### 9. DDP Slower Than Single GPU for Phase 0 (Session 4 continued: Apr 8 2026)

**Symptom:** After fixing BUG-19 (DDP forward bypass), both GPUs are active, but training speed dropped from 1.85 it/s (1 GPU) to 1.38 it/s (2 GPU DDP) — 34% slower.

**Root cause:** Phase 0 trains only TPN (887K params) but DDP wraps the entire 7.6B model. The gradient all-reduce overhead + `find_unused_parameters=True` graph traversal + disabled P2P (`NCCL_P2P_DISABLE=1`) costs more than the data parallelism gains.

| Bug | Fix |
|---|---|
| BUG-20 | Phase 0: skip DDP — rank 1 exits cleanly, rank 0 runs single-GPU. Phase 1 (35M trainable params) still uses DDP where the overhead is justified. |

### 10. Local import shadows module-level dist (Session 4: Apr 8 2026)

**Error:** `UnboundLocalError: local variable 'dist' referenced before assignment` on rank 1 at `dist.barrier()`.

**Root cause:** `train()` method had `import torch.distributed as dist` inside an `if self.is_main:` block. Python's compiler treats `dist` as local to the entire function. Rank 1 never enters the block, so `dist` is unbound when it hits the `dist.barrier()` calls added in BUG-19b.

| Bug | Fix |
|---|---|
| BUG-19d | Removed the local `import torch.distributed as dist` inside `if self.is_main:`. The module-level import at line 16 suffices. |

### 11. Phase 0 validation hangs at step 500 (Session 4: Apr 8-9 2026)

**Symptom:** Training logs stop at exactly step 500. The process doesn't crash — GPU stays at 95% — but no new output for ~1 hour. Confirmed across multiple runs: log files show last entry at step 500, progress bar stuck at 499/10000.

**Root cause:** `eval_interval: 500` in `configs/phase0.yaml` triggers `validate()` at step 500. The validation loop iterates the entire val set through the 7.6B model forward pass. With VQA loss disabled in Phase 0, this is pure waste that takes ~1 hour per call.

| Bug | Fix |
|---|---|
| BUG-21a | `eval_interval: 500 → 999999` in `configs/phase0.yaml` — skip validation entirely in Phase 0 (VQA=0, nothing to validate) |
| BUG-21b | Added `PYTHONUNBUFFERED=1` to `scripts/launch_multigpu.sh` — fixes stdout buffering when piped through `tee`, so logs appear in real-time |

### 12. Phase 1 Pre-Launch Audit Fixes (Session 4: Apr 9 2026)

Pre-launch audit of Phase 1 code path found 4 blocking issues:

| Bug | Error / Problem | Files | Fix |
|---|---|---|---|
| BUG-22a | `load_state_dict()` strict mode — Phase 0 checkpoint missing LoRA/projector keys → RuntimeError | `trainer.py:451` | Added `strict=False` + log missing/unexpected keys |
| BUG-22b | `best.ckpt` never saved — Phase 0 validation disabled, so `is_best=True` never triggered | `trainer.py:193` | Final checkpoint now saves as `best.ckpt` too |
| BUG-22c | Validation hangs every 1000 steps — full val set (13K batches, ~2h) | `phase1.yaml`, `trainer.py:336` | `eval_interval: 5000`, `max_val_batches: 100` (~55s per val) |
| BUG-22d | BF16/FP16 autocast mismatch — `_llava_forward()` used FP16, trainer used BF16 | `tc_vitok_model.py:450` | Detect caller's autocast context, skip nested autocast if already active |

**Known limitation:** GQA data not loaded in Phase 1 — `datasets: ["vqav2", "gqa"]` config key is ignored because `build_loaders()` reads singular `dataset` key. Phase 1 trains on VQAv2 only. ConcatDataset + unified vocab needed for joint training (future work).

### 13. All Bugs Fixed in download_assets.sh

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

### Current Status (as of 2026-04-14)

| Run | Steps | Best val_vqa | val_loss | Status |
|---|---|---|---|---|
| phase0_v4 | 10,000 | — | — | ✅ COMPLETE |
| phase1_v3 | 50,000 | **0.3334** | 0.2309 | ✅ COMPLETE — BASELINE |
| phase1p5_v1 | ~21k | 0.4610 | 0.3608 | ❌ KILLED — distill dominated |
| phase1p5_v2 | ~10k | — | — | ❌ KILLED — pooler_warmup shock |
| phase1p5_v3 | ~15k | — | — | ❌ KILLED — spread hacking |
| phase1p5_v4 | ~5k | 0.4476 | — | ❌ KILLED — slot init collapse |
| phase1p5_v5 | ~23.6k | 0.4230 | — | ❌ KILLED — aggregate KL collapse |
| phase1p5_v6 | ~12k | 0.4353 | — | ❌ KILLED — negative total loss + moving target |
| phase1p5_v7 | ~13k | 0.4432 | — | ❌ KILLED — regression persisted |
| phase1p5_v8 | 63k | **0.3663** (step 45k) | 0.3276 | ❌ KILLED — oscillation 30k→60k |
| **phase1p5_v8b** | ~5k/55k | 0.4540 | 0.4022 | 🔄 **RUNNING** — frozen pooler |

### Phase 1.5 v8b — Live Training
- Running: `runs/phase1p5_v8b/phase1/`, tmux session `phase1p5_v8b`
- Config: `configs/phase1p5_v8b.yaml`
- Started from: `runs/phase1p5_v8/phase1/best.ckpt` (v8 step 45k, val_vqa=0.3663)
- Key change: **lr_pooler=0.0** (pooler completely frozen), lambda_residual_distill=0, lambda_slot_orth=0
- tau_max=0.595 (continuation from v8 step 45k), max_steps=55000
- Step 5k val_vqa=0.4540 — catapult from fresh optimizer (expected, recovering rapidly)
- **Decision point: step 15k val** — if val_vqa < 0.40 then frozen pooler is helping

### Phase 1 v3 — Final Metrics (COMPLETE — BASELINE)
- val loss: 0.2309, val VQA: **0.3334**, val entropy: -5.1252
- Checkpoint: `runs/phase1_v3/phase1/best.ckpt`

### Key checkpoints on server
```bash
runs/phase0_v4/phase0/best.ckpt          # Phase 0 — start of all Phase 1 runs
runs/phase1_v3/phase1/best.ckpt          # Phase 1 v3 — baseline (0.3334)
runs/phase1p5_v8/phase1/best.ckpt        # Phase 1.5 v8 step 45k — best ever (0.3663)
runs/phase1p5_v8b/phase1/best.ckpt       # Phase 1.5 v8b — current (updating live)
```

### Restart v8b (if tmux session lost)
```bash
tmux new -s phase1p5_v8b
bash scripts/launch_multigpu.sh --config configs/phase1p5_v8b.yaml --output runs/phase1p5_v8b
```

### Verified correct parameter counts
- **Before `set_training_phase()`**: 324,012,493 trainable (LoRA + mm_proj + VQA head + TPN)
- **After `set_training_phase(0)`**: 887,955 trainable (TPN only — Phase 0 correct)
- These two numbers appearing sequentially in logs is **normal and correct**

### GPU utilization during training

**Phase 0 (single GPU):**
```
GPU 0: ~19500MiB / 49140MiB, 95% utilization, ~1.85 it/s
GPU 1: idle (15MiB, DDP skipped for Phase 0)
```

**Phase 1 (2 GPU DDP):**
```
GPU 0: ~31000MiB / 49140MiB, ~100% utilization
GPU 1: ~31000MiB / 49140MiB, ~100% utilization
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

> **RUNNING** — Phase 0 single-GPU, validation disabled. All bugs fixed (BUG-19 through BUG-21).

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
| `batch_size` | 16 | Per GPU |
| `grad_accum_steps` | 4 | Effective batch = 128 total |
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

---

## Part 6 — Phase 1 Completion + Evaluation Bugs (Session 5: Apr 9 2026)

### Phase 1 Training Completed

Phase 1 ran to completion: **50,000 steps**, final val loss **3.916**.
Best checkpoint: `runs/phase1/best.ckpt`.

### Evaluation Script Created

`scripts/run_eval.sh` was created to orchestrate all evaluations:
- TC-ViTok main model (k=144)
- Baseline A2: random token drop
- Baseline A3: image-only TPN (zero query embedding)

Usage:
```bash
tmux new -s eval
bash scripts/run_eval.sh runs/phase1/best.ckpt
```

Logs: `logs/eval_<TIMESTAMP>.log` (master), `logs/eval_tcvitok_<TIMESTAMP>.log`, etc.
Results JSON: `runs/eval_results/*.json`

---

## Part 7 — Critical Fix: Classification → Generative Training Rewrite (Session 5)

### Root Cause Analysis

Phase 1 trained with a **3129-class VQA classification head** (`vqa_head: Linear(4096 → 3129)`), but the paper's evaluation protocol expects **generative free-form answers** from the LLM (LLaVA-style string generation).

These two objectives are fundamentally incompatible:

| Metric | Classification head ceiling | Paper target |
|---|---|---|
| VQAv2 VQA-Score | ~25% (yes/no bias) | ~76% |
| GQA accuracy | ~20% | ~61% |
| POPE macro-F1 | ~50% | ~84% |

The classification head collapsed to predicting "yes"/"no" (top training vocab answers) for every question. This gives ~25% VQA-Score which equals a constant "yes" predictor — the model learned nothing meaningful.

**Why generative evaluation is required:**
- VQAv2 VQA-Score uses soft-accuracy: `min(count(pred in GT) / 3, 1)` across 10 annotator answers
- This can only reach ~76% with free-form generation — a closed vocab of 3129 classes can never soft-match the diversity of 10 annotators
- POPE expects "yes" or "no" generated strings — classification indices don't map to natural language
- The paper explicitly describes LLaVA-style generative evaluation

### What Changed (Session 5 Generative Rewrite)

Full rewrite across 6 files to switch from classification to next-token prediction:

#### `src/tc_vitok/data/datasets.py`
- Removed `answer_to_idx`, `answer_list`, `vocab_size` from all dataset classes
- All datasets now return `answer_str: str` (raw answer string) instead of `labels: tensor`
- `collate_fn` passes `answer_str` as `List[str]`, no stacking

#### `src/tc_vitok/data/caching.py`
- `CachedFeatureDataset`: returns `answer_str` instead of `labels`
- `cached_collate_fn`: passes `answer_str` as list

#### `src/tc_vitok/models/tc_vitok_model.py`
- Removed `vqa_head` entirely from `_init_llava()`
- Added `_use_llava_lm_head` flag: `False` for OLD arch (logits in output), `True` for NEW arch (need `llava.lm_head`)
- Completely rewrote `_llava_forward()` for generative next-token prediction:
  - Training: builds `[visual_prefix | USER: q\nASSISTANT: | answer</s>]` per sample
  - Label masking: `-100` for visual+question positions, answer token IDs for answer positions
  - Right-pads to batch max length
  - Returns `logits [B, seq_len, vocab_size]` + `seq_labels [B, seq_len]`
- Removed `predict_class()` method
- `forward()` now accepts `answer_strings: List[str]` instead of `labels: tensor`
- `set_training_phase()`: removed vqa_head unfreeze (no longer exists)

#### `src/tc_vitok/training/trainer.py`
- `train_step`: reads `answer_strings = batch.get("answer_str")`, passes to model
- Loss now uses `outputs.get("seq_labels")` from model output (not batch `labels`)
- `validate()`: same changes
- `save_checkpoint()`: removed `vqa_head.*` from trainable state filter

#### `src/tc_vitok/training/optimizer.py`
- Removed `vqa_head` parameter group

#### `scripts/evaluate.py`
- Removed `_build_vqav2_train_vocab()` and all `predict_class()` calls
- All three evaluators now use `model.generate()` (generative decoding)
- VQAv2: `generate()` → score against 10 GT answers
- GQA: `generate()` → exact match against single GT answer
- POPE: `generate(max_new_tokens=5)` → `extract_yes_no()` → F1

#### `scripts/train.py`
- Removed `answer_list=` from `VQAv2Dataset` and `GQADataset` construction

### Retraining Required

**Must retrain from scratch** — the Phase 1 checkpoint's LoRA weights are shaped by 50k steps of classification loss, NOT next-token prediction. Loading them for generative fine-tuning would give meaningless initialization.

**Correct retraining procedure:**

```bash
# 1. Pull all changes
git -c credential.helper= pull origin main

# 2. Find Phase 0 checkpoint (TPN pre-trained, LoRA at random init)
ls -t runs/phase0/checkpoint_step_*.pt | head -3
# OR use: runs/phase0/best.ckpt

# 3. Start Phase 1 training from Phase 0 checkpoint
tmux new -s training2
export DATA_ROOT=~/tc-vitok-data
bash scripts/launch_multigpu.sh \
    --config configs/phase1.yaml \
    --output runs/phase1_gen/ \
    --resume runs/phase0/best.ckpt \
    --no-wandb
```

**Why not reuse Phase 1 checkpoint:**
- 50k steps of `vqa_head` CE loss shaped LoRA to map visual features → class index space
- The LM head expects hidden states in language model space, not class index space
- These two spaces are fundamentally different — the LoRA cannot generalize

**Why Phase 0 checkpoint is correct starting point:**
- Phase 0 trained only TPN (L1 sparsity + entropy, no LM loss)
- LoRA adapters were initialized with `get_peft_model()` but received ZERO gradient during Phase 0
- TPN has learned good spatial selection behavior (entropy ~5.9, close to max 6.36)
- Phase 1 with next-token prediction can now shape LoRA from a clean initialization

### Evaluation Bugs Fixed (Session 5, before the rewrite was done)

| Bug | Error / Problem | Fix |
|---|---|---|
| BUG-23 | `run_eval.sh` `$?` always 0 in error handler | `local rc=0; python ... \|\| rc=$?` instead of `if python ...; then` (which consumes `$?`) |
| BUG-24 | Summary table reads `r["vqav2"]["accuracy"]` — wrong key | Fixed to `r["benchmarks"]["vqav2"]["vqa_score"]` |
| BUG-25 | `AttributeError: 'LlamaModel' has no generate()` in `_llava_generate` | PEFT `PeftModel.get_base_model()` returns `LlamaModel` (no lm_head) — use `gen_fn = lm if hasattr(base, "lm_head") else self.llava` |
| BUG-26 | VQA-Score 0.00% — evaluation called `generate()` but model trained with `vqa_head` | Fixed at root: switched to generative training (see above); evaluation now always uses `generate()` |
| BUG-27 | VQA-Score 25% plateau — `vqa_head` collapses to yes/no bias | Fixed at root: removed `vqa_head`, replaced with next-token prediction |

### Expected Results After Generative Retraining

| Benchmark | Current (classification) | Expected (generative, ~LLaVA-1.5 baseline) | Paper target |
|---|---|---|---|
| VQAv2 VQA-Score | ~25% | ~75-77% | ~76% |
| GQA overall | ~20% | ~60-62% | ~61% |
| GQA spatial | ~18% | ~58-60% | ~59% |
| POPE macro-F1 | ~50% | ~83-85% | ~84% |

The generative LM with LoRA should perform near LLaVA-1.5-7B baseline. TC-ViTok's contribution is achieving these numbers with k=144 tokens (25% of N=576) rather than all 576 tokens — demonstrating that learned token selection maintains LLM performance while reducing compute.

### After Phase 1 Completes (Generative Version)

Run evaluation:
```bash
tmux new -s eval
bash scripts/run_eval.sh runs/phase1_gen/best.ckpt
```

Expected evaluation time: ~3-4 hours total (VQAv2 + GQA + POPE × 3 variants).

---

## Part 8 — Static Crop & Gradient Flow Fixes (Session 6: Apr 10 2026)

### Problem: TPN Collapses to Static Crop Despite BUG-29–32 Fixes

After applying BUG-29/30/31/32, val entropy was -4.97 (= log(144) = static crop). Training appeared healthy (train entropy -5.9), but val proved TPN selected the same 144 tokens for every image at inference. Five root causes were found and fixed.

### BUG-33: Dead VQA Loss in Phase 0

**Symptom:** Phase 0 training shows VQA = 0.0000 every step. TPN receives zero gradient; entropy and diversity losses fire but TPN never learns query conditioning.

**Root cause:** `trainer.py` had:
```python
skip_vqa = (self.phase == 0)
if skip_vqa:
    logits = None
```
This set logits=None for all Phase 0 steps, silently zeroing the VQA loss. The original intent was to avoid VQA cost during TPN warmup, but it prevented TPN from learning query conditioning entirely.

**Fix:** Removed the `skip_vqa` condition. Phase 0 now computes full VQA loss. TPN receives query-conditioning gradient from step 0.

| Bug | File | Fix |
|---|---|---|
| BUG-33 | `trainer.py` | Remove `skip_vqa = (self.phase == 0)` block entirely |

### BUG-34: LLM `torch.no_grad()` Blocks STE Gradient

**Symptom:** After enabling VQA in Phase 0, VQA loss fires but TPN gradient is still zero. Tau-independent STE correction is computed but VQA loss gradient never reaches TPN.

**Root cause:** `tc_vitok_model.py` had:
```python
_lm_ctx = torch.no_grad() if _lm_frozen else nullcontext()
```
During Phase 0, LLM params are frozen → `_lm_frozen=True` → entire LLM forward runs inside `no_grad()`. This severs the gradient tape from logits back through the LLM, through `v_selected`, and back to TPN scores — even though TPN params are NOT frozen and the tau-independent STE correction is non-zero.

**Fix:**
```python
_lm_ctx = torch.no_grad() if (_lm_frozen and not self.training) else nullcontext()
```
During training, gradient always flows through LLM regardless of frozen status. The `no_grad()` optimization only applies at eval/inference.

| Bug | File | Fix |
|---|---|---|
| BUG-34 | `tc_vitok_model.py:_run_llm` | `no_grad` only when `_lm_frozen AND not self.training` |

### BUG-35: TPN Score Explosion → tanh Saturation → Instance Norm

**Symptom:** With `lambda_max=0.0` (no L1 penalty) and VQA gradient flowing through TPN, Adam pushes TPN scores to O(93) in ~1700 steps. Tried adding `tanh` activation to bound scores — it saturated at ±1 in ~1000 steps with gradient ≈ 0.001 (vanishing gradient in tanh tail).

**Root cause chain:**
1. Instance norm missing → unbounded scores → Adam overshoots
2. tanh fix → saturated at ±1 → gradient dead
3. Both approaches failed to give TPN meaningful gradient

**Fix: Instance normalization on TPN output:**
```python
s_N = s_upsampled.view(B, -1)           # (B, N)
s_N_mean = s_N.mean(dim=-1, keepdim=True)
s_N_std = s_N.std(dim=-1, keepdim=True).clamp(min=1e-8)
s_N = (s_N - s_N_mean) / s_N_std        # zero-mean, unit-std per sample
```
- No saturation possible (scores always have std=1)
- E[|s_N|] ≈ 0.798 (constant for N(0,1)) → `lambda_sparsity` has no effect → set to 0.0
- Gumbel noise diversifies topk selection at training; deterministic at inference

| Bug | File | Fix |
|---|---|---|
| BUG-35 | `tpn.py:forward` | Replace tanh with instance norm (mean=0, std=1 per sample) after bilinear upsample |

### BUG-36: Diversity Loss Dead — mask_ste Gumbel Noise Kills Cosine Similarity

**Symptom:** `diversity: 0.0000` every step despite `lambda_diversity=0.1`. Loss never fires.

**Root cause:** The original diversity loss used `mask_ste` (gumbel-noisy binary masks):
```python
mask_sim = F.cosine_similarity(mask_ste[:half], mask_ste[half:], dim=-1)
loss_div = (mask_sim - gamma).clamp(min=0) * (1 - q_sim)
```
Gumbel-sigmoid binary masks for different samples are random (different noise realizations) → `cosine_sim ≈ 0` by chance → `clamp(0 - 0.5, min=0) = 0` always. The `gamma=0.5` threshold was never crossed.

**Fix: Score-based diversity (no threshold):**
```python
score_sim = F.cosine_similarity(scores[:half].float(), scores[half:2*half].float(), dim=-1)
q_sim = F.cosine_similarity(q_embed[:half].float(), q_embed[half:2*half].float(), dim=-1)
loss_div = (score_sim * (1.0 - q_sim.detach())).mean()
```
- Raw `s_N` (instance-normalized) reveals true spatial bias — static-crop TPN produces similar score peaks for all samples → `score_sim ≈ 1.0` → loss fires
- No gamma/clamp → always differentiable, gradient active every step
- Negative diversity = anticorrelated score peaks per query = **good** (TPN routing per query)

| Bug | File | Fix |
|---|---|---|
| BUG-36 | `losses.py:compute_total_loss` | Switch from mask_ste cosine+clamp to raw score cosine, no threshold |

### BUG-37: Phase 0 OOM When LLM Activations Stored for STE

**Symptom:** Phase 0 OOMs at `batch_size=16`. Removing `no_grad()` from frozen LLM (BUG-34 fix) requires storing all intermediate LLM activations for the backward pass — consuming ~30GB/GPU extra.

**Fix:** Reduce batch size, increase grad accumulation to maintain effective batch size:

| Config | Old | New | Effective batch |
|---|---|---|---|
| `batch_size` | 16 | 4 | unchanged |
| `grad_accum_steps` | 4 | 16 | 4 × 16 × 2 GPUs = 128 ✓ |

Applied to both `configs/phase0.yaml` and `configs/phase1.yaml`.

| Bug | File | Fix |
|---|---|---|
| BUG-37 | `configs/phase0.yaml`, `configs/phase1.yaml` | batch_size 16→4, grad_accum 4→16 |

### Additional Phase 1 Config Changes (this session)

| Parameter | Old | New | Reason |
|---|---|---|---|
| `lambda_max` | 0.01 | 0.0 | Instance norm makes E[|s_N|]≈0.798 constant; L1 has zero gradient |
| `lambda_entropy` | 0.001 | 0.02 | 0.001 contributed only -0.006 to total loss vs VQA ≈ 0.3; needs to compete |
| `lambda_diversity` | 0.0 | 0.1 | Score-based diversity now active; ≈33% of typical VQA loss |
| `lora_warmfreeze_steps` | — | 3000 | Zero LoRA grads for first 3k steps; forces VQA to optimize TPN before LoRA compensates |
| `lr_tpn` | 1e-3 | 3e-3 | TPN starts near-zero after instance norm; higher LR accelerates recovery |
| `eval_interval` (Phase 0) | 999999 | 2500 | VQA now active; val/entropy meaningful; monitor for static crop early |
| `max_val_batches` (Phase 0) | — | 50 | Limit Phase 0 val to ~60s |

### Phase 1 v3 Training Results (current run)

| Step | val/loss | val/vqa | val/entropy | Notes |
|---|---|---|---|---|
| 5000 | 0.3374 | 0.4411 | -5.1851 | Entropy healthy (static crop = -4.97) |
| 10000 | 0.3030 | 0.4053 | -5.1183 | Slight regression |
| 15000 | 0.3003 | 0.4032 | -5.1412 | Recovered — oscillating, not collapsing |
| 20000 | 0.2591 | 0.3617 | -5.1334 | Stable; val loss improving 23% |

Val entropy stable at -5.11 to -5.19 (not heading toward -4.97). Val VQA improving each checkpoint. Training at 2.21 it/s, ~3h45m remaining from step 20k.

### After Phase 1 v3 Completes

```bash
tmux new -s eval
bash scripts/run_eval.sh runs/phase1_v3/phase1/best.ckpt
```

Monitor progress:
```bash
cat $(ls -t ~/TC-ViTok/logs/train_*.log | head -1) | tr '\r' '\n' | grep "^\[Step.*val"
```

---

## Part 9 — Phase 1.5: Full Architecture + Training Analysis (Session 7: Apr 11-12 2026)

### Phase 1 v3 Final Results

- **Completed**: 50,000 steps at ~2.20 it/s (~6h18m total)
- **Final metrics**: val loss 0.2309, val VQA 0.3334, val entropy -5.1252
- **Checkpoint**: `runs/phase1_v3/phase1/best.ckpt` (539MB)
- **Eval**: run_eval.sh completed [OK] for TC-ViTok — read actual VQA-score % with `cat runs/eval_results/*.json`
- **TPN routing diagnostic**: entropy 5.850 vs random 6.329; center bias 13.1%; query conditioning gap only 0.019 (very weak — model routes foreground but not query-conditioned)

### Phase 1.5 Architecture Implemented

All five Phase 1.5 components implemented and running. See `innovations.md` for full spec.

**New files/changes:**
- `tc_vitok_model.py`: `complement_gather`, `sinusoidal_2d_pe`, `UnselectedPooler`, orthogonal deflation in `forward()`, `_get_teacher_importance()` with 3 bug fixes
- `losses.py`: `spatial_spread_loss`, `attention_distillation_loss` (teacher_scale=10.0), extended `compute_total_loss`
- `curriculum.py`: `get_lambda_spread`
- `optimizer.py`: UnselectedPooler param group
- `trainer.py`: teacher pass, lambda_distill, lambda_spread, mask_soft
- `scripts/train.py`: ConcatDataset for `datasets: ["vqav2", "gqa"]`
- `configs/phase1p5.yaml`: LoRA rank 64, lambda_distill=0.2, lambda_spread=0.005
- `tests/test_phase1p5.py`: 52 tests covering all new components

**Bugs fixed during Phase 1.5 implementation:**

| Bug | Issue | Fix |
|---|---|---|
| BUG-38 | Padding corruption in teacher importance (h[i, ans_start:] averages padding zeros) | Store seq_lens before padding; slice h[i, ans_start:seq_lens[i]] |
| BUG-39 | Wrong LLaVA prompt structure in teacher pass (visual before BOS) | Split: prefix("USER: ") + visual + suffix("\nQ\nASSISTANT:"); track visual_starts |
| BUG-40 | VRAM spikes — lm_out (~19GB) held alive | Explicit del chain at earliest possible points |
| BUG-41 | Teacher distribution flat — T=2.0, cosine_sim std≈0.1 → p_top only 1.16× uniform | teacher_scale=10.0, T=1.0 → ~19× concentration |
| BUG-42 | configs/phase1p5.yaml wrong checkpoint path (runs/phase0_v4/best.ckpt) | Fixed to runs/phase0_v4/phase0/best.ckpt |
| BUG-43 | diagnose_routing.py used nonexistent TCViTokModel.from_pretrained() | Fixed to constructor + load_state_dict; separate preprocessors for 98px/336px |

### Phase 1.5 Launch Command

```bash
# Smoke test first (3 steps, verify compilation)
PYTHONPATH=src python scripts/train.py \
    --config configs/phase1p5.yaml \
    --output runs/smoke_p15 \
    --dummy-data --max-steps 3 --no-wandb

# Full training (after smoke passes)
tmux new -s phase1p5
bash scripts/launch_multigpu.sh configs/phase1p5.yaml runs/phase1p5_v1
```

### Phase 1.5 Training Metrics (live)

| Step | val_loss | val_vqa | val_entropy | Gap vs p1v3 |
|------|----------|---------|-------------|-------------|
| 5k   | 0.4123   | 0.5129  | -5.0266     | +0.072 |
| 10k  | 0.3950   | 0.4951  | -5.0071     | +0.090 |
| 15k  | 0.4000   | 0.5001  | -5.0057     | +0.097 |
| 20k  | 0.3608   | 0.4610  | -5.0112     | +0.099 |

**Status:** Gap vs Phase 1 v3 is widening (0.072→0.099). Phase 1.5 converges 35% slower due to:
1. UnselectedPooler randomly initialized — noisy prefix tokens for first ~15k steps
2. lambda_distill=0.2 contributes 43% of VQA gradient magnitude, redirecting TPN toward teacher's 576-token preferences

**Positive signs:** Distill declining (1.2-1.4 → 0.7-1.1), spread increasing (-107→-113), no collapse.

### Step 25k Decision: Ride Out vs Restart

Check at step 25k:
```bash
cat $(ls -t ~/TC-ViTok/logs/train_*.log | head -1) | tr '\r' '\n' | grep "^\[Step.*val"
```

| Metric | Green (add decay, extend 100k) | Yellow (watch 30k) | Red (restart) |
|--------|-------------------------------|--------------------|---------------|
| val_vqa | < 0.42 | 0.42-0.45 | > 0.45 |
| Gap vs p1v3 | < 0.085 (narrowing) | 0.085-0.095 | > 0.095 |
| distill | < 0.70 | 0.70-0.85 | > 0.85 |

**If Green/Yellow:** Add distill annealing in trainer.py (no restart):
```python
lambda_distill = max(0.02, 0.2 * (1.0 - (step - 20000) / 30000))
```
Extend max_steps to 100k.

**If Red:** Restart from `runs/phase0_v4/phase0/best.ckpt` with lambda_distill=0.05, pooler_warmup_steps=5000.

### Available Checkpoints (as of 2026-04-12)

```
runs/phase0_v4/phase0/best.ckpt     186MB  ← use for Phase 1.5 restarts
runs/phase1_v3/phase1/best.ckpt     539MB  ← best Phase 1 baseline
runs/phase1p5_v1/phase1/...         ongoing ← Phase 1.5 live
runs/eval_results/*.json             ← Phase 1 eval results (read these!)
```
