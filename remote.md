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

### Current Status (as of 2026-04-09)
- **Dry run**: PASSED (2 steps, 2 GPUs, BF16, stub mode) ✓
- **Real-data test**: PASSED (2 steps, 2 GPUs, BF16, real VQAv2 + all encoders) ✓
- **First full training attempt**: CRASHED at ~13:33 with SIGABRT (BUG-11 through BUG-15 fixed)
- **Second full training attempt**: CRASHED with DDP param mismatch (BUG-16 fixed)
- **Third full training attempt**: CRASHED with NCCL ALLREDUCE timeout (BUG-17 — HuggingFace network timeouts desync ranks)
- **Fourth full training attempt**: CRASHED with NCCL ALLGATHER timeout (BUG-18 — `device_map="cpu"` prevented `.to(cuda)`)
- **Fifth training attempt**: FAILED — DDP rank 1 dies silently (BUG-19 — forward bypass + validation desync)
- **Sixth training attempt**: DDP working but 34% SLOWER than single GPU (BUG-20 — DDP overhead for 887K params)
- **Seventh training attempt**: Crashed — `UnboundLocalError: dist` on rank 1 (BUG-19d — local import shadowed module-level)
- **Eighth training attempt**: Hangs at step 500 — validation loop runs full 7.6B model on val set (BUG-21)
- **Status**: All fixes applied (BUG-19 through BUG-21). Phase 0 running single-GPU with validation disabled.

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
| Phase | Steps | GPUs | Est. duration | Est. done |
|---|---|---|---|---|
| Phase 0 (TPN only, 887k params) | 10,000 | 1 GPU (DDP skipped) | ~90 min | ~Apr 8 night |
| Phase 1 (TPN + LoRA + MLP) | 50,000 | 2 GPU DDP | ~55-70 hours | ~Apr 10-11 |

Phase 0 runs single-GPU (~1.85 it/s) because DDP overhead on 7.6B model with only 887K trainable params makes 2-GPU 34% slower.

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
