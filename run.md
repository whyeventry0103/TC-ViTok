# TC-ViTok: Complete Run Guide

A step-by-step guide for setting up, training, evaluating, and visualizing
TC-ViTok (Task-Conditioned Visual Tokenizer). Follow these instructions in
order. Each section states its prerequisites.

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Model Downloads](#2-model-downloads)
3. [Dataset Downloads](#3-dataset-downloads)
4. [Verify Installation](#4-verify-installation)
5. [Feature Caching](#5-feature-caching)
6. [Training](#6-training)
7. [Evaluation](#7-evaluation)
8. [Visualization](#8-visualization)
9. [Jupyter Notebooks](#9-jupyter-notebooks)
10. [Expected Outputs](#10-expected-outputs)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Environment Setup

### 1.1 Create Conda Environment

```bash
conda create -n torch-m1 python=3.11 -y
conda activate torch-m1
```

Python 3.11 is recommended. Python 3.12 works but 3.14 does not have PyTorch
packages available.

### 1.2 Install PyTorch

**Apple Silicon (M1/M2/M3):**

```bash
pip install torch==2.5.1 torchvision==0.20.1
```

**CUDA (Linux with NVIDIA GPU, recommended for actual training):**

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

> **Note on PyTorch version:** The project was developed with PyTorch 2.5.1.
> The `transformers` library (v4.36+) raises a `ValueError` about
> CVE-2025-32434 when calling `from_pretrained()` on PyTorch < 2.6, because
> `torch.load` defaults to `weights_only=False`. See Section 11 for the
> workaround. If possible, upgrade to PyTorch >= 2.6:
>
> ```bash
> pip install torch>=2.6.0 torchvision>=0.21.0
> ```

### 1.3 Install Dependencies

```bash
cd /Users/apple/Desktop/Junior_spring/TC-ViTok
pip install -r requirements.txt
```

The full dependency list (`requirements.txt`):

| Package | Purpose |
|---|---|
| torch >= 2.0.0 | Core ML framework |
| torchvision >= 0.15.0 | Image transforms |
| transformers >= 4.36.0 | LLaVA-1.5, CLIP, DINOv2 model loading |
| accelerate >= 0.25.0 | HuggingFace model loading utilities |
| peft >= 0.7.0 | LoRA adapter for LLaVA |
| timm >= 0.9.12 | DINOv2 model registry |
| datasets >= 2.16.0 | HuggingFace datasets |
| h5py >= 3.10.0 | HDF5 feature caching |
| einops >= 0.7.0 | Tensor rearrangement |
| omegaconf >= 2.3.0 | YAML config loading |
| wandb >= 0.16.0 | Experiment logging |
| tqdm >= 4.66.0 | Progress bars |
| numpy >= 1.24.0 | Numerical operations |
| scipy >= 1.11.0 | Statistical utilities |
| pillow >= 10.0.0 | Image loading |
| matplotlib | Visualization (for `visualize_masks.py` and notebooks) |
| pytest >= 7.4.0 | Testing |

Install matplotlib separately if not pulled in:

```bash
pip install matplotlib
```

### 1.4 Environment Variable

Every command in this project must be prefixed with (or have this exported):

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
```

This resolves an OpenMP duplicate library conflict on macOS. The training and
evaluation scripts set this automatically via `os.environ.setdefault`, but
test scripts and notebooks need it set externally.

### 1.5 Project Structure

```
TC-ViTok/
  configs/
    base.yaml          # Shared hyperparameters (k=144, LR, loss weights)
    phase0.yaml        # Phase 0: TPN warmup (frozen LLM)
    phase1.yaml        # Phase 1: Joint TPN + LoRA + MLP
    phase2.yaml        # Phase 2: TextVQA domain transfer (optional)
  src/tc_vitok/
    models/            # TPN, masking, encoders, tc_vitok_model
    training/          # losses, curriculum, optimizer, trainer
    data/              # datasets, caching, spatial_labels
    evaluation/        # metrics, collapse_detection, spatial_eval, pareto
  scripts/
    train.py           # Training CLI
    evaluate.py        # Evaluation CLI
    cache_features.py  # Feature pre-computation
    visualize_masks.py # Token selection visualization
    test_core.py       # Lightweight verification (no downloads)
    test_pipeline.py   # Full pipeline test (needs model downloads)
  tests/               # pytest test suite (52 tests)
  notebooks/           # Analysis notebooks (01-04)
  data/                # Datasets go here (not tracked by git)
  runs/                # Training outputs go here
```

---

## 2. Model Downloads

Three frozen encoder models are used. They download automatically from
HuggingFace on first use, but you can trigger them explicitly.

### 2.1 DINOv2-Small (Coarse Visual Encoder)

- **HuggingFace ID:** `facebook/dinov2-small`
- **Loaded via:** `timm.create_model("vit_small_patch14_dinov2.lvd142m")`
- **Size:** ~88 MB
- **Cache location:** `~/.cache/huggingface/hub/` (or `~/.cache/torch/hub/` for timm)
- **Output:** (B, 49, 384) from 98x98 input

To pre-download:

```python
import timm
model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True)
```

### 2.2 CLIP ViT-B/32 (Text Encoder)

- **HuggingFace ID:** `openai/clip-vit-base-patch32`
- **Size:** ~340 MB
- **Cache location:** `~/.cache/huggingface/hub/models--openai--clip-vit-base-patch32/`
- **Output:** (B, 512) [EOS] embedding (text encoder half only; vision side is not used)

To pre-download:

```bash
python -c "from transformers import CLIPModel; CLIPModel.from_pretrained('openai/clip-vit-base-patch32')"
```

### 2.3 LLaVA-1.5-7B (LLM Backbone)

- **HuggingFace ID:** `llava-hf/llava-1.5-7b-hf`
- **Size:** ~13.5 GB
- **Cache location:** `~/.cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/`
- **Includes:** CLIP ViT-L/14 vision tower (304M frozen params), Vicuna-7B LLM, mm_projector

To pre-download:

```bash
python -c "from transformers import LlavaForConditionalGeneration; LlavaForConditionalGeneration.from_pretrained('llava-hf/llava-1.5-7b-hf')"
```

> **CVE-2025-32434 blocker:** If you are on PyTorch < 2.6, the above command
> will fail with a `ValueError`. See Section 11 for the workaround.

### 2.4 CLIP ViT-L/14 (Full Visual Encoder)

This model is used in two ways:

1. **Bundled inside LLaVA-1.5-7B** as its `vision_tower` — downloaded automatically with LLaVA (Section 2.3).
2. **Standalone `FullVisualEncoder`** — loaded separately by `cache_features.py` and `test_pipeline.py` when computing full visual features.

- **Standalone HuggingFace ID:** `openai/clip-vit-large-patch14-336`
- **Size:** ~890 MB
- **Cache location:** `~/.cache/huggingface/hub/models--openai--clip-vit-large-patch14-336/`
- **Output:** (B, 576, 1024) from 336×336 input (24×24 spatial grid)

To pre-download the standalone model:

```bash
python -c "from transformers import CLIPModel; CLIPModel.from_pretrained('openai/clip-vit-large-patch14-336')"
```

---

## 3. Dataset Downloads

All datasets go under `data/` in the project root. Create the directory first:

```bash
mkdir -p data/{vqav2,gqa,textvqa,pope,coco}
```

### 3.1 VQAv2 (Primary Benchmark)

**Size:** ~25 GB (images) + ~200 MB (annotations)

**Images (COCO 2014):**

```bash
# Training images (~13 GB, 82,783 images)
cd data/coco
wget http://images.cocodataset.org/zips/train2014.zip
unzip train2014.zip

# Validation images (~6 GB, 40,504 images)
wget http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip
```

**Annotations:**

```bash
cd data/vqav2

# Questions
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip
unzip v2_Questions_Train_mscoco.zip
unzip v2_Questions_Val_mscoco.zip

# Annotations
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip
unzip v2_Annotations_Train_mscoco.zip
unzip v2_Annotations_Val_mscoco.zip
```

**Expected directory structure:**

```
data/vqav2/
  v2_mscoco_train2014_annotations.json
  v2_mscoco_val2014_annotations.json
  v2_OpenEnded_mscoco_train2014_questions.json
  v2_OpenEnded_mscoco_val2014_questions.json

data/coco/
  train2014/
    COCO_train2014_000000000009.jpg
    ...
  val2014/
    COCO_val2014_000000000042.jpg
    ...
```

> **Note:** The VQAv2Dataset loader expects COCO image filenames in the format
> `COCO_{split}2014_{image_id:012d}.jpg`. The `image_dir` argument should
> point to the directory containing the images (e.g., `data/coco/train2014`).

> **Important — create symlinks now:** `evaluate.py` and `cache_features.py` look
> for VQAv2 images under `data/vqav2/train2014` and `data/vqav2/val2014`. Create
> symlinks immediately after downloading COCO images so later steps don't fail:
>
> ```bash
> ln -s ../coco/train2014 data/vqav2/train2014
> ln -s ../coco/val2014 data/vqav2/val2014
> ```

### 3.2 GQA (Spatial Evaluation)

**Size:** ~20 GB (images) + ~1.5 GB (questions/scene graphs)

Download from https://cs.stanford.edu/people/dorarad/gqa/download.html:

```bash
cd data/gqa

# Images (~20 GB)
wget https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip
unzip images.zip

# Questions (~1 GB)
wget https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip
unzip questions1.2.zip

# Scene graphs (optional, for spatial classification) (~100 MB)
wget https://downloads.cs.stanford.edu/nlp/data/gqa/sceneGraphs.zip
unzip sceneGraphs.zip
```

**Expected directory structure:**

```
data/gqa/
  images/
    000001.jpg
    ...
  train_balanced_questions.json
  val_balanced_questions.json
  testdev_balanced_questions.json
  train_sceneGraphs.json   (optional)
  val_sceneGraphs.json     (optional)
```

### 3.3 TextVQA (Phase 2 Domain Transfer)

**Size:** ~8 GB (images) + ~50 MB (annotations)

Download from https://textvqa.org/dataset/:

```bash
cd data/textvqa

# Images
wget https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip
unzip train_val_images.zip

# Annotations
wget https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_train.json
wget https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json
```

**Expected directory structure:**

```
data/textvqa/
  train_images/
    <image_id>.jpg
    ...
  TextVQA_0.5.1_train.json
  TextVQA_0.5.1_val.json
```

### 3.4 POPE (Hallucination Benchmark)

**Size:** ~1 MB (annotations only; uses COCO val2014 images)

```bash
cd data/pope

# Clone POPE annotations
git clone https://github.com/AoiDragon/POPE.git pope_repo
cp pope_repo/output/coco/coco_pope_popular.jsonl .
cp pope_repo/output/coco/coco_pope_random.jsonl .
cp pope_repo/output/coco/coco_pope_adversarial.jsonl .
rm -rf pope_repo
```

**Expected directory structure:**

```
data/pope/
  coco_pope_popular.jsonl
  coco_pope_random.jsonl
  coco_pope_adversarial.jsonl
```

POPE reuses COCO val2014 images from `data/coco/val2014/`.

---

## 4. Verify Installation

### 4.1 Core Tests (No Downloads Required)

These 7 tests verify TPN architecture, masking, losses, and curriculum using
only synthetic tensors. They require zero model or dataset downloads.

```bash
cd /Users/apple/Desktop/Junior_spring/TC-ViTok
KMP_DUPLICATE_LIB_OK=TRUE /Users/apple/miniforge3/envs/torch-m1/bin/python scripts/test_core.py
```

**Expected output:** All 7 tests pass, TPN params = 887,955.

### 4.2 Pytest Suite (No Downloads Required)

52 tests covering TPN (12), masking (16), and spatial labels (24):

```bash
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=src /Users/apple/miniforge3/envs/torch-m1/bin/python -m pytest tests/ -v
```

**Expected output:** 52 passed.

### 4.3 Pipeline Test (Requires DINOv2 + CLIP Downloads)

End-to-end forward + backward pass with StubLLM. Triggers DINOv2-S and CLIP
ViT-B/32 downloads (~400 MB total) on first run:

```bash
KMP_DUPLICATE_LIB_OK=TRUE /Users/apple/miniforge3/envs/torch-m1/bin/python scripts/test_pipeline.py
```

### 4.4 Training Dry Run (StubLLM, No Real Data)

Verifies the full training loop with synthetic data and StubLLM:

```bash
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=src /Users/apple/miniforge3/envs/torch-m1/bin/python \
    scripts/train.py \
    --config configs/phase0.yaml \
    --max-steps 1 \
    --stub \
    --dummy-data \
    --no-wandb
```

**Expected output:** Model created, 1 training step completes, metrics printed.

---

## 5. Feature Caching

Pre-computing DINOv2-S coarse features and CLIP text embeddings into HDF5
avoids re-encoding them every epoch. This is optional but significantly speeds
up training.

**Prerequisites:** DINOv2-S and CLIP ViT-B/32 downloaded (Section 2.1, 2.2),
dataset downloaded (Section 3).

### 5.1 Cache VQAv2 Training Features

```bash
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=src /Users/apple/miniforge3/envs/torch-m1/bin/python \
    scripts/cache_features.py \
    --dataset vqav2 \
    --split train \
    --output data/cache/ \
    --annotations-path data/vqav2/v2_mscoco_train2014_annotations.json \
    --questions-path data/vqav2/v2_OpenEnded_mscoco_train2014_questions.json \
    --image-dir data/coco/train2014 \
    --batch-size 32 \
    --device mps
```

### 5.2 Cache VQAv2 Validation Features

```bash
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=src /Users/apple/miniforge3/envs/torch-m1/bin/python \
    scripts/cache_features.py \
    --dataset vqav2 \
    --split val \
    --output data/cache/ \
    --annotations-path data/vqav2/v2_mscoco_val2014_annotations.json \
    --questions-path data/vqav2/v2_OpenEnded_mscoco_val2014_questions.json \
    --image-dir data/coco/val2014 \
    --batch-size 32 \
    --device mps
```

### 5.3 Cache GQA Features

```bash
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=src /Users/apple/miniforge3/envs/torch-m1/bin/python \
    scripts/cache_features.py \
    --dataset gqa \
    --split train \
    --output data/cache/ \
    --questions-path data/gqa/train_balanced_questions.json \
    --image-dir data/gqa/images \
    --batch-size 32 \
    --device mps
```

### 5.4 Output

Each cache command produces an HDF5 file:

```
data/cache/
  vqav2_train.h5    # v_coarse (num_samples, 49, 384) float16 + q_embed (num_samples, 512) float16
  vqav2_val.h5
  gqa_train.h5
```

HDF5 keys: `v_coarse`, `q_embed`, `labels`, `question_ids`.

**Approximate sizes:** ~2 GB per 100K samples (dominated by v_coarse).

### 5.5 Optional Arguments

| Flag | Default | Description |
|---|---|---|
| `--dataset` | required | `vqav2`, `gqa`, or `textvqa` |
| `--split` | `train` | Dataset split |
| `--output` | required | Output directory for .h5 files |
| `--annotations-path` | None | Annotations file (VQAv2 only) |
| `--questions-path` | None | Questions file |
| `--image-dir` | None | Image directory |
| `--device` | auto | `cuda`, `mps`, or `cpu` |
| `--batch-size` | 32 | Batch size for feature extraction |
| `--max-samples` | None | Limit samples (for testing) |

---

## 6. Training

### 6.1 Overview

Training proceeds in two mandatory phases and one optional phase:

| Phase | Config | What Trains | Loss | Duration |
|---|---|---|---|---|
| Phase 0 | `configs/phase0.yaml` | TPN only | L_VQA (lambda=0, mu=0) | 1 epoch, 10K steps |
| Phase 1 | `configs/phase1.yaml` | TPN + LoRA + MLP | All terms (tau anneal, lambda warmup) | 3 epochs, 50K steps |
| Phase 2 | `configs/phase2.yaml` | TPN + LoRA + MLP | L_VQA + entropy/diversity (tau=0.1 fixed, L_comp disabled: lambda_max=0.0) | 1 epoch, 10K steps |

### 6.2 Phase 0: TPN Warmup

**Prerequisites:** LLaVA-1.5-7B downloaded (Section 2.3), VQAv2 downloaded
(Section 3.1).

```bash
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=src /Users/apple/miniforge3/envs/torch-m1/bin/python \
    scripts/train.py \
    --config configs/phase0.yaml \
    --output runs/
```

With feature caching:

```bash
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=src /Users/apple/miniforge3/envs/torch-m1/bin/python \
    scripts/train.py \
    --config configs/phase0.yaml \
    --output runs/ \
    --cache-path data/cache/vqav2_train.h5
```

**Key Phase 0 settings:** tau=1.0 (fixed), lambda=0 (no sparsity), lr_tpn=1e-3.

### 6.3 Phase 1: Joint Fine-tuning

**Prerequisites:** Phase 0 checkpoint exists at `runs/phase0/best.ckpt`.

```bash
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=src /Users/apple/miniforge3/envs/torch-m1/bin/python \
    scripts/train.py \
    --config configs/phase1.yaml \
    --output runs/ \
    --resume runs/phase0/best.ckpt
```

**Key Phase 1 settings:** tau 1.0 -> 0.1 linear, lambda 0 -> 0.1 warmup over
5000 steps, lr_tpn=1e-3, lr_lora=2e-4, lr_mlp=5e-4.

### 6.4 Run Both Phases Sequentially

Use `--run-both-phases` to run Phase 0 then Phase 1 automatically. Phase 1
config is auto-detected from the same `configs/` directory:

```bash
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=src /Users/apple/miniforge3/envs/torch-m1/bin/python \
    scripts/train.py \
    --config configs/phase0.yaml \
    --output runs/ \
    --run-both-phases
```

### 6.5 Phase 2: TextVQA Domain Transfer (Optional)

```bash
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=src /Users/apple/miniforge3/envs/torch-m1/bin/python \
    scripts/train.py \
    --config configs/phase2.yaml \
    --output runs/ \
    --resume runs/phase1/best.ckpt
```

### 6.6 Dry Run (1 Step, No Real Data)

Verify the training loop works without any data or model downloads:

```bash
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=src /Users/apple/miniforge3/envs/torch-m1/bin/python \
    scripts/train.py \
    --config configs/phase0.yaml \
    --max-steps 1 \
    --stub \
    --dummy-data \
    --no-wandb
```

### 6.7 CLI Reference: scripts/train.py

| Flag | Default | Description |
|---|---|---|
| `--config` | required | Path to YAML config file |
| `--output` | `runs` | Output directory for checkpoints |
| `--resume` | None | Checkpoint path to resume from |
| `--wandb-project` | `tc-vitok` | WandB project name |
| `--max-steps` | None | Override config max_steps (dry run: `--max-steps 1`) |
| `--run-both-phases` | False | Auto-run Phase 1 after Phase 0 |
| `--device` | auto | `cuda`, `mps`, or `cpu` |
| `--stub` | False | Use StubLLM instead of LLaVA |
| `--dummy-data` | False | Use synthetic data (256 train, 64 val) |
| `--cache-path` | None | Path to HDF5 feature cache |
| `--no-wandb` | False | Disable WandB logging |

### 6.8 Config YAML Structure

Each phase config is loaded **independently** via `OmegaConf.load()` — there is no
automatic inheritance from `base.yaml` at runtime. The `defaults: - base` key in
each phase YAML is a documentation annotation only. Every key a phase needs must be
explicitly present in that phase's YAML (or it will be missing from the merged
config). To see all available keys and their defaults, consult `configs/base.yaml`.

Key fields in `base.yaml`:

```yaml
model:
  k: 144                    # Token budget
  tpn:
    n_heads: 4
    d_model: 384
    d_coarse: 384
    d_h: 512
    d_text: 512
    M: 49                   # Coarse tokens (7x7)
    N: 576                  # Full tokens (24x24)
  lora:
    rank: 32
    alpha: 64
    targets: [q_proj, k_proj, v_proj, o_proj]

training:
  batch_size: 16
  grad_accum_steps: 4       # Effective batch = 64
  max_steps: 50000
  lr_tpn: 1e-3
  lr_lora: 2e-4
  lr_mlp: 5e-4
  tau_max: 1.0
  tau_min: 0.1
  lambda_max: 0.1
  T_warmup: 5000
  lambda_entropy: 0.01
  lambda_diversity: 0.01
```

---

## 7. Evaluation

### 7.1 Evaluate on VQAv2

**Prerequisites:** Trained checkpoint, VQAv2 val data downloaded.

```bash
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=src /Users/apple/miniforge3/envs/torch-m1/bin/python \
    scripts/evaluate.py \
    --checkpoint runs/phase1/best.ckpt \
    --benchmarks vqav2 \
    --data-dir data \
    --output results/vqav2_results.json
```

### 7.2 Evaluate on Multiple Benchmarks

```bash
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=src /Users/apple/miniforge3/envs/torch-m1/bin/python \
    scripts/evaluate.py \
    --checkpoint runs/phase1/best.ckpt \
    --benchmarks vqav2 gqa pope \
    --data-dir data \
    --output results/all_benchmarks.json
```

### 7.3 GQA with Spatial Stratification

Splits GQA results into spatial-reasoning vs. global-reasoning questions:

```bash
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=src /Users/apple/miniforge3/envs/torch-m1/bin/python \
    scripts/evaluate.py \
    --checkpoint runs/phase1/best.ckpt \
    --benchmarks gqa \
    --spatial-eval \
    --data-dir data \
    --output results/gqa_spatial.json
```

### 7.4 Token Budget Sweep (Pareto Curve)

Evaluates at multiple token budgets and computes Pareto frontier:

```bash
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=src /Users/apple/miniforge3/envs/torch-m1/bin/python \
    scripts/evaluate.py \
    --checkpoint runs/phase1/best.ckpt \
    --benchmarks vqav2 gqa \
    --token-budgets 72 144 216 288 576 \
    --data-dir data \
    --output results/pareto_sweep.json
```

### 7.5 Quick Debug Evaluation

```bash
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=src /Users/apple/miniforge3/envs/torch-m1/bin/python \
    scripts/evaluate.py \
    --checkpoint runs/phase1/best.ckpt \
    --benchmarks vqav2 \
    --max-samples 100 \
    --stub \
    --data-dir data \
    --output results/debug.json
```

### 7.6 CLI Reference: scripts/evaluate.py

| Flag | Default | Description |
|---|---|---|
| `--checkpoint` | required | Path to model checkpoint |
| `--benchmarks` | `[vqav2]` | Space-separated list: `vqav2 gqa textvqa pope` |
| `--token-budgets` | `[144]` | Space-separated list of k values |
| `--output` | `evaluation_results.json` | Output JSON path |
| `--device` | auto | `cuda`, `mps`, or `cpu` |
| `--spatial-eval` | False | Enable GQA spatial stratification |
| `--data-dir` | `data` | Root data directory |
| `--max-samples` | None | Limit samples (debugging) |
| `--batch-size` | 16 | Inference batch size |
| `--stub` | False | Use StubLLM for pipeline testing |

### 7.7 Expected Data Paths

The evaluation script expects this layout under `--data-dir`:

```
<data-dir>/
  vqav2/
    v2_mscoco_val2014_annotations.json
    v2_OpenEnded_mscoco_val2014_questions.json
    val2014/            (symlink to data/coco/val2014 or direct images)
  gqa/
    val_balanced_questions.json
    images/
  pope/
    coco_pope_popular.jsonl
  coco/
    val2014/            (POPE uses these images)
```

If you followed Section 3.1, the symlinks from `data/vqav2/{train,val}2014` →
`data/coco/{train,val}2014` are already in place. If not, create them now:

```bash
ln -s ../coco/train2014 data/vqav2/train2014
ln -s ../coco/val2014 data/vqav2/val2014
```

---

## 8. Visualization

### 8.1 Visualize Token Selection Masks

**Prerequisites:** Trained checkpoint.

```bash
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=src /Users/apple/miniforge3/envs/torch-m1/bin/python \
    scripts/visualize_masks.py \
    --checkpoint runs/phase1/best.ckpt \
    --images path/to/image1.jpg path/to/image2.jpg \
    --questions "What is on the left side?" "What color is the car?" \
    --output visualizations/ \
    --k 144
```

### 8.2 Single Image with Default Question

```bash
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=src /Users/apple/miniforge3/envs/torch-m1/bin/python \
    scripts/visualize_masks.py \
    --checkpoint runs/phase1/best.ckpt \
    --images path/to/image.jpg \
    --output visualizations/
```

If `--questions` is omitted, the default question is "What is in this image?".

### 8.3 With StubLLM (No LLaVA Required)

```bash
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=src /Users/apple/miniforge3/envs/torch-m1/bin/python \
    scripts/visualize_masks.py \
    --checkpoint runs/phase0/best.ckpt \
    --images path/to/image.jpg \
    --stub \
    --output visualizations/
```

### 8.4 Output

Each image produces a 3-panel PNG:

1. **Left:** Original image
2. **Center:** 24x24 token selection heatmap (hot colormap)
3. **Right:** Heatmap overlaid on original image

Files are saved as `visualizations/mask_000.png`, `mask_001.png`, etc.

### 8.5 CLI Reference: scripts/visualize_masks.py

| Flag | Default | Description |
|---|---|---|
| `--checkpoint` | required | Path to model checkpoint |
| `--images` | required | Space-separated image paths |
| `--questions` | None | Space-separated questions (one per image, or one for all) |
| `--output` | `visualizations` | Output directory |
| `--k` | 144 | Token budget |
| `--device` | auto | `cuda`, `mps`, or `cpu` |
| `--stub` | False | Use StubLLM |

---

## 9. Jupyter Notebooks

### 9.1 Setup

```bash
conda activate torch-m1
pip install jupyter
cd /Users/apple/Desktop/Junior_spring/TC-ViTok
KMP_DUPLICATE_LIB_OK=TRUE jupyter notebook
```

Make sure to set `PYTHONPATH` in the notebook kernel. Add this cell at the top
of every notebook:

```python
import sys, os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, "src")
```

### 9.2 Notebook Descriptions

Run them in order. Each notebook has prerequisites from previous steps.

#### Notebook 01: `notebooks/01_baseline_reproduction.ipynb`

**Purpose:** Reproduce the LLaVA-1.5 baseline (VQAv2 79.1%) to establish the
performance ceiling.

**Prerequisites:**
- LLaVA-1.5-7B downloaded (Section 2.3)
- VQAv2 val data downloaded (Section 3.1)

**What it does:**
1. Loads pretrained LLaVA-1.5-7B
2. Evaluates on VQAv2 validation set
3. Reports accuracy (target: ~79.1%)

**Status:** Skeleton only (TODO cells). Needs implementation before use.

#### Notebook 02: `notebooks/02_token_visualization.ipynb`

**Purpose:** Visualize CLIP ViT-L/14 visual tokens using PCA to understand
spatial feature distribution.

**Prerequisites:**
- LLaVA-1.5-7B downloaded (provides CLIP ViT-L/14)
- Sample images

**What it does:**
1. Extracts 576 visual tokens from CLIP ViT-L/14 (each 1024-dim)
2. Applies PCA to reduce to 3 dimensions (RGB)
3. Reshapes to 24x24 spatial grid
4. Visualizes as RGB image

**Status:** Skeleton only (TODO cells).

#### Notebook 03: `notebooks/03_tpn_debug.ipynb`

**Purpose:** Debug and verify Gumbel-Sigmoid STE gradient flow through the TPN.

**Prerequisites:**
- Core tests passing (Section 4.1)
- No model downloads needed (uses synthetic data)

**What it does:**
1. Verifies `mask.grad_fn is not None` (STE has gradient function)
2. Confirms forward pass produces hard binary mask
3. Confirms backward pass uses soft Gumbel-Sigmoid gradients
4. Tests temperature annealing behavior

**Status:** Skeleton only (TODO cells).

#### Notebook 04: `notebooks/04_results_analysis.ipynb`

**Purpose:** Analyze experimental results, plot Pareto curves, generate tables
for the paper.

**Prerequisites:**
- Evaluation results JSON files from Section 7
- Training complete for at least one token budget

**What it does:**
1. Plots accuracy vs. tokens Pareto curves for k in {72, 144, 216, 288, 576}
2. Compares TC-ViTok vs. LLaVA-1.5 baseline across all benchmarks
3. Spatial vs. global GQA stratification analysis
4. Collapse detection and failure mode diagnostics

**Status:** Skeleton only (TODO cells).

---

## 10. Expected Outputs

### After Feature Caching (Section 5)

```
data/cache/
  vqav2_train.h5     (~2-4 GB)
  vqav2_val.h5        (~1-2 GB)
  gqa_train.h5        (~2-4 GB)
```

### After Training (Section 6)

```
runs/
  phase0/
    best.ckpt          # Best Phase 0 checkpoint
    last.ckpt           # Final Phase 0 checkpoint
    logs/               # TensorBoard/WandB logs
  phase1/
    best.ckpt
    last.ckpt
    logs/
  phase2/              # Only if Phase 2 was run
    best.ckpt
    last.ckpt
    logs/
```

### After Evaluation (Section 7)

```
results/
  vqav2_results.json
  all_benchmarks.json
  gqa_spatial.json
  pareto_sweep.json
```

JSON format:

```json
{
  "144": {
    "vqav2": {"mean": 0.781, "median": 1.0, "std": 0.35, "count": 40504},
    "gqa": {"mean": 0.62, "spatial": {"mean": 0.64}, "global": {"mean": 0.60}}
  }
}
```

### After Visualization (Section 8)

```
visualizations/
  mask_000.png    # 3-panel: original | heatmap | overlay
  mask_001.png
  ...
```

---

## 11. Troubleshooting

### 11.1 CVE-2025-32434: PyTorch < 2.6 + transformers Conflict

**Symptom:** `ValueError: You are using an unsafe version of PyTorch` when
calling `from_pretrained()`.

**Root cause:** `transformers` >= 4.40 enforces `weights_only=True` in
`torch.load`, which requires PyTorch >= 2.6.

**Fix (preferred):** Upgrade PyTorch:

```bash
pip install torch>=2.6.0 torchvision>=0.21.0
```

**Workaround (if upgrade is not possible):** Monkey-patch before importing
transformers:

```python
import torch
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.pop("weights_only", None)
    return _original_load(*args, **kwargs)
torch.load = _patched_load
```

Or pin an older transformers version (not recommended):

```bash
pip install transformers==4.38.2
```

### 11.2 KMP_DUPLICATE_LIB_OK Error

**Symptom:** `OMP: Error #15: Initializing libiomp5.dylib, but found
libiomp5.dylib already initialized.`

**Fix:** Set the environment variable before every command:

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
```

Or prefix each command: `KMP_DUPLICATE_LIB_OK=TRUE python ...`

The training and evaluation scripts set this automatically, but test scripts
and notebooks need it set externally.

### 11.3 Wrong Python Interpreter

**Symptom:** `ModuleNotFoundError: No module named 'torch'`

**Fix:** Always use the conda environment Python:

```bash
/Users/apple/miniforge3/envs/torch-m1/bin/python scripts/train.py ...
```

Or activate the environment first:

```bash
conda activate torch-m1
python scripts/train.py ...
```

Do not use the system Python (`/usr/bin/python3`) which may be Python 3.14
without ML packages.

### 11.4 PYTHONPATH Not Set

**Symptom:** `ModuleNotFoundError: No module named 'tc_vitok'`

**Fix:** Add `PYTHONPATH=src` before the command:

```bash
PYTHONPATH=src python scripts/train.py ...
```

The `test_core.py` and `test_pipeline.py` scripts handle this with
`sys.path.insert(0, ".")`, but `train.py`, `evaluate.py`, `cache_features.py`,
and `visualize_masks.py` import from `tc_vitok` directly and need `PYTHONPATH=src`.

### 11.5 MPS Out of Memory

**Symptom:** `RuntimeError: MPS backend out of memory`

**Fix:** Reduce batch size. MPS on Apple Silicon has limited VRAM (8-16 GB
unified memory). Use `--batch-size 4` or `--batch-size 8`. For actual training,
a CUDA GPU with >= 24 GB VRAM is recommended.

### 11.6 src/tc_vitok/data/ Not in Git

The `.gitignore` file has a `data/` pattern that is too broad and matches
`src/tc_vitok/data/` as well. All data module source code exists only locally.
If cloning fresh, you will need to obtain these files separately.

### 11.7 Known Training Failure Modes

| Symptom | Likely Cause | Fix |
|---|---|---|
| TPN always selects center tokens | Spatial collapse | Increase lambda_entropy: 0.01 -> 0.05 |
| Same tokens selected regardless of question | Query-ignoring (IoU > 0.9) | Extend Phase 0; enable L_div |
| Same token mask every batch | Degenerate fixed mask | Reduce lambda_max by 10x |
| TPN gradients near zero | STE gradient vanishing | Slow tau annealing; set tau_min=0.2 |
| VQA loss plateaus but TPN loss does not improve | LLM bypassing visual tokens | Lower lr_lora; extend Phase 0 |
| VQA loss stuck after 100 steps in Phase 0 | LLM not frozen properly | Check optimizer groups, verify `set_training_phase(0)` |

---

## Quick Reference: Full Pipeline

```bash
# 0. Activate environment
conda activate torch-m1
export KMP_DUPLICATE_LIB_OK=TRUE
cd /Users/apple/Desktop/Junior_spring/TC-ViTok

# 1. Verify setup
PYTHONPATH=src python scripts/test_core.py
PYTHONPATH=src python -m pytest tests/ -v

# 2. Cache features (after downloading datasets)
PYTHONPATH=src python scripts/cache_features.py \
    --dataset vqav2 --split train --output data/cache/ \
    --annotations-path data/vqav2/v2_mscoco_train2014_annotations.json \
    --questions-path data/vqav2/v2_OpenEnded_mscoco_train2014_questions.json \
    --image-dir data/coco/train2014

# 3. Train Phase 0 + Phase 1
PYTHONPATH=src python scripts/train.py \
    --config configs/phase0.yaml \
    --output runs/ \
    --run-both-phases

# 4. Evaluate
PYTHONPATH=src python scripts/evaluate.py \
    --checkpoint runs/phase1/best.ckpt \
    --benchmarks vqav2 gqa pope \
    --token-budgets 72 144 216 288 576 \
    --spatial-eval \
    --data-dir data \
    --output results/full_eval.json

# 5. Visualize
PYTHONPATH=src python scripts/visualize_masks.py \
    --checkpoint runs/phase1/best.ckpt \
    --images data/coco/val2014/COCO_val2014_000000000042.jpg \
    --questions "What is on the left?" \
    --output visualizations/
```
