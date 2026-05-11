# TC-ViTok v3 — Run Plan

## Architecture summary

v3 is the **diagnostic + adaptive-budget** version, justified by the v1 ablations
(E2, E3) and consistent with concurrent text-agnostic methods (FasterVLM,
VisPruner, VisionZip).

- **TPN**: `TokenScorerV3` — per-token MLP + 1 self-attention block, operates
  directly on CLIP ViT-L/14 features at native 24×24 resolution. ~790K params.
- **No DINOv2** coarse encoder. **No CLIP text** encoder.
- **No token merge** (E2 showed β=0.3 is mechanistically a no-op).
- **Adaptive per-sample budget** via `BudgetPredictor`, k ∈ [36, 288].
- **LoRA**: rank 64, targets q/k/v/o/gate/up.
- **Data**: VQAv2 + GQA + TextVQA + LLaVA-Instruct-150K.
- **LR**: lr_tpn=1e-3 (was 3e-3), warmup=3000 (was 1000), max_grad_norm=5.0 (was 1.0).

## Prerequisites — data on server

The server already has VQAv2, GQA, TextVQA, MMBench, MME. We need to add
**LLaVA-Instruct-150K + COCO train2017** for the Phase 1 mixture.

```bash
cd ~/tc-vitok-data
mkdir -p llava_instruct coco/train2017

# 1) LLaVA-Instruct-150K JSON (~250 MB)
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='liuhaotian/LLaVA-Instruct-150K',
    filename='llava_instruct_150k.json',
    repo_type='dataset',
    local_dir='/home/sk10741/tc-vitok-data/llava_instruct',
)
print('Downloaded llava_instruct_150k.json')
"

# 2) COCO train2017 images (~18 GB). Single archive, fastest path:
cd ~/tc-vitok-data/coco
wget -c http://images.cocodataset.org/zips/train2017.zip
unzip -q train2017.zip   # creates ./train2017/ with 118K jpgs
rm train2017.zip

# Verify
ls /home/sk10741/tc-vitok-data/llava_instruct/llava_instruct_150k.json
ls /home/sk10741/tc-vitok-data/coco/train2017/ | head -3
ls /home/sk10741/tc-vitok-data/coco/train2017/ | wc -l   # should be ~118287
```

If `huggingface_hub` is missing: `pip install huggingface_hub`. Disk needed:
~18.5 GB total. You said 52 GB free, so this fits.

## Step 1 — Pull v3 code on the server

```bash
cd ~/TC-ViTok
git -c credential.helper= pull origin main
git log -1 --oneline   # should be the v3 commit
```

## Step 2 — Phase 0 v3 (TPN warmup, ~12–18 hours)

Trains `TokenScorerV3` from scratch with LLaVA frozen. Single GPU is faster
than DDP for Phase 0 (only ~790K trainable params).

```bash
cd ~/TC-ViTok
tmux new -s phase0_v3
# inside tmux:
export DATA_ROOT=/home/sk10741/tc-vitok-data
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
    --config configs/phase0_v3.yaml \
    --output runs/phase0_v3 \
    2>&1 | tee logs/phase0_v3_$(date +%Y%m%d_%H%M%S).log
```

Detach: `Ctrl-b d`. Re-attach: `tmux attach -t phase0_v3`.

**Sanity checks (after 5 min):**
```bash
tmux capture-pane -t phase0_v3 -p | tail -30
# Expect to see:
#   "tpn_version: v3" in config dump
#   "Total params: ... Trainable: ~790,000"
#   "Training on concatenated datasets: ['vqav2', 'gqa']"
#   First "Step 50 | loss: ..." within ~2 min
```

**Decision points:**
- Step 5k val_vqa < 0.50: TPN is learning. Continue.
- Step 5k val_vqa > 0.70: TPN may have collapsed (selecting same tokens).
  Check `entropy_loss` is decreasing. If not, kill and lower lr_tpn to 5e-4.

Best checkpoint: `runs/phase0_v3/phase0/best.ckpt`.

## Step 3 — Phase 1 v3 (full training, ~6–7 days)

Loads from Phase 0, trains TPN + LoRA + mm_projector + BudgetPredictor.

```bash
# In the SAME server session (kill phase0_v3 tmux first if still attached):
tmux kill-session -t phase0_v3   # only after step 2 finishes
tmux new -s phase1_v3
# inside tmux:
cd ~/TC-ViTok
export DATA_ROOT=/home/sk10741/tc-vitok-data
bash scripts/launch_multigpu.sh \
    --config configs/phase1_v3.yaml \
    --output runs/phase1_v3
```

(`launch_multigpu.sh` wraps `torchrun --nproc_per_node=2 scripts/train.py`.)

**Sanity checks (after 5 min):**
```bash
tmux capture-pane -t phase1_v3 -p | tail -40
# Expect:
#   "Phase 1 resuming from runs/phase0_v3/phase0/best.ckpt"
#   "adaptive_budget=True"
#   "Training on concatenated datasets: ['vqav2', 'gqa', 'textvqa', 'llava_instruct']"
#   Sample count: ~1.8M total (440K + 1.2M + 34K + 150K)
#   First "Step 50 | loss: ... | k_eff_mean: ~144" within 2 min
```

**What to watch:**
- `k_effective_mean` should drift from ~144 (init) toward whatever the budget
  loss + VQA loss balance produces (likely 100–180 by step 50k).
- `val_loss` should drop below v1 Phase 1 Final's 0.20 by step 30k.

## Step 4 — Full evaluation (after Phase 1 completes)

### 4a. Headline eval at adaptive budget

```bash
cd ~/TC-ViTok
python scripts/evaluate.py \
    --checkpoint runs/phase1_v3/phase1/best.ckpt \
    --tpn-version v3 \
    --adaptive-budget \
    --benchmarks vqav2 gqa pope mmbench mme textvqa \
    --data-root /home/sk10741/tc-vitok-data \
    --output runs/eval_results/v3_adaptive.json \
    --batch-size 4 --spatial-eval
```

### 4b. Pareto curve — fixed k sweep

Run one eval per budget. This is the headline figure for the paper.

```bash
for K in 36 72 108 144 216 288; do
  python scripts/evaluate.py \
      --checkpoint runs/phase1_v3/phase1/best.ckpt \
      --tpn-version v3 \
      --k $K \
      --benchmarks vqav2 gqa pope \
      --data-root /home/sk10741/tc-vitok-data \
      --output runs/eval_results/v3_fixed_k${K}.json \
      --batch-size 4 --spatial-eval
done
```

### 4c. LoRA-576 control on v3 checkpoint

Reuses the v3 LoRA / mm_projector at full token count for the apples-to-apples
compression-cost number.

```bash
python scripts/evaluate.py \
    --checkpoint runs/phase1_v3/phase1/best.ckpt \
    --tpn-version v3 \
    --baseline full_576 \
    --benchmarks vqav2 gqa pope mmbench mme textvqa \
    --data-root /home/sk10741/tc-vitok-data \
    --output runs/eval_results/v3_full576.json \
    --batch-size 4
```

## Files changed in this v3 update

```
src/tc_vitok/models/tpn.py            (+TokenScorerV3 class)
src/tc_vitok/models/tc_vitok_model.py (+tpn_version branch in __init__, forward, _select_tokens)
src/tc_vitok/data/datasets.py         (+LLaVAInstructDataset class)
scripts/train.py                      (+llava_instruct dispatch, tpn_version wiring)
scripts/evaluate.py                   (+--tpn-version flag, v3 model creation)
configs/phase0_v3.yaml                (new — Phase 0 TPN warmup)
configs/phase1_v3.yaml                (new — Phase 1 full training)
```

## Troubleshooting

**`FileNotFoundError: LLaVA-Instruct image not found`**: This is non-fatal —
the loader retries adjacent samples. If you see many of these in the log, it
means COCO train2017 isn't fully present; either complete the download or
remove `llava_instruct` from `datasets:` in `phase1_v3.yaml`.

**OOM in Phase 1**: lower `batch_size` from 4 → 2 and raise `grad_accum_steps`
from 16 → 32 to keep effective batch=128.

**`tpn_version not found in checkpoint`**: not an error — checkpoints don't
serialize the version flag. The CLI/config flag controls model construction.
