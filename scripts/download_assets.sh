#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# download_assets.sh — download all TC-ViTok datasets and HuggingFace models
#
# Downloads everything into $DATA_ROOT (defaults to ./data).
# Run this once after mounting Google Drive (setup_gdrive.sh --mount).
#
# Usage:
#   # Download everything to mounted Google Drive:
#   export DATA_ROOT=/mnt/gdrive/tc-vitok-data
#   bash scripts/download_assets.sh
#
#   # Download only specific components:
#   bash scripts/download_assets.sh --models-only
#   bash scripts/download_assets.sh --vqav2-only
#   bash scripts/download_assets.sh --gqa-only
#
# After this script completes, start training with:
#   export DATA_ROOT=/mnt/gdrive/tc-vitok-data
#   bash scripts/launch_multigpu.sh --config configs/phase0.yaml --output runs/
# ─────────────────────────────────────────────────────────────────────────────
# Do NOT use set -e here — we want the script to keep going even if one
# download has a transient failure, and we check results explicitly.
set -uo pipefail

DATA_ROOT="${DATA_ROOT:-$(pwd)/data}"
HF_HOME="${HF_HOME:-${DATA_ROOT}/hf_cache}"
TORCH_HOME="${TORCH_HOME:-${DATA_ROOT}/torch_cache}"
export HF_HOME
export TRANSFORMERS_CACHE="${HF_HOME}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TORCH_HOME
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

echo "DATA_ROOT   = ${DATA_ROOT}"
echo "HF_HOME     = ${HF_HOME}"
echo "TORCH_HOME  = ${TORCH_HOME}"
mkdir -p "${DATA_ROOT}" "${HF_HOME}" "${TORCH_HOME}"

# ── Helpers ───────────────────────────────────────────────────────────────────
need_cmd() { command -v "$1" &>/dev/null || { echo "ERROR: '$1' not found. Install it first."; exit 1; }; }
hr() { echo ""; echo "──────────────────────────────────────────"; echo " $1"; echo "──────────────────────────────────────────"; }

# Robust wget: resume (-c), 5 retries, 120s timeout, no clobber on complete files
WGET="wget -c --tries=5 --timeout=120 --show-progress -q"

# ── HuggingFace model pre-download ───────────────────────────────────────────
download_models() {
    hr "Pre-downloading HuggingFace models → ${HF_HOME}"
    need_cmd python3

    python3 - <<'PYEOF'
import os
hf_home = os.environ["HF_HOME"]
torch_home = os.environ["TORCH_HOME"]
print(f"HF cache  : {hf_home}")
print(f"timm cache: {torch_home}")

# ── 1. DINOv2-S via timm (~85 MB) ─────────────────────────────────────
# CoarseVisualEncoder uses timm.create_model("vit_small_patch14_dinov2.lvd142m")
# timm downloads to TORCH_HOME/hub/
print("\n[1/4] Downloading DINOv2-S via timm ...")
import timm
m = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True)
del m
print("  ✓ DINOv2-S done")

# ── 2. CLIP ViT-B/32 (~340 MB) ────────────────────────────────────────
# TextEncoder uses openai/clip-vit-base-patch32 (512-dim text embeddings)
print("\n[2/4] Downloading CLIP ViT-B/32 ...")
from transformers import CLIPTextModel, CLIPProcessor
CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir=hf_home)
CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir=hf_home)
print("  ✓ CLIP ViT-B/32 done")

# ── 3. CLIP ViT-L/14-336 (~1.7 GB) ───────────────────────────────────
# FullVisualEncoder uses openai/clip-vit-large-patch14-336
# MUST be the -336 variant (24×24=576 patches at 336px input)
print("\n[3/4] Downloading CLIP ViT-L/14-336 ...")
from transformers import CLIPModel, CLIPProcessor
CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336", cache_dir=hf_home)
CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336", cache_dir=hf_home)
print("  ✓ CLIP ViT-L/14-336 done")

# ── 4. LLaVA-1.5-7B (~14 GB) ─────────────────────────────────────────
# TCViTokModel._init_llava uses LlavaForConditionalGeneration
print("\n[4/4] Downloading LLaVA-1.5-7B (this takes a while, ~14 GB) ...")
from transformers import LlavaForConditionalGeneration, LlavaProcessor
LlavaProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf", cache_dir=hf_home)
LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-1.5-7b-hf",
    torch_dtype="bfloat16",
    low_cpu_mem_usage=True,
    cache_dir=hf_home,
)
print("  ✓ LLaVA-1.5-7B done")

print("\nAll models downloaded successfully.")
PYEOF
}

# ── VQAv2 dataset ─────────────────────────────────────────────────────────────
download_vqav2() {
    hr "Downloading VQAv2 → ${DATA_ROOT}/vqav2"
    need_cmd wget
    need_cmd unzip

    VQAV2_DIR="${DATA_ROOT}/vqav2"
    mkdir -p "${VQAV2_DIR}"

    BASE="https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa"
    IMG_BASE="https://images.cocodataset.org/zips"

    # Annotations — map zip name → expected extracted JSON name
    declare -A ANNOTATION_FILES=(
        ["v2_Annotations_Train_mscoco.zip"]="v2_mscoco_train2014_annotations.json"
        ["v2_Annotations_Val_mscoco.zip"]="v2_mscoco_val2014_annotations.json"
        ["v2_Questions_Train_mscoco.zip"]="v2_OpenEnded_mscoco_train2014_questions.json"
        ["v2_Questions_Val_mscoco.zip"]="v2_OpenEnded_mscoco_val2014_questions.json"
    )
    for ZIP_FILE in "${!ANNOTATION_FILES[@]}"; do
        JSON_FILE="${ANNOTATION_FILES[$ZIP_FILE]}"
        if [ ! -f "${VQAV2_DIR}/${JSON_FILE}" ]; then
            echo "Downloading ${ZIP_FILE} ..."
            $WGET -P "${VQAV2_DIR}" "${BASE}/${ZIP_FILE}"
            unzip -o "${VQAV2_DIR}/${ZIP_FILE}" -d "${VQAV2_DIR}"
            rm -f "${VQAV2_DIR}/${ZIP_FILE}"
        else
            echo "  ✓ ${JSON_FILE} already present"
        fi
    done

    # COCO images (train2014 ~13 GB, val2014 ~6 GB)
    for SPLIT in train2014 val2014; do
        IMG_DIR="${VQAV2_DIR}/${SPLIT}"
        ZIP="${VQAV2_DIR}/${SPLIT}.zip"
        if [ ! -d "${IMG_DIR}" ]; then
            echo "Downloading COCO ${SPLIT} images ..."
            wget -c --tries=5 --timeout=120 --show-progress -q \
                -O "${ZIP}" "${IMG_BASE}/${SPLIT}.zip"
            if [ -f "${ZIP}" ]; then
                echo "Extracting ${SPLIT}.zip ..."
                unzip -o "${ZIP}" -d "${VQAV2_DIR}"
                rm -f "${ZIP}"
            else
                echo "ERROR: ${ZIP} not found after download — re-run script to retry"
            fi
        else
            echo "  ✓ ${SPLIT} already present"
        fi
    done
    echo "  ✓ VQAv2 done"
}

# ── GQA dataset ───────────────────────────────────────────────────────────────
download_gqa() {
    hr "Downloading GQA → ${DATA_ROOT}/gqa"
    need_cmd wget
    need_cmd unzip

    GQA_DIR="${DATA_ROOT}/gqa"
    mkdir -p "${GQA_DIR}"

    BASE="https://downloads.cs.stanford.edu/nlp/data/gqa"

    # Questions (balanced split only — ~660 MB)
    for FILE in \
        "train_balanced_questions.json" \
        "val_balanced_questions.json" \
        "testdev_balanced_questions.json"
    do
        if [ ! -f "${GQA_DIR}/${FILE}" ]; then
            echo "Downloading ${FILE} ..."
            $WGET -P "${GQA_DIR}" "${BASE}/${FILE}"
        else
            echo "  ✓ ${FILE} already present"
        fi
    done

    # Images (~20 GB)
    IMG_DIR="${GQA_DIR}/images"
    ZIP="${GQA_DIR}/images.zip"
    if [ ! -d "${IMG_DIR}" ]; then
        echo "Downloading GQA images (~20 GB) ..."
        $WGET -P "${GQA_DIR}" "${BASE}/images.zip"
        echo "Extracting GQA images.zip ..."
        unzip -o "${ZIP}" -d "${GQA_DIR}"
        rm -f "${ZIP}"
    else
        echo "  ✓ GQA images already present"
    fi
    echo "  ✓ GQA done"
}

# ── Dispatch ──────────────────────────────────────────────────────────────────
case "${1:-all}" in
    --models-only) download_models ;;
    --vqav2-only)  download_vqav2  ;;
    --gqa-only)    download_gqa    ;;
    all|"")
        download_models
        download_vqav2
        download_gqa
        ;;
    *)
        echo "Usage: $0 [--models-only | --vqav2-only | --gqa-only]"
        exit 1
        ;;
esac

hr "All downloads complete"
echo "DATA_ROOT is ready at: ${DATA_ROOT}"
echo ""
echo "Next steps:"
echo "  export DATA_ROOT=${DATA_ROOT}"
echo "  bash scripts/launch_multigpu.sh --config configs/phase0.yaml --output runs/ --run-both-phases"
