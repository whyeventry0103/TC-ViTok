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
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-$(pwd)/data}"
HF_HOME="${HF_HOME:-${DATA_ROOT}/hf_cache}"
export HF_HOME
export TRANSFORMERS_CACHE="${HF_HOME}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"

echo "DATA_ROOT = ${DATA_ROOT}"
echo "HF_HOME   = ${HF_HOME}"
mkdir -p "${DATA_ROOT}" "${HF_HOME}"

# ── Helpers ───────────────────────────────────────────────────────────────────
need_cmd() { command -v "$1" &>/dev/null || { echo "ERROR: '$1' not found. Install it first."; exit 1; }; }
hr() { echo ""; echo "──────────────────────────────────────────"; echo " $1"; echo "──────────────────────────────────────────"; }

# ── HuggingFace model pre-download ───────────────────────────────────────────
download_models() {
    hr "Pre-downloading HuggingFace models → ${HF_HOME}"
    need_cmd python3

    python3 - <<'PYEOF'
import os, sys
hf_home = os.environ["HF_HOME"]
print(f"Saving models to: {hf_home}")

# DINOv2-S (coarse encoder, ~85 MB)
print("\n[1/3] Downloading DINOv2-S ...")
from transformers import AutoModel, AutoFeatureExtractor
AutoFeatureExtractor.from_pretrained("facebook/dinov2-small", cache_dir=hf_home)
AutoModel.from_pretrained("facebook/dinov2-small", cache_dir=hf_home)
print("  ✓ DINOv2-S done")

# CLIP ViT-L/14 (text encoder + full visual, ~890 MB)
print("\n[2/3] Downloading CLIP ViT-L/14 ...")
from transformers import CLIPModel, CLIPProcessor
CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", cache_dir=hf_home)
CLIPModel.from_pretrained("openai/clip-vit-large-patch14", cache_dir=hf_home)
print("  ✓ CLIP ViT-L/14 done")

# LLaVA-1.5-7B (~14 GB — this is the big one)
print("\n[3/3] Downloading LLaVA-1.5-7B (this takes a while, ~14 GB) ...")
from transformers import AutoTokenizer, AutoModelForCausalLM
AutoTokenizer.from_pretrained("llava-hf/llava-1.5-7b-hf", cache_dir=hf_home)
AutoModelForCausalLM.from_pretrained(
    "llava-hf/llava-1.5-7b-hf",
    torch_dtype="bfloat16",
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
    IMG_BASE="http://images.cocodataset.org/zips"

    # Annotations — check for the actual extracted JSON names, not the zip stem
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
            wget -q --show-progress -P "${VQAV2_DIR}" "${BASE}/${ZIP_FILE}"
            unzip -q -o "${VQAV2_DIR}/${ZIP_FILE}" -d "${VQAV2_DIR}"
            rm "${VQAV2_DIR}/${ZIP_FILE}"
        else
            echo "  ✓ ${JSON_FILE} already present"
        fi
    done

    # COCO images (train2014 ~13 GB, val2014 ~6 GB)
    for SPLIT in train2014 val2014; do
        IMG_DIR="${VQAV2_DIR}/${SPLIT}"
        ZIP="${VQAV2_DIR}/${SPLIT}.zip"
        if [ ! -d "${IMG_DIR}" ]; then
            echo "Downloading COCO ${SPLIT} images (~6-13 GB) ..."
            wget -q --show-progress -P "${VQAV2_DIR}" "${IMG_BASE}/${SPLIT}.zip"
            unzip -q -o "${ZIP}" -d "${VQAV2_DIR}"
            rm "${ZIP}"
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
            wget -q --show-progress -P "${GQA_DIR}" "${BASE}/${FILE}"
        fi
    done

    # Images (~20 GB)
    IMG_DIR="${GQA_DIR}/images"
    if [ ! -d "${IMG_DIR}" ]; then
        echo "Downloading GQA images (~20 GB) ..."
        wget -q --show-progress -P "${GQA_DIR}" "${BASE}/images.zip"
        unzip -q -o "${GQA_DIR}/images.zip" -d "${GQA_DIR}"
        rm "${GQA_DIR}/images.zip"
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
