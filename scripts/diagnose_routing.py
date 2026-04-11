"""
diagnose_routing.py — Quick spatial diagnostic for TPN token selection.

Runs 50 VQAv2 samples through:
  1. TPN routing (query-conditioned)
  2. Random-drop routing (uniform random)
  3. Image-only TPN routing (no query)

Prints:
  - Spatial heatmap of selected tokens (24x24 grid, ASCII)
  - Selection entropy and center-bias score
  - Side-by-side comparison of all three

Usage:
  PYTHONPATH=src python scripts/diagnose_routing.py \
      --checkpoint runs/phase1_v3/phase1/best.ckpt \
      --data-root ~/tc-vitok-data \
      --n-samples 50
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# ── Args ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT", "data"))
parser.add_argument("--n-samples", type=int, default=50)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)

GRID = 24   # 24×24 = 576 tokens
K    = 144  # token budget

# ── Load model ────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  TC-ViTok Routing Diagnostic")
print("="*60)
print(f"  Checkpoint : {args.checkpoint}")
print(f"  Samples    : {args.n_samples}")
print(f"  Grid       : {GRID}×{GRID} = {GRID**2} tokens, k={K}")
print("="*60 + "\n")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from tc_vitok.models.tc_vitok_model import TCViTokModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[diag] Loading model on {device}...")

tpn_config = {
    "n_heads": 4,
    "d_model": 384,
    "d_coarse": 384,
    "d_h": 512,
    "d_text": 512,
    "M": 49,
    "N": 576,
}
lora_config = {
    "rank": 32,
    "alpha": 64,
    "targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "dropout": 0.05,
}

model = TCViTokModel(
    k=K,
    tpn_config=tpn_config,
    use_stub_llm=False,
    llava_model_name="llava-hf/llava-1.5-7b-hf",
    lora_config=lora_config,
    load_encoders=True,
)

ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
state_dict = ckpt.get("model_state_dict", ckpt)
model.load_state_dict(state_dict, strict=False)

model.eval()
model.set_training_phase(1)
model = model.to(device)
print("[diag] Model loaded.\n")

# ── Load a few VQAv2 samples ──────────────────────────────────────────────────
ann_path = Path(args.data_root) / "vqav2" / "v2_mscoco_val2014_annotations.json"
ques_path = Path(args.data_root) / "vqav2" / "v2_OpenEnded_mscoco_val2014_questions.json"
img_dir   = Path(args.data_root) / "vqav2" / "val2014"

print("[diag] Loading VQAv2 val annotations...")
with open(ann_path) as f:
    anns = json.load(f)["annotations"]
with open(ques_path) as f:
    questions_raw = json.load(f)["questions"]

# Build question_id → question/image lookup
qid2q = {q["question_id"]: q for q in questions_raw}

# Sample n_samples
sampled = random.sample(anns, args.n_samples)
print(f"[diag] Sampled {len(sampled)} questions.\n")

# ── Accumulate selection heatmaps ─────────────────────────────────────────────
heatmap_tpn    = np.zeros(GRID * GRID, dtype=np.float32)
heatmap_random = np.zeros(GRID * GRID, dtype=np.float32)
heatmap_imgonly = np.zeros(GRID * GRID, dtype=np.float32)

from torchvision import transforms
preprocess_coarse = transforms.Compose([
    transforms.Resize((98, 98)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
preprocess_full = transforms.Compose([
    transforms.Resize((336, 336)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                         std=[0.26862954, 0.26130258, 0.27577711]),
])

print("[diag] Running samples...")
n_done = 0

with torch.no_grad():
    for ann in sampled:
        qid = ann["question_id"]
        if qid not in qid2q:
            continue
        q_entry = qid2q[qid]
        question = q_entry["question"]
        img_id   = q_entry["image_id"]
        img_file = img_dir / f"COCO_val2014_{img_id:012d}.jpg"

        if not img_file.exists():
            continue

        try:
            img = Image.open(img_file).convert("RGB")
            img_coarse = preprocess_coarse(img).unsqueeze(0).to(device)  # [1, 3, 98, 98]
            img_full   = preprocess_full(img).unsqueeze(0).to(device)    # [1, 3, 336, 336]
        except Exception:
            continue

        # ── 1. TPN routing (query-conditioned) ───────────────────────────────
        outputs = model(
            images_coarse=img_coarse,
            images_full=img_full,
            questions=[question],
            answer_strings=None,
            return_idx=True,
        )
        idx_tpn = outputs["idx"][0].cpu().numpy()          # [k]
        heatmap_tpn[idx_tpn] += 1

        # ── 2. Image-only TPN (zero out query embedding) ─────────────────────
        # Temporarily zero the query embedding by using a blank question
        outputs_io = model(
            images_coarse=img_coarse,
            images_full=img_full,
            questions=[""],          # empty question → near-zero query embedding
            answer_strings=None,
            return_idx=True,
        )
        idx_imgonly = outputs_io["idx"][0].cpu().numpy()   # [k]
        heatmap_imgonly[idx_imgonly] += 1

        # ── 3. Random drop ────────────────────────────────────────────────────
        idx_random = np.random.choice(GRID * GRID, K, replace=False)
        heatmap_random[idx_random] += 1

        n_done += 1
        if n_done % 10 == 0:
            print(f"  [{n_done}/{args.n_samples}] done")

print(f"\n[diag] Completed {n_done} samples.\n")

# ── Analysis functions ────────────────────────────────────────────────────────

def entropy(heatmap: np.ndarray) -> float:
    p = heatmap / heatmap.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))

def center_bias(heatmap: np.ndarray, grid: int = GRID) -> float:
    """Fraction of selections in the center 8x8 region."""
    h2d = heatmap.reshape(grid, grid)
    c = grid // 2
    r = 4  # 8x8 center
    center = h2d[c-r:c+r, c-r:c+r].sum()
    return float(center / heatmap.sum())

def ascii_heatmap(heatmap: np.ndarray, grid: int = GRID, width: int = 24) -> str:
    """Render heatmap as ASCII art. Darker = more selected."""
    h2d = heatmap.reshape(grid, grid)
    # Downsample to width columns
    scale = grid // width
    h2d_ds = h2d.reshape(width, scale, width, scale).mean(axis=(1, 3))
    mx = h2d_ds.max()
    if mx == 0:
        return "(empty)"
    chars = " ░▒▓█"
    lines = []
    for row in h2d_ds:
        line = ""
        for val in row:
            idx_c = int((val / mx) * (len(chars) - 1))
            line += chars[idx_c] * 2  # double-wide for square cells
        lines.append(line)
    return "\n".join(lines)

def top_regions(heatmap: np.ndarray, grid: int = GRID, n: int = 5) -> str:
    """Report the top-n most selected positions as (row, col)."""
    top_idx = np.argsort(heatmap)[::-1][:n]
    parts = []
    for i in top_idx:
        r, c = divmod(i, grid)
        parts.append(f"({r:2d},{c:2d})={heatmap[i]:.0f}")
    return "  ".join(parts)

# ── Print results ─────────────────────────────────────────────────────────────

for label, hm in [("TPN (query-conditioned)", heatmap_tpn),
                  ("Image-only TPN (no query)", heatmap_imgonly),
                  ("Random drop", heatmap_random)]:
    ent = entropy(hm)
    cb  = center_bias(hm)
    print("="*60)
    print(f"  {label}")
    print(f"  Entropy     : {ent:.4f}  (max={np.log(GRID*GRID):.4f}, static crop={np.log(K):.4f})")
    print(f"  Center bias : {cb*100:.1f}%  (center 8×8 of 24×24 = expected {K/(GRID**2)*100:.1f}%)")
    print(f"  Top-5 spots : {top_regions(hm)}")
    print()
    print(ascii_heatmap(hm))
    print()

# ── Side-by-side comparison ───────────────────────────────────────────────────
print("="*60)
print("  SUMMARY")
print("="*60)
print(f"  {'Metric':<25} {'TPN':>10} {'Img-only':>10} {'Random':>10}")
print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")
for label, fn in [("Entropy", entropy), ("Center bias %", lambda h: center_bias(h)*100)]:
    vals = [fn(heatmap_tpn), fn(heatmap_imgonly), fn(heatmap_random)]
    print(f"  {label:<25} {vals[0]:>10.3f} {vals[1]:>10.3f} {vals[2]:>10.3f}")

print()
print("  Interpretation:")
print(f"  - If TPN entropy ≈ random entropy → TPN not learning spatial routing")
print(f"  - If TPN center bias >> random    → TPN has spatial bias (static crop)")
print(f"  - If TPN entropy > img-only       → query conditioning is working")
print(f"  - Static crop entropy             : {np.log(K):.4f}")
print(f"  - Max possible entropy            : {np.log(GRID*GRID):.4f}")
print("="*60 + "\n")
