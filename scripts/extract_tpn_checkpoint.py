#!/usr/bin/env python3
"""Extract TPN + mm_projector weights from Phase 1 checkpoint.

Drops classification-shaped LoRA and vqa_head weights.
Saves a pseudo-Phase-0 checkpoint so Phase 1 generative training
can resume with good TPN weights but clean LoRA initialization.

Usage:
    python3 scripts/extract_tpn_checkpoint.py
"""
import torch
from pathlib import Path

src = "runs/phase1/best.ckpt"
dst = "runs/phase0_extracted.ckpt"

print(f"Loading {src} ...")
ckpt = torch.load(src, map_location="cpu", weights_only=False)
state = ckpt["model_state_dict"]

phase0_state = {
    k: v for k, v in state.items()
    if k.startswith("tpn.") or "multi_modal_projector." in k
}

new_ckpt = {
    "global_step": 0,
    "epoch": 0,
    "model_state_dict": phase0_state,
    "config": {"phase": 0},
}

Path(dst).parent.mkdir(parents=True, exist_ok=True)
torch.save(new_ckpt, dst)
print(f"Saved {len(phase0_state)} keys to {dst}")
for k in sorted(phase0_state)[:5]:
    print(f"  {k}")
print("  ...")
