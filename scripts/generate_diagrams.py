#!/usr/bin/env python3
"""Generate architecture diagrams for TC-ViTok README.

Run:
    python scripts/generate_diagrams.py

Outputs PNGs and SVGs to assets/ directory.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

# ── Color palette ──
FROZEN_COLOR = "#D5D5D5"       # gray — frozen
FROZEN_EDGE = "#888888"
TRAINABLE_COLOR = "#C8E6C9"   # green — trainable
TRAINABLE_EDGE = "#388E3C"
NEW_COLOR = "#FFCC80"          # orange — new adaptive component
NEW_EDGE = "#E65100"
OP_COLOR = "#BBDEFB"           # blue — operations
OP_EDGE = "#1565C0"
DATA_COLOR = "#E1BEE7"         # purple — data tensors
DATA_EDGE = "#7B1FA2"
LOSS_COLOR = "#FFCDD2"         # red — loss
LOSS_EDGE = "#C62828"


def _box(ax, x, y, w, h, text, color, edge, fontsize=9, bold=False):
    """Draw a rounded box with centered text."""
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.1",
        facecolor=color, edgecolor=edge, linewidth=1.5,
    )
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            fontweight=weight, wrap=True)


def _arrow(ax, x1, y1, x2, y2, label="", color="black", style="-|>"):
    """Draw an arrow between two points with optional label."""
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle=style, color=color, lw=1.5),
    )
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx + 0.15, my, label, fontsize=7, color="#333333",
                ha="left", va="center")


def generate_system_architecture():
    """Generate the main TC-ViTok system architecture diagram with adaptive budget."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 11))
    ax.set_xlim(-1, 15)
    ax.set_ylim(-1, 11)
    ax.axis("off")
    ax.set_aspect("equal")

    # Title
    ax.text(7, 10.5, "TC-ViTok: System Architecture with Adaptive Budget",
            ha="center", va="center", fontsize=14, fontweight="bold")

    # ── Row 1: Inputs ──
    _box(ax, 2, 9, 2.5, 0.8, "Image\n336x336", DATA_COLOR, DATA_EDGE)
    _box(ax, 6, 9, 2.5, 0.8, "Image\n98x98", DATA_COLOR, DATA_EDGE)
    _box(ax, 10.5, 9, 2.5, 0.8, "Question Q", DATA_COLOR, DATA_EDGE)

    # ── Row 2: Frozen encoders ──
    _box(ax, 2, 7.5, 2.8, 0.9, "CLIP ViT-L/14\n(frozen)", FROZEN_COLOR, FROZEN_EDGE)
    _box(ax, 6, 7.5, 2.8, 0.9, "DINOv2-S\n(frozen)", FROZEN_COLOR, FROZEN_EDGE)
    _box(ax, 10.5, 7.5, 2.8, 0.9, "CLIP Text\n(frozen)", FROZEN_COLOR, FROZEN_EDGE)

    # Arrows: inputs → encoders
    _arrow(ax, 2, 8.6, 2, 7.95)
    _arrow(ax, 6, 8.6, 6, 7.95)
    _arrow(ax, 10.5, 8.6, 10.5, 7.95)

    # ── Row 3: Encoder outputs ──
    ax.text(2, 6.8, "[B, 576, 1024]", ha="center", fontsize=7, color="#666")
    ax.text(6, 6.8, "[B, 49, 384]", ha="center", fontsize=7, color="#666")
    ax.text(10.5, 6.8, "[B, 512]", ha="center", fontsize=7, color="#666")

    # ── Row 3: TPN ──
    _box(ax, 7.5, 5.5, 3.5, 1.2, "TPN (887K)\nCross-Attn + Score MLP\n+ Instance Norm",
         TRAINABLE_COLOR, TRAINABLE_EDGE, bold=True)

    # Arrows: coarse + text → TPN
    _arrow(ax, 6, 7.05, 7, 6.1)
    _arrow(ax, 10.5, 7.05, 8.5, 6.1)

    # TPN outputs
    ax.text(6, 4.6, "s_N [B, 576]", ha="center", fontsize=7, color="#333")
    ax.text(9.5, 4.6, "ctx [B, 384]", ha="center", fontsize=7, color="#333")
    _arrow(ax, 6.8, 4.9, 6, 4.75, color=TRAINABLE_EDGE)
    _arrow(ax, 8.5, 4.9, 9.5, 4.75, color=TRAINABLE_EDGE)

    # ── Budget Predictor (NEW) ──
    _box(ax, 10.5, 3.8, 2.8, 1.0,
         "BudgetPredictor (25K)\nMLP: 384\u219264\u21921\nsigmoid \u2192 k_frac",
         NEW_COLOR, NEW_EDGE, bold=True)
    _arrow(ax, 9.5, 4.5, 10.5, 4.3, color=NEW_EDGE)
    ax.text(11.5, 3.1, "k_frac [B]", ha="center", fontsize=7, color=NEW_EDGE)
    # Label "NEW"
    ax.text(12.2, 4.35, "NEW", fontsize=8, fontweight="bold", color=NEW_EDGE,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFF3E0", edgecolor=NEW_EDGE))

    # ── Adaptive Gumbel-STE (NEW) ──
    _box(ax, 5, 3.2, 3.5, 1.0,
         "Adaptive Gumbel-STE\nPer-sample threshold\nVariable top-k + STE",
         NEW_COLOR, NEW_EDGE, bold=True)
    _arrow(ax, 6, 4.5, 5.5, 3.7, color=TRAINABLE_EDGE)  # scores → STE
    _arrow(ax, 10.5, 3.3, 6.75, 3.2, color=NEW_EDGE)     # k_frac → STE
    ax.text(7.2, 3.85, "NEW", fontsize=8, fontweight="bold", color=NEW_EDGE,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFF3E0", edgecolor=NEW_EDGE))

    # STE outputs
    ax.text(3.5, 2.4, "idx [B, k_max]", ha="center", fontsize=7, color="#333")
    ax.text(7, 2.4, "k_effective [B]", ha="center", fontsize=7, color=NEW_EDGE)
    _arrow(ax, 4.2, 2.7, 3.5, 2.55)
    _arrow(ax, 6, 2.7, 7, 2.55, color=NEW_EDGE)

    # ── Gather + Merge ──
    _box(ax, 2.5, 1.8, 2.5, 0.8, "Gather + Merge\n(variable length)", OP_COLOR, OP_EDGE)
    _arrow(ax, 2, 7.05, 2, 2.2, color=FROZEN_EDGE)  # full features → gather
    ax.text(1.5, 5.0, "v_full", fontsize=7, color="#666", rotation=90)
    _arrow(ax, 3.5, 2.35, 2.8, 2.2)  # idx → gather

    # ── Budget Loss (NEW) ──
    _box(ax, 10.5, 2.0, 2.5, 0.7, "Budget Loss\n\u03bb * mean(k_eff/N)", LOSS_COLOR, LOSS_EDGE)
    _arrow(ax, 7, 2.35, 10.5, 2.35, color=NEW_EDGE)

    # ── Gather output ──
    ax.text(2.5, 1.1, "v_selected [B, k_i, 1024]", ha="center", fontsize=7, color="#333")

    # ── LLaVA LLM ──
    _box(ax, 2.5, 0.3, 3, 0.9, "LLaVA LLM (LoRA r=64)\n+ Attention Mask", TRAINABLE_COLOR, TRAINABLE_EDGE, bold=True)
    _arrow(ax, 2.5, 1.0, 2.5, 0.75)

    # ── Answer ──
    _box(ax, 2.5, -0.7, 2, 0.6, "Answer", DATA_COLOR, DATA_EDGE)
    _arrow(ax, 2.5, -0.15, 2.5, -0.4)

    # ── Legend ──
    legend_y = 0.5
    legend_x = 12
    ax.text(legend_x, legend_y + 1.5, "Legend:", fontsize=9, fontweight="bold")
    for i, (label, color, edge) in enumerate([
        ("Frozen", FROZEN_COLOR, FROZEN_EDGE),
        ("Trainable", TRAINABLE_COLOR, TRAINABLE_EDGE),
        ("NEW (Adaptive)", NEW_COLOR, NEW_EDGE),
        ("Operation", OP_COLOR, OP_EDGE),
        ("Data / Loss", DATA_COLOR, DATA_EDGE),
    ]):
        y = legend_y + 1.0 - i * 0.4
        box = FancyBboxPatch(
            (legend_x - 0.3, y - 0.12), 0.6, 0.24,
            boxstyle="round,pad=0.05", facecolor=color, edgecolor=edge, lw=1.2,
        )
        ax.add_patch(box)
        ax.text(legend_x + 0.6, y, label, fontsize=8, va="center")

    # Save
    for ext in ("png", "svg"):
        path = os.path.join(ASSETS_DIR, f"tc_vitok_architecture.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"Saved {path}")
    plt.close(fig)


def generate_tpn_architecture():
    """Generate TPN internal architecture diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 9))
    ax.set_xlim(-1, 13)
    ax.set_ylim(-1, 9)
    ax.axis("off")
    ax.set_aspect("equal")

    ax.text(6, 8.5, "TPN Internal Architecture + BudgetPredictor",
            ha="center", fontsize=14, fontweight="bold")

    # ── Inputs ──
    _box(ax, 2, 7.5, 2.5, 0.7, "v_coarse [B, 49, 384]", DATA_COLOR, DATA_EDGE)
    _box(ax, 8, 7.5, 2.5, 0.7, "q_embed [B, 512]", DATA_COLOR, DATA_EDGE)

    # ── Cross-Attention ──
    _box(ax, 5, 6.2, 3.5, 0.9, "Cross-Attention (H=4)\nQ=q_e, K=V=v_c",
         TRAINABLE_COLOR, TRAINABLE_EDGE, bold=True)
    _arrow(ax, 2, 7.1, 4.2, 6.65)
    _arrow(ax, 8, 7.1, 6, 6.65)

    # Outputs: ctx and alpha_avg
    ax.text(3.5, 5.4, "ctx [B, 384]", ha="center", fontsize=7, color="#333")
    ax.text(7, 5.4, "\u03b1_avg [B, 49]", ha="center", fontsize=7, color="#333")
    _arrow(ax, 4.2, 5.75, 3.5, 5.55)
    _arrow(ax, 6, 5.75, 7, 5.55)

    # ── Concat + Score MLP ──
    _box(ax, 5, 4.5, 3.5, 0.9, "Concat [ctx; \u03b1_avg] (433-dim)\nScore MLP \u2192 49 scores",
         TRAINABLE_COLOR, TRAINABLE_EDGE)
    _arrow(ax, 3.5, 5.3, 4.2, 4.95)
    _arrow(ax, 7, 5.3, 6, 4.95)

    # ── Bilinear Upsample ──
    _box(ax, 5, 3.3, 3.5, 0.7, "Bilinear Upsample 7\u00d77 \u2192 24\u00d724",
         OP_COLOR, OP_EDGE)
    _arrow(ax, 5, 4.05, 5, 3.65)

    # ── Instance Norm ──
    _box(ax, 5, 2.4, 3, 0.6, "Instance Norm (mean=0, std=1)",
         OP_COLOR, OP_EDGE)
    _arrow(ax, 5, 2.95, 5, 2.7)
    ax.text(5, 1.9, "s_N [B, 576]", ha="center", fontsize=7, color="#333")

    # ── BudgetPredictor branch ──
    _box(ax, 10.5, 4.5, 2.8, 1.0,
         "BudgetPredictor\n384 \u2192 64 \u2192 1\nsigmoid \u2192 k_frac",
         NEW_COLOR, NEW_EDGE, bold=True)
    # ctx → BudgetPredictor
    _arrow(ax, 3.5, 5.2, 3.5, 4.8)  # ctx goes down
    ax.annotate(
        "", xy=(9.1, 4.5), xytext=(3.5, 4.8),
        arrowprops=dict(arrowstyle="-|>", color=NEW_EDGE, lw=1.5,
                        connectionstyle="arc3,rad=-0.15"),
    )
    ax.text(6.5, 4.15, "ctx", fontsize=7, color=NEW_EDGE)

    ax.text(10.5, 3.7, "k_frac [B] \u2208 [0.0625, 0.5]", ha="center",
            fontsize=7, color=NEW_EDGE)
    ax.text(12.2, 5.0, "NEW", fontsize=8, fontweight="bold", color=NEW_EDGE,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFF3E0", edgecolor=NEW_EDGE))

    # ── Outputs box ──
    _box(ax, 5, 0.8, 4.5, 1.0,
         "Gumbel-Sigmoid STE\nFixed-k (k=144) or Adaptive (k_frac per sample)\nOutputs: idx, mask_ste, k_effective",
         TRAINABLE_COLOR, TRAINABLE_EDGE)
    _arrow(ax, 5, 1.8, 5, 1.3)
    _arrow(ax, 10.5, 3.6, 7.2, 1.3, color=NEW_EDGE)  # k_frac → STE

    # Save
    for ext in ("png", "svg"):
        path = os.path.join(ASSETS_DIR, f"tpn_architecture.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"Saved {path}")
    plt.close(fig)


def generate_training_pipeline():
    """Generate training pipeline diagram with adaptive budget phase."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [1.2, 1]})

    # ── Top: Phase boxes ──
    ax = axes[0]
    ax.set_xlim(-0.5, 14)
    ax.set_ylim(-0.5, 4)
    ax.axis("off")

    ax.text(7, 3.7, "Training Curriculum", ha="center", fontsize=14, fontweight="bold")

    phases = [
        ("Phase 0\nTPN Warmup", "1 epoch\n~10K steps",
         "Train TPN only\nLLaVA frozen\n\u03c4=1.0, \u03bb=0",
         FROZEN_COLOR, FROZEN_EDGE),
        ("Phase 1\nJoint Finetuning", "3 epochs\n~50K steps",
         "TPN + LoRA + proj\nVQAv2 + GQA\n\u03c4: 1.0\u21920.1",
         TRAINABLE_COLOR, TRAINABLE_EDGE),
        ("Phase 1-Adaptive\nAdaptive Budget", "40K steps",
         "From Phase 1 ckpt\n+BudgetPredictor\n\u03bb_budget sweep",
         NEW_COLOR, NEW_EDGE),
        ("Phase 2\nDomain Transfer", "1 epoch\n~10K steps",
         "TextVQA finetuning\nLow LR\n\u03c4=0.1",
         "#E3F2FD", "#1565C0"),
    ]

    for i, (title, duration, desc, color, edge) in enumerate(phases):
        x = 0.5 + i * 3.5
        # Phase box
        box = FancyBboxPatch(
            (x - 1.3, 0.3), 2.6, 3.0,
            boxstyle="round,pad=0.15", facecolor=color, edgecolor=edge, lw=2,
        )
        ax.add_patch(box)
        ax.text(x, 3.0, title, ha="center", va="center", fontsize=10, fontweight="bold")
        ax.text(x, 2.3, duration, ha="center", va="center", fontsize=8, color="#555")
        ax.text(x, 1.3, desc, ha="center", va="center", fontsize=8)

        # Arrow between phases
        if i < len(phases) - 1:
            ax.annotate(
                "", xy=(x + 1.5, 1.8), xytext=(x + 1.3, 1.8),
                arrowprops=dict(arrowstyle="-|>", color="#333", lw=2),
            )

    # Label new phase
    ax.text(7.5, 0.0, "NEW", fontsize=9, fontweight="bold", color=NEW_EDGE,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFF3E0", edgecolor=NEW_EDGE))

    # ── Bottom: Schedule curves ──
    ax2 = axes[1]
    steps = np.linspace(0, 40000, 1000)

    # Tau annealing
    tau = 1.0 - (1.0 - 0.3) * np.clip(steps / 40000, 0, 1)
    ax2.plot(steps, tau, "b-", lw=2, label="Temperature \u03c4 (1.0 \u2192 0.3)")

    # Budget warmup
    budget_active = np.where(steps < 2000, 0, 1)
    ax2.fill_between(steps, 0, budget_active * 0.3, alpha=0.15, color=NEW_EDGE,
                     label="Budget predictor active (after 2K steps)")

    # LoRA warmfreeze
    lora_active = np.where(steps < 1000, 0, 1)
    ax2.fill_between(steps, 0, lora_active * 0.15, alpha=0.15, color=TRAINABLE_EDGE,
                     label="LoRA active (after 1K steps)")

    ax2.set_xlabel("Training Steps", fontsize=11)
    ax2.set_ylabel("Value", fontsize=11)
    ax2.set_title("Adaptive Budget Training Schedule", fontsize=12, fontweight="bold")
    ax2.legend(loc="center right", fontsize=9)
    ax2.set_xlim(0, 40000)
    ax2.set_ylim(-0.05, 1.15)
    ax2.axvline(x=1000, color=TRAINABLE_EDGE, ls="--", alpha=0.5)
    ax2.axvline(x=2000, color=NEW_EDGE, ls="--", alpha=0.5)
    ax2.text(1000, 1.05, "LoRA\nunfreezes", ha="center", fontsize=7, color=TRAINABLE_EDGE)
    ax2.text(2000, 1.05, "Budget\nunfreezes", ha="center", fontsize=7, color=NEW_EDGE)

    fig.tight_layout()

    for ext in ("png", "svg"):
        path = os.path.join(ASSETS_DIR, f"training_pipeline.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"Saved {path}")
    plt.close(fig)


def generate_gumbel_ste():
    """Generate Gumbel-Sigmoid STE mechanism diagram (fixed + adaptive)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # ── Left: Fixed-k STE ──
    ax = axes[0]
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 8)
    ax.axis("off")
    ax.set_aspect("equal")
    ax.set_title("Fixed-k Gumbel-Sigmoid STE", fontsize=12, fontweight="bold", pad=15)

    _box(ax, 5, 7, 3, 0.7, "s_N [B, 576] (TPN scores)", DATA_COLOR, DATA_EDGE)
    _box(ax, 2.5, 5.5, 2.5, 0.7, "Gumbel Noise\ng ~ Gumbel(0,1)", OP_COLOR, OP_EDGE)
    _box(ax, 7.5, 5.5, 2.5, 0.7, "noisy = s + g", OP_COLOR, OP_EDGE)
    _arrow(ax, 5, 6.65, 2.5, 5.85)
    _arrow(ax, 5, 6.65, 7.5, 5.85)
    _arrow(ax, 2.5, 5.15, 7.5, 5.15)

    _box(ax, 3, 4, 2.5, 0.7, "Soft: \u03c3(noisy/\u03c4)\nm_soft [B, 576]",
         TRAINABLE_COLOR, TRAINABLE_EDGE)
    _box(ax, 7.5, 4, 2.5, 0.7, "Hard: topk(noisy, k)\nm_hard [B, 576]",
         "#FFCDD2", LOSS_EDGE)
    _arrow(ax, 7.5, 5.15, 3, 4.35)
    _arrow(ax, 7.5, 5.15, 7.5, 4.35)

    _box(ax, 5, 2.5, 4.5, 0.8, "STE: m_hard - m_soft.detach() + m_soft\nForward = hard, Backward = soft grad",
         TRAINABLE_COLOR, TRAINABLE_EDGE, bold=True)
    _arrow(ax, 3, 3.65, 4, 2.9)
    _arrow(ax, 7.5, 3.65, 6.5, 2.9)

    _box(ax, 5, 1.3, 3, 0.6, "mask_ste, idx [B, k]", DATA_COLOR, DATA_EDGE)
    _arrow(ax, 5, 2.1, 5, 1.6)

    # ── Right: Adaptive STE ──
    ax = axes[1]
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 8)
    ax.axis("off")
    ax.set_aspect("equal")
    ax.set_title("Adaptive Gumbel-Sigmoid STE (NEW)", fontsize=12, fontweight="bold",
                 pad=15, color=NEW_EDGE)

    _box(ax, 3.5, 7, 3, 0.7, "s_N [B, 576]", DATA_COLOR, DATA_EDGE)
    _box(ax, 8, 7, 2.5, 0.7, "k_frac [B]\nfrom BudgetPredictor", NEW_COLOR, NEW_EDGE)

    # Threshold computation
    _box(ax, 5, 5.8, 4, 0.7, "Per-sample threshold\nfrom sorted scores + k_frac",
         NEW_COLOR, NEW_EDGE)
    _arrow(ax, 3.5, 6.65, 4, 6.15)
    _arrow(ax, 8, 6.65, 6.5, 6.15)

    # Shifted scores
    _box(ax, 2.5, 4.5, 2.8, 0.7, "shifted = noisy - \u03b8\nm_soft = \u03c3(shifted/\u03c4)",
         TRAINABLE_COLOR, TRAINABLE_EDGE)
    _box(ax, 7.5, 4.5, 2.8, 0.7, "Per-sample topk(k_hard)\nm_hard [B, N]",
         "#FFCDD2", LOSS_EDGE)
    _arrow(ax, 5, 5.45, 2.5, 4.85)
    _arrow(ax, 5, 5.45, 7.5, 4.85)

    # k_effective
    _box(ax, 2.5, 3.2, 2.8, 0.7, "k_effective\n= \u03a3 m_soft\n(differentiable!)",
         NEW_COLOR, NEW_EDGE)
    _arrow(ax, 2.5, 4.15, 2.5, 3.55)

    # STE
    _box(ax, 5, 2.2, 4.5, 0.8, "STE: m_hard - m_soft.detach() + m_soft\n+ padding mask for variable k",
         TRAINABLE_COLOR, TRAINABLE_EDGE, bold=True)
    _arrow(ax, 2.5, 3.9, 4, 2.6)
    _arrow(ax, 7.5, 4.15, 6.5, 2.6)

    # Outputs
    _box(ax, 5, 0.9, 4, 0.7, "mask_ste, idx [B, k_max], k_eff [B]",
         DATA_COLOR, DATA_EDGE)
    _arrow(ax, 5, 1.8, 5, 1.25)

    # Budget loss
    _box(ax, 9, 2.0, 2, 0.6, "Budget Loss\n\u03bb * k_eff / N", LOSS_COLOR, LOSS_EDGE)
    _arrow(ax, 2.5, 3.0, 2.5, 2.5)
    ax.annotate(
        "", xy=(8, 2.0), xytext=(3.5, 2.0),
        arrowprops=dict(arrowstyle="-|>", color=NEW_EDGE, lw=1.5,
                        connectionstyle="arc3,rad=-0.2"),
    )

    fig.tight_layout()

    for ext in ("png", "svg"):
        path = os.path.join(ASSETS_DIR, f"gumbel_ste.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"Saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    print("Generating TC-ViTok architecture diagrams...")
    generate_system_architecture()
    generate_tpn_architecture()
    generate_training_pipeline()
    generate_gumbel_ste()
    print("\nAll diagrams generated in assets/")
