"""Optimizer configuration for TC-ViTok training.

This module sets up per-component learning rates and optimizer groups.
Different components are trained with different learning rates:
- TPN: 1e-3 (learns quickly, lightweight)
- LoRA: 2e-4 (careful finetuning of LM)
- Vision projector (MLP): 5e-4 (adapts vision features to LM)

Paper reference: Section 7.3 (Optimization Details)
"""

from typing import List, Dict, Any

import torch.nn as nn
from torch.optim import AdamW, Optimizer


def create_optimizer(
    model: nn.Module,
    lr_tpn: float = 1e-3,
    lr_lora: float = 2e-4,
    lr_mlp: float = 5e-4,
    weight_decay: float = 0.01,
    phase: int = 1,
) -> Optimizer:
    """Create AdamW optimizer with per-component learning rates.

    Different model components have different learning rate requirements:
    - TPN: Higher LR (1e-3) — small network, learns quickly
    - LoRA: Lower LR (2e-4) — careful finetuning to preserve pretrained weights
    - Projector: Medium LR (5e-4) — adapts frozen vision features

    Args:
        model: Full TC-ViTok model
        lr_tpn: Learning rate for TPN
        lr_lora: Learning rate for LoRA adapters
        lr_mlp: Learning rate for vision projector
        weight_decay: Weight decay for AdamW (all groups)
        phase: Training phase (0, 1, or 2)
                Phase 0: only TPN is trained
                Phase 1+: TPN + LoRA + projector

    Returns:
        optimizer: AdamW optimizer with parameter groups

    Example:
        >>> optimizer = create_optimizer(model, phase=0)
        >>> # Only TPN parameters have requires_grad=True in phase 0

    Paper reference: Section 7.3, Table 3 (Hyperparameters)
    """
    # TODO (Phase C): Set up parameter groups with different learning rates

    param_groups: List[Dict[str, Any]] = []

    if phase == 0:
        # Phase 0: Train TPN only
        tpn_params = [p for p in model.tpn.parameters() if p.requires_grad]
        param_groups.append({
            "params": tpn_params,
            "lr": lr_tpn,
            "weight_decay": weight_decay,
        })

    else:
        # Phase 1+: Train TPN + LoRA + projector (or StubLLM)

        # TPN parameters
        tpn_params = [p for p in model.tpn.parameters() if p.requires_grad]
        if tpn_params:
            param_groups.append({
                "params": tpn_params,
                "lr": lr_tpn,
                "weight_decay": weight_decay,
            })

        if hasattr(model, 'use_stub_llm') and model.use_stub_llm:
            # StubLLM: single param group for the stub
            stub_params = [p for p in model.llm.parameters() if p.requires_grad]
            if stub_params:
                param_groups.append({
                    "params": stub_params,
                    "lr": lr_lora,
                    "weight_decay": weight_decay,
                })
        elif hasattr(model, 'llava'):
            # Real LLaVA: LoRA + projector param groups
            # Use model._llm and model._mm_proj (canonical refs that work across
            # both old and new transformers LLaVA architectures)
            lora_params = []
            for name, param in model._llm.named_parameters():
                if param.requires_grad and "lora" in name.lower():
                    lora_params.append(param)
            if lora_params:
                param_groups.append({
                    "params": lora_params,
                    "lr": lr_lora,
                    "weight_decay": weight_decay,
                })

            # Vision projector (mm_projector) + VQA classification head
            # Both are trained at lr_mlp (medium LR, adapts visual→LM interface)
            projector_params = [
                p for p in model._mm_proj.parameters()
                if p.requires_grad
            ]
            if hasattr(model, "vqa_head"):
                projector_params += [
                    p for p in model.vqa_head.parameters()
                    if p.requires_grad
                ]
            if projector_params:
                param_groups.append({
                    "params": projector_params,
                    "lr": lr_mlp,
                    "weight_decay": weight_decay,
                })

    # Create AdamW optimizer
    optimizer = AdamW(param_groups)

    return optimizer


def get_optimizer_info(optimizer: Optimizer) -> Dict[str, Any]:
    """Get information about optimizer parameter groups.

    Useful for logging and debugging.

    Args:
        optimizer: PyTorch optimizer

    Returns:
        info: Dictionary with optimizer stats:
            - num_groups: Number of parameter groups
            - total_params: Total trainable parameters
            - group_info: List of dicts with per-group info
    """
    # TODO (Phase D): Extract optimizer group information

    num_groups = len(optimizer.param_groups)
    total_params = sum(
        sum(p.numel() for p in group["params"])
        for group in optimizer.param_groups
    )

    group_info = []
    for i, group in enumerate(optimizer.param_groups):
        num_params = sum(p.numel() for p in group["params"])
        group_info.append({
            "group_id": i,
            "lr": group["lr"],
            "weight_decay": group.get("weight_decay", 0.0),
            "num_params": num_params,
        })

    return {
        "num_groups": num_groups,
        "total_params": total_params,
        "group_info": group_info,
    }
