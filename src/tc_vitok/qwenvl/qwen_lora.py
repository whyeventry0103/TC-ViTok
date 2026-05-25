"""LoRA helpers for the Qwen2-VL FM1 experiment."""

from __future__ import annotations

from typing import Iterable


def apply_qwen_lora(model, rank: int = 32, alpha: int = 64, targets: Iterable[str] | None = None):
    """Attach LoRA adapters to Qwen2-VL's language-model projection layers."""
    from peft import LoraConfig, TaskType, get_peft_model

    target_modules = list(targets or ("q_proj", "k_proj", "v_proj", "o_proj"))
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.05,
        target_modules=target_modules,
        bias="none",
    )
    return get_peft_model(model, config)


def mark_only_lora_trainable(model) -> None:
    """Freeze everything except LoRA adapter weights."""
    for name, param in model.named_parameters():
        param.requires_grad = "lora_" in name

