"""Curriculum scheduling for temperature and regularization strength.

This module implements the annealing schedules for:
1. Gumbel-Sigmoid temperature τ (controls exploration-exploitation)
2. Sparsity regularization strength λ (encourages token selection)

These schedules are critical for stable training of the TPN.

Paper reference: Section 7.2 (Curriculum Learning)
"""

import math


def get_tau(
    step: int,
    T: int,
    tau_min: float = 0.5,
    tau_max: float = 5.0,
) -> float:
    """Compute Gumbel-Sigmoid temperature at given training step.

    The temperature τ is annealed from high to low using cosine decay.
    This implements a curriculum from exploration (high τ) to exploitation (low τ).

    Schedule (Eq. 13):
        τ(t) = τ_min + 0.5 * (τ_max - τ_min) * (1 + cos(πt / T))

    Intuition:
        - High τ → soft selection, smooth gradients, exploration
        - Low τ → sharp selection, discrete masks, exploitation
        - Cosine decay provides smooth transition

    Args:
        step: Current training step (0-indexed)
        T: Total training steps
        tau_min: Minimum temperature (at end of training)
        tau_max: Maximum temperature (at start of training)

    Returns:
        tau: Temperature at current step

    Example:
        >>> get_tau(step=0, T=10000, tau_min=0.5, tau_max=5.0)
        5.0  # Start with high temperature
        >>> get_tau(step=10000, T=10000, tau_min=0.5, tau_max=5.0)
        0.5  # End with low temperature

    Paper reference: Eq. 13, Section 7.2
    """
    # TODO (Phase C): Implement cosine temperature schedule

    # Cosine annealing: starts at tau_max, ends at tau_min
    progress = min(step / T, 1.0)  # Clamp to [0, 1]
    tau = tau_min + 0.5 * (tau_max - tau_min) * (1 + math.cos(math.pi * progress))

    return tau


def get_lambda(
    step: int,
    T_warmup: int,
    lambda_max: float = 0.01,
) -> float:
    """Compute sparsity regularization strength at given training step.

    The regularization strength λ is warmed up from 0 to λ_max linearly.
    This prevents the TPN from collapsing to always selecting k tokens
    before learning meaningful selection patterns.

    Schedule (Eq. 14):
        λ(t) = λ_max * min(t / T_warmup, 1)

    Intuition:
        - λ = 0 initially → TPN learns to use all information
        - λ increases → TPN learns to be selective
        - After warmup → constant λ_max for steady sparsity pressure

    Args:
        step: Current training step (0-indexed)
        T_warmup: Number of warmup steps
        lambda_max: Maximum regularization strength

    Returns:
        lambda_val: Regularization strength at current step

    Example:
        >>> get_lambda(step=0, T_warmup=5000, lambda_max=0.01)
        0.0  # No regularization at start
        >>> get_lambda(step=2500, T_warmup=5000, lambda_max=0.01)
        0.005  # Halfway through warmup
        >>> get_lambda(step=5000, T_warmup=5000, lambda_max=0.01)
        0.01  # Full regularization after warmup

    Paper reference: Eq. 14, Section 7.2
    """
    # TODO (Phase C): Implement linear lambda warmup

    # Linear warmup: 0 → lambda_max over T_warmup steps
    progress = min(step / T_warmup, 1.0)  # Clamp to [0, 1]
    lambda_val = lambda_max * progress

    return lambda_val


def get_lr_schedule(
    step: int,
    warmup_steps: int,
    total_steps: int,
    lr_base: float,
    lr_min: float = 0.0,
) -> float:
    """Compute learning rate with warmup and cosine decay.

    Standard schedule used for all optimizers (TPN, LoRA, projector).

    Schedule:
        - Warmup: 0 → lr_base linearly over warmup_steps
        - Decay: lr_base → lr_min via cosine decay

    Args:
        step: Current training step
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
        lr_base: Base learning rate (after warmup)
        lr_min: Minimum learning rate (at end)

    Returns:
        lr: Learning rate at current step

    Paper reference: Section 7.3 (Optimization)
    """
    # TODO (Phase C): Implement warmup + cosine decay schedule

    if step < warmup_steps:
        # Linear warmup
        lr = lr_base * step / warmup_steps
    else:
        # Cosine decay
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        progress = min(progress, 1.0)
        lr = lr_min + 0.5 * (lr_base - lr_min) * (1 + math.cos(math.pi * progress))

    return lr
