"""Main training loop for TC-ViTok with multi-phase curriculum.

This module orchestrates the 3-phase training procedure:
- Phase 0: TPN warmup with frozen LLaVA
- Phase 1: Joint finetuning of TPN + LoRA + projector
- Phase 2: Domain transfer to TextVQA (optional)

Paper reference: Section 7 (Training Procedure)
"""

from typing import Optional, Dict, Any
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from .losses import compute_total_loss
from .curriculum import get_tau, get_lambda, get_lr_schedule
from .optimizer import create_optimizer, get_optimizer_info


class TCViTokTrainer:
    """Multi-phase trainer for TC-ViTok model.

    Handles all aspects of training:
    - Phase-specific parameter freezing
    - Curriculum scheduling (τ, λ, LR)
    - Loss computation and backpropagation
    - Logging and checkpointing
    - Evaluation during training

    Args:
        model: TC-ViTok model instance
        config: Training configuration dict
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
        device: Device to train on

    Paper reference: Section 7
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Extract config
        self.phase = config.get("phase", 1)
        self.max_steps = config["training"]["max_steps"]
        self.grad_accum_steps = config["training"]["grad_accum_steps"]
        self.max_grad_norm = config["training"]["max_grad_norm"]

        # Curriculum params
        self.tau_min = config["training"]["tau_min"]
        self.tau_max = config["training"]["tau_max"]
        self.lambda_max = config["training"]["lambda_max"]
        self.T_warmup = config["training"]["T_warmup"]

        # Regularization weights
        self.lambda_diversity = config["training"]["lambda_diversity"]
        self.lambda_entropy = config["training"]["lambda_entropy"]

        # Logging
        self.log_interval = config["logging"]["log_interval"]
        self.eval_interval = config["logging"]["eval_interval"]
        self.save_interval = config["logging"]["save_interval"]
        self.output_dir = Path(config["logging"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # TODO (Phase C): Set up phase-specific training
        self.model.set_training_phase(self.phase)

        # TODO (Phase C): Create optimizer
        self.optimizer = create_optimizer(
            self.model,
            lr_tpn=config["training"]["lr_tpn"],
            lr_lora=config["training"]["lr_lora"],
            lr_mlp=config["training"]["lr_mlp"],
            weight_decay=config["training"]["weight_decay"],
            phase=self.phase,
        )

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")

    def train(self):
        """Run full training loop for current phase.

        Returns:
            final_metrics: Dict with final training metrics
        """
        # TODO (Phase C): Implement main training loop

        print(f"\n{'='*60}")
        print(f"Starting Phase {self.phase} Training")
        print(f"{'='*60}")
        print(f"Max steps: {self.max_steps}")
        print(f"Device: {self.device}")

        # Log optimizer info
        opt_info = get_optimizer_info(self.optimizer)
        print(f"Optimizer: {opt_info['num_groups']} parameter groups")
        print(f"Total trainable params: {opt_info['total_params']:,}")

        self.model.train()

        # Progress bar
        pbar = tqdm(total=self.max_steps, desc=f"Phase {self.phase}")

        while self.global_step < self.max_steps:
            for batch in self.train_loader:
                # Check if training is complete
                if self.global_step >= self.max_steps:
                    break

                # Training step
                loss_dict = self.train_step(batch)

                # Logging
                if self.global_step % self.log_interval == 0:
                    self.log_metrics(loss_dict)

                # Validation
                if self.val_loader and self.global_step % self.eval_interval == 0:
                    val_metrics = self.validate()
                    self.log_metrics(val_metrics, prefix="val")

                # Checkpointing
                if self.global_step % self.save_interval == 0:
                    self.save_checkpoint()

                pbar.update(1)

            self.epoch += 1

        pbar.close()
        print(f"\n{'='*60}")
        print(f"Phase {self.phase} Training Complete")
        print(f"{'='*60}\n")

        return {"final_step": self.global_step, "final_epoch": self.epoch}

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Single training step.

        Args:
            batch: Batch dict with:
                - images_coarse: [B, 3, 98, 98]
                - images_full: [B, 3, 336, 336]
                - questions: List[str]
                - labels: [B, seq_len]

        Returns:
            loss_dict: Dictionary with all loss components
        """
        # TODO (Phase C): Implement training step

        # Get current curriculum parameters
        tau = get_tau(self.global_step, self.max_steps, self.tau_min, self.tau_max)
        lambda_sparsity = get_lambda(
            self.global_step, self.T_warmup, self.lambda_max
        )

        # Move batch to device
        images_coarse = batch["images_coarse"].to(self.device)
        images_full = batch["images_full"].to(self.device)
        questions = batch["questions"]
        labels = batch["labels"].to(self.device)

        # Forward pass (always return mask for loss computation)
        outputs = self.model(
            images_coarse=images_coarse,
            images_full=images_full,
            questions=questions,
            labels=labels,
            tau=tau,
            return_mask=True,  # needed for entropy/diversity losses
        )

        # Compute losses (scores required for L1 sparsity on raw scores)
        loss_dict = compute_total_loss(
            logits=outputs.get("logits"),
            labels=labels,
            scores=outputs["scores"],
            mask=outputs.get("mask", outputs["scores"]),
            k=self.model.k,
            lambda_sparsity=lambda_sparsity,
            lambda_entropy=self.lambda_entropy,
            lambda_diversity=self.lambda_diversity,
        )

        # Backpropagation with gradient accumulation
        loss = loss_dict["total"] / self.grad_accum_steps
        loss.backward()

        if (self.global_step + 1) % self.grad_accum_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.max_grad_norm,
            )

            # Optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad()

        self.global_step += 1

        # Add curriculum params to loss dict for logging
        loss_dict["tau"] = tau
        loss_dict["lambda"] = lambda_sparsity

        return {k: v.detach() if isinstance(v, torch.Tensor) else v
                for k, v in loss_dict.items()}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation loop.

        Returns:
            metrics: Dictionary with validation metrics
        """
        # TODO (Phase D): Implement validation loop
        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            # Validation step (similar to train_step but without backprop)
            # Placeholder for now
            pass

        self.model.train()

        return {"loss": total_loss / max(num_batches, 1)}

    def log_metrics(self, metrics: Dict[str, Any], prefix: str = "train"):
        """Log metrics to console and wandb.

        Args:
            metrics: Dictionary of metrics to log
            prefix: Prefix for metric names ("train" or "val")
        """
        # TODO (Phase D): Implement logging

        # Console logging
        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        tqdm.write(f"[Step {self.global_step}] {prefix} | {metric_str}")

        # WandB logging
        if wandb.run is not None:
            wandb.log(
                {f"{prefix}/{k}": v for k, v in metrics.items()},
                step=self.global_step,
            )

    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint.

        Args:
            is_best: Whether this is the best checkpoint so far
        """
        # TODO (Phase D): Implement checkpointing

        checkpoint = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }

        # Save latest checkpoint
        ckpt_path = self.output_dir / f"checkpoint_step_{self.global_step}.pt"
        torch.save(checkpoint, ckpt_path)

        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / "best.ckpt"
            torch.save(checkpoint, best_path)

        print(f"Checkpoint saved: {ckpt_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        # TODO (Phase D): Implement checkpoint loading

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.epoch = checkpoint["epoch"]

        print(f"Checkpoint loaded: {checkpoint_path}")
        print(f"Resuming from step {self.global_step}")
