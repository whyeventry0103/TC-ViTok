"""Main training loop for TC-ViTok with multi-phase curriculum.

This module orchestrates the 3-phase training procedure:
- Phase 0: TPN warmup with frozen LLaVA
- Phase 1: Joint finetuning of TPN + LoRA + projector
- Phase 2: Domain transfer to TextVQA (optional)

Paper reference: Section 7 (Training Procedure)
"""

from contextlib import nullcontext
from typing import Optional, Dict, Any
from pathlib import Path

import torch
import torch.distributed as dist
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
    - Loss computation and backpropagation (BF16 mixed precision)
    - Logging and checkpointing (rank-0 only in DDP)
    - Evaluation during training

    Args:
        model: TC-ViTok model instance (may be wrapped in DDP by caller)
        config: Training configuration dict
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
        device: Device to train on
        local_rank: DDP local rank (-1 = single-GPU / non-DDP)

    Paper reference: Section 7
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: str = "cuda",
        local_rank: int = -1,
    ):
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.local_rank = local_rank
        self.is_ddp = local_rank >= 0
        # rank-0 is the only process that logs / saves
        self.is_main = (local_rank <= 0)

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
        self.lambda_entropy = config["training"]["lambda_entropy"]

        # Logging
        self.log_interval = config["logging"]["log_interval"]
        self.eval_interval = config["logging"]["eval_interval"]
        self.save_interval = config["logging"]["save_interval"]
        self.output_dir = Path(config["logging"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Mixed precision: use BF16 when available (RTX Ada / A100 / H100)
        mp_setting = config.get("training", {}).get("mixed_precision", "no")
        if mp_setting == "bf16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            self.amp_dtype = torch.bfloat16
        elif mp_setting == "fp16" and torch.cuda.is_available():
            self.amp_dtype = torch.float16
        else:
            self.amp_dtype = None  # full precision

        # Set up phase-specific training (call on underlying module for DDP)
        raw_model = self.model.module if self.is_ddp else self.model
        raw_model.set_training_phase(self.phase)

        # Create optimizer (operates on raw model parameters)
        self.optimizer = create_optimizer(
            raw_model,
            lr_tpn=config["training"]["lr_tpn"],
            lr_lora=config["training"]["lr_lora"],
            lr_mlp=config["training"]["lr_mlp"],
            weight_decay=config["training"]["weight_decay"],
            phase=self.phase,
        )

        # Store initial learning rates for scheduling
        for param_group in self.optimizer.param_groups:
            param_group["initial_lr"] = param_group["lr"]

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")

    def train(self):
        """Run full training loop for current phase.

        Returns:
            final_metrics: Dict with final training metrics
        """
        if self.is_main:
            print(f"\n{'='*60}")
            print(f"Starting Phase {self.phase} Training")
            print(f"{'='*60}")
            print(f"Max steps: {self.max_steps}")
            print(f"Device: {self.device}")
            if self.amp_dtype is not None:
                print(f"Mixed precision: {self.amp_dtype}")
            if self.is_ddp:
                print(f"DDP world size: {dist.get_world_size()}")

            # Log optimizer info
            opt_info = get_optimizer_info(self.optimizer)
            print(f"Optimizer: {opt_info['num_groups']} parameter groups")
            print(f"Total trainable params: {opt_info['total_params']:,}")

        self.model.train()

        # Progress bar — only on rank 0 to avoid stdout interleaving
        pbar = tqdm(total=self.max_steps, desc=f"Phase {self.phase}",
                    disable=not self.is_main)

        while self.global_step < self.max_steps:
            # Shuffle data differently each epoch in DDP
            if self.is_ddp and hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(self.epoch)

            for batch in self.train_loader:
                # Check if training is complete
                if self.global_step >= self.max_steps:
                    break

                # Training step
                loss_dict = self.train_step(batch)

                # Logging (rank-0 only)
                if self.is_main and self.global_step % self.log_interval == 0:
                    self.log_metrics(loss_dict)

                # Validation — rank 0 runs validate(), other ranks wait at barrier.
                # IMPORTANT: outer condition must be deterministic across all ranks
                # (based on global_step, not rank-local values) to prevent deadlock.
                if self.val_loader and self.global_step % self.eval_interval == 0:
                    if self.is_main:
                        val_metrics = self.validate()
                        self.log_metrics(val_metrics, prefix="val")
                    if self.is_ddp:
                        dist.barrier()  # sync all ranks after validation

                # Checkpointing — rank 0 saves, other ranks wait at barrier
                if self.global_step % self.save_interval == 0:
                    if self.is_main:
                        self.save_checkpoint()
                    if self.is_ddp:
                        dist.barrier()  # sync all ranks after checkpoint

                pbar.update(1)

            self.epoch += 1

        pbar.close()
        if self.is_main:
            print(f"\n{'='*60}")
            print(f"Phase {self.phase} Training Complete")
            print(f"{'='*60}\n")
            # Always save a final checkpoint at the end of training
            # Save as best.ckpt too — Phase 1 loads this for resumption
            self.save_checkpoint(is_best=True)
        if self.is_ddp:
            dist.barrier()  # sync all ranks before returning

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
        # Get current curriculum parameters
        tau = get_tau(self.global_step, self.max_steps, self.tau_min, self.tau_max)
        lambda_sparsity = get_lambda(
            self.global_step, self.T_warmup, self.lambda_max
        )

        # Move batch to device
        images_full = batch["images_full"].to(self.device)
        labels = batch["labels"].to(self.device)

        # BF16 / FP16 autocast context
        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=self.amp_dtype)
            if self.amp_dtype is not None and self.device != "cpu"
            else nullcontext()
        )

        # Access raw model for attributes only (k, etc.) — never for forward()
        raw_model = self.model.module if self.is_ddp else self.model

        # Build kwargs for unified forward() — DDP requires all calls go
        # through self.model (the wrapper) so gradient all-reduce hooks fire.
        with amp_ctx:
            fwd_kwargs = dict(
                images_full=images_full,
                labels=labels,
                tau=tau,
                return_idx=True,
            )
            if "v_coarse" in batch and "q_embed" in batch:
                fwd_kwargs["v_coarse"] = batch["v_coarse"].to(self.device)
                fwd_kwargs["q_embed"] = batch["q_embed"].to(self.device)
                if "v_full" in batch:
                    fwd_kwargs["v_full"] = batch["v_full"].to(self.device)
                fwd_kwargs["questions"] = batch.get("questions")
            else:
                fwd_kwargs["images_coarse"] = batch["images_coarse"].to(self.device)
                fwd_kwargs["questions"] = batch["questions"]

            outputs = self.model(**fwd_kwargs)

        # Compute losses (scores for L1 sparsity, idx for entropy)
        # Phase 0: skip VQA loss — TPN receives no gradient from VQA (gather is
        # non-differentiable) and the randomly-initialized vqa_head produces
        # large float16 logits that overflow to NaN, corrupting total loss.
        skip_vqa = (self.phase == 0)
        loss_dict = compute_total_loss(
            logits=None if skip_vqa else outputs.get("logits"),
            labels=None if skip_vqa else labels,
            scores=outputs["scores"],
            idx=outputs["idx"],
            k=raw_model.k,
            lambda_sparsity=lambda_sparsity,
            lambda_entropy=self.lambda_entropy,
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

            # Update learning rates per group
            warmup_steps = self.config["training"].get("warmup_steps", 1000)
            for param_group in self.optimizer.param_groups:
                base_lr = param_group.get("initial_lr", param_group["lr"])
                param_group["lr"] = get_lr_schedule(
                    self.global_step, warmup_steps, self.max_steps, base_lr,
                )

        self.global_step += 1

        # Add curriculum params to loss dict for logging
        loss_dict["tau"] = tau
        loss_dict["lambda"] = lambda_sparsity

        return {k: v.detach() if isinstance(v, torch.Tensor) else v
                for k, v in loss_dict.items()}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation loop.

        Uses the raw (unwrapped) model to avoid DDP deadlock: only rank 0
        calls validate(), so using the DDP wrapper would hang waiting for
        rank 1 to participate in the all-reduce.

        Returns:
            metrics: Dictionary with validation metrics
        """
        # Use raw model — DDP forward requires all ranks; validate runs on rank 0 only
        raw_model = self.model.module if self.is_ddp else self.model
        raw_model.eval()

        total_loss = 0.0
        total_vqa = 0.0
        total_sparsity = 0.0
        total_entropy = 0.0
        num_batches = 0

        # Current curriculum parameters
        tau = get_tau(self.global_step, self.max_steps, self.tau_min, self.tau_max)
        lambda_sparsity = get_lambda(
            self.global_step, self.T_warmup, self.lambda_max
        )

        # Use same autocast context as training to avoid dtype mismatches
        # (mm_projector weights are float16, inputs are float32 without autocast)
        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=self.amp_dtype)
            if self.amp_dtype is not None and self.device != "cpu"
            else nullcontext()
        )

        # Limit validation to 100 batches to prevent multi-hour hangs.
        # Full VQAv2 val set is ~13K batches — at ~1.8 it/s that's ~2 hours.
        max_val_batches = self.config.get("logging", {}).get("max_val_batches", 100)
        for batch in self.val_loader:
            if num_batches >= max_val_batches:
                break
            images_full = batch["images_full"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Use cached coarse/text features when available (no images_coarse key)
            with amp_ctx:
                if "v_coarse" in batch and "q_embed" in batch:
                    outputs = raw_model.forward_cached(
                        v_coarse=batch["v_coarse"].to(self.device),
                        q_embed=batch["q_embed"].to(self.device),
                        images_full=images_full,
                        questions=batch.get("questions"),
                        labels=labels,
                        tau=tau,
                        return_idx=True,
                    )
                else:
                    outputs = raw_model(
                        images_coarse=batch["images_coarse"].to(self.device),
                        images_full=images_full,
                        questions=batch["questions"],
                        labels=labels,
                        tau=tau,
                        return_idx=True,
                    )

            # Compute losses (skip VQA in Phase 0 — same as train_step)
            skip_vqa = (self.phase == 0)
            loss_dict = compute_total_loss(
                logits=None if skip_vqa else outputs.get("logits"),
                labels=None if skip_vqa else labels,
                scores=outputs["scores"],
                idx=outputs["idx"],
                k=raw_model.k,
                lambda_sparsity=lambda_sparsity,
                lambda_entropy=self.lambda_entropy,
            )

            total_loss += loss_dict["total"].item()
            total_vqa += loss_dict["vqa"].item()
            total_sparsity += loss_dict["sparsity"].item()
            total_entropy += loss_dict["entropy"].item()
            num_batches += 1

        raw_model.train()

        n = max(num_batches, 1)
        val_loss = total_loss / n
        metrics = {
            "loss": val_loss,
            "vqa": total_vqa / n,
            "sparsity": total_sparsity / n,
            "entropy": total_entropy / n,
        }

        # Track best validation loss
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.save_checkpoint(is_best=True)

        return metrics

    def log_metrics(self, metrics: Dict[str, Any], prefix: str = "train"):
        """Log metrics to console and wandb.

        Args:
            metrics: Dictionary of metrics to log
            prefix: Prefix for metric names ("train" or "val")
        """
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
        # Unwrap DDP so checkpoint is always a plain state dict
        raw_model = self.model.module if self.is_ddp else self.model
        checkpoint = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }

        ckpt_path = self.output_dir / f"checkpoint_step_{self.global_step}.pt"
        try:
            torch.save(checkpoint, ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

            if is_best:
                best_path = self.output_dir / "best.ckpt"
                torch.save(checkpoint, best_path)
                print(f"Best checkpoint saved: {best_path}")

            # Clean up old checkpoints to prevent disk full — keep last 2 + best
            self._cleanup_old_checkpoints(keep=2)

        except (RuntimeError, OSError) as e:
            print(f"WARNING: Checkpoint save failed: {e}")
            print("Training will continue — prior checkpoints are still valid.")
            # Remove corrupt partial file
            if ckpt_path.exists():
                ckpt_path.unlink()

    def _cleanup_old_checkpoints(self, keep: int = 2):
        """Remove old checkpoints, keeping the N most recent + best.ckpt."""
        import glob
        pattern = str(self.output_dir / "checkpoint_step_*.pt")
        ckpt_files = sorted(glob.glob(pattern), key=lambda f: int(f.split("_step_")[1].split(".")[0]))
        # Keep the last `keep` checkpoints
        for old_ckpt in ckpt_files[:-keep]:
            try:
                Path(old_ckpt).unlink()
                print(f"Cleaned up old checkpoint: {old_ckpt}")
            except OSError:
                pass

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint.

        For same-phase resume (e.g., crash recovery), restores step/epoch/optimizer.
        For cross-phase resume (e.g., Phase 0 → Phase 1), loads model weights only
        and resets step/epoch to 0 with fresh optimizer state.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)

        raw_model = self.model.module if self.is_ddp else self.model
        # strict=False: Phase transitions change trainable params — Phase 0
        # checkpoint won't have LoRA/projector weights needed by Phase 1.
        result = raw_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        if result.missing_keys:
            print(f"Checkpoint missing keys (expected for phase change): {len(result.missing_keys)} keys")
        if result.unexpected_keys:
            print(f"Checkpoint unexpected keys: {result.unexpected_keys}")

        # Detect cross-phase resume: checkpoint phase != current trainer phase
        ckpt_phase = checkpoint.get("config", {}).get("phase", self.phase)
        cross_phase = (ckpt_phase != self.phase)

        if cross_phase:
            # Cross-phase: keep model weights but reset training state
            print(f"Cross-phase resume (phase {ckpt_phase} → {self.phase}): "
                  f"loaded model weights, reset step/epoch/optimizer to 0")
        else:
            # Same-phase: restore full training state
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except (ValueError, RuntimeError) as e:
                print(f"WARNING: Could not load optimizer state: {e}")
                print("Continuing with fresh optimizer state.")
            self.global_step = checkpoint["global_step"]
            self.epoch = checkpoint["epoch"]
            print(f"Same-phase resume from step {self.global_step}")

        print(f"Checkpoint loaded: {checkpoint_path}")
