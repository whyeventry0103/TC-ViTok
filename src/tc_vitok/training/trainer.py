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
from .curriculum import get_tau, get_lambda, get_lambda_spread, get_lr_schedule
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
        self.lambda_diversity = config["training"].get("lambda_diversity", 0.0)
        # Phase 1.5: spatial spread penalty and teacher distillation
        self.lambda_spread_max = config["training"].get("lambda_spread", 0.0)
        self.lambda_distill = config["training"].get("lambda_distill", 0.0)
        self.spread_decay_start = config["training"].get("spread_decay_start", 30000)
        self.spread_decay_end   = config["training"].get("spread_decay_end",   50000)
        # Phase 1.5 v4: boundary margin + slot orthogonality (replaces spread + KL distill)
        self.lambda_boundary = config["training"].get("lambda_boundary", 0.0)
        self.lambda_slot_orth = config["training"].get("lambda_slot_orth", 0.0)
        self.boundary_margin = config["training"].get("boundary_margin", 0.1)
        # Phase 1.5 v5: residual pooler distillation
        self.lambda_residual_distill = config["training"].get("lambda_residual_distill", 0.0)

        # LoRA warm-freeze: keep LoRA gradients zeroed for the first N steps so TPN
        # has time to learn query-conditioned routing before LoRA compensates.
        self.lora_warmfreeze_steps = config["training"].get("lora_warmfreeze_steps", 0)

        # Pooler warm-up: zero TPN gradients for the first N steps so UnselectedPooler
        # learns stable background representations before TPN routing starts moving.
        # Steps 0-pooler_warmup: only pooler + LoRA + projector update.
        # Steps pooler_warmup+: everything updates normally.
        self.pooler_warmup_steps = config["training"].get("pooler_warmup_steps", 0)

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
                - answer_str: List[str]  — raw answer strings for next-token prediction

        Returns:
            loss_dict: Dictionary with all loss components
        """
        # Get current curriculum parameters
        tau = get_tau(self.global_step, self.max_steps, self.tau_min, self.tau_max)
        lambda_sparsity = get_lambda(
            self.global_step, self.T_warmup, self.lambda_max
        )
        lambda_spread = get_lambda_spread(
            self.global_step,
            max_val=self.lambda_spread_max,
            decay_start=self.spread_decay_start,
            decay_end=self.spread_decay_end,
        )

        # Move batch to device
        images_full = batch["images_full"].to(self.device)
        answer_strings = batch.get("answer_str")  # List[str] — stays on CPU

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
        # ── Teacher distillation pass (Phase 1.5) ────────────────────────────
        # Run frozen base LLaVA with all 576 tokens BEFORE student forward.
        # Sequential execution: teacher activations freed before student allocates.
        # Skipped when: stub mode, no teacher-dependent loss, or no answer_strings.
        # Fires for: KL distill, boundary margin, or residual pooler distillation.
        teacher_importance = None
        if (
            (self.lambda_distill > 0 or self.lambda_boundary > 0
             or self.lambda_residual_distill > 0)
            and not raw_model.use_stub_llm
            and answer_strings is not None
            and "images_coarse" in batch    # full pipeline only (not cached path)
        ):
            images_coarse_t = batch["images_coarse"].to(self.device)
            questions_t = batch["questions"]
            with torch.no_grad():
                v_coarse_t = raw_model.coarse_encoder(images_coarse_t)
                q_embed_t  = raw_model.text_encoder(questions_t)
                del v_coarse_t, q_embed_t  # not needed for teacher
                v_full_t = raw_model.full_encoder(images_full)
                teacher_importance = raw_model._get_teacher_importance(
                    v_full_t, questions_t, answer_strings
                )
                del v_full_t  # free before student forward

        with amp_ctx:
            fwd_kwargs = dict(
                images_full=images_full,
                answer_strings=answer_strings,
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

        # mask_soft for spatial_spread_loss: sigmoid(s_N / tau).
        # Computed here to avoid storing it inside model (no extra backward graph).
        scores = outputs["scores"]
        mask_soft = torch.sigmoid(scores / (tau + 1e-8)) if lambda_spread > 0 else None

        # Compute losses.
        # mask_ste: routes entropy gradient to ALL N tokens (not just selected k).
        # mask_soft: used for spatial spread loss (sigmoid, differentiable).
        # q_embed: enables within-batch diversity loss.
        # teacher_importance: dense KL supervision from frozen base model.
        loss_dict = compute_total_loss(
            logits=outputs.get("logits"),
            labels=outputs.get("seq_labels"),
            scores=scores,
            idx=outputs["idx"],
            k=raw_model.k,
            lambda_sparsity=lambda_sparsity,
            lambda_entropy=self.lambda_entropy,
            lambda_diversity=self.lambda_diversity,
            lambda_spread=lambda_spread,
            lambda_distill=self.lambda_distill,
            lambda_boundary=self.lambda_boundary,
            lambda_slot_orth=self.lambda_slot_orth,
            lambda_residual_distill=self.lambda_residual_distill,
            boundary_margin=self.boundary_margin,
            mask_ste=outputs.get("mask_ste"),
            mask_soft=mask_soft,
            q_embed=outputs.get("q_embed"),
            teacher_importance=teacher_importance,
            v_summary_prenorm=outputs.get("v_summary_prenorm"),
            pooler_attn=outputs.get("pooler_attn"),
            unsel_idx=outputs.get("unsel_idx"),
        )

        # NaN/Inf guard: skip backward entirely to prevent weight corruption.
        # A single NaN backward pass applies NaN gradients to TPN/LoRA weights,
        # permanently corrupting them for all subsequent steps.
        if torch.isnan(loss_dict["total"]) or torch.isinf(loss_dict["total"]):
            if self.is_main:
                tqdm.write(f"[Step {self.global_step}] WARNING: NaN/Inf loss detected — skipping backward")
            self.optimizer.zero_grad()
            self.global_step += 1
            loss_dict["tau"] = tau
            loss_dict["lambda"] = lambda_sparsity
            return {k: v.detach() if isinstance(v, torch.Tensor) else v
                    for k, v in loss_dict.items()}

        # Backpropagation with gradient accumulation
        loss = loss_dict["total"] / self.grad_accum_steps
        loss.backward()

        if (self.global_step + 1) % self.grad_accum_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.max_grad_norm,
            )

            # LoRA warm-freeze: zero LoRA gradients for the first N steps so TPN
            # receives strong directional signal from entropy+diversity losses before
            # LoRA can compensate by memorizing a fixed-crop VQA mapping.
            if self.lora_warmfreeze_steps > 0 and self.global_step < self.lora_warmfreeze_steps:
                for name, param in raw_model.named_parameters():
                    if "lora_" in name:
                        param.grad = None  # None → AdamW skips entirely (no weight decay); zero_() doesn't

            # Pooler warm-up: zero TPN gradients so UnselectedPooler stabilizes
            # before TPN routing competes. TPN still participates in forward
            # (gumbel noise gives varied routing each batch), but its weights
            # don't update — the pooler learns on a slowly-drifting token set.
            if self.pooler_warmup_steps > 0 and self.global_step < self.pooler_warmup_steps:
                for param in raw_model.tpn.parameters():
                    param.grad = None  # None → AdamW skips entirely (no weight decay); zero_() doesn't

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
            answer_strings = batch.get("answer_str")  # List[str]

            # Use cached coarse/text features when available (no images_coarse key)
            with amp_ctx:
                if "v_coarse" in batch and "q_embed" in batch:
                    outputs = raw_model.forward_cached(
                        v_coarse=batch["v_coarse"].to(self.device),
                        q_embed=batch["q_embed"].to(self.device),
                        images_full=images_full,
                        questions=batch.get("questions"),
                        answer_strings=answer_strings,
                        tau=tau,
                        return_idx=True,
                    )
                else:
                    outputs = raw_model(
                        images_coarse=batch["images_coarse"].to(self.device),
                        images_full=images_full,
                        questions=batch["questions"],
                        answer_strings=answer_strings,
                        tau=tau,
                        return_idx=True,
                    )

            # No mask_ste at eval time (training=False → TPN returns None).
            # Entropy falls back to entropy_from_idx; diversity/spread/distill = 0.
            loss_dict = compute_total_loss(
                logits=outputs.get("logits"),
                labels=outputs.get("seq_labels"),
                scores=outputs["scores"],
                idx=outputs["idx"],
                k=raw_model.k,
                lambda_sparsity=lambda_sparsity,
                lambda_entropy=self.lambda_entropy,
                lambda_diversity=0.0,
                lambda_spread=0.0,
                lambda_distill=0.0,
                mask_ste=None,
                mask_soft=None,
                q_embed=None,
                teacher_importance=None,
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

        Only saves trainable parameters (TPN + LoRA adapters +
        mm_projector) rather than the full model state dict. This reduces
        checkpoint size from ~16GB to ~1-2GB by skipping the frozen 7B LLM
        backbone and frozen encoders. load_checkpoint uses strict=False so
        the missing frozen weights are simply kept from the pretrained model.

        Args:
            is_best: Whether this is the best checkpoint so far
        """
        # Unwrap DDP so checkpoint is always a plain state dict
        raw_model = self.model.module if self.is_ddp else self.model

        # Save only trainable parameters — skips frozen 7B LLM + frozen encoders
        # Keys use the module hierarchy path, not Python attribute names:
        #   tpn.*                          — TPN weights
        #   *lora_*                        — LoRA adapter weights (PEFT naming)
        #   *multi_modal_projector.*       — mm_projector (old+new LLaVA arch)
        trainable_state = {
            k: v for k, v in raw_model.state_dict().items()
            if k.startswith("tpn.")
            or "lora_" in k
            or "multi_modal_projector." in k
            or k.startswith("unsel_pooler.")   # Phase 1.5: 4-slot cross-attention pooler
        }

        checkpoint = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "model_state_dict": trainable_state,
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
