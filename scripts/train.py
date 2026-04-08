#!/usr/bin/env python3
"""CLI entry point for training TC-ViTok.

Usage:
    python scripts/train.py --config configs/phase0.yaml --output runs/phase0/
    python scripts/train.py --config configs/phase0.yaml --output runs/ --run-both-phases
    python scripts/train.py --config configs/phase1.yaml --resume runs/phase0/best.ckpt
    python scripts/train.py --config configs/base.yaml --max-steps 1 --stub --dummy-data

This script:
1. Loads configuration from YAML
2. Initializes model, datasets, and trainer
3. Runs Phase 0 (TPN warmup) and optionally Phase 1 (joint finetuning)
4. Saves checkpoints and logs to WandB

Paper reference: Section 7 (Training Procedure)
"""

import argparse
import os
from pathlib import Path
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DistributedSampler
import wandb
from omegaconf import OmegaConf

from tc_vitok.models.tc_vitok_model import TCViTokModel
from tc_vitok.data.datasets import (
    DummyVQADataset, VQAv2Dataset, GQADataset, collate_fn,
)
from tc_vitok.data.caching import CachedFeatureDataset, cached_collate_fn
from tc_vitok.training.trainer import TCViTokTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train TC-ViTok")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file (e.g., configs/phase0.yaml)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="runs",
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="tc-vitok",
        help="WandB project name",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override max training steps from config (useful for dry runs: --max-steps 1)",
    )
    parser.add_argument(
        "--run-both-phases",
        action="store_true",
        help=(
            "When config is phase0.yaml, automatically run Phase 1 after Phase 0 "
            "using configs/phase1.yaml from the same directory"
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to train on (auto-detected if not specified)",
    )
    parser.add_argument(
        "--stub",
        action="store_true",
        help="Use StubLLM instead of real LLaVA (for pipeline testing)",
    )
    parser.add_argument(
        "--dummy-data",
        action="store_true",
        help="Use DummyVQADataset for pipeline testing without real data",
    )
    parser.add_argument(
        "--cache-path",
        type=str,
        default=None,
        help="Path to HDF5 cache file (overrides config)",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable WandB logging",
    )
    return parser.parse_args()


def init_ddp() -> tuple[int, bool]:
    """Initialize DDP if LOCAL_RANK is set by torchrun.

    Returns:
        (local_rank, is_ddp) — local_rank=-1 means single-GPU / CPU mode
    """
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank < 0:
        return -1, False
    torch.cuda.set_device(local_rank)
    # Use a generous timeout for init — model loading can delay ranks significantly
    from datetime import timedelta
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
    return local_rank, True


def detect_device(local_rank: int = -1) -> str:
    """Auto-detect the best available device."""
    if local_rank >= 0:
        return f"cuda:{local_rank}"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_loaders(data_config, batch_size, num_workers, args, is_ddp: bool = False):
    """Build train and val DataLoaders from config and args.

    When is_ddp=True, train DataLoader uses DistributedSampler (no shuffle kwarg).

    Returns:
        train_loader, val_loader (val_loader may be None)
    """
    if args.dummy_data:
        print("Using DummyVQADataset for pipeline testing")
        stub_mode = args.stub
        train_dataset = DummyVQADataset(num_samples=256, stub_mode=stub_mode)
        val_dataset = DummyVQADataset(num_samples=64, stub_mode=stub_mode)
        # stub mode: encoders are None, so use cached_collate_fn (v_coarse/q_embed path)
        train_collate = cached_collate_fn if stub_mode else collate_fn
        val_collate = cached_collate_fn if stub_mode else collate_fn
    elif args.cache_path or data_config.get("use_cached_features", False):
        cache_path = args.cache_path or data_config.get("cache_path_train")
        print(f"Using cached features from {cache_path}")
        train_dataset = CachedFeatureDataset(cache_path=cache_path)
        val_dataset = None
        train_collate = cached_collate_fn
        val_collate = cached_collate_fn
    else:
        # Real dataset — requires annotation paths in config
        dataset_name = data_config.get("dataset", "vqav2")
        if dataset_name == "vqav2":
            train_dataset = VQAv2Dataset(
                annotations_path=data_config["train_annotations"],
                questions_path=data_config["train_questions"],
                image_dir=data_config["train_image_dir"],
                split="train",
                max_samples=data_config.get("max_samples"),
            )
            val_dataset = VQAv2Dataset(
                annotations_path=data_config["val_annotations"],
                questions_path=data_config["val_questions"],
                image_dir=data_config["val_image_dir"],
                split="val",
                answer_list=list(train_dataset.answer_to_idx.keys()),
                max_samples=data_config.get("max_samples"),
            )
        elif dataset_name == "gqa":
            train_dataset = GQADataset(
                questions_path=data_config["gqa_train_questions"],
                image_dir=data_config["gqa_train_image_dir"],
                max_samples=data_config.get("max_samples"),
            )
            val_dataset = GQADataset(
                questions_path=data_config["gqa_val_questions"],
                image_dir=data_config["gqa_val_image_dir"],
                answer_list=list(train_dataset.answer_to_idx.keys()),
                max_samples=data_config.get("max_samples"),
            )
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        train_collate = collate_fn
        val_collate = collate_fn

    pin_memory = data_config.get("pin_memory", False) and torch.cuda.is_available()

    if is_ddp:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            collate_fn=train_collate,
            pin_memory=pin_memory,
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=train_collate,
            pin_memory=pin_memory,
        )

    val_loader = None
    if val_dataset is not None:
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=val_collate,
            pin_memory=pin_memory,
        )

    return train_loader, val_loader


def run_phase(model, config_dict, train_loader, val_loader, device, resume=None, local_rank=-1):
    """Create trainer, optionally resume, and run one training phase.

    Args:
        model: TCViTokModel instance (already wrapped in DDP if multi-GPU)
        config_dict: Config dict for this phase
        train_loader: Training DataLoader
        val_loader: Validation DataLoader (may be None)
        device: Training device string
        resume: Optional checkpoint path to resume from
        local_rank: DDP local rank (-1 = single-GPU)

    Returns:
        final_metrics: Dict from trainer.train()
        phase_output_dir: Path where checkpoints were saved
    """
    trainer = TCViTokTrainer(
        model=model,
        config=config_dict,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        local_rank=local_rank,
    )

    if resume:
        trainer.load_checkpoint(resume)

    final_metrics = trainer.train()
    return final_metrics, trainer.output_dir


def load_config(config_path: str) -> dict:
    """Load a config YAML, merging with base.yaml when defaults: [base] is present.

    Phase configs only override keys they need; all other keys come from base.yaml.
    This replicates Hydra's defaults-list behaviour using plain OmegaConf.
    """
    raw = OmegaConf.load(config_path)
    raw_dict = OmegaConf.to_container(raw, resolve=False)

    if "defaults" in raw_dict:
        base_path = Path(config_path).parent / "base.yaml"
        if base_path.exists():
            base = OmegaConf.load(base_path)
            # Phase config wins; base fills in everything else
            merged = OmegaConf.merge(base, raw)
            result = OmegaConf.to_container(merged, resolve=True)
            result.pop("defaults", None)
            return result
        else:
            import warnings
            warnings.warn(
                f"Config {config_path} has 'defaults: [base]' but base.yaml was not "
                f"found at {base_path}. Keys missing from base may cause KeyError."
            )

    return OmegaConf.to_container(raw, resolve=True)


def main():
    args = parse_args()

    # MPS/Apple Silicon compatibility (no-op on Linux/CUDA)
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    # If DATA_ROOT is set, redirect all model caches to Google Drive.
    # HF_HOME  → transformers models (CLIP, LLaVA)
    # TORCH_HOME → timm models (DINOv2-S via PyTorch Hub)
    data_root = os.environ.get("DATA_ROOT")
    if data_root:
        hf_home = os.path.join(data_root, "hf_cache")
        torch_home = os.path.join(data_root, "torch_cache")
        os.environ.setdefault("HF_HOME", hf_home)
        os.environ.setdefault("TRANSFORMERS_CACHE", hf_home)
        os.environ.setdefault("TORCH_HOME", torch_home)
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    # DDP initialization (sets LOCAL_RANK via torchrun env var)
    local_rank, is_ddp = init_ddp()
    is_main = local_rank <= 0  # rank 0 or single-GPU

    # Load config (merges base.yaml automatically for phase configs)
    config_dict = load_config(args.config)
    config_path = Path(args.config)
    phase = config_dict.get("phase", 1)

    # Apply --max-steps override (useful for dry runs: --max-steps 1)
    if args.max_steps is not None:
        config_dict["training"]["max_steps"] = args.max_steps

    device = args.device or detect_device(local_rank)

    if is_main:
        print(f"Loaded config from {args.config}")
        print(f"Phase: {phase}")
        print(f"Device: {device}")
        if is_ddp:
            print(f"DDP enabled — local_rank={local_rank}, world_size={dist.get_world_size()}")
        if args.max_steps is not None:
            print(f"max_steps overridden to {args.max_steps}")

    # Initialize model (shared across phases when running both)
    tpn_config = config_dict.get("model", {}).get("tpn", {})
    lora_config = config_dict.get("model", {}).get("lora", {})
    k = config_dict.get("model", {}).get("k", 144)

    # Disable HuggingFace Hub online checks — all models are pre-downloaded.
    # This prevents HTTP HEAD requests that timeout and stall one rank while the
    # other continues, leading to NCCL timeout during DDP init.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    model = TCViTokModel(
        k=k,
        tpn_config=tpn_config,
        use_stub_llm=args.stub,
        lora_config=lora_config,
        load_encoders=not args.stub,
    )

    # Log param count on ALL ranks for DDP debugging
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = model.get_num_params()
    print(f"[Rank {local_rank}] Model created (stub={args.stub}, k={k})")
    print(f"[Rank {local_rank}] Total params: {n_params:,}, Trainable: {n_trainable:,}")
    import sys
    sys.stdout.flush()

    # Phase 0 optimization: skip DDP — only 887K TPN params train, so the
    # gradient all-reduce overhead on the 7.6B model wrapper makes 2-GPU DDP
    # ~34% SLOWER than single GPU. Run Phase 0 on rank 0 only; rank 1 exits.
    # DDP is enabled for Phase 1 (35M trainable params, overhead is justified).
    skip_ddp = is_ddp and phase == 0
    if skip_ddp:
        if local_rank != 0:
            # Non-main ranks exit cleanly for Phase 0
            print(f"[Rank {local_rank}] Phase 0 runs single-GPU (DDP overhead > benefit for 887K params). Exiting.")
            sys.stdout.flush()
            dist.destroy_process_group()
            return
        else:
            print(f"[Rank 0] Phase 0: skipping DDP (887K trainable params — single GPU is faster)")
            sys.stdout.flush()
            # Treat as non-DDP for the rest of this run
            is_ddp = False
            is_main = True
            local_rank = -1
            device = "cuda:0"

    # Move model to GPU and wrap in DDP
    if is_ddp:
        try:
            mem_before = torch.cuda.memory_allocated(local_rank) / 1e6
            print(f"[Rank {local_rank}] Moving model to {device}... (GPU mem before: {mem_before:.0f}MB)")
            sys.stdout.flush()
            model = model.to(device)
            mem_after = torch.cuda.memory_allocated(local_rank) / 1e6
            print(f"[Rank {local_rank}] Model moved to {device} (GPU mem after: {mem_after:.0f}MB)")
            sys.stdout.flush()
        except RuntimeError as e:
            print(f"[Rank {local_rank}] FATAL: model.to({device}) failed: {e}")
            print(f"[Rank {local_rank}] Hint: run 'nvidia-smi' — zombie GPU processes may hold VRAM")
            dist.destroy_process_group()
            raise

        # Verify all params are on the correct device before DDP wrapping
        n_params_on_device = 0
        n_params_off_device = 0
        for name, p in model.named_parameters():
            if p.device == torch.device(device):
                n_params_on_device += 1
            else:
                n_params_off_device += 1
                if n_params_off_device <= 5:  # print first 5 misplaced params
                    print(f"[Rank {local_rank}] WARNING: param {name} on {p.device}, expected {device}")
        print(f"[Rank {local_rank}] Params on {device}: {n_params_on_device}, off-device: {n_params_off_device}")
        sys.stdout.flush()

        if n_params_off_device > 0:
            print(f"[Rank {local_rank}] FATAL: {n_params_off_device} params not on {device}. "
                  f"This causes DDP param mismatch. Check that device_map is NOT set in from_pretrained().")
            dist.destroy_process_group()
            raise RuntimeError(f"Model params not fully on {device}")

        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            find_unused_parameters=True,  # frozen vision_tower params never used in forward
        )
        print(f"[Rank {local_rank}] DDP wrapping complete")
        sys.stdout.flush()
    else:
        # Single-GPU mode (non-DDP or Phase 0 DDP-skip)
        model = model.to(device)
        print(f"Model moved to {device}")
        sys.stdout.flush()

    # ------------------------------------------------------------------ #
    # Training (runs whichever phase the config specifies)                  #
    # ------------------------------------------------------------------ #
    phase_output = Path(args.output) / f"phase{phase}"
    config_dict["logging"]["output_dir"] = str(phase_output)

    # Initialize WandB for Phase 0 (rank-0 only)
    if is_main and not args.no_wandb:
        run_name = config_dict.get("logging", {}).get("wandb_run_name", f"phase{phase}")
        wandb.init(project=args.wandb_project, name=run_name, config=config_dict)

    data_config = config_dict.get("data", {})
    batch_size = config_dict.get("training", {}).get("batch_size", 16)
    num_workers = data_config.get("num_workers", 0)

    train_loader, val_loader = build_loaders(data_config, batch_size, num_workers, args, is_ddp=is_ddp)

    phase_metrics, phase_dir = run_phase(
        model, config_dict, train_loader, val_loader, device,
        resume=args.resume, local_rank=local_rank,
    )
    if is_main:
        print(f"Phase {phase} complete. Metrics: {phase_metrics}")

    if is_main and not args.no_wandb and wandb.run is not None:
        wandb.finish()

    # ------------------------------------------------------------------ #
    # Phase 1 training (only when --run-both-phases and config is Phase 0) #
    # ------------------------------------------------------------------ #
    if args.run_both_phases and phase == 0:
        phase1_config_path = config_path.parent / "phase1.yaml"
        if not phase1_config_path.exists():
            if is_main:
                print(f"WARNING: --run-both-phases requested but {phase1_config_path} not found. Skipping Phase 1.")
        else:
            if is_main:
                print(f"\nProceeding to Phase 1 using {phase1_config_path}")

            phase1_dict = load_config(str(phase1_config_path))

            # Apply --max-steps override to Phase 1 too (for dry runs)
            if args.max_steps is not None:
                phase1_dict["training"]["max_steps"] = args.max_steps

            phase1_output = Path(args.output) / "phase1"
            phase1_dict["logging"]["output_dir"] = str(phase1_output)

            # Phase 1 resumes from Phase 0 best checkpoint
            phase0_best = phase_dir / "best.ckpt"
            phase1_resume = str(phase0_best) if phase0_best.exists() else None
            if is_main:
                if phase1_resume:
                    print(f"Phase 1 resuming from {phase1_resume}")
                else:
                    print("WARNING: Phase 0 best.ckpt not found; Phase 1 starts from scratch")

            if is_main and not args.no_wandb:
                run_name1 = phase1_dict.get("logging", {}).get("wandb_run_name", "phase1")
                wandb.init(project=args.wandb_project, name=run_name1, config=phase1_dict)

            # Rebuild loaders with Phase 1 data config (may include GQA)
            data_config1 = phase1_dict.get("data", {})
            batch_size1 = phase1_dict.get("training", {}).get("batch_size", batch_size)
            num_workers1 = data_config1.get("num_workers", num_workers)
            train_loader1, val_loader1 = build_loaders(
                data_config1, batch_size1, num_workers1, args, is_ddp=is_ddp
            )

            phase1_metrics, _ = run_phase(
                model, phase1_dict, train_loader1, val_loader1, device,
                resume=phase1_resume, local_rank=local_rank,
            )
            if is_main:
                print(f"Phase 1 complete. Metrics: {phase1_metrics}")

            if is_main and not args.no_wandb and wandb.run is not None:
                wandb.finish()

    if is_main:
        print("\nTraining complete!")

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
