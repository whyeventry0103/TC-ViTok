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


def detect_device():
    """Auto-detect the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_loaders(data_config, batch_size, num_workers, args):
    """Build train and val DataLoaders from config and args.

    Returns:
        train_loader, val_loader (val_loader may be None)
    """
    if args.dummy_data:
        print("Using DummyVQADataset for pipeline testing")
        train_dataset = DummyVQADataset(num_samples=256)
        val_dataset = DummyVQADataset(num_samples=64)
        train_collate = collate_fn
        val_collate = collate_fn
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
                questions_path=data_config["train_questions"],
                image_dir=data_config["train_image_dir"],
                max_samples=data_config.get("max_samples"),
            )
            val_dataset = GQADataset(
                questions_path=data_config["val_questions"],
                image_dir=data_config["val_image_dir"],
                answer_list=list(train_dataset.answer_to_idx.keys()),
                max_samples=data_config.get("max_samples"),
            )
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        train_collate = collate_fn
        val_collate = collate_fn

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=train_collate,
        pin_memory=data_config.get("pin_memory", False),
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=val_collate,
            pin_memory=data_config.get("pin_memory", False),
        )

    return train_loader, val_loader


def run_phase(model, config_dict, train_loader, val_loader, device, resume=None):
    """Create trainer, optionally resume, and run one training phase.

    Args:
        model: TCViTokModel instance (will have set_training_phase called by trainer)
        config_dict: Config dict for this phase
        train_loader: Training DataLoader
        val_loader: Validation DataLoader (may be None)
        device: Training device
        resume: Optional checkpoint path to resume from

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
    )

    if resume:
        trainer.load_checkpoint(resume)

    final_metrics = trainer.train()
    return final_metrics, trainer.output_dir


def main():
    args = parse_args()

    # MPS/Apple Silicon compatibility
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    # Load config
    config = OmegaConf.load(args.config)
    config_dict = OmegaConf.to_container(config, resolve=True)
    config_path = Path(args.config)
    phase = config_dict.get("phase", 1)

    # Apply --max-steps override (useful for dry runs: --max-steps 1)
    if args.max_steps is not None:
        config_dict["training"]["max_steps"] = args.max_steps

    device = args.device or detect_device()

    print(f"Loaded config from {args.config}")
    print(f"Phase: {phase}")
    print(f"Device: {device}")
    if args.max_steps is not None:
        print(f"max_steps overridden to {args.max_steps}")

    # Initialize model (shared across phases when running both)
    tpn_config = config_dict.get("model", {}).get("tpn", {})
    lora_config = config_dict.get("model", {}).get("lora", {})
    k = config_dict.get("model", {}).get("k", 144)

    model = TCViTokModel(
        k=k,
        tpn_config=tpn_config,
        use_stub_llm=args.stub,
        lora_config=lora_config,
    )
    print(f"Model created (stub={args.stub}, k={k})")
    print(f"Trainable params: {model.get_num_params():,}")

    # ------------------------------------------------------------------ #
    # Phase 0 training                                                     #
    # ------------------------------------------------------------------ #
    phase0_output = Path(args.output) / "phase0"
    config_dict["logging"]["output_dir"] = str(phase0_output)

    # Initialize WandB for Phase 0
    if not args.no_wandb:
        run_name = config_dict.get("logging", {}).get("wandb_run_name", f"phase{phase}")
        wandb.init(project=args.wandb_project, name=run_name, config=config_dict)

    data_config = config_dict.get("data", {})
    batch_size = config_dict.get("training", {}).get("batch_size", 16)
    num_workers = data_config.get("num_workers", 0)

    train_loader, val_loader = build_loaders(data_config, batch_size, num_workers, args)

    phase0_metrics, phase0_dir = run_phase(
        model, config_dict, train_loader, val_loader, device, resume=args.resume
    )
    print(f"Phase {phase} complete. Metrics: {phase0_metrics}")

    if not args.no_wandb and wandb.run is not None:
        wandb.finish()

    # ------------------------------------------------------------------ #
    # Phase 1 training (only when --run-both-phases and config is Phase 0) #
    # ------------------------------------------------------------------ #
    if args.run_both_phases and phase == 0:
        phase1_config_path = config_path.parent / "phase1.yaml"
        if not phase1_config_path.exists():
            print(f"WARNING: --run-both-phases requested but {phase1_config_path} not found. Skipping Phase 1.")
        else:
            print(f"\nProceeding to Phase 1 using {phase1_config_path}")

            phase1_config = OmegaConf.load(phase1_config_path)
            phase1_dict = OmegaConf.to_container(phase1_config, resolve=True)

            # Apply --max-steps override to Phase 1 too (for dry runs)
            if args.max_steps is not None:
                phase1_dict["training"]["max_steps"] = args.max_steps

            phase1_output = Path(args.output) / "phase1"
            phase1_dict["logging"]["output_dir"] = str(phase1_output)

            # Phase 1 resumes from Phase 0 best checkpoint
            phase0_best = phase0_dir / "best.ckpt"
            phase1_resume = str(phase0_best) if phase0_best.exists() else None
            if phase1_resume:
                print(f"Phase 1 resuming from {phase1_resume}")
            else:
                print("WARNING: Phase 0 best.ckpt not found; Phase 1 starts from scratch")

            if not args.no_wandb:
                run_name1 = phase1_dict.get("logging", {}).get("wandb_run_name", "phase1")
                wandb.init(project=args.wandb_project, name=run_name1, config=phase1_dict)

            # Rebuild loaders with Phase 1 data config (may include GQA)
            data_config1 = phase1_dict.get("data", {})
            batch_size1 = phase1_dict.get("training", {}).get("batch_size", batch_size)
            num_workers1 = data_config1.get("num_workers", num_workers)
            train_loader1, val_loader1 = build_loaders(
                data_config1, batch_size1, num_workers1, args
            )

            phase1_metrics, _ = run_phase(
                model, phase1_dict, train_loader1, val_loader1, device,
                resume=phase1_resume,
            )
            print(f"Phase 1 complete. Metrics: {phase1_metrics}")

            if not args.no_wandb and wandb.run is not None:
                wandb.finish()

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
