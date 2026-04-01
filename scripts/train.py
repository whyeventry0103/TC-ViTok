#!/usr/bin/env python3
"""CLI entry point for training TC-ViTok.

Usage:
    python scripts/train.py --config configs/base.yaml --output runs/phase1/
    python scripts/train.py --config configs/base.yaml --resume runs/phase0/best.ckpt
    python scripts/train.py --config configs/base.yaml --stub  # Use StubLLM for testing

This script:
1. Loads configuration from YAML
2. Initializes model, datasets, and trainer
3. Runs multi-phase training loop
4. Saves checkpoints and logs to WandB

Paper reference: Section 7 (Training Procedure)
"""

import argparse
import os
from pathlib import Path
import yaml
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
        help="Path to config YAML file (e.g., configs/base.yaml)",
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


def main():
    args = parse_args()

    # MPS/Apple Silicon compatibility
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    # Load config
    config = OmegaConf.load(args.config)
    config_dict = OmegaConf.to_container(config, resolve=True)

    # Override output dir
    config_dict["logging"]["output_dir"] = args.output

    device = args.device or detect_device()
    phase = config_dict.get("phase", 1)

    print(f"Loaded config from {args.config}")
    print(f"Phase: {phase}")
    print(f"Device: {device}")

    # Initialize WandB
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=config_dict.get("logging", {}).get(
                "wandb_run_name", f"phase{phase}"
            ),
            config=config_dict,
        )

    # Initialize model
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

    # Load datasets
    data_config = config_dict.get("data", {})
    batch_size = config_dict.get("training", {}).get("batch_size", 16)
    num_workers = data_config.get("num_workers", 0)

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

    # Create data loaders
    from tc_vitok.data.caching import cached_collate_fn

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

    # Create trainer
    trainer = TCViTokTrainer(
        model=model,
        config=config_dict,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )

    # Resume from checkpoint if provided
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Run training
    final_metrics = trainer.train()

    print("Training complete!")
    print(f"Final metrics: {final_metrics}")

    if not args.no_wandb and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
