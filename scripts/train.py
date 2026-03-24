#!/usr/bin/env python3
"""CLI entry point for training TC-ViTok.

Usage:
    python scripts/train.py --config configs/phase0.yaml --output runs/phase0/
    python scripts/train.py --config configs/phase1.yaml --resume runs/phase0/best.ckpt

This script:
1. Loads configuration from YAML
2. Initializes model, datasets, and trainer
3. Runs multi-phase training loop
4. Saves checkpoints and logs to WandB

Paper reference: Section 7 (Training Procedure)
"""

import argparse
from pathlib import Path
import yaml
import torch
import wandb
from omegaconf import OmegaConf

# TODO (Phase C): Import TC-ViTok modules
# from tc_vitok.models import TCViTokModel
# from tc_vitok.data import VQAv2Dataset, GQADataset
# from tc_vitok.training import TCViTokTrainer


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
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    config = OmegaConf.load(args.config)
    print(f"Loaded config from {args.config}")
    print(f"Phase: {config.get('phase', 1)}")
    print(f"Device: {args.device}")

    # TODO (Phase C): Initialize WandB
    wandb.init(
        project=args.wandb_project,
        name=config.logging.get("wandb_run_name", f"phase{config.get('phase', 1)}"),
        config=OmegaConf.to_container(config, resolve=True),
    )

    # TODO (Phase C): Initialize model
    # model = TCViTokModel(
    #     k=config.model.k,
    #     tpn_config=config.model.tpn,
    #     lora_config=config.model.lora,
    # )

    # TODO (Phase C): Load datasets
    # train_dataset = ...
    # val_dataset = ...

    # TODO (Phase C): Create trainer
    # trainer = TCViTokTrainer(
    #     model=model,
    #     config=OmegaConf.to_container(config, resolve=True),
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     device=args.device,
    # )

    # TODO (Phase C): Resume from checkpoint if provided
    # if args.resume:
    #     trainer.load_checkpoint(args.resume)

    # TODO (Phase C): Run training
    # final_metrics = trainer.train()

    print("Training complete!")
    # wandb.finish()


if __name__ == "__main__":
    main()
