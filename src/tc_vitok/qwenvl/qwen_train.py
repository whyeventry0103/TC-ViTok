"""Train Qwen2-VL + trainable pooler bridge for FM1 reproduction."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm import tqdm

from tc_vitok.evaluation.metrics import gqa_accuracy
from .qwen_backbone import QwenVLBackbone
from .qwen_pooler import QwenTrainablePooler


class RawVQADataset(Dataset):
    def __init__(self, samples: list[dict]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        return {
            "image": image,
            "question": item["question"],
            "answer": item["answer"],
        }


def load_vqav2(data_root: str, split: str = "train", max_samples: int | None = None) -> RawVQADataset:
    ann_path = Path(data_root) / "vqav2" / f"v2_mscoco_{split}2014_annotations.json"
    q_path = Path(data_root) / "vqav2" / f"v2_OpenEnded_mscoco_{split}2014_questions.json"
    img_dir = Path(data_root) / "vqav2" / f"{split}2014"
    questions = {q["question_id"]: q for q in json.load(open(q_path))["questions"]}
    samples = []
    for ann in json.load(open(ann_path))["annotations"]:
        q = questions.get(ann["question_id"])
        if q is None:
            continue
        answers = [a["answer"] for a in ann["answers"]]
        answer = max(set(answers), key=answers.count)
        samples.append({
            "image_path": img_dir / f"COCO_{split}2014_{ann['image_id']:012d}.jpg",
            "question": q["question"],
            "answer": answer,
        })
        if max_samples and len(samples) >= max_samples:
            break
    return RawVQADataset(samples)


def load_gqa(data_root: str, split: str = "train", max_samples: int | None = None) -> RawVQADataset:
    q_path = Path(data_root) / "gqa" / f"{split}_balanced_questions.json"
    img_dir = Path(data_root) / "gqa" / "images"
    samples = []
    for _, q in json.load(open(q_path)).items():
        samples.append({
            "image_path": img_dir / f"{q['imageId']}.jpg",
            "question": q["question"],
            "answer": q["answer"],
        })
        if max_samples and len(samples) >= max_samples:
            break
    return RawVQADataset(samples)


def collate_raw(batch):
    return {
        "images": [b["image"] for b in batch],
        "questions": [b["question"] for b in batch],
        "answers": [b["answer"] for b in batch],
    }


class QwenFM1Model(torch.nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self.backbone = QwenVLBackbone(
            model_name=cfg.get("model_name", "Qwen/Qwen2-VL-7B-Instruct"),
            lora_rank=int(cfg.get("lora_rank", 32)),
            lora_alpha=int(cfg.get("lora_alpha", 64)),
            device=device,
        )
        # Probe dimensions with a tiny synthetic tensor after first image batch.
        self.pooler = None
        self.k = int(cfg.get("k", 144))
        self.d_model = int(cfg.get("d_model", 384))
        self.device_name = device

    def ensure_pooler(self, visual_tokens, q_embed):
        if self.pooler is not None:
            return
        self.pooler = QwenTrainablePooler(
            visual_dim=visual_tokens.shape[-1],
            question_dim=q_embed.shape[-1],
            d_model=self.d_model,
            k=self.k,
        ).to(visual_tokens.device)

    def forward_loss(self, images, questions, answers, tau):
        device = next(self.backbone.parameters()).device
        visual = self.backbone.encode_image(images, device)
        q_embed = self.backbone.question_embedding(questions, device)
        self.ensure_pooler(visual, q_embed)
        pooled = self.pooler(visual, q_embed, tau=tau)
        logits, labels = self.backbone.forward_text_prefix(
            pooled["llm_visual_tokens"], questions, answers=answers
        )
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        return loss

    @torch.no_grad()
    def generate_answers(self, images, questions):
        self.eval()
        device = next(self.backbone.parameters()).device
        visual = self.backbone.encode_image(images, device)
        q_embed = self.backbone.question_embedding(questions, device)
        self.ensure_pooler(visual, q_embed)
        pooled = self.pooler(visual, q_embed, tau=0.1)
        ids = self.backbone.forward_text_prefix(pooled["llm_visual_tokens"], questions)
        decoded = self.backbone.tokenizer.batch_decode(ids, skip_special_tokens=True)
        answers = []
        for text in decoded:
            if "Answer:" in text:
                text = text.split("Answer:")[-1]
            answers.append(text.strip().split("\n")[0])
        return answers


def evaluate_gqa(model, loader, max_batches=None):
    correct, total = 0.0, 0
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        preds = model.generate_answers(batch["images"], batch["questions"])
        for pred, answer in zip(preds, batch["answers"]):
            correct += gqa_accuracy(pred, answer)
            total += 1
    return correct / max(total, 1)


def tau_at(step, steps, tau_max=1.0, tau_min=0.3):
    progress = min(max(step / max(steps, 1), 0.0), 1.0)
    return tau_max + progress * (tau_min - tau_max)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-steps", type=int, default=None, help="Override config steps for smoke tests.")
    p.add_argument("--eval-samples", type=int, default=None, help="Override GQA validation sample count.")
    p.add_argument("--log-every", type=int, default=None, help="Override validation/log interval.")
    p.add_argument("--save-every", type=int, default=None, help="Override checkpoint interval.")
    p.add_argument(
        "--train-samples-per-dataset",
        type=int,
        default=None,
        help="Override train sample cap per dataset for smoke tests.",
    )
    args = p.parse_args()

    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    if args.max_steps is not None:
        cfg["steps"] = args.max_steps
    if args.eval_samples is not None:
        cfg["eval_samples"] = args.eval_samples
    if args.log_every is not None:
        cfg["log_every"] = args.log_every
    if args.save_every is not None:
        cfg["save_every"] = args.save_every
    if args.train_samples_per_dataset is not None:
        cfg["train_samples_per_dataset"] = args.train_samples_per_dataset
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    data_root = os.environ.get("DATA_ROOT", cfg.get("data_root", "data"))
    seed = int(cfg.get("seed", 42))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train = ConcatDataset([
        load_vqav2(data_root, "train", cfg.get("train_samples_per_dataset")),
        load_gqa(data_root, "train", cfg.get("train_samples_per_dataset")),
    ])
    train_loader = DataLoader(
        train,
        batch_size=int(cfg.get("batch_size", 1)),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 2)),
        collate_fn=collate_raw,
    )
    val_loader = DataLoader(
        load_gqa(data_root, "val", cfg.get("eval_samples")),
        batch_size=int(cfg.get("eval_batch_size", 1)),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 2)),
        collate_fn=collate_raw,
    )

    model = QwenFM1Model(cfg, device).to(device)
    # Build pooler before optimizer from one batch.
    first = next(iter(train_loader))
    with torch.no_grad():
        visual = model.backbone.encode_image(first["images"], torch.device(device))
        q_embed = model.backbone.question_embedding(first["questions"], torch.device(device))
        model.ensure_pooler(visual, q_embed)

    params = [
        {"params": model.pooler.parameters(), "lr": float(cfg["eta_selector"]), "weight_decay": 0.01},
        {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": float(cfg["eta_lora"]), "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(params)
    steps = int(cfg.get("steps", 12000))
    grad_accum = int(cfg.get("grad_accum", 128))
    log_every = int(cfg.get("log_every", 500))
    save_every = int(cfg.get("save_every", 2000))
    csv_path = out / "log.csv"
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["step", "val_loss", "val_acc"])

    step = 0
    pbar = tqdm(total=steps, desc="Qwen FM1")
    while step < steps:
        for batch in train_loader:
            if step >= steps:
                break
            model.train()
            tau = tau_at(step, steps, cfg.get("tau_max", 1.0), cfg.get("tau_min", 0.3))
            loss = model.forward_loss(batch["images"], batch["questions"], batch["answers"], tau)
            (loss / grad_accum).backward()
            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
            step += 1
            pbar.update(1)
            if step % log_every == 0:
                val_acc = evaluate_gqa(model, val_loader, max_batches=cfg.get("eval_max_batches"))
                with open(csv_path, "a", newline="") as f:
                    csv.writer(f).writerow([step, float(loss.detach().cpu()), val_acc])
                print(f"[step {step}] train_loss={loss.item():.4f} val_acc={val_acc:.4f}", flush=True)
            if step % save_every == 0:
                torch.save({
                    "step": step,
                    "pooler": model.pooler.state_dict(),
                    "config": cfg,
                }, out / f"checkpoint_step_{step}.pt")
    pbar.close()
    torch.save({"step": step, "pooler": model.pooler.state_dict(), "config": cfg}, out / "best.ckpt")
    json.dump({"final_step": step, "eta_selector": cfg["eta_selector"]}, open(out / "summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
