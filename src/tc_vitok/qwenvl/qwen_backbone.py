"""Qwen2-VL backbone wrapper for FM1 reproduction."""

from __future__ import annotations

import os
from contextlib import nullcontext
from typing import Sequence

import torch
from torch import nn

from .qwen_lora import apply_qwen_lora, mark_only_lora_trainable


class QwenVLBackbone(nn.Module):
    """Minimal wrapper exposing visual embeddings and text-prefix LM calls."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        lora_rank: int = 32,
        lora_alpha: int = 64,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(model_name)
        base = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if "cuda" in str(device) else torch.float32,
            low_cpu_mem_usage=True,
        )
        for param in base.parameters():
            param.requires_grad = False
        self.model = apply_qwen_lora(base, rank=lora_rank, alpha=lora_alpha)
        mark_only_lora_trainable(self.model)

    @property
    def tokenizer(self):
        return self.processor.tokenizer

    def get_language_model(self):
        candidates = [
            getattr(self.model, "language_model", None),
            getattr(getattr(self.model, "model", None), "language_model", None),
            getattr(getattr(getattr(self.model, "model", None), "model", None), "language_model", None),
            getattr(self.model, "model", None),
        ]
        for candidate in candidates:
            if candidate is not None and hasattr(candidate, "get_input_embeddings"):
                return candidate
        return self.model

    def get_lm_head(self):
        for obj in (self.model, getattr(self.model, "model", None), getattr(getattr(self.model, "model", None), "model", None)):
            if obj is not None and hasattr(obj, "lm_head"):
                return obj.lm_head
        raise AttributeError("Could not locate Qwen2-VL lm_head")

    def get_visual_module(self):
        for obj in (self.model, getattr(self.model, "model", None), getattr(getattr(self.model, "model", None), "model", None)):
            visual = getattr(obj, "visual", None)
            if visual is not None:
                return visual
        raise AttributeError("Could not locate Qwen2-VL visual module")

    def prepare_images(self, images: Sequence) -> dict[str, torch.Tensor]:
        """Use Qwen processor to produce pixel values and image grid metadata."""
        text = ["<|vision_start|><|image_pad|><|vision_end|>"] * len(images)
        batch = self.processor(text=text, images=list(images), return_tensors="pt", padding=True)
        return {k: v for k, v in batch.items() if k in ("pixel_values", "image_grid_thw")}

    def encode_image(self, images: Sequence, device: torch.device) -> torch.Tensor:
        """Encode images to Qwen visual tokens in language hidden dimension."""
        batch = self.prepare_images(images)
        visual = self.get_visual_module()
        visual_dtype = next(visual.parameters()).dtype
        pixel_values = batch["pixel_values"].to(device=device, dtype=visual_dtype)
        grid_thw = batch.get("image_grid_thw")
        if grid_thw is not None:
            grid_thw = grid_thw.to(device)
        with torch.no_grad():
            try:
                embeds = visual(pixel_values, grid_thw=grid_thw)
            except TypeError:
                embeds = visual(pixel_values, image_grid_thw=grid_thw)
        if embeds.dim() == 2:
            # Qwen returns concatenated image tokens; split by grid size.
            merge = int(getattr(visual, "spatial_merge_size", 2))
            counts = (
                grid_thw[:, 0]
                * torch.div(grid_thw[:, 1], merge, rounding_mode="floor")
                * torch.div(grid_thw[:, 2], merge, rounding_mode="floor")
            ).tolist()
            if sum(int(c) for c in counts) != embeds.shape[0]:
                raise RuntimeError(
                    "Qwen visual token count mismatch: "
                    f"grid-derived={sum(int(c) for c in counts)} encoded={embeds.shape[0]}"
                )
            chunks = torch.split(embeds, [int(c) for c in counts], dim=0)
            max_len = max(c.shape[0] for c in chunks)
            padded = embeds.new_zeros(len(chunks), max_len, embeds.shape[-1])
            for i, chunk in enumerate(chunks):
                padded[i, : chunk.shape[0]] = chunk
            embeds = padded
        return embeds

    def question_embedding(self, questions: Sequence[str], device: torch.device) -> torch.Tensor:
        tokenizer = self.tokenizer
        lm = self.get_language_model()
        text = [f"Question: {q}\nAnswer:" for q in questions]
        toks = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        input_ids = toks.input_ids.to(device)
        attn = toks.attention_mask.to(device).unsqueeze(-1)
        with torch.no_grad():
            emb = lm.get_input_embeddings()(input_ids)
        return (emb * attn).sum(dim=1) / attn.sum(dim=1).clamp_min(1)

    def forward_text_prefix(
        self,
        visual_prefix: torch.Tensor,
        questions: Sequence[str],
        answers: Sequence[str] | None = None,
        max_new_tokens: int = 20,
    ):
        """Run Qwen language model with a custom visual prefix."""
        device = visual_prefix.device
        tokenizer = self.tokenizer
        lm = self.get_language_model()
        lm_head = self.get_lm_head()
        embed = lm.get_input_embeddings()
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda" else nullcontext()
        )
        prompts = [f"Question: {q}\nAnswer:" for q in questions]
        prompt = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=128)
        prompt_ids = prompt.input_ids.to(device)
        prompt_attn = prompt.attention_mask.to(device)
        with autocast:
            prompt_emb = embed(prompt_ids)
            if answers is None:
                inputs_embeds = torch.cat([visual_prefix, prompt_emb], dim=1)
                vis_attn = torch.ones(visual_prefix.shape[:2], device=device, dtype=torch.long)
                attention_mask = torch.cat([vis_attn, prompt_attn], dim=1)
                gen_obj = lm if hasattr(lm, "generate") else self.model
                return gen_obj.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            all_embeds, all_labels, all_masks = [], [], []
            for i, answer in enumerate(answers):
                ans_ids = tokenizer(
                    f" {answer}",
                    return_tensors="pt",
                    add_special_tokens=False,
                    truncation=True,
                    max_length=32,
                ).input_ids[0].to(device)
                eos = torch.tensor([tokenizer.eos_token_id], device=device, dtype=torch.long)
                ans_ids = torch.cat([ans_ids, eos])
                ans_emb = embed(ans_ids)
                seq = torch.cat([visual_prefix[i], prompt_emb[i], ans_emb], dim=0)
                labels = torch.cat([
                    torch.full((visual_prefix.shape[1] + prompt_ids.shape[1] - 1,), -100, device=device, dtype=torch.long),
                    ans_ids,
                    torch.tensor([-100], device=device, dtype=torch.long),
                ])
                mask = torch.ones(seq.shape[0], device=device, dtype=torch.long)
                all_embeds.append(seq)
                all_labels.append(labels)
                all_masks.append(mask)
            max_len = max(x.shape[0] for x in all_embeds)
            B, H = len(all_embeds), all_embeds[0].shape[-1]
            padded = visual_prefix.new_zeros(B, max_len, H)
            labels = torch.full((B, max_len), -100, device=device, dtype=torch.long)
            masks = torch.zeros(B, max_len, device=device, dtype=torch.long)
            for i, seq in enumerate(all_embeds):
                L = seq.shape[0]
                padded[i, :L] = seq
                labels[i, :L] = all_labels[i][:L]
                masks[i, :L] = all_masks[i]
            outputs = lm(inputs_embeds=padded, attention_mask=masks, output_hidden_states=True)
            hidden = getattr(outputs, "last_hidden_state", None)
            logits = outputs.logits if hasattr(outputs, "logits") else lm_head(hidden)
            return logits.float(), labels
