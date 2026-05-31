#!/usr/bin/env python3
"""Extract text-to-visual attention features for Track C C4."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.evaluate import load_model  # noqa: E402
from tc_vitok.data.datasets import GQADataset, collate_fn  # noqa: E402


def decoder_layers_from_llm(lm):
    """Locate LLaMA decoder layers through PEFT / HF wrapper variants."""
    candidates = [
        ("base_model.model.model.layers", lm),
        ("base_model.model.layers", lm),
        ("model.model.layers", lm),
        ("model.layers", lm),
        ("layers", lm),
    ]
    for path, root in candidates:
        obj = root
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            if len(obj) > 0:
                return obj, path
        except AttributeError:
            continue
    raise RuntimeError(f"Could not locate decoder layers under {type(lm)}")


def rotary_embedding_from_llm(lm):
    """Locate model-level rotary embedding module through PEFT / HF wrappers."""
    candidates = [
        "base_model.model.model.rotary_emb",
        "base_model.model.rotary_emb",
        "model.model.rotary_emb",
        "model.rotary_emb",
        "rotary_emb",
    ]
    for path in candidates:
        obj = lm
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj, path
        except AttributeError:
            continue
    return None, None


def manual_text_to_visual_for_layer(
    layer_module,
    hidden: torch.Tensor,
    text_mask: torch.Tensor,
    visual_tokens: int,
    vis_start: int,
    vis_end: int,
    rotary_emb=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute true text-query attention mass to visual keys from q/k projections."""
    attn_mod = layer_module.self_attn
    config = getattr(attn_mod, "config", None)
    normed = layer_module.input_layernorm(hidden)
    batch_size, seq_len, _ = normed.shape
    num_heads = int(getattr(attn_mod, "num_heads", getattr(config, "num_attention_heads", 0)))
    if num_heads <= 0:
        raise RuntimeError("Could not infer number of LLaMA attention heads")
    num_kv_heads = int(
        getattr(attn_mod, "num_key_value_heads", getattr(config, "num_key_value_heads", num_heads))
    )
    head_dim = int(
        getattr(attn_mod, "head_dim", getattr(config, "hidden_size", normed.shape[-1]) // num_heads)
    )
    kv_groups = int(
        getattr(attn_mod, "num_key_value_groups", getattr(config, "num_key_value_groups", num_heads // num_kv_heads))
    )

    q = attn_mod.q_proj(normed)
    k = attn_mod.k_proj(normed)
    q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    k = k.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)

    position_ids = torch.arange(seq_len, device=hidden.device).unsqueeze(0).expand(batch_size, -1)
    if rotary_emb is not None:
        try:
            cos, sin = rotary_emb(k, position_ids)
        except TypeError:
            cos, sin = rotary_emb(normed, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
    elif hasattr(attn_mod, "rotary_emb"):
        try:
            cos, sin = attn_mod.rotary_emb(k, position_ids)
        except TypeError:
            cos, sin = attn_mod.rotary_emb(normed, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
    else:
        print("WARN: no rotary embedding provider found; computing qk without RoPE.")
    if num_kv_heads != num_heads:
        k = repeat_kv(k, kv_groups)

    scale = float(head_dim) ** -0.5
    all_positions = torch.arange(seq_len, device=hidden.device)
    per_sample_mean = []
    per_sample_head = []
    for b in range(batch_size):
        valid_text = text_mask[b].bool()
        text_offsets = torch.arange(text_mask.shape[1], device=hidden.device)[valid_text]
        if not bool(text_offsets.numel()):
            raise RuntimeError("Batch contains an item with no valid text tokens")
        query_rows = text_offsets + visual_tokens
        q_text = q[b, :, query_rows, :]
        logits = torch.matmul(q_text, k[b].transpose(-2, -1)) * scale

        valid_keys = torch.cat([
            torch.ones(visual_tokens, device=hidden.device, dtype=torch.bool),
            valid_text,
        ])
        causal = all_positions.unsqueeze(0) <= query_rows.unsqueeze(1)
        keep = valid_keys.unsqueeze(0) & causal
        logits = logits.masked_fill(~keep.unsqueeze(0), torch.finfo(logits.dtype).min)
        attn = torch.softmax(logits.float(), dim=-1)
        text_to_visual = attn[:, :, vis_start:vis_end]
        per_sample_mean.append(text_to_visual.mean(dim=0).mean(dim=0).cpu())
        per_sample_head.append(text_to_visual.mean(dim=1).transpose(0, 1).cpu())
    return torch.stack(per_sample_mean).to(torch.bfloat16), torch.stack(per_sample_head).to(torch.bfloat16)


@torch.no_grad()
def llm_text_to_visual_attention(
    model,
    v_full: torch.Tensor,
    questions: list[str],
    layers: list[int],
    vis_start: int,
    vis_end: int,
):
    """Return text-query to visual-key attention summaries.

    The LLaVA path used in this repo concatenates [visual tokens | text tokens].
    We mean-pool only over non-padding text query positions.
    """
    device = v_full.device
    lm = model._llm
    decoder_layers, decoder_path = decoder_layers_from_llm(lm)
    rotary_emb, rotary_path = rotary_embedding_from_llm(lm)
    tokenizer = model.processor.tokenizer
    embed_fn = lm.get_input_embeddings()
    lm_dtype = next(lm.parameters()).dtype

    prompts = [f"USER: {q}\nASSISTANT:" for q in questions]
    text_inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
        add_special_tokens=True,
    )
    input_ids = text_inputs.input_ids.to(device)
    text_mask = text_inputs.attention_mask.to(device)

    use_cuda = "cuda" in str(device)
    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_cuda
        else torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=False)
    )
    with amp_ctx:
        v_projected = model._mm_proj(v_full.to(dtype=lm_dtype))
        text_embeds = embed_fn(input_ids).to(dtype=v_projected.dtype)
        combined_embeds = torch.cat([v_projected, text_embeds], dim=1)
        vis_mask = torch.ones(v_full.shape[0], v_full.shape[1], device=device, dtype=torch.long)
        combined_mask = torch.cat([vis_mask, text_mask], dim=1)
        outputs = lm(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            output_hidden_states=True,
            output_attentions=False,
            return_dict=True,
            use_cache=False,
        )

    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None:
        raise RuntimeError("LLM did not return hidden states for manual attention.")
    max_layer = min(len(decoder_layers), len(hidden_states)) - 1
    result_mean: dict[str, torch.Tensor] = {}
    result_perhead: dict[str, torch.Tensor] = {}

    for layer in layers:
        if layer > max_layer:
            raise RuntimeError(f"Requested attention L{layer}, but only 0..{max_layer} are available")
        key = f"L{layer}"
        mean, perhead = manual_text_to_visual_for_layer(
            decoder_layers[layer],
            hidden_states[layer],
            text_mask,
            visual_tokens=v_full.shape[1],
            vis_start=vis_start,
            vis_end=vis_end,
            rotary_emb=rotary_emb,
        )
        result_mean[key] = mean
        result_perhead[key] = perhead

    return result_mean, result_perhead, {
        "vis_start": int(vis_start),
        "vis_end": int(vis_end),
        "visual_width": int(vis_end - vis_start),
        "projected_visual_tokens": int(v_full.shape[1]),
        "sequence_length": int(combined_embeds.shape[1]),
        "text_length": int(text_mask.shape[1]),
        "decoder_layer_path": decoder_path,
        "rotary_embedding_path": rotary_path,
        "decoder_layer_count": len(decoder_layers),
        "hidden_state_count": len(hidden_states),
    }


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--n-images", type=int, default=1000)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--layers", nargs="+", type=int, default=[1, 2, 4, 8, 16, 28])
    parser.add_argument("--vis-start", type=int, default=0)
    parser.add_argument("--vis-end", type=int, default=576)
    parser.add_argument(
        "--llava-attn-implementation",
        default="eager",
        help="Attention backend for LLaVA. Use eager to make output_attentions available.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data_root = args.data_root or os.environ.get("DATA_ROOT")
    if not data_root:
        raise SystemExit("Set --data-root or DATA_ROOT")
    data_root = str(Path(data_root).expanduser())
    if args.vis_end <= args.vis_start:
        raise SystemExit("--vis-end must be greater than --vis-start")

    print(f"Extracting text-to-visual attention layers: {args.layers}")
    print(f"Visual token slice: [{args.vis_start}:{args.vis_end}]")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = load_model(
        checkpoint_path=args.checkpoint,
        device=device,
        k=576,
        baseline="full_576",
        auto_adaptive_budget=False,
        llava_attn_implementation=args.llava_attn_implementation,
    ).eval()

    dataset = GQADataset(
        questions_path=f"{data_root}/gqa/val_balanced_questions.json",
        image_dir=f"{data_root}/gqa/images",
        max_samples=args.start_index + args.n_images,
    )
    if args.start_index:
        dataset.samples = dataset.samples[args.start_index:args.start_index + args.n_images]
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    accum_mean = {f"L{layer}": [] for layer in args.layers}
    accum_perhead = {f"L{layer}": [] for layer in args.layers}
    metadata = []
    layout_checks = []
    for batch in tqdm(loader, desc="attention features"):
        images_full = batch["images_full"].to(device)
        questions = batch["questions"]
        v_full = model.full_encoder(images_full)
        mean_feats, perhead_feats, layout = llm_text_to_visual_attention(
            model,
            v_full,
            questions,
            args.layers,
            vis_start=args.vis_start,
            vis_end=args.vis_end,
        )
        for key, value in mean_feats.items():
            accum_mean[key].append(value)
        for key, value in perhead_feats.items():
            accum_perhead[key].append(value)
        if len(layout_checks) < 5:
            layout_checks.append(layout)
        for qid, image_id, question in zip(
            batch.get("question_id", [None] * len(questions)),
            batch.get("image_id", [None] * len(questions)),
            questions,
        ):
            metadata.append({"question_id": qid, "image_id": image_id, "question": question})

    save_dict = {
        "metadata": metadata,
        "layout_checks": layout_checks,
        "layers": args.layers,
        "vis_start": args.vis_start,
        "vis_end": args.vis_end,
        "config": {
            "checkpoint": args.checkpoint,
            "n_images": args.n_images,
            "start_index": args.start_index,
            "batch_size": args.batch_size,
            "layers": args.layers,
            "visual_slice": [args.vis_start, args.vis_end],
        },
    }
    for key in accum_mean:
        save_dict[f"{key}_mean"] = torch.cat(accum_mean[key], dim=0)
        save_dict[f"{key}_perhead"] = torch.cat(accum_perhead[key], dim=0)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(save_dict, out)

    summary = {
        "output": str(out),
        "layers": {
            key: list(value.shape)
            for key, value in save_dict.items()
            if isinstance(value, torch.Tensor)
        },
        "layout_checks": layout_checks,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
