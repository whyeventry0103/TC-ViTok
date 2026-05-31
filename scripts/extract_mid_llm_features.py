#!/usr/bin/env python3
"""Extract LLaVA hidden states at visual-token positions for Track C C2."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.evaluate import load_model  # noqa: E402
from tc_vitok.data.datasets import GQADataset, collate_fn  # noqa: E402


@torch.no_grad()
def llm_hidden_visual_tokens(
    model,
    v_full: torch.Tensor,
    questions: list[str],
    layers: list[int],
    vis_start: int,
    vis_end: int,
):
    """Return hidden states for visual positions from [visual | text]."""
    device = v_full.device
    lm = model._llm
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
            return_dict=True,
        )

    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None:
        raise RuntimeError("LLM did not return hidden states")
    max_layer = len(hidden_states) - 1
    result = {}
    for layer in layers:
        if layer > max_layer:
            raise RuntimeError(f"Requested L{layer}, but only 0..{max_layer} hidden states exist")
        result[f"L{layer}"] = hidden_states[layer][:, vis_start:vis_end, :].detach().cpu().to(torch.bfloat16)
    return result, {
        "vis_start": int(vis_start),
        "vis_end": int(vis_end),
        "visual_width": int(vis_end - vis_start),
        "projected_visual_tokens": int(v_full.shape[1]),
        "sequence_length": int(combined_embeds.shape[1]),
        "text_length": int(text_mask.shape[1]),
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
    parser.add_argument("--layers", default="0,2,4,8,16,24,28")
    parser.add_argument("--layer", type=int, default=None, help="If set, extract only this LLM layer.")
    parser.add_argument("--vis-start", type=int, default=0)
    parser.add_argument("--vis-end", type=int, default=576)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata-output", default=None)
    args = parser.parse_args()

    data_root = args.data_root or os.environ.get("DATA_ROOT")
    if not data_root:
        raise SystemExit("Set --data-root or DATA_ROOT")
    data_root = str(Path(data_root).expanduser())
    layers = [args.layer] if args.layer is not None else [int(x) for x in args.layers.split(",") if x.strip()]
    if args.vis_end <= args.vis_start:
        raise SystemExit("--vis-end must be greater than --vis-start")
    print(f"Extracting layers: {layers}")
    print(f"Visual token slice: [{args.vis_start}:{args.vis_end}]")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = load_model(
        checkpoint_path=args.checkpoint,
        device=device,
        k=576,
        baseline="full_576",
        auto_adaptive_budget=False,
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

    accum = {f"L{layer}": [] for layer in layers}
    metadata = []
    layout_checks = []
    for batch in tqdm(loader, desc="mid-LLM features"):
        images_full = batch["images_full"].to(device)
        questions = batch["questions"]
        v_full = model.full_encoder(images_full)
        layer_features, layout = llm_hidden_visual_tokens(
            model,
            v_full,
            questions,
            layers,
            vis_start=args.vis_start,
            vis_end=args.vis_end,
        )
        expected_width = args.vis_end - args.vis_start
        for key, value in layer_features.items():
            if value.shape[1] != expected_width:
                raise RuntimeError(
                    f"{key} visual slice has width {value.shape[1]}, expected {expected_width}"
                )
        for key, value in layer_features.items():
            accum[key].append(value)
        if len(layout_checks) < 5:
            layout_checks.append(layout)
        for qid, image_id, question in zip(
            batch.get("question_id", [None] * len(questions)),
            batch.get("image_id", [None] * len(questions)),
            questions,
        ):
            metadata.append({"question_id": qid, "image_id": image_id, "question": question})

    save_dict = {key: torch.cat(parts, dim=0) for key, parts in accum.items()}
    save_dict["metadata"] = metadata
    save_dict["layout_checks"] = layout_checks
    save_dict["vis_start"] = args.vis_start
    save_dict["vis_end"] = args.vis_end
    save_dict["layers"] = layers
    save_dict["config"] = {
        "checkpoint": args.checkpoint,
        "n_images": args.n_images,
        "start_index": args.start_index,
        "batch_size": args.batch_size,
        "layers": layers,
        "visual_slice": [args.vis_start, args.vis_end],
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(save_dict, out)

    if args.metadata_output:
        meta = Path(args.metadata_output)
        meta.parent.mkdir(parents=True, exist_ok=True)
        meta.write_text(json.dumps({"metadata": metadata, "layout_checks": layout_checks}, indent=2))

    summary = {
        "output": str(out),
        "layers": {key: list(value.shape) for key, value in save_dict.items() if isinstance(value, torch.Tensor)},
        "layout_checks": layout_checks,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
