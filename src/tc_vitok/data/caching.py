"""HDF5 feature caching for TC-ViTok.

Pre-computes and caches DINOv2-S coarse features and CLIP text embeddings
to HDF5, avoiding repeated frozen encoder forward passes during training.

Full visual features (CLIP ViT-L/14, 576x1024) are NOT cached due to size
(~2.4MB per sample). The ViT-L forward pass runs live during training.

Usage:
    build_cache(dataset, cache_path, coarse_encoder, text_encoder)
    cached_dataset = CachedFeatureDataset(cache_path, image_dir, ...)

Paper reference: planv2.tex Stage B (Data Pipeline)
"""

from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from .datasets import CLIP_TRANSFORM


def build_cache(
    dataset: Dataset,
    cache_path: str,
    coarse_encoder: torch.nn.Module,
    text_encoder: torch.nn.Module,
    batch_size: int = 32,
    device: str = "cuda",
    num_workers: int = 0,
) -> None:
    """Pre-compute and save frozen encoder features to HDF5.

    Caches:
        - v_coarse: DINOv2-S spatial features, shape (N, 49, 384), float16
        - q_embed: CLIP text [EOS] embeddings, shape (N, 512), float16

    Args:
        dataset: Source dataset (VQAv2Dataset, GQADataset, etc.)
        cache_path: Output HDF5 file path
        coarse_encoder: Frozen DINOv2-S encoder (on device)
        text_encoder: Frozen CLIP text encoder (on device)
        batch_size: Batch size for feature extraction
        device: Device for encoding
        num_workers: DataLoader workers (0 for HDF5 compatibility)
    """
    import h5py
    import numpy as np

    n = len(dataset)
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Simple collate for caching (only needs coarse images and questions)
    def cache_collate(batch):
        return {
            "images_coarse": torch.stack([b["images_coarse"] for b in batch]),
            "questions": [b["questions"] for b in batch],
            "image_id": [b.get("image_id", i) for i, b in enumerate(batch)],
        }

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=cache_collate,
    )

    coarse_encoder.eval()
    text_encoder.eval()

    with h5py.File(str(cache_path), "w") as f:
        v_coarse_ds = f.create_dataset(
            "v_coarse", shape=(n, 49, 384), dtype="float16",
        )
        q_embed_ds = f.create_dataset(
            "q_embed", shape=(n, 512), dtype="float16",
        )

        offset = 0
        for batch in tqdm(loader, desc="Caching features"):
            images_coarse = batch["images_coarse"].to(device)
            questions = batch["questions"]
            bs = images_coarse.size(0)

            with torch.no_grad():
                v_c = coarse_encoder(images_coarse)  # (B, 49, 384)
                q_e = text_encoder(questions)          # (B, 512)

            v_coarse_ds[offset:offset + bs] = v_c.cpu().half().numpy()
            q_embed_ds[offset:offset + bs] = q_e.cpu().half().numpy()
            offset += bs

        # Store metadata
        f.attrs["num_samples"] = n
        f.attrs["v_coarse_shape"] = [49, 384]
        f.attrs["q_embed_shape"] = [512]

    print(f"Cache saved: {cache_path} ({n} samples, {cache_path.stat().st_size / 1e6:.1f} MB)")


class CachedFeatureDataset(Dataset):
    """Dataset that reads pre-computed features from HDF5.

    Bypasses the frozen DINOv2-S and CLIP text encoders by reading
    cached v_coarse and q_embed directly. Still loads full-resolution
    images for CLIP ViT-L/14 (not cached due to size).

    The forward path in TCViTokModel must accept v_coarse and q_embed
    directly when using this dataset (via forward_cached or optional kwargs).

    Args:
        cache_path: Path to HDF5 cache file
        image_dataset: Base dataset for loading full-res images and metadata
        use_images: If True, load full-res images (for ViT-L). If False,
                    return random tensors (for testing without images)
    """

    def __init__(
        self,
        cache_path: str,
        image_dataset: Optional[Dataset] = None,
        use_images: bool = True,
    ):
        import h5py
        import warnings

        self.cache_path = str(cache_path)
        self.image_dataset = image_dataset
        self.use_images = use_images

        if use_images and image_dataset is None:
            warnings.warn(
                "CachedFeatureDataset: use_images=True but image_dataset=None. "
                "images_full will be random noise and labels will be zeros. "
                "Pass image_dataset= for real training.",
                UserWarning,
                stacklevel=2,
            )

        # Read metadata (open briefly, then close)
        with h5py.File(self.cache_path, "r") as f:
            self.num_samples = f.attrs["num_samples"]

        # HDF5 file handle (opened lazily per worker for multiprocessing)
        self._h5_file = None

    def _get_h5(self):
        """Lazily open HDF5 file (per-worker for DataLoader compatibility)."""
        import h5py
        if self._h5_file is None:
            self._h5_file = h5py.File(self.cache_path, "r", swmr=True)
        return self._h5_file

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        h5 = self._get_h5()

        v_coarse = torch.from_numpy(h5["v_coarse"][idx]).float()  # (49, 384)
        q_embed = torch.from_numpy(h5["q_embed"][idx]).float()    # (512,)

        result = {
            "v_coarse": v_coarse,
            "q_embed": q_embed,
        }

        # Load full-res image and metadata from base dataset
        if self.image_dataset is not None and self.use_images:
            base_sample = self.image_dataset[idx]
            result["images_full"] = base_sample["images_full"]
            result["answer_str"] = base_sample.get("answer_str", "")
            result["questions"] = base_sample.get("questions", "")
            # Pass through metadata
            for key in ["answers", "question_id", "image_id", "gqa_annotation"]:
                if key in base_sample:
                    result[key] = base_sample[key]
        else:
            # Testing mode: random full-res image
            result["images_full"] = torch.randn(3, 336, 336)
            result["answer_str"] = "dummy"
            result["questions"] = ""

        return result

    def __del__(self):
        if self._h5_file is not None:
            self._h5_file.close()


def cached_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for CachedFeatureDataset batches.

    Stacks cached feature tensors and passes through metadata.
    answer_str is passed as a list of strings for generative LM training.
    """
    result = {
        "v_coarse": torch.stack([b["v_coarse"] for b in batch]),
        "q_embed": torch.stack([b["q_embed"] for b in batch]),
        "images_full": torch.stack([b["images_full"] for b in batch]),
    }

    # Stack v_full if provided (stub mode pre-computes it to skip full_encoder)
    if "v_full" in batch[0]:
        result["v_full"] = torch.stack([b["v_full"] for b in batch])

    # answer_str: raw answer strings for next-token prediction training
    if "answer_str" in batch[0]:
        result["answer_str"] = [b["answer_str"] for b in batch]

    # Pass through optional metadata
    for key in ["questions", "answers", "question_id", "image_id",
                "gqa_annotation"]:
        if key in batch[0]:
            result[key] = [b[key] for b in batch]

    return result
