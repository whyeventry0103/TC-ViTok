"""Needle-in-image synthetic task."""

from __future__ import annotations

import random
from typing import NamedTuple

import numpy as np
from PIL import Image, ImageDraw


COLORS = {
    "red": (220, 30, 30),
    "green": (40, 180, 40),
    "blue": (40, 60, 220),
    "yellow": (220, 200, 30),
    "purple": (160, 30, 200),
    "orange": (240, 140, 30),
}


class SyntheticSample(NamedTuple):
    image: Image.Image
    question: str
    answer: str
    target_patch_idx: int


def gen_sample(rng: random.Random, image_size: int = 336, patch_size: int = 14) -> SyntheticSample:
    grid = image_size // patch_size
    arr = np.random.default_rng(rng.randint(0, 2**31 - 1)).integers(
        120, 140, size=(image_size, image_size, 3), dtype=np.uint8
    )
    image = Image.fromarray(arr)
    draw = ImageDraw.Draw(image)

    row = rng.randint(0, grid - 1)
    col = rng.randint(0, grid - 1)
    color_name = rng.choice(list(COLORS))
    cx = col * patch_size + patch_size // 2
    cy = row * patch_size + patch_size // 2
    radius = patch_size // 2 - 1
    draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=COLORS[color_name])

    if rng.random() < 0.5:
        question = f"What color is the shape at row {row}, column {col}?"
    else:
        question = "What color is the shape in the image?"
    return SyntheticSample(
        image=image,
        question=question,
        answer=color_name,
        target_patch_idx=row * grid + col,
    )


def build_dataset(n: int, seed: int = 7) -> list[SyntheticSample]:
    rng = random.Random(seed)
    return [gen_sample(rng) for _ in range(n)]

