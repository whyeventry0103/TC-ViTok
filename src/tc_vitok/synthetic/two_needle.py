"""Two-shape synthetic task with a location-addressed target."""

from __future__ import annotations

import random

import numpy as np
from PIL import Image, ImageDraw

from .needle import COLORS, SyntheticSample


def gen_sample(rng: random.Random, image_size: int = 336, patch_size: int = 14) -> SyntheticSample:
    grid = image_size // patch_size
    arr = np.random.default_rng(rng.randint(0, 2**31 - 1)).integers(
        120, 140, size=(image_size, image_size, 3), dtype=np.uint8
    )
    image = Image.fromarray(arr)
    draw = ImageDraw.Draw(image)

    positions = set()
    while len(positions) < 2:
        positions.add((rng.randint(0, grid - 1), rng.randint(0, grid - 1)))
    pos_list = list(positions)
    colors = rng.sample(list(COLORS), 2)

    for (row, col), color_name in zip(pos_list, colors):
        cx = col * patch_size + patch_size // 2
        cy = row * patch_size + patch_size // 2
        radius = patch_size // 2 - 1
        draw.rectangle([cx - radius, cy - radius, cx + radius, cy + radius], fill=COLORS[color_name])

    target = rng.randint(0, 1)
    row, col = pos_list[target]
    answer = colors[target]
    return SyntheticSample(
        image=image,
        question=f"What color is the square at row {row}, column {col}?",
        answer=answer,
        target_patch_idx=row * grid + col,
    )


def build_dataset(n: int, seed: int = 13) -> list[SyntheticSample]:
    rng = random.Random(seed)
    return [gen_sample(rng) for _ in range(n)]

