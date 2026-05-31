"""Dense text-position synthetic task."""

from __future__ import annotations

import random

import numpy as np
from PIL import Image, ImageDraw

from .needle import SyntheticSample


WORDS = ["cat", "bus", "red", "two", "left", "yes", "box", "cup", "7", "sun"]


def gen_sample(rng: random.Random, image_size: int = 336, patch_size: int = 14) -> SyntheticSample:
    grid = image_size // patch_size
    arr = np.full((image_size, image_size, 3), 235, dtype=np.uint8)
    image = Image.fromarray(arr)
    draw = ImageDraw.Draw(image)

    n_words = rng.randint(5, 10)
    positions = set()
    while len(positions) < n_words:
        positions.add((rng.randint(1, grid - 2), rng.randint(1, grid - 3)))

    placed = []
    for row, col in positions:
        word = rng.choice(WORDS)
        x = col * patch_size + 1
        y = row * patch_size
        draw.text((x, y), word, fill=(20, 20, 20))
        placed.append((row, col, word))

    row, col, answer = rng.choice(placed)
    return SyntheticSample(
        image=image,
        question=f"What is written at row {row}, column {col}?",
        answer=answer,
        target_patch_idx=row * grid + col,
    )


def build_dataset(n: int, seed: int = 29) -> list[SyntheticSample]:
    rng = random.Random(seed)
    return [gen_sample(rng) for _ in range(n)]

