"""Shared fixtures: a synthetic on-disk dataset with sharp/blurry/dup images."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageFilter


def _noise_image(size: int = 768, seed: int = 0) -> Image.Image:
    """A high-frequency noise image — sharp, passes the blur filter."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    """Build a nested dataset and return its root.

    Layout (rel_path keying matters — same basename in two sub-folders):
      personA/s1/img.png   sharp, unique
      personA/s2/img.png   sharp, unique (same basename, different folder)
      personA/s1/dup.png   near-duplicate of img.png
      personB/blurry.png   blurry -> rejected
      personB/tiny.png     low-res -> rejected
    """
    root = tmp_path / "raw"

    a1 = root / "personA" / "s1"
    a2 = root / "personA" / "s2"
    b = root / "personB"
    for d in (a1, a2, b):
        d.mkdir(parents=True, exist_ok=True)

    base = _noise_image(seed=1)
    base.save(a1 / "img.png")

    # Near-duplicate: same base with a tiny tweak.
    dup = base.copy()
    dup.putpixel((0, 0), (0, 0, 0))
    dup.save(a1 / "dup.png")

    # Different sharp image, same basename in another sub-folder.
    _noise_image(seed=2).save(a2 / "img.png")

    # Blurry -> rejected by blur_threshold.
    blurry = _noise_image(seed=3).filter(ImageFilter.GaussianBlur(radius=8))
    blurry.save(b / "blurry.png")

    # Too small -> rejected by min_short_side.
    _noise_image(size=128, seed=4).save(b / "tiny.png")

    return root
