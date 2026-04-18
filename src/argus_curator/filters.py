"""Phase-1 hard filters — CPU-only, safe to parallelise."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageFilter


def sharpness(img: Image.Image) -> float:
    """Variance of the Laplacian edge response. High = sharp, low = blurry."""
    edges = img.convert("L").filter(ImageFilter.FIND_EDGES)
    return float(np.array(edges, dtype=np.float32).var())


def artifact_score(img: Image.Image) -> float:
    """JPEG block-artefact estimate. Returns [0, 1] where 1.0 = clean.

    Compares mean absolute difference at every 8-pixel DCT boundary to the
    overall inter-pixel variation.  A high boundary/internal ratio indicates
    visible blocking from heavy JPEG compression.
    """
    gray = np.array(img.convert("L"), dtype=np.float32)
    h, w = gray.shape

    h_b = float(np.abs(gray[8::8, :] - gray[7:-1:8, :]).mean()) if h > 16 else 0.0
    v_b = float(np.abs(gray[:, 8::8] - gray[:, 7:-1:8]).mean()) if w > 16 else 0.0
    boundary = (h_b + v_b) / 2.0

    internal_h = float(np.abs(np.diff(gray, axis=0)).mean())
    internal_v = float(np.abs(np.diff(gray, axis=1)).mean())
    internal = (internal_h + internal_v) / 2.0

    if internal < 1e-6:
        return 1.0
    ratio = boundary / (internal + 1e-6)
    return float(max(0.0, min(1.0, 1.0 - (ratio - 1.0) / 0.6)))


def check_resolution(img: Image.Image, min_short_side: int) -> str | None:
    """Return a reject reason string, or None if the image passes."""
    w, h = img.size
    short = min(w, h)
    if short < min_short_side:
        return f"resolution too low ({short}px short side, min {min_short_side})"
    return None


def check_aspect(img: Image.Image, max_ratio: float) -> str | None:
    w, h = img.size
    short, long = min(w, h), max(w, h)
    ratio = long / short if short else 0.0
    if ratio > max_ratio:
        return f"aspect ratio {ratio:.2f} > {max_ratio}"
    return None


def check_blur(img: Image.Image, threshold: float) -> tuple[float, str | None]:
    """Return (sharpness_value, reject_reason_or_None)."""
    s = sharpness(img)
    if s < threshold:
        return s, f"blurry (sharpness={s:.1f} < {threshold})"
    return s, None


def open_image(data: bytes) -> tuple[Image.Image, str | None]:
    """Open an image from bytes. Returns (img, error_str_or_None)."""
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        return img, None
    except Exception as exc:
        return Image.new("RGB", (1, 1)), str(exc)
