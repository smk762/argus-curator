"""Crops (watermarks, borders) applied to image bytes in memory, never to the source.

A box comes from :class:`CropConfig.boxes` or a ``<image>.crop.json`` sidecar
beside the source. The scan crops each boxed image once, right after reading
it, so every check judges the cropped frame; export writes the cropped file
from the source again. The source and its sidecar are only ever read.

Boxes are in pixels of the image as displayed (after its EXIF orientation),
which is what an editor or a browser shows when a crop is drawn. The cropped
image is upright with no orientation tag, so a viewer shows it the same way.
"""

from __future__ import annotations

import io
import json
import math
import shutil
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Literal

import structlog
from PIL import Image, ImageOps
from pydantic import ValidationError

from argus_curator.models import CropBox, CropConfig, WatermarkDetection

logger = structlog.get_logger()

# reject_reason prefix for an image whose crop could not be applied.
PREFIX = "crop:"
SIDECAR_SUFFIX = ".crop.json"
# Re-encode quality for JPEG and lossy WebP. Checks and the exported file both
# get this; it is well above what any check or trainer can tell apart.
QUALITY = 95
_EXIF_ORIENTATION = 0x0112
# More hits than this is a tiled watermark: no edge cut removes it.
MAX_HITS = 6


class CropError(ValueError):
    """A box that cannot be read or does not fit its image.

    The message is ``<kind> (<detail>)`` with kind ``unreadable``, ``invalid``,
    ``failed``, ``unchecked`` or ``overlay``, so the scan's reject tally groups
    them as ``crop:<kind>``.
    """


def sidecar_for(path: str | Path) -> Path:
    """``photo.jpg`` -> ``photo.jpg.crop.json``: keyed by the full name, so ``a.jpg`` and ``a.png`` do not share one."""
    return Path(f"{path}{SIDECAR_SUFFIX}")


def read_sidecar(path: str | Path) -> CropBox | None:
    """The box in *path*'s crop sidecar, ``None`` when it has none; :class:`CropError` when it cannot be read."""
    side = sidecar_for(path)
    try:
        # Inside the try: before Python 3.13 this raises on a name too long to take the suffix.
        if not side.is_file():
            return None
        return CropBox.model_validate(json.loads(side.read_text(encoding="utf-8")))
    except (OSError, ValueError, ValidationError) as exc:
        raise CropError(f"unreadable ({side.name}: {exc})") from exc


def crop_image(img: Image.Image, box: CropBox) -> Image.Image:
    """*img* upright, cut to *box*. :class:`CropError` when the box is empty or leaves the frame."""
    upright = ImageOps.exif_transpose(img)
    w, h = upright.size
    if box.right <= box.left or box.bottom <= box.top:
        raise CropError(f"invalid (empty box {box.left},{box.top},{box.right},{box.bottom})")
    if box.right > w or box.bottom > h:
        raise CropError(f"invalid (box {box.left},{box.top},{box.right},{box.bottom} leaves the {w}x{h} frame)")
    return upright.crop((box.left, box.top, box.right, box.bottom))


def _webp_lossless(data: bytes) -> bool:
    """Whether a WebP's image chunk is ``VP8L``. Pillow does not report it, so walk the RIFF chunks."""
    pos = 12
    while pos + 8 <= len(data):
        tag = data[pos : pos + 4]
        if tag in (b"VP8L", b"VP8 "):
            return tag == b"VP8L"
        size = int.from_bytes(data[pos + 4 : pos + 8], "little")
        pos += 8 + size + (size & 1)
    return False


def _encode(src: Image.Image, cropped: Image.Image, data: bytes) -> bytes:
    """*cropped* in the format of *src* (decoded from *data*), keeping its colour profile and EXIF less the orientation."""
    fmt = (src.format or "PNG").upper()
    if fmt == "MPO":
        # A JPEG with a second frame, as many phones write. Only the first is cropped, so it is a plain JPEG now.
        fmt = "JPEG"
    params: dict[str, object] = {}
    if icc := src.info.get("icc_profile"):
        params["icc_profile"] = icc
    exif = src.getexif()
    if exif:
        exif.pop(_EXIF_ORIENTATION, None)
        params["exif"] = exif.tobytes()
    if fmt == "JPEG":
        if cropped.mode not in ("RGB", "L", "CMYK"):
            cropped = cropped.convert("RGB")
        params["quality"] = QUALITY
        try:
            from PIL import JpegImagePlugin

            sampling = JpegImagePlugin.get_sampling(src)
            if sampling >= 0:
                params["subsampling"] = sampling
        except Exception:  # pragma: no cover - older Pillow
            pass
    elif fmt == "WEBP":
        params["lossless"] = _webp_lossless(data)
        params["quality"] = 100 if params["lossless"] else QUALITY
    buf = io.BytesIO()
    cropped.save(buf, format=fmt, **params)
    return buf.getvalue()


def crop_bytes(data: bytes, box: CropBox) -> bytes:
    """*data* cut to *box*, in the same format. :class:`CropError` when it cannot be."""
    try:
        src = Image.open(io.BytesIO(data))
        src.load()
    except Exception as exc:
        raise CropError(f"failed (cannot open: {exc})") from exc
    cropped = crop_image(src, box)
    try:
        return _encode(src, cropped, data)
    except Exception as exc:
        raise CropError(f"failed (cannot encode: {exc})") from exc


def resolve_box(rel: str, abs_path: str, cfg: CropConfig) -> CropBox | None:
    """The box for one image: an inline box first, then its sidecar (when *cfg* reads them)."""
    if rel in cfg.boxes:
        return cfg.boxes[rel]
    return read_sidecar(abs_path) if cfg.sidecars else None


def plan_crop(
    width: int, height: int, hits: list[WatermarkDetection], cfg: CropConfig
) -> CropBox | Literal["overlay"] | None:
    """The largest frame left after cutting every confident hit away along an edge.

    Each hit is cut off by one of the four edges, plus ``cfg.margin``; every
    way of assigning hits to edges is tried, and one where an edge loses more
    than ``cfg.max_cut`` of the frame's side is out. ``None`` when no hit
    reaches ``cfg.min_conf``, ``"overlay"`` when no allowed cut removes them all.
    """
    boxes = [h.bbox for h in hits if h.score >= cfg.min_conf]
    if not boxes:
        return None
    if len(boxes) > MAX_HITS:
        return "overlay"
    m = cfg.margin
    max_x, max_y = width * cfg.max_cut, height * cfg.max_cut
    best: tuple[int, CropBox] | None = None
    for edges in product(range(4), repeat=len(boxes)):
        top, bottom, left, right = 0.0, float(height), 0.0, float(width)
        for (x, y, w, h), edge in zip(boxes, edges, strict=True):
            if edge == 0:
                top = max(top, y + h + m)
            elif edge == 1:
                bottom = min(bottom, y - m)
            elif edge == 2:
                left = max(left, x + w + m)
            else:
                right = min(right, x - m)
        t, b = max(0, math.ceil(top)), min(height, math.floor(bottom))
        lf, r = max(0, math.ceil(left)), min(width, math.floor(right))
        if t > max_y or height - b > max_y or lf > max_x or width - r > max_x or r <= lf or b <= t:
            continue
        area = (r - lf) * (b - t)
        if best is None or area > best[0]:
            best = (area, CropBox(left=lf, top=t, right=r, bottom=b))
    return "overlay" if best is None else best[1]


@dataclass
class CropOutcome:
    """What :func:`apply` did to a scan's items."""

    items: list[tuple[str, str, bytes]]
    # rel_path -> the box now baked into its bytes, and where the box came from.
    applied: dict[str, CropBox] = field(default_factory=dict)
    sources: dict[str, Literal["manual", "detected"]] = field(default_factory=dict)
    # rel_path -> why it is a crop: reject (``<kind> (<detail>)``).
    errors: dict[str, str] = field(default_factory=dict)
    # rel_path -> the detector's hits, for every image it checked.
    watermarks: dict[str, list[WatermarkDetection]] = field(default_factory=dict)


def apply(
    items: list[tuple[str, str, bytes]],
    cfg: CropConfig,
    *,
    progress: Callable[[int, int], None] | None = None,
    cancelled: Callable[[], None] | None = None,
    workers: int = 1,
) -> CropOutcome:
    """Crop the items that have a box, by hand or (with ``cfg.detect``) found.

    An item whose crop cannot be applied, that the detector could not check,
    or (with ``cfg.reject_overlay``) whose watermark no edge cut removes keeps
    its original bytes and is named in ``errors`` for the scan to reject, so
    no later check judges it. With *cfg* off the items come back unchanged. *progress* and *cancelled* follow the detector pass;
    *workers* images are cropped at a time.
    """
    if not cfg.enabled:
        return CropOutcome(items=items)
    out = CropOutcome(items=[])
    boxes: dict[str, CropBox] = {}
    for rel, abs_path, _data in items:
        try:
            if (box := resolve_box(rel, abs_path, cfg)) is not None:
                boxes[rel] = box
                out.sources[rel] = "manual"
        except CropError as exc:
            out.errors[rel] = str(exc)
    unknown = sorted(set(cfg.boxes) - {rel for rel, _a, _d in items})
    if unknown:
        logger.warning("crop_boxes_unmatched", rel_paths=unknown[:20], count=len(unknown))

    if cfg.detect:
        from argus_curator import watermarks

        todo = {rel: data for rel, _a, data in items if rel not in boxes and rel not in out.errors}
        for rel, checked in watermarks.detect_all(todo, progress, cancelled).items():
            if isinstance(checked, str):
                out.errors[rel] = f"unchecked ({checked})"
                continue
            hits, (w, h) = checked
            out.watermarks[rel] = hits
            plan = plan_crop(w, h, hits, cfg)
            if plan == "overlay":
                if cfg.reject_overlay:
                    n = sum(1 for d in hits if d.score >= cfg.min_conf)
                    out.errors[rel] = f"overlay ({n} watermark(s) no edge cut of {cfg.max_cut:.0%} removes)"
            elif plan is not None:
                boxes[rel] = plan
                out.sources[rel] = "detected"

    # Decode + re-encode is the slow part; Pillow drops the GIL for both, so crop side by side.
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {rel: pool.submit(crop_bytes, data, boxes[rel]) for rel, _a, data in items if rel in boxes}
        for rel, abs_path, data in items:
            if (fut := futures.get(rel)) is not None:
                try:
                    data = fut.result()
                    out.applied[rel] = boxes[rel]
                except CropError as exc:
                    out.errors[rel] = str(exc)
                    out.sources.pop(rel, None)
            out.items.append((rel, abs_path, data))
    for rel, why in out.errors.items():
        logger.warning("crop_failed", rel_path=rel, error=why)
    logger.info("crops_applied", applied=len(out.applied), failed=len(out.errors), checked=len(out.watermarks))
    return out


def write_cropped(src: Path, dst: Path, box: CropBox) -> None:
    """Write *src* cut to *box* at *dst*. The source is only read: *dst* being *src* raises, as a copy would."""
    if dst.exists() and not dst.is_symlink() and dst.samefile(src):
        raise shutil.SameFileError(f"{src} and {dst} are the same file")
    data = crop_bytes(src.read_bytes(), box)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink():
        # A link left by an earlier symlink export: writing through it would overwrite the source.
        dst.unlink()
    dst.write_bytes(data)
