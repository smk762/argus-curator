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
from pathlib import Path

import structlog
from PIL import Image, ImageOps
from pydantic import ValidationError

from argus_curator.models import CropBox, CropConfig, ImageResult

logger = structlog.get_logger()

# reject_reason prefix for an image whose crop could not be applied.
PREFIX = "crop:"
SIDECAR_SUFFIX = ".crop.json"
# Re-encode quality for JPEG and lossy WebP. Checks and the exported file both
# get this; it is well above what any check or trainer can tell apart.
QUALITY = 95
_EXIF_ORIENTATION = 0x0112


class CropError(ValueError):
    """A box that cannot be read or does not fit its image.

    The message is ``<kind> (<detail>)`` with kind ``unreadable``, ``invalid``
    or ``failed``, so the scan's reject tally groups them as ``crop:<kind>``.
    """


def sidecar_for(path: str | Path) -> Path:
    """``photo.jpg`` -> ``photo.jpg.crop.json``: keyed by the full name, so ``a.jpg`` and ``a.png`` do not share one."""
    return Path(f"{path}{SIDECAR_SUFFIX}")


def read_sidecar(path: str | Path) -> CropBox | None:
    """The box in *path*'s crop sidecar, ``None`` when it has none; :class:`CropError` when it cannot be read."""
    side = sidecar_for(path)
    if not side.is_file():
        return None
    try:
        return CropBox.model_validate(json.loads(side.read_text(encoding="utf-8")))
    except (OSError, ValueError, ValidationError) as exc:
        raise CropError(f"unreadable ({side.name}: {exc})") from exc


def is_crop_reject(result: ImageResult) -> bool:
    """Whether the scan rejected *result* because its crop could not be applied."""
    return not result.passed and (result.reject_reason or "").startswith(PREFIX)


def crop_image(img: Image.Image, box: CropBox) -> Image.Image:
    """*img* upright, cut to *box*. :class:`CropError` when the box is empty or leaves the frame."""
    upright = ImageOps.exif_transpose(img)
    w, h = upright.size
    if box.right <= box.left or box.bottom <= box.top:
        raise CropError(f"invalid (empty box {box.left},{box.top},{box.right},{box.bottom})")
    if box.right > w or box.bottom > h:
        raise CropError(f"invalid (box {box.left},{box.top},{box.right},{box.bottom} leaves the {w}x{h} frame)")
    return upright.crop((box.left, box.top, box.right, box.bottom))


def _encode(src: Image.Image, cropped: Image.Image) -> bytes:
    """*cropped* in *src*'s format, keeping its colour profile and EXIF less the orientation."""
    fmt = (src.format or "PNG").upper()
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
        params["lossless"] = bool(src.info.get("lossless"))
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
        return _encode(src, cropped)
    except Exception as exc:
        raise CropError(f"failed (cannot encode: {exc})") from exc


def resolve_box(rel: str, abs_path: str, cfg: CropConfig) -> CropBox | None:
    """The box for one image: an inline box first, then its sidecar (when *cfg* reads them)."""
    if rel in cfg.boxes:
        return cfg.boxes[rel]
    return read_sidecar(abs_path) if cfg.sidecars else None


def apply(
    items: list[tuple[str, str, bytes]], cfg: CropConfig
) -> tuple[list[tuple[str, str, bytes]], dict[str, CropBox], dict[str, str]]:
    """Crop the boxed items. Returns ``(items, applied, errors)``.

    ``applied`` maps rel_path to the box now baked into its bytes. ``errors``
    maps rel_path to why its box could not be applied; that item keeps its
    original bytes (so NudeNet still scores what is on disk) and the scan
    rejects it. With *cfg* off the items come back unchanged.
    """
    if not cfg.enabled:
        return items, {}, {}
    out: list[tuple[str, str, bytes]] = []
    applied: dict[str, CropBox] = {}
    errors: dict[str, str] = {}
    for rel, abs_path, data in items:
        try:
            box = resolve_box(rel, abs_path, cfg)
            if box is not None:
                data = crop_bytes(data, box)
                applied[rel] = box
        except CropError as exc:
            errors[rel] = str(exc)
            logger.warning("crop_failed", rel_path=rel, error=str(exc))
        out.append((rel, abs_path, data))
    unknown = sorted(set(cfg.boxes) - {rel for rel, _a, _d in items})
    if unknown:
        logger.warning("crop_boxes_unmatched", rel_paths=unknown[:20], count=len(unknown))
    logger.info("crops_applied", applied=len(applied), failed=len(errors))
    return out, applied, errors


def write_cropped(src: Path, dst: Path, box: CropBox) -> None:
    """Write *src* cut to *box* at *dst*. The source is only read."""
    data = crop_bytes(src.read_bytes(), box)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink():
        # A link left by an earlier symlink export: writing through it would overwrite the source.
        dst.unlink()
    dst.write_bytes(data)
