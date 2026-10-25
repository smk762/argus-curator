"""Fashionpedia garment detection: what someone is wearing, where, what colour.

Runs ``valentinafevu/yolos-fashionpedia`` (YOLOS-small fine-tuned on
Fashionpedia; model MIT, dataset CC BY 4.0) on each passing image. Fashionpedia
has 27 garment classes and 19 garment *parts* (sleeve, pocket, zipper, ...);
only the garments are recorded — on 300 Fashionpedia validation images the
parts scored AP50 0.105 against 0.327 for garments, too weak to curate on.

Each detection carries its softmax probability, its box, whether that box
touches the frame edge (a garment likely cut off), and a dominant colour from
the box's central region — garment types alone cannot tell a red dress from a
black one, which is what outfit diversity needs.

The model runs under onnxruntime; ``argus-curator garments-convert`` downloads
the pinned checkpoint, verifies it, and exports ONNX once.
"""

from __future__ import annotations

import io
from collections.abc import Callable
from pathlib import Path

import numpy as np
import structlog
from PIL import Image

from argus_curator.models import GARMENT_LABELS, GarmentConfig, GarmentDetection, ImageResult
from argus_curator.runtime import (
    ModelUnavailable,
    configured_path,
    load_session,
    onnxruntime_available,
    providers,
    sha256,
)

logger = structlog.get_logger()

MODEL_ENV = "ARGUS_CURATOR_GARMENTS_MODEL"
DEFAULT_MODEL_NAME = "yolos_fashionpedia.onnx"

HF_REPO = "valentinafevu/yolos-fashionpedia"
HF_REVISION = "0cfdb77c01cfbf0838b940ecff9c572f84c9d4f4"
HF_FILES = ("config.json", "pytorch_model.bin")
WEIGHTS_SHA256 = "1b5f4cc272f8f32b2f25b129f9ce7df4756cb030d5fbd234c790b6a9eef962a1"

# YolosImageProcessor settings from the checkpoint's preprocessor_config.json.
SHORT_EDGE = 512
MAX_EDGE = 1333
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# Touching the frame within this fraction of the image size counts as cut off.
_EDGE_MARGIN = 0.01


def model_path() -> Path:
    """Where the converted garment-detector ONNX model is read from."""
    return configured_path(MODEL_ENV, DEFAULT_MODEL_NAME)


def garments_available() -> bool:
    return onnxruntime_available() and model_path().is_file()


def get_session(provs: list[str]):
    return load_session(
        model_path(),
        provs,
        missing_hint=f"Create it with `argus-curator garments-convert` or set {MODEL_ENV}.",
        extra="scenes",
    )


def resize_size(width: int, height: int) -> tuple[int, int]:
    """YOLOS's target size: short edge 512, long edge <= 1333, both floored to /16."""
    short, long_ = min(width, height), max(width, height)
    size = SHORT_EDGE
    if long_ / short * size > MAX_EDGE:
        size = int(round(MAX_EDGE * short / long_))
    if width < height:
        new_w, new_h = size, int(size * height / width)
    else:
        new_w, new_h = int(size * width / height), size
    return new_w - new_w % 16, new_h - new_h % 16


def preprocess(image: Image.Image) -> np.ndarray:
    """``(1, 3, H, W)`` ImageNet-normalised input at YOLOS's resize."""
    rgb = image.convert("RGB")
    rgb = rgb.resize(resize_size(*rgb.size), Image.Resampling.BILINEAR)
    arr = (np.asarray(rgb, dtype=np.float32) / 255.0 - _MEAN) / _STD
    return arr.transpose(2, 0, 1)[None].astype(np.float32)


def postprocess(logits: np.ndarray, boxes: np.ndarray, width: int, height: int, min_conf: float):
    """``[(label_index, conf, [x, y, w, h])]`` for garment-class queries >= *min_conf*.

    Softmax over the 46 classes plus the trailing no-object class, which is then
    dropped (as ``post_process_object_detection`` does); boxes come out as
    normalised centre/size and are scaled to the original image.
    """
    logits = np.asarray(logits, dtype=np.float64)
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = (e / e.sum(axis=-1, keepdims=True))[:, :-1]
    labels, scores = probs.argmax(axis=-1), probs.max(axis=-1)
    out = []
    for lbl, conf, (cx, cy, bw, bh) in zip(labels, scores, boxes, strict=True):
        if lbl >= len(GARMENT_LABELS) or conf < min_conf:
            continue
        x1 = max(0.0, (cx - bw / 2) * width)
        y1 = max(0.0, (cy - bh / 2) * height)
        x2 = min(float(width), (cx + bw / 2) * width)
        y2 = min(float(height), (cy + bh / 2) * height)
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        out.append((int(lbl), float(conf), [x1, y1, x2 - x1, y2 - y1]))
    return out


def dominant_color(image: Image.Image, bbox: list[float]) -> str:
    """Mean colour of the box's central half, as ``#rrggbb``.

    The centre avoids the background a detection box always includes; a mean
    rather than a mode keeps this cheap and stable on patterned fabric.
    """
    x, y, w, h = bbox
    x1, y1 = int(x + w / 4), int(y + h / 4)
    # At least one pixel, so a sliver of a box still yields a colour.
    crop = image.crop((x1, y1, max(x1 + 1, int(x + 3 * w / 4)), max(y1 + 1, int(y + 3 * h / 4))))
    r, g, b = (int(round(v)) for v in np.asarray(crop.convert("RGB"), dtype=np.float64).reshape(-1, 3).mean(axis=0))
    return f"#{r:02x}{g:02x}{b:02x}"


def touches_edge(bbox: list[float], width: int, height: int) -> bool:
    x, y, w, h = bbox
    mx, my = _EDGE_MARGIN * width, _EDGE_MARGIN * height
    return x <= mx or y <= my or x + w >= width - mx or y + h >= height - my


def detect(session, image: Image.Image, min_conf: float) -> list[GarmentDetection]:
    """Garments in one image, most confident first."""
    rgb = image.convert("RGB")
    input_name = session.get_inputs()[0].name
    logits, boxes = session.run(None, {input_name: preprocess(rgb)})
    found = postprocess(logits[0], boxes[0], rgb.width, rgb.height, min_conf)
    dets = [
        GarmentDetection(
            label=GARMENT_LABELS[lbl],
            conf=round(conf, 4),
            bbox=[round(v, 1) for v in bbox],
            touches_edge=touches_edge(bbox, rgb.width, rgb.height),
            color=dominant_color(rgb, bbox),
        )
        for lbl, conf, bbox in found
    ]
    return sorted(dets, key=lambda d: -d.conf)


def detect_images(
    results: list[ImageResult],
    items: list[tuple[str, str, bytes]],
    cfg: GarmentConfig,
    progress: Callable[[int, int], None] | None = None,
) -> None:
    """Set ``garments`` on every passing result (``[]`` when none were found)."""
    if not cfg.enabled:
        return
    session = get_session(providers(cfg.device))
    data_by_rel = {rel: data for rel, _abs, data in items}
    pending = [r for r in results if r.passed and data_by_rel.get(r.rel_path) is not None]
    total = len(pending)
    step = max(1, total // 100)
    if progress is not None:
        progress(0, total)
    for i, r in enumerate(pending):
        if progress is not None and i % step == 0:
            progress(i, total)
        try:
            r.garments = detect(session, Image.open(io.BytesIO(data_by_rel[r.rel_path])), cfg.min_conf)
        except Exception as exc:
            logger.warning("garment_detect_failed", rel_path=r.rel_path, error=str(exc))
    if progress is not None:
        progress(total, total)


# ---------------------------------------------------------------------------
# Outfit similarity (for diversity-aware selection)
# ---------------------------------------------------------------------------


def _lab(hex_color: str) -> np.ndarray:
    """sRGB ``#rrggbb`` -> CIE L*a*b* (D65)."""
    rgb = np.array([int(hex_color[i : i + 2], 16) for i in (1, 3, 5)], dtype=np.float64) / 255.0
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = (np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]) @ lin) / np.array(
        [0.95047, 1.0, 1.08883]
    )
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def outfit(garments: list[GarmentDetection] | None, min_conf: float) -> dict[str, np.ndarray]:
    """Label -> Lab colour of the most confident detection of each garment type."""
    sig: dict[str, np.ndarray] = {}
    for g in sorted(garments or [], key=lambda d: -d.conf):
        if g.conf >= min_conf and g.label not in sig:
            sig[g.label] = _lab(g.color)
    return sig


def outfit_similarity(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> float:
    """0..1: shared garment types, each weighted by how close their colours are.

    A shared type counts fully at identical colour and fades to nothing at a
    CIE76 distance of 50 (roughly "clearly a different colour").
    """
    union = a.keys() | b.keys()
    if not union:
        return 0.0
    shared = a.keys() & b.keys()
    return sum(max(0.0, 1.0 - float(np.linalg.norm(a[k] - b[k])) / 50.0) for k in shared) / len(union)


# ---------------------------------------------------------------------------
# One-time conversion
# ---------------------------------------------------------------------------


def fetch_weights() -> Path:
    """Download the pinned checkpoint from the Hugging Face Hub (cached there)."""
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        raise ModelUnavailable("Conversion requires: pip install argus-curator[garments-convert]") from exc
    paths = [Path(hf_hub_download(HF_REPO, f, revision=HF_REVISION)) for f in HF_FILES]
    return paths[0].parent


def export_onnx(checkpoint_dir: Path, output: Path) -> Path:
    """Convert the pinned YOLOS checkpoint into a single-file ONNX model.

    The weights load with ``weights_only`` (the checkpoint is a pickled state
    dict; nothing else is unpickled) and strictly, after renaming the one key
    newer transformers moved — a lax load would leave it randomly initialised.
    Needs ``pip install argus-curator[garments-convert]``.
    """
    try:
        import torch
        from transformers import YolosConfig, YolosForObjectDetection
    except Exception as exc:
        raise ModelUnavailable("Conversion requires: pip install argus-curator[garments-convert]") from exc

    weights = checkpoint_dir / "pytorch_model.bin"
    digest = sha256(weights)
    if digest != WEIGHTS_SHA256:
        raise ModelUnavailable(f"{weights} is not the pinned {HF_REPO} checkpoint (sha256 {digest})")

    model = YolosForObjectDetection(YolosConfig.from_pretrained(str(checkpoint_dir)))
    state = torch.load(str(weights), map_location="cpu", weights_only=True)
    state = {
        k.replace("vit.encoder.mid_position_embeddings", "vit.encoder.interpolation.mid_position_embeddings"): v
        for k, v in state.items()
    }
    model.load_state_dict(state, strict=True)
    model.eval()

    class _Outputs(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            out = self.inner(pixel_values=x)
            return out.logits, out.pred_boxes

    output.parent.mkdir(parents=True, exist_ok=True)
    patches = torch.export.Dim
    torch.onnx.export(
        _Outputs(model).eval(),
        (torch.zeros(1, 3, SHORT_EDGE, 768),),
        str(output),
        input_names=["input"],
        output_names=["logits", "boxes"],
        # YOLOS interpolates its position embeddings to the input, so height
        # and width stay dynamic (in 16-px patches) rather than fixing a canvas.
        dynamic_shapes=({2: 16 * patches("hp", min=8, max=84), 3: 16 * patches("wp", min=8, max=84)},),
        opset_version=18,
        dynamo=True,
        external_data=False,
    )
    return output
