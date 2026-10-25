"""Places365 scene tags for whole images: what kind of place, indoors or out.

Runs a Places365-CNN (Zhou et al., TPAMI 2017; weights CC BY, code MIT) on
each passing image and records its top scene categories with their softmax
probabilities, plus an indoor/outdoor call. Useful for the ``setting`` target
category ("beach only") and for varying backgrounds within an identity set.

Indoor/outdoor: upstream's demo votes over the top-10 categories' IO flags,
unweighted. We instead take the probability mass on outdoor categories, so
``environment_conf`` is a real probability rather than a vote share.

The model runs under onnxruntime; ``argus-curator places365-convert`` fetches
the published weights (plain HTTP upstream, so pinned by sha256) and exports
ONNX once.
"""

from __future__ import annotations

import io
import urllib.request
from collections.abc import Callable
from pathlib import Path

import numpy as np
import structlog
from PIL import Image

from argus_curator._places365_labels import CATEGORIES, OUTDOOR
from argus_curator.models import ImageResult, SceneConfig, SceneLabel, SceneTags
from argus_curator.runtime import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    MODEL_DIR,
    ModelUnavailable,
    configured_path,
    load_session,
    onnxruntime_available,
    providers,
    sha256,
    softmax,
)

logger = structlog.get_logger()

MODEL_ENV = "ARGUS_CURATOR_PLACES365_MODEL"
DEFAULT_MODEL_NAME = "places365_resnet18.onnx"

# Published PyTorch checkpoints. Upstream serves them over plain HTTP only, so
# the hash is what makes the download trustworthy. best_prec1 (top-1 on the
# Places365 validation set, from the checkpoints themselves): 54.5 / 55.6.
WEIGHTS = {
    "resnet18": (
        "http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar",
        "2f4759217d470da2b803f8f66cd4488a066406b555a5fb95ee9a4663f9f05588",
    ),
    "resnet50": (
        "http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar",
        "46529c86902bd0cfb0ea562a30b2850c28d2620d96282b3db9c318e1d774f6c5",
    ),
}

RESIZE = 256
INPUT_SIZE = 224
BATCH = 16
_OUTDOOR = np.array(OUTDOOR, dtype=np.float64)


def model_path() -> Path:
    """Where the converted Places365 ONNX model is read from."""
    return configured_path(MODEL_ENV, DEFAULT_MODEL_NAME)


def scenes_available() -> bool:
    """True when onnxruntime is importable and the Places365 model is in place."""
    return onnxruntime_available() and model_path().is_file()


def get_session(provs: list[str]):
    return load_session(
        model_path(),
        provs,
        missing_hint=f"Create it with `argus-curator places365-convert` or set {MODEL_ENV}.",
        extra="scenes",
    )


def preprocess(images: list[Image.Image]) -> np.ndarray:
    """Upstream's transform: resize to 256x256 (no aspect keep), centre-crop 224."""
    off = (RESIZE - INPUT_SIZE) // 2
    arr = np.stack(
        [
            np.asarray(
                im.convert("RGB")
                .resize((RESIZE, RESIZE), Image.Resampling.BILINEAR)
                .crop((off, off, off + INPUT_SIZE, off + INPUT_SIZE)),
                dtype=np.float32,
            )
            / 255.0
            for im in images
        ]
    )
    return ((arr - IMAGENET_MEAN) / IMAGENET_STD).transpose(0, 3, 1, 2).astype(np.float32)


def to_tags(probs: np.ndarray, top_k: int) -> SceneTags:
    """Top-k categories and the indoor/outdoor call from one ``(365,)`` row."""
    order = np.argsort(-probs)[:top_k]
    p_out = float(probs @ _OUTDOOR)
    outdoor = p_out >= 0.5
    return SceneTags(
        categories=[SceneLabel(label=CATEGORIES[i], conf=round(float(probs[i]), 4)) for i in order],
        environment="outdoor" if outdoor else "indoor",
        environment_conf=round(p_out if outdoor else 1.0 - p_out, 4),
    )


def predict(session, images: list[Image.Image]) -> np.ndarray:
    """``(N, 365)`` probabilities for a batch of images."""
    input_name = session.get_inputs()[0].name
    (logits,) = session.run(None, {input_name: preprocess(images)})
    return softmax(logits)


def tag_images(
    results: list[ImageResult],
    items: list[tuple[str, str, bytes]],
    cfg: SceneConfig,
    progress: Callable[[int, int], None] | None = None,
) -> None:
    """Set ``scene`` on every passing result, in batches of :data:`BATCH`."""
    if not cfg.enabled:
        return
    session = get_session(providers(cfg.device))
    data_by_rel = {rel: data for rel, _abs, data in items}
    pending = [r for r in results if r.passed and data_by_rel.get(r.rel_path) is not None]
    total = len(pending)
    if progress is not None:
        progress(0, total)
    for start in range(0, total, BATCH):
        batch: list[tuple[ImageResult, Image.Image]] = []
        for r in pending[start : start + BATCH]:
            try:
                # convert() forces the decode: Image.open is lazy, and a truncated
                # file failing later in predict() would cost the whole batch.
                batch.append((r, Image.open(io.BytesIO(data_by_rel[r.rel_path])).convert("RGB")))
            except Exception as exc:
                logger.warning("scene_decode_failed", rel_path=r.rel_path, error=str(exc))
        if batch:
            try:
                probs = predict(session, [im for _r, im in batch])
            except Exception as exc:
                logger.warning("scene_batch_failed", first=batch[0][0].rel_path, error=str(exc))
            else:
                for (r, _im), row in zip(batch, probs, strict=True):
                    r.scene = to_tags(row, cfg.top_k)
        if progress is not None:
            progress(min(start + BATCH, total), total)


def fetch_weights(arch: str) -> Path:
    """Download (once) and verify the published checkpoint for *arch*."""
    url, expected = WEIGHTS[arch]
    dest = MODEL_DIR / "src" / url.rsplit("/", 1)[1]
    if not (dest.is_file() and sha256(dest) == expected):
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        with urllib.request.urlopen(url, timeout=60) as resp, tmp.open("wb") as f:  # noqa: S310 - fixed URL, hash-checked
            while chunk := resp.read(1 << 20):
                f.write(chunk)
        digest = sha256(tmp)
        if digest != expected:
            tmp.unlink(missing_ok=True)
            raise ModelUnavailable(f"download of {url} does not match the pinned sha256 (got {digest})")
        tmp.replace(dest)
    return dest


def export_onnx(weights: Path, arch: str, output: Path) -> Path:
    """Convert a Places365 ``.pth.tar`` checkpoint into an ONNX model.

    Needs torch + torchvision (``pip install argus-curator[places365-convert]``)
    only for this one-time step; inference uses onnxruntime.
    """
    try:
        import torch
        import torchvision
    except Exception as exc:
        raise ModelUnavailable("Conversion requires: pip install argus-curator[places365-convert]") from exc

    _url, expected = WEIGHTS[arch]
    digest = sha256(weights)
    if digest != expected:
        raise ModelUnavailable(f"{weights} is not the pinned {arch} Places365 checkpoint (sha256 {digest})")

    # weights_only: the checkpoint is a plain dict of tensors + metadata, so it
    # loads without unpickling arbitrary objects.
    checkpoint = torch.load(str(weights), map_location="cpu", weights_only=True)
    state = {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()}
    model = torchvision.models.__dict__[arch](num_classes=len(CATEGORIES))
    model.load_state_dict(state, strict=True)
    model.eval()

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE),
        str(output),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    return output
