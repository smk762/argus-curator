"""MiVOLO v2 age (years) and gender for detected faces.

MiVOLO (Kuprashevich & Tolstykh, 2023; ``iitolstykh/mivolo_v2``, Apache-2.0)
is a VOLO transformer that reads a face crop and a person crop side by side.
With ``FaceConfig.mivolo_body`` each face is given its body from InsightFace's
person detector (see :mod:`bodies`); otherwise, or for a face with no usable
body, the person channels carry the "absent" image upstream feeds when
``body_crops`` is ``None`` (zeros, then normalised). Face-only is a supported
mode, at some cost in accuracy against the face+body numbers MiVOLO publishes.

Output is ``[gender logit x2, normalised age]``; age in years is
``raw * (max_age - min_age) + avg_age`` from the checkpoint's config. Age is a
regression, so unlike FairFace it has no probability; gender's softmax does.

The model runs under onnxruntime; ``argus-curator mivolo-convert`` downloads the
pinned checkpoint, verifies it, and exports ONNX once.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from argus_curator.runtime import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    ModelUnavailable,
    configured_path,
    load_session,
    sha256,
    softmax,
)

MODEL_ENV = "ARGUS_CURATOR_MIVOLO_MODEL"
DEFAULT_MODEL_NAME = "mivolo_v2_d1_384.onnx"

# The checkpoint mivolo-convert fetches, pinned by commit and content hash so a
# re-pushed `main` cannot change what gets converted.
HF_REPO = "iitolstykh/mivolo_v2"
HF_REVISION = "53393526c220e34cdd7b722b36d22b6f9e5f4241"
HF_FILENAME = "model.safetensors"
WEIGHTS_SHA256 = "96efb47051c038ebeec74b73b4253c5fd000433e5afcab7deee0bd8f3fa7bf18"

INPUT_SIZE = 384
# config.json of the checkpoint: min_age 0, max_age 122, avg_age 61.
_MIN_AGE, _MAX_AGE, _AVG_AGE = 0.0, 122.0, 61.0
# Upstream's stand-in for a missing person crop: a black image, normalised.
_ABSENT = np.broadcast_to(((0.0 - IMAGENET_MEAN) / IMAGENET_STD)[:, None, None], (3, INPUT_SIZE, INPUT_SIZE)).astype(
    np.float32
)


def model_path() -> Path:
    """Where the converted MiVOLO ONNX model is read from."""
    return configured_path(MODEL_ENV, DEFAULT_MODEL_NAME)


def get_session(providers: list[str]):
    return load_session(
        model_path(),
        providers,
        missing_hint=f"Create it with `argus-curator mivolo-convert` or set {MODEL_ENV}.",
    )


def letterbox(crop: Image.Image) -> Image.Image:
    """Fit *crop* into a black 384x384 square, aspect preserved, centred.

    Mirrors upstream ``class_letterbox`` (which also scales up small crops).
    """
    w, h = crop.size
    r = min(INPUT_SIZE / h, INPUT_SIZE / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    if (nw, nh) != (w, h):
        crop = crop.resize((nw, nh), Image.Resampling.BILINEAR)
    left = int(round((INPUT_SIZE - nw) / 2 - 0.1))
    top = int(round((INPUT_SIZE - nh) / 2 - 0.1))
    canvas = Image.new("RGB", (INPUT_SIZE, INPUT_SIZE))
    canvas.paste(crop.convert("RGB"), (left, top))
    return canvas


def face_crop(image: Image.Image, bbox_xywh: list[float]) -> Image.Image | None:
    """The detector box, clamped to the image — MiVOLO takes it unaligned."""
    x, y, w, h = bbox_xywh
    x1, y1 = max(0, int(x)), max(0, int(y))
    x2, y2 = min(image.width, int(x + w)), min(image.height, int(y + h))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return image.crop((x1, y1, x2, y2))


def _channels(crop: Image.Image) -> np.ndarray:
    """``(3, 384, 384)``: *crop* letterboxed and normalised as MiVOLO was trained."""
    pixels = np.asarray(letterbox(crop), dtype=np.float32) / 255.0
    return ((pixels - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)


def preprocess(crops: list[Image.Image], bodies: list[Image.Image | None] | None = None) -> np.ndarray:
    """``(N, 6, 384, 384)``: normalised face channels + person channels (absent when no body)."""
    bodies = bodies if bodies is not None else [None] * len(crops)
    faces = np.stack([_channels(c) for c in crops])
    persons = np.stack([_ABSENT if b is None else _channels(b) for b in bodies])
    return np.concatenate([faces, persons], axis=1).astype(np.float32)


def decode(output: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(gender_probs (N, 2) [male, female], age_years (N,))`` from raw output."""
    output = np.asarray(output, dtype=np.float64)
    gender = softmax(output[:, :2])
    age = output[:, 2] * (_MAX_AGE - _MIN_AGE) + _AVG_AGE
    return gender, np.clip(age, _MIN_AGE, _MAX_AGE)


def predict(
    session,
    image: Image.Image,
    bboxes: list[list[float]],
    body_crops: list[Image.Image | None] | None = None,
) -> list[tuple[np.ndarray, float] | None]:
    """Per face: ``(gender_probs, age_years)``, or ``None`` for a degenerate box.

    *body_crops*, one per box, are the faces' masked bodies (:func:`bodies.crops_for_faces`);
    ``None`` for a face, or for all of them, runs that face face-only.
    """
    crops = [face_crop(image, b) for b in bboxes]
    idx = [i for i, c in enumerate(crops) if c is not None]
    out: list[tuple[np.ndarray, float] | None] = [None] * len(bboxes)
    if not idx:
        return out
    batch = preprocess([crops[i] for i in idx], [body_crops[i] for i in idx] if body_crops is not None else None)
    input_name = session.get_inputs()[0].name
    (raw,) = session.run(None, {input_name: batch})
    gender, age = decode(raw)
    for j, i in enumerate(idx):
        out[i] = (gender[j], float(age[j]))
    return out


def predict_bodies(session, body_crops: list[Image.Image]) -> list[tuple[np.ndarray, float]]:
    """Per body-only crop: ``(gender_probs, age_years)``, face channels absent.

    Upstream's missing-face stand-in is the same black image used for a missing
    person. A frame that shows a body and no face still gets an age.
    """
    if not body_crops:
        return []
    faces = np.broadcast_to(_ABSENT, (len(body_crops), *_ABSENT.shape))
    persons = np.stack([_channels(crop) for crop in body_crops])
    batch = np.concatenate([faces, persons], axis=1).astype(np.float32)
    input_name = session.get_inputs()[0].name
    (raw,) = session.run(None, {input_name: batch})
    gender, age = decode(raw)
    return [(gender[i], float(age[i])) for i in range(len(body_crops))]


def fetch_weights() -> Path:
    """Download the pinned checkpoint from the Hugging Face Hub (cached there)."""
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        raise ModelUnavailable("Conversion requires: pip install argus-curator[mivolo-convert]") from exc
    return Path(hf_hub_download(HF_REPO, HF_FILENAME, revision=HF_REVISION))


def export_onnx(weights: Path, output: Path) -> Path:
    """Convert a MiVOLO v2 ``model.safetensors`` into a single-file ONNX model.

    The checkpoint must hash to :data:`WEIGHTS_SHA256`: the architecture in
    ``_mivolo_arch`` is the one verified against it, and the output decoding
    here assumes its config. Needs ``pip install argus-curator[mivolo-convert]``.
    """
    try:
        import torch
        from safetensors.torch import load_file

        from argus_curator._mivolo_arch import MiVOLOModel
    except Exception as exc:
        raise ModelUnavailable("Conversion requires: pip install argus-curator[mivolo-convert]") from exc

    digest = sha256(weights)
    if digest != WEIGHTS_SHA256:
        raise ModelUnavailable(
            f"{weights} is not the pinned MiVOLO v2 checkpoint (sha256 {digest}, expected {WEIGHTS_SHA256})"
        )

    state = {k.removeprefix("mivolo.model."): v.float() for k, v in load_file(str(weights)).items()}
    model = MiVOLOModel()
    model.load_state_dict(state, strict=True)
    model.eval()

    output.parent.mkdir(parents=True, exist_ok=True)
    # The TorchScript exporter cannot lower VOLO's F.fold (Col2Im needs a static
    # output size); the dynamo exporter traces it with concrete shapes.
    torch.onnx.export(
        model,
        (torch.zeros(2, 6, INPUT_SIZE, INPUT_SIZE),),
        str(output),
        input_names=["input"],
        output_names=["output"],
        dynamic_shapes=({0: torch.export.Dim("batch", min=1, max=256)},),
        opset_version=18,
        dynamo=True,
        external_data=False,
    )
    return output
