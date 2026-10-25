"""FairFace race / gender / age bracket for detected faces.

Runs the FairFace ResNet-34 (Kärkkäinen & Joo, WACV 2021; weights CC BY 4.0)
on each face InsightFace already found. The 18 logits split into three heads —
race (7), gender (2), age bracket (9) — and each head is softmaxed, so every
label carries its probability as a confidence, exactly as the reference
``predict.py`` reports ``*_scores_fair``.

The model runs under onnxruntime like the rest of the face stack; the official
PyTorch weights are converted once with ``argus-curator fairface-convert``.

FairFace was trained on dlib ``get_face_chip(size=300, padding=0.25)`` crops.
We have InsightFace's 5 keypoints instead of dlib's landmarks, so the chip is
reproduced by mapping those keypoints onto dlib's own mean-face template under
the same padding — an approximation of the training crop, not a bit-exact one.
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
    softmax,
)

MODEL_ENV = "ARGUS_CURATOR_FAIRFACE_MODEL"
DEFAULT_MODEL_NAME = "fairface_res34_race7.onnx"

INPUT_SIZE = 224
_PADDING = 0.25

# dlib's normalised mean face (face_utilities.h), reduced to InsightFace's five
# keypoints in image order: left eye, right eye, nose tip, mouth left/right.
# Eyes are the centroid of dlib points 36-41 / 42-47; nose is 30; mouth 48 / 54.
_DLIB_KPS = np.array(
    [
        [0.2255, 0.1974],
        [0.7548, 0.1974],
        [0.4901, 0.4182],
        [0.2541, 0.6317],
        [0.7261, 0.6317],
    ],
    dtype=np.float64,
)
# get_face_chip pads the template by `padding` on every side before scaling.
_TEMPLATE = (_DLIB_KPS + _PADDING) / (1.0 + 2.0 * _PADDING) * INPUT_SIZE


def model_path() -> Path:
    """Where the converted FairFace ONNX model is read from."""
    return configured_path(MODEL_ENV, DEFAULT_MODEL_NAME)


def get_session(providers: list[str]):
    return load_session(
        model_path(),
        providers,
        missing_hint=f"Convert the official weights with `argus-curator fairface-convert "
        f"<fairface_alldata_20191111.pt>` or set {MODEL_ENV}.",
    )


def _similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity transform (Umeyama) mapping *src* points onto *dst*.

    Returns the 2x3 affine matrix ``[[a, b, tx], [c, d, ty]]``.
    """
    n = src.shape[0]
    src_mean, dst_mean = src.mean(axis=0), dst.mean(axis=0)
    src_c, dst_c = src - src_mean, dst - dst_mean
    cov = dst_c.T @ src_c / n
    u, s, vt = np.linalg.svd(cov)
    d = np.ones(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        d[-1] = -1.0
    rot = u @ np.diag(d) @ vt
    var_src = (src_c**2).sum() / n
    scale = (s * d).sum() / var_src if var_src > 0 else 1.0
    t = dst_mean - scale * rot @ src_mean
    return np.hstack([scale * rot, t[:, None]])


def align_face(image: Image.Image, kps: np.ndarray) -> Image.Image:
    """Warp the face with 5 keypoints *kps* into a 224x224 FairFace-style chip."""
    # PIL's affine transform maps output pixels back to input pixels, so solve
    # template -> image rather than image -> template.
    inv = _similarity(_TEMPLATE, np.asarray(kps, dtype=np.float64))
    coeffs = tuple(inv.flatten())
    return image.transform((INPUT_SIZE, INPUT_SIZE), Image.Transform.AFFINE, coeffs, resample=Image.Resampling.BILINEAR)


def preprocess(chips: list[Image.Image]) -> np.ndarray:
    """Stack RGB chips into an ImageNet-normalised NCHW float32 batch."""
    arr = np.stack([np.asarray(c.convert("RGB"), dtype=np.float32) / 255.0 for c in chips])
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr.transpose(0, 3, 1, 2).astype(np.float32)


def split_probs(logits: np.ndarray) -> np.ndarray:
    """Softmax each FairFace head of an ``(N, 18)`` logit batch independently."""
    logits = np.asarray(logits, dtype=np.float64)
    return np.concatenate([softmax(logits[:, :7]), softmax(logits[:, 7:9]), softmax(logits[:, 9:18])], axis=1)


def predict(session, image: Image.Image, kps_list: list[np.ndarray]) -> np.ndarray:
    """Per-head probabilities ``(N, 18)`` for each face (one batched forward pass)."""
    if not kps_list:
        return np.zeros((0, 18))
    batch = preprocess([align_face(image, kps) for kps in kps_list])
    input_name = session.get_inputs()[0].name
    (logits,) = session.run(None, {input_name: batch})
    return split_probs(logits)


def export_onnx(weights: Path, output: Path) -> Path:
    """Convert the official FairFace ``.pt`` state dict to an ONNX model.

    Needs torch + torchvision (``pip install argus-curator[fairface-convert]``)
    only for this one-time step; inference uses onnxruntime.
    """
    try:
        import torch
        import torchvision
    except Exception as exc:
        raise ModelUnavailable("Conversion requires: pip install argus-curator[fairface-convert]") from exc

    model = torchvision.models.resnet34(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 18)
    state = torch.load(str(weights), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    output.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE)
    torch.onnx.export(
        model,
        dummy,
        str(output),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    return output
