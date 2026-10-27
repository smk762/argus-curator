"""Shared onnxruntime plumbing for the optional classifier models.

Face attributes (FairFace, MiVOLO) and image tags (Places365) all run as ONNX
models converted once from their published weights. Each model's location is
process configuration — an env var with a default under :data:`MODEL_DIR` —
never a request field: a model path is a file the server would load.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

MODEL_DIR = Path.home() / ".cache" / "argus_curator" / "models"

_SESSION_CACHE: dict[str, object] = {}

# Every model here was trained on ImageNet-normalised RGB in [0, 1].
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ModelUnavailable(RuntimeError):
    """Raised when an optional model is requested but cannot be run."""


class ScanCancelled(RuntimeError):
    """The scan was stopped by its caller before it finished.

    Raised by the scanner from inside a phase's progress or per-image hook, so
    it crosses the check modules on its way out; they must let it through.
    """


def configured_path(env: str, default_name: str) -> Path:
    configured = os.environ.get(env, "").strip()
    return Path(configured).expanduser() if configured else MODEL_DIR / default_name


def onnxruntime_available() -> bool:
    try:
        import onnxruntime  # noqa: F401

        return True
    except Exception:
        return False


def providers(device: str) -> list[str]:
    """onnxruntime execution providers for a device hint ("auto" | "cpu" | "cuda")."""
    dev = (device or "auto").lower()
    if dev == "cpu":
        return ["CPUExecutionProvider"]
    try:
        import onnxruntime as ort

        available = set(ort.get_available_providers())
    except Exception:
        available = set()
    if dev == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    # auto: prefer CUDA when present.
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def load_session(path: Path, providers: list[str], *, missing_hint: str, extra: str = "faces"):
    """Build (and cache) an onnxruntime session for the model at *path*."""
    key = f"{path}:{','.join(providers)}"
    if key in _SESSION_CACHE:
        return _SESSION_CACHE[key]
    try:
        import onnxruntime as ort
    except Exception as exc:  # pragma: no cover - exercised only without deps
        raise ModelUnavailable(f"This model requires onnxruntime: pip install argus-curator[{extra}]") from exc
    if not path.is_file():
        raise ModelUnavailable(f"Model not found at {path}. {missing_hint}")
    session = ort.InferenceSession(str(path), providers=providers)
    _SESSION_CACHE[key] = session
    return session


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax in float64."""
    x = np.asarray(x, dtype=np.float64)
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
