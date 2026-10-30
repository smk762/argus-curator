"""Shared onnxruntime plumbing for the optional classifier models.

Face attributes (FairFace, MiVOLO) and image tags (Places365) all run as ONNX
models converted once from their published weights. Each model's location is
process configuration — an env var with a default under :data:`MODEL_DIR` —
never a request field: a model path is a file the server would load.
"""

from __future__ import annotations

import contextlib
import os
import threading
from pathlib import Path

import numpy as np
from argus_cortex.gpu import GpuCoordinator, LeaseCoordinator, coordinator_from_env

MODEL_DIR = Path.home() / ".cache" / "argus_curator" / "models"

_SESSION_CACHE: dict[str, object] = {}
# Threads (a scan, POST /faces) can ask for the same session at once; build it once.
_SESSION_LOCK = threading.Lock()

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


_CUDA = ["CUDAExecutionProvider", "CPUExecutionProvider"]
_cuda_preloaded = False
_cuda_preload_lock = threading.Lock()


def _preload_cuda(ort: object) -> None:
    """Load the CUDA and cuDNN libraries from NVIDIA's pip wheels, once.

    onnxruntime-gpu[cuda,cudnn] installs them under site-packages, off the
    loader's path, and without this the CUDA provider fails to load and every
    session silently falls back to CPU. A no-op when they come from the system
    instead, or with an onnxruntime too old to have ``preload_dlls``.
    """
    global _cuda_preloaded
    with _cuda_preload_lock:
        if _cuda_preloaded:
            return
        _cuda_preloaded = True
        preload = getattr(ort, "preload_dlls", None)
        if preload is not None:
            # A failure leaves system CUDA, if any, to serve the provider.
            with contextlib.suppress(Exception):
                preload()


def providers(device: str) -> list[str]:
    """onnxruntime execution providers for a device hint ("auto" | "cpu" | "cuda").

    CUDA is listed only when onnxruntime actually has ``CUDAExecutionProvider``.
    ``device="cuda"`` on a CPU-only install used to still name CUDA (and take
    the host GPU lease) while sessions fell back to CPU.
    """
    dev = (device or "auto").lower()
    if dev == "cpu":
        return ["CPUExecutionProvider"]
    try:
        import onnxruntime as ort

        available = set(ort.get_available_providers())
    except Exception:
        return ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in available:
        _preload_cuda(ort)
        return list(_CUDA)
    return ["CPUExecutionProvider"]


# Rough VRAM, in MB, a GPU lease asks to be free before curator runs a model:
# "load" for building the model (weights plus the CUDA context and arena), "run"
# for one image (or one Places365 batch) once it is resident.
VRAM_MB: dict[str, dict[str, int]] = {
    "faces": {"load": 1500, "run": 512},  # InsightFace buffalo_l, plus MiVOLO and FairFace
    "scenes": {"load": 400, "run": 512},  # one Places365 batch (up to 16); NVML is a snapshot
    "garments": {"load": 600, "run": 512},
}

_coordinator_lock = threading.Lock()
_coordinator: GpuCoordinator | None = None


def gpu_coordinator() -> GpuCoordinator:
    """The host-wide GPU lease from ``ARGUS_GPU_*`` (argus-cortex), built once. ``none`` unless configured."""
    global _coordinator
    with _coordinator_lock:
        if _coordinator is None:
            _coordinator = coordinator_from_env()
        return _coordinator


def gpu_lease(
    device: str, model: str, phase: str = "run", *, wait_s: float | None = None
) -> contextlib.AbstractContextManager[None]:
    """Hold the host's GPU lease while curator runs *model* on CUDA; a no-op on CPU.

    Every argus process on the host shares the lease (``ARGUS_GPU_LEASE_PATH`` on
    a shared volume), and it waits for ``VRAM_MB[model][phase]`` free across the
    card, so ComfyUI or Ollama on the same GPU are counted. *wait_s* caps the wait
    below the configured timeout, for a caller someone is waiting on (``POST
    /faces``). Raises :class:`argus_cortex.gpu.GpuLeaseTimeout`.
    """
    if "CUDAExecutionProvider" not in providers(device):
        return contextlib.nullcontext()
    coordinator = gpu_coordinator()
    if wait_s is not None and isinstance(coordinator, LeaseCoordinator) and wait_s < coordinator.timeout_s:
        coordinator = LeaseCoordinator(
            lock_path=coordinator.lock_path,
            timeout_s=wait_s,
            poll_s=coordinator.poll_s,
            device=coordinator.device,
            probe=coordinator.probe,
        )
    return coordinator.lease(caller=f"argus-curator:{model}:{phase}", min_vram_mb=VRAM_MB[model][phase])


def load_session(path: Path, providers: list[str], *, missing_hint: str, extra: str = "faces"):
    """Build (and cache) an onnxruntime session for the model at *path*."""
    key = f"{path}:{','.join(providers)}"
    if key in _SESSION_CACHE:
        return _SESSION_CACHE[key]
    with _SESSION_LOCK:
        if key not in _SESSION_CACHE:
            _SESSION_CACHE[key] = _build_session(path, providers, missing_hint=missing_hint, extra=extra)
        return _SESSION_CACHE[key]


def _build_session(path: Path, providers: list[str], *, missing_hint: str, extra: str):
    try:
        import onnxruntime as ort
    except Exception as exc:  # pragma: no cover - exercised only without deps
        raise ModelUnavailable(f"This model requires onnxruntime: pip install argus-curator[{extra}]") from exc
    if not path.is_file():
        raise ModelUnavailable(f"Model not found at {path}. {missing_hint}")
    return ort.InferenceSession(str(path), providers=providers)


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
