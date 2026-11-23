"""Shared onnxruntime plumbing for the optional classifier models.

Face attributes (FairFace, MiVOLO) and image tags (Places365) all run as ONNX
models converted once from their published weights. Each model's location is
process configuration — an env var with a default under :data:`MODEL_DIR` —
never a request field: a model path is a file the server would load.
"""

from __future__ import annotations

import contextlib
import copy
import os
import threading
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from argus_cortex.gpu import (
    GpuCoordinator,
    GpuLeaseRefused,
    GpuLeaseTimeout,
    LeaseCoordinator,
    coordinator_from_env,
    free_vram_mb,
)

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


# ONNX Runtime's CUDA arena, by default, grows in power-of-two steps with no
# ceiling and never hands memory back, so a scan's varying input sizes pushed one
# curator process to ~20 GB while each lease asked for 512 MB, and its own next
# lease then waited on memory it held itself (#189). Every CUDA session curator
# builds (its own, and InsightFace's) takes these options instead: the arena grows
# only by what an allocation asks, up to GPU_MEM_LIMIT_ENV per session.
GPU_MEM_LIMIT_ENV = "ARGUS_CURATOR_GPU_MEM_LIMIT_MB"
DEFAULT_GPU_MEM_LIMIT_MB = 1024


def gpu_mem_limit_mb() -> int:
    """Each CUDA session's arena cap, in MB: ``ARGUS_CURATOR_GPU_MEM_LIMIT_MB``, else the default.

    A value that is not a positive whole number falls back to the default.
    """
    raw = os.environ.get(GPU_MEM_LIMIT_ENV, "").strip()
    try:
        mb = int(raw)
    except ValueError:
        return DEFAULT_GPU_MEM_LIMIT_MB
    return mb if mb > 0 else DEFAULT_GPU_MEM_LIMIT_MB


def cuda_provider_options() -> dict[str, str]:
    """The CUDA execution provider options every curator session is built with."""
    return {
        "arena_extend_strategy": "kSameAsRequested",
        "gpu_mem_limit": str(gpu_mem_limit_mb() * 1024 * 1024),
    }


def session_providers(names: list[str], **extra: str) -> list[str | tuple[str, dict[str, str]]]:
    """*names* (from :func:`providers`) as onnxruntime takes them, CUDA with its options.

    *extra* adds or overrides CUDA options for one caller. :func:`providers`
    itself stays a list of names, which is what every "is this CUDA?" check reads.
    """
    return [(n, {**cuda_provider_options(), **extra}) if n == "CUDAExecutionProvider" else n for n in names]


def process_vram_mb() -> int | None:
    """VRAM this process holds across every GPU, in MB, from NVML; ``None`` when it cannot tell.

    NVML lists processes by host PID, so inside a container (its own PID
    namespace) curator is not found and this is ``None``: run the container
    with ``pid: host`` for the broker to get the figure.
    """
    # cortex owns NVML's one-time init; free_vram_mb is None when NVML is unusable.
    if free_vram_mb(0) is None:
        return None
    import pynvml

    pid = os.getpid()
    total: int | None = None
    try:
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
                if proc.pid == pid and proc.usedGpuMemory is not None:
                    total = (total or 0) + int(proc.usedGpuMemory) // (1024 * 1024)
    except Exception:
        return None
    return total


@contextlib.contextmanager
def _reporting_peak(lease: contextlib.AbstractContextManager[object]) -> Iterator[object]:
    """Hold *lease*, and on release report the VRAM this process holds as its peak.

    The arena keeps what it reserves, so what the process holds at release is
    its high-water mark so far: the figure the broker should plan around.
    """
    with lease as handle:
        try:
            yield handle
        finally:
            report = getattr(handle, "report_peak_vram", None)
            if report is not None:
                mb = process_vram_mb()
                if mb is not None:
                    report(mb)


# Rough VRAM, in MB, a GPU lease asks to be free before curator runs a model:
# "load" for building the model (weights plus the CUDA context and arena), "run"
# for one image (or one Places365 batch) once it is resident.
# Measured on a 3090 with the arena cap (#189), as process VRAM over a 600-image
# scan: faces 900 MB once loaded (CUDA context included) and 1,470 MB at peak;
# garments 436 and 1,176 (its input size follows each image); scenes 322 and 468.
# Each figure is the measurement plus some headroom.
VRAM_MB: dict[str, dict[str, int]] = {
    "faces": {"load": 1000, "run": 600},  # InsightFace buffalo_l, plus MiVOLO and FairFace
    "scenes": {"load": 400, "run": 256},  # one Places365 batch (up to 16); NVML is a snapshot
    "garments": {"load": 500, "run": 768},
    "bodies": {"load": 300, "run": 256},  # a Cheetah pack's person detector alone
    # CLIP ViT-L/14 image tower, fp32, one batch of 8: 1,432 MB once loaded (CUDA
    # context included) and 1,680 MB at peak over 30 batches (3090, 2 GB arena cap).
    "embeddings": {"load": 1600, "run": 384},
    # DNSMOS P.835 (speech scans): 270 MB once loaded (CUDA context included) and
    # 650 MB at peak over 30 windows of 9.01 s, one per pass (3090, 1 GB arena cap).
    "mos": {"load": 300, "run": 384},
    # WeSpeaker ECAPA-TDNN speaker embeddings (speech scans, #187): 298 MB once
    # loaded (CUDA context included) and 420 MB at peak over 30 passes of one
    # clip's 4 windows of 3 s (3090, 1 GB arena cap).
    "speakers": {"load": 350, "run": 192},
    # SpeechBrain VoxLingua107 ECAPA-TDNN language ID (speech scans, #208):
    # 352 MB once loaded (CUDA context included) and 810 MB at peak over 30
    # passes of one clip's 3 windows of 6 s (3090, 1 GB arena cap). 7 ms a
    # clip on the GPU against 700 ms on CPU, so unlike Silero it leases.
    "language": {"load": 400, "run": 512},
}

# What gpu_lease raises when no GPU slot was granted: not a bad image, so a
# phase lets it through and fails the scan (or the request) instead.
GPU_UNAVAILABLE: tuple[type[Exception], ...] = (GpuLeaseTimeout, GpuLeaseRefused)

_coordinator_lock = threading.Lock()
_coordinator: GpuCoordinator | None = None
# wait_s -> (the shared coordinator, its capped copy). Built once per cap, so a
# broker copy keeps its registration and its "unreachable" warning state
# across calls instead of re-registering (and re-warning) on every POST /faces.
_capped_cache: dict[float, tuple[GpuCoordinator, GpuCoordinator]] = {}


def gpu_coordinator() -> GpuCoordinator:
    """The host-wide GPU lease from ``ARGUS_GPU_*`` (argus-cortex), built once. ``none`` unless configured."""
    global _coordinator
    with _coordinator_lock:
        if _coordinator is None:
            _coordinator = coordinator_from_env()
        return _coordinator


def _capped(coordinator: GpuCoordinator, wait_s: float) -> GpuCoordinator:
    """*coordinator* with its wait for a GPU slot cut to *wait_s*, when that is shorter."""
    if isinstance(coordinator, LeaseCoordinator) and wait_s < coordinator.timeout_s:
        return LeaseCoordinator(
            lock_path=coordinator.lock_path,
            timeout_s=wait_s,
            poll_s=coordinator.poll_s,
            device=coordinator.device,
            probe=coordinator.probe,
        )
    if coordinator.name == "broker":
        # The broker client's wait and the local lease it falls back to are both capped.
        fallback = _capped(coordinator.fallback, wait_s)  # type: ignore[attr-defined]
        if wait_s >= coordinator.timeout_s and fallback is coordinator.fallback:  # type: ignore[attr-defined]
            return coordinator
        capped = copy.copy(coordinator)
        capped.timeout_s = min(capped.timeout_s, wait_s)  # type: ignore[attr-defined]
        capped.fallback = fallback  # type: ignore[attr-defined]
        return capped
    return coordinator


def _capped_coordinator(wait_s: float) -> GpuCoordinator:
    """The shared coordinator capped to *wait_s* (:func:`_capped`), built once per cap."""
    coordinator = gpu_coordinator()
    with _coordinator_lock:
        hit = _capped_cache.get(wait_s)
        if hit is None or hit[0] is not coordinator:
            hit = (coordinator, _capped(coordinator, wait_s))
            _capped_cache[wait_s] = hit
        return hit[1]


def gpu_lease(
    device: str, model: str, phase: str = "run", *, wait_s: float | None = None
) -> contextlib.AbstractContextManager[object]:
    """Hold the host's GPU lease while curator runs *model* on CUDA; a no-op on CPU.

    Every argus process on the host shares the lease (``ARGUS_GPU_LEASE_PATH`` on
    a shared volume), and it waits for ``VRAM_MB[model][phase]`` free across the
    card, so ComfyUI or Ollama on the same GPU are counted. With
    ``ARGUS_GPU_BROKER_URL`` set the lease comes from argus-iris instead, as
    the work kind *model* that ``gpu-tenant.yaml`` declares. *wait_s* caps the
    wait below the configured timeout, for a caller someone is waiting on (``POST
    /faces``), and asks the broker for ``interactive`` priority; a scan takes the
    kind's own (``batch``). Raises :class:`argus_cortex.gpu.GpuLeaseTimeout`, or
    :class:`argus_cortex.gpu.GpuLeaseRefused` when the broker refuses the lease.
    """
    if "CUDAExecutionProvider" not in providers(device):
        return contextlib.nullcontext()
    coordinator = gpu_coordinator() if wait_s is None else _capped_coordinator(wait_s)
    return _reporting_peak(
        coordinator.lease(
            caller=f"argus-curator:{model}:{phase}",
            min_vram_mb=VRAM_MB[model][phase],
            kind=model,
            priority="interactive" if wait_s is not None else None,
        )
    )


def load_session(
    path: Path,
    providers: list[str],
    *,
    missing_hint: str,
    extra: str = "faces",
    cuda_options: dict[str, str] | None = None,
    threads: int | None = None,
):
    """Build (and cache) an onnxruntime session for the model at *path*.

    *cuda_options* override :func:`cuda_provider_options` for this one model (a
    larger ``gpu_mem_limit`` for weights past the default cap); the arena stays
    ``kSameAsRequested`` and capped either way. *threads* caps the session's
    intra-op threads, for a tiny model that several scan threads run at once.
    """
    options = cuda_options or {}
    key = f"{path}:{','.join(providers)}:{sorted(options.items())}:{threads}"
    if key in _SESSION_CACHE:
        return _SESSION_CACHE[key]
    with _SESSION_LOCK:
        if key not in _SESSION_CACHE:
            _SESSION_CACHE[key] = _build_session(
                path, providers, missing_hint=missing_hint, extra=extra, cuda_options=options, threads=threads
            )
        return _SESSION_CACHE[key]


def _build_session(
    path: Path,
    providers: list[str],
    *,
    missing_hint: str,
    extra: str,
    cuda_options: dict[str, str] | None = None,
    threads: int | None = None,
):
    try:
        import onnxruntime as ort
    except Exception as exc:  # pragma: no cover - exercised only without deps
        raise ModelUnavailable(f"This model requires onnxruntime: pip install argus-curator[{extra}]") from exc
    if not path.is_file():
        raise ModelUnavailable(f"Model not found at {path}. {missing_hint}")
    provs = session_providers(providers, **(cuda_options or {}))
    if threads is None:
        return ort.InferenceSession(str(path), providers=provs)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=opts, providers=provs)


# Faces per forward pass of an attribute model (MiVOLO, FairFace). A pass used to
# take every face in the image at once, so a crowd shot was one large batch: each
# new batch size grew the CUDA arena (#189), and under its cap a big one fails
# and loses those faces' ages. Measured on a 3090 over 600 images: a MiVOLO pass
# of 8 faces overruns a 1 GB arena, 4 at a time peaks the process at ~1.8-2.1 GB,
# and one at a time at ~1.5 GB, and is also the fastest (one shape, so the
# arena reuses the same blocks every pass).
ATTRIBUTE_BATCH = 1


def run_batched(session, batch: np.ndarray, size: int = ATTRIBUTE_BATCH) -> np.ndarray:
    """*session*'s first output for *batch*, run *size* rows at a time and joined in order."""
    input_name = session.get_inputs()[0].name
    outputs = [session.run(None, {input_name: batch[i : i + size]})[0] for i in range(0, len(batch), size)]
    return np.concatenate(outputs)


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
