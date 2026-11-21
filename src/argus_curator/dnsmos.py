"""DNSMOS P.835: a non-intrusive MOS predictor for a speech scan (#186).

Microsoft's DNSMOS (Reddy et al., ICASSP 2022; DNS-Challenge repo, code MIT,
other content CC BY 4.0) predicts three 1-5 opinion scores from a clip alone:
the speech (SIG), the background (BAK) and overall quality (OVRL). Its
``sig_bak_ovr.onnx`` takes raw 16 kHz audio in 9.01 s windows, so it runs on
onnxruntime and numpy with no feature extractor. ``argus-curator dnsmos-fetch``
downloads it, pinned by commit and sha256, and a model that does not hash to
:data:`MODEL_SHA256` is refused at load.

Each window is one fixed-shape pass (batch 1, :data:`INPUT_SAMPLES`), so the
capped CUDA arena never grows with a clip's length. A clip is scored over at
most :data:`MAX_WINDOWS` windows, spread evenly, under one GPU lease.
"""

from __future__ import annotations

import threading
import urllib.request
from pathlib import Path

import numpy as np

from argus_curator.models import SpeechMos
from argus_curator.runtime import (
    ModelUnavailable,
    configured_path,
    gpu_lease,
    load_session,
    onnxruntime_available,
    providers,
    sha256,
)

MODEL_ENV = "ARGUS_CURATOR_DNSMOS_MODEL"
DEFAULT_MODEL_NAME = "dnsmos_sig_bak_ovr.onnx"
MODEL_COMMIT = "82f1b17e7776a43eee395d0f45bae8abb700ad00"
MODEL_URL = f"https://github.com/microsoft/DNS-Challenge/raw/{MODEL_COMMIT}/DNSMOS/DNSMOS/sig_bak_ovr.onnx"
MODEL_SHA256 = "269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd"

SAMPLE_RATE = 16000
INPUT_SAMPLES = int(9.01 * SAMPLE_RATE)  # 144160, the model's fixed input
HOP_SAMPLES = SAMPLE_RATE  # upstream's 1 s hop
MAX_WINDOWS = 8

# Upstream's (dnsmos_local.py) polynomial maps from raw output to MOS, non-personalised.
_P_SIG = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
_P_BAK = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
_P_OVR = np.poly1d([-0.06766283, 1.11546468, 0.04602535])

_verified: dict[tuple[str, int, float], bool] = {}
_verified_lock = threading.Lock()


def model_path() -> Path:
    """Where the DNSMOS ONNX model is read from."""
    return configured_path(MODEL_ENV, DEFAULT_MODEL_NAME)


def dnsmos_available() -> bool:
    """True when onnxruntime is importable and the DNSMOS model is in place (its hash is checked at load)."""
    return onnxruntime_available() and model_path().is_file()


def _verify(path: Path) -> None:
    """Refuse a model that is not the pinned file. Hashed once per (path, size, mtime)."""
    try:
        stat = path.stat()
    except OSError as exc:
        raise ModelUnavailable(
            f"DNSMOS model not found. Fetch it with `argus-curator dnsmos-fetch` or set {MODEL_ENV}."
        ) from exc
    key = (str(path), stat.st_size, stat.st_mtime)
    with _verified_lock:
        if _verified.get(key):
            return
        digest = sha256(path)
        if digest != MODEL_SHA256:
            # Names neither the path nor the URL: this reaches a client.
            raise ModelUnavailable(
                f"the DNSMOS model is not the pinned file (sha256 {digest[:12]}..., see {MODEL_ENV})"
            )
        _verified[key] = True


def get_session(device: str):
    """The DNSMOS session, built under the GPU lease with the capped arena (runtime.load_session)."""
    path = model_path()
    _verify(path)
    with gpu_lease(device, "mos", "load"):
        return load_session(
            path,
            providers(device),
            missing_hint=f"Fetch it with `argus-curator dnsmos-fetch` or set {MODEL_ENV}.",
            extra="speech",
        )


def resample(x: np.ndarray, rate: int, target: int) -> np.ndarray:
    """*x* (mono float) at *target* Hz, by FFT (band-limited; the clip's length bounds the cost)."""
    if rate == target or len(x) == 0:
        return np.asarray(x, dtype=np.float32)
    n_out = max(1, round(len(x) * target / rate))
    spectrum = np.fft.rfft(x.astype(np.float64))
    keep = n_out // 2 + 1
    out = np.zeros(keep, dtype=np.complex128)
    m = min(keep, len(spectrum))
    out[:m] = spectrum[:m]
    return (np.fft.irfft(out, n_out) * (n_out / len(x))).astype(np.float32)


def windows(audio16k: np.ndarray) -> np.ndarray:
    """``(n, INPUT_SAMPLES)`` windows as upstream cuts them: the clip tiled up to 9.01 s, then 1 s hops.

    More than :data:`MAX_WINDOWS` hops are thinned to that many, spread evenly.
    """
    audio = np.asarray(audio16k, dtype=np.float32)
    if len(audio) == 0:
        raise ValueError("no audio")
    while len(audio) < INPUT_SAMPLES:
        audio = np.concatenate([audio, audio])
    starts = list(range(0, len(audio) - INPUT_SAMPLES + 1, HOP_SAMPLES))
    if len(starts) > MAX_WINDOWS:
        starts = [starts[round(i)] for i in np.linspace(0, len(starts) - 1, MAX_WINDOWS)]
    return np.stack([audio[s : s + INPUT_SAMPLES] for s in starts])


def predict(session, segments: np.ndarray) -> SpeechMos:
    """Mean SIG/BAK/OVRL over *segments*, one fixed-shape window per pass."""
    name = session.get_inputs()[0].name
    raw = np.concatenate([session.run(None, {name: segments[i : i + 1]})[0] for i in range(len(segments))])
    sig, bak, ovr = (float(np.mean(p(raw[:, i]))) for i, p in enumerate((_P_SIG, _P_BAK, _P_OVR)))
    return SpeechMos(sig=round(sig, 3), bak=round(bak, 3), ovrl=round(ovr, 3))


def score(session, mono: np.ndarray, rate: int, device: str) -> SpeechMos:
    """DNSMOS for one mono clip at *rate*, under one GPU lease."""
    segments = windows(resample(mono, rate, SAMPLE_RATE))
    with gpu_lease(device, "mos"):
        return predict(session, segments)


def fetch_model(dest: Path | None = None) -> Path:
    """Download (once) the pinned DNSMOS model to *dest* (default :func:`model_path`) and verify it."""
    dest = dest or model_path()
    if dest.is_file() and sha256(dest) == MODEL_SHA256:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(MODEL_URL, timeout=60) as resp, tmp.open("wb") as f:  # noqa: S310 - fixed URL, hash-checked
        while chunk := resp.read(1 << 20):
            f.write(chunk)
    digest = sha256(tmp)
    if digest != MODEL_SHA256:
        tmp.unlink(missing_ok=True)
        raise ModelUnavailable(f"download of {MODEL_URL} does not match the pinned sha256 (got {digest})")
    tmp.replace(dest)
    return dest
