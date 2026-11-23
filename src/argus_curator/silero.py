"""Silero VAD: a model voice activity detector for a speech scan (#204).

The energy VAD (:func:`argus_curator.speech.vad`) marks a frame as speech by
its loudness alone, so on a noisy clip it undercounts speech and loud non-speech
counts. With ``SpeechConfig.vad = "silero"`` the speech ratio and the SNR split
read Silero VAD's mask instead (Silero Team, MIT). ``argus-curator vad-fetch``
downloads its ``silero_vad.onnx``, pinned by release commit and sha256, and a
model that does not hash to :data:`MODEL_SHA256` is refused at load.

The model is stateful: it reads 16 kHz audio in :data:`WINDOW` samples (32 ms)
with the previous window's last :data:`CONTEXT` samples before them, carries a
``(2, 1, 128)`` state from one window to the next, and gives each window a
speech probability. Every pass is the same fixed shape, ``(1, WINDOW + CONTEXT)``.

**CPU only, so no GPU lease.** One pass is a fraction of a millisecond on CPU
and launch-bound on CUDA: measured with onnxruntime-gpu 1.32, a pass took
0.12-0.24 ms on one CPU thread and 1.6-2.0 ms on a 3090, about ten times
slower, and a clip is hundreds of passes that each need the last one's state,
so there is no batch to win back. The session is built with the CPU provider alone, so it
never touches the GPU, holds no CUDA arena and takes no lease (no
``gpu-tenant.yaml`` kind), whatever device the scan's other models use. It is
built with one intra-op thread: the scan's clip threads (``max_workers``) are
the parallelism, as Silero's own wrapper does.
"""

from __future__ import annotations

import threading
import urllib.request
from pathlib import Path

import numpy as np

from argus_curator.dnsmos import resample
from argus_curator.runtime import (
    ModelUnavailable,
    configured_path,
    load_session,
    onnxruntime_available,
    sha256,
)

MODEL_ENV = "ARGUS_CURATOR_VAD_MODEL"
DEFAULT_MODEL_NAME = "silero_vad.onnx"
MODEL_RELEASE = "v6.2.3"
MODEL_COMMIT = "5cd7945676eb32225748052e2e6a0580e4686a08"
MODEL_URL = f"https://github.com/snakers4/silero-vad/raw/{MODEL_COMMIT}/src/silero_vad/data/silero_vad.onnx"
MODEL_SHA256 = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"

# The only providers the session is ever built with (see the module docstring).
PROVIDERS = ["CPUExecutionProvider"]

SAMPLE_RATE = 16000
WINDOW = 512  # 32 ms, the model's window at 16 kHz
CONTEXT = 64  # samples of the previous window prepended to each pass
STATE_SHAPE = (2, 1, 128)

# Silero's own defaults (get_speech_timestamps): a window starts speech at
# THRESHOLD and speech ends below NEG_THRESHOLD; gaps under 100 ms are closed
# and speech runs under 250 ms dropped.
THRESHOLD = 0.5
NEG_THRESHOLD = 0.35
MIN_SILENCE_WINDOWS = 3  # 96 ms
MIN_SPEECH_WINDOWS = 8  # 256 ms
# The SNR split leaves this many 10 ms frames either side of Silero's speech out
# of the noise: its mask ends inside a word's decay, and counting that as noise
# read clean clips ~20 dB low (docs/speech.md, "Silero VAD").
NOISE_GUARD_FRAMES = 5

_verified: dict[tuple[str, int, int, int, int, int], bool] = {}
_verified_lock = threading.Lock()


def model_path() -> Path:
    """Where the Silero VAD ONNX model is read from."""
    return configured_path(MODEL_ENV, DEFAULT_MODEL_NAME)


def vad_model_available() -> bool:
    """True when onnxruntime is importable and the Silero model is in place (its hash is checked at load)."""
    return onnxruntime_available() and model_path().is_file()


def problem() -> str | None:
    """Why Silero VAD cannot run here, else None. Names no path."""
    if not vad_model_available():
        return (
            "the Silero VAD was requested but the model is not in place "
            f"(argus-curator vad-fetch, or {MODEL_ENV}; needs onnxruntime: argus-curator[speech])"
        )
    return None


def _verify(path: Path) -> None:
    """Refuse a model that is not the pinned file. Hashed once per (path, device, inode, size, mtime, ctime).

    The inode and ctime are in the key because mtime can be set back from
    userspace: a same-size swap with its mtime restored still changes them.
    """
    try:
        stat = path.stat()
    except OSError as exc:
        raise ModelUnavailable(
            f"Silero VAD model not found. Fetch it with `argus-curator vad-fetch` or set {MODEL_ENV}."
        ) from exc
    key = (str(path), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    with _verified_lock:
        if _verified.get(key):
            return
        digest = sha256(path)
        if digest != MODEL_SHA256:
            # Names neither the path nor the URL: this reaches a client.
            raise ModelUnavailable(
                f"the Silero VAD model is not the pinned file (sha256 {digest[:12]}..., see {MODEL_ENV})"
            )
        _verified[key] = True


def get_session():
    """The Silero VAD session: CPU provider only, one intra-op thread, no GPU lease (see the module docstring)."""
    path = model_path()
    _verify(path)
    return load_session(
        path,
        list(PROVIDERS),
        missing_hint=f"Fetch it with `argus-curator vad-fetch` or set {MODEL_ENV}.",
        extra="speech",
        threads=1,
    )


def probabilities(session, mono: np.ndarray, rate: int) -> np.ndarray:
    """Silero's speech probability for each 32 ms window of *mono* (at *rate*), one fixed-shape pass each."""
    audio = resample(np.asarray(mono, dtype=np.float32), rate, SAMPLE_RATE)
    if len(audio) == 0:
        raise ValueError("no audio")
    n = -(-len(audio) // WINDOW)
    audio = np.pad(audio, (CONTEXT, n * WINDOW - len(audio)))
    state = np.zeros(STATE_SHAPE, dtype=np.float32)
    sr = np.array(SAMPLE_RATE, dtype=np.int64)
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        # The window and the CONTEXT samples before it (zeros before the first).
        x = audio[i * WINDOW : (i + 1) * WINDOW + CONTEXT][None, :]
        prob, state = session.run(None, {"input": x, "state": state, "sr": sr})
        out[i] = float(np.asarray(prob).reshape(-1)[0])
    return out


def _runs(mask: np.ndarray) -> np.ndarray:
    """``(k, 2)`` start and stop indices of the True runs in *mask*."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(np.int8), [0]])))
    return edges.reshape(-1, 2)


def speech_windows(probs: np.ndarray) -> np.ndarray:
    """Boolean speech mask over Silero's windows: hysteresis, short gaps closed, short runs dropped."""
    mask = np.zeros(len(probs), dtype=bool)
    on = False
    for i, p in enumerate(probs):
        if p >= THRESHOLD:
            on = True
        elif p < NEG_THRESHOLD:
            on = False
        mask[i] = on
    for start, stop in _runs(~mask):
        if start > 0 and stop < len(mask) and stop - start < MIN_SILENCE_WINDOWS:
            mask[start:stop] = True
    for start, stop in _runs(mask):
        if stop - start < MIN_SPEECH_WINDOWS:
            mask[start:stop] = False
    return mask


def frame_mask(windows: np.ndarray, n_frames: int, frame_s: float, hop_s: float) -> np.ndarray:
    """*windows* (Silero's mask) on the scan's frame grid: each frame takes the window its centre falls in."""
    if len(windows) == 0:
        return np.zeros(n_frames, dtype=bool)
    centres = np.arange(n_frames) * hop_s + frame_s / 2
    idx = np.minimum((centres * SAMPLE_RATE / WINDOW).astype(int), len(windows) - 1)
    return windows[idx]


def guarded(frames: np.ndarray, guard: int = NOISE_GUARD_FRAMES) -> np.ndarray:
    """*frames* grown by *guard* frames either side of each True run."""
    if guard <= 0 or not frames.any():
        return frames.copy()
    # "full" then sliced: "same" returns max(len(frames), kernel) samples, longer
    # than *frames* when the clip is shorter than the kernel.
    return np.convolve(frames, np.ones(2 * guard + 1), "full")[guard : guard + len(frames)] > 0


def fetch_model(dest: Path | None = None) -> Path:
    """Download (once) the pinned Silero VAD model to *dest* (default :func:`model_path`) and verify it."""
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
