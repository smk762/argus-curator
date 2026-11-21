"""Speaker embeddings and speaker clustering for a speech scan (#187).

The counterpart of face clustering for a voice corpus. Each clip that passed
the scan's checks gets a 192-d speaker embedding from WeSpeaker's ECAPA-TDNN
(``ECAPA_TDNN_GLOB_c512-ASTP-emb192``, large-margin fine-tuned, trained on
VoxCeleb2 dev; WeSpeaker's code is Apache-2.0 and its VoxCeleb models CC BY
4.0). ``argus-curator speaker-fetch`` downloads the ONNX export WeSpeaker
publishes, pinned by Hugging Face revision and sha256, and a file that does
not hash to :data:`MODEL_SHA256` is refused at load.

The model reads 80-bin Kaldi filterbanks (25 ms frames every 10 ms, Hamming
window, 16 kHz, 16-bit scale) with the clip's mean taken off each bin;
:func:`fbank` computes them in numpy. A clip is cut into :data:`WINDOWS`
windows of :data:`WINDOW_FRAMES` frames spread evenly across it (a short clip
is tiled up to one window first), and the windows run as one fixed-shape batch
per clip, so the capped CUDA arena never grows with a clip's length. The
clip's embedding is the mean of its windows', unit length.

Clips are clustered by cosine distance: each joins the nearest speaker whose
centroid is within ``SpeechConfig.speaker_threshold`` or starts a new one,
then every clip is reassigned to its nearest centroid a few times. That is
linear in clips times speakers, so a corpus of hundreds of thousands of clips
clusters without a pairwise matrix. Where the corpus names speakers,
:func:`agreement` flags the clips its labels and the clusters disagree on.

Embeddings stay in memory: the scan keeps them on ``ScanSummary.embeddings``
and :class:`argus_curator.store.ScanStore` holds them per scan and drops them
with it. A cluster's centroid is a voiceprint, so it is never stored either.
"""

from __future__ import annotations

import threading
import urllib.request
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from argus_curator.dnsmos import resample
from argus_curator.models import ClipResult, SpeakerCluster
from argus_curator.runtime import (
    ModelUnavailable,
    configured_path,
    gpu_lease,
    load_session,
    onnxruntime_available,
    providers,
    sha256,
)

MODEL_ENV = "ARGUS_CURATOR_SPEAKER_MODEL"
DEFAULT_MODEL_NAME = "wespeaker_voxceleb_ecapa512_lm.onnx"
MODEL_REPO = "Wespeaker/wespeaker-ecapa-tdnn512-LM"
MODEL_REVISION = "a2f3dcb1c8702caccc7a55ceb57f5e8d1842112b"
MODEL_URL = f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/voxceleb_ECAPA512_LM.onnx"
MODEL_SHA256 = "d71b85d9b48058ef68004f04f1b78acebefb9dfcf542e19b976a12a5ad1f10b0"
EMBEDDING_DIM = 192

FEATURE_RATE = 16000
N_MELS = 80
FRAME_SAMPLES = 400  # 25 ms
HOP_SAMPLES = 160  # 10 ms
N_FFT = 512
PREEMPHASIS = 0.97
LOW_HZ = 20.0
# Three seconds of frames per window (Kaldi frames, snip_edges): (48000 - 400) // 160 + 1.
WINDOW_SAMPLES = 3 * FEATURE_RATE
WINDOW_FRAMES = (WINDOW_SAMPLES - FRAME_SAMPLES) // HOP_SAMPLES + 1
# Windows per clip, run as one batch: every pass has the shape (WINDOWS, WINDOW_FRAMES, N_MELS).
WINDOWS = 4

# Reassignment passes after the first clustering pass.
REFINE_PASSES = 3

_verified: dict[tuple[str, int, float], bool] = {}
_verified_lock = threading.Lock()


def model_path() -> Path:
    """Where the speaker-embedding ONNX model is read from."""
    return configured_path(MODEL_ENV, DEFAULT_MODEL_NAME)


def speaker_model_available() -> bool:
    """True when onnxruntime is importable and the speaker model is in place (its hash is checked at load)."""
    return onnxruntime_available() and model_path().is_file()


def problem() -> str | None:
    """Why speaker embeddings cannot run here, else None. Names no path."""
    if not speaker_model_available():
        return (
            "speaker embeddings requested but the model is not in place "
            f"(argus-curator speaker-fetch, or {MODEL_ENV}; needs onnxruntime: argus-curator[speech])"
        )
    return None


def _verify(path: Path) -> None:
    """Refuse a model that is not the pinned file. Hashed once per (path, size, mtime)."""
    try:
        stat = path.stat()
    except OSError as exc:
        raise ModelUnavailable(
            f"speaker model not found. Fetch it with `argus-curator speaker-fetch` or set {MODEL_ENV}."
        ) from exc
    key = (str(path), stat.st_size, stat.st_mtime)
    with _verified_lock:
        if _verified.get(key):
            return
        digest = sha256(path)
        if digest != MODEL_SHA256:
            # Names neither the path nor the URL: this reaches a client.
            raise ModelUnavailable(
                f"the speaker model is not the pinned file (sha256 {digest[:12]}..., see {MODEL_ENV})"
            )
        _verified[key] = True


def get_session(device: str):
    """The speaker-embedding session, built under the GPU lease with the capped arena (runtime.load_session)."""
    path = model_path()
    _verify(path)
    with gpu_lease(device, "speakers", "load"):
        return load_session(
            path,
            providers(device),
            missing_hint=f"Fetch it with `argus-curator speaker-fetch` or set {MODEL_ENV}.",
            extra="speech",
        )


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


def _mel(hz: np.ndarray | float) -> np.ndarray:
    return 1127.0 * np.log(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def _mel_banks() -> np.ndarray:
    """Kaldi's triangular mel filters, ``(N_MELS, N_FFT // 2 + 1)`` (the Nyquist bin weighs nothing)."""
    bins = N_FFT // 2
    mel = _mel(np.arange(bins) * FEATURE_RATE / N_FFT)
    low, high = _mel(LOW_HZ), _mel(FEATURE_RATE / 2)
    step = (high - low) / (N_MELS + 1)
    left = low + np.arange(N_MELS)[:, None] * step
    center, right = left + step, left + 2 * step
    rising = (mel[None] - left) / (center - left)
    falling = (right - mel[None]) / (right - center)
    return np.pad(np.maximum(0.0, np.minimum(rising, falling)), ((0, 0), (0, 1)))


_BANKS = _mel_banks()
_WINDOW = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(FRAME_SAMPLES) / (FRAME_SAMPLES - 1))
_LOG_FLOOR = float(np.finfo(np.float32).eps)


def fbank(audio16k: np.ndarray) -> np.ndarray:
    """``(frames, N_MELS)`` log mel filterbanks of mono 16 kHz audio in [-1, 1], as Kaldi computes them.

    The same as ``torchaudio.compliance.kaldi.fbank(x * 32768, num_mel_bins=80,
    window_type="hamming", dither=0)``: per frame, the DC offset off, a 0.97
    pre-emphasis, a Hamming window, the power spectrum of a 512-point FFT
    through the mel filters, and its log. Audio shorter than a frame gives no frames.
    """
    x = np.asarray(audio16k, dtype=np.float64) * 32768.0
    if len(x) < FRAME_SAMPLES:
        return np.zeros((0, N_MELS), dtype=np.float32)
    n = 1 + (len(x) - FRAME_SAMPLES) // HOP_SAMPLES
    frames = x[np.arange(FRAME_SAMPLES)[None] + HOP_SAMPLES * np.arange(n)[:, None]]
    frames = frames - frames.mean(axis=1, keepdims=True)
    frames = frames - PREEMPHASIS * np.concatenate([frames[:, :1], frames[:, :-1]], axis=1)
    power = np.abs(np.fft.rfft(frames * _WINDOW, N_FFT)) ** 2
    return np.log(np.maximum(power @ _BANKS.T, _LOG_FLOOR)).astype(np.float32)


def windows(mono: np.ndarray, rate: int) -> np.ndarray:
    """The model's input for one clip: ``(WINDOWS, WINDOW_FRAMES, N_MELS)``, mean-normalised.

    The clip is resampled to 16 kHz, tiled up to one window when shorter, and
    its filterbanks normalised by their mean over the clip (WeSpeaker's CMN).
    The windows start evenly across it; a clip of one window repeats it.
    """
    audio = resample(np.asarray(mono, dtype=np.float32), rate, FEATURE_RATE)
    if len(audio) == 0:
        raise ValueError("no audio")
    while len(audio) < WINDOW_SAMPLES:
        audio = np.concatenate([audio, audio])
    feats = fbank(audio)
    feats = feats - feats.mean(axis=0, keepdims=True)
    starts = np.linspace(0, len(feats) - WINDOW_FRAMES, WINDOWS).round().astype(int)
    return np.stack([feats[s : s + WINDOW_FRAMES] for s in starts]).astype(np.float32)


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / max(float(np.linalg.norm(v)), 1e-9)


def predict(session, batch: np.ndarray) -> np.ndarray:
    """The unit speaker embedding for one clip's windows (one fixed-shape pass)."""
    name = session.get_inputs()[0].name
    out = np.asarray(session.run(None, {name: batch})[0], dtype=np.float32)
    return _unit(out.mean(axis=0))


def embed(session, mono: np.ndarray, rate: int, device: str) -> np.ndarray:
    """One clip's unit speaker embedding, under one GPU lease."""
    batch = windows(mono, rate)
    with gpu_lease(device, "speakers"):
        return predict(session, batch)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


def cluster(embeddings: np.ndarray, threshold: float) -> np.ndarray:
    """A speaker label per row of *embeddings* (unit vectors), by cosine distance to centroids.

    First pass in row order: a row joins the nearest centroid within
    *threshold*, else starts a speaker. Then :data:`REFINE_PASSES` rounds of
    reassigning every row to its nearest centroid and recomputing the
    centroids (a speaker left with no rows is dropped). Deterministic for a
    given row order. Labels are 0..k-1, the largest speaker first.
    """
    n = len(embeddings)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    x = np.asarray(embeddings, dtype=np.float32)
    sums = np.zeros((0, x.shape[1]), dtype=np.float64)
    labels = np.empty(n, dtype=np.int64)
    for i, v in enumerate(x):
        if len(sums):
            centroids = sums / np.linalg.norm(sums, axis=1, keepdims=True)
            sims = centroids @ v
            best = int(np.argmax(sims))
            if 1.0 - float(sims[best]) <= threshold:
                labels[i] = best
                sums[best] += v
                continue
        labels[i] = len(sums)
        sums = np.vstack([sums, v[None].astype(np.float64)])
    for _ in range(REFINE_PASSES):
        k = int(labels.max()) + 1
        sums = np.zeros((k, x.shape[1]), dtype=np.float64)
        np.add.at(sums, labels, x)
        live = np.linalg.norm(sums, axis=1) > 0
        centroids = sums[live] / np.linalg.norm(sums[live], axis=1, keepdims=True)
        new = np.argmax(x @ centroids.T.astype(np.float32), axis=1)
        if np.array_equal(new, labels):
            break
        labels = new
    # Largest speaker first; ties by first appearance.
    counts = Counter(labels.tolist())
    first = {lbl: i for i, lbl in reversed(list(enumerate(labels.tolist())))}
    order = sorted(counts, key=lambda lbl: (-counts[lbl], first[lbl]))
    remap = {old: new for new, old in enumerate(order)}
    return np.array([remap[lbl] for lbl in labels.tolist()], dtype=np.int64)


def assign_clusters(clips: Sequence[ClipResult], embeddings: Mapping[str, np.ndarray], threshold: float) -> int:
    """Cluster *clips* that have an embedding, set each one's ``speaker_cluster`` and distance in place.

    Returns how many speakers were found. A clip without an embedding gets
    no cluster.
    """
    for c in clips:
        c.speaker_cluster, c.speaker_distance = None, None
    members = [c for c in clips if c.rel_path in embeddings]
    if not members:
        return 0
    x = np.stack([_unit(embeddings[c.rel_path]) for c in members])
    labels = cluster(x, threshold)
    for lbl in range(int(labels.max()) + 1):
        rows = np.flatnonzero(labels == lbl)
        centroid = _unit(x[rows].mean(axis=0))
        for i in rows:
            members[i].speaker_cluster = f"spk_{lbl + 1}"
            members[i].speaker_distance = round(max(0.0, 1.0 - float(x[i] @ centroid)), 4)
    return int(labels.max()) + 1


def agreement(clips: Sequence[ClipResult]) -> tuple[list[SpeakerCluster], int]:
    """Compare corpus speaker labels with the clusters: the clusters, and how many clips disagree.

    A clip is a mismatch (``speaker_mismatch``, set in place) when it has both
    a label and a cluster and either its label's clips mostly sit in another
    cluster, or its cluster mostly holds another label. A cluster is a
    mismatch when it holds more than one label, or one of its labels has
    clips elsewhere.
    """
    clustered = [c for c in clips if c.speaker_cluster is not None]
    for c in clips:
        c.speaker_mismatch = False
    by_cluster: dict[str, list[ClipResult]] = {}
    for c in clustered:
        by_cluster.setdefault(c.speaker_cluster or "", []).append(c)
    clusters_of_label: dict[str, Counter[str]] = {}
    for c in clustered:
        if c.speaker:
            clusters_of_label.setdefault(c.speaker, Counter())[c.speaker_cluster or ""] += 1

    def top(counter: Counter[str]) -> str:
        return sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

    label_home = {label: top(counts) for label, counts in clusters_of_label.items()}
    out: list[SpeakerCluster] = []
    mismatches = 0
    for cid, members in by_cluster.items():
        labels = Counter(c.speaker for c in members if c.speaker)
        main = top(labels) if labels else None
        split_label = any(len(clusters_of_label[label]) > 1 for label in labels)
        for c in members:
            if c.speaker and (label_home[c.speaker] != cid or c.speaker != main):
                c.speaker_mismatch = True
                mismatches += 1
        out.append(
            SpeakerCluster(
                cluster_id=cid,
                clips=len(members),
                seconds=round(sum(c.duration_s or 0.0 for c in members), 2),
                corpus_speakers=dict(sorted(labels.items())),
                mismatch=len(labels) > 1 or split_label,
            )
        )
    out.sort(key=lambda s: int(s.cluster_id.removeprefix("spk_")))
    return out, mismatches


def fetch_model(dest: Path | None = None) -> Path:
    """Download (once) the pinned speaker model to *dest* (default :func:`model_path`) and verify it."""
    dest = dest or model_path()
    if dest.is_file() and sha256(dest) == MODEL_SHA256:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(MODEL_URL, timeout=120) as resp, tmp.open("wb") as f:  # noqa: S310 - fixed URL, hash-checked
        while chunk := resp.read(1 << 20):
            f.write(chunk)
    digest = sha256(tmp)
    if digest != MODEL_SHA256:
        tmp.unlink(missing_ok=True)
        raise ModelUnavailable(f"download of {MODEL_URL} does not match the pinned sha256 (got {digest})")
    tmp.replace(dest)
    return dest
