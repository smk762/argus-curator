"""Speech scan (#186): which clips of a voice corpus are usable, and which repeat.

The counterpart of :mod:`argus_curator.scanner` for a folder of WAV/FLAC clips
and their transcripts (:mod:`argus_curator.speech_corpus`). Each clip is read
through ``argus_cortex.limits.open_audio`` (the size, channel, rate and
duration caps are checked before a sample is decoded) and gets:

* **Basic checks**, the counterpart of ``min_short_side``: duration, sample
  rate, channels, clipping ratio, and leading or trailing silence.
* **Quality**: an SNR estimate, and with ``SpeechConfig.mos`` a DNSMOS score
  (:mod:`argus_curator.dnsmos`, under the GPU lease).
* **Speech ratio**: the share of the clip a VAD marks as speech: the energy VAD
  by default, or Silero VAD with ``SpeechConfig.vad = "silero"`` (:mod:`argus_curator.silero`).
* **Transcript fit**: the speaking rate (letters and digits per second of the
  clip less its edge silence) and, given an STT engine, the CER between the
  transcript and its pass (the :class:`Transcriber` hook; argus-echo#4).
* **Near-duplicates**: a 64-bit spectrogram fingerprint, the counterpart of pHash.
* **Language labels**: with ``SpeechConfig.language_id``, a spoken language-ID
  model's reading beside the corpus label, and a flag where it disputes it
  (:mod:`argus_curator.language_id`, under the GPU lease).

Every threshold is a prefilter: a clip past one is rejected and names the
check (``ClipResult.reject_reason``, ``failed_checks``, ``reject_detail``). A
clip the corpus marks as under 18 is rejected before it is decoded. Clips are
never modified: curator decides and exports, it does not rewrite sources.

Decoding needs soundfile: ``pip install argus-curator[speech]``.
"""

from __future__ import annotations

import contextlib
import string
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import structlog
from argus_cortex.limits import DEFAULT_SAMPLE_RATES, BadInput, TooLarge, open_audio

from argus_curator import dnsmos, language_id, silero, speakers
from argus_curator.models import (
    MAX_CLIP_CHANNELS,
    SPEECH_CHECKS,
    SPEECH_EXTS,
    ClipResult,
    FaceConfig,
    ScanConfig,
    ScanSummary,
    SpeakerCluster,
    SpeechCheck,
    SpeechConfig,
    SpeechTargetProfile,
)
from argus_curator.runtime import GPU_UNAVAILABLE, ModelUnavailable, ScanCancelled
from argus_curator.scanner import CHUNK_SIZE, CancelFn, ChunkFn, ProgressFn, SaveFn, contained_files, pending_checks
from argus_curator.speech_corpus import Transcript, load_transcripts, under_18_label

logger = structlog.get_logger()

# Rates the decoder takes. A clip at another rate (or with more channels than
# MAX_CLIP_CHANNELS) is "unreadable", with cortex's reason as the detail.
DECODE_RATES: tuple[int, ...] = (*DEFAULT_SAMPLE_RATES, 64000, 88200, 96000)
# Worst case bytes per second of audio a file may hold: 64-bit samples at the
# highest rate on every channel. A file bigger than this for max_duration_s is
# longer than it, and is not read.
_MAX_BYTES_PER_S = max(DECODE_RATES) * MAX_CLIP_CHANNELS * 8

FRAME_S = 0.030
HOP_S = 0.010
# |x| at or past this is a clipped sample.
CLIP_LEVEL = 0.999
# A clip whose peak is under -80 dBFS has nothing in it.
SILENT_PEAK = 1e-4
# The energy VAD: a frame is speech above both the clip's noise floor (its 10th
# percentile frame) plus this margin and its loudest frame plus silence_db.
VAD_MARGIN_DB = 10.0
# Gaps in the VAD mask shorter than this many frames (~200 ms) are closed.
VAD_CLOSE_FRAMES = 21

# The fingerprint: log energies in 32 bands (100-3800 Hz) of the 8 kHz clip,
# trimmed to its loud part and floored 40 dB under its loudest and resampled to 32 time steps; the 8x8 lowest 2D DCT
# coefficients against their median give 64 bits, as pHash does for an image.
_FP_RATE = 8000
_FP_FFT = 256
_FP_BANDS = 32
_FP_STEPS = 32
_FP_SIZE = 8
_FP_FLOOR = 1e-4  # -40 dB under the loudest band energy
_FP_TRIM_DB = -25.0
# Clips whose lengths differ by more than this share are never duplicates.
DUPLICATE_LENGTH_TOLERANCE = 0.10


class Transcriber(Protocol):
    """An STT engine for the CER check: mono float32 samples at *sample_rate* in, text out.

    The hook argus-echo's speech-to-text (argus-echo#4) plugs into; a scan
    without one measures no CER and refuses ``SpeechConfig.max_cer``.
    """

    def transcribe(self, samples: np.ndarray, sample_rate: int, language: str | None = None) -> str: ...


def speech_available() -> bool:
    """True when clips can be decoded (soundfile, from ``argus-curator[speech]``)."""
    return find_spec("soundfile") is not None


def require_speech() -> None:
    """Raise :class:`ModelUnavailable` when clips cannot be decoded."""
    if not speech_available():
        raise ModelUnavailable("speech scans need soundfile: pip install argus-curator[speech]")


def requested_checks(cfg: SpeechConfig, transcriber: Transcriber | None = None) -> list[str]:
    """The slow checks a speech scan runs after its first pass (``pending_checks`` in its events)."""
    checks = (
        ("mos", cfg.mos),
        ("cer", transcriber is not None),
        ("language", cfg.language_id),
        ("speakers", cfg.speakers),
    )
    return [name for name, on in checks if on]


def problem(cfg: SpeechConfig, transcriber: Transcriber | None = None) -> str | None:
    """Why *cfg* cannot run here, else None. Names no path or URL."""
    if not speech_available():
        return "speech scans need soundfile: pip install argus-curator[speech]"
    if cfg.max_cer is not None and transcriber is None:
        return "max_cer needs a speech-to-text engine (argus-echo#4), and none is configured"
    if cfg.mos and not dnsmos.dnsmos_available():
        return f"DNSMOS requested but the model is not in place (argus-curator dnsmos-fetch, or {dnsmos.MODEL_ENV})"
    if cfg.vad == "silero" and (why := silero.problem()) is not None:
        return why
    if cfg.language_id and (why := language_id.problem()) is not None:
        return why
    if cfg.speakers:
        return speakers.problem()
    return None


# ---------------------------------------------------------------------------
# Measurements (numpy only; each takes decoded samples)
# ---------------------------------------------------------------------------


def frame_db(mono: np.ndarray, rate: int) -> np.ndarray:
    """Each 30 ms frame's power in dBFS, every 10 ms."""
    n = max(1, round(FRAME_S * rate))
    hop = max(1, round(HOP_S * rate))
    x = np.asarray(mono, dtype=np.float32)
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    frames = np.lib.stride_tricks.sliding_window_view(x, n)[::hop]
    power = np.einsum("ij,ij->i", frames, frames, dtype=np.float64) / n
    return 10.0 * np.log10(power + 1e-12)


def clipping_ratio(samples: np.ndarray) -> float:
    """Share of samples (every channel) at or past :data:`CLIP_LEVEL`."""
    return float(np.mean(np.abs(samples) >= CLIP_LEVEL))


def edge_silence(db: np.ndarray, silence_db: float) -> tuple[int, int]:
    """Frames of silence before the first and after the last frame within *silence_db* of the loudest."""
    loud = np.flatnonzero(db > db.max() + silence_db)
    if len(loud) == 0:
        return len(db), 0
    return int(loud[0]), int(len(db) - 1 - loud[-1])


def vad(db: np.ndarray, silence_db: float) -> np.ndarray:
    """Boolean speech mask over the frames of *db* (an energy VAD with short gaps closed)."""
    threshold = max(float(np.percentile(db, 10)) + VAD_MARGIN_DB, float(db.max()) + silence_db)
    mask = db > threshold
    if len(mask) >= VAD_CLOSE_FRAMES:
        kernel = np.ones(VAD_CLOSE_FRAMES)
        dilated = np.convolve(mask, kernel, "same") > 0
        mask = ~(np.convolve(~dilated, kernel, "same") > 0)
    return mask


def snr_db(db: np.ndarray, mask: np.ndarray, not_noise: np.ndarray | None = None) -> float | None:
    """SNR of the speech frames over the rest (or the quietest tenth when the VAD left too few).

    *not_noise* (default *mask*) marks the frames kept out of the noise side:
    a model VAD's speech plus what it is unsure of (:func:`speech_mask`).
    """
    power = 10.0 ** (db / 10.0)
    speech, noise = power[mask], power[~(mask if not_noise is None else not_noise)]
    if speech.size == 0:
        return None
    noise_p = float(noise.mean()) if noise.size >= 5 else float(np.percentile(power, 10))
    noise_p = max(noise_p, 1e-12)
    ratio = max(float(speech.mean()) - noise_p, 1e-12) / noise_p
    return float(np.clip(10.0 * np.log10(ratio), -20.0, 100.0))


def spoken_chars(text: str) -> int:
    """Letters and digits in *text*: what a speaking rate counts."""
    return sum(ch.isalnum() for ch in text)


_PUNCT = str.maketrans("", "", string.punctuation + "‘’“”«»¿¡")


def _normalise(text: str) -> str:
    return " ".join(text.lower().translate(_PUNCT).split())


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate of *hypothesis* against *reference*, case and punctuation folded.

    Levenshtein distance over characters divided by the reference's length (an
    empty reference scores 0 against an empty hypothesis, else 1).
    """
    ref, hyp = _normalise(reference), _normalise(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        cur = [i]
        for j, hc in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1] / len(ref)


def _resample(x: np.ndarray, rate: int, target: int) -> np.ndarray:
    return dnsmos.resample(x, rate, target)


def _dct_matrix(n: int) -> np.ndarray:
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    return np.cos(np.pi * (2 * i + 1) * k / (2 * n))


_DCT = _dct_matrix(_FP_STEPS)


def fingerprint(mono: np.ndarray, rate: int) -> str:
    """64-bit fingerprint (16 hex digits) of a clip's spectrogram; near-duplicates are a few bits apart.

    The clip is cut to the frames from the first to the last within 25 dB of
    its loudest, so padding or a noise floor at the edges does not shift it,
    and band energies more than 40 dB under the loudest are floored, so a copy
    with a different noise floor in its quiet parts still matches.
    """
    lead, trail = edge_silence(frame_db(mono, rate), _FP_TRIM_DB)
    hop = max(1, round(HOP_S * rate))
    start = lead * hop
    stop = max(len(mono) - trail * hop, start + 1)
    x = _resample(np.asarray(mono[start:stop], dtype=np.float32), rate, _FP_RATE)
    if len(x) < _FP_FFT:
        x = np.pad(x, (0, _FP_FFT - len(x)))
    frames = np.lib.stride_tricks.sliding_window_view(x, _FP_FFT)[:: _FP_FFT // 2] * np.hanning(_FP_FFT)
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    edges = np.round(np.geomspace(100, 3800, _FP_BANDS + 1) / (_FP_RATE / _FP_FFT)).astype(int)
    bands = np.stack(
        [power[:, lo : max(hi, lo + 1)].sum(axis=1) for lo, hi in zip(edges[:-1], edges[1:], strict=True)],
        axis=1,
    )
    log = np.log(bands + max(float(bands.max()), 1e-10) * _FP_FLOOR)
    pos = np.linspace(0, len(log), _FP_STEPS + 1)
    steps = np.stack(
        [log[int(a) : max(int(b), int(a) + 1)].mean(axis=0) for a, b in zip(pos[:-1], pos[1:], strict=True)]
    )
    coeffs = (_DCT @ steps @ _DCT.T)[:_FP_SIZE, :_FP_SIZE]
    bits = (coeffs > np.median(coeffs)).flatten()
    return f"{int(''.join('1' if b else '0' for b in bits), 2):016x}"


def hamming(a: str, b: str) -> int:
    """Bits that differ between two hex fingerprints."""
    return (int(a, 16) ^ int(b, 16)).bit_count()


# ---------------------------------------------------------------------------
# One clip
# ---------------------------------------------------------------------------


class _Rejected(Exception):
    def __init__(self, check: SpeechCheck, detail: str) -> None:
        super().__init__(detail)
        self.check, self.detail = check, detail


def decode(path: Path, cfg: SpeechConfig):
    """The clip at *path* as cortex ``Audio``, through ``limits.open_audio``; :class:`_Rejected` when it cannot be."""
    cap = int(cfg.max_duration_s * _MAX_BYTES_PER_S) + (1 << 20)
    try:
        with path.open("rb") as fh:
            data = fh.read(cap + 1)
    except OSError as exc:
        raise _Rejected("unreadable", f"could not be read: {exc.strerror or exc}") from exc
    if len(data) > cap:
        raise _Rejected("duration", f"the file is longer than max_duration_s {cfg.max_duration_s:g} s allows")
    try:
        return open_audio(
            data, max_seconds=cfg.max_duration_s, max_channels=MAX_CLIP_CHANNELS, allowed_rates=DECODE_RATES
        )
    except TooLarge as exc:
        raise _Rejected("duration", f"longer than max_duration_s {cfg.max_duration_s:g} s") from exc
    except BadInput as exc:
        raise _Rejected("unreadable", str(exc)) from exc


def _settle(row: ClipResult, failed: dict[SpeechCheck, str]) -> None:
    """Add *failed* (check -> what it measured) to *row*'s failures, kept in SPEECH_CHECKS order."""
    known: dict[SpeechCheck, str] = dict.fromkeys(row.failed_checks, "")
    if row.reject_reason is not None:
        known[row.reject_reason] = row.reject_detail or ""
    for check, detail in failed.items():
        known.setdefault(check, detail)
    ordered = [c for c in SPEECH_CHECKS if c in known]
    row.failed_checks = ordered
    row.passed = not ordered
    row.reject_reason = ordered[0] if ordered else None
    row.reject_detail = (known[ordered[0]] or None) if ordered else None


def speech_mask(
    mono: np.ndarray, rate: int, db: np.ndarray, cfg: SpeechConfig, vad_session: Any = None
) -> tuple[np.ndarray, np.ndarray]:
    """``(speech, not_noise)`` masks over the frames of *db*, from the VAD *cfg* names.

    ``speech`` is what the speech ratio counts and the SNR's speech side;
    ``not_noise`` is kept off the SNR's noise side. For the energy VAD they are
    the same. For Silero (run on *vad_session*) ``not_noise`` adds every window
    it gives at least :data:`silero.NEG_THRESHOLD` (a word too short to count
    as speech is not noise either) and a guard band around them.
    """
    if cfg.vad == "silero":
        if vad_session is None:
            raise ModelUnavailable("the Silero VAD was requested but no session was given")
        probs = silero.probabilities(vad_session, mono, rate)
        speech = silero.speech_windows(probs)
        return (
            silero.frame_mask(speech, len(db), FRAME_S, HOP_S),
            silero.guarded(silero.frame_mask(speech | (probs >= silero.NEG_THRESHOLD), len(db), FRAME_S, HOP_S)),
        )
    mask = vad(db, cfg.silence_db)
    return mask, mask


def measure_clip(
    rel: str, path: Path, transcript: Transcript | None, cfg: SpeechConfig, vad_session: Any = None
) -> ClipResult:
    """Decode one clip, measure it, and apply every first-pass threshold of *cfg*.

    *vad_session* is the Silero VAD session when ``cfg.vad`` is ``"silero"``.
    """
    row = ClipResult(rel_path=rel, abs_path=str(path))
    if transcript is not None:
        row.transcript = transcript.text or None
        row.transcript_source = transcript.source if transcript.text else None
        row.speaker, row.accent = transcript.speaker, transcript.accent
        row.language, row.age = transcript.language, transcript.age
        row.gender = transcript.gender
    failed: dict[SpeechCheck, str] = {}
    if under_18_label(row.age):
        # Never decoded: nothing measured on it could make it usable.
        _settle(row, {"corpus_under_18": f"the corpus marks the speaker's age as {row.age!r}"})
        return row
    try:
        audio = decode(path, cfg)
    except _Rejected as exc:
        _settle(row, {exc.check: exc.detail})
        return row

    samples = audio.samples
    rate = int(audio.sample_rate)
    row.sample_rate, row.channels = rate, int(samples.shape[1])
    row.duration_s = round(len(samples) / rate, 3)
    if row.duration_s < cfg.min_duration_s:
        failed.setdefault("duration", f"{row.duration_s:g} s is under min_duration_s {cfg.min_duration_s:g}")
    if rate < cfg.min_sample_rate:
        failed.setdefault("sample_rate", f"{rate} Hz is under min_sample_rate {cfg.min_sample_rate}")
    if row.channels > cfg.max_channels:
        failed.setdefault("channels", f"{row.channels} channels; max_channels is {cfg.max_channels}")

    if float(np.max(np.abs(samples))) < SILENT_PEAK:
        failed.setdefault("silent", "the clip peaks under -80 dBFS")
        _settle(row, failed)
        return row

    row.clipping_ratio = round(clipping_ratio(samples), 6)
    if cfg.max_clipping_ratio is not None and row.clipping_ratio > cfg.max_clipping_ratio:
        failed.setdefault(
            "clipping", f"{row.clipping_ratio:g} of samples clip; max_clipping_ratio is {cfg.max_clipping_ratio:g}"
        )

    mono = samples.mean(axis=1, dtype=np.float32)
    db = frame_db(mono, rate)
    lead, trail = edge_silence(db, cfg.silence_db)
    row.leading_silence_s, row.trailing_silence_s = round(lead * HOP_S, 3), round(trail * HOP_S, 3)
    if cfg.max_leading_silence_s is not None and row.leading_silence_s > cfg.max_leading_silence_s:
        failed.setdefault(
            "leading_silence", f"{row.leading_silence_s:g} s; max_leading_silence_s is {cfg.max_leading_silence_s:g}"
        )
    if cfg.max_trailing_silence_s is not None and row.trailing_silence_s > cfg.max_trailing_silence_s:
        failed.setdefault(
            "trailing_silence",
            f"{row.trailing_silence_s:g} s; max_trailing_silence_s is {cfg.max_trailing_silence_s:g}",
        )

    try:
        mask, not_noise = speech_mask(mono, rate, db, cfg, vad_session)
    except Exception as exc:
        # Neither measured: the energy VAD is not swapped in behind the request's back.
        row.vad_error = str(exc) or type(exc).__name__
        logger.warning("speech_vad_failed", rel_path=rel, vad=cfg.vad, error=row.vad_error)
    else:
        row.vad = cfg.vad
        row.speech_ratio = round(float(mask.mean()), 4)
        snr = snr_db(db, mask, not_noise)
        row.snr_db = round(snr, 2) if snr is not None else None

    if row.transcript is None:
        if cfg.require_transcript:
            failed.setdefault("transcript", "no transcript (a .txt sidecar or a corpus metadata row)")
    else:
        span = max(row.duration_s - row.leading_silence_s - row.trailing_silence_s, HOP_S)
        row.chars_per_s = round(spoken_chars(row.transcript) / span, 2)
        if cfg.min_chars_per_s is not None and row.chars_per_s < cfg.min_chars_per_s:
            failed.setdefault(
                "speaking_rate", f"{row.chars_per_s:g} chars/s is under min_chars_per_s {cfg.min_chars_per_s:g}"
            )
        if cfg.max_chars_per_s is not None and row.chars_per_s > cfg.max_chars_per_s:
            failed.setdefault(
                "speaking_rate", f"{row.chars_per_s:g} chars/s is over max_chars_per_s {cfg.max_chars_per_s:g}"
            )
    no_vad = f"the {cfg.vad} VAD could not read it" if row.vad_error is not None else None
    if cfg.min_snr_db is not None and (row.snr_db is None or row.snr_db < cfg.min_snr_db):
        failed.setdefault(
            "snr",
            f"{no_vad}, and min_snr_db is set"
            if no_vad
            else f"SNR {row.snr_db} dB is under min_snr_db {cfg.min_snr_db:g}",
        )
    if cfg.min_speech_ratio is not None:
        if row.speech_ratio is None:
            failed.setdefault("speech_ratio", f"{no_vad or 'no speech ratio'}, and min_speech_ratio is set")
        elif row.speech_ratio < cfg.min_speech_ratio:
            failed.setdefault(
                "speech_ratio", f"{row.speech_ratio:g} is under min_speech_ratio {cfg.min_speech_ratio:g}"
            )

    row.fingerprint = fingerprint(mono, rate)
    _settle(row, failed)
    return row


def finalize_score(row: ClipResult, cfg: SpeechConfig) -> None:
    """The composite score (0-1): the measured terms of MOS, SNR and speech ratio, weighted."""
    terms: dict[str, tuple[float, float]] = {}
    if row.mos is not None:
        terms["mos"] = (min(max((row.mos.ovrl - 1.0) / 4.0, 0.0), 1.0), cfg.weight_mos)
    if row.snr_db is not None:
        terms["snr"] = (min(max(row.snr_db / cfg.snr_ref_db, 0.0), 1.0), cfg.weight_snr)
    if row.speech_ratio is not None:
        terms["speech_ratio"] = (row.speech_ratio, cfg.weight_speech_ratio)
    total = sum(w for _v, w in terms.values())
    row.score_breakdown = {k: round(v, 4) for k, (v, _w) in terms.items()}
    row.score = round(sum(v * w for v, w in terms.values()) / total, 4) if total > 0 else 0.0


def mark_duplicates(rows: list[ClipResult], distance: int) -> int:
    """Tag near-duplicate passing clips in place, the best-scoring kept as representative.

    Returns the number of groups with more than one clip. A negative *distance* disables grouping.
    """
    for r in rows:
        r.is_duplicate, r.duplicate_of = False, None
    if distance >= 0:
        reps: list[ClipResult] = []
        rep_fp = np.zeros(0, dtype=np.uint64)
        rep_len = np.zeros(0, dtype=np.float64)
        for r in sorted((r for r in rows if r.passed and r.fingerprint), key=lambda r: (-r.score, r.rel_path)):
            fp = np.uint64(int(r.fingerprint, 16))
            length = float(r.duration_s or 0.0)
            if len(reps):
                bits = np.unpackbits((rep_fp ^ fp).view(np.uint8)).reshape(-1, 64).sum(axis=1)
                close = np.abs(rep_len - length) <= DUPLICATE_LENGTH_TOLERANCE * np.maximum(rep_len, length)
                hits = np.flatnonzero((bits <= distance) & close)
                if len(hits):
                    r.is_duplicate, r.duplicate_of = True, reps[int(hits[0])].rel_path
                    continue
            reps.append(r)
            rep_fp = np.append(rep_fp, fp)
            rep_len = np.append(rep_len, length)
    rep_of = {r.rel_path: r.duplicate_of if r.is_duplicate and r.duplicate_of else r.rel_path for r in rows}
    group: dict[str, int] = {}
    size: dict[str, int] = {}
    for rep in rep_of.values():
        size[rep] = size.get(rep, 0) + 1
        group.setdefault(rep, len(group) + 1)
    for r in rows:
        rep = rep_of[r.rel_path]
        r.similar_group, r.group_size, r.is_representative = group[rep], size[rep], rep == r.rel_path
    return sum(1 for n in size.values() if n > 1)


def drop_clips(summary: ScanSummary, gone: set[str] | list[str]) -> None:
    """Remove *gone* clips from a finished speech scan and bring what is derived from them up to date.

    The speech counterpart of :func:`argus_curator.scanner.drop_results`
    (a move export takes its clips out of the source). Near-duplicate groups
    are rebuilt, and the speaker clusters recounted from the clips that stay
    (no clip changes cluster). The store drops the gone clips' embeddings.
    """
    gone = set(gone)
    clips = [c for c in summary.clips if c.rel_path not in gone]
    cfg = summary.speech_config or SpeechConfig()
    summary.similar_clusters = mark_duplicates(clips, cfg.duplicate_distance)
    if summary.speaker_clusters:
        summary.speaker_clusters, summary.speaker_mismatches = speakers.agreement(clips)
    summary.language_mismatches = language_id.count_mismatches(clips)
    for rel in gone:
        summary.embeddings.pop(rel, None)
        summary.voice_windows.pop(rel, None)
    passed = [c for c in clips if c.passed]
    reasons: dict[str, int] = {}
    for c in clips:
        if c.reject_reason is not None:
            reasons[c.reject_reason] = reasons.get(c.reject_reason, 0) + 1
    summary.clips = clips
    summary.total = len(clips)
    summary.passed = len(passed)
    summary.rejected = len(clips) - len(passed)
    summary.duplicates = sum(1 for c in passed if c.is_duplicate)
    summary.reject_reasons = reasons
    summary.returned = len(clips)


# ---------------------------------------------------------------------------
# The scan
# ---------------------------------------------------------------------------


def _call(sink: Callable[[dict[str, Any]], None] | None, payload: dict[str, Any]) -> None:
    if sink is None:
        return
    # Never let a flaky consumer (a disconnected SSE client) break a scan.
    with contextlib.suppress(Exception):  # pragma: no cover
        sink(payload)


def scan_speech_folder(
    folder: str | Path,
    profile: object | None = None,
    cfg: SpeechConfig | None = None,
    *,
    transcriber: Transcriber | None = None,
    scan_id: str | None = None,
    progress: ProgressFn | None = None,
    chunk: ChunkFn | None = None,
    save: SaveFn | None = None,
    cancelled: CancelFn | None = None,
) -> ScanSummary:
    """Scan the WAV/FLAC clips under *folder* (recursively) and score each.

    Events and saves work as an image scan's (:func:`argus_curator.scanner.scan_items`):
    *progress* gets ``{phase, done, total}``, *chunk* the clips finished since
    the last one (``clips``, with ``modality: "speech"``), *save* a snapshot
    now and then, and *cancelled* stops the scan between clips. Phases:
    ``collecting``, ``scoring`` (decode, measure, first-pass thresholds),
    ``mos`` (with ``cfg.mos``), ``cer`` (with a *transcriber*), ``language``
    (with ``cfg.language_id``) and ``speakers`` (with ``cfg.speakers``:
    embeddings, then the clustering).
    """
    # A speech scan takes a speech profile only: an image one is refused here.
    profile = SpeechTargetProfile.model_validate(profile) if profile is not None else SpeechTargetProfile()
    cfg = cfg or SpeechConfig()
    if (why := problem(cfg, transcriber)) is not None:
        raise ModelUnavailable(why)
    root = Path(folder)
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")
    scan_id = scan_id or uuid.uuid4().hex
    folder_str = str(root.resolve())

    def check_cancelled() -> None:
        if cancelled is not None and cancelled():
            raise ScanCancelled("scan cancelled")

    _call(progress, {"phase": "collecting", "done": 0, "total": 0})
    clips = contained_files(root, SPEECH_EXTS)
    transcripts = load_transcripts(root, [rel for rel, _ in clips], cfg.transcripts)
    total = len(clips)
    logger.info("speech_scan_start", folder=folder_str, count=total)
    _call(progress, {"phase": "collecting", "done": total, "total": total})

    requested = requested_checks(cfg, transcriber)
    completed: set[str] = set()
    rows: dict[str, ClipResult] = {}
    order = [rel for rel, _ in clips]
    lock = threading.Lock()
    saved_at = 0.0
    multi = 0

    def snapshot() -> ScanSummary:
        ordered = [rows[rel] for rel in order if rel in rows]
        passed = [r for r in ordered if r.passed]
        reasons: dict[str, int] = {}
        for r in ordered:
            if r.reject_reason is not None:
                reasons[r.reject_reason] = reasons.get(r.reject_reason, 0) + 1
        return ScanSummary(
            scan_id=scan_id,
            folder=folder_str,
            target_profile=profile,
            config=ScanConfig(),
            faces_config=FaceConfig(),
            speech_config=cfg,
            total=total,
            passed=len(passed),
            rejected=len(ordered) - len(passed),
            duplicates=sum(1 for r in passed if r.is_duplicate),
            similar_clusters=multi,
            reject_reasons=reasons,
            clips=ordered,
            returned=len(ordered),
        )

    def maybe_save(force: bool = False) -> None:
        nonlocal saved_at
        if save is None:
            return
        now = time.monotonic()
        if not force and saved_at and now - saved_at < 2:
            return
        try:
            save(snapshot())
            saved_at = now
        except Exception as exc:
            logger.warning("scan_snapshot_save_failed", scan_id=scan_id, error=str(exc))

    def deliver(phase: str, done: int, phase_total: int, batch: list[ClipResult], finished: str | None = None) -> None:
        check_cancelled()
        with lock:
            if finished is not None:
                completed.add(finished)
            if not batch:
                return
            _call(
                chunk,
                {
                    "scan_id": scan_id,
                    "folder": folder_str,
                    "modality": "speech",
                    "phase": phase,
                    "done": done,
                    "total": phase_total,
                    "pending_checks": pending_checks(requested, completed),
                    "clips": [r.model_dump(mode="json") for r in batch],
                },
            )
            maybe_save(force=phase_total > 0 and done >= phase_total)

    def emit(phase: str, done: int, phase_total: int) -> None:
        check_cancelled()
        _call(
            progress,
            {
                "phase": phase,
                "done": done,
                "total": phase_total,
                "pending_checks": pending_checks(requested, completed),
            },
        )

    # Phase 1: decode, measure and the first-pass thresholds. Silero VAD, when
    # asked for, runs here on CPU (no GPU lease: see argus_curator.silero).
    vad_session = silero.get_session() if cfg.vad == "silero" else None
    emit("scoring", 0, total)
    done = 0
    batch: list[ClipResult] = []
    with ThreadPoolExecutor(max_workers=cfg.max_workers) as pool:
        futures = {
            pool.submit(measure_clip, rel, path, transcripts.get(rel), cfg, vad_session): rel for rel, path in clips
        }
        try:
            for fut in as_completed(futures):
                row = fut.result()
                finalize_score(row, cfg)
                rows[row.rel_path] = row
                done += 1
                batch.append(row)
                if done == total or done % 5 == 0:
                    emit("scoring", done, total)
                if done == total or len(batch) >= CHUNK_SIZE:
                    deliver("scoring", done, total, batch)
                    batch = []
        except BaseException:
            for other in futures:
                other.cancel()
            raise

    path_of = dict(clips)

    def mono_of(row: ClipResult) -> tuple[np.ndarray, int] | None:
        try:
            audio = decode(path_of[row.rel_path], cfg)
        except _Rejected:
            return None
        return audio.samples.mean(axis=1, dtype=np.float32), int(audio.sample_rate)

    # Phase 2: DNSMOS on the clips that passed, one at a time under the GPU lease.
    if cfg.mos:
        session = dnsmos.get_session(cfg.mos_device)
        pending = [rows[rel] for rel in order if rel in rows and rows[rel].passed]
        emit("mos", 0, len(pending))
        batch = []
        for i, row in enumerate(pending, 1):
            check_cancelled()
            decoded = mono_of(row)
            try:
                if decoded is None:
                    raise ValueError("the clip could not be read again")
                row.mos = dnsmos.score(session, decoded[0], decoded[1], cfg.mos_device)
            except GPU_UNAVAILABLE:
                raise  # no GPU slot is not a bad clip: fail the scan
            except Exception as exc:
                row.mos_error = str(exc) or type(exc).__name__
                logger.warning("speech_mos_failed", rel_path=row.rel_path, error=row.mos_error)
            if cfg.min_mos is not None:
                if row.mos is None:
                    _settle(row, {"mos": "DNSMOS could not score it, and min_mos is set"})
                elif row.mos.ovrl < cfg.min_mos:
                    _settle(row, {"mos": f"DNSMOS OVRL {row.mos.ovrl:g} is under min_mos {cfg.min_mos:g}"})
            finalize_score(row, cfg)
            batch.append(row)
            if i == len(pending) or len(batch) >= CHUNK_SIZE:
                emit("mos", i, len(pending))
                deliver("mos", i, len(pending), batch, finished="mos" if i == len(pending) else None)
                batch = []
        with lock:
            completed.add("mos")

    # Phase 3: the CER hook (argus-echo#4), on the clips still passing that have a transcript.
    if transcriber is not None:
        pending = [rows[rel] for rel in order if rel in rows and rows[rel].passed and rows[rel].transcript]
        emit("cer", 0, len(pending))
        batch = []
        for i, row in enumerate(pending, 1):
            check_cancelled()
            decoded = mono_of(row)
            try:
                if decoded is None:
                    raise ValueError("the clip could not be read again")
                row.stt_transcript = transcriber.transcribe(decoded[0], decoded[1], row.language)
                row.cer = round(cer(row.transcript or "", row.stt_transcript), 4)
            except GPU_UNAVAILABLE:
                raise  # no GPU slot is not a bad clip: fail the scan
            except Exception as exc:
                logger.warning("speech_stt_failed", rel_path=row.rel_path, error=str(exc))
            if cfg.max_cer is not None:
                if row.cer is None:
                    _settle(row, {"cer": "no STT pass, and max_cer is set"})
                elif row.cer > cfg.max_cer:
                    _settle(row, {"cer": f"CER {row.cer:g} is over max_cer {cfg.max_cer:g}"})
            batch.append(row)
            if i == len(pending) or len(batch) >= CHUNK_SIZE:
                emit("cer", i, len(pending))
                deliver("cer", i, len(pending), batch, finished="cer" if i == len(pending) else None)
                batch = []
        with lock:
            completed.add("cer")

    # Phase 4: language ID on the clips still passing (#208), one clip's
    # fixed-size windows per pass under the GPU lease. A dispute is flagged;
    # it rejects the clip only with reject_language_mismatch.
    if cfg.language_id:
        session = language_id.get_session(cfg.language_id_device)
        pending = [rows[rel] for rel in order if rel in rows and rows[rel].passed]
        emit("language", 0, len(pending))
        batch = []
        for i, row in enumerate(pending, 1):
            check_cancelled()
            decoded = mono_of(row)
            try:
                if decoded is None:
                    raise ValueError("the clip could not be read again")
                probs = language_id.identify(session, decoded[0], decoded[1], cfg.language_id_device)
                language_id.judge(row, probs, cfg.language_dispute_below)
            except GPU_UNAVAILABLE:
                raise
            except Exception as exc:
                row.language_error = str(exc) or type(exc).__name__
                logger.warning("speech_language_id_failed", rel_path=row.rel_path, error=row.language_error)
            if cfg.reject_language_mismatch and row.language_mismatch:
                _settle(
                    row,
                    {
                        "language": f"the language-ID model gives the corpus label {row.language!r} "
                        f"p={row.language_label_confidence:g} (it hears {row.language_predicted!r}, "
                        f"p={row.language_confidence:g})"
                    },
                )
            batch.append(row)
            if i == len(pending) or len(batch) >= CHUNK_SIZE:
                emit("language", i, len(pending))
                deliver("language", i, len(pending), batch, finished="language" if i == len(pending) else None)
                batch = []
        with lock:
            completed.add("language")

    # Phase 5: speaker embeddings on the clips still passing, one clip per pass
    # under the GPU lease, then the clustering. With cfg.second_voice, each
    # clip's speech (the scan's VAD) is also embedded in short windows (#217),
    # in fixed-shape passes under one lease. Both stay in memory.
    embeddings: dict[str, np.ndarray] = {}
    voice_windows: dict[str, np.ndarray] = {}
    if cfg.speakers:
        session = speakers.get_session(cfg.speaker_device)
        pending = [rows[rel] for rel in order if rel in rows and rows[rel].passed]
        emit("speakers", 0, len(pending))
        for i, row in enumerate(pending, 1):
            check_cancelled()
            decoded = mono_of(row)
            try:
                if decoded is None:
                    raise ValueError("the clip could not be read again")
                embeddings[row.rel_path] = speakers.embed(session, decoded[0], decoded[1], cfg.speaker_device)
            except GPU_UNAVAILABLE:
                raise
            except Exception as exc:
                row.speaker_error = str(exc) or type(exc).__name__
                logger.warning("speech_speaker_failed", rel_path=row.rel_path, error=row.speaker_error)
            if cfg.second_voice and decoded is not None and row.rel_path in embeddings:
                try:
                    mono, rate = decoded
                    mask, _ = speech_mask(mono, rate, frame_db(mono, rate), cfg, vad_session)
                    voice_windows[row.rel_path] = speakers.embed_segments(session, mono, rate, mask, cfg.speaker_device)
                except GPU_UNAVAILABLE:
                    raise
                except Exception as exc:
                    # No windows: a consented export leaves the clip out, it never skips the check.
                    row.second_voice_error = str(exc) or type(exc).__name__
                    logger.warning("speech_second_voice_failed", rel_path=row.rel_path, error=row.second_voice_error)
            if i == len(pending) or i % 5 == 0:
                emit("speakers", i, len(pending))

    check_cancelled()
    multi = mark_duplicates([rows[rel] for rel in order if rel in rows], cfg.duplicate_distance)
    speaker_clusters: list[SpeakerCluster] = []
    mismatches = 0
    if cfg.speakers:
        ordered = [rows[rel] for rel in order if rel in rows]
        speakers.assign_clusters(ordered, embeddings, cfg.speaker_threshold)
        speaker_clusters, mismatches = speakers.agreement(ordered)
        with lock:
            completed.add("speakers")
    summary = snapshot()
    summary.speaker_clusters, summary.speaker_mismatches = speaker_clusters, mismatches
    summary.language_mismatches = language_id.count_mismatches(summary.clips)
    summary.embeddings.update(embeddings)
    summary.voice_windows.update(voice_windows)
    logger.info("speech_scan_done", folder=folder_str, total=summary.total, passed=summary.passed)
    return summary
