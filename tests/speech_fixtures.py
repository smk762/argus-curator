"""Tiny synthetic clips for the speech-scan tests, written with the stdlib ``wave`` module."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def voice(seconds: float, rate: int = 16000, *, f0: float = 140.0, seed: int = 0, level: float = 0.3) -> np.ndarray:
    """Something speech-like: a harmonic buzz whose pitch drifts, in ~4 Hz syllables, each
    with its own vowel (random harmonic weights, as formants would give)."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * rate)) / rate
    pitch = f0 * (1 + 0.1 * np.sin(2 * np.pi * (0.5 + rng.random()) * t))
    phase = 2 * np.pi * np.cumsum(pitch) / rate
    syllable_hz = 3.5 + rng.random()
    cycle = 2 * np.pi * syllable_hz * t + rng.random() * 6
    syllables = np.clip(np.sin(cycle), 0, None) ** 0.5
    vowel = (cycle // (2 * np.pi)).astype(int)
    vowel -= vowel.min()
    weights = rng.random((vowel.max() + 1, 24)) ** 2
    x = sum(weights[vowel, k - 1] * np.sin(k * phase) for k in range(1, 25)) * syllables
    return (level * x / np.max(np.abs(x))).astype(np.float32)


def silence(seconds: float, rate: int = 16000) -> np.ndarray:
    return np.zeros(int(seconds * rate), dtype=np.float32)


def write_wav(path: Path, samples: np.ndarray, rate: int = 16000) -> Path:
    """16-bit PCM WAV; *samples* is ``(n,)`` mono or ``(n, channels)``, floats in [-1, 1]."""
    data = np.asarray(samples, dtype=np.float64)
    if data.ndim == 1:
        data = data[:, None]
    pcm = (np.clip(data, -1.0, 32767 / 32768) * 32768).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(data.shape[1])
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return path


def clip(path: Path, seconds: float = 2.0, rate: int = 16000, *, lead: float = 0.2, trail: float = 0.2, **kw) -> Path:
    """A clean clip with short edge silences."""
    return write_wav(
        path, np.concatenate([silence(lead, rate), voice(seconds, rate, **kw), silence(trail, rate)]), rate
    )
