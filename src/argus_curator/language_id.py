"""Spoken language ID for a speech scan: check the corpus's language labels (#208).

A speech scan carries the corpus's ``language`` label verbatim, and
spontaneous-speech corpora mislabel some clips. With
``SpeechConfig.language_id`` each clip that passed is classified by
SpeechBrain's VoxLingua107 ECAPA-TDNN language-ID model
(``speechbrain/lang-id-voxlingua107-ecapa``, Apache-2.0, trained on
VoxLingua107, CC BY 4.0), and a clip whose label the model disputes is
flagged (``ClipResult.language_mismatch``) with the model's confidence. The
label is never changed, and nothing is rejected unless the scan asks for it
(``SpeechConfig.reject_language_mismatch``).

SpeechBrain publishes the model as PyTorch checkpoints only, so
``argus-curator lid-convert`` (``pip install argus-curator[lid-convert]``)
fetches them pinned by Hugging Face revision and sha256 and exports the ECAPA
encoder and its classifier to ONNX once. The scan needs onnxruntime alone:
torch is never imported here at run time. The converter writes
``<model>.json`` beside the ONNX file naming the pinned sources and the ONNX
file's own sha256, and a model whose sources are not the pinned ones, or that
does not hash to what its record says, is refused at load.

The model reads 60-bin log mel filterbanks of 16 kHz audio, as SpeechBrain's
``Fbank`` computes them (:func:`fbank`, numpy), with the clip's mean taken off
each bin. A clip is cut into :data:`WINDOWS` windows of :data:`WINDOW_FRAMES`
frames spread evenly across it (a short clip is tiled up to one window first),
run as one fixed-shape batch, so the capped CUDA arena never grows with a
clip's length. The clip's language probabilities are the mean of its windows'.
"""

from __future__ import annotations

import json
import math
import threading
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from argus_curator.dnsmos import resample
from argus_curator.models import ClipResult
from argus_curator.runtime import (
    ModelUnavailable,
    configured_path,
    gpu_lease,
    load_session,
    onnxruntime_available,
    providers,
    sha256,
)

MODEL_ENV = "ARGUS_CURATOR_LID_MODEL"
DEFAULT_MODEL_NAME = "voxlingua107_ecapa_lid.onnx"
HF_REPO = "speechbrain/lang-id-voxlingua107-ecapa"
HF_REVISION = "0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9"
# The files lid-convert reads, pinned. label_encoder.txt is the class order
# LANGUAGES spells out; the converter checks the two agree.
SOURCE_SHA256 = {
    "embedding_model.ckpt": "ab750d5c06d713477045fa798fab5d33e959dbc0dfe4de510a9a47844c79a19a",
    "classifier.ckpt": "a50d9024ff58d317031c9787d4c6c614d454a87a8ef32f9d36338cd3ff57adbc",
    "label_encoder.txt": "9f566d83c4f19168be4a0bf86c0c7dac7d3264a95105bcbf33a7c32b83ccc17f",
}

# VoxLingua107's classes in the model's output order: (code, name). The codes
# are the model's own, so Hebrew is ``iw`` and Javanese ``jw``.
LANGUAGES: tuple[tuple[str, str], ...] = (
    ("ab", "Abkhazian"),
    ("af", "Afrikaans"),
    ("am", "Amharic"),
    ("ar", "Arabic"),
    ("as", "Assamese"),
    ("az", "Azerbaijani"),
    ("ba", "Bashkir"),
    ("be", "Belarusian"),
    ("bg", "Bulgarian"),
    ("bn", "Bengali"),
    ("bo", "Tibetan"),
    ("br", "Breton"),
    ("bs", "Bosnian"),
    ("ca", "Catalan"),
    ("ceb", "Cebuano"),
    ("cs", "Czech"),
    ("cy", "Welsh"),
    ("da", "Danish"),
    ("de", "German"),
    ("el", "Greek"),
    ("en", "English"),
    ("eo", "Esperanto"),
    ("es", "Spanish"),
    ("et", "Estonian"),
    ("eu", "Basque"),
    ("fa", "Persian"),
    ("fi", "Finnish"),
    ("fo", "Faroese"),
    ("fr", "French"),
    ("gl", "Galician"),
    ("gn", "Guarani"),
    ("gu", "Gujarati"),
    ("gv", "Manx"),
    ("ha", "Hausa"),
    ("haw", "Hawaiian"),
    ("hi", "Hindi"),
    ("hr", "Croatian"),
    ("ht", "Haitian"),
    ("hu", "Hungarian"),
    ("hy", "Armenian"),
    ("ia", "Interlingua"),
    ("id", "Indonesian"),
    ("is", "Icelandic"),
    ("it", "Italian"),
    ("iw", "Hebrew"),
    ("ja", "Japanese"),
    ("jw", "Javanese"),
    ("ka", "Georgian"),
    ("kk", "Kazakh"),
    ("km", "Central Khmer"),
    ("kn", "Kannada"),
    ("ko", "Korean"),
    ("la", "Latin"),
    ("lb", "Luxembourgish"),
    ("ln", "Lingala"),
    ("lo", "Lao"),
    ("lt", "Lithuanian"),
    ("lv", "Latvian"),
    ("mg", "Malagasy"),
    ("mi", "Maori"),
    ("mk", "Macedonian"),
    ("ml", "Malayalam"),
    ("mn", "Mongolian"),
    ("mr", "Marathi"),
    ("ms", "Malay"),
    ("mt", "Maltese"),
    ("my", "Burmese"),
    ("ne", "Nepali"),
    ("nl", "Dutch"),
    ("nn", "Norwegian Nynorsk"),
    ("no", "Norwegian"),
    ("oc", "Occitan"),
    ("pa", "Panjabi"),
    ("pl", "Polish"),
    ("ps", "Pushto"),
    ("pt", "Portuguese"),
    ("ro", "Romanian"),
    ("ru", "Russian"),
    ("sa", "Sanskrit"),
    ("sco", "Scots"),
    ("sd", "Sindhi"),
    ("si", "Sinhala"),
    ("sk", "Slovak"),
    ("sl", "Slovenian"),
    ("sn", "Shona"),
    ("so", "Somali"),
    ("sq", "Albanian"),
    ("sr", "Serbian"),
    ("su", "Sundanese"),
    ("sv", "Swedish"),
    ("sw", "Swahili"),
    ("ta", "Tamil"),
    ("te", "Telugu"),
    ("tg", "Tajik"),
    ("th", "Thai"),
    ("tk", "Turkmen"),
    ("tl", "Tagalog"),
    ("tr", "Turkish"),
    ("tt", "Tatar"),
    ("uk", "Ukrainian"),
    ("ur", "Urdu"),
    ("uz", "Uzbek"),
    ("vi", "Vietnamese"),
    ("war", "Waray"),
    ("yi", "Yiddish"),
    ("yo", "Yoruba"),
    ("zh", "Chinese"),
)
CODES: tuple[str, ...] = tuple(code for code, _ in LANGUAGES)
N_LANGUAGES = len(LANGUAGES)

# Corpus spellings the model's codes don't use: today's ISO 639-1 for the two
# legacy codes, Bokmål as Norwegian, Filipino as Tagalog, and Mandarin.
_ALIASES = {"he": "iw", "jv": "jw", "nb": "no", "fil": "tl", "cmn": "zh"}
_BY_NAME = {name.lower(): code for code, name in LANGUAGES}
# Languages a listener can barely tell apart by ear, which the model confuses
# with each other: a prediction inside the label's group is not a dispute.
SAME_LANGUAGE = (
    frozenset({"bs", "hr", "sr"}),
    frozenset({"no", "nn"}),
    frozenset({"hi", "ur"}),
    frozenset({"id", "ms"}),
    frozenset({"cs", "sk"}),
)

FEATURE_RATE = 16000
N_MELS = 60
N_FFT = 400  # 25 ms frames, the window as long as the FFT
HOP_SAMPLES = 160  # 10 ms
TOP_DB = 80.0
AMIN = 1e-10
# Each window: WINDOW_SECONDS of audio, center-padded frames (SpeechBrain's STFT).
WINDOW_SECONDS = 6
WINDOW_SAMPLES = WINDOW_SECONDS * FEATURE_RATE
WINDOW_FRAMES = WINDOW_SAMPLES // HOP_SAMPLES + 1
# Windows per clip, run as one batch: every pass is (WINDOWS, WINDOW_FRAMES, N_MELS).
WINDOWS = 3

_verified: dict[tuple[str, int, float], bool] = {}
_verified_lock = threading.Lock()


def model_path() -> Path:
    """Where the converted language-ID ONNX model is read from."""
    return configured_path(MODEL_ENV, DEFAULT_MODEL_NAME)


def record_path(path: Path) -> Path:
    """The provenance record lid-convert writes beside the model."""
    return path.with_name(path.name + ".json")


def lid_available() -> bool:
    """True when onnxruntime is importable and the model is in place (its record is checked at load)."""
    return onnxruntime_available() and model_path().is_file()


def problem() -> str | None:
    """Why language ID cannot run here, else None. Names no path."""
    if not lid_available():
        return (
            "language ID requested but the model is not in place "
            f"(argus-curator lid-convert, or {MODEL_ENV}; needs onnxruntime: argus-curator[speech])"
        )
    return None


def _verify(path: Path) -> None:
    """Refuse a model not converted from the pinned sources, or not the file its record names.

    Checked once per (path, size, mtime). The messages name neither the path nor
    a URL: they reach a client.
    """
    hint = f"Create it with `argus-curator lid-convert` or set {MODEL_ENV}."
    try:
        stat = path.stat()
    except OSError as exc:
        raise ModelUnavailable(f"language-ID model not found. {hint}") from exc
    key = (str(path), stat.st_size, stat.st_mtime)
    with _verified_lock:
        if _verified.get(key):
            return
        try:
            record = json.loads(record_path(path).read_text())
        except (OSError, ValueError) as exc:
            raise ModelUnavailable(f"the language-ID model has no conversion record beside it. {hint}") from exc
        if not isinstance(record, dict) or record.get("sources") != SOURCE_SHA256:
            raise ModelUnavailable(f"the language-ID model was not converted from the pinned checkpoints. {hint}")
        digest = sha256(path)
        if digest != record.get("onnx_sha256"):
            raise ModelUnavailable(
                f"the language-ID model is not the file its conversion record names (sha256 {digest[:12]}...). {hint}"
            )
        _verified[key] = True


def get_session(device: str):
    """The language-ID session, built under the GPU lease with the capped arena (runtime.load_session)."""
    path = model_path()
    _verify(path)
    with gpu_lease(device, "language", "load"):
        return load_session(
            path,
            providers(device),
            missing_hint=f"Create it with `argus-curator lid-convert` or set {MODEL_ENV}.",
            extra="speech",
        )


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


def _mel(hz: float) -> float:
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def _mel_banks() -> np.ndarray:
    """SpeechBrain's triangular filters (``Filterbank``, frozen), ``(N_FFT // 2 + 1, N_MELS)``.

    The centres are evenly spaced in mel from 0 Hz to Nyquist; each filter's
    slope on both sides is set by the gap to the centre below it, as
    SpeechBrain builds it.
    """
    mel = np.linspace(_mel(0.0), _mel(FEATURE_RATE / 2), N_MELS + 2)
    hz = 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
    band = (hz[1:] - hz[:-1])[:-1]
    centre = hz[1:-1]
    freqs = np.linspace(0, FEATURE_RATE // 2, N_FFT // 2 + 1)
    slope = (freqs[None, :] - centre[:, None]) / band[:, None]
    return np.maximum(0.0, np.minimum(slope + 1.0, 1.0 - slope)).T


_BANKS = _mel_banks()
# torch.hamming_window(N_FFT), periodic.
_WINDOW = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)


def fbank(audio16k: np.ndarray) -> np.ndarray:
    """``(frames, N_MELS)`` log mel filterbanks of mono 16 kHz audio, as SpeechBrain's ``Fbank`` computes them.

    Its STFT is centred (the signal zero-padded by half a frame each side), a
    periodic Hamming window over 400 samples every 160; the power spectrum
    goes through the mel filters, to dB (10 log10, floored at 1e-10), clipped
    at 80 dB under the loudest bin of the whole input.
    """
    x = np.pad(np.asarray(audio16k, dtype=np.float64), N_FFT // 2)
    n = 1 + (len(x) - N_FFT) // HOP_SAMPLES
    frames = x[np.arange(N_FFT)[None] + HOP_SAMPLES * np.arange(n)[:, None]]
    power = np.abs(np.fft.rfft(frames * _WINDOW, N_FFT)) ** 2
    db = 10.0 * np.log10(np.maximum(power @ _BANKS, AMIN))
    return np.maximum(db, db.max() - TOP_DB).astype(np.float32)


def windows(mono: np.ndarray, rate: int) -> np.ndarray:
    """The model's input for one clip: ``(WINDOWS, WINDOW_FRAMES, N_MELS)``, mean-normalised.

    The clip is resampled to 16 kHz, tiled up to one window when shorter, and
    its filterbanks normalised by their mean over the clip (SpeechBrain's
    sentence normalisation, without the variance). The windows start evenly
    across it.
    """
    audio = resample(np.asarray(mono, dtype=np.float32), rate, FEATURE_RATE)
    if len(audio) == 0 or not np.any(audio):
        raise ValueError("no audio")
    while len(audio) < WINDOW_SAMPLES:
        audio = np.concatenate([audio, audio])
    feats = fbank(audio)
    feats = feats - feats.mean(axis=0, keepdims=True)
    starts = np.linspace(0, len(feats) - WINDOW_FRAMES, WINDOWS).round().astype(int)
    return np.stack([feats[s : s + WINDOW_FRAMES] for s in starts]).astype(np.float32)


def predict(session, batch: np.ndarray) -> np.ndarray:
    """The clip's probability per language (:data:`LANGUAGES` order) from its windows, one fixed-shape pass.

    The model gives log-probabilities per window; the clip's are the mean of
    the windows' probabilities.
    """
    name = session.get_inputs()[0].name
    out = np.asarray(session.run(None, {name: batch})[0], dtype=np.float64)
    out = out.reshape(len(batch), -1)
    if out.shape[1] != N_LANGUAGES:
        raise ValueError(f"the language-ID model gave {out.shape[1]} classes, not {N_LANGUAGES}")
    probs = np.exp(out - out.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    return probs.mean(axis=0)


def identify(session, mono: np.ndarray, rate: int, device: str) -> np.ndarray:
    """One clip's language probabilities, under one GPU lease."""
    batch = windows(mono, rate)
    with gpu_lease(device, "language"):
        return predict(session, batch)


# ---------------------------------------------------------------------------
# Judging the corpus label
# ---------------------------------------------------------------------------


def model_code(label: str | None) -> str | None:
    """The model's code for a corpus language label, or None when the model doesn't know it.

    Reads a BCP-47 / Common Voice locale by its primary subtag (``pt-BR`` is
    ``pt``, ``zh-CN`` is ``zh``), a few aliases (``he``, ``jv``, ``nb``,
    ``fil``, ``cmn``) and the English language names the model lists. A label
    it can't place (Common Voice's ``fy-NL``, ``yue``) gets None: no judgement.
    """
    if not label:
        return None
    text = label.strip().lower()
    if text in _BY_NAME:
        return _BY_NAME[text]
    primary = text.replace("_", "-").split("-")[0]
    primary = _ALIASES.get(primary, primary)
    return primary if primary in CODES else None


def _agrees(predicted: str, labelled: str) -> bool:
    return predicted == labelled or any(predicted in g and labelled in g for g in SAME_LANGUAGE)


def label_probability(probs: np.ndarray, labelled: str) -> float:
    """The probability the model gives *labelled*, its :data:`SAME_LANGUAGE` group summed."""
    return float(sum(probs[i] for i, code in enumerate(CODES) if _agrees(code, labelled)))


def judge(row: ClipResult, probs: np.ndarray, dispute_below: float) -> None:
    """Record the model's reading of *row* beside its corpus label, in place.

    ``language_predicted`` is the model's top language and
    ``language_confidence`` its probability; ``language_label_confidence`` the
    probability the model gives the corpus label (with the languages the ear
    can't split from it, :data:`SAME_LANGUAGE`), None when the label is missing
    or one the model doesn't know. The clip is a ``language_mismatch`` when
    that probability is under *dispute_below*: the model all but rules the
    label out. That reads the label, not the model's top guess, because on
    noisy or telephone audio the model is often unsure what a clip is while
    still sure what it is not (docs/speech.md). The corpus label is never changed.
    """
    best = int(np.argmax(probs))
    row.language_predicted = CODES[best]
    row.language_confidence = round(float(probs[best]), 4)
    labelled = model_code(row.language)
    if labelled is None:
        row.language_label_confidence, row.language_mismatch = None, False
        return
    p = label_probability(probs, labelled)
    row.language_label_confidence = round(p, 4)
    row.language_mismatch = p < dispute_below


def count_mismatches(clips: Sequence[ClipResult]) -> int:
    return sum(1 for c in clips if c.language_mismatch)


# ---------------------------------------------------------------------------
# lid-convert: the one-time export (torch and SpeechBrain, behind an extra)
# ---------------------------------------------------------------------------

_CONVERT_HINT = "Conversion requires: pip install argus-curator[lid-convert]"


def fetch_sources() -> Path:
    """Download (once) the pinned SpeechBrain checkpoints; the directory holding them."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - exercised only with the extra
        raise ModelUnavailable(_CONVERT_HINT) from exc
    return Path(snapshot_download(HF_REPO, revision=HF_REVISION, allow_patterns=list(SOURCE_SHA256)))


def check_sources(source_dir: Path) -> None:
    """Refuse sources that are not the pinned files, or a label order that is not :data:`LANGUAGES`."""
    for name, expected in SOURCE_SHA256.items():
        path = source_dir / name
        if not path.is_file():
            raise ModelUnavailable(f"{name} is missing from {source_dir}")
        digest = sha256(path)
        if digest != expected:
            raise ModelUnavailable(f"{name} is not the pinned {HF_REPO} file (sha256 {digest})")
    lines = [ln for ln in (source_dir / "label_encoder.txt").read_text().splitlines() if "=>" in ln]
    order = [ln.split("'")[1].split(":")[0] for ln in lines if ln.startswith("'") and ":" in ln]
    if tuple(order) != CODES:
        raise ModelUnavailable("label_encoder.txt's class order is not the one the code names")


def export_onnx(source_dir: Path, output: Path) -> Path:
    """Export the pinned VoxLingua107 ECAPA encoder and classifier to ONNX at *output*, with its record.

    Input ``feats`` ``(WINDOWS, WINDOW_FRAMES, N_MELS)`` (see :func:`windows`),
    output ``log_probs`` ``(WINDOWS, N_LANGUAGES)``. The architecture is
    SpeechBrain's from the model's ``hyperparams.yaml``, built from its classes
    and loaded strictly from the pinned checkpoints. Needs ``pip install
    argus-curator[lid-convert]``.
    """
    check_sources(source_dir)
    try:
        import torch
        from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
        from speechbrain.lobes.models.Xvector import Classifier
    except ImportError as exc:  # pragma: no cover - exercised only with the extra
        raise ModelUnavailable(_CONVERT_HINT) from exc

    encoder = ECAPA_TDNN(
        input_size=N_MELS,
        channels=[1024, 1024, 1024, 1024, 3072],
        kernel_sizes=[5, 3, 3, 3, 1],
        dilations=[1, 2, 3, 4, 1],
        attention_channels=128,
        lin_neurons=256,
    )
    classifier = Classifier(
        input_shape=[None, None, 256],
        activation=torch.nn.LeakyReLU,
        lin_blocks=1,
        lin_neurons=512,
        out_neurons=N_LANGUAGES,
    )
    for module, name in ((encoder, "embedding_model.ckpt"), (classifier, "classifier.ckpt")):
        state = torch.load(source_dir / name, map_location="cpu", weights_only=True)
        module.load_state_dict(state, strict=True)

    class _Lid(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder, self.classifier = encoder, classifier

        def forward(self, feats):
            return self.classifier(self.encoder(feats)).squeeze(1)

    model = _Lid().eval()
    dummy = torch.zeros(WINDOWS, WINDOW_FRAMES, N_MELS)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".part")
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy,),
            str(tmp),
            input_names=["feats"],
            output_names=["log_probs"],
            opset_version=17,
            dynamo=False,
        )
    tmp.replace(output)
    record = {
        "model": HF_REPO,
        "revision": HF_REVISION,
        "sources": SOURCE_SHA256,
        "onnx_sha256": sha256(output),
        "input": [WINDOWS, WINDOW_FRAMES, N_MELS],
        "torch": torch.__version__,
    }
    record_path(output).write_text(json.dumps(record, indent=2) + "\n")
    _verified.clear()
    return output
