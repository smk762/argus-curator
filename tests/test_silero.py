"""Silero VAD for a speech scan (#204): the opt-in model VAD behind speech ratio and SNR.

No model, GPU or onnxruntime is needed: Silero runs against a fake session that
answers from each window's loudness. The real model runs only when
``ARGUS_CURATOR_VAD_MODEL`` names it (``argus-curator vad-fetch``).
"""

from __future__ import annotations

import contextlib
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from speech_fixtures import clip, silence, voice, write_wav

from argus_curator import runtime, silero, speech
from argus_curator.models import SpeechConfig
from argus_curator.runtime import ModelUnavailable
from argus_curator.server.app import create_app
from argus_curator.speech import scan_speech_folder

PASS_SHAPE = (1, silero.WINDOW + silero.CONTEXT)


class _Silero:
    """Stands in for Silero: checks each pass is the fixed shape and threads the state, answers by loudness."""

    def __init__(self, prob: float | None = None) -> None:
        self.prob, self.passes = prob, 0

    def run(self, _outputs: object, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        x, state, sr = feeds["input"], feeds["state"], feeds["sr"]
        assert x.shape == PASS_SHAPE and x.dtype == np.float32
        assert state.shape == silero.STATE_SHAPE and state.dtype == np.float32
        assert sr.dtype == np.int64 and int(sr) == silero.SAMPLE_RATE
        # The state carried in is zeros (a clip's first pass) or the one the last pass gave back.
        assert float(state.flat[0]) in (0.0, self.passes)
        self.passes += 1
        window = x[0, silero.CONTEXT :]
        loud = float(np.sqrt(np.mean(window**2))) > 0.01
        prob = self.prob if self.prob is not None else (0.9 if loud else 0.05)
        return [np.array([[prob]], dtype=np.float32), np.full(silero.STATE_SHAPE, self.passes, dtype=np.float32)]


@pytest.fixture
def fake_silero(monkeypatch: pytest.MonkeyPatch) -> _Silero:
    session = _Silero()
    monkeypatch.setattr(silero, "vad_model_available", lambda: True)
    monkeypatch.setattr(silero, "get_session", lambda: session)
    return session


def _with_line(path: Path) -> Path:
    path.with_suffix(".txt").write_text("the quick brown fox jumps over", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# The scan: opt-in, recorded on each row, fixed-shape passes
# ---------------------------------------------------------------------------


def test_the_energy_vad_stays_the_default_and_rows_name_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _with_line(clip(tmp_path / "a.wav"))
    monkeypatch.setattr(silero, "get_session", lambda: pytest.fail("a default scan loaded Silero"))
    assert SpeechConfig().vad == "energy"
    row = scan_speech_folder(tmp_path).clips[0]
    assert row.vad == "energy" and row.vad_error is None
    assert row.speech_ratio is not None and row.snr_db is not None


def test_silero_reads_the_clip_in_fixed_shape_passes_and_rows_name_it(tmp_path: Path, fake_silero: _Silero) -> None:
    # 1 s of a steady vowel between 0.5 s silences, at 22.05 kHz: resampled to 16 kHz first.
    rate = 22050
    vowel = 0.3 * np.sin(2 * np.pi * 140 * np.arange(rate) / rate).astype(np.float32)
    write_wav(tmp_path / "a.wav", np.concatenate([silence(0.5, rate), vowel, silence(0.5, rate)]), rate)
    _with_line(tmp_path / "a.wav")
    row = scan_speech_folder(tmp_path, cfg=SpeechConfig(vad="silero")).clips[0]
    assert row.vad == "silero" and row.vad_error is None
    assert fake_silero.passes == -(-2 * silero.SAMPLE_RATE // silero.WINDOW)
    assert row.speech_ratio == pytest.approx(0.5, abs=0.08)
    assert row.snr_db is not None and row.snr_db > 30


def test_the_rows_speech_ratio_and_snr_follow_silero_not_loudness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A loud clip Silero hears no speech in has no speech ratio worth the name, and no SNR."""
    _with_line(clip(tmp_path / "a.wav"))
    monkeypatch.setattr(silero, "vad_model_available", lambda: True)
    monkeypatch.setattr(silero, "get_session", lambda: _Silero(prob=0.01))
    row = scan_speech_folder(tmp_path, cfg=SpeechConfig(vad="silero")).clips[0]
    assert row.vad == "silero" and row.speech_ratio == 0.0 and row.snr_db is None
    rejected = scan_speech_folder(tmp_path, cfg=SpeechConfig(vad="silero", min_speech_ratio=0.3)).clips[0]
    assert rejected.reject_reason == "speech_ratio"
    assert scan_speech_folder(tmp_path).clips[0].speech_ratio > 0.5  # the energy VAD hears the same clip


def test_a_silero_failure_is_recorded_and_never_falls_back_to_energy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _with_line(clip(tmp_path / "a.wav"))

    class _Broken:
        def run(self, *_a: object) -> None:
            raise RuntimeError("bad window")

    monkeypatch.setattr(silero, "vad_model_available", lambda: True)
    monkeypatch.setattr(silero, "get_session", lambda: _Broken())
    row = scan_speech_folder(tmp_path, cfg=SpeechConfig(vad="silero")).clips[0]
    assert row.vad is None and row.vad_error == "bad window"
    assert row.speech_ratio is None and row.snr_db is None
    assert row.passed  # no threshold reads it
    cfg = SpeechConfig(vad="silero", min_speech_ratio=0.2, min_snr_db=10)
    row = scan_speech_folder(tmp_path, cfg=cfg).clips[0]
    assert row.failed_checks == ["snr", "speech_ratio"]
    assert "silero VAD could not read it" in (row.reject_detail or "")


def test_silero_without_the_model_is_refused_before_any_clip_is_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip(tmp_path / "a.wav")
    monkeypatch.setenv(silero.MODEL_ENV, str(tmp_path / "missing.onnx"))
    monkeypatch.setattr(speech, "decode", lambda *a: pytest.fail("decoded before refusing"))
    with pytest.raises(ModelUnavailable, match="Silero") as exc:
        scan_speech_folder(tmp_path, cfg=SpeechConfig(vad="silero"))
    assert str(tmp_path) not in str(exc.value)


def test_the_server_refuses_silero_without_the_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "corpus").mkdir()
    clip(tmp_path / "corpus" / "a.wav")
    monkeypatch.setenv(silero.MODEL_ENV, str(tmp_path / "missing.onnx"))
    client = TestClient(create_app(source_root=tmp_path, export_root=tmp_path / "out"))
    body = {
        "folder": "corpus",
        "target_profile": {"modality": "speech", "target_category": "voice"},
        "speech": {"vad": "silero"},
    }
    resp = client.post("/scan/folder", json=body)
    assert resp.status_code == 400 and "vad-fetch" in resp.json()["detail"]
    assert str(tmp_path) not in resp.text
    assert client.post("/scan/folder", json={**body, "speech": {"vad": "webrtc"}}).status_code == 422
    assert client.get("/detectors").json()["silero_vad"] is False


# ---------------------------------------------------------------------------
# The session: pinned, CPU only, never under the GPU lease
# ---------------------------------------------------------------------------


def test_silero_is_built_on_cpu_alone_with_one_thread_and_takes_no_gpu_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even where CUDA is available: Silero never holds a CUDA arena or asks for a lease."""

    class _NoLease:
        name = "recorder"

        def lease(self, **_options: object) -> contextlib.AbstractContextManager[object]:
            pytest.fail("Silero took a GPU lease")

    built: dict[str, object] = {}
    model = tmp_path / "silero.onnx"
    model.write_bytes(b"model")
    monkeypatch.setenv(silero.MODEL_ENV, str(model))
    monkeypatch.setattr(silero, "MODEL_SHA256", runtime.sha256(model))
    monkeypatch.setattr(runtime, "providers", lambda device: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(runtime, "_coordinator", _NoLease())
    monkeypatch.setattr(runtime, "_SESSION_CACHE", {})

    def fake_build(path, provs, **kwargs):  # noqa: ANN001, ANN202
        built.update(providers=runtime.session_providers(provs), threads=kwargs.get("threads"))
        return _Silero()

    monkeypatch.setattr(runtime, "_build_session", fake_build)
    monkeypatch.setattr(silero, "onnxruntime_available", lambda: True)
    session = silero.get_session()
    assert built == {"providers": ["CPUExecutionProvider"], "threads": 1}
    silero.probabilities(session, voice(1.0), 16000)
    # A whole scan with Silero, with CUDA on offer, takes no lease either.
    _with_line(clip(tmp_path / "corpus" / "a.wav"))
    assert scan_speech_folder(tmp_path / "corpus", cfg=SpeechConfig(vad="silero")).clips[0].vad == "silero"


def test_load_session_caps_intra_op_threads_when_asked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    class _Options:
        intra_op_num_threads = 0
        inter_op_num_threads = 0

    fake = types.ModuleType("onnxruntime")
    fake.SessionOptions = _Options  # type: ignore[attr-defined]
    fake.InferenceSession = lambda path, **kw: seen.update(kw)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "onnxruntime", fake)
    monkeypatch.setattr(runtime, "_SESSION_CACHE", {})
    model = tmp_path / "m.onnx"
    model.write_bytes(b"model")
    runtime.load_session(model, ["CPUExecutionProvider"], missing_hint="", threads=1)
    opts = seen["sess_options"]
    assert (opts.intra_op_num_threads, opts.inter_op_num_threads) == (1, 1)
    assert seen["providers"] == ["CPUExecutionProvider"]


def test_a_silero_model_that_is_not_the_pinned_file_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = tmp_path / "silero.onnx"
    model.write_bytes(b"something else")
    monkeypatch.setenv(silero.MODEL_ENV, str(model))
    monkeypatch.setattr(runtime, "_build_session", lambda *a, **k: pytest.fail("loaded an unpinned model"))
    with pytest.raises(ModelUnavailable, match="not the pinned file") as exc:
        silero.get_session()
    assert str(tmp_path) not in str(exc.value)


# ---------------------------------------------------------------------------
# The mask
# ---------------------------------------------------------------------------


def test_speech_windows_use_hysteresis_close_short_gaps_and_drop_short_runs() -> None:
    p = np.zeros(60, dtype=np.float32)
    p[5:15] = 0.9  # speech
    p[15:17] = 0.1  # a 64 ms gap: closed
    p[17:25] = 0.9
    p[25:30] = 0.4  # between the thresholds after speech: still speech
    p[40:44] = 0.9  # 128 ms of speech alone: dropped
    mask = silero.speech_windows(p)
    assert mask[5:30].all() and not mask[:5].any() and not mask[30:].any()
    # Between the thresholds with no speech before it is not speech.
    assert not silero.speech_windows(np.full(20, 0.4, dtype=np.float32)).any()


def test_frame_mask_maps_windows_onto_the_scan_frames() -> None:
    windows = np.array([False, True, False])  # 0-32, 32-64, 64-96 ms
    frames = silero.frame_mask(windows, 9, speech.FRAME_S, speech.HOP_S)
    # Frame i is centred at 10 i + 15 ms.
    assert frames.tolist() == [False, False, True, True, True, False, False, False, False]
    assert not silero.frame_mask(np.zeros(0, dtype=bool), 4, speech.FRAME_S, speech.HOP_S).any()


def test_the_snr_split_keeps_unsure_frames_and_a_guard_band_off_the_noise_side() -> None:
    db = np.full(200, -80.0)
    db[50:100] = -20.0  # speech
    db[100:104] = -30.0  # a word's decay, outside the speech mask
    mask = np.zeros(200, dtype=bool)
    mask[50:100] = True
    plain = speech.snr_db(db, mask)
    guarded = speech.snr_db(db, mask, silero.guarded(mask))
    assert plain is not None and guarded is not None
    assert guarded == pytest.approx(60.0, abs=0.1) and plain < 45


@pytest.mark.skipif(
    not (os.environ.get(silero.MODEL_ENV) and runtime.onnxruntime_available()),
    reason="a real Silero run needs the model (argus-curator vad-fetch) and onnxruntime",
)
def test_real_silero_gives_a_probability_per_window() -> None:  # pragma: no cover - needs the model
    probs = silero.probabilities(silero.get_session(), np.concatenate([silence(0.5), voice(1.0)]), 16000)
    assert len(probs) == -(-int(1.5 * 16000) // silero.WINDOW)
    assert ((probs >= 0) & (probs <= 1)).all()
