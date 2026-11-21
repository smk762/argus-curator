"""Speech scans (#186): prefilters, measurements, transcripts, fingerprints, DNSMOS and the STT hook.

Every clip is synthesised with the stdlib ``wave`` module (tests/speech_fixtures.py);
no model, GPU or network is needed. DNSMOS runs against a fake session.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from speech_fixtures import clip, silence, voice, write_wav

from argus_curator import dnsmos, runtime, speech, speech_corpus
from argus_curator.models import (
    ClipResult,
    ExportRequest,
    ManifestRow,
    ScanRequest,
    ScanSummary,
    SpeechConfig,
    SpeechTargetProfile,
    TargetProfile,
)
from argus_curator.runtime import ModelUnavailable, ScanCancelled
from argus_curator.scanner import scan_folder
from argus_curator.server.app import create_app
from argus_curator.speech import scan_speech_folder
from argus_curator.store import ScanStore

LINE = "the quick brown fox jumps over"  # 25 letters: ~12 chars/s over 2 s


def _by_rel(summary: ScanSummary) -> dict[str, ClipResult]:
    return {c.rel_path: c for c in summary.clips}


def _with_line(path: Path, text: str = LINE) -> Path:
    path.with_suffix(".txt").write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# One clip: measurements and the named prefilters
# ---------------------------------------------------------------------------


def test_a_clean_clip_passes_with_its_measurements(tmp_path: Path) -> None:
    _with_line(clip(tmp_path / "a.wav", 2.0, lead=0.3, trail=0.2))
    row = scan_speech_folder(tmp_path).clips[0]
    assert row.passed and row.reject_reason is None and row.failed_checks == []
    assert (row.sample_rate, row.channels) == (16000, 1)
    assert row.duration_s == pytest.approx(2.5, abs=0.01)
    assert row.leading_silence_s == pytest.approx(0.3, abs=0.05)
    assert row.trailing_silence_s == pytest.approx(0.2, abs=0.05)
    assert row.clipping_ratio == 0.0
    assert row.snr_db is not None and row.snr_db > 40
    assert 0.5 < (row.speech_ratio or 0) < 1.0
    assert row.transcript == LINE and row.transcript_source == "sidecar"
    assert row.chars_per_s == pytest.approx(25 / 2.0, rel=0.1)
    assert len(row.fingerprint) == 16
    assert 0 < row.score <= 1 and set(row.score_breakdown) == {"snr", "speech_ratio"}


@pytest.mark.parametrize(
    ("make", "cfg", "check"),
    [
        (lambda p: clip(p, 0.4, lead=0.05, trail=0.05), {}, "duration"),
        (lambda p: clip(p, 3.0), {"max_duration_s": 2.0}, "duration"),
        (lambda p: clip(p, 2.0, rate=8000), {}, "sample_rate"),
        (lambda p: write_wav(p, np.stack([voice(2.0)] * 2, axis=1)), {}, "channels"),
        (lambda p: write_wav(p, np.clip(voice(2.0, level=1.0) * 3, -1, 1)), {}, "clipping"),
        (lambda p: clip(p, 2.0, lead=2.0), {}, "leading_silence"),
        (lambda p: clip(p, 2.0, trail=2.0), {}, "trailing_silence"),
        (lambda p: write_wav(p, silence(2.0)), {}, "silent"),
        (lambda p: p.write_bytes(b"RIFF\x00\x00\x00\x00WAVEnot really"), {}, "unreadable"),
        (lambda p: clip(p, 2.0), {"min_snr_db": 200.0}, "snr"),
        (lambda p: clip(p, 2.0), {"min_speech_ratio": 0.99}, "speech_ratio"),
    ],
)
def test_each_prefilter_names_the_check_it_failed(tmp_path: Path, make, cfg: dict, check: str) -> None:
    path = tmp_path / "x.wav"
    make(path)
    _with_line(path)
    row = scan_speech_folder(tmp_path, cfg=SpeechConfig(**cfg)).clips[0]
    assert not row.passed
    assert row.reject_reason == check
    assert check in row.failed_checks and row.reject_detail


def test_transcript_and_speaking_rate_checks(tmp_path: Path) -> None:
    clip(tmp_path / "none.wav")
    _with_line(clip(tmp_path / "fast.wav"), "a" * 400)
    _with_line(clip(tmp_path / "slow.wav"), "hi")
    rows = _by_rel(scan_speech_folder(tmp_path))
    assert rows["none.wav"].reject_reason == "transcript" and rows["none.wav"].chars_per_s is None
    assert rows["fast.wav"].reject_reason == "speaking_rate" and "max_chars_per_s" in rows["fast.wav"].reject_detail
    assert rows["slow.wav"].reject_reason == "speaking_rate" and "min_chars_per_s" in rows["slow.wav"].reject_detail
    lenient = _by_rel(scan_speech_folder(tmp_path, cfg=SpeechConfig(require_transcript=False)))
    assert lenient["none.wav"].passed


def test_every_failed_check_is_listed_in_order(tmp_path: Path) -> None:
    write_wav(tmp_path / "x.wav", np.concatenate([silence(2.0), voice(0.5)]), rate=8000)
    row = scan_speech_folder(tmp_path).clips[0]
    assert row.failed_checks == ["sample_rate", "leading_silence", "transcript"]
    assert row.reject_reason == "sample_rate" and "8000 Hz" in row.reject_detail


def test_a_long_file_is_never_read_past_its_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A file bigger than max_duration_s could hold is a duration reject, and open_audio never sees it."""
    clip(tmp_path / "x.wav", 2.0)
    monkeypatch.setattr(speech, "_MAX_BYTES_PER_S", 1)
    monkeypatch.setattr(speech, "open_audio", lambda *a, **k: pytest.fail("decoded a file over the byte cap"))
    cfg = SpeechConfig(max_duration_s=0.001, min_duration_s=0.0)
    path = tmp_path / "x.wav"
    path.write_bytes(path.read_bytes() + b"\0" * (2 << 20))
    row = scan_speech_folder(tmp_path, cfg=cfg).clips[0]
    assert row.reject_reason == "duration"


def test_the_scan_reads_no_clip_outside_its_folder_through_a_symlink(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    root = tmp_path / "root"
    _with_line(clip(outside / "secret.wav"))
    root.mkdir()
    (root / "link.wav").symlink_to(outside / "secret.wav")
    (root / "dir").symlink_to(outside, target_is_directory=True)
    assert scan_speech_folder(root).clips == []


# ---------------------------------------------------------------------------
# Transcripts and corpus labels
# ---------------------------------------------------------------------------


def test_ljspeech_metadata_maps_clips_by_stem(tmp_path: Path) -> None:
    clip(tmp_path / "wavs" / "LJ001-0001.wav")
    clip(tmp_path / "wavs" / "LJ001-0002.wav")
    (tmp_path / "metadata.csv").write_text(
        "LJ001-0001|Printing, in the only sense|printing in the only sense\nLJ001-0002|In being comparatively modern|\n",
        encoding="utf-8",
    )
    rows = _by_rel(scan_speech_folder(tmp_path))
    first, second = rows["wavs/LJ001-0001.wav"], rows["wavs/LJ001-0002.wav"]
    assert (first.transcript, first.transcript_source) == ("printing in the only sense", "ljspeech")
    assert second.transcript == "In being comparatively modern"  # no normalized text: the transcription


def test_common_voice_tsv_carries_its_labels(tmp_path: Path) -> None:
    clip(tmp_path / "clips" / "common_voice_pt_1.wav")
    (tmp_path / "validated.tsv").write_text(
        "client_id\tpath\tsentence\tup_votes\tage\tgender\taccents\tlocale\n"
        'abc\tcommon_voice_pt_1.mp3\tEle disse "olá" ao vizinho\t2\tthirties\tmale\tLisboa\tpt\n',
        encoding="utf-8",
    )
    row = scan_speech_folder(tmp_path).clips[0]
    assert row.transcript == 'Ele disse "olá" ao vizinho' and row.transcript_source == "common_voice"
    assert (row.speaker, row.accent, row.language, row.age) == ("abc", "Lisboa", "pt", "thirties")


def test_a_sidecar_wins_and_the_layout_narrows_the_sources(tmp_path: Path) -> None:
    clip(tmp_path / "a.wav")
    _with_line(tmp_path / "a.wav", "from the sidecar")
    (tmp_path / "metadata.csv").write_text("a|from ljspeech|\n", encoding="utf-8")
    assert scan_speech_folder(tmp_path).clips[0].transcript == "from the sidecar"
    only_lj = scan_speech_folder(tmp_path, cfg=SpeechConfig(transcripts="ljspeech")).clips[0]
    assert (only_lj.transcript, only_lj.transcript_source) == ("from ljspeech", "ljspeech")


def test_corpus_metadata_outside_the_folder_is_never_read(tmp_path: Path) -> None:
    root = tmp_path / "root"
    clip(root / "a.wav")
    (tmp_path / "metadata.csv").write_text("a|leaked|\n", encoding="utf-8")
    (tmp_path / "a.txt").write_text("leaked", encoding="utf-8")
    (root / "metadata.csv").symlink_to(tmp_path / "metadata.csv")
    (root / "a.txt").symlink_to(tmp_path / "a.txt")
    row = scan_speech_folder(root).clips[0]
    assert row.transcript is None and row.reject_reason == "transcript"


def test_a_speaker_the_corpus_marks_under_18_is_rejected_undecoded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Common Voice's "teens" (13-19) counts as under 18 whatever else the clip has, a sidecar included."""
    _with_line(clip(tmp_path / "young.wav"), "a sidecar must not hide the corpus age")
    (tmp_path / "validated.tsv").write_text(
        "client_id\tpath\tsentence\tage\nkid\tyoung.mp3\tola\tteens\n", encoding="utf-8"
    )
    monkeypatch.setattr(speech, "decode", lambda *a: pytest.fail("decoded an under-18 speaker's clip"))
    row = scan_speech_folder(tmp_path, cfg=SpeechConfig(transcripts="sidecar")).clips[0]
    assert not row.passed and row.reject_reason == "corpus_under_18" and row.age == "teens"


@pytest.mark.parametrize(
    ("label", "young"), [("teens", True), ("TEENS", True), ("16", True), ("twenties", False), (None, False)]
)
def test_under_18_labels(label: str | None, young: bool) -> None:
    assert speech_corpus.under_18_label(label) is young


# ---------------------------------------------------------------------------
# Near-duplicates
# ---------------------------------------------------------------------------


def test_a_re_encoded_quieter_copy_is_a_duplicate_and_a_different_line_is_not(tmp_path: Path) -> None:
    base = np.concatenate([silence(0.2), voice(2.0, seed=1), silence(0.2)])
    write_wav(tmp_path / "a.wav", base)
    noisy = base * 0.5 + np.random.default_rng(0).normal(0, 0.002, len(base)).astype(np.float32)
    write_wav(tmp_path / "b.wav", np.interp(np.arange(0, len(base), 16000 / 22050), np.arange(len(base)), noisy), 22050)
    write_wav(tmp_path / "c.wav", np.concatenate([silence(0.2), voice(2.0, f0=230, seed=7), silence(0.2)]))
    for name in ("a", "b", "c"):
        _with_line(tmp_path / f"{name}.wav")
    summary = scan_speech_folder(tmp_path)
    rows = _by_rel(summary)
    assert speech.hamming(rows["a.wav"].fingerprint, rows["b.wav"].fingerprint) <= 6
    assert speech.hamming(rows["a.wav"].fingerprint, rows["c.wav"].fingerprint) > 12
    dup = rows["a.wav"] if rows["a.wav"].is_duplicate else rows["b.wav"]
    rep = rows["b.wav"] if dup is rows["a.wav"] else rows["a.wav"]
    assert dup.duplicate_of == rep.rel_path and rep.is_representative and rep.score >= dup.score
    assert not rows["c.wav"].is_duplicate
    assert summary.duplicates == 1 and summary.similar_clusters == 1
    assert not any(r.is_duplicate for r in scan_speech_folder(tmp_path, cfg=SpeechConfig(duplicate_distance=-1)).clips)


def test_clips_of_very_different_lengths_are_never_duplicates() -> None:
    a = ClipResult(rel_path="a", abs_path="a", passed=True, fingerprint="0" * 16, duration_s=2.0, score=0.9)
    b = ClipResult(rel_path="b", abs_path="b", passed=True, fingerprint="0" * 16, duration_s=4.0, score=0.5)
    assert speech.mark_duplicates([a, b], 6) == 0 and not b.is_duplicate


# ---------------------------------------------------------------------------
# DNSMOS: fake session, GPU lease, capped arena, pinned hash
# ---------------------------------------------------------------------------


class _Session:
    """Stands in for DNSMOS: records each pass's shape, answers a fixed raw score."""

    def __init__(self, raw: float = 3.0) -> None:
        self.raw, self.shapes = raw, []

    def get_inputs(self):  # noqa: ANN202
        return [type("I", (), {"name": "input_1"})()]

    def run(self, _outputs: object, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        x = feeds["input_1"]
        self.shapes.append(x.shape)
        return [np.full((len(x), 3), self.raw, dtype=np.float32)]


@pytest.fixture
def fake_dnsmos(monkeypatch: pytest.MonkeyPatch) -> _Session:
    session = _Session()
    monkeypatch.setattr(dnsmos, "dnsmos_available", lambda: True)
    monkeypatch.setattr(dnsmos, "get_session", lambda device: session)
    return session


def test_mos_scores_each_passing_clip_in_fixed_size_windows(tmp_path: Path, fake_dnsmos: _Session) -> None:
    _with_line(clip(tmp_path / "a.wav", 2.0, rate=22050))
    _with_line(clip(tmp_path / "long.wav", 18.0), "word " * 60)
    clip(tmp_path / "rejected.wav")  # no transcript: never scored
    rows = _by_rel(scan_speech_folder(tmp_path, cfg=SpeechConfig(mos=True)))
    expected = round(float(dnsmos._P_OVR(3.0)), 3)
    assert rows["a.wav"].mos is not None and rows["a.wav"].mos.ovrl == expected
    assert rows["rejected.wav"].mos is None
    assert "mos" in rows["a.wav"].score_breakdown
    # Every pass is one window of the model's fixed input; a long clip is thinned to MAX_WINDOWS.
    assert set(fake_dnsmos.shapes) == {(1, dnsmos.INPUT_SAMPLES)}
    assert len(fake_dnsmos.shapes) == 1 + dnsmos.MAX_WINDOWS


def test_min_mos_rejects_a_low_score_and_an_unscored_clip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _with_line(clip(tmp_path / "a.wav"))
    monkeypatch.setattr(dnsmos, "dnsmos_available", lambda: True)
    monkeypatch.setattr(dnsmos, "get_session", lambda device: _Session(raw=1.0))
    row = scan_speech_folder(tmp_path, cfg=SpeechConfig(mos=True, min_mos=3.0)).clips[0]
    assert row.reject_reason == "mos" and "min_mos" in row.reject_detail

    def boom(*_a: object) -> None:
        raise RuntimeError("bad window")

    monkeypatch.setattr(dnsmos, "score", boom)
    row = scan_speech_folder(tmp_path, cfg=SpeechConfig(mos=True, min_mos=3.0)).clips[0]
    assert row.reject_reason == "mos" and row.mos_error == "bad window"
    assert scan_speech_folder(tmp_path, cfg=SpeechConfig(mos=True)).clips[0].passed  # no floor: kept unscored


def test_mos_loads_and_runs_under_the_gpu_lease_with_the_capped_arena(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    leases: list[str] = []
    built: dict[str, object] = {}

    class _Recorder:
        name = "recorder"

        def lease(self, *, caller: str, min_vram_mb: int, **_options: object):  # noqa: ANN202
            import contextlib

            leases.append(caller)
            assert min_vram_mb == runtime.VRAM_MB["mos"][caller.rsplit(":", 1)[1]]
            return contextlib.nullcontext()

    model = tmp_path / "dnsmos.onnx"
    model.write_bytes(b"model")
    monkeypatch.setenv(dnsmos.MODEL_ENV, str(model))
    monkeypatch.setattr(dnsmos, "MODEL_SHA256", runtime.sha256(model))
    monkeypatch.setattr(runtime, "providers", lambda device: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(runtime, "_coordinator", _Recorder())
    monkeypatch.setattr(dnsmos, "providers", runtime.providers)
    monkeypatch.setattr(runtime, "_SESSION_CACHE", {})

    def fake_build(path, provs, **kwargs):  # noqa: ANN001, ANN202
        assert leases == ["argus-curator:mos:load"], "the session was built outside the GPU lease"
        built["providers"] = runtime.session_providers(provs)
        return _Session()

    monkeypatch.setattr(runtime, "_build_session", fake_build)
    session = dnsmos.get_session("cuda")
    assert built["providers"][0] == ("CUDAExecutionProvider", runtime.cuda_provider_options())
    assert runtime.cuda_provider_options()["arena_extend_strategy"] == "kSameAsRequested"
    dnsmos.score(session, voice(1.0), 16000, "cuda")
    assert leases == ["argus-curator:mos:load", "argus-curator:mos:run"]


def test_a_model_that_is_not_the_pinned_file_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = tmp_path / "dnsmos.onnx"
    model.write_bytes(b"something else")
    monkeypatch.setenv(dnsmos.MODEL_ENV, str(model))
    monkeypatch.setattr(runtime, "_build_session", lambda *a, **k: pytest.fail("loaded an unpinned model"))
    with pytest.raises(ModelUnavailable, match="not the pinned file") as exc:
        dnsmos.get_session("cpu")
    assert str(tmp_path) not in str(exc.value)


def test_mos_without_the_model_is_refused_before_any_clip_is_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip(tmp_path / "a.wav")
    monkeypatch.setenv(dnsmos.MODEL_ENV, str(tmp_path / "missing.onnx"))
    monkeypatch.setattr(speech, "decode", lambda *a: pytest.fail("decoded before refusing"))
    with pytest.raises(ModelUnavailable, match="DNSMOS"):
        scan_speech_folder(tmp_path, cfg=SpeechConfig(mos=True))


def test_resample_keeps_a_tone_and_its_length() -> None:
    t = np.arange(22050) / 22050
    tone = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    out = dnsmos.resample(tone, 22050, 16000)
    assert len(out) == 16000
    peak = np.argmax(np.abs(np.fft.rfft(out)))
    assert abs(peak - 440) <= 1


@pytest.mark.skipif(
    not (os.environ.get(dnsmos.MODEL_ENV) and runtime.onnxruntime_available()),
    reason="a real DNSMOS run needs the model (argus-curator dnsmos-fetch) and onnxruntime",
)
def test_real_dnsmos_scores_in_range() -> None:  # pragma: no cover - needs the model
    mos = dnsmos.score(dnsmos.get_session("cpu"), voice(3.0), 16000, "cpu")
    assert 1.0 <= mos.ovrl <= 5.0


# ---------------------------------------------------------------------------
# The STT hook (argus-echo#4)
# ---------------------------------------------------------------------------


def test_cer() -> None:
    assert speech.cer("Hello, world!", "hello world") == 0.0
    assert speech.cer("abcd", "abed") == pytest.approx(0.25)
    assert speech.cer("", "") == 0.0 and speech.cer("", "x") == 1.0


def test_a_transcriber_gives_cer_and_max_cer_rejects(tmp_path: Path) -> None:
    _with_line(clip(tmp_path / "good.wav"))
    _with_line(clip(tmp_path / "drift.wav", seed=3))

    calls: list[int] = []

    class _Echo:
        def transcribe(self, samples: np.ndarray, sample_rate: int, language: str | None = None) -> str:
            assert samples.ndim == 1
            calls.append(sample_rate)
            return LINE if len(calls) == 1 else "something else entirely"

    rows = scan_speech_folder(tmp_path, cfg=SpeechConfig(max_cer=0.2), transcriber=_Echo()).clips
    cers = sorted(r.cer for r in rows)
    assert cers[0] == 0.0 and cers[1] > 0.2
    assert sorted(r.reject_reason or "" for r in rows) == ["", "cer"]


def test_max_cer_without_an_stt_engine_is_refused(tmp_path: Path) -> None:
    clip(tmp_path / "a.wav")
    with pytest.raises(ModelUnavailable, match="argus-echo#4"):
        scan_speech_folder(tmp_path, cfg=SpeechConfig(max_cer=0.1))


# ---------------------------------------------------------------------------
# Profiles: speech goes to the speech path only
# ---------------------------------------------------------------------------


def test_a_speech_scan_refuses_an_image_profile(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        scan_speech_folder(tmp_path, TargetProfile())


def test_a_speech_profile_is_still_never_scored_as_an_image(tmp_path: Path) -> None:
    """The image scanner, the manifest row and the export request keep refusing a speech profile."""
    from PIL import Image

    Image.new("RGB", (64, 64)).save(tmp_path / "a.png")
    with pytest.raises(ValidationError, match="curates images"):
        scan_folder(tmp_path, SpeechTargetProfile())
    with pytest.raises(ValidationError, match="curates images"):
        TargetProfile(modality="speech")
    row = {
        "rel_path": "a",
        "abs_path": "/a",
        "exported_path": "a",
        "exported_abs_path": "/a",
        "score": 1,
        "similar_group": 1,
    }
    with pytest.raises(ValidationError):
        ManifestRow(**row, target_profile={"modality": "speech"})
    assert "target_profile" not in ExportRequest.model_fields


def test_scan_request_routes_by_modality_and_refuses_image_checks_on_speech() -> None:
    assert isinstance(ScanRequest(folder="").target_profile, TargetProfile)
    speech_req = ScanRequest(folder="", target_profile={"modality": "speech", "target_category": "accent"})
    assert isinstance(speech_req.target_profile, SpeechTargetProfile) and speech_req.is_speech
    with pytest.raises(ValidationError, match="runs no image checks: turn off faces"):
        ScanRequest(folder="", target_profile={"modality": "speech"}, faces={"enabled": True})
    with pytest.raises(ValidationError, match="turn off consent"):
        ScanRequest(folder="", target_profile={"modality": "speech"}, consent={"record": "consent.json"})
    with pytest.raises(ValidationError):
        ScanRequest(folder="", target_profile={"modality": "speech", "target_category": "identity"})
    with pytest.raises(ValidationError):
        ScanRequest(folder="", target_profile={"modality": "video"})


def test_a_speech_summary_round_trips_through_the_store(tmp_path: Path) -> None:
    _with_line(clip(tmp_path / "src" / "a.wav"))
    _with_line(clip(tmp_path / "src" / "b.wav", seed=4, f0=200))
    summary = scan_speech_folder(tmp_path / "src")
    store = ScanStore(tmp_path / "cache")
    store.save(summary)
    loaded = store.load(summary.scan_id)
    assert loaded is not None and loaded.is_speech and isinstance(loaded.target_profile, SpeechTargetProfile)
    assert [c.rel_path for c in loaded.clips] == ["a.wav", "b.wav"]
    page = store.load_page(summary.scan_id, offset=1, limit=1)
    assert page is not None and [c.rel_path for c in page.clips] == ["b.wav"] and page.returned == 1
    assert store.delete_all() == 1 and store.load(summary.scan_id) is None


def test_a_cancelled_speech_scan_stops(tmp_path: Path) -> None:
    for i in range(3):
        clip(tmp_path / f"{i}.wav")
    with pytest.raises(ScanCancelled):
        scan_speech_folder(tmp_path, cancelled=lambda: True)


# ---------------------------------------------------------------------------
# CLI: scan --modality speech
# ---------------------------------------------------------------------------


def test_cli_speech_scan_saves_the_scan_and_exports_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from typer.testing import CliRunner

    from argus_curator.cli import app

    monkeypatch.setenv("CURATOR_CACHE_DIR", str(tmp_path / "cache"))
    source = tmp_path / "corpus"
    _with_line(clip(source / "a.wav"))
    clip(source / "b.wav", seed=3)
    (tmp_path / "cfg.json").write_text(json.dumps({"require_transcript": False}), encoding="utf-8")
    out = tmp_path / "summary.json"
    args = [
        "scan",
        str(source),
        "--modality",
        "speech",
        "--speech-config",
        str(tmp_path / "cfg.json"),
        "--json",
        str(out),
    ]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "Clips scanned:         2" in result.output and "Passed:                2" in result.output
    summary = ScanSummary.model_validate_json(out.read_text(encoding="utf-8"))
    assert summary.is_speech and len(summary.clips) == 2
    assert ScanStore(tmp_path / "cache").load(summary.scan_id) is not None


@pytest.mark.parametrize(
    "extra", [["--copy-to", "out"], ["--faces"], ["--target-category", "identity"], ["--modality", "video"]]
)
def test_cli_speech_scan_refuses_image_options(tmp_path: Path, extra: list[str]) -> None:
    from typer.testing import CliRunner

    from argus_curator.cli import app

    clip(tmp_path / "a.wav")
    result = CliRunner().invoke(app, ["scan", str(tmp_path), "--modality", "speech", *extra])
    assert result.exit_code == 2, result.output
    assert not (tmp_path / "out").exists()


def test_cli_speech_options_need_the_speech_modality(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from argus_curator.cli import app

    result = CliRunner().invoke(app, ["scan", str(tmp_path), "--mos"])
    assert result.exit_code == 2 and "--modality speech" in result.output


# ---------------------------------------------------------------------------
# Server: POST /scan/folder and its stream with modality speech
# ---------------------------------------------------------------------------


@pytest.fixture
def speech_server(tmp_path: Path) -> tuple[TestClient, Path]:
    source = tmp_path / "corpus"
    _with_line(clip(source / "spk1" / "a.wav"))
    _with_line(clip(source / "spk1" / "b.wav", seed=2, f0=190))
    clip(source / "spk2" / "untranscribed.wav", seed=5)
    exports = tmp_path / "exports"
    exports.mkdir()
    app = create_app(cache_dir=str(tmp_path / "cache"), source_root=str(source), export_root=str(exports))
    return TestClient(app), source


SPEECH_SCAN = {"folder": "", "target_profile": {"modality": "speech"}}


def test_post_scan_with_a_speech_profile_scans_clips(speech_server: tuple[TestClient, Path], monkeypatch) -> None:
    client, _ = speech_server
    from argus_curator import scanner

    monkeypatch.setattr(scanner, "scan_folder", lambda *a, **k: pytest.fail("a speech scan reached the image scanner"))
    resp = client.post("/scan/folder", json=SPEECH_SCAN)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["target_profile"]["modality"] == "speech" and body["results"] == []
    rows = {c["rel_path"]: c for c in body["clips"]}
    assert rows["spk2/untranscribed.wav"]["reject_reason"] == "transcript"
    assert body["total"] == 3 and body["passed"] == 2 and body["reject_reasons"] == {"transcript": 1}
    page = client.get(f"/scan/{body['scan_id']}?offset=0&limit=1").json()
    assert len(page["clips"]) == 1 and page["returned"] == 1


def test_the_speech_stream_sends_clip_chunks_then_the_summary(speech_server: tuple[TestClient, Path]) -> None:
    client, _ = speech_server
    resp = client.post("/scan/folder/stream", json=SPEECH_SCAN)
    assert resp.status_code == 200
    chunks = [
        json.loads(block.split("data: ", 1)[1]) for block in resp.text.split("\n\n") if block.startswith("event: chunk")
    ]
    assert chunks and all(c["modality"] == "speech" for c in chunks)
    assert sorted(r["rel_path"] for c in chunks for r in c["clips"]) == [
        "spk1/a.wav",
        "spk1/b.wav",
        "spk2/untranscribed.wav",
    ]
    complete = json.loads(resp.text.split("event: complete\ndata: ", 1)[1].split("\n", 1)[0])
    assert len(complete["clips"]) == 3
    assert client.get("/scans").json()["scans"][0]["scan_id"] == complete["scan_id"]


def test_speech_scans_refuse_image_checks_and_unconfigured_models(speech_server: tuple[TestClient, Path]) -> None:
    client, _ = speech_server
    resp = client.post("/scan/folder", json={**SPEECH_SCAN, "faces": {"enabled": True}})
    assert resp.status_code == 422 and "runs no image checks" in resp.text
    resp = client.post("/scan/folder", json={**SPEECH_SCAN, "speech": {"max_cer": 0.1}})
    assert resp.status_code == 400 and "argus-echo#4" in resp.text
    resp = client.post("/scan/folder", json={**SPEECH_SCAN, "speech": {"mos": True}})
    assert resp.status_code == 400 and "DNSMOS" in resp.text


def test_a_speech_scan_folder_is_contained(speech_server: tuple[TestClient, Path]) -> None:
    client, _ = speech_server
    resp = client.post("/scan/folder", json={**SPEECH_SCAN, "folder": "../"})
    assert resp.status_code == 400


def test_a_speech_scan_has_no_select_or_export_yet(speech_server: tuple[TestClient, Path]) -> None:
    client, _ = speech_server
    scan_id = client.post("/scan/folder", json=SPEECH_SCAN).json()["scan_id"]
    resp = client.post(f"/scan/{scan_id}/select", json={})
    assert resp.status_code == 400 and "#187" in resp.text
    resp = client.post("/export", json={"scan_id": scan_id, "dest": "out", "selection": ["spk1/a.wav"]})
    assert resp.status_code == 400 and "#187" in resp.text


def test_export_selection_refuses_a_speech_scan(tmp_path: Path) -> None:
    from argus_curator.export import export_selection

    _with_line(clip(tmp_path / "a.wav"))
    summary = scan_speech_folder(tmp_path)
    with pytest.raises(ValueError, match="#187"):
        export_selection(summary, ExportRequest(scan_id=summary.scan_id, dest=str(tmp_path / "out")))
    assert not (tmp_path / "out").exists()


def test_health_and_detectors_report_speech(speech_server: tuple[TestClient, Path]) -> None:
    client, _ = speech_server
    assert client.get("/health").json()["speech_scan"] is True
    detectors = client.get("/detectors").json()
    assert detectors["speech"] is True and isinstance(detectors["dnsmos"], bool)
