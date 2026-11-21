"""Spoken language ID (#208): checking a speech corpus's language labels with a model.

No model or GPU is needed: the language-ID model is replaced by a fake that
reads a clip's language off its peak level (each language is written at its
own level), as test_speech_export's fake speaker model reads its voice. The
features are pinned against SpeechBrain's ``Fbank``, and the lease, the arena
and the pin are checked with a recording coordinator and a fake session.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from speech_fixtures import clip, voice
from typer.testing import CliRunner

from argus_curator import language_id, runtime, speech
from argus_curator.cli import app
from argus_curator.export import export_selection
from argus_curator.models import ClipResult, ExportRequest, ScanSummary, SpeechConfig, SpeechManifestRow
from argus_curator.runtime import ModelUnavailable
from argus_curator.speech import scan_speech_folder
from argus_curator.speech_export import SPEECH_MANIFEST_NAME
from argus_curator.speech_select import plan_speech_selection
from argus_curator.store import ScanStore

LINE = "the quick brown fox jumps over"
CORPUS = {"corpus": "Common Voice pt", "license": "CC0-1.0"}

# (client_id, corpus locale, the language actually spoken). s3 is labelled pt but speaks Spanish.
PEOPLE = [("s1", "pt", "pt"), ("s2", "pt-BR", "pt"), ("s3", "pt", "es"), ("s4", "es", "es")]
CLIPS_PER_SPEAKER = 2
_SPOKEN = {"pt": 0.2, "es": 0.3, "hr": 0.4, "en": 0.5}


def _probs(code: str, p: float = 0.95) -> np.ndarray:
    out = np.full(language_id.N_LANGUAGES, (1.0 - p) / (language_id.N_LANGUAGES - 1))
    out[language_id.CODES.index(code)] = p
    return out


@pytest.fixture
def fake_lid(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, ...]]:
    """The language-ID model, replaced: a clip's language is read off its peak level, at p=0.95."""
    shapes: list[tuple[int, ...]] = []
    by_level = {level: code for code, level in _SPOKEN.items()}

    def identify(_session: object, mono: np.ndarray, rate: int, device: str) -> np.ndarray:
        shapes.append(language_id.windows(mono, rate).shape)
        return _probs(by_level[round(float(np.max(np.abs(mono))), 1)])

    monkeypatch.setattr(language_id, "problem", lambda: None)
    monkeypatch.setattr(language_id, "get_session", lambda device: object())
    monkeypatch.setattr(language_id, "identify", identify)
    return shapes


def _corpus(root: Path, people=PEOPLE) -> Path:
    rows = ["client_id\tpath\tsentence\taccents\tlocale\tage\tgender"]
    seed = 0
    for client, locale, spoken in people:
        for i in range(CLIPS_PER_SPEAKER):
            seed += 1
            name = f"{client}_{i}"
            clip(root / "clips" / f"{name}.wav", 2.0, seed=seed, f0=120 + 10 * seed, level=_SPOKEN[spoken])
            rows.append(f"{client}\t{name}.mp3\t{LINE}\t\t{locale}\tthirties\tfemale")
    (root / "validated.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return root


def _scan(root: Path, **cfg: object) -> ScanSummary:
    return scan_speech_folder(root, {"modality": "speech", "target_category": "language"}, SpeechConfig(**cfg))


# ---------------------------------------------------------------------------
# Features, fixed-shape windows, the lease, the pin
# ---------------------------------------------------------------------------


def test_fbank_matches_speechbrain() -> None:
    """Pinned against speechbrain.lobes.features.Fbank(n_mels=60, left_frames=0, right_frames=0)."""
    t = np.arange(1600) / 16000
    x = (0.3 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.sin(2 * np.pi * 2500 * t)).astype(np.float32)
    feats = language_id.fbank(x)
    assert feats.shape == (11, language_id.N_MELS)
    expected = [-43.836, 25.452, -36.651, -27.454, -23.05, -31.469, -32.629]
    np.testing.assert_allclose(feats[3, [0, 10, 20, 30, 40, 50, 59]], expected, atol=2e-3)


@pytest.mark.parametrize(("seconds", "rate"), [(0.4, 16000), (2.0, 22050), (19.5, 48000), (110.0, 16000)])
def test_every_clip_is_one_pass_of_the_same_shape(seconds: float, rate: int) -> None:
    """Short clips are tiled, long ones sampled: the capped arena never grows with a clip's length."""
    batch = language_id.windows(voice(seconds, rate), rate)
    assert batch.shape == (language_id.WINDOWS, language_id.WINDOW_FRAMES, language_id.N_MELS)
    assert batch.dtype == np.float32


def _converted(tmp_path: Path, payload: bytes = b"model", **record: object) -> Path:
    model = tmp_path / "lid.onnx"
    model.write_bytes(payload)
    body = {"sources": language_id.SOURCE_SHA256, "onnx_sha256": runtime.sha256(model), **record}
    language_id.record_path(model).write_text(json.dumps(body))
    return model


def test_language_model_loads_and_runs_under_the_gpu_lease_with_the_capped_arena(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    leases: list[str] = []
    built: dict[str, object] = {}

    class _Recorder:
        name = "recorder"

        def lease(self, *, caller: str, min_vram_mb: int, **_options: object):  # noqa: ANN202
            import contextlib

            leases.append(caller)
            assert min_vram_mb == runtime.VRAM_MB["language"][caller.rsplit(":", 1)[1]]
            return contextlib.nullcontext()

    class _Session:
        def get_inputs(self):  # noqa: ANN202
            return [type("I", (), {"name": "feats"})()]

        def run(self, _outputs: object, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
            x = feeds["feats"]
            assert x.shape == (language_id.WINDOWS, language_id.WINDOW_FRAMES, language_id.N_MELS)
            return [np.log(np.stack([_probs("pt")] * len(x))).astype(np.float32)]

    monkeypatch.setenv(language_id.MODEL_ENV, str(_converted(tmp_path)))
    monkeypatch.setattr(runtime, "providers", lambda device: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(runtime, "_coordinator", _Recorder())
    monkeypatch.setattr(language_id, "providers", runtime.providers)
    monkeypatch.setattr(runtime, "_SESSION_CACHE", {})

    def fake_build(path, provs, **kwargs):  # noqa: ANN001, ANN202
        assert leases == ["argus-curator:language:load"], "the session was built outside the GPU lease"
        built["providers"] = runtime.session_providers(provs)
        return _Session()

    monkeypatch.setattr(runtime, "_build_session", fake_build)
    session = language_id.get_session("cuda")
    assert built["providers"][0] == ("CUDAExecutionProvider", runtime.cuda_provider_options())
    probs = language_id.identify(session, voice(1.0), 16000, "cuda")
    assert leases == ["argus-curator:language:load", "argus-curator:language:run"]
    assert probs.shape == (language_id.N_LANGUAGES,) and abs(float(probs.sum()) - 1.0) < 1e-6
    assert language_id.CODES[int(np.argmax(probs))] == "pt"


@pytest.mark.parametrize(
    "spoil",
    [
        "no-record",
        "other-sources",
        "other-file",
    ],
)
def test_a_language_model_not_converted_from_the_pinned_sources_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spoil: str
) -> None:
    model = _converted(tmp_path)
    if spoil == "no-record":
        language_id.record_path(model).unlink()
    elif spoil == "other-sources":
        sources = {**language_id.SOURCE_SHA256, "classifier.ckpt": "0" * 64}
        language_id.record_path(model).write_text(
            json.dumps({"sources": sources, "onnx_sha256": runtime.sha256(model)})
        )
    else:
        model.write_bytes(b"something else")
    monkeypatch.setenv(language_id.MODEL_ENV, str(model))
    monkeypatch.setattr(runtime, "_build_session", lambda *a, **k: pytest.fail("loaded an unpinned model"))
    with pytest.raises(ModelUnavailable, match="lid-convert") as exc:
        language_id.get_session("cpu")
    assert str(tmp_path) not in str(exc.value)


def test_language_id_without_the_model_is_refused_before_any_clip_is_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip(tmp_path / "a.wav")
    monkeypatch.setenv(language_id.MODEL_ENV, str(tmp_path / "missing.onnx"))
    monkeypatch.setattr(speech, "decode", lambda *a: pytest.fail("decoded before refusing"))
    with pytest.raises(ModelUnavailable, match="lid-convert"):
        scan_speech_folder(tmp_path, cfg=SpeechConfig(language_id=True))


def test_lid_convert_refuses_sources_that_are_not_the_pinned_files(tmp_path: Path) -> None:
    """The source check runs before torch is imported, so it is testable without the extra."""
    for name in language_id.SOURCE_SHA256:
        (tmp_path / name).write_bytes(b"not the checkpoint")
    with pytest.raises(ModelUnavailable, match="not the pinned"):
        language_id.export_onnx(tmp_path, tmp_path / "out.onnx")
    assert not (tmp_path / "out.onnx").exists()


def test_the_label_order_is_voxlingua107s() -> None:
    assert language_id.N_LANGUAGES == 107 and len(set(language_id.CODES)) == 107
    assert language_id.CODES[:3] == ("ab", "af", "am") and language_id.CODES[-1] == "zh"
    assert language_id.CODES[language_id.CODES.index("pt")] == "pt"


# ---------------------------------------------------------------------------
# Judging the corpus label
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "code"),
    [
        ("pt", "pt"),
        ("pt-BR", "pt"),
        ("pt_PT", "pt"),
        ("zh-CN", "zh"),
        ("he", "iw"),
        ("nb-NO", "no"),
        ("Portuguese", "pt"),
        ("fy-NL", None),
        ("yue", None),
        ("", None),
        (None, None),
    ],
)
def test_model_code_reads_locales_aliases_and_names(label: str | None, code: str | None) -> None:
    assert language_id.model_code(label) == code


def _row(language: str | None) -> ClipResult:
    return ClipResult(rel_path="a.wav", abs_path="/a.wav", passed=True, language=language)


def test_a_dispute_needs_a_known_label_another_language_and_the_confidence() -> None:
    row = _row("pt-BR")
    language_id.judge(row, _probs("es", 0.9), 0.8)
    assert row.language_mismatch and row.language_predicted == "es" and row.language_confidence == 0.9
    assert row.language == "pt-BR"  # never relabelled
    assert row.language_label_confidence is not None and row.language_label_confidence < 0.01

    unsure = _row("pt")
    language_id.judge(unsure, _probs("es", 0.6), 0.8)
    assert not unsure.language_mismatch and unsure.language_predicted == "es"

    agrees = _row("pt")
    language_id.judge(agrees, _probs("pt", 0.99), 0.8)
    assert not agrees.language_mismatch and agrees.language_label_confidence == 0.99

    unknown = _row("fy-NL")  # a language the model doesn't know: no judgement
    language_id.judge(unknown, _probs("nl", 0.99), 0.8)
    assert not unknown.language_mismatch and unknown.language_label_confidence is None

    unlabelled = _row(None)
    language_id.judge(unlabelled, _probs("en", 0.99), 0.8)
    assert not unlabelled.language_mismatch and unlabelled.language_predicted == "en"


@pytest.mark.parametrize(("label", "heard"), [("hr", "sr"), ("bs", "hr"), ("nn", "no"), ("ur", "hi"), ("ms", "id")])
def test_languages_the_ear_cannot_split_are_not_a_dispute(label: str, heard: str) -> None:
    row = _row(label)
    language_id.judge(row, _probs(heard, 0.99), 0.8)
    assert not row.language_mismatch


# ---------------------------------------------------------------------------
# The scan, select and export
# ---------------------------------------------------------------------------


def test_a_scan_flags_a_clip_whose_corpus_language_is_wrong_and_keeps_its_label(
    tmp_path: Path, fake_lid: list[tuple[int, ...]]
) -> None:
    summary = _scan(_corpus(tmp_path), language_id=True)
    assert set(fake_lid) == {(language_id.WINDOWS, language_id.WINDOW_FRAMES, language_id.N_MELS)}
    flagged = {c.speaker for c in summary.clips if c.language_mismatch}
    assert flagged == {"s3"} and summary.language_mismatches == CLIPS_PER_SPEAKER
    for c in summary.clips:
        assert c.passed, "a dispute alone never rejects"
        assert c.language_confidence == 0.95 and c.language_error is None
    s3 = [c for c in summary.clips if c.speaker == "s3"]
    assert {c.language for c in s3} == {"pt"} and {c.language_predicted for c in s3} == {"es"}


def test_a_scan_without_language_id_judges_nothing(tmp_path: Path) -> None:
    summary = _scan(_corpus(tmp_path))
    assert summary.language_mismatches == 0
    assert all(c.language_predicted is None and not c.language_mismatch for c in summary.clips)


def test_rejecting_a_disputed_clip_is_opt_in_and_names_the_check(tmp_path: Path, fake_lid: object) -> None:
    summary = _scan(_corpus(tmp_path), language_id=True, reject_language_mismatch=True)
    rejected = [c for c in summary.clips if not c.passed]
    assert {c.speaker for c in rejected} == {"s3"}
    for c in rejected:
        assert c.reject_reason == "language" and c.failed_checks == ["language"]
        assert "'es'" in (c.reject_detail or "") and "'pt'" in (c.reject_detail or "")
    assert summary.reject_reasons == {"language": CLIPS_PER_SPEAKER}


def test_a_clip_the_model_cannot_read_records_why_and_still_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*_a: object) -> np.ndarray:
        raise ValueError("bad window")

    monkeypatch.setattr(language_id, "problem", lambda: None)
    monkeypatch.setattr(language_id, "get_session", lambda device: object())
    monkeypatch.setattr(language_id, "identify", broken)
    summary = _scan(_corpus(tmp_path), language_id=True, reject_language_mismatch=True)
    assert all(c.passed and c.language_error == "bad window" and not c.language_mismatch for c in summary.clips)


def test_exclude_language_mismatch_at_select_and_export(tmp_path: Path, fake_lid: object) -> None:
    summary = _scan(_corpus(tmp_path / "src"), language_id=True)
    req = ExportRequest(dest=".", speech={**CORPUS, "exclude_language_mismatch": True})
    plan = plan_speech_selection(summary.clips, req)
    assert {rel for rel, why in plan.keep_reason.items() if why == "language-mismatch"} == {
        c.rel_path for c in summary.clips if c.speaker == "s3"
    }

    out = tmp_path / "out"
    export_selection(
        summary,
        ExportRequest(scan_id=summary.scan_id, dest=str(out), speech={**CORPUS, "exclude_language_mismatch": True}),
    )
    rows = [
        SpeechManifestRow.model_validate_json(line)
        for line in (out / SPEECH_MANIFEST_NAME).read_text(encoding="utf-8").splitlines()
    ]
    assert {r.speaker_label for r in rows} == {"s1", "s2", "s4"}
    # The speech manifest row is unchanged: the model's reading lives in the scan and the report.
    assert not any(
        k.startswith("language_") for k in json.loads((out / SPEECH_MANIFEST_NAME).read_text().splitlines()[0])
    )
    with (out / "curation_report.csv").open(encoding="utf-8") as fh:
        report = {r["rel_path"]: r for r in csv.DictReader(fh)}
    s3 = [r for r in report.values() if r["rel_path"].startswith("clips/s3")]
    assert s3 and all(r["language_mismatch"] == "True" and r["language_predicted"] == "es" for r in s3)
    assert all(r["keep_reason"] == "language-mismatch" for r in s3)
    assert all(r["language"] == "pt" for r in s3)


def test_exclude_language_mismatch_is_refused_on_a_scan_that_never_ran_it(tmp_path: Path) -> None:
    summary = _scan(_corpus(tmp_path / "src"))
    req = ExportRequest(
        scan_id=summary.scan_id, dest=str(tmp_path / "out"), speech={**CORPUS, "exclude_language_mismatch": True}
    )
    with pytest.raises(ValueError, match="language_id"):
        export_selection(summary, req)
    assert not (tmp_path / "out").exists()


def test_language_probabilities_are_never_stored(tmp_path: Path, fake_lid: object) -> None:
    """Only the top language and two probabilities are kept per clip; the 107-way vector is dropped."""
    summary = _scan(_corpus(tmp_path / "src"), language_id=True)
    store = ScanStore(tmp_path / "cache")
    store.save(summary)
    on_disk = json.loads((tmp_path / "cache" / f"{summary.scan_id}.json").read_text(encoding="utf-8"))

    def longest_list(node: object) -> int:
        if isinstance(node, list):
            return max([len(node), *(longest_list(v) for v in node)])
        if isinstance(node, dict):
            return max([0, *(longest_list(v) for v in node.values())])
        return 0

    assert all(longest_list(c) < language_id.N_LANGUAGES for c in on_disk["clips"])
    assert summary.embeddings == {}


def test_cli_language_id_and_its_export_flag(tmp_path: Path, fake_lid: object, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CURATOR_CACHE_DIR", str(tmp_path / "cache"))
    root = _corpus(tmp_path / "corpus")
    runner = CliRunner()
    refused = runner.invoke(
        app,
        [
            "scan",
            str(root),
            "--modality",
            "speech",
            "--copy-to",
            str(tmp_path / "x"),
            "--exclude-language-mismatch",
            "--corpus",
            "Common Voice pt",
            "--license",
            "CC0-1.0",
        ],
    )
    assert refused.exit_code == 2 and "--language-id" in refused.output
    result = runner.invoke(
        app,
        [
            "scan",
            str(root),
            "--modality",
            "speech",
            "--language-id",
            "-v",
            "--copy-to",
            str(tmp_path / "out"),
            "--exclude-language-mismatch",
            "--corpus",
            "Common Voice pt",
            "--license",
            "CC0-1.0",
        ],
    )
    assert result.exit_code == 0, result.output
    assert f"Language disputes:     {CLIPS_PER_SPEAKER}" in result.output
    assert "language=pt?es(0.95)" in result.output
    rows = (tmp_path / "out" / SPEECH_MANIFEST_NAME).read_text(encoding="utf-8").splitlines()
    assert len(rows) == 3 * CLIPS_PER_SPEAKER
    image_only = runner.invoke(app, ["scan", str(root), "--language-id"])
    assert image_only.exit_code == 2 and "--modality speech" in image_only.output


# ---------------------------------------------------------------------------
# The real model (skipped unless converted here)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not language_id.lid_available(), reason="needs the converted model (argus-curator lid-convert)")
def test_real_language_model_reads_a_clip() -> None:  # pragma: no cover - needs the model
    session = language_id.get_session("cpu")
    probs = language_id.identify(session, voice(4.0), 16000, "cpu")
    assert probs.shape == (language_id.N_LANGUAGES,) and abs(float(probs.sum()) - 1.0) < 1e-5
