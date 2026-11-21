"""Speech export from the CLI (#207): ``scan --modality speech --copy-to/--move-to/--symlink-to``.

The CLI builds an ``ExportRequest`` with ``speech`` and hands it to
``export_selection``, the path ``POST /export`` takes, so these tests check the
flags reach it, that it writes what the server writes, and that a missing
provenance or an image option is refused before a clip is read. Clips and the
speaker model are test_speech_export's synthetic ones: no model or GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from test_speech_export import (  # noqa: F401 - fake_speakers is a fixture
    CLIPS_PER_SPEAKER,
    LINE,
    SPEAKERS,
    _corpus,
    _record,
    _rows,
    fake_speakers,
)
from typer.testing import CliRunner

from argus_curator import cli
from argus_curator.cli import app
from argus_curator.server.app import create_app
from argus_curator.speech_export import SPEECH_MANIFEST_NAME

CORPUS_FLAGS = ["--corpus", "Common Voice pt", "--license", "CC0-1.0"]


@pytest.fixture
def cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("CURATOR_CACHE_DIR", str(tmp_path / "cache"))
    return tmp_path / "cache"


def _run(root: Path, *flags: str) -> object:
    return CliRunner().invoke(app, ["scan", str(root), "--modality", "speech", *flags])


def _comparable(manifest: Path) -> list[dict]:
    """The manifest's rows less the paths that name the export root, which differs between two exports."""
    rows = [r.model_dump(mode="json") for r in _rows(manifest)]
    for row in rows:
        del row["exported_abs_path"], row["transcript_abs_path"]
    return sorted(rows, key=lambda r: r["rel_path"])


@pytest.mark.usefixtures("fake_speakers")
def test_cli_speech_export_writes_what_post_export_writes(tmp_path: Path, cache: Path) -> None:
    root = _corpus(tmp_path / "corpus", SPEAKERS[:6])
    exports = tmp_path / "exports"
    flags = [
        *CORPUS_FLAGS,
        "--speakers",
        "--accent",
        "br,pt",
        "--max-hours-per-speaker",
        "0.001",
        "--held-out",
        "2",
        "--split-part",
        "held_out",
        "--json",
        str(tmp_path / "summary.json"),
    ]
    result = _run(root, "--copy-to", str(exports / "cli"), *flags)
    assert result.exit_code == 0, result.output
    assert "clips with their transcripts (held_out part of the split)" in result.output
    scan_id = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))["scan_id"]

    client = TestClient(create_app(cache_dir=str(cache), source_root=str(root), export_root=str(exports)))
    body = {
        "scan_id": scan_id,
        "dest": "http",
        "split": {"held_out": 2, "group_by": "speaker"},
        "split_part": "held_out",
        "speech": {
            "corpus": "Common Voice pt",
            "license": "CC0-1.0",
            "accents": ["br", "pt"],
            "max_hours_per_speaker": 0.001,
        },
    }
    resp = client.post("/export", json=body)
    assert resp.status_code == 200, resp.text

    cli_rows = _comparable(exports / "cli" / SPEECH_MANIFEST_NAME)
    assert cli_rows and cli_rows == _comparable(exports / "http" / SPEECH_MANIFEST_NAME)
    assert {r["split"] for r in cli_rows} == {"held_out"}
    assert {r["accent"] for r in cli_rows} <= {"br", "pt"}
    assert all(r["corpus"] == "Common Voice pt" and r["provenance"]["kind"] == "stock" for r in cli_rows)
    for row in cli_rows:
        txt = exports / "cli" / row["transcript_path"]
        assert txt.read_text(encoding="utf-8") == LINE + "\n"
        assert txt.read_bytes() == (exports / "http" / row["transcript_path"]).read_bytes()
        assert (exports / "cli" / row["exported_path"]).is_file()
    assert (exports / "cli" / "curation_report.csv").is_file()


def test_cli_speech_export_parts_are_speaker_disjoint(tmp_path: Path, cache: Path) -> None:
    root = _corpus(tmp_path / "corpus", SPEAKERS[:6])
    for part in ("train", "held_out"):
        flags = ["--held-out-fraction", "0.34", "--split-method", "random", "--split-seed", "7", "--split-part", part]
        result = _run(root, "--symlink-to", str(tmp_path / part), *CORPUS_FLAGS, *flags)
        assert result.exit_code == 0, result.output
    train = _rows(tmp_path / "train" / SPEECH_MANIFEST_NAME)
    held = _rows(tmp_path / "held_out" / SPEECH_MANIFEST_NAME)
    assert train and held and len(train) + len(held) == 6 * CLIPS_PER_SPEAKER
    assert not {r.speaker_id for r in train} & {r.speaker_id for r in held}
    assert all(Path(r.exported_abs_path).is_symlink() for r in train)


def test_cli_speech_export_max_keep_and_speaker_filter(tmp_path: Path, cache: Path) -> None:
    root = _corpus(tmp_path / "corpus", SPEAKERS[:4])
    result = _run(root, "--copy-to", str(tmp_path / "out"), *CORPUS_FLAGS, "--speaker-id", "s1,s2", "--max-keep", "4")
    assert result.exit_code == 0, result.output
    rows = _rows(tmp_path / "out" / SPEECH_MANIFEST_NAME)
    assert len(rows) == 4 and {r.speaker_id for r in rows} == {"s1", "s2"}


def test_cli_speech_export_moves_clips_out_of_the_source(tmp_path: Path, cache: Path) -> None:
    """The CLI is your own shell: a move needs no gate, as with images."""
    root = _corpus(tmp_path / "corpus", SPEAKERS[:1])
    result = _run(root, "--move-to", str(tmp_path / "out"), *CORPUS_FLAGS)
    assert result.exit_code == 0, result.output
    rows = _rows(tmp_path / "out" / SPEECH_MANIFEST_NAME)
    assert len(rows) == CLIPS_PER_SPEAKER
    assert all(r.abs_path == r.exported_abs_path and Path(r.abs_path).is_file() for r in rows)
    assert not list((root / "clips").glob("*.wav"))


@pytest.mark.usefixtures("fake_speakers")
def test_cli_speech_export_under_a_voice_consent_record(tmp_path: Path, cache: Path) -> None:
    root = _corpus(tmp_path / "corpus", SPEAKERS[:3])
    _record(root / "consent.json", [root / "clips" / "s2_0.wav"])
    consent = ["--voice-consent", "consent.json", "--consent-delivery", "neutral"]
    result = _run(root, "--copy-to", str(tmp_path / "out"), "--speakers", *consent)
    assert result.exit_code == 0, result.output
    rows = _rows(tmp_path / "out" / SPEECH_MANIFEST_NAME)
    assert {r.speaker_label for r in rows} == {"s2"} and len(rows) == CLIPS_PER_SPEAKER
    assert all(r.provenance.kind == "consented" and r.consent_release_sha256 == "a" * 64 for r in rows)
    # --voice-consent runs the second-voice check's windows in the scan (#217) and reports its measurement.
    report = (tmp_path / "out" / "curation_report.csv").read_text(encoding="utf-8")
    assert "second_voice_distance" in report.splitlines()[0]
    scan = json.loads(next(cache.glob("*.json")).read_text(encoding="utf-8"))
    assert scan["speech_config"]["second_voice"] is True

    strict = _run(
        root, "--copy-to", str(tmp_path / "strict"), "--speakers", *consent, "--second-voice-max-distance", "0"
    )
    assert strict.exit_code == 2 and "second_voice_max_distance" in strict.output

    # The ceiling is the CLI's to set (#209), and checked as the library checks it.
    ceiling = ["--voice-consent-max-distance", "0"]
    bad = _run(root, "--copy-to", str(tmp_path / "bad"), "--speakers", *consent, *ceiling)
    assert bad.exit_code == 2 and "consent_max_distance" in bad.output

    refused = _run(root, "--copy-to", str(tmp_path / "late"), "--speakers", *consent[:2], "--consent-delivery", "angry")
    assert refused.exit_code == 2 and "consent refused" in refused.output
    assert not (tmp_path / "late" / SPEECH_MANIFEST_NAME).exists()


@pytest.mark.parametrize(
    ("flags", "says"),
    [
        ([], "--corpus and --license for a licensed corpus"),
        (["--corpus", "Common Voice pt"], "--corpus and --license go together"),
        (["--voice-consent", "consent.json"], "needs --consent-delivery"),
        ([*CORPUS_FLAGS, "--voice-consent", "c.json", "--consent-delivery", "neutral"], "not both"),
        (["--voice-consent", "missing.json", "--consent-delivery", "neutral", "--speakers"], "no such file"),
        ([*CORPUS_FLAGS, "--voice-consent-max-distance", "0.2"], "needs --voice-consent"),
        ([*CORPUS_FLAGS, "--second-voice-max-distance", "0.2"], "--second-voice-max-distance needs --voice-consent"),
        ([*CORPUS_FLAGS, "--split-part", "held_out"], "need a split"),
        ([*CORPUS_FLAGS, "--held-out", "1", "--held-out-fraction", "0.5"], "not both"),
        ([*CORPUS_FLAGS, "--exclude-speaker-mismatch"], "add --speakers"),
        ([*CORPUS_FLAGS, "--min-hours-per-speaker", "2", "--max-hours-per-speaker", "1"], "min_hours_per_speaker"),
        ([*CORPUS_FLAGS, "--held-out", "1", "--split-part", "test"], "split_part"),
    ],
)
def test_cli_speech_export_refuses_before_any_clip_is_read(
    tmp_path: Path, cache: Path, monkeypatch: pytest.MonkeyPatch, flags: list[str], says: str
) -> None:
    def no_scan(*_a: object, **_k: object) -> None:
        raise AssertionError("scanned a refused export")

    monkeypatch.setattr(cli, "_scan_speech", no_scan)
    result = _run(tmp_path, "--copy-to", str(tmp_path / "out"), *flags)
    assert result.exit_code == 2, result.output
    assert says in result.output
    assert not (tmp_path / "out").exists()


def test_cli_speech_export_with_no_provenance_names_the_flags(tmp_path: Path, cache: Path) -> None:
    root = _corpus(tmp_path / "corpus", SPEAKERS[:1])
    result = _run(root, "--copy-to", str(tmp_path / "out"))
    assert result.exit_code == 2
    assert "--voice-consent and --consent-delivery for a consented voice" in result.output
    assert not (tmp_path / "out").exists()


def test_cli_speech_export_split_that_cannot_be_made_is_refused_after_the_scan(tmp_path: Path, cache: Path) -> None:
    root = _corpus(tmp_path / "corpus", SPEAKERS[:1])  # one speaker: no speaker-disjoint split
    result = _run(root, "--copy-to", str(tmp_path / "out"), *CORPUS_FLAGS, "--held-out", "1")
    assert result.exit_code == 2 and "Error:" in result.output
    assert "Scan saved as" in result.output  # the scan is kept for a re-export
    assert not (tmp_path / "out" / SPEECH_MANIFEST_NAME).exists()


@pytest.mark.parametrize(
    "flags",
    [["--pose", "frontal"], ["--garment", "dress"], ["--nsfw-threshold", "0.5"], ["--carry-sidecars"], ["--landmarks"]],
)
def test_cli_speech_export_refuses_image_only_flags(tmp_path: Path, cache: Path, flags: list[str]) -> None:
    result = _run(tmp_path, "--copy-to", str(tmp_path / "out"), *CORPUS_FLAGS, *flags)
    assert result.exit_code == 2 and "image" in result.output and flags[0] in result.output
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("flags", [CORPUS_FLAGS, ["--balance-accents"], ["--held-out", "2"]])
def test_speech_export_flags_need_an_export(tmp_path: Path, cache: Path, flags: list[str]) -> None:
    result = _run(tmp_path, *flags)
    assert result.exit_code == 2 and "add --copy-to, --move-to or --symlink-to" in result.output


@pytest.mark.parametrize("flags", [CORPUS_FLAGS, ["--accent", "br"], ["--held-out-fraction", "0.1"]])
def test_speech_export_flags_need_the_speech_modality(tmp_path: Path, flags: list[str]) -> None:
    result = CliRunner().invoke(app, ["scan", str(tmp_path), "--copy-to", str(tmp_path / "out"), *flags])
    assert result.exit_code == 2 and "need --modality speech" in result.output
    assert not (tmp_path / "out").exists()
