"""Speech export (#187): speakers, accents, the speaker-disjoint split, provenance and the speech manifest.

Clips are synthesised (tests/speech_fixtures.py) and labelled through a Common
Voice ``validated.tsv``. No model or GPU is needed: the speaker model is
replaced by synthetic embeddings, one random unit vector per voice, which a
clip's peak level encodes (each voice is written at its own level).
"""

from __future__ import annotations

import json
import os
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from speech_fixtures import clip, voice

from argus_curator import runtime, speakers, speech, speech_select
from argus_curator.consent import ConsentRefused
from argus_curator.export import export_selection
from argus_curator.models import (
    SECOND_VOICE_MAX_DISTANCE,
    VOICE_CONSENT_MAX_DISTANCE,
    ClipResult,
    ExportRequest,
    ScanSummary,
    SpeechConfig,
    SpeechManifestRow,
    SpeechSelection,
    SplitConfig,
)
from argus_curator.runtime import ModelUnavailable
from argus_curator.selection import SplitError, split_held_out
from argus_curator.server.app import create_app
from argus_curator.speech import scan_speech_folder
from argus_curator.speech_export import SPEECH_MANIFEST_NAME, export_speech, voice_consent_gate
from argus_curator.store import ScanStore

LINE = "the quick brown fox jumps over"
CORPUS = {"corpus": "Common Voice pt", "license": "CC0-1.0"}

# (client_id, accent, gender, age, voice): nine speakers over three accents and both sexes.
SPEAKERS = [
    ("s1", "br", "female", "twenties", 1),
    ("s2", "br", "male", "thirties", 2),
    ("s3", "br", "female", "", 3),  # no age label: recorded unknown, never guessed
    ("s4", "pt", "male", "forties", 4),
    ("s5", "pt", "female", "twenties", 5),
    ("s6", "pt", "male", "thirties", 6),
    ("s7", "ao", "female", "fifties", 7),
    ("s8", "ao", "male", "twenties", 8),
    ("s9", "ao", "female", "thirties", 9),
]
CLIPS_PER_SPEAKER = 3


def _level(voice_id: int) -> float:
    return round(0.08 + 0.03 * voice_id, 2)


def _vector(voice_id: int) -> np.ndarray:
    v = np.random.default_rng(1000 + voice_id).standard_normal(speakers.EMBEDDING_DIM)
    return (v / np.linalg.norm(v)).astype(np.float32)


@pytest.fixture
def fake_speakers(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, ...]]:
    """The speaker model, replaced: a clip's embedding is its voice's vector, read off its peak level."""
    shapes: list[tuple[int, ...]] = []
    levels = {_level(v): v for v in range(1, 30)}

    def embed(_session: object, mono: np.ndarray, rate: int, device: str) -> np.ndarray:
        shapes.append(speakers.windows(mono, rate).shape)
        peak = round(float(np.max(np.abs(mono))), 2)
        return _vector(levels[peak])

    def embed_segments(_session: object, mono: np.ndarray, rate: int, speech: object, device: str) -> np.ndarray:
        """The second-voice windows (#217): the real windowing over the scan's VAD mask, each its voice's vector."""
        segments = speakers.segment_features(mono, rate, speech)  # type: ignore[arg-type]
        shapes.append(segments.shape)
        peak = round(float(np.max(np.abs(mono))), 2)
        return np.tile(_vector(levels[peak]), (len(segments), 1)).astype(np.float16)

    monkeypatch.setattr(speakers, "problem", lambda: None)
    monkeypatch.setattr(speakers, "get_session", lambda device: object())
    monkeypatch.setattr(speakers, "embed", embed)
    monkeypatch.setattr(speakers, "embed_segments", embed_segments)
    return shapes


def _corpus(root: Path, people=SPEAKERS, *, extra_rows: list[str] | None = None) -> Path:
    rows = ["client_id\tpath\tsentence\taccents\tlocale\tage\tgender"]
    seed = 0
    for client, accent, gender, age, voice_id in people:
        for i in range(CLIPS_PER_SPEAKER):
            seed += 1
            name = f"{client}_{i}"
            clip(root / "clips" / f"{name}.wav", 2.0, seed=seed, f0=110 + 9 * voice_id, level=_level(voice_id))
            rows.append(f"{client}\t{name}.mp3\t{LINE}\t{accent}\tpt\t{age}\t{gender}")
    rows.extend(extra_rows or [])
    (root / "validated.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return root


def _scan(root: Path, **cfg: object) -> ScanSummary:
    return scan_speech_folder(root, {"modality": "speech", "target_category": "accent"}, SpeechConfig(**cfg))


def _rows(path: Path) -> list[SpeechManifestRow]:
    return [SpeechManifestRow.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _export(summary: ScanSummary, dest: Path, **fields: object) -> object:
    speech_fields = {**CORPUS, **fields.pop("speech", {})}  # type: ignore[arg-type]
    return export_selection(
        summary, ExportRequest(scan_id=summary.scan_id, dest=str(dest), speech=speech_fields, **fields)
    )


# ---------------------------------------------------------------------------
# The speaker model: features, fixed-shape windows, the lease, the pin
# ---------------------------------------------------------------------------


def test_fbank_matches_kaldi() -> None:
    """Pinned against torchaudio.compliance.kaldi.fbank(x * 32768, 80 bins, hamming, dither 0)."""
    t = np.arange(1600) / 16000
    x = (0.3 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.sin(2 * np.pi * 2500 * t)).astype(np.float32)
    feats = speakers.fbank(x)
    assert feats.shape == (8, speakers.N_MELS)
    expected = [13.516, 13.581, 13.883, 11.962, 14.035, 15.913, 13.206, 12.341, 12.153]
    np.testing.assert_allclose(feats[3, [0, 10, 20, 30, 40, 50, 60, 70, 79]], expected, atol=2e-3)


@pytest.mark.parametrize(("seconds", "rate"), [(0.4, 16000), (2.0, 22050), (19.5, 48000)])
def test_every_clip_is_one_pass_of_the_same_shape(seconds: float, rate: int) -> None:
    """Short clips are tiled, long ones sampled: the capped arena never grows with a clip's length."""
    batch = speakers.windows(voice(seconds, rate), rate)
    assert batch.shape == (speakers.WINDOWS, speakers.WINDOW_FRAMES, speakers.N_MELS)
    assert batch.dtype == np.float32


def test_speaker_model_loads_and_runs_under_the_gpu_lease_with_the_capped_arena(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    leases: list[str] = []
    built: dict[str, object] = {}

    class _Recorder:
        name = "recorder"

        def lease(self, *, caller: str, min_vram_mb: int, **_options: object):  # noqa: ANN202
            import contextlib

            leases.append(caller)
            assert min_vram_mb == runtime.VRAM_MB["speakers"][caller.rsplit(":", 1)[1]]
            return contextlib.nullcontext()

    class _Session:
        def get_inputs(self):  # noqa: ANN202
            return [type("I", (), {"name": "feats"})()]

        def run(self, _outputs: object, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
            x = feeds["feats"]
            assert x.shape == (speakers.WINDOWS, speakers.WINDOW_FRAMES, speakers.N_MELS)
            return [np.ones((len(x), speakers.EMBEDDING_DIM), dtype=np.float32)]

    model = tmp_path / "ecapa.onnx"
    model.write_bytes(b"model")
    monkeypatch.setenv(speakers.MODEL_ENV, str(model))
    monkeypatch.setattr(speakers, "MODEL_SHA256", runtime.sha256(model))
    monkeypatch.setattr(runtime, "providers", lambda device: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(runtime, "_coordinator", _Recorder())
    monkeypatch.setattr(speakers, "providers", runtime.providers)
    monkeypatch.setattr(runtime, "_SESSION_CACHE", {})

    def fake_build(path, provs, **kwargs):  # noqa: ANN001, ANN202
        assert leases == ["argus-curator:speakers:load"], "the session was built outside the GPU lease"
        built["providers"] = runtime.session_providers(provs)
        return _Session()

    monkeypatch.setattr(runtime, "_build_session", fake_build)
    session = speakers.get_session("cuda")
    assert built["providers"][0] == ("CUDAExecutionProvider", runtime.cuda_provider_options())
    vec = speakers.embed(session, voice(1.0), 16000, "cuda")
    assert leases == ["argus-curator:speakers:load", "argus-curator:speakers:run"]
    assert vec.shape == (speakers.EMBEDDING_DIM,) and abs(float(np.linalg.norm(vec)) - 1.0) < 1e-5


def test_a_speaker_model_that_is_not_the_pinned_file_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "ecapa.onnx"
    model.write_bytes(b"something else")
    monkeypatch.setenv(speakers.MODEL_ENV, str(model))
    monkeypatch.setattr(runtime, "_build_session", lambda *a, **k: pytest.fail("loaded an unpinned model"))
    with pytest.raises(ModelUnavailable, match="not the pinned file") as exc:
        speakers.get_session("cpu")
    assert str(tmp_path) not in str(exc.value)


def test_speakers_without_the_model_are_refused_before_any_clip_is_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip(tmp_path / "a.wav")
    monkeypatch.setenv(speakers.MODEL_ENV, str(tmp_path / "missing.onnx"))
    monkeypatch.setattr(speech, "decode", lambda *a: pytest.fail("decoded before refusing"))
    with pytest.raises(ModelUnavailable, match="speaker-fetch"):
        scan_speech_folder(tmp_path, cfg=SpeechConfig(speakers=True))


# ---------------------------------------------------------------------------
# Clustering and the corpus labels
# ---------------------------------------------------------------------------


def test_clustering_separates_voices_and_is_deterministic() -> None:
    rng = np.random.default_rng(0)
    centres = [_vector(v) for v in range(1, 6)]
    rows = []
    for i in range(200):
        v = centres[i % 5] + 0.04 * rng.standard_normal(speakers.EMBEDDING_DIM)
        rows.append(v / np.linalg.norm(v))
    x = np.stack(rows)
    labels = speakers.cluster(x, 0.5)
    assert len(set(labels.tolist())) == 5
    for c in range(5):
        assert len(set(labels[c::5].tolist())) == 1
    assert np.array_equal(labels, speakers.cluster(x, 0.5))


def test_a_scan_clusters_speakers_and_flags_where_the_corpus_disagrees(
    tmp_path: Path, fake_speakers: list[tuple[int, ...]]
) -> None:
    # s2 and s3 are one voice under two labels: the clustering merges them, and says so.
    people = [p if p[0] != "s3" else (*p[:4], 2) for p in SPEAKERS]
    summary = _scan(_corpus(tmp_path, people), speakers=True)
    assert set(fake_speakers) == {(speakers.WINDOWS, speakers.WINDOW_FRAMES, speakers.N_MELS)}
    assert len(summary.speaker_clusters) == 8
    merged = [s for s in summary.speaker_clusters if len(s.corpus_speakers) > 1]
    assert len(merged) == 1 and merged[0].corpus_speakers == {"s2": 3, "s3": 3} and merged[0].mismatch
    flagged = {c.speaker for c in summary.clips if c.speaker_mismatch}
    assert flagged == {"s3"} and summary.speaker_mismatches == 3
    assert all(c.speaker_cluster and c.speaker_distance is not None for c in summary.clips)
    assert len(summary.embeddings) == len(summary.clips)  # in memory, beside the summary


def test_speaker_embeddings_are_held_in_memory_and_purged_with_the_scan(
    tmp_path: Path, fake_speakers: list[tuple[int, ...]]
) -> None:
    summary = _scan(_corpus(tmp_path / "src"), speakers=True)
    store = ScanStore(tmp_path / "cache")
    store.save(summary)
    on_disk = (tmp_path / "cache" / f"{summary.scan_id}.json").read_text(encoding="utf-8")
    assert len(store.held_embeddings(summary.scan_id)) == len(summary.clips)
    assert str(float(_vector(1)[0]))[:8] not in on_disk
    loaded = store.load(summary.scan_id)
    assert loaded is not None and len(loaded.embeddings) == len(summary.clips)
    store.delete(summary.scan_id)
    assert store.held_embeddings(summary.scan_id) == {}


# ---------------------------------------------------------------------------
# The split: speaker-disjoint, stratified
# ---------------------------------------------------------------------------


def test_an_export_of_a_multi_speaker_multi_accent_corpus_has_no_speaker_in_both_splits(
    tmp_path: Path, fake_speakers: list[tuple[int, ...]]
) -> None:
    """The issue's done-when: train and held-out share no speaker, by label or by cluster."""
    summary = _scan(_corpus(tmp_path / "src"), speakers=True)
    split = {"held_out_fraction": 0.34, "group_by": "speaker"}
    parts = {}
    for part in ("train", "held_out"):
        result = _export(summary, tmp_path / part, split=split, split_part=part)
        parts[part] = _rows(Path(result.manifest_path))
        assert {r.split for r in parts[part]} == {part}
    train, held = parts["train"], parts["held_out"]
    assert train and held
    for field in ("speaker_id", "speaker_label", "speaker_cluster"):
        assert not {getattr(r, field) for r in train} & {getattr(r, field) for r in held}, field
    assert len(train) + len(held) == len(SPEAKERS) * CLIPS_PER_SPEAKER
    # Stratified: each accent gives a held-out speaker and keeps two to train on.
    assert {r.accent for r in held} == {"br", "pt", "ao"}
    for accent in ("br", "pt", "ao"):
        assert len({r.speaker_id for r in train if r.accent == accent}) == 2


def test_label_and_cluster_join_into_one_split_group(tmp_path: Path, fake_speakers: list[tuple[int, ...]]) -> None:
    """Where the corpus and the clustering disagree, both readings stay in one part."""
    people = [p if p[0] != "s3" else (*p[:4], 2) for p in SPEAKERS]
    summary = _scan(_corpus(tmp_path, people), speakers=True)
    groups = speech_select.speaker_groups(summary.clips)
    by_label = {c.speaker: groups[c.rel_path] for c in summary.clips}
    assert by_label["s2"] == by_label["s3"] and len(set(groups.values())) == 8
    for seed in range(5):
        req = ExportRequest(
            dest=".",
            speech=CORPUS,
            split={"held_out": 6, "group_by": "speaker", "method": "random", "seed": seed},
        )
        plan = speech_select.plan_speech_selection(summary.clips, req)
        label_parts = {(c.speaker, plan.split[c.rel_path]) for c in plan.chosen if c.speaker in ("s2", "s3")}
        assert len({part for _, part in label_parts}) == 1


def test_a_split_refuses_clips_with_no_speaker(tmp_path: Path) -> None:
    for i in range(4):
        clip(tmp_path / f"{i}.wav", seed=i + 1)
        (tmp_path / f"{i}.txt").write_text(LINE, encoding="utf-8")
    summary = _scan(tmp_path)
    req = ExportRequest(dest=".", speech=CORPUS, split={"held_out": 1, "group_by": "speaker"})
    with pytest.raises(SplitError, match="no speaker"):
        speech_select.plan_speech_selection(summary.clips, req)


def test_a_speech_split_must_group_by_speaker(tmp_path: Path) -> None:
    summary = _scan(_corpus(tmp_path))
    req = ExportRequest(dest=str(tmp_path / "out"), speech=CORPUS, split={"held_out": 3})
    with pytest.raises(ValueError, match='group_by to "speaker"'):
        export_selection(summary, req)
    assert not (tmp_path / "out").exists()


def test_stratified_split_shares_held_out_groups_across_strata() -> None:
    keys = [f"g{i}" for i in range(12)]
    strata = ["a"] * 6 + ["b"] * 3 + ["c"] * 2 + ["d"]
    parts = split_held_out(keys, SplitConfig(held_out=4, group_by="speaker"), strata=strata)
    held = {strata[i] for i, p in enumerate(parts) if p == "held_out"}
    assert parts.count("held_out") == 4 and held >= {"a", "b"}
    assert "d" not in held  # a stratum with one group keeps it to train on
    # Without strata, the split is exactly the unstratified one (#183).
    plain = SplitConfig(held_out_fraction=0.25)
    assert split_held_out(keys, plain) == split_held_out(keys, plain, strata=None)


# ---------------------------------------------------------------------------
# Selection: hours per speaker and accent, caps, balance
# ---------------------------------------------------------------------------


def test_hours_caps_and_accent_balance(tmp_path: Path) -> None:
    people = [*SPEAKERS, ("s10", "br", "male", "twenties", 10)]  # br has four speakers, the others three
    summary = _scan(_corpus(tmp_path, people))
    clip_s = summary.clips[0].duration_s or 0.0

    def plan(**speech: object):  # noqa: ANN202
        return speech_select.plan_speech_selection(summary.clips, ExportRequest(dest=".", speech={**CORPUS, **speech}))

    capped = plan(max_hours_per_speaker=2.5 * clip_s / 3600)
    assert all(round(h * 3600 / clip_s) == 2 for h in capped.stats.hours_by_speaker.values())
    assert {capped.keep_reason[c.rel_path] for c in summary.clips} == {"", "speaker-cap"}

    balanced = plan(balance_accents=True)
    hours = balanced.stats.hours_by_accent
    assert len(set(hours.values())) == 1, hours  # br trimmed to the others' hours
    br = [c for c in balanced.chosen if c.accent == "br"]
    assert len({c.speaker for c in br}) == 4  # trimmed across its speakers, not one dropped

    assert {c.accent for c in plan(accents=["pt"]).chosen} == {"pt"}
    floor = plan(min_hours_per_speaker=4 * clip_s / 3600)
    assert not floor.chosen and set(floor.keep_reason.values()) == {"speaker-under-min-hours"}


def test_unlabelled_accents_never_set_the_accent_balance() -> None:
    def row(name: str, accent: str | None) -> ClipResult:
        return ClipResult(
            rel_path=name, abs_path=f"/x/{name}", passed=True, score=0.9, transcript="hi",
            duration_s=1.0, speaker=name, accent=accent,
        )  # fmt: skip

    clips = [row(f"br{i}", "br") for i in range(4)] + [row(f"pt{i}", "pt") for i in range(3)] + [row("u0", None)]
    plan = speech_select.plan_speech_selection(
        clips, ExportRequest(dest=".", speech={**CORPUS, "balance_accents": True})
    )
    # The cap is pt's three clips, not the one unlabelled clip's.
    kept = [c.accent for c in plan.chosen]
    assert (kept.count("br"), kept.count("pt"), kept.count(None)) == (3, 3, 1)


def test_max_keep_takes_speakers_in_turn(tmp_path: Path) -> None:
    summary = _scan(_corpus(tmp_path))
    req = ExportRequest(dest=".", speech=CORPUS, max_keep=9)
    plan = speech_select.plan_speech_selection(summary.clips, req)
    assert len(plan.chosen) == 9 and len({c.speaker for c in plan.chosen}) == 9


# ---------------------------------------------------------------------------
# Export: audio, transcript sidecars, rows, provenance
# ---------------------------------------------------------------------------


def test_export_writes_each_clip_its_transcript_and_a_speech_manifest_row(tmp_path: Path) -> None:
    summary = _scan(_corpus(tmp_path / "src"))
    result = _export(summary, tmp_path / "out", preserve_structure=False)
    assert result.modality == "speech" and result.copied == len(SPEAKERS) * CLIPS_PER_SPEAKER
    assert result.transcripts_written == result.copied
    assert Path(result.manifest_path).name == SPEECH_MANIFEST_NAME
    assert not (tmp_path / "out" / "manifest.jsonl").exists()  # never read as an image manifest
    rows = _rows(Path(result.manifest_path))
    assert len(rows) == result.copied
    row = next(r for r in rows if r.rel_path == "clips/s3_0.wav")
    assert row.speech_manifest_version == "1.1" and row.modality == "speech"
    assert row.exported_path == "s3_0.wav" and Path(row.exported_abs_path).is_file()
    assert row.transcript_path == "s3_0.txt"
    assert Path(row.transcript_abs_path).read_text(encoding="utf-8") == LINE + "\n"
    assert (row.speaker_id, row.speaker_label, row.accent, row.language, row.gender) == (
        "s3",
        "s3",
        "br",
        "pt",
        "female",
    )
    assert row.age is None  # the corpus gave none: unknown, never estimated
    listed = next(c for c in summary.clips if c.rel_path == "clips/s3_0.wav")
    assert row.duration_s and (row.duration_s, row.sample_rate, row.channels) == (listed.duration_s, 16000, 1)
    assert (row.corpus, row.license) == ("Common Voice pt", "CC0-1.0")
    assert row.provenance.kind == "stock" and row.provenance.license == "CC0-1.0"
    assert row.target_profile.modality == "speech" and row.target_profile.target_category == "accent"
    assert row.split is None and row.consent_release_sha256 is None


def test_a_speech_export_must_name_where_its_clips_came_from(tmp_path: Path) -> None:
    summary = _scan(_corpus(tmp_path / "src"))
    with pytest.raises(ValueError, match=r"speech\.corpus and speech\.license"):
        export_selection(summary, ExportRequest(scan_id=summary.scan_id, dest=str(tmp_path / "out")))
    with pytest.raises(ValidationError, match="go together"):
        SpeechSelection(corpus="x")
    assert not (tmp_path / "out").exists()


def test_an_under_18_speaker_is_never_exported_even_by_name(tmp_path: Path) -> None:
    young = [("kid", "br", "male", "teens", 20)]
    root = _corpus(tmp_path / "src", [*SPEAKERS[:3], *young])
    summary = _scan(root)
    kid = [c for c in summary.clips if c.speaker == "kid"]
    assert kid and all(c.reject_reason == "corpus_under_18" for c in kid)
    everything = [c.rel_path for c in summary.clips]
    for fields in ({}, {"selection": everything}):
        result = _export(summary, tmp_path / f"out{len(fields)}", **fields)
        assert not any("kid" in rel for rel in result.exported_paths)
        assert all(r.speaker_id != "kid" for r in _rows(Path(result.manifest_path)))
    # A clip that passed with an under-18 label (a cached scan from before the check) is still out.
    summary.clips[0].age = "teens"
    assert speech_select.hard_exclusion(summary.clips[0]) == "corpus_under_18"


def test_image_options_are_refused_on_a_speech_scan_and_speech_on_an_image_scan(tmp_path: Path) -> None:
    summary = _scan(_corpus(tmp_path / "src"))
    for fields in ({"face_clusters": ["face_1"]}, {"rank": "identity"}, {"caption_url": "http://lens/x"}):
        with pytest.raises(ValueError, match="image options"):
            _export(summary, tmp_path / "out", **fields)
    assert not (tmp_path / "out").exists()
    from PIL import Image

    from argus_curator.scanner import scan_folder

    pictures = tmp_path / "pics"
    pictures.mkdir()
    Image.new("RGB", (600, 600)).save(pictures / "a.png")
    image_summary = scan_folder(pictures)
    from argus_curator.selection import selection_problem

    req = ExportRequest(dest=".", speech=CORPUS)
    assert "image scan" in (selection_problem(req, image_summary) or "")
    speaker_split = ExportRequest(dest=".", split={"held_out": 1, "group_by": "speaker"})
    assert "speech scan" in (selection_problem(speaker_split, image_summary) or "")


def test_symlink_export_into_the_source_folder_never_deletes_a_clip_or_its_transcript(tmp_path: Path) -> None:
    src = tmp_path / "src"
    clip(src / "a.wav")
    (src / "a.txt").write_text(LINE, encoding="utf-8")
    summary = _scan(src)
    for mode in ("symlink", "copy"):
        result = _export(summary, src, mode=mode)
        assert result.copied == 0 and result.skipped == 1
        assert (src / "a.wav").is_file() and not (src / "a.wav").is_symlink()
        assert (src / "a.txt").read_text(encoding="utf-8") == LINE
        assert _rows(Path(result.manifest_path)) == []


def test_symlink_export_links_the_clip_and_writes_its_own_transcript(tmp_path: Path) -> None:
    src = tmp_path / "src"
    clip(src / "a.wav")
    (src / "a.txt").write_text(LINE, encoding="utf-8")
    summary = _scan(src)
    for _ in range(2):  # a re-export over the first one's link
        result = _export(summary, tmp_path / "out", mode="symlink")
        assert result.copied == 1
    out = tmp_path / "out"
    assert (out / "a.wav").is_symlink() and (out / "a.wav").resolve() == (src / "a.wav").resolve()
    assert not (out / "a.txt").is_symlink()  # writing a caption into the export never reaches the source
    (out / "a.txt").write_text("edited", encoding="utf-8")
    assert (src / "a.txt").read_text(encoding="utf-8") == LINE


def test_move_rows_read_from_the_destination(tmp_path: Path) -> None:
    summary = _scan(_corpus(tmp_path / "src", SPEAKERS[:2]))
    moved: list[str] = []
    result = export_speech(
        summary,
        ExportRequest(scan_id=summary.scan_id, dest=str(tmp_path / "out"), mode="move", speech=CORPUS),
        moved=moved,
    )
    rows = _rows(Path(result.manifest_path))
    assert len(moved) == len(rows) == 2 * CLIPS_PER_SPEAKER
    for r in rows:
        assert r.abs_path == r.exported_abs_path and Path(r.abs_path).is_file()
        assert not (tmp_path / "src" / r.rel_path).exists()


def test_a_failed_transfer_has_no_row_and_leaves_no_transcript(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from argus_curator import speech_export

    summary = _scan(_corpus(tmp_path / "src", SPEAKERS[:1]))
    real = speech_export.transfer_one

    def flaky(src: Path, dst: Path, mode: str) -> None:
        if src.name == "s1_1.wav":
            raise OSError("disk full")
        real(src, dst, mode)

    monkeypatch.setattr(speech_export, "transfer_one", flaky)
    result = _export(summary, tmp_path / "out")
    rows = _rows(Path(result.manifest_path))
    assert result.skipped == 1 and {r.rel_path for r in rows} == {"clips/s1_0.wav", "clips/s1_2.wav"}
    assert not (tmp_path / "out" / "clips" / "s1_1.txt").exists()


def test_a_failed_transfer_keeps_the_transcript_an_earlier_export_left(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from argus_curator import speech_export

    summary = _scan(_corpus(tmp_path / "src", SPEAKERS[:1]))
    out = tmp_path / "out"
    assert _export(summary, out).copied == CLIPS_PER_SPEAKER
    txt = out / "clips" / "s1_1.txt"
    txt.write_text("edited after the first export\n", encoding="utf-8")
    real = speech_export.transfer_one

    def flaky(src: Path, dst: Path, mode: str) -> None:
        if src.name == "s1_1.wav":
            raise OSError("disk full")
        real(src, dst, mode)

    monkeypatch.setattr(speech_export, "transfer_one", flaky)
    assert _export(summary, out).skipped == 1
    assert txt.read_text(encoding="utf-8") == "edited after the first export\n"
    assert (out / "clips" / "s1_1.wav").is_file()  # the earlier clip still has its transcript
    assert sorted(p.name for p in (out / "clips").iterdir() if p.name.startswith(".")) == []


# ---------------------------------------------------------------------------
# Consent: a real person's voice outside a licensed corpus
# ---------------------------------------------------------------------------


_DAY_BEFORE = (date.today() - timedelta(days=1)).isoformat()


def _record(path: Path, references: list[Path], **over: object) -> Path:
    import hashlib

    today = date.today()
    record = {
        "release_sha256": "a" * 64,
        "subject": "Test Subject",
        "collected_by": "a test",
        "signed_on": (today - timedelta(days=10)).isoformat(),
        "expires_on": None,
        "age_check": {"checked_by": "a test", "checked_on": (today - timedelta(days=10)).isoformat()},
        "face": False,
        "voice": True,
        "deliveries": ["neutral"],
        "uses": ["training"],
        "reference_recordings": [hashlib.sha256(p.read_bytes()).hexdigest() for p in references],
        "revocable": True,
        "notice_days": 0,
        "on_revoke": "delete",
        **over,
    }
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def test_a_consented_voice_exports_only_the_consented_speaker(
    tmp_path: Path, fake_speakers: list[tuple[int, ...]]
) -> None:
    root = _corpus(tmp_path / "src", SPEAKERS[:3])
    _record(root / "consent.json", [root / "clips" / "s2_0.wav"])
    summary = _scan(root, speakers=True, second_voice=True)
    speech = {"consent_record": "consent.json", "consent_delivery": "neutral"}
    result = export_selection(
        summary, ExportRequest(scan_id=summary.scan_id, dest=str(tmp_path / "out"), speech=speech)
    )
    rows = _rows(Path(result.manifest_path))
    assert {r.speaker_label for r in rows} == {"s2"} and len(rows) == CLIPS_PER_SPEAKER
    row = rows[0]
    assert row.provenance.kind == "consented" and row.provenance.release_sha256 == "a" * 64
    assert row.consent_release_sha256 == "a" * 64 and row.consent_record_sha256 is not None
    report = (tmp_path / "out" / "curation_report.csv").read_text(encoding="utf-8")
    assert "consent:other-speaker" in report


@pytest.mark.parametrize("mode", ["copy", "move", "symlink"])
def test_consent_subject_name_appears_nowhere_in_the_speech_export(
    tmp_path: Path, fake_speakers: list[tuple[int, ...]], mode: str
) -> None:
    """Must never: a consented voice export naming the consenting person or who collected the release.

    The speech manifest, the transcript sidecars and the CSV report name the subject only
    by cortex's subject ref, and the ref still matches the record.
    """
    from argus_cortex.consent import ConsentRecord, consent_matches

    root = _corpus(tmp_path / "src", SPEAKERS[:3])
    record_path = _record(
        root / "consent.json", [root / "clips" / "s2_0.wav"], subject="Jo Speaker", collected_by="Example Voice Studio"
    )
    summary = _scan(root, speakers=True, second_voice=True)
    speech = {"consent_record": "consent.json", "consent_delivery": "neutral"}
    dest = tmp_path / "out"
    result = export_selection(summary, ExportRequest(scan_id=summary.scan_id, dest=str(dest), speech=speech, mode=mode))
    files = {
        str(f.relative_to(dest)): f.read_text(encoding="utf-8")
        for f in dest.rglob("*")
        if f.is_file() and f.suffix != ".wav"
    }
    assert SPEECH_MANIFEST_NAME in files and "curation_report.csv" in files
    assert any(name.endswith(".txt") for name in files)
    for name, text in files.items():
        assert "Jo Speaker" not in text, name
        assert "Example Voice Studio" not in text, name
    rows = _rows(Path(result.manifest_path))
    assert rows and all(r.provenance.subject == "consent:" + "a" * 64 for r in rows)
    record = ConsentRecord.model_validate_json(record_path.read_bytes())
    assert consent_matches(record, rows[0].provenance)


@pytest.mark.parametrize(
    ("over", "match"),
    [
        ({"uses": ["speech"]}, "use-not-covered"),
        ({"deliveries": ["warm"]}, "delivery-not-covered"),
        ({"revocation": {"requested_on": _DAY_BEFORE, "effective_on": _DAY_BEFORE, "by": "subject"}}, "revoked"),
    ],
)
def test_a_voice_consent_record_that_does_not_cover_the_set_is_refused(
    tmp_path: Path, fake_speakers: list[tuple[int, ...]], over: dict, match: str
) -> None:
    root = _corpus(tmp_path / "src", SPEAKERS[:2])
    _record(root / "consent.json", [root / "clips" / "s1_0.wav"], **over)
    summary = _scan(root, speakers=True, second_voice=True)
    speech = {"consent_record": "consent.json", "consent_delivery": "neutral"}
    with pytest.raises(ConsentRefused, match=match):
        export_selection(summary, ExportRequest(scan_id=summary.scan_id, dest=str(tmp_path / "out"), speech=speech))
    assert not (tmp_path / "out").exists()


def test_a_voice_consent_needs_its_reference_recording_in_the_scan(
    tmp_path: Path, fake_speakers: list[tuple[int, ...]]
) -> None:
    root = _corpus(tmp_path / "src", SPEAKERS[:2])
    elsewhere = clip(tmp_path / "elsewhere.wav", seed=99)
    _record(root / "consent.json", [elsewhere])
    summary = _scan(root, speakers=True)
    speech = SpeechSelection(consent_record="consent.json", consent_delivery="neutral")
    with pytest.raises(ConsentRefused, match="reference recording"):
        export_speech(summary, ExportRequest(dest=str(tmp_path / "out"), speech=speech))


# ---------------------------------------------------------------------------
# Binding a consented voice by distance to its reference recordings (#209)
# ---------------------------------------------------------------------------

CONSENT = {"consent_record": "consent.json", "consent_delivery": "neutral"}


def _at_distance(centroid: np.ndarray, distance: float, seed: int) -> np.ndarray:
    """A unit vector whose cosine distance to the unit *centroid* is *distance* (a synthetic voice)."""
    other = np.random.default_rng(seed).standard_normal(len(centroid))
    other -= (other @ centroid) * centroid
    other /= np.linalg.norm(other)
    cos = 1.0 - distance
    return (cos * centroid + np.sqrt(max(0.0, 1.0 - cos * cos)) * other).astype(np.float32)


def _consented_scan(root: Path, distances: dict[str, float], **cfg: object) -> ScanSummary:
    """Two speakers in one loose cluster; s2_0 is the reference, every other clip at the given distance from it."""
    _corpus(root, SPEAKERS[:2])
    _record(root / "consent.json", [root / "clips" / "s2_0.wav"])
    summary = _scan(root, speakers=True, second_voice=True, speaker_threshold=1.9, **cfg)
    reference = _vector(2)
    for i, rel in enumerate(sorted(summary.embeddings)):
        if rel != "clips/s2_0.wav":
            summary.embeddings[rel] = _at_distance(reference, distances.get(rel, 0.05), seed=i)
        # One voice throughout: every window is the clip's own embedding.
        summary.voice_windows[rel] = np.tile(summary.embeddings[rel], (4, 1)).astype(np.float16)
    return summary


def _report(dest: Path) -> dict[str, dict[str, str]]:
    import csv

    with (dest / "curation_report.csv").open(encoding="utf-8") as fh:
        return {row["rel_path"]: row for row in csv.DictReader(fh)}


def test_a_consented_voice_is_bound_by_distance_not_by_cluster(
    tmp_path: Path, fake_speakers: list[tuple[int, ...]]
) -> None:
    """Must never: a clip of another voice kept under a voice record because it shares the subject's cluster."""
    root = tmp_path / "src"
    similar = {f"clips/s1_{i}.wav": 0.45 for i in range(CLIPS_PER_SPEAKER)}
    summary = _consented_scan(root, similar)
    speakers.assign_clusters(summary.clips, summary.embeddings, 1.9)
    assert len({c.speaker_cluster for c in summary.clips}) == 1  # the clustering merged the two voices
    dest = tmp_path / "out"
    result = export_selection(summary, ExportRequest(scan_id=summary.scan_id, dest=str(dest), speech=CONSENT))
    rows = _rows(Path(result.manifest_path))
    assert {r.speaker_label for r in rows} == {"s2"} and len(rows) == CLIPS_PER_SPEAKER
    report = _report(dest)
    for rel in similar:
        assert report[rel]["keep_reason"] == "consent:other-speaker"
        assert float(report[rel]["consent_distance"]) == pytest.approx(0.45, abs=1e-3)
    assert float(report["clips/s2_0.wav"]["consent_distance"]) == pytest.approx(0.0, abs=1e-3)


@pytest.mark.parametrize(
    ("distance", "ceiling", "kept"),
    [(0.2, None, True), (0.28, None, False), (0.2, 0.15, False), (0.28, 0.3, True)],
)
def test_consent_max_distance_is_the_ceiling(
    tmp_path: Path, fake_speakers: list[tuple[int, ...]], distance: float, ceiling: float | None, kept: bool
) -> None:
    probe = "clips/s2_1.wav"
    summary = _consented_scan(tmp_path / "src", {probe: distance})
    speech = {**CONSENT, **({"consent_max_distance": ceiling} if ceiling is not None else {})}
    hold = voice_consent_gate(summary, SpeechSelection(**speech))
    assert hold is not None and hold.distances[probe] == pytest.approx(distance, abs=1e-3)
    assert (probe not in hold.excluded) is kept
    assert hold.max_distance == (ceiling if ceiling is not None else VOICE_CONSENT_MAX_DISTANCE)


def test_the_reference_voice_is_the_centroid_of_every_reference_recording(
    tmp_path: Path, fake_speakers: list[tuple[int, ...]]
) -> None:
    root = _corpus(tmp_path / "src", SPEAKERS[:2])
    _record(root / "consent.json", [root / "clips" / "s2_0.wav", root / "clips" / "s2_1.wav"])
    summary = _scan(root, speakers=True, second_voice=True)
    a, b = _vector(2), _vector(3)
    summary.embeddings["clips/s2_0.wav"], summary.embeddings["clips/s2_1.wav"] = a, b
    summary.embeddings["clips/s2_2.wav"] = (a + b) / np.linalg.norm(a + b)
    hold = voice_consent_gate(summary, SpeechSelection(**CONSENT))
    assert hold is not None and hold.references == ("clips/s2_0.wav", "clips/s2_1.wav")
    assert hold.distances["clips/s2_2.wav"] == pytest.approx(0.0, abs=1e-3)
    # The references themselves are measured against the centroid too, like every clip.
    assert hold.distances["clips/s2_0.wav"] > 0.0


def test_a_voice_consent_never_falls_back_to_clusters_without_a_reference_embedding(
    tmp_path: Path, fake_speakers: list[tuple[int, ...]]
) -> None:
    """Must never: a consented voice export bound by speaker cluster because no reference voice could be built."""
    root = tmp_path / "src"
    summary = _consented_scan(root, {})
    assert summary.speaker_clusters  # the clusters are there to fall back on, and are not used
    speech = SpeechSelection(**CONSENT)
    # The reference recording's embedding alone is gone (it failed, or was dropped).
    del summary.embeddings["clips/s2_0.wav"]
    with pytest.raises(ConsentRefused, match="no speaker embedding"):
        voice_consent_gate(summary, speech)
    # All of them gone: a restart drops the in-memory embeddings.
    summary.embeddings.clear()
    with pytest.raises(ConsentRefused, match="rescan with speech.speakers on"):
        export_speech(summary, ExportRequest(dest=str(tmp_path / "out"), speech=speech))
    assert not (tmp_path / "out").exists()
    # A scan that never embedded speakers.
    plain = _scan(root)
    with pytest.raises(ConsentRefused, match="no speaker embedding"):
        voice_consent_gate(plain, speech)
    # An embedding that is not a usable vector is no reference either.
    summary = _consented_scan(root, {})
    summary.embeddings["clips/s2_0.wav"] = np.zeros(speakers.EMBEDDING_DIM, dtype=np.float32)
    with pytest.raises(ConsentRefused, match="no speaker embedding"):
        voice_consent_gate(summary, speech)


def test_a_clip_with_no_speaker_embedding_is_never_bound(tmp_path: Path, fake_speakers: list[tuple[int, ...]]) -> None:
    summary = _consented_scan(tmp_path / "src", {})
    del summary.embeddings["clips/s2_2.wav"]
    hold = voice_consent_gate(summary, SpeechSelection(**CONSENT))
    assert hold is not None and hold.excluded["clips/s2_2.wav"] == "consent:no-embedding"
    assert "clips/s2_2.wav" not in hold.distances and "clips/s2_1.wav" not in hold.excluded


def test_a_voice_binding_keeps_plain_distances_and_never_the_voiceprint(
    tmp_path: Path, fake_speakers: list[tuple[int, ...]]
) -> None:
    """Must never: the reference voice's centroid (a voiceprint) held, stored or served; only distances are."""
    summary = _consented_scan(tmp_path / "src", {"clips/s1_0.wav": 0.6})
    hold = voice_consent_gate(summary, SpeechSelection(**CONSENT))
    assert hold is not None
    # The exact shape held: any added field (a centroid by another name) fails here.
    assert set(vars(hold)) == {
        "release_sha256",
        "record_sha256",
        "provenance",
        "max_distance",
        "references",
        "distances",
        "excluded",
        "second_voice_max_distance",
        "second_voice",
    }

    def no_vector(node: object, where: str) -> None:
        # Recursive, and not only ndarrays: a centroid kept as ``.tolist()`` or nested is still a voiceprint.
        assert not isinstance(node, np.ndarray | np.generic), where
        if isinstance(node, dict):
            for key, value in node.items():
                no_vector(value, f"{where}[{key!r}]")
        elif isinstance(node, list | tuple):
            assert not any(isinstance(v, float) for v in node), f"{where} is a float sequence"
            for i, value in enumerate(node):
                no_vector(value, f"{where}[{i}]")

    for name, value in vars(hold).items():
        no_vector(value.model_dump() if name == "provenance" else value, name)
    assert all(type(r) is str for r in hold.references)
    assert all(type(d) is float for d in hold.distances.values())
    assert hold.second_voice and all(type(d) is float for d in hold.second_voice.values())


# ---------------------------------------------------------------------------
# A second voice inside a consented clip (#217)
# ---------------------------------------------------------------------------


def _with_second_voice(summary: ScanSummary, rel: str, distance: float, windows: int = 1, of: int = 12) -> None:
    """*rel* reads as the subject overall, but *windows* of its *of* windows are another voice, *distance* away."""
    reference = _vector(2)
    subject = [_at_distance(reference, 0.1, seed=100 + i) for i in range(of)]
    other = _at_distance(reference, distance, seed=999)
    start = of // 2
    for i in range(start, start + windows):
        subject[i] = other
    summary.embeddings[rel] = _at_distance(reference, 0.08, seed=7)
    summary.voice_windows[rel] = np.stack(subject).astype(np.float16)


def test_a_clip_with_a_second_voice_is_never_kept_in_a_consented_voice_export(
    tmp_path: Path, fake_speakers: list[tuple[int, ...]]
) -> None:
    """Must never: a clip with a second voice in it kept in a consented voice export.

    The clip's own embedding sits well inside consent_max_distance (the subject
    speaks most of it), and one window of 1.5 s is another voice: it is left out.
    """
    probe = "clips/s2_1.wav"
    summary = _consented_scan(tmp_path / "src", {})
    _with_second_voice(summary, probe, 0.6)
    dest = tmp_path / "out"
    result = export_selection(summary, ExportRequest(scan_id=summary.scan_id, dest=str(dest), speech=CONSENT))
    rows = _rows(Path(result.manifest_path))
    # Every other clip reads as the subject throughout and is kept.
    assert {r.rel_path for r in rows} == {c.rel_path for c in summary.clips} - {probe}
    report = _report(dest)
    assert report[probe]["keep_reason"] == "consent:second-voice"
    assert float(report[probe]["consent_distance"]) < VOICE_CONSENT_MAX_DISTANCE  # the binding alone keeps it
    assert float(report[probe]["second_voice_distance"]) == pytest.approx(0.6, abs=2e-3)
    assert float(report["clips/s2_2.wav"]["second_voice_distance"]) <= SECOND_VOICE_MAX_DISTANCE


@pytest.mark.parametrize(
    ("distance", "ceiling", "kept"),
    [(0.3, None, True), (SECOND_VOICE_MAX_DISTANCE + 0.03, None, False), (0.3, 0.25, False), (0.6, 0.7, True)],
)
def test_second_voice_max_distance_is_the_ceiling(
    tmp_path: Path, fake_speakers: list[tuple[int, ...]], distance: float, ceiling: float | None, kept: bool
) -> None:
    probe = "clips/s2_1.wav"
    summary = _consented_scan(tmp_path / "src", {})
    _with_second_voice(summary, probe, distance)
    speech = {**CONSENT, **({"second_voice_max_distance": ceiling} if ceiling is not None else {})}
    hold = voice_consent_gate(summary, SpeechSelection(**speech))
    assert hold is not None and hold.second_voice[probe] == pytest.approx(distance, abs=2e-3)
    assert (probe not in hold.excluded) is kept
    assert hold.second_voice_max_distance == (ceiling if ceiling is not None else SECOND_VOICE_MAX_DISTANCE)


def test_a_consented_voice_export_refuses_without_the_second_voice_check(
    tmp_path: Path, fake_speakers: list[tuple[int, ...]]
) -> None:
    """Must never: the second-voice check skipped. No windows held means refusal, never a pass."""
    root = tmp_path / "src"
    speech = SpeechSelection(**CONSENT)
    # A scan that embedded speakers but never asked for the windows.
    _corpus(root, SPEAKERS[:2])
    _record(root / "consent.json", [root / "clips" / "s2_0.wav"])
    plain = _scan(root, speakers=True)
    assert plain.embeddings and not plain.voice_windows
    with pytest.raises(ConsentRefused, match="speech.second_voice"):
        voice_consent_gate(plain, speech)
    # Windows that a restart dropped while the embeddings were somehow still held.
    summary = _consented_scan(tmp_path / "again", {})
    summary.voice_windows.clear()
    with pytest.raises(ConsentRefused, match="second-voice windows"):
        export_speech(summary, ExportRequest(dest=str(tmp_path / "out"), speech=speech))
    assert not (tmp_path / "out").exists()
    # Windows held, but the summary says the scan never ran the check: not trusted either.
    summary = _consented_scan(tmp_path / "third", {})
    assert summary.speech_config is not None
    summary.speech_config = summary.speech_config.model_copy(update={"second_voice": False})
    with pytest.raises(ConsentRefused, match="speech.second_voice"):
        voice_consent_gate(summary, speech)


def test_the_second_voice_check_measures_against_the_reference_recordings_windows(
    tmp_path: Path, fake_speakers: list[tuple[int, ...]]
) -> None:
    """Windows are measured against the references' own windows, not their whole-clip embedding."""
    summary = _consented_scan(tmp_path / "src", {})
    elsewhere = _vector(5)
    summary.voice_windows["clips/s2_0.wav"] = np.tile(elsewhere, (4, 1)).astype(np.float16)
    summary.voice_windows["clips/s2_1.wav"] = np.tile(elsewhere, (6, 1)).astype(np.float16)
    hold = voice_consent_gate(summary, SpeechSelection(**CONSENT))
    assert hold is not None and hold.second_voice["clips/s2_1.wav"] == pytest.approx(0.0, abs=2e-3)
    # The reference recordings with no windows of their own: no voice to check against, so refused.
    del summary.voice_windows["clips/s2_0.wav"]
    with pytest.raises(ConsentRefused, match="no second-voice windows"):
        voice_consent_gate(summary, SpeechSelection(**CONSENT))


def test_a_clip_with_no_second_voice_windows_is_never_kept(
    tmp_path: Path, fake_speakers: list[tuple[int, ...]]
) -> None:
    summary = _consented_scan(tmp_path / "src", {})
    del summary.voice_windows["clips/s2_2.wav"]
    summary.voice_windows["clips/s2_1.wav"] = np.zeros((3, speakers.EMBEDDING_DIM), dtype=np.float16)
    hold = voice_consent_gate(summary, SpeechSelection(**CONSENT))
    assert hold is not None
    assert hold.excluded["clips/s2_2.wav"] == "consent:no-voice-windows"
    assert hold.excluded["clips/s2_1.wav"] == "consent:no-voice-windows"
    assert "clips/s2_0.wav" not in hold.excluded


def test_the_second_voice_flag_changes_nothing_outside_a_consented_export(
    tmp_path: Path, fake_speakers: list[tuple[int, ...]]
) -> None:
    """Without a consent record there is no reference voice to measure against: a corpus export is unchanged."""
    summary = _consented_scan(tmp_path / "src", {})
    _with_second_voice(summary, "clips/s2_1.wav", 0.9, windows=6)
    result = _export(summary, tmp_path / "out")
    assert "clips/s2_1.wav" in {r.rel_path for r in _rows(Path(result.manifest_path))}
    assert _report(tmp_path / "out")["clips/s2_1.wav"]["second_voice_distance"] == ""


def test_select_route_reports_the_second_voice_distance(tmp_path: Path, fake_speakers: list[tuple[int, ...]]) -> None:
    source = _corpus(tmp_path / "corpus", SPEAKERS[:2])
    _record(source / "consent.json", [source / "clips" / "s2_0.wav"])
    (tmp_path / "exports").mkdir()
    client = TestClient(
        create_app(cache_dir=str(tmp_path / "cache"), source_root=str(source), export_root=str(tmp_path / "exports"))
    )
    scan = {"folder": "", "target_profile": {"modality": "speech"}, "speech": {"speakers": True}}
    scan_id = client.post("/scan/folder", json=scan).json()["scan_id"]
    resp = client.post(f"/scan/{scan_id}/select", json={"speech": CONSENT})
    assert resp.status_code == 400 and "speech.second_voice" in resp.text, resp.text

    scan["speech"] = {"speakers": True, "second_voice": True}
    scan_id = client.post("/scan/folder", json=scan).json()["scan_id"]
    resp = client.post(f"/scan/{scan_id}/select", json={"speech": CONSENT})
    assert resp.status_code == 200, resp.text
    rows = {r["rel_path"]: r for r in resp.json()["rows"]}
    assert rows["clips/s2_1.wav"]["second_voice_distance"] == pytest.approx(0.0, abs=2e-3)
    assert not rows["clips/s2_1.wav"]["keep_reason"]
    served = client.get(f"/scan/{scan_id}").text
    assert "second_voice_distance" not in served  # the measurement is the select's, never the stored scan's


def test_second_voice_windows_are_held_in_memory_and_purged_with_the_scan(
    tmp_path: Path, fake_speakers: list[tuple[int, ...]]
) -> None:
    """Must never: a voiceprint persisted. The windows live beside the embeddings and go with the scan."""
    summary = _scan(_corpus(tmp_path / "src", SPEAKERS[:2]), speakers=True, second_voice=True)
    assert set(summary.voice_windows) == set(summary.embeddings)
    window = next(iter(summary.voice_windows.values()))
    assert window.ndim == 2 and window.shape[1] == speakers.EMBEDDING_DIM and window.dtype == np.float16
    store = ScanStore(tmp_path / "cache")
    store.save(summary)
    assert {p.name for p in (tmp_path / "cache").iterdir()} == {f"{summary.scan_id}.json"}
    on_disk = json.loads((tmp_path / "cache" / f"{summary.scan_id}.json").read_text(encoding="utf-8"))
    assert "voice_windows" not in on_disk and "embeddings" not in on_disk
    # A fresh store reads only the disk: nothing of the windows comes back.
    assert ScanStore(tmp_path / "cache").held_voice_windows(summary.scan_id) == {}
    loaded = store.load(summary.scan_id)
    assert loaded is not None and set(loaded.voice_windows) == set(summary.voice_windows)
    # A dropped clip's windows go with it, and all of them with the scan.
    gone = sorted(summary.voice_windows)[0]
    speech.drop_clips(loaded, {gone})
    store.save(loaded)
    assert gone not in store.held_voice_windows(summary.scan_id)
    store.delete(summary.scan_id)
    assert store.held_voice_windows(summary.scan_id) == {}


def test_second_voice_needs_speakers() -> None:
    with pytest.raises(ValidationError, match="turn speakers on"):
        SpeechConfig(second_voice=True)


def test_second_voice_windows_cover_only_speech() -> None:
    """Silence is dropped before windowing, speech under one window is tiled, and none gives no windows."""
    x = voice(6.0)
    frames = len(speakers.fbank(x))
    every = speakers.segment_features(x, 16000, None)
    assert every.shape[1:] == (speakers.SEGMENT_FRAMES, speakers.N_MELS) and every.dtype == np.float32
    half = np.zeros(frames, dtype=bool)
    half[: frames // 2] = True
    assert len(speakers.segment_features(x, 16000, half)) < len(every)
    # The windows step by the hop, and the last one ends on the last speech frame.
    starts = len(every) - 1
    assert starts == -(-(frames - speakers.SEGMENT_FRAMES) // speakers.SEGMENT_HOP_FRAMES)
    short = np.zeros(frames, dtype=bool)
    short[:40] = True
    assert speakers.segment_features(x, 16000, short).shape == (1, speakers.SEGMENT_FRAMES, speakers.N_MELS)
    assert speakers.segment_features(x, 16000, np.zeros(frames, dtype=bool)).shape[0] == 0
    # A mask of another length is clipped or padded, never an error.
    assert len(speakers.segment_features(x, 16000, np.ones(frames + 50, dtype=bool))) == len(every)


def test_second_voice_windows_run_in_fixed_shape_passes_under_the_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every pass is (SEGMENT_BATCH, SEGMENT_FRAMES, N_MELS), whatever the clip's length, under the GPU lease."""
    leases: list[str] = []
    shapes: list[tuple[int, ...]] = []

    class _Recorder:
        name = "recorder"

        def lease(self, *, caller: str, min_vram_mb: int, **_options: object):  # noqa: ANN202
            import contextlib

            leases.append(caller)
            return contextlib.nullcontext()

    class _Session:
        def get_inputs(self):  # noqa: ANN202
            return [type("I", (), {"name": "feats"})()]

        def run(self, _outputs: object, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
            x = feeds["feats"]
            shapes.append(x.shape)
            # Each row's embedding carries its window's first feature, so the order can be checked.
            out = np.zeros((len(x), speakers.EMBEDDING_DIM), dtype=np.float32)
            out[:, 0] = 1.0
            out[:, 1] = x[:, 0, 0]
            return [out]

    monkeypatch.setattr(runtime, "providers", lambda device: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(runtime, "_coordinator", _Recorder())
    for seconds in (1.0, 7.3, 19.5):
        shapes.clear()
        leases.clear()
        x = voice(seconds)
        windows = speakers.embed_segments(_Session(), x, 16000, None, "cuda")
        assert leases == ["argus-curator:speakers:run"]
        assert set(shapes) == {(speakers.SEGMENT_BATCH, speakers.SEGMENT_FRAMES, speakers.N_MELS)}
        expected = speakers.segment_features(x, 16000, None)
        assert windows.shape == (len(expected), speakers.EMBEDDING_DIM) and windows.dtype == np.float16
        assert len(shapes) == -(-len(expected) // speakers.SEGMENT_BATCH)
        # Row i is window i: no reordering across passes, and the padded tail of the last pass is dropped.
        ratio = windows[:, 1].astype(np.float32) / windows[:, 0].astype(np.float32)
        np.testing.assert_allclose(ratio, expected[:, 0, 0], rtol=1e-2, atol=1e-2)
    # A pass holds no more frames than a clip's embedding pass, so the speakers VRAM figure covers it.
    assert speakers.SEGMENT_BATCH * speakers.SEGMENT_FRAMES <= speakers.WINDOWS * speakers.WINDOW_FRAMES


def test_second_voice_distance_takes_the_farthest_run_of_windows() -> None:
    centroid = _vector(2)
    near = [_at_distance(centroid, 0.1, seed=i) for i in range(8)]
    assert speakers.second_voice_distance(np.stack(near), centroid) == pytest.approx(0.1, abs=1e-3)
    stray = list(near)
    stray[3] = _at_distance(centroid, 0.7, seed=50)
    assert speakers.second_voice_distance(np.stack(stray), centroid, run=1) == pytest.approx(0.7, abs=1e-3)
    # With a run of two, one stray window is not enough; two in a row are.
    assert speakers.second_voice_distance(np.stack(stray), centroid, run=2) == pytest.approx(0.1, abs=1e-3)
    stray[4] = _at_distance(centroid, 0.6, seed=51)
    assert speakers.second_voice_distance(np.stack(stray), centroid, run=2) == pytest.approx(0.6, abs=1e-3)
    # Fewer windows than the run: all of them.
    assert speakers.second_voice_distance(np.stack(stray[3:4]), centroid, run=2) == pytest.approx(0.7, abs=1e-3)
    for bad in (
        np.zeros((0, speakers.EMBEDDING_DIM)),
        np.zeros((2, speakers.EMBEDDING_DIM)),
        np.full((2, speakers.EMBEDDING_DIM), np.nan),
        np.ones((2, 5)),
    ):
        assert speakers.second_voice_distance(bad, centroid) is None


# ---------------------------------------------------------------------------
# The server: select and export a speech scan
# ---------------------------------------------------------------------------


@pytest.fixture
def served(tmp_path: Path, fake_speakers: list[tuple[int, ...]]) -> tuple[TestClient, str, Path]:
    source = _corpus(tmp_path / "corpus")
    exports = tmp_path / "exports"
    exports.mkdir()
    client = TestClient(
        create_app(cache_dir=str(tmp_path / "cache"), source_root=str(source), export_root=str(exports))
    )
    resp = client.post(
        "/scan/folder", json={"folder": "", "target_profile": {"modality": "speech"}, "speech": {"speakers": True}}
    )
    assert resp.status_code == 200, resp.text
    return client, resp.json()["scan_id"], exports


def test_select_route_reports_hours_and_a_speaker_disjoint_split(served: tuple[TestClient, str, Path]) -> None:
    client, scan_id, exports = served
    body = {"split": {"held_out_fraction": 0.34, "group_by": "speaker"}, "speech": {"accents": ["br", "pt"]}}
    resp = client.post(f"/scan/{scan_id}/select", json=body)
    assert resp.status_code == 200, resp.text
    out = resp.json()
    assert out["kept"] == 6 * CLIPS_PER_SPEAKER and out["train"] + out["held_out"] == out["kept"]
    parts: dict[str, set[str]] = {}
    for row in out["rows"]:
        if row["split"]:
            parts.setdefault(row["split"], set()).add(row["speaker_id"])
    assert parts["train"] and parts["held_out"] and not parts["train"] & parts["held_out"]
    assert set(out["speech"]["hours_by_accent"]) == {"br", "pt"}
    assert out["speech"]["speakers_by_split"] == {"held_out": len(parts["held_out"]), "train": len(parts["train"])}
    assert not any(exports.iterdir())  # select writes nothing


def test_export_route_exports_a_speech_scan(served: tuple[TestClient, str, Path]) -> None:
    client, scan_id, exports = served
    resp = client.post("/export", json={"scan_id": scan_id, "dest": "set", "speech": CORPUS})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["modality"] == "speech" and body["speech_manifest_version"] == "1.1"
    assert (exports / "set" / SPEECH_MANIFEST_NAME).is_file()
    health = client.get("/health").json()
    assert health["speech_export"] is True and health["speech_manifest_version"] == "1.1"


def test_speech_export_paths_are_contained(served: tuple[TestClient, str, Path], tmp_path: Path) -> None:
    client, scan_id, exports = served
    resp = client.post("/export", json={"scan_id": scan_id, "dest": "../escape", "speech": CORPUS})
    assert resp.status_code == 400 and not (tmp_path / "escape").exists()
    speech = {"consent_record": "../../outside.json", "consent_delivery": "neutral"}
    resp = client.post("/export", json={"scan_id": scan_id, "dest": "set", "speech": speech})
    assert resp.status_code == 400 and "escapes" in resp.text
    resp = client.post(f"/scan/{scan_id}/select", json={"speech": speech})
    assert resp.status_code == 400 and "escapes" in resp.text


def test_speech_move_needs_the_gate_and_drops_the_moved_clips(tmp_path: Path, fake_speakers) -> None:  # noqa: ANN001
    source = _corpus(tmp_path / "corpus", SPEAKERS[:2])
    exports = tmp_path / "exports"
    exports.mkdir()
    body = {"dest": "set", "mode": "move", "speech": CORPUS}
    scan = {"folder": "", "target_profile": {"modality": "speech"}}
    closed = TestClient(create_app(cache_dir=str(tmp_path / "c1"), source_root=str(source), export_root=str(exports)))
    scan_id = closed.post("/scan/folder", json=scan).json()["scan_id"]
    before = sorted(p.name for p in (source / "clips").iterdir())
    assert len(before) == 2 * CLIPS_PER_SPEAKER
    resp = closed.post("/export", json={**body, "scan_id": scan_id})
    assert resp.status_code == 403
    assert sorted(p.name for p in (source / "clips").iterdir()) == before  # the refused move moved nothing
    assert not any(exports.iterdir())
    app = create_app(cache_dir=str(tmp_path / "c2"), source_root=str(source), export_root=str(exports), allow_move=True)
    client = TestClient(app)
    scan_id = client.post("/scan/folder", json=scan).json()["scan_id"]
    resp = client.post("/export", json={**body, "scan_id": scan_id, "speech": {**CORPUS, "speakers": ["s1"]}})
    assert resp.status_code == 200, resp.text
    remaining = client.get(f"/scan/{scan_id}").json()
    assert {c["speaker"] for c in remaining["clips"]} == {"s2"} and remaining["total"] == CLIPS_PER_SPEAKER


def test_select_and_export_routes_bind_a_consented_voice_and_refuse_after_a_restart(
    tmp_path: Path, fake_speakers: list[tuple[int, ...]]
) -> None:
    """The routes bind by distance, report it per row, and refuse once a restart has dropped the embeddings."""
    source = _corpus(tmp_path / "corpus", SPEAKERS[:2])
    _record(source / "consent.json", [source / "clips" / "s2_0.wav"])
    exports = tmp_path / "exports"
    exports.mkdir()

    def app() -> TestClient:
        return TestClient(
            create_app(cache_dir=str(tmp_path / "cache"), source_root=str(source), export_root=str(exports))
        )

    client = app()
    scan = {"folder": "", "target_profile": {"modality": "speech"}, "speech": {"speakers": True, "second_voice": True}}
    scan_id = client.post("/scan/folder", json=scan).json()["scan_id"]
    resp = client.post(f"/scan/{scan_id}/select", json={"speech": CONSENT})
    assert resp.status_code == 200, resp.text
    rows = {r["rel_path"]: r for r in resp.json()["rows"]}
    assert {rel for rel, r in rows.items() if not r["keep_reason"]} == {f"clips/s2_{i}.wav" for i in range(3)}
    assert rows["clips/s2_1.wav"]["consent_distance"] == pytest.approx(0.0, abs=1e-3)
    assert rows["clips/s1_0.wav"]["keep_reason"] == "consent:other-speaker"
    assert rows["clips/s1_0.wav"]["consent_distance"] > VOICE_CONSENT_MAX_DISTANCE
    served = client.get(f"/scan/{scan_id}").text
    assert "consent_distance" not in served  # the binding is the select's, never the stored scan's

    restarted = app()
    for path, body in ((f"/scan/{scan_id}/select", {"speech": CONSENT}), ("/export", {"speech": CONSENT})):
        resp = restarted.post(path, json={**body, "scan_id": scan_id, "dest": "set"})
        assert resp.status_code == 400 and "rescan with speech.speakers on" in resp.text, resp.text
    assert not (exports / "set").exists()


def test_speech_select_refuses_an_impossible_split(served: tuple[TestClient, str, Path]) -> None:
    client, scan_id, _ = served
    resp = client.post(f"/scan/{scan_id}/select", json={"split": {"held_out": 1, "group_by": "speaker"}})
    assert resp.status_code == 400 and "split refused" in resp.text


def test_a_clip_record_is_unchanged_by_selection(tmp_path: Path) -> None:
    summary = _scan(_corpus(tmp_path, SPEAKERS[:2]))
    before = [c.model_dump() for c in summary.clips]
    speech_select.plan_speech_selection(summary.clips, ExportRequest(dest=".", speech=CORPUS, max_keep=2))
    assert [c.model_dump() for c in summary.clips] == before


def test_clip_results_carry_the_corpus_gender_verbatim(tmp_path: Path) -> None:
    summary = _scan(_corpus(tmp_path, SPEAKERS[:2]))
    assert {c.speaker: c.gender for c in summary.clips} == {"s1": "female", "s2": "male"}


def test_cli_speech_scan_clusters_speakers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_speakers: list[tuple[int, ...]]
) -> None:
    from typer.testing import CliRunner

    from argus_curator.cli import app

    monkeypatch.setenv("CURATOR_CACHE_DIR", str(tmp_path / "cache"))
    root = _corpus(tmp_path / "corpus", SPEAKERS[:3])
    out = tmp_path / "summary.json"
    result = CliRunner().invoke(app, ["scan", str(root), "--modality", "speech", "--speakers", "--json", str(out)])
    assert result.exit_code == 0, result.output
    assert "Speakers (clustered):  3" in result.output
    assert len(ScanSummary.model_validate_json(out.read_text(encoding="utf-8")).speaker_clusters) == 3
    refused = CliRunner().invoke(app, ["scan", str(root), "--speakers"])
    assert refused.exit_code == 2 and "--modality speech" in refused.output


@pytest.mark.skipif(
    not (os.environ.get(speakers.MODEL_ENV) and runtime.onnxruntime_available()),
    reason="a real speaker-embedding run needs the model (argus-curator speaker-fetch) and onnxruntime",
)
def test_real_speaker_model_embeds_a_clip() -> None:  # pragma: no cover - needs the model
    session = speakers.get_session("cpu")
    a = speakers.embed(session, voice(3.0, seed=1), 16000, "cpu")
    b = speakers.embed(session, voice(3.0, seed=1), 16000, "cpu")
    assert a.shape == (speakers.EMBEDDING_DIM,) and float(a @ b) > 0.99


def test_a_speech_manifest_replaces_a_symlink_and_keeps_the_previous_file_when_a_row_raises(
    tmp_path: Path,
) -> None:
    from argus_curator.speech_export import write_manifest

    victim = tmp_path / "victim.jsonl"
    victim.write_text("keep me\n")
    path = tmp_path / "out" / "speech_manifest.jsonl"
    path.parent.mkdir()
    path.symlink_to(victim)
    write_manifest([], path)
    assert victim.read_text() == "keep me\n"
    assert path.is_file() and not path.is_symlink() and path.read_text(encoding="utf-8") == ""

    path.write_text('{"ok": true}\n', encoding="utf-8")

    class _Boom:
        def model_dump_json(self) -> str:
            raise KeyError("nope")

    with pytest.raises(KeyError):
        write_manifest([_Boom()], path)  # type: ignore[list-item]
    assert path.read_text(encoding="utf-8") == '{"ok": true}\n'


def test_the_speech_report_neutralises_formula_cells(tmp_path: Path) -> None:
    import csv

    from argus_curator.models import ClipResult
    from argus_curator.speech_export import write_report

    clip = ClipResult(rel_path=" =HYPERLINK(1).wav", abs_path="/tmp/x.wav", duplicate_of="+cmd.wav", accent="@x")
    report = tmp_path / "curation_report.csv"
    write_report([clip], {}, {}, {}, report)
    row = next(csv.DictReader(report.open(encoding="utf-8")))
    assert row["rel_path"] == "' =HYPERLINK(1).wav"
    assert row["duplicate_of"] == "'+cmd.wav"
    assert row["accent"] == "'@x"


def test_a_transcript_never_writes_through_a_link_planted_at_its_temp_name(tmp_path: Path) -> None:
    from argus_curator import speech_export

    victim = tmp_path / "victim.txt"
    victim.write_text("keep me")
    out = tmp_path / "out"
    out.mkdir()
    target = out / "clip.txt"
    # The temp name a writer could predict: the old one was .<name>.<pid>.tmp.
    (out / f".{target.name}.{os.getpid()}.tmp").symlink_to(victim)
    speech_export._write_transcript(target, "a line")
    assert victim.read_text() == "keep me"
    assert target.read_text(encoding="utf-8") == "a line\n" and not target.is_symlink()
