"""Scan prefilter: the rules, where the scan applies them, and what skips the dropped images."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from argus_cortex.moderation import NsfwResult
from test_guard import _png, singguard  # noqa: F401 - the fake SingGuard server fixture
from test_guard import _SingGuard as _FakeGuard

from argus_curator import guard, prefilter
from argus_curator.models import (
    FaceAttributes,
    FaceConfig,
    FaceDetection,
    GarmentConfig,
    GuardConfig,
    ImageResult,
    NsfwConfig,
    PersonDetection,
    PrefilterConfig,
    ScanConfig,
    ScanSummary,
    SceneConfig,
    SetAge,
    TargetProfile,
)
from argus_curator.scanner import scan_folder

_ALL = PrefilterConfig(drop_nsfw=True, drop_no_face=True, drop_no_age=True)


def _face(years: float | None = 30.0, source: str = "mivolo") -> FaceDetection:
    attrs = FaceAttributes(age_years=years, age_source=source, age="20-29") if years is not None else None
    return FaceDetection(bbox=[0, 0, 10, 10], det_score=0.9, primary=True, attributes=attrs)


def _row(
    *,
    exposed: float | None = None,
    faces: list[FaceDetection] | None = None,
    passed: bool = True,
    reason: str | None = None,
) -> ImageResult:
    row = ImageResult(rel_path="x.png", abs_path="/d/x.png", passed=passed, reject_reason=reason)
    if exposed is not None:
        labels = {"FEMALE_BREAST_EXPOSED": exposed} if exposed else {}
        row.nsfw = NsfwResult(labels=labels, exposed_score=exposed)
    row.faces = faces if faces is not None else [_face()]
    row.face_count = len(row.faces)
    return row


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------


def test_nsfw_drops_an_exposed_reading_of_at_least_0_2() -> None:
    cfg = PrefilterConfig(drop_nsfw=True)
    assert prefilter.reason(_row(exposed=0.2), cfg) == "prefilter:nsfw"
    assert prefilter.reason(_row(exposed=0.19), cfg) is None
    assert prefilter.reason(_row(exposed=0.0), cfg) is None
    # Either reading counts: the recorded score or an exposed class's label score.
    only_label = _row()
    only_label.nsfw = NsfwResult(labels={"ANUS_EXPOSED": 0.5})
    assert prefilter.reason(only_label, cfg) == "prefilter:nsfw"
    only_score = _row()
    only_score.nsfw = NsfwResult(exposed_score=0.4)
    assert prefilter.reason(only_score, cfg) == "prefilter:nsfw"
    # A covered class is not exposure.
    covered = _row()
    covered.nsfw = NsfwResult(labels={"FEMALE_BREAST_COVERED": 0.9})
    assert prefilter.reason(covered, cfg) is None


def test_nsfw_keeps_an_image_nudenet_could_not_score() -> None:
    assert prefilter.reason(_row(exposed=None), PrefilterConfig(drop_nsfw=True)) is None


def test_no_face_drops_a_faceless_image() -> None:
    cfg = PrefilterConfig(drop_no_face=True)
    assert prefilter.reason(_row(faces=[]), cfg) == "prefilter:no-face"
    assert prefilter.reason(_row(), cfg) is None


def test_no_age_needs_a_mivolo_body_only_or_set_age() -> None:
    cfg = PrefilterConfig(drop_no_age=True)
    assert prefilter.reason(_row(faces=[_face(30.0)]), cfg) is None
    # A FairFace bracket is not a MiVOLO age; nor is a face with no attributes.
    assert prefilter.reason(_row(faces=[_face(30.0, source="fairface")]), cfg) == "prefilter:no-age"
    assert prefilter.reason(_row(faces=[_face(None)]), cfg) == "prefilter:no-age"
    assert prefilter.reason(_row(faces=[]), cfg) == "prefilter:no-age"

    body_only = _row(faces=[])
    body_only.people = [PersonDetection(bbox=[0, 0, 5, 9], det_score=0.8, age_years=31.0)]
    assert prefilter.reason(body_only, cfg) is None
    # A person matched to a face is that face's body, not a body-only age.
    matched = _row(faces=[_face(None)])
    matched.people = [PersonDetection(bbox=[0, 0, 5, 9], det_score=0.8, face_index=0, age_years=31.0)]
    assert prefilter.reason(matched, cfg) == "prefilter:no-age"

    set_aged = _row(faces=[])
    set_aged.set_age = SetAge(years=27.0, cluster_id="face_1", faces=6, share=1.0)
    assert prefilter.reason(set_aged, cfg) is None


def test_the_first_matching_rule_names_the_reason() -> None:
    assert prefilter.reason(_row(exposed=0.9, faces=[]), _ALL) == "prefilter:nsfw"
    assert prefilter.reason(_row(exposed=0.0, faces=[]), _ALL) == "prefilter:no-face"
    assert prefilter.reason(_row(exposed=0.0, faces=[_face(None)]), _ALL) == "prefilter:no-age"
    assert prefilter.reason(_row(exposed=0.0), _ALL) is None


def test_off_by_default() -> None:
    assert prefilter.reason(_row(exposed=0.9, faces=[]), PrefilterConfig()) is None


def test_apply_rejects_in_place_and_leaves_hard_rejects_alone() -> None:
    exposed = _row(exposed=0.9)
    blurry = _row(exposed=0.9, faces=[], passed=False, reason="blurry (sharpness=3.0 < 100.0)")
    kept = _row(exposed=0.0)
    dropped = prefilter.apply([exposed, blurry, kept], _ALL)
    assert dropped == [exposed]
    assert (exposed.passed, exposed.reject_reason) == (False, "prefilter:nsfw")
    assert blurry.reject_reason.startswith("blurry")
    assert kept.passed and kept.reject_reason is None
    assert prefilter.is_prefiltered(exposed) and not prefilter.is_prefiltered(blurry)


def test_problem_names_the_missing_face_check() -> None:
    assert prefilter.problem(PrefilterConfig(drop_nsfw=True), FaceConfig()) is None
    assert "faces.enabled" in prefilter.problem(PrefilterConfig(drop_no_face=True), FaceConfig())
    assert "faces.enabled" in prefilter.problem(PrefilterConfig(drop_no_age=True), FaceConfig())
    assert "faces.mivolo" in prefilter.problem(PrefilterConfig(drop_no_age=True), FaceConfig(enabled=True))
    assert prefilter.problem(_ALL, FaceConfig(mivolo=True)) is None


def test_scan_refuses_a_face_rule_without_face_detection(dataset: Path) -> None:
    with pytest.raises(ValueError, match="face detection"):
        scan_folder(dataset, prefilter_cfg=PrefilterConfig(drop_no_face=True))


# ---------------------------------------------------------------------------
# SingGuard skips prefiltered images
# ---------------------------------------------------------------------------


def test_check_images_skips_prefiltered_rows(singguard: type[_FakeGuard]) -> None:  # noqa: F811
    items = [(f"{n}.png", f"/d/{n}", _png(10)) for n in ("kept", "dropped", "blurry")]
    results = [
        ImageResult(rel_path="kept.png", abs_path="/d/kept", passed=True),
        ImageResult(rel_path="dropped.png", abs_path="/d/dropped", passed=False, reject_reason="prefilter:nsfw"),
        ImageResult(rel_path="blurry.png", abs_path="/d/blurry", passed=False, reject_reason="blurry"),
    ]
    seen: list[tuple[int, int]] = []
    done: list[tuple[str, int, int]] = []
    cfg = GuardConfig(variants=["adult"])
    assert guard.check_images(
        results,
        items,
        cfg,
        progress=lambda d, t: seen.append((d, t)),
        on_done=lambda r, d, t: done.append((r.rel_path, d, t)),
    )
    by = {r.rel_path: r for r in results}
    assert by["dropped.png"].guard is None and by["dropped.png"].guard_variants is None
    assert by["kept.png"].guard is not None and set(by["kept.png"].guard_variants) == {"default", "adult"}
    assert by["blurry.png"].guard is not None  # other rejects are still judged
    # Two images under two policies: the totals never count the dropped one.
    assert seen[-1] == (4, 4) and {t for _r, _d, t in done} == {4}
    assert "dropped.png" not in {r for r, _d, _t in done}
    assert len(singguard.seen) == 4


# ---------------------------------------------------------------------------
# In a scan
# ---------------------------------------------------------------------------


class _Input:
    name = "input"


class _SceneSession:
    def get_inputs(self) -> list[_Input]:
        return [_Input()]

    def run(self, _outputs, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        return [np.zeros((feeds["input"].shape[0], 365))]


class _GarmentSession:
    def get_inputs(self) -> list[_Input]:
        return [_Input()]

    def run(self, _outputs, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        n = feeds["input"].shape[0]
        logits = np.zeros((n, 100, 47))
        logits[:, :, 46] = 10.0  # every query: no object
        return [logits, np.tile([0.5, 0.5, 0.4, 0.6], (n, 100, 1))]


_KEPT, _EXPOSED, _FACELESS = "personA/s1/img.png", "personA/s1/dup.png", "personA/s2/img.png"


def _fake_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Faces: an age on the two personA/s1 frames, none on s2. NudeNet: dup.png and
    the blurry reject exposed, s2 unscored, the rest clean."""
    from argus_curator import faces as faces_mod
    from argus_curator import garments as garments_mod
    from argus_curator import nsfw as nsfw_mod
    from argus_curator import scenes as scenes_mod
    from argus_curator.attributes import RawAttributes

    same_person = np.zeros(512, dtype=np.float32)
    same_person[0] = 1.0

    def fake_faces(results, items, cfg, progress=None, on_done=None):
        detected = faces_mod.DetectedFaces([], [], [])
        passing = [r for r in results if r.passed]
        for i, r in enumerate(passing):
            if r.rel_path != _FACELESS:
                face = _face(30.0)
                r.faces, r.face_count, r.primary_face_attributes = [face], 1, face.attributes
                detected.embeddings.append(same_person)
                detected.meta.append((r.rel_path, 0, [0, 0, 10, 10], 0.9, 1.0, 100.0))
                detected.raws.append(RawAttributes(mivolo_age=30.0))
            if on_done is not None:
                on_done(r, i + 1, len(passing))
        return detected

    def fake_nsfw(results, items, cfg, progress=None, on_done=None):
        for i, r in enumerate(results):
            if r.rel_path in (_EXPOSED, "personB/blurry.png"):
                r.nsfw = NsfwResult(labels={"FEMALE_GENITALIA_EXPOSED": 0.7}, exposed_score=0.7)
            elif r.rel_path != _FACELESS:
                r.nsfw = NsfwResult()
            if on_done is not None:
                on_done(r, i + 1, len(results))
        return True

    monkeypatch.setattr(faces_mod, "detect_faces", fake_faces)
    monkeypatch.setattr(nsfw_mod, "check_images", fake_nsfw)
    monkeypatch.setattr(nsfw_mod, "resolve_enabled", lambda cfg: True)
    monkeypatch.setattr(scenes_mod, "get_session", lambda provs: _SceneSession())
    monkeypatch.setattr(garments_mod, "get_session", lambda provs: _GarmentSession())


def test_scan_drops_before_scenes_garments_and_singguard(
    dataset: Path,
    monkeypatch: pytest.MonkeyPatch,
    singguard: type[_FakeGuard],  # noqa: F811
) -> None:
    _fake_checks(monkeypatch)
    chunks: list[dict] = []
    summary = scan_folder(
        dataset,
        faces_cfg=FaceConfig(mivolo=True),
        scene_cfg=SceneConfig(enabled=True),
        garment_cfg=GarmentConfig(enabled=True),
        nsfw_cfg=NsfwConfig(),
        prefilter_cfg=PrefilterConfig(drop_nsfw=True, drop_no_face=True),
        chunk=chunks.append,
    )
    by = {r.rel_path: r for r in summary.results}
    assert len(summary.results) == 5  # dropped images stay in the results
    assert (by[_EXPOSED].passed, by[_EXPOSED].reject_reason) == (False, "prefilter:nsfw")
    assert (by[_FACELESS].passed, by[_FACELESS].reject_reason) == (False, "prefilter:no-face")
    assert by["personB/blurry.png"].reject_reason.startswith("blurry")  # a hard reject keeps its reason
    assert by[_KEPT].passed and by[_KEPT].reject_reason is None
    assert summary.passed == 1 and summary.rejected == 4
    assert summary.reject_reasons["prefilter:nsfw"] == 1 and summary.reject_reasons["prefilter:no-face"] == 1
    assert summary.prefilter_config == PrefilterConfig(drop_nsfw=True, drop_no_face=True)

    # Scenes and garments ran on the kept image only.
    assert by[_KEPT].scene is not None and by[_KEPT].garments == []
    for rel in (_EXPOSED, _FACELESS):
        assert by[rel].scene is None and by[rel].garments is None
    # SingGuard judged the kept image and the two hard rejects, never the dropped ones.
    assert by[_EXPOSED].guard is None and by[_FACELESS].guard is None
    assert all(by[rel].guard is not None for rel in (_KEPT, "personB/blurry.png", "personB/tiny.png"))
    assert len(singguard.seen) == 3
    # The near-duplicate pair no longer pairs: dup.png is out of the passing set.
    assert not by[_KEPT].is_duplicate and not by[_EXPOSED].is_duplicate and summary.duplicates == 0

    # A streaming client hears about the drops before the late phases.
    phases = [c["phase"] for c in chunks]
    (drop_chunk,) = [c for c in chunks if c["phase"] == "prefilter"]
    assert phases.index("prefilter") > phases.index("identities")
    assert phases.index("prefilter") < min(phases.index(p) for p in ("scenes", "garments", "guard"))
    assert {r["rel_path"]: (r["passed"], r["reject_reason"]) for r in drop_chunk["results"]} == {
        _EXPOSED: (False, "prefilter:nsfw"),
        _FACELESS: (False, "prefilter:no-face"),
    }


def test_the_set_age_counts_as_an_age(dataset: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_checks(monkeypatch)
    no_age = PrefilterConfig(drop_no_age=True)
    with_set = scan_folder(
        dataset, faces_cfg=FaceConfig(set_age=True, set_age_min_faces=2), nsfw_cfg=NsfwConfig(), prefilter_cfg=no_age
    )
    by = {r.rel_path: r for r in with_set.results}
    assert by[_FACELESS].set_age is not None and by[_FACELESS].passed

    without = scan_folder(dataset, faces_cfg=FaceConfig(mivolo=True), nsfw_cfg=NsfwConfig(), prefilter_cfg=no_age)
    by = {r.rel_path: r for r in without.results}
    assert by[_FACELESS].reject_reason == "prefilter:no-age"
    assert by[_KEPT].passed and by[_EXPOSED].passed  # drop_nsfw is off


def test_a_scan_without_a_prefilter_drops_nothing(dataset: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_checks(monkeypatch)
    summary = scan_folder(dataset, faces_cfg=FaceConfig(mivolo=True), nsfw_cfg=NsfwConfig())
    assert summary.prefilter_config == PrefilterConfig()
    assert not any((r.reject_reason or "").startswith("prefilter:") for r in summary.results)
    assert summary.passed == 3


def test_prefilter_config_round_trips_on_the_summary() -> None:
    summary = ScanSummary(
        scan_id="s",
        folder="/d",
        target_profile=TargetProfile(),
        config=ScanConfig(),
        faces_config=FaceConfig(),
        prefilter_config=PrefilterConfig(drop_nsfw=True, drop_no_age=True),
        total=0,
        passed=0,
        rejected=0,
        duplicates=0,
        similar_clusters=0,
    )
    dumped = summary.model_dump(mode="json")
    assert dumped["prefilter_config"] == {"drop_nsfw": True, "drop_no_face": False, "drop_no_age": True}
    assert ScanSummary.model_validate_json(summary.model_dump_json()).prefilter_config == summary.prefilter_config
    # A summary stored before the prefilter existed still loads, with every rule off.
    del dumped["prefilter_config"]
    assert ScanSummary.model_validate(dumped).prefilter_config == PrefilterConfig()
