"""Body crops for MiVOLO (FaceConfig.mivolo_body): matching, masking, and the wiring into face detection."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image
from test_attributes import _App, _Face, _mv_row, _png
from test_attributes import _Session as _QueuedSession

from argus_curator import bodies, faces, fairface, mivolo
from argus_curator.models import FaceConfig, ImageResult


def _white(w: int = 400, h: int = 400) -> Image.Image:
    return Image.new("RGB", (w, h), (255, 255, 255))


def test_mivolo_body_is_off_by_default_and_implies_mivolo() -> None:
    cfg = FaceConfig()
    assert not cfg.mivolo_body and cfg.body_model == "cheetah_l"
    assert (cfg.body_min_score, cfg.body_min_size, cfg.body_min_visible, cfg.body_face_match) == (0.5, 50, 0.4, 0.0001)
    on = FaceConfig(mivolo_body=True)
    assert on.mivolo and on.enabled
    with pytest.raises(ValueError):
        FaceConfig(body_min_visible=1.5)


def test_only_the_insightface_modules_curator_reads_are_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    seen: dict = {}

    class _FaceAnalysis:
        def __init__(self, **kwargs) -> None:  # noqa: ANN003
            seen.update(kwargs)

        def prepare(self, **kwargs) -> None:  # noqa: ANN003
            pass

    app_mod = types.ModuleType("insightface.app")
    app_mod.FaceAnalysis = _FaceAnalysis  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "insightface", types.ModuleType("insightface"))
    monkeypatch.setitem(sys.modules, "insightface.app", app_mod)
    faces._build_app(FaceConfig(enabled=True, device="cpu"))
    assert seen["allowed_modules"] == ["detection", "recognition", "landmark_3d_68"]


def test_each_face_gets_its_own_body() -> None:
    people = np.array([[0, 0, 200, 400], [200, 0, 400, 400]], np.float32)
    left_face, right_face, stray = [60.0, 20.0, 80.0, 80.0], [260.0, 20.0, 80.0, 80.0], [500.0, 20.0, 40.0, 40.0]
    assert bodies.assign(people, [right_face, left_face, stray], 0.0001) == [1, 0, None]
    assert bodies.assign(np.empty((0, 4), np.float32), [left_face], 0.0001) == [None]
    # One body is never given two faces: the better match wins, the other runs face-only.
    two_on_one = bodies.assign(people[:1], [left_face, [70.0, 30.0, 60.0, 60.0]], 0.0001)
    assert sorted(two_on_one, key=str) == [0, None]


def test_match_score_is_mivolos_over_second_iou() -> None:
    body = np.array([0, 0, 100, 100], np.float64)
    inside = np.array([10, 10, 30, 30], np.float64)  # all of the face is inside the body
    assert bodies.match_score(body, inside) == pytest.approx((400 / 10000 + 1.0) / 2)
    assert bodies.match_score(body, np.array([200, 200, 210, 210], np.float64)) == 0.0


def test_body_crop_blacks_out_every_face_and_other_people() -> None:
    image = _white()
    people = np.array([[0, 0, 200, 400], [45, 300, 400, 400]], np.float32)
    own_face = [60.0, 20.0, 80.0, 80.0]
    crop = bodies.body_crop(image, people, 0, [own_face], min_size=50, min_visible=0.1)
    assert crop is not None and crop.size == (200, 400)
    px = np.asarray(crop)
    assert px[60, 100].tolist() == [0, 0, 0]  # its own face: the face channel carries that
    assert px[350, 100].tolist() == [0, 0, 0]  # the person overlapping it
    # That person's left edge, 45 px in, is within 30% of the crop's left border: it snaps to 0.
    assert px[350, 10].tolist() == [0, 0, 0]
    assert px[250, 100].tolist() == [255, 255, 255] and px[60, 20].tolist() == [255, 255, 255]

    # A face is masked as detected, never snapped to an edge.
    face_near_edge = [20.0, 200.0, 30.0, 30.0]
    px = np.asarray(bodies.body_crop(image, people[:1], 0, [face_near_edge], min_size=50, min_visible=0.1))
    assert px[210, 5].tolist() == [255, 255, 255] and px[210, 30].tolist() == [0, 0, 0]


def test_small_or_mostly_masked_bodies_are_not_used() -> None:
    image = _white()
    tiny = np.array([[0, 0, 40, 400]], np.float32)
    assert bodies.body_crop(image, tiny, 0, [], min_size=50, min_visible=0.4) is None
    body = np.array([[0, 0, 200, 200]], np.float32)
    big_face = [0.0, 0.0, 200.0, 150.0]  # 75% of the crop
    assert bodies.body_crop(image, body, 0, [big_face], min_size=50, min_visible=0.4) is None
    assert bodies.body_crop(image, body, 0, [big_face], min_size=50, min_visible=0.2) is not None


def test_mivolo_person_channels_carry_the_body_or_the_absent_image() -> None:
    face, body = Image.new("RGB", (60, 80), (200, 100, 50)), Image.new("RGB", (100, 300), (10, 200, 30))
    batch = mivolo.preprocess([face, face], [body, None])
    assert batch.shape == (2, 6, 384, 384)
    assert np.allclose(batch[1, 3:], mivolo._ABSENT)
    assert not np.allclose(batch[0, 3:], mivolo._ABSENT)
    assert np.allclose(batch[0, 3:], mivolo._channels(body))
    assert np.allclose(mivolo.preprocess([face]), mivolo.preprocess([face], [None]))  # face-only unchanged


class _Session(_QueuedSession):
    """The queued-answer MiVOLO session, keeping every batch it was given."""

    def __init__(self, rows: list[np.ndarray], shape: tuple[int, ...]) -> None:
        super().__init__(rows, shape)
        self.batches: list[np.ndarray] = []

    def run(self, outputs, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:  # noqa: ANN001
        self.batches.append(feeds["input"])
        return super().run(outputs, feeds)


class _Detector:
    """A person detector answering fixed boxes, recording the score floor it was asked for."""

    def __init__(self, rows: list[list[float]]) -> None:
        self.rows = np.array(rows, np.float32)
        self.asked: list[float] = []

    def detect_candidates(self, bgr: np.ndarray, min_score: float) -> np.ndarray:
        self.asked.append(min_score)
        return self.rows[self.rows[:, 4] >= min_score]


def _scan(monkeypatch: pytest.MonkeyPatch, cfg: FaceConfig, detector: _Detector | None) -> tuple[ImageResult, _Session]:
    app = _App({400: [_Face([100, 100, 200, 200], [1, 0, 0]), _Face([250, 20, 280, 50], [0, 1, 0])]})
    session = _Session([_mv_row(True, 30.0), _mv_row(False, 50.0)], (6, 384, 384))
    monkeypatch.setattr(faces, "_get_app", lambda cfg, **kw: app)
    monkeypatch.setattr(mivolo, "get_session", lambda providers: session)
    monkeypatch.setattr(fairface, "get_session", lambda providers: pytest.fail("FairFace was not asked for"))
    monkeypatch.setattr(bodies, "get_detector", lambda cfg: detector)
    r = ImageResult(rel_path="x.png", abs_path="/d/x.png", passed=True)
    faces.detect_faces([r], [("x.png", "/d/x.png", _png(400))], cfg)
    return r, session


def test_detect_faces_gives_mivolo_the_bodies_it_found(monkeypatch: pytest.MonkeyPatch) -> None:
    # A body around the first face only; the second face has none and runs face-only.
    detector = _Detector([[50, 50, 250, 400, 0.9], [300, 0, 400, 400, 0.3]])
    r, session = _scan(monkeypatch, FaceConfig(mivolo_body=True, body_min_score=0.6), detector)
    assert detector.asked == [0.6]
    assert [f.attributes.age_body for f in r.faces] == [True, False]
    assert [f.attributes.age_years for f in r.faces] == [30.0, 50.0]
    persons = session.batches[0][:, 3:]
    assert not np.allclose(persons[0], mivolo._ABSENT) and np.allclose(persons[1], mivolo._ABSENT)
    assert len(r.people) == 1 and r.people[0].face_index == 0 and r.people[0].age_years == 30.0
    assert r.faces[0].person_index == 0 and r.faces[1].person_index is None


def test_without_mivolo_body_no_body_is_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    r, session = _scan(monkeypatch, FaceConfig(mivolo=True), None)
    assert [f.attributes.age_body for f in r.faces] == [False, False]
    assert np.allclose(session.batches[0][:, 3:], mivolo._ABSENT)


def test_a_failed_body_detection_leaves_faces_face_only(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Broken(_Detector):
        def detect_candidates(self, bgr: np.ndarray, min_score: float) -> np.ndarray:
            raise RuntimeError("onnx exploded")

    r, _session = _scan(monkeypatch, FaceConfig(mivolo_body=True), _Broken([]))
    assert [f.attributes.age_body for f in r.faces] == [False, False]
    assert [f.attributes.age_years for f in r.faces] == [30.0, 50.0]  # ages still read, face-only
    assert r.people == []


def test_unmatched_body_gets_a_body_only_age(monkeypatch: pytest.MonkeyPatch) -> None:
    # A person box far from both faces; MiVOLO runs with absent face channels.
    detector = _Detector([[50, 50, 250, 400, 0.9], [0, 300, 80, 399, 0.85]])
    session = _Session([_mv_row(True, 30.0), _mv_row(False, 50.0), _mv_row(True, 22.0)], (6, 384, 384))
    app = _App({400: [_Face([100, 100, 200, 200], [1, 0, 0]), _Face([250, 20, 280, 50], [0, 1, 0])]})
    monkeypatch.setattr(faces, "_get_app", lambda cfg, **kw: app)
    monkeypatch.setattr(mivolo, "get_session", lambda providers: session)
    monkeypatch.setattr(fairface, "get_session", lambda providers: pytest.fail("FairFace was not asked for"))
    monkeypatch.setattr(bodies, "get_detector", lambda cfg: detector)
    r = ImageResult(rel_path="x.png", abs_path="/d/x.png", passed=True)
    faces.detect_faces([r], [("x.png", "/d/x.png", _png(400))], FaceConfig(mivolo_body=True))
    unmatched = [p for p in r.people if p.face_index is None]
    assert len(unmatched) == 1 and unmatched[0].age_years == 22.0 and unmatched[0].age_body is True
    assert len(session.batches) == 2  # faces, then body-only


def test_associate_nsfw_gives_each_detection_to_the_overlapping_person() -> None:
    from argus_cortex.moderation import NsfwDetection, NsfwResult

    from argus_curator.models import PersonDetection
    from argus_curator.selection import minor_exposed

    young = PersonDetection(bbox=[0, 0, 100, 200], det_score=0.9, age_years=16.0, age_source="mivolo", age_body=True)
    adult = PersonDetection(bbox=[200, 0, 100, 200], det_score=0.8, age_years=32.0, age_source="mivolo", age_body=True)
    r = ImageResult(
        rel_path="x.png",
        abs_path="/d/x.png",
        passed=True,
        people=[young, adult],
        nsfw=NsfwResult(
            detections=[
                NsfwDetection(label="FEMALE_BREAST_EXPOSED", score=0.8, bbox=[210, 80, 40, 40]),
                NsfwDetection(label="FACE_FEMALE", score=0.9, bbox=[10, 10, 30, 30]),
            ],
            labels={"FEMALE_BREAST_EXPOSED": 0.8, "FACE_FEMALE": 0.9},
            exposed_score=0.8,
        ),
    )
    bodies.associate_nsfw([r])
    assert r.people[1].exposed_score == 0.8 and r.people[0].exposed_score == 0.0
    assert "FACE_FEMALE" in r.people[0].nsfw_labels
    # Exposed is on the adult; the younger person does not overlap it.
    assert minor_exposed(r) is False
    on_young = NsfwDetection(label="FEMALE_BREAST_EXPOSED", score=0.8, bbox=[20, 80, 40, 40])
    r.nsfw.detections.append(on_young)
    assert minor_exposed(r) is True
    r.nsfw.detections.remove(on_young)

    # Exposure in the overlap of a young and an adult box goes to one of them, but flags.
    overlapping = r.model_copy(deep=True)
    overlapping.people[0].bbox = [150, 0, 200, 250]
    bodies.associate_nsfw([overlapping])
    assert overlapping.people[1].exposed_score == 0.8 and minor_exposed(overlapping) is True

    # A young face with no person box: its body is not located, so exposure fails closed.
    from argus_curator.models import FaceAttributes, FaceDetection

    faceless_body = r.model_copy(deep=True)
    faceless_body.people = [faceless_body.people[1]]
    faceless_body.faces = [
        FaceDetection(
            bbox=[400, 10, 30, 30], det_score=0.9, attributes=FaceAttributes(age_years=15.0, age_source="mivolo")
        )
    ]
    bodies.associate_nsfw([faceless_body])
    assert minor_exposed(faceless_body) is True

    # An exposed box that overlaps nobody still flags any young age (fail closed).
    r.people[0].exposed_score = 0.0
    r.people[0].nsfw_labels = {}
    r.people[1].exposed_score = 0.0
    r.people[1].nsfw_labels = {}
    r.nsfw = NsfwResult(
        detections=[NsfwDetection(label="FEMALE_BREAST_EXPOSED", score=0.8, bbox=[900, 900, 20, 20])],
        labels={"FEMALE_BREAST_EXPOSED": 0.8},
        exposed_score=0.8,
    )
    bodies.associate_nsfw([r])
    assert minor_exposed(r) is True
