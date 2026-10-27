"""Face attributes (FairFace + MiVOLO): maths, crops, merging, sweep, filters, report.

None of this needs onnxruntime, torch or InsightFace: the detector and the
model sessions are stand-ins, so the tests pin our side of each contract.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from argus_curator import attributes, faces, fairface, mivolo, runtime
from argus_curator.attributes import RawAttributes
from argus_curator.export import write_report
from argus_curator.models import (
    FACE_AGES,
    FACE_RACES,
    ExportRequest,
    FaceAttributes,
    FaceCluster,
    FaceConfig,
    FaceDetection,
    ImageResult,
)
from argus_curator.selection import decide_selection


def _logits(race: int, gender: int, age: int, peak: float = 4.0) -> np.ndarray:
    row = np.zeros(18)
    row[race] = peak
    row[7 + gender] = peak
    row[9 + age] = peak
    return row


def _ff(race: int, gender: int, age: int, peak: float = 4.0) -> np.ndarray:
    return fairface.split_probs(_logits(race, gender, age, peak)[None])[0]


# ---------------------------------------------------------------------------
# FairFace
# ---------------------------------------------------------------------------


def test_label_order_matches_fairface_reference() -> None:
    # predict.py: race_scores_fair / age_scores_fair index order.
    assert FACE_RACES == (
        "white",
        "black",
        "latino_hispanic",
        "east_asian",
        "southeast_asian",
        "indian",
        "middle_eastern",
    )
    assert FACE_AGES[0] == "0-2" and FACE_AGES[-1] == "70+" and len(FACE_AGES) == 9


def test_fairface_heads_are_softmaxed_independently() -> None:
    probs = fairface.split_probs(np.vstack([_logits(3, 1, 4), np.zeros(18)]))
    assert probs.shape == (2, 18)
    for row in probs:
        assert row[:7].sum() == pytest.approx(1.0)
        assert row[7:9].sum() == pytest.approx(1.0)
        assert row[9:].sum() == pytest.approx(1.0)
    a = attributes.to_attributes(RawAttributes(fairface=probs[0]))
    assert (a.race, a.gender, a.age) == ("east_asian", "female", "30-39")
    assert a.gender_source == a.age_source == "fairface"
    assert a.age_years is None
    assert a.race_conf == pytest.approx(np.exp(4.0) / (np.exp(4.0) + 6), abs=1e-4)
    # Uniform logits: confidence is exactly 1 / classes for each head.
    flat = attributes.to_attributes(RawAttributes(fairface=probs[1]))
    assert flat.race_conf == pytest.approx(1 / 7, abs=1e-4)
    assert flat.gender_conf == pytest.approx(0.5)


def test_similarity_recovers_a_known_transform() -> None:
    theta, scale, t = np.deg2rad(20), 1.7, np.array([12.0, -5.0])
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    src = fairface._TEMPLATE
    dst = src @ (scale * rot).T + t
    m = fairface._similarity(src, dst)
    np.testing.assert_allclose(m[:, :2], scale * rot, atol=1e-9)
    np.testing.assert_allclose(m[:, 2], t, atol=1e-9)


def test_align_face_maps_keypoints_onto_template() -> None:
    # A face twice chip size, keypoints exactly where the template expects them
    # scaled x2: the chip must be that image downsampled, i.e. a pure x0.5 warp.
    size = fairface.INPUT_SIZE * 2
    yy, xx = np.mgrid[0:size, 0:size]
    arr = np.stack([xx % 256, yy % 256, (xx // 2) % 256], axis=-1).astype(np.uint8)
    chip = fairface.align_face(Image.fromarray(arr), fairface._TEMPLATE * 2)
    assert chip.size == (fairface.INPUT_SIZE, fairface.INPUT_SIZE)
    out = np.asarray(chip)
    # Pixel (x, y) of the chip samples source (2x, 2y).
    assert abs(int(out[10, 50, 0]) - 100) <= 1
    assert abs(int(out[50, 10, 1]) - 100) <= 1


def test_fairface_preprocess_is_imagenet_nchw() -> None:
    chip = Image.new("RGB", (fairface.INPUT_SIZE,) * 2, (255, 0, 0))
    batch = fairface.preprocess([chip, chip])
    assert batch.shape == (2, 3, 224, 224) and batch.dtype == np.float32
    assert batch[0, 0, 0, 0] == pytest.approx((1.0 - 0.485) / 0.229, rel=1e-5)
    assert batch[0, 1, 0, 0] == pytest.approx((0.0 - 0.456) / 0.224, rel=1e-5)


@pytest.mark.parametrize("module", [fairface, mivolo])
def test_model_path_from_env(module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    assert module.model_path() == runtime.MODEL_DIR / module.DEFAULT_MODEL_NAME
    monkeypatch.setenv(module.MODEL_ENV, str(tmp_path / "m.onnx"))
    assert module.model_path() == tmp_path / "m.onnx"
    assert not attributes.fairface_available() or module is mivolo
    assert not attributes.mivolo_available() or module is fairface


# ---------------------------------------------------------------------------
# MiVOLO
# ---------------------------------------------------------------------------


def test_mivolo_letterbox_centres_and_pads_black() -> None:
    # 100x50 white crop -> scaled to 384x192, centred vertically with black bars.
    out = np.asarray(mivolo.letterbox(Image.new("RGB", (100, 50), (255, 255, 255))))
    assert out.shape == (384, 384, 3)
    assert out[0, 192].tolist() == [0, 0, 0] and out[383, 192].tolist() == [0, 0, 0]
    assert out[192, 0].tolist() == [255, 255, 255] and out[192, 383].tolist() == [255, 255, 255]
    assert out[95, 192].sum() == 0 and out[96, 192].sum() == 765  # bar is exactly 96 px


def test_mivolo_preprocess_feeds_absent_person() -> None:
    batch = mivolo.preprocess([Image.new("RGB", (384, 384), (255, 0, 0))])
    assert batch.shape == (1, 6, 384, 384) and batch.dtype == np.float32
    assert batch[0, 0, 5, 5] == pytest.approx((1.0 - 0.485) / 0.229, rel=1e-5)
    # Upstream's missing-body input: an all-black image, normalised.
    assert batch[0, 3, 5, 5] == pytest.approx(-0.485 / 0.229, rel=1e-5)
    assert batch[0, 5, 5, 5] == pytest.approx(-0.406 / 0.225, rel=1e-5)


def test_mivolo_decode_matches_checkpoint_config() -> None:
    gender, age = mivolo.decode(np.array([[2.0, 0.0, 0.0], [0.0, 1.0, -0.5], [0.0, 0.0, -9.0]]))
    assert gender[0].argmax() == 0 and gender[1].argmax() == 1
    np.testing.assert_allclose(gender.sum(axis=1), 1.0)
    # age = raw * (122 - 0) + 61, clipped to the model's range.
    assert age.tolist() == pytest.approx([61.0, 0.0, 0.0])


def test_mivolo_face_crop_clamps_and_rejects_degenerate_boxes() -> None:
    image = Image.new("RGB", (100, 80))
    assert mivolo.face_crop(image, [-10, -10, 50, 50]).size == (40, 40)
    assert mivolo.face_crop(image, [90, 70, 40, 40]).size == (10, 10)
    assert mivolo.face_crop(image, [200, 200, 10, 10]) is None


# ---------------------------------------------------------------------------
# Merging both models
# ---------------------------------------------------------------------------


def test_age_bracket_boundaries() -> None:
    assert attributes.age_bracket(0) == "0-2"
    assert attributes.age_bracket(2.9) == "0-2"
    assert attributes.age_bracket(3) == "3-9"
    assert attributes.age_bracket(29.99) == "20-29"
    assert attributes.age_bracket(70) == "70+"
    assert attributes.age_bracket(110) == "70+"


def test_mivolo_supersedes_fairface_for_gender_and_age() -> None:
    raw = RawAttributes(fairface=_ff(0, 0, 3), mivolo_gender=np.array([0.1, 0.9]), mivolo_age=34.26)
    a = attributes.to_attributes(raw)
    assert a.race == "white" and a.race_conf is not None  # race only ever from FairFace
    assert (a.gender, a.gender_conf, a.gender_source) == ("female", 0.9, "mivolo")
    assert (a.age, a.age_years, a.age_conf, a.age_source) == ("30-39", 34.3, None, "mivolo")


def test_mivolo_alone_has_no_race() -> None:
    a = attributes.to_attributes(RawAttributes(mivolo_gender=np.array([0.8, 0.2]), mivolo_age=8.0))
    assert a.race is None and a.race_conf is None
    assert (a.gender, a.age) == ("male", "3-9")
    assert attributes.to_attributes(RawAttributes()) is None


def test_pool_averages_probabilities_and_takes_median_age() -> None:
    raws = [
        RawAttributes(fairface=_ff(0, 0, 3), mivolo_gender=np.array([0.9, 0.1]), mivolo_age=30.0),
        RawAttributes(fairface=_ff(0, 0, 3), mivolo_gender=np.array([0.8, 0.2]), mivolo_age=31.0),
        RawAttributes(fairface=_ff(5, 1, 3, peak=2.0), mivolo_gender=np.array([0.2, 0.8]), mivolo_age=70.0),
        RawAttributes(),  # a face neither model could read
    ]
    a = attributes.pool(raws)
    assert a.race == "white" and a.gender == "male"
    assert a.age_years == 31.0  # an outlier frame cannot drag the median
    assert a.gender_conf == pytest.approx((0.9 + 0.8 + 0.2) / 3, abs=1e-4)
    assert attributes.pool([RawAttributes()]) is None


# ---------------------------------------------------------------------------
# Sweep wiring with a stand-in detector and model sessions
# ---------------------------------------------------------------------------


class _Face:
    def __init__(self, bbox: list[float], emb: list[float]) -> None:
        self.bbox = np.array(bbox, dtype=np.float32)
        self.det_score = 0.9
        v = np.array(emb, dtype=np.float32)
        self.normed_embedding = v / np.linalg.norm(v)
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        self.kps = np.array(
            [
                [x1 + 0.3 * w, y1 + 0.4 * h],
                [x1 + 0.7 * w, y1 + 0.4 * h],
                [x1 + 0.5 * w, y1 + 0.6 * h],
                [x1 + 0.35 * w, y1 + 0.8 * h],
                [x1 + 0.65 * w, y1 + 0.8 * h],
            ],
            dtype=np.float32,
        )
        self.pose = None


class _App:
    def __init__(self, by_width: dict[int, list[_Face]]) -> None:
        self.by_width = by_width

    def get(self, bgr: np.ndarray) -> list[_Face]:
        return self.by_width[bgr.shape[1]]


class _Input:
    name = "input"


class _Session:
    """Returns queued output rows, one per face, in call order."""

    def __init__(self, rows: list[np.ndarray], shape: tuple[int, ...]) -> None:
        self.rows = list(rows)
        self.shape = shape

    def get_inputs(self) -> list[_Input]:
        return [_Input()]

    def run(self, _outputs, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        n = feeds["input"].shape[0]
        assert feeds["input"].shape[1:] == self.shape
        out, self.rows = self.rows[:n], self.rows[n:]
        return [np.vstack(out)]


def _png(width: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, 400), (128, 128, 128)).save(buf, "PNG")
    return buf.getvalue()


def _mv_row(female: bool, years: float) -> np.ndarray:
    return np.array([0.0, 3.0, (years - 61.0) / 122.0] if female else [3.0, 0.0, (years - 61.0) / 122.0])


def _sweep(monkeypatch: pytest.MonkeyPatch, cfg: FaceConfig, ff_rows=None, mv_rows=None):
    same = [1.0, 0.0, 0.0]
    app = _App(
        {
            400: [_Face([100, 100, 300, 300], same)],
            401: [_Face([100, 100, 300, 300], same), _Face([0, 0, 20, 20], [0.0, 1.0, 0.0])],
            402: [_Face([100, 100, 300, 300], same)],
        }
    )
    ff_session = _Session(ff_rows or [], (3, 224, 224))
    mv_session = _Session(mv_rows or [], (6, 384, 384))
    monkeypatch.setattr(faces, "_get_app", lambda cfg: app)
    monkeypatch.setattr(fairface, "get_session", lambda providers: ff_session)
    monkeypatch.setattr(mivolo, "get_session", lambda providers: mv_session)

    items = [(f"{n}.png", f"/d/{n}.png", _png(w)) for n, w in (("x", 400), ("a", 401), ("y", 402))]
    results = [ImageResult(rel_path=rel, abs_path=abs_, passed=True) for rel, abs_, _ in items]
    clusters = faces.detect_and_cluster(results, items, cfg)
    return {r.rel_path: r for r in results}, clusters, ff_session, mv_session


def test_sweep_records_face_primary_and_pooled_cluster_attributes(monkeypatch: pytest.MonkeyPatch) -> None:
    # Identity face_1 is "white" in two frames and "indian" in one; pooling the
    # probabilities must keep the majority label for the cluster.
    ff_rows = [_logits(0, 0, 3), _logits(0, 0, 3), _logits(1, 1, 5), _logits(5, 0, 3, peak=2.0)]
    mv_rows = [_mv_row(False, 27.0), _mv_row(False, 29.0), _mv_row(True, 52.0), _mv_row(False, 60.0)]
    by_rel, clusters, ff_s, mv_s = _sweep(monkeypatch, FaceConfig(fairface=True, mivolo=True), ff_rows, mv_rows)

    a = by_rel["a.png"]
    assert [f.attributes.race for f in a.faces] == ["white", "black"]
    assert [f.attributes.age_years for f in a.faces] == [29.0, 52.0]
    assert a.primary_face_attributes.race == "white"
    assert a.primary_face_attributes.gender_source == "mivolo"
    assert by_rel["y.png"].primary_face_attributes.race == "indian"

    main = next(c for c in clusters if c.size == 3)
    assert (main.attributes.race, main.attributes.gender) == ("white", "male")
    assert main.attributes.age_years == 29.0 and main.attributes.age == "20-29"
    assert main.attributes.race_conf < by_rel["x.png"].primary_face_attributes.race_conf
    assert ff_s.rows == [] and mv_s.rows == []  # every face went through each model once


def test_sweep_runs_only_the_enabled_model(monkeypatch: pytest.MonkeyPatch) -> None:
    mv_rows = [_mv_row(True, 40.0)] * 4
    by_rel, clusters, ff_s, _mv = _sweep(monkeypatch, FaceConfig(mivolo=True), mv_rows=mv_rows)
    attrs = by_rel["x.png"].primary_face_attributes
    assert attrs.race is None and attrs.age_years == 40.0 and attrs.gender == "female"
    assert ff_s.rows == []  # never consulted


def test_sweep_without_attributes_leaves_them_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    def _no_session(providers):  # pragma: no cover - must not be reached
        raise AssertionError("attribute model loaded although none was enabled")

    monkeypatch.setattr(faces, "_get_app", lambda cfg: _App({400: [_Face([100, 100, 300, 300], [1, 0, 0])]}))
    monkeypatch.setattr(fairface, "get_session", _no_session)
    monkeypatch.setattr(mivolo, "get_session", _no_session)
    items = [("x.png", "/d/x.png", _png(400))]
    results = [ImageResult(rel_path="x.png", abs_path="/d/x.png", passed=True)]
    clusters = faces.detect_and_cluster(results, items, FaceConfig(enabled=True))
    assert results[0].primary_face_attributes is None
    assert clusters[0].attributes is None


def test_attribute_models_imply_face_detection() -> None:
    assert FaceConfig(fairface=True).enabled
    assert FaceConfig(mivolo=True).enabled
    assert not FaceConfig().enabled


# ---------------------------------------------------------------------------
# Curate-side filters and the report
# ---------------------------------------------------------------------------


def _attrs(race, gender, age, conf=0.9, years=None) -> FaceAttributes:
    return FaceAttributes(
        race=race, race_conf=conf, gender=gender, gender_conf=conf, age=age, age_conf=conf, age_years=years
    )


def _img(rel: str, attrs: FaceAttributes | None) -> ImageResult:
    return ImageResult(rel_path=rel, abs_path=f"/data/{rel}", score=0.9, passed=True, primary_face_attributes=attrs)


def test_attribute_filters() -> None:
    results = [
        _img("a.jpg", _attrs("east_asian", "female", "20-29")),
        _img("b.jpg", _attrs("white", "female", "20-29")),
        _img("c.jpg", _attrs("east_asian", "male", "20-29")),
        _img("d.jpg", _attrs("east_asian", "female", "50-59")),
        _img("e.jpg", _attrs("east_asian", "female", "30-39", conf=0.4)),
        _img("f.jpg", None),
        _img("g.jpg", _attrs(None, "female", "20-29")),  # MiVOLO only: no race
    ]
    req = ExportRequest(
        dest="/out",
        min_score=0.0,
        face_races=["east_asian"],
        face_genders=["female"],
        face_ages=["20-29", "30-39"],
        min_attribute_conf=0.6,
    )
    selected, keep_reason = decide_selection(results, req, 0.0)
    assert {r.rel_path for r in selected} == {"a.jpg"}
    assert keep_reason["b.jpg"] == "face-race-excluded"
    assert keep_reason["c.jpg"] == "face-gender-excluded"
    assert keep_reason["d.jpg"] == "face-age-excluded"
    assert keep_reason["e.jpg"] == "face-race-low-confidence"
    assert keep_reason["f.jpg"] == "face-attributes-missing"
    assert keep_reason["g.jpg"] == "face-race-missing"


def test_age_years_range_and_regression_confidence() -> None:
    mv = FaceAttributes(gender="female", gender_conf=0.95, age="20-29", age_years=24.0, age_source="mivolo")
    results = [
        _img("a.jpg", mv),
        _img("b.jpg", mv.model_copy(update={"age_years": 41.0, "age": "40-49"})),
        _img("c.jpg", _attrs("white", "female", "20-29")),  # FairFace only: bracket, no years
    ]
    req = ExportRequest(dest="/out", min_score=0.0, min_age_years=18, max_age_years=35)
    selected, keep_reason = decide_selection(results, req, 0.0)
    assert {r.rel_path for r in selected} == {"a.jpg"}
    assert keep_reason["b.jpg"] == "face-age-out-of-range"
    assert keep_reason["c.jpg"] == "face-age-years-missing"

    # A regression age has no confidence, so a floor cannot reject it.
    req = ExportRequest(dest="/out", min_score=0.0, face_ages=["20-29"], min_attribute_conf=0.99)
    selected, _ = decide_selection(results[:1], req, 0.0)
    assert [r.rel_path for r in selected] == ["a.jpg"]


def test_no_attribute_filter_keeps_images_without_attributes() -> None:
    results = [_img("a.jpg", None), _img("b.jpg", _attrs("black", "male", "40-49", conf=0.2))]
    # A confidence floor alone filters nothing: it only qualifies active filters.
    req = ExportRequest(dest="/out", min_score=0.0, min_attribute_conf=0.9)
    selected, _ = decide_selection(results, req, 0.0)
    assert {r.rel_path for r in selected} == {"a.jpg", "b.jpg"}


def test_report_includes_attribute_columns(tmp_path: Path) -> None:
    mv = FaceAttributes(gender="male", gender_conf=0.88, gender_source="mivolo", age="60-69", age_years=63.2)
    results = [_img("a.jpg", _attrs("indian", "male", "60-69", conf=0.77)), _img("b.jpg", None), _img("c.jpg", mv)]
    report = tmp_path / "r.csv"
    write_report(results, {}, set(), {}, report)
    rows = {r["rel_path"]: r for r in csv.DictReader(report.open())}
    assert rows["a.jpg"]["primary_face_race"] == "indian"
    assert rows["a.jpg"]["primary_face_race_conf"] == "0.77"
    assert rows["a.jpg"]["primary_face_age"] == "60-69"
    assert rows["b.jpg"]["primary_face_gender"] == ""
    assert rows["c.jpg"]["primary_face_race"] == ""
    assert rows["c.jpg"]["primary_face_age_years"] == "63.2"
    assert rows["c.jpg"]["primary_face_gender_source"] == "mivolo"


def test_inverted_age_range_is_rejected() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="above max_age_years"):
        ExportRequest(dest="/out", min_age_years=40, max_age_years=25)
    assert ExportRequest(dest="/out", min_age_years=25, max_age_years=25).min_age_years == 25


# ---------------------------------------------------------------------------
# Set age: frames with no face take the set's age when one identity holds the scan
# ---------------------------------------------------------------------------


def _faced(rel: str, cluster: str, years: float | None, *, passed: bool = True, source: str = "mivolo") -> ImageResult:
    attrs = FaceAttributes(age_years=years, age_source=source) if years is not None else None
    face = FaceDetection(bbox=[0, 0, 1, 1], det_score=0.9, primary=True, cluster_id=cluster, attributes=attrs)
    return ImageResult(
        rel_path=rel, abs_path=f"/d/{rel}", passed=passed, faces=[face], face_count=1, primary_face_attributes=attrs
    )


def _faceless(rel: str, *, passed: bool = True) -> ImageResult:
    return ImageResult(rel_path=rel, abs_path=f"/d/{rel}", passed=passed)


def _clusters(*sizes: tuple[str, int]) -> list[FaceCluster]:
    return [FaceCluster(cluster_id=cid, size=n, representative_rel_path="x.jpg") for cid, n in sizes]


def test_set_age_is_the_dominant_identitys_median_and_lands_on_faceless_frames() -> None:
    cfg = FaceConfig(set_age=True)
    assert cfg.mivolo and cfg.enabled  # a set age is read off MiVOLO ages
    results = [
        *(
            _faced(f"f{i}.jpg", "face_1", age)
            for i, age in enumerate([18.2, 19.4, 18.9, 21.0, 17.5, 19.1, 18.4, 19.6, 20.2, 18.6])
        ),
        _faced("other.jpg", "face_2", 44.0),  # one stray face in eleven does not break the set
        _faceless("body1.jpg"),
        _faceless("body2.jpg"),
        _faceless("blurry.jpg", passed=False),  # rejected: detection never looked
    ]
    age, why = faces.infer_set_age(results, _clusters(("face_1", 10), ("face_2", 1)), cfg)
    assert why is None and age is not None
    assert age.years == 19.0 and age.cluster_id == "face_1" and age.faces == 10 and age.share == round(10 / 11, 3)
    by = {r.rel_path: r for r in results}
    assert by["body1.jpg"].set_age == age and by["body2.jpg"].set_age == age
    assert by["blurry.jpg"].set_age is None
    assert all(r.set_age is None for r in results if r.faces)


def test_set_age_is_withheld_when_the_scan_is_not_one_subject() -> None:
    cfg = FaceConfig(set_age=True)
    faceless = [_faceless("body.jpg")]

    age, why = faces.infer_set_age(faceless, [], cfg)
    assert age is None and why == "no faces were found in this scan"

    mixed = [
        *(_faced(f"a{i}.jpg", "face_1", 20.0) for i in range(6)),
        *(_faced(f"b{i}.jpg", "face_2", 30.0) for i in range(4)),
        *faceless,
    ]
    age, why = faces.infer_set_age(mixed, _clusters(("face_1", 6), ("face_2", 4)), cfg)
    assert age is None and why == "the largest identity (face_1) has 60% of the faces, 90% needed"
    assert faceless[0].set_age is None

    few = [*(_faced(f"a{i}.jpg", "face_1", 20.0) for i in range(3)), *faceless]
    age, why = faces.infer_set_age(few, _clusters(("face_1", 3)), cfg)
    assert age is None and why == "face_1 has 3 MiVOLO ages, 5 needed"

    # FairFace brackets are not years and do not count.
    bracketed = [*(_faced(f"a{i}.jpg", "face_1", 20.0, source="fairface") for i in range(6)), *faceless]
    age, why = faces.infer_set_age(bracketed, _clusters(("face_1", 6)), cfg)
    assert age is None and why == "face_1 has 0 MiVOLO ages, 5 needed"

    # The thresholds are the caller's to set.
    age, why = faces.infer_set_age(few, _clusters(("face_1", 3)), FaceConfig(set_age=True, set_age_min_faces=3))
    assert why is None and age is not None and age.faces == 3
