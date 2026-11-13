"""Face-source measurements and the face-source preset (#130)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from argus_curator.export import write_manifest, write_report
from argus_curator.faces import (
    DetectedFaces,
    _extract_pose,
    cluster_identities,
    eye_mouth_openness,
    face_sharpness,
    measure_reference,
)
from argus_curator.models import (
    STORE_CONTEXT,
    ExportRequest,
    FaceAttributes,
    FaceConfig,
    FaceDetection,
    FaceLandmarks,
    FaceSourceFilter,
    ImageResult,
    ScanConfig,
    ScanSummary,
    SelectRequest,
    SetAge,
    TargetProfile,
)
from argus_curator.selection import decide_selection, face_source_problem

# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------


def _landmarks(eye_gap: float, mouth_gap: float) -> np.ndarray:
    """68 points with each eye 30 px wide and *eye_gap* tall, the inner lip 40 px wide and *mouth_gap* open."""
    pts = np.zeros((68, 3), dtype=np.float32)
    for base, x0 in ((36, 100.0), (42, 200.0)):
        pts[base + 0] = (x0, 100, 0)
        pts[base + 3] = (x0 + 30, 100, 0)
        pts[base + 1] = (x0 + 10, 100 - eye_gap / 2, 0)
        pts[base + 2] = (x0 + 20, 100 - eye_gap / 2, 0)
        pts[base + 5] = (x0 + 10, 100 + eye_gap / 2, 0)
        pts[base + 4] = (x0 + 20, 100 + eye_gap / 2, 0)
    pts[60] = (140, 200, 0)
    pts[64] = (180, 200, 0)
    for top, bottom, x in ((61, 67, 150), (62, 66, 160), (63, 65, 170)):
        pts[top] = (x, 200 - mouth_gap / 2, 0)
        pts[bottom] = (x, 200 + mouth_gap / 2, 0)
    return pts


def test_eye_and_mouth_openness() -> None:
    eye, mouth = eye_mouth_openness(_landmarks(eye_gap=9.0, mouth_gap=0.0))
    assert eye == pytest.approx(0.3)
    assert mouth == 0.0
    eye, mouth = eye_mouth_openness(_landmarks(eye_gap=1.5, mouth_gap=12.0))
    assert eye == pytest.approx(0.05)
    assert mouth == pytest.approx(0.3)


def test_openness_needs_68_points() -> None:
    assert eye_mouth_openness(np.zeros((5, 2))) == (None, None)
    assert eye_mouth_openness("nonsense") == (None, None)


def test_pose_keeps_roll() -> None:
    class Face:
        pose = [4.0, -12.5, 7.25]

    assert _extract_pose(Face()) == (-12.5, 4.0, 7.25)

    class TwoAngles:
        pose = [4.0, -12.5]

    assert _extract_pose(TwoAngles()) == (-12.5, 4.0, None)


def test_face_sharpness_reads_the_face_box_only() -> None:
    img = Image.new("L", (400, 200), 128)
    draw = ImageDraw.Draw(img)
    for x in range(200, 400, 4):  # fine stripes on the right half only
        draw.line([(x, 0), (x, 199)], fill=255)
    sharp = face_sharpness(img.convert("RGB"), [220.0, 20.0, 150.0, 150.0])
    flat = face_sharpness(img.convert("RGB"), [20.0, 20.0, 150.0, 150.0])
    assert sharp is not None and flat is not None
    assert sharp > 10 * max(flat, 1.0)
    assert face_sharpness(img.convert("RGB"), [500.0, 0.0, 10.0, 10.0]) is None


# ---------------------------------------------------------------------------
# Identity distances
# ---------------------------------------------------------------------------


def _unit(*v: float) -> np.ndarray:
    a = np.asarray(v, dtype=np.float32)
    return a / np.linalg.norm(a)


def _detected(rows: dict[str, list[np.ndarray]]) -> tuple[list[ImageResult], DetectedFaces]:
    results, found = [], DetectedFaces([], [], [])
    from argus_curator.attributes import RawAttributes

    for rel, embs in rows.items():
        r = ImageResult(rel_path=rel, abs_path=f"/{rel}", passed=True)
        for i, emb in enumerate(embs):
            bbox = [0.0, 0.0, 100.0 - i, 100.0 - i]  # the first face is the largest, so primary
            r.faces.append(FaceDetection(bbox=bbox, det_score=0.9))
            found.embeddings.append(emb)
            found.meta.append((rel, i, bbox, 0.9, 1.0, bbox[2] * bbox[3]))
            found.raws.append(RawAttributes())
        r.face_count = len(embs)
        results.append(r)
    return results, found


def test_cluster_distance_is_to_the_identity_centroid() -> None:
    results, found = _detected({"a.png": [_unit(1, 0.05, 0)], "b.png": [_unit(1, -0.05, 0)], "c.png": [_unit(0, 0, 1)]})
    cluster_identities(results, found, FaceConfig(enabled=True))
    a, b, c = (r.faces[0] for r in results)
    assert a.cluster_id == b.cluster_id != c.cluster_id
    assert a.cluster_distance == pytest.approx(b.cluster_distance, abs=1e-4)
    assert 0 < a.cluster_distance < 0.01
    assert c.cluster_distance == 0.0


def test_reference_distance_from_the_reference_images() -> None:
    results, found = _detected(
        {"ref.png": [_unit(1, 0, 0)], "near.png": [_unit(1, 0.1, 0)], "far.png": [_unit(0, 1, 0), _unit(1, 0, 0)]}
    )
    cfg = FaceConfig(enabled=True, reference_paths=["ref.png"])
    cluster_identities(results, found, cfg)
    assert measure_reference(results, found, cfg) is None
    ref, near, far = results
    assert ref.faces[0].reference_distance == 0.0
    assert 0 < near.faces[0].reference_distance < 0.01
    assert far.faces[0].reference_distance == pytest.approx(1.0)
    assert far.faces[1].reference_distance == 0.0  # every face is measured, not just the primary


def test_reference_skipped_reasons() -> None:
    results, found = _detected({"a.png": [_unit(1, 0, 0)]})
    assert measure_reference(results, found, FaceConfig(enabled=True)) is None
    reason = measure_reference(results, found, FaceConfig(reference_paths=["missing.png"]))
    assert reason is not None and "missing.png" in reason
    results.append(ImageResult(rel_path="blank.png", abs_path="/blank.png", passed=True))
    assert (
        measure_reference(results, found, FaceConfig(reference_paths=["blank.png"])) == "no reference image has a face"
    )
    assert results[0].faces[0].reference_distance is None


def test_a_reference_implies_face_detection() -> None:
    assert FaceConfig(reference_paths=["a.png"]).enabled


# ---------------------------------------------------------------------------
# The face-source preset
# ---------------------------------------------------------------------------


def _source(rel: str, *, score: float = 0.9, faces: int = 1, age: float | None = 30.0, **face: object) -> ImageResult:
    values: dict[str, object] = {
        "bbox": [0.0, 0.0, 400.0, 400.0],
        "det_score": 0.9,
        "cluster_id": "face_1",
        "primary": True,
        "yaw": 2.0,
        "pitch": -3.0,
        "roll": 1.0,
        "sharpness": 400.0,
        "eye_open": 0.3,
        "mouth_open": 0.0,
        "cluster_distance": 0.1,
        "attributes": FaceAttributes(age_years=age, age_source="mivolo") if age is not None else None,
    }
    values.update(face)
    primary = FaceDetection.model_validate(values)
    extras = [FaceDetection(bbox=[0.0, 0.0, 50.0, 50.0], det_score=0.9) for _ in range(faces - 1)]
    return ImageResult(
        rel_path=rel,
        abs_path=f"/{rel}",
        passed=True,
        score=score,
        faces=[primary, *extras],
        face_count=faces,
        primary_face_cluster=primary.cluster_id,
        primary_face_attributes=primary.attributes,
    )


def _select(results: list[ImageResult], set_age: SetAge | None = None, **fs: object):
    req = ExportRequest(dest=".", min_score=0.0, face_source=FaceSourceFilter(**fs))
    return decide_selection(results, req, 0.0, set_age=set_age)


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"faces": 2}, "face-source:faces"),
        ({"cluster_id": "face_2"}, "face-source:identity"),
        ({"yaw": -20.0}, "face-source:yaw"),
        ({"pitch": 16.0}, "face-source:pitch"),
        ({"roll": -11.0}, "face-source:roll"),
        ({"roll": None}, "face-source:roll"),
        ({"bbox": [0.0, 0.0, 400.0, 200.0]}, "face-source:face-size"),
        ({"sharpness": 50.0}, "face-source:face-sharpness"),
        ({"eye_open": 0.1}, "face-source:eyes"),
        ({"mouth_open": 0.3}, "face-source:mouth"),
        ({"cluster_distance": 0.6}, "face-source:identity-distance"),
        ({"age": None}, "face-source:no-age"),
        ({"age": 16.0}, "face-source:age"),
    ],
)
def test_each_check_keeps_an_image_out(kwargs: dict, reason: str) -> None:
    good = [_source(f"good{i}.png") for i in range(3)]
    bad = _source("bad.png", **kwargs)
    chosen, reasons = _select([*good, bad])
    assert reasons["bad.png"] == reason
    assert [r.rel_path for r in chosen] == ["good0.png", "good1.png", "good2.png"]


def test_a_check_set_to_none_is_off() -> None:
    _chosen, reasons = _select([_source("a.png", roll=40.0, mouth_open=0.5)], max_roll=None, max_mouth_open=None)
    assert reasons["a.png"] == ""


def test_the_reference_distance_wins_over_the_cluster_distance() -> None:
    near = _source("near.png", cluster_distance=0.9, reference_distance=0.1)
    far = _source("far.png", cluster_distance=0.0, reference_distance=0.9)
    _chosen, reasons = _select([near, far])
    assert reasons == {"near.png": "", "far.png": "face-source:identity-distance"}


def test_a_young_set_age_keeps_everything_out() -> None:
    young = SetAge(years=15.0, cluster_id="face_1", faces=5, share=1.0)
    chosen, reasons = _select([_source("a.png")], set_age=young)
    assert not chosen and reasons["a.png"] == "face-source:age"


def test_the_largest_identity_is_the_default() -> None:
    rows = [_source("a.png", cluster_id="face_2"), _source("b.png", cluster_id="face_2"), _source("c.png")]
    _chosen, reasons = _select(rows)
    assert reasons == {"a.png": "", "b.png": "", "c.png": "face-source:identity"}
    _chosen, reasons = _select(rows, cluster="face_1")
    assert reasons["c.png"] == ""


def test_kept_images_are_ranked_best_first() -> None:
    rows = [
        _source("low.png", score=0.7),
        _source("tie_far.png", score=0.9, cluster_distance=0.3),
        _source("tie_near.png", score=0.9, cluster_distance=0.05),
    ]
    chosen, _ = _select(rows)
    assert [r.rel_path for r in chosen] == ["tie_near.png", "tie_far.png", "low.png"]


def _summary(results: list[ImageResult], faces: FaceConfig) -> ScanSummary:
    return ScanSummary(
        scan_id="s",
        folder="/x",
        target_profile=TargetProfile(),
        config=ScanConfig(),
        faces_config=faces,
        total=len(results),
        passed=len(results),
        rejected=0,
        duplicates=0,
        similar_clusters=0,
        results=results,
    )


def test_face_source_problem_names_the_missing_scan_step() -> None:
    req = ExportRequest(dest=".", face_source=FaceSourceFilter())
    assert "face detection" in face_source_problem(req, _summary([], FaceConfig()))
    assert "MiVOLO" in face_source_problem(req, _summary([], FaceConfig(enabled=True)))
    assert face_source_problem(req, _summary([], FaceConfig(mivolo=True))) is None
    lax = ExportRequest(dest=".", face_source=FaceSourceFilter(require_adult=False))
    assert face_source_problem(lax, _summary([], FaceConfig(enabled=True))) is None
    assert face_source_problem(ExportRequest(dest="."), _summary([], FaceConfig())) is None


def test_select_request_carries_the_preset() -> None:
    req = SelectRequest.model_validate({"face_source": {}}).as_export_request()
    assert req.face_source == FaceSourceFilter()


# ---------------------------------------------------------------------------
# Manifest, report and store
# ---------------------------------------------------------------------------


def _landmarked(rel: str) -> ImageResult:
    r = _source(rel)
    r.faces[0].landmarks = FaceLandmarks(kps=[[1.0, 2.0]] * 5, landmark_3d_68=[[1.0, 2.0, 3.0]] * 68)
    r.primary_face_pitch, r.primary_face_roll = r.faces[0].pitch, r.faces[0].roll
    return r


def test_manifest_rows_carry_the_primary_face(tmp_path: Path) -> None:
    r = _landmarked("a.png")
    summary = _summary([r], FaceConfig(mivolo=True))
    summary.set_age = SetAge(years=30.0, cluster_id="face_1", faces=5, share=1.0)
    path = tmp_path / "manifest.jsonl"
    write_manifest([r], summary, path, {"a.png": "a.png"})
    row = json.loads(path.read_text())
    face = row["primary_face"]
    assert face["bbox"] == [0.0, 0.0, 400.0, 400.0]
    assert (face["yaw"], face["pitch"], face["roll"]) == (2.0, -3.0, 1.0)
    assert face["sharpness"] == 400.0 and face["eye_open"] == 0.3 and face["age_years"] == 30.0
    assert face["landmarks"] is None  # opt-in
    assert row["set_age"]["years"] == 30.0

    write_manifest([r], summary, path, {"a.png": "a.png"}, landmarks=True)
    face = json.loads(path.read_text())["primary_face"]
    assert len(face["landmarks"]["kps"]) == 5 and len(face["landmarks"]["landmark_3d_68"]) == 68


def test_a_row_without_a_face_has_no_primary_face(tmp_path: Path) -> None:
    r = ImageResult(rel_path="a.png", abs_path="/a.png", passed=True)
    path = tmp_path / "manifest.jsonl"
    write_manifest([r], _summary([r], FaceConfig()), path, {"a.png": "a.png"})
    assert json.loads(path.read_text())["primary_face"] is None


def test_report_has_pitch_roll_and_face_quality(tmp_path: Path) -> None:
    import csv

    rows = [_landmarked("a.png"), ImageResult(rel_path="b.png", abs_path="/b.png")]
    path = tmp_path / "report.csv"
    write_report(rows, {}, set(), {}, path)
    by_rel = {row["rel_path"]: row for row in csv.DictReader(path.open())}
    a = by_rel["a.png"]
    assert (a["primary_face_pitch"], a["primary_face_roll"], a["primary_face_px"]) == ("-3.0", "1.0", "400")
    assert a["primary_face_eye_open"] == "0.3" and a["primary_face_cluster_distance"] == "0.1"
    assert by_rel["b.png"]["primary_face_px"] == "" and by_rel["b.png"]["primary_face_roll"] == ""


def test_landmarks_live_in_the_scan_cache_only() -> None:
    face = _landmarked("a.png").faces[0]
    assert "landmarks" not in face.model_dump()
    assert "landmarks" not in json.loads(face.model_dump_json())
    stored = face.model_dump(context=STORE_CONTEXT)
    assert len(stored["landmarks"]["landmark_3d_68"]) == 68
    assert FaceDetection.model_validate(stored).landmarks == face.landmarks
