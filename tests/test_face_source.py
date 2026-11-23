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


def test_clusters_are_listed_in_face_n_order_when_sizes_tie(monkeypatch: pytest.MonkeyPatch) -> None:
    from argus_curator import faces as faces_mod

    # Twelve singletons, first seen in the reverse of their raw labels: ids follow the labels.
    monkeypatch.setattr(faces_mod, "_cluster_embeddings", lambda emb, eps: list(range(len(emb)))[::-1])
    results, found = _detected({f"{i}.png": [_unit(1, i, 0)] for i in range(12)})
    clusters = cluster_identities(results, found, FaceConfig(enabled=True))
    assert [c.cluster_id for c in clusters] == [f"face_{n}" for n in range(1, 13)]


def test_a_face_that_fails_partway_leaves_found_in_step(monkeypatch: pytest.MonkeyPatch) -> None:
    from argus_curator import faces as faces_mod

    class Face:
        det_score = 0.9
        normed_embedding = _unit(1, 0, 0)
        pose = None
        kps = None
        landmark_3d_68 = None

        def __init__(self, x: float) -> None:
            self.bbox = [x, 0.0, x + 10.0, 10.0]

    class App:
        def get(self, bgr: np.ndarray) -> list[Face]:
            return [Face(0.0), Face(20.0)]

    calls: list[object] = []

    def sharpness(image: object, bbox: list[float]) -> float:
        calls.append(bbox)
        if len(calls) == 2:
            raise ValueError("degenerate box")
        return 1.0

    monkeypatch.setattr(faces_mod, "face_sharpness", sharpness)
    found = DetectedFaces([], [], [])
    r = ImageResult(rel_path="a.png", abs_path="/a.png", passed=True)
    bgr = np.zeros((40, 40, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="degenerate"):
        faces_mod._detect_one(App(), (None, None), bgr, "a.png", r, FaceConfig(enabled=True), found, strict=False)
    assert len(calls) == 2
    assert found.embeddings == [] and found.meta == [] and found.raws == []


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
    # The caller's own values, not just the defaults, reach the export request.
    custom = {"max_yaw": 30.0, "min_face_px": 512, "require_adult": False, "cluster": "face_2"}
    req = SelectRequest.model_validate({"face_source": custom}).as_export_request()
    assert req.face_source == FaceSourceFilter(**custom) != FaceSourceFilter()
    assert SelectRequest.model_validate({}).as_export_request().face_source is None


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


def test_a_reference_lifts_the_largest_identity_default() -> None:
    # The reference's person is face_2, the smaller identity: the reference
    # distance names who to keep, not the cluster count.
    rows = [
        _source("other1.png", cluster_id="face_1", reference_distance=0.8),
        _source("other2.png", cluster_id="face_1", reference_distance=0.8),
        _source("ref.png", cluster_id="face_2", reference_distance=0.05),
    ]
    _chosen, reasons = _select(rows)
    assert reasons == {
        "other1.png": "face-source:identity-distance",
        "other2.png": "face-source:identity-distance",
        "ref.png": "",
    }


def test_face_source_problem_refuses_an_unknown_cluster_and_a_failed_reference() -> None:
    from argus_curator.models import FaceCluster

    summary = _summary([], FaceConfig(mivolo=True))
    summary.face_clusters = [FaceCluster(cluster_id="face_1", size=3, representative_rel_path="a.png")]
    unknown = ExportRequest(dest=".", face_source=FaceSourceFilter(cluster="face_9"))
    assert "face_9" in face_source_problem(unknown, summary)
    assert face_source_problem(ExportRequest(dest=".", face_source=FaceSourceFilter(cluster="face_1")), summary) is None

    referenced = _summary([], FaceConfig(mivolo=True, reference_paths=["gone.png"]))
    referenced.reference_skipped = "no reference image is in the scan (gone.png)"
    req = ExportRequest(dest=".", face_source=FaceSourceFilter())
    assert "reference" in face_source_problem(req, referenced)
    off = ExportRequest(dest=".", face_source=FaceSourceFilter(max_identity_distance=None))
    assert face_source_problem(off, referenced) is None


# ---------------------------------------------------------------------------
# A caller-supplied reference (#130 §3; consumed by #133)
# ---------------------------------------------------------------------------


class _RefFace:
    def __init__(self, emb: np.ndarray, side: float, det_score: float = 0.9) -> None:
        self.normed_embedding = emb
        self.bbox = [0.0, 0.0, side, side]
        self.det_score = det_score


def _png(color: int = 128) -> bytes:
    import io

    buf = io.BytesIO()
    Image.new("RGB", (32, 32), (color, color, color)).save(buf, format="PNG")
    return buf.getvalue()


def _fake_app(monkeypatch: pytest.MonkeyPatch, faces_by_color: dict[int, list[_RefFace]]) -> list[tuple]:
    """Stand InsightFace in with one keyed by each image's grey level; return the leases taken."""
    import contextlib

    from argus_curator import faces as faces_mod

    class App:
        def get(self, bgr: np.ndarray) -> list[_RefFace]:
            return faces_by_color.get(int(bgr[0, 0, 0]), [])

    leases: list[tuple] = []

    def lease(*args: object, **kwargs: object) -> contextlib.AbstractContextManager[None]:
        leases.append(args)
        return contextlib.nullcontext()

    monkeypatch.setattr(faces_mod, "_get_app", lambda cfg, lease_wait_s=None: App())
    monkeypatch.setattr(faces_mod, "gpu_lease", lease)
    return leases


def test_reference_embedding_averages_each_images_primary_face(monkeypatch: pytest.MonkeyPatch) -> None:
    from argus_curator.faces import reference_distance, reference_embedding

    leases = _fake_app(
        monkeypatch,
        {
            # The larger face is the primary; the small one is someone in the background.
            10: [_RefFace(_unit(0, 0, 1), side=20.0), _RefFace(_unit(1, 0.2, 0), side=100.0)],
            20: [_RefFace(_unit(1, -0.2, 0), side=100.0)],
            30: [],  # no face: left out, not an error while another image has one
        },
    )
    centroid = reference_embedding([_png(10), _png(20), _png(30)])
    assert np.asarray(centroid) == pytest.approx(_unit(1, 0, 0), abs=1e-6)
    assert len(leases) == 3 and all(args[1] == "faces" for args in leases)  # every image under the GPU lease
    assert reference_distance(_unit(1, 0, 0), centroid) == 0.0
    assert reference_distance(_unit(0, 1, 0), [2.0, 0.0, 0.0]) == pytest.approx(1.0)  # neither needs unit length
    with pytest.raises(ValueError, match="values"):
        reference_distance(_unit(1, 0), centroid)
    with pytest.raises(ValueError, match="no reference image has a face"):
        reference_embedding([_png(30)])


def test_a_supplied_reference_embedding_sets_every_reference_distance() -> None:
    results, found = _detected({"a.png": [_unit(1, 0, 0)], "b.png": [_unit(0, 1, 0), _unit(1, 0.1, 0)]})
    cfg = FaceConfig(reference_embedding=[3.0, 0.0, 0.0])
    assert cfg.enabled and cfg.has_reference
    cluster_identities(results, found, cfg)
    assert measure_reference(results, found, cfg) is None
    a, b = results
    assert a.faces[0].reference_distance == 0.0
    assert b.faces[0].reference_distance == pytest.approx(1.0)
    assert 0 < b.faces[1].reference_distance < 0.01


def test_a_reference_embedding_of_the_wrong_length_is_skipped_with_a_reason() -> None:
    results, found = _detected({"a.png": [_unit(1, 0, 0)]})
    reason = measure_reference(results, found, FaceConfig(reference_embedding=[1.0, 0.0]))
    assert reason is not None and "2 values" in reason and "3" in reason
    assert results[0].faces[0].reference_distance is None


def test_reference_embedding_is_validated() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="not both"):
        FaceConfig(reference_paths=["a.png"], reference_embedding=[1.0])
    for bad in ([0.0, 0.0], [float("nan"), 1.0], []):
        with pytest.raises(ValidationError):
            FaceConfig(reference_embedding=bad)


def test_a_scan_summary_never_holds_the_reference_embedding() -> None:
    cfg = FaceConfig(reference_embedding=[0.28, 0.96])
    summary = _summary([], cfg)
    assert summary.faces_config.reference_embedding is None
    assert cfg.reference_embedding == [0.28, 0.96]  # the scan's own config is left as it was
    for dumped in (summary.model_dump_json(), summary.model_dump_json(context=STORE_CONTEXT)):
        assert "0.28" not in dumped and "0.96" not in dumped
    stored = json.loads(summary.model_dump_json(context=STORE_CONTEXT))
    stored["faces_config"]["reference_embedding"] = [0.6, 0.8]  # even a cache someone edited
    assert ScanSummary.model_validate(stored).faces_config.reference_embedding is None


def test_a_rejected_reference_image_is_embedded_from_the_scans_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    leases = _fake_app(monkeypatch, {50: [_RefFace(_unit(1, 0, 0), side=100.0)]})
    results, found = _detected({"a.png": [_unit(1, 0, 0)], "b.png": [_unit(0, 1, 0)]})
    portrait = ImageResult(rel_path="portrait.png", abs_path="/portrait.png", passed=False, reject_reason="blurry")
    results.append(portrait)
    cfg = FaceConfig(enabled=True, reference_paths=["portrait.png"])
    cluster_identities(results, found, cfg)
    items = [("portrait.png", "/portrait.png", _png(50))]
    assert measure_reference(results, found, cfg) == "no reference image has a face"  # no bytes, no embedding
    assert measure_reference(results, found, cfg, items) is None
    assert results[0].faces[0].reference_distance == 0.0
    assert results[1].faces[0].reference_distance == pytest.approx(1.0)
    assert not portrait.passed and portrait.faces == []  # embedded only: the row stays rejected
    assert len(leases) == 1


def test_face_source_problem_refuses_a_failed_supplied_reference() -> None:
    # The summary cannot say a reference_embedding was given; reference_skipped does.
    summary = _summary([], FaceConfig(mivolo=True, reference_embedding=[1.0, 0.0]))
    summary.reference_skipped = "reference_embedding has 2 values; this face model's embeddings have 512"
    req = ExportRequest(dest=".", face_source=FaceSourceFilter())
    assert "reference" in face_source_problem(req, summary)


def test_cli_reads_a_reference_from_outside_the_folder(tmp_path: Path) -> None:
    import typer

    from argus_curator.cli import _outside_references

    folder = tmp_path / "set"
    folder.mkdir()
    (folder / "a.png").write_bytes(_png())
    outside = tmp_path / "consent.png"
    outside.write_bytes(_png())
    assert _outside_references(folder, ["a.png"]) is None
    assert _outside_references(folder, ["a.png", str(outside)]) == [folder / "a.png", outside]
    assert _outside_references(folder, [str(folder / "a.png")]) == [folder / "a.png"]  # an absolute path is a file
    with pytest.raises(typer.Exit):
        _outside_references(folder, [str(tmp_path / "missing.png")])


def test_cli_reads_a_dotdot_reference_as_a_file_not_a_rel_path(tmp_path: Path) -> None:
    """A ref through ".." matches no scanned rel path, so it is read like an outside reference."""
    from argus_curator.cli import _outside_references

    folder = tmp_path / "set"
    folder.mkdir()
    (folder / "a.png").write_bytes(_png())
    (tmp_path / "consent").mkdir()
    (tmp_path / "consent" / "portrait.png").write_bytes(_png())
    assert _outside_references(folder, ["../consent/portrait.png"]) == [folder / "../consent/portrait.png"]
    # Even one that comes back into the folder is not the rel path a scan records.
    assert _outside_references(folder, ["../set/a.png"]) == [folder / "../set/a.png"]
