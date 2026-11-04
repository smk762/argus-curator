"""An identity's pooled attributes and the set age follow the images a delete removes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from argus_curator.attributes import RawAttributes, from_readings, pool, to_attributes, to_readings  # noqa: E402
from argus_curator.models import (  # noqa: E402
    FaceCluster,
    FaceConfig,
    FaceDetection,
    FaceReadings,
    ImageResult,
    ScanConfig,
    ScanSummary,
    SetAge,
    TargetProfile,
    wire_schema,
)
from argus_curator.scanner import drop_results  # noqa: E402
from argus_curator.server import create_app  # noqa: E402
from argus_curator.store import ScanStore  # noqa: E402

_MALE, _FEMALE = np.array([0.9, 0.1]), np.array([0.2, 0.8])


def _raw(age: float, gender: np.ndarray = _MALE) -> RawAttributes:
    return RawAttributes(mivolo_gender=gender, mivolo_age=age)


def _row(rel: str, cluster: str | None = "face_1", raw: RawAttributes | None = None, **extra: object) -> ImageResult:
    faces = []
    if cluster is not None:
        raw = raw or _raw(30.0)
        faces = [
            FaceDetection(
                bbox=[0, 0, 10, 10],
                det_score=0.9,
                cluster_id=cluster,
                attributes=to_attributes(raw),
                readings=to_readings(raw),
            )
        ]
    return ImageResult(rel_path=rel, abs_path=f"/d/{rel}", passed=True, faces=faces, face_count=len(faces), **extra)


def _summary(rows: list[ImageResult], clusters: dict[str, int], set_age: SetAge | None = None) -> ScanSummary:
    raws = {cid: [] for cid in clusters}
    for r in rows:
        for f in r.faces:
            raws[f.cluster_id].append(from_readings(f.readings) if f.readings else _raw(30.0))
    return ScanSummary(
        scan_id="s",
        folder="/d",
        target_profile=TargetProfile(),
        config=ScanConfig(cluster_distance=-1),
        faces_config=FaceConfig(set_age=True, mivolo=True, set_age_min_faces=2, set_age_min_share=0.8),
        total=len(rows),
        passed=len(rows),
        rejected=0,
        duplicates=0,
        similar_clusters=0,
        face_clusters=[
            FaceCluster(cluster_id=cid, size=n, representative_rel_path=rows[0].rel_path, attributes=pool(raws[cid]))
            for cid, n in clusters.items()
        ],
        set_age=set_age,
        results=rows,
        returned=len(rows),
    )


def test_readings_round_trip_to_the_same_labels() -> None:
    raw = RawAttributes(
        fairface=np.linspace(0.01, 0.2, 18, dtype=np.float32),
        mivolo_gender=np.array([0.3, 0.7], dtype=np.float32),
        mivolo_age=31.4,
    )
    saved = FaceReadings.model_validate_json(to_readings(raw).model_dump_json())
    assert pool([from_readings(saved)]) == pool([raw])


def test_a_nan_reading_does_not_stop_the_scan_loading(tmp_path: Path) -> None:
    raw = RawAttributes(fairface=np.full(18, np.nan), mivolo_gender=np.array([np.nan, np.nan]), mivolo_age=float("nan"))
    store = ScanStore(tmp_path)
    store.save(_summary([_row("a", raw=raw), _row("b")], {"face_1": 2}))

    assert store.load("s").results[0].faces[0].readings == FaceReadings()


@pytest.mark.parametrize("field,values", [("fairface", [0.5] * 17), ("mivolo_gender", [1.0])])
def test_a_reading_of_the_wrong_length_is_refused_on_load(field: str, values: list[float]) -> None:
    with pytest.raises(ValidationError):
        FaceReadings.model_validate({field: values})


def test_deleting_faces_pools_the_identity_again() -> None:
    rows = [_row("a", raw=_raw(20.0, _FEMALE)), _row("b", raw=_raw(20.0, _FEMALE)), _row("c", raw=_raw(60.0))]
    summary = _summary(rows, {"face_1": 3})
    assert summary.face_clusters[0].attributes.gender == "female"

    drop_results(summary, ["a", "b"])

    (cluster,) = summary.face_clusters
    assert cluster.attributes.gender == "male" and cluster.attributes.age_years == 60.0


def test_a_scan_cached_without_readings_keeps_its_pooled_attributes() -> None:
    rows = [_row("a", raw=_raw(20.0)), _row("b", raw=_raw(60.0))]
    summary = _summary(rows, {"face_1": 2})
    before = summary.face_clusters[0].attributes
    for r in rows:
        r.faces[0].readings = None

    drop_results(summary, ["a"])

    assert summary.face_clusters[0].attributes == before


def test_an_identity_with_one_face_missing_readings_keeps_its_pooled_attributes() -> None:
    rows = [_row("a", raw=_raw(20.0)), _row("b", raw=_raw(40.0)), _row("c", raw=_raw(60.0))]
    summary = _summary(rows, {"face_1": 3})
    before = summary.face_clusters[0].attributes
    rows[1].faces[0].readings = None

    drop_results(summary, ["a"])

    assert summary.face_clusters[0].attributes == before


def test_a_prefiltered_face_still_counts_toward_the_pool() -> None:
    dropped = _row("c", raw=_raw(50.0))
    dropped.passed, dropped.reject_reason = False, "prefilter:nsfw"
    summary = _summary([_row("a", raw=_raw(20.0)), _row("b", raw=_raw(30.0)), dropped], {"face_1": 2})

    drop_results(summary, ["a"])

    assert summary.face_clusters[0].attributes.age_years == 40.0  # the median of 30 and 50


def _set_rows() -> tuple[list[ImageResult], SetAge]:
    age = SetAge(years=30.0, cluster_id="face_1", faces=4, share=1.0)
    rows = [_row(n, raw=_raw(y)) for n, y in (("a", 20.0), ("b", 20.0), ("c", 40.0), ("d", 40.0))]
    return [*rows, _row("faceless", cluster=None, set_age=age)], age


def test_deleting_faces_reads_the_set_age_again() -> None:
    rows, age = _set_rows()
    summary = _summary(rows, {"face_1": 4}, age)

    drop_results(summary, ["a", "b"])

    assert summary.set_age.years == 40.0 and summary.set_age.faces == 2 and summary.set_age_skipped is None
    assert summary.results[-1].set_age == summary.set_age


def test_a_set_age_too_few_faces_support_is_taken_back() -> None:
    rows, age = _set_rows()
    summary = _summary(rows, {"face_1": 4}, age)

    drop_results(summary, ["a", "b", "c"])

    assert summary.set_age is None and "MiVOLO ages" in summary.set_age_skipped
    assert summary.results[-1].set_age is None


def test_a_dropped_faceless_frame_follows_the_set_age() -> None:
    rows, age = _set_rows()
    rows[-1].passed, rows[-1].reject_reason = False, "prefilter:nsfw"
    summary = _summary(rows, {"face_1": 4}, age)

    drop_results(summary, ["a", "b"])

    assert summary.results[-1].set_age.years == 40.0


def test_deleting_the_other_identity_can_give_a_set_age() -> None:
    rows = [_row("a"), _row("b"), _row("c", cluster="face_2"), _row("d", cluster="face_2")]
    faceless = _row("faceless", cluster=None)
    summary = _summary([*rows, faceless], {"face_1": 2, "face_2": 2})
    summary.set_age_skipped = "the largest identity (face_1) has 50% of the faces, 80% needed"

    drop_results(summary, ["c", "d"])

    assert summary.set_age.cluster_id == "face_1" and summary.set_age_skipped is None
    assert summary.results[-1].set_age == summary.set_age


def test_a_tie_for_largest_identity_keeps_the_lowest_id_whatever_the_row_order() -> None:
    rows = [_row("a", cluster="face_2"), _row("b", cluster="face_2"), _row("c"), _row("d")]
    summary = _summary([*rows, _row("gone", cluster=None)], {"face_1": 2, "face_2": 2})
    summary.faces_config = FaceConfig(set_age=True, set_age_min_faces=2, set_age_min_share=0.5)

    drop_results(summary, ["gone"])

    assert summary.set_age.cluster_id == "face_1"


def test_a_scan_that_did_not_ask_for_a_set_age_gets_none() -> None:
    rows, _age = _set_rows()
    rows[-1].set_age = None
    summary = _summary(rows, {"face_1": 4})
    summary.faces_config = FaceConfig()

    drop_results(summary, ["a"])

    assert summary.set_age is None and summary.set_age_skipped is None


def test_readings_stay_out_of_every_api_dump() -> None:
    summary = _summary([_row("a")], {"face_1": 1})
    row = summary.results[0]
    assert row.faces[0].readings is not None
    assert "readings" not in summary.model_dump_json()
    assert "readings" not in str(summary.model_dump(mode="json"))  # the SSE complete event
    assert "readings" not in str(row.model_dump(mode="json"))  # scan chunks
    assert "readings" not in json.dumps(wire_schema())


def test_get_scan_does_not_serve_readings(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    ScanStore(cache).save(_summary([_row("a")], {"face_1": 1}))
    client = TestClient(create_app(cache_dir=str(cache)))

    resp = client.get("/scan/s")

    assert resp.status_code == 200 and resp.json()["results"][0]["faces"]
    assert "readings" not in resp.text and "readings" in (cache / "s.json").read_text()


def test_a_saved_scan_can_still_be_pooled_again(tmp_path: Path) -> None:
    store = ScanStore(tmp_path)
    store.save(_summary([_row("a", raw=_raw(20.0)), _row("b", raw=_raw(60.0))], {"face_1": 2}))

    summary = store.load("s")
    drop_results(summary, ["a"])
    store.save(summary)

    saved = store.load("s")
    assert saved.face_clusters[0].attributes.age_years == 60.0
    assert saved.results[0].faces[0].readings.mivolo_age == 60.0


def test_readings_leave_with_the_scan(tmp_path: Path) -> None:
    """They are in the scan's own file: purging a run leaves nothing to go stale."""
    cache = tmp_path / "cache"
    store = ScanStore(cache)
    store.save(_summary([_row("a"), _row("b")], {"face_1": 2}))
    assert store.load("s").results[0].faces[0].readings is not None
    assert [p.name for p in cache.iterdir()] == ["s.json"]

    client = TestClient(create_app(cache_dir=str(cache)))
    assert client.post("/scans/purge").status_code == 200
    assert list(cache.iterdir()) == []


def test_a_deleted_images_readings_leave_the_saved_scan(tmp_path: Path) -> None:
    store = ScanStore(tmp_path)
    summary = _summary([_row("a", raw=_raw(21.5)), _row("b", raw=_raw(60.0))], {"face_1": 2})

    drop_results(summary, ["a"])
    store.save(summary)

    saved = (tmp_path / "s.json").read_text()
    assert "21.5" not in saved and '"mivolo_age":60.0' in saved
