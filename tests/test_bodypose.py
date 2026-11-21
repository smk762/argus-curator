"""The body-pose stage on argus-puppet: parsing its wire, the scan stage, clustering, caps, failures, the manifest.

No live service: a fake client answers with tiny hand-built puppet payloads, and
a small HTTP server stands in for puppet where the real urllib path matters
(the 503 retry, the query and headers sent).
"""

from __future__ import annotations

import csv
import io
import json
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest
from PIL import Image, ImageOps

from argus_curator import bodypose, remote
from argus_curator.export import write_manifest, write_report
from argus_curator.models import (
    STORE_CONTEXT,
    BodyPoseLabels,
    FaceConfig,
    FaceDetection,
    ImageResult,
    ManifestRow,
    PoseConfig,
    ScanConfig,
    ScanRequest,
    TargetProfile,
)
from argus_curator.runtime import ModelUnavailable
from argus_curator.scanner import drop_results, scan_items
from argus_curator.store import ScanStore

PUPPET = "http://puppet.internal:8220"
# A coordinate no other part of a scan would hold: if it shows up in a dump, a keypoint leaked.
MARK = 123.456


def _png(red: int, size: tuple[int, int] = (600, 600)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (red, 0, 0)).save(buf, "PNG")
    return buf.getvalue()


def _person(*, primary: bool, labels: dict[str, str] | None = None) -> dict[str, Any]:
    person: dict[str, Any] = {
        "bbox": [10.0, 10.0, 100.0, 300.0],
        "det_score": 0.9,
        "keypoints": [[MARK, 7.0, 0.9]] * 133,
        "primary": primary,
        "part_scores": {"body": 0.9, "feet": 0.5, "face": 0.9, "left_hand": 0.5, "right_hand": 0.5},
    }
    if labels is not None:
        person["labels"] = labels
    return person


def _pose(people: list[dict[str, Any]], primary: int | None, *, version: str = "1.1", **extra: Any) -> dict[str, Any]:
    return {
        "puppet_version": version,
        "image": "upload",
        "image_sha256": "0" * 64,
        "width": 600,
        "height": 600,
        "layout": "coco_wholebody_133",
        "people": people,
        "primary": primary,
        "estimator": {"name": "topdown"},
        **extra,
    }


STANDING = {"stance": "standing", "arms": "down", "facing": "front"}
SEATED = {"stance": "seated", "arms": "out", "facing": "left"}


class FakePuppet:
    """Answers ``/pose`` by the image's red channel and ``/stats`` with a canned PoseSetStats."""

    def __init__(self, poses: dict[int, dict[str, Any] | Exception], stats: Any = None) -> None:
        self.poses = poses
        self.stats_answer = stats
        self.faces: dict[str, str | None] = {}
        self.stats_bodies: list[dict[str, Any]] = []
        self.calls = 0

    def pose(self, data: bytes, rel_path: str, face: str | None = None) -> bodypose.Posed:
        self.calls += 1
        self.faces[rel_path] = face
        red = Image.open(io.BytesIO(data)).getpixel((0, 0))[0]
        answer = self.poses[red]
        if isinstance(answer, Exception):
            raise answer
        return bodypose.parse_pose(answer)

    def stats(self, body: bytes) -> Any:
        self.stats_bodies.append(json.loads(body))
        if isinstance(self.stats_answer, Exception):
            raise self.stats_answer
        return self.stats_answer


def _stats(items: list[dict[str, Any]], clusters: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    return {
        "puppet_version": "1.1",
        "count": len(items),
        "usable": sum(1 for i in items if i.get("usable")),
        "comparable_pairs": 1,
        "comparable_share": 1.0,
        "diversity": 0.42,
        "mean_pairwise_distance": 0.3,
        "cluster_count": len(clusters),
        "largest_cluster_share": max((c["share"] for c in clusters), default=0.0),
        "clusters": clusters,
        "items": items,
        **extra,
    }


@pytest.fixture
def puppet(monkeypatch: pytest.MonkeyPatch) -> FakePuppet:
    fake = FakePuppet({})
    monkeypatch.setenv(bodypose.URL_ENV, PUPPET)
    monkeypatch.setattr(bodypose, "client", lambda: fake)
    return fake


def _scan(items: list[tuple[str, str, bytes]], **kw: Any):
    return scan_items(items, TargetProfile(), ScanConfig(blur_threshold=0.0), FaceConfig(), **kw)


THREE = [("a.png", "/d/a.png", _png(10)), ("b.png", "/d/b.png", _png(20)), ("c.png", "/d/c.png", _png(30))]


# --- the wire ---------------------------------------------------------------


def test_a_1_1_answer_gives_the_primary_persons_labels_and_keeps_only_that_person() -> None:
    other = _person(primary=False, labels=SEATED)
    posed = bodypose.parse_pose(_pose([other, _person(primary=True, labels=STANDING)], 1, extra_field="ignored"))
    assert posed.has_person and posed.version == "1.1"
    assert posed.labels == BodyPoseLabels(stance="standing", arms="down", facing="front")
    assert posed.result["primary"] == 0 and len(posed.result["people"]) == 1
    assert posed.result["people"][0]["labels"] == STANDING
    assert posed.result["extra_field"] == "ignored"  # unknown fields travel on to /stats untouched


def test_a_1_0_answer_has_no_labels_and_a_label_this_build_does_not_know_reads_unknown() -> None:
    assert bodypose.parse_pose(_pose([_person(primary=True)], 0, version="1.0")).labels is None
    odd = bodypose.parse_pose(_pose([_person(primary=True, labels={"stance": "floating", "arms": "up"})], 0))
    assert odd.labels == BodyPoseLabels(stance="unknown", arms="up", facing="unknown")


def test_no_person_another_major_or_a_bad_primary() -> None:
    nobody = bodypose.parse_pose(_pose([], None))
    assert not nobody.has_person and nobody.labels is None and nobody.result["people"] == []
    with pytest.raises(bodypose.PoseCheckFailed, match="puppet_version 2.0 is not supported"):
        bodypose.parse_pose(_pose([], None, version="2.0"))
    with pytest.raises(bodypose.PoseCheckFailed, match="primary"):
        bodypose.parse_pose(_pose([_person(primary=True)], 3))
    with pytest.raises(bodypose.PoseCheckFailed):
        bodypose.parse_pose(["not", "a", "PoseResult"])


@pytest.mark.parametrize("orientation", range(1, 9))
def test_the_face_hint_turns_with_the_exif_orientation(orientation: int) -> None:
    # A stored 120x80 frame with a white box; Pillow's exif_transpose is the frame puppet reads the hint in.
    stored = Image.new("L", (120, 80), 0)
    box = [10, 20, 30, 15]
    stored.paste(255, (box[0], box[1], box[0] + box[2], box[1] + box[3]))
    exif = Image.Exif()
    exif[0x0112] = orientation
    buf = io.BytesIO()
    stored.save(buf, "PNG", exif=exif.tobytes())
    upright = ImageOps.exif_transpose(Image.open(io.BytesIO(buf.getvalue())))
    left, top, right, bottom = upright.getbbox()
    assert bodypose.upright_box([float(v) for v in box], orientation, 120, 80) == [
        left,
        top,
        right - left,
        bottom - top,
    ]

    row = ImageResult(rel_path="x.png", abs_path="/d/x.png", passed=True)
    row.faces = [FaceDetection(bbox=[float(v) for v in box], det_score=0.9, primary=True)]
    want = ",".join(f"{v:.1f}" for v in (left, top, right - left, bottom - top))
    assert bodypose.face_hint(row, buf.getvalue()) == want


def test_no_face_no_hint() -> None:
    assert bodypose.face_hint(ImageResult(rel_path="x.png", abs_path="/d/x.png"), _png(1)) is None


# --- the stage --------------------------------------------------------------


def test_the_scan_records_labels_and_scan_level_clusters(puppet: FakePuppet) -> None:
    puppet.poses = {
        10: _pose([_person(primary=True, labels=STANDING)], 0),
        20: _pose([_person(primary=True, labels=STANDING)], 0),
        30: _pose([_person(primary=True, labels=SEATED)], 0),
    }
    puppet.stats_answer = _stats(
        [
            {"image": "a.png", "usable": True, "cluster": 0, "labels": STANDING},
            {"image": "b.png", "usable": True, "cluster": 0, "labels": STANDING},
            {"image": "c.png", "usable": True, "cluster": 1, "labels": SEATED},
        ],
        [
            {"id": 0, "size": 2, "share": 0.6667, "medoid": "a.png"},
            {"id": 1, "size": 1, "share": 0.3333, "medoid": "c.png"},
        ],
    )
    summary = _scan(THREE)
    rows = {r.rel_path: r for r in summary.results}
    assert summary.pose_config.enabled is True
    assert {k: (r.pose_check, r.pose_cluster) for k, r in rows.items()} == {
        "a.png": ("ok", 0),
        "b.png": ("ok", 0),
        "c.png": ("ok", 1),
    }
    assert rows["c.png"].pose_labels == BodyPoseLabels(**SEATED)
    pose = summary.pose
    assert pose is not None and pose.skipped is None
    assert (pose.posed, pose.no_person, pose.failed, pose.usable) == (3, 0, 0, 3)
    assert pose.diversity == 0.42 and pose.largest_cluster_share == 0.6667
    assert [(c.id, c.size, c.medoid_rel_path) for c in pose.clusters] == [(0, 2, "a.png"), (1, 1, "c.png")]
    # One /stats call over the whole scan, each image named by its rel_path, at most one person each.
    (body,) = puppet.stats_bodies
    assert [r["image"] for r in body["results"]] == ["a.png", "b.png", "c.png"]
    assert all(len(r["people"]) == 1 and r["primary"] == 0 for r in body["results"])


def test_a_1_0_server_gets_its_labels_from_the_stats_items(puppet: FakePuppet) -> None:
    puppet.poses = {10: _pose([_person(primary=True)], 0, version="1.0")}
    puppet.stats_answer = _stats(
        [{"image": "a.png", "usable": True, "cluster": 0, "labels": SEATED}],
        [{"id": 0, "size": 1, "share": 1.0, "medoid": "a.png"}],
        puppet_version="1.0",
    )
    (row,) = _scan(THREE[:1]).results
    assert row.pose_labels == BodyPoseLabels(**SEATED) and row.pose_cluster == 0


def test_a_1_0_server_whose_clustering_is_skipped_leaves_labels_none(
    puppet: FakePuppet, monkeypatch: pytest.MonkeyPatch
) -> None:
    puppet.poses = {10: _pose([_person(primary=True)], 0, version="1.0")}
    puppet.stats_answer = bodypose.PoseCheckFailed("argus-puppet returned HTTP 500")
    summary = _scan(THREE[:1])
    (row,) = summary.results
    assert row.pose_check == "ok" and row.pose_labels is None and row.pose_cluster is None
    assert summary.pose.skipped.startswith("the /stats call failed")


def test_no_person_several_people_and_an_unusable_pose(puppet: FakePuppet) -> None:
    puppet.poses = {
        10: _pose([], None),
        20: _pose([_person(primary=False, labels=SEATED), _person(primary=True, labels=STANDING)], 1),
        30: _pose([_person(primary=True, labels={"stance": "unknown", "arms": "up", "facing": "unknown"})], 0),
    }
    puppet.stats_answer = _stats(
        [
            {"image": "a.png", "usable": False, "cluster": None},
            {"image": "b.png", "usable": True, "cluster": 0, "labels": STANDING},
            {"image": "c.png", "usable": False, "cluster": None, "labels": {"arms": "up"}},
        ],
        [{"id": 0, "size": 1, "share": 1.0, "medoid": "b.png"}],
    )
    summary = _scan(THREE)
    rows = {r.rel_path: r for r in summary.results}
    assert (rows["a.png"].pose_check, rows["a.png"].pose_labels, rows["a.png"].pose_cluster) == (
        "no_person",
        None,
        None,
    )
    assert rows["b.png"].pose_labels == BodyPoseLabels(**STANDING) and rows["b.png"].pose_cluster == 0
    # Unusable: the labels the rules could decide stay, the cluster is None.
    assert rows["c.png"].pose_check == "unusable" and rows["c.png"].pose_cluster is None
    assert rows["c.png"].pose_labels == BodyPoseLabels(stance="unknown", arms="up", facing="unknown")
    assert (summary.pose.posed, summary.pose.no_person) == (3, 1)
    assert summary.pose.puppet_version == "1.1"


def test_the_primary_face_goes_as_the_hint(puppet: FakePuppet) -> None:
    puppet.poses = {10: _pose([], None)}
    puppet.stats_answer = _stats([], [])
    row = ImageResult(rel_path="a.png", abs_path="/d/a.png", passed=True)
    row.faces = [
        FaceDetection(bbox=[1.0, 2.0, 3.0, 4.0], det_score=0.5),
        FaceDetection(bbox=[50.0, 60.0, 70.0, 80.0], det_score=0.9, primary=True),
    ]
    bodypose.pose_images([row], [("a.png", "/d/a.png", _png(10))])
    assert puppet.faces == {"a.png": "50.0,60.0,70.0,80.0"}


def test_rejected_images_are_not_sent(puppet: FakePuppet) -> None:
    summary = _scan([("t.png", "/d/t.png", _png(10, (100, 100)))])  # under min_short_side
    assert puppet.calls == 0 and summary.results[0].pose_check is None
    assert summary.pose.skipped == "no image was posed"


def test_a_scan_opted_out_or_without_a_url_records_the_stage_off(puppet: FakePuppet) -> None:
    off = _scan(THREE[:1], pose_cfg=PoseConfig(enabled=False))
    assert off.pose_config.enabled is False and off.pose is None and puppet.calls == 0
    assert off.results[0].pose_check is None


def test_required_without_a_url_and_a_malformed_url_are_refused_without_echoing_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ModelUnavailable, match=bodypose.URL_ENV):
        _scan(THREE[:1], pose_cfg=PoseConfig(enabled=True))
    monkeypatch.setenv(bodypose.URL_ENV, "ftp://user:secret@puppet")
    with pytest.raises(ModelUnavailable) as info:
        _scan(THREE[:1])
    assert bodypose.URL_ENV in str(info.value) and "secret" not in str(info.value)
    assert "secret" not in repr(info.value.__cause__)


# --- caps -------------------------------------------------------------------


def test_a_scan_over_the_set_cap_is_not_clustered_and_says_why(
    puppet: FakePuppet, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(bodypose.MAX_SET_ENV, "2")
    puppet.poses = {r: _pose([_person(primary=True, labels=STANDING)], 0) for r in (10, 20, 30)}
    summary = _scan(THREE)
    assert puppet.stats_bodies == []  # never truncated to fit: not sent at all
    assert "over the /stats set cap of 2" in summary.pose.skipped
    assert all(r.pose_labels == BodyPoseLabels(**STANDING) and r.pose_cluster is None for r in summary.results)
    assert all(r.pose_check == "ok" for r in summary.results)


def test_a_body_over_the_byte_cap_is_not_sent(puppet: FakePuppet, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(bodypose.MAX_JSON_ENV, "4096")
    puppet.poses = {r: _pose([_person(primary=True, labels=STANDING)], 0) for r in (10, 20, 30)}
    summary = _scan(THREE)
    assert puppet.stats_bodies == [] and "over the cap of 4096" in summary.pose.skipped


def test_a_set_puppet_refuses_as_too_large_is_recorded(puppet: FakePuppet) -> None:
    import urllib.error

    puppet.poses = {10: _pose([_person(primary=True, labels=STANDING)], 0)}
    refused = bodypose.PoseCheckFailed("argus-puppet returned HTTP 413")
    refused.__cause__ = urllib.error.HTTPError(PUPPET, 413, "too large", {}, None)  # type: ignore[arg-type]
    puppet.stats_answer = refused
    summary = _scan(THREE[:1])
    assert summary.pose.skipped == "argus-puppet refused the set as too large (HTTP 413)"
    assert summary.results[0].pose_labels == BodyPoseLabels(**STANDING)


# --- failures ---------------------------------------------------------------


def test_a_failed_image_keeps_none_and_the_rest_are_posed(puppet: FakePuppet) -> None:
    puppet.poses = {
        10: bodypose.PoseCheckFailed("argus-puppet returned HTTP 415"),
        20: _pose([_person(primary=True, labels=STANDING)], 0),
        30: ValueError("anything at all"),
    }
    puppet.stats_answer = _stats(
        [{"image": "b.png", "usable": True, "cluster": 0, "labels": STANDING}],
        [{"id": 0, "size": 1, "share": 1.0, "medoid": "b.png"}],
    )
    summary = _scan(THREE)
    rows = {r.rel_path: r for r in summary.results}
    assert {k: r.pose_check for k, r in rows.items()} == {"a.png": "failed", "b.png": "ok", "c.png": "failed"}
    assert rows["a.png"].pose_labels is None and rows["a.png"].pose_cluster is None
    assert (summary.pose.posed, summary.pose.failed) == (1, 2)
    assert [r["image"] for r in puppet.stats_bodies[0]["results"]] == ["b.png"]


def test_a_down_server_is_given_up_on_and_the_scan_finishes(
    puppet: FakePuppet, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(bodypose.WORKERS_ENV, "1")

    class Down(FakePuppet):
        def pose(self, data: bytes, rel_path: str, face: str | None = None) -> bodypose.Posed:
            self.calls += 1
            time.sleep(0.02)
            raise bodypose.PoseCheckFailed("argus-puppet request failed: connection refused")

    down = Down({})
    monkeypatch.setattr(bodypose, "client", lambda: down)
    items = [(f"{i}.png", f"/d/{i}.png", _png(10)) for i in range(30)]
    summary = _scan(items)
    assert summary.passed == 30
    assert down.calls <= remote.MAX_FAILURES_WITHOUT_SUCCESS + 2
    assert all(r.pose_check == "failed" and r.pose_labels is None and r.pose_cluster is None for r in summary.results)
    assert summary.pose.posed == 0 and summary.pose.skipped == "no image was posed"
    assert down.stats_bodies == []


def test_a_garbled_stats_answer_never_fails_the_scan(puppet: FakePuppet) -> None:
    puppet.poses = {10: _pose([_person(primary=True, labels=STANDING)], 0)}
    puppet.stats_answer = {"puppet_version": "1.1", "items": "nope"}
    summary = _scan(THREE[:1])
    assert summary.pose.skipped.startswith("the /stats call failed")
    assert summary.results[0].pose_cluster is None and summary.results[0].pose_labels is not None


# --- what persists ----------------------------------------------------------


def _clustered_scan(puppet: FakePuppet):
    puppet.poses = {r: _pose([_person(primary=True, labels=STANDING)], 0) for r in (10, 20, 30)}
    puppet.stats_answer = _stats(
        [{"image": k, "usable": True, "cluster": 0, "labels": STANDING} for k in ("a.png", "b.png", "c.png")],
        [{"id": 0, "size": 3, "share": 1.0, "medoid": "b.png"}],
    )
    return _scan(THREE)


def test_no_keypoint_is_persisted_anywhere(puppet: FakePuppet, tmp_path: Path) -> None:
    summary = _clustered_scan(puppet)
    assert puppet.stats_bodies and str(MARK) in json.dumps(puppet.stats_bodies)  # they did go to /stats
    store = ScanStore(tmp_path / "store")
    saved = store.save(summary)
    manifest = tmp_path / "out" / "manifest.jsonl"
    write_manifest(summary.results, summary, manifest, {r.rel_path: r.rel_path for r in summary.results})
    report = tmp_path / "out" / "report.csv"
    write_report(summary.results, {}, set(), {}, report)
    dumps = [
        summary.model_dump_json(),
        json.dumps(summary.model_dump(mode="json", context=STORE_CONTEXT)),
        saved.read_text(),
        manifest.read_text(),
        report.read_text(),
    ]
    for text in dumps:
        assert "keypoints" not in text and str(MARK) not in text
    assert '"keypoints":' not in json.dumps(ImageResult.model_json_schema())  # no such field, at any depth


def test_the_manifest_row_and_report_carry_labels_and_cluster(puppet: FakePuppet, tmp_path: Path) -> None:
    summary = _clustered_scan(puppet)
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(summary.results, summary, manifest, {r.rel_path: r.rel_path for r in summary.results})
    rows = [ManifestRow.model_validate_json(line) for line in manifest.read_text().splitlines()]
    assert rows[0].manifest_version == "2.7"
    assert all(r.pose_labels == BodyPoseLabels(**STANDING) and r.pose_cluster == 0 for r in rows)

    report = tmp_path / "report.csv"
    write_report(summary.results, {}, set(), {}, report)
    first = next(csv.DictReader(report.open(encoding="utf-8")))
    assert (
        first["pose_check"],
        first["pose_stance"],
        first["pose_arms"],
        first["pose_facing"],
        first["pose_cluster"],
    ) == (
        "ok",
        "standing",
        "down",
        "front",
        "0",
    )


def test_unchecked_rows_write_null_never_a_label(tmp_path: Path) -> None:
    summary = _scan(THREE[:1])  # no puppet configured
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(summary.results, summary, manifest, {"a.png": "a.png"})
    row = json.loads(manifest.read_text())
    assert row["pose_labels"] is None and row["pose_cluster"] is None


def test_dropped_rows_recount_the_clusters_and_mark_the_rest_stale(puppet: FakePuppet) -> None:
    summary = _clustered_scan(puppet)
    drop_results(summary, {"b.png"})
    pose = summary.pose
    assert pose.stale and pose.usable == 2
    assert [(c.id, c.size, c.share, c.medoid_rel_path) for c in pose.clusters] == [(0, 2, 1.0, None)]
    assert pose.diversity == 0.42  # as measured: it cannot be redone without the keypoints


def test_no_request_field_names_a_puppet_url() -> None:
    assert set(PoseConfig.model_fields) == {"enabled"}
    req = ScanRequest.model_validate({"folder": "", "pose": {"enabled": True, "url": "http://evil.example"}})
    assert "evil" not in req.model_dump_json()


def test_a_speech_scan_refuses_the_pose_stage() -> None:
    with pytest.raises(ValueError, match="pose"):
        ScanRequest.model_validate({"folder": "", "target_profile": {"modality": "speech"}, "pose": {"enabled": True}})


# --- the real client against a stand-in server ------------------------------


class _Puppet(BaseHTTPRequestHandler):
    requests: list[dict[str, Any]] = []
    busy_first: int = 0

    def log_message(self, *args: Any) -> None:
        pass

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers["Content-Length"]))
        parts = urlsplit(self.path)
        type(self).requests.append(
            {
                "path": parts.path,
                "query": parse_qs(parts.query),
                "priority": self.headers.get("X-Argus-Priority"),
                "content_type": self.headers.get("Content-Type"),
                "body": body,
            }
        )
        if type(self).busy_first > 0:
            type(self).busy_first -= 1
            self.send_response(503)
            self.send_header("Retry-After", "1")
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")
            return
        answer = _pose([_person(primary=True, labels=STANDING)], 0) if parts.path == "/pose" else _stats([], [])
        out = json.dumps(answer).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    _Puppet.requests = []
    _Puppet.busy_first = 0
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Puppet)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    waits: list[float] = []
    monkeypatch.setattr(bodypose, "_sleep", waits.append)
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_the_client_sends_raw_bytes_with_name_face_and_batch_priority(server: str) -> None:
    posed = bodypose.PuppetClient(server, 5).pose(b"\x89PNG", "sub dir/my photo.png", "1.0,2.0,3.0,4.0")
    assert posed.labels == BodyPoseLabels(**STANDING)
    (req,) = _Puppet.requests
    assert req["path"] == "/pose" and req["priority"] == "batch"
    assert req["content_type"] == "application/octet-stream" and req["body"] == b"\x89PNG"
    assert req["query"] == {"name": ["my_photo.png"], "face": ["1.0,2.0,3.0,4.0"]}


def test_a_503_is_retried_after_its_retry_after(server: str) -> None:
    _Puppet.busy_first = 2
    client = bodypose.PuppetClient(server, 5)
    assert client.pose(b"x", "a.png").has_person
    assert len(_Puppet.requests) == 3
    assert client.stats(b'{"results": []}')["puppet_version"] == "1.1"
    assert _Puppet.requests[-1]["path"] == "/stats" and _Puppet.requests[-1]["priority"] == "batch"


def test_retries_are_bounded(server: str) -> None:
    _Puppet.busy_first = 100
    with pytest.raises(bodypose.PoseCheckFailed, match="HTTP 503"):
        bodypose.PuppetClient(server, 5).pose(b"x", "a.png")
    assert len(_Puppet.requests) == bodypose.MAX_RETRIES + 1


def test_images_go_only_to_the_configured_server(server: str, monkeypatch: pytest.MonkeyPatch) -> None:
    # The URL comes from the environment alone: no scan field can point the stage elsewhere.
    monkeypatch.setenv(bodypose.URL_ENV, server)
    cfg = PoseConfig.model_validate({"enabled": True, "url": "http://evil.example"})
    summary = _scan(THREE[:2], pose_cfg=cfg)
    assert summary.pose.posed == 2
    assert [r["path"] for r in _Puppet.requests] == ["/pose", "/pose", "/stats"]


def test_without_the_url_no_image_is_sent(server: str, monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[str] = []
    monkeypatch.setattr(bodypose.PuppetClient, "_post", lambda self, url, *a: sent.append(url))
    summary = _scan(THREE)
    assert summary.pose_config.enabled is False and sent == [] and _Puppet.requests == []
