"""NudeNet over HTTP: configuration, the client against a fake server, filters, report.

The fake server speaks NudeNet's documented ``/infer`` API, so the real
urllib client, multipart encoding and response parsing are all exercised.
"""

from __future__ import annotations

import csv
import io
import json
import threading
import time
from collections.abc import Iterator
from email.parser import BytesParser
from email.policy import HTTP
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from PIL import Image
from pydantic import ValidationError

from argus_curator import nsfw, remote
from argus_curator.export import write_report
from argus_curator.models import (
    ExportRequest,
    FaceConfig,
    ImageResult,
    NsfwConfig,
    NsfwResult,
    ScanConfig,
    ScanSummary,
    TargetProfile,
)
from argus_curator.runtime import ModelUnavailable
from argus_curator.scanner import scan_items
from argus_curator.selection import decide_selection, nsfw_filter_problem

# Canned detections, keyed by the image's red channel (see _png).
_DETECTIONS = {
    10: [],
    20: [
        {"class": "FACE_FEMALE", "score": 0.91, "box": [10, 10, 30, 30]},
        {"class": "BELLY_EXPOSED", "score": 0.4, "box": [10, 50, 30, 20]},
    ],
    30: [
        {"class": "FEMALE_BREAST_EXPOSED", "score": 0.72, "box": [5, 5, 20, 20]},
        {"class": "FEMALE_BREAST_EXPOSED", "score": 0.66, "box": [30, 5, 20, 20]},
        {"class": "FACE_FEMALE", "score": 0.8, "box": [10, 0, 10, 10]},
    ],
}
_FAIL = 99  # this red value gets an HTTP 500


def _png(red: int, size: tuple[int, int] = (64, 48)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (red, 0, 0)).save(buf, "PNG")
    return buf.getvalue()


class _NudeNet(BaseHTTPRequestHandler):
    seen: list[dict] = []

    def log_message(self, *args) -> None:  # keep test output quiet
        pass

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers["Content-Length"]))
        msg = BytesParser(policy=HTTP).parsebytes(
            f"Content-Type: {self.headers['Content-Type']}\r\n\r\n".encode() + body
        )
        parts = list(msg.iter_parts())
        part = parts[0]
        red = Image.open(io.BytesIO(part.get_content())).convert("RGB").getpixel((0, 0))[0]
        type(self).seen.append(
            {"path": self.path, "field": part.get_param("name", header="content-disposition"),
             "filename": part.get_filename(), "parts": len(parts)}
        )  # fmt: skip
        if red == _FAIL:
            self.send_response(500)
            self.end_headers()
            return
        out = json.dumps({"prediction": [_DETECTIONS.get(red, [])], "success": True}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


@pytest.fixture
def nudenet(monkeypatch: pytest.MonkeyPatch) -> Iterator[type[_NudeNet]]:
    _NudeNet.seen = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _NudeNet)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv(nsfw.URL_ENV, f"http://127.0.0.1:{server.server_port}")
    try:
        yield _NudeNet
    finally:
        server.shutdown()
        server.server_close()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_endpoint_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    assert nsfw.endpoint() is None and not nsfw.configured()
    monkeypatch.setenv(nsfw.URL_ENV, "http://nudenet:8080/")
    assert nsfw.endpoint() == "http://nudenet:8080/infer"
    monkeypatch.setenv(nsfw.URL_ENV, "https://guard.lan/x/infer")
    assert nsfw.endpoint() == "https://guard.lan/x/infer"
    monkeypatch.setenv(nsfw.URL_ENV, "file:///etc/passwd")
    assert nsfw.configured()  # set, so a scan must not silently skip the check...
    with pytest.raises(ModelUnavailable, match="http"):
        nsfw.endpoint()  # ...and fails on use instead


@pytest.mark.parametrize(
    ("url", "enabled", "expected"),
    [(None, None, False), ("http://n:8080", None, True), ("http://n:8080", False, False), (None, True, True)],
)
def test_opt_out_semantics(monkeypatch: pytest.MonkeyPatch, url, enabled, expected) -> None:
    if url:
        monkeypatch.setenv(nsfw.URL_ENV, url)
    assert nsfw.resolve_enabled(NsfwConfig(enabled=enabled)) is expected


def test_to_result_scores_only_exposed_classes() -> None:
    r = nsfw.to_result(
        [
            {"class": "FACE_FEMALE", "score": 0.95, "box": [0, 0, 1, 1]},  # a face is not nudity
            {"class": "BUTTOCKS_EXPOSED", "score": 0.31, "box": [0, 0, 1, 1]},
            {"class": "BUTTOCKS_EXPOSED", "score": 0.6, "box": [0, 0, 1, 1]},
            {"class": "NOT_A_CLASS", "score": 0.99, "box": [0, 0, 1, 1]},
            {"class": "ANUS_EXPOSED", "score": "high", "box": [0, 0, 1, 1]},
        ]
    )
    assert r.labels == {"FACE_FEMALE": 0.95, "BUTTOCKS_EXPOSED": 0.6}
    assert list(r.labels) == ["FACE_FEMALE", "BUTTOCKS_EXPOSED"]  # highest first
    assert r.exposed_score == 0.6
    assert len(r.detections) == 3


# ---------------------------------------------------------------------------
# Client against the fake server
# ---------------------------------------------------------------------------


def test_check_images_posts_each_image_and_records_results(nudenet: type[_NudeNet]) -> None:
    items = [
        ("clean.png", "/d/clean.png", _png(10)),
        ('we"ird\r\nname.png', "/d/w", _png(20)),
        ("topless.png", "/d/t", _png(30)),
        ("broken.png", "/d/b", _png(_FAIL)),
        ("blurry.png", "/d/blurry", _png(30)),
    ]
    results = [ImageResult(rel_path=rel, abs_path=a, passed=rel != "blurry.png") for rel, a, _ in items]
    seen: list[tuple[int, int]] = []
    assert nsfw.check_images(results, items, NsfwConfig(), progress=lambda d, t: seen.append((d, t)))

    by = {r.rel_path: r for r in results}
    assert by["clean.png"].nsfw == NsfwResult()
    assert by['we"ird\r\nname.png'].nsfw.exposed_score == 0.0  # face + belly: not exposed nudity
    assert by['we"ird\r\nname.png'].nsfw.labels["BELLY_EXPOSED"] == 0.4
    assert by["topless.png"].nsfw.exposed_score == 0.72
    assert by["broken.png"].nsfw is None  # failed check: unknown, not clean
    assert by["blurry.png"].nsfw.exposed_score == 0.72  # quality rejects are checked too
    assert seen[0] == (0, 5) and seen[-1] == (5, 5)

    assert {s["path"] for s in nudenet.seen} == {"/infer"}
    assert {s["field"] for s in nudenet.seen} == {"f1"} and {s["parts"] for s in nudenet.seen} == {1}
    assert "we_ird__name.png" in {s["filename"] for s in nudenet.seen}  # header-safe


def test_disabled_check_sends_nothing(nudenet: type[_NudeNet]) -> None:
    results = [ImageResult(rel_path="a.png", abs_path="/d/a", passed=True)]
    assert not nsfw.check_images(results, [("a.png", "/d/a", _png(30))], NsfwConfig(enabled=False))
    assert nudenet.seen == [] and results[0].nsfw is None


def test_required_check_without_url_fails() -> None:
    results = [ImageResult(rel_path="a.png", abs_path="/d/a", passed=True)]
    with pytest.raises(ModelUnavailable, match=nsfw.URL_ENV):
        nsfw.check_images(results, [("a.png", "/d/a", _png(30))], NsfwConfig(enabled=True))


@pytest.mark.parametrize(
    "payload",
    [
        {"prediction": [[]], "success": False},
        {"prediction": [[], []], "success": True},
        {"success": True},
        ["not", "an", "object"],
    ],
)
def test_bad_responses_are_failures(monkeypatch: pytest.MonkeyPatch, payload) -> None:
    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc) -> None:
            self.close()

    monkeypatch.setattr(remote._OPENER, "open", lambda req, timeout: _Resp(json.dumps(payload).encode()))
    with pytest.raises(nsfw.NsfwCheckFailed):
        nsfw.check(_png(10), "a.png", "http://n/infer", 5.0)


def test_malformed_detections_are_skipped() -> None:
    result = nsfw.to_result(
        [
            "not-a-dict",
            {"class": "BUTTOCKS_EXPOSED", "score": 0.7, "box": [0, 0, "w", None]},
            {"class": "BUTTOCKS_EXPOSED", "score": True, "box": [0, 0, 1, 1]},
            {"class": "BELLY_EXPOSED", "score": 0.5, "box": [0, 0, 1, 1]},
        ]
    )
    assert result.labels == {"BELLY_EXPOSED": 0.5} and result.exposed_score == 0.0


def test_unexpected_errors_leave_the_image_unchecked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(nsfw.URL_ENV, "http://n")

    def boom(*_args):
        raise ValueError("unexpected")

    monkeypatch.setattr(nsfw, "check", boom)
    results = [ImageResult(rel_path="a.png", abs_path="/d/a", passed=True)]
    assert nsfw.check_images(results, [("a.png", "/d/a", _png(30))], NsfwConfig()) is True
    assert results[0].nsfw is None


def test_unreachable_server_is_given_up_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(nsfw.URL_ENV, "http://n")
    monkeypatch.setenv(nsfw.WORKERS_ENV, "1")
    calls = []

    def down(*_args):
        calls.append(1)
        time.sleep(0.02)  # a real outage waits out a timeout; give the loop time to react
        raise nsfw.NsfwCheckFailed("NudeNet request failed: timed out")

    monkeypatch.setattr(nsfw, "check", down)
    results = [ImageResult(rel_path=f"{i}.png", abs_path=f"/d/{i}", passed=True) for i in range(50)]
    items = [(r.rel_path, r.abs_path, b"x") for r in results]
    assert nsfw.check_images(results, items, NsfwConfig()) is True
    assert len(calls) <= remote.MAX_FAILURES_WITHOUT_SUCCESS + 2
    assert all(r.nsfw is None for r in results)


def test_scan_runs_nudenet_when_configured(nudenet: type[_NudeNet]) -> None:
    items = [("a.png", "/d/a", _png(30, (600, 600))), ("b.png", "/d/b", _png(10, (600, 600)))]
    summary = scan_items(items, TargetProfile(), ScanConfig(), FaceConfig())
    assert summary.nsfw_config.enabled is True
    assert {r.rel_path: r.nsfw.exposed_score for r in summary.results} == {"a.png": 0.72, "b.png": 0.0}

    opted_out = scan_items(items, TargetProfile(), ScanConfig(), FaceConfig(), nsfw_cfg=NsfwConfig(enabled=False))
    assert opted_out.nsfw_config.enabled is False
    assert all(r.nsfw is None for r in opted_out.results)


def test_scan_without_url_records_check_off() -> None:
    summary = scan_items([("a.png", "/d/a", _png(30, (600, 600)))], TargetProfile(), ScanConfig(), FaceConfig())
    assert summary.nsfw_config.enabled is False


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


def _img(rel: str, labels: dict[str, float] | None) -> ImageResult:
    result = (
        None
        if labels is None
        else nsfw.to_result([{"class": k, "score": v, "box": [0, 0, 1, 1]} for k, v in labels.items()])
    )
    return ImageResult(rel_path=rel, abs_path=f"/d/{rel}", score=0.9, passed=True, nsfw=result)


def test_total_and_per_class_thresholds() -> None:
    results = [
        _img("clean.jpg", {}),
        _img("portrait.jpg", {"FACE_FEMALE": 0.93}),
        _img("at-threshold.jpg", {"BUTTOCKS_EXPOSED": 0.6}),
        _img("below.jpg", {"FEMALE_BREAST_EXPOSED": 0.59}),
        _img("midriff.jpg", {"BELLY_EXPOSED": 0.85}),
        _img("unchecked.jpg", None),
    ]
    req = ExportRequest(dest="/o", min_score=0.0, nsfw_threshold=0.6, nsfw_label_thresholds={"BELLY_EXPOSED": 0.8})
    selected, why = decide_selection(results, req, 0.0)
    assert {r.rel_path for r in selected} == {"clean.jpg", "portrait.jpg", "below.jpg"}
    assert why["at-threshold.jpg"] == "nsfw-exposed"  # thresholds are inclusive
    assert why["midriff.jpg"] == "nsfw-belly-exposed"
    assert why["unchecked.jpg"] == "nsfw-unchecked"  # fails closed


def test_exposed_set_is_configurable() -> None:
    results = [_img("a.jpg", {"MALE_BREAST_EXPOSED": 0.9}), _img("b.jpg", {"BUTTOCKS_EXPOSED": 0.9})]
    req = ExportRequest(dest="/o", min_score=0.0, nsfw_threshold=0.5, nsfw_exposed_labels=["MALE_BREAST_EXPOSED"])
    selected, why = decide_selection(results, req, 0.0)
    assert [r.rel_path for r in selected] == ["b.jpg"] and why["a.jpg"] == "nsfw-exposed"


def test_no_nsfw_filter_keeps_unchecked_images() -> None:
    selected, _ = decide_selection([_img("a.jpg", None)], ExportRequest(dest="/o", min_score=0.0), 0.0)
    assert [r.rel_path for r in selected] == ["a.jpg"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"nsfw_threshold": 0.1},
        {"nsfw_label_thresholds": {"FEMALE_BREAST_EXPOSED": 0.05}},
        {"nsfw_label_thresholds": {"NOT_A_CLASS": 0.5}},
        {"nsfw_exposed_labels": ["NOT_A_CLASS"]},
        {"nsfw_exposed_labels": []},  # would make nsfw_threshold pass everything
    ],
)
def test_invalid_nsfw_filters_are_rejected(kwargs) -> None:
    with pytest.raises(ValidationError):
        ExportRequest(dest="/o", **kwargs)


def test_filters_on_a_scan_without_nudenet_are_a_problem() -> None:
    req = ExportRequest(dest="/o", nsfw_threshold=0.5)
    base = {
        "scan_id": "s", "folder": "/d", "target_profile": TargetProfile().model_dump(),
        "config": ScanConfig().model_dump(), "faces_config": FaceConfig().model_dump(),
        "total": 0, "passed": 0, "rejected": 0, "duplicates": 0, "similar_clusters": 0,
    }  # fmt: skip
    legacy = ScanSummary.model_validate(base)  # cached before NudeNet existed
    assert "rescan" in nsfw_filter_problem(req, legacy)
    ran = ScanSummary.model_validate({**base, "nsfw_config": {"enabled": True}})
    assert nsfw_filter_problem(req, ran) is None
    assert nsfw_filter_problem(ExportRequest(dest="/o"), legacy) is None


def test_report_includes_nsfw_columns(tmp_path: Path) -> None:
    results = [_img("a.jpg", {"FACE_FEMALE": 0.9, "BUTTOCKS_EXPOSED": 0.61}), _img("b.jpg", None)]
    report = tmp_path / "r.csv"
    write_report(results, {}, set(), {}, report)
    rows = {r["rel_path"]: r for r in csv.DictReader(report.open())}
    assert rows["a.jpg"]["nsfw_exposed_score"] == "0.61"
    assert rows["a.jpg"]["nsfw_labels"] == "FACE_FEMALE:0.90;BUTTOCKS_EXPOSED:0.61"
    assert rows["b.jpg"]["nsfw_exposed_score"] == "" and rows["b.jpg"]["nsfw_labels"] == ""
