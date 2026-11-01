"""EraX over HTTP: mapping onto NudeNet labels, fill-in when NudeNet misses."""

from __future__ import annotations

import io
import json
import threading
import time
from collections.abc import Iterator
from email.parser import BytesParser
from email.policy import HTTP
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from PIL import Image

from argus_curator import erax, nsfw
from argus_curator.models import (
    EraxConfig,
    ExportRequest,
    FaceConfig,
    ImageResult,
    NsfwConfig,
    ScanConfig,
    TargetProfile,
)
from argus_curator.runtime import ModelUnavailable, ScanCancelled
from argus_curator.scanner import scan_items
from argus_curator.selection import decide_selection

_DETECTIONS = {
    10: [],
    20: [{"class": "nipple", "score": 0.81, "box": [8, 12, 16, 14]}],
    30: [
        {"class": "penis", "score": 0.7, "box": [4, 4, 10, 12]},
        {"class": "make_love", "score": 0.55, "box": [0, 0, 40, 40]},
    ],
}
_FAIL = 99


def _png(red: int, size: tuple[int, int] = (64, 48)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (red, 0, 0)).save(buf, "PNG")
    return buf.getvalue()


class _EraX(BaseHTTPRequestHandler):
    seen: list[dict] = []

    def log_message(self, *args) -> None:
        pass

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers["Content-Length"]))
        msg = BytesParser(policy=HTTP).parsebytes(
            f"Content-Type: {self.headers['Content-Type']}\r\n\r\n".encode() + body
        )
        part = next(msg.iter_parts())
        red = Image.open(io.BytesIO(part.get_content())).convert("RGB").getpixel((0, 0))[0]
        type(self).seen.append({"path": self.path, "filename": part.get_filename()})
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
def erax_server(monkeypatch: pytest.MonkeyPatch) -> Iterator[type[_EraX]]:
    _EraX.seen = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _EraX)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv(erax.URL_ENV, f"http://127.0.0.1:{server.server_port}")
    try:
        yield _EraX
    finally:
        server.shutdown()
        server.server_close()


def test_endpoint_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    assert erax.endpoint() is None and not erax.configured()
    monkeypatch.setenv(erax.URL_ENV, "http://erax:8080/")
    assert erax.endpoint() == "http://erax:8080/infer"
    monkeypatch.setenv(erax.URL_ENV, "file:///etc/passwd")
    assert erax.configured()
    with pytest.raises(ModelUnavailable, match="http"):
        erax.endpoint()


def test_resolve_enabled_follows_nudenet(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(erax.URL_ENV, "http://erax:8080")
    assert erax.resolve_enabled(NsfwConfig()) is False  # NudeNet not configured
    monkeypatch.setenv(nsfw.URL_ENV, "http://nudenet:8080")
    assert erax.resolve_enabled(NsfwConfig()) is True
    assert erax.resolve_enabled(NsfwConfig(enabled=False)) is False


def test_to_erax_result_drops_unknown_and_low_scores() -> None:
    r = erax.to_erax_result(
        [
            {"class": "nipple", "score": 0.81, "box": [1, 2, 3, 4]},
            {"class": "nipple", "score": 0.4, "box": [0, 0, 1, 1]},
            {"class": "FACE_FEMALE", "score": 0.99, "box": [0, 0, 1, 1]},
            {"class": "penis", "score": 0.19, "box": [0, 0, 1, 1]},
        ]
    )
    assert r.labels == {"nipple": 0.81}
    assert r.detections[0].bbox == [1.0, 2.0, 3.0, 4.0]


def test_mapped_detections_cover_make_love() -> None:
    result = erax.to_erax_result([{"class": "make_love", "score": 0.6, "box": [0, 0, 8, 8]}])
    mapped = erax.mapped_detections(result)
    assert {d["class"] for d in mapped} == {"FEMALE_GENITALIA_EXPOSED", "MALE_GENITALIA_EXPOSED"}
    assert all(d["score"] == 0.6 for d in mapped)


def test_fill_nsfw_when_nudenet_missed() -> None:
    missed = ImageResult(
        rel_path="a.png",
        abs_path="/d/a",
        passed=True,
        nsfw=nsfw.to_result([]),
        erax=erax.to_erax_result([{"class": "nipple", "score": 0.77, "box": [2, 2, 6, 6]}]),
    )
    scored = ImageResult(
        rel_path="b.png",
        abs_path="/d/b",
        passed=True,
        nsfw=nsfw.to_result([{"class": "BUTTOCKS_EXPOSED", "score": 0.9, "box": [0, 0, 4, 4]}]),
        erax=erax.to_erax_result([{"class": "anus", "score": 0.8, "box": [1, 1, 2, 2]}]),
    )
    failed = ImageResult(
        rel_path="c.png",
        abs_path="/d/c",
        passed=True,
        nsfw=None,
        erax=erax.to_erax_result([{"class": "vagina", "score": 0.5, "box": [0, 0, 3, 3]}]),
    )
    assert erax.fill_nsfw([missed, scored, failed]) == 2
    assert missed.erax_filled is True
    assert missed.nsfw.labels["FEMALE_BREAST_EXPOSED"] == 0.77
    assert missed.nsfw.exposed_score == 0.77
    assert scored.erax_filled is False
    assert scored.nsfw.labels == {"BUTTOCKS_EXPOSED": 0.9}
    assert failed.erax_filled is True
    assert failed.nsfw.labels["FEMALE_GENITALIA_EXPOSED"] == 0.5
    assert failed.nudenet_failed is True
    assert missed.nudenet_failed is False


def test_fill_keeps_non_exposed_nudenet_detections() -> None:
    row = ImageResult(
        rel_path="a.png",
        abs_path="/d/a",
        passed=True,
        nsfw=nsfw.to_result([{"class": "FACE_FEMALE", "score": 0.91, "box": [0, 0, 4, 4]}]),
        erax=erax.to_erax_result([{"class": "penis", "score": 0.6, "box": [5, 5, 4, 8]}]),
    )
    erax.fill_nsfw([row])
    assert row.nsfw.labels["FACE_FEMALE"] == 0.91
    assert row.nsfw.labels["MALE_GENITALIA_EXPOSED"] == 0.6
    assert row.nsfw.exposed_score == 0.6


def test_check_images_posts_and_records(erax_server: type[_EraX]) -> None:
    items = [("clean.png", "/d/c", _png(10)), ("hit.png", "/d/h", _png(20)), ("broken.png", "/d/b", _png(_FAIL))]
    results = [ImageResult(rel_path=rel, abs_path=a, passed=True) for rel, a, _ in items]
    assert erax.check_images(results, items) is True
    by = {r.rel_path: r for r in results}
    assert by["clean.png"].erax == erax.to_erax_result([])
    assert by["hit.png"].erax.labels["nipple"] == 0.81
    assert by["broken.png"].erax is None
    assert {s["path"] for s in erax_server.seen} == {"/infer"}


def test_scan_fills_when_nudenet_returns_empty(erax_server: type[_EraX], monkeypatch: pytest.MonkeyPatch) -> None:
    class _EmptyNudeNet:
        def check(self, data: bytes, filename: str = "image"):
            return nsfw.to_result([])

    monkeypatch.setenv(nsfw.URL_ENV, "http://nudenet:8080")
    monkeypatch.setattr(nsfw, "client", lambda: _EmptyNudeNet())
    items = [("a.png", "/d/a", _png(20, (600, 600)))]
    summary = scan_items(items, TargetProfile(), ScanConfig(), FaceConfig())
    assert summary.nsfw_config.enabled is True
    assert summary.erax_config == EraxConfig(enabled=True)
    row = summary.results[0]
    assert row.erax_filled is True
    assert row.nsfw.exposed_score == 0.81
    assert row.erax.labels["nipple"] == 0.81


def test_scan_without_erax_url_leaves_nudenet_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    class _EmptyNudeNet:
        def check(self, data: bytes, filename: str = "image"):
            return nsfw.to_result([])

    monkeypatch.setenv(nsfw.URL_ENV, "http://nudenet:8080")
    monkeypatch.setattr(nsfw, "client", lambda: _EmptyNudeNet())
    items = [("a.png", "/d/a", _png(20, (600, 600)))]
    summary = scan_items(items, TargetProfile(), ScanConfig(), FaceConfig())
    assert summary.erax_config.enabled is False
    assert summary.results[0].nsfw.exposed_score == 0.0
    assert summary.results[0].erax is None


def test_scan_fails_on_malformed_erax_url_before_nudenet(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class _EmptyNudeNet:
        def check(self, data: bytes, filename: str = "image"):
            calls.append(filename)
            return nsfw.to_result([])

    monkeypatch.setenv(nsfw.URL_ENV, "http://nudenet:8080")
    monkeypatch.setenv(erax.URL_ENV, "file:///etc/passwd")
    monkeypatch.setattr(nsfw, "client", lambda: _EmptyNudeNet())
    items = [("a.png", "/d/a", _png(20, (600, 600)))]
    with pytest.raises(ModelUnavailable):
        scan_items(items, TargetProfile(), ScanConfig(), FaceConfig())
    assert calls == []


def test_cancel_is_noticed_during_erax(erax_server: type[_EraX], monkeypatch: pytest.MonkeyPatch) -> None:
    class _EmptyNudeNet:
        def check(self, data: bytes, filename: str = "image"):
            return nsfw.to_result([])

    monkeypatch.setenv(nsfw.URL_ENV, "http://nudenet:8080")
    monkeypatch.setattr(nsfw, "client", lambda: _EmptyNudeNet())
    monkeypatch.setenv(erax.WORKERS_ENV, "1")
    items = [(f"{i}.png", f"/d/{i}", _png(20, (600, 600))) for i in range(12)]
    stop = {"now": False}
    real = erax.EraxClient.check

    def check_then_stop(self, data: bytes, filename: str = "image"):
        stop["now"] = True
        return real(self, data, filename)

    monkeypatch.setattr(erax.EraxClient, "check", check_then_stop)
    with pytest.raises(ScanCancelled):
        scan_items(items, TargetProfile(), ScanConfig(), FaceConfig(), cancelled=lambda: stop["now"])
    # Stopped at the next progress step, not after every image went to EraX.
    assert len(erax_server.seen) < len(items)


def test_nipple_on_a_nudenet_male_chest_is_not_filled() -> None:
    chest = ImageResult(
        rel_path="a.png",
        abs_path="/d/a",
        passed=True,
        nsfw=nsfw.to_result([{"class": "MALE_BREAST_EXPOSED", "score": 0.8, "box": [0, 0, 20, 20]}]),
        erax=erax.to_erax_result([{"class": "nipple", "score": 0.7, "box": [5, 5, 3, 3]}]),
    )
    both = ImageResult(
        rel_path="b.png",
        abs_path="/d/b",
        passed=True,
        nsfw=nsfw.to_result([{"class": "MALE_BREAST_EXPOSED", "score": 0.8, "box": [0, 0, 20, 20]}]),
        erax=erax.to_erax_result(
            [
                {"class": "nipple", "score": 0.7, "box": [5, 5, 3, 3]},
                {"class": "nipple", "score": 0.6, "box": [60, 5, 3, 3]},
            ]
        ),
    )
    assert erax.fill_nsfw([chest, both]) == 1
    assert chest.erax_filled is False and chest.nsfw.exposed_score == 0.0
    assert chest.erax.labels["nipple"] == 0.7  # the raw EraX result is still kept
    # The nipple away from the male chest still counts.
    assert both.nsfw.labels["FEMALE_BREAST_EXPOSED"] == 0.6
    assert both.nsfw.labels["MALE_BREAST_EXPOSED"] == 0.8


def test_erax_alone_under_threshold_stays_unchecked() -> None:
    def failed(rel: str, score: float) -> ImageResult:
        row = ImageResult(
            rel_path=rel,
            abs_path=f"/d/{rel}",
            score=0.9,
            passed=True,
            nsfw=None,
            erax=erax.to_erax_result([{"class": "nipple", "score": score, "box": [0, 0, 3, 3]}]),
        )
        erax.fill_one(row)
        return row

    rows = [failed("weak.png", 0.25), failed("strong.png", 0.9)]
    _, why = decide_selection(rows, ExportRequest(dest="/o", min_score=0.0, nsfw_threshold=0.5), 0.0)
    assert why["weak.png"] == "nsfw-unchecked"  # EraX has no buttocks class: not a pass
    assert why["strong.png"] == "nsfw-exposed"
    assert rows[0].nsfw.exposed_score == 0.25  # still blurred and paired


class _ExposedOrEmptyNudeNet:
    """NudeNet that scores red 30 as exposed and sees nothing on anything else."""

    def check(self, data: bytes, filename: str = "image"):
        red = Image.open(io.BytesIO(data)).convert("RGB").getpixel((0, 0))[0]
        if red == 30:
            return nsfw.to_result([{"class": "BUTTOCKS_EXPOSED", "score": 0.9, "box": [0, 0, 4, 4]}])
        return nsfw.to_result([])


def test_nudenet_exposed_rows_do_not_wait_for_erax(monkeypatch: pytest.MonkeyPatch) -> None:
    exposed_sent = threading.Event()

    class _SlowEraX:
        url = "http://erax:8080/infer"

        def check(self, data: bytes, filename: str = "image"):
            # Held until the NudeNet-exposed row has gone out (or 5 s, to fail rather than hang).
            exposed_sent.wait(5)
            return erax.to_erax_result([{"class": "nipple", "score": 0.7, "box": [0, 0, 3, 3]}])

    monkeypatch.setenv(nsfw.URL_ENV, "http://nudenet:8080")
    monkeypatch.setenv(erax.URL_ENV, "http://erax:8080")
    monkeypatch.setattr(nsfw, "client", lambda: _ExposedOrEmptyNudeNet())
    monkeypatch.setattr(erax, "client", lambda: _SlowEraX())
    chunks: list[dict] = []

    def on_chunk(payload: dict) -> None:
        if payload["phase"] != "nsfw":
            return
        chunks.append(payload)
        if any(r["rel_path"] == "exposed.png" for r in payload["results"]):
            exposed_sent.set()

    items = [("exposed.png", "/d/e", _png(30, (600, 600))), ("missed.png", "/d/m", _png(20, (600, 600)))]
    started = time.monotonic()
    summary = scan_items(items, TargetProfile(), ScanConfig(), FaceConfig(), chunk=on_chunk)
    assert time.monotonic() - started < 4, "the exposed row waited for EraX"
    first = [r["rel_path"] for r in chunks[0]["results"]]
    assert first == ["exposed.png"]
    assert [r["rel_path"] for c in chunks for r in c["results"]].count("missed.png") == 1
    assert chunks[-1]["done"] == chunks[-1]["total"] == 2
    by = {r.rel_path: r for r in summary.results}
    # EraX results are kept on every row, the NudeNet-exposed one included.
    assert by["exposed.png"].erax.labels["nipple"] == 0.7
    assert by["exposed.png"].erax_filled is False
    assert by["missed.png"].erax_filled is True
    assert summary.erax_config.enabled is True


def test_fill_when_erax_answers_before_nudenet(monkeypatch: pytest.MonkeyPatch) -> None:
    erax_answered = threading.Event()

    class _LateNudeNet:
        def check(self, data: bytes, filename: str = "image"):
            erax_answered.wait(5)
            time.sleep(0.05)  # let the EraX row be recorded first
            return nsfw.to_result([])

    class _QuickEraX:
        url = "http://erax:8080/infer"

        def check(self, data: bytes, filename: str = "image"):
            erax_answered.set()
            return erax.to_erax_result([{"class": "vagina", "score": 0.66, "box": [0, 0, 3, 3]}])

    monkeypatch.setenv(nsfw.URL_ENV, "http://nudenet:8080")
    monkeypatch.setenv(erax.URL_ENV, "http://erax:8080")
    monkeypatch.setattr(nsfw, "client", lambda: _LateNudeNet())
    monkeypatch.setattr(erax, "client", lambda: _QuickEraX())
    chunks: list[dict] = []
    items = [("a.png", "/d/a", _png(20, (600, 600)))]
    summary = scan_items(
        items,
        TargetProfile(),
        ScanConfig(),
        FaceConfig(),
        chunk=lambda p: chunks.append(p) if p["phase"] == "nsfw" else None,
    )
    row = summary.results[0]
    assert row.erax_filled is True and row.nsfw.exposed_score == 0.66
    sent = [r for c in chunks for r in c["results"]]
    assert len(sent) == 1 and sent[0]["nsfw"]["exposed_score"] == 0.66
