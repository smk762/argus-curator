"""SingGuard over HTTP: configuration, the client against a fake server, filters, report.

The fake server speaks SingGuard's ``/api/guard/check`` API (response shapes
taken from a live SingGuard), so the real urllib client, request body and
response parsing are all exercised.
"""

from __future__ import annotations

import base64
import csv
import io
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from PIL import Image
from pydantic import ValidationError

from argus_curator import guard
from argus_curator.export import write_report
from argus_curator.models import (
    ExportRequest,
    FaceConfig,
    GuardConfig,
    GuardResult,
    ImageResult,
    ScanConfig,
    ScanSummary,
    TargetProfile,
)
from argus_curator.runtime import ModelUnavailable
from argus_curator.scanner import scan_items
from argus_curator.selection import decide_selection, guard_filter_problem

_CRIME = "B. Real-World Crimes & Public Safety"
_GORE = "C. Unethical Behavior"
_SAFE = {"safe": True, "verdict": "safe", "rule": "Safe", "explanation": "Safe. No rule in the active policy matched."}
# Canned answers, keyed by the image's red channel (see _png).
_ANSWERS: dict[int, dict] = {
    10: {**_SAFE, "categories": []},
    20: {
        "safe": False,
        "verdict": "unsafe",
        "rule": _CRIME,
        "explanation": f"Triggered rule: {_CRIME}.",
        "categories": [{"code": _CRIME, "label": _CRIME, "why": "-"}, {"code": _GORE, "label": _GORE, "why": "-"}],
    },
    30: {"safe": False, "verdict": "unsafe", "rule": _GORE, "explanation": "", "categories": []},
    40: {"verdict": "safe"},  # no boolean verdict: a failed check, not a pass
}
_FAIL = 99  # this red value gets an HTTP 500


def _png(red: int, size: tuple[int, int] = (64, 48)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (red, 0, 0)).save(buf, "PNG")
    return buf.getvalue()


class _SingGuard(BaseHTTPRequestHandler):
    seen: list[dict] = []

    def log_message(self, *args) -> None:  # keep test output quiet
        pass

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        image = Image.open(io.BytesIO(base64.b64decode(body["image_base64"])))
        red = image.convert("RGB").getpixel((0, 0))[0]
        type(self).seen.append(
            {"path": self.path, "size": image.size, **{k: v for k, v in body.items() if k != "image_base64"}}
        )
        if red == _FAIL:
            self.send_response(500)
            self.end_headers()
            return
        answer = {**_ANSWERS.get(red, _ANSWERS[10]), "model": "SingGuard-4b", "elapsed_ms": 5}
        out = json.dumps(answer).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


@pytest.fixture
def singguard(monkeypatch: pytest.MonkeyPatch) -> Iterator[type[_SingGuard]]:
    _SingGuard.seen = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _SingGuard)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv(guard.URL_ENV, f"http://127.0.0.1:{server.server_port}")
    try:
        yield _SingGuard
    finally:
        server.shutdown()
        server.server_close()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_endpoint_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    assert guard.endpoint() is None and not guard.configured()
    monkeypatch.setenv(guard.URL_ENV, "http://192.168.1.198:9150/")
    assert guard.endpoint() == "http://192.168.1.198:9150/api/guard/check"
    monkeypatch.setenv(guard.URL_ENV, "gopher://guard")
    assert guard.configured()
    with pytest.raises(ModelUnavailable, match="http"):
        guard.endpoint()


@pytest.mark.parametrize(
    ("url", "enabled", "expected"),
    [(None, None, False), ("http://g:9150", None, True), ("http://g:9150", False, False), (None, True, True)],
)
def test_opt_out_semantics(monkeypatch: pytest.MonkeyPatch, url, enabled, expected) -> None:
    if url:
        monkeypatch.setenv(guard.URL_ENV, url)
    assert guard.resolve_enabled(GuardConfig(enabled=enabled)) is expected


def test_thinking_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    assert guard.thinking() == "fast"
    monkeypatch.setenv(guard.THINKING_ENV, "fast-slow")
    assert guard.thinking() == "fast-slow"
    monkeypatch.setenv(guard.THINKING_ENV, "slow")
    with pytest.raises(ModelUnavailable, match="fast-slow"):
        guard.thinking()


def test_policy_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    assert guard.policy() == ("default", None)
    policy = tmp_path / "adult.txt"
    policy.write_text("## Risk Categories\n### A. Minors\n  - Anyone who appears under 18.\n")
    monkeypatch.setenv(guard.POLICY_FILE_ENV, str(policy))
    name, text = guard.policy()
    assert name.startswith("custom-") and len(name) == len("custom-") + 12
    assert text.startswith("## Risk Categories")
    policy.write_text("  \n")
    with pytest.raises(ModelUnavailable, match="empty"):
        guard.policy()
    monkeypatch.setenv(guard.POLICY_FILE_ENV, str(tmp_path / "secret-name" / "absent.txt"))
    with pytest.raises(ModelUnavailable) as exc:
        guard.policy()
    assert "secret-name" not in str(exc.value)  # the server echoes this message


@pytest.mark.parametrize(
    ("fired", "wanted", "expected"),
    [(_CRIME, "B", True), (_CRIME, "b", True), (_CRIME, _CRIME.lower(), True), (_CRIME, "B. Real", False),
     (_CRIME, "C", False), ("Minors", "minors", True), (_CRIME, "B.", True)],
)  # fmt: skip
def test_rule_matches(fired: str, wanted: str, expected: bool) -> None:
    assert guard.rule_matches(fired, wanted) is expected


# ---------------------------------------------------------------------------
# Client against the fake server
# ---------------------------------------------------------------------------


def test_check_images_posts_each_image_and_records_results(singguard: type[_SingGuard]) -> None:
    items = [
        ("cat.png", "/d/cat", _png(10)),
        ("knife.png", "/d/knife", _png(20)),
        ("gore.png", "/d/gore", _png(30)),
        ("vague.png", "/d/vague", _png(40)),
        ("broken.png", "/d/broken", _png(_FAIL)),
        ("blurry.png", "/d/blurry", _png(20)),
    ]
    results = [ImageResult(rel_path=rel, abs_path=a, passed=rel != "blurry.png") for rel, a, _ in items]
    seen: list[tuple[int, int]] = []
    assert guard.check_images(results, items, GuardConfig(), progress=lambda d, t: seen.append((d, t)))

    by = {r.rel_path: r.guard for r in results}
    assert by["cat.png"] == GuardResult(
        safe=True, explanation=_SAFE["explanation"], model="SingGuard-4b", policy="default"
    )
    assert by["knife.png"].rules == [_CRIME, _GORE] and not by["knife.png"].safe
    assert by["gore.png"].rules == [_GORE]  # no categories: the rule itself
    assert by["gore.png"].explanation is None
    assert by["vague.png"] is None and by["broken.png"] is None  # unknown, not clean
    assert by["blurry.png"] == by["knife.png"]  # quality rejects are checked too
    assert seen[0] == (0, 6) and seen[-1] == (6, 6)

    assert {s["path"] for s in singguard.seen} == {"/api/guard/check"}
    assert {s["input"] for s in singguard.seen} == {""} and {s["thinking"] for s in singguard.seen} == {"fast"}
    assert all("policy" not in s for s in singguard.seen)  # default rules


def test_policy_file_is_sent_and_recorded(
    singguard: type[_SingGuard], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    policy = tmp_path / "p.txt"
    policy.write_text("### A. Minors\n")
    monkeypatch.setenv(guard.POLICY_FILE_ENV, str(policy))
    monkeypatch.setenv(guard.THINKING_ENV, "fast-slow")
    results = [ImageResult(rel_path="a.png", abs_path="/d/a", passed=True)]
    guard.check_images(results, [("a.png", "/d/a", _png(10))], GuardConfig())
    assert singguard.seen[0]["policy"] == "### A. Minors" and singguard.seen[0]["thinking"] == "fast-slow"
    assert results[0].guard.policy == guard.policy()[0]


def test_large_images_are_downscaled(singguard: type[_SingGuard]) -> None:
    results = [ImageResult(rel_path="big.png", abs_path="/d/big", passed=True)]
    guard.check_images(results, [("big.png", "/d/big", _png(10, (4000, 3000)))], GuardConfig())
    assert singguard.seen[0]["size"] == (1536, 1152)
    assert guard._prepare(small := _png(10)) is small  # small images go as they are


def test_undecodable_image_is_a_failed_check(singguard: type[_SingGuard]) -> None:
    results = [ImageResult(rel_path="a.png", abs_path="/d/a", passed=True)]
    guard.check_images(results, [("a.png", "/d/a", b"not an image")], GuardConfig())
    assert results[0].guard is None and singguard.seen == []


def test_disabled_check_sends_nothing(singguard: type[_SingGuard]) -> None:
    results = [ImageResult(rel_path="a.png", abs_path="/d/a", passed=True)]
    assert not guard.check_images(results, [("a.png", "/d/a", _png(20))], GuardConfig(enabled=False))
    assert singguard.seen == [] and results[0].guard is None


def test_required_check_without_url_fails() -> None:
    results = [ImageResult(rel_path="a.png", abs_path="/d/a", passed=True)]
    with pytest.raises(ModelUnavailable, match=guard.URL_ENV):
        guard.check_images(results, [("a.png", "/d/a", _png(20))], GuardConfig(enabled=True))


def test_scan_runs_singguard_when_configured(singguard: type[_SingGuard]) -> None:
    items = [("a.png", "/d/a", _png(20, (600, 600))), ("b.png", "/d/b", _png(10, (600, 600)))]
    summary = scan_items(items, TargetProfile(), ScanConfig(), FaceConfig())
    assert summary.guard_config.enabled is True
    assert {r.rel_path: r.guard.safe for r in summary.results} == {"a.png": False, "b.png": True}

    opted_out = scan_items(items, TargetProfile(), ScanConfig(), FaceConfig(), guard_cfg=GuardConfig(enabled=False))
    assert opted_out.guard_config.enabled is False
    assert all(r.guard is None for r in opted_out.results)


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


def _img(rel: str, rules: list[str] | None) -> ImageResult:
    verdict = None if rules is None else GuardResult(safe=not rules, rules=rules)
    return ImageResult(rel_path=rel, abs_path=f"/d/{rel}", score=0.9, passed=True, guard=verdict)


_RESULTS = [
    _img("cat.jpg", []),
    _img("knife.jpg", [_CRIME]),
    _img("gore.jpg", [_GORE]),
    _img("unchecked.jpg", None),
]


def test_guard_unsafe_drops_every_unsafe_verdict() -> None:
    selected, why = decide_selection(_RESULTS, ExportRequest(dest="/o", min_score=0.0, guard_unsafe=True), 0.0)
    assert [r.rel_path for r in selected] == ["cat.jpg"]
    assert why["knife.jpg"] == why["gore.jpg"] == "guard-unsafe"
    assert why["unchecked.jpg"] == "guard-unchecked"  # fails closed


def test_guard_rules_drop_only_the_listed_rules() -> None:
    selected, why = decide_selection(_RESULTS, ExportRequest(dest="/o", min_score=0.0, guard_rules=["c"]), 0.0)
    assert {r.rel_path for r in selected} == {"cat.jpg", "knife.jpg"}
    assert why["gore.jpg"] == "guard-rule-c" and why["unchecked.jpg"] == "guard-unchecked"
    _, why = decide_selection(_RESULTS, ExportRequest(dest="/o", min_score=0.0, guard_rules=[_CRIME]), 0.0)
    assert why["knife.jpg"] == "guard-rule-b-real-world-crimes-public-safety"


def test_guard_rules_drop_an_unsafe_verdict_naming_no_rule() -> None:
    nameless = ImageResult(
        rel_path="x.jpg", abs_path="/d/x.jpg", score=0.9, passed=True, guard=GuardResult(safe=False, rules=[])
    )
    selected, why = decide_selection([nameless], ExportRequest(dest="/o", min_score=0.0, guard_rules=["C"]), 0.0)
    assert selected == [] and why["x.jpg"] == "guard-rule-unknown"  # fails closed


def test_malformed_categories_are_ignored_not_fatal() -> None:
    result = guard.to_result({"safe": False, "rule": _GORE, "categories": 5}, "default")
    assert result.rules == [_GORE]


def test_rule_letter() -> None:
    assert guard.rule_letter(_CRIME) == "B" and guard.rule_letter("Minors") == "Minors"


def test_downscale_keeps_alpha_and_orientation() -> None:
    buf = io.BytesIO()
    Image.new("RGBA", (4000, 10), (0, 0, 0, 0)).save(buf, "PNG")
    out = Image.open(io.BytesIO(guard._prepare(buf.getvalue())))
    assert out.format == "PNG" and out.mode == "RGBA"
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees to view
    buf = io.BytesIO()
    Image.new("RGB", (4000, 3000)).save(buf, "JPEG", exif=exif)
    assert Image.open(io.BytesIO(guard._prepare(buf.getvalue()))).size == (1152, 1536)


def test_no_guard_filter_keeps_unchecked_images() -> None:
    selected, _ = decide_selection([_img("a.jpg", None)], ExportRequest(dest="/o", min_score=0.0), 0.0)
    assert [r.rel_path for r in selected] == ["a.jpg"]


def test_empty_rule_names_are_rejected() -> None:
    with pytest.raises(ValidationError):
        ExportRequest(dest="/o", guard_rules=["A", " "])
    assert ExportRequest(dest="/o", guard_rules=[" A "]).guard_rules == ["A"]


def test_filters_on_a_scan_without_singguard_are_a_problem() -> None:
    base = {
        "scan_id": "s", "folder": "/d", "target_profile": TargetProfile().model_dump(),
        "config": ScanConfig().model_dump(), "faces_config": FaceConfig().model_dump(),
        "total": 0, "passed": 0, "rejected": 0, "duplicates": 0, "similar_clusters": 0,
    }  # fmt: skip
    legacy = ScanSummary.model_validate(base)  # cached before SingGuard existed
    for req in (ExportRequest(dest="/o", guard_unsafe=True), ExportRequest(dest="/o", guard_rules=["A"])):
        assert "rescan" in guard_filter_problem(req, legacy)
    ran = ScanSummary.model_validate({**base, "guard_config": {"enabled": True}})
    assert guard_filter_problem(ExportRequest(dest="/o", guard_unsafe=True), ran) is None
    assert guard_filter_problem(ExportRequest(dest="/o"), legacy) is None


def test_report_includes_guard_columns(tmp_path: Path) -> None:
    results = [_img("a.jpg", [_CRIME, _GORE]), _img("b.jpg", []), _img("c.jpg", None)]
    report = tmp_path / "r.csv"
    write_report(results, {}, set(), {}, report)
    rows = {r["rel_path"]: r for r in csv.DictReader(report.open())}
    assert rows["a.jpg"]["guard_safe"] == "false" and rows["a.jpg"]["guard_rules"] == f"{_CRIME};{_GORE}"
    assert rows["b.jpg"]["guard_safe"] == "true" and rows["b.jpg"]["guard_rules"] == ""
    assert rows["c.jpg"]["guard_safe"] == "" and rows["c.jpg"]["guard_rules"] == ""
