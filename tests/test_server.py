"""Server: route contract via FastAPI TestClient (skips if fastapi absent)."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from argus_curator.server import create_app  # noqa: E402


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    """A rootless client — scan/export/upload must refuse on this one."""
    app = create_app(cache_dir=str(tmp_path / "cache"))
    return TestClient(app)


@pytest.fixture
def contained(dataset: Path, tmp_path: Path) -> tuple[TestClient, Path]:
    """A client with source root = the dataset and a writable export root."""
    exports = tmp_path / "exports"
    exports.mkdir()
    app = create_app(
        cache_dir=str(tmp_path / "cache"),
        source_root=str(dataset),
        export_root=str(exports),
    )
    return TestClient(app), exports


def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["service"] == "argus-curator"
    assert isinstance(body["faces"], bool)


def test_health_declares_manifest_version(client: TestClient) -> None:
    """Issue #11: clients must be able to ask, not sniff which fields turned up.

    The package version moves independently of the manifest contract, so it
    cannot stand in for it — assert they are reported separately.
    """
    from argus_curator.models import MANIFEST_VERSION

    body = client.get("/health").json()
    assert body["manifest_version"] == MANIFEST_VERSION
    assert body["manifest_version"].split(".")[0] == "2"
    assert "version" in body and body["version"] != body["manifest_version"]


def test_folders_browse(dataset: Path, tmp_path: Path) -> None:
    app = create_app(cache_dir=str(tmp_path / "c2"), source_root=str(dataset))
    client = TestClient(app)

    assert client.get("/health").json()["source_root"] == str(dataset.resolve())

    root = client.get("/folders").json()
    names = {f["name"] for f in root["folders"]}
    assert {"personA", "personB"} <= names
    assert root["parent"] is None

    person_a = next(f for f in root["folders"] if f["name"] == "personA")
    assert person_a["rel_path"] == "personA"
    assert person_a["image_count"] >= 2  # recursive across s1/s2
    assert person_a["subfolder_count"] == 2

    sub = client.get("/folders", params={"path": "personA"}).json()
    assert {f["name"] for f in sub["folders"]} == {"s1", "s2"}
    assert sub["parent"] == ""

    assert client.get("/folders", params={"path": "../../etc"}).status_code == 400


def test_folders_requires_source_root(client: TestClient) -> None:
    assert client.get("/folders").status_code == 400


def test_detectors_keys(client: TestClient) -> None:
    resp = client.get("/detectors")
    assert resp.status_code == 200
    assert set(resp.json()) == {
        "torch",
        "cuda",
        "clip",
        "insightface",
        "onnxruntime",
        "fairface",
        "mivolo",
        "places365",
        "garments",
        "nudenet",
        "singguard",
        "singguard_policies",
    }
    assert resp.json()["singguard_policies"] == []  # no SingGuard server configured


def test_detectors_list_policies_when_singguard_is_configured(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ARGUS_CURATOR_SINGGUARD_URL", "http://guard:9150")
    body = client.get("/detectors").json()
    assert body["singguard"] is True and body["singguard_policies"] == [
        "default",
        "adult",
        "adult-granular",
        "adult-cautious",
    ]


def test_guard_policies_serve_builtin_text_and_singguard_default(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from argus_curator import guard

    monkeypatch.delenv("ARGUS_CURATOR_SINGGUARD_URL", raising=False)
    monkeypatch.delenv(guard.POLICY_FILE_ENV, raising=False)
    assert client.get("/health").json()["guard_policy_text"] is True
    body = client.get("/guard/policies").json()
    assert body["singguard"] is False
    by_name = {p["name"]: p for p in body["policies"]}
    assert list(by_name) == list(guard.POLICIES)
    assert by_name["default"] == {
        "name": "default",
        "recorded": "default",
        "source": "singguard",
        "text": None,
        "error": None,
    }
    assert by_name["adult"]["source"] == "builtin" and by_name["adult"]["recorded"] == "adult"
    assert by_name["adult"]["text"] == guard.ADULT_POLICY
    assert by_name["adult-granular"]["text"] == guard.ADULT_GRANULAR_POLICY


def test_guard_policies_serve_the_operator_file(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from argus_curator import guard

    policy_file = tmp_path / "policy.txt"
    policy_file.write_text("### A. Minors\n", encoding="utf-8")
    monkeypatch.setenv(guard.POLICY_FILE_ENV, str(policy_file))
    default = client.get("/guard/policies").json()["policies"][0]
    assert default["source"] == "file" and default["text"] == "### A. Minors"
    assert default["recorded"] == guard.policy_name("### A. Minors")

    policy_file.write_text("  \n", encoding="utf-8")
    default = client.get("/guard/policies").json()["policies"][0]
    assert default["text"] is None and default["recorded"] is None
    assert guard.POLICY_FILE_ENV in default["error"] and str(tmp_path) not in default["error"]


def test_scan_refuses_an_unknown_guard_policy(contained: tuple[TestClient, Path]) -> None:
    client, _exports = contained
    resp = client.post("/scan/folder", json={"folder": "", "guard": {"policy": "anything-goes"}})
    assert resp.status_code == 422


def test_scan_then_paginate_then_export(contained: tuple[TestClient, Path]) -> None:
    client, exports = contained
    resp = client.post(
        "/scan/folder",
        json={
            "folder": "",  # scan the source root itself
            "target_profile": {"target_category": "identity"},
            "config": {"min_short_side": 512},
            "faces": {"enabled": False},
        },
    )
    assert resp.status_code == 200
    summary = resp.json()
    scan_id = summary["scan_id"]
    assert summary["total"] == 5

    # Pagination.
    page = client.get(f"/scan/{scan_id}", params={"offset": 0, "limit": 2}).json()
    assert page["returned"] == 2
    assert len(page["results"]) == 2

    # Thumb for a passing image.
    passing = next(r for r in summary["results"] if r["passed"])
    thumb = client.get("/thumb", params={"path": passing["rel_path"], "scan_id": scan_id})
    assert thumb.status_code == 200
    assert thumb.headers["content-type"] == "image/webp"

    # Export to a destination relative to the export root.
    exp = client.post(
        "/export",
        json={"scan_id": scan_id, "dest": "out", "mode": "copy", "min_score": 0.0},
    )
    assert exp.status_code == 200
    assert exp.json()["copied"] >= 1
    assert (exports / "out" / "manifest.jsonl").exists()


def _parse_sse(body: str) -> list[tuple[str, dict]]:
    """Parse ``event:``/``data:`` SSE frames into (event, payload) tuples."""
    import json

    events = []
    for frame in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in frame.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def test_scan_stream_emits_progress_then_complete(contained: tuple[TestClient, Path]) -> None:
    client, _exports = contained
    resp = client.post(
        "/scan/folder/stream",
        json={"folder": "", "faces": {"enabled": False}},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse(resp.text)
    kinds = [k for k, _ in events]
    assert "progress" in kinds
    assert kinds[-1] == "complete"

    summary = events[-1][1]
    assert summary["total"] == 5
    # The completed scan is persisted and fetchable like a non-streamed one.
    assert client.get(f"/scan/{summary['scan_id']}").status_code == 200

    chunks = [payload for kind, payload in events if kind == "chunk"]
    assert chunks, "expected image batches before the scan finished"
    assert chunks[0]["results"]
    assert chunks[0]["scan_id"] == summary["scan_id"]
    listed = client.get("/scans").json()
    assert listed["active"] is None
    assert any(row["scan_id"] == summary["scan_id"] for row in listed["scans"])


def test_scan_refuses_while_one_is_running(contained: tuple[TestClient, Path]) -> None:
    client, _exports = contained
    gate = client.app.state.scan_gate
    gate.start("busy", "/data/images", ["guard"])
    try:
        refused = client.post("/scan/folder", json={"folder": "", "faces": {"enabled": False}})
        assert refused.status_code == 409
        assert "already running" in refused.json()["detail"]
    finally:
        gate.finish("busy")


def test_export_stream_emits_progress_then_complete(contained: tuple[TestClient, Path]) -> None:
    client, exports = contained
    scan_id = client.post(
        "/scan/folder",
        json={"folder": "", "faces": {"enabled": False}},
    ).json()["scan_id"]

    dest = exports / "out"
    resp = client.post(
        "/export/stream",
        json={"scan_id": scan_id, "dest": "out", "mode": "copy", "min_score": 0.0},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse(resp.text)
    progress = [p for k, p in events if k == "progress"]
    assert progress, "expected at least one progress frame"
    assert all(p["phase"] == "transferring" for p in progress)
    total = progress[0]["total"]
    assert progress[-1]["done"] == total

    kind, result = events[-1]
    assert kind == "complete"
    assert result["copied"] >= 1
    assert result["copied"] + result["skipped"] == total
    assert (dest / "manifest.jsonl").exists()


def test_export_stream_unknown_scan_404(client: TestClient) -> None:
    resp = client.post(
        "/export/stream",
        json={"scan_id": "does-not-exist", "dest": "/tmp/unused"},
    )
    assert resp.status_code == 404


def test_unknown_scan_404(client: TestClient) -> None:
    assert client.get("/scan/does-not-exist").status_code == 404


def test_delete_removes_only_an_under_18_exposed_file(tmp_path: Path) -> None:
    from argus_curator.models import (
        FaceAttributes,
        FaceConfig,
        FaceDetection,
        ImageResult,
        ScanConfig,
        ScanSummary,
        TargetProfile,
    )
    from argus_curator.nsfw import to_result
    from argus_curator.store import ScanStore

    root = tmp_path / "mount"
    root.mkdir()
    minor = root / "minor.png"
    minor_b = root / "minor-b.png"
    adult = root / "adult.png"
    minor.write_bytes(_png_bytes())
    minor_b.write_bytes(_png_bytes())
    adult.write_bytes(_png_bytes())

    def row(name: str, years: float, label: str) -> ImageResult:
        return ImageResult(
            rel_path=name,
            abs_path=str(root / name),
            score=0.9,
            passed=True,
            faces=[
                FaceDetection(
                    bbox=[0, 0, 4, 4],
                    det_score=0.9,
                    primary=True,
                    attributes=FaceAttributes(age_years=years, age_source="mivolo"),
                )
            ],
            primary_face_attributes=FaceAttributes(age_years=years, age_source="mivolo"),
            nsfw=to_result([{"class": label, "score": 0.4, "box": [0, 0, 1, 1]}]),
        )

    cache = tmp_path / "cache"
    ScanStore(cache).save(
        ScanSummary(
            scan_id="scan",
            folder=str(root),
            target_profile=TargetProfile(),
            config=ScanConfig(),
            faces_config=FaceConfig(),
            total=3,
            passed=3,
            rejected=0,
            duplicates=0,
            similar_clusters=0,
            results=[
                row("minor.png", 16.0, "FEMALE_BREAST_EXPOSED"),
                row("minor-b.png", 15.0, "BUTTOCKS_EXPOSED"),
                row("adult.png", 30.0, "FEMALE_BREAST_EXPOSED"),
            ],
        )
    )
    client = TestClient(create_app(cache_dir=str(cache), source_root=str(root)))
    refused = client.post("/scan/scan/delete", json={"rel_paths": ["minor.png", "adult.png"]})
    assert refused.status_code == 403
    assert minor.exists() and adult.exists()
    deleted = client.post("/scan/scan/delete", json={"rel_paths": ["minor.png", "minor-b.png"]})
    assert deleted.status_code == 200
    assert deleted.json()["total"] == 1
    assert not minor.exists() and not minor_b.exists()
    assert adult.exists()
    hidden = client.get("/image", params={"path": "adult.png", "scan_id": "scan"})
    assert hidden.status_code == 200
    gone = client.get("/image", params={"path": "minor.png", "scan_id": "scan"})
    assert gone.status_code == 404


def _png_bytes() -> bytes:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (200, 30, 30)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def upload_setup(tmp_path: Path) -> tuple[TestClient, Path]:
    """A client with a tmp source root, plus that root's path."""
    root = tmp_path / "mount"
    root.mkdir()
    app = create_app(cache_dir=str(tmp_path / "cache"), source_root=str(root))
    return TestClient(app), root


def test_upload_happy_path(upload_setup: tuple[TestClient, Path]) -> None:
    client, root = upload_setup
    resp = client.post(
        "/upload",
        data={"folder": "uploads/session-1"},
        files=[
            ("files", ("a.png", _png_bytes(), "image/png")),
            ("files", ("b.png", _png_bytes(), "image/png")),
        ],
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body == {"folder": "uploads/session-1", "saved": 2, "skipped": [], "errors": []}
    assert (root / "uploads" / "session-1" / "a.png").is_file()
    assert (root / "uploads" / "session-1" / "b.png").is_file()


def test_upload_requires_source_root(client: TestClient) -> None:
    resp = client.post(
        "/upload",
        data={"folder": "uploads"},
        files=[("files", ("a.png", _png_bytes(), "image/png"))],
    )
    assert resp.status_code == 400


def test_upload_folder_traversal_blocked(upload_setup: tuple[TestClient, Path]) -> None:
    client, _root = upload_setup
    resp = client.post(
        "/upload",
        data={"folder": "../escape"},
        files=[("files", ("a.png", _png_bytes(), "image/png"))],
    )
    assert resp.status_code == 400


def test_upload_filename_sanitized_to_basename(upload_setup: tuple[TestClient, Path]) -> None:
    client, root = upload_setup
    resp = client.post(
        "/upload",
        data={"folder": "uploads"},
        files=[("files", ("../../sneaky.png", _png_bytes(), "image/png"))],
    )
    assert resp.status_code == 200
    assert resp.json()["saved"] == 1
    assert (root / "uploads" / "sneaky.png").is_file()
    assert not (root.parent / "sneaky.png").exists()


def test_upload_skips_non_images(upload_setup: tuple[TestClient, Path]) -> None:
    client, root = upload_setup
    resp = client.post(
        "/upload",
        data={"folder": "uploads"},
        files=[
            ("files", ("notes.txt", b"not an image", "text/plain")),
            ("files", ("a.png", _png_bytes(), "image/png")),
        ],
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["saved"] == 1
    assert body["skipped"] == ["notes.txt"]
    assert not (root / "uploads" / "notes.txt").exists()


def test_upload_skips_existing_names(upload_setup: tuple[TestClient, Path]) -> None:
    client, root = upload_setup
    existing = root / "uploads"
    existing.mkdir()
    (existing / "a.png").write_bytes(b"original")

    resp = client.post(
        "/upload",
        data={"folder": "uploads"},
        files=[("files", ("a.png", _png_bytes(), "image/png"))],
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["saved"] == 0
    assert body["skipped"] == ["a.png"]
    assert (existing / "a.png").read_bytes() == b"original"  # not overwritten


def test_thumb_path_traversal_blocked(contained: tuple[TestClient, Path]) -> None:
    client, _exports = contained
    summary = client.post(
        "/scan/folder",
        json={"folder": "", "faces": {"enabled": False}},
    ).json()
    resp = client.get("/thumb", params={"path": "../../etc/passwd", "scan_id": summary["scan_id"]})
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Containment (issue #3): scan/export must not accept arbitrary paths
# ---------------------------------------------------------------------------


def test_scan_requires_source_root(client: TestClient) -> None:
    for endpoint in ("/scan/folder", "/scan/folder/stream"):
        resp = client.post(endpoint, json={"folder": "", "faces": {"enabled": False}})
        assert resp.status_code == 400, endpoint
        assert "no source root configured" in resp.json()["detail"]


def test_scan_folder_traversal_blocked(contained: tuple[TestClient, Path]) -> None:
    client, _exports = contained
    for folder in ("..", "../..", "/home", "/etc"):
        for endpoint in ("/scan/folder", "/scan/folder/stream"):
            resp = client.post(endpoint, json={"folder": folder, "faces": {"enabled": False}})
            assert resp.status_code == 400, (endpoint, folder)
            # Assert the containment message specifically: a bare 400 would also
            # be produced by the is_dir() check, so this entry would still pass
            # if the escape check were dropped or reordered after it.
            assert "escapes the mount root" in resp.json()["detail"], (endpoint, folder)


def test_scan_accepts_absolute_path_inside_root(contained: tuple[TestClient, Path], dataset: Path) -> None:
    """UIs may echo back /folders abs_path values — allowed iff inside the root."""
    client, _exports = contained
    resp = client.post("/scan/folder", json={"folder": str(dataset / "personA"), "faces": {"enabled": False}})
    assert resp.status_code == 200
    assert resp.json()["total"] == 3


def test_export_requires_export_root(dataset: Path, tmp_path: Path) -> None:
    app = create_app(cache_dir=str(tmp_path / "cache"), source_root=str(dataset))
    client = TestClient(app)
    scan_id = client.post("/scan/folder", json={"folder": "", "faces": {"enabled": False}}).json()["scan_id"]
    for endpoint in ("/export", "/export/stream"):
        resp = client.post(endpoint, json={"scan_id": scan_id, "dest": "out", "min_score": 0.0})
        assert resp.status_code == 400, endpoint
        assert "no export root configured" in resp.json()["detail"]


def test_export_dest_traversal_blocked(contained: tuple[TestClient, Path]) -> None:
    client, _exports = contained
    scan_id = client.post("/scan/folder", json={"folder": "", "faces": {"enabled": False}}).json()["scan_id"]
    for dest in ("../escape", "/home/steal"):
        for endpoint in ("/export", "/export/stream"):
            resp = client.post(endpoint, json={"scan_id": scan_id, "dest": dest, "min_score": 0.0})
            assert resp.status_code == 400, (endpoint, dest)
            assert "escapes the mount root" in resp.json()["detail"], (endpoint, dest)


def test_export_move_rejected_by_default(contained: tuple[TestClient, Path]) -> None:
    client, _exports = contained
    scan_id = client.post("/scan/folder", json={"folder": "", "faces": {"enabled": False}}).json()["scan_id"]
    for endpoint in ("/export", "/export/stream"):
        resp = client.post(endpoint, json={"scan_id": scan_id, "dest": "out", "mode": "move", "min_score": 0.0})
        assert resp.status_code == 403, endpoint


def test_export_move_allowed_with_flag(dataset: Path, tmp_path: Path) -> None:
    exports = tmp_path / "exports"
    exports.mkdir()
    app = create_app(
        cache_dir=str(tmp_path / "cache"),
        source_root=str(dataset),
        export_root=str(exports),
        allow_move=True,
    )
    client = TestClient(app)
    assert client.get("/health").json()["allow_move"] is True
    scan_id = client.post("/scan/folder", json={"folder": "", "faces": {"enabled": False}}).json()["scan_id"]
    resp = client.post("/export", json={"scan_id": scan_id, "dest": "out", "mode": "move", "min_score": 0.0})
    assert resp.status_code == 200
    assert resp.json()["copied"] >= 1


def test_health_reports_roots_and_move_gate(contained: tuple[TestClient, Path]) -> None:
    client, exports = contained
    body = client.get("/health").json()
    assert body["export_root"] == str(exports.resolve())
    assert body["allow_move"] is False


# ---------------------------------------------------------------------------
# CORS (issue #3): no wildcard-with-credentials reflection
# ---------------------------------------------------------------------------


def test_cors_defaults_to_localhost_not_wildcard(tmp_path: Path) -> None:
    client = TestClient(create_app(cors=True, cache_dir=str(tmp_path / "cache")))
    ok = client.get("/health", headers={"Origin": "http://localhost:3000"})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:3000"
    evil = client.get("/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in evil.headers


def test_cors_explicit_origins(tmp_path: Path) -> None:
    client = TestClient(create_app(cors_origins=["https://demo.argus.example"], cache_dir=str(tmp_path / "cache")))
    ok = client.get("/health", headers={"Origin": "https://demo.argus.example"})
    assert ok.headers.get("access-control-allow-origin") == "https://demo.argus.example"
    other = client.get("/health", headers={"Origin": "http://localhost:3000"})
    assert "access-control-allow-origin" not in other.headers


def test_cors_wildcard_is_credentialless(tmp_path: Path) -> None:
    client = TestClient(create_app(cors_allow_any=True, cache_dir=str(tmp_path / "cache")))
    resp = client.get("/health", headers={"Origin": "https://anywhere.example"})
    assert resp.headers.get("access-control-allow-origin") == "*"
    assert "access-control-allow-credentials" not in resp.headers


@pytest.mark.parametrize("via", ["arg", "env"])
def test_cors_explicit_star_never_reflects_with_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, via: str
) -> None:
    """A "*" in the allow-list must degrade to the credential-less wildcard.

    Taken literally with allow_credentials=True, Starlette reflects whatever
    Origin it is sent — the exact hole the allow-list exists to close.
    """
    if via == "env":
        monkeypatch.setenv("CURATOR_CORS_ORIGINS", "*")
        app = create_app(cache_dir=str(tmp_path / "cache"))
    else:
        app = create_app(cors_origins=["*"], cache_dir=str(tmp_path / "cache"))
    resp = TestClient(app).get("/health", headers={"Origin": "https://evil.example"})
    assert resp.headers.get("access-control-allow-origin") != "https://evil.example"
    assert resp.headers.get("access-control-allow-origin") == "*"
    assert "access-control-allow-credentials" not in resp.headers


def test_cors_origins_from_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """CURATOR_CORS_ORIGINS is the deployment knob — cover its parsing."""
    monkeypatch.setenv("CURATOR_CORS_ORIGINS", " https://a.example , https://b.example ")
    client = TestClient(create_app(cache_dir=str(tmp_path / "cache")))
    for origin in ("https://a.example", "https://b.example"):
        resp = client.get("/health", headers={"Origin": origin})
        assert resp.headers.get("access-control-allow-origin") == origin
    evil = client.get("/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in evil.headers


# ---------------------------------------------------------------------------
# A cached scan is data, not config: its folder must be re-checked (issue #3
# follow-up). The CLI and pre-containment builds both write to the same store.
# ---------------------------------------------------------------------------


@pytest.fixture
def outside_scan(dataset: Path, tmp_path: Path) -> tuple[TestClient, str, Path]:
    """A client rooted at `dataset`, plus a cached scan of a dir outside it."""
    from argus_curator import scanner
    from argus_curator.store import ScanStore

    outside = tmp_path / "outside"
    outside.mkdir()
    _noise_image_file(outside / "private.png")

    cache = tmp_path / "cache"
    summary = scanner.scan_folder(outside)
    ScanStore(str(cache)).save(summary)

    exports = tmp_path / "exports"
    exports.mkdir()
    app = create_app(cache_dir=str(cache), source_root=str(dataset), export_root=str(exports))
    return TestClient(app), summary.scan_id, outside


def _noise_image_file(path: Path) -> None:
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(9)
    Image.fromarray(rng.integers(0, 255, size=(768, 768, 3), dtype=np.uint8), "RGB").save(path)


def test_thumb_rejects_scan_outside_source_root(outside_scan: tuple[TestClient, str, Path]) -> None:
    client, scan_id, _outside = outside_scan
    resp = client.get("/thumb", params={"path": "private.png", "scan_id": scan_id})
    assert resp.status_code == 400
    assert "outside the source root" in resp.json()["detail"]


def test_export_rejects_scan_outside_source_root(outside_scan: tuple[TestClient, str, Path]) -> None:
    client, scan_id, _outside = outside_scan
    for endpoint in ("/export", "/export/stream"):
        resp = client.post(endpoint, json={"scan_id": scan_id, "dest": "out", "min_score": 0.0})
        assert resp.status_code == 400, endpoint
        assert "outside the source root" in resp.json()["detail"], endpoint


def test_thumb_still_serves_scan_inside_source_root(contained: tuple[TestClient, Path]) -> None:
    """Positive control: the legitimate scan_id path keeps working."""
    client, _exports = contained
    summary = client.post("/scan/folder", json={"folder": "", "faces": {"enabled": False}}).json()
    rel = summary["results"][0]["rel_path"]
    resp = client.get("/thumb", params={"path": rel, "scan_id": summary["scan_id"]})
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/webp"


def test_malformed_path_is_400_not_500(contained: tuple[TestClient, Path]) -> None:
    """A NUL byte makes resolve() raise ValueError — a client error, not a 500."""
    client, _exports = contained
    resp = client.post("/scan/folder", json={"folder": "a\x00b", "faces": {"enabled": False}})
    assert resp.status_code == 400
    assert "invalid path" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Cross-site writes: CORS is not a write boundary. /upload takes multipart —
# a safelisted content type — so a browser sends it with no preflight.
# ---------------------------------------------------------------------------


def _upload(client: TestClient, **kwargs: object) -> object:
    return client.post(
        "/upload",
        data={"folder": "uploads"},
        files=[("files", ("evil.png", _png_bytes(), "image/png"))],
        **kwargs,  # type: ignore[arg-type]
    )


def test_upload_from_evil_origin_is_refused(upload_setup: tuple[TestClient, Path]) -> None:
    """The CSRF repro: a page on evil.example driving /upload with no preflight."""
    client, root = upload_setup
    resp = _upload(client, headers={"Origin": "https://evil.example"})
    assert resp.status_code == 403  # type: ignore[attr-defined]
    assert "cross-site" in resp.json()["detail"]  # type: ignore[attr-defined]
    # The point of the gate: nothing was written.
    assert not (root / "uploads" / "evil.png").exists()


def test_upload_without_origin_still_works(upload_setup: tuple[TestClient, Path]) -> None:
    """Non-browser clients (curl, the CLI) send no Origin and must be unaffected."""
    client, root = upload_setup
    resp = _upload(client)
    assert resp.status_code == 200  # type: ignore[attr-defined]
    assert (root / "uploads" / "evil.png").is_file()


def test_upload_from_allowlisted_origin_works(tmp_path: Path) -> None:
    root = tmp_path / "mount"
    root.mkdir()
    app = create_app(
        cache_dir=str(tmp_path / "cache"),
        source_root=str(root),
        cors_origins=["https://studio.example"],
    )
    client = TestClient(app)
    resp = _upload(client, headers={"Origin": "https://studio.example"})
    assert resp.status_code == 200  # type: ignore[attr-defined]
    assert (root / "uploads" / "evil.png").is_file()


def test_upload_from_same_origin_works(upload_setup: tuple[TestClient, Path]) -> None:
    """A UI proxied onto this host is same-origin and needs no allow-list entry."""
    client, root = upload_setup
    resp = _upload(client, headers={"Origin": "http://testserver"})
    assert resp.status_code == 200  # type: ignore[attr-defined]
    assert (root / "uploads" / "evil.png").is_file()


def test_cors_any_does_not_grant_cross_site_writes(tmp_path: Path) -> None:
    """--cors-any is anonymous READ from anywhere, not a public upload target."""
    root = tmp_path / "mount"
    root.mkdir()
    app = create_app(cache_dir=str(tmp_path / "cache"), source_root=str(root), cors_allow_any=True)
    client = TestClient(app)
    assert client.get("/health", headers={"Origin": "https://any.example"}).status_code == 200
    resp = _upload(client, headers={"Origin": "https://any.example"})
    assert resp.status_code == 403  # type: ignore[attr-defined]
    assert not (root / "uploads" / "evil.png").exists()


def test_cross_site_guard_covers_scan_and_export(contained: tuple[TestClient, Path]) -> None:
    """The guard is middleware, so every write route is covered, not just /upload."""
    client, _exports = contained
    evil = {"Origin": "https://evil.example"}
    for endpoint, body in (
        ("/scan/folder", {"folder": "", "faces": {"enabled": False}}),
        ("/scan/folder/stream", {"folder": "", "faces": {"enabled": False}}),
        ("/export", {"scan_id": "x", "dest": "out"}),
        ("/export/stream", {"scan_id": "x", "dest": "out"}),
        ("/scan/x/select", {}),
        ("/faces", {}),
    ):
        assert client.post(endpoint, json=body, headers=evil).status_code == 403, endpoint


def test_cross_site_read_is_unaffected(contained: tuple[TestClient, Path]) -> None:
    """Only state-changing methods are gated; GETs stay CORS's business."""
    client, _exports = contained
    assert client.get("/health", headers={"Origin": "https://evil.example"}).status_code == 200


def test_refused_write_still_carries_cors_headers(tmp_path: Path) -> None:
    """The guard must stay *inside* CORS, or the 403 is an opaque browser error.

    Registration order is the whole reason WriteGuard is added before
    CORSMiddleware. Assert the observable consequence, not the ordering: without
    the CORS headers the calling page cannot read the JSON detail explaining why
    its upload was refused. The guard now lives in argus-cortex, so this is the
    only thing pinning that contract from inside this repo.
    """
    root = tmp_path / "mount"
    root.mkdir()
    app = create_app(cache_dir=str(tmp_path / "cache"), source_root=str(root), cors_allow_any=True)
    resp = _upload(TestClient(app), headers={"Origin": "https://any.example"})
    assert resp.status_code == 403  # type: ignore[attr-defined]
    assert resp.headers.get("access-control-allow-origin") == "*"  # type: ignore[attr-defined]
    assert "cross-site" in resp.json()["detail"]  # type: ignore[attr-defined]


def test_wildcard_does_not_revoke_writes_from_a_named_origin(tmp_path: Path) -> None:
    """A "*" co-listed with a real origin must not collapse the whole allow-list.

    Naming an origin is the one documented way to grant it cross-site writes, so
    a stray "*" alongside it must degrade the *wildcard* to read-only without
    taking the named origin down with it.
    """
    root = tmp_path / "mount"
    root.mkdir()
    app = create_app(
        cache_dir=str(tmp_path / "cache"),
        source_root=str(root),
        cors_origins=["*", "https://studio.example"],
    )
    client = TestClient(app)
    # The wildcard still grants anonymous reads but no cross-site write...
    assert client.get("/health", headers={"Origin": "https://evil.example"}).status_code == 200
    assert _upload(client, headers={"Origin": "https://evil.example"}).status_code == 403  # type: ignore[attr-defined]
    assert not (root / "uploads" / "evil.png").exists()
    # ...while the origin the operator actually named keeps its writes.
    assert _upload(client, headers={"Origin": "https://studio.example"}).status_code == 200  # type: ignore[attr-defined]
    assert (root / "uploads" / "evil.png").is_file()


# ---------------------------------------------------------------------------
# CURATOR_ALLOW_MOVE is the only gate on the destructive transfer mode, and it
# is parsed by argus_cortex.server.env_flag. Every other move test passes
# allow_move= directly, so without these the env path — the one operators
# actually use — is never executed.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1", True),
        ("true", True),
        ("TRUE", True),
        ("yes", True),
        ("on", True),
        (" 1 ", True),  # surrounding whitespace is stripped (env_file / secret-file shapes)
        ("0", False),
        ("false", False),
        ("", False),
        ("enabled", False),  # unrecognised -> off, with a warning
    ],
)
def test_allow_move_env_parsing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str, expected: bool) -> None:
    monkeypatch.setenv("CURATOR_ALLOW_MOVE", value)
    client = TestClient(create_app(cache_dir=str(tmp_path / "cache")))
    assert client.get("/health").json()["allow_move"] is expected


def test_allow_move_env_defaults_off_when_unset(tmp_path: Path) -> None:
    client = TestClient(create_app(cache_dir=str(tmp_path / "cache")))
    assert client.get("/health").json()["allow_move"] is False


def test_allow_move_argument_overrides_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit allow_move= wins; the env var is only the default."""
    monkeypatch.setenv("CURATOR_ALLOW_MOVE", "1")
    client = TestClient(create_app(cache_dir=str(tmp_path / "cache"), allow_move=False))
    assert client.get("/health").json()["allow_move"] is False


def test_scan_refuses_models_it_does_not_have(contained: tuple[TestClient, Path]) -> None:
    """Fail before scoring the whole folder, and without naming server paths."""
    client, _exports = contained
    cases = (
        ("faces", "fairface", "FairFace", "ARGUS_CURATOR_FAIRFACE_MODEL"),
        ("faces", "mivolo", "MiVOLO", "ARGUS_CURATOR_MIVOLO_MODEL"),
        ("scenes", "enabled", "Places365", "ARGUS_CURATOR_PLACES365_MODEL"),
        ("garments", "enabled", "Fashionpedia", "ARGUS_CURATOR_GARMENTS_MODEL"),
    )
    for section, flag, name, env in cases:
        for endpoint in ("/scan/folder", "/scan/folder/stream"):
            resp = client.post(endpoint, json={"folder": "", section: {flag: True}})
            assert resp.status_code == 400, (endpoint, flag)
            detail = resp.json()["detail"]
            assert name in detail and env in detail
            assert "/" not in detail.split("(")[0]


def test_select_filters_without_writing_the_export_root(contained: tuple[TestClient, Path]) -> None:
    """A filter changes kept and keep_reason, and never touches the export root."""
    client, exports = contained
    before = sorted(p.name for p in exports.iterdir())
    scan = client.post("/scan/folder", json={"folder": "", "faces": {"enabled": False}}).json()
    scan_id = scan["scan_id"]

    loose = client.post(
        f"/scan/{scan_id}/select",
        json={"min_score": 0.0, "include_rejected": True, "keep_similar": True},
    )
    assert loose.status_code == 200
    loose_body = loose.json()
    assert loose_body["total"] == scan["total"]
    assert loose_body["kept"] == loose_body["total"]
    assert all(row["keep_reason"] == "" for row in loose_body["rows"])
    assert {row["rel_path"] for row in loose_body["rows"]} == {r["rel_path"] for r in scan["results"]}

    tight = client.post(f"/scan/{scan_id}/select", json={"min_score": 0.99})
    assert tight.status_code == 200
    tight_body = tight.json()
    assert tight_body["kept"] < loose_body["kept"]
    assert any(row["keep_reason"] == "below-score" for row in tight_body["rows"])
    assert all(row["abs_path"] for row in tight_body["rows"])

    assert sorted(p.name for p in exports.iterdir()) == before
    assert client.post("/scan/missing/select", json={}).status_code == 404


def test_select_refuses_scan_outside_source_root(outside_scan: tuple[TestClient, str, Path]) -> None:
    client, scan_id, _outside = outside_scan
    resp = client.post(f"/scan/{scan_id}/select", json={"min_score": 0.0})
    assert resp.status_code == 400
    assert "outside the source root" in resp.json()["detail"]


def test_select_refuses_garment_floor_below_what_the_scan_recorded(contained: tuple[TestClient, Path]) -> None:
    client, exports = contained
    scan = client.post("/scan/folder", json={"folder": "", "faces": {"enabled": False}}).json()
    resp = client.post(
        f"/scan/{scan['scan_id']}/select",
        json={"garments": ["dress"], "min_garment_conf": 0.2},
    )
    assert resp.status_code == 400
    assert "rescan" in resp.json()["detail"]
    assert list(exports.iterdir()) == []


def test_select_refuses_moderation_filters_on_unchecked_scan(contained: tuple[TestClient, Path]) -> None:
    client, _exports = contained
    scan = client.post("/scan/folder", json={"folder": "", "faces": {"enabled": False}}).json()
    for body, needs in (
        ({"nsfw_threshold": 0.5}, "NudeNet"),
        ({"nsfw_label_thresholds": {"FEMALE_BREAST_EXPOSED": 0.5}}, "NudeNet"),
        ({"guard_unsafe": True}, "SingGuard"),
        ({"guard_rules": ["A"]}, "SingGuard"),
    ):
        resp = client.post(f"/scan/{scan['scan_id']}/select", json=body)
        assert resp.status_code == 400, body
        assert needs in resp.json()["detail"]


def test_select_validates_like_export(contained: tuple[TestClient, Path]) -> None:
    """ExportRequest's validators reject a bad select body as 422, not a 500."""
    client, _exports = contained
    scan = client.post("/scan/folder", json={"folder": "", "faces": {"enabled": False}}).json()
    for body in (
        {"min_age_years": 40, "max_age_years": 20},
        {"scenes": ["beech"]},
        {"nsfw_label_thresholds": {"FEMALE_BREAST_EXPOSED": 0.1}},
        {"guard_rules": [" "]},
        {"nsfw_exposed_labels": []},
        {"guard_policies": []},
        {"guard_policies": ["nsfw"]},
    ):
        assert client.post(f"/scan/{scan['scan_id']}/select", json=body).status_code == 422, body


def test_select_request_has_every_export_filter() -> None:
    """A new ExportRequest filter must be added to SelectRequest too."""
    from argus_curator.models import ExportRequest, SelectRequest

    not_filters = {
        "scan_id",
        "selection",
        "dest",
        "mode",
        "preserve_structure",
        "write_manifest",
        "caption_url",
        "write_xmp",
        "face_names",
    }
    trims = {"max_keep", "diversify_scenes", "diversify_outfits"}
    assert set(SelectRequest.model_fields) == set(ExportRequest.model_fields) - not_filters - trims


def test_image_serves_original_bytes(contained: tuple[TestClient, Path]) -> None:
    client, _exports = contained
    summary = client.post("/scan/folder", json={"folder": "", "faces": {"enabled": False}}).json()
    rel = summary["results"][0]["rel_path"]
    resp = client.get("/image", params={"path": rel, "scan_id": summary["scan_id"]})
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert resp.content == Path(summary["results"][0]["abs_path"]).read_bytes()


def test_image_path_traversal_blocked(contained: tuple[TestClient, Path]) -> None:
    client, _exports = contained
    summary = client.post("/scan/folder", json={"folder": "", "faces": {"enabled": False}}).json()
    resp = client.get("/image", params={"path": "../../etc/passwd", "scan_id": summary["scan_id"]})
    assert resp.status_code == 400


def test_image_rejects_scan_outside_source_root(outside_scan: tuple[TestClient, str, Path]) -> None:
    client, scan_id, _outside = outside_scan
    resp = client.get("/image", params={"path": "private.png", "scan_id": scan_id})
    assert resp.status_code == 400
    assert "outside the source root" in resp.json()["detail"]


def test_export_refuses_garment_floor_below_what_the_scan_recorded(contained: tuple[TestClient, Path]) -> None:
    client, _exports = contained
    scan = client.post("/scan/folder", json={"folder": "", "faces": {"enabled": False}}).json()
    body = {"scan_id": scan["scan_id"], "dest": "out", "garments": ["dress"], "min_garment_conf": 0.2}
    for endpoint in ("/export", "/export/stream"):
        resp = client.post(endpoint, json=body)
        assert resp.status_code == 400, endpoint
        assert "rescan" in resp.json()["detail"]


def test_scan_requiring_nudenet_without_url_is_refused(contained: tuple[TestClient, Path]) -> None:
    client, _exports = contained
    for endpoint in ("/scan/folder", "/scan/folder/stream"):
        resp = client.post(endpoint, json={"folder": "", "nsfw": {"enabled": True}})
        assert resp.status_code == 400, endpoint
        assert "ARGUS_CURATOR_NUDENET_URL is not set" in resp.json()["detail"]


def test_malformed_nudenet_url_is_refused_without_echoing_it(
    contained: tuple[TestClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _exports = contained
    monkeypatch.setenv("ARGUS_CURATOR_NUDENET_URL", "ftp://user:secret@nudenet")
    resp = client.post("/scan/folder", json={"folder": ""})  # opt-out default: the check is on
    assert resp.status_code == 400
    assert "not a valid http(s) URL" in resp.json()["detail"]
    assert "secret" not in resp.json()["detail"]


def test_export_with_nsfw_thresholds_on_unchecked_scan_is_refused(contained: tuple[TestClient, Path]) -> None:
    client, _exports = contained
    scan = client.post("/scan/folder", json={"folder": "", "faces": {"enabled": False}}).json()
    assert scan["nsfw_config"]["enabled"] is False
    body = {"scan_id": scan["scan_id"], "dest": "out", "nsfw_threshold": 0.5}
    for endpoint in ("/export", "/export/stream"):
        resp = client.post(endpoint, json=body)
        assert resp.status_code == 400, endpoint
        assert "NudeNet" in resp.json()["detail"]


def test_export_with_content_ratings_on_unchecked_scan_is_refused(contained: tuple[TestClient, Path]) -> None:
    client, _exports = contained
    scan = client.post("/scan/folder", json={"folder": "", "faces": {"enabled": False}}).json()
    body = {"scan_id": scan["scan_id"], "dest": "out", "content_ratings": ["safe"]}
    for endpoint in ("/export", "/export/stream"):
        resp = client.post(endpoint, json=body)
        assert resp.status_code == 400, endpoint
        assert "NudeNet" in resp.json()["detail"]


def test_export_write_xmp_stays_under_the_export_root(
    contained: tuple[TestClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    from argus_cortex.xmp import XmpData, sidecar_path

    monkeypatch.setattr("argus_curator.export.xmp_data", lambda *_a, **_k: XmpData(content_rating="safe"))
    client, exports = contained
    scan = client.post("/scan/folder", json={"folder": "", "faces": {"enabled": False}}).json()
    resp = client.post(
        "/export",
        json={"scan_id": scan["scan_id"], "dest": "out", "mode": "copy", "min_score": 0.0, "write_xmp": True},
    )
    assert resp.status_code == 200
    assert resp.json()["xmp_written"] >= 1
    exported = resp.json()["exported_abs_paths"]
    for abs_path in exported.values():
        sidecar = sidecar_path(Path(abs_path))
        assert sidecar.is_file()
        assert exports.resolve() in sidecar.resolve().parents


def test_scan_requiring_singguard_without_url_is_refused(contained: tuple[TestClient, Path]) -> None:
    client, _exports = contained
    for endpoint in ("/scan/folder", "/scan/folder/stream"):
        resp = client.post(endpoint, json={"folder": "", "guard": {"enabled": True}})
        assert resp.status_code == 400, endpoint
        assert "ARGUS_CURATOR_SINGGUARD_URL is not set" in resp.json()["detail"]


def test_unreadable_singguard_policy_is_refused_without_echoing_its_path(
    contained: tuple[TestClient, Path], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client, _exports = contained
    monkeypatch.setenv("ARGUS_CURATOR_SINGGUARD_URL", "http://guard:9150")
    monkeypatch.setenv("ARGUS_CURATOR_SINGGUARD_POLICY_FILE", str(tmp_path / "private-dir" / "policy.txt"))
    resp = client.post("/scan/folder", json={"folder": ""})  # opt-out default: the check is on
    assert resp.status_code == 400
    assert "ARGUS_CURATOR_SINGGUARD_POLICY_FILE cannot be read" in resp.json()["detail"]
    assert "private-dir" not in resp.json()["detail"]


def test_export_with_guard_filters_on_unchecked_scan_is_refused(contained: tuple[TestClient, Path]) -> None:
    client, _exports = contained
    scan = client.post("/scan/folder", json={"folder": "", "faces": {"enabled": False}}).json()
    assert scan["guard_config"]["enabled"] is False
    for body in ({"guard_unsafe": True}, {"guard_rules": ["A"]}):
        resp = client.post("/export", json={"scan_id": scan["scan_id"], "dest": "out", **body})
        assert resp.status_code == 400, body
        assert "SingGuard" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Stopping a scan, and purging the cache
# ---------------------------------------------------------------------------


def _summary(scan_id: str, folder: str, total: int):
    from argus_curator.models import FaceConfig, ScanConfig, ScanSummary, TargetProfile

    return ScanSummary(
        scan_id=scan_id,
        folder=folder,
        target_profile=TargetProfile(),
        config=ScanConfig(),
        faces_config=FaceConfig(),
        total=total,
        passed=0,
        rejected=0,
        duplicates=0,
        similar_clusters=0,
    )


def _slow_scan(monkeypatch: pytest.MonkeyPatch, started: threading.Event) -> None:
    """Stand in for the scanner: report progress until asked to stop."""
    import time

    from argus_curator.runtime import ScanCancelled
    from argus_curator.server import app as app_mod

    def fake_scan_folder(folder, profile, cfg, faces_cfg, *, scan_id, progress, cancelled, save=None, **_):
        for i in range(200):
            progress({"phase": "guard", "done": i, "total": 200, "pending_checks": ["guard"]})
            if save is not None:
                save(_summary(scan_id, str(folder), total=200))
            started.set()
            if cancelled():
                raise ScanCancelled(f"scan {scan_id} was cancelled")
            time.sleep(0.02)
        raise AssertionError("the scan was never stopped")

    monkeypatch.setattr(app_mod.scanner, "scan_folder", fake_scan_folder)


def test_cancel_stops_the_running_scan_and_drops_its_snapshot(
    contained: tuple[TestClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _exports = contained
    started = threading.Event()
    _slow_scan(monkeypatch, started)

    assert client.post("/scan/nothing/cancel").status_code == 404

    events: list = []

    def run() -> None:
        resp = client.post("/scan/folder/stream", json={"folder": "", "faces": {"enabled": False}})
        events.extend(_parse_sse(resp.text))

    worker = threading.Thread(target=run)
    worker.start()
    assert started.wait(5)
    active = client.get("/scans").json()["active"]
    assert active and active["cancelling"] is False
    scan_id = active["scan_id"]
    assert client.post("/scan/some-other-id/cancel").status_code == 404

    resp = client.post(f"/scan/{scan_id}/cancel")
    assert resp.status_code == 200 and resp.json() == {"scan_id": scan_id, "cancelling": True}
    worker.join(5)
    assert not worker.is_alive()
    assert [k for k, _ in events][-1] == "cancelled" and events[-1][1] == {"scan_id": scan_id}
    listed = client.get("/scans").json()
    assert listed["active"] is None
    assert all(row["scan_id"] != scan_id for row in listed["scans"]), "a cancelled scan is not a result"
    assert client.get(f"/scan/{scan_id}").status_code == 404
    # A second cancel of a scan that is over is a 404 too, not an error.
    assert client.post(f"/scan/{scan_id}/cancel").status_code == 404


def test_purge_stops_the_scan_and_empties_the_cache(
    contained: tuple[TestClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _exports = contained
    store = client.app.state.scan_store
    for i in range(3):
        store.save(_summary(f"old{i}", "/data/images", total=1))
    assert len(client.get("/scans").json()["scans"]) == 3

    # With nothing running, a purge just empties the cache.
    resp = client.post("/scans/purge")
    assert resp.status_code == 200 and resp.json() == {"cancelled": None, "deleted": 3}
    assert client.get("/scans").json() == {"active": None, "scans": []}

    started = threading.Event()
    _slow_scan(monkeypatch, started)
    events: list = []
    worker = threading.Thread(
        target=lambda: events.extend(
            _parse_sse(client.post("/scan/folder/stream", json={"folder": "", "faces": {"enabled": False}}).text)
        )
    )
    worker.start()
    assert started.wait(5)
    scan_id = client.get("/scans").json()["active"]["scan_id"]

    resp = client.post("/scans/purge")
    worker.join(5)
    assert resp.status_code == 200
    body = resp.json()
    assert body["cancelled"] == scan_id and body["deleted"] >= 0
    assert [k for k, _ in events][-1] == "cancelled"
    assert client.get("/scans").json() == {"active": None, "scans": []}


def test_health_says_it_can_stop_and_purge(client: TestClient) -> None:
    assert client.get("/health").json()["scan_control"] is True


def test_cancel_and_purge_are_writes(contained: tuple[TestClient, Path]) -> None:
    """A cross-site page cannot stop or purge."""
    client, _exports = contained
    evil = {"Origin": "https://evil.example"}
    assert client.post("/scans/purge", headers=evil).status_code == 403
    assert client.post("/scan/x/cancel", headers=evil).status_code == 403


def _face(years: float | None, source: str | None = "mivolo"):
    from argus_curator.models import FaceAttributes, FaceDetection

    attrs = None if years is None else FaceAttributes(age_years=years, age_source=source)
    return FaceDetection(bbox=[10.0, 20.0, 30.0, 40.0], det_score=0.9, attributes=attrs)


def _stub_detect(monkeypatch: pytest.MonkeyPatch, faces: list) -> None:
    def detect(results, items, cfg, **kwargs) -> None:  # noqa: ANN001, ANN003
        results[0].faces.extend(faces)

    monkeypatch.setattr("argus_curator.server.app.faces_available", lambda: True)
    monkeypatch.setattr("argus_curator.server.app.mivolo_available", lambda: True)
    monkeypatch.setattr("argus_curator.faces.detect_faces", detect)


def test_faces_returns_estimates_and_writes_nothing(
    dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exports = tmp_path / "exports"
    exports.mkdir()
    app = create_app(cache_dir=str(tmp_path / "cache"), source_root=str(dataset), export_root=str(exports))
    client = TestClient(app)
    before = {p.relative_to(dataset) for p in dataset.rglob("*")}
    _stub_detect(monkeypatch, [_face(34.2)])
    resp = client.post("/faces", content=_png_bytes(), headers={"content-type": "image/png"})
    assert resp.status_code == 200
    assert resp.json() == {
        "source": "mivolo",
        "faces": [{"age_estimate_years": 34.2, "bbox": [10.0, 20.0, 30.0, 40.0]}],
    }
    assert {p.relative_to(dataset) for p in dataset.rglob("*")} == before
    assert list(exports.iterdir()) == []


def test_faces_empty_when_there_is_no_face(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_detect(monkeypatch, [])
    resp = client.post("/faces", content=_png_bytes())
    assert resp.status_code == 200
    assert resp.json() == {"source": "mivolo", "faces": []}


def test_faces_503_when_mivolo_is_down(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("argus_curator.server.app.faces_available", lambda: True)
    monkeypatch.setattr("argus_curator.server.app.mivolo_available", lambda: False)
    resp = client.post("/faces", content=_png_bytes())
    assert resp.status_code == 503
    assert "MiVOLO" in resp.json()["detail"]


def test_faces_503_when_a_face_has_no_estimate(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_detect(monkeypatch, [_face(None)])
    resp = client.post("/faces", content=_png_bytes())
    assert resp.status_code == 503
    assert "no age" in resp.json()["detail"]


def test_faces_400_when_bytes_are_not_an_image(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_detect(monkeypatch, [])
    resp = client.post("/faces", content=b"not-an-image")
    assert resp.status_code == 400
    assert "undecodable" in resp.json()["detail"]


def test_faces_accepts_multipart(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_detect(monkeypatch, [])
    resp = client.post("/faces", files={"file": ("x.png", _png_bytes(), "image/png")})
    assert resp.status_code == 200
    assert resp.json()["faces"] == []


def test_faces_without_origin_works(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Server-to-server (darkroom) sends no Origin."""
    _stub_detect(monkeypatch, [])
    assert client.post("/faces", content=_png_bytes()).status_code == 200


def test_faces_503_when_the_detector_fails(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """A detector error must not come back as 200 with no faces (fail-open age check)."""

    def detect(results, items, cfg, **kwargs) -> None:  # noqa: ANN001, ANN003
        assert kwargs.get("strict") is True
        raise RuntimeError("onnx exploded")

    monkeypatch.setattr("argus_curator.server.app.faces_available", lambda: True)
    monkeypatch.setattr("argus_curator.server.app.mivolo_available", lambda: True)
    monkeypatch.setattr("argus_curator.faces.detect_faces", detect)
    resp = client.post("/faces", content=_png_bytes())
    assert resp.status_code == 503
    assert "face detection failed" in resp.json()["detail"]


def test_faces_413_when_the_body_is_too_large(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_detect(monkeypatch, [])
    monkeypatch.setattr("argus_curator.server.app._FACES_MAX_BYTES", 16)
    resp = client.post("/faces", content=_png_bytes())
    assert resp.status_code == 413


def test_faces_503_with_retry_after_when_busy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Past the in-flight limit a call is refused at once, not queued: each one buffers an image."""
    entered, release = threading.Event(), threading.Event()

    def detect(results, items, cfg, **kwargs) -> None:  # noqa: ANN001, ANN003
        entered.set()
        assert release.wait(5)

    monkeypatch.setattr("argus_curator.server.app._FACES_CONCURRENCY", 1)
    monkeypatch.setattr("argus_curator.server.app.faces_available", lambda: True)
    monkeypatch.setattr("argus_curator.server.app.mivolo_available", lambda: True)
    monkeypatch.setattr("argus_curator.faces.detect_faces", detect)
    with TestClient(create_app(cache_dir=str(tmp_path / "cache"))) as client:
        first: list = []
        worker = threading.Thread(target=lambda: first.append(client.post("/faces", content=_png_bytes())))
        worker.start()
        assert entered.wait(5)
        busy = client.post("/faces", content=_png_bytes())
        release.set()
        worker.join(5)
    assert busy.status_code == 503 and busy.headers["retry-after"] == "1"
    assert first[0].status_code == 200


def test_faces_third_call_503_at_default_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Default cap is two: a third call is 503 while both slots are taken."""
    in_detect = threading.Semaphore(0)
    release = threading.Event()

    def detect(results, items, cfg, **kwargs) -> None:  # noqa: ANN001, ANN003
        in_detect.release()
        assert release.wait(5)

    monkeypatch.setattr("argus_curator.server.app.faces_available", lambda: True)
    monkeypatch.setattr("argus_curator.server.app.mivolo_available", lambda: True)
    monkeypatch.setattr("argus_curator.faces.detect_faces", detect)
    with TestClient(create_app(cache_dir=str(tmp_path / "cache"))) as client:
        held: list = []

        def hold() -> None:
            held.append(client.post("/faces", content=_png_bytes()))

        workers = [threading.Thread(target=hold) for _ in range(2)]
        for t in workers:
            t.start()
        assert in_detect.acquire(timeout=5)
        assert in_detect.acquire(timeout=5)
        busy = client.post("/faces", content=_png_bytes())
        release.set()
        for t in workers:
            t.join(5)
    assert busy.status_code == 503 and busy.headers["retry-after"] == "1"
    assert [r.status_code for r in held] == [200, 200]


async def test_faces_burst_does_not_queue_past_a_free_slot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When one slot is free, extra concurrent callers get 503 — they must not wait on acquire()."""
    import asyncio
    import time

    httpx = pytest.importorskip("httpx")

    in_detect = 0
    count_lock = threading.Lock()
    first_entered = threading.Event()
    release = threading.Event()

    def detect(results, items, cfg, **kwargs) -> None:  # noqa: ANN001, ANN003
        nonlocal in_detect
        with count_lock:
            in_detect += 1
            n = in_detect
        if n == 1:
            first_entered.set()
        assert release.wait(5)

    monkeypatch.setattr("argus_curator.server.app.faces_available", lambda: True)
    monkeypatch.setattr("argus_curator.server.app.mivolo_available", lambda: True)
    monkeypatch.setattr("argus_curator.faces.detect_faces", detect)
    app = create_app(cache_dir=str(tmp_path / "cache"))
    transport = httpx.ASGITransport(app=app)
    png = _png_bytes()
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        holder = asyncio.create_task(ac.post("/faces", content=png))
        assert await asyncio.to_thread(first_entered.wait, 5)
        racers = [asyncio.create_task(ac.post("/faces", content=png)) for _ in range(8)]
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if sum(t.done() for t in racers) >= 7:
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("expected 7 refusals while one slot stayed in flight")
        with count_lock:
            running = in_detect
        assert running == 2
        codes = [t.result().status_code for t in racers if t.done()]
        assert codes.count(503) == 7
        release.set()
        rest = await asyncio.gather(holder, *racers)
    assert [r.status_code for r in rest].count(200) == 2
    assert [r.status_code for r in rest].count(503) == 7


def test_faces_503_with_retry_after_when_the_gpu_is_busy(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """No GPU slot within the wait is "try again", never "no faces"."""
    from argus_cortex.gpu import GpuLeaseTimeout

    waits: list[float | None] = []

    def detect(results, items, cfg, **kwargs) -> None:  # noqa: ANN001, ANN003
        waits.append(kwargs.get("lease_wait_s"))
        raise GpuLeaseTimeout("GPU 0 had 900 MB free, argus-curator:faces:run needs 512 MB")

    monkeypatch.setattr("argus_curator.server.app.faces_available", lambda: True)
    monkeypatch.setattr("argus_curator.server.app.mivolo_available", lambda: True)
    monkeypatch.setattr("argus_curator.faces.detect_faces", detect)
    resp = client.post("/faces", content=_png_bytes())
    assert resp.status_code == 503 and resp.headers["retry-after"] == "5"
    assert resp.json()["detail"].startswith("GPU busy:")
    assert waits == [30.0]


def test_health_reports_the_gpu_coordinator(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from argus_cortex.gpu import NullCoordinator

    monkeypatch.setattr("argus_curator.runtime._coordinator", NullCoordinator())
    assert client.get("/health").json()["gpu_coordinator"] == "none"


def test_health_reports_lease_when_configured(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("argus_curator.runtime._coordinator", None)
    monkeypatch.setenv("ARGUS_GPU_COORDINATOR", "lease")
    monkeypatch.setenv("ARGUS_GPU_LEASE_PATH", str(tmp_path / "gpu.lock"))
    assert client.get("/health").json()["gpu_coordinator"] == "lease"


def test_scan_503_when_the_gpu_is_busy(contained: tuple[TestClient, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    from argus_cortex.gpu import GpuLeaseTimeout

    from argus_curator.server import app as app_mod

    def boom(*_a, **_k):  # noqa: ANN002, ANN003
        raise GpuLeaseTimeout("busy")

    monkeypatch.setattr(app_mod.scanner, "scan_folder", boom)
    client, _exports = contained
    resp = client.post("/scan/folder", json={"folder": "", "faces": {"enabled": False}})
    assert resp.status_code == 503 and resp.headers["retry-after"] == "5"
    assert resp.json()["detail"].startswith("GPU busy:")


def test_scan_stream_drops_a_partial_cache_when_the_gpu_is_busy(
    contained: tuple[TestClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    from argus_cortex.gpu import GpuLeaseTimeout

    from argus_curator.server import app as app_mod

    def fake_scan_folder(folder, profile, cfg, faces_cfg, *, scan_id, progress, cancelled, save=None, **_):  # noqa: ANN001, ANN003
        if save is not None:
            save(_summary(scan_id, str(folder), total=3))
        raise GpuLeaseTimeout("busy")

    monkeypatch.setattr(app_mod.scanner, "scan_folder", fake_scan_folder)
    client, _exports = contained
    resp = client.post("/scan/folder/stream", json={"folder": "", "faces": {"enabled": False}})
    events = _parse_sse(resp.text)
    assert events[-1][0] == "error"
    assert events[-1][1]["detail"].startswith("GPU busy:")
    assert events[-1][1]["retry_after"] == 5
    assert client.get("/scans").json()["scans"] == []


def test_faces_body_query_asks_mivolo_for_bodies(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from argus_curator.models import FaceAttributes, FaceDetection

    asked: list[bool] = []

    def detect(results, items, cfg, **kwargs) -> None:  # noqa: ANN001, ANN003
        asked.append(cfg.mivolo_body)
        attrs = FaceAttributes(age_years=34.2, age_source="mivolo", age_body=cfg.mivolo_body)
        results[0].faces.append(FaceDetection(bbox=[10.0, 20.0, 30.0, 40.0], det_score=0.9, attributes=attrs))

    monkeypatch.setattr("argus_curator.server.app.faces_available", lambda: True)
    monkeypatch.setattr("argus_curator.server.app.mivolo_available", lambda: True)
    monkeypatch.setattr("argus_curator.faces.detect_faces", detect)
    plain = client.post("/faces", content=_png_bytes()).json()["faces"][0]
    with_body = client.post("/faces?body=true", content=_png_bytes()).json()["faces"][0]
    assert asked == [False, True]
    assert "age_body" not in plain  # unchanged for callers that do not ask
    assert with_body["age_body"] is True and with_body["age_estimate_years"] == 34.2
