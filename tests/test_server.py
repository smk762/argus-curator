"""Server: route contract via FastAPI TestClient (skips if fastapi absent)."""

from __future__ import annotations

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
    assert resp.json()["service"] == "argus-curator"


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
    assert set(resp.json()) == {"torch", "cuda", "clip", "insightface", "onnxruntime"}


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
