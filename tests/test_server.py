"""Server: route contract via FastAPI TestClient (skips if fastapi absent)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from argus_curator.server import create_app  # noqa: E402


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    app = create_app(cache_dir=str(tmp_path / "cache"))
    return TestClient(app)


def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["service"] == "argus-curator"


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


def test_scan_then_paginate_then_export(client: TestClient, dataset: Path, tmp_path: Path) -> None:
    resp = client.post(
        "/scan/folder",
        json={
            "folder": str(dataset),
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

    # Export.
    dest = tmp_path / "out"
    exp = client.post(
        "/export",
        json={"scan_id": scan_id, "dest": str(dest), "mode": "copy", "min_score": 0.0},
    )
    assert exp.status_code == 200
    assert exp.json()["copied"] >= 1
    assert (dest / "manifest.jsonl").exists()


def _parse_sse(body: str) -> list[tuple[str, dict]]:
    """Parse ``event:``/``data:`` SSE frames into (event, payload) tuples."""
    import json

    events = []
    for frame in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in frame.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def test_scan_stream_emits_progress_then_complete(client: TestClient, dataset: Path) -> None:
    resp = client.post(
        "/scan/folder/stream",
        json={"folder": str(dataset), "faces": {"enabled": False}},
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


def test_export_stream_emits_progress_then_complete(client: TestClient, dataset: Path, tmp_path: Path) -> None:
    scan_id = client.post(
        "/scan/folder",
        json={"folder": str(dataset), "faces": {"enabled": False}},
    ).json()["scan_id"]

    dest = tmp_path / "out"
    resp = client.post(
        "/export/stream",
        json={"scan_id": scan_id, "dest": str(dest), "mode": "copy", "min_score": 0.0},
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


def test_thumb_path_traversal_blocked(client: TestClient, dataset: Path) -> None:
    summary = client.post(
        "/scan/folder",
        json={"folder": str(dataset), "faces": {"enabled": False}},
    ).json()
    resp = client.get("/thumb", params={"path": "../../etc/passwd", "scan_id": summary["scan_id"]})
    assert resp.status_code == 400
