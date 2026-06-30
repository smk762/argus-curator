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


def test_unknown_scan_404(client: TestClient) -> None:
    assert client.get("/scan/does-not-exist").status_code == 404


def test_thumb_path_traversal_blocked(client: TestClient, dataset: Path) -> None:
    summary = client.post(
        "/scan/folder",
        json={"folder": str(dataset), "faces": {"enabled": False}},
    ).json()
    resp = client.get("/thumb", params={"path": "../../etc/passwd", "scan_id": summary["scan_id"]})
    assert resp.status_code == 400
