"""Web session helpers and upload API (no GPU)."""

from __future__ import annotations

import io
import zipfile

import pytest
import numpy as np
from PIL import Image
from starlette.testclient import TestClient

from argus_curator.server import create_app
from argus_curator.web_sessions import (
    SUPPORTED_IMAGE_UPLOAD_EXT,
    collect_zip_members,
    sanitize_relative_path,
)


def _noisy_jpeg() -> bytes:
    arr = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def test_collect_zip_members(tmp_path) -> None:
    root = tmp_path / "in"
    root.mkdir()
    (root / "a").mkdir()
    p = root / "a" / "x.jpg"
    p.write_bytes(b"x")
    pairs = collect_zip_members(root, ["a/x.jpg", "missing.jpg"])
    assert len(pairs) == 1
    assert pairs[0][1] == "a/x.jpg"


class TestSanitize:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("a/b.jpg", "a/b.jpg"),
            ("./x/./y.png", "x/y.png"),
            ("../evil.jpg", None),
            ("ok/../x.jpg", "x.jpg"),
            ("", None),
        ],
    )
    def test_sanitize_relative_path(self, raw: str, expected: str | None) -> None:
        assert sanitize_relative_path(raw) == expected

    def test_supported_exts(self) -> None:
        assert ".jpg" in SUPPORTED_IMAGE_UPLOAD_EXT


class TestSessionAPI:
    @pytest.fixture
    def client(self) -> TestClient:
        return TestClient(create_app(cors=True))

    def test_session_upload_scan_export_delete(self, client: TestClient) -> None:
        r = client.post("/sessions")
        assert r.status_code == 200
        sid = r.json()["session_id"]

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("nested/a.jpg", _noisy_jpeg())
            zf.writestr("b.jpg", _noisy_jpeg())
        buf.seek(0)

        up = client.post(
            f"/sessions/{sid}/upload-zip",
            files={"archive": ("set.zip", buf.getvalue(), "application/zip")},
        )
        assert up.status_code == 200, up.text
        assert up.json()["saved"] == 2

        scan_body = {
            "objective": "identity",
            "target_style": "photo",
            "apply_preset": False,
            "embeddings": {"use_clip": False, "use_dino": False},
            "detectors": {"use_yolo": False, "use_mtcnn": False},
        }
        sc = client.post(f"/sessions/{sid}/scan", json=scan_body)
        assert sc.status_code == 200, sc.text
        data = sc.json()
        assert data["session_id"] == sid
        assert data["total"] == 2
        names = data["selected_names"]
        assert len(names) >= 1

        zp = client.post(f"/sessions/{sid}/export-zip", json={"selected_names": names})
        assert zp.status_code == 200
        assert zp.headers.get("content-type", "").startswith("application/zip")
        zread = zipfile.ZipFile(io.BytesIO(zp.content))
        assert len(zread.namelist()) == len(names)

        dl = client.delete(f"/sessions/{sid}")
        assert dl.status_code == 200

        sc2 = client.post(f"/sessions/{sid}/scan", json=scan_body)
        assert sc2.status_code == 404


def test_cors_enabled_via_env_star_origin(monkeypatch) -> None:
    monkeypatch.setenv("ARGUS_CURATOR_CORS", "1")
    monkeypatch.delenv("ARGUS_CURATOR_CORS_ORIGINS", raising=False)
    app = create_app(cors=None)
    client = TestClient(app)
    r = client.get("/", headers={"Origin": "https://demo.example"})
    assert r.headers.get("access-control-allow-origin") == "*"


def test_cors_enabled_via_env_specific_origin(monkeypatch) -> None:
    monkeypatch.setenv("ARGUS_CURATOR_CORS", "true")
    monkeypatch.setenv("ARGUS_CURATOR_CORS_ORIGINS", "https://demo.example,http://localhost:3000")
    app = create_app(cors=None)
    client = TestClient(app)
    r = client.get("/", headers={"Origin": "https://demo.example"})
    assert r.headers.get("access-control-allow-origin") == "https://demo.example"


def test_cors_disabled_explicit_overrides_env(monkeypatch) -> None:
    monkeypatch.setenv("ARGUS_CURATOR_CORS", "1")
    app = create_app(cors=False)
    client = TestClient(app)
    r = client.get("/", headers={"Origin": "https://demo.example"})
    assert r.headers.get("access-control-allow-origin") is None
