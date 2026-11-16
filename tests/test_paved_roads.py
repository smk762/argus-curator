"""argus-cortex 0.8's shared helpers: upload caps (``limits``) and the caption hand-off (``net``).

Nothing here touches the network: ``fetch`` and ``check_url`` get an injected resolver, and
``fetch`` an ``httpx.MockTransport``.
"""

from __future__ import annotations

import functools
from pathlib import Path

import argus_cortex.net as cortex_net
import httpx
import pytest
from structlog.testing import capture_logs

from argus_curator import export, scan_folder
from argus_curator.export import _post_manifest_to_lens, export_selection
from argus_curator.models import ExportRequest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from argus_curator.server import create_app  # noqa: E402

LENS = "http://lens:8100/caption"


def _png_bytes(pad: int = 0) -> bytes:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (200, 30, 30)).save(buf, format="PNG")
    return buf.getvalue() + b"\0" * pad


# --- uploads ------------------------------------------------------------------


@pytest.fixture
def capped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, Path]:
    """A server with a 1 KiB upload cap and a tmp source root."""
    monkeypatch.setenv("CURATOR_MAX_UPLOAD_MB", str(1 / 1024))
    root = tmp_path / "mount"
    root.mkdir()
    return TestClient(create_app(cache_dir=str(tmp_path / "cache"), source_root=str(root))), root


def test_upload_under_the_cap_is_saved(capped: tuple[TestClient, Path]) -> None:
    client, root = capped
    resp = client.post("/upload", data={"folder": "in"}, files=[("files", ("a.png", _png_bytes(), "image/png"))])
    assert resp.status_code == 200
    assert resp.json()["saved"] == 1
    assert (root / "in" / "a.png").read_bytes() == _png_bytes()


def test_upload_over_the_cap_is_413_and_saves_nothing(capped: tuple[TestClient, Path]) -> None:
    client, root = capped
    resp = client.post(
        "/upload",
        data={"folder": "in"},
        files=[
            ("files", ("small.png", _png_bytes(), "image/png")),
            ("files", ("big.png", _png_bytes(pad=2048), "image/png")),
        ],
    )
    assert resp.status_code == 413
    assert "big.png" in resp.json()["detail"]
    assert not any((root / "in").iterdir())


async def test_read_upload_stops_at_the_cap_when_the_size_is_unknown() -> None:
    """The guard itself, for an upload whose size the parser did not record."""
    from argus_cortex.limits import TooLarge, read_upload

    class Upload:
        size = None

        def __init__(self, data: bytes) -> None:
            self.data, self.read_total = data, 0

        async def read(self, n: int = -1) -> bytes:
            part = self.data[self.read_total : self.read_total + n]
            self.read_total += len(part)
            return part

    big = Upload(b"x" * 10_000)
    with pytest.raises(TooLarge):
        await read_upload(big, 1024)
    assert big.read_total <= 1025


def test_faces_multipart_over_the_upload_cap_is_413(capped: tuple[TestClient, Path], monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr("argus_curator.server.app.faces_available", lambda: True)
    monkeypatch.setattr("argus_curator.server.app.mivolo_available", lambda: True)
    client, _ = capped
    resp = client.post("/faces", files={"image": ("big.png", _png_bytes(pad=2048), "image/png")})
    assert resp.status_code == 413


@pytest.mark.parametrize("value", ["0", "-5", "lots", "inf", "nan"])
def test_a_bad_upload_cap_refuses_to_start(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("CURATOR_MAX_UPLOAD_MB", value)
    with pytest.raises(ValueError, match="CURATOR_MAX_UPLOAD_MB"):
        create_app(cache_dir=str(tmp_path / "cache"))


# --- the caption hand-off ------------------------------------------------------


def _lens(monkeypatch: pytest.MonkeyPatch, address: str = "10.0.0.5") -> list[httpx.Request]:
    """Point every fetch/check_url at *address* through a mock lens; returns the requests it got."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"ok": True})

    def resolve(host: str, port: int) -> list[str]:
        return [address]

    monkeypatch.setattr(
        cortex_net,
        "fetch",
        functools.partial(cortex_net.fetch, resolve=resolve, transport=httpx.MockTransport(handler)),
    )
    monkeypatch.setattr("argus_curator.server.app.check_url", functools.partial(cortex_net.check_url, resolve=resolve))
    return seen


def _manifest(tmp_path: Path) -> Path:
    path = tmp_path / export.MANIFEST_NAME
    path.write_text('{"rel_path": "a.png"}\n')
    return path


def test_caption_handoff_to_a_private_address_is_refused_without_an_allow_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _lens(monkeypatch)
    with capture_logs() as logs:
        assert _post_manifest_to_lens(_manifest(tmp_path), LENS) is False
    assert seen == []
    failed = [e for e in logs if e["event"] == "caption_handoff_failed"]
    assert failed and failed[0]["host"] == "lens" and "url" not in failed[0]


def test_caption_handoff_reaches_an_allow_listed_internal_host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _lens(monkeypatch)
    monkeypatch.setenv("CURATOR_CAPTION_ALLOWED_HOSTS", " other , lens:8100")
    assert _post_manifest_to_lens(_manifest(tmp_path), LENS) is True
    (req,) = seen
    assert req.method == "POST" and req.headers["host"] == "lens:8100"
    assert req.headers["content-type"].startswith("multipart/form-data; boundary=")
    body = req.content
    assert b'name="manifest"; filename="manifest.jsonl"' in body and b'{"rel_path": "a.png"}' in body


def test_caption_handoff_allow_list_entry_is_port_specific(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _lens(monkeypatch)
    monkeypatch.setenv("CURATOR_CAPTION_ALLOWED_HOSTS", "lens:9999")
    assert _post_manifest_to_lens(_manifest(tmp_path), LENS) is False
    assert seen == []


def test_caption_handoff_logs_the_host_never_the_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _lens(monkeypatch)
    with capture_logs() as logs:
        assert _post_manifest_to_lens(_manifest(tmp_path), "http://user:hunter2@lens:8100/caption") is False
    assert "hunter2" not in repr(logs)
    assert [e["host"] for e in logs if e["event"] == "caption_handoff_failed"] == ["lens"]


def test_caption_handoff_does_not_follow_a_redirect(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(307, headers={"location": "http://lens:8100/elsewhere"})

    monkeypatch.setattr(
        cortex_net,
        "fetch",
        functools.partial(cortex_net.fetch, resolve=lambda h, p: ["10.0.0.5"], transport=httpx.MockTransport(handler)),
    )
    monkeypatch.setenv("CURATOR_CAPTION_ALLOWED_HOSTS", "lens")
    assert _post_manifest_to_lens(_manifest(tmp_path), LENS) is False
    assert len(seen) == 1


def test_engine_export_reports_a_refused_handoff_as_not_captioned(
    dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _lens(monkeypatch)
    summary = scan_folder(dataset)
    req = ExportRequest(scan_id=summary.scan_id, dest=str(tmp_path / "out"), min_score=0.0, caption_url=LENS)
    result = export_selection(summary, req)
    assert result.copied >= 1 and result.captioned is False


@pytest.fixture
def scanned(dataset: Path, tmp_path: Path) -> tuple[TestClient, Path, str]:
    exports = tmp_path / "exports"
    exports.mkdir()
    client = TestClient(
        create_app(cache_dir=str(tmp_path / "cache"), source_root=str(dataset), export_root=str(exports))
    )
    resp = client.post(
        "/scan/folder",
        json={"folder": "", "target_profile": {"target_category": "identity"}, "faces": {"enabled": False}},
    )
    assert resp.status_code == 200
    return client, exports, resp.json()["scan_id"]


@pytest.mark.parametrize("route", ["/export", "/export/stream"])
def test_export_refuses_a_private_caption_url_up_front(
    scanned: tuple[TestClient, Path, str], monkeypatch: pytest.MonkeyPatch, route: str
) -> None:
    client, exports, scan_id = scanned
    seen = _lens(monkeypatch)
    resp = client.post(route, json={"scan_id": scan_id, "dest": "out", "min_score": 0.0, "caption_url": LENS})
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "CURATOR_CAPTION_ALLOWED_HOSTS" in detail and LENS not in detail
    assert not (exports / "out").exists() and seen == []


def test_export_hands_off_to_an_allow_listed_caption_url(
    scanned: tuple[TestClient, Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, exports, scan_id = scanned
    seen = _lens(monkeypatch)
    monkeypatch.setenv("CURATOR_CAPTION_ALLOWED_HOSTS", "lens:8100")
    resp = client.post("/export", json={"scan_id": scan_id, "dest": "out", "min_score": 0.0, "caption_url": LENS})
    assert resp.status_code == 200
    assert resp.json()["captioned"] is True
    assert len(seen) == 1 and (exports / "out" / "manifest.jsonl").exists()


def test_export_allows_a_public_caption_url_without_an_allow_list(
    scanned: tuple[TestClient, Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _, scan_id = scanned
    seen = _lens(monkeypatch, address="93.184.215.14")
    resp = client.post(
        "/export",
        json={"scan_id": scan_id, "dest": "out", "min_score": 0.0, "caption_url": "https://lens.example.com/caption"},
    )
    assert resp.status_code == 200 and resp.json()["captioned"] is True
    assert len(seen) == 1
