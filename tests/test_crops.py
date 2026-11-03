"""Crops: applied to the bytes in memory, seen by every check, written cropped on export, never to the source."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from argus_curator import crops, scan_folder
from argus_curator.export import export_selection
from argus_curator.models import CropBox, CropConfig, ExportRequest, FaceConfig, NsfwConfig
from argus_curator.selection import decide_selection
from argus_curator.server.app import create_app

NO_FACES = FaceConfig(enabled=False)
NO_NSFW = NsfwConfig(enabled=False)


def _noise(w: int, h: int, seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8), "RGB")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _box(left: int, top: int, right: int, bottom: int) -> dict[str, int]:
    return {"left": left, "top": top, "right": right, "bottom": bottom}


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    """Two sharp 900x800 frames: ``marked.png`` with a crop sidecar, ``plain.png`` without."""
    root = tmp_path / "src"
    root.mkdir()
    _noise(900, 800, 1).save(root / "marked.png")
    _noise(900, 800, 2).save(root / "plain.png")
    crops.sidecar_for(root / "marked.png").write_text(json.dumps(_box(0, 0, 900, 700)))
    return root


def _scan(root: Path, cfg: CropConfig | None = None):  # type: ignore[no-untyped-def]
    return scan_folder(root, faces_cfg=NO_FACES, nsfw_cfg=NO_NSFW, crop_cfg=cfg)


def test_crop_follows_exif_orientation_and_drops_the_tag() -> None:
    img = _noise(800, 600)
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 CW to display: shown as 600x800
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif.tobytes(), quality=95)

    out = Image.open(io.BytesIO(crops.crop_bytes(buf.getvalue(), CropBox(**_box(0, 0, 600, 700)))))
    assert out.format == "JPEG"
    assert out.size == (600, 700)
    assert out.getexif().get(0x0112) is None


@pytest.mark.parametrize(
    ("box", "kind"),
    [(_box(0, 0, 901, 800), "invalid"), (_box(100, 0, 100, 800), "invalid")],
)
def test_a_box_outside_the_frame_or_empty_is_refused(box: dict[str, int], kind: str) -> None:
    buf = io.BytesIO()
    _noise(900, 800).save(buf, format="PNG")
    with pytest.raises(crops.CropError, match=f"^{kind} "):
        crops.crop_bytes(buf.getvalue(), CropBox(**box))


def test_every_check_sees_the_cropped_frame_and_the_source_is_untouched(folder: Path) -> None:
    before = _sha(folder / "marked.png")
    summary = _scan(folder, CropConfig(enabled=True))
    by_rel = {r.rel_path: r for r in summary.results}

    marked = by_rel["marked.png"]
    assert marked.crop == CropBox(**_box(0, 0, 900, 700))
    assert (marked.width, marked.height) == (900, 700)
    assert marked.passed
    assert by_rel["plain.png"].crop is None
    assert (by_rel["plain.png"].width, by_rel["plain.png"].height) == (900, 800)
    assert summary.crop_config.enabled
    assert _sha(folder / "marked.png") == before


def test_crops_are_off_unless_asked_for(folder: Path) -> None:
    marked = next(r for r in _scan(folder).results if r.rel_path == "marked.png")
    assert marked.crop is None
    assert marked.height == 800


def test_an_inline_box_wins_over_the_sidecar_and_sidecars_can_be_ignored(folder: Path) -> None:
    inline = CropConfig(enabled=True, boxes={"plain.png": CropBox(**_box(50, 50, 850, 750))}, sidecars=False)
    by_rel = {r.rel_path: r for r in _scan(folder, inline).results}
    assert (by_rel["plain.png"].width, by_rel["plain.png"].height) == (800, 700)
    assert by_rel["marked.png"].crop is None

    override = CropConfig(enabled=True, boxes={"marked.png": CropBox(**_box(0, 100, 900, 800))})
    marked = next(r for r in _scan(folder, override).results if r.rel_path == "marked.png")
    assert marked.crop == CropBox(**_box(0, 100, 900, 800))


def test_a_crop_that_cannot_be_applied_is_a_reject_no_export_keeps(folder: Path) -> None:
    crops.sidecar_for(folder / "plain.png").write_text("{not json")
    bad = CropConfig(enabled=True, boxes={"marked.png": CropBox(**_box(0, 0, 2000, 700))})
    summary = _scan(folder, bad)
    by_rel = {r.rel_path: r for r in summary.results}

    assert by_rel["marked.png"].reject_reason.startswith("crop:invalid (")
    assert by_rel["plain.png"].reject_reason.startswith("crop:unreadable (")
    assert summary.reject_reasons == {"crop:invalid": 1, "crop:unreadable": 1}

    req = ExportRequest(dest="unused", min_score=0.0, include_rejected=True)
    selected, keep_reason = decide_selection(summary.results, req, 0.0)
    assert selected == []
    assert keep_reason["marked.png"].startswith("crop:invalid")


def test_export_writes_the_crop_and_the_manifest_names_it(folder: Path, tmp_path: Path) -> None:
    summary = _scan(folder, CropConfig(enabled=True))
    before = _sha(folder / "marked.png")
    dest = tmp_path / "out"
    result = export_selection(summary, ExportRequest(dest=str(dest), min_score=0.0, keep_similar=True))

    assert result.copied == 2
    assert result.cropped == 1
    assert Image.open(dest / "marked.png").size == (900, 700)
    assert Image.open(dest / "plain.png").size == (900, 800)
    assert not crops.sidecar_for(dest / "marked.png").exists()
    assert _sha(folder / "marked.png") == before

    rows = {row["rel_path"]: row for row in map(json.loads, (dest / "manifest.jsonl").read_text().splitlines())}
    assert rows["marked.png"]["crop"] == _box(0, 0, 900, 700)
    assert rows["plain.png"]["crop"] is None


@pytest.mark.parametrize("mode", ["symlink", "move"])
def test_a_cropped_selection_exports_only_by_copy(folder: Path, tmp_path: Path, mode: str) -> None:
    summary = _scan(folder, CropConfig(enabled=True))
    with pytest.raises(ValueError, match='mode "copy"'):
        export_selection(summary, ExportRequest(dest=str(tmp_path / "out"), mode=mode, min_score=0.0))
    assert (folder / "marked.png").exists()
    assert not (tmp_path / "out").exists()


def test_export_over_an_earlier_symlink_does_not_write_through_it(folder: Path, tmp_path: Path) -> None:
    dest = tmp_path / "out"
    dest.mkdir()
    (dest / "marked.png").symlink_to(folder / "marked.png")
    before = _sha(folder / "marked.png")
    summary = _scan(folder, CropConfig(enabled=True))
    export_selection(summary, ExportRequest(dest=str(dest), min_score=0.0, keep_similar=True))
    assert not (dest / "marked.png").is_symlink()
    assert _sha(folder / "marked.png") == before


def test_server_scans_with_crops_and_refuses_a_symlink_export(folder: Path, tmp_path: Path) -> None:
    app = create_app(cache_dir=str(tmp_path / "cache"), source_root=str(folder), export_root=str(tmp_path / "exp"))
    client = TestClient(app)
    assert client.get("/health").json()["crops"] is True

    resp = client.post(
        "/scan/folder",
        json={"folder": "", "faces": {"enabled": False}, "nsfw": {"enabled": False}, "crops": {"enabled": True}},
    )
    assert resp.status_code == 200
    body = resp.json()
    marked = next(r for r in body["results"] if r["rel_path"] == "marked.png")
    assert marked["crop"] == _box(0, 0, 900, 700)

    for endpoint in ("/export", "/export/stream"):
        resp = client.post(endpoint, json={"scan_id": body["scan_id"], "dest": "out", "mode": "symlink"})
        assert resp.status_code == 400, endpoint
        assert 'mode "copy"' in resp.json()["detail"]
