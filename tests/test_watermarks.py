"""Watermark detection: hits planned into edge crops, overlays and failures rejected, manual boxes first."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from argus_curator import crops, scan_folder, watermarks
from argus_curator.models import CropBox, CropConfig, FaceConfig, NsfwConfig, WatermarkDetection
from argus_curator.server.app import create_app

NO_FACES = FaceConfig(enabled=False)
NO_NSFW = NsfwConfig(enabled=False)
DETECT = CropConfig(enabled=True, detect=True)


def _hit(x: float, y: float, w: float, h: float, score: float = 0.9) -> WatermarkDetection:
    return WatermarkDetection(label="watermark", score=score, bbox=[x, y, w, h])


def _noise(w: int, h: int, seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8), "RGB")


class _FakeDetector:
    """Answers from a table keyed by the upright frame's size; records what it was sent."""

    def __init__(self, hits: dict[tuple[int, int], list[WatermarkDetection] | Exception]) -> None:
        self.hits = hits
        self.sizes: list[tuple[int, int]] = []

    def check(self, data: bytes, filename: str = "image.jpg") -> list[WatermarkDetection]:
        size = Image.open(io.BytesIO(data)).size
        self.sizes.append(size)
        answer = self.hits.get(size, [])
        if isinstance(answer, Exception):
            raise answer
        return answer


@pytest.fixture
def detector(monkeypatch: pytest.MonkeyPatch) -> _FakeDetector:
    fake = _FakeDetector({})
    monkeypatch.setenv(watermarks.URL_ENV, "http://watermark:8080")
    monkeypatch.setattr(watermarks, "client", lambda: fake)
    return fake


# --- planning ---------------------------------------------------------------


def test_a_corner_mark_is_cut_along_the_edge_that_loses_least() -> None:
    # 1000x800, a 200x40 stamp in the bottom-right: cutting the bottom loses 44 rows,
    # cutting the right 204 columns, so the bottom goes.
    box = crops.plan_crop(1000, 800, [_hit(780, 740, 200, 40)], CropConfig(margin=4))
    assert box == CropBox(left=0, top=0, right=1000, bottom=736)


def test_marks_on_two_edges_are_both_cut() -> None:
    hits = [_hit(10, 10, 150, 30), _hit(800, 760, 180, 30)]
    box = crops.plan_crop(1000, 800, hits, CropConfig(margin=0))
    assert box == CropBox(left=0, top=40, right=1000, bottom=760)


def test_a_central_mark_is_an_overlay_and_a_faint_one_is_ignored() -> None:
    assert crops.plan_crop(1000, 800, [_hit(400, 350, 200, 100)], CropConfig()) == "overlay"
    assert crops.plan_crop(1000, 800, [_hit(400, 350, 200, 100, score=0.1)], CropConfig()) is None
    assert crops.plan_crop(1000, 800, [], CropConfig()) is None


def test_max_cut_bounds_each_edge() -> None:
    hit = [_hit(0, 600, 1000, 100)]  # 200 rows to cut from the bottom of 800
    assert crops.plan_crop(1000, 800, hit, CropConfig(max_cut=0.15, margin=0)) == "overlay"
    assert crops.plan_crop(1000, 800, hit, CropConfig(max_cut=0.3, margin=0)) == CropBox(
        left=0, top=0, right=1000, bottom=600
    )


def test_a_tiled_watermark_is_an_overlay() -> None:
    hits = [_hit(10 + 5 * i, 10, 20, 20) for i in range(crops.MAX_HITS + 1)]
    assert crops.plan_crop(1000, 800, hits, CropConfig()) == "overlay"


# --- in a scan --------------------------------------------------------------


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    root = tmp_path / "src"
    root.mkdir()
    _noise(1000, 800, 1).save(root / "corner.png")
    _noise(1001, 800, 2).save(root / "overlay.png")
    _noise(1002, 800, 3).save(root / "clean.png")
    return root


def _by_rel(root: Path, cfg: CropConfig) -> dict:  # type: ignore[type-arg]
    return {r.rel_path: r for r in scan_folder(root, faces_cfg=NO_FACES, nsfw_cfg=NO_NSFW, crop_cfg=cfg).results}


def test_detected_marks_are_cut_overlays_rejected_clean_frames_untouched(folder: Path, detector: _FakeDetector) -> None:
    detector.hits = {(1000, 800): [_hit(780, 740, 200, 40)], (1001, 800): [_hit(400, 350, 200, 100)]}
    by_rel = _by_rel(folder, DETECT)

    corner = by_rel["corner.png"]
    assert corner.crop == CropBox(left=0, top=0, right=1000, bottom=736)
    assert corner.crop_source == "detected"
    assert (corner.width, corner.height) == (1000, 736)
    assert corner.watermarks == [_hit(780, 740, 200, 40)]
    assert corner.passed

    assert by_rel["overlay.png"].reject_reason.startswith("crop:overlay (")
    assert by_rel["overlay.png"].crop is None

    clean = by_rel["clean.png"]
    assert clean.watermarks == []
    assert clean.crop is None
    assert clean.passed


def test_keeping_overlays_records_them_uncropped(folder: Path, detector: _FakeDetector) -> None:
    detector.hits = {(1001, 800): [_hit(400, 350, 200, 100)]}
    overlay = _by_rel(folder, DETECT.model_copy(update={"reject_overlay": False}))["overlay.png"]
    assert overlay.passed
    assert overlay.crop is None
    assert overlay.watermarks == [_hit(400, 350, 200, 100)]


def test_an_image_the_detector_could_not_check_is_rejected(folder: Path, detector: _FakeDetector) -> None:
    detector.hits = {(1002, 800): watermarks.WatermarkCheckFailed("boom")}
    clean = _by_rel(folder, DETECT)["clean.png"]
    assert clean.reject_reason.startswith("crop:unchecked (")
    assert clean.watermarks is None


def test_a_box_by_hand_wins_and_skips_the_detector(folder: Path, detector: _FakeDetector) -> None:
    crops.sidecar_for(folder / "corner.png").write_text(json.dumps({"left": 0, "top": 0, "right": 1000, "bottom": 700}))
    corner = _by_rel(folder, DETECT)["corner.png"]
    assert corner.crop_source == "manual"
    assert corner.crop == CropBox(left=0, top=0, right=1000, bottom=700)
    assert (1000, 800) not in detector.sizes


def test_the_detector_is_sent_the_upright_frame(tmp_path: Path, detector: _FakeDetector) -> None:
    root = tmp_path / "src"
    root.mkdir()
    exif = Image.Exif()
    exif[0x0112] = 6  # stored 900x700, displayed 700x900
    _noise(900, 700).save(root / "rotated.jpg", exif=exif.tobytes(), quality=95)
    detector.hits = {(700, 900): [_hit(0, 860, 300, 30)]}
    rotated = _by_rel(root, DETECT)["rotated.jpg"]
    assert detector.sizes == [(700, 900)]
    assert rotated.crop == CropBox(left=0, top=0, right=700, bottom=856)


def test_detection_without_a_detector_is_refused(folder: Path) -> None:
    with pytest.raises(ValueError, match=watermarks.URL_ENV):
        _by_rel(folder, DETECT)


def test_server_refuses_detection_without_a_detector(folder: Path, tmp_path: Path) -> None:
    client = TestClient(create_app(cache_dir=str(tmp_path / "cache"), source_root=str(folder)))
    assert client.get("/health").json()["watermark_detect"] is True
    assert client.get("/detectors").json()["watermark"] is False
    resp = client.post(
        "/scan/folder",
        json={"folder": "", "faces": {"enabled": False}, "crops": {"enabled": True, "detect": True}},
    )
    assert resp.status_code == 400
    assert watermarks.URL_ENV in resp.json()["detail"]


def test_to_detections_drops_malformed_rows() -> None:
    rows = [
        {"class": "watermark", "score": 0.5, "box": [1, 2, 3, 4]},
        {"class": "watermark", "score": 0.9, "box": [1, 2, 0, 4]},
        {"class": "watermark", "score": "high", "box": [1, 2, 3, 4]},
        {"class": "logo", "score": 0.7, "box": [5, 6, 7, 8]},
        "junk",
    ]
    out = watermarks.to_detections(rows)
    assert [(d.label, d.score) for d in out] == [("logo", 0.7), ("watermark", 0.5)]
