"""Core scanner tests — no GPU required."""

from __future__ import annotations

import io
import struct
import zlib

import pytest
from PIL import Image

from argus_curator.filters import sharpness, artifact_score, check_resolution, check_aspect, check_blur
from argus_curator.types import CurateConfig, FilterConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_image(width: int, height: int, noise: bool = True) -> bytes:
    """Create a minimal JPEG-encoded image."""
    mode = "RGB"
    if noise:
        import random
        import numpy as np
        arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        img = Image.fromarray(arr, mode)
    else:
        img = Image.new(mode, (width, height), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _pil(width: int, height: int, noise: bool = True) -> Image.Image:
    if noise:
        import numpy as np
        arr = (255 * __import__("numpy").random.rand(height, width, 3)).astype("uint8")
        return Image.fromarray(arr, "RGB")
    return Image.new("RGB", (width, height), 128)


# ---------------------------------------------------------------------------
# Filter tests
# ---------------------------------------------------------------------------

class TestFilters:
    def test_sharpness_noisy_higher_than_flat(self):
        flat = _pil(256, 256, noise=False)
        noisy = _pil(256, 256, noise=True)
        assert sharpness(noisy) > sharpness(flat)

    def test_artifact_score_range(self):
        img = _pil(256, 256)
        s = artifact_score(img)
        assert 0.0 <= s <= 1.0

    def test_resolution_reject(self):
        img = _pil(200, 200)
        reason = check_resolution(img, min_short_side=512)
        assert reason is not None
        assert "resolution" in reason

    def test_resolution_pass(self):
        img = _pil(800, 600)
        assert check_resolution(img, min_short_side=512) is None

    def test_aspect_reject(self):
        img = _pil(1000, 100)  # ratio 10:1
        reason = check_aspect(img, max_ratio=3.0)
        assert reason is not None
        assert "aspect" in reason

    def test_aspect_pass(self):
        img = _pil(1000, 500)  # ratio 2:1
        assert check_aspect(img, max_ratio=3.0) is None

    def test_blur_flat_image_rejected(self):
        flat = _pil(256, 256, noise=False)
        s, reason = check_blur(flat, threshold=100.0)
        assert reason is not None
        assert "blurry" in reason

    def test_blur_noisy_passes(self):
        noisy = _pil(512, 512, noise=True)
        s, reason = check_blur(noisy, threshold=100.0)
        # noisy images should have high Laplacian variance
        assert s > 0.0


# ---------------------------------------------------------------------------
# Scanner integration
# ---------------------------------------------------------------------------

class TestScanner:
    def test_scan_bytes_batch_rejects_tiny(self):
        from argus_curator.scanner import scan_bytes_batch
        cfg = CurateConfig()
        cfg.embeddings.use_clip = False
        cfg.embeddings.use_dino = False
        cfg.detectors.use_yolo = False
        cfg.detectors.use_mtcnn = False

        tiny = _make_image(64, 64)
        summary = scan_bytes_batch([("tiny.jpg", "bytes:tiny.jpg", tiny)], cfg)
        assert summary.total == 1
        assert summary.rejected_filters == 1
        assert summary.selected == 0

    def test_scan_bytes_batch_passes_good_image(self):
        from argus_curator.scanner import scan_bytes_batch
        cfg = CurateConfig()
        cfg.embeddings.use_clip = False
        cfg.embeddings.use_dino = False
        cfg.detectors.use_yolo = False
        cfg.detectors.use_mtcnn = False

        good = _make_image(800, 600, noise=True)
        summary = scan_bytes_batch([("good.jpg", "bytes:good.jpg", good)], cfg)
        assert summary.total == 1
        assert summary.rejected_filters == 0

    def test_duplicate_detection(self):
        from argus_curator.scanner import scan_bytes_batch
        cfg = CurateConfig()
        cfg.embeddings.use_clip = False
        cfg.embeddings.use_dino = False
        cfg.detectors.use_yolo = False
        cfg.detectors.use_mtcnn = False
        cfg.duplicates.phash_hamming_distance = 10

        img_bytes = _make_image(800, 600, noise=True)
        items = [
            ("img_a.jpg", "bytes:img_a.jpg", img_bytes),
            ("img_b.jpg", "bytes:img_b.jpg", img_bytes),  # identical
        ]
        summary = scan_bytes_batch(items, cfg)
        dupes = [r for r in summary.results if r["is_duplicate"]]
        assert len(dupes) == 1

    def test_summary_fields_present(self):
        from argus_curator.scanner import scan_bytes_batch
        cfg = CurateConfig()
        cfg.embeddings.use_clip = False
        cfg.embeddings.use_dino = False
        cfg.detectors.use_yolo = False
        cfg.detectors.use_mtcnn = False

        good = _make_image(800, 600, noise=True)
        summary = scan_bytes_batch([("x.jpg", "bytes:x.jpg", good)], cfg)
        d = summary.to_dict()
        for key in ("total", "selected", "rejected_filters", "duplicates_removed",
                    "objective", "results", "selected_names"):
            assert key in d


# ---------------------------------------------------------------------------
# Preset tests
# ---------------------------------------------------------------------------

class TestPresets:
    def test_identity_enables_detectors(self):
        cfg = CurateConfig.for_objective("identity")
        assert cfg.detectors.use_mtcnn is True
        assert cfg.detectors.use_yolo is True
        assert cfg.scoring.weight_subject > 0.15

    def test_style_raises_aesthetic_weight(self):
        cfg = CurateConfig.for_objective("style")
        assert cfg.scoring.weight_aesthetic >= 0.30

    def test_wardrobe_raises_resolution_floor(self):
        cfg = CurateConfig.for_objective("wardrobe")
        assert cfg.filters.min_short_side >= 640

    def test_unknown_objective_does_not_crash(self):
        cfg = CurateConfig.for_objective("unknown_xyz")
        assert cfg.objective == "unknown_xyz"


# ---------------------------------------------------------------------------
# Clustering / selection tests
# ---------------------------------------------------------------------------

class TestClustering:
    def test_mark_duplicates_keeps_best(self):
        from argus_curator.clustering import mark_duplicates
        from argus_curator.types import ImageResult

        import imagehash
        from PIL import Image as PILImage
        ph = str(imagehash.phash(PILImage.new("RGB", (64, 64))))

        r1 = ImageResult(
            name="a.jpg", source="bytes:a", width=100, height=100,
            short_side=100, aspect_ratio=1.0, sharpness=500.0,
            artifact_score=0.9, phash=ph, passed=True, reject_reason=None,
            is_duplicate=False, duplicate_of=None, score=0.9,
        )
        r2 = ImageResult(
            name="b.jpg", source="bytes:b", width=100, height=100,
            short_side=100, aspect_ratio=1.0, sharpness=200.0,
            artifact_score=0.7, phash=ph, passed=True, reject_reason=None,
            is_duplicate=False, duplicate_of=None, score=0.5,
        )
        mark_duplicates([r1, r2], max_hamming=10)
        assert not r1.is_duplicate
        assert r2.is_duplicate
        assert r2.duplicate_of == "a.jpg"

    def test_selection_respects_target_count(self):
        from argus_curator.selection import select
        from argus_curator.types import ImageResult

        candidates = []
        for i in range(10):
            r = ImageResult(
                name=f"img_{i}.jpg", source=f"bytes:img_{i}", width=800, height=600,
                short_side=600, aspect_ratio=1.33, sharpness=500.0,
                artifact_score=0.9, phash=f"{'0' * 16}", passed=True,
                reject_reason=None, is_duplicate=False, duplicate_of=None,
                score=float(i) / 10.0,
            )
            candidates.append(r)

        selected = select(candidates, target_n=5, diversity_weight=0.0,
                          use_embedding_clusters=False)
        assert len(selected) == 5
