"""Fashionpedia garments: labels, YOLOS pre/post-processing, sweep, filters, outfits.

The model session is a stand-in, so none of this needs onnxruntime or torch.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from argus_curator import garments
from argus_curator.export import write_report
from argus_curator.models import (
    GARMENT_LABELS,
    ExportRequest,
    GarmentConfig,
    GarmentDetection,
    ImageResult,
    ScanSummary,
)
from argus_curator.selection import decide_selection, select_diverse

# id2label of valentinafevu/yolos-fashionpedia, first 27 entries (the garments).
_MODEL_LABELS = [
    "shirt, blouse", "top, t-shirt, sweatshirt", "sweater", "cardigan", "jacket", "vest", "pants", "shorts",
    "skirt", "coat", "dress", "jumpsuit", "cape", "glasses", "hat", "headband, head covering, hair accessory",
    "tie", "glove", "watch", "belt", "leg warmer", "tights, stockings", "sock", "shoe", "bag, wallet", "scarf",
    "umbrella",
]  # fmt: skip


def test_slugs_follow_model_label_order() -> None:
    assert len(GARMENT_LABELS) == 27
    for slug, name in zip(GARMENT_LABELS, _MODEL_LABELS, strict=True):
        assert slug.split("_")[0] == name.split(",")[0].split(" ")[0].replace("-", ""), (slug, name)
    assert all("," not in s for s in GARMENT_LABELS)  # survives comma-separated CLI lists


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        ((1024, 683), (752, 512)),  # short edge 512, long floored to /16
        ((683, 1024), (512, 752)),
        ((512, 512), (512, 512)),
        ((3000, 500), (1328, 208)),  # long edge capped at 1333 first
        ((300, 200), (768, 512)),  # small images are scaled up
    ],
)
def test_resize_matches_yolos_processor(size: tuple[int, int], expected: tuple[int, int]) -> None:
    assert garments.resize_size(*size) == expected


def test_preprocess_shape_and_normalisation() -> None:
    x = garments.preprocess(Image.new("RGB", (1024, 683), (255, 0, 0)))
    assert x.shape == (1, 3, 512, 752) and x.dtype == np.float32
    assert x[0, 0, 0, 0] == pytest.approx((1.0 - 0.485) / 0.229, rel=1e-5)


def _query(cls: int, box: tuple[float, float, float, float], peak: float = 8.0) -> tuple[np.ndarray, list[float]]:
    logits = np.zeros(47)
    logits[cls] = peak
    return logits, list(box)


def test_postprocess_keeps_confident_garments_only() -> None:
    queries = [
        _query(10, (0.5, 0.5, 0.2, 0.4)),  # dress
        _query(31, (0.5, 0.3, 0.1, 0.1)),  # sleeve: a part, never recorded
        _query(46, (0.2, 0.2, 0.1, 0.1)),  # no-object
        _query(23, (0.9, 0.95, 0.4, 0.2), peak=0.5),  # shoe, but unsure
    ]
    logits = np.stack([q[0] for q in queries])
    boxes = np.array([q[1] for q in queries])
    out = garments.postprocess(logits, boxes, 1000, 500, min_conf=0.5)
    assert len(out) == 1
    lbl, conf, (x, y, w, h) = out[0]
    assert GARMENT_LABELS[lbl] == "dress" and conf > 0.9
    assert (x, y, w, h) == pytest.approx((400.0, 150.0, 200.0, 200.0))
    # The unsure shoe appears once the floor drops; its box is clipped to the frame.
    low = garments.postprocess(logits, boxes, 1000, 500, min_conf=0.0)
    shoe = next(o for o in low if GARMENT_LABELS[o[0]] == "shoe")
    assert shoe[2][0] + shoe[2][2] == pytest.approx(1000.0)


def test_touches_edge_and_dominant_color() -> None:
    assert garments.touches_edge([0, 100, 50, 50], 1000, 1000)
    assert garments.touches_edge([900, 100, 99, 50], 1000, 1000)
    assert not garments.touches_edge([100, 100, 50, 50], 1000, 1000)
    img = Image.new("RGB", (100, 100), (0, 0, 255))
    img.paste((255, 0, 0), (25, 25, 75, 75))  # red centre, blue background
    assert garments.dominant_color(img, [0, 0, 100, 100]) == "#ff0000"
    assert garments.dominant_color(img, [50, 50, 1, 1]) == "#ff0000"  # a sliver still yields a colour


def test_outfit_similarity() -> None:
    def det(label: str, color: str, conf: float = 0.9) -> GarmentDetection:
        return GarmentDetection(label=label, conf=conf, bbox=[0, 0, 1, 1], color=color)

    red_dress = garments.outfit([det("dress", "#c01010"), det("shoe", "#101010")], 0.5)
    red_dress_2 = garments.outfit([det("dress", "#b81512"), det("shoe", "#151515")], 0.5)
    black_dress = garments.outfit([det("dress", "#101010"), det("shoe", "#101010")], 0.5)
    jeans = garments.outfit([det("pants", "#2040a0"), det("shoe", "#101010")], 0.5)
    assert garments.outfit_similarity(red_dress, red_dress_2) > 0.9
    assert 0.4 < garments.outfit_similarity(red_dress, black_dress) < 0.6  # shoes match, dress colour does not
    assert garments.outfit_similarity(red_dress, jeans) == pytest.approx(1 / 3, abs=0.01)  # shoes only, of 3 types
    assert garments.outfit_similarity({}, {}) == 0.0
    # Detections under the floor are not part of the outfit.
    assert garments.outfit([det("hat", "#ffffff", conf=0.3)], 0.5) == {}


# ---------------------------------------------------------------------------
# Detection sweep
# ---------------------------------------------------------------------------


class _Input:
    name = "input"


class _Session:
    """A dress on bright images, nothing on dark ones."""

    def get_inputs(self) -> list[_Input]:
        return [_Input()]

    def run(self, _outputs, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        x = feeds["input"]
        logits = np.zeros((1, 100, 47))
        logits[0, :, 46] = 10.0  # every query: no-object
        if x.mean() > 0:
            logits[0, 0, 46], logits[0, 0, 10] = 0.0, 10.0
        boxes = np.tile([0.5, 0.5, 0.4, 0.6], (1, 100, 1))
        return [logits, boxes]


def _png(value: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (120, 80), (value, value // 2, 0)).save(buf, "PNG")
    return buf.getvalue()


def test_detect_images_sets_garments_on_passing_images(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(garments, "get_session", lambda provs: _Session())
    items = [("bright.png", "/d/b", _png(240)), ("dark.png", "/d/d", _png(0)), ("bad.png", "/d/x", b"nope")]
    items.append(("rejected.png", "/d/r", _png(240)))
    results = [ImageResult(rel_path=rel, abs_path=a, passed=rel != "rejected.png") for rel, a, _ in items]
    garments.detect_images(results, items, GarmentConfig(enabled=True))
    by = {r.rel_path: r for r in results}
    (dress,) = by["bright.png"].garments
    assert dress.label == "dress" and dress.color == "#f07800" and not dress.touches_edge
    assert by["dark.png"].garments == []  # ran, found nothing
    assert by["bad.png"].garments is None  # could not run
    assert by["rejected.png"].garments is None  # not a candidate


def test_scan_summary_without_garments_config_still_loads() -> None:
    from argus_curator.models import FaceConfig, ScanConfig, TargetProfile

    legacy = {
        "scan_id": "s",
        "folder": "/d",
        "target_profile": TargetProfile().model_dump(),
        "config": ScanConfig().model_dump(),
        "faces_config": FaceConfig().model_dump(),
        "total": 0,
        "passed": 0,
        "rejected": 0,
        "duplicates": 0,
        "similar_clusters": 0,
    }
    assert ScanSummary.model_validate(legacy).garments_config == GarmentConfig()


# ---------------------------------------------------------------------------
# Curate-side filters, outfit diversity, report
# ---------------------------------------------------------------------------


def _g(label: str, conf: float = 0.9, color: str = "#808080", edge: bool = False) -> GarmentDetection:
    return GarmentDetection(label=label, conf=conf, bbox=[10, 10, 50, 50], touches_edge=edge, color=color)


def _img(rel: str, worn: list[GarmentDetection] | None, score: float = 0.9) -> ImageResult:
    return ImageResult(rel_path=rel, abs_path=f"/d/{rel}", score=score, passed=True, garments=worn, phash="f" * 16)


def test_garment_filters() -> None:
    results = [
        _img("dress.jpg", [_g("dress"), _g("shoe")]),
        _img("dress-bag.jpg", [_g("dress"), _g("bag_wallet", 0.8)]),
        _img("unsure.jpg", [_g("dress", 0.35)]),
        _img("jeans.jpg", [_g("pants")]),
        _img("cut.jpg", [_g("dress", edge=True)]),
        _img("faint-bag.jpg", [_g("dress"), _g("bag_wallet", 0.31)]),
        _img("unscanned.jpg", None),
    ]
    req = ExportRequest(dest="/o", min_score=0.0, garments=["dress"], exclude_garments=["bag_wallet"])
    selected, why = decide_selection(results, req, 0.0)
    assert {r.rel_path for r in selected} == {"dress.jpg", "cut.jpg", "faint-bag.jpg"}
    assert why["dress-bag.jpg"] == "garment-excluded"
    assert why["unsure.jpg"] == "garment-low-confidence"
    assert why["jeans.jpg"] == "garment-not-found"
    assert why["unscanned.jpg"] == "garments-missing"

    req = req.model_copy(update={"garments_uncropped": True})
    selected, why = decide_selection(results, req, 0.0)
    assert why["cut.jpg"] == "garment-cropped"
    assert "dress.jpg" in {r.rel_path for r in selected}


def test_no_garment_filter_keeps_unscanned_images() -> None:
    selected, _ = decide_selection([_img("a.jpg", None)], ExportRequest(dest="/o", min_score=0.0), 0.0)
    assert [r.rel_path for r in selected] == ["a.jpg"]


def test_diversify_outfits_spreads_selection() -> None:
    # Identical pHash; three shots in the same red dress outscore one in jeans.
    red = [_g("dress", color="#c01010"), _g("shoe", color="#101010")]
    pool = [
        _img("red1.jpg", red, 0.95),
        _img("red2.jpg", red, 0.94),
        _img("red3.jpg", red, 0.93),
        _img("jeans.jpg", [_g("pants", color="#2040a0"), _g("top_tshirt_sweatshirt", color="#f0f0f0")], 0.80),
    ]
    assert {r.rel_path for r in select_diverse(pool, 2, 0.5)} == {"red1.jpg", "red2.jpg"}
    assert {r.rel_path for r in select_diverse(pool, 2, 0.5, outfits=True)} == {"red1.jpg", "jeans.jpg"}

    req = ExportRequest(dest="/o", min_score=0.0, max_keep=2, diversify_outfits=True)
    selected, why = decide_selection(pool, req, 0.5)
    assert {r.rel_path for r in selected} == {"red1.jpg", "jeans.jpg"}
    assert why["red3.jpg"] == "diversity-trimmed"


def test_report_includes_garments_column(tmp_path: Path) -> None:
    results = [
        _img("a.jpg", [_g("dress", 0.912, "#c01010"), _g("shoe", 0.5, "#101010", edge=True)]),
        _img("b.jpg", None),
    ]
    report = tmp_path / "r.csv"
    write_report(results, {}, set(), {}, report)
    rows = {r["rel_path"]: r for r in csv.DictReader(report.open())}
    assert rows["a.jpg"]["garments"] == "dress:0.91:#c01010;shoe:0.50:#101010*"
    assert rows["b.jpg"]["garments"] == ""
