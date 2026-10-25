"""Places365 scene tags: labels, preprocessing, tagging sweep, filters, diversity.

The model session is a stand-in, so none of this needs onnxruntime or torch.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from argus_curator import runtime, scenes
from argus_curator._places365_labels import CATEGORIES, OUTDOOR
from argus_curator.export import write_report
from argus_curator.models import ExportRequest, ImageResult, ScanSummary, SceneConfig, SceneLabel, SceneTags
from argus_curator.selection import decide_selection, select_diverse


def test_labels_match_upstream_files() -> None:
    assert len(CATEGORIES) == len(OUTDOOR) == 365
    assert sum(OUTDOOR) == 204  # IO_places365.txt: 161 indoor, 204 outdoor
    assert CATEGORIES[0] == "airfield" and OUTDOOR[0]
    assert CATEGORIES[1] == "airplane_cabin" and not OUTDOOR[1]
    assert "apartment_building/outdoor" in CATEGORIES  # nested names keep their suffix
    assert len(set(CATEGORIES)) == 365


def test_preprocess_resizes_then_centre_crops() -> None:
    # Left half red, right half blue, on a wide image: the non-aspect-preserving
    # 256x256 resize keeps the split at the centre column of the 224 crop.
    arr = np.zeros((100, 400, 3), dtype=np.uint8)
    arr[:, :200] = (255, 0, 0)
    arr[:, 200:] = (0, 0, 255)
    batch = scenes.preprocess([Image.fromarray(arr), Image.fromarray(arr).convert("L")])
    assert batch.shape == (2, 3, 224, 224) and batch.dtype == np.float32
    red = (1.0 - 0.485) / 0.229
    assert batch[0, 0, 112, 10] == pytest.approx(red, rel=1e-4)
    assert batch[0, 0, 112, 213] < 0  # right of the split: no red
    assert batch[1, 0, 0, 0] == pytest.approx(batch[1, 1, 0, 0] * 0.224 / 0.229 + (0.456 - 0.485) / 0.229, abs=1e-4)


def _probs(peaks: dict[str, float]) -> np.ndarray:
    p = np.full(365, 0.0)
    for label, v in peaks.items():
        p[CATEGORIES.index(label)] = v
    rest = 1.0 - p.sum()
    p[p == 0] = rest / (p == 0).sum()
    return p


def test_to_tags_top_k_and_environment_mass() -> None:
    tags = scenes.to_tags(_probs({"beach": 0.5, "coast": 0.2, "bedroom": 0.1}), top_k=3)
    assert [c.label for c in tags.categories] == ["beach", "coast", "bedroom"]
    assert tags.categories[0].conf == 0.5
    expected_out = 0.7 + (0.2 / 362) * (204 - 2)
    assert tags.environment == "outdoor"
    assert tags.environment_conf == pytest.approx(expected_out, abs=1e-4)

    indoor = scenes.to_tags(_probs({"bedroom": 0.9}), top_k=1)
    assert indoor.environment == "indoor" and indoor.environment_conf >= 0.9


def test_softmax_rows_sum_to_one() -> None:
    p = scenes.softmax(np.random.default_rng(0).standard_normal((3, 365)) * 5)
    np.testing.assert_allclose(p.sum(axis=1), 1.0)


def test_model_path_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    assert scenes.model_path() == runtime.MODEL_DIR / scenes.DEFAULT_MODEL_NAME
    monkeypatch.setenv(scenes.MODEL_ENV, str(tmp_path / "p.onnx"))
    assert scenes.model_path() == tmp_path / "p.onnx"
    assert not scenes.scenes_available()


# ---------------------------------------------------------------------------
# Tagging sweep
# ---------------------------------------------------------------------------


class _Input:
    name = "input"


class _Session:
    def __init__(self) -> None:
        self.batches: list[int] = []

    def get_inputs(self) -> list[_Input]:
        return [_Input()]

    def run(self, _outputs, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        x = feeds["input"]
        assert x.shape[1:] == (3, 224, 224)
        self.batches.append(x.shape[0])
        # Bright images -> "beach", dark -> "bedroom", via a big logit.
        logits = np.zeros((x.shape[0], 365))
        for i, img in enumerate(x):
            label = "beach" if img.mean() > 0 else "bedroom"
            logits[i, CATEGORIES.index(label)] = 10.0
        return [logits]


def _png(value: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), (value,) * 3).save(buf, "PNG")
    return buf.getvalue()


def test_tag_images_batches_passing_images_only(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _Session()
    monkeypatch.setattr(scenes, "get_session", lambda provs: session)
    items = [(f"{i}.png", f"/d/{i}.png", _png(250 if i % 2 else 5)) for i in range(20)]
    items.append(("broken.png", "/d/broken.png", b"not an image"))
    results = [ImageResult(rel_path=rel, abs_path=a, passed=rel != "0.png") for rel, a, _ in items]
    seen: list[tuple[int, int]] = []

    scenes.tag_images(results, items, SceneConfig(enabled=True, top_k=2), progress=lambda d, t: seen.append((d, t)))

    by_rel = {r.rel_path: r for r in results}
    assert by_rel["0.png"].scene is None  # failed hard filters: not tagged
    assert by_rel["broken.png"].scene is None  # undecodable: skipped, sweep goes on
    assert by_rel["1.png"].scene.categories[0].label == "beach"
    assert by_rel["2.png"].scene.categories[0].label == "bedroom"
    assert by_rel["2.png"].scene.environment == "indoor"
    assert len(by_rel["1.png"].scene.categories) == 2
    assert session.batches == [16, 3]  # 20 passing (incl. broken), minus the broken one
    assert seen[0] == (0, 20) and seen[-1] == (20, 20)


def test_tag_images_disabled_is_a_no_op(monkeypatch: pytest.MonkeyPatch) -> None:
    def _no_session(provs):  # pragma: no cover - must not be reached
        raise AssertionError("Places365 loaded although scenes are disabled")

    monkeypatch.setattr(scenes, "get_session", _no_session)
    results = [ImageResult(rel_path="a.png", abs_path="/d/a.png", passed=True)]
    scenes.tag_images(results, [("a.png", "/d/a.png", _png(9))], SceneConfig())
    assert results[0].scene is None


def test_fetch_weights_refuses_a_bad_download(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc) -> None:
            self.close()

    monkeypatch.setattr(scenes, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(scenes.urllib.request, "urlopen", lambda url, timeout: _Resp(b"tampered"))
    with pytest.raises(runtime.ModelUnavailable, match="does not match the pinned sha256"):
        scenes.fetch_weights("resnet18")
    assert not any(tmp_path.rglob("*.pth.tar*"))  # nothing unverified left behind


def test_scan_summary_without_scenes_config_still_loads() -> None:
    """Scans cached before scene tagging existed have no scenes_config."""
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
    assert ScanSummary.model_validate(legacy).scenes_config == SceneConfig()


# ---------------------------------------------------------------------------
# Curate-side filters, diversity, report
# ---------------------------------------------------------------------------


def _tags(env: str, env_conf: float, *labels: tuple[str, float]) -> SceneTags:
    return SceneTags(
        categories=[SceneLabel(label=lbl, conf=c) for lbl, c in labels], environment=env, environment_conf=env_conf
    )


def _img(rel: str, scene: SceneTags | None, phash: str = "", score: float = 0.9) -> ImageResult:
    return ImageResult(rel_path=rel, abs_path=f"/d/{rel}", score=score, passed=True, scene=scene, phash=phash)


def test_scene_filters() -> None:
    results = [
        _img("beach.jpg", _tags("outdoor", 0.95, ("beach", 0.6), ("coast", 0.2))),
        _img("coast-tail.jpg", _tags("outdoor", 0.9, ("harbor", 0.7), ("coast", 0.05))),
        _img("bedroom.jpg", _tags("indoor", 0.97, ("bedroom", 0.8))),
        _img("unsure.jpg", _tags("outdoor", 0.52, ("coast", 0.3))),
        _img("untagged.jpg", None),
    ]
    req = ExportRequest(
        dest="/o", min_score=0.0, scenes=["beach", "coast"], scene_environments=["outdoor"], min_scene_conf=0.6
    )
    selected, why = decide_selection(results, req, 0.0)
    assert {r.rel_path for r in selected} == {"beach.jpg"}
    assert why["coast-tail.jpg"] == "scene-low-confidence"
    assert why["bedroom.jpg"] == "scene-environment-excluded"
    assert why["unsure.jpg"] == "scene-environment-low-confidence"
    assert why["untagged.jpg"] == "scene-missing"

    # Default floor (0.1) still rejects a label that only appears in the tail.
    selected, why = decide_selection(results, ExportRequest(dest="/o", min_score=0.0, scenes=["coast"]), 0.0)
    assert {r.rel_path for r in selected} == {"beach.jpg", "unsure.jpg"}
    assert why["coast-tail.jpg"] == "scene-low-confidence"
    assert why["bedroom.jpg"] == "scene-excluded"


def test_unknown_scene_label_is_rejected() -> None:
    with pytest.raises(ValidationError, match="beech"):
        ExportRequest(dest="/o", scenes=["beach", "beech"])


def test_no_scene_filter_keeps_untagged_images() -> None:
    selected, _ = decide_selection([_img("a.jpg", None)], ExportRequest(dest="/o", min_score=0.0), 0.0)
    assert [r.rel_path for r in selected] == ["a.jpg"]


def test_diversify_scenes_spreads_backdrops() -> None:
    # Same pHash everywhere, so pHash alone gives no reason to prefer anything
    # over score; three near-identical bedrooms outscore one beach.
    h = "ffffffffffffffff"
    bedroom = _tags("indoor", 0.9, ("bedroom", 0.8), ("hotel_room", 0.1))
    beach = _tags("outdoor", 0.9, ("beach", 0.8), ("coast", 0.1))
    pool = [
        _img("bed1.jpg", bedroom, h, 0.95),
        _img("bed2.jpg", bedroom, h, 0.94),
        _img("bed3.jpg", bedroom, h, 0.93),
        _img("beach.jpg", beach, h, 0.80),
    ]
    plain = select_diverse(pool, 2, 0.5)
    spread = select_diverse(pool, 2, 0.5, scenes=True)
    assert {r.rel_path for r in plain} == {"bed1.jpg", "bed2.jpg"}
    assert {r.rel_path for r in spread} == {"bed1.jpg", "beach.jpg"}

    req = ExportRequest(dest="/o", min_score=0.0, max_keep=2, diversify_scenes=True)
    selected, why = decide_selection(pool, req, 0.5)
    assert {r.rel_path for r in selected} == {"bed1.jpg", "beach.jpg"}
    assert why["bed2.jpg"] == "diversity-trimmed"


def test_report_includes_scene_columns(tmp_path: Path) -> None:
    results = [_img("a.jpg", _tags("outdoor", 0.93, ("beach", 0.61))), _img("b.jpg", None)]
    report = tmp_path / "r.csv"
    write_report(results, {}, set(), {}, report)
    rows = {r["rel_path"]: r for r in csv.DictReader(report.open())}
    assert rows["a.jpg"]["scene"] == "beach" and rows["a.jpg"]["scene_conf"] == "0.61"
    assert rows["a.jpg"]["scene_environment"] == "outdoor"
    assert rows["b.jpg"]["scene"] == "" and rows["b.jpg"]["scene_environment_conf"] == ""
