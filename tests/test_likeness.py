"""Likeness: a consistency measure for subjects without a usable face (#191).

Synthetic vectors only: no model, no GPU.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from argus_curator import embeddings, scanner
from argus_curator.embeddings import measure_likeness, set_centroid
from argus_curator.export import export_selection
from argus_curator.models import (
    STORE_CONTEXT,
    EmbeddingConfig,
    ExportRequest,
    FaceConfig,
    ImageResult,
    ScanConfig,
    ScanRequest,
    ScanSummary,
    SelectRequest,
    TargetProfile,
)
from argus_curator.selection import (
    decide_selection,
    likeness_problem,
    plan_selection,
    scan_references,
    selection_problem,
)
from argus_curator.store import ScanStore


def _unit(*xs: float) -> np.ndarray:
    v = np.asarray(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


def _row(rel: str, score: float = 0.9, distance: float | None = None, passed: bool = True, **kw) -> ImageResult:
    return ImageResult(rel_path=rel, abs_path=f"/d/{rel}", passed=passed, score=score, likeness_distance=distance, **kw)


def _summary(
    results: list[ImageResult],
    *,
    enabled: bool = True,
    folder: str = "/d",
    source: str | None = "reference",
    skipped: str | None = None,
    cfg: EmbeddingConfig | None = None,
) -> ScanSummary:
    return ScanSummary(
        scan_id="s1",
        folder=folder,
        target_profile=TargetProfile(),
        config=ScanConfig(),
        faces_config=FaceConfig(),
        embeddings_config=cfg or EmbeddingConfig(enabled=enabled),
        likeness_source=source if enabled else None,
        likeness_skipped=skipped,
        total=len(results),
        passed=sum(r.passed for r in results),
        rejected=sum(not r.passed for r in results),
        duplicates=0,
        similar_clusters=0,
        results=results,
    )


def _plan(results: list[ImageResult], **fields: object):
    return plan_selection(results, SelectRequest(min_score=0.0, **fields).as_export_request(), 0.4)


# -- the measurement -----------------------------------------------------------------------------------


def test_distance_is_to_the_mean_of_the_reference_images() -> None:
    results = [_row("a.png"), _row("b.png"), _row("ref1.png"), _row("ref2.png")]
    vectors = {
        "a.png": _unit(1, 1, 0),
        "b.png": _unit(0, 0, 1),
        "ref1.png": _unit(1, 0.2, 0),
        "ref2.png": _unit(0.2, 1, 0),
    }
    cfg = EmbeddingConfig(reference_paths=["ref1.png", "ref2.png"])
    assert measure_likeness(results, vectors, cfg) == ("reference", None)
    by = {r.rel_path: r.likeness_distance for r in results}
    assert by["a.png"] == pytest.approx(0.0, abs=1e-4)  # on the references' mean
    assert by["b.png"] == pytest.approx(1.0, abs=1e-4)  # orthogonal to it
    assert all(isinstance(d, float) for d in by.values())


def test_a_rejected_reference_image_counts_but_gets_no_distance() -> None:
    ref = _row("ref.png", passed=False, reject_reason="blurry")
    results = [_row("a.png"), ref]
    vectors = {"a.png": _unit(1, 0), "ref.png": _unit(1, 0)}
    assert measure_likeness(results, vectors, EmbeddingConfig(reference_paths=["ref.png"])) == ("reference", None)
    assert results[0].likeness_distance == 0.0
    assert ref.likeness_distance is None and not ref.passed


def test_a_supplied_reference_embedding_is_the_subject() -> None:
    results = [_row("a.png"), _row("b.png")]
    vectors = {"a.png": _unit(0, 1), "b.png": _unit(1, 0)}
    cfg = EmbeddingConfig(reference_embedding=[3.0, 0.0])  # any length-matched vector; normalised here
    assert measure_likeness(results, vectors, cfg) == ("reference", None)
    assert [r.likeness_distance for r in results] == [pytest.approx(1.0), 0.0]


def test_without_a_reference_the_subject_is_the_sets_trimmed_centroid() -> None:
    subject = [_unit(1, 0.05 * i, 0) for i in range(6)]
    stray = [_unit(0, 0, 1), _unit(0, 0.1, 1)]
    results = [_row(f"s{i}.png") for i in range(6)] + [_row("x0.png"), _row("x1.png")]
    vectors = {r.rel_path: v for r, v in zip(results, subject + stray, strict=True)}
    assert measure_likeness(results, vectors, EmbeddingConfig(enabled=True)) == ("centroid", None)
    near = [r.likeness_distance for r in results[:6]]
    far = [r.likeness_distance for r in results[6:]]
    assert max(near) < 0.05 and min(far) > 0.9  # the strays barely move the centre
    plain = embeddings.normalise(np.stack(subject + stray).mean(axis=0))
    assert float(set_centroid(subject + stray) @ _unit(1, 0.1, 0)) > float(plain @ _unit(1, 0.1, 0))


@pytest.mark.parametrize(
    ("cfg", "why"),
    [
        (EmbeddingConfig(reference_paths=["gone.png"]), "no reference image is in the scan"),
        (EmbeddingConfig(reference_paths=["cut.png"]), "no reference image could be embedded"),
        (EmbeddingConfig(reference_embedding=[1.0, 0.0, 0.0]), "reference_embedding has 3 values"),
    ],
)
def test_an_unusable_reference_measures_nothing_and_never_falls_back(cfg: EmbeddingConfig, why: str) -> None:
    results = [_row("a.png"), _row("b.png"), _row("cut.png")]
    vectors = {"a.png": _unit(1, 0), "b.png": _unit(0, 1)}
    source, skipped = measure_likeness(results, vectors, cfg)
    assert source is None and skipped is not None and why in skipped
    assert all(r.likeness_distance is None for r in results)  # not the centroid's instead


def test_nothing_embedded_measures_nothing() -> None:
    assert measure_likeness([_row("a.png")], {}, EmbeddingConfig(enabled=True)) == (
        None,
        "no image could be embedded",
    )
    assert measure_likeness([_row("a.png")], {"a.png": _unit(1)}, EmbeddingConfig()) == (None, None)


def test_a_reference_turns_embeddings_on_and_takes_one_kind() -> None:
    assert EmbeddingConfig(reference_paths=["r.png"]).enabled
    assert EmbeddingConfig(reference_embedding=[1.0]).enabled
    with pytest.raises(ValidationError, match="not both"):
        EmbeddingConfig(reference_paths=["r.png"], reference_embedding=[1.0])
    with pytest.raises(ValidationError, match="finite and not all zero"):
        EmbeddingConfig(reference_embedding=[0.0, 0.0])
    assert ScanRequest(folder="x", embeddings={"reference_paths": ["r.png"]}).embeddings.enabled


# -- never stored ------------------------------------------------------------------------------------


def test_a_scan_summary_never_holds_the_likeness_reference_embedding() -> None:
    cfg = EmbeddingConfig(reference_embedding=[0.6, 0.8])
    summary = _summary([_row("a.png", distance=0.25)], cfg=cfg)
    assert summary.embeddings_config.reference_embedding is None
    assert cfg.reference_embedding == [0.6, 0.8]  # the scan's own config is left as it was
    for dumped in (summary.model_dump_json(), summary.model_dump_json(context=STORE_CONTEXT)):
        assert "0.6" not in dumped and "0.8" not in dumped
        assert '"likeness_distance":0.25' in dumped  # the distance is what persists
    stored = json.loads(summary.model_dump_json(context=STORE_CONTEXT))
    stored["embeddings_config"]["reference_embedding"] = [0.6, 0.8]  # even a cache someone edited
    assert ScanSummary.model_validate(stored).embeddings_config.reference_embedding is None


class _Input:
    name = "input"


class _Session:
    """A fake image tower: each image's vector is its mean colour (and a constant)."""

    def get_inputs(self) -> list[_Input]:
        return [_Input()]

    def run(self, _outputs, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        x = feeds["input"]
        return [np.concatenate([x.mean(axis=(2, 3)), np.zeros((len(x), 1))], axis=1)]


def _png(rgb: tuple[int, int, int], size: tuple[int, int] = (600, 600)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, rgb).save(buf, "PNG")
    return buf.getvalue()


def test_scan_records_distances_on_rows_and_keeps_no_vector_of_a_rejected_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(embeddings, "get_session", lambda provs: _Session())
    items = [
        ("red.png", "/d/red", _png((250, 10, 10))),
        ("blue.png", "/d/blue", _png((10, 10, 250))),
        ("anchor.png", "/d/anchor", _png((250, 10, 10), size=(40, 40))),  # too small: rejected
    ]
    summary = scanner.scan_items(
        items,
        TargetProfile(),
        ScanConfig(blur_threshold=0.0),
        FaceConfig(),
        embed_cfg=EmbeddingConfig(reference_paths=["anchor.png"], device="cpu"),
    )
    by = {r.rel_path: r for r in summary.results}
    assert not by["anchor.png"].passed and by["anchor.png"].likeness_distance is None
    assert summary.likeness_source == "reference" and summary.likeness_skipped is None
    assert by["red.png"].likeness_distance < by["blue.png"].likeness_distance
    assert set(summary.embeddings) == {"red.png", "blue.png"}  # the anchor's vector is gone

    off = scanner.scan_items(items, TargetProfile(), ScanConfig(blur_threshold=0.0), FaceConfig())
    assert off.likeness_source is None and off.likeness_skipped is None
    assert all(r.likeness_distance is None for r in off.results)


# -- rank and filter ---------------------------------------------------------------------------------


def test_likeness_rank_puts_the_closest_first_and_unmeasured_last() -> None:
    rows = [_row("far.png", 0.99, 0.6), _row("none.png", 0.95), _row("near.png", 0.7, 0.1), _row("mid.png", 0.8, 0.3)]
    plan = _plan(rows, rank="likeness")
    assert [r.rel_path for r in plan.chosen] == ["near.png", "mid.png", "far.png", "none.png"]


def test_likeness_rank_ties_fall_back_to_score_then_path() -> None:
    rows = [_row("b.png", 0.7, 0.2), _row("c.png", 0.9, 0.2), _row("a.png", 0.7, 0.2)]
    assert [r.rel_path for r in _plan(rows, rank="likeness").chosen] == ["c.png", "a.png", "b.png"]


def test_max_keep_under_likeness_keeps_the_closest() -> None:
    rows = [_row(f"{i}.png", 0.9, i / 10) for i in range(8)]
    plan = _plan(rows, rank="likeness", max_keep=3, diversity_weight=0.0)
    assert [r.rel_path for r in plan.chosen] == ["0.png", "1.png", "2.png"]


def test_the_ceiling_keeps_out_the_far_and_the_unmeasured() -> None:
    rows = [_row("near.png", distance=0.1), _row("far.png", distance=0.5), _row("none.png")]
    plan = _plan(rows, max_likeness_distance=0.3)
    assert [r.rel_path for r in plan.chosen] == ["near.png"]
    assert plan.keep_reason["far.png"] == plan.keep_reason["none.png"] == "likeness-distance"


def test_split_by_similar_group_keeps_near_duplicates_in_one_part() -> None:
    rows = []
    for g in range(1, 6):
        for k in range(2):
            rows.append(_row(f"g{g}-{k}.png", 0.9, (g + k / 10) / 10, similar_group=g, group_size=2))
    plan = _plan(rows, rank="likeness", split={"held_out": 4, "group_by": "similar_group"})
    parts: dict[int, set[str]] = {}
    for r in plan.chosen:
        parts.setdefault(r.similar_group, set()).add(plan.split[r.rel_path])
    assert all(len(p) == 1 for p in parts.values())
    assert sum(p == "held_out" for p in plan.split.values()) == 4


# -- refused, never ignored --------------------------------------------------------------------------


@pytest.mark.parametrize("fields", [{"rank": "likeness"}, {"max_likeness_distance": 0.4}])
def test_likeness_is_refused_on_a_scan_without_distances(fields: dict) -> None:
    req = ExportRequest(dest=".", **fields)
    measured = [_row("a.png", distance=0.2)]
    off = _summary([_row("a.png")], enabled=False)
    assert "rescan with embeddings on" in likeness_problem(req, off)
    skipped = _summary([_row("a.png")], source=None, skipped="no reference image is in the scan (r.png)")
    assert "no reference image is in the scan" in likeness_problem(req, skipped)
    # Embeddings on, but cached before likeness existed: no row has a distance.
    legacy = _summary([_row("a.png")], source=None)
    assert "has none" in likeness_problem(req, legacy)
    assert selection_problem(req, off) == likeness_problem(req, off)
    assert likeness_problem(req, _summary(measured)) is None
    assert likeness_problem(ExportRequest(dest="."), off) is None


def test_a_speech_request_refuses_the_likeness_fields() -> None:
    from argus_curator.speech_select import IMAGE_ONLY_FIELDS

    assert {"max_likeness_distance", "exclude_references"} <= set(IMAGE_ONLY_FIELDS)


# -- reference images out of the selection -----------------------------------------------------------


def test_exclude_references_keeps_the_anchors_out_of_the_cap_and_the_split() -> None:
    rows = [_row("anchor.png", distance=0.0)] + [_row(f"{i}.png", distance=0.1 + i / 10) for i in range(5)]
    summary = _summary(rows, cfg=EmbeddingConfig(reference_paths=["anchor.png"]))
    refs = scan_references(summary)
    assert refs == {"anchor.png"}
    req = SelectRequest(
        min_score=0.0, rank="likeness", exclude_references=True, max_keep=3, diversity_weight=0.0, split={"held_out": 1}
    ).as_export_request()
    plan = plan_selection(rows, req, 0.4, references=refs)
    assert [r.rel_path for r in plan.chosen] == ["0.png", "1.png", "2.png"]
    assert plan.keep_reason["anchor.png"] == "reference" and "anchor.png" not in plan.split
    # Without the flag the anchor is the closest image, and takes a place.
    assert _plan(rows, rank="likeness", max_keep=3, diversity_weight=0.0).chosen[0].rel_path == "anchor.png"


def test_exclude_references_never_silently_does_nothing() -> None:
    req = ExportRequest(dest=".", exclude_references=True)
    with pytest.raises(ValueError, match="scan_references"):
        decide_selection([_row("a.png")], req, 0.4)  # a caller that forgot them
    none_named = _summary([_row("a.png", distance=0.1)])
    assert "named no reference images" in selection_problem(req, none_named)
    faces = _summary([_row("a.png", distance=0.1)])
    faces.faces_config = FaceConfig(reference_paths=["a.png"])
    assert selection_problem(req, faces) is None and scan_references(faces) == {"a.png"}


def test_select_and_export_keep_the_same_rows_in_the_same_order_under_likeness(tmp_path: Path) -> None:
    rows = []
    for i in range(8):
        rel = f"img{i}.png"
        Image.new("RGB", (8, 8), (i, 0, 0)).save(tmp_path / rel)
        rows.append(
            ImageResult(
                rel_path=rel, abs_path=str(tmp_path / rel), passed=True, score=0.9, likeness_distance=(7 - i) / 10
            )
        )
    fields = {"rank": "likeness", "max_likeness_distance": 0.55}
    plan = _plan(rows, **fields)
    dest = tmp_path / "out"
    export_selection(_summary(rows, folder=str(tmp_path)), ExportRequest(dest=str(dest), min_score=0.0, **fields))
    manifest = [json.loads(line)["rel_path"] for line in (dest / "manifest.jsonl").read_text().splitlines()]
    assert manifest == [r.rel_path for r in plan.chosen] == [f"img{i}.png" for i in range(7, 1, -1)]


# -- HTTP --------------------------------------------------------------------------------------------


def test_select_route_ranks_by_recorded_distances_after_a_restart_and_refuses_without(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from argus_curator.server import create_app

    source = tmp_path / "src"
    source.mkdir()
    exports = tmp_path / "exports"
    exports.mkdir()
    rows = [_row("anchor.png", 0.9, distance=0.0)] + [_row(f"{i}.png", 0.9, distance=i / 10) for i in range(6)]
    # Saved without its vectors, as after a restart: the distances alone serve likeness.
    cfg = EmbeddingConfig(reference_paths=["anchor.png"])
    ScanStore(str(tmp_path / "cache")).save(_summary(rows, folder=str(source), cfg=cfg))
    client = TestClient(
        create_app(cache_dir=str(tmp_path / "cache"), source_root=str(source), export_root=str(exports))
    )
    body = {
        "min_score": 0.0,
        "rank": "likeness",
        "max_likeness_distance": 0.35,
        "exclude_references": True,
        "split": {"held_out": 1},
    }
    resp = client.post("/scan/s1/select", json=body)
    assert resp.status_code == 200, resp.text
    out = resp.json()
    assert out["rank"] == "likeness" and out["kept"] == 4 and out["train"] == 3 and out["held_out"] == 1
    assert [r["rel_path"] for r in out["rows"][:4]] == ["0.png", "1.png", "2.png", "3.png"]
    assert {r["rel_path"]: r["keep_reason"] for r in out["rows"][4:]} == {
        "anchor.png": "reference",
        "4.png": "likeness-distance",
        "5.png": "likeness-distance",
    }
    assert client.get("/scan/s1").json()["results"][4]["likeness_distance"] == 0.3

    ScanStore(str(tmp_path / "cache")).save(_summary([_row("a.png")], enabled=False, folder=str(source)))
    for route, payload in (
        ("/scan/s1/select", {"rank": "likeness"}),
        ("/export", {"scan_id": "s1", "dest": "out", "max_likeness_distance": 0.4}),
    ):
        resp = client.post(route, json=payload)
        assert resp.status_code == 400 and "embeddings" in resp.json()["detail"]
    assert not (exports / "out").exists()
