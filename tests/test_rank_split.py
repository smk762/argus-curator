"""/select's ranking, cap and train/held-out split, shared with export (#183)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image
from pydantic import ValidationError

from argus_curator.export import export_selection
from argus_curator.models import (
    ExportRequest,
    FaceConfig,
    FaceDetection,
    ImageResult,
    ScanConfig,
    ScanSummary,
    SelectRequest,
    SplitConfig,
    TargetProfile,
)
from argus_curator.selection import (
    SplitError,
    plan_selection,
    rank_split_problem,
    split_held_out,
)


def _row(
    rel: str, score: float, distance: float | None = None, cluster: str | None = "face_1", root: str = ""
) -> ImageResult:
    faces = []
    if cluster is not None:
        faces = [
            FaceDetection(
                bbox=[0.0, 0.0, 300.0, 300.0],
                det_score=0.9,
                primary=True,
                cluster_id=cluster,
                cluster_distance=distance,
            )
        ]
    return ImageResult(
        rel_path=rel,
        abs_path=f"{root}/{rel}",
        passed=True,
        score=score,
        faces=faces,
        face_count=len(faces),
        primary_face_cluster=cluster,
    )


def _summary(results: list[ImageResult], folder: str = "/x", faces: bool = True) -> ScanSummary:
    return ScanSummary(
        scan_id="s",
        folder=folder,
        target_profile=TargetProfile(),
        config=ScanConfig(),
        faces_config=FaceConfig(enabled=faces),
        total=len(results),
        passed=len(results),
        rejected=0,
        duplicates=0,
        similar_clusters=0,
        results=results,
    )


def _plan(results: list[ImageResult], **fields: object):
    req = SelectRequest(min_score=0.0, **fields).as_export_request()
    return plan_selection(results, req, 0.4)


# -- ranking -------------------------------------------------------------------------------------------


def test_kept_images_come_back_ranked_by_score_not_scan_order() -> None:
    rows = [_row("a.png", 0.6), _row("b.png", 0.9), _row("c.png", 0.7)]
    assert [r.rel_path for r in _plan(rows).chosen] == ["b.png", "c.png", "a.png"]


def test_identity_rank_puts_the_closest_face_first_and_no_face_last() -> None:
    rows = [
        _row("best_score.png", 0.95, distance=0.3),
        _row("closest.png", 0.7, distance=0.05),
        _row("faceless.png", 0.99, cluster=None),
        _row("middle.png", 0.8, distance=0.2),
    ]
    chosen = _plan(rows, rank="identity").chosen
    assert [r.rel_path for r in chosen] == ["closest.png", "middle.png", "best_score.png", "faceless.png"]


def test_identity_rank_needs_a_face_scan() -> None:
    req = ExportRequest(dest=".", rank="identity")
    assert "face detection" in rank_split_problem(req, _summary([], faces=False))
    assert rank_split_problem(req, _summary([], faces=True)) is None
    grouped = ExportRequest(dest=".", split=SplitConfig(held_out=1, group_by="face_cluster"))
    assert "face detection" in rank_split_problem(grouped, _summary([], faces=False))


# -- max_keep ------------------------------------------------------------------------------------------


def test_max_keep_trims_select_to_the_best_by_rank() -> None:
    rows = [_row(f"{i}.png", 0.5 + i / 100, distance=1.0 - i / 10) for i in range(8)]
    plan = _plan(rows, max_keep=3, diversity_weight=0.0, rank="identity")
    assert [r.rel_path for r in plan.chosen] == ["7.png", "6.png", "5.png"]
    assert sum(1 for v in plan.keep_reason.values() if v == "diversity-trimmed") == 5


def test_select_and_export_keep_the_same_rows_in_the_same_order(tmp_path: Path) -> None:
    rows = _files(tmp_path, 10)
    fields = {"max_keep": 6, "diversity_weight": 0.0, "rank": "identity"}
    plan = _plan(rows, **fields)
    dest = tmp_path / "out"
    export_selection(_summary(rows, str(tmp_path)), ExportRequest(dest=str(dest), min_score=0.0, **fields))
    manifest = [json.loads(line)["rel_path"] for line in (dest / "manifest.jsonl").read_text().splitlines()]
    assert manifest == [r.rel_path for r in plan.chosen]


# -- the split -----------------------------------------------------------------------------------------


def test_spread_split_matches_mints_rule() -> None:
    """One image per group: held-out images at int((j + 0.5) * n / k) through the ranking."""
    n, k = 12, 4
    parts = split_held_out([f"k{i}" for i in range(n)], SplitConfig(held_out=k))
    held = [i for i, p in enumerate(parts) if p == "held_out"]
    assert held == sorted({int((j + 0.5) * n / k) for j in range(k)})


def test_no_group_lands_in_both_parts() -> None:
    keys = [f"id{i % 5}" for i in range(23)] + [None, None]
    for method in ("spread", "random"):
        for seed in range(20):
            cfg = SplitConfig(held_out_fraction=0.3, group_by="face_cluster", method=method, seed=seed)
            parts = split_held_out(keys, cfg)
            by_key: dict[str, set[str]] = {}
            for key, part in zip(keys, parts, strict=True):
                if key is not None:
                    by_key.setdefault(key, set()).add(part)
            assert all(len(p) == 1 for p in by_key.values()), (method, seed, by_key)
            assert "train" in parts and "held_out" in parts
            assert parts.count("held_out") <= round(0.3 * len(keys))


def test_split_is_deterministic_and_seeded() -> None:
    keys = [f"k{i}" for i in range(30)]
    cfg = SplitConfig(held_out=6, method="random", seed=7)
    assert split_held_out(keys, cfg) == split_held_out(keys, cfg)
    others = {tuple(split_held_out(keys, SplitConfig(held_out=6, method="random", seed=s))) for s in range(10)}
    assert len(others) > 1


def test_a_split_that_leaves_a_part_empty_is_refused() -> None:
    with pytest.raises(SplitError, match="none to train"):
        split_held_out(["a", "b"], SplitConfig(held_out=2))
    with pytest.raises(SplitError, match="at least two groups"):
        split_held_out(["a", "a", "a"], SplitConfig(held_out=1, group_by="face_cluster"))
    with pytest.raises(SplitError, match="no face_cluster group fits"):
        split_held_out(["a", "a", "b", "b"], SplitConfig(held_out=1, group_by="face_cluster"))


def test_split_config_needs_exactly_one_size() -> None:
    with pytest.raises(ValidationError):
        SplitConfig()
    with pytest.raises(ValidationError):
        SplitConfig(held_out=2, held_out_fraction=0.2)


def test_split_refuses_a_named_selection() -> None:
    with pytest.raises(ValidationError, match="named selection"):
        ExportRequest(dest=".", selection=["a.png"], split=SplitConfig(held_out=1))


def test_plan_splits_by_identity() -> None:
    rows = [_row(f"{c}{i}.png", 0.9 - i / 100, cluster=c) for c in ("a", "b", "c", "d") for i in range(3)]
    plan = _plan(rows, split={"held_out": 3, "group_by": "face_cluster"})
    held = {rel for rel, p in plan.split.items() if p == "held_out"}
    assert len(held) == 3 and len({rel[0] for rel in held}) == 1
    assert set(plan.split) == {r.rel_path for r in plan.chosen}


# -- export --------------------------------------------------------------------------------------------


def _files(root: Path, n: int, clusters: int = 0) -> list[ImageResult]:
    rows = []
    for i in range(n):
        rel = f"img{i:02d}.png"
        Image.new("RGB", (8, 8), (i, 0, 0)).save(root / rel)
        cluster = f"face_{i % clusters}" if clusters else "face_1"
        rows.append(_row(rel, 0.6 + i / 100, distance=0.5 - i / 100, cluster=cluster, root=str(root)))
    return rows


def test_export_writes_one_part_and_rows_name_it(tmp_path: Path) -> None:
    rows = _files(tmp_path, 10)
    summary = _summary(rows, str(tmp_path))
    split = SplitConfig(held_out=3)
    seen: dict[str, set[str]] = {}
    for part in ("train", "held_out"):
        dest = tmp_path / part
        result = export_selection(
            summary, ExportRequest(dest=str(dest), min_score=0.0, split=split, split_part=part, rank="identity")
        )
        manifest = [json.loads(line) for line in (dest / "manifest.jsonl").read_text().splitlines()]
        assert {row["split"] for row in manifest} == {part}
        seen[part] = set(result.selected_rel_paths)
    assert len(seen["held_out"]) == 3 and len(seen["train"]) == 7
    assert not seen["train"] & seen["held_out"]
    plan = _plan(rows, split={"held_out": 3}, rank="identity")
    assert seen["held_out"] == {rel for rel, p in plan.split.items() if p == "held_out"}


def test_export_without_a_split_writes_a_null_split(tmp_path: Path) -> None:
    rows = _files(tmp_path, 3)
    dest = tmp_path / "out"
    export_selection(_summary(rows, str(tmp_path)), ExportRequest(dest=str(dest), min_score=0.0))
    assert {json.loads(line)["split"] for line in (dest / "manifest.jsonl").read_text().splitlines()} == {None}


# -- HTTP ----------------------------------------------------------------------------------------------


@pytest.fixture
def served(tmp_path: Path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from argus_curator.server import create_app
    from argus_curator.store import ScanStore

    source = tmp_path / "src"
    source.mkdir()
    exports = tmp_path / "exports"
    exports.mkdir()
    rows = _files(source, 12, clusters=4)
    summary = _summary(rows, str(source))
    ScanStore(str(tmp_path / "cache")).save(summary)
    app = create_app(cache_dir=str(tmp_path / "cache"), source_root=str(source), export_root=str(exports))
    return TestClient(app), summary.scan_id, exports


def test_select_route_ranks_caps_and_splits(served) -> None:
    client, scan_id, _exports = served
    body = {
        "min_score": 0.0,
        "rank": "identity",
        "max_keep": 8,
        "diversity_weight": 0.0,
        "split": {"held_out": 2, "group_by": "face_cluster", "seed": 3, "method": "random"},
    }
    resp = client.post(f"/scan/{scan_id}/select", json=body)
    assert resp.status_code == 200, resp.text
    out = resp.json()
    assert out["kept"] == 8 and out["total"] == 12 and out["rank"] == "identity"
    kept = [r for r in out["rows"] if not r["keep_reason"]]
    assert [r["rank"] for r in kept] == list(range(1, 9))
    assert out["rows"][:8] == kept  # kept rows first, best first
    assert [r["rel_path"] for r in kept] == [f"img{i:02d}.png" for i in range(11, 3, -1)]
    assert all(r["rank"] is None and r["split"] is None for r in out["rows"][8:])
    assert out["held_out"] == 2 and out["train"] == 6
    clusters = {r["rel_path"]: f"face_{int(r['rel_path'][3:5]) % 4}" for r in kept}
    parts: dict[str, set[str]] = {}
    for r in kept:
        parts.setdefault(clusters[r["rel_path"]], set()).add(r["split"])
    assert all(len(p) == 1 for p in parts.values())
    assert client.post(f"/scan/{scan_id}/select", json=body).json() == out  # deterministic


def test_select_route_refuses_an_impossible_split(served) -> None:
    client, scan_id, _exports = served
    resp = client.post(f"/scan/{scan_id}/select", json={"min_score": 0.0, "split": {"held_out": 12}})
    assert resp.status_code == 400
    assert "split refused" in resp.json()["detail"]


def test_export_route_refuses_an_impossible_split(served) -> None:
    client, scan_id, exports = served
    resp = client.post("/export", json={"scan_id": scan_id, "dest": "out", "min_score": 0.0, "split": {"held_out": 12}})
    assert resp.status_code == 400
    assert not (exports / "out").exists()
