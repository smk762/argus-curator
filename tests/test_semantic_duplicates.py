"""Semantic near-duplicates from CLIP embeddings (#196).

No model and no GPU: grouping is fed synthetic unit vectors, and the scan phase
runs on the fake session from test_embeddings.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from test_embeddings import _png, _Session

from argus_curator import embeddings, scanner
from argus_curator.models import (
    EmbeddingConfig,
    FaceConfig,
    ImageResult,
    ScanConfig,
    ScanRequest,
    ScanSummary,
    TargetProfile,
)
from argus_curator.server.app import create_app
from argus_curator.store import ScanStore

# Hashes 64 bits apart: pHash never links them.
H0 = "0000000000000000"
H1 = "ffffffffffffffff"
SEM = ScanConfig(semantic_duplicates=True, semantic_duplicate_cosine=0.95)


def _img(rel: str, score: float, phash: str = H0) -> ImageResult:
    return ImageResult(rel_path=rel, abs_path=f"/d/{rel}", score=score, passed=True, phash=phash)


def _unit(*xs: float) -> np.ndarray:
    v = np.array(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


def _grouped(results: list[ImageResult]) -> dict[str, tuple[str | None, str | None]]:
    return {r.rel_path: (r.duplicate_of, r.duplicate_kind) for r in results}


def _mixed_pool() -> tuple[list[ImageResult], dict[str, np.ndarray]]:
    # a/b: a recolour pHash misses (far hashes, near vectors).
    # c/d: a re-encode pHash catches, vectors far apart.
    # e/f: both tests hold.  g: alone.
    pool = [
        _img("a.jpg", 0.90, H0),
        _img("b.jpg", 0.95, H1),
        _img("c.jpg", 0.80, H0),
        _img("d.jpg", 0.70, H0),
        _img("e.jpg", 0.60, H1),
        _img("f.jpg", 0.50, H1),
        _img("g.jpg", 0.40, H1),
    ]
    vecs = {
        "a.jpg": _unit(1, 0.05, 0, 0, 0),
        "b.jpg": _unit(1, 0, 0, 0, 0),
        "c.jpg": _unit(0, 1, 0, 0, 0),
        "d.jpg": _unit(0, 0, 1, 0, 0),
        "e.jpg": _unit(0, 0, 0, 1, 0),
        "f.jpg": _unit(0, 0, 0, 1, 0.05),
        "g.jpg": _unit(0, 0, 0, 0, 1),
    }
    return pool, vecs


def test_semantic_duplicates_are_off_by_default() -> None:
    pool, vecs = _mixed_pool()
    assert ScanConfig().semantic_duplicates is False
    scanner._mark_duplicates(pool, ScanConfig(), vecs)
    # pHash alone: a/c/d share H0 (b, the best H1, heads e/f/g).
    assert _grouped(pool)["a.jpg"] == (None, None)
    assert {r.rel_path for r in pool if r.duplicate_kind == "semantic"} == set()


def test_union_of_both_tests_with_one_keep_best_rule_and_a_kind_per_link() -> None:
    pool, vecs = _mixed_pool()
    pool = [r for r in pool if r.rel_path != "g.jpg"]
    scanner._mark_duplicates(
        pool, ScanConfig(cluster_distance=4, semantic_duplicates=True, semantic_duplicate_cosine=0.95), vecs
    )
    got = _grouped(pool)
    assert got["b.jpg"] == (None, None)  # best of its group: the representative
    assert got["a.jpg"] == ("b.jpg", "semantic")  # 64 bits apart, cosine ~0.999
    # a joined b's group, so it is no representative: c shares a's hash but heads its own group.
    assert got["c.jpg"] == (None, None)
    assert got["d.jpg"] == ("c.jpg", "phash")
    assert got["e.jpg"] == ("b.jpg", "phash")
    # f is near e semantically, but e is no representative: f joins b, by pHash.
    assert got["f.jpg"] == ("b.jpg", "phash")


def test_kind_both_when_both_tests_hold_against_the_representative() -> None:
    pool = [_img("a.jpg", 0.9), _img("b.jpg", 0.8)]
    vecs = {"a.jpg": _unit(1, 0), "b.jpg": _unit(1, 0.01)}
    scanner._mark_duplicates(pool, SEM, vecs)
    assert _grouped(pool)["b.jpg"] == ("a.jpg", "both")
    assert scanner._assign_clusters(pool) == 1 and pool[0].is_representative and not pool[1].is_representative


def test_semantic_grouping_runs_with_phash_grouping_off() -> None:
    pool = [_img("a.jpg", 0.9), _img("b.jpg", 0.8), _img("c.jpg", 0.7)]
    vecs = {"a.jpg": _unit(1, 0), "b.jpg": _unit(1, 0.01), "c.jpg": _unit(0, 1)}
    scanner._mark_duplicates(
        pool, ScanConfig(cluster_distance=-1, semantic_duplicates=True, semantic_duplicate_cosine=0.95), vecs
    )
    assert _grouped(pool) == {"a.jpg": (None, None), "b.jpg": ("a.jpg", "semantic"), "c.jpg": (None, None)}


def test_duplicate_of_is_always_a_direct_match_never_a_chain() -> None:
    # a~b and b~c are over the threshold, a~c is not: c is not pulled into a's group via b.
    pool = [_img("a.jpg", 0.9, H0), _img("b.jpg", 0.8, H1), _img("c.jpg", 0.7, "00000000ffffffff")]
    theta = np.radians(15)
    vecs = {
        "a.jpg": _unit(1, 0),
        "b.jpg": _unit(np.cos(theta), np.sin(theta)),
        "c.jpg": _unit(np.cos(2 * theta), np.sin(2 * theta)),
    }
    scanner._mark_duplicates(pool, ScanConfig(semantic_duplicates=True, semantic_duplicate_cosine=0.95), vecs)
    assert _grouped(pool) == {"a.jpg": (None, None), "b.jpg": ("a.jpg", "semantic"), "c.jpg": (None, None)}


def test_an_unembedded_image_has_no_semantic_link() -> None:
    pool = [_img("a.jpg", 0.9, H0), _img("b.jpg", 0.8, H1), _img("c.jpg", 0.7, H1)]
    scanner._mark_duplicates(pool, SEM, {"a.jpg": _unit(1, 0), "c.jpg": _unit(1, 0)})
    # b has no embedding: not semantically anything's duplicate, and c matches a (not b) semantically.
    assert _grouped(pool) == {"a.jpg": (None, None), "b.jpg": (None, None), "c.jpg": ("a.jpg", "semantic")}


def test_a_rejected_image_is_never_grouped() -> None:
    pool = [_img("a.jpg", 0.9, H0), _img("b.jpg", 0.95, H1)]
    pool[1].passed = False
    scanner._mark_duplicates(pool, SEM, {"a.jpg": _unit(1, 0), "b.jpg": _unit(1, 0)})
    assert _grouped(pool) == {"a.jpg": (None, None), "b.jpg": (None, None)}


def test_phash_only_grouping_is_unchanged_by_held_embeddings() -> None:
    rng = np.random.default_rng(196)
    hashes = [f"{int(x):016x}" for x in rng.integers(0, 2**63, 40)]
    hashes += [h[:-1] + "0" for h in hashes[:10]]  # near pHash twins

    def pool() -> list[ImageResult]:
        return [_img(f"{i}.jpg", float(rng.random()), h) for i, h in enumerate(hashes)]

    plain, with_vecs = pool(), pool()
    for a, b in zip(plain, with_vecs, strict=True):
        b.score = a.score
    vecs = {r.rel_path: _unit(1, 0) for r in with_vecs}  # all identical: would all group if read
    scanner._mark_duplicates(plain, ScanConfig())
    scanner._mark_duplicates(with_vecs, ScanConfig(), vecs)
    assert _grouped(plain) == _grouped(with_vecs)
    assert any(r.is_duplicate for r in plain) and all(r.duplicate_kind in (None, "phash") for r in plain)


# -- off unless the vectors exist: refused, never silently pHash-only ------------------------


def test_a_scan_asking_for_semantic_duplicates_without_embeddings_is_refused() -> None:
    with pytest.raises(ValidationError, match="needs the scan's embeddings"):
        ScanRequest(folder="", config={"semantic_duplicates": True})
    assert ScanRequest(folder="", config={"semantic_duplicates": True}, embeddings={"enabled": True})
    # The speech refusal, not the missing-embeddings one (whose message also names semantic_duplicates).
    with pytest.raises(
        ValidationError, match=r"a speech scan runs no image checks: turn off config\.semantic_duplicates"
    ):
        ScanRequest(
            folder="",
            target_profile={"modality": "speech"},
            config={"semantic_duplicates": True},
        )
    with pytest.raises(ValueError, match="needs the scan's embeddings"):
        scanner.scan_items([], TargetProfile(), SEM, FaceConfig())
    with pytest.raises(ValidationError):
        ScanConfig(semantic_duplicates=True, semantic_duplicate_cosine=1.5)
    # The threshold alone turns nothing on: the switch is semantic_duplicates.
    assert ScanRequest(folder="", config={"semantic_duplicate_cosine": 0.9})


def test_scan_route_refuses_semantic_duplicates_without_embeddings(dataset: Path, tmp_path: Path) -> None:
    client = TestClient(create_app(cache_dir=str(tmp_path / "cache"), source_root=str(dataset)))
    resp = client.post("/scan/folder", json={"folder": "", "config": {"semantic_duplicates": True}})
    assert resp.status_code == 422 and "embeddings" in resp.text


def test_the_scan_groups_semantic_duplicates_and_keeps_only_plain_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(embeddings, "get_session", lambda provs: _Session())
    # The fake session embeds the mean colour: reds 200 and 205 are near, 20 is not.
    items = [
        ("a.png", "/d/a", _png(200, (600, 600))),
        ("b.png", "/d/b", _png(205, (640, 600))),
        ("c.png", "/d/c", _png(20, (600, 600))),
    ]
    summary = scanner.scan_items(
        items,
        TargetProfile(),
        ScanConfig(blur_threshold=0.0, cluster_distance=-1, semantic_duplicates=True, semantic_duplicate_cosine=0.999),
        FaceConfig(),
        embed_cfg=EmbeddingConfig(enabled=True, device="cpu"),
    )
    kinds = {r.rel_path: r.duplicate_kind for r in summary.results}
    assert sorted(kinds.values(), key=str) == [None, None, "semantic"] and kinds["c.png"] is None
    assert summary.duplicates == 1 and summary.similar_clusters == 1
    dup = next(r for r in summary.results if r.is_duplicate)
    assert dup.duplicate_of in {"a.png", "b.png"} and dup.duplicate_of != dup.rel_path
    # The result fields are plain values; the vectors never reach the dump.
    dumped = summary.model_dump_json()
    assert '"duplicate_kind":"semantic"' in dumped and "_embeddings" not in dumped
    assert ScanSummary.model_validate_json(dumped).embeddings == {}


# -- regrouping after rows are dropped -------------------------------------------------------


def _scanned(vecs: dict[str, np.ndarray]) -> ScanSummary:
    pool = [_img("a.jpg", 0.9, H0), _img("b.jpg", 0.8, H1), _img("c.jpg", 0.7, "00000000ffffffff")]
    pool.append(_img("z.jpg", 0.1, "0f0f0f0f0f0f0f0f"))
    scanner._mark_duplicates(pool, SEM, vecs)
    summary = ScanSummary(
        scan_id="s1",
        folder="/d",
        target_profile=TargetProfile(),
        config=SEM,
        faces_config=FaceConfig(),
        embeddings_config=EmbeddingConfig(enabled=True),
        total=len(pool),
        passed=len(pool),
        rejected=0,
        duplicates=sum(r.is_duplicate for r in pool),
        similar_clusters=scanner._assign_clusters(pool),
        results=pool,
    )
    summary.embeddings.update(vecs)
    return summary


# b and c both duplicate a; b and c are near each other too. z is unrelated.
VECS = {
    "a.jpg": _unit(1, 0.1, 0.1),
    "b.jpg": _unit(1, 0.12, 0.1),
    "c.jpg": _unit(1, 0.1, 0.12),
    "z.jpg": _unit(0, 0, 1),
}


def test_drop_results_regroups_with_the_held_embeddings() -> None:
    summary = _scanned(VECS)
    assert _grouped(summary.results)["c.jpg"] == ("a.jpg", "semantic")
    scanner.drop_results(summary, ["a.jpg"])
    assert _grouped(summary.results) == {"b.jpg": (None, None), "c.jpg": ("b.jpg", "semantic"), "z.jpg": (None, None)}
    assert summary.duplicates == 1 and summary.semantic_groups_stale is False


def test_drop_results_without_embeddings_keeps_the_recorded_semantic_links(tmp_path: Path) -> None:
    store = ScanStore(tmp_path)
    store.save(_scanned(VECS))
    before = _grouped(_scanned(VECS).results)

    restarted = ScanStore(tmp_path).load("s1")  # another process: the vectors are gone
    assert restarted is not None and restarted.embeddings == {}
    scanner.drop_results(restarted, ["z.jpg"])
    # Not silently turned pHash-only: the semantic groups stand.
    assert _grouped(restarted.results) == {k: v for k, v in before.items() if k != "z.jpg"}
    assert restarted.semantic_groups_stale is False

    # The representative goes: b and c can no longer be compared, and the summary says so.
    scanner.drop_results(restarted, ["a.jpg"])
    assert _grouped(restarted.results) == {"b.jpg": (None, None), "c.jpg": (None, None)}
    assert restarted.semantic_groups_stale is True
    assert ScanSummary.model_validate_json(restarted.model_dump_json()).semantic_groups_stale is True


def test_drop_results_with_the_embeddings_held_is_never_stale(tmp_path: Path) -> None:
    store = ScanStore(tmp_path)
    store.save(_scanned(VECS))
    current = store.load("s1")
    assert current is not None and set(current.embeddings) == set(VECS)
    scanner.drop_results(current, ["a.jpg"])
    store.save(current)
    assert current.semantic_groups_stale is False and set(store.held_embeddings("s1")) == {"b.jpg", "c.jpg", "z.jpg"}
