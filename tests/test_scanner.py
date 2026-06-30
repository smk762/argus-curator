"""Scanner: scoring, rel-path keying, dedup, and the summary contract."""

from __future__ import annotations

from pathlib import Path

from argus_curator import scan_folder
from argus_curator.models import FaceConfig, ScanConfig, TargetProfile


def test_relative_path_keying_no_clobber(dataset: Path) -> None:
    summary = scan_folder(dataset)
    rels = {r.rel_path for r in summary.results}
    # Same basename in two sub-folders must produce two distinct records.
    assert "personA/s1/img.png" in rels
    assert "personA/s2/img.png" in rels
    assert len(summary.results) == 5


def test_hard_filters_reject_blurry_and_tiny(dataset: Path) -> None:
    summary = scan_folder(dataset)
    by_rel = {r.rel_path: r for r in summary.results}
    assert by_rel["personB/tiny.png"].passed is False
    assert by_rel["personB/blurry.png"].passed is False
    assert by_rel["personA/s1/img.png"].passed is True
    assert summary.rejected == 2
    assert summary.passed == 3


def test_near_duplicate_clustering(dataset: Path) -> None:
    summary = scan_folder(dataset, cfg=ScanConfig(cluster_distance=10))
    by_rel = {r.rel_path: r for r in summary.results}
    img = by_rel["personA/s1/img.png"]
    dup = by_rel["personA/s1/dup.png"]
    # img and dup share a cluster; exactly one is the representative.
    assert img.similar_group == dup.similar_group
    assert img.group_size == 2
    assert (img.is_representative, dup.is_representative).count(True) == 1
    assert summary.duplicates == 1
    assert summary.similar_clusters == 1


def test_no_cluster_disables_grouping(dataset: Path) -> None:
    summary = scan_folder(dataset, cfg=ScanConfig(cluster_distance=-1))
    assert summary.duplicates == 0
    assert all(r.group_size == 1 for r in summary.results)


def test_scores_bounded_and_breakdown_present(dataset: Path) -> None:
    summary = scan_folder(dataset)
    for r in summary.results:
        assert 0.0 <= r.score <= 1.0
        if r.passed:
            assert r.score_breakdown
            assert "target_bonus" in r.score_breakdown


def test_target_profile_roundtrips_into_summary(dataset: Path) -> None:
    profile = TargetProfile(target_style="anime", target_category="wardrobe", target_backend="flux-dev-1")
    summary = scan_folder(dataset, profile=profile, faces_cfg=FaceConfig(enabled=False))
    assert summary.target_profile.target_style == "anime"
    assert summary.target_profile.target_category == "wardrobe"
