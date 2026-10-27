"""Scanner: scoring, rel-path keying, dedup, and the summary contract."""

from __future__ import annotations

from pathlib import Path

import pytest

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


def test_pending_checks_name_the_unfinished_ones_in_order() -> None:
    from argus_curator.models import GuardConfig, NsfwConfig
    from argus_curator.scanner import pending_checks, requested_checks

    assert pending_checks(["faces", "nsfw", "guard"], set()) == ["faces", "nsfw", "guard"]
    assert pending_checks(["faces", "nsfw", "guard"], {"faces"}) == ["nsfw", "guard"]
    assert pending_checks(["nsfw", "guard"], {"nsfw"}) == ["guard"]
    assert pending_checks(["guard"], {"guard", "nsfw"}) == []
    # Nothing configured: no check is requested, whatever faces say.
    assert requested_checks(FaceConfig(enabled=False), NsfwConfig(), GuardConfig()) == []
    assert requested_checks(FaceConfig(enabled=True), NsfwConfig(), GuardConfig()) == ["faces"]


def test_nsfw_runs_right_after_faces_and_chunks_say_what_is_still_ahead(
    dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A viewer needs the age and the exposed score to lift a blur, so NudeNet
    follows faces at once and nothing waits on scenes or garments."""
    from argus_cortex.moderation import NsfwResult

    from argus_curator import faces as faces_mod
    from argus_curator import nsfw as nsfw_mod
    from argus_curator.models import NsfwConfig

    def fake_faces(results, items, cfg, progress=None, on_done=None):
        passing = [r for r in results if r.passed]
        for i, r in enumerate(passing):
            if progress is not None:
                progress(i, len(passing))
            if on_done is not None:
                on_done(r, i + 1, len(passing))
        return faces_mod.DetectedFaces([], [], [])

    def fake_nsfw(results, items, cfg, progress=None, on_done=None):
        for i, r in enumerate(results):
            r.nsfw = NsfwResult()
            if progress is not None:
                progress(i, len(results))
            if on_done is not None:
                on_done(r, i + 1, len(results))
        return True

    monkeypatch.setattr(faces_mod, "detect_faces", fake_faces)
    monkeypatch.setattr(nsfw_mod, "check_images", fake_nsfw)
    monkeypatch.setattr(nsfw_mod, "resolve_enabled", lambda cfg: True)

    events: list[dict] = []
    chunks: list[dict] = []
    summary = scan_folder(
        dataset, faces_cfg=FaceConfig(enabled=True), nsfw_cfg=NsfwConfig(), progress=events.append, chunk=chunks.append
    )

    phases = [e["phase"] for e in events]
    first = {phase: phases.index(phase) for phase in ("scoring", "faces", "nsfw", "identities", "clustering")}
    assert first["scoring"] < first["faces"] < first["nsfw"] < first["identities"] < first["clustering"]
    # Progress events carry the same list the chunks do.
    assert all("pending_checks" in e for e in events if e["phase"] != "collecting")
    assert events[phases.index("nsfw")]["pending_checks"] == ["nsfw"]

    by_phase: dict[str, list[dict]] = {}
    for c in chunks:
        by_phase.setdefault(c["phase"], []).append(c)
    assert all(c["pending_checks"] == ["faces", "nsfw"] for c in by_phase["scoring"])
    assert by_phase["faces"][-1]["pending_checks"] == ["nsfw"]
    assert by_phase["nsfw"][-1]["pending_checks"] == []
    assert by_phase["identities"][-1]["pending_checks"] == []
    assert summary.nsfw_config.enabled is True
    assert all(r.nsfw is not None for r in summary.results)


def test_target_profile_roundtrips_into_summary(dataset: Path) -> None:
    profile = TargetProfile(target_style="anime", target_category="wardrobe", target_backend="flux-dev-1")
    summary = scan_folder(dataset, profile=profile, faces_cfg=FaceConfig(enabled=False))
    assert summary.target_profile.target_style == "anime"
    assert summary.target_profile.target_category == "wardrobe"
