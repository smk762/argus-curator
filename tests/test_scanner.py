"""Scanner: scoring, rel-path keying, dedup, and the summary contract."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter

from argus_curator import scan_folder
from argus_curator.models import FaceConfig, ScanConfig, TargetProfile
from argus_curator.scanner import _sharpness


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


def _scene(side: int) -> Image.Image:
    """The same flat-shaded scene drawn at *side* pixels: a sharp picture at any size."""
    s = side / 1024
    img = Image.new("RGB", (side, side), (235, 235, 235))
    draw = ImageDraw.Draw(img)
    rng = np.random.default_rng(1)
    for _ in range(60):
        x, y = rng.integers(0, 1024, 2)
        w, h = rng.integers(20, 200, 2)
        colour = tuple(int(v) for v in rng.integers(0, 255, 3))
        box = [x * s, y * s, (x + w) * s, (y + h) * s]
        if rng.random() < 0.5:
            draw.rectangle(box, fill=colour)
        else:
            draw.ellipse(box, fill=colour)
    return img


def test_sharpness_does_not_depend_on_resolution() -> None:
    """A sharp 4800px studio frame measured at native size scored about a
    quarter of its 2048px copy and fell under blur_threshold, so faces,
    MiVOLO ages and the quality score were never computed for it. The same
    scene must measure about the same at any size at or above 1024px."""
    at_1024 = _sharpness(_scene(1024))
    at_4096 = _sharpness(_scene(4096))
    assert at_1024 > 100.0
    assert abs(at_4096 - at_1024) / at_1024 < 0.15
    # Below 1024 the image is measured as it is.
    assert _sharpness(_scene(768)) > at_1024
    # The same blur, scaled with the picture, measures the same too.
    blurred_1024 = _sharpness(_scene(1024).filter(ImageFilter.GaussianBlur(radius=30)))
    blurred_4096 = _sharpness(_scene(4096).filter(ImageFilter.GaussianBlur(radius=120)))
    assert blurred_1024 < at_1024 / 3
    assert abs(blurred_4096 - blurred_1024) / blurred_1024 < 0.15


def test_large_sharp_images_pass_the_blur_gate(tmp_path: Path) -> None:
    _scene(4096).save(tmp_path / "large.png")
    _scene(1024).save(tmp_path / "small.png")
    by_rel = {r.rel_path: r for r in scan_folder(tmp_path).results}
    assert by_rel["large.png"].passed is True
    assert by_rel["small.png"].passed is True
    assert by_rel["large.png"].reject_reason is None
    assert by_rel["large.png"].sharpness > 100.0


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


def test_faces_and_nsfw_run_side_by_side_and_chunks_say_what_is_still_ahead(
    dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A viewer needs the age and the exposed score to lift a blur. Face
    detection and NudeNet run at the same time, each ahead of scenes, garments
    and SingGuard, and every chunk names the checks still unfinished."""
    import threading

    from argus_cortex.moderation import NsfwResult

    from argus_curator import faces as faces_mod
    from argus_curator import nsfw as nsfw_mod
    from argus_curator.models import NsfwConfig

    # Each fake waits for the other to have started. Run one after the other,
    # this would hang; the timeouts turn a hang into a failure.
    faces_started = threading.Event()
    nsfw_started = threading.Event()

    def fake_faces(results, items, cfg, progress=None, on_done=None):
        faces_started.set()
        assert nsfw_started.wait(5), "NudeNet did not run alongside face detection"
        passing = [r for r in results if r.passed]
        for i, r in enumerate(passing):
            if progress is not None:
                progress(i, len(passing))
            if on_done is not None:
                on_done(r, i + 1, len(passing))
        return faces_mod.DetectedFaces([], [], [])

    def fake_nsfw(results, items, cfg, progress=None, on_done=None):
        nsfw_started.set()
        assert faces_started.wait(5), "face detection did not run alongside NudeNet"
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
    last_check = max(len(phases) - 1 - phases[::-1].index(phase) for phase in ("faces", "nsfw"))
    assert first["scoring"] < first["faces"] and first["scoring"] < first["nsfw"]
    # Identities follow the faces at once: the set age read off them is context for the guard pass.
    assert last_check < first["identities"] < first["clustering"]
    # Progress events carry the same list the chunks do.
    assert all("pending_checks" in e for e in events if e["phase"] != "collecting")

    by_phase: dict[str, list[dict]] = {}
    for c in chunks:
        by_phase.setdefault(c["phase"], []).append(c)
    assert all(c["pending_checks"] == ["faces", "nsfw"] for c in by_phase["scoring"])
    # The chunk carrying a check's last result no longer lists that check.
    assert "faces" not in by_phase["faces"][-1]["pending_checks"]
    assert "nsfw" not in by_phase["nsfw"][-1]["pending_checks"]
    assert by_phase["identities"][-1]["pending_checks"] == []
    assert summary.nsfw_config.enabled is True
    assert all(r.nsfw is not None for r in summary.results)


def test_set_age_reaches_the_faceless_frames_before_the_guard_pass(
    dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The dominant identity's median age lands on the passing frames with no
    face during the identities phase, which now runs before scenes, garments
    and guard, so SingGuard can be told what the set says."""
    from argus_curator import faces as faces_mod
    from argus_curator.attributes import RawAttributes
    from argus_curator.models import FaceAttributes, FaceDetection

    same_person = np.zeros(512, dtype=np.float32)
    same_person[0] = 1.0

    def fake_faces(results, items, cfg, progress=None, on_done=None):
        detected = faces_mod.DetectedFaces([], [], [])
        passing = [r for r in results if r.passed]
        for i, r in enumerate(passing):
            if r.rel_path != "personA/s2/img.png":  # one passing frame shows no face
                age = 18.0 + i
                attrs = FaceAttributes(age_years=age, age_source="mivolo", age="10-19" if age < 20 else "20-29")
                r.faces = [FaceDetection(bbox=[0, 0, 10, 10], det_score=0.9, primary=True, attributes=attrs)]
                r.face_count = 1
                r.primary_face_attributes = attrs
                detected.embeddings.append(same_person)
                detected.meta.append((r.rel_path, 0, [0, 0, 10, 10], 0.9, 1.0, 100.0))
                detected.raws.append(RawAttributes(mivolo_age=age))
            if on_done is not None:
                on_done(r, i + 1, len(passing))
        return detected

    monkeypatch.setattr(faces_mod, "detect_faces", fake_faces)
    events: list[dict] = []
    summary = scan_folder(dataset, faces_cfg=FaceConfig(set_age=True, set_age_min_faces=2), progress=events.append)

    phases = [e["phase"] for e in events]
    assert phases.index("identities") < phases.index("clustering")
    assert summary.set_age is not None and summary.set_age_skipped is None
    assert summary.set_age.cluster_id == "face_1" and summary.set_age.faces == 2 and summary.set_age.share == 1.0
    by_rel = {r.rel_path: r for r in summary.results}
    assert by_rel["personA/s2/img.png"].set_age == summary.set_age
    assert by_rel["personA/s1/img.png"].set_age is None  # it has a face of its own
    assert by_rel["personB/blurry.png"].set_age is None  # rejected before detection

    # Not asked: nothing is said either way.
    plain = scan_folder(dataset, faces_cfg=FaceConfig(mivolo=True))
    assert plain.set_age is None and plain.set_age_skipped is None


def test_nsfw_failure_still_lets_faces_finish(dataset: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from argus_curator import faces as faces_mod
    from argus_curator import nsfw as nsfw_mod
    from argus_curator.models import NsfwConfig
    from argus_curator.runtime import ModelUnavailable

    def fake_faces(results, items, cfg, progress=None, on_done=None):
        for i, r in enumerate([r for r in results if r.passed]):
            if on_done is not None:
                on_done(r, i + 1, 3)
        return faces_mod.DetectedFaces([], [], [])

    def fake_nsfw(results, items, cfg, progress=None, on_done=None):
        raise ModelUnavailable("NudeNet is down")

    monkeypatch.setattr(faces_mod, "detect_faces", fake_faces)
    monkeypatch.setattr(nsfw_mod, "check_images", fake_nsfw)
    monkeypatch.setattr(nsfw_mod, "resolve_enabled", lambda cfg: True)

    chunks: list[dict] = []
    with pytest.raises(ModelUnavailable):
        scan_folder(dataset, faces_cfg=FaceConfig(enabled=True), nsfw_cfg=NsfwConfig(), chunk=chunks.append)
    # Face detection ran to the end, and no chunk was emitted after the failure half-finished.
    assert [c["phase"] for c in chunks].count("faces") >= 1
    assert chunks[-1]["phase"] == "faces"


def test_target_profile_roundtrips_into_summary(dataset: Path) -> None:
    profile = TargetProfile(target_style="anime", target_category="wardrobe", target_backend="flux-dev-1")
    summary = scan_folder(dataset, profile=profile, faces_cfg=FaceConfig(enabled=False))
    assert summary.target_profile.target_style == "anime"
    assert summary.target_profile.target_category == "wardrobe"


def test_cancelled_scan_raises_and_drops_queued_work(dataset: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A stop is noticed at the next progress step; the work still queued for
    a remote server is dropped rather than waited for; nothing after the stop
    is delivered."""
    import threading

    from argus_curator import nsfw as nsfw_mod
    from argus_curator import remote
    from argus_curator.models import NsfwConfig
    from argus_curator.scanner import ScanCancelled

    stop = threading.Event()
    sent: list[str] = []
    gate = threading.Lock()

    def slow_check(data: bytes, rel_path: str) -> None:
        with gate:
            sent.append(rel_path)
        stop.set()  # the first check asks the scan to stop

    def fake_nsfw(results, items, cfg, progress=None, on_done=None):
        remote.check_all(results, items, slow_check, "nsfw", workers=1, progress=progress, on_done=on_done)
        return True

    monkeypatch.setattr(nsfw_mod, "check_images", fake_nsfw)
    monkeypatch.setattr(nsfw_mod, "resolve_enabled", lambda cfg: True)
    chunks: list[dict] = []
    with pytest.raises(ScanCancelled):
        scan_folder(
            dataset,
            faces_cfg=FaceConfig(enabled=False),
            nsfw_cfg=NsfwConfig(),
            progress=lambda payload: None,
            chunk=chunks.append,
            cancelled=stop.is_set,
        )
    # Scoring finished (the stop came during NudeNet), and the pool was drained:
    # with one worker, at most the checks before the next progress tick ran.
    assert [c["phase"] for c in chunks].count("scoring") >= 1
    assert all(c["phase"] == "scoring" for c in chunks)
    assert len(sent) < 5


def test_a_scan_nobody_stops_finishes(dataset: Path) -> None:
    summary = scan_folder(dataset, faces_cfg=FaceConfig(enabled=False), cancelled=lambda: False)
    assert summary.total == 5


def test_drop_results_regroups_when_the_representative_is_gone() -> None:
    from argus_curator.models import FaceCluster, FaceConfig, FaceDetection, ImageResult, ScanSummary
    from argus_curator.scanner import _assign_clusters, _mark_duplicates, drop_results

    phash = "0" * 16
    rows = [
        ImageResult(
            rel_path=name,
            abs_path=f"/x/{name}",
            score=score,
            passed=True,
            phash=phash,
            faces=[FaceDetection(bbox=[0, 0, 10, 10], det_score=0.9, cluster_id="face_1")],
        )
        for name, score in (("best.png", 0.9), ("b.png", 0.8), ("c.png", 0.7))
    ]
    _mark_duplicates(rows, ScanConfig())
    _assign_clusters(rows)
    assert rows[1].duplicate_of == "best.png"
    summary = ScanSummary(
        scan_id="s",
        folder="/x",
        target_profile=TargetProfile(),
        config=ScanConfig(),
        faces_config=FaceConfig(),
        total=3,
        passed=3,
        rejected=0,
        duplicates=2,
        similar_clusters=1,
        face_clusters=[FaceCluster(cluster_id="face_1", size=3, representative_rel_path="best.png")],
        results=rows,
    )
    drop_results(summary, ["best.png"])
    by = {r.rel_path: r for r in summary.results}
    assert not by["b.png"].is_duplicate and by["b.png"].is_representative
    assert by["c.png"].duplicate_of == "b.png"
    assert summary.total == 2 and summary.duplicates == 1 and summary.similar_clusters == 1
    assert summary.face_clusters[0].size == 2
    assert summary.face_clusters[0].representative_rel_path in {"b.png", "c.png"}
