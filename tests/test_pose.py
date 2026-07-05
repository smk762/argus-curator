"""Head-pose bucketing + report/manifest column contract (no InsightFace needed)."""

from __future__ import annotations

import csv
from pathlib import Path

from argus_curator.export import write_manifest, write_report
from argus_curator.faces import classify_pose
from argus_curator.models import (
    ExportRequest,
    FaceConfig,
    FaceDetection,
    ImageResult,
    ScanConfig,
    ScanSummary,
    TargetProfile,
)
from argus_curator.selection import decide_selection


def test_classify_pose_buckets() -> None:
    cfg = FaceConfig()  # frontal<=15, profile>45
    assert classify_pose(0.0, cfg) == "frontal"
    assert classify_pose(15.0, cfg) == "frontal"
    assert classify_pose(-15.0, cfg) == "frontal"  # sign-independent
    assert classify_pose(30.0, cfg) == "three_quarter"
    assert classify_pose(-45.0, cfg) == "three_quarter"
    assert classify_pose(60.0, cfg) == "profile"
    assert classify_pose(None, cfg) is None


def test_classify_pose_respects_custom_thresholds() -> None:
    cfg = FaceConfig(frontal_max_yaw=10.0, profile_min_yaw=30.0)
    assert classify_pose(12.0, cfg) == "three_quarter"
    assert classify_pose(31.0, cfg) == "profile"


def test_face_detection_carries_pose() -> None:
    fd = FaceDetection(bbox=[0, 0, 10, 10], det_score=0.9, yaw=52.3, pitch=-3.1, pose="profile")
    assert fd.pose == "profile"
    assert fd.yaw == 52.3


def _img(rel: str, pose: str | None, yaw: float | None) -> ImageResult:
    return ImageResult(
        rel_path=rel,
        abs_path=f"/data/{rel}",
        score=0.9,
        passed=True,
        primary_face_pose=pose,
        primary_face_yaw=yaw,
    )


def test_report_and_manifest_include_pose_columns(tmp_path: Path) -> None:
    results = [_img("a.jpg", "frontal", 4.0), _img("b.jpg", "profile", 61.0)]
    report = tmp_path / "curation_report.csv"
    write_report(results, {r.rel_path: "" for r in results}, {"a.jpg", "b.jpg"}, report)

    rows = list(csv.DictReader(report.open()))
    assert "primary_face_pose" in rows[0]
    assert "primary_face_yaw" in rows[0]
    by_rel = {r["rel_path"]: r for r in rows}
    assert by_rel["a.jpg"]["primary_face_pose"] == "frontal"
    assert by_rel["b.jpg"]["primary_face_pose"] == "profile"

    summary = ScanSummary(
        scan_id="s1",
        folder="/data",
        target_profile=TargetProfile(),
        config=ScanConfig(),
        faces_config=FaceConfig(),
        total=2,
        passed=2,
        rejected=0,
        duplicates=0,
        similar_clusters=0,
        results=results,
    )
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(results, summary, manifest, {r.rel_path: r.rel_path for r in results})
    import json

    lines = [json.loads(line) for line in manifest.read_text().splitlines()]
    assert {row["primary_face_pose"] for row in lines} == {"frontal", "profile"}


def test_export_pose_filter() -> None:
    results = [_img("a.jpg", "frontal", 4.0), _img("b.jpg", "profile", 61.0), _img("c.jpg", "three_quarter", 30.0)]
    req = ExportRequest(dest="/out", min_score=0.0, face_poses=["frontal", "three_quarter"])
    selected, keep_reason = decide_selection(results, req, 0.0)
    rels = {r.rel_path for r in selected}
    assert rels == {"a.jpg", "c.jpg"}
    assert keep_reason["b.jpg"] == "face-pose-excluded"
