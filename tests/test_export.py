"""Export: selection logic, structure-preserving transfer, and the manifest."""

from __future__ import annotations

import json
from pathlib import Path

from argus_curator import scan_folder
from argus_curator.export import export_selection
from argus_curator.models import ExportRequest
from argus_curator.selection import decide_selection


def test_copy_preserves_structure_and_writes_manifest(dataset: Path, tmp_path: Path) -> None:
    summary = scan_folder(dataset)
    dest = tmp_path / "out"
    req = ExportRequest(scan_id=summary.scan_id, dest=str(dest), mode="copy", min_score=0.0)
    result = export_selection(summary, req)

    assert result.copied >= 1
    assert (dest / "manifest.jsonl").exists()
    assert (dest / "curation_report.csv").exists()
    # Structure preserved: a copied keeper sits at its rel_path under dest.
    for rel in result.selected_rel_paths:
        assert (dest / rel).exists()


def test_manifest_rows_carry_target_profile(dataset: Path, tmp_path: Path) -> None:
    summary = scan_folder(dataset)
    dest = tmp_path / "out"
    export_selection(summary, ExportRequest(scan_id=summary.scan_id, dest=str(dest), min_score=0.0))

    lines = (dest / "manifest.jsonl").read_text().strip().splitlines()
    assert lines
    row = json.loads(lines[0])
    assert set(row) == {
        "rel_path",
        "abs_path",
        "target_profile",
        "primary_face_cluster",
        "primary_face_pose",
        "score",
        "similar_group",
    }
    assert row["target_profile"]["target_category"] == "identity"


def test_min_score_threshold_excludes_with_reason(dataset: Path) -> None:
    summary = scan_folder(dataset)
    req = ExportRequest(scan_id=summary.scan_id, dest="/tmp/unused", min_score=1.1)
    selected, keep_reason = decide_selection(summary.results, req, summary.config.diversity_weight)
    assert selected == []
    assert all(reason for reason in keep_reason.values())


def test_duplicates_excluded_unless_keep_similar(dataset: Path) -> None:
    summary = scan_folder(dataset)
    req = ExportRequest(scan_id=summary.scan_id, dest="/tmp/unused", min_score=0.0, keep_similar=False)
    selected, keep_reason = decide_selection(summary.results, req, summary.config.diversity_weight)
    sel_rel = {r.rel_path for r in selected}
    dup = next(r for r in summary.results if r.is_duplicate)
    assert dup.rel_path not in sel_rel
    assert keep_reason[dup.rel_path].startswith("similar-to:")

    req_keep = ExportRequest(scan_id=summary.scan_id, dest="/tmp/unused", min_score=0.0, keep_similar=True)
    selected_keep, _ = decide_selection(summary.results, req_keep, summary.config.diversity_weight)
    assert dup.rel_path in {r.rel_path for r in selected_keep}


def test_symlink_mode(dataset: Path, tmp_path: Path) -> None:
    summary = scan_folder(dataset)
    dest = tmp_path / "linked"
    req = ExportRequest(scan_id=summary.scan_id, dest=str(dest), mode="symlink", min_score=0.0)
    result = export_selection(summary, req)
    assert result.copied >= 1
    a_link = dest / result.selected_rel_paths[0]
    assert a_link.is_symlink()
