"""Export: selection logic, structure-preserving transfer, and the manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from argus_curator import scan_folder
from argus_curator.export import _plan_dest_paths, export_selection
from argus_curator.models import MANIFEST_VERSION, ExportRequest, ImageResult
from argus_curator.selection import decide_selection


def _res(rel: str) -> ImageResult:
    return ImageResult(rel_path=rel, abs_path=f"/src/{rel}")


def test_copy_preserves_structure_and_writes_manifest(dataset: Path, tmp_path: Path) -> None:
    summary = scan_folder(dataset)
    dest = tmp_path / "out"
    req = ExportRequest(scan_id=summary.scan_id, dest=str(dest), mode="copy", min_score=0.0)
    result = export_selection(summary, req)

    assert result.copied >= 1
    assert (dest / "manifest.jsonl").exists()
    assert (dest / "curation_report.csv").exists()
    # Structure preserved: a copied keeper sits at its rel_path under dest,
    # and the result maps every transferred row to that same path.
    for rel in result.selected_rel_paths:
        assert (dest / rel).exists()
    assert result.exported_paths == {rel: rel for rel in result.selected_rel_paths}


def test_manifest_rows_carry_target_profile(dataset: Path, tmp_path: Path) -> None:
    summary = scan_folder(dataset)
    dest = tmp_path / "out"
    export_selection(summary, ExportRequest(scan_id=summary.scan_id, dest=str(dest), min_score=0.0))

    lines = (dest / "manifest.jsonl").read_text().strip().splitlines()
    assert lines
    row = json.loads(lines[0])
    assert set(row) == {
        "manifest_version",
        "rel_path",
        "abs_path",
        "exported_path",
        "exported_abs_path",
        "target_profile",
        "primary_face_cluster",
        "primary_face_pose",
        "score",
        "similar_group",
    }
    assert row["manifest_version"] == MANIFEST_VERSION
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


def test_flattened_export_decollides_basenames(dataset: Path, tmp_path: Path) -> None:
    """preserve_structure=False must not silently overwrite colliding basenames."""
    summary = scan_folder(dataset)
    dest = tmp_path / "flat"
    colliding = ["personA/s1/img.png", "personA/s2/img.png"]
    req = ExportRequest(
        selection=colliding,
        dest=str(dest),
        mode="copy",
        preserve_structure=False,
    )
    result = export_selection(summary, req)
    assert result.copied == 2

    lines = (dest / "manifest.jsonl").read_text().strip().splitlines()
    rows = {row["rel_path"]: row for row in map(json.loads, lines)}
    assert set(rows) == set(colliding)

    exported = {rows[rel]["exported_path"] for rel in colliding}
    assert len(exported) == 2, "colliding basenames must map to distinct exported paths"
    # De-collision suffixes every group member: nothing may sit at (or leak to)
    # the plain colliding basename.
    assert not (dest / "img.png").exists()
    for rel in colliding:
        out = dest / rows[rel]["exported_path"]
        assert out.exists()
        # Each exported file carries its own source's bytes — nothing was overwritten.
        assert out.read_bytes() == Path(rows[rel]["abs_path"]).read_bytes()
    assert result.exported_paths == {rel: rows[rel]["exported_path"] for rel in colliding}


def test_flattened_export_unique_basename_keeps_plain_name(dataset: Path, tmp_path: Path) -> None:
    summary = scan_folder(dataset)
    dest = tmp_path / "flat"
    req = ExportRequest(selection=["personA/s1/dup.png"], dest=str(dest), preserve_structure=False)
    export_selection(summary, req)
    assert (dest / "dup.png").exists()
    row = json.loads((dest / "manifest.jsonl").read_text().strip())
    assert row["exported_path"] == "dup.png"


def test_plan_decollides_case_insensitive_basenames() -> None:
    """Case-variant basenames are one directory entry on APFS/NTFS/exFAT dests."""
    dests = _plan_dest_paths([_res("a/IMG.png"), _res("b/img.png")], preserve_structure=False)
    assert len({d.casefold() for d in dests.values()}) == 2
    # Both members of the case-folded collision group get suffixed.
    assert all(d not in {"IMG.png", "img.png"} for d in dests.values())


def test_plan_extends_digest_when_suffixed_name_is_taken() -> None:
    """A selected file already named like a generated suffix must not be clobbered."""
    taken = f"img-{hashlib.sha1(b'a/img.png').hexdigest()[:8]}.png"
    sel = [_res("a/img.png"), _res("b/img.png"), _res(f"old/{taken}")]
    dests = _plan_dest_paths(sel, preserve_structure=False)
    assert len(set(dests.values())) == 3
    assert dests[f"old/{taken}"] == taken  # unique basename keeps its plain name


def test_skipped_sources_are_excluded_from_manifest(dataset: Path, tmp_path: Path) -> None:
    """Manifest rows exist only for files actually written under dest."""
    summary = scan_folder(dataset)
    (dataset / "personA" / "s2" / "img.png").unlink()  # vanishes between scan and export
    dest = tmp_path / "flat"
    req = ExportRequest(
        selection=["personA/s1/img.png", "personA/s2/img.png"],
        dest=str(dest),
        preserve_structure=False,
    )
    result = export_selection(summary, req)
    assert (result.copied, result.skipped) == (1, 1)
    assert set(result.exported_paths) == {"personA/s1/img.png"}

    rows = [json.loads(line) for line in (dest / "manifest.jsonl").read_text().strip().splitlines()]
    assert [r["rel_path"] for r in rows] == ["personA/s1/img.png"]
    assert (dest / rows[0]["exported_path"]).exists()


def test_symlink_mode(dataset: Path, tmp_path: Path) -> None:
    summary = scan_folder(dataset)
    dest = tmp_path / "linked"
    req = ExportRequest(scan_id=summary.scan_id, dest=str(dest), mode="symlink", min_score=0.0)
    result = export_selection(summary, req)
    assert result.copied >= 1
    a_link = dest / result.selected_rel_paths[0]
    assert a_link.is_symlink()


# ---------------------------------------------------------------------------
# Manifest/result path contract (issues #9, #10, #11). The recurring failure is
# a consumer having to reconstruct the server's layout — or being handed a path
# the server already deleted.
# ---------------------------------------------------------------------------


def test_move_manifest_abs_path_points_at_the_moved_file(dataset: Path, tmp_path: Path) -> None:
    """Issue #9: under move the source is gone, so abs_path must name the destination.

    argus-lens opens rows strictly by abs_path, so writing the source there made
    a move export's manifest unusable — every row a dead path.
    """
    summary = scan_folder(dataset)
    dest = tmp_path / "moved"
    result = export_selection(
        summary,
        ExportRequest(scan_id=summary.scan_id, dest=str(dest), mode="move", min_score=0.0),
    )
    assert result.copied >= 1

    rows = [json.loads(line) for line in (dest / "manifest.jsonl").read_text().strip().splitlines()]
    assert rows
    for row in rows:
        # The whole point: every path the row hands a consumer is openable.
        assert Path(row["abs_path"]).is_file(), row["abs_path"]
        assert Path(row["exported_abs_path"]).is_file()
        assert row["abs_path"] == row["exported_abs_path"]
        # ...and it is the destination, not the source the transfer deleted.
        assert Path(row["abs_path"]).is_relative_to(dest.resolve())
        assert not Path(dataset / row["rel_path"]).exists()


def test_copy_manifest_abs_path_still_names_the_source(dataset: Path, tmp_path: Path) -> None:
    """The move fix must not disturb copy: the source survives and stays authoritative."""
    summary = scan_folder(dataset)
    dest = tmp_path / "copied"
    export_selection(
        summary,
        ExportRequest(scan_id=summary.scan_id, dest=str(dest), mode="copy", min_score=0.0),
    )
    rows = [json.loads(line) for line in (dest / "manifest.jsonl").read_text().strip().splitlines()]
    assert rows
    for row in rows:
        assert Path(row["abs_path"]).is_relative_to(dataset.resolve())
        assert Path(row["abs_path"]).is_file()
        # The copy is still reachable, just via the dedicated field.
        assert Path(row["exported_abs_path"]).is_file()
        assert Path(row["exported_abs_path"]).is_relative_to(dest.resolve())


def test_symlink_exported_abs_path_is_the_link_not_its_target(dataset: Path, tmp_path: Path) -> None:
    """Resolving the destination would follow the link back to the source.

    That would report the very location the export exists to move away from, so
    the root is resolved and the destination joined onto it.
    """
    summary = scan_folder(dataset)
    dest = tmp_path / "linked"
    result = export_selection(
        summary,
        ExportRequest(scan_id=summary.scan_id, dest=str(dest), mode="symlink", min_score=0.0),
    )
    assert result.exported_abs_paths
    for abs_path in result.exported_abs_paths.values():
        assert Path(abs_path).is_symlink()
        assert Path(abs_path).is_relative_to(dest.resolve())


def test_exported_abs_paths_survive_flattened_de_collision(dataset: Path, tmp_path: Path) -> None:
    """Issue #10: with preserve_structure=False the client cannot derive the name.

    De-collided basenames are stem-<sha1 prefix>.ext, so a client joining dest
    onto rel_path lands nowhere. The absolute mapping is the only usable answer.
    """
    summary = scan_folder(dataset)
    dest = tmp_path / "flat"
    result = export_selection(
        summary,
        ExportRequest(
            scan_id=summary.scan_id,
            dest=str(dest),
            min_score=0.0,
            preserve_structure=False,
        ),
    )
    assert result.exported_abs_paths
    assert set(result.exported_abs_paths) == set(result.exported_paths)
    for rel, abs_path in result.exported_abs_paths.items():
        assert Path(abs_path).is_file()
        assert Path(abs_path) == dest.resolve() / result.exported_paths[rel]
        # Absolute, so no client-side join against dest is needed at all.
        assert Path(abs_path).is_absolute()


def test_export_result_declares_manifest_version_even_without_a_manifest(dataset: Path, tmp_path: Path) -> None:
    """Issue #11: the version must not depend on a manifest being requested."""
    summary = scan_folder(dataset)
    result = export_selection(
        summary,
        ExportRequest(
            scan_id=summary.scan_id,
            dest=str(tmp_path / "nomanifest"),
            min_score=0.0,
            write_manifest=False,
        ),
    )
    assert result.manifest_path is None
    assert result.manifest_version == MANIFEST_VERSION
    # Still populated, which is the point of exported_paths existing at all.
    assert result.exported_abs_paths


def test_empty_export_is_distinguishable_from_a_legacy_server(dataset: Path, tmp_path: Path) -> None:
    """The sniffing failure mode #11 describes: an empty mapping is not "old server".

    A 2.x curator that transferred nothing sends {} — identical to what a
    presence check would read as legacy — so the declared version is the only
    thing that separates the two.
    """
    summary = scan_folder(dataset)
    result = export_selection(
        summary,
        ExportRequest(
            scan_id=summary.scan_id,
            dest=str(tmp_path / "none"),
            min_score=99.0,  # nothing clears this
        ),
    )
    assert result.copied == 0
    assert result.exported_paths == {}
    assert result.exported_abs_paths == {}
    assert result.manifest_version == MANIFEST_VERSION


def test_selected_rel_paths_is_not_a_record_of_what_landed(dataset: Path, tmp_path: Path) -> None:
    """The truthfulness nit in #9: selection is computed before the transfer loop.

    A source that vanishes between scan and export is still "selected" but never
    lands, so only exported_paths answers "what is on disk".
    """
    summary = scan_folder(dataset)
    doomed = summary.results[0]
    Path(doomed.abs_path).unlink()

    result = export_selection(
        summary,
        ExportRequest(scan_id=summary.scan_id, dest=str(tmp_path / "partial"), min_score=0.0),
    )
    assert doomed.rel_path in result.selected_rel_paths
    assert doomed.rel_path not in result.exported_paths
    assert doomed.rel_path not in result.exported_abs_paths
    assert result.skipped >= 1
