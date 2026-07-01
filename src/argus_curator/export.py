"""Export — structure-preserving transfer + the JSONL manifest argus-lens consumes.

Ported from imogen PR #14's ``_transfer`` / ``_write_csv`` and extended with the
handoff manifest. The manifest is one JSON object per selected image carrying
the shared ``target_profile``, so argus-lens can batch-caption it with no
category remapping (section 8 of the brief).
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import structlog

from argus_curator.models import ExportRequest, ExportResult, ImageResult, ScanSummary
from argus_curator.selection import decide_selection

logger = structlog.get_logger()

MANIFEST_NAME = "manifest.jsonl"
REPORT_NAME = "curation_report.csv"


def _dest_path(dest_root: Path, r: ImageResult, preserve_structure: bool) -> Path:
    if preserve_structure:
        return dest_root / r.rel_path
    return dest_root / Path(r.rel_path).name


def _transfer_one(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "move":
        shutil.move(str(src), str(dst))
    elif mode == "symlink":
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())
    else:  # copy
        shutil.copy2(src, dst)


def write_manifest(
    selected: list[ImageResult],
    summary: ScanSummary,
    path: Path,
) -> None:
    """Write the JSONL handoff manifest (one selected image per line)."""
    profile = summary.target_profile.model_dump()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in selected:
            row = {
                "rel_path": r.rel_path,
                "abs_path": r.abs_path,
                "target_profile": profile,
                "primary_face_cluster": r.primary_face_cluster,
                "primary_face_pose": r.primary_face_pose,
                "score": round(r.score, 4),
                "similar_group": r.similar_group,
            }
            f.write(json.dumps(row) + "\n")


def write_report(
    results: list[ImageResult],
    keep_reason: dict[str, str],
    selected_rel: set[str],
    path: Path,
) -> None:
    """Write the full per-image CSV report (HITL transparency, grouped by cluster)."""
    fieldnames = [
        "rel_path",
        "score",
        "keep",
        "keep_reason",
        "similar_group",
        "group_size",
        "is_representative",
        "passed",
        "reject_reason",
        "is_duplicate",
        "duplicate_of",
        "sharpness",
        "artifact_score",
        "face_count",
        "primary_face_cluster",
        "primary_face_pose",
        "primary_face_yaw",
        "width",
        "height",
        "abs_path",
    ]
    ordered = sorted(results, key=lambda r: (r.similar_group, -r.score))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in ordered:
            w.writerow(
                {
                    "rel_path": r.rel_path,
                    "score": round(r.score, 4),
                    "keep": r.rel_path in selected_rel,
                    "keep_reason": keep_reason.get(r.rel_path, ""),
                    "similar_group": r.similar_group,
                    "group_size": r.group_size,
                    "is_representative": r.is_representative,
                    "passed": r.passed,
                    "reject_reason": r.reject_reason or "",
                    "is_duplicate": r.is_duplicate,
                    "duplicate_of": r.duplicate_of or "",
                    "sharpness": round(r.sharpness, 2),
                    "artifact_score": round(r.artifact_score, 4),
                    "face_count": r.face_count,
                    "primary_face_cluster": r.primary_face_cluster or "",
                    "primary_face_pose": r.primary_face_pose or "",
                    "primary_face_yaw": "" if r.primary_face_yaw is None else round(r.primary_face_yaw, 1),
                    "width": r.width,
                    "height": r.height,
                    "abs_path": r.abs_path,
                }
            )


def _post_manifest_to_lens(manifest_path: Path, caption_url: str) -> bool:
    """Optionally POST the manifest to argus-lens for a one-click curate->caption run."""
    try:
        import httpx
    except Exception:
        logger.warning("caption_handoff_skipped", reason="httpx not installed")
        return False
    try:
        with manifest_path.open("rb") as f:
            resp = httpx.post(
                caption_url,
                files={"manifest": (MANIFEST_NAME, f, "application/x-ndjson")},
                timeout=30.0,
            )
        resp.raise_for_status()
        return True
    except Exception as exc:
        logger.warning("caption_handoff_failed", url=caption_url, error=str(exc))
        return False


def export_selection(summary: ScanSummary, req: ExportRequest) -> ExportResult:
    """Select per the request, transfer files, and write the manifest/report."""
    results = summary.results

    if req.selection is not None:
        wanted = set(req.selection)
        selected = [r for r in results if r.rel_path in wanted]
        keep_reason = {r.rel_path: ("" if r.rel_path in wanted else "not-selected") for r in results}
    else:
        selected, keep_reason = decide_selection(results, req, summary.config.diversity_weight)

    selected_rel = {r.rel_path for r in selected}
    dest_root = Path(req.dest)

    copied = 0
    skipped = 0
    for r in selected:
        src = Path(r.abs_path)
        if not src.exists():
            logger.warning("export_source_missing", rel_path=r.rel_path)
            skipped += 1
            continue
        dst = _dest_path(dest_root, r, req.preserve_structure)
        try:
            _transfer_one(src, dst, req.mode)
            copied += 1
        except Exception as exc:
            logger.warning("export_transfer_failed", rel_path=r.rel_path, error=str(exc))
            skipped += 1

    manifest_path: Path | None = None
    if req.write_manifest:
        manifest_path = dest_root / MANIFEST_NAME
        write_manifest(selected, summary, manifest_path)
        write_report(results, keep_reason, selected_rel, dest_root / REPORT_NAME)

    captioned = False
    if req.caption_url and manifest_path is not None:
        captioned = _post_manifest_to_lens(manifest_path, req.caption_url)

    logger.info(
        "export_done",
        dest=str(dest_root),
        mode=req.mode,
        copied=copied,
        skipped=skipped,
        selected=len(selected),
    )

    return ExportResult(
        manifest_path=str(manifest_path) if manifest_path else None,
        copied=copied,
        skipped=skipped,
        dest=str(dest_root.resolve()) if dest_root.exists() else str(dest_root),
        mode=req.mode,
        selected_rel_paths=sorted(selected_rel),
        captioned=captioned,
    )
