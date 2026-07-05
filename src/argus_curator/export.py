"""Export — structure-preserving transfer + the JSONL manifest argus-lens consumes.

Ported from imogen PR #14's ``_transfer`` / ``_write_csv`` and extended with the
handoff manifest. The manifest is one JSON object per selected image carrying
the shared ``target_profile``, so argus-lens can batch-caption it with no
category remapping (section 8 of the brief).
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import structlog

from argus_curator.models import MANIFEST_VERSION, ExportRequest, ExportResult, ImageResult, ScanSummary
from argus_curator.selection import decide_selection

logger = structlog.get_logger()

# A progress sink: called with {"phase": str, "done": int, "total": int}.
ProgressFn = Callable[[dict[str, Any]], None]

MANIFEST_NAME = "manifest.jsonl"
REPORT_NAME = "curation_report.csv"


def _plan_dest_paths(selected: list[ImageResult], dest_root: Path, preserve_structure: bool) -> dict[str, Path]:
    """Map each selected rel_path to its destination, de-colliding flattened basenames.

    With ``preserve_structure=False`` two rel_paths can share a basename
    (``a/IMG_0001.jpg`` + ``b/IMG_0001.jpg``); a naive flatten would silently
    overwrite one with the other. Every member of a colliding basename group is
    suffixed with a short hash of its rel_path (order-independent, so the same
    selection always yields the same names).
    """
    if preserve_structure:
        return {r.rel_path: dest_root / r.rel_path for r in selected}

    by_name: dict[str, list[ImageResult]] = {}
    for r in selected:
        by_name.setdefault(Path(r.rel_path).name, []).append(r)

    dests: dict[str, Path] = {}
    for name, rows in by_name.items():
        if len(rows) == 1:
            dests[rows[0].rel_path] = dest_root / name
        else:
            logger.warning("export_basename_collision", basename=name, rel_paths=[r.rel_path for r in rows])
            for r in rows:
                p = Path(r.rel_path)
                digest = hashlib.sha1(r.rel_path.encode("utf-8")).hexdigest()[:8]
                dests[r.rel_path] = dest_root / f"{p.stem}-{digest}{p.suffix}"
    return dests


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
    exported_paths: dict[str, str],
) -> None:
    """Write the JSONL handoff manifest (one selected image per line).

    ``exported_paths`` maps rel_path to the path actually written under the
    export root (posix, relative). Consumers must use it instead of re-deriving
    a destination from ``rel_path`` — flattened exports de-collide basenames,
    so the two can differ.
    """
    profile = summary.target_profile.model_dump()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in selected:
            row = {
                "manifest_version": MANIFEST_VERSION,
                "rel_path": r.rel_path,
                "abs_path": r.abs_path,
                "exported_path": exported_paths[r.rel_path],
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


def export_selection(summary: ScanSummary, req: ExportRequest, progress: ProgressFn | None = None) -> ExportResult:
    """Select per the request, transfer files, and write the manifest/report.

    When *progress* is supplied it is called with ``{phase, done, total}`` dicts
    during the ``transferring`` phase (one update per file), so a caller can
    drive a live progress bar (see the /export/stream SSE endpoint).
    """
    results = summary.results

    if req.selection is not None:
        wanted = set(req.selection)
        selected = [r for r in results if r.rel_path in wanted]
        keep_reason = {r.rel_path: ("" if r.rel_path in wanted else "not-selected") for r in results}
    else:
        selected, keep_reason = decide_selection(results, req, summary.config.diversity_weight)

    selected_rel = {r.rel_path for r in selected}
    dest_root = Path(req.dest)
    dest_paths = _plan_dest_paths(selected, dest_root, req.preserve_structure)

    total = len(selected)
    if progress is not None:
        progress({"phase": "transferring", "done": 0, "total": total})

    copied = 0
    skipped = 0
    for i, r in enumerate(selected):
        src = Path(r.abs_path)
        if not src.exists():
            logger.warning("export_source_missing", rel_path=r.rel_path)
            skipped += 1
        else:
            dst = dest_paths[r.rel_path]
            try:
                _transfer_one(src, dst, req.mode)
                copied += 1
            except Exception as exc:
                logger.warning("export_transfer_failed", rel_path=r.rel_path, error=str(exc))
                skipped += 1
        if progress is not None:
            progress({"phase": "transferring", "done": i + 1, "total": total})

    manifest_path: Path | None = None
    if req.write_manifest:
        manifest_path = dest_root / MANIFEST_NAME
        exported_rel = {rel: dst.relative_to(dest_root).as_posix() for rel, dst in dest_paths.items()}
        write_manifest(selected, summary, manifest_path, exported_rel)
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
