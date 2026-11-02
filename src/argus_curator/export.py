"""Export — structure-preserving transfer + the JSONL manifest argus-lens consumes.

Ported from imogen PR #14's ``_transfer`` / ``_write_csv`` and extended with the
handoff manifest. The manifest is one JSON object per exported image carrying
the shared ``target_profile``, so argus-lens can batch-caption it with no
category remapping (section 8 of the brief).
"""

from __future__ import annotations

import csv
import hashlib
import shutil
import unicodedata
from collections import Counter
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

import structlog
from argus_cortex.moderation import rule_letter
from argus_cortex.xmp import XmpData, XmpError, XmpRegion, XmpRegions, sidecar_path, write_xmp

from argus_curator.models import ExportRequest, ExportResult, FaceDetection, ImageResult, ManifestRow, ScanSummary
from argus_curator.selection import decide_selection, hard_exclusion

logger = structlog.get_logger()

# A progress sink: called with {"phase": str, "done": int, "total": int}.
ProgressFn = Callable[[dict[str, Any]], None]

MANIFEST_NAME = "manifest.jsonl"
REPORT_NAME = "curation_report.csv"


def _norm_name(name: str) -> str:
    """Collision key for a basename.

    Destination filesystems may be case-insensitive (APFS/NTFS/exFAT/SMB) or
    Unicode-normalising, so names differing only in case or normal form must
    count as one destination entry.
    """
    return unicodedata.normalize("NFC", name).casefold()


def _suffixed(rel_path: str, digest_len: int) -> str:
    # surrogateescape keeps un-decodable filesystem bytes hashable; the hash is
    # naming-only, so opt out of FIPS restrictions on sha1.
    digest = hashlib.sha1(rel_path.encode("utf-8", "surrogateescape"), usedforsecurity=False).hexdigest()
    p = PurePosixPath(rel_path)
    return f"{p.stem}-{digest[:digest_len]}{p.suffix}"


def _plan_dest_paths(selected: list[ImageResult], preserve_structure: bool) -> dict[str, str]:
    """Map each selected rel_path to its destination relative to the export root.

    With ``preserve_structure=False`` two rel_paths can share a basename
    (``a/IMG_0001.jpg`` + ``b/IMG_0001.jpg``); a naive flatten would silently
    overwrite one with the other. Basenames are grouped under a case-folded,
    Unicode-normalised key and every member of a colliding group is suffixed
    with a short hash of its rel_path (order-independent, so the same selection
    always yields the same names). The finished plan must then be collision-free
    as a whole: a generated name that still clashes with anything (e.g. a
    selected file literally named ``stem-<hash>.ext``) gets a longer digest, and
    the export fails loudly rather than overwrite if uniqueness is unreachable.
    """
    if preserve_structure:
        return {r.rel_path: r.rel_path for r in selected}

    by_name: dict[str, list[str]] = {}
    for r in selected:
        by_name.setdefault(_norm_name(PurePosixPath(r.rel_path).name), []).append(r.rel_path)

    dests: dict[str, str] = {}
    digest_len: dict[str, int] = {}
    for rels in by_name.values():
        if len(rels) == 1:
            dests[rels[0]] = PurePosixPath(rels[0]).name
        else:
            logger.warning("export_basename_collision", basename=PurePosixPath(rels[0]).name, rel_paths=rels)
            digest_len.update(dict.fromkeys(rels, 8))

    pending = set(digest_len)
    while pending:
        for rel in pending:
            dests[rel] = _suffixed(rel, digest_len[rel])
        counts = Counter(_norm_name(d) for d in dests.values())
        pending = {rel for rel in digest_len if counts[_norm_name(dests[rel])] > 1}
        stuck = sorted(rel for rel in pending if digest_len[rel] >= 40)
        if stuck:
            raise ValueError(f"cannot de-collide flattened basenames for: {stuck}")
        for rel in pending:
            digest_len[rel] += 8
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
    exported: list[ImageResult],
    summary: ScanSummary,
    path: Path,
    exported_paths: dict[str, str],
    exported_abs_paths: dict[str, str] | None = None,
    mode: str = "copy",
) -> None:
    """Write the JSONL handoff manifest (one :class:`ManifestRow` per line).

    ``exported`` must contain only images whose transfer actually succeeded —
    the manifest is the captioner's work list and must not claim files that
    are not on disk. ``exported_paths`` maps rel_path to the path written under
    the export root (posix, relative); consumers must use it instead of
    re-deriving a destination from ``rel_path`` — flattened exports de-collide
    basenames, so the two can differ. ``exported_abs_paths`` is that mapping
    made absolute, so each row is usable without knowing the export root.

    Under ``mode="move"`` the transfer deleted the source, so ``abs_path`` —
    which consumers open the image from — is written as the *destination*.
    Writing the source there names a file that no longer exists, which is the
    whole of issue #9: argus-lens reads rows strictly by ``abs_path``, so a
    moved export produced a manifest it could not use at all.
    """
    abs_paths = exported_abs_paths or {}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in exported:
            exported_abs = abs_paths.get(r.rel_path) or str(path.parent / exported_paths[r.rel_path])
            row = ManifestRow(
                rel_path=r.rel_path,
                abs_path=exported_abs if mode == "move" else r.abs_path,
                exported_path=exported_paths[r.rel_path],
                exported_abs_path=exported_abs,
                target_profile=summary.target_profile,
                primary_face_cluster=r.primary_face_cluster,
                primary_face_pose=r.primary_face_pose,
                score=round(r.score, 4),
                similar_group=r.similar_group,
            )
            f.write(row.model_dump_json() + "\n")


def _blank(value: object) -> object:
    return "" if value is None else value


def write_report(
    results: list[ImageResult],
    keep_reason: dict[str, str],
    selected_rel: set[str],
    exported_paths: dict[str, str],
    path: Path,
) -> None:
    """Write the full per-image CSV report (HITL transparency, grouped by cluster).

    ``exported_paths`` fills the ``exported_path`` column for rows whose
    transfer succeeded (empty otherwise), so a reviewer can map report rows to
    the possibly de-collided files on disk without joining the manifest.
    """
    fieldnames = [
        "rel_path",
        "score",
        "keep",
        "keep_reason",
        "exported_path",
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
        "primary_face_race",
        "primary_face_race_conf",
        "primary_face_gender",
        "primary_face_gender_conf",
        "primary_face_age",
        "primary_face_age_conf",
        "primary_face_age_years",
        "primary_face_gender_source",
        "primary_face_age_source",
        "primary_face_age_body",
        "people_count",
        "body_ages",
        "scene",
        "scene_conf",
        "scene_environment",
        "scene_environment_conf",
        "garments",
        "nsfw_exposed_score",
        "nsfw_labels",
        "content_rating",
        "guard_safe",
        "guard_rules",
        "guard_variants",
        "width",
        "height",
        "abs_path",
    ]
    attr_fields = (
        "race",
        "race_conf",
        "gender",
        "gender_conf",
        "age",
        "age_conf",
        "age_years",
        "gender_source",
        "age_source",
        "age_body",
    )
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
                    "exported_path": exported_paths.get(r.rel_path, ""),
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
                    **{f"primary_face_{k}": _blank(getattr(r.primary_face_attributes, k, None)) for k in attr_fields},
                    "people_count": len(r.people),
                    "body_ages": ";".join(
                        f"{p.age_years:.1f}" for p in r.people if p.face_index is None and p.age_years is not None
                    ),
                    "scene": r.scene.categories[0].label if r.scene and r.scene.categories else "",
                    "scene_conf": r.scene.categories[0].conf if r.scene and r.scene.categories else "",
                    "scene_environment": r.scene.environment if r.scene else "",
                    "scene_environment_conf": r.scene.environment_conf if r.scene else "",
                    # "label:conf:#color" per detection; "*" marks one cut off at the frame edge.
                    "garments": ";".join(
                        f"{g.label}:{g.conf:.2f}:{g.color}{'*' if g.touches_edge else ''}" for g in r.garments or []
                    ),
                    "nsfw_exposed_score": r.nsfw.exposed_score if r.nsfw else "",
                    # "LABEL:score" per detected class, highest first.
                    "nsfw_labels": ";".join(f"{k}:{v:.2f}" for k, v in r.nsfw.labels.items()) if r.nsfw else "",
                    "content_rating": r.content_rating or "",
                    "guard_safe": "" if r.guard is None else str(r.guard.safe).lower(),
                    "guard_rules": ";".join(r.guard.rules) if r.guard else "",
                    # "policy=safe" or "policy=unsafe[A,H]" per policy that judged the image.
                    "guard_variants": ";".join(
                        f"{name}={'safe' if g.safe else 'unsafe[' + ','.join(rule_letter(x) for x in g.rules) + ']'}"
                        for name, g in (r.guard_variants or {}).items()
                    ),
                    "width": r.width,
                    "height": r.height,
                    "abs_path": r.abs_path,
                }
            )


def _region(face: FaceDetection, width: int, height: int, names: dict[str, str]) -> XmpRegion:
    """*face*'s pixel ``[x, y, w, h]`` as an MWG region: centre and size as fractions of the frame."""
    x, y, w, h = face.bbox

    def frac(value: float, whole: int) -> float:
        return min(1.0, max(0.0, value / whole))

    return XmpRegion(
        x=frac(x + w / 2, width),
        y=frac(y + h / 2, height),
        w=frac(w, width),
        h=frac(h, height),
        name=names.get(face.cluster_id) if face.cluster_id else None,
    )


def xmp_data(r: ImageResult, face_names: dict[str, str] | None = None) -> XmpData | None:
    """What curator records about *r* in a sidecar, or ``None`` when it knows nothing worth writing.

    A field curator has nothing for stays ``None``, so merging leaves whatever
    the sidecar already holds there. Age estimates are never written: a MiVOLO
    guess in a photo library would read as a fact.
    """
    names = face_names or {}
    regions = None
    if r.faces and r.width > 0 and r.height > 0:
        regions = XmpRegions(
            width=r.width, height=r.height, regions=[_region(face, r.width, r.height, names) for face in r.faces]
        )
    persons = sorted({names[f.cluster_id] for f in r.faces if f.cluster_id in names})
    if regions is None and not persons and r.content_rating is None:
        return None
    return XmpData(regions=regions, persons=persons or None, content_rating=r.content_rating)


def _source_sidecar(src: Path) -> Path | None:
    """The sidecar already beside *src*: ``photo.jpg.xmp`` (lens, digiKam), else ``photo.xmp`` (Lightroom)."""
    for candidate in (sidecar_path(src), src.with_suffix(".xmp")):
        if candidate.is_file():
            return candidate
    return None


def _write_export_sidecar(src: Path, dst: Path, data: XmpData) -> None:
    """Merge *data* into *dst*'s sidecar, starting from *src*'s when *dst* has none yet.

    The seed is copied first so lens keywords and a photo app's stars survive.
    If the merge then fails, a sidecar we just created is removed: the export
    must not leave a broken ``photo.jpg.xmp`` that was not there before.
    """
    seed = _source_sidecar(src)
    target = sidecar_path(dst)
    created = False
    if seed is not None and not target.exists():
        shutil.copyfile(seed, target)
        created = True
    try:
        write_xmp(dst, data)
    except Exception:
        if created:
            target.unlink(missing_ok=True)
        raise


def write_source_sidecars(results: list[ImageResult], face_names: dict[str, str] | None = None) -> int:
    """Merge curator's metadata into the sidecar beside each scanned original. Returns how many were written.

    Every result with something to record is written, including rejected and
    below-threshold images — this is a library annotate, not an export of
    keepers. This writes into the scanned folder, so only the CLI offers it
    (``--xmp-sources``); the server's write paths stay under the export root.
    Merging keeps every property curator does not set. Regions are replaced as
    a whole, so face regions drawn by hand in a photo app are lost on an image
    with detected faces.
    """
    written = 0
    for r in results:
        data = xmp_data(r, face_names)
        if data is None:
            continue
        try:
            write_xmp(r.abs_path, data)
            written += 1
        except (OSError, XmpError) as exc:
            logger.warning("xmp_write_failed", rel_path=r.rel_path, error=str(exc))
    return written


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

        # A named list overrides neither the under-18 exposed rejection nor the prefilter.
        keep_reason = {
            r.rel_path: hard_exclusion(r) or ("" if r.rel_path in wanted else "not-selected") for r in results
        }
        selected = [r for r in results if keep_reason[r.rel_path] == ""]
    else:
        selected, keep_reason = decide_selection(results, req, summary.config.diversity_weight)

    selected_rel = {r.rel_path for r in selected}
    dest_root = Path(req.dest)
    dest_paths = _plan_dest_paths(selected, req.preserve_structure)

    total = len(selected)
    if progress is not None:
        progress({"phase": "transferring", "done": 0, "total": total})

    copied = 0
    skipped = 0
    xmp_written = 0
    transferred: list[ImageResult] = []
    for i, r in enumerate(selected):
        src = Path(r.abs_path)
        if not src.exists():
            logger.warning("export_source_missing", rel_path=r.rel_path)
            skipped += 1
        else:
            dst = dest_root / dest_paths[r.rel_path]
            try:
                _transfer_one(src, dst, req.mode)
                copied += 1
                transferred.append(r)
            except Exception as exc:
                logger.warning("export_transfer_failed", rel_path=r.rel_path, error=str(exc))
                skipped += 1
            else:
                if req.write_xmp and (data := xmp_data(r, req.face_names)) is not None:
                    try:
                        _write_export_sidecar(src, dst, data)
                        xmp_written += 1
                    except (OSError, XmpError) as exc:
                        # The file landed; a sidecar that cannot be written does not undo that.
                        logger.warning("xmp_write_failed", rel_path=r.rel_path, error=str(exc))
        if progress is not None:
            progress({"phase": "transferring", "done": i + 1, "total": total})

    # Only files that actually landed under dest_root — the manifest and the
    # result must not claim exports that never happened.
    exported = {r.rel_path: dest_paths[r.rel_path] for r in transferred}
    # Resolve the root, not each destination: under mode="symlink" resolving a
    # destination would follow the link straight back to the source, reporting
    # the location the export exists to move away from.
    dest_abs_root = dest_root.resolve()
    exported_abs = {rel: str(dest_abs_root / dest) for rel, dest in exported.items()}

    manifest_path: Path | None = None
    if req.write_manifest:
        manifest_path = dest_root / MANIFEST_NAME
        write_manifest(transferred, summary, manifest_path, exported, exported_abs, req.mode)
        write_report(results, keep_reason, selected_rel, exported, dest_root / REPORT_NAME)

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
        exported_paths=exported,
        exported_abs_paths=exported_abs,
        captioned=captioned,
        xmp_written=xmp_written,
    )
