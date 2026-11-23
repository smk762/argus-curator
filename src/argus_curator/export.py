"""Export — structure-preserving transfer + the JSONL manifest argus-lens consumes.

Ported from imogen PR #14's ``_transfer`` / ``_write_csv`` and extended with the
handoff manifest. The manifest is one JSON object per exported image carrying
the shared ``target_profile``, so argus-lens can batch-caption it with no
category remapping (section 8 of the brief).
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import secrets
import shutil
import unicodedata
from collections import Counter
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

import structlog
from argus_cortex.moderation import rule_letter
from argus_cortex.provenance import Provenance
from argus_cortex.xmp import XmpData, XmpError, XmpRegion, XmpRegions, sidecar_path, write_xmp

from argus_curator import crops
from argus_curator.consent import ConsentHold, ConsentRefused, FaceSourceGate, face_source_gate
from argus_curator.models import (
    ExportMode,
    ExportRequest,
    ExportResult,
    FaceDetection,
    ImageResult,
    ManifestFace,
    ManifestRow,
    ScanSummary,
    SplitPart,
)
from argus_curator.selection import (
    consent_set_problem,
    hard_exclusion,
    plan_selection,
    primary_face,
    scan_references,
)

logger = structlog.get_logger()

# A progress sink: called with {"phase": str, "done": int, "total": int}.
ProgressFn = Callable[[dict[str, Any]], None]

MANIFEST_NAME = "manifest.jsonl"
REPORT_NAME = "curation_report.csv"
RUN_STATS_NAME = "run_stats.json"
# Internal hosts the export's caption_url hand-off may reach (see caption_allowed_hosts).
CAPTION_HOSTS_ENV = "CURATOR_CAPTION_ALLOWED_HOSTS"

# The XMP namings, the one to merge into first (see _source_sidecar).
_XMP_NAMES: tuple[Callable[[Path], Path], ...] = (
    sidecar_path,  # photo.jpg.xmp (lens, digiKam)
    lambda p: p.with_suffix(".xmp"),  # photo.xmp (Lightroom)
)

# Where each sidecar ExportRequest.carry_sidecars carries lives, as a function
# of its image's path. Applied to the source to find it and to the destination
# to name it, so a de-collided "IMG-1a2b3c4d.jpg" gets "IMG-1a2b3c4d.txt".
_SIDECAR_NAMES: tuple[Callable[[Path], Path], ...] = (
    lambda p: p.with_suffix(".txt"),  # argus-lens captions
    *_XMP_NAMES,
    lambda p: p.with_suffix(".json"),
)


def _norm_name(name: str) -> str:
    """Collision key for a basename.

    Destination filesystems may be case-insensitive (APFS/NTFS/exFAT/SMB) or
    Unicode-normalising, so names differing only in case or normal form must
    count as one destination entry.
    """
    return unicodedata.normalize("NFC", name).casefold()


# The export's own files sit at the root of the destination. With sidecars
# carried, an image there may not take one of these stems: "run_stats.png"
# would carry a "run_stats.json" that the run stats then overwrite.
_RESERVED_STEMS = frozenset(_norm_name(PurePosixPath(n).stem) for n in (MANIFEST_NAME, REPORT_NAME, RUN_STATS_NAME))


def _suffixed(rel_path: str, digest_len: int) -> str:
    # surrogateescape keeps un-decodable filesystem bytes hashable; the hash is
    # naming-only, so opt out of FIPS restrictions on sha1.
    digest = hashlib.sha1(rel_path.encode("utf-8", "surrogateescape"), usedforsecurity=False).hexdigest()
    p = PurePosixPath(rel_path)
    return f"{p.stem}-{digest[:digest_len]}{p.suffix}"


def _plan_dest_paths(selected: list[ImageResult], preserve_structure: bool, by_stem: bool = False) -> dict[str, str]:
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

    With *by_stem* (carried sidecars) names collide on their stem instead:
    ``a/IMG.jpg`` and ``b/IMG.png`` would otherwise both carry to ``IMG.txt``,
    one caption overwriting the other. A name at the destination root also
    collides with :data:`_RESERVED_STEMS`, flattened or not: with
    *preserve_structure* that is the one case a file does not keep its rel_path.
    """

    def key(name: str) -> str:
        return _norm_name(PurePosixPath(name).stem if by_stem else name)

    def dest_key(dest: str) -> tuple[str, str]:
        # Collisions are per directory: a/IMG.jpg and b/IMG.jpg never meet.
        p = PurePosixPath(dest)
        return _norm_name(str(p.parent)), key(p.name)

    reserved = _RESERVED_STEMS if by_stem else frozenset()
    dests: dict[str, str] = {}
    digest_len: dict[str, int] = {}
    if preserve_structure:
        dests = {r.rel_path: r.rel_path for r in selected}
        by_key: dict[tuple[str, str], list[str]] = {}
        for rel in dests:
            by_key.setdefault(dest_key(rel), []).append(rel)
        for (parent, name), rels in by_key.items():
            # Same directory, same key: a/IMG.jpg + a/IMG.png would carry to one a/IMG.txt.
            if len(rels) > 1 or (parent == "." and name in reserved):
                logger.warning("export_basename_collision", basename=PurePosixPath(rels[0]).name, rel_paths=rels)
                digest_len.update(dict.fromkeys(rels, 8))
    else:
        by_name: dict[str, list[str]] = {}
        for r in selected:
            by_name.setdefault(key(PurePosixPath(r.rel_path).name), []).append(r.rel_path)
        for name, rels in by_name.items():
            if len(rels) == 1 and name not in reserved:
                dests[rels[0]] = PurePosixPath(rels[0]).name
            else:
                logger.warning("export_basename_collision", basename=PurePosixPath(rels[0]).name, rel_paths=rels)
                digest_len.update(dict.fromkeys(rels, 8))

    pending = set(digest_len)
    while pending:
        for rel in pending:
            name = _suffixed(rel, digest_len[rel])
            parent = PurePosixPath(rel).parent
            dests[rel] = str(parent / name) if preserve_structure and str(parent) != "." else name
        counts = Counter(dest_key(d) for d in dests.values())
        pending = {rel for rel in digest_len if counts[dest_key(dests[rel])] > 1}
        stuck = sorted(rel for rel in pending if digest_len[rel] >= 40)
        if stuck:
            raise ValueError(f"cannot de-collide flattened basenames for: {stuck}")
        for rel in pending:
            digest_len[rel] += 8
    return dests


def _same_entry(a: Path, b: Path) -> bool:
    """Whether *a* and *b* name one directory entry, a link itself rather than its target."""
    try:
        return os.path.samestat(os.lstat(a), os.lstat(b))
    except OSError:
        return False


def _transfer_one(src: Path, dst: Path, mode: ExportMode) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    # A source that is itself a link may be dst (dest == source folder, or a parent
    # symlink): unlinking dst would delete it. Left alone, copy2 refuses it as the
    # same file and move renames it onto itself.
    if mode != "symlink" and dst.is_symlink() and _same_entry(src, dst):
        if mode == "copy":
            raise shutil.SameFileError(f"{dst} is the source file")
        return
    if mode == "move":
        if dst.is_symlink():
            # A link left by an earlier symlink export: across filesystems shutil.move
            # falls back to copy2, which would write through it onto the link's target.
            dst.unlink()
        shutil.move(str(src), str(dst))
    elif mode == "symlink":
        target = src.resolve()
        if dst.is_symlink():
            dst.unlink()  # removes only the link (e.g. an earlier symlink export's)
        elif dst.exists():
            # dst may *be* the source (dest == source folder, or a parent symlink):
            # unlinking it would delete the original, as copy refuses to do.
            if os.path.samefile(dst, target):
                raise shutil.SameFileError(f"{dst} is the source file")
            dst.unlink()
        dst.symlink_to(target)
    else:  # copy
        if dst.is_symlink():
            # A link left by an earlier symlink export: copying through it would overwrite its target.
            dst.unlink()
        shutil.copy2(src, dst)


def _write_text_atomically(path: Path, text: str) -> None:
    """Write *text* to *path* through a fresh temp file and a rename; a failure leaves *path* as it was.

    The temp file gets a random name opened ``O_EXCL | O_NOFOLLOW``, so a link
    planted at it is never written through, and ``os.replace`` swaps a link at
    *path* for the file rather than following it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o666)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _sidecar_users(results: list[ImageResult]) -> Counter[Path]:
    """How many of the scan's images each sidecar path belongs to.

    An image the export leaves behind counts, so a move does not take the
    sidecar it shares with a keeper. An image already gone from the source does
    not: it will never take its sidecar, and counting it would strand the
    original in the source folder.
    """
    users: Counter[Path] = Counter()
    for r in results:
        src = Path(r.abs_path)
        if src.exists():
            users.update({name(src) for name in _SIDECAR_NAMES})
    return users


def _write_pseudonymised_sidecar(side: Path, target: Path, provenance: Provenance | None) -> None:
    """Write a copy of the ``<stem>.json`` *side* to *target*, its ``provenance`` swapped for *provenance*.

    For a consented face set: the source sidecar's provenance names the subject (cortex's
    ``consent_matches`` needs it to), and the export must not. *provenance* is the image's
    pseudonymised provenance from the :class:`~argus_curator.consent.ConsentHold`. A sidecar
    with no ``provenance`` is copied as it is. The source is only read, under every mode:
    a move leaves it where it is rather than carry the name along.

    Raises when the sidecar has a provenance and *provenance* is None (the image is not
    one the hold covers), so the name is never carried; and when *target* is *side*.
    """
    data = json.loads(side.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "provenance" in data:
        if provenance is None:
            raise ValueError("its sidecar names a provenance the consent hold does not cover")
        data["provenance"] = provenance.model_dump(mode="json")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not target.is_symlink() and os.path.samefile(target, side):
        raise shutil.SameFileError(f"{target} is the source sidecar")
    tmp = target.with_name(f".{target.name}.tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        # os.replace swaps a link left at *target* for the file, never writing through it.
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)


def _carry_sidecars(
    src: Path,
    dst: Path,
    mode: ExportMode,
    rel_path: str,
    waiting: Counter[Path],
    consent: ConsentHold | None = None,
) -> tuple[int, int]:
    """Transfer the sidecars beside *src* to the matching names beside *dst*. Returns ``(carried, failed)``.

    Same mode and the same overwrite semantics as the image (:func:`_transfer_one`):
    an existing destination sidecar is replaced, as an existing destination
    image would be. The exception is ``mode="symlink"``: sidecars are *copied*,
    because downstream tools (argus-lens captioning) write ``<stem>.txt`` /
    ``.json`` beside the exported image, and through a link that write would
    land in the source library. Called only after the image landed; a sidecar that fails is
    logged and counted, never undoing the image.

    *waiting* counts the images that still need each sidecar where it is (see
    :func:`_sidecar_users`). Under ``mode="move"`` a sidecar two images share
    (``photo.jpg`` and ``photo.png`` both use ``photo.txt``) is copied until
    only this image is waiting, so every image gets it, as under copy, and an
    image that is not exported keeps it. A failed
    transfer never counts down: an image that could not leave, or a sidecar that
    did not reach it, keeps the source sidecar where it is.

    With *consent* (a consented face set) the ``<stem>.json`` is written as a
    pseudonymised copy (:func:`_write_pseudonymised_sidecar`) under every mode,
    so no carried sidecar names the subject and the source is never changed.
    """
    carried = failed = 0
    seen: set[Path] = set()
    for name in _SIDECAR_NAMES:
        side = name(src)
        if side in seen or not side.is_file():
            continue
        seen.add(side)
        target = name(dst)
        try:
            if target.is_dir() and not target.is_symlink():
                # copy2 / move would put the sidecar *inside* it, under a name no consumer looks for.
                raise IsADirectoryError(f"destination is a directory: {target}")
            if consent is not None and side.suffix.lower() == ".json":
                _write_pseudonymised_sidecar(side, target, consent.provenance.get(rel_path))
            else:
                # A symlinked image is read-only by nature; a symlinked sidecar would be a write path into the source.
                _transfer_one(side, target, "copy" if mode == "symlink" or waiting[side] > 1 else mode)
            waiting[side] -= 1
            carried += 1
        except Exception as exc:
            logger.warning("export_sidecar_failed", rel_path=rel_path, sidecar=side.name, error=str(exc))
            failed += 1
    return carried, failed


def write_manifest(
    exported: list[ImageResult],
    summary: ScanSummary,
    path: Path,
    exported_paths: dict[str, str],
    exported_abs_paths: dict[str, str] | None = None,
    mode: ExportMode = "copy",
    *,
    landmarks: bool = False,
    split: dict[str, SplitPart] | None = None,
    consent: ConsentHold | None = None,
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

    Each row carries the primary face's measurements (``primary_face``), with
    its landmarks only when *landmarks* asks, and the scan's set age; with
    *split* (rel_path -> part), its part of the train/held-out split. With
    *consent* (a consented face set), the record's hashes and the image's provenance.
    """
    abs_paths = exported_abs_paths or {}
    parts = split or {}
    # Every row is built before the file is touched: a row that raises must not
    # leave the captioner a truncated work list where the last manifest was.
    lines: list[str] = []
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
            crop=r.crop,
            primary_face=_manifest_face(r, landmarks),
            set_age=summary.set_age,
            split=parts.get(r.rel_path),
            consent_release_sha256=consent.release_sha256 if consent else None,
            consent_record_sha256=consent.record_sha256 if consent else None,
            provenance=consent.provenance[r.rel_path] if consent else None,
            pose_labels=r.pose_labels,
            pose_cluster=r.pose_cluster,
        )
        lines.append(row.model_dump_json() + "\n")
    _write_text_atomically(path, "".join(lines))


def _manifest_face(r: ImageResult, landmarks: bool) -> ManifestFace | None:
    face = primary_face(r)
    if face is None:
        return None
    return ManifestFace(
        bbox=face.bbox,
        yaw=face.yaw,
        pitch=face.pitch,
        roll=face.roll,
        sharpness=face.sharpness,
        eye_open=face.eye_open,
        mouth_open=face.mouth_open,
        cluster_distance=face.cluster_distance,
        reference_distance=face.reference_distance,
        age_years=face.attributes.age_years if face.attributes is not None else None,
        landmarks=face.landmarks if landmarks else None,
    )


def _blank(value: object) -> object:
    return "" if value is None else value


# A spreadsheet reads a cell starting with one of these as a formula (or DDE).
_FORMULA_LEADS = ("=", "+", "-", "@", "\t", "\r")


def _text_cell(value: str) -> str:
    """*value* (a path or label from outside) made safe to open in a spreadsheet: a formula lead gets a quote."""
    return f"'{value}" if value.startswith(_FORMULA_LEADS) else value


def _face_columns(face: FaceDetection | None) -> dict[str, object]:
    """The report's face-quality columns for the primary face (blank without one)."""
    if face is None:
        return dict.fromkeys(
            (
                "primary_face_px",
                "primary_face_sharpness",
                "primary_face_eye_open",
                "primary_face_mouth_open",
                "primary_face_cluster_distance",
                "primary_face_reference_distance",
            ),
            "",
        )
    return {
        "primary_face_px": round(min(face.bbox[2], face.bbox[3])),
        "primary_face_sharpness": _blank(face.sharpness),
        "primary_face_eye_open": _blank(face.eye_open),
        "primary_face_mouth_open": _blank(face.mouth_open),
        "primary_face_cluster_distance": _blank(face.cluster_distance),
        "primary_face_reference_distance": _blank(face.reference_distance),
    }


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
        "duplicate_kind",
        "profile_fit",
        "junk",
        "junk_kind",
        "sharpness",
        "artifact_score",
        "face_count",
        "primary_face_cluster",
        "primary_face_pose",
        "primary_face_yaw",
        "primary_face_pitch",
        "primary_face_roll",
        "primary_face_px",
        "primary_face_sharpness",
        "primary_face_eye_open",
        "primary_face_mouth_open",
        "primary_face_cluster_distance",
        "primary_face_reference_distance",
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
        "pose_check",
        "pose_stance",
        "pose_arms",
        "pose_facing",
        "pose_cluster",
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
                    "rel_path": _text_cell(r.rel_path),
                    "score": round(r.score, 4),
                    "keep": r.rel_path in selected_rel,
                    "keep_reason": keep_reason.get(r.rel_path, ""),
                    "exported_path": _text_cell(exported_paths.get(r.rel_path, "")),
                    "similar_group": r.similar_group,
                    "group_size": r.group_size,
                    "is_representative": r.is_representative,
                    "passed": r.passed,
                    "reject_reason": r.reject_reason or "",
                    "is_duplicate": r.is_duplicate,
                    "duplicate_of": _text_cell(r.duplicate_of or ""),
                    "duplicate_kind": r.duplicate_kind or "",
                    "profile_fit": ""
                    if r.zero_shot is None or r.zero_shot.profile_fit is None
                    else r.zero_shot.profile_fit,
                    "junk": "" if r.zero_shot is None or r.zero_shot.junk is None else r.zero_shot.junk,
                    "junk_kind": (r.zero_shot.junk_kind if r.zero_shot else None) or "",
                    "sharpness": round(r.sharpness, 2),
                    "artifact_score": round(r.artifact_score, 4),
                    "face_count": r.face_count,
                    "primary_face_cluster": r.primary_face_cluster or "",
                    "primary_face_pose": r.primary_face_pose or "",
                    "primary_face_yaw": "" if r.primary_face_yaw is None else round(r.primary_face_yaw, 1),
                    "primary_face_pitch": "" if r.primary_face_pitch is None else round(r.primary_face_pitch, 1),
                    "primary_face_roll": "" if r.primary_face_roll is None else round(r.primary_face_roll, 1),
                    **_face_columns(primary_face(r)),
                    **{f"primary_face_{k}": _blank(getattr(r.primary_face_attributes, k, None)) for k in attr_fields},
                    "people_count": len(r.people),
                    "body_ages": ";".join(
                        f"{p.age_years:.1f}" for p in r.people if p.face_index is None and p.age_years is not None
                    ),
                    "pose_check": r.pose_check or "",
                    "pose_stance": r.pose_labels.stance if r.pose_labels else "",
                    "pose_arms": r.pose_labels.arms if r.pose_labels else "",
                    "pose_facing": r.pose_labels.facing if r.pose_labels else "",
                    "pose_cluster": _blank(r.pose_cluster),
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
                    "abs_path": _text_cell(r.abs_path),
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


def xmp_data(r: ImageResult, face_names: dict[str, str] | None = None, *, regions: bool = True) -> XmpData | None:
    """What curator records about *r* in a sidecar, or ``None`` when it knows nothing worth writing.

    Face regions are fractions of the frame curator scanned; *regions* False
    leaves them out for a file that is not that frame.

    A field curator has nothing for stays ``None``, so merging leaves whatever
    the sidecar already holds there. Age estimates are never written: a MiVOLO
    guess in a photo library would read as a fact.
    """
    names = face_names or {}
    face_regions = None
    if regions and r.faces and r.width > 0 and r.height > 0:
        face_regions = XmpRegions(
            width=r.width, height=r.height, regions=[_region(face, r.width, r.height, names) for face in r.faces]
        )
    persons = sorted({names[f.cluster_id] for f in r.faces if f.cluster_id in names})
    if face_regions is None and not persons and r.content_rating is None:
        return None
    return XmpData(regions=face_regions, persons=persons or None, content_rating=r.content_rating)


def _source_sidecar(src: Path) -> Path | None:
    """The sidecar already beside *src*: ``photo.jpg.xmp`` (lens, digiKam), else ``photo.xmp`` (Lightroom)."""
    for name in _XMP_NAMES:
        if (candidate := name(src)).is_file():
            return candidate
    return None


def _write_export_sidecar(src: Path, dst: Path, data: XmpData, carried: bool = False) -> None:
    """Merge *data* into *dst*'s sidecar, starting from *src*'s when *dst* has none yet.

    The seed is copied first so lens keywords and a photo app's stars survive.
    If the merge then fails, a sidecar we just created is removed: the export
    must not leave a broken ``photo.jpg.xmp`` that was not there before.

    With *carried* the source's sidecars were already transferred beside *dst*
    (under ``mode="move"`` they are no longer beside *src*), so a carried
    ``photo.jpg.xmp`` is merged in place and a carried Lightroom ``photo.xmp``
    is the seed. Under ``mode="symlink"`` the carried sidecars are copies, so
    the merge never reaches the source.
    """
    seed = (_source_sidecar(dst) if carried else None) or _source_sidecar(src)
    target = sidecar_path(dst)
    # Whatever is at *target* after a failed merge and was not there before is ours
    # to remove, whether the seed put it there or write_xmp started it from nothing.
    created = not target.exists()
    try:
        if seed is not None and created:
            shutil.copyfile(seed, target)
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
    with detected faces. A cropped image gets no regions: curator's are of the
    cropped frame, and this sidecar describes the uncropped original.
    """
    written = 0
    for r in results:
        data = xmp_data(r, face_names, regions=r.crop is None)
        if data is None:
            continue
        try:
            write_xmp(r.abs_path, data)
            written += 1
        except (OSError, XmpError) as exc:
            logger.warning("xmp_write_failed", rel_path=r.rel_path, error=str(exc))
    return written


def caption_allowed_hosts() -> list[str]:
    """``CURATOR_CAPTION_ALLOWED_HOSTS``: the only hosts a ``caption_url`` may name.

    Comma-separated ``host`` or ``host:port`` entries (a lens such as
    ``lens:8100``). Empty by default, which turns the hand-off off. Read on every
    call, so a test or a restart picks up a change.
    """
    return [v.strip() for v in os.environ.get(CAPTION_HOSTS_ENV, "").split(",") if v.strip()]


def caption_host(caption_url: str) -> str:
    """The host of *caption_url* for logs and messages: never the full URL, which may carry credentials."""
    try:
        return urlsplit(caption_url).hostname or "?"
    except ValueError:
        return "?"


def caption_url_problem(caption_url: str) -> str | None:
    """Why *caption_url* may not be sent the manifest, or ``None`` when its host is configured.

    Only a host listed in ``CURATOR_CAPTION_ALLOWED_HOSTS`` is reached, public
    or not: a ``host`` entry allows any port, a ``host:port`` entry that port.
    Never names the URL, which may carry credentials.
    """
    allowed = caption_allowed_hosts()
    if not allowed:
        return f"no caption host is configured: set {CAPTION_HOSTS_ENV} (e.g. lens:8100)"
    try:
        parts = urlsplit(caption_url)
        host, port = (parts.hostname or "").lower(), parts.port
    except ValueError:
        return "caption_url is not a valid URL"
    if parts.scheme not in ("http", "https") or not host:
        return "caption_url must be an http(s) URL"
    port = port or (443 if parts.scheme == "https" else 80)
    for entry in (e.lower() for e in allowed):
        if entry.startswith("["):  # [::1] or [::1]:8100
            name, _, rest = entry[1:].partition("]")
            entry_port = rest.removeprefix(":")
        elif entry.count(":") == 1:
            name, _, entry_port = entry.partition(":")
        else:  # a bare name, or a bare IPv6 address
            name, entry_port = entry, ""
        if name == host and (not entry_port or entry_port == str(port)):
            return None
    return f"{host}:{port} is not listed in {CAPTION_HOSTS_ENV}"


def _post_manifest_to_lens(manifest_path: Path, caption_url: str) -> bool:
    """Optionally POST the manifest to argus-lens for a one-click curate->caption run.

    Only a host listed in ``CURATOR_CAPTION_ALLOWED_HOSTS`` is sent it
    (:func:`caption_url_problem`), public or not. It goes through
    :func:`argus_cortex.net.fetch`, which pins the resolved address and never
    re-sends the POST to a redirect. Any refusal or failure is logged (host
    only) and returns False: the export itself has already succeeded.
    """
    host = caption_host(caption_url)
    if (problem := caption_url_problem(caption_url)) is not None:
        logger.warning("caption_handoff_failed", host=host, error=problem)
        return False
    try:
        import httpx
    except Exception:
        logger.warning("caption_handoff_skipped", reason="httpx not installed")
        return False
    try:
        from argus_cortex.net import fetch
    except Exception:
        logger.warning("caption_handoff_skipped", reason="argus-cortex>=0.8 (argus_cortex.net) not installed")
        return False
    try:
        # httpx encodes the multipart body; fetch sends it.
        encoded = httpx.Request(
            "POST",
            "http://encode.invalid/",
            files={"manifest": (MANIFEST_NAME, manifest_path.read_bytes(), "application/x-ndjson")},
        )
        reply = fetch(
            caption_url,
            method="POST",
            content=encoded.read(),
            headers={"Content-Type": encoded.headers["Content-Type"]},
            allow_hosts=caption_allowed_hosts(),
            timeout=30.0,
        )
    except Exception as exc:
        logger.warning("caption_handoff_failed", host=host, error=str(exc))
        return False
    if not reply.ok:
        logger.warning("caption_handoff_failed", host=host, error=f"{host} answered {reply.status}")
        return False
    return True


def export_selection(
    summary: ScanSummary,
    req: ExportRequest,
    progress: ProgressFn | None = None,
    moved: list[str] | None = None,
) -> ExportResult:
    """Select per the request, transfer files, and write the manifest/report.

    When *progress* is supplied it is called with ``{phase, done, total}`` dicts
    during the ``transferring`` phase (one update per file), so a caller can
    drive a live progress bar (see the /export/stream SSE endpoint).

    Under ``mode="move"``, *moved* collects each rel_path as its file leaves the
    source. It is filled as the export goes, so a caller still learns what moved
    when a later step raises (a full disk while writing the report) and no
    :class:`ExportResult` comes back.
    """
    if summary.is_speech:
        # A speech scan's clips get their own rows (SpeechManifestRow) in their own file (#187).
        from argus_curator.speech_export import export_speech

        return export_speech(summary, req, progress=progress, moved=moved)
    results = summary.results
    split: dict[str, SplitPart] = {}
    gate: FaceSourceGate | None = None

    if req.selection is not None:
        wanted = set(req.selection)
        fs = req.face_source
        if fs is not None and fs.consent:
            # ExportRequest refuses this pairing; one built without validation must not skip the consent gate.
            raise ConsentRefused("a consented face set is chosen by face_source, not by a named selection")
        # argus-quarry's likeness_consent: false, read from disk now, holds over a named list too.
        gate = face_source_gate(req, summary)
        refused = gate.excluded if gate is not None else {}

        # A named list overrides neither the under-18 exposed rejection nor any reject.
        keep_reason = {
            r.rel_path: hard_exclusion(r) or refused.get(r.rel_path) or ("" if r.rel_path in wanted else "not-selected")
            for r in results
        }
        selected = [r for r in results if keep_reason[r.rel_path] == ""]
    else:
        # The same plan /scan/{id}/select reports: filter, cap, ranking, split.
        # A consented set is checked from the scan, then against the record as it reads now.
        fs = req.face_source
        if fs is not None and fs.consent and (problem := consent_set_problem(summary)) is not None:
            raise ConsentRefused(problem)
        gate = face_source_gate(req, summary)
        plan = plan_selection(
            results,
            req,
            summary.config.diversity_weight,
            set_age=summary.set_age,
            embeddings=summary.embeddings,
            gate=gate,
            references=scan_references(summary),
        )
        selected, keep_reason = plan.chosen, plan.keep_reason
        if req.split is not None:
            split = plan.split
            # Only split_part is transferred; the other part is accounted for, not lost.
            for r in selected:
                if split[r.rel_path] != req.split_part:
                    keep_reason[r.rel_path] = f"split-{split[r.rel_path].replace('_', '-')}"
            selected = [r for r in selected if split[r.rel_path] == req.split_part]
        if gate is not None and gate.consent is not None and not selected:
            raise ConsentRefused("no image qualifies for the consented face set; nothing was exported")

    selected_rel = {r.rel_path for r in selected}
    # A cropped image is written cropped; a link or a move would hand on the
    # source, watermark and all. Refused before anything is transferred.
    cropped_sel = [r.rel_path for r in selected if r.crop is not None]
    if cropped_sel and req.mode != "copy":
        raise ValueError(
            f'{len(cropped_sel)} selected image(s) were scanned cropped and export only by mode "copy" '
            f"(e.g. {cropped_sel[0]})"
        )
    dest_root = Path(req.dest)
    dest_paths = _plan_dest_paths(selected, req.preserve_structure, by_stem=req.carry_sidecars)

    total = len(selected)
    if progress is not None:
        progress({"phase": "transferring", "done": 0, "total": total})

    copied = 0
    skipped = 0
    xmp_written = 0
    sidecars_carried = 0
    sidecars_failed = 0
    cropped = 0
    transferred: list[ImageResult] = []
    waiting = _sidecar_users(results) if req.carry_sidecars and req.mode == "move" else Counter()
    for i, r in enumerate(selected):
        src = Path(r.abs_path)
        if not src.exists():
            logger.warning("export_source_missing", rel_path=r.rel_path)
            skipped += 1
        else:
            dst = dest_root / dest_paths[r.rel_path]
            try:
                if r.crop is not None:
                    crops.write_cropped(src, dst, r.crop)
                    cropped += 1
                else:
                    _transfer_one(src, dst, req.mode)
                copied += 1
                transferred.append(r)
                if req.mode == "move" and moved is not None:
                    moved.append(r.rel_path)
            except Exception as exc:
                logger.warning("export_transfer_failed", rel_path=r.rel_path, error=str(exc))
                skipped += 1
            else:
                # Carry first, then merge: write_xmp then lands on the carried
                # sidecar, so under move nothing is left behind and the result
                # is the source's sidecar plus curator's data.
                if req.carry_sidecars:
                    hold = gate.consent if gate is not None else None
                    carried, failed = _carry_sidecars(src, dst, req.mode, r.rel_path, waiting, hold)
                    sidecars_carried += carried
                    sidecars_failed += failed
                if req.write_xmp and (data := xmp_data(r, req.face_names)) is not None:
                    try:
                        _write_export_sidecar(src, dst, data, carried=req.carry_sidecars)
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
        write_manifest(
            transferred,
            summary,
            manifest_path,
            exported,
            exported_abs,
            req.mode,
            landmarks=req.landmarks,
            split=split or None,
            consent=gate.consent if gate is not None else None,
        )
        write_report(results, keep_reason, selected_rel, exported, dest_root / REPORT_NAME)

    run_stats_path: Path | None = None
    # Only where the export produced a folder: a run that transferred nothing
    # and wrote no manifest should not create dest just to hold the stats.
    if req.run_stats is not None and (transferred or req.write_manifest):
        target = dest_root / RUN_STATS_NAME
        try:
            dest_root.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(req.run_stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            run_stats_path = target
        except OSError as exc:
            logger.warning("run_stats_write_failed", path=str(target), error=str(exc))

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
        sidecars_carried=sidecars_carried,
        sidecars_failed=sidecars_failed,
        cropped=cropped,
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
        sidecars_carried=sidecars_carried,
        sidecars_failed=sidecars_failed,
        cropped=cropped,
        run_stats_path=str(run_stats_path) if run_stats_path else None,
    )
