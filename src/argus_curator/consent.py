"""Real-person face sets bound to a documented consent record (#133).

A face set of a real person may be exported only when every image is of the
person who consented, and the consent is current. Curator collects no consent
and gives no legal advice: it reads the record argus-mask wrote
(:class:`argus_cortex.consent.ConsentRecord`), checks it with cortex's
:func:`~argus_cortex.consent.consent_valid` for the ``avatar`` use of the face,
and binds the set to the record's identity.

- :func:`prepare` (before the scan): read the record, check it, find its
  reference images in the scan by sha256 and build the identity reference from
  them (:func:`argus_curator.faces.reference_embedding`).
- :func:`bind` (after the scan): give each image a
  :class:`~argus_curator.models.ConsentBinding`: its primary face within the
  threshold of the reference, nobody else in the frame, and no sidecar saying
  its subject did not consent.
- :func:`face_source_gate` (at select and export): read the record again, so a
  record revoked since the scan blocks the export, and read each image's
  sidecar again for its provenance and argus-quarry's likeness mark.

Nothing here is kept beyond the scan. The scan stores only the verdict, the
record's hashes and the per-image bindings, all inside the scan's own summary,
so deleting or purging the scan removes them. The record and the reference
embedding are held in memory only.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import structlog
from argus_cortex.consent import ConsentRecord, consent_matches, consent_valid, subject_ref
from argus_cortex.provenance import Provenance, pseudonymise
from pydantic import ValidationError

from argus_curator.models import (
    ConsentBinding,
    ConsentCheck,
    ConsentConfig,
    ConsentExclusion,
    ConsentIneligible,
    ExportRequest,
    FaceConfig,
    ImageResult,
    ScanSummary,
)
from argus_curator.selection import primary_face

logger = structlog.get_logger()

# The use a face source is for: a live face on a screen (argus-mask's avatar).
FACE_SET_USE = "avatar"


class ConsentRefused(ValueError):
    """A consented face set cannot be selected or exported; the message says why."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _now(at: datetime | None) -> datetime:
    return at if at is not None else datetime.now(UTC)


def _record_path(folder: Path, record: str) -> Path:
    # The CLI is unconstrained (an absolute path is taken as given); the server
    # resolves the record under the scan folder before it gets here.
    return folder / record


@dataclass(frozen=True)
class _Read:
    """A record file as read: the record, its file's sha256, or why it is unusable."""

    record: ConsentRecord | None
    record_sha256: str | None
    reason: ConsentIneligible | None = None
    detail: str = ""


def _validation_detail(exc: ValidationError) -> str:
    # Never the input values: a record names a person, and the detail is served.
    return "; ".join(f"{'.'.join(str(p) for p in e['loc']) or 'record'}: {e['msg']}" for e in exc.errors())


def read_record(path: Path) -> _Read:
    """Read and parse the record at *path*. The detail never echoes the path or the record's contents."""
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return _Read(None, None, "missing", "there is no consent record at the scan's consent.record")
    except OSError as exc:
        return _Read(None, None, "missing", f"the consent record cannot be read ({type(exc).__name__})")
    digest = _sha256(data)
    try:
        record = ConsentRecord.model_validate_json(data)
    except ValidationError as exc:
        return _Read(None, digest, "invalid", _validation_detail(exc))
    return _Read(record, digest)


def verdict_problem(record: ConsentRecord, at: datetime) -> tuple[ConsentIneligible, str] | None:
    """Why *record* does not cover a face set at *at* (cortex's refusal and detail), else None."""
    verdict = consent_valid(record, use=FACE_SET_USE, likeness="face", at=at)
    if verdict.ok:
        return None
    assert verdict.reason is not None
    return verdict.reason, verdict.detail


@dataclass
class Prepared:
    """What :func:`prepare` read: the check so far and, when eligible, the reference embedding.

    ``embedding`` is a biometric template: it goes into the scan's
    ``FaceConfig.reference_embedding`` (which a summary never stores) and nowhere else.
    """

    check: ConsentCheck
    record: ConsentRecord | None = None
    embedding: list[float] | None = None


def faces_for_consent(faces_cfg: FaceConfig) -> FaceConfig:
    """*faces_cfg* with what a consent scan needs: faces, MiVOLO and the set age.

    Raises ``ValueError`` when it already names a reference: the record's
    reference images are the reference, and a second one would bind the set to
    someone else.
    """
    if faces_cfg.has_reference:
        raise ValueError(
            "a consent scan takes its identity reference from the record's reference images; "
            "drop faces.reference_paths / reference_embedding"
        )
    return faces_cfg.model_copy(update={"enabled": True, "mivolo": True, "set_age": True})


def prepare(
    folder: Path,
    cfg: ConsentConfig,
    items: list[tuple[str, str, bytes]],
    faces_cfg: FaceConfig,
    *,
    at: datetime | None = None,
    lease_wait_s: float | None = None,
) -> Prepared:
    """Read and check the record, and build the identity reference from its reference images.

    *items* are the scan's ``(rel_path, abs_path, bytes)``; a reference image
    is any of them whose bytes hash to one the record lists. An ineligible
    record comes back as a check that says why, with no embedding: the scan
    still runs, and the export refuses the set.
    """
    assert cfg.record is not None
    at = _now(at)

    def check(**kwargs: object) -> ConsentCheck:
        return ConsentCheck(record=cfg.record, checked_at=at, max_distance=cfg.max_distance, **kwargs)  # type: ignore[arg-type]

    read = read_record(_record_path(folder, cfg.record))
    if read.record is None:
        return Prepared(check(eligible=False, reason=read.reason, detail=read.detail, record_sha256=read.record_sha256))
    record = read.record
    hashes = {"release_sha256": record.release_sha256, "record_sha256": read.record_sha256}
    if (problem := verdict_problem(record, at)) is not None:
        return Prepared(check(eligible=False, reason=problem[0], detail=problem[1], **hashes), record)

    wanted = set(record.reference_images)
    found: dict[str, tuple[str, bytes]] = {}
    for rel, _abs, data in items:
        digest = _sha256(data)
        if digest in wanted and digest not in found:
            found[digest] = (rel, data)
    missing = len(wanted) - len(found)
    references = sorted(rel for rel, _ in found.values())
    if not found:
        return Prepared(
            check(
                eligible=False,
                reason="no-reference-image",
                detail=f"none of the record's {len(wanted)} reference image(s) is in the scanned folder",
                missing_references=missing,
                **hashes,
            ),
            record,
        )
    from argus_curator import faces as faces_mod

    try:
        embedding = faces_mod.reference_embedding(
            [data for _, data in found.values()], faces_cfg, lease_wait_s=lease_wait_s
        )
    except ValueError as exc:
        return Prepared(
            check(
                eligible=False,
                reason="no-reference-face",
                detail=str(exc),
                references=references,
                missing_references=missing,
                **hashes,
            ),
            record,
        )
    if missing:
        logger.warning("consent_references_missing", missing=missing, found=len(found))
    return Prepared(
        check(eligible=True, references=references, missing_references=missing, **hashes),
        record,
        embedding,
    )


@dataclass(frozen=True)
class SidecarFacts:
    """What an image's ``<stem>.json`` sidecar says about consent and provenance.

    ``likeness_consent`` is argus-quarry's mark (False: the subject did not
    consent to their likeness being used). ``provenance`` is the sidecar's
    ``provenance``, read as cortex's :class:`Provenance`. ``problem`` is why
    the sidecar could not be read, or why its provenance is unusable.
    """

    likeness_consent: bool | None = None
    provenance: Provenance | None = None
    problem: str | None = None


def _within(root: Path, path: Path) -> bool:
    try:
        resolved, root = path.resolve(), root.resolve()
    except (OSError, ValueError):
        return False
    return resolved == root or root in resolved.parents


def read_sidecar(abs_path: str, root: Path) -> SidecarFacts:
    """Read the ``<stem>.json`` beside the image at *abs_path* (argus-quarry's provenance sidecar).

    No sidecar is no facts. One that resolves outside *root* (a symlink out of
    the folder), cannot be parsed, or holds a provenance cortex refuses, is a
    problem: the image is refused rather than taken on trust.
    """
    sidecar = Path(abs_path).with_suffix(".json")
    if not sidecar.is_file():
        return SidecarFacts()
    if not _within(root, sidecar):
        return SidecarFacts(problem="its sidecar is outside the scanned folder")
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        return SidecarFacts(problem=f"its sidecar cannot be read ({type(exc).__name__})")
    if not isinstance(data, dict):
        # Some other tool's <stem>.json; it says nothing about consent.
        return SidecarFacts()
    mark = data.get("likeness_consent")
    likeness = mark if isinstance(mark, bool) else None
    if mark is not None and likeness is None:
        return SidecarFacts(problem="its sidecar's likeness_consent is not true or false")
    raw = data.get("provenance")
    if raw is None:
        return SidecarFacts(likeness_consent=likeness)
    try:
        provenance = Provenance.model_validate(raw)
    except ValidationError as exc:
        return SidecarFacts(
            likeness_consent=likeness, problem=f"its provenance is invalid ({exc.error_count()} error(s))"
        )
    return SidecarFacts(likeness_consent=likeness, provenance=provenance)


def _under_release(facts: SidecarFacts, record: ConsentRecord) -> bool:
    """Whether a sidecar's provenance, if any, is this release (cortex's ``consent_matches`` rule)."""
    return facts.provenance is None or consent_matches(record, facts.provenance)


def binding(result: ImageResult, facts: SidecarFacts, record: ConsentRecord, max_distance: float) -> ConsentBinding:
    """Bind one image to *record*: no refusal in its sidecar, one person, and the record's face."""
    face = primary_face(result)
    distance = face.reference_distance if face is not None else None

    def out(reason: ConsentExclusion) -> ConsentBinding:
        return ConsentBinding(bound=False, distance=distance, reason=reason)

    if facts.likeness_consent is False:
        return out("not-consented")
    if facts.problem is not None or not _under_release(facts, record):
        return out("provenance")
    if result.face_error is not None:
        return out("unchecked")
    if face is None:
        return out("no-face")
    # Anyone else in the frame did not consent: another face, or another
    # person the body detector found (a turned head has no face to count).
    if result.face_count > 1 or len(result.people) > 1:
        return out("bystander")
    if distance is None or distance > max_distance:
        return out("mismatch")
    return ConsentBinding(bound=True, distance=distance)


def bind(summary: ScanSummary, prepared: Prepared) -> None:
    """Record the consent check on *summary* and, when eligible, bind every image."""
    check = prepared.check
    if check.eligible and summary.reference_skipped:
        check = check.model_copy(
            update={"eligible": False, "reason": "no-reference-face", "detail": summary.reference_skipped}
        )
    summary.consent = check
    if not check.eligible or prepared.record is None:
        return
    root = Path(summary.folder)
    for r in summary.results:
        r.consent = binding(r, read_sidecar(r.abs_path, root), prepared.record, check.max_distance)


# ---------------------------------------------------------------------------
# Select and export
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConsentHold:
    """A consent record as re-checked for one export: its hashes and each bound image's provenance.

    Each provenance is pseudonymised (cortex's :func:`~argus_cortex.provenance.pseudonymise`): its
    ``subject`` is the record's subject ref, ``consent:<release_sha256>``, never the person's name,
    because it goes into the manifest and into carried sidecars.
    """

    release_sha256: str
    record_sha256: str
    provenance: dict[str, Provenance] = field(default_factory=dict)


@dataclass(frozen=True)
class FaceSourceGate:
    """What the face-source preset learns from disk at select and export time.

    ``excluded`` maps a rel_path to its ``face-source:*`` keep reason: an image
    whose sidecar (read again now) marks its subject as not consenting, or,
    for a consented set, one whose provenance is no longer this release's.
    ``consent`` is set for a consented set (``face_source.consent``).
    """

    excluded: dict[str, str] = field(default_factory=dict)
    consent: ConsentHold | None = None


def _default_provenance(record: ConsentRecord) -> Provenance:
    """An image with no provenance of its own: captured under the release.

    It names no one: the subject is the record's subject ref, and the source leaves out
    who collected the release (``collected_by`` can be a person too).
    """
    return Provenance(
        kind="consented",
        source="captured under the subject's likeness release",
        license="the subject's signed likeness release (release_sha256)",
        subject=subject_ref(record),
        release_sha256=record.release_sha256,
    )


def _consent_hold(summary: ScanSummary, at: datetime) -> tuple[ConsentHold, dict[str, str]]:
    """Re-read and re-check the scan's record; raise :class:`ConsentRefused` when it no longer covers the set."""
    check = summary.consent
    if check is None or not check.eligible:
        raise ConsentRefused("face_source.consent needs a scan whose consent record made the set eligible")
    read = read_record(_record_path(Path(summary.folder), check.record))
    if read.record is None:
        raise ConsentRefused(f"the consent record no longer reads ({read.reason}): {read.detail}")
    record = read.record
    # A re-signed release has a new hash, so a new record: the scan's binding was to the old one.
    if record.release_sha256 != check.release_sha256:
        raise ConsentRefused("the consent record is for another release than the one the scan bound; rescan")
    if (problem := verdict_problem(record, at)) is not None:
        raise ConsentRefused(f"the consent record no longer covers this set ({problem[0]}): {problem[1]}")
    root = Path(summary.folder)
    provenance: dict[str, Provenance] = {}
    excluded: dict[str, str] = {}
    default = _default_provenance(record)
    for r in summary.results:
        if r.consent is None or not r.consent.bound:
            continue
        facts = read_sidecar(r.abs_path, root)
        if facts.likeness_consent is False:
            excluded[r.rel_path] = "face-source:not-consented"
        elif facts.problem is not None or not _under_release(facts, record):
            excluded[r.rel_path] = "face-source:consent-provenance"
        else:
            # A sidecar's own provenance names the subject (consent_matches needs it to); the row must not.
            provenance[r.rel_path] = pseudonymise(facts.provenance) if facts.provenance else default
    assert read.record_sha256 is not None  # a parsed record was read from bytes
    return ConsentHold(record.release_sha256, read.record_sha256, provenance), excluded


def face_source_gate(req: ExportRequest, summary: ScanSummary, *, at: datetime | None = None) -> FaceSourceGate | None:
    """The disk checks behind *req*'s face-source preset, made now; None without the preset.

    Every face-source selection reads each candidate's sidecar again and keeps
    out one argus-quarry marks as not consented: a licence to a photo is not
    its subject's consent. With ``face_source.consent`` the scan's record is
    read and checked again too, so a record revoked, expired or replaced since
    the scan raises :class:`ConsentRefused`, and each bound image gets its
    provenance. Select and export both call this, so they keep the same images.
    """
    fs = req.face_source
    if fs is None:
        return None
    at = _now(at)
    hold: ConsentHold | None = None
    excluded: dict[str, str] = {}
    if fs.consent:
        hold, excluded = _consent_hold(summary, at)
    root = Path(summary.folder)
    for r in summary.results:
        if not r.passed or r.rel_path in excluded or (hold is not None and r.rel_path in hold.provenance):
            continue
        if read_sidecar(r.abs_path, root).likeness_consent is False:
            excluded[r.rel_path] = "face-source:not-consented"
    return FaceSourceGate(excluded=excluded, consent=hold)
