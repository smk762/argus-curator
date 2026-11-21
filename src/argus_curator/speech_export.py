"""Speech export (#187): the kept clips, a ``.txt`` transcript beside each, and ``speech_manifest.jsonl``.

The counterpart of :func:`argus_curator.export.export_selection` for a speech
scan, and reached through it. Clips move by the same rules as images (copy,
symlink or move; ``export._transfer_one``, which never deletes the source a
symlink is to point at) and are named by the same de-collision plan, by stem,
so ``a/x.wav`` and ``a/x.flac`` cannot write one ``a/x.txt``. Each exported
clip gets ``<stem>.txt`` holding its transcript, which is how argus-forge pairs
a training file with its caption. The transcript is written first and the clip
second, and a clip whose transcript or transfer failed has no manifest row.

Every clip carries where it came from. A request names either the licensed
corpus (``speech.corpus`` and ``speech.license``), or a cortex
``ConsentRecord`` for a real person's voice outside one
(``speech.consent_record``): it is read and checked at export, for
``training`` of the ``voice`` at ``speech.consent_delivery``, and binds the
export to the voice of the record's reference recordings (found in the scan
by sha256): each clip's speaker embedding must be within
``speech.consent_max_distance`` of their centroid (#209), and its farthest
1.5 s of speech within ``speech.second_voice_max_distance`` of the centroid
of their own 1.5 s windows (#217).
Clips of anyone else, and clips with a second voice in them, are left out.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import numpy as np
import structlog
from argus_cortex.consent import consent_valid, subject_ref
from argus_cortex.provenance import Provenance

from argus_curator import speakers
from argus_curator.consent import ConsentRefused, read_record
from argus_curator.export import REPORT_NAME, RUN_STATS_NAME, ProgressFn, _plan_dest_paths, _transfer_one
from argus_curator.models import (
    ClipResult,
    ExportRequest,
    ExportResult,
    ScanSummary,
    SpeechManifestRow,
    SpeechSelection,
    SpeechTargetProfile,
    SplitPart,
)
from argus_curator.speech_select import hard_exclusion, plan_speech_selection, speaker_id, speech_problem

logger = structlog.get_logger()

SPEECH_MANIFEST_NAME = "speech_manifest.jsonl"
# The use and likeness a voice consent record must cover for a training set.
VOICE_USE = "training"
CONSENTED_CORPUS = "consented recordings"


def _sha256_file(path: Path) -> str | None:
    h = hashlib.sha256()
    try:
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return None
    return h.hexdigest()


@dataclass(frozen=True)
class VoiceConsentHold:
    """A voice consent record as checked for one export: its hashes, the provenance, and who it binds.

    ``distances`` holds each clip's cosine distance to the reference voice (a
    plain number, for review), ``second_voice`` its farthest window's (#217);
    ``excluded`` the clips the record does not cover, with their keep reason.
    The reference centroid and the window vectors are never held.
    """

    release_sha256: str
    record_sha256: str
    provenance: Provenance
    max_distance: float
    references: tuple[str, ...]
    distances: dict[str, float] = field(default_factory=dict)
    excluded: dict[str, str] = field(default_factory=dict)
    second_voice_max_distance: float = 0.0
    second_voice: dict[str, float] = field(default_factory=dict)


# Why no reference voice could be built. It never falls back to the speaker
# clusters: a cluster can hold a similar voice (its threshold is loose on purpose).
NO_REFERENCE_VOICE = (
    "the record's reference recordings have no speaker embedding in this process (the scan ran without "
    "speech.speakers, embedding them failed, or a restart since the scan dropped the embeddings, which "
    "are held in memory only); rescan with speech.speakers on"
)
# Why the second-voice check (#217) cannot run. A consented export never skips it.
NO_SECOND_VOICE_CHECK = (
    "the scan holds no second-voice windows in this process (it ran without speech.second_voice, or a "
    "restart since the scan dropped them, which are held in memory only), so a clip with a second voice "
    "in it could not be told apart; rescan with speech.speakers and speech.second_voice on"
)
NO_REFERENCE_WINDOWS = (
    "the record's reference recordings have no second-voice windows in this process (the VAD found no "
    "speech in them, or embedding them failed), so the check has no voice to measure against"
)


def voice_consent_gate(
    summary: ScanSummary, speech: SpeechSelection, *, at: datetime | None = None
) -> VoiceConsentHold | None:
    """Read and check ``speech.consent_record`` now, and bind every clip to its voice; None without one.

    The record must cover ``training`` of the subject's ``voice`` at
    ``speech.consent_delivery``, and at least one of its reference recordings
    must be a passing clip in the scan, by sha256, with a speaker embedding
    held for the scan. The centroid of those embeddings is the consented
    voice (#209): a clip is bound when its own embedding is within
    ``speech.consent_max_distance`` of it, whatever cluster it joined, and
    holds no second voice (#217): its second-voice windows' farthest stretch
    (:func:`speakers.second_voice_distance`) is within
    ``speech.second_voice_max_distance`` of the centroid of the reference
    recordings' own windows. Every other clip is excluded,
    as ``consent:other-speaker`` (too far), ``consent:second-voice`` (another
    voice inside it), ``consent:no-embedding`` or ``consent:no-voice-windows``
    (nothing to measure). Raises :class:`ConsentRefused` with the reason
    otherwise: a scan holding no second-voice windows refuses outright, it
    never skips the check, and it never falls back to the clusters. The CLI
    takes the record path as given; the server resolves it under the scan's
    folder first.
    """
    if speech.consent_record is None:
        return None
    at = at if at is not None else datetime.now(UTC)
    read = read_record(Path(summary.folder) / speech.consent_record)
    if read.record is None:
        raise ConsentRefused(f"the consent record does not read ({read.reason}): {read.detail}")
    record = read.record
    verdict = consent_valid(record, use=VOICE_USE, likeness="voice", at=at, delivery=speech.consent_delivery)
    if not verdict.ok:
        raise ConsentRefused(f"the consent record does not cover this set ({verdict.reason}): {verdict.detail}")
    wanted = set(record.reference_recordings)
    references = [c for c in summary.clips if c.passed and _sha256_file(Path(c.abs_path)) in wanted]
    if not references:
        raise ConsentRefused(
            f"none of the record's {len(wanted)} reference recording(s) is a passing clip in the scanned folder"
        )
    held = summary.embeddings
    centroid = speakers.reference_centroid([held[c.rel_path] for c in references if c.rel_path in held])
    if centroid is None:
        raise ConsentRefused(NO_REFERENCE_VOICE)
    windows = summary.voice_windows
    cfg = summary.speech_config
    if not windows or cfg is None or not cfg.second_voice:
        raise ConsentRefused(NO_SECOND_VOICE_CHECK)
    # The second-voice check measures 1.5 s windows against the centroid of the reference
    # recordings' own 1.5 s windows: short windows sit farther from a whole clip's centroid,
    # and matching them halved the false flags (docs/speech.md). A voiceprint too: one check only.
    window_centroid = speakers.reference_centroid(
        [row for c in references if c.rel_path in windows for row in np.asarray(windows[c.rel_path])]
    )
    if window_centroid is None:
        raise ConsentRefused(NO_REFERENCE_WINDOWS)
    distances: dict[str, float] = {}
    second_voice: dict[str, float] = {}
    excluded: dict[str, str] = {}
    for c in summary.clips:
        vector = held.get(c.rel_path)
        distance = speakers.reference_distance(vector, centroid) if vector is not None else None
        if distance is None:
            excluded[c.rel_path] = "consent:no-embedding"
            continue
        distances[c.rel_path] = distance
        segments = windows.get(c.rel_path)
        spread = speakers.second_voice_distance(segments, window_centroid) if segments is not None else None
        if spread is not None:
            second_voice[c.rel_path] = spread
        if distance > speech.consent_max_distance:
            excluded[c.rel_path] = "consent:other-speaker"
        elif spread is None:
            excluded[c.rel_path] = "consent:no-voice-windows"
        elif spread > speech.second_voice_max_distance:
            excluded[c.rel_path] = "consent:second-voice"
    # It names no one: the subject is cortex's subject ref (consent:<release_sha256>), and the
    # source leaves out who collected the release. Every row of the export carries it.
    provenance = Provenance(
        kind="consented",
        source="recorded under the subject's voice release",
        license="the subject's signed likeness release (release_sha256)",
        subject=subject_ref(record),
        release_sha256=record.release_sha256,
    )
    assert read.record_sha256 is not None  # a parsed record was read from bytes
    return VoiceConsentHold(
        record.release_sha256,
        read.record_sha256,
        provenance,
        speech.consent_max_distance,
        tuple(sorted(c.rel_path for c in references)),
        distances,
        excluded,
        speech.second_voice_max_distance,
        second_voice,
    )


def provenance_problem(req: ExportRequest) -> str | None:
    """Why *req* cannot export a speech scan's clips for want of their provenance, else None."""
    if req.speech is None or not req.speech.has_provenance:
        return (
            "a speech export names where its clips came from: speech.corpus and speech.license for a licensed "
            "corpus, or speech.consent_record for a consented voice"
        )
    return None


def _transcript_path(clip_dest: Path) -> Path:
    return clip_dest.with_suffix(".txt")


def _is_source(dst: Path, src: Path) -> bool:
    """True when *dst* already is the file *src* (not a link to it, which an export may replace)."""
    try:
        return dst.exists() and not dst.is_symlink() and src.exists() and os.path.samefile(dst, src)
    except OSError:
        return False


def _write_transcript(path: Path, text: str) -> None:
    """Write *text* to *path* through a temp file and a rename, so a link at *path* is replaced, never followed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(text + "\n", encoding="utf-8")
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def write_manifest(rows: list[SpeechManifestRow], path: Path) -> None:
    """One JSON object per line, one line per exported clip."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(row.model_dump_json() + "\n")


def write_report(
    clips: list[ClipResult],
    keep_reason: dict[str, str],
    exported: dict[str, str],
    split: dict[str, SplitPart],
    path: Path,
    consent_distance: dict[str, float] | None = None,
    second_voice_distance: dict[str, float] | None = None,
) -> None:
    """The per-clip CSV report: every clip, kept or not, and why.

    *consent_distance* (a consented voice, #209) is each clip's distance to
    the reference voice, and *second_voice_distance* (#217) its farthest
    stretch of speech's, so the borderline ones can be reviewed.
    """
    consent_distance = consent_distance or {}
    second_voice_distance = second_voice_distance or {}
    fields = [
        "rel_path",
        "score",
        "keep",
        "keep_reason",
        "exported_path",
        "split",
        "speaker_id",
        "speaker_cluster",
        "speaker_mismatch",
        "consent_distance",
        "second_voice_distance",
        "accent",
        "language",
        "language_predicted",
        "language_confidence",
        "language_mismatch",
        "gender",
        "age",
        "duration_s",
        "passed",
        "reject_reason",
        "is_duplicate",
        "duplicate_of",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for c in sorted(clips, key=lambda c: (speaker_id(c) or "", -c.score, c.rel_path)):
            w.writerow(
                {
                    "rel_path": c.rel_path,
                    "score": round(c.score, 4),
                    "keep": keep_reason.get(c.rel_path, "") == "" and c.rel_path in exported,
                    "keep_reason": keep_reason.get(c.rel_path, ""),
                    "exported_path": exported.get(c.rel_path, ""),
                    "split": split.get(c.rel_path, ""),
                    "speaker_id": speaker_id(c) or "",
                    "speaker_cluster": c.speaker_cluster or "",
                    "speaker_mismatch": c.speaker_mismatch,
                    "consent_distance": consent_distance.get(c.rel_path, ""),
                    "second_voice_distance": second_voice_distance.get(c.rel_path, ""),
                    "accent": c.accent or "",
                    "language": c.language or "",
                    "language_predicted": c.language_predicted or "",
                    "language_confidence": "" if c.language_confidence is None else c.language_confidence,
                    "language_mismatch": c.language_mismatch,
                    "gender": c.gender or "",
                    "age": c.age or "",
                    "duration_s": "" if c.duration_s is None else round(c.duration_s, 3),
                    "passed": c.passed,
                    "reject_reason": c.reject_reason or "",
                    "is_duplicate": c.is_duplicate,
                    "duplicate_of": c.duplicate_of or "",
                }
            )


def export_speech(
    summary: ScanSummary,
    req: ExportRequest,
    progress: ProgressFn | None = None,
    moved: list[str] | None = None,
) -> ExportResult:
    """Select the speech scan's clips per *req*, transfer them with their transcripts, and write the manifest.

    *progress* and *moved* work as :func:`argus_curator.export.export_selection`'s.
    Raises ``ValueError`` for a request a speech scan cannot take (image
    options, no provenance), :class:`argus_curator.selection.SplitError` for
    a split that cannot be made and :class:`ConsentRefused` for a consent
    record that does not cover the set; all of them before anything is transferred.
    """
    if (problem := speech_problem(req, summary)) is not None:
        raise ValueError(problem)
    if (problem := provenance_problem(req)) is not None:
        raise ValueError(problem)
    speech = req.speech or SpeechSelection()
    hold = voice_consent_gate(summary, speech)
    profile = SpeechTargetProfile.model_validate(summary.target_profile)
    clips = summary.clips
    split: dict[str, SplitPart] = {}

    if req.selection is not None:
        wanted = set(req.selection)
        excluded = hold.excluded if hold is not None else {}
        keep_reason = {
            c.rel_path: hard_exclusion(c)
            or excluded.get(c.rel_path)
            or ("no-transcript" if not c.transcript else "")
            or ("" if c.rel_path in wanted else "not-selected")
            for c in clips
        }
        selected = [c for c in clips if keep_reason[c.rel_path] == ""]
    else:
        plan = plan_speech_selection(clips, req, hold.excluded if hold is not None else None)
        selected, keep_reason = plan.chosen, plan.keep_reason
        if req.split is not None:
            split = plan.split
            for c in selected:
                if split[c.rel_path] != req.split_part:
                    keep_reason[c.rel_path] = f"split-{split[c.rel_path].replace('_', '-')}"
            selected = [c for c in selected if split[c.rel_path] == req.split_part]
    if hold is not None and not selected:
        raise ConsentRefused("no clip of the consented voice qualifies; nothing was exported")

    corpus = speech.corpus or CONSENTED_CORPUS
    licence = speech.license or (hold.provenance.license if hold is not None else "")
    provenance = hold.provenance if hold is not None else Provenance(kind="stock", source=corpus, license=licence)
    dest_root = Path(req.dest)
    dest_paths = _plan_dest_paths(selected, req.preserve_structure, by_stem=True)  # type: ignore[arg-type]
    total = len(selected)
    if progress is not None:
        progress({"phase": "transferring", "done": 0, "total": total})

    copied = skipped = 0
    transferred: list[ClipResult] = []
    for i, c in enumerate(selected):
        src = Path(c.abs_path)
        dst = dest_root / dest_paths[c.rel_path]
        txt = _transcript_path(dst)
        if not src.exists():
            logger.warning("export_source_missing", rel_path=c.rel_path)
            skipped += 1
        else:
            wrote_txt = False
            try:
                # Exporting onto the source (dest is the scanned folder, or links into it)
                # would overwrite, then remove, the source's own transcript sidecar.
                if _is_source(dst, src) or _is_source(txt, src.with_suffix(".txt")):
                    raise shutil.SameFileError(f"{dst} is the source file")
                # The transcript first: a clip never lands (or leaves its source) without one.
                _write_transcript(txt, c.transcript or "")
                wrote_txt = True
                _transfer_one(src, dst, req.mode)
                copied += 1
                transferred.append(c)
                if req.mode == "move" and moved is not None:
                    moved.append(c.rel_path)
            except Exception as exc:
                logger.warning("export_transfer_failed", rel_path=c.rel_path, error=str(exc))
                if wrote_txt:
                    txt.unlink(missing_ok=True)
                skipped += 1
        if progress is not None:
            progress({"phase": "transferring", "done": i + 1, "total": total})

    exported = {c.rel_path: dest_paths[c.rel_path] for c in transferred}
    dest_abs_root = dest_root.resolve()
    exported_abs = {rel: str(dest_abs_root / dest) for rel, dest in exported.items()}

    manifest_path: Path | None = None
    if req.write_manifest:
        rows = []
        for c in transferred:
            rel_dest = exported[c.rel_path]
            txt_rel = str(PurePosixPath(rel_dest).with_suffix(".txt"))
            rows.append(
                SpeechManifestRow(
                    rel_path=c.rel_path,
                    abs_path=exported_abs[c.rel_path] if req.mode == "move" else c.abs_path,
                    exported_path=rel_dest,
                    exported_abs_path=exported_abs[c.rel_path],
                    transcript_path=txt_rel,
                    transcript_abs_path=str(dest_abs_root / txt_rel),
                    target_profile=profile,
                    transcript=c.transcript or "",
                    duration_s=round(float(c.duration_s or 0.0), 3),
                    sample_rate=int(c.sample_rate or 0),
                    channels=int(c.channels or 0),
                    speaker_id=speaker_id(c),
                    speaker_label=c.speaker,
                    speaker_cluster=c.speaker_cluster,
                    speaker_mismatch=c.speaker_mismatch,
                    accent=c.accent,
                    language=c.language,
                    gender=c.gender,
                    age=c.age,
                    score=round(c.score, 4),
                    similar_group=c.similar_group,
                    split=split.get(c.rel_path),
                    corpus=corpus,
                    license=licence,
                    provenance=provenance,
                    consent_release_sha256=hold.release_sha256 if hold is not None else None,
                    consent_record_sha256=hold.record_sha256 if hold is not None else None,
                )
            )
        manifest_path = dest_root / SPEECH_MANIFEST_NAME
        write_manifest(rows, manifest_path)
        write_report(
            clips,
            keep_reason,
            exported,
            split,
            dest_root / REPORT_NAME,
            hold.distances if hold is not None else None,
            hold.second_voice if hold is not None else None,
        )

    run_stats_path: Path | None = None
    if req.run_stats is not None and (transferred or req.write_manifest):
        target = dest_root / RUN_STATS_NAME
        try:
            dest_root.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(req.run_stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            run_stats_path = target
        except OSError as exc:
            logger.warning("run_stats_write_failed", path=str(target), error=str(exc))

    logger.info(
        "speech_export_done", dest=str(dest_root), mode=req.mode, copied=copied, skipped=skipped, selected=total
    )
    return ExportResult(
        modality="speech",
        manifest_path=str(manifest_path) if manifest_path else None,
        copied=copied,
        skipped=skipped,
        dest=str(dest_abs_root) if dest_root.exists() else str(dest_root),
        mode=req.mode,
        selected_rel_paths=sorted(c.rel_path for c in selected),
        exported_paths=exported,
        exported_abs_paths=exported_abs,
        transcripts_written=len(transferred),
        run_stats_path=str(run_stats_path) if run_stats_path else None,
    )
