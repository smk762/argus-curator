"""Transcripts and corpus labels for a speech scan (#186).

A clip's transcript comes from one of three layouts, each read from inside the
scanned folder only (a file whose resolved path leaves it is ignored, as the
scan's own clips are):

* **sidecar**: ``<clip>.txt`` beside the clip (``a/b.wav`` -> ``a/b.txt``).
* **ljspeech**: LJSpeech's ``metadata.csv`` at the folder's top, pipe-separated
  ``id|transcription|normalized transcription`` with no header; the normalized
  text wins when it is there. ``id`` is the clip's file stem.
* **common_voice**: Common Voice's TSVs at the folder's top (``validated.tsv``,
  then ``train.tsv``, ``dev.tsv``, ``test.tsv``: the first that names a clip
  wins). ``path`` names the clip (Common Voice ships MP3; a scan reads WAV or
  FLAC, so a converted clip matches by stem), ``sentence`` is its transcript,
  and ``client_id``, ``accents`` (or ``accent``), ``locale`` and ``age`` are
  carried through verbatim. ``invalidated.tsv`` and ``other.tsv`` are never read.

A corpus maps clips by file stem, so two clips sharing a stem in different
sub-folders both take the row. The Blizzard 2027 layout is not read yet: its
data is not published (#202).
"""

from __future__ import annotations

import csv
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath

import structlog

from argus_curator.models import TranscriptLayout, TranscriptSource

logger = structlog.get_logger()

# A sidecar transcript longer than this is not a clip's transcript.
MAX_SIDECAR_BYTES = 64 * 1024
LJSPEECH_FILE = "metadata.csv"
COMMON_VOICE_FILES = ("validated.tsv", "train.tsv", "dev.tsv", "test.tsv")

# Common Voice age buckets that are, or may be, under 18. "teens" spans 13-19,
# so it is treated as under 18 as a whole: unknown is never read as adult.
UNDER_18_AGE_LABELS = frozenset({"teens", "teen", "child", "children", "kid", "kids"})


@dataclass(frozen=True)
class Transcript:
    """A clip's transcript and the corpus labels that came with it (verbatim, never estimated)."""

    text: str
    source: TranscriptSource
    speaker: str | None = None
    accent: str | None = None
    language: str | None = None
    age: str | None = None


def under_18_label(age: str | None) -> bool:
    """True when a corpus age label marks the speaker as (possibly) under 18.

    A number below 18 counts too. An absent label is not a reading of age, so it is False.
    """
    if age is None:
        return False
    label = age.strip().lower()
    if label in UNDER_18_AGE_LABELS:
        return True
    try:
        return float(label) < 18
    except ValueError:
        return False


def _inside(root: Path, path: Path) -> Path | None:
    """*path* resolved, when it is a regular file inside *root*; else None."""
    try:
        resolved = path.resolve()
    except (OSError, RuntimeError):
        return None
    if not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
        return None
    return resolved


def _clean(text: str | None) -> str | None:
    if text is None:
        return None
    text = " ".join(text.split())
    return text or None


def _sidecar(root: Path, rel: str) -> Transcript | None:
    path = _inside(root, (root / rel).with_suffix(".txt"))
    if path is None:
        return None
    try:
        with path.open("rb") as fh:
            data = fh.read(MAX_SIDECAR_BYTES + 1)
    except OSError as exc:
        logger.warning("speech_sidecar_read_error", rel_path=rel, error=str(exc))
        return None
    if len(data) > MAX_SIDECAR_BYTES:
        logger.warning("speech_sidecar_too_large", rel_path=rel)
        return None
    text = _clean(data.decode("utf-8", errors="replace"))
    return Transcript(text, "sidecar") if text else None


def _text_lines(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        yield from fh


def _ljspeech(root: Path, stems: set[str]) -> dict[str, Transcript]:
    path = _inside(root, root / LJSPEECH_FILE)
    if path is None:
        return {}
    found: dict[str, Transcript] = {}
    try:
        for line in _text_lines(path):
            parts = line.rstrip("\r\n").split("|")
            if len(parts) < 2 or parts[0] not in stems:
                continue
            text = _clean(parts[2] if len(parts) > 2 and parts[2].strip() else parts[1])
            if text:
                found.setdefault(parts[0], Transcript(text, "ljspeech"))
    except OSError as exc:
        logger.warning("speech_metadata_read_error", file=LJSPEECH_FILE, error=str(exc))
    return found


def _rows(path: Path) -> Iterator[dict[str, str]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        # QUOTE_NONE: Common Voice sentences carry bare quotes, and its TSVs are not CSV-quoted.
        yield from csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)


def _common_voice(root: Path, stems: set[str]) -> dict[str, Transcript]:
    found: dict[str, Transcript] = {}
    for name in COMMON_VOICE_FILES:
        path = _inside(root, root / name)
        if path is None:
            continue
        try:
            for row in _rows(path):
                stem = PurePosixPath((row.get("path") or "").strip()).stem
                if stem not in stems or stem in found:
                    continue
                text = _clean(row.get("sentence"))
                if not text:
                    continue
                found[stem] = Transcript(
                    text,
                    "common_voice",
                    speaker=_clean(row.get("client_id")),
                    accent=_clean(row.get("accents") or row.get("accent")),
                    language=_clean(row.get("locale")),
                    age=_clean(row.get("age")),
                )
        except (OSError, csv.Error) as exc:
            logger.warning("speech_metadata_read_error", file=name, error=str(exc))
    return found


def load_transcripts(root: Path, rel_paths: list[str], layout: TranscriptLayout = "auto") -> dict[str, Transcript]:
    """Each clip's :class:`Transcript` by ``rel_path``, from *layout* (see the module docstring)."""
    stem_of = {rel: PurePosixPath(rel).stem for rel in rel_paths}
    stems = set(stem_of.values())
    corpora: list[dict[str, Transcript]] = []
    if layout in ("auto", "ljspeech"):
        corpora.append(_ljspeech(root, stems))
    # Common Voice's labels are read whatever the layout: a sidecar transcript
    # must not hide the corpus marking a speaker as under 18.
    voice = _common_voice(root, stems)
    if layout in ("auto", "common_voice"):
        corpora.append(voice)
    out: dict[str, Transcript] = {}
    for rel in rel_paths:
        found = _sidecar(root, rel) if layout in ("auto", "sidecar") else None
        for corpus in corpora:
            if found is not None:
                break
            found = corpus.get(stem_of[rel])
        labels = voice.get(stem_of[rel])
        if labels is not None and (found is None or found.source != "common_voice"):
            # The labels without the transcript when the layout reads none of its own.
            text, source = (found.text, found.source) if found is not None else ("", "common_voice")
            found = replace(labels, text=text, source=source)
        if found is not None:
            out[rel] = found
    return out
