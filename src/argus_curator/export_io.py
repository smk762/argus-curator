"""Export I/O: the modality-neutral half of an export, shared by the image and speech exports.

Destination naming (:func:`plan_dest_paths`), the file transfer that never
deletes its source (:func:`transfer_one`), atomic text writes
(:func:`write_text_atomically`), spreadsheet-safe CSV cells (:func:`text_cell`)
and the names of the export's own files. It imports neither ``export`` nor
``speech_export``, so both can import it.
"""

from __future__ import annotations

import hashlib
import os
import secrets
import shutil
import unicodedata
from collections import Counter
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

import structlog

from argus_curator.models import ExportMode, ImageResult

logger = structlog.get_logger()

# A progress sink: called with {"phase": str, "done": int, "total": int}.
ProgressFn = Callable[[dict[str, Any]], None]

MANIFEST_NAME = "manifest.jsonl"
REPORT_NAME = "curation_report.csv"
RUN_STATS_NAME = "run_stats.json"


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


def plan_dest_paths(selected: list[ImageResult], preserve_structure: bool, by_stem: bool = False) -> dict[str, str]:
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


def transfer_one(src: Path, dst: Path, mode: ExportMode) -> None:
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


def write_text_atomically(path: Path, text: str, *, newline: str | None = None) -> None:
    """Write *text* to *path* through a fresh temp file and a rename; a failure leaves *path* as it was.

    The temp file gets a random name opened ``O_EXCL | O_NOFOLLOW``, so a link
    planted at it is never written through, and ``os.replace`` swaps a link at
    *path* for the file rather than following it. The data is fsynced before
    the rename and the directory after it, so a crash leaves the old file or
    the new one, never an empty or partial *path*.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o666)
    try:
        # newline="" keeps a CSV's \r\n from being translated again on Windows.
        with os.fdopen(fd, "w", encoding="utf-8", newline=newline) as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    # The rename is a directory entry: sync the directory too. Best effort, since
    # Windows cannot open one and some filesystems refuse; the data is already down.
    try:
        dir_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    except OSError:
        pass
    finally:
        os.close(dir_fd)


# A spreadsheet reads a cell starting with one of these as a formula (or DDE).
_FORMULA_LEADS = ("=", "+", "-", "@", "\t", "\r")


def text_cell(value: str) -> str:
    """*value* (a path or label from outside) made safe to open in a spreadsheet.

    A formula lead (``= + - @``, tab, CR) gets a quote, including one that sits
    behind whitespace a spreadsheet strips before it decides the cell is a formula.
    """
    lead = value.lstrip(" \t\r\n")
    if value[:1] in _FORMULA_LEADS or lead[:1] in _FORMULA_LEADS:
        return f"'{value}"
    return value
