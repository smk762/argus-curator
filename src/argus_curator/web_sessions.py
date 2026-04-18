"""Ephemeral browser-upload sessions — temp dirs only, no external object storage."""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
import uuid
from pathlib import Path

_SESSION_LOCK = threading.Lock()
_SESSION_ROOTS: dict[str, Path] = {}

SUPPORTED_IMAGE_UPLOAD_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def sanitize_relative_path(raw: str) -> str | None:
    """Return a normalised POSIX relative path, or None if unsafe."""
    if not raw or raw.startswith("/") or (os.name == "nt" and len(raw) >= 2 and raw[1] == ":"):
        return None
    p = Path(raw.replace("\\", "/"))
    parts: list[str] = []
    for part in p.parts:
        if part in (".", ""):
            continue
        if part == "..":
            if not parts:
                return None
            parts.pop()
        else:
            parts.append(part)
    if not parts:
        return None
    return str(Path(*parts).as_posix())


def create_session() -> str:
    sid = str(uuid.uuid4())
    root = Path(tempfile.mkdtemp(prefix="argus-curator-web-"))
    (root / "in").mkdir(parents=True, exist_ok=True)
    with _SESSION_LOCK:
        _SESSION_ROOTS[sid] = root
    return sid


def session_in_dir(session_id: str) -> Path | None:
    with _SESSION_LOCK:
        root = _SESSION_ROOTS.get(session_id)
    if root is None:
        return None
    return root / "in"


def session_root(session_id: str) -> Path | None:
    with _SESSION_LOCK:
        return _SESSION_ROOTS.get(session_id)


def destroy_session(session_id: str) -> bool:
    with _SESSION_LOCK:
        root = _SESSION_ROOTS.pop(session_id, None)
    if root is None:
        return False
    shutil.rmtree(root, ignore_errors=True)
    return True


def clear_in_dir(in_dir: Path) -> None:
    if in_dir.exists():
        shutil.rmtree(in_dir)
    in_dir.mkdir(parents=True, exist_ok=True)


def resolve_member_under(root: Path, rel: str) -> Path | None:
    """Resolve *rel* under *root*; return None if path escapes *root*."""
    root = root.resolve()
    cand = (root / rel).resolve()
    try:
        cand.relative_to(root)
    except ValueError:
        return None
    return cand


def collect_zip_members(
    in_dir: Path,
    selected_names: list[str],
) -> list[tuple[Path, str]]:
    """Return (absolute_path, archive_name) pairs for existing files."""
    pairs: list[tuple[Path, str]] = []
    for name in selected_names:
        rel = sanitize_relative_path(name)
        if rel is None:
            continue
        path = resolve_member_under(in_dir, rel)
        if path is None or not path.is_file():
            continue
        pairs.append((path, rel))
    return pairs
