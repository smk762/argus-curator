"""On-disk scan cache keyed by ``scan_id``.

Open decision #3 (resolved): scans are cached rather than recomputed, which is
what makes paginated ``GET /scan/{id}`` and export-by-id possible. The store is
a flat directory of ``<scan_id>.json`` files — no database, easy to inspect,
easy to wipe.
"""

from __future__ import annotations

import contextlib
import glob
import json
import os
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

from argus_curator.models import STORE_CONTEXT, ScanSummary

logger = structlog.get_logger()


def default_cache_dir() -> Path:
    env = os.environ.get("CURATOR_CACHE_DIR")
    if env:
        return Path(env)
    return Path(os.path.expanduser("~/.cache/argus_curator/scans"))


class ScanStore:
    """Persist and retrieve :class:`ScanSummary` objects by ``scan_id``."""

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else default_cache_dir()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # scan_id -> {rel_path: semantic embedding, or a speech scan's speaker
        # embedding (#187)}, in memory only (#188): never
        # written to the cache dir, gone on restart, and dropped with the scan
        # or its rows by delete, delete_all and save.
        self._embeddings: dict[str, dict[str, Any]] = {}
        self._embeddings_lock = threading.Lock()

    def _hold_embeddings(self, summary: ScanSummary) -> None:
        """Keep *summary*'s embeddings, or prune the held ones to its rows.

        A summary carrying embeddings (a finished scan) replaces what is held.
        One without (a snapshot, or a reloaded copy saved after rows were
        dropped) prunes the held set to its rows, so a deleted or moved image's
        vector goes with its row.
        """
        # A speech scan's rows are its clips, and its embeddings speaker ones (#187).
        rows = {r.rel_path for r in (summary.clips if summary.is_speech else summary.results)}
        with self._embeddings_lock:
            source = summary.embeddings or self._embeddings.get(summary.scan_id)
            if not source:
                return
            kept = {rel: v for rel, v in source.items() if rel in rows}
            if kept:
                self._embeddings[summary.scan_id] = kept
            else:
                self._embeddings.pop(summary.scan_id, None)

    def held_embeddings(self, scan_id: str) -> dict[str, Any]:
        """A copy of the embeddings held for *scan_id* (empty when none are)."""
        with self._embeddings_lock:
            return dict(self._embeddings.get(scan_id, {}))

    def _path(self, scan_id: str) -> Path:
        # Guard against path traversal via a crafted scan_id.
        safe = scan_id.replace("/", "_").replace("\\", "_").replace("..", "_")
        return self.cache_dir / f"{safe}.json"

    def _marker(self, scan_id: str) -> Path:
        return self._path(scan_id).with_suffix(".running")

    def save(self, summary: ScanSummary) -> Path:
        path = self._path(summary.scan_id)
        # Replace atomically so a reader never sees a half-written summary. Each
        # writer has its own temp file: two saves of one scan (a running scan's
        # snapshot and a delete, say) must not write into, or move, each other's.
        tmp = self.cache_dir / f"{path.stem}.{uuid.uuid4().hex}.json.tmp"
        try:
            with tmp.open("x", encoding="utf-8") as fh:
                fh.write(summary.model_dump_json(context=STORE_CONTEXT))
            tmp.replace(path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        self._hold_embeddings(summary)
        return path

    def _temps(self, scan_id: str) -> list[Path]:
        """This scan's temp files: ``<id>.<random>.json.tmp``, and the older ``<id>.json.tmp``."""
        stem = self._path(scan_id).stem
        found = [self.cache_dir / f"{stem}.json.tmp"]
        for tmp in self.cache_dir.glob(f"{glob.escape(stem)}.*.json.tmp"):
            middle = tmp.name[len(stem) + 1 : -len(".json.tmp")]
            if "." not in middle:  # not another scan's, e.g. "<id>.v2"
                found.append(tmp)
        return found

    def save_snapshot(self, summary: ScanSummary) -> Path:
        """Save a scan still running, so a server killed mid-scan can be told from one that finished.

        The marker is a file of its own, not a field: rows are re-saved from
        loaded copies, and a stale copy must not make a finished scan look unfinished.
        """
        # Marker first: a kill between the two then leaves nothing unmarked.
        self._marker(summary.scan_id).touch()
        return self.save(summary)

    def mark_finished(self, scan_id: str) -> None:
        """The saved summary is the result: it survives :meth:`drop_unfinished` from now on."""
        self._marker(scan_id).unlink(missing_ok=True)

    def drop_unfinished(self) -> int:
        """Remove scans a dead server left part-way. Returns how many had a snapshot.

        For startup only, when nothing here is running. One that cannot be
        removed is logged and left marked, so the next start tries again.
        """
        count = 0
        for marker in self.cache_dir.glob("*.running"):
            try:
                count += self.delete(marker.stem)
            except OSError as exc:
                logger.warning("unfinished_scan_drop_failed", scan_id=marker.stem, error=str(exc))
        return count

    def load(self, scan_id: str) -> ScanSummary | None:
        path = self._path(scan_id)
        if not path.exists():
            return None
        summary = ScanSummary.model_validate_json(path.read_text(encoding="utf-8"))
        summary.embeddings.update(self.held_embeddings(scan_id))
        return summary

    def load_page(
        self,
        scan_id: str,
        offset: int = 0,
        limit: int | None = None,
    ) -> ScanSummary | None:
        """Load a scan but return only a slice of ``results`` (for the grid)."""
        summary = self.load(scan_id)
        if summary is None:
            return None
        # A speech scan's rows are its clips (#186); an image scan's, its results.
        rows = summary.clips if summary.is_speech else summary.results
        sliced = rows[offset:] if limit is None else rows[offset : offset + limit]
        if summary.is_speech:
            summary.clips = sliced
        else:
            summary.results = sliced
        summary.offset = offset
        summary.limit = limit
        summary.returned = len(sliced)
        return summary

    def list_ids(self) -> list[str]:
        return sorted(p.stem for p in self.cache_dir.glob("*.json"))

    def list_meta(self) -> list[dict[str, object]]:
        """Scan id, folder, and image count, newest first. Skips unreadable files."""
        rows: list[dict[str, object]] = []
        for path in self.cache_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            updated = datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat()
            rows.append(
                {
                    "scan_id": data.get("scan_id") or path.stem,
                    "folder": data.get("folder") or "",
                    "total": int(data.get("total") or 0),
                    "updated_at": updated,
                }
            )
        rows.sort(key=lambda row: str(row["updated_at"]), reverse=True)
        return rows

    def delete(self, scan_id: str) -> bool:
        path = self._path(scan_id)
        # A save that failed part-way leaves this behind, and nothing else would remove it.
        for tmp in self._temps(scan_id):
            tmp.unlink(missing_ok=True)
        existed = path.exists()
        path.unlink(missing_ok=True)
        with self._embeddings_lock:
            self._embeddings.pop(scan_id, None)
        # Last: if the snapshot would not go, the next start still knows to drop it.
        self._marker(scan_id).unlink(missing_ok=True)
        return existed

    def delete_all(self) -> int:
        """Remove every cached scan, half-written ones and markers included. Returns how many."""
        with self._embeddings_lock:
            self._embeddings.clear()
        count = 0
        paths = [path for pattern in ("*.json", "*.json.tmp", "*.running") for path in self.cache_dir.glob(pattern)]
        for path in paths:
            with contextlib.suppress(FileNotFoundError):
                path.unlink()
                count += path.suffix == ".json"
        return count
