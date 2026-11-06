"""On-disk scan cache keyed by ``scan_id``.

Open decision #3 (resolved): scans are cached rather than recomputed, which is
what makes paginated ``GET /scan/{id}`` and export-by-id possible. The store is
a flat directory of ``<scan_id>.json`` files — no database, easy to inspect,
easy to wipe.
"""

from __future__ import annotations

import contextlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

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

    def _path(self, scan_id: str) -> Path:
        # Guard against path traversal via a crafted scan_id.
        safe = scan_id.replace("/", "_").replace("\\", "_").replace("..", "_")
        return self.cache_dir / f"{safe}.json"

    def _marker(self, scan_id: str) -> Path:
        return self._path(scan_id).with_suffix(".running")

    def save(self, summary: ScanSummary) -> Path:
        path = self._path(summary.scan_id)
        # Replace atomically so a reader never sees a half-written summary.
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(summary.model_dump_json(context=STORE_CONTEXT), encoding="utf-8")
        tmp.replace(path)
        return path

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
        return ScanSummary.model_validate_json(path.read_text(encoding="utf-8"))

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
        all_results = summary.results
        sliced = all_results[offset:] if limit is None else all_results[offset : offset + limit]
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
        path.with_suffix(".json.tmp").unlink(missing_ok=True)
        existed = path.exists()
        path.unlink(missing_ok=True)
        # Last: if the snapshot would not go, the next start still knows to drop it.
        self._marker(scan_id).unlink(missing_ok=True)
        return existed

    def delete_all(self) -> int:
        """Remove every cached scan, half-written ones and markers included. Returns how many."""
        count = 0
        paths = [path for pattern in ("*.json", "*.json.tmp", "*.running") for path in self.cache_dir.glob(pattern)]
        for path in paths:
            with contextlib.suppress(FileNotFoundError):
                path.unlink()
                count += path.suffix == ".json"
        return count
