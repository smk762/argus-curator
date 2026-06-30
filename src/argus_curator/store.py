"""On-disk scan cache keyed by ``scan_id``.

Open decision #3 (resolved): scans are cached rather than recomputed, which is
what makes paginated ``GET /scan/{id}`` and export-by-id possible. The store is
a flat directory of ``<scan_id>.json`` files — no database, easy to inspect,
easy to wipe.
"""

from __future__ import annotations

import os
from pathlib import Path

from argus_curator.models import ScanSummary


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

    def save(self, summary: ScanSummary) -> Path:
        path = self._path(summary.scan_id)
        path.write_text(summary.model_dump_json(), encoding="utf-8")
        return path

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

    def delete(self, scan_id: str) -> bool:
        path = self._path(scan_id)
        if path.exists():
            path.unlink()
            return True
        return False
