"""FastAPI micro-server for argus-curator (peer to argus-lens on :8101).

Routes (section 4 of the brief):

    GET  /health
    GET  /detectors        -> { torch, cuda, clip, insightface, onnxruntime }
    POST /scan/folder      -> ScanSummary
    GET  /scan/{scan_id}   -> ScanSummary   (paginated via ?offset=&limit=)
    GET  /thumb?path=<rel> -> image/webp    (served from the mount)
    POST /export           -> ExportResult
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import queue
import threading
from importlib.util import find_spec
from pathlib import Path
from typing import Any

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import Response, StreamingResponse
except ImportError as exc:  # pragma: no cover
    raise ImportError("Server requires: pip install argus-curator[server]") from exc

from PIL import Image

from argus_curator import __version__, export, scanner
from argus_curator.faces import faces_available
from argus_curator.models import SUPPORTED_EXTS, ExportRequest, ExportResult, ScanRequest, ScanSummary
from argus_curator.store import ScanStore

THUMB_MAX = 384  # longest-edge px for /thumb webp output
_COUNT_CAP = 5000  # per-folder recursive image-count ceiling (keeps browsing snappy)


def _detectors() -> dict[str, bool]:
    torch_ok = find_spec("torch") is not None
    cuda_ok = False
    if torch_ok:
        try:
            import torch

            cuda_ok = bool(torch.cuda.is_available())
        except Exception:
            cuda_ok = False
    return {
        "torch": torch_ok,
        "cuda": cuda_ok,
        "clip": find_spec("clip") is not None or find_spec("open_clip") is not None,
        "insightface": faces_available(),
        "onnxruntime": find_spec("onnxruntime") is not None or find_spec("onnxruntime_gpu") is not None,
    }


def _resolve_within(root: Path, rel: str) -> Path:
    """Resolve ``rel`` under ``root`` and refuse path traversal escapes."""
    candidate = (root / rel).resolve()
    root_resolved = root.resolve()
    if root_resolved not in candidate.parents and candidate != root_resolved:
        raise HTTPException(status_code=400, detail="path escapes the mount root")
    return candidate


def _count_images(directory: Path, cap: int = _COUNT_CAP) -> int:
    """Recursive count of supported images under *directory* (capped)."""
    n = 0
    try:
        for p in directory.rglob("*"):
            if p.suffix.lower() in SUPPORTED_EXTS and p.is_file():
                n += 1
                if n >= cap:
                    break
    except OSError:
        pass
    return n


def _browse_folders(root: Path, rel: str) -> dict[str, Any]:
    """List sub-directories (with recursive image counts) under root/rel."""
    base = _resolve_within(root, rel)
    if not base.is_dir():
        raise HTTPException(status_code=404, detail=f"not a directory: {rel or '.'}")

    folders: list[dict[str, Any]] = []
    direct_images = 0
    try:
        for entry in sorted(base.iterdir(), key=lambda p: p.name.lower()):
            if entry.is_dir() and not entry.name.startswith("."):
                sub_rel = str(Path(rel) / entry.name) if rel else entry.name
                subfolders = sum(1 for c in entry.iterdir() if c.is_dir() and not c.name.startswith("."))
                folders.append(
                    {
                        "name": entry.name,
                        "rel_path": sub_rel,
                        "abs_path": str(entry.resolve()),
                        "image_count": _count_images(entry),
                        "subfolder_count": subfolders,
                    }
                )
            elif entry.is_file() and entry.suffix.lower() in SUPPORTED_EXTS:
                direct_images += 1
    except OSError as exc:
        raise HTTPException(status_code=400, detail=f"cannot read directory: {exc}") from exc

    parent = None
    if rel:
        parent_path = str(Path(rel).parent)
        parent = "" if parent_path == "." else parent_path

    return {
        "root": str(root.resolve()),
        "path": rel,
        "abs_path": str(base.resolve()),
        "parent": parent,
        "direct_image_count": direct_images,
        "folders": folders,
    }


def create_app(
    cors: bool = False,
    cors_origins: list[str] | None = None,
    cache_dir: str | None = None,
    source_root: str | None = None,
) -> FastAPI:
    """Create the curator FastAPI application."""
    app = FastAPI(
        title="Argus Curator",
        description="LoRA-native dataset curation: score, dedup, face-cluster, export.",
        version=__version__,
    )

    if cors:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins or ["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    store = ScanStore(cache_dir)
    # Mount root for /thumb when a scan_id is not supplied (NEXT_PUBLIC_CURATOR_SOURCE_PATH).
    default_source = source_root or os.environ.get("CURATOR_SOURCE_PATH")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "service": "argus-curator",
            "version": __version__,
            "source_root": str(Path(default_source).resolve()) if default_source else None,
        }

    @app.get("/detectors")
    async def detectors() -> dict[str, bool]:
        return _detectors()

    @app.get("/folders")
    async def folders(
        path: str = Query("", description="folder path relative to the mount root"),
    ) -> dict[str, Any]:
        """Browse Docker-mounted folders under the configured source root."""
        if not default_source:
            raise HTTPException(status_code=400, detail="no source root configured (set CURATOR_SOURCE_PATH)")
        root = Path(default_source)
        if not root.is_dir():
            raise HTTPException(status_code=400, detail=f"source root is not a directory: {default_source}")
        return await asyncio.to_thread(_browse_folders, root, path)

    @app.post("/scan/folder", response_model=ScanSummary)
    async def scan_folder(req: ScanRequest) -> ScanSummary:
        folder = Path(req.folder)
        if not folder.is_dir():
            raise HTTPException(status_code=400, detail=f"Not a directory: {req.folder}")
        try:
            summary = await asyncio.to_thread(
                scanner.scan_folder,
                req.folder,
                req.target_profile,
                req.config,
                req.faces,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"scan failed: {exc}") from exc
        store.save(summary)
        return summary

    @app.post("/scan/folder/stream")
    async def scan_folder_stream(req: ScanRequest) -> StreamingResponse:
        """Same as POST /scan/folder, but streams live progress over SSE.

        Emits ``event: progress`` frames ({phase, done, total}) as the scan runs,
        then a single ``event: complete`` frame carrying the full ScanSummary (the
        identical payload the non-streaming endpoint returns), or ``event: error``.
        """
        folder = Path(req.folder)
        if not folder.is_dir():
            raise HTTPException(status_code=400, detail=f"Not a directory: {req.folder}")

        # The scan is blocking CPU work, so it runs in a worker thread and hands
        # events back through a thread-safe queue that the async generator drains.
        events: queue.Queue[tuple[str, Any]] = queue.Queue()
        _DONE = "__done__"

        def on_progress(payload: dict[str, Any]) -> None:
            events.put(("progress", payload))

        def run_scan() -> None:
            try:
                summary = scanner.scan_folder(
                    req.folder,
                    req.target_profile,
                    req.config,
                    req.faces,
                    progress=on_progress,
                )
                store.save(summary)
                events.put(("complete", summary.model_dump(mode="json")))
            except Exception as exc:  # surface as a stream error, not a 500
                events.put(("error", {"detail": f"scan failed: {exc}"}))
            finally:
                events.put((_DONE, None))

        worker = threading.Thread(target=run_scan, name="curator-scan", daemon=True)
        worker.start()

        async def event_stream() -> Any:
            while True:
                kind, payload = await asyncio.to_thread(events.get)
                if kind == _DONE:
                    break
                yield f"event: {kind}\ndata: {json.dumps(payload)}\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/scan/{scan_id}", response_model=ScanSummary)
    async def get_scan(
        scan_id: str,
        offset: int = Query(0, ge=0),
        limit: int | None = Query(None, ge=1),
    ) -> ScanSummary:
        summary = store.load_page(scan_id, offset=offset, limit=limit)
        if summary is None:
            raise HTTPException(status_code=404, detail=f"unknown scan_id: {scan_id}")
        return summary

    @app.get("/thumb")
    async def thumb(
        path: str = Query(..., description="image path relative to the scan/mount root"),
        scan_id: str | None = Query(None, description="resolve relative to this scan's folder"),
    ) -> Response:
        if scan_id:
            summary = store.load(scan_id)
            if summary is None:
                raise HTTPException(status_code=404, detail=f"unknown scan_id: {scan_id}")
            root = Path(summary.folder)
        elif default_source:
            root = Path(default_source)
        else:
            raise HTTPException(status_code=400, detail="no scan_id and no configured source root")

        target = _resolve_within(root, path)
        if not target.is_file():
            raise HTTPException(status_code=404, detail=f"not found: {path}")

        def _render() -> bytes:
            with Image.open(target) as img:
                img = img.convert("RGB")
                img.thumbnail((THUMB_MAX, THUMB_MAX))
                buf = io.BytesIO()
                img.save(buf, format="WEBP", quality=80)
                return buf.getvalue()

        try:
            data = await asyncio.to_thread(_render)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"cannot render thumb: {exc}") from exc
        return Response(content=data, media_type="image/webp")

    @app.post("/export", response_model=ExportResult)
    async def run_export(req: ExportRequest) -> ExportResult:
        if not req.scan_id and req.selection is None:
            raise HTTPException(status_code=400, detail="export requires scan_id or inline selection")
        summary = store.load(req.scan_id) if req.scan_id else None
        if req.scan_id and summary is None:
            raise HTTPException(status_code=404, detail=f"unknown scan_id: {req.scan_id}")
        if summary is None:
            raise HTTPException(
                status_code=400,
                detail="inline selection export requires a scan_id for the manifest target_profile",
            )
        try:
            return await asyncio.to_thread(export.export_selection, summary, req)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"export failed: {exc}") from exc

    return app
