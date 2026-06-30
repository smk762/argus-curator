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
import os
from importlib.util import find_spec
from pathlib import Path
from typing import Any

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import Response
except ImportError as exc:  # pragma: no cover
    raise ImportError("Server requires: pip install argus-curator[server]") from exc

from PIL import Image

from argus_curator import __version__, export, scanner
from argus_curator.faces import faces_available
from argus_curator.models import ExportRequest, ExportResult, ScanRequest, ScanSummary
from argus_curator.store import ScanStore

THUMB_MAX = 384  # longest-edge px for /thumb webp output


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
        return {"status": "ok", "service": "argus-curator", "version": __version__}

    @app.get("/detectors")
    async def detectors() -> dict[str, bool]:
        return _detectors()

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
