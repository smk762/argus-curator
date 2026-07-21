"""FastAPI micro-server for argus-curator (peer to argus-lens on :8101).

Routes (section 4 of the brief):

    GET  /health
    GET  /detectors        -> { torch, cuda, clip, insightface, onnxruntime }
    POST /scan/folder      -> ScanSummary
    GET  /scan/{scan_id}   -> ScanSummary   (paginated via ?offset=&limit=)
    GET  /thumb?path=<rel> -> image/webp    (served from the mount)
    POST /upload           -> save images into a folder under the source root
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
    from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
    from fastapi.responses import Response, StreamingResponse
except ImportError as exc:  # pragma: no cover
    raise ImportError("Server requires: pip install argus-curator[server]") from exc

from argus_cortex.server import WriteGuard, cross_site_refuse, env_flag
from PIL import Image

from argus_curator import __version__, export, scanner
from argus_curator.faces import faces_available
from argus_curator.models import SUPPORTED_EXTS, ExportRequest, ExportResult, ScanRequest, ScanSummary
from argus_curator.store import ScanStore

THUMB_MAX = 384  # longest-edge px for /thumb webp output
_COUNT_CAP = 5000  # per-folder recursive image-count ceiling (keeps browsing snappy)

# Default CORS origins for `--cors`: the argus-studio dev frontend. Anything
# else must be allow-listed explicitly (or use the credential-less wildcard).
_LOCALHOST_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]


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
    try:
        candidate = (root / rel).resolve()
        root_resolved = root.resolve()
    except (ValueError, OSError) as exc:
        # e.g. an embedded NUL byte — a malformed path is a client error, not a
        # 500 with a traceback from the security-critical resolve step.
        raise HTTPException(status_code=400, detail="invalid path") from exc
    if root_resolved not in candidate.parents and candidate != root_resolved:
        raise HTTPException(status_code=400, detail="path escapes the mount root")
    return candidate


def _is_within(root: Path, candidate: Path) -> bool:
    """True when *candidate* is *root* or lies under it (both resolved)."""
    try:
        candidate, root = candidate.resolve(), root.resolve()
    except (ValueError, OSError):
        return False
    return candidate == root or root in candidate.parents


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
    cors_allow_any: bool = False,
    cache_dir: str | None = None,
    source_root: str | None = None,
    export_root: str | None = None,
    allow_move: bool | None = None,
) -> FastAPI:
    """Create the curator FastAPI application.

    Request-supplied paths are untrusted: scan folders resolve under
    *source_root* and export destinations under *export_root* (both refusing
    traversal escapes), and the endpoints refuse outright when their root is
    not configured. ``mode="move"`` — the one destructive transfer mode — is
    rejected unless *allow_move* is enabled.
    """
    app = FastAPI(
        title="Argus Curator",
        description="LoRA-native dataset curation: score, dedup, face-cluster, export.",
        version=__version__,
    )

    if cors_origins is None and (env_origins := os.environ.get("CURATOR_CORS_ORIGINS")):
        cors_origins = [o.strip() for o in env_origins.split(",") if o.strip()]

    # A literal wildcard is only honoured by browsers without credentials; with
    # allow_credentials=True Starlette reflects any Origin back, which defeats
    # the allow-list entirely. An explicit "*" in the allow-list means the same
    # thing as --cors-any, so it takes the same safe path rather than silently
    # becoming origin reflection.
    wildcard = cors_allow_any or bool(cors_origins and "*" in cors_origins)
    # Origins the operator has actually named. The wildcard grants anonymous
    # READ access from anywhere, but never a cross-site write: a public demo
    # must not double as an upload target for any page its users visit. To
    # allow cross-site writes, name the origin with --cors-origin.
    trusted_origins: list[str] = [] if wildcard else list(cors_origins or (_LOCALHOST_ORIGINS if cors else []))

    # CORS is not a write boundary — see cross_site_refuse for why an unauthed
    # LAN/localhost server must gate unsafe methods on Origin itself.
    #
    # Registered before CORSMiddleware so CORS ends up the outer layer
    # (add_middleware inserts at 0) and can still annotate a rejected write.
    # That matters under the wildcard, where CORS would allow the origin a read
    # but the guard refuses the write: the page gets a readable 403 rather than
    # an opaque CORS error. For a non-allow-listed origin CORS correctly adds
    # nothing — it must not advertise itself to an origin it does not trust.
    app.add_middleware(WriteGuard, refuse=cross_site_refuse(trusted_origins))

    if cors or cors_origins or cors_allow_any:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"] if wildcard else (cors_origins or _LOCALHOST_ORIGINS),
            allow_credentials=not wildcard,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    store = ScanStore(cache_dir)
    # Containment roots. Scans/thumbs/uploads resolve under the source root;
    # export destinations under the export root. The ARGUS_* names are the
    # deployment-facing ones (argus-halo); CURATOR_* are kept for compose.
    default_source = source_root or os.environ.get("ARGUS_CURATOR_SCAN_ROOT") or os.environ.get("CURATOR_SOURCE_PATH")
    default_export = export_root or os.environ.get("ARGUS_CURATOR_EXPORT_ROOT") or os.environ.get("CURATOR_EXPORT_PATH")
    if allow_move is None:
        allow_move = env_flag("CURATOR_ALLOW_MOVE")

    def _root_or_400(configured: str | None, kind: str, env: str) -> Path:
        if not configured:
            raise HTTPException(status_code=400, detail=f"no {kind} root configured (set {env})")
        root = Path(configured)
        if not root.is_dir():
            raise HTTPException(status_code=400, detail=f"{kind} root is not a directory: {configured}")
        return root

    def _scan_root_or_400(summary: ScanSummary) -> Path:
        """The scan's own folder, once proven to lie under the source root.

        A summary is *data* — it comes from the on-disk store, which the CLI
        (unconstrained by design) and pre-containment server builds both write
        to. Trusting ``summary.folder`` as a containment root would let any
        cached scan of an arbitrary directory read or export files the source
        root is supposed to fence off, so it is re-checked against config here.
        """
        root = _root_or_400(default_source, "source", "CURATOR_SOURCE_PATH")
        folder = Path(summary.folder)
        if not _is_within(root, folder):
            raise HTTPException(
                status_code=400,
                detail=f"scan {summary.scan_id} lies outside the source root",
            )
        return folder

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "service": "argus-curator",
            "version": __version__,
            "source_root": str(Path(default_source).resolve()) if default_source else None,
            "export_root": str(Path(default_export).resolve()) if default_export else None,
            "allow_move": allow_move,
        }

    @app.get("/detectors")
    async def detectors() -> dict[str, bool]:
        return _detectors()

    @app.get("/folders")
    async def folders(
        path: str = Query("", description="folder path relative to the mount root"),
    ) -> dict[str, Any]:
        """Browse Docker-mounted folders under the configured source root."""
        root = _root_or_400(default_source, "source", "CURATOR_SOURCE_PATH")
        return await asyncio.to_thread(_browse_folders, root, path)

    def _resolve_scan_folder(requested: str) -> Path:
        """Resolve a request-supplied scan folder under the source root.

        ``requested`` is canonically relative to the root; an absolute path is
        tolerated only when it already lies inside the root (compat with UIs
        that echo back ``abs_path`` from ``/folders``).
        """
        root = _root_or_400(default_source, "source", "CURATOR_SOURCE_PATH")
        folder = _resolve_within(root, requested)
        if not folder.is_dir():
            raise HTTPException(status_code=400, detail=f"not a directory under the scan root: {requested}")
        return folder

    @app.post("/scan/folder", response_model=ScanSummary)
    async def scan_folder(req: ScanRequest) -> ScanSummary:
        folder = _resolve_scan_folder(req.folder)
        try:
            summary = await asyncio.to_thread(
                scanner.scan_folder,
                folder,
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
        folder = _resolve_scan_folder(req.folder)

        # The scan is blocking CPU work, so it runs in a worker thread and hands
        # events back through a thread-safe queue that the async generator drains.
        events: queue.Queue[tuple[str, Any]] = queue.Queue()
        _DONE = "__done__"

        def on_progress(payload: dict[str, Any]) -> None:
            events.put(("progress", payload))

        def run_scan() -> None:
            try:
                summary = scanner.scan_folder(
                    folder,
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
            root = _scan_root_or_400(summary)
        else:
            root = _root_or_400(default_source, "source", "CURATOR_SOURCE_PATH")

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

    @app.post("/upload")
    async def upload(
        files: list[UploadFile] = File(..., description="image files to save"),
        folder: str = Form(..., description="target folder relative to the source root"),
    ) -> dict[str, Any]:
        """Save uploaded images into a folder under the configured source root.

        Non-image files and names that already exist in the target folder are
        reported in ``skipped`` (existing files are never overwritten).
        """
        root = _root_or_400(default_source, "source", "CURATOR_SOURCE_PATH")
        target = _resolve_within(root, folder)
        rel = target.relative_to(root.resolve()).as_posix()
        rel = "" if rel == "." else rel
        await asyncio.to_thread(target.mkdir, parents=True, exist_ok=True)

        saved = 0
        skipped: list[str] = []
        errors: list[dict[str, str]] = []
        for item in files:
            name = Path(item.filename or "").name  # basename only — no client-supplied paths
            if not name:
                errors.append({"name": item.filename or "", "detail": "missing filename"})
                continue
            if Path(name).suffix.lower() not in SUPPORTED_EXTS:
                skipped.append(name)
                continue
            dest = target / name
            if dest.exists():
                skipped.append(name)
                continue
            try:
                data = await item.read()
                await asyncio.to_thread(dest.write_bytes, data)
                saved += 1
            except OSError as exc:
                errors.append({"name": name, "detail": f"cannot save: {exc}"})

        return {"folder": rel, "saved": saved, "skipped": skipped, "errors": errors}

    def _validate_export(req: ExportRequest) -> tuple[ScanSummary, ExportRequest]:
        """Shared gate for both export endpoints.

        Loads the scan, refuses one whose folder is outside the source root,
        rejects ``mode="move"`` unless the server allows it, and rewrites
        ``req.dest`` to the containment-checked absolute path under the export
        root — all before anything touches the filesystem.
        """
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
        # The scan's images are read straight from `abs_path`, so a scan rooted
        # outside the source root would export files the root should fence off.
        _scan_root_or_400(summary)
        if req.mode == "move" and not allow_move:
            raise HTTPException(
                status_code=403,
                detail='mode "move" is disabled on this server (start with --allow-move / CURATOR_ALLOW_MOVE=1)',
            )
        root = _root_or_400(default_export, "export", "CURATOR_EXPORT_PATH")
        dest = _resolve_within(root, req.dest)
        return summary, req.model_copy(update={"dest": str(dest)})

    @app.post("/export", response_model=ExportResult)
    async def run_export(req: ExportRequest) -> ExportResult:
        summary, req = _validate_export(req)
        try:
            return await asyncio.to_thread(export.export_selection, summary, req)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"export failed: {exc}") from exc

    @app.post("/export/stream")
    async def run_export_stream(req: ExportRequest) -> StreamingResponse:
        """Same as POST /export, but streams file-transfer progress over SSE.

        Emits ``event: progress`` frames ({phase: "transferring", done, total})
        as files are copied, then one ``event: complete`` frame carrying the
        ExportResult, or ``event: error``.
        """
        summary, req = _validate_export(req)

        events: queue.Queue[tuple[str, Any]] = queue.Queue()
        _DONE = "__done__"

        def on_progress(payload: dict[str, Any]) -> None:
            events.put(("progress", payload))

        def run() -> None:
            try:
                result = export.export_selection(summary, req, progress=on_progress)
                events.put(("complete", result.model_dump(mode="json")))
            except Exception as exc:
                events.put(("error", {"detail": f"export failed: {exc}"}))
            finally:
                events.put((_DONE, None))

        worker = threading.Thread(target=run, name="curator-export", daemon=True)
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

    return app
