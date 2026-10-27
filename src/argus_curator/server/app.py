"""FastAPI micro-server for argus-curator (peer to argus-lens on :8101).

Routes (section 4 of the brief):

    GET  /health
    GET  /detectors        -> { torch, cuda, clip, insightface, onnxruntime, fairface, mivolo, places365, garments, nudenet, singguard }
    POST /scan/folder      -> ScanSummary
    GET  /scan/{scan_id}   -> ScanSummary   (paginated via ?offset=&limit=)
    GET  /thumb?path=<rel> -> image/webp    (served from the mount)
    GET  /image?path=<rel> -> original bytes (same containment as /thumb)
    POST /scan/{id}/select -> SelectResult  (decide_selection, writes nothing)
    POST /scan/{id}/delete -> remove one under-18 exposed file from the source folder
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
import uuid
from importlib.util import find_spec
from pathlib import Path
from typing import Any

try:
    # Inside the guard with fastapi: argus_cortex.server ships in argus-cortex 0.2.0+
    # behind its own [server] extra, so a stale cortex fails here too and must get the
    # same actionable message rather than a bare ModuleNotFoundError naming a package
    # the user never installed directly.
    from argus_cortex.server import WriteGuard, cross_site_refuse, env_flag
    from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
    from fastapi.responses import FileResponse, Response, StreamingResponse
except ImportError as exc:  # pragma: no cover
    raise ImportError("Server requires: pip install argus-curator[server]") from exc

from PIL import Image
from pydantic import BaseModel, Field

from argus_curator import __version__, export, scanner
from argus_curator import guard as guard_mod
from argus_curator import nsfw as nsfw_mod
from argus_curator.attributes import fairface_available, mivolo_available
from argus_curator.faces import faces_available
from argus_curator.garments import garments_available
from argus_curator.models import (
    MANIFEST_VERSION,
    SUPPORTED_EXTS,
    ExportRequest,
    ExportResult,
    ScanRequest,
    ScanSummary,
    SelectRequest,
    SelectResult,
    SelectRow,
)
from argus_curator.runtime import ModelUnavailable
from argus_curator.scenes import scenes_available
from argus_curator.selection import (
    decide_selection,
    garment_floor_problem,
    guard_filter_problem,
    minor_exposed,
    nsfw_filter_problem,
)
from argus_curator.store import ScanStore

THUMB_MAX = 384  # longest-edge px for /thumb webp output
_IMAGE_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}
_COUNT_CAP = 5000  # per-folder recursive image-count ceiling (keeps browsing snappy)


class DeleteRequest(BaseModel):
    """One image, or a batch. Every path has to be an under-18 exposed detection."""

    rel_path: str | None = None
    rel_paths: list[str] = Field(default_factory=list, max_length=200)

    def targets(self) -> list[str]:
        ordered: list[str] = []
        seen: set[str] = set()
        for path in ([self.rel_path] if self.rel_path else []) + self.rel_paths:
            if path and path not in seen:
                seen.add(path)
                ordered.append(path)
        return ordered


class ScanGate:
    """The one scan this process is allowed to run at a time."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active: dict[str, Any] | None = None

    def start(self, scan_id: str, folder: str, pending: list[str]) -> dict[str, Any] | None:
        """Register *scan_id*. Return the scan already running, if there is one."""
        with self._lock:
            if self.active is not None:
                return dict(self.active)
            self.active = {
                "scan_id": scan_id,
                "folder": folder,
                "phase": "starting",
                "done": 0,
                "total": 0,
                "pending_checks": list(pending),
            }
            return None

    def update(self, scan_id: str, **fields: Any) -> None:
        with self._lock:
            if self.active and self.active["scan_id"] == scan_id:
                self.active.update(fields)

    def finish(self, scan_id: str) -> None:
        with self._lock:
            if self.active and self.active["scan_id"] == scan_id:
                self.active = None

    def snapshot(self) -> dict[str, Any] | None:
        with self._lock:
            return dict(self.active) if self.active else None


def _busy_detail(active: dict[str, Any]) -> str:
    return f"A scan is already running ({active['phase']} {active['done']}/{active['total']})."


# Default CORS origins for `--cors`: the argus-studio dev frontend. Anything
# else must be allow-listed explicitly (or use the credential-less wildcard).
_LOCALHOST_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]


def _detectors() -> dict[str, bool | list[str]]:
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
        "fairface": fairface_available(),
        "mivolo": mivolo_available(),
        "places365": scenes_available(),
        "garments": garments_available(),
        # A NudeNet server URL is configured; reachability is not probed here.
        "nudenet": nsfw_mod.configured(),
        "singguard": guard_mod.configured(),
        # Policy names a scan can pass as guard.policy; empty without a server.
        "singguard_policies": list(guard_mod.POLICIES) if guard_mod.configured() else [],
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
    #
    # The "*" is dropped rather than allowed to collapse the whole list: a
    # wildcard co-listed with a real origin (`--cors-origin '*' --cors-origin
    # https://studio.example`, or CURATOR_CORS_ORIGINS=*,https://studio.example)
    # must still grant that named origin its writes, or naming it — the one
    # documented way to get them — silently does nothing.
    named = [o for o in (cors_origins or []) if o != "*"]
    trusted_origins: list[str] = named or ([] if wildcard else (_LOCALHOST_ORIGINS if cors else []))

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
    gate = ScanGate()
    app.state.scan_gate = gate
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
            # The *manifest* contract version, which moves independently of the
            # package version above. Clients branch on this to refuse an unknown
            # major, instead of sniffing which fields an ExportResult happens to
            # carry — a test that cannot tell an old server from a new one that
            # transferred nothing.
            "manifest_version": MANIFEST_VERSION,
            "source_root": str(Path(default_source).resolve()) if default_source else None,
            "export_root": str(Path(default_export).resolve()) if default_export else None,
            "allow_move": allow_move,
        }

    @app.get("/detectors")
    async def detectors() -> dict[str, bool | list[str]]:
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

    def _require_models(req: ScanRequest) -> None:
        """Refuse up front when an optional model is requested but not configured.

        Without this the scan would score every image first and only then fail.
        The detail names the env var, not the server's filesystem layout.
        """
        wanted = (
            (req.faces.fairface, fairface_available, "FairFace", "ARGUS_CURATOR_FAIRFACE_MODEL", "fairface-convert"),
            (req.faces.mivolo, mivolo_available, "MiVOLO", "ARGUS_CURATOR_MIVOLO_MODEL", "mivolo-convert"),
            (req.scenes.enabled, scenes_available, "Places365", "ARGUS_CURATOR_PLACES365_MODEL", "places365-convert"),
            (
                req.garments.enabled,
                garments_available,
                "Fashionpedia",
                "ARGUS_CURATOR_GARMENTS_MODEL",
                "garments-convert",
            ),
        )
        for requested, available, name, env, command in wanted:
            if requested and not available():
                raise HTTPException(
                    status_code=400,
                    detail=f"{name} requested but this server has no {name} model "
                    f"(set {env}; see `argus-curator {command}`)",
                )
        if nsfw_mod.resolve_enabled(req.nsfw):
            try:
                url = nsfw_mod.endpoint()
            except ModelUnavailable:
                url = None
            if url is None:
                # Explicitly requested without a URL, or a malformed URL: say
                # which, without echoing the configured value (it may hold a
                # credential).
                state = "is not a valid http(s) URL" if nsfw_mod.configured() else "is not set"
                raise HTTPException(
                    status_code=400,
                    detail=f"NudeNet check requested but {nsfw_mod.URL_ENV} {state}",
                )
        if guard_mod.resolve_enabled(req.guard):
            try:
                url = guard_mod.endpoint()
            except ModelUnavailable:
                url = None
            if url is None:
                state = "is not a valid http(s) URL" if guard_mod.configured() else "is not set"
                raise HTTPException(
                    status_code=400,
                    detail=f"SingGuard check requested but {guard_mod.URL_ENV} {state}",
                )
            try:
                guard_mod.thinking()
                for name in (req.guard.policy, *req.guard.variants):
                    guard_mod.policy(name)
            except ModelUnavailable as exc:  # these name no URL or path
                raise HTTPException(status_code=400, detail=f"SingGuard check requested but {exc}") from exc

    def _requested_checks(req: ScanRequest) -> list[str]:
        return scanner.requested_checks(req.faces, req.nsfw, req.guard)

    def _claim_scan(folder: Path, req: ScanRequest) -> str:
        scan_id = uuid.uuid4().hex
        busy = gate.start(scan_id, str(folder.resolve()), _requested_checks(req))
        if busy is not None:
            raise HTTPException(status_code=409, detail=_busy_detail(busy))
        return scan_id

    def _note_progress(scan_id: str, requested: list[str], payload: dict[str, Any]) -> None:
        done = int(payload.get("done") or 0)
        total = int(payload.get("total") or 0)
        phase = str(payload.get("phase") or "")
        # The scanner says which checks are still running. Events from before
        # scoring carry no list; every requested check is ahead then.
        pending = payload.get("pending_checks")
        gate.update(
            scan_id,
            phase=phase,
            done=done,
            total=total,
            pending_checks=list(pending) if isinstance(pending, list) else list(requested),
        )

    @app.get("/scans")
    async def list_scans() -> dict[str, Any]:
        """Scans on disk, plus the one currently running, if any."""
        return {"active": gate.snapshot(), "scans": store.list_meta()}

    @app.post("/scan/folder", response_model=ScanSummary)
    async def scan_folder(req: ScanRequest) -> ScanSummary:
        _require_models(req)
        folder = _resolve_scan_folder(req.folder)
        scan_id = _claim_scan(folder, req)
        requested = _requested_checks(req)
        try:
            summary = await asyncio.to_thread(
                scanner.scan_folder,
                folder,
                req.target_profile,
                req.config,
                req.faces,
                scene_cfg=req.scenes,
                garment_cfg=req.garments,
                nsfw_cfg=req.nsfw,
                guard_cfg=req.guard,
                scan_id=scan_id,
                progress=lambda payload: _note_progress(scan_id, requested, payload),
            )
            store.save(summary)
            return summary
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"scan failed: {exc}") from exc
        finally:
            gate.finish(scan_id)

    @app.post("/scan/folder/stream")
    async def scan_folder_stream(req: ScanRequest) -> StreamingResponse:
        """Same as POST /scan/folder, but streams progress and image batches over SSE.

        Emits ``event: progress`` frames ({phase, done, total}) and ``event: chunk``
        frames (those counts plus the images finished since the previous chunk),
        then a single ``event: complete`` frame carrying the full ScanSummary, or
        ``event: error``. A second scan is refused while one is still running.
        """
        _require_models(req)
        folder = _resolve_scan_folder(req.folder)
        scan_id = _claim_scan(folder, req)
        requested = _requested_checks(req)

        # The scan is blocking CPU work, so it runs in a worker thread and hands
        # events back through a thread-safe queue that the async generator drains.
        # The slot stays taken until that thread finishes, even if the browser
        # disconnects, so a refresh cannot start a second scan behind it.
        events: queue.Queue[tuple[str, Any]] = queue.Queue()
        _DONE = "__done__"

        def on_progress(payload: dict[str, Any]) -> None:
            _note_progress(scan_id, requested, payload)
            events.put(("progress", payload))

        def on_chunk(payload: dict[str, Any]) -> None:
            events.put(("chunk", payload))

        def run_scan() -> None:
            try:
                summary = scanner.scan_folder(
                    folder,
                    req.target_profile,
                    req.config,
                    req.faces,
                    scene_cfg=req.scenes,
                    garment_cfg=req.garments,
                    nsfw_cfg=req.nsfw,
                    guard_cfg=req.guard,
                    scan_id=scan_id,
                    progress=on_progress,
                    chunk=on_chunk,
                    save=store.save,
                )
                store.save(summary)
                events.put(("complete", summary.model_dump(mode="json")))
            except Exception as exc:  # surface as a stream error, not a 500
                events.put(("error", {"detail": f"scan failed: {exc}"}))
            finally:
                gate.finish(scan_id)
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

    def _media_root(scan_id: str | None) -> Path:
        """Source root for /thumb and /image, re-checked when a scan id is given."""
        if scan_id:
            summary = store.load(scan_id)
            if summary is None:
                raise HTTPException(status_code=404, detail=f"unknown scan_id: {scan_id}")
            return _scan_root_or_400(summary)
        return _root_or_400(default_source, "source", "CURATOR_SOURCE_PATH")

    def _image_file(path: str, scan_id: str | None) -> Path:
        target = _resolve_within(_media_root(scan_id), path)
        if not target.is_file():
            raise HTTPException(status_code=404, detail=f"not found: {path}")
        if target.suffix.lower() not in _IMAGE_TYPES:
            raise HTTPException(status_code=400, detail="unsupported image type")
        return target

    @app.post("/scan/{scan_id}/select", response_model=SelectResult)
    async def select_scan(scan_id: str, req: SelectRequest) -> SelectResult:
        """Apply the export filter to a cached scan and return keep reasons.

        Nothing is copied, symlinked, or moved, and the export root is not
        created. ``keep_reason`` is empty for images the filter keeps.
        """
        summary = store.load(scan_id)
        if summary is None:
            raise HTTPException(status_code=404, detail=f"unknown scan_id: {scan_id}")
        _scan_root_or_400(summary)
        export_req = req.as_export_request()
        for problem in (
            garment_floor_problem(export_req, summary.garments_config.min_conf),
            nsfw_filter_problem(export_req, summary),
            guard_filter_problem(export_req, summary),
        ):
            if problem is not None:
                raise HTTPException(status_code=400, detail=problem)
        _selected, reasons = await asyncio.to_thread(
            decide_selection, summary.results, export_req, summary.config.diversity_weight
        )
        rows = [
            SelectRow(rel_path=r.rel_path, abs_path=r.abs_path, keep_reason=reasons.get(r.rel_path, ""))
            for r in summary.results
        ]
        return SelectResult(
            scan_id=scan_id,
            total=len(rows),
            kept=sum(1 for row in rows if row.keep_reason == ""),
            rows=rows,
        )

    @app.post("/scan/{scan_id}/delete")
    async def delete_image(scan_id: str, req: DeleteRequest) -> dict[str, Any]:
        """Delete source files that MiVOLO reads as under 18 with an exposed score.

        One path or a batch. If any path is not that kind of image, nothing is
        deleted. A scan that is still running is refused too, because its next
        save would put the rows back.
        """
        paths = req.targets()
        if not paths:
            raise HTTPException(status_code=400, detail="no images to delete")
        active = gate.snapshot()
        if active and active["scan_id"] == scan_id:
            raise HTTPException(status_code=409, detail=_busy_detail(active))
        summary = store.load(scan_id)
        if summary is None:
            raise HTTPException(status_code=404, detail=f"unknown scan_id: {scan_id}")
        _scan_root_or_400(summary)
        by_path = {r.rel_path: r for r in summary.results}
        missing = [path for path in paths if path not in by_path]
        if missing:
            raise HTTPException(status_code=404, detail=f"not in scan: {missing[0]}")
        refused = [path for path in paths if not minor_exposed(by_path[path])]
        if refused:
            raise HTTPException(
                status_code=403,
                detail="Only an image with a MiVOLO age under 18 and an exposed score of at least 0.2 can be deleted here.",
            )

        targets = [_image_file(path, scan_id) for path in paths]

        def _unlink() -> None:
            for target in targets:
                target.unlink()

        try:
            await asyncio.to_thread(_unlink)
        except OSError as exc:
            raise HTTPException(status_code=400, detail=f"could not delete: {exc}") from exc
        gone = set(paths)
        summary.results = [r for r in summary.results if r.rel_path not in gone]
        summary.total = len(summary.results)
        summary.passed = sum(1 for r in summary.results if r.passed)
        summary.rejected = summary.total - summary.passed
        summary.duplicates = sum(1 for r in summary.results if r.is_duplicate)
        summary.returned = len(summary.results)
        await asyncio.to_thread(store.save, summary)
        body: dict[str, Any] = {"rel_paths": paths, "total": summary.total}
        if len(paths) == 1:
            body["rel_path"] = paths[0]
        return body

    @app.get("/thumb")
    async def thumb(
        path: str = Query(..., description="image path relative to the scan/mount root"),
        scan_id: str | None = Query(None, description="resolve relative to this scan's folder"),
    ) -> Response:
        target = _image_file(path, scan_id)

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

    @app.get("/image")
    async def image(
        path: str = Query(..., description="image path relative to the scan/mount root"),
        scan_id: str | None = Query(None, description="resolve relative to this scan's folder"),
    ) -> FileResponse:
        """Serve the original file. Same containment rules as ``/thumb``.

        An under-18 image with an exposed score is not served. The grid keeps a
        blurred thumbnail, and the file itself is only removed by delete.
        """
        if scan_id:
            summary = store.load(scan_id)
            match = next((r for r in summary.results if r.rel_path == path), None) if summary else None
            if match is not None and minor_exposed(match):
                raise HTTPException(status_code=403, detail="This image stays blurred.")
        target = _image_file(path, scan_id)
        return FileResponse(target, media_type=_IMAGE_TYPES[target.suffix.lower()])

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
        for problem in (
            garment_floor_problem(req, summary.garments_config.min_conf),
            nsfw_filter_problem(req, summary),
            guard_filter_problem(req, summary),
        ):
            if problem is not None:
                raise HTTPException(status_code=400, detail=problem)
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
