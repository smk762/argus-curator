"""FastAPI server for argus-curator."""

from __future__ import annotations

import asyncio
import io
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Annotated, Any

from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, model_validator
from starlette.background import BackgroundTask

from argus_curator.presets import PRESET_DESCRIPTIONS
from argus_curator.types import (
    CurateConfig,
    DetectorConfig,
    DuplicateConfig,
    EmbeddingConfig,
    FilterConfig,
    ScoringConfig,
    SelectionConfig,
    TRAINING_OBJECTIVES,
)
from argus_curator.web_sessions import (
    SUPPORTED_IMAGE_UPLOAD_EXT,
    clear_in_dir,
    collect_zip_members,
    create_session,
    destroy_session,
    resolve_member_under,
    sanitize_relative_path,
    session_in_dir,
)


# ---------------------------------------------------------------------------
# Pydantic request models
# ---------------------------------------------------------------------------

class FilterConfigRequest(BaseModel):
    min_short_side: int = Field(default=512, ge=64, le=4096)
    max_aspect_ratio: float = Field(default=3.0, gt=1.0, le=10.0)
    blur_threshold: float = Field(default=100.0, ge=0.0)


class DuplicateConfigRequest(BaseModel):
    phash_hamming_distance: int = Field(default=10, ge=0, le=64)


class EmbeddingConfigRequest(BaseModel):
    use_clip: bool = True
    clip_model: str = "openai/clip-vit-large-patch14"
    use_dino: bool = False
    dino_model: str = "facebook/dinov2-base"
    batch_size: int = Field(default=16, ge=1, le=128)
    device: str = "auto"


class DetectorConfigRequest(BaseModel):
    use_yolo: bool = False
    yolo_model: str = "yolov8n.pt"
    yolo_confidence: float = Field(default=0.40, ge=0.0, le=1.0)
    use_mtcnn: bool = False
    mtcnn_confidence: float = Field(default=0.90, ge=0.0, le=1.0)
    batch_size: int = Field(default=16, ge=1, le=128)
    device: str = "auto"


class ScoringConfigRequest(BaseModel):
    weight_sharpness: float = Field(default=0.35, ge=0.0, le=1.0)
    weight_resolution: float = Field(default=0.25, ge=0.0, le=1.0)
    weight_artifact: float = Field(default=0.15, ge=0.0, le=1.0)
    weight_aesthetic: float = Field(default=0.15, ge=0.0, le=1.0)
    weight_subject: float = Field(default=0.10, ge=0.0, le=1.0)
    sharpness_ref: float = Field(default=800.0, gt=0.0)
    resolution_ref: int = Field(default=1024, ge=64)


class SelectionConfigRequest(BaseModel):
    target_count: int | None = Field(default=None, ge=1, le=10000)
    top_percent: float = Field(default=80.0, gt=0.0, le=100.0)
    diversity_weight: float = Field(default=0.40, ge=0.0, le=1.0)
    use_embedding_clusters: bool = True
    n_clusters: int | None = Field(default=None, ge=2, le=200)
    caption_tags: dict[str, list[str]] | None = None


class CurateConfigRequest(BaseModel):
    objective: str = Field(
        default="identity",
        description=f"Training objective preset. One of: {', '.join(TRAINING_OBJECTIVES)}.",
    )
    target_style: str = Field(default="photo", description="'photo' or 'anime'.")
    apply_preset: bool = Field(
        default=True,
        description="Apply objective-specific defaults before overrides.",
    )
    filters: FilterConfigRequest = Field(default_factory=FilterConfigRequest)
    duplicates: DuplicateConfigRequest = Field(default_factory=DuplicateConfigRequest)
    embeddings: EmbeddingConfigRequest = Field(default_factory=EmbeddingConfigRequest)
    detectors: DetectorConfigRequest = Field(default_factory=DetectorConfigRequest)
    scoring: ScoringConfigRequest = Field(default_factory=ScoringConfigRequest)
    selection: SelectionConfigRequest = Field(default_factory=SelectionConfigRequest)
    max_workers: int = Field(default=4, ge=1, le=32)


class ScanFolderRequest(CurateConfigRequest):
    """Folder path plus flat curation fields (same JSON shape as the demo SPA)."""

    folder: str = Field(description="Absolute path to a folder of images.")

    @model_validator(mode="before")
    @classmethod
    def _merge_legacy_nested_config(cls, data: Any) -> Any:
        if isinstance(data, dict) and isinstance(data.get("config"), dict):
            merged = {**data["config"], **{k: v for k, v in data.items() if k != "config"}}
            return merged
        return data


class ExportZipRequest(BaseModel):
    selected_names: list[str] = Field(
        default_factory=list,
        description="Basenames or relative paths returned by a scan (e.g. selected_names).",
    )


# ---------------------------------------------------------------------------
# Request → config conversion
# ---------------------------------------------------------------------------

def _to_curate_config(req: CurateConfigRequest) -> CurateConfig:
    cfg = CurateConfig(
        objective=req.objective,
        target_style=req.target_style,
        filters=FilterConfig(
            min_short_side=req.filters.min_short_side,
            max_aspect_ratio=req.filters.max_aspect_ratio,
            blur_threshold=req.filters.blur_threshold,
        ),
        duplicates=DuplicateConfig(
            phash_hamming_distance=req.duplicates.phash_hamming_distance,
        ),
        embeddings=EmbeddingConfig(
            use_clip=req.embeddings.use_clip,
            clip_model=req.embeddings.clip_model,
            use_dino=req.embeddings.use_dino,
            dino_model=req.embeddings.dino_model,
            batch_size=req.embeddings.batch_size,
            device=req.embeddings.device,
        ),
        detectors=DetectorConfig(
            use_yolo=req.detectors.use_yolo,
            yolo_model=req.detectors.yolo_model,
            yolo_confidence=req.detectors.yolo_confidence,
            use_mtcnn=req.detectors.use_mtcnn,
            mtcnn_confidence=req.detectors.mtcnn_confidence,
            batch_size=req.detectors.batch_size,
            device=req.detectors.device,
        ),
        scoring=ScoringConfig(
            weight_sharpness=req.scoring.weight_sharpness,
            weight_resolution=req.scoring.weight_resolution,
            weight_artifact=req.scoring.weight_artifact,
            weight_aesthetic=req.scoring.weight_aesthetic,
            weight_subject=req.scoring.weight_subject,
            sharpness_ref=req.scoring.sharpness_ref,
            resolution_ref=req.scoring.resolution_ref,
        ),
        selection=SelectionConfig(
            target_count=req.selection.target_count,
            top_percent=req.selection.top_percent,
            diversity_weight=req.selection.diversity_weight,
            use_embedding_clusters=req.selection.use_embedding_clusters,
            n_clusters=req.selection.n_clusters,
            caption_tags=req.selection.caption_tags,
        ),
        max_workers=req.max_workers,
    )
    if req.apply_preset:
        from argus_curator.presets import apply_preset
        apply_preset(cfg)
    return cfg


def _export_dest_path(target_path: Path, src_path: Path, dest_name: str | None) -> Path:
    if dest_name:
        safe = sanitize_relative_path(dest_name.replace("\\", "/"))
        if safe is None:
            return target_path / src_path.name
        return target_path.joinpath(*Path(safe).parts)
    return target_path / src_path.name


def _run_export(
    *,
    sources: list[str],
    target: str,
    dest_names: list[str] | None,
    move: bool,
) -> dict[str, Any]:
    if not target:
        raise HTTPException(status_code=400, detail="target_folder is required.")
    target_path = Path(target)
    target_path.mkdir(parents=True, exist_ok=True)

    done, errors = 0, []
    for i, src in enumerate(sources):
        if not src.startswith("local:"):
            continue
        p = Path(src[len("local:"):])
        hint = dest_names[i] if dest_names and i < len(dest_names) else None
        dest = _export_dest_path(target_path, p, hint)
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            if move:
                shutil.move(str(p), str(dest))
            else:
                shutil.copy2(str(p), str(dest))
            done += 1
        except Exception as exc:
            errors.append({"source": src, "error": str(exc)})

    key_moved = "moved" if move else "copied"
    return {key_moved: done, "errors": errors, "target_folder": target}


# ---------------------------------------------------------------------------
# CORS (demo SPA / browser clients)
# ---------------------------------------------------------------------------

def _env_truthy(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _cors_allow_origins_from_env() -> list[str]:
    """Comma-separated `ARGUS_CURATOR_CORS_ORIGINS`; empty → all origins (`*`)."""
    raw = os.environ.get("ARGUS_CURATOR_CORS_ORIGINS", "")
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return parts if parts else ["*"]


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(*, cors: bool | None = None) -> FastAPI:
    """Create the FastAPI app.

    CORS is enabled when ``cors`` is true, or when ``cors`` is ``None`` and
    ``ARGUS_CURATOR_CORS`` is truthy (``1``, ``true``, ``yes``, ``on``).
    Allowed origins: ``ARGUS_CURATOR_CORS_ORIGINS`` (comma-separated), or
    ``["*"]`` if unset/empty.
    """
    app = FastAPI(
        title="argus-curator",
        description="Dataset curation API — quality filtering, diversity clustering, optimal subset selection.",
        version=_package_version()["version"] or "0.1.0",
    )

    if cors is None:
        cors_enabled = _env_truthy("ARGUS_CURATOR_CORS", default=False)
    else:
        cors_enabled = cors

    if cors_enabled:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=_cors_allow_origins_from_env(),
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.get("/")
    async def health() -> dict[str, Any]:
        return {"status": "ok", **_package_version()}

    @app.get("/version")
    async def version() -> dict[str, Any]:
        return _package_version()

    @app.get("/presets")
    async def presets() -> dict[str, Any]:
        return PRESET_DESCRIPTIONS

    @app.get("/detectors")
    async def detector_availability() -> dict[str, Any]:
        from argus_curator import detectors, embeddings
        info = await asyncio.to_thread(embeddings.availability)
        info.update(await asyncio.to_thread(detectors.availability))
        return info

    @app.post("/scan/folder")
    async def scan_folder(body: ScanFolderRequest) -> dict[str, Any]:
        folder = body.folder.strip()
        if not folder:
            raise HTTPException(status_code=400, detail="folder is required.")
        if not os.path.isdir(folder):
            raise HTTPException(status_code=422, detail=f"Not a directory: {folder!r}")

        cfg = _to_curate_config(body)
        from argus_curator.scanner import scan_folder as _scan
        try:
            summary = await asyncio.to_thread(_scan, folder, cfg)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return summary.to_dict()

    @app.post("/export/move")
    async def export_move(body: dict[str, Any]) -> dict[str, Any]:
        """Body: {sources: ["local:/abs/path", ...], target_folder, dest_names?: [...]}."""
        sources: list[str] = body.get("sources", [])
        dest_names: list[str] | None = body.get("dest_names")
        target: str = body.get("target_folder", "")
        return _run_export(sources=sources, target=target, dest_names=dest_names, move=True)

    @app.post("/export/copy")
    async def export_copy(body: dict[str, Any]) -> dict[str, Any]:
        """Body: {sources: [...], target_folder, dest_names?: [...]}."""
        sources: list[str] = body.get("sources", [])
        dest_names: list[str] | None = body.get("dest_names")
        target: str = body.get("target_folder", "")
        return _run_export(sources=sources, target=target, dest_names=dest_names, move=False)

    # ── Ephemeral browser sessions (temp dir on server) ───────────────────

    @app.post("/sessions")
    async def create_browser_session() -> dict[str, str]:
        return {"session_id": create_session()}

    @app.delete("/sessions/{session_id}")
    async def delete_browser_session(session_id: str) -> dict[str, bool]:
        ok = destroy_session(session_id)
        if not ok:
            raise HTTPException(status_code=404, detail="Unknown session_id.")
        return {"deleted": True}

    @app.post("/sessions/{session_id}/upload")
    async def session_upload_files(
        session_id: str,
        files: Annotated[list[UploadFile], File()],
    ) -> dict[str, Any]:
        in_dir = session_in_dir(session_id)
        if in_dir is None:
            raise HTTPException(status_code=404, detail="Unknown session_id.")
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded.")

        clear_in_dir(in_dir)
        saved = 0
        for uf in files:
            raw_name = (uf.filename or "image.jpg").replace("\\", "/")
            rel = sanitize_relative_path(raw_name)
            if rel is None:
                continue
            suf = Path(rel).suffix.lower()
            if suf not in SUPPORTED_IMAGE_UPLOAD_EXT:
                continue
            dest = resolve_member_under(in_dir, rel)
            if dest is None:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                data = await uf.read()
                dest.write_bytes(data)
                saved += 1
            finally:
                await uf.close()
        if saved == 0:
            raise HTTPException(
                status_code=400,
                detail="No supported images (.jpg, .jpeg, .png, .webp) were saved.",
            )
        return {"saved": saved, "session_id": session_id}

    @app.post("/sessions/{session_id}/upload-zip")
    async def session_upload_zip(session_id: str, archive: UploadFile = File(...)) -> dict[str, Any]:
        in_dir = session_in_dir(session_id)
        if in_dir is None:
            raise HTTPException(status_code=404, detail="Unknown session_id.")
        raw = await archive.read()
        await archive.close()
        if not raw:
            raise HTTPException(status_code=400, detail="Empty archive.")

        clear_in_dir(in_dir)
        saved = 0
        try:
            zf = zipfile.ZipFile(io.BytesIO(raw))
        except zipfile.BadZipFile as exc:
            raise HTTPException(status_code=400, detail="Not a valid zip file.") from exc

        with zf:
            for member in zf.infolist():
                if member.is_dir():
                    continue
                rel = sanitize_relative_path(member.filename.replace("\\", "/"))
                if rel is None:
                    continue
                suf = Path(rel).suffix.lower()
                if suf not in SUPPORTED_IMAGE_UPLOAD_EXT:
                    continue
                dest = resolve_member_under(in_dir, rel)
                if dest is None:
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                try:
                    dest.write_bytes(zf.read(member))
                    saved += 1
                except OSError:
                    continue

        if saved == 0:
            raise HTTPException(
                status_code=400,
                detail="Zip contained no supported images (.jpg, .jpeg, .png, .webp).",
            )
        return {"saved": saved, "session_id": session_id}

    @app.post("/sessions/{session_id}/scan")
    async def session_scan(session_id: str, body: CurateConfigRequest = Body(...)) -> dict[str, Any]:
        in_dir = session_in_dir(session_id)
        if in_dir is None:
            raise HTTPException(status_code=404, detail="Unknown session_id.")
        has_image = any(
            p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_UPLOAD_EXT
            for p in in_dir.rglob("*")
        )
        if not in_dir.is_dir() or not has_image:
            raise HTTPException(status_code=400, detail="Session upload folder is empty.")

        cfg = _to_curate_config(body)
        from argus_curator.scanner import scan_folder as _scan
        try:
            summary = await asyncio.to_thread(_scan, str(in_dir), cfg)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        out = summary.to_dict()
        out["session_id"] = session_id
        return out

    @app.post("/sessions/{session_id}/export-zip")
    async def session_export_zip(session_id: str, body: ExportZipRequest) -> FileResponse:
        in_dir = session_in_dir(session_id)
        if in_dir is None:
            raise HTTPException(status_code=404, detail="Unknown session_id.")

        names = body.selected_names
        if not names:
            raise HTTPException(status_code=400, detail="selected_names is required.")

        pairs = collect_zip_members(in_dir, names)
        if not pairs:
            raise HTTPException(
                status_code=422,
                detail="None of the requested files exist in this session.",
            )

        tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
        tmp_path = tmp.name
        tmp.close()
        try:
            with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for abs_path, arcname in pairs:
                    zf.write(abs_path, arcname=arcname)
        except Exception:
            os.unlink(tmp_path)
            raise

        return FileResponse(
            tmp_path,
            filename="curated-subset.zip",
            media_type="application/zip",
            background=BackgroundTask(lambda p=tmp_path: os.unlink(p) if os.path.exists(p) else None),
        )

    return app


# ---------------------------------------------------------------------------
# Version helper
# ---------------------------------------------------------------------------

def _package_version() -> dict[str, str | None]:
    try:
        from argus_curator import _version as v
        cid = getattr(v, "commit_id", None) or getattr(v, "__commit_id__", None)
        return {"version": v.__version__, "commit": cid}
    except Exception:
        try:
            from importlib.metadata import version as pkg_version
            return {"version": pkg_version("argus-curator"), "commit": None}
        except Exception:
            return {"version": "unknown", "commit": None}
