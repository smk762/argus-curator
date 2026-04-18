"""FastAPI server for argus-curator."""

from __future__ import annotations

import asyncio
import os
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from argus_curator.presets import PRESET_DESCRIPTIONS
from argus_curator.types import (
    CurateConfig, DetectorConfig, DuplicateConfig, EmbeddingConfig,
    FilterConfig, ScoringConfig, SelectionConfig, TRAINING_OBJECTIVES,
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


class ScanFolderRequest(BaseModel):
    folder: str = Field(description="Absolute path to a folder of images.")
    config: CurateConfigRequest = Field(default_factory=CurateConfigRequest)


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


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(*, cors: bool = False) -> FastAPI:
    app = FastAPI(
        title="argus-curator",
        description="Dataset curation API — quality filtering, diversity clustering, optimal subset selection.",
        version=_package_version()["version"] or "0.1.0",
    )

    if cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
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

        cfg = _to_curate_config(body.config)
        from argus_curator.scanner import scan_folder as _scan
        try:
            summary = await asyncio.to_thread(_scan, folder, cfg)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return summary.to_dict()

    @app.post("/export/move")
    async def export_move(body: dict[str, Any]) -> dict[str, Any]:
        """Move selected images to a target folder.

        Body: {sources: ["local:/abs/path", ...], target_folder: "/abs/path"}
        """
        import shutil
        sources: list[str] = body.get("sources", [])
        target: str = body.get("target_folder", "")
        if not target:
            raise HTTPException(status_code=400, detail="target_folder is required.")

        target_path = Path(target)
        target_path.mkdir(parents=True, exist_ok=True)

        moved, errors = 0, []
        for src in sources:
            if src.startswith("local:"):
                p = Path(src[len("local:"):])
                try:
                    shutil.move(str(p), str(target_path / p.name))
                    moved += 1
                except Exception as exc:
                    errors.append({"source": src, "error": str(exc)})

        return {"moved": moved, "errors": errors, "target_folder": target}

    @app.post("/export/copy")
    async def export_copy(body: dict[str, Any]) -> dict[str, Any]:
        """Copy selected images to a target folder."""
        import shutil
        sources: list[str] = body.get("sources", [])
        target: str = body.get("target_folder", "")
        if not target:
            raise HTTPException(status_code=400, detail="target_folder is required.")

        target_path = Path(target)
        target_path.mkdir(parents=True, exist_ok=True)

        copied, errors = 0, []
        for src in sources:
            if src.startswith("local:"):
                p = Path(src[len("local:"):])
                try:
                    shutil.copy2(str(p), str(target_path / p.name))
                    copied += 1
                except Exception as exc:
                    errors.append({"source": src, "error": str(exc)})

        return {"copied": copied, "errors": errors, "target_folder": target}

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


# ---------------------------------------------------------------------------
# Path import needed at module level for server.py
# ---------------------------------------------------------------------------
from pathlib import Path  # noqa: E402
