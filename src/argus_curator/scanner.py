"""Main curation orchestrator.

Pipeline:
  Phase 1 (CPU threads)   — open, filter, sharpness, artifacts, pHash
  Phase 2 (GPU, optional) — CLIP/DINOv2 embeddings + aesthetic scoring
  Phase 2b (GPU, optional)— YOLO person + MTCNN face detection
  Phase 3 (CPU)           — de-duplication, scoring, clustering, selection
"""

from __future__ import annotations

import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import imagehash
import structlog
from PIL import Image

from argus_curator import clustering, filters, scoring, selection
from argus_curator.types import CurateConfig, ImageResult, ScanSummary

logger = structlog.get_logger()

_SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


# ---------------------------------------------------------------------------
# Phase 1: per-image CPU assessment
# ---------------------------------------------------------------------------

def _phase1(name: str, source: str, data: bytes, cfg: CurateConfig) -> tuple[ImageResult, Image.Image | None]:
    """Assess one image in a thread.  Returns (result, pil_image_or_None)."""
    img, err = filters.open_image(data)
    if err:
        return ImageResult(
            name=name, source=source, width=0, height=0,
            short_side=0, aspect_ratio=0.0,
            sharpness=0.0, artifact_score=0.0, phash="",
            passed=False, reject_reason=f"cannot open: {err}",
            is_duplicate=False, duplicate_of=None,
        ), None

    w, h = img.size
    short, long = min(w, h), max(w, h)
    aspect = long / short if short else 0.0

    if reason := filters.check_resolution(img, cfg.filters.min_short_side):
        return ImageResult(
            name=name, source=source, width=w, height=h,
            short_side=short, aspect_ratio=round(aspect, 3),
            sharpness=0.0, artifact_score=0.0,
            phash=str(imagehash.phash(img, hash_size=8)),
            passed=False, reject_reason=reason,
            is_duplicate=False, duplicate_of=None,
        ), None

    if reason := filters.check_aspect(img, cfg.filters.max_aspect_ratio):
        return ImageResult(
            name=name, source=source, width=w, height=h,
            short_side=short, aspect_ratio=round(aspect, 3),
            sharpness=0.0, artifact_score=0.0,
            phash=str(imagehash.phash(img, hash_size=8)),
            passed=False, reject_reason=reason,
            is_duplicate=False, duplicate_of=None,
        ), None

    sharp, blur_reason = filters.check_blur(img, cfg.filters.blur_threshold)
    ph = str(imagehash.phash(img, hash_size=8))
    art = filters.artifact_score(img)

    result = ImageResult(
        name=name, source=source, width=w, height=h,
        short_side=short, aspect_ratio=round(aspect, 3),
        sharpness=round(sharp, 2), artifact_score=round(art, 4),
        phash=ph, passed=blur_reason is None,
        reject_reason=blur_reason,
        is_duplicate=False, duplicate_of=None,
    )
    return result, img if blur_reason is None else None


# ---------------------------------------------------------------------------
# Batch entry points
# ---------------------------------------------------------------------------

def scan_bytes_batch(
    items: list[tuple[str, str, bytes]],
    cfg: CurateConfig | None = None,
) -> ScanSummary:
    """Curate a list of (name, source, bytes).  Returns a ScanSummary."""
    cfg = cfg or CurateConfig()
    return _run(items, cfg)


def scan_folder(folder: str | Path, cfg: CurateConfig | None = None) -> ScanSummary:
    """Curate all supported images under *folder*."""
    cfg = cfg or CurateConfig()
    root = Path(folder)
    paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in _SUPPORTED_EXTS)
    if not paths:
        logger.warning("scan_folder_empty", folder=str(root))
        return _empty_summary(cfg)

    logger.info("scan_folder_start", folder=str(root), n=len(paths))
    items: list[tuple[str, str, bytes]] = []
    for p in paths:
        try:
            rel = p.relative_to(root).as_posix()
            items.append((rel, f"local:{p}", p.read_bytes()))
        except OSError as exc:
            logger.warning("scan_read_error", path=str(p), error=str(exc))

    return _run(items, cfg)


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def _run(
    items: list[tuple[str, str, bytes]],
    cfg: CurateConfig,
) -> ScanSummary:
    n = len(items)
    logger.info("curator_start", n=n, objective=cfg.objective)

    # ── Phase 1: parallel CPU assessment ──────────────────────────────────
    results_map: dict[str, ImageResult] = {}
    images_map: dict[str, Image.Image] = {}

    def _worker(item: tuple[str, str, bytes]) -> tuple[ImageResult, Image.Image | None]:
        name, source, data = item
        r, img = _phase1(name, source, data, cfg)
        return r, img

    with ThreadPoolExecutor(max_workers=cfg.max_workers) as pool:
        futures = {pool.submit(_worker, item): item[0] for item in items}
        for fut in as_completed(futures):
            r, img = fut.result()
            results_map[r.name] = r
            if img is not None:
                images_map[r.name] = img

    # Preserve input order
    results: list[ImageResult] = [results_map[name] for name, _, _ in items]
    passing = [r for r in results if r.passed]

    logger.info("phase1_done", passing=len(passing), rejected=len(results) - len(passing))

    # ── Phase 2a: GPU embeddings + aesthetic scoring ───────────────────────
    use_embeddings = (
        (cfg.embeddings.use_clip or cfg.embeddings.use_dino)
        and len(passing) > 0
    )

    if use_embeddings:
        _run_embeddings(passing, images_map, cfg)
    else:
        # Phase-1-only score without aesthetic component
        has_det = cfg.detectors.use_yolo or cfg.detectors.use_mtcnn
        for r in passing:
            scoring.compute_score(r, cfg, aesthetic_score=0.5, has_detectors=has_det)

    # ── Phase 2b: GPU detectors ────────────────────────────────────────────
    if (cfg.detectors.use_yolo or cfg.detectors.use_mtcnn) and passing:
        _run_detectors(passing, images_map, cfg)
        # Rescore with detector results
        for r in passing:
            scoring.compute_score(
                r, cfg,
                aesthetic_score=r.aesthetic_score,
                has_detectors=True,
            )

    # ── Phase 3: de-duplication ────────────────────────────────────────────
    clustering.mark_duplicates(results, cfg.duplicates.phash_hamming_distance)

    # ── Phase 3: caption tag boost ─────────────────────────────────────────
    if cfg.selection.caption_tags:
        clustering.apply_tag_boost(results, cfg.selection.caption_tags)

    # ── Phase 3: embedding clustering ─────────────────────────────────────
    embedding_clustering = False
    if use_embeddings and cfg.selection.use_embedding_clusters:
        k = clustering.cluster_by_embeddings(results, cfg.selection.n_clusters)
        embedding_clustering = k > 1
        logger.info("clustering_done", n_clusters=k)

    # ── Phase 3: selection ─────────────────────────────────────────────────
    candidates = [r for r in results if r.passed and not r.is_duplicate]
    sel_cfg = cfg.selection
    if sel_cfg.target_count is not None:
        target_n = min(sel_cfg.target_count, len(candidates))
    else:
        target_n = max(1, round(len(candidates) * sel_cfg.top_percent / 100))

    selected = selection.select(
        results, target_n,
        sel_cfg.diversity_weight,
        sel_cfg.use_embedding_clusters,
    )
    for r in results:
        r.selected = False
    for r in selected:
        r.selected = True

    logger.info(
        "curator_done",
        total=len(results),
        selected=len(selected),
        rejected=sum(1 for r in results if not r.passed),
        duplicates=sum(1 for r in results if r.is_duplicate),
    )

    return _build_summary(results, cfg, embedding_clustering=embedding_clustering)


def _run_embeddings(
    passing: list[ImageResult],
    images_map: dict[str, Image.Image],
    cfg: CurateConfig,
) -> None:
    from argus_curator.embeddings import EmbeddingPool

    pil_images = [images_map[r.name] for r in passing]
    pool = EmbeddingPool(
        clip_model=cfg.embeddings.clip_model if cfg.embeddings.use_clip else None,
        dino_model=cfg.embeddings.dino_model if cfg.embeddings.use_dino else None,
        batch_size=cfg.embeddings.batch_size,
        device=cfg.embeddings.device,
    )
    embeddings_out = pool.run(pil_images)

    has_det = cfg.detectors.use_yolo or cfg.detectors.use_mtcnn
    for r, (clip_emb, dino_emb, aes) in zip(passing, embeddings_out):
        r.clip_embedding = clip_emb
        r.dino_embedding = dino_emb
        r.aesthetic_score = round(aes, 4)
        scoring.compute_score(r, cfg, aesthetic_score=aes, has_detectors=has_det)


def _run_detectors(
    passing: list[ImageResult],
    images_map: dict[str, Image.Image],
    cfg: CurateConfig,
) -> None:
    from argus_curator.detectors import DetectorPool

    pool = DetectorPool(
        use_yolo=cfg.detectors.use_yolo,
        yolo_model=cfg.detectors.yolo_model,
        yolo_confidence=cfg.detectors.yolo_confidence,
        use_mtcnn=cfg.detectors.use_mtcnn,
        mtcnn_confidence=cfg.detectors.mtcnn_confidence,
        device=cfg.detectors.device,
        batch_size=cfg.detectors.batch_size,
    )
    pil_images = [images_map[r.name] for r in passing]
    det_results = pool.run(pil_images)
    for r, dr in zip(passing, det_results):
        r.face_count = dr.face_count
        r.person_detected = dr.person_detected
        r.person_confidence = dr.person_confidence


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def _build_summary(
    results: list[ImageResult],
    cfg: CurateConfig,
    embedding_clustering: bool,
) -> ScanSummary:
    rejected = [r for r in results if not r.passed]
    dupes = [r for r in results if r.passed and r.is_duplicate]
    candidates = [r for r in results if r.passed and not r.is_duplicate]
    selected = [r for r in results if r.selected]

    tally: dict[str, int] = {}
    for r in rejected:
        key = (r.reject_reason or "unknown").split("(")[0].strip()
        tally[key] = tally.get(key, 0) + 1

    return ScanSummary(
        total=len(results),
        rejected_filters=len(rejected),
        duplicates_removed=len(dupes),
        candidates=len(candidates),
        selected=len(selected),
        objective=cfg.objective,
        target_style=cfg.target_style,
        diversity_weight=cfg.selection.diversity_weight,
        embedding_clustering=embedding_clustering,
        reject_reasons=tally,
        selected_names=[r.name for r in selected],
        results=[r.to_dict() for r in results],
    )


def _empty_summary(cfg: CurateConfig) -> ScanSummary:
    return ScanSummary(
        total=0, rejected_filters=0, duplicates_removed=0,
        candidates=0, selected=0,
        objective=cfg.objective, target_style=cfg.target_style,
        diversity_weight=cfg.selection.diversity_weight,
        embedding_clustering=False,
        reject_reasons={}, selected_names=[], results=[],
    )
