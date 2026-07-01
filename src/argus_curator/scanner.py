"""Training-suitability scanner — ported from imogen ``gallery/image_scanner.py``.

Differences from the imogen original (intentional):

  * **Relative-path keying.** Images are keyed by their path relative to the
    scanned root, not by basename. imogen's ``scan_folder`` clobbers same-named
    files across sub-folders; PR #14 fixed this for curation and we keep that.
  * **Pydantic results.** Emits :class:`argus_curator.models.ImageResult` so the
    output is the server's wire format directly.
  * **Faces, not generic subject detection.** The CLIP/YOLO/MTCNN Phase-2 is
    replaced by the dedicated InsightFace pipeline in :mod:`argus_curator.faces`.

Phase 1 (parallel CPU threads, no GPU):
    1. Resolution / aspect-ratio filter
    2. Sharpness  — Laplacian-edge variance via PIL FIND_EDGES
    3. Artifact   — JPEG 8x8 block-boundary ratio
    4. pHash      — perceptual hash for near-duplicate clustering
    5. Target-aware composite score
"""

from __future__ import annotations

import io
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import imagehash
import numpy as np
import structlog
from PIL import Image, ImageFilter

from argus_curator.models import (
    SUPPORTED_EXTS,
    FaceConfig,
    ImageResult,
    ScanConfig,
    ScanSummary,
    TargetCategory,
    TargetProfile,
)

logger = structlog.get_logger()

# A progress sink: called with {"phase": str, "done": int, "total": int}.
# Kept deliberately tolerant — any raised exception is swallowed so a flaky
# consumer (e.g. a disconnected SSE client) can never abort the scan itself.
ProgressFn = Callable[[dict[str, Any]], None]


def _emit(progress: ProgressFn | None, phase: str, done: int, total: int) -> None:
    if progress is None:
        return
    try:
        progress({"phase": phase, "done": done, "total": total})
    except Exception:  # pragma: no cover - never let reporting break a scan
        pass


# ---------------------------------------------------------------------------
# Per-image Phase-1 metrics
# ---------------------------------------------------------------------------


def _sharpness(img: Image.Image) -> float:
    """Variance of the Laplacian edge response — high = sharp, low = blurry."""
    edges = img.convert("L").filter(ImageFilter.FIND_EDGES)
    return float(np.array(edges, dtype=np.float32).var())


def _artifact_score(img: Image.Image) -> float:
    """Estimate JPEG block-artifact level from 8x8 boundary discontinuities.

    Returns [0, 1] where 1.0 = clean. Compares the mean absolute difference at
    every 8-pixel boundary to the overall inter-pixel variation; a high ratio
    indicates visible blocking typical of heavy JPEG compression.
    """
    gray = np.array(img.convert("L"), dtype=np.float32)
    h, w = gray.shape

    h_b = float(np.abs(gray[8::8, :] - gray[7:-1:8, :]).mean()) if h > 16 else 0.0
    v_b = float(np.abs(gray[:, 8::8] - gray[:, 7:-1:8]).mean()) if w > 16 else 0.0
    boundary = (h_b + v_b) / 2.0

    internal_h = float(np.abs(np.diff(gray, axis=0)).mean())
    internal_v = float(np.abs(np.diff(gray, axis=1)).mean())
    internal = (internal_h + internal_v) / 2.0

    if internal < 1e-6:
        return 1.0
    ratio = boundary / (internal + 1e-6)
    return float(max(0.0, min(1.0, 1.0 - (ratio - 1.0) / 0.6)))


# ---------------------------------------------------------------------------
# Target-aware scoring
# ---------------------------------------------------------------------------


def _face_penalty(face_count: int, category: TargetCategory) -> float:
    """Penalise images that have the wrong number of faces for the target.

    Identity training wants exactly one prominent face; other categories are
    progressively more tolerant. Ported from imogen's ``_subject_face_penalty``.
    """
    if category == "identity":
        if face_count == 0:
            return 0.5
        if face_count >= 2:
            return 0.4
        return 1.0
    if category == "wardrobe":
        if face_count == 0:
            return 0.75
        if face_count >= 2:
            return 0.6
        return 1.0
    if category == "pose_composition":
        if face_count == 0:
            return 0.85
        if face_count >= 2:
            return 0.7
        return 1.0
    # setting
    if face_count >= 2:
        return 0.85
    return 1.0


def _target_bonus(result: ImageResult, profile: TargetProfile, faces_known: bool) -> float:
    """Small additive bonus rewarding framing/composition that fits the target."""
    category = profile.target_category
    orientation = result.width / max(result.height, 1)
    bonus = 0.0

    if category == "identity":
        if faces_known and result.face_count == 1:
            bonus += 0.08
        if faces_known and result.face_count >= 1:
            bonus += 0.03
        if 0.7 <= orientation <= 1.35:
            bonus += 0.03
    elif category == "wardrobe":
        if faces_known and result.face_count >= 1:
            bonus += 0.06
        if orientation <= 0.8:
            bonus += 0.08
        elif orientation <= 1.0:
            bonus += 0.04
        if faces_known and result.face_count == 1:
            bonus += 0.02
    elif category == "pose_composition":
        if faces_known and result.face_count >= 1:
            bonus += 0.04
        if orientation <= 0.82 or orientation >= 1.18:
            bonus += 0.07
        elif orientation <= 0.95 or orientation >= 1.05:
            bonus += 0.03
    elif category == "setting":
        short_side = min(result.width, result.height)
        if orientation >= 1.15:
            bonus += 0.08
        elif orientation >= 1.0:
            bonus += 0.03
        if short_side >= int(1024 * 0.9):
            bonus += 0.03

    if profile.target_style == "anime":
        if category in {"identity", "wardrobe"}:
            bonus += 0.01
        if result.artifact_score >= 0.85:
            bonus += 0.01

    return min(0.15, bonus)


def _base_score(result: ImageResult, cfg: ScanConfig) -> tuple[float, dict[str, float]]:
    short = min(result.width, result.height)
    sharp_norm = min(1.0, result.sharpness / cfg.sharpness_ref) if cfg.sharpness_ref else 0.0
    res_norm = min(1.0, short / cfg.resolution_ref) if cfg.resolution_ref else 0.0
    total_w = cfg.weight_sharpness + cfg.weight_resolution + cfg.weight_artifact
    breakdown = {
        "sharpness": cfg.weight_sharpness * sharp_norm,
        "resolution": cfg.weight_resolution * res_norm,
        "artifact": cfg.weight_artifact * result.artifact_score,
    }
    base = sum(breakdown.values()) / (total_w or 1.0)
    return base, breakdown


def finalize_score(result: ImageResult, profile: TargetProfile, cfg: ScanConfig, faces_known: bool) -> None:
    """(Re)compute the composite score in-place from stored metrics + face_count."""
    if not result.passed:
        return
    base, breakdown = _base_score(result, cfg)
    bonus = _target_bonus(result, profile, faces_known)
    breakdown["target_bonus"] = bonus
    score = base + bonus
    if faces_known:
        penalty = _face_penalty(result.face_count, profile.target_category)
        breakdown["face_penalty"] = penalty
        score *= penalty
    result.score = round(min(1.0, max(0.0, score)), 4)
    result.score_breakdown = {k: round(v, 4) for k, v in breakdown.items()}


# ---------------------------------------------------------------------------
# Phase-1 single-image assessment
# ---------------------------------------------------------------------------


def _make_rejected(rel: str, abs_path: str, img: Image.Image | None, reason: str, ph: str) -> ImageResult:
    w, h = img.size if img is not None else (0, 0)
    return ImageResult(
        rel_path=rel,
        abs_path=abs_path,
        passed=False,
        reject_reason=reason,
        width=w,
        height=h,
        phash=ph,
    )


def _score_image(rel: str, abs_path: str, data: bytes, profile: TargetProfile, cfg: ScanConfig) -> ImageResult:
    """Phase-1 assessment for one image (no GPU, thread-safe)."""
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as exc:
        logger.warning("scan_image_open_failed", rel_path=rel, error=str(exc))
        return _make_rejected(rel, abs_path, None, f"cannot open: {exc}", "")

    ph = str(imagehash.phash(img, hash_size=8))
    w, h = img.size
    short, long = min(w, h), max(w, h)
    aspect = (long / short) if short else 0.0

    if short < cfg.min_short_side:
        return _make_rejected(rel, abs_path, img, f"resolution too low ({short}px short side)", ph)
    if aspect > cfg.max_aspect_ratio:
        return _make_rejected(rel, abs_path, img, f"aspect ratio {aspect:.2f} > {cfg.max_aspect_ratio}", ph)

    sharp = _sharpness(img)
    if sharp < cfg.blur_threshold:
        return _make_rejected(rel, abs_path, img, f"blurry (sharpness={sharp:.1f} < {cfg.blur_threshold})", ph)

    art = _artifact_score(img)
    result = ImageResult(
        rel_path=rel,
        abs_path=abs_path,
        passed=True,
        width=w,
        height=h,
        sharpness=round(sharp, 2),
        artifact_score=round(art, 4),
        phash=ph,
    )
    finalize_score(result, profile, cfg, faces_known=False)
    return result


# ---------------------------------------------------------------------------
# Image collection (relative-path keyed)
# ---------------------------------------------------------------------------


def collect_images(root: Path) -> list[tuple[str, str, bytes]]:
    """Load ``(rel_path, abs_path, bytes)`` for every supported image under *root*."""
    items: list[tuple[str, str, bytes]] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in SUPPORTED_EXTS:
            continue
        rel = p.relative_to(root).as_posix()
        try:
            items.append((rel, str(p.resolve()), p.read_bytes()))
        except OSError as exc:
            logger.warning("scan_collect_read_error", path=str(p), error=str(exc))
    return items


# ---------------------------------------------------------------------------
# Near-duplicate clustering
# ---------------------------------------------------------------------------


def _mark_duplicates(results: list[ImageResult], cfg: ScanConfig) -> None:
    """Tag near-duplicates in-place; keep the highest-scoring representative.

    A negative ``cluster_distance`` disables grouping entirely (every image is
    its own cluster).
    """
    if cfg.cluster_distance < 0:
        return
    passing = sorted((r for r in results if r.passed), key=lambda r: -r.score)
    seen: list[tuple[imagehash.ImageHash, str]] = []
    for result in passing:
        if not result.phash:
            continue
        ph = imagehash.hex_to_hash(result.phash)
        for ref_hash, ref_rel in seen:
            if (ph - ref_hash) <= cfg.cluster_distance:
                result.is_duplicate = True
                result.duplicate_of = ref_rel
                break
        else:
            seen.append((ph, result.rel_path))


def _assign_clusters(results: list[ImageResult]) -> int:
    """Fill ``similar_group`` / ``group_size`` / ``is_representative`` in-place.

    Returns the number of multi-member clusters (for the summary).
    """
    rep_of = {r.rel_path: (r.duplicate_of if r.is_duplicate and r.duplicate_of else r.rel_path) for r in results}
    group_id: dict[str, int] = {}
    sizes: dict[str, int] = {}
    for rep in rep_of.values():
        sizes[rep] = sizes.get(rep, 0) + 1
        if rep not in group_id:
            group_id[rep] = len(group_id) + 1

    multi = 0
    for r in results:
        rep = rep_of[r.rel_path]
        r.similar_group = group_id[rep]
        r.group_size = sizes[rep]
        r.is_representative = rep == r.rel_path
    multi = sum(1 for rep, size in sizes.items() if size > 1)
    return multi


def _tally_reasons(rejected: list[ImageResult]) -> dict[str, int]:
    tally: dict[str, int] = {}
    for r in rejected:
        key = (r.reject_reason or "unknown").split("(")[0].strip()
        tally[key] = tally.get(key, 0) + 1
    return tally


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def scan_items(
    items: list[tuple[str, str, bytes]],
    profile: TargetProfile,
    cfg: ScanConfig,
    faces_cfg: FaceConfig,
    *,
    folder: str = "",
    scan_id: str | None = None,
    progress: ProgressFn | None = None,
) -> ScanSummary:
    """Score, (optionally) face-cluster, and de-duplicate a batch of images.

    When *progress* is supplied it is called with ``{phase, done, total}`` dicts
    as the scan advances through its ``scoring`` -> ``faces`` -> ``clustering``
    phases, so a caller can drive a live progress UI (see the SSE endpoint).
    """
    scan_id = scan_id or uuid.uuid4().hex
    total = len(items)

    # Phase 1: per-image scoring — the long pole, reported incrementally.
    _emit(progress, "scoring", 0, total)
    results_by_rel: dict[str, ImageResult] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, cfg.max_workers)) as pool:
        futures = {pool.submit(_score_image, rel, abs_path, data, profile, cfg): rel for rel, abs_path, data in items}
        for fut in as_completed(futures):
            r = fut.result()
            results_by_rel[r.rel_path] = r
            done += 1
            # Throttle: every 5 images (and the final one) is plenty for a bar.
            if done == total or done % 5 == 0:
                _emit(progress, "scoring", done, total)

    results = [results_by_rel[rel] for rel, _, _ in items]

    # M2: face detection + identity clustering (optional, GPU-friendly).
    face_clusters = []
    if faces_cfg.enabled:
        from argus_curator import faces as faces_mod

        _emit(progress, "faces", 0, total)
        face_clusters = faces_mod.detect_and_cluster(results, items, faces_cfg)
        for r in results:
            finalize_score(r, profile, cfg, faces_known=True)
        _emit(progress, "faces", total, total)

    _emit(progress, "clustering", 0, total)
    _mark_duplicates(results, cfg)
    multi = _assign_clusters(results)
    _emit(progress, "clustering", total, total)

    rejected = [r for r in results if not r.passed]
    passed = [r for r in results if r.passed]
    duplicates = [r for r in passed if r.is_duplicate]

    return ScanSummary(
        scan_id=scan_id,
        folder=folder,
        target_profile=profile,
        config=cfg,
        faces_config=faces_cfg,
        total=len(results),
        passed=len(passed),
        rejected=len(rejected),
        duplicates=len(duplicates),
        similar_clusters=multi,
        reject_reasons=_tally_reasons(rejected),
        face_clusters=face_clusters,
        results=results,
        returned=len(results),
    )


def scan_folder(
    folder: str | Path,
    profile: TargetProfile | None = None,
    cfg: ScanConfig | None = None,
    faces_cfg: FaceConfig | None = None,
    *,
    scan_id: str | None = None,
    progress: ProgressFn | None = None,
) -> ScanSummary:
    """Recursively scan all supported images under *folder*.

    Pass *progress* to receive ``{phase, done, total}`` updates (used by the
    SSE endpoint to stream live scan progress to the browser).
    """
    profile = profile or TargetProfile()
    cfg = cfg or ScanConfig()
    faces_cfg = faces_cfg or FaceConfig()

    root = Path(folder)
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    _emit(progress, "collecting", 0, 0)
    items = collect_images(root)
    logger.info("scan_folder_start", folder=str(root), count=len(items))
    if not items:
        logger.warning("scan_folder_empty", folder=str(root))
    _emit(progress, "collecting", len(items), len(items))

    return scan_items(
        items, profile, cfg, faces_cfg, folder=str(root.resolve()), scan_id=scan_id, progress=progress
    )
