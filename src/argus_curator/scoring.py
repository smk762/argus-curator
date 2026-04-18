"""Composite quality scoring with per-objective face/orientation bonuses."""

from __future__ import annotations

from argus_curator.types import CurateConfig, ImageResult


# ---------------------------------------------------------------------------
# Face-count penalty per objective
# ---------------------------------------------------------------------------

def _face_penalty(face_count: int | None, objective: str) -> float:
    if face_count is None:
        return 1.0
    if objective == "identity":
        if face_count == 0:
            return 0.50
        if face_count >= 2:
            return 0.40
        return 1.0
    if objective == "wardrobe":
        if face_count == 0:
            return 0.75
        if face_count >= 2:
            return 0.60
        return 1.0
    if objective == "style":
        return 1.0  # style doesn't care about faces
    if face_count >= 2:
        return 0.85
    return 1.0


# ---------------------------------------------------------------------------
# Orientation bonus per objective
# ---------------------------------------------------------------------------

def _orientation_bonus(result: ImageResult, objective: str) -> float:
    orientation = result.width / max(result.height, 1)
    bonus = 0.0
    if objective == "identity":
        if result.face_count == 1:
            bonus += 0.08
        if result.person_detected:
            bonus += 0.03
        if 0.7 <= orientation <= 1.35:
            bonus += 0.03
    elif objective == "wardrobe":
        if result.person_detected:
            bonus += 0.06
        if orientation <= 0.8:
            bonus += 0.08
        elif orientation <= 1.0:
            bonus += 0.04
        if result.face_count == 1:
            bonus += 0.02
    elif objective == "style":
        # wide landscape compositions are common in style datasets
        if orientation >= 1.3:
            bonus += 0.05
    elif objective == "concept":
        # square / slight-portrait compositions are good for object focus
        if 0.85 <= orientation <= 1.15:
            bonus += 0.04
    return min(0.15, bonus)


# ---------------------------------------------------------------------------
# Composite score
# ---------------------------------------------------------------------------

def compute_score(
    result: ImageResult,
    cfg: CurateConfig,
    *,
    aesthetic_score: float = 0.0,
    has_detectors: bool = False,
) -> float:
    """Compute and attach the composite quality score to *result* in-place.

    Returns the final score.
    """
    s = cfg.scoring
    sharp_norm = min(1.0, result.sharpness / s.sharpness_ref) if s.sharpness_ref else 0.0
    res_norm = min(1.0, result.short_side / s.resolution_ref) if s.resolution_ref else 0.0

    face_penalty = _face_penalty(result.face_count, cfg.objective)
    subject_raw = result.subject_score * face_penalty

    total_w = (
        s.weight_sharpness
        + s.weight_resolution
        + s.weight_artifact
        + s.weight_aesthetic
        + (s.weight_subject if has_detectors else 0.0)
    ) or 1.0

    base = (
        s.weight_sharpness * sharp_norm
        + s.weight_resolution * res_norm
        + s.weight_artifact * result.artifact_score
        + s.weight_aesthetic * aesthetic_score
        + (s.weight_subject * subject_raw if has_detectors else 0.0)
    ) / total_w

    bonus = _orientation_bonus(result, cfg.objective)
    score = round(min(1.0, base + bonus), 4)

    result.sharpness_score = round(s.weight_sharpness * sharp_norm / total_w, 4)
    result.resolution_score = round(s.weight_resolution * res_norm / total_w, 4)
    result.artifact_sub = round(s.weight_artifact * result.artifact_score / total_w, 4)
    result.aesthetic_score = round(aesthetic_score, 4)
    result.subject_score = round(subject_raw, 4)
    result.score = score
    result.score_breakdown = {
        "sharpness": result.sharpness_score,
        "resolution": result.resolution_score,
        "artifact": result.artifact_sub,
        "aesthetic": round(s.weight_aesthetic * aesthetic_score / total_w, 4),
        "orientation_bonus": bonus,
        **({"subject": round(s.weight_subject * subject_raw / total_w, 4)} if has_detectors else {}),
    }
    return score
