"""Selection — score threshold + optional diversity cap.

Ported from imogen PR #14's ``_select_diverse`` / ``_decide_keep``. Nothing is
ever silently dropped: every excluded image carries a ``keep_reason`` so the
HITL surfaces can explain *why* it was left out.
"""

from __future__ import annotations

import imagehash

from argus_curator.models import ExportRequest, ImageResult

_MAX_DIST = 64.0  # maximum 64-bit pHash Hamming distance


def _scene_overlap(a: ImageResult, b: ImageResult) -> float:
    """Histogram intersection of two images' recorded scene probabilities (0..1)."""
    if a.scene is None or b.scene is None:
        return 0.0
    pb = {c.label: c.conf for c in b.scene.categories}
    return sum(min(c.conf, pb.get(c.label, 0.0)) for c in a.scene.categories)


def select_diverse(
    candidates: list[ImageResult],
    n: int,
    diversity_weight: float,
    *,
    scenes: bool = False,
) -> list[ImageResult]:
    """Greedy diversity-aware selection.

    Each iteration picks the candidate maximising
    ``score_norm * (1 - dw) + diversity_norm * dw`` where ``diversity_norm`` is
    the minimum pHash Hamming distance to any already-selected image. ``dw=0``
    reduces to pure score ranking; ``dw=1`` maximises visual spread.

    ``scenes`` adds a novelty term — one minus the largest scene overlap with
    anything selected — and the diversity term becomes the mean of pHash
    spread and that novelty, so near-identical backdrops are spread out even
    when the pixels differ. An image without scene tags contributes no term.
    """
    if n >= len(candidates) or diversity_weight <= 0.0:
        return sorted(candidates, key=lambda r: -r.score)[:n]

    max_score = max((r.score for r in candidates), default=1.0) or 1.0
    hashes: list[imagehash.ImageHash | None] = []
    for r in candidates:
        try:
            hashes.append(imagehash.hex_to_hash(r.phash) if r.phash else None)
        except Exception:
            hashes.append(None)

    selected_idx: list[int] = []
    selected_hashes: list[imagehash.ImageHash] = []
    selected_scenes: list[ImageResult] = []
    remaining = list(range(len(candidates)))

    for _ in range(n):
        if not remaining:
            break
        best_idx, best_val = None, -1.0
        for i in remaining:
            score_norm = candidates[i].score / max_score
            if not selected_hashes or hashes[i] is None:
                div_norm = 1.0
            else:
                div_norm = min(hashes[i] - sh for sh in selected_hashes) / _MAX_DIST
            terms = [div_norm]
            if scenes and candidates[i].scene is not None and selected_scenes:
                terms.append(1.0 - max(_scene_overlap(candidates[i], o) for o in selected_scenes))
            div_norm = sum(terms) / len(terms)
            val = score_norm * (1.0 - diversity_weight) + div_norm * diversity_weight
            if val > best_val:
                best_val, best_idx = val, i
        if best_idx is None:
            break
        selected_idx.append(best_idx)
        if hashes[best_idx] is not None:
            selected_hashes.append(hashes[best_idx])
        if candidates[best_idx].scene is not None:
            selected_scenes.append(candidates[best_idx])
        remaining.remove(best_idx)

    return [candidates[i] for i in selected_idx]


def _attribute_exclusion(r: ImageResult, req: ExportRequest) -> str | None:
    """Keep-reason when *r* fails the face-attribute filters, else ``None``.

    A filtered attribute matches only when its label is allowed *and* its
    confidence reaches ``min_attribute_conf``; the two failures get distinct
    reasons so a reviewer can tell "different" from "the model wasn't sure".
    A confidence of ``None`` (a MiVOLO regression age) is not held to the floor.
    """
    filters = (("race", req.face_races), ("gender", req.face_genders), ("age", req.face_ages))
    active = [(name, set(allowed)) for name, allowed in filters if allowed]
    age_range = req.min_age_years is not None or req.max_age_years is not None
    if not active and not age_range:
        return None
    attrs = r.primary_face_attributes
    if attrs is None:
        return "face-attributes-missing"
    for name, allowed in active:
        value = getattr(attrs, name)
        if value is None:
            return f"face-{name}-missing"
        if value not in allowed:
            return f"face-{name}-excluded"
        conf = getattr(attrs, f"{name}_conf")
        if conf is not None and conf < req.min_attribute_conf:
            return f"face-{name}-low-confidence"
    if age_range:
        years = attrs.age_years
        if years is None:
            return "face-age-years-missing"
        if (req.min_age_years is not None and years < req.min_age_years) or (
            req.max_age_years is not None and years > req.max_age_years
        ):
            return "face-age-out-of-range"
    return None


def _scene_exclusion(r: ImageResult, req: ExportRequest) -> str | None:
    """Keep-reason when *r* fails the Places365 filters, else ``None``."""
    if not req.scenes and not req.scene_environments:
        return None
    if r.scene is None:
        return "scene-missing"
    if req.scene_environments:
        if r.scene.environment not in req.scene_environments:
            return "scene-environment-excluded"
        if r.scene.environment_conf < req.min_scene_conf:
            return "scene-environment-low-confidence"
    if req.scenes:
        wanted = set(req.scenes)
        hits = [c for c in r.scene.categories if c.label in wanted]
        if not hits:
            return "scene-excluded"
        if max(c.conf for c in hits) < req.min_scene_conf:
            return "scene-low-confidence"
    return None


def decide_selection(
    results: list[ImageResult],
    req: ExportRequest,
    diversity_weight: float,
) -> tuple[list[ImageResult], dict[str, str]]:
    """Return ``(selected_results, keep_reason_by_rel)`` for an export request.

    Mirrors PR #14's ``_decide_keep``: applies score threshold, rejected/duplicate
    gates, optional face-cluster / pose / attribute / scene filters, then an
    optional diversity cap.
    """
    keep_reason: dict[str, str] = {}
    face_filter = set(req.face_clusters) if req.face_clusters else None
    pose_filter = set(req.face_poses) if req.face_poses else None

    eligible: list[ImageResult] = []
    for r in results:
        if r.score < req.min_score:
            keep_reason[r.rel_path] = "below-score"
        elif not r.passed and not req.include_rejected:
            keep_reason[r.rel_path] = r.reject_reason or "rejected"
        elif r.is_duplicate and not req.keep_similar:
            keep_reason[r.rel_path] = f"similar-to:{r.duplicate_of}"
        elif face_filter is not None and r.primary_face_cluster not in face_filter:
            keep_reason[r.rel_path] = "face-cluster-excluded"
        elif pose_filter is not None and r.primary_face_pose not in pose_filter:
            keep_reason[r.rel_path] = "face-pose-excluded"
        elif (attr_reason := _attribute_exclusion(r, req)) is not None:
            keep_reason[r.rel_path] = attr_reason
        elif (scene_reason := _scene_exclusion(r, req)) is not None:
            keep_reason[r.rel_path] = scene_reason
        else:
            eligible.append(r)

    chosen = eligible
    if req.max_keep is not None and len(eligible) > req.max_keep:
        chosen = select_diverse(
            eligible,
            req.max_keep,
            diversity_weight,
            scenes=req.diversify_scenes,
        )
        chosen_set = {r.rel_path for r in chosen}
        for r in eligible:
            if r.rel_path not in chosen_set:
                keep_reason[r.rel_path] = "diversity-trimmed"

    for r in chosen:
        keep_reason[r.rel_path] = ""
    return chosen, keep_reason
