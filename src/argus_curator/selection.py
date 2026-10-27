"""Selection — score threshold + optional diversity cap.

Ported from imogen PR #14's ``_select_diverse`` / ``_decide_keep``. Nothing is
ever silently dropped: every excluded image carries a ``keep_reason`` so the
HITL surfaces can explain *why* it was left out.
"""

from __future__ import annotations

import re

import imagehash
from argus_cortex.moderation.nudenet import MIN_REPORTED_SCORE

from argus_curator.garments import outfit, outfit_similarity
from argus_curator.guard import rule_matches
from argus_curator.models import ExportRequest, FaceAttributes, ImageResult, ScanSummary
from argus_curator.nsfw import exposed_score

# A MiVOLO age under this, together with any reported exposed detection, is never kept.
MINOR_AGE_YEARS = 18.0

_MAX_DIST = 64.0  # maximum 64-bit pHash Hamming distance


def _scene_overlap(a: ImageResult, b: ImageResult) -> float:
    """Histogram intersection of two images' recorded scene probabilities (0..1).

    Two untagged images count as the same (unknown) scene, one untagged
    against a tagged one as different.
    """
    if a.scene is None or b.scene is None:
        return 1.0 if a.scene is None and b.scene is None else 0.0
    pb = {c.label: c.conf for c in b.scene.categories}
    return sum(min(c.conf, pb.get(c.label, 0.0)) for c in a.scene.categories)


def select_diverse(
    candidates: list[ImageResult],
    n: int,
    diversity_weight: float,
    *,
    scenes: bool = False,
    outfits: bool = False,
    outfit_conf: float = 0.5,
) -> list[ImageResult]:
    """Greedy diversity-aware selection.

    Each iteration picks the candidate maximising
    ``score_norm * (1 - dw) + diversity_norm * dw`` where ``diversity_norm`` is
    the minimum pHash Hamming distance to any already-selected image. ``dw=0``
    reduces to pure score ranking; ``dw=1`` maximises visual spread.

    ``scenes`` and ``outfits`` each add a novelty term — one minus the largest
    scene overlap / outfit similarity (garment types and colours, detections
    >= ``outfit_conf``) with anything selected — and the diversity term becomes
    the mean of pHash spread and the enabled novelties, so near-identical
    backdrops or outfits are spread out even when the pixels differ.

    A term, once in play, scores every candidate: an untagged image or one
    with nothing detected is itself a "scene" / "outfit" that repeats, rather
    than skipping the term and being judged on pHash alone (which would
    systematically favour exactly the images the term knows least about). A
    term is in play only if some candidate carries the data at all.
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
    sigs = [outfit(r.garments, outfit_conf) for r in candidates] if outfits else []
    use_scenes = scenes and any(r.scene is not None for r in candidates)
    use_outfits = outfits and any(sigs)
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
            if use_scenes and selected_idx:
                terms.append(1.0 - max(_scene_overlap(candidates[i], candidates[j]) for j in selected_idx))
            if use_outfits and selected_idx:
                terms.append(1.0 - max(outfit_similarity(sigs[i], sigs[j]) for j in selected_idx))
            div_norm = sum(terms) / len(terms)
            val = score_norm * (1.0 - diversity_weight) + div_norm * diversity_weight
            if val > best_val:
                best_val, best_idx = val, i
        if best_idx is None:
            break
        selected_idx.append(best_idx)
        if hashes[best_idx] is not None:
            selected_hashes.append(hashes[best_idx])
        remaining.remove(best_idx)

    return [candidates[i] for i in selected_idx]


def _mivolo_years(attrs: FaceAttributes | None) -> float | None:
    """MiVOLO's age in years, ignoring a FairFace bracket that has no year value."""
    if attrs is None or attrs.age_years is None:
        return None
    if attrs.age_source not in (None, "mivolo"):
        return None
    return attrs.age_years


def minor_exposed(result: ImageResult) -> bool:
    """True when any MiVOLO age is under 18 and an exposed class scored at least 0.2.

    0.2 is the lowest score NudeNet reports, so this is any detection of the
    default exposed classes. Covered parts, faces, belly, feet, and chest do
    not count. The check uses every face on the image, not only the primary.
    """
    if result.nsfw is None or exposed_score(result.nsfw.labels) < MIN_REPORTED_SCORE:
        return False
    years = [_mivolo_years(face.attributes) for face in result.faces]
    years.append(_mivolo_years(result.primary_face_attributes))
    return any(age is not None and age < MINOR_AGE_YEARS for age in years)


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


def _garment_exclusion(r: ImageResult, req: ExportRequest) -> str | None:
    """Keep-reason when *r* fails the Fashionpedia filters, else ``None``."""
    if not req.garments and not req.exclude_garments:
        return None
    if r.garments is None:
        return "garments-missing"
    worn = [g for g in r.garments if g.conf >= req.min_garment_conf]
    if req.exclude_garments:
        banned = set(req.exclude_garments)
        if any(g.label in banned for g in worn):
            return "garment-excluded"
    if req.garments:
        wanted = set(req.garments)
        hits = [g for g in worn if g.label in wanted]
        if not hits:
            # Distinguish "not there" from "there, but the model wasn't sure".
            doubtful = any(g.label in wanted for g in r.garments)
            return "garment-low-confidence" if doubtful else "garment-not-found"
        if req.garments_uncropped and all(g.touches_edge for g in hits):
            return "garment-cropped"
    return None


def _nsfw_exclusion(r: ImageResult, req: ExportRequest) -> str | None:
    """Keep-reason when *r* fails the NudeNet thresholds, else ``None``.

    Thresholds are inclusive (a score equal to one excludes). An image that was
    never checked, or whose check failed, is excluded rather than waved
    through: an NSFW filter that passes unknowns is not a filter.
    """
    if req.nsfw_threshold is None and not req.nsfw_label_thresholds:
        return None
    if r.nsfw is None:
        return "nsfw-unchecked"
    if req.nsfw_threshold is not None and exposed_score(r.nsfw.labels, req.nsfw_exposed_labels) >= req.nsfw_threshold:
        return "nsfw-exposed"
    for label, threshold in (req.nsfw_label_thresholds or {}).items():
        if r.nsfw.labels.get(label, 0.0) >= threshold:
            return f"nsfw-{label.lower().replace('_', '-')}"
    return None


def _guard_exclusion(r: ImageResult, req: ExportRequest) -> str | None:
    """Keep-reason when *r* fails the SingGuard filters, else ``None``.

    Unchecked images are excluded, as with NudeNet. With only ``guard_rules``,
    an unsafe image whose rules are all unlisted is kept; one that names no
    rule at all is excluded, since it cannot be shown to be unlisted.
    """
    if not req.guard_unsafe and not req.guard_rules:
        return None
    if r.guard is None:
        return "guard-unchecked"
    if r.guard.safe:
        return None
    if req.guard_unsafe:
        return "guard-unsafe"
    if not r.guard.rules:
        return "guard-rule-unknown"
    for wanted in req.guard_rules or []:
        if any(rule_matches(fired, wanted) for fired in r.guard.rules):
            return f"guard-rule-{re.sub(r'[^a-z0-9]+', '-', wanted.casefold()).strip('-')}"
    return None


def guard_filter_problem(req: ExportRequest, summary: ScanSummary) -> str | None:
    """Why *req*'s SingGuard filters cannot be applied to *summary*, else ``None``."""
    if (req.guard_unsafe or req.guard_rules) and not summary.guard_config.enabled:
        return (
            "SingGuard filters need a scan that ran SingGuard; this one did not "
            "(set ARGUS_CURATOR_SINGGUARD_URL and rescan)"
        )
    return None


def nsfw_filter_problem(req: ExportRequest, summary: ScanSummary) -> str | None:
    """Why *req*'s NSFW filters cannot be applied to *summary*, else ``None``."""
    uses_nsfw = req.nsfw_threshold is not None or bool(req.nsfw_label_thresholds)
    if uses_nsfw and not summary.nsfw_config.enabled:
        return (
            "NSFW thresholds need a scan that ran NudeNet; this one did not (set ARGUS_CURATOR_NUDENET_URL and rescan)"
        )
    return None


def garment_floor_problem(req: ExportRequest, recorded_from: float) -> str | None:
    """Why *req*'s garment floor cannot be honoured by a scan, else ``None``.

    A scan stores only detections >= its ``GarmentConfig.min_conf``; a lower
    export floor would silently behave as that one.
    """
    uses_garments = bool(req.garments or req.exclude_garments or req.diversify_outfits)
    if uses_garments and req.min_garment_conf < recorded_from:
        return (
            f"min_garment_conf {req.min_garment_conf} is below {recorded_from}, the probability this scan "
            "recorded garments from; rescan with a lower garments.min_conf"
        )
    return None


def decide_selection(
    results: list[ImageResult],
    req: ExportRequest,
    diversity_weight: float,
) -> tuple[list[ImageResult], dict[str, str]]:
    """Return ``(selected_results, keep_reason_by_rel)`` for an export request.

    Mirrors PR #14's ``_decide_keep``: applies score threshold, rejected/duplicate
    gates, optional face-cluster / pose / attribute / scene / garment / NSFW /
    SingGuard filters, then an optional diversity cap.

    An under-18 face with a reported exposed score is rejected first, whatever
    the request asked to keep.
    """
    keep_reason: dict[str, str] = {}
    face_filter = set(req.face_clusters) if req.face_clusters else None
    pose_filter = set(req.face_poses) if req.face_poses else None

    eligible: list[ImageResult] = []
    for r in results:
        if minor_exposed(r):
            keep_reason[r.rel_path] = "minor-exposed"
        elif r.score < req.min_score:
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
        elif (garment_reason := _garment_exclusion(r, req)) is not None:
            keep_reason[r.rel_path] = garment_reason
        elif (nsfw_reason := _nsfw_exclusion(r, req)) is not None:
            keep_reason[r.rel_path] = nsfw_reason
        elif (guard_reason := _guard_exclusion(r, req)) is not None:
            keep_reason[r.rel_path] = guard_reason
        else:
            eligible.append(r)

    chosen = eligible
    if req.max_keep is not None and len(eligible) > req.max_keep:
        chosen = select_diverse(
            eligible,
            req.max_keep,
            diversity_weight,
            scenes=req.diversify_scenes,
            outfits=req.diversify_outfits,
            outfit_conf=req.min_garment_conf,
        )
        chosen_set = {r.rel_path for r in chosen}
        for r in eligible:
            if r.rel_path not in chosen_set:
                keep_reason[r.rel_path] = "diversity-trimmed"

    for r in chosen:
        keep_reason[r.rel_path] = ""
    return chosen, keep_reason
