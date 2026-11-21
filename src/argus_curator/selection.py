"""Selection — filters, an optional diversity cap, a ranking and an optional split.

Ported from imogen PR #14's ``_select_diverse`` / ``_decide_keep``. Nothing is
ever silently dropped: every excluded image carries a ``keep_reason`` so the
HITL surfaces can explain *why* it was left out.

:func:`plan_selection` is the one code path ``POST /scan/{id}/select`` and
every export run, so a dry run and an export with the same fields cannot
disagree on what is kept, in what order, or in which part of a split.
"""

from __future__ import annotations

import os
import random
import re
from collections import Counter
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, get_args

import imagehash
import numpy as np
from argus_cortex.moderation import DEFAULT_EXPOSED_LABELS, rule_matches
from argus_cortex.moderation.nudenet import MIN_REPORTED_SCORE

from argus_curator.garments import outfit, outfit_similarity
from argus_curator.models import (
    ExportRequest,
    FaceAttributes,
    FaceDetection,
    FaceSourceFilter,
    GuardResult,
    ImageResult,
    RankBy,
    ScanSummary,
    SetAge,
    SplitConfig,
    SplitPart,
)
from argus_curator.nsfw import exposed_score, is_exposed

# A MiVOLO age under this, together with any reported exposed detection, is never kept.
MINOR_AGE_YEARS = 18.0

# Which under-18 rule hard_exclusion applies (see minor_exposed). "frame", the
# default and argus-darkroom's rule: a young age anywhere in the frame plus any
# exposure. "overlap": curator's older rule, which also places the exposure on a
# young person's box. Operator configuration only, never a request field.
MINOR_EXPOSED_ENV = "ARGUS_CURATOR_MINOR_EXPOSED"
MinorExposedMode = Literal["frame", "overlap"]
MINOR_EXPOSED_MODES: tuple[str, ...] = get_args(MinorExposedMode)
DEFAULT_MINOR_EXPOSED: MinorExposedMode = "frame"


class MinorExposedConfigError(ValueError):
    """``ARGUS_CURATOR_MINOR_EXPOSED`` (or ``--minor-exposed``) holds an unknown value."""


def check_minor_exposed_mode(value: str) -> MinorExposedMode:
    """*value* as a mode; raise :class:`MinorExposedConfigError` for anything else.

    Never falls back: a typo must not quietly pick a rule.
    """
    mode = value.strip().lower()
    if mode not in MINOR_EXPOSED_MODES:
        raise MinorExposedConfigError(
            f"{MINOR_EXPOSED_ENV} must be one of {', '.join(MINOR_EXPOSED_MODES)}, got {value!r}"
        )
    return mode  # type: ignore[return-value]


def minor_exposed_mode() -> MinorExposedMode:
    """The configured under-18 rule: ``ARGUS_CURATOR_MINOR_EXPOSED``, ``frame`` when unset or blank."""
    raw = os.environ.get(MINOR_EXPOSED_ENV, "").strip()
    return check_minor_exposed_mode(raw) if raw else DEFAULT_MINOR_EXPOSED


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


# An identity distance is a cosine distance, 0..2; a missing one ranks after any real one.
_NO_DISTANCE = 2.0


def _distance(r: ImageResult) -> float:
    face = primary_face(r)
    d = identity_distance(face) if face is not None else None
    return d if d is not None else _NO_DISTANCE


def rank_key(rank: RankBy) -> Callable[[ImageResult], tuple[float, float, str]]:
    """Sort key putting the best image first under *rank* (see ``models.RankBy``).

    ``"score"``: the curation score, then the identity distance. ``"identity"``:
    the identity distance (a face's ``reference_distance``, else its
    ``cluster_distance``; no face or no distance last), then the score. Both
    end on ``rel_path``, so the order never depends on scan order.
    """
    if rank == "identity":
        return lambda r: (_distance(r), -r.score, r.rel_path)
    return lambda r: (-r.score, _distance(r), r.rel_path)


def _quality(candidates: list[ImageResult], rank: RankBy) -> list[float]:
    """Each candidate's 0..1 merit under *rank*, the term diversity is blended against."""
    if rank == "identity":
        return [1.0 - _distance(r) / _NO_DISTANCE for r in candidates]
    max_score = max((r.score for r in candidates), default=1.0) or 1.0
    return [r.score / max_score for r in candidates]


def select_diverse(
    candidates: list[ImageResult],
    n: int,
    diversity_weight: float,
    *,
    scenes: bool = False,
    outfits: bool = False,
    outfit_conf: float = 0.5,
    semantic: bool = False,
    embeddings: Mapping[str, Any] | None = None,
    rank: RankBy = "score",
) -> list[ImageResult]:
    """Greedy diversity-aware selection.

    Each iteration picks the candidate maximising
    ``quality * (1 - dw) + diversity_norm * dw`` where ``diversity_norm`` is
    the minimum pHash Hamming distance to any already-selected image, and
    ``quality`` is the score over the best score (``rank="score"``) or one
    minus half the identity distance (``rank="identity"``). ``dw=0`` reduces
    to the best *n* by :func:`rank_key`; ``dw=1`` maximises visual spread.

    ``scenes`` and ``outfits`` each add a novelty term — one minus the largest
    scene overlap / outfit similarity (garment types and colours, detections
    >= ``outfit_conf``) with anything selected — and the diversity term becomes
    the mean of pHash spread and the enabled novelties, so near-identical
    backdrops or outfits are spread out even when the pixels differ.

    ``semantic`` adds a fourth: one minus the largest cosine similarity between
    the candidate's semantic embedding (*embeddings*, unit vectors by
    ``rel_path``) and any selected image's, so burst shots, re-encodes and
    small crops of one frame are spread out even when their pHashes are far
    apart. A selected image with no embedding is similar to nothing.

    A term, once in play, scores every candidate: an untagged image, one with
    nothing detected, or one with no embedding is itself a "scene" / "outfit" /
    embedding that repeats, rather
    than skipping the term and being judged on pHash alone (which would
    systematically favour exactly the images the term knows least about). A
    term is in play only if some candidate carries the data at all.
    """
    if n >= len(candidates) or diversity_weight <= 0.0:
        return sorted(candidates, key=rank_key(rank))[:n]

    quality = _quality(candidates, rank)
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
    vectors = [embeddings.get(r.rel_path) for r in candidates] if semantic and embeddings else []
    use_semantic = any(v is not None for v in vectors)
    if use_semantic:
        dim = next(len(v) for v in vectors if v is not None)
        has_vec = np.array([v is not None for v in vectors])
        matrix = np.stack(
            [np.asarray(v, dtype=np.float32) if v is not None else np.zeros(dim, np.float32) for v in vectors]
        )
        # Each candidate's largest cosine to anything selected so far, updated per pick.
        nearest = np.full(len(candidates), -1.0, dtype=np.float32)
    remaining = list(range(len(candidates)))

    for _ in range(n):
        if not remaining:
            break
        best_idx, best_val = None, -1.0
        for i in remaining:
            score_norm = quality[i]
            if not selected_idx:
                div_norm = 1.0
            elif hashes[i] is None:
                # No pHash: its spread is unknown, so it counts as a repeat, as an
                # untagged scene or outfit does, rather than as the widest spread.
                div_norm = 0.0
            elif not selected_hashes:
                div_norm = 1.0
            else:
                div_norm = min(hashes[i] - sh for sh in selected_hashes) / _MAX_DIST
            terms = [div_norm]
            if use_scenes and selected_idx:
                terms.append(1.0 - max(_scene_overlap(candidates[i], candidates[j]) for j in selected_idx))
            if use_outfits and selected_idx:
                terms.append(1.0 - max(outfit_similarity(sigs[i], sigs[j]) for j in selected_idx))
            if use_semantic and selected_idx:
                # No embedding: a repeat. Nothing embedded selected yet: as far as it gets.
                terms.append(min(1.0, max(0.0, 1.0 - float(nearest[i]))) if has_vec[i] else 0.0)
            div_norm = sum(terms) / len(terms)
            val = score_norm * (1.0 - diversity_weight) + div_norm * diversity_weight
            if val > best_val:
                best_val, best_idx = val, i
        if best_idx is None:
            break
        selected_idx.append(best_idx)
        if hashes[best_idx] is not None:
            selected_hashes.append(hashes[best_idx])
        if use_semantic and has_vec[best_idx]:
            np.maximum(nearest, matrix @ matrix[best_idx], out=nearest)
        remaining.remove(best_idx)

    return [candidates[i] for i in selected_idx]


def _mivolo_years(attrs: FaceAttributes | None) -> float | None:
    """MiVOLO's age in years, ignoring a FairFace bracket that has no year value."""
    if attrs is None or attrs.age_years is None:
        return None
    if attrs.age_source not in (None, "mivolo"):
        return None
    return attrs.age_years


def _young_mivolo(years: float | None) -> bool:
    return years is not None and years < MINOR_AGE_YEARS


def minor_exposed(result: ImageResult, mode: MinorExposedMode | None = None) -> bool:
    """True when a MiVOLO age under 18 shares the frame with an exposed reading.

    0.2 is the lowest score NudeNet reports. Covered parts, faces, belly, feet,
    and chest do not count. Ages come from every face and from unmatched people
    (body-only MiVOLO); a FairFace bracket is not an age.

    *mode* defaults to :func:`minor_exposed_mode`. Under ``frame`` (the
    default, argus-darkroom's rule) a young age and an exposure anywhere in the
    frame are enough. Under ``overlap``, with person boxes (else face boxes),
    an exposed detection flags the image when it overlaps a young box or
    overlaps no box at all; one that overlaps only adults does not. It fails
    closed: a young face with no person box, or a young age with no box to
    place it, flags any exposure.
    """
    if result.nsfw is None or not is_exposed(result):
        return False
    face_years = [_mivolo_years(face.attributes) for face in result.faces]
    years = [*face_years, _mivolo_years(result.primary_face_attributes)]
    years.extend(person.age_years for person in result.people)
    if not any(_young_mivolo(age) for age in years):
        return False
    # Resolved only for a young, exposed frame: the per-image selection loops
    # do not read the environment for every other row.
    mode = minor_exposed_mode() if mode is None else check_minor_exposed_mode(mode)
    if mode == "frame":
        return True
    if result.people:
        if any(
            _young_mivolo(age) and face.person_index is None for face, age in zip(result.faces, face_years, strict=True)
        ):
            return True  # a young face whose body was not found: any exposure may be theirs
        boxes = [p.bbox for p in result.people]
        # A person is young by their own body age or by a young face matched to them:
        # when the two disagree, the younger reading wins.
        young_idx = {i for i, p in enumerate(result.people) if _young_mivolo(p.age_years)}
        young_idx |= {
            face.person_index
            for face, age in zip(result.faces, face_years, strict=True)
            if _young_mivolo(age) and face.person_index is not None
        }
        young = [p.bbox for i, p in enumerate(result.people) if i in young_idx]
    else:
        boxes = [f.bbox for f in result.faces]
        young = [f.bbox for f, age in zip(result.faces, face_years, strict=True) if _young_mivolo(age)]
    exposed = [d for d in result.nsfw.detections if d.label in DEFAULT_EXPOSED_LABELS and d.score >= MIN_REPORTED_SCORE]
    if not young or not exposed:
        return True  # the young age (or the exposure) has no box to compare
    from argus_curator.bodies import overlaps

    return any(any(overlaps(d.bbox, b) for b in young) or not any(overlaps(d.bbox, b) for b in boxes) for d in exposed)


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
    # NudeNet never saw it; EraX alone under the thresholds is not a pass.
    return "nsfw-unchecked" if r.nudenet_failed else None


def _rating_exclusion(r: ImageResult, req: ExportRequest) -> str | None:
    """Keep-reason when *r*'s content rating is not one *req* keeps, else ``None``.

    Like the NudeNet thresholds, an image NudeNet never checked is excluded.
    """
    if req.content_ratings is None:
        return None
    if r.content_rating is None:
        return "nsfw-unchecked"
    if r.content_rating not in req.content_ratings:
        return f"content-rating:{r.content_rating}"
    return None


def _verdict_exclusion(verdict: GuardResult | None, req: ExportRequest) -> str | None:
    """Keep-reason when one verdict fails the SingGuard filters, else ``None``.

    Unchecked images are excluded, as with NudeNet. With only ``guard_rules``,
    an unsafe image whose rules are all unlisted is kept; one that names no
    rule at all is excluded, since it cannot be shown to be unlisted.
    """
    if verdict is None:
        return "guard-unchecked"
    if verdict.safe:
        return None
    if req.guard_unsafe:
        return "guard-unsafe"
    if not verdict.rules:
        return "guard-rule-unknown"
    for wanted in req.guard_rules or []:
        if any(rule_matches(fired, wanted) for fired in verdict.rules):
            return f"guard-rule-{re.sub(r'[^a-z0-9]+', '-', wanted.casefold()).strip('-')}"
    return None


def _guard_exclusion(r: ImageResult, req: ExportRequest) -> str | None:
    """Keep-reason when *r* fails the SingGuard filters, else ``None``.

    With ``guard_policies`` every listed verdict is tested, and the first that
    fails names its policy (``guard-unsafe:adult-granular``). A policy the scan
    ran as a variant but whose check failed for *r* is unchecked; it never
    falls back to the official verdict. :func:`guard_filter_problem` refuses
    names the scan did not run.
    """
    if not req.guard_unsafe and not req.guard_rules:
        return None
    if not req.guard_policies:
        return _verdict_exclusion(r.guard, req)
    for name in req.guard_policies:
        # A scan without variants has only the official verdict.
        verdict = r.guard_variants.get(name) if r.guard_variants is not None else r.guard
        if (reason := _verdict_exclusion(verdict, req)) is not None:
            return f"{reason}:{name}"
    return None


def guard_filter_problem(req: ExportRequest, summary: ScanSummary) -> str | None:
    """Why *req*'s SingGuard filters cannot be applied to *summary*, else ``None``."""
    # guard_policies alone counts too: naming the default policy must not pass on a scan with no verdicts.
    if (req.guard_unsafe or req.guard_rules or req.guard_policies) and not summary.guard_config.enabled:
        return (
            "SingGuard filters need a scan that ran SingGuard; this one did not "
            "(set ARGUS_CURATOR_SINGGUARD_URL and rescan)"
        )
    ran = [name for name in (summary.guard_config.policy, *summary.guard_config.variants) if name]
    missing = [name for name in req.guard_policies or [] if name not in ran]
    if missing:
        return (
            f"guard_policies {', '.join(missing)}: this scan judged only {', '.join(ran)} (rescan with guard.variants)"
        )
    return None


def face_source_problem(req: ExportRequest, summary: ScanSummary) -> str | None:
    """Why *req*'s face-source preset cannot be applied to *summary*, else ``None``.

    Without these the preset would keep nothing, every image failing on a
    measurement the scan never took.
    """
    fs = req.face_source
    if fs is None:
        return None
    if not summary.faces_config.enabled:
        return "the face-source preset needs a scan that ran face detection; rescan with faces on"
    if fs.require_adult and not summary.faces_config.mivolo:
        return (
            "the face-source preset's adult check needs MiVOLO ages; rescan with faces.mivolo, "
            "or set face_source.require_adult to false"
        )
    if fs.cluster is not None and fs.cluster not in {c.cluster_id for c in summary.face_clusters}:
        return f"face_source.cluster {fs.cluster!r} is not an identity in this scan"
    # reference_skipped is set only when the scan asked for a reference (its
    # reference_embedding, never stored, cannot be read back here).
    if fs.max_identity_distance is not None and summary.reference_skipped:
        # The identity check would fall back to the cluster distance: another
        # person's identity could pass as the reference's.
        return (
            f"the face-source identity check needs the scan's reference, which gave no distances "
            f"({summary.reference_skipped}); rescan with a usable reference, "
            "or set face_source.max_identity_distance to null"
        )
    return None


def nsfw_filter_problem(req: ExportRequest, summary: ScanSummary) -> str | None:
    """Why *req*'s NSFW filters cannot be applied to *summary*, else ``None``."""
    uses_nsfw = req.nsfw_threshold is not None or bool(req.nsfw_label_thresholds) or req.content_ratings is not None
    if uses_nsfw and not summary.nsfw_config.enabled:
        return (
            "NSFW thresholds and content ratings need a scan that ran NudeNet; this one did not "
            "(set ARGUS_CURATOR_NUDENET_URL and rescan)"
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


def rank_split_problem(req: ExportRequest, summary: ScanSummary) -> str | None:
    """Why *req*'s ranking or split cannot be applied to *summary*, else ``None``."""
    faces = summary.faces_config.enabled
    if req.rank == "identity" and not faces:
        return 'rank "identity" needs a scan that ran face detection; rescan with faces on'
    if req.split is not None and req.split.group_by == "face_cluster" and not faces:
        return 'split.group_by "face_cluster" needs a scan that ran face detection; rescan with faces on'
    return None


def semantic_problem(req: ExportRequest, summary: ScanSummary) -> str | None:
    """Why ``req.diversify_semantic`` cannot be applied to *summary*, else ``None``.

    Embeddings live in memory only, so a scan that ran them loses them when
    the server restarts; the term would then silently do nothing.
    """
    if not req.diversify_semantic:
        return None
    if not summary.embeddings_config.enabled:
        return "diversify_semantic needs a scan that computed embeddings; rescan with embeddings on"
    if not summary.embeddings:
        return (
            "diversify_semantic needs this scan's embeddings, which are held in memory only and are gone "
            "(the server restarted since the scan, or no image could be embedded); rescan with embeddings on"
        )
    return None


def selection_problem(req: ExportRequest, summary: ScanSummary) -> str | None:
    """The first reason *req*'s selection cannot be applied to *summary*, else ``None``.

    Every check select and export share, so the two routes refuse the same requests.
    """
    for problem in (
        garment_floor_problem(req, summary.garments_config.min_conf),
        nsfw_filter_problem(req, summary),
        face_source_problem(req, summary),
        guard_filter_problem(req, summary),
        rank_split_problem(req, summary),
        semantic_problem(req, summary),
    ):
        if problem is not None:
            return problem
    return None


def primary_face(result: ImageResult) -> FaceDetection | None:
    return next((f for f in result.faces if f.primary), None)


def identity_distance(face: FaceDetection) -> float | None:
    """The reference distance when the scan had a reference, else the distance to the face's identity."""
    return face.reference_distance if face.reference_distance is not None else face.cluster_distance


def _over(value: float | None, limit: float | None, *, absolute: bool = False) -> bool:
    """True when a limit is set and *value* is missing or above it."""
    if limit is None:
        return False
    if value is None:
        return True
    return (abs(value) if absolute else value) > limit


def _under(value: float | None, floor: float | None) -> bool:
    """True when a floor is set and *value* is missing or below it."""
    return floor is not None and (value is None or value < floor)


def _face_source_exclusion(
    r: ImageResult, fs: FaceSourceFilter, cluster: str | None, set_age: SetAge | None
) -> str | None:
    """Why the face-source preset keeps *r* out (``face-source:<check>``), or None.

    A measurement the scan does not have fails its check: a face source is
    better missed than taken on trust.
    """
    face = primary_face(r)
    if face is None:
        return "face-source:no-face"
    if fs.single_face and r.face_count != 1:
        return "face-source:faces"
    if cluster is not None and r.primary_face_cluster != cluster:
        return "face-source:identity"
    if _over(face.yaw, fs.max_yaw, absolute=True):
        return "face-source:yaw"
    if _over(face.pitch, fs.max_pitch, absolute=True):
        return "face-source:pitch"
    if _over(face.roll, fs.max_roll, absolute=True):
        return "face-source:roll"
    if _under(min(face.bbox[2], face.bbox[3]), fs.min_face_px):
        return "face-source:face-size"
    if _under(face.sharpness, fs.min_face_sharpness):
        return "face-source:face-sharpness"
    if _under(face.eye_open, fs.min_eye_open):
        return "face-source:eyes"
    if _over(face.mouth_open, fs.max_mouth_open):
        return "face-source:mouth"
    if _over(identity_distance(face), fs.max_identity_distance):
        return "face-source:identity-distance"
    if fs.require_adult:
        years = _mivolo_years(face.attributes)
        if years is None:
            return "face-source:no-age"
        if years < MINOR_AGE_YEARS or (set_age is not None and set_age.years < MINOR_AGE_YEARS):
            return "face-source:age"
    return None


def hard_exclusion(result: ImageResult) -> str | None:
    """Why no request can keep *result* (a named selection included), or ``None``.

    An under-18 age with a reported exposed score first (under the configured
    :func:`minor_exposed_mode`; no request field changes it), then any rejected
    image, by its own ``reject_reason``. A rejected frame gets no age reading,
    so the under-18 rule cannot speak for it; that covers a prefilter drop and
    a crop that could not be applied too.
    """
    if minor_exposed(result):
        return "minor-exposed"
    if not result.passed:
        return result.reject_reason or "rejected"
    return None


class SplitError(ValueError):
    """A requested split cannot be made from the kept images (too few, or one group)."""


# Group keys a split can keep together (models.SplitGroup). A key of None makes
# the item a group of its own. Another kind of item reuses split_held_out with
# its own key (#187: a speaker), so this table is the only image-specific part.
SPLIT_GROUP_KEYS: dict[str, Callable[[ImageResult], str | None]] = {
    "image": lambda r: r.rel_path,
    "face_cluster": lambda r: r.primary_face_cluster,
}


def split_held_out(keys: Sequence[str | None], cfg: SplitConfig) -> list[SplitPart]:
    """Each ranked item's part of a train/held-out split, from its group key.

    *keys* holds one group key per item, best item first; items sharing a key
    always land in the same part, and a ``None`` key is a group of its own.
    Groups are ordered by their best item. ``cfg.method`` "spread" takes
    held-out groups evenly through that order (with one item per group,
    exactly argus-mint's ``int((j + 0.5) * n / k)`` picks); "random" shuffles
    the groups with ``random.Random(cfg.seed)``. Groups are then taken in that
    order, skipping any that would push the held-out part past its target.

    Pure and deterministic. Raises :class:`SplitError` when either part would
    be empty.
    """
    n = len(keys)
    target = cfg.held_out if cfg.held_out is not None else max(1, round((cfg.held_out_fraction or 0.0) * n))
    if target >= n:
        raise SplitError(f"{n} kept image(s): holding out {target} leaves none to train")
    members: dict[Hashable, list[int]] = {}
    for i, key in enumerate(keys):
        members.setdefault(key if key is not None else ("ungrouped", i), []).append(i)
    groups = list(members.values())
    m = len(groups)
    if m < 2:
        raise SplitError(f"all {n} kept image(s) share one {cfg.group_by}; a split needs at least two groups")
    if cfg.method == "random":
        order = list(range(m))
        random.Random(cfg.seed).shuffle(order)
    else:
        k = min(m - 1, max(1, round(target * m / n)))
        picks = list(dict.fromkeys(int((j + 0.5) * m / k) for j in range(k)))
        picked = set(picks)
        order = picks + [g for g in range(m) if g not in picked]
    held: set[int] = set()
    count = 0
    for g in order:
        if count >= target:
            break
        if count + len(groups[g]) <= target:
            held.update(groups[g])
            count += len(groups[g])
    if not held:
        smallest = min(len(g) for g in groups)
        raise SplitError(f"no {cfg.group_by} group fits in {target} held-out image(s); the smallest holds {smallest}")
    return ["held_out" if i in held else "train" for i in range(n)]


@dataclass(frozen=True)
class SelectionPlan:
    """What a selection keeps: ``chosen`` best first, why the rest are out, and the split.

    ``split`` maps each chosen ``rel_path`` to its part; it is empty without a split.
    """

    chosen: list[ImageResult]
    keep_reason: dict[str, str]
    split: dict[str, SplitPart] = field(default_factory=dict)


def plan_selection(
    results: list[ImageResult],
    req: ExportRequest,
    diversity_weight: float,
    *,
    set_age: SetAge | None = None,
    embeddings: Mapping[str, Any] | None = None,
) -> SelectionPlan:
    """The filter, cap, ranking and split for *req*: the one path select and export share.

    *diversity_weight* is the scan's; ``req.diversity_weight`` overrides it.
    *embeddings* are the scan's in-memory ones (``ScanSummary.embeddings``),
    read by ``req.diversify_semantic``. Raises :class:`SplitError` when
    ``req.split`` cannot be made.
    """
    chosen, keep_reason = decide_selection(results, req, diversity_weight, set_age=set_age, embeddings=embeddings)
    split: dict[str, SplitPart] = {}
    if req.split is not None:
        key = SPLIT_GROUP_KEYS[req.split.group_by]
        parts = split_held_out([key(r) for r in chosen], req.split)
        split = {r.rel_path: part for r, part in zip(chosen, parts, strict=True)}
    return SelectionPlan(chosen=chosen, keep_reason=keep_reason, split=split)


def decide_selection(
    results: list[ImageResult],
    req: ExportRequest,
    diversity_weight: float,
    *,
    set_age: SetAge | None = None,
    embeddings: Mapping[str, Any] | None = None,
) -> tuple[list[ImageResult], dict[str, str]]:
    """Return ``(selected_results, keep_reason_by_rel)`` for an export request.

    Mirrors PR #14's ``_decide_keep``: applies score threshold, rejected/duplicate
    gates, optional face-cluster / pose / attribute / scene / garment / NSFW /
    SingGuard filters, then an optional diversity cap (``req.max_keep``, blended
    by ``req.diversity_weight`` or else *diversity_weight*).

    An under-18 face with a reported exposed score is rejected first, whatever
    the request asked to keep. Any rejected image comes next: it is never kept,
    and its keep reason is its own ``reject_reason`` (``prefilter:nsfw`` and so on).

    With ``req.face_source`` the preset runs last (*set_age* is the scan's, for
    its adult check). The selection always comes back ranked best first by
    :func:`rank_key` under ``req.rank``. Use :func:`plan_selection` for the split too.
    """
    keep_reason: dict[str, str] = {}
    face_filter = set(req.face_clusters) if req.face_clusters else None
    pose_filter = set(req.face_poses) if req.face_poses else None
    fs = req.face_source
    fs_cluster = None
    if fs is not None:
        fs_cluster = fs.cluster
        # With a reference the reference distance names the identity; the
        # largest identity need not be the reference's, so it is no default.
        referenced = any(f.reference_distance is not None for r in results for f in r.faces)
        if fs_cluster is None and not referenced:
            counts = Counter(r.primary_face_cluster for r in results if r.passed and r.primary_face_cluster)
            fs_cluster = counts.most_common(1)[0][0] if counts else None

    eligible: list[ImageResult] = []
    for r in results:
        if (hard := hard_exclusion(r)) is not None:
            keep_reason[r.rel_path] = hard
        elif r.score < req.min_score:
            keep_reason[r.rel_path] = "below-score"
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
        elif (rating_reason := _rating_exclusion(r, req)) is not None:
            keep_reason[r.rel_path] = rating_reason
        elif (guard_reason := _guard_exclusion(r, req)) is not None:
            keep_reason[r.rel_path] = guard_reason
        elif fs is not None and (fs_reason := _face_source_exclusion(r, fs, fs_cluster, set_age)) is not None:
            keep_reason[r.rel_path] = fs_reason
        else:
            eligible.append(r)

    chosen = eligible
    if req.max_keep is not None and len(eligible) > req.max_keep:
        chosen = select_diverse(
            eligible,
            req.max_keep,
            req.diversity_weight if req.diversity_weight is not None else diversity_weight,
            scenes=req.diversify_scenes,
            outfits=req.diversify_outfits,
            outfit_conf=req.min_garment_conf,
            semantic=req.diversify_semantic,
            embeddings=embeddings,
            rank=req.rank,
        )
        chosen_set = {r.rel_path for r in chosen}
        for r in eligible:
            if r.rel_path not in chosen_set:
                keep_reason[r.rel_path] = "diversity-trimmed"

    chosen = sorted(chosen, key=rank_key(req.rank))
    for r in chosen:
        keep_reason[r.rel_path] = ""
    return chosen, keep_reason
