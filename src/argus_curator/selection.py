"""Selection — score threshold + optional diversity cap.

Ported from imogen PR #14's ``_select_diverse`` / ``_decide_keep``. Nothing is
ever silently dropped: every excluded image carries a ``keep_reason`` so the
HITL surfaces can explain *why* it was left out.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from typing import Literal, get_args

import imagehash
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
    ScanSummary,
    SetAge,
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
    mode = minor_exposed_mode() if mode is None else check_minor_exposed_mode(mode)
    if result.nsfw is None or not is_exposed(result):
        return False
    face_years = [_mivolo_years(face.attributes) for face in result.faces]
    years = [*face_years, _mivolo_years(result.primary_face_attributes)]
    years.extend(person.age_years for person in result.people)
    if not any(_young_mivolo(age) for age in years):
        return False
    if mode == "frame":
        return True
    if result.people:
        if any(
            _young_mivolo(age) and face.person_index is None for face, age in zip(result.faces, face_years, strict=True)
        ):
            return True  # a young face whose body was not found: any exposure may be theirs
        boxes = [p.bbox for p in result.people]
        young = [p.bbox for p in result.people if _young_mivolo(p.age_years)]
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
    if (req.guard_unsafe or req.guard_rules) and not summary.guard_config.enabled:
        return (
            "SingGuard filters need a scan that ran SingGuard; this one did not "
            "(set ARGUS_CURATOR_SINGGUARD_URL and rescan)"
        )
    ran = [summary.guard_config.policy, *summary.guard_config.variants]
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
    if fs.max_identity_distance is not None and summary.faces_config.reference_paths and summary.reference_skipped:
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


def _face_source_rank(r: ImageResult) -> tuple[float, float]:
    """Best first: the curation score, then the closest identity match."""
    face = primary_face(r)
    distance = identity_distance(face) if face is not None else None
    return (-r.score, distance if distance is not None else 2.0)


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


def decide_selection(
    results: list[ImageResult],
    req: ExportRequest,
    diversity_weight: float,
    *,
    set_age: SetAge | None = None,
) -> tuple[list[ImageResult], dict[str, str]]:
    """Return ``(selected_results, keep_reason_by_rel)`` for an export request.

    Mirrors PR #14's ``_decide_keep``: applies score threshold, rejected/duplicate
    gates, optional face-cluster / pose / attribute / scene / garment / NSFW /
    SingGuard filters, then an optional diversity cap.

    An under-18 face with a reported exposed score is rejected first, whatever
    the request asked to keep. Any rejected image comes next: it is never kept,
    and its keep reason is its own ``reject_reason`` (``prefilter:nsfw`` and so on).

    With ``req.face_source`` the preset runs last (*set_age* is the scan's, for
    its adult check) and the selection comes back ranked best first.
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
            diversity_weight,
            scenes=req.diversify_scenes,
            outfits=req.diversify_outfits,
            outfit_conf=req.min_garment_conf,
        )
        chosen_set = {r.rel_path for r in chosen}
        for r in eligible:
            if r.rel_path not in chosen_set:
                keep_reason[r.rel_path] = "diversity-trimmed"

    if fs is not None:
        chosen = sorted(chosen, key=_face_source_rank)
    for r in chosen:
        keep_reason[r.rel_path] = ""
    return chosen, keep_reason
