"""Scan prefilter: opt-in drops before the late phases (see :class:`PrefilterConfig`).

Applied once, right after the identities phase (so a set age counts), to the
images that still pass. A dropped image becomes a reject with a
``prefilter:<rule>`` reason and stays in the results; scenes and garments
skip it like any reject, and SingGuard skips it too (other rejects it still
judges), which is where the time goes. Unlike other rejects, a dropped image
is never kept or exported, ``include_rejected`` or a named selection
notwithstanding (see :func:`selection.decide_selection`).
"""

from __future__ import annotations

from argus_cortex.moderation.nudenet import MIN_REPORTED_SCORE

from argus_curator.guard_context import body_only_ages, mivolo_ages
from argus_curator.models import FaceConfig, ImageResult, PrefilterConfig
from argus_curator.nsfw import exposed_reading

PREFIX = "prefilter:"
NSFW = f"{PREFIX}nsfw"
NO_FACE = f"{PREFIX}no-face"
NO_AGE = f"{PREFIX}no-age"


def is_prefiltered(result: ImageResult) -> bool:
    """Whether the prefilter dropped *result* (its ``reject_reason`` says so)."""
    return not result.passed and (result.reject_reason or "").startswith(PREFIX)


def has_age(result: ImageResult) -> bool:
    """A MiVOLO age on a face, a body-only age, or the set's age."""
    return bool(mivolo_ages(result) or body_only_ages(result)) or result.set_age is not None


def reason(result: ImageResult, cfg: PrefilterConfig) -> str | None:
    """The first rule *cfg* turns on that drops *result*, in nsfw, no-face, no-age order."""
    if cfg.drop_nsfw:
        reading = exposed_reading(result)
        # An image NudeNet and EraX could not score is kept: no reading, no drop.
        if reading is not None and reading >= MIN_REPORTED_SCORE:
            return NSFW
    # A failed detection is unchecked, not faceless (see ImageResult.face_error).
    if cfg.drop_no_face and result.face_count == 0 and result.face_error is None:
        return NO_FACE
    if cfg.drop_no_age and not has_age(result):
        return NO_AGE
    return None


def apply(results: list[ImageResult], cfg: PrefilterConfig) -> list[ImageResult]:
    """Drop the passing images *cfg* matches, in place; return the ones dropped."""
    dropped: list[ImageResult] = []
    for result in results:
        if not result.passed:
            continue  # a hard reject keeps its own reason
        why = reason(result, cfg)
        if why is not None:
            result.passed = False
            result.reject_reason = why
            dropped.append(result)
    return dropped


# What each requirement means, for the messages below.
_NEEDS = {
    "faces.enabled": "face detection",
    "faces.mivolo": "MiVOLO ages",
    "nsfw": "the NudeNet check (ARGUS_CURATOR_NUDENET_URL set, nsfw.enabled not false)",
}


def missing(cfg: PrefilterConfig, faces: FaceConfig, nsfw_enabled: bool) -> tuple[str, str] | None:
    """The first rule *cfg* turns on whose check will not run, and the setting it needs.

    *nsfw_enabled* is whether the scan runs NudeNet (:func:`nsfw.resolve_enabled`).
    Without it ``drop_nsfw`` would drop nothing, and without face detection
    every image would look faceless.
    """
    if (cfg.drop_no_face or cfg.drop_no_age) and not faces.enabled:
        return ("drop_no_face" if cfg.drop_no_face else "drop_no_age"), "faces.enabled"
    if cfg.drop_no_age and not faces.mivolo:
        return "drop_no_age", "faces.mivolo"
    if cfg.drop_nsfw and not nsfw_enabled:
        return "drop_nsfw", "nsfw"
    return None


def problem(cfg: PrefilterConfig, faces: FaceConfig, nsfw_enabled: bool) -> str | None:
    """Why *cfg* cannot run, or ``None`` (see :func:`missing`)."""
    if (gap := missing(cfg, faces, nsfw_enabled)) is None:
        return None
    rule, setting = gap
    return f"prefilter.{rule} needs {_NEEDS[setting]}" + ("" if setting == "nsfw" else f" ({setting})")
