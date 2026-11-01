"""EraX-NSFW as an optional second detector beside NudeNet.

Curator never imports Ultralytics or EraX: it posts each image to a server
that speaks NudeNet's ``/infer`` API with EraX class names, through the
same HTTP helpers as :mod:`argus_curator.nsfw`. Running that server is
the deployer's choice; curator stays MIT.

When NudeNet returns no exposed score (empty detections, or the check
failed), EraX hits are mapped onto NudeNet's exposed labels so filters,
blur, pairing and cascade keep using ``nsfw``. An image NudeNet already
scored as exposed is left alone.

Configured by ``ARGUS_CURATOR_ERAX_URL``. Empty or unset turns it off.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import PurePosixPath

from argus_cortex.moderation import ModerationConfigError
from argus_cortex.moderation._http import CheckFailed, post_json, service_url

# The request shape and number check are NudeNet's: EraX speaks the same API.
from argus_cortex.moderation.nudenet import MIN_REPORTED_SCORE, _is_number, _multipart, to_result

from argus_curator.models import EraxDetection, EraxLabel, EraxResult, ImageResult, NsfwConfig
from argus_curator.remote import check_all, env_set, env_value, float_env
from argus_curator.runtime import ModelUnavailable

URL_ENV = "ARGUS_CURATOR_ERAX_URL"
TIMEOUT_ENV = "ARGUS_CURATOR_ERAX_TIMEOUT"
WORKERS_ENV = "ARGUS_CURATOR_ERAX_WORKERS"
DEFAULT_TIMEOUT = 30.0
DEFAULT_WORKERS = 4

ERAX_LABELS: tuple[EraxLabel, ...] = EraxLabel.__args__

# EraX's five classes → NudeNet exposed labels. ``nipple`` is the exposed-
# breast signal (EraX does not split male/female). ``make_love`` is a sex
# act, not a body part: both genitalia labels get the box so exposed_score
# and pairing still fire.
_TO_NUDENET: dict[EraxLabel, tuple[str, ...]] = {
    "anus": ("ANUS_EXPOSED",),
    "penis": ("MALE_GENITALIA_EXPOSED",),
    "vagina": ("FEMALE_GENITALIA_EXPOSED",),
    "nipple": ("FEMALE_BREAST_EXPOSED",),
    "make_love": ("FEMALE_GENITALIA_EXPOSED", "MALE_GENITALIA_EXPOSED"),
}


class EraxCheckFailed(CheckFailed):
    """One image could not be checked by EraX."""


def to_erax_result(raw: list) -> EraxResult:
    """One image's EraX detections as recorded, plus the per-class maxima."""
    known = set(ERAX_LABELS)
    dets: list[EraxDetection] = []
    for d in raw:
        if not isinstance(d, dict):
            continue
        label, score, box = d.get("class"), d.get("score"), d.get("box")
        if (
            not isinstance(label, str)
            or label not in known
            or not _is_number(score)
            or score < MIN_REPORTED_SCORE
            or not isinstance(box, list)
            or len(box) != 4
            or not all(_is_number(v) for v in box)
        ):
            continue
        dets.append(EraxDetection(label=label, score=round(float(score), 4), bbox=[round(float(v), 1) for v in box]))
    dets.sort(key=lambda d: -d.score)
    labels: dict[str, float] = {}
    for d in dets:
        labels.setdefault(d.label, d.score)
    return EraxResult(detections=dets, labels=labels)


def mapped_detections(erax: EraxResult) -> list[dict]:
    """EraX hits as NudeNet-shaped dicts for :func:`to_result`."""
    out: list[dict] = []
    for det in erax.detections:
        for label in _TO_NUDENET[det.label]:
            out.append({"class": label, "score": det.score, "box": det.bbox})
    return out


def needs_fill(result: ImageResult) -> bool:
    """NudeNet left this image with no exposed score (miss or failed check)."""
    return result.nsfw is None or result.nsfw.exposed_score == 0.0


def fill_nsfw(results: list[ImageResult]) -> int:
    """Copy EraX hits into ``nsfw`` when NudeNet did not score the image as exposed.

    Existing non-exposed NudeNet detections (a face, a covered class) are
    kept and the mapped EraX boxes are added. Returns how many images were
    filled.
    """
    filled = 0
    for result in results:
        if result.erax is None or not result.erax.detections or not needs_fill(result):
            continue
        raw = mapped_detections(result.erax)
        if not raw:
            continue
        if result.nsfw is not None:
            raw = [{"class": d.label, "score": d.score, "box": d.bbox} for d in result.nsfw.detections] + raw
        result.nsfw = to_result(raw)
        result.erax_filled = True
        filled += 1
    return filled


class EraxClient:
    """An EraX server at *base_url* (its origin, e.g. ``http://erax:8080``)."""

    def __init__(self, base_url: str, timeout: float = DEFAULT_TIMEOUT) -> None:
        self.url = service_url(base_url, "/infer", "EraX")
        self.timeout = timeout

    def check(self, data: bytes, filename: str = "image") -> EraxResult:
        body, content_type = _multipart(data, filename)
        payload = post_json(self.url, body, content_type, self.timeout, "EraX", EraxCheckFailed)
        prediction = payload.get("prediction") if isinstance(payload, dict) else None
        if not (isinstance(payload, dict) and payload.get("success") is True and isinstance(prediction, list)):
            raise EraxCheckFailed("EraX did not report success")
        if len(prediction) != 1 or not isinstance(prediction[0], list):
            raise EraxCheckFailed(f"EraX returned {len(prediction)} results for one image")
        return to_erax_result(prediction[0])


def client() -> EraxClient | None:
    """A client for the configured server, or ``None`` when EraX is not configured."""
    base = env_value(URL_ENV)
    if base is None:
        return None
    try:
        return EraxClient(base, timeout=float_env(TIMEOUT_ENV, DEFAULT_TIMEOUT))
    except ModerationConfigError as exc:
        raise ModelUnavailable(f"{URL_ENV} must be an http(s) URL, got {base!r}") from exc


def endpoint() -> str | None:
    """The configured ``/infer`` URL, or ``None`` when EraX is not configured."""
    c = client()
    return c.url if c else None


def configured() -> bool:
    """An EraX URL is set. A malformed one still counts, so that it fails the
    scan loudly instead of silently switching the check off."""
    return env_set(URL_ENV)


def resolve_enabled(nsfw_cfg: NsfwConfig) -> bool:
    """EraX runs when its URL is set and the NudeNet check was not opted out.

    It is a fill-in for NudeNet, not a replacement: ``--no-nsfw`` turns both
    off.
    """
    from argus_curator.nsfw import resolve_enabled as nsfw_enabled

    return configured() and nsfw_enabled(nsfw_cfg)


def check_images(
    results: list[ImageResult],
    items: list[tuple[str, str, bytes]],
    progress: Callable[[int, int], None] | None = None,
    on_done: Callable[..., None] | None = None,
) -> bool:
    """Set ``erax`` on every image; return whether the check ran at all."""
    erax = client()
    if erax is None:
        return False
    check_all(
        results,
        items,
        lambda data, rel: erax.check(data, PurePosixPath(rel).name),
        "erax",
        int(float_env(WORKERS_ENV, DEFAULT_WORKERS)),
        progress,
        on_done=on_done,
    )
    return True
