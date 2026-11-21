"""Body pose from a remote argus-puppet server (:class:`PoseConfig`; docs/body-pose.md).

Curator never imports argus-puppet: it speaks puppet's wire. Each passing image
goes to ``POST /pose`` (the bytes the faces saw, with the primary face as the
``face=x,y,w,h`` hint, at ``X-Argus-Priority: batch``); then one ``POST
/stats`` over every posed image, inline, gives the scan-level clusters. What
lands on a row is plain values: the primary person's labels, the cluster id and
how the check went. The keypoints live only in this module's memory until the
``/stats`` call is answered: no row, snapshot, cache file or manifest holds one.

The server is configured by environment only (``ARGUS_CURATOR_PUPPET_URL``, see
:mod:`argus_curator.remote`): a request-supplied URL would let any API caller
make this server send images wherever they liked. The stage is opt-out like
NudeNet: on whenever the URL is set, unless a scan says otherwise. It never
fails a scan: a failed image keeps ``None``, and after
``MAX_FAILURES_WITHOUT_SUCCESS`` failures with no success the server is taken
to be down (:func:`remote.check_all`).

Puppet's ``/pose`` never calls curator, so the two can run side by side; each
takes its own argus-iris GPU lease.
"""

from __future__ import annotations

import io
import json
import os
import re
import time
import urllib.error
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, get_args
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import structlog
from argus_cortex.moderation import ModerationConfigError
from argus_cortex.moderation._http import CheckFailed, _request_json, is_number, service_url, sibling_url
from argus_cortex.wire import check_version
from PIL import Image

from argus_curator.models import (
    BodyArms,
    BodyFacing,
    BodyPoseLabels,
    BodyStance,
    ImageResult,
    PoseCluster,
    PoseConfig,
    PoseSummary,
    ScanSummary,
)
from argus_curator.remote import RemoteService, check_all, env_value, float_env
from argus_curator.runtime import ModelUnavailable
from argus_curator.selection import primary_face

logger = structlog.get_logger()

URL_ENV = "ARGUS_CURATOR_PUPPET_URL"
TIMEOUT_ENV = "ARGUS_CURATOR_PUPPET_TIMEOUT"
WORKERS_ENV = "ARGUS_CURATOR_PUPPET_WORKERS"
# Puppet's own caps on one /stats set (ARGUS_PUPPET_MAX_SET_RESULTS, ARGUS_PUPPET_MAX_JSON_BYTES):
# set these to match a server that lowered them. A scan over either is not clustered.
MAX_SET_ENV = "ARGUS_CURATOR_PUPPET_MAX_SET"
MAX_JSON_ENV = "ARGUS_CURATOR_PUPPET_MAX_JSON_BYTES"
DEFAULT_TIMEOUT = 60.0
# Puppet admits ARGUS_PUPPET_POSE_CONCURRENCY (default 2) /pose calls at once and answers the rest 503.
DEFAULT_WORKERS = 2
DEFAULT_MAX_SET = 5000
DEFAULT_MAX_JSON_BYTES = 128 * 1024 * 1024
# Puppet counts a body's people against max_json_bytes / this.
PERSON_JSON_FLOOR = 2048
# /stats is the whole scan's distance matrix in one call: give it longer than an image.
STATS_TIMEOUT_FACTOR = 5
SUPPORTED_MAJORS = ("1",)
PRIORITY = "batch"
# A 503 (puppet busy, or its GPU lease refused) is retried this many times, waiting
# Retry-After seconds (capped) between tries; then the image counts as failed.
MAX_RETRIES = 3
MAX_RETRY_WAIT_S = 10.0
SERVICE_NAME = "argus-puppet"

_STANCES = frozenset(get_args(BodyStance))
_ARMS = frozenset(get_args(BodyArms))
_FACINGS = frozenset(get_args(BodyFacing))
_EXIF_ORIENTATION = 0x0112

# time.sleep, through a name tests can replace.
_sleep = time.sleep


class PoseCheckFailed(CheckFailed):
    """One image (or the /stats call) could not be posed by argus-puppet."""


@dataclass
class Posed:
    """One image's ``/pose`` answer, as curator keeps it until ``/stats``.

    ``result`` is the PoseResult trimmed to its primary person (the only one
    the set metrics read), keypoints and all: memory only, never on a row.
    """

    version: str
    labels: BodyPoseLabels | None
    has_person: bool
    result: dict[str, Any] = field(repr=False)


def _labels(raw: object) -> BodyPoseLabels | None:
    """Puppet's ``labels`` object, a value this build does not know read as ``unknown``; None when absent."""
    if not isinstance(raw, dict):
        return None
    stance, arms, facing = raw.get("stance"), raw.get("arms"), raw.get("facing")
    return BodyPoseLabels(
        stance=stance if stance in _STANCES else "unknown",
        arms=arms if arms in _ARMS else "unknown",
        facing=facing if facing in _FACINGS else "unknown",
    )


def _version(payload: dict[str, Any]) -> str:
    version = payload.get("puppet_version")
    if not isinstance(version, str):
        raise PoseCheckFailed(f"{SERVICE_NAME} sent no puppet_version")
    check_version(version, SUPPORTED_MAJORS, label="puppet_version", error=PoseCheckFailed)
    return version


def _version_key(version: str) -> tuple[int, ...]:
    """``"1.10"`` after ``"1.9"``: compare versions by number, not text."""
    return tuple(int(part) if part.isdigit() else 0 for part in version.split("."))


def parse_pose(payload: object) -> Posed:
    """A ``POST /pose`` answer (a PoseResult): its primary person's labels, trimmed for ``/stats``.

    Tolerant of fields it does not know; refuses another ``puppet_version``
    major, or a ``primary`` that names no person.
    """
    if not isinstance(payload, dict):
        raise PoseCheckFailed(f"{SERVICE_NAME} answered with something other than a PoseResult")
    version = _version(payload)
    people, primary = payload.get("people"), payload.get("primary")
    if not isinstance(people, list):
        raise PoseCheckFailed(f"{SERVICE_NAME} sent no people list")
    trimmed = {k: v for k, v in payload.items() if k not in ("people", "primary")}
    if primary is None:
        return Posed(version, None, False, {**trimmed, "people": [], "primary": None})
    if not isinstance(primary, int) or isinstance(primary, bool) or not 0 <= primary < len(people):
        raise PoseCheckFailed(f"{SERVICE_NAME} named a primary person it did not send")
    person = people[primary]
    if not isinstance(person, dict):
        raise PoseCheckFailed(f"{SERVICE_NAME} sent a person that is not an object")
    return Posed(version, _labels(person.get("labels")), True, {**trimmed, "people": [person], "primary": 0})


def _name(rel_path: str) -> str:
    """A plain file name for puppet's ``name=`` (it refuses paths): the basename, header-safe."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", PurePosixPath(rel_path).name).lstrip(".")[-100:] or "image"


def upright_box(box: list[float], orientation: int, width: int, height: int) -> list[float]:
    """*box* ``[x, y, w, h]`` in the stored pixels of a *width* x *height* image, in its upright frame.

    Curator's faces are found on the stored pixels; puppet applies the EXIF
    orientation before it reads the hint, so the box turns with the image
    (Pillow's ``exif_transpose`` for each of the eight orientations).
    """
    x, y, w, h = box
    match orientation:
        case 2:  # mirrored left-right
            return [width - x - w, y, w, h]
        case 3:  # rotated 180
            return [width - x - w, height - y - h, w, h]
        case 4:  # mirrored top-bottom
            return [x, height - y - h, w, h]
        case 5:  # transposed
            return [y, x, h, w]
        case 6:  # rotated 90 clockwise to view
            return [height - y - h, x, h, w]
        case 7:  # transversed
            return [height - y - h, width - x - w, h, w]
        case 8:  # rotated 90 counter-clockwise to view
            return [y, width - x - w, h, w]
        case _:
            return [x, y, w, h]


def face_hint(result: ImageResult, data: bytes) -> str | None:
    """The primary face as puppet's ``face=x,y,w,h``, in the upright pixels puppet sees; None without one."""
    face = primary_face(result)
    if face is None or face.bbox[2] <= 0 or face.bbox[3] <= 0:
        return None
    try:
        with Image.open(io.BytesIO(data)) as img:
            orientation = int(img.getexif().get(_EXIF_ORIENTATION, 1) or 1)
            width, height = img.size
    except Exception:  # an image that will not open still goes; puppet says what it makes of it
        return None
    return ",".join(f"{v:.1f}" for v in upright_box(face.bbox, orientation, width, height))


def _with_query(url: str, **params: str) -> str:
    parts = urlsplit(url)
    query = [*parse_qsl(parts.query, keep_blank_values=True), *params.items()]
    return urlunsplit(parts._replace(query=urlencode(query)))


def _retry_after(exc: BaseException) -> float | None:
    """The wait a 503 asked for, or None when *exc* is not a 503."""
    cause = exc.__cause__
    if not isinstance(cause, urllib.error.HTTPError) or cause.code != 503:
        return None
    raw = cause.headers.get("Retry-After") if cause.headers is not None else None
    try:
        wait = float(raw) if raw is not None else 1.0
    except ValueError:
        wait = 1.0
    return min(max(wait, 0.0), MAX_RETRY_WAIT_S)


def http_status(exc: BaseException) -> int | None:
    cause = exc.__cause__
    return cause.code if isinstance(cause, urllib.error.HTTPError) else None


class PuppetClient:
    """An argus-puppet server at *base_url* (its origin, e.g. ``http://puppet:8220``)."""

    def __init__(self, base_url: str, timeout: float = DEFAULT_TIMEOUT) -> None:
        self.pose_url = service_url(base_url, "/pose", SERVICE_NAME)
        self.stats_url = sibling_url(self.pose_url, "/pose", "/stats")
        self.timeout = timeout

    def _post(self, url: str, body: bytes, content_type: str, timeout: float) -> object:
        headers = {"Content-Type": content_type, "X-Argus-Priority": PRIORITY}
        for attempt in range(MAX_RETRIES + 1):
            try:
                return _request_json("POST", url, body, headers, timeout, SERVICE_NAME, PoseCheckFailed)
            except PoseCheckFailed as exc:
                wait = _retry_after(exc)
                if wait is None or attempt == MAX_RETRIES:
                    raise
                _sleep(wait)
        raise AssertionError("unreachable")  # pragma: no cover

    def pose(self, data: bytes, rel_path: str, face: str | None = None) -> Posed:
        params = {"name": _name(rel_path)}
        if face is not None:
            params["face"] = face
        url = _with_query(self.pose_url, **params)
        return parse_pose(self._post(url, data, "application/octet-stream", self.timeout))

    def stats(self, body: bytes) -> object:
        return self._post(self.stats_url, body, "application/json", self.timeout * STATS_TIMEOUT_FACTOR)


def _make_client(base: str, timeout: float) -> PuppetClient:
    return PuppetClient(base, timeout)


SERVICE = RemoteService(URL_ENV, TIMEOUT_ENV, WORKERS_ENV, _make_client, DEFAULT_TIMEOUT, DEFAULT_WORKERS)


def client() -> PuppetClient | None:
    """A client for the configured server, or ``None`` when ``ARGUS_CURATOR_PUPPET_URL`` is unset.

    A malformed URL raises :class:`ModelUnavailable` naming the variable, never its value (it can hold a credential).
    """
    base = env_value(URL_ENV)
    if base is None:
        return None
    try:
        return _make_client(base, float_env(TIMEOUT_ENV, DEFAULT_TIMEOUT))
    except ModerationConfigError:
        raise ModelUnavailable(f"{URL_ENV} must be an http(s) URL") from None


def configured() -> bool:
    """A puppet URL is set (a malformed one still counts, so it fails loudly)."""
    return SERVICE.configured()


def resolve_enabled(cfg: PoseConfig) -> bool:
    """Whether a scan with *cfg* runs the pose stage: explicit choice, else configured."""
    return cfg.enabled if cfg.enabled is not None else configured()


def _int_env(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, ""))
    except ValueError:
        return default
    return value if value > 0 else default


def _stats_body(posed: dict[str, Posed]) -> bytes:
    """The ``POST /stats`` body: every posed image, each named by its rel_path (unique in a scan)."""
    results = [{**p.result, "image": rel} for rel, p in sorted(posed.items())]
    return json.dumps({"results": results}, separators=(",", ":"), allow_nan=False).encode()


def _over_cap(posed: dict[str, Posed], body: bytes) -> str | None:
    """Why *body* is over puppet's /stats caps (as configured here), else None. Checked, never truncated."""
    max_set, max_json = _int_env(MAX_SET_ENV, DEFAULT_MAX_SET), _int_env(MAX_JSON_ENV, DEFAULT_MAX_JSON_BYTES)
    if len(posed) > max_set:
        return f"{len(posed)} posed images is over the /stats set cap of {max_set} ({MAX_SET_ENV})"
    if len(body) > max_json:
        return f"the /stats body ({len(body)} bytes) is over the cap of {max_json} ({MAX_JSON_ENV})"
    people = sum(1 for p in posed.values() if p.has_person)
    if people > max_json // PERSON_JSON_FLOOR:
        return f"{people} people is over the /stats cap of {max_json // PERSON_JSON_FLOOR} ({MAX_JSON_ENV} / 2048)"
    return None


def _unit(value: object) -> float | None:
    return round(float(value), 4) if is_number(value) else None


def apply_stats(results: list[ImageResult], summary: PoseSummary, payload: object) -> None:
    """Record a ``PoseSetStats`` answer: each image's cluster (and labels a 1.0 server left out), and the scan's numbers."""
    if not isinstance(payload, dict):
        raise PoseCheckFailed(f"{SERVICE_NAME} answered /stats with something other than PoseSetStats")
    _version(payload)
    items, clusters = payload.get("items"), payload.get("clusters")
    if not isinstance(items, list) or not isinstance(clusters, list):
        raise PoseCheckFailed(f"{SERVICE_NAME} sent PoseSetStats without items or clusters")
    by_rel = {r.rel_path: r for r in results}
    known: set[int] = set()
    parsed: list[PoseCluster] = []
    for c in clusters:
        if not isinstance(c, dict) or not isinstance(c.get("id"), int) or not isinstance(c.get("size"), int):
            continue
        medoid = c.get("medoid")
        parsed.append(
            PoseCluster(
                id=c["id"],
                size=c["size"],
                share=_unit(c.get("share")) or 0.0,
                medoid_rel_path=medoid if isinstance(medoid, str) and medoid in by_rel else None,
            )
        )
        known.add(c["id"])
    for item in items:
        if not isinstance(item, dict) or (row := by_rel.get(item.get("image"))) is None:  # type: ignore[arg-type]
            continue
        if row.pose_check not in ("ok", "unusable"):
            continue  # nobody in frame: nothing to cluster or label
        cluster = item.get("cluster")
        usable = item.get("usable") is True and isinstance(cluster, int) and cluster in known
        row.pose_cluster = cluster if usable else None
        row.pose_check = "ok" if usable else "unusable"
        if row.pose_labels is None:  # a 1.0 server put no labels on its people
            row.pose_labels = _labels(item.get("labels"))
    summary.usable = payload.get("usable") if isinstance(payload.get("usable"), int) else None
    summary.comparable_share = _unit(payload.get("comparable_share"))
    summary.diversity = _unit(payload.get("diversity"))
    summary.largest_cluster_share = _unit(payload.get("largest_cluster_share"))
    summary.clusters = sorted(parsed, key=lambda c: c.id)


def pose_images(
    results: list[ImageResult],
    items: list[tuple[str, str, bytes]],
    progress: Callable[[int, int], None] | None = None,
) -> PoseSummary:
    """Pose every image in *results* (the scanner passes the passing ones), then cluster the scan.

    Sets ``pose_check``, ``pose_labels`` and ``pose_cluster`` on each row and
    returns the scan-level :class:`PoseSummary`. Never raises for a failed
    image, a down server or a refused ``/stats``: those leave ``None`` and say
    why. Raises :class:`ModelUnavailable` only for an unset or malformed URL,
    which the scanner checks before any work.
    """
    puppet = client()
    if puppet is None:
        raise ModelUnavailable(f"body pose requested but {URL_ENV} is not set")
    data_by_rel = {rel: data for rel, _abs, data in items}
    by_rel = {r.rel_path: r for r in results}
    for r in results:
        if data_by_rel.get(r.rel_path) is not None:
            r.pose_check = "failed"  # until its answer lands
    posed: dict[str, Posed] = {}

    def check(data: bytes, rel: str) -> Posed:
        return puppet.pose(data, rel, face_hint(by_rel[rel], data))

    def store(row: ImageResult, answer: Posed) -> None:
        posed[row.rel_path] = answer
        row.pose_labels = answer.labels if answer.has_person else None
        row.pose_check = "ok" if answer.has_person else "no_person"

    check_all(results, items, check, "pose", SERVICE.workers(), progress, store=store)

    summary = PoseSummary(
        puppet_version=min((p.version for p in posed.values()), key=_version_key, default=None),
        posed=len(posed),
        no_person=sum(1 for p in posed.values() if not p.has_person),
        failed=sum(1 for r in results if r.pose_check == "failed"),
    )
    try:
        if not posed:
            summary.skipped = "no image was posed"
            return summary
        body = _stats_body(posed)
        if (why := _over_cap(posed, body)) is not None:
            summary.skipped = why
            logger.warning("pose_clustering_skipped", reason=why)
            return summary
        try:
            apply_stats(results, summary, puppet.stats(body))
        except (PoseCheckFailed, ValueError) as exc:
            status = http_status(exc)
            summary.skipped = (
                f"{SERVICE_NAME} refused the set as too large (HTTP 413)"
                if status == 413
                else f"the /stats call failed: {exc}"
            )
            logger.warning("pose_clustering_failed", error=str(exc))
            for r in results:
                r.pose_cluster = None
        return summary
    finally:
        posed.clear()  # the keypoints go now


def recount(summary: ScanSummary, results: list[ImageResult]) -> None:
    """Bring the scan's pose clusters in line with *results* after rows were dropped.

    Sizes and shares are recounted from the rows left; ``diversity`` and
    ``comparable_share`` cannot be without the keypoints, so ``stale`` says so.
    """
    pose = summary.pose
    if pose is None or not pose.clusters:
        return
    sizes: dict[int, int] = {}
    for r in results:
        if r.pose_cluster is not None:
            sizes[r.pose_cluster] = sizes.get(r.pose_cluster, 0) + 1
    usable = sum(sizes.values())
    if usable == pose.usable and all(sizes.get(c.id, 0) == c.size for c in pose.clusters):
        return
    rels = {r.rel_path for r in results}
    kept = []
    for c in pose.clusters:
        if not sizes.get(c.id):
            continue
        c.size = sizes[c.id]
        c.share = round(c.size / usable, 4)
        if c.medoid_rel_path not in rels:
            c.medoid_rel_path = None
        kept.append(c)
    pose.clusters = kept
    pose.usable = usable
    pose.largest_cluster_share = max((c.share for c in kept), default=0.0)
    pose.stale = True
