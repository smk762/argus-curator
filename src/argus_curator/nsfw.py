"""NudeNet nudity detection through a remote NudeNet server.

Curator never imports NudeNet: it posts each image to a NudeNet server
(``ghcr.io/notai-tech/nudenet``, AGPL-3.0) over HTTP and records the
detections. Running that server is the deployer's choice; curator itself
stays MIT and gains no dependency.

The server is configured by environment only (``ARGUS_CURATOR_NUDENET_URL``),
never by a request: a request-supplied URL would let any API caller make this
server send images wherever they liked. When the URL is set, every scan checks
its images unless the scan opts out (``NsfwConfig.enabled = False``).

API (NudeNet's README): ``POST /infer`` with the image as multipart field
``f1``, answered by ``{"prediction": [[{"class", "score", "box"}, ...]],
"success": true}``. ``box`` is ``[x, y, w, h]`` in source pixels, and the
server drops detections scoring under 0.2.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import PurePosixPath
from urllib.parse import urlsplit

import structlog

from argus_curator.models import (
    DEFAULT_EXPOSED_LABELS,
    NUDENET_LABELS,
    ImageResult,
    NsfwConfig,
    NsfwDetection,
    NsfwResult,
)
from argus_curator.runtime import ModelUnavailable

logger = structlog.get_logger()

URL_ENV = "ARGUS_CURATOR_NUDENET_URL"
TIMEOUT_ENV = "ARGUS_CURATOR_NUDENET_TIMEOUT"
WORKERS_ENV = "ARGUS_CURATOR_NUDENET_WORKERS"
DEFAULT_TIMEOUT = 30.0
DEFAULT_WORKERS = 4


class NsfwCheckFailed(RuntimeError):
    """One image could not be checked (network error, bad response)."""


def endpoint() -> str | None:
    """The configured ``/infer`` URL, or ``None`` when NudeNet is not configured.

    The variable holds the server's origin (``http://nudenet:8080``); a value
    already ending in ``/infer`` is used as is.
    """
    base = os.environ.get(URL_ENV, "").strip().rstrip("/")
    if not base:
        return None
    parts = urlsplit(base)
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        raise ModelUnavailable(f"{URL_ENV} must be an http(s) URL, got {base!r}")
    return base if base.endswith("/infer") else f"{base}/infer"


def configured() -> bool:
    """A NudeNet URL is set. A malformed one still counts, so that it fails the
    scan loudly instead of silently switching the check off."""
    return bool(os.environ.get(URL_ENV, "").strip())


def resolve_enabled(cfg: NsfwConfig) -> bool:
    """Whether a scan with *cfg* checks images: explicit choice, else configured."""
    return cfg.enabled if cfg.enabled is not None else configured()


def _float_env(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, ""))
    except ValueError:
        return default
    return value if value > 0 else default


def _multipart(data: bytes, filename: str) -> tuple[bytes, str]:
    # The name lands in a header: keep it to characters that cannot break it.
    filename = re.sub(r"[^A-Za-z0-9._-]", "_", filename)[-100:] or "image"
    boundary = uuid.uuid4().hex
    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="f1"; filename="{filename}"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
    ).encode()
    return head + data + f"\r\n--{boundary}--\r\n".encode(), f"multipart/form-data; boundary={boundary}"


def to_result(raw: list[dict]) -> NsfwResult:
    """Detections as recorded, plus the per-class maxima and exposed score."""
    known = set(NUDENET_LABELS)
    dets: list[NsfwDetection] = []
    for d in raw:
        label, score, box = d.get("class"), d.get("score"), d.get("box")
        if label not in known or not isinstance(score, int | float) or not isinstance(box, list) or len(box) != 4:
            continue  # an unknown class or malformed entry is not evidence either way
        dets.append(NsfwDetection(label=label, score=round(float(score), 4), bbox=[round(float(v), 1) for v in box]))
    dets.sort(key=lambda d: -d.score)
    labels: dict[str, float] = {}
    for d in dets:
        labels.setdefault(d.label, d.score)
    return NsfwResult(
        detections=dets,
        labels=labels,
        exposed_score=exposed_score(labels, DEFAULT_EXPOSED_LABELS),
    )


def exposed_score(labels: dict[str, float], exposed: tuple[str, ...] | list[str]) -> float:
    """The highest score among the *exposed* classes (0.0 when none was detected)."""
    return max((labels[c] for c in exposed if c in labels), default=0.0)


def check(data: bytes, filename: str, url: str, timeout: float) -> NsfwResult:
    """Post one image to the NudeNet server and parse its answer."""
    body, content_type = _multipart(data, filename)
    req = urllib.request.Request(url, data=body, headers={"Content-Type": content_type}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - operator-configured URL
            payload = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        raise NsfwCheckFailed(f"NudeNet returned HTTP {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise NsfwCheckFailed(f"NudeNet request failed: {exc}") from exc
    except ValueError as exc:
        raise NsfwCheckFailed("NudeNet returned a body that is not JSON") from exc
    prediction = payload.get("prediction") if isinstance(payload, dict) else None
    if not (isinstance(payload, dict) and payload.get("success") is True and isinstance(prediction, list)):
        raise NsfwCheckFailed("NudeNet did not report success")
    if len(prediction) != 1 or not isinstance(prediction[0], list):
        raise NsfwCheckFailed(f"NudeNet returned {len(prediction)} results for one image")
    return to_result(prediction[0])


def check_images(
    results: list[ImageResult],
    items: list[tuple[str, str, bytes]],
    cfg: NsfwConfig,
    progress: Callable[[int, int], None] | None = None,
) -> bool:
    """Set ``nsfw`` on every image; return whether the check ran at all.

    Every image is checked, including ones that failed the quality filters: a
    blurry explicit image is still explicit, and ``include_rejected`` exports
    them. An image whose check fails keeps ``nsfw = None``, which NSFW filters
    treat as a reason to exclude it.
    """
    if not resolve_enabled(cfg):
        return False
    url = endpoint()
    if url is None:
        raise ModelUnavailable(f"NudeNet check requested but {URL_ENV} is not set")
    timeout = _float_env(TIMEOUT_ENV, DEFAULT_TIMEOUT)
    workers = int(_float_env(WORKERS_ENV, DEFAULT_WORKERS))
    data_by_rel = {rel: data for rel, _abs, data in items}
    pending = [r for r in results if data_by_rel.get(r.rel_path) is not None]
    total, done, failed = len(pending), 0, 0
    if progress is not None:
        progress(0, total)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(check, data_by_rel[r.rel_path], PurePosixPath(r.rel_path).name, url, timeout): r
            for r in pending
        }
        for fut in as_completed(futures):
            r = futures[fut]
            try:
                r.nsfw = fut.result()
            except NsfwCheckFailed as exc:
                failed += 1
                logger.warning("nsfw_check_failed", rel_path=r.rel_path, error=str(exc))
            done += 1
            if progress is not None and (done == total or done % 5 == 0):
                progress(done, total)
    if failed:
        logger.warning("nsfw_checks_failed", failed=failed, total=total)
    return True
