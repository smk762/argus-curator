"""NudeNet nudity detection through a remote NudeNet server.

Curator never imports NudeNet: it posts each image to a NudeNet server
(``ghcr.io/notai-tech/nudenet``, AGPL-3.0) over HTTP, through argus-cortex's
:class:`~argus_cortex.moderation.NudeNetClient`, and records the detections.
Running that server is the deployer's choice; curator itself stays MIT.

The server is configured by environment only (``ARGUS_CURATOR_NUDENET_URL``;
see :mod:`argus_curator.remote`). When the URL is set, every scan checks its
images unless the scan opts out (``NsfwConfig.enabled = False``).

"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import PurePosixPath

from argus_cortex.moderation import NudeNetCheckFailed as NsfwCheckFailed  # noqa: F401 - re-exported
from argus_cortex.moderation import (
    NudeNetClient,
    exposed_score,  # noqa: F401 - re-exported for selection
)
from argus_cortex.moderation.nudenet import to_result  # noqa: F401 - re-exported

from argus_curator.models import ImageResult, NsfwConfig
from argus_curator.remote import RemoteService, check_all
from argus_curator.runtime import ModelUnavailable

URL_ENV = "ARGUS_CURATOR_NUDENET_URL"
TIMEOUT_ENV = "ARGUS_CURATOR_NUDENET_TIMEOUT"
WORKERS_ENV = "ARGUS_CURATOR_NUDENET_WORKERS"
DEFAULT_TIMEOUT = 30.0
DEFAULT_WORKERS = 4

SERVICE = RemoteService(URL_ENV, TIMEOUT_ENV, WORKERS_ENV, NudeNetClient, DEFAULT_TIMEOUT, DEFAULT_WORKERS)


def client() -> NudeNetClient | None:
    """A client for the configured server, or ``None`` when NudeNet is not configured."""
    return SERVICE.client()


def endpoint() -> str | None:
    """The configured ``/infer`` URL, or ``None`` when NudeNet is not configured.

    The variable holds the server's origin (``http://nudenet:8080``); a value
    already ending in ``/infer`` is used as is.
    """
    c = client()
    return c.url if c else None


def configured() -> bool:
    """A NudeNet URL is set (a malformed one still counts; see :meth:`RemoteService.configured`)."""
    return SERVICE.configured()


def resolve_enabled(cfg: NsfwConfig) -> bool:
    """Whether a scan with *cfg* checks images: explicit choice, else configured."""
    return cfg.enabled if cfg.enabled is not None else configured()


def check_images(
    results: list[ImageResult],
    items: list[tuple[str, str, bytes]],
    cfg: NsfwConfig,
    progress: Callable[[int, int], None] | None = None,
    on_done: Callable[..., None] | None = None,
) -> bool:
    """Set ``nsfw`` on every image (see :func:`remote.check_all`); return whether
    the check ran at all."""
    if not resolve_enabled(cfg):
        return False
    nudenet = client()
    if nudenet is None:
        raise ModelUnavailable(f"NudeNet check requested but {URL_ENV} is not set")
    check_all(
        results,
        items,
        lambda data, rel: nudenet.check(data, PurePosixPath(rel).name),
        "nsfw",
        SERVICE.workers(),
        progress,
        on_done=on_done,
    )
    return True


def exposed_reading(result: ImageResult) -> float | None:
    """The image's exposed-nudity reading, or ``None`` when NudeNet (and EraX) did not score it.

    The higher of the recorded ``exposed_score`` and the exposed classes' label
    scores. Compared against ``MIN_REPORTED_SCORE`` (0.2) by :func:`selection.minor_exposed`
    and the scan prefilter.
    """
    if result.nsfw is None:
        return None
    return max(result.nsfw.exposed_score, exposed_score(result.nsfw.labels))
