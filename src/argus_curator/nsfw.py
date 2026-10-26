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

from argus_cortex.moderation import (
    ModerationConfigError,
    NudeNetClient,
    exposed_score,  # noqa: F401 - re-exported for selection
)
from argus_cortex.moderation import NudeNetCheckFailed as NsfwCheckFailed  # noqa: F401 - re-exported
from argus_cortex.moderation.nudenet import to_result  # noqa: F401 - re-exported

from argus_curator.models import ImageResult, NsfwConfig
from argus_curator.remote import check_all, env_set, env_value, float_env
from argus_curator.runtime import ModelUnavailable

URL_ENV = "ARGUS_CURATOR_NUDENET_URL"
TIMEOUT_ENV = "ARGUS_CURATOR_NUDENET_TIMEOUT"
WORKERS_ENV = "ARGUS_CURATOR_NUDENET_WORKERS"
DEFAULT_TIMEOUT = 30.0
DEFAULT_WORKERS = 4


def client() -> NudeNetClient | None:
    """A client for the configured server, or ``None`` when NudeNet is not configured."""
    base = env_value(URL_ENV)
    if base is None:
        return None
    try:
        return NudeNetClient(base, timeout=float_env(TIMEOUT_ENV, DEFAULT_TIMEOUT))
    except ModerationConfigError as exc:
        raise ModelUnavailable(f"{URL_ENV} must be an http(s) URL, got {base!r}") from exc


def endpoint() -> str | None:
    """The configured ``/infer`` URL, or ``None`` when NudeNet is not configured.

    The variable holds the server's origin (``http://nudenet:8080``); a value
    already ending in ``/infer`` is used as is.
    """
    c = client()
    return c.url if c else None


def configured() -> bool:
    """A NudeNet URL is set. A malformed one still counts, so that it fails the
    scan loudly instead of silently switching the check off."""
    return env_set(URL_ENV)


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
        int(float_env(WORKERS_ENV, DEFAULT_WORKERS)),
        progress,
        on_done=on_done,
    )
    return True
