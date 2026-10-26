"""Shared plumbing for the checks curator runs on a remote server (NudeNet, SingGuard).

Each server is configured by environment only, never by a request: a
request-supplied URL would let any API caller make this server send images
wherever they liked. Each check is opt-out: on whenever its URL is set, unless
a scan says otherwise.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from urllib.parse import urlsplit

import structlog

from argus_curator.models import ImageResult
from argus_curator.runtime import ModelUnavailable

logger = structlog.get_logger()


class CheckFailed(RuntimeError):
    """One image could not be checked (network error, bad response)."""


def service_url(env: str, path: str) -> str | None:
    """The URL in *env* with *path* appended, or ``None`` when *env* is unset.

    The variable holds the server's origin; a value already ending in *path*
    is used as is.
    """
    base = os.environ.get(env, "").strip().rstrip("/")
    if not base:
        return None
    parts = urlsplit(base)
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        raise ModelUnavailable(f"{env} must be an http(s) URL, got {base!r}")
    return base if base.endswith(path) else f"{base}{path}"


def env_set(env: str) -> bool:
    """*env* holds a value. A malformed one still counts, so that it fails the
    scan loudly instead of silently switching the check off."""
    return bool(os.environ.get(env, "").strip())


def float_env(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, ""))
    except ValueError:
        return default
    return value if value > 0 else default


def check_all(
    results: list[ImageResult],
    items: list[tuple[str, str, bytes]],
    check: Callable[[bytes, str], Any],
    field: str,
    workers: int,
    progress: Callable[[int, int], None] | None = None,
) -> None:
    """Set ``field`` on every image to ``check(data, rel_path)``, *workers* at a time.

    Every image is checked, including ones that failed the quality filters: a
    blurry explicit image is still explicit, and ``include_rejected`` exports
    them. An image whose check fails keeps ``None``, which the filters treat as
    a reason to exclude it.
    """
    data_by_rel = {rel: data for rel, _abs, data in items}
    pending = [r for r in results if data_by_rel.get(r.rel_path) is not None]
    total, done, failed = len(pending), 0, 0
    if progress is not None:
        progress(0, total)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(check, data_by_rel[r.rel_path], r.rel_path): r for r in pending}
        for fut in as_completed(futures):
            r = futures[fut]
            try:
                setattr(r, field, fut.result())
            except CheckFailed as exc:
                failed += 1
                logger.warning("remote_check_failed", check=field, rel_path=r.rel_path, error=str(exc))
            done += 1
            if progress is not None and (done == total or done % 5 == 0):
                progress(done, total)
    if failed:
        logger.warning("remote_checks_failed", check=field, failed=failed, total=total)
