"""Shared plumbing for the checks curator runs on a remote server (NudeNet, SingGuard).

The HTTP clients themselves are argus-cortex's (:mod:`argus_cortex.moderation`);
this module holds what is curator's: configuration from the environment, and
checking a whole scan's images in parallel.

Each server is configured by environment only, never by a request: a
request-supplied URL would let any API caller make this server send images
wherever they liked. Each check is opt-out: on whenever its URL is set, unless
a scan says otherwise.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Callable
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from typing import Any

import structlog

from argus_curator.models import ImageResult

logger = structlog.get_logger()

# Give up on the rest of a scan after this many failures with no success at all:
# the server is down or unreachable, and every further image would only wait
# out its timeout.
MAX_FAILURES_WITHOUT_SUCCESS = 8


def env_value(env: str) -> str | None:
    """*env*'s value, stripped, or ``None`` when it is unset or blank."""
    return os.environ.get(env, "").strip() or None


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
    on_done: Callable[[ImageResult, int, int], None] | None = None,
    store: Callable[[ImageResult, Any], None] | None = None,
) -> None:
    """Set ``field`` on every image to ``check(data, rel_path)``, *workers* at a time.

    *store* replaces the ``setattr`` when the result goes somewhere else;
    ``field`` still names the check in logs.

    Every image is checked, including ones that failed the quality filters: a
    blurry explicit image is still explicit, and ``include_rejected`` exports
    them. An image whose check fails keeps ``None``, which the filters treat as
    a reason to exclude it. After ``MAX_FAILURES_WITHOUT_SUCCESS`` failures with
    no success the server is taken to be down and the images not yet sent stay
    unchecked, rather than each waiting out its timeout.
    """
    data_by_rel = {rel: data for rel, _abs, data in items}
    pending = [r for r in results if data_by_rel.get(r.rel_path) is not None]
    total, done, failed, succeeded = len(pending), 0, 0, 0
    if progress is not None:
        progress(0, total)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(check, data_by_rel[r.rel_path], r.rel_path): r for r in pending}
        for fut in as_completed(futures):
            r = futures[fut]
            try:
                value = fut.result()
                if store is None:
                    setattr(r, field, value)
                else:
                    store(r, value)
                succeeded += 1
            except CancelledError:
                pass  # given up on below; the image stays unchecked
            except Exception as exc:  # one bad image must not fail the whole scan
                failed += 1
                logger.warning("remote_check_failed", check=field, rel_path=r.rel_path, error=str(exc))
                if not succeeded and failed == MAX_FAILURES_WITHOUT_SUCCESS:
                    logger.warning("remote_server_unreachable", check=field, failed=failed, total=total)
                    for other in futures:
                        other.cancel()
            done += 1
            if on_done is not None:
                # A chunk consumer must not be counted as a failed check.
                with contextlib.suppress(Exception):
                    on_done(r, done, total)
            if progress is not None and (done == total or done % 5 == 0):
                progress(done, total)
    if failed:
        logger.warning("remote_checks_failed", check=field, failed=failed, total=total)
