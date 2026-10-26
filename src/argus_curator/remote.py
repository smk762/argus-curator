"""Shared plumbing for the checks curator runs on a remote server (NudeNet, SingGuard).

Each server is configured by environment only, never by a request: a
request-supplied URL would let any API caller make this server send images
wherever they liked. Each check is opt-out: on whenever its URL is set, unless
a scan says otherwise.
"""

from __future__ import annotations

import http.client
import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from typing import Any
from urllib.parse import urlsplit

import structlog

from argus_curator.models import ImageResult
from argus_curator.runtime import ModelUnavailable

logger = structlog.get_logger()

# Give up on the rest of a scan after this many failures with no success at all:
# the server is down or unreachable, and every further image would only wait
# out its timeout.
MAX_FAILURES_WITHOUT_SUCCESS = 8

# Ignore HTTP(S)_PROXY: images go straight to the operator-configured server,
# never through whatever proxy happens to be in the environment.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


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


def post_json(
    url: str, body: bytes, content_type: str, timeout: float, service: str, failed: type[CheckFailed]
) -> object:
    """POST *body* to *url* and return the decoded JSON answer.

    Every transport or decoding problem is raised as *failed*, so that
    :func:`check_all` records it against the one image instead of aborting the scan.
    """
    req = urllib.request.Request(url, data=body, headers={"Content-Type": content_type}, method="POST")
    try:
        with _OPENER.open(req, timeout=timeout) as resp:  # noqa: S310 - operator-configured URL
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        raise failed(f"{service} returned HTTP {exc.code}") from exc
    except (urllib.error.URLError, http.client.HTTPException, TimeoutError, OSError) as exc:
        raise failed(f"{service} request failed: {exc}") from exc
    except ValueError as exc:
        raise failed(f"{service} returned a body that is not JSON") from exc


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
                setattr(r, field, fut.result())
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
            if progress is not None and (done == total or done % 5 == 0):
                progress(done, total)
    if failed:
        logger.warning("remote_checks_failed", check=field, failed=failed, total=total)
