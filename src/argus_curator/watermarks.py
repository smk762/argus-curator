"""Watermark detection on a remote server, for :class:`CropConfig.detect`.

Curator never imports Ultralytics: it posts each image to a server that
speaks NudeNet's ``POST /infer`` shape (multipart ``f1`` →
``{"prediction": [[{"class", "score", "box": [x, y, w, h]}, …]], "success": true}``),
as the EraX sidecar does. Running that server is the deployer's choice;
curator stays MIT.

Images are sent upright (EXIF orientation applied), so the boxes that come
back are in the displayed pixels crop boxes use.

Configured by ``ARGUS_CURATOR_WATERMARK_URL``. Unlike NudeNet it is not
opt-out: it runs only when a scan asks for ``crops.detect``.
"""

from __future__ import annotations

import http.client
import io
import urllib.error
import urllib.request
from collections.abc import Callable
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from urllib.parse import urlsplit, urlunsplit

import structlog
from argus_cortex.moderation._http import _OPENER, CheckFailed, _split_credentials, post_json, service_url
from argus_cortex.moderation.nudenet import _is_number, _multipart
from PIL import Image, ImageOps

from argus_curator.models import WatermarkDetection
from argus_curator.remote import MAX_FAILURES_WITHOUT_SUCCESS, RemoteService

logger = structlog.get_logger()

URL_ENV = "ARGUS_CURATOR_WATERMARK_URL"
TIMEOUT_ENV = "ARGUS_CURATOR_WATERMARK_TIMEOUT"
WORKERS_ENV = "ARGUS_CURATOR_WATERMARK_WORKERS"
DEFAULT_TIMEOUT = 60.0
DEFAULT_WORKERS = 2
# A scan waits on this before it starts, so it is far shorter than an image's timeout.
HEALTH_TIMEOUT = 5.0


class WatermarkCheckFailed(CheckFailed):
    """One image could not be checked by the watermark detector."""


def to_detections(raw: list) -> list[WatermarkDetection]:
    """One image's hits, most confident first. Malformed rows are dropped."""
    out: list[WatermarkDetection] = []
    for d in raw:
        if not isinstance(d, dict):
            continue
        label, score, box = d.get("class"), d.get("score"), d.get("box")
        if (
            not isinstance(label, str)
            or not _is_number(score)
            or not isinstance(box, list)
            or len(box) != 4
            or not all(_is_number(v) for v in box)
            or box[2] <= 0
            or box[3] <= 0
        ):
            continue
        out.append(
            WatermarkDetection(label=label, score=round(float(score), 4), bbox=[round(float(v), 1) for v in box])
        )
    out.sort(key=lambda d: -d.score)
    return out


class WatermarkClient:
    """A watermark detector at *base_url* (its origin, e.g. ``http://watermark:8080``)."""

    def __init__(self, base_url: str, timeout: float = DEFAULT_TIMEOUT) -> None:
        self.url = service_url(base_url, "/infer", "watermark detector")
        self.timeout = timeout

    def check(self, data: bytes, filename: str = "image.jpg") -> list[WatermarkDetection]:
        body, content_type = _multipart(data, filename)
        payload = post_json(self.url, body, content_type, self.timeout, "watermark detector", WatermarkCheckFailed)
        prediction = payload.get("prediction") if isinstance(payload, dict) else None
        if not (isinstance(payload, dict) and payload.get("success") is True and isinstance(prediction, list)):
            raise WatermarkCheckFailed("watermark detector did not report success")
        if len(prediction) != 1 or not isinstance(prediction[0], list):
            raise WatermarkCheckFailed(f"watermark detector returned {len(prediction)} results for one image")
        return to_detections(prediction[0])

    def unreachable(self) -> str | None:
        """Why this server cannot take a scan, or ``None`` when it answers.

        ``/detectors`` only says the URL is set; this is what stops a scan
        rejecting every frame as ``crop:unchecked``. Only ``POST /infer`` is the
        contract, so a server with no ``/health`` still counts as up: any answer
        below 500 does. The message leaves the URL out, as it can hold a credential.
        """
        parts = urlsplit(self.url)
        health = parts._replace(path=parts.path.removesuffix("/infer") + "/health", query="")
        url, headers = _split_credentials(urlunsplit(health))
        try:
            with _OPENER.open(urllib.request.Request(url, headers=headers), timeout=HEALTH_TIMEOUT):  # noqa: S310
                return None
        except urllib.error.HTTPError as exc:
            exc.close()
            return f"returned HTTP {exc.code}" if exc.code >= 500 else None
        except (urllib.error.URLError, http.client.HTTPException, TimeoutError, OSError) as exc:
            return f"is not answering: {getattr(exc, 'reason', exc)}"


SERVICE = RemoteService(URL_ENV, TIMEOUT_ENV, WORKERS_ENV, WatermarkClient, DEFAULT_TIMEOUT, DEFAULT_WORKERS)


def client() -> WatermarkClient | None:
    """A client for the configured server, or ``None`` when none is configured."""
    return SERVICE.client()


def configured() -> bool:
    return SERVICE.configured()


def upright(data: bytes) -> tuple[bytes, tuple[int, int]]:
    """*data* as an upright JPEG for the detector, and the displayed ``(width, height)``."""
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data)))
    size = img.size
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue(), size


Checked = tuple[list[WatermarkDetection], tuple[int, int]]


def detect_all(
    items: dict[str, bytes],
    progress: Callable[[int, int], None] | None = None,
    cancelled: Callable[[], None] | None = None,
) -> dict[str, Checked | str]:
    """``(hits, displayed size)`` for each of *items* (rel_path → source bytes), or why its check failed.

    Each image is made upright in the worker that sends it, so a scan never
    holds a second copy of every image. A file that will not open is reported
    like a failed check, but says nothing about the server.

    *cancelled* is called after each image and may raise to stop the pass.
    After ``MAX_FAILURES_WITHOUT_SUCCESS`` failures with no success the server
    is taken to be down and the rest are reported unchecked.
    """
    det = client()
    if det is None:
        raise WatermarkCheckFailed(f"{URL_ENV} is not set")

    def one(data: bytes) -> Checked | str:
        try:
            sent, size = upright(data)
        except Exception as exc:
            return f"cannot open: {exc}"
        return det.check(sent), size

    out: dict[str, Checked | str] = {}
    total, done, failed, succeeded = len(items), 0, 0, 0
    if progress is not None:
        progress(0, total)
    with ThreadPoolExecutor(max_workers=max(1, SERVICE.workers())) as pool:
        futures = {pool.submit(one, data): rel for rel, data in items.items()}
        try:
            for fut in as_completed(futures):
                rel = futures[fut]
                try:
                    checked = fut.result()
                    out[rel] = checked
                    if not isinstance(checked, str):
                        succeeded += 1
                except CancelledError:
                    out[rel] = "the watermark detector is unreachable"
                except Exception as exc:
                    failed += 1
                    out[rel] = str(exc)
                    logger.warning("watermark_check_failed", rel_path=rel, error=str(exc))
                    if not succeeded and failed == MAX_FAILURES_WITHOUT_SUCCESS:
                        logger.warning("watermark_server_unreachable", failed=failed, total=total)
                        for other in futures:
                            other.cancel()
                done += 1
                # Throttled as remote.check_all is: every 5 images (and the last) is plenty for a bar.
                if progress is not None and (done == total or done % 5 == 0):
                    progress(done, total)
                if cancelled is not None:
                    cancelled()
        except BaseException:
            for other in futures:
                other.cancel()
            raise
    return out
