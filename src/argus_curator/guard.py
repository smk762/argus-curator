"""SingGuard policy check through a remote SingGuard server.

Where NudeNet scores nudity per body part, SingGuard judges an image against a
whole moderation policy and answers with a verdict and the rule that fired.
Its default rules are A. Sexual Content Risk, B. Real-World Crimes & Public
Safety, C. Unethical Behavior (hate, self-harm, gore), D. Cybersecurity,
E. Agent Safety, F. Politically Sensitive and G. Animal Abuse. A policy file
(``ARGUS_CURATOR_SINGGUARD_POLICY_FILE``) replaces them, e.g. one that allows
consensual adult nudity but still flags apparent minors.

SingGuard gives no scores, so its filters are by verdict and by rule. It runs
a vision language model: expect about a second per image in ``fast`` mode and
several with ``fast-slow``, which also writes why the rule fired.

Configured by environment only, like NudeNet (see :mod:`argus_curator.remote`).

API: ``POST /api/guard/check`` with JSON ``{input, image_base64, policy,
thinking}``, answered by ``{safe, verdict, rule, explanation, categories:
[{code, label, why}], model, elapsed_ms}``. ``input`` is the text judged with
the image; curator has none and sends it empty.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
from collections.abc import Callable
from pathlib import Path

from PIL import Image, ImageOps

from argus_curator.models import GuardConfig, GuardResult, ImageResult
from argus_curator.remote import CheckFailed, check_all, env_set, float_env, post_json, service_url
from argus_curator.runtime import ModelUnavailable

URL_ENV = "ARGUS_CURATOR_SINGGUARD_URL"
TIMEOUT_ENV = "ARGUS_CURATOR_SINGGUARD_TIMEOUT"
WORKERS_ENV = "ARGUS_CURATOR_SINGGUARD_WORKERS"
THINKING_ENV = "ARGUS_CURATOR_SINGGUARD_THINKING"
POLICY_FILE_ENV = "ARGUS_CURATOR_SINGGUARD_POLICY_FILE"
DEFAULT_TIMEOUT = 90.0
DEFAULT_WORKERS = 2
THINKING_MODES = ("fast", "fast-slow")

# Larger images are downscaled before sending: the guard model sees far fewer
# pixels than a camera original, and base64 JSON inflates every byte by 4/3.
MAX_EDGE = 1536
MAX_BYTES = 4 * 1024 * 1024
_EXPLANATION_MAX = 4000
_FIELD_MAX = 200


class GuardCheckFailed(CheckFailed):
    """One image could not be checked by SingGuard."""


def endpoint() -> str | None:
    """The configured ``/api/guard/check`` URL, or ``None`` when SingGuard is not configured."""
    return service_url(URL_ENV, "/api/guard/check")


def configured() -> bool:
    return env_set(URL_ENV)


def resolve_enabled(cfg: GuardConfig) -> bool:
    """Whether a scan with *cfg* checks images: explicit choice, else configured."""
    return cfg.enabled if cfg.enabled is not None else configured()


def thinking() -> str:
    mode = os.environ.get(THINKING_ENV, "").strip() or "fast"
    if mode not in THINKING_MODES:
        raise ModelUnavailable(f"{THINKING_ENV} must be one of {', '.join(THINKING_MODES)}, got {mode!r}")
    return mode


def policy() -> tuple[str, str | None]:
    """``(name, text)`` of the policy to judge against; text ``None`` = SingGuard's default."""
    path = os.environ.get(POLICY_FILE_ENV, "").strip()
    if not path:
        return "default", None
    try:
        text = Path(path).read_text(encoding="utf-8").strip()
    except OSError as exc:
        # The path is operator configuration; the server echoes this message.
        raise ModelUnavailable(f"{POLICY_FILE_ENV} cannot be read ({exc.strerror})") from exc
    if not text:
        raise ModelUnavailable(f"{POLICY_FILE_ENV} is empty (unset it to use SingGuard's default rules)")
    return f"custom-{hashlib.sha256(text.encode()).hexdigest()[:12]}", text


def _prepare(data: bytes) -> bytes:
    """*data* as is when small enough, else downscaled to a JPEG (a PNG when it has alpha)."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            if max(im.size) <= MAX_EDGE and len(data) <= MAX_BYTES:
                return data
            # Upright, as a viewer would see it: re-encoding drops the EXIF tag.
            out = ImageOps.exif_transpose(im)
            out.thumbnail((MAX_EDGE, MAX_EDGE))
            buf = io.BytesIO()
            if out.mode in ("RGBA", "LA", "PA") or (out.mode == "P" and "transparency" in out.info):
                # Flattening onto black would hide dark content drawn on a
                # transparent background; keep the alpha, as a small PNG goes.
                out.save(buf, "PNG", optimize=True)
            else:
                out.convert("RGB").save(buf, "JPEG", quality=90)
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise GuardCheckFailed(f"image could not be prepared for SingGuard: {exc}") from exc
    return buf.getvalue()


def _clip(value: object, limit: int) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()[:limit]


def to_result(payload: object, policy_name: str) -> GuardResult:
    """SingGuard's answer as recorded. A missing verdict is a failure, not a pass."""
    if not isinstance(payload, dict) or not isinstance(payload.get("safe"), bool):
        raise GuardCheckFailed("SingGuard returned no verdict")
    safe = payload["safe"]
    rules: list[str] = []
    if not safe:
        categories = payload.get("categories")
        for cat in categories if isinstance(categories, list) else []:
            code = _clip(cat.get("code"), _FIELD_MAX) if isinstance(cat, dict) else None
            if code and code not in rules:
                rules.append(code)
        rule = _clip(payload.get("rule"), _FIELD_MAX)
        if not rules and rule:
            rules.append(rule)
    return GuardResult(
        safe=safe,
        rules=rules,
        explanation=_clip(payload.get("explanation"), _EXPLANATION_MAX),
        model=_clip(payload.get("model"), _FIELD_MAX),
        policy=policy_name,
    )


def rule_letter(rule: str) -> str:
    """*rule* by its letter (``"A"`` for ``"A. Sexual Content Risk"``), else in full."""
    head, sep, _ = rule.strip().partition(".")
    return head if sep and len(head) == 1 else rule


def rule_matches(fired: str, wanted: str) -> bool:
    """*wanted* names *fired*: in full, or by its letter (``"A"`` or ``"A."`` for ``"A. Sexual Content Risk"``)."""
    fired, wanted = fired.strip().casefold(), wanted.strip().casefold()
    return fired == wanted or fired.startswith(f"{wanted.removesuffix('.')}.")


def check(data: bytes, url: str, timeout: float, mode: str, policy_name: str, policy_text: str | None) -> GuardResult:
    """Post one image to the SingGuard server and parse its answer."""
    body: dict[str, str] = {
        "input": "",
        "image_base64": base64.b64encode(_prepare(data)).decode("ascii"),
        "thinking": mode,
    }
    if policy_text is not None:
        body["policy"] = policy_text
    payload = post_json(url, json.dumps(body).encode(), "application/json", timeout, "SingGuard", GuardCheckFailed)
    return to_result(payload, policy_name)


def check_images(
    results: list[ImageResult],
    items: list[tuple[str, str, bytes]],
    cfg: GuardConfig,
    progress: Callable[[int, int], None] | None = None,
) -> bool:
    """Set ``guard`` on every image (see :func:`remote.check_all`); return whether
    the check ran at all."""
    if not resolve_enabled(cfg):
        return False
    url = endpoint()
    if url is None:
        raise ModelUnavailable(f"SingGuard check requested but {URL_ENV} is not set")
    mode = thinking()
    policy_name, policy_text = policy()
    timeout = float_env(TIMEOUT_ENV, DEFAULT_TIMEOUT)
    check_all(
        results,
        items,
        lambda data, _rel: check(data, url, timeout, mode, policy_name, policy_text),
        "guard",
        int(float_env(WORKERS_ENV, DEFAULT_WORKERS)),
        progress,
    )
    return True
