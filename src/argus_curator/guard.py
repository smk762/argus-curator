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
The HTTP client is argus-cortex's :class:`~argus_cortex.moderation.SingGuardClient`,
which also downscales images over 1536 px before sending.

"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from argus_cortex.moderation import (
    THINKING_MODES,
    GuardCheckFailed,  # noqa: F401 - re-exported
    ModerationConfigError,
    SingGuardClient,
    policy_name,
    rule_letter,  # noqa: F401 - re-exported for the CLI
    rule_matches,  # noqa: F401 - re-exported for selection
)

from argus_curator.models import GuardConfig, ImageResult
from argus_curator.remote import check_all, env_set, env_value, float_env
from argus_curator.runtime import ModelUnavailable

URL_ENV = "ARGUS_CURATOR_SINGGUARD_URL"
TIMEOUT_ENV = "ARGUS_CURATOR_SINGGUARD_TIMEOUT"
WORKERS_ENV = "ARGUS_CURATOR_SINGGUARD_WORKERS"
THINKING_ENV = "ARGUS_CURATOR_SINGGUARD_THINKING"
POLICY_FILE_ENV = "ARGUS_CURATOR_SINGGUARD_POLICY_FILE"
DEFAULT_TIMEOUT = 90.0
DEFAULT_WORKERS = 2


def endpoint() -> str | None:
    """The configured ``/api/guard/check`` URL, or ``None`` when SingGuard is not configured.

    Checks the URL only; :func:`thinking` and :func:`policy` check the rest.
    """
    base = env_value(URL_ENV)
    if base is None:
        return None
    try:
        return SingGuardClient(base).url
    except ModerationConfigError as exc:
        raise ModelUnavailable(f"{URL_ENV} must be an http(s) URL, got {base!r}") from exc


def configured() -> bool:
    return env_set(URL_ENV)


def resolve_enabled(cfg: GuardConfig) -> bool:
    """Whether a scan with *cfg* checks images: explicit choice, else configured."""
    return cfg.enabled if cfg.enabled is not None else configured()


def thinking() -> str:
    mode = env_value(THINKING_ENV) or "fast"
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
    return policy_name(text), text


def client() -> SingGuardClient | None:
    """A client for the configured server and policy, or ``None`` when SingGuard is not configured."""
    url = endpoint()
    if url is None:
        return None
    _name, text = policy()
    return SingGuardClient(url, timeout=float_env(TIMEOUT_ENV, DEFAULT_TIMEOUT), thinking=thinking(), policy=text)


def check_images(
    results: list[ImageResult],
    items: list[tuple[str, str, bytes]],
    cfg: GuardConfig,
    progress: Callable[[int, int], None] | None = None,
    on_done: Callable[..., None] | None = None,
) -> bool:
    """Set ``guard`` on every image (see :func:`remote.check_all`); return whether
    the check ran at all."""
    if not resolve_enabled(cfg):
        return False
    guard = client()
    if guard is None:
        raise ModelUnavailable(f"SingGuard check requested but {URL_ENV} is not set")
    check_all(
        results,
        items,
        lambda data, _rel: guard.check(data),
        "guard",
        int(float_env(WORKERS_ENV, DEFAULT_WORKERS)),
        progress,
        on_done=on_done,
    )
    return True
