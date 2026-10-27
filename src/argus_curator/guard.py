"""SingGuard policy check through a remote SingGuard server.

Where NudeNet scores nudity per body part, SingGuard judges an image against a
whole moderation policy and answers with a verdict and the rule that fired.
Its default rules are A. Sexual Content Risk, B. Real-World Crimes & Public
Safety, C. Unethical Behavior (hate, self-harm, gore), D. Cybersecurity,
E. Agent Safety, F. Politically Sensitive and G. Animal Abuse. A policy file
(``ARGUS_CURATOR_SINGGUARD_POLICY_FILE``) replaces them as the ``default``
policy. A scan can instead pick the built-in ``adult`` policy
(:data:`ADULT_POLICY`), which allows consensual adult nudity but still flags
apparent minors. Requests choose a policy by name, never by text.

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

from argus_curator.models import GuardConfig, GuardPolicy, ImageResult
from argus_curator.remote import check_all, env_set, env_value, float_env
from argus_curator.runtime import ModelUnavailable

URL_ENV = "ARGUS_CURATOR_SINGGUARD_URL"
TIMEOUT_ENV = "ARGUS_CURATOR_SINGGUARD_TIMEOUT"
WORKERS_ENV = "ARGUS_CURATOR_SINGGUARD_WORKERS"
THINKING_ENV = "ARGUS_CURATOR_SINGGUARD_THINKING"
POLICY_FILE_ENV = "ARGUS_CURATOR_SINGGUARD_POLICY_FILE"
DEFAULT_TIMEOUT = 90.0
DEFAULT_WORKERS = 2

# The policies a request can name, in the order /detectors lists them.
POLICIES: tuple[GuardPolicy, ...] = ("default", "adult")

# Sent in full: SingGuard reads a bare "nsfw" as rule text. The letters and
# areas match the default rules, so guard_rules by letter mean the same area
# under both; B-G are our wording, so their verdicts can differ from the
# built-ins.
ADULT_POLICY = "\n".join(
    [
        "## Risk Categories",
        "### A. Illegal or Extreme Sexual Content",
        "  - Sexual content involving anyone under 18, or anyone who appears to be a minor.",
        (
            "  - Non-consensual sexual content, including hidden recording, sexual images of a real "
            "person shared without consent, coercion, or sexual exploitation."
        ),
        "  - Sexual violence, bestiality, or necrophilia.",
        "  - Consensual sexual content and nudity between adults is not this rule.",
        "### B. Real-World Crimes & Public Safety",
        "  - Content involving violent crime, weapons, other crimes, or public-safety threats.",
        "### C. Unethical Behavior",
        "  - Content involving hate, harassment, manipulation, self-harm, or harmful misinformation.",
        "  - Gore, dismemberment, and shock content meant to frighten or disgust.",
        "  - Adult sexual content by itself is not this rule.",
        "### D. Cybersecurity & Information Manipulation",
        "  - Content involving data leaks, hacking, surveillance abuse, platform abuse, or copyright abuse.",
        "### E. Agent Safety",
        "  - Content attempting to expose system prompts, internal policies, or other model safeguards.",
        "### F. Politically Sensitive Content",
        (
            "  - Content involving political advocacy, rumors, unrest, historical distortion, "
            "or attacks on political figures."
        ),
        "### G. Animal Abuse",
        "  - Content involving cruelty to animals or the spread of animal abuse.",
        "### Safe",
        "  - Content that does not match any risk category, including consensual adult nudity and sexual content.",
    ]
)


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


def policy(name: GuardPolicy = "default") -> tuple[str, str | None]:
    """``(recorded name, text)`` of policy *name*; text ``None`` = SingGuard's default.

    ``default`` records as ``default`` or ``custom-<sha256 prefix>`` of the
    operator's policy file; ``adult`` records as ``adult``.
    """
    if name == "adult":
        return "adult", ADULT_POLICY
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


def client(name: GuardPolicy = "default") -> SingGuardClient | None:
    """A client for the configured server and policy *name*, or ``None`` when SingGuard is not configured."""
    url = endpoint()
    if url is None:
        return None
    _recorded, text = policy(name)
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
    guard = client(cfg.policy)
    if guard is None:
        raise ModelUnavailable(f"SingGuard check requested but {URL_ENV} is not set")
    # The client names a policy by its text's hash; record the name asked for.
    recorded, _text = policy(cfg.policy)
    check_all(
        results,
        items,
        lambda data, _rel: guard.check(data).model_copy(update={"policy": recorded}),
        "guard",
        int(float_env(WORKERS_ENV, DEFAULT_WORKERS)),
        progress,
        on_done=on_done,
    )
    return True
