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
several with ``fast-slow`` or ``slow``. The default is ``cascade`` (``fast``
for every image, ``slow`` for what ``fast`` flags and for a MiVOLO age under
18 with an exposed NudeNet score of at least 0.2).

``ARGUS_CURATOR_SINGGUARD_CONTEXT`` sends what the scan already knows about an
image (its MiVOLO age, its folder, its caption sidecar, and optionally scene,
NudeNet, garments) as the request text, framed as recorded context rather than
a request; see :mod:`guard_context`.

Configured by environment only, like NudeNet (see :mod:`argus_curator.remote`).
The HTTP client is argus-cortex's :class:`~argus_cortex.moderation.SingGuardClient`,
which also downscales images over 1536 px before sending.

"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from pathlib import Path

import structlog
from argus_cortex.moderation import (
    THINKING_MODES,
    GuardCheckFailed,  # noqa: F401 - re-exported
    ModerationConfigError,
    SingGuardClient,
    policy_name,
    rule_letter,  # noqa: F401 - re-exported for the CLI
    rule_matches,  # noqa: F401 - re-exported
)
from argus_cortex.moderation.nudenet import MIN_REPORTED_SCORE
from argus_cortex.moderation.policies import POLICIES
from argus_cortex.moderation.policies import policy_text as cortex_policy_text

from argus_curator.guard_context import CONTEXT_KINDS, body_only_ages, image_context, mivolo_ages
from argus_curator.models import GuardConfig, GuardPolicy, GuardResult, ImageResult
from argus_curator.nsfw import exposed_score
from argus_curator.prefilter import is_prefiltered
from argus_curator.remote import check_all, env_set, env_value, float_env
from argus_curator.runtime import ModelUnavailable
from argus_curator.selection import minor_exposed

URL_ENV = "ARGUS_CURATOR_SINGGUARD_URL"
TIMEOUT_ENV = "ARGUS_CURATOR_SINGGUARD_TIMEOUT"
WORKERS_ENV = "ARGUS_CURATOR_SINGGUARD_WORKERS"
THINKING_ENV = "ARGUS_CURATOR_SINGGUARD_THINKING"
POLICY_FILE_ENV = "ARGUS_CURATOR_SINGGUARD_POLICY_FILE"
CONTEXT_ENV = "ARGUS_CURATOR_SINGGUARD_CONTEXT"
DEFAULT_TIMEOUT = 90.0
DEFAULT_WORKERS = 2

# The policies a request can name, in the order /detectors lists them, and
# their texts: argus-cortex's, shared with argus-darkroom. Adult variants
# besides "adult" are under evaluation against "adult" (scripts/guard_eval.py);
# keep them out of UIs until one is chosen. Built from cortex so a new name
# such as "adult-cinematic" is offered as soon as cortex ships it.
_POLICY_TEXT: dict[str, str] = {name: text for name in POLICIES if (text := cortex_policy_text(name))}
ADULT_POLICY = _POLICY_TEXT["adult"]
ADULT_GRANULAR_POLICY = _POLICY_TEXT["adult-granular"]
ADULT_CAUTIOUS_POLICY = _POLICY_TEXT["adult-cautious"]
ADULT_CINEMATIC_POLICY = _POLICY_TEXT.get("adult-cinematic")


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
    mode = env_value(THINKING_ENV) or "cascade"
    if mode not in THINKING_MODES:
        raise ModelUnavailable(f"{THINKING_ENV} must be one of {', '.join(THINKING_MODES)}, got {mode!r}")
    return mode


def context_kinds() -> frozenset[str]:
    """The context ``ARGUS_CURATOR_SINGGUARD_CONTEXT`` asks to send with each image (see :mod:`guard_context`)."""
    raw = env_value(CONTEXT_ENV)
    if raw is None:
        return frozenset()
    kinds = frozenset(part.strip().lower() for part in raw.split(",") if part.strip())
    unknown = kinds - CONTEXT_KINDS
    if unknown:
        raise ModelUnavailable(
            f"{CONTEXT_ENV} must name only {', '.join(sorted(CONTEXT_KINDS))}, got {', '.join(sorted(unknown))}"
        )
    return kinds


def policy(name: GuardPolicy = "default") -> tuple[str, str | None]:
    """``(recorded name, text)`` of policy *name*; text ``None`` = SingGuard's default.

    ``default`` records as ``default`` or ``custom-<sha256 prefix>`` of the
    operator's policy file; a built-in policy records as its own name.
    """
    if name in _POLICY_TEXT:
        return name, _POLICY_TEXT[name]
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


def describe_policies() -> list[dict[str, str | None]]:
    """Every policy a scan can name, with the text it sends and the name results record.

    ``source`` is ``builtin``, ``file`` (``default`` reads the operator's policy
    file) or ``singguard`` (``default`` sends no text, so SingGuard applies its
    own rules and ``text`` is ``None``). A policy file that cannot be read is
    listed with ``text`` ``None`` and an ``error``, as a scan naming it would fail.
    """
    described: list[dict[str, str | None]] = []
    for name in POLICIES:
        source = "builtin" if name in _POLICY_TEXT else "file" if env_value(POLICY_FILE_ENV) else "singguard"
        try:
            recorded, text = policy(name)
        except ModelUnavailable as exc:  # names the env var, never the path
            described.append({"name": name, "recorded": None, "source": source, "text": None, "error": str(exc)})
            continue
        described.append({"name": name, "recorded": recorded, "source": source, "text": text, "error": None})
    return described


logger = structlog.get_logger()

# Modes only some servers take (argus-cull's and agent-composer's singguard);
# a stock SingGuard server answers HTTP 400 to them.
EXTENDED_MODES = frozenset({"slow", "cascade"})
FALLBACK_THINKING = "fast"


class FallbackClient:
    """A SingGuard client that asks for an extended mode and drops to ``fast`` when the server refuses it.

    On HTTP 400 the same check is retried in ``fast``; only if that succeeds is the
    server taken not to know the mode, and every later check goes straight to
    ``fast``. A 400 that ``fast`` also gets (a bad image, a refused policy) is
    raised as it came, and the mode is kept. A verdict judged in the fallback says
    ``thinking: "fast"``, so a scan records the mode that actually judged it.
    """

    def __init__(self, url: str, timeout: float, mode: str, policy_text: str | None) -> None:
        self.primary = SingGuardClient(url, timeout=timeout, thinking=mode, policy=policy_text)
        self.fallback = (
            SingGuardClient(url, timeout=timeout, thinking=FALLBACK_THINKING, policy=policy_text)
            if mode in EXTENDED_MODES
            else None
        )
        self.fell_back = False
        self._lock = threading.Lock()

    @property
    def thinking(self) -> str:
        return FALLBACK_THINKING if self.fell_back else self.primary.thinking

    def check(self, data: bytes, text: str = "") -> GuardResult:
        if self.fell_back and self.fallback is not None:
            return self._fallback_check(data, text)
        try:
            return self.primary.check(data, text=text)
        except GuardCheckFailed as exc:
            if self.fallback is None or "HTTP 400" not in str(exc):
                raise
            try:
                result = self._fallback_check(data, text)
            except GuardCheckFailed:
                raise exc from None
            with self._lock:
                if not self.fell_back:
                    self.fell_back = True
                    logger.warning("singguard_mode_unsupported", asked=self.primary.thinking, using=FALLBACK_THINKING)
            return result

    def _fallback_check(self, data: bytes, text: str) -> GuardResult:
        result = self.fallback.check(data, text=text)  # type: ignore[union-attr]
        return result if result.thinking else result.model_copy(update={"thinking": FALLBACK_THINKING})


def client(name: GuardPolicy = "default", *, mode: str | None = None) -> FallbackClient | None:
    """A client for the configured server and policy *name*, or ``None`` when SingGuard is not configured.

    *mode* overrides :func:`thinking` for one client: cascade sends most frames
    in ``cascade``, and the under-18 + exposure pairing in ``slow``.
    """
    url = endpoint()
    if url is None:
        return None
    _recorded, text = policy(name)
    return FallbackClient(url, float_env(TIMEOUT_ENV, DEFAULT_TIMEOUT), mode or thinking(), text)


def force_slow(result: ImageResult, mode: str) -> bool:
    """Under cascade, skip the fast screen when a young age shares the frame with exposure.

    Fast is only there to save work. A MiVOLO age under 18 with an exposed
    NudeNet score of at least 0.2 already holds and deletes the frame, so it
    is cheap to send it to slow. Measured faces and body-only ages go through
    :func:`minor_exposed`. A faceless frame that took the set's age uses the
    same pairing.
    """
    if mode != "cascade":
        return False
    if minor_exposed(result):
        return True
    if mivolo_ages(result) or body_only_ages(result):
        return False
    if result.set_age is None or result.set_age.years >= 18:
        return False
    if result.nsfw is None:
        return False
    return exposed_score(result.nsfw.labels) >= MIN_REPORTED_SCORE


def check_images(
    results: list[ImageResult],
    items: list[tuple[str, str, bytes]],
    cfg: GuardConfig,
    progress: Callable[[int, int], None] | None = None,
    on_done: Callable[..., None] | None = None,
) -> bool:
    """Set ``guard`` on every image (see :func:`remote.check_all`); return whether
    the check ran at all.

    With ``cfg.variants``, every image is then judged again under each variant,
    one pass per policy, and ``guard_variants`` holds every verdict by policy
    name, the official one included. Progress counts images times policies.

    Images the scan prefilter dropped are not judged (nor counted); other
    rejects are, since a blurry explicit image is still explicit.
    """
    if not resolve_enabled(cfg):
        return False
    results = [r for r in results if not is_prefiltered(r)]
    names = [cfg.policy, *cfg.variants]
    clients = {name: client(name) for name in names}
    if clients[cfg.policy] is None:
        raise ModelUnavailable(f"SingGuard check requested but {URL_ENV} is not set")
    mode = thinking()
    # Cascade's fast pass is a screen. Under-18 + exposure skips it and is judged slow.
    slow_clients = {name: client(name, mode="slow") for name in names} if mode == "cascade" else {}
    by_rel = {r.rel_path: r for r in results}
    workers = int(float_env(WORKERS_ENV, DEFAULT_WORKERS))
    has_data = {rel for rel, _abs, data in items if data is not None}
    pending = sum(1 for r in results if r.rel_path in has_data)
    grand_total = pending * len(names)
    # The same context goes with the image under every policy, built once.
    kinds = context_kinds()
    contexts = {r.rel_path: image_context(r, kinds) for r in results if r.rel_path in has_data} if kinds else {}

    for offset, name in enumerate(names):
        guard = clients[name]
        slow = slow_clients.get(name)
        # The client names a policy by its text's hash; record the name asked for.
        recorded, _text = policy(name)
        official = name == cfg.policy
        base = offset * pending

        def store(r: ImageResult, value: GuardResult, name: str = name, official: bool = official) -> None:
            if official:
                r.guard = value
            if cfg.variants:
                r.guard_variants = {**(r.guard_variants or {}), name: value}

        def check(
            data: bytes,
            rel: str,
            guard: FallbackClient | None = guard,
            slow: FallbackClient | None = slow,
            recorded: str = recorded,
            mode: str = mode,
        ) -> GuardResult:
            row = by_rel.get(rel)
            chosen = slow if slow is not None and row is not None and force_slow(row, mode) else guard
            assert chosen is not None
            return chosen.check(data, text=contexts.get(rel, "")).model_copy(update={"policy": recorded})

        check_all(
            results,
            items,
            check,
            "guard" if official else f"guard:{name}",
            workers,
            (lambda done, _total, base=base: progress(base + done, grand_total)) if progress else None,
            on_done=(lambda r, done, _total, base=base: on_done(r, base + done, grand_total)) if on_done else None,
            store=store,
        )
    return True
