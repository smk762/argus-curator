"""Shared fixtures: a synthetic on-disk dataset with sharp/blurry/dup images."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageFilter

# Every env var that feeds server/store configuration. These are exactly the
# ones docker-compose.yaml and the README tell operators to export, so a dev
# box or CI job with them set would otherwise reconfigure the app under test —
# flipping the "refuses without a root" assertions and letting a failing upload
# test write real files into the ambient source root.
_CONFIG_ENV = (
    "ARGUS_CURATOR_SCAN_ROOT",
    "ARGUS_CURATOR_EXPORT_ROOT",
    "CURATOR_SOURCE_PATH",
    "CURATOR_EXPORT_PATH",
    "CURATOR_ALLOW_MOVE",
    "CURATOR_CORS_ORIGINS",
    "CURATOR_TRUSTED_HOSTS",
    "CURATOR_CACHE_DIR",
    "CURATOR_MAX_UPLOAD_MB",
    "CURATOR_CAPTION_ALLOWED_HOSTS",
    "ARGUS_CURATOR_FAIRFACE_MODEL",
    "ARGUS_CURATOR_MIVOLO_MODEL",
    "ARGUS_CURATOR_PLACES365_MODEL",
    "ARGUS_CURATOR_GARMENTS_MODEL",
    # A real NudeNet server in the shell would turn the opt-out check on for
    # every scan under test.
    "ARGUS_CURATOR_NUDENET_URL",
    "ARGUS_CURATOR_NUDENET_TIMEOUT",
    "ARGUS_CURATOR_NUDENET_WORKERS",
    "ARGUS_CURATOR_SINGGUARD_URL",
    "ARGUS_CURATOR_SINGGUARD_TIMEOUT",
    "ARGUS_CURATOR_SINGGUARD_WORKERS",
    "ARGUS_CURATOR_SINGGUARD_THINKING",
    "ARGUS_CURATOR_SINGGUARD_THINKING_ALLOWED",
    "ARGUS_CURATOR_SINGGUARD_POLICY_FILE",
    "ARGUS_CURATOR_WATERMARK_URL",
    "ARGUS_CURATOR_WATERMARK_TIMEOUT",
    "ARGUS_CURATOR_WATERMARK_WORKERS",
    # The body-pose stage is opt-out like NudeNet: a puppet URL in the shell would turn it on.
    "ARGUS_CURATOR_PUPPET_URL",
    "ARGUS_CURATOR_PUPPET_TIMEOUT",
    "ARGUS_CURATOR_PUPPET_WORKERS",
    "ARGUS_CURATOR_PUPPET_MAX_SET",
    "ARGUS_CURATOR_PUPPET_MAX_JSON_BYTES",
    # The under-18 rule; tests pick a mode explicitly where it matters.
    "ARGUS_CURATOR_MINOR_EXPOSED",
    # The GPU coordinator: a broker URL alone selects argus-iris (cortex 0.15), and
    # the server would register with it at startup.
    "ARGUS_GPU_COORDINATOR",
    "ARGUS_GPU_BROKER_URL",
    "ARGUS_GPU_BROKER_TOKEN",
    "ARGUS_GPU_TENANT",
    "ARGUS_GPU_TENANT_MANIFEST",
    # The suite-wide peer URLs (argus_cortex.peers): each is the fallback for curator's own variable above, so
    # one in the shell would configure that server for every scan under test.
    "ARGUS_NUDENET_URL",
    "ARGUS_ERAX_URL",
    "ARGUS_WATERMARK_URL",
    "ARGUS_SINGGUARD_URL",
    "ARGUS_PUPPET_URL",
    # What a container is told about itself; /health reports them.
    "ARGUS_BUILD",
    "ARGUS_PEER",
)


@pytest.fixture(autouse=True)
def hermetic_env(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory) -> None:
    """Config comes from create_app arguments only, never the ambient shell.

    Models too: with the env vars cleared, a model resolves under MODEL_DIR, and
    a dev box that has run ``mivolo-convert`` has one there.
    """
    for name in _CONFIG_ENV:
        monkeypatch.delenv(name, raising=False)
    # A developer's suite file (~/.config/argus/suite.env) is never read: it holds real URLs and tokens.
    monkeypatch.setenv("ARGUS_ENV_FILE", "off")
    # The CLI saves each scan to the default store. Unset, that is
    # ~/.cache/argus_curator/scans, which a running curator serves from, so its
    # scan list filled with pytest folders.
    monkeypatch.setenv("CURATOR_CACHE_DIR", str(tmp_path_factory.mktemp("scans")))
    from argus_curator import runtime

    monkeypatch.setattr(runtime, "MODEL_DIR", tmp_path_factory.mktemp("models"))
    # Built once per process from the environment; each test builds its own.
    monkeypatch.setattr(runtime, "_coordinator", None)
    monkeypatch.setattr(runtime, "_capped_cache", {})


def _noise_image(size: int = 768, seed: int = 0) -> Image.Image:
    """A high-frequency noise image — sharp, passes the blur filter."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    """Build a nested dataset and return its root.

    Layout (rel_path keying matters — same basename in two sub-folders):
      personA/s1/img.png   sharp, unique
      personA/s2/img.png   sharp, unique (same basename, different folder)
      personA/s1/dup.png   near-duplicate of img.png
      personB/blurry.png   blurry -> rejected
      personB/tiny.png     low-res -> rejected
    """
    root = tmp_path / "raw"

    a1 = root / "personA" / "s1"
    a2 = root / "personA" / "s2"
    b = root / "personB"
    for d in (a1, a2, b):
        d.mkdir(parents=True, exist_ok=True)

    base = _noise_image(seed=1)
    base.save(a1 / "img.png")

    # Near-duplicate: same base with a tiny tweak.
    dup = base.copy()
    dup.putpixel((0, 0), (0, 0, 0))
    dup.save(a1 / "dup.png")

    # Different sharp image, same basename in another sub-folder.
    _noise_image(seed=2).save(a2 / "img.png")

    # Blurry -> rejected by blur_threshold.
    blurry = _noise_image(seed=3).filter(ImageFilter.GaussianBlur(radius=8))
    blurry.save(b / "blurry.png")

    # Too small -> rejected by min_short_side.
    _noise_image(size=128, seed=4).save(b / "tiny.png")

    return root
