"""Zero-shot profile fit and junk checks from the scan's CLIP embeddings (#197).

Each passing image's CLIP image vector (:mod:`argus_curator.embeddings`) is
compared with text-prompt vectors from the same checkpoint's text tower:

- **Profile fit**: how well the image reads as the kind of image the
  :class:`TargetProfile` asks for: the probability of the profile's category
  in one softmax over the four framing prompts (identity, wardrobe,
  pose_composition, setting). ``concept`` (a style LoRA) has no framing, so
  its fit is the probability of its style (photo or anime). The style is not
  folded into the other categories' fit: on real photos it was too unsure to
  weigh (docs/scoring.md).
- **Junk**: the probability that the image is a screenshot, a collage, a
  page of text or a meme, from one softmax over those prompts and a few
  ordinary-image prompts. ``junk_kind`` names the likeliest kind once the
  probability reaches ``ScanConfig.junk_threshold``.

Both feed the composite score (``scanner._target_bonus``) and neither rejects
an image on its own; ``PrefilterConfig.drop_junk`` is the opt-in rule that
does.

The prompt vectors are computed once, offline, by ``argus-curator
embeddings-convert --prompts`` and committed as ``data/clip_prompts.json``,
pinned by :data:`PROMPTS_SHA256`. Every profile prompt is in it (the
taxonomy has two styles and five categories), so no text model or tokenizer
runs at scan time. The image vectors stay in memory as before; only the
scores (plain floats and a label) land on the rows.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np

from argus_curator.embeddings import EMBEDDING_SPACE, normalise, space
from argus_curator.models import JUNK_KINDS, ImageResult, ScanConfig, TargetProfile, ZeroShotResult
from argus_curator.runtime import ModelUnavailable, sha256, softmax

PROMPTS_FILE = "clip_prompts.json"
# sha256 of data/clip_prompts.json. Regenerate the file with
# `argus-curator embeddings-convert --prompts` and update this with it.
PROMPTS_SHA256 = "0b6ce5a7c49541956e0f6b54e7812a969d920aad3e6729d5196ddf0d79a377ce"

# The prompts each label's vector averages (prompt ensembling, as in the CLIP
# paper). Changing one means regenerating the file: a test compares them. The
# wording was chosen on the calibration sets in docs/scoring.md.
CATEGORY_PROMPTS: dict[str, tuple[str, ...]] = {
    "identity": (
        "a close-up portrait of a person's face.",
        "a headshot of one person.",
        "a head and shoulders portrait.",
    ),
    "wardrobe": (
        "a full-length photo of a person showing their outfit.",
        "a person wearing a full outfit, head to toe.",
        "a fashion photo of clothing.",
    ),
    "pose_composition": (
        "a person in a full-body pose.",
        "a person posing, their whole body in frame.",
        "a dynamic action pose.",
    ),
    "setting": ("a place with no people in it.", "an interior of a room.", "a landscape.", "a building."),
}
# ``concept`` has no framing prompt: a style LoRA's fit is its style's.
PROFILE_CATEGORIES = tuple(CATEGORY_PROMPTS)
# The photo side names other non-anime looks too, so a painting or a render
# is not pushed onto anime for want of anything closer.
STYLE_PROMPTS: dict[str, tuple[str, ...]] = {
    "photo": (
        "a photo.",
        "a photograph.",
        "a realistic photo.",
        "a painting.",
        "a digital render.",
        "an old photograph.",
    ),
    "anime": ("an anime illustration.", "an anime drawing.", "a manga drawing.", "anime art."),
}

JUNK_PROMPTS: dict[str, tuple[str, ...]] = {
    "screenshot": (
        "a screenshot of a website.",
        "a screenshot of a phone app.",
        "a screenshot of a computer screen with windows and menus.",
        "a screen capture of a user interface.",
    ),
    "collage": (
        "a collage of several photos.",
        "a grid of photos.",
        "a contact sheet of thumbnail images.",
        "a photo montage made of many pictures.",
    ),
    "text": (
        "a page of text.",
        "a scanned document.",
        "a screenshot of a text document.",
        "a slide with bullet points.",
    ),
    "meme": (
        "a meme.",
        "an internet meme with a caption in bold white letters.",
        "a photo with meme text at the top and bottom.",
        "a reaction image with a funny caption.",
    ),
}
# What an ordinary image is, each its own label, for the junk softmax to weigh
# junk against. With only "a photo" and "a painting" here, archival prints and
# generated portraits read as junk (7.8% of clean images at 0.5; 1.1% with these).
CLEAN_PROMPTS: dict[str, tuple[str, ...]] = {
    "photo": ("a photo.", "a photograph."),
    "portrait": ("a portrait photo of a person.", "a photo of a person."),
    "people": ("a photo of people.", "a fashion photo of a model."),
    "place": ("a photo of a place.", "a landscape photo.", "a photo of a room."),
    "old": ("an old black and white photograph.", "a vintage portrait.", "a sepia photo."),
    "print": ("an engraving.", "an old printed portrait.", "a lithograph."),
    "art": ("a painting.", "an illustration.", "a drawing."),
    "anime": ("an anime illustration.", "a manga drawing."),
    "render": ("a digital render.", "a CGI image.", "digital art."),
}


def prompt_labels() -> dict[str, tuple[str, ...]]:
    """Every label the prompts file must hold, with the prompts its vector averages."""
    labels: dict[str, tuple[str, ...]] = {}
    for category, prompts in CATEGORY_PROMPTS.items():
        labels[f"category:{category}"] = prompts
    for style, prompts in STYLE_PROMPTS.items():
        labels[f"style:{style}"] = prompts
    for kind, prompts in JUNK_PROMPTS.items():
        labels[f"junk:{kind}"] = prompts
    for kind, prompts in CLEAN_PROMPTS.items():
        labels[f"clean:{kind}"] = prompts
    return labels


@dataclass(frozen=True)
class PromptVectors:
    """The pinned prompt vectors: unit rows by label, and CLIP's logit scale."""

    logit_scale: float
    vectors: Mapping[str, np.ndarray]

    def matrix(self, labels: list[str]) -> np.ndarray:
        return np.stack([self.vectors[label] for label in labels])


def _prompts_text() -> bytes:
    return resources.files("argus_curator").joinpath("data", PROMPTS_FILE).read_bytes()


@lru_cache(maxsize=1)
def prompt_vectors() -> PromptVectors:
    """Load and check the committed prompt vectors (once per process).

    Refuses a file whose sha256 is not :data:`PROMPTS_SHA256`, one computed in
    another embedding space, or one missing a label: a score against the wrong
    vectors would look plausible and mean nothing.
    """
    import hashlib

    raw = _prompts_text()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != PROMPTS_SHA256:
        raise ModelUnavailable(f"{PROMPTS_FILE} is not the pinned prompt set (sha256 {digest})")
    doc = json.loads(raw)
    sp = space()
    if doc.get("space") != EMBEDDING_SPACE or doc.get("weights_sha256") != sp.weights_sha256:
        raise ModelUnavailable(f"{PROMPTS_FILE} was not computed in the {EMBEDDING_SPACE} space")
    vectors = {label: np.asarray(entry["vector"], dtype=np.float32) for label, entry in doc["labels"].items()}
    missing = [label for label in prompt_labels() if label not in vectors]
    if missing or any(v.shape != (sp.dim,) for v in vectors.values()):
        raise ModelUnavailable(f"{PROMPTS_FILE} does not match the prompt set (missing {missing})")
    return PromptVectors(logit_scale=float(doc["logit_scale"]), vectors={k: normalise(v) for k, v in vectors.items()})


def profile_fit(vectors: np.ndarray, profile: TargetProfile, prompts: PromptVectors) -> np.ndarray:
    """``(N,)`` profile-fit probabilities for unit image *vectors* ``(N, dim)``.

    The profile's category against the other framings; for ``concept``, its
    style against the other style.
    """
    if profile.target_category == "concept":
        labels = [f"style:{s}" for s in STYLE_PROMPTS]
        own = f"style:{profile.target_style}"
    else:
        labels = [f"category:{c}" for c in PROFILE_CATEGORIES]
        own = f"category:{profile.target_category}"
    probs = softmax(prompts.logit_scale * vectors @ prompts.matrix(labels).T)
    return probs[:, labels.index(own)]


def junk(vectors: np.ndarray, prompts: PromptVectors) -> tuple[np.ndarray, list[str]]:
    """``(N,)`` junk probabilities and each image's likeliest junk kind."""
    junk_labels = [f"junk:{k}" for k in JUNK_KINDS]
    labels = junk_labels + [f"clean:{k}" for k in CLEAN_PROMPTS]
    probs = softmax(prompts.logit_scale * vectors @ prompts.matrix(labels).T)
    junk_probs = probs[:, : len(junk_labels)]
    kinds = [JUNK_KINDS[i] for i in junk_probs.argmax(axis=1)]
    return junk_probs.sum(axis=1), kinds


def wanted(cfg: ScanConfig, drop_junk: bool = False) -> bool:
    """Whether the scan asks for any zero-shot score."""
    return cfg.profile_fit or cfg.junk_check or drop_junk


def score_images(
    results: list[ImageResult],
    embeddings: Mapping[str, Any],
    profile: TargetProfile,
    cfg: ScanConfig,
    *,
    junk_check: bool | None = None,
    prompts: PromptVectors | None = None,
) -> int:
    """Set ``zero_shot`` on each passing image that has an embedding; return how many.

    *junk_check* overrides ``cfg.junk_check`` (``PrefilterConfig.drop_junk``
    needs the junk score even when the score itself was not asked for). An
    image with no embedding (it failed to decode, or its batch failed) keeps
    ``zero_shot`` None: unscored, not a perfect fit and not clean.
    """
    with_junk = cfg.junk_check if junk_check is None else junk_check
    if not (cfg.profile_fit or with_junk):
        return 0
    rows = [r for r in results if r.passed and embeddings.get(r.rel_path) is not None]
    if not rows:
        return 0
    prompts = prompts or prompt_vectors()
    vectors = normalise(np.stack([np.asarray(embeddings[r.rel_path], dtype=np.float32) for r in rows]))
    fits = profile_fit(vectors, profile, prompts) if cfg.profile_fit else None
    junk_probs, kinds = junk(vectors, prompts) if with_junk else (None, None)
    for i, row in enumerate(rows):
        junk_p = None if junk_probs is None else round(float(junk_probs[i]), 4)
        row.zero_shot = ZeroShotResult(
            profile_fit=None if fits is None else round(float(fits[i]), 4),
            junk=junk_p,
            junk_kind=kinds[i] if kinds is not None and junk_p is not None and junk_p >= cfg.junk_threshold else None,
        )
    return len(rows)


# ---------------------------------------------------------------------------
# One-time prompt export (embeddings-convert --prompts)
# ---------------------------------------------------------------------------

# The checkpoint files the text tower and its tokenizer need, at the pinned revision.
PROMPT_HF_FILES = (
    "config.json",
    "pytorch_model.bin",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
)


def default_prompts_path() -> Path:
    """Where ``--prompts`` writes by default: the package's own data file (a source checkout)."""
    return Path(__file__).parent / "data" / PROMPTS_FILE


def export_prompts(checkpoint_dir: Path, output: Path) -> tuple[Path, str]:
    """Embed every :func:`prompt_labels` prompt with the pinned text tower; write the file.

    Each label's vector is the normalised mean of its prompts' unit
    embeddings. Values are rounded to 6 decimals, so the file (and its
    sha256) is stable across runs. Returns the path and its sha256, which
    :data:`PROMPTS_SHA256` must then name. Needs ``argus-curator[embeddings-convert]``.
    """
    try:
        import torch
        from transformers import CLIPConfig, CLIPTextModelWithProjection, CLIPTokenizer
    except Exception as exc:
        raise ModelUnavailable("Conversion requires: pip install argus-curator[embeddings-convert]") from exc

    sp = space()
    weights = checkpoint_dir / "pytorch_model.bin"
    digest = sha256(weights)
    if digest != sp.weights_sha256:
        raise ModelUnavailable(f"{weights} is not the pinned {sp.hf_repo} checkpoint (sha256 {digest})")
    config = CLIPConfig.from_pretrained(str(checkpoint_dir))
    text = config.text_config
    text.projection_dim = config.projection_dim  # as for the image tower: the top-level one is real
    if text.projection_dim != sp.dim:
        raise ModelUnavailable(f"{sp.hf_repo} projects to {text.projection_dim}, not {sp.dim}")
    model = CLIPTextModelWithProjection(text)
    state = torch.load(str(weights), map_location="cpu", weights_only=True)
    logit_scale = float(state["logit_scale"].exp())
    state = {
        k: v
        for k, v in state.items()
        if k.startswith(("text_model.", "text_projection.")) and not k.endswith("position_ids")
    }
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if not k.endswith("position_ids")]
    if missing or unexpected:
        raise ModelUnavailable(f"checkpoint does not match the text tower (missing {missing}, unexpected {unexpected})")
    model.eval()
    tokenizer = CLIPTokenizer.from_pretrained(str(checkpoint_dir))

    labels: dict[str, Any] = {}
    with torch.no_grad():
        for label, prompts in prompt_labels().items():
            tokens = tokenizer(list(prompts), padding=True, return_tensors="pt")
            embeds = normalise(model(**tokens).text_embeds.numpy())
            mean = normalise(embeds.mean(axis=0))
            labels[label] = {"prompts": list(prompts), "vector": [round(float(x), 6) for x in mean]}
    doc = {
        "space": EMBEDDING_SPACE,
        "hf_repo": sp.hf_repo,
        "hf_revision": sp.hf_revision,
        "weights_sha256": sp.weights_sha256,
        "logit_scale": round(logit_scale, 4),
        "labels": labels,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(doc, separators=(",", ":")) + "\n")
    return output, sha256(output)
