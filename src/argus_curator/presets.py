"""Training objective presets — tune CurateConfig defaults per LoRA goal."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from argus_curator.types import CurateConfig


def apply_preset(cfg: "CurateConfig") -> "CurateConfig":
    """Mutate *cfg* in-place with objective-specific defaults and return it."""
    obj = cfg.objective.lower().strip()

    if obj == "identity":
        _identity(cfg)
    elif obj == "style":
        _style(cfg)
    elif obj == "wardrobe":
        _wardrobe(cfg)
    elif obj == "concept":
        _concept(cfg)
    # unknown objectives fall through with whatever defaults the caller set

    return cfg


# ---------------------------------------------------------------------------
# Preset implementations
# ---------------------------------------------------------------------------

def _identity(cfg: "CurateConfig") -> None:
    """Person/character LoRA: face angles, clean single-subject shots.

    Priority hierarchy:
      1. Clear face visibility (MTCNN face count = 1 is ideal)
      2. Multiple distinct angles (front, 3/4, profile)
      3. Expression variety
      4. Technical quality
    """
    s = cfg.scoring
    s.weight_sharpness = 0.30
    s.weight_resolution = 0.20
    s.weight_artifact = 0.10
    s.weight_aesthetic = 0.10
    s.weight_subject = 0.30   # face/person detection matters most

    sel = cfg.selection
    sel.diversity_weight = 0.45   # more diversity — avoid 10 identical head shots
    sel.use_embedding_clusters = True

    # Enable MTCNN by default for face counting
    cfg.detectors.use_mtcnn = True
    cfg.detectors.use_yolo = True

    cfg.filters.blur_threshold = 120.0  # stricter — soft faces degrade identity training


def _style(cfg: "CurateConfig") -> None:
    """Style/aesthetic LoRA: broad composition and lighting variety.

    Priority hierarchy:
      1. Aesthetic quality (composition, colour, lighting)
      2. Maximum compositional diversity
      3. Technical quality
      4. Subject presence is optional
    """
    s = cfg.scoring
    s.weight_sharpness = 0.25
    s.weight_resolution = 0.20
    s.weight_artifact = 0.15
    s.weight_aesthetic = 0.35   # aesthetic proxy carries most weight
    s.weight_subject = 0.05

    sel = cfg.selection
    sel.diversity_weight = 0.55   # maximise style/composition spread
    sel.use_embedding_clusters = True

    cfg.filters.blur_threshold = 80.0   # less strict — artistic blur can be valid


def _wardrobe(cfg: "CurateConfig") -> None:
    """Clothing/outfit LoRA: full-body shots, clothing clearly visible.

    Priority hierarchy:
      1. Full body / clothing visibility (portrait / aspect bonuses)
      2. Person detected (YOLO)
      3. Clothing variety across images
      4. Technical quality
    """
    s = cfg.scoring
    s.weight_sharpness = 0.25
    s.weight_resolution = 0.25
    s.weight_artifact = 0.15
    s.weight_aesthetic = 0.15
    s.weight_subject = 0.20   # person detection matters for verifying clothing is visible

    sel = cfg.selection
    sel.diversity_weight = 0.40

    cfg.detectors.use_yolo = True
    cfg.filters.min_short_side = 640   # need enough resolution to read clothing


def _concept(cfg: "CurateConfig") -> None:
    """Object/concept LoRA: canonical views + edge case coverage.

    Priority hierarchy:
      1. Coverage of object variations (angles, contexts, scales)
      2. Technical quality
      3. Aesthetic quality
    """
    s = cfg.scoring
    s.weight_sharpness = 0.30
    s.weight_resolution = 0.30
    s.weight_artifact = 0.20
    s.weight_aesthetic = 0.15
    s.weight_subject = 0.05

    sel = cfg.selection
    sel.diversity_weight = 0.50   # maximise coverage, not quality
    sel.use_embedding_clusters = True
    sel.n_clusters = None  # auto-compute based on dataset size


# ---------------------------------------------------------------------------
# Public preset descriptions (for API / UI)
# ---------------------------------------------------------------------------

PRESET_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "identity": {
        "label": "Identity / Character",
        "description": (
            "Optimised for training a person or character LoRA. "
            "Prioritises clear single-face shots, multiple face angles "
            "(front, 3/4, profile), and expression variety. "
            "Strict blur filtering. MTCNN + YOLO enabled."
        ),
    },
    "style": {
        "label": "Style / Aesthetic",
        "description": (
            "Optimised for style transfer LoRAs. "
            "Maximises compositional and lighting diversity. "
            "Aesthetic scoring weighted heavily. "
            "Subject presence is optional."
        ),
    },
    "wardrobe": {
        "label": "Wardrobe / Outfit",
        "description": (
            "Optimised for clothing or outfit LoRAs. "
            "Prefers full-body shots where clothing is clearly visible. "
            "YOLO person detection enabled. Higher resolution minimum."
        ),
    },
    "concept": {
        "label": "Concept / Object",
        "description": (
            "Optimised for object or concept LoRAs. "
            "Maximises coverage of object variations — different angles, "
            "scales, and contexts. Equal weight on canonical and edge-case images."
        ),
    },
}
