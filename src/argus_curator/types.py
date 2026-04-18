"""Core data types for argus-curator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Training objective presets
# ---------------------------------------------------------------------------

TRAINING_OBJECTIVES: tuple[str, ...] = (
    "identity",   # person/character LoRA — face angles, clear subject, single face
    "style",      # style/aesthetic LoRA — composition/lighting variety
    "wardrobe",   # clothing/outfit LoRA — full body, clothing visible
    "concept",    # object/concept LoRA — canonical + edge case coverage
)

TARGET_STYLES: tuple[str, ...] = ("photo", "anime")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FilterConfig:
    """Phase-1 hard-filter thresholds (no GPU required)."""
    min_short_side: int = 512
    max_aspect_ratio: float = 3.0
    blur_threshold: float = 100.0


@dataclass
class DuplicateConfig:
    """Near-duplicate detection settings."""
    phash_hamming_distance: int = 10
    """Maximum pHash Hamming distance for two images to be considered near-duplicates."""


@dataclass
class EmbeddingConfig:
    """CLIP / DINOv2 embedding extraction settings."""
    use_clip: bool = True
    clip_model: str = "openai/clip-vit-large-patch14"
    """ViT-L/14 gives meaningfully better semantic clustering than base-patch32."""
    use_dino: bool = False
    dino_model: str = "facebook/dinov2-base"
    """DINOv2 captures structural/compositional features CLIP misses."""
    batch_size: int = 16
    device: str = "auto"


@dataclass
class DetectorConfig:
    """Optional GPU subject detectors (YOLO + MTCNN)."""
    use_yolo: bool = False
    yolo_model: str = "yolov8n.pt"
    yolo_confidence: float = 0.40
    use_mtcnn: bool = False
    mtcnn_confidence: float = 0.90
    batch_size: int = 16
    device: str = "auto"


@dataclass
class ScoringConfig:
    """Composite quality score weights. Weights are re-normalised internally."""
    weight_sharpness: float = 0.35
    weight_resolution: float = 0.25
    weight_artifact: float = 0.15
    weight_aesthetic: float = 0.15
    weight_subject: float = 0.10
    """Subject weight only active when detectors are enabled."""
    sharpness_ref: float = 800.0
    resolution_ref: int = 1024


@dataclass
class SelectionConfig:
    """Diversity-aware subset selection settings."""
    target_count: int | None = None
    """Hard limit on selected images. If None, top_percent is used."""
    top_percent: float = 80.0
    """Fraction of de-duplicated candidates to select when target_count is None."""
    diversity_weight: float = 0.40
    """0 = pure quality ranking, 1 = pure diversity. 0.4 = balanced default."""
    use_embedding_clusters: bool = True
    """When CLIP embeddings are available, cluster by embedding and pick top-N per cluster
    instead of greedy pHash diversity. This gives better semantic coverage."""
    n_clusters: int | None = None
    """Target cluster count. None = auto (√n_candidates, capped at 20)."""
    caption_tags: dict[str, list[str]] | None = None
    """Optional {filename: [tag, ...]} from argus-lens output.
    When provided, underrepresented-tag images get a selection boost."""


@dataclass
class CurateConfig:
    """Top-level configuration for a curation run."""
    objective: str = "identity"
    target_style: str = "photo"
    filters: FilterConfig = field(default_factory=FilterConfig)
    duplicates: DuplicateConfig = field(default_factory=DuplicateConfig)
    embeddings: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    detectors: DetectorConfig = field(default_factory=DetectorConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    max_workers: int = 4

    @classmethod
    def for_objective(cls, objective: str, **kwargs: Any) -> "CurateConfig":
        """Return a config pre-tuned for the given training objective."""
        from argus_curator.presets import apply_preset
        cfg = cls(objective=objective, **kwargs)
        return apply_preset(cfg)


# ---------------------------------------------------------------------------
# Per-image result
# ---------------------------------------------------------------------------

@dataclass
class ImageResult:
    name: str
    source: str            # "local:/abs/path" or "bytes:<name>"
    width: int
    height: int
    short_side: int
    aspect_ratio: float

    # Phase-1 quality metrics
    sharpness: float
    artifact_score: float
    phash: str
    passed: bool
    reject_reason: str | None

    # Duplicate clustering
    is_duplicate: bool
    duplicate_of: str | None

    # Embedding-based clustering (populated when embeddings run)
    cluster_id: int | None = None
    clip_embedding: list[float] | None = None  # not serialised to dict
    dino_embedding: list[float] | None = None  # not serialised to dict

    # Phase-2 detector results
    face_count: int | None = None
    person_detected: bool | None = None
    person_confidence: float = 0.0

    # Scores
    sharpness_score: float = 0.0
    resolution_score: float = 0.0
    artifact_sub: float = 0.0
    aesthetic_score: float = 0.0
    subject_score: float = 0.0
    score: float = 0.0
    tag_boost: float = 0.0

    selected: bool = False
    score_breakdown: dict[str, float] = field(default_factory=dict)

    def to_dict(self, *, include_embeddings: bool = False) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "source": self.source,
            "width": self.width,
            "height": self.height,
            "short_side": self.short_side,
            "aspect_ratio": round(self.aspect_ratio, 3),
            "sharpness": round(self.sharpness, 2),
            "artifact_score": round(self.artifact_score, 4),
            "phash": self.phash,
            "passed": self.passed,
            "reject_reason": self.reject_reason,
            "is_duplicate": self.is_duplicate,
            "duplicate_of": self.duplicate_of,
            "cluster_id": self.cluster_id,
            "score": round(self.score, 4),
            "aesthetic_score": round(self.aesthetic_score, 4),
            "selected": self.selected,
        }
        if self.score_breakdown:
            d["score_breakdown"] = {k: round(v, 4) for k, v in self.score_breakdown.items()}
        if self.face_count is not None:
            d["face_count"] = self.face_count
        if self.person_detected is not None:
            d["person_detected"] = self.person_detected
        if self.subject_score:
            d["subject_score"] = round(self.subject_score, 4)
        if self.tag_boost:
            d["tag_boost"] = round(self.tag_boost, 4)
        if include_embeddings:
            d["clip_embedding"] = self.clip_embedding
            d["dino_embedding"] = self.dino_embedding
        return d


# ---------------------------------------------------------------------------
# Scan summary
# ---------------------------------------------------------------------------

@dataclass
class ScanSummary:
    total: int
    rejected_filters: int
    duplicates_removed: int
    candidates: int
    selected: int
    objective: str
    target_style: str
    diversity_weight: float
    embedding_clustering: bool
    reject_reasons: dict[str, int]
    selected_names: list[str]
    results: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "rejected_filters": self.rejected_filters,
            "duplicates_removed": self.duplicates_removed,
            "candidates": self.candidates,
            "selected": self.selected,
            "objective": self.objective,
            "target_style": self.target_style,
            "diversity_weight": self.diversity_weight,
            "embedding_clustering": self.embedding_clustering,
            "reject_reasons": self.reject_reasons,
            "selected_names": self.selected_names,
            "results": self.results,
        }
