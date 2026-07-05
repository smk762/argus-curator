"""Pydantic models — the curation stage's data contract.

The :class:`TargetProfile` is the shared moat between argus-curator and
argus-lens: both services speak the same ``style / backend / checkpoint /
category`` taxonomy, so a manifest emitted here is consumed by the captioner
verbatim with no remapping. Keep this schema in lockstep with argus-lens (or,
eventually, hoist it into a shared ``argus-core`` package).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

TargetStyle = Literal["photo", "anime"]
TargetCategory = Literal["identity", "wardrobe", "pose_composition", "setting"]

# Head-pose bucket of the primary face, derived from InsightFace yaw. Lets the UI
# pick a balanced subset by orientation (head-on / three-quarter / side profile).
FacePose = Literal["frontal", "three_quarter", "profile"]

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# Version of the JSONL handoff manifest argus-lens consumes. Bump on any
# breaking change to the per-row shape or semantics; every manifest row carries
# it so a consumer can refuse an incompatible manifest instead of misreading it.
# 2.0: rows exist only for files actually transferred and carry exported_path —
# the real path under the export root, which flattened exports de-collide.
# Consumers must use exported_path instead of deriving locations from rel_path.
MANIFEST_VERSION = "2.0"


# ---------------------------------------------------------------------------
# Shared target profile (the moat)
# ---------------------------------------------------------------------------


class TargetProfile(BaseModel):
    """What we are curating *for* — shared verbatim with argus-lens.

    The curator uses it to weight scoring and label exports; the captioner
    inherits it to pick caption variants. This is the reason generic CV tools
    (FiftyOne / fastdup) structurally cannot replace the suite.
    """

    target_style: TargetStyle = "photo"
    target_backend: str | None = "sdxl"
    checkpoint: str | None = None
    target_category: TargetCategory = "identity"


# ---------------------------------------------------------------------------
# Scan configuration
# ---------------------------------------------------------------------------


class ScanConfig(BaseModel):
    """Tunable parameters for a training-suitability scan."""

    # Hard filters
    min_short_side: int = 512
    max_aspect_ratio: float = 3.0
    blur_threshold: float = 100.0

    # Near-duplicate clustering (pHash Hamming distance). -1 disables grouping.
    cluster_distance: int = 10

    # Composite score weights
    weight_sharpness: float = 0.35
    weight_resolution: float = 0.30
    weight_artifact: float = 0.15
    weight_subject: float = 0.20

    # Normalisation ceilings
    sharpness_ref: float = 800.0
    resolution_ref: int = 1024

    # Diversity blend used by the optional selection cap (0=score, 1=spread)
    diversity_weight: float = 0.40

    # Runtime
    max_workers: int = 4


class FaceConfig(BaseModel):
    """InsightFace detection + identity-clustering configuration (M2)."""

    enabled: bool = False
    model: str = "buffalo_l"
    min_det_score: float = 0.5
    # Cosine-distance threshold for grouping face embeddings into identities.
    cluster_eps: float = 0.5
    # Compute device hint for the InsightFace runtime: "auto" | "cpu" | "cuda".
    device: str = "auto"

    # Head-pose bucketing (absolute yaw, degrees) for the primary face:
    #   |yaw| <= frontal_max_yaw            -> "frontal"   (head-on)
    #   frontal_max_yaw < |yaw| <= profile_min_yaw -> "three_quarter"
    #   |yaw| > profile_min_yaw             -> "profile"   (side)
    frontal_max_yaw: float = 15.0
    profile_min_yaw: float = 45.0


# ---------------------------------------------------------------------------
# Per-image result
# ---------------------------------------------------------------------------


class FaceDetection(BaseModel):
    """One detected face within an image."""

    bbox: list[float] = Field(..., min_length=4, max_length=4, description="[x, y, w, h]")
    det_score: float
    cluster_id: str | None = None
    primary: bool = False
    # Head pose (degrees) from InsightFace, when available. Positive yaw = turned
    # to one side; ``pose`` is the derived bucket (frontal / three_quarter / profile).
    yaw: float | None = None
    pitch: float | None = None
    pose: FacePose | None = None


class ImageResult(BaseModel):
    """Per-image curation record — superset of imogen's scanner output.

    ``rel_path`` is the path relative to the scanned root and is the stable key
    everywhere (it stays unique across sub-folders, unlike a bare basename).
    """

    rel_path: str
    abs_path: str

    # Scoring / hard-filter outcome
    score: float = 0.0
    passed: bool = False
    reject_reason: str | None = None

    # Near-duplicate clustering
    similar_group: int = 0
    group_size: int = 1
    is_representative: bool = True
    is_duplicate: bool = False
    duplicate_of: str | None = None
    keep_reason: str = ""

    # Raw metrics
    sharpness: float = 0.0
    artifact_score: float = 0.0
    width: int = 0
    height: int = 0

    # Faces (M2)
    faces: list[FaceDetection] = Field(default_factory=list)
    face_count: int = 0
    primary_face_cluster: str | None = None
    # Orientation of the primary face (for pose-balanced subset selection).
    primary_face_pose: FacePose | None = None
    primary_face_yaw: float | None = None

    # Internal: perceptual hash + score breakdown (kept for HITL transparency)
    phash: str = ""
    score_breakdown: dict[str, float] = Field(default_factory=dict)


class FaceCluster(BaseModel):
    """A clustered identity across the scanned dataset."""

    cluster_id: str
    size: int
    representative_rel_path: str
    representative_bbox: list[float] | None = None


class ScanSummary(BaseModel):
    """Full result of a scan — persisted on disk and paginated by the server."""

    scan_id: str
    folder: str
    target_profile: TargetProfile
    config: ScanConfig
    faces_config: FaceConfig

    total: int
    passed: int
    rejected: int
    duplicates: int
    similar_clusters: int

    reject_reasons: dict[str, int] = Field(default_factory=dict)
    face_clusters: list[FaceCluster] = Field(default_factory=list)
    results: list[ImageResult] = Field(default_factory=list)

    # Pagination metadata (server fills these when slicing ``results``)
    offset: int = 0
    limit: int | None = None
    returned: int = 0


# ---------------------------------------------------------------------------
# Request / response bodies for the FastAPI server
# ---------------------------------------------------------------------------


class ScanRequest(BaseModel):
    folder: str
    target_profile: TargetProfile = Field(default_factory=TargetProfile)
    config: ScanConfig = Field(default_factory=ScanConfig)
    faces: FaceConfig = Field(default_factory=FaceConfig)


class ExportRequest(BaseModel):
    scan_id: str | None = None
    # Inline selection of rel_paths (alternative to scan_id-driven selection).
    selection: list[str] | None = None

    dest: str
    mode: Literal["copy", "symlink", "move"] = "copy"
    preserve_structure: bool = True

    min_score: float = 0.6
    include_rejected: bool = False
    keep_similar: bool = False
    max_keep: int | None = None

    # Optional: export only images whose primary face is in these clusters.
    face_clusters: list[str] | None = None
    # Optional: export only images whose primary-face pose is in these buckets.
    face_poses: list[FacePose] | None = None

    write_manifest: bool = True
    # Optionally POST the manifest straight to argus-lens /caption (M4 stretch).
    caption_url: str | None = None


class ManifestRow(BaseModel):
    """One line of the JSONL handoff manifest (written by ``export.write_manifest``).

    ``exported_path`` is the path actually written under the export root
    (posix, relative to it). Consumers must use it instead of re-deriving a
    destination from ``rel_path`` — flattened exports de-collide basenames, so
    the two can differ. Rows exist only for files whose transfer succeeded.
    """

    manifest_version: str = MANIFEST_VERSION
    rel_path: str
    abs_path: str
    exported_path: str
    target_profile: TargetProfile
    primary_face_cluster: str | None = None
    primary_face_pose: FacePose | None = None
    score: float
    similar_group: int


class ExportResult(BaseModel):
    manifest_path: str | None = None
    copied: int
    skipped: int
    dest: str
    mode: str
    selected_rel_paths: list[str] = Field(default_factory=list)
    # rel_path -> path actually written under dest (posix, relative), only for
    # transfers that succeeded — the same mapping the manifest rows carry, so
    # API callers get the de-collided names even with write_manifest=false.
    exported_paths: dict[str, str] = Field(default_factory=dict)
    captioned: bool = False


# ---------------------------------------------------------------------------
# Wire contract (published as JSON Schema for consumers to codegen against)
# ---------------------------------------------------------------------------

# The models that make up the public HTTP/manifest contract. Everything a
# consumer (argus-lens, the frontend) needs to speak to the curator is reachable
# from these via their nested $defs — including the manifest row itself.
WIRE_MODELS: tuple[type[BaseModel], ...] = (ScanRequest, ScanSummary, ExportRequest, ExportResult, ManifestRow)


def wire_schema() -> dict:
    """Combined JSON Schema for the curator's wire contract (all WIRE_MODELS)."""
    from pydantic.json_schema import models_json_schema

    _, schema = models_json_schema(
        [(m, "serialization") for m in WIRE_MODELS],
        title="argus-curator wire contract",
        ref_template="#/$defs/{model}",
    )
    return schema
