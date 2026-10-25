"""Pydantic models — the curation stage's data contract.

The :class:`TargetProfile` is the shared moat between argus-curator and
argus-lens: both services speak the same ``style / backend / checkpoint /
category`` taxonomy, so a manifest emitted here is consumed by the captioner
verbatim with no remapping. Keep this schema in lockstep with argus-lens (or,
eventually, hoist it into a shared ``argus-core`` package).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

TargetStyle = Literal["photo", "anime"]
TargetCategory = Literal["identity", "wardrobe", "pose_composition", "setting"]

# Head-pose bucket of the primary face, derived from InsightFace yaw. Lets the UI
# pick a balanced subset by orientation (head-on / three-quarter / side profile).
FacePose = Literal["frontal", "three_quarter", "profile"]

# FairFace attribute labels, in the model's output order (see attributes.py).
# These are a model's prediction from appearance, not ground truth about anyone.
FaceRace = Literal["white", "black", "latino_hispanic", "east_asian", "southeast_asian", "indian", "middle_eastern"]
FaceGender = Literal["male", "female"]
FaceAge = Literal["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]
FACE_RACES: tuple[FaceRace, ...] = FaceRace.__args__
FACE_GENDERS: tuple[FaceGender, ...] = FaceGender.__args__
FACE_AGES: tuple[FaceAge, ...] = FaceAge.__args__
AttributeSource = Literal["fairface", "mivolo"]

# Places365 indoor/outdoor call for a whole image.
SceneEnvironment = Literal["indoor", "outdoor"]

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# Version of the JSONL handoff manifest argus-lens consumes. Bump on any
# breaking change to the per-row shape or semantics; every manifest row carries
# it so a consumer can refuse an incompatible manifest instead of misreading it.
# 2.0: rows exist only for files actually transferred and carry exported_path —
# the real path under the export root, which flattened exports de-collide.
# Consumers must use exported_path instead of deriving locations from rel_path.
# 2.1: rows carry exported_abs_path (the transferred file's absolute location),
# and abs_path is now guaranteed readable — under mode="move" it points at the
# destination, since the source it used to name has been deleted by the time the
# manifest is written. Additive plus a bug fix, so the major stays 2: a consumer
# that accepts 2.x keeps working, and one that already required exported_path
# (argus-forge's MAJORS_REQUIRING_EXPORTED_PATH) is unaffected.
MANIFEST_VERSION = "2.1"


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

    # Face-attribute models (see attributes.py). Model files are process
    # configuration (ARGUS_CURATOR_FAIRFACE_MODEL / ARGUS_CURATOR_MIVOLO_MODEL),
    # deliberately not fields here. Either one implies face detection.
    fairface: bool = False  # race, gender, age bracket
    mivolo: bool = False  # age in years, gender (supersedes FairFace's)

    @model_validator(mode="after")
    def _attributes_need_faces(self) -> FaceConfig:
        # Attributes are read off detected faces; asking for them without
        # detection would otherwise silently yield nothing.
        if self.fairface or self.mivolo:
            self.enabled = True
        return self


class SceneConfig(BaseModel):
    """Places365 scene tagging of whole images (see scenes.py).

    The model file is process configuration (ARGUS_CURATOR_PLACES365_MODEL),
    deliberately not a field here.
    """

    enabled: bool = False
    # How many top categories to record per image (and match filters against).
    top_k: int = Field(5, ge=1, le=20)
    # Compute device hint for onnxruntime: "auto" | "cpu" | "cuda".
    device: str = "auto"


# ---------------------------------------------------------------------------
# Per-image result
# ---------------------------------------------------------------------------


class FaceAttributes(BaseModel):
    """Predicted attributes of one face (or a cluster's pooled prediction).

    Each ``*_conf`` is the softmax probability of the chosen label, so a low
    value means the model was torn between labels. ``race`` comes only from
    FairFace. Gender and age come from MiVOLO when it ran, else FairFace (see
    ``*_source``); a MiVOLO age is a regression, so ``age_years`` is set,
    ``age`` is its derived bracket, and ``age_conf`` is ``None``.
    """

    race: FaceRace | None = None
    race_conf: float | None = None
    gender: FaceGender | None = None
    gender_conf: float | None = None
    gender_source: AttributeSource | None = None
    age: FaceAge | None = None
    age_conf: float | None = None
    age_years: float | None = None
    age_source: AttributeSource | None = None


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
    attributes: FaceAttributes | None = None


class SceneLabel(BaseModel):
    label: str
    conf: float


class SceneTags(BaseModel):
    """Places365 prediction for one image.

    ``categories`` are the top-k scene labels, most probable first, each with
    its softmax probability. ``environment_conf`` is the probability mass on
    categories of the chosen environment, so it is always at least 0.5.
    """

    categories: list[SceneLabel] = Field(default_factory=list)
    environment: SceneEnvironment
    environment_conf: float


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
    # Attributes of the primary face (when FaceConfig.fairface / .mivolo ran).
    primary_face_attributes: FaceAttributes | None = None

    # Places365 scene tags (when SceneConfig.enabled ran).
    scene: SceneTags | None = None

    # Internal: perceptual hash + score breakdown (kept for HITL transparency)
    phash: str = ""
    score_breakdown: dict[str, float] = Field(default_factory=dict)


class FaceCluster(BaseModel):
    """A clustered identity across the scanned dataset."""

    cluster_id: str
    size: int
    representative_rel_path: str
    representative_bbox: list[float] | None = None
    # Pooled over every member face: per-head probabilities are averaged before
    # taking the top label, so one odd frame cannot flip an identity's label.
    attributes: FaceAttributes | None = None


class ScanSummary(BaseModel):
    """Full result of a scan — persisted on disk and paginated by the server."""

    scan_id: str
    folder: str
    target_profile: TargetProfile
    config: ScanConfig
    faces_config: FaceConfig
    # Defaulted so scans cached before scene tagging existed still load.
    scenes_config: SceneConfig = Field(default_factory=SceneConfig)

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
    # The server resolves this under its configured scan root and refuses
    # traversal escapes; relative paths are canonical ("" scans the root).
    folder: str
    target_profile: TargetProfile = Field(default_factory=TargetProfile)
    config: ScanConfig = Field(default_factory=ScanConfig)
    faces: FaceConfig = Field(default_factory=FaceConfig)
    scenes: SceneConfig = Field(default_factory=SceneConfig)


class ExportRequest(BaseModel):
    scan_id: str | None = None
    # Inline selection of rel_paths (alternative to scan_id-driven selection).
    selection: list[str] | None = None

    # The server resolves this under its configured export root and refuses
    # traversal escapes; relative paths are canonical. Library/CLI callers
    # pass a filesystem path directly.
    dest: str
    # "move" is destructive; the server rejects it unless started with
    # --allow-move / CURATOR_ALLOW_MOVE=1.
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
    # Optional: export only images whose primary face's predicted attributes
    # are in these labels. An image without them is excluded once any is set.
    face_races: list[FaceRace] | None = None
    face_genders: list[FaceGender] | None = None
    face_ages: list[FaceAge] | None = None
    # Optional inclusive age range in years (needs MiVOLO's age_years).
    min_age_years: float | None = Field(None, ge=0.0)
    max_age_years: float | None = Field(None, ge=0.0)
    # Minimum confidence a filtered attribute must reach to count as a match.
    # Applies only where a confidence exists: MiVOLO ages are regressions.
    min_attribute_conf: float = Field(0.0, ge=0.0, le=1.0)

    # Optional: export only images with one of these Places365 scenes among
    # their recorded top-k at >= min_scene_conf (e.g. ["beach", "coast"]).
    scenes: list[str] | None = None
    scene_environments: list[SceneEnvironment] | None = None
    # Floor for a scene label (or the indoor/outdoor call) to count as a match.
    # Non-zero by default: the top-k tail is often a few percent, not a claim.
    min_scene_conf: float = Field(0.1, ge=0.0, le=1.0)
    # With max_keep: also spread the selection across scenes, not just pHash.
    diversify_scenes: bool = False

    write_manifest: bool = True
    # Optionally POST the manifest straight to argus-lens /caption (M4 stretch).
    caption_url: str | None = None

    @field_validator("scenes")
    @classmethod
    def _known_scenes(cls, value: list[str] | None) -> list[str] | None:
        # A typo ("beech") would otherwise silently match nothing.
        if value is None:
            return None
        from argus_curator._places365_labels import CATEGORIES

        unknown = sorted(set(value) - set(CATEGORIES))
        if unknown:
            raise ValueError(f"unknown Places365 scene(s): {', '.join(unknown)}")
        return value


class ManifestRow(BaseModel):
    """One line of the JSONL handoff manifest (written by ``export.write_manifest``).

    ``exported_path`` is the path actually written under the export root
    (posix, relative to it). Consumers must use it instead of re-deriving a
    destination from ``rel_path`` — flattened exports de-collide basenames, so
    the two can differ. Rows exist only for files whose transfer succeeded.

    ``exported_abs_path`` is that same file's absolute location, so a row is
    self-contained: the manifest is often read somewhere other than the export
    root it was written into (argus-lens receives it as an upload), and nothing
    in the row names the root to join ``exported_path`` onto.

    ``abs_path`` is where the image can be *read*, which is not always where it
    came from: ``mode="move"`` deletes the source during transfer, so for a
    moved export it equals ``exported_abs_path``. Use ``rel_path`` for the
    original location within the scan.
    """

    manifest_version: str = MANIFEST_VERSION
    rel_path: str
    abs_path: str
    exported_path: str
    exported_abs_path: str
    target_profile: TargetProfile
    primary_face_cluster: str | None = None
    primary_face_pose: FacePose | None = None
    score: float
    similar_group: int


class ExportResult(BaseModel):
    # Version of the manifest this curator writes, declared whether or not a
    # manifest was requested. Without it a client can only compat-detect by
    # sniffing which fields are present, which cannot distinguish "old server"
    # from "new server, nothing transferred" (both yield an empty mapping).
    manifest_version: str = MANIFEST_VERSION
    manifest_path: str | None = None
    copied: int
    skipped: int
    dest: str
    mode: str
    # What the selection *chose*, before transfer. A file here can still be
    # missing from exported_paths — its source vanished, or the transfer failed
    # — so this is not a record of what landed. Use exported_paths for that.
    selected_rel_paths: list[str] = Field(default_factory=list)
    # rel_path -> path actually written under dest (posix, relative), only for
    # transfers that succeeded — the same mapping the manifest rows carry, so
    # API callers get the de-collided names even with write_manifest=false.
    exported_paths: dict[str, str] = Field(default_factory=dict)
    # The same mapping, absolute. Joining dest onto exported_paths means a
    # client re-implements the server's layout rule; with preserve_structure
    # false the curator de-collides basenames into stem-<hash>.ext, which no
    # client can reproduce. The process that owns the rule publishes the answer.
    exported_abs_paths: dict[str, str] = Field(default_factory=dict)
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
