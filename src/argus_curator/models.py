"""Pydantic models — the curation stage's data contract.

The :class:`TargetProfile` is the shared moat between argus-curator and
argus-lens: both services speak the same ``style / backend / checkpoint /
category`` taxonomy, so a manifest emitted here is consumed by the captioner
verbatim with no remapping. Keep this schema in lockstep with argus-lens (or,
eventually, hoist it into a shared ``argus-core`` package).
"""

from __future__ import annotations

from typing import Literal

from argus_cortex.moderation import (  # re-exported: part of curator's wire contract
    DEFAULT_EXPOSED_LABELS,
    GuardResult,
    NsfwResult,
    NudeNetLabel,
)
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

# Fashionpedia's 27 garment classes, in model output order (ids 0-26), as
# comma-free slugs so they survive comma-separated CLI lists. Parts (sleeve,
# pocket, ...; ids 27-45) are not recorded — see garments.py.
GarmentLabel = Literal[
    "shirt_blouse",  # shirt, blouse
    "top_tshirt_sweatshirt",  # top, t-shirt, sweatshirt
    "sweater",
    "cardigan",
    "jacket",
    "vest",
    "pants",
    "shorts",
    "skirt",
    "coat",
    "dress",
    "jumpsuit",
    "cape",
    "glasses",
    "hat",
    "headband_hair_accessory",  # headband, head covering, hair accessory
    "tie",
    "glove",
    "watch",
    "belt",
    "leg_warmer",
    "tights_stockings",  # tights, stockings
    "sock",
    "shoe",
    "bag_wallet",  # bag, wallet
    "scarf",
    "umbrella",
]
GARMENT_LABELS: tuple[GarmentLabel, ...] = GarmentLabel.__args__

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


class GarmentConfig(BaseModel):
    """Fashionpedia garment detection on whole images (see garments.py).

    The model file is process configuration (ARGUS_CURATOR_GARMENTS_MODEL),
    deliberately not a field here.
    """

    enabled: bool = False
    # Detections below this probability are not recorded at all. Lower than
    # the export filter's default (0.5) so reviewers still see the doubtful ones.
    min_conf: float = Field(0.3, ge=0.0, le=1.0)
    # Compute device hint for onnxruntime: "auto" | "cpu" | "cuda".
    device: str = "auto"


class NsfwConfig(BaseModel):
    """NudeNet nudity detection via a remote NudeNet server (see nsfw.py).

    The server URL is process configuration (ARGUS_CURATOR_NUDENET_URL),
    deliberately not a field here: a request-supplied URL would let any API
    caller make this server send images to an address of their choosing.
    """

    # None (default) = on whenever a NudeNet URL is configured, i.e. opt-out;
    # True = required (a scan fails if no URL is configured); False = opted out.
    # A finished scan records what actually happened as True or False.
    enabled: bool | None = None


class GuardConfig(BaseModel):
    """SingGuard policy check via a remote SingGuard server (see guard.py).

    Like NudeNet, the URL (ARGUS_CURATOR_SINGGUARD_URL) and the policy
    (ARGUS_CURATOR_SINGGUARD_POLICY_FILE) are process configuration.
    """

    # Same opt-out semantics as NsfwConfig.enabled.
    enabled: bool | None = None


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


class GarmentDetection(BaseModel):
    """One detected garment."""

    label: GarmentLabel
    conf: float
    bbox: list[float] = Field(..., min_length=4, max_length=4, description="[x, y, w, h]")
    # The box reaches the frame edge, so the garment is likely cut off.
    touches_edge: bool = False
    # Mean colour of the box's central half, "#rrggbb".
    color: str


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
    # Fashionpedia garments, most confident first (None = detection not run).
    garments: list[GarmentDetection] | None = None
    # NudeNet nudity detection (None = not checked, or the check failed).
    nsfw: NsfwResult | None = None
    # SingGuard policy verdict (None = not checked, or the check failed).
    guard: GuardResult | None = None

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
    garments_config: GarmentConfig = Field(default_factory=GarmentConfig)
    nsfw_config: NsfwConfig = Field(default_factory=NsfwConfig)
    guard_config: GuardConfig = Field(default_factory=GuardConfig)

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
    garments: GarmentConfig = Field(default_factory=GarmentConfig)
    nsfw: NsfwConfig = Field(default_factory=NsfwConfig)
    guard: GuardConfig = Field(default_factory=GuardConfig)


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

    # Optional: export only images wearing at least one of these garments, and
    # none of the excluded ones, counting detections >= min_garment_conf.
    garments: list[GarmentLabel] | None = None
    exclude_garments: list[GarmentLabel] | None = None
    min_garment_conf: float = Field(0.5, ge=0.0, le=1.0)
    # With `garments`: the matching garment must not touch the frame edge.
    garments_uncropped: bool = False
    # With max_keep: also spread the selection across outfits (garment types
    # and their colours), so an identity set is not one outfit throughout.
    diversify_outfits: bool = False

    # Optional: drop images whose NudeNet exposed score (the highest score
    # among nsfw_exposed_labels) is at or above this. Scores under 0.2 are
    # never reported, so 0.2 is the strictest threshold that means anything.
    nsfw_threshold: float | None = Field(None, ge=0.2, le=1.0)
    # Optional: per-class thresholds; drop images where that class scores at
    # or above its threshold, e.g. {"FEMALE_BREAST_EXPOSED": 0.5}.
    nsfw_label_thresholds: dict[NudeNetLabel, float] | None = None
    # The classes nsfw_threshold is taken over.
    # Never empty: an empty set scores every image 0.0, so nsfw_threshold
    # would silently pass everything.
    nsfw_exposed_labels: list[NudeNetLabel] = Field(default_factory=lambda: list(DEFAULT_EXPOSED_LABELS), min_length=1)

    # Optional: drop images SingGuard judged unsafe, under any rule.
    guard_unsafe: bool = False
    # Optional: drop images where one of these rules fired, named by letter
    # ("A") or in full ("A. Sexual Content Risk"), case-insensitive.
    guard_rules: list[str] | None = None

    write_manifest: bool = True
    # Optionally POST the manifest straight to argus-lens /caption (M4 stretch).
    caption_url: str | None = None

    @field_validator("nsfw_label_thresholds")
    @classmethod
    def _thresholds_in_range(cls, value: dict[str, float] | None) -> dict[str, float] | None:
        for label, threshold in (value or {}).items():
            if not 0.2 <= threshold <= 1.0:
                raise ValueError(f"{label}: threshold {threshold} must be within 0.2..1 (NudeNet reports from 0.2)")
        return value

    @field_validator("guard_rules")
    @classmethod
    def _rules_named(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        rules = [r.strip() for r in value]
        if not all(rules):
            raise ValueError("guard_rules entries must be non-empty")
        return rules

    @model_validator(mode="after")
    def _age_range_in_order(self) -> ExportRequest:
        if (
            self.min_age_years is not None
            and self.max_age_years is not None
            and self.min_age_years > self.max_age_years
        ):
            raise ValueError(f"min_age_years ({self.min_age_years}) is above max_age_years ({self.max_age_years})")
        return self

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


class SelectRow(BaseModel):
    """One image under a dry-run selection.

    ``keep_reason`` is empty when the image is in the selection. A non-empty
    value is why ``decide_selection`` left it out — the same strings an export
    report would write, without copying anything.
    """

    rel_path: str
    abs_path: str
    keep_reason: str = ""


class SelectRequest(BaseModel):
    """Filter body for ``POST /scan/{id}/select``.

    The filter fields of :class:`ExportRequest`, and none of the transfer
    fields (``dest``, ``mode``, ``selection``). ``max_keep`` and the diversify
    flags are absent on purpose: this route never diversity-trims.
    """

    min_score: float = 0.6
    include_rejected: bool = False
    keep_similar: bool = False

    face_clusters: list[str] | None = None
    face_poses: list[FacePose] | None = None
    face_races: list[FaceRace] | None = None
    face_genders: list[FaceGender] | None = None
    face_ages: list[FaceAge] | None = None
    min_age_years: float | None = Field(None, ge=0.0)
    max_age_years: float | None = Field(None, ge=0.0)
    min_attribute_conf: float = Field(0.0, ge=0.0, le=1.0)

    scenes: list[str] | None = None
    scene_environments: list[SceneEnvironment] | None = None
    min_scene_conf: float = Field(0.1, ge=0.0, le=1.0)

    garments: list[GarmentLabel] | None = None
    exclude_garments: list[GarmentLabel] | None = None
    min_garment_conf: float = Field(0.5, ge=0.0, le=1.0)
    garments_uncropped: bool = False

    nsfw_threshold: float | None = Field(None, ge=0.2, le=1.0)
    nsfw_label_thresholds: dict[NudeNetLabel, float] | None = None
    nsfw_exposed_labels: list[NudeNetLabel] = Field(default_factory=lambda: list(DEFAULT_EXPOSED_LABELS), min_length=1)

    guard_unsafe: bool = False
    guard_rules: list[str] | None = None

    @model_validator(mode="after")
    def _valid_as_export(self) -> SelectRequest:
        # ExportRequest owns the cross-field rules (age order, known scenes,
        # NSFW threshold range, rule names); run them here rather than copy them.
        self.as_export_request()
        return self

    def as_export_request(self) -> ExportRequest:
        """A transfer-free request so ``decide_selection`` can run unchanged.

        ``dest`` is a placeholder. Callers must not pass this object to
        ``export_selection``.
        """
        return ExportRequest.model_validate({**self.model_dump(), "dest": "."})


class SelectResult(BaseModel):
    """What the current filter would keep, without writing the export root."""

    scan_id: str
    total: int
    kept: int
    rows: list[SelectRow] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Wire contract (published as JSON Schema for consumers to codegen against)
# ---------------------------------------------------------------------------

# The models that make up the public HTTP/manifest contract. Everything a
# consumer (argus-lens, the frontend) needs to speak to the curator is reachable
# from these via their nested $defs — including the manifest row itself.
WIRE_MODELS: tuple[type[BaseModel], ...] = (
    ScanRequest,
    ScanSummary,
    ExportRequest,
    ExportResult,
    ManifestRow,
    SelectRequest,
    SelectResult,
)


def wire_schema() -> dict:
    """Combined JSON Schema for the curator's wire contract (all WIRE_MODELS)."""
    from pydantic.json_schema import models_json_schema

    _, schema = models_json_schema(
        [(m, "serialization") for m in WIRE_MODELS],
        title="argus-curator wire contract",
        ref_template="#/$defs/{model}",
    )
    return schema
