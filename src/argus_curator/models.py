"""Pydantic models — the curation stage's data contract.

The :class:`TargetProfile` is the shared moat between argus-curator and
argus-lens: both services speak the same ``style / backend / checkpoint /
category`` taxonomy, so a manifest emitted here is consumed by the captioner
verbatim with no remapping. The style and category literals come from
``argus_cortex.taxonomy``, the one copy the whole suite shares.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from typing import Annotated, Any, Literal

from argus_cortex.consent import ConsentRefusal
from argus_cortex.moderation import (  # re-exported: part of curator's wire contract
    DEFAULT_EXPOSED_LABELS,
    GuardResult,
    NsfwResult,
    NudeNetLabel,
    exposed_score,
)
from argus_cortex.moderation.nudenet import MIN_REPORTED_SCORE
from argus_cortex.moderation.policies import PolicyName
from argus_cortex.provenance import SHA256_PATTERN, Provenance
from argus_cortex.taxonomy import (  # re-exported: the suite's one taxonomy, not a local copy
    SPEECH_CATEGORIES,  # noqa: F401
    TARGET_CATEGORIES,  # noqa: F401
    TARGET_STYLES,  # noqa: F401
    SpeechCategory,
    TargetCategory,
    TargetStyle,  # noqa: F401
)
from argus_cortex.taxonomy import TargetProfile as _CortexTargetProfile
from argus_cortex.wire import render_schema as _cortex_render_schema
from argus_cortex.wire import wire_schema as _cortex_wire_schema
from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    PrivateAttr,
    SerializationInfo,
    SerializerFunctionWrapHandler,
    Tag,
    computed_field,
    field_validator,
    model_serializer,
    model_validator,
)

ExportMode = Literal["copy", "symlink", "move"]

# For models only ever sent, never received: a defaulted field is always in the
# payload, so the wire schema lists it as required. A model a request also
# carries must not use this, since a client may leave its defaulted fields out
# (tests/test_schema.py checks which models carry it).
RESPONSE_ONLY = ConfigDict(json_schema_serialization_defaults_required=True)

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
# 2.2: rows carry crop, the box baked into the exported file (in the source's
# displayed pixels), or null when the file is the source as it was. Additive.
# 2.3: rows carry primary_face (box, yaw / pitch / roll, face sharpness, eye and
# mouth openness, identity distances, age, and landmarks when asked) and the
# scan's set_age. Additive (#130).
# 2.4: rows carry split ("train" / "held_out") when the export asked for a
# train/held-out split, else null. Additive (#183).
# 2.5: target_profile carries modality, always "image" (argus-cortex 0.12's
# TargetProfile, narrowed to the image modality: target_category keeps its
# image-only enum). Additive.
# 2.6: rows of a consented face set (face_source.consent) carry the consent
# record's release_sha256 and the sha256 of the record file as read at export,
# and the image's own provenance (cortex's Provenance); null otherwise.
# Additive (#133).
# 2.7: rows carry pose_labels (the primary person's stance / arms / facing from
# argus-puppet) and pose_cluster (the scan-level body-pose cluster), both null
# when the pose stage did not run or could not say. Additive.
MANIFEST_VERSION = "2.7"

# Version of a speech export's manifest (#187), its own row shape
# (SpeechManifestRow) in its own file (speech_manifest.jsonl). It moves
# independently of MANIFEST_VERSION and of the package version, under the same
# rule: the major for a breaking row change, the minor for an additive one
# (schema/speech-manifest-row.snapshot.json, `make manifest-snapshot`).
SPEECH_MANIFEST_VERSION = "1.0"

# SpeechSelection.consent_max_distance's default (#209): the cosine distance,
# speaker embedding to the reference recordings' centroid, under which a clip
# is the consenting voice. Measured with the pinned ECAPA model (docs/speech.md).
VOICE_CONSENT_MAX_DISTANCE = 0.25

# Cap on ExportRequest.run_stats once serialised: it is opaque client JSON
# written to disk verbatim, so a runaway payload must not fill the export root.
RUN_STATS_MAX_BYTES = 5 * 1024 * 1024


# ---------------------------------------------------------------------------
# Shared target profile (the moat)
# ---------------------------------------------------------------------------


# Not a local copy: a subclass of argus-cortex's that inherits every field,
# default and validator, so an instance is a cortex TargetProfile and flows to
# lens, forge and proof unchanged. It narrows modality and target_category to
# images because curator's scoring (face counts, framing) has no meaning for a
# speech target, and keeps the manifest's target_category enum image-only (a
# consumer validating it never meets a speech value). from_attributes lets a
# cortex image profile built by a library caller convert; a cortex speech
# profile is refused the same way.
class TargetProfile(_CortexTargetProfile):
    """What a dataset was curated for, shared verbatim across the suite.

    argus-cortex's ``TargetProfile`` for an image dataset: ``modality`` is
    always ``image`` and ``target_category`` an image category. argus-curator
    curates images, so a speech profile is refused.
    """

    model_config = ConfigDict(from_attributes=True)

    modality: Literal["image"] = "image"
    target_category: TargetCategory = "identity"

    @model_validator(mode="before")
    @classmethod
    def _image_only(cls, data: Any) -> Any:
        modality = data.get("modality", "image") if isinstance(data, dict) else getattr(data, "modality", "image")
        if modality != "image":
            raise ValueError(f"argus-curator curates images: a {modality!r} target profile is not supported")
        return data


# The speech counterpart (#186): a subclass of cortex's narrowed the other way,
# so a speech scan's profile is a cortex TargetProfile too. Only the speech scan
# path takes it (ScanRequest / ScanSummary through ScanProfile); ManifestRow and
# the image scanner keep TargetProfile, which refuses it.
class SpeechTargetProfile(_CortexTargetProfile):
    """argus-cortex's ``TargetProfile`` for a speech dataset: ``modality`` is ``speech``.

    ``target_category`` is a speech category (``voice`` by default), and
    ``target_backend`` names a TTS model family (``xtts-v2``, ``qwen3-tts``) or
    is ``None``. Accepted by the speech scan only; image scoring and the image
    manifest never see it.
    """

    model_config = ConfigDict(from_attributes=True)

    modality: Literal["speech"] = "speech"
    target_category: SpeechCategory = "voice"
    target_backend: str | None = None


def _profile_modality(value: Any) -> str:
    # A profile without modality reads as image, as cortex's does.
    modality = value.get("modality", "image") if isinstance(value, dict) else getattr(value, "modality", "image")
    return modality if isinstance(modality, str) else "image"


# The profile a scan request (and its stored summary) carries: an image profile,
# or with ``modality: "speech"`` a speech one, which routes the scan to the
# speech path. Anything that writes manifest rows reads TargetProfile instead.
ScanProfile = Annotated[
    Annotated[TargetProfile, Tag("image")] | Annotated[SpeechTargetProfile, Tag("speech")],
    Discriminator(_profile_modality),
]


# ---------------------------------------------------------------------------
# Scan configuration
# ---------------------------------------------------------------------------


# The smallest short side a scan accepts, and a floor on ScanConfig.min_short_side.
# A frame below it is rejected, and a rejected frame gets no face, age, scene, or
# garment reading, so the under-18 rule could not fire on it. A request cannot
# lower this.
MIN_SHORT_SIDE = 512

# The exposed-score threshold to use when filtering nudity (#156): any exposure
# NudeNet reports at all. It's NudeNet's reporting floor, and darkroom's default.
# Conservative on purpose: on 640m, a 0.5 cutoff let through exposed images
# scored 0.27-0.49, and no clean public-domain image scored 0.2 or more. Only a
# default value: a request that sets no nsfw_threshold still filters nothing.
DEFAULT_NSFW_THRESHOLD = MIN_REPORTED_SCORE

_INCLUDE_REJECTED_IGNORED = "Ignored: a rejected image is never kept. Accepted so older clients still validate."


SEMANTIC_DUPLICATES_NEED_EMBEDDINGS = "config.semantic_duplicates needs the scan's embeddings: turn embeddings on"
ZERO_SHOT_NEEDS_EMBEDDINGS = "{what} needs the scan's embeddings: turn embeddings on"


class ScanConfig(BaseModel):
    """Tunable parameters for a training-suitability scan."""

    # Hard filters
    min_short_side: int = Field(MIN_SHORT_SIDE, ge=MIN_SHORT_SIDE)
    max_aspect_ratio: float = 3.0
    blur_threshold: float = 100.0

    # Near-duplicate clustering (pHash Hamming distance). -1 disables grouping.
    cluster_distance: int = 10
    # Semantic near-duplicates (#196): also group images whose CLIP embeddings
    # have a cosine similarity of at least ``semantic_duplicate_cosine`` (mirrors,
    # small crops and rescans that pHash puts far apart). Off by default. Needs
    # the scan's embeddings (EmbeddingConfig.enabled): a scan asking for it
    # without them is refused. Either test links an image to a group's
    # representative (see scanner._mark_duplicates); a -1 cluster_distance turns
    # off the pHash test only. 0.95 sits above every pair of distinct shots
    # measured (same person or same scene: at most 0.944) and below rescans of
    # one photo (0.98+) and most mirrors (median 0.989); see docs/scoring.md.
    semantic_duplicates: bool = False
    semantic_duplicate_cosine: float = Field(0.95, ge=0.0, le=1.0)

    # Zero-shot CLIP scores (#197, see zeroshot.py), each off by default and
    # needing the scan's embeddings (a scan asking without them is refused).
    # ``profile_fit`` adds ``profile_fit_weight`` x the image's fit to the
    # TargetProfile to its target bonus; ``junk_check`` takes ``junk_weight`` x
    # its junk probability off it. Neither rejects an image: that is the opt-in
    # PrefilterConfig.drop_junk, which drops those at or above
    # ``junk_threshold``. Calibration: docs/scoring.md.
    profile_fit: bool = False
    profile_fit_weight: float = Field(0.10, ge=0.0, le=1.0)
    junk_check: bool = False
    junk_weight: float = Field(0.15, ge=0.0, le=1.0)
    junk_threshold: float = Field(0.8, ge=0.0, le=1.0)

    # Composite score weights
    weight_sharpness: float = 0.35
    weight_resolution: float = 0.30
    weight_artifact: float = 0.15
    weight_subject: float = 0.20

    # Normalisation ceilings. 0 turns the term off (it scores 0); a negative
    # ceiling would invert the ranking, so it is refused.
    sharpness_ref: float = Field(800.0, ge=0.0)
    resolution_ref: int = Field(1024, ge=0)

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

    # Read an age for the frames with no face off the set they belong to (see
    # faces.infer_set_age). Only when the scan looks like one subject: at
    # least set_age_min_faces MiVOLO ages, and at least set_age_min_share of
    # the faces in one identity. The frames with no face are often the most
    # explicit ones, and MiVOLO has nothing to read on them. Off by default:
    # in a mixed folder the inherited age would be someone else's.
    set_age: bool = False
    set_age_min_faces: int = Field(5, ge=1)
    set_age_min_share: float = Field(0.9, ge=0.0, le=1.0)

    # Give MiVOLO each face's body as well (see bodies.py). MiVOLO reads a face
    # and a person crop side by side and is published on the pair; without a
    # body every face runs face-only. Bodies come from InsightFace's person
    # detector (body_model, a Cheetah pack). Under evaluation
    # (scripts/age_eval.py): off by default. The thresholds are MiVOLO's own.
    mivolo_body: bool = False
    body_model: str = "cheetah_l"
    # A person box below this detector score is not a body.
    body_min_score: float = Field(0.5, ge=0.0, le=1.0)
    # A body crop narrower or shorter than this many pixels is not used.
    body_min_size: int = Field(50, ge=1)
    # Share of a body crop left once every face and other person on it is blacked
    # out; below it, the crop is not used.
    body_min_visible: float = Field(0.4, ge=0.0, le=1.0)
    # Least match score (mean of IoU and the share of the face inside the body)
    # for a face to be given a body.
    body_face_match: float = Field(0.0001, ge=0.0, le=1.0)

    # The identity reference: each face's reference_distance is the cosine
    # distance from it. Either images in this scan (rel_paths), whose primary
    # faces' embeddings are averaged (a frontal portrait for a multi-view set),
    # or a centroid the caller supplies (reference_embedding), such as
    # faces.reference_embedding() over a consenting person's reference photos
    # that are not part of the set. Not both. None = no reference distance.
    # A reference image the scan's filters rejected is still embedded (from the
    # bytes the scan read) but stays rejected. Embeddings are never stored: the
    # distance is measured during the scan, and ScanSummary drops
    # reference_embedding from its faces_config, so neither the scan cache nor
    # GET /scan/{id} ever holds it.
    reference_paths: list[str] = Field(default_factory=list)
    reference_embedding: list[float] | None = Field(default=None, min_length=1)

    @field_validator("reference_embedding")
    @classmethod
    def _usable_embedding(cls, value: list[float] | None) -> list[float] | None:
        if value is None:
            return None
        if not all(math.isfinite(v) for v in value) or not any(value):
            raise ValueError("reference_embedding must be finite and not all zero")
        return value

    @model_validator(mode="after")
    def _attributes_need_faces(self) -> FaceConfig:
        if self.reference_paths and self.reference_embedding is not None:
            raise ValueError("give reference_paths or reference_embedding, not both")
        # Attributes are read off detected faces; asking for them without
        # detection would otherwise silently yield nothing. A set age is read
        # off MiVOLO ages.
        if self.set_age or self.mivolo_body:
            self.mivolo = True
        if self.fairface or self.mivolo or self.has_reference:
            self.enabled = True
        return self

    @property
    def has_reference(self) -> bool:
        """Whether this config asks for reference distances."""
        return bool(self.reference_paths) or self.reference_embedding is not None


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


class EmbeddingConfig(BaseModel):
    """Semantic image embeddings of whole images, a CLIP image tower (see embeddings.py).

    The vectors are held in memory for the scan only, never written, and are
    gone once the scan is deleted or the server restarts. The model (and so the
    embedding space) is process configuration (ARGUS_CURATOR_EMBEDDINGS_MODEL,
    ``embeddings.EMBEDDING_SPACE``), deliberately not a field here.
    """

    enabled: bool = False
    # Compute device hint for onnxruntime: "auto" | "cpu" | "cuda".
    device: str = "auto"

    # The subject reference for likeness (#191): each passing image's
    # ImageResult.likeness_distance is the cosine distance from its embedding
    # to it. Either images in this scan (rel_paths), whose embeddings are
    # averaged (a rejected one is still embedded, from the bytes the scan read,
    # and stays rejected), or a centroid the caller supplies in this embedding
    # space (reference_embedding). Not both. Neither = the distance to the set's
    # own centroid (see embeddings.measure_likeness). Either reference turns
    # embeddings on. Vectors are never stored: ScanSummary drops
    # reference_embedding from its embeddings_config.
    reference_paths: list[str] = Field(default_factory=list)
    reference_embedding: list[float] | None = Field(default=None, min_length=1)

    @field_validator("reference_embedding")
    @classmethod
    def _usable_embedding(cls, value: list[float] | None) -> list[float] | None:
        if value is None:
            return None
        if not all(math.isfinite(v) for v in value) or not any(value):
            raise ValueError("reference_embedding must be finite and not all zero")
        return value

    @model_validator(mode="after")
    def _reference_needs_embeddings(self) -> EmbeddingConfig:
        if self.reference_paths and self.reference_embedding is not None:
            raise ValueError("give reference_paths or reference_embedding, not both")
        if self.has_reference:
            self.enabled = True
        return self

    @property
    def has_reference(self) -> bool:
        """Whether this config asks for likeness distances to a reference (else to the set's centroid)."""
        return bool(self.reference_paths) or self.reference_embedding is not None


# Where an image's likeness_distance is measured from (#191): "reference" =
# EmbeddingConfig.reference_paths or reference_embedding; "centroid" = the
# scan's own passing images' (trimmed) centroid.
LikenessSource = Literal["reference", "centroid"]


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


class PoseConfig(BaseModel):
    """Body pose from a remote argus-puppet server (see bodypose.py).

    The server URL is process configuration (ARGUS_CURATOR_PUPPET_URL), never a
    request field, as for NudeNet. Same opt-out rule: None (default) = on
    whenever the URL is set; True = required (the scan is refused without a
    URL); False = opted out. A finished scan records whether the stage ran as
    True or False.
    """

    enabled: bool | None = None


class ConsentConfig(BaseModel):
    """Bind a real person's face set to their consent record (#133; see consent.py).

    ``record`` is the :class:`argus_cortex.consent.ConsentRecord` file, a path
    relative to the scanned folder (the server refuses one that resolves
    outside it). None = no consent check. With a record the scan reads it,
    checks it for the ``avatar`` use of the subject's face, builds the identity
    reference from the record's reference images (found in the scan by their
    sha256), and binds each image: its primary face within ``max_distance`` of
    that reference, and no one else in the frame. Turning it on turns on face
    detection, MiVOLO and the set age, which the export gate needs.
    """

    record: str | None = None
    # Ceiling on the primary face's reference_distance for an image to be bound.
    max_distance: float = Field(0.4, ge=0.0, le=2.0)

    @field_validator("record")
    @classmethod
    def _named(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("consent.record names the record file (e.g. consent.json), or is null")
        return value


# Why a consent record makes a scan's set ineligible: cortex's refusals from
# consent_valid, or curator's own. "missing": no record file. "invalid": not a
# ConsentRecord. "no-reference-image": none of the record's reference images
# is in the scan. "no-reference-face": none of them has a face, or the
# reference gave no distances.
ConsentIneligible = Literal["missing", "invalid", "no-reference-image", "no-reference-face"] | ConsentRefusal

# Why an image is not bound to the scan's consent record. "not-consented": its
# <stem>.json sidecar marks its subject as not consenting (argus-quarry's
# likeness_consent: false). "provenance": its sidecar's provenance is
# unreadable, or names another source than this release. "unchecked": face
# detection failed on it. "no-face": no face. "bystander": someone else is in
# the frame. "mismatch": the primary face is not within max_distance of the
# record's reference.
ConsentExclusion = Literal["not-consented", "provenance", "unchecked", "no-face", "bystander", "mismatch"]


class ConsentCheck(BaseModel):
    """What the scan made of its consent record (ScanSummary.consent).

    Never the record: a record names a person, so the scan keeps only the
    verdict, the record's hashes and which images are the references. It is
    part of the scan, so deleting or purging the scan removes it.
    """

    model_config = RESPONSE_ONLY

    # ConsentConfig.record as the scan was given it.
    record: str
    eligible: bool
    reason: ConsentIneligible | None = None
    detail: str = ""
    # The record's key (the signed release's sha256) and the sha256 of the
    # record file the scan read; None when there was no readable record.
    release_sha256: str | None = Field(None, pattern=SHA256_PATTERN)
    record_sha256: str | None = Field(None, pattern=SHA256_PATTERN)
    checked_at: datetime
    max_distance: float
    # rel_paths of the scanned images that are the record's reference images.
    references: list[str] = Field(default_factory=list)
    # Reference images the record lists that the scan did not find.
    missing_references: int = 0


class ConsentBinding(BaseModel):
    """Whether one image is bound to the scan's consent record (ImageResult.consent)."""

    model_config = RESPONSE_ONLY

    bound: bool
    # The primary face's reference_distance, kept for review of the borderline ones.
    distance: float | None = None
    reason: ConsentExclusion | None = None


EraxLabel = Literal["anus", "make_love", "nipple", "penis", "vagina"]


class EraxDetection(BaseModel):
    """One EraX-NSFW detection. ``bbox`` is ``[x, y, w, h]`` in source pixels."""

    model_config = RESPONSE_ONLY

    label: EraxLabel
    score: float
    bbox: list[float] = Field(..., min_length=4, max_length=4, description="[x, y, w, h]")


class EraxResult(BaseModel):
    """EraX-NSFW's view of one image. Classes are EraX's own, not NudeNet's."""

    model_config = RESPONSE_ONLY

    detections: list[EraxDetection] = Field(default_factory=list)
    labels: dict[EraxLabel, float] = Field(default_factory=dict)


class EraxConfig(BaseModel):
    """Optional EraX-NSFW fill-in via a remote server (see erax.py).

    The URL (ARGUS_CURATOR_ERAX_URL) is process configuration, like NudeNet.
    A finished scan records whether the check ran as True or False.
    """

    model_config = RESPONSE_ONLY

    enabled: bool | None = None


# "default" is the operator's policy (ARGUS_CURATOR_SINGGUARD_POLICY_FILE, else
# SingGuard's own rules); "adult" is curator's built-in policy that allows
# consensual adult nudity and still flags minors (see guard.ADULT_POLICY).
# "adult-granular", "adult-cautious" and "adult-cinematic" are variants under
# evaluation. The names and texts are argus-cortex's (argus_cortex.moderation.policies).
GuardPolicy = PolicyName


class GuardConfig(BaseModel):
    """SingGuard policy check via a remote SingGuard server (see guard.py).

    Like NudeNet, the URL (ARGUS_CURATOR_SINGGUARD_URL) is process
    configuration. A request picks a policy by name, never by text.

    ``thinking``: a request may pick the SingGuard mode (``fast``,
    ``fast-slow``, ``slow`` or ``cascade``); null runs the server's default
    (ARGUS_CURATOR_SINGGUARD_THINKING). The server refuses a mode outside
    ARGUS_CURATOR_SINGGUARD_THINKING_ALLOWED with 422. A finished scan records
    the mode it ran in, whichever set it. ``fell_back`` is recorded by the scan
    only (a value a request sends is replaced): whether the server refused the
    mode so checks dropped to ``fast``.
    """

    # Same opt-out semantics as NsfwConfig.enabled.
    enabled: bool | None = None
    # The official verdict (ImageResult.guard) and what the filters use by default.
    policy: GuardPolicy = "default"
    # Extra policies to judge every image under too, for comparison: one more
    # SingGuard pass per image each. Results land in ImageResult.guard_variants.
    variants: list[GuardPolicy] = Field(default_factory=list)
    # The SingGuard mode the scan asks for: the request's, else the server's
    # default (ARGUS_CURATOR_SINGGUARD_THINKING); the scan records the one it
    # ran in. fell_back is the scan's alone (a value sent is replaced): whether
    # the server refused the mode so checks dropped to "fast". Each
    # verdict's own GuardResult.thinking says which mode judged it; under
    # cascade that is "fast" for every frame not escalated, so only this field
    # tells a cascade scan from a fast one. None until the check has run.
    thinking: Literal["fast", "fast-slow", "slow", "cascade"] | None = None
    fell_back: bool | None = None

    @model_validator(mode="after")
    def _variants_distinct(self) -> GuardConfig:
        # The official policy already runs; a repeat would only double the cost.
        self.variants = [v for i, v in enumerate(self.variants) if v != self.policy and v not in self.variants[:i]]
        return self


class PrefilterConfig(BaseModel):
    """Opt-in drops applied before the late scan phases (see prefilter.py).

    Right after the identities phase, a passing image that matches a rule is
    turned into a reject (``passed`` False, ``reject_reason``
    ``prefilter:nsfw`` / ``prefilter:no-face`` / ``prefilter:no-age``, the
    first that applies in that order). It stays in the results, but scenes,
    garments and SingGuard never run on it. Hard rejects keep their own reason.

    ``drop_nsfw``: an exposed NudeNet reading of at least 0.2 (an image NudeNet
    and EraX could not score is kept; needs the NudeNet check). ``drop_no_face``:
    no face detected (needs face detection; an image whose detection failed,
    ``face_error``, is kept). ``drop_no_age``: no MiVOLO age on any face, no
    body-only age, and no set age (needs MiVOLO; a failed detection has no age,
    so it is dropped).
    """

    drop_nsfw: bool = False
    drop_no_face: bool = False
    drop_no_age: bool = False
    # A zero-shot junk probability at or above ScanConfig.junk_threshold
    # (#197): ``prefilter:junk``. Needs the scan's embeddings; applied right
    # after the embeddings phase, so only SingGuard is spared for those.
    drop_junk: bool = False


class CropBox(BaseModel):
    """A crop in pixels of the image as displayed: after its EXIF orientation, ``right``/``bottom`` exclusive.

    Recorded beside a source as ``<image>.crop.json`` (``{"left": .., "top": ..,
    "right": .., "bottom": ..}``) or passed inline in :class:`CropConfig`. A box
    must lie inside the frame and be non-empty; one that does not makes the
    image a ``crop:`` reject rather than letting it through uncropped.
    """

    left: int = Field(ge=0)
    top: int = Field(ge=0)
    right: int = Field(gt=0)
    bottom: int = Field(gt=0)


class WatermarkDetection(BaseModel):
    """One watermark the detector found: ``[x, y, w, h]`` in displayed pixels of the uncropped frame."""

    model_config = RESPONSE_ONLY

    label: str
    score: float
    bbox: list[float] = Field(..., min_length=4, max_length=4, description="[x, y, w, h]")


class CropConfig(BaseModel):
    """Opt-in crops (watermarks, borders) applied to the bytes in memory, never to the source.

    With ``enabled``, an image with a box is cropped once after it is read, and
    every check (scoring, faces, NudeNet, SingGuard ...) sees the cropped frame;
    face boxes and sizes are in its pixels. Export writes the cropped image, so
    a cropped scan exports only by ``mode="copy"``.

    A box is given by hand (``boxes``, keyed by rel_path, wins over a
    ``<image>.crop.json`` sidecar; ``sidecars`` False reads none) or, with
    ``detect``, found: every image without a box by hand goes to the watermark
    detector (``ARGUS_CURATOR_WATERMARK_URL``), and the hits of at least
    ``min_conf`` are cut away along the frame's edges. Each edge may lose at
    most ``max_cut`` of the frame's side; ``margin`` (pixels) is cut beyond a
    hit's edge. A hit that no such cut removes is an overlay: with
    ``reject_overlay`` the image is a ``crop:overlay`` reject, else it is kept
    uncropped with its hits recorded. An image the detector could not check is
    a ``crop:unchecked`` reject. An image with no box is scanned as it is.
    """

    enabled: bool = False
    sidecars: bool = True
    boxes: dict[str, CropBox] = Field(default_factory=dict)
    detect: bool = False
    min_conf: float = Field(default=0.4, ge=0.0, le=1.0)
    max_cut: float = Field(default=0.15, gt=0.0, lt=0.5)
    margin: int = Field(default=4, ge=0)
    reject_overlay: bool = True


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

    model_config = RESPONSE_ONLY

    race: FaceRace | None = None
    race_conf: float | None = None
    gender: FaceGender | None = None
    gender_conf: float | None = None
    gender_source: AttributeSource | None = None
    age: FaceAge | None = None
    age_conf: float | None = None
    age_years: float | None = None
    age_source: AttributeSource | None = None
    # MiVOLO read this face's body as well as its face (FaceConfig.mivolo_body).
    # None on a pooled cluster, or when MiVOLO did not run.
    age_body: bool | None = None


class SetAge(BaseModel):
    """An age for a frame with no face, read off the set the frame belongs to.

    ``years`` is the median MiVOLO age of the faces in the scan's dominant
    identity, ``cluster_id`` that identity, ``faces`` how many ages it is read
    from, and ``share`` the fraction of the scan's faces in it. It is an
    inference about the set, not a reading of this frame: a viewer shows it as
    such, and treats it as it treats a measured age where a wrong age would be
    the costlier mistake.
    """

    model_config = RESPONSE_ONLY

    years: float
    # Read before the prefilter: an identity it emptied is gone from face_clusters but still named here.
    cluster_id: str
    faces: int
    share: float


# Serialization context for the scan cache, the one place FaceDetection.readings is written.
STORE_CONTEXT = "scan-store"


class FaceReadings(BaseModel):
    """One face's raw model outputs, kept so its identity can be pooled again after a delete.

    They live on the face, so they go with its row and with the scan: nothing
    outlives a deleted image or a purged scan. ``None`` on a face from a scan
    cached before readings were kept. Only the scan cache holds them: they are
    not on the wire (see :data:`STORE_CONTEXT`).
    """

    # Sized here so a bad cache fails on load, not while pooling after the files are gone.
    fairface: list[float] | None = Field(None, min_length=18, max_length=18)
    mivolo_gender: list[float] | None = Field(None, min_length=2, max_length=2)
    mivolo_age: float | None = None


class FaceLandmarks(BaseModel):
    """InsightFace's landmarks for one face, in the image's pixels.

    ``kps`` are the detector's 5 keypoints (eyes, nose tip, mouth corners) as
    ``[x, y]``; ``landmark_3d_68`` the 68 iBUG points as ``[x, y, z]``. Kept in
    the scan cache only, and written to a manifest when an export asks
    (``ExportRequest.landmarks``): they are the largest per-face payload.
    """

    model_config = RESPONSE_ONLY

    kps: list[list[float]] | None = None
    landmark_3d_68: list[list[float]] | None = None


class FaceDetection(BaseModel):
    """One detected face within an image."""

    model_config = RESPONSE_ONLY

    bbox: list[float] = Field(..., min_length=4, max_length=4, description="[x, y, w, h]")
    det_score: float
    cluster_id: str | None = None
    primary: bool = False
    # Head pose (degrees) from InsightFace, when available. Positive yaw = turned
    # to one side; ``pose`` is the derived bucket (frontal / three_quarter / profile).
    yaw: float | None = None
    pitch: float | None = None
    roll: float | None = None
    pose: FacePose | None = None
    # Laplacian variance over the face box alone, in the units of
    # ImageResult.sharpness: a sharp face on a plain background can sit in an
    # image whose overall sharpness is low.
    sharpness: float | None = None
    # Eye aspect ratio (mean of both eyes) and inner-lip opening over mouth
    # width, from the 68 landmarks. Measurements, not an expression classifier:
    # open eyes read about 0.25-0.35 and a closed mouth about 0. None without
    # the 3D-68 landmark model.
    eye_open: float | None = None
    mouth_open: float | None = None
    # Cosine distance from this face's embedding to its identity's centroid,
    # and to the scan's reference (FaceConfig.reference_paths or
    # reference_embedding; None without one).
    cluster_distance: float | None = None
    reference_distance: float | None = None
    attributes: FaceAttributes | None = None
    readings: FaceReadings | None = Field(default=None, exclude=True)
    # Scan cache only, like readings (see FaceLandmarks).
    landmarks: FaceLandmarks | None = Field(default=None, exclude=True)
    # Index into ImageResult.people when FaceConfig.mivolo_body matched this face
    # to a person box; NudeNet detections assigned to that person live there.
    person_index: int | None = None
    # NudeNet classes assigned to this face's box when no person list was recorded.
    nsfw_labels: dict[NudeNetLabel, float] = Field(default_factory=dict)
    exposed_score: float = 0.0

    # No return annotation: one would replace this model's properties in the wire schema.
    @model_serializer(mode="wrap")
    def _readings_for_the_store(self, handler: SerializerFunctionWrapHandler, info: SerializationInfo):
        data = handler(self)
        if info.context == STORE_CONTEXT:
            if self.readings is not None:
                data["readings"] = self.readings.model_dump(mode=info.mode)
            if self.landmarks is not None:
                data["landmarks"] = self.landmarks.model_dump(mode=info.mode)
        return data


PeopleCheck = Literal["ok", "detection_failed", "body_ages_failed"]


class PersonDetection(BaseModel):
    """One person box from InsightFace's Cheetah detector (FaceConfig.mivolo_body).

    ``face_index`` is the face it was given, or ``None`` when no face matched
    (a body-only crop; MiVOLO then runs with an absent face). ``nsfw_labels``
    are NudeNet detections whose box overlaps this person.
    """

    model_config = RESPONSE_ONLY

    bbox: list[float] = Field(..., min_length=4, max_length=4, description="[x, y, w, h]")
    det_score: float
    face_index: int | None = None
    age_years: float | None = None
    age_source: AttributeSource | None = None
    age_body: bool | None = None
    nsfw_labels: dict[NudeNetLabel, float] = Field(default_factory=dict)
    exposed_score: float = 0.0


class SceneLabel(BaseModel):
    model_config = RESPONSE_ONLY

    label: str
    conf: float


class SceneTags(BaseModel):
    """Places365 prediction for one image.

    ``categories`` are the top-k scene labels, most probable first, each with
    its softmax probability. ``environment_conf`` is the probability mass on
    categories of the chosen environment, so it is always at least 0.5.
    """

    model_config = RESPONSE_ONLY

    categories: list[SceneLabel] = Field(default_factory=list)
    environment: SceneEnvironment
    environment_conf: float


class GarmentDetection(BaseModel):
    """One detected garment."""

    model_config = RESPONSE_ONLY

    label: GarmentLabel
    conf: float
    bbox: list[float] = Field(..., min_length=4, max_length=4, description="[x, y, w, h]")
    # The box reaches the frame edge, so the garment is likely cut off.
    touches_edge: bool = False
    # Mean colour of the box's central half, "#rrggbb".
    color: str


# A content rating read off NudeNet's detections, for a facet in a UI and for
# argus:ContentRating in an XMP sidecar. "explicit": an exposed class was
# reported (the five DEFAULT_EXPOSED_LABELS, from NudeNet's 0.2 floor, the
# suite's one definition of nudity). "suggestive": otherwise, a covered intimate
# class was (lingerie, swimwear). "safe": neither. Unchecked images have none.
ContentRating = Literal["safe", "suggestive", "explicit"]
SUGGESTIVE_LABELS: tuple[NudeNetLabel, ...] = (
    "FEMALE_BREAST_COVERED",
    "FEMALE_GENITALIA_COVERED",
    "BUTTOCKS_COVERED",
    "ANUS_COVERED",
)


def rate_content(nsfw: NsfwResult | None) -> ContentRating | None:
    """*nsfw*'s :data:`ContentRating`, or ``None`` when NudeNet did not check the image."""
    if nsfw is None:
        return None
    if exposed_score(nsfw.labels) >= MIN_REPORTED_SCORE:
        return "explicit"
    if any(nsfw.labels.get(label, 0.0) >= MIN_REPORTED_SCORE for label in SUGGESTIVE_LABELS):
        return "suggestive"
    return "safe"


# Which near-duplicate test linked an image to its group's representative (#196).
DuplicateKind = Literal["phash", "semantic", "both"]

# The junk kinds the zero-shot check names (see zeroshot.py), in prompt order.
JunkKind = Literal["screenshot", "collage", "text", "meme"]
JUNK_KINDS: tuple[JunkKind, ...] = ("screenshot", "collage", "text", "meme")


class ZeroShotResult(BaseModel):
    """Zero-shot CLIP scores for one image (#197; see zeroshot.py).

    Plain scores only: the image's embedding itself is never stored. A field
    is None when its check was not asked for.
    """

    model_config = RESPONSE_ONLY

    # The probability that the image is the kind the TargetProfile asks for
    # (one softmax over the style x category prompts; for concept, its style's).
    profile_fit: float | None = None
    # The probability that the image is a screenshot, collage, page of text or
    # meme rather than an ordinary photo or artwork.
    junk: float | None = None
    # The likeliest junk kind, only once ``junk`` reaches ScanConfig.junk_threshold.
    junk_kind: JunkKind | None = None


# argus-puppet's coarse body-pose labels (geometry rules, not a classifier;
# "unknown" when the joints a rule needs are not confident). Facing is the
# subject's as the camera sees it: "left" means they face the image's left.
BodyStance = Literal["standing", "jumping", "crouching", "seated", "lying", "unknown"]
BodyArms = Literal["up", "out", "down", "mixed", "unknown"]
BodyFacing = Literal["front", "back", "left", "right", "unknown"]

# How the body-pose stage went for one image (None = not checked: the stage was
# off, or the image was rejected before it). "ok": a primary person was found
# (pose_labels holds their labels when the server gave them; pose_cluster is set
# unless the scan's clustering was skipped, ScanSummary.pose.skipped says why).
# "unusable": a person was found but too few body joints were confident for
# clustering, so pose_cluster is None. "no_person": nobody in the frame.
# "failed": the request failed or the server was unreachable.
PoseCheck = Literal["ok", "unusable", "no_person", "failed"]


class BodyPoseLabels(BaseModel):
    """The primary person's body-pose labels, as argus-puppet derived them. Keypoints are never stored."""

    model_config = RESPONSE_ONLY

    stance: BodyStance
    arms: BodyArms
    facing: BodyFacing


class ImageResult(BaseModel):
    """Per-image curation record — superset of imogen's scanner output.

    ``rel_path`` is the path relative to the scanned root and is the stable key
    everywhere (it stays unique across sub-folders, unlike a bare basename).
    """

    model_config = RESPONSE_ONLY

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
    # Which test tied this image to ``duplicate_of`` (None when not a duplicate):
    # "phash" (Hamming distance within cluster_distance), "semantic" (embedding
    # cosine at least semantic_duplicate_cosine, with semantic_duplicates on) or "both".
    duplicate_kind: DuplicateKind | None = None
    keep_reason: str = ""

    # Zero-shot profile fit and junk scores (ScanConfig.profile_fit /
    # junk_check, #197). None = not asked for, or the image has no embedding.
    zero_shot: ZeroShotResult | None = None

    # Raw metrics
    sharpness: float = 0.0
    artifact_score: float = 0.0
    width: int = 0
    height: int = 0

    # Faces (M2)
    faces: list[FaceDetection] = Field(default_factory=list)
    face_count: int = 0
    # Why face detection failed on this image (the detector raised, or the
    # image could not be decoded), else None. With it set, ``face_count`` 0
    # means unchecked, not faceless.
    face_error: str | None = None
    # Person boxes from Cheetah when FaceConfig.mivolo_body ran. Empty means it
    # was not asked, or it found none — unlike garments, this is never None.
    people: list[PersonDetection] = Field(default_factory=list)
    # How the person check went (FaceConfig.mivolo_body): None = not asked for,
    # or face detection on the image failed before it;
    # "ok"; "detection_failed" = the person detector failed, so `people` is empty
    # and the faces ran face-only; "body_ages_failed" = people were found but
    # MiVOLO's body-only ages failed. An empty `people` with "ok" means none found.
    people_check: PeopleCheck | None = None
    people_error: str | None = None
    primary_face_cluster: str | None = None
    # Orientation of the primary face (for pose-balanced subset selection).
    primary_face_pose: FacePose | None = None
    primary_face_yaw: float | None = None
    primary_face_pitch: float | None = None
    primary_face_roll: float | None = None
    # Attributes of the primary face (when FaceConfig.fairface / .mivolo ran).
    primary_face_attributes: FaceAttributes | None = None
    # Only on a passing frame with no face, when FaceConfig.set_age found one subject.
    set_age: SetAge | None = None
    # Only when the scan read an eligible consent record (ConsentConfig).
    consent: ConsentBinding | None = None
    # Cosine distance (0..2) from this image's CLIP embedding to the subject
    # (#191): the scan's likeness reference, else the set's centroid
    # (ScanSummary.likeness_source says which). The non-face analogue of a
    # face's reference_distance / cluster_distance. None = embeddings off, the
    # image rejected or not embedded, or no distance (ScanSummary.likeness_skipped).
    likeness_distance: float | None = None

    # Places365 scene tags (when SceneConfig.enabled ran).
    scene: SceneTags | None = None
    # Fashionpedia garments, most confident first (None = detection not run).
    garments: list[GarmentDetection] | None = None
    # NudeNet nudity detection (None = not checked, or the check failed).
    nsfw: NsfwResult | None = None
    # EraX-NSFW (None = not checked, or the check failed). Own classes.
    erax: EraxResult | None = None
    # True when ``nsfw`` was filled from EraX because NudeNet did not score it.
    erax_filled: bool = False
    # True when NudeNet's check failed and ``nsfw`` holds EraX's hits alone.
    # EraX has no buttocks class, so the NudeNet filters still exclude the
    # image as unchecked unless those hits reach a threshold.
    nudenet_failed: bool = False
    # SingGuard policy verdict (None = not checked, or the check failed).
    guard: GuardResult | None = None
    # Only when GuardConfig.variants ran: every policy's verdict by name, the
    # official one included. A missing name means that check failed.
    guard_variants: dict[GuardPolicy, GuardResult] | None = None

    # The crop every check saw (CropConfig), in the source's displayed pixels.
    # width/height and every box are of the cropped frame. None = uncropped.
    crop: CropBox | None = None
    # Where the crop came from: "manual" (a request box or a .crop.json
    # sidecar) or "detected" (CropConfig.detect). None when uncropped.
    crop_source: Literal["manual", "detected"] | None = None
    # Watermarks the detector found (CropConfig.detect), on the uncropped
    # frame; empty = none found. None = not checked (detection off, a box by
    # hand, or the check failed).
    watermarks: list[WatermarkDetection] | None = None

    # Body pose from argus-puppet (PoseConfig; see bodypose.py). pose_check says
    # how it went (None = not checked). pose_labels: the primary person's labels,
    # None without a person or a reading. pose_cluster: the scan-level cluster id
    # (0 = the largest; ScanSummary.pose.clusters), None when the pose is
    # unusable or the scan's clustering was skipped. Clusters are formed over
    # the whole scan, never a selection. No keypoints are kept.
    pose_check: PoseCheck | None = None
    pose_labels: BodyPoseLabels | None = None
    pose_cluster: int | None = None

    # Internal: perceptual hash + score breakdown (kept for HITL transparency)
    phash: str = ""
    score_breakdown: dict[str, float] = Field(default_factory=dict)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def content_rating(self) -> ContentRating | None:
        """Derived from :attr:`nsfw` on read, so a scan cached before ratings existed has one too."""
        return rate_content(self.nsfw)


# ---------------------------------------------------------------------------
# Speech scans (#186)
# ---------------------------------------------------------------------------


SPEECH_EXTS = {".wav", ".flac"}

# The checks a speech scan runs, in the order a clip meets them. A clip that
# fails any is rejected: ``reject_reason`` names the first and
# ``failed_checks`` every one measured. ``unreadable``: not a WAV/FLAC clip the
# scan can decode (``reject_detail`` says why). ``corpus_under_18``: the
# corpus's own metadata marks the speaker as under 18 (a hard exclusion, never
# configurable). ``cer`` needs an STT engine (argus-echo#4).
SpeechCheck = Literal[
    "unreadable",
    "corpus_under_18",
    "duration",
    "sample_rate",
    "channels",
    "silent",
    "clipping",
    "leading_silence",
    "trailing_silence",
    "transcript",
    "speaking_rate",
    "snr",
    "speech_ratio",
    "mos",
    "cer",
]
SPEECH_CHECKS: tuple[SpeechCheck, ...] = SpeechCheck.__args__

# Where a clip's transcript came from: a ``<clip>.txt`` beside it, LJSpeech's
# ``metadata.csv``, or a Common Voice ``*.tsv``, both at the scanned folder's top.
TranscriptSource = Literal["sidecar", "ljspeech", "common_voice"]
TranscriptLayout = Literal["auto", "sidecar", "ljspeech", "common_voice"]

# Ceilings a request cannot raise: they bound what decoding one clip costs.
MAX_CLIP_SECONDS = 120.0
MAX_CLIP_CHANNELS = 2


class SpeechConfig(BaseModel):
    """Thresholds and options for a speech scan (``target_profile.modality == "speech"``).

    Every threshold is a prefilter: a clip past it is rejected with the check
    named (:data:`SpeechCheck`). ``None`` turns a threshold off. Silence is
    measured against the clip's loudest 30 ms frame: a frame ``silence_db``
    or more below it is silent. ``chars_per_s`` counts a transcript's letters
    and digits over the clip less its leading and trailing silence.
    """

    # Which transcripts to read. "auto": a sidecar, else LJSpeech's
    # metadata.csv, else Common Voice TSVs (the first that names the clip).
    transcripts: TranscriptLayout = "auto"
    require_transcript: bool = True

    min_duration_s: float = Field(1.0, ge=0.0)
    max_duration_s: float = Field(20.0, gt=0.0, le=MAX_CLIP_SECONDS)
    min_sample_rate: int = Field(16000, ge=8000)
    max_channels: int = Field(1, ge=1, le=MAX_CLIP_CHANNELS)
    # Share of samples at or past full scale (|x| >= 0.999).
    max_clipping_ratio: float | None = Field(0.001, ge=0.0, le=1.0)
    silence_db: float = Field(-40.0, lt=0.0)
    max_leading_silence_s: float | None = Field(1.0, ge=0.0)
    max_trailing_silence_s: float | None = Field(1.0, ge=0.0)
    min_chars_per_s: float | None = Field(4.0, ge=0.0)
    max_chars_per_s: float | None = Field(30.0, gt=0.0)
    min_snr_db: float | None = None
    min_speech_ratio: float | None = Field(None, ge=0.0, le=1.0)

    # DNSMOS (ARGUS_CURATOR_DNSMOS_MODEL), run on each clip that passed the
    # checks above. min_mos is on its overall (OVRL) score, 1-5.
    mos: bool = False
    mos_device: str = "auto"
    min_mos: float | None = Field(None, ge=1.0, le=5.0)
    # The character error rate between the transcript and an STT pass. Needs
    # an STT engine (argus-echo#4): a scan that sets it without one is refused.
    max_cer: float | None = Field(None, ge=0.0)

    # Near-duplicate clips: Hamming distance between 64-bit spectrogram
    # fingerprints, among clips whose lengths are within 10%. -1 disables.
    duplicate_distance: int = Field(6, ge=-1, le=64)

    # Speaker embeddings (WeSpeaker ECAPA-TDNN, ARGUS_CURATOR_SPEAKER_MODEL),
    # run on each clip that passed, then clustered into speakers: a clip joins
    # the nearest speaker whose centroid is within speaker_threshold (cosine
    # distance). The embeddings stay in memory only (ScanSummary.embeddings).
    speakers: bool = False
    speaker_device: str = "auto"
    speaker_threshold: float = Field(0.5, gt=0.0, lt=2.0)

    # Composite score weights (a term the scan did not measure drops out and
    # the rest are renormalised), and the SNR that scores 1.
    weight_mos: float = Field(0.5, ge=0.0)
    weight_snr: float = Field(0.3, ge=0.0)
    weight_speech_ratio: float = Field(0.2, ge=0.0)
    snr_ref_db: float = Field(40.0, gt=0.0)

    max_workers: int = Field(4, ge=1)

    @model_validator(mode="after")
    def _ranges(self) -> SpeechConfig:
        if self.min_duration_s >= self.max_duration_s:
            raise ValueError("min_duration_s must be below max_duration_s")
        low, high = self.min_chars_per_s, self.max_chars_per_s
        if low is not None and high is not None and low >= high:
            raise ValueError("min_chars_per_s must be below max_chars_per_s")
        return self


class SpeechMos(BaseModel):
    """DNSMOS P.835 predictions for a clip, each 1-5: speech (SIG), background (BAK), overall (OVRL)."""

    model_config = RESPONSE_ONLY

    model: Literal["dnsmos"] = "dnsmos"
    sig: float
    bak: float
    ovrl: float


class ClipResult(BaseModel):
    """Per-clip record of a speech scan, keyed by ``rel_path`` like :class:`ImageResult`.

    A measurement is ``None`` when the scan never got to it (the clip could not
    be decoded, or the check was not asked for). The corpus labels
    (``speaker``, ``accent``, ``language``, ``age``) are copied verbatim from
    the corpus metadata, never estimated.
    """

    model_config = RESPONSE_ONLY

    rel_path: str
    abs_path: str

    score: float = 0.0
    passed: bool = False
    # The first failed check, and every failed one in SPEECH_CHECKS order.
    reject_reason: SpeechCheck | None = None
    failed_checks: list[SpeechCheck] = Field(default_factory=list)
    # What the first failed check measured against its threshold.
    reject_detail: str | None = None

    # Near-duplicate clustering (as ImageResult's).
    similar_group: int = 0
    group_size: int = 1
    is_representative: bool = True
    is_duplicate: bool = False
    duplicate_of: str | None = None

    duration_s: float | None = None
    sample_rate: int | None = None
    channels: int | None = None
    clipping_ratio: float | None = None
    leading_silence_s: float | None = None
    trailing_silence_s: float | None = None
    snr_db: float | None = None
    speech_ratio: float | None = None

    transcript: str | None = None
    transcript_source: TranscriptSource | None = None
    chars_per_s: float | None = None

    mos: SpeechMos | None = None
    # Why DNSMOS gave no score for a clip it was asked to score, else None.
    mos_error: str | None = None
    # The STT hook (argus-echo#4): its transcript, and the CER against ``transcript``.
    stt_transcript: str | None = None
    cer: float | None = None

    speaker: str | None = None
    accent: str | None = None
    language: str | None = None
    age: str | None = None
    # The corpus's gender (sex) label, verbatim (Common Voice's ``gender``).
    gender: str | None = None

    # SpeechConfig.speakers (#187): the speaker cluster the clip's embedding
    # joined (``spk_1`` is the largest), its cosine distance to that cluster's
    # centroid, and whether the corpus's speaker label disagrees with the
    # clustering (the label's clips span clusters, or the cluster holds more
    # than one label). None and False when no clustering ran.
    speaker_cluster: str | None = None
    speaker_distance: float | None = None
    speaker_mismatch: bool = False
    # Why no speaker embedding was made for a clip that passed, else None.
    speaker_error: str | None = None

    # 64-bit spectrogram fingerprint (hex), the counterpart of ImageResult.phash.
    fingerprint: str = ""
    score_breakdown: dict[str, float] = Field(default_factory=dict)


class SpeakerCluster(BaseModel):
    """A speaker found by clustering a speech scan's embeddings (#187).

    ``corpus_speakers`` counts the corpus speaker labels among its clips (empty
    when the corpus names none). ``mismatch`` is True when the clustering and
    the corpus disagree about it: it holds more than one label, or a label it
    holds also has clips in another cluster. A centroid is a voiceprint, so
    none is stored.
    """

    model_config = RESPONSE_ONLY

    cluster_id: str
    clips: int
    seconds: float
    corpus_speakers: dict[str, int] = Field(default_factory=dict)
    mismatch: bool = False


class FaceCluster(BaseModel):
    """A clustered identity across the scanned dataset."""

    model_config = RESPONSE_ONLY

    cluster_id: str
    # Size and representative leave out images the prefilter dropped, which can never be kept.
    size: int
    representative_rel_path: str
    representative_bbox: list[float] | None = Field(None, min_length=4, max_length=4, description="[x, y, w, h]")
    # Pooled over every member face, dropped images included: per-head probabilities are averaged before
    # taking the top label, so one odd frame cannot flip an identity's label.
    attributes: FaceAttributes | None = None


class PoseCluster(BaseModel):
    """One scan-level body-pose cluster from argus-puppet's ``POST /stats`` (average linkage over the scan)."""

    model_config = RESPONSE_ONLY

    # 0 for the largest, as ImageResult.pose_cluster names it.
    id: int
    size: int
    # size / ScanSummary.pose.usable.
    share: float
    # The image whose pose is most central to the cluster; None once that row was dropped.
    medoid_rel_path: str | None = None


class PoseSummary(BaseModel):
    """What the body-pose stage found over the whole scan (see bodypose.py and docs/body-pose.md).

    ``posed`` images came back from argus-puppet (``no_person`` of them with
    nobody in frame), ``failed`` did not. The rest comes from one ``POST
    /stats`` over every posed image, or is None/empty with ``skipped`` saying
    why (the scan is over the set cap, the server refused or failed the call,
    nothing was posed). ``stale``: rows were dropped since, so cluster sizes and
    shares are recounted from the rows left but ``diversity`` and
    ``comparable_share`` are as measured.
    """

    model_config = RESPONSE_ONLY

    puppet_version: str | None = None
    posed: int = 0
    no_person: int = 0
    failed: int = 0
    skipped: str | None = None
    usable: int | None = None
    comparable_share: float | None = None
    diversity: float | None = None
    largest_cluster_share: float | None = None
    clusters: list[PoseCluster] = Field(default_factory=list)
    stale: bool = False


class ScanSummary(BaseModel):
    """Full result of a scan — persisted on disk and paginated by the server."""

    model_config = RESPONSE_ONLY

    scan_id: str
    folder: str
    # A speech profile marks a speech scan: its rows are in ``clips``, and
    # ``results`` and every image config stay empty.
    target_profile: ScanProfile
    config: ScanConfig
    faces_config: FaceConfig
    # Defaulted so scans cached before scene tagging existed still load.
    scenes_config: SceneConfig = Field(default_factory=SceneConfig)
    garments_config: GarmentConfig = Field(default_factory=GarmentConfig)
    # Whether the scan computed semantic embeddings. The vectors themselves are
    # not here: see ``embeddings``.
    embeddings_config: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    nsfw_config: NsfwConfig = Field(default_factory=NsfwConfig)
    erax_config: EraxConfig = Field(default_factory=EraxConfig)
    guard_config: GuardConfig = Field(default_factory=GuardConfig)
    # Defaulted so scans cached before the prefilter existed still load.
    prefilter_config: PrefilterConfig = Field(default_factory=PrefilterConfig)
    crop_config: CropConfig = Field(default_factory=CropConfig)
    # Whether the body-pose stage ran, and what it found over the scan (None when it did not run).
    pose_config: PoseConfig = Field(default_factory=PoseConfig)
    pose: PoseSummary | None = None

    total: int
    passed: int
    rejected: int
    duplicates: int
    similar_clusters: int
    # True once rows were dropped (a delete or a move) after this process lost
    # the scan's embeddings (a restart) and some image's semantic representative
    # went with them: the groups were rebuilt from pHash and the semantic links
    # the rows recorded, and such an image may have a semantic near-duplicate the
    # regroup could no longer see. Rescan with embeddings on to group it afresh.
    semantic_groups_stale: bool = False

    reject_reasons: dict[str, int] = Field(default_factory=dict)
    face_clusters: list[FaceCluster] = Field(default_factory=list)
    # When FaceConfig.set_age asked: the set age given to the frames with no
    # face, or why none was (``set_age_skipped``). Both None when not asked.
    set_age: SetAge | None = None
    set_age_skipped: str | None = None
    # Why the scan's reference (FaceConfig.reference_paths or
    # reference_embedding) gave no reference distances (no such image in the
    # scan, none with a face, or an embedding of the wrong length), else None.
    # Set only when a reference was asked for.
    reference_skipped: str | None = None
    # What each image's likeness_distance was measured from (#191), or why
    # none was measured (no reference image in the scan, none embedded, an
    # embedding of the wrong length). Both None when embeddings were off.
    likeness_source: LikenessSource | None = None
    likeness_skipped: str | None = None
    # When ConsentConfig.record asked: whether the record makes the set
    # eligible, and why not. None when not asked.
    consent: ConsentCheck | None = None
    results: list[ImageResult] = Field(default_factory=list)
    # A speech scan's thresholds and per-clip rows (#186); None and empty for an
    # image scan. ``total``, ``passed`` ... count clips on a speech scan.
    speech_config: SpeechConfig | None = None
    clips: list[ClipResult] = Field(default_factory=list)
    # SpeechConfig.speakers (#187): the speakers found, largest first, and how
    # many clips' corpus speaker labels disagree with them.
    speaker_clusters: list[SpeakerCluster] = Field(default_factory=list)
    speaker_mismatches: int = 0

    # The min_short_side a scan cached before MIN_SHORT_SIDE existed actually ran
    # with, when it was below the floor; None otherwise. ``config`` reads the
    # floor (it is a valid config to scan with again), so this is what says the
    # pass/reject counts were made with the lower threshold.
    legacy_min_short_side: int | None = None

    # Pagination metadata (server fills these when slicing ``results``)
    offset: int = 0
    limit: int | None = None
    returned: int = 0

    # Each passing image's unit semantic embedding (a numpy vector), by rel_path;
    # on a speech scan each passing clip's unit speaker embedding. Memory only: a private attribute is neither serialised nor in the schema,
    # so it never reaches the store's JSON or a response. ScanStore holds it per
    # scan and drops it with the scan (#188).
    _embeddings: dict[str, Any] = PrivateAttr(default_factory=dict)

    @property
    def embeddings(self) -> dict[str, Any]:
        """The scan's in-memory semantic embeddings by ``rel_path`` (empty when none are held)."""
        return self._embeddings

    @property
    def is_speech(self) -> bool:
        """True for a speech scan (#186): its rows are ``clips``, never image ``results``."""
        return self.target_profile.modality == "speech"

    @field_validator("faces_config")
    @classmethod
    def _no_reference_template(cls, value: FaceConfig) -> FaceConfig:
        # A caller's reference_embedding is a biometric template: a summary is
        # stored and served, so it never carries one (reference_skipped and
        # each face's reference_distance say what it measured).
        if value.reference_embedding is None:
            return value
        return value.model_copy(update={"reference_embedding": None})

    @field_validator("embeddings_config")
    @classmethod
    def _no_likeness_template(cls, value: EmbeddingConfig) -> EmbeddingConfig:
        # Like the face reference above: a summary is stored and served, so a
        # caller's likeness centroid never rides on it (likeness_source,
        # likeness_skipped and each image's likeness_distance say what it measured).
        if value.reference_embedding is None:
            return value
        return value.model_copy(update={"reference_embedding": None})

    @model_validator(mode="before")
    @classmethod
    def _config_below_the_floor(cls, value: object) -> object:
        # The floor governs new scans. One cached with a lower value must still load,
        # and must still say what it ran with: that moves to legacy_min_short_side.
        if isinstance(value, dict) and isinstance(config := value.get("config"), dict):
            side = config.get("min_short_side")
            if isinstance(side, int) and side < MIN_SHORT_SIDE:
                return {
                    **value,
                    "config": {**config, "min_short_side": MIN_SHORT_SIDE},
                    "legacy_min_short_side": side,
                }
        return value


# ---------------------------------------------------------------------------
# Request / response bodies for the FastAPI server
# ---------------------------------------------------------------------------


def zero_shot_asked(cfg: ScanConfig, prefilter: PrefilterConfig | None = None) -> str | None:
    """The first zero-shot setting *cfg* / *prefilter* turn on (they need embeddings), else None."""
    for name, on in (
        ("config.profile_fit", cfg.profile_fit),
        ("config.junk_check", cfg.junk_check),
        ("prefilter.drop_junk", prefilter is not None and prefilter.drop_junk),
    ):
        if on:
            return name
    return None


class ScanRequest(BaseModel):
    # The server resolves this under its configured scan root and refuses
    # traversal escapes; relative paths are canonical ("" scans the root).
    folder: str
    # ``{"modality": "speech", ...}`` scans the folder's wav/flac clips with
    # ``speech`` instead of its images (#186). The image checks below are
    # then refused, never silently skipped.
    target_profile: ScanProfile = Field(default_factory=TargetProfile)
    config: ScanConfig = Field(default_factory=ScanConfig)
    faces: FaceConfig = Field(default_factory=FaceConfig)
    scenes: SceneConfig = Field(default_factory=SceneConfig)
    garments: GarmentConfig = Field(default_factory=GarmentConfig)
    embeddings: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    nsfw: NsfwConfig = Field(default_factory=NsfwConfig)
    guard: GuardConfig = Field(default_factory=GuardConfig)
    prefilter: PrefilterConfig = Field(default_factory=PrefilterConfig)
    crops: CropConfig = Field(default_factory=CropConfig)
    pose: PoseConfig = Field(default_factory=PoseConfig)
    consent: ConsentConfig = Field(default_factory=ConsentConfig)
    speech: SpeechConfig = Field(default_factory=SpeechConfig)

    @property
    def is_speech(self) -> bool:
        return self.target_profile.modality == "speech"

    @model_validator(mode="after")
    def _semantic_duplicates_need_embeddings(self) -> ScanRequest:
        # Refused, never silently pHash-only: the grouping needs the vectors.
        if self.is_speech:
            return self  # refused below as an image check
        if self.config.semantic_duplicates and not self.embeddings.enabled:
            raise ValueError(SEMANTIC_DUPLICATES_NEED_EMBEDDINGS)
        if not self.embeddings.enabled and (what := zero_shot_asked(self.config, self.prefilter)):
            raise ValueError(ZERO_SHOT_NEEDS_EMBEDDINGS.format(what=what))
        return self

    @model_validator(mode="after")
    def _no_image_checks_on_speech(self) -> ScanRequest:
        if not self.is_speech:
            return self
        asked = [
            name
            for name, on in (
                ("faces", self.faces.enabled),
                ("scenes", self.scenes.enabled),
                ("garments", self.garments.enabled),
                ("embeddings", self.embeddings.enabled),
                ("config.semantic_duplicates", self.config.semantic_duplicates),
                ("config.profile_fit", self.config.profile_fit),
                ("config.junk_check", self.config.junk_check),
                ("nsfw", self.nsfw.enabled is True),
                ("guard", self.guard.enabled is True),
                ("crops", self.crops.enabled),
                ("pose", self.pose.enabled is True),
                ("prefilter", any(self.prefilter.model_dump().values())),
                ("consent", self.consent.record is not None),
            )
            if on
        ]
        if asked:
            raise ValueError(f"a speech scan runs no image checks: turn off {', '.join(asked)}")
        return self


class FaceSourceFilter(BaseModel):
    """The face-source preset: images a head can be built from (argus-mask).

    Every check reads the primary face; ``None`` turns a check off. The
    defaults are the preset, so ``{}`` asks for all of it: one identity, one
    face, frontal and level, large and sharp, eyes open, mouth closed, close
    to the identity, and an adult age. An image that fails is kept out with
    ``face-source:<check>`` as its keep reason, and the kept images are ranked
    best first.
    """

    # The identity to keep (a face cluster id). None = the scan's largest, or
    # with a scan reference any (the reference distance names it).
    cluster: str | None = None
    # Exactly one face in the frame.
    single_face: bool = True
    # Head pose limits, absolute degrees.
    max_yaw: float | None = Field(15.0, ge=0.0)
    max_pitch: float | None = Field(15.0, ge=0.0)
    max_roll: float | None = Field(10.0, ge=0.0)
    # Floor on the face box's short side, in pixels.
    min_face_px: int | None = Field(256, ge=1)
    # Floor on FaceDetection.sharpness.
    min_face_sharpness: float | None = Field(100.0, ge=0.0)
    # Floor on FaceDetection.eye_open, ceiling on FaceDetection.mouth_open.
    min_eye_open: float | None = Field(0.18, ge=0.0)
    max_mouth_open: float | None = Field(0.1, ge=0.0)
    # Ceiling on the identity distance: reference_distance when the scan had
    # a reference (FaceConfig.reference_paths or reference_embedding), else
    # cluster_distance.
    max_identity_distance: float | None = Field(0.4, ge=0.0, le=2.0)
    # A MiVOLO age of at least 18 on the face, and an adult set age when the
    # scan read one. An image with no age is kept out.
    require_adult: bool = True
    # A consented face set (#133): keep only images bound to the scan's consent
    # record (ConsentConfig), re-checked at export, with an adult set age and no
    # matched face under 18; refuse when none qualifies. Rows then carry the
    # record's hashes and each image's provenance.
    consent: bool = False

    @model_validator(mode="after")
    def _consent_is_adult_only(self) -> FaceSourceFilter:
        if self.consent and not self.require_adult:
            raise ValueError("face_source.consent keeps adults only; it cannot run with require_adult false")
        return self


# How the kept images are ordered, best first (selection.rank_key). "score":
# the curation score, then the primary face's identity distance. "identity":
# the identity distance (reference_distance, else cluster_distance; no face
# last), then the score. "likeness" (#191): the image's likeness_distance
# (its CLIP embedding's distance to the subject; none last), then the score,
# for a subject without a usable face. Ties fall back to rel_path, so the
# order is total.
RankBy = Literal["score", "identity", "likeness"]

# Which part of a train/held-out split an image is in.
SplitPart = Literal["train", "held_out"]

# What keeps images together across a split (selection.SPLIT_GROUP_KEYS):
# "image" = every image is its own group; "face_cluster" = the primary face's
# identity, so one identity never lands in both parts (an image with no
# clustered face is a group of its own). "speaker" (speech scans only, and the
# only grouping a speech split takes): a speaker never lands in both parts (see
# speech_select.speaker_groups). "similar_group": the scan's near-duplicate
# group (pHash, and CLIP embeddings with ScanConfig.semantic_duplicates), so
# no image has a near-duplicate in the other part (#191).
SplitGroup = Literal["image", "face_cluster", "similar_group", "speaker"]


class SplitConfig(BaseModel):
    """A deterministic train/held-out split of the kept images (#183).

    Set exactly one of ``held_out`` (images) or ``held_out_fraction``. Whole
    groups (``group_by``) go to one part, never both; a group that would take
    the held-out part past its target is skipped, so the held-out count can
    fall short of it, never exceed it. Both parts must end up non-empty.

    ``method`` "spread" takes held-out groups evenly through the ranking, so
    the held-out images are as good as the training ones (argus-mint's rule);
    "random" shuffles the groups with ``seed``. The same request on the same
    scan always gives the same split.
    """

    held_out: int | None = Field(None, ge=1)
    held_out_fraction: float | None = Field(None, gt=0.0, lt=1.0)
    group_by: SplitGroup = "image"
    method: Literal["spread", "random"] = "spread"
    seed: int = 0

    @model_validator(mode="after")
    def _one_size(self) -> SplitConfig:
        if (self.held_out is None) == (self.held_out_fraction is None):
            raise ValueError("split needs exactly one of held_out or held_out_fraction")
        return self


class SpeechSelection(BaseModel):
    """How a speech scan's clips are selected, and where they came from (#187).

    The speech part of :class:`ExportRequest` and :class:`SelectRequest`, and
    the only part a speech scan reads besides ``min_score``, ``keep_similar``,
    ``max_keep``, ``split`` and the transfer fields; the image filters are
    refused on a speech scan.

    Filters: ``speakers`` (by ``speaker_id``), ``accents`` and ``languages``
    (the corpus labels, exact). ``exclude_speaker_mismatch`` drops clips whose
    corpus speaker label the clustering disputes. Caps, in hours of audio,
    best-scoring clips first: ``min_hours_per_speaker`` drops a speaker with
    less, ``max_hours_per_speaker`` caps one, ``max_hours_per_accent`` caps an
    accent, and ``balance_accents`` caps every accent at the smallest one's
    hours. An accent cap takes clips from its speakers in turn, so it keeps
    them as even as it can.

    Provenance, required to export (not to select): ``corpus`` and
    ``license`` name the licensed corpus every clip comes from; or
    ``consent_record`` names a cortex ``ConsentRecord`` (a path under the
    scanned folder, as ``ConsentConfig.record``) for a real person's voice
    outside a licensed corpus, checked for ``training`` of the ``voice`` at
    ``consent_delivery``. The record binds the export to its reference
    recordings' voice (#209): a clip is kept only when its speaker embedding
    is within ``consent_max_distance`` (cosine distance) of their centroid.
    Nothing estimates a speaker's age: the corpus's label is carried, and an
    unknown age stays unknown.
    """

    speakers: list[str] | None = Field(None, min_length=1)
    accents: list[str] | None = Field(None, min_length=1)
    languages: list[str] | None = Field(None, min_length=1)
    exclude_speaker_mismatch: bool = False

    min_hours_per_speaker: float | None = Field(None, gt=0.0)
    max_hours_per_speaker: float | None = Field(None, gt=0.0)
    max_hours_per_accent: float | None = Field(None, gt=0.0)
    balance_accents: bool = False

    corpus: str | None = Field(None, min_length=1, max_length=200)
    license: str | None = Field(None, min_length=1, max_length=200)
    consent_record: str | None = Field(None, min_length=1)
    consent_delivery: str | None = Field(None, min_length=1, max_length=100)
    # Ceiling on a clip's cosine distance to the centroid of the consent
    # record's reference recordings (#209). Strict on purpose: a clip of
    # another voice let in is the harmful error, a clip of the subject's
    # left out the safe one (docs/speech.md has the measurements).
    consent_max_distance: float = Field(VOICE_CONSENT_MAX_DISTANCE, gt=0.0, lt=2.0)

    @model_validator(mode="after")
    def _provenance(self) -> SpeechSelection:
        if (self.corpus is None) != (self.license is None):
            raise ValueError("speech.corpus and speech.license go together: name the corpus and its licence")
        if (self.consent_record is None) != (self.consent_delivery is None):
            raise ValueError("speech.consent_record needs speech.consent_delivery (the voice delivery it covers)")
        if self.corpus is not None and self.consent_record is not None:
            raise ValueError("a speech export is from a licensed corpus or under a consent record, not both")
        low, high = self.min_hours_per_speaker, self.max_hours_per_speaker
        if low is not None and high is not None and low > high:
            raise ValueError("speech.min_hours_per_speaker is above speech.max_hours_per_speaker")
        return self

    @property
    def has_provenance(self) -> bool:
        return self.corpus is not None or self.consent_record is not None


class ExportRequest(BaseModel):
    scan_id: str | None = None
    # Inline selection of rel_paths (alternative to scan_id-driven selection).
    selection: list[str] | None = None

    # The server resolves this under its configured export root and refuses
    # traversal escapes; relative paths are canonical. Library/CLI callers
    # pass a filesystem path directly.
    dest: str
    # "move" is destructive; the server rejects it unless started with
    # --allow-move / CURATOR_ALLOW_MOVE=1. An image the scan cropped is written
    # cropped, which neither a link nor a move can do: selecting one refuses
    # any mode but "copy".
    mode: ExportMode = "copy"
    preserve_structure: bool = True

    min_score: float = 0.6
    include_rejected: bool = Field(False, description=_INCLUDE_REJECTED_IGNORED)
    keep_similar: bool = False
    # Cap on the kept images: the best by `rank`, spread by diversity_weight
    # (and the diversify flags). /scan/{id}/select applies the same cap.
    max_keep: int | None = Field(None, ge=0)
    # The max_keep blend, 0 = best by rank only, 1 = widest spread. None = the
    # scan's ScanConfig.diversity_weight.
    diversity_weight: float | None = Field(None, ge=0.0, le=1.0)
    # How kept images are ordered (RankBy), in the manifest too. "identity"
    # needs a scan that ran face detection, "likeness" one that measured
    # likeness distances (EmbeddingConfig).
    rank: RankBy = "score"
    # Ceiling on ImageResult.likeness_distance (#191); an image without one is
    # kept out ("likeness-distance"). Needs a scan that measured them.
    max_likeness_distance: float | None = Field(None, ge=0.0, le=2.0)
    # Keep the scan's reference images (faces.reference_paths and
    # embeddings.reference_paths) out of the selection, keep reason
    # "reference": the yardstick, not a candidate (argus-mint's anchors), so a
    # cap or a split never spends a place on one. Refused on a scan with none.
    exclude_references: bool = False
    # Optional train/held-out split of the kept images; the export transfers
    # only split_part of it, and each manifest row names its part. Two exports
    # with the same filter and split, one per part, give disjoint sets.
    split: SplitConfig | None = None
    split_part: SplitPart = "train"

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
    # With max_keep: also spread the selection semantically (one minus the
    # largest embedding cosine to anything selected), so burst shots, re-encodes
    # and small crops of one frame do not crowd out real variety. Needs a scan
    # that ran EmbeddingConfig and whose embeddings this process still holds.
    diversify_semantic: bool = False

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
    # Optional: keep only images with one of these content ratings. An image
    # NudeNet never checked has no rating and is excluded once this is set.
    content_ratings: list[ContentRating] | None = Field(None, min_length=1)

    # Optional: drop images SingGuard judged unsafe, under any rule.
    guard_unsafe: bool = False
    # Optional: drop images where one of these rules fired, named by letter
    # ("A") or in full ("A. Sexual Content Risk"), case-insensitive.
    guard_rules: list[str] | None = None
    # Which policies' verdicts guard_unsafe / guard_rules read (None = the
    # official one). An image is dropped if any listed verdict fails, or is
    # missing: the union, so an export matches a "flagged by any" view.
    guard_policies: list[GuardPolicy] | None = Field(None, min_length=1)

    # Optional: the face-source preset (FaceSourceFilter). Kept images are
    # then ordered best first, in the manifest too.
    face_source: FaceSourceFilter | None = None

    write_manifest: bool = True
    # Write the primary face's landmarks (FaceLandmarks) into manifest rows.
    landmarks: bool = False
    # Optionally POST the manifest straight to argus-lens /caption (M4 stretch).
    caption_url: str | None = None
    # Write an XMP sidecar beside each exported file (<file>.xmp): face regions,
    # the names in face_names, and the content rating. A sidecar beside the
    # source (lens's captions, a photo app's edits) is the starting point and
    # its other properties are kept. The source's own sidecar is never changed.
    write_xmp: bool = False
    # Person names by face cluster id ("face_1" -> "Ada"), for the regions and
    # Iptc4xmpExt:PersonInImage. Two ids may share a name. Unnamed clusters get
    # an unnamed region and no person entry: "face_1" is not a person's name.
    face_names: dict[str, str] | None = None
    # Transfer the sidecars beside each exported source image with it, in the
    # same mode (but copied under mode="symlink", so writing a caption into the
    # export never reaches the source), named after the destination file:
    # <stem>.txt (lens captions), <file>.xmp (lens, digiKam), <stem>.xmp
    # (Lightroom) and <stem>.json. Under mode="move" they would otherwise be
    # orphaned in the source folder. With
    # write_xmp, curator's data is merged onto the carried <file>.xmp.
    carry_sidecars: bool = False
    # A speech scan's selection and provenance (#187); refused on an image scan.
    speech: SpeechSelection | None = None
    # Opaque client JSON (e.g. a cull session's stats), written pretty-printed
    # to <dest>/run_stats.json when the export produced a destination folder
    # (a file was transferred, or write_manifest is on). At most
    # RUN_STATS_MAX_BYTES once serialised.
    run_stats: dict[str, Any] | None = None

    @field_validator("run_stats")
    @classmethod
    def _run_stats_size(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value is None:
            return None
        try:
            size = len(json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"run_stats is not JSON-serialisable: {exc}") from exc
        if size > RUN_STATS_MAX_BYTES:
            raise ValueError(f"run_stats is {size} bytes serialised; the limit is {RUN_STATS_MAX_BYTES}")
        return value

    @field_validator("face_names")
    @classmethod
    def _names_are_names(cls, value: dict[str, str] | None) -> dict[str, str] | None:
        if value is None:
            return None
        names = {cluster.strip(): name.strip() for cluster, name in value.items()}
        for cluster, name in names.items():
            if not cluster or not name or len(name) > 200:
                raise ValueError(f"face_names[{cluster!r}]: a name must be 1-200 characters")
        return names

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
    def _split_needs_the_filter(self) -> ExportRequest:
        if self.split is not None and self.selection is not None:
            raise ValueError("split applies to the filter's selection; a named selection is already split")
        return self

    @model_validator(mode="after")
    def _consent_needs_the_filter(self) -> ExportRequest:
        if self.face_source is not None and self.face_source.consent and self.selection is not None:
            raise ValueError("a consented face set is chosen by face_source, not by a named selection")
        return self

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


class ManifestFace(BaseModel):
    """The primary face of a manifest row (manifest 2.3)."""

    model_config = RESPONSE_ONLY

    bbox: list[float] = Field(..., min_length=4, max_length=4, description="[x, y, w, h]")
    yaw: float | None = None
    pitch: float | None = None
    roll: float | None = None
    sharpness: float | None = None
    eye_open: float | None = None
    mouth_open: float | None = None
    cluster_distance: float | None = None
    reference_distance: float | None = None
    # MiVOLO's age in years, when it ran.
    age_years: float | None = None
    # Only when the export asked (ExportRequest.landmarks).
    landmarks: FaceLandmarks | None = None


class ManifestRow(BaseModel):
    """One line of the JSONL handoff manifest (written by ``export.write_manifest``).

    ``exported_path`` is the path actually written under the export root
    (posix, relative to it). Consumers must use it instead of re-deriving a
    destination from ``rel_path`` — flattened exports de-collide basenames, and
    with carried sidecars a root image cannot take the stem of an export file,
    so the two can differ. Rows exist only for files whose transfer succeeded.

    ``exported_abs_path`` is that same file's absolute location, so a row is
    self-contained: the manifest is often read somewhere other than the export
    root it was written into (argus-lens receives it as an upload), and nothing
    in the row names the root to join ``exported_path`` onto.

    ``abs_path`` is where the image can be *read*, which is not always where it
    came from: ``mode="move"`` deletes the source during transfer, so for a
    moved export it equals ``exported_abs_path``. Use ``rel_path`` for the
    original location within the scan.
    """

    model_config = RESPONSE_ONLY

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
    # The crop baked into the exported file (manifest 2.2), in the source's
    # displayed pixels. None = the file is the source as it was.
    crop: CropBox | None = None
    # Manifest 2.3: the primary face's measurements (None = no face), and the
    # scan's set age when FaceConfig.set_age read one.
    primary_face: ManifestFace | None = None
    set_age: SetAge | None = None
    # Manifest 2.4: the row's part of ExportRequest.split, None without one.
    split: SplitPart | None = None
    # Manifest 2.5, a consented face set only (face_source.consent): the
    # consent record's key (its release's sha256), the sha256 of the record
    # file as read at export, and where this image came from (its <stem>.json
    # sidecar's provenance, else captured under the release). None otherwise.
    consent_release_sha256: str | None = None
    consent_record_sha256: str | None = None
    provenance: Provenance | None = None
    # Manifest 2.7: the primary person's body-pose labels and the scan-level
    # pose cluster (ImageResult.pose_labels / pose_cluster). None = the pose
    # stage did not run, found nobody, failed, or (cluster) the pose was
    # unusable or clustering was skipped. Clusters span the whole scan.
    pose_labels: BodyPoseLabels | None = None
    pose_cluster: int | None = None


class SpeechManifestRow(BaseModel):
    """One line of a speech export's ``speech_manifest.jsonl`` (#187; ``speech_export.write_manifest``).

    Its own shape and version (:data:`SPEECH_MANIFEST_VERSION`), never mixed
    with image rows: a consumer reads ``speech_manifest_version`` and refuses a
    major it does not know. Rows exist only for clips whose transfer and
    transcript sidecar both succeeded.

    Paths follow :class:`ManifestRow`: ``exported_path`` and
    ``exported_abs_path`` are the de-collided files on disk, ``abs_path`` is
    where the clip can be read (the destination under ``mode="move"``), and
    ``transcript_path`` / ``transcript_abs_path`` name the ``<stem>.txt``
    beside the exported clip holding ``transcript``.

    ``speaker_id`` is the corpus's speaker label when it has one, else the
    speaker cluster; ``speaker_label`` and ``speaker_cluster`` are each on its
    own. A split keeps every speaker (label and cluster alike) in one
    part. ``accent``, ``language``, ``gender`` and ``age`` are the corpus's
    labels verbatim; ``age`` is null when the corpus gives none, and is never
    estimated. ``corpus``, ``license`` and ``provenance`` say where every clip
    came from; a clip used under a consent record also names its hashes.
    """

    model_config = RESPONSE_ONLY

    speech_manifest_version: str = SPEECH_MANIFEST_VERSION
    modality: Literal["speech"] = "speech"
    rel_path: str
    abs_path: str
    exported_path: str
    exported_abs_path: str
    transcript_path: str
    transcript_abs_path: str
    target_profile: SpeechTargetProfile
    transcript: str
    duration_s: float
    sample_rate: int
    channels: int
    speaker_id: str | None = None
    speaker_label: str | None = None
    speaker_cluster: str | None = None
    speaker_mismatch: bool = False
    accent: str | None = None
    language: str | None = None
    gender: str | None = None
    age: str | None = None
    score: float
    similar_group: int
    split: SplitPart | None = None
    corpus: str
    license: str
    provenance: Provenance
    consent_release_sha256: str | None = None
    consent_record_sha256: str | None = None


class ExportResult(BaseModel):
    model_config = RESPONSE_ONLY

    # Version of the manifest this curator writes, declared whether or not a
    # manifest was requested. Without it a client can only compat-detect by
    # sniffing which fields are present, which cannot distinguish "old server"
    # from "new server, nothing transferred" (both yield an empty mapping).
    manifest_version: str = MANIFEST_VERSION
    manifest_path: str | None = None
    copied: int
    skipped: int
    dest: str
    mode: ExportMode
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
    # client can reproduce, and with carry_sidecars a root image with a reserved
    # stem gets the same suffix. The process that owns the rule publishes the answer.
    exported_abs_paths: dict[str, str] = Field(default_factory=dict)
    captioned: bool = False
    # Sidecars written beside exported files (write_xmp). A file with nothing
    # to record (no faces, never checked by NudeNet) gets none. Transfer can
    # succeed while a sidecar fails (copied > xmp_written); the image still
    # landed, and the failure is logged rather than undoing it.
    xmp_written: int = 0
    # Sidecar files transferred beside exported images (carry_sidecars). One
    # image can carry several (.txt, .xmp, .json). A sidecar that fails to
    # transfer does not undo its image: it is logged and counted in
    # sidecars_failed instead.
    sidecars_carried: int = 0
    sidecars_failed: int = 0
    # Exported images written cropped (ImageResult.crop), counted in copied too.
    cropped: int = 0
    # Where run_stats was written (<dest>/run_stats.json), or None when the
    # request carried none, nothing was exported, or the write failed.
    run_stats_path: str | None = None
    # What was exported: an image scan's images, or a speech scan's clips
    # (#187), whose manifest is speech_manifest.jsonl, versioned by
    # speech_manifest_version (declared whatever was exported, as
    # manifest_version is).
    modality: Literal["image", "speech"] = "image"
    speech_manifest_version: str = SPEECH_MANIFEST_VERSION
    # Speech: the <stem>.txt transcripts written beside the exported clips.
    transcripts_written: int = 0


class SelectRow(BaseModel):
    """One image under a dry-run selection.

    ``keep_reason`` is empty when the image is in the selection. A non-empty
    value is why ``decide_selection`` left it out — the same strings an export
    report would write, without copying anything.
    """

    model_config = RESPONSE_ONLY

    rel_path: str
    abs_path: str
    keep_reason: str = ""
    # 1-based position among the kept images, best first (SelectRequest.rank);
    # None for an image left out.
    rank: int | None = None
    # Its part of SelectRequest.split; None without a split or when left out.
    split: SplitPart | None = None
    # A speech scan's clip (#187): its speaker_id (SpeechManifestRow), else None.
    speaker_id: str | None = None
    # Under speech.consent_record (#209): the clip's cosine distance to the
    # record's reference voice, for review of the borderline ones; None
    # otherwise, or for a clip with no speaker embedding. Never the vector.
    consent_distance: float | None = None


class SelectRequest(BaseModel):
    """Filter body for ``POST /scan/{id}/select``.

    The filter, ranking, cap and split fields of :class:`ExportRequest`, and
    none of the transfer fields (``dest``, ``mode``, ``selection``). Select and
    export run the same ``selection.plan_selection``, so an export with the
    same fields keeps exactly the rows this route marks kept, in this order.
    """

    min_score: float = 0.6
    include_rejected: bool = Field(False, description=_INCLUDE_REJECTED_IGNORED)
    keep_similar: bool = False
    max_keep: int | None = Field(None, ge=0)
    diversity_weight: float | None = Field(None, ge=0.0, le=1.0)
    diversify_scenes: bool = False
    diversify_outfits: bool = False
    diversify_semantic: bool = False
    rank: RankBy = "score"
    max_likeness_distance: float | None = Field(None, ge=0.0, le=2.0)
    exclude_references: bool = False
    split: SplitConfig | None = None

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
    content_ratings: list[ContentRating] | None = Field(None, min_length=1)

    guard_unsafe: bool = False
    guard_rules: list[str] | None = None
    guard_policies: list[GuardPolicy] | None = Field(None, min_length=1)

    face_source: FaceSourceFilter | None = None
    speech: SpeechSelection | None = None

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


class SpeechSelectStats(BaseModel):
    """What a speech selection keeps, in hours of audio (#187)."""

    model_config = RESPONSE_ONLY

    hours: float
    hours_by_speaker: dict[str, float] = Field(default_factory=dict)
    # Keyed by the corpus accent label; "" for clips without one.
    hours_by_accent: dict[str, float] = Field(default_factory=dict)
    # With a split: hours and speakers in each part.
    hours_by_split: dict[str, float] = Field(default_factory=dict)
    speakers_by_split: dict[str, int] = Field(default_factory=dict)


class SelectResult(BaseModel):
    """What the current filter would keep, without writing the export root.

    ``rows`` lists the kept images first, best first by ``rank`` (each row's
    ``rank`` is its position), then the images left out, in scan order.
    """

    model_config = RESPONSE_ONLY

    scan_id: str
    total: int
    kept: int
    rank: RankBy = "score"
    # With a split: how many kept images each part holds (train + held_out == kept).
    train: int | None = None
    held_out: int | None = None
    rows: list[SelectRow] = Field(default_factory=list)
    # A speech scan's selection: hours by speaker, accent and split part.
    speech: SpeechSelectStats | None = None


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
    SpeechManifestRow,
    SelectRequest,
    SelectResult,
)


WIRE_SCHEMA_TITLE = "argus-curator wire contract"


def wire_schema() -> dict:
    """Combined JSON Schema for the curator's wire contract (all WIRE_MODELS)."""
    return _cortex_wire_schema(WIRE_MODELS, title=WIRE_SCHEMA_TITLE)


def render_wire_schema() -> str:
    """The committed ``schema/curator-wire.schema.json`` text, formatted by argus-cortex."""
    return _cortex_render_schema(WIRE_MODELS, title=WIRE_SCHEMA_TITLE)
