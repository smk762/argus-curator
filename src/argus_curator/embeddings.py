"""Semantic image embeddings: one CLIP image vector per passing image (#188).

Runs the image tower of DataComp-1B's CLIP ViT-L/14
(``laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K``, MIT; 79.2% zero-shot top-1 on
ImageNet-1k) on each passing image and keeps its L2-normalised 768-d vector.
Selection reads them for a semantic novelty term (``diversify_semantic``):
frames with the same pose and framing but different pixels (burst shots,
re-encodes, small crops, mirrors) are near each other here even when their
pHashes are far apart. The scan also measures each image's likeness distance
to the subject (:func:`measure_likeness`, #191), a plain float on its row.

The vectors are never written anywhere. They ride on the scan's summary as a
private attribute (``ScanSummary.embeddings``), which neither the store's JSON
nor a response carries, and :class:`~argus_curator.store.ScanStore` holds them
in memory for the scan and drops them with it.

The model runs under onnxruntime; ``argus-curator embeddings-convert``
downloads the pinned checkpoint, verifies it, and exports the image tower to
ONNX once.
"""

from __future__ import annotations

import io
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import structlog
from PIL import Image

from argus_curator.models import EmbeddingConfig, ImageResult, LikenessSource
from argus_curator.runtime import (
    GPU_UNAVAILABLE,
    ModelUnavailable,
    configured_path,
    gpu_lease,
    gpu_mem_limit_mb,
    load_session,
    onnxruntime_available,
    providers,
    sha256,
)

logger = structlog.get_logger()


@dataclass(frozen=True)
class EmbeddingSpace:
    """A pinned CLIP image tower: where its weights come from and how it reads an image."""

    hf_repo: str
    hf_revision: str
    weights_sha256: str
    onnx_name: str
    dim: int
    input_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

EMBEDDING_SPACES: dict[str, EmbeddingSpace] = {
    "datacomp-xl-vit-l-14": EmbeddingSpace(
        hf_repo="laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K",
        hf_revision="84c9828e63dc9a9351d1fe637c346d4c1c4db341",
        # pytorch_model.bin (the transformers CLIPModel checkpoint) at that revision.
        weights_sha256="cbae707ef72c3d2748f3c9482e9ca8645d8bbd915d59a86e4f11107c9b341068",
        onnx_name="clip_datacomp_xl_vit_l_14_image.onnx",
        dim=768,
        input_size=224,
        mean=_CLIP_MEAN,
        std=_CLIP_STD,
    ),
}

# The one embedding space curator computes. Vectors from different spaces are
# not comparable (argus-quarry uses OpenAI's ViT-L/14, the space PD12M
# publishes), so if the suite ever compares them across stages this choice
# moves to argus-cortex. Change it here only, never per call.
EMBEDDING_SPACE = "datacomp-xl-vit-l-14"

MODEL_ENV = "ARGUS_CURATOR_EMBEDDINGS_MODEL"

# Images per forward pass. Every pass has this shape (a short last batch is
# padded), so the capped CUDA arena reuses the same blocks each time (#189).
BATCH = 8

# The image tower's fp32 weights alone are ~1.2 GB, past the default per-session
# arena cap (runtime.DEFAULT_GPU_MEM_LIMIT_MB): under a 1 GB cap the session
# fails to initialise. So this session's cap is at least this; it stays
# kSameAsRequested and capped (measured peak 1,680 MB process VRAM).
ARENA_MB = 2048

HF_FILES = ("config.json", "pytorch_model.bin")


def space() -> EmbeddingSpace:
    return EMBEDDING_SPACES[EMBEDDING_SPACE]


def model_path() -> Path:
    """Where the converted image-tower ONNX model is read from."""
    return configured_path(MODEL_ENV, space().onnx_name)


def embeddings_available() -> bool:
    """True when onnxruntime is importable and the image-tower model is in place."""
    return onnxruntime_available() and model_path().is_file()


def arena_mb() -> int:
    """This session's CUDA arena cap: the configured per-session cap, raised to :data:`ARENA_MB`."""
    return max(gpu_mem_limit_mb(), ARENA_MB)


def get_session(provs: list[str]):
    return load_session(
        model_path(),
        provs,
        missing_hint=f"Create it with `argus-curator embeddings-convert` or set {MODEL_ENV}.",
        extra="scenes",
        cuda_options={"gpu_mem_limit": str(arena_mb() * 1024 * 1024)},
    )


def preprocess(images: list[Image.Image]) -> np.ndarray:
    """CLIP's transform: short side to 224 (bicubic), centre-crop 224, CLIP-normalised."""
    sp = space()
    size = sp.input_size
    rows = []
    for im in images:
        rgb = im.convert("RGB")
        w, h = rgb.size
        scale = size / min(w, h)
        nw, nh = max(size, round(w * scale)), max(size, round(h * scale))
        rgb = rgb.resize((nw, nh), Image.Resampling.BICUBIC)
        left, top = (nw - size) // 2, (nh - size) // 2
        rows.append(np.asarray(rgb.crop((left, top, left + size, top + size)), dtype=np.float32) / 255.0)
    arr = (np.stack(rows) - np.array(sp.mean, dtype=np.float32)) / np.array(sp.std, dtype=np.float32)
    return arr.transpose(0, 3, 1, 2).astype(np.float32)


def normalise(vectors: np.ndarray) -> np.ndarray:
    """Rows scaled to unit length (a zero row stays zero), so a dot product is the cosine."""
    v = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, norms, out=np.zeros_like(v), where=norms > 0)


def embed(session, images: list[Image.Image]) -> np.ndarray:
    """``(N, dim)`` unit vectors for up to :data:`BATCH` images, run as one full batch."""
    batch = preprocess(images)
    n = len(batch)
    if n < BATCH:
        batch = np.concatenate([batch, np.zeros((BATCH - n, *batch.shape[1:]), dtype=np.float32)])
    input_name = session.get_inputs()[0].name
    (out,) = session.run(None, {input_name: batch})
    return normalise(out[:n])


def embed_images(
    results: list[ImageResult],
    items: list[tuple[str, str, bytes]],
    cfg: EmbeddingConfig,
    progress: Callable[[int, int], None] | None = None,
    *,
    also: Collection[str] = (),
) -> dict[str, np.ndarray]:
    """Each passing image's unit embedding, by ``rel_path``, in batches of :data:`BATCH`.

    *also* names images embedded even when rejected (the likeness reference
    images, :func:`measure_likeness`); the caller drops their vectors after.
    An image that fails to decode, or whose batch fails, is left out: selection
    reads a missing embedding as a repeat. Nothing is written to *results*.
    """
    vectors: dict[str, np.ndarray] = {}
    if not cfg.enabled:
        return vectors
    with gpu_lease(cfg.device, "embeddings", "load"):
        session = get_session(providers(cfg.device))
    data_by_rel = {rel: data for rel, _abs, data in items}
    extra = set(also)
    pending = [r for r in results if (r.passed or r.rel_path in extra) and data_by_rel.get(r.rel_path) is not None]
    total = len(pending)
    if progress is not None:
        progress(0, total)
    for start in range(0, total, BATCH):
        batch: list[tuple[ImageResult, Image.Image]] = []
        for r in pending[start : start + BATCH]:
            try:
                # convert() forces the decode, so a truncated file fails here, not the batch.
                batch.append((r, Image.open(io.BytesIO(data_by_rel[r.rel_path])).convert("RGB")))
            except Exception as exc:
                logger.warning("embedding_decode_failed", rel_path=r.rel_path, error=str(exc))
        if batch:
            try:
                with gpu_lease(cfg.device, "embeddings"):
                    out = embed(session, [im for _r, im in batch])
            except GPU_UNAVAILABLE:
                raise  # no GPU slot is not a bad batch: fail the scan, do not skip the images
            except Exception as exc:
                logger.warning("embedding_batch_failed", first=batch[0][0].rel_path, error=str(exc))
            else:
                for (r, _im), row in zip(batch, out, strict=True):
                    vectors[r.rel_path] = row
        if progress is not None:
            progress(min(start + BATCH, total), total)
    return vectors


# ---------------------------------------------------------------------------
# Likeness: each image's distance to the subject (#191)
# ---------------------------------------------------------------------------

# Without a reference the subject is the set's centroid, recentred once on
# this share of the images closest to the plain mean, so a minority of strays
# (another subject, a failed render) pulls it less.
CENTROID_KEEP = 0.75


def _cosine_distance(vector: np.ndarray, unit_centre: np.ndarray) -> float:
    """``1 - cos`` (0..2, floored at 0 and rounded to 4 places), as faces' distances are."""
    return round(max(0.0, 1.0 - float(np.dot(normalise(vector), unit_centre))), 4)


def set_centroid(vectors: list[np.ndarray]) -> np.ndarray:
    """The unit centroid of *vectors*, recentred on the :data:`CENTROID_KEEP` share closest to the plain mean."""
    matrix = normalise(np.stack(vectors))
    centre = normalise(matrix.mean(axis=0))
    keep = max(1, int(np.ceil(CENTROID_KEEP * len(matrix))))
    if keep < len(matrix):
        closest = np.argsort(-(matrix @ centre), kind="stable")[:keep]
        centre = normalise(matrix[closest].mean(axis=0))
    return centre


def measure_likeness(
    results: list[ImageResult],
    vectors: Mapping[str, np.ndarray],
    cfg: EmbeddingConfig,
) -> tuple[LikenessSource | None, str | None]:
    """Set each passing image's ``likeness_distance``; return ``(source, skipped)``.

    The subject is ``cfg.reference_embedding`` when given, else the mean of
    ``cfg.reference_paths``' embeddings (in *vectors*: pass them to
    :func:`embed_images` as ``also``), else the set's own centroid
    (:func:`set_centroid` over the passing images). ``source`` says which;
    ``skipped`` why no distance was measured (a reference is never swapped
    for the centroid). Only the distances are kept: the vectors are the
    caller's, and nothing here stores them.
    """
    if not cfg.enabled:
        return None, None
    passing = [r for r in results if r.passed and r.rel_path in vectors]
    if not passing:
        return None, "no image could be embedded"
    dim = len(vectors[passing[0].rel_path])
    if cfg.reference_embedding is not None:
        if len(cfg.reference_embedding) != dim:
            return None, f"reference_embedding has {len(cfg.reference_embedding)} values; embeddings have {dim}"
        centre = normalise(np.asarray(cfg.reference_embedding, dtype=np.float32))
        source: LikenessSource = "reference"
    elif cfg.reference_paths:
        known = {r.rel_path for r in results}
        missing = [p for p in cfg.reference_paths if p not in known]
        if len(missing) == len(cfg.reference_paths):
            return None, f"no reference image is in the scan ({', '.join(missing)})"
        refs = [vectors[p] for p in cfg.reference_paths if p in vectors]
        if not refs:
            return None, "no reference image could be embedded"
        if missing:
            logger.warning("likeness_reference_paths_missing", missing=missing)
        centre = normalise(normalise(np.stack(refs)).mean(axis=0))
        source = "reference"
    else:
        centre = set_centroid([vectors[r.rel_path] for r in passing])
        source = "centroid"
    if not centre.any():
        return None, "the references cancel out (their mean is zero)"
    for r in passing:
        r.likeness_distance = _cosine_distance(vectors[r.rel_path], centre)
    return source, None


# ---------------------------------------------------------------------------
# One-time conversion
# ---------------------------------------------------------------------------


def fetch_weights(files: tuple[str, ...] = HF_FILES) -> Path:
    """Download the pinned checkpoint's *files* from the Hugging Face Hub (cached there)."""
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        raise ModelUnavailable("Conversion requires: pip install argus-curator[embeddings-convert]") from exc
    sp = space()
    paths = [Path(hf_hub_download(sp.hf_repo, f, revision=sp.hf_revision)) for f in files]
    return paths[0].parent


def export_onnx(checkpoint_dir: Path, output: Path) -> Path:
    """Export the pinned checkpoint's image tower (with its projection) to ONNX.

    The checkpoint is the full CLIP model; only the ``vision_model.*`` and
    ``visual_projection.*`` weights are kept, loaded with ``weights_only`` and
    strictly, so nothing is left randomly initialised. The ONNX model takes
    ``(N, 3, 224, 224)`` CLIP-normalised pixels and returns the unnormalised
    ``(N, 768)`` image embeddings. Needs ``pip install argus-curator[embeddings-convert]``.
    """
    try:
        import torch
        from transformers import CLIPConfig, CLIPVisionModelWithProjection
    except Exception as exc:
        raise ModelUnavailable("Conversion requires: pip install argus-curator[embeddings-convert]") from exc

    sp = space()
    weights = checkpoint_dir / "pytorch_model.bin"
    digest = sha256(weights)
    if digest != sp.weights_sha256:
        raise ModelUnavailable(f"{weights} is not the pinned {sp.hf_repo} checkpoint (sha256 {digest})")

    config = CLIPConfig.from_pretrained(str(checkpoint_dir))
    vision = config.vision_config
    # The checkpoint's vision_config carries a stale projection_dim (512); the
    # projection it ships is the top-level one (768).
    vision.projection_dim = config.projection_dim
    if vision.projection_dim != sp.dim:
        raise ModelUnavailable(f"{sp.hf_repo} projects to {vision.projection_dim}, not {sp.dim}")
    model = CLIPVisionModelWithProjection(vision)
    state = torch.load(str(weights), map_location="cpu", weights_only=True)
    # position_ids is a constant arange buffer, which newer transformers rebuild
    # rather than load, so it is left out on both sides; every weight must match.
    state = {
        k: v
        for k, v in state.items()
        if k.startswith(("vision_model.", "visual_projection.")) and not k.endswith("position_ids")
    }
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if not k.endswith("position_ids")]
    if missing or unexpected:
        raise ModelUnavailable(
            f"checkpoint does not match the image tower (missing {missing}, unexpected {unexpected})"
        )
    model.eval()

    class _Embeds(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            return self.inner(pixel_values=x).image_embeds

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        _Embeds(model).eval(),
        (torch.zeros(BATCH, 3, sp.input_size, sp.input_size),),
        str(output),
        input_names=["input"],
        output_names=["embeds"],
        dynamic_axes={"input": {0: "batch"}, "embeds": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    return output
