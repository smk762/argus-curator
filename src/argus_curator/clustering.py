"""Duplicate detection and embedding-based diversity clustering.

Two-tier approach:
  Tier 1 — pHash Hamming distance: fast, CPU-only near-duplicate detection.
            Runs before any GPU work so we don't waste embedding time on
            burst-shot duplicates.

  Tier 2 — CLIP embedding cosine distance: semantic diversity clustering.
            Groups images by visual/semantic similarity so the selector
            can pick the best representative from each cluster rather than
            greedily maximising pHash spread.
"""

from __future__ import annotations

import math

import imagehash
import numpy as np

from argus_curator.types import ImageResult


# ---------------------------------------------------------------------------
# Tier 1: pHash near-duplicate detection
# ---------------------------------------------------------------------------

def mark_duplicates(results: list[ImageResult], max_hamming: int) -> None:
    """Tag near-duplicates in-place; keep the highest-scoring representative.

    Images are processed in descending score order so the best copy
    is always the one that survives.
    """
    passing = sorted(
        (r for r in results if r.passed),
        key=lambda r: -r.score,
    )
    seen: list[tuple[imagehash.ImageHash, str]] = []
    for result in passing:
        if not result.phash:
            continue
        try:
            ph = imagehash.hex_to_hash(result.phash)
        except Exception:
            continue
        for ref_hash, ref_name in seen:
            if (ph - ref_hash) <= max_hamming:
                result.is_duplicate = True
                result.duplicate_of = ref_name
                break
        else:
            seen.append((ph, result.name))


# ---------------------------------------------------------------------------
# Tier 2: embedding cosine-distance clustering
# ---------------------------------------------------------------------------

def _cosine_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Pairwise cosine distances (already normalised embeddings → 1 - dot)."""
    # embeddings shape: (n, dim), already L2-normalised
    dots = embeddings @ embeddings.T
    dists = 1.0 - np.clip(dots, -1.0, 1.0)
    np.fill_diagonal(dists, 0.0)
    return dists.astype(np.float64)


def cluster_by_embeddings(
    results: list[ImageResult],
    n_clusters: int | None,
) -> int:
    """Assign cluster_id to each candidate (non-duplicate, passing) result.

    Uses agglomerative clustering with cosine distance — no GPU required
    after the embeddings are extracted.

    Returns the actual number of clusters formed.
    """
    from sklearn.cluster import AgglomerativeClustering

    candidates = [r for r in results if r.passed and not r.is_duplicate]
    if len(candidates) < 2:
        for i, r in enumerate(candidates):
            r.cluster_id = 0
        return min(1, len(candidates))

    # Gather CLIP embeddings (fall back to DINOv2 if CLIP absent)
    emb_list = []
    for r in candidates:
        emb = r.clip_embedding or r.dino_embedding
        if emb is not None:
            emb_list.append(emb)
        else:
            emb_list.append(None)

    if all(e is None for e in emb_list):
        # No embeddings available — all in one cluster
        for r in candidates:
            r.cluster_id = 0
        return 1

    # Replace missing embeddings with zero-vector (they'll cluster together)
    dim = next(len(e) for e in emb_list if e is not None)
    matrix = np.array(
        [e if e is not None else [0.0] * dim for e in emb_list],
        dtype=np.float32,
    )
    # Re-normalise (embeddings should already be normalised, but be safe)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    matrix = matrix / norms

    n = len(candidates)
    k = n_clusters or max(2, min(20, int(math.sqrt(n))))
    k = min(k, n)

    clustering = AgglomerativeClustering(
        n_clusters=k,
        metric="cosine",
        linkage="average",
    )
    labels = clustering.fit_predict(matrix)

    for r, label in zip(candidates, labels):
        r.cluster_id = int(label)

    return k


# ---------------------------------------------------------------------------
# Caption-aware tag boost
# ---------------------------------------------------------------------------

def apply_tag_boost(
    results: list[ImageResult],
    caption_tags: dict[str, list[str]],
    boost_scale: float = 0.10,
) -> None:
    """Boost selection weight for images with underrepresented tags.

    Computes tag frequency across all images that have captions.
    Images whose tags include low-frequency tags get a positive boost
    proportional to how rare those tags are.

    Mutates result.tag_boost in-place.
    """
    if not caption_tags:
        return

    # Count tag frequency
    tag_freq: dict[str, int] = {}
    for tags in caption_tags.values():
        for tag in tags:
            tag_freq[tag] = tag_freq.get(tag, 0) + 1

    if not tag_freq:
        return

    max_freq = max(tag_freq.values())

    for result in results:
        tags = caption_tags.get(result.name, [])
        if not tags:
            continue
        # Rarity score: average (1 - freq/max_freq) across the image's tags
        rarity = sum(1.0 - tag_freq.get(t, max_freq) / max_freq for t in tags) / len(tags)
        result.tag_boost = round(float(rarity) * boost_scale, 4)
        result.score = round(min(1.0, result.score + result.tag_boost), 4)
