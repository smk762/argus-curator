"""Subset selection strategies.

Two strategies, chosen automatically based on whether embeddings are available:

  cluster-based (preferred when CLIP embeddings present)
    Pick top-N images per cluster proportional to cluster size.
    Within each cluster, rank by score.
    This gives guaranteed semantic coverage across all clusters.

  greedy pHash diversity (fallback)
    Each iteration picks the candidate that maximises:
        score_norm * (1 - dw) + phash_diversity_norm * dw
    Classic tradeoff between quality and visual spread.
"""

from __future__ import annotations

import imagehash

from argus_curator.types import ImageResult


# ---------------------------------------------------------------------------
# Cluster-based selection
# ---------------------------------------------------------------------------

def select_by_clusters(
    candidates: list[ImageResult],
    target_n: int,
    diversity_weight: float,
) -> list[ImageResult]:
    """Select *target_n* images with guaranteed per-cluster coverage.

    Images must have cluster_id set (by clustering.cluster_by_embeddings).
    Within each cluster, images are ranked by score.  The selection quota
    per cluster is proportional to cluster size but biased toward quality
    via diversity_weight (higher dw → more even spread across clusters).
    """
    from collections import defaultdict
    import math

    clusters: dict[int, list[ImageResult]] = defaultdict(list)
    no_cluster: list[ImageResult] = []
    for r in candidates:
        if r.cluster_id is None:
            no_cluster.append(r)
        else:
            clusters[r.cluster_id].append(r)

    # Sort within each cluster by score (descending)
    for c in clusters.values():
        c.sort(key=lambda r: -r.score)

    n_clusters = len(clusters)
    if n_clusters == 0:
        return _greedy_phash(no_cluster, target_n, diversity_weight)

    # Quota per cluster: blend between proportional and equal
    cluster_sizes = {cid: len(imgs) for cid, imgs in clusters.items()}
    total = sum(cluster_sizes.values())

    quotas: dict[int, int] = {}
    for cid, size in cluster_sizes.items():
        prop_quota = target_n * (size / total)
        equal_quota = target_n / n_clusters
        # diversity_weight blends toward equal distribution
        quota = prop_quota * (1.0 - diversity_weight) + equal_quota * diversity_weight
        quotas[cid] = max(1, round(quota))

    # Normalise quotas so they sum to target_n
    total_quota = sum(quotas.values())
    if total_quota != target_n:
        # Adjust the largest cluster to absorb rounding difference
        largest = max(quotas, key=lambda k: cluster_sizes[k])
        quotas[largest] += target_n - total_quota

    selected: list[ImageResult] = []
    for cid, imgs in clusters.items():
        q = min(quotas.get(cid, 1), len(imgs))
        selected.extend(imgs[:q])

    # If we're short (rounding, empty clusters), backfill with best remaining
    if len(selected) < target_n:
        selected_names = {r.name for r in selected}
        remaining = sorted(
            [r for r in candidates if r.name not in selected_names],
            key=lambda r: -r.score,
        )
        selected.extend(remaining[: target_n - len(selected)])

    return selected[:target_n]


# ---------------------------------------------------------------------------
# Greedy pHash diversity fallback
# ---------------------------------------------------------------------------

def _greedy_phash(
    candidates: list[ImageResult],
    n: int,
    diversity_weight: float,
) -> list[ImageResult]:
    """Greedy diversity-aware selection using pHash Hamming distance."""
    if not candidates:
        return []
    if n >= len(candidates) or diversity_weight <= 0.0:
        return sorted(candidates, key=lambda r: -r.score)[:n]

    _MAX_DIST = 64.0
    max_score = max(r.score for r in candidates) or 1.0

    hashes: list[imagehash.ImageHash | None] = []
    for r in candidates:
        try:
            hashes.append(imagehash.hex_to_hash(r.phash) if r.phash else None)
        except Exception:
            hashes.append(None)

    selected_idx: list[int] = []
    selected_hashes: list[imagehash.ImageHash] = []
    remaining = list(range(len(candidates)))

    for _ in range(n):
        if not remaining:
            break
        best_idx, best_val = None, -1.0
        for i in remaining:
            score_norm = candidates[i].score / max_score
            if not selected_hashes or hashes[i] is None:
                div_norm = 1.0
            else:
                min_dist = min(hashes[i] - sh for sh in selected_hashes if sh is not None)
                div_norm = min_dist / _MAX_DIST
            val = score_norm * (1.0 - diversity_weight) + div_norm * diversity_weight
            if val > best_val:
                best_val = val
                best_idx = i
        if best_idx is None:
            break
        selected_idx.append(best_idx)
        if hashes[best_idx] is not None:
            selected_hashes.append(hashes[best_idx])
        remaining.remove(best_idx)

    return [candidates[i] for i in selected_idx]


# ---------------------------------------------------------------------------
# Public selector
# ---------------------------------------------------------------------------

def select(
    results: list[ImageResult],
    target_n: int,
    diversity_weight: float,
    use_embedding_clusters: bool,
) -> list[ImageResult]:
    """Select *target_n* images from the candidate set.

    Candidates are non-duplicate passing images.  cluster_id must be set
    before calling this function when use_embedding_clusters=True.
    """
    candidates = [r for r in results if r.passed and not r.is_duplicate]
    target_n = min(target_n, len(candidates))
    if target_n <= 0:
        return []

    has_clusters = use_embedding_clusters and any(r.cluster_id is not None for r in candidates)
    if has_clusters:
        return select_by_clusters(candidates, target_n, diversity_weight)
    return _greedy_phash(candidates, target_n, diversity_weight)
