"""Selection — score threshold + optional diversity cap.

Ported from imogen PR #14's ``_select_diverse`` / ``_decide_keep``. Nothing is
ever silently dropped: every excluded image carries a ``keep_reason`` so the
HITL surfaces can explain *why* it was left out.
"""

from __future__ import annotations

import imagehash

from argus_curator.models import ExportRequest, ImageResult

_MAX_DIST = 64.0  # maximum 64-bit pHash Hamming distance


def select_diverse(candidates: list[ImageResult], n: int, diversity_weight: float) -> list[ImageResult]:
    """Greedy diversity-aware selection.

    Each iteration picks the candidate maximising
    ``score_norm * (1 - dw) + diversity_norm * dw`` where ``diversity_norm`` is
    the minimum pHash Hamming distance to any already-selected image. ``dw=0``
    reduces to pure score ranking; ``dw=1`` maximises visual spread.
    """
    if n >= len(candidates) or diversity_weight <= 0.0:
        return sorted(candidates, key=lambda r: -r.score)[:n]

    max_score = max((r.score for r in candidates), default=1.0) or 1.0
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
                div_norm = min(hashes[i] - sh for sh in selected_hashes) / _MAX_DIST
            val = score_norm * (1.0 - diversity_weight) + div_norm * diversity_weight
            if val > best_val:
                best_val, best_idx = val, i
        if best_idx is None:
            break
        selected_idx.append(best_idx)
        if hashes[best_idx] is not None:
            selected_hashes.append(hashes[best_idx])
        remaining.remove(best_idx)

    return [candidates[i] for i in selected_idx]


def decide_selection(
    results: list[ImageResult],
    req: ExportRequest,
    diversity_weight: float,
) -> tuple[list[ImageResult], dict[str, str]]:
    """Return ``(selected_results, keep_reason_by_rel)`` for an export request.

    Mirrors PR #14's ``_decide_keep``: applies score threshold, rejected/duplicate
    gates, optional face-cluster filter, then an optional diversity cap.
    """
    keep_reason: dict[str, str] = {}
    face_filter = set(req.face_clusters) if req.face_clusters else None
    pose_filter = set(req.face_poses) if req.face_poses else None

    eligible: list[ImageResult] = []
    for r in results:
        if r.score < req.min_score:
            keep_reason[r.rel_path] = "below-score"
        elif not r.passed and not req.include_rejected:
            keep_reason[r.rel_path] = r.reject_reason or "rejected"
        elif r.is_duplicate and not req.keep_similar:
            keep_reason[r.rel_path] = f"similar-to:{r.duplicate_of}"
        elif face_filter is not None and r.primary_face_cluster not in face_filter:
            keep_reason[r.rel_path] = "face-cluster-excluded"
        elif pose_filter is not None and r.primary_face_pose not in pose_filter:
            keep_reason[r.rel_path] = "face-pose-excluded"
        else:
            eligible.append(r)

    chosen = eligible
    if req.max_keep is not None and len(eligible) > req.max_keep:
        chosen = select_diverse(eligible, req.max_keep, diversity_weight)
        chosen_set = {r.rel_path for r in chosen}
        for r in eligible:
            if r.rel_path not in chosen_set:
                keep_reason[r.rel_path] = "diversity-trimmed"

    for r in chosen:
        keep_reason[r.rel_path] = ""
    return chosen, keep_reason
