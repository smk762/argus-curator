"""InsightFace detect + embed + cluster — the identity-aware differentiator (M2).

Uses InsightFace ``buffalo_l`` (RetinaFace detection + ArcFace embeddings) — the
same model family Immich uses, CPU-capable and GPU-accelerated via
``onnxruntime-gpu``. Every detected face is embedded; embeddings are clustered
across the whole dataset into stable ``face_<n>`` identities so the UI can
"show only identity X", "single-face only", or "drop unknown faces".

All heavy imports are deferred and guarded: ``import argus_curator.faces`` is
always safe; the InsightFace stack is only required when face detection is
actually enabled (``pip install argus-curator[faces]`` or ``[gpu]``).
"""

from __future__ import annotations

import io
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import structlog

from argus_curator.attributes import RawAttributes, pool, to_attributes
from argus_curator.models import FaceCluster, FaceConfig, FaceDetection, ImageResult
from argus_curator.runtime import providers

logger = structlog.get_logger()

_APP_CACHE: dict[str, object] = {}


class FacesUnavailable(RuntimeError):
    """Raised when face detection is requested but optional deps are missing."""


def faces_available() -> bool:
    """True when the InsightFace stack is importable."""
    try:
        import insightface  # noqa: F401

        return True
    except Exception:
        return False


def _get_app(cfg: FaceConfig):
    """Build (and cache) an InsightFace ``FaceAnalysis`` app for the model."""
    key = f"{cfg.model}:{cfg.device}"
    if key in _APP_CACHE:
        return _APP_CACHE[key]
    try:
        from insightface.app import FaceAnalysis
    except Exception as exc:  # pragma: no cover - exercised only without deps
        raise FacesUnavailable("Face detection requires: pip install argus-curator[faces]  (or [gpu])") from exc

    app = FaceAnalysis(name=cfg.model, providers=providers(cfg.device))
    ctx_id = 0 if "CUDAExecutionProvider" in providers(cfg.device) else -1
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    _APP_CACHE[key] = app
    return app


def _to_bgr(data: bytes) -> np.ndarray | None:
    try:
        from PIL import Image

        img = Image.open(io.BytesIO(data)).convert("RGB")
        arr = np.array(img)
        return arr[:, :, ::-1].copy()  # RGB -> BGR for InsightFace/cv2
    except Exception:
        return None


def classify_pose(yaw: float | None, cfg: FaceConfig) -> str | None:
    """Bucket a head yaw (degrees) into frontal / three_quarter / profile.

    Uses absolute yaw so left and right map to the same bucket; the signed
    ``yaw`` is stored alongside if you need to distinguish direction.
    """
    if yaw is None:
        return None
    ay = abs(yaw)
    if ay <= cfg.frontal_max_yaw:
        return "frontal"
    if ay <= cfg.profile_min_yaw:
        return "three_quarter"
    return "profile"


def _extract_pose(face: object) -> tuple[float | None, float | None]:
    """Return ``(yaw, pitch)`` in degrees from an InsightFace face, if present.

    ``buffalo_l`` ships the 3D-68 landmark model, which populates
    ``face.pose = [pitch, yaw, roll]``. Older packs may omit it.
    """
    pose_arr = getattr(face, "pose", None)
    if pose_arr is None:
        return None, None
    try:
        pitch = float(pose_arr[0])
        yaw = float(pose_arr[1])
    except (TypeError, ValueError, IndexError):
        return None, None
    return yaw, pitch


def _centrality(bbox_xywh: list[float], w: int, h: int) -> float:
    """How central a face is (1 = dead centre) — used to pick the primary face."""
    if w <= 0 or h <= 0:
        return 0.0
    cx = bbox_xywh[0] + bbox_xywh[2] / 2.0
    cy = bbox_xywh[1] + bbox_xywh[3] / 2.0
    dx = abs(cx - w / 2.0) / (w / 2.0)
    dy = abs(cy - h / 2.0) / (h / 2.0)
    return max(0.0, 1.0 - (dx + dy) / 2.0)


def _cluster_embeddings(embeddings: np.ndarray, eps: float) -> list[int]:
    """Cluster L2-normalised embeddings by cosine distance.

    Prefers scikit-learn's agglomerative clustering; falls back to a
    deterministic greedy assignment when sklearn is unavailable.
    """
    n = len(embeddings)
    if n == 0:
        return []
    if n == 1:
        return [0]

    try:
        from sklearn.cluster import AgglomerativeClustering

        model = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=eps,
        )
        return model.fit_predict(embeddings).tolist()
    except Exception:
        # Greedy cosine clustering: assign each face to the first centroid within eps.
        labels = [-1] * n
        centroids: list[np.ndarray] = []
        for i in range(n):
            v = embeddings[i]
            best_lbl, best_dist = -1, eps
            for lbl, c in enumerate(centroids):
                dist = 1.0 - float(np.dot(v, c))
                if dist < best_dist:
                    best_dist, best_lbl = dist, lbl
            if best_lbl == -1:
                centroids.append(v.copy())
                labels[i] = len(centroids) - 1
            else:
                labels[i] = best_lbl
        return labels


def _attribute_sessions(cfg: FaceConfig) -> tuple[object | None, object | None]:
    """``(fairface, mivolo)`` onnxruntime sessions for the enabled models."""
    provs = providers(cfg.device)
    ff = mv = None
    if cfg.fairface:
        from argus_curator import fairface

        ff = fairface.get_session(provs)
    if cfg.mivolo:
        from argus_curator import mivolo

        mv = mivolo.get_session(provs)
    return ff, mv


def _face_attributes(
    sessions: tuple[object | None, object | None],
    bgr: np.ndarray,
    kps_by_face: list[np.ndarray | None],
    bboxes: list[list[float]],
    rel: str,
) -> list[RawAttributes]:
    """Raw model outputs per face of one image; a model that fails is skipped."""
    raws = [RawAttributes() for _ in bboxes]
    ff_session, mv_session = sessions
    if (ff_session is None and mv_session is None) or not bboxes:
        return raws
    from PIL import Image

    image = Image.fromarray(bgr[:, :, ::-1])
    if ff_session is not None:
        from argus_curator import fairface

        # FairFace aligns on the 5 keypoints; a face without them is skipped.
        idx = [i for i, kps in enumerate(kps_by_face) if kps is not None]
        try:
            probs = fairface.predict(ff_session, image, [kps_by_face[i] for i in idx])
            for i, row in zip(idx, probs, strict=True):
                raws[i].fairface = row
        except Exception as exc:
            logger.warning("fairface_failed", rel_path=rel, error=str(exc))
    if mv_session is not None:
        from argus_curator import mivolo

        try:
            for raw, pred in zip(raws, mivolo.predict(mv_session, image, bboxes), strict=True):
                if pred is not None:
                    raw.mivolo_gender, raw.mivolo_age = pred
        except Exception as exc:
            logger.warning("mivolo_failed", rel_path=rel, error=str(exc))
    return raws


@dataclass
class DetectedFaces:
    """Face embeddings held until identity clustering. Not written into the scan."""

    embeddings: list[np.ndarray]
    # (rel_path, face_index_within_image, bbox_xywh, det_score, centrality, area)
    meta: list[tuple[str, int, list[float], float, float, float]]
    raws: list[RawAttributes]


def _assign_primary(result: ImageResult) -> None:
    """Largest, most central face becomes the primary. A cluster id is copied when it exists."""
    if not result.faces:
        return
    primary_i = max(
        range(len(result.faces)),
        key=lambda i: (result.faces[i].bbox[2] * result.faces[i].bbox[3]) * (0.5 + 0.5 * result.faces[i].det_score),
    )
    for i, face in enumerate(result.faces):
        face.primary = i == primary_i
    primary = result.faces[primary_i]
    result.primary_face_cluster = primary.cluster_id
    result.primary_face_pose = primary.pose
    result.primary_face_yaw = primary.yaw
    result.primary_face_attributes = primary.attributes


def detect_faces(
    results: list[ImageResult],
    items: list[tuple[str, str, bytes]],
    cfg: FaceConfig,
    progress: Callable[[int, int], None] | None = None,
    on_done: Callable[[ImageResult, int, int], None] | None = None,
) -> DetectedFaces:
    """Detect faces and run MiVOLO and FairFace. Identity clustering is separate.

    ``on_done(image, done, total)`` fires once each passing image has its faces,
    so a caller can show ages and race before the rest of the scan. ``total`` is
    the number of passing images.
    """
    empty = DetectedFaces([], [], [])
    if not cfg.enabled:
        return empty

    app = _get_app(cfg)
    sessions = _attribute_sessions(cfg)
    data_by_rel = {rel: data for rel, _abs, data in items}
    result_by_rel = {r.rel_path: r for r in results}
    found = DetectedFaces([], [], [])

    pending = [(rel, r) for rel, r in result_by_rel.items() if r.passed and data_by_rel.get(rel) is not None]
    total = len(pending)
    step = max(1, total // 100)  # throttle to ~100 updates over the sweep
    if progress is not None:
        progress(0, total)

    for i, (rel, r) in enumerate(pending):
        if progress is not None and i % step == 0:
            progress(i, total)
        bgr = _to_bgr(data_by_rel[rel])
        if bgr is not None:
            try:
                detections = app.get(bgr)
            except Exception as exc:
                logger.warning("face_detect_failed", rel_path=rel, error=str(exc))
                detections = None
            if detections is not None:
                h, w = bgr.shape[:2]
                kept = 0
                kps_by_face: list[np.ndarray | None] = []
                for face in detections:
                    det_score = float(getattr(face, "det_score", 0.0))
                    if det_score < cfg.min_det_score:
                        continue
                    x1, y1, x2, y2 = (float(v) for v in face.bbox)
                    bbox = [round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1)]
                    emb = getattr(face, "normed_embedding", None)
                    if emb is None:
                        emb = getattr(face, "embedding", None)
                        if emb is None:
                            continue
                        emb = emb / (np.linalg.norm(emb) + 1e-9)
                    area = bbox[2] * bbox[3]
                    yaw, pitch = _extract_pose(face)
                    found.embeddings.append(np.asarray(emb, dtype=np.float32))
                    found.meta.append((rel, kept, bbox, det_score, _centrality(bbox, w, h), area))
                    r.faces.append(
                        FaceDetection(
                            bbox=bbox,
                            det_score=round(det_score, 4),
                            yaw=round(yaw, 2) if yaw is not None else None,
                            pitch=round(pitch, 2) if pitch is not None else None,
                            pose=classify_pose(yaw, cfg),
                        )
                    )
                    kps_by_face.append(getattr(face, "kps", None))
                    kept += 1
                r.face_count = kept
                raws = _face_attributes(sessions, bgr, kps_by_face, [f.bbox for f in r.faces], rel)
                for f, raw in zip(r.faces, raws, strict=True):
                    f.attributes = to_attributes(raw)
                found.raws.extend(raws)
                _assign_primary(r)
        if on_done is not None:
            on_done(r, i + 1, total)

    if progress is not None:
        progress(total, total)
    return found


def cluster_identities(
    results: list[ImageResult],
    detected: DetectedFaces,
    cfg: FaceConfig,
) -> list[FaceCluster]:
    """Group the embeddings from :func:`detect_faces` into ``face_<n>`` identities."""
    if not detected.embeddings:
        return []

    result_by_rel = {r.rel_path: r for r in results}
    labels = _cluster_embeddings(np.vstack(detected.embeddings), cfg.cluster_eps)

    # Stable, size-ordered cluster ids: the largest identity becomes face_1.
    raw_sizes: dict[int, int] = {}
    for lbl in labels:
        raw_sizes[lbl] = raw_sizes.get(lbl, 0) + 1
    order = sorted(raw_sizes, key=lambda lbl: (-raw_sizes[lbl], lbl))
    label_to_id = {lbl: f"face_{i + 1}" for i, lbl in enumerate(order)}

    # Track, per cluster, the best representative face (highest det_score x area).
    cluster_members: dict[str, list[tuple[str, list[float], float]]] = {}
    cluster_raw: dict[str, list[RawAttributes]] = {}

    for (rel, face_idx, bbox, det_score, _central, area), raw, lbl in zip(
        detected.meta, detected.raws, labels, strict=True
    ):
        cluster_id = label_to_id[lbl]
        result_by_rel[rel].faces[face_idx].cluster_id = cluster_id
        cluster_members.setdefault(cluster_id, []).append((rel, bbox, det_score * (area or 1.0)))
        cluster_raw.setdefault(cluster_id, []).append(raw)

    for r in results:
        _assign_primary(r)

    clusters: list[FaceCluster] = []
    for cluster_id, members in cluster_members.items():
        rep_rel, rep_bbox, _ = max(members, key=lambda m: m[2])
        clusters.append(
            FaceCluster(
                cluster_id=cluster_id,
                size=len(members),
                representative_rel_path=rep_rel,
                representative_bbox=rep_bbox,
                attributes=pool(cluster_raw[cluster_id]),
            )
        )
    clusters.sort(key=lambda c: -c.size)
    return clusters


def detect_and_cluster(
    results: list[ImageResult],
    items: list[tuple[str, str, bytes]],
    cfg: FaceConfig,
    progress: Callable[[int, int], None] | None = None,
) -> list[FaceCluster]:
    """Detect, attribute, and cluster. Ages are set before clustering starts."""
    detected = detect_faces(results, items, cfg, progress=progress)
    return cluster_identities(results, detected, cfg)
