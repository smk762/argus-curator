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
import threading
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import structlog

from argus_curator.attributes import RawAttributes, pool, to_attributes
from argus_curator.models import FaceCluster, FaceConfig, FaceDetection, ImageResult, PersonDetection, SetAge
from argus_curator.runtime import gpu_lease, providers

logger = structlog.get_logger()

_APP_CACHE: dict[str, object] = {}
# The InsightFace models curator reads: boxes and keypoints, identity embeddings,
# and the 3D-68 landmarks behind head pose. A pack's gender/age model (MiVOLO
# replaces it) and its 2D-106 landmarks are never loaded.
FACE_MODULES = ("detection", "recognition", "landmark_3d_68")
# A scan thread and POST /faces can ask for the same app at once; build it once.
_APP_LOCK = threading.Lock()
# FaceAnalysis.get is not thread-safe (bbox/result mix-ups under concurrent
# callers). A scan's faces phase and POST /faces share the cached app, so every
# get() is serialized. onnxruntime session.run (MiVOLO, FairFace) is fine.
_INFER_LOCK = threading.Lock()


class FacesUnavailable(RuntimeError):
    """Raised when face detection is requested but optional deps are missing."""


def faces_available() -> bool:
    """True when the InsightFace stack is importable."""
    try:
        import insightface  # noqa: F401

        return True
    except Exception:
        return False


def _get_app(cfg: FaceConfig, *, lease_wait_s: float | None = None):
    """Build (and cache) an InsightFace ``FaceAnalysis`` app for the model."""
    key = f"{cfg.model}:{cfg.device}"
    if key in _APP_CACHE:
        return _APP_CACHE[key]
    with _APP_LOCK:
        if key not in _APP_CACHE:
            _APP_CACHE[key] = _build_app(cfg, lease_wait_s=lease_wait_s)
        return _APP_CACHE[key]


def _build_app(cfg: FaceConfig, *, lease_wait_s: float | None = None):
    try:
        from insightface.app import FaceAnalysis
    except Exception as exc:  # pragma: no cover - exercised only without deps
        raise FacesUnavailable("Face detection requires: pip install argus-curator[faces]  (or [gpu])") from exc

    provs = providers(cfg.device)
    # FaceAnalysis() fetches the pack to disk; prepare() builds CUDA sessions.
    # Keep the download off the GPU lease so a cold start does not block lens.
    app = FaceAnalysis(name=cfg.model, providers=provs, allowed_modules=list(FACE_MODULES))
    ctx_id = 0 if "CUDAExecutionProvider" in provs else -1
    with gpu_lease(cfg.device, "faces", "load", wait_s=lease_wait_s):
        app.prepare(ctx_id=ctx_id, det_size=(640, 640))
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


def _body_detector(cfg: FaceConfig) -> object | None:
    """The person detector when ``cfg.mivolo_body`` asks for body crops, else ``None``."""
    if not cfg.mivolo_body:
        return None
    from argus_curator import bodies

    return bodies.get_detector(cfg)


def _face_attributes(
    sessions: tuple[object | None, object | None],
    bgr: np.ndarray,
    kps_by_face: list[np.ndarray | None],
    bboxes: list[list[float]],
    rel: str,
    *,
    body_crops: list[object] | None = None,
) -> list[RawAttributes]:
    """Raw model outputs per face of one image; a model that fails is skipped.

    *body_crops*, one per face, are the masked bodies from :mod:`bodies`; a
    ``None`` crop (or *body_crops* itself ``None``) runs that face face-only.
    """
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
            preds = mivolo.predict(mv_session, image, bboxes, body_crops)
            for i, (raw, pred) in enumerate(zip(raws, preds, strict=True)):
                if pred is not None:
                    raw.mivolo_gender, raw.mivolo_age = pred
                    raw.mivolo_body = body_crops is not None and body_crops[i] is not None
        except Exception as exc:
            logger.warning("mivolo_failed", rel_path=rel, error=str(exc))
    return raws


def _record_people(
    body_detector: object,
    bgr: np.ndarray,
    image: object,
    face_bboxes: list[list[float]],
    cfg: FaceConfig,
    rel: str,
) -> tuple[list[PersonDetection], list[object] | None, object]:
    """Detect people, match them to *face_bboxes*, return people, per-face crops, and xyxy boxes."""
    from argus_curator import bodies

    try:
        boxes, scores = bodies.detect(body_detector, bgr, cfg.body_min_score)
    except Exception as exc:
        logger.warning("body_detect_failed", rel_path=rel, error=str(exc))
        return [], None, None
    matched = bodies.assign(boxes, face_bboxes, cfg.body_face_match)
    crops = [
        (
            None
            if body is None
            else bodies.body_crop(image, boxes, body, face_bboxes, cfg.body_min_size, cfg.body_min_visible)
        )
        for body in matched
    ]
    return bodies.to_people(boxes, scores, matched), crops, boxes


def _link_people(
    r: ImageResult,
    mv_session: object | None,
    image: object,
    boxes_xyxy: object,
    cfg: FaceConfig,
    rel: str,
) -> None:
    """Copy face ages onto matched people; MiVOLO body-only ages onto the rest."""
    from argus_curator import bodies

    for i, person in enumerate(r.people):
        if person.face_index is None:
            continue
        face = r.faces[person.face_index]
        face.person_index = i
        attrs = face.attributes
        if attrs is not None:
            person.age_years = attrs.age_years
            person.age_source = attrs.age_source
            person.age_body = attrs.age_body
    if mv_session is None or boxes_xyxy is None:
        return
    from argus_curator import mivolo

    unmatched: list[tuple[int, object]] = []
    for i, person in enumerate(r.people):
        if person.face_index is not None:
            continue
        crop = bodies.body_crop(
            image, boxes_xyxy, i, [f.bbox for f in r.faces], cfg.body_min_size, cfg.body_min_visible
        )
        if crop is not None:
            unmatched.append((i, crop))
    if not unmatched:
        return
    try:
        preds = mivolo.predict_bodies(mv_session, [crop for _, crop in unmatched])
    except Exception as exc:
        logger.warning("mivolo_body_only_failed", rel_path=rel, error=str(exc))
        return
    for (i, _crop), pred in zip(unmatched, preds, strict=True):
        _gender, years = pred
        r.people[i].age_years = round(years, 1)
        r.people[i].age_source = "mivolo"
        r.people[i].age_body = True


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
    strict: bool = False,
    lease_wait_s: float | None = None,
) -> DetectedFaces:
    """Detect faces and run MiVOLO and FairFace. Identity clustering is separate.

    ``on_done(image, done, total)`` fires once each passing image has its faces,
    so a caller can show ages and race before the rest of the scan. ``total`` is
    the number of passing images.

    A scan logs and skips an image the detector cannot read. With *strict* that
    raises instead, so a single-image caller (``POST /faces``) cannot mistake a
    failed detection for "no faces".

    On CUDA each image runs under the host's GPU lease (:func:`runtime.gpu_lease`),
    taken per image so another argus process can use the card between them.
    *lease_wait_s* caps each wait. A lease that times out raises
    :class:`~argus_cortex.gpu.GpuLeaseTimeout`, failing the call rather than
    reading as an image with no faces.
    """
    empty = DetectedFaces([], [], [])
    if not cfg.enabled:
        return empty

    app = _get_app(cfg, lease_wait_s=lease_wait_s)
    with gpu_lease(cfg.device, "faces", "load", wait_s=lease_wait_s):
        sessions = _attribute_sessions(cfg)
        body_detector = _body_detector(cfg)
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
        if bgr is None and strict:
            raise ValueError(f"cannot decode {rel} for face detection")
        if bgr is not None:
            with gpu_lease(cfg.device, "faces", wait_s=lease_wait_s):
                _detect_one(app, sessions, bgr, rel, r, cfg, found, strict, body_detector)
        if on_done is not None:
            on_done(r, i + 1, total)

    if progress is not None:
        progress(total, total)
    return found


def _detect_one(
    app: object,
    sessions: tuple[object | None, object | None],
    bgr: np.ndarray,
    rel: str,
    r: ImageResult,
    cfg: FaceConfig,
    found: DetectedFaces,
    strict: bool,
    body_detector: object | None = None,
) -> None:
    """Faces, embeddings and attributes for one decoded image, recorded on *r* and *found*."""
    from PIL import Image

    try:
        with _INFER_LOCK:
            detections = app.get(bgr)  # type: ignore[attr-defined]
    except Exception as exc:
        if strict:
            raise
        logger.warning("face_detect_failed", rel_path=rel, error=str(exc))
        detections = []
    if detections is None:
        detections = []
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
    image = Image.fromarray(bgr[:, :, ::-1])
    body_crops = None
    boxes_xyxy = None
    if body_detector is not None:
        r.people, body_crops, boxes_xyxy = _record_people(
            body_detector, bgr, image, [f.bbox for f in r.faces], cfg, rel
        )
    raws = _face_attributes(sessions, bgr, kps_by_face, [f.bbox for f in r.faces], rel, body_crops=body_crops)
    for f, raw in zip(r.faces, raws, strict=True):
        f.attributes = to_attributes(raw)
    found.raws.extend(raws)
    _link_people(r, sessions[1], image, boxes_xyxy, cfg, rel)
    _assign_primary(r)


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


def infer_set_age(
    results: list[ImageResult], clusters: list[FaceCluster], cfg: FaceConfig
) -> tuple[SetAge | None, str | None]:
    """Give the passing frames with no face an age read off the set, when the scan looks like one subject.

    The dominant identity must hold at least ``cfg.set_age_min_share`` of the
    scan's faces and have at least ``cfg.set_age_min_faces`` MiVOLO ages; its
    median age becomes each faceless frame's :class:`SetAge`. Returns the age
    and ``None``, or ``None`` and why nothing was given. Frames the hard
    filters rejected are left alone: face detection never looked at them.
    """
    if not clusters:
        return None, "no faces were found in this scan"
    total = sum(cluster.size for cluster in clusters)
    top = max(clusters, key=lambda cluster: cluster.size)
    share = top.size / total if total else 0.0
    if share < cfg.set_age_min_share:
        return (
            None,
            f"the largest identity ({top.cluster_id}) has {share:.0%} of the faces, {cfg.set_age_min_share:.0%} needed",
        )
    ages = [
        face.attributes.age_years
        for r in results
        for face in r.faces
        if face.cluster_id == top.cluster_id
        and face.attributes is not None
        and face.attributes.age_years is not None
        and face.attributes.age_source in (None, "mivolo")
    ]
    if len(ages) < cfg.set_age_min_faces:
        return None, f"{top.cluster_id} has {len(ages)} MiVOLO ages, {cfg.set_age_min_faces} needed"
    set_age = SetAge(
        years=round(float(np.median(ages)), 1), cluster_id=top.cluster_id, faces=len(ages), share=round(share, 3)
    )
    for r in results:
        if r.passed and not r.faces and not any(p.age_years is not None for p in r.people):
            r.set_age = set_age
    return set_age, None


def detect_and_cluster(
    results: list[ImageResult],
    items: list[tuple[str, str, bytes]],
    cfg: FaceConfig,
    progress: Callable[[int, int], None] | None = None,
) -> list[FaceCluster]:
    """Detect, attribute, and cluster. Ages are set before clustering starts."""
    detected = detect_faces(results, items, cfg, progress=progress)
    return cluster_identities(results, detected, cfg)
