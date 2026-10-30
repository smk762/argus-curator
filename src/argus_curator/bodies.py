"""Body crops for MiVOLO: detect people, give each face its body, mask the crop.

MiVOLO reads a face crop and a person crop side by side, and its published
accuracy is for the pair. Without a person detector curator ran every face
face-only (see :mod:`mivolo`). InsightFace 2.1's Cheetah packs ship a person
detector; only that model is loaded here (``pp_det_med.onnx`` in ``cheetah_l``,
30 MB), not the pack's re-identification or its second copy of the face models.

The rules follow MiVOLO's own ``structures.py`` / ``data/misc.py``:

* A face is given the body it best matches by one-to-one assignment, scoring
  each pair as the mean of their IoU and the share of the face inside the body.
  A pair under ``FaceConfig.body_face_match`` is no match.
* A body crop is the person box with **every** face on it blacked out, its own
  included (the face channel carries the face), and every other person on it
  too. A masked person box whose edge falls within 30% of the crop's edge is
  stretched to that edge.
* A crop under ``body_min_size`` pixels on a side, or with less than
  ``body_min_visible`` of it left after masking, is not used: that face runs
  face-only, as before.

The pretrained Cheetah models carry InsightFace's non-commercial licence
(``MODEL.LICENSE``), the same terms as ``buffalo_l``.
"""

from __future__ import annotations

import threading

import numpy as np
from argus_cortex.moderation import DEFAULT_EXPOSED_LABELS
from PIL import Image

from argus_curator.models import FaceConfig, ImageResult, PersonDetection
from argus_curator.runtime import ModelUnavailable, providers

# MiVOLO's CROP_ROUND_RATE: a masked person edge this close to the crop's edge snaps to it.
CROP_ROUND_RATE = 0.3

_DETECTOR_CACHE: dict[str, object] = {}
_DETECTOR_LOCK = threading.Lock()


def bodies_available() -> bool:
    """Whether InsightFace has the Cheetah person-package loader (2.1+)."""
    try:
        from insightface.model_zoo.person_detection import PersonDetection  # noqa: F401
        from insightface.model_zoo.person_package import ensure_person_package  # noqa: F401
    except Exception:
        return False
    return True


def get_detector(cfg: FaceConfig) -> object:
    """The person detector of ``cfg.body_model``, built once per model and device."""
    key = f"{cfg.body_model}:{cfg.device}"
    if key in _DETECTOR_CACHE:
        return _DETECTOR_CACHE[key]
    with _DETECTOR_LOCK:
        if key not in _DETECTOR_CACHE:
            _DETECTOR_CACHE[key] = _build_detector(cfg)
        return _DETECTOR_CACHE[key]


def _build_detector(cfg: FaceConfig) -> object:
    try:
        from insightface.model_zoo.person_detection import PersonDetection
        from insightface.model_zoo.person_package import ensure_person_package
    except Exception as exc:
        raise ModelUnavailable(
            "Body crops need insightface 2.1+: pip install argus-curator[faces]  (or [gpu])"
        ) from exc
    # Fetches the pack on first use into ~/.insightface, beside buffalo_l.
    package = ensure_person_package(cfg.body_model)
    return PersonDetection(package["tasks"]["person_detection"], providers(cfg.device))


def detect(detector: object, bgr: np.ndarray, min_score: float) -> tuple[np.ndarray, np.ndarray]:
    """Person boxes ``[x1, y1, x2, y2]`` and their detector scores, at least *min_score*."""
    rows = np.asarray(detector.detect_candidates(bgr, min_score), dtype=np.float32)  # type: ignore[attr-defined]
    if not rows.size:
        return np.empty((0, 4), np.float32), np.empty((0,), np.float32)
    boxes = rows[:, :4]
    scores = rows[:, 4] if rows.shape[1] > 4 else np.ones(len(rows), np.float32)
    return boxes, scores


def xywh(box_xyxy: np.ndarray) -> list[float]:
    """``[x, y, w, h]``, the same box convention faces use."""
    x1, y1, x2, y2 = (float(v) for v in box_xyxy)
    return [round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1)]


def _xyxy(bbox_xywh: list[float]) -> np.ndarray:
    x, y, w, h = bbox_xywh
    return np.array([x, y, x + w, y + h], dtype=np.float64)


def _area(box: np.ndarray) -> float:
    return float(max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]))


def _intersection(a: np.ndarray, b: np.ndarray) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return float(w * h) if w > 0 and h > 0 else 0.0


def match_score(body: np.ndarray, face: np.ndarray) -> float:
    """MiVOLO's ``box_iou(..., over_second=True)``: mean of IoU and the share of *face* inside *body*."""
    inter = _intersection(body, face)
    if inter == 0.0:
        return 0.0
    union = _area(body) + _area(face) - inter
    return (inter / union + inter / _area(face)) / 2


def assign(bodies: np.ndarray, faces_xywh: list[list[float]], min_score: float) -> list[int | None]:
    """Per face, the index of its body, or ``None``. One body per face, one face per body."""
    from scipy.optimize import linear_sum_assignment

    out: list[int | None] = [None] * len(faces_xywh)
    if not len(bodies) or not faces_xywh:
        return out
    faces = [_xyxy(f) for f in faces_xywh]
    scores = np.array([[match_score(np.asarray(b, np.float64), f) for f in faces] for b in bodies])
    for b, f in zip(*linear_sum_assignment(scores, maximize=True), strict=True):
        if scores[b, f] > min_score:
            out[f] = int(b)
    return out


def body_crop(
    image: Image.Image,
    bodies: np.ndarray,
    index: int,
    faces_xywh: list[list[float]],
    min_size: int,
    min_visible: float,
) -> Image.Image | None:
    """Body *index* cut out of *image* with every face and every other body on it blacked out.

    ``None`` when the crop is too small, or too little of it is left after masking.
    """
    body = np.asarray(bodies[index], np.float64)
    x1, y1 = max(0, int(body[0])), max(0, int(body[1]))
    x2, y2 = min(image.width, int(body[2])), min(image.height, int(body[3]))
    w, h = x2 - x1, y2 - y1
    if w < min_size or h < min_size:
        return None
    crop = np.array(image.crop((x1, y1, x2, y2)).convert("RGB"))
    others = [(_xyxy(f), False) for f in faces_xywh]
    others += [(np.asarray(b, np.float64), True) for i, b in enumerate(bodies) if i != index]
    for box, is_person in others:
        if _intersection(body, box) == 0.0:
            continue
        ox1, oy1 = max(0, int(box[0]) - x1), max(0, int(box[1]) - y1)
        ox2, oy2 = min(w, int(box[2]) - x1), min(h, int(box[3]) - y1)
        if is_person:
            if ox1 < w * CROP_ROUND_RATE:
                ox1 = 0
            if w - ox2 < w * CROP_ROUND_RATE:
                ox2 = w
            if oy1 < h * CROP_ROUND_RATE:
                oy1 = 0
            if h - oy2 < h * CROP_ROUND_RATE:
                oy2 = h
        crop[oy1:oy2, ox1:ox2] = 0
    if np.count_nonzero(crop) / crop.size < min_visible:
        return None
    return Image.fromarray(crop)


def crops_for_faces(
    image: Image.Image, bodies: np.ndarray, faces_xywh: list[list[float]], cfg: FaceConfig
) -> list[Image.Image | None]:
    """Per face, its masked body crop, or ``None`` when it has no usable body."""
    matched = assign(bodies, faces_xywh, cfg.body_face_match)
    return [
        None if b is None else body_crop(image, bodies, b, faces_xywh, cfg.body_min_size, cfg.body_min_visible)
        for b in matched
    ]


def to_people(boxes_xyxy: np.ndarray, scores: np.ndarray, matched: list[int | None]) -> list[PersonDetection]:
    """Person boxes in face-box convention, with ``face_index`` filled in."""
    face_of_body = {body: face for face, body in enumerate(matched) if body is not None}
    return [
        PersonDetection(bbox=xywh(box), det_score=round(float(score), 4), face_index=face_of_body.get(i))
        for i, (box, score) in enumerate(zip(boxes_xyxy, scores, strict=True))
    ]


def _iou_xywh(a: list[float], b: list[float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter == 0.0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def associate_nsfw(results: list[ImageResult]) -> None:
    """Give each NudeNet detection to the person (else face) its box overlaps most.

    A detection that overlaps nothing is left unassigned; :func:`selection.minor_exposed`
    treats those as image-level (fail closed). Called after faces and NudeNet both
    finish, before the guard pass.
    """
    for result in results:
        if result.nsfw is None:
            continue
        for person in result.people:
            person.nsfw_labels = {}
            person.exposed_score = 0.0
        for face in result.faces:
            face.nsfw_labels = {}
            face.exposed_score = 0.0
        targets: list[tuple[list[float], PersonDetection | object]] = [(p.bbox, p) for p in result.people] or [
            (f.bbox, f) for f in result.faces
        ]
        if not targets:
            continue
        for det in result.nsfw.detections:
            best, best_iou = None, 0.0
            for bbox, obj in targets:
                iou = _iou_xywh(det.bbox, bbox)
                if iou > best_iou:
                    best, best_iou = obj, iou
            if best is None or best_iou <= 0.0:
                continue
            labels = best.nsfw_labels
            labels[det.label] = max(float(labels.get(det.label, 0.0)), det.score)
            if det.label in DEFAULT_EXPOSED_LABELS:
                best.exposed_score = max(best.exposed_score, det.score)
