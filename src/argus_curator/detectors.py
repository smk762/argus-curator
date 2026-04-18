"""YOLO person detection and MTCNN face detection.

Both are optional GPU-accelerated detectors.  They run on images that
already passed Phase-1 quality filters to avoid wasting GPU time on rejects.
"""

from __future__ import annotations

import structlog
from PIL import Image

logger = structlog.get_logger()


def _resolve_device(spec: str) -> str:
    if spec != "auto":
        return spec
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


# ---------------------------------------------------------------------------
# YOLO person detector
# ---------------------------------------------------------------------------

class _YOLODetector:
    _PERSON_CLASS = 0  # COCO class 0

    def __init__(self, model_name: str, device: str, confidence: float) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "YOLO detection requires 'ultralytics'. "
                "Install with: pip install 'argus-curator[gpu]'"
            ) from exc
        logger.info("yolo_load_start", model=model_name, device=device)
        self._model = YOLO(model_name)
        self._model.to(device)
        self._confidence = confidence
        logger.info("yolo_load_done", model=model_name)

    def detect_batch(self, images: list[Image.Image]) -> list[tuple[bool, float]]:
        """Returns (person_detected, best_confidence) per image."""
        results = self._model(images, verbose=False, conf=self._confidence)
        out = []
        for r in results:
            best = 0.0
            for box in r.boxes:
                if int(box.cls[0]) == self._PERSON_CLASS:
                    conf = float(box.conf[0])
                    if conf > best:
                        best = conf
            out.append((best > 0.0, best))
        return out


# ---------------------------------------------------------------------------
# MTCNN face detector
# ---------------------------------------------------------------------------

class _MTCNNDetector:
    def __init__(self, device: str, confidence: float) -> None:
        try:
            from facenet_pytorch import MTCNN
        except ImportError as exc:
            raise ImportError(
                "MTCNN requires 'facenet-pytorch'. "
                "Install with: pip install 'argus-curator[gpu]'"
            ) from exc
        import torch
        logger.info("mtcnn_load_start", device=device)
        self._mtcnn = MTCNN(keep_all=True, device=torch.device(device), post_process=False)
        self._min_prob = confidence
        logger.info("mtcnn_load_done")

    def detect_batch(self, images: list[Image.Image]) -> list[int]:
        """Returns face count per image."""
        _, probs_list = self._mtcnn.detect(images)
        counts = []
        for probs in probs_list:
            if probs is None:
                counts.append(0)
            else:
                counts.append(int(sum(1 for p in probs if p is not None and p >= self._min_prob)))
        return counts


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

from dataclasses import dataclass

@dataclass
class DetectorResult:
    face_count: int = 0
    person_detected: bool = False
    person_confidence: float = 0.0


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class DetectorPool:
    """Lazy-loading pool of YOLO + MTCNN.

    Only images that passed Phase-1 filters are processed here.
    """

    def __init__(
        self,
        *,
        use_yolo: bool,
        yolo_model: str,
        yolo_confidence: float,
        use_mtcnn: bool,
        mtcnn_confidence: float,
        device: str,
        batch_size: int,
    ) -> None:
        self._use_yolo = use_yolo
        self._yolo_args = (yolo_model, _resolve_device(device), yolo_confidence)
        self._use_mtcnn = use_mtcnn
        self._mtcnn_args = (_resolve_device(device), mtcnn_confidence)
        self._batch_size = batch_size
        self._yolo: _YOLODetector | None = None
        self._mtcnn: _MTCNNDetector | None = None

    def _get_yolo(self) -> _YOLODetector:
        if self._yolo is None:
            self._yolo = _YOLODetector(*self._yolo_args)
        return self._yolo

    def _get_mtcnn(self) -> _MTCNNDetector:
        if self._mtcnn is None:
            self._mtcnn = _MTCNNDetector(*self._mtcnn_args)
        return self._mtcnn

    def run(self, images: list[Image.Image]) -> list[DetectorResult]:
        n = len(images)
        results = [DetectorResult() for _ in range(n)]

        if self._use_yolo:
            try:
                detections: list[tuple[bool, float]] = []
                for start in range(0, n, self._batch_size):
                    batch = images[start: start + self._batch_size]
                    logger.info("yolo_batch", start=start, total=n)
                    detections.extend(self._get_yolo().detect_batch(batch))
                for i, (detected, conf) in enumerate(detections):
                    results[i].person_detected = detected
                    results[i].person_confidence = conf
            except Exception as exc:
                logger.error("yolo_failed", error=str(exc))

        if self._use_mtcnn:
            try:
                counts: list[int] = []
                for start in range(0, n, self._batch_size):
                    batch = images[start: start + self._batch_size]
                    logger.info("mtcnn_batch", start=start, total=n)
                    counts.extend(self._get_mtcnn().detect_batch(batch))
                for i, count in enumerate(counts):
                    results[i].face_count = count
            except Exception as exc:
                logger.error("mtcnn_failed", error=str(exc))

        return results


def availability() -> dict[str, object]:
    info: dict[str, object] = {}
    for name, pkg in [("yolo", "ultralytics"), ("mtcnn", "facenet_pytorch")]:
        try:
            __import__(pkg)
            info[name] = True
        except ImportError:
            info[name] = False
    return info
