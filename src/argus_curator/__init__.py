"""argus-curator — the curation stage of the Argus suite.

Decides *which images, of whom, at what quality* belong in a LoRA dataset:
training-suitability scoring, near-duplicate dedup, and identity-aware face
clustering. Emits a manifest that argus-lens captions verbatim — they share one
:class:`~argus_curator.models.TargetProfile`.
"""

from __future__ import annotations

__version__ = "0.1.0"

from argus_curator.models import (
    ExportRequest,
    ExportResult,
    FaceCluster,
    FaceConfig,
    FaceDetection,
    ImageResult,
    ScanConfig,
    ScanRequest,
    ScanSummary,
    TargetProfile,
)
from argus_curator.scanner import scan_folder, scan_items

__all__ = [
    "__version__",
    "TargetProfile",
    "ScanConfig",
    "FaceConfig",
    "FaceDetection",
    "FaceCluster",
    "ImageResult",
    "ScanSummary",
    "ScanRequest",
    "ExportRequest",
    "ExportResult",
    "scan_folder",
    "scan_items",
]
