"""argus-curator — the curation stage of the Argus suite.

Decides *which images, of whom, at what quality* belong in a LoRA dataset:
training-suitability scoring, near-duplicate dedup, and identity-aware face
clustering. Emits a manifest that argus-lens captions verbatim — they share one
:class:`~argus_curator.models.TargetProfile`. A speech scan
(:func:`~argus_curator.speech.scan_speech_folder`) scores voice clips the same way.
"""

from __future__ import annotations

try:
    # Written by hatch-vcs at build time (see pyproject [tool.hatch.build.hooks.vcs]).
    from argus_curator._version import __version__
except ImportError:  # running from a source checkout that hasn't been built
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("argus-curator")
    except PackageNotFoundError:
        __version__ = "0.0.0+unknown"

from argus_curator.models import (
    ClipResult,
    ExportRequest,
    ExportResult,
    FaceCluster,
    FaceConfig,
    FaceDetection,
    ImageResult,
    ScanConfig,
    ScanRequest,
    ScanSummary,
    SpeechConfig,
    SpeechTargetProfile,
    TargetProfile,
)
from argus_curator.scanner import scan_folder, scan_items
from argus_curator.speech import scan_speech_folder

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
    "SpeechTargetProfile",
    "SpeechConfig",
    "ClipResult",
    "scan_speech_folder",
]
