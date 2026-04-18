"""argus-curator — dataset curation for LoRA training."""

from argus_curator.scanner import scan_folder, scan_bytes_batch
from argus_curator.types import CurateConfig, ScanSummary, ImageResult

try:
    from argus_curator._version import __version__
except ImportError:
    __version__ = "0.1.0"

__all__ = [
    "scan_folder",
    "scan_bytes_batch",
    "CurateConfig",
    "ScanSummary",
    "ImageResult",
    "__version__",
]
