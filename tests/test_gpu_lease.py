"""The host-wide GPU lease (argus_cortex.gpu) around curator's CUDA work."""

from __future__ import annotations

import contextlib
from pathlib import Path

import pytest
from argus_cortex.gpu import GpuLeaseTimeout, LeaseCoordinator, NullCoordinator
from test_attributes import _App, _Face, _png

from argus_curator import faces, runtime
from argus_curator.models import FaceConfig, ImageResult

_CUDA = ["CUDAExecutionProvider", "CPUExecutionProvider"]


class _Recorder:
    """A coordinator that grants every lease and records what was asked."""

    name = "recorder"

    def __init__(self) -> None:
        self.leases: list[tuple[str, int]] = []

    @contextlib.contextmanager
    def lease(self, *, caller: str, min_vram_mb: int):  # noqa: ANN201
        self.leases.append((caller, min_vram_mb))
        yield


@pytest.fixture
def on_cuda(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    recorder = _Recorder()
    monkeypatch.setattr(runtime, "providers", lambda device: _CUDA)
    monkeypatch.setattr(runtime, "_coordinator", recorder)
    return recorder


def test_no_lease_on_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime, "providers", lambda device: ["CPUExecutionProvider"])
    monkeypatch.setattr(runtime, "gpu_coordinator", lambda: pytest.fail("a CPU run took the GPU lease"))
    with runtime.gpu_lease("auto", "faces"):
        pass


def test_lease_asks_for_the_model_footprint(on_cuda: _Recorder) -> None:
    with runtime.gpu_lease("cuda", "faces", "load"), runtime.gpu_lease("cuda", "scenes"):
        pass
    assert on_cuda.leases == [
        ("argus-curator:faces:load", runtime.VRAM_MB["faces"]["load"]),
        ("argus-curator:scenes:run", runtime.VRAM_MB["scenes"]["run"]),
    ]


def test_coordinator_comes_from_the_environment_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(runtime, "_coordinator", None)
    monkeypatch.setenv("ARGUS_GPU_COORDINATOR", "lease")
    monkeypatch.setenv("ARGUS_GPU_LEASE_PATH", str(tmp_path / "gpu.lock"))
    first = runtime.gpu_coordinator()
    assert isinstance(first, LeaseCoordinator) and first.lock_path == str(tmp_path / "gpu.lock")
    assert runtime.gpu_coordinator() is first
    monkeypatch.setattr(runtime, "_coordinator", None)
    monkeypatch.delenv("ARGUS_GPU_COORDINATOR")
    assert isinstance(runtime.gpu_coordinator(), NullCoordinator)


def test_wait_s_caps_the_wait_on_a_held_lease(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Another argus process (here, a second lease on the same file) holds the GPU."""
    lock = str(tmp_path / "gpu.lock")
    monkeypatch.setattr(runtime, "providers", lambda device: _CUDA)
    monkeypatch.setattr(runtime, "_coordinator", LeaseCoordinator(lock_path=lock, timeout_s=300, probe=lambda d: None))
    holder = LeaseCoordinator(lock_path=lock, probe=lambda d: None)
    with (
        holder.lease(caller="argus-lens", min_vram_mb=0),
        pytest.raises(GpuLeaseTimeout),
        runtime.gpu_lease("cuda", "faces", wait_s=0.1),
    ):
        pass
    with runtime.gpu_lease("cuda", "faces", wait_s=0.1):  # free once the other lets go
        pass


def _items(*widths: int) -> tuple[list[ImageResult], list[tuple[str, str, bytes]]]:
    items = [(f"{w}.png", f"/d/{w}.png", _png(w)) for w in widths]
    return [ImageResult(rel_path=rel, abs_path=abs_, passed=True) for rel, abs_, _ in items], items


def test_detect_faces_takes_the_lease_per_image(on_cuda: _Recorder, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _App({400: [_Face([100, 100, 300, 300], [1, 0, 0])], 401: []})
    monkeypatch.setattr(faces, "_get_app", lambda cfg: app)
    monkeypatch.setattr(faces, "_attribute_sessions", lambda cfg: (None, None))
    results, items = _items(400, 401)
    faces.detect_faces(results, items, FaceConfig(enabled=True, device="cuda"))
    assert [caller for caller, _mb in on_cuda.leases] == [
        "argus-curator:faces:load",
        "argus-curator:faces:run",
        "argus-curator:faces:run",
    ]
    assert results[0].face_count == 1


def test_a_lease_timeout_fails_a_scan_rather_than_skipping_faces(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime, "providers", lambda device: _CUDA)

    class _Busy(_Recorder):
        @contextlib.contextmanager
        def lease(self, *, caller: str, min_vram_mb: int):  # noqa: ANN201
            if caller.endswith(":run"):
                raise GpuLeaseTimeout("busy")
            yield

    monkeypatch.setattr(runtime, "_coordinator", _Busy())
    monkeypatch.setattr(faces, "_get_app", lambda cfg: _App({400: []}))
    monkeypatch.setattr(faces, "_attribute_sessions", lambda cfg: (None, None))
    results, items = _items(400)
    with pytest.raises(GpuLeaseTimeout):
        faces.detect_faces(results, items, FaceConfig(enabled=True, device="cuda"))  # strict=False, a scan
