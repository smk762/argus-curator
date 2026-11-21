"""The host-wide GPU lease (argus_cortex.gpu) around curator's CUDA work."""

from __future__ import annotations

import contextlib
from pathlib import Path

import pytest
from argus_cortex.gpu import GpuLease, GpuLeaseTimeout, LeaseCoordinator, NullCoordinator
from argus_cortex.gpu_broker import BrokerCoordinator
from argus_cortex.gpu_wire import load_tenant_manifest
from test_attributes import _App, _Face, _png

from argus_curator import faces, garments, runtime, scenes
from argus_curator.models import FaceConfig, GarmentConfig, ImageResult, SceneConfig

_CUDA = ["CUDAExecutionProvider", "CPUExecutionProvider"]


class _Recorder:
    """A coordinator that grants every lease and records what was asked."""

    name = "recorder"

    def __init__(self) -> None:
        self.leases: list[tuple[str, int]] = []
        self.options: list[dict[str, object]] = []

    @contextlib.contextmanager
    def lease(self, *, caller: str, min_vram_mb: int, **options: object):  # noqa: ANN201
        self.leases.append((caller, min_vram_mb))
        self.options.append(options)
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


def test_no_lease_when_cuda_was_asked_but_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """device=cuda with only the CPU EP must not serialize the host GPU."""
    import sys
    import types

    fake = types.ModuleType("onnxruntime")
    fake.get_available_providers = lambda: ["CPUExecutionProvider"]  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "onnxruntime", fake)
    monkeypatch.setattr(runtime, "_cuda_preloaded", False)
    monkeypatch.setattr(runtime, "gpu_coordinator", lambda: pytest.fail("took the GPU lease without CUDA"))
    assert runtime.providers("cuda") == ["CPUExecutionProvider"]
    with runtime.gpu_lease("cuda", "faces"):
        pass


def test_no_lease_when_onnxruntime_cannot_be_probed(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    fake = types.ModuleType("onnxruntime")

    def _boom() -> list[str]:
        raise RuntimeError("ort broken")

    fake.get_available_providers = _boom  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "onnxruntime", fake)
    monkeypatch.setattr(runtime, "gpu_coordinator", lambda: pytest.fail("took the GPU lease without CUDA"))
    assert runtime.providers("cuda") == ["CPUExecutionProvider"]
    with runtime.gpu_lease("cuda", "faces"):
        pass


def test_lease_asks_for_the_model_footprint(on_cuda: _Recorder) -> None:
    with runtime.gpu_lease("cuda", "faces", "load"), runtime.gpu_lease("cuda", "scenes"):
        pass
    assert on_cuda.leases == [
        ("argus-curator:faces:load", runtime.VRAM_MB["faces"]["load"]),
        ("argus-curator:scenes:run", runtime.VRAM_MB["scenes"]["run"]),
    ]


def test_lease_names_the_kind_and_asks_interactive_only_when_someone_waits(on_cuda: _Recorder) -> None:
    with runtime.gpu_lease("cuda", "faces", wait_s=30), runtime.gpu_lease("cuda", "scenes"):
        pass
    assert on_cuda.options == [
        {"kind": "faces", "priority": "interactive"},
        {"kind": "scenes", "priority": None},  # the kind's own from gpu-tenant.yaml: batch
    ]


def test_gpu_tenant_manifest_declares_every_leased_model() -> None:
    """gpu-tenant.yaml validates as cortex's TenantManifest and covers every kind gpu_lease names."""
    manifest = load_tenant_manifest(Path(__file__).parent.parent / "gpu-tenant.yaml")
    assert manifest.tenant == "argus-curator"
    leased = {"faces", "scenes", "garments", "embeddings", "mos", "speakers"}  # bodies run inside the faces lease
    assert {k.kind for k in manifest.kinds} == leased
    for model in leased:
        kind = manifest.kind_for(model)
        assert kind is not None and kind.model_mb == runtime.VRAM_MB[model]["load"]
        assert kind.vram_mb == runtime.VRAM_MB[model]["load"] + runtime.VRAM_MB[model]["run"]


def test_wait_s_caps_a_broker_lease_and_its_fallback(tmp_path: Path) -> None:
    fallback = LeaseCoordinator(lock_path=str(tmp_path / "gpu.lock"), timeout_s=300, probe=lambda d: None)
    broker = BrokerCoordinator("http://iris:8120", tenant="argus-curator", fallback=fallback, timeout_s=600)
    capped = runtime._capped(broker, 30)
    assert isinstance(capped, BrokerCoordinator) and capped is not broker
    assert capped.timeout_s == 30 and capped.fallback.timeout_s == 30  # type: ignore[attr-defined]
    assert broker.timeout_s == 600 and broker.fallback is fallback  # the shared coordinator is untouched
    assert fallback.timeout_s == 300 and capped.fallback is not fallback  # copied, not cut in place


def test_capped_broker_is_built_once_per_cap(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """POST /faces reuses one capped copy, so it keeps its broker registration between calls."""
    fallback = LeaseCoordinator(lock_path=str(tmp_path / "gpu.lock"), timeout_s=300, probe=lambda d: None)
    broker = BrokerCoordinator("http://iris:8120", tenant="argus-curator", fallback=fallback, timeout_s=600)
    monkeypatch.setattr(runtime, "_coordinator", broker)
    monkeypatch.setattr(runtime, "_capped_cache", {})
    first = runtime._capped_coordinator(30)
    assert first is not broker and runtime._capped_coordinator(30) is first
    assert runtime._capped(broker, 900) is broker  # a cap above both waits changes nothing


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
    monkeypatch.setattr(faces, "_get_app", lambda cfg, **_k: app)
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
        def lease(self, *, caller: str, min_vram_mb: int, **_options: object):  # noqa: ANN201
            if caller.endswith(":run"):
                raise GpuLeaseTimeout("busy")
            yield

    monkeypatch.setattr(runtime, "_coordinator", _Busy())
    monkeypatch.setattr(faces, "_get_app", lambda cfg, **_k: _App({400: []}))
    monkeypatch.setattr(faces, "_attribute_sessions", lambda cfg: (None, None))
    results, items = _items(400)
    with pytest.raises(GpuLeaseTimeout):
        faces.detect_faces(results, items, FaceConfig(enabled=True, device="cuda"))  # strict=False, a scan


def test_a_load_lease_timeout_fails_a_scan_rather_than_skipping_faces(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime, "providers", lambda device: _CUDA)

    class _BusyLoad(_Recorder):
        @contextlib.contextmanager
        def lease(self, *, caller: str, min_vram_mb: int, **_options: object):  # noqa: ANN201
            if caller.endswith(":load"):
                raise GpuLeaseTimeout("busy")
            yield

    monkeypatch.setattr(runtime, "_coordinator", _BusyLoad())
    monkeypatch.setattr(faces, "_get_app", lambda cfg, **_k: _App({400: []}))
    monkeypatch.setattr(faces, "_attribute_sessions", lambda cfg: (None, None))
    results, items = _items(400)
    with pytest.raises(GpuLeaseTimeout):
        faces.detect_faces(results, items, FaceConfig(enabled=True, device="cuda"))


def test_scene_lease_timeout_fails_the_scan_rather_than_skipping_a_batch(
    on_cuda: _Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(scenes, "get_session", lambda provs: object())

    class _BusyRun(_Recorder):
        @contextlib.contextmanager
        def lease(self, *, caller: str, min_vram_mb: int, **_options: object):  # noqa: ANN201
            if caller.endswith(":run"):
                raise GpuLeaseTimeout("busy")
            yield

    monkeypatch.setattr(runtime, "_coordinator", _BusyRun())
    results, items = _items(400)
    with pytest.raises(GpuLeaseTimeout):
        scenes.tag_images(results, items, SceneConfig(enabled=True, device="cuda"))
    assert results[0].scene is None


def test_garment_lease_timeout_fails_the_scan_rather_than_skipping_an_image(
    on_cuda: _Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(garments, "get_session", lambda provs: object())

    class _BusyRun(_Recorder):
        @contextlib.contextmanager
        def lease(self, *, caller: str, min_vram_mb: int, **_options: object):  # noqa: ANN201
            if caller.endswith(":run"):
                raise GpuLeaseTimeout("busy")
            yield

    monkeypatch.setattr(runtime, "_coordinator", _BusyRun())
    results, items = _items(400)
    with pytest.raises(GpuLeaseTimeout):
        garments.detect_images(results, items, GarmentConfig(enabled=True, device="cuda"))
    assert results[0].garments is None


def test_cuda_libraries_are_preloaded_once_and_only_for_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """onnxruntime-gpu's pip-wheel CUDA libraries are off the loader path; without the preload, CPU fallback."""
    import sys
    import types

    calls: list[int] = []
    fake = types.ModuleType("onnxruntime")
    fake.get_available_providers = lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]  # type: ignore[attr-defined]
    fake.preload_dlls = lambda: calls.append(1)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "onnxruntime", fake)
    monkeypatch.setattr(runtime, "_cuda_preloaded", False)
    assert runtime.providers("cpu") == ["CPUExecutionProvider"] and calls == []
    assert runtime.providers("auto") == _CUDA and runtime.providers("cuda") == _CUDA
    assert calls == [1]
    fake.get_available_providers = lambda: ["CPUExecutionProvider"]  # type: ignore[attr-defined]
    assert runtime.providers("auto") == runtime.providers("cuda") == ["CPUExecutionProvider"]


# --- The CUDA arena cap and the peak reported on release (#189) ----------------------------

_CAPPED = {"arena_extend_strategy": "kSameAsRequested", "gpu_mem_limit": str(1024 * 1024 * 1024)}


def test_cuda_sessions_get_a_capped_arena_that_grows_only_as_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    """ORT's default arena doubles with no ceiling and keeps it: one scan held ~20 GB on 512 MB leases."""
    monkeypatch.delenv(runtime.GPU_MEM_LIMIT_ENV, raising=False)
    assert runtime.session_providers(_CUDA) == [("CUDAExecutionProvider", _CAPPED), "CPUExecutionProvider"]
    assert runtime.session_providers(["CPUExecutionProvider"]) == ["CPUExecutionProvider"]
    extra = runtime.session_providers(_CUDA, use_tf32="0")[0]
    assert extra == ("CUDAExecutionProvider", {**_CAPPED, "use_tf32": "0"})


@pytest.mark.parametrize(("raw", "mb"), [("2048", 2048), ("", 1024), ("0", 1024), ("-5", 1024), ("lots", 1024)])
def test_the_arena_cap_comes_from_the_environment(monkeypatch: pytest.MonkeyPatch, raw: str, mb: int) -> None:
    monkeypatch.setenv(runtime.GPU_MEM_LIMIT_ENV, raw)
    assert runtime.cuda_provider_options()["gpu_mem_limit"] == str(mb * 1024 * 1024)


def test_curators_own_sessions_are_built_with_the_cap(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import sys
    import types

    seen: dict[str, object] = {}
    fake = types.ModuleType("onnxruntime")
    fake.InferenceSession = lambda path, providers: seen.update(providers=providers)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "onnxruntime", fake)
    monkeypatch.delenv(runtime.GPU_MEM_LIMIT_ENV, raising=False)
    model = tmp_path / "m.onnx"
    model.write_bytes(b"")
    runtime._build_session(model, _CUDA, missing_hint="", extra="faces")
    assert seen["providers"] == [("CUDAExecutionProvider", _CAPPED), "CPUExecutionProvider"]


def test_insightface_sessions_are_built_with_the_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """FaceAnalysis and the body detector build their own sessions, so they are handed the options too."""
    import sys
    import types

    from argus_curator import bodies

    seen: dict[str, object] = {}

    class _FaceAnalysis:
        def __init__(self, **kwargs: object) -> None:
            seen["faces"] = kwargs["providers"]

        def prepare(self, **kwargs: object) -> None:
            pass

    app_mod = types.ModuleType("insightface.app")
    app_mod.FaceAnalysis = _FaceAnalysis  # type: ignore[attr-defined]
    utils_mod = types.ModuleType("insightface.utils")
    utils_mod.ensure_available = lambda *a, **k: None  # type: ignore[attr-defined]
    det_mod = types.ModuleType("insightface.model_zoo.person_detection")
    det_mod.PersonDetection = lambda descriptor, providers: seen.update(bodies=providers)  # type: ignore[attr-defined]
    pkg_mod = types.ModuleType("insightface.model_zoo.person_package")
    pkg_mod.ensure_person_package = lambda name: {"tasks": {"person_detection": {}}}  # type: ignore[attr-defined]
    for name, mod in {
        "insightface": types.ModuleType("insightface"),
        "insightface.app": app_mod,
        "insightface.utils": utils_mod,
        "insightface.model_zoo": types.ModuleType("insightface.model_zoo"),
        "insightface.model_zoo.person_detection": det_mod,
        "insightface.model_zoo.person_package": pkg_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(faces, "providers", lambda device: _CUDA)
    monkeypatch.setattr(bodies, "providers", lambda device: _CUDA)
    monkeypatch.setattr(faces, "gpu_lease", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.delenv(runtime.GPU_MEM_LIMIT_ENV, raising=False)
    cfg = FaceConfig(enabled=True, device="cuda")
    faces._build_app(cfg)
    bodies._build_detector(cfg)
    assert seen["faces"] == [("CUDAExecutionProvider", _CAPPED), "CPUExecutionProvider"]
    # insightface drops its own TF32-off for a provider given as a tuple, so it rides along.
    assert seen["bodies"] == [("CUDAExecutionProvider", {**_CAPPED, "use_tf32": "0"}), "CPUExecutionProvider"]


class _Granting:
    """A coordinator that grants a real GpuLease and keeps it for the test to read."""

    name = "granting"

    def __init__(self) -> None:
        self.handles: list[GpuLease] = []

    @contextlib.contextmanager
    def lease(self, **_: object):  # noqa: ANN201
        handle = GpuLease()
        self.handles.append(handle)
        yield handle


def test_the_lease_reports_the_vram_the_process_holds_on_release(monkeypatch: pytest.MonkeyPatch) -> None:
    granting = _Granting()
    monkeypatch.setattr(runtime, "providers", lambda device: _CUDA)
    monkeypatch.setattr(runtime, "_coordinator", granting)
    monkeypatch.setattr(runtime, "process_vram_mb", lambda: 1234)
    with runtime.gpu_lease("cuda", "faces") as handle:
        assert handle.peak_vram_mb is None
    assert granting.handles[0].peak_vram_mb == 1234
    # Unknown (a container's PID namespace hides the process from NVML): nothing is reported.
    monkeypatch.setattr(runtime, "process_vram_mb", lambda: None)
    with runtime.gpu_lease("cuda", "faces"):
        pass
    assert granting.handles[1].peak_vram_mb is None


def test_a_lease_that_yields_no_handle_still_works(on_cuda: _Recorder, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime, "process_vram_mb", lambda: pytest.fail("measured with nowhere to report"))
    with runtime.gpu_lease("cuda", "faces") as handle:
        assert handle is None


def test_process_vram_counts_only_this_process_on_every_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    import os
    import sys
    import types

    mb = 1024 * 1024
    proc = types.SimpleNamespace
    by_gpu = {
        0: [proc(pid=os.getpid(), usedGpuMemory=700 * mb), proc(pid=os.getpid() + 1, usedGpuMemory=9000 * mb)],
        1: [proc(pid=os.getpid(), usedGpuMemory=300 * mb)],
    }
    fake = types.ModuleType("pynvml")
    fake.nvmlInit = lambda: None  # type: ignore[attr-defined]
    fake.nvmlDeviceGetCount = lambda: 2  # type: ignore[attr-defined]
    fake.nvmlDeviceGetHandleByIndex = lambda i: i  # type: ignore[attr-defined]
    fake.nvmlDeviceGetComputeRunningProcesses = lambda h: by_gpu[h]  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    monkeypatch.setattr(runtime, "free_vram_mb", lambda device: 5000)
    assert runtime.process_vram_mb() == 1000
    # Not listed (the host PID differs inside a container): unknown, not zero.
    by_gpu[0], by_gpu[1] = [proc(pid=os.getpid() + 1, usedGpuMemory=9000 * mb)], []
    assert runtime.process_vram_mb() is None
    # No NVML at all.
    monkeypatch.setattr(runtime, "free_vram_mb", lambda device: None)
    assert runtime.process_vram_mb() is None


def test_attribute_models_run_a_crowd_in_fixed_size_chunks() -> None:
    """One pass per crowd shot grew the arena with every new face count, and under the cap it could lose ages."""
    import numpy as np

    from argus_curator import fairface, mivolo

    class _Session:
        def __init__(self, width: int) -> None:
            self.width, self.sizes = width, []

        def get_inputs(self):  # noqa: ANN202
            return [type("I", (), {"name": "x"})()]

        def run(self, _outputs: object, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
            x = feeds["x"]
            self.sizes.append(len(x))
            # Each row's output carries a per-row value so the order can be checked.
            return [np.repeat(x.reshape(len(x), -1)[:, :1], self.width, axis=1)]

    batch = np.arange(19, dtype=np.float32)[:, None, None, None] * np.ones((19, 6, 2, 2), np.float32)
    session = _Session(3)
    out = runtime.run_batched(session, batch, size=8)
    assert session.sizes == [8, 8, 3]
    assert out[:, 0].tolist() == list(range(19))

    from PIL import Image

    image = Image.new("RGB", (800, 800))
    mv = _Session(3)
    # The models themselves run ATTRIBUTE_BATCH (one) face per pass.
    assert runtime.ATTRIBUTE_BATCH == 1
    assert len(mivolo.predict(mv, image, [[10.0 * i, 10.0, 40.0, 40.0] for i in range(19)])) == 19
    assert mv.sizes == [1] * 19
    mv.sizes.clear()
    assert len(mivolo.predict_bodies(mv, [Image.new("RGB", (60, 120))] * 10)) == 10
    assert mv.sizes == [1] * 10
    ff = _Session(18)
    kps = np.array([[30, 40], [70, 40], [50, 60], [35, 80], [65, 80]], np.float64)
    assert fairface.predict(ff, image, [kps] * 9).shape == (9, 18)
    assert ff.sizes == [1] * 9
