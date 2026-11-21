"""Semantic embeddings (#188): the selection term, the in-memory store and the scan phase.

No model and no GPU: the ONNX session is faked, and selection is fed vectors directly.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from argus_curator import embeddings, runtime, scanner
from argus_curator.models import (
    EmbeddingConfig,
    ExportRequest,
    FaceConfig,
    ImageResult,
    ScanConfig,
    ScanSummary,
    SelectRequest,
    TargetProfile,
)
from argus_curator.selection import decide_selection, select_diverse, selection_problem
from argus_curator.store import ScanStore

H = "ffffffffffffffff"


def _img(rel: str, phash: str = H, score: float = 0.9) -> ImageResult:
    return ImageResult(rel_path=rel, abs_path=f"/d/{rel}", score=score, passed=True, phash=phash)


def _unit(*xs: float) -> np.ndarray:
    v = np.array(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


# -- selection ---------------------------------------------------------------------------------


def _burst_pool() -> tuple[list[ImageResult], dict[str, np.ndarray]]:
    # Same pHash everywhere, so pHash alone gives no reason to prefer anything
    # over score; three near-identical burst frames outscore one other shot.
    pool = [_img("burst1.jpg", score=0.95), _img("burst2.jpg", score=0.94), _img("burst3.jpg", score=0.93)]
    pool.append(_img("other.jpg", score=0.80))
    vecs = {
        "burst1.jpg": _unit(1.0, 0.0, 0.0),
        "burst2.jpg": _unit(1.0, 0.02, 0.0),
        "burst3.jpg": _unit(1.0, 0.0, 0.02),
        "other.jpg": _unit(0.0, 1.0, 0.0),
    }
    return pool, vecs


def test_diversify_semantic_spreads_near_duplicates() -> None:
    pool, vecs = _burst_pool()
    plain = select_diverse(pool, 2, 0.5)
    spread = select_diverse(pool, 2, 0.5, semantic=True, embeddings=vecs)
    assert {r.rel_path for r in plain} == {"burst1.jpg", "burst2.jpg"}
    assert {r.rel_path for r in spread} == {"burst1.jpg", "other.jpg"}

    req = ExportRequest(dest="/o", min_score=0.0, max_keep=2, diversify_semantic=True)
    selected, why = decide_selection(pool, req, 0.5, embeddings=vecs)
    assert {r.rel_path for r in selected} == {"burst1.jpg", "other.jpg"}
    assert why["burst2.jpg"] == "diversity-trimmed"


def test_semantic_term_is_off_by_default() -> None:
    pool, vecs = _burst_pool()
    assert ExportRequest(dest="/o").diversify_semantic is False
    assert SelectRequest().diversify_semantic is False
    assert EmbeddingConfig().enabled is False
    # Embeddings on hand change nothing until the flag asks for the term.
    assert select_diverse(pool, 2, 0.5, embeddings=vecs) == select_diverse(pool, 2, 0.5)
    req = ExportRequest(dest="/o", min_score=0.0, max_keep=2)
    assert decide_selection(pool, req, 0.5, embeddings=vecs)[0] == decide_selection(pool, req, 0.5)[0]


def test_semantic_term_is_in_play_only_when_some_candidate_has_an_embedding() -> None:
    pool, _vecs = _burst_pool()
    plain = select_diverse(pool, 2, 0.5)
    assert select_diverse(pool, 2, 0.5, semantic=True, embeddings={}) == plain
    assert select_diverse(pool, 2, 0.5, semantic=True, embeddings=None) == plain
    # Vectors for images outside the pool leave the term out too.
    assert select_diverse(pool, 2, 0.5, semantic=True, embeddings={"elsewhere.jpg": _unit(1.0, 0.0)}) == plain


def test_unembedded_images_are_not_favoured_by_semantic_diversity() -> None:
    """Without an embedding, an image must still face the semantic term, or it wins on pHash alone."""
    pool = [
        _img("a.jpg", "ffffffffffffffff", 0.95),
        _img("bare1.jpg", "0000000000000000", 0.94),
        _img("bare2.jpg", "0000000000000001", 0.93),
        _img("b.jpg", "0000000000000003", 0.92),
    ]
    vecs = {"a.jpg": _unit(1.0, 0.0), "b.jpg": _unit(0.0, 1.0)}
    picked = [r.rel_path for r in select_diverse(pool, 3, 0.6, semantic=True, embeddings=vecs)]
    # The image with no embedding counts as a repeat, so the far embedded one wins.
    assert picked[:2] == ["a.jpg", "b.jpg"]


def test_semantic_term_combines_with_the_others() -> None:
    pool, vecs = _burst_pool()
    spread = select_diverse(pool, 2, 0.5, scenes=True, outfits=True, semantic=True, embeddings=vecs)
    assert {r.rel_path for r in spread} == {"burst1.jpg", "other.jpg"}


# -- selection_problem ------------------------------------------------------------------------


def _summary(enabled: bool, vecs: dict[str, np.ndarray] | None = None) -> ScanSummary:
    pool, _ = _burst_pool()
    summary = ScanSummary(
        scan_id="s1",
        folder="/d",
        target_profile=TargetProfile(),
        config=ScanConfig(),
        faces_config=FaceConfig(),
        embeddings_config=EmbeddingConfig(enabled=enabled),
        total=len(pool),
        passed=len(pool),
        rejected=0,
        duplicates=0,
        similar_clusters=0,
        results=pool,
    )
    summary.embeddings.update(vecs or {})
    return summary


def test_diversify_semantic_is_refused_without_embeddings() -> None:
    req = ExportRequest(dest="/o", diversify_semantic=True)
    assert "rescan with embeddings on" in (selection_problem(req, _summary(False)) or "")
    # A scan that ran them, once they are gone (a restart): refused, not silently ignored.
    assert "held in memory only" in (selection_problem(req, _summary(True)) or "")
    assert selection_problem(req, _summary(True, _burst_pool()[1])) is None
    assert selection_problem(ExportRequest(dest="/o"), _summary(False)) is None


# -- never persisted, purged with the scan --------------------------------------------------


def test_embeddings_are_never_serialised() -> None:
    summary = _summary(True, _burst_pool()[1])
    dumped = summary.model_dump_json()
    assert "burst1.jpg" in dumped  # the rows are there
    assert "_embeddings" not in dumped and "0.99" not in dumped
    assert "embeddings" not in ScanSummary.model_json_schema()["properties"]
    assert ScanSummary.model_validate_json(dumped).embeddings == {}


def test_store_holds_embeddings_in_memory_only(tmp_path: Path) -> None:
    store = ScanStore(tmp_path)
    _, vecs = _burst_pool()
    path = store.save(_summary(True, vecs))
    assert "_embeddings" not in path.read_text() and list(tmp_path.iterdir()) == [path]
    loaded = store.load("s1")
    assert loaded is not None and set(loaded.embeddings) == set(vecs)
    # Another process (a restart) reads the same file and has none.
    again = ScanStore(tmp_path).load("s1")
    assert again is not None and again.embeddings == {}


def test_store_drops_embeddings_with_the_scan_and_its_rows(tmp_path: Path) -> None:
    store = ScanStore(tmp_path)
    _, vecs = _burst_pool()
    store.save(_summary(True, vecs))

    # A row dropped (delete, move) and the reloaded copy saved: its vector goes too.
    current = store.load("s1")
    assert current is not None
    scanner.drop_results(current, ["burst2.jpg"])
    store.save(current)
    assert set(store.held_embeddings("s1")) == {"burst1.jpg", "burst3.jpg", "other.jpg"}

    store.delete("s1")
    assert store.held_embeddings("s1") == {}

    store.save(_summary(True, vecs))
    store.delete_all()
    assert store.held_embeddings("s1") == {}


def test_a_snapshot_never_holds_embeddings(tmp_path: Path) -> None:
    store = ScanStore(tmp_path)
    store.save_snapshot(_summary(True))
    assert store.held_embeddings("s1") == {}
    store.drop_unfinished()
    assert store.held_embeddings("s1") == {}


# -- the scan phase -------------------------------------------------------------------------


class _Input:
    name = "input"


class _Session:
    def __init__(self) -> None:
        self.batches: list[int] = []

    def get_inputs(self) -> list[_Input]:
        return [_Input()]

    def run(self, _outputs, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        x = feeds["input"]
        assert x.shape[1:] == (3, 224, 224) and x.dtype == np.float32
        self.batches.append(x.shape[0])
        # Unnormalised: a vector along the mean colour, scaled by 3.
        means = x.mean(axis=(2, 3))
        return [np.concatenate([means, np.ones((len(x), 1))], axis=1) * 3.0]


def _png(red: int, size: tuple[int, int] = (300, 200)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (red, 40, 0)).save(buf, "PNG")
    return buf.getvalue()


def test_embed_images_runs_full_batches_and_skips_undecodable(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _Session()
    monkeypatch.setattr(embeddings, "get_session", lambda provs: session)
    leases: list[tuple[str, str]] = []
    monkeypatch.setattr(embeddings, "gpu_lease", lambda device, model, phase="run": _Lease(leases, model, phase))
    items = [(f"{i}.png", f"/d/{i}", _png(20 * i)) for i in range(embeddings.BATCH + 2)]
    items.append(("cut.png", "/d/cut", _png(5)[:40]))
    results = [ImageResult(rel_path=rel, abs_path=a, passed=True) for rel, a, _ in items]
    results.append(ImageResult(rel_path="rejected.png", abs_path="/d/r", passed=False))
    vecs = embeddings.embed_images(results, items, EmbeddingConfig(enabled=True))
    assert set(vecs) == {f"{i}.png" for i in range(embeddings.BATCH + 2)}
    # Every pass has one shape: the short last batch is padded (#189).
    assert session.batches == [embeddings.BATCH, embeddings.BATCH]
    assert all(abs(float(np.linalg.norm(v)) - 1.0) < 1e-5 for v in vecs.values())
    assert all(r.model_dump() == ImageResult(**r.model_dump()).model_dump() for r in results)
    assert leases[0] == ("embeddings", "load") and set(leases[1:]) == {("embeddings", "run")}


class _Lease:
    def __init__(self, seen: list[tuple[str, str]], model: str, phase: str) -> None:
        seen.append((model, phase))

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> None:
        return None


def test_embed_images_is_a_no_op_when_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(embeddings, "get_session", lambda provs: pytest.fail("no session when off"))
    assert embeddings.embed_images([], [], EmbeddingConfig()) == {}


def test_a_gpu_lease_timeout_fails_the_scan_rather_than_skipping(monkeypatch: pytest.MonkeyPatch) -> None:
    from argus_cortex.gpu import GpuLeaseTimeout

    monkeypatch.setattr(embeddings, "get_session", lambda provs: _Session())

    def lease(device: str, model: str, phase: str = "run"):
        if phase == "run":
            raise GpuLeaseTimeout("busy")
        return _Lease([], model, phase)

    monkeypatch.setattr(embeddings, "gpu_lease", lease)
    items = [("a.png", "/d/a", _png(10))]
    results = [ImageResult(rel_path="a.png", abs_path="/d/a", passed=True)]
    with pytest.raises(GpuLeaseTimeout):
        embeddings.embed_images(results, items, EmbeddingConfig(enabled=True))


def test_scan_puts_embeddings_on_the_summary_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(embeddings, "get_session", lambda provs: _Session())
    items = [("a.png", "/d/a", _png(20, (600, 600))), ("b.png", "/d/b", _png(200, (600, 600)))]
    summary = scanner.scan_items(
        items,
        TargetProfile(),
        ScanConfig(blur_threshold=0.0),
        FaceConfig(),
        embed_cfg=EmbeddingConfig(enabled=True, device="cpu"),
    )
    assert summary.embeddings_config.enabled is True
    assert set(summary.embeddings) == {"a.png", "b.png"}
    assert "embedding" not in summary.results[0].model_dump_json()

    off = scanner.scan_items(items, TargetProfile(), ScanConfig(blur_threshold=0.0), FaceConfig())
    assert off.embeddings_config.enabled is False and off.embeddings == {}


def test_preprocess_is_clip_resize_and_centre_crop() -> None:
    x = embeddings.preprocess([Image.new("RGB", (640, 320), (255, 255, 255)), Image.new("RGB", (100, 300))])
    assert x.shape == (2, 3, 224, 224) and x.dtype == np.float32
    white = (1.0 - np.array(embeddings.space().mean)) / np.array(embeddings.space().std)
    assert np.allclose(x[0][:, 112, 112], white, atol=1e-4)


def test_the_embedding_space_is_one_pinned_choice() -> None:
    sp = embeddings.space()
    assert embeddings.EMBEDDING_SPACES[embeddings.EMBEDDING_SPACE] is sp
    assert len(sp.weights_sha256) == 64 and len(sp.hf_revision) == 40
    assert embeddings.model_path().name == sp.onnx_name


def test_embedding_session_is_built_with_a_capped_arena_big_enough_for_its_weights(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import sys
    import types

    seen: dict[str, object] = {}
    fake = types.ModuleType("onnxruntime")
    fake.InferenceSession = lambda path, providers: seen.update(providers=providers)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "onnxruntime", fake)
    monkeypatch.delenv(runtime.GPU_MEM_LIMIT_ENV, raising=False)
    model = tmp_path / "clip.onnx"
    model.write_bytes(b"")
    monkeypatch.setenv(embeddings.MODEL_ENV, str(model))
    monkeypatch.setattr(runtime, "_SESSION_CACHE", {})
    embeddings.get_session(["CUDAExecutionProvider", "CPUExecutionProvider"])
    cuda, cpu = seen["providers"]  # type: ignore[misc]
    assert cpu == "CPUExecutionProvider"
    assert cuda == (
        "CUDAExecutionProvider",
        {"arena_extend_strategy": "kSameAsRequested", "gpu_mem_limit": str(embeddings.ARENA_MB * 1024 * 1024)},
    )
    # A larger operator cap wins; it is never lowered below what the weights need.
    monkeypatch.setenv(runtime.GPU_MEM_LIMIT_ENV, "4096")
    assert embeddings.arena_mb() == 4096


# -- HTTP -----------------------------------------------------------------------------------


def test_select_route_uses_held_embeddings_and_refuses_without_them(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from argus_curator.server import create_app

    source = tmp_path / "src"
    source.mkdir()
    app = create_app(cache_dir=str(tmp_path / "cache"), source_root=str(source))
    client = TestClient(app)
    pool, vecs = _burst_pool()
    summary = _summary(True)
    summary.folder = str(source)
    store: ScanStore = app.state.scan_store
    store.save(summary)  # the file alone, as after a restart
    body = {"min_score": 0.0, "max_keep": 2, "diversity_weight": 0.5, "diversify_semantic": True}
    resp = client.post("/scan/s1/select", json=body)
    assert resp.status_code == 400 and "held in memory only" in resp.json()["detail"]

    summary.embeddings.update(vecs)
    store.save(summary)
    resp = client.post("/scan/s1/select", json=body)
    assert resp.status_code == 200, resp.text
    kept = {r["rel_path"] for r in resp.json()["rows"] if not r["keep_reason"]}
    assert kept == {"burst1.jpg", "other.jpg"}
    assert "0.99" not in client.get("/scan/s1").text

    assert client.delete("/scan/s1").status_code == 200
    assert store.held_embeddings("s1") == {}
