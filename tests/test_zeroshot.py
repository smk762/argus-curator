"""Zero-shot profile fit and junk checks from CLIP embeddings (#197).

No model and no GPU: the scores are fed synthetic unit vectors and synthetic
prompt vectors; the scan phase runs on the fake session from test_embeddings.
The committed prompt file is checked as data (its pin, its labels, its space).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from test_embeddings import _png, _Session
from typer.testing import CliRunner

from argus_curator import embeddings, prefilter, scanner, zeroshot
from argus_curator.cli import app
from argus_curator.models import (
    JUNK_KINDS,
    EmbeddingConfig,
    FaceConfig,
    ImageResult,
    PrefilterConfig,
    ScanConfig,
    ScanRequest,
    ScanSummary,
    TargetProfile,
    ZeroShotResult,
)
from argus_curator.server.app import create_app

runner = CliRunner()


# -- the committed prompt file ---------------------------------------------------------------


def test_the_prompt_file_is_the_pinned_one_and_loads() -> None:
    raw = zeroshot.default_prompts_path().read_bytes()
    assert hashlib.sha256(raw).hexdigest() == zeroshot.PROMPTS_SHA256
    prompts = zeroshot.prompt_vectors()
    assert set(prompts.vectors) == set(zeroshot.prompt_labels())
    dim = embeddings.space().dim
    assert all(v.shape == (dim,) and abs(float(np.linalg.norm(v)) - 1.0) < 1e-4 for v in prompts.vectors.values())
    assert prompts.logit_scale > 1


def test_the_prompt_file_holds_exactly_the_prompts_the_code_names_in_this_space() -> None:
    # Editing a prompt without regenerating the file (embeddings-convert --prompts) fails here.
    doc = json.loads(zeroshot.default_prompts_path().read_text())
    assert {label: tuple(entry["prompts"]) for label, entry in doc["labels"].items()} == zeroshot.prompt_labels()
    sp = embeddings.space()
    assert doc["space"] == embeddings.EMBEDDING_SPACE
    assert (doc["hf_repo"], doc["hf_revision"], doc["weights_sha256"]) == (
        sp.hf_repo,
        sp.hf_revision,
        sp.weights_sha256,
    )


def test_the_prompt_file_ships_in_the_package() -> None:
    # Read through importlib.resources, as an installed wheel reads it.
    assert zeroshot._prompts_text() == zeroshot.default_prompts_path().read_bytes()


def test_an_unpinned_prompt_file_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = zeroshot.default_prompts_path().read_bytes()
    monkeypatch.setattr(zeroshot, "_prompts_text", lambda: raw.replace(b'"logit_scale":', b'"logit_scale": '))
    zeroshot.prompt_vectors.cache_clear()
    try:
        with pytest.raises(zeroshot.ModelUnavailable, match="not the pinned prompt set"):
            zeroshot.prompt_vectors()
    finally:
        zeroshot.prompt_vectors.cache_clear()


def test_every_profile_has_its_prompts() -> None:
    # Two styles and five categories, all precomputed: no text tower at scan time.
    from argus_cortex.taxonomy import TARGET_CATEGORIES, TARGET_STYLES

    labels = zeroshot.prompt_labels()
    assert {f"style:{s}" for s in TARGET_STYLES} <= set(labels)
    assert {f"category:{c}" for c in TARGET_CATEGORIES if c != "concept"} <= set(labels)
    assert {f"junk:{k}" for k in JUNK_KINDS} <= set(labels)


# -- scoring on synthetic vectors ------------------------------------------------------------

# A toy space: one axis per category, style, junk kind and clean kind.
_PROFILE = [f"category:{c}" for c in zeroshot.PROFILE_CATEGORIES] + [f"style:{s}" for s in zeroshot.STYLE_PROMPTS]
_LABELS = _PROFILE + [f"junk:{k}" for k in JUNK_KINDS] + [f"clean:{k}" for k in zeroshot.CLEAN_PROMPTS]
DIM = len(_LABELS)


def _axis(label: str, other: str | None = None) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[_LABELS.index(label)] = 1.0
    if other is not None:
        v[_LABELS.index(other)] = 1.0
    return v / np.linalg.norm(v)


PROMPTS = zeroshot.PromptVectors(logit_scale=100.0, vectors={label: _axis(label) for label in _LABELS})


def _img(rel: str, passed: bool = True) -> ImageResult:
    return ImageResult(rel_path=rel, abs_path=f"/d/{rel}", passed=passed, width=1024, height=1024, sharpness=900)


def test_profile_fit_is_the_profiles_own_probability_and_concept_takes_its_style() -> None:
    vecs = np.stack(
        [_axis("category:identity", "style:photo"), _axis("category:setting", "style:anime"), _axis("junk:meme")]
    )
    identity = zeroshot.profile_fit(vecs, TargetProfile(target_category="identity"), PROMPTS)
    assert identity[0] > 0.99 and identity[1] < 0.01
    concept_photo = zeroshot.profile_fit(vecs, TargetProfile(target_category="concept"), PROMPTS)
    concept_anime = zeroshot.profile_fit(vecs, TargetProfile(target_category="concept", target_style="anime"), PROMPTS)
    assert concept_photo[0] > 0.99 and concept_anime[1] > 0.99
    np.testing.assert_allclose(concept_photo + concept_anime, 1.0, atol=1e-6)
    # Nothing profile-like: an even spread, not a confident fit.
    assert abs(identity[2] - 1 / len(zeroshot.PROFILE_CATEGORIES)) < 1e-6
    assert abs(concept_photo[2] - 0.5) < 1e-6


def test_junk_is_the_summed_junk_probability_with_its_likeliest_kind() -> None:
    vecs = np.stack([_axis("junk:screenshot"), _axis("junk:collage", "clean:photo"), _axis("clean:art")])
    probs, kinds = zeroshot.junk(vecs, PROMPTS)
    assert probs[0] > 0.99 and kinds[0] == "screenshot"
    assert abs(probs[1] - 0.5) < 1e-3 and kinds[1] == "collage"
    assert probs[2] < 0.01


def test_score_images_writes_plain_scores_and_names_a_kind_only_past_the_threshold() -> None:
    results = [_img("junk.jpg"), _img("half.jpg"), _img("clean.jpg"), _img("none.jpg"), _img("rej.jpg", False)]
    vecs = {
        "junk.jpg": _axis("junk:text"),
        "half.jpg": _axis("junk:meme", "clean:photo"),
        "clean.jpg": _axis("category:identity", "clean:photo"),
        "rej.jpg": _axis("junk:text"),
    }
    cfg = ScanConfig(profile_fit=True, junk_check=True, junk_threshold=0.6)
    assert zeroshot.score_images(results, vecs, TargetProfile(), cfg, prompts=PROMPTS) == 3
    by = {r.rel_path: r.zero_shot for r in results}
    assert by["junk.jpg"] is not None and by["junk.jpg"].junk_kind == "text" and by["junk.jpg"].junk > 0.99
    assert by["half.jpg"] is not None and by["half.jpg"].junk_kind is None  # 0.5 < 0.6
    assert by["clean.jpg"] is not None and by["clean.jpg"].profile_fit > 0.99 and by["clean.jpg"].junk < 0.5
    # No embedding: unscored, never a perfect fit or clean. A reject is never scored.
    assert by["none.jpg"] is None and by["rej.jpg"] is None
    assert all(isinstance(v, float) for zs in by.values() if zs for v in zs.model_dump().values() if v and v != "text")


def test_only_the_asked_scores_are_computed() -> None:
    results = [_img("a.jpg")]
    vecs = {"a.jpg": _axis("junk:text")}
    assert zeroshot.score_images(results, vecs, TargetProfile(), ScanConfig(), prompts=PROMPTS) == 0
    assert results[0].zero_shot is None
    zeroshot.score_images(results, vecs, TargetProfile(), ScanConfig(profile_fit=True), prompts=PROMPTS)
    assert results[0].zero_shot is not None and results[0].zero_shot.junk is None
    assert results[0].zero_shot.profile_fit is not None


# -- the score: the target bonus, never a reject ---------------------------------------------


def _scored(zs: ZeroShotResult | None, cfg: ScanConfig) -> ImageResult:
    r = _img("a.jpg")
    r.artifact_score = 0.5
    r.zero_shot = zs
    scanner.finalize_score(r, TargetProfile(target_category="setting"), cfg, faces_known=False)
    return r


def test_profile_fit_raises_and_junk_lowers_the_target_bonus() -> None:
    cfg = ScanConfig(profile_fit=True, junk_check=True, profile_fit_weight=0.1, junk_weight=0.2)
    plain = _scored(None, cfg)
    fit = _scored(ZeroShotResult(profile_fit=0.9, junk=0.0), cfg)
    junk = _scored(ZeroShotResult(profile_fit=0.0, junk=1.0, junk_kind="meme"), cfg)
    assert fit.score_breakdown["target_bonus"] == pytest.approx(plain.score_breakdown["target_bonus"] + 0.09)
    assert junk.score_breakdown["target_bonus"] == pytest.approx(plain.score_breakdown["target_bonus"] - 0.2)
    assert fit.score > plain.score > junk.score
    assert fit.score_breakdown["zero_shot_profile_fit"] == pytest.approx(0.09)
    assert junk.score_breakdown["zero_shot_junk"] == pytest.approx(-0.2)
    # Never a reject on its own, however junky.
    assert junk.passed and junk.reject_reason is None


def test_the_score_reads_only_the_switched_on_terms() -> None:
    zs = ZeroShotResult(profile_fit=0.9, junk=1.0, junk_kind="meme")
    off = _scored(zs, ScanConfig())
    assert off.score == _scored(None, ScanConfig()).score
    assert not any(k.startswith("zero_shot_") for k in off.score_breakdown)


def test_junk_prefilter_drops_only_a_named_junk_kind() -> None:
    cfg = PrefilterConfig(drop_junk=True)
    flagged, below, unscored = _img("a.jpg"), _img("b.jpg"), _img("c.jpg")
    flagged.zero_shot = ZeroShotResult(junk=0.9, junk_kind="screenshot")
    below.zero_shot = ZeroShotResult(junk=0.4)
    assert [r.rel_path for r in prefilter.apply([flagged, below, unscored], cfg)] == ["a.jpg"]
    assert flagged.reject_reason == prefilter.JUNK and prefilter.is_prefiltered(flagged)
    assert below.passed and unscored.passed
    # Off by default: nothing drops.
    other = _img("d.jpg")
    other.zero_shot = ZeroShotResult(junk=0.9, junk_kind="screenshot")
    assert prefilter.apply([other], PrefilterConfig()) == [] and other.passed


# -- refused without the vectors, never silently ignored -------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        {"config": {"profile_fit": True}},
        {"config": {"junk_check": True}},
        {"prefilter": {"drop_junk": True}},
    ],
)
def test_a_scan_asking_for_zero_shot_scores_without_embeddings_is_refused(body: dict) -> None:
    with pytest.raises(ValidationError, match="needs the scan's embeddings"):
        ScanRequest(folder="", **body)
    assert ScanRequest(folder="", embeddings={"enabled": True}, **body)
    cfg = ScanConfig(**body.get("config", {}))
    pre = PrefilterConfig(**body.get("prefilter", {}))
    with pytest.raises(ValueError, match="needs the scan's embeddings"):
        scanner.scan_items([], TargetProfile(), cfg, FaceConfig(), prefilter_cfg=pre)


def test_the_weights_and_threshold_alone_turn_nothing_on() -> None:
    assert ScanRequest(folder="", config={"junk_threshold": 0.3, "profile_fit_weight": 0.5, "junk_weight": 0.5})
    with pytest.raises(ValidationError):
        ScanConfig(junk_threshold=1.5)


def test_a_speech_scan_refuses_zero_shot_scores() -> None:
    with pytest.raises(ValidationError, match="config.junk_check"):
        ScanRequest(folder="", target_profile={"modality": "speech"}, config={"junk_check": True})


def test_scan_route_refuses_zero_shot_without_embeddings(dataset: Path, tmp_path: Path) -> None:
    client = TestClient(create_app(cache_dir=str(tmp_path / "cache"), source_root=str(dataset)))
    resp = client.post("/scan/folder", json={"folder": "", "config": {"profile_fit": True}})
    assert resp.status_code == 422 and "embeddings" in resp.text


@pytest.mark.parametrize("flag", ["--profile-fit", "--junk-check", "--drop-junk"])
def test_cli_zero_shot_without_embeddings_fails_before_scanning(dataset: Path, flag: str) -> None:
    result = runner.invoke(app, ["scan", str(dataset), flag])
    assert result.exit_code == 2, result.output
    assert flag in result.output and "--embeddings" in result.output and "Scanning" not in result.output


# -- the scan phase --------------------------------------------------------------------------

# The fake session embeds (r, g, b, 1) of the mean colour: put one prompt per axis.
_FAKE_LABELS = list(zeroshot.prompt_labels())


def _fake_prompts() -> zeroshot.PromptVectors:
    vectors = {label: np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32) for label in _FAKE_LABELS}
    # A red image reads as a screenshot, a dark one as an ordinary image.
    vectors["junk:screenshot"] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    for kind in zeroshot.CLEAN_PROMPTS:
        vectors[f"clean:{kind}"] = embeddings.normalise(np.array([-1.0, -1.0, -1.0, 0.0], dtype=np.float32))
    return zeroshot.PromptVectors(logit_scale=100.0, vectors=vectors)


def _scan(monkeypatch: pytest.MonkeyPatch, cfg: ScanConfig, pre: PrefilterConfig | None = None) -> ScanSummary:
    monkeypatch.setattr(embeddings, "get_session", lambda provs: _Session())
    monkeypatch.setattr(zeroshot, "prompt_vectors", _fake_prompts)
    items = [("red.png", "/d/red", _png(250, (600, 600))), ("dark.png", "/d/dark", _png(5, (600, 600)))]
    return scanner.scan_items(
        items,
        TargetProfile(),
        cfg.model_copy(update={"blur_threshold": 0.0, "cluster_distance": -1}),
        FaceConfig(),
        embed_cfg=EmbeddingConfig(enabled=True, device="cpu"),
        prefilter_cfg=pre,
    )


def test_the_scan_scores_and_keeps_only_plain_values(monkeypatch: pytest.MonkeyPatch) -> None:
    summary = _scan(monkeypatch, ScanConfig(profile_fit=True, junk_check=True))
    by = {r.rel_path: r for r in summary.results}
    red, dark = by["red.png"], by["dark.png"]
    assert red.zero_shot is not None and red.zero_shot.junk_kind == "screenshot"
    assert dark.zero_shot is not None and dark.zero_shot.junk_kind is None
    # Scored, docked, and still passing: no hard reject by default.
    assert red.passed and "zero_shot_junk" in red.score_breakdown
    dumped = summary.model_dump_json()
    assert '"junk_kind":"screenshot"' in dumped and "_embeddings" not in dumped
    reloaded = ScanSummary.model_validate_json(dumped)
    assert reloaded.embeddings == {} and reloaded.results[0].zero_shot is not None


def test_the_scan_drops_junk_only_with_the_prefilter(monkeypatch: pytest.MonkeyPatch) -> None:
    summary = _scan(monkeypatch, ScanConfig(), PrefilterConfig(drop_junk=True))
    by = {r.rel_path: r for r in summary.results}
    assert not by["red.png"].passed and by["red.png"].reject_reason == prefilter.JUNK
    assert by["dark.png"].passed
    assert summary.reject_reasons.get(prefilter.JUNK) == 1
    # A dropped image is a reject, and a reject holds no vector.
    assert set(summary.embeddings) == {"dark.png"}
    # drop_junk computes the junk score it needs, but leaves the score alone.
    assert "zero_shot_junk" not in by["dark.png"].score_breakdown
