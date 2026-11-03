"""Export: carrying sidecars with their images (carry_sidecars) and the run_stats.json file."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from argus_cortex.xmp import XmpData, read_xmp, sidecar_path, write_xmp
from pydantic import ValidationError

from argus_curator import scan_folder
from argus_curator.export import RUN_STATS_NAME, _plan_dest_paths, export_selection
from argus_curator.models import RUN_STATS_MAX_BYTES, ExportRequest, ImageResult, NsfwResult, ScanSummary


def _image(path: Path, rel: str | None = None, nsfw: NsfwResult | None = None) -> ImageResult:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG not really " + path.name.encode())
    return ImageResult(
        rel_path=rel or path.name, abs_path=str(path), score=1.0, passed=True, width=100, height=100, nsfw=nsfw
    )


def _summary(tmp_path: Path, results: list[ImageResult]) -> ScanSummary:
    empty = tmp_path / "empty"
    empty.mkdir(exist_ok=True)
    return scan_folder(empty).model_copy(update={"results": results})


def _with_sidecars(img: ImageResult) -> dict[str, Path]:
    """Write a caption, both XMP namings and a JSON sidecar beside *img*."""
    src = Path(img.abs_path)
    write_xmp(src, XmpData(keywords=["lens"], description="from lens"))
    sides = {
        "txt": src.with_suffix(".txt"),
        "file_xmp": sidecar_path(src),
        "stem_xmp": src.with_suffix(".xmp"),
        "json": src.with_suffix(".json"),
    }
    sides["txt"].write_text("a caption")
    sides["stem_xmp"].write_text(sides["file_xmp"].read_text().replace("from lens", "from lightroom"))
    sides["json"].write_text('{"k": 1}')
    return sides


@pytest.mark.parametrize("mode", ["copy", "move", "symlink"])
def test_sidecars_travel_with_their_image(tmp_path: Path, mode: str) -> None:
    img = _image(tmp_path / "src" / "sub" / "photo.png", rel="sub/photo.png")
    sides = _with_sidecars(img)
    contents = {k: p.read_bytes() for k, p in sides.items()}
    dest = tmp_path / "out"

    result = export_selection(
        _summary(tmp_path, [img]),
        ExportRequest(scan_id="s", dest=str(dest), mode=mode, min_score=0.0, carry_sidecars=True),
    )

    assert result.copied == 1 and result.sidecars_carried == 4 and result.sidecars_failed == 0
    out = dest / "sub" / "photo.png"
    expected = {
        "txt": out.with_suffix(".txt"),
        "file_xmp": sidecar_path(out),
        "stem_xmp": out.with_suffix(".xmp"),
        "json": out.with_suffix(".json"),
    }
    for key, path in expected.items():
        assert path.read_bytes() == contents[key], key
        # Sidecars are copied even under symlink: they are a write path, the image is not.
        assert not path.is_symlink()
        # Under move nothing is left behind in the source folder.
        assert sides[key].exists() is (mode != "move")


def test_without_carry_sidecars_stay_put(tmp_path: Path) -> None:
    img = _image(tmp_path / "src" / "photo.png")
    sides = _with_sidecars(img)
    dest = tmp_path / "out"
    result = export_selection(
        _summary(tmp_path, [img]), ExportRequest(scan_id="s", dest=str(dest), mode="move", min_score=0.0)
    )
    assert result.copied == 1 and result.sidecars_carried == 0
    assert all(p.exists() for p in sides.values())
    assert not (dest / "photo.txt").exists()


def test_only_existing_sidecars_are_carried(tmp_path: Path) -> None:
    img = _image(tmp_path / "src" / "photo.png")
    Path(img.abs_path).with_suffix(".txt").write_text("caption only")
    dest = tmp_path / "out"
    result = export_selection(
        _summary(tmp_path, [img]),
        ExportRequest(scan_id="s", dest=str(dest), min_score=0.0, carry_sidecars=True),
    )
    assert result.sidecars_carried == 1
    assert (dest / "photo.txt").read_text() == "caption only"
    assert sorted(p.name for p in dest.iterdir() if p.name.startswith("photo")) == ["photo.png", "photo.txt"]


def test_flattened_sidecars_follow_the_de_collided_name(tmp_path: Path) -> None:
    src = tmp_path / "src"
    a = _image(src / "a" / "img.png", rel="a/img.png")
    b = _image(src / "b" / "img.png", rel="b/img.png")
    for img, caption in ((a, "caption a"), (b, "caption b")):
        Path(img.abs_path).with_suffix(".txt").write_text(caption)
        sidecar_path(img.abs_path).write_text(f"<xmp {caption}>")
    dest = tmp_path / "flat"

    result = export_selection(
        _summary(tmp_path, [a, b]),
        ExportRequest(
            scan_id="s", dest=str(dest), mode="move", min_score=0.0, preserve_structure=False, carry_sidecars=True
        ),
    )

    assert result.copied == 2 and result.sidecars_carried == 4
    assert not (dest / "img.txt").exists() and not (dest / "img.png.xmp").exists()
    for rel, caption in (("a/img.png", "caption a"), ("b/img.png", "caption b")):
        out = dest / result.exported_paths[rel]
        assert out.name != "img.png"
        assert out.with_suffix(".txt").read_text() == caption
        assert sidecar_path(out).read_text() == f"<xmp {caption}>"


def test_an_existing_destination_sidecar_is_replaced_like_the_image(tmp_path: Path) -> None:
    img = _image(tmp_path / "src" / "photo.png")
    Path(img.abs_path).with_suffix(".txt").write_text("new caption")
    dest = tmp_path / "out"
    dest.mkdir()
    (dest / "photo.png").write_bytes(b"old image")
    (dest / "photo.txt").write_text("old caption")
    result = export_selection(
        _summary(tmp_path, [img]),
        ExportRequest(scan_id="s", dest=str(dest), min_score=0.0, carry_sidecars=True),
    )
    assert result.sidecars_carried == 1
    assert (dest / "photo.png").read_bytes() == Path(img.abs_path).read_bytes()
    assert (dest / "photo.txt").read_text() == "new caption"


def test_a_failed_sidecar_does_not_undo_the_image(tmp_path: Path) -> None:
    img = _image(tmp_path / "src" / "photo.png")
    Path(img.abs_path).with_suffix(".txt").write_text("caption")
    Path(img.abs_path).with_suffix(".json").write_text("{}")
    dest = tmp_path / "out"
    # A directory where the caption should land: the copy fails, the image stays.
    (dest / "photo.txt").mkdir(parents=True)
    result = export_selection(
        _summary(tmp_path, [img]),
        ExportRequest(scan_id="s", dest=str(dest), min_score=0.0, carry_sidecars=True),
    )
    assert result.copied == 1 and result.exported_paths == {"photo.png": "photo.png"}
    assert result.sidecars_carried == 1 and result.sidecars_failed == 1
    assert (dest / "photo.json").exists()
    assert json.loads((dest / "manifest.jsonl").read_text())["rel_path"] == "photo.png"


@pytest.mark.parametrize("lightroom", [False, True])
def test_write_xmp_merges_onto_the_carried_sidecar_under_move(tmp_path: Path, lightroom: bool) -> None:
    img = _image(tmp_path / "src" / "photo.png", nsfw=NsfwResult(labels={"BUTTOCKS_COVERED": 0.4}, exposed_score=0.4))
    src = Path(img.abs_path)
    write_xmp(src, XmpData(keywords=["lens"], description="from lens", rating=4))
    seed = sidecar_path(src)
    if lightroom:  # only a Lightroom-named <stem>.xmp beside the source
        seed = seed.rename(src.with_suffix(".xmp"))
    dest = tmp_path / "out"

    result = export_selection(
        _summary(tmp_path, [img]),
        ExportRequest(scan_id="s", dest=str(dest), mode="move", min_score=0.0, carry_sidecars=True, write_xmp=True),
    )

    assert result.copied == 1 and result.sidecars_carried == 1 and result.xmp_written == 1
    out = read_xmp(dest / "photo.png")
    # The source sidecar's content plus curator's data, in one file.
    assert out.keywords == ["lens"] and out.description == "from lens" and out.rating == 4
    assert out.content_rating == "suggestive"
    # Nothing left behind in the source folder.
    assert not src.exists() and not seed.exists()
    assert not sidecar_path(src).exists() and not src.with_suffix(".xmp").exists()


def test_writing_a_caption_into_a_symlink_export_leaves_the_source_alone(tmp_path: Path) -> None:
    img = _image(tmp_path / "src" / "photo.png")
    sides = _with_sidecars(img)
    dest = tmp_path / "out"
    export_selection(
        _summary(tmp_path, [img]),
        ExportRequest(scan_id="s", dest=str(dest), mode="symlink", min_score=0.0, carry_sidecars=True),
    )
    assert (dest / "photo.png").is_symlink()
    # What argus-lens does after captioning the manifest.
    (dest / "photo.txt").write_text("new caption")
    (dest / "photo.json").write_text('{"k": 2}')
    assert sides["txt"].read_text() == "a caption"
    assert sides["json"].read_text() == '{"k": 1}'


def test_write_xmp_in_a_symlink_export_leaves_the_source_alone(tmp_path: Path) -> None:
    img = _image(tmp_path / "src" / "photo.png", nsfw=NsfwResult(labels={}, exposed_score=0.0))
    write_xmp(img.abs_path, XmpData(description="from lens"))
    before = sidecar_path(img.abs_path).read_bytes()
    dest = tmp_path / "out"
    result = export_selection(
        _summary(tmp_path, [img]),
        ExportRequest(scan_id="s", dest=str(dest), mode="symlink", min_score=0.0, carry_sidecars=True, write_xmp=True),
    )
    assert result.xmp_written == 1
    merged = sidecar_path(dest / "photo.png")
    assert not merged.is_symlink()
    assert read_xmp(dest / "photo.png").content_rating == "safe"
    assert read_xmp(dest / "photo.png").description == "from lens"
    assert sidecar_path(img.abs_path).read_bytes() == before


def test_run_stats_written_beside_the_manifest(tmp_path: Path) -> None:
    img = _image(tmp_path / "src" / "photo.png")
    dest = tmp_path / "out"
    stats = {"session": "abc", "kept": 1, "nested": {"ünïcode": [1, 2]}}
    result = export_selection(
        _summary(tmp_path, [img]), ExportRequest(scan_id="s", dest=str(dest), min_score=0.0, run_stats=stats)
    )
    path = dest / RUN_STATS_NAME
    assert result.run_stats_path == str(path)
    assert json.loads(path.read_text(encoding="utf-8")) == stats
    assert "\n  " in path.read_text(encoding="utf-8")  # pretty-printed


def test_run_stats_not_written_when_nothing_was_exported(tmp_path: Path) -> None:
    dest = tmp_path / "out"
    result = export_selection(
        _summary(tmp_path, []),
        ExportRequest(scan_id="s", dest=str(dest), min_score=0.0, write_manifest=False, run_stats={"a": 1}),
    )
    assert result.run_stats_path is None and not dest.exists()
    # With a manifest the folder exists anyway, so the stats go with it.
    result = export_selection(
        _summary(tmp_path, []), ExportRequest(scan_id="s", dest=str(dest), min_score=0.0, run_stats={"a": 1})
    )
    assert result.run_stats_path == str(dest / RUN_STATS_NAME)


def test_run_stats_absent_by_default(tmp_path: Path) -> None:
    img = _image(tmp_path / "src" / "photo.png")
    dest = tmp_path / "out"
    result = export_selection(_summary(tmp_path, [img]), ExportRequest(scan_id="s", dest=str(dest), min_score=0.0))
    assert result.run_stats_path is None and not (dest / RUN_STATS_NAME).exists()


def test_run_stats_size_is_capped() -> None:
    ExportRequest(dest="x", run_stats={"blob": "x" * (RUN_STATS_MAX_BYTES - 100)})
    with pytest.raises(ValidationError, match="run_stats"):
        ExportRequest(dest="x", run_stats={"blob": "x" * RUN_STATS_MAX_BYTES})


def test_flattened_carry_decollides_on_stem(tmp_path: Path) -> None:
    """a/IMG.jpg and b/IMG.png flatten to distinct basenames but would share IMG.txt."""
    sel = [ImageResult(rel_path=p, abs_path=str(tmp_path / p)) for p in ("a/IMG.jpg", "b/IMG.png")]
    assert _plan_dest_paths(sel, preserve_structure=False) == {"a/IMG.jpg": "IMG.jpg", "b/IMG.png": "IMG.png"}
    dests = _plan_dest_paths(sel, preserve_structure=False, by_stem=True)
    stems = {Path(d).stem.casefold() for d in dests.values()}
    assert len(stems) == 2


def test_run_stats_rejects_nan() -> None:
    with pytest.raises(ValidationError):
        ExportRequest(dest="/tmp/x", run_stats={"ratio": float("nan")})
