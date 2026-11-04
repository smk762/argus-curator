"""Export: carrying sidecars with their images (carry_sidecars) and the run_stats.json file."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from argus_cortex.xmp import XmpData, read_xmp, sidecar_path, write_xmp
from pydantic import ValidationError

from argus_curator import export as export_mod
from argus_curator import scan_folder
from argus_curator.export import MANIFEST_NAME, RUN_STATS_NAME, _plan_dest_paths, export_selection
from argus_curator.models import (
    RUN_STATS_MAX_BYTES,
    ExportRequest,
    ExportResult,
    ImageResult,
    NsfwResult,
    ScanSummary,
)


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


def _shared_pair(tmp_path: Path) -> tuple[ImageResult, ImageResult, Path]:
    """Two images in one folder that share photo.txt."""
    first = _image(tmp_path / "src" / "photo.jpg")
    second = _image(tmp_path / "src" / "photo.png")
    side = tmp_path / "src" / "photo.txt"
    side.write_text("shared caption")
    return first, second, side


def _move(tmp_path: Path, images: list[ImageResult], dest: Path) -> ExportResult:
    return export_selection(
        _summary(tmp_path, images),
        ExportRequest(
            scan_id="s",
            dest=str(dest),
            mode="move",
            min_score=0.0,
            keep_similar=True,
            preserve_structure=False,
            carry_sidecars=True,
        ),
    )


def test_move_gives_a_shared_sidecar_to_every_image(tmp_path: Path) -> None:
    first, second, side = _shared_pair(tmp_path)
    dest = tmp_path / "out"

    result = _move(tmp_path, [first, second], dest)

    assert result.copied == 2 and result.sidecars_carried == 2 and result.sidecars_failed == 0
    for exported in result.exported_paths.values():
        assert (dest / exported).with_suffix(".txt").read_text() == "shared caption"
    assert not side.exists()  # the last image took the original


def test_move_counts_down_a_sidecar_three_images_share(tmp_path: Path) -> None:
    first, second, side = _shared_pair(tmp_path)
    third = _image(tmp_path / "src" / "photo.webp")
    dest = tmp_path / "out"

    result = _move(tmp_path, [first, second, third], dest)

    assert result.sidecars_carried == 3 and not side.exists()
    for exported in result.exported_paths.values():
        assert (dest / exported).with_suffix(".txt").read_text() == "shared caption"


def test_move_leaves_a_shared_sidecar_for_an_image_that_stayed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    first, second, side = _shared_pair(tmp_path)
    real = export_mod._transfer_one

    def stuck(src: Path, dst: Path, mode: str) -> None:
        if src.name == "photo.jpg":
            raise OSError("stuck")
        real(src, dst, mode)

    monkeypatch.setattr(export_mod, "_transfer_one", stuck)
    dest = tmp_path / "out"

    result = _move(tmp_path, [first, second], dest)

    assert result.copied == 1 and result.skipped == 1
    assert Path(first.abs_path).exists() and side.read_text() == "shared caption"
    assert (dest / result.exported_paths["photo.png"]).with_suffix(".txt").read_text() == "shared caption"


def test_move_keeps_a_shared_sidecar_that_failed_to_reach_an_image(tmp_path: Path) -> None:
    first, second, side = _shared_pair(tmp_path)
    dest = tmp_path / "out"
    names = _plan_dest_paths([first, second], preserve_structure=False, by_stem=True)
    (dest / names["photo.jpg"]).with_suffix(".txt").mkdir(parents=True)

    result = _move(tmp_path, [first, second], dest)

    assert result.copied == 2 and result.sidecars_carried == 1 and result.sidecars_failed == 1
    assert side.read_text() == "shared caption"


@pytest.mark.parametrize("gone", ["photo.jpg", "photo.png"])
def test_move_does_not_keep_a_shared_sidecar_for_an_image_already_gone(tmp_path: Path, gone: str) -> None:
    first, second, side = _shared_pair(tmp_path)
    (tmp_path / "src" / gone).unlink()
    dest = tmp_path / "out"

    result = _move(tmp_path, [first, second], dest)

    assert result.copied == 1 and result.skipped == 1 and result.sidecars_carried == 1
    assert not side.exists()
    (exported,) = result.exported_paths.values()
    assert (dest / exported).with_suffix(".txt").read_text() == "shared caption"


@pytest.mark.parametrize("preserve", [False, True])
def test_run_stats_does_not_overwrite_a_carried_sidecar(tmp_path: Path, preserve: bool) -> None:
    img = _image(tmp_path / "src" / "run_stats.png")
    Path(img.abs_path).with_suffix(".json").write_text('{"mine": 1}')
    dest = tmp_path / "out"

    result = export_selection(
        _summary(tmp_path, [img]),
        ExportRequest(
            scan_id="s",
            dest=str(dest),
            min_score=0.0,
            preserve_structure=preserve,
            carry_sidecars=True,
            run_stats={"client": True},
        ),
    )

    exported = result.exported_paths["run_stats.png"]
    assert exported != "run_stats.png" and (dest / exported).is_file()
    assert json.loads((dest / exported).with_suffix(".json").read_text()) == {"mine": 1}
    assert json.loads((dest / RUN_STATS_NAME).read_text()) == {"client": True}
    assert [json.loads(line)["exported_path"] for line in (dest / MANIFEST_NAME).read_text().splitlines()] == [exported]


@pytest.mark.parametrize("name", ["run_stats.png", "RUN_STATS.png", "manifest.png", "curation_report.png"])
def test_reserved_stems_are_suffixed_only_when_sidecars_are_carried(tmp_path: Path, name: str) -> None:
    sel = [ImageResult(rel_path=name, abs_path=str(tmp_path / name))]
    assert _plan_dest_paths(sel, preserve_structure=False) == {name: name}
    assert _plan_dest_paths(sel, preserve_structure=True) == {name: name}
    assert _plan_dest_paths(sel, preserve_structure=False, by_stem=True)[name] != name
    assert _plan_dest_paths(sel, preserve_structure=True, by_stem=True)[name] != name


def test_a_reserved_stem_in_a_subfolder_keeps_its_path(tmp_path: Path) -> None:
    sel = [ImageResult(rel_path="sub/run_stats.png", abs_path=str(tmp_path / "x"))]
    assert _plan_dest_paths(sel, preserve_structure=True, by_stem=True) == {"sub/run_stats.png": "sub/run_stats.png"}


def test_moved_names_what_left_even_when_the_export_then_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    img = _image(tmp_path / "src" / "a.png")
    missing = ImageResult(rel_path="gone.png", abs_path=str(tmp_path / "src" / "gone.png"), score=1.0, passed=True)

    def disk_full(*_args: object, **_kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(export_mod, "write_report", disk_full)
    moved: list[str] = []
    with pytest.raises(OSError, match="disk full"):
        export_selection(
            _summary(tmp_path, [img, missing]),
            ExportRequest(scan_id="s", dest=str(tmp_path / "out"), mode="move", min_score=0.0, keep_similar=True),
            moved=moved,
        )
    assert moved == ["a.png"]


def test_moved_stays_empty_for_a_copy(tmp_path: Path) -> None:
    moved: list[str] = []
    export_selection(
        _summary(tmp_path, [_image(tmp_path / "src" / "a.png")]),
        ExportRequest(scan_id="s", dest=str(tmp_path / "out"), min_score=0.0),
        moved=moved,
    )
    assert moved == []
