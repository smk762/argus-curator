"""XMP sidecars on export (#16), the NudeNet content rating (#18), and named face groups (#17)."""

from __future__ import annotations

from pathlib import Path

import pytest
from argus_cortex.xmp import XmpData, read_xmp, sidecar_path, write_xmp

from argus_curator import scan_folder
from argus_curator.export import export_selection, write_source_sidecars, xmp_data
from argus_curator.models import (
    ExportRequest,
    FaceAttributes,
    FaceDetection,
    ImageResult,
    NsfwResult,
    ScanSummary,
    SelectRequest,
    rate_content,
)
from argus_curator.selection import decide_selection, nsfw_filter_problem


def _nsfw(**labels: float) -> NsfwResult:
    return NsfwResult(labels=labels, exposed_score=max(labels.values(), default=0.0))


def _image(path: Path, *, nsfw: NsfwResult | None = None, faces: list[FaceDetection] | None = None) -> ImageResult:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG not really")
    return ImageResult(
        rel_path=path.name,
        abs_path=str(path),
        score=1.0,
        passed=True,
        width=1000,
        height=500,
        nsfw=nsfw,
        faces=faces or [],
    )


def _face(cluster: str | None, bbox: list[float] = [100.0, 50.0, 200.0, 100.0]) -> FaceDetection:  # noqa: B006
    attrs = FaceAttributes(age_years=34.0, age_source="mivolo")
    return FaceDetection(bbox=bbox, det_score=0.9, cluster_id=cluster, attributes=attrs)


def _summary(tmp_path: Path, results: list[ImageResult]) -> ScanSummary:
    empty = tmp_path / "empty"
    empty.mkdir(exist_ok=True)
    return scan_folder(empty).model_copy(update={"results": results})


@pytest.mark.parametrize(
    ("labels", "rating"),
    [
        ({}, "safe"),
        ({"FACE_FEMALE": 0.9, "BELLY_EXPOSED": 0.8, "MALE_BREAST_EXPOSED": 0.7, "FEET_EXPOSED": 0.6}, "safe"),
        ({"BUTTOCKS_COVERED": 0.3}, "suggestive"),
        ({"FEMALE_BREAST_COVERED": 0.25, "FACE_FEMALE": 0.9}, "suggestive"),
        ({"FEMALE_BREAST_EXPOSED": 0.21, "BUTTOCKS_COVERED": 0.9}, "explicit"),
        ({"MALE_GENITALIA_EXPOSED": 0.5}, "explicit"),
    ],
)
def test_content_rating_from_detections(labels: dict[str, float], rating: str) -> None:
    assert rate_content(_nsfw(**labels)) == rating


def test_content_rating_is_none_when_unchecked_and_survives_a_cached_round_trip() -> None:
    assert rate_content(None) is None
    unchecked = ImageResult(rel_path="a", abs_path="/a")
    assert unchecked.content_rating is None
    rated = ImageResult(rel_path="a", abs_path="/a", nsfw=_nsfw(BUTTOCKS_COVERED=0.4))
    dumped = rated.model_dump()
    assert dumped["content_rating"] == "suggestive"  # served to the UI with the result
    assert ImageResult.model_validate(dumped).content_rating == "suggestive"


def test_content_ratings_filter(tmp_path: Path) -> None:
    results = [
        _image(tmp_path / "safe.png", nsfw=_nsfw()),
        _image(tmp_path / "lingerie.png", nsfw=_nsfw(BUTTOCKS_COVERED=0.5)),
        _image(tmp_path / "nude.png", nsfw=_nsfw(FEMALE_BREAST_EXPOSED=0.8)),
        _image(tmp_path / "unchecked.png"),
    ]
    req = SelectRequest(min_score=0.0, content_ratings=["safe", "suggestive"]).as_export_request()
    selected, reasons = decide_selection(results, req, 0.0)
    assert [r.rel_path for r in selected] == ["safe.png", "lingerie.png"]
    assert reasons["nude.png"] == "content-rating:explicit"
    assert reasons["unchecked.png"] == "nsfw-unchecked"
    summary = _summary(tmp_path, results)
    assert "content ratings need a scan that ran NudeNet" in (nsfw_filter_problem(req, summary) or "")
    with pytest.raises(ValueError):
        ExportRequest(dest=".", content_ratings=["nsfw"])


def test_xmp_data_names_only_named_clusters_and_never_writes_age() -> None:
    r = ImageResult(
        rel_path="a.png",
        abs_path="/a.png",
        width=1000,
        height=500,
        nsfw=_nsfw(),
        faces=[_face("face_1"), _face("face_2", [0.0, 0.0, 2000.0, 100.0]), _face(None)],
    )
    data = xmp_data(r, {"face_1": "Ada"})
    assert data is not None and data.content_rating == "safe" and data.persons == ["Ada"]
    first, wide, unclustered = data.regions.regions
    assert (first.x, first.y, first.w, first.h, first.name) == (0.2, 0.2, 0.2, 0.2, "Ada")  # centre, as fractions
    assert (wide.w, wide.name) == (1.0, None)  # clamped to the frame; "face_2" is not a name
    assert unclustered.name is None
    assert "34" not in data.model_dump_json()  # the MiVOLO estimate stays out
    # Two clusters may be one person; a name is listed once.
    assert xmp_data(r, {"face_1": "Ada", "face_2": "Ada"}).persons == ["Ada"]
    assert xmp_data(ImageResult(rel_path="b", abs_path="/b")) is None  # nothing known, nothing written
    assert xmp_data(ImageResult(rel_path="b", abs_path="/b", nsfw=_nsfw())).regions is None


def test_export_sidecar_starts_from_the_source_sidecar_and_leaves_it_alone(tmp_path: Path) -> None:
    src = tmp_path / "src"
    face = _image(src / "face.png", nsfw=_nsfw(BUTTOCKS_COVERED=0.4), faces=[_face("face_1")])
    plain = _image(src / "plain.png")  # no faces, never checked: nothing to record
    lightroom = _image(src / "lr.png", nsfw=_nsfw())
    # lens's captions and a photo app's stars, plus a property curator knows nothing about.
    write_xmp(face.abs_path, XmpData(keywords=["portrait"], description="Ada in a studio", rating=4))
    seed = sidecar_path(face.abs_path).read_text()
    foreign = '<rdf:Description rdf:about="" xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/" crs:Exposure2012="+0.50"/>'
    sidecar_path(face.abs_path).write_text(seed.replace("</rdf:RDF>", foreign + "</rdf:RDF>"))
    before = sidecar_path(face.abs_path).read_bytes()
    (src / "lr.xmp").write_text(  # Lightroom names a sidecar after the stem
        sidecar_path(face.abs_path).read_text().replace("Ada in a studio", "Lightroom caption")
    )

    dest = tmp_path / "out"
    req = ExportRequest(scan_id="s", dest=str(dest), min_score=0.0, write_xmp=True, face_names={"face_1": "Ada"})
    result = export_selection(_summary(tmp_path, [face, plain, lightroom]), req)

    assert result.xmp_written == 2
    out = read_xmp(dest / "face.png")
    assert out.keywords == ["portrait"] and out.description == "Ada in a studio" and out.rating == 4
    assert out.persons == ["Ada"] and out.content_rating == "suggestive"
    assert [region.name for region in out.regions.regions] == ["Ada"]
    assert 'crs:Exposure2012="+0.50"' in sidecar_path(dest / "face.png").read_text()
    assert sidecar_path(face.abs_path).read_bytes() == before  # the source's sidecar is not changed
    assert not sidecar_path(dest / "plain.png").exists()
    assert read_xmp(dest / "lr.png").description == "Lightroom caption"


def test_export_without_write_xmp_writes_no_sidecar(tmp_path: Path) -> None:
    face = _image(tmp_path / "src" / "face.png", nsfw=_nsfw(), faces=[_face("face_1")])
    dest = tmp_path / "out"
    result = export_selection(_summary(tmp_path, [face]), ExportRequest(scan_id="s", dest=str(dest), min_score=0.0))
    assert result.xmp_written == 0 and not list(dest.glob("*.xmp"))


def test_an_unreadable_source_sidecar_does_not_fail_the_export(tmp_path: Path) -> None:
    face = _image(tmp_path / "src" / "face.png", nsfw=_nsfw())
    sidecar_path(face.abs_path).write_text("<not xmp")
    dest = tmp_path / "out"
    result = export_selection(
        _summary(tmp_path, [face]), ExportRequest(scan_id="s", dest=str(dest), min_score=0.0, write_xmp=True)
    )
    assert result.copied == 1 and result.xmp_written == 0
    assert sidecar_path(face.abs_path).read_text() == "<not xmp"  # never clobbered


def test_write_source_sidecars_merges_beside_the_originals(tmp_path: Path) -> None:
    face = _image(tmp_path / "face.png", nsfw=_nsfw(), faces=[_face("face_1")])
    plain = _image(tmp_path / "plain.png")
    write_xmp(face.abs_path, XmpData(keywords=["portrait"]))
    assert write_source_sidecars([face, plain], {"face_1": "Ada"}) == 1
    merged = read_xmp(face.abs_path)
    assert merged.keywords == ["portrait"] and merged.persons == ["Ada"] and merged.content_rating == "safe"
    assert not sidecar_path(plain.abs_path).exists()


def test_face_names_are_validated() -> None:
    assert ExportRequest(dest=".", face_names={" face_1 ": " Ada "}).face_names == {"face_1": "Ada"}
    for bad in ({"face_1": " "}, {"": "Ada"}, {"face_1": "x" * 201}):
        with pytest.raises(ValueError, match="face_names"):
            ExportRequest(dest=".", face_names=bad)
