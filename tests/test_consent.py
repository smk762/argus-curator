"""Real-person face sets bound to a consent record: read, bind, gate and record (#133).

Face detection is stubbed: each image's faces (embedding, age, size) come from
``PLAN``, and the real identity clustering, reference distances, set age,
binding, selection and export run on them.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from argus_cortex.consent import AgeCheck, ConsentRecord, Revocation, consent_matches
from argus_cortex.provenance import Provenance
from fastapi.testclient import TestClient
from pydantic import ValidationError

from argus_curator import faces as faces_mod
from argus_curator.attributes import RawAttributes
from argus_curator.consent import ConsentRefused, face_source_gate
from argus_curator.export import _write_pseudonymised_sidecar, export_selection
from argus_curator.models import (
    MANIFEST_VERSION,
    ConsentConfig,
    ExportRequest,
    FaceAttributes,
    FaceConfig,
    FaceDetection,
    FaceSourceFilter,
    ScanSummary,
)
from argus_curator.scanner import scan_folder
from argus_curator.selection import consent_set_problem, decide_selection
from argus_curator.server.app import create_app

DIM = 8


def _unit(*v: float) -> list[float]:
    a = np.zeros(DIM, dtype=np.float32)
    a[: len(v)] = v
    return list(a / np.linalg.norm(a))


SUBJECT = _unit(1.0)
OTHER = _unit(0.0, 1.0)

# rel_path -> faces, largest (primary) first: (embedding, MiVOLO age).
PLAN: dict[str, list[tuple[list[float], float]]] = {
    "ref.png": [(SUBJECT, 30.0)],
    **{f"a{i}.png": [(_unit(1.0, 0.02 * i), 30.0)] for i in range(6)},
    "bystander.png": [(SUBJECT, 30.0), (OTHER, 31.0)],
    "stranger.png": [(OTHER, 31.0)],
}
SUBJECT_IMAGES = {"ref.png", *(f"a{i}.png" for i in range(6))}


def _noise(seed: int) -> bytes:
    from io import BytesIO

    from PIL import Image

    arr = np.random.default_rng(seed).integers(0, 255, size=(512, 512, 3), dtype=np.uint8)
    buf = BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def faces_stub(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[tuple[list[float], float]]]:
    """Stub face detection and the reference embedding with ``PLAN`` (a copy a test may change)."""
    plan = {rel: list(faces) for rel, faces in PLAN.items()}

    def detect(results, items, cfg, progress=None, on_done=None, **kwargs):  # noqa: ANN001, ANN003, ANN202
        found = faces_mod.DetectedFaces([], [], [])
        for r in results:
            if not r.passed:
                continue
            for i, (emb, age) in enumerate(plan.get(r.rel_path, [])):
                side = 400.0 - 100.0 * i
                bbox = [0.0, 0.0, side, side]
                r.faces.append(
                    FaceDetection(
                        bbox=bbox,
                        det_score=0.9,
                        yaw=0.0,
                        pitch=0.0,
                        roll=0.0,
                        sharpness=400.0,
                        eye_open=0.3,
                        mouth_open=0.0,
                        attributes=FaceAttributes(age_years=age, age_source="mivolo"),
                    )
                )
                found.embeddings.append(np.asarray(emb, dtype=np.float32))
                found.meta.append((r.rel_path, i, bbox, 0.9, 1.0, side * side))
                found.raws.append(RawAttributes())
            r.face_count = len(r.faces)
        return found

    def reference(images, cfg=None, *, lease_wait_s=None):  # noqa: ANN001, ANN202
        refs = [rel for rel, data in _FILES.items() if data in images]
        embs = [np.asarray(plan[rel][0][0]) for rel in refs if plan.get(rel)]
        if not embs:
            raise ValueError("no reference image has a face")
        mean = np.vstack(embs).mean(axis=0)
        return [float(v) for v in mean / np.linalg.norm(mean)]

    monkeypatch.setattr(faces_mod, "detect_faces", detect)
    monkeypatch.setattr(faces_mod, "reference_embedding", reference)
    return plan


_FILES: dict[str, bytes] = {}


def _record(ref_sha: str, **overrides: object) -> ConsentRecord:
    values: dict[str, object] = {
        "release_sha256": "a" * 64,
        "subject": "Jo Example",
        "collected_by": "Example Studio",
        "signed_on": date(2025, 1, 1),
        "expires_on": None,
        "age_check": AgeCheck(checked_by="Example Studio", checked_on=date(2019, 1, 1)),
        "face": True,
        "voice": False,
        "uses": frozenset({"avatar"}),
        "reference_images": (ref_sha,),
        "revocable": True,
        "notice_days": 0,
        "on_revoke": "delete",
    }
    values.update(overrides)
    return ConsentRecord.model_validate(values)


def _write_record(folder: Path, record: ConsentRecord) -> None:
    (folder / "consent.json").write_text(record.model_dump_json(), encoding="utf-8")


@pytest.fixture
def consent_set(tmp_path: Path) -> Path:
    """A folder of ``PLAN``'s images with a consent record whose reference is ref.png."""
    root = tmp_path / "set"
    root.mkdir()
    _FILES.clear()
    for seed, rel in enumerate(PLAN, start=10):
        data = _noise(seed)
        (root / rel).write_bytes(data)
        _FILES[rel] = data
    _write_record(root, _record(hashlib.sha256(_FILES["ref.png"]).hexdigest()))
    return root


def _faces_cfg() -> FaceConfig:
    # The set holds a bystander and a stranger, so the subject has 8 of 10 faces.
    return FaceConfig(set_age_min_share=0.5, set_age_min_faces=5)


def _scan(root: Path, record: str | None = "consent.json") -> ScanSummary:
    return scan_folder(root, faces_cfg=_faces_cfg(), consent_cfg=ConsentConfig(record=record))


def _req(dest: Path, **fs: object) -> ExportRequest:
    return ExportRequest(dest=str(dest), min_score=0.0, face_source=FaceSourceFilter(consent=True, **fs))


def _rows(dest: Path) -> list[dict]:
    return [json.loads(line) for line in (dest / "manifest.jsonl").read_text().splitlines()]


# ---------------------------------------------------------------------------
# Read the record
# ---------------------------------------------------------------------------


def test_a_valid_record_makes_the_set_eligible_and_names_its_references(faces_stub: dict, consent_set: Path) -> None:
    summary = _scan(consent_set)
    check = summary.consent
    assert check is not None and check.eligible and check.reason is None
    assert check.references == ["ref.png"]
    assert check.release_sha256 == "a" * 64
    assert check.record_sha256 == hashlib.sha256((consent_set / "consent.json").read_bytes()).hexdigest()
    # A consent scan reads the set age the export gate needs.
    assert summary.faces_config.set_age and summary.set_age is not None


def test_a_missing_record_makes_the_set_ineligible(faces_stub: dict, consent_set: Path) -> None:
    (consent_set / "consent.json").unlink()
    check = _scan(consent_set).consent
    assert check is not None and not check.eligible and check.reason == "missing"
    assert str(consent_set) not in check.detail


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"signed_on": date(2020, 1, 1), "expires_on": date(2021, 1, 1)}, "expired"),
        (
            {"revocation": Revocation(requested_on=date(2025, 6, 1), effective_on=date(2025, 6, 1), by="subject")},
            "revoked",
        ),
        ({"uses": frozenset({"speech"})}, "use-not-covered"),
        ({"signed_on": date(2999, 1, 1), "age_check": AgeCheck(checked_by="x", checked_on=date(2999, 1, 1))}, None),
    ],
)
def test_an_expired_revoked_or_out_of_scope_record_makes_the_set_ineligible(
    faces_stub: dict, consent_set: Path, overrides: dict, reason: str | None
) -> None:
    ref = hashlib.sha256(_FILES["ref.png"]).hexdigest()
    _write_record(consent_set, _record(ref, **overrides))
    summary = _scan(consent_set)
    check = summary.consent
    assert check is not None and not check.eligible
    assert check.reason == (reason or "not-yet-signed")
    # An ineligible set binds nothing.
    assert all(r.consent is None for r in summary.results)


def test_an_invalid_record_is_ineligible_without_echoing_its_contents(faces_stub: dict, consent_set: Path) -> None:
    data = json.loads((consent_set / "consent.json").read_text())
    data["face"] = False  # a face release must opt in to the face
    data["unexpected"] = "Jo Example's secret"
    (consent_set / "consent.json").write_text(json.dumps(data))
    check = _scan(consent_set).consent
    assert check is not None and check.reason == "invalid"
    assert "Jo Example" not in check.detail and "secret" not in check.detail


def test_a_record_whose_reference_images_are_not_in_the_folder_is_ineligible(
    faces_stub: dict, consent_set: Path
) -> None:
    _write_record(consent_set, _record("b" * 64))
    check = _scan(consent_set).consent
    assert check is not None and check.reason == "no-reference-image" and check.missing_references == 1


def test_a_reference_image_without_a_face_is_ineligible(faces_stub: dict, consent_set: Path) -> None:
    faces_stub["ref.png"] = []
    check = _scan(consent_set).consent
    assert check is not None and check.reason == "no-reference-face"


# ---------------------------------------------------------------------------
# Bind
# ---------------------------------------------------------------------------


def test_matching_single_person_images_are_bound_with_their_distance(faces_stub: dict, consent_set: Path) -> None:
    by_rel = {r.rel_path: r for r in _scan(consent_set).results}
    for rel in SUBJECT_IMAGES:
        binding = by_rel[rel].consent
        assert binding is not None and binding.bound, rel
        assert binding.distance is not None and binding.distance < 0.05
    assert by_rel["ref.png"].consent.distance == pytest.approx(0.0, abs=1e-5)


def test_a_bystander_image_is_excluded(faces_stub: dict, consent_set: Path, tmp_path: Path) -> None:
    summary = _scan(consent_set)
    bystander = next(r for r in summary.results if r.rel_path == "bystander.png")
    # Its primary face is the subject's; the other person did not consent.
    assert bystander.consent is not None and not bystander.consent.bound
    assert bystander.consent.reason == "bystander" and bystander.consent.distance < 0.05
    result = export_selection(summary, _req(tmp_path / "out"))
    assert "bystander.png" not in result.selected_rel_paths


def test_a_mismatched_face_is_excluded(faces_stub: dict, consent_set: Path, tmp_path: Path) -> None:
    summary = _scan(consent_set)
    stranger = next(r for r in summary.results if r.rel_path == "stranger.png")
    assert stranger.consent is not None and stranger.consent.reason == "mismatch"
    assert stranger.consent.distance == pytest.approx(1.0, abs=1e-5)
    req = _req(tmp_path / "out")
    _chosen, reasons = decide_selection(
        summary.results, req, 0.0, set_age=summary.set_age, gate=face_source_gate(req, summary)
    )
    assert reasons["stranger.png"] == "face-source:consent-mismatch"
    assert reasons["bystander.png"] == "face-source:consent-bystander"


def test_the_binding_threshold_is_the_scans(faces_stub: dict, consent_set: Path) -> None:
    summary = scan_folder(
        consent_set, faces_cfg=_faces_cfg(), consent_cfg=ConsentConfig(record="consent.json", max_distance=0.0001)
    )
    by_rel = {r.rel_path: r.consent for r in summary.results}
    assert by_rel["ref.png"].bound
    assert by_rel["a5.png"].reason == "mismatch"


def test_a_consent_scan_refuses_a_second_reference(faces_stub: dict, consent_set: Path) -> None:
    with pytest.raises(ValueError, match="reference"):
        scan_folder(
            consent_set,
            faces_cfg=FaceConfig(reference_paths=["a0.png"]),
            consent_cfg=ConsentConfig(record="consent.json"),
        )


# ---------------------------------------------------------------------------
# Gate the export
# ---------------------------------------------------------------------------


def test_a_consented_export_keeps_the_bound_images_and_records_consent_on_each_row(
    faces_stub: dict, consent_set: Path, tmp_path: Path
) -> None:
    summary = _scan(consent_set)
    dest = tmp_path / "out"
    result = export_selection(summary, _req(dest))
    assert set(result.selected_rel_paths) == SUBJECT_IMAGES
    record_sha = hashlib.sha256((consent_set / "consent.json").read_bytes()).hexdigest()
    rows = _rows(dest)
    assert {row["rel_path"] for row in rows} == SUBJECT_IMAGES
    for row in rows:
        assert row["manifest_version"] == MANIFEST_VERSION == "2.8"
        assert row["consent_release_sha256"] == "a" * 64
        assert row["consent_record_sha256"] == record_sha
        prov = row["provenance"]
        assert prov["kind"] == "consented" and prov["release_sha256"] == "a" * 64
        # The row names the release, not the person, nor who collected the release.
        assert prov["subject"] == "consent:" + "a" * 64 and prov["license"]
        assert "Example Studio" not in prov["source"]


def test_an_image_sidecar_provenance_goes_on_its_row(faces_stub: dict, consent_set: Path, tmp_path: Path) -> None:
    own = {
        "kind": "consented",
        "source": "shot by Ann on 2025-02-01",
        "license": "release",
        "subject": "Jo Example",
        "release_sha256": "a" * 64,
    }
    (consent_set / "a0.json").write_text(json.dumps({"provenance": own}))
    dest = tmp_path / "out"
    export_selection(_scan(consent_set), _req(dest))
    by_rel = {row["rel_path"]: row for row in _rows(dest)}
    assert by_rel["a0.png"]["provenance"]["source"] == "shot by Ann on 2025-02-01"
    # Pseudonymised like the default: the sidecar's subject is not carried onto the row.
    assert by_rel["a0.png"]["provenance"]["subject"] == "consent:" + "a" * 64


def _all_text(dest: Path) -> dict[str, str]:
    files = (f for f in dest.rglob("*") if f.is_file() and f.suffix != ".png")
    return {str(f.relative_to(dest)): f.read_text(encoding="utf-8") for f in files}


@pytest.mark.parametrize("mode", ["copy", "move", "symlink"])
def test_consent_subject_name_appears_nowhere_in_the_export(
    faces_stub: dict, consent_set: Path, tmp_path: Path, mode: str
) -> None:
    """Must never: a consented export naming the consenting person (or who collected the release).

    The manifest, the CSV report, curator's XMP sidecars and the carried ``<stem>.json``
    sidecars name the subject only by cortex's subject ref; the source sidecars keep the name.
    """
    named = {
        "kind": "consented",
        "source": "a capture session",
        "license": "release",
        "subject": "Jo Example",
        "release_sha256": "a" * 64,
    }
    sidecar = {"producer": "argus-mask", "likeness_consent": True, "provenance": named}
    (consent_set / "a0.json").write_text(json.dumps(sidecar))
    source_before = (consent_set / "a0.json").read_bytes()
    summary = _scan(consent_set)
    dest = tmp_path / "out"
    req = _req(dest).model_copy(update={"mode": mode, "write_xmp": True, "carry_sidecars": True})
    result = export_selection(summary, req)
    assert "a0.png" in result.selected_rel_paths
    files = _all_text(dest)
    assert "manifest.jsonl" in files and "curation_report.csv" in files
    assert any(name.endswith(".xmp") for name in files)
    for name, text in files.items():
        assert "Jo Example" not in text, name
        assert "Example Studio" not in text, name
    # The carried copy is the sidecar with its provenance pseudonymised, and still matches the record.
    carried = json.loads(files["a0.json"])
    assert carried["producer"] == "argus-mask" and carried["likeness_consent"] is True
    prov = Provenance.model_validate(carried["provenance"])
    assert prov.subject == "consent:" + "a" * 64
    assert consent_matches(ConsentRecord.model_validate_json((consent_set / "consent.json").read_bytes()), prov)
    # The source sidecar is never changed, nor moved away, whatever the mode.
    assert (consent_set / "a0.json").read_bytes() == source_before
    assert not (dest / "a0.json").is_symlink()


def test_a_pseudonymised_sidecar_never_writes_through_a_planted_temp_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    side = tmp_path / "x.json"
    side.write_text(json.dumps({"producer": "argus-mask"}))
    victim = tmp_path / "victim.json"
    victim.write_text("keep me")
    target = tmp_path / "out" / "x.json"
    target.parent.mkdir()
    # The old temp name was predictable; a link there must not receive the sidecar.
    (target.parent / f".{target.name}.tmp").symlink_to(victim)
    monkeypatch.setattr("argus_curator.export.secrets.token_hex", lambda _n: "ab" * 8)
    planted = target.parent / f".{target.name}.{'ab' * 8}.tmp"
    planted.symlink_to(victim)

    with pytest.raises(FileExistsError):
        _write_pseudonymised_sidecar(side, target, None)

    assert victim.read_text() == "keep me"
    assert not target.exists()


def test_a_carried_sidecar_the_hold_does_not_cover_is_not_carried(tmp_path: Path) -> None:
    side = tmp_path / "x.json"
    side.write_text(json.dumps({"provenance": {"subject": "Jo Example"}}))
    target = tmp_path / "out" / "x.json"
    with pytest.raises(ValueError, match="does not cover"):
        _write_pseudonymised_sidecar(side, target, None)
    assert not target.exists()


def test_an_image_whose_provenance_names_another_release_is_excluded(
    faces_stub: dict, consent_set: Path, tmp_path: Path
) -> None:
    other = {"kind": "stock", "source": "https://example.org/x", "license": "CC0"}
    (consent_set / "a1.json").write_text(json.dumps({"provenance": other}))
    summary = _scan(consent_set)
    assert next(r for r in summary.results if r.rel_path == "a1.png").consent.reason == "provenance"
    assert "a1.png" not in export_selection(summary, _req(tmp_path / "out")).selected_rel_paths


def test_a_record_revoked_between_scan_and_export_blocks_the_export(
    faces_stub: dict, consent_set: Path, tmp_path: Path
) -> None:
    summary = _scan(consent_set)
    today = datetime.now(UTC).date()
    ref = hashlib.sha256(_FILES["ref.png"]).hexdigest()
    revoked = Revocation(requested_on=today, effective_on=today, by="the subject")
    _write_record(consent_set, _record(ref, revocation=revoked))
    dest = tmp_path / "out"
    with pytest.raises(ConsentRefused, match="revoked"):
        export_selection(summary, _req(dest))
    assert not dest.exists()


def test_a_record_deleted_or_replaced_since_the_scan_blocks_the_export(
    faces_stub: dict, consent_set: Path, tmp_path: Path
) -> None:
    summary = _scan(consent_set)
    ref = hashlib.sha256(_FILES["ref.png"]).hexdigest()
    _write_record(consent_set, _record(ref, release_sha256="c" * 64))
    with pytest.raises(ConsentRefused, match="another release"):
        export_selection(summary, _req(tmp_path / "out"))
    (consent_set / "consent.json").unlink()
    with pytest.raises(ConsentRefused, match="missing"):
        export_selection(summary, _req(tmp_path / "out"))


def test_an_expiry_passing_since_the_scan_blocks_the_export(
    faces_stub: dict, consent_set: Path, tmp_path: Path
) -> None:
    summary = _scan(consent_set)
    ref = hashlib.sha256(_FILES["ref.png"]).hexdigest()
    _write_record(consent_set, _record(ref, expires_on=datetime.now(UTC).date()))
    with pytest.raises(ConsentRefused, match="expired"):
        export_selection(summary, _req(tmp_path / "out"))


def test_an_under_18_estimate_on_a_matched_face_refuses_the_set(
    faces_stub: dict, consent_set: Path, tmp_path: Path
) -> None:
    faces_stub["a3.png"] = [(_unit(1.0, 0.06), 16.0)]
    summary = _scan(consent_set)
    assert summary.set_age is not None and summary.set_age.years >= 18  # the median is adult
    problem = consent_set_problem(summary)
    assert problem is not None and "under 18" in problem and "a3.png" in problem
    with pytest.raises(ConsentRefused, match="under 18"):
        export_selection(summary, _req(tmp_path / "out"))


def test_an_under_18_set_age_refuses_the_set(faces_stub: dict, consent_set: Path) -> None:
    for rel in [*SUBJECT_IMAGES, "bystander.png"]:
        faces_stub[rel] = [(emb, 16.0) for emb, _ in faces_stub[rel]]
    problem = consent_set_problem(_scan(consent_set))
    assert problem is not None and "under 18" in problem


def test_a_consented_set_needs_a_set_age(faces_stub: dict, consent_set: Path) -> None:
    summary = scan_folder(
        consent_set,
        faces_cfg=FaceConfig(set_age_min_share=0.99),
        consent_cfg=ConsentConfig(record="consent.json"),
    )
    assert summary.set_age is None
    problem = consent_set_problem(summary)
    assert problem is not None and "set age" in problem


def test_an_ineligible_set_is_refused_at_export(faces_stub: dict, consent_set: Path, tmp_path: Path) -> None:
    (consent_set / "consent.json").unlink()
    summary = _scan(consent_set)
    with pytest.raises(ConsentRefused, match="ineligible"):
        export_selection(summary, _req(tmp_path / "out"))
    no_record = scan_folder(consent_set, faces_cfg=_faces_cfg())
    assert "consent record" in consent_set_problem(no_record)


def test_an_export_where_nothing_qualifies_is_refused(faces_stub: dict, consent_set: Path, tmp_path: Path) -> None:
    summary = _scan(consent_set)
    dest = tmp_path / "out"
    with pytest.raises(ConsentRefused, match="no image qualifies"):
        export_selection(summary, _req(dest, max_identity_distance=0.0, min_face_px=10_000))
    assert not dest.exists()


def test_quarry_images_marked_not_consented_are_refused(faces_stub: dict, consent_set: Path, tmp_path: Path) -> None:
    quarry = {
        "producer": "argus-quarry",
        "likeness_consent": False,
        "provenance": {"kind": "stock", "source": "x", "license": "CC0"},
    }
    (consent_set / "a2.json").write_text(json.dumps(quarry))
    summary = _scan(consent_set)
    assert next(r for r in summary.results if r.rel_path == "a2.png").consent.reason == "not-consented"
    req = _req(tmp_path / "out")
    _chosen, reasons = decide_selection(
        summary.results, req, 0.0, set_age=summary.set_age, gate=face_source_gate(req, summary)
    )
    assert reasons["a2.png"] == "face-source:not-consented"


def test_a_quarry_mark_added_after_the_scan_is_read_at_export(
    faces_stub: dict, consent_set: Path, tmp_path: Path
) -> None:
    summary = _scan(consent_set)
    (consent_set / "a4.json").write_text(json.dumps({"producer": "argus-quarry", "likeness_consent": False}))
    result = export_selection(summary, _req(tmp_path / "out"))
    assert "a4.png" not in result.selected_rel_paths and "a0.png" in result.selected_rel_paths


def test_any_face_source_export_refuses_a_quarry_image(faces_stub: dict, consent_set: Path, tmp_path: Path) -> None:
    summary = scan_folder(consent_set, faces_cfg=FaceConfig(mivolo=True, reference_paths=["ref.png"]))
    (consent_set / "a0.json").write_text(json.dumps({"producer": "argus-quarry", "likeness_consent": False}))
    req = ExportRequest(dest=str(tmp_path / "out"), min_score=0.0, face_source=FaceSourceFilter())
    result = export_selection(summary, req)
    assert "a0.png" not in result.selected_rel_paths and "a1.png" in result.selected_rel_paths


def test_a_named_selection_with_face_source_refuses_a_quarry_image(
    faces_stub: dict, consent_set: Path, tmp_path: Path
) -> None:
    summary = scan_folder(consent_set, faces_cfg=FaceConfig(mivolo=True, reference_paths=["ref.png"]))
    (consent_set / "a0.json").write_text(json.dumps({"producer": "argus-quarry", "likeness_consent": False}))
    req = ExportRequest(
        dest=str(tmp_path / "out"), selection=["a0.png", "a1.png"], face_source=FaceSourceFilter(), min_score=0.0
    )
    assert export_selection(summary, req).selected_rel_paths == ["a1.png"]


def test_a_named_selection_never_skips_the_consent_gate(faces_stub: dict, consent_set: Path, tmp_path: Path) -> None:
    summary = _scan(consent_set)
    req = _req(tmp_path / "out").model_copy(update={"selection": ["a0.png"]})  # validation skipped
    with pytest.raises(ConsentRefused, match="named selection"):
        export_selection(summary, req)
    assert not (tmp_path / "out").exists()


def test_a_sidecar_outside_the_folder_is_refused(faces_stub: dict, consent_set: Path, tmp_path: Path) -> None:
    outside = tmp_path / "elsewhere.json"
    outside.write_text(json.dumps({"likeness_consent": True}))
    (consent_set / "a5.json").symlink_to(outside)
    summary = _scan(consent_set)
    assert next(r for r in summary.results if r.rel_path == "a5.png").consent.reason == "provenance"


def test_a_consented_set_is_never_chosen_without_the_gate(faces_stub: dict, consent_set: Path) -> None:
    summary = _scan(consent_set)
    with pytest.raises(ValueError, match="consent gate"):
        decide_selection(summary.results, _req(Path(".")), 0.0, set_age=summary.set_age)


def test_consent_requests_are_adult_only_and_by_filter() -> None:
    with pytest.raises(ValidationError, match="adult"):
        FaceSourceFilter(consent=True, require_adult=False)
    with pytest.raises(ValidationError, match="named selection"):
        ExportRequest(dest=".", selection=["a.png"], face_source=FaceSourceFilter(consent=True))
    with pytest.raises(ValidationError):
        ConsentConfig(record="  ")


# ---------------------------------------------------------------------------
# Server: containment, the gate over HTTP, and nothing kept past the scan
# ---------------------------------------------------------------------------


@pytest.fixture
def server(faces_stub: dict, consent_set: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr("argus_curator.server.app.faces_available", lambda: True)
    monkeypatch.setattr("argus_curator.server.app.mivolo_available", lambda: True)
    exports = tmp_path / "exports"
    exports.mkdir()
    app = create_app(cache_dir=str(tmp_path / "cache"), source_root=str(consent_set.parent), export_root=str(exports))
    return TestClient(app)


@pytest.fixture
def cli(faces_stub: dict, monkeypatch: pytest.MonkeyPatch):  # noqa: ANN201
    """``argus-curator scan`` with faces stubbed: unconstrained, as the CLI is."""
    pytest.importorskip("typer")
    from typer.testing import CliRunner

    from argus_curator.cli import app

    monkeypatch.setattr("argus_curator.attributes.mivolo_available", lambda: True)
    # One person throughout, so the default set-age share holds.
    faces_stub["bystander.png"] = []
    faces_stub["stranger.png"] = []
    runner = CliRunner()
    return lambda *args: runner.invoke(app, ["scan", *args])


_SCAN_BODY = {"folder": "set", "faces": {"set_age_min_share": 0.5}, "consent": {"record": "consent.json"}}


def test_server_scan_and_export_record_consent(server: TestClient, tmp_path: Path) -> None:
    resp = server.post("/scan/folder", json=_SCAN_BODY)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["consent"]["eligible"] is True
    # Never the record, and never the reference template.
    assert "Jo Example" not in resp.text and body["faces_config"]["reference_embedding"] is None
    scan_id = body["scan_id"]
    sel = server.post(f"/scan/{scan_id}/select", json={"min_score": 0.0, "face_source": {"consent": True}})
    assert sel.status_code == 200, sel.text
    assert {r["rel_path"] for r in sel.json()["rows"] if r["rank"]} == SUBJECT_IMAGES
    exp = server.post(
        "/export", json={"scan_id": scan_id, "dest": "out", "min_score": 0.0, "face_source": {"consent": True}}
    )
    assert exp.status_code == 200, exp.text
    assert set(exp.json()["selected_rel_paths"]) == SUBJECT_IMAGES


def test_server_refuses_a_revoked_record_at_select_and_export(
    server: TestClient, consent_set: Path, tmp_path: Path
) -> None:
    scan_id = server.post("/scan/folder", json=_SCAN_BODY).json()["scan_id"]
    today = datetime.now(UTC).date()
    ref = hashlib.sha256(_FILES["ref.png"]).hexdigest()
    _write_record(consent_set, _record(ref, revocation=Revocation(requested_on=today, effective_on=today, by="s")))
    body = {"min_score": 0.0, "face_source": {"consent": True}}
    sel = server.post(f"/scan/{scan_id}/select", json=body)
    assert sel.status_code == 400 and "revoked" in sel.json()["detail"]
    exp = server.post("/export", json={**body, "scan_id": scan_id, "dest": "out"})
    assert exp.status_code == 400 and "revoked" in exp.json()["detail"]
    assert not (tmp_path / "exports" / "out").exists()


def test_server_consent_record_traversal_blocked(server: TestClient, consent_set: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside.json"
    outside.write_text((consent_set / "consent.json").read_text())
    resp = server.post("/scan/folder", json={**_SCAN_BODY, "consent": {"record": "../outside.json"}})
    assert resp.status_code == 400
    (consent_set / "linked.json").symlink_to(outside)
    resp = server.post("/scan/folder", json={**_SCAN_BODY, "consent": {"record": "linked.json"}})
    assert resp.status_code == 400
    resp = server.post("/scan/folder", json={**_SCAN_BODY, "consent": {"record": str(outside)}})
    assert resp.status_code == 400


def test_server_export_consent_record_traversal_blocked(server: TestClient, consent_set: Path, tmp_path: Path) -> None:
    """The stored record path is data: one edited to leave the scan folder is refused at export."""
    scan_id = server.post("/scan/folder", json=_SCAN_BODY).json()["scan_id"]
    # A valid record outside the folder, so only containment can refuse it.
    (consent_set.parent / "outside.json").write_bytes((consent_set / "consent.json").read_bytes())
    path = tmp_path / "cache" / f"{scan_id}.json"
    data = json.loads(path.read_text())
    data["consent"]["record"] = "../outside.json"
    path.write_text(json.dumps(data))
    body = {"min_score": 0.0, "face_source": {"consent": True}}
    sel = server.post(f"/scan/{scan_id}/select", json=body)
    assert sel.status_code == 400 and "escapes" in sel.json()["detail"]
    exp = server.post("/export", json={**body, "scan_id": scan_id, "dest": "out"})
    assert exp.status_code == 400 and "escapes" in exp.json()["detail"]


def test_cli_exports_a_consented_set_with_a_record_outside_the_folder(cli, consent_set: Path, tmp_path: Path) -> None:
    record = tmp_path / "records" / "jo.json"
    record.parent.mkdir()
    record.write_bytes((consent_set / "consent.json").read_bytes())
    (consent_set / "consent.json").unlink()
    dest = tmp_path / "out"
    result = cli(
        str(consent_set), "--consent", str(record), "--face-source-consent", "--min-score", "0", "--copy-to", str(dest)
    )
    assert result.exit_code == 0, result.output
    assert "Consent:               eligible; 7 image(s) bound" in result.output
    rows = _rows(dest)
    assert {row["rel_path"] for row in rows} == SUBJECT_IMAGES
    assert all(row["consent_release_sha256"] == "a" * 64 for row in rows)


def test_cli_refuses_an_ineligible_set(cli, consent_set: Path, tmp_path: Path) -> None:
    (consent_set / "consent.json").unlink()
    dest = tmp_path / "out"
    result = cli(str(consent_set), "--consent", "consent.json", "--face-source-consent", "--copy-to", str(dest))
    assert result.exit_code == 2
    assert "INELIGIBLE (missing)" in result.output and "ineligible" in result.output
    assert not dest.exists()


def test_cli_consent_export_needs_a_record(cli, consent_set: Path) -> None:
    result = cli(str(consent_set), "--face-source-consent")
    assert result.exit_code == 2 and "--consent" in result.output and "Scanning" not in result.output


def test_server_refuses_a_consent_scan_with_its_own_reference(server: TestClient) -> None:
    body = {**_SCAN_BODY, "faces": {"reference_paths": ["a0.png"]}}
    resp = server.post("/scan/folder", json=body)
    assert resp.status_code == 400 and "reference" in resp.json()["detail"]


def test_nothing_about_consent_survives_a_purge(server: TestClient, consent_set: Path, tmp_path: Path) -> None:
    record_sha = hashlib.sha256((consent_set / "consent.json").read_bytes()).hexdigest()
    assert server.post("/scan/folder", json=_SCAN_BODY).status_code == 200
    cache = tmp_path / "cache"

    def mentions() -> list[Path]:
        return [
            p
            for p in cache.rglob("*")
            if p.is_file() and any(s in p.read_text(errors="ignore") for s in ("a" * 64, record_sha, '"consent"'))
        ]

    assert mentions()  # the scan holds its consent check...
    assert server.post("/scans/purge").status_code == 200
    assert mentions() == []  # ...and nothing about it outlives the scan
    assert [p for p in cache.rglob("*") if p.is_file()] == []


def test_consent_check_dates_are_the_scans(faces_stub: dict, consent_set: Path) -> None:
    before = datetime.now(UTC) - timedelta(seconds=1)
    check = _scan(consent_set).consent
    assert check is not None and before <= check.checked_at <= datetime.now(UTC) + timedelta(seconds=1)
