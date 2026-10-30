"""scripts/age_eval.py: the face-only against face+body table, without models."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from argus_curator import faces
from argus_curator.models import FaceAttributes

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("age_eval", ROOT / "scripts" / "age_eval.py")
age_eval = importlib.util.module_from_spec(_spec)
sys.modules["age_eval"] = age_eval
_spec.loader.exec_module(age_eval)  # type: ignore[union-attr]


def test_table_compares_modes_and_counts_minors_read_as_adults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("a.png", "b.png", "c.png", "noface.png"):
        (tmp_path / name).write_bytes(b"png")
    (tmp_path / "labels.csv").write_text("path,age\na.png,30\nb.png,16\nc.png,40\nnoface.png,25\n")
    # face-only: 34, 19 (a minor read as an adult), 40; face+body: 31 (with body), 17, 40.
    readings = {
        False: {"a.png": (34.0, False), "b.png": (19.0, False), "c.png": (40.0, False)},
        True: {"a.png": (31.0, True), "b.png": (17.0, True), "c.png": (40.0, False)},
    }

    def detect(results, items, cfg, **kwargs) -> None:  # noqa: ANN001, ANN003
        r = results[0]
        if r.rel_path in readings[cfg.mivolo_body]:
            years, body = readings[cfg.mivolo_body][r.rel_path]
            r.primary_face_attributes = FaceAttributes(age_years=years, age_source="mivolo", age_body=body)

    monkeypatch.setattr(faces, "detect_faces", detect)
    out = tmp_path / "runs" / "ages.jsonl"
    assert age_eval.main([str(tmp_path / "labels.csv"), "--out", str(out)]) == 0
    assert len(out.read_text().splitlines()) == 8  # every case, in both modes

    cases = age_eval.load_cases(tmp_path / "labels.csv", tmp_path)
    table = age_eval.summarise(
        age_eval.read_ages(cases, age_eval.FaceConfig(mivolo=True), "face")
        + age_eval.read_ages(cases, age_eval.FaceConfig(mivolo=True, mivolo_body=True), "face+body"),
        ["face", "face+body"],
    ).splitlines()
    face, body = table[1].split(), table[2].split()
    assert face == ["face", "4", "3", "0", "2.33", "3.00", "100%", "1/1"]
    assert body == ["face+body", "4", "3", "2", "0.67", "1.00", "100%", "0/1"]


def test_labels_need_a_path_and_age_header(tmp_path: Path) -> None:
    (tmp_path / "labels.csv").write_text("file,years\na.png,30\n")
    with pytest.raises(SystemExit, match="path and age"):
        age_eval.load_cases(tmp_path / "labels.csv", tmp_path)
