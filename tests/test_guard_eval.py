"""scripts/guard_eval.py: the committed probes load, and the table scores verdicts per policy."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("guard_eval", ROOT / "scripts" / "guard_eval.py")
guard_eval = importlib.util.module_from_spec(_spec)
sys.modules["guard_eval"] = guard_eval  # dataclasses look the module up
_spec.loader.exec_module(guard_eval)


def test_committed_probes_cover_every_expectation() -> None:
    cases = guard_eval.load_cases(ROOT / "eval" / "guard_probes.jsonl", None)
    assert {c.expect for c in cases} == set(guard_eval.EXPECTS)
    assert len({c.id for c in cases}) == len(cases)
    assert all(c.text and c.image is None for c in cases)


def test_image_folders_become_cases(tmp_path: Path) -> None:
    (tmp_path / "adult").mkdir()
    (tmp_path / "adult" / "a.jpg").write_bytes(b"x")
    (tmp_path / "adult" / "notes.txt").write_text("skip")
    cases = guard_eval.load_cases(None, tmp_path)
    assert [(c.id, c.expect) for c in cases] == [("img:adult/a.jpg", "adult")]


def test_adult_content_is_correct_to_flag_only_under_default() -> None:
    assert guard_eval.should_flag("default", "adult") is True
    assert guard_eval.should_flag("adult-granular", "adult") is False
    assert guard_eval.should_flag("adult-cinematic", "adult") is False
    assert all(guard_eval.should_flag(p, "hard") for p in ("default", "adult", "adult-cautious"))


def test_cinematic_crime_passes_only_under_adult_cinematic() -> None:
    assert guard_eval.should_flag("default", "cinematic") is True
    assert guard_eval.should_flag("adult", "cinematic") is True
    assert guard_eval.should_flag("adult-granular", "cinematic") is True
    assert guard_eval.should_flag("adult-cinematic", "cinematic") is False
    assert guard_eval.should_flag("adult-cinematic", "other") is True


def test_summary_counts_correct_runs_and_hard_letters() -> None:
    rows = [
        {"policy": "adult", "expect": "hard", "safe": False, "letters": ["A"], "correct": True, "ms": 900},
        {"policy": "adult", "expect": "hard", "safe": True, "letters": [], "correct": False, "ms": 1100},
        {"policy": "adult", "expect": "adult", "safe": True, "letters": [], "correct": True, "ms": 1000},
        {"policy": "adult", "expect": "safe", "error": "SingGuard returned no verdict"},
    ]
    table = guard_eval.summarise(rows, ["adult"])
    assert "| adult | 1/2 | - | 1/1 | - | - | - | A:1 | 1 | 1000 |" in table
