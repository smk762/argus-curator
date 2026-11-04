"""scripts/guard_eval.py: the committed probes load, and the table scores verdicts per policy."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

from argus_curator.models import (
    CropBox,
    FaceAttributes,
    FaceConfig,
    FaceDetection,
    GuardResult,
    ImageResult,
    NsfwResult,
    ScanConfig,
    ScanSummary,
    TargetProfile,
)

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


def test_image_cases_are_found_in_subfolders(tmp_path: Path) -> None:
    for rel in ("other/gore/a.jpg", "other/hate/a.jpg", "other/b.jpg"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(b"x")
    cases = guard_eval.load_cases(None, tmp_path)
    assert [c.id for c in cases] == ["img:other/b.jpg", "img:other/gore/a.jpg", "img:other/hate/a.jpg"]


def test_a_folder_named_like_an_image_is_not_a_case(tmp_path: Path) -> None:
    (tmp_path / "other" / "set.jpg").mkdir(parents=True)
    (tmp_path / "other" / "set.jpg" / "a.jpg").write_bytes(b"x")
    assert [c.id for c in guard_eval.load_cases(None, tmp_path)] == ["img:other/set.jpg/a.jpg"]


def test_a_symlinked_subfolder_is_named_not_skipped_quietly(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "elsewhere" / "a.jpg").write_bytes(b"x")
    (tmp_path / "cases" / "other").mkdir(parents=True)
    (tmp_path / "cases" / "other" / "linked").symlink_to(tmp_path / "elsewhere")

    assert guard_eval.load_cases(None, tmp_path / "cases") == []
    assert "other/linked" in capsys.readouterr().err


def test_a_dangling_link_stays_a_case_so_the_run_reports_it(tmp_path: Path) -> None:
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "a.jpg").symlink_to(tmp_path / "gone.jpg")
    assert [c.id for c in guard_eval.load_cases(None, tmp_path)] == ["img:other/a.jpg"]


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
    assert "| adult | 1/2 | - | 1/1 | - | - | - | A:1 | - | 1 | 1000 |" in table


def test_summary_counts_the_deciding_pass() -> None:
    rows = [
        {"policy": "adult", "expect": "other", "correct": True, "ms": 1, "thinking": "fast"},
        {"policy": "adult", "expect": "other", "correct": True, "ms": 1, "thinking": "fast"},
        {"policy": "adult", "expect": "hard", "correct": True, "ms": 1, "thinking": "slow"},
    ]
    assert "| fast:2 slow:1 |" in guard_eval.summarise(rows, ["adult"])


class _FakeClient:
    """Stands in for SingGuardClient: passes everything and records what it was sent."""

    made: list[dict] = []
    sent: list[tuple[bytes, str]] = []
    judged_in: list[str] = []

    def __init__(self, url: str, **kwargs: object) -> None:
        self.made.append(kwargs)
        self.thinking = kwargs["thinking"]

    def check(self, data: bytes, text: str = "") -> GuardResult:
        self.sent.append((data, text))
        self.judged_in.append(self.thinking)
        return GuardResult(safe=True, thinking="fast")


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch) -> type[_FakeClient]:
    monkeypatch.setattr(_FakeClient, "made", [])
    monkeypatch.setattr(_FakeClient, "sent", [])
    monkeypatch.setattr(_FakeClient, "judged_in", [])
    monkeypatch.setattr(guard_eval, "SingGuardClient", _FakeClient)
    return _FakeClient


def test_a_candidate_is_scored_as_its_parent_with_its_own_text(tmp_path: Path, fake_client: type[_FakeClient]) -> None:
    wording = tmp_path / "wording.txt"
    wording.write_text("candidate text")
    candidate = guard_eval.parse_candidate(f"adult-cinematic-wm=adult-cinematic:{wording}")
    cases = [guard_eval.Case(id="c", expect="cinematic", text="a heist still")]

    rows = guard_eval.run("http://g", ["adult"], cases, "fast", 1, 1.0, [candidate])

    by = {r["policy"]: r for r in rows}
    assert by["adult"]["correct"] is False  # a pass is wrong under adult
    assert by["adult-cinematic-wm"]["correct"] is True  # and right under the parent
    assert by["adult-cinematic-wm"]["thinking"] == "fast"
    assert fake_client.made[-1]["policy"] == "candidate text"


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ("nofile=adult", "NAME=PARENT:FILE"),
        ("x=nope:file.txt", "unknown parent"),
        ("adult=adult:file.txt", "built-in"),
        ("x=adult:/does/not/exist", "cannot read"),
    ],
)
def test_a_bad_candidate_says_why(raw: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        guard_eval.parse_candidate(raw)


def test_an_empty_candidate_is_refused_before_anything_runs(tmp_path: Path) -> None:
    wording = tmp_path / "wording.txt"
    wording.write_text(" \n")
    with pytest.raises(ValueError, match="is empty"):
        guard_eval.parse_candidate(f"x=adult:{wording}")


def test_a_candidate_file_can_be_under_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "wording.txt").write_text("candidate text")
    assert guard_eval.parse_candidate("x=adult:~/wording.txt").text == "candidate text"


def _aged_row(path: Path, rel: str, **extra: object) -> ImageResult:
    attrs = FaceAttributes(age_years=30.0, age_source="mivolo")
    face = FaceDetection(bbox=[0, 0, 10, 10], det_score=0.9, primary=True, attributes=attrs)
    return ImageResult(rel_path=rel, abs_path=str(path), faces=[face], **extra)


def _case(path: Path, expect: str = "adult") -> object:
    return guard_eval.Case(id=f"img:{expect}/{path.name}", expect=expect, image=path)


def test_scan_context_matches_by_path_then_by_unique_name(tmp_path: Path) -> None:
    sorted_copy = tmp_path / "cases" / "adult" / "a.jpg"
    exact = tmp_path / "cases" / "adult" / "b.jpg"
    rows = [_aged_row(tmp_path / "library" / "a.jpg", "a.jpg"), _aged_row(exact, "adult/b.jpg")]
    probe = guard_eval.Case(id="p", expect="safe", text="a probe")

    cases, unmatched = guard_eval.with_scan_context([_case(sorted_copy), _case(exact), probe], rows, frozenset({"age"}))

    assert unmatched == []
    assert all("Apparent age" in c.text for c in cases[:2])
    assert cases[2].text == "a probe"


def test_scan_context_names_cases_with_no_row(tmp_path: Path) -> None:
    twin = "a.jpg"
    rows = [_aged_row(tmp_path / "x" / twin, "x/a.jpg"), _aged_row(tmp_path / "y" / twin, "y/a.jpg")]
    cases, unmatched = guard_eval.with_scan_context([_case(tmp_path / twin)], rows, frozenset({"age"}))
    assert unmatched == ["img:adult/a.jpg"]  # two rows have that name: a guess could be the wrong image
    assert cases[0].text == ""


def test_a_name_two_images_share_matches_neither(tmp_path: Path) -> None:
    row = _aged_row(tmp_path / "library" / "a.jpg", "a.jpg")
    pair = [_case(tmp_path / "adult" / "a.jpg", "adult"), _case(tmp_path / "safe" / "a.jpg", "safe")]
    cases, unmatched = guard_eval.with_scan_context(pair, [row], frozenset({"age"}))
    assert unmatched == ["img:adult/a.jpg", "img:safe/a.jpg"]
    assert [c.text for c in cases] == ["", ""]


def test_a_path_match_survives_a_shared_name(tmp_path: Path) -> None:
    exact = tmp_path / "adult" / "a.jpg"
    pair = [_case(exact, "adult"), _case(tmp_path / "safe" / "a.jpg", "safe")]
    cases, unmatched = guard_eval.with_scan_context(pair, [_aged_row(exact, "adult/a.jpg")], frozenset({"age"}))
    assert unmatched == ["img:safe/a.jpg"]
    assert "Apparent age" in cases[0].text


@pytest.mark.parametrize("rel", ["adult/a.jpg", "Adult/a.jpg"])
def test_folder_context_does_not_give_the_label_away(tmp_path: Path, rel: str) -> None:
    path = tmp_path / "a.jpg"
    cases, _ = guard_eval.with_scan_context([_case(path)], [_aged_row(path, rel)], frozenset({"folder"}))
    assert cases[0].text == ""


def test_folder_context_keeps_the_rest_of_the_path(tmp_path: Path) -> None:
    path = tmp_path / "a.jpg"
    row = _aged_row(path, "adult/shoot-1/a.jpg")
    cases, _ = guard_eval.with_scan_context([_case(path)], [row], frozenset({"folder"}))
    assert "Folder: shoot-1" in cases[0].text and "adult" not in cases[0].text


def test_an_image_case_is_sent_as_the_scan_cropped_it(tmp_path: Path, fake_client: type[_FakeClient]) -> None:
    path = tmp_path / "a.png"
    Image.new("RGB", (40, 40)).save(path)
    row = _aged_row(path, "a.png", crop=CropBox(left=0, top=0, right=20, bottom=10))
    cases, _ = guard_eval.with_scan_context([_case(path)], [row], frozenset({"age"}))

    guard_eval.run("http://g", ["adult"], cases, "fast", 1, 1.0)

    data, text = fake_client.sent[0]
    assert Image.open(io.BytesIO(data)).size == (20, 10)
    assert "Apparent age" in text


def test_a_crop_that_does_not_fit_is_an_error_row_not_a_crash(tmp_path: Path, fake_client: type[_FakeClient]) -> None:
    path = tmp_path / "a.png"
    Image.new("RGB", (10, 10)).save(path)  # an export already cut to size, say
    row = _aged_row(path, "a.png", crop=CropBox(left=0, top=0, right=20, bottom=10))
    cases, _ = guard_eval.with_scan_context([_case(path)], [row], frozenset({"age"}))

    rows = guard_eval.run("http://g", ["adult"], cases, "fast", 2, 1.0)

    assert [r["run"] for r in rows] == [0, 1]
    assert all(r["error"].startswith("crop ") and "leaves the 10x10 frame" in r["error"] for r in rows)
    assert fake_client.sent == []


def test_a_crop_is_made_once_for_every_policy(
    tmp_path: Path, fake_client: type[_FakeClient], monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "a.png"
    Image.new("RGB", (40, 40)).save(path)
    calls: list[CropBox] = []
    real = guard_eval.crops.crop_bytes
    monkeypatch.setattr(guard_eval.crops, "crop_bytes", lambda data, box: calls.append(box) or real(data, box))
    case = guard_eval.Case(id="c", expect="adult", image=path, crop=CropBox(left=0, top=0, right=20, bottom=10))

    guard_eval.run("http://g", ["adult", "default"], [case], "fast", 1, 1.0)

    assert len(calls) == 1 and len(fake_client.sent) == 2


@pytest.mark.parametrize(("thinking", "modes"), [("cascade", ["slow", "cascade"]), ("fast", ["fast", "fast"])])
def test_an_exposed_row_skips_the_fast_screen_as_a_scan_does(
    tmp_path: Path, fake_client: type[_FakeClient], thinking: str, modes: list[str]
) -> None:
    exposed, plain = tmp_path / "exposed.png", tmp_path / "plain.png"
    for path in (exposed, plain):
        Image.new("RGB", (8, 8)).save(path)
    nsfw = NsfwResult(detections=[], labels={"BUTTOCKS_EXPOSED": 0.3}, exposed_score=0.3)
    rows = [_aged_row(exposed, "exposed.png", nsfw=nsfw), _aged_row(plain, "plain.png")]
    cases, _ = guard_eval.with_scan_context([_case(exposed), _case(plain)], rows, frozenset({"age"}))

    guard_eval.run("http://g", ["adult"], cases, thinking, 1, 1.0)

    assert fake_client.judged_in == modes


def test_an_image_gone_mid_run_is_an_error_row(
    tmp_path: Path, fake_client: type[_FakeClient], capsys: pytest.CaptureFixture[str]
) -> None:
    kept = tmp_path / "kept.png"
    Image.new("RGB", (8, 8)).save(kept)
    cases = [_case(tmp_path / "gone.png"), _case(kept)]

    rows = guard_eval.run("http://g", ["adult"], cases, "fast", 2, 1.0)

    assert ["error" in r for r in rows] == [True, True, False, False]
    assert rows[0]["error"].startswith("read ")
    assert "| 2 |" in guard_eval.summarise(rows, ["adult"])
    assert "img:adult/gone.png: read " in capsys.readouterr().err


def test_a_run_that_dies_keeps_the_verdicts_so_far(
    tmp_path: Path, fake_client: type[_FakeClient], monkeypatch: pytest.MonkeyPatch
) -> None:
    images = tmp_path / "cases"
    (images / "adult").mkdir(parents=True)
    for stem in ("a", "b"):
        Image.new("RGB", (8, 8)).save(images / "adult" / f"{stem}.png")
    real = _FakeClient.check

    def dies_on_the_second(self: _FakeClient, data: bytes, text: str = "") -> GuardResult:
        if fake_client.sent:
            raise KeyboardInterrupt
        return real(self, data, text)

    monkeypatch.setattr(_FakeClient, "check", dies_on_the_second)
    out = tmp_path / "runs" / "guard.jsonl"

    with pytest.raises(KeyboardInterrupt):
        guard_eval.main(
            ["--url", "http://g", "--no-probes", "--images", str(images), "--policies", "adult"] + ["--out", str(out)]
        )

    assert [json.loads(line)["case"] for line in out.read_text().splitlines()] == ["img:adult/a.png"]


def _write_scan(path: Path, rows: list[ImageResult]) -> Path:
    summary = ScanSummary(
        scan_id="s",
        folder="/d",
        target_profile=TargetProfile(),
        config=ScanConfig(),
        faces_config=FaceConfig(),
        total=len(rows),
        passed=len(rows),
        rejected=0,
        duplicates=0,
        similar_clusters=0,
        results=rows,
        returned=len(rows),
    )
    path.write_text(summary.model_dump_json(), encoding="utf-8")
    return path


def test_main_sends_scan_context_and_reports_unmatched(
    tmp_path: Path, fake_client: type[_FakeClient], capsys: pytest.CaptureFixture[str]
) -> None:
    images = tmp_path / "cases"
    (images / "adult").mkdir(parents=True)
    known, stray = images / "adult" / "known.png", images / "adult" / "stray.png"
    for path in (known, stray):
        Image.new("RGB", (8, 8)).save(path)
    scan = _write_scan(tmp_path / "scan.json", [_aged_row(known, "adult/known.png")])

    args = ["--url", "http://g", "--no-probes", "--images", str(images), "--policies", "adult"]
    assert guard_eval.main([*args, "--scan-json", str(scan), "--context", "age"]) == 0

    assert sorted(bool(text) for _data, text in fake_client.sent) == [False, True]
    captured = capsys.readouterr()
    assert "img:adult/stray.png" in captured.err and "| adult |" in captured.out
    assert "1 of 2 cases had no scan row" in captured.err


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--context", "age"], "--context needs --scan-json"),
        (["--scan-json", "s.json"], "needs context kinds"),
        (["--scan-json", "s.json", "--context", "age,nope"], "--context must name only"),
        (["--scan-json", "/does/not/exist.json", "--context", "age"], "--scan-json /does/not/exist.json"),
    ],
)
def test_main_refuses_a_context_it_cannot_send(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    extra: list[str],
    message: str,
) -> None:
    monkeypatch.delenv(guard_eval.guard.CONTEXT_ENV, raising=False)
    with pytest.raises(SystemExit):
        guard_eval.main(["--url", "http://g", "--images", str(tmp_path), *extra])
    assert message in capsys.readouterr().err


def test_main_refuses_a_scan_that_is_not_one(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    scan = tmp_path / "scan.json"
    scan.write_text("{}")
    with pytest.raises(SystemExit):
        guard_eval.main(["--url", "http://g", "--images", str(tmp_path), "--scan-json", str(scan), "--context", "age"])
    assert "--scan-json" in capsys.readouterr().err


def test_main_refuses_a_bad_context_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(guard_eval.guard.CONTEXT_ENV, "age,bogus")
    with pytest.raises(SystemExit):
        guard_eval.main(["--url", "http://g", "--images", str(tmp_path), "--scan-json", "s.json"])
    assert f"{guard_eval.guard.CONTEXT_ENV} must name only" in capsys.readouterr().err


def test_main_refuses_a_scan_with_no_images(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        guard_eval.main(["--url", "http://g", "--scan-json", "s.json", "--context", "age"])
    assert "--scan-json needs --images" in capsys.readouterr().err
