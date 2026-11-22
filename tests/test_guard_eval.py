"""scripts/guard_eval.py: the case file, interleaved cached runs and the report, with no live model."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest
from argus_cortex.moderation import GuardCheckFailed
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

CASES = guard_eval.load_case_file(ROOT / "eval" / "guard_cases.toml")
NO_SPLIT = guard_eval.CaseFile(CASES.buckets)  # every case on the tune side


# --- the case file ---------------------------------------------------------------------------------


def test_committed_probes_cover_every_bucket() -> None:
    cases = guard_eval.load_cases(ROOT / "eval" / "guard_probes.jsonl", None, CASES)
    assert {c.expect for c in cases} == set(CASES.buckets)
    assert len({c.id for c in cases}) == len(cases)
    assert all(c.text and c.image is None and c.kind == "text" for c in cases)


@pytest.mark.parametrize(
    ("policy", "flags"),
    [
        ("default", {"hard", "ambiguous", "adult", "other", "cinematic"}),
        ("adult", {"hard", "ambiguous", "other", "cinematic"}),
        ("adult-granular", {"hard", "ambiguous", "other", "cinematic"}),
        ("adult-cautious", {"hard", "ambiguous", "other", "cinematic"}),
        ("adult-cinematic", {"hard", "ambiguous", "other"}),
    ],
)
def test_committed_case_file_keeps_the_expectations_the_script_had(policy: str, flags: set[str]) -> None:
    wants = {bucket: CASES.want("any", bucket, policy) for bucket in CASES.buckets}
    assert {b for b, w in wants.items() if w == "flag"} == flags
    assert {b for b, w in wants.items() if w == "pass"} == set(CASES.buckets) - flags
    assert CASES.gates == {"hard"}


def _case_file(extra: str = "") -> object:
    import tomllib

    raw = tomllib.loads('[buckets.other]\n"*" = "flag"\n[buckets.safe]\n"*" = "pass"\n' + extra)
    return guard_eval.parse_case_file(raw)


def test_a_carve_out_is_a_case_file_entry_and_later_ones_win() -> None:
    casefile = _case_file(
        "[[override]]\ncase = 'img:other/drugs/*'\nwant = { adult-cinematic = 'either' }\nreason = 'r'\n"
        "[[override]]\ncase = 'img:other/drugs/b.jpg'\nwant = { '*' = 'pass' }\nreason = 'r'\n"
    )
    assert casefile.want("img:other/drugs/a.jpg", "other", "adult-cinematic") == "either"
    assert casefile.want("img:other/drugs/a.jpg", "other", "adult") == "flag"
    assert casefile.want("img:other/drugs/b.jpg", "other", "adult-cinematic") == "pass"
    assert casefile.want("img:other/gore/a.jpg", "other", "adult-cinematic") == "flag"


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ('[buckets.x]\nadult = "flag"\n', '"\\*" entry'),
        ('[buckets.x]\n"*" = "maybe"\n', "must be one of flag, pass, either"),
        ('[buckets.x]\n"*" = "flag"\nadlut = "pass"\n', "unknown policy 'adlut'"),
        ('[buckets.X]\n"*" = "flag"\n', "folder name"),
        ('[buckets.x]\n"*" = "flag"\ngate = "yes"\n', "true or false"),
        ("[[override]]\ncase = 'a'\nwant = { adult = 'pass' }\n", "exactly case, want and reason"),
        ("[[override]]\ncase = 'a'\nwant = { adult = 'pass' }\nreason = ' '\n", "needs a reason"),
        ("[split]\nheldout = 1.5\n", "split.heldout"),
        ("[split]\nheld = 0.2\n", "unknown keys held"),
        ("[extra]\n", "unknown sections extra"),
    ],
)
def test_a_bad_case_file_says_why(extra: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _case_file(extra)


def test_the_held_out_split_is_stable_and_near_its_share() -> None:
    casefile = _case_file("[split]\nheldout = 0.25\nsalt = 's'\ncases = ['pinned-*']\n")
    ids = [f"case-{n}" for n in range(2000)]
    held = [i for i in ids if casefile.split(i) == "heldout"]
    assert 0.2 < len(held) / len(ids) < 0.3
    assert held == [i for i in ids if casefile.split(i) == "heldout"]  # stable
    assert casefile.split("pinned-1") == "heldout"
    other_salt = _case_file("[split]\nheldout = 0.25\nsalt = 't'\n")
    assert held != [i for i in ids if other_salt.split(i) == "heldout"]


def test_a_run_takes_only_the_split_it_names() -> None:
    cases = [guard_eval.Case(id=f"c{n}", expect="safe", text="t") for n in range(40)]
    casefile = _case_file("[split]\nheldout = 0.5\nsalt = 's'\n")
    tune = guard_eval.select_split(cases, casefile, "tune")
    held = guard_eval.select_split(cases, casefile, "heldout")
    assert tune and held and {c.id for c in tune}.isdisjoint(c.id for c in held)
    assert len(tune) + len(held) == 40 == len(guard_eval.select_split(cases, casefile, "all"))


def test_a_probe_with_a_bucket_the_case_file_lacks_is_refused(tmp_path: Path) -> None:
    probes = tmp_path / "p.jsonl"
    probes.write_text('{"id": "a", "expect": "nope", "text": "t"}\n')
    with pytest.raises(SystemExit, match="bucket of the case file"):
        guard_eval.load_cases(probes, None, CASES)


# --- image cases -----------------------------------------------------------------------------------


def test_image_folders_become_cases(tmp_path: Path) -> None:
    (tmp_path / "adult").mkdir()
    (tmp_path / "adult" / "a.jpg").write_bytes(b"x")
    (tmp_path / "adult" / "notes.txt").write_text("skip")
    cases = guard_eval.load_cases(None, tmp_path, CASES)
    assert [(c.id, c.expect, c.group, c.kind) for c in cases] == [("img:adult/a.jpg", "adult", "adult", "image-bare")]


def test_image_cases_are_found_in_subfolders_and_grouped_by_them(tmp_path: Path) -> None:
    for rel in ("other/gore/a.jpg", "other/hate/a.jpg", "other/b.jpg"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(b"x")
    cases = guard_eval.load_cases(None, tmp_path, CASES)
    assert [c.id for c in cases] == ["img:other/b.jpg", "img:other/gore/a.jpg", "img:other/hate/a.jpg"]
    assert [c.group for c in cases] == ["other", "other/gore", "other/hate"]


def test_a_new_bucket_needs_only_the_case_file(tmp_path: Path) -> None:
    (tmp_path / "watermark").mkdir()
    (tmp_path / "watermark" / "a.jpg").write_bytes(b"x")
    casefile = _case_file('[buckets.watermark]\n"*" = "pass"\n')
    assert [c.expect for c in guard_eval.load_cases(None, tmp_path, casefile)] == ["watermark"]


def test_a_symlinked_subfolder_is_named_not_skipped_quietly(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "elsewhere" / "a.jpg").write_bytes(b"x")
    (tmp_path / "cases" / "other").mkdir(parents=True)
    (tmp_path / "cases" / "other" / "linked").symlink_to(tmp_path / "elsewhere")

    assert guard_eval.load_cases(None, tmp_path / "cases", CASES) == []
    assert "other/linked" in capsys.readouterr().err


def test_a_dangling_link_stays_a_case_so_the_run_reports_it(tmp_path: Path) -> None:
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "a.jpg").symlink_to(tmp_path / "gone.jpg")
    assert [c.id for c in guard_eval.load_cases(None, tmp_path, CASES)] == ["img:other/a.jpg"]


# --- candidates ------------------------------------------------------------------------------------


def test_built_in_policies_have_no_heading_gaps() -> None:
    for name in guard_eval.guard.POLICIES:
        text = guard_eval.guard.policy(name)[1]
        assert text is None or guard_eval.heading_gaps(text) is None, name


def test_a_candidate_with_a_gap_in_its_letters_is_refused(tmp_path: Path) -> None:
    wording = tmp_path / "w.txt"
    wording.write_text("### A. Sexual\n- x\n### C. Unethical\n- y\n### Safe\n- z\n")
    with pytest.raises(ValueError, match="rule headings run A, C; letter them A, B"):
        guard_eval.parse_candidate(f"x=adult:{wording}")


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


# --- runs ------------------------------------------------------------------------------------------


class _FakeClient:
    """Stands in for SingGuardClient: passes everything (or what ``verdicts`` says) and records each call."""

    made: list[dict] = []
    calls: list[dict] = []
    verdicts: dict[str, bool] = {}  # case text -> safe
    failures: list[Exception] = []

    def __init__(self, url: str, **kwargs: object) -> None:
        self.made.append(kwargs)
        self.thinking = kwargs["thinking"]
        self.policy = kwargs["policy"]

    def _answer(self, text: str) -> GuardResult:
        if self.failures:
            raise self.failures.pop(0)
        return GuardResult(safe=self.verdicts.get(text, True), thinking="fast", model="m")

    def check(self, data: bytes, text: str = "") -> GuardResult:
        self.calls.append({"how": "image", "data": data, "text": text, "mode": self.thinking, "policy": self.policy})
        return self._answer(text)

    def check_text(self, text: str, role: str = "user") -> GuardResult:
        self.calls.append({"how": "text", "text": text, "mode": self.thinking, "policy": self.policy})
        return self._answer(text)


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch) -> type[_FakeClient]:
    monkeypatch.setattr(_FakeClient, "made", [])
    monkeypatch.setattr(_FakeClient, "calls", [])
    monkeypatch.setattr(_FakeClient, "verdicts", {})
    monkeypatch.setattr(_FakeClient, "failures", [])
    monkeypatch.setattr(guard_eval, "SingGuardClient", _FakeClient)
    monkeypatch.setattr(guard_eval, "_sleep", lambda _s: None)
    return _FakeClient


def _run(cases: list, policies: list[str] = ["adult"], repeat: int = 1, **kwargs: object) -> list[dict]:  # noqa: B006
    kwargs.setdefault("casefile", NO_SPLIT)
    candidates = kwargs.pop("candidates", None)
    return guard_eval.run(
        "http://g", policies, cases, kwargs.pop("thinking", "fast"), repeat, 1.0, candidates, **kwargs
    )


_AT = guard_eval.guard.policy("adult")[1]


def test_text_probes_go_through_the_text_check_with_no_image(fake_client: type[_FakeClient]) -> None:
    rows = _run([guard_eval.Case(id="p", expect="other", text="a dismembered body")])
    assert fake_client.calls == [{"how": "text", "text": "a dismembered body", "mode": "fast", "policy": _AT}]
    assert rows[0]["kind"] == "text" and rows[0]["correct"] is False


def test_a_candidate_is_scored_as_its_parent_with_its_own_text(tmp_path: Path, fake_client: type[_FakeClient]) -> None:
    wording = tmp_path / "wording.txt"
    wording.write_text("candidate text")
    candidate = guard_eval.parse_candidate(f"adult-cinematic-wm=adult-cinematic:{wording}")
    cases = [guard_eval.Case(id="c", expect="cinematic", text="a heist still")]

    rows = _run(cases, candidates=[candidate])

    by = {r["policy"]: r for r in rows}
    assert by["adult"]["correct"] is False  # a pass is wrong under adult
    assert by["adult-cinematic-wm"]["correct"] is True  # and right under the parent
    assert by["adult-cinematic-wm"]["scored_as"] == "adult-cinematic"
    assert {c["policy"] for c in fake_client.calls} == {_AT, "candidate text"}


def test_a_candidate_named_like_a_policy_never_replaces_its_text(fake_client: type[_FakeClient]) -> None:
    # parse_candidate refuses this name; run() must hold on its own when handed one directly.
    clash = guard_eval.Candidate(name="adult", parent="adult", text="candidate text")
    cases = [guard_eval.Case(id="c", expect="cinematic", text="a heist still")]

    _run(cases, candidates=[clash])

    assert sorted(c["policy"] for c in fake_client.calls) == sorted([_AT, "candidate text"])


def test_each_pass_is_every_case_under_every_policy_shuffled(fake_client: type[_FakeClient]) -> None:
    cases = [guard_eval.Case(id=f"c{n}", expect="safe", text=f"t{n}") for n in range(6)]

    rows = _run(cases, ["adult", "default"], repeat=3, seed=7)

    for attempt in range(3):
        mine = [(r["case"], r["policy"]) for r in rows if r["run"] == attempt]
        assert sorted(mine) == sorted((c.id, p) for c in cases for p in ("adult", "default"))
    assert [r["run"] for r in rows] == sorted(r["run"] for r in rows)  # whole passes, one after another
    orders = {tuple((r["case"], r["policy"]) for r in rows if r["run"] == a) for a in range(3)}
    assert len(orders) > 1  # shuffled, not the same order each pass
    nested = [(c.id, p) for p in ("adult", "default") for c in cases]
    assert [(r["case"], r["policy"]) for r in rows if r["run"] == 0] != nested
    # The same seed replays the same order.
    replay = _run(cases, ["adult", "default"], repeat=3, seed=7)
    assert [(r["case"], r["policy"], r["run"]) for r in replay] == [(r["case"], r["policy"], r["run"]) for r in rows]


def test_the_cache_resumes_a_run_and_keeps_repeats_apart(tmp_path: Path, fake_client: type[_FakeClient]) -> None:
    cases = [guard_eval.Case(id="c", expect="safe", text="t")]
    cache = guard_eval.VerdictCache(tmp_path / "v.jsonl")
    _run(cases, repeat=2, cache=cache, digest=lambda: "d1")
    assert len(fake_client.calls) == 2  # two passes are two draws, never one cached twice
    cache.close()

    cache = guard_eval.VerdictCache(tmp_path / "v.jsonl")
    rows = _run(cases, repeat=3, cache=cache, digest=lambda: "d1")
    assert len(fake_client.calls) == 3  # only the new third pass ran
    assert [r["cached"] for r in rows] == [True, True, False]
    assert all(r["digest"] == "d1" for r in rows)


def test_a_different_model_digest_misses_the_cache(tmp_path: Path, fake_client: type[_FakeClient]) -> None:
    cases = [guard_eval.Case(id="c", expect="safe", text="t")]
    cache = guard_eval.VerdictCache(tmp_path / "v.jsonl")
    _run(cases, cache=cache, digest=lambda: "d1")
    rows = _run(cases, cache=cache, digest=lambda: "d2")
    assert len(fake_client.calls) == 2 and rows[0]["cached"] is False


def test_the_cache_key_names_every_input() -> None:
    base = {"image_sha": "i", "policy_sha": "p", "context": "c", "mode": "fast", "digest": "d", "run": 0}
    key = guard_eval.cache_key(**base)
    for field, other in [
        ("image_sha", None),
        ("policy_sha", "q"),
        ("context", "c2"),
        ("mode", "slow"),
        ("digest", "e"),
        ("run", 1),
    ]:
        assert guard_eval.cache_key(**{**base, field: other}) != key, field
    assert guard_eval.cache_key(**base) == key


def test_errors_are_never_cached(tmp_path: Path, fake_client: type[_FakeClient]) -> None:
    fake_client.failures.append(GuardCheckFailed("SingGuard returned HTTP 422"))
    cases = [guard_eval.Case(id="c", expect="safe", text="t")]
    cache = guard_eval.VerdictCache(tmp_path / "v.jsonl")
    assert "error" in _run(cases, cache=cache, digest=lambda: "d")[0]
    assert "error" not in _run(cases, cache=cache, digest=lambda: "d")[0]


def test_a_cache_line_cut_short_is_skipped(tmp_path: Path) -> None:
    path = tmp_path / "v.jsonl"
    path.write_text('{"key": "a", "safe": true}\n{"key": "b", "sa')
    assert set(guard_eval.VerdictCache(path).entries) == {"a"}


def test_a_verdict_appended_after_a_cut_short_line_starts_its_own(tmp_path: Path) -> None:
    path = tmp_path / "v.jsonl"
    path.write_text('{"key": "a", "safe": true}\n{"key": "b", "sa')
    cache = guard_eval.VerdictCache(path)
    cache.put("c", {"safe": False})
    cache.close()
    assert set(guard_eval.VerdictCache(path).entries) == {"a", "c"}


def test_a_cache_needs_a_digest(tmp_path: Path, fake_client: type[_FakeClient]) -> None:
    cache = guard_eval.VerdictCache(tmp_path / "v.jsonl")
    with pytest.raises(guard_eval.RunStopped, match="needs the model digest"):
        _run([guard_eval.Case(id="c", expect="safe", text="t")], cache=cache, digest=lambda: None)


def test_a_digest_that_changes_mid_run_stops_it(fake_client: type[_FakeClient]) -> None:
    digests = iter(["d1", "d1", "d2"])
    with pytest.raises(guard_eval.RunStopped, match="changed from d1 to d2 before pass 3"):
        _run([guard_eval.Case(id="c", expect="safe", text="t")], repeat=3, digest=lambda: next(digests))
    assert len(fake_client.calls) == 2


def test_an_unreadable_digest_mid_run_does_not_stop_it(fake_client: type[_FakeClient]) -> None:
    digests = iter(["d1", None])
    rows = _run([guard_eval.Case(id="c", expect="safe", text="t")], repeat=2, digest=lambda: next(digests))
    assert len(rows) == 2


def test_a_connection_failure_is_retried(fake_client: type[_FakeClient]) -> None:
    fake_client.failures += [GuardCheckFailed("SingGuard request failed: refused")] * 2
    rows = _run([guard_eval.Case(id="c", expect="safe", text="t")], retries=3)
    assert "error" not in rows[0] and len(fake_client.calls) == 3


def test_a_refusal_is_not_retried(fake_client: type[_FakeClient]) -> None:
    fake_client.failures.append(GuardCheckFailed("SingGuard returned HTTP 422"))
    rows = _run([guard_eval.Case(id="c", expect="safe", text="t")], retries=3)
    assert rows[0]["error"] == "SingGuard returned HTTP 422" and len(fake_client.calls) == 1


def test_a_server_that_stays_down_stops_the_run(fake_client: type[_FakeClient]) -> None:
    fake_client.failures += [GuardCheckFailed("SingGuard returned HTTP 503")] * 100
    cases = [guard_eval.Case(id=f"c{n}", expect="safe", text="t") for n in range(10)]
    with pytest.raises(guard_eval.RunStopped, match="3 checks in a row"):
        _run(cases, retries=1, max_errors=3)
    assert len(fake_client.calls) == 6  # three checks, each tried twice


def test_digest_sources() -> None:
    tags = {"models": [{"name": "singguard:4b", "model": "singguard:4b", "digest": "abc"}]}
    assert guard_eval.ollama_digest("http://o:11434/", "singguard:4b", lambda url: tags) == "abc"
    assert guard_eval.ollama_digest("http://o:11434", "other", lambda url: tags) is None
    latest = {"models": [{"name": "llama-guard3:latest", "model": "llama-guard3:latest", "digest": "def"}]}
    assert guard_eval.ollama_digest("http://o", "llama-guard3", lambda url: latest) == "def"
    assert guard_eval.ollama_digest("http://o", "llama-guard3:8b", lambda url: latest) is None
    registry = {"models": [{"name": "reg:5000/guard:latest", "model": "reg:5000/guard:latest", "digest": "r"}]}
    assert guard_eval.ollama_digest("http://o", "reg:5000/guard", lambda url: registry) == "r"
    health = {"model": "inclusionAI/SingGuard-4b", "revision": "4e08"}
    assert guard_eval.health_digest("http://g", lambda url: health) == "inclusionAI/SingGuard-4b@4e08"
    assert guard_eval.health_digest("http://g", lambda url: {"status": "ok"}) is None

    def down(url: str) -> object:
        raise OSError("refused")

    assert guard_eval.health_digest("http://g", down) is None
    assert guard_eval.digest_source("http://g", "pinned", None, None, down)() == "pinned"
    assert guard_eval.digest_source(None, None, None, None) is None


# --- context ---------------------------------------------------------------------------------------


def _aged_row(path: Path, rel: str, age: float = 30.0, **extra: object) -> ImageResult:
    attrs = FaceAttributes(age_years=age, age_source="mivolo")
    face = FaceDetection(bbox=[0, 0, 10, 10], det_score=0.9, primary=True, attributes=attrs)
    return ImageResult(rel_path=rel, abs_path=str(path), faces=[face], **extra)


def _case(path: Path, expect: str = "adult") -> object:
    return guard_eval.Case(id=f"img:{expect}/{path.name}", expect=expect, image=path, group=expect)


def test_scan_context_matches_by_path_then_by_unique_name(tmp_path: Path) -> None:
    sorted_copy = tmp_path / "cases" / "adult" / "a.jpg"
    exact = tmp_path / "cases" / "adult" / "b.jpg"
    rows = [_aged_row(tmp_path / "library" / "a.jpg", "a.jpg"), _aged_row(exact, "adult/b.jpg")]
    probe = guard_eval.Case(id="p", expect="safe", text="a probe")

    cases, unmatched = guard_eval.with_scan_context([_case(sorted_copy), _case(exact), probe], rows, frozenset({"age"}))

    assert unmatched == []
    assert all("Apparent age" in c.text and c.kind == "image" for c in cases[:2])
    assert cases[2].text == "a probe"


def test_scan_context_names_cases_with_no_row(tmp_path: Path) -> None:
    twin = "a.jpg"
    rows = [_aged_row(tmp_path / "x" / twin, "x/a.jpg"), _aged_row(tmp_path / "y" / twin, "y/a.jpg")]
    cases, unmatched = guard_eval.with_scan_context([_case(tmp_path / twin)], rows, frozenset({"age"}))
    assert unmatched == ["img:adult/a.jpg"]  # two rows have that name: a guess could be the wrong image
    assert cases[0].text == "" and cases[0].kind == "image-bare"


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


def test_the_folder_context_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "a.jpg"
    with pytest.raises(ValueError, match="hands the model the label"):
        guard_eval.with_scan_context([_case(path)], [_aged_row(path, "adult/a.jpg")], frozenset({"folder"}))
    with pytest.raises(SystemExit):
        guard_eval.main(["--url", "http://g", "--images", str(tmp_path), "--scan-json", "s", "--context", "age,folder"])
    assert "folder context kind is refused" in capsys.readouterr().err


def test_an_image_case_is_sent_as_the_scan_cropped_it(tmp_path: Path, fake_client: type[_FakeClient]) -> None:
    path = tmp_path / "a.png"
    Image.new("RGB", (40, 40)).save(path)
    row = _aged_row(path, "a.png", crop=CropBox(left=0, top=0, right=20, bottom=10))
    cases, _ = guard_eval.with_scan_context([_case(path)], [row], frozenset({"age"}))

    _run(cases)

    call = fake_client.calls[0]
    assert call["how"] == "image" and Image.open(io.BytesIO(call["data"])).size == (20, 10)
    assert "Apparent age" in call["text"]


def test_a_crop_that_does_not_fit_is_an_error_row_not_a_crash(tmp_path: Path, fake_client: type[_FakeClient]) -> None:
    path = tmp_path / "a.png"
    Image.new("RGB", (10, 10)).save(path)  # an export already cut to size, say
    row = _aged_row(path, "a.png", crop=CropBox(left=0, top=0, right=20, bottom=10))
    cases, _ = guard_eval.with_scan_context([_case(path)], [row], frozenset({"age"}))

    rows = _run(cases, repeat=2)

    assert [r["run"] for r in rows] == [0, 1]
    assert all(r["error"].startswith("crop ") and "leaves the 10x10 frame" in r["error"] for r in rows)
    assert fake_client.calls == []


@pytest.mark.parametrize(("thinking", "modes"), [("cascade", ["slow", "cascade"]), ("fast", ["fast", "fast"])])
def test_an_exposed_row_skips_the_fast_screen_as_a_scan_does(
    tmp_path: Path, fake_client: type[_FakeClient], thinking: str, modes: list[str]
) -> None:
    exposed, plain = tmp_path / "exposed.png", tmp_path / "plain.png"
    Image.new("RGB", (8, 8), (255, 0, 0)).save(exposed)
    Image.new("RGB", (8, 8), (0, 0, 255)).save(plain)
    nsfw = NsfwResult(detections=[], labels={"BUTTOCKS_EXPOSED": 0.3}, exposed_score=0.3)
    rows = [_aged_row(exposed, "exposed.png", nsfw=nsfw), _aged_row(plain, "plain.png")]
    cases, _ = guard_eval.with_scan_context([_case(exposed), _case(plain)], rows, frozenset({"age"}))

    _run(cases, thinking=thinking, seed=0)

    judged = {("exposed" if c["data"] == exposed.read_bytes() else "plain"): c["mode"] for c in fake_client.calls}
    assert [judged["exposed"], judged["plain"]] == modes


def test_an_image_gone_mid_run_is_an_error_row(
    tmp_path: Path, fake_client: type[_FakeClient], capsys: pytest.CaptureFixture[str]
) -> None:
    kept = tmp_path / "kept.png"
    Image.new("RGB", (8, 8)).save(kept)
    cases = [_case(tmp_path / "gone.png"), _case(kept)]

    rows = _run(cases, repeat=2)

    gone = [r for r in rows if r["case"] == "img:adult/gone.png"]
    assert len(gone) == 2 and all(r["error"].startswith("read ") for r in gone)
    assert all("error" not in r for r in rows if r["case"] == "img:adult/kept.png")
    assert "| 2 |" in guard_eval.summarise(rows)
    assert "img:adult/gone.png: read " in capsys.readouterr().err


# --- preflight -------------------------------------------------------------------------------------


def test_preflight_lists_exposed_safe_frames_and_anything_minor_exposed(tmp_path: Path) -> None:
    exposed = NsfwResult(detections=[], labels={"FEMALE_BREAST_EXPOSED": 0.4}, exposed_score=0.4)
    cases = [_case(tmp_path / n, b) for n, b in [("s.jpg", "safe"), ("m.jpg", "adult"), ("ok.jpg", "adult")]]
    rows = {
        "img:safe/s.jpg": _aged_row(tmp_path / "s.jpg", "s.jpg", nsfw=exposed),
        "img:adult/m.jpg": _aged_row(tmp_path / "m.jpg", "m.jpg", age=15.0, nsfw=exposed),
        "img:adult/ok.jpg": _aged_row(tmp_path / "ok.jpg", "ok.jpg", nsfw=exposed),
    }
    found = dict(guard_eval.preflight(cases, rows, CASES))
    assert set(found) == {"img:safe/s.jpg", "img:adult/m.jpg"}
    assert found["img:adult/m.jpg"].startswith("minor-exposed")


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


def test_a_run_stops_at_the_preflight_until_someone_has_looked(
    tmp_path: Path, fake_client: type[_FakeClient], capsys: pytest.CaptureFixture[str]
) -> None:
    images = tmp_path / "cases"
    (images / "safe").mkdir(parents=True)
    path = images / "safe" / "a.png"
    Image.new("RGB", (8, 8)).save(path)
    exposed = NsfwResult(detections=[], labels={"BUTTOCKS_EXPOSED": 0.3}, exposed_score=0.3)
    scan = _write_scan(tmp_path / "scan.json", [_aged_row(path, "safe/a.png", nsfw=exposed)])
    args = ["--url", "http://g", "--no-probes", "--images", str(images), "--policies", "adult", "--split", "all"]
    args += ["--scan-json", str(scan), "--context", "age"]

    assert guard_eval.main(args) == 2
    assert fake_client.calls == []
    assert "review img:safe/a.png: exposed reading in the safe bucket" in capsys.readouterr().err
    assert guard_eval.main([*args, "--preflight"]) == 0 and fake_client.calls == []
    assert guard_eval.main([*args, "--preflight-ok"]) == 0 and len(fake_client.calls) == 1


# --- reports ---------------------------------------------------------------------------------------


def _row(policy: str, case: str, run: int, safe: bool, expect: str = "hard", **extra: object) -> dict:
    scored_as = extra.pop("scored_as", policy)
    want = extra.pop("want", "flag" if expect == "hard" else "pass")
    kind = extra.pop("kind", "image")
    return {
        "policy": policy,
        "scored_as": scored_as,
        "case": case,
        "expect": expect,
        "group": expect,
        "kind": kind,
        "run": run,
        "want": want,
        "safe": safe,
        "letters": [] if safe else ["A"],
        "correct": None if want == "either" else (not safe) == (want == "flag"),
        **extra,
    }


def _passes(policy: str, case: str, safes: list[bool], **extra: object) -> list[dict]:
    return [_row(policy, case, n, safe, **extra) for n, safe in enumerate(safes)]


def test_a_split_frame_counts_as_flagged_and_is_listed() -> None:
    rows = _passes("adult", "h1", [False, True, False]) + _passes("adult", "s1", [True, False], expect="safe")
    frame = guard_eval.frames(rows)
    assert frame[("adult", "h1")].outcome == "split" and frame[("adult", "h1")].correct is True
    assert frame[("adult", "s1")].correct is False  # flagged once: it goes to review
    assert frame[("adult", "h1")].flips == (2, 2)
    assert [f.case for f in guard_eval.splitting(rows)] == ["h1", "s1"]
    text, _failed = guard_eval.report(rows)
    assert "Frames that keep splitting (2, counted as flagged)" in text
    assert "| 1/1 (1 split) | 0/1 (1 split) |" in text


def test_either_is_neither_right_nor_wrong() -> None:
    rows = _passes("adult", "o1", [True], expect="other", want="either")
    assert guard_eval.frames(rows)[("adult", "o1")].correct is None
    assert "| adult | image | - | 1 |" in guard_eval.summarise(rows, buckets=["other"])


def test_no_context_frames_are_counted_apart() -> None:
    rows = _passes("adult", "a", [False]) + _passes("adult", "b", [True], kind="image-bare")
    table = guard_eval.summarise(rows, buckets=["hard"])
    assert "| adult | image | 1/1 |" in table and "| adult | image-bare | 0/1 |" in table
    assert "different condition from production frames" in guard_eval.report(rows)[0]


def test_per_subfolder_noise_is_reported() -> None:
    rows = _passes("adult", "h1", [False, True, False]) + _passes("adult", "h2", [False, False, False])
    assert "| hard | 2/2 (1 split); flips 2/4 |" in guard_eval.subfolder_table(rows)


def test_changes_between_arms_are_weighed_against_each_frames_noise() -> None:
    cand = {"scored_as": "adult"}
    rows = (
        _passes("adult", "steady", [False, False, False])
        + _passes("cand", "steady", [True, True, True], **cand)
        + _passes("adult", "noisy", [False, True, False])
        + _passes("cand", "noisy", [True, True, True], **cand)
        + _passes("adult", "same", [False, False])
        + _passes("cand", "same", [False, False], **cand)
    )
    found = {c.case: c for c in guard_eval.changes(rows, "cand", "adult")}
    assert set(found) == {"steady", "noisy"}
    assert found["steady"].beyond_noise and not found["noisy"].beyond_noise
    assert found["steady"].direction == "worse"


def test_two_arms_that_agree_are_no_change_beyond_noise() -> None:
    def frame(arm: str) -> guard_eval.Frame:
        return guard_eval.Frame(arm, "c", "safe", "", "text", "pass", verdicts=[(1, False), (2, False)])

    assert not guard_eval.Change("c", frame("adult"), frame("cand")).beyond_noise


def test_a_candidate_that_passes_a_hard_frame_its_parent_flagged_fails() -> None:
    cand = {"scored_as": "adult"}
    rows = (
        _passes("adult", "h1", [False, True], kind="text")
        + _passes("cand", "h1", [True, True], kind="text", **cand)
        + _passes("adult", "h2", [False, False], kind="text")
        + _passes("cand", "h2", [False, True], kind="text", **cand)
    )
    assert [c.case for c in guard_eval.gate_failures(rows, "cand", "adult", frozenset({"hard"}))] == ["h1"]
    text, failed = guard_eval.report(rows)
    assert failed == ["cand"]
    assert "**FAIL**: cand passed 1 hard frames adult flagged: h1" in text
    assert "judged on text probes only, which doesn't prove image behaviour" in text
    assert guard_eval.TEXT_PROBE_NOTE in text


def test_a_candidate_that_holds_the_gate_passes() -> None:
    rows = _passes("adult", "h1", [False, False]) + _passes("cand", "h1", [False, True], scored_as="adult")
    text, failed = guard_eval.report(rows)
    assert failed == [] and "Gate held" in text and guard_eval.TEXT_PROBE_NOTE not in text


def test_baseline_compares_every_policy() -> None:
    rows = _passes("adult", "h1", [False]) + _passes("default", "h1", [True])
    assert guard_eval.comparisons(rows) == []
    assert guard_eval.comparisons(rows, "adult") == [("default", "adult", False)]


def test_main_report_exits_1_when_a_candidate_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rows = _passes("adult", "h1", [False]) + _passes("cand", "h1", [True], scored_as="adult")
    saved = tmp_path / "rows.jsonl"
    saved.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert guard_eval.main(["--report", str(saved)]) == 1
    assert "**FAIL**" in capsys.readouterr().out


# --- main ------------------------------------------------------------------------------------------


def test_a_run_that_dies_keeps_the_verdicts_so_far(
    tmp_path: Path, fake_client: type[_FakeClient], monkeypatch: pytest.MonkeyPatch
) -> None:
    images = tmp_path / "cases"
    (images / "adult").mkdir(parents=True)
    for stem in ("a", "b"):
        Image.new("RGB", (8, 8)).save(images / "adult" / f"{stem}.png")
    real = _FakeClient.check

    def dies_on_the_second(self: _FakeClient, data: bytes, text: str = "") -> GuardResult:
        if fake_client.calls:
            raise KeyboardInterrupt
        return real(self, data, text)

    monkeypatch.setattr(_FakeClient, "check", dies_on_the_second)
    out = tmp_path / "runs" / "guard.jsonl"
    args = ["--url", "http://g", "--no-probes", "--images", str(images), "--policies", "adult", "--split", "all"]

    with pytest.raises(KeyboardInterrupt):
        guard_eval.main([*args, "--out", str(out)])

    assert len(out.read_text().splitlines()) == 1


def test_main_adds_a_candidates_parent(
    tmp_path: Path, fake_client: type[_FakeClient], capsys: pytest.CaptureFixture[str]
) -> None:
    wording = tmp_path / "w.txt"
    wording.write_text("candidate text")
    args = ["--url", "http://g", "--policies", "adult", "--candidate", f"cand=adult-cinematic:{wording}"]
    guard_eval.main([*args, "--model-digest", "d"])
    assert {c["policy"] for c in fake_client.calls} == {
        _AT,
        guard_eval.guard.policy("adult-cinematic")[1],
        "candidate text",
    }
    captured = capsys.readouterr()
    assert "added adult-cinematic" in captured.err and "cand against adult-cinematic" in captured.out


def test_main_sends_scan_context_and_reports_unmatched(
    tmp_path: Path, fake_client: type[_FakeClient], capsys: pytest.CaptureFixture[str]
) -> None:
    images = tmp_path / "cases"
    (images / "adult").mkdir(parents=True)
    known, stray = images / "adult" / "known.png", images / "adult" / "stray.png"
    for path in (known, stray):
        Image.new("RGB", (8, 8)).save(path)
    scan = _write_scan(tmp_path / "scan.json", [_aged_row(known, "adult/known.png")])

    args = ["--url", "http://g", "--no-probes", "--images", str(images), "--policies", "adult", "--split", "all"]
    assert guard_eval.main([*args, "--scan-json", str(scan), "--context", "age"]) == 0

    assert sorted(bool(c["text"]) for c in fake_client.calls) == [False, True]
    captured = capsys.readouterr()
    assert "img:adult/stray.png" in captured.err and "| adult | image |" in captured.out
    assert "| adult | image-bare |" in captured.out
    assert "1 of 2 cases had no scan row" in captured.err


def test_main_resumes_from_a_cache(tmp_path: Path, fake_client: type[_FakeClient]) -> None:
    args = ["--url", "http://g", "--policies", "adult", "--model-digest", "d", "--cache", str(tmp_path / "v.jsonl")]
    assert guard_eval.main(args) == 0
    first = len(fake_client.calls)
    assert first > 0
    assert guard_eval.main(args) == 0
    assert len(fake_client.calls) == first


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--context", "age"], "--context needs --scan-json"),
        (["--scan-json", "s.json"], "needs context kinds"),
        (["--scan-json", "s.json", "--context", "age,nope"], "--context must name only"),
        (["--scan-json", "/does/not/exist.json", "--context", "age"], "--scan-json /does/not/exist.json"),
        (["--baseline", "nope"], "--baseline nope is not in the run"),
    ],
)
def test_main_refuses_what_it_cannot_do(
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


def test_main_needs_a_url_to_run_checks(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        guard_eval.main(["--policies", "adult"])
    assert "--url is required" in capsys.readouterr().err


# --- the prompt-cache check ------------------------------------------------------------------------


class _FakeOllama:
    """Answers like Ollama with a prompt cache: a repeat of the last probe is cached and scores a touch differently."""

    def __init__(self, evicts_after: int) -> None:
        self.evicts_after = evicts_after
        self.since_probe = 0
        self.warm = False
        self.posts: list[tuple[str, dict]] = []

    def __call__(self, url: str, body: dict) -> dict:
        self.posts.append((url, body))
        if body.get("keep_alive") == 0:
            self.warm = False
            return {"done_reason": "unload"}
        if "Name " in json.dumps(body):  # a filler
            self.since_probe += 1
            if self.since_probe >= self.evicts_after:
                self.warm = False
            return {"response": "carrots", "prompt_eval_count": 20, "prompt_eval_cached_count": 0}
        cached, self.warm, self.since_probe = self.warm, True, 0
        return {
            "response": "unsafe",
            "prompt_eval_count": 400,
            "prompt_eval_cached_count": 399 if cached else 0,
            "logprobs": [{"token": "unsafe", "logprob": -0.51 if cached else -0.49}],
        }


def test_the_prompt_cache_check_tells_cached_from_fresh_answers() -> None:
    fake = _FakeOllama(evicts_after=99)
    text, data = guard_eval.prompt_cache_check("http://o", "m", repeats=3, fillers=2, post=fake)
    assert [r["cached"] for r in data["back-to-back"]] == [0, 399, 399]
    assert all(r["cached"] == 399 for r in data["interleaved"])
    assert all(r["cached"] == 0 for r in data["cold"])
    assert "Back-to-back repeats reuse the prompt cache." in text
    assert "2 unrelated prompts between repeats did not evict it." in text
    assert "the answer depends on how much of the prompt was cached" in text
    assert sum(1 for _url, body in fake.posts if body.get("keep_alive") == 0) == 3


def test_the_prompt_cache_check_sees_eviction_and_sends_an_image_by_chat() -> None:
    fake = _FakeOllama(evicts_after=2)
    text, _data = guard_eval.prompt_cache_check("http://o", "m", repeats=2, image=b"png", fillers=2, post=fake)
    assert "2 unrelated prompts between repeats evicted it." in text
    probes = [(url, body) for url, body in fake.posts if "messages" in body and body["messages"][0]["content"]]
    assert probes and all(url.endswith("/api/chat") and body["messages"][1]["images"] for url, body in probes[:1])


def test_answers_that_follow_the_cache_state_are_called_out() -> None:
    def run(cached: int, lp: float) -> dict:
        return {"prompt": 1479, "cached": cached, "answer": str(lp), "logprobs": [lp]}

    data = {
        "model": "m",
        "image": True,
        "fillers": 8,
        "back-to-back": [run(415, -0.077), run(1478, -0.069), run(1478, -0.069)],
        "interleaved": [run(1478, -0.069)],
        "cold": [run(0, -0.056), run(0, -0.056)],
    }
    text = guard_eval.summarise_cache_check(data)
    assert "3 (1 at 0 cached, 1 at 1478 cached, 1 at 415 cached)" in text
    assert "depends on how much of the prompt was cached" in text
    data["cold"][1] = run(0, -0.01)
    assert "noise the cache doesn't explain" in guard_eval.summarise_cache_check(data)


def test_a_cache_check_with_no_evaluations_draws_no_conclusion() -> None:
    data = {"model": "m", "image": False, "fillers": 2, "back-to-back": [], "interleaved": [], "cold": []}
    text = guard_eval.summarise_cache_check(data)
    assert "No evaluations were recorded" in text
    assert "depends on how much of the prompt was cached" not in text
