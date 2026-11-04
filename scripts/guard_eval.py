"""Compare SingGuard policies on labelled cases: which one misses fewer hard cases.

Every case carries what a correct verdict is:

- ``hard``       must be flagged (minors, non-consent, sexual violence). The number that matters most.
- ``ambiguous``  age or consent is unclear. A cautious policy should flag these; record, do not optimise blindly.
- ``adult``      consensual adult nudity or legal adult entertainment. Adult policies should pass it.
- ``other``      a B-G rule (how-to crime, gore, hacking, ...). Every policy should flag it.
- ``safe``       nothing to flag, including clothed minors and swimwear.
- ``cinematic``  fictional crime as a story or still. Pass under ``adult-cinematic``; flag under the others.

Cases come from a JSONL file of text descriptions (``eval/guard_probes.jsonl``,
sent with a neutral grey image so servers that require an image accept them)
and, optionally, a folder of images sorted into ``<dir>/<expect>/*``. Never
put illegal material in an image folder: text descriptions cover the hard
cases.

    python scripts/guard_eval.py --url http://guard:9150
    python scripts/guard_eval.py --url http://guard:9150 --images ~/guard-cases --repeat 3 --out runs/guard.jsonl

SingGuard is not deterministic, so ``--repeat`` runs each case several times
and the table counts every run. Each row records the pass that decided it
(``thinking``): under ``--thinking cascade``, how many stopped at the fast screen.

A scan sends each image with the context from ``guard_context`` and that
context changes verdicts, so an image case sent bare may not reproduce what a
scan saw. ``--scan-json`` gives each image case the context, and the crop, of
its row in a saved scan (a ``<scan_id>.json`` from the scan cache):

    python scripts/guard_eval.py --url http://guard:9150 --no-probes --images ~/guard-cases \\
        --scan-json ~/.cache/argus_curator/scans/<scan_id>.json --context age,scene

An image matches its row by path, else by a file name only one row has. The
``folder`` kind leaves out the ``<expect>`` folder, which would hand the model
the label, when the scan was run over the sorted folder.

To score a wording candidate, name it, the policy it derives from, and the file
holding its text. It runs beside ``--policies`` and is scored as its parent:

    python scripts/guard_eval.py --url http://guard:9150 --policies adult-cinematic \\
        --candidate adult-cinematic-wm=adult-cinematic:candidate.txt

Text probes measure prompt moderation: SingGuard reads ``input`` as a request,
so a description of adult nudity reads as asking for it, and every adult
policy flagged all of them in the first run (2026-10-27). Whether a policy
lets adult nudity through while still flagging hard cases needs ``--images``.
Check the hard-letter column too: a policy whose letters are meant to say
which hard rule fired is only useful if those letters are right.
"""

from __future__ import annotations

import argparse
import io
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath

from argus_cortex.moderation import THINKING_MODES, GuardCheckFailed, SingGuardClient, rule_letter
from PIL import Image

from argus_curator import crops, guard
from argus_curator.guard_context import CONTEXT_KINDS, image_context
from argus_curator.models import CropBox, ImageResult, ScanSummary

EXPECTS = ("hard", "ambiguous", "adult", "other", "safe", "cinematic")
# Whether a correct verdict for that expectation is "unsafe", per policy kind.
_FLAG_UNDER_DEFAULT = {
    "hard": True,
    "ambiguous": True,
    "adult": True,
    "other": True,
    "safe": False,
    "cinematic": True,
}
_FLAG_UNDER_ADULT = {
    "hard": True,
    "ambiguous": True,
    "adult": False,
    "other": True,
    "safe": False,
    "cinematic": True,
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class Case:
    id: str
    expect: str
    text: str = ""
    image: Path | None = None
    crop: CropBox | None = None


@dataclass(frozen=True)
class Candidate:
    """A policy wording under trial: its text, and the policy whose expectations score it."""

    name: str
    parent: str
    text: str


def parse_candidate(raw: str) -> Candidate:
    """``NAME=PARENT:FILE`` as a :class:`Candidate`. ValueError names what is wrong."""
    name, _, rest = raw.partition("=")
    parent, _, file = rest.partition(":")
    if not (name and parent and file):
        raise ValueError(f"--candidate must be NAME=PARENT:FILE, got {raw!r}")
    if name in guard.POLICIES:
        raise ValueError(f"--candidate {name}: that name is a built-in policy")
    if parent not in guard.POLICIES:
        raise ValueError(f"--candidate {name}: unknown parent {parent} (known: {', '.join(guard.POLICIES)})")
    try:
        return Candidate(name, parent, Path(file).read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"--candidate {name}: cannot read {file}: {exc}") from exc


def _grey_png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (128, 128, 128)).save(buf, "PNG")
    return buf.getvalue()


def load_cases(probes: Path | None, images: Path | None) -> list[Case]:
    cases: list[Case] = []
    if probes is not None:
        for n, line in enumerate(probes.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip() or line.lstrip().startswith("//"):
                continue
            row = json.loads(line)
            if row.get("expect") not in EXPECTS:
                raise SystemExit(f"{probes}:{n}: expect must be one of {', '.join(EXPECTS)}")
            cases.append(Case(id=row["id"], expect=row["expect"], text=row["text"]))
    if images is not None:
        for expect in EXPECTS:
            for path in sorted((images / expect).glob("*")):
                if path.suffix.lower() in IMAGE_SUFFIXES:
                    cases.append(Case(id=f"img:{expect}/{path.name}", expect=expect, image=path))
    return cases


def _without_label(row: ImageResult) -> ImageResult:
    """*row* with any ``<expect>`` folder taken out of its path, so ``folder`` context cannot give the label away."""
    parts = [part for part in PurePosixPath(row.rel_path).parts if part.lower() not in EXPECTS]
    return row.model_copy(update={"rel_path": "/".join(parts)})


def with_scan_context(
    cases: list[Case], rows: list[ImageResult], kinds: frozenset[str]
) -> tuple[list[Case], list[str]]:
    """Give each image case the context and crop of its scan row; return the cases and the ids with no row.

    Text probes are left alone. A case with no row stays as it was, and is named
    so a run that reproduces nothing cannot pass for one that did.
    """
    by_path = {str(Path(r.abs_path).resolve()): r for r in rows}
    names = Counter(Path(r.abs_path).name for r in rows)
    by_name = {Path(r.abs_path).name: r for r in rows if names[Path(r.abs_path).name] == 1}
    out: list[Case] = []
    unmatched: list[str] = []
    for case in cases:
        if case.image is None:
            out.append(case)
            continue
        row = by_path.get(str(case.image.resolve())) or by_name.get(case.image.name)
        if row is None:
            unmatched.append(case.id)
            out.append(case)
        else:
            out.append(replace(case, text=image_context(_without_label(row), kinds), crop=row.crop))
    return out, unmatched


def should_flag(policy: str, expect: str) -> bool:
    if expect == "cinematic":
        return policy != "adult-cinematic"
    return (_FLAG_UNDER_DEFAULT if policy == "default" else _FLAG_UNDER_ADULT)[expect]


def _case_bytes(case: Case, grey: bytes) -> bytes:
    if case.image is None:
        return grey
    data = case.image.read_bytes()
    return crops.crop_bytes(data, case.crop) if case.crop is not None else data


def run(
    url: str,
    policies: list[str],
    cases: list[Case],
    thinking: str,
    repeat: int,
    timeout: float,
    candidates: list[Candidate] | None = None,
) -> list[dict]:
    grey = _grey_png()
    rows: list[dict] = []
    trials = {c.name: c for c in candidates or []}
    for name in [*policies, *trials]:
        trial = trials.get(name)
        text = trial.text if trial else guard.policy(name)[1]
        scored_as = trial.parent if trial else name
        client = SingGuardClient(url, timeout=timeout, thinking=thinking, policy=text)
        for case in cases:
            data = _case_bytes(case, grey)
            for attempt in range(repeat):
                started = time.monotonic()
                try:
                    result = client.check(data, text=case.text)
                except GuardCheckFailed as exc:
                    rows.append(
                        {"policy": name, "case": case.id, "expect": case.expect, "run": attempt, "error": str(exc)}
                    )
                    continue
                rows.append(
                    {
                        "policy": name,
                        "case": case.id,
                        "expect": case.expect,
                        "run": attempt,
                        "safe": result.safe,
                        "rules": result.rules,
                        "letters": sorted({rule_letter(r) for r in result.rules}),
                        "correct": (not result.safe) == should_flag(scored_as, case.expect),
                        "thinking": result.thinking,
                        "ms": round((time.monotonic() - started) * 1000),
                    }
                )
            print(f"  {name:15} {case.id}", file=sys.stderr)
    return rows


def summarise(rows: list[dict], policies: list[str]) -> str:
    """A Markdown table: per policy, the share of runs judged correctly for each expectation."""
    by: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        by[(row["policy"], row["expect"])].append(row)
    header = ["policy", *(f"{e} ok" for e in EXPECTS), "hard letters", "decided by", "errors", "median ms"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for name in policies:
        mine = [r for r in rows if r["policy"] == name]
        cells = [name]
        for expect in EXPECTS:
            judged = [r for r in by[(name, expect)] if "error" not in r]
            ok = sum(r["correct"] for r in judged)
            cells.append(f"{ok}/{len(judged)}" if judged else "-")
        letters = defaultdict(int)
        for r in by[(name, "hard")]:
            for letter in r.get("letters", []):
                letters[letter] += 1
        cells.append(" ".join(f"{k}:{v}" for k, v in sorted(letters.items())) or "-")
        passes = Counter(r["thinking"] for r in mine if r.get("thinking"))
        cells.append(" ".join(f"{k}:{v}" for k, v in sorted(passes.items())) or "-")
        cells.append(str(sum("error" in r for r in mine)))
        timings = [r["ms"] for r in mine if "ms" in r]
        cells.append(str(round(statistics.median(timings))) if timings else "-")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", required=True, help="SingGuard server origin, e.g. http://guard:9150")
    parser.add_argument(
        "--probes", type=Path, default=Path(__file__).resolve().parent.parent / "eval" / "guard_probes.jsonl"
    )
    parser.add_argument("--no-probes", action="store_true", help="Only run the --images cases")
    parser.add_argument(
        "--images", type=Path, help="Folder with hard/ ambiguous/ adult/ other/ safe/ cinematic/ subfolders"
    )
    parser.add_argument("--policies", default=",".join(guard.POLICIES), help="Comma-separated policy names")
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        metavar="NAME=PARENT:FILE",
        help="A policy wording to try: its text is in FILE, and it is scored as PARENT. Repeatable",
    )
    parser.add_argument(
        "--scan-json", type=Path, help="A saved scan: each --images case gets the context and crop of its row"
    )
    parser.add_argument(
        "--context",
        help=f"With --scan-json: context kinds to send ({', '.join(sorted(CONTEXT_KINDS))}). "
        f"Default: {guard.CONTEXT_ENV}",
    )
    parser.add_argument("--thinking", default="fast", choices=THINKING_MODES)
    parser.add_argument("--repeat", type=int, default=1, help="Runs per case (SingGuard is not deterministic)")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--out", type=Path, help="Write every verdict here as JSONL")
    args = parser.parse_args(argv)

    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    unknown = sorted(set(policies) - set(guard.POLICIES))
    if unknown:
        parser.error(f"unknown policies: {', '.join(unknown)} (known: {', '.join(guard.POLICIES)})")
    try:
        candidates = [parse_candidate(raw) for raw in args.candidate]
    except ValueError as exc:
        parser.error(str(exc))
    names = [*policies, *(c.name for c in candidates)]
    if len(set(names)) != len(names):
        parser.error("each policy and --candidate name can run only once")
    if args.scan_json is not None and args.images is None:
        parser.error("--scan-json needs --images: it gives image cases their context")
    if args.context is not None and args.scan_json is None:
        parser.error("--context needs --scan-json: the context comes from a scan's results")
    cases = load_cases(None if args.no_probes else args.probes, args.images)
    if not cases:
        parser.error("no cases: give --probes or --images")
    if args.scan_json is not None:
        if args.context is None:
            kinds = guard.context_kinds()
        else:
            kinds = frozenset(part.strip().lower() for part in args.context.split(",") if part.strip())
        if unknown_kinds := sorted(kinds - CONTEXT_KINDS):
            parser.error(
                f"unknown context kinds: {', '.join(unknown_kinds)} (known: {', '.join(sorted(CONTEXT_KINDS))})"
            )
        if not kinds:
            parser.error(f"--scan-json needs context kinds: give --context or set {guard.CONTEXT_ENV}")
        summary = ScanSummary.model_validate_json(args.scan_json.read_text(encoding="utf-8"))
        cases, unmatched = with_scan_context(cases, summary.results, kinds)
        if unmatched:
            print(f"no scan row, sent without context: {', '.join(unmatched)}", file=sys.stderr)

    rows = run(args.url, policies, cases, args.thinking, max(1, args.repeat), args.timeout, candidates)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    print(summarise(rows, names))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
