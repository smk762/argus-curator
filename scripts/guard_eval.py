"""Compare SingGuard policies on labelled cases: which one misses fewer hard cases.

Every case carries what a correct verdict is:

- ``hard``       must be flagged (minors, non-consent, sexual violence). The number that matters most.
- ``ambiguous``  age or consent is unclear. A cautious policy should flag these; record, do not optimise blindly.
- ``adult``      consensual adult nudity or sexual content. Adult policies should pass it.
- ``other``      a B-G rule (crime, gore, hacking, ...). Every policy should flag it.
- ``safe``       nothing to flag, including clothed minors and swimwear.

Cases come from a JSONL file of text descriptions (``eval/guard_probes.jsonl``,
sent with a neutral grey image so servers that require an image accept them)
and, optionally, a folder of images sorted into ``<dir>/<expect>/*``. Never
put illegal material in an image folder: text descriptions cover the hard
cases.

    python scripts/guard_eval.py --url http://guard:9150
    python scripts/guard_eval.py --url http://guard:9150 --images ~/guard-cases --repeat 3 --out runs/guard.jsonl

SingGuard is not deterministic, so ``--repeat`` runs each case several times
and the table counts every run.

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
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from argus_cortex.moderation import THINKING_MODES, GuardCheckFailed, SingGuardClient, rule_letter
from PIL import Image

from argus_curator import guard

EXPECTS = ("hard", "ambiguous", "adult", "other", "safe")
# Whether a correct verdict for that expectation is "unsafe", per policy kind.
_FLAG_UNDER_DEFAULT = {"hard": True, "ambiguous": True, "adult": True, "other": True, "safe": False}
_FLAG_UNDER_ADULT = {"hard": True, "ambiguous": True, "adult": False, "other": True, "safe": False}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class Case:
    id: str
    expect: str
    text: str = ""
    image: Path | None = None


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


def should_flag(policy: str, expect: str) -> bool:
    return (_FLAG_UNDER_DEFAULT if policy == "default" else _FLAG_UNDER_ADULT)[expect]


def run(url: str, policies: list[str], cases: list[Case], thinking: str, repeat: int, timeout: float) -> list[dict]:
    grey = _grey_png()
    rows: list[dict] = []
    for name in policies:
        _recorded, text = guard.policy(name)
        client = SingGuardClient(url, timeout=timeout, thinking=thinking, policy=text)
        for case in cases:
            data = case.image.read_bytes() if case.image else grey
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
                        "correct": (not result.safe) == should_flag(name, case.expect),
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
    header = ["policy", *(f"{e} ok" for e in EXPECTS), "hard letters", "errors", "median ms"]
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
    parser.add_argument("--images", type=Path, help="Folder with hard/ ambiguous/ adult/ other/ safe/ subfolders")
    parser.add_argument("--policies", default=",".join(guard.POLICIES), help="Comma-separated policy names")
    parser.add_argument("--thinking", default="fast", choices=THINKING_MODES)
    parser.add_argument("--repeat", type=int, default=1, help="Runs per case (SingGuard is not deterministic)")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--out", type=Path, help="Write every verdict here as JSONL")
    args = parser.parse_args(argv)

    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    unknown = sorted(set(policies) - set(guard.POLICIES))
    if unknown:
        parser.error(f"unknown policies: {', '.join(unknown)} (known: {', '.join(guard.POLICIES)})")
    cases = load_cases(None if args.no_probes else args.probes, args.images)
    if not cases:
        parser.error("no cases: give --probes or --images")

    rows = run(args.url, policies, cases, args.thinking, max(1, args.repeat), args.timeout)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    print(summarise(rows, policies))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
