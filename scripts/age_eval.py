"""Compare MiVOLO face-only with face+body (FaceConfig.mivolo_body) on images of known age.

Each case is an image and the true age of its main subject, from a CSV with a
header row::

    path,age
    sets/ada/001.jpg,34
    sets/ada/002.jpg,34

``path`` is relative to ``--root`` (default: the CSV's folder). The age read
is the primary face's (the largest, most confident), so give one subject per
image or make sure the labelled subject is the main one.

    python scripts/age_eval.py labels.csv
    python scripts/age_eval.py labels.csv --device cuda --out runs/ages.jsonl

The table gives, per mode, how many images had a face, the mean and median
absolute error in years, the share within 5 years, and the count that matters
most for the minor checks: subjects under 18 read as 18 or over. For face+body
it also counts how many ages were read with a body; a crowded frame masks most
of each body out and falls back to face-only, as MiVOLO itself does.

Ages are tool estimates: this measures how wrong they are, it does not make
them right. Never put sexual images of anyone under 18 in a labelled set.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

from argus_curator import faces
from argus_curator.models import FaceConfig, ImageResult

ADULT = 18.0


@dataclass
class Case:
    path: Path
    age: float


@dataclass
class Reading:
    case: Case
    mode: str
    years: float | None
    with_body: bool


def load_cases(csv_path: Path, root: Path) -> list[Case]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"path", "age"} <= set(reader.fieldnames):
            raise SystemExit(f"{csv_path}: needs a header row with path and age")
        return [Case(root / row["path"], float(row["age"])) for row in reader]


def read_ages(cases: list[Case], cfg: FaceConfig, mode: str) -> list[Reading]:
    out: list[Reading] = []
    for case in cases:
        r = ImageResult(rel_path=case.path.name, abs_path=str(case.path), passed=True)
        faces.detect_faces([r], [(r.rel_path, r.abs_path, case.path.read_bytes())], cfg)
        attrs = r.primary_face_attributes
        years = attrs.age_years if attrs is not None else None
        out.append(Reading(case, mode, years, bool(attrs is not None and attrs.age_body)))
    return out


def summarise(readings: list[Reading], modes: list[str]) -> str:
    lines = [
        f"{'mode':10} {'cases':>5} {'faces':>5} {'body':>5} {'MAE':>6} {'median':>6} {'<=5y':>5} {'minor>=18':>9}",
    ]
    for mode in modes:
        rows = [r for r in readings if r.mode == mode]
        read = [r for r in rows if r.years is not None]
        errors = [abs(r.years - r.case.age) for r in read]  # type: ignore[operator]
        minors = [r for r in read if r.case.age < ADULT]
        missed = sum(1 for r in minors if r.years >= ADULT)  # type: ignore[operator]
        mae = f"{statistics.fmean(errors):6.2f}" if errors else f"{'-':>6}"
        med = f"{statistics.median(errors):6.2f}" if errors else f"{'-':>6}"
        within = f"{sum(e <= 5 for e in errors) / len(errors):5.0%}" if errors else f"{'-':>5}"
        body = sum(r.with_body for r in read)
        lines.append(
            f"{mode:10} {len(rows):5d} {len(read):5d} {body:5d} {mae} {med} {within} {f'{missed}/{len(minors)}':>9}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("labels", type=Path, help="CSV with path,age")
    parser.add_argument("--root", type=Path, help="Folder the CSV's paths are relative to (default: the CSV's)")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--body-model", default=FaceConfig().body_model, help="InsightFace Cheetah pack")
    parser.add_argument("--out", type=Path, help="Write every reading here as JSONL")
    args = parser.parse_args(argv)

    cases = load_cases(args.labels, args.root or args.labels.parent)
    if not cases:
        print(f"{args.labels}: no cases", file=sys.stderr)
        return 1
    base = {"mivolo": True, "device": args.device, "body_model": args.body_model}
    readings = read_ages(cases, FaceConfig(**base), "face") + read_ages(
        cases, FaceConfig(**base, mivolo_body=True), "face+body"
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as handle:
            for r in readings:
                row = {"path": str(r.case.path), "age": r.case.age, "mode": r.mode, "years": r.years}
                handle.write(json.dumps({**row, "with_body": r.with_body}) + "\n")
    print(summarise(readings, ["face", "face+body"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
