"""Compare SingGuard policies on labelled cases: which one misses fewer hard cases.

What a correct verdict is lives in a case file (``eval/guard_cases.toml``), not
in this script: each bucket says, per policy, whether a correct verdict is
``flag``, ``pass`` or ``either``, and an ``[[override]]`` carves out one case or
a pattern of cases. The committed buckets are:

- ``hard``       must be flagged (minors, non-consent, sexual violence). The gate: see below.
- ``ambiguous``  age or consent is unclear. A cautious policy should flag these; record, do not optimise blindly.
- ``adult``      consensual adult nudity or legal adult entertainment. Adult policies should pass it.
- ``other``      a B-G rule (how-to crime, gore, hacking, ...). Every policy should flag it.
- ``safe``       nothing to flag, including clothed minors and swimwear.
- ``cinematic``  fictional crime as a story or still. Pass under ``adult-cinematic``; flag under the others.

Cases come from a JSONL file of text descriptions (``eval/guard_probes.jsonl``)
and, optionally, a folder of images sorted into ``<dir>/<bucket>/``, in
subfolders or not; a symlinked subfolder is named on stderr and not followed.
Text probes go through SingGuard's text-only check (``check_text``), never
beside a blank image: the model noticed the image was blank and passed probes
it flags as text. A server that takes no text-only check (argus-cull's
``singguard``) answers each probe with an error row. Never put illegal material
in an image folder: text probes cover the hard cases.

    python scripts/guard_eval.py --url http://guard:9150
    python scripts/guard_eval.py --url http://guard:9150 --images ~/guard-cases --repeat 3 \\
        --cache ~/guard-runs/verdicts.jsonl --out ~/guard-runs/run1.jsonl

Runs
----
SingGuard is not deterministic, so ``--repeat`` runs every case several times.
Each repeat is a whole pass: every case under every policy once, in a shuffled
order (``--seed`` to replay one), so cases and policies interleave and repeats of
one case land far apart. Images are read from disk for each check.

``--cache FILE`` keeps every verdict, keyed by the bytes sent (the image after
its crop, or the probe text), the policy text, the context text, the mode, the
model digest and the pass number. A rerun with the same cache resumes where a
dead run stopped, and a later comparison reuses the passes an arm already has.
The pass number is in the key so repeats stay independent draws; ask for more
passes and only the new ones run. The digest comes from ``--model-digest``, from
Ollama's ``/api/tags`` (``--ollama-url`` and ``--ollama-model``), or from the
guard server's ``/health`` (``model@revision``, as argus-cull's server reports
it). It is checked again before every pass and the run stops if it changed: a
composer rebuild mid-run must not mix two models' verdicts. A cache needs a
digest. Keep the cache and ``--out`` off ``/tmp``: a reboot wipes them.

A connection failure (or an HTTP 5xx) is retried with backoff (``--retries``),
and after ``--max-errors`` such failures in a row the run stops instead of
recording hundreds of instant failures. Errors are never cached.

``--split`` picks the cases to run: ``tune`` (the default), ``heldout`` or
``all``. The case file holds a share of the cases out by a stable hash of the
id, so wording isn't tuned on the cases that judge it. Run ``heldout`` once,
for the wording you mean to ship.

Context
-------
A scan sends each image with the context from ``guard_context`` and that
context changes verdicts, so an image case sent bare may not reproduce what a
scan saw. ``--scan-json`` gives each image case the context, and the crop, of
its row in a saved scan (a ``<scan_id>.json`` from the scan cache):

    python scripts/guard_eval.py --url http://guard:9150 --no-probes --images ~/guard-cases \\
        --scan-json ~/.cache/argus_curator/scans/<scan_id>.json --context age,scene --thinking cascade

An image matches its row by path, else by a file name only one row and one
image have. Under ``--thinking cascade`` a row with an exposed reading skips the
fast screen, as it does in a scan (``guard.force_slow``). The ``folder`` kind is
refused: over a sorted set it hands the model the label (``Folder: adult``).
Frames that go with no context (no row, or a row with nothing to say, as for a
rejected low-resolution frame) test a different condition from production
frames, so the report counts them apart.

Before any check, a run with ``--scan-json`` lists the frames a person must
look at first: any frame ``minor_exposed`` catches, and any frame in a
``preflight`` bucket (``safe``) with an exposed reading. The run stops there
until ``--preflight-ok`` says someone has. ``--preflight`` lists them and exits.

Candidates
----------
To score a wording candidate, name it, the policy it derives from, and the file
holding its text. It runs beside ``--policies`` and is scored as its parent;
the parent is added to the run if it is not there:

    python scripts/guard_eval.py --url http://guard:9150 --policies adult-cinematic \\
        --candidate adult-cinematic-wm=adult-cinematic:candidate.txt

A candidate whose rule headings skip a letter (A, C, D) is refused: the model
renumbers the rules in its reasoning and the letters it reports stop meaning
anything.

Reports
-------
A frame is one case under one policy over every pass. It passed if every pass
passed, flagged if every pass flagged, and split otherwise; a split frame
counts as flagged (it goes to review). Flag-or-pass is the score; the rule
letter is secondary (the ``hard letters`` column). The report gives a table
per policy and kind of case (text probe, image with context, image without),
one per subfolder, the baseline noise per frame (consecutive passes that
flipped: noise depends on the image, so there is no single floor), the frames
that keep splitting, and every frame whose outcome changed between a candidate
and its parent (``--baseline NAME`` compares every policy with NAME too), each
marked within or beyond that frame's noise.

A candidate fails, and the exit status is 1, when it passes a frame in a
``gate`` bucket (``hard``) that its parent flagged or split. Text probes
measure prompt moderation: SingGuard reads ``input`` as a request, so a pass or
fail on a text probe doesn't show how the policy judges an image, and a gate
held on text probes alone says nothing about images.

``--report ROWS.jsonl`` prints the report for a saved ``--out`` file.

Prompt cache
------------
``--prompt-cache-check`` asks whether repeats reuse Ollama's prompt cache, and
whether a cached evaluation answers differently from a fresh one. Straight to
Ollama (``--ollama-url``, ``--ollama-model``), it sends one prompt
``--cache-check-repeats`` times back to back, as often again with
``--cache-check-fillers`` unrelated prompts before each repeat (as other cases
come between repeats in a run), and as often again with the model unloaded
before each repeat, so nothing is cached. It compares the cached token counts
and the token log-probabilities. ``--cache-check-image`` sends an image through
the chat endpoint, as the guard router does for image cases. It unloads the
model, so don't run it beside a run that shares the server.
"""

from __future__ import annotations

import argparse
import base64
import fnmatch
import hashlib
import json
import random
import re
import statistics
import sys
import time
import tomllib
import urllib.request
from collections import Counter
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, TextIO

from argus_cortex.moderation import THINKING_MODES, GuardCheckFailed, SingGuardClient, rule_letter

from argus_curator import crops, guard
from argus_curator.guard_context import CONTEXT_KINDS, image_context, parse_kinds
from argus_curator.models import SUPPORTED_EXTS, CropBox, ImageResult, ScanSummary
from argus_curator.nsfw import is_exposed
from argus_curator.runtime import ModelUnavailable
from argus_curator.selection import minor_exposed

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROBES = ROOT / "eval" / "guard_probes.jsonl"
DEFAULT_CASES = ROOT / "eval" / "guard_cases.toml"
WANTS = ("flag", "pass", "either")
SPLITS = ("tune", "heldout", "all")
# Over a sorted set the folder is the label.
EVAL_CONTEXT_KINDS = CONTEXT_KINDS - {"folder"}
_BUCKET_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_HEADING = re.compile(r"^\s*(?:#{1,6}\s*)?([A-Z])\.\s+\S", re.MULTILINE)
TEXT_PROBE_NOTE = (
    "Text probes measure prompt moderation through check_text: a pass or fail on one doesn't show "
    "how the policy judges an image."
)


# --- the case file ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class Bucket:
    name: str
    want: dict[str, str]  # policy name (or "*") -> flag / pass / either
    gate: bool = False
    preflight: bool = False


@dataclass(frozen=True)
class Override:
    case: str  # an id or an fnmatch pattern over ids
    want: dict[str, str]
    reason: str


@dataclass(frozen=True)
class CaseFile:
    buckets: dict[str, Bucket]
    overrides: tuple[Override, ...] = ()
    heldout: float = 0.0
    salt: str = ""
    heldout_cases: tuple[str, ...] = ()

    def want(self, case_id: str, bucket: str, policy: str) -> str:
        """What a correct verdict on *case_id* is under *policy*: ``flag``, ``pass`` or ``either``."""
        table = self.buckets[bucket].want
        want = table.get(policy, table["*"])
        for override in self.overrides:
            if fnmatch.fnmatchcase(case_id, override.case):
                want = override.want.get(policy, override.want.get("*", want))
        return want

    def split(self, case_id: str) -> str:
        """``heldout`` or ``tune``: pinned by ``cases``, else a stable hash of the id and the salt."""
        if any(fnmatch.fnmatchcase(case_id, pattern) for pattern in self.heldout_cases):
            return "heldout"
        digest = hashlib.sha256(f"{self.salt}\0{case_id}".encode()).digest()
        return "heldout" if int.from_bytes(digest[:8], "big") / 2**64 < self.heldout else "tune"

    @property
    def gates(self) -> frozenset[str]:
        return frozenset(name for name, bucket in self.buckets.items() if bucket.gate)


def _want_table(raw: object, where: str, *, need_default: bool) -> dict[str, str]:
    if not isinstance(raw, dict):
        raise ValueError(f"{where}: must be a table of policy = flag|pass|either")
    table: dict[str, str] = {}
    for policy, want in raw.items():
        if policy != "*" and policy not in guard.POLICIES:
            raise ValueError(f"{where}: unknown policy {policy!r} (known: *, {', '.join(guard.POLICIES)})")
        if want not in WANTS:
            raise ValueError(f"{where}.{policy}: must be one of {', '.join(WANTS)}, got {want!r}")
        table[policy] = want
    if need_default and "*" not in table:
        raise ValueError(f'{where}: needs a "*" entry for the policies it does not name')
    return table


def parse_case_file(raw: dict[str, Any], source: str = "case file") -> CaseFile:
    """A :class:`CaseFile` from parsed TOML. ValueError names what is wrong, and where."""
    if unknown := set(raw) - {"buckets", "override", "split"}:
        raise ValueError(f"{source}: unknown sections {', '.join(sorted(unknown))}")
    buckets: dict[str, Bucket] = {}
    for name, body in (raw.get("buckets") or {}).items():
        where = f"{source}: buckets.{name}"
        if not _BUCKET_NAME.match(name):
            raise ValueError(f"{where}: a bucket name is a folder name (lower case, digits, - and _)")
        if not isinstance(body, dict):
            raise ValueError(f"{where}: must be a table")
        flags = {key: body[key] for key in ("gate", "preflight") if key in body}
        if any(not isinstance(value, bool) for value in flags.values()):
            raise ValueError(f"{where}: gate and preflight are true or false")
        want = _want_table({k: v for k, v in body.items() if k not in flags}, where, need_default=True)
        buckets[name] = Bucket(name, want, **flags)
    if not buckets:
        raise ValueError(f"{source}: no [buckets]")
    overrides: list[Override] = []
    for n, body in enumerate(raw.get("override") or [], 1):
        where = f"{source}: override {n}"
        if not isinstance(body, dict) or set(body) != {"case", "want", "reason"}:
            raise ValueError(f"{where}: needs exactly case, want and reason")
        if not isinstance(body["case"], str) or not body["case"].strip():
            raise ValueError(f"{where}: case must name an id or a pattern")
        if not isinstance(body["reason"], str) or not body["reason"].strip():
            raise ValueError(f"{where}: a carve-out needs a reason")
        overrides.append(Override(body["case"], _want_table(body["want"], where, need_default=False), body["reason"]))
    split = raw.get("split") or {}
    if unknown := set(split) - {"heldout", "salt", "cases"}:
        raise ValueError(f"{source}: split: unknown keys {', '.join(sorted(unknown))}")
    heldout = split.get("heldout", 0.0)
    if not isinstance(heldout, int | float) or isinstance(heldout, bool) or not 0 <= heldout < 1:
        raise ValueError(f"{source}: split.heldout must be a share from 0 up to (not including) 1")
    cases = split.get("cases", [])
    if not isinstance(cases, list) or not all(isinstance(c, str) for c in cases):
        raise ValueError(f"{source}: split.cases must be a list of ids or patterns")
    return CaseFile(buckets, tuple(overrides), float(heldout), str(split.get("salt", "")), tuple(cases))


def load_case_file(path: Path) -> CaseFile:
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"{path}: {exc}") from exc
    return parse_case_file(raw, str(path))


# --- cases -----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    id: str
    expect: str  # the bucket
    text: str = ""
    image: Path | None = None
    crop: CropBox | None = None
    # Under cascade, judged slow with no fast screen, as a scan judges this row.
    slow: bool = False
    group: str = ""  # the subfolder it is reported under

    @property
    def kind(self) -> str:
        """``text`` (a probe), ``image`` (sent with context) or ``image-bare`` (sent with none)."""
        if self.image is None:
            return "text"
        return "image" if self.text else "image-bare"


def load_cases(probes: Path | None, images: Path | None, casefile: CaseFile | None = None) -> list[Case]:
    casefile = casefile or load_case_file(DEFAULT_CASES)
    buckets = list(casefile.buckets)
    cases: list[Case] = []
    if probes is not None:
        for n, line in enumerate(probes.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip() or line.lstrip().startswith("//"):
                continue
            row = json.loads(line)
            if row.get("expect") not in buckets:
                raise SystemExit(f"{probes}:{n}: expect must be a bucket of the case file ({', '.join(buckets)})")
            cases.append(Case(id=row["id"], expect=row["expect"], text=row["text"], group=f"text:{row['expect']}"))
    if images is not None:
        for expect in buckets:
            folder = images / expect
            for path in sorted(folder.rglob("*")):
                if path.is_dir():
                    if path.is_symlink():
                        print(f"symlinked folder not followed: {path}", file=sys.stderr)
                elif path.suffix.lower() in SUPPORTED_EXTS:
                    rel = path.relative_to(folder)
                    group = "/".join([expect, *rel.parent.parts])
                    cases.append(Case(id=f"img:{expect}/{rel.as_posix()}", expect=expect, image=path, group=group))
    return cases


def select_split(cases: list[Case], casefile: CaseFile, split: str) -> list[Case]:
    return cases if split == "all" else [c for c in cases if casefile.split(c.id) == split]


def match_rows(cases: list[Case], rows: list[ImageResult]) -> dict[str, ImageResult]:
    """Each image case's scan row, by path, else by a file name only one row and one case have.

    A file name two images share matches neither: at most one of them is that row's image.
    """
    by_path = {str(Path(r.abs_path).resolve()): r for r in rows}
    names = Counter(Path(r.abs_path).name for r in rows)
    by_name = {Path(r.abs_path).name: r for r in rows if names[Path(r.abs_path).name] == 1}
    case_names = Counter(c.image.name for c in cases if c.image is not None)
    matched: dict[str, ImageResult] = {}
    for case in cases:
        if case.image is None:
            continue
        row = by_path.get(str(case.image.resolve()))
        if row is None and case_names[case.image.name] == 1:
            row = by_name.get(case.image.name)
        if row is not None:
            matched[case.id] = row
    return matched


def with_scan_context(
    cases: list[Case], rows: list[ImageResult], kinds: frozenset[str]
) -> tuple[list[Case], list[str]]:
    """Give each image case the context and crop of its scan row; return the cases and the ids with no row.

    Text probes are left alone. A case with no row stays as it was, and is named
    so a run that reproduces nothing cannot pass for one that did.
    """
    if "folder" in kinds:
        raise ValueError("the folder context hands the model the label over a sorted set")
    matched = match_rows(cases, rows)
    out: list[Case] = []
    unmatched: list[str] = []
    for case in cases:
        row = matched.get(case.id)
        if case.image is None:
            out.append(case)
        elif row is None:
            unmatched.append(case.id)
            out.append(case)
        else:
            text = image_context(row, kinds)
            out.append(replace(case, text=text, crop=row.crop, slow=guard.force_slow(row, "cascade")))
    return out, unmatched


def preflight(cases: list[Case], matched: dict[str, ImageResult], casefile: CaseFile) -> list[tuple[str, str]]:
    """``(case id, why)`` for every frame a person must look at before any check runs.

    Anything the under-18 rule catches (judged by its strictest mode, ``frame``),
    and anything in a ``preflight`` bucket with an exposed reading.
    """
    found: list[tuple[str, str]] = []
    for case in cases:
        row = matched.get(case.id)
        if row is None:
            continue
        if minor_exposed(row, "frame"):
            found.append((case.id, "minor-exposed: an under-18 age with an exposed reading"))
        elif casefile.buckets[case.expect].preflight and is_exposed(row):
            found.append((case.id, f"exposed reading in the {case.expect} bucket"))
    return found


# --- candidates ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """A policy wording under trial: its text, and the policy whose expectations score it."""

    name: str
    parent: str
    text: str


def heading_gaps(text: str) -> str | None:
    """Why the lettered rule headings of *text* don't run A, B, C... with no gap, or ``None`` when they do."""
    letters = _HEADING.findall(text)
    if not letters:
        return None
    expected = [chr(ord("A") + i) for i in range(len(letters))]
    if letters != expected:
        return f"rule headings run {', '.join(letters)}; letter them {', '.join(expected)}"
    return None


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
        text = Path(file).expanduser().read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"--candidate {name}: cannot read {file}: {exc}") from exc
    if not text.strip():
        raise ValueError(f"--candidate {name}: {file} is empty")
    if gaps := heading_gaps(text):
        raise ValueError(f"--candidate {name}: {gaps} (the model renumbers gapped rules, so its letters mislead)")
    return Candidate(name, parent, text)


@dataclass(frozen=True)
class Arm:
    """One policy wording in a run, and the policy whose expectations score it."""

    name: str
    text: str | None
    scored_as: str
    candidate: bool = False

    @property
    def text_sha(self) -> str:
        return _sha(self.text if self.text is not None else "\0singguard-default")


def make_arms(policies: list[str], candidates: list[Candidate]) -> list[Arm]:
    # Pair each name with its own text up front: a candidate sharing a policy's
    # name must never stand in for that policy's own pass.
    arms = [Arm(p, guard.policy(p)[1], p) for p in policies]
    arms += [Arm(c.name, c.text, c.parent, candidate=True) for c in candidates]
    return arms


# --- model digest and HTTP -------------------------------------------------------------------------


def _sha(text: str | bytes) -> str:
    return hashlib.sha256(text.encode() if isinstance(text, str) else text).hexdigest()


def _http_json(url: str, body: dict | None = None, timeout: float = 10.0) -> Any:
    """GET (or POST *body* to) *url* and decode the JSON answer. Raises OSError or ValueError."""
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - operator-given URL
        return json.loads(response.read())


def ollama_digest(ollama_url: str, model: str, get: Callable[[str], Any] = _http_json) -> str | None:
    """The digest Ollama's ``/api/tags`` lists for *model*, or ``None`` when it lists no such model."""
    for entry in get(ollama_url.rstrip("/") + "/api/tags").get("models", []):
        if model in (entry.get("name"), entry.get("model")):
            return entry.get("digest")
    return None


def health_digest(url: str, get: Callable[[str], Any] = _http_json) -> str | None:
    """``model@revision`` from the guard server's ``/health``, or ``None`` when it reports no revision."""
    try:
        body = get(url.rstrip("/") + "/health")
    except (OSError, ValueError):
        return None
    if isinstance(body, dict) and body.get("model") and body.get("revision"):
        return f"{body['model']}@{body['revision']}"
    return None


def digest_source(
    url: str | None,
    explicit: str | None,
    ollama_url: str | None,
    ollama_model: str | None,
    get: Callable[[str], Any] = _http_json,
) -> Callable[[], str | None] | None:
    """A callable reading the guard model's digest now, or ``None`` when there is no way to read one."""
    if explicit:
        return lambda: explicit
    if ollama_url and ollama_model:
        return lambda: ollama_digest(ollama_url, ollama_model, get)
    if url:
        return lambda: health_digest(url, get)
    return None


# --- the verdict cache -----------------------------------------------------------------------------


def cache_key(*, image_sha: str | None, policy_sha: str, context: str, mode: str, digest: str | None, run: int) -> str:
    """The key a verdict is cached under. *image_sha* is ``None`` for a text-only check."""
    parts = {
        "image": image_sha,
        "policy": policy_sha,
        "context": _sha(context),
        "mode": mode,
        "digest": digest,
        "run": run,
    }
    return _sha(json.dumps(parts, sort_keys=True))


class VerdictCache:
    """Verdicts by :func:`cache_key`, in a JSONL file appended to as each lands, so a dead run keeps them."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.entries: dict[str, dict] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    entry = json.loads(line)
                except ValueError:
                    continue  # a line cut short when a run died
                if isinstance(entry, dict) and "key" in entry:
                    self.entries[entry["key"]] = entry
        path.parent.mkdir(parents=True, exist_ok=True)
        self._out = path.open("a", encoding="utf-8")

    def get(self, key: str) -> dict | None:
        return self.entries.get(key)

    def put(self, key: str, entry: dict) -> None:
        entry = {"key": key, **entry}
        self.entries[key] = entry
        self._out.write(json.dumps(entry) + "\n")
        self._out.flush()

    def close(self) -> None:
        self._out.close()


# --- runs ------------------------------------------------------------------------------------------


class RunStopped(RuntimeError):
    """The run cannot go on: the model changed under it, or the server stopped answering."""


def _transient(exc: GuardCheckFailed) -> bool:
    message = str(exc)
    return "request failed" in message or re.search(r"HTTP 5\d\d", message) is not None


def _case_bytes(case: Case) -> bytes | None:
    """What to send for *case*, read from disk now: ``None`` for a text probe."""
    if case.image is None:
        return None
    data = case.image.read_bytes()
    return data if case.crop is None else crops.crop_bytes(data, case.crop)


_sleep = time.sleep


def run(
    url: str,
    policies: list[str],
    cases: list[Case],
    thinking: str,
    repeat: int,
    timeout: float,
    candidates: list[Candidate] | None = None,
    out: TextIO | None = None,
    *,
    casefile: CaseFile | None = None,
    seed: int | None = None,
    cache: VerdictCache | None = None,
    digest: Callable[[], str | None] | None = None,
    retries: int = 3,
    backoff: float = 2.0,
    max_errors: int = 5,
) -> list[dict]:
    """Every verdict as a row, each also written to *out* as it lands, so a run that dies keeps what it has.

    Each of *repeat* passes judges every case under every arm once, in a shuffled order.
    """
    casefile = casefile or load_case_file(DEFAULT_CASES)
    arms = make_arms(policies, candidates or [])
    rng = random.Random(seed)
    rows: list[dict] = []
    clients: dict[tuple[str, str], SingGuardClient] = {}
    current = digest() if digest is not None else None
    if cache is not None and current is None:
        raise RunStopped("a verdict cache needs the model digest, and none could be read")
    failures_in_a_row = 0

    def client(arm: Arm, mode: str) -> SingGuardClient:
        if (arm.text_sha, mode) not in clients:
            clients[(arm.text_sha, mode)] = SingGuardClient(url, timeout=timeout, thinking=mode, policy=arm.text)
        return clients[(arm.text_sha, mode)]

    def record(row: dict) -> None:
        rows.append(row)
        if out is not None:
            out.write(json.dumps(row) + "\n")
            out.flush()

    for attempt in range(repeat):
        if digest is not None and attempt:
            now = digest()
            if now is None:
                print(f"  could not read the model digest before pass {attempt + 1}; going on", file=sys.stderr)
            elif now != current:
                raise RunStopped(f"the model digest changed from {current} to {now} before pass {attempt + 1}")
        work = [(case, arm) for case in cases for arm in arms]
        rng.shuffle(work)
        for case, arm in work:
            want = casefile.want(case.id, case.expect, arm.scored_as)
            base = {
                "policy": arm.name,
                "scored_as": arm.scored_as,
                "case": case.id,
                "expect": case.expect,
                "group": case.group,
                "kind": case.kind,
                "split": casefile.split(case.id),
                "run": attempt,
                "want": want,
                "digest": current,
            }
            try:
                data = _case_bytes(case)
            except (crops.CropError, OSError) as exc:
                # A row matched by name can be a different copy, such as an export already cropped;
                # a labelled folder can change during a long run. Neither should lose the rows so far.
                error = f"{'crop' if isinstance(exc, crops.CropError) else 'read'} {exc}"
                print(f"  {arm.name:15} {case.id}: {error}", file=sys.stderr)
                record({**base, "t": time.time(), "error": error})
                continue
            mode = "slow" if case.slow and thinking == "cascade" and data is not None else thinking
            key = cache_key(
                image_sha=None if data is None else _sha(data),
                policy_sha=arm.text_sha,
                context=case.text,
                mode=mode,
                digest=current,
                run=attempt,
            )
            hit = cache.get(key) if cache is not None else None
            started = time.monotonic()
            if hit is None:
                try:
                    result = _check(client(arm, mode), data, case.text, retries, backoff)
                except GuardCheckFailed as exc:
                    failures_in_a_row = failures_in_a_row + 1 if _transient(exc) else 0
                    record({**base, "t": time.time(), "error": str(exc)})
                    if failures_in_a_row >= max_errors:
                        raise RunStopped(f"{failures_in_a_row} checks in a row could not reach the server") from exc
                    continue
                failures_in_a_row = 0
                verdict = {
                    "safe": result.safe,
                    "rules": result.rules,
                    "thinking": result.thinking,
                    "model": result.model,
                    "explanation": result.explanation,
                    "ms": round((time.monotonic() - started) * 1000),
                    "t": time.time(),  # when it was judged: a cached row keeps its own
                }
                if cache is not None:
                    cache.put(key, verdict)
            else:
                verdict = {k: hit.get(k) for k in ("safe", "rules", "thinking", "model", "explanation", "ms", "t")}
            rules = verdict["rules"] or []
            record(
                {
                    **base,
                    **verdict,
                    "rules": rules,
                    "letters": sorted({rule_letter(r) for r in rules}),
                    "correct": None if want == "either" else (not verdict["safe"]) == (want == "flag"),
                    "cached": hit is not None,
                }
            )
            print(f"  {attempt + 1}/{repeat} {arm.name:15} {case.id}", file=sys.stderr)
        print(f"  pass {attempt + 1}/{repeat} done", file=sys.stderr)
    return rows


def _check(client: SingGuardClient, data: bytes | None, text: str, retries: int, backoff: float):
    """One verdict, retried with backoff when the server cannot be reached or answers 5xx."""
    for attempt in range(retries + 1):
        try:
            return client.check_text(text) if data is None else client.check(data, text=text)
        except GuardCheckFailed as exc:
            if attempt == retries or not _transient(exc):
                raise
            _sleep(backoff * 2**attempt)
    raise AssertionError("unreachable")


# --- reports ---------------------------------------------------------------------------------------


@dataclass
class Frame:
    """One case under one arm, over every pass."""

    arm: str
    case: str
    expect: str
    group: str
    kind: str
    want: str
    verdicts: list[tuple[int, bool]] = field(default_factory=list)  # (pass, flagged)
    errors: int = 0
    letters: Counter = field(default_factory=Counter)

    @property
    def outcome(self) -> str | None:
        flags = {flagged for _run, flagged in self.verdicts}
        if not flags:
            return None
        return "split" if len(flags) == 2 else "flag" if True in flags else "pass"

    @property
    def flagged(self) -> bool:
        """A split frame counts as flagged: it goes to review."""
        return self.outcome in ("flag", "split")

    @property
    def correct(self) -> bool | None:
        if self.want == "either" or self.outcome is None:
            return None
        return self.flagged == (self.want == "flag")

    @property
    def flips(self) -> tuple[int, int]:
        """Consecutive passes that flipped, of the consecutive pairs judged."""
        ordered = [flagged for _run, flagged in sorted(self.verdicts)]
        return sum(a != b for a, b in zip(ordered, ordered[1:], strict=False)), max(0, len(ordered) - 1)

    def counts(self) -> str:
        return f"{sum(f for _r, f in self.verdicts)}/{len(self.verdicts)}"


def frames(rows: list[dict]) -> dict[tuple[str, str], Frame]:
    out: dict[tuple[str, str], Frame] = {}
    for row in rows:
        key = (row["policy"], row["case"])
        frame = out.get(key)
        if frame is None:
            frame = out[key] = Frame(
                row["policy"],
                row["case"],
                row["expect"],
                row.get("group", row["expect"]),
                row.get("kind", "text"),
                row.get("want", "either"),
            )
        if "error" in row:
            frame.errors += 1
            continue
        frame.verdicts.append((row.get("run", len(frame.verdicts)), not row["safe"]))
        frame.letters.update(row.get("letters", []))
    return out


def _arms_in(rows: list[dict]) -> list[str]:
    return list(dict.fromkeys(r["policy"] for r in rows))


def _buckets_in(rows: list[dict]) -> list[str]:
    return list(dict.fromkeys(r["expect"] for r in rows))


def _cell(group: list[Frame]) -> str:
    judged = [f for f in group if f.correct is not None]
    if not judged:
        return "-"
    splits = sum(f.outcome == "split" for f in judged)
    return f"{sum(f.correct for f in judged)}/{len(judged)}" + (f" ({splits} split)" if splits else "")


def _table(header: list[str], body: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(cells) + " |" for cells in body]
    return "\n".join(lines)


def summarise(rows: list[dict], policies: list[str] | None = None, buckets: list[str] | None = None) -> str:
    """Per arm and kind of case, the frames judged correctly for each bucket (a split counts as flagged)."""
    policies = policies or _arms_in(rows)
    buckets = buckets or _buckets_in(rows)
    by_frame = frames(rows)
    kinds = [k for k in ("text", "image", "image-bare") if any(f.kind == k for f in by_frame.values())]
    header = ["policy", "cases", *(f"{b} ok" for b in buckets), "either", "hard letters", "decided by"]
    header += ["errors", "median ms"]
    body = []
    for name in policies:
        for kind in kinds:
            mine = [f for f in by_frame.values() if f.arm == name and f.kind == kind]
            if not mine:
                continue
            raw = [r for r in rows if r["policy"] == name and r.get("kind", "text") == kind]
            cells = [name, kind]
            cells += [_cell([f for f in mine if f.expect == b]) for b in buckets]
            cells.append(str(sum(f.want == "either" for f in mine)))
            letters: Counter = Counter()
            for f in mine:
                if f.expect == "hard":
                    letters.update(f.letters)
            cells.append(" ".join(f"{k}:{v}" for k, v in sorted(letters.items())) or "-")
            passes = Counter(r["thinking"] for r in raw if r.get("thinking"))
            cells.append(" ".join(f"{k}:{v}" for k, v in sorted(passes.items())) or "-")
            cells.append(str(sum("error" in r for r in raw)))
            timings = [r["ms"] for r in raw if r.get("ms") is not None and not r.get("cached")]
            cells.append(str(round(statistics.median(timings))) if timings else "-")
            body.append(cells)
    return _table(header, body)


def subfolder_table(rows: list[dict], policies: list[str] | None = None) -> str:
    """Per subfolder and arm: frames correct, and consecutive passes that flipped (its noise)."""
    policies = policies or _arms_in(rows)
    by_frame = frames(rows)
    groups = sorted({f.group for f in by_frame.values()})
    body = []
    for group in groups:
        cells = [group]
        for name in policies:
            mine = [f for f in by_frame.values() if f.arm == name and f.group == group]
            flips = [f.flips for f in mine]
            flipped, pairs = sum(a for a, _b in flips), sum(b for _a, b in flips)
            cells.append(f"{_cell(mine)}; flips {flipped}/{pairs}" if mine else "-")
        body.append(cells)
    return _table(["subfolder", *policies], body)


def splitting(rows: list[dict]) -> list[Frame]:
    """Frames whose passes disagree, most evenly split first."""
    split = [f for f in frames(rows).values() if f.outcome == "split"]
    return sorted(split, key=lambda f: (-f.flips[0], f.case, f.arm))


@dataclass(frozen=True)
class Change:
    case: str
    baseline: Frame
    other: Frame

    @property
    def beyond_noise(self) -> bool:
        """Both arms agreed with themselves on every pass, and disagree with each other."""
        return self.baseline.outcome != "split" and self.other.outcome != "split"

    @property
    def direction(self) -> str:
        before, after = self.baseline.correct, self.other.correct
        if before is None or after is None or before == after:
            return "-"
        return "better" if after else "worse"


def changes(rows: list[dict], other: str, baseline: str) -> list[Change]:
    """Frames whose outcome differs between *other* and *baseline*."""
    by_frame = frames(rows)
    found = []
    for (arm, case), frame in by_frame.items():
        if arm != other or (base := by_frame.get((baseline, case))) is None:
            continue
        if frame.outcome is not None and base.outcome is not None and frame.outcome != base.outcome:
            found.append(Change(case, base, frame))
    return sorted(found, key=lambda c: (not c.beyond_noise, c.case))


def gate_failures(rows: list[dict], candidate: str, parent: str, gates: frozenset[str]) -> list[Change]:
    """Gate frames (``hard``) the candidate passed on every pass where its parent flagged or split."""
    return [
        c
        for c in changes(rows, candidate, parent)
        if c.other.expect in gates and c.other.outcome == "pass" and c.baseline.flagged
    ]


def comparisons(rows: list[dict], baseline: str | None = None) -> list[tuple[str, str, bool]]:
    """``(arm, compared with, is a candidate against its parent)`` for every comparison the rows allow."""
    arms = _arms_in(rows)
    pairs: list[tuple[str, str, bool]] = []
    for arm in arms:
        parent = next((r["scored_as"] for r in rows if r["policy"] == arm and "scored_as" in r), arm)
        if parent != arm and parent in arms:
            pairs.append((arm, parent, True))
        if baseline and arm != baseline and baseline in arms and (arm, baseline, True) not in pairs:
            pairs.append((arm, baseline, False))
    return pairs


def report(
    rows: list[dict],
    policies: list[str] | None = None,
    buckets: list[str] | None = None,
    gates: frozenset[str] = frozenset({"hard"}),
    baseline: str | None = None,
) -> tuple[str, list[str]]:
    """The whole report as Markdown, and the candidates that failed the gate."""
    policies = policies or _arms_in(rows)
    out = ["## Frames correct (a split counts as flagged)", "", summarise(rows, policies, buckets), ""]
    out += ["## Per subfolder", "", subfolder_table(rows, policies), ""]
    split = splitting(rows)
    out.append(f"## Frames that keep splitting ({len(split)}, counted as flagged)")
    out.append("")
    if split:
        out.append(
            _table(
                ["case", "policy", "flagged", "flips"],
                [[f.case, f.arm, f.counts(), f"{f.flips[0]}/{f.flips[1]}"] for f in split],
            )
        )
    else:
        out.append("None.")
    out.append("")
    failed: list[str] = []
    for arm, other, is_candidate in comparisons(rows, baseline):
        found = changes(rows, arm, other)
        beyond = sum(c.beyond_noise for c in found)
        out.append(f"## {arm} against {other}: {len(found)} frames changed, {beyond} beyond noise")
        out.append("")
        if found:
            body = [
                [
                    c.case,
                    f"{c.baseline.outcome} {c.baseline.counts()} (flips {c.baseline.flips[0]}/{c.baseline.flips[1]})",
                    f"{c.other.outcome} {c.other.counts()} (flips {c.other.flips[0]}/{c.other.flips[1]})",
                    "beyond" if c.beyond_noise else "within",
                    c.direction,
                ]
                for c in found
            ]
            out.append(_table(["case", other, arm, "noise", "direction"], body))
            out.append("")
        if is_candidate:
            fails = gate_failures(rows, arm, other, gates)
            gate_rows = [r for r in rows if r["policy"] == arm and r["expect"] in gates]
            text_only = bool(gate_rows) and all(r.get("kind", "text") == "text" for r in gate_rows)
            if fails:
                failed.append(arm)
                out.append(
                    f"**FAIL**: {arm} passed {len(fails)} {'/'.join(sorted(gates))} frames {other} flagged: "
                    + ", ".join(c.case for c in fails)
                )
            else:
                out.append(f"Gate held: {arm} passed no {'/'.join(sorted(gates))} frame that {other} flagged.")
            if text_only:
                out.append("The gate was judged on text probes only, which doesn't prove image behaviour.")
            out.append("")
    if any(r.get("kind", "text") == "text" for r in rows):
        out.append(TEXT_PROBE_NOTE)
    if any(r.get("kind") == "image-bare" for r in rows):
        out.append("Images with no context (`image-bare`) test a different condition from production frames.")
    return "\n".join(out).rstrip() + "\n", failed


# --- the prompt-cache check ------------------------------------------------------------------------

_CACHE_CHECK_SYSTEM = guard.ADULT_POLICY
_CACHE_CHECK_PROMPT = "Description of an image: a nude person whose age is hard to tell. Is it safe or unsafe?"
# Unrelated prompts with no shared prefix, sent between repeats as other cases would be.
_CACHE_CHECK_FILLERS = tuple(
    (f"You are assistant {n}. Answer in one short sentence.", f"Name {n + 2} vegetables that grow underground.")
    for n in range(8)
)
CACHE_CHECK_CONDITIONS = ("back-to-back", "interleaved", "cold")


def _ollama_call(post: Callable[[str, dict], Any], url: str, model: str, system: str, prompt: str, image: str | None):
    options = {"temperature": 0, "num_predict": 32, "num_ctx": 16384}
    if image is None:
        body = {"model": model, "system": system, "prompt": prompt, "stream": False, "logprobs": True}
        return post(url + "/api/generate", {**body, "options": options})
    messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt, "images": [image]}]
    body = {"model": model, "messages": messages, "stream": False, "logprobs": True, "options": options}
    return post(url + "/api/chat", body)


def _answer(body: dict) -> str:
    return body.get("response") or (body.get("message") or {}).get("content", "")


def prompt_cache_check(
    ollama_url: str,
    model: str,
    repeats: int = 5,
    image: bytes | None = None,
    fillers: int = 4,
    post: Callable[[str, dict], Any] = lambda url, body: _http_json(url, body, timeout=300.0),
) -> tuple[str, dict]:
    """Whether repeats hit Ollama's prompt cache, and whether a cached evaluation answers as a fresh one does.

    Three conditions, *repeats* times each: ``back-to-back``; ``interleaved``, with
    *fillers* unrelated prompts before each repeat, as other cases come between
    repeats in a run; and ``cold``, with the model unloaded before each repeat, so
    nothing can be cached. Returns a Markdown summary and the raw measurements.
    """
    url = ollama_url.rstrip("/")
    b64 = base64.b64encode(image).decode("ascii") if image is not None else None
    fillers = max(1, min(fillers, len(_CACHE_CHECK_FILLERS)))

    def probe() -> dict:
        body = _ollama_call(post, url, model, _CACHE_CHECK_SYSTEM, _CACHE_CHECK_PROMPT, b64)
        return {
            "prompt": body.get("prompt_eval_count"),
            "cached": body.get("prompt_eval_cached_count"),
            "answer": _answer(body),
            "logprobs": [t.get("logprob") for t in body.get("logprobs") or []],
        }

    def filler(n: int) -> None:
        _ollama_call(post, url, model, *_CACHE_CHECK_FILLERS[n], None)

    def unload() -> None:
        post(url + "/api/generate", {"model": model, "keep_alive": 0})

    data: dict[str, Any] = {"model": model, "image": image is not None, "fillers": fillers}
    filler(0)  # load the model
    data["back-to-back"] = [probe() for _ in range(repeats)]
    data["interleaved"] = []
    for _ in range(repeats):
        for n in range(fillers):
            filler(n)
        data["interleaved"].append(probe())
    data["cold"] = []
    for _ in range(repeats):
        unload()
        data["cold"].append(probe())
    return summarise_cache_check(data), data


def _hit(run: dict) -> bool:
    """Most of the prompt came from the cache."""
    return bool(run.get("cached") and run.get("prompt") and run["cached"] >= run["prompt"] / 2)


def summarise_cache_check(data: dict) -> str:
    def tokens(runs: list[dict]) -> str:
        if all(r.get("cached") is None for r in runs):
            return "not reported"
        return ", ".join(f"{r.get('cached') or 0}/{r.get('prompt')}" for r in runs)

    runs = [r for c in CACHE_CHECK_CONDITIONS for r in data.get(c, [])]
    # What decides the answer, if the model is otherwise deterministic: how much of the prompt was cached.
    by_state: dict[object, set[tuple]] = {}
    for r in runs:
        by_state.setdefault(r.get("cached") or 0, set()).add(tuple(r["logprobs"]))
    sequences = {seq for seqs in by_state.values() for seq in seqs}
    answers = {r["answer"] for r in runs}
    kind = "image, /api/chat" if data["image"] else "text, /api/generate"
    lines = [f"Prompt-cache check on {data['model']} ({kind})", ""]
    for condition in CACHE_CHECK_CONDITIONS:
        mine = data.get(condition, [])
        lines.append(f"- {condition}: cached/prompt tokens {tokens(mine)}; {sum(map(_hit, mine))}/{len(mine)} hit")
    states = ", ".join(f"{len(seqs)} at {state} cached" for state, seqs in sorted(by_state.items(), key=str))
    lines.append(
        f"- distinct log-probability sequences: {len(sequences)} ({states}); {len(answers)} distinct answer texts"
    )
    lines.append("")
    b2b, mixed = data.get("back-to-back", []), data.get("interleaved", [])
    later = b2b[1:]  # the first follows a filler
    if later and all(map(_hit, later)):
        lines.append("Back-to-back repeats reuse the prompt cache.")
    elif any(map(_hit, later)):
        lines.append("Back-to-back repeats sometimes reuse the prompt cache.")
    else:
        lines.append("Back-to-back repeats did not reuse the prompt cache.")
    if mixed and all(map(_hit, mixed)):
        lines.append(f"{data.get('fillers')} unrelated prompts between repeats did not evict it.")
    elif any(map(_hit, mixed)):
        lines.append(f"{data.get('fillers')} unrelated prompts between repeats evicted it only sometimes.")
    elif mixed:
        lines.append(f"{data.get('fillers')} unrelated prompts between repeats evicted it.")
    deterministic = all(len(seqs) == 1 for seqs in by_state.values())
    if len(sequences) == 1:
        lines.append("Every evaluation scored the same: the cache did not change the answer here.")
    elif deterministic:
        lines.append(
            "Each cache state scored the same every time, and the states scored differently: the answer "
            "depends on how much of the prompt was cached, so repeats that share a cache state "
            "understate noise."
        )
    else:
        lines.append("Evaluations in one cache state scored differently, so there is noise the cache doesn't explain.")
    return "\n".join(lines)


# --- main ------------------------------------------------------------------------------------------


def _warn_tmp(path: Path, what: str) -> None:
    if str(path.expanduser().resolve()).startswith("/tmp/"):
        print(f"{what} {path} is under /tmp: a reboot wipes it", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", help="SingGuard server origin, e.g. http://guard:9150")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES, help="The case file: what a correct verdict is")
    parser.add_argument("--probes", type=Path, default=DEFAULT_PROBES)
    parser.add_argument("--no-probes", action="store_true", help="Only run the --images cases")
    parser.add_argument("--images", type=Path, help="Folder with one subfolder per bucket of the case file")
    parser.add_argument("--split", choices=SPLITS, default="tune", help="Which cases to run (default: tune)")
    parser.add_argument("--policies", default=",".join(guard.POLICIES), help="Comma-separated policy names")
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        metavar="NAME=PARENT:FILE",
        help="A policy wording to try: its text is in FILE, and it is scored as PARENT. Repeatable",
    )
    parser.add_argument("--baseline", help="Also compare every policy with this one, frame by frame")
    parser.add_argument(
        "--scan-json", type=Path, help="A saved scan: each --images case gets the context and crop of its row"
    )
    parser.add_argument(
        "--context",
        help=f"With --scan-json: context kinds to send ({', '.join(sorted(EVAL_CONTEXT_KINDS))}). "
        f"Default: {guard.CONTEXT_ENV}",
    )
    parser.add_argument("--preflight", action="store_true", help="List the frames to review first, then exit")
    parser.add_argument("--preflight-ok", action="store_true", help="A person has reviewed the preflight list")
    parser.add_argument("--thinking", default="fast", choices=THINKING_MODES)
    parser.add_argument("--repeat", type=int, default=1, help="Whole passes (SingGuard is not deterministic)")
    parser.add_argument("--seed", type=int, help="Seed for the pass order, to replay one")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--retries", type=int, default=3, help="Retries for a check that cannot reach the server")
    parser.add_argument("--max-errors", type=int, default=5, help="Stop after this many unreachable checks in a row")
    parser.add_argument("--cache", type=Path, help="Verdict cache (JSONL); a rerun resumes from it")
    parser.add_argument("--model-digest", help="The guard model's digest, when the server cannot report it")
    parser.add_argument("--ollama-url", help="Ollama origin: reads the model digest, and runs --prompt-cache-check")
    parser.add_argument("--ollama-model", help="The guard model's name in Ollama")
    parser.add_argument("--out", type=Path, help="Write every verdict here as JSONL")
    parser.add_argument("--report", type=Path, metavar="ROWS", help="Print the report for a saved --out file")
    parser.add_argument("--prompt-cache-check", action="store_true", help="Test Ollama's prompt cache, then exit")
    parser.add_argument("--cache-check-repeats", type=int, default=5)
    parser.add_argument("--cache-check-fillers", type=int, default=4, help="Unrelated prompts between repeats")
    parser.add_argument("--cache-check-image", type=Path, help="Send this image in the prompt-cache check")
    args = parser.parse_args(argv)

    if args.prompt_cache_check:
        if not (args.ollama_url and args.ollama_model):
            parser.error("--prompt-cache-check needs --ollama-url and --ollama-model")
        image = args.cache_check_image.read_bytes() if args.cache_check_image else None
        text, data = prompt_cache_check(
            args.ollama_url, args.ollama_model, max(2, args.cache_check_repeats), image, args.cache_check_fillers
        )
        print(text)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        return 0

    try:
        casefile = load_case_file(args.cases)
    except ValueError as exc:
        parser.error(str(exc))

    if args.report is not None:
        rows = [json.loads(line) for line in args.report.read_text(encoding="utf-8").splitlines() if line.strip()]
        text, failed = report(rows, buckets=list(casefile.buckets), gates=casefile.gates, baseline=args.baseline)
        print(text)
        return 1 if failed else 0

    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    unknown = sorted(set(policies) - set(guard.POLICIES))
    if unknown:
        parser.error(f"unknown policies: {', '.join(unknown)} (known: {', '.join(guard.POLICIES)})")
    try:
        candidates = [parse_candidate(raw) for raw in args.candidate]
    except ValueError as exc:
        parser.error(str(exc))
    for candidate in candidates:
        if candidate.parent not in policies:
            policies.append(candidate.parent)
            print(f"added {candidate.parent}: {candidate.name} is compared with its parent", file=sys.stderr)
    names = [*policies, *(c.name for c in candidates)]
    if len(set(names)) != len(names):
        parser.error("each policy and --candidate name can run only once")
    if args.baseline is not None and args.baseline not in names:
        parser.error(f"--baseline {args.baseline} is not in the run")
    if args.scan_json is not None and args.images is None:
        parser.error("--scan-json needs --images: it gives image cases their context")
    if args.context is not None and args.scan_json is None:
        parser.error("--context needs --scan-json: the context comes from a scan's results")
    if args.preflight and args.scan_json is None:
        parser.error("--preflight needs --scan-json: it reads the scan's readings")
    unmatched: list[str] = []
    cases = load_cases(None if args.no_probes else args.probes, args.images, casefile)
    if not cases:
        parser.error("no cases: give --probes or --images")
    cases = select_split(cases, casefile, args.split)
    if not cases:
        parser.error(f"no cases in the {args.split} split")
    if args.scan_json is not None:
        try:
            kinds = guard.context_kinds() if args.context is None else parse_kinds(args.context)
        except ModelUnavailable as exc:
            parser.error(str(exc))
        except ValueError as exc:
            parser.error(f"--context {exc}")
        if not kinds:
            parser.error(f"--scan-json needs context kinds: give --context or set {guard.CONTEXT_ENV}")
        if "folder" in kinds:
            parser.error("the folder context kind is refused: over a sorted set it hands the model the label")
        try:
            summary = ScanSummary.model_validate_json(args.scan_json.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            parser.error(f"--scan-json {args.scan_json}: {exc}")
        review = preflight(cases, match_rows(cases, summary.results), casefile)
        if args.preflight or (review and not args.preflight_ok):
            for case_id, why in review:
                print(f"review {case_id}: {why}", file=sys.stderr)
            if args.preflight:
                print(f"{len(review)} frames to review", file=sys.stderr)
                return 0
            print(f"{len(review)} frames to review before any check; rerun with --preflight-ok", file=sys.stderr)
            return 2
        cases, unmatched = with_scan_context(cases, summary.results, kinds)
        if unmatched:
            print(f"no scan row, sent without context: {', '.join(unmatched)}", file=sys.stderr)

    if not args.url:
        parser.error("--url is required to run checks")
    digest = digest_source(args.url, args.model_digest, args.ollama_url, args.ollama_model)
    cache = None
    if args.cache is not None:
        _warn_tmp(args.cache, "--cache")
        cache = VerdictCache(args.cache)
    if args.out:
        _warn_tmp(args.out, "--out")
        args.out.parent.mkdir(parents=True, exist_ok=True)
    seed = args.seed if args.seed is not None else random.SystemRandom().randrange(2**32)
    print(f"{len(cases)} cases, {len(names)} policies, {args.split} split, seed {seed}", file=sys.stderr)
    try:
        with args.out.open("w", encoding="utf-8") if args.out else nullcontext() as out:
            rows = run(
                args.url,
                policies,
                cases,
                args.thinking,
                max(1, args.repeat),
                args.timeout,
                candidates,
                out,
                casefile=casefile,
                seed=seed,
                cache=cache,
                digest=digest,
                retries=max(0, args.retries),
                max_errors=max(1, args.max_errors),
            )
    except RunStopped as exc:
        print(f"run stopped: {exc}", file=sys.stderr)
        return 3
    finally:
        if cache is not None:
            cache.close()
    text, failed = report(rows, names, list(casefile.buckets), casefile.gates, args.baseline)
    print(text)
    if unmatched:
        # Named again: the per-case progress has long since scrolled the first notice away.
        print(f"{len(unmatched)} of {len(cases)} cases had no scan row", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
