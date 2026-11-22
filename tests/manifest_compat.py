"""Hold ``MANIFEST_VERSION`` to the ``ManifestRow`` it describes (#168).

argus-lens and argus-forge read the JSONL manifest and branch on the major of
``MANIFEST_VERSION``. So a breaking row change must bump the major, and an
additive one the minor. ``schema/manifest-row.snapshot.json`` records the row's
schema for the version last released; ``tests/test_manifest_version.py`` compares the
live row against it.

Seen from a consumer reading rows, a change is **breaking** when:

- a field is removed, or is no longer always sent (it left ``required``);
- a field's type changes (``type``, ``anyOf`` arms, ``format``, a ``const``, ...);
- an enum gains a value: a consumer validating the field (argus-forge reads
  ``target_profile`` into the ``Literal`` fields of cortex's ``TargetProfile``)
  rejects a row that carries it;
- an enum loses a value: a consumer branching on it never sees it again.

It is **additive** when it only adds a field. Docs
(``title``/``description``) and ``default`` are ignored: ``RESPONSE_ONLY``
always sends every field, and ``manifest_version``'s default is the version itself.

Refresh the snapshot after bumping the version: ``make manifest-snapshot``. It
refuses a breaking change under the same major, and any change without a bump.

A speech export's ``SpeechManifestRow`` (#187) is held to
``SPEECH_MANIFEST_VERSION`` the same way, against
``schema/speech-manifest-row.snapshot.json`` (:data:`SPEECH`); the image
row and its version are :data:`IMAGE`. ``make manifest-snapshot`` refreshes both.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SCHEMA_DIR = Path(__file__).resolve().parent.parent / "schema"

_IGNORED = frozenset({"title", "description", "default", "examples"})
# Keywords whose value is a list of subschemas, or one subschema: walked, so a
# `$ref` inside them is resolved rather than compared as a string.
_SCHEMA_LISTS = frozenset({"anyOf", "oneOf", "allOf", "prefixItems"})
_SCHEMA_VALUES = frozenset({"items", "additionalProperties", "not", "if", "then", "else", "contains"})


@dataclass(frozen=True)
class Target:
    """A manifest row model, the version constant that describes it, and its snapshot file."""

    row: str  # the model's name in argus_curator.models
    version: str  # the version constant's name there
    path: Path

    def live(self) -> dict[str, Any]:
        from argus_curator import models

        return {
            "manifest_version": getattr(models, self.version),
            "schema": getattr(models, self.row).model_json_schema(mode="serialization"),
        }


IMAGE = Target("ManifestRow", "MANIFEST_VERSION", _SCHEMA_DIR / "manifest-row.snapshot.json")
SPEECH = Target("SpeechManifestRow", "SPEECH_MANIFEST_VERSION", _SCHEMA_DIR / "speech-manifest-row.snapshot.json")
TARGETS = (IMAGE, SPEECH)
SNAPSHOT_PATH = IMAGE.path


def live_snapshot(target: Target = IMAGE) -> dict[str, Any]:
    return target.live()


def major(version: str) -> str:
    return version.split(".", 1)[0]


def normalize(node: Any) -> Any:
    """The schema without the keys that don't change what a consumer reads."""
    if isinstance(node, dict):
        return {k: normalize(v) for k, v in node.items() if k not in _IGNORED}
    if isinstance(node, list):
        return [normalize(v) for v in node]
    return node


def breaking_changes(old: dict[str, Any], new: dict[str, Any], row: str = "ManifestRow") -> list[str]:
    """Every way ``new`` would break a consumer of rows that match ``old``."""
    old, new = normalize(old), normalize(new)
    old_defs, new_defs = old.get("$defs", {}), new.get("$defs", {})
    problems: list[str] = []
    seen: set[tuple[str, str]] = set()

    def resolve(node: Any, defs: dict[str, Any]) -> tuple[Any, str | None]:
        ref = node.get("$ref") if isinstance(node, dict) else None
        if ref is None:
            return node, None
        name = ref.rsplit("/", 1)[-1]
        return defs.get(name, {}), name

    def walk(o: Any, n: Any, path: str) -> None:
        o, o_ref = resolve(o, old_defs)
        n, n_ref = resolve(n, new_defs)
        # Only a pair of refs can recurse. A ref the new schema inlined is compared
        # at every use: each inline copy can differ.
        if o_ref is not None and n_ref is not None:
            if (o_ref, n_ref) in seen:
                return
            seen.add((o_ref, n_ref))
        if not isinstance(o, dict) or not isinstance(n, dict):
            if o != n:
                problems.append(f"{path}: changed from {o!r} to {n!r}")
            return
        for key in sorted(set(o) | set(n)):
            if key in ("$defs", "$ref"):
                continue
            here = f"{path}.{key}"
            if key not in n:
                problems.append(f"{here}: removed")
            elif key not in o:
                # A new keyword constrains values the old schema allowed (e.g. a new
                # `const` or `enum` on a field): a type change.
                problems.append(f"{here}: added")
            elif key == "properties":
                for prop, sub in o["properties"].items():
                    if prop not in n["properties"]:
                        problems.append(f"{path}.{prop}: field removed")
                    else:
                        walk(sub, n["properties"][prop], f"{path}.{prop}")
            elif key == "required":
                for prop in sorted(set(o["required"]) - set(n["required"])):
                    problems.append(f"{path}.{prop}: no longer always sent")
            elif key == "enum":
                for value in o["enum"]:
                    if value not in n["enum"]:
                        problems.append(f"{here}: lost {value!r}")
                for value in n["enum"]:
                    if value not in o["enum"]:
                        problems.append(f"{here}: gained {value!r}")
            elif key in _SCHEMA_LISTS and isinstance(o[key], list) and isinstance(n[key], list):
                if len(o[key]) != len(n[key]):
                    problems.append(f"{here}: changed from {len(o[key])} to {len(n[key])} subschemas")
                    continue
                # Walked, not compared raw: an arm is often just a `$ref`, whose
                # string stays the same while the def it names changes.
                for i, (a, b) in enumerate(zip(o[key], n[key], strict=True)):
                    walk(a, b, f"{here}[{i}]")
            elif key in _SCHEMA_VALUES and isinstance(o[key], dict):
                walk(o[key], n[key], here)
            elif o[key] != n[key]:
                problems.append(f"{here}: changed from {o[key]!r} to {n[key]!r}")

    walk(old, new, row)
    return problems


def _key(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


def verdict(snapshot: dict[str, Any], live: dict[str, Any], target: Target = IMAGE) -> str | None:
    """Why the snapshot can't stand for the live row, or None when it does."""
    row, const = target.row, target.version
    old_v, new_v = snapshot["manifest_version"], live["manifest_version"]
    if _key(new_v) < _key(old_v):
        return f"{const} went backwards: {old_v} -> {new_v}. A released version never moves down."
    same_schema = normalize(snapshot["schema"]) == normalize(live["schema"])
    if major(old_v) == major(new_v):
        broken = breaking_changes(snapshot["schema"], live["schema"], row)
        if broken:
            return (
                f"{row} changed in a way that breaks consumers, but {const} is still "
                f"{major(new_v)}.x ({old_v} -> {new_v}). Bump the major, or make the change additive:\n  "
                + "\n  ".join(broken)
            )
    if same_schema and old_v == new_v:
        return None
    if not same_schema and old_v == new_v:
        return (
            f"{row} changed but {const} is still {new_v}. Bump the minor for an additive "
            "change (the major for a breaking one), then run `make manifest-snapshot`."
        )
    return f"{const} is {new_v} but the snapshot records {old_v}: run `make manifest-snapshot`."


def _refresh(target: Target) -> int:
    live = target.live()
    if target.path.exists():
        snapshot = json.loads(target.path.read_text(encoding="utf-8"))
        problem = verdict(snapshot, live, target)
        if problem is None:
            print(f"{target.path.name} is current ({live['manifest_version']}).")
            return 0
        # Only a stale version is fixed by writing; a missing bump or an unbumped break is refused.
        if not problem.startswith(f"{target.version} is "):
            print(problem, file=sys.stderr)
            return 1
    target.path.write_text(json.dumps(live, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {target.path} ({live['manifest_version']})")
    return 0


def refresh() -> int:
    return max(_refresh(target) for target in TARGETS)


if __name__ == "__main__":
    sys.exit(refresh())
