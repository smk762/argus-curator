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
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

SNAPSHOT_PATH = Path(__file__).resolve().parent.parent / "schema" / "manifest-row.snapshot.json"

_IGNORED = frozenset({"title", "description", "default", "examples"})


def live_snapshot() -> dict[str, Any]:
    from argus_curator.models import MANIFEST_VERSION, ManifestRow

    return {
        "manifest_version": MANIFEST_VERSION,
        "schema": ManifestRow.model_json_schema(mode="serialization"),
    }


def major(version: str) -> str:
    return version.split(".", 1)[0]


def normalize(node: Any) -> Any:
    """The schema without the keys that don't change what a consumer reads."""
    if isinstance(node, dict):
        return {k: normalize(v) for k, v in node.items() if k not in _IGNORED}
    if isinstance(node, list):
        return [normalize(v) for v in node]
    return node


def breaking_changes(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
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
            elif key == "anyOf" and len(o[key]) == len(n[key]):
                for i, (a, b) in enumerate(zip(o[key], n[key], strict=True)):
                    walk(a, b, f"{here}[{i}]")
            elif key in ("items", "additionalProperties") and isinstance(o[key], dict):
                walk(o[key], n[key], here)
            elif o[key] != n[key]:
                problems.append(f"{here}: changed from {o[key]!r} to {n[key]!r}")

    walk(old, new, "ManifestRow")
    return problems


def _key(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


def verdict(snapshot: dict[str, Any], live: dict[str, Any]) -> str | None:
    """Why the snapshot can't stand for the live row, or None when it does."""
    old_v, new_v = snapshot["manifest_version"], live["manifest_version"]
    if _key(new_v) < _key(old_v):
        return f"MANIFEST_VERSION went backwards: {old_v} -> {new_v}. A released version never moves down."
    same_schema = normalize(snapshot["schema"]) == normalize(live["schema"])
    if major(old_v) == major(new_v):
        broken = breaking_changes(snapshot["schema"], live["schema"])
        if broken:
            return (
                f"ManifestRow changed in a way that breaks consumers, but MANIFEST_VERSION is still "
                f"{major(new_v)}.x ({old_v} -> {new_v}). Bump the major, or make the change additive:\n  "
                + "\n  ".join(broken)
            )
    if same_schema and old_v == new_v:
        return None
    if not same_schema and old_v == new_v:
        return (
            f"ManifestRow changed but MANIFEST_VERSION is still {new_v}. Bump the minor for an additive "
            "change (the major for a breaking one), then run `make manifest-snapshot`."
        )
    return f"MANIFEST_VERSION is {new_v} but the snapshot records {old_v}: run `make manifest-snapshot`."


def refresh() -> int:
    live = live_snapshot()
    if SNAPSHOT_PATH.exists():
        snapshot = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
        problem = verdict(snapshot, live)
        if problem is None:
            print(f"{SNAPSHOT_PATH.name} is current ({live['manifest_version']}).")
            return 0
        # Only a stale version is fixed by writing; a missing bump or an unbumped break is refused.
        if not problem.startswith("MANIFEST_VERSION is "):
            print(problem, file=sys.stderr)
            return 1
    SNAPSHOT_PATH.write_text(json.dumps(live, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {SNAPSHOT_PATH} ({live['manifest_version']})")
    return 0


if __name__ == "__main__":
    sys.exit(refresh())
