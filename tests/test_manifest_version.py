"""MANIFEST_VERSION moves with ManifestRow: major for a break, minor for an addition (#168)."""

from __future__ import annotations

import copy
import json

import pytest
from manifest_compat import SNAPSHOT_PATH, breaking_changes, live_snapshot, verdict


def _snapshot() -> dict:
    assert SNAPSHOT_PATH.exists(), "run `make manifest-snapshot` to record the manifest row"
    return json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))


def test_manifest_version_matches_the_recorded_row() -> None:
    problem = verdict(_snapshot(), live_snapshot())
    assert problem is None, problem


# The checks the test above relies on, against planted changes to the live row.


def _bumped(snap: dict, version: str) -> dict:
    return {"manifest_version": version, "schema": snap["schema"]}


@pytest.fixture
def live() -> dict:
    return live_snapshot()


def test_removed_field_under_the_same_major_is_refused(live: dict) -> None:
    new = copy.deepcopy(live)
    del new["schema"]["properties"]["exported_abs_path"]
    new["schema"]["required"].remove("exported_abs_path")
    problem = verdict(live, _bumped(new, "2.99"))
    assert problem is not None and "exported_abs_path: field removed" in problem


def test_retyped_field_under_the_same_major_is_refused(live: dict) -> None:
    new = copy.deepcopy(live)
    new["schema"]["properties"]["score"] = {"type": "string"}
    assert any("score.type" in p for p in breaking_changes(live["schema"], new["schema"]))


def test_field_no_longer_always_sent_is_breaking(live: dict) -> None:
    new = copy.deepcopy(live)
    new["schema"]["required"].remove("rel_path")
    assert "ManifestRow.rel_path: no longer always sent" in breaking_changes(live["schema"], new["schema"])


def test_nested_model_change_is_breaking(live: dict) -> None:
    new = copy.deepcopy(live)
    target = new["schema"]["$defs"]["TargetProfile"]
    del target["properties"][next(iter(target["properties"]))]
    assert breaking_changes(live["schema"], new["schema"])


def test_an_enum_that_gains_or_loses_a_value_is_breaking() -> None:
    """A consumer that validates the field rejects a new value; one that branches on it misses a lost one."""

    def row(*poses: str) -> dict:
        pose = {"anyOf": [{"enum": list(poses), "type": "string"}, {"type": "null"}]}
        return {"type": "object", "properties": {"pose": pose}, "required": ["pose"]}

    assert breaking_changes(row("front", "side"), row("front")) == ["ManifestRow.pose.anyOf[0].enum: lost 'side'"]
    assert breaking_changes(row("front"), row("front", "side")) == ["ManifestRow.pose.anyOf[0].enum: gained 'side'"]
    assert breaking_changes(row("front", "side"), row("side", "front")) == []


def test_a_shared_ref_inlined_by_the_new_schema_is_compared_at_every_use() -> None:
    money = {"type": "object", "properties": {"amount": {"type": "number"}}, "required": ["amount"]}
    old = {
        "type": "object",
        "properties": {"a": {"$ref": "#/$defs/Money"}, "b": {"$ref": "#/$defs/Money"}},
        "$defs": {"Money": money},
    }
    retyped = {"type": "object", "properties": {"amount": {"type": "string"}}, "required": ["amount"]}
    new = {"type": "object", "properties": {"a": copy.deepcopy(money), "b": retyped}}
    assert breaking_changes(old, new) == ["ManifestRow.b.amount.type: changed from 'number' to 'string'"]


def test_breaking_change_with_a_major_bump_only_asks_for_a_refresh(live: dict) -> None:
    new = copy.deepcopy(live)
    del new["schema"]["properties"]["score"]
    problem = verdict(live, _bumped(new, "3.0"))
    assert problem is not None and "make manifest-snapshot" in problem and "breaks" not in problem


def test_additive_field_needs_a_minor_bump(live: dict) -> None:
    new = copy.deepcopy(live)
    new["schema"]["properties"]["new_field"] = {"type": "string"}
    same = verdict(live, new)
    assert same is not None and "Bump the minor" in same
    bumped = verdict(live, _bumped(new, live["manifest_version"] + ".1"))
    assert bumped is not None and "make manifest-snapshot" in bumped and "breaks" not in bumped


def test_docs_and_defaults_are_not_changes(live: dict) -> None:
    new = copy.deepcopy(live)
    new["schema"]["description"] = "reworded"
    new["schema"]["properties"]["manifest_version"]["default"] = "9.9"
    assert verdict(live, new) is None


def test_version_never_moves_down(live: dict) -> None:
    problem = verdict(_bumped(live, "2.10"), live)
    assert problem is not None and "went backwards" in problem
