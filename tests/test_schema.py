"""The committed wire-contract schema must stay in lockstep with the models."""

from __future__ import annotations

import json
from pathlib import Path

from argus_curator.models import wire_schema

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schema" / "curator-wire.schema.json"


def test_committed_schema_is_current() -> None:
    assert SCHEMA_PATH.exists(), "run `argus-curator schema` to generate the committed contract"
    committed = SCHEMA_PATH.read_text(encoding="utf-8")
    rendered = json.dumps(wire_schema(), indent=2, sort_keys=True) + "\n"
    assert committed == rendered, "schema is stale — run `argus-curator schema` and commit the result"
