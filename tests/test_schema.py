"""The committed wire-contract schema must stay in lockstep with the models."""

from __future__ import annotations

import json
from pathlib import Path

from argus_curator.models import render_wire_schema, wire_schema

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schema" / "curator-wire.schema.json"


def test_committed_schema_is_current() -> None:
    assert SCHEMA_PATH.exists(), "run `argus-curator schema` to generate the committed contract"
    committed = SCHEMA_PATH.read_text(encoding="utf-8")
    assert committed == render_wire_schema(), "schema is stale — run `argus-curator schema` and commit the result"


def _reachable(root: type) -> set[type]:
    import typing

    from pydantic import BaseModel

    seen: set[type] = set()
    todo = [root]
    while todo:
        model = todo.pop()
        if model in seen:
            continue
        seen.add(model)
        for field in model.model_fields.values():
            if field.exclude:  # never on the wire, so it has no schema to flag
                continue
            stack = [field.annotation]
            while stack:
                tp = stack.pop()
                if isinstance(tp, type) and issubclass(tp, BaseModel):
                    todo.append(tp)
                stack.extend(typing.get_args(tp))
    return seen


def test_only_models_no_request_carries_list_their_defaults_as_required() -> None:
    from argus_curator import models

    def reach(*roots: type) -> set[type]:
        return set().union(*(_reachable(root) for root in roots))

    received = reach(models.ScanRequest, models.ExportRequest, models.SelectRequest)
    sent = reach(models.ScanSummary, models.ExportResult, models.ManifestRow, models.SelectResult)
    # Cortex's result types are built there; they can't take curator's config.
    ours = {m for m in sent - received if m.__module__ == models.__name__}
    flagged = {
        m for m in reach(*models.WIRE_MODELS) if m.model_config.get("json_schema_serialization_defaults_required")
    }

    assert ours and flagged == ours


_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "object": (dict,),
    "array": (list,),
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "null": (type(None),),
}


def _type_matches(value: object, schema: dict, defs: dict) -> bool:
    """Whether *value* is of the type *schema* declares; a schema that declares none matches."""
    if "$ref" in schema:
        return _type_matches(value, defs[schema["$ref"].rsplit("/", 1)[-1]], defs)
    for key in ("anyOf", "oneOf"):
        if key in schema:
            return any(_type_matches(value, b, defs) for b in schema[key])
    declared = schema.get("type")
    if declared is None:
        return True
    kinds = tuple(t for name in ([declared] if isinstance(declared, str) else declared) for t in _JSON_TYPES[name])
    # JSON booleans are not integers, though Python's bool is an int.
    return isinstance(value, kinds) and not (isinstance(value, bool) and bool not in kinds)


def _missing_required(value: object, schema: dict, defs: dict, path: str = "$") -> list[str]:
    """Paths where *value* lacks a key its schema requires (a small subset of JSON Schema)."""
    if "$ref" in schema:
        return _missing_required(value, defs[schema["$ref"].rsplit("/", 1)[-1]], defs, path)
    for key in ("anyOf", "oneOf"):
        if key in schema:  # only a branch whose type matches the value can decide
            branches = [b for b in schema[key] if _type_matches(value, b, defs)]
            if not branches:
                return [f"{path} (matches no {key} branch)"]
            return min((_missing_required(value, b, defs, path) for b in branches), key=len)
    out: list[str] = []
    if isinstance(value, dict):
        out += [f"{path}.{name}" for name in schema.get("required", []) if name not in value]
        for name, sub in schema.get("properties", {}).items():
            if name in value:
                out += _missing_required(value[name], sub, defs, f"{path}.{name}")
        extra = schema.get("additionalProperties")
        if isinstance(extra, dict):
            for name, item in value.items():
                if name not in schema.get("properties", {}):
                    out += _missing_required(item, extra, defs, f"{path}[{name!r}]")
    elif isinstance(value, list) and isinstance(schema.get("items"), dict):
        for i, item in enumerate(value):
            out += _missing_required(item, schema["items"], defs, f"{path}[{i}]")
    return out


def test_what_the_server_sends_has_every_field_the_schema_requires(dataset: Path, tmp_path: Path) -> None:
    import pytest

    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from argus_curator.server import create_app

    exports = tmp_path / "exports"
    exports.mkdir()
    client = TestClient(
        create_app(cache_dir=str(tmp_path / "cache"), source_root=str(dataset), export_root=str(exports))
    )
    defs = wire_schema()["$defs"]

    def check(model: str, payload: object) -> None:
        assert _missing_required(payload, defs[model], defs) == [], model

    scan = client.post("/scan/folder", json={"folder": "", "faces": {"enabled": False}})
    assert scan.status_code == 200, scan.text
    summary = scan.json()
    assert summary["results"]
    check("ScanSummary", summary)
    check("ScanSummary", client.get(f"/scan/{summary['scan_id']}?offset=0&limit=2").json())

    selected = client.post(f"/scan/{summary['scan_id']}/select", json={"min_score": 0.0})
    assert selected.status_code == 200, selected.text
    check("SelectResult", selected.json())

    exported = client.post(
        "/export", json={"scan_id": summary["scan_id"], "dest": "out", "min_score": 0.0, "write_manifest": True}
    )
    assert exported.status_code == 200, exported.text
    result = exported.json()
    check("ExportResult", result)
    rows = [json.loads(line) for line in Path(result["manifest_path"]).read_text().splitlines()]
    assert rows
    for row in rows:
        check("ManifestRow", row)


def test_a_union_branch_of_another_type_cannot_hide_a_missing_key() -> None:
    defs = {"M": {"type": "object", "required": ["a", "b"], "properties": {"a": {}, "b": {}}}}
    union = {"anyOf": [{"$ref": "#/$defs/M"}, {"type": "string"}, {"type": "null"}]}
    assert _missing_required({"a": 1}, union, defs) == ["$.b"]
    assert _missing_required({"a": 1, "b": 2}, union, defs) == []
    assert _missing_required("text", union, defs) == []
    assert _missing_required(None, union, defs) == []
    assert _missing_required(3, union, defs) == ["$ (matches no anyOf branch)"]


def test_a_face_sent_with_its_cache_only_readings_still_has_every_required_field() -> None:
    from argus_curator.models import FaceAttributes, FaceDetection, FaceReadings

    face = FaceDetection(
        bbox=[1.0, 2.0, 3.0, 4.0], det_score=0.9, attributes=FaceAttributes(), readings=FaceReadings(mivolo_age=30.0)
    )
    defs = wire_schema()["$defs"]
    sent = face.model_dump(mode="json")

    assert "readings" not in sent
    assert _missing_required(sent, defs["FaceDetection"], defs) == []
