"""CLI: request mistakes fail before the scan, not after it."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("typer")
from typer.testing import CliRunner  # noqa: E402

from argus_curator.cli import app  # noqa: E402

runner = CliRunner()

_MODEL_ENVS = (
    "ARGUS_CURATOR_FAIRFACE_MODEL",
    "ARGUS_CURATOR_MIVOLO_MODEL",
    "ARGUS_CURATOR_PLACES365_MODEL",
    "ARGUS_CURATOR_GARMENTS_MODEL",
)


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--scene", "beech"], "unknown Places365 scene"),
        (["--garment", "tshirt"], "garments"),
        (["--min-age", "40", "--max-age", "25"], "above max_age_years"),
    ],
)
def test_bad_filters_fail_before_scanning(dataset: Path, tmp_path: Path, args: list[str], message: str) -> None:
    result = runner.invoke(app, ["scan", str(dataset), "--csv", str(tmp_path / "r.csv"), *args])
    assert result.exit_code == 2, result.output
    assert message in result.output
    assert "Scanning" not in result.output


@pytest.mark.parametrize(
    ("flag", "command"),
    [
        ("--fairface", "fairface-convert"),
        ("--mivolo", "mivolo-convert"),
        ("--scenes", "places365-convert"),
        ("--garments", "garments-convert"),
    ],
)
def test_missing_model_fails_before_scanning(dataset: Path, tmp_path: Path, flag: str, command: str) -> None:
    env = {name: str(tmp_path / "absent.onnx") for name in _MODEL_ENVS}
    result = runner.invoke(app, ["scan", str(dataset), flag], env=env)
    assert result.exit_code == 1, result.output
    assert command in result.output
    assert "Scanning" not in result.output
