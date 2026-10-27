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


def test_nsfw_thresholds_without_nudenet_fail_before_scanning(dataset: Path) -> None:
    result = runner.invoke(app, ["scan", str(dataset), "--nsfw-threshold", "0.5"])
    assert result.exit_code == 1, result.output
    assert "NudeNet" in result.output and "Scanning" not in result.output


def test_nsfw_required_without_url_fails_before_scanning(dataset: Path) -> None:
    result = runner.invoke(app, ["scan", str(dataset), "--nsfw"])
    assert result.exit_code == 1, result.output
    assert "--nudenet-url" in result.output and "Scanning" not in result.output


def test_opting_out_while_filtering_on_nsfw_is_an_error(dataset: Path) -> None:
    result = runner.invoke(
        app, ["scan", str(dataset), "--no-nsfw", "--nsfw-threshold", "0.5", "--nudenet-url", "http://n:1"]
    )
    assert result.exit_code == 1, result.output
    assert "--no-nsfw" in result.output


@pytest.mark.parametrize("flags", [["--guard-unsafe"], ["--guard-rules", "A,C"]])
def test_guard_filters_without_singguard_fail_before_scanning(dataset: Path, flags: list[str]) -> None:
    result = runner.invoke(app, ["scan", str(dataset), *flags])
    assert result.exit_code == 1, result.output
    assert "SingGuard" in result.output and "Scanning" not in result.output


def test_guard_required_without_url_fails_before_scanning(dataset: Path) -> None:
    result = runner.invoke(app, ["scan", str(dataset), "--guard"])
    assert result.exit_code == 1, result.output
    assert "--singguard-url" in result.output and "Scanning" not in result.output


def test_bad_singguard_thinking_mode_fails_before_scanning(dataset: Path) -> None:
    result = runner.invoke(
        app,
        ["scan", str(dataset), "--singguard-url", "http://g:1"],
        env={"ARGUS_CURATOR_SINGGUARD_THINKING": "slower"},
    )
    assert result.exit_code == 1, result.output
    assert "fast-slow" in result.output and "Scanning" not in result.output


def test_guard_policies_must_be_judged_before_scanning(dataset: Path) -> None:
    args = ["scan", str(dataset), "--singguard-url", "http://g:1", "--guard-policy", "adult", "--guard-unsafe"]
    result = runner.invoke(app, [*args, "--guard-policies", "adult,adult-cautious"])
    assert result.exit_code == 1, result.output
    assert "--guard-variants" in result.output and "Scanning" not in result.output
    result = runner.invoke(app, [*args, "--guard-variants", "adult-nope"])
    assert result.exit_code == 1 and "adult-nope" in result.output


def test_unknown_guard_policy_fails_before_scanning(dataset: Path) -> None:
    result = runner.invoke(app, ["scan", str(dataset), "--singguard-url", "http://g:1", "--guard-policy", "nsfw"])
    assert result.exit_code == 1, result.output
    assert "default, adult" in result.output and "Scanning" not in result.output


@pytest.mark.parametrize(
    "value", ["FEMALE_BREAST_EXPOSED", "FEMALE_BREAST_EXPOSED=lots", "=0.5", "NOT_A_CLASS=0.5", "BELLY_EXPOSED=0.1"]
)
def test_bad_nsfw_label_thresholds_fail_before_scanning(dataset: Path, value: str) -> None:
    result = runner.invoke(app, ["scan", str(dataset), "--nsfw-label", value, "--nudenet-url", "http://n:1"])
    assert result.exit_code == 2, result.output
    assert "Scanning" not in result.output
