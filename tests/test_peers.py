"""Where curator finds its detector servers, and what it says about itself (argus-cortex 0.16).

Each server's URL is curator's own variable first, then the suite-wide ``ARGUS_<PEER>_URL``, each from the process
environment and then the suite file (``argus_cortex.peers``). The rules that must survive that: nothing set is still
"not configured" (a required check refuses), and a malformed URL is never "not configured".

``GET /health`` is cortex's ``ServiceHealth`` with every field curator served before it.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

import pytest
from argus_cortex.service import ServiceHealth
from fastapi.testclient import TestClient

from argus_curator import bodypose, erax, guard, nsfw, watermarks
from argus_curator.models import MANIFEST_VERSION, SPEECH_MANIFEST_VERSION
from argus_curator.runtime import ModelUnavailable
from argus_curator.server.app import create_app

# (module, curator's own variable, the suite-wide one)
PEERS = [
    (nsfw, "ARGUS_CURATOR_NUDENET_URL", "ARGUS_NUDENET_URL"),
    (erax, "ARGUS_CURATOR_ERAX_URL", "ARGUS_ERAX_URL"),
    (watermarks, "ARGUS_CURATOR_WATERMARK_URL", "ARGUS_WATERMARK_URL"),
    (guard, "ARGUS_CURATOR_SINGGUARD_URL", "ARGUS_SINGGUARD_URL"),
    (bodypose, "ARGUS_CURATOR_PUPPET_URL", "ARGUS_PUPPET_URL"),
]
_IDS = [own.removeprefix("ARGUS_CURATOR_").removesuffix("_URL").lower() for _mod, own, _suite in PEERS]
peers = pytest.mark.parametrize(("mod", "own", "suite"), PEERS, ids=_IDS)


def _url(mod: ModuleType) -> str | None:
    """The URL *mod* would call, whatever its accessor is named."""
    if mod is guard:
        return guard.endpoint()
    client = mod.client()
    if client is None:
        return None
    return client.pose_url if mod is bodypose else client.url


def _env_in_use(mod: ModuleType) -> str:
    return guard.env_in_use() if mod is guard else mod.SERVICE.env_in_use()


@peers
def test_nothing_set_is_not_configured(mod: ModuleType, own: str, suite: str) -> None:
    assert own == mod.URL_ENV
    assert mod.configured() is False
    assert _url(mod) is None
    assert _env_in_use(mod) == own


@peers
def test_own_variable_wins_then_the_suite_wide_one(
    mod: ModuleType, own: str, suite: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(suite, "http://suite-wide.test:8130")
    assert mod.configured() is True
    assert "suite-wide.test:8130" in (_url(mod) or "")
    assert _env_in_use(mod) == suite

    monkeypatch.setenv(own, "http://curators-own.test:9000")
    assert "curators-own.test:9000" in (_url(mod) or "")
    assert _env_in_use(mod) == own

    # Blank is unset (what compose's ${NAME-} gives), so it falls through to the suite-wide name.
    monkeypatch.setenv(own, "  ")
    assert "suite-wide.test:8130" in (_url(mod) or "")


@peers
def test_off_in_curators_variable_opts_out_of_a_suite_wide_server(
    mod: ModuleType, own: str, suite: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(suite, "http://suite-wide.test:8130")
    monkeypatch.setenv(own, "off")
    assert mod.configured() is False
    assert _url(mod) is None


@peers
def test_a_malformed_url_counts_as_set_and_fails_naming_its_variable_only(
    mod: ModuleType, own: str, suite: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A typo must not read as "not configured": that would switch an opt-out check off."""
    for env in (suite, own):
        monkeypatch.setenv(env, "ftp://user:hunter2@detector.test")
        assert mod.configured() is True
        with pytest.raises(ModelUnavailable) as err:
            _url(mod)
        assert env in str(err.value)
        assert "hunter2" not in str(err.value) and "detector.test" not in str(err.value)
        assert _env_in_use(mod) == env


@peers
def test_the_suite_file_is_read_after_the_environment(
    mod: ModuleType, own: str, suite: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    suite_file = tmp_path / "suite.env"
    suite_file.write_text(f"{suite}=http://from-the-file.test:8130\n", encoding="utf-8")
    suite_file.chmod(0o600)
    monkeypatch.setenv("ARGUS_ENV_FILE", str(suite_file))
    assert mod.configured() is True
    assert "from-the-file.test:8130" in (_url(mod) or "")

    monkeypatch.setenv(suite, "http://from-the-environment.test:8130")
    assert "from-the-environment.test:8130" in (_url(mod) or "")
    monkeypatch.setenv(own, "http://curators-own.test:9000")
    assert "curators-own.test:9000" in (_url(mod) or "")


def test_a_suite_file_others_can_read_configures_nothing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    suite_file = tmp_path / "suite.env"
    suite_file.write_text("ARGUS_NUDENET_URL=http://from-the-file.test:8130\n", encoding="utf-8")
    suite_file.chmod(0o644)
    monkeypatch.setenv("ARGUS_ENV_FILE", str(suite_file))
    assert nsfw.configured() is False and nsfw.client() is None


@pytest.fixture
def client(dataset: Path, tmp_path: Path) -> TestClient:
    return TestClient(create_app(cache_dir=str(tmp_path / "cache"), source_root=str(dataset)))


def test_a_required_check_with_no_url_anywhere_is_still_refused(client: TestClient) -> None:
    quiet = {"folder": "", "faces": {"enabled": False}}
    for body, wanted in (
        ({"nsfw": {"enabled": True}}, "ARGUS_CURATOR_NUDENET_URL is not set"),
        ({"guard": {"enabled": True}, "nsfw": {"enabled": False}}, "ARGUS_CURATOR_SINGGUARD_URL is not set"),
        ({"pose": {"enabled": True}, "nsfw": {"enabled": False}}, "ARGUS_CURATOR_PUPPET_URL is not set"),
        (
            {"crops": {"enabled": True, "detect": True}, "nsfw": {"enabled": False}},
            "ARGUS_CURATOR_WATERMARK_URL is not set",
        ),
    ):
        resp = client.post("/scan/folder", json={**quiet, **body})
        assert resp.status_code == 400, body
        assert wanted in resp.json()["detail"]


def test_a_malformed_suite_wide_url_refuses_the_scan_and_names_that_variable(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ARGUS_NUDENET_URL", "ftp://user:hunter2@nudenet.test")
    resp = client.post("/scan/folder", json={"folder": "", "faces": {"enabled": False}})  # opt-out: on once set
    assert resp.status_code == 400
    assert resp.json()["detail"] == "NudeNet check requested but ARGUS_NUDENET_URL is not a valid http(s) URL"


def test_detectors_reports_a_suite_wide_server_as_configured(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = client.get("/detectors").json()
    assert not any(before[name] for name in ("nudenet", "erax", "watermark", "singguard", "puppet"))
    monkeypatch.setenv("ARGUS_SINGGUARD_URL", "http://suite-wide.test:9150")
    after = client.get("/detectors").json()
    assert after["singguard"] is True and after["nudenet"] is False


# Every field GET /health served before it became cortex's ServiceHealth. UIs, argus-iris, darkroom and the
# tests of other repos read these by name: none may go, and none may change meaning.
_LEGACY_HEALTH_FIELDS = {
    "status",
    "service",
    "version",
    "manifest_version",
    "source_root",
    "export_root",
    "allow_move",
    "minor_exposed",
    "scan_control",
    "scan_delete",
    "set_age",
    "guard_policy_text",
    "prefilter",
    "crops",
    "guard_thinking",
    "watermark_detect",
    "export_sidecars",
    "delete_any",
    "faces",
    "speech_scan",
    "speech_export",
    "speech_manifest_version",
    "gpu_coordinator",
}


def test_health_is_service_health_and_keeps_every_field_it_had(client: TestClient, dataset: Path) -> None:
    from argus_curator import __version__

    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) >= _LEGACY_HEALTH_FIELDS
    assert (body["status"], body["service"], body["version"]) == ("ok", "argus-curator", __version__)
    assert body["manifest_version"] == MANIFEST_VERSION
    assert body["speech_manifest_version"] == SPEECH_MANIFEST_VERSION
    assert body["source_root"] == str(dataset.resolve()) and body["export_root"] is None
    assert body["allow_move"] is False and body["minor_exposed"] == "frame"
    assert body["gpu_coordinator"] == "none"
    assert isinstance(body["faces"], bool) and isinstance(body["speech_scan"], bool)

    health = ServiceHealth.model_validate(body)
    assert health.service == "argus-curator" and health.ready
    assert health.wire == {"manifest_version": MANIFEST_VERSION, "speech_manifest_version": SPEECH_MANIFEST_VERSION}
    assert health.checks == {} and health.peer is None


def test_health_answers_the_cheap_probe_and_names_its_peer_label(
    dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ARGUS_PEER", "curator")
    client = TestClient(create_app(cache_dir=str(tmp_path / "cache"), source_root=str(dataset)))
    resp = client.get("/health", params={"checks": "0"})
    assert resp.status_code == 200
    assert resp.json()["peer"] == "curator" and resp.json()["status"] == "ok"
