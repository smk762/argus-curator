"""argus-iris registration (argus-cortex 0.15): the URL selects the broker, the server registers at startup."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
import httpx  # noqa: E402
from argus_cortex.gpu_broker import BrokerCoordinator  # noqa: E402
from argus_cortex.gpu_wire import load_tenant_manifest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from argus_curator import runtime  # noqa: E402
from argus_curator.server.app import create_app  # noqa: E402

_REPO = Path(__file__).parent.parent


def test_a_broker_url_alone_selects_iris_with_the_manifests_tenant(monkeypatch: pytest.MonkeyPatch) -> None:
    """No ARGUS_GPU_COORDINATOR, tenant or manifest path: the URL picks the broker, the manifest is found in the
    working directory (/app in the image), and the tenant is the manifest's."""
    monkeypatch.chdir(_REPO)
    monkeypatch.setenv("ARGUS_GPU_BROKER_URL", "http://127.0.0.1:1")
    coordinator = runtime.gpu_coordinator()
    assert isinstance(coordinator, BrokerCoordinator)
    assert coordinator.tenant == "argus-curator"
    assert coordinator.manifest == load_tenant_manifest(_REPO / "gpu-tenant.yaml")


def test_the_server_registers_with_iris_at_startup_and_signs_off_at_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[str, str, dict]] = []
    registered = threading.Event()

    def iris(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        seen.append((request.method, request.url.path, body))
        if request.method == "POST" and request.url.path == "/tenants":
            registered.set()
        return httpx.Response(200, json={})

    broker = BrokerCoordinator(
        "http://iris:8120",
        tenant="argus-curator",
        token="caller-token",
        manifest=load_tenant_manifest(_REPO / "gpu-tenant.yaml"),
        async_transport=httpx.MockTransport(iris),
        detect_container=False,
    )
    monkeypatch.setattr(runtime, "_coordinator", broker)
    with TestClient(create_app()) as client:
        assert registered.wait(5), "no registration at startup"
        assert client.get("/health").json()["gpu_coordinator"] == "broker"
    method, path, body = seen[0]
    assert (method, path, body["tenant"]) == ("POST", "/tenants", "argus-curator")
    assert {kind["kind"] for kind in body["kinds"]} >= {"faces", "scenes"}
    assert ("DELETE", "/tenants/argus-curator") in [(m, p) for m, p, _ in seen]
    assert not [p for _, p, _ in seen if p.startswith("/leases")]  # registered before, and without, any lease


def test_an_unreachable_broker_does_not_stop_the_server_starting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(_REPO)
    monkeypatch.setenv("ARGUS_GPU_BROKER_URL", "http://127.0.0.1:1")
    with TestClient(create_app()) as client:
        assert client.get("/health").status_code == 200
