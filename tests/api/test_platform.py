"""Auth, request ids, error format, health, backends listing."""

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from imaginer.backends import registry as registry_module
from imaginer.backends.base import BackendHealth, ChatDelta, ChatRequest
from imaginer.backends.registry import BackendRegistry
from imaginer.core.config import (
    BackendConfig,
    DiffusersBackendConfig,
    LlamaCppBackendConfig,
    RoutingConfig,
    SdCppBackendConfig,
    ServerConfig,
)
from imaginer.domain.enums import BackendStatus, Capability
from imaginer.main import create_app
from tests.conftest import SettingsFactory, llama_config
from tests.fakes.fake_llama import FakeLlama


@pytest.fixture
async def secured(make_settings: SettingsFactory) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(make_settings(server=ServerConfig(auth_token=SecretStr("t0ken"))))
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            yield client


async def test_auth_required_when_token_configured(secured: httpx.AsyncClient) -> None:
    missing = await secured.get("/v1/conversations")
    assert missing.status_code == 401
    assert missing.headers["www-authenticate"] == "Bearer"
    assert missing.json()["code"] == "unauthorized"
    wrong = await secured.get("/v1/conversations", headers={"Authorization": "Bearer nope"})
    assert wrong.status_code == 401
    ok = await secured.get("/v1/conversations", headers={"Authorization": "Bearer t0ken"})
    assert ok.status_code == 200
    assert (await secured.get("/healthz")).status_code == 200


async def test_no_auth_when_token_unset(client: httpx.AsyncClient) -> None:
    assert (await client.get("/v1/conversations")).status_code == 200


async def test_request_id_echo_and_problem_body(client: httpx.AsyncClient) -> None:
    resp = await client.get("/v1/conversations/missing", headers={"X-Request-ID": "trace-123"})
    assert resp.headers["x-request-id"] == "trace-123"
    body = resp.json()
    assert body["request_id"] == "trace-123"
    assert body["status"] == 404
    assert body["type"] == "urn:imaginer:error:not_found"

    generated = await client.get("/healthz", headers={"X-Request-ID": "has spaces;rm -rf"})
    assert generated.headers["x-request-id"] != "has spaces;rm -rf"
    assert len(generated.headers["x-request-id"]) == 36


async def test_unknown_route_is_problem_json(client: httpx.AsyncClient) -> None:
    resp = await client.get("/v1/nope")
    assert resp.status_code == 404
    assert resp.headers["content-type"] == "application/problem+json"


async def test_unhandled_errors_are_500_problems(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch, app: Any
) -> None:
    async def boom(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("kaput")

    monkeypatch.setattr(app.state.container.conversations, "list", boom)
    resp = await client.get("/v1/conversations", headers={"X-Request-ID": "r-500"})
    assert resp.status_code == 500
    body = resp.json()
    assert (body["code"], body["request_id"]) == ("internal_error", "r-500")
    assert "kaput" not in resp.text


async def test_healthz_and_openapi(client: httpx.AsyncClient) -> None:
    assert (await client.get("/healthz")).json()["status"] == "ok"
    spec = (await client.get("/openapi.json")).json()
    for path in [
        "/v1/conversations",
        "/v1/conversations/{conversation_id}",
        "/v1/conversations/{conversation_id}/messages",
        "/v1/conversations/{conversation_id}/artifacts",
        "/v1/artifacts/{artifact_id}",
        "/v1/artifacts/{artifact_id}/content",
        "/v1/artifacts/{artifact_id}/lineage",
        "/v1/backends",
    ]:
        assert path in spec["paths"], path


async def test_backends_listed_with_reasons(
    make_settings: SettingsFactory, fake_llama: FakeLlama, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(registry_module, "missing_dependencies", lambda: ["torch"])
    fake_llama.vision = False
    backends: dict[str, BackendConfig] = {
        "llama": llama_config(fake_llama.url),
        "diff": DiffusersBackendConfig(type="diffusers"),
        "sd": SdCppBackendConfig(type="sdcpp", base_url="http://127.0.0.1:1", enabled=False),
    }
    app = create_app(make_settings(backends=backends))
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            listed = {b["name"]: b for b in (await client.get("/v1/backends")).json()}
    assert listed["sd"]["status"] == "unavailable"
    assert listed["sd"]["detail"] == "disabled in config"
    assert listed["sd"]["kind"] == "image"
    assert listed["diff"]["status"] == "unavailable"
    assert "missing torch" in listed["diff"]["detail"]  # optional extra not installed
    llama = listed["llama"]
    assert (llama["kind"], llama["status"]) == ("llm", "ok")
    assert llama["capabilities"] == ["chat", "stream", "tools"]
    assert llama["detail"] == "model Qwen3VL-8B-Instruct-Q4_K_M.gguf; engine lacks: vision_in"
    assert llama["latency_ms"] is not None
    assert llama["model"] == "Qwen3VL-8B-Instruct-Q4_K_M.gguf"
    # Routing is implicit here: each kind's only enabled backend is its default.
    assert llama["default_for"] == ["chat"]
    assert listed["diff"]["default_for"] == ["t2i", "edit"]
    assert (listed["sd"]["default_for"], listed["sd"]["model"]) == ([], None)


class _FakeLLM:
    def __init__(self, name: str, delay: float = 0.0) -> None:
        self.name = name
        self._delay = delay

    @property
    def model(self) -> str:
        return "fake"

    def capabilities(self) -> set[Capability]:
        return {Capability.CHAT, Capability.STREAM}

    async def chat_stream(self, req: ChatRequest) -> AsyncIterator[ChatDelta]:
        yield ChatDelta(content="hi")

    async def health(self) -> BackendHealth:
        await asyncio.sleep(self._delay)
        return BackendHealth(status=BackendStatus.OK, latency_ms=1.0)

    async def aclose(self) -> None:
        pass


async def test_registry_reports_health_and_capabilities(
    make_settings: SettingsFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = make_settings(
        backends={
            "fast": LlamaCppBackendConfig(type="llamacpp", base_url="http://x:1"),
            "slow": LlamaCppBackendConfig(type="llamacpp", base_url="http://x:2"),
        },
        routing=RoutingConfig(chat="fast"),
    )

    def factory(name: str, config: BackendConfig) -> _FakeLLM:
        return _FakeLLM(name, delay=5 if name == "slow" else 0)

    monkeypatch.setattr(registry_module, "HEALTH_TIMEOUT_S", 0.05)
    registry = BackendRegistry.from_settings(settings, {"llamacpp": factory})
    described = {b.name: b for b in await registry.describe()}
    assert described["fast"].status is BackendStatus.OK
    assert described["fast"].capabilities == [Capability.CHAT, Capability.STREAM]
    assert described["slow"].status is BackendStatus.DOWN
    await registry.aclose()


async def test_registry_survives_factory_errors(make_settings: SettingsFactory) -> None:
    settings = make_settings(
        backends={"llama": LlamaCppBackendConfig(type="llamacpp", base_url="http://x:1")}
    )

    def broken(name: str, config: BackendConfig) -> _FakeLLM:
        raise ImportError("torch is not installed")

    registry = BackendRegistry.from_settings(settings, {"llamacpp": broken})
    (info,) = await registry.describe()
    assert info.status is BackendStatus.UNAVAILABLE
    assert info.detail == "failed to initialise: torch is not installed"


async def test_error_extras_cannot_clobber_standard_members(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch, app: Any
) -> None:
    from imaginer.domain.errors import ConflictError

    async def clash(*args: Any, **kwargs: Any) -> None:
        raise ConflictError("busy", status="not-a-number", code="nope", turn_id="t1")

    monkeypatch.setattr(app.state.container.conversations, "list", clash)
    resp = await client.get("/v1/conversations")
    assert resp.status_code == 409
    body = resp.json()
    assert (body["status"], body["code"], body["turn_id"]) == (409, "conflict", "t1")
