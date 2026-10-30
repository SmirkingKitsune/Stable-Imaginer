import asyncio
from collections.abc import AsyncIterator, Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from imaginer.backends.llamacpp import LlamaCppBackend
from imaginer.backends.openai_http.transport import HttpTransport
from imaginer.container import Container
from imaginer.core.config import (
    BackendConfig,
    ImageDefaults,
    LlamaCppBackendConfig,
    LoggingConfig,
    OrchestrationConfig,
    RetryConfig,
    RoutingConfig,
    SdCppBackendConfig,
    Settings,
    StorageConfig,
    TimeoutConfig,
)
from imaginer.db import migrate
from imaginer.db.engine import Database
from imaginer.main import create_app
from tests.fakes.fake_llama import FakeLlama
from tests.fakes.fake_sdcpp import FakeSdCpp
from tests.fakes.live import LiveServer, refusing_client

SettingsFactory = Callable[..., Settings]


@pytest.fixture
def make_settings(tmp_path: Path) -> SettingsFactory:
    """Settings isolated in tmp_path, with periodic sweeps off. Override any field."""

    def factory(**overrides: Any) -> Settings:
        fields: dict[str, Any] = {
            "data_root": tmp_path / "data",
            "storage": StorageConfig(sweep_interval_s=0),
            "logging": LoggingConfig(level="WARNING"),
        }
        fields.update(overrides)
        return Settings(**fields)

    return factory


@pytest.fixture
def settings(make_settings: SettingsFactory) -> Settings:
    return make_settings()


@pytest.fixture
async def app(settings: Settings) -> AsyncIterator[FastAPI]:
    application = create_app(settings)
    async with application.router.lifespan_context(application):
        yield application


@pytest.fixture
def container(app: FastAPI) -> Container:
    result: Container = app.state.container
    return result


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        yield http


@pytest.fixture
async def database(tmp_path: Path) -> AsyncIterator[Database]:
    """A migrated SQLite database, without the HTTP app."""
    url = f"sqlite+aiosqlite:///{(tmp_path / 'test.sqlite3').as_posix()}"
    await asyncio.to_thread(migrate.upgrade, url)  # alembic runs its own event loop
    db = Database(url)
    try:
        yield db
    finally:
        await db.dispose()


# --- fake engines -------------------------------------------------------------


@pytest.fixture(scope="session")
def _fake_llama_server() -> Iterator[tuple[FakeLlama, LiveServer]]:
    fake = FakeLlama()
    with LiveServer(fake.app()) as server:
        fake.url = server.url
        yield fake, server


@pytest.fixture
def fake_llama(_fake_llama_server: tuple[FakeLlama, LiveServer]) -> FakeLlama:
    """A running fake llama-server, reset for this test."""
    fake, _ = _fake_llama_server
    fake.reset()
    return fake


def llama_config(url: str, **overrides: Any) -> LlamaCppBackendConfig:
    """Fast-failing config: two attempts, millisecond backoff."""
    fields: dict[str, Any] = {
        "type": "llamacpp",
        "base_url": url,
        "retry": RetryConfig(attempts=2, backoff_initial_s=0.01, backoff_max_s=0.02),
        "timeouts": TimeoutConfig(connect=2, read=10, write=10, pool=2),
    }
    fields.update(overrides)
    return LlamaCppBackendConfig(**fields)


@pytest.fixture
def chat_settings(make_settings: SettingsFactory, fake_llama: FakeLlama) -> Settings:
    """Settings whose only chat backend is the fake llama-server (re-probed on every use)."""
    return make_settings(
        backends={"llama": llama_config(fake_llama.url)},
        routing=RoutingConfig(chat="llama"),
        orchestration=OrchestrationConfig(probe_ttl_s=0),
    )


def refusing_llama(name: str, config: BackendConfig) -> LlamaCppBackend:
    """Backend factory whose engine refuses connections instantly (see ``refusing_client``)."""
    assert isinstance(config, LlamaCppBackendConfig)
    transport = HttpTransport(
        name=name,
        base_url=config.base_url,
        timeouts=config.timeouts,
        retry=config.retry,
        client=refusing_client(),
    )
    return LlamaCppBackend(name, config, transport=transport)


@pytest.fixture(scope="session")
def _fake_sd_server() -> Iterator[tuple[FakeSdCpp, LiveServer]]:
    fake = FakeSdCpp()
    with LiveServer(fake.app()) as server:
        fake.url = server.url
        yield fake, server


@pytest.fixture
def fake_sd(_fake_sd_server: tuple[FakeSdCpp, LiveServer]) -> Iterator[FakeSdCpp]:
    """A running fake sd-server, reset for this test."""
    fake, _ = _fake_sd_server
    fake.reset()
    yield fake
    if fake.hold is not None:
        fake.hold.set()  # never leave the fake's worker thread blocked


def sd_config(url: str, **overrides: Any) -> SdCppBackendConfig:
    """Small, fast images: 256x256, 3 steps, 10 ms polling."""
    fields: dict[str, Any] = {
        "type": "sdcpp",
        "base_url": url,
        "poll_interval_s": 0.01,
        "retry": RetryConfig(attempts=2, backoff_initial_s=0.01, backoff_max_s=0.02),
        "timeouts": TimeoutConfig(connect=2, read=10, write=10, pool=2),
        "defaults": ImageDefaults(width=256, height=256, steps=3, cfg_scale=6.0, sampler="euler"),
    }
    fields.update(overrides)
    return SdCppBackendConfig(**fields)


@pytest.fixture
def studio_settings(
    make_settings: SettingsFactory, fake_llama: FakeLlama, fake_sd: FakeSdCpp
) -> Settings:
    """Chat on the fake llama-server, images on the fake sd-server."""
    return make_settings(
        backends={"llama": llama_config(fake_llama.url), "sd": sd_config(fake_sd.url)},
        routing=RoutingConfig(chat="llama", t2i="sd", edit="sd"),
        orchestration=OrchestrationConfig(probe_ttl_s=0, max_tool_iterations=2),
    )
