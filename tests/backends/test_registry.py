import asyncio
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

from imaginer.backends.base import BackendHealth, LLMBackend
from imaginer.backends.registry import BackendRegistry
from imaginer.core.clock import utcnow
from imaginer.core.config import (
    DiffusersBackendConfig,
    LlamaCppLaunch,
    OrchestrationConfig,
    RoutingConfig,
    SdCppBackendConfig,
    Settings,
)
from imaginer.domain.enums import BackendStatus, Capability
from imaginer.domain.errors import CapabilityUnavailableError, EngineUnavailableError
from tests.conftest import SettingsFactory, llama_config, refusing_llama
from tests.fakes.fake_llama import FakeLlama
from tests.fakes.live import dead_url


def _sd(enabled: bool = True) -> SdCppBackendConfig:
    return SdCppBackendConfig(type="sdcpp", base_url="http://127.0.0.1:1", enabled=enabled)


async def test_default_chat_backend(chat_settings: Settings) -> None:
    registry = BackendRegistry.from_settings(chat_settings)
    backend = (await registry.chat_backend()).backend
    assert backend.name == "llama"
    assert Capability.TOOLS in backend.capabilities()
    await registry.aclose()


async def test_single_enabled_backend_needs_no_routing(
    make_settings: SettingsFactory, fake_llama: FakeLlama
) -> None:
    settings = make_settings(
        backends={"a": llama_config(fake_llama.url), "b": llama_config(dead_url(), enabled=False)}
    )
    assert settings.default_backend("chat") == "a"
    registry = BackendRegistry.from_settings(settings)
    assert (await registry.chat_backend()).backend.name == "a"
    await registry.aclose()


def test_ambiguous_routing_is_a_config_error(fake_llama: FakeLlama) -> None:
    with pytest.raises(ValidationError, match=r"routing\.chat is not set"):
        Settings(backends={"a": llama_config(fake_llama.url), "b": llama_config(fake_llama.url)})
    with pytest.raises(ValidationError, match=r"routing\.t2i is not set"):
        Settings(backends={"x": _sd(), "y": DiffusersBackendConfig(type="diffusers")})


@pytest.mark.parametrize(
    ("name", "match"),
    [
        ("ghost", "unknown backend 'ghost'"),
        ("off", "unavailable: disabled in config"),
        ("sd", "image backend and cannot serve chat"),
    ],
)
async def test_unroutable_requests_are_422(
    make_settings: SettingsFactory, fake_llama: FakeLlama, name: str, match: str
) -> None:
    settings = make_settings(
        backends={
            "llama": llama_config(fake_llama.url),
            "off": llama_config(fake_llama.url, enabled=False),
            "sd": _sd(),
        },
        routing=RoutingConfig(chat="llama", t2i="sd"),
    )
    registry = BackendRegistry.from_settings(settings)
    with pytest.raises(CapabilityUnavailableError, match=match):
        await registry.chat_backend(name)
    await registry.aclose()


async def test_no_chat_backend_configured(make_settings: SettingsFactory) -> None:
    registry = BackendRegistry.from_settings(make_settings())
    with pytest.raises(CapabilityUnavailableError, match="no llm backend is configured"):
        await registry.chat_backend()


async def test_missing_capability_is_422_never_a_fallback(
    chat_settings: Settings, fake_llama: FakeLlama
) -> None:
    fake_llama.vision = False
    registry = BackendRegistry.from_settings(chat_settings)
    with pytest.raises(CapabilityUnavailableError) as info:
        await registry.chat_backend(require={Capability.CHAT, Capability.VISION_IN})
    assert info.value.extra["missing"] == [Capability.VISION_IN]
    await registry.aclose()


async def test_down_or_loading_engine_is_503(
    make_settings: SettingsFactory, fake_llama: FakeLlama
) -> None:
    down = BackendRegistry.from_settings(
        make_settings(backends={"llama": llama_config("http://refused")}),
        {"llamacpp": refusing_llama},
    )
    with pytest.raises(EngineUnavailableError, match="'llama' is down"):
        await down.chat_backend()
    await down.aclose()

    fake_llama.loading = True
    loading = BackendRegistry.from_settings(
        make_settings(backends={"llama": llama_config(fake_llama.url)})
    )
    with pytest.raises(EngineUnavailableError, match="degraded: model is loading"):
        await loading.chat_backend()
    fake_llama.loading = False
    assert (await loading.chat_backend()).backend.name == "llama"  # recovers without a restart
    await loading.aclose()


async def test_healthy_probe_is_reused_within_ttl(
    make_settings: SettingsFactory, fake_llama: FakeLlama
) -> None:
    settings = make_settings(
        backends={"llama": llama_config(fake_llama.url)},
        orchestration=OrchestrationConfig(probe_ttl_s=60),
    )
    registry = BackendRegistry.from_settings(settings)
    for _ in range(3):
        await registry.chat_backend()
    assert fake_llama.props_hits == 1
    await registry.aclose()


async def test_refresh_and_describe(chat_settings: Settings) -> None:
    registry = BackendRegistry.from_settings(chat_settings)
    await registry.refresh()
    (info,) = await registry.describe()
    assert info.status is BackendStatus.OK
    assert info.capabilities == sorted(
        [Capability.CHAT, Capability.STREAM, Capability.TOOLS, Capability.VISION_IN]
    )
    await registry.aclose()


# --- reconfiguring ---------------------------------------------------------------------


class _Stub:
    """An adapter that is always up and remembers being closed."""

    def __init__(self, name: str, config: object) -> None:
        self.name, self.model, self.closed = name, "stub", False

    def capabilities(self) -> set[Capability]:
        return {Capability.CHAT, Capability.STREAM}

    async def health(self) -> BackendHealth:
        return BackendHealth(status=BackendStatus.OK, probed_at=utcnow())

    async def aclose(self) -> None:
        self.closed = True


def _closed(backend: object) -> bool:
    return cast(_Stub, backend).closed


def _stub_registry(make_settings: SettingsFactory) -> BackendRegistry:
    settings = make_settings(backends={"llama": llama_config("http://127.0.0.1:1")})
    return BackendRegistry.from_settings(
        settings, factories={"llamacpp": lambda name, config: cast(LLMBackend, _Stub(name, config))}
    )


async def test_a_replaced_adapter_closes_after_its_last_lease(
    make_settings: SettingsFactory,
) -> None:
    registry = _stub_registry(make_settings)
    first, second = await registry.chat_backend(), await registry.chat_backend()
    old = first.backend
    moved = registry.config("llama").model_copy(update={"base_url": "http://127.0.0.1:2"})
    assert registry.reconfigure("llama", moved) is True
    fresh = await registry.chat_backend()
    assert fresh.backend is not first.backend

    first.release()
    first.release()  # a second release doesn't count again
    await asyncio.sleep(0)
    assert not _closed(old)  # the second lease still uses it
    second.release()
    await asyncio.sleep(0)
    assert _closed(old)

    unused = fresh.backend
    fresh.release()
    registry.reconfigure("llama", moved.model_copy(update={"base_url": "http://127.0.0.1:3"}))
    await asyncio.sleep(0)
    assert _closed(unused)  # nothing held it, so it closes at once
    await registry.aclose()


async def test_launch_only_changes_keep_the_adapter(make_settings: SettingsFactory) -> None:
    registry = _stub_registry(make_settings)
    lease = await registry.chat_backend()
    launched = registry.config("llama").model_copy(
        update={"launch": LlamaCppLaunch(model=Path("q.gguf"))}
    )
    assert registry.reconfigure("llama", launched) is False
    assert registry.config("llama") is launched
    again = await registry.chat_backend()
    assert again.backend is lease.backend and not _closed(lease.backend)
    await registry.aclose()


async def test_shutdown_closes_retired_adapters_still_leased(
    make_settings: SettingsFactory,
) -> None:
    registry = _stub_registry(make_settings)
    lease = await registry.chat_backend()
    moved = registry.config("llama").model_copy(update={"base_url": "http://127.0.0.1:2"})
    registry.reconfigure("llama", moved)
    await registry.aclose()
    assert _closed(lease.backend)
