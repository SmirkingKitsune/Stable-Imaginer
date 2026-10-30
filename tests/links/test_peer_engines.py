"""Serving a shared engine: starting a cold one on demand, refusing what may not be done,
and the linked adapter never replaying a stream that already started."""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import httpx
import pytest

from imaginer.backends import llamacpp
from imaginer.backends.base import BackendHealth, ChatDelta, ChatRequest
from imaginer.backends.linked import LinkedLLMBackend, linked_backend
from imaginer.backends.openai_http.transport import HttpTransport
from imaginer.backends.registry import BackendRegistry
from imaginer.core.config import (
    BackendConfig,
    LinkedBackendConfig,
    LinksConfig,
    LinkShare,
    RetryConfig,
    Settings,
    TimeoutConfig,
)
from imaginer.domain import models as dto
from imaginer.domain.enums import BackendKind, BackendStatus, Capability
from imaginer.domain.errors import EngineError, EngineInterruptedError, EngineUnavailableError
from imaginer.engines import gpu
from imaginer.links import serve
from imaginer.links.serve import PeerEngines
from imaginer.links.state import Link
from imaginer.orchestration.gates import EngineGates
from tests.conftest import llama_config
from tests.links.conftest import linked

# --- a cold engine ------------------------------------------------------------------------


@dataclass
class FakeChat:
    name: str = "llama"
    up: bool = False

    @property
    def model(self) -> str:
        return "fake"

    def capabilities(self) -> set[Capability]:
        return {Capability.CHAT}

    async def chat_stream(self, req: ChatRequest) -> AsyncIterator[ChatDelta]:
        yield ChatDelta(content="hi")
        yield ChatDelta(finish_reason="stop")


@dataclass
class FakeLease:
    backend: FakeChat
    released: bool = False

    def release(self) -> None:
        self.released = True


@dataclass
class FakeRegistry:
    chat: FakeChat

    async def probe(self, name: str) -> BackendHealth:
        status = BackendStatus.OK if self.chat.up else BackendStatus.DOWN
        return BackendHealth(status=status, detail="probed")

    async def chat_backend(self, name: str, **_: Any) -> FakeLease:
        return FakeLease(self.chat)


@dataclass
class FakeProcess:
    alive: bool = True
    run: Any = None


@dataclass
class FakeSupervisor:
    procs: dict[str, FakeProcess] = field(default_factory=dict)

    def peek(self, name: str) -> FakeProcess | None:
        return self.procs.get(name)


@dataclass
class FakeEngines:
    chat: FakeChat
    exe: Path
    starts: int = 0
    supervisor: FakeSupervisor = field(default_factory=FakeSupervisor)

    def executable(self, config: Any) -> Path:
        return self.exe

    async def start(self, name: str) -> None:
        self.starts += 1
        self.supervisor.procs[name] = FakeProcess()

        async def load() -> None:
            await asyncio.sleep(0.05)
            self.chat.up = True

        asyncio.get_running_loop().create_task(load())


@pytest.fixture
def cold(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[PeerEngines, FakeEngines]:
    monkeypatch.setattr(serve, "START_POLL_S", 0.01)
    monkeypatch.setattr(serve, "gpu_memory", lambda: gpu.GpuMemory("GPU", 16384, 1024))
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF" + bytes(1024 * 1024))
    exe = tmp_path / "llama-server"
    exe.write_bytes(b"")
    settings = Settings(
        data_root=tmp_path,
        backends={"llama": llama_config("http://127.0.0.1:1", launch={"model": model})},
        links=LinksConfig(share={"llama": LinkShare(remote_start=True)}, start_timeout_s=5),
    )
    chat = FakeChat()
    engines = FakeEngines(chat, exe)
    peers = PeerEngines(
        settings, cast(Any, FakeRegistry(chat)), cast(Any, engines), EngineGates({})
    )
    return peers, engines


async def _collect(events: AsyncIterator[tuple[str, dict[str, Any]]]) -> list[tuple[str, Any]]:
    return [event async for event in events]


async def test_a_cold_engine_starts_once_for_concurrent_requests(
    cold: tuple[PeerEngines, FakeEngines],
) -> None:
    peers, engines = cold
    body = dto.PeerChatRequest(messages=[{"role": "user", "content": "hi"}])
    first, second = await asyncio.gather(
        _collect(peers.chat("llama", body)), _collect(peers.chat("llama", body))
    )
    assert engines.starts == 1
    for events in (first, second):
        names = [e for e, _ in events]
        assert names[0] == "status" and events[0][1]["state"] == "loading"
        assert names[-3:] == ["delta", "delta", "done"]


async def test_only_its_owner_starts_an_engine_without_remote_start(
    cold: tuple[PeerEngines, FakeEngines],
) -> None:
    peers, engines = cold
    peers._settings.links.share = {"llama": LinkShare(remote_start=False)}
    body = dto.PeerChatRequest(messages=[])
    with pytest.raises(EngineUnavailableError, match="only its owner may start it"):
        await _collect(peers.chat("llama", body))
    assert engines.starts == 0


async def test_a_model_that_wont_fit_is_not_started(
    cold: tuple[PeerEngines, FakeEngines], monkeypatch: pytest.MonkeyPatch
) -> None:
    peers, engines = cold
    monkeypatch.setattr(serve, "gpu_memory", lambda: gpu.GpuMemory("GPU", 2048, 1800))
    with pytest.raises(EngineUnavailableError, match="won't fit"):
        await _collect(peers.chat("llama", dto.PeerChatRequest(messages=[])))
    assert engines.starts == 0


async def _stopped(name: str, max_age_s: float = 0.0) -> dto.BackendInfo:
    return dto.BackendInfo(
        name=name,
        type="llamacpp",
        kind=BackendKind.LLM,
        enabled=True,
        status=BackendStatus.DOWN,
        capabilities=[],  # what a stopped engine reports
    )


async def test_status_reports_whether_the_model_fits(
    cold: tuple[PeerEngines, FakeEngines], monkeypatch: pytest.MonkeyPatch
) -> None:
    peers, _ = cold
    peers._registry.describe_one = _stopped  # type: ignore[method-assign]
    [engine] = await peers.status()
    assert engine.availability.state == "cold"
    assert engine.availability.need_mb == 1  # 1 MiB of weights, plus a fifth, rounded
    assert engine.availability.vram_free_mb == 16384 - 1024


async def test_a_cold_engine_can_be_routed_to_from_the_linked_instance(
    cold: tuple[PeerEngines, FakeEngines], tmp_path: Path
) -> None:
    """The caller routes by capabilities, so a cold engine reports those it will have: with
    none, its first request would be refused there and never come here to start it."""
    peers, _ = cold
    peers._registry.describe_one = _stopped  # type: ignore[method-assign]
    [engine] = await peers.status()
    assert set(engine.capabilities) == {Capability.CHAT, Capability.STREAM, Capability.TOOLS}

    link, _ = _link(lambda _: httpx.Response(500))
    link.engines = {"llama": engine}
    settings = Settings(
        data_root=tmp_path / "caller",
        backends={"llama_nas": linked("nas", "llama", BackendKind.LLM)},
    )

    def factory(name: str, config: BackendConfig) -> Any:
        assert isinstance(config, LinkedBackendConfig)
        return linked_backend(name, config, Directory(link))

    registry = BackendRegistry.from_settings(settings, {"linked": factory})
    lease = await registry.chat_backend("llama_nas")
    lease.release()
    await link.transport.aclose()


def test_vision_is_expected_only_with_a_projector(tmp_path: Path) -> None:
    model = {"model": tmp_path / "m.gguf"}
    without = llama_config("http://127.0.0.1:1", launch=model)
    with_projector = llama_config(
        "http://127.0.0.1:1", launch={**model, "mmproj": tmp_path / "p.gguf"}
    )
    assert Capability.VISION_IN not in llamacpp.expected_capabilities(without)
    assert Capability.VISION_IN in llamacpp.expected_capabilities(with_projector)
    narrowed = llama_config(
        "http://127.0.0.1:1", launch=model, declared_capabilities={Capability.CHAT}
    )
    assert llamacpp.expected_capabilities(narrowed) == {Capability.CHAT}


# --- the linked adapter ------------------------------------------------------------------


@dataclass
class Directory:
    link: Link

    def by_name(self, name: str) -> Link | None:
        return self.link

    async def fresh(self, name: str, max_age_s: float) -> Link | None:
        return self.link


def _link(handler: Any) -> tuple[Link, list[httpx.Request]]:
    sent: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        result: httpx.Response = handler(request)
        return result

    client = httpx.AsyncClient(transport=httpx.MockTransport(record), base_url="http://peer")
    transport = HttpTransport(
        name="link nas",
        base_url="http://peer",
        timeouts=TimeoutConfig(),
        retry=RetryConfig(attempts=3, backoff_initial_s=0, backoff_max_s=0),
        client=client,
    )
    link = Link(
        id="l1",
        name="nas",
        peer_instance_id="p",
        peer_name="nas",
        url="http://peer",
        created_at=datetime.now(UTC),
        transport=transport,
        status="ok",
    )
    return link, sent


def _adapter(handler: Any) -> tuple[LinkedLLMBackend, list[httpx.Request]]:
    link, sent = _link(handler)
    config = linked("nas", "llama", BackendKind.LLM)
    return LinkedLLMBackend("llama_nas", config, Directory(link)), sent


def _sse(*events: tuple[str, str]) -> httpx.Response:
    body = "".join(f"event: {e}\ndata: {d}\n\n" for e, d in events)
    return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})


async def test_a_started_stream_is_never_replayed() -> None:
    backend, sent = _adapter(lambda _: _sse(("delta", '{"content": "par"}')))
    seen = []
    with pytest.raises(EngineInterruptedError):
        async for delta in backend.chat_stream(ChatRequest(messages=[])):
            seen.append(delta.content)
    assert seen == ["par"]
    assert len(sent) == 1


async def test_peer_errors_keep_their_kind() -> None:
    problem = '{"code": "engine_unavailable", "detail": "llama is down"}'
    backend, _ = _adapter(lambda _: _sse(("status", "{}"), ("error", problem)))
    with pytest.raises(EngineUnavailableError, match="nas: llama is down"):
        async for _ in backend.chat_stream(ChatRequest(messages=[])):
            pass
    odd = '{"code": "something_new", "detail": "?"}'
    backend, _ = _adapter(lambda _: _sse(("error", odd)))
    with pytest.raises(EngineError):
        async for _ in backend.chat_stream(ChatRequest(messages=[])):
            pass
