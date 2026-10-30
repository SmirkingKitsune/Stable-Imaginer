from typing import Any

import pytest

from imaginer.backends.base import ChatRequest
from imaginer.backends.llamacpp import LlamaCppBackend
from imaginer.backends.openai_http.transport import HttpTransport
from imaginer.domain.enums import BackendStatus, Capability
from imaginer.domain.errors import EngineError, EngineUnavailableError
from imaginer.orchestration.assembler import MessageAssembler
from tests.conftest import llama_config
from tests.fakes.fake_llama import FakeLlama, Script
from tests.fakes.live import dead_url, refusing_client

ALL = {Capability.CHAT, Capability.STREAM, Capability.TOOLS, Capability.VISION_IN}


def _backend(url: str, **overrides: Any) -> LlamaCppBackend:
    return LlamaCppBackend("llama", llama_config(url, **overrides))


async def _collect(backend: LlamaCppBackend, **kwargs: Any) -> MessageAssembler:
    asm = MessageAssembler()
    request = ChatRequest(messages=[{"role": "user", "content": "hi"}], **kwargs)
    async for delta in backend.chat_stream(request):
        asm.add(delta)
    return asm


async def test_unprobed_backend_advertises_nothing(fake_llama: FakeLlama) -> None:
    backend = _backend(fake_llama.url)
    assert backend.capabilities() == set()
    await backend.aclose()


async def test_health_probes_capabilities_and_model(fake_llama: FakeLlama) -> None:
    backend = _backend(fake_llama.url)
    health = await backend.health()
    assert health.status is BackendStatus.OK
    assert health.latency_ms is not None and health.probed_at is not None
    assert backend.capabilities() == ALL
    assert backend.model == "Qwen3VL-8B-Instruct-Q4_K_M.gguf"
    await backend.aclose()


async def test_configured_model_name_wins(fake_llama: FakeLlama) -> None:
    backend = _backend(fake_llama.url, model="qwen3-vl-8b-instruct")
    await backend.health()
    assert backend.model == "qwen3-vl-8b-instruct"
    await backend.aclose()


async def test_missing_engine_features_are_not_advertised(fake_llama: FakeLlama) -> None:
    fake_llama.vision = False  # started without --mmproj
    fake_llama.tools = False  # template without tool support
    backend = _backend(fake_llama.url)
    health = await backend.health()
    assert health.status is BackendStatus.OK
    assert backend.capabilities() == {Capability.CHAT, Capability.STREAM}
    assert health.detail is not None and "engine lacks: tools, vision_in" in health.detail
    await backend.aclose()


async def test_declared_capabilities_only_narrow(fake_llama: FakeLlama) -> None:
    backend = _backend(fake_llama.url, declared_capabilities={Capability.CHAT, Capability.STREAM})
    await backend.health()
    assert backend.capabilities() == {Capability.CHAT, Capability.STREAM}
    await backend.aclose()


async def test_loading_model_is_degraded(fake_llama: FakeLlama) -> None:
    fake_llama.loading = True
    backend = _backend(fake_llama.url)
    health = await backend.health()
    assert health.status is BackendStatus.DEGRADED
    assert health.detail == "model is loading"
    await backend.aclose()


async def test_unreachable_engine_is_down() -> None:
    backend = _backend(dead_url())
    health = await backend.health()
    assert health.status is BackendStatus.DOWN
    assert backend.capabilities() == set()
    await backend.aclose()


async def test_stream_text_and_request_body(fake_llama: FakeLlama) -> None:
    fake_llama.queue(FakeLlama.text("Hello there, friend", reasoning="greet them"))
    backend = _backend(fake_llama.url, model="m", sampling={"temperature": 0.3, "top_p": 0.9})
    asm = await _collect(backend, params={"temperature": 0.1})
    assert asm.text == "Hello there, friend"
    assert asm.reasoning == "greet them"
    assert asm.finish_reason == "stop"
    assert asm.usage is not None and asm.usage["timings"]["predicted_n"] == 4
    body = fake_llama.requests[-1]
    assert body["stream"] is True
    assert body["model"] == "m"
    assert body["temperature"] == 0.1  # per-request params beat configured sampling
    assert body["top_p"] == 0.9
    assert "tools" not in body
    await backend.aclose()


async def test_stream_tool_call(fake_llama: FakeLlama) -> None:
    fake_llama.queue(
        FakeLlama.tool_call("generate_image", {"prompt": "a red fox", "n": 1}, grow_name=True)
    )
    backend = _backend(fake_llama.url)
    tools = [{"type": "function", "function": {"name": "generate_image", "parameters": {}}}]
    asm = await _collect(backend, tools=tools, tool_choice="auto")
    assert asm.finish_reason == "tool_calls"
    (call,) = asm.tool_calls()
    assert call["id"] == "call_fake_1"
    assert call["function"]["name"] == "generate_image"
    assert call["function"]["arguments"] == '{"prompt": "a red fox", "n": 1}'
    body = fake_llama.requests[-1]
    assert body["tools"] == tools
    assert body["tool_choice"] == "auto"
    assert body["parallel_tool_calls"] is False
    await backend.aclose()


async def test_http_error_at_start(fake_llama: FakeLlama) -> None:
    fake_llama.queue(FakeLlama.http_error(400, "the request exceeds the available context size"))
    backend = _backend(fake_llama.url)
    with pytest.raises(EngineError, match="exceeds the available context"):
        await _collect(backend)
    await backend.aclose()


async def test_error_mid_stream(fake_llama: FakeLlama) -> None:
    fake_llama.queue(Script(chunks=[{"content": "partial"}], mid_stream_error="slot crashed"))
    backend = _backend(fake_llama.url)
    with pytest.raises(EngineError, match="slot crashed"):
        await _collect(backend)
    await backend.aclose()


async def test_api_key(fake_llama: FakeLlama, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_llama.api_key = "k3y"
    unauthorized = _backend(fake_llama.url)
    with pytest.raises(EngineError, match="401"):
        await _collect(unauthorized)
    await unauthorized.aclose()

    monkeypatch.setenv("IMAGINER_TEST_LLAMA_KEY", "k3y")
    authorized = _backend(fake_llama.url, api_key_env="IMAGINER_TEST_LLAMA_KEY")
    assert (await _collect(authorized)).text.startswith("You said")
    await authorized.aclose()


async def test_unreachable_stream() -> None:
    config = llama_config("http://refused")
    transport = HttpTransport(
        name="llama",
        base_url=config.base_url,
        timeouts=config.timeouts,
        retry=config.retry,
        client=refusing_client(),
    )
    backend = LlamaCppBackend("llama", config, transport=transport)
    with pytest.raises(EngineUnavailableError):
        await _collect(backend)
    await backend.aclose()
