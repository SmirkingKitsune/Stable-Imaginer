"""Checks against a real llama-server running Qwen3-VL-8B-Instruct (GGUF + mmproj).

Opt in::

    IMAGINER_IT_LLAMA_URL=http://127.0.0.1:8080 uv run pytest -m integration

Set ``IMAGINER_IT_LLAMA_KEY`` too if the server was started with an API key.
"""

import base64
import json
import os
from collections.abc import AsyncIterator
from typing import Any

import pytest

from imaginer.backends.base import ChatRequest
from imaginer.backends.llamacpp import LlamaCppBackend
from imaginer.core.config import LlamaCppBackendConfig, TimeoutConfig
from imaginer.domain.enums import BackendStatus, Capability
from imaginer.orchestration.assembler import MessageAssembler
from tests.helpers import image_bytes

URL = os.environ.get("IMAGINER_IT_LLAMA_URL")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not URL, reason="set IMAGINER_IT_LLAMA_URL to run against llama-server"),
]

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
}


@pytest.fixture
async def backend() -> AsyncIterator[LlamaCppBackend]:
    assert URL
    config = LlamaCppBackendConfig(
        type="llamacpp",
        base_url=URL,
        api_key_env="IMAGINER_IT_LLAMA_KEY",
        timeouts=TimeoutConfig(read=300),
        sampling={"temperature": 0},
    )
    llama = LlamaCppBackend("it", config)
    health = await llama.health()
    assert health.status is BackendStatus.OK, health.detail
    yield llama
    await llama.aclose()


async def _ask(
    backend: LlamaCppBackend, messages: list[dict[str, Any]], **kw: Any
) -> MessageAssembler:
    asm = MessageAssembler()
    async for delta in backend.chat_stream(ChatRequest(messages=messages, **kw)):
        asm.add(delta)
    return asm


async def test_probe_finds_tools_and_vision(backend: LlamaCppBackend) -> None:
    # Qwen3-VL's template supports tools; vision needs --mmproj at server start.
    assert backend.capabilities() == {
        Capability.CHAT,
        Capability.STREAM,
        Capability.TOOLS,
        Capability.VISION_IN,
    }


async def test_plain_streaming(backend: LlamaCppBackend) -> None:
    asm = await _ask(backend, [{"role": "user", "content": "Reply with exactly one word: pong"}])
    assert "pong" in asm.text.lower()
    assert asm.finish_reason == "stop"


async def test_streamed_tool_call_parses(backend: LlamaCppBackend) -> None:
    asm = await _ask(
        backend,
        [{"role": "user", "content": "What is the weather in Paris right now?"}],
        tools=[WEATHER_TOOL],
        tool_choice="required",
    )
    (call,) = asm.tool_calls()
    assert call["function"]["name"] == "get_weather"
    assert "paris" in json.loads(call["function"]["arguments"])["city"].lower()
    assert asm.finish_reason == "tool_calls"


async def test_tool_result_round_trip(backend: LlamaCppBackend) -> None:
    call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
    }
    asm = await _ask(
        backend,
        [
            {"role": "user", "content": "What is the weather in Paris?"},
            {"role": "assistant", "content": "", "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "call_1", "content": '{"temp_c": 31, "sky": "clear"}'},
        ],
        tools=[WEATHER_TOOL],
    )
    assert "31" in asm.text


async def test_vision_input(backend: LlamaCppBackend) -> None:
    red = base64.b64encode(image_bytes(64, 64, color=(220, 20, 20))).decode()
    asm = await _ask(
        backend,
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What single color fills this image? One word."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{red}"}},
                ],
            }
        ],
    )
    assert "red" in asm.text.lower()
