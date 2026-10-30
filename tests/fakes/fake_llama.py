"""A fake ``llama-server``: /health, /props, /v1/models and streaming /v1/chat/completions.

Responses are scripted per test with ``FakeLlama.queue(...)``. Without a script it echoes
the last user message. Chunk shapes match llama.cpp's ``server_chat_msg_diff_to_json_oaicompat``:
``delta`` carries ``content``, ``reasoning_content`` and ``tool_calls[{index, id, type,
function{name, arguments}}]``, and usage/timings arrive with the final chunk.
"""

import asyncio
import json
import re
import threading
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route


@dataclass
class Script:
    chunks: list[dict[str, Any]] = field(default_factory=list)  # delta dicts
    finish_reason: str = "stop"
    status: int = 200
    error_message: str = ""
    mid_stream_error: str | None = None  # sent as an error chunk after the deltas
    cut_off: bool = False  # close the stream after the deltas, without finishing it
    delay_s: float = 0.0  # between chunks
    hold: threading.Event | None = None  # after the first chunk, wait until set


@dataclass
class FakeLlama:
    url: str = ""
    model_path: str = "/models/Qwen3VL-8B-Instruct-Q4_K_M.gguf"
    loading: bool = False
    vision: bool = True
    tools: bool = True
    api_key: str | None = None
    scripts: deque[Script] = field(default_factory=deque)
    requests: list[dict[str, Any]] = field(default_factory=list)
    props_hits: int = 0

    def reset(self) -> None:
        self.loading = False
        self.vision = True
        self.tools = True
        self.api_key = None
        self.scripts.clear()
        self.requests.clear()
        self.props_hits = 0

    def queue(self, *scripts: Script) -> None:
        self.scripts.extend(scripts)

    # --- script builders ------------------------------------------------------

    @staticmethod
    def text(
        text: str,
        *,
        pieces: int = 3,
        reasoning: str | None = None,
        delay_s: float = 0.0,
        hold: threading.Event | None = None,
    ) -> Script:
        chunks: list[dict[str, Any]] = []
        if reasoning:
            chunks.append({"reasoning_content": reasoning})
        size = max(1, -(-len(text) // pieces))
        chunks += [{"content": text[i : i + size]} for i in range(0, len(text), size)]
        return Script(chunks=chunks, delay_s=delay_s, hold=hold)

    @staticmethod
    def tool_call(
        name: str,
        arguments: dict[str, Any],
        *,
        call_id: str = "call_fake_1",
        index: int = 0,
        pieces: int = 4,
        grow_name: bool = False,
    ) -> Script:
        args = json.dumps(arguments)
        size = max(1, -(-len(args) // pieces))
        first_name = name[: max(1, len(name) // 2)] if grow_name else name
        chunks: list[dict[str, Any]] = [
            {
                "tool_calls": [
                    {
                        "index": index,
                        "id": call_id,
                        "type": "function",
                        "function": {"name": first_name},
                    }
                ]
            }
        ]
        if grow_name:  # llama.cpp re-sends id and the full name when the parse grows
            chunks.append(
                {
                    "tool_calls": [
                        {
                            "index": index,
                            "id": call_id,
                            "type": "function",
                            "function": {"name": name},
                        }
                    ]
                }
            )
        for i in range(0, len(args), size):
            chunks.append(
                {"tool_calls": [{"index": index, "function": {"arguments": args[i : i + size]}}]}
            )
        return Script(chunks=chunks, finish_reason="tool_calls")

    @staticmethod
    def http_error(status: int, message: str) -> Script:
        return Script(status=status, error_message=message)

    # --- app ------------------------------------------------------------------

    def app(self) -> Starlette:
        return Starlette(
            routes=[
                Route("/health", self._health),
                Route("/props", self._props),
                Route("/v1/models", self._models),
                Route("/v1/chat/completions", self._chat, methods=["POST"]),
            ]
        )

    def _unauthorized(self, request: Request) -> Response | None:
        if self.api_key and request.headers.get("authorization") != f"Bearer {self.api_key}":
            return JSONResponse(
                {
                    "error": {
                        "code": 401,
                        "message": "Invalid API Key",
                        "type": "authentication_error",
                    }
                },
                status_code=401,
            )
        return None

    async def _health(self, request: Request) -> Response:
        if self.loading:
            return JSONResponse(
                {"error": {"code": 503, "message": "Loading model", "type": "unavailable_error"}},
                status_code=503,
            )
        return JSONResponse({"status": "ok"})

    async def _props(self, request: Request) -> Response:
        self.props_hits += 1
        caps = (
            {"supports_tools": True, "supports_tool_calls": True, "supports_system_role": True}
            if self.tools
            else {}
        )
        return JSONResponse(
            {
                "model_path": self.model_path,
                "chat_template_caps": caps,
                "modalities": {"vision": self.vision},
                "total_slots": 1,
                "build_info": "b0-fake",
                "is_sleeping": False,
            }
        )

    async def _models(self, request: Request) -> Response:
        return JSONResponse(
            {"object": "list", "data": [{"id": self.model_path, "object": "model"}]}
        )

    async def _chat(self, request: Request) -> Response:
        if (denied := self._unauthorized(request)) is not None:
            return denied
        body = await request.json()
        self.requests.append(body)
        script = self.scripts.popleft() if self.scripts else self._echo(body)
        if script.status != 200:
            return JSONResponse(
                {"error": {"code": script.status, "message": script.error_message}},
                status_code=script.status,
            )
        if not body.get("stream"):
            text = "".join(c.get("content", "") for c in script.chunks)
            return JSONResponse(
                {"choices": [{"index": 0, "message": {"role": "assistant", "content": text}}]}
            )
        return StreamingResponse(self._stream(script), media_type="text/event-stream")

    async def _stream(self, script: Script) -> AsyncIterator[str]:
        for position, delta in enumerate(script.chunks):
            yield _sse(_chunk(delta))
            if position == 0 and script.hold is not None:
                await asyncio.to_thread(script.hold.wait, 30)
            if script.delay_s:
                await asyncio.sleep(script.delay_s)
        if script.mid_stream_error:
            yield _sse({"error": {"code": 500, "message": script.mid_stream_error}})
            return
        if script.cut_off:
            return
        final = _chunk({}, script.finish_reason)
        final["usage"] = {
            "prompt_tokens": 12,
            "completion_tokens": len(script.chunks),
            "total_tokens": 12 + len(script.chunks),
        }
        final["timings"] = {"predicted_n": len(script.chunks), "predicted_per_second": 42.0}
        yield _sse(final)
        yield "data: [DONE]\n\n"

    @staticmethod
    def _echo(body: dict[str, Any]) -> Script:
        """Unscripted behaviour, handy for manual and UI work: "draw <x>" calls
        generate_image and "edit img-N: <x>" calls edit_image (when tools are offered), a
        tool result gets a short summary, and anything else is echoed back."""
        final = body["messages"][-1]
        if final["role"] == "tool":
            result = json.loads(final["content"])
            if result.get("status") == "succeeded":
                labels = ", ".join(i["label"] for i in result["images"])
                return FakeLlama.text(f"Here you go: {labels}.")
            return FakeLlama.text(f"That didn't work: {result['error']['detail']}")
        if body.get("tools") and body.get("tool_choice") != "none" and final["role"] == "user":
            text = final["content"] if isinstance(final["content"], str) else ""
            if text.lower().startswith("draw "):
                return FakeLlama.tool_call("generate_image", {"prompt": text[5:].strip()})
            edit = re.match(r"edit (img-\d+)[:,]?\s+(.+)", text.strip(), re.IGNORECASE)
            if edit:
                arguments = {"images": [edit.group(1).lower()], "prompt": edit.group(2)}
                return FakeLlama.tool_call("edit_image", arguments)
        last = next((m for m in reversed(body["messages"]) if m["role"] == "user"), None)
        content = last["content"] if last else ""
        if isinstance(content, list):
            texts = [p["text"] for p in content if p.get("type") == "text"]
            images = sum(1 for p in content if p.get("type") == "image_url")
            content = " ".join(texts) + f" (saw {images} image{'s' if images != 1 else ''})"
        return FakeLlama.text(f"You said: {content}")


def _chunk(delta: dict[str, Any], finish_reason: str | None = None) -> dict[str, Any]:
    return {
        "id": "chatcmpl-fake",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "fake",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}\n\n"
