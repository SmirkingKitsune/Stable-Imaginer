"""OpenAI chat-completions streaming: SSE framing and chunk parsing.

A stream is finished once a chunk has a ``finish_reason`` or ``data: [DONE]`` arrives. If
the connection closes before either, the answer is cut off, and that is an error rather
than a short reply.
"""

import json
from collections.abc import AsyncIterable, AsyncIterator
from typing import Any

from imaginer.backends.base import ChatDelta, ToolCallFragment
from imaginer.domain.errors import EngineError, EngineInterruptedError

DONE = "[DONE]"


async def iter_sse_data(lines: AsyncIterable[str]) -> AsyncIterator[str]:
    """Yield the ``data`` payload of each server-sent event (multi-line data joined by \\n)."""
    data: list[str] = []
    async for raw in lines:
        line = raw.rstrip("\r")
        if not line:
            if data:
                yield "\n".join(data)
                data = []
            continue
        if line.startswith(":"):  # comment / keep-alive
            continue
        field, _, value = line.partition(":")
        if field == "data":
            data.append(value.removeprefix(" "))
    if data:
        yield "\n".join(data)


async def iter_sse_events(lines: AsyncIterable[str]) -> AsyncIterator[tuple[str, str]]:
    """Yield ``(event, data)`` for each server-sent event; ``event`` defaults to "message"."""
    event, data = "message", list[str]()
    async for raw in lines:
        line = raw.rstrip("\r")
        if not line:
            if data:
                yield event, "\n".join(data)
            event, data = "message", []
            continue
        if line.startswith(":"):
            continue
        name, _, value = line.partition(":")
        value = value.removeprefix(" ")
        if name == "event":
            event = value
        elif name == "data":
            data.append(value)
    if data:
        yield event, "\n".join(data)


async def iter_chat_deltas(backend: str, lines: AsyncIterable[str]) -> AsyncIterator[ChatDelta]:
    finished = False
    chunks = 0
    async for data in iter_sse_data(lines):
        if data == DONE:
            if not finished:
                # The server ended the stream itself, just without a reason.
                yield ChatDelta(finish_reason="stop")
            return
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError as exc:
            raise EngineError(f"{backend} sent malformed stream data: {data[:200]!r}") from exc
        if not isinstance(chunk, dict):
            raise EngineError(f"{backend} sent an unexpected stream chunk: {data[:200]!r}")
        if "error" in chunk:
            raise EngineError(f"{backend} reported an error mid-stream: {_error_text(chunk)}")
        delta = parse_chunk(chunk)
        finished = finished or delta.finish_reason is not None
        chunks += 1
        yield delta
    if not finished:
        raise EngineInterruptedError(
            f"{backend} closed the stream before finishing the answer (after {chunks} chunks)",
            backend=backend,
        )


def parse_chunk(chunk: dict[str, Any]) -> ChatDelta:
    choices = chunk.get("choices") or []
    choice: dict[str, Any] = choices[0] if choices else {}
    delta: dict[str, Any] = choice.get("delta") or {}
    fragments = []
    for position, call in enumerate(delta.get("tool_calls") or []):
        function = call.get("function") or {}
        fragments.append(
            ToolCallFragment(
                index=call.get("index", position),
                id=call.get("id") or None,
                name=function.get("name") or None,
                arguments=function.get("arguments") or "",
            )
        )
    usage = chunk.get("usage")
    timings = chunk.get("timings")
    if timings is not None:
        usage = {**(usage or {}), "timings": timings}
    return ChatDelta(
        content=delta.get("content") or None,
        reasoning=delta.get("reasoning_content") or None,
        tool_calls=tuple(fragments),
        finish_reason=choice.get("finish_reason"),
        usage=usage,
    )


def _error_text(chunk: dict[str, Any]) -> str:
    error = chunk["error"]
    if isinstance(error, dict):
        return str(error.get("message") or error)
    return str(error)
