"""Reassembles a streamed assistant message from ``ChatDelta`` pieces."""

from dataclasses import dataclass, field
from typing import Any

from imaginer.backends.base import ChatDelta
from imaginer.core.ids import new_id


@dataclass
class _ToolCallBuffer:
    id: str | None = None
    name: str = ""
    arguments: list[str] = field(default_factory=list)


class MessageAssembler:
    def __init__(self) -> None:
        self._content: list[str] = []
        self._reasoning: list[str] = []
        self._calls: dict[int, _ToolCallBuffer] = {}
        self.finish_reason: str | None = None
        self.usage: dict[str, Any] | None = None

    def add(self, delta: ChatDelta) -> None:
        if delta.content:
            self._content.append(delta.content)
        if delta.reasoning:
            self._reasoning.append(delta.reasoning)
        for fragment in delta.tool_calls:
            buffer = self._calls.setdefault(fragment.index, _ToolCallBuffer())
            if fragment.id:
                buffer.id = fragment.id
            if fragment.name:
                # Replace, don't append: llama.cpp re-sends the whole (longer) name when its
                # incremental parser revises it. OpenAI sends the name once. Both work.
                buffer.name = fragment.name
            buffer.arguments.append(fragment.arguments)
        if delta.finish_reason:
            self.finish_reason = delta.finish_reason
        if delta.usage:
            self.usage = {**(self.usage or {}), **delta.usage}

    @property
    def text(self) -> str:
        return "".join(self._content)

    @property
    def reasoning(self) -> str | None:
        return "".join(self._reasoning) or None

    def tool_calls(self) -> list[dict[str, Any]]:
        """OpenAI-format tool calls, in index order. Missing ids get a generated one."""
        calls = []
        for index in sorted(self._calls):
            buffer = self._calls[index]
            if buffer.id is None:
                buffer.id = f"call_{new_id().replace('-', '')[:24]}"
            calls.append(
                {
                    "id": buffer.id,
                    "type": "function",
                    "function": {"name": buffer.name, "arguments": "".join(buffer.arguments)},
                }
            )
        return calls
