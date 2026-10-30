"""Tool plumbing: schemas for the LLM, argument validation, and error containment.

A tool call never crashes a turn. Bad JSON, invalid arguments, a missing capability or an
engine failure all come back to the LLM as a structured error result that it can explain
to the user or correct and retry.
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

from pydantic import BaseModel, ValidationError

from imaginer.core.logging import get_logger
from imaginer.domain.errors import ImaginerError
from imaginer.domain.models import ImageRequestOptions
from imaginer.orchestration.jobs import Emit

log = get_logger(__name__)


@dataclass(frozen=True, kw_only=True)
class ToolContext:
    conversation_id: str
    message_id: str  # the assistant message that made the call
    tool_call_id: str
    emit: Emit
    image_options: ImageRequestOptions | None = None  # the user's picks for this turn


@dataclass(kw_only=True)
class ToolOutcome:
    content: dict[str, Any]  # the JSON result shown to the LLM
    artifact_ids: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.content.get("status") != "error"

    @classmethod
    def error(cls, code: str, detail: str, **extra: Any) -> "ToolOutcome":
        return cls(content={"status": "error", "error": {"code": code, "detail": detail, **extra}})


class Tool(ABC):
    name: ClassVar[str]
    description: ClassVar[str]
    guidance: ClassVar[str]  # one line for the system prompt when the tool is offered
    args_model: ClassVar[type[BaseModel]]

    @abstractmethod
    def available(self) -> bool:
        """Whether to offer the tool at all (some backend could serve it)."""

    @abstractmethod
    async def run(self, args: Any, ctx: ToolContext) -> ToolOutcome: ...

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": json_schema(self.args_model),
            },
        }


class ToolRegistry:
    def __init__(self, tools: list[Tool]) -> None:
        self._tools = {tool.name: tool for tool in tools}

    def offered(self) -> list[Tool]:
        return [tool for tool in self._tools.values() if tool.available()]

    async def execute(self, name: str, arguments: str, ctx: ToolContext) -> ToolOutcome:
        tool = self._tools.get(name)
        if tool is None or not tool.available():
            return ToolOutcome.error("unknown_tool", f"there is no tool named {name!r}")
        try:
            raw = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError as exc:
            return ToolOutcome.error("invalid_arguments", f"arguments are not valid JSON: {exc}")
        try:
            args = tool.args_model.model_validate(raw)
        except ValidationError as exc:
            problems = [
                f"{'.'.join(str(p) for p in e['loc']) or '(root)'}: {e['msg']}"
                for e in exc.errors(include_url=False)
            ]
            return ToolOutcome.error("invalid_arguments", "; ".join(problems))
        try:
            return await tool.run(args, ctx)
        except ImaginerError as exc:
            return ToolOutcome.error(exc.code, exc.detail)
        except Exception:
            log.exception("tool.crashed", tool=name)
            return ToolOutcome.error("internal_error", "the tool failed unexpectedly")


def json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """The model's JSON schema without pydantic's ``title`` noise."""
    schema: dict[str, Any] = _strip_titles(model.model_json_schema())
    return schema


def _strip_titles(node: Any) -> Any:
    if isinstance(node, dict):
        return {
            k: _strip_titles(v)
            for k, v in node.items()
            if not (k == "title" and isinstance(v, str))
        }
    if isinstance(node, list):
        return [_strip_titles(v) for v in node]
    return node
