"""Backend protocols and the request/result types that cross them.

The orchestrator only ever talks to these protocols. Which engine, process or OS sits
behind them is configuration.
"""

from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from imaginer.domain.enums import BackendStatus, Capability

__all__ = [
    "BackendHealth",
    "Capability",
    "ChatDelta",
    "ChatRequest",
    "EditRequest",
    "GenProgress",
    "GenRequest",
    "ImageBackend",
    "ImageInput",
    "ImageResult",
    "LLMBackend",
    "ProgressCallback",
    "SubmittedCallback",
    "ToolCallFragment",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class BackendHealth:
    status: BackendStatus
    latency_ms: float | None = None
    detail: str | None = None
    probed_at: datetime | None = None


# --- images -------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class ImageInput:
    data: bytes
    mime: str
    label: str | None = None
    width: int | None = None
    height: int | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class GenProgress:
    step: int
    total_steps: int


ProgressCallback = Callable[[GenProgress], Awaitable[None]]
SubmittedCallback = Callable[[str], Awaitable[None]]  # receives the engine's own job id


@dataclass(frozen=True, slots=True, kw_only=True)
class GenRequest:
    """One image. Backends that batch internally still get one request per output, so each
    output has its own seed and can be reproduced on its own."""

    prompt: str
    seed: int  # always concrete: the orchestrator picks one so every output is reproducible
    negative_prompt: str | None = None
    width: int | None = None  # None: from aspect_ratio, else the backend's configured default
    height: int | None = None
    aspect_ratio: str | None = None  # one of sizing.ASPECT_RATIOS
    alpha: bool = False  # transparent (RGBA) output; requires Capability.RGBA
    params: dict[str, Any] = field(default_factory=dict)  # per-request sampler overrides
    on_progress: ProgressCallback | None = None
    on_submitted: SubmittedCallback | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class EditRequest(GenRequest):
    images: tuple[ImageInput, ...]  # ordered; more than one requires Capability.MULTI_REF
    mask: ImageInput | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class ImageResult:
    data: bytes
    mime: str
    width: int
    height: int
    has_alpha: bool
    seed: int  # the seed the engine actually used
    params: dict[str, Any] = field(default_factory=dict)  # resolved settings, for reproduction
    engine_meta: dict[str, Any] = field(default_factory=dict)


# --- chat ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class ChatRequest:
    messages: list[dict[str, Any]]  # OpenAI chat-completions message objects
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    params: dict[str, Any] = field(default_factory=dict)  # temperature, max_tokens, ...


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolCallFragment:
    """One streamed piece of a tool call. Fragments with the same index belong together."""

    index: int
    id: str | None = None
    name: str | None = None
    arguments: str = ""


@dataclass(frozen=True, slots=True, kw_only=True)
class ChatDelta:
    content: str | None = None
    reasoning: str | None = None
    tool_calls: tuple[ToolCallFragment, ...] = ()
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None


# --- protocols ----------------------------------------------------------------


class LLMBackend(Protocol):
    name: str

    @property
    def model(self) -> str:
        """Model identifier as reported by the engine (or configured)."""
        ...

    def capabilities(self) -> set[Capability]:
        """Cached: declared capabilities narrowed by the last successful probe."""
        ...

    def chat_stream(self, req: ChatRequest) -> AsyncIterator[ChatDelta]: ...

    async def health(self) -> BackendHealth:
        """Probe the engine. Also refreshes the cached capabilities."""
        ...

    async def aclose(self) -> None: ...


class ImageBackend(Protocol):
    name: str

    @property
    def model(self) -> str: ...

    @property
    def max_ref_images(self) -> int: ...

    def capabilities(self) -> set[Capability]: ...

    async def generate(self, req: GenRequest) -> list[ImageResult]: ...

    async def edit(self, req: EditRequest) -> list[ImageResult]: ...

    async def health(self) -> BackendHealth: ...

    async def aclose(self) -> None: ...
