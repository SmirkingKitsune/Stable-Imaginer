"""Engines a linked Imaginer instance shares, over its peer API (docs/linked-instances.md).

The whole model runs on that instance: requests go to ``/v1/peer/engines/{name}/chat`` or
``/images`` and come back as server-sent events. Health and capabilities come from the
peer's status, which the link service polls; the adapter only narrows them by
``declared_capabilities``.

A stream is never replayed once it has started: retries happen only until response
headers arrive (``HttpTransport.stream``), so a reply can't be duplicated. Cancelling the
caller closes the stream, and the peer then cancels its own engine job.
"""

import base64
import json
import time
from collections.abc import AsyncIterator
from typing import Any

from imaginer.backends.base import (
    BackendHealth,
    ChatDelta,
    ChatRequest,
    EditRequest,
    GenProgress,
    GenRequest,
    ImageInput,
    ImageResult,
    ToolCallFragment,
)
from imaginer.backends.openai_http.chat import iter_sse_events
from imaginer.backends.openai_http.transport import HttpTransport
from imaginer.core.clock import utcnow
from imaginer.core.config import LinkedBackendConfig
from imaginer.domain import models as dto
from imaginer.domain.enums import BackendKind, BackendStatus, Capability
from imaginer.domain.errors import (
    CapabilityUnavailableError,
    EngineError,
    EngineInterruptedError,
    EngineTimeoutError,
    EngineUnavailableError,
    ImaginerError,
    InvalidRequestError,
)
from imaginer.links.state import Link, LinkDirectory

# The registry already re-probes only every probe_ttl_s; backends on one link that are
# probed at the same moment still share a single request (LinkService.poll).
STATUS_MAX_AGE_S = 0.0
_ERRORS: dict[str, type[ImaginerError]] = {
    EngineUnavailableError.code: EngineUnavailableError,
    EngineTimeoutError.code: EngineTimeoutError,
    EngineInterruptedError.code: EngineInterruptedError,
    CapabilityUnavailableError.code: CapabilityUnavailableError,
    InvalidRequestError.code: InvalidRequestError,
}


class _LinkedBackend:
    def __init__(self, name: str, config: LinkedBackendConfig, links: LinkDirectory) -> None:
        self.name = name
        self._config = config
        self._links = links

    @property
    def model(self) -> str:
        engine = self._engine()
        return engine.model or "unknown" if engine is not None else "unknown"

    def capabilities(self) -> set[Capability]:
        engine = self._engine()
        if engine is None:
            return set()
        offered = set(engine.capabilities)
        declared = self._config.declared_capabilities
        return offered & declared if declared is not None else offered

    async def health(self) -> BackendHealth:
        started = time.perf_counter()
        link = await self._links.fresh(self._config.link, STATUS_MAX_AGE_S)
        status, detail = _health(self._config, link)
        return BackendHealth(
            status=status,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
            detail=detail,
            probed_at=utcnow(),
        )

    async def aclose(self) -> None:
        """The transport belongs to the link, not to this adapter."""

    def _engine(self) -> dto.SharedEngine | None:
        link = self._links.by_name(self._config.link)
        return link.engine(self._config.engine) if link is not None else None

    def _transport(self) -> tuple[Link, HttpTransport]:
        link = self._links.by_name(self._config.link)
        if link is None:
            raise EngineUnavailableError(
                f"{self.name}: no link named '{self._config.link}'", backend=self.name
            )
        return link, link.transport

    def _path(self, what: str) -> str:
        return f"/v1/peer/engines/{self._config.engine}/{what}"


class LinkedLLMBackend(_LinkedBackend):
    async def chat_stream(self, req: ChatRequest) -> AsyncIterator[ChatDelta]:
        link, transport = self._transport()
        body = {
            "messages": req.messages,
            "tools": req.tools,
            "tool_choice": req.tool_choice,
            "params": req.params,
        }
        async with transport.stream("POST", self._path("chat"), json=body) as response:
            async for event, data in iter_sse_events(response.aiter_lines()):
                if event == "delta":
                    yield _delta(json.loads(data))
                elif event == "error":
                    raise _error(link, self.name, json.loads(data))
                elif event == "done":
                    return
        raise EngineInterruptedError(
            f"{self.name}: {link.peer_name} closed the stream before the answer was finished",
            backend=self.name,
        )


class LinkedImageBackend(_LinkedBackend):
    @property
    def max_ref_images(self) -> int:
        engine = self._engine()
        return engine.max_ref_images or 1 if engine is not None else 1

    @property
    def image_defaults(self) -> dto.ImageDefaultsInfo | None:
        engine = self._engine()
        return engine.image_defaults if engine is not None else None

    async def generate(self, req: GenRequest) -> list[ImageResult]:
        return await self._run("generate", req, (), None)

    async def edit(self, req: EditRequest) -> list[ImageResult]:
        return await self._run("edit", req, req.images, req.mask)

    async def _run(
        self,
        op: str,
        req: GenRequest,
        images: tuple[ImageInput, ...],
        mask: ImageInput | None,
    ) -> list[ImageResult]:
        link, transport = self._transport()
        body = {
            "op": op,
            "prompt": req.prompt,
            "seed": req.seed,
            "negative_prompt": req.negative_prompt,
            "width": req.width,
            "height": req.height,
            "aspect_ratio": req.aspect_ratio,
            "alpha": req.alpha,
            "params": req.params,
            "images": [_image(i) for i in images],
            "mask": _image(mask) if mask is not None else None,
        }
        results: list[ImageResult] = []
        async with transport.stream("POST", self._path("images"), json=body) as response:
            async for event, data in iter_sse_events(response.aiter_lines()):
                payload = json.loads(data)
                if event == "submitted" and req.on_submitted is not None:
                    await req.on_submitted(f"{link.peer_name}:{payload['request_id']}")
                elif event == "progress" and req.on_progress is not None:
                    await req.on_progress(
                        GenProgress(step=payload["step"], total_steps=payload["total_steps"])
                    )
                elif event == "result":
                    results.append(_result(payload))
                elif event == "error":
                    raise _error(link, self.name, payload)
                elif event == "done":
                    return results
        raise EngineInterruptedError(
            f"{self.name}: {link.peer_name} closed the stream before the image was finished",
            backend=self.name,
        )


def _health(config: LinkedBackendConfig, link: Link | None) -> tuple[BackendStatus, str]:
    if link is None:
        return BackendStatus.DOWN, f"not linked to '{config.link}': pair on the Links screen"
    if link.status != "ok":
        return BackendStatus.DOWN, link.detail or f"{link.peer_name} is {link.status}"
    engine = link.engine(config.engine)
    if engine is None:
        return BackendStatus.DOWN, f"{link.peer_name} doesn't share '{config.engine}'"
    if engine.kind is not config.kind:
        return BackendStatus.DOWN, (
            f"'{config.engine}' on {link.peer_name} is a {engine.kind.value} backend, "
            f"not {config.kind.value}"
        )
    where = f"on {link.peer_name}"
    availability = engine.availability
    if availability.state == "ready":
        return BackendStatus.OK, f"model {engine.model or 'unknown'} {where}"
    if availability.state == "cold":
        return BackendStatus.OK, f"loads {where} on first use"
    return BackendStatus.DOWN, f"{where}: {availability.detail or availability.state}"


def _error(link: Link, backend: str, problem: dict[str, Any]) -> ImaginerError:
    """The peer's problem details as the same kind of error here."""
    cls = _ERRORS.get(str(problem.get("code")), EngineError)
    detail = f"{link.peer_name}: {problem.get('detail') or problem.get('title') or 'error'}"
    return cls(detail, backend=backend)


def _delta(data: dict[str, Any]) -> ChatDelta:
    return ChatDelta(
        content=data.get("content"),
        reasoning=data.get("reasoning"),
        tool_calls=tuple(
            ToolCallFragment(
                index=int(f["index"]),
                id=f.get("id"),
                name=f.get("name"),
                arguments=f.get("arguments") or "",
            )
            for f in data.get("tool_calls") or ()
        ),
        finish_reason=data.get("finish_reason"),
        usage=data.get("usage"),
    )


def _image(image: ImageInput) -> dict[str, Any]:
    return {
        "data_b64": base64.b64encode(image.data).decode(),
        "mime": image.mime,
        "label": image.label,
        "width": image.width,
        "height": image.height,
    }


def _result(data: dict[str, Any]) -> ImageResult:
    return ImageResult(
        data=base64.b64decode(data["data_b64"]),
        mime=data["mime"],
        width=int(data["width"]),
        height=int(data["height"]),
        has_alpha=bool(data["has_alpha"]),
        seed=int(data["seed"]),
        params=data.get("params") or {},
        engine_meta=data.get("engine_meta") or {},
    )


def linked_backend(
    name: str, config: LinkedBackendConfig, links: LinkDirectory
) -> LinkedLLMBackend | LinkedImageBackend:
    if config.kind is BackendKind.LLM:
        return LinkedLLMBackend(name, config, links)
    return LinkedImageBackend(name, config, links)
