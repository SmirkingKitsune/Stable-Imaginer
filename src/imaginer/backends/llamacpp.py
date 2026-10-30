"""llama.cpp ``llama-server`` adapter (OpenAI-compatible chat completions).

Capabilities are probed, not assumed. ``GET /props`` reports
``chat_template_caps.supports_tools`` (from the model's Jinja template; ``--jinja`` is on by
default in current llama-server) and ``modalities.vision`` (only when started with
``--mmproj``). Configured ``declared_capabilities`` can only narrow that.
"""

import time
from collections.abc import AsyncIterator
from pathlib import PurePath
from typing import Any

from imaginer.backends.base import BackendHealth, ChatDelta, ChatRequest
from imaginer.backends.openai_http.chat import iter_chat_deltas
from imaginer.backends.openai_http.transport import HttpTransport
from imaginer.core.clock import utcnow
from imaginer.core.config import LlamaCppBackendConfig
from imaginer.domain.enums import BackendStatus, Capability
from imaginer.domain.errors import EngineError, ImaginerError

HEALTH_TIMEOUT_S = 5.0
SUPPORTED: frozenset[Capability] = frozenset(
    {Capability.CHAT, Capability.STREAM, Capability.TOOLS, Capability.VISION_IN}
)


def expected_capabilities(config: LlamaCppBackendConfig) -> set[Capability]:
    """What the engine its launch block starts will report once loaded, before it can be
    probed: tool calling is assumed (the template decides), vision needs ``--mmproj``."""
    offered = set(SUPPORTED)
    if config.launch is None or config.launch.mmproj is None:
        offered.discard(Capability.VISION_IN)
    declared = config.declared_capabilities
    return offered & declared if declared is not None else offered


class LlamaCppBackend:
    def __init__(
        self,
        name: str,
        config: LlamaCppBackendConfig,
        *,
        transport: HttpTransport | None = None,
    ) -> None:
        self.name = name
        self._config = config
        self._transport = transport or HttpTransport(
            name=name,
            base_url=config.base_url,
            timeouts=config.timeouts,
            retry=config.retry,
            api_key=config.api_key(),
        )
        declared = config.declared_capabilities
        self._declared = SUPPORTED & declared if declared is not None else SUPPORTED
        self._probed: set[Capability] = set()
        self._model = config.model or "unknown"

    @property
    def model(self) -> str:
        return self._model

    def capabilities(self) -> set[Capability]:
        return set(self._declared & self._probed)

    async def health(self) -> BackendHealth:
        started = time.perf_counter()
        try:
            response = await self._transport.request(
                "GET",
                "/health",
                timeout_s=HEALTH_TIMEOUT_S,
                attempts=1,
                accept_status=frozenset({503}),
            )
            if response.status_code == 503:
                return self._health(BackendStatus.DEGRADED, started, "model is loading")
            props = await self._transport.get_json("/props", timeout_s=HEALTH_TIMEOUT_S, attempts=1)
        except ImaginerError as exc:
            self._probed = set()
            return self._health(BackendStatus.DOWN, started, exc.detail)
        if not isinstance(props, dict):
            return self._health(BackendStatus.DEGRADED, started, "unexpected /props response")
        self._probed = _capabilities_from_props(props)
        self._model = self._config.model or _model_name(props) or self._model
        missing = sorted(SUPPORTED - self._probed)
        detail = f"model {self._model}"
        if missing:
            detail += f"; engine lacks: {', '.join(missing)}"
        return self._health(BackendStatus.OK, started, detail)

    async def chat_stream(self, req: ChatRequest) -> AsyncIterator[ChatDelta]:
        body: dict[str, Any] = {**self._config.sampling, **req.params}
        body["messages"] = req.messages
        body["stream"] = True
        if self._config.model:
            body["model"] = self._config.model
        if req.tools:
            body["tools"] = req.tools
            body["parallel_tool_calls"] = False  # one tool at a time keeps turns simple
            if req.tool_choice is not None:
                body["tool_choice"] = req.tool_choice
        async with self._transport.stream("POST", "/v1/chat/completions", json=body) as response:
            content_type = response.headers.get("content-type", "")
            if "text/event-stream" not in content_type:
                raise EngineError(
                    f"{self.name} did not stream (content-type {content_type or 'missing'})"
                )
            async for delta in iter_chat_deltas(self.name, response.aiter_lines()):
                yield delta

    async def aclose(self) -> None:
        await self._transport.aclose()

    def _health(self, status: BackendStatus, started: float, detail: str) -> BackendHealth:
        return BackendHealth(
            status=status,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
            detail=detail,
            probed_at=utcnow(),
        )


def _capabilities_from_props(props: dict[str, Any]) -> set[Capability]:
    caps = {Capability.CHAT, Capability.STREAM}
    template_caps = props.get("chat_template_caps") or {}
    if template_caps.get("supports_tools") and template_caps.get("supports_tool_calls", True):
        caps.add(Capability.TOOLS)
    modalities = props.get("modalities") or {}
    if modalities.get("vision"):
        caps.add(Capability.VISION_IN)
    return caps


def _model_name(props: dict[str, Any]) -> str | None:
    alias = props.get("model_alias")
    if isinstance(alias, str) and alias:
        return alias
    path = props.get("model_path")
    if isinstance(path, str) and path:
        # Paths may come from either OS; take the last component either way.
        return PurePath(path.replace("\\", "/")).name
    return None
