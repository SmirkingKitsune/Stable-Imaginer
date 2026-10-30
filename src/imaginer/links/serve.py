"""The engines this instance shares with linked instances: the peer API's backend.

Only engines named in ``links.share`` are served, and never a ``linked`` one, so requests
can't hop on to a third instance. Image requests wait at the same gate as this instance's
own jobs. A stopped engine is started for a linked instance only when its share allows
``remote_start`` and the model fits; the request then reports ``loading`` until it is ready.
"""

import asyncio
import base64
import contextlib
import time
from collections.abc import AsyncIterator
from typing import Any, Literal

from fastapi.encoders import jsonable_encoder

from imaginer.backends import llamacpp, sdcpp
from imaginer.backends.base import (
    ChatDelta,
    ChatRequest,
    EditRequest,
    GenProgress,
    GenRequest,
    ImageInput,
    ImageResult,
)
from imaginer.backends.registry import BackendRegistry
from imaginer.core.config import (
    BackendConfig,
    LinkedBackendConfig,
    LlamaCppBackendConfig,
    SdCppBackendConfig,
    Settings,
    backend_kind,
)
from imaginer.core.ids import new_id
from imaginer.domain import models as dto
from imaginer.domain.enums import BackendKind, BackendStatus, Capability
from imaginer.domain.errors import (
    ConflictError,
    EngineTimeoutError,
    EngineUnavailableError,
    InvalidRequestError,
    NotFoundError,
)
from imaginer.engines import launch
from imaginer.engines.gpu import gpu_memory
from imaginer.engines.service import EngineService
from imaginer.links import fit
from imaginer.orchestration.gates import EngineGates

Event = tuple[str, dict[str, Any]]
START_POLL_S = 1.0


class PeerEngines:
    def __init__(
        self,
        settings: Settings,
        registry: BackendRegistry,
        engines: EngineService,
        gates: EngineGates,
    ) -> None:
        self._settings = settings
        self._registry = registry
        self._engines = engines
        self._gates = gates
        self._starting: dict[str, asyncio.Lock] = {}

    def shared(self, name: str, kind: BackendKind | None = None) -> BackendConfig:
        config = self._settings.backends.get(name)
        if (
            config is None
            or name not in self._settings.links.shared()
            or isinstance(config, LinkedBackendConfig)
        ):
            raise NotFoundError(f"this instance doesn't share an engine named '{name}'")
        if kind is not None and backend_kind(config) is not kind:
            raise InvalidRequestError(f"'{name}' is not an {kind.value} engine")
        return config

    # --- status ----------------------------------------------------------------------

    async def status(self) -> list[dto.SharedEngine]:
        gpu = await asyncio.to_thread(gpu_memory)
        names = [
            name
            for name in self._settings.links.shared()
            if name in self._settings.backends
            and not isinstance(self._settings.backends[name], LinkedBackendConfig)
        ]
        return list(await asyncio.gather(*(self._describe(name, gpu) for name in names)))

    async def _describe(self, name: str, gpu: Any) -> dto.SharedEngine:
        config = self._settings.backends[name]
        info = await self._registry.describe_one(name)
        not_startable = await asyncio.to_thread(self._not_startable, config)
        need = await asyncio.to_thread(fit.need_mb, config)
        availability = fit.assess(
            status=info.status,
            remote_start=self._remote_start(name),
            not_startable=not_startable,
            need=need,
            gpu=gpu,
        )
        kind = backend_kind(config)
        capabilities = info.capabilities
        if availability.state == "cold" and not capabilities:
            # A stopped engine reports none, but the caller routes by them: without these
            # its first request would be refused there and never get here to start it.
            capabilities = sorted(_expected(config))
        return dto.SharedEngine(
            name=name,
            type=config.type,
            kind=kind,
            model=info.model,
            status=info.status,
            detail=info.detail,
            capabilities=capabilities,
            max_ref_images=getattr(config, "max_ref_images", None),
            image_defaults=info.image_defaults,
            queue_depth=self._gates.queue_depth(name) if kind is BackendKind.IMAGE else 0,
            availability=availability,
        )

    def _remote_start(self, name: str) -> bool:
        share = self._settings.links.shared().get(name)
        return share is not None and share.remote_start

    def _not_startable(self, config: BackendConfig) -> str | None:
        """Why this engine can't be started here, or None if it can. Blocking (disk)."""
        if not config.enabled:
            return "disabled here"
        if getattr(config, "launch", None) is None:
            return "stopped, and it isn't started by this instance"
        exe = self._engines.executable(config)
        if exe is None or not exe.is_file():
            return "stopped, and its runtime isn't installed"
        try:
            launch.command(config, exe)
        except launch.LaunchError as exc:
            return f"stopped: {exc}"
        return None

    # --- readiness -----------------------------------------------------------------------

    async def _ensure_ready(self, name: str) -> AsyncIterator[dict[str, Any]]:
        """Return once the engine is healthy, starting it if it is cold and that's allowed.
        Yields ``loading`` statuses while it loads."""
        health = await self._registry.probe(name)
        if health is not None and health.status is BackendStatus.OK:
            return
        loading = health is not None and health.status is BackendStatus.DEGRADED
        if not loading:
            config = self._settings.backends[name]
            proc = self._engines.supervisor.peek(name)
            if proc is None or not proc.alive:
                await self._start(name, config)
        yield {"state": "loading", "detail": f"loading {name}"}
        deadline = time.monotonic() + self._settings.links.start_timeout_s
        while True:
            await asyncio.sleep(START_POLL_S)
            health = await self._registry.probe(name)
            if health is not None and health.status is BackendStatus.OK:
                yield {"state": "ready"}
                return
            proc = self._engines.supervisor.peek(name)
            if proc is not None and not proc.alive:
                summary = proc.run.last_error if proc.run is not None else None
                raise EngineUnavailableError(
                    f"'{name}' stopped while loading" + (f": {summary}" if summary else ""),
                    backend=name,
                )
            if time.monotonic() > deadline:
                raise EngineTimeoutError(
                    f"'{name}' did not load within {self._settings.links.start_timeout_s:.0f} s",
                    backend=name,
                )

    async def _start(self, name: str, config: BackendConfig) -> None:
        if not self._remote_start(name):
            raise EngineUnavailableError(
                f"'{name}' is stopped, and only its owner may start it", backend=name
            )
        reason = await asyncio.to_thread(self._not_startable, config)
        if reason is not None:
            raise EngineUnavailableError(f"'{name}' is {reason}", backend=name)
        gpu = await asyncio.to_thread(gpu_memory)
        need = await asyncio.to_thread(fit.need_mb, config)
        verdict = fit.assess(
            status=BackendStatus.DOWN, remote_start=True, not_startable=None, need=need, gpu=gpu
        )
        if verdict.state == "unfit":
            raise EngineUnavailableError(f"'{name}' won't fit: {verdict.detail}", backend=name)
        async with self._starting.setdefault(name, asyncio.Lock()):
            proc = self._engines.supervisor.peek(name)
            if proc is not None and proc.alive:
                return  # another request started it while we waited
            with contextlib.suppress(ConflictError):  # started meanwhile from the Engines screen
                await self._engines.start(name)

    # --- requests ------------------------------------------------------------------------

    async def chat(self, name: str, body: dto.PeerChatRequest) -> AsyncIterator[Event]:
        async for state in self._ensure_ready(name):
            yield "status", state
        lease = await self._registry.chat_backend(name)
        try:
            request = ChatRequest(
                messages=body.messages,
                tools=body.tools,
                tool_choice=body.tool_choice,
                params=body.params,
            )
            async for delta in lease.backend.chat_stream(request):
                yield "delta", _delta(delta)
        finally:
            lease.release()
        yield "done", {}

    async def images(self, name: str, body: dto.PeerImageRequest) -> AsyncIterator[Event]:
        if body.op == "edit" and not body.images:
            raise InvalidRequestError("edit needs at least one input image")
        require: set[Capability] = set()
        if body.alpha:
            require.add(Capability.RGBA)
        if len(body.images) > 1:
            require.add(Capability.MULTI_REF)
        async for state in self._ensure_ready(name):
            yield "status", state
        task: Literal["t2i", "edit"] = "edit" if body.op == "edit" else "t2i"
        lease = await self._registry.image_backend(task, name, require=require)
        request_id = new_id()
        try:
            if self._gates.queue_depth(name):
                yield "status", {"state": "queued", "queue_depth": self._gates.queue_depth(name)}
            async with self._gates.hold(name):
                yield "submitted", {"request_id": request_id}
                yield "status", {"state": "running"}
                progress: asyncio.Queue[Event] = asyncio.Queue()

                async def on_progress(p: GenProgress) -> None:
                    progress.put_nowait(
                        ("progress", {"step": p.step, "total_steps": p.total_steps})
                    )

                request = _request(body, on_progress)
                if isinstance(request, EditRequest):
                    work = asyncio.ensure_future(lease.backend.edit(request))
                else:
                    work = asyncio.ensure_future(lease.backend.generate(request))
                try:
                    async for event in _drain(progress, work):
                        yield event
                    results = work.result()
                finally:
                    if not work.done():  # the linked instance went away: free the GPU
                        work.cancel()
                        with contextlib.suppress(asyncio.CancelledError, Exception):
                            await work
            for result in results:
                yield "result", _result(result)
        finally:
            lease.release()
        yield "done", {}


async def _drain(queue: asyncio.Queue[Event], work: asyncio.Future[Any]) -> AsyncIterator[Event]:
    """Yield queued events until ``work`` is done, then whatever is left."""
    while not work.done():
        getter = asyncio.ensure_future(queue.get())
        try:
            await asyncio.wait({work, getter}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            if not getter.done():
                getter.cancel()
        if getter.done() and not getter.cancelled():
            yield getter.result()
    while not queue.empty():
        yield queue.get_nowait()


def _expected(config: BackendConfig) -> set[Capability]:
    """What a stopped engine will offer once loaded. The loaded engine is still checked
    before each request runs on it."""
    if isinstance(config, LlamaCppBackendConfig):
        return llamacpp.expected_capabilities(config)
    if isinstance(config, SdCppBackendConfig):
        return sdcpp.expected_capabilities(config)
    return set()


def _image(image: dto.PeerImage) -> ImageInput:
    try:
        data = base64.b64decode(image.data_b64, validate=True)
    except ValueError as exc:
        raise InvalidRequestError("an input image is not valid base64") from exc
    return ImageInput(
        data=data, mime=image.mime, label=image.label, width=image.width, height=image.height
    )


def _request(body: dto.PeerImageRequest, on_progress: Any) -> GenRequest | EditRequest:
    common: dict[str, Any] = {
        "prompt": body.prompt,
        "seed": body.seed,
        "negative_prompt": body.negative_prompt,
        "width": body.width,
        "height": body.height,
        "aspect_ratio": body.aspect_ratio,
        "alpha": body.alpha,
        "params": body.params,
        "on_progress": on_progress,
    }
    if body.op == "edit":
        return EditRequest(
            images=tuple(_image(i) for i in body.images),
            mask=_image(body.mask) if body.mask is not None else None,
            **common,
        )
    return GenRequest(**common)


def _delta(delta: ChatDelta) -> dict[str, Any]:
    return {
        "content": delta.content,
        "reasoning": delta.reasoning,
        "tool_calls": [
            {"index": f.index, "id": f.id, "name": f.name, "arguments": f.arguments}
            for f in delta.tool_calls
        ],
        "finish_reason": delta.finish_reason,
        "usage": delta.usage,
    }


def _result(result: ImageResult) -> dict[str, Any]:
    return {
        "data_b64": base64.b64encode(result.data).decode(),
        "mime": result.mime,
        "width": result.width,
        "height": result.height,
        "has_alpha": result.has_alpha,
        "seed": result.seed,
        "params": jsonable_encoder(result.params),
        "engine_meta": jsonable_encoder(result.engine_meta),
    }
