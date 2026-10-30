"""The Engines screen: engine processes, model files, endpoints and runtime installs.

Starting an engine runs a local executable with paths from the config, so these routes
are as sensitive as the config itself: keep the server on loopback, or set a token.
"""

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter
from fastapi.responses import FileResponse
from sse_starlette import EventSourceResponse, ServerSentEvent

from imaginer.api.deps import ContainerDep
from imaginer.api.sse import PING_INTERVAL_S
from imaginer.domain import models as dto
from imaginer.domain.errors import NotFoundError
from imaginer.engines.installer import InstallJob
from imaginer.engines.service import _install

router = APIRouter(prefix="/v1/engines", tags=["engines"])

_STREAM_DOC: dict[int | str, dict[str, Any]] = {
    200: {"content": {"text/event-stream": {"schema": {"type": "string"}}}},
}


@router.get("")
async def list_engines(container: ContainerDep) -> list[dto.EngineInfo]:
    """Every configured backend as an engine: config, health, process, runtime, install."""
    return await container.engines.all()


@router.get("/{name}")
async def get_engine(name: str, container: ContainerDep) -> dto.EngineInfo:
    return await container.engines.get(name)


@router.patch("/{name}")
async def update_engine(
    name: str, body: dto.EngineUpdate, container: ContainerDep
) -> dto.EngineInfo:
    """Change the endpoint, capabilities or launch files. Applied at once and saved to
    ``config/<profile>.local.yaml``; the profile file itself is left as it is."""
    return await container.engines.update(name, body)


@router.post("/{name}/start")
async def start_engine(name: str, container: ContainerDep) -> dto.EngineInfo:
    """Start the engine's executable with its launch files, listening on its base_url."""
    return await container.engines.start(name)


@router.post("/{name}/stop")
async def stop_engine(name: str, container: ContainerDep) -> dto.EngineInfo:
    return await container.engines.stop(name)


@router.get("/{name}/log")
async def engine_log(
    name: str, container: ContainerDep, errors: bool = False
) -> list[dto.EngineLogLine]:
    """The last run's output (newest 2000 lines), or only the lines that look like errors."""
    return container.engines.log(name, errors_only=errors)


@router.get("/{name}/log/file", response_class=FileResponse)
async def engine_log_file(name: str, container: ContainerDep) -> FileResponse:
    path = container.engines.log_file(name)
    if path is None or not path.is_file():
        raise NotFoundError(f"{name} has no log yet")
    return FileResponse(path, media_type="text/plain; charset=utf-8", filename=path.name)


@router.get("/{name}/events", response_class=EventSourceResponse, responses=_STREAM_DOC)
async def engine_events(name: str, container: ContainerDep) -> EventSourceResponse:
    """Live `line` and `state` events from the engine process, and `install` progress."""
    service = container.engines
    await service.get(name)  # 404 for unknown names
    proc = service.supervisor.get(name)
    queue: asyncio.Queue[ServerSentEvent] = asyncio.Queue(maxsize=1000)

    def put(event: ServerSentEvent) -> None:
        if not queue.full():
            queue.put_nowait(event)

    def on_process(kind: str, payload: object) -> None:
        if kind == "line":
            line = dto.EngineLogLine(at=payload.at, text=payload.text, error=payload.error)  # type: ignore[attr-defined]
            put(ServerSentEvent(data=line.model_dump_json(), event="line"))
        else:
            put(ServerSentEvent(data=f'"{payload}"', event="state"))

    def on_install(job: InstallJob) -> None:
        info = _install(job)
        if info is not None:
            put(ServerSentEvent(data=info.model_dump_json(), event="install"))

    unsubscribe = [proc.subscribe(on_process)]
    engine_type = (await service.get(name)).type
    unsubscribe.append(service.installer.subscribe(engine_type, on_install))

    async def events() -> AsyncIterator[ServerSentEvent]:
        try:
            while True:
                yield await queue.get()
        finally:
            for off in unsubscribe:
                off()

    return EventSourceResponse(events(), ping=PING_INTERVAL_S, headers={"X-Accel-Buffering": "no"})


@router.post("/{name}/install/plan")
async def install_plan(
    name: str, body: dto.EngineInstallRequest, container: ContainerDep
) -> dict[str, Any]:
    """What an install would do, in order, and the hardware it detected. Nothing is
    downloaded."""
    return await container.engines.install_plan(name, body)


@router.post("/{name}/install")
async def install_runtime(
    name: str, body: dto.EngineInstallRequest, container: ContainerDep
) -> dto.EngineInfo:
    """Download (or build) the engine's runtime in the background; follow /events."""
    return await container.engines.install(name, body)


@router.post("/{name}/install/cancel")
async def cancel_install(name: str, container: ContainerDep) -> dto.EngineInfo:
    return await container.engines.cancel_install(name)


@router.get("/{name}/update")
async def check_update(name: str, container: ContainerDep) -> dict[str, Any]:
    """Whether a newer runtime release exists. Asks GitHub; never installs."""
    return await container.engines.check_update(name)
