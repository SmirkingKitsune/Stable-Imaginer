"""The Hugging Face Hub: search, repo layouts matched to an engine, and model downloads.

Downloads write to this machine's models folder and can repoint an engine, so these routes
are for the user only, like the Engines screen's.
"""

import asyncio
from collections.abc import AsyncIterator
from typing import Any, Literal

from fastapi import APIRouter, Query, status
from sse_starlette import EventSourceResponse, ServerSentEvent

from imaginer.api.deps import ContainerDep
from imaginer.api.sse import PING_INTERVAL_S
from imaginer.domain import models as dto
from imaginer.hub.downloads import DownloadJob
from imaginer.hub.service import download_view

router = APIRouter(prefix="/v1/hub", tags=["hub"])

_STREAM_DOC: dict[int | str, dict[str, Any]] = {
    200: {"content": {"text/event-stream": {"schema": {"type": "string"}}}},
}


@router.get("")
async def hub_status(container: ContainerDep) -> dto.HubStatus:
    """The endpoint, whether a Hub token is set (never the token), and where models go."""
    return container.hub.status()


@router.patch("")
async def update_hub(body: dto.HubSettingsUpdate, container: ContainerDep) -> dto.HubStatus:
    """Change where downloads go. Saved to ``config/<profile>.local.yaml``."""
    return await container.hub.update(body)


@router.get("/search")
async def search(
    container: ContainerDep,
    q: str = Query(default="", max_length=200),
    format: Literal["gguf", "diffusers", "safetensors", "any"] = "any",
    base_model: str | None = Query(
        default=None, max_length=300, description="Repos derived from this one."
    ),
    relation: Literal["quantized", "finetune", "adapter", "merge"] | None = None,
    task: str | None = Query(
        default=None, max_length=64, pattern=r"^[a-z0-9-]+$", description="A pipeline tag."
    ),
    sort: Literal["downloads", "trending", "likes", "updated"] = "downloads",
    limit: int = Query(default=30, ge=1, le=100),
) -> list[dto.HubRepoSummary]:
    """Model repos on the Hub. ``base_model`` with ``relation=quantized`` lists a model's
    GGUF and other quantized versions."""
    return await container.hub.search(
        q, fmt=format, base_model=base_model, relation=relation, task=task, sort=sort, limit=limit
    )


@router.get("/repo")
async def repo(
    container: ContainerDep,
    ref: str = Query(
        min_length=3,
        max_length=500,
        description="owner/name, owner/name:QUANT, a Hub URL or an hf:// path.",
    ),
    engine: str | None = Query(default=None, description="Match the files to this engine."),
) -> dto.HubRepo:
    """A repo's layout: its loadable files (quants, shards, projectors, VAEs, text encoders)
    or Diffusers pipeline, what is already on this machine, and, for an engine, which file
    can fill each of its settings."""
    return await container.hub.repo(ref, engine=engine)


@router.get("/downloads")
async def list_downloads(container: ContainerDep) -> list[dto.HubDownload]:
    return container.hub.list_downloads()


@router.post("/downloads", status_code=status.HTTP_202_ACCEPTED)
async def start_download(body: dto.HubDownloadRequest, container: ContainerDep) -> dto.HubDownload:
    """Download a variant (or a Diffusers pipeline) into the models folder in the background;
    with ``engine`` and ``key``, that setting points at it once it's done. Follow /events."""
    return await container.hub.download(body)


@router.post("/downloads/{download_id}/cancel")
async def cancel_download(download_id: str, container: ContainerDep) -> dto.HubDownload:
    """Stop a download. What it got so far is kept, and resumes when it's asked for again."""
    return container.hub.cancel(download_id)


@router.get("/events", response_class=EventSourceResponse, responses=_STREAM_DOC)
async def hub_events(container: ContainerDep) -> EventSourceResponse:
    """A `download` event whenever a download progresses or ends. Downloads still running
    when the stream opens are sent first, as they stand."""
    manager = container.hub.downloads
    queue: asyncio.Queue[ServerSentEvent] = asyncio.Queue(maxsize=1000)

    def put(job: DownloadJob) -> None:
        if not queue.full():
            queue.put_nowait(
                ServerSentEvent(data=download_view(job).model_dump_json(), event="download")
            )

    unsubscribe = manager.subscribe(put)
    for job in manager.jobs():
        if job.active:
            put(job)

    async def events() -> AsyncIterator[ServerSentEvent]:
        try:
            while True:
                yield await queue.get()
        finally:
            unsubscribe()

    return EventSourceResponse(events(), ping=PING_INTERVAL_S, headers={"X-Accel-Buffering": "no"})
