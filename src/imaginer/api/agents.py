"""The agent library (the web UI's #/agents): an agent per chat model, with its persona.

An agent is a llama-server model (its GGUF, vision projector and chat template) on the chat
engine it runs on, plus a persona: name, system prompt, portrait, tools and sampling. Using
one on an engine launched here restarts it, so these routes are for the user only.
"""

from typing import Annotated

from fastapi import APIRouter, File, Response, UploadFile, status
from fastapi.responses import FileResponse

from imaginer.agents.portraits import PORTRAIT_MAX_BYTES
from imaginer.api.deps import ContainerDep
from imaginer.domain import models as dto
from imaginer.domain.errors import NotFoundError, PayloadTooLargeError

router = APIRouter(prefix="/v1/agents", tags=["agents"])


@router.get("")
async def list_agents(container: ContainerDep) -> dto.AgentLibraryView:
    """Every agent, with its engine, whether it is ready and running, its portrait and how
    much it has been used; the chat engines; the agent new prompts go to."""
    return await container.agents.overview()


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_agent(body: dto.AgentCreate, container: ContainerDep) -> dto.AgentDetail:
    """A new agent: from a GGUF here, from a Hub reference (download its files into it
    next), or for whatever an engine runs."""
    return await container.agents.create(body)


@router.get("/{agent_id}")
async def get_agent(agent_id: str, container: ContainerDep) -> dto.AgentDetail:
    """One agent, with its files checked on disk, its system prompt, tools and sampling."""
    return await container.agents.get(agent_id)


@router.patch("/{agent_id}")
async def update_agent(
    agent_id: str, body: dto.AgentUpdate, container: ContainerDep
) -> dto.AgentDetail:
    """Change its name, engine, files, system prompt, viewer, tools, sampling, context size,
    notes, tags or pin. An engine running it takes new files when it is restarted."""
    return await container.agents.update(agent_id, body)


@router.delete("/{agent_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_agent(agent_id: str, container: ContainerDep) -> Response:
    """Forget the agent. Its files and the messages it wrote stay. Not while an engine runs
    it."""
    await container.agents.delete(agent_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{agent_id}/use")
async def use_agent(agent_id: str, body: dto.AgentUse, container: ContainerDep) -> dto.AgentDetail:
    """Run its model on its engine: one launched here restarts with it (409 while a turn
    uses it)."""
    return await container.agents.use(agent_id, body)


@router.post("/{agent_id}/stop")
async def stop_agent(agent_id: str, container: ContainerDep) -> dto.AgentDetail:
    """Stop its engine (one launched here, running it)."""
    return await container.agents.stop(agent_id)


@router.put("/{agent_id}/portrait")
async def set_agent_portrait(
    agent_id: str, body: dto.AgentPortraitSet, container: ContainerDep
) -> dto.AgentDetail:
    """Show one of its portraits, an image from a conversation, its Hub uploader's avatar
    fetched again, or none (its monogram)."""
    return await container.agents.set_portrait(agent_id, body)


@router.post("/{agent_id}/portrait/upload")
async def upload_agent_portrait(
    agent_id: str,
    container: ContainerDep,
    file: Annotated[UploadFile, File(description="An image; it is cropped to a square.")],
) -> dto.AgentDetail:
    if file.size is not None and file.size > PORTRAIT_MAX_BYTES:
        raise PayloadTooLargeError(f"a portrait is at most {PORTRAIT_MAX_BYTES} bytes")
    data = await file.read(PORTRAIT_MAX_BYTES + 1)
    if len(data) > PORTRAIT_MAX_BYTES:
        raise PayloadTooLargeError(f"a portrait is at most {PORTRAIT_MAX_BYTES} bytes")
    return await container.agents.upload_portrait(agent_id, data)


@router.post("/{agent_id}/portrait/paint", status_code=status.HTTP_202_ACCEPTED)
async def paint_agent_portrait(agent_id: str, container: ContainerDep) -> dto.AgentPainted:
    """Have it paint another self-portrait: a new conversation in which it writes its own
    prompt and calls set_portrait. Follow the conversation's turn for its progress."""
    return await container.agents.paint(agent_id)


@router.get("/{agent_id}/portraits/{key}", response_class=FileResponse)
async def agent_portrait(agent_id: str, key: str, container: ContainerDep) -> FileResponse:
    """One of its portraits (square WebP): hub, upload or self-<n>. Its URL carries a
    version, so it can be cached for good."""
    path = container.agents.portrait_file(agent_id, key)
    if path is None:
        raise NotFoundError(f"agent {agent_id} has no portrait {key!r}")
    return FileResponse(
        path, media_type="image/webp", headers={"Cache-Control": "private, max-age=31536000"}
    )
