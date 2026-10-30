from typing import Annotated

from fastapi import APIRouter, Query, Response, status

from imaginer.api.deps import ContainerDep
from imaginer.domain import models as dto
from imaginer.domain.errors import InvalidRequestError

router = APIRouter(prefix="/v1/conversations", tags=["conversations"])


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_conversation(
    body: dto.ConversationCreate, container: ContainerDep
) -> dto.Conversation:
    return await container.conversations.create(body)


@router.get("")
async def list_conversations(
    container: ContainerDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    cursor: Annotated[str | None, Query(description="next_cursor from the previous page")] = None,
    include_archived: bool = False,
) -> dto.Page[dto.ConversationSummary]:
    """Most recently updated first, each with its image count, latest image and turn state."""
    page = await container.conversations.list(
        limit=limit, cursor=cursor, include_archived=include_archived
    )
    for item in page.items:
        channel = container.turns.get(item.id)
        item.turn_running = channel is not None and not channel.closed
    return page


@router.get("/search")
async def search_conversations(
    container: ContainerDep,
    q: Annotated[str, Query(min_length=1, max_length=200, description="Text to find")],
    fields: Annotated[
        str, Query(description="Comma list of title, prompt, label, reply")
    ] = "title,prompt,label,reply",
    include_archived: bool = False,
    limit: Annotated[int, Query(ge=1, le=200, description="Conversations per page")] = 60,
    cursor: Annotated[str | None, Query(description="next_cursor from the previous page")] = None,
) -> dto.SearchPage:
    """Search titles, your prompts, image prompts, labels and replies, one page of
    conversations at a time, most recently updated first. Each hit says where it matched."""
    chosen = [f.strip() for f in fields.split(",") if f.strip()]
    unknown = sorted(set(chosen) - set(dto.SEARCH_FIELDS))
    if unknown or not chosen:
        raise InvalidRequestError(
            f"fields must name some of {', '.join(dto.SEARCH_FIELDS)}", unknown=unknown
        )
    page = await container.conversations.search(
        q.strip() or q,
        fields=[f for f in dto.SEARCH_FIELDS if f in chosen],
        limit=limit,
        cursor=cursor,
        include_archived=include_archived,
    )
    for item in page.items:
        channel = container.turns.get(item.id)
        item.turn_running = channel is not None and not channel.closed
    return page


@router.get("/{conversation_id}")
async def get_conversation(conversation_id: str, container: ContainerDep) -> dto.Conversation:
    return await container.conversations.get(conversation_id)


@router.patch("/{conversation_id}")
async def update_conversation(
    conversation_id: str, body: dto.ConversationUpdate, container: ContainerDep
) -> dto.Conversation:
    """Partial update: only fields present in the body change. ``archived`` toggles archival."""
    return await container.conversations.update(conversation_id, body)


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(conversation_id: str, container: ContainerDep) -> Response:
    """Deletes the conversation with all its messages, artifacts (rows and files) and jobs.

    The conversation is gone for readers as soon as this returns. If its files could not be
    removed yet (e.g. a file is still open on Windows), a background sweep finishes the job.
    """
    await container.conversations.delete(conversation_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
