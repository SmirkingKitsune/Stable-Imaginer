from typing import Annotated

from fastapi import APIRouter, Header, Query
from sse_starlette import EventSourceResponse

from imaginer.api.deps import ContainerDep
from imaginer.api.sse import SSE_RESPONSE_DOC, turn_event_stream
from imaginer.domain import models as dto
from imaginer.domain.errors import NotFoundError
from imaginer.orchestration.event_bus import TurnChannel

router = APIRouter(prefix="/v1/conversations/{conversation_id}", tags=["messages"])


@router.get("/messages")
async def list_messages(
    conversation_id: str,
    container: ContainerDep,
    after_seq: Annotated[int, Query(ge=0, description="Only messages with seq > after_seq")] = 0,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    leaf_id: Annotated[
        str | None, Query(description="Show the branch ending at this message (any message).")
    ] = None,
    all_branches: Annotated[
        bool, Query(description="Every message of every branch; rebuild the tree by parent_id.")
    ] = False,
) -> dto.MessagePage:
    """Messages of one branch in order: the latest branch unless ``leaf_id`` names another.

    Resume or page with ``after_seq`` = the last ``seq`` you have.
    """
    return await container.conversations.list_messages(
        conversation_id,
        after_seq=after_seq,
        limit=limit,
        leaf_id=leaf_id,
        all_branches=all_branches,
    )


@router.get("/branches")
async def list_branches(conversation_id: str, container: ContainerDep) -> list[dto.Branch]:
    """Every branch, main first, each followed by the branches forked from it."""
    return await container.conversations.branches(conversation_id)


@router.post("/messages", response_class=EventSourceResponse, responses=SSE_RESPONSE_DOC)
async def post_message(
    conversation_id: str, body: dto.MessageCreate, container: ContainerDep
) -> EventSourceResponse:
    """Send a user message and stream the assistant's turn as server-sent events.

    The message continues the latest branch, or ``parent_id``. With ``edit_of`` it replaces
    one of your earlier messages on a new branch (edit & branch).

    Problems found before the turn starts are returned as regular HTTP errors:
    404 (no conversation), 409 (a turn is already running), 422 (bad attachments, an unknown
    ``parent_id``/``edit_of``, or no backend with the needed capability), 503 (chat engine
    unreachable). After that, failures arrive as an ``error`` event. The turn keeps running
    if the client disconnects. Re-attach with ``GET .../turn/events`` or re-read
    ``GET .../messages``.
    """
    channel = await container.turn_runner.start(conversation_id, body)
    return turn_event_stream(channel)


@router.post(
    "/messages/{message_id}/retry",
    response_class=EventSourceResponse,
    responses=SSE_RESPONSE_DOC,
)
async def retry_tool_call(
    conversation_id: str, message_id: str, container: ContainerDep
) -> EventSourceResponse:
    """Run a tool call again as a new attempt, on a new branch, and stream that turn.

    ``message_id`` is the call's tool result (any attempt). The call runs with the same
    arguments and image options but a new seed. Its result becomes a sibling of the old one,
    and the assistant then replies on the new branch. Errors before the turn starts: 404 (no
    such message), 409 (a turn is running), 422 (not a finished tool result), 503.
    """
    channel = await container.turn_runner.retry(conversation_id, message_id)
    return turn_event_stream(channel)


@router.get("/turn")
async def get_turn(conversation_id: str, container: ContainerDep) -> dto.TurnInfo:
    """The conversation's running turn, if any (404 otherwise)."""
    return _info(_active(container.turns.get(conversation_id), conversation_id))


@router.get("/turn/events", response_class=EventSourceResponse, responses=SSE_RESPONSE_DOC)
async def attach_turn(
    conversation_id: str,
    container: ContainerDep,
    last_event_id: Annotated[int, Header(ge=0)] = 0,
) -> EventSourceResponse:
    """Re-attach to the running turn's event stream. Replays events after ``Last-Event-ID``."""
    channel = _active(container.turns.get(conversation_id), conversation_id)
    return turn_event_stream(channel, after_id=last_event_id)


@router.post("/turn/cancel")
async def cancel_turn(conversation_id: str, container: ContainerDep) -> dto.TurnInfo:
    """Stop the running turn. Text streamed so far is kept, marked ``cancelled``."""
    channel = _active(await container.turns.cancel(conversation_id), conversation_id)
    return _info(channel)


def _active(channel: TurnChannel | None, conversation_id: str) -> TurnChannel:
    if channel is None:
        raise NotFoundError(f"no turn is running in conversation {conversation_id}")
    return channel


def _info(channel: TurnChannel) -> dto.TurnInfo:
    return dto.TurnInfo(
        turn_id=channel.turn_id,
        conversation_id=channel.conversation_id,
        started_at=channel.started_at,
        running=not channel.closed,
        last_event_id=channel.last_event_id,
    )
