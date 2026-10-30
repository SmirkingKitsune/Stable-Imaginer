from collections.abc import AsyncIterator
from typing import Any

from sse_starlette import EventSourceResponse, ServerSentEvent

from imaginer.orchestration.event_bus import TurnChannel

PING_INTERVAL_S = 15

SSE_RESPONSE_DOC: dict[int | str, dict[str, Any]] = {
    200: {
        "description": (
            "Server-sent events: `message`, `delta`, `tool_call`, `job_update`, "
            "`artifact_created`, then `done` or `error`. Each event has an `id` usable as "
            "`Last-Event-ID`."
        ),
        "content": {"text/event-stream": {"schema": {"type": "string"}}},
    }
}


def turn_event_stream(channel: TurnChannel, after_id: int = 0) -> EventSourceResponse:
    """Stream a turn's events. Disconnecting stops the stream, never the turn."""

    async def events() -> AsyncIterator[ServerSentEvent]:
        async for event in channel.subscribe(after_id):
            yield ServerSentEvent(
                data=event.data.model_dump_json(), event=event.type.value, id=str(event.id)
            )

    return EventSourceResponse(
        events(),
        ping=PING_INTERVAL_S,
        headers={"X-Turn-ID": channel.turn_id, "X-Accel-Buffering": "no"},
    )
