"""Per-turn event channels.

A turn runs as its own task, independent of any HTTP connection. Its events go into a
channel that keeps the full history of the turn, so any number of subscribers can attach
at any time (``Last-Event-ID`` resume) and a disconnected client loses nothing.
"""

import asyncio
from collections.abc import AsyncIterator
from datetime import datetime

from pydantic import BaseModel

from imaginer.core.clock import utcnow
from imaginer.core.ids import new_id
from imaginer.domain.errors import ConflictError
from imaginer.domain.events import EventType, TurnEvent

CANCEL_WAIT_S = 10.0


class TurnChannel:
    def __init__(self, conversation_id: str) -> None:
        self.turn_id = new_id()
        self.conversation_id = conversation_id
        self.started_at: datetime = utcnow()
        self.task: asyncio.Task[None] | None = None
        self._events: list[TurnEvent] = []
        self._closed = False
        self._changed = asyncio.Condition()

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def last_event_id(self) -> int:
        return len(self._events)

    async def publish(self, type_: EventType, data: BaseModel) -> TurnEvent:
        if self._closed:
            raise RuntimeError(f"turn {self.turn_id} is closed")
        event = TurnEvent(id=len(self._events) + 1, type=type_, data=data)
        self._events.append(event)
        async with self._changed:
            self._changed.notify_all()
        return event

    async def close(self) -> None:
        self._closed = True
        async with self._changed:
            self._changed.notify_all()

    async def subscribe(self, after_id: int = 0) -> AsyncIterator[TurnEvent]:
        """Every event with ``id > after_id``, then live events until the turn ends."""
        index = max(after_id, 0)
        while True:
            while index < len(self._events):
                yield self._events[index]
                index += 1
            if self._closed:
                return

            def has_news(seen: int = index) -> bool:
                return len(self._events) > seen or self._closed

            async with self._changed:
                await self._changed.wait_for(has_news)


class TurnManager:
    """At most one running turn per conversation."""

    def __init__(self) -> None:
        self._active: dict[str, TurnChannel] = {}

    def open(self, conversation_id: str) -> TurnChannel:
        current = self._active.get(conversation_id)
        if current is not None:
            raise ConflictError(
                "a turn is already running in this conversation", turn_id=current.turn_id
            )
        channel = TurnChannel(conversation_id)
        self._active[conversation_id] = channel
        return channel

    def get(self, conversation_id: str) -> TurnChannel | None:
        return self._active.get(conversation_id)

    def release(self, channel: TurnChannel) -> None:
        if self._active.get(channel.conversation_id) is channel:
            del self._active[channel.conversation_id]

    async def cancel(self, conversation_id: str) -> TurnChannel | None:
        """Cancel the conversation's running turn and wait (briefly) for it to wind down."""
        channel = self._active.get(conversation_id)
        if channel is None or channel.task is None:
            return channel
        channel.task.cancel()
        await asyncio.wait({channel.task}, timeout=CANCEL_WAIT_S)
        return channel

    async def cancel_all(self) -> None:
        for conversation_id in list(self._active):
            await self.cancel(conversation_id)
