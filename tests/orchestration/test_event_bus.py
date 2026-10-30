import asyncio

import pytest

from imaginer.domain.errors import ConflictError
from imaginer.domain.events import DeltaEventData, EventType
from imaginer.orchestration.event_bus import TurnChannel, TurnManager


async def _drain(channel: TurnChannel, after: int = 0) -> list[int]:
    return [event.id async for event in channel.subscribe(after)]


async def test_late_and_concurrent_subscribers_see_everything() -> None:
    channel = TurnChannel("conv")
    early = asyncio.create_task(_drain(channel))
    for i in range(3):
        await channel.publish(EventType.DELTA, DeltaEventData(message_id="m", content=str(i)))
        await asyncio.sleep(0)
    late = asyncio.create_task(_drain(channel))
    resumed = asyncio.create_task(_drain(channel, after=2))
    await channel.publish(EventType.DELTA, DeltaEventData(message_id="m", content="3"))
    await channel.close()
    assert await early == [1, 2, 3, 4]
    assert await late == [1, 2, 3, 4]
    assert await resumed == [3, 4]
    with pytest.raises(RuntimeError):
        await channel.publish(EventType.DELTA, DeltaEventData(message_id="m"))


async def test_a_cancelled_subscriber_does_not_disturb_others() -> None:
    channel = TurnChannel("conv")
    quitter = asyncio.create_task(_drain(channel))
    stayer = asyncio.create_task(_drain(channel))
    await asyncio.sleep(0)
    quitter.cancel()
    await channel.publish(EventType.DELTA, DeltaEventData(message_id="m"))
    await channel.close()
    assert await stayer == [1]


async def test_one_turn_per_conversation() -> None:
    turns = TurnManager()
    first = turns.open("a")
    turns.open("b")  # other conversations are independent
    with pytest.raises(ConflictError):
        turns.open("a")
    turns.release(first)
    turns.open("a")


async def test_cancel_waits_for_the_task() -> None:
    turns = TurnManager()
    channel = turns.open("a")
    finished = asyncio.Event()

    async def work() -> None:
        try:
            await asyncio.sleep(30)
        finally:
            finished.set()
            turns.release(channel)

    channel.task = asyncio.create_task(work())
    await asyncio.sleep(0)
    assert await turns.cancel("a") is channel
    assert finished.is_set()
    assert turns.get("a") is None
    assert await turns.cancel("a") is None
