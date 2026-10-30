"""One queue in front of each image engine, shared by everything that uses it.

This instance's own jobs and the requests of linked instances wait at the same gate, so an
engine with ``concurrency: 1`` never runs two images at once, whoever asked.
"""

import asyncio
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager


class EngineGates:
    def __init__(self, concurrency: Mapping[str, int]) -> None:
        self._semaphores = {name: asyncio.Semaphore(n) for name, n in concurrency.items()}
        self._in_flight: dict[str, int] = {}

    def set_limit(self, name: str, concurrency: int) -> None:
        """For backends added at runtime. An existing gate keeps its limit until restart:
        replacing it would let queued work through twice."""
        self._semaphores.setdefault(name, asyncio.Semaphore(concurrency))

    @asynccontextmanager
    async def hold(self, name: str) -> AsyncIterator[None]:
        """Wait for a turn on the engine, and hold it until the block ends."""
        semaphore = self._semaphores.setdefault(name, asyncio.Semaphore(1))
        self._in_flight[name] = self._in_flight.get(name, 0) + 1
        try:
            async with semaphore:
                yield
        finally:
            self._in_flight[name] -= 1

    def queue_depth(self, name: str) -> int:
        """Requests waiting for or holding the engine."""
        return self._in_flight.get(name, 0)
