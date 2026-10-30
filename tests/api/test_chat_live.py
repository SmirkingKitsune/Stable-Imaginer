"""Turn behaviour that needs real streaming over real sockets: concurrency, cancel,
re-attach and client disconnects. The app runs in a live uvicorn server."""

import asyncio
import json
import threading
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
import pytest

from imaginer.core.config import Settings
from imaginer.main import create_app
from tests.fakes.fake_llama import FakeLlama
from tests.fakes.live import LiveServer

Event = tuple[str, int, dict[str, Any]]


@pytest.fixture
def server(chat_settings: Settings) -> Iterator[LiveServer]:
    with LiveServer(create_app(chat_settings), lifespan="on") as live:
        yield live


@pytest.fixture
def hold(fake_llama: FakeLlama) -> Iterator[threading.Event]:
    """The fake engine sends one chunk, then waits for this event."""
    event = threading.Event()
    fake_llama.queue(FakeLlama.text("partial and more", pieces=2, hold=event))
    yield event
    event.set()  # never leave the fake's worker thread blocked


async def events(response: httpx.Response) -> AsyncIterator[Event]:
    fields: dict[str, str] = {}
    data: list[str] = []
    async for line in response.aiter_lines():
        if line:
            if not line.startswith(":"):
                name, _, value = line.partition(":")
                value = value.removeprefix(" ")
                if name == "data":
                    data.append(value)
                else:
                    fields[name] = value
            continue
        if "event" in fields:
            yield fields["event"], int(fields["id"]), json.loads("\n".join(data))
        fields, data = {}, []


async def _until_delta(stream: AsyncIterator[Event]) -> list[Event]:
    seen = []
    async for event in stream:
        seen.append(event)
        if event[0] == "delta":
            return seen
    raise AssertionError(f"stream ended without a delta: {seen}")


async def _messages(client: httpx.AsyncClient, cid: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = (await client.get(f"/v1/conversations/{cid}/messages")).json()[
        "items"
    ]
    return items


async def test_conflict_attach_and_cancel(server: LiveServer, hold: threading.Event) -> None:
    async with httpx.AsyncClient(base_url=server.url, timeout=10) as client:
        cid = (await client.post("/v1/conversations", json={})).json()["id"]
        url = f"/v1/conversations/{cid}/messages"
        async with client.stream("POST", url, json={"content": "go"}) as response:
            stream = events(response)
            head = await _until_delta(stream)
            assert [e[0] for e in head] == ["message", "message", "delta"]

            busy = await client.post(url, json={"content": "me too"})
            assert busy.status_code == 409
            assert busy.json()["turn_id"] == response.headers["x-turn-id"]

            info = (await client.get(f"/v1/conversations/{cid}/turn")).json()
            assert info["running"] is True
            assert info["turn_id"] == response.headers["x-turn-id"]
            assert info["last_event_id"] == 3

            # A second client re-attaches and gets everything after event 1.
            async with client.stream(
                "GET", f"/v1/conversations/{cid}/turn/events", headers={"Last-Event-ID": "1"}
            ) as attached:
                attached_stream = events(attached)
                replay = await _until_delta(attached_stream)
                assert [(e[0], e[1]) for e in replay] == [("message", 2), ("delta", 3)]

                cancelled = await client.post(f"/v1/conversations/{cid}/turn/cancel")
                assert cancelled.status_code == 200
                assert cancelled.json()["running"] is False

                tail = [e async for e in attached_stream]
            rest = [e async for e in stream]

    assert rest == tail  # both subscribers saw the same ending
    assert [e[0] for e in rest] == ["message", "done"]
    assert rest[0][2]["message"]["status"] == "cancelled"
    assert rest[1][2]["finish_reason"] == "cancelled"

    async with httpx.AsyncClient(base_url=server.url) as client:
        stored = await _messages(client, cid)
        assert [(m["role"], m["status"], m["content_text"]) for m in stored] == [
            ("user", "complete", "go"),
            ("assistant", "cancelled", "partial "),
        ]
        assert (await client.get(f"/v1/conversations/{cid}/turn")).status_code == 404


async def test_turn_survives_client_disconnect(server: LiveServer, hold: threading.Event) -> None:
    async with httpx.AsyncClient(base_url=server.url, timeout=10) as client:
        cid = (await client.post("/v1/conversations", json={})).json()["id"]
        async with client.stream(
            "POST", f"/v1/conversations/{cid}/messages", json={"content": "go"}
        ) as response:
            await _until_delta(events(response))
        # The client hung up mid-turn. The engine now finishes its answer.
        hold.set()
        for _ in range(100):
            stored = await _messages(client, cid)
            if stored[-1]["status"] == "complete":
                break
            await asyncio.sleep(0.05)
        assert stored[-1]["content_text"] == "partial and more"


async def test_deleting_a_conversation_stops_its_turn(
    server: LiveServer, hold: threading.Event
) -> None:
    async with httpx.AsyncClient(base_url=server.url, timeout=10) as client:
        cid = (await client.post("/v1/conversations", json={})).json()["id"]
        async with client.stream(
            "POST", f"/v1/conversations/{cid}/messages", json={"content": "go"}
        ) as response:
            stream = events(response)
            await _until_delta(stream)
            assert (await client.delete(f"/v1/conversations/{cid}")).status_code == 204
            rest = [e async for e in stream]
        assert rest[-1][0] == "done"
        assert rest[-1][2]["finish_reason"] == "cancelled"
        assert (await client.get(f"/v1/conversations/{cid}")).status_code == 404
