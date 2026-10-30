"""Cancelling a turn while its image job is generating, over real sockets."""

import asyncio
import json
import threading
from collections.abc import Iterator

import httpx
import pytest

from imaginer.core.config import Settings
from imaginer.main import create_app
from tests.api.test_chat_live import events
from tests.fakes.fake_llama import FakeLlama
from tests.fakes.fake_sdcpp import FakeSdCpp
from tests.fakes.live import LiveServer


@pytest.fixture
def server(studio_settings: Settings) -> Iterator[LiveServer]:
    with LiveServer(create_app(studio_settings), lifespan="on") as live:
        yield live


async def test_cancel_during_generation(
    server: LiveServer, fake_llama: FakeLlama, fake_sd: FakeSdCpp
) -> None:
    fake_sd.hold = threading.Event()
    fake_llama.queue(FakeLlama.tool_call("generate_image", {"prompt": "a slow masterpiece"}))
    async with httpx.AsyncClient(base_url=server.url, timeout=15) as client:
        cid = (await client.post("/v1/conversations", json={})).json()["id"]
        async with client.stream(
            "POST", f"/v1/conversations/{cid}/messages", json={"content": "paint"}
        ) as response:
            stream = events(response)
            async for kind, _, data in stream:
                if kind == "job_update" and data["job"]["status"] == "running":
                    break
            assert await asyncio.to_thread(fake_sd.started.wait, 10)
            assert (await client.post(f"/v1/conversations/{cid}/turn/cancel")).status_code == 200
            rest = [e async for e in stream]

        statuses = [d["job"]["status"] for k, _, d in rest if k == "job_update"]
        assert statuses[-1] == "cancelled"
        assert rest[-1][0] == "done" and rest[-1][2]["finish_reason"] == "cancelled"
        assert len(fake_sd.cancels) == 1  # the engine was told to stop

        messages = (await client.get(f"/v1/conversations/{cid}/messages")).json()["items"]
        assert [m["role"] for m in messages] == ["user", "assistant", "tool"]
        result = json.loads(messages[2]["content_text"])
        assert result["status"] == "error" and result["error"]["code"] == "cancelled"
        assert messages[2]["tool_call_id"] == messages[1]["tool_calls"][0]["id"]
        jobs = (await client.get(f"/v1/conversations/{cid}/jobs")).json()["items"]
        assert [j["status"] for j in jobs] == ["cancelled"]

        # The history is still well-formed: the next turn works and sees the cancelled call.
        fake_sd.hold.set()
        fake_llama.queue(FakeLlama.text("Okay, stopped."))
        reply = await client.post(f"/v1/conversations/{cid}/messages", json={"content": "ok"})
        assert reply.status_code == 200
        sent = fake_llama.requests[-1]["messages"]
        assert [m["role"] for m in sent] == ["system", "user", "assistant", "tool", "user"]


async def test_cancelling_only_the_job_lets_the_turn_finish(
    server: LiveServer, fake_llama: FakeLlama, fake_sd: FakeSdCpp
) -> None:
    fake_sd.hold = threading.Event()
    fake_llama.queue(
        FakeLlama.tool_call("generate_image", {"prompt": "a slow masterpiece"}),
        FakeLlama.text("No problem, I stopped it."),
    )
    async with httpx.AsyncClient(base_url=server.url, timeout=15) as client:
        cid = (await client.post("/v1/conversations", json={})).json()["id"]
        async with client.stream(
            "POST", f"/v1/conversations/{cid}/messages", json={"content": "paint"}
        ) as response:
            stream = events(response)
            async for kind, _, data in stream:
                if kind == "job_update" and data["job"]["status"] == "running":
                    job_id = data["job"]["id"]
                    break
            assert await asyncio.to_thread(fake_sd.started.wait, 10)
            cancelled = await client.post(f"/v1/jobs/{job_id}/cancel")
            assert cancelled.status_code == 200
            assert cancelled.json()["status"] == "cancelled"
            rest = [e async for e in stream]

        assert rest[-1][0] == "done"
        assert rest[-1][2]["finish_reason"] == "stop"  # the turn itself was not cancelled
        assert len(fake_sd.cancels) == 1
        messages = (await client.get(f"/v1/conversations/{cid}/messages")).json()["items"]
        assert [m["role"] for m in messages] == ["user", "assistant", "tool", "assistant"]
        result = json.loads(messages[2]["content_text"])
        assert result["error"]["code"] == "job_cancelled"
        assert messages[3]["content_text"] == "No problem, I stopped it."
        assert fake_llama.requests[-1]["messages"][-1]["role"] == "tool"


async def test_cancelling_a_retry_still_answers_its_attempt(
    server: LiveServer, fake_llama: FakeLlama, fake_sd: FakeSdCpp
) -> None:
    fake_llama.queue(
        FakeLlama.tool_call("generate_image", {"prompt": "a lighthouse"}),
        FakeLlama.text("Here is img-1."),
    )
    async with httpx.AsyncClient(base_url=server.url, timeout=15) as client:
        cid = (await client.post("/v1/conversations", json={})).json()["id"]
        assert (
            await client.post(f"/v1/conversations/{cid}/messages", json={"content": "paint"})
        ).status_code == 200
        result = (await client.get(f"/v1/conversations/{cid}/messages")).json()["items"][2]

        fake_sd.hold = threading.Event()
        async with client.stream(
            "POST", f"/v1/conversations/{cid}/messages/{result['id']}/retry"
        ) as response:
            stream = events(response)
            async for kind, _, data in stream:
                if kind == "job_update" and data["job"]["status"] == "running":
                    break
            assert await asyncio.to_thread(fake_sd.started.wait, 10)
            assert (await client.post(f"/v1/conversations/{cid}/turn/cancel")).status_code == 200
            rest = [e async for e in stream]
        assert rest[-1][0] == "done" and rest[-1][2]["finish_reason"] == "cancelled"

        # The attempt is on its own branch, finished with the cancellation as its result.
        branch = (await client.get(f"/v1/conversations/{cid}/messages")).json()["items"]
        assert [m["role"] for m in branch] == ["user", "assistant", "tool"]
        attempt = branch[2]
        assert attempt["id"] != result["id"] and attempt["parent_id"] == result["parent_id"]
        assert attempt["status"] == "complete"
        assert json.loads(attempt["content_text"])["error"]["code"] == "cancelled"

        # And it can be continued like any other branch.
        fake_sd.hold.set()
        fake_llama.queue(FakeLlama.text("Okay."))
        reply = await client.post(
            f"/v1/conversations/{cid}/messages",
            json={"content": "never mind", "parent_id": attempt["id"]},
        )
        assert reply.status_code == 200
        sent = fake_llama.requests[-1]["messages"]
        assert [m["role"] for m in sent] == ["system", "user", "assistant", "tool", "user"]
        assert json.loads(sent[3]["content"])["error"]["code"] == "cancelled"
