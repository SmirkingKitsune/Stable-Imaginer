"""Branching end to end: edit & branch, continuing any branch, and retrying a tool call."""

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from imaginer.core.config import Settings
from imaginer.core.ids import new_id
from tests.api.test_generation import _client_for, _of
from tests.fakes.fake_llama import FakeLlama
from tests.fakes.fake_sdcpp import FakeSdCpp
from tests.helpers import parse_sse

Event = tuple[str, int, dict[str, Any]]


@pytest.fixture
async def chat(chat_settings: Settings) -> AsyncIterator[httpx.AsyncClient]:
    async for client in _client_for(chat_settings):
        yield client


@pytest.fixture
async def studio(studio_settings: Settings) -> AsyncIterator[httpx.AsyncClient]:
    async for client in _client_for(studio_settings):
        yield client


async def _send(client: httpx.AsyncClient, cid: str, content: str, **extra: Any) -> list[Event]:
    resp = await client.post(
        f"/v1/conversations/{cid}/messages", json={"content": content, **extra}
    )
    assert resp.status_code == 200, resp.text
    events = parse_sse(resp.text)
    assert events[-1][0] == "done", events[-1]
    return events


async def _messages(client: httpx.AsyncClient, cid: str, **params: Any) -> list[dict[str, Any]]:
    resp = await client.get(f"/v1/conversations/{cid}/messages", params=params)
    assert resp.status_code == 200, resp.text
    items: list[dict[str, Any]] = resp.json()["items"]
    return items


async def _branches(client: httpx.AsyncClient, cid: str) -> list[dict[str, Any]]:
    resp = await client.get(f"/v1/conversations/{cid}/branches")
    assert resp.status_code == 200, resp.text
    items: list[dict[str, Any]] = resp.json()
    return items


def _texts(messages: list[dict[str, Any]]) -> list[str | None]:
    return [m["content_text"] for m in messages]


async def test_linear_messages_chain_by_parent(
    chat: httpx.AsyncClient, fake_llama: FakeLlama
) -> None:
    cid = (await chat.post("/v1/conversations", json={})).json()["id"]
    await _send(chat, cid, "one")
    await _send(chat, cid, "two")
    messages = await _messages(chat, cid)
    assert _texts(messages) == ["one", "You said: one", "two", "You said: two"]
    assert [m["parent_id"] for m in messages] == [None, *[m["id"] for m in messages[:-1]]]

    (main,) = await _branches(chat, cid)
    assert main["name"] == "main" and main["kind"] == "main" and main["latest"] is True
    assert main["leaf_id"] == messages[-1]["id"]
    assert (main["message_count"], main["fork_id"], main["parent_leaf_id"]) == (4, None, None)


async def test_edit_and_branch_keeps_the_original_branch(
    chat: httpx.AsyncClient, fake_llama: FakeLlama
) -> None:
    cid = (await chat.post("/v1/conversations", json={})).json()["id"]
    await _send(chat, cid, "a lighthouse")
    await _send(chat, cid, "at night")
    main = await _messages(chat, cid)
    first = main[0]

    events = await _send(chat, cid, "a lighthouse, wide crop", edit_of=first["id"])
    edited = _of(events, "message")[0]["message"]
    assert edited["parent_id"] is None  # an alternative to the first message

    # The model saw only the new branch.
    sent = fake_llama.requests[-1]["messages"]
    assert [(m["role"], m["content"]) for m in sent[1:]] == [("user", "a lighthouse, wide crop")]

    # Without leaf_id, the latest branch; the original is still there in full.
    assert _texts(await _messages(chat, cid)) == [
        "a lighthouse, wide crop",
        "You said: a lighthouse, wide crop",
    ]
    assert _texts(await _messages(chat, cid, leaf_id=main[-1]["id"])) == _texts(main)
    assert len(await _messages(chat, cid, all_branches=True)) == 6

    branches = await _branches(chat, cid)
    assert [(b["name"], b["kind"], b["depth"], b["latest"]) for b in branches] == [
        ("main", "main", 0, False),
        ("message 1, edited", "edit", 1, True),
    ]
    fork = branches[1]
    assert fork["fork_id"] == edited["id"]
    assert fork["parent_leaf_id"] == main[-1]["id"]
    assert fork["origin"] == "from your first message"
    assert fork["message_count"] == 2

    # Continuing main by its leaf makes it the latest branch again, with its own history.
    await _send(chat, cid, "add a boat", parent_id=main[-1]["id"])
    sent = fake_llama.requests[-1]["messages"]
    assert [m["content"] for m in sent[1:] if m["role"] == "user"] == [
        "a lighthouse",
        "at night",
        "add a boat",
    ]
    branches = await _branches(chat, cid)
    assert [(b["name"], b["message_count"], b["latest"]) for b in branches] == [
        ("main", 6, True),
        ("message 1, edited", 2, False),
    ]


async def test_reply_from_the_middle_forks_a_branch(
    chat: httpx.AsyncClient, fake_llama: FakeLlama
) -> None:
    cid = (await chat.post("/v1/conversations", json={})).json()["id"]
    await _send(chat, cid, "one")
    await _send(chat, cid, "two")
    first_reply = (await _messages(chat, cid))[1]

    await _send(chat, cid, "two, differently", parent_id=first_reply["id"])
    assert _texts(await _messages(chat, cid)) == [
        "one",
        "You said: one",
        "two, differently",
        "You said: two, differently",
    ]
    names = [(b["name"], b["origin"]) for b in await _branches(chat, cid)]
    assert names == [("main", None), ("message 2, edited", "from your second message")]


async def test_bad_anchors_are_refused_before_anything_is_stored(
    chat: httpx.AsyncClient, fake_llama: FakeLlama
) -> None:
    cid = (await chat.post("/v1/conversations", json={})).json()["id"]
    await _send(chat, cid, "hello")
    user, assistant = await _messages(chat, cid)
    url = f"/v1/conversations/{cid}/messages"

    unknown = await chat.post(url, json={"content": "x", "parent_id": new_id()})
    assert unknown.status_code == 422 and "parent_id" in unknown.json()["detail"]
    not_mine = await chat.post(url, json={"content": "x", "edit_of": assistant["id"]})
    assert not_mine.status_code == 422
    both = await chat.post(
        url, json={"content": "x", "parent_id": user["id"], "edit_of": user["id"]}
    )
    assert both.status_code == 422
    other = (await chat.post("/v1/conversations", json={})).json()["id"]
    foreign = await chat.post(
        f"/v1/conversations/{other}/messages", json={"content": "x", "parent_id": user["id"]}
    )
    assert foreign.status_code == 422

    assert len(await _messages(chat, cid, all_branches=True)) == 2
    assert (await chat.get(f"/v1/conversations/{cid}/turn")).status_code == 404  # released
    missing = await chat.get(f"/v1/conversations/{cid}/messages", params={"leaf_id": new_id()})
    assert missing.status_code == 404


async def _draw(client: httpx.AsyncClient, fake_llama: FakeLlama, **arguments: Any) -> str:
    fake_llama.queue(
        FakeLlama.tool_call("generate_image", {"prompt": "a lighthouse", **arguments}),
        FakeLlama.text("Here is img-1."),
    )
    cid: str = (await client.post("/v1/conversations", json={})).json()["id"]
    await _send(client, cid, "draw a lighthouse")
    return cid


async def _retry(client: httpx.AsyncClient, cid: str, message_id: str) -> list[Event]:
    resp = await client.post(f"/v1/conversations/{cid}/messages/{message_id}/retry")
    assert resp.status_code == 200, resp.text
    return parse_sse(resp.text)


async def test_retry_runs_the_call_again_on_a_new_branch(
    studio: httpx.AsyncClient, fake_llama: FakeLlama, fake_sd: FakeSdCpp
) -> None:
    fake_sd.step_delay_s = 0.0
    cid = await _draw(studio, fake_llama, seed=7)
    user, caller, result, reply = await _messages(studio, cid)

    fake_llama.queue(FakeLlama.text("Another take: img-2."))
    events = await _retry(studio, cid, result["id"])
    assert events[-1][0] == "done"

    # The attempt is stored first, beside the old result, and filled in once the job is done.
    stored = [d["message"] for d in _of(events, "message")]
    attempt = stored[0]
    assert attempt["role"] == "tool" and attempt["status"] == "streaming"
    assert attempt["parent_id"] == caller["id"]
    assert attempt["tool_call_id"] == result["tool_call_id"]
    (call,) = _of(events, "tool_call")
    assert call["message_id"] == caller["id"]
    assert "seed" not in json.loads(call["arguments"])  # a new seed, same everything else
    assert json.loads(call["arguments"])["prompt"] == "a lighthouse"
    (created,) = _of(events, "artifact_created")
    assert created["artifact"]["label"] == "img-2"  # labels stay unique across branches
    assert fake_sd.requests[0]["seed"] == 7 and fake_sd.requests[1]["seed"] != 7

    branch = await _messages(studio, cid)
    assert [m["id"] for m in branch[:3]] == [user["id"], caller["id"], attempt["id"]]
    assert branch[2]["status"] == "complete"
    assert branch[2]["attachments"] == [created["artifact"]["id"]]
    assert json.loads(branch[2]["content_text"])["images"][0]["label"] == "img-2"
    assert branch[3]["content_text"] == "Another take: img-2."
    assert branch[3]["parent_id"] == attempt["id"]

    # The model saw the new attempt, and only this branch's images.
    sent = fake_llama.requests[-1]["messages"]
    assert [m["role"] for m in sent] == ["system", "user", "assistant", "tool"]
    assert "img-2" in sent[3]["content"]
    assert "- img-2: generated image" in sent[0]["content"]
    assert "- img-1:" not in sent[0]["content"]

    jobs = (await studio.get(f"/v1/conversations/{cid}/jobs")).json()["items"]
    assert [(j["request_message_id"], j["tool_call_id"]) for j in jobs] == [
        (caller["id"], result["tool_call_id"])
    ] * 2

    branches = await _branches(studio, cid)
    assert [
        (b["name"], b["kind"], b["message_count"], b["latest_image"]["label"], b["latest"])
        for b in branches
    ] == [
        ("main", "main", 2, "img-1", False),
        ("retry of img-1", "retry", 2, "img-2", True),
    ]
    assert branches[1]["origin"] == "from assistant reply 1"
    assert branches[1]["fork_id"] == attempt["id"]
    assert branches[1]["parent_leaf_id"] == reply["id"]

    # A retry of the retry is a third attempt at the same call.
    fake_llama.queue(FakeLlama.text("Third: img-3."))
    await _retry(studio, cid, attempt["id"])
    attempts = [
        m
        for m in await _messages(studio, cid, all_branches=True)
        if m["role"] == "tool" and m["parent_id"] == caller["id"]
    ]
    assert len(attempts) == 3
    assert [b["name"] for b in await _branches(studio, cid)] == [
        "main",
        "retry of img-1",
        "retry of img-1",
    ]


async def test_retry_keeps_the_users_options_except_the_seed(
    studio: httpx.AsyncClient, fake_llama: FakeLlama, fake_sd: FakeSdCpp
) -> None:
    fake_sd.step_delay_s = 0.0
    fake_llama.queue(
        FakeLlama.tool_call("generate_image", {"prompt": "a fox"}),
        FakeLlama.text("Two foxes."),
    )
    cid = (await studio.post("/v1/conversations", json={})).json()["id"]
    options = {"aspect_ratio": "3:4", "n": 2, "seed": 41207}
    await _send(studio, cid, "draw a fox", image_options=options)
    result = (await _messages(studio, cid))[2]

    fake_llama.queue(FakeLlama.text("Two more."))
    events = await _retry(studio, cid, result["id"])
    created = [d["artifact"] for d in _of(events, "artifact_created")]
    assert [a["label"] for a in created] == ["img-3", "img-4"]
    assert all(a["height"] > a["width"] for a in created)  # still 3:4
    retried = [r["seed"] for r in fake_sd.requests[2:]]
    assert len(retried) == 2 and retried[0] != 41207
    note = json.loads((await _messages(studio, cid))[2]["content_text"])
    assert note["user_options"] == {"aspect_ratio": "3:4", "n": 2}


async def test_retry_refusals(
    studio: httpx.AsyncClient, fake_llama: FakeLlama, fake_sd: FakeSdCpp
) -> None:
    fake_sd.step_delay_s = 0.0
    cid = await _draw(studio, fake_llama)
    user = (await _messages(studio, cid))[0]
    url = f"/v1/conversations/{cid}/messages/{{}}/retry"

    assert (await studio.post(url.format(new_id()))).status_code == 404
    not_tool = await studio.post(url.format(user["id"]))
    assert not_tool.status_code == 422
    assert "tool result" in not_tool.json()["detail"]
    gone = (await studio.post("/v1/conversations", json={})).json()["id"]
    assert (
        await studio.post(f"/v1/conversations/{gone}/messages/{user['id']}/retry")
    ).status_code == 404
    assert len(await _messages(studio, cid, all_branches=True)) == 4
