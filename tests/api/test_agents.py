"""Agents end to end (docs/agent-library.md): every chat engine gets one, a turn answers as the
agent (its system prompt, tools and sampling), set_portrait paints a self-portrait, and a
llama-server launched here runs the agent it is set to."""

import asyncio
import io
import json
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from PIL import Image

from imaginer.core.config import LlamaCppLaunch, RoutingConfig, Settings
from imaginer.main import create_app
from tests.conftest import SettingsFactory, llama_config
from tests.fakes.fake_hub import gguf
from tests.fakes.fake_llama import FakeLlama
from tests.fakes.fake_sdcpp import FakeSdCpp
from tests.helpers import image_bytes, parse_sse


async def _client_for(settings: Settings) -> AsyncIterator[httpx.AsyncClient]:
    app: FastAPI = create_app(settings)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


@pytest.fixture
async def chat(chat_settings: Settings) -> AsyncIterator[httpx.AsyncClient]:
    async for client in _client_for(chat_settings):
        yield client


@pytest.fixture
async def studio(studio_settings: Settings) -> AsyncIterator[httpx.AsyncClient]:
    async for client in _client_for(studio_settings):
        yield client


async def _only_agent(client: httpx.AsyncClient) -> dict[str, Any]:
    resp = await client.get("/v1/agents")
    assert resp.status_code == 200, resp.text
    agent: dict[str, Any]
    [agent] = resp.json()["items"]
    return agent


async def _say(
    client: httpx.AsyncClient, content: str, cid: str | None = None, **body: Any
) -> tuple[str, list[tuple[str, int, dict[str, Any]]]]:
    if cid is None:
        cid = (await client.post("/v1/conversations", json={})).json()["id"]
    resp = await client.post(f"/v1/conversations/{cid}/messages", json={"content": content, **body})
    assert resp.status_code == 200, resp.text
    return cid, parse_sse(resp.text)


async def _turn_over(client: httpx.AsyncClient, cid: str) -> None:
    deadline = time.monotonic() + 15
    while True:
        resp = await client.get(f"/v1/conversations/{cid}/turn")
        if resp.status_code == 404 or not resp.json()["running"]:
            return
        assert time.monotonic() < deadline, "the turn didn't end"
        await asyncio.sleep(0.05)


async def test_every_chat_engine_gets_an_agent(chat: httpx.AsyncClient) -> None:
    lib = (await chat.get("/v1/agents")).json()
    [agent] = lib["items"]
    # Named after the model the engine reports, and the one new prompts go to.
    assert agent["name"] == "Qwen3VL-8B-Instruct"
    assert (agent["engine"], agent["engine_set"], agent["default"]) == ("llama", "llama", True)
    assert lib["default"] == agent["id"]
    assert agent["use"]["kind"] == "remote" and agent["use"]["running"]
    assert agent["own_files"] is False and agent["ready"] and agent["vision"]
    assert agent["portrait"]["key"] is None and agent["portrait"]["monogram"] == "Q"
    assert [e["name"] for e in lib["engines"]] == ["llama"]


async def test_a_turn_answers_as_the_agent(chat: httpx.AsyncClient, fake_llama: FakeLlama) -> None:
    agent = await _only_agent(chat)
    resp = await chat.patch(
        f"/v1/agents/{agent['id']}",
        json={
            "name": "Director",
            "system_prompt": "You are an image director. Keep replies short.",
            "sampling": {"temperature": 0.3, "top_k": "40"},
        },
    )
    assert resp.status_code == 200, resp.text
    detail = resp.json()
    assert detail["system_prompt_custom"]
    assert detail["sampling"] == {"temperature": 0.3, "top_k": 40}

    cid, events = await _say(chat, "hello there")
    assert events[-1][0] == "done"
    sent = fake_llama.requests[-1]
    assert sent["messages"][0]["content"].startswith("You are an image director.")
    assert (sent["temperature"], sent["top_k"]) == (0.3, 40)

    messages = (await chat.get(f"/v1/conversations/{cid}/messages")).json()["items"]
    [reply] = [m for m in messages if m["role"] == "assistant"]
    assert reply["agent"] == agent["id"]
    conversation = (await chat.get(f"/v1/conversations/{cid}")).json()
    assert conversation["metadata"]["agent"] == agent["id"]
    assert (await _only_agent(chat))["turns"] == 1

    # Back to the built-in prompt; sampling settings go one by one.
    resp = await chat.patch(
        f"/v1/agents/{agent['id']}", json={"system_prompt": "", "sampling": {"top_k": None}}
    )
    assert not resp.json()["system_prompt_custom"]
    assert resp.json()["sampling"] == {"temperature": 0.3}
    bad = await chat.patch(f"/v1/agents/{agent['id']}", json={"sampling": {"seed": 3}})
    assert bad.status_code == 422


async def test_tools_the_agent_has_off_are_not_offered(
    studio: httpx.AsyncClient, fake_llama: FakeLlama
) -> None:
    agent = await _only_agent(studio)
    resp = await studio.patch(
        f"/v1/agents/{agent['id']}", json={"tools": {"set_portrait": False, "edit_image": False}}
    )
    tools = {t["name"]: t["enabled"] for t in resp.json()["tool_list"]}
    assert tools == {"generate_image": True, "edit_image": False, "set_portrait": False}
    await _say(studio, "hello")
    assert [t["function"]["name"] for t in fake_llama.requests[-1]["tools"]] == ["generate_image"]


async def test_a_backend_without_an_agent_gets_no_set_portrait(
    studio: httpx.AsyncClient, fake_llama: FakeLlama
) -> None:
    cid, events = await _say(studio, "hello", backend="llama")
    assert events[-1][0] == "done"
    offered = [t["function"]["name"] for t in fake_llama.requests[-1]["tools"]]
    assert offered == ["generate_image", "edit_image"]
    messages = (await studio.get(f"/v1/conversations/{cid}/messages")).json()["items"]
    assert [m["agent"] for m in messages if m["role"] == "assistant"] == [None]


async def test_set_portrait_paints_a_self_portrait(
    studio: httpx.AsyncClient, fake_llama: FakeLlama, fake_sd: FakeSdCpp
) -> None:
    agent = await _only_agent(studio)
    fake_llama.queue(
        FakeLlama.tool_call("set_portrait", {"prompt": "a lens-shaped eye in a square frame"}),
        FakeLlama.text("That's me now."),
    )
    cid, events = await _say(studio, "paint yourself a new portrait")
    assert events[-1][0] == "done"

    detail = (await studio.get(f"/v1/agents/{agent['id']}")).json()
    p = detail["portrait"]
    assert (p["key"], p["source"], p["origin"]) == ("self-1", "self", "SELF-PORTRAIT")
    [image] = p["images"]
    assert image["prompt"] == "a lens-shaped eye in a square frame"
    assert (image["conversation_id"], image["label"], image["version"]) == (cid, "img-1", 1)
    assert p["prompt"] == image["prompt"]

    # The painting stays in the conversation; the portrait is a square copy of it.
    got = await studio.get(image["url"])
    assert got.status_code == 200 and got.headers["content-type"] == "image/webp"
    with Image.open(io.BytesIO(got.content)) as img:
        assert img.width == img.height
    jobs = (await studio.get(f"/v1/conversations/{cid}/jobs")).json()["items"]
    assert jobs[0]["params"]["requested"]["aspect_ratio"] == "1:1"
    stored = [d["message"] for e, _, d in events if e == "message"]
    result = next(m for m in stored if m["role"] == "tool")
    assert json.loads(result["content_text"])["portrait"] == {"version": 1, "label": "img-1"}
    assert detail["images"] == 1


async def test_paint_another_runs_in_a_conversation_of_its_own(
    studio: httpx.AsyncClient, fake_llama: FakeLlama, fake_sd: FakeSdCpp
) -> None:
    agent = await _only_agent(studio)
    for version in (1, 2):
        resp = await studio.post(f"/v1/agents/{agent['id']}/portrait/paint")
        assert resp.status_code == 202, resp.text
        cid = resp.json()["conversation_id"]
        await _turn_over(studio, cid)
        detail = (await studio.get(f"/v1/agents/{agent['id']}")).json()
        assert detail["portrait"]["key"] == f"self-{version}"
        assert detail["portrait"]["painting"] is None
        # Only set_portrait was offered, and the model had to call it.
        sent = next(r for r in fake_llama.requests if r["messages"][-1]["role"] == "user")
        assert [t["function"]["name"] for t in sent["tools"]] == ["set_portrait"]
        assert sent["tool_choice"] == "required"
        fake_llama.requests.clear()
    conversation = (await studio.get(f"/v1/conversations/{cid}")).json()
    assert conversation["title"] == f"Self-portrait · {agent['name']}"
    # An earlier self-portrait can be shown again.
    resp = await studio.put(f"/v1/agents/{agent['id']}/portrait", json={"key": "self-1"})
    assert resp.json()["portrait"]["key"] == "self-1"


async def test_an_upload_becomes_the_portrait(chat: httpx.AsyncClient) -> None:
    agent = await _only_agent(chat)
    resp = await chat.post(
        f"/v1/agents/{agent['id']}/portrait/upload",
        files={"file": ("me.png", image_bytes(300, 200), "image/png")},
    )
    assert resp.status_code == 200, resp.text
    p = resp.json()["portrait"]
    assert (p["key"], p["origin"]) == ("upload", "UPLOADED")
    with Image.open(io.BytesIO((await chat.get(p["url"])).content)) as img:
        assert img.size == (200, 200)
    shown = await chat.put(f"/v1/agents/{agent['id']}/portrait", json={"key": "none"})
    assert shown.json()["portrait"]["key"] is None
    missing = await chat.put(f"/v1/agents/{agent['id']}/portrait", json={"key": "self-3"})
    assert missing.status_code == 422
    junk = await chat.post(
        f"/v1/agents/{agent['id']}/portrait/upload", files={"file": ("x.png", b"nope")}
    )
    assert junk.status_code == 422


async def test_a_launched_engine_runs_its_agents_model(
    make_settings: SettingsFactory, fake_llama: FakeLlama, tmp_path: Path
) -> None:
    qwen = tmp_path / "models" / "qwen" / "Qwen3-VL-8B-Instruct-Q4_K_M.gguf"
    qwen.parent.mkdir(parents=True)
    qwen.write_bytes(gguf(4000, 1))
    (qwen.parent / "mmproj-F16.gguf").write_bytes(gguf(1000, 2))
    gemma = tmp_path / "models" / "gemma" / "gemma-3-12b-it-Q5_K_M.gguf"
    gemma.parent.mkdir()
    gemma.write_bytes(gguf(5000, 3))
    settings = make_settings(
        backends={"llama": llama_config(fake_llama.url, launch=LlamaCppLaunch(model=qwen))},
        routing=RoutingConfig(chat="llama"),
    )
    async for client in _client_for(settings):
        first = await _only_agent(client)
        assert (first["name"], first["own_files"], first["quant"]) == (
            "Qwen3-VL-8B-Instruct",
            True,
            "Q4_K_M",
        )
        # Its folder has a vision projector it isn't using yet.
        assert first["missing"] == ["mmproj"]
        engine = (await client.get("/v1/engines/llama")).json()
        assert engine["agent"] == first["id"]

        resp = await client.post(
            "/v1/agents", json={"path": str(gemma), "engine": "llama", "name": "Gemma"}
        )
        assert resp.status_code == 201, resp.text
        second = resp.json()
        assert second["use"]["runs_it"] is False
        assert second["use"]["runs_other"] == "Qwen3-VL-8B-Instruct"

        # A turn as an agent its engine doesn't run is refused, until the engine runs it.
        cid = (await client.post("/v1/conversations", json={})).json()["id"]
        refused = await client.post(
            f"/v1/conversations/{cid}/messages", json={"content": "hi", "agent": second["id"]}
        )
        assert refused.status_code == 409, refused.text
        used = await client.post(f"/v1/agents/{second['id']}/use", json={"start": False})
        assert used.status_code == 200, used.text
        engine = (await client.get("/v1/engines/llama")).json()
        assert engine["agent"] == second["id"]
        assert next(f for f in engine["files"] if f["key"] == "model")["path"] == str(gemma)
        _, events = await _say(client, "hi", cid=cid, agent=second["id"])
        assert events[-1][0] == "done"

        # One an engine runs can't be forgotten; the other can.
        assert (await client.delete(f"/v1/agents/{second['id']}")).status_code == 409
        assert (await client.delete(f"/v1/agents/{first['id']}")).status_code == 204
        names = [a["name"] for a in (await client.get("/v1/agents")).json()["items"]]
        assert names == ["Gemma"]
