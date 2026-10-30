"""POST /messages end to end, with the fake llama-server as the chat engine."""

from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from imaginer.backends import registry as registry_module
from imaginer.core.config import Settings
from imaginer.main import create_app
from tests.conftest import SettingsFactory, llama_config, refusing_llama
from tests.fakes.fake_llama import FakeLlama, Script
from tests.helpers import image_bytes, parse_sse

Event = tuple[str, int, dict[str, Any]]


@pytest.fixture
async def chat_app(chat_settings: Settings) -> AsyncIterator[FastAPI]:
    app = create_app(chat_settings)
    async with app.router.lifespan_context(app):
        yield app


@pytest.fixture
async def chat(chat_app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=chat_app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


async def _conversation(client: httpx.AsyncClient) -> str:
    return str((await client.post("/v1/conversations", json={})).json()["id"])


async def _send(client: httpx.AsyncClient, cid: str, **body: Any) -> httpx.Response:
    return await client.post(f"/v1/conversations/{cid}/messages", json=body)


async def _turn(client: httpx.AsyncClient, cid: str, **body: Any) -> list[Event]:
    resp = await _send(client, cid, **body)
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert resp.headers["x-turn-id"]
    return parse_sse(resp.text)


async def test_streamed_turn(chat: httpx.AsyncClient, fake_llama: FakeLlama) -> None:
    fake_llama.queue(FakeLlama.text("Hello! How can I help?", pieces=3))
    cid = await _conversation(chat)
    events = await _turn(chat, cid, content="hi there")

    assert [e[0] for e in events] == [
        "message",  # user message stored
        "message",  # assistant placeholder (streaming)
        "delta",
        "delta",
        "delta",
        "message",  # assistant finalised
        "done",
    ]
    assert [e[1] for e in events] == list(range(1, 8))
    user = events[0][2]["message"]
    assert (user["role"], user["content_text"], user["seq"]) == ("user", "hi there", 1)
    placeholder = events[1][2]["message"]
    assert (placeholder["role"], placeholder["status"]) == ("assistant", "streaming")
    assert "".join(e[2]["content"] for e in events if e[0] == "delta") == "Hello! How can I help?"
    assert {e[2]["message_id"] for e in events if e[0] == "delta"} == {placeholder["id"]}
    final = events[5][2]["message"]
    assert (final["status"], final["content_text"]) == ("complete", "Hello! How can I help?")
    assert (final["backend"], final["model"]) == ("llama", "Qwen3VL-8B-Instruct-Q4_K_M.gguf")
    done = events[6][2]
    assert done["finish_reason"] == "stop"
    assert done["message_ids"] == [user["id"], placeholder["id"]]
    assert done["usage"]["timings"]["predicted_n"] == 3

    stored = (await chat.get(f"/v1/conversations/{cid}/messages")).json()["items"]
    assert [(m["role"], m["status"], m["content_text"]) for m in stored] == [
        ("user", "complete", "hi there"),
        ("assistant", "complete", "Hello! How can I help?"),
    ]
    assert stored[1]["metadata"]["finish_reason"] == "stop"


async def test_history_and_reasoning(chat: httpx.AsyncClient, fake_llama: FakeLlama) -> None:
    cid = await _conversation(chat)
    await _turn(chat, cid, content="my name is Ada")
    fake_llama.queue(FakeLlama.text("Ada.", reasoning="The user said it earlier."))
    events = await _turn(chat, cid, content="what is my name?")

    sent = fake_llama.requests[-1]["messages"]
    assert [m["role"] for m in sent] == ["system", "user", "assistant", "user"]
    assert sent[2]["content"] == "You said: my name is Ada"
    first_delta = next(e[2] for e in events if e[0] == "delta")
    assert first_delta["reasoning"] == "The user said it earlier."
    final = [e for e in events if e[0] == "message"][-1][2]["message"]
    assert final["reasoning_text"] == "The user said it earlier."


async def test_image_attachments_reach_a_vision_model(
    chat: httpx.AsyncClient, fake_llama: FakeLlama
) -> None:
    cid = await _conversation(chat)
    upload = await chat.post(
        f"/v1/conversations/{cid}/artifacts",
        files={"file": ("cat.png", image_bytes(64, 64), "image/png")},
    )
    artifact = upload.json()
    events = await _turn(chat, cid, content="what's in this picture?", attachments=[artifact["id"]])

    user = events[0][2]["message"]
    assert user["attachments"] == [artifact["id"]]
    parts = fake_llama.requests[-1]["messages"][-1]["content"]
    assert [p["type"] for p in parts] == ["text", "text", "image_url"]
    assert parts[2]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert "img-1" in fake_llama.requests[-1]["messages"][0]["content"]  # manifest
    final = [e for e in events if e[0] == "message"][-1][2]["message"]
    assert final["content_text"].endswith("(saw 1 image)")
    # The upload now records the message that introduced it.
    assert (await chat.get(f"/v1/artifacts/{artifact['id']}")).json()["message_id"] == user["id"]


async def test_image_to_a_blind_model_is_refused(
    chat: httpx.AsyncClient, fake_llama: FakeLlama
) -> None:
    fake_llama.vision = False
    cid = await _conversation(chat)
    upload = await chat.post(
        f"/v1/conversations/{cid}/artifacts", files={"file": ("a.png", image_bytes(), "image/png")}
    )
    resp = await _send(chat, cid, content="see?", attachments=[upload.json()["id"]])
    assert resp.status_code == 422
    assert resp.json()["code"] == "capability_unavailable"
    assert resp.json()["missing"] == ["vision_in"]
    assert (await chat.get(f"/v1/conversations/{cid}/messages")).json()["items"] == []


@pytest.mark.parametrize(
    ("body", "status", "code"),
    [
        ({"content": "   "}, 422, "invalid_request"),
        (
            {"content": "x", "attachments": ["00000000-0000-7000-8000-000000000000"]},
            422,
            "invalid_request",
        ),
        ({"content": "x", "backend": "nope"}, 422, "capability_unavailable"),
        ({"content": "x", "temperature": 2}, 422, "validation_error"),
    ],
)
async def test_rejected_before_streaming(
    chat: httpx.AsyncClient, body: dict[str, Any], status: int, code: str
) -> None:
    cid = await _conversation(chat)
    resp = await _send(chat, cid, **body)
    assert resp.status_code == status
    assert resp.headers["content-type"] == "application/problem+json"
    assert resp.json()["code"] == code


async def test_attachment_from_another_conversation(chat: httpx.AsyncClient) -> None:
    a, b = await _conversation(chat), await _conversation(chat)
    upload = await chat.post(
        f"/v1/conversations/{a}/artifacts", files={"file": ("a.png", image_bytes(), "image/png")}
    )
    resp = await _send(chat, b, content="x", attachments=[upload.json()["id"]])
    assert resp.status_code == 422
    assert resp.json()["missing"] == [upload.json()["id"]]


async def test_unknown_conversation(chat: httpx.AsyncClient) -> None:
    resp = await _send(chat, "00000000-0000-7000-8000-000000000000", content="x")
    assert resp.status_code == 404


async def test_engine_down_is_503_and_stores_nothing(
    make_settings: SettingsFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(registry_module.FACTORIES, "llamacpp", refusing_llama)
    app = create_app(make_settings(backends={"llama": llama_config("http://refused")}))
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            cid = await _conversation(client)
            resp = await _send(client, cid, content="anyone there?")
            assert resp.status_code == 503
            assert resp.json()["code"] == "engine_unavailable"
            assert (await client.get(f"/v1/conversations/{cid}/messages")).json()["items"] == []


async def test_engine_error_becomes_an_error_event(
    chat: httpx.AsyncClient, fake_llama: FakeLlama
) -> None:
    fake_llama.queue(FakeLlama.http_error(400, "the request exceeds the available context size"))
    cid = await _conversation(chat)
    events = await _turn(chat, cid, content="x" * 50)
    assert events[-1][0] == "error"
    error = events[-1][2]
    assert (error["code"], error["status"]) == ("engine_error", 502)
    assert "exceeds the available context" in error["detail"]
    stored = (await chat.get(f"/v1/conversations/{cid}/messages")).json()["items"]
    assert stored[1]["status"] == "error"
    assert stored[1]["metadata"]["error"]["code"] == "engine_error"


async def test_partial_text_survives_a_mid_stream_failure(
    chat: httpx.AsyncClient, fake_llama: FakeLlama
) -> None:
    fake_llama.queue(
        Script(chunks=[{"content": "The answer "}, {"content": "is"}], mid_stream_error="crash")
    )
    cid = await _conversation(chat)
    events = await _turn(chat, cid, content="question")
    assert [e[0] for e in events][-2:] == ["message", "error"]
    final = events[-2][2]["message"]
    assert (final["status"], final["content_text"]) == ("error", "The answer is")


async def test_a_stream_cut_off_is_an_error_not_an_answer(
    chat: httpx.AsyncClient, fake_llama: FakeLlama
) -> None:
    fake_llama.queue(Script(chunks=[{"content": "The answer "}, {"content": "is"}], cut_off=True))
    cid = await _conversation(chat)
    events = await _turn(chat, cid, content="question")
    assert [e[0] for e in events][-2:] == ["message", "error"]
    final = events[-2][2]["message"]
    assert (final["status"], final["content_text"]) == ("error", "The answer is")
    assert final["metadata"]["error"]["code"] == "engine_interrupted"
    assert events[-1][2]["code"] == "engine_interrupted"


async def test_a_conversation_can_continue_after_a_failure(
    chat: httpx.AsyncClient, fake_llama: FakeLlama
) -> None:
    fake_llama.queue(FakeLlama.http_error(500, "boom"))
    cid = await _conversation(chat)
    assert (await _turn(chat, cid, content="one"))[-1][0] == "error"
    assert (await _turn(chat, cid, content="two"))[-1][0] == "done"
    seqs = [m["seq"] for m in (await chat.get(f"/v1/conversations/{cid}/messages")).json()["items"]]
    assert seqs == [1, 2, 3, 4]


async def test_no_turn_running(chat: httpx.AsyncClient) -> None:
    cid = await _conversation(chat)
    assert (await chat.get(f"/v1/conversations/{cid}/turn")).status_code == 404
    assert (await chat.post(f"/v1/conversations/{cid}/turn/cancel")).status_code == 404
    assert (await chat.get(f"/v1/conversations/{cid}/turn/events")).status_code == 404
