"""Editing an upload through the chat, lineage over HTTP, and the job cancel endpoint."""

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from imaginer.core.config import Settings
from imaginer.main import create_app
from tests.fakes.fake_llama import FakeLlama
from tests.fakes.fake_sdcpp import FakeSdCpp
from tests.helpers import image_bytes, parse_sse

Event = tuple[str, int, dict[str, Any]]


@pytest.fixture
async def studio(studio_settings: Settings) -> AsyncIterator[httpx.AsyncClient]:
    app: FastAPI = create_app(studio_settings)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


async def _upload(client: httpx.AsyncClient, cid: str, data: bytes) -> dict[str, Any]:
    resp = await client.post(
        f"/v1/conversations/{cid}/artifacts", files={"file": ("in.png", data, "image/png")}
    )
    body: dict[str, Any] = resp.json()
    return body


async def test_edit_an_upload_through_chat(
    studio: httpx.AsyncClient, fake_llama: FakeLlama, fake_sd: FakeSdCpp
) -> None:
    cid = (await studio.post("/v1/conversations", json={})).json()["id"]
    photo = await _upload(studio, cid, image_bytes(320, 256, color=(40, 120, 40)))
    fake_llama.queue(
        FakeLlama.tool_call("edit_image", {"images": ["img-1"], "prompt": "make the field purple"}),
        FakeLlama.text("Done: img-2 is the purple version."),
    )
    resp = await studio.post(
        f"/v1/conversations/{cid}/messages",
        json={"content": "make the grass purple", "attachments": [photo["id"]]},
    )
    events: list[Event] = parse_sse(resp.text)
    assert events[-1][0] == "done"

    # The model saw the upload as pixels with its label, and the edit tool was offered.
    first = fake_llama.requests[-2]
    assert {t["function"]["name"] for t in first["tools"]} == {"generate_image", "edit_image"}
    user_parts = first["messages"][-1]["content"]
    assert {"type": "text", "text": "img-1:"} in user_parts

    (created,) = [d for e, _, d in events if e == "artifact_created"]
    edited = created["artifact"]
    assert (edited["label"], edited["source"]) == ("img-2", "edited")
    assert (edited["width"], edited["height"]) == (320, 256)

    child = (await studio.get(f"/v1/artifacts/{edited['id']}/lineage")).json()
    assert [(p["label"], p["relation"]) for p in child["parents"]] == [("img-1", "edit_source")]
    parent = (await studio.get(f"/v1/artifacts/{photo['id']}/lineage")).json()
    assert [(c["label"], c["relation"]) for c in parent["children"]] == [("img-2", "edit_source")]

    tool = next(
        d["message"] for e, _, d in events if e == "message" and d["message"]["role"] == "tool"
    )
    assert json.loads(tool["content_text"])["edited"] == ["img-1"]
    assert tool["attachments"] == [edited["id"]]

    job = (await studio.get(f"/v1/conversations/{cid}/jobs")).json()["items"][0]
    assert (job["task"], job["status"]) == ("edit", "succeeded")
    assert job["params"]["requested"]["images"] == ["img-1"]


async def test_cancel_endpoint_errors(studio: httpx.AsyncClient, fake_llama: FakeLlama) -> None:
    missing = await studio.post("/v1/jobs/00000000-0000-7000-8000-000000000000/cancel")
    assert missing.status_code == 404

    fake_llama.queue(FakeLlama.tool_call("generate_image", {"prompt": "x"}), FakeLlama.text("ok"))
    cid = (await studio.post("/v1/conversations", json={})).json()["id"]
    await studio.post(f"/v1/conversations/{cid}/messages", json={"content": "draw"})
    job = (await studio.get(f"/v1/conversations/{cid}/jobs")).json()["items"][0]
    finished = await studio.post(f"/v1/jobs/{job['id']}/cancel")
    assert finished.status_code == 409
    assert finished.json()["code"] == "conflict"


async def test_jobs_of_deleted_conversations_are_gone(
    studio: httpx.AsyncClient, fake_llama: FakeLlama
) -> None:
    fake_llama.queue(FakeLlama.tool_call("generate_image", {"prompt": "x"}), FakeLlama.text("ok"))
    cid = (await studio.post("/v1/conversations", json={})).json()["id"]
    await studio.post(f"/v1/conversations/{cid}/messages", json={"content": "draw"})
    job_id = (await studio.get(f"/v1/conversations/{cid}/jobs")).json()["items"][0]["id"]
    assert (await studio.delete(f"/v1/conversations/{cid}")).status_code == 204
    assert (await studio.get(f"/v1/jobs/{job_id}")).status_code == 404
    assert (await studio.post(f"/v1/jobs/{job_id}/cancel")).status_code == 404
