"""The tool loop end to end: fake llama-server calls generate_image, fake sd-server draws."""

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from imaginer.core.config import OrchestrationConfig, RoutingConfig, Settings
from imaginer.domain.enums import Capability
from imaginer.main import create_app
from tests.conftest import SettingsFactory, llama_config, sd_config
from tests.fakes.fake_llama import FakeLlama
from tests.fakes.fake_sdcpp import FakeSdCpp
from tests.helpers import parse_sse

Event = tuple[str, int, dict[str, Any]]


async def _client_for(settings: Settings) -> AsyncIterator[httpx.AsyncClient]:
    app: FastAPI = create_app(settings)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


@pytest.fixture
async def studio(studio_settings: Settings) -> AsyncIterator[httpx.AsyncClient]:
    async for client in _client_for(studio_settings):
        yield client


async def _turn(client: httpx.AsyncClient, content: str) -> tuple[str, list[Event]]:
    cid = (await client.post("/v1/conversations", json={})).json()["id"]
    resp = await client.post(f"/v1/conversations/{cid}/messages", json={"content": content})
    assert resp.status_code == 200, resp.text
    return cid, parse_sse(resp.text)


def _of(events: list[Event], kind: str) -> list[dict[str, Any]]:
    return [data for event, _, data in events if event == kind]


async def test_generate_image_round_trip(
    studio: httpx.AsyncClient, fake_llama: FakeLlama, fake_sd: FakeSdCpp
) -> None:
    fake_sd.step_delay_s = 0.02
    fake_llama.queue(
        FakeLlama.tool_call(
            "generate_image", {"prompt": "a lighthouse at dusk", "aspect_ratio": "16:9"}
        ),
        FakeLlama.text("Here is your lighthouse: img-1."),
    )
    cid, events = await _turn(studio, "draw a lighthouse")

    kinds = [e[0] for e in events]
    assert kinds[-1] == "done"
    assert kinds.index("tool_call") < kinds.index("job_update") < kinds.index("artifact_created")
    (call,) = _of(events, "tool_call")
    assert call["name"] == "generate_image"
    assert json.loads(call["arguments"])["aspect_ratio"] == "16:9"

    statuses = [d["job"]["status"] for d in _of(events, "job_update")]
    assert statuses[0] == "queued" and statuses[-1] == "succeeded" and "running" in statuses
    progress = [d["job"]["progress"] for d in _of(events, "job_update") if d["job"]["progress"]]
    assert progress[-1] == {"image": 1, "images": 1, "step": 3, "total_steps": 3}

    (created,) = _of(events, "artifact_created")
    artifact = created["artifact"]
    assert (artifact["label"], artifact["source"]) == ("img-1", "generated")
    assert (artifact["width"], artifact["height"]) == (448, 256)

    messages = (await studio.get(f"/v1/conversations/{cid}/messages")).json()["items"]
    assert [(m["role"], m["status"]) for m in messages] == [
        ("user", "complete"),
        ("assistant", "complete"),
        ("tool", "complete"),
        ("assistant", "complete"),
    ]
    caller, tool, final = messages[1], messages[2], messages[3]
    assert caller["tool_calls"][0]["id"] == call["tool_call_id"]
    assert tool["tool_call_id"] == call["tool_call_id"]
    assert tool["attachments"] == [artifact["id"]]
    result = json.loads(tool["content_text"])
    assert result["status"] == "succeeded"
    assert result["images"][0]["label"] == "img-1"
    assert final["content_text"] == "Here is your lighthouse: img-1."
    assert artifact["message_id"] == caller["id"]
    assert created["job_id"] == result["job_id"]

    # What the model saw: tools offered with guidance, then its own call and the result.
    first, second = fake_llama.requests[-2:]
    assert [t["function"]["name"] for t in first["tools"]] == ["generate_image", "edit_image"]
    assert "generate_image" in first["messages"][0]["content"]
    assert [m["role"] for m in second["messages"]] == ["system", "user", "assistant", "tool"]
    assert second["messages"][2]["tool_calls"][0]["function"]["name"] == "generate_image"
    assert "img-1: generated image, 448x256" in second["messages"][0]["content"]

    # The job is queryable, with per-output seeds.
    jobs = (await studio.get(f"/v1/conversations/{cid}/jobs")).json()["items"]
    assert len(jobs) == 1
    job = (await studio.get(f"/v1/jobs/{jobs[0]['id']}")).json()
    assert job["status"] == "succeeded"
    assert job["outputs"][0]["artifact_id"] == artifact["id"]
    assert job["outputs"][0]["seed"] == job["seed"]
    assert job["tool_call_id"] == call["tool_call_id"]
    assert job["request_message_id"] == caller["id"]
    assert job["params"]["resolved"]["width"] == 448


async def test_user_image_options_override_the_model(
    studio: httpx.AsyncClient, fake_llama: FakeLlama, fake_sd: FakeSdCpp
) -> None:
    fake_sd.step_delay_s = 0.0
    fake_llama.queue(
        FakeLlama.tool_call(
            "generate_image",
            {"prompt": "a fox", "aspect_ratio": "16:9", "width": 640, "n": 1, "seed": 5},
        ),
        FakeLlama.text("Two foxes."),
    )
    cid = (await studio.post("/v1/conversations", json={})).json()["id"]
    options = {"aspect_ratio": "3:4", "n": 2, "seed": 41207, "backend": "sd"}
    resp = await studio.post(
        f"/v1/conversations/{cid}/messages",
        json={"content": "draw a fox", "image_options": options},
    )
    events = parse_sse(resp.text)
    assert events[-1][0] == "done"

    assert [r["seed"] for r in fake_sd.requests] == [41207, 41208]
    created = [d["artifact"] for d in _of(events, "artifact_created")]
    assert len(created) == 2
    assert all(a["height"] > a["width"] for a in created)  # 3:4 won over 16:9 and width=640

    messages = (await studio.get(f"/v1/conversations/{cid}/messages")).json()["items"]
    assert messages[0]["metadata"] == {"image_options": options}
    result = json.loads(messages[2]["content_text"])
    assert result["user_options"] == options
    (job,) = (await studio.get(f"/v1/conversations/{cid}/jobs")).json()["items"]
    assert job["params"]["requested"]["n"] == 2
    assert "width" not in job["params"]["requested"]


async def test_exact_size_steps_cfg_and_negative_from_the_user(
    studio: httpx.AsyncClient, fake_llama: FakeLlama, fake_sd: FakeSdCpp
) -> None:
    fake_sd.step_delay_s = 0.0
    fake_llama.queue(
        FakeLlama.tool_call("generate_image", {"prompt": "a fox", "aspect_ratio": "16:9"}),
        FakeLlama.text("Done."),
    )
    cid = (await studio.post("/v1/conversations", json={})).json()["id"]
    options = {
        "width": 640,
        "height": 480,
        "steps": 8,
        "cfg_scale": 4.5,
        "negative_prompt": "text",
    }
    resp = await studio.post(
        f"/v1/conversations/{cid}/messages",
        json={"content": "draw a fox", "image_options": options},
    )
    assert parse_sse(resp.text)[-1][0] == "done"
    (body,) = fake_sd.requests
    assert (body["width"], body["height"]) == (640, 480)
    assert body["sample_params"]["sample_steps"] == 8
    assert body["sample_params"]["guidance"] == {"txt_cfg": 4.5}
    assert body["negative_prompt"] == "text"
    (job,) = (await studio.get(f"/v1/conversations/{cid}/jobs")).json()["items"]
    assert job["params"]["requested"]["steps"] == 8
    assert "aspect_ratio" not in job["params"]["requested"]


@pytest.mark.parametrize(
    ("options", "fragment"),
    [
        ({"width": 640}, "give width and height together"),
        ({"mask": "img-1"}, "should match pattern"),
        ({"mask": "mask-9"}, "no mask labelled 'mask-9'"),
    ],
)
async def test_bad_user_image_options_are_rejected(
    studio: httpx.AsyncClient, options: dict[str, Any], fragment: str
) -> None:
    cid = (await studio.post("/v1/conversations", json={})).json()["id"]
    resp = await studio.post(
        f"/v1/conversations/{cid}/messages", json={"content": "x", "image_options": options}
    )
    assert resp.status_code in (400, 422), resp.text
    assert fragment in resp.text


async def test_image_backends_report_their_defaults(studio: httpx.AsyncClient) -> None:
    backends = {b["name"]: b for b in (await studio.get("/v1/backends")).json()}
    assert backends["llama"]["image_defaults"] is None
    defaults = backends["sd"]["image_defaults"]
    assert set(defaults) == {"width", "height", "steps", "cfg_scale", "sampler", "size_multiple"}
    assert defaults["size_multiple"] == 32


async def test_unusable_image_backend_is_rejected_before_the_turn(
    studio: httpx.AsyncClient, fake_llama: FakeLlama
) -> None:
    cid = (await studio.post("/v1/conversations", json={})).json()["id"]
    for backend in ["nope", "llama"]:
        resp = await studio.post(
            f"/v1/conversations/{cid}/messages",
            json={"content": "draw", "image_options": {"backend": backend}},
        )
        assert resp.status_code == 422, resp.text
        assert resp.json()["code"] == "capability_unavailable"
    assert (await studio.get(f"/v1/conversations/{cid}/messages")).json()["items"] == []
    assert fake_llama.requests == []


async def test_engine_failure_is_explained_not_fatal(
    studio: httpx.AsyncClient, fake_llama: FakeLlama, fake_sd: FakeSdCpp
) -> None:
    fake_sd.fail_next = "CUDA out of memory"
    fake_llama.queue(
        FakeLlama.tool_call("generate_image", {"prompt": "x"}),
        FakeLlama.text("Sorry, the image engine ran out of memory."),
    )
    _cid, events = await _turn(studio, "draw")
    assert events[-1][0] == "done"
    assert [d["job"]["status"] for d in _of(events, "job_update")][-1] == "failed"
    tool = next(d["message"] for d in _of(events, "message") if d["message"]["role"] == "tool")
    result = json.loads(tool["content_text"])
    assert result["error"]["code"] == "engine_error"
    assert "CUDA out of memory" in result["error"]["detail"]
    assert fake_llama.requests[-1]["messages"][-1]["role"] == "tool"


async def test_bad_tool_arguments_reach_the_model(
    studio: httpx.AsyncClient, fake_llama: FakeLlama, fake_sd: FakeSdCpp
) -> None:
    fake_llama.queue(
        FakeLlama.tool_call("generate_image", {"prompt": "x", "n": 99}),
        FakeLlama.text("Let me fix that."),
    )
    _, events = await _turn(studio, "draw 99")
    tool = next(d["message"] for d in _of(events, "message") if d["message"]["role"] == "tool")
    assert json.loads(tool["content_text"])["error"]["code"] == "invalid_arguments"
    assert fake_sd.requests == []


async def test_tool_rounds_are_capped(
    studio: httpx.AsyncClient, fake_llama: FakeLlama, fake_sd: FakeSdCpp
) -> None:
    # max_tool_iterations=2: after two rounds of calls, the model must answer.
    for i in range(3):
        fake_llama.queue(
            FakeLlama.tool_call("generate_image", {"prompt": f"p{i}"}, call_id=f"c{i}")
        )
    cid, events = await _turn(studio, "loop forever")
    assert events[-1][0] == "done"
    assert len(fake_sd.requests) == 2
    assert [r.get("tool_choice") for r in fake_llama.requests[-3:]] == [None, None, "none"]
    final = (await studio.get(f"/v1/conversations/{cid}/messages")).json()["items"][-1]
    assert final["role"] == "assistant"
    assert final["tool_calls"] is None  # the ignored third call is not stored


async def test_no_tools_without_tool_support(
    studio: httpx.AsyncClient, fake_llama: FakeLlama
) -> None:
    fake_llama.tools = False
    await _turn(studio, "hello")
    assert "tools" not in fake_llama.requests[-1]
    assert "generate_image" not in fake_llama.requests[-1]["messages"][0]["content"]


async def test_no_tools_without_an_image_backend(
    chat_settings: Settings, fake_llama: FakeLlama
) -> None:
    async for client in _client_for(chat_settings):
        await _turn(client, "hello")
    assert "tools" not in fake_llama.requests[-1]


async def test_transparent_needs_rgba(
    make_settings: SettingsFactory, fake_llama: FakeLlama, fake_sd: FakeSdCpp
) -> None:
    settings = make_settings(
        backends={
            "llama": llama_config(fake_llama.url),
            "sd": sd_config(fake_sd.url, declared_capabilities={Capability.T2I}),
        },
        routing=RoutingConfig(chat="llama", t2i="sd"),
        orchestration=OrchestrationConfig(probe_ttl_s=0),
    )
    fake_llama.queue(
        FakeLlama.tool_call("generate_image", {"prompt": "logo", "transparent": True}),
        FakeLlama.text("That engine can't do transparency."),
    )
    async for client in _client_for(settings):
        _, events = await _turn(client, "transparent logo please")
    tool = next(d["message"] for d in _of(events, "message") if d["message"]["role"] == "tool")
    assert json.loads(tool["content_text"])["error"]["code"] == "capability_unavailable"
    assert fake_sd.requests == []


async def test_unknown_job(studio: httpx.AsyncClient) -> None:
    assert (await studio.get("/v1/jobs/00000000-0000-7000-8000-000000000000")).status_code == 404
    cid = (await studio.post("/v1/conversations", json={})).json()["id"]
    assert (await studio.get(f"/v1/conversations/{cid}/jobs")).json()["items"] == []
