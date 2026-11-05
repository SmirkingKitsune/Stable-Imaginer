"""Prompt modes (raw, interpret, curate) and the quality check, end to end on the fakes."""

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from imaginer.core.config import OrchestrationConfig, QAConfig, RoutingConfig, Settings
from imaginer.domain.enums import PromptMode
from imaginer.main import create_app
from imaginer.orchestration.modes import MODE_GUIDANCE
from imaginer.orchestration.qa import QA_GUIDANCE, parse
from tests.conftest import SettingsFactory, llama_config, sd_config
from tests.fakes.fake_llama import FakeLlama, Script
from tests.fakes.fake_sdcpp import FakeSdCpp
from tests.helpers import image_bytes, parse_sse

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


@pytest.fixture
async def strict(
    make_settings: SettingsFactory, fake_llama: FakeLlama, fake_sd: FakeSdCpp
) -> AsyncIterator[httpx.AsyncClient]:
    """Like ``studio``, but the check may redo a call only once."""
    settings = make_settings(
        backends={"llama": llama_config(fake_llama.url), "sd": sd_config(fake_sd.url)},
        routing=RoutingConfig(chat="llama", t2i="sd", edit="sd"),
        orchestration=OrchestrationConfig(probe_ttl_s=0, qa=QAConfig(max_retries=1)),
    )
    async for client in _client_for(settings):
        yield client


@pytest.fixture
async def images_only(
    make_settings: SettingsFactory, fake_sd: FakeSdCpp
) -> AsyncIterator[httpx.AsyncClient]:
    """No chat backend at all: only raw mode can make anything."""
    settings = make_settings(
        backends={"sd": sd_config(fake_sd.url)},
        routing=RoutingConfig(t2i="sd", edit="sd"),
        orchestration=OrchestrationConfig(probe_ttl_s=0),
    )
    async for client in _client_for(settings):
        yield client


async def _conversation(client: httpx.AsyncClient) -> str:
    cid: str = (await client.post("/v1/conversations", json={})).json()["id"]
    return cid


async def _send(client: httpx.AsyncClient, cid: str, **body: Any) -> list[Event]:
    resp = await client.post(f"/v1/conversations/{cid}/messages", json=body)
    assert resp.status_code == 200, resp.text
    events = parse_sse(resp.text)
    assert events[-1][0] == "done", events[-1]
    return events


async def _messages(client: httpx.AsyncClient, cid: str) -> list[dict[str, Any]]:
    """Every message, on every branch (a redone attempt sits beside the kept one), by seq."""
    resp = await client.get(f"/v1/conversations/{cid}/messages", params={"all_branches": True})
    items: list[dict[str, Any]] = resp.json()["items"]
    return items


async def _jobs(client: httpx.AsyncClient, cid: str) -> list[dict[str, Any]]:
    items = (await client.get(f"/v1/conversations/{cid}/jobs")).json()["items"]
    return sorted(items, key=lambda j: j["id"])


def _verdict(**fields: Any) -> Script:
    return FakeLlama.text(json.dumps({"issues": [], **fields}), pieces=2)


def _system(request: dict[str, Any]) -> str:
    content: str = request["messages"][0]["content"]
    return content


# --- raw ----------------------------------------------------------------------


async def test_raw_mode_sends_the_words_to_the_image_model_as_they_are(
    studio: httpx.AsyncClient, fake_llama: FakeLlama, fake_sd: FakeSdCpp
) -> None:
    cid = await _conversation(studio)
    prompt = "masterpiece, a red fox in snow, (detailed fur:1.2)"
    events = await _send(studio, cid, content=prompt, mode="raw", qa=False)

    assert fake_llama.requests == []  # no LLM in the turn
    (body,) = fake_sd.requests
    assert body["prompt"] == prompt
    user, caller, result = await _messages(studio, cid)
    assert user["metadata"] == {"mode": "raw", "qa": False}
    assert caller["role"] == "assistant" and caller["backend"] is None
    assert caller["metadata"]["mode"] == "raw"
    (call,) = caller["tool_calls"]
    assert call["function"]["name"] == "generate_image"
    assert json.loads(call["function"]["arguments"]) == {"prompt": prompt}
    assert result["tool_call_id"] == call["id"]
    assert json.loads(result["content_text"])["status"] == "succeeded"
    assert result["attachments"]
    assert "qa" not in (result["metadata"] or {})
    assert events[-1][2]["finish_reason"] is None


async def test_raw_mode_with_an_attached_image_edits_it(
    studio: httpx.AsyncClient, fake_llama: FakeLlama, fake_sd: FakeSdCpp
) -> None:
    cid = await _conversation(studio)
    upload = await studio.post(
        f"/v1/conversations/{cid}/artifacts",
        files={"file": ("cat.png", image_bytes(64, 64), "image/png")},
    )
    artifact = upload.json()
    await _send(
        studio, cid, content="make it night", attachments=[artifact["id"]], mode="raw", qa=False
    )

    _, caller, result = await _messages(studio, cid)
    (call,) = caller["tool_calls"]
    assert call["function"]["name"] == "edit_image"
    assert json.loads(call["function"]["arguments"]) == {
        "images": [artifact["label"]],
        "prompt": "make it night",
    }
    assert json.loads(result["content_text"])["edited"] == [artifact["label"]]
    assert fake_llama.requests == []


async def test_raw_mode_needs_no_chat_backend_and_skips_the_check(
    images_only: httpx.AsyncClient,
) -> None:
    cid = await _conversation(images_only)
    await _send(images_only, cid, content="a lighthouse", mode="raw")

    *_, result = await _messages(images_only, cid)
    qa = result["metadata"]["qa"]
    assert qa["status"] == "skipped"
    assert qa["detail"] == "no chat engine that can see images is ready"
    assert json.loads(result["content_text"])["qa"] == qa


@pytest.mark.parametrize(
    ("body", "fragment"),
    [
        ({"content": "  ", "mode": "raw", "attachments": ["x"]}, "raw mode"),
        ({"content": "hi", "mode": "loud"}, "chat"),
    ],
)
async def test_bad_modes_are_rejected(
    studio: httpx.AsyncClient, body: dict[str, Any], fragment: str
) -> None:
    cid = await _conversation(studio)
    resp = await studio.post(f"/v1/conversations/{cid}/messages", json=body)
    assert resp.status_code in (400, 422), resp.text
    assert fragment in resp.text


async def test_modes_that_draw_need_an_image_backend(
    chat_client: httpx.AsyncClient,
) -> None:
    cid = await _conversation(chat_client)
    for mode in ("interpret", "curate", "raw"):
        resp = await chat_client.post(
            f"/v1/conversations/{cid}/messages", json={"content": "a fox", "mode": mode}
        )
        assert resp.status_code in (400, 422), resp.text
        assert "backend is available" in resp.text


@pytest.fixture
async def chat_client(chat_settings: Settings) -> AsyncIterator[httpx.AsyncClient]:
    async for client in _client_for(chat_settings):
        yield client


# --- interpret and curate -----------------------------------------------------


async def test_interpret_mode_must_call_the_image_tool_first(
    studio: httpx.AsyncClient, fake_llama: FakeLlama
) -> None:
    fake_llama.queue(
        FakeLlama.tool_call("generate_image", {"prompt": "a small boat on a calm lake"}),
        FakeLlama.text("Here it is: img-1."),
    )
    cid = await _conversation(studio)
    await _send(studio, cid, content="boat, lake, calm", mode="interpret")

    first, second = fake_llama.requests
    assert first["tool_choice"] == "required"
    assert MODE_GUIDANCE[PromptMode.INTERPRET] in _system(first)
    assert "tool_choice" not in second  # the reply after the image is free
    assert (await _messages(studio, cid))[0]["metadata"] == {"mode": "interpret"}


async def test_curate_mode_may_ask_before_drawing(
    studio: httpx.AsyncClient, fake_llama: FakeLlama, fake_sd: FakeSdCpp
) -> None:
    fake_llama.queue(
        FakeLlama.text("A cat doing what, and in which style? Or: a cosy oil painting.")
    )
    cid = await _conversation(studio)
    await _send(studio, cid, content="cat", mode="curate")

    (request,) = fake_llama.requests
    assert "tool_choice" not in request
    assert MODE_GUIDANCE[PromptMode.CURATE] in _system(request)
    assert fake_sd.requests == []
    assert [m["role"] for m in await _messages(studio, cid)] == ["user", "assistant"]


async def test_chat_mode_adds_no_mode_guidance(
    studio: httpx.AsyncClient, fake_llama: FakeLlama
) -> None:
    fake_llama.queue(FakeLlama.text("Hello!"))
    cid = await _conversation(studio)
    await _send(studio, cid, content="hi")

    system = _system(fake_llama.requests[0])
    assert not any(text in system for text in MODE_GUIDANCE.values())
    assert QA_GUIDANCE not in system  # studio's default has the check off


# --- the quality check --------------------------------------------------------


async def test_a_passing_check_keeps_the_image(
    studio: httpx.AsyncClient, fake_llama: FakeLlama, fake_sd: FakeSdCpp
) -> None:
    fake_llama.queue(
        FakeLlama.tool_call("generate_image", {"prompt": "a lighthouse at dusk"}),
        _verdict(verdict="pass"),
        FakeLlama.text("Here is img-1."),
    )
    cid = await _conversation(studio)
    events = await _send(studio, cid, content="draw a lighthouse", qa=True)

    caller, check, reply = fake_llama.requests
    assert QA_GUIDANCE in _system(caller)
    assert check["response_format"]["type"] == "json_schema"
    assert "tools" not in check
    parts = check["messages"][1]["content"]
    assert "Request: draw a lighthouse" in parts[0]["text"]
    assert "Prompt: a lighthouse at dusk" in parts[0]["text"]
    assert "steps 3" in parts[0]["text"]
    assert [p["type"] for p in parts[1:]] == ["text", "image_url"]

    shown = [
        d["message"]["metadata"].get("qa", {}).get("status")
        for e, _, d in events
        if e == "message" and d["message"]["role"] == "tool"
    ]
    assert shown == [None, "checking", "passed"]
    *_, result, final = await _messages(studio, cid)
    assert result["metadata"]["qa"] == {"status": "passed", "attempt": 1}
    assert json.loads(result["content_text"])["qa"]["status"] == "passed"
    assert final["content_text"] == "Here is img-1."
    assert len(fake_sd.requests) == 1
    # The reply sees the checked result.
    assert json.loads(reply["messages"][-1]["content"])["qa"]["status"] == "passed"


async def test_a_failed_check_redoes_the_call_with_its_changes(
    studio: httpx.AsyncClient, fake_llama: FakeLlama, fake_sd: FakeSdCpp
) -> None:
    fake_llama.queue(
        FakeLlama.tool_call("generate_image", {"prompt": "a hand holding a cup"}),
        _verdict(verdict="redo", issues=["six fingers"], steps=12, new_seed=False),
        _verdict(verdict="pass"),
        FakeLlama.text("Here is img-2."),
    )
    cid = await _conversation(studio)
    await _send(studio, cid, content="draw a hand holding a cup", qa=True)

    first, second = fake_sd.requests
    assert first["sample_params"]["sample_steps"] == 3
    assert second["sample_params"]["sample_steps"] == 12
    assert second["seed"] == first["seed"]  # same seed: refine, don't reroll
    assert second["prompt"] == first["prompt"]

    messages = await _messages(studio, cid)
    results = [m for m in messages if m["role"] == "tool"]
    rejected, kept = results
    assert rejected["parent_id"] == kept["parent_id"]  # attempts of one call
    assert rejected["metadata"]["qa"] == {
        "status": "redone",
        "attempt": 1,
        "issues": ["six fingers"],
        "changes": {"steps": 12},
    }
    assert kept["metadata"]["qa"] == {
        "status": "passed",
        "attempt": 2,
        "earlier": [{"issues": ["six fingers"], "changes": {"steps": 12}}],
    }
    assert kept["metadata"]["redo_of"] == rejected["id"]
    final = messages[-1]
    assert final["role"] == "assistant" and final["parent_id"] == kept["id"]
    branches = (await studio.get(f"/v1/conversations/{cid}/branches")).json()
    assert [b["name"] for b in branches] == ["main", "QA redo of img-1"]
    jobs = await _jobs(studio, cid)
    assert [j["params"]["requested"].get("steps") for j in jobs] == [None, 12]


async def test_the_check_revises_the_prompt_and_rerolls(
    studio: httpx.AsyncClient, fake_llama: FakeLlama, fake_sd: FakeSdCpp
) -> None:
    fake_llama.queue(
        FakeLlama.tool_call("generate_image", {"prompt": "a sign", "seed": 7}),
        _verdict(
            verdict="redo",
            issues=["the sign is blank"],
            prompt='a wooden sign that reads "OPEN"',
            cfg_scale=99,
            new_seed=True,
        ),
        _verdict(verdict="pass"),
        FakeLlama.text("Done."),
    )
    cid = await _conversation(studio)
    await _send(studio, cid, content='draw a sign saying "OPEN"', qa=True)

    first, second = fake_sd.requests
    assert first["seed"] == 7
    assert second["prompt"] == 'a wooden sign that reads "OPEN"'
    assert second["sample_params"]["guidance"] == {"txt_cfg": 30.0}  # clamped
    assert second["seed"] != 7


async def test_the_check_leaves_the_users_settings_alone(
    studio: httpx.AsyncClient, fake_llama: FakeLlama, fake_sd: FakeSdCpp
) -> None:
    fake_llama.queue(
        FakeLlama.tool_call("generate_image", {"prompt": "a fox"}),
        _verdict(verdict="redo", issues=["noisy"], steps=40, new_seed=False),
        _verdict(verdict="pass"),
        FakeLlama.text("Done."),
    )
    cid = await _conversation(studio)
    await _send(studio, cid, content="draw a fox", qa=True, image_options={"steps": 5})

    check = fake_llama.requests[1]
    assert "leave them null: steps" in _system(check)
    first, second = fake_sd.requests
    assert first["sample_params"]["sample_steps"] == second["sample_params"]["sample_steps"] == 5
    assert second["seed"] != first["seed"]  # nothing else it may change: a new seed
    rejected = next(m for m in await _messages(studio, cid) if m["role"] == "tool")
    assert rejected["metadata"]["qa"]["changes"] == {"new_seed": True}


async def test_attempts_run_out_and_the_last_image_is_kept_flagged(
    strict: httpx.AsyncClient, fake_llama: FakeLlama, fake_sd: FakeSdCpp
) -> None:
    fake_llama.queue(
        FakeLlama.tool_call("generate_image", {"prompt": "a horse"}),
        _verdict(verdict="redo", issues=["five legs"], new_seed=True),
        _verdict(verdict="redo", issues=["five legs again"], new_seed=True),
        FakeLlama.text("It may still have an extra leg."),
    )
    cid = await _conversation(strict)
    await _send(strict, cid, content="draw a horse")  # the server's default: on

    assert len(fake_sd.requests) == 2
    kept = [m for m in await _messages(strict, cid) if m["role"] == "tool"][-1]
    qa = kept["metadata"]["qa"]
    assert (qa["status"], qa["attempt"], qa["issues"]) == ("flagged", 2, ["five legs again"])
    assert "changes" not in qa


async def test_an_unreadable_check_is_skipped(
    studio: httpx.AsyncClient, fake_llama: FakeLlama, fake_sd: FakeSdCpp
) -> None:
    fake_llama.queue(
        FakeLlama.tool_call("generate_image", {"prompt": "a fox"}),
        FakeLlama.text("Looks great to me!"),
        FakeLlama.text("Done."),
    )
    cid = await _conversation(studio)
    await _send(studio, cid, content="draw a fox", qa=True)

    result = next(m for m in await _messages(studio, cid) if m["role"] == "tool")
    assert result["metadata"]["qa"]["status"] == "skipped"
    assert "could not be read" in result["metadata"]["qa"]["detail"]
    assert len(fake_sd.requests) == 1


async def test_a_chat_model_without_vision_skips_the_check(
    studio: httpx.AsyncClient, fake_llama: FakeLlama
) -> None:
    fake_llama.vision = False
    fake_llama.queue(
        FakeLlama.tool_call("generate_image", {"prompt": "a fox"}),
        FakeLlama.text("Done."),
    )
    cid = await _conversation(studio)
    await _send(studio, cid, content="draw a fox", qa=True)

    assert len(fake_llama.requests) == 2  # no check request
    assert QA_GUIDANCE not in _system(fake_llama.requests[0])
    result = next(m for m in await _messages(studio, cid) if m["role"] == "tool")
    assert result["metadata"]["qa"] == {
        "status": "skipped",
        "attempt": 1,
        "detail": "llama can't see images",
    }


async def test_in_raw_mode_the_check_keeps_the_prompt(
    studio: httpx.AsyncClient, fake_llama: FakeLlama, fake_sd: FakeSdCpp
) -> None:
    fake_llama.queue(
        _verdict(verdict="redo", issues=["blurry"], prompt="a sharper fox", steps=20),
        _verdict(verdict="pass"),
    )
    cid = await _conversation(studio)
    await _send(studio, cid, content="fox, snow", mode="raw", qa=True)

    check, _ = fake_llama.requests
    assert "used exactly as written" in _system(check)
    first, second = fake_sd.requests
    assert first["prompt"] == second["prompt"] == "fox, snow"
    assert second["sample_params"]["sample_steps"] == 20
    roles = [m["role"] for m in await _messages(studio, cid)]
    assert roles == ["user", "assistant", "tool", "tool"]  # two attempts, no reply


async def test_retrying_a_raw_result_runs_no_llm(
    studio: httpx.AsyncClient, fake_llama: FakeLlama, fake_sd: FakeSdCpp
) -> None:
    cid = await _conversation(studio)
    await _send(studio, cid, content="a fox", mode="raw", qa=False)
    result = (await _messages(studio, cid))[-1]

    resp = await studio.post(f"/v1/conversations/{cid}/messages/{result['id']}/retry")
    assert resp.status_code == 200, resp.text
    assert parse_sse(resp.text)[-1][0] == "done"

    assert fake_llama.requests == []
    first, second = fake_sd.requests
    assert first["prompt"] == second["prompt"] == "a fox"
    messages = await _messages(studio, cid)
    assert [m["role"] for m in messages] == ["user", "assistant", "tool", "tool"]
    assert messages[2]["parent_id"] == messages[3]["parent_id"]


# --- the verdict parser -------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "verdict"),
    [
        ('{"verdict": "pass", "issues": []}', "pass"),
        ('```json\n{"verdict": "redo", "steps": "30"}\n```', "redo"),
        ('Sure! {"verdict": "redo", "issues": ["blur"]} Hope that helps.', "redo"),
        ('{"verdict": "maybe"}', None),
        ("Looks fine.", None),
    ],
)
def test_parse_finds_the_verdict(text: str, verdict: str | None) -> None:
    parsed = parse(text)
    assert (parsed.verdict if parsed else None) == verdict
