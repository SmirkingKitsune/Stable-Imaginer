"""Agents and the Hub: a new agent from a Hub reference shows its uploader's avatar, and its
model and vision projector download into it."""

import asyncio
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from fastapi import FastAPI

from imaginer.core.config import load_settings
from imaginer.hub import downloads
from imaginer.main import create_app
from tests.fakes.fake_hub import FakeHub, FakeRepo, gguf
from tests.helpers import image_bytes
from tests.hub.conftest import LLM_REPO

_FAST = {"retry": {"attempts": 1}, "timeouts": {"connect": 0.2, "read": 0.5}}


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(downloads, "RETRY_BACKOFF_S", 0.0)


@pytest.fixture
async def client(tmp_path: Path, hub: FakeHub, hub_url: str) -> AsyncIterator[httpx.AsyncClient]:
    hub.avatars["acme"] = image_bytes(64, 48, color=(20, 90, 160))
    path = tmp_path / "config" / "box.yaml"
    path.parent.mkdir()
    path.write_text(
        yaml.safe_dump(
            {
                "data_root": str(tmp_path / "data"),
                "storage": {"sweep_interval_s": 0},
                "logging": {"level": "WARNING"},
                "hub": {"endpoint": hub_url},
                "backends": {
                    "llama": {
                        "type": "llamacpp",
                        "base_url": "http://127.0.0.1:1",
                        "launch": {},
                        **_FAST,
                    }
                },
                "routing": {"chat": "llama"},
            }
        ),
        encoding="utf-8",
    )
    app: FastAPI = create_app(load_settings(str(path)))
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
            yield http


async def _until(client: httpx.AsyncClient, agent_id: str, ok: Any) -> dict[str, Any]:
    deadline = time.monotonic() + 15
    while True:
        detail: dict[str, Any] = (await client.get(f"/v1/agents/{agent_id}")).json()
        if ok(detail):
            return detail
        assert time.monotonic() < deadline, detail
        await asyncio.sleep(0.05)


async def test_an_agent_from_the_hub(client: httpx.AsyncClient) -> None:
    resp = await client.post("/v1/agents", json={"ref": f"{LLM_REPO}:Q4_K_M", "engine": "llama"})
    assert resp.status_code == 201, resp.text
    agent = resp.json()
    assert agent["name"] == "Chat-8B"
    assert agent["missing"] == ["model"]

    # The uploader's avatar, fetched in the background, is its portrait until it has another.
    detail = await _until(client, agent["id"], lambda d: d["portrait"]["key"] == "hub")
    assert (detail["portrait"]["origin"], detail["portrait"]["hub_owner"]) == ("HUB · ACME", "acme")
    portrait = await client.get(detail["portrait"]["url"])
    assert portrait.status_code == 200 and portrait.headers["content-type"] == "image/webp"

    # Its files download into it.
    for key, variant in (("model", "Chat-8B-Q4_K_M.gguf"), ("mmproj", "mmproj-Chat-8B-F16.gguf")):
        job = await client.post(
            "/v1/hub/downloads",
            json={"repo": LLM_REPO, "variant": variant, "agent": agent["id"], "key": key},
        )
        assert job.status_code == 202, job.text
        assert job.json()["assign"] == [f"agent:{agent['id']}.{key}"]

    def downloaded(d: dict[str, Any]) -> bool:
        return all(f["path"] for f in d["files"] if f["key"] != "chat_template")

    detail = await _until(client, agent["id"], downloaded)
    assert detail["ready"] and detail["vision"]
    model = next(f for f in detail["files"] if f["key"] == "model")
    assert model["path"].endswith("Chat-8B-Q4_K_M.gguf") and model["exists"]
    assert model["source"]["repo"] == LLM_REPO

    # Refreshing from the Hub fetches the avatar again; it stays in use.
    again = await client.put(f"/v1/agents/{agent['id']}/portrait", json={"refresh": True})
    assert again.json()["portrait"]["key"] == "hub"


TOOL_TEMPLATE = "{%- if tools %}{% for tool in tools %}{{ tool | tojson }}{% endfor %}{% endif %}"
PLAIN_TEMPLATE = "{% for m in messages %}{{ m['content'] }}{% endfor %}"


def _vision_repos(hub: FakeHub) -> None:
    """Three more GGUF chat repos: one sees images but takes no tools, one is text only."""
    hub.repos["acme/Look-4B-GGUF"] = FakeRepo(
        {"Look-4B-Q4_K_M.gguf": gguf(3000, 4), "mmproj-Look-4B-F16.gguf": gguf(800, 5)},
        pipeline_tag="image-text-to-text",
        tags=["gguf"],
        downloads=5000,
        chat_template=PLAIN_TEMPLATE,
        architecture="gemma3",
        params=4_300_000_000,
    )
    hub.repos["acme/Text-3B-GGUF"] = FakeRepo(
        {"Text-3B-Q4_K_M.gguf": gguf(2000, 6)},
        pipeline_tag="text-generation",
        tags=["gguf"],
        downloads=9000,
        chat_template=TOOL_TEMPLATE,
    )
    # The Hub read its projector, not its model: nothing is known of its template.
    hub.repos["acme/Odd-2B-GGUF"] = FakeRepo(
        {"Odd-2B-Q4_K_M.gguf": gguf(2000, 7), "Odd-2B-mmproj-f16.gguf": gguf(700, 8)},
        tags=["gguf"],
        downloads=7000,
        architecture="clip",
        params=420_000_000,
    )
    hub.repos[LLM_REPO].chat_template = TOOL_TEMPLATE
    hub.repos[LLM_REPO].architecture = "qwen3vl"
    hub.repos[LLM_REPO].params = 8_190_735_360


async def test_the_hub_search_for_agents_lists_only_models_that_see_images(
    client: httpx.AsyncClient, hub: FakeHub
) -> None:
    _vision_repos(hub)
    found = (await client.get("/v1/hub/search", params={"vision": "true"})).json()
    # The text-only repo is left out; the one that calls tools comes first, though it has
    # the fewest downloads, then the others in the Hub's order.
    assert [(r["id"], r["vision"], r["tools"]) for r in found] == [
        (LLM_REPO, True, True),
        ("acme/Odd-2B-GGUF", True, None),
        ("acme/Look-4B-GGUF", True, False),
    ]
    assert (found[0]["params"], found[0]["architecture"]) == ("8.2B", "qwen3vl")
    assert (found[1]["params"], found[1]["architecture"]) == (None, None)
    # Without vision, the search is as it was.
    plain = (await client.get("/v1/hub/search", params={"format": "gguf"})).json()
    assert "acme/Text-3B-GGUF" in [r["id"] for r in plain] and plain[0]["vision"] is None


async def test_detect_checks_a_hub_repo_and_create_refuses_one_that_cant_see(
    client: httpx.AsyncClient, hub: FakeHub
) -> None:
    _vision_repos(hub)
    found = (await client.post("/v1/agents/detect", json={"ref": f"{LLM_REPO}:Q8_0"})).json()
    assert (found["kind"], found["name"], found["repo"], found["owner"]) == (
        "ref",
        "Chat-8B",
        LLM_REPO,
        "acme",
    )
    # The asked-for quant first, then the preferred one; the F16 projector.
    assert [m["quant"] for m in found["models"]] == ["Q8_0", "Q4_K_M"]
    assert [p["id"] for p in found["projectors"]] == ["mmproj-Chat-8B-F16.gguf"]
    assert found["fit"]["allowed"] and found["fit"]["tools"] is True
    assert found["fit"]["context_length"] == 32768

    look = (await client.post("/v1/agents/detect", json={"ref": "acme/Look-4B-GGUF"})).json()
    assert look["fit"]["allowed"] and look["fit"]["tools"] is False

    text = (await client.post("/v1/agents/detect", json={"ref": "acme/Text-3B-GGUF"})).json()
    assert not text["fit"]["allowed"] and text["fit"]["vision"] is False
    assert "vision projector" in text["fit"]["problems"][0]
    refused = await client.post("/v1/agents", json={"ref": "acme/Text-3B-GGUF:Q4_K_M"})
    assert refused.status_code == 422 and "vision projector" in refused.json()["detail"]

    # One that can see is made, and knows it needs its projector; the Hub's word on tools
    # stands until its model is here.
    made = (await client.post("/v1/agents", json={"ref": "acme/Look-4B-GGUF:Q4_K_M"})).json()
    assert made["missing"] == ["model"]
    assert next(f for f in made["files"] if f["key"] == "mmproj")["required"]
    assert made["template_tools"] is False and made["tools"] is False
