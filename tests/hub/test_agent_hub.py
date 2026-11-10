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
from tests.fakes.fake_hub import FakeHub
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
