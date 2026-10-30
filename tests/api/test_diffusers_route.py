"""The Diffusers backend is a drop-in: route t2i/edit to it and the same tool flow works."""

import json

import httpx
import pytest

from imaginer.backends import registry as registry_module
from imaginer.backends.diffusers_inproc import DiffusersBackend
from imaginer.core.config import (
    BackendConfig,
    DiffusersBackendConfig,
    OrchestrationConfig,
    RoutingConfig,
)
from imaginer.main import create_app
from tests.backends.test_diffusers import diffusers_config
from tests.conftest import SettingsFactory, llama_config
from tests.fakes.fake_diffusers import FakeLoader
from tests.fakes.fake_llama import FakeLlama
from tests.helpers import parse_sse


async def test_generate_then_edit_on_diffusers(
    make_settings: SettingsFactory, fake_llama: FakeLlama, monkeypatch: pytest.MonkeyPatch
) -> None:
    loader = FakeLoader()

    def factory(name: str, config: BackendConfig) -> DiffusersBackend:
        assert isinstance(config, DiffusersBackendConfig)
        return DiffusersBackend(name, config, loader=loader)

    monkeypatch.setitem(registry_module.FACTORIES, "diffusers", factory)
    settings = make_settings(
        backends={"llama": llama_config(fake_llama.url), "diff": diffusers_config()},
        routing=RoutingConfig(chat="llama", t2i="diff", edit="diff"),
        orchestration=OrchestrationConfig(probe_ttl_s=0),
    )
    fake_llama.queue(
        FakeLlama.tool_call("generate_image", {"prompt": "a red barn", "transparent": True}),
        FakeLlama.text("Here is img-1."),
        FakeLlama.tool_call("edit_image", {"images": ["img-1"], "prompt": "make it blue"}),
        FakeLlama.text("Here is img-2."),
    )
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            backends = {b["name"]: b for b in (await client.get("/v1/backends")).json()}
            assert backends["diff"]["status"] == "ok"
            assert set(backends["diff"]["capabilities"]) == {"t2i", "edit", "multi_ref", "rgba"}

            cid = (await client.post("/v1/conversations", json={})).json()["id"]
            for text in ("draw a barn sticker", "now blue"):
                resp = await client.post(
                    f"/v1/conversations/{cid}/messages", json={"content": text}
                )
                assert parse_sse(resp.text)[-1][0] == "done"

            artifacts = (await client.get(f"/v1/conversations/{cid}/artifacts")).json()["items"]
            assert [(a["label"], a["source"], a["has_alpha"]) for a in artifacts] == [
                ("img-1", "generated", True),
                ("img-2", "edited", False),
            ]
            jobs = (await client.get(f"/v1/conversations/{cid}/jobs")).json()["items"]
            assert [(j["backend"], j["model"], j["task"]) for j in jobs] == [
                ("diff", "Qwen/Qwen-Image-2.1", "t2i"),
                ("diff", "Qwen/Qwen-Image-2.1", "edit"),
            ]
            assert jobs[0]["progress"]["step"] == jobs[0]["progress"]["total_steps"] == 4
            messages = (await client.get(f"/v1/conversations/{cid}/messages")).json()["items"]
            results = [json.loads(m["content_text"]) for m in messages if m["role"] == "tool"]
            assert [r["status"] for r in results] == ["succeeded", "succeeded"]
    assert loader.loads == 1  # loaded once, reused for the edit
    assert len(loader.pipeline.calls[-1]["image"]) == 1
