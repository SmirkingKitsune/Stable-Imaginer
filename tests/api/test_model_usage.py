"""An image made on an engine that runs a model card: made with the card's defaults, and
recorded with the card, which then counts it and shows it."""

from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from imaginer.core.config import Settings
from imaginer.main import create_app
from tests.fakes.fake_sdcpp import FakeSdCpp
from tests.helpers import parse_sse
from tests.image_models import headers


@pytest.fixture
async def studio(studio_settings: Settings) -> AsyncIterator[httpx.AsyncClient]:
    app: FastAPI = create_app(studio_settings)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


async def test_an_image_is_made_with_and_recorded_against_the_card(
    studio: httpx.AsyncClient, fake_sd: FakeSdCpp, tmp_path: Path
) -> None:
    model = headers.safetensors(tmp_path / "flux1-schnell.safetensors", headers.FLUX_SCHNELL)
    card = (await studio.post("/v1/models", json={"path": str(model)})).json()
    await studio.patch(f"/v1/models/{card['id']}", json={"defaults": {"width": 256, "height": 256}})
    # The fake engine isn't launched here: the card only says what it runs.
    used = await studio.post(f"/v1/models/{card['id']}/use", json={"engine": "sd"})
    assert used.status_code == 200, used.text
    conv = (await studio.post("/v1/conversations", json={"title": "fox"})).json()
    resp = await studio.post(
        f"/v1/conversations/{conv['id']}/messages",
        json={"content": "a red fox in snow", "mode": "raw", "qa": False},
    )
    assert resp.status_code == 200
    assert [e for e in parse_sse(resp.text) if e[0] == "error"] == []
    body = fake_sd.requests[-1]
    assert (body["width"], body["height"]) == (256, 256)
    assert body["sample_params"]["sample_steps"] == 4  # FLUX.1 schnell's recipe
    assert body["sample_params"]["guidance"] == {"txt_cfg": 1}
    jobs = (await studio.get(f"/v1/conversations/{conv['id']}/jobs")).json()
    job = (jobs["items"] if isinstance(jobs, dict) else jobs)[0]
    assert job["status"] == "succeeded" and job["params"]["model_card"] == card["id"]
    detail = (await studio.get(f"/v1/models/{card['id']}")).json()
    assert detail["uses"] == 1 and len(detail["recent_images"]) == 1
