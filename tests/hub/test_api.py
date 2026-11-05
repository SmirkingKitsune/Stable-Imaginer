"""/v1/hub end to end: the app against a fake Hub over real sockets. Downloads land in the
models folder and set the engine setting they were started for, saved beside the profile."""

import asyncio
import json
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from fastapi import FastAPI

from imaginer.core.config import load_settings, local_overrides_path
from imaginer.hub import downloads
from imaginer.main import create_app
from tests.fakes.fake_hub import TOKEN, FakeHub, gguf
from tests.fakes.live import LiveServer
from tests.hub.conftest import GATED_REPO, IMAGE_REPO, LLM_REPO, PIPELINE_REPO

_FAST = {"retry": {"attempts": 1}, "timeouts": {"connect": 0.2, "read": 0.5}}


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(downloads, "RETRY_BACKOFF_S", 0.0)


@pytest.fixture
def profile(tmp_path: Path, hub_url: str) -> Path:
    path = tmp_path / "config" / "box.yaml"
    path.parent.mkdir()
    backends = {
        "llama": {"type": "llamacpp", "base_url": "http://127.0.0.1:1", "launch": {}, **_FAST},
        "sd": {"type": "sdcpp", "base_url": "http://127.0.0.1:2", "launch": {}, **_FAST},
        "remote": {"type": "llamacpp", "base_url": "http://127.0.0.1:3", **_FAST},
        "diff": {"type": "diffusers", "enabled": False},
    }
    path.write_text(
        yaml.safe_dump(
            {
                "data_root": str(tmp_path / "data"),
                "storage": {"sweep_interval_s": 0},
                "logging": {"level": "WARNING"},
                "hub": {"endpoint": hub_url},
                "backends": backends,
                "routing": {"chat": "llama", "t2i": "sd", "edit": "sd"},
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def models(tmp_path: Path) -> Path:
    return tmp_path / "data" / "models"


@pytest.fixture
async def client(profile: Path) -> AsyncIterator[httpx.AsyncClient]:
    app: FastAPI = create_app(load_settings(str(profile)))
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
            yield http


def _log_in(hf_home: Path, token: str) -> None:
    """What `hf auth login` leaves behind."""
    hf_home.mkdir()
    (hf_home / "token").write_text(f"{token}\n", encoding="utf-8")


async def _finished(client: httpx.AsyncClient, job: dict[str, Any]) -> dict[str, Any]:
    deadline = time.monotonic() + 15
    while job["state"] in {"queued", "running"}:
        assert time.monotonic() < deadline, job
        await asyncio.sleep(0.05)
        jobs = (await client.get("/v1/hub/downloads")).json()
        job = next(j for j in jobs if j["id"] == job["id"])
    return job


async def _download(client: httpx.AsyncClient, **body: Any) -> dict[str, Any]:
    resp = await client.post("/v1/hub/downloads", json=body)
    assert resp.status_code == 202, resp.text
    return await _finished(client, resp.json())


async def test_status_says_whether_a_token_is_set_but_never_shows_it(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch, isolated_hf: Path, models: Path
) -> None:
    status = (await client.get("/v1/hub")).json()
    assert (status["token"], status["token_source"]) == (False, None)
    assert status["models_dir"] == str(models)
    assert status["hf_cache"] == str(isolated_hf / "hub")
    _log_in(isolated_hf, "hf_from_login")
    status = (await client.get("/v1/hub")).json()
    assert (status["token"], status["token_source"]) == (True, "hf auth login")
    monkeypatch.setenv("HF_TOKEN", "hf_secret_value")
    resp = await client.get("/v1/hub")
    assert (resp.json()["token"], resp.json()["token_source"]) == (True, "$HF_TOKEN")
    assert "hf_secret_value" not in resp.text and "hf_from_login" not in resp.text


async def test_search_filters_by_format_task_and_base_model(client: httpx.AsyncClient) -> None:
    async def ids(**params: str) -> list[str]:
        return [r["id"] for r in (await client.get("/v1/hub/search", params=params)).json()]

    assert set(await ids(format="gguf")) == {LLM_REPO, IMAGE_REPO, GATED_REPO}
    assert set(await ids(task="text-to-image")) == {IMAGE_REPO, PIPELINE_REPO}
    assert await ids(base_model="acme/Chat-8B", relation="quantized") == [LLM_REPO]
    assert await ids(q="image-2", format="diffusers") == [PIPELINE_REPO]
    found = (await client.get("/v1/hub/search", params={"q": "gated"})).json()
    assert (found[0]["gated"], found[0]["formats"]) == (True, ["gguf"])


async def test_a_repo_is_matched_to_an_engines_settings(client: httpx.AsyncClient) -> None:
    resp = await client.get("/v1/hub/repo", params={"ref": f"{LLM_REPO}:Q8_0", "engine": "llama"})
    assert resp.status_code == 200, resp.text
    repo = resp.json()
    assert (repo["kind"], repo["quant"], repo["base_models"]) == ("gguf", "Q8_0", ["acme/Chat-8B"])
    roles = {r["key"]: r for r in repo["roles"]}
    assert roles["model"]["flag"] == "-m"
    assert roles["model"]["candidates"] == ["Chat-8B-Q4_K_M.gguf", "Chat-8B-Q8_0.gguf"]
    assert roles["model"]["suggested"] == "Chat-8B-Q8_0.gguf"  # the quant asked for
    assert roles["mmproj"]["suggested"] == "mmproj-Chat-8B-F16.gguf"
    assert all(v["local"] is None for v in repo["variants"])


async def test_paths_leaving_the_repo_are_dropped_from_its_listing(
    client: httpx.AsyncClient, hub: FakeHub
) -> None:
    hub.repos[LLM_REPO].files["../../escape-Q4_K_M.gguf"] = gguf(100)
    repo = (await client.get("/v1/hub/repo", params={"ref": LLM_REPO, "engine": "llama"})).json()
    assert not any("escape" in v["id"] for v in repo["variants"])


async def test_a_download_sets_the_engine_setting_and_saves_it(
    client: httpx.AsyncClient, profile: Path, models: Path
) -> None:
    job = await _download(
        client, repo=LLM_REPO, variant="Chat-8B-Q4_K_M.gguf", engine="llama", key="model"
    )
    target = models / "acme" / "Chat-8B-GGUF" / "Chat-8B-Q4_K_M.gguf"
    assert (job["state"], job["target"], job["assign"]) == (
        "succeeded",
        str(target),
        ["llama.model"],
    )
    llama = (await client.get("/v1/engines/llama")).json()
    model = next(f for f in llama["files"] if f["key"] == "model")
    assert (model["path"], model["format"], model["problem"]) == (str(target), "gguf", None)
    saved = yaml.safe_load(local_overrides_path(profile).read_text(encoding="utf-8"))
    assert saved["backends"]["llama"]["launch"]["model"] == str(target)
    repo = (await client.get("/v1/hub/repo", params={"ref": LLM_REPO, "engine": "llama"})).json()
    q4 = next(v for v in repo["variants"] if v["id"] == "Chat-8B-Q4_K_M.gguf")
    assert q4["local"] == {"source": "models_dir", "path": str(target), "in_use": ["llama.model"]}


async def test_sd_server_takes_a_diffusers_component_through_its_index(
    client: httpx.AsyncClient, models: Path
) -> None:
    repo = (await client.get("/v1/hub/repo", params={"ref": PIPELINE_REPO, "engine": "sd"})).json()
    roles = {r["key"]: r for r in repo["roles"]}
    index = "transformer/diffusion_pytorch_model.safetensors.index.json"
    assert roles["diffusion_model"]["suggested"] == index
    assert roles["vae"]["suggested"] == "vae/diffusion_pytorch_model.safetensors"
    variant = next(v for v in repo["variants"] if v["id"] == index)
    assert (variant["name"], variant["shards"], len(variant["files"])) == ("transformer", 2, 3)
    job = await _download(
        client, repo=PIPELINE_REPO, variant=index, engine="sd", key="diffusion_model"
    )
    assert job["state"] == "succeeded", job
    sd = (await client.get("/v1/engines/sd")).json()
    diffusion = next(f for f in sd["files"] if f["key"] == "diffusion_model")
    assert diffusion["path"] == str(models / "acme" / "Image-2" / Path(index))
    assert (diffusion["format"], diffusion["size_bytes"], diffusion["problem"]) == (
        "safetensors",
        4000,  # both shards, not the index
        None,
    )


async def test_a_pipeline_download_becomes_a_diffusers_model_id(
    client: httpx.AsyncClient, models: Path
) -> None:
    repo = (
        await client.get("/v1/hub/repo", params={"ref": PIPELINE_REPO, "engine": "diff"})
    ).json()
    assert repo["pipeline"]["compatible"] is True
    assert repo["roles"] == [
        {"key": "model_id", "flag": None, "what": "Diffusers pipeline",
         "current": "Qwen/Qwen-Image-2.1", "candidates": [], "suggested": None}
    ]  # fmt: skip
    job = await _download(client, repo=PIPELINE_REPO, engine="diff", key="model_id")
    folder = models / "acme" / "Image-2"
    assert (job["state"], job["target"]) == ("succeeded", str(folder))
    assert (await client.get("/v1/engines/diff")).json()["model_id"] == str(folder)
    assert (
        json.loads((folder / "model_index.json").read_text())["_class_name"]
        == "QwenImage21Pipeline"
    )
    assert (folder / "transformer" / "diffusion_pytorch_model-00002-of-00002.safetensors").is_file()
    assert not (folder / "assets").exists()


async def test_gated_repos_need_the_hub_token(
    client: httpx.AsyncClient, hub: FakeHub, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = {"repo": GATED_REPO, "variant": "g-Q4_K_M.gguf", "engine": "llama", "key": "model"}
    job = await _download(client, **body)
    assert job["state"] == "failed"
    assert "gated" in job["error"] and "HF_TOKEN" in job["error"]
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    assert (await _download(client, **body))["state"] == "succeeded"
    assert hub.requests[-1]["auth"] == f"Bearer {TOKEN}"


def _get(variant: str, engine: str | None = None, key: str | None = None) -> dict[str, Any]:
    body = {"repo": LLM_REPO, "variant": variant}
    return body | {"engine": engine, "key": key} if engine else body


@pytest.mark.parametrize(
    ("method", "path", "body", "status", "fragment"),
    [
        ("GET", "/v1/hub/repo?ref=acme/missing", None, 404, "not found"),
        ("GET", "/v1/hub/repo?ref=https://huggingface.co/datasets/a/b", None, 422, "model repos"),
        ("GET", "/v1/hub/repo?ref=acme/x&engine=nobody", None, 404, "no backend"),
        ("POST", "/v1/hub/downloads", _get("mmproj-Chat-8B-F16.gguf", "llama", "model"),
         422, "can't be loaded as -m"),
        ("POST", "/v1/hub/downloads", _get("Chat-8B-Q4_K_M.gguf", "remote", "model"),
         422, "set up launch"),
        ("POST", "/v1/hub/downloads", _get("Chat-8B-Q4_K_M.gguf", "sd", "vae"),
         422, "can't be loaded as --vae"),
        ("POST", "/v1/hub/downloads", {"repo": LLM_REPO}, 422, "not a Diffusers pipeline"),
        ("POST", "/v1/hub/downloads", _get("nope.gguf"), 404, "no model file"),
    ],
)  # fmt: skip
async def test_requests_that_cant_work_say_why(
    client: httpx.AsyncClient,
    method: str,
    path: str,
    body: dict[str, Any] | None,
    status: int,
    fragment: str,
) -> None:
    resp = await client.request(method, path, json=body)
    assert resp.status_code == status, resp.text
    assert fragment in resp.json()["detail"]


async def test_the_models_folder_can_be_changed_and_is_saved(
    client: httpx.AsyncClient, profile: Path, tmp_path: Path
) -> None:
    folder = tmp_path / "elsewhere"
    resp = await client.patch("/v1/hub", json={"models_dir": str(folder)})
    assert resp.status_code == 200, resp.text
    assert resp.json()["models_dir"] == str(folder)
    saved = yaml.safe_load(local_overrides_path(profile).read_text(encoding="utf-8"))
    assert saved["hub"] == {"models_dir": str(folder)}
    job = await _download(client, repo=IMAGE_REPO, variant="image-2-Q4_K.gguf")
    assert job["target"] == str(folder / "acme" / "Image-2-GGUF" / "image-2-Q4_K.gguf")
    resp = await client.patch("/v1/hub", json={"models_dir": "relative/models"})
    assert resp.status_code == 422 and "absolute" in resp.json()["detail"]


async def test_events_stream_download_progress(profile: Path) -> None:
    with LiveServer(create_app(load_settings(str(profile))), lifespan="on") as live:
        async with (
            httpx.AsyncClient(base_url=live.url, timeout=10) as http,
            http.stream("GET", "/v1/hub/events") as stream,
        ):
            body = {"repo": IMAGE_REPO, "variant": "image-2-Q4_K.gguf"}
            started = (await http.post("/v1/hub/downloads", json=body)).json()
            states = []
            async for line in stream.aiter_lines():
                if line.startswith("data:"):
                    event = json.loads(line.removeprefix("data:"))
                    assert event["id"] == started["id"]
                    states.append(event["state"])
                    if event["state"] not in {"queued", "running"}:
                        break
    assert states[-1] == "succeeded"
    assert "running" in states
