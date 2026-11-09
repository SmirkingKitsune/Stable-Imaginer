"""The LoRA library (GET/PATCH /v1/loras): one entry per LoRA file, what the user adds to
it, its compatibility with the engines' models, and usage from the jobs that used it."""

import hashlib
import json
import struct
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from imaginer.backends import registry as registry_module
from imaginer.backends.diffusers_inproc import DiffusersBackend
from imaginer.core.config import (
    BackendConfig,
    DiffusersBackendConfig,
    HubConfig,
    OrchestrationConfig,
    QAConfig,
    RoutingConfig,
)
from imaginer.loras import files
from imaginer.main import create_app
from tests.backends.test_diffusers import diffusers_config
from tests.conftest import SettingsFactory, llama_config
from tests.fakes.fake_diffusers import FakeLoader
from tests.fakes.fake_llama import FakeLlama
from tests.helpers import parse_sse

WIDTH = 32


def tensors_file(path: Path, tensors: dict[str, list[int]], metadata: dict[str, str]) -> Path:
    """A safetensors file whose header names ``tensors`` (no weights: only headers are read)."""
    header: dict[str, Any] = {"__metadata__": metadata}
    header |= {n: {"dtype": "F16", "shape": s, "data_offsets": [0, 0]} for n, s in tensors.items()}
    raw = json.dumps(header).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw)
    return path


def lora_tensors(width: int) -> dict[str, list[int]]:
    out = {}
    for module in ("to_q", "to_k"):
        out[f"transformer_blocks.0.attn.{module}.lora_A.weight"] = [4, width]
        out[f"transformer_blocks.0.attn.{module}.lora_B.weight"] = [64, 4]
    return out


@pytest.fixture
def loader() -> FakeLoader:
    return FakeLoader()


@pytest.fixture
async def library(
    tmp_path: Path,
    make_settings: SettingsFactory,
    fake_llama: FakeLlama,
    loader: FakeLoader,
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[httpx.AsyncClient]:
    # The model: a Diffusers folder whose transformer has the modules the LoRAs patch.
    model = tmp_path / "Qwen-Image-2.1"
    tensors_file(
        model / "transformer" / "diffusion_pytorch_model.safetensors",
        {f"transformer_blocks.0.attn.{m}.weight": [64, WIDTH] for m in ("to_q", "to_k")},
        {},
    )
    loras = tmp_path / "loras"
    tensors_file(
        loras / "styles" / "clay.safetensors",
        lora_tensors(WIDTH),
        {"trigger_words": "clay sculpture, claymation", "base_model": "Qwen/Qwen-Image-2.1"},
    )
    tensors_file(loras / "other" / "sdxl-ink.safetensors", lora_tensors(16), {})

    def factory(name: str, config: BackendConfig) -> DiffusersBackend:
        assert isinstance(config, DiffusersBackendConfig)
        return DiffusersBackend(name, config, loader=loader)

    monkeypatch.setitem(registry_module.FACTORIES, "diffusers", factory)
    presets = {"clay": {"loras": [{"path": "styles/clay.safetensors", "weight": 0.8}]}}
    settings = make_settings(
        backends={
            "llama": llama_config(fake_llama.url),
            "diff": diffusers_config(model_id=str(model), lora_dir=loras, presets=presets),
        },
        routing=RoutingConfig(chat="llama", t2i="diff", edit="diff"),
        orchestration=OrchestrationConfig(probe_ttl_s=0, qa=QAConfig(enabled=False)),
        hub=HubConfig(use_hf_cache=False),
    )
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def _items(client: httpx.AsyncClient) -> dict[str, dict[str, Any]]:
    resp = await client.get("/v1/loras")
    assert resp.status_code == 200, resp.text
    return {i["file_name"]: i for i in resp.json()["items"]}


async def test_the_library_lists_each_lora_with_what_its_file_says(
    library: httpx.AsyncClient, tmp_path: Path
) -> None:
    body = (await library.get("/v1/loras")).json()
    items = {i["file_name"]: i for i in body["items"]}
    clay = items["clay.safetensors"]
    assert clay["id"] == clay["sha256"] == _sha(tmp_path / "loras" / "styles" / "clay.safetensors")
    assert (clay["name"], clay["path"], clay["folder"], clay["local"]) == (
        "clay",
        "styles/clay.safetensors",
        "styles",
        True,
    )
    assert [t["text"] for t in clay["triggers"]] == ["clay sculpture", "claymation"]
    assert clay["engines"] == [
        {
            "engine": "diff",
            "path": "styles/clay.safetensors",
            "model": "qwen-image-2.1",
            "compat": "ok",
        }
    ]
    assert clay["compat"]["state"] == "ok"
    # Same module names, another width: made for another model.
    ink = items["sdxl-ink.safetensors"]
    assert ink["compat"]["state"] == "no"
    assert ink["compat"]["bad"] == "works with no model here"
    assert body["folders"] == ["other", "styles"]
    assert body["local_count"] == 2
    assert [m["name"] for m in body["models"]] == ["Qwen-Image-2.1"]

    detail = (await library.get(f"/v1/loras/{clay['id']}")).json()
    (row,) = detail["compat_models"]
    assert (row["detected"], row["detected_text"], row["call"]) == ("ok", "2/2 keys", "auto")
    assert (detail["format"], detail["rank"], detail["base"]) == ("PEFT keys", 4, "Qwen-Image-2.1")
    assert detail["triggers_from"] == "the file's metadata (trigger_words)"
    (preset,) = detail["presets"]
    assert (preset["name"], preset["included"], preset["weight"]) == ("clay", True, 0.8)
    assert (await library.get("/v1/loras/nope")).status_code == 404


async def test_the_users_words_name_and_calls_are_kept(
    library: httpx.AsyncClient, loader: FakeLoader, fake_llama: FakeLlama
) -> None:
    clay = (await _items(library))["clay.safetensors"]
    resp = await library.patch(
        f"/v1/loras/{clay['id']}",
        json={
            "name": "Clay Sculpture",
            "default_weight": 0.8,
            "triggers": [
                {"text": "clay sculpture", "on": True},
                {"text": "claymation", "on": False},
            ],
            "tags": ["Style", "3D", "style"],
            "pinned": True,
            "notes": "  Strong at 0.8. ",
        },
    )
    assert resp.status_code == 200, resp.text
    detail = resp.json()
    assert (detail["name"], detail["custom_name"], detail["default_weight"]) == (
        "Clay Sculpture",
        True,
        0.8,
    )
    assert detail["tags"] == ["Style", "3D"]
    assert (detail["pinned"], detail["notes"], detail["triggers_from"]) == (
        True,
        "Strong at 0.8.",
        None,
    )

    # An image with it gets the words that are on, and the job counts as a use.
    fake_llama.queue(
        FakeLlama.tool_call("generate_image", {"prompt": "a fox"}), FakeLlama.text("Done.")
    )
    cid = (await library.post("/v1/conversations", json={"title": "Clay foxes"})).json()["id"]
    resp = await library.post(
        f"/v1/conversations/{cid}/messages",
        json={"content": "a fox", "image_options": {"loras": [{"path": "clay", "weight": 0.7}]}},
    )
    assert parse_sse(resp.text)[-1][0] == "done"
    assert loader.pipeline.calls[-1]["prompt"] == "a fox, clay sculpture"
    detail = (await library.get(f"/v1/loras/{clay['id']}")).json()
    assert (detail["uses"], detail["conversations"]) == (1, 1)
    (used,) = detail["used_in"]
    assert (used["conversation_id"], used["title"], used["images"], used["weight"]) == (
        cid,
        "Clay foxes",
        1,
        0.7,
    )
    assert used["artifact"]["label"] == "img-1"
    assert [a["label"] for a in detail["recent_images"]] == ["img-1"]

    # The user's call wins over what was detected; auto goes back to it.
    key = detail["compat_models"][0]["key"]
    detail = (await library.patch(f"/v1/loras/{clay['id']}", json={"compat": {key: "no"}})).json()
    assert detail["compat_models"][0]["state"] == "no"
    assert detail["engines"][0]["compat"] == "no"
    detail = (await library.patch(f"/v1/loras/{clay['id']}", json={"compat": {key: "auto"}})).json()
    assert detail["engines"][0]["compat"] == "ok"

    # Back to the file's words.
    detail = (await library.patch(f"/v1/loras/{clay['id']}", json={"reset_triggers": True})).json()
    assert [t["on"] for t in detail["triggers"]] == [True, True]


async def test_the_llm_is_told_which_loras_the_user_picked(
    library: httpx.AsyncClient, fake_llama: FakeLlama
) -> None:
    clay = (await _items(library))["clay.safetensors"]
    resp = await library.patch(
        f"/v1/loras/{clay['id']}",
        json={"name": "Clay Sculpture", "tags": ["style"], "notes": "Soft light.\nBest at 0.8."},
    )
    assert resp.status_code == 200, resp.text
    fake_llama.queue(FakeLlama.text("What should they show?"))
    cid = (await library.post("/v1/conversations", json={})).json()["id"]
    resp = await library.post(
        f"/v1/conversations/{cid}/messages",
        json={
            "content": "make @lora:clay:0.8 portraits",
            "image_options": {"loras": [{"path": "other/sdxl-ink.safetensors", "weight": 0.5}]},
        },
    )
    assert parse_sse(resp.text)[-1][0] == "done"
    system, user = fake_llama.requests[-1]["messages"][:2]
    assert "Some of the user's messages pick LoRAs" in system["content"]
    assert user["content"] == (
        "make Clay Sculpture portraits\n\n"
        "[LoRAs the user picked for this message's images:\n"
        "- sdxl-ink, weight 0.5\n"
        '- Clay Sculpture, weight 0.8, trigger words "clay sculpture", "claymation", '
        'tags style, notes "Soft light. Best at 0.8."]'
    )


async def test_a_cover_comes_from_a_conversations_image(
    library: httpx.AsyncClient, fake_llama: FakeLlama
) -> None:
    fake_llama.queue(
        FakeLlama.tool_call("generate_image", {"prompt": "a fox"}), FakeLlama.text("Done.")
    )
    cid = (await library.post("/v1/conversations", json={"title": "Foxes"})).json()["id"]
    resp = await library.post(f"/v1/conversations/{cid}/messages", json={"content": "a fox"})
    assert parse_sse(resp.text)[-1][0] == "done"
    (art,) = (await library.get(f"/v1/conversations/{cid}/artifacts")).json()["items"]
    clay = (await _items(library))["clay.safetensors"]
    resp = await library.put(f"/v1/loras/{clay['id']}/cover", json={"artifact_id": art["id"]})
    assert resp.status_code == 200, resp.text
    cover = resp.json()["cover"]
    assert cover["source"] == "img-1 · Foxes"
    image = await library.get(cover["url"])
    assert image.status_code == 200
    assert image.headers["content-type"] == "image/webp"
    assert (await library.delete(f"/v1/loras/{clay['id']}/cover")).json()["cover"] is None
    assert (await library.get(cover["url"])).status_code == 404


async def test_moving_a_lora_points_presets_at_its_new_path(
    library: httpx.AsyncClient, tmp_path: Path
) -> None:
    clay = (await _items(library))["clay.safetensors"]
    resp = await library.post(f"/v1/loras/{clay['id']}/move", json={"folder": "3d/clay"})
    assert resp.status_code == 200, resp.text
    moved = resp.json()
    assert (moved["id"], moved["path"]) == (clay["id"], "3d/clay/clay.safetensors")
    assert (tmp_path / "loras" / "3d" / "clay" / "clay.safetensors").is_file()
    assert not (tmp_path / "loras" / "styles").exists()  # left empty: removed
    (preset,) = moved["presets"]
    assert (preset["included"], preset["weight"]) == (True, 0.8)
    backends = {b["name"]: b for b in (await library.get("/v1/backends")).json()}
    (info,) = backends["diff"]["image_presets"]
    assert info["loras"] == [{"path": "3d/clay/clay.safetensors", "weight": 0.8}]
    bad = await library.post(f"/v1/loras/{clay['id']}/move", json={"folder": "../out"})
    assert bad.status_code == 422


async def test_presets_take_and_drop_a_lora(library: httpx.AsyncClient) -> None:
    ink = (await _items(library))["sdxl-ink.safetensors"]
    url = f"/v1/loras/{ink['id']}/presets"
    detail = (
        await library.post(url, json={"engine": "diff", "preset": "clay", "weight": 0.5})
    ).json()
    assert detail["presets"][0]["included"] is True
    backends = {b["name"]: b for b in (await library.get("/v1/backends")).json()}
    assert backends["diff"]["image_presets"][0]["loras"] == [
        {"path": "styles/clay.safetensors", "weight": 0.8},
        {"path": "other/sdxl-ink.safetensors", "weight": 0.5},
    ]
    detail = (
        await library.post(url, json={"engine": "diff", "preset": "clay", "weight": None})
    ).json()
    assert detail["presets"][0]["included"] is False
    resp = await library.post(
        url, json={"engine": "diff", "preset": "ink", "weight": 1.0, "create": True}
    )
    assert {p["name"]: p["included"] for p in resp.json()["presets"]} == {
        "clay": False,
        "ink": True,
    }
    again = await library.post(
        url, json={"engine": "diff", "preset": "ink", "weight": 1.0, "create": True}
    )
    assert again.status_code == 409


async def test_deleting_a_lora_takes_it_out_of_presets(
    library: httpx.AsyncClient, tmp_path: Path
) -> None:
    clay = (await _items(library))["clay.safetensors"]
    assert (await library.delete(f"/v1/loras/{clay['id']}")).status_code == 204
    assert not (tmp_path / "loras" / "styles" / "clay.safetensors").exists()
    assert "clay.safetensors" not in await _items(library)
    backends = {b["name"]: b for b in (await library.get("/v1/backends")).json()}
    assert backends["diff"]["image_presets"][0]["loras"] == []


def test_lineage_takes_later_versions() -> None:
    assert files.lineage_matches("Qwen-Image 2.x", "Qwen-Image-2.1-Turbo")
    assert not files.lineage_matches("Qwen-Image 2.x", "Qwen-Image 1.0")
    assert files.model_name("D:/m/qwen-image-2.1-turbo-Q6_K_XL.gguf") == "qwen-image-2.1-turbo"
    assert files.model_quant("qwen-image-2.1-turbo-Q6_K_XL.gguf") == "Q6_K_XL"
    assert files.model_name("Qwen/Qwen-Image-2.1") == "Qwen-Image-2.1"
