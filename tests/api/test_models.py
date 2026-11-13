"""The model library (/v1/models): cards made from model files here, their components and
settings, running one on an sd-server engine, and the card an engine set up by hand gets."""

import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from fastapi import FastAPI

from imaginer.backends.registry import Lease
from imaginer.container import Container
from imaginer.core.config import load_settings, local_overrides_path
from imaginer.engines.launch import command
from imaginer.main import create_app
from tests.image_models import headers

_FAST = {"retry": {"attempts": 1}, "timeouts": {"connect": 0.3}}  # its engines are never up
PRESETS = {"draft": {"description": "fast", "steps": 4, "cfg_scale": 1.0}}


@pytest.fixture
def weights(tmp_path: Path) -> dict[str, Path]:
    m = tmp_path / "models"
    return {
        "qwen": headers.safetensors(m / "qwen_image_2.1_bf16.safetensors", ["img_in.weight"]),
        "qwen_vae": headers.safetensors(m / "qwen_image_2.1_vae_bf16.safetensors", ["x"]),
        "qwen_llm": headers.gguf(m / "Qwen3VL-8B-Instruct-Q4_K_M.gguf", ["x"]),
        "schnell": headers.safetensors(m / "my-fast-model.safetensors", headers.FLUX_SCHNELL),
        "ae": headers.safetensors(m / "ae.safetensors", ["decoder.conv_in.weight"]),
        "sdxl": headers.safetensors(m / "juggernautXL_v9.safetensors", headers.SDXL),
        "edit": headers.gguf(m / "qwen-image-edit-2511-Q4_K_M.gguf", ["x"], arch="qwen_image"),
        "edit_llm": headers.safetensors(m / "qwen_2.5_vl_7b.safetensors", ["x"]),
        "edit_llm_gguf": headers.gguf(m / "Qwen2.5-VL-7B-Instruct.Q4_K_M.gguf", ["x"]),
        "edit_vae": headers.safetensors(m / "qwen_image_vae.safetensors", ["x"]),
    }


@pytest.fixture
def profile(tmp_path: Path, weights: dict[str, Path]) -> Path:
    path = tmp_path / "config" / "box.yaml"
    path.parent.mkdir()
    backends = {
        "sd": {"type": "sdcpp", "base_url": "http://127.0.0.1:1", "launch": {}, **_FAST},
        # Set up by hand before the library: its files are in its launch block.
        "old": {
            "type": "sdcpp",
            "base_url": "http://127.0.0.1:2",
            "launch": {
                "executable": sys.executable,
                "diffusion_model": str(weights["qwen"]),
                "vae": str(weights["qwen_vae"]),
                "llm": str(weights["qwen_llm"]),
            },
            "defaults": {"width": 1024, "height": 1024, "steps": 9, "cfg_scale": 2.5},
            "presets": PRESETS,
            **_FAST,
        },
        "remote": {"type": "sdcpp", "base_url": "http://127.0.0.1:3", **_FAST},
    }
    path.write_text(
        yaml.safe_dump(
            {
                "data_root": str(tmp_path / "data"),
                "storage": {"sweep_interval_s": 0},
                "logging": {"level": "WARNING"},
                "backends": backends,
                "routing": {"t2i": "sd", "edit": "sd"},
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
async def app(profile: Path) -> AsyncIterator[FastAPI]:
    application = create_app(load_settings(str(profile)))
    async with application.router.lifespan_context(application):
        yield application


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        yield http


def _saved(profile: Path) -> dict[str, Any]:
    data = yaml.safe_load(local_overrides_path(profile).read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


async def _create(client: httpx.AsyncClient, **body: Any) -> dict[str, Any]:
    resp = await client.post("/v1/models", json=body)
    assert resp.status_code == 201, resp.text
    card: dict[str, Any] = resp.json()
    return card


async def test_an_engine_set_up_by_hand_gets_a_card(
    client: httpx.AsyncClient, profile: Path, weights: dict[str, Path]
) -> None:
    library = (await client.get("/v1/models")).json()
    (card,) = library["items"]
    assert (card["name"], card["family"], card["ready"]) == (
        "Qwen Image 2.1",
        "qwen-image-21",
        True,
    )
    assert [e["engine"] for e in card["engines"]] == ["old"]
    assert _saved(profile)["backends"]["old"] == {"model_card": card["id"]}
    detail = (await client.get(f"/v1/models/{card['id']}")).json()
    assert detail["presets"]["draft"] == PRESETS["draft"]  # its settings came along
    assert (detail["defaults"]["steps"], detail["defaults"]["cfg_scale"]) == (9, 2.5)
    engine = (await client.get("/v1/engines/old")).json()
    assert (engine["model_card"], engine["family"]) == (card["id"], "qwen-image-21")
    assert {f["key"]: f["path"] for f in engine["files"] if f["path"]} == {
        "diffusion_model": str(weights["qwen"]),
        "vae": str(weights["qwen_vae"]),
        "llm": str(weights["qwen_llm"]),
    }
    presets = (await client.get("/v1/backends")).json()
    old = next(b for b in presets if b["name"] == "old")
    assert [p["name"] for p in old["image_presets"]] == ["draft"]


async def test_a_card_from_a_file_detects_its_family_and_lists_what_it_lacks(
    client: httpx.AsyncClient, weights: dict[str, Path]
) -> None:
    card = await _create(client, path=str(weights["schnell"]))
    assert (card["family"], card["name"]) == ("flux1-schnell", "FLUX.1 schnell")
    assert card["missing"] == ["clip_l", "t5xxl", "vae or tae"] and not card["ready"]
    assert card["defaults"]["steps"] == 4 and card["defaults"]["width"] == 1024
    components = {c["slot"]: c for c in card["components"]}
    assert components["diffusion_model"]["path"] == str(weights["schnell"])
    assert components["clip_l"]["sources"][0] == (
        "comfyanonymous/flux_text_encoders/clip_l.safetensors"
    )
    assert card["detected"][0]["family"] == "flux1-schnell"
    checkpoint = await _create(client, path=str(weights["sdxl"]), name="Juggernaut")
    assert (checkpoint["family"], checkpoint["name"]) == ("sdxl-checkpoint", "Juggernaut")
    assert {c["slot"]: c["path"] for c in checkpoint["components"]}["model"] == str(weights["sdxl"])
    assert checkpoint["ready"] is True


async def test_editing_a_card_and_sharing_a_component(
    client: httpx.AsyncClient, weights: dict[str, Path], tmp_path: Path
) -> None:
    card = await _create(client, path=str(weights["schnell"]))
    t5 = headers.safetensors(tmp_path / "models" / "t5xxl_fp16.safetensors", ["x"])
    clip = headers.safetensors(tmp_path / "models" / "clip_l.safetensors", ["x"])
    edited = (
        await client.patch(
            f"/v1/models/{card['id']}",
            json={
                "components": {"clip_l": str(clip), "t5xxl": str(t5), "vae": str(weights["ae"])},
                "defaults": {"steps": 6, "guidance": 3.0},
                "presets": {"one": {"steps": 1, "cfg_scale": 1.0}},
                "launch": {"extra_args": ["--offload-to-cpu"]},
                "tags": ["fast", "fast", "flux"],
            },
        )
    ).json()
    assert edited["ready"] is True and edited["missing"] == []
    assert (edited["defaults"]["steps"], edited["defaults"]["guidance"]) == (6, 3.0)
    assert edited["presets"] == {"one": {"steps": 1, "cfg_scale": 1.0, "description": ""}}
    assert edited["tags"] == ["fast", "flux"] and edited["launch"]["extra_args"] == [
        "--offload-to-cpu"
    ]
    other = await _create(client, family="chroma", components={"vae": str(weights["ae"])})
    detail = (await client.get(f"/v1/models/{other['id']}")).json()
    vae = next(c for c in detail["components"] if c["slot"] == "vae")
    assert vae["shared_with"] == ["FLUX.1 schnell"]
    library = (await client.get("/v1/models")).json()
    shared = next(c for c in library["components"] if c["path"] == str(weights["ae"]))
    assert sorted(shared["cards"]) == sorted([card["id"], other["id"]])
    bad = await client.patch(f"/v1/models/{card['id']}", json={"defaults": {"steps": 0}})
    assert bad.status_code == 422
    assert (
        await client.patch(f"/v1/models/{card['id']}", json={"components": {"x": "y"}})
    ).status_code == 422


async def test_model_arguments_survive_save_and_launch(
    client: httpx.AsyncClient, app: FastAPI, weights: dict[str, Path]
) -> None:

    card = await _create(client, path=str(weights["sdxl"]))
    args = ["--model-dir", "C:\\My Models\\", "--name", "", "--separator", " "]
    response = await client.patch(f"/v1/models/{card['id']}", json={"launch": {"extra_args": args}})
    assert response.status_code == 200, response.text
    saved = (await client.get(f"/v1/models/{card['id']}")).json()
    assert saved["launch"]["extra_args"] == args
    response = await client.post(
        f"/v1/models/{card['id']}/use", json={"engine": "sd", "start": False}
    )
    assert response.status_code == 200, response.text
    container: Container = app.state.container
    argv = command(container.engines.effective("sd"), Path("sd-server"))
    assert argv[-len(args) :] == args


async def test_using_a_card_on_an_engine(
    client: httpx.AsyncClient, app: FastAPI, profile: Path, weights: dict[str, Path]
) -> None:
    card = await _create(client, path=str(weights["sdxl"]))
    resp = await client.post(f"/v1/models/{card['id']}/use", json={"engine": "sd", "start": False})
    assert resp.status_code == 200, resp.text
    assert resp.json()["engines"] == [
        {
            "engine": "sd",
            "role": "main",
            "launchable": True,
            "running": False,
            "loaded": False,
            "needs_restart": False,
        }
    ]
    assert _saved(profile)["backends"]["sd"] == {"model_card": card["id"]}
    engine = (await client.get("/v1/engines/sd")).json()
    assert engine["family"] == "sdxl-checkpoint"
    assert [f["key"] for f in engine["files"]] == ["model", "vae", "tae"]
    backends = {b["name"]: b for b in (await client.get("/v1/backends")).json()}
    assert backends["sd"]["image_defaults"]["size_multiple"] == 64
    # Presets set on the engine are the card's now, not the profile's.
    await client.patch("/v1/engines/sd", json={"presets": {"lightning": {"steps": 4}}})
    assert (await client.get(f"/v1/models/{card['id']}")).json()["presets"]["lightning"][
        "steps"
    ] == 4
    assert "presets" not in _saved(profile)["backends"]["sd"]
    # A card an engine runs can't be deleted; nor can the engine change models mid-image.
    assert (await client.delete(f"/v1/models/{card['id']}")).status_code == 409
    container: Container = app.state.container
    entry = container.registry._entries["sd"]
    assert entry.backend is not None
    lease = Lease(container.registry, entry.backend)  # type: ignore[type-var]
    try:
        busy = await client.post(f"/v1/models/{card['id']}/use", json={"engine": "sd"})
        assert busy.status_code == 409
    finally:
        lease.release()


async def test_a_card_describes_an_engine_started_elsewhere(
    client: httpx.AsyncClient, weights: dict[str, Path]
) -> None:
    card = await _create(client, path=str(weights["schnell"]))
    resp = await client.post(f"/v1/models/{card['id']}/use", json={"engine": "remote"})
    assert resp.status_code == 200, resp.text  # not ready, but nothing is started here
    backends = {b["name"]: b for b in (await client.get("/v1/backends")).json()}
    assert backends["remote"]["image_defaults"]["steps"] == 4


async def test_families_and_detection(client: httpx.AsyncClient, weights: dict[str, Path]) -> None:
    families = (await client.get("/v1/models/families")).json()
    ids = {f["id"] for f in families}
    assert {"flux1-dev", "qwen-image-edit-2511", "flux1-kontext"} <= ids
    assert "sdxl-controlnet" not in ids
    every = (await client.get("/v1/models/families", params={"hidden": True})).json()
    assert "sdxl-controlnet" in {f["id"] for f in every}
    flux = (await client.get("/v1/models/families/flux1-dev")).json()
    assert [s["id"] for s in flux["slots"] if s["required"]] == [
        "diffusion_model",
        "clip_l",
        "t5xxl",
    ]
    assert flux["required_any"] == [["vae", "tae"]]
    found = (await client.post("/v1/models/detect", json={"path": str(weights["sdxl"])})).json()
    assert (found["guesses"][0]["family"], found["slot"]) == ("sdxl-checkpoint", "model")
    missing = await client.post("/v1/models/detect", json={"path": str(weights["sdxl"]) + "x"})
    assert missing.status_code == 422
    assert (await client.get("/v1/models/families/nope")).status_code == 404


async def test_a_dropped_file_is_found_by_its_name_in_the_cards_folders(
    client: httpx.AsyncClient, weights: dict[str, Path]
) -> None:
    # The hand-made card's files are in models/: a file next to them is found by name and
    # size, as a browser reports a dropped file.
    target = weights["schnell"]
    resp = await client.post(
        "/v1/models/locate", json={"name": target.name.upper(), "size_bytes": target.stat().st_size}
    )
    assert resp.status_code == 200, resp.text
    found = resp.json()
    assert found["paths"] == [str(target)]
    assert str(target.parent) in found["searched"] or str(target.parent.parent) in found["searched"]
    # Another size: still offered, as the only file of that name.
    other = (
        await client.post("/v1/models/locate", json={"name": target.name, "size_bytes": 1})
    ).json()
    assert other["paths"] == [str(target)]
    nothing = (await client.post("/v1/models/locate", json={"name": "nowhere.gguf"})).json()
    assert nothing["paths"] == []


async def _edit_card(
    client: httpx.AsyncClient, weights: dict[str, Path], llm: str
) -> dict[str, Any]:
    return await _create(
        client,
        path=str(weights["edit"]),
        components={"llm": str(weights[llm]), "vae": str(weights["edit_vae"])},
    )


async def test_an_edit_model_is_detected_and_needs_its_vision(
    client: httpx.AsyncClient, weights: dict[str, Path]
) -> None:
    blind = await _edit_card(client, weights, "edit_llm_gguf")
    assert blind["family"] == "qwen-image-edit-2511"
    # A GGUF text encoder can't see the images without its vision projector.
    assert "llm_vision" in blind["missing"] and not blind["ready"]
    assert blind["capabilities"] == ["lora"]
    seeing = await _edit_card(client, weights, "edit_llm")
    assert seeing["ready"] and seeing["missing"] == []
    assert set(seeing["capabilities"]) == {"edit", "multi_ref", "lora"}  # no text-to-image


async def test_an_engine_switches_to_its_edit_card_for_edits(
    client: httpx.AsyncClient, app: FastAPI, profile: Path, weights: dict[str, Path]
) -> None:
    main = await _create(client, path=str(weights["sdxl"]))
    edit = await _edit_card(client, weights, "edit_llm")
    await client.post(f"/v1/models/{main['id']}/use", json={"engine": "sd", "start": False})
    resp = await client.post(f"/v1/models/{edit['id']}/use", json={"engine": "sd", "role": "edit"})
    assert resp.status_code == 200, resp.text
    assert [(u["engine"], u["role"]) for u in resp.json()["engines"]] == [("sd", "edit")]
    assert _saved(profile)["backends"]["sd"] == {"model_card": main["id"], "edit_card": edit["id"]}
    engine = (await client.get("/v1/engines/sd")).json()
    assert engine["model_card"] == main["id"] and engine["edit_card"] == edit["id"]
    assert engine["edit_card_name"] == edit["name"]
    # The engine makes images with its card and edits with the edit card's settings.
    backend = {b["name"]: b for b in (await client.get("/v1/backends")).json()}["sd"]
    assert backend["image_defaults"]["size_multiple"] == 64  # SDXL's
    assert backend["edit"]["card"] == edit["id"] and backend["edit"]["max_ref_images"] == 3
    assert backend["edit"]["mask"] is False  # SDXL takes no mask either
    assert backend["edit"]["mask_card"] is None
    container: Container = app.state.container
    entry = container.registry._entries["sd"]
    assert entry.backend is not None
    face = entry.backend.edit_face  # type: ignore[union-attr]
    assert face is not None and face.family == "qwen-image-edit-2511"
    assert face.model_card == edit["id"]
    view = container.engines.effective("sd").edit_view  # type: ignore[union-attr]
    assert view is not None
    argv = command(view, Path("sd-server"))  # what an edit restarts the engine with
    assert str(weights["edit"]) in argv and "qwen_image_zero_cond_t=true" in argv
    # A card used for edits can't be deleted either; nor be both cards.
    assert (await client.delete(f"/v1/models/{edit['id']}")).status_code == 409
    same = await client.post(f"/v1/models/{main['id']}/use", json={"engine": "sd", "role": "edit"})
    assert same.status_code == 422
    # A card that can't edit can't be the edit card.
    sdxl = await client.post(
        f"/v1/models/{main['id']}/use", json={"engine": "remote", "role": "edit"}
    )
    assert sdxl.status_code == 422
    # Running the edit card for everything ends its use for edits.
    await client.post(f"/v1/models/{edit['id']}/use", json={"engine": "sd", "start": False})
    assert _saved(profile)["backends"]["sd"] == {"model_card": edit["id"], "edit_card": None}
    # The engine's own Engines-screen edit: clearing it.
    cleared = await client.patch("/v1/engines/sd", json={"edit_card": ""})
    assert cleared.status_code == 200 and cleared.json()["edit_card"] is None


async def test_masked_edits_run_the_model_card_when_the_edit_card_takes_no_mask(
    client: httpx.AsyncClient, app: FastAPI, weights: dict[str, Path]
) -> None:
    (main,) = (await client.get("/v1/models")).json()["items"]  # "old" runs Qwen-Image-2.1
    edit = await _edit_card(client, weights, "edit_llm")
    resp = await client.post(f"/v1/models/{edit['id']}/use", json={"engine": "old", "role": "edit"})
    assert resp.status_code == 200, resp.text
    backend = {b["name"]: b for b in (await client.get("/v1/backends")).json()}["old"]
    assert backend["edit"]["card"] == edit["id"]
    # Qwen-Image-Edit-2511 takes no mask and Qwen-Image-2.1 does: masked edits run that.
    assert backend["edit"]["mask"] is True
    assert (backend["edit"]["mask_card"], backend["edit"]["mask_card_name"]) == (
        main["id"],
        main["name"],
    )
    container: Container = app.state.container
    hint = container.models.prompt_hint("old") or ""
    assert f"An edit_image call with a mask runs {main['name']} instead." in hint


async def test_an_edit_card_needs_an_engine_started_here(
    client: httpx.AsyncClient, weights: dict[str, Path]
) -> None:
    edit = await _edit_card(client, weights, "edit_llm")
    resp = await client.post(
        f"/v1/models/{edit['id']}/use", json={"engine": "remote", "role": "edit"}
    )
    assert resp.status_code == 422 and "isn't started here" in resp.json()["detail"]
