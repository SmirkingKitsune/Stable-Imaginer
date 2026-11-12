"""The sd-server adapter by model family: what it can do, how sizes snap, and the sampling
fields families beyond Qwen-Image-2.1 need (distilled guidance, CLIP skip, extra ones)."""

from typing import Any

import pytest

from imaginer.backends.base import EditRequest, GenRequest, ImageInput
from imaginer.backends.sdcpp import SdCppBackend
from imaginer.core.config import ImageDefaults
from imaginer.domain.enums import Capability
from imaginer.domain.errors import InvalidRequestError
from imaginer.image_models import recipes
from tests.conftest import sd_config
from tests.fakes.fake_sdcpp import FakeSdCpp
from tests.helpers import image_bytes


async def _backend(url: str, **overrides: Any) -> SdCppBackend:
    backend = SdCppBackend("sd", sd_config(url, **overrides))
    await backend.health()
    return backend


async def test_a_familys_features_bound_what_the_engine_can_do(fake_sd: FakeSdCpp) -> None:
    flux = await _backend(fake_sd.url, family="flux1-dev")
    assert flux.capabilities() == {Capability.T2I}  # no reference images, no RGBA template
    klein = await _backend(fake_sd.url, family="flux2-klein-4b")
    assert klein.capabilities() == {Capability.T2I, Capability.EDIT, Capability.MULTI_REF}
    qwen = await _backend(fake_sd.url, family="qwen-image-21")
    assert Capability.RGBA in qwen.capabilities() and Capability.EDIT in qwen.capabilities()
    for backend in (flux, klein, qwen):
        await backend.aclose()


async def test_guidance_clip_skip_and_extra_sample_params(fake_sd: FakeSdCpp) -> None:
    defaults = ImageDefaults(
        width=256,
        height=256,
        steps=3,
        cfg_scale=1.0,
        guidance=3.5,
        clip_skip=2,
        extra={"sample_params": {"eta": 0.5}, "embed_image_metadata": False},
        extra_sample={"shifted_timestep": 250},
    )
    backend = await _backend(fake_sd.url, family="flux1-dev", defaults=defaults)
    await backend.generate(GenRequest(prompt="a fox", seed=1, params={"guidance": 4.0}))
    body = fake_sd.requests[-1]
    assert body["sample_params"] == {
        "eta": 0.5,  # from extra, merged rather than overwritten
        "shifted_timestep": 250,
        "sample_steps": 3,
        "guidance": {"txt_cfg": 1.0, "distilled_guidance": 4.0},  # the request's guidance
    }
    assert body["clip_skip"] == 2 and body["embed_image_metadata"] is False
    await backend.aclose()


async def test_sizes_snap_to_the_familys_multiple(fake_sd: FakeSdCpp) -> None:
    backend = await _backend(fake_sd.url, family="sdxl-checkpoint")
    assert backend.size_multiple == 64
    await backend.generate(GenRequest(prompt="a fox", seed=1, width=1000, height=700))
    body = fake_sd.requests[-1]
    assert (body["width"] % 64, body["height"] % 64) == (0, 0)
    await backend.aclose()


async def test_a_family_without_a_mask_instruction_refuses_masks(fake_sd: FakeSdCpp) -> None:
    backend = await _backend(fake_sd.url, family="flux2-klein-4b")
    photo = ImageInput(data=image_bytes(), mime="image/png")
    with pytest.raises(InvalidRequestError, match="can't edit with a mask"):
        await backend.edit(EditRequest(prompt="a lake", seed=1, images=(photo,), mask=photo))
    await backend.edit(EditRequest(prompt="a lake", seed=1, images=(photo,)))
    assert fake_sd.requests[-1]["prompt"] == "a lake"  # no mask instruction
    await backend.aclose()


async def test_pixel_space_families_make_no_previews(fake_sd: FakeSdCpp) -> None:
    backend = await _backend(fake_sd.url, family="chroma-radiance", preview="proj")
    await backend.generate(GenRequest(prompt="a fox", seed=1))
    assert fake_sd.requests[-1]["preview"] == "none"
    await backend.aclose()


async def test_the_engines_recipes_are_read_once(fake_sd: FakeSdCpp) -> None:
    backend = await _backend(fake_sd.url, family="qwen-image-21")
    await backend.health()
    family = recipes.family("qwen-image-21")
    assert family is not None and family.origin == "engine"
    assert family.mask is not None  # our notes still apply to the engine's recipe
    await backend.aclose()


async def test_image_guidance_and_reference_args_go_with_edits(fake_sd: FakeSdCpp) -> None:
    defaults = ImageDefaults(
        width=256, height=256, img_cfg_scale=1.5, ref_image_args="resize_before_vae=false"
    )
    photo = ImageInput(data=image_bytes(), mime="image/png")
    kontext = await _backend(fake_sd.url, family="flux1-kontext", defaults=defaults)
    await kontext.generate(GenRequest(prompt="a fox", seed=1))
    body = fake_sd.requests[-1]
    assert "img_cfg" not in body.get("sample_params", {}).get("guidance", {})
    assert "ref_image_args" not in body  # only edits take reference images
    await kontext.edit(EditRequest(prompt="make it red", seed=1, images=(photo,)))
    body = fake_sd.requests[-1]
    assert body["sample_params"]["guidance"]["img_cfg"] == 1.5
    assert body["ref_image_args"] == "resize_before_vae=false"
    await kontext.aclose()
    # Fibo 2 has no image CFG (sd.cpp would warn and use 1): none is sent.
    fibo = await _backend(fake_sd.url, family="fibo2-edit-turbo", defaults=defaults)
    await fibo.edit(EditRequest(prompt="make it red", seed=1, images=(photo,)))
    assert "img_cfg" not in fake_sd.requests[-1].get("sample_params", {}).get("guidance", {})
    await fibo.aclose()


async def test_an_edit_model_takes_as_many_images_as_it_handles(fake_sd: FakeSdCpp) -> None:
    photo = ImageInput(data=image_bytes(), mime="image/png")
    kontext = await _backend(fake_sd.url, family="flux1-kontext")
    assert kontext.max_ref_images == 1
    assert Capability.MULTI_REF not in kontext.capabilities()
    with pytest.raises(InvalidRequestError, match=r"Kontext dev takes at most 1 input image"):
        await kontext.edit(EditRequest(prompt="mix", seed=1, images=(photo, photo)))
    with pytest.raises(InvalidRequestError, match="can't edit with a mask"):
        await kontext.edit(EditRequest(prompt="a lake", seed=1, images=(photo,), mask=photo))
    await kontext.aclose()
    edit = await _backend(fake_sd.url, family="qwen-image-edit-2511", max_ref_images=10)
    assert edit.max_ref_images == 3
    assert edit.capabilities() == {Capability.EDIT, Capability.MULTI_REF}  # no text-to-image
    await edit.edit(EditRequest(prompt="mix", seed=1, images=(photo, photo, photo)))
    assert len(fake_sd.requests[-1]["ref_images"]) == 3
    await edit.aclose()


async def test_an_edit_needs_a_gguf_text_encoders_vision_projector(fake_sd: FakeSdCpp) -> None:
    from pathlib import Path

    from imaginer.core.config import SdCppLaunch

    launch = SdCppLaunch(
        diffusion_model=Path("qwen-image-edit-2511-Q4_K_M.gguf"),
        llm=Path("Qwen2.5-VL-7B-Instruct.Q4_K_M.gguf"),
        vae=Path("qwen_image_vae.safetensors"),
    )
    blind = await _backend(fake_sd.url, family="qwen-image-edit-2511", launch=launch)
    assert Capability.EDIT not in blind.capabilities()
    await blind.aclose()
    seeing = launch.model_copy(update={"llm_vision": Path("mmproj-f16.gguf")})
    edit = await _backend(fake_sd.url, family="qwen-image-edit-2511", launch=seeing)
    assert Capability.EDIT in edit.capabilities()
    await edit.aclose()
