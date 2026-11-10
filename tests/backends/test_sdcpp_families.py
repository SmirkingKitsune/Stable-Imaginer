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
