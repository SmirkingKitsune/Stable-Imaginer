import asyncio
import base64
import threading
from typing import Any

import pytest

from imaginer.backends.base import EditRequest, GenProgress, GenRequest, ImageInput
from imaginer.backends.sdcpp import SdCppBackend
from imaginer.domain.enums import BackendStatus, Capability
from imaginer.domain.errors import EngineError, InvalidRequestError
from imaginer.services.media import inspect_image_bytes
from tests.conftest import sd_config
from tests.fakes.fake_sdcpp import RGBA_MARKER, FakeSdCpp
from tests.fakes.live import dead_url
from tests.helpers import image_bytes

ALL = {Capability.T2I, Capability.EDIT, Capability.MULTI_REF, Capability.RGBA}


async def _backend(url: str, **overrides: Any) -> SdCppBackend:
    backend = SdCppBackend("sd", sd_config(url, **overrides))
    await backend.health()
    return backend


async def test_health_and_capabilities(fake_sd: FakeSdCpp) -> None:
    backend = SdCppBackend("sd", sd_config(fake_sd.url))
    assert backend.capabilities() == set()  # nothing until probed
    health = await backend.health()
    assert health.status is BackendStatus.OK
    assert backend.capabilities() == ALL
    assert backend.model == "qwen_image_2.1-Q4_K.gguf"
    await backend.aclose()


async def test_declared_capabilities_narrow(fake_sd: FakeSdCpp) -> None:
    # e.g. sd-server started without --llm_vision: no editing.
    backend = await _backend(fake_sd.url, declared_capabilities={Capability.T2I, Capability.RGBA})
    assert backend.capabilities() == {Capability.T2I, Capability.RGBA}
    await backend.aclose()


async def test_recipe_without_ref_images_cannot_edit(fake_sd: FakeSdCpp) -> None:
    fake_sd.origin = "recipe"
    fake_sd.ref_images_feature = False
    backend = await _backend(fake_sd.url)
    assert backend.capabilities() == {Capability.T2I, Capability.RGBA}
    await backend.aclose()


async def test_command_line_model_reports_every_feature(fake_sd: FakeSdCpp) -> None:
    # Command-line models report all features; declared_capabilities is the real limit.
    fake_sd.ref_images_feature = False
    backend = await _backend(fake_sd.url)
    assert Capability.EDIT in backend.capabilities()
    await backend.aclose()


@pytest.mark.parametrize(("state", "detail"), [("loading", "model is loading"), ("empty", "empty")])
async def test_not_ready_is_degraded(fake_sd: FakeSdCpp, state: str, detail: str) -> None:
    fake_sd.state = state
    backend = SdCppBackend("sd", sd_config(fake_sd.url))
    health = await backend.health()
    assert health.status is BackendStatus.DEGRADED
    assert health.detail is not None and detail in health.detail
    await backend.aclose()


async def test_unreachable_is_down() -> None:
    backend = SdCppBackend("sd", sd_config(dead_url()))
    assert (await backend.health()).status is BackendStatus.DOWN
    await backend.aclose()


async def test_generate_maps_config_and_request(fake_sd: FakeSdCpp) -> None:
    fake_sd.step_delay_s = 0.03  # slow enough for the 10 ms poll to observe steps
    backend = await _backend(fake_sd.url, preview="proj")
    progress: list[GenProgress] = []
    submitted: list[str] = []

    async def on_progress(p: GenProgress) -> None:
        progress.append(p)

    async def on_submitted(job_id: str) -> None:
        submitted.append(job_id)

    (result,) = await backend.generate(
        GenRequest(
            prompt="a lighthouse at dusk",
            seed=1234,
            aspect_ratio="16:9",
            params={"steps": 5, "cfg_scale": 4.5},
            on_progress=on_progress,
            on_submitted=on_submitted,
        )
    )
    body = fake_sd.requests[-1]
    assert body["prompt"] == "a lighthouse at dusk"
    assert (body["width"], body["height"]) == (448, 256)  # 16:9, short side at the minimum
    assert body["seed"] == 1234
    assert body["batch_count"] == 1
    assert body["sample_params"] == {
        "sample_steps": 5,
        "sample_method": "euler",
        "guidance": {"txt_cfg": 4.5},
    }
    assert body["preview"] == "proj"
    assert "ref_images" not in body
    steps = [p.step for p in progress]
    assert len(steps) >= 2 and steps == sorted(set(steps)) and steps[-1] <= 5
    assert {p.total_steps for p in progress} == {5}
    assert submitted == list(fake_sd.jobs)
    assert (result.mime, result.width, result.height) == ("image/png", 448, 256)
    assert result.has_alpha is False
    assert result.seed == 1234  # read back from generation_info
    assert result.params["steps"] == 5 and result.params["cfg_scale"] == 4.5
    assert inspect_image_bytes(result.data).width == 448
    await backend.aclose()


async def test_transparent_uses_rgba_prompt_and_png(fake_sd: FakeSdCpp) -> None:
    backend = await _backend(fake_sd.url)
    (result,) = await backend.generate(
        GenRequest(prompt="a cute dragon sticker.", seed=7, alpha=True)
    )
    prompt = fake_sd.requests[-1]["prompt"]
    assert prompt == (
        f"{RGBA_MARKER} a cute dragon sticker. "
        "The image has alpha channel and the background is transparent."
    )
    assert result.has_alpha is True
    assert result.mime == "image/png"
    await backend.aclose()


async def test_sigma_presets_expand(fake_sd: FakeSdCpp) -> None:
    backend = await _backend(fake_sd.url)
    await backend.generate(
        GenRequest(prompt="x", seed=1, params={"sigmas": "qwen-image-2.1-turbo"})
    )
    assert fake_sd.requests[-1]["sample_params"]["custom_sigmas"][:2] == [1.0, 0.978453]
    with pytest.raises(ValueError, match="unknown sigma preset"):
        SdCppBackend("sd", sd_config(fake_sd.url, defaults={"sigmas": "nope"}))
    await backend.aclose()


async def test_unsupported_preview_mode_falls_back(fake_sd: FakeSdCpp) -> None:
    fake_sd.preview_modes = ["none"]
    backend = await _backend(fake_sd.url, preview="tae")
    await backend.generate(GenRequest(prompt="x", seed=1))
    assert fake_sd.requests[-1]["preview"] == "none"
    await backend.aclose()


async def test_edit_sends_ordered_reference_images(fake_sd: FakeSdCpp) -> None:
    backend = await _backend(fake_sd.url)
    first = ImageInput(data=image_bytes(color=(255, 0, 0)), mime="image/png")
    second = ImageInput(data=image_bytes(fmt="JPEG"), mime="image/jpeg")
    mask = ImageInput(data=image_bytes(color=(255, 255, 255)), mime="image/png")
    (result,) = await backend.edit(
        EditRequest(prompt="put the hat on the cat", seed=3, images=(first, second), mask=mask)
    )
    refs = fake_sd.requests[-1]["ref_images"]
    assert [r.split(";")[0] for r in refs] == [
        "data:image/png",
        "data:image/jpeg",
        "data:image/png",
    ]
    assert base64.b64decode(refs[1].split(",", 1)[1]) == second.data
    assert result.engine_meta["generation_info"]["ref_images"] == 3
    await backend.aclose()


async def test_edit_limits(fake_sd: FakeSdCpp) -> None:
    backend = await _backend(fake_sd.url, max_ref_images=2)
    image = ImageInput(data=image_bytes(), mime="image/png")
    with pytest.raises(InvalidRequestError, match="at most 2"):
        await backend.edit(EditRequest(prompt="x", seed=1, images=(image, image, image)))
    with pytest.raises(InvalidRequestError, match="at least one"):
        await backend.edit(EditRequest(prompt="x", seed=1, images=()))
    await backend.aclose()


async def test_engine_failure(fake_sd: FakeSdCpp) -> None:
    fake_sd.fail_next = "CUDA out of memory"
    backend = await _backend(fake_sd.url)
    with pytest.raises(EngineError, match="CUDA out of memory"):
        await backend.generate(GenRequest(prompt="x", seed=1))
    await backend.aclose()


async def test_queue_full_is_an_engine_error(fake_sd: FakeSdCpp) -> None:
    fake_sd.queue_full = True
    backend = await _backend(fake_sd.url)
    with pytest.raises(EngineError, match="429"):
        await backend.generate(GenRequest(prompt="x", seed=1))
    await backend.aclose()


async def test_cancelling_cancels_the_engine_job(fake_sd: FakeSdCpp) -> None:
    fake_sd.hold = threading.Event()
    backend = await _backend(fake_sd.url)
    task = asyncio.create_task(backend.generate(GenRequest(prompt="slow", seed=1)))
    assert await asyncio.to_thread(fake_sd.started.wait, 10)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    (job_id,) = fake_sd.jobs
    assert fake_sd.cancels == [job_id]
    fake_sd.hold.set()
    for _ in range(100):
        if fake_sd.jobs[job_id]["status"] == "cancelled":
            break
        await asyncio.sleep(0.01)
    assert fake_sd.jobs[job_id]["status"] == "cancelled"
    await backend.aclose()


async def test_cancel_during_submit_still_cancels_the_engine_job(fake_sd: FakeSdCpp) -> None:
    fake_sd.hold = threading.Event()
    backend = await _backend(fake_sd.url)
    task = asyncio.create_task(backend.generate(GenRequest(prompt="x", seed=1)))
    await asyncio.sleep(0)  # the submit request is now in flight
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert fake_sd.cancels == list(fake_sd.jobs)  # the late-arriving job id was cancelled
    await backend.aclose()


async def test_edit_size_follows_first_image_and_refs_are_fitted(fake_sd: FakeSdCpp) -> None:
    backend = await _backend(fake_sd.url, max_ref_pixels=40_000)
    wide = ImageInput(data=image_bytes(800, 400, noise=True), mime="image/png")
    (result,) = await backend.edit(EditRequest(prompt="brighter", seed=1, images=(wide,)))
    body = fake_sd.requests[-1]
    assert (body["width"], body["height"]) == (512, 256)  # 2:1 like the input
    sent = inspect_image_bytes(base64.b64decode(body["ref_images"][0].split(",", 1)[1]))
    assert (sent.width, sent.height) == (282, 141)  # downscaled to <= 40k pixels
    assert (result.width, result.height) == (512, 256)
    await backend.aclose()


async def test_mask_goes_last_with_an_instruction(fake_sd: FakeSdCpp) -> None:
    backend = await _backend(fake_sd.url)
    photo = ImageInput(data=image_bytes(), mime="image/png")
    mask = ImageInput(data=image_bytes(color=(255, 255, 255)), mime="image/png")
    await backend.edit(EditRequest(prompt="make it a lake", seed=1, images=(photo,), mask=mask))
    body = fake_sd.requests[-1]
    assert len(body["ref_images"]) == 2
    assert body["prompt"].startswith("make it a lake The last image is a mask")
    await backend.aclose()
