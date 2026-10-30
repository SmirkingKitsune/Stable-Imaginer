"""Checks against a real sd-server running Qwen-Image-2.1 (tested with master-1024-77b72a2).

Opt in::

    IMAGINER_IT_SD_URL=http://127.0.0.1:1234 uv run pytest -m integration

``IMAGINER_IT_SD_STEPS`` (default 8) and ``IMAGINER_IT_SD_CFG`` (default 6.0) tune speed.
Edit needs sd-server started with ``--llm_vision`` (the Qwen3-VL mmproj).

These settle, on real hardware, what the docs leave open: the per-image seed in
``generation_info``, that PNG output really carries alpha, and that engine-side cancel works.
"""

import asyncio
import io
import os
from collections.abc import AsyncIterator

import pytest
from PIL import Image

from imaginer.backends.base import EditRequest, GenRequest, ImageInput
from imaginer.backends.sdcpp import SdCppBackend
from imaginer.core.config import ImageDefaults, SdCppBackendConfig, TimeoutConfig
from imaginer.domain.enums import BackendStatus, Capability
from tests.helpers import image_bytes

URL = os.environ.get("IMAGINER_IT_SD_URL")
STEPS = int(os.environ.get("IMAGINER_IT_SD_STEPS", "8"))
CFG = float(os.environ.get("IMAGINER_IT_SD_CFG", "6.0"))

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not URL, reason="set IMAGINER_IT_SD_URL to run against sd-server"),
]


@pytest.fixture
async def backend() -> AsyncIterator[SdCppBackend]:
    assert URL
    config = SdCppBackendConfig(
        type="sdcpp",
        base_url=URL,
        poll_interval_s=0.5,
        timeouts=TimeoutConfig(read=120),
        defaults=ImageDefaults(width=512, height=512, steps=STEPS, cfg_scale=CFG, sampler="euler"),
    )
    sd = SdCppBackend("it", config)
    health = await sd.health()
    assert health.status is BackendStatus.OK, health.detail
    yield sd
    await sd.aclose()


async def test_capabilities(backend: SdCppBackend) -> None:
    assert {Capability.T2I, Capability.RGBA} <= backend.capabilities()


async def test_t2i_reports_the_seed_it_used(backend: SdCppBackend) -> None:
    (result,) = await backend.generate(GenRequest(prompt="a red apple on a white table", seed=4242))
    assert (result.mime, result.width, result.height) == ("image/png", 512, 512)
    assert result.seed == 4242
    assert "seed" in result.engine_meta["generation_info"]


async def test_rgba_output_has_real_alpha(backend: SdCppBackend) -> None:
    (result,) = await backend.generate(
        GenRequest(prompt="a cute cartoon dragon sticker", seed=7, alpha=True)
    )
    assert result.mime == "image/png"
    assert result.has_alpha, "expected transparent pixels in the PNG"


async def test_two_reference_edit(backend: SdCppBackend) -> None:
    if Capability.MULTI_REF not in backend.capabilities():
        pytest.skip("backend does not advertise multi_ref")
    red = ImageInput(data=image_bytes(256, 256, color=(220, 30, 30)), mime="image/png")
    blue = ImageInput(data=image_bytes(256, 256, color=(30, 30, 220)), mime="image/png")
    (result,) = await backend.edit(
        EditRequest(
            prompt=(
                "Combine the two colors into a split image: left half like image 1, "
                "right half like image 2"
            ),
            seed=11,
            images=(red, blue),
            width=512,
            height=512,
        )
    )
    assert (result.width, result.height) == (512, 512)


async def test_cancel_reaches_the_engine(backend: SdCppBackend) -> None:
    submitted: asyncio.Future[str] = asyncio.get_running_loop().create_future()

    async def on_submitted(job_id: str) -> None:
        submitted.set_result(job_id)

    task = asyncio.create_task(
        backend.generate(
            GenRequest(
                prompt="a detailed city", seed=1, params={"steps": 50}, on_submitted=on_submitted
            )
        )
    )
    job_id = await asyncio.wait_for(submitted, 30)
    await asyncio.sleep(2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    transport = backend._transport
    for _ in range(120):
        job = await transport.get_json(f"/sdcpp/v1/jobs/{job_id}")
        if job["status"] in ("cancelled", "completed", "failed"):
            break
        await asyncio.sleep(0.5)
    assert job["status"] == "cancelled"


async def test_mask_edit(backend: SdCppBackend) -> None:
    """Qwen-Image-2.1's "separate mask" path: the mask goes last, the prompt says so.

    Only checks that the engine accepts it and returns an image shaped like the input; how
    faithfully the model keeps the unmasked area is for a human to judge (look at the file
    written next to the test's tmp dir if needed).
    """
    if Capability.MULTI_REF not in backend.capabilities():
        pytest.skip("a mask is a second input image: needs multi_ref")
    photo = ImageInput(data=image_bytes(640, 384, color=(60, 140, 60)), mime="image/png")
    mask_im = Image.new("L", (640, 384), 0)
    mask_im.paste(255, (320, 0, 640, 384))  # change only the right half
    buf = io.BytesIO()
    mask_im.save(buf, format="PNG")
    mask = ImageInput(data=buf.getvalue(), mime="image/png")
    (result,) = await backend.edit(
        EditRequest(prompt="paint this area bright blue", seed=5, images=(photo,), mask=mask)
    )
    assert abs(result.width / result.height - 640 / 384) < 0.1
