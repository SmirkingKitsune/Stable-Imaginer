import asyncio
import subprocess
import sys
import threading
from typing import Any

import pytest
from PIL import Image

from imaginer.backends import registry as registry_module
from imaginer.backends.base import EditRequest, GenProgress, GenRequest, ImageInput
from imaginer.backends.diffusers_inproc import DiffusersBackend, missing_dependencies
from imaginer.backends.registry import BackendRegistry
from imaginer.core.config import DiffusersBackendConfig, ImageDefaults
from imaginer.domain.enums import BackendStatus, Capability
from imaginer.domain.errors import EngineError, EngineUnavailableError
from tests.conftest import SettingsFactory
from tests.fakes.fake_diffusers import FakeLoader
from tests.fakes.fake_sdcpp import RGBA_MARKER
from tests.helpers import image_bytes

ALL = {Capability.T2I, Capability.EDIT, Capability.MULTI_REF, Capability.RGBA}


def diffusers_config(**overrides: Any) -> DiffusersBackendConfig:
    fields: dict[str, Any] = {
        "type": "diffusers",
        "device": "cuda:0",
        "defaults": ImageDefaults(width=512, height=512, steps=4, cfg_scale=1.0),
    }
    fields.update(overrides)
    return DiffusersBackendConfig(**fields)


def _backend(
    loader: FakeLoader | None = None, **overrides: Any
) -> tuple[DiffusersBackend, FakeLoader]:
    loader = loader or FakeLoader()
    return DiffusersBackend("diff", diffusers_config(**overrides), loader=loader), loader


# --- packaging ------------------------------------------------------------------


def test_every_backend_module_imports_without_torch() -> None:
    """The [diffusers] extra is optional: nothing may import torch at module level."""
    code = (
        "import sys\n"
        "for name in ('torch', 'diffusers', 'transformers', 'accelerate'):\n"
        "    sys.modules[name] = None  # any import of them now raises ImportError\n"
        "import imaginer.backends.diffusers_inproc, imaginer.backends.registry\n"
        "import imaginer.backends.llamacpp, imaginer.backends.sdcpp, imaginer.main\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


async def test_missing_extra_is_reported_not_fatal(
    make_settings: SettingsFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(registry_module, "missing_dependencies", lambda: ["torch", "diffusers"])
    registry = BackendRegistry.from_settings(make_settings(backends={"diff": diffusers_config()}))
    (info,) = await registry.describe()
    assert info.status is BackendStatus.UNAVAILABLE
    assert info.detail is not None
    assert "missing torch, diffusers" in info.detail and "--extra diffusers" in info.detail


def test_missing_dependencies_checks_without_importing() -> None:
    assert set(missing_dependencies()) <= {"torch", "diffusers", "transformers", "accelerate"}


# --- health and loading -----------------------------------------------------------


async def test_health_lazy_load_and_capabilities() -> None:
    backend, loader = _backend()
    health = await backend.health()
    assert health.status is BackendStatus.OK
    assert health.detail == "Qwen/Qwen-Image-2.1 loads on first use"
    assert backend.capabilities() == ALL
    assert loader.loads == 0  # health never loads 20+ GB of weights by itself
    await backend.generate(GenRequest(prompt="x", seed=1))
    assert (await backend.health()).detail == "Qwen/Qwen-Image-2.1 loaded on cuda:0 (bfloat16)"
    await backend.aclose()
    assert loader.closed == 1


async def test_environment_problems_are_down() -> None:
    loader = FakeLoader()
    loader.problem = "torch 2.14.1+cpu cannot use CUDA"
    backend, _ = _backend(loader)
    health = await backend.health()
    assert (health.status, health.detail) == (
        BackendStatus.DOWN,
        "torch 2.14.1+cpu cannot use CUDA",
    )
    assert backend.capabilities() == set()


async def test_slow_environment_check_is_degraded_meanwhile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from imaginer.backends import diffusers_inproc

    monkeypatch.setattr(diffusers_inproc, "ENV_CHECK_TIMEOUT_S", 0.05)
    loader = FakeLoader()
    loader.check_gate = threading.Event()
    backend, _ = _backend(loader)
    first = await backend.health()
    assert (first.status, first.detail) == (BackendStatus.DEGRADED, "importing torch and diffusers")
    loader.check_gate.set()
    await asyncio.sleep(0.05)
    assert (await backend.health()).status is BackendStatus.OK
    assert loader.checks == 1  # the check ran once, not once per probe


async def test_preload() -> None:
    backend, loader = _backend(preload=True)
    assert "loading" in ((await backend.health()).detail or "")
    for _ in range(100):
        if loader.loads and "loaded on" in ((await backend.health()).detail or ""):
            break
        await asyncio.sleep(0.01)
    assert loader.loads == 1
    await backend.aclose()


async def test_load_failure_is_reported_and_retried() -> None:
    loader = FakeLoader()
    loader.load_error = "CUDA out of memory"
    backend, _ = _backend(loader)
    with pytest.raises(EngineUnavailableError, match=r"could not load .*CUDA out of memory"):
        await backend.generate(GenRequest(prompt="x", seed=1))
    health = await backend.health()
    assert (health.status, health.detail) == (
        BackendStatus.OK,
        "last load failed: CUDA out of memory; next request retries loading",
    )
    loader.load_error = None
    await backend.generate(GenRequest(prompt="x", seed=1))
    assert loader.loads == 2


@pytest.mark.parametrize("preload", [False, True])
async def test_registry_retries_failed_load_on_request(
    make_settings: SettingsFactory,
    preload: bool,
) -> None:
    loader = FakeLoader()
    loader.load_error = "temporary download failure"
    config = diffusers_config(preload=preload)
    backend = DiffusersBackend("diff", config, loader=loader)
    registry = BackendRegistry.from_settings(
        make_settings(backends={"diff": config}),
        factories={"diffusers": lambda name, config: backend},
    )
    try:
        routed = (await registry.image_backend("t2i")).backend
        with pytest.raises(EngineUnavailableError, match="temporary download failure"):
            await routed.generate(GenRequest(prompt="x", seed=1))
        # Force a new probe: cached healthy state must not hide the failure path.
        await registry.refresh()
        await registry.refresh()
        assert loader.loads == 1  # polling never retries downloads
        loader.load_error = None
        routed = (await registry.image_backend("t2i")).backend
        await routed.generate(GenRequest(prompt="recovered", seed=1))
        assert loader.loads == 2
        assert "loaded on" in ((await backend.health()).detail or "")
    finally:
        await registry.aclose()


@pytest.mark.parametrize(
    ("schedule", "expected_steps"),
    [("qwen-image-2.1-turbo", 8), ([1.0, 0.5, 0.0], 2), ([1.0, 0.5], 2)],
)
async def test_sigma_steps_match_progress_and_result(
    schedule: str | list[float],
    expected_steps: int,
) -> None:
    backend, loader = _backend(
        defaults=ImageDefaults(width=256, height=256, steps=40, sigmas=schedule),
    )
    progress: list[GenProgress] = []

    async def report(update: GenProgress) -> None:
        progress.append(update)

    try:
        (result,) = await backend.generate(GenRequest(prompt="x", seed=1, on_progress=report))
        assert loader.pipeline.steps_run == expected_steps
        assert result.params["steps"] == expected_steps
        assert [p.step for p in progress] == list(range(1, expected_steps + 1))
        assert {p.total_steps for p in progress} == {expected_steps}
    finally:
        await backend.aclose()


# --- generation -------------------------------------------------------------------


async def test_generate_maps_the_pipeline_call() -> None:
    backend, loader = _backend()
    progress: list[GenProgress] = []

    async def on_progress(update: GenProgress) -> None:
        progress.append(update)

    (result,) = await backend.generate(
        GenRequest(
            prompt="a lighthouse", seed=77, aspect_ratio="16:9", negative_prompt="fog",
            on_progress=on_progress,
        )
    )  # fmt: skip
    call = loader.pipeline.calls[-1]
    assert call["prompt"] == "a lighthouse"
    assert (call["width"], call["height"]) == (672, 384)  # 16:9 at the 512x512 area
    assert call["num_inference_steps"] == 4
    assert call["true_cfg_scale"] == 1.0
    assert "negative_prompt" not in call  # ignored by the pipeline without guidance
    assert call["generator"] == 77
    assert [p.step for p in progress] == [1, 2, 3, 4]  # all delivered before returning
    assert {p.total_steps for p in progress} == {4}
    assert (result.mime, result.width, result.height) == ("image/png", 672, 384)
    assert result.has_alpha is False  # opaque RGBA from the VAE is stored as RGB
    assert Image.open(__import__("io").BytesIO(result.data)).mode == "RGB"
    assert (result.seed, result.params["steps"], result.params["cfg_scale"]) == (77, 4, 1.0)
    await backend.generate(GenRequest(prompt="again", seed=1))
    assert loader.loads == 1


async def test_guidance_and_sigma_presets() -> None:
    backend, loader = _backend()
    await backend.generate(
        GenRequest(
            prompt="x",
            seed=1,
            negative_prompt="blurry",
            params={"cfg_scale": 4.0, "sigmas": "qwen-image-2.1-turbo"},
        )
    )
    call = loader.pipeline.calls[-1]
    assert (call["true_cfg_scale"], call["negative_prompt"]) == (4.0, "blurry")
    assert call["num_inference_steps"] == 8  # one per sigma, trailing 0.0 dropped
    assert call["sigmas"][0] == 1.0 and 0.0 not in call["sigmas"]


async def test_transparent_output() -> None:
    backend, loader = _backend()
    (result,) = await backend.generate(GenRequest(prompt="a dragon sticker", seed=3, alpha=True))
    assert loader.pipeline.calls[-1]["prompt"].startswith(RGBA_MARKER)
    assert (result.mime, result.has_alpha) == ("image/png", True)


async def test_edit_passes_images_in_order_with_mask_last() -> None:
    backend, loader = _backend()
    photo = ImageInput(data=image_bytes(400, 200, color=(255, 0, 0)), mime="image/png")
    ref = ImageInput(data=image_bytes(100, 100, color=(0, 255, 0)), mime="image/png")
    mask = ImageInput(data=image_bytes(400, 200, color=(255, 255, 255)), mime="image/png")
    (result,) = await backend.edit(
        EditRequest(prompt="add a tree", seed=2, images=(photo, ref), mask=mask)
    )
    call = loader.pipeline.calls[-1]
    assert [im.size for im in call["image"]] == [(400, 200), (100, 100), (400, 200)]
    assert call["image"][1].getpixel((0, 0))[:3] == (0, 255, 0)
    assert call["prompt"].startswith("add a tree The last image is a mask")
    assert (result.width, result.height) == (
        512,
        256,
    )  # 2:1 like the first input, never smaller than 256


async def test_cancel_stops_within_a_step() -> None:
    backend, loader = _backend(defaults=ImageDefaults(width=512, height=512, steps=50))
    loader.pipeline.hold = threading.Event()
    task = asyncio.create_task(backend.generate(GenRequest(prompt="slow", seed=1)))
    assert await asyncio.to_thread(loader.pipeline.started.wait, 10)
    await asyncio.sleep(0.05)
    task.cancel()
    loader.pipeline.hold.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert loader.pipeline.finished.is_set()  # the worker thread had stopped before we returned
    assert loader.pipeline.steps_run < 50


async def test_pipeline_errors_are_engine_errors() -> None:
    backend, loader = _backend()
    loader.pipeline.fail = "shape mismatch"
    with pytest.raises(EngineError, match="shape mismatch"):
        await backend.generate(GenRequest(prompt="x", seed=1))
