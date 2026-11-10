import asyncio
import base64
import hashlib
import threading
from pathlib import Path
from typing import Any

import httpx
import pytest

from imaginer.backends.base import EditRequest, GenProgress, GenRequest, ImageInput
from imaginer.backends.common import lora_matches
from imaginer.backends.openai_http.transport import HttpTransport
from imaginer.backends.sdcpp import SdCppBackend
from imaginer.core.config import LoraRef
from imaginer.domain.enums import BackendStatus, Capability
from imaginer.domain.errors import EngineError, EngineTimeoutError, InvalidRequestError
from imaginer.services.media import inspect_image_bytes
from tests.conftest import sd_config
from tests.fakes.fake_sdcpp import RGBA_MARKER, FakeSdCpp
from tests.fakes.live import dead_url
from tests.helpers import image_bytes, safetensors_file

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
    assert fake_sd.cancels == []  # already terminal: do not send a redundant cancellation
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


@pytest.mark.parametrize("cancel_fails", [False, True])
async def test_poll_failure_cancels_remote_job_and_preserves_error(cancel_fails: bool) -> None:
    calls: list[tuple[str, str]] = []

    def reply(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        if request.url.path.endswith("/img_gen"):
            return httpx.Response(202, json={"id": "job1"})
        if request.url.path.endswith("/cancel"):
            return httpx.Response(500 if cancel_fails else 200, json={})
        raise httpx.ReadTimeout("lost poll", request=request)

    config = sd_config("http://engine", retry={"attempts": 1})
    transport = HttpTransport(
        name="sd",
        base_url=config.base_url,
        timeouts=config.timeouts,
        retry=config.retry,
        client=httpx.AsyncClient(base_url=config.base_url, transport=httpx.MockTransport(reply)),
    )
    backend = SdCppBackend("sd", config, transport=transport)
    try:
        with pytest.raises(EngineTimeoutError, match="lost poll"):
            await backend.generate(GenRequest(prompt="x", seed=1))
        assert calls == [
            ("POST", "/sdcpp/v1/img_gen"),
            ("GET", "/sdcpp/v1/jobs/job1"),
            ("POST", "/sdcpp/v1/jobs/job1/cancel"),
        ]
    finally:
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


# --- LoRAs and presets ------------------------------------------------------------

DISTILL = "qwen/p_q_image_2.1_8_steps_v0.1_comfyui_bf16.safetensors"
FAST = {
    "fast": {
        "description": "8-step draft",
        "loras": [{"path": "p_q_image_2.1_8_steps_v0.1_comfyui_bf16"}],
        "steps": 2,
        "cfg_scale": 1.0,
    },
    "plain": {"loras": []},
}


def _defaults(**fields: Any) -> dict[str, Any]:
    return {"width": 256, "height": 256, "steps": 3, "cfg_scale": 6.0, **fields}


async def test_lora_capability_comes_with_a_lora_listing(fake_sd: FakeSdCpp) -> None:
    backend = await _backend(fake_sd.url)  # an older server: no listing, no lora field
    assert Capability.LORA not in backend.capabilities()
    listed: list[str] | None = backend.available_loras
    assert listed is None
    fake_sd.loras = []  # a newer server with an empty folder can still take LoRAs
    await backend.health()
    assert Capability.LORA in backend.capabilities()
    listed = backend.available_loras
    assert listed == []
    fake_sd.loras, fake_sd.origin, fake_sd.lora_feature = [DISTILL], "recipe", False
    await backend.health()
    assert Capability.LORA not in backend.capabilities()
    await backend.aclose()


async def test_preset_sends_loras_and_its_settings(fake_sd: FakeSdCpp) -> None:
    fake_sd.loras = [DISTILL, "style/ink.safetensors"]
    backend = await _backend(fake_sd.url, presets=FAST)
    (result,) = await backend.generate(GenRequest(prompt="x", seed=1, preset="fast"))
    body = fake_sd.requests[-1]
    # Named by its stem in the config; sent as the path the server listed.
    assert body["lora"] == [{"path": DISTILL, "multiplier": 1.0}]
    assert body["sample_params"]["sample_steps"] == 2
    assert body["sample_params"]["guidance"] == {"txt_cfg": 1.0}
    assert body["sample_params"]["sample_method"] == "euler"  # unset in the preset: default
    assert result.params["preset"] == "fast"
    assert result.params["loras"] == [{"path": DISTILL, "weight": 1.0}]
    assert result.params["steps"] == 2
    await backend.aclose()


async def test_request_settings_win_over_the_preset(fake_sd: FakeSdCpp) -> None:
    fake_sd.loras = [DISTILL]
    backend = await _backend(fake_sd.url, presets=FAST)
    await backend.generate(GenRequest(prompt="x", seed=1, preset="fast", params={"steps": 5}))
    assert fake_sd.requests[-1]["sample_params"]["sample_steps"] == 5
    await backend.aclose()


async def test_preset_steps_replace_a_default_sigma_schedule(fake_sd: FakeSdCpp) -> None:
    presets = {"slow": {"steps": 20}, "turbo": {"sigmas": "qwen-image-2.1-turbo"}}
    defaults = _defaults(sigmas="qwen-image-2.1-turbo")
    backend = await _backend(fake_sd.url, defaults=defaults, presets=presets)
    await backend.generate(GenRequest(prompt="x", seed=1, preset="slow"))
    sample = fake_sd.requests[-1]["sample_params"]
    assert sample["sample_steps"] == 20 and "custom_sigmas" not in sample
    await backend.generate(GenRequest(prompt="x", seed=1, preset="turbo"))
    assert "custom_sigmas" in fake_sd.requests[-1]["sample_params"]
    await backend.aclose()


@pytest.mark.parametrize("preset", [None, "turbo"])
@pytest.mark.parametrize("explicit_sigmas", [False, True])
async def test_request_steps_replace_only_inherited_sigmas(
    fake_sd: FakeSdCpp, preset: str | None, explicit_sigmas: bool
) -> None:
    backend = await _backend(
        fake_sd.url,
        defaults=_defaults(sigmas="qwen-image-2.1-turbo"),
        presets={"turbo": {"sigmas": "qwen-image-2.1-turbo"}},
    )
    params: dict[str, Any] = {"steps": 5}
    if explicit_sigmas:
        params["sigmas"] = [1.0, 0.5, 0.0]
    try:
        (result,) = await backend.generate(
            GenRequest(prompt="x", seed=1, preset=preset, params=params)
        )
        sample = fake_sd.requests[-1]["sample_params"]
        assert sample["sample_steps"] == 5
        if explicit_sigmas:
            assert sample["custom_sigmas"] == params["sigmas"]
        else:
            assert "custom_sigmas" not in sample
            assert "sigmas" not in result.params
    finally:
        await backend.aclose()


async def test_default_loras_apply_unless_a_preset_drops_them(fake_sd: FakeSdCpp) -> None:
    fake_sd.loras = ["style/ink.safetensors"]
    defaults = _defaults(loras=[{"path": r"style\ink.safetensors", "weight": 0.6}])
    backend = await _backend(fake_sd.url, defaults=defaults, presets=FAST)
    await backend.generate(GenRequest(prompt="x", seed=1))
    assert fake_sd.requests[-1]["lora"] == [{"path": "style/ink.safetensors", "multiplier": 0.6}]
    (result,) = await backend.generate(GenRequest(prompt="x", seed=1, preset="plain"))
    assert "lora" not in fake_sd.requests[-1]
    assert "loras" not in result.params
    await backend.aclose()


async def test_lora_added_after_the_probe_is_found(fake_sd: FakeSdCpp) -> None:
    fake_sd.loras = []
    backend = await _backend(fake_sd.url, presets=FAST)
    fake_sd.loras = [DISTILL]  # dropped into the folder since: sd-server rescans per request
    await backend.generate(GenRequest(prompt="x", seed=1, preset="fast"))
    assert fake_sd.requests[-1]["lora"][0]["path"] == DISTILL
    await backend.aclose()


async def test_a_requests_loras_go_on_top_of_the_presets(fake_sd: FakeSdCpp) -> None:
    fake_sd.loras = [DISTILL, "style/ink.safetensors"]
    backend = await _backend(fake_sd.url, presets=FAST)
    picks = (LoraRef(path="ink", weight=0.7), LoraRef(path=DISTILL, weight=0.5))
    (result,) = await backend.generate(GenRequest(prompt="x", seed=1, preset="fast", loras=picks))
    # The preset's LoRA keeps its place at the request's weight; the request's own follows.
    applied = [{"path": DISTILL, "weight": 0.5}, {"path": "style/ink.safetensors", "weight": 0.7}]
    assert fake_sd.requests[-1]["lora"] == [
        {"path": x["path"], "multiplier": x["weight"]} for x in applied
    ]
    assert result.params["loras"] == applied
    # Weight 0 turns one off.
    off = (LoraRef(path="p_q_image_2.1_8_steps_v0.1_comfyui_bf16.safetensors", weight=0),)
    (result,) = await backend.generate(GenRequest(prompt="x", seed=1, preset="fast", loras=off))
    assert "lora" not in fake_sd.requests[-1]
    assert "loras" not in result.params
    assert result.params["steps"] == 2  # the rest of the preset still applies
    await backend.aclose()


async def test_the_loras_of_an_engine_launched_here_are_hashed(
    fake_sd: FakeSdCpp, tmp_path: Path
) -> None:
    folder = tmp_path / "loras"
    (folder / "style").mkdir(parents=True)
    (folder / "style" / "ink.safetensors").write_bytes(b"ink weights")
    (tmp_path / "outside.safetensors").write_bytes(b"not in the folder")
    fake_sd.loras = ["style/ink.safetensors", "gone.safetensors", "../outside.safetensors"]
    backend = await _backend(fake_sd.url, launch={"lora_model_dir": folder})
    picks = (LoraRef(path="ink"), LoraRef(path="gone"), LoraRef(path="../outside.safetensors"))
    (result,) = await backend.generate(GenRequest(prompt="x", seed=1, loras=picks))
    ink = hashlib.sha256(b"ink weights").hexdigest()
    assert result.params["loras"] == [
        {"path": "style/ink.safetensors", "weight": 1.0, "sha256": ink},
        {"path": "gone.safetensors", "weight": 1.0},  # listed, but no file to read
        {"path": "../outside.safetensors", "weight": 1.0},  # never read outside the folder
    ]
    assert all("sha256" not in x for x in fake_sd.requests[-1]["lora"])
    await backend.aclose()


async def test_a_loras_trigger_phrase_is_added_to_the_prompt(
    fake_sd: FakeSdCpp, tmp_path: Path
) -> None:
    folder = tmp_path / "loras"
    folder.mkdir()
    safetensors_file(folder / "clay.safetensors", {"trigger_word": "clay sculpture style"})
    safetensors_file(folder / "plain.safetensors", {"ss_network_dim": "16"})
    fake_sd.loras = ["clay.safetensors", "plain.safetensors"]
    backend = await _backend(fake_sd.url, launch={"lora_model_dir": folder})
    picks = (LoraRef(path="clay"), LoraRef(path="plain"))
    (result,) = await backend.generate(GenRequest(prompt="a fox.", seed=1, loras=picks))
    assert fake_sd.requests[-1]["prompt"] == "a fox. clay sculpture style"
    assert [x.get("trigger") for x in result.params["loras"]] == ["clay sculpture style", None]
    (again,) = await backend.generate(
        GenRequest(prompt="A fox in Clay Sculpture Style", seed=1, loras=picks)
    )
    assert fake_sd.requests[-1]["prompt"] == "A fox in Clay Sculpture Style"  # has it already
    assert "trigger" not in again.params["loras"][0]
    await backend.aclose()

    remote = await _backend(fake_sd.url)  # its LoRA files aren't on this machine
    await remote.generate(GenRequest(prompt="a fox", seed=1, loras=picks))
    assert fake_sd.requests[-1]["prompt"] == "a fox"
    await remote.aclose()


@pytest.mark.parametrize(
    ("listed", "match"),
    [
        ([], "it lists none .*--lora-model-dir"),
        (["style/ink.safetensors"], r"has no LoRA .* \(it has: style/ink\.safetensors\)"),
        (
            ["a/p_q_image_2.1_8_steps_v0.1_comfyui_bf16.safetensors", DISTILL],
            "could be any of a/p_q.*, qwen/p_q",
        ),
    ],
)
async def test_lora_the_engine_cannot_name(
    fake_sd: FakeSdCpp, listed: list[str], match: str
) -> None:
    fake_sd.loras = listed
    backend = await _backend(fake_sd.url, presets=FAST)
    with pytest.raises(InvalidRequestError, match=match):
        await backend.generate(GenRequest(prompt="x", seed=1, preset="fast"))
    assert fake_sd.requests == []  # nothing was sent
    await backend.aclose()


async def test_unknown_preset(fake_sd: FakeSdCpp) -> None:
    backend = await _backend(fake_sd.url, presets=FAST)
    with pytest.raises(
        InvalidRequestError, match=r"no image preset 'slow' \(it has: fast, plain\)"
    ):
        await backend.generate(GenRequest(prompt="x", seed=1, preset="slow"))
    with pytest.raises(ValueError, match="unknown sigma preset"):
        SdCppBackend("sd", sd_config(fake_sd.url, presets={"t": {"sigmas": "nope"}}))
    await backend.aclose()


def test_lora_matches() -> None:
    catalog: dict[str, dict[str, Any]] = {
        "a/one.safetensors": {"name": "one"},
        "b/one.gguf": {"name": "one"},
        "two.safetensors": {"name": "two"},
    }
    assert lora_matches("two.safetensors", catalog) == ["two.safetensors"]
    assert lora_matches(r"a\one.safetensors", catalog) == ["a/one.safetensors"]
    assert lora_matches("one.gguf", catalog) == ["b/one.gguf"]
    assert lora_matches("two", catalog) == ["two.safetensors"]
    assert lora_matches("one", catalog) == ["a/one.safetensors", "b/one.gguf"]
    assert lora_matches("three", catalog) == []
