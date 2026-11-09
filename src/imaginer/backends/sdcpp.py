"""stable-diffusion.cpp ``sd-server`` adapter, over its native async API.

Flow: ``POST /sdcpp/v1/img_gen`` (202 + job id), poll ``GET /sdcpp/v1/jobs/{id}`` until
``completed|failed|cancelled``, and ``POST /sdcpp/v1/jobs/{id}/cancel`` if our side is
cancelled. A generating job stops at the engine's next checkpoint.

Verified against sd.cpp ``master-1024-77b72a2`` (examples/server/async_jobs.cpp,
routes_sdcpp.cpp, model_catalog.cpp):

* Each result image carries ``generation_info``, the ``sdcpp.generation/v1`` snapshot, with
  the seed actually used (a decimal string).
* Step progress exists only through previews (``preview`` != ``none``).
* Edit and multi-reference both go through ``ref_images``. For a model loaded from the command
  line the server reports every feature as available, so whether edit really works (it needs
  ``--llm_vision`` with a GGUF text encoder) comes from ``declared_capabilities``. For models
  loaded through a recipe, ``features_by_mode.img_gen.ref_images`` narrows it.
* Transparency is prompt-driven for Qwen-Image-2.1. Alpha survives only in PNG/WebP.
* LoRAs go in the request's ``lora`` list (``path``, ``multiplier``); ``<lora:...>`` prompt
  tags are ignored by the server API. A path must be one the server lists under ``loras`` in
  its capabilities: it only applies LoRAs it found in ``--lora-model-dir`` (or the model
  roots' ``loras/`` folders), and it rescans them on every request. A listed ``path`` is
  relative to that folder, so for an engine launched here the image's record gets each
  LoRA's SHA-256 too. Checked with ``master-1028-37bc88b``: it lists ``{name, path,
  display_path}`` per LoRA, and applies a PEFT-format Qwen-Image-2.1 LoRA
  (``transformer_blocks.N...lora_A.default.weight``). A LoRA whose file names a trigger
  phrase in its metadata gets it added to the prompt (``common.with_triggers``), here too
  only for an engine launched here.
"""

import asyncio
import base64
import contextlib
import time
from dataclasses import replace
from typing import Any

from imaginer.backends.base import (
    BackendHealth,
    EditRequest,
    GenProgress,
    GenRequest,
    ImageResult,
)
from imaginer.backends.common import (
    WordsSource,
    check_sigmas,
    default_max_ref_pixels,
    find_preset,
    mask_prompt,
    match_loras,
    merge_loras,
    prepare_inputs,
    resolve_params,
    sigmas,
    styled_prompt,
    with_sha256,
    with_triggers,
)
from imaginer.backends.openai_http.transport import HttpTransport
from imaginer.core.clock import utcnow
from imaginer.core.config import BackendConfig, SdCppBackendConfig
from imaginer.core.logging import get_logger
from imaginer.domain.enums import BackendStatus, Capability
from imaginer.domain.errors import EngineError, ImaginerError, InvalidRequestError
from imaginer.services.media import inspect_image_bytes

log = get_logger(__name__)

HEALTH_TIMEOUT_S = 5.0
CANCEL_TIMEOUT_S = 5.0
SUPPORTED: frozenset[Capability] = frozenset(
    {Capability.T2I, Capability.EDIT, Capability.MULTI_REF, Capability.RGBA, Capability.LORA}
)
UNLISTED = "start sd-server with --lora-model-dir, launch.lora_model_dir here"
LIVE_FIELDS = frozenset({"presets", "defaults"})  # read per request: no new adapter


def _lora_dir_of(config: SdCppBackendConfig) -> object:
    return config.launch.lora_model_dir if config.launch is not None else None


def expected_capabilities(config: SdCppBackendConfig) -> set[Capability]:
    """What a started sd-server will report once loaded, before it can be probed: what a
    command-line launch reports, narrowed by ``declared_capabilities``."""
    declared = config.declared_capabilities
    return set(SUPPORTED & declared if declared is not None else SUPPORTED)


class SdCppBackend:
    lora_words: WordsSource | None = None  # the LoRA library's trigger words, if it has any

    def __init__(
        self,
        name: str,
        config: SdCppBackendConfig,
        *,
        transport: HttpTransport | None = None,
    ) -> None:
        self.name = name
        self._config = config
        self._transport = transport or HttpTransport(
            name=name,
            base_url=config.base_url,
            timeouts=config.timeouts,
            retry=config.retry,
            api_key=config.api_key(),
        )
        declared = config.declared_capabilities
        self._declared = SUPPORTED & declared if declared is not None else SUPPORTED
        self._probed: set[Capability] = set()
        self._model = "unknown"
        self._max_side = 4096
        self._preview_modes: set[str] = set()
        self._loras: dict[str, dict[str, Any]] | None = None  # path -> entry; None: not listed
        check_sigmas(config.defaults, config.presets)

    def update_config(self, config: BackendConfig) -> bool:
        """Take new presets or defaults as they are: requests read them. False when
        anything else changed, which needs a new adapter."""
        if not isinstance(config, SdCppBackendConfig):
            return False
        try:
            check_sigmas(config.defaults, config.presets)
        except ValueError:
            return False
        live = set(LIVE_FIELDS)
        if config.model_dump(exclude=live | {"launch"}) != self._config.model_dump(
            exclude=live | {"launch"}
        ) or _lora_dir_of(config) != _lora_dir_of(self._config):
            return False
        self._config = config
        return True

    @property
    def model(self) -> str:
        return self._model

    @property
    def max_ref_images(self) -> int:
        return self._config.max_ref_images

    @property
    def available_loras(self) -> list[str] | None:
        """The LoRA paths the engine listed at its last probe; None if it lists none."""
        return sorted(self._loras) if self._loras is not None else None

    def capabilities(self) -> set[Capability]:
        return set(self._declared & self._probed)

    async def health(self) -> BackendHealth:
        started = time.perf_counter()
        try:
            caps = await self._transport.get_json(
                "/sdcpp/v1/capabilities", timeout_s=HEALTH_TIMEOUT_S, attempts=1
            )
        except ImaginerError as exc:
            self._probed = set()
            return self._health(BackendStatus.DOWN, started, exc.detail)
        if not isinstance(caps, dict):
            return self._health(BackendStatus.DEGRADED, started, "unexpected capabilities response")

        runtime = caps.get("runtime") or {}
        state = runtime.get("state", "ready")
        if state != "ready":
            detail = "model is loading" if state == "loading" else f"model state: {state}"
            if runtime.get("error"):
                detail += f" ({runtime['error']})"
            return self._health(BackendStatus.DEGRADED, started, detail)
        if "img_gen" not in (caps.get("supported_modes") or ["img_gen"]):
            self._probed = set()
            return self._health(
                BackendStatus.DEGRADED, started, "loaded model cannot generate images"
            )

        probed = {Capability.T2I, Capability.RGBA, Capability.EDIT, Capability.MULTI_REF}
        features = (caps.get("features_by_mode") or {}).get("img_gen") or caps.get("features") or {}
        if runtime.get("origin") != "command_line" and not features.get("ref_images", True):
            probed -= {Capability.EDIT, Capability.MULTI_REF}
        # Servers that take a request's LoRAs list the ones they can apply (maybe none).
        self._loras = _lora_catalog(caps.get("loras"))
        if self._loras is not None and features.get("lora", True):
            probed.add(Capability.LORA)
        self._probed = probed
        self._model = (caps.get("model") or {}).get("name") or self._model
        limits = caps.get("limits") or {}
        self._max_side = int(min(limits.get("max_width", 4096), limits.get("max_height", 4096)))
        self._preview_modes = set(caps.get("preview_modes") or [])
        detail = f"model {self._model}"
        if self._config.declared_capabilities is None:
            detail += "; edit/rgba support assumed (set declared_capabilities to restrict)"
        return self._health(BackendStatus.OK, started, detail)

    async def generate(self, req: GenRequest) -> list[ImageResult]:
        return await self._run(req, [])

    async def edit(self, req: EditRequest) -> list[ImageResult]:
        # Qwen-Image-2.1 takes "separate masks" as one more reference image, said in the prompt.
        fitted = await prepare_inputs(
            self.name,
            req.images,
            req.mask,
            max_inputs=self._config.max_ref_images,
            max_pixels=default_max_ref_pixels(self._config.defaults, self._config.max_ref_pixels),
        )
        if req.mask is not None:
            req = replace(req, prompt=mask_prompt(req.prompt, self._config.mask_instruction))
        base = (fitted[0].width, fitted[0].height)
        refs = [_data_url(f.data, f.mime) for f in fitted]
        return await self._run(req, refs, base_size=base)

    async def aclose(self) -> None:
        await self._transport.aclose()

    # --- internals ------------------------------------------------------------

    def resolved_params(
        self, req: GenRequest, base_size: tuple[int, int] | None = None
    ) -> dict[str, Any]:
        return resolve_params(
            req,
            self._config.defaults,
            size_multiple=self._config.size_multiple,
            max_side=self._max_side,
            base_size=base_size,
            preset=find_preset(self.name, self._config.presets, req.preset),
        )

    async def _engine_loras(self, wanted: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """``wanted`` with each path as the engine lists it; a LoRA named twice (a preset's,
        then the request's) gets its last weight, and one at weight 0 is left out. One
        missing from the last listing asks for a new listing first: the server rescans its
        folder every time."""
        try:
            matched = match_loras(self.name, wanted, self._loras, unlisted=UNLISTED)
        except InvalidRequestError:
            await self.health()
            matched = match_loras(self.name, wanted, self._loras, unlisted=UNLISTED)
        return merge_loras(matched)

    def _body(self, req: GenRequest, params: dict[str, Any], refs: list[str]) -> dict[str, Any]:
        prompt = styled_prompt(req, self._config.rgba_prompt_template)
        sample: dict[str, Any] = {}
        if "steps" in params:
            sample["sample_steps"] = params["steps"]
        if "sampler" in params:
            sample["sample_method"] = params["sampler"]
        if "scheduler" in params:
            sample["scheduler"] = params["scheduler"]
        if "flow_shift" in params:
            sample["flow_shift"] = params["flow_shift"]
        if (schedule := sigmas(params.get("sigmas"))) is not None:
            sample["custom_sigmas"] = schedule
        if "cfg_scale" in params:
            sample["guidance"] = {"txt_cfg": params["cfg_scale"]}
        preview = self._config.preview
        if self._preview_modes and preview not in self._preview_modes:
            preview = "none"
        body: dict[str, Any] = {
            **self._config.defaults.extra,
            "prompt": prompt,
            "negative_prompt": params.get("negative_prompt", ""),
            "width": params["width"],
            "height": params["height"],
            "seed": req.seed,
            "batch_count": 1,
            "output_format": params["output_format"],
            "preview": preview,
            "preview_interval": self._config.preview_interval,
        }
        if sample:
            body["sample_params"] = sample
        if refs:
            body["ref_images"] = refs
        if loras := params.get("loras"):
            body["lora"] = [{"path": x["path"], "multiplier": x["weight"]} for x in loras]
        return body

    async def _run(
        self, req: GenRequest, refs: list[str], base_size: tuple[int, int] | None = None
    ) -> list[ImageResult]:
        params = self.resolved_params(req, base_size)
        # Only an engine launched here has its LoRA folder on this machine, the folder its
        # listed paths are relative to: their trigger phrases and hashes come from there.
        launch = self._config.launch
        folder = launch.lora_model_dir if launch is not None else None
        if params.get("loras"):
            params["loras"] = await self._engine_loras(params["loras"]) or None
            params = {k: v for k, v in params.items() if v is not None}
        if params.get("loras"):
            prompt, params["loras"] = await with_triggers(
                folder, req.prompt, params["loras"], self.lora_words
            )
            req = replace(req, prompt=prompt)
        body = self._body(req, params, refs)
        # Hashed while the image is made.
        hashing = asyncio.ensure_future(with_sha256(folder, params.get("loras", [])))
        try:
            return await self._follow(req, params, body, hashing)
        finally:
            hashing.cancel()

    async def _follow(
        self,
        req: GenRequest,
        params: dict[str, Any],
        body: dict[str, Any],
        hashing: asyncio.Future[list[dict[str, Any]]],
    ) -> list[ImageResult]:
        job_id = await self._submit(body)
        log.info("sdcpp.submitted", backend=self.name, engine_job_id=job_id, seed=req.seed)
        last_step = -1
        try:
            if req.on_submitted is not None:
                await req.on_submitted(job_id)
            while True:
                await asyncio.sleep(self._config.poll_interval_s)
                job = await self._transport.get_json(f"/sdcpp/v1/jobs/{job_id}")
                status = job.get("status")
                preview = job.get("preview") or {}
                step = preview.get("step")
                if req.on_progress is not None and isinstance(step, int) and step != last_step:
                    last_step = step
                    total = int(preview.get("total_steps") or 0)
                    await req.on_progress(GenProgress(step=step, total_steps=total))
                if status == "completed":
                    if loras := await hashing:
                        params = {**params, "loras": loras}
                    return await self._results(job, req, params, job_id)
                if status in ("failed", "cancelled"):
                    error = job.get("error") or {}
                    raise EngineError(
                        f"{self.name} job {status}: {error.get('message') or error.get('code')}",
                        backend=self.name,
                        engine_job_id=job_id,
                    )
        except asyncio.CancelledError:
            # Free the GPU: the engine would otherwise finish an image nobody wants.
            await asyncio.shield(self._cancel_remote(job_id))
            raise

    async def _submit(self, body: dict[str, Any]) -> str:
        """Submit a job. If we are cancelled while the request is in flight, the engine may
        already have queued it: wait briefly for its id and cancel it, so no orphan runs."""
        submit = asyncio.ensure_future(self._transport.post_json("/sdcpp/v1/img_gen", body))
        try:
            submitted = await asyncio.shield(submit)
        except asyncio.CancelledError:
            with contextlib.suppress(Exception):
                late = await asyncio.wait_for(submit, CANCEL_TIMEOUT_S)
                await self._cancel_remote(str(late["id"]))
            raise
        return str(submitted["id"])

    async def _cancel_remote(self, job_id: str) -> None:
        with contextlib.suppress(ImaginerError):
            await self._transport.request(
                "POST",
                f"/sdcpp/v1/jobs/{job_id}/cancel",
                timeout_s=CANCEL_TIMEOUT_S,
                attempts=1,
                accept_status=frozenset({404, 409, 410}),
            )
            log.info("sdcpp.cancelled", backend=self.name, engine_job_id=job_id)

    async def _results(
        self, job: dict[str, Any], req: GenRequest, params: dict[str, Any], job_id: str
    ) -> list[ImageResult]:
        images = (job.get("result") or {}).get("images") or []
        if not images:
            raise EngineError(f"{self.name} job {job_id} completed without images")
        results = []
        for item in images:
            data = base64.b64decode(item["b64_json"])
            info = await asyncio.to_thread(inspect_image_bytes, data)
            generation = item.get("generation_info") or {}
            seed = generation.get("seed", req.seed) if isinstance(generation, dict) else req.seed
            results.append(
                ImageResult(
                    data=data,
                    mime=info.mime,
                    width=info.width,
                    height=info.height,
                    has_alpha=info.has_alpha,
                    seed=int(seed),
                    params=params,
                    engine_meta={"engine_job_id": job_id, "generation_info": generation},
                )
            )
        return results

    def _health(self, status: BackendStatus, started: float, detail: str) -> BackendHealth:
        return BackendHealth(
            status=status,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
            detail=detail,
            probed_at=utcnow(),
        )


def _data_url(data: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def _lora_catalog(listed: object) -> dict[str, dict[str, Any]] | None:
    """The capabilities' ``loras`` by path; None when the server sends no list."""
    if not isinstance(listed, list):
        return None
    return {
        str(entry["path"]): entry
        for entry in listed
        if isinstance(entry, dict) and isinstance(entry.get("path"), str) and entry["path"]
    }
