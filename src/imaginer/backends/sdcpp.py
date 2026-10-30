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
    default_max_ref_pixels,
    mask_prompt,
    prepare_inputs,
    resolve_params,
    sigmas,
    styled_prompt,
)
from imaginer.backends.openai_http.transport import HttpTransport
from imaginer.core.clock import utcnow
from imaginer.core.config import SdCppBackendConfig
from imaginer.core.logging import get_logger
from imaginer.domain.enums import BackendStatus, Capability
from imaginer.domain.errors import EngineError, ImaginerError
from imaginer.services.media import inspect_image_bytes

log = get_logger(__name__)

HEALTH_TIMEOUT_S = 5.0
CANCEL_TIMEOUT_S = 5.0
SUPPORTED: frozenset[Capability] = frozenset(
    {Capability.T2I, Capability.EDIT, Capability.MULTI_REF, Capability.RGBA}
)


def expected_capabilities(config: SdCppBackendConfig) -> set[Capability]:
    """What a started sd-server will report once loaded, before it can be probed: what a
    command-line launch reports, narrowed by ``declared_capabilities``."""
    declared = config.declared_capabilities
    return set(SUPPORTED & declared if declared is not None else SUPPORTED)


class SdCppBackend:
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
        sigmas(config.defaults.sigmas)  # fail fast on an unknown preset name

    @property
    def model(self) -> str:
        return self._model

    @property
    def max_ref_images(self) -> int:
        return self._config.max_ref_images

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
        )

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
        return body

    async def _run(
        self, req: GenRequest, refs: list[str], base_size: tuple[int, int] | None = None
    ) -> list[ImageResult]:
        params = self.resolved_params(req, base_size)
        body = self._body(req, params, refs)
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
