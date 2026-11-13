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
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Protocol

from imaginer.backends.base import (
    BackendHealth,
    EditRequest,
    GenProgress,
    GenRequest,
    ImageResult,
    LoadingCallback,
    ProgressCallback,
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
from imaginer.core.config import (
    DEFAULT_MASK_INSTRUCTION,
    DEFAULT_RGBA_PROMPT_TEMPLATE,
    BackendConfig,
    SdCppBackendConfig,
)
from imaginer.core.logging import get_logger
from imaginer.domain.enums import BackendStatus, Capability
from imaginer.domain.errors import EngineError, ImaginerError, InvalidRequestError
from imaginer.image_models import recipes
from imaginer.services.media import inspect_image_bytes

log = get_logger(__name__)

HEALTH_TIMEOUT_S = 5.0
CANCEL_TIMEOUT_S = 5.0
SUPPORTED: frozenset[Capability] = frozenset(
    {Capability.T2I, Capability.EDIT, Capability.MULTI_REF, Capability.RGBA, Capability.LORA}
)
UNLISTED = "start sd-server with --lora-model-dir, launch.lora_model_dir here"
LIVE_FIELDS = frozenset({"presets", "defaults"})  # read per request: no new adapter
LEGACY_SIZE_MULTIPLE = 32  # Qwen-Image-2.1's, for an engine with no family


def _lora_dir_of(config: SdCppBackendConfig) -> object:
    return config.launch.lora_model_dir if config.launch is not None else None


def family_capabilities(config: SdCppBackendConfig) -> frozenset[Capability]:
    """What the engine's model family can do at most (all of it, without a family): with
    the files it launches, no edits from a GGUF text encoder missing its vision projector."""
    family = recipes.family(config.family)
    if family is None:
        return SUPPORTED
    files = (
        {slot: getattr(config.launch, slot, None) for slot in recipes.SLOT_IDS}
        if config.launch is not None
        else None
    )
    return SUPPORTED & family.capabilities(files)


def expected_capabilities(config: SdCppBackendConfig) -> set[Capability]:
    """What a started sd-server will report once loaded, before it can be probed: what a
    command-line launch reports for its model's family, narrowed by
    ``declared_capabilities``."""
    declared = config.declared_capabilities
    supported = family_capabilities(config)
    return set(supported & declared if declared is not None else supported)


def ref_limit(config: SdCppBackendConfig) -> int:
    """The most input images an edit takes: the config's limit, or fewer if the model family
    handles fewer well (FLUX.1 Kontext edits one image)."""
    family = recipes.family(config.family)
    if family is not None and family.max_refs is not None:
        return min(config.max_ref_images, family.max_refs)
    return config.max_ref_images


def size_multiple(config: SdCppBackendConfig | None) -> int:
    """What sizes snap to: the config's, else its model family's, else Qwen-Image-2.1's."""
    if config is None:
        return LEGACY_SIZE_MULTIPLE
    if config.size_multiple is not None:
        return config.size_multiple
    family = recipes.family(config.family)
    return family.size_multiple if family is not None else LEGACY_SIZE_MULTIPLE


def mask_instruction(config: SdCppBackendConfig) -> str | None:
    """How an edit's mask is explained to the model; None when its family takes no mask."""
    if config.mask_instruction is not None:
        return config.mask_instruction
    family = recipes.family(config.family)
    return DEFAULT_MASK_INSTRUCTION if family is None else family.mask


Face = Literal["main", "edit"]


def mask_face(config: SdCppBackendConfig) -> Face | None:
    """Which card an edit with a mask runs on an engine with an edit card (None without
    one): the edit card, unless it takes no mask and the model card edits with one, as
    Qwen-Image-2.1 does next to FLUX.1 Kontext."""
    view = config.edit_view
    if view is None:
        return None
    if (
        mask_instruction(view) is None
        and mask_instruction(config) is not None
        and Capability.EDIT in expected_capabilities(config)
    ):
        return "main"
    return "edit"


class CardSwitcher(Protocol):
    """Switches an engine between its model card and its edit card (EngineService)."""

    def loading(self, engine: str) -> str | None:
        """The card the engine is switching to, while it does."""
        ...

    def card_name(self, card_id: str | None) -> str | None: ...

    async def ensure(
        self,
        engine: str,
        face: Face,
        config: SdCppBackendConfig,
        *,
        notify: LoadingCallback | None,
        probe: Callable[[], Awaitable[BackendHealth]],
    ) -> None:
        """Return once the engine runs ``config``'s files, restarting it with them first if
        it runs the other card's."""
        ...


@dataclass(slots=True)
class _ServerState:
    """What the running sd-server reported at its last probe. An engine with an edit card
    has two faces (one per card) on one server, so they share it."""

    probed: set[Capability] = field(default_factory=set)
    model: str = "unknown"
    max_side: int = 4096
    preview_modes: set[str] = field(default_factory=set)
    loras: dict[str, dict[str, Any]] | None = None  # path -> entry; None: not listed
    recipes_read: bool = False


@dataclass(slots=True)
class _LoadingLabel:
    """A card switch's "loading" label on the job. sd-server says it is ready before it has
    read the weights, and reads them as it starts the first image, so the label stays until
    that image's first step takes its place. A model that reports no steps drops it as soon
    as the engine is ready."""

    on_loading: LoadingCallback | None
    held: bool = False  # the switch is over, and the label waits for a step

    async def notify(self, card: str | None) -> None:
        if self.on_loading is None:
            return
        if card is None:
            self.held = True
        else:
            await self.on_loading(card)

    def until_step(self, on_progress: ProgressCallback) -> ProgressCallback:
        async def progress(step: GenProgress) -> None:
            self.held = False  # the step is the job's progress now
            await on_progress(step)

        return progress

    async def drop(self) -> None:
        if self.held and self.on_loading is not None:
            self.held = False
            await self.on_loading(None)


class SdCppBackend:
    """One sd-server. With an edit card (``config.edit_view``) it has a second face,
    ``edit_face``: the same server as it runs the edit card, with that card's family,
    defaults and presets. Each face makes sure the server runs its card before it makes an
    image, so an edit switches the engine to the edit card and text-to-image switches back."""

    lora_words: WordsSource | None = None  # the LoRA library's trigger words, if it has any

    def __init__(
        self,
        name: str,
        config: SdCppBackendConfig,
        *,
        transport: HttpTransport | None = None,
        server: _ServerState | None = None,
        face: Face | None = None,
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
        self._family = recipes.family(config.family)
        self._declared = frozenset(expected_capabilities(config))
        self._server = server or _ServerState()
        self.switcher: CardSwitcher | None = None
        self.edit_face: SdCppBackend | None = None
        self._face: Face | None = face
        check_sigmas(config.defaults, config.presets)
        if config.edit_view is not None and face is None:
            self._face = "main"
            self.edit_face = SdCppBackend(
                name, config.edit_view, transport=self._transport, server=self._server, face="edit"
            )

    @property
    def face(self) -> Face | None:
        """Which card this face runs (none: the engine has no edit card)."""
        return self._face

    @property
    def config(self) -> SdCppBackendConfig:
        return self._config

    @property
    def declared(self) -> frozenset[Capability]:
        """What this face can do at most: its family's, narrowed by declared_capabilities."""
        return self._declared

    @property
    def loading(self) -> str | None:
        """The card the engine is switching to, while it switches."""
        if self._face is None or self.switcher is None:
            return None
        return self.switcher.loading(self.name)

    @property
    def takes_mask(self) -> bool:
        return self._mask_instruction() is not None

    @property
    def size_multiple(self) -> int:
        return size_multiple(self._config)

    @property
    def family(self) -> str | None:
        return self._family.id if self._family is not None else None

    @property
    def model_card(self) -> str | None:
        """The model card the engine runs, if it runs one (recorded with each image)."""
        return self._config.model_card

    def _rgba_template(self) -> str:
        if self._config.rgba_prompt_template is not None:
            return self._config.rgba_prompt_template
        if self._family is not None and self._family.rgba:
            return self._family.rgba
        return DEFAULT_RGBA_PROMPT_TEMPLATE

    def _takes(self, feature: str) -> bool:
        """Whether the engine's family has a recipe feature (any, without a family)."""
        return self._family is None or feature in self._family.features

    def _mask_instruction(self) -> str | None:
        return mask_instruction(self._config)

    def update_config(self, config: BackendConfig) -> bool:
        """Take new presets or defaults as they are: requests read them. False when
        anything else changed, which needs a new adapter (for the edit face too)."""
        if not isinstance(config, SdCppBackendConfig) or not self._takes_config(config):
            return False
        if (config.edit_view is None) != (self.edit_face is None):
            return False
        if self.edit_face is not None:
            assert config.edit_view is not None
            if not self.edit_face._takes_config(config.edit_view):
                return False
            self.edit_face._config = config.edit_view
        self._config = config
        return True

    def _takes_config(self, config: SdCppBackendConfig) -> bool:
        try:
            check_sigmas(config.defaults, config.presets)
        except ValueError:
            return False
        live = set(LIVE_FIELDS)
        return config.model_dump(exclude=live | {"launch"}) == self._config.model_dump(
            exclude=live | {"launch"}
        ) and _lora_dir_of(config) == _lora_dir_of(self._config)

    @property
    def model(self) -> str:
        return self._server.model

    @property
    def max_ref_images(self) -> int:
        return ref_limit(self._config)

    @property
    def available_loras(self) -> list[str] | None:
        """The LoRA paths the engine listed at its last probe; None if it lists none."""
        loras = self._server.loras
        return sorted(loras) if loras is not None else None

    def capabilities(self) -> set[Capability]:
        if self.loading is not None:
            # The engine is down while it switches cards: say what it will do once loaded,
            # so work waits for it rather than failing.
            return set(self._declared)
        return set(self._declared & self._server.probed)

    async def health(self) -> BackendHealth:
        if (card := self.loading) is not None:
            return BackendHealth(
                status=BackendStatus.DEGRADED, detail=f"loading {card}", probed_at=utcnow()
            )
        return await self.probe()

    async def probe(self) -> BackendHealth:
        """Ask the server how it is, even while it switches cards."""
        started = time.perf_counter()
        server = self._server
        try:
            caps = await self._transport.get_json(
                "/sdcpp/v1/capabilities", timeout_s=HEALTH_TIMEOUT_S, attempts=1
            )
        except ImaginerError as exc:
            server.probed = set()
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
            server.probed = set()
            return self._health(
                BackendStatus.DEGRADED, started, "loaded model cannot generate images"
            )

        probed = {Capability.T2I, Capability.RGBA, Capability.EDIT, Capability.MULTI_REF}
        features = (caps.get("features_by_mode") or {}).get("img_gen") or caps.get("features") or {}
        if runtime.get("origin") != "command_line" and not features.get("ref_images", True):
            probed -= {Capability.EDIT, Capability.MULTI_REF}
        # Servers that take a request's LoRAs list the ones they can apply (maybe none).
        server.loras = _lora_catalog(caps.get("loras"))
        if server.loras is not None and features.get("lora", True):
            probed.add(Capability.LORA)
        server.probed = probed
        server.model = (caps.get("model") or {}).get("name") or server.model
        limits = caps.get("limits") or {}
        server.max_side = int(min(limits.get("max_width", 4096), limits.get("max_height", 4096)))
        server.preview_modes = set(caps.get("preview_modes") or [])
        if not server.recipes_read:
            await self._read_recipes()
        detail = f"model {server.model}"
        if self._config.declared_capabilities is None:
            detail += "; edit/rgba support assumed (set declared_capabilities to restrict)"
        return self._health(BackendStatus.OK, started, detail)

    async def _read_recipes(self) -> None:
        """The engine's own recipes, which describe its build's families better than the
        bundled snapshot. Builds before the recipe API have none; that's fine."""
        self._server.recipes_read = True
        with contextlib.suppress(ImaginerError):
            listed = await self._transport.get_json(
                "/sdcpp/v1/recipes", timeout_s=HEALTH_TIMEOUT_S, attempts=1
            )
            if isinstance(listed, dict):
                taken = recipes.catalog().take_engine_recipes(listed.get("recipes"))
                log.debug("sdcpp.recipes", backend=self.name, recipes=taken)

    async def generate(self, req: GenRequest) -> list[ImageResult]:
        return await self._run(req, [])

    async def edit(self, req: EditRequest) -> list[ImageResult]:
        # Qwen-Image-2.1 takes "separate masks" as one more reference image, said in the prompt.
        instruction = self._mask_instruction()
        if req.mask is not None and instruction is None:
            name = self._family.name if self._family is not None else "its model"
            raise InvalidRequestError(
                f"{self.name} can't edit with a mask: {name} takes reference images only",
                backend=self.name,
            )
        limited = self.max_ref_images < self._config.max_ref_images and self._family is not None
        fitted = await prepare_inputs(
            self._family.name if limited and self._family is not None else self.name,
            req.images,
            req.mask,
            max_inputs=self.max_ref_images,
            max_pixels=default_max_ref_pixels(self._config.defaults, self._config.max_ref_pixels),
        )
        if req.mask is not None and instruction is not None:
            req = replace(req, prompt=mask_prompt(req.prompt, instruction))
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
            size_multiple=self.size_multiple,
            max_side=self._server.max_side,
            base_size=base_size,
            preset=find_preset(self.name, self._config.presets, req.preset),
        )

    async def _engine_loras(self, wanted: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """``wanted`` with each path as the engine lists it; a LoRA named twice (a preset's,
        then the request's) gets its last weight, and one at weight 0 is left out. One
        missing from the last listing asks for a new listing first: the server rescans its
        folder every time."""
        try:
            matched = match_loras(self.name, wanted, self._server.loras, unlisted=UNLISTED)
        except InvalidRequestError:
            await self.probe()
            matched = match_loras(self.name, wanted, self._server.loras, unlisted=UNLISTED)
        return merge_loras(matched)

    def _body(self, req: GenRequest, params: dict[str, Any], refs: list[str]) -> dict[str, Any]:
        prompt = styled_prompt(req, self._rgba_template())
        extra = dict(self._config.defaults.extra)
        sample: dict[str, Any] = _merged(
            extra.pop("sample_params", None) or {}, self._config.defaults.extra_sample
        )
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
        guidance: dict[str, Any] = dict(sample.get("guidance") or {})
        if "cfg_scale" in params:
            guidance["txt_cfg"] = params["cfg_scale"]
        if "guidance" in params:
            guidance["distilled_guidance"] = params["guidance"]
        if refs and "img_cfg_scale" in params and self._takes("img_cfg"):
            guidance["img_cfg"] = params["img_cfg_scale"]  # LLaDA, Fibo 2 have none
        if guidance:
            sample["guidance"] = guidance
        preview = self._config.preview
        if self._server.preview_modes and preview not in self._server.preview_modes:
            preview = "none"
        if self._family is not None and not self._family.previews:
            preview = "none"  # pixel-space models make no previews
        body: dict[str, Any] = {
            **extra,
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
        if "clip_skip" in params:
            body["clip_skip"] = params["clip_skip"]
        if sample:
            body["sample_params"] = sample
        if refs:
            body["ref_images"] = refs
            if params.get("ref_image_args"):
                body["ref_image_args"] = params["ref_image_args"]
        if loras := params.get("loras"):
            body["lora"] = [{"path": x["path"], "multiplier": x["weight"]} for x in loras]
        return body

    async def _run(
        self, req: GenRequest, refs: list[str], base_size: tuple[int, int] | None = None
    ) -> list[ImageResult]:
        label = _LoadingLabel(req.on_loading)
        if self._face is not None and self.switcher is not None:
            # The engine may run the other card: switch it to this face's first.
            await self.switcher.ensure(
                self.name, self._face, self._config, notify=label.notify, probe=self.probe
            )
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
        if body["preview"] == "none" or req.on_progress is None:
            await label.drop()  # no step will take its place
        elif label.held:
            req = replace(req, on_progress=label.until_step(req.on_progress))
        # Hashed while the image is made.
        hashing = asyncio.ensure_future(with_sha256(folder, params.get("loras", [])))
        try:
            results = await self._follow(req, params, body, hashing)
        finally:
            hashing.cancel()
        await label.drop()  # polling saw none of its steps
        return results

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
        terminal = False
        try:
            if req.on_submitted is not None:
                await req.on_submitted(job_id)
            while True:
                await asyncio.sleep(self._config.poll_interval_s)
                job = await self._transport.get_json(f"/sdcpp/v1/jobs/{job_id}")
                status = job.get("status")
                terminal = status in ("completed", "failed", "cancelled")
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
        except (Exception, asyncio.CancelledError):
            # Abandoning polling must free the GPU too. Preserve the original failure
            # if the best-effort cancellation cannot reach the engine.
            if not terminal:
                with contextlib.suppress(Exception):
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


def _merged(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """``over`` on ``base``, nested mappings merged too."""
    out = dict(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merged(out[key], value)
        else:
            out[key] = value
    return out


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
