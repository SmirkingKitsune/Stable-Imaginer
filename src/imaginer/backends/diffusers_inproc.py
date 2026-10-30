"""In-process Qwen-Image-2.1 through Hugging Face Diffusers (``pip install ".[diffusers]"``).

The reference implementation for edit and RGBA output, behind the same ``ImageBackend``
protocol as sd-server. This module imports without torch: torch, diffusers and
transformers are imported only by ``TorchLoader`` when the environment is checked or the
pipeline is loaded.

* **Loading** is lazy: the model loads on first use, or at startup with ``preload: true``.
  Loading can take minutes, and downloads the weights to the Hugging Face cache the first time.
* **Inference** runs in a worker thread. The job manager's per-backend semaphore keeps it to
  one job at a time per GPU.
* **Cancellation:** ``callback_on_step_end`` runs after every denoising step and raises once the
  job is cancelled, so the GPU stops within one step. The VAE decode is skipped.
* **Seeds** use a CPU ``torch.Generator`` to improve reproducibility. Identical output across
  different hardware and software versions is not guaranteed.
* **Guidance:** Qwen-Image-2.1 samples without guidance (``true_cfg_scale`` 1.0, the default).
  A negative prompt is only sent when ``cfg_scale`` is above 1.
"""

import asyncio
import concurrent.futures
import contextlib
import importlib.util
import io
import threading
import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass, replace
from functools import partial
from typing import Any, Protocol

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
from imaginer.core.clock import utcnow
from imaginer.core.config import DiffusersBackendConfig
from imaginer.core.logging import get_logger
from imaginer.domain.enums import BackendStatus, Capability
from imaginer.domain.errors import EngineError, EngineUnavailableError
from imaginer.services.media import FittedImage, inspect_image_bytes

log = get_logger(__name__)

PIPELINE_CLASS = "QwenImage21Pipeline"
REQUIRED_MODULES = ("torch", "diffusers", "transformers", "accelerate")
ENV_CHECK_TIMEOUT_S = 4.0
MAX_SIDE = 4096
SUPPORTED: frozenset[Capability] = frozenset(
    {Capability.T2I, Capability.EDIT, Capability.MULTI_REF, Capability.RGBA}
)
IGNORED_PARAMS = ("sampler", "scheduler", "flow_shift")  # fixed by the pipeline's scheduler


def missing_dependencies() -> list[str]:
    """Optional packages that are not installed (checked without importing them)."""
    return [name for name in REQUIRED_MODULES if importlib.util.find_spec(name) is None]


@dataclass(frozen=True, kw_only=True)
class Runtime:
    """A loaded pipeline plus the torch bits the adapter needs, behind plain callables."""

    pipe: Callable[..., Any]
    generator: Callable[[int], Any]  # seed -> torch.Generator
    inference: Callable[[], AbstractContextManager[Any]]  # e.g. torch.inference_mode
    device: str
    close: Callable[[], None] = lambda: None


class PipelineLoader(Protocol):
    def check(self, config: DiffusersBackendConfig) -> str | None:
        """None if the environment can run the pipeline, else why not. Blocking."""
        ...

    def load(self, config: DiffusersBackendConfig) -> Runtime:
        """Load the pipeline onto its device. Blocking, slow."""
        ...


class TorchLoader:
    def check(self, config: DiffusersBackendConfig) -> str | None:
        import diffusers
        import torch

        if not hasattr(diffusers, PIPELINE_CLASS):
            return f"diffusers {diffusers.__version__} has no {PIPELINE_CLASS}; need >= 0.42.1"
        if config.device.startswith("cuda"):
            if not torch.cuda.is_available():
                return (
                    f"torch {torch.__version__} cannot use CUDA (CPU-only build or no driver); "
                    "install a CUDA build, see README"
                )
            index = int(config.device.partition(":")[2] or 0)
            if index >= torch.cuda.device_count():
                return f"{config.device} not found ({torch.cuda.device_count()} CUDA device(s))"
        return None

    def load(self, config: DiffusersBackendConfig) -> Runtime:
        import diffusers
        import torch

        pipeline_class = getattr(diffusers, PIPELINE_CLASS)
        pipe = pipeline_class.from_pretrained(
            config.model_id, torch_dtype=getattr(torch, config.dtype)
        )
        if config.cpu_offload:
            pipe.enable_model_cpu_offload(device=config.device)
        else:
            pipe.to(config.device)
        pipe.set_progress_bar_config(disable=True)

        def generator(seed: int) -> Any:
            return torch.Generator(device="cpu").manual_seed(seed)

        def close() -> None:
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return Runtime(
            pipe=pipe,
            generator=generator,
            inference=torch.inference_mode,
            device=config.device,
            close=close,
        )


class _Interrupted(Exception):
    """Raised from the step callback to stop a cancelled generation."""


class DiffusersBackend:
    def __init__(
        self,
        name: str,
        config: DiffusersBackendConfig,
        *,
        loader: PipelineLoader | None = None,
    ) -> None:
        self.name = name
        self._config = config
        self._loader = loader or TorchLoader()
        declared = config.declared_capabilities
        self._declared = SUPPORTED & declared if declared is not None else SUPPORTED
        self._probed: set[Capability] = set()
        self._env_check: asyncio.Task[str | None] | None = None
        self._runtime: Runtime | None = None
        self._loading: asyncio.Task[Runtime] | None = None
        self._load_error: str | None = None
        sigmas(config.defaults.sigmas)  # fail fast on an unknown preset name
        ignored = [k for k in IGNORED_PARAMS if getattr(config.defaults, k) is not None]
        if ignored:
            log.warning("diffusers.params_ignored", backend=name, params=ignored)

    @property
    def model(self) -> str:
        return self._config.model_id

    @property
    def max_ref_images(self) -> int:
        return self._config.max_ref_images

    def capabilities(self) -> set[Capability]:
        return set(self._declared & self._probed)

    # --- health and loading -----------------------------------------------------

    async def health(self) -> BackendHealth:
        started = time.perf_counter()
        if self._env_check is None:
            self._env_check = asyncio.create_task(
                asyncio.to_thread(self._loader.check, self._config)
            )
        try:
            problem = await asyncio.wait_for(asyncio.shield(self._env_check), ENV_CHECK_TIMEOUT_S)
        except TimeoutError:
            return self._health(BackendStatus.DEGRADED, started, "importing torch and diffusers")
        except Exception as exc:
            problem = f"environment check failed: {exc!r}"
        if problem:
            self._probed = set()
            return self._health(BackendStatus.DOWN, started, problem)
        self._probed = set(SUPPORTED)
        if self._load_error is not None:
            # Like an unloaded lazy backend, this can accept a request. Reporting DOWN
            # would make the registry block the request before _start_loading can retry.
            return self._health(
                BackendStatus.OK,
                started,
                f"last load failed: {self._load_error}; next request retries loading",
            )
        if self._runtime is not None:
            detail = f"{self.model} loaded on {self._runtime.device} ({self._config.dtype})"
        elif self._loading is not None:
            detail = f"loading {self.model} (requests wait for it)"
        else:
            detail = f"{self.model} loads on first use"
            if self._config.preload:
                self._start_loading()
                detail = f"loading {self.model} (requests wait for it)"
        return self._health(BackendStatus.OK, started, detail)

    def _start_loading(self) -> asyncio.Task[Runtime]:
        if self._loading is None or (self._loading.done() and self._runtime is None):
            self._load_error = None
            self._loading = asyncio.create_task(self._load(), name=f"load-{self.name}")
            # Preloading (or a cancelled waiter) may leave nobody awaiting this task.
            # Retrieve its exception without changing what future awaiters receive.
            self._loading.add_done_callback(
                lambda task: None if task.cancelled() else task.exception()
            )
        return self._loading

    async def _load(self) -> Runtime:
        log.info(
            "diffusers.loading", backend=self.name, model=self.model, device=self._config.device
        )
        started = time.perf_counter()
        try:
            runtime = await asyncio.to_thread(self._loader.load, self._config)
        except Exception as exc:
            self._load_error = str(exc) or repr(exc)
            log.exception("diffusers.load_failed", backend=self.name)
            raise EngineUnavailableError(
                f"{self.name} could not load {self.model}: {self._load_error}", backend=self.name
            ) from exc
        self._runtime = runtime
        log.info(
            "diffusers.loaded", backend=self.name, seconds=round(time.perf_counter() - started, 1)
        )
        return runtime

    async def _ensure_loaded(self) -> Runtime:
        if self._runtime is not None:
            return self._runtime
        # Shielded: a cancelled job must not abort a load that later jobs will need.
        return await asyncio.shield(self._start_loading())

    async def aclose(self) -> None:
        if self._loading is not None and not self._loading.done():
            with contextlib.suppress(BaseException):
                await self._loading  # a load thread cannot be interrupted; let it finish
        runtime, self._runtime = self._runtime, None
        if runtime is not None:
            await asyncio.to_thread(runtime.close)

    # --- generation -------------------------------------------------------------

    async def generate(self, req: GenRequest) -> list[ImageResult]:
        return await self._run(req, [], None)

    async def edit(self, req: EditRequest) -> list[ImageResult]:
        fitted = await prepare_inputs(
            self.name,
            req.images,
            req.mask,
            max_inputs=self._config.max_ref_images,
            max_pixels=default_max_ref_pixels(self._config.defaults, self._config.max_ref_pixels),
        )
        if req.mask is not None:
            req = replace(req, prompt=mask_prompt(req.prompt, self._config.mask_instruction))
        return await self._run(req, fitted, (fitted[0].width, fitted[0].height))

    def resolved_params(
        self, req: GenRequest, base_size: tuple[int, int] | None = None
    ) -> dict[str, Any]:
        params = resolve_params(
            req,
            self._config.defaults,
            size_multiple=self._config.size_multiple,
            max_side=MAX_SIDE,
            base_size=base_size,
        )
        for key in IGNORED_PARAMS:
            params.pop(key, None)
        params.setdefault("steps", 40)
        params.setdefault("cfg_scale", 1.0)
        if (schedule := sigmas(params.get("sigmas"))) is not None:
            params["steps"] = len(schedule) - int(bool(schedule) and schedule[-1] == 0)
        return params

    async def _run(
        self, req: GenRequest, inputs: list[FittedImage], base_size: tuple[int, int] | None
    ) -> list[ImageResult]:
        runtime = await self._ensure_loaded()
        params = self.resolved_params(req, base_size)
        prompt = styled_prompt(req, self._config.rgba_prompt_template)
        loop = asyncio.get_running_loop()
        cancelled = threading.Event()
        progress: list[concurrent.futures.Future[None]] = []

        async def report(update: GenProgress) -> None:
            if req.on_progress is not None:
                await req.on_progress(update)

        def on_step(pipe: Any, step: int, timestep: Any, tensors: dict[str, Any]) -> dict[str, Any]:
            # Runs on the worker thread after every denoising step.
            if cancelled.is_set():
                raise _Interrupted
            if req.on_progress is not None:
                update = GenProgress(step=step + 1, total_steps=params["steps"])
                progress.append(asyncio.run_coroutine_threadsafe(report(update), loop))
            return tensors

        call = partial(self._infer, runtime, req.seed, prompt, params, inputs, on_step)
        work = asyncio.ensure_future(asyncio.to_thread(call))
        try:
            images = await asyncio.shield(work)
        except asyncio.CancelledError:
            cancelled.set()
            # Hold the job (and its GPU slot) until the worker thread has really stopped.
            with contextlib.suppress(BaseException):
                await work
            raise
        except _Interrupted as exc:  # pragma: no cover - only after cancellation
            raise asyncio.CancelledError from exc
        except Exception as exc:
            raise EngineError(f"{self.name} generation failed: {exc!r}", backend=self.name) from exc
        finally:
            # Progress updates still in flight must land before the job is finalised.
            await asyncio.gather(
                *(asyncio.wrap_future(f) for f in progress), return_exceptions=True
            )
        return [await asyncio.to_thread(_encode, image, req.seed, params) for image in images]

    @staticmethod
    def _infer(
        runtime: Runtime,
        seed: int,
        prompt: str,
        params: dict[str, Any],
        inputs: list[FittedImage],
        on_step: Callable[..., dict[str, Any]],
    ) -> list[Any]:
        from PIL import Image

        kwargs: dict[str, Any] = {
            "prompt": prompt,
            "width": params["width"],
            "height": params["height"],
            "num_inference_steps": params["steps"],
            "true_cfg_scale": params["cfg_scale"],
            "generator": runtime.generator(seed),
            "callback_on_step_end": on_step,
        }
        if params["cfg_scale"] > 1 and params.get("negative_prompt"):
            kwargs["negative_prompt"] = params["negative_prompt"]
        if (schedule := sigmas(params.get("sigmas"))) is not None:
            # Diffusers wants one sigma per step, without the final 0.0.
            steps = schedule[:-1] if schedule and schedule[-1] == 0 else schedule
            kwargs["sigmas"] = steps
            kwargs["num_inference_steps"] = len(steps)
        if inputs:
            kwargs["image"] = [Image.open(io.BytesIO(f.data)) for f in inputs]
        with runtime.inference():
            output = runtime.pipe(**kwargs)
        return list(output.images)

    def _health(self, status: BackendStatus, started: float, detail: str) -> BackendHealth:
        return BackendHealth(
            status=status,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
            detail=detail,
            probed_at=utcnow(),
        )


def _encode(image: Any, seed: int, params: dict[str, Any]) -> ImageResult:
    """PIL image -> stored bytes. Opaque RGBA becomes RGB; transparency forces PNG."""
    alpha = "A" in image.getbands() and image.getchannel("A").getextrema()[0] < 255
    image = image.convert("RGBA" if alpha else "RGB")
    fmt = "png" if alpha else params.get("output_format", "png")
    buf = io.BytesIO()
    image.save(buf, format={"png": "PNG", "webp": "WEBP", "jpeg": "JPEG"}[fmt])
    data = buf.getvalue()
    info = inspect_image_bytes(data)
    return ImageResult(
        data=data,
        mime=info.mime,
        width=info.width,
        height=info.height,
        has_alpha=info.has_alpha,
        seed=seed,
        params=params,
    )
