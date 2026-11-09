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
* **LoRAs** come from ``lora_dir``: its ``.safetensors`` files, listed under
  ``available_loras`` by their path in it and listed again when a request names one that
  isn't listed. Diffusers loads one as a PEFT adapter the first time an image uses it
  (converting ComfyUI and kohya files), and ``set_adapters`` gives each its weight. The last
  few stay loaded (``LORAS_KEPT``), so the next image with the same LoRAs starts at once.
  Pickled weights are never loaded. The image's record gets each LoRA's SHA-256, and a
  LoRA's trigger words (the library's, else the phrase its file names) are added to the
  prompt (``with_triggers``).
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
from pathlib import Path
from typing import Any, Protocol

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
    lora_file,
    mask_prompt,
    match_loras,
    merge_loras,
    prepare_inputs,
    resolve_params,
    scan_loras,
    sigmas,
    styled_prompt,
    with_sha256,
    with_triggers,
)
from imaginer.core.clock import utcnow
from imaginer.core.config import BackendConfig, DiffusersBackendConfig
from imaginer.core.logging import get_logger
from imaginer.domain.enums import BackendStatus, Capability
from imaginer.domain.errors import EngineError, EngineUnavailableError, InvalidRequestError
from imaginer.services.media import FittedImage, inspect_image_bytes

log = get_logger(__name__)

PIPELINE_CLASS = "QwenImage21Pipeline"
REQUIRED_MODULES = ("torch", "torchvision", "diffusers", "transformers", "accelerate")
ENV_CHECK_TIMEOUT_S = 4.0
MAX_SIDE = 4096
SUPPORTED: frozenset[Capability] = frozenset(
    {Capability.T2I, Capability.EDIT, Capability.MULTI_REF, Capability.RGBA, Capability.LORA}
)
IGNORED_PARAMS = ("sampler", "scheduler", "flow_shift")  # fixed by the pipeline's scheduler
LORA_EXTENSIONS = frozenset({".safetensors"})
LORAS_KEPT = 4  # LoRAs left loaded for later images (more while one image uses more)
UNLISTED = "set lora_dir, a folder of LoRAs"
LIVE_FIELDS = frozenset({"lora_dir", "presets", "defaults"})  # no model reload for these


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

    def lora_problem(self) -> str | None:
        """None if the pipeline can load LoRAs, else why not."""
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

    def lora_problem(self) -> str | None:
        if importlib.util.find_spec("peft") is None:
            return "LoRAs need peft: uv sync --extra diffusers"
        return None

    def load(self, config: DiffusersBackendConfig) -> Runtime:
        import diffusers
        import torch

        pipeline_class = getattr(diffusers, PIPELINE_CLASS)
        pipe = pipeline_class.from_pretrained(config.model_id, dtype=getattr(torch, config.dtype))
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


class LoraLoadError(Exception):
    """Diffusers couldn't load a LoRA into the pipeline (most likely one made for another
    model)."""

    def __init__(self, path: Path, cause: Exception) -> None:
        super().__init__(f"{path.name}: {cause}")
        self.path = path
        self.cause = cause


@dataclass
class _Adapter:
    name: str
    stamp: tuple[int, int]  # the file's size and modification time when it was loaded


class LoraAdapters:
    """The LoRAs loaded into one pipeline, by file. Used on the worker thread, under
    ``lock`` together with the pipeline call, so an image runs with the LoRAs set for it."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self._loaded: dict[Path, _Adapter] = {}  # least recently used first
        self._made = 0

    def apply(self, pipe: Any, loras: list[tuple[Path, float]]) -> None:
        """Make ``loras`` (file, weight) the ones the next image uses; none turns LoRAs off."""
        if not loras:
            if self._loaded:
                pipe.disable_lora()
            return
        wanted = {path for path, _ in loras}
        self._make_room(pipe, wanted)
        names = [self._load(pipe, path) for path, _ in loras]
        pipe.enable_lora()
        pipe.set_adapters(names, adapter_weights=[weight for _, weight in loras])

    def _make_room(self, pipe: Any, wanted: set[Path]) -> None:
        """Unload the least recently used LoRAs this image doesn't use before loading new
        ones, so no more than ``LORAS_KEPT`` (or the ones it uses) are ever loaded."""
        new = sum(1 for path in wanted if path not in self._loaded)
        extra = len(self._loaded) + new - max(LORAS_KEPT, len(wanted))
        for path in list(self._loaded):
            if extra <= 0:
                break
            if path not in wanted:
                self._unload(pipe, path)
                extra -= 1

    def _load(self, pipe: Any, path: Path) -> str:
        stat = path.stat()
        stamp = (stat.st_size, stat.st_mtime_ns)
        adapter = self._loaded.pop(path, None)
        if adapter is not None and adapter.stamp == stamp:
            self._loaded[path] = adapter  # now the most recently used
            return adapter.name
        if adapter is not None:  # the file changed: load it again
            self._loaded[path] = adapter
            self._unload(pipe, path)
        self._made += 1
        name = f"lora_{self._made}"
        try:
            pipe.load_lora_weights(str(path.parent), weight_name=path.name, adapter_name=name)
        except Exception as exc:
            with contextlib.suppress(Exception):
                pipe.delete_adapters(name)
            raise LoraLoadError(path, exc) from exc
        self._loaded[path] = _Adapter(name, stamp)
        return name

    def _unload(self, pipe: Any, path: Path) -> None:
        adapter = self._loaded.pop(path)
        with contextlib.suppress(Exception):
            pipe.delete_adapters(adapter.name)


class DiffusersBackend:
    lora_words: WordsSource | None = None  # the LoRA library's trigger words, if it has any

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
        self._loras: dict[str, dict[str, Any]] | None = None  # lora_dir's; None: not listed
        self._adapters = LoraAdapters()
        check_sigmas(config.defaults, config.presets)
        ignored = [k for k in IGNORED_PARAMS if getattr(config.defaults, k) is not None]
        if ignored:
            log.warning("diffusers.params_ignored", backend=name, params=ignored)

    @property
    def model(self) -> str:
        return self._config.model_id

    @property
    def max_ref_images(self) -> int:
        return self._config.max_ref_images

    @property
    def available_loras(self) -> list[str] | None:
        """The LoRAs in ``lora_dir`` at the last probe; None without a folder they can come
        from."""
        return sorted(self._loras) if self._loras is not None else None

    def capabilities(self) -> set[Capability]:
        return set(self._declared & self._probed)

    def update_config(self, config: BackendConfig) -> bool:
        """Take a new LoRA folder, presets or defaults without reloading the model. False
        when anything else changed: that needs a new adapter."""
        if not isinstance(config, DiffusersBackendConfig):
            return False
        live = set(LIVE_FIELDS)
        if config.model_dump(exclude=live) != self._config.model_dump(exclude=live):
            return False
        self._config = config
        self._loras = None  # listed again at the next probe
        self._probed.discard(Capability.LORA)
        return True

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
        lora_problem = await self._list_loras()
        self._probed = set(SUPPORTED) - ({Capability.LORA} if self._loras is None else set())
        if self._load_error is not None:
            # Like an unloaded lazy backend, this can accept a request. Reporting DOWN
            # would make the registry block the request before _start_loading can retry.
            return self._health(
                BackendStatus.OK,
                started,
                f"last load failed: {self._load_error}; next request retries loading",
                lora_problem,
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
        return self._health(BackendStatus.OK, started, detail, lora_problem)

    async def _list_loras(self) -> str | None:
        """List ``lora_dir``; returns why its LoRAs can't be applied, if they can't."""
        folder = self._config.lora_dir
        problem = self._loader.lora_problem() if folder is not None else None
        catalog = None
        if folder is not None and problem is None:
            catalog = await asyncio.to_thread(scan_loras, folder, LORA_EXTENSIONS)
            if catalog is None:
                problem = f"LoRA folder {folder} not found"
        self._loras = catalog
        return problem

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
        self._runtime, self._adapters = runtime, LoraAdapters()
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
        self._adapters = LoraAdapters()
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
            preset=find_preset(self.name, self._config.presets, req.preset),
        )
        for key in IGNORED_PARAMS:
            params.pop(key, None)
        params.setdefault("steps", 40)
        params.setdefault("cfg_scale", 1.0)
        if (schedule := sigmas(params.get("sigmas"))) is not None:
            params["steps"] = len(schedule) - int(bool(schedule) and schedule[-1] == 0)
        return params

    async def _engine_loras(self, wanted: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """``wanted`` with each path as listed, merged (``merge_loras``). One missing from
        the last listing lists the folder again first."""
        try:
            matched = match_loras(self.name, wanted, self._loras, unlisted=UNLISTED)
        except InvalidRequestError:
            await self._list_loras()
            matched = match_loras(self.name, wanted, self._loras, unlisted=UNLISTED)
        return merge_loras(matched)

    def _lora_files(self, loras: list[dict[str, Any]]) -> list[tuple[Path, float]]:
        folder = self._config.lora_dir
        files = []
        for lora in loras:
            path = lora_file(folder, lora["path"]) if folder is not None else None
            if path is None:
                raise InvalidRequestError(
                    f"{self.name}: LoRA '{lora['path']}' is no longer in {folder}",
                    backend=self.name,
                    lora=lora["path"],
                )
            files.append((path, float(lora["weight"])))
        return files

    async def _run(
        self, req: GenRequest, inputs: list[FittedImage], base_size: tuple[int, int] | None
    ) -> list[ImageResult]:
        params = self.resolved_params(req, base_size)
        if params.get("loras"):  # checked before a model load that can take minutes
            params["loras"] = await self._engine_loras(params["loras"]) or None
            params = {k: v for k, v in params.items() if v is not None}
        loras = self._lora_files(params.get("loras", []))
        if loras:
            folder = self._config.lora_dir
            prompt, params["loras"] = await with_triggers(
                folder, req.prompt, params["loras"], self.lora_words
            )
            req = replace(req, prompt=prompt)
        runtime = await self._ensure_loaded()
        hashing = asyncio.ensure_future(with_sha256(self._config.lora_dir, params.get("loras", [])))
        try:
            images = await self._images(runtime, req, params, inputs, loras)
            if loras:
                params = {**params, "loras": await hashing}
        finally:
            hashing.cancel()
        return [await asyncio.to_thread(_encode, image, req.seed, params) for image in images]

    async def _images(
        self,
        runtime: Runtime,
        req: GenRequest,
        params: dict[str, Any],
        inputs: list[FittedImage],
        loras: list[tuple[Path, float]],
    ) -> list[Any]:
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

        call = partial(
            self._infer, runtime, self._adapters, loras, req.seed, prompt, params, inputs, on_step
        )
        work = asyncio.ensure_future(asyncio.to_thread(call))
        try:
            return await asyncio.shield(work)
        except asyncio.CancelledError:
            cancelled.set()
            # Hold the job (and its GPU slot) until the worker thread has really stopped.
            with contextlib.suppress(BaseException):
                await work
            raise
        except _Interrupted as exc:  # pragma: no cover - only after cancellation
            raise asyncio.CancelledError from exc
        except LoraLoadError as exc:
            pairs = zip(params["loras"], loras, strict=True)  # the same LoRAs, in order
            listed = next((x["path"] for x, (path, _) in pairs if path == exc.path), exc.path.name)
            raise InvalidRequestError(
                f"{self.name} can't load LoRA '{listed}' into {self.model}: {exc.cause}",
                backend=self.name,
                lora=listed,
            ) from exc
        except Exception as exc:
            raise EngineError(f"{self.name} generation failed: {exc!r}", backend=self.name) from exc
        finally:
            # Progress updates still in flight must land before the job is finalised.
            await asyncio.gather(
                *(asyncio.wrap_future(f) for f in progress), return_exceptions=True
            )

    @staticmethod
    def _infer(
        runtime: Runtime,
        adapters: LoraAdapters,
        loras: list[tuple[Path, float]],
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
        with adapters.lock:
            adapters.apply(runtime.pipe, loras)  # outside inference mode: it loads weights
            with runtime.inference():
                output = runtime.pipe(**kwargs)
        return list(output.images)

    def _health(
        self, status: BackendStatus, started: float, detail: str, note: str | None = None
    ) -> BackendHealth:
        return BackendHealth(
            status=status,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
            detail=f"{detail}; {note}" if note else detail,
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
