"""A stand-in for ``QwenImage21Pipeline`` plus a loader, so the Diffusers adapter is tested
without torch. It follows the real call contract: keyword arguments, ``callback_on_step_end
(pipe, step, timestep, kwargs)`` after every step, and ``.images`` holding RGBA PIL images.
LoRAs go through the same methods as Diffusers' (``load_lora_weights``, ``set_adapters``,
``enable_lora``/``disable_lora``, ``delete_adapters``); a file that starts with
``NOT_A_LORA`` fails to load, like one made for another model."""

import contextlib
import io
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from PIL import Image

from imaginer.backends.diffusers_inproc import Runtime
from imaginer.core.config import DiffusersBackendConfig
from tests.fakes.fake_sdcpp import RGBA_MARKER, render

NOT_A_LORA = b"not a lora"


class FakePipeline:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.hold: threading.Event | None = None  # pause after the first step until set
        self.fail: str | None = None
        self.started = threading.Event()
        self.finished = threading.Event()
        self.steps_run = 0
        self.adapters: dict[str, str] = {}  # adapter name -> the file name it was loaded from
        self.active: list[tuple[str, float]] = []
        self.lora_enabled = True
        self.loaded: list[str] = []  # file names, in the order they were loaded
        self.lora_calls: list[list[tuple[str, float]]] = []  # (file, weight)s each image used

    def load_lora_weights(self, folder: str, *, weight_name: str, adapter_name: str) -> None:
        if adapter_name in self.adapters:
            raise ValueError(f"adapter {adapter_name} is already loaded")
        if (Path(folder) / weight_name).read_bytes().startswith(NOT_A_LORA):
            raise ValueError("Target modules {'img_in'} not found in the base model.")
        self.adapters[adapter_name] = weight_name
        self.loaded.append(weight_name)

    def set_adapters(self, names: list[str], adapter_weights: list[float]) -> None:
        missing = [n for n in names if n not in self.adapters]
        if missing:
            raise ValueError(f"adapters {missing} are not loaded")
        self.active = list(zip(names, adapter_weights, strict=True))

    def enable_lora(self) -> None:
        self.lora_enabled = True

    def disable_lora(self) -> None:
        self.lora_enabled = False

    def delete_adapters(self, names: str | list[str]) -> None:
        for name in [names] if isinstance(names, str) else names:
            self.adapters.pop(name, None)
        self.active = [(n, w) for n, w in self.active if n in self.adapters]

    def __call__(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        used = [(self.adapters[n], w) for n, w in self.active] if self.lora_enabled else []
        self.lora_calls.append(used)
        self.started.set()
        try:
            if self.fail:
                raise RuntimeError(self.fail)
            for step in range(kwargs["num_inference_steps"]):
                if step == 1 and self.hold is not None:
                    self.hold.wait(30)
                kwargs["callback_on_step_end"](self, step, 1000 - step, {"latents": None})
                self.steps_run += 1
            seed = kwargs["generator"]  # the fake runtime's generator is the seed itself
            png = render(seed, kwargs["width"], kwargs["height"], RGBA_MARKER in kwargs["prompt"])
            image = Image.open(io.BytesIO(png)).convert("RGBA")  # the VAE always decodes RGBA
            return SimpleNamespace(images=[image])
        finally:
            self.finished.set()


class FakeLoader:
    def __init__(self, pipeline: FakePipeline | None = None) -> None:
        self.pipeline = pipeline or FakePipeline()
        self.problem: str | None = None
        self.no_loras: str | None = None  # why LoRAs can't be loaded (peft missing)
        self.load_error: str | None = None
        self.check_gate: threading.Event | None = None
        self.checks = 0
        self.loads = 0
        self.closed = 0

    def check(self, config: DiffusersBackendConfig) -> str | None:
        self.checks += 1
        if self.check_gate is not None:
            self.check_gate.wait(30)
        return self.problem

    def lora_problem(self) -> str | None:
        return self.no_loras

    def load(self, config: DiffusersBackendConfig) -> Runtime:
        self.loads += 1
        if self.load_error:
            raise RuntimeError(self.load_error)

        def close() -> None:
            self.closed += 1

        return Runtime(
            pipe=self.pipeline,
            generator=lambda seed: seed,
            inference=contextlib.nullcontext,
            device=config.device,
            close=close,
        )
