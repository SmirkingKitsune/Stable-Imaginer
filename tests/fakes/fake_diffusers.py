"""A stand-in for ``QwenImage21Pipeline`` plus a loader, so the Diffusers adapter is tested
without torch. It follows the real call contract: keyword arguments, ``callback_on_step_end
(pipe, step, timestep, kwargs)`` after every step, and ``.images`` holding RGBA PIL images."""

import contextlib
import io
import threading
from types import SimpleNamespace
from typing import Any

from PIL import Image

from imaginer.backends.diffusers_inproc import Runtime
from imaginer.core.config import DiffusersBackendConfig
from tests.fakes.fake_sdcpp import RGBA_MARKER, render


class FakePipeline:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.hold: threading.Event | None = None  # pause after the first step until set
        self.fail: str | None = None
        self.started = threading.Event()
        self.finished = threading.Event()
        self.steps_run = 0

    def __call__(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
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
