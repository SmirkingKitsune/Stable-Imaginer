"""Engine runtime installs from the Engines screen (designs 1b, 1c).

The provisioners are synchronous (see docs/*-provisioning.md), so an install runs in a
worker thread. Its progress, output and outcome are kept here and published to
subscribers; cancelling sets the flag the provisioner polls.

On shutdown, running installs are cancelled and waited for: a cancelled provisioner still
stops its build, releases the install lock and finishes or undoes the runtime swap. The
workers aren't daemon threads, so even past the wait the process doesn't exit under them.
"""

import asyncio
import contextlib
import shutil
import subprocess
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

from imaginer.core.clock import utcnow
from imaginer.core.logging import get_logger

log = get_logger(__name__)

InstallState = Literal["running", "succeeded", "failed", "cancelled"]
OUTPUT_LINES = 200
SHUTDOWN_WAIT_S = 60.0  # for cancelled installs to clean up before shutdown carries on


def engine_module(engine_type: str) -> tuple[ModuleType, type[Any]]:
    """The provisioning module and provisioner class for an engine type."""
    from imaginer.provisioning import llamacpp, sdcpp

    if engine_type == "sdcpp":
        return sdcpp, sdcpp.SdCppProvisioner
    if engine_type == "llamacpp":
        return llamacpp, llamacpp.LlamaProvisioner
    raise ValueError(f"no installer for '{engine_type}' engines")


@dataclass(frozen=True, slots=True)
class InstallRequest:
    method: str = "auto"
    accelerator: str = ""
    version: str = "latest"
    force: bool = False


def provision_config(engine_type: str, root: Path, req: InstallRequest) -> Any:
    module, _ = engine_module(engine_type)
    return module.ProvisionConfig(
        provision_dir=root,
        install_method=req.method,
        version=req.version,
        accelerator=req.accelerator,
        force_reinstall=req.force,
    ).normalized()


def plan(engine_type: str, root: Path, req: InstallRequest) -> list[dict[str, Any]]:
    """The attempts an install would make, in order, without downloading anything."""
    module, runner = engine_module(engine_type)
    config = provision_config(engine_type, root, req)
    provisioner = runner(config)
    attempts = []
    for label, attempt in provisioner.install_attempts():
        steps = module.build_install_plan(attempt)
        attempts.append(
            {
                "attempt": label,
                "method": attempt.install_method,
                "accelerator": attempt.accelerator,
                "steps": [step.label for step in steps],
            }
        )
    return attempts


def detect_hardware(engine_type: str) -> dict[str, Any]:
    """What the installer would pick: accelerator, CUDA arch and driver, GPU name."""
    module, _ = engine_module(engine_type)
    from imaginer.provisioning import llamacpp as shared

    out: dict[str, Any] = {"accelerator": None, "cuda_arch": None, "driver_cuda": None, "gpu": None}
    with contextlib.suppress(Exception):
        out["accelerator"] = shared.detect_accelerator()
    if out["accelerator"] == "cuda":
        with contextlib.suppress(Exception):
            out["cuda_arch"] = shared.detect_cuda_arch() or None
        with contextlib.suppress(Exception):
            version = shared.detect_driver_cuda_version()
            out["driver_cuda"] = ".".join(map(str, version)) if version else None
        out["gpu"] = _gpu_name()
    accelerators = getattr(module, "ACCELERATORS", ())
    out["accelerators"] = list(accelerators)
    out["methods"] = list(getattr(module, "INSTALL_METHODS", ("auto", "release", "source")))
    return out


def _gpu_name() -> str | None:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        done = subprocess.run(
            [exe, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    first = done.stdout.strip().splitlines()
    return first[0].strip() if first else None


@dataclass
class InstallJob:
    engine: str
    request: InstallRequest
    started_at: datetime
    state: InstallState = "running"
    step: int = 0
    total: int = 0
    label: str = ""
    fraction: float = -1.0
    finished_at: datetime | None = None
    error: str | None = None
    warning: str | None = None
    executable: str | None = None
    log_path: str | None = None
    output: deque[str] = field(default_factory=lambda: deque(maxlen=OUTPUT_LINES))
    cancel: threading.Event = field(default_factory=threading.Event)


Listener = Callable[[InstallJob], None]


class Installer:
    """At most one install per engine type at a time (they share a runtime directory)."""

    def __init__(self) -> None:
        self._jobs: dict[str, InstallJob] = {}  # engine type -> its latest job
        self._listeners: dict[str, set[Listener]] = {}
        self._workers: dict[str, threading.Thread] = {}  # engine type -> its latest worker

    def job(self, engine_type: str) -> InstallJob | None:
        return self._jobs.get(engine_type)

    def subscribe(self, engine_type: str, listener: Listener) -> Callable[[], None]:
        self._listeners.setdefault(engine_type, set()).add(listener)
        return lambda: self._listeners.get(engine_type, set()).discard(listener)

    def _publish(self, job: InstallJob) -> None:
        for listener in list(self._listeners.get(job.engine, ())):
            listener(job)

    def start(self, engine_type: str, root: Path, req: InstallRequest) -> InstallJob:
        current = self._jobs.get(engine_type)
        if current is not None and current.state == "running":
            raise RuntimeError(f"a {engine_type} install is already running")
        _, runner = engine_module(engine_type)
        config = provision_config(engine_type, root, req)
        job = InstallJob(engine=engine_type, request=req, started_at=utcnow())
        self._jobs[engine_type] = job
        loop = asyncio.get_running_loop()

        def publish() -> None:
            # The loop may be gone if the install outlives shutdown's wait.
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(self._publish, job)

        def on_line(text: str, is_stderr: bool) -> None:
            job.output.append(text.rstrip())
            publish()

        def on_progress(progress: Any) -> None:
            job.step, job.total = progress.step, progress.total
            job.label, job.fraction = progress.label, progress.fraction
            publish()

        provisioner = runner(
            config, log_sink=on_line, progress_sink=on_progress, cancel_check=job.cancel.is_set
        )

        def work() -> None:
            try:
                binary = provisioner.ensure_runtime()
                job.executable = str(binary)
                job.warning = provisioner.target_mismatch or None
                job.state = "succeeded"
            except BaseException as exc:
                job.state = "cancelled" if job.cancel.is_set() else "failed"
                job.error = str(exc) or type(exc).__name__
                log.warning("engine.install_failed", engine=engine_type, error=job.error)
            finally:
                path = getattr(provisioner, "build_log_path", None)
                job.log_path = str(path) if path else None
                job.finished_at = utcnow()
                publish()

        worker = threading.Thread(target=work, name=f"install-{engine_type}", daemon=False)
        self._workers[engine_type] = worker
        worker.start()
        log.info("engine.install_started", engine=engine_type, **vars_of(req))
        return job

    def cancel(self, engine_type: str) -> InstallJob | None:
        job = self._jobs.get(engine_type)
        if job is not None and job.state == "running":
            job.cancel.set()
        return job

    def cancel_all(self) -> None:
        for job in self._jobs.values():
            job.cancel.set()

    async def aclose(self, timeout_s: float = SHUTDOWN_WAIT_S) -> None:
        """Cancel running installs and wait for their workers to finish cleaning up."""
        self.cancel_all()
        running = {t: w for t, w in self._workers.items() if w.is_alive()}
        if not running:
            return
        log.info("engine.install_waiting", engines=sorted(running))
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        for engine_type, worker in running.items():
            await asyncio.to_thread(worker.join, max(0.0, deadline - loop.time()))
            if worker.is_alive():
                log.warning("engine.install_still_running", engine=engine_type)


def vars_of(req: InstallRequest) -> dict[str, Any]:
    return {"method": req.method, "accelerator": req.accelerator, "version": req.version}
