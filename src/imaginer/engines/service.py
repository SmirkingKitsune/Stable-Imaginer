"""The Engines screen's backend: one view per configured engine, combining its config,
live health, the process this server started, the managed runtime and any install.

Changes made from the screen are validated as a whole backend config, applied to the
registry at once, and saved to ``config/<profile>.local.yaml``, which is loaded over the
profile; the hand-written profile is never rewritten. Secrets are not part of it: API
keys stay in the environment and are only named. Saves are serialized, since every
backend shares that one file.

Starting, stopping and installing take one lock per engine type, because an install
replaces the runtime that every engine of that type is started from.

Runtimes install on this machine, so only local engines (launched here, or on a loopback
address) show a runtime or accept an install; a remote engine is updated on its own host.
"""

import asyncio
import contextlib
import os
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import yaml
from pydantic import TypeAdapter, ValidationError

from imaginer.backends.registry import BackendRegistry
from imaginer.core.config import (
    BackendConfig,
    ConfigError,
    LinkedBackendConfig,
    Settings,
    local_overrides_path,
    merge_mappings,
    resolve_profile_path,
)
from imaginer.core.logging import get_logger
from imaginer.domain import models as dto
from imaginer.domain.errors import ConflictError, InvalidRequestError, NotFoundError
from imaginer.engines import launch
from imaginer.engines.installer import (
    Installer,
    InstallJob,
    InstallRequest,
    detect_hardware,
    engine_module,
)
from imaginer.engines.installer import plan as install_plan
from imaginer.engines.supervisor import EngineProcess, Supervisor

log = get_logger(__name__)

_CONFIG = TypeAdapter[BackendConfig](BackendConfig)


class EngineService:
    def __init__(self, settings: Settings, registry: BackendRegistry) -> None:
        self._settings = settings
        self._registry = registry
        self.supervisor = Supervisor(settings.data_root / "engines" / "logs")
        self.installer = Installer()
        self._tasks: set[asyncio.Task[None]] = set()
        self._lifecycle: dict[str, asyncio.Lock] = {}  # engine type -> start/stop/install
        self._saving = asyncio.Lock()  # one settings update at a time: they share one file

    # --- views -----------------------------------------------------------------

    def names(self) -> list[str]:
        return list(self._settings.backends)

    def _require(self, name: str) -> BackendConfig:
        config = self._settings.backends.get(name)
        if config is None:
            raise NotFoundError(f"no backend named {name!r}")
        return config

    def _lock(self, engine_type: str) -> asyncio.Lock:
        return self._lifecycle.setdefault(engine_type, asyncio.Lock())

    async def all(self) -> list[dto.EngineInfo]:
        return list(await asyncio.gather(*(self.get(n) for n in self.names())))

    async def get(self, name: str) -> dto.EngineInfo:
        config = self._require(name)
        info = await self._registry.describe_one(name)
        host, port = launch.endpoint(config) if hasattr(config, "base_url") else (None, None)
        launch_cfg = getattr(config, "launch", None)
        declared = getattr(config, "declared_capabilities", None)
        exe = self._executable(config)
        return dto.EngineInfo(
            name=name,
            type=config.type,
            kind=info.kind,
            enabled=config.enabled,
            status=info.status,
            detail=info.detail,
            model=info.model,
            capabilities=info.capabilities,
            declared_capabilities=sorted(declared) if declared is not None else None,
            default_for=info.default_for,
            base_url=getattr(config, "base_url", None),
            host=host,
            port=port,
            api_key_env=getattr(config, "api_key_env", None),
            loopback=host is not None and launch.is_loopback(host),
            launchable=launch_cfg is not None,
            files=[dto.EngineFile(**vars_of(c)) for c in launch.check_files(config)]
            if launch.file_specs(config)
            else [],
            extra_args=list(launch_cfg.extra_args) if launch_cfg is not None else [],
            autostart=bool(launch_cfg and launch_cfg.autostart),
            executable=str(exe) if exe is not None else None,
            process=_process(self.supervisor.peek(name)),
            runtime=self._runtime(config.type) if _local(config) else None,
            install=_install(self.installer.job(config.type)) if _local(config) else None,
            settings_file=str(self._overrides_path()) if self._overrides_path() else None,
        )

    def executable(self, config: BackendConfig) -> Path | None:
        """The executable a local engine starts from, if it has one."""
        return self._executable(config)

    def _executable(self, config: BackendConfig) -> Path | None:
        if config.type not in launch.RUNTIME_FOLDERS:
            return None
        module, _ = engine_module(config.type)
        return launch.executable(self._settings, config, module.read_active_runtime)

    def _runtime(self, engine_type: str) -> dto.EngineRuntime | None:
        root = launch.runtime_dir(self._settings, engine_type)
        if root is None:
            return None
        module, _ = engine_module(engine_type)
        active = module.read_active_runtime(root)
        exe = active.get("executable")
        return dto.EngineRuntime(
            installed=bool(exe and Path(str(exe)).is_file()),
            version=_str(active.get("version")),
            method=_str(active.get("method")),
            accelerator=_str(active.get("accelerator")),
            executable=_str(exe),
            installed_at=_str(active.get("installed_at")),
            directory=str(root),
        )

    def log(self, name: str, *, errors_only: bool = False) -> list[dto.EngineLogLine]:
        self._require(name)
        proc = self.supervisor.peek(name)
        if proc is None or proc.run is None:
            return []
        return [
            dto.EngineLogLine(at=line.at, text=line.text, error=line.error)
            for line in proc.run.lines
            if line.error or not errors_only
        ]

    def log_file(self, name: str) -> Path | None:
        self._require(name)
        proc = self.supervisor.peek(name)
        return proc.run.log_path if proc is not None and proc.run is not None else None

    # --- process -------------------------------------------------------------------

    async def start(self, name: str) -> dto.EngineInfo:
        config = self._require(name)
        async with self._lock(config.type):
            config = self._require(name)  # an update may have landed while we waited
            if self.supervisor.get(name).alive:
                raise ConflictError(f"{name} is already running")
            if self._installing(config.type):
                raise ConflictError(
                    f"a {config.type} runtime install is running; start {name} once it is done"
                )
            exe = self._executable(config)
            if getattr(config, "launch", None) is None:
                raise InvalidRequestError(
                    f"{name} has no launch settings, so it isn't started here"
                )
            if exe is None or not exe.is_file():
                raise InvalidRequestError(
                    f"no {config.type} executable: install the runtime or set launch.executable"
                )
            try:
                argv = launch.command(config, exe)
            except launch.LaunchError as exc:
                raise InvalidRequestError(str(exc)) from exc
            try:
                await self.supervisor.start(name, argv)
            except OSError as exc:
                raise InvalidRequestError(f"could not start {exe.name}: {exc}") from exc
        self._watch_health(name)
        return await self.get(name)

    async def stop(self, name: str) -> dto.EngineInfo:
        config = self._require(name)
        async with self._lock(config.type):
            await self.supervisor.stop(name)
        self._spawn(self._probe_soon(name, delays=(0.5,)))
        return await self.get(name)

    def _installing(self, engine_type: str) -> bool:
        job = self.installer.job(engine_type)
        return job is not None and job.state == "running"

    def _watch_health(self, name: str) -> None:
        """Probe while the engine loads, so the status line turns ready without waiting
        for the next poll."""
        self._spawn(self._probe_soon(name, delays=(1, 2, 3, 5, 8, 13, 21, 34)))

    async def _probe_soon(self, name: str, delays: tuple[float, ...]) -> None:
        for delay in delays:
            await asyncio.sleep(delay)
            with contextlib.suppress(Exception):
                health = await self._registry.probe(name)
                if health is not None and health.status.value == "ok":
                    return
            proc = self.supervisor.peek(name)
            if proc is not None and not proc.alive:
                return

    async def autostart(self) -> None:
        for name, config in self._settings.backends.items():
            launch_cfg = getattr(config, "launch", None)
            if config.enabled and launch_cfg is not None and launch_cfg.autostart:
                try:
                    await self.start(name)
                except Exception as exc:
                    log.warning("engine.autostart_failed", engine=name, error=str(exc))

    # --- settings ----------------------------------------------------------------------

    async def save_settings(self, patch: dict[str, Any]) -> Path | None:
        """Merge ``patch`` into ``config/<profile>.local.yaml``, keeping everything else in it
        (the Links screen saves its changes here too). Returns where it was saved."""
        path = self._overrides_path()
        async with self._saving:
            if path is not None:
                await asyncio.to_thread(save_patch, path, patch)
        return path

    def settings_file(self) -> Path | None:
        return self._overrides_path()

    def _overrides_path(self) -> Path | None:
        if not self._settings.profile:
            return None
        try:
            return local_overrides_path(resolve_profile_path(self._settings.profile))
        except ConfigError:
            return None

    async def update(self, name: str, body: dto.EngineUpdate) -> dto.EngineInfo:
        self._require(name)
        async with self._saving:
            await self._update(name, body)
        return await self.get(name)

    async def _update(self, name: str, body: dto.EngineUpdate) -> None:
        config = self._require(name)
        patch: dict[str, Any] = {}
        if body.host is not None or body.port is not None:
            if not hasattr(config, "base_url"):
                raise InvalidRequestError(f"{name} has no endpoint")
            host, port = launch.endpoint(config)
            host = body.host.strip() if body.host is not None else host
            port = body.port if body.port is not None else port
            patch["base_url"] = _with_endpoint(str(config.base_url), host, port)
        if body.declared_capabilities is not None:
            patch["declared_capabilities"] = sorted(c.value for c in body.declared_capabilities)
        if body.launch is not None:
            if not launch.file_specs(config):
                raise InvalidRequestError(f"{config.type} engines can't be launched from here")
            keys = {key for key, _, _ in launch.file_specs(config)}
            unknown = sorted(set(body.launch.files) - keys)
            if unknown:
                raise InvalidRequestError(f"unknown launch files: {', '.join(unknown)}")
            block: dict[str, Any] = {
                k: (v.strip() or None) if v else None for k, v in body.launch.files.items()
            }
            if body.launch.extra_args is not None:
                block["extra_args"] = body.launch.extra_args
            if body.launch.autostart is not None:
                block["autostart"] = body.launch.autostart
            if body.launch.executable is not None:
                block["executable"] = body.launch.executable.strip() or None
            patch["launch"] = block
        if not patch:
            return

        current = config.model_dump(mode="json", exclude_unset=False)
        try:
            updated = _CONFIG.validate_python(merge_mappings(current, patch))
        except ValidationError as exc:
            raise InvalidRequestError(
                "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
            ) from exc
        path = self._overrides_path()
        if path is not None:
            await asyncio.to_thread(_save_overrides, path, name, patch)
        # Turns and jobs already using the old adapter keep it until they end.
        if self._registry.reconfigure(name, updated):
            self._spawn(self._probe_soon(name, delays=(0,)))
        log.info("engine.updated", engine=name, fields=sorted(patch), saved_to=str(path))

    # --- install -------------------------------------------------------------------------

    def _install_root(self, name: str) -> tuple[str, Path]:
        config = self._require(name)
        root = launch.runtime_dir(self._settings, config.type)
        if root is None:
            raise InvalidRequestError(f"{config.type} engines have no managed runtime")
        if not _local(config):
            host, _ = launch.endpoint(config)
            raise InvalidRequestError(
                f"{name} runs on {host}, and runtimes install on this machine: "
                "install or update it on that host"
            )
        return config.type, root

    async def install_plan(self, name: str, body: dto.EngineInstallRequest) -> dict[str, Any]:
        engine_type, root = self._install_root(name)
        req = InstallRequest(body.method, body.accelerator, body.version, body.force)
        hardware = await asyncio.to_thread(detect_hardware, engine_type)
        if not req.accelerator and hardware.get("accelerator"):
            req = InstallRequest(req.method, hardware["accelerator"], req.version, req.force)
        try:
            attempts = await asyncio.to_thread(install_plan, engine_type, root, req)
        except Exception as exc:
            raise InvalidRequestError(f"no install plan: {exc}") from exc
        return {
            "engine": engine_type,
            "directory": str(root),
            "hardware": hardware,
            "attempts": attempts,
        }

    async def install(self, name: str, body: dto.EngineInstallRequest) -> dto.EngineInfo:
        engine_type, root = self._install_root(name)
        async with self._lock(engine_type):
            if any(
                self.supervisor.get(n).alive
                for n, c in self._settings.backends.items()
                if c.type == engine_type
            ):
                raise ConflictError("stop the engine before installing a new runtime")
            req = InstallRequest(body.method, body.accelerator, body.version, body.force)
            try:
                self.installer.start(engine_type, root, req)
            except RuntimeError as exc:
                raise ConflictError(str(exc)) from exc
            except Exception as exc:
                raise InvalidRequestError(str(exc)) from exc
        return await self.get(name)

    async def cancel_install(self, name: str) -> dto.EngineInfo:
        engine_type, _ = self._install_root(name)
        self.installer.cancel(engine_type)
        return await self.get(name)

    async def check_update(self, name: str) -> dict[str, Any]:
        engine_type, root = self._install_root(name)
        module, _ = engine_module(engine_type)
        active = module.read_active_runtime(root)
        if not active.get("version"):
            return {"installed": False}
        from dataclasses import asdict

        config = module.ProvisionConfig(
            provision_dir=root,
            install_method=str(active.get("method", "auto")),
            accelerator=str(active.get("accelerator", "")),
        ).normalized()
        try:
            status = await asyncio.to_thread(
                module.check_for_update, config, str(active["version"])
            )
        except Exception as exc:
            return {"installed": True, "error": str(exc)}
        return {"installed": True, **asdict(status)}

    # --- lifecycle -------------------------------------------------------------------------

    def _spawn(self, coro: Any) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def aclose(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        # Installs first: a cancelled one still stops its build, releases the install lock
        # and finishes or undoes the runtime swap, and the process must not exit before.
        await self.installer.aclose()
        await self.supervisor.aclose()


def _local(config: BackendConfig) -> bool:
    """Whether the engine runs on this machine: launched here, or on a loopback address."""
    if isinstance(config, LinkedBackendConfig):
        return False
    if getattr(config, "launch", None) is not None:
        return True
    if not hasattr(config, "base_url"):
        return True  # in process
    host, _ = launch.endpoint(config)
    return launch.is_loopback(host)


def _with_endpoint(base_url: str, host: str, port: int) -> str:
    """``base_url`` with a new host and port, keeping its scheme, credentials, path (an engine
    behind a path-based proxy), query and fragment."""
    url = urlsplit(base_url)
    credentials = url.netloc.rpartition("@")[0]
    netloc = f"[{host}]:{port}" if ":" in host else f"{host}:{port}"
    if credentials:
        netloc = f"{credentials}@{netloc}"
    return urlunsplit(url._replace(netloc=netloc))


def _save_overrides(path: Path, name: str, patch: dict[str, Any]) -> None:
    """Merge one backend's changes into the sidecar, keeping everything else in it."""
    save_patch(path, {"backends": {name: patch}})


def save_patch(path: Path, patch: dict[str, Any]) -> None:
    """Merge ``patch`` into the sidecar, keeping everything else in it.

    Callers serialize saves; the temporary file still gets a unique name, so a save can
    never replace the file with another's half-written one."""
    data: dict[str, Any] = {}
    if path.is_file():
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if isinstance(loaded, dict):
            data = loaded
    data = merge_mappings(data, patch)
    header = (
        "# Saved from the Engines and Links screens. Loaded over the profile with the same\n"
        "# name; edit or delete freely. Secrets never go here: they stay in the environment.\n"
    )
    text = header + yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        tmp.replace(path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise


def _process(proc: EngineProcess | None) -> dto.EngineProcess | None:
    if proc is None or proc.run is None:
        return None
    run = proc.run
    return dto.EngineProcess(
        state=proc.state,
        pid=run.pid,
        exit_code=run.exit_code,
        started_at=run.started_at,
        exited_at=run.exited_at,
        summary=run.last_error,
        command=run.argv,
        log_path=str(run.log_path),
    )


def _install(job: InstallJob | None) -> dto.EngineInstall | None:
    if job is None:
        return None
    return dto.EngineInstall(
        state=job.state,
        method=job.request.method,
        accelerator=job.request.accelerator,
        version=job.request.version,
        step=job.step,
        total=job.total,
        label=job.label,
        fraction=job.fraction,
        started_at=job.started_at,
        finished_at=job.finished_at,
        error=job.error,
        warning=job.warning,
        executable=job.executable,
        log_path=job.log_path,
        output=list(job.output)[-40:],
    )


def vars_of(check: launch.FileCheck) -> dict[str, Any]:
    return {
        "key": check.key,
        "flag": check.flag,
        "what": check.what,
        "path": check.path,
        "exists": check.exists,
        "size_bytes": check.size_bytes,
        "format": check.format,
        "problem": check.problem,
    }


def _str(value: object) -> str | None:
    return None if value in (None, "") else str(value)
