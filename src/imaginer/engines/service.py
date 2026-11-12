"""The Engines screen's backend: one view per configured engine, combining its config,
live health, the process this server started, the managed runtime and any install.

Changes made from the screen are validated as a whole backend config, applied to the
registry at once, and saved to ``config/<profile>.local.yaml``, which is loaded over the
profile; the hand-written profile is never rewritten. Secrets are not part of it: API
keys stay in the environment and are only named. Saves are serialized, since every
backend shares that one file.

Starting, stopping and installing take one lock per engine type, because an install
replaces the runtime that every engine of that type is started from.

An sd-server with an edit card switches between its model card and the edit card per task
(:meth:`EngineService.ensure`, called by the adapter as a job starts, inside the engine's
gate): it restarts with the other card's files and waits for them to load. Which card is
loaded is not saved: a started engine always starts with its model card.

Runtimes install on this machine, so only local engines (launched here, or on a loopback
address) show a runtime or accept an install; a remote engine is updated on its own host.
"""

import asyncio
import contextlib
import os
import tempfile
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import yaml
from pydantic import TypeAdapter, ValidationError

from imaginer.backends import diffusers_inproc
from imaginer.backends.base import BackendHealth, LoadingCallback
from imaginer.backends.registry import BackendRegistry
from imaginer.backends.sdcpp import Face
from imaginer.core.config import (
    BackendConfig,
    ConfigError,
    DiffusersBackendConfig,
    LinkedBackendConfig,
    LlamaCppBackendConfig,
    SdCppBackendConfig,
    Settings,
    local_overrides_path,
    merge_mappings,
    resolve_profile_path,
)
from imaginer.core.logging import get_logger
from imaginer.domain import models as dto
from imaginer.domain.enums import BackendStatus
from imaginer.domain.errors import (
    ConflictError,
    EngineTimeoutError,
    EngineUnavailableError,
    InvalidRequestError,
    NotFoundError,
)
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
from imaginer.image_models import recipes

log = get_logger(__name__)

_CONFIG = TypeAdapter[BackendConfig](BackendConfig)
# A backend's config as it runs: for an sd-server with a model card, the card's files and
# settings over its own (ModelLibrary.effective); for a llama-server with an agent, the
# agent's (AgentLibrary.effective).
ConfigView = Callable[[str, BackendConfig], BackendConfig]
# The engines whose config as it runs can differ from their own.
_VIEWED = (SdCppBackendConfig, LlamaCppBackendConfig)
# Where an engine's presets and default LoRAs go while it runs a model card: the card.
CardSettings = Callable[
    [str, dict[str, dict[str, Any]] | None, list[dict[str, Any]] | None], Awaitable[None]
]
SWITCH_POLL_S = 1.0  # how often a switching engine is asked whether it has loaded


class EngineService:
    """``settings.backends`` holds each backend's config as it runs (the registry's); this
    keeps an sd-server's own config too (``raw``), which changes are made to and saved
    from, so a model card's files and settings never end up in the profile."""

    def __init__(
        self,
        settings: Settings,
        registry: BackendRegistry,
        *,
        raw: dict[str, BackendConfig] | None = None,
        view: ConfigView | None = None,
    ) -> None:
        self._settings = settings
        self._registry = registry
        source = raw if raw is not None else settings.backends
        self._raw: dict[str, BackendConfig] = {
            n: c for n, c in source.items() if isinstance(c, _VIEWED)
        }
        self._view: ConfigView = view or (lambda _name, config: config)
        self.on_card_settings: CardSettings | None = None
        self.card_name: Callable[[str | None], str | None] = lambda card: card
        self._loading: dict[str, str] = {}  # engine -> the card it is switching to
        self._switches: dict[str, asyncio.Task[None]] = {}  # engine -> its stop-and-start
        self.supervisor = Supervisor(settings.data_root / "engines" / "logs")
        self.installer = Installer()
        self._tasks: set[asyncio.Task[None]] = set()
        self._lifecycle: dict[str, asyncio.Lock] = {}  # engine type -> start/stop/install
        self._saving = asyncio.Lock()  # one settings update at a time: they share one file

    # --- views -----------------------------------------------------------------

    def names(self) -> list[str]:
        return list(self._settings.backends)

    def _require(self, name: str) -> BackendConfig:
        """The backend's own config, as set in the profile and on this screen."""
        config = self._settings.backends.get(name)
        if config is None:
            raise NotFoundError(f"no backend named {name!r}")
        return self._raw.get(name, config)

    def own_config(self, name: str) -> BackendConfig:
        return self._require(name)

    def effective(self, name: str) -> BackendConfig:
        """The backend's config as it runs: its model card's files and settings over its
        own."""
        self._require(name)
        return self._settings.backends[name]

    def running(self, name: str) -> bool:
        return self.supervisor.get(name).alive

    def running_command(self, name: str) -> list[str] | None:
        proc = self.supervisor.peek(name)
        if proc is None or not proc.alive or proc.run is None:
            return None
        return list(proc.run.argv)

    def command(self, name: str) -> list[str] | None:
        """The command line the engine would start with now, if it can start."""
        config = self.effective(name)
        exe = self._executable(config)
        if exe is None or getattr(config, "launch", None) is None:
            return None
        try:
            return launch.command(config, exe)
        except launch.LaunchError:
            return None

    def edit_command(self, name: str) -> list[str] | None:
        """The command line the engine would switch to for edits, if it has an edit card."""
        view = getattr(self.effective(name), "edit_view", None)
        exe = self._executable(view) if view is not None else None
        if view is None or exe is None or view.launch is None:
            return None
        try:
            return launch.command(view, exe)
        except launch.LaunchError:
            return None

    def loaded_face(self, name: str) -> Face | None:
        """Which of its cards a running engine has loaded: its model card (main) or its edit
        card. None while it is stopped or switching, or when it runs other files (its card
        changed since it started)."""
        if name in self._loading:
            return None
        now = self.running_command(name)
        if now is None:
            return None
        if now == self.command(name):
            return "main"
        return "edit" if now == self.edit_command(name) else None

    def loading(self, engine: str) -> str | None:
        """The card the engine is switching to, while it switches."""
        return self._loading.get(engine)

    async def ensure(
        self,
        engine: str,
        face: Face,
        config: SdCppBackendConfig,
        *,
        notify: LoadingCallback | None,
        probe: Callable[[], Awaitable[BackendHealth]],
    ) -> None:
        """Return once the engine runs ``config``'s files (a face's card), loaded. One that
        runs the other card's is restarted with these first; one that is stopped stays
        stopped (it fails, as it would without an edit card). A job calls this inside the
        engine's gate, so nothing else is using the engine."""
        own = self._require(engine)
        exe = self._executable(own)
        if exe is None or config.launch is None:
            return  # not started here: it runs what it runs
        try:
            # Before anything stops: missing files fail now and leave the engine running.
            argv = await asyncio.to_thread(launch.command, config, exe)
        except launch.LaunchError as exc:
            raise InvalidRequestError(f"{engine} can't switch cards: {exc}") from exc
        card = self.card_name(config.model_card) or config.model_card or face
        running = self.running_command(engine)
        if running is None and engine not in self._switches:
            raise EngineUnavailableError(f"{engine} is stopped: start it first", backend=engine)
        deadline = time.monotonic() + config.launch.swap_timeout_s
        if running != argv:
            log.info("engine.switching", engine=engine, face=face, card=config.model_card)
            self._loading[engine] = card
            if notify is not None:
                await notify(card)
            switch = self._switches.get(engine)
            if switch is None or switch.done():
                switch = asyncio.ensure_future(self._restart(engine, own.type, argv))
                self._switches[engine] = switch
                switch.add_done_callback(lambda _t: self._switches.pop(engine, None))
            # Shielded: a cancelled job must not leave the engine half restarted.
            await asyncio.shield(switch)
        try:
            await self._loaded(engine, card, probe, deadline)
        except asyncio.CancelledError:
            # The engine goes on loading: say so until it is ready (or gone).
            self._spawn(self._settle(engine, card, probe, deadline))
            raise
        except BaseException:
            self._loading.pop(engine, None)
            raise
        if self._loading.pop(engine, None) is not None:
            log.info("engine.switched", engine=engine, face=face, card=config.model_card)
            if notify is not None:
                await notify(None)

    async def _restart(self, engine: str, engine_type: str, argv: list[str]) -> None:
        async with self._lock(engine_type):
            await self.supervisor.stop(engine)
            await self.supervisor.start(engine, argv)

    async def _loaded(
        self,
        engine: str,
        card: str,
        probe: Callable[[], Awaitable[BackendHealth]],
        deadline: float,
    ) -> None:
        """Wait until the engine answers ready, without holding a lock: it can be stopped
        meanwhile (which ends the wait), and other engines started."""
        while True:
            health = await probe()
            if health.status is BackendStatus.OK:
                return
            proc = self.supervisor.peek(engine)
            if proc is None or not proc.alive:
                summary = proc.run.last_error if proc is not None and proc.run else None
                raise EngineUnavailableError(
                    f"{engine} stopped while loading {card}" + (f": {summary}" if summary else ""),
                    backend=engine,
                )
            if time.monotonic() > deadline:
                raise EngineTimeoutError(
                    f"{engine} did not load {card} in time (launch.swap_timeout_s)",
                    backend=engine,
                )
            await asyncio.sleep(SWITCH_POLL_S)

    async def _settle(
        self,
        engine: str,
        card: str,
        probe: Callable[[], Awaitable[BackendHealth]],
        deadline: float,
    ) -> None:
        try:
            await self._loaded(engine, card, probe, deadline)
        finally:
            self._loading.pop(engine, None)

    def refresh(self, name: str) -> None:
        """Apply the view again (a model card changed): the registry takes the new config
        as it runs. A running engine keeps its files until it is restarted."""
        raw = self._require(name)
        if self._registry.reconfigure(name, self._view(name, raw)):
            self._spawn(self._probe_soon(name, delays=(0,)))

    def busy(self, name: str) -> bool:
        """Whether work holds the engine's adapter (a turn or an image job)."""
        return self._registry.leased(name)

    def _lock(self, engine_type: str) -> asyncio.Lock:
        return self._lifecycle.setdefault(engine_type, asyncio.Lock())

    async def all(self) -> list[dto.EngineInfo]:
        return list(await asyncio.gather(*(self.get(n) for n in self.names())))

    async def get(self, name: str) -> dto.EngineInfo:
        config = self._require(name)
        running = self.effective(name)
        info = await self._registry.describe_one(name)
        host, port = launch.endpoint(config) if hasattr(config, "base_url") else (None, None)
        launch_cfg = getattr(config, "launch", None)
        declared = getattr(config, "declared_capabilities", None)
        exe = self._executable(config)
        family = recipes.family(getattr(running, "family", None))
        return dto.EngineInfo(
            name=name,
            type=config.type,
            kind=info.kind,
            enabled=config.enabled,
            status=info.status,
            detail=info.detail,
            model=info.model,
            model_card=getattr(config, "model_card", None),
            edit_card=getattr(config, "edit_card", None),
            edit_card_name=self.card_name(getattr(config, "edit_card", None)),
            loaded=self.loaded_face(name) if getattr(config, "edit_card", None) else None,
            loading=self.loading(name),
            agent=getattr(config, "agent", None),
            family=family.id if family is not None else getattr(running, "family", None),
            family_name=family.name if family is not None else None,
            capabilities=info.capabilities,
            declared_capabilities=sorted(declared) if declared is not None else None,
            default_for=info.default_for,
            base_url=getattr(config, "base_url", None),
            host=host,
            port=port,
            api_key_env=getattr(config, "api_key_env", None),
            loopback=host is not None and launch.is_loopback(host),
            launchable=launch_cfg is not None,
            files=_files(family, await asyncio.to_thread(launch.check_files, running))
            if launch.file_specs(running)
            else [],
            loras=await asyncio.to_thread(self._loras, running),
            model_id=config.model_id if isinstance(config, DiffusersBackendConfig) else None,
            extra_args=list(launch_cfg.extra_args) if launch_cfg is not None else [],
            autostart=bool(launch_cfg and launch_cfg.autostart),
            executable=str(exe) if exe is not None else None,
            process=_process(self.supervisor.peek(name)),
            runtime=self._runtime(config.type) if _local(config) else None,
            install=_install(self.installer.job(config.type)) if _local(config) else None,
            settings_file=str(self._overrides_path()) if self._overrides_path() else None,
        )

    def _loras(self, config: BackendConfig) -> dto.EngineLoraFolder | None:
        """The LoRA folder of an engine that has one here: an sd-server launched here, or a
        Diffusers backend (which lists only .safetensors files)."""
        if isinstance(config, DiffusersBackendConfig):
            check = launch.check_lora_folder(config.lora_dir, diffusers_inproc.LORA_EXTENSIONS)
            mode = None
        elif isinstance(config, SdCppBackendConfig) and config.launch is not None:
            check = launch.check_lora_folder(config.launch.lora_model_dir)
            mode = config.launch.lora_apply_mode
        else:
            return None
        return dto.EngineLoraFolder(
            path=check.path,
            exists=check.exists,
            count=check.count,
            problem=check.problem,
            warning=check.warning,
            apply_mode=mode,
            default_path=str(launch.default_lora_dir(self._settings)),
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
                argv = launch.command(self.effective(name), exe)
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
            block |= await _lora_settings(config, body.launch)
            patch["launch"] = block
        if body.model_card is not None:
            if not isinstance(config, SdCppBackendConfig):
                raise InvalidRequestError(
                    f"{name} is not an sd-server engine; it runs no model card"
                )
            patch["model_card"] = body.model_card.strip() or None
        if body.edit_card is not None:
            if not isinstance(config, SdCppBackendConfig):
                raise InvalidRequestError(
                    f"{name} is not an sd-server engine; it runs no model card"
                )
            patch["edit_card"] = body.edit_card.strip() or None
        if body.agent is not None:
            if not isinstance(config, LlamaCppBackendConfig):
                raise InvalidRequestError(f"{name} is not a llama-server engine; it runs no agent")
            patch["agent"] = body.agent.strip() or None
        if body.model_id is not None:
            if not isinstance(config, DiffusersBackendConfig):
                raise InvalidRequestError(f"{name} is not a Diffusers backend; it has no model_id")
            patch["model_id"] = body.model_id.strip()
        if body.lora_dir is not None:
            if not isinstance(config, DiffusersBackendConfig):
                raise InvalidRequestError(
                    f"{name} is not a Diffusers backend: an sd-server's LoRA folder is "
                    "launch.lora_model_dir"
                )
            text = body.lora_dir.strip()
            if text:
                await asyncio.to_thread(_make_lora_folder, text)
            patch["lora_dir"] = text or None
        if body.presets is not None or body.default_loras is not None:
            if not isinstance(config, SdCppBackendConfig | DiffusersBackendConfig):
                raise InvalidRequestError(f"{name} has no presets or default LoRAs")
            loras = (
                [lora.model_dump() for lora in body.default_loras]
                if body.default_loras is not None
                else None
            )
            card = getattr(config, "model_card", None)
            if card and self.on_card_settings is not None:
                # They belong to the model the engine runs, not to the engine.
                await self.on_card_settings(card, body.presets, loras)
            else:
                if body.presets is not None:
                    patch["presets"] = body.presets
                if loras is not None:
                    patch["defaults"] = {"loras": loras}
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
        if isinstance(updated, _VIEWED):
            self._raw[name] = updated
        # Turns and jobs already using the old adapter keep it until they end.
        if self._registry.reconfigure(name, self._view(name, updated)):
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


async def _lora_settings(config: BackendConfig, body: dto.EngineLaunchUpdate) -> dict[str, Any]:
    """An sd-server's LoRA folder and apply mode from the screen. A folder that isn't
    there yet is made, so Hugging Face can install LoRAs into it."""
    if body.lora_model_dir is None and body.lora_apply_mode is None:
        return {}
    if not isinstance(config, SdCppBackendConfig):
        raise InvalidRequestError(f"{config.type} engines take no LoRAs")
    block: dict[str, Any] = {}
    if body.lora_apply_mode is not None:
        block["lora_apply_mode"] = body.lora_apply_mode or None
    if body.lora_model_dir is not None:
        text = body.lora_model_dir.strip()
        if text:
            await asyncio.to_thread(_make_lora_folder, text)
        block["lora_model_dir"] = text or None
    return block


def _make_lora_folder(text: str) -> None:
    folder = launch.expand(text)
    if not folder.is_absolute():
        raise InvalidRequestError("the LoRA folder must be an absolute path")
    if folder.exists() and not folder.is_dir():
        raise InvalidRequestError(f"{folder} is a file, not a folder")
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise InvalidRequestError(f"can't make the LoRA folder {folder}: {exc}") from exc


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


def _files(family: recipes.Family | None, checks: list[launch.FileCheck]) -> list[dto.EngineFile]:
    """The engine's files, each with whether its model family still needs it (a slot of a
    group, like a VAE or a tiny autoencoder, once none of the group is set) and its name."""
    filled = {c.key for c in checks if c.path is not None}
    out = []
    for check in checks:
        extra: dict[str, Any] = {}
        slot = family.slot(check.key) if family is not None else None
        if family is not None and slot is not None:
            open_group = any(
                check.key in group and not filled & set(group) for group in family.required_any
            )
            extra = {"required": slot.required or open_group, "label": slot.label}
        out.append(dto.EngineFile(**vars_of(check), **extra))
    return out


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
