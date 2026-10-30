"""Builds backends from config, keeps their health fresh, and routes requests to them.

Routing never falls back silently. A request goes to the backend it names, or to the
configured default for its task. If that backend lacks a needed capability, the request fails with
422 ``capability_unavailable``. If it is unreachable, it fails with 503 ``engine_unavailable``.

A backend that is disabled, or whose adapter can't be built (missing optional extra, or
not shipped yet), stays visible in ``GET /v1/backends`` with status ``unavailable`` and a
reason, instead of silently disappearing.

Whoever uses an adapter holds a :class:`Lease` on it: a turn for its whole run, an image
job until it ends. When a backend is reconfigured, its old adapter is closed only once the
last lease on it is released, so a long job keeps its client.
"""

import asyncio
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from imaginer.backends.base import BackendHealth, ImageBackend, LLMBackend
from imaginer.backends.diffusers_inproc import DiffusersBackend, missing_dependencies
from imaginer.backends.llamacpp import LlamaCppBackend
from imaginer.backends.sdcpp import SdCppBackend
from imaginer.core.clock import utcnow
from imaginer.core.config import (
    BackendConfig,
    DiffusersBackendConfig,
    LlamaCppBackendConfig,
    SdCppBackendConfig,
    Settings,
    backend_kind,
)
from imaginer.core.logging import get_logger
from imaginer.domain.enums import BackendKind, BackendStatus, Capability
from imaginer.domain.errors import CapabilityUnavailableError, EngineUnavailableError
from imaginer.domain.models import BackendInfo, ImageDefaultsInfo

log = get_logger(__name__)

HEALTH_TIMEOUT_S = 5.0

Backend = LLMBackend | ImageBackend
BackendFactory = Callable[[str, BackendConfig], Backend]
# Config fields the adapter doesn't read: changing only these keeps the adapter.
_NOT_ADAPTER_FIELDS = frozenset({"launch"})
DESCRIBE_MAX_AGE_S = 5.0
Task = Literal["chat", "t2i", "edit"]
_TASKS: tuple[Task, ...] = ("chat", "t2i", "edit")


def _llamacpp(name: str, config: BackendConfig) -> Backend:
    assert isinstance(config, LlamaCppBackendConfig)
    return LlamaCppBackend(name, config)


def _sdcpp(name: str, config: BackendConfig) -> Backend:
    assert isinstance(config, SdCppBackendConfig)
    return SdCppBackend(name, config)


def _diffusers(name: str, config: BackendConfig) -> Backend:
    assert isinstance(config, DiffusersBackendConfig)
    missing = missing_dependencies()
    if missing:
        raise RuntimeError(
            f"missing {', '.join(missing)}; install the extra: uv sync --extra diffusers "
            "(or pip install '.[diffusers]')"
        )
    return DiffusersBackend(name, config)


# Adapter type -> factory. An SGLang Diffusion adapter would register here (docs/adapters.md).
FACTORIES: dict[str, BackendFactory] = {
    "llamacpp": _llamacpp,
    "sdcpp": _sdcpp,
    "diffusers": _diffusers,
}


@dataclass
class _Entry:
    name: str
    config: BackendConfig
    kind: BackendKind
    backend: Backend | None
    reason: str | None  # why backend is None
    health: BackendHealth | None = None
    probed_at: float | None = None  # time.monotonic()
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class Lease[B: (LLMBackend, ImageBackend)]:
    """A hold on one adapter. Release it when done; releasing again does nothing."""

    def __init__(self, registry: "BackendRegistry", backend: B) -> None:
        self.backend: B = backend
        self._registry: BackendRegistry | None = registry
        registry._hold(backend)

    def release(self) -> None:
        registry, self._registry = self._registry, None
        if registry is not None:
            registry._unhold(self.backend)


class BackendRegistry:
    def __init__(
        self,
        entries: list[_Entry],
        settings: Settings,
        factories: Mapping[str, BackendFactory] | None = None,
    ) -> None:
        self._entries = {e.name: e for e in entries}
        self._settings = settings
        self._factories = FACTORIES if factories is None else factories
        self._leases: dict[int, int] = {}  # id(adapter) -> leases held on it
        self._retired: dict[int, Backend] = {}  # replaced adapters that are still leased
        self._closing: set[asyncio.Task[None]] = set()

    @classmethod
    def from_settings(
        cls, settings: Settings, factories: Mapping[str, BackendFactory] | None = None
    ) -> "BackendRegistry":
        factories = FACTORIES if factories is None else factories
        entries = [_build(name, config, factories) for name, config in settings.backends.items()]
        return cls(entries, settings, factories)

    def reconfigure(self, name: str, config: BackendConfig) -> bool:
        """Use ``config`` for backend ``name`` from now on (e.g. a new endpoint).

        A change the adapter doesn't read (launch settings) keeps the adapter and its health.
        Otherwise a new adapter is built and the old one retired: turns and jobs holding a
        lease on it keep using it, and it is closed once the last of them is done. Returns
        whether the adapter was replaced.
        """
        old = self._entries[name]
        self._settings.backends[name] = config
        if _adapter_fields(old.config) == _adapter_fields(config):
            old.config = config
            return False
        self._entries[name] = _build(name, config, self._factories)
        if old.backend is not None:
            self._retire(old.backend)
        return True

    def add(self, name: str, config: BackendConfig) -> None:
        """Add a backend created at runtime (e.g. one a linked instance shares), or
        reconfigure it if it already exists."""
        if name in self._entries:
            self.reconfigure(name, config)
            return
        self._settings.backends[name] = config
        self._entries[name] = _build(name, config, self._factories)

    def rebuild(self, name: str) -> None:
        """Build a new adapter from the same config, because something it was built from
        changed outside the config (a linked instance's address). Leases keep the old one."""
        old = self._entries[name]
        self._entries[name] = _build(name, old.config, self._factories)
        if old.backend is not None:
            self._retire(old.backend)

    def config(self, name: str) -> BackendConfig:
        return self._entries[name].config

    def names(self) -> list[str]:
        return list(self._entries)

    # --- leases ---------------------------------------------------------------

    def _hold(self, backend: Backend) -> None:
        key = id(backend)
        self._leases[key] = self._leases.get(key, 0) + 1

    def _unhold(self, backend: Backend) -> None:
        key = id(backend)
        left = self._leases.get(key, 0) - 1
        if left > 0:
            self._leases[key] = left
            return
        self._leases.pop(key, None)
        retired = self._retired.pop(key, None)
        if retired is not None:
            self._close_soon(retired)

    def _retire(self, backend: Backend) -> None:
        leases = self._leases.get(id(backend), 0)
        if leases:
            self._retired[id(backend)] = backend
            log.info("backend.retiring", backend=backend.name, leases=leases)
        else:
            self._close_soon(backend)

    def _close_soon(self, backend: Backend) -> None:
        task = asyncio.get_running_loop().create_task(_close(backend))
        self._closing.add(task)
        task.add_done_callback(self._closing.discard)

    # --- health ---------------------------------------------------------------

    async def probe(self, name: str) -> BackendHealth | None:
        entry = self._entries[name]
        if entry.backend is None:
            return None
        return await self._probe(entry)

    async def refresh(self) -> None:
        """Probe every built backend (in parallel)."""
        await asyncio.gather(
            *(self._probe(e) for e in self._entries.values() if e.backend is not None)
        )

    async def _probe(self, entry: _Entry) -> BackendHealth:
        assert entry.backend is not None
        async with entry.lock:
            try:
                health = await asyncio.wait_for(entry.backend.health(), HEALTH_TIMEOUT_S)
            except TimeoutError:
                health = BackendHealth(
                    status=BackendStatus.DOWN, detail="health probe timed out", probed_at=utcnow()
                )
            previous = entry.health.status if entry.health else None
            entry.health, entry.probed_at = health, time.monotonic()
        if health.status is not previous:
            log.info(
                "backend.status",
                backend=entry.name,
                status=health.status.value,
                detail=health.detail,
                capabilities=sorted(c.value for c in entry.backend.capabilities()),
            )
        return health

    async def _fresh(self, entry: _Entry) -> BackendHealth:
        """Last probe if it is recent and healthy, otherwise probe now."""
        ttl = self._settings.orchestration.probe_ttl_s
        if (
            entry.health is not None
            and entry.probed_at is not None
            and entry.health.status is BackendStatus.OK
            and time.monotonic() - entry.probed_at < ttl
        ):
            return entry.health
        return await self._probe(entry)

    # --- routing --------------------------------------------------------------

    async def chat_backend(
        self, name: str | None = None, *, require: Iterable[Capability] = (Capability.CHAT,)
    ) -> Lease[LLMBackend]:
        """A lease on the ready chat backend; release it when the turn is over."""
        entry = await self._ready("chat", name, BackendKind.LLM, set(require))
        return Lease(self, cast(LLMBackend, entry.backend))

    async def image_backend(
        self,
        task: Literal["t2i", "edit"],
        name: str | None = None,
        *,
        require: Iterable[Capability] = (),
    ) -> Lease[ImageBackend]:
        """A lease on the ready image backend; release it when the job is over."""
        base = Capability.T2I if task == "t2i" else Capability.EDIT
        entry = await self._ready(task, name, BackendKind.IMAGE, {base, *require})
        return Lease(self, cast(ImageBackend, entry.backend))

    async def _ready(
        self, task: Task, name: str | None, kind: BackendKind, required: set[Capability]
    ) -> _Entry:
        """The selected backend once it is healthy and capable. If it is reconfigured while
        being probed, the new one is checked instead: no lease starts on a retired adapter."""
        while True:
            entry = self._select(task, name, kind)
            await self._ensure_ready(entry, required)
            if self._entries.get(entry.name) is entry:
                return entry

    def validate_image_backend(self, name: str) -> None:
        """Fail now, not mid-turn, if ``name`` is not a usable image backend. Health and
        capabilities are still checked when a tool actually needs it."""
        self._select("t2i", name, BackendKind.IMAGE)

    def offers(self, task: Task) -> bool:
        """Whether a request for ``task`` could be routed at all (a built backend exists).

        Says nothing about health: an engine that is down still "offers" the task, and a
        request then fails with 503 rather than the tool silently disappearing."""
        name = self._settings.default_backend(task)
        entry = self._entries.get(name) if name else None
        return entry is not None and entry.backend is not None

    def _select(self, task: Task, name: str | None, kind: BackendKind) -> _Entry:
        chosen = name or self._settings.default_backend(task)
        if chosen is None:
            raise CapabilityUnavailableError(
                f"no {kind.value} backend is configured for {task}", task=task
            )
        entry = self._entries.get(chosen)
        if entry is None:
            raise CapabilityUnavailableError(f"unknown backend '{chosen}'", backend=chosen)
        if entry.kind is not kind:
            raise CapabilityUnavailableError(
                f"backend '{chosen}' is a {entry.kind.value} backend and cannot serve {task}",
                backend=chosen,
            )
        if entry.backend is None:
            raise CapabilityUnavailableError(
                f"backend '{chosen}' is unavailable: {entry.reason}", backend=chosen
            )
        return entry

    async def _ensure_ready(self, entry: _Entry, required: set[Capability]) -> None:
        assert entry.backend is not None
        health = await self._fresh(entry)
        if health.status is not BackendStatus.OK:
            raise EngineUnavailableError(
                f"backend '{entry.name}' is {health.status.value}: {health.detail}",
                backend=entry.name,
            )
        missing = required - entry.backend.capabilities()
        if missing:
            raise CapabilityUnavailableError(
                f"backend '{entry.name}' lacks {', '.join(sorted(missing))}",
                backend=entry.name,
                missing=sorted(missing),
            )

    # --- reporting ------------------------------------------------------------

    async def describe(self) -> list[BackendInfo]:
        return list(await asyncio.gather(*(self._describe(e) for e in self._entries.values())))

    async def describe_one(self, name: str, max_age_s: float = DESCRIBE_MAX_AGE_S) -> BackendInfo:
        """Like describe(), for one backend, reusing a probe younger than ``max_age_s``:
        screens that poll shouldn't wait on an unreachable engine every time."""
        return await self._describe(self._entries[name], max_age_s)

    async def _describe(self, entry: _Entry, max_age_s: float = 0.0) -> BackendInfo:
        info = BackendInfo(
            name=entry.name,
            type=entry.config.type,
            kind=entry.kind,
            enabled=entry.config.enabled,
            status=BackendStatus.UNAVAILABLE,
            capabilities=[],
            default_for=[
                task for task in _TASKS if self._settings.default_backend(task) == entry.name
            ],
            detail=entry.reason,
            image_defaults=_image_defaults(entry.config, entry.backend),
        )
        if entry.backend is None:
            return info
        recent = (
            entry.health is not None
            and entry.probed_at is not None
            and time.monotonic() - entry.probed_at < max_age_s
        )
        health = entry.health if recent and entry.health is not None else await self._probe(entry)
        return info.model_copy(
            update={
                "model": entry.backend.model,
                "status": health.status,
                "detail": health.detail,
                "latency_ms": health.latency_ms,
                "probed_at": health.probed_at,
                "capabilities": sorted(entry.backend.capabilities()),
            }
        )

    async def aclose(self) -> None:
        """Close every adapter, retired ones still leased included (this is shutdown)."""
        backends = [e.backend for e in self._entries.values() if e.backend is not None]
        backends += self._retired.values()
        self._retired.clear()
        await asyncio.gather(*self._closing, *(_close(b) for b in backends))


async def _close(backend: Backend) -> None:
    try:
        await backend.aclose()
    except Exception:
        log.exception("backend.close_failed", backend=backend.name)


def _adapter_fields(config: BackendConfig) -> dict[str, Any]:
    return config.model_dump(exclude=set(_NOT_ADAPTER_FIELDS))


def _image_defaults(config: BackendConfig, backend: Backend | None) -> ImageDefaultsInfo | None:
    defaults = getattr(config, "defaults", None)
    if defaults is None:
        # An adapter whose defaults live elsewhere (a linked instance's engine) reports them.
        reported = getattr(backend, "image_defaults", None)
        return reported if isinstance(reported, ImageDefaultsInfo) else None
    return ImageDefaultsInfo(
        width=defaults.width,
        height=defaults.height,
        steps=defaults.steps,
        cfg_scale=defaults.cfg_scale,
        sampler=defaults.sampler,
        size_multiple=config.size_multiple,  # type: ignore[union-attr]
    )


def _build(name: str, config: BackendConfig, factories: Mapping[str, BackendFactory]) -> _Entry:
    backend: Backend | None = None
    reason: str | None = None
    if not config.enabled:
        reason = "disabled in config"
    elif config.type not in factories:
        reason = f"no '{config.type}' adapter is available in this build"
    else:
        try:
            backend = factories[config.type](name, config)
        except Exception as exc:
            reason = f"failed to initialise: {exc}"
            log.exception("backend.init_failed", backend=name)
    return _Entry(name, config, backend_kind(config), backend, reason)
