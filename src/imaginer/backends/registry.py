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

An sd-server with an edit card has two faces, one per card (``SdCppBackend.edit_face``):
an edit gets the edit face, with that card's presets and capabilities, and the face
switches the engine to its card when the job runs. An edit with a mask gets the model
card's face instead when only that card takes a mask (FLUX.1 Kontext takes none,
Qwen-Image-2.1 does). A lease on either face counts as one on the engine's adapter.
"""

import asyncio
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from imaginer.backends.base import BackendHealth, ImageBackend, LLMBackend
from imaginer.backends.common import WordsSource, find_preset, preset_loras
from imaginer.backends.diffusers_inproc import DiffusersBackend, missing_dependencies
from imaginer.backends.llamacpp import LlamaCppBackend
from imaginer.backends.sdcpp import CardSwitcher, SdCppBackend
from imaginer.backends.sdcpp import mask_face as sdcpp_mask_face
from imaginer.backends.sdcpp import size_multiple as sdcpp_size_multiple
from imaginer.core.clock import utcnow
from imaginer.core.config import (
    BackendConfig,
    DiffusersBackendConfig,
    ImagePreset,
    LlamaCppBackendConfig,
    LoraRef,
    SdCppBackendConfig,
    Settings,
    backend_kind,
)
from imaginer.core.logging import get_logger
from imaginer.domain.enums import BackendKind, BackendStatus, Capability
from imaginer.domain.errors import (
    CapabilityUnavailableError,
    EngineUnavailableError,
    ImaginerError,
    InvalidRequestError,
)
from imaginer.domain.models import (
    BackendEditInfo,
    BackendInfo,
    ImageDefaultsInfo,
    ImagePresetInfo,
    LoraInfo,
)

log = get_logger(__name__)

HEALTH_TIMEOUT_S = 5.0

Backend = LLMBackend | ImageBackend
BackendFactory = Callable[[str, BackendConfig], Backend]
# Config fields the adapter doesn't read: changing only these keeps the adapter. Of the launch
# settings, sd-server's adapter reads one: the LoRA folder, to hash the LoRAs it applies.
_NOT_ADAPTER_FIELDS = frozenset({"launch"})
_ADAPTER_LAUNCH_FIELDS = frozenset({"lora_model_dir"})
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
    """A hold on one adapter. Release it when done; releasing again does nothing. On an
    engine's edit face, the hold is on the engine's adapter (``owner``), which closes both."""

    def __init__(
        self, registry: "BackendRegistry", backend: B, owner: Backend | None = None
    ) -> None:
        self.backend: B = backend
        self._owner: Backend = owner if owner is not None else backend
        self._registry: BackendRegistry | None = registry
        registry._hold(self._owner)

    def release(self) -> None:
        registry, self._registry = self._registry, None
        if registry is not None:
            registry._unhold(self._owner)


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
        self._lora_words: WordsSource | None = None
        self._switcher: CardSwitcher | None = None

    def set_lora_words(self, words: WordsSource | None) -> None:
        """Where image adapters get a LoRA's trigger words (the LoRA library), for every
        adapter there is and every one built later."""
        self._lora_words = words
        for entry in self._entries.values():
            self._attach(entry)

    def set_switcher(self, switcher: CardSwitcher | None) -> None:
        """What switches an sd-server between its model card and its edit card (the engine
        service), for every adapter there is and every one built later."""
        self._switcher = switcher
        for entry in self._entries.values():
            self._attach(entry)

    def _attach(self, entry: _Entry) -> _Entry:
        for face in _faces(entry.backend):
            if hasattr(face, "lora_words"):
                face.lora_words = self._lora_words
            if isinstance(face, SdCppBackend):
                face.switcher = self._switcher
        return entry

    @classmethod
    def from_settings(
        cls, settings: Settings, factories: Mapping[str, BackendFactory] | None = None
    ) -> "BackendRegistry":
        factories = FACTORIES if factories is None else factories
        entries = [_build(name, config, factories) for name, config in settings.backends.items()]
        return cls(entries, settings, factories)

    def reconfigure(self, name: str, config: BackendConfig) -> bool:
        """Use ``config`` for backend ``name`` from now on (e.g. a new endpoint).

        A change the adapter doesn't read (launch settings) keeps the adapter and its health,
        and so does one it takes as it runs (a Diffusers backend's LoRA folder), which only
        needs a new probe. Otherwise a new adapter is built and the old one retired: turns
        and jobs holding a lease on it keep using it, and it is closed once the last of them
        is done. Returns whether the adapter was replaced.
        """
        old = self._entries[name]
        self._settings.backends[name] = config
        if _adapter_fields(old.config) == _adapter_fields(config):
            old.config = config
            return False
        update = getattr(old.backend, "update_config", None)
        if callable(update) and update(config):
            old.config, old.probed_at = config, None  # its last probe no longer counts
            return False
        self._entries[name] = self._attach(_build(name, config, self._factories))
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
        self._entries[name] = self._attach(_build(name, config, self._factories))

    def rebuild(self, name: str) -> None:
        """Build a new adapter from the same config, because something it was built from
        changed outside the config (a linked instance's address). Leases keep the old one."""
        old = self._entries[name]
        self._entries[name] = self._attach(_build(name, old.config, self._factories))
        if old.backend is not None:
            self._retire(old.backend)

    def config(self, name: str) -> BackendConfig:
        return self._entries[name].config

    def names(self) -> list[str]:
        return list(self._entries)

    # --- leases ---------------------------------------------------------------

    def leased(self, name: str) -> bool:
        """Whether a turn or job holds the backend's adapter (or a replaced one it built)."""
        entry = self._entries.get(name)
        if entry is not None and entry.backend is not None and self._leases.get(id(entry.backend)):
            return True
        return any(b.name == name for b in self._retired.values())

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
        preset: str | None = None,
        mask: bool = False,
    ) -> Lease[ImageBackend]:
        """A lease on the ready image backend; release it when the job is over. ``preset``
        must be one of its presets; with LoRAs to apply it must also offer ``lora``. An edit
        with a ``mask`` may get the engine's model card rather than its edit card."""
        base = Capability.T2I if task == "t2i" else Capability.EDIT
        entry = await self._ready(task, name, BackendKind.IMAGE, {base, *require}, preset, mask)
        face = _face(entry, _route(entry, task, mask))
        return Lease(self, cast(ImageBackend, face), owner=entry.backend)

    async def _ready(
        self,
        task: Task,
        name: str | None,
        kind: BackendKind,
        required: set[Capability],
        preset: str | None = None,
        mask: bool = False,
    ) -> _Entry:
        """The selected backend once it is healthy and capable. If it is reconfigured while
        being probed, the new one is checked instead: no lease starts on a retired adapter."""
        while True:
            entry = self._select(task, name, kind)
            card = _route(entry, task, mask)
            needs = set(required)
            if kind is BackendKind.IMAGE:
                chosen = find_preset(_card_label(entry, card), _presets(entry, card), preset)
                if _loras(entry, chosen, card):
                    needs.add(Capability.LORA)
            await self._ensure_ready(entry, needs, card)
            if self._entries.get(entry.name) is entry:
                return entry

    def validate_image_backend(self, name: str) -> None:
        """Fail now, not mid-turn, if ``name`` is not a usable image backend. Health and
        capabilities are still checked when a tool actually needs it."""
        self._select("t2i", name, BackendKind.IMAGE)

    def validate_image_preset(self, preset: str, name: str | None = None) -> None:
        """Fail now, not mid-turn, if no image backend the message could use has ``preset``:
        ``name``, or else the default backends for t2i and edit."""
        tasks: tuple[Literal["t2i", "edit"], ...] = ("t2i", "edit")
        candidates: dict[str, _Entry]
        if name is not None:
            named = self._select("t2i", name, BackendKind.IMAGE)
            # Either card's: an edit card's presets are for its edits.
            both = {p: v for task in tasks for p, v in _presets(named, task).items()}
            find_preset(name, both, preset)
            return
        candidates = {
            chosen: entry
            for task in tasks
            if (chosen := self._settings.default_backend(task)) is not None
            and (entry := self._entries.get(chosen)) is not None
            and entry.kind is BackendKind.IMAGE
        }
        # A preset of either card counts: the edit card's are for edits.
        offered = {p for e in candidates.values() for task in tasks for p in _presets(e, task)}
        if preset not in offered:
            known = sorted(offered)
            where = ", ".join(sorted(candidates)) or "no image backend"
            raise InvalidRequestError(
                f"no image preset '{preset}' on {where} (presets: {', '.join(known) or 'none'})",
                preset=preset,
            )

    def image_preset(
        self, task: Literal["t2i", "edit"], name: str | None, preset: str, *, mask: bool = False
    ) -> ImagePreset | None:
        """The preset as the card a ``task`` call would run has it (for an edit with a
        ``mask``, maybe the model card); None if it doesn't."""
        try:
            entry = self._select(task, name, BackendKind.IMAGE)
            return find_preset("", _presets(entry, _route(entry, task, mask)), preset)
        except ImaginerError:
            return None

    def offers(self, task: Task) -> bool:
        """Whether a request for ``task`` could be routed at all: a built backend exists, and
        its model could do it (an edit-only model makes no images from text).

        Says nothing about health: an engine that is down still "offers" the task, and a
        request then fails with 503 rather than the tool silently disappearing."""
        name = self._settings.default_backend(task)
        entry = self._entries.get(name) if name else None
        if entry is None or entry.backend is None:
            return False
        if task == "chat":
            return True
        declared = getattr(_face(entry, task), "declared", None)
        base = Capability.T2I if task == "t2i" else Capability.EDIT
        return declared is None or base in declared

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

    async def _ensure_ready(
        self, entry: _Entry, required: set[Capability], task: Task = "chat"
    ) -> None:
        assert entry.backend is not None
        health = await self._fresh(entry)
        # An engine switching between its cards is down for a while: work waits for it at
        # the engine's gate rather than failing.
        switching = getattr(entry.backend, "loading", None) is not None
        if health.status is not BackendStatus.OK and not switching:
            raise EngineUnavailableError(
                f"backend '{entry.name}' is {health.status.value}: {health.detail}",
                backend=entry.name,
            )
        missing = required - _face(entry, task).capabilities()
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
            image_presets=_preset_infos(_presets(entry)),
        )
        if entry.backend is None:
            return info
        recent = (
            entry.health is not None
            and entry.probed_at is not None
            and time.monotonic() - entry.probed_at < max_age_s
        )
        health = entry.health if recent and entry.health is not None else await self._probe(entry)
        capabilities = set(entry.backend.capabilities())
        edit = _face(entry, "edit")
        if edit is not entry.backend:
            capabilities |= edit.capabilities() & {Capability.EDIT, Capability.MULTI_REF}
        return info.model_copy(
            update={
                "model": entry.backend.model,
                "status": health.status,
                "detail": health.detail,
                "latency_ms": health.latency_ms,
                "probed_at": health.probed_at,
                "capabilities": sorted(capabilities),
                "available_loras": getattr(entry.backend, "available_loras", None),
                "edit": self._edit_info(entry) if entry.kind is BackendKind.IMAGE else None,
                "loading": getattr(entry.backend, "loading", None),
            }
        )

    def _edit_info(self, entry: _Entry) -> BackendEditInfo | None:
        """How the backend edits: its edit card's settings, or its own."""
        face = _face(entry, "edit")
        caps = face.capabilities()
        if Capability.EDIT not in caps and Capability.EDIT not in getattr(face, "declared", caps):
            return None
        config = _face_config(entry, "edit")
        card = getattr(entry.config, "edit_card", None) if face is not entry.backend else None
        # An edit card that takes no mask leaves masked edits to the model card, if it can.
        masked_on_model_card = _route(entry, "edit", mask=True) == "t2i"
        mask_card = getattr(entry.config, "model_card", None) if masked_on_model_card else None
        return BackendEditInfo(
            card=card,
            card_name=self._card_name(card),
            capabilities=sorted(caps),
            max_ref_images=getattr(face, "max_ref_images", None),
            mask=masked_on_model_card or bool(getattr(face, "takes_mask", True)),
            mask_card=mask_card,
            mask_card_name=self._card_name(mask_card),
            image_defaults=_image_defaults(config, face),
            image_presets=_preset_infos(_presets(entry, "edit")),
        )

    def _card_name(self, card: str | None) -> str | None:
        return self._switcher.card_name(card) if card and self._switcher else None

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
    fields = config.model_dump(exclude=set(_NOT_ADAPTER_FIELDS))
    launch = getattr(config, "launch", None)
    read = set(_ADAPTER_LAUNCH_FIELDS)
    fields["launch"] = launch.model_dump(include=read, exclude_none=True) if launch else {}
    edit_view = getattr(config, "edit_view", None)  # excluded from dumps, read all the same
    if edit_view is not None:
        fields["edit_view"] = _adapter_fields(edit_view)
    return fields


def _faces(backend: Backend | None) -> list[Backend]:
    """An adapter, and its edit face if it has one."""
    if backend is None:
        return []
    edit = getattr(backend, "edit_face", None)
    return [backend, edit] if edit is not None else [backend]


def _face(entry: _Entry, task: Task) -> Backend:
    """What serves ``task`` on the backend: an edit gets an sd-server's edit face."""
    assert entry.backend is not None
    edit = getattr(entry.backend, "edit_face", None)
    return edit if task == "edit" and edit is not None else entry.backend


def _route(entry: _Entry, task: Task, mask: bool = False) -> Task:
    """Whose card serves a request, named by the task that card is for: an edit with a mask
    runs on an engine's model card ("t2i") when its edit card takes no mask and the model
    card edits with one."""
    if task == "edit" and mask and isinstance(entry.config, SdCppBackendConfig):
        return "t2i" if sdcpp_mask_face(entry.config) == "main" else task
    return task


def _face_config(entry: _Entry, task: Task) -> BackendConfig:
    """The config ``task`` runs with: for an edit on an engine with an edit card, the card's."""
    edit_view = getattr(entry.config, "edit_view", None)
    return edit_view if task == "edit" and edit_view is not None else entry.config


def _card_label(entry: _Entry, task: Task) -> str:
    """How errors name what a task runs on: the backend, and for an edit card that too."""
    if task == "edit" and getattr(entry.config, "edit_view", None) is not None:
        return f"{entry.name} (its edit card)"
    return entry.name


def _presets(entry: _Entry, task: Task = "t2i") -> dict[str, ImagePreset]:
    """Its configured presets (an edit card's, for an edit), or for an engine a linked
    instance shares, the presets that instance reports."""
    presets = getattr(_face_config(entry, task), "presets", None)
    if isinstance(presets, dict):
        return presets
    reported = getattr(entry.backend, "image_presets", None)
    return reported if isinstance(reported, dict) else {}


def _preset_infos(presets: Mapping[str, ImagePreset]) -> list[ImagePresetInfo]:
    return [
        ImagePresetInfo(
            name=key,
            description=p.description,
            loras=_lora_infos(p.loras) if p.loras is not None else None,
            steps=p.steps,
            cfg_scale=p.cfg_scale,
            sampler=p.sampler,
        )
        for key, p in presets.items()
    ]


def _loras(entry: _Entry, preset: ImagePreset | None, task: Task = "t2i") -> list[LoraRef]:
    """The LoRAs a request with ``preset`` applies on this backend."""
    defaults = getattr(_face_config(entry, task), "defaults", None)
    if defaults is not None:
        return preset_loras(defaults, preset)
    if preset is not None and preset.loras is not None:
        return list(preset.loras)
    reported = getattr(entry.backend, "image_defaults", None)  # a linked instance's engine
    if not isinstance(reported, ImageDefaultsInfo):
        return []
    return [LoraRef(path=x.path, weight=x.weight) for x in reported.loras]


def _lora_infos(loras: Iterable[LoraRef]) -> list[LoraInfo]:
    return [LoraInfo(path=x.path, weight=x.weight) for x in loras]


def _image_defaults(config: BackendConfig, backend: Backend | None) -> ImageDefaultsInfo | None:
    defaults = getattr(config, "defaults", None)
    if defaults is None:
        # An adapter whose defaults live elsewhere (a linked instance's engine) reports them.
        reported = getattr(backend, "image_defaults", None)
        return reported if isinstance(reported, ImageDefaultsInfo) else None
    if isinstance(config, SdCppBackendConfig):
        multiple = sdcpp_size_multiple(config)
    else:
        multiple = getattr(config, "size_multiple", None) or sdcpp_size_multiple(None)
    return ImageDefaultsInfo(
        width=defaults.width,
        height=defaults.height,
        steps=defaults.steps,
        cfg_scale=defaults.cfg_scale,
        sampler=defaults.sampler,
        size_multiple=multiple,
        loras=_lora_infos(defaults.loras),
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
