"""The model library: a card per image model the user imports for sd-server.

**What a card is.** One model as sd-server loads it: its family (an sd-server recipe,
``recipes.py``), the file in each of the family's component slots (``--diffusion-model``,
``--vae``, ``--t5xxl``...), the launch arguments that go with it, and its own generation
defaults and presets: what suits SDXL is wrong for FLUX, so they belong to the model, not
the engine. Plus what the user adds: notes, tags, a pin, a cover.

**Running one.** An sd-server engine runs the card its ``model_card`` names. Its config
as it runs (``effective``) is its own with the card's files, family, defaults and presets
over it; the engine's own config, and the profile, keep only the card's id. Using a card
on an engine launched here restarts it with the card's files; on an engine started
elsewhere, the card only says what it runs.

**Components are shared.** A file is named by its path, so two cards (FLUX.1 dev and
schnell) can use one VAE and one T5. The library lists every component file with the
cards that use it, to reuse one rather than download it again.

**Moving in.** An engine set up before the library (its files in its own launch block)
gets a card made from them at startup, with its defaults and presets; nothing it does
changes.
"""

import asyncio
import contextlib
import os
import re
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePath
from typing import TYPE_CHECKING, Any

import sqlalchemy as sa
from PIL import UnidentifiedImageError
from pydantic import ValidationError

from imaginer.backends.common import expand_path
from imaginer.core.clock import utcnow
from imaginer.core.config import (
    BackendConfig,
    ImageDefaults,
    ImagePreset,
    SdCppBackendConfig,
    SdCppLaunch,
    Settings,
)
from imaginer.core.ids import new_id
from imaginer.core.logging import get_logger
from imaginer.db.engine import Database
from imaginer.db.models import Artifact, Conversation, GenerationJob, ImageModelRecord, JobOutput
from imaginer.domain import models as dto
from imaginer.domain.enums import JobStatus
from imaginer.domain.errors import ConflictError, InvalidRequestError, NotFoundError
from imaginer.engines import launch
from imaginer.hub.refs import RefError, parse_ref
from imaginer.image_models import detect as detection
from imaginer.image_models import recipes
from imaginer.loras import files
from imaginer.services import covers, mappers
from imaginer.services.covers import COVER_MAX_BYTES

if TYPE_CHECKING:
    from imaginer.engines.service import EngineService
    from imaginer.hub.service import HubService
    from imaginer.services.artifacts import ArtifactService

log = get_logger(__name__)

RECENT_IMAGES = 24
_CHUNK = 500
_ID = re.compile(r"^[0-9a-z-]{1,64}$")
DEFAULT_SIZE = 1024  # for a family whose recipe names no size


@dataclass(frozen=True, slots=True)
class Card:
    id: str
    name: str
    family: str | None
    components: Mapping[str, str]
    sources: Mapping[str, Mapping[str, Any]]
    vae_format: str | None
    model_args: Mapping[str, str]
    extra_args: tuple[str, ...]
    defaults: ImageDefaults
    presets: Mapping[str, ImagePreset]
    prompt_hint: str | None
    notes: str | None
    tags: tuple[str, ...]
    pinned: bool
    cover: Mapping[str, Any] | None
    created_at: datetime
    updated_at: datetime

    @property
    def main_file(self) -> str | None:
        family = recipes.family(self.family)
        slots = (family.main_slot,) if family is not None else ()
        for slot in (*slots, *recipes.MAIN_SLOTS):
            if self.components.get(slot):
                return self.components[slot]
        return None


def seed_defaults(family_id: str | None) -> ImageDefaults:
    """A new card's defaults: its family's recipe defaults, with our notes over them."""
    family = recipes.family(family_id)
    values = family.seed_defaults() if family is not None else {}
    values.setdefault("width", DEFAULT_SIZE)
    values.setdefault("height", DEFAULT_SIZE)
    known = ImageDefaults.model_fields
    return ImageDefaults.model_validate({k: v for k, v in values.items() if k in known})


def _card(row: ImageModelRecord) -> Card:
    try:
        defaults = (
            ImageDefaults.model_validate(row.defaults)
            if row.defaults
            else seed_defaults(row.family)
        )
    except ValidationError as exc:
        log.warning("models.bad_defaults", card=row.id, error=str(exc))
        defaults = seed_defaults(row.family)
    presets: dict[str, ImagePreset] = {}
    for name, raw in (row.presets or {}).items():
        try:
            presets[str(name)] = ImagePreset.model_validate(raw)
        except ValidationError as exc:
            log.warning("models.bad_preset", card=row.id, preset=name, error=str(exc))
    launch_cfg = row.launch or {}
    return Card(
        id=row.id,
        name=row.name,
        family=row.family,
        components={str(k): str(v) for k, v in (row.components or {}).items() if v},
        sources={str(k): dict(v) for k, v in (row.sources or {}).items() if isinstance(v, dict)},
        vae_format=launch_cfg.get("vae_format") or None,
        model_args={str(k): str(v) for k, v in (launch_cfg.get("model_args") or {}).items()},
        extra_args=tuple(str(a) for a in launch_cfg.get("extra_args") or []),
        defaults=defaults,
        presets=presets,
        prompt_hint=row.prompt_hint,
        notes=row.notes,
        tags=tuple(row.tags or ()),
        pinned=row.pinned,
        cover=row.cover,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


@dataclass(slots=True)
class _Use:
    images: int = 0
    last: datetime | None = None


class ModelLibrary:
    def __init__(self, settings: Settings, db: Database) -> None:
        self._settings = settings
        self._db = db
        self._cards: dict[str, Card] = {}
        self.covers_dir = settings.data_root / "model_covers"
        self._engines: EngineService | None = None
        self._hub: HubService | None = None
        self._artifacts: ArtifactService | None = None
        self._lock = asyncio.Lock()  # one change to the cards at a time

    async def load(self) -> None:
        """Read every card. Done before the registry is built, so engines start out
        running their cards."""
        async with self._db.session() as session, session.begin():
            rows = (await session.execute(sa.select(ImageModelRecord))).scalars().all()
            self._cards = {row.id: _card(row) for row in rows}

    def attach(
        self, *, engines: "EngineService", hub: "HubService", artifacts: "ArtifactService"
    ) -> None:
        self._engines = engines
        self._hub = hub
        self._artifacts = artifacts
        engines.on_card_settings = self._engine_settings
        hub.on_model_install = self.install_component
        hub.model_config = self.card_config
        hub.model_components = self.component_list

    @property
    def engines(self) -> "EngineService":
        assert self._engines is not None, "attach() first"
        return self._engines

    def card(self, card_id: str | None) -> Card | None:
        return self._cards.get(card_id) if card_id else None

    def card_config(self, card_id: str) -> SdCppBackendConfig:
        """A card as an sd-server engine config (its family and files), for matching a Hub
        repo's files to its slots."""
        card = self._require(card_id)
        files_in = {slot: Path(path) for slot, path in card.components.items()}
        return SdCppBackendConfig(
            type="sdcpp",
            base_url="http://127.0.0.1:1",
            model_card=card.id,
            family=card.family,
            launch=SdCppLaunch.model_validate(files_in),
        )

    def component_list(self) -> list[tuple[str, str, str]]:
        """(card id, slot, path) of every card's component."""
        return [
            (card.id, slot, path)
            for card in self._cards.values()
            for slot, path in card.components.items()
        ]

    # --- what engines run --------------------------------------------------------------------

    def effective(self, name: str, config: BackendConfig) -> BackendConfig:
        """An engine's config as it runs: for an sd-server with a model card, the card's
        files, family, defaults and presets over its own. A card that is gone leaves the
        engine as it is configured."""
        if not isinstance(config, SdCppBackendConfig) or not config.model_card:
            return config
        card = self._cards.get(config.model_card)
        if card is None:
            log.warning("models.card_missing", engine=name, card=config.model_card)
            return config
        update: dict[str, Any] = {
            "family": card.family,
            "defaults": card.defaults,
            "presets": dict(card.presets),
        }
        if config.launch is not None:
            slots = {
                slot: Path(card.components[slot]) if card.components.get(slot) else None
                for slot in recipes.SLOT_IDS
            }
            update["launch"] = config.launch.model_copy(
                update={
                    **slots,
                    "vae_format": card.vae_format,
                    "model_args": dict(card.model_args),
                    "extra_args": [*config.launch.extra_args, *card.extra_args],
                }
            )
        return config.model_copy(update=update)

    def engine_card(self, engine: str) -> Card | None:
        """The card an engine runs, if it runs one."""
        if self._engines is None:
            return None
        with contextlib.suppress(NotFoundError):
            config = self._engines.own_config(engine)
            return self.card(getattr(config, "model_card", None))
        return None

    def prompt_hint(self, engine: str) -> str | None:
        """What the chat model should know to prompt the model an engine runs (tags for
        SD 1.x/SDXL, JSON for Ideogram 4); None for natural language."""
        if self._engines is None:
            return None
        card = self.engine_card(engine)
        if card is not None and card.prompt_hint:
            return card.prompt_hint
        with contextlib.suppress(NotFoundError):
            config = self._engines.effective(engine)
            family = recipes.family(getattr(config, "family", None))
            return family.prompt_hint() if family is not None else None
        return None

    def _sdcpp_engines(self) -> list[tuple[str, SdCppBackendConfig]]:
        out = []
        for name in self.engines.names():
            config = self.engines.own_config(name)
            if isinstance(config, SdCppBackendConfig):
                out.append((name, config))
        return out

    def _users(self, card_id: str) -> list[str]:
        return [n for n, c in self._sdcpp_engines() if c.model_card == card_id]

    def _refresh_engines(self, card_id: str) -> None:
        for name in self._users(card_id):
            self.engines.refresh(name)

    # --- moving in -----------------------------------------------------------------------------

    async def adopt_engines(self) -> list[str]:
        """Give each sd-server engine set up by hand (files in its launch block, no card) a
        card made from them, with its defaults, presets and default LoRAs, and point it at
        it. Returns the cards made."""
        made = []
        for name, config in self._sdcpp_engines():
            if config.model_card or config.launch is None:
                continue
            components = {
                slot: str(getattr(config.launch, slot))
                for slot in recipes.SLOT_IDS
                if getattr(config.launch, slot) is not None
            }
            if not components:
                continue
            existing = next(
                (c for c in self._cards.values() if _same(c.components, components)), None
            )
            if existing is None:
                card = await self._adopt(name, config, components)
                made.append(card.id)
            else:
                card = existing
            await self.engines.update(name, dto.EngineUpdate(model_card=card.id))
            log.info("models.adopted", engine=name, card=card.id, name=card.name)
        return made

    async def _adopt(
        self, engine: str, config: SdCppBackendConfig, components: dict[str, str]
    ) -> Card:
        main = components.get("model") or components.get("diffusion_model")
        family_id = config.family
        if family_id is None and main is not None:
            guess = await asyncio.to_thread(
                detection.best, detection.Evidence(path=expand_path(main))
            )
            family_id = guess.family if guess is not None else None
        family = recipes.family(family_id)
        name = family.name if family is not None else files.model_name(main or engine)
        launch_cfg = config.launch
        assert launch_cfg is not None
        model_args = {k: recipes.arg_text(v) for k, v in launch_cfg.model_args.items()}
        now = utcnow()
        row = ImageModelRecord(
            id=new_id(),
            name=name,
            family=family_id,
            components=components,
            sources=None,
            launch=_launch_json(launch_cfg.vae_format, model_args, []),
            defaults=config.defaults.model_dump(mode="json"),
            presets={
                k: p.model_dump(mode="json", exclude_none=True) for k, p in config.presets.items()
            }
            or None,
            notes=f"Made from {engine}'s settings.",
            pinned=False,
            created_at=now,
            updated_at=now,
        )
        async with self._lock, self._db.session() as session, session.begin():
            session.add(row)
            await session.flush()
            card = self._cards[row.id] = _card(row)
        return card

    # --- views ---------------------------------------------------------------------------------

    def families(self, *, hidden: bool = False) -> list[dto.ModelFamily]:
        return [_family_view(f) for f in recipes.catalog().families(hidden=hidden)]

    def family(self, family_id: str) -> dto.ModelFamily:
        family = recipes.family(family_id)
        if family is None:
            raise NotFoundError(f"no model family {family_id!r}")
        return _family_view(family)

    async def overview(self) -> dto.ModelLibraryView:
        cards = sorted(self._cards.values(), key=lambda c: (not c.pinned, c.name.casefold()))
        checks = await asyncio.to_thread(self._checks, cards)
        usage = await self._usage()
        items = [await asyncio.to_thread(self._item, c, checks[c.id], usage) for c in cards]
        used_families = {c.family for c in cards if c.family}
        families = [
            dto.ModelFamilyBrief(id=f.id, name=f.name, group=f.group, hidden=f.hidden)
            for f in recipes.catalog().families(hidden=True)
            if not f.hidden or f.id in used_families
        ]
        shared: dict[str, dto.SharedComponent] = {}
        for card in cards:
            for slot, path in card.components.items():
                key = _norm(path)
                check = next((c for c in checks[card.id] if c.key == slot), None)
                entry = shared.setdefault(
                    key,
                    dto.SharedComponent(
                        path=path,
                        slots=[],
                        cards=[],
                        size_bytes=check.size_bytes if check else None,
                        exists=bool(check and check.exists),
                    ),
                )
                if slot not in entry.slots:
                    entry.slots.append(slot)
                if card.id not in entry.cards:
                    entry.cards.append(card.id)
        engines = [
            dto.ModelLibraryEngine(
                name=name,
                model_card=config.model_card,
                launchable=config.launch is not None,
                running=self.engines.running(name),
            )
            for name, config in self._sdcpp_engines()
        ]
        tags = sorted({t for c in cards for t in c.tags}, key=str.casefold)
        return dto.ModelLibraryView(
            items=items,
            families=families,
            components=sorted(shared.values(), key=lambda s: s.path.casefold()),
            engines=engines,
            tags=tags,
        )

    async def get(self, card_id: str) -> dto.ModelCardDetail:
        card = self._require(card_id)
        checks = (await asyncio.to_thread(self._checks, [card]))[card.id]
        usage = await self._usage()
        item = await asyncio.to_thread(self._item, card, checks, usage)
        family = recipes.family(card.family)
        main = card.main_file
        guesses = (
            await asyncio.to_thread(detection.detect, detection.Evidence(path=expand_path(main)))
            if main
            else []
        )
        return dto.ModelCardDetail(
            **item.model_dump(),
            components=self._components(card, checks),
            launch=dto.ModelLaunch(
                vae_format=card.vae_format,
                model_args=dict(card.model_args),
                extra_args=list(card.extra_args),
            ),
            defaults=card.defaults.model_dump(mode="json"),
            presets={
                k: p.model_dump(mode="json", exclude_none=True) for k, p in card.presets.items()
            },
            prompt_hint=card.prompt_hint,
            family_prompt_hint=family.prompt_hint() if family is not None else None,
            family_note=family.note if family is not None else None,
            capabilities=sorted(family.capabilities()) if family is not None else [],
            detected=[_guess_view(g) for g in guesses[:5]],
            recent_images=await self._recent_images(card.id),
        )

    def cover_file(self, card_id: str) -> Path | None:
        if not _ID.fullmatch(card_id):
            return None
        path = self.covers_dir / f"{card_id}.webp"
        return path if path.is_file() else None

    def _require(self, card_id: str) -> Card:
        card = self._cards.get(card_id)
        if card is None:
            raise NotFoundError(f"no model card {card_id!r}")
        return card

    def _checks(self, cards: Iterable[Card]) -> dict[str, list[launch.FileCheck]]:
        """Each card's component files, checked on disk. Blocking."""
        flags = {slot: (flag, what) for slot, flag, what in recipes.SLOTS}
        out: dict[str, list[launch.FileCheck]] = {}
        for card in cards:
            checks = []
            for slot, path in card.components.items():
                flag, what = flags.get(slot, (slot, slot))
                checks.append(launch.check_file(slot, flag, what, Path(path), engine="sdcpp"))
            out[card.id] = checks
        return out

    def _item(
        self, card: Card, checks: list[launch.FileCheck], usage: dict[str, _Use]
    ) -> dto.ModelCardItem:
        family = recipes.family(card.family)
        filled = [c.key for c in checks]
        missing = family.missing(filled) if family is not None else ["a family"]
        problems = [f"{c.flag} {c.path}: {c.problem}" for c in checks if c.problem]
        if family is not None:
            problems += [
                f"{c.flag}: {family.name} has no {c.what} slot"
                for c in checks
                if c.key not in family.slot_ids
            ]
        use = usage.get(card.id, _Use())
        cover = card.cover
        return dto.ModelCardItem(
            id=card.id,
            name=card.name,
            family=card.family,
            family_name=family.name if family is not None else None,
            group=family.group if family is not None else None,
            main_file=card.main_file,
            size_bytes=sum(c.size_bytes or 0 for c in checks),
            ready=family is not None and not missing and not problems,
            missing=missing,
            problems=problems,
            engines=self._uses(card.id),
            tags=list(card.tags),
            pinned=card.pinned,
            notes=card.notes or "",
            cover=dto.ModelCover(
                url=f"/v1/models/{card.id}/cover?v={cover.get('v', 0)}",
                source=str(cover.get("source") or ""),
                width=cover.get("width"),
                height=cover.get("height"),
                conversation_id=cover.get("conversation_id"),
                artifact_id=cover.get("artifact_id"),
            )
            if cover
            else None,
            uses=use.images,
            last_used_at=use.last,
            created_at=card.created_at,
            updated_at=card.updated_at,
        )

    def _uses(self, card_id: str) -> list[dto.ModelEngineUse]:
        out = []
        for name, config in self._sdcpp_engines():
            if config.model_card != card_id:
                continue
            running = self.engines.running(name)
            stale = False
            if running and config.launch is not None:
                now = self.engines.command(name)
                stale = now is not None and now != self.engines.running_command(name)
            out.append(
                dto.ModelEngineUse(
                    engine=name,
                    launchable=config.launch is not None,
                    running=running,
                    needs_restart=stale,
                )
            )
        return out

    def _components(self, card: Card, checks: list[launch.FileCheck]) -> list[dto.ModelComponent]:
        family = recipes.family(card.family)
        by_slot = {c.key: c for c in checks}
        slots: list[recipes.Slot] = list(family.slots) if family is not None else []
        known = {s.id for s in slots}
        slots += [
            recipes.Slot(slot, False, recipes.SLOT_WHAT.get(slot, slot))
            for slot in recipes.SLOT_IDS
            if slot in by_slot and slot not in known
        ]
        flags = {slot: flag for slot, flag, _ in recipes.SLOTS}
        out = []
        for slot in slots:
            check = by_slot.get(slot.id)
            in_group = family is not None and any(slot.id in g for g in family.required_any)
            path = card.components.get(slot.id)
            out.append(
                dto.ModelComponent(
                    slot=slot.id,
                    flag=flags.get(slot.id, slot.id),
                    what=slot.what,
                    label=slot.label,
                    required=slot.required or in_group,
                    path=path,
                    exists=bool(check and check.exists),
                    size_bytes=check.size_bytes if check else None,
                    format=check.format if check else None,
                    problem=check.problem if check else None,
                    source=dict(card.sources[slot.id]) if slot.id in card.sources else None,
                    shared_with=[
                        other.name
                        for other in self._cards.values()
                        if other.id != card.id
                        and path
                        and any(_norm(p) == _norm(path) for p in other.components.values())
                    ],
                    sources=list(family.sources.get(slot.id, ())) if family is not None else [],
                )
            )
        return out

    # --- usage ---------------------------------------------------------------------------------

    async def _usage(self) -> dict[str, _Use]:
        """Images made with each card, from what each image job recorded."""
        stmt = (
            sa.select(GenerationJob.id, GenerationJob.params_json, GenerationJob.finished_at)
            .join(Conversation, Conversation.id == GenerationJob.conversation_id)
            .where(
                Conversation.deleted_at.is_(None),
                GenerationJob.status == JobStatus.SUCCEEDED,
                sa.cast(GenerationJob.params_json, sa.Text).like('%"model_card"%'),
            )
        )
        async with self._db.session() as session, session.begin():
            rows = list((await session.execute(stmt)).all())
            counts = await _output_counts(session, [r[0] for r in rows])
        usage: dict[str, _Use] = {}
        for job_id, params, finished in rows:
            card_id = (params or {}).get("model_card")
            images = counts.get(job_id, 0)
            if not isinstance(card_id, str) or not images or finished is None:
                continue
            use = usage.setdefault(card_id, _Use())
            use.images += images
            use.last = max(use.last or finished, finished)
        return usage

    async def _recent_images(self, card_id: str) -> list[dto.Artifact]:
        stmt = (
            sa.select(GenerationJob.id, GenerationJob.params_json)
            .join(Conversation, Conversation.id == GenerationJob.conversation_id)
            .where(
                Conversation.deleted_at.is_(None),
                GenerationJob.status == JobStatus.SUCCEEDED,
                sa.cast(GenerationJob.params_json, sa.Text).like(f'%"{card_id}"%'),
            )
            .order_by(GenerationJob.finished_at.desc())
            .limit(RECENT_IMAGES * 2)
        )
        async with self._db.session() as session, session.begin():
            jobs = [
                job_id
                for job_id, params in (await session.execute(stmt)).all()
                if (params or {}).get("model_card") == card_id
            ]
            if not jobs:
                return []
            found = (
                await session.execute(
                    sa.select(JobOutput.job_id, Artifact)
                    .join(Artifact, Artifact.id == JobOutput.artifact_id)
                    .where(JobOutput.job_id.in_(jobs))
                    .order_by(JobOutput.ordinal)
                )
            ).all()
        by_job: dict[str, list[dto.Artifact]] = {}
        for job_id, row in found:
            by_job.setdefault(job_id, []).append(mappers.artifact(row))
        out: list[dto.Artifact] = []
        for job_id in jobs:
            out += by_job.get(job_id, [])
        return out[:RECENT_IMAGES]

    # --- detecting a family --------------------------------------------------------------------

    async def detect(self, body: dto.ModelDetect) -> dto.ModelDetected:
        if body.path:
            path = expand_path(body.path.strip())
            if not await asyncio.to_thread(path.exists):
                raise InvalidRequestError(f"{body.path}: not found")
            guesses = await asyncio.to_thread(detection.detect, detection.Evidence(path=path))
            name_from = path.name
        elif body.ref:
            evidence = await self._ref_evidence(body.ref)
            guesses = await asyncio.to_thread(detection.detect, evidence)
            name_from = evidence.name or evidence.repo or body.ref
        else:
            raise InvalidRequestError("say what to look at: path, or ref")
        best = guesses[0] if guesses and guesses[0].score >= 0.5 else None
        family = recipes.family(best.family) if best is not None else None
        return dto.ModelDetected(
            guesses=[_guess_view(g) for g in guesses[:8]],
            slot=family.main_slot if family is not None else None,
            name=family.name if family is not None else files.model_name(name_from),
        )

    async def locate(self, body: dto.ModelLocate) -> dto.ModelLocated:
        roots = self._search_roots()
        found = await asyncio.to_thread(_find_named, roots, body.name.strip(), body.size_bytes)
        return dto.ModelLocated(paths=[str(p) for p in found], searched=[str(r) for r in roots])

    def _search_roots(self) -> list[Path]:
        """Where model files are kept here: the download folder and Hugging Face cache, each
        card's folders (and the folder above, e.g. D:\\models for D:\\models\\flux), and
        Downloads."""
        out: list[Path] = []
        if self._hub is not None:
            out.append(self._hub.models_dir)
            if self._hub.cache_dir is not None:
                out.append(self._hub.cache_dir)
        for card in self._cards.values():
            for value in card.components.values():
                parent = expand_path(value).parent
                out += [parent, parent.parent]
        out.append(Path.home() / "Downloads")
        home = Path.home()
        roots: list[Path] = []
        for path in out:
            # A drive or the home folder itself is too wide to walk.
            if path == home or path.parent == path or path in roots:
                continue
            if any(path.is_relative_to(r) for r in roots):
                continue
            roots = [r for r in roots if not r.is_relative_to(path)] + [path]
        return roots

    async def _ref_evidence(self, reference: str) -> detection.Evidence:
        if self._hub is None:
            raise InvalidRequestError("Hugging Face isn't available here")
        ref, info, layout = await self._hub.inspect(reference)
        name = None
        if ref.path and PurePath(ref.path).suffix:
            name = PurePath(ref.path).name
        else:
            mains = [v for v in layout.variants if v.component == "diffusion_model"]
            if ref.path:
                mains = [v for v in mains if v.id.startswith(ref.path.rstrip("/") + "/")] or mains
            if mains:
                name = PurePath(mains[0].id).name
        pipeline = layout.pipeline.pipeline_class if layout.pipeline is not None else None
        return detection.Evidence(
            name=name, repo=ref.repo_id, base_models=tuple(info.base_models), pipeline=pipeline
        )

    # --- changing cards ------------------------------------------------------------------------

    async def create(self, body: dto.ModelCreate) -> dto.ModelCardDetail:
        family_id = (body.family or "").strip() or None
        if family_id is not None and recipes.family(family_id) is None:
            raise InvalidRequestError(f"no model family {family_id!r}")
        components = _clean_components(body.components)
        sources: dict[str, dict[str, Any]] = {}
        name = " ".join((body.name or "").split()) or None
        if body.path:
            path = expand_path(body.path.strip())
            detected = await self.detect(dto.ModelDetect(path=str(path)))
            family_id = family_id or _best_family(detected)
            family = recipes.family(family_id)
            slot = (
                family.main_slot if family is not None else await asyncio.to_thread(_slot_of, path)
            )
            components.setdefault(slot, str(path))
            name = name or detected.name
        elif body.ref:
            detected = await self.detect(dto.ModelDetect(ref=body.ref))
            family_id = family_id or _best_family(detected)
            family = recipes.family(family_id)
            try:
                ref = parse_ref(body.ref)
            except RefError as exc:
                raise InvalidRequestError(str(exc)) from exc
            slot = family.main_slot if family is not None else "diffusion_model"
            sources[slot] = {"repo": ref.repo_id, "revision": ref.revision, "path": ref.path}
            name = name or detected.name
        family = recipes.family(family_id)
        if name is None:
            name = family.name if family is not None else "New model"
        now = utcnow()
        row = ImageModelRecord(
            id=new_id(),
            name=name[:120],
            family=family_id,
            components=components,
            sources=sources or None,
            launch=None,
            defaults=seed_defaults(family_id).model_dump(mode="json"),
            presets=None,
            pinned=False,
            created_at=now,
            updated_at=now,
        )
        async with self._lock, self._db.session() as session, session.begin():
            session.add(row)
            await session.flush()
            self._cards[row.id] = _card(row)
        log.info("models.created", card=row.id, family=family_id, name=row.name)
        return await self.get(row.id)

    async def update(self, card_id: str, body: dto.ModelUpdate) -> dto.ModelCardDetail:
        self._require(card_id)
        fields = body.model_fields_set
        async with self._lock, self._db.session() as session, session.begin():
            row = await self._row(session, card_id)
            if "name" in fields and body.name is not None:
                text = " ".join(body.name.split())
                if not text:
                    raise InvalidRequestError("a model card needs a name")
                row.name = text
            if "family" in fields:
                family_id = (body.family or "").strip() or None
                if family_id is not None and recipes.family(family_id) is None:
                    raise InvalidRequestError(f"no model family {family_id!r}")
                row.components = _moved_main(dict(row.components or {}), row.family, family_id)
                row.family = family_id
            if body.components is not None:
                merged = dict(row.components or {})
                for slot, path in body.components.items():
                    if slot not in recipes.SLOT_IDS:
                        raise InvalidRequestError(f"no component slot {slot!r}")
                    text = (path or "").strip()
                    if text:
                        merged[slot] = text
                    else:
                        merged.pop(slot, None)
                        sources = dict(row.sources or {})
                        if sources.pop(slot, None) is not None:
                            row.sources = sources or None
                row.components = merged
            if body.launch is not None:
                current = dict(row.launch or {})
                given = body.launch.model_fields_set
                if "vae_format" in given:
                    current["vae_format"] = body.launch.vae_format or None
                if "model_args" in given:
                    current["model_args"] = {
                        k.strip(): v.strip()
                        for k, v in (body.launch.model_args or {}).items()
                        if k.strip()
                    }
                if "extra_args" in given:
                    current["extra_args"] = list(body.launch.extra_args or [])
                row.launch = current or None
            if body.reset_defaults:
                row.defaults = seed_defaults(row.family).model_dump(mode="json")
            if body.defaults is not None:
                base = dict(row.defaults or seed_defaults(row.family).model_dump(mode="json"))
                try:
                    row.defaults = ImageDefaults.model_validate(base | body.defaults).model_dump(
                        mode="json"
                    )
                except ValidationError as exc:
                    raise InvalidRequestError(_errors(exc)) from exc
            if body.presets is not None:
                row.presets = _merged_presets(dict(row.presets or {}), body.presets)
            if "prompt_hint" in fields:
                row.prompt_hint = (body.prompt_hint or "").strip() or None
            if "notes" in fields:
                row.notes = (body.notes or "").strip() or None
            if body.tags is not None:
                row.tags = _clean_tags(body.tags) or None
            if body.pinned is not None:
                row.pinned = body.pinned
            row.updated_at = utcnow()
            await session.flush()
            self._cards[card_id] = _card(row)
        self._refresh_engines(card_id)
        return await self.get(card_id)

    async def delete(self, card_id: str) -> None:
        """Forget a card (its files stay). Not one an engine runs."""
        self._require(card_id)
        users = self._users(card_id)
        if users:
            raise ConflictError(
                f"{', '.join(users)} runs it: give {'them' if len(users) > 1 else 'it'} "
                "another model first"
            )
        async with self._lock, self._db.session() as session, session.begin():
            row = await session.get(ImageModelRecord, card_id)
            if row is not None:
                await session.delete(row)
            self._cards.pop(card_id, None)
        with contextlib.suppress(OSError):
            (self.covers_dir / f"{card_id}.webp").unlink()

    async def use(self, card_id: str, body: dto.ModelUse) -> dto.ModelCardDetail:
        """Run the card on an sd-server engine. One launched here is restarted with the
        card's files (or started, with ``start``); on one started elsewhere the card only
        says what it runs. Refused while work is using the engine."""
        card = self._require(card_id)
        engines = self.engines
        config = engines.own_config(body.engine)
        if not isinstance(config, SdCppBackendConfig):
            raise InvalidRequestError(f"{body.engine} is not an sd-server engine")
        if engines.busy(body.engine):
            raise ConflictError(f"{body.engine} is making an image: try again when it's done")
        launchable = config.launch is not None
        running = engines.running(body.engine)
        if launchable and (running or body.start):
            item = (await self.get(card_id)).model_dump()
            if not item["ready"]:
                gaps = [*item["missing"], *item["problems"]]
                raise InvalidRequestError(f"{card.name} isn't ready: {'; '.join(gaps)}")
        if running:
            await engines.stop(body.engine)
        await engines.update(body.engine, dto.EngineUpdate(model_card=card_id))
        if launchable and (running or body.start):
            await engines.start(body.engine)
        log.info("models.used", card=card_id, engine=body.engine, restarted=running)
        return await self.get(card_id)

    async def install_component(
        self, card_id: str, slot: str, path: Path, source: dict[str, Any]
    ) -> None:
        """A component the Hub downloaded for a card's slot."""
        if card_id not in self._cards:
            raise NotFoundError(f"no model card {card_id!r}: it was deleted meanwhile")
        if slot not in recipes.SLOT_IDS:
            raise InvalidRequestError(f"no component slot {slot!r}")
        async with self._lock, self._db.session() as session, session.begin():
            row = await self._row(session, card_id)
            row.components = {**(row.components or {}), slot: str(path)}
            row.sources = {**(row.sources or {}), slot: source}
            row.updated_at = utcnow()
            await session.flush()
            self._cards[card_id] = _card(row)
        self._refresh_engines(card_id)

    async def _engine_settings(
        self,
        card_id: str,
        presets: dict[str, dict[str, Any]] | None,
        default_loras: list[dict[str, Any]] | None,
    ) -> None:
        """Presets and default LoRAs set on an engine that runs this card (the LoRA
        library, the Engines screen): they are the card's."""
        self._require(card_id)
        async with self._lock, self._db.session() as session, session.begin():
            row = await self._row(session, card_id)
            if presets is not None:
                row.presets = _merged_presets(dict(row.presets or {}), dict(presets))
            if default_loras is not None:
                base = dict(row.defaults or seed_defaults(row.family).model_dump(mode="json"))
                try:
                    row.defaults = ImageDefaults.model_validate(
                        base | {"loras": default_loras}
                    ).model_dump(mode="json")
                except ValidationError as exc:
                    raise InvalidRequestError(_errors(exc)) from exc
            row.updated_at = utcnow()
            await session.flush()
            self._cards[card_id] = _card(row)
        self._refresh_engines(card_id)

    async def set_cover(self, card_id: str, body: dto.ModelCoverSet) -> dto.ModelCardDetail:
        card = self._require(card_id)
        if body.artifact_id:
            assert self._artifacts is not None
            artifact, data = await self._artifacts.read_bytes(body.artifact_id)
            async with self._db.session() as session, session.begin():
                conv = await session.get(Conversation, artifact.conversation_id)
            title = conv.title if conv is not None else None
            cover: dict[str, Any] = {
                "source": " · ".join(x for x in (artifact.label, title) if x),
                "conversation_id": artifact.conversation_id,
                "artifact_id": artifact.id,
            }
        elif body.hub:
            source = next(
                (
                    card.sources[s]
                    for s in (*recipes.MAIN_SLOTS, *card.sources)
                    if s in card.sources
                ),
                None,
            )
            if source is None or self._hub is None:
                raise InvalidRequestError("this model didn't come from the Hub: it has no card")
            repo = str(source["repo"])
            info = await self._hub.card(repo, str(source.get("revision") or "main"))
            if not info.images:
                raise InvalidRequestError(f"{repo}'s card shows no image")
            data = await self._hub.fetch(repo, info.sha or "main", info.images[0], COVER_MAX_BYTES)
            cover = {"source": f"Hugging Face · {repo}"}
        else:
            raise InvalidRequestError("say which image: artifact_id, or hub")
        try:
            webp, width, height = await asyncio.to_thread(covers.thumbnail, data)
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise InvalidRequestError(f"that image can't be read: {exc}") from exc
        await asyncio.to_thread(covers.write_atomic, self.covers_dir / f"{card_id}.webp", webp)
        cover |= {"width": width, "height": height, "v": int(time.time() * 1000)}
        async with self._lock, self._db.session() as session, session.begin():
            row = await self._row(session, card_id)
            row.cover = cover
            row.updated_at = utcnow()
            await session.flush()
            self._cards[card_id] = _card(row)
        return await self.get(card_id)

    async def clear_cover(self, card_id: str) -> dto.ModelCardDetail:
        self._require(card_id)
        async with self._lock, self._db.session() as session, session.begin():
            row = await self._row(session, card_id)
            row.cover = None
            row.updated_at = utcnow()
            await session.flush()
            self._cards[card_id] = _card(row)
        with contextlib.suppress(OSError):
            (self.covers_dir / f"{card_id}.webp").unlink()
        return await self.get(card_id)

    async def _row(self, session: Any, card_id: str) -> ImageModelRecord:
        row = await session.get(ImageModelRecord, card_id)
        if row is None:
            raise NotFoundError(f"no model card {card_id!r}")
        assert isinstance(row, ImageModelRecord)
        return row


# --- helpers ---------------------------------------------------------------------------------


async def _output_counts(session: Any, job_ids: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for i in range(0, len(job_ids), _CHUNK):
        stmt = (
            sa.select(JobOutput.job_id, sa.func.count())
            .where(JobOutput.job_id.in_(job_ids[i : i + _CHUNK]))
            .group_by(JobOutput.job_id)
        )
        counts |= {j: int(n) for j, n in (await session.execute(stmt)).all()}
    return counts


def _family_view(family: recipes.Family) -> dto.ModelFamily:
    flags = {slot: flag for slot, flag, _ in recipes.SLOTS}
    return dto.ModelFamily(
        id=family.id,
        name=family.name,
        group=family.group,
        hidden=family.hidden,
        description=family.description,
        model_url=family.model_url,
        origin=family.origin,
        slots=[
            dto.ModelSlot(
                id=s.id,
                flag=flags.get(s.id, s.id),
                what=s.what,
                label=s.label,
                required=s.required,
                sources=list(family.sources.get(s.id, ())),
            )
            for s in family.slots
        ],
        required_any=[list(g) for g in family.required_any],
        defaults=dict(family.seed_defaults()),
        size_multiple=family.size_multiple,
        capabilities=sorted(family.capabilities()),
        features=sorted(family.features),
        prompt=family.prompt,
        prompt_hint=family.prompt_hint(),
        note=family.note,
        vae_format=family.vae_format,
        model_args=dict(family.model_args),
    )


def _guess_view(guess: detection.Guess) -> dto.ModelGuess:
    family = recipes.family(guess.family)
    return dto.ModelGuess(
        family=guess.family,
        name=family.name if family is not None else guess.family,
        score=round(guess.score, 3),
        reasons=guess.reasons,
    )


def _best_family(detected: dto.ModelDetected) -> str | None:
    guess = detected.guesses[0] if detected.guesses else None
    return guess.family if guess is not None and guess.score >= 0.5 else None


_LOCATE_DEPTH = 6  # folders below a root
_LOCATE_BUDGET = 200_000  # entries looked at, over every root
_LOCATE_MAX = 10


def _find_named(roots: list[Path], name: str, size: int | None) -> list[Path]:
    """Files called ``name`` (any case) under the roots, same size first. Blocking."""
    want = name.casefold()
    found: list[tuple[bool, Path]] = []
    budget = _LOCATE_BUDGET
    for root in roots:
        stack = [(root, 0)]
        while stack and budget > 0:
            folder, depth = stack.pop()
            try:
                entries = list(os.scandir(folder))
            except OSError:
                continue
            budget -= len(entries)
            for entry in entries:
                try:
                    if entry.is_dir(follow_symlinks=False):
                        if depth < _LOCATE_DEPTH and not entry.name.startswith("."):
                            stack.append((Path(entry.path), depth + 1))
                    elif entry.name.casefold() == want and entry.is_file():
                        exact = size is not None and entry.stat().st_size == size
                        found.append((exact, Path(entry.path)))
                except OSError:
                    continue
    if size is not None and any(exact for exact, _ in found):
        found = [f for f in found if f[0]]
    return [p for _, p in sorted(found, key=lambda f: not f[0])][:_LOCATE_MAX]


def _slot_of(path: Path) -> str:
    """Where a model file of no known family goes: a full checkpoint (it has its text
    encoder and VAE in it) is ``model``, else it is a diffusion model. Blocking."""
    names = files.tensor_names(path) or []
    whole = any(
        n.startswith(("cond_stage_model.", "conditioner.", "first_stage_model.")) for n in names
    )
    return "model" if whole else "diffusion_model"


def _moved_main(components: dict[str, Any], old: str | None, new: str | None) -> dict[str, Any]:
    """A card's components when its family changes: the model's own file moves to the new
    family's slot for it (a checkpoint family loads it as ``model``)."""
    before, after = recipes.family(old), recipes.family(new)
    if after is None:
        return components
    source = before.main_slot if before is not None else None
    target = after.main_slot
    if source and source != target and source in components and target not in components:
        components[target] = components.pop(source)
    return components


def _merged_presets(
    current: dict[str, Any], changes: Mapping[str, dict[str, Any] | None]
) -> dict[str, Any] | None:
    for name, value in changes.items():
        if value is None:
            current.pop(name, None)
            continue
        try:
            preset = ImagePreset.model_validate({**(current.get(name) or {}), **value})
        except ValidationError as exc:
            raise InvalidRequestError(f"preset {name}: {_errors(exc)}") from exc
        current[name] = preset.model_dump(mode="json", exclude_none=True)
    return current or None


def _launch_json(
    vae_format: str | None, model_args: Mapping[str, str], extra_args: list[str]
) -> dict[str, Any] | None:
    data: dict[str, Any] = {}
    if vae_format:
        data["vae_format"] = vae_format
    if model_args:
        data["model_args"] = dict(model_args)
    if extra_args:
        data["extra_args"] = extra_args
    return data or None


def _clean_components(components: Mapping[str, str]) -> dict[str, str]:
    out = {}
    for slot, path in components.items():
        if slot not in recipes.SLOT_IDS:
            raise InvalidRequestError(f"no component slot {slot!r}")
        if path.strip():
            out[slot] = path.strip()
    return out


def _clean_tags(tags: list[str]) -> list[str]:
    out: list[str] = []
    for tag in tags:
        text = " ".join(tag.split())[:32]
        if text and text.casefold() not in {t.casefold() for t in out}:
            out.append(text)
    return out


def _errors(exc: ValidationError) -> str:
    return "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())


def _norm(path: str) -> str:
    return os.path.normcase(os.path.normpath(str(expand_path(path))))


def _same(a: Mapping[str, str], b: Mapping[str, str]) -> bool:
    return {k: _norm(v) for k, v in a.items()} == {k: _norm(v) for k, v in b.items()}


__all__ = ["Card", "ModelLibrary", "seed_defaults"]
