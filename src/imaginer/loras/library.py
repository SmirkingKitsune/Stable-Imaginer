"""The LoRA library: one list of every LoRA the image engines can apply, and what the user
adds to each.

**Where LoRAs come from.** The LoRA folders on this machine (an sd-server launched here,
a Diffusers backend; engines can share one folder), and the LoRAs an engine only lists by
name: a remote sd-server, an engine a linked instance shares. A file here is known by its
SHA-256 (hashed once per size and modification time, and kept in ``lora_file``), so the
same file in two folders is one LoRA, and what the user adds follows it through renames
and moves. One an engine only lists by name is known by the engine and that name.

**What the user adds** (``lora``): a display name, trigger words (each on or off), notes,
tags, a pin, a cover image, a default weight, and a call per model on whether it works
there. The display name is only shown: prompts and presets keep naming the file's path.

**Trigger words.** Until the user edits them, a LoRA's words are the ones its file's
metadata names, else its Hub card's. Image adapters ask ``words_for`` when an image uses a
LoRA, and add the words the prompt doesn't have yet.

**Compatibility.** For every model the image engines run, a first guess from the file
(``files.py``: the share of its modules the model has, at the same width; the model it says
it was made for; a lineage the user names, for models whose weights can't be read here),
and the user's call, which wins. A LoRA's engines list it in pickers when it works with
their model ("?" when untested); not when it doesn't.

**Usage** comes from the LoRAs each image job records (with each file's SHA-256 since
LoRAs were hashed; by engine and path before that).

**Moving and deleting** act on the files here, and rewrite the presets and defaults that
name the file by its path (``config/<profile>.local.yaml``, like the Engines screen).
"""

import asyncio
import contextlib
import hashlib
import os
import re
import time
from collections import Counter
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import sqlalchemy as sa
from PIL import UnidentifiedImageError

from imaginer.backends import diffusers_inproc
from imaginer.backends.common import expand_path, file_sha256, lora_matches, scan_loras
from imaginer.backends.registry import BackendRegistry
from imaginer.core.clock import utcnow
from imaginer.core.config import (
    BackendConfig,
    DiffusersBackendConfig,
    LinkedBackendConfig,
    LoraRef,
    SdCppBackendConfig,
    Settings,
    backend_kind,
)
from imaginer.core.logging import get_logger
from imaginer.db.engine import Database
from imaginer.db.models import (
    Artifact,
    Conversation,
    GenerationJob,
    JobOutput,
    LoraFile,
    LoraRecord,
)
from imaginer.domain import mentions
from imaginer.domain import models as dto
from imaginer.domain.enums import BackendKind, BackendStatus, JobStatus
from imaginer.domain.errors import (
    ConflictError,
    ImaginerError,
    InvalidRequestError,
    NotFoundError,
)
from imaginer.engines import launch
from imaginer.engines.service import EngineService
from imaginer.hub.client import hub_cache_dir
from imaginer.hub.service import HubService
from imaginer.image_models import recipes
from imaginer.loras import files
from imaginer.services import covers, mappers
from imaginer.services.artifacts import ArtifactService
from imaginer.services.covers import COVER_MAX_BYTES

log = get_logger(__name__)

State = Literal["ok", "maybe", "no", "unknown"]
_RANK: dict[str, int] = {"ok": 0, "maybe": 1, "unknown": 2, "no": 3}
RECENT_IMAGES = 24
USED_IN = 24
_CHUNK = 500
_REPO = re.compile(r"^[A-Za-z0-9][\w.-]{0,95}$")


@dataclass
class _Copy:
    root: Path  # the LoRA folder, resolved
    shown_root: str  # as it is set
    path: str  # relative to the folder, with "/"
    file: Path
    size: int
    mtime_ns: int
    engines: list[str]


@dataclass
class _Entry:
    id: str
    sha256: str | None = None
    copies: list[_Copy] = field(default_factory=list)
    listed: dict[str, str] = field(default_factory=dict)  # engine -> path as it lists it
    origin: str | None = None
    record: dict[str, Any] | None = None
    source: dict[str, Any] | None = None
    facts: files.LoraFacts = files.NO_FACTS
    hashed_at: datetime | None = None

    @property
    def path(self) -> str:
        if self.copies:
            return self.copies[0].path
        return next(iter(self.listed.values()), "")

    def rec(self, key: str, default: Any = None) -> Any:
        value = (self.record or {}).get(key)
        return default if value is None else value


@dataclass
class _Model:
    key: str
    name: str
    quants: list[str] = field(default_factory=list)
    engines: list[str] = field(default_factory=list)
    places: list[str] = field(default_factory=list)  # "sdcpp", "sdcpp on nas"
    loaded: bool = False
    modules: files.Modules | None = None
    remote: str | None = None  # where it runs when that isn't here
    group: str | None = None  # its family's LoRA group (families.yaml), when known

    @property
    def where(self) -> str:
        parts = ["/".join(self.quants), ", ".join(self.places)]
        if self.engines:
            parts.append("loaded" if self.loaded else "not loaded")
        else:
            parts.append("no engine runs it")
        return " · ".join(p for p in parts if p)


@dataclass
class _State:
    entries: dict[str, _Entry]
    models: dict[str, _Model]
    engine_models: dict[str, str]  # engine -> model key
    catalogs: dict[str, dict[str, dict[str, Any]]]  # engine -> its listed paths
    roots: list[str]


@dataclass
class _Conv:
    title: str | None
    images: int = 0
    last: datetime | None = None
    weight: float | None = None
    jobs: list[tuple[datetime, str]] = field(default_factory=list)  # (finished, job id)


@dataclass
class _Use:
    images: int = 0
    last: datetime | None = None
    convs: dict[str, _Conv] = field(default_factory=dict)


def listed_id(engine: str, path: str) -> str:
    """The id of a LoRA an engine only lists by name."""
    return "l-" + hashlib.sha256(f"{engine}\n{path}".encode()).hexdigest()[:40]


def _file_key(path: Path) -> str:
    return os.path.normcase(str(path))


class LoraLibrary:
    def __init__(
        self,
        settings: Settings,
        db: Database,
        registry: BackendRegistry,
        engines: EngineService,
        artifacts: ArtifactService,
        hub: HubService,
    ) -> None:
        self._settings = settings
        self._db = db
        self._registry = registry
        self._engines = engines
        self._artifacts = artifacts
        self._hub = hub
        self._hashing = asyncio.Lock()
        self._changing = asyncio.Lock()  # moves, deletes and preset edits, one at a time
        # Trigger words by SHA-256, for the image adapters (read from worker threads).
        self._record_words: dict[str, tuple[str, ...]] = {}
        self._source_words: dict[str, tuple[str, ...]] = {}

    @property
    def covers_dir(self) -> Path:
        return self._settings.data_root / "loras" / "covers"

    async def start(self) -> None:
        """Load the trigger words the user set, so images use them before the library is
        first opened."""
        async with self._db.session() as session, session.begin():
            records = list((await session.execute(sa.select(LoraRecord))).scalars())
            sources = list(
                (
                    await session.execute(sa.select(LoraFile).where(LoraFile.source.is_not(None)))
                ).scalars()
            )
        self._record_words = {
            r.id: _on_words(r.triggers) for r in records if r.triggers is not None
        }
        self._source_words = {
            f.sha256: _source_words(f.source) for f in sources if f.sha256 and f.source
        }

    # --- trigger words for the adapters --------------------------------------------------

    def words_for(self, path: Path) -> Sequence[str] | None:
        """The trigger words to add for a LoRA file: the user's, else its metadata's, else
        its Hub card's. Blocking (it may hash the file): the adapters call it in a thread."""
        try:
            digest = file_sha256(path)
        except OSError:
            return None
        if digest in self._record_words:
            return self._record_words[digest]
        facts = files.lora_facts(path)
        if facts.words:
            return facts.words
        return self._source_words.get(digest)

    # --- the Hub ---------------------------------------------------------------------------

    async def note_source(self, path: Path, source: dict[str, Any]) -> None:
        """Remember where a LoRA the Hub is installing comes from (and its card's words)."""
        key = _file_key(expand_path(path).resolve())
        async with self._db.session() as session, session.begin():
            row = await session.get(LoraFile, key)
            if row is None:
                session.add(LoraFile(path=key, source=source))
            else:
                row.source = source

    # --- what the LLM is told ----------------------------------------------------------------

    async def briefs(
        self, wanted: Collection[mentions.LoraKey]
    ) -> dict[mentions.LoraKey, mentions.LoraBrief]:
        """What the LLM is told of the LoRAs messages pick: each one's display name, the
        trigger words the image engine adds, its tags and notes. Each is named as the
        message names it, on the image backend the message picked, else on the default
        ones; one none of them lists is left out."""
        if not wanted:
            return {}
        local: dict[str, dict[str, Path]] = {}  # engine -> {listed path: file}
        for (_, _, engines), listed, file in await asyncio.to_thread(_scan, self._folders()):
            for engine, exts in engines:
                if file.suffix.lower() in exts:
                    local.setdefault(engine, {})[listed] = file
        found: dict[mentions.LoraKey, tuple[str, str, Path | None]] = {}
        for name, backend in wanted:
            for engine in self._image_engines(backend):
                hit = await self._listed(engine, name, local.get(engine))
                if hit is not None:
                    found[(name, backend)] = hit
                    break
        async with self._db.session() as session, session.begin():
            ids = sorted({record_id for record_id, _, _ in found.values()})
            stmt = sa.select(LoraRecord).where(LoraRecord.id.in_(ids))
            records = {r.id: r for r in (await session.execute(stmt)).scalars()}
        out = {}
        for key, (record_id, path, here) in found.items():
            row = records.get(record_id)
            if here is not None:
                chosen = await asyncio.to_thread(self.words_for, here)
                words = tuple(chosen) if chosen is not None else ()
            else:  # the instance that shares the engine adds its own; only the user's are known
                words = _on_words(row.triggers) if row is not None else ()
            out[key] = mentions.LoraBrief(
                name=(row.name if row is not None else None) or PurePosixPath(path).stem,
                words=words,
                tags=tuple(row.tags or ()) if row is not None else (),
                notes=row.notes if row is not None else None,
            )
        return out

    def _image_engines(self, backend: str | None) -> list[str]:
        if backend is not None:
            return [backend]
        names = (self._settings.default_backend(t) for t in ("t2i", "edit"))
        return list(dict.fromkeys(n for n in names if n is not None))

    async def _listed(
        self, engine: str, name: str, files_here: dict[str, Path] | None
    ) -> tuple[str, str, Path | None] | None:
        """The LoRA ``name`` names on ``engine``: (its library id, its path as listed, its
        file when it is on this machine)."""
        if files_here is not None:
            paths = lora_matches(name, {p: {"name": PurePosixPath(p).stem} for p in files_here})
            if len(paths) != 1:
                return None
            try:
                digest = await asyncio.to_thread(file_sha256, files_here[paths[0]])
            except OSError:
                return None
            return digest, paths[0], files_here[paths[0]]
        if engine not in self._registry.names():
            return None
        info = await self._registry.describe_one(engine)
        listed = {p: {"name": PurePosixPath(p).stem} for p in info.available_loras or []}
        paths = lora_matches(name, listed)
        return (listed_id(engine, paths[0]), paths[0], None) if len(paths) == 1 else None

    # --- reading ---------------------------------------------------------------------------

    async def overview(self) -> dto.LoraLibrary:
        state = await self._state()
        usage = await self._usage(state)
        items = [self._item(e, state, usage) for e in state.entries.values()]
        tags = Counter(t for i in items for t in i.tags)
        return dto.LoraLibrary(
            items=items,
            models=[self._model_view(m) for m in state.models.values()],
            tags=[t for t, _ in tags.most_common()],
            folders=sorted({i.folder for i in items if i.local}),
            roots=state.roots,
            total_bytes=sum(i.size_bytes or 0 for i in items if i.local),
            local_count=sum(1 for i in items if i.local),
            origins=dict(Counter(i.origin for i in items if i.origin)),
        )

    async def get(self, lora_id: str) -> dto.LoraDetail:
        state = await self._state()
        entry = _require(state, lora_id)
        usage = await self._usage(state, detail=entry)
        item = self._item(entry, state, usage)
        use = self._use_of(entry, usage)
        convs = sorted(use.convs.items(), key=lambda kv: kv[1].last or utcnow(), reverse=True)
        latest = await self._latest_images(convs)
        recent = await self._recent_images(use)
        base = _base_of(entry)
        return dto.LoraDetail(
            **item.model_dump(),
            compat_models=[self._model_compat(entry, m) for m in _models_in_order(state, entry)],
            lineage=entry.rec("lineage", ""),
            lineage_default=files.model_name(base) if base else "",
            presets=self._preset_uses(entry, state),
            used_in=[
                dto.LoraUseImage(
                    conversation_id=cid,
                    title=c.title,
                    images=c.images,
                    weight=c.weight,
                    last_used_at=c.last or utcnow(),
                    artifact=latest.get(cid),
                )
                for cid, c in convs[:USED_IN]
            ],
            recent_images=recent,
            files=[
                dto.LoraFileInfo(
                    path=str(c.file),
                    root=c.shown_root,
                    engines=c.engines,
                    size_bytes=c.size,
                    hashed_at=entry.hashed_at,
                )
                for c in entry.copies
            ],
            format=entry.facts.format,
            rank=entry.facts.rank,
            base=files.model_name(base) if base else None,
            triggers_from=_words_from(entry),
            file_triggers=list(_seed_words(entry)),
        )

    def cover_file(self, lora_id: str) -> Path | None:
        if not re.fullmatch(r"[0-9a-z-]{1,64}", lora_id):
            return None
        path = self.covers_dir / f"{lora_id}.webp"
        return path if path.is_file() else None

    # --- what the user adds ------------------------------------------------------------------

    async def update(self, lora_id: str, body: dto.LoraUpdate) -> dto.LoraDetail:
        state = await self._state()
        entry = _require(state, lora_id)
        fields = body.model_fields_set
        async with self._db.session() as session, session.begin():
            row = await self._row(session, entry)
            if "name" in fields:
                text = " ".join((body.name or "").split())
                row.name = text or None
            if "default_weight" in fields:
                row.default_weight = body.default_weight
            if body.reset_triggers:
                row.triggers = None
            elif body.triggers is not None:
                row.triggers = _clean_triggers(body.triggers)
            if "notes" in fields:
                row.notes = (body.notes or "").strip() or None
            if body.tags is not None:
                row.tags = _clean_tags(body.tags)
            if body.pinned is not None:
                row.pinned = body.pinned
            if body.compat is not None:
                calls = dict(row.compat or {})
                for key, call in body.compat.items():
                    model = state.models.get(key)
                    if call == "auto":
                        calls.pop(key, None)
                    else:
                        name = model.name if model else calls.get(key, {}).get("name", key)
                        calls[key] = {"call": call, "name": name}
                row.compat = calls or None
            if "lineage" in fields:
                row.lineage = " ".join((body.lineage or "").split()) or None
            row.updated_at = utcnow()
            if entry.sha256:
                if row.triggers is not None:
                    self._record_words[entry.sha256] = _on_words(row.triggers)
                else:
                    self._record_words.pop(entry.sha256, None)
        return await self.get(lora_id)

    async def set_cover(self, lora_id: str, body: dto.LoraCoverSet) -> dto.LoraDetail:
        state = await self._state()
        entry = _require(state, lora_id)
        if body.artifact_id:
            artifact, data = await self._artifacts.read_bytes(body.artifact_id)
            title = await self._title_of(artifact.conversation_id)
            cover: dict[str, Any] = {
                "source": " · ".join(x for x in (artifact.label, title) if x),
                "conversation_id": artifact.conversation_id,
                "artifact_id": artifact.id,
            }
        elif body.hub:
            source = _source_of(entry)
            if source is None:
                raise InvalidRequestError("this LoRA didn't come from the Hub: it has no card")
            card = await self._hub.card(source["repo"], source.get("revision") or "main")
            if not card.images:
                raise InvalidRequestError(f"{source['repo']}'s card shows no image")
            data = await self._hub.fetch(
                source["repo"], card.sha or "main", card.images[0], COVER_MAX_BYTES
            )
            cover = {"source": f"Hugging Face · {source['repo']}"}
        else:
            raise InvalidRequestError("say which image: artifact_id, or hub")
        try:
            webp, width, height = await asyncio.to_thread(covers.thumbnail, data)
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise InvalidRequestError(f"that image can't be read: {exc}") from exc
        await asyncio.to_thread(covers.write_atomic, self.covers_dir / f"{entry.id}.webp", webp)
        cover |= {"width": width, "height": height, "v": int(time.time() * 1000)}
        async with self._db.session() as session, session.begin():
            row = await self._row(session, entry)
            row.cover = cover
            row.updated_at = utcnow()
        return await self.get(lora_id)

    async def clear_cover(self, lora_id: str) -> dto.LoraDetail:
        state = await self._state()
        entry = _require(state, lora_id)
        async with self._db.session() as session, session.begin():
            row = await self._row(session, entry)
            row.cover = None
            row.updated_at = utcnow()
        with contextlib.suppress(OSError):
            (self.covers_dir / f"{entry.id}.webp").unlink()
        return await self.get(lora_id)

    async def set_preset(self, lora_id: str, body: dto.LoraPresetSet) -> dto.LoraDetail:
        """Put the LoRA in one of an engine's presets at a weight, or take it out; or make a
        preset with it (on top of the defaults' LoRAs)."""
        state = await self._state()
        entry = _require(state, lora_id)
        config = self._settings.backends.get(body.engine)
        if not isinstance(config, SdCppBackendConfig | DiffusersBackendConfig):
            raise InvalidRequestError(f"{body.engine} has no presets here")
        path = entry.listed.get(body.engine)
        if path is None:
            raise InvalidRequestError(f"{body.engine} doesn't list this LoRA")
        catalog = state.catalogs.get(body.engine, {path: {}})
        if body.create:
            if body.preset in config.presets:
                raise ConflictError(f"{body.engine} already has a preset named {body.preset}")
            if body.weight is None:
                raise InvalidRequestError("a new preset needs the LoRA's weight")
            loras = [*_refs(config.defaults.loras), {"path": path, "weight": body.weight}]
            patch: dict[str, Any] = {"description": body.description, "loras": loras}
        else:
            preset = config.presets.get(body.preset)
            if preset is None:
                raise NotFoundError(f"{body.engine} has no preset named {body.preset}")
            refs = preset.loras if preset.loras is not None else config.defaults.loras
            loras, placed = [], False
            for ref in refs:
                if lora_matches(ref.path, catalog) == [path]:
                    if body.weight is not None and not placed:
                        loras.append({"path": ref.path, "weight": body.weight})
                        placed = True
                    continue
                loras.append({"path": ref.path, "weight": ref.weight})
            if body.weight is not None and not placed:
                loras.append({"path": path, "weight": body.weight})
            patch = {"loras": loras}
        async with self._changing:
            await self._engines.update(body.engine, dto.EngineUpdate(presets={body.preset: patch}))
        return await self.get(lora_id)

    # --- files -------------------------------------------------------------------------------

    async def move(self, lora_id: str, folder: str) -> dto.LoraDetail:
        """Move the LoRA's files into ``folder`` of their LoRA folders, and point the
        presets and defaults that name the old path at the new one."""
        target = _clean_folder(folder)
        async with self._changing:
            state = await self._state()
            entry = _require(state, lora_id)
            if not entry.copies:
                raise InvalidRequestError("only a LoRA whose file is on this machine can move")
            for copy in entry.copies:
                name = PurePosixPath(copy.path).name
                new = f"{target}/{name}" if target else name
                if new == copy.path:
                    continue
                dest = copy.root / new
                await asyncio.to_thread(_move_file, copy.file, dest, copy.root)
                await self._moved(copy.file, dest)
                for engine in copy.engines:
                    await self._rewrite(engine, state, copy.path, new)
            await self._relist(entry)
        return await self.get(lora_id)

    async def delete(self, lora_id: str) -> None:
        """Delete the LoRA's files here, and take it out of the presets and defaults that
        name it. What the user added stays, for a file with the same SHA-256 later; past
        images keep their record."""
        async with self._changing:
            state = await self._state()
            entry = _require(state, lora_id)
            if not entry.copies:
                raise InvalidRequestError("only a LoRA whose file is on this machine can go")
            for copy in entry.copies:
                try:
                    await asyncio.to_thread(_delete_file, copy.file, copy.root)
                except OSError as exc:
                    raise ConflictError(
                        f"can't delete {copy.file}: {exc.strerror or exc} (in use by an engine?)"
                    ) from exc
                async with self._db.session() as session, session.begin():
                    await session.execute(
                        sa.delete(LoraFile).where(LoraFile.path == _file_key(copy.file))
                    )
                for engine in copy.engines:
                    await self._rewrite(engine, state, copy.path, None)
            await self._relist(entry)

    async def _moved(self, old: Path, new: Path) -> None:
        stat = await asyncio.to_thread(new.stat)
        resolved = await asyncio.to_thread(new.resolve)
        async with self._db.session() as session, session.begin():
            row = await session.get(LoraFile, _file_key(old))
            if row is None:
                return
            fresh = LoraFile(
                path=_file_key(resolved),
                size_bytes=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
                sha256=row.sha256,
                source=row.source,
                hashed_at=row.hashed_at,
            )
            await session.delete(row)
            await session.flush()
            await session.merge(fresh)

    async def _rewrite(self, engine: str, state: _State, old: str, new: str | None) -> None:
        """Point an engine's presets and defaults that name ``old`` at ``new``, or (None)
        take them out. A move only changes the ones that name it by its path: its file name
        still names it."""
        config = self._settings.backends.get(engine)
        if not isinstance(config, SdCppBackendConfig | DiffusersBackendConfig):
            return
        catalog = state.catalogs.get(engine, {old: {}})

        def names_it(ref: LoraRef) -> bool:
            if new is not None:
                return ref.path.replace("\\", "/") == old
            return lora_matches(ref.path, catalog) == [old]

        def rewritten(refs: Iterable[LoraRef]) -> list[dict[str, Any]] | None:
            out, changed = [], False
            for ref in refs:
                if names_it(ref):
                    changed = True
                    if new is not None:
                        out.append({"path": new, "weight": ref.weight})
                    continue
                out.append({"path": ref.path, "weight": ref.weight})
            return out if changed else None

        presets = {}
        for name, preset in config.presets.items():
            if preset.loras is not None and (loras := rewritten(preset.loras)) is not None:
                presets[name] = {"loras": loras}
        defaults = rewritten(config.defaults.loras)
        if not presets and defaults is None:
            return
        update = dto.EngineUpdate(
            presets=presets or None,
            default_loras=[dto.LoraPick(**d) for d in defaults] if defaults is not None else None,
        )
        try:
            await self._engines.update(engine, update)
        except ImaginerError as exc:
            log.warning("loras.rewrite_failed", engine=engine, error=exc.detail)

    async def _relist(self, entry: _Entry) -> None:
        """Have the engines that list the LoRA list their folders again."""
        for engine in {e for c in entry.copies for e in c.engines}:
            with contextlib.suppress(Exception):
                await self._registry.probe(engine)

    # --- building the library ------------------------------------------------------------------

    async def _state(self) -> _State:
        folders = self._folders()
        found = await asyncio.to_thread(_scan, folders)
        hashes = await self._hashes([f for _, _, f in found])
        entries: dict[str, _Entry] = {}
        catalogs: dict[str, dict[str, dict[str, Any]]] = {}
        for (root, shown, engines), path, file in found:
            if _file_key(file) not in hashes:
                continue  # gone, or unreadable, since the scan
            digest, hashed_at, source, size, mtime = hashes[_file_key(file)]
            entry = entries.setdefault(digest, _Entry(id=digest, sha256=digest))
            usable = [e for e, exts in engines if file.suffix.lower() in exts]
            entry.copies.append(_Copy(root, shown, path, file, size, mtime, usable))
            entry.hashed_at = entry.hashed_at or hashed_at
            entry.source = entry.source or source or _inferred_source(path)
            for engine in usable:
                entry.listed.setdefault(engine, path)
                catalogs.setdefault(engine, {})[path] = {"name": PurePosixPath(path).stem}
        infos = await self._infos()
        for name, info in infos.items():
            if name in catalogs or not info.available_loras:
                continue
            config = self._settings.backends[name]
            origin = config.link if isinstance(config, LinkedBackendConfig) else name
            for path in info.available_loras:
                entry = entries.setdefault(
                    listed_id(name, path), _Entry(id=listed_id(name, path), origin=origin)
                )
                entry.listed[name] = path
                catalogs.setdefault(name, {})[path] = {"name": PurePosixPath(path).stem}
        records = await self._records(list(entries))
        for entry in entries.values():
            entry.record = records.get(entry.id)
            if entry.copies:
                entry.facts = await asyncio.to_thread(files.lora_facts, entry.copies[0].file)
        models, engine_models = await self._models(infos, records.values())
        self._refresh_words(entries.values())
        roots = sorted({shown for (_, shown, _), _, _ in found} | {str(r[1]) for r in folders})
        return _State(entries, models, engine_models, catalogs, roots)

    def _folders(self) -> list[tuple[Path, str, list[tuple[str, frozenset[str]]]]]:
        """Each LoRA folder of an enabled image engine here: (resolved, as set, [(engine,
        the file types it lists)])."""
        by_root: dict[str, tuple[Path, str, list[tuple[str, frozenset[str]]]]] = {}
        for name, config in self._settings.backends.items():
            if not config.enabled:
                continue
            if isinstance(config, DiffusersBackendConfig) and config.lora_dir is not None:
                folder, exts = config.lora_dir, diffusers_inproc.LORA_EXTENSIONS
            elif (
                isinstance(config, SdCppBackendConfig)
                and config.launch is not None
                and config.launch.lora_model_dir is not None
            ):
                folder, exts = config.launch.lora_model_dir, launch.LORA_EXTENSIONS
            else:
                continue
            try:
                root = expand_path(folder).resolve()
            except OSError:
                continue
            here = by_root.setdefault(_file_key(root), (root, str(folder), []))
            here[2].append((name, exts))
        return list(by_root.values())

    async def _hashes(
        self, found: list[Path]
    ) -> dict[str, tuple[str, datetime | None, dict[str, Any] | None, int, int]]:
        """Each file's SHA-256 (from ``lora_file`` while its size and time are the same),
        when it was hashed, its source, size and time."""
        async with self._hashing:
            keys = [_file_key(f) for f in found]
            rows: dict[str, LoraFile] = {}
            async with self._db.session() as session, session.begin():
                for i in range(0, len(keys), _CHUNK):
                    stmt = sa.select(LoraFile).where(LoraFile.path.in_(keys[i : i + _CHUNK]))
                    rows |= {r.path: r for r in (await session.execute(stmt)).scalars()}
            out = {}
            fresh: list[LoraFile] = []
            for file, key in zip(found, keys, strict=True):
                try:
                    stat = await asyncio.to_thread(file.stat)
                except OSError:
                    continue
                row = rows.get(key)
                if (
                    row is not None
                    and row.sha256
                    and row.size_bytes == stat.st_size
                    and row.mtime_ns == stat.st_mtime_ns
                ):
                    out[key] = (
                        row.sha256,
                        row.hashed_at,
                        row.source,
                        stat.st_size,
                        stat.st_mtime_ns,
                    )
                    continue
                try:
                    digest = await asyncio.to_thread(file_sha256, file)
                except OSError:
                    continue
                now = utcnow()
                source = row.source if row is not None else None
                fresh.append(
                    LoraFile(
                        path=key,
                        size_bytes=stat.st_size,
                        mtime_ns=stat.st_mtime_ns,
                        sha256=digest,
                        source=source,
                        hashed_at=now,
                    )
                )
                out[key] = (digest, now, source, stat.st_size, stat.st_mtime_ns)
            if fresh:
                async with self._db.session() as session, session.begin():
                    for row in fresh:
                        await session.merge(row)
                for row in fresh:
                    if row.source and row.sha256:
                        self._source_words[row.sha256] = _source_words(row.source)
            return out

    async def _infos(self) -> dict[str, dto.BackendInfo]:
        """The enabled image engines as the registry last saw them."""
        names = [
            n
            for n, c in self._settings.backends.items()
            if c.enabled and backend_kind(c) is BackendKind.IMAGE and n in self._registry.names()
        ]
        infos = await asyncio.gather(
            *(self._registry.describe_one(n) for n in names), return_exceptions=True
        )
        return {n: i for n, i in zip(names, infos, strict=True) if isinstance(i, dto.BackendInfo)}

    async def _records(self, ids: list[str]) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        async with self._db.session() as session, session.begin():
            for i in range(0, len(ids), _CHUNK):
                stmt = sa.select(LoraRecord).where(LoraRecord.id.in_(ids[i : i + _CHUNK]))
                for row in (await session.execute(stmt)).scalars():
                    out[row.id] = _record_dict(row)
            # Calls on models no engine runs any more still show.
            stmt = sa.select(LoraRecord).where(LoraRecord.compat.is_not(None))
            for row in (await session.execute(stmt)).scalars():
                out.setdefault(row.id, _record_dict(row))
        return out

    async def _row(self, session: Any, entry: _Entry) -> LoraRecord:
        row = await session.get(LoraRecord, entry.id)
        if row is None:
            now = utcnow()
            row = LoraRecord(id=entry.id, pinned=False, created_at=now, updated_at=now)
            session.add(row)
        return row  # type: ignore[no-any-return]

    def _refresh_words(self, entries: Iterable[_Entry]) -> None:
        for entry in entries:
            if entry.sha256 is None:
                continue
            triggers = entry.rec("triggers")
            if triggers is not None:
                self._record_words[entry.sha256] = _on_words(triggers)
            else:
                self._record_words.pop(entry.sha256, None)

    async def _models(
        self, infos: dict[str, dto.BackendInfo], records: Iterable[dict[str, Any]]
    ) -> tuple[dict[str, _Model], dict[str, str]]:
        models: dict[str, _Model] = {}
        engine_models: dict[str, str] = {}
        for name, info in infos.items():
            config = self._settings.backends[name]
            text, modules, quant, remote = await self._model_of(config, info)
            if not text:
                continue
            key = files.model_key(text)
            model = models.setdefault(key, _Model(key=key, name=files.model_name(text)))
            family = recipes.family(getattr(config, "family", None))
            if family is not None and family.lora_group:
                model.group = model.group or family.lora_group
            if model.name.islower():  # "Qwen-Image-2.1" reads better than a file's lower case
                model.name = files.model_name(text)
            if quant and quant not in model.quants:
                model.quants.append(quant)
            model.engines.append(name)
            model.places.append(f"{name} on {remote}" if remote else name)
            model.loaded = model.loaded or info.status is BackendStatus.OK
            model.modules = model.modules or modules
            if model.modules is None and remote:
                model.remote = model.remote or remote
            engine_models[name] = key
        for record in records:
            for key, call in (record.get("compat") or {}).items():
                if key not in models and isinstance(call, dict):
                    models[key] = _Model(key=key, name=str(call.get("name") or key))
        return models, engine_models

    async def _model_of(
        self, config: BackendConfig, info: dto.BackendInfo
    ) -> tuple[str | None, files.Modules | None, str | None, str | None]:
        """The model an engine runs: (its name as set, its modules if they can be read
        here, its quantization, where it runs when that isn't here)."""
        if isinstance(config, SdCppBackendConfig):
            # Its own weights: a checkpoint (SD 1.x, SDXL), else its diffusion model.
            file = None
            if config.launch is not None:
                file = config.launch.model or config.launch.diffusion_model
            if file is not None:
                path = expand_path(file)
                modules = await asyncio.to_thread(files.model_file_modules, path)
                return str(file), modules, files.model_quant(str(file)), None
            host, _ = launch.endpoint(config)
            model = info.model if info.model and info.model != "unknown" else None
            where = None if launch.is_loopback(host) else host
            return model, None, files.model_quant(model) if model else None, where
        if isinstance(config, DiffusersBackendConfig):
            folder = await asyncio.to_thread(_pipeline_folder, config.model_id, self._settings)
            modules = await asyncio.to_thread(files.pipeline_modules, folder) if folder else None
            return config.model_id, modules, None, None
        if isinstance(config, LinkedBackendConfig):
            return (
                info.model,
                None,
                files.model_quant(info.model) if info.model else None,
                (config.link),
            )
        return None, None, None, None

    # --- usage -------------------------------------------------------------------------------

    async def _usage(self, state: _State, detail: _Entry | None = None) -> dict[str, _Use]:
        """Images made with each LoRA: by "sha:<SHA-256>" when the job recorded it, else by
        "<engine>\\n<path>"."""
        stmt = (
            sa.select(
                GenerationJob.id,
                GenerationJob.conversation_id,
                GenerationJob.backend,
                GenerationJob.params_json,
                GenerationJob.finished_at,
                Conversation.title,
            )
            .join(Conversation, Conversation.id == GenerationJob.conversation_id)
            .where(
                Conversation.deleted_at.is_(None),
                GenerationJob.status == JobStatus.SUCCEEDED,
                sa.cast(GenerationJob.params_json, sa.Text).like('%"loras"%'),
            )
        )
        async with self._db.session() as session, session.begin():
            rows = list((await session.execute(stmt)).all())
            counts: dict[str, int] = {}
            ids = [r[0] for r in rows]
            for i in range(0, len(ids), _CHUNK):
                count = (
                    sa.select(JobOutput.job_id, sa.func.count())
                    .where(JobOutput.job_id.in_(ids[i : i + _CHUNK]))
                    .group_by(JobOutput.job_id)
                )
                counts |= {j: int(n) for j, n in (await session.execute(count)).all()}
        usage: dict[str, _Use] = {}
        for job_id, cid, backend, params, finished, title in rows:
            resolved = (params or {}).get("resolved") or {}
            images = counts.get(job_id, 0)
            if not images or finished is None:
                continue
            for lora in resolved.get("loras") or []:
                if not isinstance(lora, dict) or not isinstance(lora.get("path"), str):
                    continue
                key = (
                    f"sha:{lora['sha256']}" if lora.get("sha256") else f"{backend}\n{lora['path']}"
                )
                use = usage.setdefault(key, _Use())
                use.images += images
                use.last = max(use.last or finished, finished)
                conv = use.convs.setdefault(cid, _Conv(title=title))
                conv.images += images
                if conv.last is None or finished >= conv.last:
                    conv.last = finished
                    weight = lora.get("weight")
                    conv.weight = float(weight) if isinstance(weight, int | float) else None
                conv.jobs.append((finished, job_id))
        return usage

    def _use_of(self, entry: _Entry, usage: dict[str, _Use]) -> _Use:
        keys = [f"sha:{entry.sha256}"] if entry.sha256 else []
        keys += [f"{engine}\n{path}" for engine, path in entry.listed.items()]
        total = _Use()
        for key in keys:
            use = usage.get(key)
            if use is None:
                continue
            total.images += use.images
            if use.last is not None:
                total.last = max(total.last or use.last, use.last)
            for cid, conv in use.convs.items():
                mine = total.convs.setdefault(cid, _Conv(title=conv.title))
                mine.images += conv.images
                mine.jobs += conv.jobs
                if conv.last is not None and (mine.last is None or conv.last >= mine.last):
                    mine.last, mine.weight = conv.last, conv.weight
        return total

    async def _latest_images(self, convs: list[tuple[str, _Conv]]) -> dict[str, dto.Artifact]:
        """The newest image made with the LoRA in each conversation shown."""
        newest = {cid: max(c.jobs)[1] for cid, c in convs[:USED_IN] if c.jobs}
        found = await self._outputs(list(newest.values()))
        out = {}
        for cid, job_id in newest.items():
            if found.get(job_id):
                out[cid] = found[job_id][0]
        return out

    async def _recent_images(self, use: _Use) -> list[dto.Artifact]:
        jobs = sorted((j for c in use.convs.values() for j in c.jobs), reverse=True)
        found = await self._outputs([job_id for _, job_id in jobs[:RECENT_IMAGES]])
        out: list[dto.Artifact] = []
        for _, job_id in jobs[:RECENT_IMAGES]:
            out += found.get(job_id, [])
        return out[:RECENT_IMAGES]

    async def _outputs(self, job_ids: list[str]) -> dict[str, list[dto.Artifact]]:
        if not job_ids:
            return {}
        stmt = (
            sa.select(JobOutput.job_id, Artifact)
            .join(Artifact, Artifact.id == JobOutput.artifact_id)
            .where(JobOutput.job_id.in_(job_ids))
            .order_by(JobOutput.job_id, JobOutput.ordinal)
        )
        out: dict[str, list[dto.Artifact]] = {}
        async with self._db.session() as session, session.begin():
            for job_id, row in (await session.execute(stmt)).all():
                out.setdefault(job_id, []).append(mappers.artifact(row))
        return out

    async def _title_of(self, conversation_id: str) -> str | None:
        async with self._db.session() as session, session.begin():
            row = await session.get(Conversation, conversation_id)
            return row.title if row is not None else None

    # --- views ---------------------------------------------------------------------------------

    def _item(self, entry: _Entry, state: _State, usage: dict[str, _Use]) -> dto.LoraItem:
        path = entry.path
        pure = PurePosixPath(path.replace("\\", "/"))
        folder = str(pure.parent) if str(pure.parent) != "." else ""
        engines = []
        for engine, listed in entry.listed.items():
            key = state.engine_models.get(engine)
            model = state.models.get(key) if key else None
            compat: State = self._state_for(entry, model)[0] if model else "unknown"
            engines.append(dto.LoraEngineUse(engine=engine, path=listed, model=key, compat=compat))
        use = self._use_of(entry, usage)
        copy = entry.copies[0] if entry.copies else None
        source = _source_of(entry)
        cover = entry.rec("cover")
        weight = entry.rec("default_weight")
        return dto.LoraItem(
            id=entry.id,
            name=entry.rec("name") or pure.stem,
            custom_name=bool(entry.rec("name")),
            file_name=pure.name,
            path=path,
            folder=folder,
            root=copy.shown_root if copy else None,
            local=copy is not None,
            origin=None if copy else entry.origin,
            size_bytes=copy.size if copy else None,
            sha256=entry.sha256,
            modified_at=datetime.fromtimestamp(copy.mtime_ns / 1e9, UTC) if copy else None,
            engines=engines,
            triggers=_triggers_of(entry),
            default_weight=float(weight) if weight is not None else 1.0,
            tags=list(entry.rec("tags", [])),
            pinned=bool(entry.rec("pinned", False)),
            notes=entry.rec("notes", ""),
            cover=dto.LoraCover(
                url=f"/v1/loras/{entry.id}/cover?v={cover.get('v', 0)}",
                source=str(cover.get("source") or ""),
                width=cover.get("width"),
                height=cover.get("height"),
                conversation_id=cover.get("conversation_id"),
                artifact_id=cover.get("artifact_id"),
            )
            if isinstance(cover, dict)
            else None,
            source=dto.LoraSourceInfo(
                repo=source["repo"],
                revision=source.get("revision"),
                url=f"{self._hub.client.endpoint}/{source['repo']}",
            )
            if source
            else None,
            compat=self._summary(entry, engines),
            uses=use.images,
            conversations=len(use.convs),
            last_used_at=use.last,
        )

    def _summary(self, entry: _Entry, engines: list[dto.LoraEngineUse]) -> dto.LoraCompat:
        if not engines:
            return dto.LoraCompat(state="unknown", text="no engine lists it")
        best = min((e.compat for e in engines), key=lambda s: _RANK[s])
        names = [e.engine for e in engines if e.compat == best]
        on = f"applied on {entry.origin}" if entry.origin and not entry.copies else None
        if best == "ok":
            return dto.LoraCompat(state="ok", text=on or ", ".join(names))
        if best == "maybe":
            return dto.LoraCompat(state="maybe", text=f"untested on {', '.join(names)}")
        if best == "unknown":
            return dto.LoraCompat(state="unknown", text=on or "can't check here")
        base = _base_of(entry)
        label = files.model_name(base) if base else _family_name(entry.facts.family)
        return dto.LoraCompat(
            state="no",
            text=label or "no model here",
            bad=f"wrong base · {label}" if label else "works with no model here",
        )

    def _model_view(self, model: _Model) -> dto.LoraModel:
        return dto.LoraModel(
            key=model.key,
            name=model.name,
            quant="/".join(model.quants) or None,
            engines=model.engines,
            where=model.where,
            loaded=model.loaded,
            checkable=model.modules is not None,
        )

    def _model_compat(self, entry: _Entry, model: _Model) -> dto.LoraModelCompat:
        detected, text = self._detect(entry, model)
        call = (entry.rec("compat", {}).get(model.key) or {}).get("call", "auto")
        state: State = "ok" if call == "works" else "no" if call == "no" else detected
        return dto.LoraModelCompat(
            key=model.key,
            name=model.name,
            where=model.where,
            loaded=model.loaded,
            detected=detected,
            detected_text=text,
            call=call if call in {"works", "no"} else "auto",
            state=state,
        )

    def _state_for(self, entry: _Entry, model: _Model) -> tuple[State, str]:
        call = (entry.rec("compat", {}).get(model.key) or {}).get("call")
        if call == "works":
            return "ok", "your call"
        if call == "no":
            return "no", "your call"
        return self._detect(entry, model)

    def _detect(self, entry: _Entry, model: _Model) -> tuple[State, str]:
        """A first guess at whether the LoRA works with a model: (state, why)."""
        facts = entry.facts
        base = _base_of(entry)
        made_for = base is not None and files.model_key(base) == model.key
        lineage = entry.rec("lineage") or (files.model_name(base) if base else "")
        group = lora_group(base, facts.family)
        same_family = group is not None and group == model.group
        other_family = group is not None and model.group is not None and group != model.group
        if model.modules is None or not facts.modules:
            if made_for:
                return "ok", "made for it · keys not checked"
            if same_family:
                return "ok", f"made for {_group_name(group)} · keys not checked"
            if other_family:
                return "no", f"made for {_group_name(group)}"
            if lineage and files.lineage_matches(lineage, model.name):
                return "maybe", "same lineage, untested"
            if model.modules is None:
                return "unknown", f"can't check on {model.remote}" if model.remote else (
                    "weights not on this machine"
                )
            return "unknown", "the LoRA's keys can't be read"
        total = len(facts.modules)
        hits = files.matched(facts.modules, model.modules)
        # A LoRA for this model's architecture finds (nearly) all its modules; one for
        # another finds none, or only the few layers two architectures happen to share.
        if hits * 2 < total:
            if same_family:
                # A checkpoint's keys and a LoRA's (kohya, Diffusers) are named differently.
                return "maybe", f"made for {_group_name(group)} · {hits}/{total} keys match"
            return "no", f"{hits}/{total} keys"
        if hits * 10 < total * 9:
            return "maybe", f"{hits}/{total} keys match"
        if base is not None and not made_for and not same_family:
            return "maybe", f"keys match, made for {files.model_name(base)}"
        return "ok", f"{hits}/{total} keys"

    def _preset_uses(self, entry: _Entry, state: _State) -> list[dto.LoraPresetUse]:
        out = []
        for engine, path in entry.listed.items():
            config = self._settings.backends.get(engine)
            if not isinstance(config, SdCppBackendConfig | DiffusersBackendConfig):
                continue
            catalog = state.catalogs.get(engine, {path: {}})
            for name, preset in config.presets.items():
                refs = preset.loras if preset.loras is not None else config.defaults.loras
                mine = [r for r in refs if lora_matches(r.path, catalog) == [path]]
                others = [
                    PurePosixPath(r.path.replace("\\", "/")).stem
                    + ("" if r.weight == 1 else f" \N{MULTIPLICATION SIGN}{r.weight:g}")
                    for r in refs
                    if r not in mine
                ]
                meta = [
                    f"with {', '.join(others)}" if others else "",
                    f"{preset.steps} steps" if preset.steps else "",
                    f"cfg {preset.cfg_scale:g}" if preset.cfg_scale is not None else "",
                ]
                out.append(
                    dto.LoraPresetUse(
                        engine=engine,
                        name=name,
                        description=preset.description,
                        included=bool(mine),
                        weight=mine[-1].weight if mine else None,
                        meta=" · ".join(m for m in meta if m),
                    )
                )
        return out


# --- helpers ---------------------------------------------------------------------------------


def _require(state: _State, lora_id: str) -> _Entry:
    entry = state.entries.get(lora_id)
    if entry is None:
        raise NotFoundError(f"no LoRA {lora_id}: its file may have moved or gone")
    return entry


def _record_dict(row: LoraRecord) -> dict[str, Any]:
    return {
        "name": row.name,
        "notes": row.notes,
        "default_weight": row.default_weight,
        "tags": row.tags,
        "triggers": row.triggers,
        "pinned": row.pinned,
        "cover": row.cover,
        "compat": row.compat,
        "lineage": row.lineage,
    }


def _on_words(triggers: list[dict[str, Any]] | None) -> tuple[str, ...]:
    return tuple(
        str(t["text"])
        for t in triggers or []
        if isinstance(t, dict) and t.get("on") and t.get("text")
    )


def _source_words(source: dict[str, Any] | None) -> tuple[str, ...]:
    words = (source or {}).get("words")
    return files.split_words(words) if isinstance(words, str) else ()


def _seed_words(entry: _Entry) -> tuple[str, ...]:
    """The words a LoRA has until the user edits them: its file's, else its Hub card's."""
    return entry.facts.words or _source_words(entry.source)


def _words_from(entry: _Entry) -> str | None:
    if entry.rec("triggers") is not None:
        return None
    if entry.facts.words:
        return f"the file's metadata ({entry.facts.words_from})"
    if _source_words(entry.source):
        return "the Hub card"
    return None


def _triggers_of(entry: _Entry) -> list[dto.LoraTrigger]:
    saved = entry.rec("triggers")
    if saved is not None:
        return [
            dto.LoraTrigger(text=str(t["text"]), on=bool(t.get("on", True)))
            for t in saved
            if isinstance(t, dict) and t.get("text")
        ]
    return [dto.LoraTrigger(text=w) for w in _seed_words(entry)]


def _clean_triggers(triggers: list[dto.LoraTrigger]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen = set()
    for t in triggers:
        text = " ".join(t.text.split())
        if text and text.casefold() not in seen:
            seen.add(text.casefold())
            out.append({"text": text, "on": t.on})
    return out


def _clean_tags(tags: list[str]) -> list[str]:
    out: list[str] = []
    for tag in tags:
        text = " ".join(tag.split())[:32]
        if text and text.casefold() not in {t.casefold() for t in out}:
            out.append(text)
    return out


def _clean_folder(folder: str) -> str:
    text = folder.strip().replace("\\", "/").strip("/")
    if not text:
        return ""
    parts = text.split("/")
    if any(p in {"", ".", ".."} or ":" in p for p in parts) or len(text) > 400:
        raise InvalidRequestError(f"{folder!r} isn't a folder inside the LoRA folder")
    return "/".join(parts)


def _base_of(entry: _Entry) -> str | None:
    """The model the LoRA says it was made for: its Hub card's, else its metadata's."""
    bases = (entry.source or {}).get("base_models") or []
    if bases and isinstance(bases[0], str):
        return str(bases[0])
    return entry.facts.base


_FAMILIES = (
    ("sdxl", "SDXL"), ("stable-diffusion-xl", "SDXL"), ("sd3", "SD 3"), ("flux", "Flux"),
    ("sd_v1", "SD 1.5"), ("sd1", "SD 1.5"), ("v1-5", "SD 1.5"), ("sd_v2", "SD 2"),
    ("pony", "Pony"), ("illustrious", "Illustrious"), ("hunyuan", "HunyuanDiT"),
    ("qwen", "Qwen-Image"), ("wan", "Wan"),
)  # fmt: skip


# Base-model names LoRA trainers write (kohya's ss_base_model_version, modelspec's
# architecture, a Hub card's base_model), by the LoRA group of the families they fit.
_GROUPS = (
    ("stable-diffusion-xl", "sdxl"), ("sdxl", "sdxl"), ("sd_xl", "sdxl"), ("pony", "sdxl"),
    ("illustrious", "sdxl"), ("noobai", "sdxl"), ("sd3.5-large", "sd35-large"),
    ("sd3.5_large", "sd35-large"), ("stable-diffusion-3.5-large", "sd35-large"),
    ("sd3.5-medium", "sd35-medium"), ("stable-diffusion-3.5-medium", "sd35-medium"),
    ("sd3", "sd3"), ("stable-diffusion-3", "sd3"), ("sd_v2", "sd2"), ("sd2", "sd2"),
    ("stable-diffusion-v2", "sd2"), ("stable-diffusion-2", "sd2"), ("sd_v1", "sd1"),
    ("sd1", "sd1"), ("v1-5", "sd1"), ("stable-diffusion-v1", "sd1"),
    ("flux.2-klein-9b", "flux2-klein-9b"), ("flux-2-klein-9b", "flux2-klein-9b"),
    ("flux.2-klein", "flux2-klein-4b"), ("flux-2-klein", "flux2-klein-4b"),
    ("flux.2-dev", "flux2-dev"), ("flux-2-dev", "flux2-dev"), ("flux.1", "flux1"),
    ("flux-1", "flux1"), ("flux1", "flux1"), ("chroma1-radiance", "chroma-radiance"),
    ("chroma", "chroma"), ("qwen-image-2.1", "qwen-image-21"),
    ("qwen_image_2.1", "qwen-image-21"), ("qwen-image", "qwen-image"),
    ("qwen_image", "qwen-image"), ("z-image-turbo", "z-image-turbo"),
    ("z_image_turbo", "z-image-turbo"), ("z-image", "z-image"),
)  # fmt: skip


def lora_group(base: str | None, family: str | None) -> str | None:
    """The LoRA group (families.yaml's lora_group) of the model a LoRA was made for, from
    the base model it names (or the family its metadata names)."""
    for text in (base, family):
        if not text:
            continue
        name = text.casefold().replace(" ", "-")
        for needle, group in _GROUPS:
            if needle in name:
                return group
    return None


def _group_name(group: str | None) -> str:
    names = {f.lora_group: f.group for f in recipes.catalog().families() if f.lora_group}
    if group is None:
        return "another model"
    return names.get(group) or group


def _family_name(family: str | None) -> str | None:
    """A model family as people call it: "sdxl_base_v1-0" is SDXL."""
    if not family:
        return None
    text = family.casefold()
    for needle, name in _FAMILIES:
        if needle in text:
            return name
    return family.split("/")[0]


def _source_of(entry: _Entry) -> dict[str, Any] | None:
    source = entry.source
    if isinstance(source, dict) and isinstance(source.get("repo"), str):
        return source
    return None


def _inferred_source(path: str) -> dict[str, Any] | None:
    """A Hub install's repo, from its place in the LoRA folder (``<owner>/<name>/...``)."""
    parts = path.split("/")
    if len(parts) >= 3 and _REPO.match(parts[0]) and _REPO.match(parts[1]):
        return {"repo": f"{parts[0]}/{parts[1]}", "inferred": True}
    return None


def _models_in_order(state: _State, entry: _Entry) -> list[_Model]:
    """The models of the LoRA's engines first, then the others."""
    mine = [state.engine_models[e] for e in entry.listed if e in state.engine_models]
    keys = list(dict.fromkeys([*mine, *state.models]))
    return [state.models[k] for k in keys]


def _refs(loras: Iterable[LoraRef]) -> list[dict[str, Any]]:
    return [{"path": r.path, "weight": r.weight} for r in loras]


def _scan(
    folders: list[tuple[Path, str, list[tuple[str, frozenset[str]]]]],
) -> list[tuple[tuple[Path, str, list[tuple[str, frozenset[str]]]], str, Path]]:
    out = []
    for folder in folders:
        root, _, engines = folder
        exts = frozenset().union(*(e for _, e in engines))
        for path in sorted(scan_loras(root, exts) or {}):
            out.append((folder, path, root / path))
    return out


def _pipeline_folder(model_id: str, settings: Settings) -> Path | None:
    """Where a Diffusers model's files are on this machine: its folder, its copy in the
    models folder, or its newest snapshot in the Hugging Face cache."""
    path = expand_path(model_id)
    if path.is_dir():
        return path
    if model_id.count("/") != 1 or ":" in model_id:
        return None
    owner, name = model_id.split("/")
    local = settings.models_dir / owner / name
    if local.is_dir():
        return local
    if not settings.hub.use_hf_cache:
        return None
    snapshots = hub_cache_dir() / f"models--{owner}--{name}" / "snapshots"
    ref = snapshots.parent / "refs" / "main"
    with contextlib.suppress(OSError):
        if ref.is_file():
            pinned = snapshots / ref.read_text(encoding="utf-8").strip()
            if pinned.is_dir():
                return pinned
        found = sorted(
            (p for p in snapshots.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime
        )
        if found:
            return found[-1]
    return None


def _move_file(src: Path, dest: Path, root: Path) -> None:
    resolved = dest.resolve()
    if not resolved.is_relative_to(root):
        raise InvalidRequestError("that folder isn't inside the LoRA folder")
    if resolved.exists():
        raise ConflictError(f"{dest} is already there")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    try:
        src.replace(resolved)
    except OSError as exc:
        raise ConflictError(f"can't move {src.name}: {exc.strerror or exc}") from exc
    _prune(src.parent, root)


def _delete_file(path: Path, root: Path) -> None:
    path.unlink()
    _prune(path.parent, root)


def _prune(folder: Path, root: Path) -> None:
    """Remove the folders a move or delete left empty, up to the LoRA folder."""
    here = folder
    while here != root and here.is_relative_to(root):
        try:
            here.rmdir()
        except OSError:
            return
        here = here.parent


__all__ = ["LoraLibrary", "listed_id"]
