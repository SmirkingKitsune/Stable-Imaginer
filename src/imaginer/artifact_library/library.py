"""The artifact library (docs/artifact-library.md).

Every image of every conversation, on this instance and on linked ones, listed once per
distinct image and newest first, so an image made or uploaded earlier can go into a new
prompt. Using one copies it into that conversation (it gets a label of its own there), so
an artifact still belongs to exactly one conversation.

A pin keeps a copy of an image's bytes in ``<data_root>/artifact_library/``, with a snapshot
of where it came from: it outlives its conversation, and the instance it came from.
"""

import asyncio
import contextlib
import re
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy.exc import IntegrityError

from imaginer.artifact_library import peers
from imaginer.core.config import Settings
from imaginer.core.ids import new_id
from imaginer.core.logging import get_logger
from imaginer.db import models as orm
from imaginer.db.engine import Database
from imaginer.db.repositories import artifact_library as lib_repo
from imaginer.db.repositories import artifacts as art_repo
from imaginer.db.repositories.artifact_library import LibraryQuery, LiveRow
from imaginer.db.types import utcnow
from imaginer.domain import models as dto
from imaginer.domain.enums import ArtifactKind, ArtifactSource
from imaginer.domain.errors import ImaginerError, InvalidRequestError, NotFoundError
from imaginer.links import gallery
from imaginer.links.service import LinkService
from imaginer.links.state import Link
from imaginer.services import mappers
from imaginer.services.artifacts import ArtifactService
from imaginer.services.covers import write_atomic
from imaginer.services.media import EXTENSIONS

log = get_logger(__name__)

LOCAL = "local"
DIRNAME = "artifact_library"
RECHECK_AFTER_S = 5.0  # how often listings ask an unreachable link again (in the background)
_PIN_FILE = re.compile(r"^[0-9a-f-]{36}\.(png|jpg|webp|gif|bmp)$")


@dataclass(frozen=True, slots=True)
class Found:
    """An image a pin or an import starts from."""

    item: dto.ArtifactLibraryItem
    link: Link | None  # where its bytes are; None: here
    metadata: dict[str, Any] | None = None  # its artifact's own (images of this instance)


class ArtifactLibrary:
    def __init__(
        self,
        settings: Settings,
        db: Database,
        artifacts: ArtifactService,
        links: LinkService,
    ) -> None:
        self._db = db
        self._artifacts = artifacts
        self._links = links
        self._root = settings.data_root / DIRNAME
        self._max_bytes = settings.storage.max_upload_bytes
        self._grace_s = settings.storage.orphan_grace_s

    async def start(self) -> None:
        await asyncio.to_thread(self._root.mkdir, parents=True, exist_ok=True)
        await self._sweep()

    async def _sweep(self) -> None:
        """Remove files no pin keeps: an interrupted pin's, or a leftover temp file."""
        async with self._db.session() as session, session.begin():
            kept = await lib_repo.pin_storage_keys(session)
        cutoff = time.time() - self._grace_s

        def scan() -> int:
            removed = 0
            for entry in self._root.iterdir():
                if entry.name in kept or not entry.is_file():
                    continue
                with contextlib.suppress(OSError):
                    if entry.stat().st_mtime < cutoff:
                        entry.unlink()
                        removed += 1
            return removed

        removed = await asyncio.to_thread(scan)
        if removed:
            log.info("artifact_library.swept", files=removed)

    # --- listing ---------------------------------------------------------------------

    async def page(
        self,
        query: LibraryQuery,
        *,
        cursor: str | None,
        limit: int,
        origin: str | None = None,
    ) -> dto.ArtifactLibraryPage:
        """This instance's images and every linked instance's, merged newest first.

        A remote image this instance already has (the same bytes, live or pinned) is left
        out, so a page can come back short while ``next_cursor`` says there is more."""
        links = self._links.all() if self._links.enabled else []
        by_id = {link.id: link for link in links}
        sources = {
            LOCAL: dto.GallerySource(id=LOCAL, name=self._links.name, local=True, status="ok")
        }
        if not query.pinned:  # pins are this instance's own
            for link in links:
                sources[link.id] = dto.GallerySource(
                    id=link.id, name=link.peer_name, local=False, status="ok"
                )
        if origin is not None:
            if origin != LOCAL and origin not in by_id:
                raise NotFoundError(f"no link {origin}")
            sources = {origin: sources[origin]} if origin in sources else {}
        positions = gallery.decode(cursor) if cursor else {}
        wanted = [s for s in sources if not (s in positions and positions[s] is None)]
        fetched = await asyncio.gather(
            *(
                self._local_source(query, positions.get(source), limit)
                if source == LOCAL
                else self._remote_source(by_id[source], query, positions.get(source), limit)
                for source in wanted
            ),
            return_exceptions=True,
        )
        pages: dict[str, gallery.SourcePage[dto.ArtifactLibraryItem]] = {}
        for source, result in zip(wanted, fetched, strict=True):
            if isinstance(result, gallery.SourcePage):
                pages[source] = result
                continue
            if not isinstance(result, Exception):
                raise result  # cancelled
            if not isinstance(result, ImaginerError):
                log.error("artifact_library.source_failed", source=source, error=repr(result))
            sources[source] = _failed(sources[source], result)
            positions[source] = None  # skip it for the rest of this listing
        items, after = gallery.merge(
            pages, positions, limit, sort_key=_by_id, position=_id_position
        )
        if len(sources) > 1:
            items = await self._without_duplicates(items)
        done = all(after.get(s, ("",)) is None for s in sources)
        return dto.ArtifactLibraryPage(
            items=items,
            next_cursor=None if done else gallery.encode(after),
            sources=list(sources.values()),
        )

    async def local_page(
        self,
        query: LibraryQuery,
        *,
        cursor: str | None,
        limit: int,
        for_link: bool = False,
    ) -> dto.Page[dto.ArtifactLibraryItem]:
        """This instance's own images: live ones and pins whose conversation is gone. A
        linked instance sees only pins of this instance's own images, and no pin details."""
        async with self._db.session() as session, session.begin():
            live = await lib_repo.live_page(session, query, after=cursor, limit=limit + 1)
            pins = await lib_repo.pin_cards(
                session,
                query,
                after=cursor,
                limit=limit + 1,
                instance_id=self._links.instance_id if for_link else None,
            )
            rows: list[LiveRow | orm.ArtifactPin] = [*live, *pins]
            rows.sort(key=_row_id, reverse=True)
            has_more = len(rows) > limit
            rows = rows[:limit]
            items = await self._items(session, rows, for_link=for_link)
        return dto.Page[dto.ArtifactLibraryItem](
            items=items, next_cursor=items[-1].id if has_more and items else None
        )

    async def local_item(
        self, *, artifact_id: str | None = None, pin_id: str | None = None, for_link: bool = False
    ) -> dto.ArtifactLibraryItem:
        """One image of this instance, by its artifact or its pin."""
        async with self._db.session() as session, session.begin():
            if artifact_id is not None:
                row = await lib_repo.live_artifact(session, artifact_id)
                if row is None:
                    raise NotFoundError(f"artifact {artifact_id} not found")
                return (await self._items(session, [row], for_link=for_link))[0]
            pin = await lib_repo.get_pin(session, pin_id or "")
            if pin is None or (for_link and pin.source_instance_id != self._links.instance_id):
                raise NotFoundError(f"pin {pin_id} not found")
            return self._pin_item(pin, for_link=for_link)

    async def _local_source(
        self, query: LibraryQuery, position: gallery.Position, limit: int
    ) -> gallery.SourcePage[dto.ArtifactLibraryItem]:
        page = await self.local_page(query, cursor=position[0] if position else None, limit=limit)
        return gallery.SourcePage(items=page.items, has_more=page.next_cursor is not None)

    async def _remote_source(
        self, link: Link, query: LibraryQuery, position: gallery.Position, limit: int
    ) -> gallery.SourcePage[dto.ArtifactLibraryItem]:
        if link.peer is not None and link.peer.gallery == "off":
            raise InvalidRequestError(f"{link.peer_name} doesn't share its images")
        params: dict[str, Any] = {"limit": limit, "kind": _kind_param(query.kinds)}
        if query.sources:
            params["source"] = [s.value for s in query.sources]
        if query.text:
            params["q"] = query.text
        if position:
            params["cursor"] = position[0]
        page = await peers.list_page(self._links, link, params, recheck_after_s=RECHECK_AFTER_S)
        return gallery.SourcePage(
            items=[self._remote(item, link) for item in page.items],
            has_more=page.next_cursor is not None,
        )

    async def _without_duplicates(
        self, items: list[dto.ArtifactLibraryItem]
    ) -> list[dto.ArtifactLibraryItem]:
        """Leave out remote images this instance has, and repeats among linked ones."""
        remote = [i.sha256 for i in items if i.origin.link_id is not None]
        if not remote:
            return items
        async with self._db.session() as session, session.begin():
            here = await lib_repo.present(session, remote)
        seen: set[str] = set()
        kept = []
        for item in items:
            if item.origin.link_id is not None and (item.sha256 in here or item.sha256 in seen):
                continue
            seen.add(item.sha256)
            kept.append(item)
        return kept

    # --- pins ------------------------------------------------------------------------

    async def pin(self, body: dto.ArtifactPinCreate) -> tuple[dto.ArtifactLibraryItem, bool]:
        """Keep an image. When its bytes are pinned already, that pin (and False)."""
        found = await self._find(body.ref)
        item = found.item
        async with self._db.session() as session, session.begin():
            existing = await lib_repo.pin_by_sha256(session, item.sha256)
        if existing is not None:
            return await self.get_pin(existing.id), False
        ext = EXTENSIONS.get(item.mime)
        if ext is None:
            raise InvalidRequestError(f"can't keep a {item.mime} file")
        data = await self._bytes(found)
        pin_id = new_id()
        key = f"{pin_id}.{ext}"
        path = self._root / key
        await asyncio.to_thread(write_atomic, path, data)
        now = utcnow()
        row = orm.ArtifactPin(
            id=pin_id,
            sort_id=item.id,
            sha256=item.sha256,
            kind=item.kind.value,
            source=item.source.value,
            mime=item.mime,
            storage_key=key,
            size_bytes=len(data),
            width=item.width,
            height=item.height,
            has_alpha=item.has_alpha,
            label=item.label,
            prompt=item.prompt,
            conversation_title=item.conversation_title,
            snapshot=_snapshot(found),
            source_instance_id=item.origin.instance_id,
            source_instance_name=item.origin.name,
            source_conversation_id=item.conversation_id,
            source_artifact_id=item.id,
            name=_text(body.name),
            notes=_text(body.notes),
            tags=_clean_tags(body.tags or []) or None,
            image_created_at=item.created_at,
            created_at=now,
            updated_at=now,
        )
        try:
            async with self._db.session() as session, session.begin():
                session.add(row)
                await session.flush()
        except IntegrityError:
            await asyncio.to_thread(path.unlink, missing_ok=True)
            async with self._db.session() as session, session.begin():
                existing = await lib_repo.pin_by_sha256(session, item.sha256)
            if existing is None:
                raise
            return await self.get_pin(existing.id), False  # pinned meanwhile
        except BaseException:
            await asyncio.to_thread(path.unlink, missing_ok=True)
            raise
        log.info("artifact_library.pinned", pin_id=pin_id, sha256=item.sha256)
        return await self.get_pin(pin_id), True

    async def get_pin(self, pin_id: str) -> dto.ArtifactLibraryItem:
        """A pinned image as the library lists it: the live image while a conversation has
        its bytes, the pin's own copy after that."""
        async with self._db.session() as session, session.begin():
            pin = await lib_repo.get_pin(session, pin_id)
            if pin is None:
                raise NotFoundError(f"pin {pin_id} not found")
            live = await lib_repo.first_copy(session, pin.sha256)
            if live is not None:
                return (await self._items(session, [live]))[0]
            return self._pin_item(pin)

    async def update_pin(self, pin_id: str, body: dto.ArtifactPinUpdate) -> dto.ArtifactLibraryItem:
        fields = body.model_fields_set
        async with self._db.session() as session, session.begin():
            pin = await lib_repo.get_pin(session, pin_id)
            if pin is None:
                raise NotFoundError(f"pin {pin_id} not found")
            if "name" in fields:
                pin.name = _text(body.name)
            if "notes" in fields:
                pin.notes = _text(body.notes)
            if "tags" in fields:
                pin.tags = _clean_tags(body.tags or []) or None
            pin.updated_at = utcnow()
        return await self.get_pin(pin_id)

    async def unpin(self, pin_id: str) -> None:
        async with self._db.session() as session, session.begin():
            pin = await lib_repo.get_pin(session, pin_id)
            if pin is None:
                raise NotFoundError(f"pin {pin_id} not found")
            key = pin.storage_key
            await session.delete(pin)
        path = self._pin_path(key)
        if path is not None:
            await asyncio.to_thread(path.unlink, missing_ok=True)
        log.info("artifact_library.unpinned", pin_id=pin_id)

    async def pin_file(
        self, pin_id: str, *, for_link: bool = False
    ) -> tuple[orm.ArtifactPin, Path]:
        async with self._db.session() as session, session.begin():
            pin = await lib_repo.get_pin(session, pin_id)
        if pin is None or (for_link and pin.source_instance_id != self._links.instance_id):
            raise NotFoundError(f"pin {pin_id} not found")
        path = self._pin_path(pin.storage_key)
        if path is None or not await asyncio.to_thread(path.is_file):
            raise NotFoundError(f"the file of pin {pin_id} is missing")
        return pin, path

    def _pin_path(self, storage_key: str) -> Path | None:
        """Where a pin's file is; None for a key that can't be one (never a path elsewhere)."""
        return self._root / storage_key if _PIN_FILE.fullmatch(storage_key) else None

    # --- using an image ----------------------------------------------------------------

    async def import_into(self, body: dto.ArtifactImport) -> tuple[dto.Artifact, bool]:
        """Copy an image into a conversation, here or on a linked instance. The artifact it
        becomes there comes back, and whether it is new: a conversation that has the image
        already keeps the one it has."""
        ref = body.ref
        if body.link_id is None:
            if ref.origin == LOCAL and ref.artifact_id:
                async with self._db.session() as session, session.begin():
                    row = await art_repo.get_active(session, ref.artifact_id)
                if row is not None and row.conversation_id == body.conversation_id:
                    return mappers.artifact(row), False
            found = await self._find(ref)
            data = await self._bytes(found)
            return await self._artifacts.import_image(
                body.conversation_id,
                data,
                kind=found.item.kind,
                metadata=_provenance(found.item),
            )
        link = self._links.get(body.link_id)
        if ref.origin == link.id and ref.artifact_id:
            there = await peers.artifact(self._links, link, ref.artifact_id)
            if there is not None and there.conversation_id == body.conversation_id:
                return there, False
        found = await self._find(ref)
        return await peers.take(
            self._links,
            link,
            body.conversation_id,
            found.item,
            lambda: self._bytes(found),
            filename=_provenance(found.item)["original_filename"],
        )

    async def _find(self, ref: dto.LibraryRef) -> Found:
        if ref.origin == LOCAL:
            if ref.artifact_id is not None:
                async with self._db.session() as session, session.begin():
                    row = await lib_repo.live_artifact(session, ref.artifact_id)
                    if row is None:
                        raise NotFoundError(f"artifact {ref.artifact_id} not found")
                    item = (await self._items(session, [row]))[0]
                return Found(item, None, row.artifact.metadata_json)
            async with self._db.session() as session, session.begin():
                pin = await lib_repo.get_pin(session, ref.pin_id or "")
            if pin is None:
                raise NotFoundError(f"pin {ref.pin_id} not found")
            return Found(self._pin_item(pin), None, (pin.snapshot or {}).get("metadata"))
        link = self._links.get(ref.origin)
        item = await peers.item(self._links, link, ref)
        return Found(self._remote(item, link), link)

    async def _bytes(self, found: Found) -> bytes:
        item = found.item
        if found.link is not None:
            return await peers.fetch(
                self._links,
                found.link,
                item.content_url,
                sha256=item.sha256,
                max_bytes=self._max_bytes,
            )
        if item.ref.artifact_id is not None:
            _, data = await self._artifacts.read_bytes(item.ref.artifact_id)
            return data
        _, path = await self.pin_file(item.ref.pin_id or "")
        return await asyncio.to_thread(path.read_bytes)

    # --- items -----------------------------------------------------------------------

    async def _items(
        self, session: Any, rows: Sequence[LiveRow | orm.ArtifactPin], *, for_link: bool = False
    ) -> list[dto.ArtifactLibraryItem]:
        live = [r for r in rows if isinstance(r, LiveRow)]
        job_ids = [j for r in live if (j := _job_id(r.artifact.metadata_json))]
        jobs = await lib_repo.jobs(session, job_ids)
        counts = await lib_repo.copies(session, [r.artifact.sha256 for r in live])
        return [
            self._live_item(r, jobs, counts, for_link=for_link)
            if isinstance(r, LiveRow)
            else self._pin_item(r, for_link=for_link)
            for r in rows
        ]

    def _origin(self) -> dto.Origin:
        return dto.Origin(instance_id=self._links.instance_id, name=self._links.name)

    def _live_item(
        self,
        row: LiveRow,
        jobs: dict[str, orm.GenerationJob],
        counts: dict[tuple[str, str], int],
        *,
        for_link: bool = False,
    ) -> dto.ArtifactLibraryItem:
        a = row.artifact
        metadata = a.metadata_json or {}
        imported = metadata.get("imported_from")
        imported = imported if isinstance(imported, dict) else {}
        job_id = _job_id(metadata)
        generation = _generation(jobs.get(job_id) if job_id else None, metadata)
        if generation is None and isinstance(imported.get("generation"), dict):
            generation = _quiet_generation(imported["generation"])
        return dto.ArtifactLibraryItem(
            id=a.id,
            ref=dto.LibraryRef(origin=LOCAL, artifact_id=a.id),
            live=True,
            kind=a.kind,
            source=a.source,
            mime=a.mime,
            sha256=a.sha256,
            size_bytes=a.size_bytes,
            width=a.width,
            height=a.height,
            has_alpha=a.has_alpha,
            created_at=a.created_at,
            content_url=f"/v1/artifacts/{a.id}/content",
            label=a.label,
            conversation_id=a.conversation_id,
            conversation_title=row.conversation_title,
            prompt=_str(metadata.get("prompt")) or _str(imported.get("prompt")),
            original_filename=_str(metadata.get("original_filename")),
            generation=generation,
            copies=counts.get((a.sha256, a.kind.value), 1),
            pin=self._pin_info(row.pin) if row.pin is not None and not for_link else None,
            origin=self._origin(),
        )

    def _pin_item(self, pin: orm.ArtifactPin, *, for_link: bool = False) -> dto.ArtifactLibraryItem:
        snapshot = pin.snapshot or {}
        generation = snapshot.get("generation")
        return dto.ArtifactLibraryItem(
            id=pin.sort_id,
            ref=dto.LibraryRef(origin=LOCAL, pin_id=pin.id),
            live=False,
            kind=_enum(ArtifactKind, pin.kind, ArtifactKind.IMAGE),
            source=_enum(ArtifactSource, pin.source, ArtifactSource.UPLOAD),
            mime=pin.mime,
            sha256=pin.sha256,
            size_bytes=pin.size_bytes,
            width=pin.width,
            height=pin.height,
            has_alpha=pin.has_alpha,
            created_at=pin.image_created_at,
            content_url=f"/v1/artifact-library/pins/{pin.id}/content",
            label=pin.label,
            conversation_title=pin.conversation_title,
            prompt=pin.prompt,
            original_filename=_str(snapshot.get("original_filename")),
            generation=_quiet_generation(generation) if isinstance(generation, dict) else None,
            copies=0,
            pin=None if for_link else self._pin_info(pin),
            origin=self._origin(),
        )

    def _pin_info(self, pin: orm.ArtifactPin) -> dto.LibraryPinInfo:
        here = pin.source_instance_id == self._links.instance_id
        link = None if here else self._links.by_peer(pin.source_instance_id)
        return dto.LibraryPinInfo(
            id=pin.id,
            name=pin.name,
            notes=pin.notes,
            tags=list(pin.tags or []),
            pinned_at=pin.created_at,
            source_instance_id=pin.source_instance_id,
            source_instance_name=pin.source_instance_name,
            source_link_id=link.id if link is not None else None,
            source_conversation_id=pin.source_conversation_id,
        )

    @staticmethod
    def _remote(item: dto.ArtifactLibraryItem, link: Link) -> dto.ArtifactLibraryItem:
        """A linked instance's image, as it is reached from here."""
        return item.model_copy(
            update={
                "ref": item.ref.model_copy(update={"origin": link.id}),
                "pin": None,
                "origin": dto.Origin(
                    link_id=link.id,
                    instance_id=link.peer_instance_id,
                    name=link.peer_name,
                    base_path=f"/v1/links/{link.id}/r",
                ),
            }
        )


# --- helpers ---------------------------------------------------------------------------


def _row_id(row: LiveRow | orm.ArtifactPin) -> str:
    return row.artifact.id if isinstance(row, LiveRow) else row.sort_id


def _by_id(item: dto.ArtifactLibraryItem) -> Any:
    return item.id


def _id_position(item: dto.ArtifactLibraryItem) -> tuple[str, ...]:
    return (item.id,)


def _failed(source: dto.GallerySource, error: BaseException) -> dto.GallerySource:
    if isinstance(error, peers.PeerTooOldError):
        return source.model_copy(update={"status": "unsupported", "detail": error.detail})
    return gallery.source_failed(source, error)


def _kind_param(kinds: tuple[ArtifactKind, ...]) -> str:
    if set(kinds) == {ArtifactKind.IMAGE, ArtifactKind.MASK}:
        return "all"
    return kinds[0].value if kinds else "image"


def _job_id(metadata: dict[str, Any] | None) -> str | None:
    job_id = (metadata or {}).get("job_id")
    return job_id if isinstance(job_id, str) and job_id else None


def _generation(
    job: orm.GenerationJob | None, metadata: dict[str, Any]
) -> dto.LibraryGeneration | None:
    if job is None:
        return None
    params = job.params_json if isinstance(job.params_json, dict) else {}
    resolved = params.get("resolved")
    card = params.get("model_card")
    seed = metadata.get("seed")
    return dto.LibraryGeneration(
        job_id=job.id,
        task=job.task.value,
        backend=job.backend,
        model=job.model,
        model_card=card if isinstance(card, str) else None,
        seed=seed if isinstance(seed, int) else job.seed,
        alpha=job.alpha,
        resolved=resolved if isinstance(resolved, dict) else None,
    )


def _quiet_generation(raw: dict[str, Any]) -> dto.LibraryGeneration | None:
    try:
        return dto.LibraryGeneration.model_validate(raw)
    except ValueError:
        return None


def _snapshot(found: Found) -> dict[str, Any]:
    item = found.item
    snapshot: dict[str, Any] = {}
    if item.generation is not None:
        snapshot["generation"] = item.generation.model_dump(mode="json", exclude_none=True)
    if item.original_filename:
        snapshot["original_filename"] = item.original_filename
    if found.metadata:
        snapshot["metadata"] = found.metadata
    return snapshot


def _provenance(item: dto.ArtifactLibraryItem) -> dict[str, Any]:
    """An imported artifact's metadata: a file name, and where it was brought in from."""
    ext = EXTENSIONS.get(item.mime, "png")
    stem = (item.original_filename or "").rsplit(".", 1)[0] or item.label or "image"
    imported: dict[str, Any] = {
        "instance_id": item.origin.instance_id,
        "instance_name": item.origin.name,
        "conversation_id": item.conversation_id,
        "conversation_title": item.conversation_title,
        "artifact_id": item.ref.artifact_id,
        "pin_id": item.ref.pin_id,
        "label": item.label,
        "source": item.source.value,
        "prompt": item.prompt,
    }
    if item.generation is not None:
        imported["generation"] = item.generation.model_dump(mode="json", exclude_none=True)
    return {
        "original_filename": f"{stem[:200]}.{ext}",
        "imported_from": {k: v for k, v in imported.items() if v is not None},
    }


def _enum[E: (ArtifactKind, ArtifactSource)](cls: type[E], value: str, default: E) -> E:
    try:
        return cls(value)
    except ValueError:
        return default


def _str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _text(value: str | None) -> str | None:
    value = (value or "").strip()
    return value or None


def _clean_tags(tags: Iterable[str]) -> list[str]:
    out: list[str] = []
    for tag in tags:
        text = " ".join(str(tag).split())[:32]
        if text and text.casefold() not in {t.casefold() for t in out}:
            out.append(text)
    return out
