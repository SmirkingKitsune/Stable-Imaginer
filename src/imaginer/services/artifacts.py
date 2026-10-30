import asyncio
import re
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from sqlalchemy.ext.asyncio import AsyncSession

from imaginer.core.ids import new_id
from imaginer.core.logging import get_logger
from imaginer.db import models as orm
from imaginer.db.engine import Database
from imaginer.db.repositories import artifacts as art_repo
from imaginer.db.repositories import conversations as conv_repo
from imaginer.db.repositories.artifacts import ParentRef
from imaginer.db.types import utcnow
from imaginer.domain import models as dto
from imaginer.domain.enums import LABEL_PREFIX, ArtifactKind, ArtifactSource
from imaginer.domain.errors import InvalidRequestError, NotFoundError
from imaginer.services import mappers
from imaginer.services.media import ImageInfo, inspect_image
from imaginer.storage.base import ArtifactStore

log = get_logger(__name__)

UPLOAD_CHUNK = 1 << 20

# Runs inside the artifact's insert transaction, e.g. to link job outputs atomically.
SessionHook = Callable[[AsyncSession, orm.Artifact], Awaitable[None]]


@dataclass(frozen=True, slots=True, kw_only=True)
class NewArtifact:
    """Everything needed to create an artifact except its id and label."""

    conversation_id: str
    kind: ArtifactKind
    source: ArtifactSource
    mime: str
    ext: str
    width: int | None = None
    height: int | None = None
    has_alpha: bool | None = None
    duration_ms: int | None = None
    message_id: str | None = None
    metadata: dict[str, Any] | None = None
    parents: tuple[ParentRef, ...] = ()

    @classmethod
    def image(
        cls,
        conversation_id: str,
        info: ImageInfo,
        source: ArtifactSource,
        kind: ArtifactKind = ArtifactKind.IMAGE,
        **kwargs: Any,
    ) -> "NewArtifact":
        return cls(
            conversation_id=conversation_id,
            kind=kind,
            source=source,
            mime=info.mime,
            ext=info.ext,
            width=info.width,
            height=info.height,
            has_alpha=info.has_alpha,
            **kwargs,
        )


@dataclass(frozen=True, slots=True)
class ArtifactContent:
    artifact: dto.Artifact
    path: Path | None  # serve directly when the store is local
    chunks: Callable[[], AsyncIterator[bytes]]


class ArtifactService:
    def __init__(self, db: Database, store: ArtifactStore, *, max_upload_bytes: int) -> None:
        self._db = db
        self._store = store
        self._max_upload_bytes = max_upload_bytes

    async def ingest(
        self,
        spec: NewArtifact,
        data: bytes | AsyncIterable[bytes],
        *,
        max_bytes: int | None = None,
        session_hook: SessionHook | None = None,
    ) -> dto.Artifact:
        """Store bytes, then record the row (and lineage) in one transaction.

        Bytes are written first and outside the transaction, so the database lock is never
        held during disk I/O. If the insert fails, the file is removed again.
        """
        artifact_id = new_id()
        stored = await self._store.put(
            spec.conversation_id, artifact_id, spec.ext, data, max_bytes=max_bytes
        )
        try:
            async with self._db.session() as session, session.begin():
                now = utcnow()
                number = await conv_repo.allocate_artifact_no(session, spec.conversation_id, now)
                row = orm.Artifact(
                    id=artifact_id,
                    conversation_id=spec.conversation_id,
                    message_id=spec.message_id,
                    kind=spec.kind,
                    source=spec.source,
                    mime=spec.mime,
                    storage_key=stored.storage_key,
                    sha256=stored.sha256,
                    size_bytes=stored.size_bytes,
                    width=spec.width,
                    height=spec.height,
                    has_alpha=spec.has_alpha,
                    duration_ms=spec.duration_ms,
                    label=f"{LABEL_PREFIX[spec.kind]}-{number}",
                    metadata_json=spec.metadata,
                    created_at=now,
                )
                await art_repo.insert(session, row)
                if spec.parents:
                    await self._check_parents(session, spec)
                    await art_repo.add_lineage(session, artifact_id, spec.parents)
                if session_hook is not None:
                    await session_hook(session, row)
        except BaseException:
            try:
                await self._store.delete_object(spec.conversation_id, stored.storage_key)
            except OSError:
                log.warning("artifact.cleanup_failed", storage_key=stored.storage_key)
            raise
        log.info(
            "artifact.created",
            conversation_id=row.conversation_id,
            artifact_id=row.id,
            label=row.label,
            source=row.source.value,
            size_bytes=row.size_bytes,
        )
        return mappers.artifact(row)

    async def ingest_upload(
        self,
        conversation_id: str,
        fileobj: BinaryIO,
        filename: str | None,
        *,
        kind: ArtifactKind = ArtifactKind.IMAGE,
        mask_for: str | None = None,
    ) -> dto.Artifact:
        """An uploaded image, or (kind=mask) a mask painted for the image labelled mask_for."""
        async with self._db.session() as session, session.begin():
            await conv_repo.require_active(session, conversation_id)
            if mask_for is not None:
                source = await art_repo.get_by_label(session, conversation_id, mask_for)
                if source is None or source.kind is not ArtifactKind.IMAGE:
                    raise InvalidRequestError(f"no image labelled {mask_for!r} to mask")
        info = await asyncio.to_thread(_inspect_from_start, fileobj)
        metadata: dict[str, Any] = {}
        if filename:
            metadata["original_filename"] = _clean_filename(filename)
        if mask_for is not None:
            metadata["mask_for"] = mask_for
        spec = NewArtifact.image(
            conversation_id, info, ArtifactSource.UPLOAD, kind, metadata=metadata or None
        )
        return await self.ingest(spec, _read_chunks(fileobj), max_bytes=self._max_upload_bytes)

    async def get(self, artifact_id: str) -> dto.Artifact:
        return mappers.artifact(await self._get_row(artifact_id))

    async def get_by_label(self, conversation_id: str, label: str) -> dto.Artifact:
        async with self._db.session() as session, session.begin():
            row = await art_repo.get_by_label(session, conversation_id, label)
        if row is None:
            raise NotFoundError(f"no artifact labelled {label!r} in this conversation")
        return mappers.artifact(row)

    async def list(
        self,
        conversation_id: str,
        *,
        kind: ArtifactKind | None = None,
        source: ArtifactSource | None = None,
        limit: int,
        cursor: str | None = None,
    ) -> dto.Page[dto.Artifact]:
        async with self._db.session() as session, session.begin():
            await conv_repo.require_active(session, conversation_id)
            rows = await art_repo.list_for_conversation(
                session, conversation_id, kind=kind, source=source, after_id=cursor, limit=limit + 1
            )
        has_more = len(rows) > limit
        rows = rows[:limit]
        return dto.Page[dto.Artifact](
            items=[mappers.artifact(r) for r in rows],
            next_cursor=rows[-1].id if has_more else None,
        )

    async def content(self, artifact_id: str) -> ArtifactContent:
        row = await self._get_row(artifact_id)
        cid, key = row.conversation_id, row.storage_key
        return ArtifactContent(
            artifact=mappers.artifact(row),
            path=self._store.local_path(cid, key),
            chunks=lambda: self._store.iter_bytes(cid, key),
        )

    async def read_bytes(self, artifact_id: str) -> tuple[dto.Artifact, bytes]:
        row = await self._get_row(artifact_id)
        data = await self._store.read_bytes(row.conversation_id, row.storage_key)
        return mappers.artifact(row), data

    async def lineage(self, artifact_id: str) -> dto.Lineage:
        async with self._db.session() as session, session.begin():
            if await art_repo.get_active(session, artifact_id) is None:
                raise NotFoundError(f"artifact {artifact_id} not found")
            parents = await art_repo.parents_of(session, artifact_id)
            children = await art_repo.children_of(session, artifact_id)
        return dto.Lineage(
            artifact_id=artifact_id,
            parents=[_edge(link) for link in parents],
            children=[_edge(link) for link in children],
        )

    async def _get_row(self, artifact_id: str) -> orm.Artifact:
        async with self._db.session() as session, session.begin():
            row = await art_repo.get_active(session, artifact_id)
        if row is None:
            raise NotFoundError(f"artifact {artifact_id} not found")
        return row

    @staticmethod
    async def _check_parents(session: AsyncSession, spec: NewArtifact) -> None:
        wanted = [p.parent_id for p in spec.parents]
        found = await art_repo.get_many_in_conversation(session, spec.conversation_id, wanted)
        missing = set(wanted) - {a.id for a in found}
        if missing:
            raise InvalidRequestError(
                "lineage parents must belong to the same conversation",
                missing=sorted(missing),
            )


def _edge(link: art_repo.LineageLink) -> dto.LineageEdge:
    return dto.LineageEdge(
        artifact_id=link.artifact.id,
        label=link.artifact.label,
        relation=link.relation,
        ordinal=link.ordinal,
    )


def _inspect_from_start(fileobj: BinaryIO) -> ImageInfo:
    fileobj.seek(0)
    try:
        return inspect_image(fileobj)
    finally:
        fileobj.seek(0)


async def _read_chunks(fileobj: BinaryIO) -> AsyncIterator[bytes]:
    while chunk := await asyncio.to_thread(fileobj.read, UPLOAD_CHUNK):
        yield chunk


def _clean_filename(name: str) -> str:
    """Keep only the final path component, whichever separator the client used."""
    return re.split(r"[\\/]", name)[-1][:255]
