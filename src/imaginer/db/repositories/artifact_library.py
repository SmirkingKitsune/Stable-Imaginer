"""Queries for the artifact library: images across conversations, and pinned copies.

Copies of the same bytes (an image imported into several conversations) are listed once,
as their earliest copy. A pin is listed on its own only once no conversation has its bytes
any more; until then it rides on the live image. Everything is listed newest first by id
(UUIDv7, so time-ordered), a pin by the id of the artifact it was taken from.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import InstrumentedAttribute, aliased

from imaginer.db.models import Artifact, ArtifactPin, Conversation, GenerationJob
from imaginer.domain.enums import ArtifactKind, ArtifactSource


@dataclass(frozen=True, slots=True)
class LibraryQuery:
    kinds: tuple[ArtifactKind, ...] = (ArtifactKind.IMAGE,)
    sources: tuple[ArtifactSource, ...] = ()  # empty: every source
    pinned: bool = False
    text: str | None = None


class LiveRow(NamedTuple):
    artifact: Artifact
    conversation_title: str | None
    pin: ArtifactPin | None


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


Column = sa.ColumnElement[Any] | InstrumentedAttribute[Any]


def _like(column: Column, text: str) -> sa.ColumnElement[bool]:
    return column.ilike(f"%{_escape(text)}%", escape="\\")


def _live() -> sa.Select[Artifact, str | None, ArtifactPin]:
    """Artifacts of conversations that aren't deleted, each with its conversation's title and
    the pin of its bytes, if any."""
    return (
        sa.select(Artifact, Conversation.title, ArtifactPin)
        .join(Conversation, Conversation.id == Artifact.conversation_id)
        .outerjoin(ArtifactPin, ArtifactPin.sha256 == Artifact.sha256)
        .where(Conversation.deleted_at.is_(None))
    )


def _has_live_copy(sha256: Column) -> sa.Exists:
    copy = aliased(Artifact)
    room = aliased(Conversation)
    return (
        sa.select(copy.id)
        .join(room, room.id == copy.conversation_id)
        .where(copy.sha256 == sha256, room.deleted_at.is_(None))
        .exists()
    )


async def live_page(
    session: AsyncSession, query: LibraryQuery, *, after: str | None, limit: int
) -> list[LiveRow]:
    """Up to ``limit`` distinct live images older than ``after``, newest first."""
    earlier = aliased(Artifact)
    room = aliased(Conversation)
    has_earlier_copy = (
        sa.select(earlier.id)
        .join(room, room.id == earlier.conversation_id)
        .where(
            earlier.sha256 == Artifact.sha256,
            earlier.kind == Artifact.kind,
            earlier.id < Artifact.id,
            room.deleted_at.is_(None),
        )
        .exists()
    )
    stmt = _live().where(Artifact.kind.in_(query.kinds), ~has_earlier_copy)
    if query.sources:
        stmt = stmt.where(Artifact.source.in_(query.sources))
    if query.pinned:
        stmt = stmt.where(ArtifactPin.id.is_not(None))
    if query.text:
        stmt = stmt.where(
            sa.or_(
                _like(sa.cast(Artifact.metadata_json, sa.Text), query.text),
                _like(Artifact.label, query.text),
                _like(Conversation.title, query.text),
                _like(ArtifactPin.name, query.text),
                _like(ArtifactPin.notes, query.text),
            )
        )
    if after is not None:
        stmt = stmt.where(Artifact.id < after)
    stmt = stmt.order_by(Artifact.id.desc()).limit(limit)
    return [LiveRow(*row) for row in await session.execute(stmt)]


async def pin_cards(
    session: AsyncSession,
    query: LibraryQuery,
    *,
    after: str | None,
    limit: int,
    instance_id: str | None = None,
) -> list[ArtifactPin]:
    """Pins whose bytes no conversation has any more, newest first. ``instance_id``: only
    pins of that instance's own images."""
    stmt = sa.select(ArtifactPin).where(
        ~_has_live_copy(ArtifactPin.sha256),
        ArtifactPin.kind.in_([k.value for k in query.kinds]),
    )
    if query.sources:
        stmt = stmt.where(ArtifactPin.source.in_([s.value for s in query.sources]))
    if instance_id is not None:
        stmt = stmt.where(ArtifactPin.source_instance_id == instance_id)
    if query.text:
        stmt = stmt.where(
            sa.or_(
                _like(ArtifactPin.prompt, query.text),
                _like(ArtifactPin.label, query.text),
                _like(ArtifactPin.conversation_title, query.text),
                _like(ArtifactPin.name, query.text),
                _like(ArtifactPin.notes, query.text),
                _like(sa.cast(ArtifactPin.snapshot, sa.Text), query.text),
            )
        )
    if after is not None:
        stmt = stmt.where(ArtifactPin.sort_id < after)
    stmt = stmt.order_by(ArtifactPin.sort_id.desc()).limit(limit)
    return list((await session.execute(stmt)).scalars())


async def live_artifact(session: AsyncSession, artifact_id: str) -> LiveRow | None:
    row = (await session.execute(_live().where(Artifact.id == artifact_id))).one_or_none()
    return LiveRow(*row) if row is not None else None


async def first_copy(session: AsyncSession, sha256: str) -> LiveRow | None:
    """The earliest live artifact with these bytes: the one the library lists."""
    stmt = _live().where(Artifact.sha256 == sha256).order_by(Artifact.id).limit(1)
    row = (await session.execute(stmt)).one_or_none()
    return LiveRow(*row) if row is not None else None


async def copies(session: AsyncSession, sha256s: Sequence[str]) -> dict[tuple[str, str], int]:
    """How many live artifacts have each of these bytes, by (sha256, kind)."""
    if not sha256s:
        return {}
    stmt = (
        sa.select(Artifact.sha256, Artifact.kind, sa.func.count())
        .join(Conversation, Conversation.id == Artifact.conversation_id)
        .where(Conversation.deleted_at.is_(None), Artifact.sha256.in_(set(sha256s)))
        .group_by(Artifact.sha256, Artifact.kind)
    )
    return {(sha, kind.value): int(n) for sha, kind, n in await session.execute(stmt)}


async def present(session: AsyncSession, sha256s: Sequence[str]) -> set[str]:
    """Which of these bytes this instance has: in a conversation, or pinned."""
    if not sha256s:
        return set()
    wanted = set(sha256s)
    live = (
        sa.select(Artifact.sha256)
        .join(Conversation, Conversation.id == Artifact.conversation_id)
        .where(Conversation.deleted_at.is_(None), Artifact.sha256.in_(wanted))
    )
    pinned = sa.select(ArtifactPin.sha256).where(ArtifactPin.sha256.in_(wanted))
    found = set((await session.execute(live)).scalars())
    return found | set((await session.execute(pinned)).scalars())


async def jobs(session: AsyncSession, job_ids: Sequence[str]) -> dict[str, GenerationJob]:
    if not job_ids:
        return {}
    stmt = sa.select(GenerationJob).where(GenerationJob.id.in_(set(job_ids)))
    return {job.id: job for job in (await session.execute(stmt)).scalars()}


async def get_pin(session: AsyncSession, pin_id: str) -> ArtifactPin | None:
    return await session.get(ArtifactPin, pin_id)


async def pin_by_sha256(session: AsyncSession, sha256: str) -> ArtifactPin | None:
    stmt = sa.select(ArtifactPin).where(ArtifactPin.sha256 == sha256)
    return (await session.execute(stmt)).scalar_one_or_none()


async def pin_storage_keys(session: AsyncSession) -> set[str]:
    return set((await session.execute(sa.select(ArtifactPin.storage_key))).scalars())
