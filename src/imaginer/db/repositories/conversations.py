from collections.abc import Iterable
from datetime import datetime

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import InstrumentedAttribute

from imaginer.db.models import Conversation
from imaginer.db.types import UTCDateTime
from imaginer.domain.errors import NotFoundError

_ACTIVE = Conversation.deleted_at.is_(None)


async def insert(session: AsyncSession, row: Conversation) -> None:
    session.add(row)
    await session.flush()


async def get_active(session: AsyncSession, conversation_id: str) -> Conversation | None:
    stmt = sa.select(Conversation).where(Conversation.id == conversation_id, _ACTIVE)
    return (await session.execute(stmt)).scalar_one_or_none()


async def require_active(session: AsyncSession, conversation_id: str) -> Conversation:
    row = await get_active(session, conversation_id)
    if row is None:
        raise NotFoundError(f"conversation {conversation_id} not found")
    return row


async def list_active(
    session: AsyncSession,
    *,
    limit: int,
    before: tuple[datetime, str] | None = None,
    include_archived: bool = False,
) -> list[Conversation]:
    """Most recently updated first. ``before`` is the (updated_at, id) of the last row seen."""
    stmt = sa.select(Conversation).where(_ACTIVE)
    if not include_archived:
        stmt = stmt.where(Conversation.archived_at.is_(None))
    if before is not None:
        stamp, last_id = before
        stmt = stmt.where(
            sa.tuple_(Conversation.updated_at, Conversation.id)
            < sa.tuple_(sa.literal(stamp, UTCDateTime()), sa.literal(last_id, sa.String()))
        )
    stmt = stmt.order_by(Conversation.updated_at.desc(), Conversation.id.desc()).limit(limit)
    return list((await session.execute(stmt)).scalars())


async def tombstone(session: AsyncSession, conversation_id: str, now: datetime) -> bool:
    """Mark deleted. False if it does not exist or is already being deleted."""
    stmt = (
        sa.update(Conversation)
        .where(Conversation.id == conversation_id, _ACTIVE)
        .values(deleted_at=now)
        .returning(Conversation.id)
    )
    return (await session.execute(stmt)).scalar_one_or_none() is not None


async def hard_delete(session: AsyncSession, conversation_id: str) -> None:
    """Remove the row; FK cascades take messages, artifacts, lineage and jobs with it."""
    await session.execute(sa.delete(Conversation).where(Conversation.id == conversation_id))


async def list_tombstoned_ids(session: AsyncSession) -> list[str]:
    stmt = sa.select(Conversation.id).where(Conversation.deleted_at.is_not(None))
    return list((await session.execute(stmt)).scalars())


async def existing_ids(session: AsyncSession, ids: Iterable[str]) -> set[str]:
    """Which of ``ids`` have a row, tombstoned or not."""
    wanted = list(ids)
    if not wanted:
        return set()
    stmt = sa.select(Conversation.id).where(Conversation.id.in_(wanted))
    return set((await session.execute(stmt)).scalars())


async def _increment(
    session: AsyncSession,
    conversation_id: str,
    column: InstrumentedAttribute[int],
    now: datetime,
) -> int:
    stmt = (
        sa.update(Conversation)
        .where(Conversation.id == conversation_id, _ACTIVE)
        .values({column: column + 1, Conversation.updated_at: now})
        .returning(column)
    )
    value = (await session.execute(stmt)).scalar_one_or_none()
    if value is None:
        raise NotFoundError(f"conversation {conversation_id} not found")
    return value


async def allocate_seq(session: AsyncSession, conversation_id: str, now: datetime) -> int:
    """Next message sequence number (1-based). Also bumps ``updated_at``."""
    return await _increment(session, conversation_id, Conversation.last_seq, now)


async def allocate_artifact_no(session: AsyncSession, conversation_id: str, now: datetime) -> int:
    """Next artifact number for labels like ``img-3`` (1-based). Also bumps ``updated_at``."""
    return await _increment(session, conversation_id, Conversation.last_artifact_no, now)
