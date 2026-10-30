from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from imaginer.db.models import Conversation, Message, MessageAttachment


async def insert(session: AsyncSession, row: Message) -> None:
    session.add(row)
    await session.flush()


async def list_for_conversation(
    session: AsyncSession, conversation_id: str, *, after_seq: int = 0, limit: int
) -> list[Message]:
    stmt = (
        sa.select(Message)
        .join(Conversation, Conversation.id == Message.conversation_id)
        .where(
            Message.conversation_id == conversation_id,
            Conversation.deleted_at.is_(None),
            Message.seq > after_seq,
        )
        .order_by(Message.seq)
        .limit(limit)
    )
    return list((await session.execute(stmt)).scalars())


async def add_attachments(
    session: AsyncSession, message_id: str, artifact_ids: Sequence[str]
) -> None:
    for ordinal, artifact_id in enumerate(artifact_ids):
        session.add(
            MessageAttachment(message_id=message_id, artifact_id=artifact_id, ordinal=ordinal)
        )
    await session.flush()


async def attachments_for(
    session: AsyncSession, message_ids: Sequence[str]
) -> dict[str, list[str]]:
    """Attached artifact ids per message, in attachment order."""
    if not message_ids:
        return {}
    stmt = (
        sa.select(MessageAttachment.message_id, MessageAttachment.artifact_id)
        .where(MessageAttachment.message_id.in_(message_ids))
        .order_by(MessageAttachment.message_id, MessageAttachment.ordinal)
    )
    result: dict[str, list[str]] = {}
    for message_id, artifact_id in await session.execute(stmt):
        result.setdefault(message_id, []).append(artifact_id)
    return result


async def latest(session: AsyncSession, conversation_id: str) -> Message | None:
    """The newest message: the leaf of the latest branch (nothing is ever added below it)."""
    stmt = (
        sa.select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.seq.desc())
        .limit(1)
    )
    return (await session.execute(stmt)).scalar_one_or_none()


async def path_to(session: AsyncSession, message_id: str) -> list[Message]:
    """The branch ending at ``message_id``: the message and its ancestors, oldest first."""
    path = (
        sa.select(Message.id, Message.parent_id)
        .where(Message.id == message_id)
        .cte("path", recursive=True)
    )
    parent = aliased(Message)
    path = path.union_all(
        sa.select(parent.id, parent.parent_id).join(path, parent.id == path.c.parent_id)
    )
    stmt = sa.select(Message).join(path, Message.id == path.c.id).order_by(Message.seq)
    return list((await session.execute(stmt)).scalars())


async def all_for_conversation(session: AsyncSession, conversation_id: str) -> list[Message]:
    """Every message of every branch, in creation order."""
    stmt = (
        sa.select(Message).where(Message.conversation_id == conversation_id).order_by(Message.seq)
    )
    return list((await session.execute(stmt)).scalars())


async def get_in_conversation(
    session: AsyncSession, conversation_id: str, message_id: str
) -> Message | None:
    stmt = sa.select(Message).where(
        Message.id == message_id, Message.conversation_id == conversation_id
    )
    return (await session.execute(stmt)).scalar_one_or_none()


async def get(session: AsyncSession, message_id: str) -> Message | None:
    return await session.get(Message, message_id)
