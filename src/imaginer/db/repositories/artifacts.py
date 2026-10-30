from collections.abc import Sequence
from typing import NamedTuple

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from imaginer.db.models import Artifact, ArtifactLineage, Conversation, MessageAttachment
from imaginer.domain.enums import ArtifactKind, ArtifactSource, LineageRelation


class LineageLink(NamedTuple):
    artifact: Artifact
    relation: LineageRelation
    ordinal: int


class ParentRef(NamedTuple):
    parent_id: str
    relation: LineageRelation
    ordinal: int = 0


def _active_artifacts() -> sa.Select[Artifact]:
    return (
        sa.select(Artifact)
        .join(Conversation, Conversation.id == Artifact.conversation_id)
        .where(Conversation.deleted_at.is_(None))
    )


async def insert(session: AsyncSession, row: Artifact) -> None:
    session.add(row)
    await session.flush()


async def get_active(session: AsyncSession, artifact_id: str) -> Artifact | None:
    stmt = _active_artifacts().where(Artifact.id == artifact_id)
    return (await session.execute(stmt)).scalar_one_or_none()


async def get_by_label(session: AsyncSession, conversation_id: str, label: str) -> Artifact | None:
    """Resolve a label like ``img-3``. Scoped to one conversation, by construction."""
    stmt = _active_artifacts().where(
        Artifact.conversation_id == conversation_id, Artifact.label == label
    )
    return (await session.execute(stmt)).scalar_one_or_none()


async def get_many_in_conversation(
    session: AsyncSession, conversation_id: str, artifact_ids: Sequence[str]
) -> list[Artifact]:
    """The subset of ``artifact_ids`` that belong to ``conversation_id``."""
    if not artifact_ids:
        return []
    stmt = _active_artifacts().where(
        Artifact.conversation_id == conversation_id, Artifact.id.in_(artifact_ids)
    )
    return list((await session.execute(stmt)).scalars())


async def list_for_conversation(
    session: AsyncSession,
    conversation_id: str,
    *,
    kind: ArtifactKind | None = None,
    source: ArtifactSource | None = None,
    after_id: str | None = None,
    limit: int,
) -> list[Artifact]:
    """Oldest first. UUIDv7 ids are time-ordered, so ``id`` doubles as the cursor."""
    stmt = _active_artifacts().where(Artifact.conversation_id == conversation_id)
    if kind is not None:
        stmt = stmt.where(Artifact.kind == kind)
    if source is not None:
        stmt = stmt.where(Artifact.source == source)
    if after_id is not None:
        stmt = stmt.where(Artifact.id > after_id)
    stmt = stmt.order_by(Artifact.id).limit(limit)
    return list((await session.execute(stmt)).scalars())


async def storage_keys(session: AsyncSession, conversation_id: str) -> set[str]:
    stmt = sa.select(Artifact.storage_key).where(Artifact.conversation_id == conversation_id)
    return set((await session.execute(stmt)).scalars())


async def add_lineage(session: AsyncSession, child_id: str, parents: Sequence[ParentRef]) -> None:
    for ref in parents:
        session.add(
            ArtifactLineage(
                child_id=child_id,
                parent_id=ref.parent_id,
                relation=ref.relation,
                ordinal=ref.ordinal,
            )
        )
    await session.flush()


async def parents_of(session: AsyncSession, artifact_id: str) -> list[LineageLink]:
    """In input order: the edit source, then references by position, then the mask."""
    parent = aliased(Artifact)
    relation_order = sa.case(
        {
            LineageRelation.EDIT_SOURCE: 0,
            LineageRelation.REFERENCE: 1,
            LineageRelation.MASK: 2,
        },
        value=ArtifactLineage.relation,
        else_=3,
    )
    stmt = (
        sa.select(parent, ArtifactLineage.relation, ArtifactLineage.ordinal)
        .join(ArtifactLineage, ArtifactLineage.parent_id == parent.id)
        .where(ArtifactLineage.child_id == artifact_id)
        .order_by(relation_order, ArtifactLineage.ordinal)
    )
    return [LineageLink(*row) for row in await session.execute(stmt)]


async def children_of(session: AsyncSession, artifact_id: str) -> list[LineageLink]:
    child = aliased(Artifact)
    stmt = (
        sa.select(child, ArtifactLineage.relation, ArtifactLineage.ordinal)
        .join(ArtifactLineage, ArtifactLineage.child_id == child.id)
        .where(ArtifactLineage.parent_id == artifact_id)
        .order_by(child.id)
    )
    return [LineageLink(*row) for row in await session.execute(stmt)]


async def unattached(session: AsyncSession, conversation_id: str) -> list[Artifact]:
    """Artifacts that no message attaches (yet), such as a fresh upload. Oldest first."""
    attached = sa.exists().where(MessageAttachment.artifact_id == Artifact.id)
    stmt = (
        sa.select(Artifact)
        .where(Artifact.conversation_id == conversation_id, ~attached)
        .order_by(Artifact.id)
    )
    return list((await session.execute(stmt)).scalars())


class ImageSummary(NamedTuple):
    images: int
    latest: Artifact


async def image_summaries(
    session: AsyncSession, conversation_ids: Sequence[str]
) -> dict[str, ImageSummary]:
    """Per conversation that has images: how many, and the most recent one."""
    if not conversation_ids:
        return {}
    ranked = (
        sa.select(
            Artifact.id,
            sa.func.row_number()
            .over(partition_by=Artifact.conversation_id, order_by=Artifact.id.desc())
            .label("rank"),
            sa.func.count().over(partition_by=Artifact.conversation_id).label("images"),
        )
        .where(Artifact.conversation_id.in_(conversation_ids), Artifact.kind == ArtifactKind.IMAGE)
        .subquery()
    )
    stmt = (
        sa.select(Artifact, ranked.c.images)
        .join(ranked, ranked.c.id == Artifact.id)
        .where(ranked.c.rank == 1)
    )
    return {
        row.conversation_id: ImageSummary(count, row) for row, count in await session.execute(stmt)
    }


async def claim_uploads(
    session: AsyncSession, artifact_ids: Sequence[str], message_id: str
) -> None:
    """Record ``message_id`` as the origin of uploads that are not yet tied to a message."""
    if not artifact_ids:
        return
    await session.execute(
        sa.update(Artifact)
        .where(
            Artifact.id.in_(artifact_ids),
            Artifact.message_id.is_(None),
            Artifact.source == ArtifactSource.UPLOAD,
        )
        .values(message_id=message_id)
    )
