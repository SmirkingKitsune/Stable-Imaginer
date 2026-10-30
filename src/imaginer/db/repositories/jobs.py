from collections.abc import Sequence
from datetime import datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from imaginer.db.models import Conversation, GenerationJob, JobOutput
from imaginer.domain.enums import JobStatus

_UNFINISHED = (JobStatus.QUEUED, JobStatus.RUNNING)


async def insert(session: AsyncSession, row: GenerationJob) -> None:
    session.add(row)
    await session.flush()


async def get(session: AsyncSession, job_id: str) -> GenerationJob | None:
    return await session.get(GenerationJob, job_id)


async def get_active(session: AsyncSession, job_id: str) -> GenerationJob | None:
    """A job whose conversation is not being deleted."""
    stmt = (
        sa.select(GenerationJob)
        .join(Conversation, Conversation.id == GenerationJob.conversation_id)
        .where(GenerationJob.id == job_id, Conversation.deleted_at.is_(None))
    )
    return (await session.execute(stmt)).scalar_one_or_none()


async def update(session: AsyncSession, job_id: str, **values: Any) -> None:
    await session.execute(
        sa.update(GenerationJob).where(GenerationJob.id == job_id).values(**values)
    )


async def list_for_conversation(
    session: AsyncSession,
    conversation_id: str,
    *,
    status: JobStatus | None = None,
    after_id: str | None = None,
    limit: int,
) -> list[GenerationJob]:
    stmt = sa.select(GenerationJob).where(GenerationJob.conversation_id == conversation_id)
    if status is not None:
        stmt = stmt.where(GenerationJob.status == status)
    if after_id is not None:
        stmt = stmt.where(GenerationJob.id > after_id)
    stmt = stmt.order_by(GenerationJob.id).limit(limit)
    return list((await session.execute(stmt)).scalars())


async def add_output(
    session: AsyncSession, job_id: str, artifact_id: str, ordinal: int, seed: int | None
) -> None:
    session.add(JobOutput(job_id=job_id, artifact_id=artifact_id, ordinal=ordinal, seed=seed))
    await session.flush()


async def outputs_for(session: AsyncSession, job_ids: Sequence[str]) -> dict[str, list[JobOutput]]:
    if not job_ids:
        return {}
    stmt = (
        sa.select(JobOutput)
        .where(JobOutput.job_id.in_(job_ids))
        .order_by(JobOutput.job_id, JobOutput.ordinal)
    )
    result: dict[str, list[JobOutput]] = {}
    for row in (await session.execute(stmt)).scalars():
        result.setdefault(row.job_id, []).append(row)
    return result


async def fail_unfinished(session: AsyncSession, error: str, now: datetime) -> list[str]:
    """Mark every queued/running job failed (after a restart nothing is running them)."""
    stmt = (
        sa.update(GenerationJob)
        .where(GenerationJob.status.in_(_UNFINISHED))
        .values(status=JobStatus.FAILED, error=error, finished_at=now)
        .returning(GenerationJob.id)
    )
    return list((await session.execute(stmt)).scalars())
