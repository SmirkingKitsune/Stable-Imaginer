from typing import Annotated

from fastapi import APIRouter, Query

from imaginer.api.deps import ContainerDep
from imaginer.domain import models as dto
from imaginer.domain.enums import JobStatus

conversation_router = APIRouter(prefix="/v1/conversations/{conversation_id}/jobs", tags=["jobs"])
router = APIRouter(prefix="/v1/jobs", tags=["jobs"])


@conversation_router.get("")
async def list_jobs(
    conversation_id: str,
    container: ContainerDep,
    status: JobStatus | None = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    cursor: Annotated[str | None, Query(description="next_cursor from the previous page")] = None,
) -> dto.Page[dto.Job]:
    """Generation jobs of one conversation, oldest first."""
    return await container.jobs.list_jobs(
        conversation_id, status=status, limit=limit, cursor=cursor
    )


@router.get("/{job_id}")
async def get_job(job_id: str, container: ContainerDep) -> dto.Job:
    """A job with its parameters (requested and resolved), per-output seeds and progress."""
    return await container.jobs.get(job_id)


@router.post("/{job_id}/cancel")
async def cancel_job(job_id: str, container: ContainerDep) -> dto.Job:
    """Cancel a queued or running job, including its engine-side job.

    The turn that started it keeps going: the model is told the job was cancelled and
    answers accordingly. Outputs already produced (for ``n`` > 1) are kept. 409 if the
    job has already finished.
    """
    return await container.jobs.cancel_job(job_id)
