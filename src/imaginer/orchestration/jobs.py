"""Generation jobs: persisted, queued per backend, cancellable.

* Every state change is written to ``generation_job`` and, when an emitter is attached,
  published as a ``job_update`` event. New artifacts are published as ``artifact_created``.
* A per-backend gate (``concurrency``, default 1) queues jobs in front of each GPU. Linked
  instances using a shared engine wait at the same gate (``orchestration/gates.py``).
* ``n`` images are made as ``n`` single-image engine calls with seeds ``seed, seed+1, ...``.
  Each output's seed is recorded in ``job_output``, so any one image can be regenerated.
* Cancelling the job's task (directly, or by cancelling the turn awaiting it) cancels the
  engine job too, and the job is recorded as ``cancelled``.
* After a restart nothing is running the jobs left ``queued``/``running``, so startup marks
  them ``failed``.
"""

import asyncio
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from imaginer.backends.base import (
    EditRequest,
    GenProgress,
    GenRequest,
    ImageBackend,
    ImageInput,
    ImageResult,
)
from imaginer.backends.registry import Lease
from imaginer.core.clock import utcnow
from imaginer.core.ids import new_id
from imaginer.core.logging import get_logger
from imaginer.db import models as orm
from imaginer.db.engine import Database
from imaginer.db.repositories import conversations as conv_repo
from imaginer.db.repositories import jobs as jobs_repo
from imaginer.db.repositories.artifacts import ParentRef
from imaginer.domain import models as dto
from imaginer.domain.enums import ArtifactSource, JobStatus, JobTask
from imaginer.domain.errors import (
    ConflictError,
    ImaginerError,
    JobCancelledError,
    NotFoundError,
)
from imaginer.domain.events import ArtifactCreatedEventData, EventType, JobUpdateEventData
from imaginer.domain.models import SEED_MAX
from imaginer.orchestration.gates import EngineGates
from imaginer.services import mappers
from imaginer.services.artifacts import ArtifactService, NewArtifact
from imaginer.services.media import EXTENSIONS, ImageInfo

log = get_logger(__name__)

CANCEL_WAIT_S = 10.0
ORPHAN_ERROR = "orphaned: the server restarted while this job was unfinished"

Emit = Callable[[EventType, BaseModel], Awaitable[None]]


def random_seed() -> int:
    return random.randint(0, SEED_MAX)


@dataclass(frozen=True, kw_only=True)
class JobSpec:
    conversation_id: str
    task: JobTask
    backend: ImageBackend
    prompt: str
    seed: int
    n: int = 1
    negative_prompt: str | None = None
    width: int | None = None
    height: int | None = None
    aspect_ratio: str | None = None
    alpha: bool = False
    params: dict[str, Any] = field(default_factory=dict)
    images: tuple[ImageInput, ...] = ()  # edit inputs, in order
    mask: ImageInput | None = None
    parents: tuple[ParentRef, ...] = ()  # lineage of every output
    requested: dict[str, Any] = field(default_factory=dict)  # as asked (e.g. tool arguments)
    request_message_id: str | None = None
    tool_call_id: str | None = None
    # The lease the backend was chosen with. The job owns it once submitted and releases
    # it when it ends, so a reconfigure doesn't close the adapter under a queued job.
    lease: Lease[ImageBackend] | None = None


class JobManager:
    def __init__(self, db: Database, artifacts: ArtifactService, gates: EngineGates) -> None:
        self._db = db
        self._artifacts = artifacts
        self._gates = gates
        self._tasks: dict[str, asyncio.Task[list[dto.Artifact]]] = {}
        self._owners: dict[str, str] = {}  # job id -> conversation id

    # --- lifecycle ------------------------------------------------------------

    async def fail_orphans(self) -> list[str]:
        async with self._db.session() as session, session.begin():
            failed = await jobs_repo.fail_unfinished(session, ORPHAN_ERROR, utcnow())
        if failed:
            log.warning("jobs.orphans_failed", count=len(failed), job_ids=failed)
        return failed

    async def submit(self, spec: JobSpec, emit: Emit | None = None) -> str:
        try:
            return await self._submit(spec, emit)
        except BaseException:
            if spec.lease is not None:
                spec.lease.release()
            raise

    async def _submit(self, spec: JobSpec, emit: Emit | None) -> str:
        job_id = new_id()
        row = orm.GenerationJob(
            id=job_id,
            conversation_id=spec.conversation_id,
            request_message_id=spec.request_message_id,
            tool_call_id=spec.tool_call_id,
            task=spec.task,
            alpha=spec.alpha,
            backend=spec.backend.name,
            model=spec.backend.model,
            params_json={"requested": spec.requested, "n": spec.n},
            seed=spec.seed,
            status=JobStatus.QUEUED,
            queued_at=utcnow(),
        )
        async with self._db.session() as session, session.begin():
            await conv_repo.require_active(session, spec.conversation_id)
            await jobs_repo.insert(session, row)
        log.info("job.queued", job_id=job_id, task=spec.task.value, backend=spec.backend.name)
        await self._publish(job_id, emit)
        self._owners[job_id] = spec.conversation_id
        # Started eagerly, so it is inside _run's handlers before anything can cancel it. A
        # task cancelled before its first step never runs them, and the job would stay queued.
        task = asyncio.Task(
            self._run(job_id, spec, emit),
            loop=asyncio.get_running_loop(),
            name=f"job-{job_id}",
            eager_start=True,
        )
        if spec.lease is not None:
            lease = spec.lease
            task.add_done_callback(lambda _: lease.release())
        self._tasks[job_id] = task
        return job_id

    async def wait(self, job_id: str) -> list[dto.Artifact]:
        """The job's artifacts once it succeeds; raises its error otherwise.

        Cancelling the waiter cancels the job (and its engine job). If instead the job alone
        is cancelled (``POST /v1/jobs/{id}/cancel``), the waiter gets ``JobCancelledError``
        rather than a ``CancelledError`` that would look like its own cancellation.
        """
        task = self._tasks.get(job_id)
        if task is None:
            raise NotFoundError(f"job {job_id} is not running")
        try:
            return await task
        except asyncio.CancelledError:
            waiter = asyncio.current_task()
            if waiter is not None and waiter.cancelling():
                raise  # we are being cancelled ourselves
            raise JobCancelledError(f"job {job_id} was cancelled", job_id=job_id) from None
        finally:
            if task.done():
                self._forget(job_id)

    async def cancel(self, job_id: str) -> bool:
        task = self._tasks.get(job_id)
        if task is None or task.done():
            return False
        task.cancel()
        await asyncio.wait({task}, timeout=CANCEL_WAIT_S)
        return True

    async def cancel_job(self, job_id: str) -> dto.Job:
        """Cancel one job on request (the API). 409 if it already finished."""
        job = await self.get(job_id)
        if job.status in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED):
            raise ConflictError(f"job {job_id} already {job.status.value}", job_status=job.status)
        if not await self.cancel(job_id):
            # Unfinished in the database but not running here: nothing else will finish it.
            await self._update(
                job_id, None, status=JobStatus.CANCELLED, finished_at=utcnow(), error="cancelled"
            )
        log.info("job.cancel_requested", job_id=job_id)
        return await self.get(job_id)

    async def cancel_conversation(self, conversation_id: str) -> None:
        for job_id, owner in list(self._owners.items()):
            if owner == conversation_id:
                await self.cancel(job_id)

    async def cancel_all(self) -> None:
        for job_id in list(self._tasks):
            await self.cancel(job_id)

    def _forget(self, job_id: str) -> None:
        self._tasks.pop(job_id, None)
        self._owners.pop(job_id, None)

    # --- queries --------------------------------------------------------------

    async def outputs(self, job_id: str) -> list[dto.Artifact]:
        """The artifacts a job has produced so far, in order."""
        job = await self.get(job_id)
        return [await self._artifacts.get(o.artifact_id) for o in job.outputs]

    async def get(self, job_id: str) -> dto.Job:
        async with self._db.session() as session, session.begin():
            row = await jobs_repo.get_active(session, job_id)
            if row is None:
                raise NotFoundError(f"job {job_id} not found")
            outputs = await jobs_repo.outputs_for(session, [job_id])
        return mappers.job(row, outputs.get(job_id))

    async def list_jobs(
        self,
        conversation_id: str,
        *,
        status: JobStatus | None = None,
        limit: int,
        cursor: str | None = None,
    ) -> dto.Page[dto.Job]:
        async with self._db.session() as session, session.begin():
            await conv_repo.require_active(session, conversation_id)
            rows = await jobs_repo.list_for_conversation(
                session, conversation_id, status=status, after_id=cursor, limit=limit + 1
            )
            has_more = len(rows) > limit
            rows = rows[:limit]
            outputs = await jobs_repo.outputs_for(session, [r.id for r in rows])
        return dto.Page[dto.Job](
            items=[mappers.job(r, outputs.get(r.id)) for r in rows],
            next_cursor=rows[-1].id if has_more else None,
        )

    # --- execution ------------------------------------------------------------

    async def _run(self, job_id: str, spec: JobSpec, emit: Emit | None) -> list[dto.Artifact]:
        outputs: list[dto.Artifact] = []
        try:
            async with self._gates.hold(spec.backend.name):
                await self._update(job_id, emit, status=JobStatus.RUNNING, started_at=utcnow())
                resolved: dict[str, Any] | None = None
                for index in range(spec.n):
                    request = self._request(job_id, spec, index, emit)
                    if spec.task is JobTask.EDIT:
                        assert isinstance(request, EditRequest)
                        results = await spec.backend.edit(request)
                    else:
                        results = await spec.backend.generate(request)
                    for result in results:
                        artifact = await self._store(job_id, spec, result, len(outputs))
                        outputs.append(artifact)
                        resolved = resolved or result.params
                        if emit is not None:
                            await emit(
                                EventType.ARTIFACT_CREATED,
                                ArtifactCreatedEventData(artifact=artifact, job_id=job_id),
                            )
                await self._update(
                    job_id,
                    emit,
                    status=JobStatus.SUCCEEDED,
                    finished_at=utcnow(),
                    params_json={"requested": spec.requested, "n": spec.n, "resolved": resolved},
                    progress_json=await self._final_progress(job_id),
                )
                log.info("job.succeeded", job_id=job_id, outputs=len(outputs))
                return outputs
        except asyncio.CancelledError:
            log.info("job.cancelled", job_id=job_id, outputs=len(outputs))
            await self._update(
                job_id, emit, status=JobStatus.CANCELLED, finished_at=utcnow(), error="cancelled"
            )
            raise
        except ImaginerError as exc:
            log.warning("job.failed", job_id=job_id, code=exc.code, detail=exc.detail)
            await self._update(
                job_id, emit, status=JobStatus.FAILED, finished_at=utcnow(), error=exc.detail
            )
            raise
        except Exception as exc:
            log.exception("job.crashed", job_id=job_id)
            await self._update(
                job_id,
                emit,
                status=JobStatus.FAILED,
                finished_at=utcnow(),
                error=f"internal error: {exc!r}",
            )
            raise

    def _request(
        self, job_id: str, spec: JobSpec, index: int, emit: Emit | None
    ) -> GenRequest | EditRequest:
        async def on_progress(progress: GenProgress) -> None:
            await self._update(
                job_id,
                emit,
                progress_json={
                    "image": index + 1,
                    "images": spec.n,
                    "step": progress.step,
                    "total_steps": progress.total_steps,
                },
            )

        async def on_submitted(engine_job_id: str) -> None:
            await self._update(job_id, None, engine_job_id=engine_job_id)

        common: dict[str, Any] = {
            "prompt": spec.prompt,
            "seed": (spec.seed + index) % (SEED_MAX + 1),
            "negative_prompt": spec.negative_prompt,
            "width": spec.width,
            "height": spec.height,
            "aspect_ratio": spec.aspect_ratio,
            "alpha": spec.alpha,
            "params": spec.params,
            "on_progress": on_progress,
            "on_submitted": on_submitted,
        }
        if spec.task is JobTask.EDIT:
            return EditRequest(images=spec.images, mask=spec.mask, **common)
        return GenRequest(**common)

    async def _store(
        self, job_id: str, spec: JobSpec, result: ImageResult, ordinal: int
    ) -> dto.Artifact:
        info = ImageInfo(
            mime=result.mime,
            ext=EXTENSIONS[result.mime],
            width=result.width,
            height=result.height,
            has_alpha=result.has_alpha,
        )
        source = ArtifactSource.EDITED if spec.task is JobTask.EDIT else ArtifactSource.GENERATED
        new = NewArtifact.image(
            spec.conversation_id,
            info,
            source,
            message_id=spec.request_message_id,
            parents=spec.parents,
            metadata={"job_id": job_id, "prompt": spec.prompt, "seed": result.seed},
        )

        async def link(session: AsyncSession, row: orm.Artifact) -> None:
            await jobs_repo.add_output(session, job_id, row.id, ordinal, result.seed)

        return await self._artifacts.ingest(new, result.data, session_hook=link)

    async def _final_progress(self, job_id: str) -> dict[str, Any] | None:
        """Progress as of completion: polling may have missed the engine's last step."""
        async with self._db.session() as session, session.begin():
            row = await jobs_repo.get(session, job_id)
        progress = dict(row.progress_json or {}) if row is not None else {}
        if not progress:
            return None
        progress["image"] = progress.get("images", progress.get("image"))
        progress["step"] = progress.get("total_steps", progress.get("step"))
        return progress

    async def _update(self, job_id: str, emit: Emit | None, **values: Any) -> None:
        async with self._db.session() as session, session.begin():
            await jobs_repo.update(session, job_id, **values)
        await self._publish(job_id, emit)

    async def _publish(self, job_id: str, emit: Emit | None) -> None:
        if emit is None:
            return
        try:
            job = await self.get(job_id)
        except NotFoundError:  # conversation deleted under us
            return
        await emit(EventType.JOB_UPDATE, JobUpdateEventData(job=job))
