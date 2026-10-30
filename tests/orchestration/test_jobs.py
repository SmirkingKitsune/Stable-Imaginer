import asyncio
import threading
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from imaginer.backends.registry import BackendRegistry
from imaginer.backends.sdcpp import SdCppBackend
from imaginer.core.clock import utcnow
from imaginer.core.ids import new_id
from imaginer.db import models as orm
from imaginer.db.engine import Database
from imaginer.db.repositories import conversations as conv_repo
from imaginer.domain.enums import ArtifactSource, Capability, JobStatus, JobTask
from imaginer.domain.errors import ConflictError, EngineError, JobCancelledError
from imaginer.domain.events import EventType
from imaginer.orchestration.gates import EngineGates
from imaginer.orchestration.jobs import ORPHAN_ERROR, JobManager, JobSpec
from imaginer.services.artifacts import ArtifactService
from imaginer.storage.local_fs import LocalFSArtifactStore
from tests.conftest import SettingsFactory, sd_config
from tests.fakes.fake_sdcpp import FakeSdCpp


class Rig:
    def __init__(self, db: Database, backend: SdCppBackend, store_root: Path) -> None:
        self.db = db
        self.backend = backend
        self.artifacts = ArtifactService(
            db, LocalFSArtifactStore(store_root), max_upload_bytes=10**8
        )
        self.jobs = JobManager(db, self.artifacts, EngineGates({"sd": 1}))
        self.events: list[tuple[EventType, Any]] = []

    async def emit(self, event_type: EventType, data: BaseModel) -> None:
        self.events.append((event_type, data))

    async def conversation(self) -> str:
        cid, now = new_id(), utcnow()
        async with self.db.session() as s, s.begin():
            await conv_repo.insert(s, orm.Conversation(id=cid, created_at=now, updated_at=now))
        return cid

    def spec(self, cid: str, **kw: Any) -> JobSpec:
        fields: dict[str, Any] = {
            "conversation_id": cid,
            "task": JobTask.T2I,
            "backend": self.backend,
            "prompt": "a red fox",
            "seed": 100,
            "requested": {"prompt": "a red fox"},
        }
        fields.update(kw)
        return JobSpec(**fields)

    def statuses(self) -> list[str]:
        return [d.job.status.value for t, d in self.events if t is EventType.JOB_UPDATE]


@pytest.fixture
async def rig(database: Database, fake_sd: FakeSdCpp, tmp_path: Path) -> AsyncIterator[Rig]:
    backend = SdCppBackend("sd", sd_config(fake_sd.url))
    await backend.health()
    yield Rig(database, backend, tmp_path / "store")
    await backend.aclose()


async def test_lifecycle_outputs_and_seeds(rig: Rig) -> None:
    cid = await rig.conversation()
    job_id = await rig.jobs.submit(rig.spec(cid, n=2, aspect_ratio="1:1"), rig.emit)
    artifacts = await rig.jobs.wait(job_id)

    assert [a.label for a in artifacts] == ["img-1", "img-2"]
    assert {a.source for a in artifacts} == {ArtifactSource.GENERATED}
    assert [(a.metadata or {})["seed"] for a in artifacts] == [100, 101]
    statuses = rig.statuses()
    assert statuses[:2] == ["queued", "running"]
    assert statuses[-1] == "succeeded"
    created = [d.artifact.id for t, d in rig.events if t is EventType.ARTIFACT_CREATED]
    assert created == [a.id for a in artifacts]

    job = await rig.jobs.get(job_id)
    assert (job.status, job.task, job.backend, job.seed) == (
        JobStatus.SUCCEEDED,
        JobTask.T2I,
        "sd",
        100,
    )
    assert job.model == "qwen_image_2.1-Q4_K.gguf"
    assert [(o.ordinal, o.seed) for o in job.outputs] == [(0, 100), (1, 101)]
    assert job.params["requested"] == {"prompt": "a red fox"}
    assert job.params["resolved"]["width"] == 256 and job.params["resolved"]["steps"] == 3
    assert job.engine_job_id is not None
    assert job.started_at is not None and job.finished_at is not None


async def test_one_job_at_a_time_per_backend(rig: Rig, fake_sd: FakeSdCpp) -> None:
    fake_sd.hold = threading.Event()
    cid = await rig.conversation()
    first = await rig.jobs.submit(rig.spec(cid))
    second = await rig.jobs.submit(rig.spec(cid, seed=5))
    assert await asyncio.to_thread(fake_sd.started.wait, 10)
    await asyncio.sleep(0.1)
    assert (await rig.jobs.get(first)).status is JobStatus.RUNNING
    assert (await rig.jobs.get(second)).status is JobStatus.QUEUED
    assert len(fake_sd.requests) == 1  # the engine never saw the second job yet
    fake_sd.hold.set()
    await rig.jobs.wait(first)
    await rig.jobs.wait(second)
    assert len(fake_sd.requests) == 2


async def test_cancel_running_job(rig: Rig, fake_sd: FakeSdCpp) -> None:
    fake_sd.hold = threading.Event()
    cid = await rig.conversation()
    job_id = await rig.jobs.submit(rig.spec(cid), rig.emit)
    assert await asyncio.to_thread(fake_sd.started.wait, 10)
    await asyncio.sleep(0.05)
    assert await rig.jobs.cancel(job_id) is True
    job = await rig.jobs.get(job_id)
    assert job.status is JobStatus.CANCELLED
    assert fake_sd.cancels == [job.engine_job_id]
    assert rig.statuses()[-1] == "cancelled"
    # The waiter was not cancelled itself, so it learns the job was, as a domain error.
    with pytest.raises(JobCancelledError):
        await rig.jobs.wait(job_id)
    assert await rig.jobs.cancel(job_id) is False


async def test_cancel_queued_job_never_reaches_the_engine(rig: Rig, fake_sd: FakeSdCpp) -> None:
    fake_sd.hold = threading.Event()
    cid = await rig.conversation()
    running = await rig.jobs.submit(rig.spec(cid))
    queued = await rig.jobs.submit(rig.spec(cid))
    assert await asyncio.to_thread(fake_sd.started.wait, 10)
    await rig.jobs.cancel(queued)
    assert (await rig.jobs.get(queued)).status is JobStatus.CANCELLED
    fake_sd.hold.set()
    await rig.jobs.wait(running)
    assert len(fake_sd.requests) == 1


async def test_a_job_cancelled_before_it_runs_is_recorded_as_cancelled(
    rig: Rig, fake_sd: FakeSdCpp
) -> None:
    cid = await rig.conversation()
    job_id = await rig.jobs.submit(rig.spec(cid), rig.emit)
    assert await rig.jobs.cancel(job_id) is True  # before the job's task had a turn
    assert (await rig.jobs.get(job_id)).status is JobStatus.CANCELLED
    assert rig.statuses()[-1] == "cancelled"
    assert fake_sd.requests == []


async def test_engine_failure_is_recorded(rig: Rig, fake_sd: FakeSdCpp) -> None:
    fake_sd.fail_next = "CUDA out of memory"
    cid = await rig.conversation()
    job_id = await rig.jobs.submit(rig.spec(cid), rig.emit)
    with pytest.raises(EngineError):
        await rig.jobs.wait(job_id)
    job = await rig.jobs.get(job_id)
    assert job.status is JobStatus.FAILED
    assert job.error is not None and "CUDA out of memory" in job.error
    assert job.outputs == []


async def test_cancel_conversation_jobs(rig: Rig, fake_sd: FakeSdCpp) -> None:
    fake_sd.hold = threading.Event()
    a, b = await rig.conversation(), await rig.conversation()
    job_a = await rig.jobs.submit(rig.spec(a))
    assert await asyncio.to_thread(fake_sd.started.wait, 10)
    job_b = await rig.jobs.submit(rig.spec(b))
    await rig.jobs.cancel_conversation(a)
    assert (await rig.jobs.get(job_a)).status is JobStatus.CANCELLED
    fake_sd.hold.set()
    await rig.jobs.wait(job_b)
    assert (await rig.jobs.get(job_b)).status is JobStatus.SUCCEEDED


async def test_orphans_fail_on_restart(rig: Rig) -> None:
    cid = await rig.conversation()
    now = utcnow()
    async with rig.db.session() as s, s.begin():
        for status in (JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.SUCCEEDED):
            s.add(
                orm.GenerationJob(
                    id=new_id(),
                    conversation_id=cid,
                    task=JobTask.T2I,
                    backend="sd",
                    model="m",
                    params_json={},
                    status=status,
                    queued_at=now,
                )
            )
    failed = await rig.jobs.fail_orphans()
    assert len(failed) == 2
    page = await rig.jobs.list_jobs(cid, limit=10)
    assert sorted(j.status.value for j in page.items) == ["failed", "failed", "succeeded"]
    assert {j.error for j in page.items if j.status is JobStatus.FAILED} == {ORPHAN_ERROR}


async def test_cancelling_the_waiter_cancels_the_job(rig: Rig, fake_sd: FakeSdCpp) -> None:
    fake_sd.hold = threading.Event()
    cid = await rig.conversation()
    job_id = await rig.jobs.submit(rig.spec(cid))
    waiter = asyncio.create_task(rig.jobs.wait(job_id))
    assert await asyncio.to_thread(fake_sd.started.wait, 10)
    await asyncio.sleep(0.05)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):  # its own cancellation, not JobCancelledError
        await waiter
    for _ in range(100):
        if (await rig.jobs.get(job_id)).status is JobStatus.CANCELLED:
            break
        await asyncio.sleep(0.01)
    assert (await rig.jobs.get(job_id)).status is JobStatus.CANCELLED
    assert len(fake_sd.cancels) == 1


async def test_cancel_job_api_semantics(rig: Rig, fake_sd: FakeSdCpp) -> None:
    cid = await rig.conversation()
    done = await rig.jobs.submit(rig.spec(cid))
    await rig.jobs.wait(done)
    with pytest.raises(ConflictError, match="already succeeded"):
        await rig.jobs.cancel_job(done)

    fake_sd.hold = threading.Event()
    running = await rig.jobs.submit(rig.spec(cid))
    assert await asyncio.to_thread(fake_sd.started.wait, 10)
    job = await rig.jobs.cancel_job(running)
    assert job.status is JobStatus.CANCELLED


async def test_partial_outputs_survive_a_failure(rig: Rig, fake_sd: FakeSdCpp) -> None:
    fake_sd.fail_at_request = 2
    cid = await rig.conversation()
    job_id = await rig.jobs.submit(rig.spec(cid, n=3))
    with pytest.raises(EngineError):
        await rig.jobs.wait(job_id)
    outputs = await rig.jobs.outputs(job_id)
    assert [a.label for a in outputs] == ["img-1"]
    assert (await rig.jobs.get(job_id)).status is JobStatus.FAILED


async def test_a_running_job_keeps_its_adapter_through_a_reconfigure(
    rig: Rig, fake_sd: FakeSdCpp, make_settings: SettingsFactory
) -> None:
    registry = BackendRegistry.from_settings(make_settings(backends={"sd": sd_config(fake_sd.url)}))
    lease = await registry.image_backend("t2i")
    old = lease.backend
    fake_sd.hold = threading.Event()
    cid = await rig.conversation()
    job_id = await rig.jobs.submit(rig.spec(cid, backend=old, lease=lease))
    assert await asyncio.to_thread(fake_sd.started.wait, 10)

    # e.g. the Engines screen changes a capability while the job runs
    changed = registry.config("sd").model_copy(update={"declared_capabilities": {Capability.T2I}})
    assert registry.reconfigure("sd", changed) is True
    await asyncio.sleep(0.1)  # the job keeps polling the engine through the old adapter
    fake_sd.hold.set()
    assert len(await rig.jobs.wait(job_id)) == 1
    await asyncio.sleep(0)
    with pytest.raises(RuntimeError, match="closed"):
        await old.health()  # released by the job, so the old adapter was closed
    await registry.aclose()
