"""Background downloads from the Hub into the models folder.

* A file is written as ``<name>.part`` and renamed into place once whole, so a path an
  engine is pointed at is never half a file.
* An interrupted download (cancelled, dropped, or the server stopped) resumes from its
  ``.part`` with a range request the next time it is asked for, and transient network
  errors are retried the same way.
* LFS files are checked against the SHA-256 the Hub lists. A mismatch is thrown away.
* All of a job's files come from one commit, so shards and pipeline parts always match.
* Two jobs run at once and the rest wait. Asking again for something that is already being
  downloaded joins that job, and only adds where its result goes.
* Files are written from worker threads: a multi-gigabyte download never blocks the loop.
* A job can name a copy already on this machine (a LoRA in the HF cache): its files are
  copied from there, checked against the Hub's SHA-256 like a download, and any that is
  missing, different or unreadable is downloaded instead.
"""

import asyncio
import contextlib
import hashlib
import os
import shutil
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO, Literal

import httpx

from imaginer.core.clock import utcnow
from imaginer.core.ids import new_id
from imaginer.core.logging import get_logger
from imaginer.domain.errors import HubError, ImaginerError, NotFoundError
from imaginer.hub.client import HubClient
from imaginer.hub.layout import RepoFile
from imaginer.hub.local import PART_SUFFIX, bytes_under, safe_relpath

log = get_logger(__name__)

DownloadState = Literal["queued", "running", "succeeded", "failed", "cancelled"]
CHUNK_BYTES = 1 << 20
PUBLISH_EVERY_S = 0.25
RETRIES = 3
RETRY_BACKOFF_S = 1.0  # doubled each time, up to 8 s
KEEP_FINISHED = 50
SPACE_MARGIN_BYTES = 256 * 1024 * 1024


@dataclass(eq=False)
class DownloadJob:
    id: str
    repo_id: str
    revision: str  # a commit, so every file comes from the same one
    label: str
    files: tuple[RepoFile, ...]
    root: Path  # <models_dir>/<owner>/<name>
    target: Path  # what an engine is pointed at: the first file, or root for a pipeline
    assign: list[tuple[str, str]] = field(default_factory=list)  # (engine, launch setting)
    source: Path | None = None  # a local copy's root, copied from instead of downloading
    state: DownloadState = "queued"
    total_bytes: int = 0
    done_bytes: int = 0
    file_index: int = 0
    current: str | None = None
    speed_bps: float | None = None
    error: str | None = None
    warning: str | None = None
    started_at: datetime = field(default_factory=utcnow)
    finished_at: datetime | None = None

    @property
    def active(self) -> bool:
        return self.state in {"queued", "running"}


Listener = Callable[[DownloadJob], None]
Finished = Callable[[DownloadJob], Awaitable[None]]


class _Transient(Exception):
    """Worth another try, resuming where the last one stopped."""


class _Stopped(Exception):
    """A copy was stopped because its job was cancelled."""


class DownloadManager:
    def __init__(
        self,
        client: HubClient,
        *,
        concurrency: int = 2,
        on_success: Finished | None = None,
    ) -> None:
        self._client = client
        self._slots = asyncio.Semaphore(concurrency)
        self._on_success = on_success
        self._jobs: dict[str, DownloadJob] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._listeners: set[Listener] = set()
        self._file_locks: dict[Path, asyncio.Lock] = {}

    # --- views and control ----------------------------------------------------------

    def jobs(self) -> list[DownloadJob]:
        """Newest first."""
        return sorted(self._jobs.values(), key=lambda j: j.started_at, reverse=True)

    def get(self, job_id: str) -> DownloadJob:
        job = self._jobs.get(job_id)
        if job is None:
            raise NotFoundError(f"no download {job_id!r}")
        return job

    def subscribe(self, listener: Listener) -> Callable[[], None]:
        self._listeners.add(listener)
        return lambda: self._listeners.discard(listener)

    def _publish(self, job: DownloadJob) -> None:
        for listener in list(self._listeners):
            listener(job)

    def start(
        self,
        *,
        repo_id: str,
        revision: str,
        label: str,
        files: tuple[RepoFile, ...],
        root: Path,
        target: Path,
        assign: tuple[str, str] | None = None,
        source: Path | None = None,
    ) -> DownloadJob:
        for job in self._jobs.values():
            if job.active and job.target == target:
                if assign is not None and assign not in job.assign:
                    job.assign.append(assign)
                    self._publish(job)
                return job
        job = DownloadJob(
            id=new_id(),
            repo_id=repo_id,
            revision=revision,
            label=label,
            files=files,
            root=root,
            target=target,
            assign=[assign] if assign is not None else [],
            source=source,
            total_bytes=sum(f.size or 0 for f in files),
        )
        self._jobs[job.id] = job
        self._prune()
        self._tasks[job.id] = asyncio.create_task(self._run(job), name=f"download-{job.id}")
        self._publish(job)
        log.info(
            "hub.download_started",
            repo=repo_id,
            label=label,
            files=len(files),
            copy_from=str(source) if source else None,
        )
        return job

    def cancel(self, job_id: str) -> DownloadJob:
        job = self.get(job_id)
        task = self._tasks.get(job_id)
        if task is not None and not task.done():
            task.cancel()
        return job

    async def aclose(self) -> None:
        """Cancel every download; their ``.part`` files stay, to resume later."""
        tasks = [t for t in self._tasks.values() if not t.done()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def _prune(self) -> None:
        finished = [j for j in self.jobs() if not j.active]
        for job in finished[KEEP_FINISHED:]:
            self._jobs.pop(job.id, None)

    # --- the work --------------------------------------------------------------------

    async def _run(self, job: DownloadJob) -> None:
        try:
            async with self._slots:
                job.state = "running"
                self._publish(job)
                await self._check_space(job)
                done = 0
                for index, f in enumerate(job.files):
                    job.file_index, job.current = index, f.path
                    done += await self._fetch_with_retries(job, f, done)
                job.current = None
                # Set what the download was for before saying it succeeded, so a client that
                # sees it done also sees the setting it changed.
                if self._on_success is not None:
                    try:
                        await self._on_success(job)
                    except ImaginerError as exc:
                        job.warning = f"downloaded, but not set: {exc.detail}"
                job.state = "succeeded"
            log.info("hub.download_done", repo=job.repo_id, label=job.label)
        except asyncio.CancelledError:
            job.state = "cancelled"
            raise
        except ImaginerError as exc:
            job.state, job.error = "failed", exc.detail
        except Exception as exc:
            job.state, job.error = "failed", str(exc) or type(exc).__name__
            log.warning("hub.download_failed", repo=job.repo_id, error=job.error)
        finally:
            job.finished_at, job.speed_bps = utcnow(), None
            self._tasks.pop(job.id, None)
            self._publish(job)

    async def _check_space(self, job: DownloadJob) -> None:
        have = await asyncio.to_thread(bytes_under, job.root, job.files)
        need = job.total_bytes - have
        if need <= 0:
            return
        probe = job.root
        while not probe.exists() and probe.parent != probe:
            probe = probe.parent
        free = (await asyncio.to_thread(shutil.disk_usage, probe)).free
        if free < need + SPACE_MARGIN_BYTES:
            raise HubError(
                f"not enough disk space: {_gb(need)} to download, {_gb(free)} free at {probe}"
            )

    async def _fetch_with_retries(self, job: DownloadJob, f: RepoFile, done: int) -> int:
        for attempt in range(RETRIES + 1):
            try:
                return await self._fetch(job, f, done)
            except (_Transient, HubError) as exc:
                if attempt == RETRIES:
                    raise HubError(f"{f.path}: {exc}") from exc
                job.warning = f"retrying {f.path}: {exc}"
                self._publish(job)
                await asyncio.sleep(min(RETRY_BACKOFF_S * 2**attempt, 8.0))
        raise AssertionError("unreachable")

    async def _fetch(self, job: DownloadJob, f: RepoFile, done: int) -> int:
        """Download one file into place; returns its size."""
        dest = job.root / safe_relpath(f.path)
        async with self._file_locks.setdefault(dest, asyncio.Lock()):
            have = await asyncio.to_thread(_size, dest)
            if have is not None and (f.size is None or have == f.size):
                job.done_bytes = done + have
                return have
            part = dest.with_name(dest.name + PART_SUFFIX)
            await asyncio.to_thread(dest.parent.mkdir, parents=True, exist_ok=True)
            if job.source is not None:
                copied = await self._copy(job, f, done, dest, part)
                if copied is not None:
                    return copied
            offset = await asyncio.to_thread(_size, part) or 0
            if f.size is not None and offset > f.size:
                offset = 0
            hasher = hashlib.sha256() if f.sha256 else None
            if hasher is not None and offset:
                await asyncio.to_thread(_hash_into, hasher, part, offset)
            async with self._client.open(job.repo_id, job.revision, f.path, start=offset) as resp:
                if resp.status_code != 416:  # 416: nothing left after offset
                    if offset and resp.status_code != 206:  # the range was ignored
                        offset, hasher = 0, hashlib.sha256() if f.sha256 else None
                    await self._stream(job, resp, part, offset, hasher, done)
            got = await asyncio.to_thread(_size, part) or 0
            if f.size is not None and got != f.size:
                if got > f.size:
                    await asyncio.to_thread(part.unlink)
                raise _Transient(f"got {_gb(got)} of {_gb(f.size)}")
            if hasher is not None and hasher.hexdigest() != f.sha256:
                await asyncio.to_thread(part.unlink)
                raise _Transient("the download doesn't match the Hub's SHA-256; starting over")
            await asyncio.to_thread(os.replace, part, dest)
            job.done_bytes = done + got
            return got

    async def _copy(
        self, job: DownloadJob, f: RepoFile, done: int, dest: Path, part: Path
    ) -> int | None:
        """Copy one file from the job's local copy; returns its size, or None to download
        it instead (it isn't there at the Hub's size, can't be read, or doesn't match)."""
        assert job.source is not None
        src = job.source / safe_relpath(f.path)
        size = await asyncio.to_thread(_size, src)
        if size is None or (f.size is not None and size != f.size):
            return None
        hasher = hashlib.sha256() if f.sha256 else None
        stop = threading.Event()

        def run() -> None:
            with src.open("rb") as fh, part.open("wb") as sink:
                copied = 0
                while block := fh.read(CHUNK_BYTES * 8):
                    if stop.is_set():
                        raise _Stopped
                    _write(sink, hasher, block)
                    copied += len(block)
                    job.done_bytes = done + copied  # read by the loop as it publishes

        work = asyncio.ensure_future(asyncio.to_thread(run))
        last_at, last_bytes = time.monotonic(), job.done_bytes
        try:
            while not work.done():
                await asyncio.wait([work], timeout=PUBLISH_EVERY_S)
                now, got = time.monotonic(), job.done_bytes
                job.speed_bps = (got - last_bytes) / max(now - last_at, 1e-6)
                last_at, last_bytes = now, got
                self._publish(job)
            work.result()
        except asyncio.CancelledError:
            stop.set()
            with contextlib.suppress(BaseException):
                await work  # the file is closed before the job counts as cancelled
            raise
        except OSError as exc:
            job.warning = f"couldn't copy {f.path} ({exc}); downloading it"
            await asyncio.to_thread(_remove, part)
            return None
        if hasher is not None and hasher.hexdigest() != f.sha256:
            job.warning = f"the copy of {f.path} here doesn't match the Hub's; downloading it"
            await asyncio.to_thread(_remove, part)
            return None
        await asyncio.to_thread(os.replace, part, dest)
        job.done_bytes = done + size
        return size

    async def _stream(
        self,
        job: DownloadJob,
        resp: httpx.Response,
        part: Path,
        offset: int,
        hasher: Any,
        done: int,
    ) -> None:
        """Bytes count as they arrive and are written a megabyte at a time. Whatever arrived
        before an error or a cancel is still written, so the next try resumes after it."""
        sink = await asyncio.to_thread(_open, part, offset > 0)
        buffer = bytearray()
        writing: asyncio.Future[None] | None = None

        async def flush() -> None:
            nonlocal writing
            data = bytes(buffer)
            buffer.clear()
            # Shielded: a cancel can't leave the file closed under a write still running.
            writing = asyncio.ensure_future(asyncio.to_thread(_write, sink, hasher, data))
            await asyncio.shield(writing)

        received = offset
        last_at, last_bytes = time.monotonic(), received
        try:
            async for chunk in resp.aiter_bytes():
                buffer += chunk
                received += len(chunk)
                job.done_bytes = done + received
                if len(buffer) >= CHUNK_BYTES:
                    await flush()
                now = time.monotonic()
                if now - last_at >= PUBLISH_EVERY_S:
                    rate = (received - last_bytes) / (now - last_at)
                    job.speed_bps = (
                        rate if job.speed_bps is None else 0.7 * job.speed_bps + 0.3 * rate
                    )
                    last_at, last_bytes = now, received
                    self._publish(job)
        except httpx.HTTPError as exc:
            raise _Transient(str(exc) or type(exc).__name__) from exc
        finally:
            if writing is not None and not writing.done():
                await asyncio.wait([writing])
            if buffer:
                await flush()
            await asyncio.to_thread(sink.close)


def _size(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except OSError:
        return None


def _remove(path: Path) -> None:
    with contextlib.suppress(OSError):
        path.unlink()


def _open(path: Path, append: bool) -> BinaryIO:
    return path.open("ab" if append else "wb")


def _write(sink: BinaryIO, hasher: Any, chunk: bytes) -> None:
    sink.write(chunk)
    if hasher is not None:
        hasher.update(chunk)


def _hash_into(hasher: Any, path: Path, length: int) -> None:
    with path.open("rb") as fh:
        remaining = length
        while remaining > 0:
            block = fh.read(min(CHUNK_BYTES * 8, remaining))
            if not block:
                break
            hasher.update(block)
            remaining -= len(block)


def _gb(size: int) -> str:
    return f"{size / 1e9:.1f} GB" if size >= 1e8 else f"{max(size, 0) / 1e6:.1f} MB"
