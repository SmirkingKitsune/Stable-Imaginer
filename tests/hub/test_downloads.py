"""Background downloads against a fake Hub over real sockets: checksums, resumes, retries,
cancelling, and jobs that ask for the same thing."""

import asyncio
import hashlib
import shutil
import time
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace

import pytest

from imaginer.hub import downloads
from imaginer.hub.client import HubClient
from imaginer.hub.downloads import DownloadJob, DownloadManager
from imaginer.hub.layout import RepoFile
from tests.fakes.fake_hub import SHA, FakeHub
from tests.hub.conftest import LLM_REPO

PATH = "Chat-8B-Q4_K_M.gguf"


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(downloads, "RETRY_BACKOFF_S", 0.0)


@pytest.fixture
async def manager(hub_url: str) -> AsyncIterator[DownloadManager]:
    client = HubClient(hub_url)
    manager = DownloadManager(client)
    yield manager
    await manager.aclose()
    await client.aclose()


def _file(hub: FakeHub, path: str = PATH) -> RepoFile:
    data = hub.repos[LLM_REPO].files[path]
    return RepoFile(path, len(data), hashlib.sha256(data).hexdigest())


def _start(
    manager: DownloadManager, hub: FakeHub, root: Path, path: str = PATH, **kw
) -> DownloadJob:
    return manager.start(
        repo_id=LLM_REPO,
        revision=SHA,
        label=path,
        files=(_file(hub, path),),
        root=root,
        target=root / path,
        **kw,
    )


def _contents(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.iterdir())


async def _settled(job: DownloadJob) -> DownloadJob:
    deadline = time.monotonic() + 15
    while job.active:
        assert time.monotonic() < deadline, f"stuck: {job}"
        await asyncio.sleep(0.02)
    return job


async def test_a_download_is_checked_and_put_in_place(
    manager: DownloadManager, hub: FakeHub, tmp_path: Path
) -> None:
    job = await _settled(_start(manager, hub, tmp_path))
    assert (job.state, job.error) == ("succeeded", None)
    assert (tmp_path / PATH).read_bytes() == hub.repos[LLM_REPO].files[PATH]
    assert not (tmp_path / f"{PATH}.part").exists()
    assert job.done_bytes == job.total_bytes == 5000
    # From the pinned commit, through the CDN redirect, without a token to send.
    assert hub.requests == [{"path": f"{LLM_REPO}/{PATH}", "range": None, "auth": None}]


async def test_a_part_left_behind_is_resumed(
    manager: DownloadManager, hub: FakeHub, tmp_path: Path
) -> None:
    (tmp_path / f"{PATH}.part").write_bytes(hub.repos[LLM_REPO].files[PATH][:1234])
    job = await _settled(_start(manager, hub, tmp_path))
    assert job.state == "succeeded"
    assert hub.requests[0]["range"] == "bytes=1234-"
    assert (tmp_path / PATH).read_bytes() == hub.repos[LLM_REPO].files[PATH]


async def test_a_dropped_connection_resumes_where_it_stopped(
    manager: DownloadManager, hub: FakeHub, tmp_path: Path
) -> None:
    hub.drop_once[f"{LLM_REPO}/{PATH}"] = 2048
    job = await _settled(_start(manager, hub, tmp_path))
    assert job.state == "succeeded", job.error
    assert [r["range"] for r in hub.requests] == [None, "bytes=2048-"]
    assert job.warning is not None and "retrying" in job.warning
    assert (tmp_path / PATH).read_bytes() == hub.repos[LLM_REPO].files[PATH]


async def test_a_corrupt_download_is_thrown_away(
    manager: DownloadManager, hub: FakeHub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(downloads, "RETRIES", 1)
    hub.corrupt.add(f"{LLM_REPO}/{PATH}")
    job = await _settled(_start(manager, hub, tmp_path))
    assert job.state == "failed"
    assert job.error is not None and "SHA-256" in job.error
    assert _contents(tmp_path) == []
    assert len(hub.requests) == 2  # tried again from the start, once


async def test_cancelling_keeps_what_arrived(
    manager: DownloadManager, hub: FakeHub, tmp_path: Path
) -> None:
    hub.slow[f"{LLM_REPO}/{PATH}"] = 0.05
    job = _start(manager, hub, tmp_path)
    deadline = time.monotonic() + 10
    while job.done_bytes < 1024:
        assert time.monotonic() < deadline
        await asyncio.sleep(0.02)
    manager.cancel(job.id)
    await _settled(job)
    assert job.state == "cancelled"
    assert 0 < (tmp_path / f"{PATH}.part").stat().st_size < 5000
    assert not (tmp_path / PATH).exists()


async def test_asking_again_joins_the_running_download(
    manager: DownloadManager, hub: FakeHub, tmp_path: Path
) -> None:
    hub.slow[f"{LLM_REPO}/{PATH}"] = 0.01
    first = _start(manager, hub, tmp_path, assign=("llama", "model"))
    second = _start(manager, hub, tmp_path, assign=("sd", "llm"))
    assert second is first
    assert first.assign == [("llama", "model"), ("sd", "llm")]
    await _settled(first)
    assert len(hub.requests) == 1


async def test_a_file_already_in_place_is_not_fetched_again(
    manager: DownloadManager, hub: FakeHub, tmp_path: Path
) -> None:
    (tmp_path / PATH).write_bytes(hub.repos[LLM_REPO].files[PATH])
    job = await _settled(_start(manager, hub, tmp_path))
    assert (job.state, hub.requests) == ("succeeded", [])


async def test_a_full_disk_fails_before_downloading(
    manager: DownloadManager, hub: FakeHub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(shutil, "disk_usage", lambda _path: SimpleNamespace(free=10))
    job = await _settled(_start(manager, hub, tmp_path))
    assert job.state == "failed"
    assert job.error is not None and job.error.startswith("not enough disk space")
    assert hub.requests == []
