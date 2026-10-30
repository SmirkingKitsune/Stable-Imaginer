import hashlib
import sys
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from imaginer.core.ids import new_id
from imaginer.domain.errors import PayloadTooLargeError
from imaginer.storage import local_fs
from imaginer.storage.local_fs import LocalFSArtifactStore


@pytest.fixture
def store(tmp_path: Path) -> LocalFSArtifactStore:
    return LocalFSArtifactStore(tmp_path, delete_attempts=3, delete_backoff_s=0.001)


async def _chunks(*parts: bytes) -> AsyncIterator[bytes]:
    for part in parts:
        yield part


async def test_put_bytes_layout_and_digest(store: LocalFSArtifactStore, tmp_path: Path) -> None:
    cid, aid = new_id(), new_id()
    stored = await store.put(cid, aid, "png", b"hello")
    assert stored.storage_key == f"artifacts/{aid}.png"
    assert stored.sha256 == hashlib.sha256(b"hello").hexdigest()
    assert stored.size_bytes == 5
    path = tmp_path / "conversations" / cid / "artifacts" / f"{aid}.png"
    assert path.read_bytes() == b"hello"
    assert store.local_path(cid, stored.storage_key) == path
    assert await store.read_bytes(cid, stored.storage_key) == b"hello"


async def test_put_stream(store: LocalFSArtifactStore) -> None:
    cid = new_id()
    stored = await store.put(cid, new_id(), "bin", _chunks(b"ab", b"cd", b"e"))
    assert stored.sha256 == hashlib.sha256(b"abcde").hexdigest()
    got = b"".join([c async for c in store.iter_bytes(cid, stored.storage_key, chunk_size=2)])
    assert got == b"abcde"


async def test_stream_over_limit_leaves_nothing_behind(store: LocalFSArtifactStore) -> None:
    cid = new_id()
    with pytest.raises(PayloadTooLargeError):
        await store.put(cid, new_id(), "bin", _chunks(b"x" * 6, b"x" * 6), max_bytes=10)
    assert await store.list_objects(cid) == []


async def test_bytes_over_limit(store: LocalFSArtifactStore) -> None:
    with pytest.raises(PayloadTooLargeError):
        await store.put(new_id(), new_id(), "bin", b"x" * 11, max_bytes=10)


@pytest.mark.parametrize(
    "key",
    ["../x.png", "artifacts/../../evil.png", "artifacts/x.png", "/etc/passwd", "artifacts\\a.png"],
)
def test_rejects_bad_keys(store: LocalFSArtifactStore, key: str) -> None:
    with pytest.raises(ValueError):
        store.local_path(new_id(), key)


@pytest.mark.parametrize("cid", ["..", "abc", "../" + new_id(), ""])
def test_rejects_bad_conversation_ids(store: LocalFSArtifactStore, cid: str) -> None:
    with pytest.raises(ValueError):
        store.conversation_dir(cid)


async def test_rejects_bad_extension(store: LocalFSArtifactStore) -> None:
    with pytest.raises(ValueError):
        await store.put(new_id(), new_id(), "../png", b"x")


async def test_delete_conversation_is_idempotent(store: LocalFSArtifactStore) -> None:
    cid = new_id()
    await store.put(cid, new_id(), "png", b"x")
    assert await store.list_conversation_ids() == [cid]
    await store.delete_conversation(cid)
    await store.delete_conversation(cid)
    await store.delete_conversation(new_id())
    assert await store.list_conversation_ids() == []


async def test_delete_retries_transient_permission_errors(
    store: LocalFSArtifactStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    cid = new_id()
    await store.put(cid, new_id(), "png", b"x")
    real = local_fs._rmtree
    failures = iter([True, True])

    def flaky(path: Path) -> None:
        if next(failures, False):
            raise PermissionError("file in use")
        real(path)

    monkeypatch.setattr(local_fs, "_rmtree", flaky)
    await store.delete_conversation(cid)
    assert not store.conversation_dir(cid).exists()


async def test_delete_gives_up_after_attempts(
    store: LocalFSArtifactStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    def always_locked(path: Path) -> None:
        raise PermissionError("file in use")

    monkeypatch.setattr(local_fs, "_rmtree", always_locked)
    with pytest.raises(PermissionError):
        await store.delete_conversation(new_id())


@pytest.mark.skipif(sys.platform != "win32", reason="only Windows refuses to delete open files")
async def test_windows_open_file_blocks_delete(store: LocalFSArtifactStore) -> None:
    cid = new_id()
    stored = await store.put(cid, new_id(), "png", b"x")
    path = store.local_path(cid, stored.storage_key)
    assert path is not None
    with path.open("rb"), pytest.raises(PermissionError):
        await store.delete_conversation(cid)
    await store.delete_conversation(cid)  # handle closed: succeeds now
    assert not store.conversation_dir(cid).exists()


async def test_list_objects_includes_leftover_temp_files(store: LocalFSArtifactStore) -> None:
    cid, aid = new_id(), new_id()
    await store.put(cid, aid, "png", b"x")
    leftover = store.conversation_dir(cid) / "artifacts" / f".{new_id()}.png.tmp"
    leftover.write_bytes(b"partial")
    keys = {o.storage_key for o in await store.list_objects(cid)}
    assert keys == {f"artifacts/{aid}.png", f"artifacts/{leftover.name}"}
    await store.delete_object(cid, f"artifacts/{leftover.name}")
    await store.delete_object(cid, f"artifacts/{leftover.name}")  # idempotent
    assert len(await store.list_objects(cid)) == 1
