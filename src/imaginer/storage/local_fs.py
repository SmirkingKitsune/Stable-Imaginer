"""Artifacts on the local filesystem.

Layout: ``<root>/conversations/<conversation_id>/artifacts/<artifact_id>.<ext>``.
"""

import asyncio
import contextlib
import hashlib
import os
import re
import shutil
import stat
import time
from collections.abc import AsyncIterable, AsyncIterator, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any

from imaginer.core.logging import get_logger
from imaginer.domain.errors import PayloadTooLargeError
from imaginer.storage.base import ObjectInfo, StoredObject

_ID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_EXT_RE = re.compile(r"[a-z0-9]{1,8}")
# Final objects, plus the hidden temp files a crashed write can leave behind.
_KEY_RE = re.compile(r"artifacts/\.?[0-9a-f-]{36}\.[a-z0-9]{1,8}(\.tmp)?")

CHUNK_SIZE = 1 << 20

log = get_logger(__name__)


class LocalFSArtifactStore:
    def __init__(
        self, root: Path, *, delete_attempts: int = 6, delete_backoff_s: float = 0.05
    ) -> None:
        self.root = root
        self.conversations_root = root / "conversations"
        self._delete_attempts = delete_attempts
        self._delete_backoff_s = delete_backoff_s

    # --- paths ----------------------------------------------------------------

    def conversation_dir(self, conversation_id: str) -> Path:
        if not _ID_RE.fullmatch(conversation_id):
            raise ValueError(f"invalid conversation id: {conversation_id!r}")
        return self.conversations_root / conversation_id

    def _path(self, conversation_id: str, storage_key: str) -> Path:
        if not _KEY_RE.fullmatch(storage_key):
            raise ValueError(f"invalid storage key: {storage_key!r}")
        base = self.conversation_dir(conversation_id)
        path = base / storage_key
        # Belt and braces: the regexes already rule out traversal.
        if not path.resolve().is_relative_to(base.resolve()):
            raise ValueError(f"storage key escapes its conversation: {storage_key!r}")
        return path

    def local_path(self, conversation_id: str, storage_key: str) -> Path | None:
        return self._path(conversation_id, storage_key)

    # --- writes ---------------------------------------------------------------

    async def put(
        self,
        conversation_id: str,
        artifact_id: str,
        ext: str,
        data: bytes | AsyncIterable[bytes],
        *,
        max_bytes: int | None = None,
    ) -> StoredObject:
        if not _ID_RE.fullmatch(artifact_id):
            raise ValueError(f"invalid artifact id: {artifact_id!r}")
        if not _EXT_RE.fullmatch(ext):
            raise ValueError(f"invalid extension: {ext!r}")
        key = f"artifacts/{artifact_id}.{ext}"
        final = self._path(conversation_id, key)
        tmp = final.with_name(f".{final.name}.tmp")
        await asyncio.to_thread(final.parent.mkdir, parents=True, exist_ok=True)
        try:
            if isinstance(data, bytes):
                if max_bytes is not None and len(data) > max_bytes:
                    raise PayloadTooLargeError(f"artifact exceeds {max_bytes} bytes")
                digest, size = await asyncio.to_thread(_write_bytes, tmp, data)
            else:
                digest, size = await _write_stream(tmp, data, max_bytes)
            # Atomic on POSIX and Windows; readers never observe a partial file.
            await asyncio.to_thread(tmp.replace, final)
        except BaseException:
            await asyncio.to_thread(tmp.unlink, missing_ok=True)
            raise
        return StoredObject(storage_key=key, sha256=digest, size_bytes=size)

    async def delete_object(self, conversation_id: str, storage_key: str) -> None:
        path = self._path(conversation_id, storage_key)
        await asyncio.to_thread(self._retry, lambda: path.unlink(missing_ok=True))

    async def delete_conversation(self, conversation_id: str) -> None:
        path = self.conversation_dir(conversation_id)
        await asyncio.to_thread(self._retry, lambda: _rmtree(path))

    def _retry(self, op: Callable[[], None]) -> None:
        """Retry on PermissionError: on Windows a file that is open (e.g. being served)
        cannot be deleted until its handle closes."""
        for attempt in range(self._delete_attempts):
            try:
                op()
                return
            except PermissionError:
                if attempt == self._delete_attempts - 1:
                    raise
                time.sleep(self._delete_backoff_s * 2**attempt)

    # --- reads ----------------------------------------------------------------

    async def read_bytes(self, conversation_id: str, storage_key: str) -> bytes:
        path = self._path(conversation_id, storage_key)
        return await asyncio.to_thread(path.read_bytes)

    async def iter_bytes(
        self, conversation_id: str, storage_key: str, *, chunk_size: int = CHUNK_SIZE
    ) -> AsyncIterator[bytes]:
        path = self._path(conversation_id, storage_key)
        fh = await asyncio.to_thread(path.open, "rb")
        try:
            while chunk := await asyncio.to_thread(fh.read, chunk_size):
                yield chunk
        finally:
            await asyncio.to_thread(fh.close)

    async def list_conversation_ids(self) -> list[str]:
        def scan() -> list[str]:
            if not self.conversations_root.is_dir():
                return []
            return sorted(
                p.name
                for p in self.conversations_root.iterdir()
                if p.is_dir() and _ID_RE.fullmatch(p.name)
            )

        return await asyncio.to_thread(scan)

    async def list_objects(self, conversation_id: str) -> list[ObjectInfo]:
        directory = self.conversation_dir(conversation_id) / "artifacts"

        def scan() -> list[ObjectInfo]:
            if not directory.is_dir():
                return []
            infos = []
            for p in directory.iterdir():
                st = p.stat()
                if not stat.S_ISREG(st.st_mode):
                    continue
                infos.append(
                    ObjectInfo(
                        storage_key=f"artifacts/{p.name}",
                        size_bytes=st.st_size,
                        modified_at=datetime.fromtimestamp(st.st_mtime, UTC),
                    )
                )
            return infos

        return await asyncio.to_thread(scan)


def _write_bytes(path: Path, data: bytes) -> tuple[str, int]:
    with path.open("wb") as fh:
        fh.write(data)
        _flush_and_sync(fh)
    return hashlib.sha256(data).hexdigest(), len(data)


async def _write_stream(
    path: Path, chunks: AsyncIterable[bytes], max_bytes: int | None
) -> tuple[str, int]:
    hasher = hashlib.sha256()
    size = 0
    fh = await asyncio.to_thread(path.open, "wb")
    try:
        async for chunk in chunks:
            size += len(chunk)
            if max_bytes is not None and size > max_bytes:
                raise PayloadTooLargeError(f"artifact exceeds {max_bytes} bytes")
            hasher.update(chunk)
            await asyncio.to_thread(fh.write, chunk)
        await asyncio.to_thread(_flush_and_sync, fh)
    finally:
        await asyncio.to_thread(fh.close)
    return hasher.hexdigest(), size


def _flush_and_sync(fh: IO[bytes]) -> None:
    fh.flush()
    os.fsync(fh.fileno())


def _rmtree(path: Path) -> None:
    def on_error(func: Callable[..., Any], failed: str, exc: BaseException) -> None:
        # Read-only files (common on Windows) refuse deletion until made writable.
        if isinstance(exc, PermissionError):
            Path(failed).chmod(stat.S_IWRITE)
            func(failed)
        elif not isinstance(exc, FileNotFoundError):
            raise exc

    with contextlib.suppress(FileNotFoundError):
        shutil.rmtree(path, onexc=on_error)
