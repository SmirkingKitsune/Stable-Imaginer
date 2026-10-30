from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True, slots=True)
class StoredObject:
    storage_key: str  # relative to the conversation's folder, e.g. "artifacts/<id>.png"
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class ObjectInfo:
    storage_key: str
    size_bytes: int
    modified_at: datetime


class ArtifactStore(Protocol):
    """Bytes for artifacts, grouped by conversation.

    Keys are relative to a conversation, so deleting a conversation is one prefix delete
    and an artifact can never be read through another conversation's id.
    """

    async def put(
        self,
        conversation_id: str,
        artifact_id: str,
        ext: str,
        data: bytes | AsyncIterable[bytes],
        *,
        max_bytes: int | None = None,
    ) -> StoredObject:
        """Write atomically. Raises ``PayloadTooLargeError`` past ``max_bytes``."""
        ...

    async def read_bytes(self, conversation_id: str, storage_key: str) -> bytes: ...

    def iter_bytes(
        self, conversation_id: str, storage_key: str, *, chunk_size: int = ...
    ) -> AsyncIterator[bytes]: ...

    def local_path(self, conversation_id: str, storage_key: str) -> Path | None:
        """A filesystem path to serve directly, or ``None`` for non-local stores."""
        ...

    async def delete_object(self, conversation_id: str, storage_key: str) -> None:
        """Idempotent."""
        ...

    async def delete_conversation(self, conversation_id: str) -> None:
        """Remove everything stored for a conversation. Idempotent; raises ``OSError``."""
        ...

    async def list_conversation_ids(self) -> list[str]: ...

    async def list_objects(self, conversation_id: str) -> list[ObjectInfo]:
        """Every object under the conversation, including leftover temp files."""
        ...
