import base64
import binascii
import builtins
import json
from collections.abc import Awaitable, Callable
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from imaginer.core.ids import new_id
from imaginer.core.logging import get_logger
from imaginer.db import models as orm
from imaginer.db.engine import Database
from imaginer.db.repositories import artifacts as art_repo
from imaginer.db.repositories import conversations as conv_repo
from imaginer.db.repositories import messages as msg_repo
from imaginer.db.repositories import search as search_repo
from imaginer.db.types import utcnow
from imaginer.domain import models as dto
from imaginer.domain.enums import ArtifactKind
from imaginer.domain.errors import InvalidRequestError, NotFoundError
from imaginer.services import mappers
from imaginer.services.branches import describe as describe_branches
from imaginer.storage.base import ArtifactStore

log = get_logger(__name__)

DeleteHook = Callable[[str], Awaitable[None]]


class ConversationService:
    def __init__(self, db: Database, store: ArtifactStore) -> None:
        self._db = db
        self._store = store
        self._delete_hooks: list[DeleteHook] = []

    def add_delete_hook(self, hook: DeleteHook) -> None:
        """Run after a conversation is tombstoned, before its files go (e.g. cancel its jobs)."""
        self._delete_hooks.append(hook)

    async def create(self, data: dto.ConversationCreate) -> dto.Conversation:
        now = utcnow()
        row = orm.Conversation(
            id=new_id(),
            title=data.title,
            metadata_json=data.metadata,
            created_at=now,
            updated_at=now,
        )
        async with self._db.session() as session, session.begin():
            await conv_repo.insert(session, row)
        log.info("conversation.created", conversation_id=row.id)
        return mappers.conversation(row)

    async def get(self, conversation_id: str) -> dto.Conversation:
        async with self._db.session() as session, session.begin():
            row = await conv_repo.require_active(session, conversation_id)
            return mappers.conversation(row)

    async def list(
        self, *, limit: int, cursor: str | None = None, include_archived: bool = False
    ) -> dto.Page[dto.ConversationSummary]:
        before = decode_cursor(cursor) if cursor else None
        async with self._db.session() as session, session.begin():
            rows = await conv_repo.list_active(
                session, limit=limit + 1, before=before, include_archived=include_archived
            )
            has_more = len(rows) > limit
            rows = rows[:limit]
            images = await art_repo.image_summaries(session, [r.id for r in rows])
        next_cursor = encode_cursor(rows[-1].updated_at, rows[-1].id) if has_more else None
        return dto.Page[dto.ConversationSummary](
            items=[mappers.conversation_summary(r, images.get(r.id)) for r in rows],
            next_cursor=next_cursor,
        )

    async def search(
        self,
        query: str,
        *,
        fields: builtins.list[dto.SearchField],
        limit: int,
        cursor: str | None = None,
        include_archived: bool = False,
    ) -> dto.SearchPage:
        """Search the next ``limit`` conversations (gallery order) for ``query``."""
        before = decode_cursor(cursor) if cursor else None
        async with self._db.session() as session, session.begin():
            total = await search_repo.count(session, include_archived=include_archived)
            rows = await conv_repo.list_active(
                session, limit=limit + 1, before=before, include_archived=include_archived
            )
            has_more = len(rows) > limit
            rows = rows[:limit]
            found = await search_repo.matches(session, rows, query, fields)
            hits = [r for r in rows if r.id in found]
            images = await art_repo.image_summaries(session, [r.id for r in hits])
        next_cursor = encode_cursor(rows[-1].updated_at, rows[-1].id) if has_more else None
        items = [
            dto.SearchHit(
                **mappers.conversation_summary(r, images.get(r.id)).model_dump(),
                match=dto.SearchMatch(
                    field=found[r.id].field, snippet=found[r.id].snippet, label=found[r.id].label
                ),
            )
            for r in hits
        ]
        return dto.SearchPage(items=items, scanned=len(rows), total=total, next_cursor=next_cursor)

    async def update(self, conversation_id: str, data: dto.ConversationUpdate) -> dto.Conversation:
        fields = data.model_fields_set
        now = utcnow()
        async with self._db.session() as session, session.begin():
            row = await conv_repo.require_active(session, conversation_id)
            if "title" in fields:
                row.title = data.title
            if "metadata" in fields:
                row.metadata_json = data.metadata
            if "archived" in fields and data.archived is not None:
                if data.archived and row.archived_at is None:
                    row.archived_at = now
                elif not data.archived:
                    row.archived_at = None
            row.updated_at = now
            await session.flush()
            return mappers.conversation(row)

    async def delete(self, conversation_id: str) -> None:
        """Delete a conversation, its rows and its files.

        Order matters for consistency: (1) tombstone the row so the conversation is gone
        for every reader, (2) remove its files, (3) remove its rows. If (2) or (3) fails,
        the tombstone stays and the sweeper finishes the job later.
        """
        async with self._db.session() as session, session.begin():
            if not await conv_repo.tombstone(session, conversation_id, utcnow()):
                raise NotFoundError(f"conversation {conversation_id} not found")
        log.info("conversation.tombstoned", conversation_id=conversation_id)
        for hook in self._delete_hooks:
            try:
                await hook(conversation_id)
            except Exception:
                log.exception("conversation.delete_hook_failed", conversation_id=conversation_id)
        await self.finish_delete(conversation_id)

    async def finish_delete(self, conversation_id: str) -> bool:
        """Steps (2) and (3) of a delete. Safe to repeat. False if it must be retried."""
        try:
            await self._store.delete_conversation(conversation_id)
        except OSError as exc:
            log.warning(
                "conversation.files_delete_deferred",
                conversation_id=conversation_id,
                error=str(exc),
            )
            return False
        async with self._db.session() as session, session.begin():
            await conv_repo.hard_delete(session, conversation_id)
        log.info("conversation.deleted", conversation_id=conversation_id)
        return True

    async def list_messages(
        self,
        conversation_id: str,
        *,
        after_seq: int = 0,
        limit: int,
        leaf_id: str | None = None,
        all_branches: bool = False,
    ) -> dto.MessagePage:
        """One branch (by default the latest), or with ``all_branches`` every message."""
        async with self._db.session() as session, session.begin():
            await conv_repo.require_active(session, conversation_id)
            if all_branches:
                rows = await msg_repo.list_for_conversation(
                    session, conversation_id, after_seq=after_seq, limit=limit + 1
                )
            else:
                leaf = await self._leaf(session, conversation_id, leaf_id)
                path = await msg_repo.path_to(session, leaf.id) if leaf else []
                rows = [m for m in path if m.seq > after_seq][: limit + 1]
            has_more = len(rows) > limit
            rows = rows[:limit]
            attachments = await msg_repo.attachments_for(session, [r.id for r in rows])
        return dto.MessagePage(
            items=[mappers.message(r, attachments.get(r.id)) for r in rows], has_more=has_more
        )

    async def branches(self, conversation_id: str) -> builtins.list[dto.Branch]:
        async with self._db.session() as session, session.begin():
            await conv_repo.require_active(session, conversation_id)
            rows = await msg_repo.all_for_conversation(session, conversation_id)
            attachments = await msg_repo.attachments_for(session, [r.id for r in rows])
            referenced = sorted({a for ids in attachments.values() for a in ids})
            artifacts = await art_repo.get_many_in_conversation(
                session, conversation_id, referenced
            )
        images = {a.id: a for a in artifacts if a.kind is ArtifactKind.IMAGE}
        return describe_branches(rows, attachments, images)

    @staticmethod
    async def _leaf(
        session: AsyncSession, conversation_id: str, leaf_id: str | None
    ) -> orm.Message | None:
        if leaf_id is None:
            return await msg_repo.latest(session, conversation_id)
        leaf = await msg_repo.get_in_conversation(session, conversation_id, leaf_id)
        if leaf is None:
            raise NotFoundError(f"message {leaf_id} not found in conversation {conversation_id}")
        return leaf


def encode_cursor(updated_at: datetime, conversation_id: str) -> str:
    raw = json.dumps([updated_at.isoformat(), conversation_id]).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def decode_cursor(cursor: str) -> tuple[datetime, str]:
    try:
        raw = base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4))
        stamp, conversation_id = json.loads(raw)
        return datetime.fromisoformat(stamp), str(conversation_id)
    except (binascii.Error, ValueError, TypeError) as exc:
        raise InvalidRequestError("invalid cursor") from exc
