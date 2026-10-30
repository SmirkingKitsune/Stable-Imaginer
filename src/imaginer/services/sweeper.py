"""Background consistency repair.

* Finishes conversation deletes that were tombstoned but not completed (crash, locked file).
* Removes conversation folders that have no database row.
* Removes artifact files no row points to (e.g. a crash between write and insert), once
  they are older than a grace period, so an in-flight write is never touched.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import timedelta

from imaginer.core.logging import get_logger
from imaginer.db.engine import Database
from imaginer.db.repositories import artifacts as art_repo
from imaginer.db.repositories import conversations as conv_repo
from imaginer.db.types import utcnow
from imaginer.services.conversations import ConversationService
from imaginer.storage.base import ArtifactStore

log = get_logger(__name__)


@dataclass
class SweepReport:
    deletes_finished: list[str] = field(default_factory=list)
    deletes_pending: list[str] = field(default_factory=list)
    orphan_dirs_removed: list[str] = field(default_factory=list)
    orphan_files_removed: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.deletes_finished or self.orphan_dirs_removed or self.orphan_files_removed)


class Sweeper:
    def __init__(
        self,
        db: Database,
        store: ArtifactStore,
        conversations: ConversationService,
        *,
        orphan_grace_s: float,
    ) -> None:
        self._db = db
        self._store = store
        self._conversations = conversations
        self._grace = timedelta(seconds=orphan_grace_s)
        self._lock = asyncio.Lock()

    async def run_once(self) -> SweepReport:
        async with self._lock:
            report = SweepReport()
            await self._finish_deletes(report)
            await self._remove_orphans(report)
            if report.changed or report.deletes_pending:
                log.info(
                    "sweep.done",
                    deletes_finished=len(report.deletes_finished),
                    deletes_pending=len(report.deletes_pending),
                    orphan_dirs=len(report.orphan_dirs_removed),
                    orphan_files=len(report.orphan_files_removed),
                )
            return report

    async def run_forever(self, interval_s: float) -> None:
        while True:
            await asyncio.sleep(interval_s)
            try:
                await self.run_once()
            except Exception:
                log.exception("sweep.failed")

    async def _finish_deletes(self, report: SweepReport) -> None:
        async with self._db.session() as session, session.begin():
            tombstoned = await conv_repo.list_tombstoned_ids(session)
        for conversation_id in tombstoned:
            if await self._conversations.finish_delete(conversation_id):
                report.deletes_finished.append(conversation_id)
            else:
                report.deletes_pending.append(conversation_id)

    async def _remove_orphans(self, report: SweepReport) -> None:
        on_disk = await self._store.list_conversation_ids()
        async with self._db.session() as session, session.begin():
            known = await conv_repo.existing_ids(session, on_disk)
        for conversation_id in on_disk:
            if conversation_id not in known:
                try:
                    await self._store.delete_conversation(conversation_id)
                    report.orphan_dirs_removed.append(conversation_id)
                except OSError as exc:
                    log.warning(
                        "sweep.orphan_dir_failed", conversation_id=conversation_id, error=str(exc)
                    )
                continue
            await self._remove_orphan_files(conversation_id, report)

    async def _remove_orphan_files(self, conversation_id: str, report: SweepReport) -> None:
        objects = await self._store.list_objects(conversation_id)
        if not objects:
            return
        async with self._db.session() as session, session.begin():
            referenced = await art_repo.storage_keys(session, conversation_id)
        cutoff = utcnow() - self._grace
        for obj in objects:
            if obj.storage_key in referenced or obj.modified_at > cutoff:
                continue
            try:
                await self._store.delete_object(conversation_id, obj.storage_key)
                report.orphan_files_removed.append(f"{conversation_id}/{obj.storage_key}")
            except (OSError, ValueError) as exc:
                log.warning(
                    "sweep.orphan_file_failed",
                    conversation_id=conversation_id,
                    storage_key=obj.storage_key,
                    error=str(exc),
                )
