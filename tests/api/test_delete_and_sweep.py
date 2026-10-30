import os
import time

import httpx
import pytest
import sqlalchemy as sa

from imaginer.container import Container
from imaginer.core.ids import new_id
from imaginer.db import models as orm
from tests.helpers import image_bytes


async def _conversation_with_image(client: httpx.AsyncClient) -> tuple[str, str]:
    cid = str((await client.post("/v1/conversations", json={})).json()["id"])
    resp = await client.post(
        f"/v1/conversations/{cid}/artifacts", files={"file": ("a.png", image_bytes(), "image/png")}
    )
    return cid, str(resp.json()["id"])


async def _row(container: Container, cid: str) -> orm.Conversation | None:
    async with container.db.session() as s, s.begin():
        return (
            await s.execute(sa.select(orm.Conversation).where(orm.Conversation.id == cid))
        ).scalar_one_or_none()


async def test_delete_removes_rows_and_folder(
    client: httpx.AsyncClient, container: Container
) -> None:
    cid, aid = await _conversation_with_image(client)
    folder = container.store.conversation_dir(cid)
    assert folder.is_dir()

    assert (await client.delete(f"/v1/conversations/{cid}")).status_code == 204
    assert not folder.exists()
    assert await _row(container, cid) is None
    assert (await client.get(f"/v1/artifacts/{aid}")).status_code == 404
    assert (await client.get(f"/v1/artifacts/{aid}/content")).status_code == 404


async def test_failed_file_removal_leaves_tombstone_and_sweeper_finishes(
    client: httpx.AsyncClient, container: Container, monkeypatch: pytest.MonkeyPatch
) -> None:
    cid, aid = await _conversation_with_image(client)
    folder = container.store.conversation_dir(cid)

    async def locked(conversation_id: str) -> None:
        raise PermissionError("file is open in another process")

    with monkeypatch.context() as patch:
        patch.setattr(container.store, "delete_conversation", locked)
        # The delete is accepted: the conversation is gone for every reader right away...
        assert (await client.delete(f"/v1/conversations/{cid}")).status_code == 204
        assert (await client.get(f"/v1/conversations/{cid}")).status_code == 404
        assert (await client.get(f"/v1/artifacts/{aid}")).status_code == 404
        listed = (await client.get("/v1/conversations")).json()["items"]
        assert cid not in [c["id"] for c in listed]
        # ...but its files and tombstoned row are still there.
        assert folder.is_dir()
        row = await _row(container, cid)
        assert row is not None and row.deleted_at is not None
        report = await container.sweeper.run_once()
        assert report.deletes_pending == [cid]

    report = await container.sweeper.run_once()
    assert report.deletes_finished == [cid]
    assert not folder.exists()
    assert await _row(container, cid) is None


async def test_delete_hooks_run_after_tombstone(container: Container) -> None:
    from imaginer.domain.models import ConversationCreate

    seen: list[tuple[str, bool]] = []

    async def hook(conversation_id: str) -> None:
        row = await _row(container, conversation_id)
        seen.append((conversation_id, row is not None and row.deleted_at is not None))
        raise RuntimeError("a failing hook must not block the delete")

    container.conversations.add_delete_hook(hook)
    conv = await container.conversations.create(ConversationCreate())
    await container.conversations.delete(conv.id)
    assert seen == [(conv.id, True)]
    assert await _row(container, conv.id) is None


async def test_sweeper_removes_orphan_folders(container: Container) -> None:
    orphan = new_id()
    await container.store.put(orphan, new_id(), "png", b"x")
    report = await container.sweeper.run_once()
    assert report.orphan_dirs_removed == [orphan]
    assert not container.store.conversation_dir(orphan).exists()


async def test_sweeper_removes_old_unreferenced_files_only(
    client: httpx.AsyncClient, container: Container
) -> None:
    cid, _ = await _conversation_with_image(client)
    old_orphan = await container.store.put(cid, new_id(), "png", b"stale")
    fresh_orphan = await container.store.put(cid, new_id(), "png", b"in-flight write")
    old_path = container.store.local_path(cid, old_orphan.storage_key)
    assert old_path is not None
    two_hours_ago = time.time() - 7200
    os.utime(old_path, (two_hours_ago, two_hours_ago))

    report = await container.sweeper.run_once()
    assert report.orphan_files_removed == [f"{cid}/{old_orphan.storage_key}"]
    keys = {o.storage_key for o in await container.store.list_objects(cid)}
    assert old_orphan.storage_key not in keys
    assert fresh_orphan.storage_key in keys  # inside the grace period
    assert len(keys) == 2  # the referenced upload is untouched
