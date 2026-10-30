import base64
import io
from typing import Any

import pytest
from PIL import Image

from imaginer.core.clock import utcnow
from imaginer.core.config import ContextConfig
from imaginer.core.ids import new_id
from imaginer.db import models as orm
from imaginer.db.engine import Database
from imaginer.db.repositories import conversations as conv_repo
from imaginer.db.repositories import messages as msg_repo
from imaginer.domain.enums import ArtifactSource, MessageStatus, Role
from imaginer.orchestration.context_builder import DEFAULT_SYSTEM_PROMPT, ContextBuilder
from imaginer.services.artifacts import ArtifactService, NewArtifact
from imaginer.services.media import inspect_image_bytes
from imaginer.storage.local_fs import LocalFSArtifactStore
from tests.helpers import image_bytes


class World:
    def __init__(self, db: Database, store: LocalFSArtifactStore) -> None:
        self.db = db
        self.store = store
        self.artifacts = ArtifactService(db, store, max_upload_bytes=10**8)

    async def conversation(self) -> str:
        cid, now = new_id(), utcnow()
        async with self.db.session() as s, s.begin():
            await conv_repo.insert(s, orm.Conversation(id=cid, created_at=now, updated_at=now))
        return cid

    async def image(self, cid: str, data: bytes, filename: str | None = None) -> str:
        info = inspect_image_bytes(data)
        meta = {"original_filename": filename} if filename else None
        spec = NewArtifact.image(cid, info, ArtifactSource.UPLOAD, metadata=meta)
        return (await self.artifacts.ingest(spec, data)).id

    async def message(
        self,
        cid: str,
        role: Role,
        text: str | None,
        attachments: list[str] | None = None,
        *,
        after: str | None = "latest",
        **extra: Any,
    ) -> str:
        """Store a message after ``after`` (a message id, None for a first message)."""
        async with self.db.session() as s, s.begin():
            if after == "latest":
                latest = await msg_repo.latest(s, cid)
                after = latest.id if latest else None
            seq = await conv_repo.allocate_seq(s, cid, utcnow())
            row = orm.Message(
                id=new_id(),
                conversation_id=cid,
                parent_id=after,
                seq=seq,
                role=role,
                content_text=text,
                created_at=utcnow(),
                **extra,
            )
            await msg_repo.insert(s, row)
            await msg_repo.add_attachments(s, row.id, attachments or [])
        return row.id

    def builder(self, **config: Any) -> ContextBuilder:
        return ContextBuilder(self.db, self.store, ContextConfig(**config))


@pytest.fixture
def world(database: Database, tmp_path: Any) -> World:
    return World(database, LocalFSArtifactStore(tmp_path / "store"))


def _decode(url: str) -> Image.Image:
    header, _, payload = url.partition(",")
    assert header.startswith("data:image/") and header.endswith(";base64")
    return Image.open(io.BytesIO(base64.b64decode(payload)))


async def test_plain_history(world: World) -> None:
    cid = await world.conversation()
    await world.message(cid, Role.USER, "hello")
    await world.message(cid, Role.ASSISTANT, "hi!")
    messages = await world.builder().build(cid, vision=True)
    assert messages == [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi!"},
    ]


async def test_manifest_lists_artifacts_by_label(world: World) -> None:
    cid = await world.conversation()
    await world.image(cid, image_bytes(30, 20), "holiday.png")
    await world.image(cid, image_bytes(mode="RGBA", color=(0, 0, 0, 0)))
    system = (await world.builder(system_prompt="Be brief.").build(cid, vision=False))[0]
    assert system["content"] == (
        "Be brief.\n\nArtifacts in this conversation (oldest first):\n"
        '- img-1: uploaded image, 30x20, file "holiday.png"\n'
        "- img-2: uploaded image, 8x8, transparent background"
    )


async def test_attachments_with_vision_are_downscaled_pixels(world: World) -> None:
    cid = await world.conversation()
    big = await world.image(cid, image_bytes(400, 200))
    clear = await world.image(cid, image_bytes(mode="RGBA", color=(1, 2, 3, 0)))
    await world.message(cid, Role.USER, "what is this?", [big, clear])
    user = (await world.builder(image_max_side=100).build(cid, vision=True))[1]
    parts = user["content"]
    assert parts[0] == {"type": "text", "text": "what is this?\n\n[Attached: img-1, img-2]"}
    assert parts[1] == {"type": "text", "text": "img-1:"}
    first = _decode(parts[2]["image_url"]["url"])
    assert (first.format, first.size) == ("JPEG", (100, 50))
    assert parts[3] == {"type": "text", "text": "img-2:"}
    second = _decode(parts[4]["image_url"]["url"])
    assert (second.format, second.mode) == ("PNG", "RGBA")  # transparency survives


async def test_without_vision_only_labels(world: World) -> None:
    cid = await world.conversation()
    art = await world.image(cid, image_bytes())
    await world.message(cid, Role.USER, "look", [art])
    user = (await world.builder().build(cid, vision=False))[1]
    assert user == {"role": "user", "content": "look\n\n[Attached: img-1]"}


async def test_only_newest_images_are_sent_as_pixels(world: World) -> None:
    cid = await world.conversation()
    old, new = await world.image(cid, image_bytes()), await world.image(cid, image_bytes())
    await world.message(cid, Role.USER, "first", [old])
    await world.message(cid, Role.ASSISTANT, "ok")
    await world.message(cid, Role.USER, "second", [new])
    messages = await world.builder(max_images=1).build(cid, vision=True)
    assert messages[1]["content"] == "first\n\n[Attached: img-1]"
    assert any(p["type"] == "image_url" for p in messages[3]["content"])


async def test_tool_exchange_and_message_window(world: World) -> None:
    cid = await world.conversation()
    call = {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
    await world.message(cid, Role.USER, "make one")
    await world.message(cid, Role.ASSISTANT, None, tool_calls_json=[call])
    await world.message(cid, Role.TOOL, '{"ok": true}', tool_call_id="c1")
    await world.message(cid, Role.ASSISTANT, "done")
    full = await world.builder().build(cid, vision=False)
    assert full[2] == {"role": "assistant", "content": "", "tool_calls": [call]}
    assert full[3] == {"role": "tool", "tool_call_id": "c1", "content": '{"ok": true}'}

    # A window that starts at the tool result must not leave it orphaned.
    cut = await world.builder(max_messages=2).build(cid, vision=False)
    assert [m["role"] for m in cut] == ["system", "assistant"]


async def test_in_progress_messages_are_skipped(world: World) -> None:
    cid = await world.conversation()
    await world.message(cid, Role.USER, "hi")
    await world.message(cid, Role.ASSISTANT, None, status=MessageStatus.STREAMING)
    messages = await world.builder().build(cid, vision=False)
    assert [m["role"] for m in messages] == ["system", "user"]


async def test_follows_one_branch_and_its_images(world: World) -> None:
    cid = await world.conversation()
    first = await world.image(cid, image_bytes(), "first.png")
    await world.image(cid, image_bytes(), "not sent yet.png")  # no message attaches it
    other = await world.image(cid, image_bytes())
    user = await world.message(cid, Role.USER, "a lighthouse", [first])
    main_leaf = await world.message(cid, Role.ASSISTANT, "here it is", [other])
    await world.message(cid, Role.USER, "a lighthouse at night", after=None)  # edit & branch
    builder = world.builder()

    # By default the latest branch: the edit, alone.
    latest = await builder.build(cid, vision=False)
    assert [m["content"] for m in latest[1:]] == ["a lighthouse at night"]
    assert latest[0]["content"].endswith('- img-2: uploaded image, 8x8, file "not sent yet.png"')

    # The original branch, with its own images (and the upload nobody attached yet).
    main = await builder.build(cid, leaf_id=main_leaf, vision=False)
    assert [m["role"] for m in main] == ["system", "user", "assistant"]
    manifest = main[0]["content"].split("(oldest first):\n")[1].splitlines()
    assert [line.split(":")[0] for line in manifest] == ["- img-1", "- img-2", "- img-3"]

    # Any message can end the path.
    assert (await builder.build(cid, leaf_id=user, vision=False))[1:] == [
        {"role": "user", "content": "a lighthouse\n\n[Attached: img-1]"}
    ]
