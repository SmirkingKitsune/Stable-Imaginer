import httpx

from imaginer.container import Container
from imaginer.core.ids import new_id
from imaginer.db import models as orm
from imaginer.db.repositories import conversations as conv_repo
from imaginer.db.repositories import messages as msg_repo
from imaginer.db.types import utcnow
from imaginer.domain.enums import MessageStatus, Role
from tests.helpers import image_bytes


async def _add_message(
    container: Container, cid: str, role: Role, text: str, **extra: object
) -> str:
    async with container.db.session() as s, s.begin():
        latest = await msg_repo.latest(s, cid)
        seq = await conv_repo.allocate_seq(s, cid, utcnow())
        row = orm.Message(
            id=new_id(),
            conversation_id=cid,
            parent_id=latest.id if latest else None,
            seq=seq,
            role=role,
            content_text=text,
            created_at=utcnow(),
            **extra,
        )
        await msg_repo.insert(s, row)
        return row.id


async def test_list_messages_in_order_with_after_seq(
    client: httpx.AsyncClient, container: Container
) -> None:
    cid = str((await client.post("/v1/conversations", json={})).json()["id"])
    upload = await client.post(
        f"/v1/conversations/{cid}/artifacts", files={"file": ("a.png", image_bytes(), "image/png")}
    )
    artifact_id = upload.json()["id"]
    first = await _add_message(container, cid, Role.USER, "draw a cat")
    async with container.db.session() as s, s.begin():
        await msg_repo.add_attachments(s, first, [artifact_id])
    await _add_message(
        container,
        cid,
        Role.ASSISTANT,
        "",
        tool_calls_json=[{"id": "call_1", "type": "function", "function": {"name": "x"}}],
    )
    await _add_message(container, cid, Role.TOOL, '{"ok": true}', tool_call_id="call_1")
    await _add_message(container, cid, Role.ASSISTANT, "done", status=MessageStatus.COMPLETE)

    page = (await client.get(f"/v1/conversations/{cid}/messages", params={"limit": 2})).json()
    assert [m["seq"] for m in page["items"]] == [1, 2]
    assert page["has_more"] is True
    assert page["items"][0]["attachments"] == [artifact_id]
    assert page["items"][1]["tool_calls"][0]["id"] == "call_1"

    rest = (await client.get(f"/v1/conversations/{cid}/messages", params={"after_seq": 2})).json()
    assert [(m["seq"], m["role"]) for m in rest["items"]] == [(3, "tool"), (4, "assistant")]
    assert rest["items"][0]["tool_call_id"] == "call_1"
    assert rest["has_more"] is False


async def test_messages_of_missing_conversation(client: httpx.AsyncClient) -> None:
    resp = await client.get(f"/v1/conversations/{new_id()}/messages")
    assert resp.status_code == 404
