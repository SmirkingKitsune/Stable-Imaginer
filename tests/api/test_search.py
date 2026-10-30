"""GET /v1/conversations/search: titles, prompts, image prompts, labels and replies."""

from typing import Any

import httpx

from imaginer.container import Container
from imaginer.core.ids import new_id
from imaginer.db import models as orm
from imaginer.db.repositories import conversations as conv_repo
from imaginer.db.repositories import messages as msg_repo
from imaginer.db.types import utcnow
from imaginer.domain.enums import ArtifactSource, Role
from imaginer.services.artifacts import NewArtifact
from imaginer.services.media import inspect_image_bytes
from tests.helpers import image_bytes


async def _conversation(client: httpx.AsyncClient, title: str | None = None) -> str:
    return str((await client.post("/v1/conversations", json={"title": title})).json()["id"])


async def _message(container: Container, cid: str, role: Role, text: str) -> None:
    async with container.db.session() as s, s.begin():
        now = utcnow()
        seq = await conv_repo.allocate_seq(s, cid, now)
        await msg_repo.insert(
            s,
            orm.Message(
                id=new_id(),
                conversation_id=cid,
                seq=seq,
                role=role,
                content_text=text,
                created_at=now,
            ),
        )


async def _image(container: Container, cid: str, prompt: str) -> str:
    data = image_bytes()
    spec = NewArtifact.image(
        cid, inspect_image_bytes(data), ArtifactSource.GENERATED, metadata={"prompt": prompt}
    )
    return (await container.artifacts.ingest(spec, data)).label


async def _search(client: httpx.AsyncClient, **params: Any) -> dict[str, Any]:
    resp = await client.get("/v1/conversations/search", params=params)
    assert resp.status_code == 200, resp.text
    body: dict[str, Any] = resp.json()
    return body


async def test_finds_each_kind_of_match(client: httpx.AsyncClient, container: Container) -> None:
    titled = await _conversation(client, "Lighthouse at dusk")
    prompted = await _conversation(client, "Coastal map icons")
    await _message(
        container,
        prompted,
        Role.USER,
        "Draw a set of icons with a tiny lighthouse glyph, 1px stroke",
    )
    imaged = await _conversation(client, "Poster series, Atlantic")
    await _image(container, imaged, "a lighthouse beam cutting fog over the bay")
    replied = await _conversation(client, "Storm studies")
    await _message(
        container, replied, Role.ASSISTANT, "I kept the LIGHTHOUSE silhouette as it was."
    )
    await _conversation(client, "Nothing here")

    page = await _search(client, q="lighthouse")
    hits = {h["id"]: h["match"] for h in page["items"]}
    assert (page["scanned"], page["total"], page["next_cursor"]) == (5, 5, None)
    assert hits[titled] == {"field": "title", "snippet": None, "label": None}
    assert hits[prompted]["field"] == "prompt"
    assert (
        hits[prompted]["snippet"] == "Draw a set of icons with a tiny lighthouse glyph, 1px stroke"
    )
    assert hits[imaged] == {
        "field": "prompt",
        "snippet": "a lighthouse beam cutting fog over the bay",
        "label": "img-1",
    }
    assert hits[replied]["field"] == "reply"
    assert "LIGHTHOUSE" in hits[replied]["snippet"]
    assert len(hits) == 4

    only_titles = await _search(client, q="lighthouse", fields="title")
    assert [h["id"] for h in only_titles["items"]] == [titled]


async def test_labels_and_wildcards(client: httpx.AsyncClient, container: Container) -> None:
    cid = await _conversation(client, "fox")
    for _ in range(2):
        await _image(container, cid, "a fox")
    other = await _conversation(client, "100% done_ok")
    assert [h["match"] for h in (await _search(client, q="IMG-2"))["items"]] == [
        {"field": "label", "snippet": None, "label": "img-2"}
    ]
    # LIKE wildcards are matched literally.
    assert [h["id"] for h in (await _search(client, q="0% d"))["items"]] == [other]
    assert (await _search(client, q="_"))["items"][0]["id"] == other


async def test_pages_and_archived(client: httpx.AsyncClient) -> None:
    ids = [await _conversation(client, f"sketch {i}") for i in range(5)]
    await client.patch(f"/v1/conversations/{ids[0]}", json={"archived": True})

    first = await _search(client, q="sketch", limit=2)
    assert (first["scanned"], first["total"], len(first["items"])) == (2, 4, 2)
    second = await _search(client, q="sketch", limit=2, cursor=first["next_cursor"])
    assert second["next_cursor"] is None
    found = [h["id"] for p in (first, second) for h in p["items"]]
    assert found == ids[:0:-1]  # newest first, archived left out

    everything = await _search(client, q="sketch", include_archived=True)
    assert everything["total"] == 5 and ids[0] in [h["id"] for h in everything["items"]]


async def test_bad_fields(client: httpx.AsyncClient) -> None:
    resp = await client.get("/v1/conversations/search", params={"q": "x", "fields": "title,body"})
    assert resp.status_code == 422
    assert "body" in resp.text
