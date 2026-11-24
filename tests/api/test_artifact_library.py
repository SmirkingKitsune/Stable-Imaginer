"""The artifact library on one instance: listing, pins, and importing into conversations."""

import hashlib
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from imaginer.container import Container
from imaginer.core.config import Settings
from imaginer.core.ids import new_id
from imaginer.main import create_app
from tests.fakes.fake_llama import FakeLlama
from tests.fakes.fake_sdcpp import FakeSdCpp
from tests.helpers import image_bytes, parse_sse

LIBRARY = "/v1/artifact-library"


async def _conversation(client: httpx.AsyncClient, title: str | None = None) -> str:
    return str((await client.post("/v1/conversations", json={"title": title})).json()["id"])


async def _upload(
    client: httpx.AsyncClient, cid: str, data: bytes, filename: str = "pic.png", **params: str
) -> dict[str, Any]:
    resp = await client.post(
        f"/v1/conversations/{cid}/artifacts",
        params=params,
        files={"file": (filename, data, "image/png")},
    )
    assert resp.status_code == 201, resp.text
    result: dict[str, Any] = resp.json()
    return result


async def _library(client: httpx.AsyncClient, **params: Any) -> dict[str, Any]:
    resp = await client.get(LIBRARY, params=params)
    assert resp.status_code == 200, resp.text
    result: dict[str, Any] = resp.json()
    return result


def _red(shade: int) -> bytes:
    return image_bytes(12, 8, color=(shade, 30, 30))


async def test_lists_images_across_conversations_newest_first(
    client: httpx.AsyncClient,
) -> None:
    a = await _conversation(client, "harbour")
    b = await _conversation(client, "forest")
    first = await _upload(client, a, _red(10), "boat.png")
    second = await _upload(client, b, _red(20))
    third = await _upload(client, a, _red(30))

    page = await _library(client)
    assert [i["id"] for i in page["items"]] == [third["id"], second["id"], first["id"]]
    assert page["next_cursor"] is None
    assert [(s["id"], s["status"]) for s in page["sources"]] == [("local", "ok")]
    oldest = page["items"][-1]
    assert oldest["ref"] == {"origin": "local", "artifact_id": first["id"], "pin_id": None}
    assert (oldest["live"], oldest["kind"], oldest["source"]) == (True, "image", "upload")
    assert (oldest["label"], oldest["conversation_id"]) == ("img-1", a)
    assert (oldest["conversation_title"], oldest["original_filename"]) == ("harbour", "boat.png")
    assert (oldest["copies"], oldest["pin"], oldest["generation"]) == (1, None, None)
    assert oldest["origin"]["link_id"] is None and oldest["origin"]["base_path"] == ""
    content = await client.get(oldest["content_url"])
    assert content.content == _red(10)


async def test_masks_only_when_asked_for(client: httpx.AsyncClient) -> None:
    cid = await _conversation(client)
    image = await _upload(client, cid, _red(10))
    mask = await _upload(
        client, cid, image_bytes(12, 8, color=(255, 255, 255)), kind="mask", mask_for="img-1"
    )
    assert [i["id"] for i in (await _library(client))["items"]] == [image["id"]]
    assert [i["id"] for i in (await _library(client, kind="mask"))["items"]] == [mask["id"]]
    both = await _library(client, kind="all")
    assert [i["kind"] for i in both["items"]] == ["mask", "image"]


async def test_filters_by_source_and_text(client: httpx.AsyncClient) -> None:
    cid = await _conversation(client, "Lighthouse study")
    under = await _upload(client, cid, _red(10), "a_b.png")
    await _upload(client, cid, _red(20), "axb.png")
    percent = await _upload(client, cid, _red(30), "100%.png")

    async def ids(**params: Any) -> list[str]:
        return [i["id"] for i in (await _library(client, **params))["items"]]

    assert await ids(q="a_b") == [under["id"]]  # _ is not a wildcard
    assert await ids(q="100%") == [percent["id"]]
    assert len(await ids(q="lighthouse")) == 3  # the conversation's title
    assert await ids(q="IMG-3") == [percent["id"]]
    assert len(await ids(source="upload")) == 3
    assert await ids(source=["generated", "edited"]) == []
    bad = await client.get(LIBRARY, params={"source": "telepathy"})
    assert bad.status_code == 422


async def test_pages_cover_everything_once(client: httpx.AsyncClient) -> None:
    cid = await _conversation(client)
    uploaded = [(await _upload(client, cid, _red(shade)))["id"] for shade in range(5)]
    seen: list[str] = []
    cursor = None
    for _ in range(10):
        page = await _library(client, limit=2, **({"cursor": cursor} if cursor else {}))
        seen += [i["id"] for i in page["items"]]
        cursor = page["next_cursor"]
        if cursor is None:
            break
    assert seen == uploaded[::-1]
    assert (await client.get(LIBRARY, params={"cursor": "nonsense!"})).status_code == 422


async def test_the_same_image_is_listed_once(client: httpx.AsyncClient) -> None:
    a, b = await _conversation(client), await _conversation(client)
    first = await _upload(client, a, _red(10))
    await _upload(client, b, _red(10))
    (item,) = (await _library(client))["items"]
    assert (item["id"], item["copies"]) == (first["id"], 2)

    # Its earliest copy goes with its conversation: the other one stands in.
    assert (await client.delete(f"/v1/conversations/{a}")).status_code == 204
    (item,) = (await _library(client))["items"]
    assert (item["conversation_id"], item["copies"]) == (b, 1)


async def test_pinning_keeps_an_image(client: httpx.AsyncClient, container: Container) -> None:
    cid = await _conversation(client, "keepers")
    art = await _upload(client, cid, _red(10), "keep.png")
    ref = {"artifact_id": art["id"]}

    made = await client.post(f"{LIBRARY}/pins", json={"ref": ref, "name": "  hero shot "})
    assert made.status_code == 201, made.text
    pin = made.json()["pin"]
    assert (pin["name"], pin["source_conversation_id"]) == ("hero shot", cid)
    again = await client.post(f"{LIBRARY}/pins", json={"ref": ref})
    assert again.status_code == 200 and again.json()["pin"]["id"] == pin["id"]
    files = list((container.settings.data_root / "artifact_library").iterdir())
    assert [f.read_bytes() for f in files] == [_red(10)]

    content = await client.get(f"{LIBRARY}/pins/{pin['id']}/content")
    assert content.content == _red(10)
    assert content.headers["etag"] == f'"{hashlib.sha256(_red(10)).hexdigest()}"'
    cached = await client.get(
        f"{LIBRARY}/pins/{pin['id']}/content", headers={"If-None-Match": content.headers["etag"]}
    )
    assert cached.status_code == 304

    patched = await client.patch(
        f"{LIBRARY}/pins/{pin['id']}", json={"notes": "for the cover", "tags": ["a", "A", "b"]}
    )
    assert patched.status_code == 200
    assert (patched.json()["pin"]["notes"], patched.json()["pin"]["tags"]) == (
        "for the cover",
        ["a", "b"],
    )
    assert [i["id"] for i in (await _library(client, q="hero"))["items"]] == [art["id"]]
    assert [i["id"] for i in (await _library(client, pinned=True))["items"]] == [art["id"]]
    assert (await client.patch(f"{LIBRARY}/pins/{pin['id']}", json={"x": 1})).status_code == 422


async def test_a_pin_outlives_its_conversation(
    client: httpx.AsyncClient, container: Container
) -> None:
    cid = await _conversation(client, "gone soon")
    art = await _upload(client, cid, _red(10), "boat.png")
    pin = (await client.post(f"{LIBRARY}/pins", json={"ref": {"artifact_id": art["id"]}})).json()
    assert (await client.delete(f"/v1/conversations/{cid}")).status_code == 204
    await container.sweeper.run_once()

    (item,) = (await _library(client))["items"]
    assert item["live"] is False
    assert item["id"] == art["id"]  # still in its place
    assert item["ref"] == {"origin": "local", "artifact_id": None, "pin_id": pin["pin"]["id"]}
    assert (item["conversation_title"], item["label"], item["original_filename"]) == (
        "gone soon",
        "img-1",
        "boat.png",
    )
    assert item["conversation_id"] is None and item["copies"] == 0
    assert (await client.get(item["content_url"])).content == _red(10)
    assert (await _library(client, pinned=True))["items"][0]["id"] == art["id"]

    # Unpinning it now lets it go for good.
    assert (await client.delete(f"{LIBRARY}/pins/{pin['pin']['id']}")).status_code == 204
    assert (await _library(client))["items"] == []
    assert list((container.settings.data_root / "artifact_library").iterdir()) == []
    assert (await client.get(item["content_url"])).status_code == 404
    assert (await client.delete(f"{LIBRARY}/pins/{pin['pin']['id']}")).status_code == 404


async def test_unpinning_a_live_image_keeps_it_listed(client: httpx.AsyncClient) -> None:
    cid = await _conversation(client)
    art = await _upload(client, cid, _red(10))
    pin = (await client.post(f"{LIBRARY}/pins", json={"ref": {"artifact_id": art["id"]}})).json()
    assert (await client.delete(f"{LIBRARY}/pins/{pin['pin']['id']}")).status_code == 204
    (item,) = (await _library(client))["items"]
    assert (item["id"], item["pin"]) == (art["id"], None)


async def test_importing_copies_an_image_into_a_conversation(client: httpx.AsyncClient) -> None:
    source = await _conversation(client, "the original")
    target = await _conversation(client)
    await _upload(client, target, _red(99))  # so the import is img-2 there
    art = await _upload(client, source, _red(10), "boat.png")
    body = {"conversation_id": target, "ref": {"artifact_id": art["id"]}}

    made = await client.post(f"{LIBRARY}/import", json=body)
    assert made.status_code == 201, made.text
    copy = made.json()
    assert (copy["conversation_id"], copy["label"], copy["source"]) == (target, "img-2", "upload")
    assert copy["id"] != art["id"] and copy["sha256"] == art["sha256"]
    assert copy["metadata"]["original_filename"] == "boat.png"
    imported = copy["metadata"]["imported_from"]
    assert (imported["conversation_id"], imported["label"], imported["artifact_id"]) == (
        source,
        "img-1",
        art["id"],
    )
    assert imported["conversation_title"] == "the original"
    assert (await client.get(copy["content_url"])).content == _red(10)

    again = await client.post(f"{LIBRARY}/import", json=body)
    assert again.status_code == 200 and again.json()["id"] == copy["id"]
    own = await client.post(
        f"{LIBRARY}/import", json={"conversation_id": source, "ref": {"artifact_id": art["id"]}}
    )
    assert own.status_code == 200 and own.json()["id"] == art["id"]
    # Listed once: the copy rides on the original.
    items = (await _library(client))["items"]
    assert [(i["id"], i["copies"]) for i in items if i["sha256"] == art["sha256"]] == [
        (art["id"], 2)
    ]


async def test_importing_a_mask_keeps_it_a_mask(client: httpx.AsyncClient) -> None:
    source, target = await _conversation(client), await _conversation(client)
    await _upload(client, source, _red(10))
    mask = await _upload(
        client, source, image_bytes(12, 8, color=(255, 255, 255)), kind="mask", mask_for="img-1"
    )
    made = await client.post(
        f"{LIBRARY}/import", json={"conversation_id": target, "ref": {"artifact_id": mask["id"]}}
    )
    assert made.status_code == 201, made.text
    assert (made.json()["kind"], made.json()["label"]) == ("mask", "mask-1")
    assert "mask_for" not in made.json()["metadata"]


async def test_importing_a_pin_whose_conversation_is_gone(client: httpx.AsyncClient) -> None:
    source, target = await _conversation(client), await _conversation(client)
    art = await _upload(client, source, _red(10))
    pin = (await client.post(f"{LIBRARY}/pins", json={"ref": {"artifact_id": art["id"]}})).json()
    await client.delete(f"/v1/conversations/{source}")
    made = await client.post(
        f"{LIBRARY}/import",
        json={"conversation_id": target, "ref": {"pin_id": pin["pin"]["id"]}},
    )
    assert made.status_code == 201, made.text
    assert made.json()["metadata"]["imported_from"]["pin_id"] == pin["pin"]["id"]
    # The image is live again: the library lists the copy, with its pin.
    (item,) = (await _library(client))["items"]
    assert (item["live"], item["conversation_id"]) == (True, target)
    assert item["pin"]["id"] == pin["pin"]["id"]


async def test_refusals(client: httpx.AsyncClient) -> None:
    cid = await _conversation(client)
    art = await _upload(client, cid, _red(10))
    gone = await _conversation(client)
    await client.delete(f"/v1/conversations/{gone}")

    async def imported(body: dict[str, Any]) -> int:
        return (await client.post(f"{LIBRARY}/import", json=body)).status_code

    ref = {"artifact_id": art["id"]}
    assert await imported({"conversation_id": new_id(), "ref": ref}) == 404
    assert await imported({"conversation_id": gone, "ref": ref}) == 404
    assert await imported({"conversation_id": cid, "ref": {"artifact_id": new_id()}}) == 404
    assert await imported({"conversation_id": cid, "ref": {"pin_id": new_id()}}) == 404
    assert await imported({"conversation_id": cid, "ref": {}}) == 422
    both = {"artifact_id": art["id"], "pin_id": new_id()}
    assert await imported({"conversation_id": cid, "ref": both}) == 422
    assert await imported({"conversation_id": cid, "link_id": new_id(), "ref": ref}) == 404
    remote_ref = {"origin": new_id(), "artifact_id": art["id"]}
    assert await imported({"conversation_id": cid, "ref": remote_ref}) == 404
    assert (await client.post(f"{LIBRARY}/pins", json={"ref": remote_ref})).status_code == 404
    assert (await client.get(f"{LIBRARY}/pins/{new_id()}")).status_code == 404
    assert (await client.get(LIBRARY, params={"origin": "elsewhere"})).status_code == 404


# --- with images made by the (fake) engines -------------------------------------------------


@pytest.fixture
async def studio(studio_settings: Settings) -> AsyncIterator[tuple[httpx.AsyncClient, FastAPI]]:
    app = create_app(studio_settings)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client, app


async def _send(client: httpx.AsyncClient, cid: str, content: str, **extra: Any) -> None:
    resp = await client.post(
        f"/v1/conversations/{cid}/messages", json={"content": content, **extra}
    )
    assert resp.status_code == 200, resp.text
    assert parse_sse(resp.text)[-1][0] == "done"


async def test_generated_images_say_how_they_were_made(
    studio: tuple[httpx.AsyncClient, FastAPI], fake_llama: FakeLlama, fake_sd: FakeSdCpp
) -> None:
    client, _ = studio
    fake_sd.step_delay_s = 0.0
    fake_llama.queue(
        FakeLlama.tool_call("generate_image", {"prompt": "a lighthouse at dusk", "seed": 7}),
        FakeLlama.text("Here is img-1."),
    )
    cid = await _conversation(client)
    await _send(client, cid, "draw a lighthouse")

    (item,) = (await _library(client))["items"]
    assert (item["source"], item["prompt"]) == ("generated", "a lighthouse at dusk")
    generation = item["generation"]
    assert (generation["task"], generation["backend"], generation["seed"]) == ("t2i", "sd", 7)
    assert generation["resolved"]["steps"] == 3
    assert generation["resolved"]["sampler"] == "euler"
    assert (await _library(client, source="generated"))["items"][0]["id"] == item["id"]
    assert [i["id"] for i in (await _library(client, q="dusk"))["items"]] == [item["id"]]


async def test_an_imported_image_can_be_edited_in_its_new_conversation(
    studio: tuple[httpx.AsyncClient, FastAPI], fake_llama: FakeLlama, fake_sd: FakeSdCpp
) -> None:
    client, _ = studio
    fake_sd.step_delay_s = 0.0
    source, target = await _conversation(client), await _conversation(client)
    art = await _upload(client, source, image_bytes(64, 64))
    copy = (
        await client.post(
            f"{LIBRARY}/import",
            json={"conversation_id": target, "ref": {"artifact_id": art["id"]}},
        )
    ).json()

    fake_llama.queue(
        FakeLlama.tool_call("edit_image", {"images": [copy["label"]], "prompt": "make it blue"}),
        FakeLlama.text("Here is img-2."),
    )
    await _send(client, target, "make @img-1 blue", attachments=[copy["id"]])
    artifacts = (await client.get(f"/v1/conversations/{target}/artifacts")).json()["items"]
    assert [(a["label"], a["source"]) for a in artifacts] == [
        ("img-1", "upload"),
        ("img-2", "edited"),
    ]
    lineage = (await client.get(f"/v1/artifacts/{artifacts[1]['id']}/lineage")).json()
    assert [p["artifact_id"] for p in lineage["parents"]] == [copy["id"]]
    # The chat model is told where the image came from.
    system = fake_llama.requests[-1]["messages"][0]["content"]
    assert "img-1: uploaded image" in system and "from the artifact library" in system
    # The source conversation is untouched.
    assert len((await client.get(f"/v1/conversations/{source}/artifacts")).json()["items"]) == 1
    assert json.dumps(copy["metadata"]["imported_from"])  # stored as plain JSON
