"""The artifact library across linked instances, over real sockets: a desk and a NAS."""

from typing import Any

import httpx
import pytest

from imaginer.backends.openai_http.transport import HttpTransport
from imaginer.core.config import RetryConfig, TimeoutConfig
from tests.helpers import image_bytes
from tests.links.conftest import Instance, InstanceFactory, pair

Pair = tuple[Instance, Instance]
LIBRARY = "/v1/artifact-library"


@pytest.fixture
def desk_and_nas(make_instance: InstanceFactory) -> Pair:
    return make_instance("desk", backends={}), make_instance("nas", backends={})


def _red(shade: int) -> bytes:
    return image_bytes(12, 8, color=(shade, 30, 30))


async def _upload(owner: Instance, data: bytes, cid: str | None = None) -> dict[str, Any]:
    async with owner.client() as client:
        if cid is None:
            cid = (await client.post("/v1/conversations", json={"title": owner.name})).json()["id"]
        resp = await client.post(
            f"/v1/conversations/{cid}/artifacts", files={"file": ("pic.png", data, "image/png")}
        )
    assert resp.status_code == 201, resp.text
    result: dict[str, Any] = resp.json()
    return result


async def _library(client: httpx.AsyncClient, **params: Any) -> dict[str, Any]:
    resp = await client.get(LIBRARY, params=params)
    assert resp.status_code == 200, resp.text
    result: dict[str, Any] = resp.json()
    return result


async def test_the_library_shows_both_instances(desk_and_nas: Pair) -> None:
    desk, nas = desk_and_nas
    link_id = await pair(desk, nas, name="nas")
    mine = await _upload(desk, _red(10))
    theirs = await _upload(nas, _red(20))
    async with desk.client() as client:
        page = await _library(client)
        assert {s["name"]: s["status"] for s in page["sources"]} == {"desk": "ok", "nas": "ok"}
        assert [i["id"] for i in page["items"]] == [theirs["id"], mine["id"]]
        remote = page["items"][0]
        assert remote["origin"]["link_id"] == link_id
        assert remote["origin"]["base_path"] == f"/v1/links/{link_id}/r"
        assert remote["ref"] == {"origin": link_id, "artifact_id": theirs["id"], "pin_id": None}
        assert remote["conversation_title"] == "nas"
        content = await client.get(remote["origin"]["base_path"] + remote["content_url"])
        assert content.content == _red(20)
        only_nas = await _library(client, origin=link_id)
        assert [i["id"] for i in only_nas["items"]] == [theirs["id"]]
        assert [i["id"] for i in (await _library(client, origin="local"))["items"]] == [mine["id"]]


async def test_pages_merge_across_instances(desk_and_nas: Pair) -> None:
    desk, nas = desk_and_nas
    await pair(desk, nas, name="nas")
    made = [(await _upload(desk if n % 2 == 0 else nas, _red(n)))["id"] for n in range(5)]
    seen: list[str] = []
    cursor = None
    async with desk.client() as client:
        for _ in range(10):
            page = await _library(client, limit=2, **({"cursor": cursor} if cursor else {}))
            seen += [i["id"] for i in page["items"]]
            cursor = page["next_cursor"]
            if cursor is None:
                break
    assert seen == made[::-1]


async def test_pinning_a_remote_image_keeps_a_copy_here(desk_and_nas: Pair) -> None:
    desk, nas = desk_and_nas
    link_id = await pair(desk, nas, name="nas")
    theirs = await _upload(nas, _red(20))
    async with desk.client() as client:
        made = await client.post(
            f"{LIBRARY}/pins", json={"ref": {"origin": link_id, "artifact_id": theirs["id"]}}
        )
        assert made.status_code == 201, made.text
        pin = made.json()
        assert (pin["pin"]["source_instance_name"], pin["pin"]["source_link_id"]) == (
            "nas",
            link_id,
        )
        assert pin["origin"]["link_id"] is None  # the copy is here
        # Listed once: the pinned copy, not the NAS's own.
        (item,) = (await _library(client))["items"]
        assert (item["id"], item["live"], item["pin"]["id"]) == (
            theirs["id"],
            False,
            pin["pin"]["id"],
        )

        nas.server.stop()
        page = await _library(client)
        assert {s["name"]: s["status"] for s in page["sources"]}["nas"] == "unreachable"
        assert [i["id"] for i in page["items"]] == [theirs["id"]]
        assert (await client.get(page["items"][0]["content_url"])).content == _red(20)


async def test_importing_a_remote_image_into_a_conversation_here(desk_and_nas: Pair) -> None:
    desk, nas = desk_and_nas
    link_id = await pair(desk, nas, name="nas")
    theirs = await _upload(nas, _red(20))
    async with desk.client() as client:
        cid = (await client.post("/v1/conversations", json={})).json()["id"]
        body = {"conversation_id": cid, "ref": {"origin": link_id, "artifact_id": theirs["id"]}}
        made = await client.post(f"{LIBRARY}/import", json=body)
        assert made.status_code == 201, made.text
        copy = made.json()
        assert (copy["conversation_id"], copy["label"]) == (cid, "img-1")
        assert copy["metadata"]["imported_from"]["instance_name"] == "nas"
        assert (await client.get(copy["content_url"])).content == _red(20)
        again = await client.post(f"{LIBRARY}/import", json=body)
        assert again.status_code == 200 and again.json()["id"] == copy["id"]
        # The desk has it now: the NAS's copy isn't listed as well.
        assert [i["id"] for i in (await _library(client))["items"]] == [copy["id"]]


async def test_importing_into_a_conversation_on_the_linked_instance(desk_and_nas: Pair) -> None:
    desk, nas = desk_and_nas
    link_id = await pair(desk, nas, name="nas")
    mine = await _upload(desk, _red(10))
    async with nas.client() as client:
        remote_cid = (await client.post("/v1/conversations", json={})).json()["id"]
    async with desk.client() as client:
        body = {
            "conversation_id": remote_cid,
            "link_id": link_id,
            "ref": {"artifact_id": mine["id"]},
        }
        made = await client.post(f"{LIBRARY}/import", json=body)
        assert made.status_code == 201, made.text
        assert (made.json()["conversation_id"], made.json()["label"]) == (remote_cid, "img-1")
        again = await client.post(f"{LIBRARY}/import", json=body)
        assert again.status_code == 200 and again.json()["id"] == made.json()["id"]
    async with nas.client() as client:
        (stored,) = (await client.get(f"/v1/conversations/{remote_cid}/artifacts")).json()["items"]
    assert stored["sha256"] == mine["sha256"]
    assert stored["metadata"]["original_filename"] == "pic.png"


async def test_a_read_only_instance_takes_no_images(make_instance: InstanceFactory) -> None:
    owner = make_instance("owner", backends={}, gallery="read")
    viewer = make_instance("viewer", backends={})
    link_id = await pair(viewer, owner)
    mine = await _upload(viewer, _red(10))
    async with owner.client() as client:
        cid = (await client.post("/v1/conversations", json={})).json()["id"]
    theirs = await _upload(owner, _red(20), cid)
    async with viewer.client() as client:
        body = {"conversation_id": cid, "link_id": link_id, "ref": {"artifact_id": mine["id"]}}
        refused = await client.post(f"{LIBRARY}/import", json=body)
        assert refused.status_code == 403, refused.text
        # Its images can still be listed and brought here.
        page = await _library(client)
        assert theirs["id"] in [i["id"] for i in page["items"]]


async def test_a_link_token_reaches_only_the_peer_routes(desk_and_nas: Pair) -> None:
    desk, nas = desk_and_nas
    link_id = await pair(desk, nas, name="nas")
    theirs = await _upload(nas, _red(20))
    token = desk.container.links._outbound[link_id]  # what the desk calls the NAS with
    async with nas.client(token) as client:
        assert (await client.get(LIBRARY)).status_code == 403
        assert (
            await client.post(f"{LIBRARY}/pins", json={"ref": {"artifact_id": "x"}})
        ).status_code == 403
        assert (await client.post(f"{LIBRARY}/import", json={})).status_code == 403
        local = await client.get(f"{LIBRARY}/local")
        assert local.status_code == 200
        assert [i["id"] for i in local.json()["items"]] == [theirs["id"]]
    async with desk.client() as client:
        base = f"/v1/links/{link_id}/r"
        assert (await client.get(f"{base}{LIBRARY}/local")).status_code == 403  # not proxied


async def test_pins_are_shared_only_for_their_own_images(desk_and_nas: Pair) -> None:
    desk, nas = desk_and_nas
    desk_link = await pair(desk, nas, name="nas")
    nas_link = nas.container.links.by_peer(desk.container.links.instance_id)
    assert nas_link is not None
    mine = await _upload(desk, _red(10))
    own = await _upload(nas, _red(20))
    async with nas.client() as client:
        # The NAS keeps the desk's image, and one of its own.
        for ref in ({"origin": nas_link.id, "artifact_id": mine["id"]}, {"artifact_id": own["id"]}):
            assert (await client.post(f"{LIBRARY}/pins", json={"ref": ref})).status_code == 201
        await client.delete(f"/v1/conversations/{own['conversation_id']}")
    async with desk.client() as client:
        await client.delete(f"/v1/conversations/{mine['conversation_id']}")
        page = await _library(client)
        # The desk's image is gone here; the NAS doesn't hand back its copy of it. The NAS's
        # own pinned image is listed, and its bytes come through the proxy.
        (item,) = page["items"]
        assert (item["id"], item["live"], item["origin"]["link_id"]) == (
            own["id"],
            False,
            desk_link,
        )
        content = await client.get(item["origin"]["base_path"] + item["content_url"])
        assert content.content == _red(20)
        cid = (await client.post("/v1/conversations", json={})).json()["id"]
        made = await client.post(
            f"{LIBRARY}/import", json={"conversation_id": cid, "ref": item["ref"]}
        )
        assert made.status_code == 201, made.text
    async with nas.client() as client:
        assert len((await _library(client))["items"]) == 2  # both of its pins


class _OldPeer(httpx.AsyncBaseTransport):
    """An instance from before the artifact library: everything else answers as usual."""

    def __init__(self) -> None:
        self._real = httpx.AsyncHTTPTransport()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith(LIBRARY):
            return httpx.Response(404, json={"detail": "Not Found"})
        return await self._real.handle_async_request(request)

    async def aclose(self) -> None:
        await self._real.aclose()


async def test_an_older_instance_is_reported_not_failed(desk_and_nas: Pair) -> None:
    desk, nas = desk_and_nas
    link_id = await pair(desk, nas, name="nas")
    mine = await _upload(desk, _red(10))
    theirs = await _upload(nas, _red(20))
    link = desk.container.links.get(link_id)
    token = desk.container.links._outbound[link_id]
    old = HttpTransport(
        name="link nas",
        base_url=nas.url,
        timeouts=TimeoutConfig(),
        retry=RetryConfig(attempts=1),
        client=httpx.AsyncClient(
            transport=_OldPeer(), base_url=nas.url, headers={"Authorization": f"Bearer {token}"}
        ),
    )
    real, link.transport = link.transport, old
    try:
        async with desk.client() as client:
            page = await _library(client)
            nas_source = next(s for s in page["sources"] if s["name"] == "nas")
            assert nas_source["status"] == "unsupported"
            assert "update it" in nas_source["detail"]
            assert [i["id"] for i in page["items"]] == [mine["id"]]
            # An image it shows in a conversation can still be brought over.
            cid = (await client.post("/v1/conversations", json={})).json()["id"]
            ref = {"origin": link_id, "artifact_id": theirs["id"]}
            made = await client.post(f"{LIBRARY}/import", json={"conversation_id": cid, "ref": ref})
            assert made.status_code == 201, made.text
            assert (await client.get(made.json()["content_url"])).content == _red(20)
    finally:
        link.transport = real
        await old.aclose()


async def test_an_offline_instance_fails_fast(desk_and_nas: Pair) -> None:
    desk, nas = desk_and_nas
    link_id = await pair(desk, nas, name="nas")
    theirs = await _upload(nas, _red(20))
    nas.server.stop()
    async with desk.client() as client:
        cid = (await client.post("/v1/conversations", json={})).json()["id"]
        ref = {"origin": link_id, "artifact_id": theirs["id"]}
        refused = await client.post(f"{LIBRARY}/import", json={"conversation_id": cid, "ref": ref})
        assert refused.status_code == 503
        assert refused.json()["code"] == "engine_unavailable"
        page = await _library(client)
        assert {s["name"]: s["status"] for s in page["sources"]}["nas"] == "unreachable"
        assert page["items"] == []
