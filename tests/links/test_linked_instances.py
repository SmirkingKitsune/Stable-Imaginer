"""Two linked instances, over real sockets: a desktop that shares its image engine and
a NAS that shares its chat engine, each using the other's."""

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from imaginer.core.config import LinkShare, RoutingConfig
from imaginer.domain.enums import BackendKind
from tests.conftest import llama_config, sd_config
from tests.fakes.fake_llama import FakeLlama
from tests.fakes.fake_sdcpp import FakeSdCpp
from tests.links.conftest import Instance, InstanceFactory, linked, pair

Pair = tuple[Instance, Instance]


@pytest.fixture
def desk_and_nas(make_instance: InstanceFactory, fake_llama: FakeLlama, fake_sd: FakeSdCpp) -> Pair:
    desk = make_instance(
        "desk",
        backends={
            "sd": sd_config(fake_sd.url),
            "llama_nas": linked("nas", "llama", BackendKind.LLM),
        },
        routing=RoutingConfig(chat="llama_nas", t2i="sd", edit="sd"),
        share={"sd": LinkShare()},
    )
    nas = make_instance(
        "nas",
        backends={
            "llama": llama_config(fake_llama.url),
            "sd_desk": linked("desk", "sd", BackendKind.IMAGE),
        },
        routing=RoutingConfig(chat="llama", t2i="sd_desk", edit="sd_desk"),
        share={"llama": LinkShare()},
    )
    return desk, nas


async def _events(response: httpx.Response) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    event, data = "message", list[str]()
    async for line in response.aiter_lines():
        if not line:
            if data:
                yield event, json.loads("\n".join(data))
            event, data = "message", []
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].strip())


async def _say(client: httpx.AsyncClient, base: str, cid: str, text: str) -> list[str]:
    seen = []
    url = f"{base}/v1/conversations/{cid}/messages"
    async with client.stream("POST", url, json={"content": text}) as response:
        assert response.status_code == 200, await response.aread()
        async for event, _ in _events(response):
            seen.append(event)
    return seen


async def _until(check: Any, within_s: float = 10.0) -> None:
    deadline = asyncio.get_running_loop().time() + within_s
    while not await check():
        assert asyncio.get_running_loop().time() < deadline, "timed out"
        await asyncio.sleep(0.05)


async def test_pairing_links_both_ways(desk_and_nas: Pair) -> None:
    desk, nas = desk_and_nas
    link_id = await pair(desk, nas, name="nas")

    async with desk.client() as client:
        overview = (await client.get("/v1/links")).json()
    [link] = overview["links"]
    assert link["id"] == link_id and link["name"] == "nas" and link["status"] == "ok"
    assert link["peer_instance_id"] == nas.container.links.instance_id
    assert [e["name"] for e in link["engines"]] == ["llama"]
    assert link["engines"][0]["availability"]["state"] == "ready"
    assert link["used_by"] == ["llama_nas"]

    async def nas_sees_desk() -> bool:
        async with nas.client() as client:
            links = (await client.get("/v1/links")).json()["links"]
        return bool(links) and links[0]["status"] == "ok"

    await _until(nas_sees_desk)  # the NAS confirms the way back on its own
    async with nas.client() as client:
        [back] = (await client.get("/v1/links")).json()["links"]
    assert back["name"] == "desk"  # from the desk's own name
    assert [e["name"] for e in back["engines"]] == ["sd"]


async def test_a_code_works_once_and_wrong_codes_burn_it(desk_and_nas: Pair) -> None:
    desk, nas = desk_and_nas
    async with nas.client() as client:
        code = (await client.post("/v1/links/pairing-code")).json()["code"]
    async with desk.client() as client:
        for _ in range(5):
            wrong = await client.post("/v1/links", json={"url": nas.url, "code": "AAAA-AAAA"})
            assert wrong.status_code == 422
            assert wrong.json()["code"] == "pairing_failed"
        assert "used up" in wrong.json()["detail"]
        burned = await client.post("/v1/links", json={"url": nas.url, "code": code})
    assert burned.status_code == 422
    assert "no pairing code is active" in burned.json()["detail"]

    async with nas.client() as client:
        code = (await client.post("/v1/links/pairing-code")).json()["code"]
    async with desk.client() as client:
        first = await client.post("/v1/links", json={"url": nas.url, "code": code.lower()})
        assert first.status_code == 201
        again = await client.post("/v1/links", json={"url": nas.url, "code": code})
    assert again.status_code == 422


async def test_chat_runs_on_the_linked_instance(desk_and_nas: Pair, fake_llama: FakeLlama) -> None:
    desk, nas = desk_and_nas
    await pair(desk, nas, name="nas")
    async with desk.client() as client:
        cid = (await client.post("/v1/conversations", json={})).json()["id"]
        events = await _say(client, "", cid, "hello over there")
        assert events[-1] == "done"
        messages = (await client.get(f"/v1/conversations/{cid}/messages")).json()["items"]
    reply = messages[-1]
    assert reply["role"] == "assistant"
    assert reply["content_text"] == "You said: hello over there"
    assert reply["backend"] == "llama_nas"
    assert fake_llama.requests  # the NAS's engine did the work
    # The conversation stays on the desk: the NAS has none.
    async with nas.client() as client:
        assert (await client.get("/v1/conversations")).json()["items"] == []


async def test_images_come_from_the_linked_instance(desk_and_nas: Pair, fake_sd: FakeSdCpp) -> None:
    desk, nas = desk_and_nas
    await pair(nas, desk, name="desk")
    async with nas.client() as client:
        cid = (await client.post("/v1/conversations", json={})).json()["id"]
        events = await _say(client, "", cid, "draw a lighthouse")
        assert "artifact_created" in events and events[-1] == "done"
        artifacts = (await client.get(f"/v1/conversations/{cid}/artifacts")).json()["items"]
        assert [a["label"] for a in artifacts] == ["img-1"]
        content = await client.get(artifacts[0]["content_url"])
        assert content.status_code == 200 and content.content[:8] == b"\x89PNG\r\n\x1a\n"
        jobs = (await client.get(f"/v1/conversations/{cid}/jobs")).json()["items"]
    assert jobs[0]["backend"] == "sd_desk" and jobs[0]["status"] == "succeeded"
    assert fake_sd.requests  # the desk's engine made it
    async with desk.client() as client:
        assert (await client.get("/v1/conversations")).json()["items"] == []


async def test_gallery_shows_both_and_remote_conversations_run_at_their_origin(
    desk_and_nas: Pair,
) -> None:
    desk, nas = desk_and_nas
    link_id = await pair(desk, nas, name="nas")
    async with nas.client() as client:
        remote = (await client.post("/v1/conversations", json={"title": "on the nas"})).json()
    async with desk.client() as client:
        local = (await client.post("/v1/conversations", json={"title": "on the desk"})).json()
        page = (await client.get("/v1/gallery")).json()
        assert {s["name"]: s["status"] for s in page["sources"]} == {"desk": "ok", "nas": "ok"}
        origins = {item["title"]: item["origin"] for item in page["items"]}
        assert origins["on the desk"]["link_id"] is None
        assert origins["on the nas"]["link_id"] == link_id
        base = origins["on the nas"]["base_path"]
        assert base == f"/v1/links/{link_id}/r"

        # Continue the NAS's conversation from the desk: the turn runs on the NAS.
        events = await _say(client, base, remote["id"], "from the desk")
        assert events[-1] == "done"
        assert (await client.get(f"/v1/conversations/{remote['id']}")).status_code == 404
        through = (await client.get(f"{base}/v1/conversations/{remote['id']}/messages")).json()
    async with nas.client() as client:
        stored = (await client.get(f"/v1/conversations/{remote['id']}/messages")).json()
        assert stored == through
        assert stored["items"][-1]["content_text"] == "You said: from the desk"
        assert stored["items"][-1]["backend"] == "llama"
        # The desk's own conversation is not listed on the NAS's own listing.
        listed = (await client.get("/v1/conversations")).json()["items"]
    assert [c["id"] for c in listed] == [remote["id"]]
    assert local["id"] not in [c["id"] for c in listed]


async def test_gallery_pages_merge_by_activity(desk_and_nas: Pair) -> None:
    desk, nas = desk_and_nas
    await pair(desk, nas, name="nas")
    titles = []
    for n in range(5):
        owner = desk if n % 2 == 0 else nas
        async with owner.client() as client:
            await client.post("/v1/conversations", json={"title": f"c{n}"})
        titles.append(f"c{n}")
    seen: list[str] = []
    cursor = None
    async with desk.client() as client:
        while True:
            params: dict[str, Any] = {"limit": 2}
            if cursor:
                params["cursor"] = cursor
            page = (await client.get("/v1/gallery", params=params)).json()
            seen += [item["title"] for item in page["items"]]
            cursor = page["next_cursor"]
            if cursor is None:
                break
    assert seen == list(reversed(titles))


async def test_a_link_token_reaches_only_what_links_may(desk_and_nas: Pair) -> None:
    desk, nas = desk_and_nas
    link_id = await pair(desk, nas, name="nas")
    token = desk.container.links._outbound[link_id]  # what the desk calls the NAS with
    async with nas.client(token) as client:
        assert (await client.get("/v1/conversations")).status_code == 200
        assert (await client.get("/v1/link/status")).status_code == 200
        for path in ("/v1/engines", "/v1/links", "/v1/gallery"):
            assert (await client.get(path)).status_code == 403, path
        proxied = await client.get(f"/v1/links/{link_id}/r/v1/conversations")
        assert proxied.status_code == 403  # no hops through a linked instance
        unshared = await client.post("/v1/peer/engines/sd_desk/images", json={})
        assert unshared.status_code in (404, 422)
        assert (
            await client.post("/v1/peer/engines/nope/chat", json={"messages": []})
        ).status_code == 404
    async with nas.client() as client:  # the user's own token can't use the peer API
        assert (await client.get("/v1/link/status")).status_code == 403


async def test_read_only_gallery(make_instance: InstanceFactory, fake_llama: FakeLlama) -> None:
    owner = make_instance("owner", backends={"llama": llama_config(fake_llama.url)}, gallery="read")
    viewer = make_instance("viewer", backends={})
    link_id = await pair(viewer, owner)
    async with owner.client() as client:
        cid = (await client.post("/v1/conversations", json={})).json()["id"]
    base = f"/v1/links/{link_id}/r"
    async with viewer.client() as client:
        assert (await client.get(f"{base}/v1/conversations/{cid}")).status_code == 200
        denied = await client.post(
            f"{base}/v1/conversations/{cid}/messages", json={"content": "hi"}
        )
        assert denied.status_code == 403
        assert (await client.get(f"{base}/v1/engines")).status_code == 403  # not proxied


async def test_an_offline_link_fails_loudly(desk_and_nas: Pair) -> None:
    desk, nas = desk_and_nas
    await pair(desk, nas, name="nas")
    nas.server.stop()
    async with desk.client() as client:
        cid = (await client.post("/v1/conversations", json={})).json()["id"]
        refused = await client.post(f"/v1/conversations/{cid}/messages", json={"content": "hi"})
        assert refused.status_code == 503
        assert refused.json()["code"] == "engine_unavailable"
        page = (await client.get("/v1/gallery")).json()
        status = {s["name"]: s["status"] for s in page["sources"]}
        assert status == {"desk": "ok", "nas": "unreachable"}
        assert [i["id"] for i in page["items"]] == [cid]


async def test_unlinking_reaches_both_sides(desk_and_nas: Pair) -> None:
    desk, nas = desk_and_nas
    link_id = await pair(desk, nas, name="nas")
    async with desk.client() as client:
        assert (await client.delete(f"/v1/links/{link_id}")).status_code == 204
        assert (await client.get("/v1/links")).json()["links"] == []
        backends = {b["name"]: b for b in (await client.get("/v1/backends")).json()}
    assert backends["llama_nas"]["status"] == "down"
    assert "not linked" in backends["llama_nas"]["detail"]
    async with nas.client() as client:
        assert (await client.get("/v1/links")).json()["links"] == []


async def test_use_routes_a_task_to_a_shared_engine(
    make_instance: InstanceFactory, fake_sd: FakeSdCpp, fake_llama: FakeLlama
) -> None:
    gpu = make_instance("gpu", backends={"sd": sd_config(fake_sd.url)}, share={"sd": LinkShare()})
    laptop = make_instance("laptop", backends={"llama": llama_config(fake_llama.url)})
    link_id = await pair(laptop, gpu, name="gpu")
    async with laptop.client() as client:
        bad = await client.post(
            f"/v1/links/{link_id}/use", json={"engine": "sd", "tasks": ["chat"]}
        )
        assert bad.status_code == 422
        used = await client.post(
            f"/v1/links/{link_id}/use", json={"engine": "sd", "tasks": ["t2i", "edit"]}
        )
        assert used.status_code == 200, used.text
        backends = {b["name"]: b for b in (await client.get("/v1/backends")).json()}
        cid = (await client.post("/v1/conversations", json={})).json()["id"]
        events = await _say(client, "", cid, "draw a boat")
    assert backends["gpu_sd"]["status"] == "ok"
    assert backends["gpu_sd"]["default_for"] == ["t2i", "edit"]
    assert backends["gpu_sd"]["image_defaults"]["width"] == 256  # the sharing side's defaults
    assert "artifact_created" in events


async def test_a_moved_link_must_prove_it_is_the_same_instance(desk_and_nas: Pair) -> None:
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    from imaginer.links import pairing
    from tests.fakes.live import LiveServer

    desk, nas = desk_and_nas
    link_id = await pair(desk, nas, name="nas")
    links = desk.container.links
    nas_id = nas.container.links.instance_id
    token = links._outbound[link_id]

    # The real NAS proves it holds the hash of the token the desk calls it with.
    async with httpx.AsyncClient(base_url=nas.url) as client:
        hello = (
            await client.get("/v1/link/hello", params={"nonce": "n1", "asker": links.instance_id})
        ).json()
    expected = pairing.identity_proof(pairing.token_hash(token), "n1", nas_id)
    assert hello["proof"] == expected

    # An impostor can claim the NAS's id (it is advertised), but not prove it, and never
    # gets the token.
    seen: list[str] = []

    async def fake_hello(request: Request) -> JSONResponse:
        seen.append(request.headers.get("authorization", ""))
        return JSONResponse(
            {
                "instance_id": nas_id,
                "name": "nas",
                "api_version": 1,
                "version": "x",
                "challenge": "c",
                "proof": "0" * 64,
            }
        )

    async def fake_status(request: Request) -> JSONResponse:
        seen.append(request.headers.get("authorization", ""))
        return JSONResponse({}, status_code=500)

    impostor = Starlette(
        routes=[
            Route("/v1/link/hello", fake_hello),
            Route("/v1/link/status", fake_status),
        ]
    )
    with LiveServer(impostor) as fake:
        assert await links._proves_identity(links.get(link_id), fake.url) is False
    assert seen == [""]  # asked for proof without a token, and stopped there
