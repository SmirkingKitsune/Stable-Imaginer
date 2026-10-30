import uuid

import httpx

from tests.helpers import image_bytes

PROBLEM = "application/problem+json"


async def _create(client: httpx.AsyncClient, title: str | None = None) -> dict[str, object]:
    resp = await client.post("/v1/conversations", json={"title": title})
    assert resp.status_code == 201, resp.text
    body: dict[str, object] = resp.json()
    return body


async def test_create_and_get(client: httpx.AsyncClient) -> None:
    created = await _create(client, "first")
    assert uuid.UUID(str(created["id"])).version == 7
    assert created["title"] == "first"
    assert created["archived_at"] is None
    got = await client.get(f"/v1/conversations/{created['id']}")
    assert got.status_code == 200
    assert got.json() == created


async def test_metadata_round_trip(client: httpx.AsyncClient) -> None:
    resp = await client.post("/v1/conversations", json={"metadata": {"ui": {"pinned": True}}})
    assert resp.json()["metadata"] == {"ui": {"pinned": True}}


async def test_unknown_fields_rejected(client: httpx.AsyncClient) -> None:
    resp = await client.post("/v1/conversations", json={"titel": "typo"})
    assert resp.status_code == 422
    assert resp.headers["content-type"] == PROBLEM
    assert resp.json()["code"] == "validation_error"


async def test_list_orders_by_recent_activity_and_paginates(client: httpx.AsyncClient) -> None:
    ids = [str((await _create(client, f"c{i}"))["id"]) for i in range(5)]
    # Touch the oldest; it becomes the most recent.
    await client.patch(f"/v1/conversations/{ids[0]}", json={"title": "bumped"})
    expected = [ids[0], ids[4], ids[3], ids[2], ids[1]]

    seen: list[str] = []
    cursor: str | None = None
    while True:
        params: dict[str, str | int] = {"limit": 2}
        if cursor:
            params["cursor"] = cursor
        page = (await client.get("/v1/conversations", params=params)).json()
        seen += [c["id"] for c in page["items"]]
        cursor = page["next_cursor"]
        if cursor is None:
            break
    assert seen == expected


async def test_bad_cursor(client: httpx.AsyncClient) -> None:
    resp = await client.get("/v1/conversations", params={"cursor": "!!not-a-cursor"})
    assert resp.status_code == 422
    assert resp.json()["code"] == "invalid_request"


async def test_patch_is_partial(client: httpx.AsyncClient) -> None:
    created = await _create(client, "keep me")
    resp = await client.patch(f"/v1/conversations/{created['id']}", json={"metadata": {"k": 1}})
    body = resp.json()
    assert body["title"] == "keep me"
    assert body["metadata"] == {"k": 1}
    cleared = await client.patch(f"/v1/conversations/{created['id']}", json={"title": None})
    assert cleared.json()["title"] is None


async def test_archive_hides_from_default_list(client: httpx.AsyncClient) -> None:
    created = await _create(client)
    cid = created["id"]
    archived = (await client.patch(f"/v1/conversations/{cid}", json={"archived": True})).json()
    assert archived["archived_at"] is not None
    listed = (await client.get("/v1/conversations")).json()["items"]
    assert cid not in [c["id"] for c in listed]
    with_archived = (
        await client.get("/v1/conversations", params={"include_archived": "true"})
    ).json()["items"]
    assert cid in [c["id"] for c in with_archived]
    restored = (await client.patch(f"/v1/conversations/{cid}", json={"archived": False})).json()
    assert restored["archived_at"] is None


async def test_delete_then_404(client: httpx.AsyncClient) -> None:
    cid = (await _create(client))["id"]
    assert (await client.delete(f"/v1/conversations/{cid}")).status_code == 204
    resp = await client.get(f"/v1/conversations/{cid}")
    assert resp.status_code == 404
    assert resp.headers["content-type"] == PROBLEM
    assert resp.json()["code"] == "not_found"
    assert (await client.delete(f"/v1/conversations/{cid}")).status_code == 404
    assert (await client.patch(f"/v1/conversations/{cid}", json={"title": "x"})).status_code == 404


async def test_unknown_id_is_404_not_500(client: httpx.AsyncClient) -> None:
    for cid in ["nope", "../../etc", str(uuid.uuid4())]:
        assert (await client.get(f"/v1/conversations/{cid}")).status_code == 404


async def test_list_summarises_images_for_a_gallery(client: httpx.AsyncClient) -> None:
    pictured, empty = (await _create(client, "pictured"))["id"], (await _create(client))["id"]
    for _ in range(2):
        resp = await client.post(
            f"/v1/conversations/{pictured}/artifacts",
            files={"file": ("p.png", image_bytes(8, 6), "image/png")},
        )
        assert resp.status_code == 201, resp.text
    listed = {c["id"]: c for c in (await client.get("/v1/conversations")).json()["items"]}
    assert listed[pictured]["image_count"] == 2
    assert listed[pictured]["cover"]["label"] == "img-2"
    assert listed[pictured]["cover"]["content_url"].endswith("/content")
    assert (listed[empty]["image_count"], listed[empty]["cover"]) == (0, None)
    assert not listed[pictured]["turn_running"]
