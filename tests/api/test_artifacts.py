import hashlib
from typing import Any

import httpx
import pytest

from imaginer.container import Container
from imaginer.core.config import StorageConfig
from imaginer.db.repositories.artifacts import ParentRef
from imaginer.domain.enums import ArtifactSource, LineageRelation
from imaginer.services.artifacts import NewArtifact
from imaginer.services.media import inspect_image_bytes
from tests.conftest import SettingsFactory
from tests.helpers import image_bytes


async def _conversation(client: httpx.AsyncClient) -> str:
    return str((await client.post("/v1/conversations", json={})).json()["id"])


async def _upload(
    client: httpx.AsyncClient,
    cid: str,
    data: bytes,
    filename: str = "pic.png",
    content_type: str = "image/png",
) -> httpx.Response:
    return await client.post(
        f"/v1/conversations/{cid}/artifacts", files={"file": (filename, data, content_type)}
    )


async def test_upload_a_mask_for_an_image(client: httpx.AsyncClient) -> None:
    cid = await _conversation(client)
    await _upload(client, cid, image_bytes(32, 24))
    resp = await client.post(
        f"/v1/conversations/{cid}/artifacts?kind=mask&mask_for=IMG-1",
        files={"file": ("mask.png", image_bytes(32, 24, color=(255, 255, 255)), "image/png")},
    )
    assert resp.status_code == 201, resp.text
    mask = resp.json()
    assert (mask["kind"], mask["label"]) == ("mask", "mask-2")
    assert mask["metadata"] == {"original_filename": "mask.png", "mask_for": "img-1"}
    images = await client.get(f"/v1/conversations/{cid}/artifacts?kind=image")
    assert [a["label"] for a in images.json()["items"]] == ["img-1"]
    (summary,) = (await client.get("/v1/conversations")).json()["items"]
    assert summary["image_count"] == 1 and summary["cover"]["label"] == "img-1"

    missing = await client.post(
        f"/v1/conversations/{cid}/artifacts?kind=mask&mask_for=img-9",
        files={"file": ("m.png", image_bytes(), "image/png")},
    )
    assert missing.status_code == 422 and "img-9" in missing.text
    stray = await client.post(
        f"/v1/conversations/{cid}/artifacts?mask_for=img-1",
        files={"file": ("m.png", image_bytes(), "image/png")},
    )
    assert stray.status_code == 422


async def test_upload_creates_labelled_artifact(client: httpx.AsyncClient) -> None:
    cid = await _conversation(client)
    data = image_bytes(16, 9, mode="RGBA", color=(1, 2, 3, 255), transparent_pixel=True)
    resp = await _upload(client, cid, data, filename="C:\\Users\\me\\Pictures\\cat.png")
    assert resp.status_code == 201, resp.text
    art = resp.json()
    assert art["conversation_id"] == cid
    assert (art["kind"], art["source"], art["label"]) == ("image", "upload", "img-1")
    assert (art["mime"], art["width"], art["height"], art["has_alpha"]) == (
        "image/png",
        16,
        9,
        True,
    )
    assert art["sha256"] == hashlib.sha256(data).hexdigest()
    assert art["size_bytes"] == len(data)
    assert art["metadata"] == {"original_filename": "cat.png"}
    assert art["content_url"] == f"/v1/artifacts/{art['id']}/content"

    second = (await _upload(client, cid, image_bytes(fmt="JPEG"), "b.jpg", "image/jpeg")).json()
    assert (second["label"], second["mime"]) == ("img-2", "image/jpeg")


async def test_mime_is_sniffed_not_trusted(client: httpx.AsyncClient) -> None:
    cid = await _conversation(client)
    resp = await _upload(client, cid, b"#!/bin/sh\necho pwned\n", "evil.png", "image/png")
    assert resp.status_code == 415
    assert resp.json()["code"] == "unsupported_media_type"
    jpeg_named_png = (await _upload(client, cid, image_bytes(fmt="JPEG"), "x.png")).json()
    assert jpeg_named_png["mime"] == "image/jpeg"


async def test_upload_to_missing_conversation(client: httpx.AsyncClient) -> None:
    resp = await _upload(client, "00000000-0000-7000-8000-000000000000", image_bytes())
    assert resp.status_code == 404


async def test_upload_size_limit(make_settings: SettingsFactory) -> None:
    from imaginer.main import create_app

    app = create_app(make_settings(storage=StorageConfig(sweep_interval_s=0, max_upload_bytes=200)))
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            cid = await _conversation(client)
            resp = await _upload(client, cid, image_bytes(64, 64, noise=True))
            assert resp.status_code == 413
            assert resp.json()["code"] == "payload_too_large"
            container: Container = app.state.container
            assert await container.store.list_objects(cid) == []


async def test_list_filters_and_pagination(client: httpx.AsyncClient, container: Container) -> None:
    cid = await _conversation(client)
    for _ in range(3):
        await _upload(client, cid, image_bytes())
    src = (await client.get(f"/v1/conversations/{cid}/artifacts")).json()["items"][0]
    info = inspect_image_bytes(image_bytes())
    await container.artifacts.ingest(
        NewArtifact.image(
            cid,
            info,
            ArtifactSource.EDITED,
            parents=(ParentRef(src["id"], LineageRelation.EDIT_SOURCE),),
        ),
        image_bytes(),
    )

    labels: list[str] = []
    cursor = None
    while True:
        params: dict[str, Any] = {"limit": 2}
        if cursor:
            params["cursor"] = cursor
        page = (await client.get(f"/v1/conversations/{cid}/artifacts", params=params)).json()
        labels += [a["label"] for a in page["items"]]
        cursor = page["next_cursor"]
        if not cursor:
            break
    assert labels == ["img-1", "img-2", "img-3", "img-4"]

    edited = await client.get(f"/v1/conversations/{cid}/artifacts", params={"source": "edited"})
    assert [a["label"] for a in edited.json()["items"]] == ["img-4"]
    files = await client.get(f"/v1/conversations/{cid}/artifacts", params={"kind": "file"})
    assert files.json()["items"] == []
    bad = await client.get(f"/v1/conversations/{cid}/artifacts", params={"kind": "hologram"})
    assert bad.status_code == 422


async def test_artifacts_never_cross_conversations(client: httpx.AsyncClient) -> None:
    a, b = await _conversation(client), await _conversation(client)
    art = (await _upload(client, a, image_bytes())).json()
    assert (await client.get(f"/v1/conversations/{b}/artifacts")).json()["items"] == []
    assert (await client.get(f"/v1/conversations/{a}/artifacts")).json()["items"][0]["id"] == art[
        "id"
    ]


async def test_content_bytes_mime_and_etag(client: httpx.AsyncClient) -> None:
    cid = await _conversation(client)
    data = image_bytes(fmt="WEBP")
    art = (await _upload(client, cid, data, "w.webp", "image/webp")).json()

    resp = await client.get(art["content_url"])
    assert resp.status_code == 200
    assert resp.content == data
    assert resp.headers["content-type"] == "image/webp"
    assert resp.headers["etag"] == f'"{art["sha256"]}"'
    assert "immutable" in resp.headers["cache-control"]
    assert resp.headers["content-disposition"].startswith("inline")

    cached = await client.get(art["content_url"], headers={"If-None-Match": resp.headers["etag"]})
    assert cached.status_code == 304
    assert cached.content == b""
    stale = await client.get(art["content_url"], headers={"If-None-Match": '"deadbeef"'})
    assert stale.status_code == 200


async def test_metadata_and_lineage_endpoints(
    client: httpx.AsyncClient, container: Container
) -> None:
    cid = await _conversation(client)
    src = (await _upload(client, cid, image_bytes())).json()
    ref = (await _upload(client, cid, image_bytes())).json()
    info = inspect_image_bytes(image_bytes())
    child = await container.artifacts.ingest(
        NewArtifact.image(
            cid,
            info,
            ArtifactSource.EDITED,
            parents=(
                ParentRef(src["id"], LineageRelation.EDIT_SOURCE),
                ParentRef(ref["id"], LineageRelation.REFERENCE, 1),
            ),
        ),
        image_bytes(),
    )
    meta = (await client.get(f"/v1/artifacts/{child.id}")).json()
    assert (meta["label"], meta["source"]) == ("img-3", "edited")

    lineage = (await client.get(f"/v1/artifacts/{child.id}/lineage")).json()
    assert lineage["children"] == []
    assert [(p["label"], p["relation"], p["ordinal"]) for p in lineage["parents"]] == [
        ("img-1", "edit_source", 0),
        ("img-2", "reference", 1),
    ]
    up = (await client.get(f"/v1/artifacts/{src['id']}/lineage")).json()
    assert [c["label"] for c in up["children"]] == ["img-3"]


async def test_lineage_parents_must_share_the_conversation(
    client: httpx.AsyncClient, container: Container
) -> None:
    a, b = await _conversation(client), await _conversation(client)
    foreign = (await _upload(client, a, image_bytes())).json()
    info = inspect_image_bytes(image_bytes())
    from imaginer.domain.errors import InvalidRequestError

    with pytest.raises(InvalidRequestError):
        await container.artifacts.ingest(
            NewArtifact.image(
                b,
                info,
                ArtifactSource.EDITED,
                parents=(ParentRef(foreign["id"], LineageRelation.EDIT_SOURCE),),
            ),
            image_bytes(),
        )
    # The rejected artifact left neither a row nor a file behind.
    assert (await client.get(f"/v1/conversations/{b}/artifacts")).json()["items"] == []
    assert await container.store.list_objects(b) == []


async def test_label_lookup_is_scoped(client: httpx.AsyncClient, container: Container) -> None:
    from imaginer.domain.errors import NotFoundError

    a, b = await _conversation(client), await _conversation(client)
    await _upload(client, a, image_bytes())
    await _upload(client, a, image_bytes())
    await _upload(client, b, image_bytes())
    assert (await container.artifacts.get_by_label(a, "img-2")).conversation_id == a
    with pytest.raises(NotFoundError):
        await container.artifacts.get_by_label(b, "img-2")


async def test_unknown_artifact(client: httpx.AsyncClient) -> None:
    missing = "00000000-0000-7000-8000-000000000000"
    for path in ["", "/content", "/lineage"]:
        assert (await client.get(f"/v1/artifacts/{missing}{path}")).status_code == 404
