import base64
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from imaginer.backends.registry import BackendRegistry
from imaginer.core.clock import utcnow
from imaginer.core.config import SdCppBackendConfig
from imaginer.core.ids import new_id
from imaginer.db import models as orm
from imaginer.db.engine import Database
from imaginer.db.repositories import conversations as conv_repo
from imaginer.domain.enums import ArtifactKind, ArtifactSource, Capability
from imaginer.domain.events import EventType
from imaginer.domain.models import ImageRequestOptions
from imaginer.orchestration.gates import EngineGates
from imaginer.orchestration.jobs import JobManager
from imaginer.services.artifacts import ArtifactService, NewArtifact
from imaginer.services.media import inspect_image_bytes
from imaginer.storage.local_fs import LocalFSArtifactStore
from imaginer.tools.base import ToolContext, ToolRegistry
from imaginer.tools.edit_image import EditImageTool
from tests.conftest import SettingsFactory, sd_config
from tests.fakes.fake_sdcpp import FakeSdCpp
from tests.helpers import image_bytes


async def _noop(event_type: EventType, data: BaseModel) -> None:
    pass


class Bench:
    def __init__(
        self, db: Database, sd: SdCppBackendConfig, make_settings: SettingsFactory, root: Path
    ) -> None:
        self.db = db
        self.registry = BackendRegistry.from_settings(make_settings(backends={"sd": sd}))
        self.artifacts = ArtifactService(db, LocalFSArtifactStore(root), max_upload_bytes=10**8)
        self.jobs = JobManager(db, self.artifacts, EngineGates({"sd": 1}))
        self.tools = ToolRegistry([EditImageTool(self.registry, self.jobs, self.artifacts)])

    async def conversation(self) -> str:
        cid, now = new_id(), utcnow()
        async with self.db.session() as s, s.begin():
            await conv_repo.insert(s, orm.Conversation(id=cid, created_at=now, updated_at=now))
        return cid

    async def upload(self, cid: str, data: bytes, kind: ArtifactKind = ArtifactKind.IMAGE) -> str:
        info = inspect_image_bytes(data)
        artifact = await self.artifacts.ingest(
            NewArtifact.image(cid, info, ArtifactSource.UPLOAD, kind), data
        )
        return artifact.label

    async def edit(
        self, cid: str, arguments: dict[str, Any], options: ImageRequestOptions | None = None
    ) -> dict[str, Any]:
        ctx = ToolContext(
            conversation_id=cid,
            message_id=None,  # type: ignore[arg-type]
            tool_call_id="c1",
            emit=_noop,
            image_options=options,
        )
        outcome = await self.tools.execute("edit_image", json.dumps(arguments), ctx)
        return outcome.content


@pytest.fixture
async def bench(
    database: Database, make_settings: SettingsFactory, fake_sd: FakeSdCpp, tmp_path: Path
) -> AsyncIterator[Bench]:
    bench = Bench(database, sd_config(fake_sd.url), make_settings, tmp_path / "store")
    yield bench
    await bench.registry.aclose()


async def test_multi_reference_edit_with_mask_and_lineage(bench: Bench, fake_sd: FakeSdCpp) -> None:
    cid = await bench.conversation()
    cat = image_bytes(320, 256, color=(200, 100, 50))
    hat = image_bytes(256, 256, color=(10, 10, 200))
    mask = image_bytes(320, 256, color=(255, 255, 255))
    for data in (cat, hat, mask):
        await bench.upload(cid, data)

    content = await bench.edit(
        cid,
        {
            "images": ["img-1", " IMG-2 "],
            "mask": "img-3",
            "prompt": "put the hat from image 2 on the cat",
        },
    )
    assert content["status"] == "succeeded"
    assert content["edited"] == ["img-1", "img-2"] and content["mask"] == "img-3"
    (output,) = content["images"]
    assert output["label"] == "img-4"

    body = fake_sd.requests[-1]
    sent = [base64.b64decode(r.split(",", 1)[1]) for r in body["ref_images"]]
    assert sent == [cat, hat, mask]  # in order, mask last, small enough to send untouched
    assert (body["width"], body["height"]) == (320, 256)  # shape of the first image

    child = await bench.artifacts.get_by_label(cid, "img-4")
    assert child.source is ArtifactSource.EDITED
    lineage = await bench.artifacts.lineage(child.id)
    assert [(p.label, p.relation.value, p.ordinal) for p in lineage.parents] == [
        ("img-1", "edit_source", 0),
        ("img-2", "reference", 1),
        ("img-3", "mask", 0),
    ]


async def test_unknown_label_lists_what_exists(bench: Bench, fake_sd: FakeSdCpp) -> None:
    cid = await bench.conversation()
    await bench.upload(cid, image_bytes())
    content = await bench.edit(cid, {"images": ["img-7"], "prompt": "x"})
    assert content["error"]["code"] == "invalid_request"
    assert (
        content["error"]["detail"] == "unknown image label(s) img-7; this conversation has: img-1"
    )
    assert fake_sd.requests == []


async def test_labels_never_cross_conversations(bench: Bench, fake_sd: FakeSdCpp) -> None:
    mine, theirs = await bench.conversation(), await bench.conversation()
    await bench.upload(mine, image_bytes())
    for _ in range(3):
        await bench.upload(theirs, image_bytes())
    # img-2 and img-3 exist, but only in the other conversation.
    content = await bench.edit(mine, {"images": ["img-1", "img-3"], "prompt": "x"})
    assert "unknown image label(s) img-3" in content["error"]["detail"]
    assert fake_sd.requests == []


@pytest.mark.parametrize(
    ("arguments", "fragment"),
    [
        ({"images": [], "prompt": "x"}, "images"),
        ({"images": ["img-1", "img-1"], "prompt": "x"}, "must not repeat"),
        ({"images": ["img-1"], "mask": "img-1", "prompt": "x"}, "mask must not"),
        ({"images": ["picture one"], "prompt": "x"}, "images.0"),
        ({"images": ["img-1"]}, "prompt"),
    ],
)
async def test_invalid_arguments(bench: Bench, arguments: dict[str, Any], fragment: str) -> None:
    cid = await bench.conversation()
    content = await bench.edit(cid, arguments)
    assert content["error"]["code"] == "invalid_arguments"
    assert fragment in content["error"]["detail"]


async def test_multi_reference_needs_the_capability(
    database: Database, make_settings: SettingsFactory, fake_sd: FakeSdCpp, tmp_path: Path
) -> None:
    config = sd_config(fake_sd.url, declared_capabilities={Capability.T2I, Capability.EDIT})
    bench = Bench(database, config, make_settings, tmp_path / "store")
    cid = await bench.conversation()
    await bench.upload(cid, image_bytes())
    await bench.upload(cid, image_bytes())
    two = await bench.edit(cid, {"images": ["img-1", "img-2"], "prompt": "combine"})
    assert two["error"]["code"] == "capability_unavailable"
    masked = await bench.edit(cid, {"images": ["img-1"], "mask": "img-2", "prompt": "x"})
    assert masked["error"]["code"] == "capability_unavailable"  # a mask is a second input
    one = await bench.edit(cid, {"images": ["img-1"], "prompt": "brighter"})
    assert one["status"] == "succeeded"
    await bench.registry.aclose()


async def test_input_limit(
    database: Database, make_settings: SettingsFactory, fake_sd: FakeSdCpp, tmp_path: Path
) -> None:
    bench = Bench(database, sd_config(fake_sd.url, max_ref_images=2), make_settings, tmp_path / "s")
    cid = await bench.conversation()
    for _ in range(3):
        await bench.upload(cid, image_bytes())
    content = await bench.edit(cid, {"images": ["img-1", "img-2"], "mask": "img-3", "prompt": "x"})
    assert (
        content["error"]["detail"]
        == "at most 2 input images (including a mask) are supported, got 3"
    )
    await bench.registry.aclose()


async def test_partial_outputs_are_reported(bench: Bench, fake_sd: FakeSdCpp) -> None:
    fake_sd.fail_at_request = 2
    cid = await bench.conversation()
    await bench.upload(cid, image_bytes())
    content = await bench.edit(cid, {"images": ["img-1"], "prompt": "x", "n": 3})
    assert content["status"] == "error"
    assert [i["label"] for i in content["error"]["images"]] == ["img-2"]


async def test_user_mask_and_sampler_overrides(bench: Bench, fake_sd: FakeSdCpp) -> None:
    cid = await bench.conversation()
    photo = image_bytes(320, 256, color=(200, 100, 50))
    mask = image_bytes(320, 256, color=(255, 255, 255))
    await bench.upload(cid, photo)
    assert await bench.upload(cid, mask, ArtifactKind.MASK) == "mask-2"

    options = ImageRequestOptions(
        mask="mask-2", steps=8, cfg_scale=4.5, negative_prompt="blurry", width=512, height=384
    )
    content = await bench.edit(cid, {"images": ["img-1"], "prompt": "stormy sky"}, options)
    assert content["status"] == "succeeded" and content["mask"] == "mask-2"
    body = fake_sd.requests[-1]
    sent = [base64.b64decode(r.split(",", 1)[1]) for r in body["ref_images"]]
    assert sent == [photo, mask]
    assert body["sample_params"]["sample_steps"] == 8
    assert body["sample_params"]["guidance"] == {"txt_cfg": 4.5}
    assert body["negative_prompt"] == "blurry"
    assert (body["width"], body["height"]) == (512, 384)

    (output,) = content["images"]
    child = await bench.artifacts.get_by_label(cid, output["label"])
    lineage = await bench.artifacts.lineage(child.id)
    assert ("mask-2", "mask") in [(p.label, p.relation.value) for p in lineage.parents]


async def test_a_mask_is_not_an_image_to_edit(bench: Bench, fake_sd: FakeSdCpp) -> None:
    cid = await bench.conversation()
    await bench.upload(cid, image_bytes())
    await bench.upload(cid, image_bytes(), ArtifactKind.MASK)
    content = await bench.edit(cid, {"images": ["mask-2"], "prompt": "x"})
    assert content["error"]["detail"] == "mask-2 is a mask, not an image"
    assert fake_sd.requests == []
