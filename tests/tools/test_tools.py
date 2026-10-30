import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from imaginer.backends.registry import BackendRegistry
from imaginer.core.clock import utcnow
from imaginer.core.config import Settings
from imaginer.core.ids import new_id
from imaginer.db import models as orm
from imaginer.db.engine import Database
from imaginer.db.repositories import conversations as conv_repo
from imaginer.domain.enums import Capability
from imaginer.domain.events import EventType
from imaginer.orchestration.gates import EngineGates
from imaginer.orchestration.jobs import JobManager
from imaginer.services.artifacts import ArtifactService
from imaginer.storage.local_fs import LocalFSArtifactStore
from imaginer.tools.base import ToolContext, ToolRegistry
from imaginer.tools.generate_image import GenerateImageTool
from tests.conftest import SettingsFactory, sd_config
from tests.fakes.fake_sdcpp import FakeSdCpp


async def _noop(event_type: EventType, data: BaseModel) -> None:
    pass


class Bench:
    def __init__(self, db: Database, settings: Settings, root: Path) -> None:
        self.db = db
        self.registry = BackendRegistry.from_settings(settings)
        artifacts = ArtifactService(db, LocalFSArtifactStore(root), max_upload_bytes=10**8)
        self.jobs = JobManager(db, artifacts, EngineGates({"sd": 1}))
        self.tool = GenerateImageTool(self.registry, self.jobs)
        self.tools = ToolRegistry([self.tool])

    async def ctx(self) -> ToolContext:
        cid, now = new_id(), utcnow()
        async with self.db.session() as s, s.begin():
            await conv_repo.insert(s, orm.Conversation(id=cid, created_at=now, updated_at=now))
        return ToolContext(conversation_id=cid, message_id=None, tool_call_id="c1", emit=_noop)  # type: ignore[arg-type]

    async def call(self, arguments: Any, name: str = "generate_image") -> dict[str, Any]:
        raw = arguments if isinstance(arguments, str) else json.dumps(arguments)
        outcome = await self.tools.execute(name, raw, await self.ctx())
        return outcome.content


@pytest.fixture
async def bench(
    database: Database, make_settings: SettingsFactory, fake_sd: FakeSdCpp, tmp_path: Path
) -> AsyncIterator[Bench]:
    settings = make_settings(backends={"sd": sd_config(fake_sd.url)})
    bench = Bench(database, settings, tmp_path / "store")
    yield bench
    await bench.registry.aclose()


def test_schema_is_clean_json_schema(bench: Bench) -> None:
    schema = bench.tool.schema()
    assert schema["type"] == "function"
    function = schema["function"]
    assert function["name"] == "generate_image"
    params = function["parameters"]
    assert params["type"] == "object"
    assert params["required"] == ["prompt"]
    assert params["additionalProperties"] is False
    assert set(params["properties"]["aspect_ratio"]["anyOf"][0]["enum"]) >= {"1:1", "16:9"}
    assert '"title"' not in json.dumps(schema)  # pydantic's titles are stripped


async def test_happy_path(bench: Bench, fake_sd: FakeSdCpp) -> None:
    content = await bench.call({"prompt": "a lighthouse", "n": 2, "seed": 9})
    assert content["status"] == "succeeded"
    assert [(i["label"], i["seed"], i["transparent"]) for i in content["images"]] == [
        ("img-1", 9, False),
        ("img-2", 10, False),
    ]
    assert len(fake_sd.requests) == 2


async def test_transparent(bench: Bench) -> None:
    content = await bench.call({"prompt": "a logo", "transparent": True})
    assert content["images"][0]["transparent"] is True


@pytest.mark.parametrize(
    ("arguments", "code", "fragment"),
    [
        ("{not json", "invalid_arguments", "not valid JSON"),
        ({"prompt": ""}, "invalid_arguments", "prompt"),
        ({"prompt": "x", "n": 9}, "invalid_arguments", "n:"),
        ({"prompt": "x", "aspect_ratio": "5:1"}, "invalid_arguments", "aspect_ratio"),
        ({"prompt": "x", "style": "noir"}, "invalid_arguments", "style"),
    ],
)
async def test_bad_arguments_come_back_as_tool_errors(
    bench: Bench, arguments: Any, code: str, fragment: str
) -> None:
    content = await bench.call(arguments)
    assert content["status"] == "error"
    assert content["error"]["code"] == code
    assert fragment in content["error"]["detail"]


async def test_unknown_tool(bench: Bench) -> None:
    assert (await bench.call({}, name="launch_rocket"))["error"]["code"] == "unknown_tool"


async def test_engine_failure_is_a_tool_error(bench: Bench, fake_sd: FakeSdCpp) -> None:
    fake_sd.fail_next = "CUDA out of memory"
    content = await bench.call({"prompt": "x"})
    assert content["status"] == "error"
    assert content["error"]["code"] == "engine_error"
    assert "job_id" in content["error"]


async def test_missing_capability_is_a_tool_error(
    database: Database, make_settings: SettingsFactory, fake_sd: FakeSdCpp, tmp_path: Path
) -> None:
    settings = make_settings(
        backends={"sd": sd_config(fake_sd.url, declared_capabilities={Capability.T2I})}
    )
    bench = Bench(database, settings, tmp_path / "store")
    content = await bench.call({"prompt": "a sticker", "transparent": True})
    assert content["error"]["code"] == "capability_unavailable"
    assert fake_sd.requests == []  # never silently generated an opaque image instead
    await bench.registry.aclose()


async def test_not_offered_without_an_image_backend(
    database: Database, make_settings: SettingsFactory, tmp_path: Path
) -> None:
    bench = Bench(database, make_settings(), tmp_path / "store")
    assert bench.tools.offered() == []
    assert (await bench.call({"prompt": "x"}))["error"]["code"] == "unknown_tool"
