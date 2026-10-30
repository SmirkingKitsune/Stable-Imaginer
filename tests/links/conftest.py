"""Two (or more) real instances on live sockets, linked the way users link them."""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from imaginer.container import Container
from imaginer.core.config import (
    BackendConfig,
    LinkedBackendConfig,
    LinksConfig,
    LinkShare,
    LoggingConfig,
    OrchestrationConfig,
    RoutingConfig,
    ServerConfig,
    Settings,
    StorageConfig,
)
from imaginer.domain.enums import BackendKind
from imaginer.main import create_app
from tests.fakes.live import LiveServer


@dataclass
class Instance:
    name: str
    settings: Settings
    app: FastAPI
    server: LiveServer
    token: str

    @property
    def url(self) -> str:
        return self.server.url

    @property
    def container(self) -> Container:
        result: Container = self.app.state.container
        return result

    def client(self, token: str | None = None) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self.url,
            timeout=20,
            headers={"Authorization": f"Bearer {token or self.token}"},
        )


def linked(link: str, engine: str, kind: BackendKind) -> LinkedBackendConfig:
    return LinkedBackendConfig(type="linked", link=link, engine=engine, kind=kind)


InstanceFactory = Callable[..., Instance]


@pytest.fixture
def make_instance(tmp_path: Path) -> Iterator[InstanceFactory]:
    started: list[Instance] = []

    def factory(
        name: str,
        *,
        backends: dict[str, BackendConfig],
        routing: RoutingConfig | None = None,
        share: dict[str, LinkShare | None] | None = None,
        **links: Any,
    ) -> Instance:
        token = f"user-token-of-{name}"
        settings = Settings(
            data_root=tmp_path / name,
            storage=StorageConfig(sweep_interval_s=0),
            logging=LoggingConfig(level="WARNING"),
            server=ServerConfig(auth_token=token),  # type: ignore[arg-type]
            backends=backends,
            routing=routing or RoutingConfig(),
            orchestration=OrchestrationConfig(probe_ttl_s=0, max_tool_iterations=2),
            links=LinksConfig(
                enabled=True,
                name=name,
                advertise_url="http://127.0.0.1:1",  # replaced once the port is known
                discovery=False,
                poll_interval_s=3600,
                share=share or {},
                **links,
            ),
        )
        app = create_app(settings)
        server = LiveServer(app, lifespan="on")
        settings.links.advertise_url = server.url
        instance = Instance(name, settings, app, server.start(), token)
        started.append(instance)
        return instance

    yield factory
    for instance in reversed(started):
        instance.server.stop()


async def pair(requester: Instance, responder: Instance, name: str | None = None) -> str:
    """Link two instances as a user would: a code on one, typed into the other."""
    async with responder.client() as client:
        code = (await client.post("/v1/links/pairing-code")).json()["code"]
    async with requester.client() as client:
        body: dict[str, Any] = {"url": responder.url, "code": code}
        if name:
            body["name"] = name
        made = await client.post("/v1/links", json=body)
        assert made.status_code == 201, made.text
        link_id: str = made.json()["id"]
        return link_id
