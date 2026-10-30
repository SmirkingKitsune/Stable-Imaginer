"""The Links screen's operations: overview, routing tasks to a linked engine, sharing.

Changes are applied at once and saved to ``config/<profile>.local.yaml``, like the
Engines screen's, so the profile itself is never rewritten.
"""

from typing import Any

from imaginer.backends.registry import BackendRegistry
from imaginer.core.config import LinkedBackendConfig, LinkShare, Settings, backend_kind
from imaginer.domain import models as dto
from imaginer.domain.enums import BackendKind
from imaginer.domain.errors import ConflictError, InvalidRequestError
from imaginer.engines.service import EngineService
from imaginer.links.service import LinkService
from imaginer.links.state import Link
from imaginer.orchestration.gates import EngineGates

_KIND_OF_TASK = {"chat": BackendKind.LLM, "t2i": BackendKind.IMAGE, "edit": BackendKind.IMAGE}


class LinkAdmin:
    def __init__(
        self,
        settings: Settings,
        links: LinkService,
        registry: BackendRegistry,
        engines: EngineService,
        gates: EngineGates,
    ) -> None:
        self._settings = settings
        self._links = links
        self._registry = registry
        self._engines = engines
        self._gates = gates

    def info(self, link: Link) -> dto.LinkInfo:
        used_by = [
            name
            for name, config in self._settings.backends.items()
            if isinstance(config, LinkedBackendConfig) and config.link == link.name
        ]
        return dto.LinkInfo(
            id=link.id,
            name=link.name,
            peer_instance_id=link.peer_instance_id,
            peer_name=link.peer_name,
            url=link.url,
            created_at=link.created_at,
            status=link.status,
            detail=link.detail,
            last_seen_at=link.seen_at,
            gallery=link.peer.gallery if link.peer is not None else None,
            engines=list(link.engines.values()) if link.status == "ok" else [],
            used_by=used_by,
        )

    def overview(self) -> dto.LinksOverview:
        config = self._settings.links
        path = self._engines.settings_file()
        return dto.LinksOverview(
            enabled=config.enabled,
            instance_id=self._links.instance_id,
            name=self._links.name,
            advertise_url=self._links.advertise_url if config.enabled else None,
            discovery=config.enabled and config.discovery,
            gallery=config.gallery,
            share={
                name: dto.LinkShareInfo(remote_start=share.remote_start)
                for name, share in config.shared().items()
            },
            links=[self.info(link) for link in self._links.all()],
            settings_file=str(path) if path is not None else None,
        )

    async def use(self, link_id: str, body: dto.LinkUse) -> dto.LinksOverview:
        """Add a local backend for an engine the linked instance shares, and route tasks to it."""
        link = self._links.get(link_id)
        await self._links.poll(link)
        engine = link.engine(body.engine)
        if engine is None:
            raise InvalidRequestError(
                f"{link.peer_name} doesn't share an engine named '{body.engine}'"
                + (f" ({link.detail})" if link.detail else "")
            )
        for task in body.tasks:
            if _KIND_OF_TASK[task] is not engine.kind:
                raise InvalidRequestError(
                    f"'{body.engine}' is an {engine.kind.value} engine and can't do {task}"
                )
        name = body.backend or _backend_name(link.name, engine.name)
        existing = self._settings.backends.get(name)
        if existing is not None and not isinstance(existing, LinkedBackendConfig):
            raise ConflictError(f"a backend named '{name}' already exists; choose another name")
        config = LinkedBackendConfig(
            type="linked", link=link.name, engine=engine.name, kind=engine.kind
        )
        patch: dict[str, object] = {
            "backends": {
                name: config.model_dump(mode="json", include={"type", "link", "engine", "kind"})
            },
        }
        if body.tasks:
            patch["routing"] = dict.fromkeys(body.tasks, name)
        await self._engines.save_settings(patch)
        self._registry.add(name, config)
        if backend_kind(config) is BackendKind.IMAGE:
            self._gates.set_limit(name, config.concurrency)
        for task in body.tasks:
            setattr(self._settings.routing, task, name)
        return self.overview()

    async def update_settings(self, body: dto.LinksSettingsUpdate) -> dto.LinksOverview:
        config = self._settings.links
        patch: dict[str, object] = {}
        if body.share is not None:
            for name in body.share:
                backend = self._settings.backends.get(name)
                if backend is None:
                    raise InvalidRequestError(f"no backend named '{name}'")
                if isinstance(backend, LinkedBackendConfig):
                    raise InvalidRequestError(
                        f"'{name}' runs on a linked instance; only local engines can be shared"
                    )
            share: dict[str, LinkShare | None] = {
                name: LinkShare(remote_start=s.remote_start) for name, s in body.share.items()
            }
            # null un-shares an engine the profile itself shares
            saved: dict[str, dict[str, Any] | None] = {
                name: None for name in config.shared() if name not in share
            }
            saved.update({n: s.model_dump() for n, s in share.items() if s is not None})
            patch["share"] = saved
            config.share = share
        if body.gallery is not None:
            patch["gallery"] = body.gallery
            config.gallery = body.gallery
        if patch:
            await self._engines.save_settings({"links": patch})
        return self.overview()


def _backend_name(link: str, engine: str) -> str:
    return f"{link}_{engine}"[:32].rstrip("_")
