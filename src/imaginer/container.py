"""Process-wide wiring: one instance per running app, created in the FastAPI lifespan."""

import asyncio
import contextlib
from dataclasses import dataclass, field

from imaginer.backends.registry import FACTORIES, BackendRegistry
from imaginer.core.config import Settings, backend_kind, is_loopback_host
from imaginer.core.logging import get_logger
from imaginer.db import migrate
from imaginer.db.engine import Database
from imaginer.domain.enums import BackendKind
from imaginer.engines.service import EngineService
from imaginer.links.admin import LinkAdmin
from imaginer.links.gallery import Gallery
from imaginer.links.serve import PeerEngines
from imaginer.links.service import LinkService
from imaginer.orchestration.context_builder import ContextBuilder
from imaginer.orchestration.event_bus import TurnManager
from imaginer.orchestration.gates import EngineGates
from imaginer.orchestration.jobs import JobManager
from imaginer.orchestration.turn_runner import TurnRunner
from imaginer.services.artifacts import ArtifactService
from imaginer.services.conversations import ConversationService
from imaginer.services.sweeper import Sweeper
from imaginer.storage.local_fs import LocalFSArtifactStore
from imaginer.tools.base import ToolRegistry
from imaginer.tools.edit_image import EditImageTool
from imaginer.tools.generate_image import GenerateImageTool

log = get_logger(__name__)


@dataclass
class Container:
    settings: Settings
    db: Database
    store: LocalFSArtifactStore
    conversations: ConversationService
    artifacts: ArtifactService
    sweeper: Sweeper
    registry: BackendRegistry
    turns: TurnManager
    jobs: JobManager
    turn_runner: TurnRunner
    engines: EngineService
    gates: EngineGates
    links: LinkService
    peers: PeerEngines
    link_admin: LinkAdmin
    gallery: Gallery
    _tasks: list[asyncio.Task[None]] = field(default_factory=list)

    @classmethod
    async def start(cls, settings: Settings) -> "Container":
        await asyncio.to_thread(settings.data_root.mkdir, parents=True, exist_ok=True)
        url = settings.database_url
        if settings.db.auto_migrate:
            # Alembic's env runs its own event loop, so keep it off ours.
            await asyncio.to_thread(migrate.upgrade, url)
        db = Database(url, echo=settings.db.echo)
        store = LocalFSArtifactStore(settings.data_root)
        conversations = ConversationService(db, store)
        artifacts = ArtifactService(db, store, max_upload_bytes=settings.storage.max_upload_bytes)
        sweeper = Sweeper(db, store, conversations, orphan_grace_s=settings.storage.orphan_grace_s)
        await sweeper.run_once()
        # Before the registry: linked backends resolve their link through it.
        links = LinkService(settings, db)
        await links.start()
        registry = BackendRegistry.from_settings(
            settings, factories={**FACTORIES, "linked": links.backend_factory}
        )
        gates = EngineGates(
            {
                name: getattr(config, "concurrency", 1)
                for name, config in settings.backends.items()
                if backend_kind(config) is BackendKind.IMAGE
            }
        )
        jobs = JobManager(db, artifacts, gates)
        await jobs.fail_orphans()
        tools = ToolRegistry(
            [GenerateImageTool(registry, jobs), EditImageTool(registry, jobs, artifacts)]
        )
        turns = TurnManager()
        turn_runner = TurnRunner(
            db=db,
            registry=registry,
            turns=turns,
            context=ContextBuilder(
                db,
                store,
                settings.context,
                feed_outputs_to_vision=settings.orchestration.feed_outputs_to_vision,
            ),
            tools=tools,
            config=settings.orchestration,
        )

        async def stop_work(conversation_id: str) -> None:
            await turns.cancel(conversation_id)
            await jobs.cancel_conversation(conversation_id)

        conversations.add_delete_hook(stop_work)
        engines = EngineService(settings, registry)
        container = cls(
            settings=settings,
            db=db,
            store=store,
            conversations=conversations,
            artifacts=artifacts,
            sweeper=sweeper,
            registry=registry,
            turns=turns,
            jobs=jobs,
            turn_runner=turn_runner,
            engines=engines,
            gates=gates,
            links=links,
            peers=PeerEngines(settings, registry, engines, gates),
            link_admin=LinkAdmin(settings, links, registry, engines, gates),
            gallery=Gallery(turns=turns, conversations=conversations, links=links),
        )
        # Probe engines in the background: the API must come up even when they are down.
        container._tasks.append(asyncio.create_task(registry.refresh()))
        container._tasks.append(asyncio.create_task(container.engines.autostart()))
        if settings.storage.sweep_interval_s > 0:
            container._tasks.append(
                asyncio.create_task(sweeper.run_forever(settings.storage.sweep_interval_s))
            )
        _warn_if_exposed(settings)
        log.info(
            "app.started",
            profile=settings.profile,
            data_root=str(settings.data_root),
            backends=registry.names(),
        )
        return container

    async def aclose(self) -> None:
        await self.turns.cancel_all()
        await self.jobs.cancel_all()
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        await self.engines.aclose()
        await self.registry.aclose()
        await self.links.aclose()
        await self.db.dispose()


def _warn_if_exposed(settings: Settings) -> None:
    host = settings.server.host
    if not is_loopback_host(host) and settings.server.auth_token is None:
        log.warning(
            "server.exposed_without_token",
            host=host,
            hint="set IMAGINER_SERVER__AUTH_TOKEN before binding beyond localhost",
        )
