"""FastAPI application factory.

``uvicorn imaginer.main:create_app --factory`` works too; the profile then comes from
``$IMAGINER_PROFILE``.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import APIRouter, Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from imaginer import __version__
from imaginer.api import artifacts, backends, conversations, engines, jobs, links, messages
from imaginer.api.auth import allow_gallery, require_auth, require_user
from imaginer.api.errors import install_error_handlers
from imaginer.api.middleware import RequestContextMiddleware
from imaginer.api.ui import install_ui
from imaginer.container import Container
from imaginer.core.config import Settings, load_settings
from imaginer.core.logging import configure_logging
from imaginer.domain.models import Health


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings if settings is not None else load_settings()
    configure_logging(settings.logging.level, settings.logging.format)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        container = await Container.start(settings)
        app.state.container = container
        try:
            yield
        finally:
            await container.aclose()

    app = FastAPI(
        title="Stable-Imaginer",
        version=__version__,
        summary="Local multimodal conversation server.",
        lifespan=lifespan,
    )
    if settings.server.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.server.cors_origins,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["X-Request-ID", "ETag"],
        )
    app.add_middleware(RequestContextMiddleware)
    install_error_handlers(app)

    @app.get("/healthz", tags=["health"])
    async def healthz() -> Health:
        """Liveness of the orchestrator itself (no auth). Engine health: GET /v1/backends."""
        return Health(status="ok", version=__version__)

    # Conversations: the user, and linked instances as far as links.gallery allows.
    shared = APIRouter(dependencies=[Depends(require_auth), Depends(allow_gallery)])
    shared.include_router(conversations.router)
    shared.include_router(messages.router)
    shared.include_router(artifacts.conversation_router)
    shared.include_router(artifacts.router)
    shared.include_router(jobs.conversation_router)
    shared.include_router(jobs.router)
    shared.include_router(backends.router)
    app.include_router(shared)
    # Administration and the merged gallery: the user only (no hops through a linked instance).
    own = APIRouter(dependencies=[Depends(require_auth), Depends(require_user)])
    own.include_router(engines.router)
    own.include_router(links.router)
    own.include_router(links.gallery_router)
    app.include_router(own)
    # Linked instances: pairing carries its own proof; the peer API needs a link token.
    app.include_router(links.pairing_router)
    app.include_router(links.peer_router)
    install_ui(app)
    return app
