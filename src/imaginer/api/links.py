"""Linked instances: the Links screen (user only), and the peer API other instances call.

``/v1/link/hello`` and ``/v1/link/pair`` need no token: they carry their own proof (the
pairing code). Everything else under ``/v1/link`` and ``/v1/peer`` is for linked instances,
with the token they were issued. All of it answers 404 while linking is off.
"""

import json
from collections.abc import AsyncIterator
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query, Request, Response, status
from fastapi.responses import StreamingResponse
from sse_starlette import EventSourceResponse, ServerSentEvent

from imaginer import __version__
from imaginer.api.auth import PrincipalDep, require_link
from imaginer.api.deps import ContainerDep
from imaginer.api.sse import PING_INTERVAL_S
from imaginer.core.logging import get_logger
from imaginer.domain import models as dto
from imaginer.domain.enums import BackendKind
from imaginer.domain.errors import ImaginerError, NotFoundError
from imaginer.links import proxy
from imaginer.links.state import PEER_API_VERSION

log = get_logger(__name__)


async def links_on(container: ContainerDep) -> None:
    if not container.settings.links.enabled:
        raise NotFoundError("linking is off on this instance (links.enabled)")


# --- the Links screen ------------------------------------------------------------------

router = APIRouter(prefix="/v1/links", tags=["links"])


@router.get("")
async def links_overview(container: ContainerDep) -> dto.LinksOverview:
    """This instance's identity and sharing, and every link with what its peer shares."""
    return container.link_admin.overview()


@router.get("/discovered")
async def discovered(container: ContainerDep) -> list[dto.DiscoveredInstance]:
    """Instances seen on the network (mDNS), linked or not."""
    discovery = container.links.discovery
    return discovery.instances() if discovery is not None else []


@router.post("/pairing-code", status_code=status.HTTP_201_CREATED, dependencies=[Depends(links_on)])
async def pairing_code(container: ContainerDep) -> dto.PairingCode:
    """A code another instance can link with, for 10 minutes. Any earlier code stops working."""
    return await container.links.mint_code()


@router.post("", status_code=status.HTTP_201_CREATED, dependencies=[Depends(links_on)])
async def create_link(body: dto.LinkCreate, container: ContainerDep) -> dto.LinkInfo:
    """Link with the instance at ``url``, using the code it shows."""
    link = await container.links.pair_with(body)
    return container.link_admin.info(link)


@router.patch("/settings")
async def update_settings(
    body: dto.LinksSettingsUpdate, container: ContainerDep
) -> dto.LinksOverview:
    """Which engines to share (and whether linked instances may start them), and what linked
    instances may do with this instance's conversations. Saved to the .local.yaml sidecar."""
    return await container.link_admin.update_settings(body)


@router.post("/{link_id}/refresh")
async def refresh_link(link_id: str, container: ContainerDep) -> dto.LinkInfo:
    link = container.links.get(link_id)
    await container.links.poll(link)
    return container.link_admin.info(link)


@router.post("/{link_id}/use", dependencies=[Depends(links_on)])
async def use_engine(link_id: str, body: dto.LinkUse, container: ContainerDep) -> dto.LinksOverview:
    """Add a local backend for an engine the linked instance shares, and optionally make it
    the default for some tasks."""
    return await container.link_admin.use(link_id, body)


@router.delete("/{link_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_link(link_id: str, container: ContainerDep) -> Response:
    """Unlink. The other instance is told to forget the link too, if it can be reached.
    Backends that used the link stay configured, and report that they are not linked."""
    await container.links.unlink(link_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.api_route(
    "/{link_id}/r/{path:path}",
    methods=["GET", "POST", "PATCH", "PUT", "DELETE"],
    dependencies=[Depends(links_on)],
    include_in_schema=False,  # a pass-through: its routes are the linked instance's own
)
async def through_link(
    link_id: str, path: str, request: Request, container: ContainerDep
) -> StreamingResponse:
    """A linked instance's conversation, artifact or job route, as if it were local."""
    return await proxy.forward(container.links.get(link_id), request, path)


# --- pairing (no token) ---------------------------------------------------------------

pairing_router = APIRouter(prefix="/v1/link", tags=["peer"], dependencies=[Depends(links_on)])


@pairing_router.get("/hello")
async def hello(
    container: ContainerDep,
    nonce: Annotated[str | None, Query(max_length=128)] = None,
    asker: Annotated[str | None, Query(max_length=36)] = None,
) -> dto.LinkHello:
    """Who this is, and a challenge to pair against (valid for two minutes). A linked
    instance that sends a nonce gets proof this is the instance it linked with."""
    return container.links.hello(nonce, asker)


@pairing_router.post("/pair")
async def pair(
    body: dto.PairRequest, request: Request, container: ContainerDep
) -> dto.PairAccepted:
    address = request.client.host if request.client is not None else None
    return await container.links.accept(body, address)


# --- the peer API (linked instances) -----------------------------------------------------

peer_router = APIRouter(tags=["peer"], dependencies=[Depends(links_on), Depends(require_link)])


@peer_router.get("/v1/link/status")
async def link_status(container: ContainerDep) -> dto.PeerStatus:
    """What this instance shares, and whether each engine can run its whole model here."""
    return dto.PeerStatus(
        instance_id=container.links.instance_id,
        name=container.links.name,
        api_version=PEER_API_VERSION,
        version=__version__,
        gallery=container.settings.links.gallery,
        engines=await container.peers.status(),
    )


@peer_router.post("/v1/link/revoke", status_code=status.HTTP_204_NO_CONTENT)
async def revoke(principal: PrincipalDep, container: ContainerDep) -> Response:
    """The calling instance unlinked: forget it here too."""
    assert principal.link_id is not None
    await container.links.forget(principal.link_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@peer_router.post("/v1/peer/engines/{name}/chat")
async def peer_chat(
    name: str, body: dto.PeerChatRequest, container: ContainerDep
) -> EventSourceResponse:
    """One chat completion. Events: ``status``, ``delta``..., then ``done`` or ``error``."""
    container.peers.shared(name, BackendKind.LLM)
    return _stream(container.peers.chat(name, body))


@peer_router.post("/v1/peer/engines/{name}/images")
async def peer_images(
    name: str, body: dto.PeerImageRequest, container: ContainerDep
) -> EventSourceResponse:
    """One image. Events: ``status``, ``submitted``, ``progress``..., ``result``, then
    ``done`` or ``error``. Disconnecting cancels it."""
    container.peers.shared(name, BackendKind.IMAGE)
    return _stream(container.peers.images(name, body))


def _stream(events: AsyncIterator[tuple[str, dict[str, Any]]]) -> EventSourceResponse:
    async def guarded() -> AsyncIterator[ServerSentEvent]:
        try:
            async for event, data in events:
                yield ServerSentEvent(event=event, data=json.dumps(data))
        except ImaginerError as exc:
            yield ServerSentEvent(event="error", data=json.dumps(_problem(exc)))
        except Exception:
            log.exception("peer.request_failed")
            yield ServerSentEvent(
                event="error",
                data=json.dumps(
                    {
                        "status": 500,
                        "code": "internal_error",
                        "title": "Internal error",
                        "detail": "an unexpected error occurred",
                    }
                ),
            )

    return EventSourceResponse(guarded(), ping=PING_INTERVAL_S, headers={"X-Accel-Buffering": "no"})


def _problem(exc: ImaginerError) -> dict[str, Any]:
    return {"status": exc.status, "code": exc.code, "title": exc.title, "detail": exc.detail}


# --- the gallery ------------------------------------------------------------------------

gallery_router = APIRouter(prefix="/v1/gallery", tags=["gallery"])


@gallery_router.get("")
async def gallery(
    container: ContainerDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    cursor: Annotated[str | None, Query(description="next_cursor from the previous page")] = None,
    include_archived: bool = False,
) -> dto.GalleryPage:
    """This instance's conversations and every linked instance's, newest activity first.
    Each says where it lives (``origin``); remote ones are opened through ``base_path``."""
    links = container.links.all() if container.settings.links.enabled else []
    return await container.gallery.page(
        links, limit=limit, cursor=cursor, include_archived=include_archived
    )
