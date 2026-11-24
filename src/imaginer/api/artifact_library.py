"""The artifact library (docs/artifact-library.md).

``router`` is the user's: the merged listing, pins, and importing an image into a
conversation. ``shared_router`` is what linked instances call (as far as this instance's
``links.gallery`` allows), so their libraries can list this instance's images and fetch
pinned ones; the user may call it too.
"""

from typing import Annotated

from fastapi import APIRouter, Header, Query, Response, status
from fastapi.responses import FileResponse

from imaginer.api.auth import PrincipalDep
from imaginer.api.deps import ContainerDep
from imaginer.db.repositories.artifact_library import LibraryQuery
from imaginer.domain import models as dto
from imaginer.domain.enums import ArtifactKind, ArtifactSource

router = APIRouter(prefix="/v1/artifact-library", tags=["artifact library"])
shared_router = APIRouter(prefix="/v1/artifact-library", tags=["artifact library"])

# A pin never gets new bytes, so its content can be cached indefinitely.
_CACHE_CONTROL = "private, max-age=31536000, immutable"

KindQuery = Annotated[
    dto.LibraryKind, Query(description="image (the default), mask, or all of them.")
]
SourceQuery = Annotated[
    list[ArtifactSource] | None,
    Query(description="Only images from these sources (repeat it); none: every source."),
]
TextQuery = Annotated[
    str | None,
    Query(
        max_length=200,
        description="Text in its prompt, file name, label, conversation title or pin.",
    ),
]


def _query(
    kind: dto.LibraryKind, source: list[ArtifactSource] | None, q: str | None, pinned: bool
) -> LibraryQuery:
    kinds = (ArtifactKind.IMAGE, ArtifactKind.MASK) if kind == "all" else (ArtifactKind(kind),)
    return LibraryQuery(
        kinds=kinds,
        sources=tuple(dict.fromkeys(source or [])),
        pinned=pinned,
        text=(q or "").strip() or None,
    )


# --- the user's ------------------------------------------------------------------------


@router.get("")
async def library(
    container: ContainerDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 60,
    cursor: Annotated[str | None, Query(description="next_cursor from the previous page")] = None,
    kind: KindQuery = "image",
    source: SourceQuery = None,
    q: TextQuery = None,
    pinned: Annotated[bool, Query(description="Only pinned images.")] = False,
    origin: Annotated[
        str | None, Query(max_length=64, description="Only 'local', or one link's images.")
    ] = None,
) -> dto.ArtifactLibraryPage:
    """Every image here and on linked instances, newest first, each distinct image once.
    A page can be short (even empty) while ``next_cursor`` says there is more: keep asking."""
    return await container.artifact_library.page(
        _query(kind, source, q, pinned), cursor=cursor, limit=limit, origin=origin
    )


@router.post(
    "/pins",
    status_code=status.HTTP_201_CREATED,
    responses={200: {"description": "Its bytes were pinned already: that pin."}},
)
async def pin(
    body: dto.ArtifactPinCreate, container: ContainerDep, response: Response
) -> dto.ArtifactLibraryItem:
    """Keep an image: its bytes are copied into the library, so it outlives its
    conversation (and the linked instance it came from)."""
    item, created = await container.artifact_library.pin(body)
    if not created:
        response.status_code = status.HTTP_200_OK
    return item


@router.get("/pins/{pin_id}")
async def get_pin(pin_id: str, container: ContainerDep) -> dto.ArtifactLibraryItem:
    return await container.artifact_library.get_pin(pin_id)


@router.patch("/pins/{pin_id}")
async def update_pin(
    pin_id: str, body: dto.ArtifactPinUpdate, container: ContainerDep
) -> dto.ArtifactLibraryItem:
    return await container.artifact_library.update_pin(pin_id, body)


@router.delete("/pins/{pin_id}", status_code=status.HTTP_204_NO_CONTENT)
async def unpin(pin_id: str, container: ContainerDep) -> None:
    """Stop keeping it. An image whose conversation is gone is gone with its pin."""
    await container.artifact_library.unpin(pin_id)


@router.post(
    "/import",
    status_code=status.HTTP_201_CREATED,
    responses={200: {"description": "The conversation has the image already: that artifact."}},
)
async def import_artifact(
    body: dto.ArtifactImport, container: ContainerDep, response: Response
) -> dto.Artifact:
    """Copy a library image into a conversation (here, or on the linked instance
    ``link_id``), where it becomes an upload with a label of its own."""
    artifact, created = await container.artifact_library.import_into(body)
    if not created:
        response.status_code = status.HTTP_200_OK
    return artifact


# --- what linked instances call ----------------------------------------------------------


@shared_router.get("/local")
async def local_library(
    container: ContainerDep,
    principal: PrincipalDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 60,
    cursor: Annotated[str | None, Query(max_length=36, description="The last id taken")] = None,
    kind: KindQuery = "image",
    source: SourceQuery = None,
    q: TextQuery = None,
) -> dto.Page[dto.ArtifactLibraryItem]:
    """This instance's own images, newest first: its live ones, and pins whose
    conversation is gone. A linked instance sees only pins of this instance's images."""
    return await container.artifact_library.local_page(
        _query(kind, source, q, pinned=False),
        cursor=cursor,
        limit=limit,
        for_link=principal.is_link,
    )


@shared_router.get("/local/artifacts/{artifact_id}")
async def local_artifact(
    artifact_id: str, container: ContainerDep, principal: PrincipalDep
) -> dto.ArtifactLibraryItem:
    return await container.artifact_library.local_item(
        artifact_id=artifact_id, for_link=principal.is_link
    )


@shared_router.get("/local/pins/{pin_id}")
async def local_pin(
    pin_id: str, container: ContainerDep, principal: PrincipalDep
) -> dto.ArtifactLibraryItem:
    return await container.artifact_library.local_item(pin_id=pin_id, for_link=principal.is_link)


@shared_router.get(
    "/pins/{pin_id}/content",
    response_class=Response,
    responses={
        200: {"description": "The pinned bytes, with their MIME type and ETag = sha256."},
        304: {"description": "Not modified (If-None-Match matched)."},
    },
)
async def pin_content(
    pin_id: str,
    container: ContainerDep,
    principal: PrincipalDep,
    if_none_match: Annotated[str | None, Header()] = None,
) -> Response:
    pin, path = await container.artifact_library.pin_file(pin_id, for_link=principal.is_link)
    etag = f'"{pin.sha256}"'
    headers = {"ETag": etag, "Cache-Control": _CACHE_CONTROL}
    if if_none_match is not None and _etag_matches(if_none_match, etag):
        return Response(status_code=status.HTTP_304_NOT_MODIFIED, headers=headers)
    return FileResponse(
        path,
        media_type=pin.mime,
        headers=headers,
        filename=f"{pin.label or 'pinned'}.{path.suffix.lstrip('.')}",
        content_disposition_type="inline",
    )


def _etag_matches(header: str, etag: str) -> bool:
    candidates = {c.strip().removeprefix("W/") for c in header.split(",")}
    return "*" in candidates or etag in candidates
