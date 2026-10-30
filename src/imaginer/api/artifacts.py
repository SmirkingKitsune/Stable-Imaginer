from typing import Annotated, Literal

from fastapi import APIRouter, File, Header, Query, Request, Response, UploadFile, status
from fastapi.responses import FileResponse, StreamingResponse

from imaginer.api.deps import ContainerDep
from imaginer.domain import models as dto
from imaginer.domain.enums import ArtifactKind, ArtifactSource
from imaginer.domain.errors import InvalidRequestError, PayloadTooLargeError

conversation_router = APIRouter(
    prefix="/v1/conversations/{conversation_id}/artifacts", tags=["artifacts"]
)
router = APIRouter(prefix="/v1/artifacts", tags=["artifacts"])

# An artifact id never gets new bytes, so content can be cached indefinitely.
_CACHE_CONTROL = "private, max-age=31536000, immutable"


@conversation_router.get("")
async def list_artifacts(
    conversation_id: str,
    container: ContainerDep,
    kind: ArtifactKind | None = None,
    source: ArtifactSource | None = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    cursor: Annotated[str | None, Query(description="next_cursor from the previous page")] = None,
) -> dto.Page[dto.Artifact]:
    """Artifacts of one conversation, oldest first."""
    return await container.artifacts.list(
        conversation_id, kind=kind, source=source, limit=limit, cursor=cursor
    )


@conversation_router.post("", status_code=status.HTTP_201_CREATED)
async def upload_artifact(
    conversation_id: str,
    container: ContainerDep,
    file: Annotated[UploadFile, File(description="PNG, JPEG, WebP, GIF or BMP image")],
    kind: Annotated[
        Literal["image", "mask"],
        Query(description="'mask': a black-and-white mask (white = change), labelled mask-N."),
    ] = "image",
    mask_for: Annotated[
        str | None,
        Query(max_length=32, description="With kind=mask: label of the image it was painted on."),
    ] = None,
) -> dto.Artifact:
    """Upload an image into the conversation. Its type is detected from the bytes."""
    limit = container.settings.storage.max_upload_bytes
    if file.size is not None and file.size > limit:
        raise PayloadTooLargeError(f"upload exceeds {limit} bytes")
    if mask_for is not None and kind != "mask":
        raise InvalidRequestError("mask_for is only for kind=mask")
    try:
        return await container.artifacts.ingest_upload(
            conversation_id,
            file.file,
            file.filename,
            kind=ArtifactKind(kind),
            mask_for=mask_for.strip().lower() if mask_for else None,
        )
    finally:
        await file.close()


@router.get("/{artifact_id}")
async def get_artifact(artifact_id: str, container: ContainerDep) -> dto.Artifact:
    return await container.artifacts.get(artifact_id)


@router.get(
    "/{artifact_id}/content",
    response_class=Response,
    responses={
        200: {"description": "The artifact's bytes, with its MIME type and ETag = sha256."},
        304: {"description": "Not modified (If-None-Match matched)."},
    },
)
async def get_artifact_content(
    artifact_id: str,
    request: Request,
    container: ContainerDep,
    if_none_match: Annotated[str | None, Header()] = None,
) -> Response:
    content = await container.artifacts.content(artifact_id)
    artifact = content.artifact
    etag = f'"{artifact.sha256}"'
    headers = {"ETag": etag, "Cache-Control": _CACHE_CONTROL}
    if if_none_match is not None and _etag_matches(if_none_match, etag):
        return Response(status_code=status.HTTP_304_NOT_MODIFIED, headers=headers)
    ext = artifact.mime.rsplit("/", 1)[-1].replace("jpeg", "jpg")
    if content.path is not None:
        return FileResponse(
            content.path,
            media_type=artifact.mime,
            headers=headers,
            filename=f"{artifact.label}.{ext}",
            content_disposition_type="inline",
        )
    return StreamingResponse(content.chunks(), media_type=artifact.mime, headers=headers)


@router.get("/{artifact_id}/lineage")
async def get_artifact_lineage(artifact_id: str, container: ContainerDep) -> dto.Lineage:
    """What this artifact was made from (parents) and what was made from it (children)."""
    return await container.artifacts.lineage(artifact_id)


def _etag_matches(header: str, etag: str) -> bool:
    candidates = {c.strip().removeprefix("W/") for c in header.split(",")}
    return "*" in candidates or etag in candidates
