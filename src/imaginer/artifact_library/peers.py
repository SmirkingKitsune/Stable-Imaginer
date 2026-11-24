"""The artifact library on a linked instance: its images, their bytes, and handing it one.

Listing and describing use the library's own peer routes; an instance too old to have them
answers 404, and the library says so instead of failing. Fetching bytes and handing an image
to one of its conversations use only routes every linked instance already has.
"""

import hashlib
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
from pydantic import ValidationError

from imaginer.domain import models as dto
from imaginer.domain.enums import ArtifactKind
from imaginer.domain.errors import (
    EngineError,
    EngineUnavailableError,
    ForbiddenError,
    ImaginerError,
    InvalidRequestError,
    NotFoundError,
    PayloadTooLargeError,
)
from imaginer.links.service import LinkService
from imaginer.links.state import Link

LIST_TIMEOUT_S = 3.0  # a listing doesn't wait long for one instance
ITEM_TIMEOUT_S = 10.0
BYTES_TIMEOUT_S = 120.0  # an image across a slow network
LOCAL_PATH = "/v1/artifact-library/local"


class PeerTooOldError(ImaginerError):
    """The linked instance runs an Imaginer without the artifact library."""

    status = 422
    code = "peer_unsupported"
    title = "Linked instance too old"


async def list_page(
    links: LinkService, link: Link, params: dict[str, Any], *, recheck_after_s: float
) -> dto.Page[dto.ArtifactLibraryItem]:
    """One page of the images the linked instance has itself (its live ones, its pins)."""
    response = await links.ask(
        link,
        "GET",
        LOCAL_PATH,
        params=params,
        timeout_s=LIST_TIMEOUT_S,
        accept_status=frozenset({403, 404}),
        recheck_after_s=recheck_after_s,
    )
    if response.status_code == 404:
        raise PeerTooOldError(
            f"{link.peer_name} runs an older Imaginer: update it to see its images"
        )
    if response.status_code == 403:
        raise InvalidRequestError(f"{link.peer_name} doesn't share its images")
    try:
        return dto.Page[dto.ArtifactLibraryItem].model_validate(response.json())
    except (ValueError, ValidationError) as exc:
        raise InvalidRequestError(f"{link.peer_name} sent an unexpected listing") from exc


async def item(links: LinkService, link: Link, ref: dto.LibraryRef) -> dto.ArtifactLibraryItem:
    """One of its images, as its library lists it."""
    path = (
        f"{LOCAL_PATH}/artifacts/{ref.artifact_id}"
        if ref.artifact_id
        else f"{LOCAL_PATH}/pins/{ref.pin_id}"
    )
    response = await links.ask(
        link, "GET", path, timeout_s=ITEM_TIMEOUT_S, accept_status=frozenset({403, 404})
    )
    if response.status_code == 403:
        raise ForbiddenError(f"{link.peer_name} doesn't share its images")
    if response.status_code == 404:
        if ref.artifact_id is None:
            raise NotFoundError(f"{link.peer_name} has no pin {ref.pin_id}")
        # Gone, or an instance without the library: an artifact is reachable there anyway.
        found = await artifact(links, link, ref.artifact_id)
        if found is None:
            raise NotFoundError(f"{link.peer_name} has no artifact {ref.artifact_id}")
        return _item_of(found, link)
    try:
        return dto.ArtifactLibraryItem.model_validate(response.json())
    except (ValueError, ValidationError) as exc:
        raise InvalidRequestError(f"{link.peer_name} sent an unexpected image") from exc


async def artifact(links: LinkService, link: Link, artifact_id: str) -> dto.Artifact | None:
    response = await links.ask(
        link,
        "GET",
        f"/v1/artifacts/{artifact_id}",
        timeout_s=ITEM_TIMEOUT_S,
        accept_status=frozenset({403, 404}),
    )
    if response.status_code == 403:
        raise ForbiddenError(f"{link.peer_name} doesn't share its conversations")
    if response.status_code == 404:
        return None
    try:
        return dto.Artifact.model_validate(response.json())
    except (ValueError, ValidationError) as exc:
        raise InvalidRequestError(f"{link.peer_name} sent an unexpected artifact") from exc


async def fetch(
    links: LinkService, link: Link, content_url: str, *, sha256: str, max_bytes: int
) -> bytes:
    """An image's bytes, checked against the hash it was listed with."""
    links.check(link)
    data = bytearray()
    try:
        async with link.transport.client.stream(
            "GET", content_url, timeout=BYTES_TIMEOUT_S
        ) as response:
            if response.status_code == 404:
                raise NotFoundError(f"{link.peer_name} no longer has this image")
            if response.status_code == 403:
                raise ForbiddenError(f"{link.peer_name} doesn't share this image")
            if not response.is_success:
                raise EngineError(
                    f"{link.peer_name} answered HTTP {response.status_code} for {content_url}"
                )
            async for chunk in response.aiter_bytes():
                data += chunk
                if len(data) > max_bytes:
                    raise PayloadTooLargeError(
                        f"the image on {link.peer_name} is larger than {max_bytes} bytes"
                    )
    except httpx.HTTPError as exc:
        links.recheck(link, 0)
        raise EngineUnavailableError(f"{link.peer_name} is unreachable: {exc!s}") from exc
    if hashlib.sha256(data).hexdigest() != sha256:
        raise EngineError(f"{link.peer_name} sent different bytes than it listed")
    return bytes(data)


async def take(
    links: LinkService,
    link: Link,
    conversation_id: str,
    image: dto.ArtifactLibraryItem,
    data: Callable[[], Awaitable[bytes]],
    *,
    filename: str,
) -> tuple[dto.Artifact, bool]:
    """Put an image into a conversation on the linked instance, as an upload there; the
    one it already has when the conversation has these bytes (and False)."""
    if link.peer is not None and link.peer.gallery != "full":
        raise ForbiddenError(f"{link.peer_name} shares its conversations read-only")
    existing = await _find(links, link, conversation_id, image.sha256, image.kind)
    if existing is not None:
        return existing, False
    payload = await data()
    links.check(link)
    try:
        response = await link.transport.client.post(
            f"/v1/conversations/{conversation_id}/artifacts",
            params={"kind": image.kind.value},
            files={"file": (filename, payload, image.mime)},
            timeout=BYTES_TIMEOUT_S,
        )
    except httpx.HTTPError as exc:
        links.recheck(link, 0)
        raise EngineUnavailableError(f"{link.peer_name} is unreachable: {exc!s}") from exc
    if response.status_code == 403:
        raise ForbiddenError(f"{link.peer_name} shares its conversations read-only")
    if response.status_code == 404:
        raise NotFoundError(f"{link.peer_name} has no conversation {conversation_id}")
    if response.status_code == 413:
        raise PayloadTooLargeError(f"the image is too large for {link.peer_name}")
    if response.status_code in (415, 422):
        raise InvalidRequestError(f"{link.peer_name} refused the image: {_detail(response)}")
    if not response.is_success:
        raise EngineError(f"{link.peer_name} answered HTTP {response.status_code}")
    try:
        return dto.Artifact.model_validate(response.json()), True
    except (ValueError, ValidationError) as exc:
        raise InvalidRequestError(f"{link.peer_name} sent an unexpected artifact") from exc


async def _find(
    links: LinkService, link: Link, conversation_id: str, sha256: str, kind: ArtifactKind
) -> dto.Artifact | None:
    """The conversation's artifact with these bytes, if it has one."""
    cursor: str | None = None
    for _ in range(50):  # 500 a page: far more than any conversation has
        params: dict[str, Any] = {"kind": kind.value, "limit": 500}
        if cursor is not None:
            params["cursor"] = cursor
        response = await links.ask(
            link,
            "GET",
            f"/v1/conversations/{conversation_id}/artifacts",
            params=params,
            timeout_s=ITEM_TIMEOUT_S,
            accept_status=frozenset({403, 404}),
        )
        if response.status_code == 404:
            raise NotFoundError(f"{link.peer_name} has no conversation {conversation_id}")
        if response.status_code == 403:
            raise ForbiddenError(f"{link.peer_name} doesn't share its conversations")
        try:
            page = dto.Page[dto.Artifact].model_validate(response.json())
        except (ValueError, ValidationError) as exc:
            raise InvalidRequestError(f"{link.peer_name} sent an unexpected listing") from exc
        for found in page.items:
            if found.sha256 == sha256:
                return found
        if page.next_cursor is None:
            return None
        cursor = page.next_cursor
    return None


def _item_of(found: dto.Artifact, link: Link) -> dto.ArtifactLibraryItem:
    """An artifact of an instance without the library, as the library would list it."""
    metadata = found.metadata or {}
    prompt = metadata.get("prompt")
    filename = metadata.get("original_filename")
    return dto.ArtifactLibraryItem(
        id=found.id,
        ref=dto.LibraryRef(origin=link.id, artifact_id=found.id),
        live=True,
        kind=found.kind,
        source=found.source,
        mime=found.mime,
        sha256=found.sha256,
        size_bytes=found.size_bytes,
        width=found.width,
        height=found.height,
        has_alpha=found.has_alpha,
        created_at=found.created_at,
        content_url=found.content_url,
        label=found.label,
        conversation_id=found.conversation_id,
        prompt=prompt if isinstance(prompt, str) else None,
        original_filename=filename if isinstance(filename, str) else None,
        origin=dto.Origin(instance_id=link.peer_instance_id, name=link.peer_name),
    )


def _detail(response: httpx.Response) -> str:
    try:
        detail = response.json().get("detail")
    except (ValueError, AttributeError):
        detail = None
    return str(detail) if detail else f"HTTP {response.status_code}"
