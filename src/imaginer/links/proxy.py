"""Pass a request on to a linked instance, as this instance's user, and stream the answer back.

This is how a conversation that lives on a linked instance is opened, continued and
edited from here: the browser only ever talks to (and holds the token of) its own
instance. Only conversation, artifact and job routes pass, plus the backend list; the
linked instance still applies its own ``links.gallery`` rules to them.
"""

import re
from collections.abc import AsyncIterator

import httpx
from fastapi import Request
from fastapi.responses import StreamingResponse

from imaginer.domain.errors import EngineUnavailableError, ForbiddenError
from imaginer.links.state import Link

_ALLOWED = re.compile(
    r"v1/(conversations(/[\w.-]+)*|artifacts(/[\w.-]+)+|jobs(/[\w.-]+)+|backends)"
)
_WITH_BODY = frozenset({"POST", "PUT", "PATCH"})
_REQUEST_HEADERS = (
    "accept",
    "content-type",
    "last-event-id",
    "if-none-match",
    "if-modified-since",
    "range",
    "x-request-id",
)
_RESPONSE_HEADERS = (
    "content-type",
    "content-length",
    "content-encoding",
    "content-disposition",
    "content-range",
    "accept-ranges",
    "etag",
    "last-modified",
    "cache-control",
    "x-turn-id",
    "x-accel-buffering",
)


async def forward(link: Link, request: Request, path: str) -> StreamingResponse:
    dot_segment = any(not part.strip(".") for part in path.split("/"))
    if (
        dot_segment
        or not _ALLOWED.fullmatch(path)
        or (path == "v1/backends" and request.method != "GET")
    ):
        raise ForbiddenError(f"/{path} can't be reached on a linked instance")
    client = link.transport.client
    upstream = client.build_request(
        request.method,
        f"/{path}",
        params=list(request.query_params.multi_items()),
        headers={k: v for k in _REQUEST_HEADERS if (v := request.headers.get(k))},
        content=request.stream() if request.method in _WITH_BODY else None,
    )
    try:
        response = await client.send(upstream, stream=True)
    except httpx.HTTPError as exc:
        raise EngineUnavailableError(
            f"{link.peer_name} is unreachable at {link.url}: {exc!s}", link=link.id
        ) from exc

    async def body() -> AsyncIterator[bytes]:
        try:
            async for chunk in response.aiter_raw():
                yield chunk
        except httpx.HTTPError:
            return  # the linked instance went away mid-stream; the client sees it end
        finally:
            await response.aclose()

    headers = {k: v for k in _RESPONSE_HEADERS if (v := response.headers.get(k))}
    return StreamingResponse(body(), status_code=response.status_code, headers=headers)
