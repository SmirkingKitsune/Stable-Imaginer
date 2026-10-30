"""One gallery over this instance's conversations and those of every linked instance.

Each source is listed in its own order (most recently updated first) from its own
position, and the pages are merged. The cursor holds every source's position: absent
means "from the top", null means "nothing more", otherwise the (updated_at, id) of the last
conversation taken from it. Conversations stay where they are: remote ones are reached
through ``/v1/links/{link_id}/r/...`` (``Origin.base_path``).

A source that doesn't answer is reported in ``sources`` and skipped for the rest of this
listing; the next listing from the top tries it again.
"""

import asyncio
import base64
import binascii
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from pydantic import ValidationError

from imaginer.domain import models as dto
from imaginer.domain.errors import ImaginerError, InvalidRequestError
from imaginer.links.state import Link
from imaginer.orchestration.event_bus import TurnManager
from imaginer.services.conversations import ConversationService, encode_cursor

if TYPE_CHECKING:
    from imaginer.links.service import LinkService

LOCAL = "local"
REMOTE_TIMEOUT_S = 3.0

Position = tuple[str, str] | None  # (updated_at ISO, id); None: the source is exhausted


@dataclass(frozen=True)
class SourcePage:
    items: list[dto.GalleryItem]  # newest first, all after the source's position
    has_more: bool


def encode(positions: Mapping[str, Position]) -> str:
    raw = json.dumps({k: list(v) if v is not None else None for k, v in positions.items()})
    return base64.urlsafe_b64encode(raw.encode()).decode().rstrip("=")


def decode(cursor: str) -> dict[str, Position]:
    try:
        raw = json.loads(base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)))
        if not isinstance(raw, dict):
            raise ValueError("not an object")
        return {str(k): (str(v[0]), str(v[1])) if v is not None else None for k, v in raw.items()}
    except (binascii.Error, ValueError, TypeError, IndexError, KeyError) as exc:
        raise InvalidRequestError("invalid cursor") from exc


def merge(
    pages: Mapping[str, SourcePage], positions: Mapping[str, Position], limit: int
) -> tuple[list[dto.GalleryItem], dict[str, Position]]:
    """The newest ``limit`` items across the pages, and every source's position after them.

    A source is exhausted once it has no more pages and all its items were taken."""
    pool = [(item, source) for source, page in pages.items() for item in page.items]
    pool.sort(key=lambda pair: (pair[0].updated_at, pair[0].id), reverse=True)
    chosen = pool[:limit]
    after = dict(positions)
    for source, page in pages.items():
        taken = [item for item, s in chosen if s == source]
        if not page.has_more and len(taken) == len(page.items):
            after[source] = None
        elif taken:
            after[source] = (taken[-1].updated_at.isoformat(), taken[-1].id)
    return [item for item, _ in chosen], after


def _source_cursor(position: Position) -> str | None:
    if position is None:
        return None
    return encode_cursor(datetime.fromisoformat(position[0]), position[1])


class Gallery:
    def __init__(
        self, conversations: ConversationService, turns: TurnManager, links: "LinkService"
    ) -> None:
        self._conversations = conversations
        self._turns = turns
        self._links = links

    async def page(
        self,
        links: list[Link],
        *,
        limit: int,
        cursor: str | None,
        include_archived: bool,
    ) -> dto.GalleryPage:
        positions: dict[str, Position] = decode(cursor) if cursor else {}
        sources = {
            LOCAL: dto.GallerySource(id=LOCAL, name=self._links.name, local=True, status="ok"),
            **{
                link.id: dto.GallerySource(
                    id=link.id, name=link.peer_name, local=False, status="ok"
                )
                for link in links
            },
        }
        wanted = [s for s in sources if not (s in positions and positions[s] is None)]
        by_id = {link.id: link for link in links}
        fetched = await asyncio.gather(
            *(
                self._local(limit, positions.get(LOCAL), include_archived)
                if source == LOCAL
                else self._remote(by_id[source], limit, positions.get(source), include_archived)
                for source in wanted
            ),
            return_exceptions=True,
        )
        pages: dict[str, SourcePage] = {}
        for source, result in zip(wanted, fetched, strict=True):
            if isinstance(result, SourcePage):
                pages[source] = result
                continue
            reachable = isinstance(result, ImaginerError) and not result.code.startswith("engine")
            sources[source] = sources[source].model_copy(
                update={
                    "status": "error" if reachable else "unreachable",
                    "detail": result.detail if isinstance(result, ImaginerError) else str(result),
                }
            )
            positions[source] = None  # skip it for the rest of this listing
        items, after = merge(pages, positions, limit)
        done = all(after.get(s, "start") is None for s in sources)
        return dto.GalleryPage(
            items=items,
            next_cursor=None if done else encode(after),
            sources=list(sources.values()),
        )

    async def _local(self, limit: int, position: Position, include_archived: bool) -> SourcePage:
        page = await self._conversations.list(
            limit=limit, cursor=_source_cursor(position), include_archived=include_archived
        )
        origin = dto.Origin(instance_id=self._links.instance_id, name=self._links.name)
        items = []
        for summary in page.items:
            channel = self._turns.get(summary.id)
            summary.turn_running = channel is not None and not channel.closed
            items.append(dto.GalleryItem(**summary.model_dump(), origin=origin))
        return SourcePage(items=items, has_more=page.next_cursor is not None)

    async def _remote(
        self, link: Link, limit: int, position: Position, include_archived: bool
    ) -> SourcePage:
        if link.status in ("unauthorized", "incompatible"):
            raise InvalidRequestError(link.detail or f"{link.peer_name} is {link.status}")
        params: dict[str, str | int] = {"limit": limit, "include_archived": include_archived}
        cursor = _source_cursor(position)
        if cursor is not None:
            params["cursor"] = cursor
        response = await link.transport.request(
            "GET",
            "/v1/conversations",
            params=params,
            timeout_s=REMOTE_TIMEOUT_S,
            attempts=1,
            accept_status=frozenset({403}),
        )
        if response.status_code == 403:
            raise InvalidRequestError(f"{link.peer_name} doesn't share its conversations")
        try:
            page = dto.Page[dto.ConversationSummary].model_validate(response.json())
        except (ValueError, ValidationError) as exc:
            raise InvalidRequestError(f"{link.peer_name} sent an unexpected listing") from exc
        origin = dto.Origin(
            link_id=link.id,
            instance_id=link.peer_instance_id,
            name=link.peer_name,
            base_path=f"/v1/links/{link.id}/r",
        )
        items = [dto.GalleryItem(**s.model_dump(), origin=origin) for s in page.items]
        return SourcePage(items=items, has_more=page.next_cursor is not None)
