"""One gallery over this instance's conversations and those of every linked instance.

Each source is listed in its own order (most recently updated first) from its own
position, and the pages are merged. The cursor holds every source's position: absent
means "from the top", null means "nothing more", otherwise the (updated_at, id) of the last
conversation taken from it. Conversations stay where they are: remote ones are reached
through ``/v1/links/{link_id}/r/...`` (``Origin.base_path``).

A source that doesn't answer is reported in ``sources`` and skipped for the rest of this
listing; the next listing from the top tries it again. A link last seen unreachable isn't
waited for (a refused connection alone can take seconds): it is reported at once and asked
again in the background, and listed again once it answers.
"""

import asyncio
import base64
import binascii
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

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
RECHECK_AFTER_S = 5.0  # how often listings ask an unreachable link again (in the background)

# Where a source's listing goes on from: the gallery keeps (updated_at ISO, id) of the last
# conversation taken, the artifact library (id,). None: the source is exhausted.
Position = tuple[str, ...] | None


@dataclass(frozen=True)
class SourcePage[T]:
    items: list[T]  # newest first, all after the source's position
    has_more: bool


def encode(positions: Mapping[str, Position]) -> str:
    raw = json.dumps({k: list(v) if v is not None else None for k, v in positions.items()})
    return base64.urlsafe_b64encode(raw.encode()).decode().rstrip("=")


def decode(cursor: str) -> dict[str, Position]:
    try:
        raw = json.loads(base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)))
        if not isinstance(raw, dict):
            raise ValueError("not an object")
        positions: dict[str, Position] = {}
        for k, v in raw.items():
            if v is not None and (not isinstance(v, list) or not v):
                raise ValueError("not a position")
            positions[str(k)] = tuple(str(part) for part in v) if v is not None else None
        return positions
    except (binascii.Error, ValueError, TypeError, IndexError, KeyError) as exc:
        raise InvalidRequestError("invalid cursor") from exc


def _by_activity(item: Any) -> Any:
    return (item.updated_at, item.id)


def _activity_position(item: Any) -> tuple[str, ...]:
    return (item.updated_at.isoformat(), item.id)


def merge[T](
    pages: Mapping[str, SourcePage[T]],
    positions: Mapping[str, Position],
    limit: int,
    *,
    sort_key: Callable[[T], Any] = _by_activity,
    position: Callable[[T], tuple[str, ...]] = _activity_position,
) -> tuple[list[T], dict[str, Position]]:
    """The newest ``limit`` items across the pages, and every source's position after them.
    By default items are gallery conversations, ordered by their latest activity.

    A source is exhausted once it has no more pages and all its items were taken."""
    pool = [(item, source) for source, page in pages.items() for item in page.items]
    pool.sort(key=lambda pair: sort_key(pair[0]), reverse=True)
    chosen = pool[:limit]
    after = dict(positions)
    for source, page in pages.items():
        taken = [item for item, s in chosen if s == source]
        if not page.has_more and len(taken) == len(page.items):
            after[source] = None
        elif taken:
            after[source] = position(taken[-1])
    return [item for item, _ in chosen], after


def source_failed(source: dto.GallerySource, error: BaseException) -> dto.GallerySource:
    """A source as listed once asking it failed: unreachable when it didn't answer."""
    reachable = isinstance(error, ImaginerError) and not error.code.startswith("engine")
    return source.model_copy(
        update={
            "status": "error" if reachable else "unreachable",
            "detail": error.detail if isinstance(error, ImaginerError) else str(error),
        }
    )


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
        pages: dict[str, SourcePage[dto.GalleryItem]] = {}
        for source, result in zip(wanted, fetched, strict=True):
            if isinstance(result, SourcePage):
                pages[source] = result
                continue
            sources[source] = source_failed(sources[source], result)
            positions[source] = None  # skip it for the rest of this listing
        items, after = merge(pages, positions, limit)
        done = all(after.get(s, "start") is None for s in sources)
        return dto.GalleryPage(
            items=items,
            next_cursor=None if done else encode(after),
            sources=list(sources.values()),
        )

    async def _local(
        self, limit: int, position: Position, include_archived: bool
    ) -> SourcePage[dto.GalleryItem]:
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
    ) -> SourcePage[dto.GalleryItem]:
        params: dict[str, str | int] = {"limit": limit, "include_archived": include_archived}
        cursor = _source_cursor(position)
        if cursor is not None:
            params["cursor"] = cursor
        response = await self._links.ask(
            link,
            "GET",
            "/v1/conversations",
            params=params,
            timeout_s=REMOTE_TIMEOUT_S,
            accept_status=frozenset({403}),
            recheck_after_s=RECHECK_AFTER_S,
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
