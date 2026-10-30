"""What this instance knows about one linked instance, kept in memory.

The link table holds who the peer is and the tokens. This adds how it answered last time:
reachable or not, and what it shares. Engine adapters read it through ``LinkDirectory``,
so they resolve the link on every call and follow it when its address changes.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Protocol

from imaginer.backends.openai_http.transport import HttpTransport
from imaginer.domain import models as dto

PEER_API_VERSION = 1

LinkStatus = Literal["unknown", "ok", "unreachable", "incompatible", "unauthorized"]


@dataclass
class Link:
    id: str
    name: str
    peer_instance_id: str
    peer_name: str
    url: str
    created_at: datetime
    transport: HttpTransport
    status: LinkStatus = "unknown"
    detail: str | None = None
    peer: dto.PeerStatus | None = None
    checked_at: float | None = None  # time.monotonic() of the last poll
    seen_at: datetime | None = None  # last time it answered
    engines: dict[str, dto.SharedEngine] = field(default_factory=dict)

    def engine(self, name: str) -> dto.SharedEngine | None:
        return self.engines.get(name) if self.status == "ok" else None


class LinkDirectory(Protocol):
    def by_name(self, name: str) -> Link | None: ...

    async def fresh(self, name: str, max_age_s: float) -> Link | None:
        """The link, polled now if its status is older than ``max_age_s``."""
        ...
